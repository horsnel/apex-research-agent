"""
Evidence Fetcher — full-text evidence pipeline for research reports.

Reports built from 1000-char search snippets are "snippet-summarized", not
"evidence-grounded". This module closes that gap by:

1. Fetching FULL TEXT for the top-ranked sources:
   - Open-access PDFs via Unpaywall (DOI → PDF) → pdf_ingest extraction
   - Live scraping with the existing 3-tier fallback (httpx → Jina → Firecrawl)
2. Chunking full text (reusing ingest/chunker, header-aware, ~400 tokens)
3. Providing targeted per-claim / per-section retrieval over the fresh
   mini-corpus:
   - Semantic scoring (Cloudflare/OpenAI embeddings) when credentials exist
   - Keyword-overlap scoring as zero-credential fallback
4. Formatting evidence blocks with stable [S{n}] source attribution so every
   generated sentence can be traced back to a source index.

All heavy imports (httpx, embedder, pdf tools) are lazy so this module can be
imported and unit-tested without optional dependencies.
"""

import asyncio
import logging
import math
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Configuration ──
EVIDENCE_TOP_SOURCES = int(os.getenv("EVIDENCE_TOP_SOURCES", "6"))
EVIDENCE_MAX_CHARS_PER_SOURCE = int(os.getenv("EVIDENCE_MAX_CHARS_PER_SOURCE", "40000"))
EVIDENCE_CHUNK_TOKENS = int(os.getenv("EVIDENCE_CHUNK_TOKENS", "400"))
EVIDENCE_CHUNK_OVERLAP = float(os.getenv("EVIDENCE_CHUNK_OVERLAP", "0.15"))
EVIDENCE_CHUNKS_PER_SECTION = int(os.getenv("EVIDENCE_CHUNKS_PER_SECTION", "6"))
EVIDENCE_FETCH_TIMEOUT = float(os.getenv("EVIDENCE_FETCH_TIMEOUT", "30"))

CHARS_PER_TOKEN = 4


@dataclass
class EvidenceChunk:
    """A chunk of full-text evidence attributed to a numbered source."""
    text: str
    source_idx: int              # 1-based index into the report's source list
    source_url: str
    source_title: str
    source_tier: str
    chunk_index: int = 0
    score: float = 0.0           # relevance score for the current query
    match_reason: str = ""       # "semantic" | "keyword"

    @property
    def citation_tag(self) -> str:
        return f"[S{self.source_idx}]"


@dataclass
class EvidenceCorpus:
    """Full-text evidence corpus for one research run."""
    chunks: List[EvidenceChunk] = field(default_factory=list)
    fetched_sources: List[str] = field(default_factory=list)   # URLs that yielded text
    failed_sources: List[str] = field(default_factory=list)    # URLs that failed
    used_semantic: bool = False
    total_chars: int = 0

    def __len__(self) -> int:
        return len(self.chunks)


# ═══════════════════════════════════════════════════════════════
# FULL-TEXT FETCHING
# ═══════════════════════════════════════════════════════════════


def _source_score(s: Dict) -> float:
    """Score used to rank which sources deserve full-text fetching."""
    score = s.get("adjusted_score", s.get("score", s.get("similarity", 0.0)) or 0.0)
    # Tier bonus: P1 evidence is worth fetching even at lower similarity
    tier_bonus = {"P1": 0.15, "P2": 0.08, "P3": 0.0, "UNV": -0.05}
    return float(score) + tier_bonus.get(s.get("tier", "UNV"), 0.0)


async def _fetch_pdf_text(source: Dict) -> str:
    """Fetch full text from an open-access PDF (Unpaywall → pdf_ingest)."""
    doi = source.get("doi") or (source.get("extra", {}) or {}).get("doi")
    if not doi:
        return ""
    try:
        from tools.search_sources import find_open_access_pdf
        from ingest.pdf_ingest import download_pdf, extract_text_from_pdf

        pdf_url = await find_open_access_pdf(doi)
        if not pdf_url:
            return ""

        pdf_path = await asyncio.wait_for(download_pdf(pdf_url), timeout=EVIDENCE_FETCH_TIMEOUT + 30)
        if not pdf_path:
            return ""

        result = extract_text_from_pdf(pdf_path)
        text = (result.get("text", "") or "") if isinstance(result, dict) else str(result)
        if text.strip():
            logger.info(f"Evidence PDF fetched via Unpaywall: {doi} ({len(text)} chars)")
        return text[:EVIDENCE_MAX_CHARS_PER_SOURCE]

    except Exception as e:
        logger.debug(f"PDF evidence fetch failed for {doi}: {e}")
        return ""


async def _fetch_scraped_text(source: Dict) -> str:
    """Fetch full text via the 3-tier scrape fallback (httpx → Jina → Firecrawl)."""
    url = source.get("url", "")
    if not url or not url.startswith("http"):
        return ""
    try:
        from tools.live_scraper import _scrape_url_with_fallback

        result = await asyncio.wait_for(_scrape_url_with_fallback(url), timeout=EVIDENCE_FETCH_TIMEOUT)
        if result and result.success and result.markdown and not result.markdown.startswith("[SCRAPE_FAILED"):
            return result.markdown[:EVIDENCE_MAX_CHARS_PER_SOURCE]
    except Exception as e:
        logger.debug(f"Scrape evidence fetch failed for {url}: {e}")
    return ""


async def _fetch_web_text(source: Dict) -> str:
    """Full text for web/grey-literature records via the bridge page_reader.

    JS-rendered sites (Wix-hosted NGO reports, Squarespace news pages) defeat
    plain httpx scraping; the bridge's page_reader renders them. Only web
    records (source_name 'zai_web' / extra.web) take this route, and an
    empty result falls through to the scrape chain — never a dead end.
    """
    is_web = _is_web_record(source)
    if not is_web:
        return ""
    try:
        from tools.search_sources import fetch_page_text

        text = await asyncio.wait_for(
            fetch_page_text(source.get("url", "")),
            timeout=EVIDENCE_FETCH_TIMEOUT,
        )
        if text and len(text) > 300:
            logger.info(
                f"Evidence page fetched via page_reader: {source.get('url', '')[:80]} ({len(text)} chars)"
            )
        return text[:EVIDENCE_MAX_CHARS_PER_SOURCE]
    except Exception as e:
        logger.debug(f"page_reader evidence fetch failed for {source.get('url', '')}: {e}")
        return ""


async def fetch_source_fulltext(source: Dict) -> str:
    """
    Fetch full text for a single source.

    Order: OA PDF (Unpaywall, best for academic sources) → bridge page_reader
    (web/grey-literature records, JS-rendered pages) → live scrape (3-tier
    fallback). Returns "" when all routes fail. Snippet-only sources still
    work downstream — they just contribute fewer evidence chunks.
    """
    # 1. Open-access PDF route (academic sources with DOIs)
    text = await _fetch_pdf_text(source)
    if text and len(text) > 500:
        return text

    # 2. Bridge page_reader route (web/grey-literature records)
    text = await _fetch_web_text(source)
    if text and len(text) > 500:
        return text

    # 3. Scrape route
    text = await _fetch_scraped_text(source)
    return text or ""


def _is_web_record(s: Dict) -> bool:
    """True for grey-literature records retrieved via the z-ai web provider."""
    return s.get("source_name") == "zai_web" or bool((s.get("extra") or {}).get("web"))


async def build_evidence_corpus(
    sources: List[Dict],
    top_n: int = EVIDENCE_TOP_SOURCES,
    web_slots: int = 2,
) -> EvidenceCorpus:
    """
    Build a full-text evidence corpus from the top-ranked sources.

    Selection: rank by tier-adjusted score, take top_n, fetch all in
    parallel, chunk each text into ~400-token pieces, and tag every chunk
    with the source's 1-BASED index in the caller's source list (so report
    citations [S{n}] match the final SOURCES section).

    WEB RESERVATION: web/grey-literature records carry no upstream score,
    so pure score ranking can leave them all unfetched even when they hold
    the empirical heart of the story (live case: the TranspariMED Nordic
    page was cited [S3] but its 475/83,903 numbers never reached a drafter
    because snippet-only records scored 0). Up to `web_slots` highest-ranked
    web records outside the top_n are therefore always fetched too.

    Args:
        sources: The final, deduplicated, ranked source list
        top_n: How many top sources to fetch full text for
        web_slots: Reserved full-text fetches for web/grey-literature records

    Returns:
        EvidenceCorpus (possibly empty — callers must handle gracefully)
    """
    corpus = EvidenceCorpus()
    if not sources:
        return corpus

    # Rank candidate sources for fetching (never mutate caller's order)
    ranked = sorted(
        range(len(sources)),
        key=lambda i: _source_score(sources[i]),
        reverse=True,
    )
    candidates = ranked[:top_n]

    # Web reservation: highest-ranked web records outside the score cut
    if web_slots > 0:
        chosen = set(candidates)
        web_reserved = [
            i for i in ranked
            if i not in chosen and _is_web_record(sources[i])
        ][:web_slots]
        if web_reserved:
            candidates = candidates + web_reserved
            logger.info(
                f"Evidence corpus: reserved {len(web_reserved)} web/grey-literature "
                f"fetch slots outside the top-{top_n} score cut"
            )

    async def _fetch_one(idx: int) -> tuple:
        text = await fetch_source_fulltext(sources[idx])
        return idx, text

    results = await asyncio.gather(*[_fetch_one(i) for i in candidates], return_exceptions=True)

    from ingest.chunker import chunk_fixed_size

    for result in results:
        if isinstance(result, Exception):
            continue
        idx, text = result
        source = sources[idx]
        url = source.get("url", "")

        if not text or len(text) < 500:
            if url:
                corpus.failed_sources.append(url)
            continue

        corpus.fetched_sources.append(url)
        corpus.total_chars += len(text)

        chunks = chunk_fixed_size(
            text,
            chunk_size_tokens=EVIDENCE_CHUNK_TOKENS,
            overlap_pct=EVIDENCE_CHUNK_OVERLAP,
        )
        for c in chunks:
            corpus.chunks.append(EvidenceChunk(
                text=c.text,
                source_idx=idx + 1,  # 1-based: matches [S1], [S2], ... citations
                source_url=url,
                source_title=source.get("title", ""),
                source_tier=source.get("tier", "UNV"),
                chunk_index=c.chunk_index,
            ))

    corpus.used_semantic = bool(os.getenv("CLOUDFLARE_API_TOKEN") or os.getenv("OPENAI_API_KEY"))
    logger.info(
        f"Evidence corpus built: {len(corpus.chunks)} chunks from "
        f"{len(corpus.fetched_sources)} sources "
        f"({len(corpus.failed_sources)} failed, {corpus.total_chars} chars)"
    )
    return corpus


# ═══════════════════════════════════════════════════════════════
# TARGETED RETRIEVAL OVER THE EVIDENCE CORPUS
# ═══════════════════════════════════════════════════════════════


def _keyword_score(query: str, text: str) -> float:
    """Zero-credential relevance: weighted term-overlap (Jaccard-like)."""
    q_terms = set(re.findall(r'\b\w{4,}\b', query.lower()))
    if not q_terms:
        return 0.0
    t_terms = set(re.findall(r'\b\w{4,}\b', text.lower()))
    overlap = q_terms & t_terms
    if not overlap:
        return 0.0
    return len(overlap) / math.sqrt(len(q_terms))


async def _semantic_scores(query: str, texts: List[str]) -> Optional[List[float]]:
    """Embed query + chunks, return cosine scores. None if embeddings unavailable."""
    try:
        from ingest.embedder import embed_texts
        import asyncio as _asyncio

        texts_to_embed = [query] + list(texts)
        embeddings = await embed_texts(texts_to_embed)
        if not embeddings or len(embeddings) < len(texts_to_embed):
            return None

        def _cosine(a: List[float], b: List[float]) -> float:
            dot = sum(x * y for x, y in zip(a, b))
            na = math.sqrt(sum(x * x for x in a)) or 1e-9
            nb = math.sqrt(sum(y * y for y in b)) or 1e-9
            return dot / (na * nb)

        q_emb = embeddings[0]
        scores = [_cosine(q_emb, e) for e in embeddings[1:]]
        # Degenerate embedding output (provider outage returns zero vectors —
        # live-seen during a Cloudflare neuron-quota outage) would rank chunks
        # arbitrarily while claiming semantic validity. Detect and fall back
        # to honest keyword scoring instead.
        if all(s <= 1e-9 for s in scores):
            logger.info("Semantic scores degenerate (zero vectors) — falling back to keyword scoring")
            return None
        return scores
    except Exception as e:
        logger.debug(f"Semantic evidence scoring unavailable ({e}); using keyword scores")
        return None


async def retrieve_evidence(
    corpus: EvidenceCorpus,
    query: str,
    k: int = EVIDENCE_CHUNKS_PER_SECTION,
    source_idxs: Optional[List[int]] = None,
) -> List[EvidenceChunk]:
    """
    Retrieve the top-k evidence chunks relevant to a query (or section draft).

    Tries semantic scoring first (embeddings); falls back to keyword overlap
    with zero credentials. Optionally restricts to specific source indexes —
    used by the critique pass to pull supporting evidence for a disputed claim.

    Args:
        corpus: Built evidence corpus
        query: Section question or claim text
        k: Number of chunks to return
        source_idxs: Optional restriction to specific 1-based source indexes

    Returns:
        Top-k EvidenceChunks sorted by score desc
    """
    if not corpus.chunks:
        return []

    pool = corpus.chunks
    if source_idxs:
        allowed = set(source_idxs)
        pool = [c for c in pool if c.source_idx in allowed]
        if not pool:
            return []

    texts = [c.text for c in pool]
    scores: Optional[List[float]] = None

    if corpus.used_semantic:
        scores = await _semantic_scores(query, texts)

    if scores is None:
        scores = [_keyword_score(query, t) for t in texts]
        corpus.used_semantic = False
    else:
        corpus.used_semantic = True

    for chunk, score in zip(pool, scores):
        chunk.score = float(score)
        chunk.match_reason = "semantic" if corpus.used_semantic else "keyword"

    ranked = sorted(pool, key=lambda c: c.score, reverse=True)
    return ranked[:k]


def format_evidence_for_context(chunks: List[EvidenceChunk]) -> str:
    """
    Format evidence chunks into a context block with [S{n}] attribution.

    The numbering matches the report's numbered SOURCES section, so the LLM
    can cite [S2] and the citation-integrity pass can verify the mapping.
    """
    if not chunks:
        return ""
    parts = []
    seen_pairs = set()
    for c in chunks:
        key = (c.source_idx, c.chunk_index)
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        parts.append(f"[S{c.source_idx}|{c.source_tier}] {c.source_title}\n{c.text}")
    return "\n\n---\n\n".join(parts)


def evidence_stats(corpus: EvidenceCorpus) -> Dict:
    """Summary stats for report metadata / observability."""
    return {
        "chunks": len(corpus.chunks),
        "fetched_sources": len(corpus.fetched_sources),
        "failed_sources": len(corpus.failed_sources),
        "total_chars": corpus.total_chars,
        "semantic_scoring": corpus.used_semantic,
    }
