"""
APEX 2.0 Competitive Research Engine — Verification, Iteration, Structured Output, and LLM Wiki.

This module implements the 5 upgrades that make APEX competitive with
Perplexity, Elicit, and Consensus, PLUS the APEX 2.0 LLM Wiki pattern:

1. Source Tier Enforcement — Hard domain-to-tier mapping + temporal decay
2. Verification Loop — Multi-source corroboration with epistemic marking
3. Parallel Orchestration — Failure handling, graceful degradation
4. Research Report Mode — Structured output with claim-evidence maps
5. Iterative Research Loop — Opt-in multi-cycle with gap identification
6. Structured Extraction — Claim/evidence extraction from P1 sources

APEX 2.0 additions:
7. LLM Wiki Pattern — Raw sources (immutable) → Wiki (LLM-compiled) → Schema (config)
8. Page Lifecycle (Synthadoc) — draft → active → stale → contradicted → archived
9. Hot Cache (Claude-Obsidian) — Session continuity across research sessions
10. SciMem Architecture — Scientific memory with provenance tracking
11. Provenance/Conflict Detection — Track where info came from, detect contradictions
12. Dialogic Wiki — Interactive dialogue around wiki topics
13. Secure Wiki Layer + Concurrency Safety — Optimistic locking, edit conflicts

Design principle: "Perplexity retrieves and summarizes. APEX verifies, structures, and compounds."
"""

import asyncio
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import httpx

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════
# UPGRADE #3: SOURCE TIER ENFORCEMENT
# ═══════════════════════════════════════════════════════════════

# Hard domain-to-tier mapping — ensures source quality before LLM sees it
SOURCE_TIER_DOMAINS = {
    "P1": {
        # Top-tier academic / government / institutional
        "arxiv.org", "pubmed.ncbi.nlm.nih.gov", "ncbi.nlm.nih.gov",
        "nejm.org", "nature.com", "science.org", "lancet.com",
        "cell.com", "pnas.org", "bmj.com",
        "who.int", "cdc.gov", "nih.gov", "fda.gov",
        "sec.gov", "eur-lex.europa.eu", "gov.uk",
        "doi.org", "dl.acm.org", "ieeexplore.ieee.org",
        "jmlr.org", "aclanthology.org", "openreview.net",
        "biorxiv.org", "medrxiv.org", "chemrxiv.org",
        "springer.com", "wiley.com", "oxfordacademic.com",
        "cambridge.org", "plos.org",
    },
    "P2": {
        # Reputable news / industry / professional orgs
        "reuters.com", "bloomberg.com", "economist.com",
        "ft.com", "wsj.com", "nytimes.com", "bbc.com",
        "apa.org", "ieee.org", "acm.org",
        "mit.edu", "stanford.edu", "harvard.edu",
        "semanticscholar.org", "openalex.org",
        "clinicaltrials.gov", "cochrane.org",
        "ourworldindata.org", "pewresearch.org",
        "nber.org", "ssrn.com",
        "hbr.org", "mckinsey.com",
    },
    "P3": {
        # Acceptable but not authoritative
        "medium.com", "towardsdatascience.com",
        "wikipedia.org", "wikidata.org",
        "stackoverflow.com", "github.com",
        "reddit.com", "hackernews.com",
        "youtube.com", "substack.com",
        "quora.com", "researchgate.net",
        "academia.edu",
    },
}

# Build reverse lookup: domain suffix → tier
_DOMAIN_TIER_MAP: Dict[str, str] = {}
for tier, domains in SOURCE_TIER_DOMAINS.items():
    for domain in domains:
        _DOMAIN_TIER_MAP[domain] = tier


def enforce_source_tier(url: str, current_tier: str = "UNV") -> str:
    """
    Enforce source tier based on hard domain mapping.

    If the URL's domain matches a known P1/P2/P3 domain, override
    whatever tier was assigned. If no match, keep current or UNV.

    This eliminates garbage before the LLM ever sees it.

    Args:
        url: Source URL
        current_tier: Currently assigned tier

    Returns:
        Enforced tier (P1/P2/P3/UNV)
    """
    if not url:
        return current_tier

    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()

        # Check exact match first
        if domain in _DOMAIN_TIER_MAP:
            return _DOMAIN_TIER_MAP[domain]

        # Check parent domain (e.g., subdomain.arxiv.org → arxiv.org)
        parts = domain.split(".")
        for i in range(len(parts) - 1):
            parent = ".".join(parts[i:])
            if parent in _DOMAIN_TIER_MAP:
                return _DOMAIN_TIER_MAP[parent]

    except Exception:
        pass

    return current_tier


def tier_enforce(sources: List[Dict]) -> List[Dict]:
    """
    Enforce tier rules across a list of sources.

    Strategy:
    - Re-classify each source by domain
    - If P1 sources exist, return up to 5 P1 sources
    - If only P2, return up to 5 P2 sources
    - P3 only as last resort, max 3, with [UNVERIFIED] warning
    - Always include at least 1 source if available

    NOTE: This hard-truncates to 5 sources and is only appropriate for the
    token-efficient CHAT path. The report path must use select_diverse_sources()
    instead, which preserves breadth (comparative analysis and debate sections
    need 20-30 diverse sources, not 5).

    Args:
        sources: List of source dicts with 'url', 'tier', 'title', etc.

    Returns:
        Filtered, tier-enforced list of sources
    """
    if not sources:
        return []

    # Re-classify each source by domain
    for s in sources:
        url = s.get("url", "")
        current = s.get("tier", "UNV")
        s["tier"] = enforce_source_tier(url, current)

    p1 = [s for s in sources if s.get("tier") == "P1"]
    p2 = [s for s in sources if s.get("tier") == "P2"]
    p3 = [s for s in sources if s.get("tier") == "P3"]
    unv = [s for s in sources if s.get("tier") == "UNV"]

    if p1:
        return p1[:5]
    if p2:
        return p2[:5]
    # P3 as last resort — flag as unverifiable
    result = p3[:3]
    for s in result:
        s["tier_warning"] = True
    # Add UNV sources with explicit flag
    if len(result) < 3 and unv:
        for s in unv[:3 - len(result)]:
            s["tier"] = "UNV"
            s["tier_warning"] = True
            result.append(s)
    return result if result else sources[:3]


# ═══════════════════════════════════════════════════════════════
# CANONICAL DEDUP + DIVERSE SOURCE SELECTION (report-grade)
# ═══════════════════════════════════════════════════════════════

def _normalize_doi(doi: Optional[str]) -> str:
    """
    Normalize a DOI to its canonical lowercase form.

    Handles: "10.1038/xxx", "https://doi.org/10.1038/xxx",
    "http://dx.doi.org/10.1038/xxx", "doi:10.1038/xxx"
    """
    if not doi:
        return ""
    d = str(doi).strip().lower()
    for prefix in ("https://doi.org/", "http://doi.org/",
                   "https://dx.doi.org/", "http://dx.doi.org/", "doi:"):
        if d.startswith(prefix):
            d = d[len(prefix):]
    return d.strip()


def _title_fingerprint(title: Optional[str]) -> str:
    """
    Produce a fuzzy fingerprint for a paper title.

    Lowercase, alphanumeric-only, first 120 chars. Catches the same paper
    returned by arXiv / Semantic Scholar / OpenAlex / Crossref with slightly
    different punctuation or capitalization.
    """
    if not title:
        return ""
    alnum = re.sub(r'[^a-z0-9]+', '', title.lower())
    return alnum[:120]


def dedup_sources_canonical(sources: List[Dict]) -> List[Dict]:
    """
    Deduplicate sources by canonical identity: DOI first, then title fingerprint.

    The same paper discovered via arXiv + Semantic Scholar + OpenAlex would
    otherwise (a) waste context slots and (b) create FALSE independence during
    verification (different domains, same underlying study).

    On merge, keeps the record with the richest metadata (longest snippet),
    accumulates alternate URLs into 'also_found_in', and preserves the best
    score / tier / citation_count / DOI / authors across duplicates.

    Args:
        sources: List of source dicts

    Returns:
        Deduplicated list (order preserved by first appearance)
    """
    if not sources:
        return []

    by_doi: Dict[str, int] = {}
    by_title: Dict[str, int] = {}
    deduped: List[Dict] = []

    def _merge(existing: Dict, dup: Dict) -> None:
        # Richest record wins canonical identity: if the duplicate carries a
        # longer snippet, adopt its URL/title/snippet as primary and demote
        # the old canonical URL to also_found_in.
        if len(dup.get("snippet", "") or "") > len(existing.get("snippet", "") or ""):
            existing_url = existing.get("url", "")
            existing["url"] = dup.get("url", existing_url)
            existing["title"] = dup.get("title", existing.get("title", ""))
            existing["snippet"] = dup.get("snippet", existing.get("snippet", ""))
            alt = existing_url
        else:
            alt = dup.get("url", "")
        if alt and alt != existing.get("url"):
            existing.setdefault("also_found_in", []).append(alt)
        # Best values across duplicates
        for key in ("doi", "published_date", "title"):
            new_val = dup.get(key)
            if new_val and not existing.get(key):
                existing[key] = new_val
        if dup.get("citation_count") is not None:
            old_cc = existing.get("citation_count") or 0
            existing["citation_count"] = max(old_cc, dup["citation_count"])
        if len(dup.get("authors", []) or []) > len(existing.get("authors", []) or []):
            existing["authors"] = dup["authors"]
        # Tier: keep the higher tier (P1 > P2 > P3 > UNV)
        tier_rank = {"P1": 4, "P2": 3, "P3": 2, "UNV": 1}
        if tier_rank.get(dup.get("tier", "UNV"), 1) > tier_rank.get(existing.get("tier", "UNV"), 1):
            existing["tier"] = dup["tier"]
        score_keys = ("adjusted_score", "score", "similarity")
        for key in score_keys:
            if dup.get(key) is not None:
                existing[key] = max(existing.get(key) or 0, dup[key])

    for s in sources:
        doi = _normalize_doi(s.get("doi") or (s.get("extra", {}) or {}).get("doi"))
        fp = _title_fingerprint(s.get("title"))

        match_idx = None
        if doi and doi in by_doi:
            match_idx = by_doi[doi]
        elif fp and fp in by_title:
            match_idx = by_title[fp]

        if match_idx is not None:
            _merge(deduped[match_idx], s)
        else:
            if doi:
                s["doi"] = doi  # persist normalized form
                by_doi[doi] = len(deduped)
            if fp:
                by_title[fp] = len(deduped)
            deduped.append(s)

    removed = len(sources) - len(deduped)
    if removed:
        logger.info(f"Canonical dedup: {len(sources)} → {len(deduped)} sources ({removed} duplicates merged)")
    return deduped


_STOPWORDS = {
    "what", "which", "that", "this", "with", "from", "have", "has", "had",
    "are", "were", "been", "their", "them", "they", "about", "into", "over",
    "than", "then", "when", "where", "while", "some", "such", "only", "more",
    "most", "much", "many", "also", "between", "among", "both", "each",
    "does", "did", "doing", "done", "make", "made", "make", "take", "took",
    "never", "made", "public", "there", "here", "will", "would", "could",
    "should", "might", "must", "shall", "may", "can", "cannot", "using",
    "used", "use", "new", "non", "not", "and", "for", "the", "its",
}


def _normalize_term(w: str) -> str:
    """Light plural normalization — trials→trial, results→result, studies→study.
    Full stemming is overkill; singular/plural confusion measurably hurts the
    relevance gate on clinical vocabulary (trial(s), result(s), patient(s))."""
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 3 and w.endswith("s") and not w.endswith(("ss", "us", "is")):
        return w[:-1]
    return w


def _content_words(text: str) -> set:
    """Lowercased alphanumeric content words (len>=4, stopwords removed,
    plural-normalized)."""
    words = re.findall(r"[a-z0-9]{4,}", (text or "").lower())
    return {_normalize_term(w) for w in words if w not in _STOPWORDS}


# ── Generic-academic vocabulary lives in citation_integrity (shared with
# scan_off_topic_padding); imported there to keep one source of truth.
from agent.citation_integrity import (  # noqa: E402
    GENERIC_ACADEMIC_TERMS,
    discriminative_vocab as _discriminative_vocab,
    shared_terms as _source_shared_terms,
)


def _source_relevance(source: Dict, query_vocab: set) -> float:
    """Fraction of query content words appearing in the source title+snippet."""
    if not query_vocab:
        return 1.0
    text = f"{(source.get('title') or '')} {(source.get('snippet') or '')}".lower()
    shared = sum(1 for w in query_vocab if w in text)
    return shared / len(query_vocab)


# ── Junk-source detection (defense-in-depth behind provider filters) ──
# Live failure: Crossref component DOIs ("Supplemental Information 1: ...",
# figure/table records) carry the parent paper's keywords, pass lexical
# relevance gates, and get cited as methodology for unrelated claims.
_JUNK_SOURCE_TITLE_RE = re.compile(
    r"^(supplemental information|supplementary (material|information|file|table|figure)|"
    r"table s?\d|figure s?\d|appendix [a-z0-9]|supporting information|"
    r"front matter|back matter|cover image|table of contents)\b",
    re.I,
)
_ENCYCLOPEDIA_DOMAINS = {"wikipedia.org", "britannica.com", "dbpedia.org", "wikidata.org"}
# A URL whose final path segment is one of these is a directory listing,
# not a citable resource (live case: https://huggingface.co/papers/ with no
# paper ID). Conservative set — only bare section words, nothing else.
_BARE_SECTION_SEGMENTS = {
    "papers", "pubs", "publications", "articles", "issues", "abstracts",
    "docs", "blog", "news", "search", "topics", "categories", "collections",
    "datasets", "models", "journals", "books", "reports", "publications",
}


def _is_junk_source(source: Dict) -> bool:
    """Non-paper artifacts no report should cite: supplementary-file records
    and URLs that point at a bare directory/section root."""
    title = (source.get("title") or "").strip()
    if title and _JUNK_SOURCE_TITLE_RE.match(title):
        return True
    url = (source.get("url") or "").strip()
    m = re.match(r"^[a-z]+://[^/]+(/.*)?$", url)
    if m:
        path = (m.group(1) or "").strip("/")
        if not path:
            return True  # bare domain, no resource at all
        last = path.rsplit("/", 1)[-1].strip().lower()
        if last in _BARE_SECTION_SEGMENTS:
            return True
    return False


def select_diverse_sources(
    sources: List[Dict],
    max_sources: int = 25,
    per_domain_cap: int = 3,
    min_p2: int = 2,
    query: str = "",
    query_variants: Optional[List[str]] = None,
    min_relevance: float = 0.15,
    relevance_floor: int = 5,
    screening: Optional[Dict] = None,
) -> List[Dict]:
    """
    Report-grade source selection: preserve BREADTH instead of hard-truncating
    to 5 P1 sources (the old tier_enforce behavior that discarded ~90% of
    gathered evidence).

    Selection strategy:
    0. RELEVANCE GATE — drop sources whose title+snippet share (nearly) no
       content words with the query or its expansion variants. Live retrieval
       grabs loosely-related papers (a drug page, an outreach program); citing
       them is padding. A small floor of top-relevance survivors is guaranteed
       so a genuinely niche query never ends up with zero sources.
    1. Rank all sources by adjusted_score (temporal decay already applied)
    2. Walk the ranked list, capping per-domain representation (default 3)
       so one prolific domain doesn't crowd out diverse perspectives
    3. Guarantee at least `min_p2` P2 sources (industry/policy perspective)
       and reserve slots so P3 counter-evidence survives when useful
    4. Stop at max_sources

    Args:
        sources: Ranked source dicts
        max_sources: Maximum sources to return (report context allows 20-30)
        per_domain_cap: Max sources per base domain
        min_p2: Minimum P2 sources to include when available
        query: Original query (drives the relevance gate when provided)
        query_variants: Expanded query variants (unioned into the vocabulary)
        min_relevance: Minimum shared-content-word fraction to keep a source
        relevance_floor: If fewer sources survive the gate, top up with the
            highest-relevance rejects so selection never degenerates

    Returns:
        Selected, diversity-balanced list of sources
    """
    if not sources:
        return []

    # Re-enforce tier from URL so selection sees accurate tiers
    for s in sources:
        s["tier"] = enforce_source_tier(s.get("url", ""), s.get("tier", "UNV"))

    # ── Pass -1: junk filter — non-paper artifacts never enter selection ──
    pre_junk = len(sources)
    sources = [s for s in sources if not _is_junk_source(s)]
    if len(sources) < pre_junk:
        logger.info(
            f"Junk filter: dropped {pre_junk - len(sources)} non-paper sources "
            f"(supplement/figure records, bare-directory URLs)"
        )
    if screening is not None:
        screening["junk_filtered"] = pre_junk - len(sources)

    # ── Pass 0: relevance gate ──
    dropped_irrelevant = 0
    if query:
        core_vocab = _content_words(query)
        disc_vocab = _discriminative_vocab(core_vocab)
        vocab = set(core_vocab)
        for v in (query_variants or []):
            vocab |= _content_words(v)
        if vocab:
            def _gated_relevance(s: Dict) -> float:
                """Core-question relevance with DISCRIMINATIVE authority.

                Two rules stack:
                1. Core-question authority (run7 repair): sources sharing any
                   core vocabulary score on core vocab alone; zero-core
                   sources fall to 0.6× expansion-vocab.
                2. Discriminative authority (run8 repair): core overlap that
                   reduces ENTIRELY to generic academic terms ({meta,
                   analysis, public, database, ...}) is zero topical
                   purchase. "Meta-analysis of hypoxic transcriptomes from
                   public databases" shares 4 core words with "Meta-analysis
                   of unpublished negative-result trials ... public
                   databases" and shares NONE of {unpublished, negative,
                   result, trial} — it must not ride the core lane.

                A source with zero discriminative overlap scores 0.0 (the
                expansion lane is NOT opened for it — the expansion vocab is
                academic paraphrase and generic-dense jargon would re-admit
                exactly this class). Canonical backfill happens AFTER
                selection, so registry-attested sources are unaffected here.
                """
                if disc_vocab:
                    if not _source_shared_terms(s, disc_vocab):
                        return 0.0
                core_rel = _source_relevance(s, core_vocab) if core_vocab else 0.0
                if core_rel > 0:
                    return core_rel
                return 0.6 * _source_relevance(s, vocab)

            scored = [(_gated_relevance(s), s) for s in sources]
            kept = [s for rel, s in scored if rel >= min_relevance]
            dropped_irrelevant = len(sources) - len(kept)
            floor_rescued: List[Dict] = []
            if len(kept) < relevance_floor:
                # Top up with the highest-relevance rejects (not highest-score)
                by_rel = sorted(scored, key=lambda t: t[0], reverse=True)
                seen = {id(s) for s in kept}
                for rel, s in by_rel:
                    if len(kept) >= relevance_floor:
                        break
                    if id(s) not in seen:
                        kept.append(s)
                        seen.add(id(s))
                        if rel < min_relevance:
                            floor_rescued.append(s)
                kept = kept[:relevance_floor]
            if screening is not None:
                screening["relevance_kept"] = len(kept)
                screening["relevance_rejected"] = [
                    {
                        "title": (s.get("title") or "")[:120],
                        "url": (s.get("url") or "")[:120],
                        "relevance": round(rel, 3),
                    }
                    for rel, s in sorted(scored, key=lambda t: t[0])
                    if rel < min_relevance and not any(s is k for k in kept)
                ][:10]
                screening["floor_rescued"] = [
                    {"title": (s.get("title") or "")[:120]} for s in floor_rescued
                ]
            if dropped_irrelevant:
                logger.info(
                    f"Relevance gate: dropped {dropped_irrelevant} low-relevance "
                    f"sources (core_vocab={len(core_vocab)}, "
                    f"discriminative={len(disc_vocab)}, full_vocab={len(vocab)}, "
                    f"threshold={min_relevance})"
                )
            sources = kept

    ranked = sorted(
        sources,
        key=lambda s: s.get("adjusted_score", s.get("score", s.get("similarity", 0.0)) or 0.0),
        reverse=True,
    )

    selected: List[Dict] = []
    domain_counts: Dict[str, int] = {}

    def _base_domain(s: Dict) -> str:
        return _extract_domain(s.get("url", "")) or "unknown"

    def _is_encyclopedia(s: Dict) -> bool:
        dom = _base_domain(s)
        return any(dom == e or dom.endswith("." + e) for e in _ENCYCLOPEDIA_DOMAINS)

    # Pass 1: score-ordered walk with per-domain caps + encyclopedia cap.
    # Encyclopedias support background only — at most ONE per report so the
    # technical claims are anchored in primary literature.
    deferred: List[Dict] = []
    encyclopedia_used = 0
    for s in ranked:
        if len(selected) >= max_sources:
            break
        if _is_encyclopedia(s):
            if encyclopedia_used >= 1:
                deferred.append(s)
                continue
            encyclopedia_used += 1
        dom = _base_domain(s)
        if domain_counts.get(dom, 0) >= per_domain_cap:
            deferred.append(s)
            continue
        selected.append(s)
        domain_counts[dom] = domain_counts.get(dom, 0) + 1

    # Pass 2: guarantee P2 representation (industry/policy perspective)
    p2_count = sum(1 for s in selected if s.get("tier") == "P2")
    if p2_count < min_p2:
        for s in ranked:
            if p2_count >= min_p2 or len(selected) >= max_sources:
                break
            if s.get("tier") != "P2" or s in selected:
                continue
            dom = _base_domain(s)
            if domain_counts.get(dom, 0) >= per_domain_cap:
                continue
            selected.append(s)
            domain_counts[dom] = domain_counts.get(dom, 0) + 1
            p2_count += 1

    # Pass 3: fill remaining slots from deferred (domain-capped) sources —
    # but never re-admit encyclopedia sources (cap is one, taken in Pass 1)
    for s in deferred:
        if len(selected) >= max_sources:
            break
        if s in selected or _is_encyclopedia(s):
            continue
        dom = _base_domain(s)
        if domain_counts.get(dom, 0) >= per_domain_cap:
            continue
        selected.append(s)
        domain_counts[dom] = domain_counts.get(dom, 0) + 1

    # Re-sort selected by score for context ordering
    selected.sort(
        key=lambda s: s.get("adjusted_score", s.get("score", s.get("similarity", 0.0)) or 0.0),
        reverse=True,
    )
    logger.info(
        f"Diverse source selection: {len(sources)} → {len(selected)} "
        f"(P1={sum(1 for s in selected if s.get('tier')=='P1')}, "
        f"P2={sum(1 for s in selected if s.get('tier')=='P2')}, "
        f"domains={len(domain_counts)})"
    )
    return selected


# ── Domain-aware temporal decay ──
# A uniform 0.95/year factor is wrong across fields: an AI paper from 2022 is
# often obsolete, while a 2015 clinical trial may still be the strongest
# evidence. Fast-moving fields decay faster; fundamental science barely decays.
# Override via env: APEX_TEMPORAL_DECAY_<DOMAIN> (e.g. APEX_TEMPORAL_DECAY_AI=0.7)
DOMAIN_DECAY_FACTORS: Dict[str, float] = {
    "ai": 0.85,          # ML/AI/NLP: fast-moving, SOTA churns quickly
    "software": 0.88,    # software engineering / systems
    "biotech": 0.90,     # genomics, biotech
    "finance": 0.90,     # markets, monetary policy
    "news": 0.80,        # news/current events decay fastest
    "medicine": 0.97,    # clinical evidence is durable
    "physics": 0.99,     # fundamental science
    "mathematics": 0.995,
}

# Keyword hints used by infer_domain()
_DOMAIN_KEYWORDS: Dict[str, re.Pattern] = {
    "ai": re.compile(r'\b(llm|gpt|transformer|neural|machine learning|deep learning|ai\b|artificial intelligence|language model|diffusion|rlhf|embedding)\b', re.I),
    "software": re.compile(r'\b(software|programming|compiler|database|distributed|cloud|api|kubernetes|microservice)\b', re.I),
    "biotech": re.compile(r'\b(genom|crispr|protein|gene|biotech|sequenc|mrna)\b', re.I),
    "finance": re.compile(r'\b(market|monetary|inflation|stock|fiscal|investment|economy|economic|central bank)\b', re.I),
    "news": re.compile(r'\b(latest|breaking|announced|this week|this month|current|recent)\b', re.I),
    "medicine": re.compile(r'\b(clinical|patient|therapy|treatment|drug|disease|cancer|vaccine|trial|diagnos|symptom)\b', re.I),
    "physics": re.compile(r'\b(quantum|particle|relativity|cosmolog|photon|thermodynamic)\b', re.I),
    "mathematics": re.compile(r'\b(theorem|proof|conjecture|lemma|topolog|algebraic|geometr)\b', re.I),
}


def infer_domain(query: str, classification: str = "") -> str:
    """
    Infer the research domain of a query for domain-aware decay and cache TTL.

    Priority: explicit classification hint > keyword match > "general".
    """
    if classification in DOMAIN_DECAY_FACTORS:
        return classification
    q = query or ""
    best_domain, best_hits = "general", 0
    for domain, pattern in _DOMAIN_KEYWORDS.items():
        hits = len(pattern.findall(q))
        if hits > best_hits:
            best_domain, best_hits = domain, hits
    return best_domain


def get_decay_factor(domain: str) -> float:
    """Get the yearly temporal-decay factor for a domain (env-overridable)."""
    default = 0.95
    if not domain or domain == "general":
        return float(os.getenv("APEX_TEMPORAL_DECAY_GENERAL", str(default)))
    return float(os.getenv(f"APEX_TEMPORAL_DECAY_{domain.upper()}", str(DOMAIN_DECAY_FACTORS.get(domain, default))))


def _extract_source_year(s: Dict) -> Optional[int]:
    """
    Extract publication year from a source, checking multiple metadata
    locations. Fixes the old behavior where undated-looking arXiv sources
    were penalized even when the year was present in metadata.
    """
    # 1. Explicit published_date ("2024-03-15" or "2024")
    pub_date = s.get("published_date", "")
    if pub_date and len(str(pub_date)) >= 4:
        try:
            year = int(str(pub_date)[:4])
            if 1900 <= year <= 2100:
                return year
        except (ValueError, TypeError):
            pass

    # 2. metadata.year / metadata.published
    meta = s.get("metadata", {}) or {}
    for key in ("year", "published", "publication_year"):
        val = meta.get(key)
        if val:
            try:
                year = int(str(val)[:4])
                if 1900 <= year <= 2100:
                    return year
            except (ValueError, TypeError):
                pass

    # 3. arXiv ID embedded in metadata or URL ("2403.12345" → 2024)
    arxiv_id = (s.get("metadata", {}) or {}).get("arxiv_id") or s.get("arxiv_id", "")
    if not arxiv_id:
        url = s.get("url", "")
        m = re.search(r'arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})', url)
        if m:
            arxiv_id = m.group(1)
    if arxiv_id:
        m = re.match(r'(\d{2})\d{2}\.', arxiv_id)
        if m:
            yy = int(m.group(1))
            year = 2000 + yy if yy <= 79 else 1900 + yy
            return year

    # 4. Top-level "year" field
    val = s.get("year")
    if val:
        try:
            year = int(str(val)[:4])
            if 1900 <= year <= 2100:
                return year
        except (ValueError, TypeError):
            pass

    return None


def apply_temporal_decay(
    sources: List[Dict],
    decay_factor: Optional[float] = None,
    current_year: Optional[int] = None,
    domain: Optional[str] = None,
) -> List[Dict]:
    """
    Apply domain-aware temporal decay to source scores.

    A 2024 meta-analysis should outweigh a 2019 RCT on the same question —
    but how much "age" matters depends on the field. AI papers decay ~15%/yr,
    clinical evidence ~3%/yr, mathematics ~0.5%/yr.

    Formula: score *= factor ^ (current_year - source_year)

    Args:
        sources: List of source dicts with 'published_date' and 'score'
        decay_factor: Explicit yearly decay override (skips domain inference)
        current_year: Override current year for testing
        domain: Domain hint ("ai", "medicine", ...). Inferred from source
                content when omitted.

    Returns:
        Sources with adjusted scores (re-sorted by adjusted_score desc)
    """
    year = current_year or datetime.now().year

    for s in sources:
        source_year = _extract_source_year(s)

        if source_year:
            age = max(0, year - source_year)
            factor = decay_factor if decay_factor is not None else (
                get_decay_factor(domain) if domain else get_decay_factor(infer_domain(
                    f"{s.get('title', '')} {s.get('snippet', '')[:200]}"
                ))
            )
            decay = factor ** age
            score = s.get("score", s.get("similarity", 1.0))
            s["adjusted_score"] = score * decay
        else:
            # No date found anywhere — slight penalty for undated sources
            score = s.get("score", s.get("similarity", 1.0))
            s["adjusted_score"] = score * 0.9

    # Re-sort by adjusted score
    sources.sort(key=lambda s: s.get("adjusted_score", 0), reverse=True)
    return sources


# ═══════════════════════════════════════════════════════════════
# UPGRADE #1: VERIFICATION LOOP
# ═══════════════════════════════════════════════════════════════

# Epistemic status markers
EPISTEMIC_ESTABLISHED = "ESTABLISHED"   # 2+ independent P1/P2 sources agree
EPISTEMIC_TENTATIVE = "TENTATIVE"       # Only 1 source, or only P3 sources
EPISTEMIC_CONTESTED = "ACTIVE_DEBATE"   # Sources conflict
EPISTEMIC_SPECULATIVE = "SPECULATIVE"   # Early inference, no direct evidence
EPISTEMIC_UNVERIFIED = "UNVERIFIED"     # No source supports this claim


@dataclass
class VerifiedClaim:
    """A claim verified across multiple sources."""
    statement: str
    epistemic_status: str  # ESTABLISHED, TENTATIVE, ACTIVE_DEBATE, SPECULATIVE, UNVERIFIED
    supporting_sources: List[Dict] = field(default_factory=list)
    conflicting_sources: List[Dict] = field(default_factory=list)
    confidence: float = 0.0  # 0.0-1.0
    evidence_type: str = ""  # RCT, cohort, meta-analysis, expert_opinion, anecdote
    sample_size: Optional[int] = None
    year: Optional[int] = None


@dataclass
class VerificationResult:
    """Result of the verification process."""
    claims: List[VerifiedClaim] = field(default_factory=list)
    total_sources_checked: int = 0
    established_count: int = 0
    tentative_count: int = 0
    contested_count: int = 0
    unverifiable_count: int = 0
    summary: str = ""


def _extract_domain(url: str) -> str:
    """Extract base domain from URL for independence checking."""
    try:
        parsed = urlparse(url)
        parts = parsed.netloc.split(".")
        # Return last 2 parts (e.g., "nature.com")
        if len(parts) >= 2:
            return ".".join(parts[-2:])
        return parsed.netloc
    except Exception:
        return url


def _sources_are_independent(source_a: Dict, source_b: Dict) -> bool:
    """
    Check if two sources are independent (different domain, different authors).

    Two sources from the same domain or same first author are not independent.
    """
    domain_a = _extract_domain(source_a.get("url", ""))
    domain_b = _extract_domain(source_b.get("url", ""))

    if domain_a and domain_b and domain_a == domain_b:
        return False

    # Check author overlap
    authors_a = set(source_a.get("authors", []))
    authors_b = set(source_b.get("authors", []))
    if authors_a and authors_b and authors_a & authors_b:
        return False

    return True


def verify_claims_from_sources(
    claims: List[str],
    sources: List[Dict],
) -> VerificationResult:
    """
    Verify claims by cross-referencing across independent sources.

    Rules:
    - ESTABLISHED: 2+ independent P1/P2 sources agree
    - TENTATIVE: Only 1 source, or only P3 sources
    - ACTIVE_DEBATE: Sources explicitly conflict
    - UNVERIFIED: No source supports the claim

    Args:
        claims: List of claim strings to verify
        sources: List of source dicts with url, tier, snippet, title, etc.

    Returns:
        VerificationResult with verified claims and statistics
    """
    verified_claims = []

    for claim in claims:
        supporting = []
        conflicting = []

        claim_lower = claim.lower()
        # Extract key terms from the claim for matching
        claim_terms = set(re.findall(r'\b\w{4,}\b', claim_lower))

        for source in sources:
            snippet = (source.get("snippet", "") or source.get("content", "") or "").lower()
            title = (source.get("title", "") or "").lower()
            source_text = f"{title} {snippet}"

            if not source_text.strip():
                continue

            # Check term overlap
            source_terms = set(re.findall(r'\b\w{4,}\b', source_text))
            overlap = claim_terms & source_terms
            overlap_ratio = len(overlap) / max(len(claim_terms), 1)

            if overlap_ratio < 0.3:
                continue  # Not relevant enough

            # Check for negation/conflict signals
            is_negative = any(
                neg in source_text
                for neg in [
                    "does not", "did not", "no evidence", "not associated",
                    "contradicts", "refutes", "fails to", "unable to replicate",
                    "not supported", "disputed", "controversial", "debated",
                    "however,", "in contrast,", "conversely,",
                ]
            )

            # Check for positive/support signals
            is_positive = any(
                pos in source_text
                for pos in [
                    "found that", "shows that", "demonstrated", "confirmed",
                    "evidence suggests", "results indicate", "significantly",
                    "associated with", "linked to", "supports", "proves",
                    "established", "meta-analysis", "systematic review",
                ]
            )

            entry = {
                "url": source.get("url", ""),
                "title": source.get("title", ""),
                "tier": source.get("tier", "UNV"),
                "overlap_ratio": round(overlap_ratio, 2),
                "supports": is_positive and not is_negative,
                "conflicts": is_negative,
            }

            if is_negative and overlap_ratio > 0.4:
                conflicting.append(entry)
            elif is_positive or overlap_ratio > 0.5:
                supporting.append(entry)

        # Determine epistemic status
        # Count independent supporting sources
        independent_supporting = []
        for s in supporting:
            is_indep = all(
                _sources_are_independent(s, existing) for existing in independent_supporting
            )
            if is_indep:
                independent_supporting.append(s)

        p1_p2_supporting = [s for s in supporting if s.get("tier") in ("P1", "P2")]

        if len(independent_supporting) >= 2 and len(p1_p2_supporting) >= 1:
            status = EPISTEMIC_ESTABLISHED
            confidence = min(0.95, 0.5 + 0.15 * len(independent_supporting))
        elif conflicting and supporting:
            status = EPISTEMIC_CONTESTED
            confidence = 0.3
        elif len(supporting) >= 1:
            if len(independent_supporting) >= 2:
                status = EPISTEMIC_ESTABLISHED
                confidence = 0.7
            elif p1_p2_supporting:
                status = EPISTEMIC_TENTATIVE
                confidence = 0.5
            else:
                status = EPISTEMIC_TENTATIVE
                confidence = 0.3
        elif conflicting:
            status = EPISTEMIC_CONTESTED
            confidence = 0.2
        else:
            status = EPISTEMIC_UNVERIFIED
            confidence = 0.0

        # Determine evidence type from sources
        evidence_type = ""
        all_text = " ".join(s.get("snippet", "") for s in supporting).lower()
        if "meta-analysis" in all_text:
            evidence_type = "meta-analysis"
        elif "systematic review" in all_text:
            evidence_type = "systematic-review"
        elif "randomized" in all_text or "rct" in all_text:
            evidence_type = "RCT"
        elif "cohort" in all_text:
            evidence_type = "cohort"
        elif "case-control" in all_text:
            evidence_type = "case-control"
        elif "cross-sectional" in all_text:
            evidence_type = "cross-sectional"
        elif any(s.get("tier") == "P1" for s in supporting):
            evidence_type = "peer-reviewed"
        elif supporting:
            evidence_type = "observation"

        verified_claims.append(VerifiedClaim(
            statement=claim,
            epistemic_status=status,
            supporting_sources=supporting[:5],
            conflicting_sources=conflicting[:5],
            confidence=round(confidence, 2),
            evidence_type=evidence_type,
        ))

    # Build statistics
    established = sum(1 for c in verified_claims if c.epistemic_status == EPISTEMIC_ESTABLISHED)
    tentative = sum(1 for c in verified_claims if c.epistemic_status == EPISTEMIC_TENTATIVE)
    contested = sum(1 for c in verified_claims if c.epistemic_status == EPISTEMIC_CONTESTED)
    unverifiable = sum(1 for c in verified_claims if c.epistemic_status == EPISTEMIC_UNVERIFIED)

    return VerificationResult(
        claims=verified_claims,
        total_sources_checked=len(sources),
        established_count=established,
        tentative_count=tentative,
        contested_count=contested,
        unverifiable_count=unverifiable,
    )


async def verify_claim_with_search(
    claim: str,
    max_sources: int = 10,
) -> VerifiedClaim:
    """
    Verify a single claim by searching for corroborating/contradicting sources.

    1. Search academic sources
    2. Search web sources
    3. Cross-reference and determine epistemic status

    Args:
        claim: The claim to verify
        max_sources: Max sources to check

    Returns:
        VerifiedClaim with epistemic status
    """
    from tools.search_sources import search_router

    # Parallel search: academic + web
    academic_task = search_router(claim, classification="academic", max_results=max_sources)
    web_task = search_router(claim, classification="web", max_results=max_sources)

    academic_results, web_results = await asyncio.gather(
        academic_task, web_task, return_exceptions=True
    )

    all_sources = []
    for results in [academic_results, web_results]:
        if isinstance(results, list):
            for r in results:
                all_sources.append({
                    "url": r.url,
                    "title": r.title,
                    "snippet": r.snippet,
                    "tier": enforce_source_tier(r.url, r.source_tier),
                    "authors": r.authors if hasattr(r, 'authors') else [],
                })

    # Run verification
    verification = verify_claims_from_sources([claim], all_sources)
    if verification.claims:
        return verification.claims[0]

    return VerifiedClaim(
        statement=claim,
        epistemic_status=EPISTEMIC_UNVERIFIED,
        confidence=0.0,
    )


# ═══════════════════════════════════════════════════════════════
# UPGRADE #5: PARALLEL ORCHESTRATION
# ═══════════════════════════════════════════════════════════════

@dataclass
class ParallelResearchResult:
    """Result from parallel research with failure tracking."""
    successful_sources: List[str] = field(default_factory=list)
    failed_sources: List[str] = field(default_factory=list)
    results: List[Dict] = field(default_factory=list)
    total_latency_ms: int = 0


async def parallel_research(
    query: str,
    classification: str = "academic",
    max_results_per_source: int = 3,
    max_sources: int = 12,
) -> ParallelResearchResult:
    """
    Execute parallel research across all sources with graceful failure handling.

    Key features:
    - All sources queried simultaneously (no sequential blocking)
    - Individual failures don't crash the pipeline
    - Failed sources tracked and reported
    - Canonical dedup (DOI/title) + diverse selection preserving breadth

    Args:
        query: Research query
        classification: Query type for routing
        max_results_per_source: Max results per source
        max_sources: Final source budget after dedup/diversity selection
            (chat callers keep the default; report callers pass 25+)

    Returns:
        ParallelResearchResult with successful/failed tracking
    """
    from tools.search_sources import search_router, SOURCE_FUNCTIONS

    start = time.time()

    # Get appropriate sources for classification
    from tools.search_sources import SOURCE_ROUTING
    routing = SOURCE_ROUTING.get(classification, SOURCE_ROUTING["academic"])
    source_names = routing.get("primary", []) + routing.get("secondary", [])

    # Launch all searches in parallel with exception handling
    tasks = {}
    for name in source_names:
        func = SOURCE_FUNCTIONS.get(name)
        if func:
            tasks[name] = func(query, max_results_per_source)

    if not tasks:
        # Absolute fallback: DuckDuckGo
        func = SOURCE_FUNCTIONS.get("duckduckgo")
        if func:
            tasks["duckduckgo"] = func(query, max_results_per_source)

    # Gather with return_exceptions=True — never crash on individual failure
    task_names = list(tasks.keys())
    task_coroutines = list(tasks.values())

    completed = await asyncio.gather(*task_coroutines, return_exceptions=True)

    successful_sources = []
    failed_sources = []
    all_results = []

    for name, result in zip(task_names, completed):
        if isinstance(result, Exception):
            failed_sources.append(f"{name}: {str(result)[:50]}")
            logger.debug(f"Source {name} failed: {result}")
        elif isinstance(result, list) and len(result) > 0:
            successful_sources.append(name)
            for r in result:
                all_results.append({
                    "url": r.url,
                    "title": r.title,
                    "snippet": r.snippet,
                    "tier": enforce_source_tier(r.url, r.source_tier),
                    "source_name": r.source_name,
                    "category": str(r.source_category) if hasattr(r, 'source_category') else "",
                    "published_date": r.published_date if hasattr(r, 'published_date') else None,
                })
        else:
            # Empty results — not a failure, just no matches
            successful_sources.append(name)

    # Canonical dedup: merge the same paper returned by multiple APIs
    # (arXiv + Semantic Scholar + OpenAlex would otherwise waste context
    # slots and fake "independent corroboration" during verification)
    all_results = dedup_sources_canonical(all_results) if all_results else []

    # Diverse selection (NOT the old tier_enforce hard-truncation to 5)
    all_results = select_diverse_sources(all_results, max_sources=max_sources) if all_results else []

    # Domain-aware temporal decay
    all_results = apply_temporal_decay(all_results, domain=infer_domain(query, classification))

    latency_ms = int((time.time() - start) * 1000)

    return ParallelResearchResult(
        successful_sources=successful_sources,
        failed_sources=failed_sources,
        results=all_results,
        total_latency_ms=latency_ms,
    )


# ═══════════════════════════════════════════════════════════════
# UPGRADE #4: RESEARCH REPORT MODE
# ═══════════════════════════════════════════════════════════════


def _web_query_rewrite(query: str, max_words: int = 8) -> str:
    """Deterministic keyword rewrite of the primary query for web indexes.

    HyDE variants wrap the question in academic prose ("Technical analysis
    of...") which web engines match far worse than concrete keywords. This
    rewrite is pure query-derived vocabulary (no hard-coded topics): stopword
    stripping + dedupe, first `max_words` content words kept.
    """
    words = re.findall(r"[a-zA-Z0-9][a-zA-Z0-9\-]{2,}", query or "")
    seen, out = set(), []
    for w in words:
        k = w.lower()
        if k in _STOPWORDS or k in seen or len(w) < 3:
            continue
        seen.add(k)
        out.append(w)
        if len(out) >= max_words:
            break
    return " ".join(out)


def _load_web_seed_urls() -> List[Dict]:
    """Operator-attested fallback URLs for grey-literature retrieval.

    Read from APEX_WEB_SEED_URLS (JSON list of {url,title,snippet,
    published_date}). Used ONLY when live web search returns zero usable
    hits; every seed still passes the same governance stack as live hits
    (canonicalization, DROP filter, tier enforcement, junk + relevance
    gates) and is labeled seeded=True for the provenance appendix.
    Malformed entries are skipped, never raised.
    """
    raw = os.getenv("APEX_WEB_SEED_URLS", "").strip()
    if not raw:
        return []
    try:
        data = json.loads(raw)
    except (ValueError, TypeError) as e:
        logger.warning(f"APEX_WEB_SEED_URLS is not valid JSON, ignoring: {e}")
        return []
    if not isinstance(data, list):
        return []
    return [d for d in data if isinstance(d, dict) and d.get("url")]


async def _web_sources_for_report(
    query: str,
    variants: List[str],
    depth: str = "quick",
) -> Tuple[List[Dict], Dict]:
    """
    Grey-literature retrieval: general web search via the keyless z-ai bridge.

    The academic fan-out structurally cannot see NGO/transparency reports,
    regulator notices, or institutional news pages — often the empirical
    material that motivates the question (live case: TranspariMED's Nordic
    unreported-trials reports are indexed by no academic API, yet carry the
    475-trial, Rigshospitalet, and Haukeland numbers the literature then
    studies). Web records enter the pool as ORDINARY members: they flow
    through canonical dedup, the junk/encyclopedia gates, the relevance
    gate, and the deterministic bibliography, honestly labeled (tier from
    grey-literature governance, category 'web', source_name 'zai_web').

    Budget is deliberately small: the primary query + at most one variant
    (quick) / two variants (thorough), 8 hits per query.

    Returns (source_dicts, stats) — stats are reported verbatim in the
    report's evidence metadata so downstream consumers know whether web
    retrieval ran at all (available=false when the bridge is down).
    """
    stats = {
        "enabled": os.getenv("APEX_WEB_SEARCH", "true").lower() in ("1", "true", "yes"),
        "available": False,
        "queries": [],
        "results": 0,
        "inserted": 0,
    }
    if not stats["enabled"]:
        return [], stats

    from tools.search_sources import search_zai_web

    # HyDE variants return the ORIGINAL query as their first element and
    # wrap the rest in academic prose that web engines match poorly (live
    # run7: the "Technical analysis of..." phrasing returned zero grey-lit
    # hits while the plain query returned the TranspariMED reports). Budget:
    # primary + deterministic keyword rewrite (always distinct vocab) + at
    # most 2 genuinely-new variants (thorough).
    extra_variants = [
        v for v in (variants or [])
        if v and v.strip().lower() != (query or "").strip().lower()
    ]
    candidates = [query, _web_query_rewrite(query)]
    if depth == "thorough":
        candidates += extra_variants[:2]
    seen_q = set()
    queries = []
    for cand in candidates:
        q = (cand or "").strip()[:300]
        if q and q.lower() not in seen_q:
            seen_q.add(q.lower())
            queries.append(q)
    stats["queries"] = queries
    if not queries:
        return [], stats

    try:
        lists = await asyncio.gather(
            *[search_zai_web(q, 10) for q in queries],
            return_exceptions=True,
        )
    except Exception as e:
        logger.warning(f"Web search failed (non-fatal): {e}")
        return [], stats

    flat: List[Any] = []
    ok_queries = 0
    seen: set = set()
    for lst in lists:
        if isinstance(lst, Exception):
            logger.debug(f"Web search query failed: {lst}")
            continue
        if not lst:
            continue
        ok_queries += 1
        for r in lst:
            if getattr(r, "url", "") in seen:
                continue
            seen.add(r.url)
            flat.append(r)

    stats["results"] = len(flat)
    stats["available"] = ok_queries > 0

    # ── SEED FALLBACK (provider redundancy, honest labeling) ──
    # When live web search yields nothing (bridge down, upstream 429, egress
    # blocked), operator-attested VERIFIED grey-literature URLs from
    # APEX_WEB_SEED_URLS (JSON list of {url,title,snippet,published_date})
    # enter the SAME governance stack — canonicalization, domain-tier
    # enforcement, DROP filtering, junk + relevance gates — and are marked
    # seeded=True in both stats and the record's extra block, so the
    # provenance appendix can state that live search was unavailable and
    # seeds were used. A dead search engine never fabricates results; it
    # only substitutes attested URLs for discovered ones.
    if not flat:
        seeds = _load_web_seed_urls()
        for seed in seeds:
            try:
                from tools.search_sources import canonicalize_web_url, _web_result_tier
                url = canonicalize_web_url(str(seed.get("url") or ""))
                if not url or url in seen:
                    continue
                tier = _web_result_tier(url)
                if tier == "DROP":
                    continue
                seen.add(url)
                flat.append(SimpleNamespace(
                    url=url,
                    title=re.sub(r"\s+", " ", str(seed.get("title") or url)).strip()[:300],
                    snippet=str(seed.get("snippet") or "")[:600],
                    source_name="web_seed",
                    source_tier=tier,
                    published_date=seed.get("published_date") or "",
                    extra={"web": True, "seeded": True},
                ))
            except Exception as e:
                logger.debug(f"Seed URL {seed.get('url', '?')!r} rejected: {e}")
        stats["seeded"] = len(flat)
        stats["search_mode"] = "seeded_fallback" if flat else "unavailable"
    else:
        stats["search_mode"] = "live"

    # ── ClinicalTrials.gov registry lane (infrastructure the academic fan-out
    # lacks entirely — Elicit's differentiator for trial questions). One
    # keyless query against the official registry; records carry
    # has_results so the pool can distinguish never-reported trials.
    # Degradation is silent ([]), never fatal.
    try:
        from tools.search_sources import search_clinical_trials, canonicalize_web_url
        # Registry-natural query: trial records never describe themselves as
        # "unpublished" — searching the academic rewrite ("Meta-analysis
        # unpublished ... databases") matches nothing. Query the registry
        # with the question's DISCRIMINATIVE domain words only, then let the
        # hasResults=False annotation identify never-reported trials.
        reg_words = [
            w for w in re.findall(r"[a-z0-9]{4,}", (query or "").lower())
            if w not in GENERIC_ACADEMIC_TERMS and w not in _STOPWORDS
        ]
        reg_query = " ".join(dict.fromkeys(reg_words))[:200] or queries[0]
        reg_rows = await search_clinical_trials(reg_query, 8)
        reg_kept = 0
        for r in reg_rows:
            u = canonicalize_web_url(getattr(r, "url", ""))
            if not u or u in seen:
                continue
            seen.add(u)
            flat.append(SimpleNamespace(
                url=u,
                title=getattr(r, "title", "") or u,
                snippet=getattr(r, "snippet", "") or "",
                source_name="clinical_trials",
                source_tier=getattr(r, "source_tier", "") or "P1",
                published_date=getattr(r, "published_date", "") or "",
                extra=dict(getattr(r, "extra", {}) or {}, web=True, registry=True),
            ))
            reg_kept += 1
        stats["registry_records"] = reg_kept
        stats["registry_no_results"] = sum(
            1 for r in reg_rows if getattr(r, "extra", {}).get("has_results") is False
        )
    except Exception as e:
        logger.debug(f"ClinicalTrials.gov lane failed (non-fatal): {e}")

    out: List[Dict] = []
    for r in flat:
        out.append({
            "url": r.url,
            "title": r.title,
            "snippet": r.snippet,
            "tier": enforce_source_tier(r.url, r.source_tier),
            "source_name": r.source_name,
            "category": "web",
            "published_date": r.published_date,
            "extra": dict(getattr(r, "extra", {}) or {}),
        })
    stats["inserted"] = len(out)
    logger.info(
        f"Web search: {len(out)} grey-literature records from {ok_queries}/{len(queries)} queries"
    )
    return out, stats

RESEARCH_REPORT_SYSTEM_PROMPT_QUICK = """You are APEX Research, an expert research analyst producing comprehensive, evidence-based reports.

You MUST produce a DETAILED report with MULTIPLE TABLES. Write 800-1500 words minimum.
Every section must have substantial prose — not just bullet points.

OUTPUT FORMAT — Produce EXACTLY this structure:

## EXECUTIVE SUMMARY
(3-5 sentences: key findings, confidence level, and why it matters)

## KEY FINDINGS
(For each major finding, write 2-3 sentences of explanation BEFORE the table)

| Finding | Evidence | Source | Tier | Status |
|---------|----------|--------|------|--------|
| (detailed finding with context) | (specific evidence) | [N] | P1/P2/P3 | ESTABLISHED/TENTATIVE/ACTIVE_DEBATE/SPECULATIVE |

## COMPARATIVE ANALYSIS
(Create a comparison table — compare approaches, methods, tools, frameworks, or perspectives)

| Dimension | Option A | Option B | Option C |
|-----------|----------|----------|----------|
| (criteria) | (details) | (details) | (details) |

(Write 2-3 paragraphs analyzing the trade-offs shown in the table)

## ACTIVE DEBATES
(If sources conflict, present BOTH sides with citations. Write 2-3 sentences per debate point)

## SPECULATIVE FINDINGS
(Early inferences that need replication. Explain WHY they are speculative and what evidence would confirm them)

## METHODOLOGY NOTES
(Discuss: sample sizes, study types, limitations of the evidence, potential biases in sources)

## PRACTICAL IMPLICATIONS
(What should the reader DO with this information? 2-3 actionable recommendations)

## SOURCES
[1] Author. Title. Journal/Domain, Year. URL

RULES:
1. EVERY claim must have a source citation and epistemic status
2. Use [ESTABLISHED] for claims with 2+ independent P1/P2 sources
3. Use [TENTATIVE] for claims with only 1 source or only P3 sources
4. Use [ACTIVE_DEBATE] when sources conflict
5. Use [SPECULATIVE] for early-stage findings
6. Use [UNVERIFIED] if no source supports a claim
7. Use MULTIPLE TABLES — at minimum: Key Findings table + Comparative Analysis table
8. Write SUBSTANTIAL PROSE between tables — explain the significance, context, and implications
9. No preamble, no filler, no "In conclusion"
10. Be COMPREHENSIVE — this is a research report, not a summary
"""

RESEARCH_REPORT_SYSTEM_PROMPT_THOROUGH = """You are APEX Research, a world-class research analyst producing deep, comprehensive, publication-quality research reports.

You MUST produce an EXTREMELY DETAILED report with 3-5+ TABLES and 2000-4000+ words.
Every section must have MULTIPLE PARAGRAPHS of analysis and explanation.
Think like an academic researcher writing for an informed audience.

OUTPUT FORMAT — Produce EXACTLY this structure:

## EXECUTIVE SUMMARY
(5-8 sentences: key findings, overall confidence assessment, critical gaps, and why this topic matters)

## KEY FINDINGS
(For each finding, write 3-5 sentences of detailed explanation with nuance and context)

| Finding | Evidence | Source | Tier | Status |
|---------|----------|--------|------|--------|
| (detailed, nuanced finding) | (specific evidence with detail) | [N] | P1/P2/P3 | ESTABLISHED/TENTATIVE/ACTIVE_DEBATE/SPECULATIVE |

(Include 8-15 findings covering ALL aspects of the query)

## COMPARATIVE ANALYSIS
(Create detailed comparison tables for relevant dimensions — approaches, tools, frameworks, methods, schools of thought)

| Dimension | Option A | Option B | Option C | Option D |
|-----------|----------|----------|----------|----------|
| (criteria 1) | (details) | (details) | (details) | (details) |
| (criteria 2) | (details) | (details) | (details) | (details) |

(Write 3-5 paragraphs of in-depth comparative analysis discussing trade-offs, use cases, and when each approach is optimal)

## EVIDENCE QUALITY MATRIX
(Assess the strength and reliability of the evidence base)

| Evidence Type | Count | Avg Confidence | Key Limitation |
|---------------|-------|----------------|----------------|
| (RCT/peer-reviewed/observation/etc.) | N | High/Med/Low | (specific limitation) |

(Write 2-3 paragraphs discussing evidence gaps and reliability concerns)

## ACTIVE DEBATES
(For EACH debate point: present BOTH sides with citations, explain the disagreement, assess which side has stronger evidence. Write 4-6 sentences per debate)

| Debate Point | Position A | Position B | Preponderance of Evidence |
|-------------|-----------|-----------|------------------------|
| (topic) | (view + sources) | (view + sources) | (A/B/Unclear + why) |

## SPECULATIVE FINDINGS
(Explain each speculative finding in detail — WHY it's speculative, what evidence would confirm/falsify it, potential impact if confirmed)

## METHODOLOGY NOTES
(Comprehensive discussion: sample sizes, study designs, potential confounds, publication bias, funding sources, geographic limitations, temporal relevance)

## PRACTICAL IMPLICATIONS & RECOMMENDATIONS
(5-8 specific, actionable recommendations with confidence levels. Prioritize by impact and evidence strength)

| Recommendation | Confidence | Key Prerequisite | Potential Risk |
|---------------|-----------|-----------------|---------------|
| (specific action) | High/Med/Low | (what's needed) | (what could go wrong) |

## FUTURE RESEARCH DIRECTIONS
(What questions remain unanswered? What studies would be most valuable? 3-5 specific research questions)

## SOURCES
[1] Author. Title. Journal/Domain, Year. URL

RULES:
1. EVERY claim must have a source citation and epistemic status
2. Use [ESTABLISHED] for claims with 2+ independent P1/P2 sources
3. Use [TENTATIVE] for claims with only 1 source or only P3 sources
4. Use [ACTIVE_DEBATE] when sources conflict
5. Use [SPECULATIVE] for early-stage findings
6. Use [UNVERIFIED] if no source supports a claim
7. Use MULTIPLE TABLES — minimum 3: Key Findings, Comparative Analysis, Evidence Quality or Recommendations
8. Write SUBSTANTIAL PROSE between and after tables — explain significance, context, nuance, and implications
9. No preamble, no filler, no "In conclusion"
10. Be EXTREMELY COMPREHENSIVE — this is a deep research report, not a surface summary
11. Cover the topic from MULTIPLE ANGLES — technical, practical, theoretical, comparative
12. When comparing options, use detailed multi-row tables with 3+ options
"""


@dataclass
class ResearchReport:
    """A structured research report with verified claims."""
    query: str
    title: str = ""  # evidence-accurate report title (may reframe the query)
    executive_summary: str = ""
    findings: List[Dict] = field(default_factory=list)
    debates: List[Dict] = field(default_factory=list)
    speculative: List[str] = field(default_factory=list)
    sources: List[Dict] = field(default_factory=list)
    verification: Optional[VerificationResult] = None
    raw_report: str = ""
    total_latency_ms: int = 0
    depth: str = "quick"  # "quick" or "thorough"
    # APEX 2.0 Wiki fields
    wiki_page_id: Optional[str] = None
    wiki_cache_id: Optional[str] = None
    wiki_lifecycle: str = "draft"
    wiki_slug: str = ""
    wiki_version: int = 0
    from_cache: bool = False
    # ── APEX 3.0 report-quality fields ──
    evidence: Dict = field(default_factory=dict)          # full-text evidence stats
    citation_metrics: Dict = field(default_factory=dict)  # citation integrity metrics
    expanded_queries: List[str] = field(default_factory=list)
    generation_mode: str = "sectionwise"                  # "sectionwise" | "single_call"
    model_used: str = ""
    provenance: Dict = field(default_factory=dict)        # PRISMA-style funnel audit trail


# ═══════════════════════════════════════════════════════════════
# APEX 3.0: SECTION-WISE REPORT GENERATION
# One call per section beats one mega-call: no truncation, targeted
# evidence per section, parallel drafting, measurable citation integrity.
# ═══════════════════════════════════════════════════════════════

SECTION_DRAFT_SYSTEM_PROMPT = """You are APEX Research, an expert research analyst drafting ONE section of a larger evidence-based report.

CITATION FORMAT (NON-NEGOTIABLE):
- Cite sources as [S1], [S2], ... matching the numbered source list provided.
- ONLY cite sources that appear in the provided list. Never invent [S15] if 10 sources are given.
- Every factual claim (numbers, findings, comparisons) MUST end with a citation.
- If a claim has no supporting source, either omit it or mark it [UNVERIFIED].
- When sources conflict, present BOTH sides with their citations.

STYLE:
- Substantial prose (not bare bullets) with tables where comparisons arise.
- No preamble, no "In conclusion", no filler transitions.
- Write ONLY the requested section — no other sections, no report title.
- Use ## for the section heading exactly as given.

CLAIM-SOURCE FIDELITY (NON-NEGOTIABLE):
- Before attaching [Sn] to a sentence, verify that source actually supports THAT claim — not merely a related topic.
- If the source addresses a related-but-different problem (e.g. accumulation bias ≠ publication bias; workflow automation ≠ making inaccessible data accessible), NEVER present it as solving the claim's problem. Restate the claim to what the source actually supports, or drop the citation and mark the sentence [UNVERIFIED].
- Encyclopedias may support background definitions only — never named methods, statistical procedures, or empirical results.
- SCOPE FIDELITY: if a source's title restricts its conclusion to a domain or population (e.g. "...in educational research", "...in mice"), NEVER state that conclusion unqualified — carry the restriction into the sentence or drop the generalization.
- News/consumer web sources are not peer-reviewed evidence: cite them only for the documented real-world events they report (investigations, audits, policy actions), never for scientific findings that a peer-reviewed source in the pool also covers.
- Describe each source's actual scope; do not inflate its contribution to fit your narrative."""


CRITIQUE_SYSTEM_PROMPT = """You are a hostile peer reviewer for a research report.

Given the report and its numbered source list, identify ONLY these defect types:
1. HALLUCINATED_CITATION: a [Sn] citation whose claim is NOT actually supported by that source
2. MISSING_CITATION: a factual claim sentence with no citation
3. UNSUPPORTED_SUPERLATIVE: sweeping claims ("most", "best", "proves") without evidence
4. STALE_EVIDENCE: claims resting on clearly outdated sources when newer ones exist
5. FRAMING_MISMATCH: the title or section structure promises an analysis the cited evidence cannot deliver (e.g. claims to study X directly when the evidence only supports an indirect estimate related to X) — the fix is a reframing instruction, never invented evidence
6. CLAIM_SOURCE_MISMATCH: a citation attached to a claim about a DIFFERENT problem than the source addresses — e.g. a source about accumulation bias (type-I error from repeatedly updating a meta-analysis) cited as if it addressed publication bias (unpublished studies); a workflow-automation tool cited as making inaccessible data accessible. The fix MUST restate the claim to what the source actually supports, or drop the claim
7. ENCYCLOPEDIA_CITATION: a named method, statistical procedure, or empirical result cited to an encyclopedia — fix: keep encyclopedias for background definitions only; attribute methods/results to primary literature or mark [UNVERIFIED]
8. SCOPE_DODGED: the SCOPE section concedes the query's literal demand cannot be met but offers only a generic hedge — no STRUCTURAL reason (why the demanded data/analysis fundamentally cannot exist, not merely that it was not found) and no presentation of documented real-world attempts/audits that exist in the source list. Fix: state the structural reason and present those attempts with citations
9. DOMAIN_GENERALIZATION: a source's conclusion is stated without the domain/population scope its title restricts it to (e.g. a finding demonstrated in educational research generalized to all research). Fix: restrict the claim to the source's scope or attribute the scope explicitly
10. NEWS_TIER_CITATION: a scientific claim cited to a news/consumer web source when a peer-reviewed source in the list supports the same point. Fix: re-anchor the citation to the peer-reviewed source; keep the news source only for the documented event it reports

Pay special attention to MECHANICALLY FLAGGED PAIRS if present — they are deterministic suspects for type 1/6/9/10.

Return ONLY valid JSON:
{"issues": [{"section_heading": "...", "type": "...", "detail": "...", "fix": "specific instruction"}]}
If the report is clean, return {"issues": []}. Max 6 issues — the worst ones only."""


DEFAULT_QUICK_SECTIONS = [
    {"id": "exec_summary", "title": "EXECUTIVE SUMMARY", "purpose": "Key findings, confidence level, why it matters", "key_questions": ["What are the most important established findings?", "What is the overall confidence?"]},
    {"id": "key_findings", "title": "KEY FINDINGS", "purpose": "Detailed findings with evidence and epistemic status", "key_questions": ["What did the strongest sources find?", "Which findings are established vs tentative?"]},
    {"id": "analysis", "title": "ANALYSIS & LIMITATIONS", "purpose": "Trade-offs, evidence quality, methodology caveats", "key_questions": ["What are the limitations of the evidence?", "What practical implications follow?"]},
]

DEFAULT_THOROUGH_SECTIONS = [
    {"id": "scope", "title": "SCOPE & EVIDENCE BASE", "purpose": "State what the evidence base actually supports: what was retrievable, what could NOT be analyzed directly, and any framing limits — BEFORE any findings. If the query's literal demand is not satisfiable, state the STRUCTURAL reason why the demanded data/analysis fundamentally cannot exist (not merely that it was not found) and present any documented real-world attempts/audits in the pool that demonstrate the limitation", "key_questions": ["What evidence was actually retrieved?", "What could not be analyzed directly and why — structurally, not just in this search?", "Does the evidence support the query's literal demand?", "What do documented real-world attempts (registries, audits, NGO investigations) in the source list show about that limitation?"]},
    {"id": "exec_summary", "title": "EXECUTIVE SUMMARY", "purpose": "Key findings, overall confidence, critical gaps, why it matters", "key_questions": ["What are the most important established findings?", "What remains uncertain or contested?", "Why does this topic matter now?"]},
    {"id": "key_findings", "title": "KEY FINDINGS", "purpose": "8-15 detailed findings with evidence table and epistemic status", "key_questions": ["What did the strongest P1/P2 sources find?", "Which findings are ESTABLISHED vs TENTATIVE?", "What numbers/effect sizes are reported?"]},
    {"id": "comparative", "title": "COMPARATIVE ANALYSIS", "purpose": "Multi-dimension comparison table of approaches/methods/options with trade-off analysis", "key_questions": ["What are the main approaches/options?", "How do they differ across key dimensions?", "When is each optimal?"]},
    {"id": "evidence_quality", "title": "EVIDENCE QUALITY MATRIX", "purpose": "Table of evidence types, counts, confidence, key limitations — ONLY evidence actually retrieved; never list the query's core subject as an evidence row unless it was directly studied", "key_questions": ["What study types back the findings?", "Where is the evidence weak or missing?"]},
    {"id": "debates", "title": "ACTIVE DEBATES", "purpose": "Points where sources conflict — both sides with citations", "key_questions": ["Where do sources disagree?", "Which side has stronger evidence?"]},
    {"id": "speculative", "title": "SPECULATIVE FINDINGS", "purpose": "Inferences extrapolated ONLY from cited evidence, needing replication, with confirmation criteria — omit the section entirely if the evidence base supports no responsible extrapolation", "key_questions": ["What do the CITED sources jointly imply that none states alone?", "What evidence would confirm or falsify it?"]},
    {"id": "methodology", "title": "METHODOLOGY NOTES", "purpose": "Sample sizes, study designs, biases, geographic/temporal limits", "key_questions": ["What are the methodology caveats?", "What biases might skew the evidence?"]},
    {"id": "implications", "title": "PRACTICAL IMPLICATIONS & RECOMMENDATIONS", "purpose": "Actionable recommendations with confidence and risks, plus future research directions", "key_questions": ["What should the reader do?", "What questions remain open?"]},
]


async def expand_query_variants(query: str, max_variants: int = 3) -> List[str]:
    """
    HyDE-style query expansion: generate paraphrase/perspective variants so
    the parallel research pass retrieves beyond the exact query wording.

    Falls back to [query] on any failure — expansion is an enhancement,
    never a blocker.
    """
    from agent.llm_router import synthesize_with_router

    prompt = f"""Generate {max_variants - 1} alternative research queries for the same information need, each emphasizing a different angle (technical, empirical, practical/state-of-the-art).

Original query: {query}

Return ONLY numbered queries, one per line, nothing else."""

    try:
        result = await synthesize_with_router(
            query="expand research query",
            context=prompt,
            max_tokens=120,
            system_prompt="You rewrite research queries for better retrieval coverage. Return only numbered items.",
            task_type="classification",  # tiny output, cheap models fine
        )
        variants = [query]
        for line in result.content.split("\n"):
            m = re.match(r'^\d+[\.\)]\s*(.+)', line.strip())
            if m and len(m.group(1).strip()) > 10:
                variant = m.group(1).strip()
                # LLMs often prefix variants with angle labels like
                # "Technical:", "Empirical:", "Practical:" — strip them, since
                # raw colons break downstream source APIs (e.g. ClinicalTrials.gov).
                variant = re.sub(
                    r'^(technical|empirical|practical|state[- ]of[- ]the[- ]art|'
                    r'methodological|theoretical|applied|perspective)\s*[:\-\u2013]\s*',
                    '', variant, flags=re.IGNORECASE,
                ).strip()
                if len(variant) > 10:
                    variants.append(variant)
        return variants[:max_variants]
    except Exception as e:
        logger.debug(f"Query expansion failed (non-fatal): {e}")
        return [query]


async def _rag_sources_for_report(query: str, max_context_tokens: int = 8000) -> List[Dict]:
    """
    Hybrid report mode: pull RAG corpus results alongside live search.
    Reports want max recall — never depend on the RAG-vs-live binary decision.

    Converts RetrievedChunks to source dicts and returns [] on any failure
    (DB-less mode must keep working).
    """
    try:
        from agent.retriever import retrieve
        chunks, avg_sim = await retrieve(
            query,
            top_k=12,
            final_k=8,
            max_context_tokens=max_context_tokens,
        )
        rag_sources = []
        for c in chunks:
            rag_sources.append({
                "url": c.source_url,
                "title": c.title or "",
                "snippet": c.raw_text,
                "tier": c.source_tier,
                "domain": c.domain,
                "authors": c.authors or [],
                "published_date": (c.metadata or {}).get("published_date", ""),
                "doi": (c.metadata or {}).get("doi") or (c.metadata or {}).get("arxiv_id", ""),
                "score": c.similarity_score,
                "origin": "rag",
            })
        logger.info(f"Hybrid RAG merge: {len(rag_sources)} corpus chunks added to report sources")
        return rag_sources
    except Exception as e:
        logger.debug(f"RAG retrieval for report failed (non-fatal): {e}")
        return []


async def _snowball_sources(sources: List[Dict], max_seeds: int = 2, per_seed: int = 4) -> List[Dict]:
    """
    Citation snowballing: expand the top Semantic Scholar hits via their
    forward citations (newer work building on them). Pure addition — the
    caller re-dedups, so overlap with existing sources is harmless.
    """
    seeds = []
    for s in sources:
        paper_id = (s.get("extra", {}) or {}).get("paperId") or ""
        if not paper_id:
            m = re.search(r'semanticscholar\.org/paper/([A-Za-z0-9]{20,})', s.get("url", ""))
            if m:
                paper_id = m.group(1)
        if paper_id:
            seeds.append(paper_id)
        if len(seeds) >= max_seeds:
            break
    if not seeds:
        return []

    try:
        from tools.search_sources import snowball_semantic_scholar
        tasks = [snowball_semantic_scholar(pid, direction="citations", max_results=per_seed) for pid in seeds]
        batches = await asyncio.gather(*tasks, return_exceptions=True)
        snowballed = []
        for batch in batches:
            if isinstance(batch, Exception):
                continue
            for r in batch:
                snowballed.append({
                    "url": r.url,
                    "title": r.title,
                    "snippet": r.snippet,
                    "tier": r.source_tier,
                    "source_name": r.source_name,
                    "published_date": r.published_date,
                    "authors": r.authors,
                    "citation_count": r.citation_count,
                    "doi": r.doi,
                    "extra": r.extra,
                    "origin": "snowball",
                })
        if snowballed:
            logger.info(f"Snowballing added {len(snowballed)} candidate papers")
        return snowballed
    except Exception as e:
        logger.debug(f"Snowballing failed (non-fatal): {e}")
        return []


async def _generate_outline(
    query: str,
    sources: List[Dict],
    verification: Optional[VerificationResult],
    depth: str,
) -> tuple:
    """
    Outline-first planning: an LLM proposes the report's sections given the
    evidence inventory (claims with epistemic status + source titles).

    Also produces an EVIDENCE-ACCURATE title: if the retrievable evidence only
    supports an indirect analysis of the query, the title must say so instead
    of parroting the query (prevents the framing bait-and-switch).

    Returns (sections, report_title). Falls back to the built-in default
    section plan and query-as-title on any failure.
    """
    from agent.llm_router import synthesize_with_router

    defaults = DEFAULT_THOROUGH_SECTIONS if depth == "thorough" else DEFAULT_QUICK_SECTIONS

    claims_blob = ""
    if verification and verification.claims:
        lines = []
        for c in verification.claims[:12]:
            lines.append(f"- [{c.epistemic_status}] {c.statement[:140]}")
        claims_blob = "\n".join(lines)

    sources_blob = "\n".join(
        f"[S{i}] [{s.get('tier', 'UNV')}] {(s.get('title') or 'Untitled')[:90]}"
        for i, s in enumerate(sources[:20], 1)
    )

    prompt = f"""Plan the section structure for a research report on:
QUERY: {query}

AVAILABLE SOURCES:
{sources_blob}

VERIFIED CLAIMS SO FAR:
{claims_blob or "(none extracted yet)"}

Choose 4-{len(DEFAULT_THOROUGH_SECTIONS)} sections. You may adapt the standard structure
(Executive Summary, Key Findings, Comparative Analysis, Evidence Quality, Active Debates, Speculative Findings, Methodology Notes, Practical Implications) to fit the evidence — drop sections with no evidence, add a section if the evidence demands it.

FRAMING HONESTY (MANDATORY):
1. report_title: an HONEST title describing what the evidence base actually supports — NOT the query verbatim. If the sources only permit an INDIRECT analysis of the query's literal subject, the title must reflect that (e.g. "What publication-bias research reveals about the scale of the unpublished-trial problem").
2. The FIRST section MUST be titled "SCOPE & EVIDENCE BASE" and must state: what evidence was actually retrievable, what could NOT be analyzed directly, and whether the query's literal demand is methodologically feasible with this evidence base. If it is not feasible, the section MUST give the structural reason (why the demanded data/analysis fundamentally cannot exist — not merely that it was not found) and point to the closest documented real-world attempts in the source list.
3. Never plan a section that implies direct evidence for the query's core subject unless such sources exist above.

Return ONLY valid JSON:
{{"report_title": "...", "sections": [{{"id": "kebab-id", "title": "SECTION TITLE", "purpose": "1 sentence", "key_questions": ["q1", "q2"]}}]}}"""

    try:
        result = await synthesize_with_router(
            query="plan report outline",
            context=prompt,
            max_tokens=900,
            system_prompt="You plan evidence-based research reports. Return ONLY valid JSON.",
            task_type="outline",
        )
        content = result.content.strip()
        if content.startswith("```"):
            content = re.sub(r"^```(?:json)?\s*", "", content)
            content = re.sub(r"\s*```$", "", content)
        m = re.search(r'\{[\s\S]*\}', content)
        if m:
            data = json.loads(m.group())
            sections = data.get("sections", [])
            report_title = str(data.get("report_title", "")).strip()[:200]
            # Validate: need id/title/purpose; cap at 9 sections
            valid = []
            for s in sections[:9]:
                if isinstance(s, dict) and s.get("title"):
                    valid.append({
                        "id": str(s.get("id", s["title"].lower().replace(" ", "-")))[:40],
                        "title": str(s["title"]).upper()[:80],
                        "purpose": str(s.get("purpose", ""))[:300],
                        "key_questions": [str(q)[:200] for q in (s.get("key_questions") or [])][:4],
                    })
            if len(valid) >= 3:
                logger.info(f"Outline generated: {len(valid)} sections via {result.model_name}")
                return valid, (report_title or query)
    except Exception as e:
        logger.debug(f"Outline generation failed, using default structure: {e}")

    return [dict(s) for s in defaults], query


def _claims_for_section(section: Dict, verification: Optional[VerificationResult], max_claims: int = 8) -> str:
    """Pick verified claims relevant to a section (keyword overlap) for its context."""
    if not verification or not verification.claims:
        return ""
    section_text = f"{section.get('title', '')} {section.get('purpose', '')} {' '.join(section.get('key_questions', []))}".lower()
    sec_terms = set(re.findall(r'\b\w{4,}\b', section_text))

    scored = []
    for c in verification.claims:
        claim_terms = set(re.findall(r'\b\w{4,}\b', c.statement.lower()))
        overlap = len(sec_terms & claim_terms)
        if overlap >= 2:
            scored.append((overlap, c))
    scored.sort(key=lambda t: t[0], reverse=True)

    lines = []
    for _, c in scored[:max_claims]:
        conflicts = f" ({len(c.conflicting_sources)} sources CONFLICT)" if c.conflicting_sources else ""
        lines.append(f"- [{c.epistemic_status}] (conf={c.confidence}, {c.evidence_type or 'n/a'}) {c.statement[:250]}{conflicts}")
    return "\n".join(lines)


async def _draft_section(
    section: Dict,
    query: str,
    sources: List[Dict],
    corpus,   # EvidenceCorpus
    verification: Optional[VerificationResult],
    depth: str,
    fulltext_count: int = 0,
) -> tuple:
    """
    Draft ONE report section with targeted evidence retrieval.

    Returns (section_id, markdown, model_used). Never raises — on total
    failure returns a minimal placeholder so assembly always produces a
    complete report skeleton.
    """
    from agent.llm_router import synthesize_with_router
    from agent.evidence_fetcher import retrieve_evidence, format_evidence_for_context

    section_query = f"{section.get('title', '')}. {section.get('purpose', '')}. " + " ".join(section.get("key_questions", []))

    # Targeted evidence for THIS section
    evidence_text = ""
    try:
        if corpus and len(corpus) > 0:
            ev_chunks = await retrieve_evidence(corpus, section_query, k=6)
            evidence_text = format_evidence_for_context(ev_chunks)
    except Exception as e:
        logger.debug(f"Evidence retrieval failed for section {section.get('id')}: {e}")

    # Source inventory (numbered, so the model can cite [S1]..[Sn])
    sources_blob = "\n".join(
        f"[S{i}] [{s.get('tier', 'UNV')}] {(s.get('title') or 'Untitled')[:100]}: {(s.get('snippet') or '')[:400]}"
        for i, s in enumerate(sources, 1)
    )

    claims_blob = _claims_for_section(section, verification)

    word_target = "250-500" if depth == "quick" else "400-800"
    depth_note = ""
    if fulltext_count < 3:
        depth_note = (
            f"\nEVIDENCE-DEPTH CONSTRAINT (NON-NEGOTIABLE): only {fulltext_count} source(s) in the pool have "
            f"fetched FULL TEXT (floor: 3). Everything else is title/abstract-level. You MUST NOT describe "
            f"any evidence as 'established', 'well-established', 'definitive', or 'conclusive' — qualify "
            f"claims as abstract-level wherever full text was not fetched.\n"
        )
    user_prompt = f"""REPORT QUERY: {query}
{depth_note}
YOUR SECTION: {section.get('title', '')}
PURPOSE: {section.get('purpose', '')}
KEY QUESTIONS TO ANSWER:
""" + "\n".join(f"- {q}" for q in section.get("key_questions", [])) + f"""

FULL-TIME EVIDENCE (retrieved passages with [S{{n}}|TIER] attribution — prefer these over snippets):
{evidence_text or "(no full-text evidence available for this section — rely on the source snippets below)"}

VERIFIED CLAIMS (use these epistemic labels verbatim):
{claims_blob or "(none)"}

NUMBERED SOURCE INVENTORY:
{sources_blob}

Write the section now. Target {word_target} words. Head it with: ## {section.get('title', '')}
Cite with [S{{n}}] tags ONLY from the inventory above."""

    try:
        result = await synthesize_with_router(
            query=section_query[:200],
            context=user_prompt,
            max_tokens=2048 if depth == "quick" else 3072,
            system_prompt=SECTION_DRAFT_SYSTEM_PROMPT,
            task_type="report",
        )
        content = result.content.strip()
        if content.startswith("[ALL_LLM_FAILED]") or len(content) < 80:
            return section.get("id", "section"), "", result.model_name
        return section.get("id", "section"), content, result.model_name
    except Exception as e:
        logger.warning(f"Section drafting failed for {section.get('id')}: {e}")
        return section.get("id", "section"), "", ""


def _assemble_report(section_markdowns: List[str], sources: List[Dict], title: str = "") -> str:
    """
    Assemble section drafts into a full report.
    The SOURCES bibliography is built PROGRAMMATICALLY (deterministic —
    the LLM never gets to invent the source list).

    The H1 title is the outline's evidence-accurate title (never the raw
    query when the evidence supports only an indirect analysis).
    """
    from agent.citation_integrity import build_sources_section

    body = "\n\n".join(md.strip() for md in section_markdowns if md and md.strip())
    header = f"# {title}\n\n" if title else ""
    return f"{header}{body}\n\n{build_sources_section(sources)}"


async def _critique_and_revise(
    report_text: str,
    query: str,
    sources: List[Dict],
    depth: str,
    extra_issues: Optional[List[str]] = None,
) -> tuple:
    """
    Critique pass: mechanical citation-integrity check + canonical-method
    scan + hostile-reviewer LLM pass. If real issues surface, ONE revision
    call regenerates the report with the issue list as feedback.

    extra_issues: pre-formatted issue strings from the late canonical
    backfill (method named in draft, primary source fetched and appended
    after drafting) — they enter the revision feedback verbatim.

    Returns (final_text, critique_summary_dict, model_used).
    """
    from agent.llm_router import synthesize_with_router
    from agent.citation_integrity import (
        check_report_citations, scan_claim_source_alignment,
        scan_concept_boundary_mismatch, scan_domain_generalization,
        scan_news_tier_citations, scan_scope_dodge,
        scan_off_topic_padding,
    )
    from agent.canonical_sources import scan_canonical_method_usage

    citation_report = check_report_citations(report_text, sources)
    alignment_flags = scan_claim_source_alignment(report_text, sources)
    boundary_flags = scan_concept_boundary_mismatch(report_text, sources)
    domain_flags = scan_domain_generalization(report_text, sources)
    news_flags = scan_news_tier_citations(report_text, sources)
    scope_flags = scan_scope_dodge(report_text)
    padding_flags = scan_off_topic_padding(query, sources)
    canonical_issues = scan_canonical_method_usage(report_text, sources)
    critique_summary = {
        "citation_precision": citation_report.citation_precision,
        "claim_coverage": citation_report.claim_coverage,
        "invalid_citations": citation_report.invalid_citations,
        "uncited_claims": len(report_text and citation_report.uncited_claim_sentences or []),
        "alignment_flags": len(alignment_flags),
        "boundary_flags": len(boundary_flags),
        "domain_flags": len(domain_flags),
        "news_flags": len(news_flags),
        "scope_flags": len(scope_flags),
        "padding_flags": len(padding_flags),
        "canonical_issues": len(canonical_issues),
        "llm_issues": 0,
        "revised": False,
    }

    # LLM critique (skip when the mechanical pass already failed badly —
    # the revision prompt will carry both mechanical and LLM issues)
    llm_issues = []
    try:
        sources_blob = "\n".join(
            f"[S{i}] {(s.get('title') or 'Untitled')[:100]}: {(s.get('snippet') or '')[:300]}"
            for i, s in enumerate(sources, 1)
        )
        alignment_blob = ""
        mechanical_flags = list(alignment_flags) + [
            {"source": f["source"], "excerpt": f["excerpt"], "reason": f.get("reason", "")}
            for f in boundary_flags
        ] + [
            {"source": f["source"], "excerpt": f["excerpt"], "reason": f.get("reason", "")}
            for f in domain_flags
        ] + [
            {"source": f["source"], "excerpt": f["excerpt"], "reason": f.get("reason", "")}
            for f in news_flags
        ] + [
            {"source": f["source"], "excerpt": f["title"], "reason": f.get("reason", "")}
            for f in padding_flags
        ]
        if mechanical_flags:
            pairs = "\n".join(
                f"  - [S{f['source']}] ← “{f['excerpt']}”"
                + (f"\n        (mechanical hypothesis: {f['reason']})" if f.get("reason") else "")
                for f in mechanical_flags
            )
            alignment_blob = (
                "\nMECHANICALLY FLAGGED PAIRS (cited source may not support the "
                "claim — verify against the source list first; a hypothesis is "
                "attached where the mismatch is conceptual, not just lexical):\n" + pairs + "\n"
            )
        critique_prompt = f"""REPORT TO REVIEW (may be truncated):
{report_text[:14000]}

SOURCE LIST:
{sources_blob}
{alignment_blob}
Find the worst hallucinated citations, missing citations, and unsupported superlatives. Return ONLY the JSON."""
        result = await synthesize_with_router(
            query="critique report",
            context=critique_prompt,
            max_tokens=700,
            system_prompt=CRITIQUE_SYSTEM_PROMPT,
            task_type="nli",
        )
        content = result.content.strip()
        if content.startswith("```"):
            content = re.sub(r"^```(?:json)?\s*", "", content)
            content = re.sub(r"\s*```$", "", content)
        m = re.search(r'\{[\s\S]*\}', content)
        if m:
            data = json.loads(m.group())
            llm_issues = [i for i in data.get("issues", []) if isinstance(i, dict) and i.get("fix")][:6]
    except Exception as e:
        logger.debug(f"LLM critique failed (non-fatal): {e}")

    critique_summary["llm_issues"] = len(llm_issues)
    canonical_issue_strs = [
        f"[{i['type']}] {i['detail']} → {i['fix']}" for i in canonical_issues
    ]
    # Scope-dodge flags carry no source number — format without the [S{n}] head
    scope_issue_strs = [
        f"[SCOPE_DODGED] {f['excerpt']} — {f['reason']}" for f in scope_flags
    ]
    domain_issue_strs = [
        f"[DOMAIN_GENERALIZATION] [S{f['source']}] “{f['excerpt']}” — {f['reason']}"
        for f in domain_flags
    ]
    news_issue_strs = [
        f"[NEWS_TIER_CITATION] [S{f['source']}] “{f['excerpt']}” — {f['reason']}"
        for f in news_flags
    ]
    padding_issue_strs = [
        f"[OFF_TOPIC_PADDING] [S{f['source']}] {f['title']} — {f['reason']}"
        for f in padding_flags
    ]
    all_issues = (
        list(extra_issues or [])
        + list(citation_report.issues)
        + canonical_issue_strs
        + scope_issue_strs
        + domain_issue_strs
        + news_issue_strs
        + padding_issue_strs
        + [f"[{i.get('type')}] {i.get('detail')} → {i.get('fix')}" for i in llm_issues]
    )

    if not all_issues:
        return report_text, critique_summary, ""

    # One revision pass with all issues as feedback
    issues_blob = "\n".join(f"{n+1}. {issue}" for n, issue in enumerate(all_issues[:10]))
    sources_blob = "\n".join(
        f"[S{i}] [{s.get('tier', 'UNV')}] {(s.get('title') or 'Untitled')[:100]}: {(s.get('snippet') or '')[:350]}"
        for i, s in enumerate(sources, 1)
    )
    def _revise_prompt(rpt_text: str) -> str:
        return f"""Fix the defects flagged by review in this research report. Change ONLY what the issues require — keep everything else verbatim.

QUERY: {query}

FLAGGED ISSUES:
{issues_blob}

NUMBERED SOURCE INVENTORY (only [S1]..[S{len(sources)}] exist):
{sources_blob}

REPORT:
{rpt_text[:14000]}

Return the FULL corrected report (same structure and LENGTH, ## sections, [Sn] citations, SOURCES section at the end). Do not shorten the report."""

    async def _attempt_revision(rpt_text: str) -> tuple:
        result = await synthesize_with_router(
            query="revise report",
            context=_revise_prompt(rpt_text),
            max_tokens=8192,
            system_prompt=SECTION_DRAFT_SYSTEM_PROMPT,
            task_type="report",
        )
        content = result.content.strip()
        ok = not content.startswith("[ALL_LLM_FAILED]") and len(content) > len(rpt_text) * 0.5
        return content, ok, result.model_name

    for attempt, (rpt_text, label) in enumerate(
        [(report_text, "full"), (report_text[:10000], "trimmed")], start=1
    ):
        try:
            content, ok, model_name = await _attempt_revision(rpt_text)
            if ok:
                critique_summary["revised"] = True
                critique_summary["revision_attempts"] = attempt
                # Re-check citations after revision
                post = check_report_citations(content, sources)
                critique_summary["citation_precision"] = post.citation_precision
                critique_summary["claim_coverage"] = post.claim_coverage
                return content, critique_summary, model_name
            logger.info(
                f"Revision attempt {attempt} ({label}) rejected: "
                f"len={len(content)} vs threshold={int(len(rpt_text) * 0.5)}"
            )
        except Exception as e:
            logger.info(f"Revision attempt {attempt} ({label}) failed: {e}")

    return report_text, critique_summary, ""


def _render_provenance_section(provenance: Dict) -> str:
    """Deterministic PRISMA-style audit-trail appendix. No LLM involved —
    this is the reproducibility record of how the report was built."""
    ev = provenance.get("evidence", {}) or {}
    lines = [
        "",
        "## METHODOLOGY & PROVENANCE",
        "",
        "Reproducible audit trail of the retrieval pipeline that produced this "
        "report (PRISMA-style flow). Generated deterministically from pipeline "
        "counters — not drafted by a model.",
        "",
        "| Pipeline stage | Count |",
        "|---|---|",
        f"| Records identified (live APIs + RAG + web) | {provenance.get('identified_records', 0)} |",
        f"| After canonical dedup | {provenance.get('post_dedup', 0)} |",
        f"| After relevance/diversity selection | {provenance.get('post_selection', 0)} |",
        f"| After snowball + gap-fill enrichment | {provenance.get('post_enrichment', 0)} |",
        f"| Canonical primary sources backfilled (by verified DOI) | {provenance.get('canonical_backfilled', 0)} |",
        f"| Final cited pool (after dead-source pruning) | {provenance.get('final_sources', 0)} |",
    ]
    tiers = provenance.get("tier_breakdown") or {}
    if tiers:
        tier_str = ", ".join(f"{k}: {v}" for k, v in sorted(tiers.items()))
        lines.append(f"| Final pool by source tier | {tier_str} |")
    if provenance.get("web_records_kept"):
        lines.append(f"| Grey-literature web records in final pool | {provenance.get('web_records_kept')} |")
    if provenance.get("web_seeded"):
        lines.append(
            f"| Web records from seeded fallback (live search unavailable) | {provenance.get('web_seeded')} |"
        )
    if provenance.get("web_search_mode") and provenance.get("web_search_mode") != "off":
        lines.append(f"| Web retrieval mode | {provenance.get('web_search_mode')} |")
    topical = provenance.get("topical") or {}
    if topical.get("dropped"):
        lines.append(
            f"| Sources rejected by embedding topical filter | {topical.get('dropped')} |"
        )
    lines += [
        f"| Full-text evidence chunks | {ev.get('chunks', 0)} |",
        f"| Sources with full text fetched | {ev.get('fetched_sources', 0)} |",
        f"| Evidence corpus size (chars) | {ev.get('total_chars', 0)} |",
        f"| Semantic scoring active | {'yes' if ev.get('semantic_scoring') else 'keyword fallback'} |",
        f"| Citation precision | {provenance.get('citation_precision', 'n/a')} |",
        f"| Claim coverage | {provenance.get('claim_coverage', 'n/a')} |",
    ]
    pruned = provenance.get("pruned_uncited") or []
    if pruned:
        lines.append(f"| Uncited sources pruned | {len(pruned)} |")
    # Metrics that refuse to lie: relevance precision exists ONLY when the
    # embedding topical filter actually ran — run9's "1.0" was padding-only
    # arithmetic printed beside an entire chronic-pain section. Unavailable
    # embeddings ⇒ "not measurable", never a number. Borderline (starvation-
    # rescued sub-floor) sources that got cited count against precision.
    rel_prec = provenance.get("relevance_precision")
    rel_mode = provenance.get("relevance_precision_mode")
    if rel_prec is not None:
        lines.append(f"| Citation precision (structural) | {provenance.get('citation_precision', 'n/a')} |")
        label = (
            "Relevance precision (topical, embedding-gated)"
            if rel_mode == "embedding_topical"
            else "Relevance precision (padding-aware)"
        )
        lines.append(f"| {label} | {rel_prec} |")
        pad_cited = provenance.get("padding_cited") or []
        if pad_cited:
            lines.append(f"| ⚠ Cited sources with zero discriminative overlap | {len(pad_cited)} ([S{'], [S'.join(str(x) for x in pad_cited)}]) |")
        border_cited = provenance.get("borderline_cited") or []
        if border_cited:
            lines.append(
                f"| ⚠ Cited sources below the embedding topical floor (starvation-rescued; topical fit unverified) | [S{'], [S'.join(str(x) for x in border_cited)}] |"
            )
    else:
        lines.append(
            "| Relevance precision (topical) | not measurable — embedding topical filter unavailable this run; lexical padding screen only |  |"
        )
    ed = provenance.get("evidence_depth") or {}
    if ed and ed.get("fulltext_sources", 0) < (ed.get("floor") or 3):
        lines.append(
            f"| ⚠ Evidence depth below floor | {ed.get('fulltext_sources', 0)} full-text source(s) vs floor {ed.get('floor', 3)} — claims rest on titles/abstracts; established language is not supported |  |"
        )
    cov = provenance.get("claim_coverage")
    if cov is not None and cov != "n/a":
        try:
            if float(cov) < 0.6:
                lines.append(f"| ⚠ Claim coverage below 0.60 — {cov} of body claims verified against retrieved source text; treat unverified claims as provisional |  |")
        except (TypeError, ValueError):
            pass
    lines += ["", "Queries executed:"]
    for q in provenance.get("queries", []):
        lines.append(f"- {q}")

    # Screening decisions — WHICH records were excluded and WHY (the
    # "counter table can't tell you what was screened out" gap).
    scr = provenance.get("screening") or {}
    rej = scr.get("relevance_rejected") or []
    resc = scr.get("floor_rescued") or []
    pad_drop = scr.get("padding_dropped") or []
    pruned_uncited = provenance.get("pruned_uncited") or []
    top_rej = topical.get("rejected") or []
    top_border = topical.get("borderline") or []
    top_adj = topical.get("adjudicated") or []
    if any([rej, resc, pad_drop, scr.get("junk_filtered"), pruned_uncited, top_rej, top_border,
            top_adj, topical.get("mode") == "unavailable",
            (topical.get("mode") == "embedding" and topical.get("anchor"))]):
        lines += ["", "Screening decisions (deterministic, auditable):"]
        if scr.get("junk_filtered"):
            lines.append(f"- Junk filter: {scr['junk_filtered']} non-paper records dropped (supplements, bare directories)")
        if topical.get("anchor") and topical.get("mode") == "embedding":
            kind = topical.get("kind") or "?"
            lines.append(
                f"- Topical anchor ({kind} — every source scored against it by embedding cosine, not keywords): "
                f"“{(topical.get('anchor') or '')[:150]}”"
            )
            if topical.get("distractor"):
                lines.append(
                    f"- Topical distractor (sources closer to this than to the phenomenon are rejected): "
                    f"“{(topical.get('distractor') or '')[:150]}”"
                )
        if topical.get("mode") == "unavailable":
            lines.append(
                "- Topical filter: UNAVAILABLE this run (embedding backend failed) — "
                "topical fit NOT verified beyond the lexical padding screen"
            )
        if top_adj:
            shown = "; ".join(
                f"“{(a.get('title') or '')[:60]}” ({a.get('score')} → {a.get('verdict')})"
                for a in top_adj[:5]
            )
            lines.append(
                f"- Topical adjudication (LLM judgment on vector-borderline sources): {shown}"
            )
        if top_rej:
            shown = "; ".join(
                f"“{(r.get('title') or '')[:70]}” ({r.get('score')})"
                for r in top_rej[:5]
            )
            lines.append(
                f"- Embedding topical filter: rejected {len(top_rej)} below floor "
                f"{topical.get('floor')}: {shown}"
            )
        if top_border:
            shown = "; ".join(
                f"“{(r.get('title') or '')[:70]}” ({r.get('score')})"
                for r in top_border[:4]
            )
            lines.append(
                f"- ⚠ Starvation-rescued below topical floor (pool would be empty — topical fit unverified): {shown}"
            )
        if rej:
            shown = "; ".join(
                f"“{(r.get('title') or '')[:70]}” ({r.get('relevance', 0)})"
                for r in rej[:5]
            )
            lines.append(f"- Relevance gate: kept {scr.get('relevance_kept', '?')}, rejected {len(rej)} — lowest scores: {shown}")
        if resc:
            shown = "; ".join(f"“{(r.get('title') or '')[:70]}”" for r in resc[:4])
            lines.append(f"- Floor rescue: {len(resc)} re-admitted below threshold to avoid an empty pool: {shown}")
        if pad_drop:
            shown = "; ".join(f"“{(t or '')[:70]}”" for t in pad_drop[:5])
            lines.append(f"- Off-topic padding: {len(pad_drop)} sources with zero discriminative overlap removed before drafting: {shown}")
        if pruned_uncited:
            lines.append(f"- Dead-source pruning: {len(pruned_uncited)} pool members the final text never cited")
    return "\n".join(lines) + "\n"


async def generate_research_report(
    query: str,
    classification: str = "academic",
    depth: str = "quick",
    max_cycles: int = 3,
    user_id: Optional[str] = None,
    apex_tier: str = "apex-free",
    use_nli: Optional[bool] = None,
) -> ResearchReport:
    """
    APEX 3.0 report pipeline:

        cache check → query expansion → HYBRID research (live ⊕ RAG ⊕ snowball)
        → canonical dedup → diverse selection → full-text EVIDENCE corpus
        → claim extraction + NLI verification → OUTLINE → parallel section
        drafting → assembly (deterministic bibliography) → critique/revise
        → wiki compile + cache

    Args:
        query: Research query
        classification: Query type for routing
        depth: "quick" (4 sections) or "thorough" (up to 8 sections + iterative loop)
        max_cycles: Max research cycles for thorough mode
        user_id: Optional user ID for cache association
        apex_tier: apex-free or apex-premium
        use_nli: Force NLI verification on/off. Default: env APEX_USE_NLI (true)

    Returns:
        ResearchReport with structured findings + wiki/cache metadata
    """
    start = time.time()
    if use_nli is None:
        use_nli = os.getenv("APEX_USE_NLI", "true").lower() in ("1", "true", "yes")

    # ── Step 0: Check LLM Wiki cache (read-side TTL enforced in check_cache) ──
    from agent.llm_wiki import get_wiki_engine
    wiki_engine = get_wiki_engine()

    cached = await wiki_engine.check_cache(query, depth, apex_tier, user_id)
    if cached and cached.report:
        logger.info(f"[APEX 2.0] Cache HIT for: {query[:50]}")
        report = _parse_report(cached.report)
        report.query = query
        report.total_latency_ms = int((time.time() - start) * 1000)
        report.depth = depth
        report.sources = cached.sources if isinstance(cached.sources, list) else []
        report.wiki_cache_id = cached.id
        report.wiki_page_id = cached.wiki_page_id
        report.from_cache = True
        return report

    # ── Step 1: Query expansion (HyDE-style variants) ──
    variants = await expand_query_variants(query, max_variants=3 if depth == "thorough" else 2)
    logger.info(f"Query expanded into {len(variants)} variants")

    # ── Step 2: HYBRID research — live search per variant ⊕ RAG corpus ⊕ web, concurrent ──
    live_tasks = [
        parallel_research(
            v,
            classification,
            max_results_per_source=3,
            max_sources=25 if depth == "thorough" else 15,
        )
        for v in variants
    ]
    rag_task = _rag_sources_for_report(query)
    web_task = _web_sources_for_report(query, variants, depth)

    live_results = await asyncio.gather(*live_tasks, return_exceptions=True)
    rag_sources = await rag_task
    web_sources, web_stats = await web_task

    all_sources: List[Dict] = list(rag_sources) + list(web_sources)
    successful: List[str] = []
    failed: List[str] = []
    for res in live_results:
        if isinstance(res, ParallelResearchResult):
            all_sources.extend(res.results)
            successful.extend(res.successful_sources)
            failed.extend(res.failed_sources)
    if web_stats.get("available"):
        successful.append("zai_web")
    elif web_stats.get("enabled"):
        failed.append("zai_web: bridge unavailable")
    funnel: Dict = {"identified": len(all_sources)}

    # ── Step 3: Canonical dedup → diverse selection → domain-aware decay ──
    all_sources = dedup_sources_canonical(all_sources)
    funnel["post_dedup"] = len(all_sources)
    screening: Dict = {}  # deterministic screening-decision audit (provenance)

    # ── Step 3.1: EMBEDDING TOPICAL FILTER (meaning, not keywords) ──
    # run9 fatal defect: the CT.gov provider stamps "Results posted: NO" into
    # every record snippet, so EVERY registry record carried the discriminative
    # token "results" — and the lexical gate waved a chronic-pain ACT/CBT
    # trial into a publication-bias report, where it became an entire
    # section. Token overlap cannot judge topical fit; embeddings can.
    # Honest degradation: if embeddings are unavailable NOTHING is dropped,
    # screening["topical_mode"]="unavailable", and the metrics layer must
    # then REFUSE to print a topical precision number (a lexical-only "1.0"
    # is exactly the lie run9 printed next to its chronic-pain section).
    try:
        from agent.topical_filter import apply_topical_filter
        pre_topical = len(all_sources)
        all_sources = await apply_topical_filter(query, all_sources, screening=screening)
        funnel["topical_dropped"] = pre_topical - len(all_sources)
    except Exception as e:
        logger.warning(f"Topical filter failed (non-fatal): {e}")
        screening.setdefault("topical_mode", "unavailable")
    funnel["post_topical"] = len(all_sources)

    all_sources = select_diverse_sources(
        all_sources,
        max_sources=28 if depth == "thorough" else 18,
        query=query,
        query_variants=variants,
        screening=screening,
    )
    funnel["post_selection"] = len(all_sources)

    # ── Step 4: Snowball top academic seeds (thorough mode adds the most) ──
    if depth == "thorough":
        snowballed = await _snowball_sources(all_sources)
        if snowballed:
            merged = dedup_sources_canonical(all_sources + snowballed)
            all_sources = select_diverse_sources(
                merged,
                max_sources=30,
                query=query,
                query_variants=variants,
                screening=screening,
            )

    # ── Step 5: Iterative gap-filling (thorough, verification-driven) ──
    if depth == "thorough" and max_cycles > 1:
        claims_pre = _extract_claims_from_sources(all_sources)
        verification_pre = verify_claims_from_sources(claims_pre, all_sources)
        enriched = await _iterative_research(
            query,
            ParallelResearchResult(results=all_sources),
            max_cycles,
            verification=verification_pre,
        )
        if len(enriched.results) > len(all_sources):
            all_sources = select_diverse_sources(
                dedup_sources_canonical(enriched.results),
                max_sources=30,
                query=query,
                query_variants=variants,
                screening=screening,
            )

    # ── Step 5.5: CANONICAL-SOURCE BACKFILL (registry-verified, deterministic) ──
    # Keyword retrieval finds papers CONTAINING the topic's words; the papers
    # that DEFINE those words (Egger's test, trim-and-fill, ROB-ME, ...) are
    # routinely missed — the live failure behind four consecutive critiques.
    # When the query/variants/pool name a registered method-or-topic, fetch
    # the actual primary literature from Crossref BY verified DOI and add it
    # to the pool. Backfilled sources are ordinary pool members: they flow
    # through NLI, evidence fetching (Unpaywall takes DOIs), drafting, and
    # the deterministic bibliography. Failures degrade silently.
    backfilled_ids: List[str] = []
    try:
        from agent.canonical_sources import ensure_canonical_sources
        pool_signal = " ".join(
            f"{s.get('title', '')} {s.get('snippet', '')}" for s in all_sources
        )
        all_sources, backfilled = await ensure_canonical_sources(
            " ".join([query, *variants, pool_signal]),
            all_sources,
            max_add=8 if depth == "thorough" else 5,
        )
        backfilled_ids = [b["entry_id"] for b in backfilled]
    except Exception as e:
        logger.warning(f"Canonical backfill failed (non-fatal): {e}")
    funnel["post_enrichment"] = len(all_sources)
    funnel["canonical_backfilled"] = len(backfilled_ids)

    # ── Step 5.55: TOPICAL FILTER, PASS 2 (snowball/iterative additions) ──
    # Sources added after pass 1 (snowballing, gap-filling cycles) never saw
    # the embedding gate. Canonical-backfilled sources stay exempt — they
    # were fetched BY DOI from the named-method registry (in-domain by
    # construction) — and carry an explicit exemption justification so that
    # EVERY accepted source keeps a one-line auditable reason.
    try:
        from agent.topical_filter import apply_topical_filter
        backfilled_set = set(backfilled_ids)
        protected = [s for s in all_sources if s.get("canonical_id") in backfilled_set]
        for s in protected:
            ex = s.setdefault("extra", {})
            ex.setdefault("topical_ok", True)
            ex.setdefault("topical_justification", "exempt: canonical backfill (registry-attested in-domain)")
        judgeable = [s for s in all_sources if s.get("canonical_id") not in backfilled_set]
        pre_topical2 = len(judgeable)
        judged = await apply_topical_filter(
            query, judgeable, screening=screening, select_new_only=True
        )
        all_sources = judged + protected
        funnel["topical_dropped_pass2"] = pre_topical2 - len(judged)
    except Exception as e:
        logger.warning(f"Topical filter pass-2 failed (non-fatal): {e}")

    # ── Step 5.6: OFF-TOPIC PADDING DROP (deterministic, pre-draft) ──
    # Sources sharing NONE of the query's discriminative vocabulary cannot
    # be evidence about the question — citing them is padding (run8: RNA-seq
    # and brain-imaging meta-analyses cited for an unpublished-trials
    # question, one rationalized into the scope argument). Canonical-
    # backfilled sources are exempt (registry-attested in-domain). Dropping
    # BEFORE drafting beats flagging after: the drafter can't rationalize
    # padding into arguments it never sees.
    try:
        from agent.citation_integrity import scan_off_topic_padding
        padding_pre = scan_off_topic_padding(query, all_sources)
        if padding_pre:
            drop_idx = {f["source"] for f in padding_pre}
            screening["padding_dropped"] = [f["title"] for f in padding_pre]
            all_sources = [
                s for i, s in enumerate(all_sources, 1) if i not in drop_idx
            ]
            funnel["padding_dropped"] = len(padding_pre)
            logger.info(
                f"Padding drop: removed {len(padding_pre)} zero-discriminative-"
                f"overlap sources pre-draft: {[f['title'][:60] for f in padding_pre]}"
            )
    except Exception as e:
        logger.warning(f"Padding drop failed (non-fatal): {e}")

    if not all_sources:
        # Absolute fallback: single-call generation on empty evidence
        raw_report = f"# Research Report\n\n## EXECUTIVE SUMMARY\n\nNo sources could be retrieved for this query. Try again or broaden the query.\n"
        report = _parse_report(raw_report)
        report.query = query
        report.depth = depth
        report.total_latency_ms = int((time.time() - start) * 1000)
        report.sources = []
        report.generation_mode = "single_call"
        return report

    # ── Step 6: FULL-TEXT EVIDENCE CORPUS (the quality lever) ──
    try:
        from agent.evidence_fetcher import build_evidence_corpus, evidence_stats
        corpus = await build_evidence_corpus(all_sources, top_n=6 if depth == "quick" else 8)
        ev_stats = None
    except Exception as e:
        logger.warning(f"Evidence corpus build failed (non-fatal): {e}")
        from agent.evidence_fetcher import EvidenceCorpus
        corpus = EvidenceCorpus()
        ev_stats = {"chunks": 0, "fetched_sources": 0, "failed_sources": 0, "total_chars": 0, "semantic_scoring": False}

    # Full-text depth for the evidence floor (run9 verdict fix 2): the count
    # is factual at build time; only semantic_scoring is corrected post-draft.
    fulltext_count_pre = 0
    try:
        from agent.evidence_fetcher import evidence_stats as _evstats
        fulltext_count_pre = int((_evstats(corpus) or {}).get("fetched_sources", 0) or 0)
    except Exception:
        pass

    # ── Step 7: Claim extraction + verification (heuristic prefilter ⊕ NLI) ──
    claims = _extract_claims_from_sources(all_sources)
    verification = verify_claims_from_sources(claims, all_sources)

    if use_nli and claims:
        try:
            from agent.verifier import verify_claims_with_nli
            nli_results = await verify_claims_with_nli(claims, all_sources)
            verification = _merge_nli_into_verification(verification, nli_results)
        except Exception as e:
            logger.warning(f"NLI verification failed, keeping heuristic statuses (non-fatal): {e}")

    # ── Step 8: OUTLINE ──
    outline, report_title = await _generate_outline(query, all_sources, verification, depth)
    logger.info(f"Report outline: {[s['title'] for s in outline]} | title={report_title!r}")

    # ── Step 9: PARALLEL SECTION DRAFTING ──
    draft_tasks = [
        _draft_section(section, query, all_sources, corpus, verification, depth,
                       fulltext_count=fulltext_count_pre)
        for section in outline
    ]
    draft_results = await asyncio.gather(*draft_tasks, return_exceptions=True)

    sections_md: List[str] = []
    models_used: List[str] = []
    for section, result in zip(outline, draft_results):
        if isinstance(result, Exception) or not isinstance(result, tuple):
            logger.warning(f"Section {section.get('id')} failed entirely: {result}")
            continue
        _sid, md, model_used = result
        if md:
            sections_md.append(md)
            if model_used:
                models_used.append(model_used)

    generation_mode = "sectionwise"

    # Fallback: every section draft failed → legacy single-call path
    if not sections_md:
        logger.warning("All section drafts failed — falling back to single-call generation")
        raw_report = await _generate_report_text(query, all_sources, verification, depth)
        generation_mode = "single_call"
    else:
        raw_report = _assemble_report(sections_md, all_sources, title=report_title)

    # ── Step 10: CRITIQUE + REVISE (citation integrity loop) ──
    # Late canonical backfill: if the DRAFT itself names a registered method
    # (e.g. "trim-and-fill") whose primary source is not in the pool, fetch
    # it now (appended as [S{n+1}...] — existing numbering untouched) and
    # make the revision cite it or drop the name.
    canonical_issues: List[str] = []
    try:
        from agent.canonical_sources import late_backfill_for_draft
        canonical_issues, late_added = await late_backfill_for_draft(raw_report, all_sources)
        backfilled_ids.extend(a["entry_id"] for a in late_added)
    except Exception as e:
        logger.warning(f"Late canonical backfill failed (non-fatal): {e}")

    # Evidence-depth enforcement (run9 verdict fix 2): with < 3 full-text
    # sources, 'established/definitive/conclusive' language enters the
    # revision feedback verbatim — the drafter reasoned from abstracts and
    # must not dress that up as settled evidence.
    established_issues: List[str] = []
    try:
        from agent.citation_integrity import scan_unfounded_established
        for f in scan_unfounded_established(raw_report, fulltext_count_pre):
            established_issues.append(
                f"EVIDENCE_DEPTH: “{f['excerpt']}” — only {f['fulltext_count']} source(s) have fetched full text "
                f"(floor {f['floor']}). Rewrite the sentence without established/definitive language and "
                f"qualify the evidence as abstract-level."
            )
    except Exception as e:
        logger.debug(f"Evidence-depth scan failed (non-fatal): {e}")

    raw_report, critique, revise_model = await _critique_and_revise(
        raw_report, query, all_sources, depth,
        extra_issues=canonical_issues + established_issues,
    )
    if revise_model:
        models_used.append(revise_model)
    critique["canonical_backfilled"] = backfilled_ids
    critique["web_search"] = web_stats

    # Evidence stats are captured AFTER drafting/critique: used_semantic starts
    # optimistic at build time and is corrected during retrieval (degenerate
    # embedding output → keyword fallback). A build-time snapshot lied about
    # semantic scoring in live runs with a provider outage.
    if ev_stats is None:
        try:
            from agent.evidence_fetcher import evidence_stats
            ev_stats = evidence_stats(corpus)
        except Exception:
            ev_stats = {"chunks": 0, "fetched_sources": 0, "failed_sources": 0, "total_chars": 0, "semantic_scoring": False}
    ev_stats["web_search"] = web_stats

    # ── Step 10.5: DETERMINISTIC BIBLIOGRAPHY + DEAD-SOURCE PRUNING ──
    # A revision pass regenerates the whole report INCLUDING its SOURCES
    # section — so first restore the canonical bibliography, then drop any
    # source the body never cites and renumber [Sn] tags deterministically.
    from agent.citation_integrity import prune_uncited_sources, rebuild_bibliography
    raw_report = rebuild_bibliography(raw_report, all_sources)
    raw_report, all_sources, prune_stats = prune_uncited_sources(raw_report, all_sources)
    critique["dead_sources_pruned"] = prune_stats.get("pruned", [])
    if prune_stats.get("pruned"):
        # Recompute metrics on the pruned text
        from agent.citation_integrity import check_report_citations
        post_prune = check_report_citations(raw_report, all_sources)
        critique["citation_precision"] = post_prune.citation_precision
        critique["claim_coverage"] = post_prune.claim_coverage
        logger.info(
            f"Pruned {len(prune_stats['pruned'])} uncited sources: "
            f"{prune_stats['before']} → {prune_stats['after']} (renumbered)"
        )

    # ── Step 10.55: RELEVANCE HONESTY (metrics that can't lie) ──
    # citation_precision counts numbering/decency, not topicality — run8's
    # "1.0" sat next to two irrelevant sources, and run9's sat next to an
    # entire chronic-pain section. Two rules now hold:
    #   1. Topical relevance precision is ONLY claimable when the embedding
    #      filter actually ran (mode == "embedding"). Borderline (starvation-
    #      rescued sub-floor) sources that got cited count against it.
    #   2. When embeddings were unavailable, relevance_precision is None and
    #      the renderer says "not measurable" — a lexical-only "1.0" is
    #      structurally incapable of detecting topical drift and must never
    #      be printed again.
    try:
        from agent.citation_integrity import scan_off_topic_padding as _sop
        from agent.citation_integrity import extract_numbered_citations as _enc
        padding_final = _sop(query, all_sources)
        critique["padding_flags_final"] = len(padding_final)
        body_split = re.split(r"\n##\s+SOURCES\b", raw_report, maxsplit=1, flags=re.I)
        cited_final = sorted(set(_enc(body_split[0])))
        padding_cited = {f["source"] for f in padding_final if f["source"] in cited_final}
        critique["padding_cited"] = sorted(padding_cited)

        topical_mode = (screening or {}).get("topical_mode")
        critique["topical_mode"] = topical_mode
        if topical_mode == "embedding":
            sub_floor_cited = [
                i for i in cited_final
                if 0 < i <= len(all_sources)
                and ((all_sources[i - 1].get("extra") or {}).get("topical_ok") is False)
            ]
            critique["borderline_cited"] = sorted(sub_floor_cited)
            critique["relevance_precision"] = (
                round(1.0 - len(sub_floor_cited) / len(cited_final), 3)
                if cited_final else None
            )
            critique["relevance_precision_mode"] = "embedding_topical"
        else:
            critique["borderline_cited"] = []
            critique["relevance_precision"] = None
            critique["relevance_precision_mode"] = "unavailable"

        # Post-revision evidence-depth check: did the 'established' language
        # survive the revision pass on a sub-floor full-text base?
        try:
            from agent.citation_integrity import scan_unfounded_established as _sue
            _ft = (ev_stats or {}).get("fetched_sources", 0) if ev_stats else 0
            critique["established_flags_final"] = len(_sue(raw_report, _ft))
        except Exception:
            critique["established_flags_final"] = None
    except Exception as e:
        logger.debug(f"Relevance-honesty metrics failed (non-fatal): {e}")
        critique["padding_flags_final"] = None
        critique["relevance_precision"] = None
        critique["relevance_precision_mode"] = "unavailable"

    # ── Step 10.6: PROVENANCE APPENDIX (deterministic audit trail) ──
    # PRISMA-style funnel rendered from pipeline counters — nothing after
    # this point rewrites the text, so the appendix stays consistent.
    provenance: Dict = {}
    if generation_mode == "sectionwise":
        tier_counts: Dict[str, int] = {}
        for s in all_sources:
            t = (s.get("tier") or "UNV").upper()
            tier_counts[t] = tier_counts.get(t, 0) + 1
        provenance = {
            "identified_records": funnel.get("identified", 0),
            "post_dedup": funnel.get("post_dedup", 0),
            "post_selection": funnel.get("post_selection", 0),
            "post_enrichment": funnel.get("post_enrichment", 0),
            "canonical_backfilled": funnel.get("canonical_backfilled", 0),
            "final_sources": len(all_sources),
            "tier_breakdown": tier_counts,
            "web_records_kept": sum(
                1 for s in all_sources if (s.get("extra") or {}).get("web")
            ),
            "web_seeded": sum(
                1 for s in all_sources if (s.get("extra") or {}).get("seeded")
            ),
            "web_search_mode": (web_stats or {}).get("search_mode", "off"),
            "evidence": ev_stats,
            "queries": [query] + [
                v for v in variants if v.strip().lower() != query.strip().lower()
            ],
            "pruned_uncited": prune_stats.get("pruned", []),
            "citation_precision": critique.get("citation_precision"),
            "relevance_precision": critique.get("relevance_precision"),
            "relevance_precision_mode": critique.get("relevance_precision_mode"),
            "borderline_cited": critique.get("borderline_cited", []),
            "padding_cited": critique.get("padding_cited", []),
            "claim_coverage": critique.get("claim_coverage"),
            "topical": {
                "mode": (screening or {}).get("topical_mode"),
                "floor": (screening or {}).get("topical_floor"),
                "anchor": (screening or {}).get("topical_anchor"),
                "kind": ((screening or {}).get("topical_anchors") or {}).get("kind"),
                "distractor": ((screening or {}).get("topical_anchors") or {}).get("distractor"),
                "dropped": (
                    (funnel.get("topical_dropped", 0) or 0)
                    + (funnel.get("topical_dropped_pass2", 0) or 0)
                ),
                "rejected": (screening or {}).get("topical_rejected", []),
                "adjudicated": (screening or {}).get("topical_adjudicated", []),
                "borderline": (screening or {}).get("topical_borderline", []),
            },
            "evidence_depth": {
                "fulltext_sources": (
                    (ev_stats or {}).get("fetched_sources", 0) if ev_stats else 0
                ),
                "floor": 3,
                "established_flags_final": critique.get("established_flags_final"),
            },
            "screening": {
                "relevance_kept": screening.get("relevance_kept"),
                "relevance_rejected": screening.get("relevance_rejected", []),
                "floor_rescued": screening.get("floor_rescued", []),
                "padding_dropped": screening.get("padding_dropped", []),
                "junk_filtered": funnel.get("junk_filtered"),
            },
            "registry": {
                "records": (web_stats or {}).get("registry_records", 0),
                "no_results_records": (web_stats or {}).get("registry_no_results", 0),
            },
        }
        raw_report = raw_report.rstrip("\n") + _render_provenance_section(provenance)

    # ── Step 11: Parse + finalize ──
    report = _parse_report(raw_report)
    report.query = query
    report.title = report_title or query
    report.verification = verification
    report.total_latency_ms = int((time.time() - start) * 1000)
    report.depth = depth
    report.sources = all_sources
    report.from_cache = False
    report.evidence = ev_stats
    report.citation_metrics = critique
    report.provenance = provenance
    report.expanded_queries = variants
    report.generation_mode = generation_mode
    report.model_used = models_used[-1] if models_used else ""

    # ── APEX 2.0 Step 12: Compile into Wiki + Cache ──
    try:
        wiki_page, cache_id = await wiki_engine.research_to_wiki(
            query=query,
            report=raw_report,
            sources=all_sources,
            verification=verification,
            mode=depth,
            apex_tier=apex_tier,
            depth=depth,
            category=classification,
            user_id=user_id,
            original_latency_ms=report.total_latency_ms,
        )
        report.wiki_page_id = wiki_page.id
        report.wiki_cache_id = cache_id
        report.wiki_lifecycle = wiki_page.lifecycle
        report.wiki_slug = wiki_page.slug
        report.wiki_version = wiki_page.version

        logger.info(
            f"[APEX 2.0] Wiki compiled: slug={wiki_page.slug}, "
            f"lifecycle={wiki_page.lifecycle}, cache_id={cache_id}"
        )
    except Exception as e:
        logger.warning(f"[APEX 2.0] Wiki compilation failed (non-fatal): {e}")
        report.wiki_page_id = None
        report.wiki_cache_id = None

    return report


def _merge_nli_into_verification(
    verification: VerificationResult,
    nli_results: Dict[str, Dict],
) -> VerificationResult:
    """
    Overlay NLI verdicts onto the heuristic verification result.
    Claims with NLI data get NLI statuses/confidence; others keep heuristic
    values. Recomputes the summary counters.
    """
    if not nli_results:
        return verification

    for claim in verification.claims:
        nli = nli_results.get(claim.statement)
        if not nli:
            continue
        claim.epistemic_status = nli["status"]
        claim.confidence = nli["confidence"]
        if nli.get("evidence_type"):
            claim.evidence_type = nli["evidence_type"]
        # Attach NLI quotes as structured supporting/conflicting entries
        nli_supports = [
            {"url": s.get("source_url", ""), "title": "", "tier": s.get("source_tier", "UNV"),
             "quote": s.get("quote", ""), "supports": True, "conflicts": False}
            for s in nli.get("supports", [])
        ]
        nli_conflicts = [
            {"url": s.get("source_url", ""), "title": "", "tier": s.get("source_tier", "UNV"),
             "quote": s.get("quote", ""), "supports": False, "conflicts": True}
            for s in nli.get("contradictions", [])
        ]
        if nli_supports:
            claim.supporting_sources = nli_supports[:5]
        if nli_conflicts:
            claim.conflicting_sources = nli_conflicts[:5]

    # Recompute summary counters
    verification.established_count = sum(1 for c in verification.claims if c.epistemic_status == EPISTEMIC_ESTABLISHED)
    verification.tentative_count = sum(1 for c in verification.claims if c.epistemic_status == EPISTEMIC_TENTATIVE)
    verification.contested_count = sum(1 for c in verification.claims if c.epistemic_status == EPISTEMIC_CONTESTED)
    verification.unverifiable_count = sum(1 for c in verification.claims if c.epistemic_status == EPISTEMIC_UNVERIFIED)

    if verification.claims:
        statuses = [c.epistemic_status for c in verification.claims]
        verification.summary = (
            f"{verification.established_count} established, "
            f"{verification.tentative_count} tentative, "
            f"{verification.contested_count} contested, "
            f"{verification.unverifiable_count} unverified "
            f"(NLI-verified: {len(nli_results)}/{len(verification.claims)})"
        )

    return verification


def _extract_claims_from_sources(sources: List[Dict]) -> List[str]:
    """
    Extract key claims from source snippets.

    Uses simple heuristics: look for sentences with factual assertions
    (contain numbers, causal language, or comparison).
    """
    claims = []
    claim_patterns = re.compile(
        r'\b(found|showed|demonstrated|indicates|suggests|reveals|proves|confirmed|associated with|'
        r'linked to|increases?|decreases?|improves?|reduces?|prevents?|causes?|'
        r'\d+%|\d+ percent|significant|effective|ineffective)\b',
        re.I,
    )

    for source in sources[:10]:  # Limit to top 10 sources
        snippet = source.get("snippet", "")
        if not snippet or len(snippet) < 50:
            continue

        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', snippet)
        for sentence in sentences:
            if len(sentence) > 20 and claim_patterns.search(sentence):
                # Clean and add as claim
                claim = sentence.strip()
                if claim not in claims:
                    claims.append(claim)

    return claims[:15]  # Max 15 claims


async def _generate_report_text(
    query: str,
    sources: List[Dict],
    verification: VerificationResult,
    depth: str = "quick",
) -> str:
    """Generate the structured report text via LLM (fallback single-call path).

    Routes with task_type="report" so mid/capable models (4k-8k output caps)
    handle it — the old code routed through the full chain starting at the
    150-token cheap models, silently truncating reports to ~110 words.
    """
    from agent.llm_router import synthesize_with_router

    # Select prompt based on depth
    is_thorough = depth == "thorough"
    system_prompt = RESEARCH_REPORT_SYSTEM_PROMPT_THOROUGH if is_thorough else RESEARCH_REPORT_SYSTEM_PROMPT_QUICK
    max_tokens = 4096 if is_thorough else 2048

    # Build context from sources and verification
    context_parts = ["RESEARCH QUERY: " + query]

    context_parts.append("\n\nSOURCES:")
    for i, s in enumerate(sources[:25], 1):  # Pass up to 25 sources (was 15)
        tier = s.get("tier", "UNV")
        title = s.get("title", "Untitled")
        snippet = s.get("snippet", "")[:1000]  # Full snippets (was 200 chars)
        url = s.get("url", "")
        context_parts.append(f"[{i}] [{tier}] {title}\n    {snippet}\n    URL: {url}")

    context_parts.append("\n\nVERIFIED CLAIMS:")
    for claim in verification.claims:
        status = claim.epistemic_status
        confidence = claim.confidence
        evidence = claim.evidence_type
        context_parts.append(
            f"- [{status}] (conf={confidence}, evidence={evidence}) {claim.statement[:500]}"  # Full claims (was 150 chars)
        )
        if claim.conflicting_sources:
            context_parts.append(f"  CONFLICTS: {len(claim.conflicting_sources)} sources disagree")

    context = "\n".join(context_parts)

    # Use depth-appropriate system prompt and token budget
    router_result = await synthesize_with_router(
        query=query,
        context=context,
        max_tokens=max_tokens,  # 2048 quick / 4096 thorough
        similarity=0.3,  # Route to strongest models
        table_needed=True,
        system_prompt=system_prompt,
        task_type="report",  # mid/capable models ONLY — never 150-token cheap tiers
    )

    content = router_result.content or ""
    # Honest total-failure: the router returns an [ALL_LLM_FAILED] sentinel when
    # every model in the chain fails (live-seen under a daily-quota outage).
    # Raise here instead of letting a stub masquerade as a report through
    # critique → pruning → parsing — callers must see the real problem.
    if content.startswith("[ALL_LLM_FAILED]") or len(content.strip()) < 80:
        raise RuntimeError(
            "APEX report generation failed: all LLM providers returned no usable "
            f"output. Router said: {content[:160]}"
        )
    return content


def _parse_report(raw: str) -> ResearchReport:
    """Parse the LLM output into a structured ResearchReport."""
    report = ResearchReport(query="")

    # Extract executive summary
    summary_match = re.search(
        r'##\s*EXECUTIVE\s+SUMMARY\s*\n(.*?)(?=\n##|\Z)', raw, re.DOTALL | re.I
    )
    if summary_match:
        report.executive_summary = summary_match.group(1).strip()

    # Extract findings table rows
    findings_match = re.search(
        r'##\s*KEY\s+FINDINGS\s*\n(.*?)(?=\n##|\Z)', raw, re.DOTALL | re.I
    )
    if findings_match:
        findings_text = findings_match.group(1)
        # Parse markdown table rows
        for line in findings_text.split("\n"):
            if "|" in line and not line.strip().startswith("|-") and not line.strip().startswith("| Finding"):
                cells = [c.strip() for c in line.split("|") if c.strip()]
                if len(cells) >= 3:
                    report.findings.append({
                        "finding": cells[0],
                        "evidence": cells[1] if len(cells) > 1 else "",
                        "source": cells[2] if len(cells) > 2 else "",
                        "tier": cells[3] if len(cells) > 3 else "",
                        "status": cells[4] if len(cells) > 4 else "",
                    })

    # Extract debates
    debate_match = re.search(
        r'##\s*ACTIVE\s+DEBATES?\s*\n(.*?)(?=\n##|\Z)', raw, re.DOTALL | re.I
    )
    if debate_match:
        debate_text = debate_match.group(1).strip()
        for line in debate_text.split("\n"):
            line = line.strip()
            if line and not line.startswith("#"):
                report.debates.append({"point": line})

    # Extract speculative findings
    spec_match = re.search(
        r'##\s*SPECULATIVE\s*(?:FINDINGS?)?\s*\n(.*?)(?=\n##|\Z)', raw, re.DOTALL | re.I
    )
    if spec_match:
        for line in spec_match.group(1).strip().split("\n"):
            line = line.strip()
            if line and not line.startswith("#"):
                report.speculative.append(line)

    report.raw_report = raw
    return report


# ═══════════════════════════════════════════════════════════════
# UPGRADE #2: ITERATIVE RESEARCH LOOP
# ═══════════════════════════════════════════════════════════════

async def _iterative_research(
    query: str,
    initial_research: ParallelResearchResult,
    max_cycles: int = 3,
    verification: Optional[VerificationResult] = None,
) -> ParallelResearchResult:
    """
    Iterative research: search, identify gaps, search again.

    Gap identification is VERIFICATION-DRIVEN when a VerificationResult is
    supplied: the gap-finder sees which claims are TENTATIVE / ACTIVE_DEBATE /
    UNVERIFIED and targets those holes specifically, instead of squinting at
    ten 100-char titles.

    Opt-in feature: only triggered by depth="thorough".

    Args:
        query: Original query
        initial_research: Results from first research cycle
        max_cycles: Maximum number of research cycles
        verification: Optional verification result driving gap discovery

    Returns:
        Enriched ParallelResearchResult with additional sources
    """
    from agent.llm_router import synthesize_with_router

    all_results = list(initial_research.results)
    all_successful = list(initial_research.successful_sources)
    all_failed = list(initial_research.failed_sources)

    for cycle in range(1, max_cycles):
        # Identify gaps using verification data (richer) or raw titles (fallback)
        context = _build_gap_context(query, all_results, verification)
        gaps = await _identify_gaps(query, context)

        if not gaps:
            logger.info(f"Research cycle {cycle}: No gaps identified, stopping.")
            break

        logger.info(f"Research cycle {cycle}: Found {len(gaps)} gaps: {gaps[:3]}")

        # Search for each gap in parallel
        gap_tasks = []
        for gap in gaps[:3]:  # Max 3 gaps per cycle
            gap_tasks.append(parallel_research(gap, max_results_per_source=2))

        gap_results = await asyncio.gather(*gap_tasks, return_exceptions=True)

        new_results_count = 0
        for result in gap_results:
            if isinstance(result, ParallelResearchResult):
                for r in result.results:
                    # Deduplicate by URL
                    existing_urls = {s.get("url") for s in all_results}
                    if r.get("url") not in existing_urls:
                        all_results.append(r)
                        new_results_count += 1
                all_successful.extend(result.successful_sources)
                all_failed.extend(result.failed_sources)

        logger.info(f"Research cycle {cycle}: Added {new_results_count} new results")

        if new_results_count == 0:
            logger.info(f"Research cycle {cycle}: No new results, stopping.")
            break

    return ParallelResearchResult(
        successful_sources=list(set(all_successful)),
        failed_sources=list(set(all_failed)),
        results=all_results,
        total_latency_ms=initial_research.total_latency_ms,
    )


def _build_gap_context(
    query: str,
    results: List[Dict],
    verification: Optional[VerificationResult] = None,
) -> str:
    """
    Build context string for gap identification.

    With verification data, leads with the epistemic holes (TENTATIVE /
    ACTIVE_DEBATE / UNVERIFIED claims) — telling the gap-finder exactly where
    the evidence is weak — instead of just listing titles.
    """
    parts = [f"Original query: {query}\n"]
    parts.append(f"Sources found so far: {len(results)}\n")

    if verification and verification.claims:
        parts.append("VERIFICATION STATUS OF CURRENT EVIDENCE:\n")
        for c in verification.claims[:12]:
            conflict_note = f" — {len(c.conflicting_sources)} conflicting sources" if c.conflicting_sources else ""
            parts.append(f"  [{c.epistemic_status}] (conf={c.confidence}) {c.statement[:160]}{conflict_note}")
        weak = sum(1 for c in verification.claims if c.epistemic_status in (EPISTEMIC_TENTATIVE, EPISTEMIC_UNVERIFIED, EPISTEMIC_CONTESTED))
        parts.append(f"\n{weak} of {len(verification.claims)} claims are weakly supported or contested.\n")

    parts.append("Key findings from sources:\n")
    for i, r in enumerate(results[:10], 1):
        parts.append(f"  [{i}] {r.get('title', 'Untitled')[:80]}: {r.get('snippet', '')[:100]}")

    return "\n".join(parts)


async def _identify_gaps(query: str, context: str) -> List[str]:
    """
    Use LLM to identify knowledge gaps in current research.

    Returns list of gap descriptions that become new search queries.
    """
    from agent.llm_router import synthesize_with_router

    gap_prompt = f"""Given this research query and the current evidence status, identify 1-3 specific knowledge gaps that need more research.

{context}

Prioritize gaps that would upgrade weak claims (TENTATIVE/UNVERIFIED) to established, or resolve conflicts (ACTIVE_DEBATE). What specific information is still missing or unclear? Return ONLY the gaps as numbered search queries, nothing else.
Format:
1. [specific gap as a search query]
2. [specific gap as a search query]
3. [specific gap as a search query]"""

    router_result = await synthesize_with_router(
        query="identify knowledge gaps",
        context=gap_prompt,
        max_tokens=150,
        similarity=0.5,  # Use mid-tier model
        system_prompt="You identify missing information in research. Be specific. Return only numbered gaps.",
    )

    # Parse gaps from LLM output
    gaps = []
    for line in router_result.content.split("\n"):
        line = line.strip()
        # Match numbered items: "1. gap text" or "- gap text"
        match = re.match(r'^\d+[\.\)]\s*(.+)', line)
        if match:
            gap = match.group(1).strip()
            if len(gap) > 10:  # Skip very short/poor gaps
                gaps.append(gap)
        elif line.startswith("- ") and len(line) > 12:
            gaps.append(line[2:].strip())

    return gaps[:3]


# ═══════════════════════════════════════════════════════════════
# UPGRADE #6: STRUCTURED EXTRACTION
# ═══════════════════════════════════════════════════════════════

EXTRACTION_PROMPT = """Extract structured claims from this text. Return ONLY valid JSON, no other text.

Schema:
{
  "claims": [
    {
      "statement": "the factual claim in one sentence",
      "evidence_type": "RCT|cohort|meta-analysis|systematic-review|case-control|expert-opinion|observation|anecdote",
      "sample_size": null or integer,
      "confidence": "high|medium|low",
      "year": null or integer
    }
  ],
  "methodology_notes": "brief notes on study design if mentioned",
  "conflicts_with": []
}

Text to extract from:"""


@dataclass
class ExtractedClaims:
    """Structured claims extracted from a source."""
    source_url: str
    source_title: str
    source_tier: str
    claims: List[Dict] = field(default_factory=list)
    methodology_notes: str = ""
    extraction_success: bool = False


async def extract_claims_from_source(
    source: Dict,
) -> ExtractedClaims:
    """
    Extract structured claims from a single P1 source.

    Uses a small LLM to extract:
    - Individual claims with evidence type and confidence
    - Methodology notes
    - Conflicts with other sources

    Only applied to P1 sources to avoid wasting tokens on low-quality content.

    Args:
        source: Source dict with url, title, snippet, tier

    Returns:
        ExtractedClaims with structured data
    """
    from agent.llm_router import synthesize_with_router

    text = source.get("snippet", "")
    if not text or len(text) < 50:
        return ExtractedClaims(
            source_url=source.get("url", ""),
            source_title=source.get("title", ""),
            source_tier=source.get("tier", "UNV"),
            extraction_success=False,
        )

    prompt = f"{EXTRACTION_PROMPT}\n\n{text[:1500]}"

    try:
        router_result = await synthesize_with_router(
            query="extract claims",
            context=prompt,
            max_tokens=200,
            similarity=0.5,
            system_prompt="Extract structured claims. Return ONLY valid JSON.",
        )

        content = router_result.content.strip()

        # Try to parse JSON from the response
        # Find JSON object in the response
        json_match = re.search(r'\{[\s\S]*\}', content)
        if json_match:
            data = json.loads(json_match.group())
            return ExtractedClaims(
                source_url=source.get("url", ""),
                source_title=source.get("title", ""),
                source_tier=source.get("tier", "UNV"),
                claims=data.get("claims", []),
                methodology_notes=data.get("methodology_notes", ""),
                extraction_success=True,
            )

    except (json.JSONDecodeError, Exception) as e:
        logger.debug(f"Claim extraction failed for {source.get('url', '')}: {e}")

    return ExtractedClaims(
        source_url=source.get("url", ""),
        source_title=source.get("title", ""),
        source_tier=source.get("tier", "UNV"),
        extraction_success=False,
    )


async def extract_claims_from_p1_sources(
    sources: List[Dict],
) -> List[ExtractedClaims]:
    """
    Extract structured claims from all P1 sources in parallel.

    Args:
        sources: List of source dicts

    Returns:
        List of ExtractedClaims (only for P1 sources)
    """
    p1_sources = [s for s in sources if s.get("tier") == "P1"]

    if not p1_sources:
        return []

    tasks = [extract_claims_from_source(s) for s in p1_sources[:5]]  # Max 5 P1 sources
    results = await asyncio.gather(*tasks, return_exceptions=True)

    extracted = []
    for result in results:
        if isinstance(result, ExtractedClaims) and result.extraction_success:
            extracted.append(result)

    return extracted


# ═══════════════════════════════════════════════════════════════
# RETRACTION DETECTION (Bonus from roadmap review)
# ═══════════════════════════════════════════════════════════════

async def check_retraction(doi: str) -> Dict:
    """
    Check if a paper with a given DOI has been retracted.

    Uses Crossref API to check retraction status.
    Low-hanging fruit: PubMed/Crossref expose retraction metadata.

    Args:
        doi: DOI string (e.g., "10.1038/s41586-020-1234-5")

    Returns:
        Dict with 'is_retracted', 'retraction_notice', 'retraction_date'
    """
    if not doi:
        return {"is_retracted": False, "checked": False}

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get(
                f"https://api.crossref.org/works/{doi}",
                headers={"User-Agent": "APEX-Research-Agent/1.0 (mailto:apex-research@example.com)"},
            )
            if r.status_code != 200:
                return {"is_retracted": False, "checked": True, "doi": doi}

            data = r.json().get("message", {})

            # Check for retraction indicators
            is_retracted = False
            retraction_notice = ""
            retraction_date = ""

            # Check assertion flag
            assertions = data.get("assertion", [])
            for assertion in assertions:
                if assertion.get("name") == "retracted" or "retract" in assertion.get("value", "").lower():
                    is_retracted = True
                    retraction_notice = assertion.get("value", "")

            # Check update-to (Crossref retraction policy)
            updates = data.get("update-to", [])
            for update in updates:
                if "retract" in update.get("type", "").lower():
                    is_retracted = True
                    retraction_notice = update.get("label", "Retracted")
                    retraction_date = update.get("updated-date", "")

            # Check title for retraction signals
            titles = data.get("title", [])
            for title in titles:
                if "retract" in title.lower():
                    is_retracted = True
                    retraction_notice = title

            return {
                "is_retracted": is_retracted,
                "retraction_notice": retraction_notice,
                "retraction_date": retraction_date,
                "doi": doi,
                "checked": True,
            }

    except Exception as e:
        logger.debug(f"Retraction check failed for {doi}: {e}")
        return {"is_retracted": False, "checked": False, "doi": doi, "error": str(e)}


# ═══════════════════════════════════════════════════════════════
# QUERY DECOMPOSITION (Bonus from roadmap review)
# ═══════════════════════════════════════════════════════════════

async def decompose_query(query: str) -> List[str]:
    """
    Decompose a complex query into sub-queries for more targeted search.

    "Is drug X effective for condition Y in elderly patients?"
    → ["drug X efficacy condition Y", "drug X elderly patients", "drug X safety elderly"]

    Uses LLM for decomposition, falls back to simple splitting.
    """
    from agent.llm_router import synthesize_with_router

    # Simple heuristic: check if query is complex enough to decompose
    word_count = len(query.split())
    if word_count < 8:
        return [query]  # Simple query, no decomposition needed

    # Check for PICO-style medical queries
    pico_patterns = re.compile(
        r'\b(effective|efficacy|safety|treatment|therapy|outcome|'
        r'compared to|versus|vs|in patients with|for the treatment of)\b',
        re.I,
    )

    if not pico_patterns.search(query) and " and " not in query.lower() and " or " not in query.lower():
        return [query]  # Not complex enough

    prompt = f"""Decompose this research query into 2-4 specific sub-queries for targeted searching.
Each sub-query should focus on one aspect of the original question.

Original: {query}

Return ONLY the sub-queries as numbered items, nothing else:"""

    try:
        router_result = await synthesize_with_router(
            query="decompose query",
            context=prompt,
            max_tokens=100,
            similarity=0.5,
            system_prompt="Decompose research queries into sub-queries. Return only numbered items.",
        )

        sub_queries = []
        for line in router_result.content.split("\n"):
            match = re.match(r'^\d+[\.\)]\s*(.+)', line.strip())
            if match:
                sub_query = match.group(1).strip()
                if len(sub_query) > 5:
                    sub_queries.append(sub_query)

        return sub_queries if sub_queries else [query]

    except Exception:
        return [query]


# ═══════════════════════════════════════════════════════════════
# CONVENIENCE: FULL RESEARCH PIPELINE
# ═══════════════════════════════════════════════════════════════

async def deep_research(
    query: str,
    classification: str = "academic",
    depth: str = "quick",
    verify: bool = True,
    extract: bool = False,
    check_retractions: bool = False,
    use_nli: Optional[bool] = None,
) -> Dict:
    """
    Full deep research pipeline combining all upgrades.

    Args:
        query: Research query
        classification: Query type
        depth: "quick" (1 cycle) or "thorough" (iterative)
        verify: Whether to verify claims
        extract: Whether to extract structured claims from P1 sources
        check_retractions: Whether to check DOIs for retraction status
        use_nli: LLM entailment verification on top of the keyword prefilter.
            Default: env APEX_USE_NLI ("false" for deep_research — cheaper;
            generate_research_report defaults to true)

    Returns:
        Complete research results dict
    """
    start = time.time()
    if use_nli is None:
        use_nli = os.getenv("APEX_USE_NLI_DEEP", "false").lower() in ("1", "true", "yes")

    # Step 1: Query decomposition
    sub_queries = await decompose_query(query)
    logger.info(f"Decomposed into {len(sub_queries)} sub-queries")

    # Step 2: Parallel research for each sub-query
    research_tasks = [
        parallel_research(sq, classification, max_results_per_source=3)
        for sq in sub_queries
    ]
    research_results = await asyncio.gather(*research_tasks, return_exceptions=True)

    # Merge results
    all_sources = []
    all_successful = []
    all_failed = []
    for result in research_results:
        if isinstance(result, ParallelResearchResult):
            all_sources.extend(result.results)
            all_successful.extend(result.successful_sources)
            all_failed.extend(result.failed_sources)

    # Deduplicate by URL
    seen_urls = set()
    unique_sources = []
    for s in all_sources:
        url = s.get("url", "")
        if url not in seen_urls:
            seen_urls.add(url)
            unique_sources.append(s)

    # Step 3: Tier enforcement + temporal decay
    unique_sources = tier_enforce(unique_sources)
    unique_sources = apply_temporal_decay(unique_sources)

    # Step 4: Iterative research (if thorough)
    if depth == "thorough":
        initial = ParallelResearchResult(
            successful_sources=list(set(all_successful)),
            failed_sources=list(set(all_failed)),
            results=unique_sources,
        )
        enriched = await _iterative_research(query, initial, max_cycles=3)
        unique_sources = enriched.results

    # Step 5: Verification
    verification = None
    if verify and unique_sources:
        claims = _extract_claims_from_sources(unique_sources)
        verification = verify_claims_from_sources(claims, unique_sources)
        if use_nli and claims:
            try:
                from agent.verifier import verify_claims_with_nli
                nli_results = await verify_claims_with_nli(claims, unique_sources)
                verification = _merge_nli_into_verification(verification, nli_results)
            except Exception as e:
                logger.warning(f"NLI verification failed, keeping heuristic statuses (non-fatal): {e}")

    # Step 6: Structured extraction (opt-in)
    extracted = None
    if extract:
        extracted = await extract_claims_from_p1_sources(unique_sources)

    # Step 7: Retraction checks (opt-in)
    retraction_results = []
    if check_retractions:
        doi_tasks = []
        for s in unique_sources:
            doi = s.get("doi") or s.get("extra", {}).get("doi")
            if doi:
                doi_tasks.append(check_retraction(doi))
        if doi_tasks:
            retraction_results = await asyncio.gather(*doi_tasks, return_exceptions=True)
            retraction_results = [
                r for r in retraction_results
                if isinstance(r, dict) and r.get("is_retracted")
            ]

    latency_ms = int((time.time() - start) * 1000)

    return {
        "query": query,
        "sub_queries": sub_queries,
        "sources": unique_sources,
        "source_count": len(unique_sources),
        "successful_sources": list(set(all_successful)),
        "failed_sources": list(set(all_failed)),
        "verification": {
            "claims": [
                {
                    "statement": c.statement,
                    "status": c.epistemic_status,
                    "confidence": c.confidence,
                    "evidence_type": c.evidence_type,
                    "supporting_count": len(c.supporting_sources),
                    "conflicting_count": len(c.conflicting_sources),
                }
                for c in verification.claims
            ] if verification else [],
            "summary": {
                "established": verification.established_count,
                "tentative": verification.tentative_count,
                "contested": verification.contested_count,
                "unverifiable": verification.unverifiable_count,
                "total_sources_checked": verification.total_sources_checked,
            } if verification else {},
        },
        "extracted_claims": [
            {
                "source_url": e.source_url,
                "claims": e.claims,
                "methodology_notes": e.methodology_notes,
            }
            for e in (extracted or [])
        ],
        "retractions": retraction_results,
        "depth": depth,
        "latency_ms": latency_ms,
    }
