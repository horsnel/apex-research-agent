"""
Evidence Fetcher — full-text evidence pipeline for research reports.

Reports built from 1000-char search snippets are "snippet-summarized", not
"evidence-grounded". This module closes that gap by:

1. Fetching FULL TEXT for the final source pool via a credential-free,
   multi-strategy stack (run10 postmortem: the old chain died on publisher
   403s because it had exactly one generic route; live-probed alternatives):

   - ClinicalTrials.gov records → API v2 fielded fetch (protocol + results
     modules): a registry record's "full text" IS its registry entry.
   - arXiv records → export.arxiv.org PDF → pdf_ingest extraction.
   - DOI-bearing academic sources → Europe PMC ID resolution → NCBI eutils
     efetch (db=pmc) full-text XML → structured text. (Europe PMC fullTextXML
     returns 500 from some egress IPs but its SEARCH endpoint resolves
     DOI→PMCID reliably; eutils serves the article body.)
   - DOI → Unpaywall OA PDF → pdf_ingest extraction.
   - DOI → OpenAlex work record → abstract reconstructed from the inverted
     index (abstract-level evidence: strictly better than a 160-char
     search snippet when no full text is reachable).
   - Web/grey-literature records → bridge page_reader (JS rendering).
   - Everything else → direct scrape with browser User-Agent.

   Every attempt is recorded as a diagnostic ({strategy, ok, chars, detail})
   so the report can show WHY a source has no full text instead of silently
   reporting "0 fetched" (run10's failure mode).
2. Chunking full text (reusing ingest/chunker, header-aware, ~400 tokens)
3. Providing targeted per-claim / per-section retrieval over the fresh
   mini-corpus:
   - Semantic scoring (Cloudflare/OpenAI embeddings) when credentials exist
   - Keyword-overlap scoring as zero-credential fallback
4. Formatting evidence blocks with stable [S{n}] source attribution so every
   generated sentence can be traced back to a source index.
5. A deterministic BODY-LEVEL provisional notice (insert_body_notice): when
   fewer than the floor of sources have full text, the report says so in its
   body — not only in the appendix audit table.

All heavy imports (httpx, embedder, pdf tools) are lazy so this module can be
imported and unit-tested without optional dependencies.
"""

import asyncio
import logging
import math
import os
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Configuration ──
EVIDENCE_TOP_SOURCES = int(os.getenv("EVIDENCE_TOP_SOURCES", "6"))
EVIDENCE_MAX_FETCH = int(os.getenv("EVIDENCE_MAX_FETCH", "64"))
EVIDENCE_MAX_CHARS_PER_SOURCE = int(os.getenv("EVIDENCE_MAX_CHARS_PER_SOURCE", "40000"))
EVIDENCE_CHUNK_TOKENS = int(os.getenv("EVIDENCE_CHUNK_TOKENS", "400"))
EVIDENCE_CHUNK_OVERLAP = float(os.getenv("EVIDENCE_CHUNK_OVERLAP", "0.15"))
EVIDENCE_CHUNKS_PER_SECTION = int(os.getenv("EVIDENCE_CHUNKS_PER_SECTION", "6"))
EVIDENCE_FETCH_TIMEOUT = float(os.getenv("EVIDENCE_FETCH_TIMEOUT", "30"))
EVIDENCE_FETCH_CONCURRENCY = int(os.getenv("EVIDENCE_FETCH_CONCURRENCY", "4"))

CHARS_PER_TOKEN = 4

# Routes that produced the document itself (or its authoritative record).
# `openalex_abstract` is EXCLUDED deliberately: it is abstract-level metadata
# from an external API — counting it toward the evidence-depth floor would be
# the same metric-inflation the run10 verdict banned ("claim coverage was
# measuring citation attachment, not evidentiary support"). A floor of 3
# means 3 sources whose text the system actually READ.
FULLTEXT_ROUTES = {"pmc_efetch", "unpaywall_pdf", "arxiv_pdf", "ctgov_api",
                   "page_reader", "scrape"}

# Browser-like headers: run10 postmortem — the "APEX-Research-Agent/1.0" UA
# was 403'd by every academic publisher (Cloudflare). Direct scraping with a
# browser UA still fails on Cloudflare-protected domains, but rescues plain
# HTTP sites (NGO blogs, government pages, open repositories).
BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


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
    # Per-URL diagnostics: which strategy produced the text ("pmc_efetch",
    # "unpaywall_pdf", "arxiv_pdf", "ctgov_api", "openalex_abstract",
    # "page_reader", "scrape") and every attempt's outcome — the "why did
    # this source have no full text" audit trail (run10's missing piece).
    fetch_routes: Dict[str, str] = field(default_factory=dict)
    attempts: Dict[str, List[Dict]] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.chunks)


# ═══════════════════════════════════════════════════════════════
# FULL-TEXT FETCHING — credential-free multi-strategy stack
# ═══════════════════════════════════════════════════════════════


def _source_score(s: Dict) -> float:
    """Score used to rank which sources deserve full-text fetching."""
    score = s.get("adjusted_score", s.get("score", s.get("similarity", 0.0)) or 0.0)
    # Tier bonus: P1 evidence is worth fetching even at lower similarity
    tier_bonus = {"P1": 0.15, "P2": 0.08, "P3": 0.0, "UNV": -0.05}
    return float(score) + tier_bonus.get(s.get("tier", "UNV"), 0.0)


# ── Identity extraction ──

_DOI_RE = re.compile(r"\b(10\.\d{4,9}/[^\s\"<>\\]+)", re.I)
_NCT_RE = re.compile(r"(NCT\d{8})", re.I)
_ARXIV_RE = re.compile(r"arxiv\.org/(?:abs|pdf)/([0-9]{4}\.[0-9]{4,5})(?:v\d+)?|arXiv:([0-9]{4}\.[0-9]{4,5})", re.I)


def _extract_doi(source: Dict) -> Optional[str]:
    """Extract a DOI from fields, extra metadata, or — critically — from URLs.

    run10 postmortem: seeded/search records carry the DOI inside the URL
    ("https://doi.org/10.1136/bmj.315.7109.629") while the old code only
    read source["doi"] — the academic routes were dead on arrival.
    """
    doi = source.get("doi") or (source.get("extra", {}) or {}).get("doi")
    if doi:
        m = _DOI_RE.search(str(doi))
        return m.group(1).rstrip(".,;)") if m else None
    url = source.get("url", "") or ""
    m = _DOI_RE.search(url)
    if m:
        return m.group(1).rstrip(".,;)")
    return None


def _extract_nct_id(source: Dict) -> Optional[str]:
    for v in (source.get("url", ""), source.get("title", ""), source.get("external_id", "") or ""):
        m = _NCT_RE.search(str(v or ""))
        if m:
            return m.group(1).upper()
    return None


def _extract_arxiv_id(source: Dict) -> Optional[str]:
    url = source.get("url", "") or ""
    m = _ARXIV_RE.search(url) or _ARXIV_RE.search(str(source.get("extra", {}) or {}))
    if m:
        return m.group(1) or m.group(2)
    return None


def _is_web_record(s: Dict) -> bool:
    """True for grey-literature records retrieved via the z-ai web provider."""
    return s.get("source_name") == "zai_web" or bool((s.get("extra") or {}).get("web"))


# ── Strategy 1: ClinicalTrials.gov API v2 (registry records) ──

_CTGOV_FIELDS = ",".join([
    "protocolSection.identificationModule",
    "protocolSection.descriptionModule",
    "protocolSection.designModule",
    "protocolSection.armsInterventionsModule",
    "protocolSection.outcomesModule",
    "protocolSection.eligibilityModule",
    "resultsSection",
])


def _json_to_text(obj, depth: int = 0, lines: Optional[List[str]] = None) -> str:
    """Flatten nested JSON into readable labelled lines (registry full text)."""
    if lines is None:
        lines = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            label = re.sub(r"([a-z])([A-Z])", r"\1 \2", k).title()
            if isinstance(v, (dict, list)):
                lines.append(f"{'  ' * depth}{label}:")
                _json_to_text(v, depth + 1, lines)
            elif v not in (None, ""):
                lines.append(f"{'  ' * depth}{label}: {v}")
    elif isinstance(obj, list):
        for item in obj:
            if isinstance(item, (dict, list)):
                _json_to_text(item, depth, lines)
            elif item not in (None, ""):
                lines.append(f"{'  ' * depth}- {item}")
    return "\n".join(lines)


async def _fetch_ctgov_text(source: Dict, client) -> Tuple[str, Dict]:
    """Registry record 'full text' = the study's protocol + results modules.

    A trial that never published has exactly one authoritative document: its
    registry entry. The API v2 fielded fetch returns it without keys.
    """
    nct = _extract_nct_id(source)
    if not nct:
        return "", {"strategy": "ctgov_api", "ok": False, "chars": 0, "detail": "no NCT id"}
    try:
        r = await client.get(
            f"https://clinicaltrials.gov/api/v2/studies/{nct}",
            params={"fields": _CTGOV_FIELDS},
        )
        if r.status_code != 200:
            return "", {"strategy": "ctgov_api", "ok": False, "chars": 0,
                        "detail": f"HTTP {r.status_code}"}
        text = _json_to_text(r.json())
        ok = len(text) > 500
        return (text[:EVIDENCE_MAX_CHARS_PER_SOURCE] if ok else "",
                {"strategy": "ctgov_api", "ok": ok, "chars": len(text),
                 "detail": nct})
    except Exception as e:
        return "", {"strategy": "ctgov_api", "ok": False, "chars": 0,
                    "detail": f"{type(e).__name__}: {str(e)[:120]}"}


# ── Strategy 2: arXiv PDF ──

async def _fetch_arxiv_text(source: Dict, client) -> Tuple[str, Dict]:
    arxiv_id = _extract_arxiv_id(source)
    if not arxiv_id:
        return "", {"strategy": "arxiv_pdf", "ok": False, "chars": 0, "detail": "no arxiv id"}
    try:
        from ingest.pdf_ingest import download_pdf, extract_text_from_pdf
        pdf_path = await asyncio.wait_for(
            download_pdf(f"https://export.arxiv.org/pdf/{arxiv_id}"),
            timeout=EVIDENCE_FETCH_TIMEOUT + 30,
        )
        if not pdf_path:
            return "", {"strategy": "arxiv_pdf", "ok": False, "chars": 0,
                        "detail": "download failed"}
        result = extract_text_from_pdf(pdf_path)
        text = (result.get("text", "") or "") if isinstance(result, dict) else str(result)
        ok = len(text.strip()) > 500
        return (text[:EVIDENCE_MAX_CHARS_PER_SOURCE] if ok else "",
                {"strategy": "arxiv_pdf", "ok": ok, "chars": len(text),
                 "detail": arxiv_id})
    except Exception as e:
        return "", {"strategy": "arxiv_pdf", "ok": False, "chars": 0,
                    "detail": f"{type(e).__name__}: {str(e)[:120]}"}


# ── Strategy 3: PMC full text via Europe PMC resolution + NCBI eutils ──

def _pmc_xml_to_text(xml_bytes: bytes) -> str:
    """Extract readable text from a PMC article-set XML (eutils efetch).

    Keeps title/abstract/section structure, drops references, tables and
    math markup (their text is unreadable when linearized).
    """
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError:
        return ""

    DROP_TAGS = {"ref-list", "table-wrap", "fig", "graphic", "inline-formula",
                 "disp-formula", "xref", "label", "fn-group", "contrib-group"}

    def local(tag: str) -> str:
        return tag.rsplit("}", 1)[-1]

    def inline_text(el) -> str:
        """Concatenated text of a subtree, skipping dropped subtrees but
        keeping tail punctuation (so 'x<ref>y</ref> z' → 'x z')."""
        parts: List[str] = []
        if el.text:
            parts.append(el.text)
        for c in el:
            if local(c.tag) in DROP_TAGS:
                if c.tail:
                    parts.append(c.tail)
                continue
            parts.append(inline_text(c))
        if el.tail:
            parts.append(el.tail)
        return "".join(parts)

    out: List[str] = []

    def emit(el):
        tag = local(el.tag)
        if tag in DROP_TAGS:
            return
        if tag == "sec":
            title_el = next((c for c in el if local(c.tag) == "title"), None)
            if title_el is not None:
                t = " ".join(inline_text(title_el).split())
                if t:
                    out.append(t.upper())
            for c in el:
                if c is not title_el:
                    emit(c)
            return
        if tag in {"p", "article-title", "title", "subtitle", "journal-title"}:
            t = " ".join(inline_text(el).split())
            if not t:
                return
            if tag in {"article-title", "subtitle", "journal-title"}:
                out.append(t)
            elif tag == "title":
                # titles outside secs (e.g. named abstracts) → headings
                out.append(t.upper())
            else:
                out.append(t)
            return
        # structural containers (front, body, abstract, article-meta, ...):
        for c in el:
            emit(c)

    emit(root)
    text = "\n\n".join(part.strip() for part in out if part.strip())
    text = re.sub(r"\n{3,}", "\n\n", text)
    return re.sub(r"[ \t]{2,}", " ", text).strip()


async def _resolve_pmc_id(client, doi: Optional[str]) -> Tuple[Optional[str], Dict]:
    """DOI → PMCID via Europe PMC search (live-probed: reliable from this
    egress; the sibling fullTextXML endpoint is not — eutils serves the body)."""
    if not doi:
        return None, {"strategy": "pmc_efetch", "ok": False, "chars": 0,
                      "detail": "no doi for pmc resolution"}
    try:
        r = await client.get(
            "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
            params={"query": f'DOI:"{doi}"', "format": "json", "resultType": "core"},
        )
        if r.status_code != 200:
            return None, {"strategy": "pmc_efetch", "ok": False, "chars": 0,
                          "detail": f"epmc search HTTP {r.status_code}"}
        results = (r.json().get("resultList") or {}).get("result") or []
        pmcid = next((h.get("pmcid") for h in results if h.get("pmcid")), None)
        if not pmcid:
            return None, {"strategy": "pmc_efetch", "ok": False, "chars": 0,
                          "detail": f"no PMCID for {doi}"}
        return pmcid, {}
    except Exception as e:
        return None, {"strategy": "pmc_efetch", "ok": False, "chars": 0,
                      "detail": f"epmc {type(e).__name__}: {str(e)[:100]}"}


async def _fetch_pmc_text(source: Dict, client, doi: Optional[str]) -> Tuple[str, Dict]:
    pmcid, fail = await _resolve_pmc_id(client, doi)
    if fail:
        return "", fail
    if not pmcid:
        return "", {"strategy": "pmc_efetch", "ok": False, "chars": 0,
                    "detail": "no PMCID"}
    try:
        numeric = pmcid.replace("PMC", "")
        r = await client.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
            params={"db": "pmc", "id": numeric, "retmode": "xml"},
        )
        if r.status_code != 200:
            return "", {"strategy": "pmc_efetch", "ok": False, "chars": 0,
                        "detail": f"efetch HTTP {r.status_code}"}
        text = _pmc_xml_to_text(r.content)
        ok = len(text) > 500
        return (text[:EVIDENCE_MAX_CHARS_PER_SOURCE] if ok else "",
                {"strategy": "pmc_efetch", "ok": ok, "chars": len(text),
                 "detail": pmcid})
    except Exception as e:
        return "", {"strategy": "pmc_efetch", "ok": False, "chars": 0,
                    "detail": f"efetch {type(e).__name__}: {str(e)[:100]}"}


# ── Strategy 4: Unpaywall OA PDF ──

async def _fetch_unpaywall_pdf_text(source: Dict, client, doi: Optional[str]) -> Tuple[str, Dict]:
    if not doi:
        return "", {"strategy": "unpaywall_pdf", "ok": False, "chars": 0,
                    "detail": "no doi"}
    try:
        email = os.getenv("CONTACT_EMAIL", "apex.research.agent@proton.me")
        r = await client.get(f"https://api.unpaywall.org/v2/{doi}",
                             params={"email": email})
        if r.status_code != 200:
            return "", {"strategy": "unpaywall_pdf", "ok": False, "chars": 0,
                        "detail": f"HTTP {r.status_code}"}
        best = r.json().get("best_oa_location") or {}
        pdf_url = best.get("url_for_pdf")
        if not pdf_url:
            return "", {"strategy": "unpaywall_pdf", "ok": False, "chars": 0,
                        "detail": "no url_for_pdf"}
        from ingest.pdf_ingest import download_pdf, extract_text_from_pdf
        pdf_path = await asyncio.wait_for(
            download_pdf(pdf_url), timeout=EVIDENCE_FETCH_TIMEOUT + 30)
        if not pdf_path:
            return "", {"strategy": "unpaywall_pdf", "ok": False, "chars": 0,
                        "detail": "download failed"}
        result = extract_text_from_pdf(pdf_path)
        text = (result.get("text", "") or "") if isinstance(result, dict) else str(result)
        ok = len(text.strip()) > 500
        return (text[:EVIDENCE_MAX_CHARS_PER_SOURCE] if ok else "",
                {"strategy": "unpaywall_pdf", "ok": ok, "chars": len(text),
                 "detail": pdf_url[:120]})
    except Exception as e:
        return "", {"strategy": "unpaywall_pdf", "ok": False, "chars": 0,
                    "detail": f"{type(e).__name__}: {str(e)[:120]}"}


# ── Strategy 5: OpenAlex abstract (abstract-level floor, not full text) ──

def _openalex_abstract(work: Dict) -> str:
    """Reconstruct the abstract from OpenAlex's inverted index."""
    inv = work.get("abstract_inverted_index")
    if not inv:
        return ""
    positions: List[Tuple[int, str]] = []
    for word, idxs in inv.items():
        for i in idxs:
            positions.append((i, word))
    positions.sort()
    return " ".join(w for _, w in positions).strip()


async def _fetch_openalex_text(source: Dict, client, doi: Optional[str]) -> Tuple[str, Dict]:
    try:
        if doi:
            r = await client.get(f"https://api.openalex.org/works/doi:{doi}")
        else:
            title = (source.get("title") or "").strip()
            if len(title) < 15:
                return "", {"strategy": "openalex_abstract", "ok": False,
                            "chars": 0, "detail": "no doi and title too short"}
            r = await client.get(
                "https://api.openalex.org/works",
                params={"filter": f"title.search:{title[:180]}", "per-page": 1},
            )
        if r.status_code != 200:
            return "", {"strategy": "openalex_abstract", "ok": False, "chars": 0,
                        "detail": f"HTTP {r.status_code}"}
        work = r.json()
        if work.get("results") is not None:
            work = (work.get("results") or [{}])[0]
        abstract = _openalex_abstract(work)
        ok = len(abstract) > 300
        return (abstract[:EVIDENCE_MAX_CHARS_PER_SOURCE] if ok else "",
                {"strategy": "openalex_abstract", "ok": ok, "chars": len(abstract),
                 "detail": f"abstract-level only (not full text)"})
    except Exception as e:
        return "", {"strategy": "openalex_abstract", "ok": False, "chars": 0,
                    "detail": f"{type(e).__name__}: {str(e)[:120]}"}


# ── Strategy 6/7: bridge page_reader + browser-UA scrape ──

async def _fetch_web_text(source: Dict, client_unused) -> Tuple[str, Dict]:
    """Full text for web/grey-literature records via the bridge page_reader.

    JS-rendered sites (Wix-hosted NGO reports, Squarespace news pages)
    defeat plain httpx scraping; the bridge's page_reader renders them.
    Only web records take this route, and an empty result falls through to
    the scrape chain — never a dead end.
    """
    if not _is_web_record(source):
        return "", {"strategy": "page_reader", "ok": False, "chars": 0,
                    "detail": "not a web record"}
    try:
        from tools.search_sources import fetch_page_text
        text = await asyncio.wait_for(
            fetch_page_text(source.get("url", "")),
            timeout=EVIDENCE_FETCH_TIMEOUT,
        )
        ok = bool(text) and len(text) > 500
        return (text[:EVIDENCE_MAX_CHARS_PER_SOURCE] if ok else "",
                {"strategy": "page_reader", "ok": ok, "chars": len(text or ""),
                 "detail": "bridge page_reader"})
    except Exception as e:
        return "", {"strategy": "page_reader", "ok": False, "chars": 0,
                    "detail": f"{type(e).__name__}: {str(e)[:120]}"}


async def _fetch_scraped_text(source: Dict, client_unused) -> Tuple[str, Dict]:
    """Fetch full text via the 3-tier scrape fallback (httpx → Jina → Firecrawl)."""
    url = source.get("url", "")
    if not url or not url.startswith("http"):
        return "", {"strategy": "scrape", "ok": False, "chars": 0, "detail": "no url"}
    try:
        from tools.live_scraper import _scrape_url_with_fallback
        result = await asyncio.wait_for(
            _scrape_url_with_fallback(url), timeout=EVIDENCE_FETCH_TIMEOUT)
        if result and result.success and result.markdown \
                and not result.markdown.startswith("[SCRAPE_FAILED"):
            ok = len(result.markdown) > 500
            return (result.markdown[:EVIDENCE_MAX_CHARS_PER_SOURCE] if ok else "",
                    {"strategy": "scrape", "ok": ok, "chars": len(result.markdown),
                     "detail": "3-tier scrape"})
        err = getattr(result, "error", None) or "empty result"
        return "", {"strategy": "scrape", "ok": False, "chars": 0, "detail": str(err)[:120]}
    except Exception as e:
        return "", {"strategy": "scrape", "ok": False, "chars": 0,
                    "detail": f"{type(e).__name__}: {str(e)[:120]}"}


async def fetch_source_fulltext(source: Dict) -> Tuple[str, List[Dict]]:
    """
    Fetch the best available text for a single source, trying every
    credential-free strategy. Returns (text, attempts).

    Strategy order reflects evidence quality:
      1. ctgov_api        — registry records: their entry IS the full text
      2. arxiv_pdf        — preprint full text
      3. pmc_efetch       — journal articles in PubMed Central (real full text)
      4. unpaywall_pdf    — OA PDF from the publisher/repository
      5. openalex_abstract— abstract-level floor (better than any snippet)
      6. page_reader      — JS-rendered grey literature (web records only)
      7. scrape           — 3-tier fallback, browser UA

    The longest successful result wins; abstract-level text is only used
    when no full-text route produced anything.
    """
    attempts: List[Dict] = []
    best_text = ""

    async def _try(strategy_fn, *args, record_even_if_empty: bool = True):
        nonlocal best_text
        try:
            text, diag = await asyncio.wait_for(
                strategy_fn(*args), timeout=EVIDENCE_FETCH_TIMEOUT + 45)
        except asyncio.TimeoutError:
            diag = {"strategy": getattr(strategy_fn, "__name__", "?"),
                    "ok": False, "chars": 0, "detail": "timeout"}
            text = ""
        except Exception as e:
            diag = {"strategy": getattr(strategy_fn, "__name__", "?"),
                    "ok": False, "chars": 0,
                    "detail": f"{type(e).__name__}: {str(e)[:100]}"}
        if record_even_if_empty or text:
            attempts.append(diag)
        if text and len(text) > len(best_text):
            best_text = text

    doi = _extract_doi(source)
    is_ct = _extract_nct_id(source) is not None
    is_web = _is_web_record(source)
    openalex_tried = False

    async with _http_client() as client:
        if is_ct:
            await _try(_fetch_ctgov_text, source, client)
        if _extract_arxiv_id(source):
            await _try(_fetch_arxiv_text, source, client)
        if doi:
            await _try(_fetch_pmc_text, source, client, doi)
            if not best_text:
                await _try(_fetch_unpaywall_pdf_text, source, client, doi)
        if not best_text:
            openalex_tried = True
            await _try(_fetch_openalex_text, source, client, doi)
        if not best_text and is_web:
            await _try(_fetch_web_text, source, client)
        if not best_text:
            await _try(_fetch_scraped_text, source, client)
        # Abstract-level upgrade when only a thin scrape succeeded
        if best_text and len(best_text) < 800 and doi and not openalex_tried:
            await _try(_fetch_openalex_text, source, client, doi)

    return best_text, attempts


class _HttpClient:
    """Shared httpx client context with browser headers (per-fetch-batch)."""

    def __init__(self):
        self._cm = None
        self.client = None

    async def __aenter__(self):
        import httpx
        self._cm = httpx.AsyncClient(
            timeout=30, follow_redirects=True, headers=BROWSER_HEADERS)
        self.client = await self._cm.__aenter__()
        return self.client

    async def __aexit__(self, *exc):
        return await self._cm.__aexit__(*exc)


def _http_client():
    return _HttpClient()


async def append_fetched_source(corpus: "EvidenceCorpus", source: Dict, idx: int) -> bool:
    """Fetch one source's full text and append it + its chunks to an EXISTING
    corpus (run11 finding: late canonical backfills entered AFTER corpus
    build, so their full text was never attempted — a cited source must get
    its fetch attempt whenever it entered the pool).

    Returns True when text was added.
    """
    url = source.get("url", "")
    text, attempts = await fetch_source_fulltext(source)
    corpus.attempts[url] = attempts
    route = next((a["strategy"] for a in attempts
                  if a.get("ok") and a["strategy"] in FULLTEXT_ROUTES), None)
    if not route:
        route = next((a["strategy"] for a in attempts if a.get("ok")), None)
    if route:
        corpus.fetch_routes[url] = route
    if not text or len(text) < 500:
        if url:
            corpus.failed_sources.append(url)
        return False
    corpus.fetched_sources.append(url)
    corpus.total_chars += len(text)
    from ingest.chunker import chunk_fixed_size
    chunks = chunk_fixed_size(
        text,
        chunk_size_tokens=EVIDENCE_CHUNK_TOKENS,
        overlap_pct=EVIDENCE_CHUNK_OVERLAP,
    )
    for c in chunks:
        corpus.chunks.append(EvidenceChunk(
            text=c.text,
            source_idx=idx + 1,  # 1-based [Sn] alignment
            source_url=url,
            source_title=source.get("title", ""),
            source_tier=source.get("tier", "UNV"),
            chunk_index=c.chunk_index,
        ))
    return True


async def build_evidence_corpus(
    sources: List[Dict],
    top_n: Optional[int] = None,
    web_slots: int = 2,
) -> EvidenceCorpus:
    """
    Build a full-text evidence corpus from the final source pool.

    Selection: rank by tier-adjusted score, take up to `top_n` (default:
    EVIDENCE_MAX_FETCH — run10 postmortem: capping at 8 left sources
    unfetched *silently*; every accepted source now gets attempted), fetch
    with bounded concurrency, record per-source diagnostics, chunk each
    text into ~400-token pieces, and tag every chunk with the source's
    1-BASED index in the caller's source list (so report citations [S{n}]
    match the final SOURCES section).

    Args:
        sources: The final, deduplicated, ranked source list
        top_n: How many sources to attempt (None → all, capped)
        web_slots: Reserved full-text fetches for web/grey-literature records
            outside the cut (kept for compatibility; the default attempts all)

    Returns:
        EvidenceCorpus (possibly empty — callers must handle gracefully)
    """
    corpus = EvidenceCorpus()
    if not sources:
        return corpus

    cap = EVIDENCE_MAX_FETCH if top_n is None else min(top_n, EVIDENCE_MAX_FETCH)
    ranked = sorted(
        range(len(sources)),
        key=lambda i: _source_score(sources[i]),
        reverse=True,
    )
    candidates = ranked[:cap]

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
                f"fetch slots outside the score cut"
            )

    sem = asyncio.Semaphore(EVIDENCE_FETCH_CONCURRENCY)

    async def _fetch_one(idx: int) -> tuple:
        async with sem:
            text, attempts = await fetch_source_fulltext(sources[idx])
        return idx, text, attempts

    results = await asyncio.gather(*[_fetch_one(i) for i in candidates],
                                   return_exceptions=True)

    from ingest.chunker import chunk_fixed_size

    for result in results:
        if isinstance(result, Exception):
            logger.warning(f"Evidence fetch task crashed: {result!r}")
            continue
        idx, text, attempts = result
        source = sources[idx]
        url = source.get("url", "")

        corpus.attempts[url] = attempts
        for diag in attempts:
            if diag.get("ok") and diag["strategy"] not in ("openalex_abstract",):
                corpus.fetch_routes[url] = diag["strategy"]
                break
            if diag.get("ok") and diag["strategy"] == "openalex_abstract" \
                    and url not in corpus.fetch_routes:
                corpus.fetch_routes[url] = "openalex_abstract"

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
        f"({len(corpus.failed_sources)} failed, {corpus.total_chars} chars) "
        f"routes={sorted(set(corpus.fetch_routes.values()))}"
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
    fetched_detail = [
        {"url": url, "route": corpus.fetch_routes.get(url, "unknown")}
        for url in corpus.fetched_sources
    ]
    failed_detail = [
        {
            "url": url,
            "strategies_tried": [a.get("strategy") for a in corpus.attempts.get(url, [])],
            "last_detail": (corpus.attempts.get(url) or [{}])[-1].get("detail", ""),
        }
        for url in corpus.failed_sources
    ]
    strict = {u: r for u, r in corpus.fetch_routes.items() if r in FULLTEXT_ROUTES}
    return {
        "chunks": len(corpus.chunks),
        "fetched_sources": len(corpus.fetched_sources),
        "failed_sources": len(corpus.failed_sources),
        "total_chars": corpus.total_chars,
        "semantic_scoring": corpus.used_semantic,
        "fetch_routes": dict(corpus.fetch_routes),
        # Strict full-text subset: openalex_abstract deliberately excluded —
        # the evidence-depth floor counts sources the system actually READ,
        # never external abstract metadata (run10's metric-inflation ban).
        "fulltext_routes": strict,
        "fulltext_sources_strict": len(strict),
        "abstract_only_routes": {
            u: r for u, r in corpus.fetch_routes.items()
            if r not in FULLTEXT_ROUTES
        },
        "fetched_detail": fetched_detail,
        "failed_detail": failed_detail,
    }


# ═══════════════════════════════════════════════════════════════
# BODY-LEVEL PROVISIONAL NOTICE (run10 verdict, deterministic)
# ═══════════════════════════════════════════════════════════════
#
# The run10 verdict: "If it can't fetch, the report should say 'claims rest
# on abstracts; treat as provisional' IN THE BODY, not just in the audit
# table at the bottom." This notice is rendered by Python from measured
# corpus state — an LLM pass cannot omit, soften, or forget it.


def _route_breakdown(ev: Dict) -> str:
    routes = ev.get("fetch_routes") or {}
    if not routes:
        return "none"
    counts: Dict[str, int] = {}
    for r in routes.values():
        counts[r] = counts.get(r, 0) + 1
    return "; ".join(f"{k}: {v}" for k, v in sorted(counts.items()))


def build_provisional_notice(fulltext_count: int, floor: int, n_sources: int,
                             ev_stats: Optional[Dict] = None) -> Optional[str]:
    """Return the body-level notice markdown when fulltext < floor, else None.

    `fulltext_count` must be the STRICT count (FULLTEXT_ROUTES only, final
    pool only) — abstract-level metadata never satisfies the floor.
    """
    if fulltext_count >= floor:
        return None
    ev = ev_stats or {}
    n_abstract = len(ev.get("abstract_only_routes") or {})
    routes = _route_breakdown(ev)
    fetched = ev.get("fetched_detail") or []
    fetched_lines = "\n".join(
        f"- Fetched ({d.get('route', 'unknown')}): {d.get('url', '')[:110]}"
        for d in fetched[:6]
    ) or "- (no source yielded text through any route)"
    failed = ev.get("failed_detail") or []
    failed_lines = "\n".join(
        f"- {d.get('url', '')[:80]} — tried [{', '.join(d.get('strategies_tried') or [])}], "
        f"last: {str(d.get('last_detail') or '')[:80]}"
        for d in failed[:6]
    )
    abstract_note = (
        f" A further {n_abstract} source(s) contributed abstract-level metadata only "
        f"(not counted toward the full-text floor)."
        if n_abstract else ""
    )
    return (
        "## EVIDENCE DEPTH NOTICE — PROVISIONAL STATUS\n\n"
        f"> **This report's claims rest on abstracts, titles, and registry summaries — "
        f"NOT on full text the system actually read.** Full text was retrieved for "
        f"{fulltext_count} of {n_sources} cited sources (floor: {floor}).{abstract_note} "
        f"Every substantive claim below is therefore provisional and abstract-level: "
        f"the system cannot distinguish what a source *says* from what its title "
        f"*implies* it says. Fetch routes that produced text: {routes}. "
        f"Do not treat any finding here as evidentially settled.\n\n"
        f"<details><summary>Full-text fetch audit (measured, not asserted)</summary>\n\n"
        f"{fetched_lines}\n"
        + (failed_lines + "\n" if failed_lines else "")
        + "\n</details>\n"
    )


def insert_body_notice(raw_report: str, notice: Optional[str]) -> str:
    """Insert the notice into the BODY — after the executive summary section.

    Body means before SOURCES/provenance appendices. If the structure is
    unrecognized, the notice is inserted directly after the H1 title so it
    can never end up buried at the bottom.
    """
    if not notice:
        return raw_report
    lines = raw_report.split("\n")

    # Find heading line indexes (## level)
    h2_idx = [i for i, ln in enumerate(lines) if re.match(r"^##\s+\S", ln)]
    insert_at = None
    for j, i in enumerate(h2_idx):
        if re.match(r"^##\s+(EXECUTIVE\s+SUMMARY|SUMMARY|ABSTRACT)", ln := lines[i], flags=re.I):
            # insert before the NEXT h2 after this one (or before SOURCES)
            nxt = [k for k in h2_idx if k > i]
            insert_at = nxt[0] if nxt else len(lines)
            break
    if insert_at is None:
        insert_at = (h2_idx[0] if h2_idx
                     else (1 if lines and lines[0].lstrip().startswith("# ") else 0))

    block = notice.rstrip("\n").split("\n")
    return "\n".join(lines[:insert_at] + block + ["", ""] + lines[insert_at:])
