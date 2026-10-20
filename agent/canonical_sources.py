"""
Canonical-source backfill — put the actual primary literature in the pool.

THE FAILURE THIS FIXES (live, 4 consecutive runs):
The pipeline retrieved papers that mention meta-analysis keywords but never
the canonical methodology papers the topic turns on. The report said
"Egger's test" and "trim-and-fill" while citing Wikipedia and a federated-
learning paper — because retrieval is keyword-driven, and keyword retrieval
ranks *containing the words* far above *being the answer*.

Design (no new hallucination surface):
1. REGISTRY — a small, curated set of canonical methodology / empirical
   papers. Every DOI was verified live against api.crossref.org at build
   time (title + year + type + identity), because blind bibliographic DOI
   guesses resolve to unrelated papers (verified: two candidate DOIs
   silently resolved to a neuroblastoma study and a CONSORT paper).
2. DETECT — pure-regex triggers over the query/variants/source-pool text
   (pre-draft) or the drafted report (post-draft). Method-name triggers
   ("trim-and-fill", "Egger's test") are high-precision; topic compounds
   ("meta-analysis" AND "unpublished/negative") cover queries that never
   name the methods.
3. BACKFILL — fetch each triggered canonical paper's real metadata from
   Crossref BY DOI (not by fuzzy search), validate the returned title
   against the registry's expected title, and append to the source pool.
   Backfilled sources are ordinary pool members: they flow through NLI,
   evidence fetching (Unpaywall uses DOIs), drafting, and the deterministic
   bibliography. If the fetch fails or the title mismatches, the source is
   skipped — the pipeline degrades honestly.
4. SCAN — after drafting, a deterministic scan flags (a) method names with
   no primary source in the pool (CANONICAL_SOURCE_MISSING) and (b) named
   methods cited to the wrong source (CANONICAL_METHOD_UNCITED), so the
   revision pass must cite the primary literature or soften the claim.

This module is stdlib-only at import time; the only network call is the
lazy Crossref fetch (injectable for tests).
"""

import asyncio
import logging
import re
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────
# Registry
# ────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class CanonicalEntry:
    """One verified canonical paper.

    identity (entry_id/doi/expected_title/expected_year) was verified live
    against Crossref before being hard-coded. trigger_any / trigger_all are
    case-insensitive regex fragments; an entry is triggered when
    (any pattern matches) OR (every group of trigger_all matches).
    method_patterns — strict method-name regexes used by the DRAFT scan
    (a draft that names the method must have its primary source).
    """
    entry_id: str
    doi: str
    expected_title: str
    expected_year: int
    venue: str
    priority: int                      # lower = injected first
    label: str                         # human-readable method/topic label
    why: str
    trigger_any: Tuple[str, ...] = ()
    trigger_all: Tuple[Tuple[str, ...], ...] = ()
    method_patterns: Tuple[str, ...] = ()


_TOPIC_COMPOUND: Tuple[Tuple[str, ...], ...] = (
    (r"meta[- ]analy|systematic review|research synthesis",),
    (r"publication bias|unpublish|unreport|negative[- ]result|missing (?:result|evidence)|never (?:made it |)public",),
)

REGISTRY: Tuple[CanonicalEntry, ...] = (
    # ── Method originals (the named procedures themselves) ──
    CanonicalEntry(
        entry_id="egger1997",
        doi="10.1136/bmj.315.7109.629",
        expected_title="Bias in meta-analysis detected by a simple, graphical test",
        expected_year=1997,
        venue="BMJ",
        priority=1,
        label="Egger's test / funnel-plot asymmetry regression (Egger et al. 1997)",
        why="the original paper for the Egger regression test of funnel-plot asymmetry",
        trigger_any=(r"egger(?:'s)? (?:test|regression)", r"funnel[- ]plot asymmetry", r"small[- ]study effects?"),
        trigger_all=_TOPIC_COMPOUND,
        method_patterns=(r"egger(?:'s)? (?:test|regression)", r"egger(?:'s)?"),
    ),
    CanonicalEntry(
        entry_id="duval2000",
        doi="10.1111/j.0006-341X.2000.00455.x",
        expected_title="Trim and Fill: A Simple Funnel-Plot-Based Method of Testing and Adjusting for Publication Bias in Meta-Analysis",
        expected_year=2000,
        venue="Biometrics",
        priority=2,
        label="trim-and-fill (Duval & Tweedie 2000)",
        why="the original paper for the trim-and-fill publication-bias adjustment",
        trigger_any=(r"trim[ -]and[ -]fill", r"trim and fill"),
        trigger_all=_TOPIC_COMPOUND,
        method_patterns=(r"trim[ -]and[ -]fill", r"trim and fill"),
    ),
    CanonicalEntry(
        entry_id="rosenthal1979",
        doi="10.1037/0033-2909.86.3.638",
        expected_title="The file drawer problem and tolerance for null results",
        expected_year=1979,
        venue="Psychological Bulletin",
        priority=4,
        label="file-drawer problem / fail-safe N (Rosenthal 1979)",
        why="the original paper for the file-drawer problem and fail-safe N",
        trigger_any=(r"file[- ]drawer", r"fail[- ]safe n\b", r"failsafe n\b"),
        trigger_all=_TOPIC_COMPOUND,
        method_patterns=(r"file[- ]drawer", r"fail[- ]safe n\b", r"failsafe n\b"),
    ),
    CanonicalEntry(
        entry_id="cochrane_robme",
        doi="10.1002/9781119536604.ch13",
        expected_title="Assessing risk of bias due to missing results in a synthesis",
        expected_year=2019,
        venue="Cochrane Handbook for Systematic Reviews of Interventions, chapter 13",
        priority=3,
        label="risk of bias due to missing results / ROB-ME (Cochrane Handbook ch. 13)",
        why="the Cochrane Handbook chapter defining ROB-ME, the standard framework for assessing (not solving) missing-evidence risk",
        trigger_any=(r"rob[- ]?me\b", r"bias due to missing", r"missing (?:results|evidence)", r"risk of bias (?:due to|from) missing"),
        trigger_all=_TOPIC_COMPOUND,
        method_patterns=(r"rob[- ]?me\b", r"bias due to missing (?:results|evidence)"),
    ),
    CanonicalEntry(
        entry_id="sterne2011",
        doi="10.1136/bmj.d4002",
        expected_title="Recommendations for examining and interpreting funnel plot asymmetry in meta-analyses of randomised controlled trials",
        expected_year=2011,
        venue="BMJ",
        priority=9,
        label="funnel-plot asymmetry assessment guidance (Sterne et al. 2011)",
        why="the standard recommendations for examining funnel-plot asymmetry",
        trigger_any=(r"funnel[- ]plot",),
        trigger_all=(),
        method_patterns=(r"funnel[- ]plot",),
    ),
    # ── Empirical backbone (how much evidence is delayed/withheld) ──
    CanonicalEntry(
        entry_id="song2000",
        doi="10.1016/S0895-4356(00)00242-0",
        expected_title="Publication and related bias in meta-analysis",
        expected_year=2000,
        venue="Journal of Clinical Epidemiology",
        priority=5,
        label="review of publication-bias evidence and detection methods (Sterne, Gavaghan & Egger 2000)",
        why="the standard review of publication-bias evidence and the power of detection methods",
        trigger_any=(r"publication bias", r"publication and related bias", r"biased (?:publication|reporting)"),
        trigger_all=_TOPIC_COMPOUND,
        method_patterns=(),
    ),
    CanonicalEntry(
        entry_id="stern1997",
        doi="10.1136/bmj.315.7109.640",
        expected_title="Publication bias: evidence of delayed publication in a cohort study of clinical research projects",
        expected_year=1997,
        venue="BMJ",
        priority=6,
        label="cohort evidence of delayed/withheld publication (Stern & Simes 1997)",
        why="cohort evidence that studies with negative results are published later or not at all",
        trigger_any=(r"publication bias", r"delayed publication", r"unpublished (?:trials?|studies?|results?|data)", r"unreported (?:trials?|studies?|results?)"),
        trigger_all=(),
        method_patterns=(),
    ),
    CanonicalEntry(
        entry_id="ioannidis1998",
        doi="10.1001/jama.279.4.281",
        expected_title="Effect of the Statistical Significance of Results on the Time to Completion and Publication of Randomized",
        expected_year=1998,
        venue="JAMA",
        priority=8,
        label="statistical significance vs time-to-publication cohort (Ioannidis 1998)",
        why="cohort evidence that trial significance drives completion and publication timing",
        trigger_any=(r"publication bias", r"time to publication", r"unpublished (?:trials?|studies?)", r"unreported (?:trials?|studies?)"),
        trigger_all=(),
        method_patterns=(),
    ),
    CanonicalEntry(
        entry_id="chan2004",
        doi="10.1001/jama.291.20.2457",
        expected_title="Empirical Evidence for Selective Reporting of Outcomes in Randomized Trials",
        expected_year=2004,
        venue="JAMA",
        priority=7,
        label="outcome-selective reporting cohort (Chan et al. 2004)",
        why="protocol-vs-publication evidence of selective outcome reporting in randomized trials",
        trigger_any=(r"outcome reporting bias", r"selective reporting", r"reporting bias", r"unreported (?:outcomes?|trials?|results?)"),
        trigger_all=(),
        method_patterns=(),
    ),
    CanonicalEntry(
        entry_id="mcgauran2010",
        doi="10.1186/1745-6215-11-37",
        expected_title="Reporting bias in medical research - a narrative review",
        expected_year=2010,
        venue="Trials",
        priority=10,
        label="narrative review of reporting-bias landscape (McGauran et al. 2010)",
        why="a narrative review mapping the full reporting-bias landscape",
        trigger_any=(r"reporting bias", r"publication bias", r"unpublished", r"unreported"),
        trigger_all=(),
        method_patterns=(),
    ),
)

# Entries whose *method names* must never be carried by secondary sources
# alone — the draft scan checks exactly these.
_METHOD_ENTRIES: Tuple[CanonicalEntry, ...] = tuple(
    e for e in REGISTRY if e.method_patterns
)


# ────────────────────────────────────────────────────────────────────
# Detection + identity helpers (pure, testable)
# ────────────────────────────────────────────────────────────────────

_TITLE_SPLIT_RE = re.compile(r"[^a-z0-9]+")
_DOI_PREFIX_RE = re.compile(r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)", re.I)


def normalize_doi(doi: str) -> str:
    return _DOI_PREFIX_RE.sub("", (doi or "").strip()).lower()


def _title_tokens(title: str) -> List[str]:
    # Normalize unicode hyphens/quotes Crossref emits (U+2010 etc.) via the
    # non-alnum split; drop tiny function words so containment is content-based.
    tokens = [t for t in _TITLE_SPLIT_RE.split((title or "").lower()) if len(t) > 2]
    return [t for t in tokens if t not in {"the", "and", "for", "of", "in", "a", "an", "to", "on"}]


def titles_match(fetched_title: str, expected_title: str, min_containment: float = 0.7) -> bool:
    """True when the fetched title contains the expected title's content
    tokens (extra fetched tokens — subtitles, truncation differences — OK;
    missing expected tokens are NOT ok). This is the anti-hallucination
    guard: a DOI that resolves to a different paper is rejected."""
    expected = _title_tokens(expected_title)
    if not expected:
        return False
    fetched = set(_title_tokens(fetched_title))
    hits = sum(1 for t in expected if t in fetched)
    return (hits / len(expected)) >= min_containment


def detect_canonical_entries(text: str) -> List[CanonicalEntry]:
    """Registry entries triggered by `text` (any-pattern OR all-compound),
    ordered by priority."""
    if not text:
        return []
    lowered = text.lower()
    triggered: List[CanonicalEntry] = []
    for entry in REGISTRY:
        any_hit = any(re.search(p, lowered) for p in entry.trigger_any)
        # bool(trigger_all) guards against vacuous all() over empty groups —
        # entries without a compound trigger must NEVER match on all() alone.
        all_hit = bool(entry.trigger_all) and all(
            any(re.search(g, lowered) for g in group)
            for group in entry.trigger_all
        )
        if any_hit or all_hit:
            triggered.append(entry)
    return sorted(triggered, key=lambda e: e.priority)


def _entry_in_pool(entry: CanonicalEntry, sources: List[Dict]) -> bool:
    """Is this canonical paper already in the pool? (DOI identity, then
    title containment — pool entries may lack a doi key.)"""
    want_doi = normalize_doi(entry.doi)
    for s in sources:
        if s.get("doi") and normalize_doi(str(s["doi"])) == want_doi:
            return True
        if s.get("canonical_id") == entry.entry_id:
            return True
        if titles_match(s.get("title", ""), entry.expected_title):
            return True
    return False


def _pool_index_for_entry(entry: CanonicalEntry, sources: List[Dict]) -> Optional[int]:
    """1-based index of the canonical source in the pool, or None."""
    want_doi = normalize_doi(entry.doi)
    for i, s in enumerate(sources, 1):
        if s.get("canonical_id") == entry.entry_id:
            return i
        if s.get("doi") and normalize_doi(str(s["doi"])) == want_doi:
            return i
        if titles_match(s.get("title", ""), entry.expected_title):
            return i
    return None


# ────────────────────────────────────────────────────────────────────
# Backfill (the only network path — injectable for tests)
# ────────────────────────────────────────────────────────────────────

Fetcher = Callable[[str], Awaitable[Optional[object]]]  # doi -> SearchResult-like | None


def _source_dict_from_result(entry: CanonicalEntry, result: object) -> Optional[Dict]:
    """Validate a fetched SearchResult against the registry entry and shape
    it into the pipeline's source dict. Returns None on identity mismatch."""
    fetched_title = (getattr(result, "title", "") or "").strip()
    if not titles_match(fetched_title, entry.expected_title):
        logger.warning(
            f"Canonical backfill REJECTED {entry.entry_id}: Crossref returned "
            f"title {fetched_title[:80]!r}, expected {entry.expected_title[:80]!r}"
        )
        return None
    year_raw = getattr(result, "published_date", None)
    try:
        year_val = int(str(year_raw)[:4])
    except (TypeError, ValueError):
        year_val = None
    if year_val is not None and abs(year_val - entry.expected_year) > 1:
        logger.warning(
            f"Canonical backfill REJECTED {entry.entry_id}: year {year_val} "
            f"vs expected {entry.expected_year}"
        )
        return None

    abstract = (getattr(result, "snippet", "") or "").strip()
    snippet = abstract if abstract else f"Canonical {entry.venue} reference for {entry.label}."
    return {
        "url": f"https://doi.org/{entry.doi}",
        "title": fetched_title,
        "snippet": snippet,
        "tier": "P1",
        "source_name": "crossref-canonical",
        "category": "SourceCategory.ACADEMIC",
        "published_date": str(year_val) if year_val is not None else str(entry.expected_year),
        "authors": list(getattr(result, "authors", None) or []),
        "citation_count": getattr(result, "citation_count", None),
        "doi": normalize_doi(entry.doi),
        "canonical_id": entry.entry_id,
        "backfilled": True,
        "why": entry.why,
        # Canonical-registry attestation: these papers enter by VERIFIED DOI
        # identity (the hand-curated methodological registry), not through
        # the embedding pipeline — run11 finding: they carry no topical
        # annotation and the every-cited-source-topically-ok gate reads that
        # as an unscreened citation. They are attested, screened-exempt, and
        # excluded from the embedding precision denominator like operator
        # seeds. Disclosed in the screening audit either way.
        "extra": {
            "attested": True,
            "topical_ok": True,
            "topical_attested": True,
            "topical_justification": (
                f"canonical primary source (identity verified by DOI: {entry.doi}) — "
                f"automated topical screening exempt"
            ),
        },
    }


async def ensure_canonical_sources(
    signal_text: str,
    sources: List[Dict],
    max_add: int = 5,
    fetcher: Optional[Fetcher] = None,
    force_entries: Optional[Tuple[CanonicalEntry, ...]] = None,
) -> Tuple[List[Dict], List[Dict]]:
    """Detect triggered canonical papers missing from the pool, fetch their
    real metadata from Crossref by DOI, validate identity, and append them.

    force_entries bypasses trigger detection and fetches exactly those
    entries (used by the late draft backfill, where asking for Sterne 2011
    must not priority-match Egger 1997 via a shared trigger word).

    Mutates `sources` in place AND returns (sources, added) where added is a
    list of {"entry_id", "doi", "title", "index", "label"} for logging and
    critique messaging. Network failures and identity mismatches are
    skipped silently (honest degradation — no invented metadata).
    """
    added: List[Dict] = []
    if force_entries is None and not signal_text:
        return sources, added

    if fetcher is None:
        from tools.search_sources import fetch_crossref_by_doi
        fetcher = fetch_crossref_by_doi

    if force_entries is not None:
        candidates = [e for e in force_entries if not _entry_in_pool(e, sources)]
    else:
        candidates = [
            e for e in detect_canonical_entries(signal_text)
            if not _entry_in_pool(e, sources)
        ][:max(0, max_add)]
    if not candidates:
        return sources, added

    results = await asyncio.gather(
        *(fetcher(e.doi) for e in candidates), return_exceptions=True
    )

    next_index = len(sources) + 1
    for entry, res in zip(candidates, results):
        if isinstance(res, BaseException) or res is None:
            if isinstance(res, BaseException):
                logger.warning(f"Canonical backfill fetch failed for {entry.entry_id}: {res!r}")
            continue
        shaped = _source_dict_from_result(entry, res)
        if shaped is None:
            continue
        sources.append(shaped)
        added.append({
            "entry_id": entry.entry_id,
            "doi": entry.doi,
            "title": shaped["title"],
            "index": next_index,
            "label": entry.label,
        })
        next_index += 1

    if added:
        logger.info(f"Canonical backfill appended: {[a['entry_id'] for a in added]}")
    return sources, added


# ────────────────────────────────────────────────────────────────────
# Draft-time scan — method names must stand on primary literature
# ────────────────────────────────────────────────────────────────────

def _report_body(text: str) -> str:
    """Report text minus the SOURCES bibliography, markdown tables and
    headings — mentions inside the bibliography or a table don't count."""
    m = re.search(r"\n##\s*SOURCES\b.*\Z", text or "", re.DOTALL | re.IGNORECASE)
    body = text[: m.start()] if m else (text or "")
    lines = [ln for ln in body.split("\n") if not ln.strip().startswith(("#", "|"))]
    return "\n".join(lines)


# A sentence that attributes a procedure to a source (vs. merely mentioning it)
_ATTRIBUTION_RE = re.compile(
    r"\b(test|statistic|regression|method|procedure|adjust(?:s|ed|ment)?|"
    r"detect(?:s|ed|ion)?|assess(?:s|ed|ment)?|estimat(?:es?|ed|ion)|"
    r"proposed|introduced|developed|quantif(?:y|ies|ied)|correct(?:s|ed|ion))\b",
    re.I,
)


def scan_canonical_method_usage(
    report_text: str,
    sources: List[Dict],
    max_issues: int = 3,
) -> List[Dict]:
    """Deterministic scan: named methods must stand on their primary source.

    Returns issue dicts shaped like LLM-critique issues:
        {"type": "CANONICAL_SOURCE_MISSING" | "CANONICAL_METHOD_UNCITED",
         "detail": ..., "fix": ...}
    so the critique pass can merge them with LLM issues verbatim.
    """
    from agent.citation_integrity import extract_numbered_citations

    issues: List[Dict] = []
    body = _report_body(report_text)
    if not body:
        return issues

    for entry in _METHOD_ENTRIES:
        if len(issues) >= max_issues:
            break
        pattern = re.compile("|".join(entry.method_patterns), re.I)
        mention_sentences = [s for s in re.split(r"(?<=[.!?])\s+", body) if pattern.search(s)]
        if not mention_sentences:
            continue

        pool_idx = _pool_index_for_entry(entry, sources)
        if pool_idx is None:
            issues.append({
                "type": "CANONICAL_SOURCE_MISSING",
                "detail": (
                    f"The report names {entry.label}, but no primary source for it "
                    f"exists in the source list."
                ),
                "fix": (
                    f"Either remove the named method or mark those sentences "
                    f"[UNVERIFIED]. Do not attribute a named method to an "
                    f"encyclopedia, a keyword-adjacent paper, or a snippet that "
                    f"does not describe it."
                ),
            })
            continue

        # Canonical source IS in the pool — it must be the one cited at the mention.
        bad_example = None
        for sentence in mention_sentences:
            cited = [n for n in extract_numbered_citations(sentence) if 1 <= n <= len(sources)]
            if not cited:
                bad_example = (sentence, "uncited")
                break
            if pool_idx not in cited and _ATTRIBUTION_RE.search(sentence):
                bad_example = (sentence, "wrong-source")
                break
        if bad_example:
            sentence, kind = bad_example
            detail = (
                f"Named method {entry.label} is described but its primary source "
                f"[S{pool_idx}] is not cited there"
                if kind == "wrong-source"
                else f"Named method {entry.label} appears without any citation"
            )
            issues.append({
                "type": "CANONICAL_METHOD_UNCITED",
                "detail": f"{detail}. Sentence: {sentence.strip()[:160]}",
                "fix": (
                    f"Cite [S{pool_idx}] ({entry.label}) at that sentence, or "
                    f"restate the claim generically without naming the method."
                ),
            })

    return issues


async def late_backfill_for_draft(
    report_text: str,
    sources: List[Dict],
    fetcher: Optional[Fetcher] = None,
) -> Tuple[List[str], List[Dict]]:
    """Post-draft safety net: if the DRAFT names a canonical method whose
    primary source is absent from the pool, fetch + append it now (in place,
    index n+1 — existing [Sn] numbering is untouched) and return critique
    issues telling the revision to cite the new source.

    Returns (issue_strings, newly_added) — issue_strings use the same
    "[TYPE] detail → fix" format as LLM issues.
    """
    added: List[Dict] = []
    issues: List[str] = []
    body = _report_body(report_text)
    if not body:
        return issues, added

    missing: List[CanonicalEntry] = []
    for entry in _METHOD_ENTRIES:
        if not any(re.search(p, body) for p in entry.method_patterns):
            continue
        if _entry_in_pool(entry, sources):
            continue
        missing.append(entry)

    if not missing:
        return issues, added

    _, added = await ensure_canonical_sources(
        "",
        sources,
        max_add=len(missing),
        fetcher=fetcher,
        force_entries=tuple(missing),  # exact fetches — no trigger re-matching
    )
    got_ids = {a["entry_id"] for a in added}
    for entry in missing:
        if entry.entry_id in got_ids:
            idx = _pool_index_for_entry(entry, sources)
            issues.append(
                f"[CANONICAL_SOURCE_ADDED] The draft names {entry.label} but no "
                f"primary source was in the pool. Its actual paper was just "
                f"retrieved from Crossref and appended as [S{idx}]. "
                f"→ Cite [S{idx}] at every sentence describing this method, or "
                f"remove the method name entirely."
            )
        else:
            issues.append(
                f"[CANONICAL_SOURCE_MISSING] The draft names {entry.label} but its "
                f"primary source could not be retrieved. "
                f"→ Remove the method name or mark those sentences [UNVERIFIED]; "
                f"do not attribute it to an unrelated or secondary source."
            )
    return issues, added
