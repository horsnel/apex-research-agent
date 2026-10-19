"""
Citation Integrity — post-generation grounding check for research reports.

A report is only trustworthy if:
1. Every numbered citation [S2] / [2] maps to a real source in the list
2. Every factual claim sentence carries a citation (or an epistemic tag)
3. Citation density is measurable so quality regressions are visible

This module is pure-stdlib and fully unit-testable (no I/O). The report
pipeline runs check_report_citations() after assembly and feeds the issues
back into the critique/revise pass.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


# Numbered citations: [S2], [2], [3,5], [S2, S3], [S2, S3, S5] — brackets
# REQUIRED; the S prefix may appear on the first element or on every element
NUMBERED_CITATION = re.compile(r'\[S?(\d{1,3}(?:\s*,\s*S?\d{1,3})*)\]')

# Tiered author citations: [Author 2024, P1], [domain.com, P2], [UNVERIFIED]
TIERED_CITATION = re.compile(r'\[[^\]]+,\s*(?:P[123]|UNV)\]|\[UNVERIFIED\]')

# Sentences that look like factual claims and require a citation
CLAIM_PATTERNS = [
    re.compile(r'\d+\.?\d*%'),
    re.compile(r'\$[\d,]+\.?\d*'),
    re.compile(r'\b(?:increased|decreased|improved|reduced|achieved|outperformed|demonstrated|showed|found)\b', re.I),
    re.compile(r'\b\d+(?:\.\d+)?x\b'),
    re.compile(r'\b(?:study|studies|trial|meta-analysis|survey|benchmark|experiment)\b', re.I),
]

EPISTEMIC_TAG = re.compile(r'\[(?:ESTABLISHED|TENTATIVE|ACTIVE_DEBATE|SPECULATIVE|UNVERIFIED)\]', re.I)


@dataclass
class CitationReport:
    """Result of the citation integrity check."""
    total_citations: int = 0
    unique_sources_cited: int = 0
    invalid_citations: List[str] = field(default_factory=list)   # e.g. "[S27]" with 20 sources
    uncited_claim_sentences: List[str] = field(default_factory=list)
    total_claim_sentences: int = 0
    citation_precision: float = 1.0   # valid / total numbered citations
    claim_coverage: float = 0.0       # cited claims / total claim sentences
    issues: List[str] = field(default_factory=list)

    @property
    def needs_revision(self) -> bool:
        return bool(self.invalid_citations) or (
            self.total_claim_sentences > 0 and self.claim_coverage < 0.6
        )


def _sentence_has_citation(sentence: str) -> bool:
    return bool(NUMBERED_CITATION.search(sentence) or TIERED_CITATION.search(sentence)
                or EPISTEMIC_TAG.search(sentence))


def extract_numbered_citations(text: str) -> List[int]:
    """Extract all source numbers referenced via [S2] / [S2, S3] / [2] / [3,5] citations."""
    numbers: List[int] = []
    for m in NUMBERED_CITATION.finditer(text):
        for part in m.group(1).split(","):
            part = part.strip().lstrip("Ss").strip()
            try:
                numbers.append(int(part))
            except ValueError:
                continue
    return numbers


def extract_tiered_citations(text: str) -> List[Tuple[str, str]]:
    """Extract (source_string, tier) pairs from [Author Year, P1]-style citations."""
    pairs = []
    for m in TIERED_CITATION.finditer(text):
        raw = m.group(0).strip("[]")
        if raw == "UNVERIFIED":
            pairs.append(("UNVERIFIED", "UNV"))
            continue
        source, _, tier = raw.rpartition(",")
        pairs.append((source.strip(), tier.strip()))
    return pairs


def check_report_citations(report_text: str, sources: List[Dict]) -> CitationReport:
    """
    Check a generated report against the actual source list.

    Args:
        report_text: Full markdown report (may include the SOURCES section —
            it is excluded from claim-sentence checks since it's bibliography)
        sources: The numbered source list used for generation (1-based [S1]..)

    Returns:
        CitationReport with precision/coverage metrics and issue list
    """
    report = CitationReport()

    # Split off the SOURCES bibliography so we don't count it as claims
    main_text = report_text
    bib_match = re.search(r'\n##\s*SOURCES\b.*\Z', report_text, re.DOTALL | re.IGNORECASE)
    if bib_match:
        main_text = report_text[:bib_match.start()]

    # 1. Numbered citation validity
    numbers = extract_numbered_citations(main_text)
    report.total_citations = len(numbers)
    n_sources = len(sources)
    invalid = sorted({n for n in numbers if n < 1 or n > n_sources})
    report.invalid_citations = [f"[S{n}]" for n in invalid]
    if report.total_citations > 0:
        report.citation_precision = 1.0 - (len(invalid) / max(len(set(numbers)), 1))
    report.unique_sources_cited = len({n for n in numbers if 1 <= n <= n_sources})

    # 2. Claim-sentence coverage
    sentences = re.split(r'(?<=[.!?])\s+', main_text)
    for sentence in sentences:
        s = sentence.strip()
        if len(s) < 40:  # skip headers / fragments
            continue
        is_claim = any(p.search(s) for p in CLAIM_PATTERNS)
        if not is_claim:
            continue
        report.total_claim_sentences += 1
        if not _sentence_has_citation(s):
            report.uncited_claim_sentences.append(s[:200])

    if report.total_claim_sentences > 0:
        cited = report.total_claim_sentences - len(report.uncited_claim_sentences)
        report.claim_coverage = round(cited / report.total_claim_sentences, 3)

    # 3. Build issue list for the critique/revise pass
    if report.invalid_citations:
        report.issues.append(
            f"Citations {', '.join(report.invalid_citations)} reference nonexistent sources — "
            f"only [S1] through [S{n_sources}] exist. Replace or remove them."
        )
    if report.uncited_claim_sentences:
        preview = "; ".join(report.uncited_claim_sentences[:3])
        report.issues.append(
            f"{len(report.uncited_claim_sentences)} factual claim sentences lack citations "
            f"(e.g. {preview}). Add [S<no>] citations from the provided sources or mark [UNVERIFIED]."
        )
    if report.total_citations >= 5 and report.unique_sources_cited < min(3, n_sources):
        report.issues.append(
            f"Only {report.unique_sources_cited} distinct sources cited across "
            f"{report.total_citations} citations — diversify the evidence base."
        )

    return report


def build_sources_section(sources: List[Dict]) -> str:
    """
    Build the SOURCES bibliography programmatically (deterministic — never
    let the LLM invent the source list). Numbering matches [S1]..[Sn] tags.
    """
    lines = ["## SOURCES"]
    for i, s in enumerate(sources, 1):
        title = (s.get("title") or "Untitled").strip()
        url = s.get("url", "")
        tier = s.get("tier", "UNV")
        year = ""
        pub = str(s.get("published_date") or "")[:4]
        if pub.isdigit():
            year = pub
        authors = s.get("authors", []) or []
        author_str = ""
        if authors:
            first = str(authors[0]).split(",")[0].strip()
            author_str = f"{first} et al. " if len(authors) > 1 else f"{first}. "
        cc = s.get("citation_count")
        cc_str = f" Cited {cc} times." if isinstance(cc, (int, float)) and cc else ""
        alt = s.get("also_found_in") or []
        alt_str = f" (also: {', '.join(alt[:2])})" if alt else ""
        lines.append(f"[S{i}] [{tier}] {author_str}{title}. {year}. {url}{cc_str}{alt_str}")
    return "\n".join(lines)


def prune_uncited_sources(report_text: str, sources: List[Dict]) -> Tuple[str, List[Dict], Dict]:
    """
    Premium bibliography hygiene: remove sources that are NEVER cited in the
    report body, then renumber the remaining [Sn] citations and rebuild the
    bibliography deterministically (no gaps, no dead entries).

    The SOURCES section is excluded from citation extraction so the
    bibliography's own [S1]..[Sn] labels don't count as "citations".

    Returns:
        (new_report_text, new_sources, stats) where stats = {
            "pruned": [old 1-based indices removed],
            "before": n_sources_before,
            "after": n_sources_after,
            "renumbered": True/False,
        }
        If every source is cited, returns (report_text, sources, {"pruned": []...}).
    """
    stats = {"pruned": [], "before": len(sources), "after": len(sources), "renumbered": False}
    if not sources or not report_text:
        return report_text, sources, stats

    # Split body from the deterministic bibliography (built by build_sources_section)
    m = re.search(r"\n## SOURCES\b", report_text)
    if not m:
        return report_text, sources, stats
    body = report_text[:m.start()]
    tail = report_text[m.start():]

    cited = sorted({n for n in extract_numbered_citations(body) if 1 <= n <= len(sources)})
    dead = [i for i in range(1, len(sources) + 1) if i not in set(cited)]
    stats["pruned"] = dead

    if not dead:
        return report_text, sources, stats

    # Old index -> new index (dense renumbering, order preserved)
    mapping: Dict[int, int] = {}
    for new_idx, old_idx in enumerate(cited, start=1):
        mapping[old_idx] = new_idx

    def _renumber(m_: "re.Match") -> str:
        parts = []
        for part in m_.group(1).split(","):
            p = part.strip()
            num = re.sub(r"^S", "", p).strip()
            try:
                old = int(num)
            except ValueError:
                parts.append(p)
                continue
            parts.append(str(mapping.get(old, old)))
        return f"[S{', '.join(parts)}]"

    new_body = re.sub(r'\[S?(\d{1,3}(?:\s*,\s*S?\d{1,3})*)\]', _renumber, body)

    new_sources = [sources[old - 1] for old in cited]
    new_tail = "\n\n" + build_sources_section(new_sources)

    stats["after"] = len(new_sources)
    stats["renumbered"] = True
    return new_body.rstrip() + new_tail, new_sources, stats


def rebuild_bibliography(report_text: str, sources: List[Dict]) -> str:
    """
    Replace whatever SOURCES section exists (e.g. an LLM-regenerated one from
    a revision pass) with the deterministic bibliography for `sources`.
    Keeps the body untouched — numbering must already match [S1]..[Sn].
    """
    m = re.search(r"\n## SOURCES\b", report_text)
    canonical = "\n\n" + build_sources_section(sources)
    if not m:
        return report_text.rstrip() + canonical
    return report_text[:m.start()] + canonical


# ── Claim↔source alignment scan ─────────────────────────────────────
# Live failure: "Egger's regression / trim-and-fill [S4]" cited a marine-
# ecology supplement whose snippet was just a DOI string. Citation NUMBERS
# were valid (precision 1.0) — but the source did not support the claim.
# This scan is the deterministic first-order detector: a cited source that
# shares ZERO content words with the claim sentence is flagged for the
# critique LLM to adjudicate (snippets can be terse even when support is real).

_ALIGNMENT_STOPWORDS = {
    "this", "that", "these", "those", "with", "from", "have", "has", "had",
    "been", "were", "their", "them", "they", "which", "while", "when", "where",
    "such", "also", "more", "most", "than", "then", "into", "over", "under",
    "between", "among", "both", "each", "other", "some", "only", "very",
    "can", "may", "will", "would", "could", "should", "must", "shall",
    "there", "here", "thus", "hence", "however", "although", "because",
    "about", "after", "before", "during", "through", "across", "against",
    "within", "without", "upon", "does", "done", "using", "used", "based",
    "study", "studies", "research", "analysis", "results", "result",
    "paper", "report", "section", "table", "figure", "approach", "method",
    "methods", "data", "evidence", "findings", "finding", "shows", "shown",
    "including", "include", "includes", "provide", "provides", "given",
}


def _alignment_content_words(text: str) -> set:
    """Content words for alignment (len>=4, stopwords removed)."""
    words = re.findall(r"[a-z0-9]{4,}", (text or "").lower())
    return {w for w in words if w not in _ALIGNMENT_STOPWORDS}


def scan_claim_source_alignment(
    report_text: str,
    sources: List[Dict],
    max_flags: int = 8,
) -> List[Dict]:
    """
    Deterministic claim↔source alignment check.

    For each sentence carrying a numbered citation, compute the shared
    content-word set between the sentence and each cited source's
    title+snippet. A cited source with ZERO shared content words is flagged
    as a suspected mis-attribution for the critique pass to verify.

    Returns a list of flags:
        {"source": n, "excerpt": "<sentence up to 160 chars>"}
    """
    flags: List[Dict] = []

    main_text = report_text
    bib_match = re.search(r'\n##\s*SOURCES\b.*\Z', report_text, re.DOTALL | re.IGNORECASE)
    if bib_match:
        main_text = report_text[:bib_match.start()]

    # Drop markdown tables and headings — alignment only applies to prose
    lines = [ln for ln in main_text.split("\n") if not ln.strip().startswith(("#", "|"))]
    prose = "\n".join(lines)
    sentences = re.split(r"(?<=[.!?])\s+", prose)

    for sentence in sentences:
        if len(sentence.split()) < 8 or len(flags) >= max_flags:
            continue
        cited = sorted({n for n in extract_numbered_citations(sentence) if 1 <= n <= len(sources)})
        if not cited:
            continue
        sent_words = _alignment_content_words(sentence)
        if not sent_words:
            continue
        for n in cited:
            src = sources[n - 1]
            src_text = f"{src.get('title') or ''} {src.get('snippet') or ''}"
            # A snippet that is only a DOI/URL carries no checkable content —
            # skip rather than false-positive on terse metadata.
            if len(re.sub(r"(doi|https?://\S+|[\s.:])", "", src_text.lower())) < 25:
                continue
            if not (sent_words & _alignment_content_words(src_text)):
                flags.append({"source": n, "excerpt": sentence.strip()[:160]})
                break  # one flag per sentence is enough for adjudication

    return flags
