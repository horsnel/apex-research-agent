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


# ── Generic-academic vocabulary (padding detection) ──
# Words that appear in ANY methods paper's title/abstract regardless of
# subject. Live run8 failure: the query "Meta-analysis of unpublished
# negative-result trials ... public databases" shares {meta, analysis,
# public, database} with "Meta-analysis of hypoxic transcriptomes from
# public databases" (RNA-seq) and "Federated Learning in Distributed Medical
# Databases" (brain imaging) — zero overlap with the DISCRIMINATIVE core
# (unpublished/negative/result/trial) — yet both passed the relevance gate
# and got cited, one even rationalized into the scope argument. Core overlap
# that reduces to generic terms is NOT topical purchase.
GENERIC_ACADEMIC_TERMS = {
    "meta", "analysis", "analyses", "systematic", "review", "scoping",
    "study", "studies", "research", "evidence", "public", "database",
    "databases", "data", "method", "methods", "methodological",
    "methodology", "literature", "survey", "overview", "report",
    "reports", "clinical", "health", "medical", "medicine", "science",
    "scientific", "paper", "papers", "approach", "approaches", "framework",
    "based", "using", "toward", "towards", "large", "scale", "real",
    "world", "new", "novel", "comparative",
}


def discriminative_vocab(core_vocab: set) -> set:
    """Core-question words that actually distinguish topic from non-topic:
    core vocabulary minus generic academic terms."""
    return {w for w in (core_vocab or set()) if w not in GENERIC_ACADEMIC_TERMS}


def shared_terms(source: Dict, vocab: set) -> set:
    """Which of `vocab` appear in the source's title+snippet (for audits)."""
    text = f"{(source.get('title') or '')} {(source.get('snippet') or '')}".lower()
    return {w for w in vocab if w in text}


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


# ── Concept-boundary mismatch scan ──────────────────────────────────
# Live failure: Silva et al. (2018), a privacy-preserving federated-learning
# paper, was cited 4 runs in a row for "accessing unpublished trials".
# Federated learning makes EXISTING institutionally-held data analyzable
# across sites; it says nothing about trials that were never published or
# registered. Zero-overlap alignment can't catch this — the words "trials"
# and "data" overlap — so the mismatch is CONCEPTUAL and needs a boundary
# table: (what the source is about) × (what the claim presupposes).

_CONCEPT_BOUNDARIES = [
    {
        "name": "existing-private-data vs never-published-data",
        # Source side: >=2 distinct markers ⇒ the source is about analyzing
        # data that exists but is held privately / across institutions.
        "source_markers": (
            r"federated", r"privacy[- ]preserving", r"privacy", r"decentralis|decentraliz",
            r"cross[- ]silo", r"multi(?:-| )[ct]", r"secure (?:computation|aggregation|multiparty)",
            r"patient[- ](?:level|record)", r"electronic health record",
        ),
        # Claim side: the sentence presupposes data that was never made public.
        "claim_markers": (
            r"unpublished", r"unregistered", r"never[- ](?:report|publish|registr)",
            r"never made (?:it |)public", r"unreport", r"missing (?:trial|study|data)",
            r"not (?:in any|in public) (?:database|registry)",
        ),
        "reason": (
            "the cited source is about analyzing data that exists but is held "
            "privately across institutions — it cannot support claims about "
            "trials/data that were never published or registered"
        ),
    },
]


def scan_concept_boundary_mismatch(
    report_text: str,
    sources: List[Dict],
    max_flags: int = 3,
) -> List[Dict]:
    """Deterministic concept-boundary check for cited sentences.

    A flag is raised when a cited source sits on one side of a declared
    conceptual boundary while the citing sentence presupposes the other
    side. Flags go to the critique LLM for adjudication, same as the
    zero-overlap alignment flags.
    """
    flags: List[Dict] = []

    main_text = report_text
    bib_match = re.search(r'\n##\s*SOURCES\b.*\Z', report_text, re.DOTALL | re.IGNORECASE)
    if bib_match:
        main_text = report_text[:bib_match.start()]

    lines = [ln for ln in main_text.split("\n") if not ln.strip().startswith(("#", "|"))]
    sentences = re.split(r"(?<=[.!?])\s+", "\n".join(lines))

    for sentence in sentences:
        if len(sentence.split()) < 8 or len(flags) >= max_flags:
            continue
        cited = sorted({n for n in extract_numbered_citations(sentence) if 1 <= n <= len(sources)})
        if not cited:
            continue
        for boundary in _CONCEPT_BOUNDARIES:
            claim_hit = any(re.search(p, sentence, re.I) for p in boundary["claim_markers"])
            if not claim_hit:
                continue
            for n in cited:
                src = sources[n - 1]
                src_text = f"{src.get('title') or ''} {src.get('snippet') or ''}".lower()
                hits = [p for p in boundary["source_markers"] if re.search(p, src_text)]
                if len(hits) >= 2:
                    flags.append({
                        "source": n,
                        "excerpt": sentence.strip()[:160],
                        "reason": boundary["reason"],
                    })
                    break
            if len(flags) and flags[-1]["source"] in cited:
                break  # one flag per sentence is enough for adjudication

    return flags


# ── Domain-generalization scan ──────────────────────────────────────
# Live failure: "Neglect of publication bias compromises meta-analyses of
# educational research" was cited for "under moderate publication bias the
# false positive rate can approach 100%" — a domain-restricted simulation
# finding stated as a general law. Zero-overlap alignment can't catch it
# (the words overlap); the failure is SCOPE, not topic.

_DOMAIN_HEAD = (
    r"(?:research|studies|trials|patients?|populations?|practice|settings?"
    r"|children|adolescents|adults?|women|men|nurses?|physicians?|schools?"
    r"|universities|cohorts?|surveys?|projects?|mice|rats|animals?"
    r"|registr(?:y|ies))"
)
# Trailing "in/of/among <up to 3 words> <head>" qualifier in a source title.
_DOMAIN_QUALIFIER_RE = re.compile(
    r"\b(?:in|of|among)\s+((?:[a-z][a-z\-']*\s+){0,3}" + _DOMAIN_HEAD + r")\b[\s.,\"']*$",
    re.I,
)
# Claims strong enough that an unstated scope restriction matters.
_STRENGTH_MARKERS_RE = re.compile(
    r"\b\d+(?:\.\d+)?\s?(?:%|percent)\b"
    r"|\bapproach(?:es|ed)?\b"
    r"|\bas (?:high|large|many|much) as\b"
    r"|\bup to\b"
    r"|\bcan (?:reach|exceed|approach|be as high as)\b"
    r"|\bnearly\b|\balmost\b"
    r"|\bmost\b|\bthe majority\b"
    r"|\bwidespread\b|\bcommonly\b"
    r"|\btypically\b|\bgenerally\b",
    re.I,
)

def scan_domain_generalization(
    report_text: str,
    sources: List[Dict],
    max_flags: int = 3,
) -> List[Dict]:
    """Deterministic domain-scope check for cited sentences.

    When a source's title restricts its conclusion to a domain/population
    (e.g. "...in educational research"), a citing sentence that asserts a
    strong quantitative/general claim WITHOUT any of the qualifier's tokens
    is flagged for the critique pass to adjudicate: either restrict the
    claim to the source's scope or attribute the scope explicitly.
    """
    flags: List[Dict] = []

    main_text = report_text
    bib_match = re.search(r'\n##\s*SOURCES\b.*\Z', report_text, re.DOTALL | re.IGNORECASE)
    if bib_match:
        main_text = report_text[:bib_match.start()]

    # Pre-compute per-source domain qualifiers (token sets, len>=4)
    quals: Dict[int, set] = {}
    for i, src in enumerate(sources, 1):
        title = (src.get("title") or "").strip()
        m = _DOMAIN_QUALIFIER_RE.search(title)
        if not m:
            continue
        toks = {t for t in re.findall(r"[a-z][a-z\-']{3,}", m.group(1).lower())}
        # Generic method contexts are not domain restrictions
        if toks and toks <= {"meta", "analyses", "analysis", "systematic",
                             "review", "reviews", "evidence", "syntheses",
                             "synthesis", "prospective", "registered"}:
            continue
        if toks:
            quals[i] = toks
    if not quals:
        return flags

    lines = [ln for ln in main_text.split("\n") if not ln.strip().startswith(("#", "|"))]
    sentences = re.split(r"(?<=[.!?])\s+", "\n".join(lines))
    for sentence in sentences:
        if len(sentence.split()) < 8 or len(flags) >= max_flags:
            continue
        if not _STRENGTH_MARKERS_RE.search(sentence):
            continue
        cited = sorted({n for n in extract_numbered_citations(sentence) if 1 <= n <= len(sources)})
        if not cited:
            continue
        sent_l = sentence.lower()
        for n in cited:
            toks = quals.get(n)
            if not toks:
                continue
            if any(t in sent_l for t in toks):
                continue  # scope carried into the claim — fine
            flags.append({
                "source": n,
                "excerpt": sentence.strip()[:160],
                "reason": (
                    "the cited source's conclusion is scoped to "
                    f"'{' '.join(sorted(toks))}' — the claim states it without "
                    "that scope restriction; restrict the claim or attribute the scope"
                ),
            })
            break

    return flags


# ── News-tier citation scan ─────────────────────────────────────────
# Live failure: Pulmonology Advisor (consumer medical news, web P3) was
# cited for a claim while peer-reviewed sources in the same pool covered
# the same point. A premium report cites peer-reviewed literature for
# scientific claims; news/web sources are for documented real-world events.

_NEWS_HOST_RE = re.compile(
    r"(?:news|advisor|media|daily|times|post|herald|tribune|chronicle"
    r"|weekly|today|magazine|forbes|bloomberg|reuters|barron|express"
    r"|buzzfeed|substack|blogspot|wordpress|medium\.com|outreach|digest)",
    re.I,
)

def scan_news_tier_citations(
    report_text: str,
    sources: List[Dict],
    max_flags: int = 2,
) -> List[Dict]:
    """Flag news/consumer-tier web sources cited for claims that a
    peer-reviewed (P1) source in the pool also supports.

    Only fires when a P1 source shares >=2 content words with the news
    source's title+snippet — i.e. a re-anchoring target actually exists.
    """
    flags: List[Dict] = []

    news_idx: Dict[int, set] = {}
    for i, src in enumerate(sources, 1):
        if (src.get("tier") or "").upper() != "P3":
            continue
        host = ""
        url = src.get("url") or (src.get("extra") or {}).get("web_url") or ""
        m = re.search(r"(?:https?://)?(?:www\.)?([^/:?#]+)", url or "")
        if m:
            host = m.group(1)
        if host and _NEWS_HOST_RE.search(host):
            news_idx[i] = _alignment_content_words(
                f"{src.get('title') or ''} {src.get('snippet') or ''}"
            )
    if not news_idx:
        return flags

    # Best peer-reviewed re-anchoring target per news source
    replacement: Dict[int, int] = {}
    for i, news_words in news_idx.items():
        best_p, best_shared = 0, 0
        for j, src in enumerate(sources, 1):
            if j == i or (src.get("tier") or "").upper() != "P1":
                continue
            shared = news_words & _alignment_content_words(
                f"{src.get('title') or ''} {src.get('snippet') or ''}"
            )
            if len(shared) > best_shared:
                best_p, best_shared = j, len(shared)
        if best_shared >= 2:
            replacement[i] = best_p
    if not replacement:
        return flags

    main_text = report_text
    bib_match = re.search(r'\n##\s*SOURCES\b.*\Z', report_text, re.DOTALL | re.IGNORECASE)
    if bib_match:
        main_text = report_text[:bib_match.start()]
    lines = [ln for ln in main_text.split("\n") if not ln.strip().startswith(("#", "|"))]
    sentences = re.split(r"(?<=[.!?])\s+", "\n".join(lines))
    for sentence in sentences:
        if len(sentence.split()) < 8 or len(flags) >= max_flags:
            continue
        cited = sorted({n for n in extract_numbered_citations(sentence) if 1 <= n <= len(sources)})
        for n in cited:
            if n in replacement and n not in {f["source"] for f in flags}:
                p = replacement[n]
                flags.append({
                    "source": n,
                    "replacement": p,
                    "excerpt": sentence.strip()[:160],
                    "reason": (
                        f"a news/consumer-tier web source is cited for a scientific "
                        f"claim while peer-reviewed [S{p}] covers the same point — "
                        f"re-anchor the claim to [S{p}], or keep [S{n}] only for the "
                        "documented real-world event it reports (never for the science)"
                    ),
                })
                break

    return flags


# ── Off-topic padding scan ──────────────────────────────────────────
# Live failure (run8 verdict): "Meta-analysis of hypoxic transcriptomes from
# public databases" (RNA-seq) and "Federated Learning in Distributed Medical
# Databases" (brain imaging) were SELECTED and CITED for a question about
# unpublished negative-result TRIALS — keyword collision on generic academic
# vocabulary, zero overlap with the discriminative core. The drafter then
# rationalized one into the scope argument, which is how padding becomes
# fake evidence.
def scan_off_topic_padding(query: str, sources: List[Dict]) -> List[Dict]:
    """Flag selected sources with ZERO discriminative overlap with the query.

    A source whose title+snippet shares none of the query's discriminative
    vocabulary (core content words minus generic academic terms) cannot be
    evidence about the question; whatever the drafter says about it is
    padding. Exemptions: canonical/backfilled sources (registry-attested
    in-domain by construction) and records without any title at all.

    Deterministic — no LLM, no thresholds to tune.
    """
    # Lazy import: engine imports this module at load time, so the cycle is
    # broken by deferring; by call time the engine is fully initialized.
    # _content_words = same stopword + plural-normalization machinery the
    # selection-time relevance gate uses — scan and gate must agree.
    from agent.research_engine import _content_words  # noqa: E402
    disc = discriminative_vocab(_content_words(query))
    flags: List[Dict] = []
    if not disc:
        return flags
    for i, s in enumerate(sources, 1):
        if s.get("canonical_id") or s.get("backfilled"):
            continue
        title = (s.get("title") or "").strip()
        if not title:
            continue
        hits = shared_terms(s, disc)
        if not hits:
            flags.append({
                "source": i,
                "title": title[:120],
                "url": (s.get("url") or "")[:120],
                "reason": (
                    "source shares none of the question's discriminative "
                    "vocabulary — selected on generic-academic keyword "
                    "collision; it cannot be evidence about this question and "
                    "must not be cited (remove citations to it and any "
                    "sentences built on it)"
                ),
            })
    return flags


# ── Evidence-depth scan (run9 verdict fix 2) ────────────────────────
# run9 fetched full text for exactly ONE source, reasoned mostly from
# titles/abstracts, and still produced well-cited prose — including a
# chronic-pain section. A drafter working from abstracts cannot know any
# claim is "established"; established language under a thin full-text base
# is exactly how abstract-level inference dresses up as settled evidence.
FULLTEXT_FLOOR_DEFAULT = 3

_ESTABLISHED_RE = re.compile(
    r"\b(well[- ]?established|firmly[- ]?established|established that|"
    r"definitively|conclusively|conclusive evidence|settled (?:science|question))\b",
    re.I,
)


def scan_unfounded_established(
    report_text: str,
    fulltext_count: int,
    floor: int = FULLTEXT_FLOOR_DEFAULT,
) -> List[Dict]:
    """Flag 'established/definitive/conclusive' language when the full-text
    evidence base is below `floor` fetched sources.

    Not a ban on the WORD — with >= floor full-text sources the scan stays
    silent. Below the floor, every paragraph using the language is flagged:
    the evidence base is title/abstract-level and the prose must say so.

    Deterministic, no I/O, fully unit-testable.
    """
    flags: List[Dict] = []
    if fulltext_count >= floor or not report_text:
        return flags
    for para in report_text.split("\n"):
        stripped = para.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith("|"):
            continue  # headings and provenance tables don't assert evidence
        m = _ESTABLISHED_RE.search(stripped)
        if m:
            flags.append({
                "excerpt": stripped[:200],
                "term": m.group(0).lower(),
                "fulltext_count": fulltext_count,
                "floor": floor,
                "reason": (
                    f"'{m.group(0)}' claimed with only {fulltext_count} source(s) "
                    f"fetched at full text (floor {floor}) — the base is "
                    f"abstract-level; qualify the sentence or drop the language"
                ),
            })
    return flags


# ── Scope-dodge scan ────────────────────────────────────────────────
# Live failure (run7 verdict): the SCOPE section conceded the query's
# literal demand could not be met but stayed in the safe zone — a generic
# hedge with no STRUCTURAL reason (why the demanded data fundamentally
# cannot exist) and no presentation of documented real-world attempts.

_SCOPE_SECTION_RE = re.compile(r"\n##\s*SCOPE\b[^\n]*\n(.*?)(?=\n##\s|\Z)", re.I | re.S)
_FEASIBILITY_NEGATION_RE = re.compile(
    r"cannot (?:be |directly |fully |systematically )?(?:perform|conduct|address|analyz|answer|compile|execute)"
    r"|not (?:methodologically |currently |directly |statistically |empirically )?feasible"
    r"|could not be (?:directly |fully |systematically )?(?:analyzed|performed|addressed|answered|compiled)"
    r"|(?:is|are) not (?:directly |currently |methodologically )?(?:possible|available|analyzable|achievable)"
    r"|unable to (?:perform|conduct|address|analyz|compile)",
    re.I,
)
_STRUCTURAL_REASON_RE = re.compile(
    r"by definition|no sampling frame|never (?:been )?(?:registered|reported|published|recorded|indexed)"
    r"|(?:does|do) not exist|no denominator|impossible to (?:enumerate|identify|count|compile|locate)"
    r"|cannot be enumerated|not systematically (?:recorded|captured|collected|indexed)"
    r"|no systematic record|no (?:way|method) to (?:enumerate|identify|count|locate|retrieve)"
    r"|hidden (?:population|fraction)|no (?:such )?(?:database|registry|index|denominator)"
    r"|no exhaustive|inherently (?:unknowable|unmeasurable|unobservable)"
    # Data-availability family (run8 regression): the structural reason for
    # never-published-trial questions is that the DATA exists in no
    # retrievable form — registries record existence, not effect sizes.
    # LLM-drafted scopes state this as "without access to the individual
    # participant data", "registries rarely contain the detailed effect size
    # data", "<noun> data ... unavailable/inaccessible". Recognize these
    # without opening the door to generic "not found in our search" hedges:
    # each marker requires an access/data noun phrase, not a bare negative.
    r"|without access to (?:the |any |its )?(?:raw|individual|underlying|effect|outcome|patient|source)"
    r"|no access to (?:the |any )?(?:raw|individual|underlying|effect|outcome|patient|source)"
    r"|(?:rarely|seldom) contain"
    r"|(?:raw|individual participant|patient-level|individual-level|underlying|source|effect.size|outcome|effect)"
    r"[\w\s,()-]{0,80}\b(?:unavailable|inaccessible|cannot be (?:retrieved|obtained|accessed|extracted))"
    ,
    re.I,
)

def scan_scope_dodge(report_text: str) -> List[Dict]:
    """Flag a SCOPE section that negates feasibility without the structural
    argument. Returns at most one flag — the fix is additive content."""
    flags: List[Dict] = []
    m = _SCOPE_SECTION_RE.search(report_text or "")
    if not m:
        return flags
    scope_text = m.group(1)
    if not _FEASIBILITY_NEGATION_RE.search(scope_text):
        return flags
    if _STRUCTURAL_REASON_RE.search(scope_text):
        return flags
    flags.append({
        "source": 0,
        "excerpt": re.sub(r"\s+", " ", scope_text).strip()[:200],
        "reason": (
            "the SCOPE section concedes the query's literal demand cannot be "
            "met but gives only a generic hedge — it must state the STRUCTURAL "
            "reason (why the demanded data/analysis fundamentally cannot exist, "
            "not merely that it was not found) and, where the source list "
            "contains documented real-world attempts/audits of the problem, "
            "present what they found"
        ),
    })
    return flags
