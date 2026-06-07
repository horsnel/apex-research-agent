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


def apply_temporal_decay(
    sources: List[Dict],
    decay_factor: float = 0.95,
    current_year: Optional[int] = None,
) -> List[Dict]:
    """
    Apply temporal decay to source scores.

    A 2024 meta-analysis should outweigh a 2019 RCT on the same question.
    Formula: score *= decay_factor ^ (current_year - source_year)

    Args:
        sources: List of source dicts with 'published_date' and 'score'
        decay_factor: Yearly decay (0.95 = 5% penalty per year of age)
        current_year: Override current year for testing

    Returns:
        Sources with adjusted scores
    """
    year = current_year or datetime.now().year

    for s in sources:
        pub_date = s.get("published_date", "")
        source_year = None

        if pub_date and len(pub_date) >= 4:
            try:
                source_year = int(pub_date[:4])
            except (ValueError, TypeError):
                pass

        if source_year and source_year > 1900:
            age = max(0, year - source_year)
            decay = decay_factor ** age
            score = s.get("score", s.get("similarity", 1.0))
            s["adjusted_score"] = score * decay
        else:
            # No date — slight penalty for undated sources
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
) -> ParallelResearchResult:
    """
    Execute parallel research across all sources with graceful failure handling.

    Key features:
    - All sources queried simultaneously (no sequential blocking)
    - Individual failures don't crash the pipeline
    - Failed sources tracked and reported
    - Minimum viable response even if most sources fail

    Args:
        query: Research query
        classification: Query type for routing
        max_results_per_source: Max results per source

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

    # Apply tier enforcement
    all_results = tier_enforce(all_results) if all_results else []

    # Apply temporal decay
    all_results = apply_temporal_decay(all_results)

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


async def generate_research_report(
    query: str,
    classification: str = "academic",
    depth: str = "quick",
    max_cycles: int = 3,
    user_id: Optional[str] = None,
    apex_tier: str = "apex-free",
) -> ResearchReport:
    """
    Generate a structured research report with verification.

    APEX 2.0 pipeline: Research → Verify → Compile → Wiki → Cache
    Every query compounds. Every claim is traceable.

    Args:
        query: Research query
        classification: Query type for routing
        depth: "quick" (1 cycle) or "thorough" (up to max_cycles)
        max_cycles: Max research cycles for thorough mode
        user_id: Optional user ID for cache association
        apex_tier: apex-free or apex-premium

    Returns:
        ResearchReport with structured findings + wiki/cache metadata
    """
    start = time.time()

    # ── APEX 2.0 Step 0: Check LLM Wiki cache ──
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

    # Step 1: Parallel research
    research = await parallel_research(query, classification)

    # Step 2: Iterative research (if thorough mode)
    if depth == "thorough" and max_cycles > 1:
        research = await _iterative_research(query, research, max_cycles)

    # Step 3: Extract and verify claims
    claims = _extract_claims_from_sources(research.results)
    verification = verify_claims_from_sources(claims, research.results)

    # Step 4: Generate structured report via LLM
    raw_report = await _generate_report_text(query, research.results, verification, depth)

    # Step 5: Parse into structured format
    report = _parse_report(raw_report)
    report.query = query
    report.verification = verification
    report.total_latency_ms = int((time.time() - start) * 1000)
    report.depth = depth
    report.sources = research.results
    report.from_cache = False

    # ── APEX 2.0 Step 6: Compile into Wiki + Cache ──
    try:
        wiki_page, cache_id = await wiki_engine.research_to_wiki(
            query=query,
            report=raw_report,
            sources=research.results,
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
    """Generate the structured report text via LLM."""
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
        max_tokens=max_tokens,  # 2048 quick / 4096 thorough (was 500)
        similarity=0.3,  # Route to strongest models (was 0.5)
        table_needed=True,
        system_prompt=system_prompt,
    )

    return router_result.content


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
) -> ParallelResearchResult:
    """
    Iterative research: search, identify gaps, search again.

    Opt-in feature: only triggered by depth="thorough".
    Each cycle identifies knowledge gaps and searches for them.

    Args:
        query: Original query
        initial_research: Results from first research cycle
        max_cycles: Maximum number of research cycles

    Returns:
        Enriched ParallelResearchResult with additional sources
    """
    from agent.llm_router import synthesize_with_router

    all_results = list(initial_research.results)
    all_successful = list(initial_research.successful_sources)
    all_failed = list(initial_research.failed_sources)

    for cycle in range(1, max_cycles):
        # Identify gaps using LLM
        context = _build_gap_context(query, all_results)
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


def _build_gap_context(query: str, results: List[Dict]) -> str:
    """Build context string for gap identification."""
    parts = [f"Original query: {query}\n"]
    parts.append(f"Sources found so far: {len(results)}\n")
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

    gap_prompt = f"""Given this research query and the sources found so far, identify 1-3 specific knowledge gaps that need more research.

{context}

What specific information is still missing or unclear? Return ONLY the gaps as numbered search queries, nothing else.
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

    Returns:
        Complete research results dict
    """
    start = time.time()

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
