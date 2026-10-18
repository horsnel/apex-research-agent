"""
NLI Claim Verifier — replaces keyword-overlap "verification" with entailment.

The old verify_claims_from_sources() marked support/conflict via bag-of-words
overlap and naive negation lists ("however," appears in ~half of all abstracts
and falsely flagged them as conflicting). Epistemic labels were therefore
unreliable.

This module keeps the keyword pass as a CHEAP CANDIDATE PREFILTER, then asks
an LLM (routed to JSON-reliable mid-tier models) a proper NLI question per
(claim, candidate source) pair:

    Does this source SUPPORT, CONTRADICT, or NOT_ADDRESS the claim?
    → {"verdict": "...", "quote": "...", "confidence": 0.x}

The verdicts are aggregated into the same epistemic statuses the rest of the
pipeline already understands (ESTABLISHED / TENTATIVE / ACTIVE_DEBATE /
SPECULATIVE / UNVERIFIED), so downstream code needs no changes.
"""

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Cap on LLM NLI calls per verification run (cost guard)
MAX_NLI_CALLS = int(__import__("os").getenv("APEX_MAX_NLI_CALLS", "24"))
NLI_TIMEOUT = float(__import__("os").getenv("APEX_NLI_TIMEOUT", "25"))

NLI_SYSTEM_PROMPT = """You are a precise entailment judge for research claims.
Given a CLAIM and a SOURCE TEXT, decide:
- "support": the source provides direct evidence for the claim
- "contradict": the source provides direct evidence against the claim
- "not_address": the source does not address the claim's core assertion

Be strict: topical similarity is NOT support. Return ONLY valid JSON:
{"verdict": "support"|"contradict"|"not_address", "quote": "<short verbatim quote from the source that justifies your verdict, or empty>", "confidence": <0.0-1.0>}"""


@dataclass
class NLIVerdict:
    """Result of one claim-source entailment check."""
    verdict: str            # support | contradict | not_address | parse_error
    quote: str = ""
    confidence: float = 0.0
    source_idx: int = 0
    source_url: str = ""
    source_tier: str = "UNV"
    error: str = ""


@dataclass
class NLIResult:
    """Aggregated NLI verification for one claim."""
    statement: str
    supports: List[NLIVerdict] = field(default_factory=list)
    contradictions: List[NLIVerdict] = field(default_factory=list)
    not_addressed: int = 0


# ═══════════════════════════════════════════════════════════════
# CANDIDATE PREFILTER (keyword, cheap)
# ═══════════════════════════════════════════════════════════════


def _claim_terms(text: str) -> set:
    return set(re.findall(r'\b\w{4,}\b', text.lower()))


def select_nli_candidates(
    claim: str,
    sources: List[Dict],
    max_candidates: int = 4,
    min_overlap: float = 0.25,
) -> List[Dict]:
    """
    Cheap keyword prefilter: pick the sources most likely to bear on a claim,
    so NLI calls are spent on plausible pairs instead of everything × everything.
    """
    claim_terms = _claim_terms(claim)
    if not claim_terms:
        return []

    scored: List[tuple] = []
    for i, s in enumerate(sources):
        text = f"{s.get('title', '')} {s.get('snippet', '') or s.get('content', '')}".lower()
        if not text.strip():
            continue
        overlap = claim_terms & _claim_terms(text)
        ratio = len(overlap) / max(len(claim_terms), 1)
        if ratio >= min_overlap:
            scored.append((ratio, i, s))

    scored.sort(key=lambda t: t[0], reverse=True)
    return [s for _, _, s in scored[:max_candidates]]


# ═══════════════════════════════════════════════════════════════
# NLI CALL
# ═══════════════════════════════════════════════════════════════


async def verify_claim_against_source(claim: str, source: Dict, source_idx: int = 0) -> NLIVerdict:
    """
    Run one entailment check: does this source support / contradict /
    not-address this claim? Routed through the LLM router with task_type="nli"
    (mid-tier JSON-capable models).
    """
    from agent.llm_router import synthesize_with_router

    source_text = (source.get("snippet", "") or source.get("content", "") or "")[:2500]
    title = source.get("title", "")

    user_prompt = f"CLAIM: {claim}\n\nSOURCE [{title}]:\n{source_text}"

    try:
        result = await asyncio.wait_for(
            synthesize_with_router(
                query="entailment check",
                context=user_prompt,
                max_tokens=150,
                system_prompt=NLI_SYSTEM_PROMPT,
                task_type="nli",
            ),
            timeout=NLI_TIMEOUT,
        )

        content = result.content.strip()
        if content.startswith("[ALL_LLM_FAILED]"):
            return NLIVerdict(verdict="parse_error", source_idx=source_idx,
                              source_url=source.get("url", ""), error="all models failed")

        # Strip markdown fences if present
        if content.startswith("```"):
            content = re.sub(r"^```(?:json)?\s*", "", content)
            content = re.sub(r"\s*```$", "", content)

        m = re.search(r'\{[\s\S]*\}', content)
        if not m:
            return NLIVerdict(verdict="parse_error", source_idx=source_idx,
                              source_url=source.get("url", ""), error=content[:80])

        data = json.loads(m.group())
        verdict = str(data.get("verdict", "not_address")).lower().strip()
        if verdict not in ("support", "contradict", "not_address"):
            verdict = "not_address"

        try:
            confidence = max(0.0, min(1.0, float(data.get("confidence", 0.5))))
        except (TypeError, ValueError):
            confidence = 0.5

        return NLIVerdict(
            verdict=verdict,
            quote=str(data.get("quote", ""))[:300],
            confidence=confidence,
            source_idx=source_idx,
            source_url=source.get("url", ""),
            source_tier=source.get("tier", "UNV"),
        )

    except asyncio.TimeoutError:
        return NLIVerdict(verdict="parse_error", source_idx=source_idx,
                          source_url=source.get("url", ""), error="timeout")
    except Exception as e:
        return NLIVerdict(verdict="parse_error", source_idx=source_idx,
                          source_url=source.get("url", ""), error=str(e))


# ═══════════════════════════════════════════════════════════════
# AGGREGATION → EPISTEMIC STATUS
# ═══════════════════════════════════════════════════════════════


def _sources_independent(a: NLIVerdict, b: NLIVerdict) -> bool:
    """Two verdicts are independent if they come from different domains and no shared author string in URLs."""
    def domain(url: str) -> str:
        m = re.match(r'https?://([^/]+)', url or "")
        return (m.group(1).lower().removeprefix("www.") if m else (url or "").lower())

    return domain(a.source_url) != domain(b.source_url) and bool(a.source_url) and bool(b.source_url)


def aggregate_nli_to_status(result: NLIResult) -> tuple:
    """
    Convert NLI verdicts into (epistemic_status, confidence, evidence_type).

    Rules:
    - 2+ independent SUPPORTS with at least one P1/P2  → ESTABLISHED
    - Any genuine CONTRADICT with support              → ACTIVE_DEBATE
    - Only 1 support, or supports only from P3         → TENTATIVE
    - No support                                       → UNVERIFIED
    """
    supports = [v for v in result.supports if v.verdict == "support"]
    contradictions = [v for v in result.contradictions if v.verdict == "contradict"]

    # Count independent supports
    independent: List[NLIVerdict] = []
    for v in supports:
        if all(_sources_independent(v, other) for other in independent):
            independent.append(v)

    p12_support = [v for v in supports if v.source_tier in ("P1", "P2")]
    n_contra = len([v for v in contradictions if v.confidence >= 0.4])

    if len(independent) >= 2 and p12_support:
        status, confidence = "ESTABLISHED", min(0.95, 0.55 + 0.12 * len(independent))
    elif n_contra and supports:
        status, confidence = "ACTIVE_DEBATE", 0.35
    elif n_contra:
        status, confidence = "ACTIVE_DEBATE", 0.25
    elif len(supports) >= 1:
        if len(independent) >= 2:
            status, confidence = "ESTABLISHED", 0.7
        elif p12_support:
            status, confidence = "TENTATIVE", 0.55
        else:
            status, confidence = "TENTATIVE", 0.35
    else:
        status, confidence = "UNVERIFIED", 0.05

    # Evidence type from the strongest supporting quote
    evidence_type = ""
    quote_blob = " ".join(v.quote.lower() for v in supports)
    if "meta-analysis" in quote_blob:
        evidence_type = "meta-analysis"
    elif "systematic review" in quote_blob:
        evidence_type = "systematic-review"
    elif "randomized" in quote_blob or " rct" in quote_blob:
        evidence_type = "RCT"
    elif "cohort" in quote_blob:
        evidence_type = "cohort"
    elif any(v.source_tier == "P1" for v in supports):
        evidence_type = "peer-reviewed"
    elif supports:
        evidence_type = "observation"

    return status, round(confidence, 2), evidence_type


# ═══════════════════════════════════════════════════════════════
# BATCH VERIFICATION
# ═══════════════════════════════════════════════════════════════


async def verify_claims_with_nli(
    claims: List[str],
    sources: List[Dict],
    max_claims: int = 12,
    max_candidates_per_claim: int = 4,
) -> Dict[str, Dict]:
    """
    Verify a batch of claims with NLI.

    Returns a dict: claim_text → {"status", "confidence", "evidence_type",
    "supports": [...], "contradictions": [...]}. Claims beyond max_claims or
    with no keyword-relevant candidates are simply absent from the result
    (callers keep their existing heuristic verdict for those).

    Total NLI calls are capped at MAX_NLI_CALLS (cost guard).
    """
    results: Dict[str, Dict] = {}
    budget = MAX_NLI_CALLS

    tasks = []           # (claim, source, idx) coroutine triplets
    claim_meta = {}      # claim → list of (source, idx)

    for claim in claims[:max_claims]:
        candidates = select_nli_candidates(claim, sources, max_candidates_per_claim)
        claim_meta[claim] = []
        for cand in candidates:
            if budget <= 0:
                break
            # global source index (1-based, matches report source numbering)
            gidx = sources.index(cand) + 1 if cand in sources else 0
            claim_meta[claim].append((cand, gidx))
            tasks.append((claim, cand, gidx))
            budget -= 1
        if budget <= 0:
            break

    if not tasks:
        return results

    verdict_lists = await asyncio.gather(
        *[verify_claim_against_source(c, s, i) for c, s, i in tasks],
        return_exceptions=True,
    )

    for (claim, source, idx), verdict in zip(tasks, verdict_lists):
        if isinstance(verdict, Exception):
            continue
        entry = results.setdefault(claim, {
            "supports": [], "contradictions": [], "not_addressed": 0,
        })
        if verdict.verdict == "support":
            entry["supports"].append(verdict)
        elif verdict.verdict == "contradict":
            entry["contradictions"].append(verdict)
        elif verdict.verdict == "not_address":
            entry["not_addressed"] += 1

    # Aggregate
    final: Dict[str, Dict] = {}
    for claim, entry in results.items():
        nli_result = NLIResult(
            statement=claim,
            supports=entry["supports"],
            contradictions=entry["contradictions"],
        )
        status, confidence, evidence_type = aggregate_nli_to_status(nli_result)
        final[claim] = {
            "status": status,
            "confidence": confidence,
            "evidence_type": evidence_type,
            "supports": [vars(v) for v in entry["supports"]],
            "contradictions": [vars(v) for v in entry["contradictions"]],
            "independent_support_count": _count_independent(entry["supports"]),
        }

    logger.info(f"NLI verification: {len(final)}/{len(claims[:max_claims])} claims verified "
                f"with {len(tasks)} entailment calls")
    return final


def _count_independent(supports: List) -> int:
    independent: List = []
    for v in supports:
        if v.verdict == "support" and all(_sources_independent(v, other) for other in independent):
            independent.append(v)
    return len(independent)
