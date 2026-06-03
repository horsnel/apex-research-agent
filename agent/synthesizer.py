"""
Synthesizer — generates minimal, token-efficient answers.

Core principles:
1. No LLM writes prose when a source answers directly. Pass through.
2. Tables > prose for comparative data.
3. Output cap: 150 tokens default, 300 for multi-item tables.
4. Every claim must have inline citation [Source, Tier] or [UNVERIFIED].
5. No reasoning emitted: Internal chain-of-thought stays internal.
6. If a tool result directly answers the query, emit it verbatim with ≤5 words of framing.

Uses the 9-model fallback router:
- similarity > 0.85 → Pass-through (no LLM call at all)
- similarity 0.72-0.85 → Cloudflare cheap models (Granite, GLM-4.7)
- similarity < 0.72 → Full fallback chain up to DeepSeek-V3
- Table queries → Mid+ capable models only
"""

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import List, Optional

import httpx

from .llm_router import synthesize_with_router, RouterResult
from .retriever import RetrievedChunk

logger = logging.getLogger(__name__)

# ── Configuration ──
DEFAULT_SYNTHESIS_TOKENS = int(os.getenv("DEFAULT_SYNTHESIS_TOKENS", "150"))
MAX_SYNTHESIS_TOKENS = int(os.getenv("MAX_SYNTHESIS_TOKENS", "300"))

# ── APEX Compression System Prompt ──
APEX_SYSTEM_PROMPT = """You are APEX, a token-efficient research synthesis engine.

RULES (NON-NEGOTIABLE):
1. OUTPUT FORMAT: Direct answer only. No preamble, no "Based on...", no "The answer is...".
2. PASS-THROUGH: If a source directly answers the query, quote it verbatim with ≤5 words of framing.
3. TABLES OVER PROSE: If the answer involves comparison or multiple items, use a markdown table.
4. TOKEN CAP: Maximum 150 tokens. 300 ONLY if a multi-row table is required.
5. CITATION MANDATORY: Every factual claim must end with [Source, Tier] or [UNVERIFIED].
   Format: [Author Year, P1] or [domain.com/path, P2]
6. NO REASONING: Do not show your chain-of-thought. Output starts at information zero.
7. SOURCE HIERARCHY: P1 > P2 > P3. If P1 exists, ignore P3 unless it's counter-evidence.
8. NO FILLER: No "Additionally", "Furthermore", "In conclusion", or similar filler.
9. CONFLICTS: If sources disagree, note the conflict with both citations in one line.
10. UNVERIFIED: If no source supports a claim, either omit it or mark it [UNVERIFIED].
"""


@dataclass
class SynthesisResult:
    """Result of the synthesis process."""
    answer: str
    token_count: int
    sources_used: List[dict]
    method: str  # "pass_through", "synthesis", "table", "raw_context"
    model_used: str = ""
    provider: str = ""
    fallback_count: int = 0


def _format_citation(chunk: RetrievedChunk) -> str:
    """Format an inline citation for a chunk."""
    domain = chunk.domain or "unknown"
    tier = chunk.source_tier or "UNV"

    # Try to create a more specific citation
    if chunk.authors:
        # Use first author + year
        first_author = chunk.authors[0].split(",")[0].strip()
        year = ""
        if chunk.metadata.get("arxiv_id"):
            year = chunk.metadata["arxiv_id"][:4]
        elif chunk.metadata.get("pmid"):
            year = ""
        citation = f"{first_author} {year}".strip()
        if citation and citation != first_author:
            return f"[{citation}, {tier}]"

    return f"[{domain}, {tier}]"


def _check_direct_answer(query: str, chunks: List[RetrievedChunk]) -> Optional[str]:
    """
    Check if any single chunk directly answers the query.

    Heuristic: If the top chunk's similarity > 0.85 and its content
    is a direct answer (not just topically related), pass through.

    Returns:
        Direct answer string if found, None otherwise
    """
    if not chunks:
        return None

    top = chunks[0]
    if top.similarity_score >= 0.85:
        # High confidence — the chunk likely directly answers
        citation = _format_citation(top)

        # Truncate if too long
        text = top.raw_text
        if len(text.split()) > DEFAULT_SYNTHESIS_TOKENS:
            # Find the most relevant sentence
            sentences = text.split(". ")
            # Return first 2-3 sentences that are most relevant
            text = ". ".join(sentences[:3])
            if not text.endswith("."):
                text += "."

        return f"{text} {citation}"

    return None


def _build_context(chunks: List[RetrievedChunk], scraped_text: Optional[str] = None) -> str:
    """
    Build the context string from retrieved chunks and/or scraped text.

    Args:
        chunks: Retrieved chunks
        scraped_text: Optional live-scraped text

    Returns:
        Formatted context string
    """
    parts = []

    for i, chunk in enumerate(chunks, 1):
        citation = _format_citation(chunk)
        parts.append(f"[Source {i} {citation}]: {chunk.raw_text}")

    if scraped_text:
        parts.append(f"[Live Source]: {scraped_text}")

    return "\n\n".join(parts)


def _estimate_tokens(text: str) -> int:
    """Estimate token count."""
    return max(1, len(text.split()))


def _detect_table_needed(query: str, chunks: List[RetrievedChunk]) -> bool:
    """
    Detect if the query requires a table (comparative/multi-item).

    Heuristics:
    - Query contains "compare", "vs", "versus", "difference", "differences"
    - Multiple distinct sources covering different items
    - Query asks about "types", "methods", "approaches"
    """
    table_patterns = re.compile(
        r"\b(compare|vs|versus|differences?|types|methods|approaches|alternatives|options|rankings?)\b",
        re.I,
    )

    if table_patterns.search(query):
        return True

    # Multiple distinct domains = likely comparative
    domains = set(c.domain for c in chunks)
    if len(domains) >= 3:
        return True

    return False


async def synthesize(
    query: str,
    chunks: List[RetrievedChunk],
    scraped_text: Optional[str] = None,
    force_synthesis: bool = False,
) -> SynthesisResult:
    """
    Main synthesis function — produces a token-efficient answer.

    Decision flow:
    1. If a single chunk directly answers -> pass through (no LLM call)
    2. Determine similarity tier for model selection
    3. If multiple sources or conflicts -> LLM synthesis via 9-model router
    4. If comparative query -> table format, higher token cap, stronger models
    5. Enforce token cap

    Args:
        query: User query
        chunks: Retrieved chunks from RAG
        scraped_text: Optional live-scraped text
        force_synthesis: Force LLM synthesis even for high-similarity

    Returns:
        SynthesisResult with answer, token count, and sources
    """
    # Track sources used
    sources_used = []
    for chunk in chunks:
        sources_used.append({
            "url": chunk.source_url,
            "tier": chunk.source_tier,
            "title": chunk.title,
            "similarity": chunk.similarity_score,
        })

    # Step 1: Check for direct pass-through
    if not force_synthesis and not scraped_text:
        direct_answer = _check_direct_answer(query, chunks)
        if direct_answer:
            return SynthesisResult(
                answer=direct_answer,
                token_count=_estimate_tokens(direct_answer),
                sources_used=sources_used,
                method="pass_through",
                model_used="pass-through",
                provider="none",
                fallback_count=0,
            )

    # Step 2: Build context for LLM
    context = _build_context(chunks, scraped_text)

    # Step 3: Determine token budget and similarity for tier selection
    table_needed = _detect_table_needed(query, chunks)
    max_tokens = MAX_SYNTHESIS_TOKENS if table_needed else DEFAULT_SYNTHESIS_TOKENS

    # Calculate average similarity for model tier selection
    avg_similarity = None
    if chunks:
        avg_similarity = sum(c.similarity_score for c in chunks) / len(chunks)

    # Step 4: LLM synthesis via 9-model fallback router
    router_result = await synthesize_with_router(
        query=query,
        context=context,
        max_tokens=max_tokens,
        similarity=avg_similarity,
        table_needed=table_needed,
        system_prompt=APEX_SYSTEM_PROMPT,
    )

    answer = router_result.content

    # Step 5: Enforce token cap (hard truncate if needed)
    token_count = _estimate_tokens(answer)
    if token_count > max_tokens:
        # Truncate at sentence boundary
        words = answer.split()
        answer = " ".join(words[:max_tokens])
        # Find last sentence boundary
        last_period = answer.rfind(".")
        if last_period > len(answer) * 0.7:
            answer = answer[:last_period + 1]
        token_count = _estimate_tokens(answer)

    method = "table" if table_needed else "synthesis"

    # If all models failed, fall back to raw context
    if answer.startswith("[ALL_LLM_FAILED]"):
        answer = context[:max_tokens * 4]
        method = "raw_context"

    return SynthesisResult(
        answer=answer,
        token_count=token_count,
        sources_used=sources_used,
        method=method,
        model_used=router_result.model_name,
        provider=router_result.provider,
        fallback_count=router_result.fallback_count,
    )
