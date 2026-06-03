"""
Query Classifier — routes queries to RAG or live scraper.

Uses a two-stage approach:
1. Rule-based fast path (zero LLM cost)
2. LLM fallback for ambiguous queries

Output: {"route": "rag" | "live", "reason": "...", "domain_hint": "..."}
"""

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

# ── Configuration ──
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.72"))

# ── Rule patterns ──
# Temporal keywords that signal "needs live data"
LIVE_KEYWORDS = re.compile(
    r"\b(latest|today|breaking|just announced|yesterday|this week|this month|current|now|recent|update)\b",
    re.I,
)

# Known academic/research domains in the vector DB
KNOWN_DOMAINS = {
    "arxiv", "pubmed", "nature", "science", "nejm", "lancet",
    "acm", "ieee", "springer", "wiley", "semanticscholar",
    "openreview", "biorxiv", "medrxiv",
}

# Research topic patterns likely in DB
RESEARCH_PATTERNS = re.compile(
    r"\b(RAG|retrieval.augmented|vector.search|embedding|transformer|"
    r"language.model|LLM|GPT|BERT|fine.tun|prompt.engineer|"
    r"attention.mechanism|neural.network|deep.learning)\b",
    re.I,
)


@dataclass
class ClassificationResult:
    """Result of query classification."""
    route: str  # "rag" or "live"
    reason: str
    domain_hint: str = ""
    confidence: float = 1.0
    method: str = "rules"  # "rules" or "llm"


def classify_rules(query: str) -> ClassificationResult:
    """
    Rule-based query classification (fast, zero cost).

    Priority:
    1. Temporal keywords -> live
    2. Known research patterns -> rag
    3. Default -> rag (with similarity check downstream)

    Args:
        query: User query string

    Returns:
        ClassificationResult
    """
    # Check for temporal signals
    if LIVE_KEYWORDS.search(query):
        return ClassificationResult(
            route="live",
            reason="Temporal keyword detected — likely needs current data",
            confidence=0.9,
            method="rules",
        )

    # Check for known domain references
    query_lower = query.lower()
    for domain in KNOWN_DOMAINS:
        if domain in query_lower:
            return ClassificationResult(
                route="rag",
                reason=f"Known domain reference: {domain}",
                domain_hint=domain,
                confidence=0.85,
                method="rules",
            )

    # Check for research topic patterns
    if RESEARCH_PATTERNS.search(query):
        return ClassificationResult(
            route="rag",
            reason="Research topic pattern match — likely in corpus",
            confidence=0.75,
            method="rules",
        )

    # Default to RAG (similarity threshold will catch low-confidence later)
    return ClassificationResult(
        route="rag",
        reason="Default RAG route — similarity check will validate",
        confidence=0.5,
        method="rules",
    )


async def classify_llm(query: str) -> ClassificationResult:
    """
    LLM-based query classification for ambiguous queries.

    Uses a lightweight model (GPT-4o-mini) to determine routing.

    Args:
        query: User query string

    Returns:
        ClassificationResult
    """
    system_prompt = """You are a query router for a research AI. Classify the query as needing:
- "rag": if it can likely be answered from a pre-loaded academic/research corpus
- "live": if it needs current/real-time data from the web

Respond ONLY with valid JSON: {"route": "rag"|"live", "reason": "...", "domain_hint": "..."}"""

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "gpt-4o-mini",
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": query},
                    ],
                    "max_tokens": 50,
                    "temperature": 0,
                },
            )
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"].strip()

            # Parse JSON from response
            # Handle possible markdown code blocks
            if content.startswith("```"):
                content = re.sub(r"^```(?:json)?\s*", "", content)
                content = re.sub(r"\s*```$", "", content)

            result = json.loads(content)
            return ClassificationResult(
                route=result.get("route", "rag"),
                reason=result.get("reason", "LLM classified"),
                domain_hint=result.get("domain_hint", ""),
                confidence=0.8,
                method="llm",
            )

    except Exception as e:
        logger.warning(f"LLM classification failed: {e}. Falling back to rules.")
        return classify_rules(query)


async def classify_query(
    query: str,
    avg_similarity: Optional[float] = None,
) -> ClassificationResult:
    """
    Classify a query, determining whether to use RAG or live scraping.

    Decision flow:
    1. Rule-based fast path
    2. If low confidence AND similarity is below threshold -> try LLM
    3. If RAG avg similarity < 0.72 after retrieval -> escalate to live

    Args:
        query: User query string
        avg_similarity: Average similarity from RAG retrieval (if already done)

    Returns:
        ClassificationResult with route decision
    """
    # Stage 1: Rules
    result = classify_rules(query)

    # Stage 2: If similarity is known and below threshold, override to live
    if avg_similarity is not None and avg_similarity < SIMILARITY_THRESHOLD:
        result = ClassificationResult(
            route="live",
            reason=f"RAG similarity too low ({avg_similarity:.2f} < {SIMILARITY_THRESHOLD})",
            confidence=0.95,
            method="rules+similarity",
        )
        return result

    # Stage 3: LLM fallback for ambiguous rules-based results
    if result.confidence < 0.6 and OPENAI_API_KEY:
        result = await classify_llm(query)

    logger.info(f"Query classified: route={result.route}, method={result.method}, reason={result.reason}")
    return result


def should_escalate_to_live(avg_similarity: float, has_p1_source: bool = False) -> bool:
    """
    Determine if RAG results are insufficient and we should fall back to live scraping.

    Args:
        avg_similarity: Average similarity score of top-k RAG results
        has_p1_source: Whether any P1 source was found in RAG results

    Returns:
        True if should escalate to live scraping
    """
    if avg_similarity < SIMILARITY_THRESHOLD:
        return True
    if avg_similarity < 0.80 and not has_p1_source:
        return True
    return False
