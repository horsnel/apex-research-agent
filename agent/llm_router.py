"""
LLM Router — 9-model fallback chain with Cloudflare + GitHub Models.

Fallback order (cheapest → most capable):
  1. Pass-through (no LLM, similarity > 0.85)
  2. Cloudflare: @cf/ibm-granite/granite-4.0-h-micro      (~3B,  $0.017/$0.112 per M tok)
  3. Cloudflare: @cf/zai-org/glm-4.7-flash                 (~4.7B, $0.060/$0.400 per M tok)
  4. Cloudflare: @cf/qwen/qwen3-30b-a3b-fp8               (30B MoE, $0.051/$0.335 per M tok)
  5. Cloudflare: @cf/mistralai/mistral-small-3.1-24b-instruct (24B, $0.351/$0.555 per M tok)
  6. GitHub:      Phi-4-mini-instruct                       (3.8B, free 150 req/day)
  7. GitHub:      Mistral-Nemo                              (12B,  free 150 req/day)
  8. GitHub:      GPT-4o-mini                               (cloud, free 150 req/day)
  9. GitHub:      DeepSeek-V3-0324                          (671B MoE, free 150 req/day)

Each provider uses OpenAI-compatible chat completions API.
On failure, the router falls through to the next model automatically.
"""

import asyncio
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════

CLOUDFLARE_API_TOKEN = os.getenv("CLOUDFLARE_API_TOKEN", "")
CLOUDFLARE_ACCOUNT_ID = os.getenv("CLOUDFLARE_ACCOUNT_ID", "")
CLOUDFLARE_BASE_URL = os.getenv(
    "CLOUDFLARE_BASE_URL",
    f"https://api.cloudflare.com/client/v4/accounts/{CLOUDFLARE_ACCOUNT_ID}/ai/v1",
)

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
GITHUB_MODELS_BASE_URL = os.getenv("GITHUB_MODELS_BASE_URL", "https://models.inference.ai.azure.com")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

DEFAULT_SYNTHESIS_TOKENS = int(os.getenv("DEFAULT_SYNTHESIS_TOKENS", "150"))
MAX_SYNTHESIS_TOKENS = int(os.getenv("MAX_SYNTHESIS_TOKENS", "300"))

# ═══════════════════════════════════════════════════════════════
# MODEL REGISTRY
# ═══════════════════════════════════════════════════════════════


class Provider(str, Enum):
    PASSTHROUGH = "passthrough"
    CLOUDFLARE = "cloudflare"
    GITHUB = "github"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


@dataclass
class ModelConfig:
    """Configuration for a single LLM in the fallback chain."""
    name: str
    provider: Provider
    model_id: str
    context_window: int
    max_output_tokens: int
    price_input_per_m: float
    price_output_per_m: float
    supports_tools: bool = False
    tier: str = "cheap"  # "cheap", "mid", "capable", "cloud"
    enabled: bool = True

    @property
    def is_configured(self) -> bool:
        """Check if the required credentials are present."""
        if self.provider == Provider.PASSTHROUGH:
            return True
        elif self.provider == Provider.CLOUDFLARE:
            return bool(CLOUDFLARE_API_TOKEN and CLOUDFLARE_ACCOUNT_ID)
        elif self.provider == Provider.GITHUB:
            return bool(GITHUB_TOKEN)
        elif self.provider == Provider.OPENAI:
            return bool(OPENAI_API_KEY)
        elif self.provider == Provider.ANTHROPIC:
            return bool(ANTHROPIC_API_KEY)
        return False


# ── The 9 Fallback Models ──

FALLBACK_CHAIN: List[ModelConfig] = [
    ModelConfig(
        name="pass-through",
        provider=Provider.PASSTHROUGH,
        model_id="none",
        context_window=0,
        max_output_tokens=DEFAULT_SYNTHESIS_TOKENS,
        price_input_per_m=0.0,
        price_output_per_m=0.0,
        tier="free",
    ),
    ModelConfig(
        name="Granite-4.0-Micro",
        provider=Provider.CLOUDFLARE,
        model_id="@cf/ibm-granite/granite-4.0-h-micro",
        context_window=131000,
        max_output_tokens=DEFAULT_SYNTHESIS_TOKENS,
        price_input_per_m=0.017,
        price_output_per_m=0.112,
        supports_tools=True,
        tier="cheap",
    ),
    ModelConfig(
        name="GLM-4.7-Flash",
        provider=Provider.CLOUDFLARE,
        model_id="@cf/zai-org/glm-4.7-flash",
        context_window=131072,
        max_output_tokens=DEFAULT_SYNTHESIS_TOKENS,
        price_input_per_m=0.060,
        price_output_per_m=0.400,
        supports_tools=True,
        tier="cheap",
    ),
    ModelConfig(
        name="Qwen3-30B-MoE",
        provider=Provider.CLOUDFLARE,
        model_id="@cf/qwen/qwen3-30b-a3b-fp8",
        context_window=32768,
        max_output_tokens=DEFAULT_SYNTHESIS_TOKENS,
        price_input_per_m=0.051,
        price_output_per_m=0.335,
        supports_tools=True,
        tier="mid",
    ),
    ModelConfig(
        name="Mistral-Small-3.1-24B",
        provider=Provider.CLOUDFLARE,
        model_id="@cf/mistralai/mistral-small-3.1-24b-instruct",
        context_window=128000,
        max_output_tokens=MAX_SYNTHESIS_TOKENS,
        price_input_per_m=0.351,
        price_output_per_m=0.555,
        supports_tools=True,
        tier="mid",
    ),
    ModelConfig(
        name="Phi-4-Mini",
        provider=Provider.GITHUB,
        model_id="Phi-4-mini-instruct",
        context_window=128000,
        max_output_tokens=DEFAULT_SYNTHESIS_TOKENS,
        price_input_per_m=0.08,
        price_output_per_m=0.30,
        tier="free",
    ),
    ModelConfig(
        name="Mistral-Nemo-12B",
        provider=Provider.GITHUB,
        model_id="Mistral-Nemo",
        context_window=128000,
        max_output_tokens=MAX_SYNTHESIS_TOKENS,
        price_input_per_m=0.15,
        price_output_per_m=0.60,
        supports_tools=True,
        tier="free",
    ),
    ModelConfig(
        name="GPT-4o-Mini",
        provider=Provider.GITHUB,
        model_id="gpt-4o-mini",
        context_window=128000,
        max_output_tokens=MAX_SYNTHESIS_TOKENS,
        price_input_per_m=0.15,
        price_output_per_m=0.60,
        supports_tools=True,
        tier="cloud",
    ),
    ModelConfig(
        name="DeepSeek-V3",
        provider=Provider.GITHUB,
        model_id="DeepSeek-V3-0324",
        context_window=128000,
        max_output_tokens=MAX_SYNTHESIS_TOKENS,
        price_input_per_m=1.14,
        price_output_per_m=4.56,
        tier="capable",
    ),
]


# ═══════════════════════════════════════════════════════════════
# ROUTER RESULT
# ═══════════════════════════════════════════════════════════════


@dataclass
class LLMCallResult:
    """Result from a single LLM call."""
    success: bool
    content: str
    model_name: str
    model_id: str
    provider: str
    latency_ms: int
    tokens_used: int = 0
    error: str = ""


@dataclass
class RouterResult:
    """Result from the full fallback router."""
    content: str
    model_name: str
    model_id: str
    provider: str
    fallback_count: int  # How many models failed before success
    total_latency_ms: int
    attempts: List[Dict[str, Any]] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════
# API CLIENTS
# ═══════════════════════════════════════════════════════════════


async def _call_cloudflare(
    model_id: str,
    messages: List[Dict[str, str]],
    max_tokens: int = DEFAULT_SYNTHESIS_TOKENS,
    temperature: float = 0.0,
) -> LLMCallResult:
    """Call a Cloudflare Workers AI model (OpenAI-compatible endpoint)."""
    start = time.time()
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{CLOUDFLARE_BASE_URL}/chat/completions",
                headers={
                    "Authorization": f"Bearer {CLOUDFLARE_API_TOKEN}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model_id,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                },
            )
            latency = int((time.time() - start) * 1000)

            if response.status_code != 200:
                error_text = response.text[:200]
                logger.warning(f"Cloudflare {model_id} returned {response.status_code}: {error_text}")
                return LLMCallResult(
                    success=False,
                    content="",
                    model_name=model_id,
                    model_id=model_id,
                    provider="cloudflare",
                    latency_ms=latency,
                    error=f"HTTP {response.status_code}: {error_text}",
                )

            data = response.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
            usage = data.get("usage", {})
            tokens_used = usage.get("total_tokens", 0)

            return LLMCallResult(
                success=True,
                content=content,
                model_name=model_id,
                model_id=model_id,
                provider="cloudflare",
                latency_ms=latency,
                tokens_used=tokens_used,
            )

    except httpx.TimeoutException:
        latency = int((time.time() - start) * 1000)
        return LLMCallResult(
            success=False, content="", model_name=model_id, model_id=model_id,
            provider="cloudflare", latency_ms=latency, error="Timeout",
        )
    except Exception as e:
        latency = int((time.time() - start) * 1000)
        return LLMCallResult(
            success=False, content="", model_name=model_id, model_id=model_id,
            provider="cloudflare", latency_ms=latency, error=str(e),
        )


async def _call_github(
    model_id: str,
    messages: List[Dict[str, str]],
    max_tokens: int = DEFAULT_SYNTHESIS_TOKENS,
    temperature: float = 0.0,
) -> LLMCallResult:
    """Call a GitHub Models endpoint (OpenAI-compatible)."""
    start = time.time()
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{GITHUB_MODELS_BASE_URL}/chat/completions",
                headers={
                    "Authorization": f"Bearer {GITHUB_TOKEN}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model_id,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                },
            )
            latency = int((time.time() - start) * 1000)

            if response.status_code == 429:
                # Rate limited — this is a signal to fall through
                return LLMCallResult(
                    success=False, content="", model_name=model_id, model_id=model_id,
                    provider="github", latency_ms=latency, error="Rate limited (429)",
                )

            if response.status_code != 200:
                error_text = response.text[:200]
                logger.warning(f"GitHub Models {model_id} returned {response.status_code}: {error_text}")
                return LLMCallResult(
                    success=False, content="", model_name=model_id, model_id=model_id,
                    provider="github", latency_ms=latency,
                    error=f"HTTP {response.status_code}: {error_text}",
                )

            data = response.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
            usage = data.get("usage", {})
            tokens_used = usage.get("total_tokens", 0)

            return LLMCallResult(
                success=True, content=content, model_name=model_id, model_id=model_id,
                provider="github", latency_ms=latency, tokens_used=tokens_used,
            )

    except httpx.TimeoutException:
        latency = int((time.time() - start) * 1000)
        return LLMCallResult(
            success=False, content="", model_name=model_id, model_id=model_id,
            provider="github", latency_ms=latency, error="Timeout",
        )
    except Exception as e:
        latency = int((time.time() - start) * 1000)
        return LLMCallResult(
            success=False, content="", model_name=model_id, model_id=model_id,
            provider="github", latency_ms=latency, error=str(e),
        )


async def _call_openai_direct(
    model_id: str,
    messages: List[Dict[str, str]],
    max_tokens: int = DEFAULT_SYNTHESIS_TOKENS,
    temperature: float = 0.0,
) -> LLMCallResult:
    """Call OpenAI directly (legacy fallback)."""
    start = time.time()
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model_id,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                },
            )
            latency = int((time.time() - start) * 1000)
            response.raise_for_status()
            data = response.json()
            content = data["choices"][0]["message"]["content"].strip()
            tokens_used = data.get("usage", {}).get("total_tokens", 0)

            return LLMCallResult(
                success=True, content=content, model_name=model_id, model_id=model_id,
                provider="openai", latency_ms=latency, tokens_used=tokens_used,
            )
    except Exception as e:
        latency = int((time.time() - start) * 1000)
        return LLMCallResult(
            success=False, content="", model_name=model_id, model_id=model_id,
            provider="openai", latency_ms=latency, error=str(e),
        )


async def _call_anthropic_direct(
    model_id: str,
    messages: List[Dict[str, str]],
    max_tokens: int = DEFAULT_SYNTHESIS_TOKENS,
    system_prompt: str = "",
) -> LLMCallResult:
    """Call Anthropic directly (legacy fallback)."""
    start = time.time()
    try:
        body: Dict[str, Any] = {
            "model": model_id,
            "max_tokens": max_tokens,
            "messages": messages,
        }
        if system_prompt:
            body["system"] = system_prompt

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": ANTHROPIC_API_KEY,
                    "anthropic-version": "2023-06-01",
                    "Content-Type": "application/json",
                },
                json=body,
            )
            latency = int((time.time() - start) * 1000)
            response.raise_for_status()
            data = response.json()
            content = data["content"][0]["text"].strip()

            return LLMCallResult(
                success=True, content=content, model_name=model_id, model_id=model_id,
                provider="anthropic", latency_ms=latency,
            )
    except Exception as e:
        latency = int((time.time() - start) * 1000)
        return LLMCallResult(
            success=False, content="", model_name=model_id, model_id=model_id,
            provider="anthropic", latency_ms=latency, error=str(e),
        )


# ═══════════════════════════════════════════════════════════════
# SMART ROUTER
# ═══════════════════════════════════════════════════════════════


def select_tier(
    similarity: Optional[float] = None,
    table_needed: bool = False,
    is_classification: bool = False,
    force_model: Optional[str] = None,
) -> List[ModelConfig]:
    """
    Select which models to try based on the task requirements.

    Tier selection logic:
    - Classification tasks: cheapest configured model only ( Granite / Phi-4-mini )
    - similarity > 0.85: pass-through only
    - similarity 0.72-0.85: cheap + mid models
    - similarity < 0.72: all models up to capable
    - table queries: mid + capable models only
    - force_model: skip to a specific model name

    Returns:
        Ordered list of ModelConfig to try
    """
    available = [m for m in FALLBACK_CHAIN if m.enabled and m.is_configured]

    if force_model:
        available = [m for m in available if m.name == force_model]
        return available if available else FALLBACK_CHAIN[:1]

    if is_classification:
        # Classification needs the cheapest model with enough smarts
        # Prefer Cloudflare (cheapest), then GitHub (free)
        class_models = [m for m in available if m.tier in ("cheap", "free") and m.provider != Provider.PASSTHROUGH]
        return class_models[:3] if class_models else available[1:3]

    if similarity is not None and similarity > 0.85:
        # High confidence — pass-through only
        return [FALLBACK_CHAIN[0]]

    if table_needed:
        # Tables need stronger models
        table_models = [m for m in available if m.tier in ("mid", "capable", "cloud")]
        return table_models if table_models else available[3:]

    if similarity is not None and similarity < 0.72:
        # Low similarity — need all options including capable
        return available

    # Default: cheap + mid models
    default_models = [m for m in available if m.tier in ("cheap", "mid", "free")]
    return default_models if default_models else available[1:6]


async def route_llm_call(
    messages: List[Dict[str, str]],
    max_tokens: int = DEFAULT_SYNTHESIS_TOKENS,
    temperature: float = 0.0,
    similarity: Optional[float] = None,
    table_needed: bool = False,
    is_classification: bool = False,
    force_model: Optional[str] = None,
    system_prompt_for_anthropic: str = "",
) -> RouterResult:
    """
    Route an LLM call through the 9-model fallback chain.

    Tries models in order based on tier selection. Falls through on:
    - API errors (4xx, 5xx)
    - Timeouts
    - Rate limits (429)
    - Empty responses
    - Missing credentials (skipped automatically)

    Args:
        messages: OpenAI-format messages list
        max_tokens: Maximum output tokens
        temperature: Sampling temperature
        similarity: RAG similarity score for tier selection
        table_needed: Whether a table response is needed
        is_classification: Whether this is a classification task
        force_model: Force a specific model name
        system_prompt_for_anthropic: System prompt for Anthropic calls

    Returns:
        RouterResult with the first successful response
    """
    start_time = time.time()
    models_to_try = select_tier(similarity, table_needed, is_classification, force_model)

    attempts: List[Dict[str, Any]] = []
    fallback_count = 0

    for model in models_to_try:
        # Skip unconfigured models
        if not model.is_configured:
            attempts.append({
                "model": model.name,
                "provider": model.provider.value,
                "status": "skipped",
                "reason": "Not configured (missing credentials)",
            })
            fallback_count += 1
            continue

        # Skip pass-through unless it's the only option
        if model.provider == Provider.PASSTHROUGH:
            attempts.append({
                "model": model.name,
                "provider": "passthrough",
                "status": "skipped",
                "reason": "Pass-through handled upstream",
            })
            fallback_count += 1
            continue

        # Enforce max_tokens per model
        model_max = min(max_tokens, model.max_output_tokens)

        # Call the appropriate provider
        if model.provider == Provider.CLOUDFLARE:
            result = await _call_cloudflare(model.model_id, messages, model_max, temperature)
        elif model.provider == Provider.GITHUB:
            result = await _call_github(model.model_id, messages, model_max, temperature)
        elif model.provider == Provider.OPENAI:
            result = await _call_openai_direct(model.model_id, messages, model_max, temperature)
        elif model.provider == Provider.ANTHROPIC:
            result = await _call_anthropic_direct(
                model.model_id, messages, model_max, system_prompt_for_anthropic,
            )
        else:
            fallback_count += 1
            continue

        # Record attempt
        attempts.append({
            "model": model.name,
            "model_id": model.model_id,
            "provider": model.provider.value,
            "status": "success" if result.success else "failed",
            "latency_ms": result.latency_ms,
            "tokens_used": result.tokens_used,
            "error": result.error if not result.success else None,
        })

        if result.success and result.content:
            total_latency = int((time.time() - start_time) * 1000)
            logger.info(
                f"LLM Router: {model.name} ({model.provider.value}) succeeded "
                f"after {fallback_count} fallback(s) in {total_latency}ms"
            )
            return RouterResult(
                content=result.content,
                model_name=model.name,
                model_id=model.model_id,
                provider=model.provider.value,
                fallback_count=fallback_count,
                total_latency_ms=total_latency,
                attempts=attempts,
            )

        # Failed — fall through to next model
        logger.warning(
            f"LLM Router: {model.name} ({model.provider.value}) failed: {result.error}. "
            f"Trying next model..."
        )
        fallback_count += 1

    # All models failed
    total_latency = int((time.time() - start_time) * 1000)
    logger.error(f"LLM Router: All {len(models_to_try)} models failed after {total_latency}ms")

    return RouterResult(
        content="[ALL_LLM_FAILED] No model could generate a response. Check API keys and rate limits.",
        model_name="none",
        model_id="none",
        provider="none",
        fallback_count=fallback_count,
        total_latency_ms=total_latency,
        attempts=attempts,
    )


# ═══════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════


async def classify_with_router(query: str) -> Dict[str, Any]:
    """
    Classify a query using the cheapest available model.

    Returns:
        Dict with "route", "reason", "domain_hint" keys
    """
    system_prompt = """You are a query router for a research AI. Classify the query as needing:
- "rag": if it can likely be answered from a pre-loaded academic/research corpus
- "live": if it needs current/real-time data from the web

Respond ONLY with valid JSON: {"route": "rag"|"live", "reason": "...", "domain_hint": "..."}"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query},
    ]

    result = await route_llm_call(
        messages=messages,
        max_tokens=50,
        temperature=0.0,
        is_classification=True,
    )

    if result.content.startswith("[ALL_LLM_FAILED]"):
        return {"route": "rag", "reason": "LLM fallback failed, defaulting to RAG", "domain_hint": ""}

    try:
        # Clean up response
        content = result.content.strip()
        if content.startswith("```"):
            content = re.sub(r"^```(?:json)?\s*", "", content)
            content = re.sub(r"\s*```$", "", content)
        parsed = json.loads(content)
        return {
            "route": parsed.get("route", "rag"),
            "reason": parsed.get("reason", "LLM classified"),
            "domain_hint": parsed.get("domain_hint", ""),
            "model_used": result.model_name,
        }
    except json.JSONDecodeError:
        return {"route": "rag", "reason": f"LLM output not JSON: {result.content[:50]}", "domain_hint": ""}


async def synthesize_with_router(
    query: str,
    context: str,
    max_tokens: int = DEFAULT_SYNTHESIS_TOKENS,
    similarity: Optional[float] = None,
    table_needed: bool = False,
    system_prompt: str = "",
) -> RouterResult:
    """
    Synthesize an answer using the tiered fallback router.

    Selects models based on:
    - similarity score (higher = cheaper model)
    - table_needed (stronger model required)
    - availability (skips unconfigured providers)

    Args:
        query: User query
        context: Formatted context string
        max_tokens: Max output tokens
        similarity: RAG similarity score
        table_needed: Whether table format is needed
        system_prompt: APEX system prompt

    Returns:
        RouterResult with synthesized answer
    """
    if not system_prompt:
        from .synthesizer import APEX_SYSTEM_PROMPT
        system_prompt = APEX_SYSTEM_PROMPT

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Query: {query}\n\nContext:\n{context}"},
    ]

    return await route_llm_call(
        messages=messages,
        max_tokens=max_tokens,
        temperature=0.0,
        similarity=similarity,
        table_needed=table_needed,
        system_prompt_for_anthropic=system_prompt,
    )


def get_router_status() -> Dict[str, Any]:
    """Get the current status of all models in the fallback chain."""
    models = []
    for m in FALLBACK_CHAIN:
        models.append({
            "name": m.name,
            "provider": m.provider.value,
            "model_id": m.model_id,
            "tier": m.tier,
            "configured": m.is_configured,
            "supports_tools": m.supports_tools,
            "price_input_per_m": m.price_input_per_m,
            "price_output_per_m": m.price_output_per_m,
            "context_window": m.context_window,
        })
    return {
        "total_models": len(FALLBACK_CHAIN),
        "configured_models": sum(1 for m in FALLBACK_CHAIN if m.is_configured),
        "models": models,
        "cloudflare_configured": bool(CLOUDFLARE_API_TOKEN and CLOUDFLARE_ACCOUNT_ID),
        "github_configured": bool(GITHUB_TOKEN),
        "openai_configured": bool(OPENAI_API_KEY),
        "anthropic_configured": bool(ANTHROPIC_API_KEY),
    }
