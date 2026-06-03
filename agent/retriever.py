"""
Retriever — hybrid search combining vector similarity + BM25 keyword boost.

Implements:
1. Vector similarity search (pgvector cosine)
2. BM25 keyword search (PostgreSQL tsvector/tsquery)
3. Score fusion (Reciprocal Rank Fusion)
4. Optional reranking via cross-encoder or Cohere
5. Token budget management
"""

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import asyncpg
import httpx

from .query_classifier import classify_query, ClassificationResult

logger = logging.getLogger(__name__)

# ── Configuration ──
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://apex:apex_secret@localhost:5432/apex_db")
CLOUDFLARE_API_TOKEN = os.getenv("CLOUDFLARE_API_TOKEN", "")
CLOUDFLARE_ACCOUNT_ID = os.getenv("CLOUDFLARE_ACCOUNT_ID", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
COHERE_API_KEY = os.getenv("COHERE_API_KEY", "")
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "cloudflare")  # "cloudflare" or "openai"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "@cf/baai/bge-base-en-v1.5")
EMBEDDING_DIMENSIONS = int(os.getenv("EMBEDDING_DIMENSIONS", "768"))
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
OPENAI_EMBEDDING_DIMENSIONS = int(os.getenv("OPENAI_EMBEDDING_DIMENSIONS", "1536"))

RAG_TOP_K = int(os.getenv("RAG_TOP_K", "5"))
RAG_FINAL_K = int(os.getenv("RAG_FINAL_K", "3"))
MAX_RAG_CONTEXT_TOKENS = int(os.getenv("MAX_RAG_CONTEXT_TOKENS", "2000"))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.72"))

CHARS_PER_TOKEN = 4  # Approximate


@dataclass
class RetrievedChunk:
    """A retrieved document chunk with relevance scoring."""
    id: str
    source_url: str
    source_tier: str
    domain: str
    doc_type: str
    title: str
    authors: List[str]
    raw_text: str
    metadata: dict
    chunk_index: int
    total_chunks: int
    similarity_score: float = 0.0
    keyword_score: float = 0.0
    fused_score: float = 0.0
    token_count: int = 0


# Auto-detected CF account ID cache
_detected_cf_account_id: Optional[str] = None


async def _detect_cf_account_id() -> Optional[str]:
    """Auto-detect Cloudflare Account ID from API token."""
    global _detected_cf_account_id
    if _detected_cf_account_id:
        return _detected_cf_account_id
    if CLOUDFLARE_ACCOUNT_ID:
        _detected_cf_account_id = CLOUDFLARE_ACCOUNT_ID
        return CLOUDFLARE_ACCOUNT_ID
    if not CLOUDFLARE_API_TOKEN:
        return None
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                "https://api.cloudflare.com/client/v4/accounts",
                headers={"Authorization": f"Bearer {CLOUDFLARE_API_TOKEN}"},
            )
            if response.status_code == 200:
                accounts = response.json().get("result", [])
                if accounts:
                    _detected_cf_account_id = accounts[0]["id"]
                    return _detected_cf_account_id
    except Exception as e:
        logger.warning(f"Failed to auto-detect CF Account ID: {e}")
    return None


async def _get_embedding_cloudflare(query: str) -> List[float]:
    """Generate embedding using Cloudflare Workers AI (primary, free)."""
    if not CLOUDFLARE_API_TOKEN:
        return []

    account_id = await _detect_cf_account_id()
    if not account_id:
        return []

    base_url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1"

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.post(
                f"{base_url}/embeddings",
                headers={
                    "Authorization": f"Bearer {CLOUDFLARE_API_TOKEN}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": EMBEDDING_MODEL,
                    "input": [query],
                },
            )
            response.raise_for_status()
            return response.json()["data"][0]["embedding"]
    except Exception as e:
        logger.error(f"CF embedding failed: {e}")
        return []


async def _get_embedding_openai(query: str) -> List[float]:
    """Generate embedding using OpenAI (fallback, paid, may be region-restricted)."""
    if not OPENAI_API_KEY:
        return []

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.post(
                "https://api.openai.com/v1/embeddings",
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": OPENAI_EMBEDDING_MODEL,
                    "input": [query],
                    "dimensions": OPENAI_EMBEDDING_DIMENSIONS,
                },
            )
            response.raise_for_status()
            return response.json()["data"][0]["embedding"]
    except Exception as e:
        logger.error(f"OpenAI embedding failed: {e}")
        return []


async def _get_embedding(query: str) -> List[float]:
    """Generate embedding for a query string.

    Primary: Cloudflare Workers AI bge-base-en-v1.5 (768 dims, free)
    Fallback: OpenAI text-embedding-3-small (1536 dims, paid)
    """
    if EMBEDDING_PROVIDER == "cloudflare":
        embedding = await _get_embedding_cloudflare(query)
        if embedding:
            return embedding
        # Try OpenAI fallback
        if OPENAI_API_KEY:
            logger.info("CF embedding failed, trying OpenAI fallback")
            embedding = await _get_embedding_openai(query)
            if embedding:
                return embedding
    else:
        embedding = await _get_embedding_openai(query)
        if embedding:
            return embedding
        # Try CF fallback
        if CLOUDFLARE_API_TOKEN:
            logger.info("OpenAI embedding failed, trying CF fallback")
            embedding = await _get_embedding_cloudflare(query)
            if embedding:
                return embedding

    logger.error("All embedding providers failed. Returning zero vector.")
    return [0.0] * EMBEDDING_DIMENSIONS


async def vector_search(
    query_embedding: List[float],
    top_k: int = RAG_TOP_K,
    domain_filter: Optional[str] = None,
    tier_filter: Optional[str] = None,
    doc_type_filter: Optional[str] = None,
) -> List[RetrievedChunk]:
    """
    Perform vector similarity search in pgvector.

    Args:
        query_embedding: Query vector
        top_k: Number of results
        domain_filter: Optional domain filter
        tier_filter: Optional tier filter
        doc_type_filter: Optional doc type filter

    Returns:
        List of RetrievedChunk objects sorted by similarity
    """
    vector_str = "[" + ",".join(str(v) for v in query_embedding) + "]"

    conn = await asyncpg.connect(DATABASE_URL)
    try:
        rows = await conn.fetch(
            "SELECT * FROM search_documents($1::vector, $2, $3, $4, $5, $6)",
            vector_str,
            0.5,  # Lower threshold for initial retrieval
            top_k,
            domain_filter,
            tier_filter,
            doc_type_filter,
        )

        chunks = []
        for row in rows:
            metadata = json.loads(row["metadata"]) if row["metadata"] else {}
            chunks.append(RetrievedChunk(
                id=str(row["id"]),
                source_url=row["source_url"],
                source_tier=row["source_tier"],
                domain=row["domain"],
                doc_type=row["doc_type"],
                title=row["title"] or "",
                authors=list(row["authors"]) if row["authors"] else [],
                raw_text=row["raw_text"],
                metadata=metadata,
                chunk_index=row["chunk_index"],
                total_chunks=row["total_chunks"],
                similarity_score=row["similarity"],
                token_count=max(1, len(row["raw_text"]) // CHARS_PER_TOKEN),
            ))

        logger.debug(f"Vector search returned {len(chunks)} results")
        return chunks

    finally:
        await conn.close()


async def keyword_search(
    query: str,
    top_k: int = RAG_TOP_K,
    tier_filter: Optional[str] = None,
) -> List[RetrievedChunk]:
    """
    Perform BM25-like keyword search using PostgreSQL tsvector.

    Args:
        query: Search query
        top_k: Number of results
        tier_filter: Optional tier filter

    Returns:
        List of RetrievedChunk objects sorted by keyword relevance
    """
    conn = await asyncpg.connect(DATABASE_URL)
    try:
        rows = await conn.fetch(
            "SELECT * FROM keyword_search($1, $2, $3)",
            query,
            top_k,
            tier_filter,
        )

        chunks = []
        for row in rows:
            chunks.append(RetrievedChunk(
                id=str(row["id"]),
                source_url=row["source_url"],
                source_tier=row["source_tier"],
                domain=row["domain"],
                doc_type="",
                title=row["title"] or "",
                authors=[],
                raw_text=row["raw_text"],
                metadata={},
                chunk_index=0,
                total_chunks=1,
                keyword_score=row["rank"],
                token_count=max(1, len(row["raw_text"]) // CHARS_PER_TOKEN),
            ))

        logger.debug(f"Keyword search returned {len(chunks)} results")
        return chunks

    finally:
        await conn.close()


def reciprocal_rank_fusion(
    vector_results: List[RetrievedChunk],
    keyword_results: List[RetrievedChunk],
    k: int = 60,
) -> List[RetrievedChunk]:
    """
    Combine vector and keyword results using Reciprocal Rank Fusion.

    RRF(score) = sum(1 / (k + rank_i)) for each result list

    Args:
        vector_results: Results from vector search
        keyword_results: Results from keyword search
        k: RRF constant (default 60)

    Returns:
        Fused and sorted list of RetrievedChunk objects
    """
    # Build score map keyed by document ID
    score_map: dict[str, float] = {}
    chunk_map: dict[str, RetrievedChunk] = {}

    # Score vector results
    for rank, chunk in enumerate(vector_results, 1):
        if chunk.id not in score_map:
            score_map[chunk.id] = 0.0
            chunk_map[chunk.id] = chunk
        score_map[chunk.id] += 1.0 / (k + rank)
        # Keep highest similarity score
        if chunk.similarity_score > chunk_map[chunk.id].similarity_score:
            chunk_map[chunk.id].similarity_score = chunk.similarity_score

    # Score keyword results
    for rank, chunk in enumerate(keyword_results, 1):
        if chunk.id not in score_map:
            score_map[chunk.id] = 0.0
            chunk_map[chunk.id] = chunk
        score_map[chunk.id] += 1.0 / (k + rank)
        if chunk.keyword_score > chunk_map[chunk.id].keyword_score:
            chunk_map[chunk.id].keyword_score = chunk.keyword_score

    # Apply fused scores and sort
    for chunk_id, fused_score in score_map.items():
        chunk_map[chunk_id].fused_score = fused_score

    results = sorted(chunk_map.values(), key=lambda c: c.fused_score, reverse=True)
    return results


async def rerank_cohere(
    query: str,
    chunks: List[RetrievedChunk],
    top_n: int = RAG_FINAL_K,
) -> List[RetrievedChunk]:
    """
    Rerank results using Cohere's rerank API, with CF Workers AI fallback.

    Strategy:
    1. Cohere rerank (if API key available) — dedicated reranker, best quality
    2. Cloudflare Workers AI LLM-as-reranker (free, uses your CF token) —
       asks a small model to score relevance of each chunk
    3. Fallback to RRF fusion scores if both fail

    Args:
        query: Original query
        chunks: Chunks to rerank
        top_n: Number of results to return

    Returns:
        Reranked list of RetrievedChunk objects
    """
    if not chunks:
        return chunks[:top_n]

    # Strategy 1: Cohere rerank
    if COHERE_API_KEY:
        try:
            documents = [chunk.raw_text for chunk in chunks]

            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.post(
                    "https://api.cohere.ai/v1/rerank",
                    headers={
                        "Authorization": f"Bearer {COHERE_API_KEY}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": "rerank-english-v3.0",
                        "query": query,
                        "documents": documents,
                        "top_n": top_n,
                    },
                )
                response.raise_for_status()
                data = response.json()

            reranked = []
            for result in data["results"]:
                idx = result["index"]
                if idx < len(chunks):
                    chunk = chunks[idx]
                    chunk.fused_score = result["relevance_score"]
                    reranked.append(chunk)

            logger.info(f"Cohere rerank: {len(reranked)} results")
            return reranked

        except Exception as e:
            logger.warning(f"Cohere rerank failed: {e}. Trying CF AI reranker.")

    # Strategy 2: Cloudflare Workers AI LLM-as-reranker
    cf_result = await _rerank_with_cloudflare_ai(query, chunks, top_n)
    if cf_result:
        return cf_result

    # Strategy 3: Fallback to fusion scores
    logger.info("All rerankers failed, using RRF fusion scores")
    return sorted(chunks, key=lambda c: c.fused_score, reverse=True)[:top_n]


async def _rerank_with_cloudflare_ai(
    query: str,
    chunks: List[RetrievedChunk],
    top_n: int,
) -> Optional[List[RetrievedChunk]]:
    """
    Use a Cloudflare Workers AI model as a reranker.

    Sends each chunk to a small CF model with a relevance scoring prompt.
    Uses Granite-4.0-Micro (cheapest) for cost efficiency.

    Returns None if CF AI is not available, so caller can fall through.
    """
    CF_TOKEN = os.getenv("CLOUDFLARE_API_TOKEN", "")
    CF_ACCOUNT_ID = os.getenv("CLOUDFLARE_ACCOUNT_ID", "")

    if not CF_TOKEN:
        return None

    # Auto-detect account ID if needed
    if not CF_ACCOUNT_ID:
        try:
            from .llm_router import detect_cloudflare_account_id
            CF_ACCOUNT_ID = await detect_cloudflare_account_id(CF_TOKEN) or ""
        except Exception:
            pass

    if not CF_ACCOUNT_ID:
        return None

    base_url = f"https://api.cloudflare.com/client/v4/accounts/{CF_ACCOUNT_ID}/ai/v1"

    try:
        # Use Granite for cheap reranking — score each chunk's relevance
        scored_chunks = []

        for chunk in chunks:
            prompt = f"""Rate the relevance of this text to the query on a scale of 0.0 to 1.0.
Respond with ONLY a number, no explanation.

Query: {query}

Text: {chunk.raw_text[:500]}

Relevance score:"""

            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    f"{base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {CF_TOKEN}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": "@cf/ibm-granite/granite-4.0-h-micro",
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": 10,
                        "temperature": 0.0,
                    },
                )

                if response.status_code == 200:
                    data = response.json()
                    content = data.get("choices", [{}])[0].get("message", {}).get("content", "0.5")
                    # Parse score
                    try:
                        score = float(re.search(r'[0-9]*\.?[0-9]+', content).group())
                        score = max(0.0, min(1.0, score))
                    except (AttributeError, ValueError):
                        score = chunk.fused_score
                else:
                    score = chunk.fused_score

            chunk.fused_score = score
            scored_chunks.append(chunk)

        reranked = sorted(scored_chunks, key=lambda c: c.fused_score, reverse=True)[:top_n]
        logger.info(f"CF AI rerank: {len(reranked)} results using Granite")
        return reranked

    except Exception as e:
        logger.warning(f"CF AI rerank failed: {e}")
        return None


def apply_token_budget(
    chunks: List[RetrievedChunk],
    max_tokens: int = MAX_RAG_CONTEXT_TOKENS,
) -> List[RetrievedChunk]:
    """
    Truncate chunk list to fit within the token budget.

    If a document is too large, returns only the most relevant chunk + metadata.

    Args:
        chunks: Retrieved chunks
        max_tokens: Maximum total tokens

    Returns:
        Budget-compliant list of RetrievedChunk objects
    """
    budget_chunks = []
    total_tokens = 0

    for chunk in chunks:
        if total_tokens + chunk.token_count <= max_tokens:
            budget_chunks.append(chunk)
            total_tokens += chunk.token_count
        else:
            # Partial inclusion: truncate text to fit remaining budget
            remaining = max_tokens - total_tokens
            if remaining > 50:  # Only include if meaningful text remains
                truncated_text = chunk.raw_text[:remaining * CHARS_PER_TOKEN]
                chunk.raw_text = truncated_text
                chunk.token_count = remaining
                budget_chunks.append(chunk)
            break

    logger.debug(f"Token budget: {total_tokens}/{max_tokens} tokens across {len(budget_chunks)} chunks")
    return budget_chunks


def apply_source_hierarchy(chunks: List[RetrievedChunk]) -> List[RetrievedChunk]:
    """
    Apply source hierarchy: P1 > P2 > P3.
    If P1 sources exist, deprioritize P3 unless flagged as counter-evidence.

    Args:
        chunks: Retrieved chunks

    Returns:
        Reordered chunks respecting source hierarchy
    """
    tier_priority = {"P1": 0, "P2": 1, "P3": 2, "UNV": 3}

    has_p1 = any(c.source_tier == "P1" for c in chunks)

    if has_p1:
        # Boost P1, keep P2, deprioritize P3/UNV
        for chunk in chunks:
            if chunk.source_tier == "P3":
                # Check if flagged as counter-evidence in metadata
                if chunk.metadata.get("counter_evidence", False):
                    continue  # Keep P3 counter-evidence
                chunk.fused_score *= 0.5  # Penalize P3 when P1 exists
            elif chunk.source_tier == "UNV":
                chunk.fused_score *= 0.3
            elif chunk.source_tier == "P1":
                chunk.fused_score *= 1.5  # Boost P1

    return sorted(chunks, key=lambda c: c.fused_score, reverse=True)


async def retrieve(
    query: str,
    top_k: int = RAG_TOP_K,
    final_k: int = RAG_FINAL_K,
    domain_filter: Optional[str] = None,
    tier_filter: Optional[str] = None,
    use_reranker: bool = True,
) -> Tuple[List[RetrievedChunk], float]:
    """
    Full hybrid retrieval pipeline.

    1. Generate query embedding
    2. Vector search + keyword search in parallel
    3. Reciprocal Rank Fusion
    4. Source hierarchy enforcement
    5. Optional reranking
    6. Token budget enforcement

    If DATABASE_URL is not set or database is unreachable, returns
    empty results gracefully (live-only mode).

    Args:
        query: User query
        top_k: Initial retrieval count
        final_k: Final result count after reranking
        domain_filter: Optional domain filter
        tier_filter: Optional tier filter
        use_reranker: Whether to use Cohere reranker

    Returns:
        Tuple of (final chunks, average similarity score)
    """
    # Check if database is available
    if not DATABASE_URL or DATABASE_URL == "postgresql://apex:apex_secret@localhost:5432/apex_db":
        # Only skip if env var is not set AND we're not on localhost
        if not os.getenv("DATABASE_URL"):
            logger.info("No DATABASE_URL configured — RAG disabled, using live-only mode")
            return [], 0.0

    try:
        # Step 1: Generate query embedding
        query_embedding = await _get_embedding(query)

        # Step 2: Parallel vector + keyword search
        vector_task = vector_search(query_embedding, top_k, domain_filter, tier_filter)
        keyword_task = keyword_search(query, top_k, tier_filter)

        vector_results, keyword_results = await asyncio.gather(
            vector_task, keyword_task, return_exceptions=True
        )

        # Handle individual search failures gracefully
        if isinstance(vector_results, Exception):
            logger.warning(f"Vector search failed: {vector_results}")
            vector_results = []
        if isinstance(keyword_results, Exception):
            logger.warning(f"Keyword search failed: {keyword_results}")
            keyword_results = []

        # Step 3: Reciprocal Rank Fusion
        fused = reciprocal_rank_fusion(vector_results, keyword_results)

        # Step 4: Source hierarchy
        fused = apply_source_hierarchy(fused)

        # Step 5: Reranking
        if use_reranker and COHERE_API_KEY:
            fused = await rerank_cohere(query, fused, final_k)
        else:
            fused = fused[:final_k]

        # Step 6: Token budget
        final_chunks = apply_token_budget(fused, MAX_RAG_CONTEXT_TOKENS)

        # Calculate average similarity
        avg_similarity = (
            sum(c.similarity_score for c in final_chunks) / len(final_chunks)
            if final_chunks else 0.0
        )

        logger.info(f"Retrieval complete: {len(final_chunks)} chunks, avg similarity={avg_similarity:.3f}")
        return final_chunks, avg_similarity

    except Exception as e:
        logger.warning(f"RAG retrieval failed, falling back to live-only: {e}")
        return [], 0.0
