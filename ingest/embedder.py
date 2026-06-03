"""
Embedder — generates vector embeddings for document chunks.

Supports:
- Cloudflare Workers AI bge-base-en-v1.5 (768 dim) — primary, free with CF token
- OpenAI text-embedding-3-small (1536 dim) — fallback if CF unavailable

Includes batching, rate limiting, and error handling.
"""

import asyncio
import logging
import os
from typing import List, Optional

import httpx
import asyncpg

logger = logging.getLogger(__name__)

# ── Configuration ──
CLOUDFLARE_API_TOKEN = os.getenv("CLOUDFLARE_API_TOKEN", "")
CLOUDFLARE_ACCOUNT_ID = os.getenv("CLOUDFLARE_ACCOUNT_ID", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Default: CF Workers AI bge-base (768 dims) — free, same token as LLMs
# Fallback: OpenAI text-embedding-3-small (1536 dims) — paid, region-restricted
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "cloudflare")  # "cloudflare" or "openai"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "@cf/baai/bge-base-en-v1.5")
EMBEDDING_DIMENSIONS = int(os.getenv("EMBEDDING_DIMENSIONS", "768"))
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
OPENAI_EMBEDDING_DIMENSIONS = int(os.getenv("OPENAI_EMBEDDING_DIMENSIONS", "1536"))

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://apex:apex_secret@localhost:5432/apex_db")

BATCH_SIZE = 50  # Max texts per API call (CF limit)
MAX_RETRIES = 3
RETRY_DELAY = 1.0  # seconds

# Auto-detected account ID cache
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
                    logger.info(f"Auto-detected CF Account ID for embeddings: {_detected_cf_account_id}")
                    return _detected_cf_account_id
    except Exception as e:
        logger.warning(f"Failed to auto-detect CF Account ID: {e}")
    return None


async def embed_texts_cloudflare(
    texts: List[str],
    model: str = EMBEDDING_MODEL,
) -> List[List[float]]:
    """
    Generate embeddings using Cloudflare Workers AI.

    Available models:
    - @cf/baai/bge-small-en-v1.5  (384 dims, fastest)
    - @cf/baai/bge-base-en-v1.5   (768 dims, best balance) ← default
    - @cf/baai/bge-large-en-v1.5  (1024 dims, highest quality)

    Args:
        texts: List of text strings to embed
        model: CF Workers AI embedding model name

    Returns:
        List of embedding vectors
    """
    if not texts:
        return []

    if not CLOUDFLARE_API_TOKEN:
        logger.warning("No CLOUDFLARE_API_TOKEN set. Returning zero vectors.")
        return [[0.0] * EMBEDDING_DIMENSIONS for _ in texts]

    account_id = await _detect_cf_account_id()
    if not account_id:
        logger.warning("Could not determine CF Account ID. Returning zero vectors.")
        return [[0.0] * EMBEDDING_DIMENSIONS for _ in texts]

    base_url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1"
    all_embeddings = []

    # Process in batches
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]

        for attempt in range(MAX_RETRIES):
            try:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    response = await client.post(
                        f"{base_url}/embeddings",
                        headers={
                            "Authorization": f"Bearer {CLOUDFLARE_API_TOKEN}",
                            "Content-Type": "application/json",
                        },
                        json={
                            "model": model,
                            "input": batch,
                        },
                    )
                    response.raise_for_status()
                    data = response.json()

                    # Sort by index to maintain order
                    embeddings = [None] * len(batch)
                    for item in data["data"]:
                        embeddings[item["index"]] = item["embedding"]

                    all_embeddings.extend(embeddings)
                    logger.debug(f"Embedded batch {i // BATCH_SIZE + 1}: {len(batch)} texts via CF")
                    break

            except httpx.HTTPStatusError as e:
                logger.warning(f"CF Embedding API error (attempt {attempt + 1}): {e.response.status_code}")
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(RETRY_DELAY * (attempt + 1))
                else:
                    logger.error(f"Failed to embed batch after {MAX_RETRIES} attempts")
                    all_embeddings.extend([[0.0] * EMBEDDING_DIMENSIONS for _ in batch])

            except Exception as e:
                logger.error(f"Unexpected CF embedding error: {e}")
                all_embeddings.extend([[0.0] * EMBEDDING_DIMENSIONS for _ in batch])
                break

    return all_embeddings


async def embed_texts_openai(
    texts: List[str],
    model: str = OPENAI_EMBEDDING_MODEL,
    dimensions: int = OPENAI_EMBEDDING_DIMENSIONS,
) -> List[List[float]]:
    """
    Generate embeddings using OpenAI's embedding API (fallback).

    Note: OpenAI may be region-restricted. CF Workers AI is the primary provider.

    Args:
        texts: List of text strings to embed
        model: OpenAI embedding model name
        dimensions: Output vector dimensions

    Returns:
        List of embedding vectors
    """
    if not texts:
        return []

    if not OPENAI_API_KEY:
        logger.warning("No OPENAI_API_KEY set. Returning zero vectors.")
        return [[0.0] * dimensions for _ in texts]

    all_embeddings = []

    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]

        for attempt in range(MAX_RETRIES):
            try:
                async with httpx.AsyncClient(timeout=60.0) as client:
                    response = await client.post(
                        "https://api.openai.com/v1/embeddings",
                        headers={
                            "Authorization": f"Bearer {OPENAI_API_KEY}",
                            "Content-Type": "application/json",
                        },
                        json={
                            "model": model,
                            "input": batch,
                            "dimensions": dimensions,
                        },
                    )
                    response.raise_for_status()
                    data = response.json()

                    embeddings = [None] * len(batch)
                    for item in data["data"]:
                        embeddings[item["index"]] = item["embedding"]

                    all_embeddings.extend(embeddings)
                    logger.debug(f"Embedded batch {i // BATCH_SIZE + 1}: {len(batch)} texts via OpenAI")
                    break

            except httpx.HTTPStatusError as e:
                logger.warning(f"OpenAI Embedding API error (attempt {attempt + 1}): {e.response.status_code}")
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(RETRY_DELAY * (attempt + 1))
                else:
                    logger.error(f"Failed to embed batch after {MAX_RETRIES} attempts")
                    all_embeddings.extend([[0.0] * dimensions for _ in batch])

            except Exception as e:
                logger.error(f"Unexpected OpenAI embedding error: {e}")
                all_embeddings.extend([[0.0] * dimensions for _ in batch])
                break

    return all_embeddings


async def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Generate embeddings using the configured provider.

    Primary: Cloudflare Workers AI (free, same token as LLMs)
    Fallback: OpenAI (paid, may be region-restricted)

    Args:
        texts: List of text strings to embed

    Returns:
        List of embedding vectors
    """
    if EMBEDDING_PROVIDER == "cloudflare":
        result = await embed_texts_cloudflare(texts)
        # If CF fails (all zeros), try OpenAI as fallback
        if result and all(all(v == 0.0 for v in emb) for emb in result):
            if OPENAI_API_KEY:
                logger.info("CF embeddings returned zeros, trying OpenAI fallback")
                return await embed_texts_openai(texts)
        return result
    else:
        return await embed_texts_openai(texts)


async def embed_single(text: str) -> List[float]:
    """Embed a single text string."""
    result = await embed_texts([text])
    return result[0] if result else [0.0] * EMBEDDING_DIMENSIONS


async def embed_unembedded_chunks(batch_size: int = 50) -> int:
    """
    Find chunks without embeddings and generate them.

    Args:
        batch_size: Number of chunks to process per batch

    Returns:
        Number of chunks embedded
    """
    conn = await asyncpg.connect(DATABASE_URL)

    try:
        rows = await conn.fetch(
            "SELECT id, raw_text FROM documents WHERE content_vector IS NULL ORDER BY created_at LIMIT $1",
            batch_size,
        )

        if not rows:
            logger.info("No unembedded chunks found.")
            return 0

        logger.info(f"Found {len(rows)} unembedded chunks")

        texts = [row["raw_text"] for row in rows]
        ids = [row["id"] for row in rows]

        embeddings = await embed_texts(texts)

        updated = 0
        for doc_id, embedding in zip(ids, embeddings):
            if embedding and any(v != 0.0 for v in embedding):
                vector_str = "[" + ",".join(str(v) for v in embedding) + "]"
                await conn.execute(
                    "UPDATE documents SET content_vector = $1::vector WHERE id = $2",
                    vector_str,
                    doc_id,
                )
                updated += 1

        logger.info(f"Embedded {updated}/{len(rows)} chunks")
        return updated

    finally:
        await conn.close()


async def embed_and_upsert(
    source_url: str,
    source_tier: str,
    domain: str,
    doc_type: str,
    title: Optional[str],
    authors: Optional[List[str]],
    published_date: Optional[str],
    chunks: list,  # List of Chunk objects from chunker
    metadata: dict = None,
) -> int:
    """
    Embed chunks and upsert into database.

    Args:
        source_url: Source document URL
        source_tier: P1/P2/P3/UNV
        domain: Source domain
        doc_type: paper/article/legal/etc
        title: Document title
        authors: List of author strings
        published_date: Publication date string
        chunks: List of Chunk objects
        metadata: Additional metadata dict

    Returns:
        Number of chunks upserted
    """
    import json

    conn = await asyncpg.connect(DATABASE_URL)

    try:
        texts = [chunk.text for chunk in chunks]
        embeddings = await embed_texts(texts)

        upserted = 0
        for chunk, embedding in zip(chunks, embeddings):
            vector_str = "[" + ",".join(str(v) for v in embedding) + "]" if embedding else None

            chunk_metadata = dict(metadata or {})
            chunk_metadata.update(chunk.metadata)

            await conn.execute(
                """
                INSERT INTO documents (
                    source_url, source_tier, domain, doc_type,
                    published_date, title, authors, raw_text,
                    content_vector, metadata, chunk_index, total_chunks
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9::vector, $10::jsonb, $11, $12)
                ON CONFLICT (source_url, chunk_index) DO UPDATE SET
                    raw_text = EXCLUDED.raw_text,
                    content_vector = EXCLUDED.content_vector,
                    metadata = EXCLUDED.metadata,
                    total_chunks = EXCLUDED.total_chunks,
                    updated_at = NOW()
                """,
                source_url,
                source_tier,
                domain,
                doc_type,
                published_date,
                title,
                authors or [],
                chunk.text,
                vector_str,
                json.dumps(chunk_metadata),
                chunk.chunk_index,
                chunk.total_chunks,
            )
            upserted += 1

        logger.info(f"Upserted {upserted} chunks for {source_url}")
        return upserted

    finally:
        await conn.close()
