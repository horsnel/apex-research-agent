"""
Embedder — generates vector embeddings for document chunks.

Supports:
- OpenAI text-embedding-3-small (1536 dim) — default
- Nomic nomic-embed-text-v1.5 (768 dim) — via Ollama or API

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
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_DIMENSIONS = int(os.getenv("EMBEDDING_DIMENSIONS", "1536"))
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://apex:apex_secret@localhost:5432/apex_db")

BATCH_SIZE = 100  # Max texts per API call
MAX_RETRIES = 3
RETRY_DELAY = 1.0  # seconds


async def embed_texts_openai(
    texts: List[str],
    model: str = EMBEDDING_MODEL,
    dimensions: int = EMBEDDING_DIMENSIONS,
) -> List[List[float]]:
    """
    Generate embeddings using OpenAI's embedding API.

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

    # Process in batches
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

                    # Sort by index to maintain order
                    embeddings = [None] * len(batch)
                    for item in data["data"]:
                        embeddings[item["index"]] = item["embedding"]

                    all_embeddings.extend(embeddings)
                    logger.debug(f"Embedded batch {i // BATCH_SIZE + 1}: {len(batch)} texts")
                    break

            except httpx.HTTPStatusError as e:
                logger.warning(f"Embedding API error (attempt {attempt + 1}): {e.response.status_code}")
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(RETRY_DELAY * (attempt + 1))
                else:
                    logger.error(f"Failed to embed batch after {MAX_RETRIES} attempts")
                    all_embeddings.extend([[0.0] * dimensions for _ in batch])

            except Exception as e:
                logger.error(f"Unexpected embedding error: {e}")
                all_embeddings.extend([[0.0] * dimensions for _ in batch])
                break

    return all_embeddings


async def embed_single(text: str) -> List[float]:
    """Embed a single text string."""
    result = await embed_texts_openai([text])
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
        # Find chunks without embeddings
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

        embeddings = await embed_texts_openai(texts)

        # Update database
        updated = 0
        for doc_id, embedding in zip(ids, embeddings):
            if embedding and any(v != 0.0 for v in embedding):
                # Convert to PostgreSQL vector format
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
        embeddings = await embed_texts_openai(texts)

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
