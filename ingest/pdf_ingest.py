"""
PDF Ingest — downloads and parses PDF documents for chunking and embedding.

Supports:
- PDF from URL
- PDF from local file path
- Text extraction with PyMuPDF (fitz)
- Metadata extraction
"""

import asyncio
import logging
import os
import tempfile
from datetime import datetime
from typing import List, Optional
from urllib.parse import urlparse

import httpx

from .chunker import chunk_text
from .embedder import embed_and_upsert
from .html_cleaner import clean_html

logger = logging.getLogger(__name__)

# Try importing PyMuPDF
try:
    import fitz  # PyMuPDF
    HAS_FITZ = True
except ImportError:
    HAS_FITZ = False
    logger.warning("PyMuPDF (fitz) not installed. PDF ingestion will be limited.")


def extract_text_from_pdf(file_path: str) -> dict:
    """
    Extract text and metadata from a PDF file.

    Args:
        file_path: Path to the PDF file

    Returns:
        Dict with 'text', 'title', 'authors', 'metadata' keys
    """
    if not HAS_FITZ:
        logger.error("PyMuPDF not installed. Cannot extract PDF text.")
        return {"text": "", "title": "", "authors": [], "metadata": {}}

    try:
        doc = fitz.open(file_path)
    except Exception as e:
        logger.error(f"Failed to open PDF {file_path}: {e}")
        return {"text": "", "title": "", "authors": [], "metadata": {}}

    # Extract metadata
    meta = doc.metadata or {}
    title = meta.get("title", "")
    authors_str = meta.get("author", "")
    authors = [a.strip() for a in authors_str.split(";") if a.strip()] if authors_str else []

    # Extract text from all pages
    pages = []
    for page_num in range(len(doc)):
        try:
            page = doc[page_num]
            page_text = page.get_text("text")
            if page_text.strip():
                pages.append(f"--- Page {page_num + 1} ---\n{page_text}")
        except Exception as e:
            logger.warning(f"Failed to extract page {page_num}: {e}")

    doc.close()

    full_text = "\n\n".join(pages)

    # Clean common PDF artifacts
    import re
    full_text = re.sub(r'\x0c', '\n', full_text)  # Form feed
    full_text = re.sub(r'-\n(\w)', r'\1', full_text)  # Hyphenation
    full_text = re.sub(r'\n{3,}', '\n\n', full_text)  # Excess newlines

    return {
        "text": full_text.strip(),
        "title": title,
        "authors": authors,
        "metadata": {
            "page_count": len(pages),
            "pdf_producer": meta.get("producer", ""),
            "pdf_creation_date": meta.get("creationDate", ""),
        },
    }


async def download_pdf(url: str, timeout: int = 60) -> Optional[str]:
    """
    Download a PDF from a URL to a temporary file.

    Args:
        url: PDF URL
        timeout: Download timeout in seconds

    Returns:
        Path to temporary PDF file, or None on failure
    """
    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            response = await client.get(url)
            response.raise_for_status()

            # Verify it's a PDF
            content_type = response.headers.get("content-type", "")
            if "pdf" not in content_type and not url.endswith(".pdf"):
                # Check magic bytes
                if not response.content.startswith(b'%PDF'):
                    logger.warning(f"URL does not appear to be a PDF: {url}")

            # Save to temp file
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
                f.write(response.content)
                temp_path = f.name

            return temp_path

    except Exception as e:
        logger.error(f"Failed to download PDF from {url}: {e}")
        return None


async def ingest_pdf_url(
    url: str,
    source_tier: str = "UNV",
    doc_type: str = "paper",
    chunk_strategy: str = "markdown",
    chunk_size: int = 512,
    overlap_pct: float = 0.20,
    title: Optional[str] = None,
    authors: Optional[List[str]] = None,
    published_date: Optional[str] = None,
) -> Optional[int]:
    """
    Ingest a PDF from a URL: download, extract, chunk, embed, upsert.

    Args:
        url: PDF URL
        source_tier: P1/P2/P3/UNV
        doc_type: Document type
        chunk_strategy: Chunking strategy
        chunk_size: Target tokens per chunk
        overlap_pct: Overlap fraction
        title: Override title (if known)
        authors: Override authors (if known)
        published_date: Publication date

    Returns:
        Number of chunks upserted, or None on failure
    """
    # Download
    temp_path = await download_pdf(url)
    if not temp_path:
        return None

    try:
        # Extract
        result = extract_text_from_pdf(temp_path)
        text = result["text"]

        if not text or len(text) < 100:
            logger.warning(f"Insufficient text extracted from PDF: {url}")
            return None

        # Use extracted metadata if not overridden
        if not title:
            title = result["title"] or urlparse(url).path.split("/")[-1]
        if not authors:
            authors = result["authors"]

        # Determine domain
        domain = urlparse(url).netloc

        # Chunk
        chunks = chunk_text(text, strategy=chunk_strategy, chunk_size_tokens=chunk_size, overlap_pct=overlap_pct)

        if not chunks:
            logger.warning(f"No chunks generated for PDF: {url}")
            return None

        # Embed and upsert
        count = await embed_and_upsert(
            source_url=url,
            source_tier=source_tier,
            domain=domain,
            doc_type=doc_type,
            title=title,
            authors=authors,
            published_date=published_date,
            chunks=chunks,
            metadata=result.get("metadata", {}),
        )

        logger.info(f"Ingested PDF {url} -> {count} chunks")
        return count

    finally:
        # Clean up temp file
        try:
            os.unlink(temp_path)
        except OSError:
            pass


async def ingest_pdf_file(
    file_path: str,
    source_url: str,
    source_tier: str = "UNV",
    doc_type: str = "paper",
    chunk_strategy: str = "markdown",
    chunk_size: int = 512,
    overlap_pct: float = 0.20,
    title: Optional[str] = None,
    authors: Optional[List[str]] = None,
    published_date: Optional[str] = None,
) -> Optional[int]:
    """
    Ingest a local PDF file: extract, chunk, embed, upsert.

    Args:
        file_path: Local path to PDF
        source_url: Canonical URL for citation
        source_tier: P1/P2/P3/UNV
        doc_type: Document type
        chunk_strategy: Chunking strategy
        chunk_size: Target tokens per chunk
        overlap_pct: Overlap fraction
        title: Override title
        authors: Override authors
        published_date: Publication date

    Returns:
        Number of chunks upserted, or None on failure
    """
    if not os.path.exists(file_path):
        logger.error(f"PDF file not found: {file_path}")
        return None

    result = extract_text_from_pdf(file_path)
    text = result["text"]

    if not text or len(text) < 100:
        logger.warning(f"Insufficient text extracted from PDF: {file_path}")
        return None

    if not title:
        title = result["title"] or os.path.basename(file_path)
    if not authors:
        authors = result["authors"]

    domain = urlparse(source_url).netloc if source_url else "local"

    chunks = chunk_text(text, strategy=chunk_strategy, chunk_size_tokens=chunk_size, overlap_pct=overlap_pct)

    if not chunks:
        return None

    count = await embed_and_upsert(
        source_url=source_url,
        source_tier=source_tier,
        domain=domain,
        doc_type=doc_type,
        title=title,
        authors=authors,
        published_date=published_date,
        chunks=chunks,
        metadata=result.get("metadata", {}),
    )

    logger.info(f"Ingested PDF {file_path} -> {count} chunks")
    return count
