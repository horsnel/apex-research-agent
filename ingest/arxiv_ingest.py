"""
arXiv Ingest — downloads and processes papers from arXiv.

Supports:
- Individual arXiv IDs
- Category-based bulk ingest
- RSS feed monitoring
"""

import asyncio
import logging
import re
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import List, Optional, Tuple
from urllib.parse import urlparse

import httpx

from .chunker import chunk_text, Chunk
from .embedder import embed_and_upsert
from .html_cleaner import clean_html

logger = logging.getLogger(__name__)

ARXIV_API_BASE = "http://export.arxiv.org/api/query"
ARXIV_ABS_BASE = "https://arxiv.org/abs"

# Rate limiting: arXiv asks for 3-second delay between requests
ARXIV_DELAY = 3.0


def _parse_arxiv_id(input_str: str) -> Optional[str]:
    """Extract arXiv ID from various URL formats or raw IDs."""
    # Direct ID: 2312.10997
    if re.match(r"^\d{4}\.\d{4,5}(v\d+)?$", input_str):
        return input_str

    # URL: https://arxiv.org/abs/2312.10997
    match = re.search(r"arxiv\.org/abs/([\d.]+(v\d+)?)", input_str)
    if match:
        return match.group(1)

    # Old format: cs/0701001
    match = re.search(r"arxiv\.org/abs/([a-z-]+/\d{7})", input_str)
    if match:
        return match.group(1)

    return None


def _parse_arxiv_entry(entry: ET.Element) -> dict:
    """Parse an arXiv API entry into a structured dict."""
    ns = {"atom": "http://www.w3.org/2005/Atom"}

    title_el = entry.find("atom:title", ns)
    title = title_el.text.strip().replace("\n", " ") if title_el is not None else "Untitled"

    # Authors
    authors = []
    for author_el in entry.findall("atom:author", ns):
        name_el = author_el.find("atom:name", ns)
        if name_el is not None and name_el.text:
            authors.append(name_el.text.strip())

    # Abstract
    summary_el = entry.find("atom:summary", ns)
    abstract = summary_el.text.strip() if summary_el is not None else ""

    # Published date
    published_el = entry.find("atom:published", ns)
    published_date = None
    if published_el is not None and published_el.text:
        try:
            published_date = datetime.strptime(published_el.text[:10], "%Y-%m-%d").strftime("%Y-%m-%d")
        except ValueError:
            pass

    # ID -> URL
    id_el = entry.find("atom:id", ns)
    source_url = id_el.text.strip() if id_el is not None else ""

    # Categories
    categories = []
    for cat_el in entry.findall("atom:category", ns):
        term = cat_el.get("term", "")
        if term:
            categories.append(term)

    # PDF link
    pdf_url = ""
    for link_el in entry.findall("atom:link", ns):
        if link_el.get("title") == "pdf":
            pdf_url = link_el.get("href", "")
            break

    return {
        "title": title,
        "authors": authors,
        "abstract": abstract,
        "published_date": published_date,
        "source_url": source_url,
        "categories": categories,
        "pdf_url": pdf_url,
    }


async def fetch_arxiv_paper(arxiv_id: str) -> Optional[dict]:
    """
    Fetch metadata for a single arXiv paper via the API.

    Args:
        arxiv_id: arXiv identifier (e.g., "2312.10997")

    Returns:
        Parsed paper metadata dict or None
    """
    url = f"{ARXIV_API_BASE}?id_list={arxiv_id}&max_results=1"

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url)
            response.raise_for_status()

        root = ET.fromstring(response.text)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        entries = root.findall("atom:entry", ns)

        if not entries:
            logger.warning(f"No arXiv entry found for ID: {arxiv_id}")
            return None

        return _parse_arxiv_entry(entries[0])

    except Exception as e:
        logger.error(f"Failed to fetch arXiv paper {arxiv_id}: {e}")
        return None


async def fetch_arxiv_category(
    category: str,
    max_results: int = 50,
) -> List[dict]:
    """
    Fetch recent papers from an arXiv category.

    Args:
        category: arXiv category (e.g., "cs.AI")
        max_results: Maximum papers to fetch

    Returns:
        List of parsed paper metadata dicts
    """
    url = f"{ARXIV_API_BASE}?search_query=cat:{category}&start=0&max_results={max_results}&sortBy=submittedDate&sortOrder=descending"

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.get(url)
            response.raise_for_status()

        root = ET.fromstring(response.text)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        entries = root.findall("atom:entry", ns)

        papers = []
        for entry in entries:
            paper = _parse_arxiv_entry(entry)
            if paper:
                papers.append(paper)

        logger.info(f"Fetched {len(papers)} papers from arXiv/{category}")
        return papers

    except Exception as e:
        logger.error(f"Failed to fetch arXiv category {category}: {e}")
        return []


async def ingest_arxiv_paper(
    arxiv_id_or_url: str,
    chunk_strategy: str = "semantic",
    chunk_size: int = 512,
    overlap_pct: float = 0.20,
) -> Optional[int]:
    """
    Ingest a single arXiv paper: fetch metadata, chunk abstract, embed, upsert.

    For full-text ingestion, the PDF would need to be downloaded and parsed
    (handled by pdf_ingest.py).

    Args:
        arxiv_id_or_url: arXiv ID or URL
        chunk_strategy: Chunking strategy
        chunk_size: Target tokens per chunk
        overlap_pct: Overlap fraction

    Returns:
        Number of chunks upserted, or None on failure
    """
    arxiv_id = _parse_arxiv_id(arxiv_id_or_url)
    if not arxiv_id:
        logger.error(f"Invalid arXiv ID: {arxiv_id_or_url}")
        return None

    paper = await fetch_arxiv_paper(arxiv_id)
    if not paper:
        return None

    # Chunk the abstract (for full-text, use pdf_ingest)
    text = f"# {paper['title']}\n\n{paper['abstract']}"
    chunks = chunk_text(text, strategy=chunk_strategy, chunk_size_tokens=chunk_size, overlap_pct=overlap_pct)

    if not chunks:
        logger.warning(f"No chunks generated for arXiv:{arxiv_id}")
        return None

    # Embed and upsert
    count = await embed_and_upsert(
        source_url=paper["source_url"],
        source_tier="P1",
        domain="arxiv.org",
        doc_type="paper",
        title=paper["title"],
        authors=paper["authors"],
        published_date=paper["published_date"],
        chunks=chunks,
        metadata={"arxiv_id": arxiv_id, "categories": paper["categories"]},
    )

    logger.info(f"Ingested arXiv:{arxiv_id} -> {count} chunks")
    return count


async def ingest_arxiv_category(
    category: str,
    max_results: int = 50,
    chunk_strategy: str = "semantic",
    chunk_size: int = 512,
    overlap_pct: float = 0.20,
) -> int:
    """
    Ingest recent papers from an arXiv category.

    Args:
        category: arXiv category (e.g., "cs.AI")
        max_results: Max papers to fetch
        chunk_strategy: Chunking strategy
        chunk_size: Target tokens per chunk
        overlap_pct: Overlap fraction

    Returns:
        Total chunks upserted
    """
    papers = await fetch_arxiv_category(category, max_results)
    total_chunks = 0

    for i, paper in enumerate(papers):
        if i > 0:
            await asyncio.sleep(ARXIV_DELAY)  # Rate limiting

        arxiv_id = _parse_arxiv_id(paper.get("source_url", ""))
        if not arxiv_id:
            continue

        count = await ingest_arxiv_paper(arxiv_id, chunk_strategy, chunk_size, overlap_pct)
        if count:
            total_chunks += count

    logger.info(f"Ingested {total_chunks} chunks from arXiv/{category}")
    return total_chunks
