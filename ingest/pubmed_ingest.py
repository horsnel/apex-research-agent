"""
PubMed Ingest — downloads and processes papers from PubMed.

Supports:
- Individual PMIDs
- Search query-based bulk ingest
- Metadata extraction via E-utilities API
"""

import asyncio
import logging
import re
from datetime import datetime
from typing import List, Optional

import httpx

from .chunker import chunk_text
from .embedder import embed_and_upsert

logger = logging.getLogger(__name__)

PUBMED_ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PUBMED_EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
PUBMED_ESUMMARY = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"

PUBMED_DELAY = 0.4  # NCBI requests 3 requests/second without API key


async def search_pubmed(query: str, max_results: int = 25) -> List[str]:
    """
    Search PubMed and return a list of PMIDs.

    Args:
        query: PubMed search query
        max_results: Maximum number of results

    Returns:
        List of PMID strings
    """
    params = {
        "db": "pubmed",
        "term": query,
        "retmax": max_results,
        "retmode": "json",
        "sort": "date",
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(PUBMED_ESEARCH, params=params)
            response.raise_for_status()
            data = response.json()

        pmids = data.get("esearchresult", {}).get("idlist", [])
        logger.info(f"PubMed search '{query}' returned {len(pmids)} results")
        return pmids

    except Exception as e:
        logger.error(f"PubMed search failed: {e}")
        return []


async def fetch_pubmed_summary(pmids: List[str]) -> List[dict]:
    """
    Fetch article summaries from PubMed.

    Args:
        pmids: List of PubMed IDs

    Returns:
        List of paper metadata dicts
    """
    if not pmids:
        return []

    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "json",
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(PUBMED_ESUMMARY, params=params)
            response.raise_for_status()
            data = response.json()

        papers = []
        result = data.get("result", {})

        for pmid in pmids:
            article = result.get(pmid, {})
            if not article or "error" in article:
                continue

            authors = [
                a.get("name", "")
                for a in article.get("authors", [])
                if a.get("name")
            ]

            pub_date = article.get("pubdate", "")
            published_date = None
            if pub_date:
                try:
                    # Format: "2023 Dec 15" or "2023"
                    for fmt in ("%Y %b %d", "%Y %b", "%Y"):
                        try:
                            published_date = datetime.strptime(pub_date.strip(), fmt).strftime("%Y-%m-%d")
                            break
                        except ValueError:
                            continue
                except Exception:
                    pass

            papers.append({
                "pmid": pmid,
                "title": article.get("title", "Untitled"),
                "authors": authors,
                "source_url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                "published_date": published_date,
                "journal": article.get("fulljournalname", ""),
                "doi": article.get("elocationid", ""),
            })

        return papers

    except Exception as e:
        logger.error(f"PubMed summary fetch failed: {e}")
        return []


async def fetch_pubmed_abstract(pmids: List[str]) -> dict:
    """
    Fetch abstracts for PubMed articles.

    Args:
        pmids: List of PubMed IDs

    Returns:
        Dict mapping PMID -> abstract text
    """
    if not pmids:
        return {}

    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "rettype": "abstract",
        "retmode": "text",
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(PUBMED_EFETCH, params=params)
            response.raise_for_status()
            text = response.text

        # Parse the text response into per-PMID abstracts
        abstracts = {}
        current_pmid = None
        current_lines = []

        for line in text.split("\n"):
            pmid_match = re.match(r"^PMID:\s*(\d+)", line)
            if pmid_match:
                if current_pmid and current_lines:
                    abstracts[current_pmid] = "\n".join(current_lines).strip()
                current_pmid = pmid_match.group(1)
                current_lines = []
            elif line.strip() and not line.startswith("DOI:"):
                current_lines.append(line.strip())

        # Last article
        if current_pmid and current_lines:
            abstracts[current_pmid] = "\n".join(current_lines).strip()

        return abstracts

    except Exception as e:
        logger.error(f"PubMed abstract fetch failed: {e}")
        return {}


async def ingest_pubmed_search(
    query: str,
    max_results: int = 25,
    chunk_strategy: str = "semantic",
    chunk_size: int = 512,
    overlap_pct: float = 0.20,
) -> int:
    """
    Ingest papers from a PubMed search.

    Args:
        query: PubMed search query
        max_results: Maximum papers to fetch
        chunk_strategy: Chunking strategy
        chunk_size: Target tokens per chunk
        overlap_pct: Overlap fraction

    Returns:
        Total chunks upserted
    """
    pmids = await search_pubmed(query, max_results)
    if not pmids:
        return 0

    await asyncio.sleep(PUBMED_DELAY)

    papers = await fetch_pubmed_summary(pmids)
    abstracts = await fetch_pubmed_abstract(pmids)

    total_chunks = 0

    for paper in papers:
        pmid = paper["pmid"]
        abstract = abstracts.get(pmid, "")

        if not abstract:
            logger.warning(f"No abstract for PMID:{pmid}")
            continue

        text = f"# {paper['title']}\n\n{abstract}"
        chunks = chunk_text(text, strategy=chunk_strategy, chunk_size_tokens=chunk_size, overlap_pct=overlap_pct)

        if not chunks:
            continue

        count = await embed_and_upsert(
            source_url=paper["source_url"],
            source_tier="P1",
            domain="pubmed.ncbi.nlm.nih.gov",
            doc_type="paper",
            title=paper["title"],
            authors=paper["authors"],
            published_date=paper["published_date"],
            chunks=chunks,
            metadata={"pmid": pmid, "journal": paper.get("journal", ""), "doi": paper.get("doi", "")},
        )

        total_chunks += count
        await asyncio.sleep(PUBMED_DELAY)

    logger.info(f"Ingested {total_chunks} chunks from PubMed search '{query}'")
    return total_chunks
