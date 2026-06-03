"""
Live Scraper — fallback web scraping when RAG doesn't have the answer.

Uses Firecrawl API for structured scraping with markdown output.
Falls back to httpx + BeautifulSoup if Firecrawl is unavailable.

Constraints:
- Max 3 URLs scraped per query
- 10s timeout per URL
- Clean markdown output, no HTML boilerplate
- Returns [SCRAPE_FAILED: url] on failure
"""

import asyncio
import logging
import os
import re
from dataclasses import dataclass
from typing import List, Optional
from urllib.parse import urlparse

import httpx

from ingest.html_cleaner import clean_html, clean_markdown

logger = logging.getLogger(__name__)

# ── Configuration ──
FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY", "")
FIRECRAWL_BASE_URL = os.getenv("FIRECRAWL_BASE_URL", "https://api.firecrawl.dev/v1")
MAX_URLS_PER_QUERY = 3
TIMEOUT_PER_URL = 10  # seconds
MAX_SCRAPE_TOKENS = int(os.getenv("MAX_LIVE_SCRAPE_TOKENS", "3000"))
CHARS_PER_TOKEN = 4


@dataclass
class ScrapeResult:
    """Result from a live scrape operation."""
    url: str
    markdown: str
    title: str = ""
    success: bool = True
    error: str = ""


async def firecrawl_scrape(url: str) -> ScrapeResult:
    """
    Scrape a single URL using the Firecrawl API.

    Args:
        url: URL to scrape

    Returns:
        ScrapeResult with clean markdown
    """
    if not FIRECRAWL_API_KEY:
        return ScrapeResult(
            url=url,
            markdown="",
            success=False,
            error="Firecrawl API key not configured",
        )

    try:
        async with httpx.AsyncClient(timeout=TIMEOUT_PER_URL + 5) as client:
            response = await client.post(
                f"{FIRECRAWL_BASE_URL}/scrape",
                headers={
                    "Authorization": f"Bearer {FIRECRAWL_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "url": url,
                    "formats": ["markdown"],
                    "onlyMainContent": True,
                    "waitFor": 1000,
                },
            )
            response.raise_for_status()
            data = response.json()

            markdown = data.get("data", {}).get("markdown", "")
            title = data.get("data", {}).get("metadata", {}).get("title", "")

            # Clean the markdown
            markdown = clean_markdown(markdown)

            return ScrapeResult(
                url=url,
                markdown=markdown,
                title=title,
                success=True,
            )

    except httpx.TimeoutException:
        logger.warning(f"Firecrawl timeout for {url}")
        return ScrapeResult(
            url=url,
            markdown=f"[SCRAPE_FAILED: {url}]",
            success=False,
            error="Timeout",
        )
    except Exception as e:
        logger.error(f"Firecrawl scrape failed for {url}: {e}")
        return ScrapeResult(
            url=url,
            markdown=f"[SCRAPE_FAILED: {url}]",
            success=False,
            error=str(e),
        )


async def firecrawl_search(query: str, max_results: int = MAX_URLS_PER_QUERY) -> List[ScrapeResult]:
    """
    Search the web using Firecrawl's search endpoint and scrape the results.

    Args:
        query: Search query
        max_results: Maximum number of URLs to scrape

    Returns:
        List of ScrapeResult objects
    """
    if not FIRECRAWL_API_KEY:
        logger.warning("Firecrawl API key not configured")
        return []

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{FIRECRAWL_BASE_URL}/search",
                headers={
                    "Authorization": f"Bearer {FIRECRAWL_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "query": query,
                    "limit": max_results,
                    "scrapeOptions": {
                        "formats": ["markdown"],
                        "onlyMainContent": True,
                    },
                },
            )
            response.raise_for_status()
            data = response.json()

        results = []
        for item in data.get("data", [])[:max_results]:
            markdown = item.get("markdown", "")
            title = item.get("metadata", {}).get("title", "")
            url = item.get("metadata", {}).get("sourceURL", item.get("url", ""))

            markdown = clean_markdown(markdown)

            results.append(ScrapeResult(
                url=url,
                markdown=markdown,
                title=title,
                success=True,
            ))

        logger.info(f"Firecrawl search returned {len(results)} results for '{query}'")
        return results

    except Exception as e:
        logger.error(f"Firecrawl search failed: {e}")
        return []


async def fallback_scrape(url: str) -> ScrapeResult:
    """
    Fallback scraper using httpx + BeautifulSoup.

    Used when Firecrawl is unavailable or fails.

    Args:
        url: URL to scrape

    Returns:
        ScrapeResult with cleaned text
    """
    try:
        async with httpx.AsyncClient(
            timeout=TIMEOUT_PER_URL,
            follow_redirects=True,
            headers={"User-Agent": "APEX-Research-Agent/1.0"},
        ) as client:
            response = await client.get(url)
            response.raise_for_status()

            content_type = response.headers.get("content-type", "")
            if "html" in content_type:
                clean_text = clean_html(response.text, url)
            else:
                clean_text = response.text

            return ScrapeResult(
                url=url,
                markdown=clean_text,
                success=True,
            )

    except httpx.TimeoutException:
        return ScrapeResult(
            url=url,
            markdown=f"[SCRAPE_FAILED: {url}]",
            success=False,
            error="Timeout",
        )
    except Exception as e:
        logger.error(f"Fallback scrape failed for {url}: {e}")
        return ScrapeResult(
            url=url,
            markdown=f"[SCRAPE_FAILED: {url}]",
            success=False,
            error=str(e),
        )


def truncate_to_token_budget(text: str, max_tokens: int = MAX_SCRAPE_TOKENS) -> str:
    """
    Truncate scraped text to the token budget.

    Args:
        text: Text to truncate
        max_tokens: Maximum allowed tokens

    Returns:
        Truncated text
    """
    max_chars = max_tokens * CHARS_PER_TOKEN
    if len(text) <= max_chars:
        return text

    # Truncate at sentence boundary
    truncated = text[:max_chars]
    last_period = truncated.rfind(".")
    if last_period > max_chars * 0.8:
        truncated = truncated[:last_period + 1]

    return truncated


async def live_scrape(
    query: str,
    urls: Optional[List[str]] = None,
) -> List[ScrapeResult]:
    """
    Main live scraping function.

    If URLs are provided, scrape them directly.
    Otherwise, use Firecrawl search to find relevant URLs.

    Constraints enforced:
    - Max 3 URLs per query
    - 10s timeout per URL
    - Total scraped content capped at MAX_SCRAPE_TOKENS

    Args:
        query: Search query
        urls: Optional specific URLs to scrape

    Returns:
        List of ScrapeResult objects
    """
    results = []

    if urls:
        # Scrape provided URLs directly
        scrape_urls = urls[:MAX_URLS_PER_QUERY]

        tasks = []
        for url in scrape_urls:
            if FIRECRAWL_API_KEY:
                tasks.append(firecrawl_scrape(url))
            else:
                tasks.append(fallback_scrape(url))

        results = await asyncio.gather(*tasks)
        results = list(results)

    else:
        # Search and scrape
        if FIRECRAWL_API_KEY:
            results = await firecrawl_search(query)
        else:
            logger.warning("No Firecrawl key and no URLs provided. Cannot live scrape.")
            return []

    # Apply token budget across all results
    total_chars = 0
    max_chars = MAX_SCRAPE_TOKENS * CHARS_PER_TOKEN

    for result in results:
        if result.success:
            remaining = max_chars - total_chars
            if remaining <= 0:
                result.markdown = f"[TRUNCATED: token budget exceeded]"
                result.success = False
            elif len(result.markdown) > remaining:
                result.markdown = truncate_to_token_budget(
                    result.markdown, remaining // CHARS_PER_TOKEN
                )
                total_chars += len(result.markdown)
            else:
                total_chars += len(result.markdown)

    successful = sum(1 for r in results if r.success)
    logger.info(f"Live scrape: {successful}/{len(results)} successful for '{query[:50]}'")
    return results
