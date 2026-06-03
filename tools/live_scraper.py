"""
Live Scraper — fallback web scraping when RAG doesn't have the answer.

Scraping strategy (3-tier fallback per URL):
1. httpx + BeautifulSoup (always works, no key, direct HTML parse)
2. Jina Reader API (better quality, requires JINA_API_KEY from jina.ai)
3. Firecrawl API (best quality, requires FIRECRAWL_API_KEY, 500 free credits/mo)

For web search (no URLs provided):
1. Firecrawl search (if key available)
2. Jina Reader search (if key available)
3. DuckDuckGo search + direct scrape (always works, free, no key)

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

# Jina Reader — free API key from https://jina.ai/
JINA_API_KEY = os.getenv("JINA_API_KEY", "")
JINA_READER_BASE_URL = os.getenv("JINA_READER_BASE_URL", "https://r.jina.ai")

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


# ═══════════════════════════════════════════════════════════════
# TIER 1: HTTPX + BEAUTIFULSOUP (ALWAYS WORKS, NO KEY)
# ═══════════════════════════════════════════════════════════════


async def direct_scrape(url: str) -> ScrapeResult:
    """
    Direct HTTP scrape using httpx + BeautifulSoup.

    Always works, no external API key needed.
    Uses readability + bleach for HTML cleaning.

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
        logger.error(f"Direct scrape failed for {url}: {e}")
        return ScrapeResult(
            url=url,
            markdown=f"[SCRAPE_FAILED: {url}]",
            success=False,
            error=str(e),
        )


# ═══════════════════════════════════════════════════════════════
# TIER 2: JINA READER (REQUIRES FREE API KEY)
# ═══════════════════════════════════════════════════════════════


async def jina_reader_scrape(url: str) -> ScrapeResult:
    """
    Scrape a URL using Jina Reader API.

    Jina Reader provides high-quality markdown extraction with
    JavaScript rendering and content extraction.
    Requires a free API key from https://jina.ai/.

    Args:
        url: URL to scrape

    Returns:
        ScrapeResult with clean markdown
    """
    if not JINA_API_KEY:
        return ScrapeResult(url=url, markdown="", success=False, error="JINA_API_KEY not set")

    try:
        async with httpx.AsyncClient(timeout=TIMEOUT_PER_URL + 5) as client:
            response = await client.get(
                f"{JINA_READER_BASE_URL}/{url}",
                headers={
                    "Authorization": f"Bearer {JINA_API_KEY}",
                    "Accept": "text/markdown",
                    "X-Return-Format": "markdown",
                },
                follow_redirects=True,
            )
            response.raise_for_status()

            markdown = response.text
            title = ""
            if markdown.startswith("Title:"):
                lines = markdown.split("\n", 1)
                title = lines[0].replace("Title:", "").strip()
                markdown = lines[1] if len(lines) > 1 else markdown

            markdown = clean_markdown(markdown)

            return ScrapeResult(url=url, markdown=markdown, title=title, success=True)

    except httpx.TimeoutException:
        return ScrapeResult(url=url, markdown=f"[SCRAPE_FAILED: {url}]", success=False, error="Jina timeout")
    except Exception as e:
        logger.warning(f"Jina Reader failed for {url}: {e}")
        return ScrapeResult(url=url, markdown=f"[SCRAPE_FAILED: {url}]", success=False, error=str(e))


async def jina_reader_search(query: str, max_results: int = MAX_URLS_PER_QUERY) -> List[ScrapeResult]:
    """Search the web using Jina Reader's search endpoint. Requires JINA_API_KEY."""
    if not JINA_API_KEY:
        return []

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{JINA_READER_BASE_URL}/search",
                headers={
                    "Authorization": f"Bearer {JINA_API_KEY}",
                    "Accept": "application/json",
                    "Content-Type": "application/json",
                },
                json={"q": query, "num": max_results},
            )
            response.raise_for_status()
            data = response.json()

        results = []
        for item in data.get("data", [])[:max_results]:
            url = item.get("url", "")
            title = item.get("title", "")
            markdown = item.get("content", "") or item.get("description", "")
            markdown = clean_markdown(markdown)
            results.append(ScrapeResult(url=url, markdown=markdown, title=title, success=True))

        logger.info(f"Jina Reader search returned {len(results)} results for '{query}'")
        return results

    except Exception as e:
        logger.warning(f"Jina Reader search failed: {e}")
        return []


# ═══════════════════════════════════════════════════════════════
# TIER 3: FIRECRAWL (PREMIUM, REQUIRES API KEY)
# ═══════════════════════════════════════════════════════════════


async def firecrawl_scrape(url: str) -> ScrapeResult:
    """
    Scrape a single URL using the Firecrawl API.

    Firecrawl provides structured scraping with JavaScript rendering,
    anti-bot bypass, and guaranteed markdown output.
    Free tier: 500 credits/month.

    Args:
        url: URL to scrape

    Returns:
        ScrapeResult with clean markdown
    """
    if not FIRECRAWL_API_KEY:
        return ScrapeResult(url=url, markdown="", success=False, error="FIRECRAWL_API_KEY not set")

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
            markdown = clean_markdown(markdown)

            return ScrapeResult(url=url, markdown=markdown, title=title, success=True)

    except httpx.TimeoutException:
        return ScrapeResult(url=url, markdown=f"[SCRAPE_FAILED: {url}]", success=False, error="Timeout")
    except Exception as e:
        logger.error(f"Firecrawl scrape failed for {url}: {e}")
        return ScrapeResult(url=url, markdown=f"[SCRAPE_FAILED: {url}]", success=False, error=str(e))


async def firecrawl_search(query: str, max_results: int = MAX_URLS_PER_QUERY) -> List[ScrapeResult]:
    """Search the web using Firecrawl's search endpoint. Requires FIRECRAWL_API_KEY."""
    if not FIRECRAWL_API_KEY:
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
            results.append(ScrapeResult(url=url, markdown=markdown, title=title, success=True))

        logger.info(f"Firecrawl search returned {len(results)} results for '{query}'")
        return results

    except Exception as e:
        logger.error(f"Firecrawl search failed: {e}")
        return []


# ═══════════════════════════════════════════════════════════════
# DUCKDUCKGO SEARCH (FREE, NO KEY, FOR URL DISCOVERY)
# ═══════════════════════════════════════════════════════════════


async def duckduckgo_search_urls(query: str, max_results: int = MAX_URLS_PER_QUERY) -> List[str]:
    """
    Search DuckDuckGo for URLs matching a query.

    Free, no API key required. Uses DuckDuckGo Lite (no JS challenge).
    Returns just URLs (content must be scraped separately).

    Args:
        query: Search query
        max_results: Max URLs to return

    Returns:
        List of URL strings
    """
    try:
        async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
            response = await client.post(
                "https://lite.duckduckgo.com/lite/",
                data={"q": query, "kl": "us-en"},
                headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"},
            )
            response.raise_for_status()

            # Extract URLs from DDG Lite results
            urls = re.findall(r'href="(https?://[^"]+)"', response.text)
            # Deduplicate while preserving order
            seen = set()
            unique_urls = []
            for u in urls:
                # Skip DDG's own URLs
                if "duckduckgo.com" in u:
                    continue
                if u not in seen:
                    seen.add(u)
                    unique_urls.append(u)
            return unique_urls[:max_results]

    except Exception as e:
        logger.warning(f"DuckDuckGo search failed: {e}")
        return []


# ═══════════════════════════════════════════════════════════════
# TOKEN BUDGET MANAGEMENT
# ═══════════════════════════════════════════════════════════════


def truncate_to_token_budget(text: str, max_tokens: int = MAX_SCRAPE_TOKENS) -> str:
    """Truncate scraped text to the token budget."""
    max_chars = max_tokens * CHARS_PER_TOKEN
    if len(text) <= max_chars:
        return text

    truncated = text[:max_chars]
    last_period = truncated.rfind(".")
    if last_period > max_chars * 0.8:
        truncated = truncated[:last_period + 1]

    return truncated


# ═══════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════


async def live_scrape(
    query: str,
    urls: Optional[List[str]] = None,
) -> List[ScrapeResult]:
    """
    Main live scraping function with 3-tier fallback.

    Strategy:
    1. httpx + BeautifulSoup (always works, no key needed)
    2. Jina Reader (if JINA_API_KEY available — better quality)
    3. Firecrawl (if FIRECRAWL_API_KEY available — best quality)

    For web search (no URLs provided):
    1. Firecrawl search → Jina search → DuckDuckGo + direct scrape

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
        # Scrape provided URLs directly with 3-tier fallback
        scrape_urls = urls[:MAX_URLS_PER_QUERY]
        tasks = [_scrape_url_with_fallback(url) for url in scrape_urls]
        results = list(await asyncio.gather(*tasks))

    else:
        # Search the web for relevant URLs, then scrape them
        # Try premium search APIs first, fall back to DuckDuckGo
        search_results = await firecrawl_search(query)

        if not search_results:
            search_results = await jina_reader_search(query)

        if search_results:
            results = search_results
        else:
            # DuckDuckGo + direct scrape (always works, no key)
            search_urls = await duckduckgo_search_urls(query)
            if search_urls:
                tasks = [direct_scrape(url) for url in search_urls]
                results = list(await asyncio.gather(*tasks))

        if not results:
            logger.warning("No search results from any source. Cannot live scrape.")
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


async def _scrape_url_with_fallback(url: str) -> ScrapeResult:
    """
    Scrape a single URL with 3-tier fallback.

    Tries direct httpx → Jina Reader → Firecrawl in order.

    Args:
        url: URL to scrape

    Returns:
        ScrapeResult from first successful scraper
    """
    # Tier 1: Direct httpx scrape (always works, no key)
    result = await direct_scrape(url)
    if result.success and result.markdown and not result.markdown.startswith("[SCRAPE_FAILED"):
        return result

    # Tier 2: Jina Reader (better quality, if key available)
    if JINA_API_KEY:
        result = await jina_reader_scrape(url)
        if result.success and result.markdown and not result.markdown.startswith("[SCRAPE_FAILED"):
            return result

    # Tier 3: Firecrawl (best quality, if key available)
    if FIRECRAWL_API_KEY:
        result = await firecrawl_scrape(url)
        if result.success and result.markdown and not result.markdown.startswith("[SCRAPE_FAILED"):
            return result

    return result
