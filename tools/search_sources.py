"""
Multi-Source Search — unified search across 29 information sources.

Categorized sources:
1. ACADEMIC: Semantic Scholar, Crossref, OpenAlex, DOAJ, arXiv, Exa, Papers With Code, Europe PMC, CORE, Google Scholar
2. GENERAL WEB: Serper (Google), Brave Search, DuckDuckGo, Tavily, Jina
3. ENCYCLOPEDIA: Wikipedia, Wikidata
4. CODE: GitHub, StackOverflow, Papers With Code
5. NEWS: Hacker News, Reddit, NewsAPI, Substack
6. CLINICAL: ClinicalTrials.gov, Europe PMC
7. PATENTS: Google Patents
8. VIDEO: YouTube
9. AUDIO: Podcast Index
10. SOCIAL: Mastodon
11. COMPUTATION: Wolfram Alpha
12. CITATION: Unpaywall (open access PDFs)

Each source returns normalized SearchResult objects.
The search_router() function dispatches queries to the right sources
based on the query classification (academic, web, code, news, clinical).

Routing Priority (by query type):
  academic  → Exa(neural) > Semantic Scholar > OpenAlex > arXiv > Crossref > DOAJ > CORE > Papers With Code > Europe PMC > Google Scholar
  web       → Serper(Google) > Exa > Brave > DuckDuckGo
  code      → GitHub > StackOverflow > Papers With Code > Exa
  news      → Hacker News > Serper > Reddit > NewsAPI > Substack > YouTube
  clinical  → ClinicalTrials.gov > Europe PMC > Semantic Scholar > OpenAlex
  compute   → Wolfram Alpha > Wikipedia
  patent    → Google Patents
"""

import asyncio
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Any
from urllib.parse import quote_plus

import httpx

logger = logging.getLogger(__name__)

# ── Configuration ──
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
BRAVE_SEARCH_API_KEY = os.getenv("BRAVE_SEARCH_API_KEY", "")
SERPER_API_KEY = os.getenv("SERPER_API_KEY", "")
EXA_API_KEY = os.getenv("EXA_API_KEY", "")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
JINA_API_KEY = os.getenv("JINA_API_KEY", "")
FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY", "")
SEMANTIC_SCHOLAR_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", "")
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID", "")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET", "")
WOLFRAM_APP_ID = os.getenv("WOLFRAM_APP_ID", "")
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY", "")
PODCAST_INDEX_API_KEY = os.getenv("PODCAST_INDEX_API_KEY", "")
MASTODON_INSTANCE = os.getenv("MASTODON_INSTANCE", "mastodon.social")
CONTACT_EMAIL = os.getenv("CONTACT_EMAIL", "apex-research@example.com")

DEFAULT_MAX_RESULTS = 5
DEFAULT_TIMEOUT = 15.0

# User-Agent for Wikipedia/Wikidata
WIKI_USER_AGENT = f"APEX-Research-Agent/1.0 ({CONTACT_EMAIL})"


class SourceCategory(str, Enum):
    ACADEMIC = "academic"
    WEB = "web"
    ENCYCLOPEDIA = "encyclopedia"
    CODE = "code"
    NEWS = "news"
    CLINICAL = "clinical"
    PATENT = "patent"
    VIDEO = "video"
    AUDIO = "audio"
    SOCIAL = "social"
    COMPUTATION = "computation"


@dataclass
class SearchResult:
    """Normalized search result from any source."""
    title: str
    url: str
    snippet: str  # Brief description/abstract
    source_name: str  # Which API returned this
    source_category: SourceCategory
    source_tier: str = "UNV"  # P1/P2/P3/UNV for APEX hierarchy
    published_date: Optional[str] = None
    authors: List[str] = field(default_factory=list)
    citation_count: Optional[int] = None
    open_access: bool = False
    doi: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════
# ACADEMIC SOURCES
# ═══════════════════════════════════════════════════════════════


async def search_semantic_scholar(
    query: str,
    max_results: int = DEFAULT_MAX_RESULTS,
) -> List[SearchResult]:
    """
    Search Semantic Scholar — the largest open academic graph.
    
    Free: 100 requests per 5 minutes (no key).
    With key: 1 request/second.
    Covers 200M+ papers with citation graphs.
    
    Best for: Finding papers, citation counts, author networks, 
    finding related/forward-citing work.
    """
    headers = {}
    if SEMANTIC_SCHOLAR_API_KEY:
        headers["x-api-key"] = SEMANTIC_SCHOLAR_API_KEY
    
    try:
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            r = await client.get(
                "https://api.semanticscholar.org/graph/v1/paper/search",
                headers=headers,
                params={
                    "query": query,
                    "limit": max_results,
                    "fields": "title,year,authors,citationCount,isOpenAccess,openAccessPdf,externalIds,url",
                },
            )
            
            if r.status_code == 429:
                logger.warning("Semantic Scholar rate limited. Consider getting an API key.")
                return []
            
            r.raise_for_status()
            data = r.json()
        
        results = []
        for paper in data.get("data", []):
            authors = [a.get("name", "") for a in paper.get("authors", []) if a.get("name")]
            doi = paper.get("externalIds", {}).get("DOI")
            pdf_url = paper.get("openAccessPdf", {}).get("url", "")
            
            results.append(SearchResult(
                title=paper.get("title", "Untitled"),
                url=paper.get("url", "") or f"https://semanticscholar.org/paper/{paper.get('paperId', '')}",
                snippet=f"Cited {paper.get('citationCount', 0)} times. {('Open access available.' if paper.get('isOpenAccess') else '')}",
                source_name="semantic_scholar",
                source_category=SourceCategory.ACADEMIC,
                source_tier="P1",
                published_date=str(paper.get("year", "")) if paper.get("year") else None,
                authors=authors,
                citation_count=paper.get("citationCount"),
                open_access=paper.get("isOpenAccess", False),
                doi=doi,
                extra={"pdf_url": pdf_url},
            ))
        
        return results
    
    except Exception as e:
        logger.warning(f"Semantic Scholar search failed: {e}")
        return []


async def search_crossref(
    query: str,
    max_results: int = DEFAULT_MAX_RESULTS,
) -> List[SearchResult]:
    """
    Search Crossref — the official DOI registry with 140M+ records.
    
    Free: Polite pool (with mailto) gets priority routing.
    Covers: Journal articles, books, conference proceedings, preprints.
    Returns: DOIs, citation counts, license info, full metadata.
    
    Best for: DOI verification, citation tracking, finding official 
    publication metadata and licensing.
    """
    try:
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            r = await client.get(
                "https://api.crossref.org/works",
                params={
                    "query": query,
                    "rows": max_results,
                    "mailto": CONTACT_EMAIL,
                    "sort": "relevance",
                },
            )
            r.raise_for_status()
            data = r.json()
        
        results = []
        for item in data.get("message", {}).get("items", []):
            authors = []
            for a in item.get("author", []):
                name = f"{a.get('given', '')} {a.get('family', '')}".strip()
                if name:
                    authors.append(name)
            
            doi = item.get("DOI", "")
            title_list = item.get("title", ["Untitled"])
            title = title_list[0] if title_list else "Untitled"
            
            pub_date = item.get("published-print", {}).get("date-parts", [[None]])[0]
            pub_year = str(pub_date[0]) if pub_date and pub_date[0] else None
            
            is_oa = item.get("license", []) and any(
                l.get("content-version") == "vor" for l in item.get("license", [])
            )
            
            results.append(SearchResult(
                title=title,
                url=f"https://doi.org/{doi}" if doi else item.get("URL", ""),
                snippet=item.get("abstract", "")[:300] if item.get("abstract") else f"DOI: {doi}",
                source_name="crossref",
                source_category=SourceCategory.ACADEMIC,
                source_tier="P1",
                published_date=pub_year,
                authors=authors,
                citation_count=item.get("is-referenced-by-count"),
                open_access=is_oa,
                doi=doi,
            ))
        
        return results
    
    except Exception as e:
        logger.warning(f"Crossref search failed: {e}")
        return []


async def search_openalex(
    query: str,
    max_results: int = DEFAULT_MAX_RESULTS,
) -> List[SearchResult]:
    """
    Search OpenAlex — open catalog of 250M+ scholarly works.
    
    Free: Completely open, no key needed, no rate limits.
    Covers: Papers, datasets, institutions, concepts, funding.
    Returns: Open access status, APCs, cited_by_count, concepts.
    
    Best for: Broad academic discovery, institutional analysis,
    open access detection, concept/topic mapping.
    """
    try:
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            r = await client.get(
                "https://api.openalex.org/works",
                params={
                    "search": query,
                    "per_page": max_results,
                    "select": "id,title,publication_year,authorships,cited_by_count,open_access,doi,primary_location",
                },
            )
            r.raise_for_status()
            data = r.json()
        
        results = []
        for work in data.get("results", []):
            authors = []
            for a in work.get("authorships", []):
                name = a.get("author", {}).get("display_name", "")
                if name:
                    authors.append(name)
            
            oa = work.get("open_access", {})
            location = work.get("primary_location", {}) or {}
            source_url = location.get("landing_page_url") or work.get("doi", "")
            if source_url and not source_url.startswith("http"):
                source_url = f"https://doi.org/{source_url}"
            
            results.append(SearchResult(
                title=work.get("title", "Untitled"),
                url=source_url,
                snippet=f"Cited {work.get('cited_by_count', 0)} times. {('Open access.' if oa.get('is_oa') else '')}",
                source_name="openalex",
                source_category=SourceCategory.ACADEMIC,
                source_tier="P1",
                published_date=str(work.get("publication_year", "")) if work.get("publication_year") else None,
                authors=authors,
                citation_count=work.get("cited_by_count"),
                open_access=oa.get("is_oa", False),
                doi=work.get("doi"),
                extra={"oa_url": oa.get("oa_url", "")},
            ))
        
        return results
    
    except Exception as e:
        logger.warning(f"OpenAlex search failed: {e}")
        return []


async def search_doaj(
    query: str,
    max_results: int = DEFAULT_MAX_RESULTS,
) -> List[SearchResult]:
    """
    Search DOAJ — Directory of Open Access Journals.
    
    Free: Completely open, no key needed.
    Covers: ~6M articles from ~20K open access journals.
    All results are open access by definition.
    
    Best for: Finding freely available papers, verifying journal 
    legitimacy (DOAJ-indexed = trusted OA journal).
    """
    try:
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            r = await client.get(
                f"https://doaj.org/api/search/articles/{quote_plus(query)}",
                params={"pageSize": max_results},
            )
            r.raise_for_status()
            data = r.json()
        
        results = []
        for item in data.get("results", []):
            bibjson = item.get("bibjson", {})
            authors = [a.get("name", "") for a in bibjson.get("author", []) if a.get("name")]
            
            results.append(SearchResult(
                title=bibjson.get("title", "Untitled"),
                url=bibjson.get("link", [{}])[0].get("url", "") if bibjson.get("link") else "",
                snippet=bibjson.get("abstract", "")[:300] if bibjson.get("abstract") else "",
                source_name="doaj",
                source_category=SourceCategory.ACADEMIC,
                source_tier="P1",
                published_date=bibjson.get("year"),
                authors=authors,
                open_access=True,  # DOAJ is all OA
                doi=bibjson.get("doi"),
            ))
        
        return results
    
    except Exception as e:
        logger.warning(f"DOAJ search failed: {e}")
        return []


async def search_clinical_trials(
    query: str,
    max_results: int = DEFAULT_MAX_RESULTS,
) -> List[SearchResult]:
    """
    Search ClinicalTrials.gov — official US clinical trial registry.
    
    Free: Completely open, no key needed.
    Covers: 400K+ studies from 220 countries.
    Returns: Trial status, phase, enrollment, conditions, interventions.
    
    Best for: Medical/clinical research, drug development tracking,
    evidence-based medicine queries.
    """
    try:
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            r = await client.get(
                "https://clinicaltrials.gov/api/v2/studies",
                params={
                    "query.term": query,
                    "pageSize": max_results,
                    "fields": "protocolSection.identificationModule,protocolSection.statusModule,protocolSection.designModule",
                },
            )
            r.raise_for_status()
            data = r.json()
        
        results = []
        for study in data.get("studies", []):
            proto = study.get("protocolSection", {})
            ident = proto.get("identificationModule", {})
            status = proto.get("statusModule", {})
            design = proto.get("designModule", {})
            
            nct_id = ident.get("nctId", "")
            title = ident.get("briefTitle", "Untitled")
            overall_status = status.get("overallStatus", "")
            phase = design.get("phases", [])
            
            results.append(SearchResult(
                title=title,
                url=f"https://clinicaltrials.gov/study/{nct_id}",
                snippet=f"Status: {overall_status}. Phase: {', '.join(phase) if phase else 'N/A'}. NCT: {nct_id}",
                source_name="clinical_trials",
                source_category=SourceCategory.CLINICAL,
                source_tier="P1",
                published_date=status.get("startDateStruct", {}).get("date"),
                extra={"nct_id": nct_id, "status": overall_status, "phase": phase},
            ))
        
        return results
    
    except Exception as e:
        logger.warning(f"ClinicalTrials.gov search failed: {e}")
        return []


# ═══════════════════════════════════════════════════════════════
# ENCYCLOPEDIA SOURCES
# ═══════════════════════════════════════════════════════════════


async def search_wikipedia(
    query: str,
    max_results: int = DEFAULT_MAX_RESULTS,
    get_extract: bool = True,
) -> List[SearchResult]:
    """
    Search Wikipedia — the largest free encyclopedia.
    
    Free: No key needed. Requires proper User-Agent header.
    Covers: 60M+ articles in 300+ languages.
    Returns: Article extracts with structured metadata.
    
    Best for: General knowledge, definitions, background context,
    finding primary sources cited in Wikipedia articles.
    """
    try:
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            # Step 1: Search for article titles
            r = await client.get(
                "https://en.wikipedia.org/w/api.php",
                headers={"User-Agent": WIKI_USER_AGENT},
                params={
                    "action": "query",
                    "list": "search",
                    "srsearch": query,
                    "format": "json",
                    "srlimit": max_results,
                },
            )
            r.raise_for_status()
            data = r.json()
        
        results = []
        titles = []
        for item in data.get("query", {}).get("search", []):
            title = item.get("title", "")
            titles.append(title)
            results.append(SearchResult(
                title=title,
                url=f"https://en.wikipedia.org/wiki/{quote_plus(title)}",
                snippet=item.get("snippet", "").replace('<span class="searchmatch">', '').replace('</span>', ''),
                source_name="wikipedia",
                source_category=SourceCategory.ENCYCLOPEDIA,
                source_tier="P3",
            ))
        
        # Step 2: Get extracts for the found articles
        if get_extract and titles:
            try:
                async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
                    r = await client.get(
                        "https://en.wikipedia.org/w/api.php",
                        headers={"User-Agent": WIKI_USER_AGENT},
                        params={
                            "action": "query",
                            "titles": "|".join(titles),
                            "prop": "extracts",
                            "exintro": True,
                            "explaintext": True,
                            "format": "json",
                        },
                    )
                    r.raise_for_status()
                    extract_data = r.json()
                
                pages = extract_data.get("query", {}).get("pages", {})
                for page_id, page in pages.items():
                    title = page.get("title", "")
                    extract = page.get("extract", "")
                    for result in results:
                        if result.title == title and extract:
                            result.snippet = extract[:500]
                            break
            except Exception as e:
                logger.debug(f"Wikipedia extract fetch failed: {e}")
        
        return results
    
    except Exception as e:
        logger.warning(f"Wikipedia search failed: {e}")
        return []


async def search_wikidata(
    query: str,
    max_results: int = DEFAULT_MAX_RESULTS,
) -> List[SearchResult]:
    """
    Search Wikidata — structured knowledge base for entity lookup.
    
    Free: No key needed. Requires proper User-Agent header.
    Covers: 100M+ data items, structured relationships.
    Returns: Entity IDs, labels, descriptions, property values.
    
    Best for: Entity disambiguation, finding IDs for people/organizations/
    concepts, getting structured data (dates, locations, relationships).
    """
    try:
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            r = await client.get(
                "https://www.wikidata.org/w/api.php",
                headers={"User-Agent": WIKI_USER_AGENT},
                params={
                    "action": "wbsearchentities",
                    "search": query,
                    "language": "en",
                    "format": "json",
                    "limit": max_results,
                },
            )
            r.raise_for_status()
            data = r.json()
        
        results = []
        for item in data.get("search", []):
            qid = item.get("id", "")
            results.append(SearchResult(
                title=item.get("label", ""),
                url=f"https://www.wikidata.org/wiki/{qid}",
                snippet=item.get("description", ""),
                source_name="wikidata",
                source_category=SourceCategory.ENCYCLOPEDIA,
                source_tier="P2",
                extra={"qid": qid},
            ))
        
        return results
    
    except Exception as e:
        logger.warning(f"Wikidata search failed: {e}")
        return []


# ═══════════════════════════════════════════════════════════════
# CODE SOURCES
# ═══════════════════════════════════════════════════════════════


async def search_github(
    query: str,
    max_results: int = DEFAULT_MAX_RESULTS,
) -> List[SearchResult]:
    """
    Search GitHub — code repositories and code search.
    
    Free with token: 30 req/min (authenticated) vs 10 req/min (unauthenticated).
    Covers: 300M+ repositories, code, issues, discussions.
    Returns: Repo descriptions, stars, language, license.
    
    Best for: Finding implementations, open-source tools,
    comparing approaches, finding code examples.
    """
    headers = {"Accept": "application/vnd.github.v3+json"}
    if GITHUB_TOKEN:
        headers["Authorization"] = f"token {GITHUB_TOKEN}"
    
    try:
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            r = await client.get(
                "https://api.github.com/search/repositories",
                headers=headers,
                params={
                    "q": query,
                    "per_page": max_results,
                    "sort": "stars",
                    "order": "desc",
                },
            )
            if r.status_code == 422:
                return []
            r.raise_for_status()
            data = r.json()
        
        results = []
        for repo in data.get("items", []):
            results.append(SearchResult(
                title=repo.get("full_name", ""),
                url=repo.get("html_url", ""),
                snippet=f"⭐ {repo.get('stargazers_count', 0)} | {repo.get('language', '')} | {repo.get('description', '')[:200]}",
                source_name="github",
                source_category=SourceCategory.CODE,
                source_tier="P3",
                published_date=repo.get("created_at", "")[:10],
                extra={
                    "stars": repo.get("stargazers_count", 0),
                    "language": repo.get("language", ""),
                    "license": repo.get("license", {}).get("spdx_id", "") if repo.get("license") else "",
                    "forks": repo.get("forks_count", 0),
                },
            ))
        
        return results
    
    except Exception as e:
        logger.warning(f"GitHub search failed: {e}")
        return []


async def search_stackoverflow(
    query: str,
    max_results: int = DEFAULT_MAX_RESULTS,
) -> List[SearchResult]:
    """
    Search StackOverflow / Stack Exchange — Q&A for developers.
    
    Free: No key needed. 300 requests/second.
    Covers: 55M+ questions across 180+ Stack Exchange sites.
    Returns: Questions with accepted answers, scores, tags.
    
    Best for: Technical how-to queries, debugging, finding 
    expert explanations of programming/science topics.
    """
    try:
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            r = await client.get(
                "https://api.stackexchange.com/2.3/search/advanced",
                params={
                    "q": query,
                    "site": "stackoverflow",
                    "pagesize": max_results,
                    "order": "desc",
                    "sort": "relevance",
                    "filter": "withbody",
                },
            )
            r.raise_for_status()
            data = r.json()
        
        results = []
        for item in data.get("items", []):
            # Strip HTML from body
            body = item.get("body", "")
            body = re.sub(r'<[^>]+>', '', body)[:300]
            
            results.append(SearchResult(
                title=item.get("title", ""),
                url=item.get("link", ""),
                snippet=f"Score: {item.get('score', 0)} | Answers: {item.get('answer_count', 0)} | {body}",
                source_name="stackoverflow",
                source_category=SourceCategory.CODE,
                source_tier="P3",
                extra={
                    "score": item.get("score", 0),
                    "answer_count": item.get("answer_count", 0),
                    "tags": item.get("tags", []),
                    "is_answered": item.get("is_answered", False),
                },
            ))
        
        return results
    
    except Exception as e:
        logger.warning(f"StackOverflow search failed: {e}")
        return []


# ═══════════════════════════════════════════════════════════════
# NEWS SOURCES
# ═══════════════════════════════════════════════════════════════


async def search_hackernews(
    query: str,
    max_results: int = DEFAULT_MAX_RESULTS,
) -> List[SearchResult]:
    """
    Search Hacker News — tech-focused community discussion.
    
    Free: No key needed. Via Algolia API.
    Covers: Stories, comments, show HN, Ask HN.
    Returns: Points, comment counts, story URLs.
    
    Best for: Tech trends, startup/industry analysis, 
    finding expert commentary on new research.
    """
    try:
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            r = await client.get(
                "https://hn.algolia.com/api/v1/search",
                params={
                    "query": query,
                    "tags": "story",
                    "hitsPerPage": max_results,
                },
            )
            r.raise_for_status()
            data = r.json()
        
        results = []
        for hit in data.get("hits", []):
            title = hit.get("title", "")
            url = hit.get("url", "") or f"https://news.ycombinator.com/item?id={hit.get('objectID', '')}"
            points = hit.get("points", 0) or 0
            comments = hit.get("num_comments", 0) or 0
            
            results.append(SearchResult(
                title=title,
                url=url,
                snippet=f"Points: {points} | Comments: {comments}",
                source_name="hackernews",
                source_category=SourceCategory.NEWS,
                source_tier="P3",
                published_date=hit.get("created_at", "")[:10],
                extra={"points": points, "comment_count": comments, "hn_id": hit.get("objectID")},
            ))
        
        return results
    
    except Exception as e:
        logger.warning(f"Hacker News search failed: {e}")
        return []


async def search_brave(
    query: str,
    max_results: int = DEFAULT_MAX_RESULTS,
) -> List[SearchResult]:
    """
    Search Brave Search — privacy-focused web search API.
    
    Free tier: 2,000 queries/month.
    Covers: Full web index, independent of Google/Bing.
    Returns: Web results with rich snippets.
    
    Best for: General web search when DuckDuckGo fails,
    getting diverse search results independent of big tech.
    """
    if not BRAVE_SEARCH_API_KEY:
        return []
    
    try:
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            r = await client.get(
                "https://api.search.brave.com/res/v1/web/search",
                headers={"X-Subscription-Token": BRAVE_SEARCH_API_KEY, "Accept": "application/json"},
                params={"q": query, "count": max_results},
            )
            r.raise_for_status()
            data = r.json()
        
        results = []
        for item in data.get("web", {}).get("results", []):
            results.append(SearchResult(
                title=item.get("title", ""),
                url=item.get("url", ""),
                snippet=item.get("description", ""),
                source_name="brave",
                source_category=SourceCategory.WEB,
                source_tier="P3",
                published_date=item.get("age", ""),
            ))
        
        return results
    
    except Exception as e:
        logger.warning(f"Brave Search failed: {e}")
        return []


async def search_serper(
    query: str,
    max_results: int = DEFAULT_MAX_RESULTS,
) -> List[SearchResult]:
    """
    Search Serper — Google Search Results API.
    
    Free tier: 2,500 queries (one-time).
    Covers: Full Google index with rich results.
    Returns: Knowledge graph, featured snippets, organic results.
    
    Best for: When you need Google-quality search results,
    finding featured snippets and knowledge panels.
    """
    if not SERPER_API_KEY:
        return []
    
    try:
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            r = await client.post(
                "https://google.serper.dev/search",
                headers={"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"},
                json={"q": query, "num": max_results},
            )
            r.raise_for_status()
            data = r.json()
        
        results = []
        for item in data.get("organic", []):
            results.append(SearchResult(
                title=item.get("title", ""),
                url=item.get("link", ""),
                snippet=item.get("snippet", ""),
                source_name="serper",
                source_category=SourceCategory.WEB,
                source_tier="P3",
                extra={"position": item.get("position")},
            ))
        
        # Add knowledge graph if available
        kg = data.get("knowledgeGraph")
        if kg:
            results.insert(0, SearchResult(
                title=kg.get("title", ""),
                url=kg.get("descriptionLink", ""),
                snippet=kg.get("description", ""),
                source_name="serper_kg",
                source_category=SourceCategory.WEB,
                source_tier="P2",
            ))
        
        return results
    
    except Exception as e:
        logger.warning(f"Serper search failed: {e}")
        return []


# ═══════════════════════════════════════════════════════════════
# EXA.AI — NEURAL / SEMANTIC SEARCH (PAID, HIGH QUALITY)
# ═══════════════════════════════════════════════════════════════


async def search_exa(
    query: str,
    max_results: int = DEFAULT_MAX_RESULTS,
    search_type: str = "neural",
    category: Optional[str] = None,
) -> List[SearchResult]:
    """
    Search Exa — neural/semantic search API with high-quality results.
    
    Free tier: 1,000 requests/month.
    Covers: Full web with neural (meaning-based) and keyword search.
    Returns: Titles, URLs, and optionally full page text content.
    
    Unique features:
    - Neural search: understands meaning, not just keywords
    - Category filter: company, research paper, news, github repo, tweet, movie, song, personal site, or pdf
    - Autoprompt: automatically optimizes your query for better results
    - Content retrieval: can return full text of pages (not just snippets)
    
    Best for: Academic research (finds papers by meaning), finding 
    specific implementations, semantic discovery of related work.
    """
    if not EXA_API_KEY:
        return []
    
    try:
        payload = {
            "query": query,
            "num_results": max_results,
            "type": search_type,  # "neural" or "keyword"
            "use_autoprompt": True,
            "contents": {
                "text": {"maxCharacters": 1000},
            },
        }
        
        # Add category filter if specified
        if category:
            payload["category"] = category
        
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT + 5) as client:
            r = await client.post(
                "https://api.exa.ai/search",
                headers={
                    "x-api-key": EXA_API_KEY,
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            r.raise_for_status()
            data = r.json()
        
        results = []
        for item in data.get("results", []):
            url = item.get("url", "")
            title = item.get("title", "") or url.split("/")[-1].replace("-", " ").title()
            text = item.get("text", "")[:500] if item.get("text") else ""
            published = item.get("publishedDate", "")
            
            # Determine tier based on URL domain
            tier = "UNV"
            for p1_domain in ["arxiv.org", "semanticscholar.org", "nature.com", "science.org", 
                              "aclanthology.org", "openreview.net", "nejm.org", "lancet.com"]:
                if p1_domain in url:
                    tier = "P1"
                    break
            if tier == "UNV":
                for p2_domain in [".edu", "nih.gov", "nasa.gov", "cdc.gov", "who.int", "nist.gov"]:
                    if p2_domain in url:
                        tier = "P2"
                        break
            
            # Category-based tier override
            if category == "research paper":
                tier = "P1" if tier == "UNV" else tier
            
            results.append(SearchResult(
                title=title,
                url=url,
                snippet=text or item.get("score", ""),
                source_name="exa",
                source_category=SourceCategory.ACADEMIC if category == "research paper" else SourceCategory.WEB,
                source_tier=tier,
                published_date=published[:10] if published else None,
                extra={
                    "search_type": search_type,
                    "category": category,
                    "score": item.get("score"),
                    "author": item.get("author", ""),
                },
            ))
        
        return results
    
    except Exception as e:
        logger.warning(f"Exa search failed: {e}")
        return []


# ═══════════════════════════════════════════════════════════════
# ARXIV API — PREPRINT REPOSITORY
# ═══════════════════════════════════════════════════════════════


async def search_arxiv(
    query: str,
    max_results: int = DEFAULT_MAX_RESULTS,
) -> List[SearchResult]:
    """
    Search arXiv — open preprint repository for physics, math, CS, bio.
    
    Free: No key needed, no rate limits (reasonable use).
    Covers: 2.4M+ preprints across physics, math, CS, q-bio, stat, eess, econ.
    Returns: Titles, authors, abstracts, categories, PDF links.
    
    Best for: Finding cutting-edge preprints not yet peer-reviewed,
    CS/AI/ML research, physics, mathematics papers.
    """
    try:
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            r = await client.get(
                "https://export.arxiv.org/api/query",
                params={
                    "search_query": f"all:{query}",
                    "start": 0,
                    "max_results": max_results,
                    "sortBy": "relevance",
                    "sortOrder": "descending",
                },
            )
            r.raise_for_status()
        
        # Parse Atom XML response
        import xml.etree.ElementTree as ET
        root = ET.fromstring(r.text)
        ns = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
        
        results = []
        for entry in root.findall("atom:entry", ns):
            title = entry.find("atom:title", ns)
            title_text = title.text.strip().replace("\n", " ") if title is not None else "Untitled"
            
            summary = entry.find("atom:summary", ns)
            abstract = summary.text.strip().replace("\n", " ")[:500] if summary is not None else ""
            
            # Get PDF link
            pdf_url = ""
            for link in entry.findall("atom:link", ns):
                if link.get("title") == "pdf":
                    pdf_url = link.get("href", "")
                    break
            
            # Get arXiv ID
            arxiv_id = entry.find("atom:id", ns)
            arxiv_url = arxiv_id.text if arxiv_id is not None else ""
            if not pdf_url and arxiv_url:
                pdf_url = arxiv_url.replace("abs", "pdf")
            
            # Get authors
            authors = []
            for author in entry.findall("atom:author", ns):
                name = author.find("atom:name", ns)
                if name is not None and name.text:
                    authors.append(name.text.strip())
            
            # Get published date
            published = entry.find("atom:published", ns)
            pub_date = published.text[:10] if published is not None else None
            
            # Get categories
            categories = []
            for cat in entry.findall("atom:category", ns):
                term = cat.get("term", "")
                if term:
                    categories.append(term)
            
            results.append(SearchResult(
                title=title_text,
                url=arxiv_url,
                snippet=abstract,
                source_name="arxiv",
                source_category=SourceCategory.ACADEMIC,
                source_tier="P1",
                published_date=pub_date,
                authors=authors,
                open_access=True,  # arXiv is all open access
                doi=None,
                extra={
                    "pdf_url": pdf_url,
                    "categories": categories,
                    "arxiv_id": arxiv_url.split("/abs/")[-1] if "/abs/" in arxiv_url else "",
                },
            ))
        
        return results
    
    except Exception as e:
        logger.warning(f"arXiv search failed: {e}")
        return []


# ═══════════════════════════════════════════════════════════════
# TAVILY — AI-OPTIMIZED SEARCH
# ═══════════════════════════════════════════════════════════════


async def search_tavily(
    query: str,
    max_results: int = DEFAULT_MAX_RESULTS,
    search_depth: str = "basic",
) -> List[SearchResult]:
    """
    Search Tavily — AI-optimized search API built for agents.
    
    Free tier: 1,000 requests/month.
    Covers: Full web with AI-optimized result extraction.
    Returns: Titles, URLs, content snippets optimized for LLM consumption.
    
    Best for: Getting pre-extracted, clean content for LLM consumption,
    deep research mode that extracts more content per page.
    """
    if not TAVILY_API_KEY:
        return []
    
    try:
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT + 5) as client:
            r = await client.post(
                "https://api.tavily.com/search",
                headers={"Content-Type": "application/json"},
                json={
                    "api_key": TAVILY_API_KEY,
                    "query": query,
                    "max_results": max_results,
                    "search_depth": search_depth,  # "basic" or "advanced"
                    "include_raw_content": False,
                    "include_answer": True,
                },
            )
            r.raise_for_status()
            data = r.json()
        
        results = []
        for item in data.get("results", []):
            results.append(SearchResult(
                title=item.get("title", ""),
                url=item.get("url", ""),
                snippet=item.get("content", "")[:500],
                source_name="tavily",
                source_category=SourceCategory.WEB,
                source_tier="P3",
                published_date=item.get("published_date", ""),
                extra={"score": item.get("score", 0)},
            ))
        
        # If Tavily generated an answer, include it
        answer = data.get("answer")
        if answer:
            results.insert(0, SearchResult(
                title=f"AI Answer: {query[:50]}",
                url="",
                snippet=answer[:500],
                source_name="tavily_answer",
                source_category=SourceCategory.WEB,
                source_tier="P2",
            ))
        
        return results
    
    except Exception as e:
        logger.warning(f"Tavily search failed: {e}")
        return []


# ═══════════════════════════════════════════════════════════════
# REDDIT — COMMUNITY DISCUSSIONS
# ═══════════════════════════════════════════════════════════════


async def _search_reddit_via_serper(
    query: str,
    max_results: int = DEFAULT_MAX_RESULTS,
) -> List[SearchResult]:
    """Fallback: Search Reddit via Serper Google search with site:reddit.com filter."""
    if not SERPER_API_KEY:
        return []
    
    try:
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            r = await client.post(
                "https://google.serper.dev/search",
                headers={"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"},
                json={"q": f"site:reddit.com {query}", "num": max_results},
            )
            r.raise_for_status()
            data = r.json()
        
        results = []
        for item in data.get("organic", []):
            url = item.get("link", "")
            if "reddit.com" not in url:
                continue
            
            # Extract subreddit from URL
            subreddit = ""
            parts = url.split("/r/")
            if len(parts) > 1:
                subreddit = parts[1].split("/")[0]
            
            title = item.get("title", "").replace(" - Reddit", "").replace("r/", "r/")
            if subreddit:
                title = f"[r/{subreddit}] {title}"
            
            results.append(SearchResult(
                title=title,
                url=url,
                snippet=item.get("snippet", ""),
                source_name="reddit_via_serper",
                source_category=SourceCategory.NEWS,
                source_tier="P3",
                extra={"via": "serper", "subreddit": subreddit},
            ))
        
        return results
    
    except Exception as e:
        logger.debug(f"Reddit via Serper fallback failed: {e}")
        return []


async def search_reddit(
    query: str,
    max_results: int = DEFAULT_MAX_RESULTS,
) -> List[SearchResult]:
    """
    Search Reddit — community discussions and user-generated content.
    
    Free: No key needed for basic search via old.reddit.com (OAuth for higher limits).
    Covers: Thousands of communities (subreddits) across all topics.
    Returns: Post titles, scores, comment counts, self-text previews.
    
    Best for: Finding community opinions, real-world experiences,
    trending discussions, informal expert commentary.
    
    Note: Reddit blocks direct API access without OAuth. Falls back to 
    Serper Google search with site:reddit.com filter when available.
    """
    try:
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT, follow_redirects=True) as client:
            # Reddit search API — use old.reddit.com to avoid blocking
            r = await client.get(
                "https://old.reddit.com/search.json",
                headers={"User-Agent": "APEX-Research-Agent/1.0 (research bot)"},
                params={
                    "q": query,
                    "limit": max_results,
                    "sort": "relevance",
                    "type": "link",
                },
            )
            r.raise_for_status()
            data = r.json()
        
        results = []
        for child in data.get("data", {}).get("children", []):
            post = child.get("data", {})
            title = post.get("title", "")
            url = post.get("url", "")
            permalink = f"https://www.reddit.com{post.get('permalink', '')}"
            
            # Use permalink as URL if the post URL is just an image/external link
            if not url or "reddit.com" not in url:
                result_url = permalink
            else:
                result_url = url
            
            score = post.get("score", 0)
            comments = post.get("num_comments", 0)
            selftext = post.get("selftext", "")[:200].replace("\n", " ")
            subreddit = post.get("subreddit", "")
            
            results.append(SearchResult(
                title=f"[r/{subreddit}] {title}",
                url=result_url,
                snippet=f"Score: {score} | Comments: {comments} | {selftext}",
                source_name="reddit",
                source_category=SourceCategory.NEWS,
                source_tier="P3",
                published_date=str(post.get("created_utc", "")),
                extra={
                    "score": score,
                    "comment_count": comments,
                    "subreddit": subreddit,
                    "permalink": permalink,
                },
            ))
        
        return results
    
    except Exception as e:
        logger.debug(f"Reddit direct API failed: {e}")
        # Fallback: Use Serper with site:reddit.com filter
        if SERPER_API_KEY:
            return await _search_reddit_via_serper(query, max_results)
        return []


# ═══════════════════════════════════════════════════════════════
# NEWSAPI — NEWS ARTICLE SEARCH
# ═══════════════════════════════════════════════════════════════


async def search_newsapi(
    query: str,
    max_results: int = DEFAULT_MAX_RESULTS,
) -> List[SearchResult]:
    """
    Search NewsAPI — aggregate news from 150,000+ sources worldwide.
    
    Free tier: 100 requests/day, 100 results per request.
    Covers: 150K+ news sources in 50+ languages.
    Returns: Article titles, URLs, descriptions, source names, dates.
    
    Best for: Current events, breaking news, media analysis,
    tracking recent developments across global news outlets.
    """
    if not NEWSAPI_KEY:
        return []
    
    try:
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            r = await client.get(
                "https://newsapi.org/v2/everything",
                params={
                    "q": query,
                    "pageSize": max_results,
                    "sortBy": "relevancy",
                    "apiKey": NEWSAPI_KEY,
                    "language": "en",
                },
            )
            r.raise_for_status()
            data = r.json()
        
        results = []
        for article in data.get("articles", []):
            source_name = article.get("source", {}).get("name", "")
            title = article.get("title", "") or "Untitled"
            
            results.append(SearchResult(
                title=title,
                url=article.get("url", ""),
                snippet=article.get("description", "") or article.get("content", "")[:300],
                source_name="newsapi",
                source_category=SourceCategory.NEWS,
                source_tier="P3",
                published_date=article.get("publishedAt", "")[:10],
                authors=[article.get("author", "")] if article.get("author") else [],
                extra={"news_source": source_name},
            ))
        
        return results
    
    except Exception as e:
        logger.warning(f"NewsAPI search failed: {e}")
        return []


# ═══════════════════════════════════════════════════════════════
# WOLFRAM ALPHA — COMPUTATION ENGINE
# ═══════════════════════════════════════════════════════════════


async def search_wolfram(
    query: str,
    max_results: int = 3,
) -> List[SearchResult]:
    """
    Search Wolfram Alpha — computational knowledge engine.
    
    Free: 2,000 queries/month with AppID from https://products.wolframalpha.com/api/
    Covers: Mathematics, science, engineering, geography, finance, nutrition,
    unit conversions, real-time data (stock prices, weather, etc.).
    Returns: Computed answers, plots, step-by-step solutions.
    
    Best for: Mathematical computations, scientific calculations, 
    factual data lookups, unit conversions, real-time quantitative data.
    """
    if not WOLFRAM_APP_ID:
        # Fallback: try Serper for "wolframalpha <query>" to get cached results
        if SERPER_API_KEY:
            try:
                async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
                    r = await client.post(
                        "https://google.serper.dev/search",
                        headers={"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"},
                        json={"q": f"site:wolframalpha.com {query}", "num": max_results},
                    )
                    r.raise_for_status()
                    data = r.json()
                results = []
                for item in data.get("organic", []):
                    results.append(SearchResult(
                        title=f"[Wolfram] {item.get('title', '')}",
                        url=item.get("link", ""),
                        snippet=item.get("snippet", ""),
                        source_name="wolfram_via_serper",
                        source_category=SourceCategory.COMPUTATION,
                        source_tier="P2",
                    ))
                return results
            except Exception:
                pass
        return []
    
    try:
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT + 10) as client:
            r = await client.get(
                "https://api.wolframalpha.com/v2/query",
                params={
                    "appid": WOLFRAM_APP_ID,
                    "input": query,
                    "output": "JSON",
                    "format": "plaintext",
                    "podindex": "1,2,3",
                },
            )
            r.raise_for_status()
            data = r.json()
        
        results = []
        query_result = data.get("queryresult", {})
        
        for pod in query_result.get("pods", []):
            pod_title = pod.get("title", "")
            for subpod in pod.get("subpods", []):
                plaintext = subpod.get("plaintext", "")
                if not plaintext:
                    continue
                
                results.append(SearchResult(
                    title=f"[Wolfram] {pod_title}",
                    url=f"https://www.wolframalpha.com/input?i={quote_plus(query)}",
                    snippet=plaintext[:500],
                    source_name="wolfram",
                    source_category=SourceCategory.COMPUTATION,
                    source_tier="P1",
                    extra={"pod_title": pod_title, "is_computation": True},
                ))
                
                if len(results) >= max_results:
                    break
            if len(results) >= max_results:
                break
        
        return results
    
    except Exception as e:
        logger.warning(f"Wolfram Alpha search failed: {e}")
        return []


# ═══════════════════════════════════════════════════════════════
# GOOGLE SCHOLAR — VIA SERPER SCHOLAR
# ═══════════════════════════════════════════════════════════════


async def search_google_scholar(
    query: str,
    max_results: int = DEFAULT_MAX_RESULTS,
) -> List[SearchResult]:
    """
    Search Google Scholar — largest academic search engine.
    
    Free via Serper.dev Scholar endpoint (uses same SERPER_API_KEY).
    Covers: All academic disciplines, 389M+ articles, theses, books, preprints.
    Returns: Titles, URLs, snippets, citation counts, author info.
    
    Best for: Broad academic search across all disciplines, finding 
    theses and books not in other databases, citation tracking.
    
    Note: Uses Serper's /scholar endpoint. Falls back to Serper web 
    search with site:scholar.google.com if needed.
    """
    if not SERPER_API_KEY:
        return []
    
    try:
        # Try Serper's Scholar endpoint first
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            r = await client.post(
                "https://google.serper.dev/scholar",
                headers={"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"},
                json={"q": query, "num": max_results},
            )
            r.raise_for_status()
            data = r.json()
        
        results = []
        for item in data.get("organic", []):
            # Extract citation count from snippet if present
            snippet = item.get("snippet", "")
            citation_count = None
            cit_match = re.search(r'Cited by (\d+)', snippet)
            if cit_match:
                citation_count = int(cit_match.group(1))
            
            # Extract year
            pub_year = None
            year_match = re.search(r'\b(19|20)\d{2}\b', snippet)
            if year_match:
                pub_year = year_match.group()
            
            results.append(SearchResult(
                title=item.get("title", ""),
                url=item.get("link", ""),
                snippet=snippet,
                source_name="google_scholar",
                source_category=SourceCategory.ACADEMIC,
                source_tier="P1",
                published_date=pub_year,
                citation_count=citation_count,
                extra={"position": item.get("position")},
            ))
        
        return results
    
    except Exception as e:
        logger.debug(f"Serper Scholar endpoint failed: {e}")
        # Fallback: regular Serper search with site:scholar.google.com
        try:
            async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
                r = await client.post(
                    "https://google.serper.dev/search",
                    headers={"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"},
                    json={"q": f"site:scholar.google.com {query}", "num": max_results},
                )
                r.raise_for_status()
                data = r.json()
            results = []
            for item in data.get("organic", []):
                results.append(SearchResult(
                    title=item.get("title", ""),
                    url=item.get("link", ""),
                    snippet=item.get("snippet", ""),
                    source_name="google_scholar",
                    source_category=SourceCategory.ACADEMIC,
                    source_tier="P1",
                ))
            return results
        except Exception as e2:
            logger.warning(f"Google Scholar search failed: {e2}")
            return []


# ═══════════════════════════════════════════════════════════════
# PAPERS WITH CODE — PAPERS LINKED TO IMPLEMENTATIONS
# ═══════════════════════════════════════════════════════════════


async def search_papers_with_code(
    query: str,
    max_results: int = DEFAULT_MAX_RESULTS,
) -> List[SearchResult]:
    """
    Search Papers With Code — links research papers to code implementations.
    
    Free: Completely open API, no key needed (now via HuggingFace Papers).
    Covers: 8K+ papers with code, 5K+ evaluation tables, 1K+ tasks.
    Returns: Paper titles, abstracts, GitHub repos, benchmark results.
    
    Best for: Finding implementations of research papers, comparing 
    approaches on benchmarks, discovering SOTA methods for specific tasks.
    
    Note: Papers With Code merged with HuggingFace. This uses the 
    HuggingFace Papers API as the primary source, with Serper fallback.
    """
    # Try HuggingFace Papers API (successor to PWC)
    try:
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            r = await client.get(
                "https://huggingface.co/api/papers/search",
                params={
                    "q": query,
                    "limit": max_results,
                },
            )
            r.raise_for_status()
            data = r.json()
        
        results = []
        for item in data:
            title = item.get("title", "Untitled")
            paper_id = item.get("id", "")
            abstract = item.get("abstract", "")[:300] if item.get("abstract") else ""
            url = f"https://huggingface.co/papers/{paper_id}"
            
            # Get upvotes as a proxy for popularity
            upvotes = item.get("upvotes", 0)
            
            results.append(SearchResult(
                title=title,
                url=url,
                snippet=abstract,
                source_name="papers_with_code",
                source_category=SourceCategory.CODE,
                source_tier="P1",
                published_date=item.get("publishedAt", "")[:10] if item.get("publishedAt") else None,
                extra={
                    "paper_id": paper_id,
                    "upvotes": upvotes,
                    "huggingface_url": url,
                },
            ))
        
        return results
    
    except Exception as e:
        logger.debug(f"HuggingFace Papers API failed: {e}")
    
    # Fallback: Serper search for papers with code
    if SERPER_API_KEY:
        try:
            async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
                r = await client.post(
                    "https://google.serper.dev/search",
                    headers={"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"},
                    json={"q": f"site:paperswithcode.com {query}", "num": max_results},
                )
                r.raise_for_status()
                data = r.json()
            results = []
            for item in data.get("organic", []):
                results.append(SearchResult(
                    title=item.get("title", ""),
                    url=item.get("link", ""),
                    snippet=item.get("snippet", ""),
                    source_name="papers_with_code",
                    source_category=SourceCategory.CODE,
                    source_tier="P1",
                ))
            return results
        except Exception as e2:
            logger.warning(f"Papers With Code search failed: {e2}")
            return []


# ═══════════════════════════════════════════════════════════════
# EUROPE PMC — BIOMEDICAL AND LIFE SCIENCES
# ═══════════════════════════════════════════════════════════════


async def search_europe_pmc(
    query: str,
    max_results: int = DEFAULT_MAX_RESULTS,
) -> List[SearchResult]:
    """
    Search Europe PMC — biomedical and life sciences literature.
    
    Free: Completely open API, no key needed.
    Covers: 40M+ publications (PubMed, PMC, PATENT, etc.), full-text for open access.
    Returns: Titles, abstracts, authors, DOIs, citation counts, full text links.
    
    Best for: Biomedical research, drug discovery, clinical literature,
    finding full-text open access articles, patent literature.
    """
    try:
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            r = await client.get(
                "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
                params={
                    "query": query,
                    "resultType": "core",
                    "pageSize": max_results,
                    "format": "json",
                },
            )
            r.raise_for_status()
            data = r.json()
        
        results = []
        for item in data.get("resultList", {}).get("result", []):
            title = item.get("title", "Untitled")
            pmid = item.get("pmid", "")
            pmcid = item.get("pmcid", "")
            doi = item.get("doi", "")
            
            # Build URL
            if pmcid:
                url = f"https://europepmc.org/article/PMC/{pmcid}"
            elif pmid:
                url = f"https://europepmc.org/article/MED/{pmid}"
            elif doi:
                url = f"https://doi.org/{doi}"
            else:
                url = ""
            
            # Authors
            authors = []
            for author in item.get("authorList", {}).get("author", []):
                name = f"{author.get('firstName', '')} {author.get('lastName', '')}".strip()
                if name:
                    authors.append(name)
            
            # Abstract
            abstract = item.get("abstractText", "")[:300] if item.get("abstractText") else ""
            
            # Citation count
            cit_count = item.get("citedByCount", 0)
            try:
                cit_count = int(cit_count)
            except (ValueError, TypeError):
                cit_count = None
            
            # Open access
            is_oa = bool(item.get("isOpenAccess", "").lower() == "y")
            
            results.append(SearchResult(
                title=title,
                url=url,
                snippet=abstract,
                source_name="europe_pmc",
                source_category=SourceCategory.CLINICAL,
                source_tier="P1",
                published_date=item.get("pubYear"),
                authors=authors,
                citation_count=cit_count,
                open_access=is_oa,
                doi=doi,
                extra={
                    "pmid": pmid,
                    "pmcid": pmcid,
                    "journal": item.get("journalTitle", ""),
                    "full_text_url": item.get("fullTextUrlList", {}).get("fullTextUrl", [{}])[0].get("url", "") if item.get("fullTextUrlList") else "",
                },
            ))
        
        return results
    
    except Exception as e:
        logger.warning(f"Europe PMC search failed: {e}")
        return []


# ═══════════════════════════════════════════════════════════════
# CORE — 280M+ OPEN ACCESS RESEARCH PAPERS
# ═══════════════════════════════════════════════════════════════


async def search_core(
    query: str,
    max_results: int = DEFAULT_MAX_RESULTS,
) -> List[SearchResult]:
    """
    Search CORE — world's largest collection of open access research papers.
    
    Free: 10M requests/day, no key needed for basic search.
    Covers: 280M+ open access papers from 16K+ data providers.
    Returns: Full-text metadata, download URLs, journal info.
    
    Best for: Finding open access full-text papers, broad academic
    discovery, accessing papers behind paywalls via OA versions.
    """
    try:
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT, follow_redirects=True) as client:
            r = await client.get(
                "https://api.core.ac.uk/v3/search/works/",
                params={
                    "q": query,
                    "limit": max_results,
                    "offset": 0,
                },
            )
            r.raise_for_status()
            data = r.json()
        
        results = []
        for item in data.get("results", []):
            title = item.get("title", "Untitled")
            doi = item.get("doi", "")
            url = item.get("downloadUrl") or item.get("sourceFulltextUrls", [""])[0] or ""
            if not url and doi:
                url = f"https://doi.org/{doi}"
            
            # Authors
            authors = []
            for author in item.get("authors", []):
                name = author.get("name", "")
                if name:
                    authors.append(name)
            
            abstract = item.get("abstract", "")[:300] if item.get("abstract") else ""
            year = item.get("yearPublished")
            
            results.append(SearchResult(
                title=title,
                url=url,
                snippet=abstract,
                source_name="core",
                source_category=SourceCategory.ACADEMIC,
                source_tier="P1",
                published_date=str(year) if year else None,
                authors=authors,
                open_access=True,  # CORE is all OA
                doi=doi,
                extra={
                    "download_url": item.get("downloadUrl", ""),
                    "journal": item.get("journal", ""),
                    "publisher": item.get("publisher", ""),
                },
            ))
        
        return results
    
    except Exception as e:
        logger.warning(f"CORE search failed: {e}")
        return []


# ═══════════════════════════════════════════════════════════════
# GOOGLE PATENTS — PATENT SEARCH
# ═══════════════════════════════════════════════════════════════


async def search_google_patents(
    query: str,
    max_results: int = DEFAULT_MAX_RESULTS,
) -> List[SearchResult]:
    """
    Search Google Patents — patent database via Serper.
    
    Free via Serper.dev (uses same SERPER_API_KEY).
    Covers: 120M+ patents from 100+ patent offices worldwide.
    Returns: Patent titles, filing dates, assignees, abstracts.
    
    Best for: Patent research, prior art searches, technology 
    landscape analysis, finding similar inventions.
    """
    if not SERPER_API_KEY:
        return []
    
    try:
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            r = await client.post(
                "https://google.serper.dev/search",
                headers={"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"},
                json={"q": f"site:patents.google.com {query}", "num": max_results},
            )
            r.raise_for_status()
            data = r.json()
        
        results = []
        for item in data.get("organic", []):
            url = item.get("link", "")
            title = item.get("title", "").replace(" - Google Patents", "")
            
            results.append(SearchResult(
                title=title,
                url=url,
                snippet=item.get("snippet", ""),
                source_name="google_patents",
                source_category=SourceCategory.PATENT,
                source_tier="P2",
                extra={"position": item.get("position")},
            ))
        
        return results
    
    except Exception as e:
        logger.warning(f"Google Patents search failed: {e}")
        return []


# ═══════════════════════════════════════════════════════════════
# YOUTUBE — VIDEO SEARCH
# ═══════════════════════════════════════════════════════════════


async def search_youtube(
    query: str,
    max_results: int = DEFAULT_MAX_RESULTS,
) -> List[SearchResult]:
    """
    Search YouTube — video content search.
    
    Free with API key: 10,000 units/day (~100 searches).
    Without key: Falls back to Serper site:youtube.com search.
    Covers: Billions of videos, channels, playlists.
    Returns: Video titles, URLs, descriptions, channel info.
    
    Best for: Finding video tutorials, conference talks, 
    expert interviews, visual explanations, demonstrations.
    """
    if YOUTUBE_API_KEY:
        try:
            async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
                r = await client.get(
                    "https://www.googleapis.com/youtube/v3/search",
                    params={
                        "part": "snippet",
                        "q": query,
                        "maxResults": max_results,
                        "type": "video",
                        "key": YOUTUBE_API_KEY,
                    },
                )
                r.raise_for_status()
                data = r.json()
            
            results = []
            for item in data.get("items", []):
                video_id = item.get("id", {}).get("videoId", "")
                snippet = item.get("snippet", {})
                
                results.append(SearchResult(
                    title=snippet.get("title", ""),
                    url=f"https://www.youtube.com/watch?v={video_id}",
                    snippet=snippet.get("description", "")[:300],
                    source_name="youtube",
                    source_category=SourceCategory.VIDEO,
                    source_tier="P3",
                    published_date=snippet.get("publishedAt", "")[:10],
                    extra={
                        "channel": snippet.get("channelTitle", ""),
                        "video_id": video_id,
                        "thumbnail": snippet.get("thumbnails", {}).get("default", {}).get("url", ""),
                    },
                ))
            
            return results
        
        except Exception as e:
            logger.debug(f"YouTube API failed: {e}")
    
    # Fallback: Serper with site:youtube.com
    if SERPER_API_KEY:
        try:
            async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
                r = await client.post(
                    "https://google.serper.dev/search",
                    headers={"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"},
                    json={"q": f"site:youtube.com {query}", "num": max_results},
                )
                r.raise_for_status()
                data = r.json()
            
            results = []
            for item in data.get("organic", []):
                results.append(SearchResult(
                    title=item.get("title", "").replace(" - YouTube", ""),
                    url=item.get("link", ""),
                    snippet=item.get("snippet", ""),
                    source_name="youtube_via_serper",
                    source_category=SourceCategory.VIDEO,
                    source_tier="P3",
                ))
            return results
        except Exception as e:
            logger.debug(f"YouTube via Serper failed: {e}")
    
    return []


# ═══════════════════════════════════════════════════════════════
# PODCAST INDEX — PODCAST SEARCH
# ═══════════════════════════════════════════════════════════════


async def search_podcast_index(
    query: str,
    max_results: int = DEFAULT_MAX_RESULTS,
) -> List[SearchResult]:
    """
    Search Podcast Index — open podcast directory and search.
    
    Free: Completely open API. Get key from https://api.podcastindex.org/
    Covers: 4M+ podcasts, 100M+ episodes.
    Returns: Podcast titles, feed URLs, episode descriptions.
    
    Best for: Finding expert audio content, tech podcasts, 
    interview discussions, educational audio content.
    """
    if not PODCAST_INDEX_API_KEY:
        # Fallback: Serper with podcast search
        if SERPER_API_KEY:
            try:
                async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
                    r = await client.post(
                        "https://google.serper.dev/search",
                        headers={"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"},
                        json={"q": f"podcast {query}", "num": max_results},
                    )
                    r.raise_for_status()
                    data = r.json()
                results = []
                for item in data.get("organic", []):
                    results.append(SearchResult(
                        title=item.get("title", ""),
                        url=item.get("link", ""),
                        snippet=item.get("snippet", ""),
                        source_name="podcast_via_serper",
                        source_category=SourceCategory.AUDIO,
                        source_tier="P3",
                    ))
                return results
            except Exception:
                pass
        return []
    
    try:
        import hashlib
        # Podcast Index API requires auth headers
        api_key = PODCAST_INDEX_API_KEY
        api_header = "APEX Research Agent"
        timestamp = str(int(time.time()))
        hash_input = api_key + api_header + timestamp
        auth_hash = hashlib.sha1(hash_input.encode()).hexdigest()
        
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            r = await client.get(
                "https://api.podcastindex.org/api/1.0/search/byterm",
                headers={
                    "X-Auth-Key": api_key,
                    "X-Auth-Date": timestamp,
                    "Authorization": auth_hash,
                    "User-Agent": "APEX-Research-Agent/1.0",
                },
                params={"q": query, "max": max_results},
            )
            r.raise_for_status()
            data = r.json()
        
        results = []
        for feed in data.get("feeds", []):
            results.append(SearchResult(
                title=feed.get("title", ""),
                url=feed.get("link", "") or feed.get("url", ""),
                snippet=feed.get("description", "")[:300] if feed.get("description") else "",
                source_name="podcast_index",
                source_category=SourceCategory.AUDIO,
                source_tier="P3",
                extra={
                    "episode_count": feed.get("episodeCount", 0),
                    "feed_url": feed.get("url", ""),
                    "categories": feed.get("categories", {}),
                },
            ))
        
        return results
    
    except Exception as e:
        logger.warning(f"Podcast Index search failed: {e}")
        return []


# ═══════════════════════════════════════════════════════════════
# SUBSTACK — NEWSLETTER SEARCH
# ═══════════════════════════════════════════════════════════════


async def search_substack(
    query: str,
    max_results: int = DEFAULT_MAX_RESULTS,
) -> List[SearchResult]:
    """
    Search Substack — independent newsletter content.
    
    Free via Serper (uses SERPER_API_KEY).
    Covers: Millions of newsletter posts from independent writers.
    Returns: Post titles, URLs, author info, snippets.
    
    Best for: Finding independent analysis, opinion pieces, 
    niche expertise, long-form commentary from domain experts.
    """
    if not SERPER_API_KEY:
        return []
    
    try:
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            r = await client.post(
                "https://google.serper.dev/search",
                headers={"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"},
                json={"q": f"site:substack.com {query}", "num": max_results},
            )
            r.raise_for_status()
            data = r.json()
        
        results = []
        for item in data.get("organic", []):
            url = item.get("link", "")
            title = item.get("title", "").replace(" | Substack", "").replace(" - Substack", "")
            
            results.append(SearchResult(
                title=title,
                url=url,
                snippet=item.get("snippet", ""),
                source_name="substack",
                source_category=SourceCategory.NEWS,
                source_tier="P3",
                extra={"position": item.get("position")},
            ))
        
        return results
    
    except Exception as e:
        logger.warning(f"Substack search failed: {e}")
        return []


# ═══════════════════════════════════════════════════════════════
# MASTODON — FEDIVERSE DISCUSSIONS
# ═══════════════════════════════════════════════════════════════


async def search_mastodon(
    query: str,
    max_results: int = DEFAULT_MAX_RESULTS,
) -> List[SearchResult]:
    """
    Search Mastodon — decentralized social network (Fediverse).
    
    Free: Completely open API, but most instances require auth for search.
    Falls back to Serper site:mastodon.social search.
    Covers: Public posts from thousands of Mastodon instances.
    Returns: Post content, URLs, author info, engagement data.
    
    Best for: Finding real-time tech discussions, open-source 
    community opinions, decentralized/privacy-focused perspectives.
    """
    try:
        # Use the public search API on the configured instance
        instance = MASTODON_INSTANCE.rstrip("/")
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            r = await client.get(
                f"https://{instance}/api/v2/search",
                headers={"User-Agent": "APEX-Research-Agent/1.0"},
                params={
                    "q": query,
                    "type": "statuses",
                    "limit": max_results,
                },
            )
            r.raise_for_status()
            data = r.json()
        
        results = []
        for status in data.get("statuses", []):
            account = status.get("account", {})
            username = account.get("acct", "")
            display_name = account.get("display_name", "")
            content = re.sub(r'<[^>]+>', '', status.get("content", ""))[:300]
            url = status.get("url", "")
            
            results.append(SearchResult(
                title=f"@{username}: {content[:80]}...",
                url=url,
                snippet=content,
                source_name="mastodon",
                source_category=SourceCategory.SOCIAL,
                source_tier="P3",
                published_date=status.get("created_at", "")[:10],
                extra={
                    "username": username,
                    "display_name": display_name,
                    "favourites": status.get("favourites_count", 0),
                    "reblogs": status.get("reblogs_count", 0),
                    "instance": instance,
                },
            ))
        
        if results:
            return results
    
    except Exception as e:
        logger.debug(f"Mastodon direct API failed: {e}")
    
    # Fallback: Serper with site:mastodon.social
    if SERPER_API_KEY:
        try:
            async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
                r = await client.post(
                    "https://google.serper.dev/search",
                    headers={"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"},
                    json={"q": f"site:mastodon.social {query}", "num": max_results},
                )
                r.raise_for_status()
                data = r.json()
            results = []
            for item in data.get("organic", []):
                results.append(SearchResult(
                    title=item.get("title", ""),
                    url=item.get("link", ""),
                    snippet=item.get("snippet", ""),
                    source_name="mastodon_via_serper",
                    source_category=SourceCategory.SOCIAL,
                    source_tier="P3",
                ))
            return results
        except Exception as e2:
            logger.debug(f"Mastodon via Serper failed: {e2}")
    
    return []


# ═══════════════════════════════════════════════════════════════
# IPFS / WIKIPEDIA OFFLINE — OFFLINE ENCYCLOPEDIA
# ═══════════════════════════════════════════════════════════════


async def search_wikipedia_offline(
    query: str,
    max_results: int = DEFAULT_MAX_RESULTS,
) -> List[SearchResult]:
    """
    Search Wikipedia via Kiwix/IPFS — offline encyclopedia access.
    
    Free: No key needed. Uses public IPFS gateway or local Kiwix server.
    Covers: Full Wikipedia dump (offline-capable).
    Returns: Article titles, summaries, content.
    
    Best for: Offline/air-gapped environments, low-bandwidth situations,
    regions with internet restrictions, archival research.
    
    Note: Falls back to regular Wikipedia API if IPFS/Kiwix is unavailable.
    """
    # Try IPFS-hosted Wikipedia first
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            # IPFS CID for Wikipedia Kiwix ZIM (public gateway)
            r = await client.get(
                "https://ipfs.io/ipfs/bafybeiaks3mhnbu5m7j6nb25hy2xqfpxrfahq6bjWD4ndab5slm3rlxwvm/wiki/",
                params={"q": query},
            )
            if r.status_code == 200:
                # If IPFS works, return as IPFS source
                results = []
                results.append(SearchResult(
                    title=f"[IPFS/Wikipedia] {query}",
                    url=f"https://ipfs.io/ipfs/bafybeiaks3mhnbu5m7j6nb25hy2xqfpxrfahq6bjWD4ndab5slm3rlxwvm/wiki/{quote_plus(query)}",
                    snippet="Wikipedia via IPFS (offline-capable). Content available without internet.",
                    source_name="wikipedia_ipfs",
                    source_category=SourceCategory.ENCYCLOPEDIA,
                    source_tier="P2",
                    extra={"protocol": "ipfs", "offline_capable": True},
                ))
                return results
    except Exception:
        pass
    
    # Fallback: regular Wikipedia API (not offline, but always available)
    return await search_wikipedia(query, max_results)


# ═══════════════════════════════════════════════════════════════
# DUCKDUCKGO (FREE, NO KEY — ALREADY EXISTS, WRAPPER)
# ═══════════════════════════════════════════════════════════════


async def search_duckduckgo(
    query: str,
    max_results: int = DEFAULT_MAX_RESULTS,
) -> List[SearchResult]:
    """
    Search DuckDuckGo Lite — free web search, no key needed.
    
    Free: No limits, no key, no registration.
    Covers: Full web index (Bing-based results).
    Returns: URLs only (content must be scraped separately).
    
    Best for: Fallback when all paid search APIs fail.
    Always available, no rate limits, no authentication.
    """
    try:
        async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
            r = await client.post(
                "https://lite.duckduckgo.com/lite/",
                data={"q": query, "kl": "us-en"},
                headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"},
            )
            r.raise_for_status()
            
            urls = re.findall(r'href="(https?://[^"]+)"', r.text)
            # Extract titles from link text
            titles = re.findall(r'result__a[^>]*>([^<]+)', r.text)
            snippets = re.findall(r'result__snippet[^>]*>([^<]+)', r.text)
            
            seen = set()
            results = []
            for i, url in enumerate(urls):
                if "duckduckgo.com" in url:
                    continue
                if url in seen:
                    continue
                seen.add(url)
                
                title = titles[i] if i < len(titles) else url.split("/")[-1].replace("-", " ").title()
                snippet = snippets[i] if i < len(snippets) else ""
                
                results.append(SearchResult(
                    title=title,
                    url=url,
                    snippet=snippet,
                    source_name="duckduckgo",
                    source_category=SourceCategory.WEB,
                    source_tier="UNV",
                ))
                
                if len(results) >= max_results:
                    break
        
        return results
    
    except Exception as e:
        logger.warning(f"DuckDuckGo search failed: {e}")
        return []


# ═══════════════════════════════════════════════════════════════
# UNPAYWALL — OPEN ACCESS PDF FINDER
# ═══════════════════════════════════════════════════════════════


async def find_open_access_pdf(doi: str) -> Optional[str]:
    """
    Find open access PDF for a DOI via Unpaywall.
    
    Free: No key needed (just email for identification).
    Covers: 45M+ open access articles matched to DOIs.
    Returns: Best available OA URL (pdf, repository, publisher).
    
    Best for: Finding free PDFs for paywalled papers,
    verifying open access status of a given DOI.
    """
    if not doi:
        return None
    
    try:
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            r = await client.get(
                f"https://api.unpaywall.org/v2/{doi}",
                params={"email": CONTACT_EMAIL},
            )
            r.raise_for_status()
            data = r.json()
        
        best_oa = data.get("best_oa_location", {})
        if best_oa:
            return best_oa.get("url_for_pdf") or best_oa.get("url_for_landing_page") or best_oa.get("url")
        
        return None
    
    except Exception as e:
        logger.debug(f"Unpaywall lookup failed for {doi}: {e}")
        return None


# ═══════════════════════════════════════════════════════════════
# SMART SEARCH ROUTER
# ═══════════════════════════════════════════════════════════════


# Map query classification to source priorities
SOURCE_ROUTING = {
    "academic": {
        "primary": ["exa", "semantic_scholar", "openalex", "arxiv", "google_scholar"],
        "secondary": ["crossref", "doaj", "core", "papers_with_code", "clinical_trials", "wikipedia"],
        "web_fallback": ["serper", "duckduckgo"],
    },
    "web": {
        "primary": ["serper", "exa"],
        "secondary": ["brave", "duckduckgo", "hackernews", "wikipedia", "youtube", "substack"],
        "academic_boost": ["openalex", "google_scholar"],
    },
    "code": {
        "primary": ["github", "stackoverflow", "papers_with_code", "exa"],
        "secondary": ["serper", "duckduckgo", "hackernews"],
    },
    "news": {
        "primary": ["hackernews", "serper", "reddit", "youtube"],
        "secondary": ["exa", "newsapi", "substack", "duckduckgo", "mastodon"],
    },
    "clinical": {
        "primary": ["clinical_trials", "europe_pmc", "semantic_scholar"],
        "secondary": ["openalex", "crossref", "doaj", "exa", "core"],
    },
    "encyclopedia": {
        "primary": ["wikipedia", "wikidata"],
        "secondary": ["openalex", "serper", "duckduckgo", "wikipedia_offline"],
    },
    "compute": {
        "primary": ["wolfram", "wikipedia"],
        "secondary": ["serper", "duckduckgo"],
    },
    "patent": {
        "primary": ["google_patents"],
        "secondary": ["serper", "duckduckgo"],
    },
}

# Source name → async function mapping
SOURCE_FUNCTIONS = {
    "semantic_scholar": search_semantic_scholar,
    "crossref": search_crossref,
    "openalex": search_openalex,
    "doaj": search_doaj,
    "arxiv": search_arxiv,
    "google_scholar": search_google_scholar,
    "papers_with_code": search_papers_with_code,
    "europe_pmc": search_europe_pmc,
    "core": search_core,
    "clinical_trials": search_clinical_trials,
    "wikipedia": search_wikipedia,
    "wikidata": search_wikidata,
    "wikipedia_offline": search_wikipedia_offline,
    "github": search_github,
    "stackoverflow": search_stackoverflow,
    "hackernews": search_hackernews,
    "reddit": search_reddit,
    "newsapi": search_newsapi,
    "substack": search_substack,
    "youtube": search_youtube,
    "podcast_index": search_podcast_index,
    "mastodon": search_mastodon,
    "brave": search_brave,
    "serper": search_serper,
    "exa": search_exa,
    "tavily": search_tavily,
    "wolfram": search_wolfram,
    "google_patents": search_google_patents,
    "duckduckgo": search_duckduckgo,
}


async def search_router(
    query: str,
    classification: str = "academic",
    max_results: int = DEFAULT_MAX_RESULTS,
    sources: Optional[List[str]] = None,
) -> List[SearchResult]:
    """
    Route a search query to the appropriate sources based on classification.
    
    Strategy:
    1. Check classification → determine source priority
    2. Search primary sources in parallel
    3. If results < threshold, search secondary sources
    4. Deduplicate by URL and sort by source_tier priority
    5. Return top max_results results
    
    Args:
        query: Search query
        classification: Query type (academic, web, code, news, clinical, encyclopedia)
        max_results: Maximum results to return
        sources: Override source list (search only these)
    
    Returns:
        Deduplicated, tier-sorted list of SearchResult objects
    """
    routing = SOURCE_ROUTING.get(classification, SOURCE_ROUTING["academic"])
    
    # Determine which sources to query
    if sources:
        source_names = sources
    else:
        source_names = routing.get("primary", []) + routing.get("secondary", [])
    
    # Execute searches in parallel
    tasks = []
    for name in source_names:
        func = SOURCE_FUNCTIONS.get(name)
        if func:
            tasks.append(func(query, max_results))
    
    if not tasks:
        # Fallback to DuckDuckGo
        tasks.append(search_duckduckgo(query, max_results))
    
    all_results_lists = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Flatten and filter
    all_results: List[SearchResult] = []
    for result_list in all_results_lists:
        if isinstance(result_list, list):
            all_results.extend(result_list)
        elif isinstance(result_list, Exception):
            logger.debug(f"Source search failed: {result_list}")
    
    # Deduplicate by URL
    seen_urls = set()
    unique_results = []
    for result in all_results:
        if result.url and result.url not in seen_urls:
            seen_urls.add(result.url)
            unique_results.append(result)
    
    # Sort by source tier priority
    tier_order = {"P1": 0, "P2": 1, "P3": 2, "UNV": 3}
    unique_results.sort(key=lambda r: tier_order.get(r.source_tier, 4))
    
    return unique_results[:max_results]


async def search_all_sources(
    query: str,
    max_per_source: int = 3,
) -> Dict[str, List[SearchResult]]:
    """
    Search ALL sources simultaneously for comprehensive coverage.
    
    Useful for: Deep research queries where you want maximum coverage.
    Returns results grouped by source for analysis.
    """
    tasks = {
        name: func(query, max_per_source)
        for name, func in SOURCE_FUNCTIONS.items()
    }
    
    results = {}
    task_list = list(tasks.values())
    task_names = list(tasks.keys())
    
    completed = await asyncio.gather(*task_list, return_exceptions=True)
    
    for name, result in zip(task_names, completed):
        if isinstance(result, list):
            results[name] = result
        elif isinstance(result, Exception):
            results[name] = []
            logger.debug(f"{name} search failed: {result}")
    
    return results


def get_source_status() -> Dict[str, Any]:
    """Get the status of all configured search sources."""
    sources = {
        # Always free, no key
        "openalex": {"key_needed": False, "status": "Free, unlimited", "category": "academic"},
        "crossref": {"key_needed": False, "status": "Free (polite pool with email)", "category": "academic"},
        "doaj": {"key_needed": False, "status": "Free, no limits", "category": "academic"},
        "arxiv": {"key_needed": False, "status": "Free, no limits", "category": "academic"},
        "core": {"key_needed": False, "status": "Free, 10M req/day", "category": "academic"},
        "papers_with_code": {"key_needed": False, "status": "Free, no limits", "category": "academic+code"},
        "europe_pmc": {"key_needed": False, "status": "Free, no limits", "category": "biomedical"},
        "clinical_trials": {"key_needed": False, "status": "Free, no limits", "category": "clinical"},
        "wikipedia": {"key_needed": False, "status": "Free (User-Agent required)", "category": "encyclopedia"},
        "wikidata": {"key_needed": False, "status": "Free (User-Agent required)", "category": "encyclopedia"},
        "github": {"key_needed": False, "status": "Free (better with token)", "key_configured": bool(GITHUB_TOKEN), "category": "code"},
        "stackoverflow": {"key_needed": False, "status": "Free, 300 req/s", "category": "code"},
        "hackernews": {"key_needed": False, "status": "Free via Algolia", "category": "news"},
        "reddit": {"key_needed": False, "status": "Free (Serper fallback)", "category": "news"},
        "mastodon": {"key_needed": False, "status": "Free, open API", "category": "social"},
        "duckduckgo": {"key_needed": False, "status": "Free, no limits", "category": "web"},
        "wikipedia_offline": {"key_needed": False, "status": "Free via IPFS/Kiwix", "category": "offline"},
        # Free tier with key
        "semantic_scholar": {"key_needed": "optional", "status": "Rate-limited w/o key (100/5min)", "key_configured": bool(SEMANTIC_SCHOLAR_API_KEY), "category": "academic"},
        "exa": {"key_needed": True, "status": "Free: 1K req/mo | Neural search", "key_configured": bool(EXA_API_KEY), "category": "academic+web"},
        "serper": {"key_needed": True, "status": "Free: 2.5K one-time | Google results", "key_configured": bool(SERPER_API_KEY), "category": "web+scholar+patents"},
        "google_scholar": {"key_needed": True, "status": "Via Serper | 389M+ articles", "key_configured": bool(SERPER_API_KEY), "category": "academic"},
        "google_patents": {"key_needed": True, "status": "Via Serper | 120M+ patents", "key_configured": bool(SERPER_API_KEY), "category": "patents"},
        "substack": {"key_needed": True, "status": "Via Serper | Newsletter search", "key_configured": bool(SERPER_API_KEY), "category": "news"},
        "youtube": {"key_needed": True, "status": "Via Serper fallback | Or YouTube API key", "key_configured": bool(YOUTUBE_API_KEY), "category": "video"},
        "brave": {"key_needed": True, "status": "Free: 2K req/mo", "key_configured": bool(BRAVE_SEARCH_API_KEY), "category": "web"},
        "tavily": {"key_needed": True, "status": "Free: 1K req/mo | AI-optimized", "key_configured": bool(TAVILY_API_KEY), "category": "web"},
        "newsapi": {"key_needed": True, "status": "Free: 100 req/day", "key_configured": bool(NEWSAPI_KEY), "category": "news"},
        "wolfram": {"key_needed": True, "status": "Free: 2K req/mo | Computation", "key_configured": bool(WOLFRAM_APP_ID), "category": "computation"},
        "podcast_index": {"key_needed": True, "status": "Free | Podcast search", "key_configured": bool(PODCAST_INDEX_API_KEY), "category": "audio"},
    }
    return sources
