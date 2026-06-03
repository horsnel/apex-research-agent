"""
Multi-Source Search — unified search across 20+ information sources.

Categorized sources:
1. ACADEMIC: Semantic Scholar, Crossref, OpenAlex, DOAJ, arXiv, PubMed, CORE, Exa
2. GENERAL WEB: Serper (Google), Brave Search, DuckDuckGo, Tavily, Jina
3. ENCYCLOPEDIA: Wikipedia, Wikidata
4. CODE: GitHub, StackOverflow
5. NEWS: Hacker News, Reddit, NewsAPI
6. CLINICAL: ClinicalTrials.gov
7. PATENTS: Google Patents (via scraping)
8. CITATION: Unpaywall (open access PDFs)

Each source returns normalized SearchResult objects.
The search_router() function dispatches queries to the right sources
based on the query classification (academic, web, code, news, clinical).

Routing Priority (by query type):
  academic  → Exa(neural) > Semantic Scholar > OpenAlex > Crossref > DOAJ
  web       → Serper(Google) > Exa > Brave > DuckDuckGo
  code      → GitHub > StackOverflow > Exa
  news      → Hacker News > Serper > Reddit > NewsAPI
  clinical  → ClinicalTrials.gov > Semantic Scholar > OpenAlex
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
        "primary": ["exa", "semantic_scholar", "openalex", "arxiv"],
        "secondary": ["crossref", "doaj", "clinical_trials", "wikipedia"],
        "web_fallback": ["serper", "duckduckgo"],
    },
    "web": {
        "primary": ["serper", "exa"],
        "secondary": ["brave", "duckduckgo", "hackernews", "wikipedia"],
        "academic_boost": ["openalex"],
    },
    "code": {
        "primary": ["github", "stackoverflow", "exa"],
        "secondary": ["serper", "duckduckgo", "hackernews"],
    },
    "news": {
        "primary": ["hackernews", "serper", "reddit"],
        "secondary": ["exa", "newsapi", "duckduckgo"],
    },
    "clinical": {
        "primary": ["clinical_trials", "semantic_scholar"],
        "secondary": ["openalex", "crossref", "doaj", "exa"],
    },
    "encyclopedia": {
        "primary": ["wikipedia", "wikidata"],
        "secondary": ["openalex", "serper", "duckduckgo"],
    },
}

# Source name → async function mapping
SOURCE_FUNCTIONS = {
    "semantic_scholar": search_semantic_scholar,
    "crossref": search_crossref,
    "openalex": search_openalex,
    "doaj": search_doaj,
    "arxiv": search_arxiv,
    "clinical_trials": search_clinical_trials,
    "wikipedia": search_wikipedia,
    "wikidata": search_wikidata,
    "github": search_github,
    "stackoverflow": search_stackoverflow,
    "hackernews": search_hackernews,
    "reddit": search_reddit,
    "newsapi": search_newsapi,
    "brave": search_brave,
    "serper": search_serper,
    "exa": search_exa,
    "tavily": search_tavily,
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
        "openalex": {"key_needed": False, "status": "✅ Free, unlimited"},
        "crossref": {"key_needed": False, "status": "✅ Free (polite pool with email)"},
        "doaj": {"key_needed": False, "status": "✅ Free, no limits"},
        "arxiv": {"key_needed": False, "status": "✅ Free, no limits"},
        "clinical_trials": {"key_needed": False, "status": "✅ Free, no limits"},
        "wikipedia": {"key_needed": False, "status": "✅ Free (User-Agent required)"},
        "wikidata": {"key_needed": False, "status": "✅ Free (User-Agent required)"},
        "github": {"key_needed": False, "status": "✅ Free (better with token)", "key_configured": bool(GITHUB_TOKEN)},
        "stackoverflow": {"key_needed": False, "status": "✅ Free, 300 req/s"},
        "hackernews": {"key_needed": False, "status": "✅ Free via Algolia"},
        "reddit": {"key_needed": False, "status": "✅ Free (basic search)"},
        "duckduckgo": {"key_needed": False, "status": "✅ Free, no limits"},
        # Free tier with key
        "semantic_scholar": {"key_needed": "optional", "status": "⚠️ Rate-limited w/o key (100/5min)", "key_configured": bool(SEMANTIC_SCHOLAR_API_KEY)},
        "exa": {"key_needed": True, "status": "🔑 Free: 1K req/mo | Neural search", "key_configured": bool(EXA_API_KEY)},
        "serper": {"key_needed": True, "status": "🔑 Free: 2.5K one-time | Google results", "key_configured": bool(SERPER_API_KEY)},
        "brave": {"key_needed": True, "status": "🔑 Free: 2K req/mo", "key_configured": bool(BRAVE_SEARCH_API_KEY)},
        "tavily": {"key_needed": True, "status": "🔑 Free: 1K req/mo | AI-optimized", "key_configured": bool(TAVILY_API_KEY)},
        "newsapi": {"key_needed": True, "status": "🔑 Free: 100 req/day", "key_configured": bool(NEWSAPI_KEY)},
    }
    return sources
