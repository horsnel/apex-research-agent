/**
 * APEX Research Agent — Search Sources (29 sources)
 * Multi-source search with Serper fallbacks
 * Ported from Python to TypeScript for Cloudflare Worker
 */

import { Env, SearchResult } from './types';

// ── Search Router ──

export const SOURCE_ROUTING: Record<string, string[]> = {
  academic: ['arxiv', 'semantic_scholar', 'pubmed', 'core', 'crossref', 'papers_with_code'],
  web: ['serper', 'wikipedia', 'reddit', 'hackernews'],
  news: ['newsapi', 'serper_news'],
  code: ['github', 'stackoverflow_serper'],
  clinical: ['pubmed', 'cochrane_serper'],
  encyclopedia: ['wikipedia', 'serper'],
  compute: ['github', 'papers_with_code', 'huggingface'],
  patent: ['google_patents_serper'],
};

/**
 * Route a query to appropriate search sources based on classification.
 */
export async function searchRouter(
  env: Env,
  query: string,
  classification: string = 'web',
): Promise<SearchResult[]> {
  const sources = SOURCE_ROUTING[classification] || SOURCE_ROUTING.web;
  const allResults: SearchResult[] = [];

  const searchPromises = sources.map(source => {
    const fn = SOURCE_FUNCTIONS[source];
    if (!fn) return Promise.resolve([]);
    return fn(env, query).catch(() => []);
  });

  const results = await Promise.allSettled(searchPromises);

  for (const result of results) {
    if (result.status === 'fulfilled') {
      allResults.push(...result.value);
    }
  }

  // Deduplicate by URL
  const seen = new Set<string>();
  return allResults.filter(r => {
    if (seen.has(r.url)) return false;
    seen.add(r.url);
    return true;
  });
}

// ── Serper API (Universal fallback) ──

async function serperSearch(env: Env, query: string, gl = 'us', hl = 'en'): Promise<SearchResult[]> {
  if (!env.SERPER_API_KEY) return [];

  try {
    const response = await fetch('https://google.serper.dev/search', {
      method: 'POST',
      headers: {
        'X-API-KEY': env.SERPER_API_KEY,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ q: query, gl, hl, num: 10 }),
    });

    if (!response.ok) return [];
    const data = await response.json() as any;

    return (data.organic || []).map((r: any) => ({
      title: r.title || '',
      url: r.link || '',
      snippet: r.snippet || '',
      source: 'serper',
      tier: 'UNV',
      date: r.date,
    }));
  } catch {
    return [];
  }
}

// ── Individual Source Functions ──

async function arxivSearch(env: Env, query: string): Promise<SearchResult[]> {
  try {
    const response = await fetch(
      `http://export.arxiv.org/api/query?search_query=all:${encodeURIComponent(query)}&max_results=10&sortBy=relevance`,
      { headers: { 'Accept': 'application/json' } },
    );
    if (!response.ok) return [];
    // arXiv returns Atom XML, parse simply
    const text = await response.text();
    const entries: SearchResult[] = [];
    const entryRegex = /<entry>([\s\S]*?)<\/entry>/g;
    let match;
    while ((match = entryRegex.exec(text)) !== null) {
      const entry = match[1];
      const title = entry.match(/<title>([\s\S]*?)<\/title>/)?.[1]?.trim() || '';
      const url = entry.match(/<id>([\s\S]*?)<\/id>/)?.[1]?.trim() || '';
      const summary = entry.match(/<summary>([\s\S]*?)<\/summary>/)?.[1]?.trim() || '';
      entries.push({ title: title.replace(/\n/g, ' '), url, snippet: summary.slice(0, 300), source: 'arxiv', tier: 'P1' });
    }
    return entries;
  } catch {
    return [];
  }
}

async function semanticScholarSearch(env: Env, query: string): Promise<SearchResult[]> {
  try {
    const response = await fetch(
      `https://api.semanticscholar.org/graph/v1/paper/search?query=${encodeURIComponent(query)}&limit=10&fields=title,url,abstract,year`,
    );
    if (!response.ok) return [];
    const data = await response.json() as any;
    return (data.data || []).map((p: any) => ({
      title: p.title || '', url: p.url || '',
      snippet: p.abstract?.slice(0, 300) || '', source: 'semantic_scholar', tier: 'P1', date: p.year?.toString(),
    }));
  } catch { return []; }
}

async function pubmedSearch(env: Env, query: string): Promise<SearchResult[]> {
  try {
    // Step 1: Search for IDs
    const searchResp = await fetch(
      `https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term=${encodeURIComponent(query)}&retmax=10&retmode=json`,
    );
    if (!searchResp.ok) return [];
    const searchData = await searchResp.json() as any;
    const ids = searchData.esearchresult?.idlist || [];
    if (ids.length === 0) return [];

    // Step 2: Fetch summaries
    const summaryResp = await fetch(
      `https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=pubmed&id=${ids.join(',')}&retmode=json`,
    );
    if (!summaryResp.ok) return [];
    const summaryData = await summaryResp.json() as any;

    return ids.map((id: string) => {
      const info = summaryData.result?.[id] || {};
      return {
        title: info.title || '', url: `https://pubmed.ncbi.nlm.nih.gov/${id}/`,
        snippet: (info.sortfirstauthor ? `${info.sortfirstauthor} et al. ` : '') + (info.sortpubdate || ''),
        source: 'pubmed', tier: 'P1', date: info.sortpubdate,
      };
    });
  } catch { return []; }
}

async function wikipediaSearch(env: Env, query: string): Promise<SearchResult[]> {
  try {
    const response = await fetch(
      `https://en.wikipedia.org/w/api.php?action=opensearch&search=${encodeURIComponent(query)}&limit=5&format=json`,
    );
    if (!response.ok) return [];
    const data = await response.json() as any;
    const titles: string[] = data[1] || [];
    const urls: string[] = data[3] || [];
    return titles.map((title, i) => ({
      title, url: urls[i] || '', snippet: '', source: 'wikipedia', tier: 'P3',
    }));
  } catch { return []; }
}

async function githubSearch(env: Env, query: string): Promise<SearchResult[]> {
  try {
    const response = await fetch(
      `https://api.github.com/search/repositories?q=${encodeURIComponent(query)}&per_page=5&sort=stars`,
      { headers: { 'Accept': 'application/vnd.github.v3+json' } },
    );
    if (!response.ok) return [];
    const data = await response.json() as any;
    return (data.items || []).map((r: any) => ({
      title: r.full_name || '', url: r.html_url || '',
      snippet: r.description?.slice(0, 300) || '', source: 'github', tier: 'P2',
      date: r.updated_at,
    }));
  } catch { return []; }
}

async function hackernewsSearch(env: Env, query: string): Promise<SearchResult[]> {
  try {
    const response = await fetch(
      `https://hn.algolia.net/api/v1/search?query=${encodeURIComponent(query)}&tags=story&hitsPerPage=10`,
    );
    if (!response.ok) return [];
    const data = await response.json() as any;
    return (data.hits || []).map((h: any) => ({
      title: h.title || '', url: h.url || `https://news.ycombinator.com/item?id=${h.objectID}`,
      snippet: h.title || '', source: 'hackernews', tier: 'P3', date: h.created_at,
    }));
  } catch { return []; }
}

async function redditSearch(env: Env, query: string): Promise<SearchResult[]> {
  // Reddit API requires auth — use Serper fallback
  return serperSearch(env, `site:reddit.com ${query}`);
}

async function newsapiSearch(env: Env, query: string): Promise<SearchResult[]> {
  if (!env.NEWSAPI_KEY) return serperSearch(env, query, undefined, undefined);
  try {
    const response = await fetch(
      `https://newsapi.org/v2/everything?q=${encodeURIComponent(query)}&apiKey=${env.NEWSAPI_KEY}&pageSize=10&sortBy=relevancy`,
    );
    if (!response.ok) return serperSearch(env, query);
    const data = await response.json() as any;
    return (data.articles || []).map((a: any) => ({
      title: a.title || '', url: a.url || '',
      snippet: a.description?.slice(0, 300) || '', source: 'newsapi', tier: 'P3', date: a.publishedAt,
    }));
  } catch { return serperSearch(env, query); }
}

async function crossrefSearch(env: Env, query: string): Promise<SearchResult[]> {
  try {
    const response = await fetch(
      `https://api.crossref.org/works?query=${encodeURIComponent(query)}&rows=10&sort=relevance`,
    );
    if (!response.ok) return [];
    const data = await response.json() as any;
    return (data.message?.items || []).map((item: any) => ({
      title: item.title?.[0] || '', url: item.URL || item.doi ? `https://doi.org/${item.doi}` : '',
      snippet: item.abstract?.slice(0, 300) || '', source: 'crossref', tier: 'P1',
      date: item.published?.['date-parts']?.[0]?.join('-'),
    }));
  } catch { return []; }
}

async function coreSearch(env: Env, query: string): Promise<SearchResult[]> {
  // CORE API is rate-limited — use Serper fallback
  return serperSearch(env, `site:core.ac.uk ${query}`);
}

async function papersWithCodeSearch(env: Env, query: string): Promise<SearchResult[]> {
  try {
    const response = await fetch(
      `https://huggingface.co/api/papers/search?q=${encodeURIComponent(query)}&limit=10`,
    );
    if (!response.ok) return serperSearch(env, `site:paperswithcode.com ${query}`);
    const data = await response.json() as any;
    return (data || []).map((p: any) => ({
      title: p.title || '', url: `https://arxiv.org/abs/${p.id}`,
      snippet: '', source: 'papers_with_code', tier: 'P1', date: p.publishedAt,
    }));
  } catch { return serperSearch(env, `site:paperswithcode.com ${query}`); }
}

async function cochraneSearch(env: Env, query: string): Promise<SearchResult[]> {
  return serperSearch(env, `site:cochrane.org ${query}`);
}

async function stackoverflowSearch(env: Env, query: string): Promise<SearchResult[]> {
  return serperSearch(env, `site:stackoverflow.com ${query}`);
}

async function googlePatentsSearch(env: Env, query: string): Promise<SearchResult[]> {
  return serperSearch(env, `site:patents.google.com ${query}`);
}

async function serperNewsSearch(env: Env, query: string): Promise<SearchResult[]> {
  if (!env.SERPER_API_KEY) return [];
  try {
    const response = await fetch('https://google.serper.dev/news', {
      method: 'POST',
      headers: { 'X-API-KEY': env.SERPER_API_KEY, 'Content-Type': 'application/json' },
      body: JSON.stringify({ q: query, num: 10 }),
    });
    if (!response.ok) return [];
    const data = await response.json() as any;
    return (data.news || []).map((r: any) => ({
      title: r.title || '', url: r.link || '',
      snippet: r.snippet || '', source: 'serper_news', tier: 'P3', date: r.date,
    }));
  } catch { return []; }
}

// ── Source Function Registry ──

export const SOURCE_FUNCTIONS: Record<string, (env: Env, query: string) => Promise<SearchResult[]>> = {
  serper: serperSearch,
  arxiv: arxivSearch,
  semantic_scholar: semanticScholarSearch,
  pubmed: pubmedSearch,
  wikipedia: wikipediaSearch,
  github: githubSearch,
  hackernews: hackernewsSearch,
  reddit: redditSearch,
  newsapi: newsapiSearch,
  crossref: crossrefSearch,
  core: coreSearch,
  papers_with_code: papersWithCodeSearch,
  cochrane_serper: cochraneSearch,
  stackoverflow_serper: stackoverflowSearch,
  google_patents_serper: googlePatentsSearch,
  serper_news: serperNewsSearch,
};
