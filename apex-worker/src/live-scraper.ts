/**
 * APEX Research Agent — Live Scraper
 * 3-tier scraper: Direct fetch → Jina Reader → Firecrawl
 */

import { Env, ScrapeResult } from './types';

/**
 * Live scrape with 3-tier fallback.
 */
export async function liveScrape(
  env: Env,
  options: { query?: string; urls?: string[] } = {},
): Promise<ScrapeResult[]> {
  const { query, urls } = options;

  // If URLs provided, scrape them directly
  if (urls && urls.length > 0) {
    const results = await Promise.allSettled(
      urls.slice(0, 5).map(url => scrapeSingleUrl(env, url)),
    );
    return results
      .filter((r): r is PromiseFulfilledResult<ScrapeResult> => r.status === 'fulfilled')
      .map(r => r.value);
  }

  // If query provided, use Serper to find URLs then scrape
  if (query) {
    const searchResults = await searchAndScrape(env, query);
    return searchResults;
  }

  return [];
}

/**
 * Scrape a single URL with 3-tier fallback.
 */
async function scrapeSingleUrl(env: Env, url: string): Promise<ScrapeResult> {
  // Tier 1: Direct fetch
  const direct = await directFetch(url);
  if (direct.success && direct.markdown.length > 100) return direct;

  // Tier 2: Jina Reader
  const jina = await jinaFetch(env, url);
  if (jina.success && jina.markdown.length > 100) return jina;

  // Tier 3: Firecrawl
  const firecrawl = await firecrawlFetch(env, url);
  if (firecrawl.success) return firecrawl;

  // Return best effort
  return direct.markdown.length > jina.markdown.length ? direct : jina;
}

/**
 * Tier 1: Direct fetch with basic text extraction.
 */
async function directFetch(url: string): Promise<ScrapeResult> {
  try {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 10000);

    const response = await fetch(url, {
      signal: controller.signal,
      headers: {
        'User-Agent': 'APEX-Research-Agent/2.0',
        'Accept': 'text/html,text/plain,application/json',
      },
    });
    clearTimeout(timeout);

    if (!response.ok) {
      return { url, markdown: '', title: '', success: false, error: `HTTP ${response.status}` };
    }

    const html = await response.text();
    const { title, text } = extractTextFromHTML(html);

    return { url, markdown: text, title, success: text.length > 50, error: '' };
  } catch (err: any) {
    return { url, markdown: '', title: '', success: false, error: err.message || String(err) };
  }
}

/**
 * Tier 2: Jina Reader API.
 */
async function jinaFetch(env: Env, url: string): Promise<ScrapeResult> {
  try {
    const headers: Record<string, string> = {
      'Accept': 'text/markdown',
    };
    if (env.JINA_API_KEY) {
      headers['Authorization'] = `Bearer ${env.JINA_API_KEY}`;
    }

    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 15000);

    const response = await fetch(`https://r.jina.ai/${url}`, {
      signal: controller.signal,
      headers,
    });
    clearTimeout(timeout);

    if (!response.ok) {
      return { url, markdown: '', title: '', success: false, error: `Jina HTTP ${response.status}` };
    }

    const markdown = await response.text();
    const title = extractTitleFromMarkdown(markdown);

    return { url, markdown, title, success: markdown.length > 50, error: '' };
  } catch (err: any) {
    return { url, markdown: '', title: '', success: false, error: err.message || String(err) };
  }
}

/**
 * Tier 3: Firecrawl API.
 */
async function firecrawlFetch(env: Env, url: string): Promise<ScrapeResult> {
  if (!env.FIRECRAWL_API_KEY) {
    return { url, markdown: '', title: '', success: false, error: 'No Firecrawl API key' };
  }

  try {
    const response = await fetch('https://api.firecrawl.dev/v1/scrape', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${env.FIRECRAWL_API_KEY}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ url, formats: ['markdown'] }),
    });

    if (!response.ok) {
      return { url, markdown: '', title: '', success: false, error: `Firecrawl HTTP ${response.status}` };
    }

    const data = await response.json() as any;
    const markdown = data.data?.markdown || '';
    const title = data.data?.metadata?.title || '';

    return { url, markdown, title, success: markdown.length > 50, error: '' };
  } catch (err: any) {
    return { url, markdown: '', title: '', success: false, error: err.message || String(err) };
  }
}

/**
 * Search for URLs and scrape them.
 */
async function searchAndScrape(env: Env, query: string): Promise<ScrapeResult[]> {
  // Use Serper to find relevant URLs
  if (!env.SERPER_API_KEY) return [];

  try {
    const searchResp = await fetch('https://google.serper.dev/search', {
      method: 'POST',
      headers: {
        'X-API-KEY': env.SERPER_API_KEY,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ q: query, num: 5 }),
    });

    if (!searchResp.ok) return [];
    const searchData = await searchResp.json() as any;

    const urls = (searchData.organic || []).slice(0, 5).map((r: any) => r.link).filter(Boolean);
    if (urls.length === 0) return [];

    // Scrape found URLs
    const results = await Promise.allSettled(
      urls.map((url: string) => scrapeSingleUrl(env, url)),
    );

    return results
      .filter((r): r is PromiseFulfilledResult<ScrapeResult> => r.status === 'fulfilled')
      .map(r => r.value)
      .filter(r => r.success);
  } catch {
    return [];
  }
}

// ── HTML Text Extraction ──

function extractTextFromHTML(html: string): { title: string; text: string } {
  // Simple extraction — remove tags, decode entities
  let title = '';
  const titleMatch = html.match(/<title[^>]*>([\s\S]*?)<\/title>/i);
  if (titleMatch) title = titleMatch[1].trim();

  // Remove script, style, nav, footer
  let text = html
    .replace(/<script[\s\S]*?<\/script>/gi, '')
    .replace(/<style[\s\S]*?<\/style>/gi, '')
    .replace(/<nav[\s\S]*?<\/nav>/gi, '')
    .replace(/<footer[\s\S]*?<\/footer>/gi, '')
    .replace(/<header[\s\S]*?<\/header>/gi, '')
    .replace(/<[^>]+>/g, ' ')
    .replace(/&nbsp;/g, ' ')
    .replace(/&amp;/g, '&')
    .replace(/&lt;/g, '<')
    .replace(/&gt;/g, '>')
    .replace(/&quot;/g, '"')
    .replace(/&#39;/g, "'")
    .replace(/\s+/g, ' ')
    .trim();

  return { title, text };
}

function extractTitleFromMarkdown(markdown: string): string {
  const match = markdown.match(/^#\s+(.+)$/m);
  return match ? match[1].trim() : '';
}
