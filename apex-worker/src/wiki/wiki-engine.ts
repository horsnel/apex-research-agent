/**
 * APEX 2.0 — Core LLM Wiki Engine
 *
 * The main engine for the persistent knowledge layer.
 * Check wiki first (0ms, free) → If stale/missing, search → Update Wiki → COMPOUNDS FOREVER
 */

import { Env } from '../types';
import { routeLLMCall } from '../llm-router';
import { embedSingle, queryVectorize, embedAndUpsert } from '../embedder';
import { liveScrape } from '../live-scraper';
import { generateUUID, hashText, Timer } from '../utils';
import {
  WikiPage,
  WikiPageState,
  WikiSource,
  WikiEntity,
  WikiLink,
  WikiIngestRequest,
  WikiIngestResult,
  WikiQueryRequest,
  WikiQueryResult,
  WikiPageRow,
  ContradictionAlert,
} from './types';
import { checkPageFreshness, transitionState, shouldReverify } from './page-lifecycle';
import { scanSource, assignTrustTier } from './security';
import { acquireWriteLock, releaseWriteLock, withWriteLock } from './concurrency';

// ── Wiki Compiler System Prompt ──

const WIKI_COMPILER_SYSTEM_PROMPT = `You are APEX Wiki Compiler. Your job is to compile raw source material into structured wiki pages that persist and compound knowledge.

Output format — produce EXACTLY this structure in markdown:

---
state: active
sources: [list source URLs used]
lastVerified: [ISO date]
---

# [Page Title]

## Summary
[2-4 sentence dense summary of the topic]

## Key Claims
- [Claim 1] [S1, P1]
- [Claim 2] [S2, P2]
- [Claim 3] [S1, P1] [CONTESTED by S3, P3]

## Details
[Detailed explanation with inline citations [Sn, Pn] where n is source number]

## Entities
- [[Person:Name]] — role/context
- [[Org:Name]] — role/context
- [[Tech:Name]] — role/context
- [[Concept:Name]] — definition

## Related Topics
- [[wiki-link-slug]] — relationship description
- [[another-topic]] — extends this concept

Rules:
1. Every factual claim MUST have a citation [Sn, Pn]
2. Use [[wiki-links]] for any concept that deserves its own page
3. Tag entities inline as [[Type:Name]]
4. If sources conflict, present BOTH positions with [CONTESTED] marker
5. Maximum information density — no filler, no repetition
6. Preserve exact numbers, dates, and proper nouns
7. Mark speculative content [SPECULATIVE]
8. Mark unverifiable claims from P3/UNV sources [UNVERIFIED]`;

// ── Slug Generation ──

function slugify(title: string): string {
  return title
    .toLowerCase()
    .trim()
    .replace(/[^\w\s-]/g, '')
    .replace(/[\s_]+/g, '-')
    .replace(/-+/g, '-')
    .replace(/^-+|-+$/g, '')
    .slice(0, 100);
}

// ── D1 Row → WikiPage Conversion ──

async function rowToWikiPage(env: Env, row: WikiPageRow): Promise<WikiPage> {
  let content = '';

  // Load full content from R2
  try {
    const r2Key = `wiki/pages/${row.slug}.md`;
    const r2Object = await env.BUCKET.get(r2Key);
    if (r2Object) {
      content = await r2Object.text();
    }
  } catch {
    // Fallback: use snippet
    content = row.content_snippet || '';
  }

  return {
    id: row.id,
    title: row.title,
    slug: row.slug,
    content,
    state: row.state as WikiPageState,
    sourceHashes: row.source_hashes ? JSON.parse(row.source_hashes) : [],
    sources: row.sources ? JSON.parse(row.sources) : [],
    entities: row.entities ? JSON.parse(row.entities) : [],
    links: row.links ? JSON.parse(row.links) : [],
    metadata: row.metadata ? JSON.parse(row.metadata) : {},
    createdAt: row.created_at,
    updatedAt: row.updated_at,
    lastVerifiedAt: row.last_verified_at,
    verificationCount: row.verification_count,
    accessCount: row.access_count,
    version: row.version,
  };
}

// ── Core: Ingest Sources into Wiki ──

export async function ingestSources(
  env: Env,
  request: WikiIngestRequest,
): Promise<WikiIngestResult> {
  const timer = new Timer();
  const result: WikiIngestResult = {
    pagesCreated: 0,
    pagesUpdated: 0,
    pagesUnchanged: 0,
    errors: [],
    totalCostMs: 0,
  };

  for (const url of request.urls) {
    try {
      // Step 1: Fetch source content
      const scrapeResults = await liveScrape(env, { urls: [url] });
      const scrapeResult = scrapeResults.find(r => r.url === url);

      if (!scrapeResult || !scrapeResult.success || scrapeResult.markdown.length < 50) {
        result.errors.push(`Failed to fetch content from ${url}: ${scrapeResult?.error || 'insufficient content'}`);
        continue;
      }

      const sourceContent = scrapeResult.markdown;

      // Step 2: Security scan
      const securityScan = await scanSource(env, sourceContent, url);
      if (!securityScan.isSafe) {
        result.errors.push(`Security scan failed for ${url}: ${securityScan.threats.join('; ')}`);
        continue;
      }

      // Step 3: Compute content hash
      const contentHash = await hashText(sourceContent);

      // Step 4: Check if source already ingested
      const existingSource = await env.DB.prepare(
        'SELECT id, content_hash, page_ids FROM wiki_sources WHERE url = ? AND content_hash = ?'
      ).bind(url, contentHash).first();

      if (existingSource && !request.forceReingest) {
        result.pagesUnchanged++;
        continue;
      }

      // Step 5: Assign trust tier
      const trustTier = await assignTrustTier(env, url, sourceContent);

      // Step 6: Record the source
      const sourceId = generateUUID();
      const now = new Date().toISOString();
      await env.DB.prepare(`
        INSERT OR REPLACE INTO wiki_sources (id, url, content_hash, tier, title, trust_tier, first_ingested_at, last_checked_at, page_ids)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
      `).bind(
        sourceId,
        url,
        contentHash,
        trustTier === 'verified' ? 'P1' : trustTier === 'internal' ? 'P2' : trustTier === 'external' ? 'P3' : 'UNV',
        scrapeResult.title || null,
        trustTier,
        now,
        now,
        JSON.stringify([]),
      ).run();

      // Step 7: Compile into wiki pages using LLM
      const wikiSource: WikiSource = {
        url,
        tier: trustTier === 'verified' ? 'P1' : trustTier === 'internal' ? 'P2' : trustTier === 'external' ? 'P3' : 'UNV',
        title: scrapeResult.title || url,
        contentHash,
        ingestedAt: now,
        lastCheckedAt: now,
      };

      const compilationResult = await compileSourceToWiki(env, sourceContent, wikiSource, request.category);

      if (compilationResult.pages.length === 0) {
        result.errors.push(`LLM compilation produced no pages for ${url}`);
        continue;
      }

      // Step 8: Upsert each compiled page
      for (const compiledPage of compilationResult.pages) {
        const existingPage = await env.DB.prepare(
          'SELECT id, version, state FROM wiki_pages WHERE slug = ?'
        ).bind(compiledPage.slug).first();

        if (existingPage) {
          // Update existing page
          const lockResult = await acquireWriteLock(env, existingPage.id as string, 'wiki-engine', 60);
          if (lockResult) {
            try {
              const newVersion = (existingPage.version as number) + 1;
              const snippet = compiledPage.content.slice(0, 500);

              await env.DB.prepare(`
                UPDATE wiki_pages
                SET title = ?, content_snippet = ?, source_hashes = ?, sources = ?,
                    entities = ?, links = ?, metadata = ?, updated_at = ?,
                    last_verified_at = ?, verification_count = verification_count + 1,
                    version = ?, state = 'active'
                WHERE id = ?
              `).bind(
                compiledPage.title,
                snippet,
                JSON.stringify(compiledPage.sourceHashes),
                JSON.stringify(compiledPage.sources),
                JSON.stringify(compiledPage.entities),
                JSON.stringify(compiledPage.links),
                JSON.stringify(compiledPage.metadata || {}),
                now,
                now,
                newVersion,
                existingPage.id,
              ).run();

              // Store full content in R2
              await env.BUCKET.put(`wiki/pages/${compiledPage.slug}.md`, compiledPage.content);

              // Update embedding in Vectorize
              await embedAndUpsert(env, `wiki-${compiledPage.slug}`, compiledPage.content, {
                type: 'wiki',
                slug: compiledPage.slug,
                state: 'active',
                category: request.category || '',
              });

              // Update source page_ids
              const sourceRow = await env.DB.prepare(
                'SELECT page_ids FROM wiki_sources WHERE id = ?'
              ).bind(sourceId).first();
              const pageIds: string[] = sourceRow?.page_ids ? JSON.parse(sourceRow.page_ids as string) : [];
              if (!pageIds.includes(existingPage.id as string)) {
                pageIds.push(existingPage.id as string);
                await env.DB.prepare('UPDATE wiki_sources SET page_ids = ? WHERE id = ?')
                  .bind(JSON.stringify(pageIds), sourceId).run();
              }

              result.pagesUpdated++;
            } finally {
              await releaseWriteLock(env, lockResult.lockId);
            }
          } else {
            result.errors.push(`Could not acquire write lock for page ${compiledPage.slug}`);
          }
        } else {
          // Create new page
          const pageId = generateUUID();
          const snippet = compiledPage.content.slice(0, 500);

          await env.DB.prepare(`
            INSERT INTO wiki_pages (id, slug, title, content_snippet, state, category, source_hashes, sources, entities, links, metadata, created_at, updated_at, last_verified_at, verification_count, access_count, version)
            VALUES (?, ?, ?, ?, 'active', ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, 0, 1)
          `).bind(
            pageId,
            compiledPage.slug,
            compiledPage.title,
            snippet,
            request.category || null,
            JSON.stringify(compiledPage.sourceHashes),
            JSON.stringify(compiledPage.sources),
            JSON.stringify(compiledPage.entities),
            JSON.stringify(compiledPage.links),
            JSON.stringify(compiledPage.metadata || {}),
            now,
            now,
            now,
          ).run();

          // Store full content in R2
          await env.BUCKET.put(`wiki/pages/${compiledPage.slug}.md`, compiledPage.content);

          // Embed and upsert to Vectorize
          await embedAndUpsert(env, `wiki-${compiledPage.slug}`, compiledPage.content, {
            type: 'wiki',
            slug: compiledPage.slug,
            state: 'active',
            category: request.category || '',
          });

          // Update source page_ids
          const sourceRow = await env.DB.prepare(
            'SELECT page_ids FROM wiki_sources WHERE id = ?'
          ).bind(sourceId).first();
          const pageIds: string[] = sourceRow?.page_ids ? JSON.parse(sourceRow.page_ids as string) : [];
          pageIds.push(pageId);
          await env.DB.prepare('UPDATE wiki_sources SET page_ids = ? WHERE id = ?')
            .bind(JSON.stringify(pageIds), sourceId).run();

          result.pagesCreated++;
        }

        // Extract and store entities from compiled page
        for (const entity of compiledPage.entities) {
          await upsertWikiEntity(env, entity);
        }

        // Create cross-references from wiki links
        for (const link of compiledPage.links) {
          await ensureWikiLinkExists(env, compiledPage.slug, link);
        }
      }
    } catch (err: any) {
      result.errors.push(`Error processing ${url}: ${err.message || String(err)}`);
    }
  }

  result.totalCostMs = timer.elapsed();
  return result;
}

// ── Compile Source to Wiki Pages using LLM ──

async function compileSourceToWiki(
  env: Env,
  sourceContent: string,
  source: WikiSource,
  category?: string,
): Promise<{ pages: CompiledWikiPage[] }> {
  const truncatedContent = sourceContent.slice(0, 8000);

  const messages = [
    { role: 'system', content: WIKI_COMPILER_SYSTEM_PROMPT },
    {
      role: 'user',
      content: `Compile the following source into wiki page(s).

Source URL: ${source.url}
Source Tier: ${source.tier}
Category: ${category || 'general'}

Source Content:
${truncatedContent}

Produce structured wiki markdown with title, summary, key claims, details, entities, and related topics.`,
    },
  ];

  const llmResult = await routeLLMCall(env, messages, 4096, 0.0, undefined, false, false);

  if (llmResult.content.startsWith('[ALL_LLM_FAILED]')) {
    return { pages: [] };
  }

  const pages = parseCompiledPages(llmResult.content, source);

  return { pages };
}

// ── Parse LLM Output into Compiled Pages ──

interface CompiledWikiPage {
  title: string;
  slug: string;
  content: string;
  sourceHashes: string[];
  sources: WikiSource[];
  entities: WikiEntity[];
  links: WikiLink[];
  metadata: Record<string, unknown>;
}

function parseCompiledPages(
  llmOutput: string,
  source: WikiSource,
): CompiledWikiPage[] {
  const pages: CompiledWikiPage[] = [];

  // Remove YAML frontmatter if present
  let content = llmOutput.replace(/^---[\s\S]*?---\n*/, '');

  // Extract title from first H1
  const titleMatch = content.match(/^#\s+(.+)$/m);
  if (!titleMatch) {
    // If no H1, use the source title
    const title = source.title || slugify(source.url);
    pages.push({
      title,
      slug: slugify(title),
      content,
      sourceHashes: [source.contentHash],
      sources: [source],
      entities: extractEntitiesFromContent(content),
      links: extractLinksFromContent(content),
      metadata: { compiledFrom: source.url },
    });
    return pages;
  }

  const title = titleMatch[1].trim();

  // Extract entities from content
  const entities = extractEntitiesFromContent(content);

  // Extract wiki links
  const links = extractLinksFromContent(content);

  pages.push({
    title,
    slug: slugify(title),
    content,
    sourceHashes: [source.contentHash],
    sources: [source],
    entities,
    links,
    metadata: { compiledFrom: source.url },
  });

  return pages;
}

// ── Extract Entities from Wiki Content ──

function extractEntitiesFromContent(content: string): WikiEntity[] {
  const entities: WikiEntity[] = [];
  const now = new Date().toISOString();
  const seen = new Set<string>();

  // Match [[Type:Name]] patterns
  const entityPattern = /\[\[(person|org|tech|concept|location|event):([^\]]+)\]\]/gi;
  let match: RegExpExecArray | null;

  while ((match = entityPattern.exec(content)) !== null) {
    const type = match[1].toLowerCase() as WikiEntity['type'];
    const name = match[2].trim();
    const key = `${type}:${name}`;

    if (!seen.has(key)) {
      seen.add(key);
      entities.push({
        name,
        type,
        mentions: 1,
        firstSeen: now,
        lastSeen: now,
      });
    } else {
      const existing = entities.find(e => `${e.type}:${e.name}` === key);
      if (existing) existing.mentions++;
    }
  }

  return entities;
}

// ── Extract Wiki Links from Content ──

function extractLinksFromContent(content: string): WikiLink[] {
  const links: WikiLink[] = [];
  const seen = new Set<string>();

  // Match [[wiki-link-slug]] patterns (not [[Type:Name]] entity patterns)
  const linkPattern = /\[\[([a-z0-9][\w-]*[a-z0-9])\]\]/gi;
  let match: RegExpExecArray | null;

  while ((match = linkPattern.exec(content)) !== null) {
    const slug = match[1];
    if (!seen.has(slug)) {
      seen.add(slug);
      links.push({
        targetSlug: slug,
        relationType: 'related',
        context: '',
      });
    }
  }

  return links;
}

// ── Upsert Wiki Entity ──

async function upsertWikiEntity(env: Env, entity: WikiEntity): Promise<void> {
  try {
    const existing = await env.DB.prepare(
      'SELECT id, mention_count FROM wiki_entities WHERE name = ? AND type = ?'
    ).bind(entity.name, entity.type).first();

    if (existing) {
      await env.DB.prepare(`
        UPDATE wiki_entities
        SET mention_count = mention_count + ?, last_seen = ?, updated_at = ?
        WHERE id = ?
      `).bind(entity.mentions, entity.lastSeen, new Date().toISOString(), existing.id).run();
    } else {
      const id = generateUUID();
      await env.DB.prepare(`
        INSERT INTO wiki_entities (id, name, type, description, mention_count, first_seen, last_seen, properties, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
      `).bind(
        id,
        entity.name,
        entity.type,
        null,
        entity.mentions,
        entity.firstSeen,
        entity.lastSeen,
        JSON.stringify({}),
        new Date().toISOString(),
        new Date().toISOString(),
      ).run();
    }
  } catch {
    // Non-critical — entity upsert failures shouldn't block ingestion
  }
}

// ── Ensure Wiki Link Target Exists ──

async function ensureWikiLinkExists(env: Env, sourceSlug: string, link: WikiLink): Promise<void> {
  try {
    const targetExists = await env.DB.prepare(
      'SELECT id FROM wiki_pages WHERE slug = ?'
    ).bind(link.targetSlug).first();

    if (!targetExists) {
      // Create a draft placeholder page for the linked topic
      const id = generateUUID();
      const now = new Date().toISOString();
      await env.DB.prepare(`
        INSERT INTO wiki_pages (id, slug, title, content_snippet, state, category, source_hashes, sources, entities, links, metadata, created_at, updated_at, verification_count, access_count, version)
        VALUES (?, ?, ?, ?, 'draft', ?, ?, ?, ?, ?, ?, ?, ?, 0, 0, 1)
      `).bind(
        id,
        link.targetSlug,
        link.targetSlug.replace(/-/g, ' ').replace(/\b\w/g, c => c.toUpperCase()),
        `Placeholder page — content to be compiled. Referenced from [[${sourceSlug}]].`,
        null,
        JSON.stringify([]),
        JSON.stringify([]),
        JSON.stringify([]),
        JSON.stringify([{ targetSlug: sourceSlug, relationType: 'related', context: 'backlink' }]),
        JSON.stringify({ isPlaceholder: true }),
        now,
        now,
      ).run();
    }
  } catch {
    // Non-critical
  }
}

// ── Core: Query the Wiki ──

export async function queryWiki(
  env: Env,
  request: WikiQueryRequest,
): Promise<WikiQueryResult> {
  const timer = new Timer();
  const maxPages = request.maxPages || 5;
  const freshnessThreshold = request.freshnessThreshold || 168; // 1 week default

  // Step 1: Embed the query
  const queryVector = await embedSingle(env, request.query);

  // Step 2: Search Vectorize for wiki pages
  const vectorResults = await queryVectorize(env, queryVector, maxPages * 2, { type: 'wiki' });

  // Step 3: Also do FTS5 keyword search
  const ftsResults = await env.DB.prepare(`
    SELECT p.id, p.slug, p.title, p.state, p.last_verified_at, p.updated_at
    FROM wiki_pages_fts f
    JOIN wiki_pages p ON p.rowid = f.rowid
    WHERE wiki_pages_fts MATCH ?
    ORDER BY rank
    LIMIT ?
  `).bind(request.query, maxPages).all();

  // Step 4: Combine results
  const candidateSlugs = new Set<string>();

  for (const match of vectorResults) {
    if (match.metadata?.slug) {
      candidateSlugs.add(match.metadata.slug);
    }
  }

  for (const row of ftsResults.results) {
    if (row.slug) {
      candidateSlugs.add(row.slug as string);
    }
  }

  // Step 5: Load wiki pages
  const pages: WikiPage[] = [];
  const stalePages: string[] = [];
  const contradictionAlerts: ContradictionAlert[] = [];

  for (const slug of Array.from(candidateSlugs).slice(0, maxPages)) {
    const page = await getPage(env, slug);
    if (page && page.state !== 'archived' && page.state !== 'draft') {
      pages.push(page);

      // Track access
      await env.DB.prepare('UPDATE wiki_pages SET access_count = access_count + 1 WHERE slug = ?')
        .bind(slug).run();

      // Check freshness
      if (shouldReverify(page, freshnessThreshold)) {
        stalePages.push(slug);
        // Trigger background re-verification
        await checkPageFreshness(env, page);
      }

      // Check for contradicted state
      if (page.state === 'contradicted') {
        contradictionAlerts.push({
          pageA: slug,
          pageB: '',
          conflictingClaims: ['This page has been marked as contradicted'],
          detectedAt: page.updatedAt,
          severity: 'medium',
        });
      }
    }
  }

  // Step 6: If no wiki pages found, return empty result
  if (pages.length === 0) {
    return {
      answer: '',
      pagesUsed: [],
      stalePages: [],
      contradictionAlerts: [],
      costSavedMs: 0,
    };
  }

  // Step 7: Synthesize answer from wiki pages
  const wikiContext = pages.map((p, i) => `[Wiki Page ${i + 1}: ${p.title} (state: ${p.state})]\n${p.content.slice(0, 3000)}`).join('\n\n');

  const synthesisPrompt = `You are APEX Wiki Answer. Answer the user's query using ONLY the wiki pages provided.

Wiki Pages:
${wikiContext}

Rules:
1. Use inline citations [Wiki N] where N is the page number
2. If a page is marked stale or contradicted, note it
3. If pages disagree, present both positions
4. Be concise and information-dense
5. If the wiki doesn't fully answer the query, say so explicitly`;

  const llmResult = await routeLLMCall(
    env,
    [
      { role: 'system', content: synthesisPrompt },
      { role: 'user', content: request.query },
    ],
    1024, 0.0, undefined, false,
  );

  // Step 8: Check for contradictions if requested
  if (request.includeContradictions) {
    try {
      const { detectContradictions } = await import('./dialogic-wiki');
      const topic = request.query;
      const contradictions = await detectContradictions(env, topic);
      for (const record of contradictions) {
        for (const position of record.positions) {
          contradictionAlerts.push({
            pageA: position.pageId,
            pageB: '',
            conflictingClaims: [position.claim],
            detectedAt: record.detectedAt,
            severity: record.severity,
          });
        }
      }
    } catch {
      // Non-critical — contradiction detection failure shouldn't block query
    }
  }

  const costSavedMs = pages.length > 0 ? timer.elapsed() * 3 : 0; // Estimated 3x cost for full RAG

  return {
    answer: llmResult.content,
    pagesUsed: pages.map(p => p.slug),
    stalePages,
    contradictionAlerts,
    costSavedMs,
  };
}

// ── Get Page by Slug ──

export async function getPage(env: Env, slug: string): Promise<WikiPage | null> {
  const row = await env.DB.prepare(
    'SELECT * FROM wiki_pages WHERE slug = ?'
  ).bind(slug).first() as WikiPageRow | null;

  if (!row) return null;

  return rowToWikiPage(env, row);
}

// ── List Pages ──

export async function listPages(
  env: Env,
  options: {
    state?: WikiPageState;
    category?: string;
    limit?: number;
    offset?: number;
    sortBy?: 'updated_at' | 'title' | 'access_count' | 'verification_count';
    sortOrder?: 'ASC' | 'DESC';
  } = {},
): Promise<{ pages: WikiPage[]; total: number }> {
  const {
    state,
    category,
    limit = 50,
    offset = 0,
    sortBy = 'updated_at',
    sortOrder = 'DESC',
  } = options;

  const conditions: string[] = [];
  const params: unknown[] = [];

  if (state) {
    conditions.push('state = ?');
    params.push(state);
  }

  if (category) {
    conditions.push('category = ?');
    params.push(category);
  }

  const whereClause = conditions.length > 0 ? `WHERE ${conditions.join(' AND ')}` : '';

  // Validate sort column to prevent SQL injection
  const validSortColumns = ['updated_at', 'title', 'access_count', 'verification_count', 'created_at'];
  const safeSortBy = validSortColumns.includes(sortBy) ? sortBy : 'updated_at';
  const safeSortOrder = sortOrder === 'ASC' ? 'ASC' : 'DESC';

  // Get total count
  const countResult = await env.DB.prepare(
    `SELECT COUNT(*) as total FROM wiki_pages ${whereClause}`
  ).bind(...params).first();
  const total = (countResult?.total as number) || 0;

  // Get pages
  const pageParams = [...params, limit, offset];
  const rows = await env.DB.prepare(
    `SELECT * FROM wiki_pages ${whereClause} ORDER BY ${safeSortBy} ${safeSortOrder} LIMIT ? OFFSET ?`
  ).bind(...pageParams).all();

  const pages: WikiPage[] = [];
  for (const row of rows.results as unknown as WikiPageRow[]) {
    pages.push(await rowToWikiPage(env, row));
  }

  return { pages, total };
}

// ── Delete (Archive) Page ──

export async function deletePage(env: Env, slug: string): Promise<boolean> {
  const page = await getPage(env, slug);
  if (!page) return false;

  await transitionState(env, page.id, 'archived', 'Manual deletion');

  return true;
}

// ── Recompile Page ──

export async function recompilePage(env: Env, slug: string): Promise<WikiPage | null> {
  const page = await getPage(env, slug);
  if (!page) return null;

  if (page.sources.length === 0) {
    return page; // Nothing to recompile from
  }

  // Re-fetch all sources
  const urls = page.sources.map(s => s.url);
  const result = await ingestSources(env, {
    urls,
    category: undefined,
    forceReingest: true,
  });

  // Return updated page
  return getPage(env, slug);
}

// ── Get Wiki Stats ──

export async function getWikiStats(env: Env): Promise<{
  totalPages: number;
  activePages: number;
  stalePages: number;
  contradictedPages: number;
  draftPages: number;
  archivedPages: number;
  totalSources: number;
  entityCount: number;
  linkCount: number;
  contradictionCount: number;
  costSavedMs: number;
}> {
  const [
    totalResult,
    activeResult,
    staleResult,
    contradictedResult,
    draftResult,
    archivedResult,
    sourcesResult,
    entitiesResult,
    linksResult,
    contradictionsResult,
  ] = await Promise.all([
    env.DB.prepare('SELECT COUNT(*) as count FROM wiki_pages').first(),
    env.DB.prepare("SELECT COUNT(*) as count FROM wiki_pages WHERE state = 'active'").first(),
    env.DB.prepare("SELECT COUNT(*) as count FROM wiki_pages WHERE state = 'stale'").first(),
    env.DB.prepare("SELECT COUNT(*) as count FROM wiki_pages WHERE state = 'contradicted'").first(),
    env.DB.prepare("SELECT COUNT(*) as count FROM wiki_pages WHERE state = 'draft'").first(),
    env.DB.prepare("SELECT COUNT(*) as count FROM wiki_pages WHERE state = 'archived'").first(),
    env.DB.prepare('SELECT COUNT(*) as count FROM wiki_sources').first(),
    env.DB.prepare('SELECT COUNT(*) as count FROM wiki_entities').first(),
    env.DB.prepare('SELECT COUNT(*) as count FROM wiki_relations').first(),
    env.DB.prepare('SELECT COUNT(*) as count FROM wiki_contradictions').first(),
  ]);

  // Estimate cost savings: each active page saves ~3x the compilation time on future queries
  const activeCount = (activeResult?.count as number) || 0;

  return {
    totalPages: (totalResult?.count as number) || 0,
    activePages: activeCount,
    stalePages: (staleResult?.count as number) || 0,
    contradictedPages: (contradictedResult?.count as number) || 0,
    draftPages: (draftResult?.count as number) || 0,
    archivedPages: (archivedResult?.count as number) || 0,
    totalSources: (sourcesResult?.count as number) || 0,
    entityCount: (entitiesResult?.count as number) || 0,
    linkCount: (linksResult?.count as number) || 0,
    contradictionCount: (contradictionsResult?.count as number) || 0,
    costSavedMs: activeCount * 500, // Rough estimate: 500ms saved per active page
  };
}
