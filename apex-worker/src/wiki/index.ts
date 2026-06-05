/**
 * APEX 2.0 — Wiki Module Index
 *
 * Exports all wiki functions, types, and the route registration function
 * that wires wiki API endpoints into the main worker router.
 */

// ── Type Exports ──

export type {
  WikiPage,
  WikiPageState,
  WikiSource,
  WikiEntity,
  WikiEntityType,
  WikiLink,
  WikiLinkRelationType,
  WikiIngestRequest,
  WikiIngestResult,
  WikiQueryRequest,
  WikiQueryResult,
  HotCacheEntry,
  PageLifecycleEvent,
  WikiSchema,
  ContradictionAlert,
  KnowledgeEntityType,
  KnowledgeRelationType,
  KnowledgeEntity,
  KnowledgeRelation,
  KnowledgeSubgraph,
  ProvenanceClaim,
  ProvenanceAuditReport,
  ContradictionPosition,
  ContradictionRecord,
  DialecticalSummary,
  TrustTier,
  SecurityScanResult,
  LockAcquisition,
  ConflictResolutionStrategy,
  WikiPageRow,
  WikiSourceRow,
  WikiSessionRow,
  WikiEntityRow,
  WikiRelationRow,
  WikiProvenanceClaimRow,
  WikiContradictionRow,
  WikiSecurityLogRow,
  WikiLockRow,
  WikiLifecycleEventRow,
  WikiSchemaRow,
} from './types';

// ── Function Exports ──

// Wiki Engine
export {
  ingestSources,
  queryWiki,
  getPage,
  listPages,
  deletePage,
  recompilePage,
  getWikiStats,
} from './wiki-engine';

// Page Lifecycle
export {
  checkPageFreshness,
  transitionState,
  detectContradictions as detectPageContradictions,
  getLifecycleHistory,
  autoTransition,
  shouldReverify,
  WIKI_LIFECYCLE_CONFIG,
} from './page-lifecycle';

// Hot Cache
export {
  createSession,
  getSession,
  updateHotCache,
  getResumptionContext,
  endSession,
  pruneStaleSessions,
} from './hot-cache';

// SciMem (Knowledge Graph)
export {
  getOrCreateEntity,
  getEntityByName,
  linkEntities,
  queryKnowledgeGraph,
  findConnections,
  mergeResearchIntoMemory,
  getMemoryEvolutionSuggestions,
  exportGraph,
  ENTITY_TYPES,
  RELATION_TYPES,
} from './sci-mem';

// Provenance
export {
  extractAndTrack,
  detectConflicts,
  getClaimLineage,
  auditProvenance,
  resolveConflict as resolveProvenanceConflict,
} from './provenance';

// Dialogic Wiki
export {
  detectContradictions,
  preserveContradiction,
  getContradictionMap,
  analyzeContradiction,
  trackContradictionEvolution,
  generateDialecticalSummary,
} from './dialogic-wiki';

// Security
export {
  scanSource,
  assignTrustTier,
  adversarialReview,
  validateCrossOriginClaims,
  getSecurityAuditLog,
  quarantinePage,
} from './security';

// Concurrency
export {
  acquireWriteLock,
  releaseWriteLock,
  extendWriteLock,
  withWriteLock,
  detectConcurrentModifications,
  resolveConflict,
  getActiveLocks,
  pruneExpiredLocks,
  CONCURRENCY_CONFIG,
} from './concurrency';

// ── Route Registration ──

import { Env } from '../types';
import { jsonResponse, errorResponse } from '../utils';

import { ingestSources, queryWiki, getPage, listPages, deletePage, recompilePage, getWikiStats } from './wiki-engine';
import { autoTransition, transitionState, getLifecycleHistory } from './page-lifecycle';
import { createSession, getSession, updateHotCache, getResumptionContext, endSession } from './hot-cache';
import { getOrCreateEntity, getEntityByName, queryKnowledgeGraph, mergeResearchIntoMemory, exportGraph } from './sci-mem';
import { extractAndTrack, auditProvenance } from './provenance';
import { detectContradictions, getContradictionMap, analyzeContradiction, generateDialecticalSummary } from './dialogic-wiki';
import { scanSource, adversarialReview } from './security';
import { acquireWriteLock, releaseWriteLock, getActiveLocks } from './concurrency';

/**
 * Register all wiki API routes on the main worker's router.
 *
 * Usage in src/index.ts:
 *   import { registerWikiRoutes } from './wiki';
 *   // Inside fetch handler, before default case:
 *   const wikiResponse = await registerWikiRoutes(request, env);
 *   if (wikiResponse) return wikiResponse;
 */
export async function registerWikiRoutes(
  request: Request,
  env: Env,
): Promise<Response | null> {
  const url = new URL(request.url);
  const path = url.pathname;
  const method = request.method;

  // ── Wiki Ingest ──
  if (method === 'POST' && path === '/wiki/ingest') {
    try {
      const body = await request.json() as { urls: string[]; category?: string; forceReingest?: boolean };
      const result = await ingestSources(env, {
        urls: body.urls || [],
        category: body.category,
        forceReingest: body.forceReingest,
      });
      return jsonResponse(result, 200, request);
    } catch (err: any) {
      return errorResponse(`Ingest failed: ${err.message}`, 500, request);
    }
  }

  // ── Wiki Query ──
  if (method === 'POST' && path === '/wiki/query') {
    try {
      const body = await request.json() as {
        query: string;
        includeContradictions?: boolean;
        maxPages?: number;
        freshnessThreshold?: number;
      };
      const result = await queryWiki(env, {
        query: body.query,
        includeContradictions: body.includeContradictions,
        maxPages: body.maxPages,
        freshnessThreshold: body.freshnessThreshold,
      });
      return jsonResponse(result, 200, request);
    } catch (err: any) {
      return errorResponse(`Wiki query failed: ${err.message}`, 500, request);
    }
  }

  // ── List Wiki Pages ──
  if (method === 'GET' && path === '/wiki/pages') {
    try {
      const state = url.searchParams.get('state') as any;
      const category = url.searchParams.get('category') || undefined;
      const limit = parseInt(url.searchParams.get('limit') || '50', 10);
      const offset = parseInt(url.searchParams.get('offset') || '0', 10);
      const sortBy = url.searchParams.get('sortBy') as any || 'updated_at';
      const sortOrder = url.searchParams.get('sortOrder') as any || 'DESC';

      const result = await listPages(env, {
        state,
        category,
        limit,
        offset,
        sortBy,
        sortOrder,
      });
      return jsonResponse(result, 200, request);
    } catch (err: any) {
      return errorResponse(`List pages failed: ${err.message}`, 500, request);
    }
  }

  // ── Get Wiki Page ──
  if (method === 'GET' && path.match(/^\/wiki\/pages\/[^/]+$/)) {
    try {
      const slug = path.split('/').pop()!;
      const page = await getPage(env, slug);
      if (!page) {
        return errorResponse('Page not found', 404, request);
      }
      return jsonResponse(page, 200, request);
    } catch (err: any) {
      return errorResponse(`Get page failed: ${err.message}`, 500, request);
    }
  }

  // ── Delete (Archive) Wiki Page ──
  if (method === 'DELETE' && path.match(/^\/wiki\/pages\/[^/]+$/)) {
    try {
      const slug = path.split('/').pop()!;
      const deleted = await deletePage(env, slug);
      if (!deleted) {
        return errorResponse('Page not found', 404, request);
      }
      return jsonResponse({ status: 'archived', slug }, 200, request);
    } catch (err: any) {
      return errorResponse(`Delete page failed: ${err.message}`, 500, request);
    }
  }

  // ── Recompile Wiki Page ──
  if (method === 'POST' && path.match(/^\/wiki\/pages\/[^/]+\/recompile$/)) {
    try {
      const parts = path.split('/');
      const slug = parts[3];
      const page = await recompilePage(env, slug);
      if (!page) {
        return errorResponse('Page not found or no sources to recompile', 404, request);
      }
      return jsonResponse(page, 200, request);
    } catch (err: any) {
      return errorResponse(`Recompile failed: ${err.message}`, 500, request);
    }
  }

  // ── Wiki Stats ──
  if (method === 'GET' && path === '/wiki/stats') {
    try {
      const stats = await getWikiStats(env);
      return jsonResponse(stats, 200, request);
    } catch (err: any) {
      return errorResponse(`Stats failed: ${err.message}`, 500, request);
    }
  }

  // ── Lifecycle: Check All Pages ──
  if (method === 'POST' && path === '/wiki/lifecycle/check') {
    try {
      const result = await autoTransition(env);
      return jsonResponse(result, 200, request);
    } catch (err: any) {
      return errorResponse(`Lifecycle check failed: ${err.message}`, 500, request);
    }
  }

  // ── Lifecycle: Manual State Transition ──
  if (method === 'POST' && path === '/wiki/lifecycle/transition') {
    try {
      const body = await request.json() as {
        pageId: string;
        newState: 'draft' | 'active' | 'stale' | 'contradicted' | 'archived';
        reason: string;
      };
      const transitioned = await transitionState(env, body.pageId, body.newState, body.reason || 'Manual transition');
      return jsonResponse({ transitioned }, 200, request);
    } catch (err: any) {
      return errorResponse(`State transition failed: ${err.message}`, 500, request);
    }
  }

  // ── Hot Cache: Get Session ──
  if (method === 'GET' && path.match(/^\/wiki\/hot-cache\/[^/]+$/)) {
    try {
      const sessionId = path.split('/').pop()!;
      const session = await getSession(env, sessionId);
      if (!session) {
        return errorResponse('Session not found', 404, request);
      }

      // Check if they want resumption context
      const full = url.searchParams.get('full');
      if (full === 'true') {
        const resumptionContext = await getResumptionContext(env, sessionId);
        return jsonResponse({ session, resumptionContext }, 200, request);
      }

      return jsonResponse(session, 200, request);
    } catch (err: any) {
      return errorResponse(`Get session failed: ${err.message}`, 500, request);
    }
  }

  // ── Hot Cache: Update Session ──
  if (method === 'POST' && path.match(/^\/wiki\/hot-cache\/[^/]+\/update$/)) {
    try {
      const parts = path.split('/');
      const sessionId = parts[3];
      const body = await request.json() as {
        query: string;
        context: string;
        sources?: string[];
      };
      const updated = await updateHotCache(
        env,
        sessionId,
        body.query,
        body.context || '',
        body.sources || [],
      );
      if (!updated) {
        return errorResponse('Session not found', 404, request);
      }
      return jsonResponse(updated, 200, request);
    } catch (err: any) {
      return errorResponse(`Update session failed: ${err.message}`, 500, request);
    }
  }

  // ── Knowledge Graph: Query ──
  if (method === 'POST' && path === '/wiki/knowledge-graph/query') {
    try {
      const body = await request.json() as {
        entityId?: string;
        entityName?: string;
        entityType?: string;
        depth?: number;
      };

      if (body.entityName) {
        // Query by name
        const entity = await getEntityByName(env, body.entityName, body.entityType as any);
        if (!entity) {
          return errorResponse('Entity not found', 404, request);
        }
        const subgraph = await queryKnowledgeGraph(env, entity.id, body.depth || 2);
        return jsonResponse(subgraph, 200, request);
      }

      if (body.entityId) {
        const subgraph = await queryKnowledgeGraph(env, body.entityId, body.depth || 2);
        return jsonResponse(subgraph, 200, request);
      }

      return errorResponse('Provide either entityId or entityName', 400, request);
    } catch (err: any) {
      return errorResponse(`Knowledge graph query failed: ${err.message}`, 500, request);
    }
  }

  // ── Knowledge Graph: Merge Research ──
  if (method === 'POST' && path === '/wiki/knowledge-graph/merge') {
    try {
      const body = await request.json() as {
        sources: Array<{ url: string; title: string; snippet: string; tier: string }>;
        claims: Array<{ statement: string; confidence: number }>;
      };
      const result = await mergeResearchIntoMemory(env, body.sources, body.claims);
      return jsonResponse(result, 200, request);
    } catch (err: any) {
      return errorResponse(`Merge failed: ${err.message}`, 500, request);
    }
  }

  // ── Knowledge Graph: Get Entity ──
  if (method === 'GET' && path.match(/^\/wiki\/knowledge-graph\/entity\/[^/]+$/)) {
    try {
      const name = decodeURIComponent(path.split('/').pop()!);
      const type = url.searchParams.get('type') || undefined;
      const entity = await getEntityByName(env, name, type as any);
      if (!entity) {
        return errorResponse('Entity not found', 404, request);
      }
      return jsonResponse(entity, 200, request);
    } catch (err: any) {
      return errorResponse(`Get entity failed: ${err.message}`, 500, request);
    }
  }

  // ── Provenance: Extract and Track ──
  if (method === 'POST' && path === '/wiki/provenance/extract') {
    try {
      const body = await request.json() as {
        sources: Array<{ url: string; title: string; content: string; tier: string }>;
        pageId: string;
      };
      const result = await extractAndTrack(env, body.sources, body.pageId);
      return jsonResponse(result, 200, request);
    } catch (err: any) {
      return errorResponse(`Provenance extraction failed: ${err.message}`, 500, request);
    }
  }

  // ── Provenance: Audit ──
  if (method === 'POST' && path.match(/^\/wiki\/provenance\/audit\/[^/]+$/)) {
    try {
      const pageId = path.split('/').pop()!;
      const report = await auditProvenance(env, pageId);
      return jsonResponse(report, 200, request);
    } catch (err: any) {
      return errorResponse(`Provenance audit failed: ${err.message}`, 500, request);
    }
  }

  // ── Contradictions: Detect ──
  if (method === 'POST' && path === '/wiki/contradictions/detect') {
    try {
      const body = await request.json() as { topic: string };
      const contradictions = await detectContradictions(env, body.topic);
      return jsonResponse(contradictions, 200, request);
    } catch (err: any) {
      return errorResponse(`Contradiction detection failed: ${err.message}`, 500, request);
    }
  }

  // ── Contradictions: Map ──
  if (method === 'GET' && path.match(/^\/wiki\/contradictions\/map\/[^/]*$/)) {
    try {
      const domain = path.split('/').pop() || undefined;
      const map = await getContradictionMap(env, domain === 'all' ? undefined : domain);
      return jsonResponse(map, 200, request);
    } catch (err: any) {
      return errorResponse(`Contradiction map failed: ${err.message}`, 500, request);
    }
  }

  // ── Contradictions: Analyze ──
  if (method === 'POST' && path.match(/^\/wiki\/contradictions\/analyze\/[^/]+$/)) {
    try {
      const recordId = path.split('/').pop()!;
      const analysis = await analyzeContradiction(env, recordId);
      return jsonResponse(analysis, 200, request);
    } catch (err: any) {
      return errorResponse(`Contradiction analysis failed: ${err.message}`, 500, request);
    }
  }

  // ── Contradictions: Dialectical Summary ──
  if (method === 'POST' && path === '/wiki/contradictions/dialectical') {
    try {
      const body = await request.json() as { topic: string };
      const summary = await generateDialecticalSummary(env, body.topic);
      return jsonResponse(summary, 200, request);
    } catch (err: any) {
      return errorResponse(`Dialectical summary failed: ${err.message}`, 500, request);
    }
  }

  // ── Security: Scan Source ──
  if (method === 'POST' && path === '/wiki/security/scan') {
    try {
      const body = await request.json() as { content: string; sourceUrl: string };
      const result = await scanSource(env, body.content, body.sourceUrl);
      return jsonResponse(result, 200, request);
    } catch (err: any) {
      return errorResponse(`Security scan failed: ${err.message}`, 500, request);
    }
  }

  // ── Security: Adversarial Review ──
  if (method === 'POST' && path.match(/^\/wiki\/security\/review\/[^/]+$/)) {
    try {
      const pageId = path.split('/').pop()!;
      const result = await adversarialReview(env, pageId);
      return jsonResponse(result, 200, request);
    } catch (err: any) {
      return errorResponse(`Adversarial review failed: ${err.message}`, 500, request);
    }
  }

  // ── Locks: Acquire ──
  if (method === 'POST' && path === '/wiki/locks/acquire') {
    try {
      const body = await request.json() as {
        pageId: string;
        holder: string;
        ttlSeconds?: number;
      };
      const lock = await acquireWriteLock(env, body.pageId, body.holder, body.ttlSeconds);
      if (!lock) {
        return errorResponse('Could not acquire lock — page may be locked by another holder', 409, request);
      }
      return jsonResponse(lock, 200, request);
    } catch (err: any) {
      return errorResponse(`Lock acquisition failed: ${err.message}`, 500, request);
    }
  }

  // ── Locks: Release ──
  if (method === 'POST' && path === '/wiki/locks/release') {
    try {
      const body = await request.json() as { lockId: string };
      const released = await releaseWriteLock(env, body.lockId);
      if (!released) {
        return errorResponse('Lock not found or already released', 404, request);
      }
      return jsonResponse({ released: true }, 200, request);
    } catch (err: any) {
      return errorResponse(`Lock release failed: ${err.message}`, 500, request);
    }
  }

  // ── Locks: List Active ──
  if (method === 'GET' && path === '/wiki/locks') {
    try {
      const locks = await getActiveLocks(env);
      return jsonResponse({ locks, count: locks.length }, 200, request);
    } catch (err: any) {
      return errorResponse(`Get locks failed: ${err.message}`, 500, request);
    }
  }

  // ── No wiki route matched ──
  return null;
}
