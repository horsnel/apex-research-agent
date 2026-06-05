/**
 * APEX Research Agent 2.1 — Cloudflare Worker Entry Point
 * Fully serverless: Workers AI + D1 + R2 + Vectorize + LLM Wiki
 *
 * Endpoints:
 *   GET  /health          — Health check
 *   POST /query           — Main research query (checks wiki first)
 *   POST /classify        — Query classification
 *   POST /search          — Vector+keyword search
 *   POST /scrape          — Live web scraping
 *   POST /research        — Deep research with structured report
 *   POST /verify          — Claim verification
 *   GET  /research/status — Research engine features
 *   POST /ingest/url      — Ingest web URL
 *   POST /ingest/embed-pending — Embed missing vectors
 *   GET  /router/status   — LLM fallback chain status
 *   GET  /router/chain    — Full chain config
 *   POST /router/test     — Test all LLM models
 *   /wiki/*               — 24 LLM Wiki endpoints (see wiki/index.ts)
 */

import { Env, QueryRequest, QueryResponse, ResearchRequest, ResearchResponse, ClassifyRequest, SearchRequest, IngestURLRequest, HealthResponse } from './types';
import { corsHeaders, handleOptions, jsonResponse, errorResponse, Timer, generateUUID, hashText, extractDomain, enforceSourceTier } from './utils';
import { classifyQuery, shouldEscalateToLive } from './query-classifier';
import { retrieve } from './retriever';
import { synthesize } from './synthesizer';
import { getRouterStatus, testAllModels, FALLBACK_CHAIN } from './llm-router';
import { liveScrape } from './live-scraper';
import { deepResearch, generateResearchReport, verifyClaimsFromSources, extractClaimsFromSources } from './research-engine';
import { embedAndUpsert, embedPendingChunks } from './embedder';
import { searchRouter } from './search-sources';
import { registerWikiRoutes } from './wiki';

export default {
  async fetch(request: Request, env: Env, ctx: ExecutionContext): Promise<Response> {
    const url = new URL(request.url);
    const path = url.pathname;

    // CORS preflight
    if (request.method === 'OPTIONS') {
      return handleOptions(request);
    }

    // Route to handler
    try {
      // ── LLM Wiki Routes (24 endpoints) ──
      if (path.startsWith('/wiki/')) {
        const wikiResponse = await registerWikiRoutes(request, env);
        if (wikiResponse) return wikiResponse;
      }

      switch (`${request.method} ${path}`) {
        case 'GET /health':
          return handleHealth(env, request);
        case 'POST /query':
          return handleQuery(env, request, ctx);
        case 'POST /classify':
          return handleClassify(env, request);
        case 'POST /search':
          return handleSearch(env, request);
        case 'POST /scrape':
          return handleScrape(env, request);
        case 'POST /research':
          return handleResearch(env, request, ctx);
        case 'POST /verify':
          return handleVerify(env, request);
        case 'GET /research/status':
          return handleResearchStatus(request);
        case 'POST /ingest/url':
          return handleIngestUrl(env, request, ctx);
        case 'POST /ingest/embed-pending':
          return handleEmbedPending(env, request);
        case 'GET /router/status':
          return handleRouterStatus(request);
        case 'GET /router/chain':
          return handleRouterChain(request);
        case 'POST /router/test':
          return handleRouterTest(env, request);
        default:
          // Try method-agnostic routing
          if (path === '/health' && request.method === 'GET') return handleHealth(env, request);
          if (path === '/query' && request.method === 'POST') return handleQuery(env, request, ctx);
          if (path === '/classify' && request.method === 'POST') return handleClassify(env, request);
          if (path === '/search' && request.method === 'POST') return handleSearch(env, request);
          if (path === '/scrape' && request.method === 'POST') return handleScrape(env, request);
          if (path === '/research' && request.method === 'POST') return handleResearch(env, request, ctx);
          if (path === '/verify' && request.method === 'POST') return handleVerify(env, request);
          if (path === '/router/status' && request.method === 'GET') return handleRouterStatus(request);
          if (path === '/router/test' && request.method === 'POST') return handleRouterTest(env, request);
          if (path.startsWith('/ingest/')) return handleIngestUrl(env, request, ctx);
          // Final wiki fallback for any unmatched /wiki/* routes
          if (path.startsWith('/wiki/')) {
            const wikiResponse = await registerWikiRoutes(request, env);
            if (wikiResponse) return wikiResponse;
          }
          return errorResponse('Not found', 404, request);
      }
    } catch (err: any) {
      console.error(`Unhandled error: ${err}`);
      return errorResponse(err.message || 'Internal server error', 500, request);
    }
  },
};

// ═══════════════════════════════════════
// ENDPOINT HANDLERS
// ═══════════════════════════════════════

async function handleHealth(env: Env, request: Request): Promise<Response> {
  let dbStatus = 'not_configured';
  let vectorizeStatus = 'not_configured';
  let r2Status = 'not_configured';

  // Check D1
  try {
    const result = await env.DB.prepare('SELECT COUNT(*) as count FROM documents').first();
    dbStatus = `connected (${result?.count || 0} docs)`;
  } catch (err: any) {
    dbStatus = `error: ${err.message?.slice(0, 80) || 'unknown'}`;
  }

  // Check Vectorize
  try {
    const testQuery = await env.VECTORIZE.query({
      vector: new Array(768).fill(0),
      topK: 1,
    });
    vectorizeStatus = 'connected';
  } catch (err: any) {
    vectorizeStatus = `error: ${err.message?.slice(0, 80) || 'unknown'}`;
  }

  // Check R2
  try {
    // Just list with limit 0 to test access
    await env.BUCKET.list({ limit: 1 });
    r2Status = 'connected';
  } catch (err: any) {
    r2Status = `error: ${err.message?.slice(0, 80) || 'unknown'}`;
  }

  // Check Wiki
  let wikiStatus = 'not_configured';
  try {
    const wikiResult = await env.DB.prepare('SELECT COUNT(*) as count FROM wiki_pages').first();
    wikiStatus = `connected (${wikiResult?.count || 0} pages)`;
  } catch (err: any) {
    wikiStatus = `not_initialized`;
  }

  const response: HealthResponse = {
    status: 'healthy',
    version: '2.1.0-worker',
    database: dbStatus,
    vectorize: vectorizeStatus,
    r2: r2Status,
    wiki: wikiStatus,
  };

  return jsonResponse(response, 200, request);
}

async function handleQuery(env: Env, request: Request, ctx: ExecutionContext): Promise<Response> {
  const body = await request.json() as QueryRequest;
  const timer = new Timer();

  // Step 1: Classify
  let classification;
  if (body.force_live) {
    classification = { route: 'live' as const, reason: 'Forced live mode', domainHint: '', confidence: 1.0, method: 'override' };
  } else {
    classification = await classifyQuery(env, body.query);
  }

  // Step 2: RAG retrieval
  let chunks: any[] = [];
  let similarityScore: number | null = null;
  let route = 'live';

  try {
    if (classification.route === 'rag' || classification.route === 'live') {
      const [retrieved, avgSim] = await retrieve(env, body.query, {
        domainFilter: body.domain_filter,
        tierFilter: body.tier_filter,
      });
      chunks = retrieved;
      similarityScore = avgSim;

      if (chunks.length > 0 && classification.route === 'rag' && !shouldEscalateToLive(avgSim, chunks.some(c => c.sourceTier === 'P1'))) {
        route = 'rag';
      } else if (chunks.length > 0) {
        route = 'rag+live';
      }
    }
  } catch {
    route = 'live';
  }

  // Step 3: Live scrape fallback
  let scrapedText: string | null = null;
  if (route === 'live' || route === 'rag+live') {
    try {
      const results = await liveScrape(env, { query: body.query });
      const successful = results.filter(r => r.success);
      if (successful.length > 0) {
        scrapedText = successful.map(r => r.markdown).join('\n\n');
      }
    } catch { /* non-critical */ }
  }

  // Step 4: Synthesize
  const synthesis = await synthesize(env, body.query, chunks, scrapedText, body.max_tokens);

  // Step 5: Verification (best-effort)
  let verificationData = null;
  try {
    const sources = chunks.map(c => ({ url: c.sourceUrl, tier: c.sourceTier, snippet: c.rawText.slice(0, 300) }));
    if (scrapedText) sources.push({ url: 'live_scrape', tier: 'P3', snippet: scrapedText.slice(0, 500) });
    if (sources.length > 0) {
      const claims = extractClaimsFromSources(sources as any);
      if (claims.length > 0) {
        const verification = await verifyClaimsFromSources(sources as any);
        verificationData = {
          claims: verification.claims.slice(0, 5).map(c => ({
            statement: c.statement.slice(0, 150),
            status: c.epistemicStatus,
            confidence: c.confidence,
            evidence_type: c.evidenceType,
          })),
          summary: {
            established: verification.establishedCount,
            tentative: verification.tentativeCount,
            contested: verification.contestedCount,
            unverifiable: verification.unverifiableCount,
          },
        };
      }
    }
  } catch { /* non-critical */ }

  const response: QueryResponse = {
    answer: synthesis.answer,
    route,
    method: synthesis.method,
    sources: synthesis.sourcesUsed,
    token_count: synthesis.tokenCount,
    latency_ms: timer.elapsed(),
    similarity_score: similarityScore,
    validation: { is_valid: true, total_claims: 0, cited_claims: 0, warnings: [] },
    model_used: synthesis.modelUsed,
    provider: synthesis.provider,
    fallback_count: synthesis.fallbackCount,
    verification: verificationData,
  };

  // Log query in background
  ctx.waitUntil(logQuery(env, body.query, route, similarityScore, synthesis.modelUsed, timer.elapsed()));

  return jsonResponse(response, 200, request);
}

async function handleClassify(env: Env, request: Request): Promise<Response> {
  const body = await request.json() as ClassifyRequest;
  const result = await classifyQuery(env, body.query);
  return jsonResponse({
    route: result.route,
    reason: result.reason,
    domain_hint: result.domainHint,
    confidence: result.confidence,
    method: result.method,
  }, 200, request);
}

async function handleSearch(env: Env, request: Request): Promise<Response> {
  const body = await request.json() as SearchRequest;
  const [chunks, avgSim] = await retrieve(env, body.query, {
    topK: body.top_k || 5,
    finalK: body.top_k || 5,
    domainFilter: body.domain,
  });

  const results = chunks.map(c => ({
    text: c.rawText,
    source_url: c.sourceUrl,
    source_tier: c.sourceTier,
    domain: c.domain,
    title: c.title,
    authors: c.authors,
    similarity: c.similarityScore,
    token_count: c.tokenCount,
  }));

  return jsonResponse({ results, avg_similarity: avgSim, count: results.length }, 200, request);
}

async function handleScrape(env: Env, request: Request): Promise<Response> {
  const body = await request.json() as { query?: string; urls?: string[] };
  const results = await liveScrape(env, { query: body.query, urls: body.urls });

  return jsonResponse({
    results: results.map(r => ({
      url: r.url, markdown: r.markdown, title: r.title,
      success: r.success, error: r.error,
    })),
    successful: results.filter(r => r.success).length,
    total: results.length,
  }, 200, request);
}

async function handleResearch(env: Env, request: Request, ctx: ExecutionContext): Promise<Response> {
  const body = await request.json() as ResearchRequest;
  const timer = new Timer();

  const researchData = await deepResearch(
    env, body.query || '',
    body.classification || 'web',
    body.depth || 'quick',
    body.verify !== false,
    body.extract || false,
  );

  const report = await generateResearchReport(
    env, body.query || '',
    researchData.sources,
    researchData.verification,
    body.depth || 'quick',
  );

  const response: ResearchResponse = {
    query: body.query || '',
    executive_summary: report.executiveSummary,
    findings: report.findings,
    debates: report.debates,
    speculative: report.speculative,
    sources: report.sources,
    verification: researchData.verification ? {
      claims: researchData.verification.claims.slice(0, 10).map(c => ({
        statement: c.statement.slice(0, 150),
        status: c.epistemicStatus,
        confidence: c.confidence,
        evidence_type: c.evidenceType,
      })),
      summary: {
        established: researchData.verification.establishedCount,
        tentative: researchData.verification.tentativeCount,
        contested: researchData.verification.contestedCount,
        unverifiable: researchData.verification.unverifiableCount,
      },
    } : null,
    extracted_claims: researchData.extracted_claims,
    sub_queries: researchData.sub_queries,
    depth: body.depth || 'quick',
    latency_ms: timer.elapsed(),
    raw_report: report.rawReport,
  };

  return jsonResponse(response, 200, request);
}

async function handleVerify(env: Env, request: Request): Promise<Response> {
  const body = await request.json() as ClassifyRequest;
  const sources = await searchRouter(env, body.query, 'academic');
  const verification = await verifyClaimsFromSources(sources);

  if (verification.claims.length > 0) {
    const claim = verification.claims[0];
    return jsonResponse({
      claim: claim.statement,
      status: claim.epistemicStatus,
      confidence: claim.confidence,
      evidence_type: claim.evidenceType,
      supporting_sources: claim.supportingSources.length,
      conflicting_sources: claim.conflictingSources.length,
    }, 200, request);
  }

  return jsonResponse({
    claim: body.query,
    status: 'UNVERIFIED',
    confidence: 0.1,
    evidence_type: 'unknown',
    supporting_sources: 0,
    conflicting_sources: 0,
  }, 200, request);
}

function handleResearchStatus(request: Request): Response {
  return jsonResponse({
    version: '2.1-worker',
    architecture: 'cloudflare_worker+d1+r2+vectorize+llm_wiki',
    upgrades: {
      tier_enforcement: { status: 'active', P1_domains: 14, P2_domains: 9, P3_domains: 5 },
      verification_loop: { status: 'active', epistemic_markers: ['ESTABLISHED', 'TENTATIVE', 'ACTIVE_DEBATE', 'SPECULATIVE', 'UNVERIFIED'] },
      parallel_orchestration: { status: 'active', graceful_degradation: true },
      research_report_mode: { status: 'active' },
      iterative_research: { status: 'active', opt_in: true, max_cycles: 3 },
      structured_extraction: { status: 'active', p1_only: true },
      temporal_decay: { status: 'active', factor: 0.95 },
    },
    endpoints: {
      '/research': 'Deep research with structured report',
      '/verify': 'Claim verification with epistemic markers',
      '/query': 'Standard pipeline with tier enforcement + verification',
      '/research/status': 'This status endpoint',
    },
  }, 200, request);
}

async function handleIngestUrl(env: Env, request: Request, ctx: ExecutionContext): Promise<Response> {
  const body = await request.json() as IngestURLRequest;
  const timer = new Timer();

  try {
    // Fetch the URL
    const fetchResp = await fetch(body.url, {
      headers: { 'User-Agent': 'APEX-Research-Agent/2.0' },
    });

    if (!fetchResp.ok) {
      return errorResponse(`Failed to fetch URL: HTTP ${fetchResp.status}`, 400, request);
    }

    const contentType = fetchResp.headers.get('content-type') || '';
    let text = await fetchResp.text();

    // Basic HTML stripping
    if (contentType.includes('html')) {
      text = text
        .replace(/<script[\s\S]*?<\/script>/gi, '')
        .replace(/<style[\s\S]*?<\/style>/gi, '')
        .replace(/<[^>]+>/g, ' ')
        .replace(/\s+/g, ' ')
        .trim();
    }

    if (text.length < 50) {
      return errorResponse('Insufficient content extracted', 400, request);
    }

    // Chunk the text (simple fixed-size chunking)
    const chunkSize = body.chunk_size || 512;
    const overlap = Math.floor(chunkSize * (body.overlap_pct || 0.2));
    const chunks: string[] = [];
    let pos = 0;

    while (pos < text.length) {
      chunks.push(text.slice(pos, pos + chunkSize));
      pos += chunkSize - overlap;
    }

    const domain = extractDomain(body.url);
    const tier = body.source_tier || 'UNV';

    // Embed and upsert each chunk
    let upserted = 0;
    for (let i = 0; i < chunks.length; i++) {
      const id = generateUUID();
      const r2Key = `docs/${await hashText(body.url)}/${i}.txt`;

      // Store full text in R2
      ctx.waitUntil(
        env.BUCKET.put(r2Key, chunks[i])
      );

      // Embed and upsert to Vectorize
      const embedding = await embedAndUpsert(env, id, chunks[i], {
        source_url: body.url,
        domain,
        tier,
      });

      // Insert metadata into D1
      await env.DB.prepare(`
        INSERT OR REPLACE INTO documents (id, source_url, source_tier, domain, doc_type, title, authors, text_snippet, r2_key, chunk_index, total_chunks, metadata, token_count)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
      `).bind(
        id, body.url, tier, domain, body.doc_type || 'article',
        body.title || null,
        body.authors ? JSON.stringify(body.authors) : null,
        chunks[i].slice(0, 500),
        r2Key, i, chunks.length,
        JSON.stringify({ publishedDate: null }),
        Math.ceil(chunks[i].length / 4),
      ).run();

      upserted++;
    }

    return jsonResponse({
      status: 'success',
      chunks_upserted: upserted,
      message: `Ingested ${upserted} chunks from ${body.url}`,
    }, 200, request);
  } catch (err: any) {
    return errorResponse(`Ingestion failed: ${err.message}`, 500, request);
  }
}

async function handleEmbedPending(env: Env, request: Request): Promise<Response> {
  const count = await embedPendingChunks(env);
  return jsonResponse({ status: 'success', embedded: count }, 200, request);
}

function handleRouterStatus(request: Request): Response {
  return jsonResponse(getRouterStatus(), 200, request);
}

function handleRouterChain(request: Request): Response {
  return jsonResponse({
    fallback_order: FALLBACK_CHAIN.map((m, i) => ({
      position: i + 1,
      name: m.name,
      provider: m.provider,
      model_id: m.modelId,
      tier: m.tier,
      price_input_per_m: m.priceInputPerM,
      price_output_per_m: m.priceOutputPerM,
      context_window: m.contextWindow,
      description: m.description,
    })),
    note: 'All models use native Workers AI binding',
  }, 200, request);
}

async function handleRouterTest(env: Env, request: Request): Promise<Response> {
  const results = await testAllModels(env);
  return jsonResponse({
    total_models: results.length,
    configured: results.filter(r => r.configured).length,
    reachable: results.filter(r => r.reachable).length,
    results: results.map(r => ({
      name: r.modelName,
      model_id: r.modelId,
      provider: r.provider,
      configured: r.configured,
      reachable: r.reachable,
      latency_ms: r.latencyMs,
      error: r.error,
      sample_output: r.sampleOutput,
    })),
  }, 200, request);
}

// ── Background Logging ──

async function logQuery(
  env: Env,
  query: string,
  route: string,
  similarity: number | null,
  modelUsed: string,
  latencyMs: number,
): Promise<void> {
  try {
    const id = generateUUID();
    await env.DB.prepare(`
      INSERT INTO query_log (id, query_text, route, similarity_score, model_used, latency_ms)
      VALUES (?, ?, ?, ?, ?, ?)
    `).bind(id, query.slice(0, 2000), route, similarity, modelUsed, latencyMs).run();
  } catch {
    // Non-critical — don't fail the request
  }
}
