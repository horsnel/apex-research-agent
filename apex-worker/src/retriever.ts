/**
 * APEX Research Agent — Hybrid Retriever
 * Vectorize (vector search) + D1 FTS5 (keyword search) + RRF fusion
 * Replaces PostgreSQL pgvector + tsvector
 */

import { Env, RetrievedChunk, DocumentRow } from './types';
import { embedSingle, queryVectorize } from './embedder';
import { enforceSourceTier, applySourceHierarchy, applyTemporalDecay, estimateTokens } from './utils';

const RAG_TOP_K = 5;
const RAG_FINAL_K = 3;
const SIMILARITY_THRESHOLD = 0.5;
const RRF_K = 60; // Reciprocal Rank Fusion constant

/**
 * Main retrieval function: embed → parallel vector+keyword → RRF → hierarchy → budget
 */
export async function retrieve(
  env: Env,
  query: string,
  options: {
    topK?: number;
    finalK?: number;
    domainFilter?: string;
    tierFilter?: string;
    docTypeFilter?: string;
    maxTokens?: number;
  } = {},
): Promise<[RetrievedChunk[], number]> {
  const topK = options.topK || RAG_TOP_K;
  const finalK = options.finalK || RAG_FINAL_K;
  const maxTokens = options.maxTokens || 2000;

  // Step 1: Embed query
  const queryVector = await embedSingle(env, query);

  // Step 2: Parallel vector + keyword search
  const [vectorResults, keywordResults] = await Promise.allSettled([
    vectorSearch(env, queryVector, topK * 2, options.domainFilter, options.tierFilter),
    keywordSearch(env, query, topK * 2, options.tierFilter),
  ]);

  const vResults = vectorResults.status === 'fulfilled' ? vectorResults.value : [];
  const kResults = keywordResults.status === 'fulfilled' ? keywordResults.value : [];

  // Step 3: Reciprocal Rank Fusion
  const fused = reciprocalRankFusion(vResults, kResults);

  // Step 4: Apply source hierarchy and temporal decay
  for (const chunk of fused) {
    chunk.sourceTier = enforceSourceTier(chunk.sourceUrl, chunk.sourceTier);
    chunk.fusedScore = applySourceHierarchy(chunk.fusedScore, chunk.sourceTier);
    if (chunk.metadata?.publishedDate) {
      chunk.fusedScore = applyTemporalDecay(
        chunk.fusedScore, chunk.metadata.publishedDate as string,
      );
    }
  }

  // Step 5: Sort by fused score, take top finalK
  fused.sort((a, b) => b.fusedScore - a.fusedScore);
  const topResults = fused.slice(0, finalK);

  // Step 6: Apply token budget
  const budgeted = applyTokenBudget(topResults, maxTokens);

  // Calculate average similarity
  const avgSimilarity = budgeted.length > 0
    ? budgeted.reduce((sum, c) => sum + c.similarityScore, 0) / budgeted.length
    : 0;

  return [budgeted, avgSimilarity];
}

/**
 * Vector search using Vectorize index.
 */
async function vectorSearch(
  env: Env,
  queryVector: Float32Array,
  topK: number,
  domainFilter?: string,
  tierFilter?: string,
): Promise<RetrievedChunk[]> {
  try {
    // Build filter for Vectorize
    const filter: Record<string, string> = {};
    if (domainFilter) filter.domain = domainFilter;
    if (tierFilter) filter.tier = tierFilter;

    const matches = await queryVectorize(env, queryVector, topK, Object.keys(filter).length > 0 ? filter : undefined);

    if (!matches || matches.length === 0) return [];

    // Fetch document metadata from D1 by IDs
    const ids = matches.map(m => m.id);
    const placeholders = ids.map(() => '?').join(',');
    const dbResult = await env.DB.prepare(
      `SELECT * FROM documents WHERE id IN (${placeholders})`
    ).bind(...ids).all();

    const docMap = new Map<string, DocumentRow>();
    for (const row of (dbResult.results || [])) {
      docMap.set(row.id as string, row as unknown as DocumentRow);
    }

    // Fetch full text from R2 for matched documents
    const chunks: RetrievedChunk[] = [];
    for (const match of matches) {
      const doc = docMap.get(match.id);
      if (!doc) continue;

      let rawText = doc.text_snippet || '';
      if (doc.r2_key) {
        try {
          const r2Object = await env.BUCKET.get(doc.r2_key);
          if (r2Object) {
            rawText = await r2Object.text();
          }
        } catch {
          // Fall back to snippet
        }
      }

      const authors: string[] = doc.authors ? JSON.parse(doc.authors) : [];

      chunks.push({
        id: doc.id,
        sourceUrl: doc.source_url,
        sourceTier: doc.source_tier,
        domain: doc.domain,
        docType: doc.doc_type,
        title: doc.title || '',
        authors,
        rawText,
        metadata: doc.metadata ? JSON.parse(doc.metadata) : {},
        chunkIndex: doc.chunk_index,
        totalChunks: doc.total_chunks,
        similarityScore: match.score || 0,
        keywordScore: 0,
        fusedScore: 0,
        tokenCount: estimateTokens(rawText),
      });
    }

    return chunks;
  } catch (err) {
    console.error(`Vector search failed: ${err}`);
    return [];
  }
}

/**
 * Keyword search using D1 FTS5.
 */
async function keywordSearch(
  env: Env,
  query: string,
  topK: number,
  tierFilter?: string,
): Promise<RetrievedChunk[]> {
  try {
    let sql = `
      SELECT d.*, rank
      FROM documents_fts f
      JOIN documents d ON d.rowid = f.rowid
      WHERE documents_fts MATCH ?
    `;
    const params: any[] = [query];

    if (tierFilter) {
      sql += ' AND d.source_tier = ?';
      params.push(tierFilter);
    }

    sql += ' ORDER BY rank LIMIT ?';
    params.push(topK);

    const result = await env.DB.prepare(sql).bind(...params).all();

    const chunks: RetrievedChunk[] = [];
    for (const row of (result.results || [])) {
      const doc = row as unknown as DocumentRow;
      const authors: string[] = doc.authors ? JSON.parse(doc.authors) : [];
      const rank = Math.abs(row.rank as number || 0);

      let rawText = doc.text_snippet || '';
      if (doc.r2_key) {
        try {
          const r2Object = await env.BUCKET.get(doc.r2_key);
          if (r2Object) rawText = await r2Object.text();
        } catch { /* fallback to snippet */ }
      }

      chunks.push({
        id: doc.id,
        sourceUrl: doc.source_url,
        sourceTier: doc.source_tier,
        domain: doc.domain,
        docType: doc.doc_type,
        title: doc.title || '',
        authors,
        rawText,
        metadata: doc.metadata ? JSON.parse(doc.metadata) : {},
        chunkIndex: doc.chunk_index,
        totalChunks: doc.total_chunks,
        similarityScore: 0,
        keywordScore: rank,
        fusedScore: 0,
        tokenCount: estimateTokens(rawText),
      });
    }

    return chunks;
  } catch (err) {
    console.error(`Keyword search failed: ${err}`);
    return [];
  }
}

/**
 * Reciprocal Rank Fusion: merge vector and keyword results.
 */
function reciprocalRankFusion(
  vectorResults: RetrievedChunk[],
  keywordResults: RetrievedChunk[],
  k = RRF_K,
): RetrievedChunk[] {
  const scoreMap = new Map<string, { chunk: RetrievedChunk; score: number }>();

  // Vector results
  vectorResults.forEach((chunk, rank) => {
    const rrfScore = 1 / (k + rank + 1);
    const existing = scoreMap.get(chunk.id);
    if (existing) {
      existing.score += rrfScore;
      existing.chunk.similarityScore = chunk.similarityScore;
    } else {
      scoreMap.set(chunk.id, { chunk: { ...chunk }, score: rrfScore });
    }
  });

  // Keyword results
  keywordResults.forEach((chunk, rank) => {
    const rrfScore = 1 / (k + rank + 1);
    const existing = scoreMap.get(chunk.id);
    if (existing) {
      existing.score += rrfScore;
      existing.chunk.keywordScore = chunk.keywordScore;
    } else {
      scoreMap.set(chunk.id, { chunk: { ...chunk }, score: rrfScore });
    }
  });

  // Set fused scores
  for (const [, entry] of scoreMap) {
    entry.chunk.fusedScore = entry.score;
  }

  return Array.from(scoreMap.values())
    .map(entry => entry.chunk)
    .sort((a, b) => b.fusedScore - a.fusedScore);
}

/**
 * Apply token budget — truncate chunks to fit within max tokens.
 */
function applyTokenBudget(chunks: RetrievedChunk[], maxTokens: number): RetrievedChunk[] {
  let totalTokens = 0;
  const result: RetrievedChunk[] = [];

  for (const chunk of chunks) {
    if (totalTokens + chunk.tokenCount <= maxTokens) {
      result.push(chunk);
      totalTokens += chunk.tokenCount;
    } else {
      // Partially include the chunk
      const remaining = maxTokens - totalTokens;
      if (remaining > 50) {
        chunk.rawText = chunk.rawText.slice(0, remaining * 4); // Rough char estimate
        chunk.tokenCount = remaining;
        result.push(chunk);
      }
      break;
    }
  }

  return result;
}
