/**
 * APEX Research Agent — Synthesizer
 * APEX compression + citations synthesis
 */

import { Env, RetrievedChunk, SynthesisResult, SourceInfo } from './types';
import { synthesizeWithRouter, APEX_SYSTEM_PROMPT } from './llm-router';
import { enforceSourceTier } from './utils';

/**
 * Synthesize an answer from retrieved chunks and/or scraped text.
 */
export async function synthesize(
  env: Env,
  query: string,
  chunks: RetrievedChunk[] = [],
  scrapedText: string | null = null,
  maxTokens?: number,
): Promise<SynthesisResult> {
  // Build context string from chunks
  const chunkContext = chunks.map((c, i) => {
    const tier = enforceSourceTier(c.sourceUrl, c.sourceTier);
    return `[${i + 1}, ${tier}] ${c.rawText}`;
  }).join('\n\n');

  // Add scraped content if available
  const fullContext = scrapedText
    ? `${chunkContext}\n\n[LIVE WEB CONTENT, P3]\n${scrapedText.slice(0, 3000)}`
    : chunkContext;

  if (!fullContext.trim()) {
    return {
      answer: 'No relevant sources found for this query.',
      method: 'raw_context',
      tokenCount: 0,
      sourcesUsed: [],
      modelUsed: 'none',
      provider: 'none',
      fallbackCount: 0,
    };
  }

  // Check for pass-through (very high similarity)
  const avgSimilarity = chunks.length > 0
    ? chunks.reduce((sum, c) => sum + c.similarityScore, 0) / chunks.length
    : 0;

  if (avgSimilarity > 0.85 && chunks.length > 0) {
    // Direct quote from best chunk
    const best = chunks[0];
    return {
      answer: best.rawText,
      method: 'pass_through',
      tokenCount: best.tokenCount,
      sourcesUsed: [{
        url: best.sourceUrl,
        tier: best.sourceTier,
        title: best.title,
        similarity: best.similarityScore,
      }],
      modelUsed: 'pass-through',
      provider: 'passthrough',
      fallbackCount: 0,
    };
  }

  // Use LLM for synthesis
  const tokenBudget = maxTokens || 150;
  const tableNeeded = query.toLowerCase().includes('table') ||
    query.toLowerCase().includes('compare') ||
    query.toLowerCase().includes('list');

  const result = await synthesizeWithRouter(
    env, query, fullContext, tokenBudget, avgSimilarity, tableNeeded,
  );

  // Build sources list
  const sourcesUsed: SourceInfo[] = chunks.slice(0, 5).map(c => ({
    url: c.sourceUrl,
    tier: enforceSourceTier(c.sourceUrl, c.sourceTier),
    title: c.title,
    similarity: c.similarityScore,
  }));

  // Add scraped sources
  if (scrapedText) {
    sourcesUsed.push({
      url: 'live_scrape', tier: 'P3', title: 'Live web content', similarity: 0,
    });
  }

  const method = tableNeeded ? 'table' : 'synthesis';

  return {
    answer: result.content.startsWith('[ALL_LLM_FAILED]')
      ? chunks[0]?.rawText || 'Unable to generate response.'
      : result.content,
    method,
    tokenCount: estimateTokens(result.content),
    sourcesUsed,
    modelUsed: result.modelName,
    provider: result.provider,
    fallbackCount: result.fallbackCount,
  };
}

function estimateTokens(text: string): number {
  return Math.ceil(text.length / 4);
}
