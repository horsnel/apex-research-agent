/**
 * APEX Research Agent — Embedder
 * Workers AI embedding via binding (replaces HTTP calls)
 * Uses @cf/baai/bge-base-en-v1.5 (768-dim)
 *
 * D1-only storage: embeddings stored as JSON TEXT columns,
 * vector search via application-side cosine similarity.
 */

import { Env } from './types';

const EMBEDDING_MODEL = '@cf/baai/bge-base-en-v1.5';
const EMBEDDING_DIMENSIONS = 768;
const BATCH_SIZE = 50;

/**
 * Generate embedding for a single text using Workers AI binding.
 */
export async function embedSingle(env: Env, text: string): Promise<Float32Array> {
  const result = await env.AI.run(EMBEDDING_MODEL, { text: [text] }) as {
    data?: Array<{ embedding?: number[] }>;
  };

  if (result.data?.[0]?.embedding) {
    return new Float32Array(result.data[0].embedding);
  }

  // Fallback: return zero vector
  return new Float32Array(EMBEDDING_DIMENSIONS);
}

/**
 * Generate embeddings for multiple texts in batches.
 */
export async function embedBatch(env: Env, texts: string[]): Promise<Float32Array[]> {
  const allEmbeddings: Float32Array[] = [];

  for (let i = 0; i < texts.length; i += BATCH_SIZE) {
    const batch = texts.slice(i, i + BATCH_SIZE);

    try {
      const result = await env.AI.run(EMBEDDING_MODEL, { text: batch }) as {
        data?: Array<{ embedding?: number[] }>;
      };

      if (result.data) {
        for (const item of result.data) {
          if (item.embedding) {
            allEmbeddings.push(new Float32Array(item.embedding));
          } else {
            allEmbeddings.push(new Float32Array(EMBEDDING_DIMENSIONS));
          }
        }
      } else {
        // All zero vectors for failed batch
        for (let j = 0; j < batch.length; j++) {
          allEmbeddings.push(new Float32Array(EMBEDDING_DIMENSIONS));
        }
      }
    } catch (err) {
      console.error(`Embedding batch failed: ${err}`);
      for (let j = 0; j < batch.length; j++) {
        allEmbeddings.push(new Float32Array(EMBEDDING_DIMENSIONS));
      }
    }
  }

  return allEmbeddings;
}

/**
 * Upsert document vector — stores embedding as JSON in D1 instead of Vectorize.
 */
export async function upsertToVectorize(
  env: Env,
  id: string,
  values: Float32Array,
  metadata: Record<string, string>,
): Promise<void> {
  const embeddingJson = JSON.stringify(Array.from(values));

  // Determine table based on metadata type
  if (metadata.type === 'wiki') {
    await env.DB.prepare('UPDATE wiki_pages SET embedding = ? WHERE slug = ?')
      .bind(embeddingJson, metadata.slug).run();
  } else {
    await env.DB.prepare('UPDATE documents SET embedding = ? WHERE id = ?')
      .bind(embeddingJson, id).run();
  }
}

/**
 * Query for similar vectors — fetches embeddings from D1 and computes
 * cosine similarity in application code instead of using Vectorize.
 */
export async function queryVectorize(
  env: Env,
  queryVector: Float32Array,
  topK = 5,
  filter?: Record<string, string>,
): Promise<Array<{ id: string; score: number; metadata: Record<string, string> }>> {
  const queryArr = Array.from(queryVector);
  const queryNorm = Math.sqrt(queryArr.reduce((sum, v) => sum + v * v, 0));

  if (queryNorm === 0) return [];

  const scored: Array<{ id: string; score: number; metadata: Record<string, string> }> = [];

  // Search wiki_pages if filter type is 'wiki' or no filter
  if (!filter || filter.type === 'wiki') {
    try {
      const wikiRows = await env.DB.prepare(
        "SELECT id, slug, embedding FROM wiki_pages WHERE embedding IS NOT NULL AND state NOT IN ('archived', 'draft') LIMIT 200"
      ).all();

      for (const row of wikiRows.results as any[]) {
        if (!row.embedding) continue;
        try {
          const embedding = JSON.parse(row.embedding as string);
          const dotProduct = embedding.reduce((sum: number, v: number, i: number) => sum + v * (queryArr[i] || 0), 0);
          const embNorm = Math.sqrt(embedding.reduce((sum: number, v: number) => sum + v * v, 0));
          const similarity = embNorm > 0 ? dotProduct / (queryNorm * embNorm) : 0;
          scored.push({
            id: row.id as string,
            score: similarity,
            metadata: { slug: row.slug as string, type: 'wiki' },
          });
        } catch { /* skip invalid embeddings */ }
      }
    } catch (err) {
      console.error(`Wiki vector search failed: ${err}`);
    }
  }

  // Search documents if no wiki-only filter
  if (!filter || filter.type !== 'wiki') {
    try {
      let sql = 'SELECT id, embedding, source_url, domain, source_tier FROM documents WHERE embedding IS NOT NULL';
      const params: any[] = [];

      if (filter?.domain) {
        sql += ' AND domain = ?';
        params.push(filter.domain);
      }
      if (filter?.tier) {
        sql += ' AND source_tier = ?';
        params.push(filter.tier);
      }

      sql += ' LIMIT 200';

      const docRows = await env.DB.prepare(sql).bind(...params).all();

      for (const row of docRows.results as any[]) {
        if (!row.embedding) continue;
        try {
          const embedding = JSON.parse(row.embedding as string);
          const dotProduct = embedding.reduce((sum: number, v: number, i: number) => sum + v * (queryArr[i] || 0), 0);
          const embNorm = Math.sqrt(embedding.reduce((sum: number, v: number) => sum + v * v, 0));
          const similarity = embNorm > 0 ? dotProduct / (queryNorm * embNorm) : 0;
          scored.push({
            id: row.id as string,
            score: similarity,
            metadata: {
              source_url: row.source_url as string || '',
              domain: row.domain as string || '',
              tier: row.source_tier as string || '',
              type: 'document',
            },
          });
        } catch { /* skip invalid embeddings */ }
      }
    } catch (err) {
      console.error(`Document vector search failed: ${err}`);
    }
  }

  return scored.sort((a, b) => b.score - a.score).slice(0, topK);
}

/**
 * Embed text and upsert to D1 in one step.
 */
export async function embedAndUpsert(
  env: Env,
  id: string,
  text: string,
  metadata: Record<string, string>,
): Promise<Float32Array> {
  const embedding = await embedSingle(env, text);
  await upsertToVectorize(env, id, embedding, metadata);
  return embedding;
}

/**
 * Embed unembedded chunks from D1.
 * Finds chunks without embeddings, embeds them, and stores in D1.
 */
export async function embedPendingChunks(env: Env): Promise<number> {
  // Get chunks from D1 that don't have embeddings yet
  const result = await env.DB.prepare(
    "SELECT id, text_snippet FROM documents WHERE text_snippet IS NOT NULL AND embedding IS NULL LIMIT 50"
  ).all();

  if (!result.results || result.results.length === 0) return 0;

  let count = 0;
  for (const row of result.results) {
    const id = row.id as string;
    const text = (row.text_snippet as string) || '';

    if (!text) continue;

    try {
      const embedding = await embedSingle(env, text);
      await upsertToVectorize(env, id, embedding, {
        source_url: '', // Would need full row data
        domain: '',
        tier: 'UNV',
      });
      count++;
    } catch (err) {
      console.error(`Failed to embed chunk ${id}: ${err}`);
    }
  }

  return count;
}
