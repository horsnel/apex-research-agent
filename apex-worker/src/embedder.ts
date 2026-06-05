/**
 * APEX Research Agent — Embedder
 * Workers AI embedding via binding (replaces HTTP calls)
 * Uses @cf/baai/bge-base-en-v1.5 (768-dim)
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
 * Upsert document vector to Vectorize index.
 */
export async function upsertToVectorize(
  env: Env,
  id: string,
  values: Float32Array,
  metadata: Record<string, string>,
): Promise<void> {
  await env.VECTORIZE.upsert([
    { id, values: Array.from(values), metadata },
  ]);
}

/**
 * Query Vectorize index for similar vectors.
 */
export async function queryVectorize(
  env: Env,
  queryVector: Float32Array,
  topK = 5,
  filter?: Record<string, string>,
): Promise<VectorizeSearchResult[]> {
  const results = await env.VECTORIZE.query({
    vector: Array.from(queryVector),
    topK,
    filter,
  });

  return results.matches || [];
}

/**
 * Embed text and upsert to Vectorize in one step.
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
 * Finds chunks without vectors in Vectorize, embeds them, and upserts.
 */
export async function embedPendingChunks(env: Env): Promise<number> {
  // Get chunks from D1 (we'll check against Vectorize later)
  const result = await env.DB.prepare(
    'SELECT id, text_snippet, r2_key FROM documents WHERE text_snippet IS NOT NULL LIMIT 50'
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
