/**
 * APEX 2.0 — Concurrency Safety Layer (Matryca-Plumber Pattern)
 *
 * Advisory write locks for multi-user/multi-agent wiki access.
 * Prevents lost updates and provides conflict resolution.
 */

import { Env } from '../types';
import { generateUUID } from '../utils';
import { LockAcquisition, ConflictResolutionStrategy } from './types';

// ── Configuration ──

const DEFAULT_LOCK_TTL_SECONDS = 60;
const MAX_RETRY_ATTEMPTS = 3;
const RETRY_DELAY_MS = 500;

// ── Acquire Write Lock ──

export async function acquireWriteLock(
  env: Env,
  pageId: string,
  holder: string,
  ttlSeconds: number = DEFAULT_LOCK_TTL_SECONDS,
): Promise<LockAcquisition | null> {
  const now = new Date();
  const nowISO = now.toISOString();
  const expiresAt = new Date(now.getTime() + ttlSeconds * 1000).toISOString();

  // First, clean up any expired locks for this page
  await pruneExpiredLocksForPage(env, pageId);

  // Check if there's an active (non-expired, non-released) lock
  const activeLock = await env.DB.prepare(`
    SELECT id, holder, expires_at FROM wiki_locks
    WHERE page_id = ? AND released_at IS NULL AND expires_at > ?
    LIMIT 1
  `).bind(pageId, nowISO).first();

  if (activeLock) {
    // Lock is held by someone else
    return null;
  }

  // Acquire the lock
  const lockId = generateUUID();
  try {
    await env.DB.prepare(`
      INSERT INTO wiki_locks (id, page_id, holder, acquired_at, expires_at, released_at)
      VALUES (?, ?, ?, ?, ?, NULL)
    `).bind(lockId, pageId, holder, nowISO, expiresAt).run();

    return {
      lockId,
      pageId,
      holder,
      acquiredAt: nowISO,
      expiresAt,
    };
  } catch {
    // Concurrent insert might fail — lock already acquired by another process
    return null;
  }
}

// ── Release Write Lock ──

export async function releaseWriteLock(
  env: Env,
  lockId: string,
): Promise<boolean> {
  const now = new Date().toISOString();

  const result = await env.DB.prepare(`
    UPDATE wiki_locks SET released_at = ? WHERE id = ? AND released_at IS NULL
  `).bind(now, lockId).run();

  return (result.meta?.changes || 0) > 0;
}

// ── Extend Write Lock ──

export async function extendWriteLock(
  env: Env,
  lockId: string,
  additionalSeconds: number = DEFAULT_LOCK_TTL_SECONDS,
): Promise<boolean> {
  // Get current lock
  const lock = await env.DB.prepare(
    'SELECT expires_at FROM wiki_locks WHERE id = ? AND released_at IS NULL'
  ).bind(lockId).first();

  if (!lock) return false;

  const currentExpiry = new Date(lock.expires_at as string);
  const newExpiry = new Date(currentExpiry.getTime() + additionalSeconds * 1000).toISOString();

  const result = await env.DB.prepare(`
    UPDATE wiki_locks SET expires_at = ? WHERE id = ? AND released_at IS NULL
  `).bind(newExpiry, lockId).run();

  return (result.meta?.changes || 0) > 0;
}

// ── With Write Lock (Higher-Order Function) ──

export async function withWriteLock<T>(
  env: Env,
  pageId: string,
  holder: string,
  fn: () => Promise<T>,
  maxRetries: number = MAX_RETRY_ATTEMPTS,
  retryDelayMs: number = RETRY_DELAY_MS,
): Promise<T | null> {
  for (let attempt = 0; attempt < maxRetries; attempt++) {
    const lock = await acquireWriteLock(env, pageId, holder);

    if (lock) {
      try {
        const result = await fn();
        return result;
      } finally {
        await releaseWriteLock(env, lock.lockId);
      }
    }

    // Wait before retrying
    if (attempt < maxRetries - 1) {
      await sleep(retryDelayMs * (attempt + 1)); // Exponential backoff
    }
  }

  return null; // Could not acquire lock after retries
}

// ── Detect Concurrent Modifications ──

export async function detectConcurrentModifications(
  env: Env,
  pageId: string,
  baseVersion: number,
): Promise<{
  hasConflict: boolean;
  currentVersion: number;
  modifiedAt: string | null;
}> {
  const row = await env.DB.prepare(
    'SELECT version, updated_at FROM wiki_pages WHERE id = ?'
  ).bind(pageId).first();

  if (!row) {
    return { hasConflict: false, currentVersion: 0, modifiedAt: null };
  }

  const currentVersion = row.version as number;
  const hasConflict = currentVersion > baseVersion;

  return {
    hasConflict,
    currentVersion,
    modifiedAt: row.updated_at as string,
  };
}

// ── Resolve Conflict ──

export async function resolveConflict(
  env: Env,
  pageId: string,
  localVersion: string,
  remoteVersion: string,
  strategy: ConflictResolutionStrategy,
): Promise<{
  resolved: boolean;
  strategy: ConflictResolutionStrategy;
  winner: string;
}> {
  switch (strategy) {
    case 'last_writer_wins': {
      // Remote version (most recent) wins
      return {
        resolved: true,
        strategy: 'last_writer_wins',
        winner: 'remote',
      };
    }

    case 'merge': {
      // Use LLM to merge the two versions
      try {
        const localContent = localVersion;
        const remoteContent = remoteVersion;

        const mergePrompt = `You are a wiki content merger. Merge these two versions of wiki content into a single coherent version that preserves information from both.

Local version:
${localContent.slice(0, 4000)}

Remote version:
${remoteContent.slice(0, 4000)}

Rules:
1. Preserve all factual claims from both versions
2. If claims conflict, note the conflict
3. Maintain consistent formatting
4. Keep all citations from both versions
5. Output the merged content only, no commentary`;

        const { routeLLMCall } = await import('../llm-router');
        const result = await routeLLMCall(
          env,
          [{ role: 'user', content: mergePrompt }],
          4096, 0.0, undefined, false, false,
        );

        if (!result.content.startsWith('[ALL_LLM_FAILED]')) {
          // Store merged content
          const slug = await env.DB.prepare('SELECT slug FROM wiki_pages WHERE id = ?').bind(pageId).first();
          if (slug?.slug) {
            await env.BUCKET.put(`wiki/pages/${slug.slug}.md`, result.content);
            await env.DB.prepare(`
              UPDATE wiki_pages SET content_snippet = ?, version = version + 1, updated_at = ? WHERE id = ?
            `).bind(
              result.content.slice(0, 500),
              new Date().toISOString(),
              pageId,
            ).run();
          }

          return {
            resolved: true,
            strategy: 'merge',
            winner: 'merged',
          };
        }
      } catch {
        // Fall through to abort
      }

      return {
        resolved: false,
        strategy: 'merge',
        winner: 'none',
      };
    }

    case 'abort':
    default: {
      // Abort the write
      return {
        resolved: false,
        strategy: 'abort',
        winner: 'none',
      };
    }
  }
}

// ── Get Active Locks ──

export async function getActiveLocks(env: Env): Promise<LockAcquisition[]> {
  const now = new Date().toISOString();

  const rows = await env.DB.prepare(`
    SELECT id, page_id, holder, acquired_at, expires_at
    FROM wiki_locks
    WHERE released_at IS NULL AND expires_at > ?
    ORDER BY acquired_at DESC
  `).bind(now).all();

  return (rows.results as any[]).map(row => ({
    lockId: row.id,
    pageId: row.page_id,
    holder: row.holder,
    acquiredAt: row.acquired_at,
    expiresAt: row.expires_at,
  }));
}

// ── Prune Expired Locks ──

export async function pruneExpiredLocks(env: Env): Promise<number> {
  const now = new Date().toISOString();

  const result = await env.DB.prepare(`
    DELETE FROM wiki_locks WHERE expires_at < ? AND released_at IS NULL
  `).bind(now).run();

  return result.meta?.changes || 0;
}

// ── Prune Expired Locks for Specific Page ──

async function pruneExpiredLocksForPage(env: Env, pageId: string): Promise<void> {
  const now = new Date().toISOString();

  await env.DB.prepare(`
    DELETE FROM wiki_locks WHERE page_id = ? AND expires_at < ? AND released_at IS NULL
  `).bind(pageId, now).run();
}

// ── Sleep Utility ──

function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}

// ── Export Configuration ──

export const CONCURRENCY_CONFIG = {
  DEFAULT_LOCK_TTL_SECONDS,
  MAX_RETRY_ATTEMPTS,
  RETRY_DELAY_MS,
};
