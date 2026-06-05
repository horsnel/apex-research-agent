/**
 * APEX 2.0 — Synthadoc's 5-State Page Lifecycle
 *
 * State machine:
 *   draft → active (when compiled and verified)
 *   active → stale (when source hash changes, or age > threshold)
 *   active → contradicted (when new source contradicts existing claims)
 *   stale → active (when re-verified and updated)
 *   contradicted → active (when contradiction resolved)
 *   Any → archived (manual or auto after long inactivity)
 */

import { Env } from '../types';
import { routeLLMCall } from '../llm-router';
import { hashText, generateUUID } from '../utils';
import {
  WikiPage,
  WikiPageState,
  PageLifecycleEvent,
  WikiPageRow,
  WikiLifecycleEventRow,
} from './types';

// ── Configuration ──

const STALE_THRESHOLD_HOURS = 168;           // 1 week
const CONTRADICTION_SIMILARITY_THRESHOLD = 0.7;
const ARCHIVE_AFTER_DAYS = 90;

// ── Valid State Transitions ──

const VALID_TRANSITIONS: Record<string, WikiPageState[]> = {
  draft: ['active', 'archived'],
  active: ['stale', 'contradicted', 'archived'],
  stale: ['active', 'archived'],
  contradicted: ['active', 'archived'],
  archived: [],  // Terminal state
};

// ── Check Page Freshness ──

export async function checkPageFreshness(
  env: Env,
  page: WikiPage,
): Promise<{ isFresh: boolean; changedSources: string[] }> {
  const changedSources: string[] = [];
  const now = new Date().toISOString();

  for (const source of page.sources) {
    try {
      // Fetch the source to check if content has changed
      const response = await fetch(source.url, {
        method: 'GET',
        headers: { 'User-Agent': 'APEX-Research-Agent/2.0' },
        signal: AbortSignal.timeout(10000),
      });

      if (!response.ok) continue;

      const content = await response.text();
      const currentHash = await hashText(content);

      // Update last checked time
      await env.DB.prepare(
        'UPDATE wiki_sources SET last_checked_at = ? WHERE url = ? AND content_hash = ?'
      ).bind(now, source.url, source.contentHash).run();

      if (currentHash !== source.contentHash) {
        changedSources.push(source.url);
      }
    } catch {
      // If we can't check a source, skip it (don't mark stale based on network errors)
    }
  }

  // Also check age-based staleness
  const ageHours = page.lastVerifiedAt
    ? (Date.now() - new Date(page.lastVerifiedAt).getTime()) / (1000 * 60 * 60)
    : STALE_THRESHOLD_HOURS + 1;

  const isFresh = changedSources.length === 0 && ageHours < STALE_THRESHOLD_HOURS;

  // If sources have changed or page is too old, mark as stale
  if (!isFresh && page.state === 'active') {
    await transitionState(
      env,
      page.id,
      'stale',
      changedSources.length > 0
        ? `${changedSources.length} source(s) changed: ${changedSources.join(', ')}`
        : `Page age exceeds ${STALE_THRESHOLD_HOURS} hours without verification`,
    );
  }

  return { isFresh, changedSources };
}

// ── Transition State ──

export async function transitionState(
  env: Env,
  pageId: string,
  newState: WikiPageState,
  reason: string,
  sourceHash?: string,
): Promise<boolean> {
  // Get current state
  const row = await env.DB.prepare(
    'SELECT state FROM wiki_pages WHERE id = ?'
  ).bind(pageId).first();

  if (!row) return false;

  const currentState = row.state as WikiPageState;

  // Validate transition
  const allowedTransitions = VALID_TRANSITIONS[currentState] || [];
  if (!allowedTransitions.includes(newState)) {
    return false; // Invalid transition
  }

  const now = new Date().toISOString();

  // Update page state
  await env.DB.prepare(`
    UPDATE wiki_pages
    SET state = ?, updated_at = ?, last_verified_at = ?
    WHERE id = ?
  `).bind(newState, now, now, pageId).run();

  // Log the lifecycle event
  const eventId = generateUUID();
  await env.DB.prepare(`
    INSERT INTO wiki_lifecycle_events (id, page_id, from_state, to_state, reason, source_hash, created_at)
    VALUES (?, ?, ?, ?, ?, ?, ?)
  `).bind(
    eventId,
    pageId,
    currentState,
    newState,
    reason,
    sourceHash || null,
    now,
  ).run();

  return true;
}

// ── Detect Contradictions Between Two Pages ──

export async function detectContradictions(
  env: Env,
  pageA: WikiPage,
  pageB: WikiPage,
): Promise<{
  hasContradictions: boolean;
  contradictions: Array<{
    claimA: string;
    claimB: string;
    severity: 'low' | 'medium' | 'high' | 'critical';
  }>;
}> {
  // Use LLM to compare claims from both pages
  const prompt = `Compare these two wiki pages and identify any CONTRADICTING claims between them.

Page A (${pageA.title}):
${pageA.content.slice(0, 3000)}

Page B (${pageB.title}):
${pageB.content.slice(0, 3000)}

For each contradiction found, respond with a JSON array:
[{"claimA": "claim from page A", "claimB": "contradicting claim from page B", "severity": "low|medium|high|critical"}]

If no contradictions exist, respond with: []`;

  const result = await routeLLMCall(
    env,
    [{ role: 'user', content: prompt }],
    1024, 0.0, undefined, false, false,
  );

  if (result.content.startsWith('[ALL_LLM_FAILED]')) {
    return { hasContradictions: false, contradictions: [] };
  }

  try {
    let content = result.content.trim();
    // Extract JSON from possible markdown code blocks
    const jsonMatch = content.match(/\[[\s\S]*\]/);
    if (!jsonMatch) {
      return { hasContradictions: false, contradictions: [] };
    }
    content = jsonMatch[0];

    const contradictions = JSON.parse(content);

    if (!Array.isArray(contradictions)) {
      return { hasContradictions: false, contradictions: [] };
    }

    // Filter by similarity threshold (use as confidence threshold)
    const significantContradictions = contradictions.filter(
      (c: any) => c.severity === 'high' || c.severity === 'critical' || c.severity === 'medium',
    );

    return {
      hasContradictions: significantContradictions.length > 0,
      contradictions: significantContradictions,
    };
  } catch {
    return { hasContradictions: false, contradictions: [] };
  }
}

// ── Get Lifecycle History ──

export async function getLifecycleHistory(
  env: Env,
  pageId: string,
): Promise<PageLifecycleEvent[]> {
  const rows = await env.DB.prepare(`
    SELECT * FROM wiki_lifecycle_events
    WHERE page_id = ?
    ORDER BY created_at DESC
  `).bind(pageId).all();

  return (rows.results as unknown as WikiLifecycleEventRow[]).map(row => ({
    pageId: row.page_id,
    fromState: row.from_state as WikiPageState,
    toState: row.to_state as WikiPageState,
    reason: row.reason || '',
    sourceHash: row.source_hash,
    timestamp: row.created_at,
  }));
}

// ── Auto Transition (Batch) ──

export async function autoTransition(env: Env): Promise<{
  checked: number;
  transitions: number;
  staleMarked: number;
  archivedCount: number;
}> {
  const now = new Date();
  const result = {
    checked: 0,
    transitions: 0,
    staleMarked: 0,
    archivedCount: 0,
  };

  // Get all active pages
  const activePages = await env.DB.prepare(
    "SELECT id, slug, title, state, sources, source_hashes, last_verified_at FROM wiki_pages WHERE state = 'active'"
  ).all();

  for (const row of activePages.results as unknown as WikiPageRow[]) {
    result.checked++;

    // Check age-based staleness
    const lastVerified = row.last_verified_at ? new Date(row.last_verified_at) : null;
    const ageHours = lastVerified
      ? (now.getTime() - lastVerified.getTime()) / (1000 * 60 * 60)
      : STALE_THRESHOLD_HOURS + 1;

    if (ageHours > STALE_THRESHOLD_HOURS) {
      const transitioned = await transitionState(
        env,
        row.id,
        'stale',
        `Auto-transition: page age (${Math.round(ageHours)}h) exceeds threshold (${STALE_THRESHOLD_HOURS}h)`,
      );
      if (transitioned) {
        result.transitions++;
        result.staleMarked++;
      }
    }
  }

  // Check for pages to archive (inactive for ARCHIVE_AFTER_DAYS)
  const archiveCutoff = new Date(now.getTime() - ARCHIVE_AFTER_DAYS * 24 * 60 * 60 * 1000).toISOString();

  const stalePages = await env.DB.prepare(`
    SELECT id FROM wiki_pages
    WHERE state IN ('stale', 'contradicted')
    AND updated_at < ?
  `).bind(archiveCutoff).all();

  for (const row of stalePages.results) {
    const transitioned = await transitionState(
      env,
      row.id as string,
      'archived',
      `Auto-archive: page inactive for more than ${ARCHIVE_AFTER_DAYS} days`,
    );
    if (transitioned) {
      result.transitions++;
      result.archivedCount++;
    }
  }

  return result;
}

// ── Should Reverify ──

export function shouldReverify(
  page: WikiPage,
  maxAgeHours: number = STALE_THRESHOLD_HOURS,
): boolean {
  if (!page.lastVerifiedAt) return true;

  const ageHours = (Date.now() - new Date(page.lastVerifiedAt).getTime()) / (1000 * 60 * 60);

  // High-access pages should be reverified more frequently
  const accessAdjustment = page.accessCount > 100 ? 0.5 : page.accessCount > 10 ? 0.75 : 1.0;

  return ageHours > maxAgeHours * accessAdjustment;
}

// ── Export Constants ──

export const WIKI_LIFECYCLE_CONFIG = {
  STALE_THRESHOLD_HOURS,
  CONTRADICTION_SIMILARITY_THRESHOLD,
  ARCHIVE_AFTER_DAYS,
  VALID_TRANSITIONS,
};
