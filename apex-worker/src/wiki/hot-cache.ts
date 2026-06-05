/**
 * APEX 2.0 — Session Hot Cache (Claude-Obsidian Pattern)
 *
 * Per-session hot cache that tracks recent queries, topics, and sources.
 * Enables session resumption: "Welcome back. Last time you were researching X..."
 */

import { Env } from '../types';
import { routeLLMCall } from '../llm-router';
import { generateUUID } from '../utils';
import { HotCacheEntry, WikiSessionRow } from './types';

// ── Configuration ──

const MAX_RECENT_TOPICS = 10;
const MAX_RECENT_SOURCES = 20;
const MAX_CONTEXT_LENGTH = 2000;
const MAX_SUMMARY_LENGTH = 1000;

// ── Row to HotCacheEntry Conversion ──

function rowToHotCacheEntry(row: WikiSessionRow): HotCacheEntry {
  return {
    sessionId: row.id,
    userId: row.user_id || '',
    lastQuery: row.last_query || '',
    lastContext: row.last_context || '',
    recentTopics: row.recent_topics ? JSON.parse(row.recent_topics) : [],
    recentSources: row.recent_sources ? JSON.parse(row.recent_sources) : [],
    sessionSummary: row.session_summary || '',
    updatedAt: row.updated_at,
  };
}

// ── Create Session ──

export async function createSession(
  env: Env,
  userId: string,
): Promise<HotCacheEntry> {
  const id = generateUUID();
  const now = new Date().toISOString();

  await env.DB.prepare(`
    INSERT INTO wiki_sessions (id, user_id, last_query, last_context, recent_topics, recent_sources, session_summary, created_at, updated_at)
    VALUES (?, ?, '', '', ?, ?, '', ?, ?)
  `).bind(
    id,
    userId,
    JSON.stringify([]),
    JSON.stringify([]),
    now,
    now,
  ).run();

  return {
    sessionId: id,
    userId,
    lastQuery: '',
    lastContext: '',
    recentTopics: [],
    recentSources: [],
    sessionSummary: '',
    updatedAt: now,
  };
}

// ── Get Session ──

export async function getSession(
  env: Env,
  sessionId: string,
): Promise<HotCacheEntry | null> {
  const row = await env.DB.prepare(
    'SELECT * FROM wiki_sessions WHERE id = ?'
  ).bind(sessionId).first() as WikiSessionRow | null;

  if (!row) return null;

  return rowToHotCacheEntry(row);
}

// ── Update Hot Cache ──

export async function updateHotCache(
  env: Env,
  sessionId: string,
  query: string,
  context: string,
  sources: string[],
): Promise<HotCacheEntry | null> {
  const existing = await getSession(env, sessionId);
  if (!existing) return null;

  const now = new Date().toISOString();

  // Update recent topics (extract topics from query)
  const newTopics = extractTopicsFromQuery(query);
  const recentTopics = [...new Set([...newTopics, ...existing.recentTopics])]
    .slice(0, MAX_RECENT_TOPICS);

  // Update recent sources
  const recentSources = [...new Set([...sources, ...existing.recentSources])]
    .slice(0, MAX_RECENT_SOURCES);

  // Truncate context
  const lastContext = context.slice(0, MAX_CONTEXT_LENGTH);
  const lastQuery = query.slice(0, 500);

  // Generate updated session summary using LLM
  let sessionSummary = existing.sessionSummary;
  try {
    const summaryPrompt = `You are a session tracker. Update the research session summary based on the new query and context.

Previous summary: ${sessionSummary || 'No previous summary.'}

New query: ${query}
New context snippet: ${lastContext.slice(0, 500)}
Recent topics: ${recentTopics.join(', ')}

Write a 1-2 sentence summary of what this user has been researching across this session.`;

    const summaryResult = await routeLLMCall(
      env,
      [{ role: 'user', content: summaryPrompt }],
      100, 0.0, undefined, false, true, // isClassification for cheap model
    );

    if (!summaryResult.content.startsWith('[ALL_LLM_FAILED]')) {
      sessionSummary = summaryResult.content.slice(0, MAX_SUMMARY_LENGTH);
    }
  } catch {
    // Keep existing summary on failure
  }

  await env.DB.prepare(`
    UPDATE wiki_sessions
    SET last_query = ?, last_context = ?, recent_topics = ?, recent_sources = ?,
        session_summary = ?, updated_at = ?
    WHERE id = ?
  `).bind(
    lastQuery,
    lastContext,
    JSON.stringify(recentTopics),
    JSON.stringify(recentSources),
    sessionSummary,
    now,
    sessionId,
  ).run();

  return getSession(env, sessionId);
}

// ── Get Resumption Context ──

export async function getResumptionContext(
  env: Env,
  sessionId: string,
): Promise<string> {
  const session = await getSession(env, sessionId);
  if (!session) return '';

  const timeSinceUpdate = Date.now() - new Date(session.updatedAt).getTime();
  const hoursSince = Math.round(timeSinceUpdate / (1000 * 60 * 60));
  const minutesSince = Math.round(timeSinceUpdate / (1000 * 60));

  let timeAgo: string;
  if (minutesSince < 60) {
    timeAgo = `${minutesSince} minute${minutesSince !== 1 ? 's' : ''} ago`;
  } else if (hoursSince < 24) {
    timeAgo = `${hoursSince} hour${hoursSince !== 1 ? 's' : ''} ago`;
  } else {
    const daysSince = Math.round(hoursSince / 24);
    timeAgo = `${daysSince} day${daysSince !== 1 ? 's' : ''} ago`;
  }

  const topicsStr = session.recentTopics.length > 0
    ? `Topics explored: ${session.recentTopics.slice(0, 5).join(', ')}.`
    : '';

  const sourcesStr = session.recentSources.length > 0
    ? `${session.recentSources.length} sources consulted.`
    : '';

  return `Welcome back. Last time you were researching: ${session.sessionSummary || session.lastQuery}. ${timeAgo}. ${topicsStr} ${sourcesStr} Here's where we left off...`;
}

// ── End Session ──

export async function endSession(
  env: Env,
  sessionId: string,
): Promise<boolean> {
  const session = await getSession(env, sessionId);
  if (!session) return false;

  // Final update to summary
  try {
    const finalSummaryPrompt = `Summarize this research session in 2-3 sentences.

Queries asked: ${session.lastQuery}
Topics explored: ${session.recentTopics.join(', ')}
Sources consulted: ${session.recentSources.length}

Write a concise summary of the user's research session.`;

    const summaryResult = await routeLLMCall(
      env,
      [{ role: 'user', content: finalSummaryPrompt }],
      100, 0.0, undefined, false, true,
    );

    if (!summaryResult.content.startsWith('[ALL_LLM_FAILED]')) {
      await env.DB.prepare(`
        UPDATE wiki_sessions SET session_summary = ?, updated_at = ? WHERE id = ?
      `).bind(
        summaryResult.content.slice(0, MAX_SUMMARY_LENGTH),
        new Date().toISOString(),
        sessionId,
      ).run();
    }
  } catch {
    // Non-critical
  }

  return true;
}

// ── Prune Stale Sessions ──

export async function pruneStaleSessions(
  env: Env,
  maxAgeHours: number = 72,
): Promise<number> {
  const cutoff = new Date(Date.now() - maxAgeHours * 60 * 60 * 1000).toISOString();

  const result = await env.DB.prepare(
    'DELETE FROM wiki_sessions WHERE updated_at < ?'
  ).bind(cutoff).run();

  return result.meta?.changes || 0;
}

// ── Extract Topics from Query ──

function extractTopicsFromQuery(query: string): string[] {
  const topics: string[] = [];

  // Simple topic extraction: key noun phrases
  // Remove common question words and stop words
  const stopWords = new Set([
    'what', 'how', 'why', 'when', 'where', 'who', 'which', 'is', 'are',
    'the', 'a', 'an', 'of', 'in', 'on', 'at', 'to', 'for', 'with', 'by',
    'does', 'do', 'did', 'can', 'could', 'would', 'should', 'will', 'shall',
    'about', 'between', 'from', 'into', 'through', 'during', 'before', 'after',
    'and', 'or', 'but', 'not', 'no', 'nor', 'so', 'yet', 'both', 'either',
    'this', 'that', 'these', 'those', 'it', 'its', 'my', 'your', 'their',
  ]);

  const words = query.toLowerCase().split(/\s+/);
  const meaningfulWords = words.filter(w => w.length > 2 && !stopWords.has(w));

  // Use the whole query as one topic if it's short enough
  if (query.length <= 60) {
    topics.push(query.trim());
  }

  // Extract bigrams (2-word phrases)
  for (let i = 0; i < meaningfulWords.length - 1; i++) {
    topics.push(`${meaningfulWords[i]} ${meaningfulWords[i + 1]}`);
  }

  // Individual meaningful words as topics (only if not already covered)
  for (const word of meaningfulWords) {
    const alreadyCovered = topics.some(t => t.includes(word));
    if (!alreadyCovered && word.length > 3) {
      topics.push(word);
    }
  }

  return topics.slice(0, 5);
}
