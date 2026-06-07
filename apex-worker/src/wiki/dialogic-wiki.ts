/**
 * APEX 2.0 — Contradiction-as-Content Pattern (Dialogic Wiki)
 *
 * When sources disagree, APEX presents the disagreement AS content,
 * not flattened. Contradictions are first-class wiki content.
 */

import { Env } from '../types';
import { routeLLMCall } from '../llm-router';
import { generateUUID } from '../utils';
import {
  ContradictionPosition,
  ContradictionRecord,
  DialecticalSummary,
  WikiContradictionRow,
} from './types';

// ── Row Conversion ──

function rowToContradictionRecord(row: WikiContradictionRow): ContradictionRecord {
  return {
    id: row.id,
    topic: row.topic,
    positions: row.positions ? JSON.parse(row.positions) : [],
    severity: row.severity as ContradictionRecord['severity'],
    status: row.status as ContradictionRecord['status'],
    detectedAt: row.detected_at,
    resolvedAt: row.resolved_at,
  };
}

// ── Detect Contradictions for a Topic ──

export async function detectContradictions(
  env: Env,
  topic: string,
): Promise<ContradictionRecord[]> {
  // Find wiki pages related to this topic
  const relatedPages = await env.DB.prepare(`
    SELECT id, slug, title, state FROM wiki_pages
    WHERE title LIKE ? OR content_snippet LIKE ?
    AND state NOT IN ('archived', 'draft')
    LIMIT 10
  `).bind(`%${topic}%`, `%${topic}%`).all();

  if (!relatedPages.results || relatedPages.results.length < 2) {
    return [];
  }

  // Use LLM to find contradictions across pages
  const pageList = (relatedPages.results as any[]).map((p, i) =>
    `[Page ${i + 1}] ${p.title} (slug: ${p.slug}, state: ${p.state})`
  ).join('\n');

  // Load page content snippets
  const pageContents: string[] = [];
  for (const page of relatedPages.results as any[]) {
    const contentRow = await env.DB.prepare(
      'SELECT content_snippet FROM wiki_pages WHERE id = ?'
    ).bind(page.id).first();
    if (contentRow?.content_snippet) {
      pageContents.push(`[Page: ${page.title}]\n${(contentRow.content_snippet as string).slice(0, 1000)}`);
    }
  }

  const contradictionPrompt = `You are a contradiction analyst. Analyze these wiki pages about "${topic}" and identify any CONTRADICTING claims between them.

Pages:
${pageContents.join('\n\n')}

For each contradiction found, respond with a JSON array:
[{
  "topic": "specific sub-topic of disagreement",
  "positions": [
    {"pageId": "page-slug-or-id", "claim": "exact claim from page", "sources": [], "confidence": 0.0-1.0, "jurisdiction": "", "context": "surrounding context"},
    {"pageId": "page-slug-or-id", "claim": "contradicting claim", "sources": [], "confidence": 0.0-1.0, "jurisdiction": "", "context": "surrounding context"}
  ],
  "severity": "low|medium|high|critical"
}]

If no contradictions exist, respond with: []`;

  const result = await routeLLMCall(
    env,
    [{ role: 'user', content: contradictionPrompt }],
    2048, 0.0, undefined, false, false,
  );

  if (result.content.startsWith('[ALL_LLM_FAILED]')) {
    return [];
  }

  // Parse and store contradictions
  const contradictions: ContradictionRecord[] = [];

  try {
    let content = result.content.trim();
    const jsonMatch = content.match(/\[[\s\S]*\]/);
    if (!jsonMatch) return [];

    const parsed = JSON.parse(jsonMatch[0]);

    if (!Array.isArray(parsed)) return [];

    const now = new Date().toISOString();

    for (const item of parsed) {
      if (!item.topic || !item.positions || item.positions.length < 2) continue;

      const id = generateUUID();

      await env.DB.prepare(`
        INSERT INTO wiki_contradictions (id, topic, positions, severity, status, detected_at, created_at, updated_at)
        VALUES (?, ?, ?, ?, 'detected', ?, ?, ?)
      `).bind(
        id,
        item.topic,
        JSON.stringify(item.positions),
        item.severity || 'medium',
        now,
        now,
        now,
      ).run();

      contradictions.push({
        id,
        topic: item.topic,
        positions: item.positions,
        severity: item.severity || 'medium',
        status: 'detected',
        detectedAt: now,
        resolvedAt: null,
      });

      // Mark related wiki pages as contradicted
      for (const position of item.positions) {
        if (position.pageId) {
          const page = await env.DB.prepare(
            'SELECT id, state FROM wiki_pages WHERE slug = ? OR id = ?'
          ).bind(position.pageId, position.pageId).first();

          if (page && page.state === 'active') {
            // Transition page state to contradicted
            const { transitionState } = await import('./page-lifecycle');
            await transitionState(
              env,
              page.id as string,
              'contradicted',
              `Contradiction detected on topic: ${item.topic}`,
            );
          }
        }
      }
    }
  } catch {
    // Non-critical
  }

  return contradictions;
}

// ── Preserve Contradiction ──

export async function preserveContradiction(
  env: Env,
  record: ContradictionRecord,
): Promise<ContradictionRecord> {
  // Instead of resolving a contradiction, preserve it as first-class content
  await env.DB.prepare(`
    UPDATE wiki_contradictions
    SET status = 'preserved', updated_at = ?
    WHERE id = ?
  `).bind(new Date().toISOString(), record.id).run();

  // Create a wiki page that presents the contradiction
  const dialecticalContent = generateContradictionPage(record);

  const slug = `contradiction-${record.topic.toLowerCase().replace(/\s+/g, '-').slice(0, 80)}`;
  const pageId = generateUUID();
  const now = new Date().toISOString();
  const snippet = dialecticalContent.slice(0, 500);

  // Check if page already exists
  const existing = await env.DB.prepare(
    'SELECT id FROM wiki_pages WHERE slug = ?'
  ).bind(slug).first();

  if (existing) {
    // Update existing contradiction page
    await env.DB.prepare(`
      UPDATE wiki_pages
      SET content_snippet = ?, content_text = ?, updated_at = ?, state = 'active'
      WHERE slug = ?
    `).bind(snippet, dialecticalContent, now, slug).run();
  } else {
    // Create new contradiction page with content_text (replaces R2)
    await env.DB.prepare(`
      INSERT INTO wiki_pages (id, slug, title, content_snippet, content_text, state, category, source_hashes, sources, entities, links, metadata, created_at, updated_at, last_verified_at, verification_count, access_count, version)
      VALUES (?, ?, ?, ?, ?, 'active', 'contradiction', ?, ?, ?, ?, ?, ?, ?, ?, 1, 0, 1)
    `).bind(
      pageId,
      slug,
      `Contradiction: ${record.topic}`,
      snippet,
      dialecticalContent,
      JSON.stringify([]),
      JSON.stringify([]),
      JSON.stringify([]),
      JSON.stringify([{ targetSlug: record.topic, relationType: 'contradicts', context: 'Dialectical page' }]),
      JSON.stringify({ type: 'contradiction', recordId: record.id, severity: record.severity }),
      now,
      now,
      now,
    ).run();
  }

  return {
    ...record,
    status: 'preserved',
  };
}

// ── Generate Contradiction Page Content ──

function generateContradictionPage(record: ContradictionRecord): string {
  let content = `# Contradiction: ${record.topic}\n\n`;
  content += `> This page documents an active disagreement between sources. Both positions are preserved as first-class content.\n\n`;
  content += `**Severity**: ${record.severity}\n`;
  content += `**Status**: ${record.status}\n`;
  content += `**Detected**: ${record.detectedAt}\n\n`;

  content += `## Positions\n\n`;

  for (let i = 0; i < record.positions.length; i++) {
    const position = record.positions[i];
    content += `### Position ${i + 1}: ${position.pageId}\n\n`;
    content += `**Claim**: ${position.claim}\n\n`;
    content += `**Confidence**: ${position.confidence}\n\n`;
    if (position.jurisdiction) {
      content += `**Jurisdiction**: ${position.jurisdiction}\n\n`;
    }
    if (position.context) {
      content += `**Context**: ${position.context}\n\n`;
    }
    if (position.sources.length > 0) {
      content += `**Sources**: ${position.sources.join(', ')}\n\n`;
    }
  }

  content += `## Nature of the Disagreement\n\n`;
  content += `The disagreement centers on ${record.topic}. `;
  content += `The positions above represent different interpretations or findings from their respective sources.\n\n`;

  content += `---\n*This contradiction page was auto-generated by APEX 2.0. It will be updated as new evidence emerges.*\n`;

  return content;
}

// ── Get Contradiction Map ──

export async function getContradictionMap(
  env: Env,
  domain?: string,
): Promise<ContradictionRecord[]> {
  let query = 'SELECT * FROM wiki_contradictions WHERE status NOT IN (\'resolved\', \'superseded\')';
  const params: unknown[] = [];

  if (domain) {
    query += ' AND topic LIKE ?';
    params.push(`%${domain}%`);
  }

  query += ' ORDER BY detected_at DESC LIMIT 50';

  const rows = await env.DB.prepare(query).bind(...params).all();

  return (rows.results as unknown as WikiContradictionRow[]).map(rowToContradictionRecord);
}

// ── Analyze Contradiction ──

export async function analyzeContradiction(
  env: Env,
  recordId: string,
): Promise<{
  record: ContradictionRecord | null;
  analysis: string;
  stakes: string;
  resolutionPaths: string[];
}> {
  const row = await env.DB.prepare(
    'SELECT * FROM wiki_contradictions WHERE id = ?'
  ).bind(recordId).first() as WikiContradictionRow | null;

  if (!row) {
    return { record: null, analysis: '', stakes: '', resolutionPaths: [] };
  }

  const record = rowToContradictionRecord(row);

  // Update status to analyzing
  await env.DB.prepare(`
    UPDATE wiki_contradictions SET status = 'analyzing', updated_at = ? WHERE id = ?
  `).bind(new Date().toISOString(), recordId).run();

  // Use LLM for deep analysis
  const positionsText = record.positions.map((p, i) =>
    `Position ${i + 1} (${p.pageId}): ${p.claim}\n  Confidence: ${p.confidence}\n  Context: ${p.context}\n  Sources: ${p.sources.join(', ')}`
  ).join('\n\n');

  const analysisPrompt = `You are a dialectical analyst. Perform a deep analysis of this contradiction.

Topic: ${record.topic}
Severity: ${record.severity}

Positions:
${positionsText}

Analyze:
1. What is the NATURE of the disagreement? (factual, methodological, interpretive, jurisdictional)
2. What are the STAKES? What depends on resolving this?
3. What would RESOLVE it? Suggest 2-3 possible paths to resolution.

Respond as JSON:
{
  "analysis": "detailed analysis of the nature of disagreement",
  "stakes": "what depends on resolving this",
  "resolutionPaths": ["path 1", "path 2", "path 3"]
}`;

  const result = await routeLLMCall(
    env,
    [{ role: 'user', content: analysisPrompt }],
    1024, 0.0, undefined, false, false,
  );

  if (result.content.startsWith('[ALL_LLM_FAILED]')) {
    return { record, analysis: 'Analysis failed — LLM unavailable', stakes: 'Unknown', resolutionPaths: [] };
  }

  try {
    let content = result.content.trim();
    const jsonMatch = content.match(/\{[\s\S]*\}/);
    if (jsonMatch) {
      const parsed = JSON.parse(jsonMatch[0]);
      return {
        record,
        analysis: parsed.analysis || '',
        stakes: parsed.stakes || '',
        resolutionPaths: parsed.resolutionPaths || [],
      };
    }
  } catch {
    // Fall through to raw text
  }

  return {
    record,
    analysis: result.content,
    stakes: 'Could not parse structured analysis',
    resolutionPaths: [],
  };
}

// ── Track Contradiction Evolution ──

export async function trackContradictionEvolution(
  env: Env,
  recordId: string,
): Promise<{
  record: ContradictionRecord | null;
  evolution: Array<{
    timestamp: string;
    event: string;
    details: string;
  }>;
}> {
  const row = await env.DB.prepare(
    'SELECT * FROM wiki_contradictions WHERE id = ?'
  ).bind(recordId).first() as WikiContradictionRow | null;

  if (!row) {
    return { record: null, evolution: [] };
  }

  const record = rowToContradictionRecord(row);

  // Build evolution timeline from lifecycle events related to involved pages
  const evolution: Array<{ timestamp: string; event: string; details: string }> = [];

  evolution.push({
    timestamp: record.detectedAt,
    event: 'Contradiction detected',
    details: `Found ${record.positions.length} conflicting positions on topic: ${record.topic}`,
  });

  // Check for state changes on involved pages
  for (const position of record.positions) {
    const pageId = position.pageId;
    const lifecycleEvents = await env.DB.prepare(`
      SELECT * FROM wiki_lifecycle_events
      WHERE page_id = ? OR page_id = (SELECT id FROM wiki_pages WHERE slug = ?)
      ORDER BY created_at ASC
    `).bind(pageId, pageId).all();

    for (const event of lifecycleEvents.results as any[]) {
      evolution.push({
        timestamp: event.created_at,
        event: `Page ${pageId}: ${event.from_state} → ${event.to_state}`,
        details: event.reason || 'State transition',
      });
    }
  }

  if (record.resolvedAt) {
    evolution.push({
      timestamp: record.resolvedAt,
      event: 'Contradiction resolved',
      details: `Status changed to ${record.status}`,
    });
  }

  // Sort by timestamp
  evolution.sort((a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime());

  return { record, evolution };
}

// ── Generate Dialectical Summary ──

export async function generateDialecticalSummary(
  env: Env,
  topic: string,
): Promise<DialecticalSummary> {
  // Get all active contradictions for this topic
  const contradictions = await getContradictionMap(env, topic);

  // Also get wiki pages about this topic
  const pages = await env.DB.prepare(`
    SELECT slug, title, content_snippet, state FROM wiki_pages
    WHERE (title LIKE ? OR content_snippet LIKE ?)
    AND state NOT IN ('archived', 'draft')
    LIMIT 10
  `).bind(`%${topic}%`, `%${topic}%`).all();

  const pageSummaries = (pages.results as any[]).map(p =>
    `[${p.title}] (${p.state}): ${(p.content_snippet || '').slice(0, 300)}`
  ).join('\n\n');

  const contradictionSummaries = contradictions.map(c =>
    `Contradiction: ${c.topic} (${c.severity})\n${c.positions.map((p, i) => `  Position ${i + 1}: ${p.claim}`).join('\n')}`
  ).join('\n\n');

  const dialecticPrompt = `Generate a dialectical summary for: "${topic}"

This is NOT a standard summary that picks one "right answer." Instead, present ALL positions, their evidence, and the nature of disagreements.

Wiki Pages:
${pageSummaries}

Active Contradictions:
${contradictionSummaries || 'No active contradictions detected.'}

Respond with JSON:
{
  "topic": "the topic",
  "positions": [
    {"label": "Position label", "claim": "what this position holds", "evidence": ["evidence 1", "evidence 2"], "confidence": 0.0-1.0}
  ],
  "disagreement": "nature of the disagreement between positions",
  "stakes": "what depends on resolving this",
  "resolutionPaths": ["possible way to resolve", "another approach"]
}`;

  const result = await routeLLMCall(
    env,
    [{ role: 'user', content: dialecticPrompt }],
    2048, 0.0, undefined, false, false,
  );

  if (result.content.startsWith('[ALL_LLM_FAILED]')) {
    return {
      topic,
      positions: [],
      disagreement: 'Could not generate dialectical summary — LLM unavailable',
      stakes: 'Unknown',
      resolutionPaths: [],
    };
  }

  try {
    let content = result.content.trim();
    const jsonMatch = content.match(/\{[\s\S]*\}/);
    if (jsonMatch) {
      const parsed = JSON.parse(jsonMatch[0]);
      return {
        topic: parsed.topic || topic,
        positions: parsed.positions || [],
        disagreement: parsed.disagreement || '',
        stakes: parsed.stakes || '',
        resolutionPaths: parsed.resolutionPaths || [],
      };
    }
  } catch {
    // Fall through
  }

  return {
    topic,
    positions: [],
    disagreement: result.content,
    stakes: 'Could not parse structured analysis',
    resolutionPaths: [],
  };
}
