/**
 * APEX 2.0 — AutoSci's Two-Tier Memory Architecture
 *
 * Structured knowledge memory with typed entities and relations.
 * Builds a knowledge graph that compounds over time.
 */

import { Env } from '../types';
import { routeLLMCall } from '../llm-router';
import { generateUUID } from '../utils';
import {
  KnowledgeEntity,
  KnowledgeEntityType,
  KnowledgeRelation,
  KnowledgeRelationType,
  KnowledgeSubgraph,
  WikiEntityRow,
  WikiRelationRow,
} from './types';

// ── Valid Entity and Relation Types ──

export const ENTITY_TYPES: KnowledgeEntityType[] = [
  'Topic', 'Paper', 'Company', 'Person', 'Technology',
  'Market', 'Concept', 'Method', 'Event', 'Location',
];

export const RELATION_TYPES: KnowledgeRelationType[] = [
  'relates_to', 'cites', 'authored_by', 'competes_with',
  'precedes', 'extends', 'contradicts', 'supports', 'uses', 'part_of',
];

// ── Row Conversion ──

function rowToKnowledgeEntity(row: WikiEntityRow): KnowledgeEntity {
  return {
    id: row.id,
    name: row.name,
    type: row.type as KnowledgeEntityType,
    description: row.description || '',
    mentionCount: row.mention_count,
    firstSeen: row.first_seen,
    lastSeen: row.last_seen,
    properties: row.properties ? JSON.parse(row.properties) : {},
  };
}

function rowToKnowledgeRelation(row: WikiRelationRow): KnowledgeRelation {
  return {
    id: row.id,
    fromEntityId: row.from_entity_id,
    toEntityId: row.to_entity_id,
    relationType: row.relation_type as KnowledgeRelationType,
    context: row.context || '',
  };
}

// ── Get or Create Entity ──

export async function getOrCreateEntity(
  env: Env,
  name: string,
  type: KnowledgeEntityType,
): Promise<KnowledgeEntity> {
  const existing = await env.DB.prepare(
    'SELECT * FROM wiki_entities WHERE name = ? AND type = ?'
  ).bind(name, type).first() as WikiEntityRow | null;

  if (existing) {
    // Update mention count and last_seen
    const now = new Date().toISOString();
    await env.DB.prepare(`
      UPDATE wiki_entities SET mention_count = mention_count + 1, last_seen = ?, updated_at = ? WHERE id = ?
    `).bind(now, now, existing.id).run();

    return {
      ...rowToKnowledgeEntity(existing),
      mentionCount: existing.mention_count + 1,
      lastSeen: now,
    };
  }

  // Create new entity
  const id = generateUUID();
  const now = new Date().toISOString();
  await env.DB.prepare(`
    INSERT INTO wiki_entities (id, name, type, description, mention_count, first_seen, last_seen, properties, created_at, updated_at)
    VALUES (?, ?, ?, ?, 1, ?, ?, ?, ?, ?)
  `).bind(
    id,
    name,
    type,
    '',
    now,
    now,
    JSON.stringify({}),
    now,
    now,
  ).run();

  return {
    id,
    name,
    type,
    description: '',
    mentionCount: 1,
    firstSeen: now,
    lastSeen: now,
    properties: {},
  };
}

// ── Get Entity by Name ──

export async function getEntityByName(
  env: Env,
  name: string,
  type?: KnowledgeEntityType,
): Promise<KnowledgeEntity | null> {
  let query = 'SELECT * FROM wiki_entities WHERE name = ?';
  const params: unknown[] = [name];

  if (type) {
    query += ' AND type = ?';
    params.push(type);
  }

  query += ' LIMIT 1';

  const row = await env.DB.prepare(query).bind(...params).first() as WikiEntityRow | null;

  if (!row) return null;

  return rowToKnowledgeEntity(row);
}

// ── Link Entities ──

export async function linkEntities(
  env: Env,
  fromId: string,
  toId: string,
  relationType: KnowledgeRelationType,
  context: string = '',
): Promise<KnowledgeRelation> {
  // Check if relation already exists
  const existing = await env.DB.prepare(
    'SELECT id FROM wiki_relations WHERE from_entity_id = ? AND to_entity_id = ? AND relation_type = ?'
  ).bind(fromId, toId, relationType).first();

  if (existing) {
    // Update context
    const now = new Date().toISOString();
    await env.DB.prepare('UPDATE wiki_relations SET context = ?, updated_at = ? WHERE id = ?')
      .bind(context, now, existing.id).run();

    return {
      id: existing.id as string,
      fromEntityId: fromId,
      toEntityId: toId,
      relationType,
      context,
    };
  }

  // Create new relation
  const id = generateUUID();
  const now = new Date().toISOString();
  await env.DB.prepare(`
    INSERT INTO wiki_relations (id, from_entity_id, to_entity_id, relation_type, context, created_at, updated_at)
    VALUES (?, ?, ?, ?, ?, ?, ?)
  `).bind(id, fromId, toId, relationType, context, now, now).run();

  return {
    id,
    fromEntityId: fromId,
    toEntityId: toId,
    relationType,
    context,
  };
}

// ── Query Knowledge Graph ──

export async function queryKnowledgeGraph(
  env: Env,
  entityId: string,
  depth: number = 2,
): Promise<KnowledgeSubgraph> {
  const entities: KnowledgeEntity[] = [];
  const relations: KnowledgeRelation[] = [];
  const visitedEntityIds = new Set<string>();

  // BFS traversal
  const queue: Array<{ id: string; currentDepth: number }> = [{ id: entityId, currentDepth: 0 }];
  visitedEntityIds.add(entityId);

  while (queue.length > 0) {
    const { id, currentDepth } = queue.shift()!;

    if (currentDepth > depth) continue;

    // Get entity
    const entityRow = await env.DB.prepare(
      'SELECT * FROM wiki_entities WHERE id = ?'
    ).bind(id).first() as WikiEntityRow | null;

    if (entityRow) {
      entities.push(rowToKnowledgeEntity(entityRow));
    }

    if (currentDepth >= depth) continue;

    // Get outgoing relations
    const outgoingRows = await env.DB.prepare(
      'SELECT * FROM wiki_relations WHERE from_entity_id = ?'
    ).bind(id).all();

    for (const row of outgoingRows.results as unknown as WikiRelationRow[]) {
      const relation = rowToKnowledgeRelation(row);
      relations.push(relation);

      if (!visitedEntityIds.has(relation.toEntityId)) {
        visitedEntityIds.add(relation.toEntityId);
        queue.push({ id: relation.toEntityId, currentDepth: currentDepth + 1 });
      }
    }

    // Get incoming relations
    const incomingRows = await env.DB.prepare(
      'SELECT * FROM wiki_relations WHERE to_entity_id = ?'
    ).bind(id).all();

    for (const row of incomingRows.results as unknown as WikiRelationRow[]) {
      const relation = rowToKnowledgeRelation(row);
      relations.push(relation);

      if (!visitedEntityIds.has(relation.fromEntityId)) {
        visitedEntityIds.add(relation.fromEntityId);
        queue.push({ id: relation.fromEntityId, currentDepth: currentDepth + 1 });
      }
    }
  }

  return { entities, relations };
}

// ── Find Connections Between Two Entities ──

export async function findConnections(
  env: Env,
  entityAId: string,
  entityBId: string,
  maxDepth: number = 4,
): Promise<KnowledgeSubgraph | null> {
  // Bidirectional BFS
  const forwardVisited = new Map<string, string>(); // entityId -> parent relation id
  const backwardVisited = new Map<string, string>();

  const forwardQueue: Array<{ id: string; depth: number }> = [{ id: entityAId, depth: 0 }];
  const backwardQueue: Array<{ id: string; depth: number }> = [{ id: entityBId, depth: 0 }];
  forwardVisited.set(entityAId, '');
  backwardVisited.set(entityBId, '');

  let meetingPoint: string | null = null;

  while (forwardQueue.length > 0 || backwardQueue.length > 0) {
    // Expand forward
    if (forwardQueue.length > 0) {
      const { id, depth } = forwardQueue.shift()!;
      if (depth >= maxDepth) continue;

      const outgoing = await env.DB.prepare(
        'SELECT * FROM wiki_relations WHERE from_entity_id = ?'
      ).bind(id).all();

      for (const row of outgoing.results as unknown as WikiRelationRow[]) {
        const targetId = row.to_entity_id;
        if (!forwardVisited.has(targetId)) {
          forwardVisited.set(targetId, row.id);
          forwardQueue.push({ id: targetId, depth: depth + 1 });

          if (backwardVisited.has(targetId)) {
            meetingPoint = targetId;
            break;
          }
        }
      }

      if (meetingPoint) break;
    }

    // Expand backward
    if (backwardQueue.length > 0) {
      const { id, depth } = backwardQueue.shift()!;
      if (depth >= maxDepth) continue;

      const incoming = await env.DB.prepare(
        'SELECT * FROM wiki_relations WHERE to_entity_id = ?'
      ).bind(id).all();

      for (const row of incoming.results as unknown as WikiRelationRow[]) {
        const sourceId = row.from_entity_id;
        if (!backwardVisited.has(sourceId)) {
          backwardVisited.set(sourceId, row.id);
          backwardQueue.push({ id: sourceId, depth: depth + 1 });

          if (forwardVisited.has(sourceId)) {
            meetingPoint = sourceId;
            break;
          }
        }
      }

      if (meetingPoint) break;
    }
  }

  if (!meetingPoint) return null;

  // Reconstruct path and collect entities/relations
  const pathEntityIds = new Set<string>();
  const pathRelationIds = new Set<string>();

  // Trace forward path from A to meeting point
  let currentId: string | null = meetingPoint;
  while (currentId && currentId !== entityAId) {
    pathEntityIds.add(currentId);
    const relationId = forwardVisited.get(currentId);
    if (relationId) pathRelationIds.add(relationId);
    // We need to find the entity that leads to currentId in forward direction
    // This is simplified — in production, store parent entity ID too
    break; // Simplified: just include the meeting point
  }
  pathEntityIds.add(entityAId);
  pathEntityIds.add(entityBId);
  pathEntityIds.add(meetingPoint);

  // Load entities and relations
  const entities: KnowledgeEntity[] = [];
  const relations: KnowledgeRelation[] = [];

  for (const eid of pathEntityIds) {
    const row = await env.DB.prepare('SELECT * FROM wiki_entities WHERE id = ?').bind(eid).first() as WikiEntityRow | null;
    if (row) entities.push(rowToKnowledgeEntity(row));
  }

  for (const rid of pathRelationIds) {
    const row = await env.DB.prepare('SELECT * FROM wiki_relations WHERE id = ?').bind(rid).first() as WikiRelationRow | null;
    if (row) relations.push(rowToKnowledgeRelation(row));
  }

  // Also get direct relations between entities in the path
  for (const eidA of pathEntityIds) {
    for (const eidB of pathEntityIds) {
      if (eidA === eidB) continue;
      const directRelations = await env.DB.prepare(
        'SELECT * FROM wiki_relations WHERE from_entity_id = ? AND to_entity_id = ?'
      ).bind(eidA, eidB).all();

      for (const row of directRelations.results as unknown as WikiRelationRow[]) {
        if (!pathRelationIds.has(row.id)) {
          pathRelationIds.add(row.id);
          relations.push(rowToKnowledgeRelation(row));
        }
      }
    }
  }

  return { entities, relations };
}

// ── Merge Research Into Memory ──

export async function mergeResearchIntoMemory(
  env: Env,
  sources: Array<{ url: string; title: string; snippet: string; tier: string }>,
  claims: Array<{ statement: string; confidence: number }>,
): Promise<{ entitiesCreated: number; relationsCreated: number }> {
  let entitiesCreated = 0;
  let relationsCreated = 0;

  // Use LLM to extract entities and relations from claims
  const claimsText = claims.slice(0, 15).map((c, i) => `[${i + 1}] ${c.statement} (confidence: ${c.confidence})`).join('\n');
  const sourcesText = sources.slice(0, 10).map((s, i) => `[${i + 1}] ${s.title} (${s.tier})`).join('\n');

  const extractionPrompt = `Extract entities and relations from these research claims.

Sources:
${sourcesText}

Claims:
${claimsText}

Respond with a JSON object:
{
  "entities": [{"name": "Entity Name", "type": "Topic|Paper|Company|Person|Technology|Market|Concept|Method|Event|Location"}],
  "relations": [{"from": "Entity A", "to": "Entity B", "type": "relates_to|cites|authored_by|competes_with|precedes|extends|contradicts|supports|uses|part_of", "context": "brief context"}]
}`;

  const result = await routeLLMCall(
    env,
    [{ role: 'user', content: extractionPrompt }],
    2048, 0.0, undefined, false, false,
  );

  if (result.content.startsWith('[ALL_LLM_FAILED]')) {
    return { entitiesCreated, relationsCreated };
  }

  try {
    let content = result.content.trim();
    const jsonMatch = content.match(/\{[\s\S]*\}/);
    if (!jsonMatch) return { entitiesCreated, relationsCreated };

    const parsed = JSON.parse(jsonMatch[0]);
    const extractedEntities: Array<{ name: string; type: KnowledgeEntityType }> = parsed.entities || [];
    const extractedRelations: Array<{ from: string; to: string; type: KnowledgeRelationType; context: string }> = parsed.relations || [];

    // Create entities
    const entityMap = new Map<string, KnowledgeEntity>();
    for (const entityDef of extractedEntities) {
      const entityType = ENTITY_TYPES.includes(entityDef.type) ? entityDef.type : 'Concept';
      const entity = await getOrCreateEntity(env, entityDef.name, entityType);
      entityMap.set(entityDef.name, entity);
      entitiesCreated++;
    }

    // Create relations
    for (const relDef of extractedRelations) {
      const fromEntity = entityMap.get(relDef.from);
      const toEntity = entityMap.get(relDef.to);

      if (fromEntity && toEntity) {
        const relationType = RELATION_TYPES.includes(relDef.type) ? relDef.type : 'relates_to';
        await linkEntities(env, fromEntity.id, toEntity.id, relationType, relDef.context || '');
        relationsCreated++;
      }
    }
  } catch {
    // Non-critical — extraction failure shouldn't block the pipeline
  }

  return { entitiesCreated, relationsCreated };
}

// ── Get Memory Evolution Suggestions ──

export async function getMemoryEvolutionSuggestions(env: Env): Promise<string[]> {
  // Get high-mention entities with few connections
  const sparseEntities = await env.DB.prepare(`
    SELECT e.id, e.name, e.type, e.mention_count,
           (SELECT COUNT(*) FROM wiki_relations WHERE from_entity_id = e.id OR to_entity_id = e.id) as relation_count
    FROM wiki_entities e
    WHERE e.mention_count > 3
    ORDER BY relation_count ASC, e.mention_count DESC
    LIMIT 10
  `).all();

  if (!sparseEntities.results || sparseEntities.results.length === 0) {
    return [];
  }

  const entityList = (sparseEntities.results as any[]).map(e =>
    `${e.name} (${e.type}): ${e.mention_count} mentions, ${e.relation_count} relations`
  ).join('\n');

  const suggestionPrompt = `You are a knowledge graph analyst. Based on these entities that have many mentions but few connections, suggest potential relationships or areas to investigate.

Entities:
${entityList}

For each, suggest 1-2 potential connections or contradictions to investigate.
Respond with a JSON array of suggestion strings.`;

  const result = await routeLLMCall(
    env,
    [{ role: 'user', content: suggestionPrompt }],
    512, 0.0, undefined, false, false,
  );

  if (result.content.startsWith('[ALL_LLM_FAILED]')) {
    return [];
  }

  try {
    let content = result.content.trim();
    const jsonMatch = content.match(/\[[\s\S]*\]/);
    if (!jsonMatch) return [];
    return JSON.parse(jsonMatch[0]);
  } catch {
    // Return raw suggestions split by newlines
    return result.content.split('\n').filter(l => l.trim().length > 10).slice(0, 10);
  }
}

// ── Export Graph ──

export async function exportGraph(
  env: Env,
  format: 'json' | 'graphml' | 'd3' = 'json',
): Promise<string> {
  const entityRows = await env.DB.prepare('SELECT * FROM wiki_entities ORDER BY mention_count DESC LIMIT 500').all();
  const relationRows = await env.DB.prepare('SELECT * FROM wiki_relations LIMIT 1000').all();

  const entities = (entityRows.results as unknown as WikiEntityRow[]).map(rowToKnowledgeEntity);
  const relations = (relationRows.results as unknown as WikiRelationRow[]).map(rowToKnowledgeRelation);

  switch (format) {
    case 'd3': {
      const d3Format = {
        nodes: entities.map(e => ({
          id: e.id,
          name: e.name,
          type: e.type,
          mentionCount: e.mentionCount,
        })),
        links: relations.map(r => ({
          source: r.fromEntityId,
          target: r.toEntityId,
          type: r.relationType,
          context: r.context,
        })),
      };
      return JSON.stringify(d3Format, null, 2);
    }

    case 'graphml': {
      let graphml = '<?xml version="1.0" encoding="UTF-8"?>\n';
      graphml += '<graphml xmlns="http://graphml.graphstruct.org/xmlns">\n';
      graphml += '  <graph id="KnowledgeGraph" edgedefault="directed">\n';

      for (const entity of entities) {
        graphml += `    <node id="${escapeXml(entity.id)}">\n`;
        graphml += `      <data key="name">${escapeXml(entity.name)}</data>\n`;
        graphml += `      <data key="type">${escapeXml(entity.type)}</data>\n`;
        graphml += `      <data key="mentions">${entity.mentionCount}</data>\n`;
        graphml += `    </node>\n`;
      }

      for (const relation of relations) {
        graphml += `    <edge source="${escapeXml(relation.fromEntityId)}" target="${escapeXml(relation.toEntityId)}">\n`;
        graphml += `      <data key="type">${escapeXml(relation.relationType)}</data>\n`;
        graphml += `      <data key="context">${escapeXml(relation.context)}</data>\n`;
        graphml += `    </edge>\n`;
      }

      graphml += '  </graph>\n';
      graphml += '</graphml>';
      return graphml;
    }

    case 'json':
    default: {
      return JSON.stringify({ entities, relations }, null, 2);
    }
  }
}

// ── XML Escaping ──

function escapeXml(str: string): string {
  return str
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&apos;');
}
