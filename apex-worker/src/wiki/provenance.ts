/**
 * APEX 2.0 — Dense-Mem's Provenance + Conflict Detection
 *
 * Claim-level provenance tracking: where each claim came from,
 * what it cost to produce, what supports or conflicts it.
 */

import { Env } from '../types';
import { routeLLMCall } from '../llm-router';
import { generateUUID } from '../utils';
import {
  ProvenanceClaim,
  ProvenanceAuditReport,
  WikiProvenanceClaimRow,
} from './types';

// ── Row Conversion ──

function rowToProvenanceClaim(row: WikiProvenanceClaimRow): ProvenanceClaim {
  return {
    id: row.id,
    statement: row.statement,
    sourceUrl: row.source_url,
    sourceTier: row.source_tier || '',
    confidence: row.confidence,
    extractionMethod: row.extraction_method,
    extractedAt: row.extracted_at,
    costToProduce: row.cost_to_produce,
    verificationStatus: row.verification_status as ProvenanceClaim['verificationStatus'],
    supportingClaims: row.supporting_claim_ids ? JSON.parse(row.supporting_claim_ids) : [],
    conflictingClaims: row.conflicting_claim_ids ? JSON.parse(row.conflicting_claim_ids) : [],
  };
}

// ── Extract and Track Claims from Sources ──

export async function extractAndTrack(
  env: Env,
  sources: Array<{
    url: string;
    title: string;
    content: string;
    tier: string;
  }>,
  wikiPageId: string,
): Promise<{ claimsExtracted: number; claimsCreated: number; errors: string[] }> {
  const result = {
    claimsExtracted: 0,
    claimsCreated: 0,
    errors: [] as string[],
  };

  for (const source of sources) {
    try {
      // Use LLM to extract claims from source content
      const truncatedContent = source.content.slice(0, 4000);

      const extractionPrompt = `Extract factual claims from this source. Each claim should be a single, testable statement.

Source: ${source.title} (${source.url})
Tier: ${source.tier}

Content:
${truncatedContent}

Respond with a JSON array of claims:
[{"statement": "exact claim text", "confidence": 0.0-1.0, "method": "direct_quote|inference|summary"}]

Rules:
- Each claim must be a single testable statement
- Include exact numbers, dates, and proper nouns
- Assign confidence based on source tier (P1: 0.8+, P2: 0.6+, P3: 0.3+, UNV: 0.1+)
- Mark extraction method: direct_quote if verbatim, inference if deduced, summary if generalized`;

      const llmResult = await routeLLMCall(
        env,
        [{ role: 'user', content: extractionPrompt }],
        2048, 0.0, undefined, false, false,
      );

      if (llmResult.content.startsWith('[ALL_LLM_FAILED]')) {
        result.errors.push(`LLM extraction failed for ${source.url}`);
        continue;
      }

      // Parse claims
      let claims: Array<{ statement: string; confidence: number; method: string }> = [];
      try {
        let content = llmResult.content.trim();
        const jsonMatch = content.match(/\[[\s\S]*\]/);
        if (jsonMatch) {
          claims = JSON.parse(jsonMatch[0]);
        }
      } catch {
        result.errors.push(`Could not parse claims from LLM output for ${source.url}`);
        continue;
      }

      if (!Array.isArray(claims)) continue;

      result.claimsExtracted += claims.length;

      // Store each claim
      const now = new Date().toISOString();
      for (const claim of claims.slice(0, 30)) {
        if (!claim.statement || claim.statement.length < 10) continue;

        const claimId = generateUUID();
        const costToProduce = estimateCostToProduce(llmResult.totalLatencyMs, source.tier);

        // Check for existing similar claims
        const existingClaim = await findSimilarClaim(env, claim.statement, wikiPageId);

        if (existingClaim) {
          // Link as supporting or conflicting
          await linkClaimToExisting(env, existingClaim.id, claimId, claim.statement);
        }

        await env.DB.prepare(`
          INSERT INTO wiki_provenance_claims (
            id, page_id, statement, source_url, source_tier, confidence,
            extraction_method, extracted_at, cost_to_produce, verification_status,
            supporting_claim_ids, conflicting_claim_ids, created_at, updated_at
          ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'unverified', ?, ?, ?, ?)
        `).bind(
          claimId,
          wikiPageId,
          claim.statement.slice(0, 2000),
          source.url,
          source.tier,
          claim.confidence || 0.5,
          claim.method || 'llm',
          now,
          costToProduce,
          JSON.stringify([]),
          JSON.stringify([]),
          now,
          now,
        ).run();

        result.claimsCreated++;
      }
    } catch (err: any) {
      result.errors.push(`Error extracting from ${source.url}: ${err.message || String(err)}`);
    }
  }

  return result;
}

// ── Find Similar Existing Claim ──

async function findSimilarClaim(
  env: Env,
  statement: string,
  pageId: string,
): Promise<ProvenanceClaim | null> {
  // Simple keyword-based matching
  const keywords = statement.toLowerCase().split(/\s+/).filter(w => w.length > 4).slice(0, 5);

  if (keywords.length === 0) return null;

  const ftsQuery = keywords.join(' OR ');
  try {
    const rows = await env.DB.prepare(`
      SELECT * FROM wiki_provenance_claims
      WHERE page_id = ? AND statement LIKE ?
      LIMIT 1
    `).bind(pageId, `%${keywords[0]}%`).all();

    if (rows.results && rows.results.length > 0) {
      return rowToProvenanceClaim(rows.results[0] as unknown as WikiProvenanceClaimRow);
    }
  } catch {
    // Fallback: no similar claim found
  }

  return null;
}

// ── Link Claim to Existing ──

async function linkClaimToExisting(
  env: Env,
  existingClaimId: string,
  newClaimId: string,
  newStatement: string,
): Promise<void> {
  const now = new Date().toISOString();

  // Get existing claim
  const existing = await env.DB.prepare(
    'SELECT supporting_claim_ids, conflicting_claim_ids FROM wiki_provenance_claims WHERE id = ?'
  ).bind(existingClaimId).first() as { supporting_claim_ids: string | null; conflicting_claim_ids: string | null } | null;

  if (!existing) return;

  const supportingIds: string[] = existing.supporting_claim_ids ? JSON.parse(existing.supporting_claim_ids) : [];

  // Add new claim as supporting (simplified — in production, use LLM to determine support vs conflict)
  supportingIds.push(newClaimId);

  await env.DB.prepare(`
    UPDATE wiki_provenance_claims
    SET supporting_claim_ids = ?, verification_status = 'supported', updated_at = ?
    WHERE id = ?
  `).bind(JSON.stringify(supportingIds), now, existingClaimId).run();
}

// ── Detect Conflicts ──

export async function detectConflicts(
  env: Env,
  pageId: string,
): Promise<Array<{
  claimA: ProvenanceClaim;
  claimB: ProvenanceClaim;
  conflictDescription: string;
}>> {
  const conflicts: Array<{
    claimA: ProvenanceClaim;
    claimB: ProvenanceClaim;
    conflictDescription: string;
  }> = [];

  // Get all claims for this page
  const pageClaims = await env.DB.prepare(
    'SELECT * FROM wiki_provenance_claims WHERE page_id = ? ORDER BY extracted_at DESC'
  ).bind(pageId).all();

  const claims = (pageClaims.results as unknown as WikiProvenanceClaimRow[]).map(rowToProvenanceClaim);

  // Also get claims from other pages that might conflict
  const otherClaims = await env.DB.prepare(`
    SELECT * FROM wiki_provenance_claims
    WHERE page_id != ? AND verification_status IN ('unverified', 'conflicted')
    ORDER BY extracted_at DESC
    LIMIT 50
  `).bind(pageId).all();

  const otherClaimList = (otherClaims.results as unknown as WikiProvenanceClaimRow[]).map(rowToProvenanceClaim);

  // Use LLM to compare claims pairwise (sample to avoid O(n^2) explosion)
  const samplePage = claims.slice(0, 10);
  const sampleOther = otherClaimList.slice(0, 10);

  if (samplePage.length === 0 || sampleOther.length === 0) {
    return conflicts;
  }

  const claimsForComparison = samplePage.map((c, i) =>
    `[A${i + 1}] ${c.statement.slice(0, 200)} (from ${c.sourceUrl}, tier: ${c.sourceTier})`
  ).join('\n');

  const otherClaimsForComparison = sampleOther.map((c, i) =>
    `[B${i + 1}] ${c.statement.slice(0, 200)} (from ${c.sourceUrl}, tier: ${c.sourceTier})`
  ).join('\n');

  const conflictPrompt = `Compare these two sets of claims and identify any CONFLICTING pairs.

Page claims:
${claimsForComparison}

Other claims:
${otherClaimsForComparison}

For each conflict, respond with a JSON array:
[{"claimA": "A1", "claimB": "B3", "description": "A1 says X while B3 says Y"}]

If no conflicts exist, respond with: []`;

  const result = await routeLLMCall(
    env,
    [{ role: 'user', content: conflictPrompt }],
    1024, 0.0, undefined, false, false,
  );

  if (result.content.startsWith('[ALL_LLM_FAILED]')) {
    return conflicts;
  }

  try {
    let content = result.content.trim();
    const jsonMatch = content.match(/\[[\s\S]*\]/);
    if (!jsonMatch) return conflicts;

    const parsed = JSON.parse(jsonMatch[0]);

    if (!Array.isArray(parsed)) return conflicts;

    for (const conflict of parsed) {
      const claimAIdx = parseInt((conflict.claimA || '').replace('A', '')) - 1;
      const claimBIdx = parseInt((conflict.claimB || '').replace('B', '')) - 1;

      if (claimAIdx >= 0 && claimAIdx < samplePage.length && claimBIdx >= 0 && claimBIdx < sampleOther.length) {
        const claimA = samplePage[claimAIdx];
        const claimB = sampleOther[claimBIdx];

        // Update both claims to reflect the conflict
        await updateClaimConflicts(env, claimA.id, claimB.id);
        await updateClaimConflicts(env, claimB.id, claimA.id);

        conflicts.push({
          claimA,
          claimB,
          conflictDescription: conflict.description || 'Claims conflict',
        });
      }
    }
  } catch {
    // Non-critical
  }

  return conflicts;
}

// ── Update Claim Conflicts ──

async function updateClaimConflicts(
  env: Env,
  claimId: string,
  conflictingClaimId: string,
): Promise<void> {
  const row = await env.DB.prepare(
    'SELECT conflicting_claim_ids FROM wiki_provenance_claims WHERE id = ?'
  ).bind(claimId).first();

  if (!row) return;

  const conflictingIds: string[] = row.conflicting_claim_ids ? JSON.parse(row.conflicting_claim_ids as string) : [];

  if (!conflictingIds.includes(conflictingClaimId)) {
    conflictingIds.push(conflictingClaimId);
    await env.DB.prepare(`
      UPDATE wiki_provenance_claims
      SET conflicting_claim_ids = ?, verification_status = 'conflicted', updated_at = ?
      WHERE id = ?
    `).bind(
      JSON.stringify(conflictingIds),
      new Date().toISOString(),
      claimId,
    ).run();
  }
}

// ── Get Claim Lineage ──

export async function getClaimLineage(
  env: Env,
  claimId: string,
): Promise<{
  claim: ProvenanceClaim | null;
  supportingClaims: ProvenanceClaim[];
  conflictingClaims: ProvenanceClaim[];
}> {
  const row = await env.DB.prepare(
    'SELECT * FROM wiki_provenance_claims WHERE id = ?'
  ).bind(claimId).first() as WikiProvenanceClaimRow | null;

  if (!row) {
    return { claim: null, supportingClaims: [], conflictingClaims: [] };
  }

  const claim = rowToProvenanceClaim(row);

  // Load supporting claims
  const supportingClaims: ProvenanceClaim[] = [];
  for (const sid of claim.supportingClaims) {
    const supportRow = await env.DB.prepare(
      'SELECT * FROM wiki_provenance_claims WHERE id = ?'
    ).bind(sid).first() as WikiProvenanceClaimRow | null;
    if (supportRow) supportingClaims.push(rowToProvenanceClaim(supportRow));
  }

  // Load conflicting claims
  const conflictingClaims: ProvenanceClaim[] = [];
  for (const cid of claim.conflictingClaims) {
    const conflictRow = await env.DB.prepare(
      'SELECT * FROM wiki_provenance_claims WHERE id = ?'
    ).bind(cid).first() as WikiProvenanceClaimRow | null;
    if (conflictRow) conflictingClaims.push(rowToProvenanceClaim(conflictRow));
  }

  return { claim, supportingClaims, conflictingClaims };
}

// ── Audit Provenance ──

export async function auditProvenance(
  env: Env,
  pageId: string,
): Promise<ProvenanceAuditReport> {
  const rows = await env.DB.prepare(
    'SELECT * FROM wiki_provenance_claims WHERE page_id = ? ORDER BY extracted_at ASC'
  ).bind(pageId).all();

  const claims = (rows.results as unknown as WikiProvenanceClaimRow[]).map(rowToProvenanceClaim);

  const verifiedClaims = claims.filter(c => c.verificationStatus === 'supported' || c.verificationStatus === 'resolved').length;
  const conflictedClaims = claims.filter(c => c.verificationStatus === 'conflicted').length;
  const totalCost = claims.reduce((sum, c) => sum + c.costToProduce, 0);

  return {
    pageId,
    claims,
    totalClaims: claims.length,
    verifiedClaims,
    conflictedClaims,
    totalCost,
  };
}

// ── Resolve Conflict ──

export async function resolveConflict(
  env: Env,
  claimAId: string,
  claimBId: string,
  resolution: string,
  resolver: string,
): Promise<boolean> {
  const now = new Date().toISOString();

  // Update claim A
  await env.DB.prepare(`
    UPDATE wiki_provenance_claims
    SET verification_status = 'resolved', updated_at = ?
    WHERE id = ?
  `).bind(now, claimAId).run();

  // Update claim B
  await env.DB.prepare(`
    UPDATE wiki_provenance_claims
    SET verification_status = 'resolved', updated_at = ?
    WHERE id = ?
  `).bind(now, claimBId).run();

  // Log the resolution in security log
  await env.DB.prepare(`
    INSERT INTO wiki_security_log (id, page_id, event_type, details, threat_level, created_at)
    VALUES (?, ?, 'conflict_resolve', ?, 'none', ?)
  `).bind(
    generateUUID(),
    null,
    JSON.stringify({
      claimA: claimAId,
      claimB: claimBId,
      resolution,
      resolver,
    }),
    now,
  ).run();

  return true;
}

// ── Estimate Cost to Produce ──

function estimateCostToProduce(latencyMs: number, sourceTier: string): number {
  // Rough cost estimation based on latency and source tier
  const baseCost = latencyMs * 0.0001; // $0.0001 per ms
  const tierMultiplier = sourceTier === 'P1' ? 1.5 : sourceTier === 'P2' ? 1.2 : 1.0;
  return baseCost * tierMultiplier;
}
