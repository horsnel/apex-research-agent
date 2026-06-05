/**
 * APEX Research Agent — Research Engine
 * Deep research with 6 competitive upgrades:
 * 1. Source Tier Enforcement
 * 2. Verification Loop (epistemic markers)
 * 3. Parallel Orchestration (Promise.allSettled)
 * 4. Research Report Mode
 * 5. Iterative Research Loop
 * 6. Structured Extraction
 */

import { Env, SearchResult, VerifiedClaim, VerificationResult, ResearchReport, Finding, Debate, EpistemicStatus, EvidenceType } from './types';
import { searchRouter } from './search-sources';
import { synthesizeWithRouter, routeLLMCall, APEX_SYSTEM_PROMPT } from './llm-router';
import { enforceSourceTier } from './utils';

/**
 * Deep research — parallel source search + verification + report generation.
 */
export async function deepResearch(
  env: Env,
  query: string,
  classification: string = 'web',
  depth: 'quick' | 'thorough' = 'quick',
  verify = true,
  extract = false,
): Promise<{
  sources: SearchResult[];
  verification: VerificationResult | null;
  extracted_claims: VerifiedClaim[];
  sub_queries: string[];
  latency_ms: number;
}> {
  const startTime = Date.now();

  // Step 1: Parallel search across all relevant sources
  const sources = await parallelResearch(env, query, classification);

  // Step 2: Verification (if enabled)
  let verification: VerificationResult | null = null;
  if (verify && sources.length > 0) {
    verification = await verifyClaimsFromSources(sources);
  }

  // Step 3: Structured extraction (if enabled)
  let extractedClaims: VerifiedClaim[] = [];
  if (extract && sources.length > 0) {
    extractedClaims = extractClaimsFromSources(
      sources.filter(s => s.tier === 'P1'),
    );
  }

  // Step 4: Iterative research (if thorough)
  let subQueries: string[] = [query];
  if (depth === 'thorough') {
    const iterative = await iterativeResearch(env, query, sources, classification);
    sources.push(...iterative.newSources);
    subQueries.push(...iterative.subQueries);
  }

  return {
    sources,
    verification,
    extracted_claims: extractedClaims,
    sub_queries: subQueries,
    latency_ms: Date.now() - startTime,
  };
}

/**
 * Parallel research — fire all search sources simultaneously with graceful degradation.
 */
export async function parallelResearch(
  env: Env,
  query: string,
  classification: string,
): Promise<SearchResult[]> {
  const results = await searchRouter(env, query, classification);

  // Enforce source tier by domain
  for (const result of results) {
    result.tier = enforceSourceTier(result.url, result.tier);
  }

  return results;
}

/**
 * Verify claims from multiple sources with epistemic marking.
 */
export async function verifyClaimsFromSources(
  sources: SearchResult[],
): Promise<VerificationResult> {
  const claims = extractClaimsFromSources(sources);

  const verifiedClaims: VerifiedClaim[] = claims.map(claim => {
    // Count how many independent sources support vs conflict
    const supporting: string[] = [];
    const conflicting: string[] = [];

    for (const source of sources) {
      const snippetLower = source.snippet.toLowerCase();
      const claimLower = claim.statement.toLowerCase().slice(0, 50);

      if (snippetLower.includes(claimLower) || similarity(snippetLower, claimLower) > 0.6) {
        supporting.push(source.url);
      }
    }

    // Determine epistemic status
    let epistemicStatus: EpistemicStatus = 'UNVERIFIED';
    let confidence = 0.3;
    let evidenceType: EvidenceType = 'unknown';

    if (supporting.length >= 3) {
      epistemicStatus = 'ESTABLISHED';
      confidence = 0.85;
      evidenceType = 'observational';
    } else if (supporting.length >= 2) {
      epistemicStatus = 'TENTATIVE';
      confidence = 0.65;
      evidenceType = 'observational';
    } else if (conflicting.length > 0 && supporting.length > 0) {
      epistemicStatus = 'ACTIVE_DEBATE';
      confidence = 0.4;
      evidenceType = 'observational';
    } else if (supporting.length === 1) {
      epistemicStatus = 'SPECULATIVE';
      confidence = 0.3;
      evidenceType = 'theoretical';
    }

    // Check for P1 sources (boost confidence)
    const p1Sources = sources.filter(s => s.tier === 'P1');
    if (p1Sources.length > 0 && supporting.length > 0) {
      confidence = Math.min(confidence + 0.1, 0.95);
    }

    return {
      statement: claim.statement,
      epistemicStatus,
      supportingSources: supporting,
      conflictingSources: conflicting,
      confidence,
      evidenceType,
    };
  });

  return {
    claims: verifiedClaims,
    totalSourcesChecked: sources.length,
    establishedCount: verifiedClaims.filter(c => c.epistemicStatus === 'ESTABLISHED').length,
    tentativeCount: verifiedClaims.filter(c => c.epistemicStatus === 'TENTATIVE').length,
    contestedCount: verifiedClaims.filter(c => c.epistemicStatus === 'ACTIVE_DEBATE').length,
    unverifiableCount: verifiedClaims.filter(c => c.epistemicStatus === 'UNVERIFIED' || c.epistemicStatus === 'SPECULATIVE').length,
  };
}

/**
 * Extract claims from source snippets using pattern matching.
 */
export function extractClaimsFromSources(sources: SearchResult[]): VerifiedClaim[] {
  const claims: VerifiedClaim[] = [];
  const causalPatterns = [
    /([^.]*?(?:causes?|leads? to|results? in|increases?|decreases?|improves?|reduces?|prevents?|enhances?)[^.]*?\.)[^.]*\./gi,
    /([^.]*?(?:found that|showed that|demonstrated that|revealed that|indicates? that|suggests? that)[^.]*?\.)[^.]*\./gi,
    /([^.]*?(?:is associated with|is correlated with|is linked to|predicts?)[^.]*?\.)[^.]*\./gi,
  ];

  for (const source of sources) {
    for (const pattern of causalPatterns) {
      const matches = source.snippet.matchAll(pattern);
      for (const match of matches) {
        const statement = match[1]?.trim();
        if (statement && statement.length > 20 && statement.length < 300) {
          // Deduplicate
          if (!claims.some(c => c.statement === statement)) {
            claims.push({
              statement,
              epistemicStatus: 'SPECULATIVE',
              supportingSources: [source.url],
              conflictingSources: [],
              confidence: 0.3,
              evidenceType: source.tier === 'P1' ? 'experimental' : 'unknown',
            });
          }
        }
      }
    }
  }

  return claims.slice(0, 20); // Cap at 20 claims
}

/**
 * Iterative research — identify gaps and search for more.
 */
async function iterativeResearch(
  env: Env,
  query: string,
  initialSources: SearchResult[],
  classification: string,
  maxCycles = 3,
): Promise<{ newSources: SearchResult[]; subQueries: string[] }> {
  const newSources: SearchResult[] = [];
  const subQueries: string[] = [];

  for (let cycle = 0; cycle < maxCycles; cycle++) {
    // Use LLM to identify gaps
    const gapPrompt = `Given this research query: "${query}"
And these sources found so far:
${initialSources.concat(newSources).slice(0, 10).map((s, i) => `[${i + 1}] ${s.title} (${s.url})`).join('\n')}

What specific sub-questions or aspects are NOT covered by these sources?
List exactly 2 follow-up search queries, one per line. Be specific.`;

    const gapResult = await routeLLMCall(
      env,
      [{ role: 'user', content: gapPrompt }],
      100, 0.0, undefined, false, false,
    );

    if (gapResult.content.startsWith('[ALL_LLM_FAILED]')) break;

    const lines = gapResult.content.split('\n').filter(l => l.trim().length > 5).slice(0, 2);
    if (lines.length === 0) break;

    for (const line of lines) {
      const subQuery = line.replace(/^\d+\.\s*/, '').replace(/^[-*]\s*/, '').trim();
      if (subQuery.length < 5 || subQuery.length > 200) continue;
      if (subQueries.includes(subQuery)) continue;

      subQueries.push(subQuery);

      const subResults = await parallelResearch(env, subQuery, classification);
      // Deduplicate
      const existingUrls = new Set([
        ...initialSources.map(s => s.url),
        ...newSources.map(s => s.url),
      ]);
      for (const result of subResults) {
        if (!existingUrls.has(result.url)) {
          newSources.push(result);
          existingUrls.add(result.url);
        }
      }
    }
  }

  return { newSources, subQueries };
}

/**
 * Generate a structured research report using LLM.
 */
export async function generateResearchReport(
  env: Env,
  query: string,
  sources: SearchResult[],
  verification: VerificationResult | null,
  depth: 'quick' | 'thorough' = 'quick',
): Promise<ResearchReport> {
  const sourceList = sources.slice(0, 15).map((s, i) =>
    `[${i + 1}] ${s.title} (${s.tier}) - ${s.url}\n    ${s.snippet.slice(0, 200)}`
  ).join('\n\n');

  const verificationSection = verification ? `
Verification Summary:
- ESTABLISHED claims: ${verification.establishedCount}
- TENTATIVE claims: ${verification.tentativeCount}
- CONTESTED claims: ${verification.contestedCount}
- UNVERIFIABLE claims: ${verification.unverifiableCount}
${verification.claims.slice(0, 5).map(c => `  - [${c.epistemicStatus}] ${c.statement.slice(0, 100)} (confidence: ${c.confidence.toFixed(2)})`).join('\n')}
` : '';

  const reportPrompt = `Generate a structured research report for: "${query}"

Sources:
${sourceList}

${verificationSection}

Format your report EXACTLY as:

## Executive Summary
[2-3 sentence dense summary of findings]

## Findings
| Claim | Evidence | Sources | Status | Confidence |
|-------|----------|---------|--------|------------|
| [claim] | [evidence] | [S1,S2] | [ESTABLISHED/TENTATIVE/SPECULATIVE] | [0.0-1.0] |

## Active Debates
- **[Topic]**: Position A [S1, P1] vs Position B [S2, P2]

## Speculative
- [speculative claim] [SPECULATIVE]

## Sources
${sources.slice(0, 15).map((s, i) => `[${i + 1}] ${s.url} (${s.tier})`).join('\n')}`;

  const tokenBudget = depth === 'thorough' ? 4096 : 2048;
  const result = await routeLLMCall(
    env,
    [{ role: 'system', content: 'You are a research report generator. Produce structured, evidence-based reports.' },
     { role: 'user', content: reportPrompt }],
    tokenBudget, 0.0, undefined, true, // tableNeeded for structured output
  );

  return parseReport(query, result.content, sources, verification);
}

/**
 * Parse LLM markdown output into structured ResearchReport.
 */
function parseReport(
  query: string,
  rawReport: string,
  sources: SearchResult[],
  verification: VerificationResult | null,
): ResearchReport {
  // Extract sections
  const execMatch = rawReport.match(/## Executive Summary\s*\n([\s\S]*?)(?=\n##|$)/i);
  const findingsMatch = rawReport.match(/## Findings\s*\n([\s\S]*?)(?=\n##|$)/i);
  const debatesMatch = rawReport.match(/## Active Debates\s*\n([\s\S]*?)(?=\n##|$)/i);
  const speculativeMatch = rawReport.match(/## Speculative\s*\n([\s\S]*?)(?=\n##|$)/i);

  // Parse findings table rows
  const findings: Finding[] = [];
  if (findingsMatch) {
    const rows = findingsMatch[1].matchAll(/\|\s*([^|]+)\|\s*([^|]+)\|\s*([^|]+)\|\s*([^|]+)\|\s*([^|]+)\|/g);
    for (const row of rows) {
      const claim = row[1]?.trim();
      if (!claim || claim.startsWith('---') || claim.toLowerCase() === 'claim') continue;
      findings.push({
        claim,
        evidence: row[2]?.trim() || '',
        sources: (row[3]?.trim() || '').split(',').map(s => s.trim()),
        epistemicStatus: normalizeEpistemic(row[4]?.trim() || 'SPECULATIVE'),
        confidence: parseFloat(row[5]?.trim() || '0.3'),
      });
    }
  }

  // Parse debates
  const debates: Debate[] = [];
  if (debatesMatch) {
    const lines = debatesMatch[1].split('\n').filter(l => l.trim().startsWith('-'));
    for (const line of lines) {
      const topicMatch = line.match(/\*\*(.+?)\*\*:\s*(.+?)\s*vs\s*(.+)/);
      if (topicMatch) {
        debates.push({
          topic: topicMatch[1],
          position_a: topicMatch[2],
          position_b: topicMatch[3],
          sources: [],
        });
      }
    }
  }

  // Parse speculative
  const speculative: string[] = [];
  if (speculativeMatch) {
    const lines = speculativeMatch[1].split('\n').filter(l => l.trim().startsWith('-'));
    for (const line of lines) {
      speculative.push(line.replace(/^-\s*/, '').replace(/\[SPECULATIVE\]/g, '').trim());
    }
  }

  return {
    query,
    executiveSummary: execMatch?.[1]?.trim() || '',
    findings,
    debates,
    speculative,
    sources: sources.slice(0, 15).map(s => ({
      url: s.url,
      tier: s.tier,
      title: s.title,
      similarity: 0,
    })),
    verification,
    rawReport,
  };
}

function normalizeEpistemic(status: string): EpistemicStatus {
  const upper = status.toUpperCase().replace(/[\[\]]/g, '');
  if (upper.includes('ESTABLISHED')) return 'ESTABLISHED';
  if (upper.includes('TENTATIVE')) return 'TENTATIVE';
  if (upper.includes('DEBATE') || upper.includes('CONTESTED')) return 'ACTIVE_DEBATE';
  if (upper.includes('SPECULATIVE')) return 'SPECULATIVE';
  return 'UNVERIFIED';
}

/**
 * Simple string similarity (Jaccard on words).
 */
function similarity(a: string, b: string): number {
  const wordsA = new Set(a.split(/\s+/));
  const wordsB = new Set(b.split(/\s+/));
  const intersection = new Set([...wordsA].filter(x => wordsB.has(x)));
  const union = new Set([...wordsA, ...wordsB]);
  return union.size > 0 ? intersection.size / union.size : 0;
}
