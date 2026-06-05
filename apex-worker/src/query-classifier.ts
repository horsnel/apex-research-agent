/**
 * APEX Research Agent — Query Classifier
 * Rules-based + LLM classification
 */

import { Env, ClassificationResult } from './types';
import { classifyWithRouter } from './llm-router';

// ── Temporal Keywords ──
const TEMPORAL_KEYWORDS = [
  'latest', 'recent', 'today', 'yesterday', 'this week', 'this month', 'this year',
  'current', 'now', '2024', '2025', '2026', 'breaking', 'update', 'new',
];

// ── Academic Keywords ──
const ACADEMIC_KEYWORDS = [
  'paper', 'study', 'research', 'arxiv', 'pubmed', 'doi', 'citation',
  'journal', 'conference', 'preprint', 'thesis', 'dissertation',
  'methodology', 'findings', 'abstract', 'peer-reviewed',
];

// ── Clinical Keywords ──
const CLINICAL_KEYWORDS = [
  'clinical trial', 'treatment', 'diagnosis', 'prognosis', 'side effects',
  'fda', 'drug', 'therapy', 'patient', 'randomized', 'cochrane',
];

/**
 * Classify a query — rules first, LLM fallback.
 */
export async function classifyQuery(env: Env, query: string): Promise<ClassificationResult> {
  const lower = query.toLowerCase();

  // Rule 1: Temporal keywords → live
  if (TEMPORAL_KEYWORDS.some(kw => lower.includes(kw))) {
    return {
      route: 'live',
      reason: 'Temporal keyword detected — likely needs current data',
      domainHint: '',
      confidence: 0.9,
      method: 'rules',
    };
  }

  // Rule 2: Academic keywords → rag
  if (ACADEMIC_KEYWORDS.some(kw => lower.includes(kw))) {
    return {
      route: 'rag',
      reason: 'Academic keyword detected — likely answerable from corpus',
      domainHint: 'academic',
      confidence: 0.85,
      method: 'rules',
    };
  }

  // Rule 3: Clinical keywords → rag (clinical)
  if (CLINICAL_KEYWORDS.some(kw => lower.includes(kw))) {
    return {
      route: 'rag',
      reason: 'Clinical keyword detected — likely answerable from medical corpus',
      domainHint: 'clinical',
      confidence: 0.85,
      method: 'rules',
    };
  }

  // Rule 4: Short factual queries → rag
  if (query.split(/\s+/).length <= 4) {
    return {
      route: 'rag',
      reason: 'Short factual query — likely answerable from corpus',
      domainHint: '',
      confidence: 0.7,
      method: 'rules',
    };
  }

  // Fallback: Use LLM classification
  try {
    const llmResult = await classifyWithRouter(env, query);
    return {
      route: llmResult.route === 'rag' ? 'rag' : 'live',
      reason: llmResult.reason,
      domainHint: llmResult.domain_hint || '',
      confidence: 0.6,
      method: 'llm',
    };
  } catch {
    // Default to live if LLM fails
    return {
      route: 'live',
      reason: 'Classification fallback — defaulting to live',
      domainHint: '',
      confidence: 0.5,
      method: 'fallback',
    };
  }
}

/**
 * Should we escalate to live search?
 */
export function shouldEscalateToLive(
  similarity: number,
  hasP1: boolean,
  threshold = 0.72,
): boolean {
  // If similarity is low, always escalate
  if (similarity < threshold) return true;
  // If no P1 sources and similarity is marginal, escalate
  if (!hasP1 && similarity < 0.80) return true;
  return false;
}
