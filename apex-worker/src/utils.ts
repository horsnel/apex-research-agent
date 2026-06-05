/**
 * APEX Research Agent — Shared Utilities
 * CORS helpers, UUID generation, timers, error formatting
 */

import { Env } from './types';

// ── CORS ──

const ALLOWED_ORIGINS = [
  'https://kovira.pages.dev',
  'https://www.kovira.pages.dev',
  'http://localhost:3000',
  'http://localhost:8000',
  'http://127.0.0.1:3000',
  'http://127.0.0.1:8000',
];

export function getAllowedOrigin(request: Request): string {
  const origin = request.headers.get('Origin') || '';
  if (ALLOWED_ORIGINS.includes(origin)) return origin;
  // Allow *.pages.dev for preview deploys
  if (origin.endsWith('.pages.dev')) return origin;
  return ALLOWED_ORIGINS[0];
}

export function corsHeaders(request: Request): Record<string, string> {
  const origin = getAllowedOrigin(request);
  return {
    'Access-Control-Allow-Origin': origin,
    'Access-Control-Allow-Methods': 'GET, POST, PUT, PATCH, DELETE, OPTIONS',
    'Access-Control-Allow-Headers': 'Content-Type, Authorization',
    'Access-Control-Allow-Credentials': 'true',
    'Access-Control-Max-Age': '86400',
    'Vary': 'Origin',
  };
}

export function handleOptions(request: Request): Response {
  return new Response(null, { status: 204, headers: corsHeaders(request) });
}

// ── Response Helpers ──

export function jsonResponse(data: unknown, status = 200, request?: Request): Response {
  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
  };
  if (request) {
    Object.assign(headers, corsHeaders(request));
  }
  return new Response(JSON.stringify(data), { status, headers });
}

export function errorResponse(message: string, status = 500, request?: Request): Response {
  return jsonResponse({ error: message, timestamp: new Date().toISOString() }, status, request);
}

// ── UUID ──

export function generateUUID(): string {
  // Crypto.randomUUID is available in Workers
  return crypto.randomUUID();
}

// ── Hashing ──

export async function hashText(text: string): Promise<string> {
  const encoder = new TextEncoder();
  const data = encoder.encode(text);
  const hashBuffer = await crypto.subtle.digest('SHA-256', data);
  const hashArray = Array.from(new Uint8Array(hashBuffer));
  return hashArray.map(b => b.toString(16).padStart(2, '0')).join('');
}

// ── Timer ──

export class Timer {
  private start: number;
  constructor() { this.start = Date.now(); }
  elapsed(): number { return Date.now() - this.start; }
}

// ── Token Estimation ──

export function estimateTokens(text: string): number {
  // Rough: 1 token ≈ 4 chars for English
  return Math.ceil(text.length / 4);
}

// ── Domain Extraction ──

export function extractDomain(url: string): string {
  try {
    return new URL(url).hostname;
  } catch {
    return '';
  }
}

// ── Source Tier Enforcement ──

const TIER_DOMAIN_MAP: Record<string, string[]> = {
  P1: [
    'arxiv.org', 'pubmed.ncbi.nlm.nih.gov', 'nature.com', 'science.org',
    'nejm.org', 'lancet.com', 'dl.acm.org', 'ieee.org', 'springer.com',
    'wiley.com', 'semanticscholar.org', 'openreview.net', 'biorxiv.org', 'medrxiv.org',
  ],
  P2: [
    'nih.gov', 'nasa.gov', 'cdc.gov', 'who.int', 'nist.gov', 'gov.uk',
  ],
  P3: [
    'medium.com', 'substack.com', 'wikipedia.org', 'stackoverflow.com', 'reddit.com',
  ],
};

export function enforceSourceTier(url: string, currentTier: string): string {
  const domain = extractDomain(url).toLowerCase();

  for (const [tier, domains] of Object.entries(TIER_DOMAIN_MAP)) {
    for (const d of domains) {
      if (d.startsWith('*.')) {
        if (domain.endsWith(d.slice(1))) return tier;
      } else if (domain === d || domain.endsWith('.' + d)) {
        return tier;
      }
    }
  }

  // Academic domain patterns
  if (domain.endsWith('.edu') || domain.endsWith('.ac.uk') || domain.endsWith('.ac.jp')) {
    return 'P2';
  }

  return currentTier;
}

// ── Temporal Decay ──

export function applyTemporalDecay(score: number, publishedDate: string | null, decayFactor = 0.95): number {
  if (!publishedDate) return score * 0.9; // Undated sources get 10% penalty

  const year = new Date(publishedDate).getFullYear();
  const currentYear = new Date().getFullYear();
  const ageInYears = currentYear - year;

  if (ageInYears <= 0) return score;
  return score * Math.pow(decayFactor, ageInYears);
}

// ── Source Hierarchy ──

export function applySourceHierarchy(score: number, tier: string): number {
  switch (tier) {
    case 'P1': return score * 1.5;
    case 'P2': return score * 1.2;
    case 'P3': return score * 1.0;
    case 'UNV': return score * 0.3;
    default: return score * 0.8;
  }
}
