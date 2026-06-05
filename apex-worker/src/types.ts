/**
 * APEX Research Agent — TypeScript Type Definitions
 * All data models for the Cloudflare Worker migration
 */

// ── Environment Bindings ──

export interface Env {
  AI: Ai;                              // Workers AI binding
  DB: D1Database;                       // D1 database binding
  BUCKET: R2Bucket;                     // R2 bucket binding
  VECTORIZE: VectorizeIndex;            // Vectorize index binding
  ENVIRONMENT: string;
  DEFAULT_SYNTHESIS_TOKENS: string;
  MAX_SYNTHESIS_TOKENS: string;
  SIMILARITY_THRESHOLD: string;
  PASS_THROUGH_THRESHOLD: string;
  RAG_TOP_K: string;
  RAG_FINAL_K: string;
  MAX_RAG_CONTEXT_TOKENS: string;
  // Secrets
  SERPER_API_KEY?: string;
  COHERE_API_KEY?: string;
  JINA_API_KEY?: string;
  FIRECRAWL_API_KEY?: string;
  NEWSAPI_KEY?: string;
  YOUTUBE_API_KEY?: string;
}

// ── LLM Router Types ──

export type Provider = 'passthrough' | 'cloudflare' | 'github' | 'openai' | 'anthropic';
export type ModelTier = 'free' | 'cheap' | 'mid' | 'capable' | 'cloud';

export interface ModelConfig {
  name: string;
  provider: Provider;
  modelId: string;
  contextWindow: number;
  maxOutputTokens: number;
  priceInputPerM: number;
  priceOutputPerM: number;
  supportsTools: boolean;
  tier: ModelTier;
  enabled: boolean;
  description: string;
}

export interface LLMCallResult {
  success: boolean;
  content: string;
  modelName: string;
  modelId: string;
  provider: string;
  latencyMs: number;
  tokensUsed: number;
  error: string;
}

export interface RouterResult {
  content: string;
  modelName: string;
  modelId: string;
  provider: string;
  fallbackCount: number;
  totalLatencyMs: number;
  attempts: RouterAttempt[];
}

export interface RouterAttempt {
  model: string;
  modelId?: string;
  provider: string;
  status: 'success' | 'failed' | 'skipped';
  latencyMs?: number;
  tokensUsed?: number;
  error?: string | null;
  reason?: string;
}

export interface ModelTestResult {
  modelName: string;
  modelId: string;
  provider: string;
  configured: boolean;
  reachable: boolean;
  latencyMs: number;
  error: string;
  sampleOutput: string;
}

// ── Retrieval Types ──

export interface RetrievedChunk {
  id: string;
  sourceUrl: string;
  sourceTier: string;
  domain: string;
  docType: string;
  title: string;
  authors: string[];
  rawText: string;
  metadata: Record<string, unknown>;
  chunkIndex: number;
  totalChunks: number;
  similarityScore: number;
  keywordScore: number;
  fusedScore: number;
  tokenCount: number;
}

// ── Query Classification ──

export interface ClassificationResult {
  route: 'rag' | 'live' | 'rag+live';
  reason: string;
  domainHint: string;
  confidence: number;
  method: string;
}

// ── Synthesis Types ──

export interface SynthesisResult {
  answer: string;
  method: string;  // 'pass_through' | 'synthesis' | 'table' | 'raw_context'
  tokenCount: number;
  sourcesUsed: SourceInfo[];
  modelUsed: string;
  provider: string;
  fallbackCount: number;
}

export interface SourceInfo {
  url: string;
  tier: string;
  title: string;
  similarity: number;
}

// ── Research Engine Types ──

export type EpistemicStatus = 'ESTABLISHED' | 'TENTATIVE' | 'ACTIVE_DEBATE' | 'SPECULATIVE' | 'UNVERIFIED';
export type EvidenceType = 'experimental' | 'observational' | 'theoretical' | 'computational' | 'survey' | 'meta_analysis' | 'editorial' | 'unknown';

export interface VerifiedClaim {
  statement: string;
  epistemicStatus: EpistemicStatus;
  supportingSources: string[];
  conflictingSources: string[];
  confidence: number;
  evidenceType: EvidenceType;
  sampleSize?: string;
  year?: number;
}

export interface VerificationResult {
  claims: VerifiedClaim[];
  totalSourcesChecked: number;
  establishedCount: number;
  tentativeCount: number;
  contestedCount: number;
  unverifiableCount: number;
}

export interface ResearchReport {
  query: string;
  executiveSummary: string;
  findings: Finding[];
  debates: Debate[];
  speculative: string[];
  sources: SourceInfo[];
  verification: VerificationResult | null;
  rawReport: string;
}

export interface Finding {
  claim: string;
  evidence: string;
  sources: string[];
  epistemicStatus: EpistemicStatus;
  confidence: number;
}

export interface Debate {
  topic: string;
  position_a: string;
  position_b: string;
  sources: string[];
}

// ── Search Sources Types ──

export interface SearchResult {
  title: string;
  url: string;
  snippet: string;
  source: string;
  tier: string;
  date?: string;
}

// ── Live Scraper Types ──

export interface ScrapeResult {
  url: string;
  markdown: string;
  title: string;
  success: boolean;
  error: string;
}

// ── API Request/Response Types ──

export interface QueryRequest {
  query: string;
  force_live?: boolean;
  domain_filter?: string;
  tier_filter?: string;
  max_tokens?: number;
  depth?: 'quick' | 'thorough';
}

export interface QueryResponse {
  answer: string;
  route: string;
  method: string;
  sources: SourceInfo[];
  token_count: number;
  latency_ms: number;
  similarity_score: number | null;
  validation: {
    is_valid: boolean;
    total_claims: number;
    cited_claims: number;
    warnings: string[];
  } | null;
  model_used: string;
  provider: string;
  fallback_count: number;
  verification: {
    claims: Array<{
      statement: string;
      status: EpistemicStatus;
      confidence: number;
      evidence_type: EvidenceType;
    }>;
    summary: {
      established: number;
      tentative: number;
      contested: number;
      unverifiable: number;
    };
  } | null;
}

export interface ResearchRequest {
  query: string;
  classification?: string;
  depth?: 'quick' | 'thorough';
  verify?: boolean;
  extract?: boolean;
  check_retractions?: boolean;
  force_live?: boolean;
}

export interface ResearchResponse {
  query: string;
  executive_summary: string;
  findings: Finding[];
  debates: Debate[];
  speculative: string[];
  sources: SourceInfo[];
  verification: {
    claims: Array<{
      statement: string;
      status: EpistemicStatus;
      confidence: number;
      evidence_type: EvidenceType;
    }>;
    summary: {
      established: number;
      tentative: number;
      contested: number;
      unverifiable: number;
    };
  } | null;
  extracted_claims: VerifiedClaim[];
  sub_queries: string[];
  depth: string;
  latency_ms: number;
  raw_report: string;
}

export interface ClassifyRequest {
  query: string;
}

export interface SearchRequest {
  query: string;
  domain?: string;
  top_k?: number;
}

export interface IngestURLRequest {
  url: string;
  source_tier?: string;
  doc_type?: string;
  title?: string;
  authors?: string[];
  chunk_strategy?: 'fixed' | 'semantic' | 'markdown';
  chunk_size?: number;
  overlap_pct?: number;
}

export interface HealthResponse {
  status: string;
  version: string;
  database: string;
  vectorize: string;
  r2: string;
  wiki?: string;
}

// ── D1 Row Types ──

export interface DocumentRow {
  id: string;
  source_url: string;
  source_tier: string;
  domain: string;
  doc_type: string;
  published_date: string | null;
  title: string | null;
  authors: string | null;  // JSON
  text_snippet: string | null;
  r2_key: string | null;
  chunk_index: number;
  total_chunks: number;
  metadata: string | null;  // JSON
  token_count: number;
  created_at: string;
  updated_at: string;
}

export interface SourceTierRule {
  id: string;
  domain_pattern: string;
  tier: string;
  doc_types: string | null;  // JSON
  boost_factor: number;
  max_age_days: number | null;
}
