/**
 * APEX 2.0 — LLM Wiki Type Definitions
 * All types for the persistent knowledge layer
 */

import { Env } from '../types';

// ── Wiki Page States ──

export type WikiPageState = 'draft' | 'active' | 'stale' | 'contradicted' | 'archived';

// ── Wiki Page ──

export interface WikiPage {
  id: string;
  title: string;
  slug: string;
  content: string;                 // Full markdown content (stored in R2, loaded on demand)
  state: WikiPageState;
  sourceHashes: string[];          // SHA-256 hashes of all source content
  sources: WikiSource[];
  entities: WikiEntity[];
  links: WikiLink[];
  metadata: Record<string, unknown>;
  createdAt: string;
  updatedAt: string;
  lastVerifiedAt: string | null;
  verificationCount: number;
  accessCount: number;
  version: number;
}

// ── Wiki Source ──

export interface WikiSource {
  url: string;
  tier: string;
  title: string;
  contentHash: string;
  ingestedAt: string;
  lastCheckedAt: string;
}

// ── Wiki Entity ──

export type WikiEntityType = 'person' | 'org' | 'tech' | 'concept' | 'location' | 'event';

export interface WikiEntity {
  name: string;
  type: WikiEntityType;
  mentions: number;
  firstSeen: string;
  lastSeen: string;
}

// ── Wiki Link ──

export type WikiLinkRelationType = 'related' | 'contradicts' | 'supports' | 'extends' | 'prerequisite';

export interface WikiLink {
  targetSlug: string;
  relationType: WikiLinkRelationType;
  context: string;
}

// ── Ingest Request/Result ──

export interface WikiIngestRequest {
  urls: string[];
  category?: string;
  forceReingest?: boolean;
}

export interface WikiIngestResult {
  pagesCreated: number;
  pagesUpdated: number;
  pagesUnchanged: number;
  errors: string[];
  totalCostMs: number;
}

// ── Query Request/Result ──

export interface WikiQueryRequest {
  query: string;
  includeContradictions?: boolean;
  maxPages?: number;
  freshnessThreshold?: number;    // Max age in hours before considering stale
}

export interface WikiQueryResult {
  answer: string;
  pagesUsed: string[];            // Slugs of pages used
  stalePages: string[];           // Slugs of stale pages found
  contradictionAlerts: ContradictionAlert[];
  costSavedMs: number;            // Estimated time saved vs full RAG
}

// ── Hot Cache ──

export interface HotCacheEntry {
  sessionId: string;
  userId: string;
  lastQuery: string;
  lastContext: string;
  recentTopics: string[];
  recentSources: string[];
  sessionSummary: string;
  updatedAt: string;
}

// ── Page Lifecycle ──

export interface PageLifecycleEvent {
  pageId: string;
  fromState: WikiPageState;
  toState: WikiPageState;
  reason: string;
  sourceHash: string | null;
  timestamp: string;
}

// ── Wiki Schema ──

export interface WikiSchema {
  wikiId: string;
  name: string;
  description: string;
  behaviorRules: string[];
  outputFormat: string;
  entityTypes: string[];
  linkTypes: string[];
}

// ── Contradiction Alert ──

export interface ContradictionAlert {
  pageA: string;
  pageB: string;
  conflictingClaims: string[];
  detectedAt: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
}

// ── Knowledge Graph Types (SciMem) ──

export type KnowledgeEntityType = 'Topic' | 'Paper' | 'Company' | 'Person' | 'Technology' | 'Market' | 'Concept' | 'Method' | 'Event' | 'Location';

export type KnowledgeRelationType =
  | 'relates_to'
  | 'cites'
  | 'authored_by'
  | 'competes_with'
  | 'precedes'
  | 'extends'
  | 'contradicts'
  | 'supports'
  | 'uses'
  | 'part_of';

export interface KnowledgeEntity {
  id: string;
  name: string;
  type: KnowledgeEntityType;
  description: string;
  mentionCount: number;
  firstSeen: string;
  lastSeen: string;
  properties: Record<string, unknown>;
}

export interface KnowledgeRelation {
  id: string;
  fromEntityId: string;
  toEntityId: string;
  relationType: KnowledgeRelationType;
  context: string;
}

export interface KnowledgeSubgraph {
  entities: KnowledgeEntity[];
  relations: KnowledgeRelation[];
}

// ── Provenance Types ──

export interface ProvenanceClaim {
  id: string;
  statement: string;
  sourceUrl: string;
  sourceTier: string;
  confidence: number;
  extractionMethod: string;
  extractedAt: string;
  costToProduce: number;
  verificationStatus: 'unverified' | 'supported' | 'conflicted' | 'resolved';
  supportingClaims: string[];
  conflictingClaims: string[];
}

export interface ProvenanceAuditReport {
  pageId: string;
  claims: ProvenanceClaim[];
  totalClaims: number;
  verifiedClaims: number;
  conflictedClaims: number;
  totalCost: number;
}

// ── Dialogic Wiki Types ──

export interface ContradictionPosition {
  pageId: string;
  claim: string;
  sources: string[];
  confidence: number;
  jurisdiction: string;
  context: string;
}

export interface ContradictionRecord {
  id: string;
  topic: string;
  positions: ContradictionPosition[];
  severity: 'low' | 'medium' | 'high' | 'critical';
  status: 'detected' | 'analyzing' | 'preserved' | 'resolved' | 'superseded';
  detectedAt: string;
  resolvedAt: string | null;
}

export interface DialecticalSummary {
  topic: string;
  positions: Array<{
    label: string;
    claim: string;
    evidence: string[];
    confidence: number;
  }>;
  disagreement: string;
  stakes: string;
  resolutionPaths: string[];
}

// ── Security Types ──

export type TrustTier = 'untrusted' | 'external' | 'partner' | 'internal' | 'verified';

export interface SecurityScanResult {
  isSafe: boolean;
  threats: string[];
  injectionDetected: boolean;
  trustScore: number;
  recommendations: string[];
}

// ── Concurrency Types ──

export interface LockAcquisition {
  lockId: string;
  pageId: string;
  holder: string;
  acquiredAt: string;
  expiresAt: string;
}

export type ConflictResolutionStrategy = 'last_writer_wins' | 'merge' | 'abort';

// ── D1 Row Types ──

export interface WikiPageRow {
  id: string;
  slug: string;
  title: string;
  content_snippet: string | null;
  content_text: string | null;        // Full markdown content (replaces R2)
  embedding: string | null;           // JSON array of floats (replaces Vectorize)
  state: string;
  category: string | null;
  source_hashes: string | null;
  sources: string | null;
  entities: string | null;
  links: string | null;
  metadata: string | null;
  schema_id: string | null;
  created_at: string;
  updated_at: string;
  last_verified_at: string | null;
  verification_count: number;
  access_count: number;
  version: number;
}

export interface WikiSourceRow {
  id: string;
  url: string;
  content_hash: string;
  tier: string | null;
  title: string | null;
  trust_tier: string | null;
  first_ingested_at: string;
  last_checked_at: string;
  page_ids: string | null;
}

export interface WikiSessionRow {
  id: string;
  user_id: string | null;
  last_query: string | null;
  last_context: string | null;
  recent_topics: string | null;
  recent_sources: string | null;
  session_summary: string | null;
  created_at: string;
  updated_at: string;
}

export interface WikiEntityRow {
  id: string;
  name: string;
  type: string;
  description: string | null;
  mention_count: number;
  first_seen: string;
  last_seen: string;
  properties: string | null;
  created_at: string;
  updated_at: string;
}

export interface WikiRelationRow {
  id: string;
  from_entity_id: string;
  to_entity_id: string;
  relation_type: string;
  context: string | null;
  created_at: string;
  updated_at: string;
}

export interface WikiProvenanceClaimRow {
  id: string;
  page_id: string;
  statement: string;
  source_url: string;
  source_tier: string | null;
  confidence: number;
  extraction_method: string;
  extracted_at: string;
  cost_to_produce: number;
  verification_status: string;
  supporting_claim_ids: string | null;
  conflicting_claim_ids: string | null;
  created_at: string;
  updated_at: string;
}

export interface WikiContradictionRow {
  id: string;
  topic: string;
  positions: string;
  severity: string;
  status: string;
  detected_at: string;
  resolved_at: string | null;
  resolution: string | null;
  created_at: string;
  updated_at: string;
}

export interface WikiSecurityLogRow {
  id: string;
  page_id: string | null;
  event_type: string;
  details: string | null;
  trust_tier: string | null;
  threat_level: string;
  created_at: string;
}

export interface WikiLockRow {
  id: string;
  page_id: string;
  holder: string;
  acquired_at: string;
  expires_at: string;
  released_at: string | null;
}

export interface WikiLifecycleEventRow {
  id: string;
  page_id: string;
  from_state: string;
  to_state: string;
  reason: string | null;
  source_hash: string | null;
  created_at: string;
}

export interface WikiSchemaRow {
  id: string;
  name: string;
  description: string | null;
  behavior_rules: string | null;
  output_format: string | null;
  entity_types: string | null;
  link_types: string | null;
  created_at: string;
  updated_at: string;
}
