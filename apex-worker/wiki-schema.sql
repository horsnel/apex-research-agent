-- APEX 2.0 — LLM Wiki D1 Schema (SQLite)
-- Persistent knowledge layer: wiki pages, knowledge graph, provenance, contradictions,
-- security audit, concurrency locks, and session hot cache
-- D1-only storage: full content and embeddings stored in D1 (no R2/Vectorize)

-- ── Wiki Pages ──
-- Full markdown content stored in content_text column (D1-only, replaces R2)
-- Embeddings stored in embedding column as JSON (D1-only, replaces Vectorize)
-- D1 stores metadata and a text snippet for FTS5
CREATE TABLE IF NOT EXISTS wiki_pages (
    id TEXT PRIMARY KEY,
    slug TEXT NOT NULL UNIQUE,
    title TEXT NOT NULL,
    content_snippet TEXT,          -- First 500 chars for preview / FTS5
    content_text TEXT,             -- Full markdown content (replaces R2)
    embedding TEXT,                -- JSON array of floats (replaces Vectorize)
    state TEXT NOT NULL CHECK(state IN ('draft','active','stale','contradicted','archived')) DEFAULT 'draft',
    category TEXT,
    source_hashes TEXT,           -- JSON array of SHA-256 hashes: '["abc123","def456"]'
    sources TEXT,                 -- JSON array of WikiSource objects
    entities TEXT,                -- JSON array of WikiEntity objects
    links TEXT,                   -- JSON array of WikiLink objects
    metadata TEXT,                -- JSON object
    schema_id TEXT,               -- References wiki_schemas.id
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now')),
    last_verified_at TEXT,
    verification_count INTEGER DEFAULT 0,
    access_count INTEGER DEFAULT 0,
    version INTEGER DEFAULT 1
);

-- ── FTS5 for wiki pages ──
CREATE VIRTUAL TABLE IF NOT EXISTS wiki_pages_fts USING fts5(
    title,
    content_snippet,
    content='wiki_pages',
    content_rowid='rowid',
    tokenize='porter unicode61'
);

-- ── FTS5 sync triggers ──
CREATE TRIGGER IF NOT EXISTS wiki_pages_ai AFTER INSERT ON wiki_pages BEGIN
    INSERT INTO wiki_pages_fts(rowid, title, content_snippet) VALUES (new.rowid, new.title, new.content_snippet);
END;

CREATE TRIGGER IF NOT EXISTS wiki_pages_ad AFTER DELETE ON wiki_pages BEGIN
    INSERT INTO wiki_pages_fts(wiki_pages_fts, rowid, title, content_snippet) VALUES('delete', old.rowid, old.title, old.content_snippet);
END;

CREATE TRIGGER IF NOT EXISTS wiki_pages_au AFTER UPDATE ON wiki_pages BEGIN
    INSERT INTO wiki_pages_fts(wiki_pages_fts, rowid, title, content_snippet) VALUES('delete', old.rowid, old.title, old.content_snippet);
    INSERT INTO wiki_pages_fts(rowid, title, content_snippet) VALUES (new.rowid, new.title, new.content_snippet);
END;

-- ── Wiki Sources ──
CREATE TABLE IF NOT EXISTS wiki_sources (
    id TEXT PRIMARY KEY,
    url TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    tier TEXT,
    title TEXT,
    trust_tier TEXT CHECK(trust_tier IN ('untrusted','external','partner','internal','verified')),
    first_ingested_at TEXT DEFAULT (datetime('now')),
    last_checked_at TEXT DEFAULT (datetime('now')),
    page_ids TEXT,                -- JSON array of wiki page IDs this source contributes to
    UNIQUE(url, content_hash)
);

-- ── Wiki Sessions (Hot Cache) ──
CREATE TABLE IF NOT EXISTS wiki_sessions (
    id TEXT PRIMARY KEY,
    user_id TEXT,
    last_query TEXT,
    last_context TEXT,
    recent_topics TEXT,           -- JSON array of strings
    recent_sources TEXT,          -- JSON array of strings
    session_summary TEXT,
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now'))
);

-- ── Wiki Entities (Knowledge Graph) ──
CREATE TABLE IF NOT EXISTS wiki_entities (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    type TEXT NOT NULL CHECK(type IN ('Topic','Paper','Company','Person','Technology','Market','Concept','Method','Event','Location')),
    description TEXT,
    mention_count INTEGER DEFAULT 1,
    first_seen TEXT DEFAULT (datetime('now')),
    last_seen TEXT DEFAULT (datetime('now')),
    properties TEXT,              -- JSON object for arbitrary properties
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now')),
    UNIQUE(name, type)
);

-- ── Wiki Relations (Knowledge Graph Edges) ──
CREATE TABLE IF NOT EXISTS wiki_relations (
    id TEXT PRIMARY KEY,
    from_entity_id TEXT NOT NULL,
    to_entity_id TEXT NOT NULL,
    relation_type TEXT NOT NULL CHECK(relation_type IN (
        'relates_to','cites','authored_by','competes_with','precedes','extends',
        'contradicts','supports','uses','part_of'
    )),
    context TEXT,
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now')),
    UNIQUE(from_entity_id, to_entity_id, relation_type)
);

-- ── Wiki Provenance Claims ──
CREATE TABLE IF NOT EXISTS wiki_provenance_claims (
    id TEXT PRIMARY KEY,
    page_id TEXT NOT NULL,
    statement TEXT NOT NULL,
    source_url TEXT NOT NULL,
    source_tier TEXT,
    confidence REAL DEFAULT 0.5,
    extraction_method TEXT DEFAULT 'llm',
    extracted_at TEXT DEFAULT (datetime('now')),
    cost_to_produce REAL DEFAULT 0.0,
    verification_status TEXT CHECK(verification_status IN ('unverified','supported','conflicted','resolved')) DEFAULT 'unverified',
    supporting_claim_ids TEXT,    -- JSON array of claim IDs
    conflicting_claim_ids TEXT,   -- JSON array of claim IDs
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now'))
);

-- ── Wiki Contradictions ──
CREATE TABLE IF NOT EXISTS wiki_contradictions (
    id TEXT PRIMARY KEY,
    topic TEXT NOT NULL,
    positions TEXT NOT NULL,      -- JSON array of ContradictionPosition objects
    severity TEXT NOT NULL CHECK(severity IN ('low','medium','high','critical')) DEFAULT 'medium',
    status TEXT NOT NULL CHECK(status IN ('detected','analyzing','preserved','resolved','superseded')) DEFAULT 'detected',
    detected_at TEXT DEFAULT (datetime('now')),
    resolved_at TEXT,
    resolution TEXT,              -- JSON object with resolution details
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now'))
);

-- ── Wiki Security Log ──
CREATE TABLE IF NOT EXISTS wiki_security_log (
    id TEXT PRIMARY KEY,
    page_id TEXT,
    event_type TEXT NOT NULL,     -- 'scan', 'review', 'quarantine', 'trust_assign', 'conflict_resolve'
    details TEXT,                 -- JSON object with event-specific details
    trust_tier TEXT CHECK(trust_tier IN ('untrusted','external','partner','internal','verified')),
    threat_level TEXT CHECK(threat_level IN ('none','low','medium','high','critical')) DEFAULT 'none',
    created_at TEXT DEFAULT (datetime('now'))
);

-- ── Wiki Locks (Concurrency) ──
CREATE TABLE IF NOT EXISTS wiki_locks (
    id TEXT PRIMARY KEY,
    page_id TEXT NOT NULL,
    holder TEXT NOT NULL,
    acquired_at TEXT DEFAULT (datetime('now')),
    expires_at TEXT NOT NULL,
    released_at TEXT
);

-- ── Wiki Lifecycle Events ──
CREATE TABLE IF NOT EXISTS wiki_lifecycle_events (
    id TEXT PRIMARY KEY,
    page_id TEXT NOT NULL,
    from_state TEXT NOT NULL,
    to_state TEXT NOT NULL,
    reason TEXT,
    source_hash TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);

-- ── Wiki Schemas ──
CREATE TABLE IF NOT EXISTS wiki_schemas (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    description TEXT,
    behavior_rules TEXT,          -- JSON array of strings
    output_format TEXT,
    entity_types TEXT,            -- JSON array of entity type names
    link_types TEXT,              -- JSON array of link type names
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now'))
);

-- ═══════════════════════════════════════
-- INDEXES
-- ═══════════════════════════════════════

-- Wiki pages
CREATE INDEX IF NOT EXISTS idx_wiki_pages_slug ON wiki_pages(slug);
CREATE INDEX IF NOT EXISTS idx_wiki_pages_state ON wiki_pages(state);
CREATE INDEX IF NOT EXISTS idx_wiki_pages_category ON wiki_pages(category);
CREATE INDEX IF NOT EXISTS idx_wiki_pages_last_verified ON wiki_pages(last_verified_at);
CREATE INDEX IF NOT EXISTS idx_wiki_pages_created_at ON wiki_pages(created_at);
CREATE INDEX IF NOT EXISTS idx_wiki_pages_schema_id ON wiki_pages(schema_id);

-- Wiki sources
CREATE INDEX IF NOT EXISTS idx_wiki_sources_url ON wiki_sources(url);
CREATE INDEX IF NOT EXISTS idx_wiki_sources_content_hash ON wiki_sources(content_hash);
CREATE INDEX IF NOT EXISTS idx_wiki_sources_tier ON wiki_sources(tier);
CREATE INDEX IF NOT EXISTS idx_wiki_sources_trust_tier ON wiki_sources(trust_tier);

-- Wiki sessions
CREATE INDEX IF NOT EXISTS idx_wiki_sessions_user_id ON wiki_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_wiki_sessions_updated_at ON wiki_sessions(updated_at);

-- Wiki entities
CREATE INDEX IF NOT EXISTS idx_wiki_entities_name ON wiki_entities(name);
CREATE INDEX IF NOT EXISTS idx_wiki_entities_type ON wiki_entities(type);
CREATE INDEX IF NOT EXISTS idx_wiki_entities_name_type ON wiki_entities(name, type);
CREATE INDEX IF NOT EXISTS idx_wiki_entities_mention_count ON wiki_entities(mention_count);

-- Wiki relations
CREATE INDEX IF NOT EXISTS idx_wiki_relations_from ON wiki_relations(from_entity_id);
CREATE INDEX IF NOT EXISTS idx_wiki_relations_to ON wiki_relations(to_entity_id);
CREATE INDEX IF NOT EXISTS idx_wiki_relations_type ON wiki_relations(relation_type);
CREATE INDEX IF NOT EXISTS idx_wiki_relations_from_type ON wiki_relations(from_entity_id, relation_type);

-- Wiki provenance claims
CREATE INDEX IF NOT EXISTS idx_wiki_provenance_page_id ON wiki_provenance_claims(page_id);
CREATE INDEX IF NOT EXISTS idx_wiki_provenance_source_url ON wiki_provenance_claims(source_url);
CREATE INDEX IF NOT EXISTS idx_wiki_provenance_verification ON wiki_provenance_claims(verification_status);
CREATE INDEX IF NOT EXISTS idx_wiki_provenance_confidence ON wiki_provenance_claims(confidence);

-- Wiki contradictions
CREATE INDEX IF NOT EXISTS idx_wiki_contradictions_topic ON wiki_contradictions(topic);
CREATE INDEX IF NOT EXISTS idx_wiki_contradictions_severity ON wiki_contradictions(severity);
CREATE INDEX IF NOT EXISTS idx_wiki_contradictions_status ON wiki_contradictions(status);
CREATE INDEX IF NOT EXISTS idx_wiki_contradictions_detected_at ON wiki_contradictions(detected_at);

-- Wiki security log
CREATE INDEX IF NOT EXISTS idx_wiki_security_page_id ON wiki_security_log(page_id);
CREATE INDEX IF NOT EXISTS idx_wiki_security_event_type ON wiki_security_log(event_type);
CREATE INDEX IF NOT EXISTS idx_wiki_security_created_at ON wiki_security_log(created_at);

-- Wiki locks
CREATE INDEX IF NOT EXISTS idx_wiki_locks_page_id ON wiki_locks(page_id);
CREATE INDEX IF NOT EXISTS idx_wiki_locks_expires_at ON wiki_locks(expires_at);
CREATE INDEX IF NOT EXISTS idx_wiki_locks_holder ON wiki_locks(holder);
CREATE INDEX IF NOT EXISTS idx_wiki_locks_active ON wiki_locks(page_id, expires_at) WHERE released_at IS NULL;

-- Wiki lifecycle events
CREATE INDEX IF NOT EXISTS idx_wiki_lifecycle_page_id ON wiki_lifecycle_events(page_id);
CREATE INDEX IF NOT EXISTS idx_wiki_lifecycle_created_at ON wiki_lifecycle_events(created_at);
CREATE INDEX IF NOT EXISTS idx_wiki_lifecycle_from_state ON wiki_lifecycle_events(from_state);
CREATE INDEX IF NOT EXISTS idx_wiki_lifecycle_to_state ON wiki_lifecycle_events(to_state);
