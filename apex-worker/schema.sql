-- APEX Research Agent — D1 Schema (SQLite)
-- Replaces PostgreSQL + pgvector with D1 + Vectorize + R2
-- Vectors are stored in Vectorize, full text in R2, metadata in D1

-- ── Main documents table ──
-- Vectors live in Vectorize index, full text lives in R2 bucket
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,  -- UUID generated in Worker
    source_url TEXT NOT NULL,
    source_tier TEXT CHECK(source_tier IN ('P1','P2','P3','UNV')) DEFAULT 'UNV',
    domain TEXT,
    doc_type TEXT CHECK(doc_type IN ('paper','article','legal','dataset','book','report','other')) DEFAULT 'other',
    published_date TEXT,  -- ISO 8601 date string
    title TEXT,
    authors TEXT,         -- JSON array: '["Author A","Author B"]'
    text_snippet TEXT,    -- First 500 chars for preview / FTS5 source
    r2_key TEXT,          -- R2 object key: 'docs/{hash}/{chunk_index}.txt'
    chunk_index INTEGER DEFAULT 0,
    total_chunks INTEGER DEFAULT 1,
    metadata TEXT,        -- JSON object
    token_count INTEGER DEFAULT 0,
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now')),
    UNIQUE(source_url, chunk_index)
);

-- ── FTS5 virtual table for keyword search (replaces tsvector) ──
CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
    text_snippet,
    content='documents',
    content_rowid='rowid',
    tokenize='porter unicode61'
);

-- ── Triggers to keep FTS5 in sync ──
CREATE TRIGGER IF NOT EXISTS documents_ai AFTER INSERT ON documents BEGIN
    INSERT INTO documents_fts(rowid, text_snippet) VALUES (new.rowid, new.text_snippet);
END;

CREATE TRIGGER IF NOT EXISTS documents_ad AFTER DELETE ON documents BEGIN
    INSERT INTO documents_fts(documents_fts, rowid, text_snippet) VALUES('delete', old.rowid, old.text_snippet);
END;

CREATE TRIGGER IF NOT EXISTS documents_au AFTER UPDATE ON documents BEGIN
    INSERT INTO documents_fts(documents_fts, rowid, text_snippet) VALUES('delete', old.rowid, old.text_snippet);
    INSERT INTO documents_fts(rowid, text_snippet) VALUES (new.rowid, new.text_snippet);
END;

-- ── Source tier rules (from source_tiers.yaml) ──
CREATE TABLE IF NOT EXISTS source_tier_rules (
    id TEXT PRIMARY KEY,
    domain_pattern TEXT NOT NULL,
    tier TEXT NOT NULL CHECK(tier IN ('P1','P2','P3','UNV')),
    doc_types TEXT,       -- JSON array of valid doc types, or NULL for all
    boost_factor REAL DEFAULT 1.0,
    max_age_days INTEGER,
    UNIQUE(domain_pattern, tier)
);

-- ── Query cache ──
CREATE TABLE IF NOT EXISTS query_cache (
    id TEXT PRIMARY KEY,
    query_hash TEXT NOT NULL UNIQUE,
    query_text TEXT NOT NULL,
    route TEXT,
    answer TEXT,
    sources TEXT,         -- JSON
    model_used TEXT,
    similarity_score REAL,
    created_at TEXT DEFAULT (datetime('now')),
    expires_at TEXT
);

-- ── Ingest queue ──
CREATE TABLE IF NOT EXISTS ingest_queue (
    id TEXT PRIMARY KEY,
    url TEXT NOT NULL,
    source_type TEXT,     -- 'url', 'arxiv', 'pubmed', 'pdf'
    status TEXT DEFAULT 'pending' CHECK(status IN ('pending','processing','completed','failed')),
    chunks_created INTEGER DEFAULT 0,
    error TEXT,
    created_at TEXT DEFAULT (datetime('now')),
    completed_at TEXT
);

-- ── Ingest audit log ──
CREATE TABLE IF NOT EXISTS ingest_log (
    id TEXT PRIMARY KEY,
    source_url TEXT NOT NULL,
    status TEXT CHECK(status IN ('success', 'failed', 'skipped')),
    chunks_created INTEGER DEFAULT 0,
    error_message TEXT,
    started_at TEXT DEFAULT (datetime('now')),
    completed_at TEXT
);

-- ── Query log for analytics ──
CREATE TABLE IF NOT EXISTS query_log (
    id TEXT PRIMARY KEY,
    query_text TEXT NOT NULL,
    route TEXT CHECK(route IN ('rag', 'live', 'rag+live')),
    similarity_score REAL,
    answer_text TEXT,
    token_count INTEGER,
    sources_used TEXT,    -- JSON array of URLs
    model_used TEXT,
    latency_ms INTEGER,
    created_at TEXT DEFAULT (datetime('now'))
);

-- ── Indexes ──
CREATE INDEX IF NOT EXISTS idx_documents_domain ON documents(domain);
CREATE INDEX IF NOT EXISTS idx_documents_source_tier ON documents(source_tier);
CREATE INDEX IF NOT EXISTS idx_documents_doc_type ON documents(doc_type);
CREATE INDEX IF NOT EXISTS idx_documents_published_date ON documents(published_date);
CREATE INDEX IF NOT EXISTS idx_documents_r2_key ON documents(r2_key);
CREATE INDEX IF NOT EXISTS idx_documents_chunk ON documents(source_url, chunk_index);
CREATE INDEX IF NOT EXISTS idx_query_cache_hash ON query_cache(query_hash);
CREATE INDEX IF NOT EXISTS idx_query_cache_expires ON query_cache(expires_at);
CREATE INDEX IF NOT EXISTS idx_ingest_queue_status ON ingest_queue(status);
CREATE INDEX IF NOT EXISTS idx_ingest_log_status ON ingest_log(status);
CREATE INDEX IF NOT EXISTS idx_query_log_route ON query_log(route);
CREATE INDEX IF NOT EXISTS idx_query_log_created ON query_log(created_at);
CREATE INDEX IF NOT EXISTS idx_source_tier_rules_domain ON source_tier_rules(domain_pattern);
