-- APEX Research Agent — Database Schema
-- PostgreSQL + pgvector for hybrid RAG + live scraper system

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;  -- For BM25-like trigram search

-- ── Main documents table ──
CREATE TABLE documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_url TEXT NOT NULL,
    source_tier TEXT CHECK (source_tier IN ('P1','P2','P3','UNV')),
    domain TEXT,
    doc_type TEXT CHECK (doc_type IN ('paper','article','legal','dataset','book','report','other')),
    published_date DATE,
    title TEXT,
    authors TEXT[],
    raw_text TEXT NOT NULL,
    content_vector VECTOR(1536),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    chunk_index INT DEFAULT 0,
    total_chunks INT DEFAULT 1,

    -- Prevent duplicate chunks for the same source
    UNIQUE(source_url, chunk_index)
);

-- ── Indexes ──
CREATE INDEX idx_documents_domain ON documents (domain);
CREATE INDEX idx_documents_source_tier ON documents (source_tier);
CREATE INDEX idx_documents_doc_type ON documents (doc_type);
CREATE INDEX idx_documents_published_date ON documents (published_date DESC);
CREATE INDEX idx_documents_metadata ON documents USING GIN (metadata);
CREATE INDEX idx_documents_authors ON documents USING GIN (authors);
CREATE INDEX idx_documents_raw_text_trgm ON documents USING GIN (raw_text gin_trgm_ops);

-- IVFFlat index for vector similarity (build after seeding with representative data)
-- Adjust lists count based on expected row count: sqrt(rows) is a good starting point
CREATE INDEX idx_documents_content_vector ON documents
    USING ivfflat (content_vector vector_cosine_ops)
    WITH (lists = 100);

-- ── Full-text search index for BM25-like keyword matching ──
ALTER TABLE documents ADD COLUMN tsv tsvector
    GENERATED ALWAYS AS (to_tsvector('english', coalesce(title, '') || ' ' || coalesce(raw_text, ''))) STORED;
CREATE INDEX idx_documents_tsv ON documents USING GIN (tsv);

-- ── Ingest audit log ──
CREATE TABLE ingest_log (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_url TEXT NOT NULL,
    status TEXT CHECK (status IN ('success', 'failed', 'skipped')),
    chunks_created INT DEFAULT 0,
    error_message TEXT,
    started_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    completed_at TIMESTAMP WITH TIME ZONE
);

CREATE INDEX idx_ingest_log_status ON ingest_log (status);
CREATE INDEX idx_ingest_log_started ON ingest_log (started_at DESC);

-- ── Query log for analytics and feedback ──
CREATE TABLE query_log (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    query_text TEXT NOT NULL,
    route TEXT CHECK (route IN ('rag', 'live', 'rag+live')),
    similarity_score FLOAT,
    answer_text TEXT,
    token_count INT,
    sources_used TEXT[],
    latency_ms INT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_query_log_route ON query_log (route);
CREATE INDEX idx_query_log_created ON query_log (created_at DESC);

-- ── Helper: cosine similarity search function ──
CREATE OR REPLACE FUNCTION search_documents(
    query_vector VECTOR(1536),
    match_threshold FLOAT DEFAULT 0.72,
    match_count INT DEFAULT 5,
    filter_domain TEXT DEFAULT NULL,
    filter_tier TEXT DEFAULT NULL,
    filter_doc_type TEXT DEFAULT NULL
)
RETURNS TABLE (
    id UUID,
    source_url TEXT,
    source_tier TEXT,
    domain TEXT,
    doc_type TEXT,
    title TEXT,
    authors TEXT[],
    raw_text TEXT,
    metadata JSONB,
    chunk_index INT,
    total_chunks INT,
    similarity FLOAT
)
LANGUAGE plpgsql STABLE
AS $$
BEGIN
    RETURN QUERY
    SELECT
        d.id,
        d.source_url,
        d.source_tier,
        d.domain,
        d.doc_type,
        d.title,
        d.authors,
        d.raw_text,
        d.metadata,
        d.chunk_index,
        d.total_chunks,
        1 - (d.content_vector <=> query_vector) AS similarity
    FROM documents d
    WHERE (1 - (d.content_vector <=> query_vector)) >= match_threshold
      AND (filter_domain IS NULL OR d.domain = filter_domain)
      AND (filter_tier IS NULL OR d.source_tier = filter_tier)
      AND (filter_doc_type IS NULL OR d.doc_type = filter_doc_type)
    ORDER BY d.content_vector <=> query_vector
    LIMIT match_count;
END;
$$;

-- ── Helper: BM25 keyword search (tsvector ranking) ──
CREATE OR REPLACE FUNCTION keyword_search(
    search_query TEXT,
    match_count INT DEFAULT 5,
    filter_tier TEXT DEFAULT NULL
)
RETURNS TABLE (
    id UUID,
    source_url TEXT,
    source_tier TEXT,
    domain TEXT,
    title TEXT,
    raw_text TEXT,
    rank FLOAT
)
LANGUAGE plpgsql STABLE
AS $$
BEGIN
    RETURN QUERY
    SELECT
        d.id,
        d.source_url,
        d.source_tier,
        d.domain,
        d.title,
        d.raw_text,
        ts_rank(d.tsv, plainto_tsquery('english', search_query)) AS rank
    FROM documents d
    WHERE d.tsv @@ plainto_tsquery('english', search_query)
      AND (filter_tier IS NULL OR d.source_tier = filter_tier)
    ORDER BY rank DESC
    LIMIT match_count;
END;
$$;

-- ── Auto-update updated_at ──
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER documents_updated_at
    BEFORE UPDATE ON documents
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at();
