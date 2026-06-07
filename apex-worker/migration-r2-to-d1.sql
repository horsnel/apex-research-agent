-- Migration: R2 + Vectorize → D1-only storage
-- Adds content_text and embedding columns to both tables
-- content_text replaces R2 object storage for full document/page content
-- embedding replaces Vectorize index for vector similarity search

-- ── documents table ──
ALTER TABLE documents ADD COLUMN content_text TEXT;
ALTER TABLE documents ADD COLUMN embedding TEXT;

-- ── wiki_pages table ──
ALTER TABLE wiki_pages ADD COLUMN content_text TEXT;
ALTER TABLE wiki_pages ADD COLUMN embedding TEXT;
