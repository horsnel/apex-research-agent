"""
Database initialization script for Railway deployment.

Creates the apex_db database (if needed) and runs the schema.
Run before starting the API server.
"""

import asyncio
import os
import sys
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


async def init_database():
    """Create the apex_db database and schema on Railway."""
    import asyncpg
    
    # Step 1: Connect to default 'postgres' database to create apex_db
    db_url = os.getenv("DATABASE_URL", "")
    if not db_url:
        logger.error("DATABASE_URL not set")
        return False
    
    # Parse the DATABASE_URL to extract components
    # Expected: postgresql://user:pass@host:port/dbname
    from urllib.parse import urlparse, urlunparse
    
    parsed = urlparse(db_url)
    current_db = parsed.path.lstrip("/")
    
    # Build a URL pointing to the 'postgres' default database
    postgres_url = urlunparse(parsed._replace(path="/postgres"))
    
    # Build the target apex_db URL
    apex_db_url = urlunparse(parsed._replace(path="/apex_db"))
    
    logger.info(f"Connecting to PostgreSQL at {parsed.hostname}...")
    
    try:
        # Connect to default postgres database
        conn = await asyncio.wait_for(asyncpg.connect(postgres_url), timeout=10.0)
    except Exception as e:
        logger.error(f"Failed to connect to PostgreSQL: {e}")
        return False
    
    try:
        # Check if apex_db exists
        exists = await conn.fetchval(
            "SELECT 1 FROM pg_database WHERE datname = 'apex_db'"
        )
        
        if exists:
            logger.info("Database apex_db already exists")
        else:
            # Create apex_db
            await conn.execute('CREATE DATABASE apex_db')
            logger.info("Created database apex_db")
        
        # Check/create the apex user if it doesn't exist
        user_exists = await conn.fetchval(
            "SELECT 1 FROM pg_roles WHERE rolname = 'apex'"
        )
        if not user_exists:
            await conn.execute("CREATE USER apex WITH PASSWORD 'apex_secret'")
            logger.info("Created user apex")
        else:
            # Update password in case it's wrong
            await conn.execute("ALTER USER apex WITH PASSWORD 'apex_secret'")
            logger.info("Updated apex user password")
        
        # Grant privileges
        await conn.execute("GRANT ALL PRIVILEGES ON DATABASE apex_db TO apex")
        logger.info("Granted privileges on apex_db to apex")
        
    finally:
        await conn.close()
    
    # Step 2: Connect to apex_db and create schema
    try:
        conn = await asyncio.wait_for(asyncpg.connect(apex_db_url), timeout=10.0)
    except Exception as e:
        logger.error(f"Failed to connect to apex_db: {e}")
        return False
    
    try:
        # Enable extensions
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        logger.info("Enabled pgvector extension")
        
        try:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
            logger.info("Enabled pg_trgm extension")
        except Exception:
            logger.warning("pg_trgm extension not available (non-critical)")
        
        # Grant schema permissions
        await conn.execute("GRANT ALL ON SCHEMA public TO apex")
        
        # Create tables if they don't exist
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                source_url TEXT NOT NULL,
                source_tier TEXT CHECK (source_tier IN ('P1','P2','P3','UNV')),
                domain TEXT,
                doc_type TEXT CHECK (doc_type IN ('paper','article','legal','dataset','book','report','other')),
                published_date DATE,
                title TEXT,
                authors TEXT[],
                raw_text TEXT NOT NULL,
                content_vector VECTOR(768),
                metadata JSONB DEFAULT '{}',
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                chunk_index INT DEFAULT 0,
                total_chunks INT DEFAULT 1,
                UNIQUE(source_url, chunk_index)
            )
        """)
        logger.info("Created documents table")
        
        # Create indexes if they don't exist
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_documents_domain ON documents (domain)",
            "CREATE INDEX IF NOT EXISTS idx_documents_source_tier ON documents (source_tier)",
            "CREATE INDEX IF NOT EXISTS idx_documents_doc_type ON documents (doc_type)",
            "CREATE INDEX IF NOT EXISTS idx_documents_published_date ON documents (published_date DESC)",
            "CREATE INDEX IF NOT EXISTS idx_documents_metadata ON documents USING GIN (metadata)",
            "CREATE INDEX IF NOT EXISTS idx_documents_authors ON documents USING GIN (authors)",
        ]
        
        for idx_sql in indexes:
            try:
                await conn.execute(idx_sql)
            except Exception as e:
                logger.warning(f"Index creation warning: {e}")
        
        # Create HNSW vector index (works with any dataset size, unlike IVFFlat)
        try:
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_documents_content_vector ON documents
                USING hnsw (content_vector vector_cosine_ops)
            """)
            logger.info("Created HNSW vector index")
        except Exception as e:
            logger.warning(f"Vector index creation warning: {e}")
            logger.info("Vector search will use brute force (fine for small datasets)")
        
        # Create trigram index for BM25-like search
        try:
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_documents_raw_text_trgm ON documents 
                USING GIN (raw_text gin_trgm_ops)
            """)
        except Exception:
            logger.warning("Trigram index not created (pg_trgm may not be available)")
        
        # Add tsvector column for full-text search if not exists
        try:
            await conn.execute("""
                ALTER TABLE documents ADD COLUMN IF NOT EXISTS tsv tsvector
                GENERATED ALWAYS AS (to_tsvector('english', coalesce(title, '') || ' ' || coalesce(raw_text, ''))) STORED
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_documents_tsv ON documents USING GIN (tsv)
            """)
        except Exception as e:
            logger.warning(f"Full-text search setup warning: {e}")
        
        # Create ingest_log table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS ingest_log (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                source_url TEXT NOT NULL,
                status TEXT CHECK (status IN ('success', 'failed', 'skipped')),
                chunks_created INT DEFAULT 0,
                error_message TEXT,
                started_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                completed_at TIMESTAMP WITH TIME ZONE
            )
        """)
        
        # Create query_log table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS query_log (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                query_text TEXT NOT NULL,
                route TEXT CHECK (route IN ('rag', 'live', 'rag+live')),
                similarity_score FLOAT,
                answer_text TEXT,
                token_count INT,
                sources_used TEXT[],
                latency_ms INT,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
        """)
        
        # Drop existing search functions (to handle signature changes)
        try:
            await conn.execute("DROP FUNCTION IF EXISTS search_documents CASCADE")
        except Exception:
            pass
        try:
            await conn.execute("DROP FUNCTION IF EXISTS keyword_search CASCADE")
        except Exception:
            pass

        # Create search_documents function
        await conn.execute("""
            CREATE OR REPLACE FUNCTION search_documents(
                query_vector VECTOR(768),
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
                    d.id, d.source_url, d.source_tier, d.domain, d.doc_type,
                    d.title, d.authors, d.raw_text, d.metadata,
                    d.chunk_index, d.total_chunks,
                    1 - (d.content_vector <=> query_vector) AS similarity
                FROM documents d
                WHERE d.content_vector IS NOT NULL
              AND (1 - (d.content_vector <=> query_vector)) >= match_threshold
                  AND (filter_domain IS NULL OR d.domain = filter_domain)
                  AND (filter_tier IS NULL OR d.source_tier = filter_tier)
                  AND (filter_doc_type IS NULL OR d.doc_type = filter_doc_type)
                ORDER BY d.content_vector <=> query_vector
                LIMIT match_count;
            END;
            $$;
        """)
        logger.info("Created search_documents function")
        
        # Create keyword_search function
        await conn.execute("""
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
                    d.id, d.source_url, d.source_tier, d.domain,
                    d.title, d.raw_text,
                    ts_rank(d.tsv, plainto_tsquery('english', search_query)) AS rank
                FROM documents d
                WHERE d.tsv @@ plainto_tsquery('english', search_query)
                  AND (filter_tier IS NULL OR d.source_tier = filter_tier)
                ORDER BY rank DESC
                LIMIT match_count;
            END;
            $$;
        """)
        logger.info("Created keyword_search function")
        
        # Grant all table permissions to apex user
        await conn.execute("GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO apex")
        await conn.execute("GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO apex")
        await conn.execute("ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO apex")
        await conn.execute("ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO apex")
        
        logger.info("Database initialization complete!")
        
        # Verify
        count = await conn.fetchval("SELECT COUNT(*) FROM documents")
        logger.info(f"Database has {count} documents")
        
        return True
        
    except Exception as e:
        logger.error(f"Schema creation failed: {e}")
        return False
    finally:
        await conn.close()


if __name__ == "__main__":
    success = asyncio.run(init_database())
    if not success:
        logger.warning("Database initialization failed - API will start in DB-less mode")
        # Don't exit with error - let the API start anyway
    sys.exit(0)
