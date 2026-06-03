"""
APEX Research Agent — Main FastAPI Application.

Orchestrates the full pipeline:
Query → Classify → Retrieve (RAG) → [Fallback: Live Scrape] → Synthesize → Answer

Also exposes:
- /ingest endpoints for document ingestion
- /mcp endpoints for MCP protocol
- /health for monitoring
"""

import asyncio
import json
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.query_classifier import classify_query, ClassificationResult, should_escalate_to_live
from agent.retriever import retrieve, RetrievedChunk
from agent.synthesizer import synthesize, SynthesisResult
from agent.llm_router import get_router_status, test_all_models, reload_config, FALLBACK_CHAIN
from tools.live_scraper import live_scrape, ScrapeResult
from tools.citation_validator import validate_citations, ValidationResult
from ingest.chunker import chunk_text
from ingest.embedder import embed_and_upsert, embed_unembedded_chunks

# ── Logging ──
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("apex")

# ── Lifespan ──
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown lifecycle."""
    logger.info("APEX Research Agent starting up...")
    yield
    logger.info("APEX Research Agent shutting down.")


# ── FastAPI App ──
app = FastAPI(
    title="APEX Research Agent",
    description="Token-efficient hybrid RAG + Live Scraper research AI",
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS: Allow kovira.pages.dev + local dev ──
ALLOWED_ORIGINS = [
    "https://kovira.pages.dev",
    "https://www.kovira.pages.dev",
    "http://localhost:3000",   # Local Next.js dev
    "http://localhost:8000",   # Local API docs
    "http://127.0.0.1:3000",
    "http://127.0.0.1:8000",
]
# In production, restrict to known origins.
# Set CORS_WILDCARD=true in .env to allow all (for dev/preview deploys).
if os.getenv("CORS_WILDCARD", "false").lower() == "true":
    ALLOWED_ORIGINS = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ═══════════════════════════════════════
# REQUEST/RESPONSE MODELS
# ═══════════════════════════════════════

class QueryRequest(BaseModel):
    """Main research query request."""
    query: str = Field(..., min_length=1, max_length=2000, description="Research query")
    force_live: bool = Field(False, description="Force live scraping bypass")
    domain_filter: Optional[str] = Field(None, description="Filter by domain")
    tier_filter: Optional[str] = Field(None, description="Filter by tier (P1/P2/P3/UNV)")
    max_tokens: Optional[int] = Field(None, ge=50, le=500, description="Override max output tokens")
    depth: str = Field("quick", pattern=r"^(quick|thorough)$", description="Research depth: quick (1 cycle) or thorough (iterative)")


class QueryResponse(BaseModel):
    """Research query response."""
    answer: str
    route: str  # "rag", "live", "rag+live"
    method: str  # "pass_through", "synthesis", "table", "raw_context"
    sources: list
    token_count: int
    latency_ms: int
    similarity_score: Optional[float] = None
    validation: Optional[dict] = None
    model_used: str = ""
    provider: str = ""
    fallback_count: int = 0
    verification: Optional[dict] = None  # Epistemic verification from research engine


class IngestURLRequest(BaseModel):
    """Ingest a URL into the corpus."""
    url: str
    source_tier: str = Field("UNV", pattern=r"^(P1|P2|P3|UNV)$")
    doc_type: str = Field("article", pattern=r"^(paper|article|legal|dataset|book|report|other)$")
    title: Optional[str] = None
    authors: Optional[list] = None
    chunk_strategy: str = Field("semantic", pattern=r"^(fixed|semantic|markdown)$")
    chunk_size: int = Field(512, ge=128, le=2048)
    overlap_pct: float = Field(0.20, ge=0.0, le=0.50)


class IngestArxivRequest(BaseModel):
    """Ingest arXiv papers."""
    arxiv_id: Optional[str] = None
    category: Optional[str] = None
    max_results: int = Field(25, ge=1, le=100)


class IngestPubMedRequest(BaseModel):
    """Ingest PubMed papers."""
    query: str
    max_results: int = Field(25, ge=1, le=100)


class IngestPDFRequest(BaseModel):
    """Ingest a PDF."""
    url: str
    source_tier: str = Field("UNV", pattern=r"^(P1|P2|P3|UNV)$")
    doc_type: str = Field("paper", pattern=r"^(paper|article|legal|dataset|book|report|other)$")
    title: Optional[str] = None
    authors: Optional[list] = None


class IngestResponse(BaseModel):
    """Response from ingest endpoints."""
    status: str
    chunks_upserted: int
    message: str


class ClassifyRequest(BaseModel):
    """Query classification request."""
    query: str


class ClassifyResponse(BaseModel):
    """Query classification response."""
    route: str
    reason: str
    domain_hint: str
    confidence: float
    method: str


class SearchRequest(BaseModel):
    """Direct corpus search request."""
    query: str
    domain: Optional[str] = None
    top_k: int = Field(5, ge=1, le=20)


class SearchResponse(BaseModel):
    """Direct corpus search response."""
    results: list
    avg_similarity: float
    count: int


class ScrapeRequest(BaseModel):
    """Live scrape request."""
    query: Optional[str] = None
    urls: Optional[list] = None


class ScrapeResponse(BaseModel):
    """Live scrape response."""
    results: list
    successful: int
    total: int


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    database: str


class ResearchRequest(BaseModel):
    """Deep research request with structured report output."""
    query: str = Field(..., min_length=1, max_length=2000, description="Research query")
    classification: str = Field("academic", description="Query type: academic, web, code, news, clinical, encyclopedia, compute, patent")
    depth: str = Field("quick", pattern=r"^(quick|thorough)$", description="Research depth: quick (1 cycle) or thorough (iterative)")
    verify: bool = Field(True, description="Verify claims with epistemic marking")
    extract: bool = Field(False, description="Extract structured claims from P1 sources")
    check_retractions: bool = Field(False, description="Check DOIs for retraction status")


class ResearchResponse(BaseModel):
    """Structured research report response."""
    query: str
    executive_summary: str = ""
    findings: list = []
    debates: list = []
    speculative: list = []
    sources: list = []
    verification: Optional[dict] = None
    extracted_claims: list = []
    retractions: list = []
    sub_queries: list = []
    depth: str = "quick"
    latency_ms: int = 0
    raw_report: str = ""


# ═══════════════════════════════════════
# CORE PIPELINE
# ═══════════════════════════════════════

async def run_pipeline(request: QueryRequest) -> QueryResponse:
    """
    Execute the full APEX research pipeline.

    Flow:
    1. Classify query → route to RAG or live
    2. Retrieve from vector DB (always attempted first)
    3. Enforce source tiers + temporal decay (Upgrade #3)
    4. If low confidence → live scrape fallback
    5. Synthesize answer with token cap
    6. Verify claims with epistemic marking (Upgrade #1)
    7. Validate citations
    8. Return result
    """
    from agent.research_engine import (
        enforce_source_tier, apply_temporal_decay,
        verify_claims_from_sources, _extract_claims_from_sources,
    )

    start_time = time.time()
    route = "rag"
    chunks: list[RetrievedChunk] = []
    scraped_text: Optional[str] = None
    similarity_score: Optional[float] = None
    verification_data = None

    # Step 1: Classify
    if request.force_live:
        classification = ClassificationResult(
            route="live",
            reason="Forced live mode",
            confidence=1.0,
            method="override",
        )
    else:
        classification = await classify_query(request.query)

    # Step 2: RAG retrieval (always attempt unless forced live-only)
    if classification.route in ("rag", "live"):
        chunks, avg_sim = await retrieve(
            query=request.query,
            domain_filter=request.domain_filter,
            tier_filter=request.tier_filter,
        )
        similarity_score = avg_sim

        # Upgrade #3: Enforce source tier by domain
        for chunk in chunks:
            chunk.source_tier = enforce_source_tier(chunk.source_url, chunk.source_tier)

        # Check if RAG is sufficient
        if classification.route == "rag" and not should_escalate_to_live(avg_sim, any(c.source_tier == "P1" for c in chunks)):
            route = "rag"
        elif classification.route == "rag" and should_escalate_to_live(avg_sim, any(c.source_tier == "P1" for c in chunks)):
            route = "rag+live"
        else:
            route = "live"

    # Step 3: Live scrape fallback
    if route in ("live", "rag+live"):
        scrape_results = await live_scrape(query=request.query)
        if scrape_results:
            successful_results = [r for r in scrape_results if r.success]
            if successful_results:
                scraped_text = "\n\n".join(r.markdown for r in successful_results)

    # Step 4: Synthesize
    synthesis = await synthesize(
        query=request.query,
        chunks=chunks,
        scraped_text=scraped_text,
    )

    # Step 5: Verify claims with epistemic marking (Upgrade #1)
    try:
        sources_for_verification = [
            {
                "url": c.source_url,
                "tier": c.source_tier,
                "title": c.title,
                "snippet": c.raw_text[:300],
                "authors": c.authors,
            }
            for c in chunks
        ]
        # Also add scraped content as a source if available
        if scraped_text:
            sources_for_verification.append({
                "url": "live_scrape",
                "tier": "P3",
                "title": "Live web content",
                "snippet": scraped_text[:500],
                "authors": [],
            })

        if sources_for_verification:
            claims = _extract_claims_from_sources(sources_for_verification)
            if claims:
                verification = verify_claims_from_sources(claims, sources_for_verification)
                verification_data = {
                    "claims": [
                        {
                            "statement": c.statement[:150],
                            "status": c.epistemic_status,
                            "confidence": c.confidence,
                            "evidence_type": c.evidence_type,
                        }
                        for c in verification.claims
                    ],
                    "summary": {
                        "established": verification.established_count,
                        "tentative": verification.tentative_count,
                        "contested": verification.contested_count,
                        "unverifiable": verification.unverifiable_count,
                    },
                }
    except Exception as e:
        logger.debug(f"Verification failed (non-critical): {e}")

    # Step 6: Validate citations
    sources_for_validation = [{"url": c.source_url, "tier": c.source_tier, "title": c.title} for c in chunks]
    validation = validate_citations(synthesis.answer, sources_for_validation)

    # Use corrected text if citations were missing
    answer = validation.corrected_text if not validation.is_valid else synthesis.answer

    # Step 7: Calculate latency
    latency_ms = int((time.time() - start_time) * 1000)

    # Log query for analytics
    logger.info(
        f"Query: '{request.query[:50]}' | Route: {route} | "
        f"Method: {synthesis.method} | Tokens: {synthesis.token_count} | "
        f"Latency: {latency_ms}ms | Similarity: {similarity_score} | "
        f"Verified: {bool(verification_data)}"
    )

    return QueryResponse(
        answer=answer,
        route=route,
        method=synthesis.method,
        sources=synthesis.sources_used,
        token_count=synthesis.token_count,
        latency_ms=latency_ms,
        similarity_score=similarity_score,
        validation={
            "is_valid": validation.is_valid,
            "total_claims": validation.total_claims,
            "cited_claims": validation.cited_claims,
            "warnings": validation.warnings,
        },
        model_used=synthesis.model_used,
        provider=synthesis.provider,
        fallback_count=synthesis.fallback_count,
        verification=verification_data,
    )


# ═══════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint — works with or without database."""
    db_status = "not_configured"
    db_url = os.getenv("DATABASE_URL", "")
    if db_url:
        try:
            import asyncpg
            conn = await asyncpg.connect(db_url)
            count = await conn.fetchval("SELECT COUNT(*) FROM documents")
            await conn.close()
            db_status = f"connected ({count} docs)"
        except Exception as e:
            db_status = f"error: {str(e)[:80]}"
    else:
        db_status = "no_database_url_set"

    return HealthResponse(status="healthy", version="1.0.0", database=db_status)


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """
    Main research query endpoint.

    Accepts a natural language query, classifies it, retrieves from RAG,
    falls back to live scraping if needed, and synthesizes a token-efficient answer.
    """
    return await run_pipeline(request)


@app.post("/classify", response_model=ClassifyResponse)
async def classify_endpoint(request: ClassifyRequest):
    """Classify a query without executing the full pipeline."""
    result = await classify_query(request.query)
    return ClassifyResponse(
        route=result.route,
        reason=result.reason,
        domain_hint=result.domain_hint,
        confidence=result.confidence,
        method=result.method,
    )


@app.post("/search", response_model=SearchResponse)
async def search_endpoint(request: SearchRequest):
    """Direct corpus search — returns raw chunks without synthesis."""
    chunks, avg_sim = await retrieve(
        query=request.query,
        top_k=request.top_k,
        final_k=request.top_k,
        domain_filter=request.domain,
    )

    results = [
        {
            "text": c.raw_text,
            "source_url": c.source_url,
            "source_tier": c.source_tier,
            "domain": c.domain,
            "title": c.title,
            "authors": c.authors,
            "similarity": c.similarity_score,
            "token_count": c.token_count,
        }
        for c in chunks
    ]

    return SearchResponse(results=results, avg_similarity=avg_sim, count=len(results))


@app.post("/scrape", response_model=ScrapeResponse)
async def scrape_endpoint(request: ScrapeRequest):
    """Direct live scrape endpoint."""
    results = await live_scrape(query=request.query or "", urls=request.urls)

    return ScrapeResponse(
        results=[
            {
                "url": r.url,
                "markdown": r.markdown,
                "title": r.title,
                "success": r.success,
                "error": r.error,
            }
            for r in results
        ],
        successful=sum(1 for r in results if r.success),
        total=len(results),
    )


# ── Research Engine Endpoints ──

@app.post("/research", response_model=ResearchResponse)
async def research_endpoint(request: ResearchRequest):
    """
    Deep research endpoint — structured research report with verification.

    This is the competitive feature: generates a mini research report
    with epistemic marking, claim-evidence tables, and debate surfacing.

    Modes:
    - depth="quick": Single-cycle research, fast (~3-5s)
    - depth="thorough": Iterative multi-cycle research (~10-15s)

    Features:
    - Source tier enforcement (P1 > P2 > P3)
    - Temporal decay (newer sources weighted higher)
    - Claim verification with epistemic markers
    - Optional structured extraction from P1 sources
    - Optional retraction detection via Crossref
    - Query decomposition for complex questions

    Output: Structured research report with executive summary,
    findings table, active debates, and source list.
    """
    from agent.research_engine import deep_research, generate_research_report

    # Run the deep research pipeline
    research_data = await deep_research(
        query=request.query,
        classification=request.classification,
        depth=request.depth,
        verify=request.verify,
        extract=request.extract,
        check_retractions=request.check_retractions,
    )

    # Generate structured research report
    report = await generate_research_report(
        query=request.query,
        classification=request.classification,
        depth=request.depth,
    )

    return ResearchResponse(
        query=request.query,
        executive_summary=report.executive_summary,
        findings=report.findings,
        debates=report.debates,
        speculative=report.speculative,
        sources=research_data.get("sources", []),
        verification=research_data.get("verification"),
        extracted_claims=research_data.get("extracted_claims", []),
        retractions=research_data.get("retractions", []),
        sub_queries=research_data.get("sub_queries", [request.query]),
        depth=request.depth,
        latency_ms=research_data.get("latency_ms", 0),
        raw_report=report.raw_report,
    )


@app.post("/verify")
async def verify_endpoint(request: ClassifyRequest):
    """
    Verify claims in a query against multiple sources.

    Returns epistemic markers: [ESTABLISHED], [TENTATIVE],
    [ACTIVE_DEBATE], [SPECULATIVE], [UNVERIFIED]
    """
    from agent.research_engine import verify_claim_with_search

    result = await verify_claim_with_search(request.query)
    return {
        "claim": result.statement,
        "status": result.epistemic_status,
        "confidence": result.confidence,
        "evidence_type": result.evidence_type,
        "supporting_sources": len(result.supporting_sources),
        "conflicting_sources": len(result.conflicting_sources),
    }


@app.get("/research/status")
async def research_status():
    """Get the status of the competitive research engine."""
    from agent.research_engine import SOURCE_TIER_DOMAINS
    return {
        "version": "2.0",
        "upgrades": {
            "tier_enforcement": {
                "status": "active",
                "P1_domains": len(SOURCE_TIER_DOMAINS["P1"]),
                "P2_domains": len(SOURCE_TIER_DOMAINS["P2"]),
                "P3_domains": len(SOURCE_TIER_DOMAINS["P3"]),
            },
            "verification_loop": {"status": "active", "epistemic_markers": ["ESTABLISHED", "TENTATIVE", "ACTIVE_DEBATE", "SPECULATIVE", "UNVERIFIED"]},
            "parallel_orchestration": {"status": "active", "graceful_degradation": True},
            "research_report_mode": {"status": "active"},
            "iterative_research": {"status": "active", "opt_in": True, "max_cycles": 3},
            "structured_extraction": {"status": "active", "p1_only": True},
            "retraction_detection": {"status": "active", "provider": "Crossref"},
            "query_decomposition": {"status": "active"},
            "temporal_decay": {"status": "active", "factor": 0.95},
        },
        "endpoints": {
            "/research": "Deep research with structured report",
            "/verify": "Claim verification with epistemic markers",
            "/query": "Standard pipeline (now with tier enforcement + verification)",
            "/research/status": "This status endpoint",
        },
    }


# ── Ingest Endpoints ──

@app.post("/ingest/url", response_model=IngestResponse)
async def ingest_url(request: IngestURLRequest, background_tasks: BackgroundTasks):
    """Ingest a web URL into the corpus."""
    from ingest.html_cleaner import clean_html
    import httpx

    try:
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            response = await client.get(request.url)
            response.raise_for_status()
            raw_content = response.text

        # Determine if HTML or plain text
        content_type = response.headers.get("content-type", "")
        if "html" in content_type:
            text = clean_html(raw_content, request.url)
        else:
            text = raw_content

        if not text or len(text) < 50:
            raise HTTPException(status_code=400, detail="Insufficient content extracted from URL")

        # Chunk
        chunks = chunk_text(
            text,
            strategy=request.chunk_strategy,
            chunk_size_tokens=request.chunk_size,
            overlap_pct=request.overlap_pct,
        )

        # Determine domain
        from urllib.parse import urlparse
        domain = urlparse(request.url).netloc

        # Embed and upsert
        count = await embed_and_upsert(
            source_url=request.url,
            source_tier=request.source_tier,
            domain=domain,
            doc_type=request.doc_type,
            title=request.title,
            authors=request.authors,
            published_date=None,
            chunks=chunks,
        )

        return IngestResponse(
            status="success",
            chunks_upserted=count,
            message=f"Ingested {count} chunks from {request.url}",
        )

    except httpx.HTTPError as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch URL: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")


@app.post("/ingest/arxiv", response_model=IngestResponse)
async def ingest_arxiv(request: IngestArxivRequest):
    """Ingest arXiv papers into the corpus."""
    from ingest.arxiv_ingest import ingest_arxiv_paper, ingest_arxiv_category

    try:
        if request.arxiv_id:
            count = await ingest_arxiv_paper(request.arxiv_id)
        elif request.category:
            count = await ingest_arxiv_category(request.category, request.max_results)
        else:
            raise HTTPException(status_code=400, detail="Provide either arxiv_id or category")

        return IngestResponse(
            status="success",
            chunks_upserted=count or 0,
            message=f"Ingested arXiv content: {count or 0} chunks",
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"arXiv ingestion failed: {e}")


@app.post("/ingest/pubmed", response_model=IngestResponse)
async def ingest_pubmed(request: IngestPubMedRequest):
    """Ingest PubMed papers into the corpus."""
    from ingest.pubmed_ingest import ingest_pubmed_search

    try:
        count = await ingest_pubmed_search(request.query, request.max_results)
        return IngestResponse(
            status="success",
            chunks_upserted=count,
            message=f"Ingested PubMed content: {count} chunks",
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PubMed ingestion failed: {e}")


@app.post("/ingest/pdf", response_model=IngestResponse)
async def ingest_pdf(request: IngestPDFRequest):
    """Ingest a PDF into the corpus."""
    from ingest.pdf_ingest import ingest_pdf_url

    try:
        count = await ingest_pdf_url(
            url=request.url,
            source_tier=request.source_tier,
            doc_type=request.doc_type,
            title=request.title,
            authors=request.authors,
        )
        return IngestResponse(
            status="success",
            chunks_upserted=count or 0,
            message=f"Ingested PDF: {count or 0} chunks",
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF ingestion failed: {e}")


@app.post("/ingest/embed-pending")
async def embed_pending():
    """Embed all chunks that don't have vectors yet."""
    try:
        count = await embed_unembedded_chunks()
        return {"status": "success", "embedded": count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding failed: {e}")


@app.get("/router/status")
async def router_status():
    """Get the status of all 9 models in the LLM fallback chain."""
    return get_router_status()


@app.get("/router/chain")
async def router_chain():
    """Get the full fallback chain configuration."""
    return {
        "fallback_order": [
            {
                "position": i + 1,
                "name": m.name,
                "provider": m.provider.value,
                "model_id": m.model_id,
                "tier": m.tier,
                "price_input_per_m": m.price_input_per_m,
                "price_output_per_m": m.price_output_per_m,
                "context_window": m.context_window,
                "supports_tools": m.supports_tools,
                "description": m.description,
            }
            for i, m in enumerate(FALLBACK_CHAIN)
        ],
        "tier_selection": {
            "similarity_gt_0.85": "pass-through (no LLM)",
            "similarity_0.72_0.85": "Granite-4.0 → Llama-1B → GLM-4.7 → Llama-3B → Qwen3-30B",
            "similarity_lt_0.72": "Full 9-model chain up to Llama-3.3-70B",
            "table_queries": "Qwen3-30B → Llama-8B → Mistral-24B → Llama-70B",
            "classification": "Cheapest configured model (Granite/Llama-1B)",
        },
    }


@app.post("/router/test")
async def router_test():
    """
    Test connectivity for all 9 models in the fallback chain.

    Sends a simple prompt to each configured model and reports:
    - Whether it responded
    - Latency in ms
    - Any errors
    - Sample output

    Use this to verify your Cloudflare API token has Workers AI permission.
    """
    results = await test_all_models()
    return {
        "total_models": len(results),
        "configured": sum(1 for r in results if r.configured),
        "reachable": sum(1 for r in results if r.reachable),
        "results": [
            {
                "name": r.model_name,
                "model_id": r.model_id,
                "provider": r.provider,
                "configured": r.configured,
                "reachable": r.reachable,
                "latency_ms": r.latency_ms,
                "error": r.error,
                "sample_output": r.sample_output,
            }
            for r in results
        ],
    }


@app.post("/router/reload")
async def router_reload():
    """Reload model configuration from config/llm_models.yaml."""
    reload_config()
    return {"status": "success", "models_loaded": len(FALLBACK_CHAIN)}


# ── MCP Mount ──
from tools.mcp_server import mcp_app
app.mount("/mcp", mcp_app)


if __name__ == "__main__":
    import uvicorn
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    uvicorn.run(app, host=host, port=port)
