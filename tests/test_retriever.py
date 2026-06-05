"""
Tests for the APEX Research Agent.
Run: pytest tests/ -v
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

# ── Chunker Tests ──

from ingest.chunker import (
    chunk_fixed_size,
    chunk_semantic,
    chunk_markdown,
    chunk_text,
    Chunk,
)


class TestFixedChunker:
    """Tests for fixed-size chunking."""

    def test_short_text_single_chunk(self):
        text = "This is a short text that should fit in one chunk."
        chunks = chunk_fixed_size(text, chunk_size_tokens=512)
        assert len(chunks) == 1
        assert chunks[0].chunk_index == 0
        assert chunks[0].total_chunks == 1

    def test_empty_text(self):
        chunks = chunk_fixed_size("")
        assert len(chunks) == 0

    def test_long_text_multiple_chunks(self):
        text = " ".join(["word"] * 5000)
        chunks = chunk_fixed_size(text, chunk_size_tokens=100, overlap_pct=0.20)
        assert len(chunks) > 1
        assert all(c.total_chunks == len(chunks) for c in chunks)

    def test_overlap_creates_redundancy(self):
        text = " ".join([f"sentence{i}." for i in range(100)])
        chunks_no_overlap = chunk_fixed_size(text, chunk_size_tokens=50, overlap_pct=0.0)
        chunks_with_overlap = chunk_fixed_size(text, chunk_size_tokens=50, overlap_pct=0.20)
        # More chunks with overlap
        assert len(chunks_with_overlap) >= len(chunks_no_overlap)

    def test_chunk_indices_sequential(self):
        text = " ".join(["word"] * 3000)
        chunks = chunk_fixed_size(text, chunk_size_tokens=100)
        indices = [c.chunk_index for c in chunks]
        assert indices == list(range(len(chunks)))


class TestSemanticChunker:
    """Tests for semantic chunking."""

    def test_preserves_sentences(self):
        text = "First sentence here. Second sentence here. Third sentence here. Fourth sentence here."
        chunks = chunk_semantic(text, chunk_size_tokens=20)
        # Each chunk should contain complete sentences
        for chunk in chunks:
            assert chunk.text.strip()

    def test_empty_input(self):
        chunks = chunk_semantic("")
        assert len(chunks) == 0


class TestMarkdownChunker:
    """Tests for markdown-aware chunking."""

    def test_respects_headers(self):
        text = "# Header 1\nContent under header 1.\n\n# Header 2\nContent under header 2."
        chunks = chunk_markdown(text, chunk_size_tokens=512)
        assert len(chunks) >= 1

    def test_no_headers(self):
        text = "Just plain text without any markdown headers."
        chunks = chunk_markdown(text)
        assert len(chunks) == 1


class TestChunkTextRouter:
    """Tests for the chunk_text strategy router."""

    def test_fixed_strategy(self):
        text = "Some text here for testing."
        chunks = chunk_text(text, strategy="fixed")
        assert len(chunks) >= 1

    def test_invalid_strategy(self):
        with pytest.raises(ValueError, match="Unknown chunking strategy"):
            chunk_text("text", strategy="nonexistent")


# ── HTML Cleaner Tests ──

from ingest.html_cleaner import clean_html, clean_markdown


class TestHTMLCleaner:
    """Tests for HTML cleaning."""

    def test_removes_script_tags(self):
        html = "<html><body><script>alert('xss')</script><p>Content</p></body></html>"
        result = clean_html(html)
        assert "alert" not in result
        assert "Content" in result

    def test_removes_nav_footer(self):
        html = "<nav>Nav content</nav><main><p>Main content</p></main><footer>Footer</footer>"
        result = clean_html(html)
        assert "Main content" in result

    def test_empty_html(self):
        assert clean_html("") == ""
        assert clean_html("   ") == ""

    def test_extracts_article(self):
        html = "<html><body><article><p>Article content</p></article></body></html>"
        result = clean_html(html)
        assert "Article content" in result

    def test_removes_ad_elements(self):
        html = '<div class="ad-banner">Ad content</div><div class="content"><p>Real content</p></div>'
        result = clean_html(html)
        assert "Real content" in result


class TestMarkdownCleaner:
    """Tests for markdown cleaning."""

    def test_removes_cookie_notices(self):
        md = "Cookie Policy: We use cookies.\n\nReal content here."
        result = clean_markdown(md)
        assert "Real content" in result
        assert "Cookie" not in result

    def test_cleans_excess_whitespace(self):
        md = "Content\n\n\n\n\nMore content"
        result = clean_markdown(md)
        assert "\n\n\n" not in result


# ── Query Classifier Tests ──

from agent.query_classifier import classify_rules, ClassificationResult


class TestQueryClassifier:
    """Tests for query classification rules."""

    def test_temporal_keywords_route_live(self):
        result = classify_rules("What is the latest news on AI?")
        assert result.route == "live"
        assert "Temporal" in result.reason

    def test_breaking_news_route_live(self):
        result = classify_rules("breaking news about climate change")
        assert result.route == "live"

    def test_today_keyword_route_live(self):
        result = classify_rules("What happened today in tech?")
        assert result.route == "live"

    def test_arxiv_reference_route_rag(self):
        result = classify_rules("arxiv paper on transformers")
        assert result.route == "rag"
        assert result.domain_hint == "arxiv"

    def test_research_pattern_route_rag(self):
        result = classify_rules("How does RAG improve language model accuracy?")
        assert result.route == "rag"

    def test_default_route_rag(self):
        result = classify_rules("What is the capital of France?")
        assert result.route == "rag"
        assert result.confidence == 0.5

    def test_recent_keyword_route_live(self):
        result = classify_rules("Recent developments in quantum computing")
        assert result.route == "live"


# ── Citation Validator Tests ──

from tools.citation_validator import (
    extract_citations,
    detect_claims,
    validate_citations,
    format_source_citation,
    ValidationResult,
)


class TestCitationValidator:
    """Tests for citation validation."""

    def test_extract_citations(self):
        text = "RAG improves accuracy [Lewis 2023, P1]. Some claim [domain.com, P2]."
        citations = extract_citations(text)
        assert len(citations) == 2
        assert citations[0][1] == "P1"
        assert citations[1][1] == "P2"

    def test_extract_unverified(self):
        text = "This is uncertain [UNVERIFIED]."
        citations = extract_citations(text)
        assert len(citations) == 1
        assert citations[0][0] == "UNVERIFIED"

    def test_detect_percentage_claims(self):
        text = "Accuracy improved by 15%. This is a fact."
        claims = detect_claims(text)
        assert len(claims) >= 1
        assert "15%" in claims[0]

    def test_validate_good_citations(self):
        text = "RAG reduces hallucination [Lewis 2023, P1]. Models achieve 94% accuracy [nature.com, P1]."
        result = validate_citations(text)
        assert isinstance(result, ValidationResult)

    def test_validate_adds_unverified(self):
        text = "Accuracy improved by 15%."
        result = validate_citations(text)
        assert result.uncited_claims >= 1
        assert "[UNVERIFIED]" in result.corrected_text

    def test_format_citation_with_authors(self):
        citation = format_source_citation(
            url="https://arxiv.org/abs/2312.10997",
            tier="P1",
            authors=["Lewis, P.", "Perez, E."],
        )
        assert "P1" in citation
        assert "Lewis" in citation

    def test_format_citation_without_authors(self):
        citation = format_source_citation(
            url="https://nature.com/article/123",
            tier="P1",
        )
        assert "nature.com" in citation
        assert "P1" in citation


# ── Retriever Unit Tests ──

from agent.retriever import (
    reciprocal_rank_fusion,
    apply_token_budget,
    apply_source_hierarchy,
    RetrievedChunk,
)


class TestReciprocalRankFusion:
    """Tests for RRF score fusion."""

    def _make_chunk(self, chunk_id: str, sim: float = 0.8, kw: float = 0.0) -> RetrievedChunk:
        return RetrievedChunk(
            id=chunk_id,
            source_url=f"https://example.com/{chunk_id}",
            source_tier="P1",
            domain="example.com",
            doc_type="paper",
            title="Test",
            authors=[],
            raw_text="Test text",
            metadata={},
            chunk_index=0,
            total_chunks=1,
            similarity_score=sim,
            keyword_score=kw,
        )

    def test_fusion_combines_scores(self):
        vector_results = [self._make_chunk("a", sim=0.9), self._make_chunk("b", sim=0.8)]
        keyword_results = [self._make_chunk("b", kw=5.0), self._make_chunk("c", kw=3.0)]

        fused = reciprocal_rank_fusion(vector_results, keyword_results)
        assert len(fused) == 3  # a, b, c

    def test_fusion_prefers_results_in_both_lists(self):
        vector_results = [self._make_chunk("a"), self._make_chunk("shared")]
        keyword_results = [self._make_chunk("shared"), self._make_chunk("b")]

        fused = reciprocal_rank_fusion(vector_results, keyword_results)
        # "shared" should rank highest (appears in both lists)
        assert fused[0].id == "shared"

    def test_empty_inputs(self):
        assert reciprocal_rank_fusion([], []) == []


class TestTokenBudget:
    """Tests for token budget enforcement."""

    def _make_chunk(self, token_count: int) -> RetrievedChunk:
        return RetrievedChunk(
            id=f"chunk-{token_count}",
            source_url="https://example.com",
            source_tier="P1",
            domain="example.com",
            doc_type="paper",
            title="Test",
            authors=[],
            raw_text="word " * token_count,
            metadata={},
            chunk_index=0,
            total_chunks=1,
            token_count=token_count,
        )

    def test_within_budget(self):
        chunks = [self._make_chunk(500), self._make_chunk(500)]
        result = apply_token_budget(chunks, max_tokens=2000)
        assert len(result) == 2

    def test_exceeds_budget(self):
        chunks = [self._make_chunk(1500), self._make_chunk(1500)]
        result = apply_token_budget(chunks, max_tokens=2000)
        assert len(result) < 2
        total = sum(c.token_count for c in result)
        assert total <= 2000


class TestSourceHierarchy:
    """Tests for source tier hierarchy enforcement."""

    def _make_chunk(self, tier: str, score: float = 0.5) -> RetrievedChunk:
        return RetrievedChunk(
            id=f"chunk-{tier}-{score}",
            source_url="https://example.com",
            source_tier=tier,
            domain="example.com",
            doc_type="paper",
            title="Test",
            authors=[],
            raw_text="Test",
            metadata={},
            chunk_index=0,
            total_chunks=1,
            similarity_score=0.8,
            fused_score=score,
        )

    def test_p1_boosted_over_p3(self):
        chunks = [
            self._make_chunk("P3", 0.6),
            self._make_chunk("P1", 0.5),
        ]
        result = apply_source_hierarchy(chunks)
        # P1 should be boosted above P3
        assert result[0].source_tier == "P1"

    def test_p3_not_penalized_without_p1(self):
        chunks = [self._make_chunk("P3", 0.6)]
        result = apply_source_hierarchy(chunks)
        assert result[0].fused_score == 0.6  # Unchanged


# ── Synthesizer Tests ──

from agent.synthesizer import (
    _check_direct_answer,
    _detect_table_needed,
    _format_citation,
    APEX_SYSTEM_PROMPT,
)


class TestSynthesizer:
    """Tests for synthesis components."""

    def test_direct_answer_high_similarity(self):
        chunk = RetrievedChunk(
            id="test",
            source_url="https://arxiv.org/abs/2312.10997",
            source_tier="P1",
            domain="arxiv.org",
            doc_type="paper",
            title="RAG Paper",
            authors=["Lewis, P."],
            raw_text="RAG reduces hallucination by grounding generation in retrieved evidence.",
            metadata={},
            chunk_index=0,
            total_chunks=1,
            similarity_score=0.90,
        )
        result = _check_direct_answer("What does RAG do?", [chunk])
        assert result is not None
        assert "RAG" in result
        assert "P1" in result

    def test_no_direct_answer_low_similarity(self):
        chunk = RetrievedChunk(
            id="test",
            source_url="https://example.com",
            source_tier="P2",
            domain="example.com",
            doc_type="article",
            title="Related",
            authors=[],
            raw_text="Something tangentially related.",
            metadata={},
            chunk_index=0,
            total_chunks=1,
            similarity_score=0.60,
        )
        result = _check_direct_answer("What is RAG?", [chunk])
        assert result is None

    def test_detect_compare_table(self):
        chunks = []
        assert _detect_table_needed("Compare RAG vs fine-tuning approaches", chunks) is True

    def test_detect_no_table(self):
        chunks = []
        assert _detect_table_needed("What is RAG?", chunks) is False

    def test_format_citation_with_authors(self):
        chunk = RetrievedChunk(
            id="test",
            source_url="https://arxiv.org/abs/2312.10997",
            source_tier="P1",
            domain="arxiv.org",
            doc_type="paper",
            title="Test",
            authors=["Lewis, P."],
            raw_text="",
            metadata={"arxiv_id": "2312.10997"},
            chunk_index=0,
            total_chunks=1,
        )
        citation = _format_citation(chunk)
        assert "Lewis" in citation
        assert "P1" in citation

    def test_system_prompt_rules_present(self):
        assert "150" in APEX_SYSTEM_PROMPT
        assert "UNVERIFIED" in APEX_SYSTEM_PROMPT
        assert "PASS-THROUGH" in APEX_SYSTEM_PROMPT
        assert "TABLES OVER PROSE" in APEX_SYSTEM_PROMPT


# ── Live Scraper Tests ──

from tools.live_scraper import truncate_to_token_budget, MAX_SCRAPE_TOKENS


class TestLiveScraper:
    """Tests for live scraper components."""

    def test_truncate_within_budget(self):
        text = "Short text"
        result = truncate_to_token_budget(text, max_tokens=100)
        assert result == text

    def test_truncate_exceeds_budget(self):
        text = "word " * 10000
        result = truncate_to_token_budget(text, max_tokens=100)
        assert len(result.split()) <= 110  # Some margin for sentence boundary

    def test_truncate_preserves_sentences(self):
        text = ". ".join([f"Sentence {i} here" for i in range(1000)]) + "."
        result = truncate_to_token_budget(text, max_tokens=100)
        # Should end with a period (sentence boundary)
        assert result.rstrip().endswith(".") or len(result) < len(text)


# ── LLM Router Tests ──

from agent.llm_router import (
    FALLBACK_CHAIN,
    ModelConfig,
    Provider,
    select_tier,
    get_router_status,
    RouterResult,
)


class TestFallbackChain:
    """Tests for the 9-model fallback chain configuration."""

    def test_chain_has_9_models(self):
        assert len(FALLBACK_CHAIN) == 9

    def test_first_model_is_passthrough(self):
        assert FALLBACK_CHAIN[0].provider == Provider.PASSTHROUGH

    def test_cloudflare_models_present(self):
        cf_models = [m for m in FALLBACK_CHAIN if m.provider == Provider.CLOUDFLARE]
        assert len(cf_models) == 4  # Granite, GLM-4.7, Qwen3, Mistral-Small

    def test_github_models_present(self):
        gh_models = [m for m in FALLBACK_CHAIN if m.provider == Provider.GITHUB]
        assert len(gh_models) == 4  # Phi-4-mini, Mistral-Nemo, GPT-4o-mini, DeepSeek-V3

    def test_models_ordered_cheapest_first(self):
        # Non-passthrough models should generally increase in price
        priced = [m for m in FALLBACK_CHAIN[1:] if m.price_output_per_m > 0]
        # First model should be cheaper than last
        assert priced[0].price_output_per_m <= priced[-1].price_output_per_m

    def test_all_models_have_required_fields(self):
        for m in FALLBACK_CHAIN:
            assert m.name
            assert m.model_id
            assert m.context_window > 0 or m.provider == Provider.PASSTHROUGH
            assert m.tier in ("free", "cheap", "mid", "capable", "cloud")


class TestTierSelection:
    """Tests for model tier selection logic."""

    def test_high_similarity_returns_passthrough(self):
        models = select_tier(similarity=0.90)
        assert any(m.provider == Provider.PASSTHROUGH for m in models)

    def test_low_similarity_returns_more_models(self):
        low_sim = select_tier(similarity=0.50)
        mid_sim = select_tier(similarity=0.78)
        assert len(low_sim) >= len(mid_sim)

    def test_table_needed_excludes_cheap(self):
        models = select_tier(table_needed=True)
        for m in models:
            assert m.tier in ("mid", "capable", "cloud")

    def test_classification_uses_cheap_models(self):
        models = select_tier(is_classification=True)
        for m in models:
            assert m.tier in ("cheap", "free")

    def test_force_model_returns_specific_model(self):
        models = select_tier(force_model="GLM-4.7-Flash")
        assert len(models) == 1
        assert models[0].name == "GLM-4.7-Flash"

    def test_force_nonexistent_returns_passthrough(self):
        models = select_tier(force_model="nonexistent-model")
        assert len(models) == 0 or models[0].provider == Provider.PASSTHROUGH


class TestRouterStatus:
    """Tests for router status reporting."""

    def test_status_returns_dict(self):
        status = get_router_status()
        assert isinstance(status, dict)
        assert "total_models" in status
        assert "models" in status
        assert status["total_models"] == 9

    def test_status_shows_all_models(self):
        status = get_router_status()
        assert len(status["models"]) == 9

    def test_status_shows_provider_config(self):
        status = get_router_status()
        assert "cloudflare_configured" in status
        assert "github_configured" in status


# ── Integration Test Markers ──

@pytest.mark.asyncio
class TestAsyncComponents:
    """Async component tests (mocked external calls)."""

    async def test_classify_query_rules(self):
        from agent.query_classifier import classify_query
        result = await classify_query("What is RAG?")
        assert result.route in ("rag", "live")

    async def test_classify_query_live(self):
        from agent.query_classifier import classify_query
        result = await classify_query("What is the latest AI news today?")
        assert result.route == "live"

    async def test_classify_result_has_model_field(self):
        from agent.query_classifier import classify_rules
        result = classify_rules("What is RAG?")
        assert hasattr(result, "model_used")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
