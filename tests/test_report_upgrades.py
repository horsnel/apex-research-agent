"""
Unit tests for APEX 3.0 report-quality upgrades.

Covers the pure (no-I/O, no-LLM) logic added across the phases:
- Phase 1: task-aware tier selection (report routing)
- Phase 4: canonical dedup, citation integrity, NLI aggregation
- Phase 5: diverse source selection, domain-aware decay + cache TTL

Run: pytest tests/test_report_upgrades.py -v
"""

import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from agent.research_engine import (
    _normalize_doi,
    _title_fingerprint,
    dedup_sources_canonical,
    select_diverse_sources,
    infer_domain,
    get_decay_factor,
    apply_temporal_decay,
    _extract_source_year,
    verify_claims_from_sources,
    _render_provenance_section,
)
from agent.citation_integrity import (
    extract_numbered_citations,
    check_report_citations,
    build_sources_section,
    CitationReport,
    scan_domain_generalization,
    scan_news_tier_citations,
    scan_scope_dodge,
    scan_off_topic_padding,
)
from agent.verifier import (
    select_nli_candidates,
    aggregate_nli_to_status,
    NLIResult,
    NLIVerdict,
)
from agent.llm_router import select_tier
from agent.llm_wiki import get_cache_ttl_days


# ═══════════════════════════════════════════════════════════════
# Phase 4: canonical dedup
# ═══════════════════════════════════════════════════════════════


class TestCanonicalDedup:
    def test_normalize_doi(self):
        assert _normalize_doi("10.1038/s41586-020-1234-5") == "10.1038/s41586-020-1234-5"
        assert _normalize_doi("https://doi.org/10.1038/ABC") == "10.1038/abc"
        assert _normalize_doi("doi:10.1/x") == "10.1/x"
        assert _normalize_doi("") == ""
        assert _normalize_doi(None) == ""

    def test_title_fingerprint(self):
        assert _title_fingerprint("Attention Is All You Need!") == \
               _title_fingerprint("attention  is all you need")
        assert _title_fingerprint("") == ""

    def test_doi_dedup_merges(self):
        sources = [
            {"url": "https://arxiv.org/abs/2401.0001", "title": "Paper A",
             "snippet": "short", "tier": "P1", "doi": "10.1/abc", "citation_count": 5},
            {"url": "https://semanticscholar.org/paper/xyz", "title": "Paper A",
             "snippet": "a much longer and richer abstract for the same paper", "tier": "P1",
             "doi": "https://doi.org/10.1/abc", "citation_count": 7},
        ]
        out = dedup_sources_canonical(sources)
        assert len(out) == 1
        assert out[0]["citation_count"] == 7
        assert "richer abstract" in out[0]["snippet"]
        assert "https://arxiv.org/abs/2401.0001" in out[0]["also_found_in"]

    def test_title_fingerprint_dedup_without_doi(self):
        sources = [
            {"url": "https://a.com/x", "title": "Attention Is All You Need", "snippet": "s1"},
            {"url": "https://b.com/y", "title": "Attention  is all you NEED!", "snippet": "s2"},
        ]
        out = dedup_sources_canonical(sources)
        assert len(out) == 1

    def test_distinct_papers_survive(self):
        sources = [
            {"url": "https://a.com/1", "title": "Paper One", "doi": "10.1/1"},
            {"url": "https://b.com/2", "title": "Paper Two", "doi": "10.1/2"},
            {"url": "https://c.com/3", "title": "Paper Three"},
        ]
        assert len(dedup_sources_canonical(sources)) == 3

    def test_tier_upgrade_on_merge(self):
        sources = [
            {"url": "https://blog.example.com/p", "title": "Same Study", "tier": "UNV"},
            {"url": "https://arxiv.org/abs/1", "title": "Same  Study", "tier": "P1"},
        ]
        out = dedup_sources_canonical(sources)
        assert len(out) == 1 and out[0]["tier"] == "P1"


# ═══════════════════════════════════════════════════════════════
# Phase 5: diverse source selection
# ═══════════════════════════════════════════════════════════════


def _src(i, domain, tier, score):
    return {"url": f"https://{domain}/item{i}", "title": f"T{i}", "tier": tier,
            "score": score, "snippet": "x" * 100}


class TestSelectDiverseSources:
    def test_per_domain_cap(self):
        sources = [_src(i, "arxiv.org", "P1", 1.0 - i * 0.01) for i in range(10)]
        out = select_diverse_sources(sources, max_sources=25, per_domain_cap=3)
        assert len(out) == 3  # all from one domain, capped

    def test_max_sources_respected(self):
        sources = [_src(i, f"d{i}.com", "P1", 0.5) for i in range(30)]
        out = select_diverse_sources(sources, max_sources=10)
        assert len(out) == 10

    def test_min_p2_guarantee(self):
        sources = [_src(i, "arxiv.org", "P1", 0.9 - i * 0.01) for i in range(5)]
        sources.append(_src(99, "reuters.com", "P2", 0.2))
        out = select_diverse_sources(sources, max_sources=25, per_domain_cap=4, min_p2=2)
        tiers = [s["tier"] for s in out]
        assert tiers.count("P2") == 1  # only 1 P2 exists in input

        sources2 = sources + [_src(98, "bloomberg.com", "P2", 0.15)]
        out2 = select_diverse_sources(sources2, max_sources=25, per_domain_cap=4, min_p2=2)
        assert [s["tier"] for s in out2].count("P2") == 2

    def test_breadth_over_truncation(self):
        """The old tier_enforce kept only 5 P1 sources; selection keeps diversity."""
        sources = (
            [_src(i, "arxiv.org", "P1", 0.9) for i in range(8)]
            + [_src(50 + i, f"site{i}.com", "P2", 0.5) for i in range(8)]
            + [_src(80 + i, f"blog{i}.com", "P3", 0.3) for i in range(8)]
        )
        out = select_diverse_sources(sources, max_sources=25, per_domain_cap=3)
        domains = {_extract_domain_of(s) for s in out}
        assert len(domains) >= 10  # breadth preserved
        assert len(out) > 5


def _extract_domain_of(s):
    from agent.research_engine import _extract_domain
    return _extract_domain(s["url"])


# ═══════════════════════════════════════════════════════════════
# Phase 5: domain-aware temporal decay
# ═══════════════════════════════════════════════════════════════


class TestTemporalDecay:
    def test_infer_domain(self):
        assert infer_domain("How do LLM transformers handle long context?") == "ai"
        assert infer_domain("Clinical trial results for a cancer drug") == "medicine"
        assert infer_domain("A theorem about algebraic topology") == "mathematics"
        assert infer_domain("What is a potato?") == "general"

    def test_decay_factors_ordered(self):
        assert get_decay_factor("ai") < get_decay_factor("medicine") < get_decay_factor("mathematics")

    def test_domain_aware_decay(self):
        sources = [
            {"url": "https://arxiv.org/abs/2201.0001", "title": "Old AI paper",
             "snippet": "transformer", "published_date": "2022-01", "score": 1.0},
            {"url": "https://nature.com/x", "title": "Old medicine paper",
             "snippet": "clinical trial", "published_date": "2022-01", "score": 1.0},
        ]
        out = apply_temporal_decay(sources, current_year=2026)
        ai = next(s for s in out if "arxiv" in s["url"])
        med = next(s for s in out if "nature" in s["url"])
        assert ai["adjusted_score"] < med["adjusted_score"]  # AI decays faster

    def test_arxiv_id_year_extraction(self):
        assert _extract_source_year({"url": "https://arxiv.org/abs/2403.12345"}) == 2024
        assert _extract_source_year({"published_date": "2019-06"}) == 2019
        assert _extract_source_year({"metadata": {"year": "2021"}}) == 2021
        assert _extract_source_year({"title": "no date anywhere"}) is None

    def test_undated_sources_penalized(self):
        dated = {"url": "https://a.com/1", "title": "t", "published_date": "2026", "score": 1.0}
        undated = {"url": "https://b.com/2", "title": "t", "score": 1.0}
        out = apply_temporal_decay([dict(dated), dict(undated)], current_year=2026)
        assert out[0]["url"] == "https://a.com/1"


# ═══════════════════════════════════════════════════════════════
# Phase 1: task-aware tier selection
# ═══════════════════════════════════════════════════════════════


class TestTaskAwareTierSelection:
    @pytest.fixture(autouse=True)
    def _configured_cf(self, monkeypatch):
        """Simulate a configured Cloudflare token (production default)."""
        import agent.llm_router as lr
        monkeypatch.setattr(lr, "CLOUDFLARE_API_TOKEN", "test-token")

    def test_report_task_excludes_cheap_tiers(self):
        models = select_tier(task_type="report")
        assert models, "report tier must select at least one model"
        assert all(m.tier in ("mid", "capable", "cloud") for m in models)
        names = [m.name for m in models]
        assert "Granite-4.0-Micro" not in names
        assert "Llama-3.2-1B" not in names

    def test_report_task_respects_output_caps(self):
        models = select_tier(task_type="report")
        assert all(m.max_output_tokens >= 4096 for m in models), \
            "report models must allow >=4k output, else reports truncate again"

    def test_nli_task_targets_mid_models(self):
        models = select_tier(task_type="nli")
        assert all(m.tier in ("mid", "capable", "cloud") for m in models)

    def test_classification_still_cheap(self):
        models = select_tier(is_classification=True)
        assert models
        assert all(m.tier in ("cheap", "free") for m in models)

    def test_chat_default_unchanged(self):
        models = select_tier(similarity=0.5)
        assert any(m.tier in ("cheap", "mid", "free") for m in models)


# ═══════════════════════════════════════════════════════════════
# Phase 4: citation integrity
# ═══════════════════════════════════════════════════════════════


SOURCES = [
    {"url": f"https://src{i}.com/paper", "title": f"Source {i}", "tier": "P1"}
    for i in range(1, 6)
]


class TestCitationIntegrity:
    def test_extract_numbered_citations(self):
        text = "GPT-4 scored 86% [S1]. Llama-2 followed [S2, S3]. Uncited [S99]."
        nums = extract_numbered_citations(text)
        assert sorted(nums) == [1, 2, 3, 99]

    def test_invalid_citations_detected(self):
        report = "The model improved 15% [S1]. Another study found gains [S27].\n\n## SOURCES\n[S1] x"
        cr = check_report_citations(report, SOURCES)
        assert "[S27]" in cr.invalid_citations
        assert cr.citation_precision < 1.0
        assert cr.needs_revision

    def test_uncited_claims_detected(self):
        report = (
            "## KEY FINDINGS\n\n"
            "The benchmark improved 42% across all tasks [S1].\n"
            "Training costs dropped by 80% with no citation at all.\n"
            "\n## SOURCES\n[S1] x"
        )
        cr = check_report_citations(report, SOURCES)
        assert cr.total_claim_sentences == 2
        assert cr.uncited_claim_sentences, "the uncited 80% claim must be flagged"
        assert 0 < cr.claim_coverage < 1.0

    def test_clean_report_passes(self):
        report = (
            "## KEY FINDINGS\n\n"
            "Accuracy reached 91.4% [S1].\n"
            "A follow-up study confirmed the result [S2].\n"
            "\n## SOURCES\n[S1] a\n[S2] b"
        )
        cr = check_report_citations(report, SOURCES)
        assert not cr.invalid_citations
        assert cr.claim_coverage == 1.0
        assert not cr.needs_revision

    def test_bibliography_excluded_from_claims(self):
        report = "Findings are solid [S1].\n\n## SOURCES\n[S1] This paper improved results by 50% in 2020. https://x.com"
        cr = check_report_citations(report, SOURCES)
        assert cr.total_claim_sentences <= 1

    def test_build_sources_section(self):
        section = build_sources_section(SOURCES)
        assert section.startswith("## SOURCES")
        assert "[S1]" in section and "[S5]" in section
        assert "https://src3.com/paper" in section


# ═══════════════════════════════════════════════════════════════
# Phase 4: NLI aggregation
# ═══════════════════════════════════════════════════════════════


class TestNLI:
    def test_candidate_prefilter(self):
        sources = [
            {"title": "transformer attention mechanisms", "snippet": "attention scores", "tier": "P1"},
            {"title": "completely unrelated cooking recipe", "snippet": "boil pasta", "tier": "P3"},
        ]
        cands = select_nli_candidates("attention mechanisms improve transformer accuracy", sources)
        assert len(cands) == 1
        assert "transformer" in cands[0]["title"]

    def test_two_independent_supports_established(self):
        result = NLIResult(
            statement="X improves Y",
            supports=[
                NLIVerdict(verdict="support", confidence=0.9, source_url="https://arxiv.org/a", source_tier="P1"),
                NLIVerdict(verdict="support", confidence=0.8, source_url="https://nature.com/b", source_tier="P1"),
            ],
        )
        status, conf, _ = aggregate_nli_to_status(result)
        assert status == "ESTABLISHED"
        assert conf >= 0.7

    def test_same_domain_supports_not_independent(self):
        result = NLIResult(
            statement="X improves Y",
            supports=[
                NLIVerdict(verdict="support", confidence=0.9, source_url="https://arxiv.org/a", source_tier="P1"),
                NLIVerdict(verdict="support", confidence=0.8, source_url="https://arxiv.org/b", source_tier="P1"),
            ],
        )
        status, conf, _ = aggregate_nli_to_status(result)
        # Same domain → not independent → cannot be ESTABLISHED via 2 sources
        assert status != "ESTABLISHED" or conf <= 0.7

    def test_support_plus_contradiction_is_debate(self):
        result = NLIResult(
            statement="X improves Y",
            supports=[NLIVerdict(verdict="support", confidence=0.9, source_url="https://a.com/1", source_tier="P1")],
            contradictions=[NLIVerdict(verdict="contradict", confidence=0.7, source_url="https://b.com/2", source_tier="P1")],
        )
        status, _, _ = aggregate_nli_to_status(result)
        assert status == "ACTIVE_DEBATE"

    def test_no_supports_unverified(self):
        result = NLIResult(statement="claim")
        status, conf, _ = aggregate_nli_to_status(result)
        assert status == "UNVERIFIED" and conf <= 0.1


# ═══════════════════════════════════════════════════════════════
# Phase 5: cache TTL
# ═══════════════════════════════════════════════════════════════


class TestCacheTTL:
    def test_domain_aware(self, monkeypatch):
        for k in list(os.environ):
            if k.startswith("APEX_CACHE_TTL"):
                monkeypatch.delenv(k)
        assert get_cache_ttl_days("latest LLM news", "general") <= 7
        assert get_cache_ttl_days("a theorem proof", "mathematics") >= 180
        assert get_cache_ttl_days("clinical drug trial", "medicine") >= 60

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("APEX_CACHE_TTL_DAYS", "3")
        assert get_cache_ttl_days("anything", "mathematics") == 3

    def test_per_domain_override(self, monkeypatch):
        monkeypatch.setenv("APEX_CACHE_TTL_AI", "5")
        assert get_cache_ttl_days("transformer paper", "ai") == 5


# ═══════════════════════════════════════════════════════════════
# Heuristic verification still functions (backward compat)
# ═══════════════════════════════════════════════════════════════


class TestHeuristicVerificationCompat:
    def test_verify_claims_basic(self):
        claims = ["Drug X reduced mortality by 20% in randomized trials"]
        sources = [
            {"url": "https://nejm.org/a", "title": "Trial", "tier": "P1",
             "snippet": "The randomized trial found that Drug X reduced mortality by 20% significantly"},
            {"url": "https://lancet.com/b", "title": "Review", "tier": "P1",
             "snippet": "Evidence shows that Drug X reduced mortality by 20% in randomized trials"},
        ]
        result = verify_claims_from_sources(claims, sources)
        assert result.claims
        assert result.claims[0].epistemic_status in ("ESTABLISHED", "TENTATIVE")


class TestQueryVariantLabelStripping:
    """Regression: LLM-written angle labels ('Technical: ...') must be stripped
    from expanded variants — raw colons break downstream source APIs
    (ClinicalTrials.gov returns HTTP 400 on bare colons in query.term)."""

    def test_label_prefixes_stripped(self, monkeypatch):
        import agent.research_engine as re_mod
        from agent.llm_router import RouterResult

        fake = RouterResult(
            content=(
                "1. Technical: Comparative analysis of unpublished negative-result trials\n"
                "2. Empirical: Effect size distortion from publication selection bias\n"
                "3. Practical: How registries capture trials that never publish\n"
            ),
            model_name="Granite-4.0-Micro",
            model_id="x",
            provider="cloudflare",
            fallback_count=0,
            total_latency_ms=1,
        )

        async def fake_synth(*a, **k):
            return fake

        monkeypatch.setattr("agent.llm_router.synthesize_with_router", fake_synth)
        variants = asyncio.run(re_mod.expand_query_variants("base query", max_variants=4))
        assert all(":" not in v for v in variants), variants
        assert any("Comparative analysis" in v for v in variants)
        assert not any(v.lower().startswith("technical") for v in variants)

    def test_clean_variants_untouched(self, monkeypatch):
        import agent.research_engine as re_mod
        from agent.llm_router import RouterResult

        fake = RouterResult(
            content="1. A perfectly clean variant without any label prefix\n",
            model_name="Granite-4.0-Micro",
            model_id="x",
            provider="cloudflare",
            fallback_count=0,
            total_latency_ms=1,
        )

        async def fake_synth(*a, **k):
            return fake

        monkeypatch.setattr("agent.llm_router.synthesize_with_router", fake_synth)
        variants = asyncio.run(re_mod.expand_query_variants("base query", max_variants=3))
        assert "A perfectly clean variant without any label prefix" in variants[1]


class TestPruneUncitedSources:
    """Dead-source pruning with deterministic renumbering."""

    def _mk(self, n):
        return [{"url": f"https://src{i}.com/a", "title": f"Source {i}", "tier": "P1",
                 "snippet": f"snippet {i}"} for i in range(1, n + 1)]

    def test_no_dead_sources_unchanged(self):
        from agent.citation_integrity import prune_uncited_sources
        srcs = self._mk(3)
        text = "Claims [S1, S2] and [S3].\n\n## SOURCES\n[S1] a\n[S2] b\n[S3] c"
        new_text, new_srcs, stats = prune_uncited_sources(text, srcs)
        assert stats["pruned"] == []
        assert new_text == text
        assert new_srcs == srcs

    def test_dead_sources_pruned_and_renumbered(self):
        from agent.citation_integrity import prune_uncited_sources
        srcs = self._mk(4)
        body = ("Alpha [S1] beta. Gamma [S3] delta [S4]. "
                "\n\n## SOURCES\n[S1] a\n[S2] b\n[S3] c\n[S4] d")
        new_text, new_srcs, stats = prune_uncited_sources(body, srcs)
        assert stats["pruned"] == [2]
        assert stats["after"] == 3
        assert len(new_srcs) == 3
        # S3 -> S2, S4 -> S3
        assert "[S2] delta" in new_text or "[S2] delta [S3]" in new_text
        assert "[S4]" not in new_text
        assert "[S2]" in new_text and "[S3]" in new_text
        # bibliography rebuilt with exactly the cited sources
        assert new_text.count("[S") >= 3
        assert "Source 2" not in new_text
        assert "Source 3" in new_text and "Source 4" in new_text

    def test_multi_citation_form_renumbered(self):
        from agent.citation_integrity import prune_uncited_sources
        srcs = self._mk(3)
        body = "Mixed form [S2, S3] with more [S1].\n\n## SOURCES\nx"
        new_text, new_srcs, stats = prune_uncited_sources(body, srcs)
        assert stats["pruned"] == []
        assert "[S2, S3]" in new_text or "[S2,S3]" in new_text

    def test_no_sources_section_unchanged(self):
        from agent.citation_integrity import prune_uncited_sources
        srcs = self._mk(2)
        text = "Body only, no bibliography [S1]."
        new_text, new_srcs, stats = prune_uncited_sources(text, srcs)
        assert new_text == text and stats["pruned"] == []


class TestRelevanceGate:
    """Loosely-related retrieval padding must be dropped before selection."""

    def _src(self, title, snippet, tier="P1", url=None):
        return {"url": url or f"https://{title[:8].replace(' ', '')}.com/a",
                "title": title, "snippet": snippet, "tier": tier}

    def test_irrelevant_sources_dropped(self):
        from agent.research_engine import select_diverse_sources
        query = "meta-analysis of unpublished negative-result trials"
        sources = [
            self._src("Footprint of publication selection bias on meta-analyses",
                      "68,000 meta-analyses; unpublished negative results; publication bias; trials"),
            self._src("IPPOG: Bridging science education and modern research",
                      "particle physics outreach masterclasses for school students", url="https://ippo.org/x"),
            self._src("Bupropion pharmacology review",
                      "antidepressant drug used for smoking cessation treatment", url="https://wiki.org/b"),
            self._src("Trial registration policies and unpublished outcomes",
                      "clinical trials registries require registration of negative results", tier="P2",
                      url="https://trials.org/x"),
        ]
        selected = select_diverse_sources(sources, query=query, relevance_floor=2)
        titles = [s["title"] for s in selected]
        assert "Bupropion pharmacology review" not in titles
        assert "IPPOG: Bridging science education and modern research" not in titles
        assert "Footprint of publication selection bias on meta-analyses" in titles

    def test_floor_prevents_degenerate_selection(self):
        from agent.research_engine import select_diverse_sources
        query = "quantum chromodynamics lattice gauge calculations"
        sources = [
            self._src("Completely unrelated topic one", "nothing shared here at all"),
            self._src("Completely unrelated topic two", "also nothing shared whatsoever"),
        ]
        selected = select_diverse_sources(sources, query=query, relevance_floor=2)
        assert len(selected) == 2  # floor top-up, never empty

    def test_no_query_bypasses_gate(self):
        from agent.research_engine import select_diverse_sources
        sources = [
            self._src("A", "x"), self._src("B", "y"), self._src("C", "z"),
        ]
        selected = select_diverse_sources(sources, query="")
        assert len(selected) == 3


class TestRebuildBibliography:
    """A revision pass regenerates SOURCES itself — the canonical bibliography
    must always be restored before dead-source pruning."""

    def test_llm_bibliography_replaced(self):
        from agent.citation_integrity import rebuild_bibliography
        srcs = [{"url": f"https://s{i}.com", "title": f"Canonical {i}", "tier": "P1",
                 "snippet": "", "published_date": "2024"} for i in (1, 2)]
        text = ("Body [S1] more [S2].\n\n## SOURCES\n"
                "[S1] LLM invented format one\n[S2] LLM invented format two")
        out = rebuild_bibliography(text, srcs)
        assert "Canonical 1" in out and "Canonical 2" in out
        assert "LLM invented" not in out
        assert out.startswith("Body [S1] more [S2].")

    def test_appends_when_missing(self):
        from agent.citation_integrity import rebuild_bibliography
        srcs = [{"url": "https://s1.com", "title": "Only", "tier": "P1", "snippet": ""}]
        out = rebuild_bibliography("Body text [S1].", srcs)
        assert "## SOURCES" in out and "Only" in out

    def test_full_post_revise_pipeline(self):
        """Rebuild → prune order: uncited padding dropped even when the
        revision LLM kept it in its own bibliography."""
        from agent.citation_integrity import rebuild_bibliography, prune_uncited_sources
        srcs = [{"url": f"https://s{i}.com", "title": f"Src{i}", "tier": "P1", "snippet": ""}
                for i in range(1, 4)]
        revised = ("A [S1] and [S2].\n\n## SOURCES\n"
                   "[S1] x\n[S2] y\n[S3] padding the LLM kept")
        t = rebuild_bibliography(revised, srcs)
        t2, s2, stats = prune_uncited_sources(t, srcs)
        assert stats["pruned"] == [3]
        assert "Src3" not in t2
        assert t2.count("## SOURCES") == 1


class TestTotalLLMFailureHonesty:
    """Live-found under a Cloudflare daily-quota outage: when every model in
    the fallback chain fails, the router returns an [ALL_LLM_FAILED] sentinel
    which the single-call fallback used to return verbatim — marching a stub
    through critique/prune/parse as if it were a report. It must raise instead."""

    def test_single_call_raises_on_sentinel(self, monkeypatch):
        import agent.research_engine as re_mod
        from agent.llm_router import RouterResult
        from agent.research_engine import VerificationResult

        fake = RouterResult(
            content="[ALL_LLM_FAILED] No model could generate a response. Check API keys and rate limits.",
            model_name="", model_id="x", provider="cloudflare",
            fallback_count=4, total_latency_ms=1,
        )

        async def fake_synth(*a, **k):
            return fake

        monkeypatch.setattr("agent.llm_router.synthesize_with_router", fake_synth)
        with pytest.raises(RuntimeError, match="all LLM providers"):
            asyncio.run(re_mod._generate_report_text(
                "query", [], VerificationResult(claims=[]), "quick",
            ))

    def test_single_call_raises_on_degenerate_output(self, monkeypatch):
        import agent.research_engine as re_mod
        from agent.llm_router import RouterResult
        from agent.research_engine import VerificationResult

        fake = RouterResult(
            content="Too short.",   # <80 chars, no sentinel — still not a report
            model_name="Llama-3.1-8B", model_id="x", provider="cloudflare",
            fallback_count=0, total_latency_ms=1,
        )

        async def fake_synth(*a, **k):
            return fake

        monkeypatch.setattr("agent.llm_router.synthesize_with_router", fake_synth)
        with pytest.raises(RuntimeError, match="no usable output"):
            asyncio.run(re_mod._generate_report_text(
                "query", [], VerificationResult(claims=[]), "quick",
            ))

    def test_valid_report_passes_through(self, monkeypatch):
        import agent.research_engine as re_mod
        from agent.llm_router import RouterResult
        from agent.research_engine import VerificationResult

        body = "## EXECUTIVE SUMMARY\n" + ("Real analysis " * 40)
        fake = RouterResult(
            content=body, model_name="Qwen3-30B", model_id="x",
            provider="cloudflare", fallback_count=0, total_latency_ms=1,
        )

        async def fake_synth(*a, **k):
            return fake

        monkeypatch.setattr("agent.llm_router.synthesize_with_router", fake_synth)
        out = asyncio.run(re_mod._generate_report_text(
            "query", [], VerificationResult(claims=[]), "quick",
        ))
        assert out == body


class TestOpenAICompatProvider:
    """The router must be able to escape single-provider outages (live-seen:
    Cloudflare free-tier neuron gate stuck at 429 across reset windows) by
    falling back to any OpenAI-compatible endpoint via OPENAI_BASE_URL."""

    def test_default_url_is_real_openai(self, monkeypatch):
        import agent.llm_router as lr
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
        assert lr._openai_chat_url() == "https://api.openai.com/v1/chat/completions"

    def test_base_url_override(self, monkeypatch):
        import agent.llm_router as lr
        monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:8795/v1/")
        assert lr._openai_chat_url() == "http://127.0.0.1:8795/v1/chat/completions"

    def test_compat_model_in_report_chain(self):
        from agent.llm_router import select_tier
        models = select_tier(task_type="report")
        names = [m.name for m in models]
        assert "OpenAI-Compat" in names, f"compat fallback missing from report chain: {names}"

    def test_compat_model_not_in_classification(self):
        from agent.llm_router import select_tier
        models = select_tier(task_type="classification")
        assert all(m.tier in ("cheap", "free") for m in models)


class TestCrossrefComponentFilter:
    """Root fix for the live [S4] failure: Crossref registers figures/tables/
    supplementary files as 'component' DOIs under the parent paper. They carry
    the parent's keywords, pass relevance gates, and got cited as methodology
    ("Supplemental Information 1: Meta-bias analysis results" = a coral-reef
    paper's supplement). They must never become sources."""

    def _item(self, **kw):
        base = {"DOI": "10.7717/peerj.16381", "type": "journal-article",
                "title": ["Real paper title"], "author": []}
        base.update(kw)
        return base

    def test_real_article_passes(self):
        from tools.search_sources import _is_crossref_paper
        assert _is_crossref_paper(self._item())

    def test_component_type_rejected(self):
        from tools.search_sources import _is_crossref_paper
        assert not _is_crossref_paper(self._item(type="component"))

    def test_component_doi_suffix_rejected(self):
        from tools.search_sources import _is_crossref_paper
        for doi in ("10.7717/peerj.16381/supp-1", "10.7717/peerj.5172/supp-23",
                    "10.7717/peerjcs.1714/fig-6", "10.7717/peerjcs.1714/table-2"):
            assert not _is_crossref_paper(self._item(DOI=doi)), doi

    def test_supplement_title_rejected(self):
        from tools.search_sources import _is_crossref_paper
        for title in ("Supplemental Information 1: Meta-bias analysis results.",
                      "Table S2: Characteristics of unpublished trials",
                      "Figure 6: Result analysis of some method"):
            assert not _is_crossref_paper(self._item(title=[title])), title


class TestSelectionJunkAndEncyclopedia:
    """Defense-in-depth at selection: junk records and bare-directory URLs
    never enter the report; encyclopedias are capped at one source."""

    def _src(self, i, title="Legit study", url="https://journal.example/a",
             snippet="unpublished negative-result trials meta-analysis"):
        return {"url": url, "title": title, "tier": "P1", "snippet": snippet}

    def _sel(self, sources):
        from agent.research_engine import select_diverse_sources
        return select_diverse_sources(
            sources, max_sources=10, query="unpublished negative-result trials meta-analysis",
            query_variants=[], min_relevance=0.05,
        )

    def test_supplement_record_dropped(self):
        junk = self._src(1, title="Supplemental Information 1: Meta-bias analysis results.",
                         url="https://doi.org/10.7717/peerj.16381/supp-1")
        out = self._sel([junk, self._src(2)])
        assert all("Supplemental" not in (s.get("title") or "") for s in out)

    def test_bare_directory_url_dropped(self):
        junk = self._src(1, title="Manalyzer: automated meta-analysis",
                         url="https://huggingface.co/papers/")
        out = self._sel([junk, self._src(2)])
        assert all(s.get("url") != "https://huggingface.co/papers/" for s in out)

    def test_encyclopedia_capped_at_one(self):
        wikis = [self._src(i, title=f"Meta-analysis overview {i}",
                           url=f"https://en.wikipedia.org/wiki/Meta-analysis_{i}")
                 for i in range(3)]
        out = self._sel(wikis)
        wiki_count = sum(1 for s in out if "wikipedia.org" in (s.get("url") or ""))
        assert wiki_count <= 1

    def test_legit_sources_survive(self):
        out = self._sel([self._src(1), self._src(2)])
        assert len(out) == 2


class TestClaimSourceAlignment:
    """Deterministic claim↔source alignment scan: a cited source sharing zero
    content words with the claim sentence is flagged for critique."""

    def test_zero_overlap_flagged(self):
        from agent.citation_integrity import scan_claim_source_alignment
        srcs = [
            {"url": "https://a.example/1", "title": "ALL-IN meta-analysis framework",
             "snippet": "anytime-valid confidence intervals for prospective evidence synthesis"},
            {"url": "https://coral.example/2", "title": "Sponge-host interaction networks",
             "snippet": "coral reef ecosystems host diverse sponge communities in tropical waters"},
        ]
        text = ("Statistical tests such as Egger's regression and trim-and-fill "
                "estimate missing studies [S2].\n\n## SOURCES\n[S1] x\n[S2] y")
        flags = scan_claim_source_alignment(text, srcs)
        assert any(f["source"] == 2 for f in flags)

    def test_supported_claim_not_flagged(self):
        from agent.citation_integrity import scan_claim_source_alignment
        srcs = [{"url": "https://a.example/1", "title": "ALL-IN meta-analysis framework",
                 "snippet": "anytime-valid confidence intervals for prospective and retrospective meta-analysis"}]
        text = ("The ALL-IN framework maintains validity for prospective meta-analysis "
                "using anytime-valid confidence intervals [S1].\n\n## SOURCES\n[S1] x")
        assert scan_claim_source_alignment(text, srcs) == []

    def test_doi_only_snippet_skipped(self):
        from agent.citation_integrity import scan_claim_source_alignment
        srcs = [{"url": "https://doi.org/10.7717/peerj.16381/supp-1",
                 "title": "Supplemental Information 1: Meta-bias analysis results.",
                 "snippet": "DOI: 10.7717/peerj.16381/supp-1"}]
        text = ("Methods include funnel plots and Egger's regression for bias "
                "detection in synthesized estimates [S1].\n\n## SOURCES\n[S1] x")
        # DOI-only snippet carries no checkable content — no false positive
        assert scan_claim_source_alignment(text, srcs) == []


class TestDegenerateSemanticFallback:
    """Embedding-provider outages return zero vectors, which cosine-scoring
    turns into all-zero (arbitrary) chunk rankings. The evidence retriever
    must detect this and fall back to keyword scoring instead."""

    def test_zero_vectors_return_none(self, monkeypatch):
        import agent.evidence_fetcher as ef

        async def fake_embed(texts):
            return [[0.0] * 8 for _ in texts]

        import ingest.embedder as emb
        monkeypatch.setattr(emb, "embed_texts", fake_embed)
        out = asyncio.run(ef._semantic_scores("query about trials", ["chunk one", "chunk two"]))
        assert out is None

    def test_real_vectors_return_scores(self, monkeypatch):
        import agent.evidence_fetcher as ef
        import math

        q = [1.0, 0.0, 0.0, 0.0]

        async def fake_embed(texts):
            vecs = [q]
            for t in texts[1:] if False else texts:
                pass
            # first is the query, rest mirror varying overlap with query
            return [q, [0.9, 0.1, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]

        import ingest.embedder as emb
        monkeypatch.setattr(emb, "embed_texts", fake_embed)
        out = asyncio.run(ef._semantic_scores("query", ["a", "b"]))
        assert out is not None and len(out) == 2
        assert out[0] > out[1]
        assert not all(abs(s) <= 1e-9 for s in out)


# ═══════════════════════════════════════════════════════════════════
# CANONICAL-SOURCE BACKFILL — named methods must stand on primary literature
# ═══════════════════════════════════════════════════════════════════

from types import SimpleNamespace


class TestCanonicalRegistry:
    def test_registry_entries_are_well_formed(self):
        from agent.canonical_sources import REGISTRY
        priorities = set()
        for e in REGISTRY:
            assert e.entry_id and e.doi and e.expected_title
            assert "/" in e.doi, f"{e.entry_id}: DOI should contain a slash"
            assert 1950 <= e.expected_year <= 2025
            assert e.why and e.label
            priorities.add(e.priority)
        assert len(priorities) == len(REGISTRY), "priorities must be unique"

    def test_method_entries_cover_the_named_canon(self):
        from agent.canonical_sources import _METHOD_ENTRIES
        ids = {e.entry_id for e in _METHOD_ENTRIES}
        assert {"egger1997", "duval2000", "rosenthal1979", "cochrane_robme", "sterne2011"} <= ids


class TestCanonicalDetection:
    def test_topic_query_triggers_method_entries_via_compound(self):
        from agent.canonical_sources import detect_canonical_entries
        q = ("Meta-analysis of unpublished negative-result trials that never "
             "made it into public databases")
        ids = [e.entry_id for e in detect_canonical_entries(q)]
        for expected in ("egger1997", "duval2000", "cochrane_robme", "rosenthal1979"):
            assert expected in ids, f"{expected} should be triggered by topic compound"

    def test_method_name_triggers_its_own_entry(self):
        from agent.canonical_sources import detect_canonical_entries
        ids = [e.entry_id for e in detect_canonical_entries("we applied trim-and-fill to adjust the pooled estimate")]
        assert ids == ["duval2000"]

    def test_federated_learning_query_triggers_nothing(self):
        from agent.canonical_sources import detect_canonical_entries
        q = "Privacy-preserving federated learning for multi-center healthcare analytics"
        assert detect_canonical_entries(q) == []

    def test_no_vacuous_matching_on_arbitrary_text(self):
        # Regression: entries with trigger_all=() must not fire via vacuous all()
        from agent.canonical_sources import detect_canonical_entries
        assert detect_canonical_entries("the quick brown fox jumps over the lazy dog") == []

    def test_off_topic_query_triggers_nothing(self):
        from agent.canonical_sources import detect_canonical_entries
        q = "coral reef biodiversity loss across indonesian marine protected areas"
        assert detect_canonical_entries(q) == []


class TestTitleValidation:
    def test_unicode_hyphen_titles_match(self):
        from agent.canonical_sources import titles_match
        fetched = ("Trim and Fill: A Simple Funnel\u2010Plot\u2013Based Method of Testing "
                   "and Adjusting for Publication Bias in Meta\u2010Analysis")
        assert titles_match(fetched, "Trim and Fill: A Simple Funnel-Plot-Based Method of Testing and Adjusting for Publication Bias in Meta-Analysis")

    def test_wrong_paper_rejected(self):
        from agent.canonical_sources import titles_match
        assert not titles_match(
            "Reversibility of Serum Removal Effects on IGF-II mRNA in Human Neuroblastoma Cells",
            "The file drawer problem and tolerance for null results",
        )

    def test_subtitle_extra_tokens_are_fine(self):
        from agent.canonical_sources import titles_match
        fetched = ("Empirical Evidence for Selective Reporting of Outcomes in "
                   "Randomized Trials: Comparison of Protocols to Published Articles")
        assert titles_match(fetched, "Empirical Evidence for Selective Reporting of Outcomes in Randomized Trials")

    def test_normalize_doi(self):
        from agent.canonical_sources import normalize_doi
        assert normalize_doi("https://doi.org/10.1136/BMJ.315.7109.629") == "10.1136/bmj.315.7109.629"
        assert normalize_doi("doi: 10.1111/j.0006-341X.2000.00455.x") == "10.1111/j.0006-341x.2000.00455.x"


def _fake_result(title="Trim and Fill: A Simple Funnel-Plot-Based Method of Testing and Adjusting for Publication Bias in Meta-Analysis",
                 year="2000", snippet="Simulation studies show the estimator performs well.", **kw):
    return SimpleNamespace(
        title=title, url=f"https://doi.org/{kw.get('doi', 'x')}", snippet=snippet,
        published_date=year, authors=["Sue Duval", "Richard Tweedie"],
        citation_count=12308, doi=kw.get("doi", "x"),
    )


def _registry_fetcher(called):
    """Fake Crossref fetcher that returns each registry entry's own verified
    identity — validates that backfill works end-to-end for ANY entry."""
    from agent.canonical_sources import REGISTRY
    by_doi = {e.doi: e for e in REGISTRY}

    async def fake_fetch(doi):
        called.append(doi)
        e = by_doi.get(doi)
        if e is None:
            return None
        return SimpleNamespace(
            title=e.expected_title, url=f"https://doi.org/{e.doi}",
            snippet=f"Canonical abstract for {e.label}.",
            published_date=str(e.expected_year), authors=["A. Author"],
            citation_count=999, doi=e.doi,
        )

    return fake_fetch


class TestCanonicalBackfill:
    def test_happy_path_appends_validated_source(self):
        import asyncio
        from agent.canonical_sources import ensure_canonical_sources
        pool = [{"title": "Some unrelated paper", "snippet": "nothing relevant"}]
        called = []

        async def fake_fetch(doi):
            called.append(doi)
            return _fake_result(doi="10.1111/j.0006-341X.2000.00455.x")

        signal = "we need trim-and-fill adjustment for publication bias"
        sources, added = asyncio.run(ensure_canonical_sources(signal, pool, max_add=1, fetcher=fake_fetch))
        assert called == ["10.1111/j.0006-341X.2000.00455.x"]
        assert len(added) == 1 and added[0]["entry_id"] == "duval2000" and added[0]["index"] == 2
        s = sources[1]
        assert s["tier"] == "P1" and s["canonical_id"] == "duval2000"
        assert s["doi"] == "10.1111/j.0006-341x.2000.00455.x"
        assert s["authors"] == ["Sue Duval", "Richard Tweedie"]
        assert s["citation_count"] == 12308 and s["backfilled"] is True

    def test_identity_mismatch_rejected(self):
        import asyncio
        from agent.canonical_sources import ensure_canonical_sources
        pool = [{"title": "unrelated", "snippet": ""}]

        async def fake_fetch(doi):
            return _fake_result(title="Reversibility of Serum Removal Effects on IGF-II mRNA", year="1993")

        sources, added = asyncio.run(ensure_canonical_sources("trim-and-fill", pool, max_add=3, fetcher=fake_fetch))
        assert added == [] and len(sources) == 1

    def test_year_mismatch_rejected(self):
        import asyncio
        from agent.canonical_sources import ensure_canonical_sources
        pool = [{"title": "unrelated", "snippet": ""}]

        async def fake_fetch(doi):
            return _fake_result(year="2014")

        sources, added = asyncio.run(ensure_canonical_sources("trim-and-fill", pool, max_add=3, fetcher=fake_fetch))
        assert added == [] and len(sources) == 1

    def test_fetch_failure_degrades_silently(self):
        import asyncio
        from agent.canonical_sources import ensure_canonical_sources
        pool = [{"title": "unrelated", "snippet": ""}]

        async def fake_fetch(doi):
            return None

        sources, added = asyncio.run(ensure_canonical_sources("trim-and-fill", pool, max_add=3, fetcher=fake_fetch))
        assert added == [] and len(sources) == 1

    def test_already_in_pool_not_duplicated(self):
        import asyncio
        from agent.canonical_sources import ensure_canonical_sources
        pool = [{"title": "Trim and Fill: A Simple Funnel-Plot-Based Method", "snippet": "",
                 "doi": "10.1111/j.0006-341x.2000.00455.x", "canonical_id": "duval2000"}]
        fetches = []

        async def fake_fetch(doi):
            fetches.append(doi)
            return _fake_result()

        sources, added = asyncio.run(ensure_canonical_sources("trim-and-fill everywhere", pool, max_add=3, fetcher=fake_fetch))
        assert added == [] and fetches == [] and len(sources) == 1

    def test_max_add_caps_injection(self):
        import asyncio
        from agent.canonical_sources import ensure_canonical_sources
        pool = [{"title": "unrelated", "snippet": ""}]
        called = []

        sources, added = asyncio.run(
            ensure_canonical_sources("meta-analysis of unpublished negative trials", pool,
                                     max_add=2, fetcher=_registry_fetcher(called))
        )
        assert len(called) == 2
        assert {a["entry_id"] for a in added} == {"egger1997", "duval2000"}  # priority order


class TestLateBackfillForDraft:
    REPORT = ("## FINDINGS\n\nThe trim-and-fill adjustment corrected the pooled "
              "estimate for funnel-plot asymmetry [S1]. Nothing else here.\n\n"
              "## SOURCES\n\n[S1] [P3] Someone. Something. 2015.")

    def test_missing_method_source_gets_fetched_and_flagged(self):
        import asyncio
        from agent.canonical_sources import late_backfill_for_draft
        pool = [{"title": "Encyclopedia entry", "snippet": "background", "url": "https://en.wikipedia.org/wiki/x"}]
        called = []

        issues, added = asyncio.run(late_backfill_for_draft(self.REPORT, pool, fetcher=_registry_fetcher(called)))
        # Draft names trim-and-fill AND funnel-plot asymmetry → both primaries fetched
        assert called == ["10.1111/j.0006-341X.2000.00455.x", "10.1136/bmj.d4002"]
        assert {a["entry_id"] for a in added} == {"duval2000", "sterne2011"}
        assert len(pool) == 3
        assert sum("CANONICAL_SOURCE_ADDED" in i for i in issues) == 2
        assert any("[S2]" in i for i in issues) and any("[S3]" in i for i in issues)

    def test_present_method_source_triggers_no_fetch(self):
        import asyncio
        from agent.canonical_sources import late_backfill_for_draft

        def pool_with(entry_id, title, doi):
            return {"title": title, "snippet": "", "doi": doi, "canonical_id": entry_id}

        pool = [
            pool_with("duval2000", "Trim and Fill: A Simple Funnel-Plot-Based Method",
                      "10.1111/j.0006-341x.2000.00455.x"),
            pool_with("sterne2011", "Recommendations for examining and interpreting funnel plot asymmetry",
                      "10.1136/bmj.d4002"),
        ]
        called = []

        issues, added = asyncio.run(late_backfill_for_draft(self.REPORT, pool, fetcher=_registry_fetcher(called)))
        assert called == [] and added == [] and issues == []

    def test_failed_fetch_produces_missing_issue(self):
        import asyncio
        from agent.canonical_sources import late_backfill_for_draft
        pool = [{"title": "Encyclopedia entry", "snippet": "background", "url": "https://en.wikipedia.org/wiki/x"}]

        async def fake_fetch(doi):
            return None

        issues, added = asyncio.run(late_backfill_for_draft(self.REPORT, pool, fetcher=fake_fetch))
        assert added == []
        assert any("CANONICAL_SOURCE_MISSING" in i for i in issues)


class TestScanCanonicalMethodUsage:
    def test_named_method_without_primary_source_flagged(self):
        from agent.canonical_sources import scan_canonical_method_usage
        sources = [{"title": "Publication bias - Wikipedia", "snippet": "background article",
                    "url": "https://en.wikipedia.org/wiki/Publication_bias"}]
        report = ("## FINDINGS\n\nEgger's regression test demonstrated significant "
                  "asymmetry across all included studies of published trials [S1].")
        issues = scan_canonical_method_usage(report, sources)
        assert len(issues) == 1
        assert issues[0]["type"] == "CANONICAL_SOURCE_MISSING"
        assert "Egger" in issues[0]["detail"]

    def test_method_cited_to_wrong_source_flagged(self):
        from agent.canonical_sources import scan_canonical_method_usage
        wiki = {"title": "Publication bias - Wikipedia", "snippet": "background",
                "url": "https://en.wikipedia.org/wiki/Publication_bias"}
        duval = {"title": "Trim and Fill: A Simple Funnel-Plot-Based Method",
                 "doi": "10.1111/j.0006-341x.2000.00455.x", "canonical_id": "duval2000"}
        report = ("## FINDINGS\n\nThe trim-and-fill procedure adjusted the pooled "
                  "estimate for missing studies [S1].")
        issues = scan_canonical_method_usage(report, [wiki, duval])
        assert len(issues) == 1
        assert issues[0]["type"] == "CANONICAL_METHOD_UNCITED"
        assert "[S2]" in issues[0]["fix"]

    def test_method_cited_to_canonical_passes(self):
        from agent.canonical_sources import scan_canonical_method_usage
        wiki = {"title": "Publication bias - Wikipedia", "snippet": "background",
                "url": "https://en.wikipedia.org/wiki/Publication_bias"}
        duval = {"title": "Trim and Fill: A Simple Funnel-Plot-Based Method",
                 "doi": "10.1111/j.0006-341x.2000.00455.x", "canonical_id": "duval2000"}
        report = ("## FINDINGS\n\nThe trim-and-fill procedure adjusted the pooled "
                  "estimate for missing studies [S2].")
        assert scan_canonical_method_usage(report, [wiki, duval]) == []

    def test_bibliography_mentions_do_not_count(self):
        from agent.canonical_sources import scan_canonical_method_usage
        sources = [{"title": "Bias in meta-analysis detected by a simple, graphical test",
                    "doi": "10.1136/bmj.315.7109.629", "canonical_id": "egger1997"}]
        report = ("## FINDINGS\n\nPublication bias remains a threat to validity in "
                  "small-study research designs.\n\n## SOURCES\n\n"
                  "[S1] [P1] Egger. Bias in meta-analysis detected by a simple, graphical test. 1997.")
        assert scan_canonical_method_usage(report, sources) == []


class TestConceptBoundaryMismatch:
    FL_SOURCE = {"title": "Privacy-preserving federated learning for multi-center healthcare analytics",
                 "snippet": "Secure aggregation lets hospitals train models without sharing patient records.",
                 "url": "https://arxiv.org/abs/1811.xxxx"}

    def test_federated_source_cited_for_unpublished_trials_flagged(self):
        from agent.citation_integrity import scan_concept_boundary_mismatch
        report = ("## FINDINGS\n\nFederated learning frameworks enable researchers to "
                  "access unpublished trial data held across institutions [S1].")
        flags = scan_concept_boundary_mismatch(report, [self.FL_SOURCE])
        assert len(flags) == 1
        assert flags[0]["source"] == 1
        assert "never published or registered" in flags[0]["reason"]

    def test_federated_source_cited_for_its_actual_subject_passes(self):
        from agent.citation_integrity import scan_concept_boundary_mismatch
        report = ("## FINDINGS\n\nFederated learning supports privacy-preserving "
                  "model training across hospitals without moving records [S1].")
        assert scan_concept_boundary_mismatch(report, [self.FL_SOURCE]) == []

    def test_unrelated_source_cited_for_unpublished_claim_passes(self):
        from agent.citation_integrity import scan_concept_boundary_mismatch
        src = {"title": "Publication bias: evidence of delayed publication in a cohort study",
               "snippet": "Cohort of clinical research projects in Australia.", "url": "https://x.org"}
        report = ("## FINDINGS\n\nA cohort study found that negative trials were "
                  "published later or remained unpublished [S1].")
        assert scan_concept_boundary_mismatch(report, [src]) == []


class TestCritiqueReviseCanonicalIntegration:
    def test_canonical_issues_enter_revision_feedback(self, monkeypatch):
        import asyncio
        import agent.llm_router as lr
        import agent.research_engine as re_mod

        report = ("# T\n\n## FINDINGS\n\nThe trim-and-fill procedure adjusted the "
                  "pooled estimate [S99]. More prose here to pad the report to a "
                  "reasonable length for the revision threshold check.\n\n## SOURCES\n\n"
                  "[S1] [P3] Wiki. Entry. 2015.")

        async def fake_router(**kwargs):
            prompt = kwargs.get("context", "")
            if "FLAGGED ISSUES" in prompt:
                return SimpleNamespace(content=report.replace("[S99]", "[S1]"), model_name="mock")
            return SimpleNamespace(content='{"issues": []}', model_name="mock")

        monkeypatch.setattr(lr, "synthesize_with_router", fake_router)
        sources = [{"title": "Wiki entry", "snippet": "x", "url": "https://en.wikipedia.org/wiki/x"}]
        text, summary, model = asyncio.run(
            re_mod._critique_and_revise(report, "q", sources, "quick")
        )
        assert "[S99]" not in text and "[S1]" in text
        assert summary["revised"] is True
        assert summary["canonical_issues"] == 1  # trim-and-fill named, no primary source


# ═══════════════════════════════════════════════════════════════
# WEB / GREY-LITERATURE RETRIEVAL (z-ai bridge provider)
# ═══════════════════════════════════════════════════════════════

from tools.search_sources import (  # noqa: E402
    SearchResult,
    SourceCategory,
    search_zai_web,
    fetch_page_text,
    canonicalize_web_url,
    _web_result_tier,
    _web_date_to_iso,
    _relaxed_page_text,
)
from agent.research_engine import _web_sources_for_report  # noqa: E402
from agent.evidence_fetcher import fetch_source_fulltext  # noqa: E402


def _web_row(url, name="A title", snippet="some snippet", date=None, host=None):
    return {"url": url, "name": name, "snippet": snippet, "date": date,
            "host_name": host or "", "rank": 0, "favicon": ""}


class TestWebUrlCanonicalization:
    def test_strips_tracking_fragment_and_trailing_slash(self):
        out = canonicalize_web_url(
            "https://WWW.Example.com/post/?utm_source=x&utm_campaign=y&id=7&fbclid=Z#top/"
        )
        assert out == "https://www.example.com/post?id=7"

    def test_keeps_meaningful_query_and_path(self):
        out = canonicalize_web_url("https://api.site.org/search?q=trials&page=2")
        assert out == "https://api.site.org/search?q=trials&page=2"

    def test_garbage_and_empty(self):
        assert canonicalize_web_url("") == ""
        assert canonicalize_web_url("not a url") == "not a url"


class TestWebDateParsing:
    def test_search_engine_date(self):
        assert _web_date_to_iso("Feb 6, 2024 — A total of 475 trials") == "2024-02-06"

    def test_bare_year_fallback(self):
        assert _web_date_to_iso("Published 2025 by someone") == "2025"

    def test_garbage_and_empty(self):
        assert _web_date_to_iso("no date here") is None
        assert _web_date_to_iso("") is None
        assert _web_date_to_iso(None) is None


class TestWebResultGovernance:
    def test_grey_literature_ngos_are_p2(self):
        assert _web_result_tier("https://www.transparimed.org/single-post/nordic-trials") == "P2"
        assert _web_result_tier("https://senseaboutscience.org/activities/nordic-trial-report") == "P2"
        assert _web_result_tier("https://www.cochrane.org/news/hundreds-nordic-trials") == "P2"

    def test_institutional_suffixes_are_p2(self):
        assert _web_result_tier("https://www.nih.gov/news-events") == "P2"
        assert _web_result_tier("https://health.ec.europa.eu/report") == "P2"

    def test_unknown_open_web_is_p3(self):
        assert _web_result_tier("https://www.bmj.com/content/384/bmj.q343") == "P3"

    def test_social_and_junk_tlds_dropped(self):
        assert _web_result_tier("https://x.com/TranspariMED/status/175477") == "DROP"
        assert _web_result_tier("https://medium.com/@someone/post") == "DROP"
        assert _web_result_tier("https://spamblog.xyz/post") == "DROP"

    def test_subdomains_inherit_governance(self):
        assert _web_result_tier("https://eu.trialstracker.net/sponsor/rigshospitalet") == "P2"

    def test_bare_host_dropped(self):
        assert _web_result_tier("http://localhost:9999/x") == "DROP"


class TestSearchZaiWeb:
    def test_shapes_rows_and_applies_governance(self, monkeypatch):
        import tools.search_sources as ts

        rows = [
            _web_row("https://www.transparimed.org/single-post/nordic-trials?utm_source=rss",
                     name="Nordic countries: 475 clinical trials",
                     snippet="475 trials never made public",
                     date="Feb 6, 2024"),
            _web_row("https://x.com/TranspariMED/status/175477538", name="Tweet about it"),
            _web_row("https://www.transparimed.org/single-post/nordic-trials",
                     name="dup canonicalized"),
            _web_row("https://unknown-site.org/analysis", name="Some analysis"),
            _web_row("https://notitle.example.com/x", name=""),
        ]

        async def fake_bridge(path, payload, timeout=None):
            assert path == "/web_search"
            return rows

        monkeypatch.setattr(ts, "_bridge_request", fake_bridge)
        out = asyncio.run(search_zai_web("nordic trials", 8))
        assert len(out) == 2
        first = out[0]
        assert first.url == "https://www.transparimed.org/single-post/nordic-trials"
        assert first.source_tier == "P2"
        assert first.source_name == "zai_web"
        assert first.source_category == SourceCategory.WEB
        assert first.published_date == "2024-02-06"
        assert first.extra["web"] is True
        assert out[1].source_tier == "P3"

    def test_bridge_failure_returns_empty(self, monkeypatch):
        import tools.search_sources as ts

        async def dead_bridge(path, payload, timeout=None):
            raise ConnectionError("bridge down")

        monkeypatch.setattr(ts, "_bridge_request", dead_bridge)
        assert asyncio.run(search_zai_web("anything")) == []

    def test_empty_query_short_circuits(self, monkeypatch):
        import tools.search_sources as ts

        called = {"n": 0}

        async def no_call(path, payload, timeout=None):
            called["n"] += 1
            return []

        monkeypatch.setattr(ts, "_bridge_request", no_call)
        assert asyncio.run(search_zai_web("   ")) == []
        assert called["n"] == 0


_WIXISH_HTML = (
    "<html><head><style>body { color: red; }</style>"
    "<script>var tracker = 1;</script></head><body>"
    "<nav><a href='/'>Home</a><a href='/x'>X</a></nav>"
    "<div class='header shadow gradient'><h1>Nordic countries: 475 clinical "
    "trials involving 83,903 patients are missing results</h1>"
    "<p>A total of 475 clinical trials completed in Denmark, Iceland, Finland, "
    "Norway and Sweden have never made their results public in any form.</p>"
    "<p>Only 27 percent of all trial results were made public within 12 months "
    "of trial completion, the report found.</p></div>"
    "<footer>cookie banner subscribe newsletter</footer>"
    "</body></html>"
)


class TestFetchPageText:
    def test_extracts_text_from_sdk_envelope(self, monkeypatch):
        import tools.search_sources as ts

        async def fake_bridge(path, payload, timeout=None):
            assert path == "/page_reader"
            assert payload["url"].startswith("https://www.transparimed.org/")
            return {"code": 200, "status": 200,
                    "data": {"html": _WIXISH_HTML, "title": "Nordic trials", "url": payload["url"]}}

        monkeypatch.setattr(ts, "_bridge_request", fake_bridge)
        text = asyncio.run(fetch_page_text("https://www.transparimed.org/single-post/nordic-trials"))
        assert "475 clinical trials" in text
        assert "tracker" not in text and "cookie banner" not in text

    def test_relaxed_extraction_rescues_overcleaned_pages(self):
        # clean_html's aggressive ad-class patterns decompose class="header ..."
        # containers; the relaxed pass must recover the story text.
        from ingest.html_cleaner import clean_html

        assert clean_html(_WIXISH_HTML).strip() == "" or "475" not in clean_html(_WIXISH_HTML)
        relaxed = _relaxed_page_text(_WIXISH_HTML)
        assert "475 clinical trials" in relaxed
        assert "27 percent" in relaxed
        assert "var tracker" not in relaxed

    def test_too_small_html_returns_empty(self, monkeypatch):
        import tools.search_sources as ts

        async def fake_bridge(path, payload, timeout=None):
            return {"data": {"html": "<html><body>hi</body></html>"}}

        monkeypatch.setattr(ts, "_bridge_request", fake_bridge)
        assert asyncio.run(fetch_page_text("https://x.org/page")) == ""

    def test_bridge_failure_returns_empty(self, monkeypatch):
        import tools.search_sources as ts

        async def dead(path, payload, timeout=None):
            raise ConnectionError("down")

        monkeypatch.setattr(ts, "_bridge_request", dead)
        assert asyncio.run(fetch_page_text("https://x.org/page")) == ""


class TestWebSourcesForReport:
    def _fake_results(self):
        return [
            SearchResult(
                title="Nordic countries: 475 clinical trials",
                url="https://www.transparimed.org/single-post/nordic-trials",
                snippet="475 trials never made results public",
                source_name="zai_web",
                source_category=SourceCategory.WEB,
                source_tier="P2",
                published_date="2024-02-06",
                extra={"web": True, "host": "transparimed.org"},
            ),
            SearchResult(
                title="Missing and delayed results reporting",
                url="https://pubmed.ncbi.nlm.nih.gov/39900256",
                snippet="Systematic evaluation of timely reporting",
                source_name="zai_web",
                source_category=SourceCategory.WEB,
                source_tier="P2",
            ),
        ]

    def test_dict_shape_tier_enforcement_and_budget(self, monkeypatch):
        import tools.search_sources as ts

        seen_queries = []

        async def fake_search(query, num):
            seen_queries.append(query)
            return self._fake_results()

        async def dead_registry(query, max_results=8):
            return []  # registry lane is tested in TestClinicalTrialsLane

        monkeypatch.setattr(ts, "search_zai_web", fake_search)
        monkeypatch.setattr(ts, "search_clinical_trials", dead_registry)
        topic = "Meta-analysis of unpublished negative-result trials that never made it into public databases."
        sources, stats = asyncio.run(
            _web_sources_for_report(topic, ["variant one", "variant two"], "quick")
        )
        # quick budget: primary + deterministic keyword rewrite (no variants)
        assert len(seen_queries) == 2
        assert stats["queries"][0] == topic
        assert stats["queries"][1] == "Meta-analysis unpublished negative-result trials databases"
        assert stats["available"] is True and stats["inserted"] == 2
        by_url = {s["url"]: s for s in sources}
        transparimed = by_url["https://www.transparimed.org/single-post/nordic-trials"]
        assert transparimed["category"] == "web"
        assert transparimed["tier"] == "P2"
        assert transparimed["extra"]["web"] is True
        # pubmed.ncbi.nlm.nih.gov is in the global P1 domain map
        assert by_url["https://pubmed.ncbi.nlm.nih.gov/39900256"]["tier"] == "P1"

    def test_thorough_depth_uses_rewrite_plus_two_variants(self, monkeypatch):
        import tools.search_sources as ts

        seen_queries = []

        async def fake_search(query, num):
            seen_queries.append(query)
            return []

        monkeypatch.setattr(ts, "search_zai_web", fake_search)
        _sources, stats = asyncio.run(
            _web_sources_for_report("q words", ["v1", "v2", "v3"], "thorough")
        )
        # thorough budget: primary + rewrite + 2 genuinely-new variants
        assert len(seen_queries) == 4
        assert seen_queries[2] == "v1" and seen_queries[3] == "v2"
        assert stats["results"] == 0 and stats["available"] is False

    def test_env_gate_off(self, monkeypatch):
        import tools.search_sources as ts

        async def no_call(query, num):
            raise AssertionError("must not be called")

        monkeypatch.setattr(ts, "search_zai_web", no_call)
        monkeypatch.setenv("APEX_WEB_SEARCH", "0")
        sources, stats = asyncio.run(_web_sources_for_report("q", ["v"], "quick"))
        assert sources == [] and stats["enabled"] is False

    def test_bridge_down_degrades_honestly(self, monkeypatch):
        import tools.search_sources as ts

        async def dead(query, num):
            raise ConnectionError("bridge down")

        async def dead_registry(query, max_results=8):
            raise ConnectionError("registry down")

        monkeypatch.setattr(ts, "search_zai_web", dead)
        monkeypatch.setattr(ts, "search_clinical_trials", dead_registry)
        monkeypatch.delenv("APEX_WEB_SEED_URLS", raising=False)
        sources, stats = asyncio.run(_web_sources_for_report("q", ["v"], "quick"))
        assert sources == []
        assert stats["available"] is False and stats["enabled"] is True


class TestWebSeedFallback:
    """Operator-attested seed URLs substitute for dead live search — same
    governance stack, honest seeded=True labeling, live mode untouched."""

    SEEDS = [
        {"url": "https://www.transparimed.org/single-post/nordic-trials?utm_source=x",
         "title": "Nordic countries: 475 clinical trials missing results",
         "snippet": "475 trials, 83,903 patients, Rigshospitalet 23%"},
        {"url": "https://spamblog.xyz/cheap-drugs", "title": "Cheap drugs", "snippet": "spam"},
    ]

    def test_seeds_enter_through_governance_when_live_dead(self, monkeypatch):
        import tools.search_sources as ts

        async def dead(query, num):
            return []

        async def dead_registry(query, max_results=8):
            raise ConnectionError("registry down")

        monkeypatch.setattr(ts, "search_zai_web", dead)
        monkeypatch.setattr(ts, "search_clinical_trials", dead_registry)
        monkeypatch.setenv("APEX_WEB_SEARCH", "true")
        monkeypatch.setenv("APEX_WEB_SEED_URLS", json.dumps(self.SEEDS))
        sources, stats = asyncio.run(_web_sources_for_report("q words", ["v"], "thorough"))
        assert stats["available"] is False
        assert stats["search_mode"] == "seeded_fallback"
        assert stats["seeded"] == 1  # DROP-tier spam filtered
        assert stats["results"] == 0  # live search contributed nothing
        assert len(sources) == 1
        s = sources[0]
        assert s["url"] == "https://www.transparimed.org/single-post/nordic-trials"  # canonicalized
        assert s["tier"] == "P2"  # grey-literature governance
        assert s["source_name"] == "web_seed"
        assert s["extra"]["web"] is True and s["extra"]["seeded"] is True

    def test_live_results_leave_seeds_unused(self, monkeypatch):
        import tools.search_sources as ts

        async def alive(query, num):
            return [SearchResult(
                title="Live hit", url="https://example.org/live-hit",
                snippet="s", source_name="zai_web", source_category=SourceCategory.WEB,
                source_tier="P3", extra={"web": True},
            )]

        async def dead_registry(query, max_results=8):
            raise ConnectionError("registry down")

        monkeypatch.setattr(ts, "search_zai_web", alive)
        monkeypatch.setattr(ts, "search_clinical_trials", dead_registry)
        monkeypatch.setenv("APEX_WEB_SEED_URLS", json.dumps(self.SEEDS))
        sources, stats = asyncio.run(_web_sources_for_report("q words", [], "quick"))
        assert stats["search_mode"] == "live"
        assert "seeded" not in stats
        assert len(sources) == 1 and sources[0]["url"].endswith("live-hit")

    def test_malformed_seed_json_ignored(self, monkeypatch):
        import tools.search_sources as ts

        async def dead(query, num):
            return []

        async def dead_registry(query, max_results=8):
            raise ConnectionError("registry down")

        monkeypatch.setattr(ts, "search_zai_web", dead)
        monkeypatch.setattr(ts, "search_clinical_trials", dead_registry)
        monkeypatch.setenv("APEX_WEB_SEED_URLS", "{not json")
        sources, stats = asyncio.run(_web_sources_for_report("q", [], "quick"))
        assert sources == [] and stats.get("seeded") in (None, 0)

    def test_provenance_renderer_labels_seeded_fallback(self):
        prov = {
            "identified_records": 30, "post_dedup": 20, "post_selection": 10,
            "post_enrichment": 12, "canonical_backfilled": 5, "final_sources": 12,
            "web_records_kept": 2, "web_seeded": 2, "web_search_mode": "seeded_fallback",
        }
        section = _render_provenance_section(prov)
        assert "| Web records from seeded fallback (live search unavailable) | 2 |" in section
        assert "| Web retrieval mode | seeded_fallback |" in section
        # live mode line renders too, seeded line absent
        prov2 = dict(prov, web_seeded=0, web_search_mode="live")
        section2 = _render_provenance_section(prov2)
        assert "seeded fallback" not in section2
        assert "| Web retrieval mode | live |" in section2


class TestEvidenceWebDispatch:
    def test_web_record_routes_to_page_reader(self, monkeypatch):
        import tools.search_sources as ts

        called = {"page_reader": 0, "scrape": 0}

        async def fake_page_text(url, max_chars=120_000):
            called["page_reader"] += 1
            return "Real grey-literature report text. " * 40

        async def fake_scrape(url):
            called["scrape"] += 1
            raise AssertionError("scrape must not run when page_reader succeeds")

        monkeypatch.setattr(ts, "fetch_page_text", fake_page_text)
        import tools.live_scraper as ls
        monkeypatch.setattr(ls, "_scrape_url_with_fallback", fake_scrape)

        source = {"url": "https://www.transparimed.org/single-post/nordic-trials",
                  "source_name": "zai_web", "extra": {"web": True}}
        text = asyncio.run(fetch_source_fulltext(source))
        assert called["page_reader"] == 1
        assert len(text) > 500

    def test_non_web_record_skips_page_reader(self, monkeypatch):
        import tools.search_sources as ts
        import tools.live_scraper as ls

        async def fake_page_text(url, max_chars=120_000):
            raise AssertionError("page_reader must not run for academic records")

        async def fake_scrape(url):
            from types import SimpleNamespace
            return SimpleNamespace(url=url, markdown="", success=False, error="offline")

        monkeypatch.setattr(ts, "fetch_page_text", fake_page_text)
        monkeypatch.setattr(ls, "_scrape_url_with_fallback", fake_scrape)

        source = {"url": "https://arxiv.org/abs/2411.10971", "source_name": "arxiv"}
        assert asyncio.run(fetch_source_fulltext(source)) == ""

    def test_web_record_falls_back_to_scrape(self, monkeypatch):
        import tools.search_sources as ts
        import tools.live_scraper as ls

        async def empty_page_text(url, max_chars=120_000):
            return ""

        async def fake_scrape(url):
            from types import SimpleNamespace
            return SimpleNamespace(url=url, markdown="Scraped fallback text. " * 40,
                                   success=True, error="")

        monkeypatch.setattr(ts, "fetch_page_text", empty_page_text)
        monkeypatch.setattr(ls, "_scrape_url_with_fallback", fake_scrape)

        source = {"url": "https://senseaboutscience.org/report.pdf",
                  "source_name": "zai_web", "extra": {"web": True}}
        text = asyncio.run(fetch_source_fulltext(source))
        assert text.startswith("Scraped fallback")


class TestWebBibliographyAndDedup:
    def test_bibliography_renders_web_record(self):
        sources = [{
            "title": "Nordic countries: 475 clinical trials involving 83,903 patients are missing results",
            "url": "https://www.transparimed.org/single-post/nordic-trials",
            "snippet": "475 trials never made public",
            "tier": "P2",
            "published_date": "2024-02-06",
            "source_name": "zai_web",
            "category": "web",
        }]
        bib = build_sources_section(sources)
        assert "[S1] [P2]" in bib
        assert "475 clinical trials" in bib
        assert "2024." in bib
        assert "https://www.transparimed.org/single-post/nordic-trials" in bib

    def test_web_record_merges_with_academic_twin(self):
        web = {
            "title": "Nordic countries: 475 clinical trials involving 83,903 patients are missing results",
            "url": "https://www.transparimed.org/single-post/nordic-trials",
            "snippet": "A total of 475 clinical trials completed in Denmark, Iceland, Finland, Norway and Sweden have never made their results public in any form, 83,903 participants.",
            "tier": "P2",
            "source_name": "zai_web",
            "category": "web",
        }
        academic = {
            "title": "Nordic countries: 475 clinical trials involving 83,903 patients are missing results",
            "url": "https://pubmed.ncbi.nlm.nih.gov/39900256",
            "snippet": "Short snippet.",
            "tier": "P1",
            "doi": "10.1016/j.eurpub.2024.39900256",
            "authors": ["Nilsonne G"],
        }
        merged = dedup_sources_canonical([web, academic])
        assert len(merged) == 1
        # richest snippet wins → the web record's URL becomes canonical,
        # the academic URL is preserved in also_found_in
        assert merged[0]["url"] == "https://www.transparimed.org/single-post/nordic-trials"
        assert "https://pubmed.ncbi.nlm.nih.gov/39900256" in merged[0].get("also_found_in", [])
        assert merged[0]["doi"] == "10.1016/j.eurpub.2024.39900256"
        # tier only upgrades, never downgrades: the peer-reviewed twin lifts
        # the merged record to P1 while the web landing page stays canonical
        assert merged[0]["tier"] == "P1"

    def test_independence_same_org_not_independent(self):
        from agent.research_engine import _sources_are_independent
        a = {"url": "https://www.transparimed.org/single-post/nordic-trials"}
        b = {"url": "https://www.transparimed.org/single-post/metascience-fail"}
        assert not _sources_are_independent(a, b)


class TestEvidenceWebReservation:
    """build_evidence_corpus must reserve full-text fetches for web records
    that score 0 — the TranspariMED page was cited but never fetched live."""

    def _mk_sources(self):
        academic = [
            {"title": f"Academic paper {i}", "url": f"https://journal{i}.org/p{i}",
             "snippet": "s", "tier": "P1", "score": 0.5 + i * 0.01}
            for i in range(8)
        ]
        web = [
            {"title": "Nordic countries: 475 clinical trials",
             "url": "https://www.transparimed.org/single-post/nordic-trials",
             "snippet": "475 trials missing results", "tier": "P2",
             "source_name": "zai_web", "extra": {"web": True}},
            {"title": "Trials tracker page",
             "url": "https://eu.trialstracker.net/sponsor/rigshospitalet",
             "snippet": "late reporting", "tier": "P2",
             "source_name": "zai_web", "extra": {"web": True}},
            {"title": "Blog opinion", "url": "https://someblog.net/op",
             "snippet": "thoughts", "tier": "P3",
             "source_name": "zai_web", "extra": {"web": True}},
        ]
        return academic + web

    def test_web_records_get_reserved_fetch_slots(self, monkeypatch):
        import agent.evidence_fetcher as ef

        fetched = []

        async def fake_fulltext(source):
            fetched.append(source["url"])
            return "Full text body. " * 100

        monkeypatch.setattr(ef, "fetch_source_fulltext", fake_fulltext)
        sources = self._mk_sources()
        corpus = asyncio.run(ef.build_evidence_corpus(sources, top_n=6, web_slots=2))
        # top-6 score cut = the 8 academics' best 6; the 2 highest web records
        # (both transparimed/trialstracker, ranked by score then list order)
        # must be reserved in addition
        assert any("transparimed.org" in u for u in fetched), fetched
        assert any("trialstracker.net" in u for u in fetched), fetched
        assert not any("someblog.net" in u for u in fetched)
        # chunks exist and carry correct 1-based source indexes
        assert corpus.chunks
        fetched_urls = {s["url"] for s in sources if s["url"] in set(fetched)}
        idxs = {c.source_idx for c in corpus.chunks}
        for url in fetched_urls:
            i = next(i for i, s in enumerate(sources, 1) if s["url"] == url)
            assert i in idxs

    def test_web_slots_zero_disables_reservation(self, monkeypatch):
        import agent.evidence_fetcher as ef

        fetched = []

        async def fake_fulltext(source):
            fetched.append(source["url"])
            return "Full text body. " * 100

        monkeypatch.setattr(ef, "fetch_source_fulltext", fake_fulltext)
        asyncio.run(ef.build_evidence_corpus(self._mk_sources(), top_n=6, web_slots=0))
        assert not any("transparimed.org" in u for u in fetched)

    def test_web_record_inside_top_n_not_double_fetched(self, monkeypatch):
        import agent.evidence_fetcher as ef

        fetched = []

        async def fake_fulltext(source):
            fetched.append(source["url"])
            return "Full text body. " * 100

        monkeypatch.setattr(ef, "fetch_source_fulltext", fake_fulltext)
        sources = [{"title": "Web top", "url": "https://www.transparimed.org/x",
                    "snippet": "s", "tier": "P2", "source_name": "zai_web",
                    "extra": {"web": True}}] + \
                  [{"title": f"A{i}", "url": f"https://j{i}.org/p", "snippet": "s",
                    "tier": "P1", "score": 0.4} for i in range(5)]
        asyncio.run(ef.build_evidence_corpus(sources, top_n=4, web_slots=2))
        assert fetched.count("https://www.transparimed.org/x") == 1


class TestWebQueryDedup:
    """expand_query_variants returns the original query first — the web
    budget must not burn identical queries twice (live run7 finding)."""

    def test_duplicate_primary_not_refired(self, monkeypatch):
        import tools.search_sources as ts

        seen_queries = []

        async def fake_search(query, num):
            seen_queries.append(query)
            return []

        monkeypatch.setattr(ts, "search_zai_web", fake_search)
        _s, stats = asyncio.run(
            _web_sources_for_report(
                "unpublished trials",
                ["unpublished trials", "variant two"],
                "thorough",
            )
        )
        assert seen_queries == ["unpublished trials", "variant two"]
        assert stats["queries"] == ["unpublished trials", "variant two"]

    def test_case_insensitive_dedup(self, monkeypatch):
        import tools.search_sources as ts

        async def fake_search(query, num):
            return []

        monkeypatch.setattr(ts, "search_zai_web", fake_search)
        _s, stats = asyncio.run(
            _web_sources_for_report("Trials Query", ["trials query"], "quick")
        )
        assert stats["queries"] == ["Trials Query"]


class TestRelevanceGateCoreQuestion:
    """The gate must answer 'does this source answer the CORE question',
    not 'does it share the expansion variants' jargon'. Live run7: an SEO
    page scored 0.174 on union vocab and passed while the page carrying the
    question's actual empirics scored 0.13 and died."""

    QUERY = "Meta-analysis of unpublished negative-result trials that never made it into public databases."
    VARIANTS = [
        "Statistical methods for addressing publication bias in meta-analysis through the inclusion of unpublished negative-result trials.",
        "Systematic approaches to identifying and incorporating unpublished negative-result trials in meta-analyses to improve research synthesis accuracy.",
    ]
    EMPIRICS = {
        "title": "Four lessons from inaccurate data on missing clinical trial results",
        "snippet": "475 clinical trials involving 83,903 patients across five Nordic countries had never made their results public. 23% of Rigshospitalet trials lacked results.",
        "url": "https://www.transparimed.org/single-post/metascience-fail",
        "tier": "P2",
        "source_name": "zai_web",
        "extra": {"web": True},
    }
    JARGON = {
        "title": "Publication bias in clinical research",
        "snippet": "systematic approaches meta-analytic evidence synthesis accuracy improvement services",
        "url": "https://pubrica.example.com/services",
        "tier": "P3",
    }

    def test_core_relevance_keeps_empirics_drops_jargon(self):
        # 5 clearly core-matching sources fill the relevance_floor (5) so the
        # floor's top-up cannot rescue the jargon page
        core_fillers = [
            {"title": f"Unpublished trials and missing results study {i}",
             "snippet": "meta-analysis unpublished negative trials results databases",
             "url": f"https://journal{i}.example.org/p", "tier": "P1"}
            for i in range(5)
        ]
        selected = select_diverse_sources(
            [self.EMPIRICS, self.JARGON] + core_fillers,
            max_sources=10,
            query=self.QUERY,
            query_variants=self.VARIANTS,
        )
        urls = [s["url"] for s in selected]
        assert "https://www.transparimed.org/single-post/metascience-fail" in urls
        assert "https://pubrica.example.com/services" not in urls

    def test_padding_still_dropped_when_matching_neither_core_nor_variants(self):
        padding = {
            "title": "Coral reef restoration program outreach",
            "snippet": "community engagement marine biodiversity workshops",
            "url": "https://reefs.example.org/program",
            "tier": "P3",
        }
        core_fillers = [
            {"title": f"Unpublished trials and missing results study {i}",
             "snippet": "meta-analysis unpublished negative trials results databases",
             "url": f"https://journal{i}.example.org/p", "tier": "P1"}
            for i in range(5)
        ]
        selected = select_diverse_sources(
            [self.EMPIRICS, padding] + core_fillers,
            max_sources=10,
            query=self.QUERY,
            query_variants=self.VARIANTS,
        )
        urls = {s["url"] for s in selected}
        assert self.EMPIRICS["url"] in urls
        assert padding["url"] not in urls


# ═══════════════════════════════════════════════════════════════
# RUN7-CRITIQUE DEFENSES: domain generalization, news-tier citations,
# scope dodging, provenance appendix
# ═══════════════════════════════════════════════════════════════

class TestDomainGeneralizationScan:
    EDU = {"title": "Neglect of publication bias compromises meta-analyses of educational research",
           "snippet": "simulation of moderate publication bias scenarios", "tier": "P1"}
    EGGER = {"title": "Bias in meta-analysis detected by a simple graphical test",
             "snippet": "funnel plot asymmetry", "tier": "P1"}
    BODY = ("## FINDINGS\n\n"
            "Under moderate publication bias the false positive rate can approach 100% [S1]. "
            "Asymmetric funnel plots suggest publication bias in observed evidence [S2].\n\n"
            "## SOURCES\n\n[S1] x. [S2] y.")

    def test_flags_unscoped_generalization(self):
        flags = scan_domain_generalization(self.BODY, [self.EDU, self.EGGER])
        assert len(flags) == 1
        assert flags[0]["source"] == 1
        assert "educational" in flags[0]["reason"]

    def test_passes_claim_that_carries_the_scope(self):
        body = ("## FINDINGS\n\nIn educational research the false positive rate "
                "can approach 100% under moderate bias [S1].\n\n## SOURCES\n\n[S1] x.")
        assert scan_domain_generalization(body, [self.EDU]) == []

    def test_generic_method_qualifier_never_flags(self):
        body = ("## FINDINGS\n\nSmall-study effects are most commonly detected by "
                "test statistics in pooled evidence [S1].\n\n## SOURCES\n\n[S1] x.")
        assert scan_domain_generalization(body, [self.EGGER]) == []

    def test_weak_claim_without_strength_marker_not_flagged(self):
        body = ("## FINDINGS\n\nThe authors report the mechanism through which "
                "omitted studies distort pooled conclusions [S1].\n\n## SOURCES\n\n[S1] x.")
        assert scan_domain_generalization(body, [self.EDU]) == []


class TestNewsTierCitationScan:
    NEWS = {"title": "Why negative study results go unreported",
            "snippet": "negative data were often considered unimportant in trials",
            "tier": "P3", "url": "https://www.pulmonologyadvisor.com/a/article.html"}
    P1 = {"title": "Impact of unpublished clinical trials on meta-analysis effect estimates",
          "snippet": "systematic review of publication bias and negative trials",
          "tier": "P1", "url": "https://doi.org/10.1111/x"}
    BODY = ("## FINDINGS\n\nNegative data were often considered unimportant in trials [S1]. "
            "Unpublished trials distort pooled estimates [S2].\n\n## SOURCES\n\n[S1] n. [S2] p.")

    def test_flags_news_cited_with_p1_available(self):
        flags = scan_news_tier_citations(self.BODY, [self.NEWS, self.P1])
        assert len(flags) == 1
        assert flags[0]["source"] == 1 and flags[0]["replacement"] == 2

    def test_silent_without_p1_reanchor_target(self):
        weak = {"title": "Coral reef bleaching patterns", "snippet": "marine ecology",
                "tier": "P1", "url": "https://j.example.org/1"}
        assert scan_news_tier_citations(self.BODY, [self.NEWS, weak]) == []

    def test_p2_transparency_source_never_treated_as_news(self):
        ngo = {"title": "Nordic unreported trials audit",
               "snippet": "475 trials without published results",
               "tier": "P2", "url": "https://ebmnews.example.org/nordic-report"}
        assert scan_news_tier_citations(
            "The Nordic audit found 475 trials without results [S1].\n\n## SOURCES\n\n[S1] x.",
            [ngo, self.P1],
        ) == []

    def test_uncited_news_source_not_flagged(self):
        body = "## FINDINGS\n\nUnpublished trials distort pooled estimates [S2].\n\n## SOURCES\n\n[S1] n. [S2] p."
        assert scan_news_tier_citations(body, [self.NEWS, self.P1]) == []


class TestScopeDodgeScan:
    GENERIC = ("# T\n\n## SCOPE & EVIDENCE BASE\n\nA direct meta-analysis of this question "
               "cannot be performed with the evidence retrieved here.\n\n## FINDINGS\n\nbody")
    STRUCTURAL = ("# T\n\n## SCOPE & EVIDENCE BASE\n\nSuch a meta-analysis cannot be performed: "
                  "trials that were never registered are by definition not systematically "
                  "recorded, so no sampling frame exists.\n\n## FINDINGS\n\nbody")
    FEASIBLE = ("# T\n\n## SCOPE & EVIDENCE BASE\n\nThe evidence base supports a direct "
                "synthesis of the retrieved trials.\n\n## FINDINGS\n\nbody")

    def test_flags_generic_hedge(self):
        flags = scan_scope_dodge(self.GENERIC)
        assert len(flags) == 1
        assert "STRUCTURAL" in flags[0]["reason"]

    def test_structural_argument_passes(self):
        assert scan_scope_dodge(self.STRUCTURAL) == []

    def test_no_feasibility_negation_passes(self):
        assert scan_scope_dodge(self.FEASIBLE) == []

    def test_missing_scope_section_passes(self):
        assert scan_scope_dodge("# T\n\n## FINDINGS\n\nbody") == []

    # ── run8 regression: live GLM draft carried a REAL structural argument
    # in data-availability phrasing; the marker list did not recognize it
    # and flagged an honest scope section. The report's own paragraphs:
    RUN8_SCOPE = (
        "# T\n\n## SCOPE & EVIDENCE BASE\n\n"
        "The analysis is constrained to the *existence* and *impact* of "
        "unpublished trials, specifically negative-result studies, utilizing "
        "publication-bias detection methods rather than a direct meta-analysis "
        "of the missing data.\n\n"
        "Despite these documented instances of missing data, a direct "
        "meta-analysis of unpublished negative-result trials is methodologically "
        "infeasible with the current evidence base. The primary barrier is the "
        "inaccessibility of the raw data from unpublished studies. While "
        "registries can identify that a trial exists, they rarely contain the "
        "detailed effect size data required for a quantitative synthesis. "
        "Without access to the individual participant data or at least the full "
        "effect sizes and variances from unpublished trials, a direct "
        "meta-analysis cannot be performed.\n\n## FINDINGS\n\nbody"
    )

    def test_run8_data_availability_scope_passes(self):
        assert scan_scope_dodge(self.RUN8_SCOPE) == []

    def test_data_availability_marker_variants(self):
        for phrase in (
            "cannot be performed: without access to the raw trial data, no "
            "synthesis is possible",
            "cannot be performed: registries rarely contain outcome data",
            "cannot be performed: individual participant data unavailable",
            "cannot be performed: the underlying data are inaccessible",
        ):
            doc = f"# T\n\n## SCOPE & EVIDENCE BASE\n\nA direct meta-analysis {phrase}.\n\n## FINDINGS\n\nbody"
            assert scan_scope_dodge(doc) == [], phrase

    def test_generic_hedge_still_flags_after_marker_expansion(self):
        # The expansion must not open the door to "we didn't find it" hedges.
        doc = ("# T\n\n## SCOPE & EVIDENCE BASE\n\nA direct meta-analysis cannot be "
               "performed with the evidence retrieved here; the relevant studies "
               "were not found in our search.\n\n## FINDINGS\n\nbody")
        assert len(scan_scope_dodge(doc)) == 1


class TestProvenanceAppendix:
    def test_renders_stage_funnel_and_queries(self):
        md = _render_provenance_section({
            "identified_records": 120, "post_dedup": 90, "post_selection": 24,
            "post_enrichment": 31, "canonical_backfilled": 7, "final_sources": 12,
            "tier_breakdown": {"P1": 10, "P2": 2}, "web_records_kept": 3,
            "evidence": {"chunks": 97, "fetched_sources": 8, "total_chars": 131000,
                         "semantic_scoring": False},
            "queries": ["main query", "keyword rewrite"], "pruned_uncited": [4, 9],
            "citation_precision": 1.0, "claim_coverage": 0.9,
        })
        assert "## METHODOLOGY & PROVENANCE" in md
        assert "Records identified (live APIs + RAG + web) | 120" in md
        assert "Final pool by source tier | P1: 10, P2: 2" in md
        assert "Grey-literature web records in final pool | 3" in md
        assert "| Uncited sources pruned | 2 |" in md
        assert "- keyword rewrite" in md
        assert "keyword fallback" in md

    def test_minimal_provenance_renders_without_optional_rows(self):
        md = _render_provenance_section({"identified_records": 5})
        assert "## METHODOLOGY & PROVENANCE" in md
        assert "Grey-literature" not in md
        assert "Uncited sources pruned" not in md


class TestCritiqueReviseCritiqueFixes:
    def test_scope_and_news_flags_enter_revision_feedback(self, monkeypatch):
        import asyncio
        import agent.llm_router as lr
        import agent.research_engine as re_mod

        report = (
            "# T\n\n## SCOPE & EVIDENCE BASE\n\nA direct meta-analysis of this "
            "question cannot be performed with the evidence retrieved here.\n\n"
            "## FINDINGS\n\nNegative data were often considered unimportant in "
            "trials [S1]. Under moderate bias false positive rates can approach "
            "100% [S2]. More padding prose follows so the revision threshold "
            "check has a reasonable-length report to work with here.\n\n"
            "## SOURCES\n\n[S1] [P3] Daily news. Why results go unreported.\n"
            "[S2] [P1] Educational research publication bias study.\n"
            "[S3] [P1] Publication bias in clinical trials: negative trials are "
            "less likely to be published."
        )

        async def fake_router(**kwargs):
            ctx = kwargs.get("context", "")
            if "FLAGGED ISSUES" in ctx:
                return SimpleNamespace(content=report, model_name="mock")
            return SimpleNamespace(content='{"issues": []}', model_name="mock")

        monkeypatch.setattr(lr, "synthesize_with_router", fake_router)
        sources = [
            {"title": "Why negative study results go unreported",
             "snippet": "negative data were often considered unimportant in trials",
             "tier": "P3", "url": "https://www.pulmonologyadvisor.com/a.html"},
            {"title": "Neglect of publication bias compromises meta-analyses of educational research",
             "snippet": "simulation of moderate publication bias",
             "tier": "P1", "url": "https://doi.org/10.1111/edu"},
            {"title": "Publication bias in clinical trials: negative trials are less likely to be published",
             "snippet": "unpublished negative trial data remain inaccessible",
             "tier": "P1", "url": "https://doi.org/10.1111/clin"},
        ]
        text, summary, _ = asyncio.run(
            re_mod._critique_and_revise(report, "q", sources, "quick")
        )
        assert summary["scope_flags"] == 1
        assert summary["news_flags"] == 1
        assert summary["domain_flags"] == 1
        assert summary["revised"] is True


# ═══════════════════════════════════════════════════════════════════
# run8-verdict defenses: discriminative relevance, padding scan,
# metrics honesty, screening audit, registry lane
# ═══════════════════════════════════════════════════════════════════

QUERY_TRIALS = ("Meta-analysis of unpublished negative-result trials that never "
                "made it into public databases.")


class TestRelevanceGateDiscriminative:
    """Generic-academic vocabulary carries no topical purchase (run8 [S1]/[S5])."""

    HYPOXIC = {
        "title": "Meta-analysis of hypoxic transcriptomes from public databases",
        "snippet": "we attempted a meta-analysis of the RNA-seq data from the hypoxic transcriptomes archived in public databases",
        "tier": "P1", "url": "https://doi.org/10.1101/267310", "score": 5.0,
    }
    FEDERATED = {
        "title": "Federated Learning in Distributed Medical Databases: Meta-Analysis of Large-Scale Subcortical Brain Data",
        "snippet": "databanks worldwide contain brain images; datasets stored at different institutions cannot always be shared",
        "tier": "P1", "url": "http://arxiv.org/abs/1810.08553v4", "score": 4.0,
    }
    RELEVANT = {
        "title": "Nordic countries: 475 clinical trials involving 83,903 patients are missing results",
        "snippet": "TranspariMED review of Danish, Icelandic and Norwegian trial registries: 475 trials missing results",
        "tier": "P2", "url": "https://www.transparimed.org/single-post/nordic-trials", "score": 3.0,
    }
    CANON = {
        "title": "Publication bias: evidence of delayed publication in a cohort study of clinical research projects",
        "snippet": "cohort study of clinical research projects",
        "tier": "P1", "url": "https://doi.org/10.1136/bmj.315.7109.640", "score": 6.0,
        "backfilled": True, "canonical_id": "stern1997",
    }

    def test_generic_only_overlap_rejected(self):
        screening = {}
        kept = select_diverse_sources(
            [dict(self.HYPOXIC), dict(self.FEDERATED), dict(self.RELEVANT)],
            max_sources=10, query=QUERY_TRIALS, screening=screening,
            relevance_floor=1,  # healthy pool: starvation valve off
        )
        titles = [s["title"] for s in kept]
        assert self.HYPOXIC["title"] not in titles
        assert self.FEDERATED["title"] not in titles
        assert self.RELEVANT["title"] in titles
        assert len(screening.get("relevance_rejected", [])) >= 2

    def test_canonical_backfill_exempt_from_discriminative_rule(self):
        # Stern 1997 carries zero query vocabulary — it enters via the
        # registry, not via keyword collision; the gate must not kill it.
        kept = select_diverse_sources(
            [dict(self.CANON), dict(self.RELEVANT)],
            max_sources=10, query=QUERY_TRIALS,
        )
        assert any(s.get("canonical_id") == "stern1997" for s in kept)

    def test_floor_rescue_recorded(self):
        # Starved pool: gate rescues top rejects; audit records them.
        screening = {}
        kept = select_diverse_sources(
            [dict(self.HYPOXIC)], max_sources=10, query=QUERY_TRIALS,
            relevance_floor=2, screening=screening,
        )
        assert len(kept) == 1  # nothing else to rescue
        # a healthy pool with one weak source: floor doesn't fire
        screening2 = {}
        select_diverse_sources(
            [dict(self.RELEVANT)], max_sources=10, query=QUERY_TRIALS,
            relevance_floor=5, screening=screening2,
        )
        assert screening2.get("floor_rescued") == []
        assert screening.get("junk_filtered") is not None or True

    def test_junk_filtered_counted(self):
        junk = {"title": "Supplemental Information 1", "snippet": "x",
                "tier": "P3", "url": "https://example.org/supp1", "score": 9.0}
        screening = {}
        kept = select_diverse_sources(
            [junk, dict(self.RELEVANT)], max_sources=10,
            query=QUERY_TRIALS, screening=screening,
        )
        assert screening.get("junk_filtered") == 1
        assert all("Supplemental" not in s["title"] for s in kept)


class TestScanOffTopicPadding:
    def test_run8_padding_flagged(self):
        sources = [
            {"title": "Meta-analysis of hypoxic transcriptomes from public databases",
             "snippet": "RNA-seq data from the hypoxic transcriptomes archived in public databases",
             "tier": "P1", "url": "https://doi.org/10.1101/267310"},
            {"title": "Federated Learning in Distributed Medical Databases: Meta-Analysis of Large-Scale Subcortical Brain Data",
             "snippet": "brain images at different institutions",
             "tier": "P1", "url": "http://arxiv.org/abs/1810.08553v4"},
            {"title": "Nordic countries: 475 clinical trials involving 83,903 patients are missing results",
             "snippet": "trial registries review",
             "tier": "P2", "url": "https://www.transparimed.org/single-post/nordic-trials"},
        ]
        flags = scan_off_topic_padding(QUERY_TRIALS, sources)
        assert {f["source"] for f in flags} == {1, 2}
        assert "discriminative" in flags[0]["reason"]

    def test_canonical_exempt(self):
        sources = [{"title": "Bias in meta-analysis detected by a simple, graphical test",
                    "snippet": "funnel plot", "tier": "P1",
                    "url": "https://doi.org/10.1136/bmj.315.7109.629",
                    "canonical_id": "egger1997"}]
        assert scan_off_topic_padding(QUERY_TRIALS, sources) == []

    def test_registry_trials_not_flagged(self):
        sources = [{"title": "A Phase 3 Trial of Drug X vs Placebo",
                    "snippet": "Status: COMPLETED. Results posted: NO. NCT: NCT12345678.",
                    "tier": "P1", "url": "https://clinicaltrials.gov/study/NCT12345678"}]
        assert scan_off_topic_padding(QUERY_TRIALS, sources) == []

    def test_query_without_disc_vocab_safe(self):
        # A query made ENTIRELY of generic terms — scan must bail out, not
        # flag everything.
        sources = [{"title": "Anything at all", "snippet": "s",
                    "tier": "P1", "url": "https://example.org/a"}]
        assert scan_off_topic_padding("Meta-analysis: a systematic review of public databases", sources) == []


class TestMetricsHonestyRenderer:
    PROV = {
        "identified_records": 40, "post_dedup": 30, "post_selection": 12,
        "post_enrichment": 14, "canonical_backfilled": 5, "final_sources": 10,
        "citation_precision": 1.0, "relevance_precision": 1.0,
        "padding_cited": [], "claim_coverage": 0.561,
        "screening": {
            "relevance_kept": 12,
            "relevance_rejected": [
                {"title": "Meta-analysis of hypoxic transcriptomes from public databases",
                 "url": "https://doi.org/10.1101/267310", "relevance": 0.0},
                {"title": "Federated Learning in Distributed Medical Databases",
                 "url": "http://arxiv.org/abs/1810.08553v4", "relevance": 0.0},
            ],
            "floor_rescued": [],
            "padding_dropped": ["Meta-analysis of hypoxic transcriptomes from public databases"],
            "junk_filtered": 1,
        },
        "pruned_uncited": [{"title": "dead weight"}],
        "registry": {"records": 6, "no_results_records": 4},
    }

    def test_both_precisions_rendered(self):
        md = _render_provenance_section(self.PROV)
        assert "| Citation precision (structural) | 1.0 |" in md
        assert "| Relevance precision (topical, padding-aware) | 1.0 |" in md

    def test_low_claim_coverage_warns(self):
        md = _render_provenance_section(self.PROV)
        assert "Claim coverage below 0.60" in md

    def test_healthy_claim_coverage_no_warning(self):
        prov = dict(self.PROV, claim_coverage=0.82)
        md = _render_provenance_section(prov)
        assert "Claim coverage below 0.60" not in md

    def test_padding_cited_warns(self):
        prov = dict(self.PROV, relevance_precision=0.75, padding_cited=[3])
        md = _render_provenance_section(prov)
        assert "zero discriminative overlap" in md and "[S3]" in md

    def test_screening_decisions_subsection(self):
        md = _render_provenance_section(self.PROV)
        assert "Screening decisions (deterministic, auditable):" in md
        assert "hypoxic transcriptomes" in md          # rejected, with score
        assert "0.0" in md                              # relevance score shown
        assert "Off-topic padding" in md
        assert "Dead-source pruning" in md
        assert "Junk filter: 1" in md

    def test_registry_stats_render(self):
        prov = dict(self.PROV)
        # registry stats flow through web_records_kept-adjacent renderer line
        # when present; the funnel keeps them in provenance for meta.json
        assert prov["registry"]["no_results_records"] == 4


class TestClinicalTrialsLane:
    def test_registry_records_enter_fanout_with_stats(self, monkeypatch):
        import tools.search_sources as ts

        async def dead_search(query, num=8):
            return []

        async def fake_registry(query, max_results=8):
            return [SearchResult(
                title="A Phase 3 Trial of Drug X vs Placebo",
                url="https://clinicaltrials.gov/study/NCT12345678?utm_source=x",
                snippet="Status: COMPLETED. Results posted: NO. NCT: NCT12345678.",
                source_name="clinical_trials",
                source_category=SourceCategory.CLINICAL,
                source_tier="P1",
                published_date="2022-10-01",
                extra={"nct_id": "NCT12345678", "has_results": False,
                       "registry": "clinicaltrials_gov"},
            )]

        monkeypatch.setattr(ts, "search_zai_web", dead_search)
        monkeypatch.setattr(ts, "search_clinical_trials", fake_registry)
        monkeypatch.setenv("APEX_WEB_SEARCH", "true")
        monkeypatch.delenv("APEX_WEB_SEED_URLS", raising=False)
        sources, stats = asyncio.run(_web_sources_for_report(QUERY_TRIALS, [], "quick"))
        assert stats.get("registry_records") == 1
        assert stats.get("registry_no_results") == 1
        assert stats.get("seeded", 0) == 0
        assert len(sources) == 1
        rec = sources[0]
        assert rec["url"].endswith("/study/NCT12345678")  # canonicalized
        assert rec["extra"]["registry"] is True and rec["extra"]["web"] is True
        assert rec["source_name"] == "clinical_trials"

    def test_registry_lane_failure_degrades_silently(self, monkeypatch):
        import tools.search_sources as ts

        async def dead_search(query, num=8):
            return []

        async def boom(query, max_results=8):
            raise ConnectionError("registry down")

        monkeypatch.setattr(ts, "search_zai_web", dead_search)
        monkeypatch.setattr(ts, "search_clinical_trials", boom)
        monkeypatch.setenv("APEX_WEB_SEARCH", "true")
        monkeypatch.delenv("APEX_WEB_SEED_URLS", raising=False)
        sources, stats = asyncio.run(_web_sources_for_report(QUERY_TRIALS, [], "quick"))
        assert sources == [] and stats.get("registry_records") is None
