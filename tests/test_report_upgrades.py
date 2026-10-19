"""
Unit tests for APEX 3.0 report-quality upgrades.

Covers the pure (no-I/O, no-LLM) logic added across the phases:
- Phase 1: task-aware tier selection (report routing)
- Phase 4: canonical dedup, citation integrity, NLI aggregation
- Phase 5: diverse source selection, domain-aware decay + cache TTL

Run: pytest tests/test_report_upgrades.py -v
"""

import asyncio
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
)
from agent.citation_integrity import (
    extract_numbered_citations,
    check_report_citations,
    build_sources_section,
    CitationReport,
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
