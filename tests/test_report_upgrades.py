"""
Unit tests for APEX 3.0 report-quality upgrades.

Covers the pure (no-I/O, no-LLM) logic added across the phases:
- Phase 1: task-aware tier selection (report routing)
- Phase 4: canonical dedup, citation integrity, NLI aggregation
- Phase 5: diverse source selection, domain-aware decay + cache TTL

Run: pytest tests/test_report_upgrades.py -v
"""

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
