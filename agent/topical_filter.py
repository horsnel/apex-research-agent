"""
Topical relevance filter — EMBEDDING-based, meaning-level (run9 verdict fix).

run9's fatal defect: a chronic-pain ACT-vs-CBT trial (NCT02778737) became an
entire section of a publication-bias report. Root causes:

1. The ClinicalTrials.gov provider injects "Results posted: NO" into every
   record's snippet — handing EVERY registry record the discriminative token
   "results" for free, so the lexical gate passed any trial on any topic.
2. Lexical gates score token overlap, not topical fit. No amount of
   re-running fixes a scoring function that doesn't understand the topic.

CALIBRATION (live bge-base-en-v1.5, 2026-10, scripts/probe_topical_floor*.py,
probe_anchor_prompts.py, probe_bipolar_anchor.py):

  Cosine against the RAW question does NOT separate gold from junk: the
  question embeds as a design+topic mix ("meta-analysis ... trials ...
  databases") and junk rides that baseline — chronic-pain trial 0.636 vs
  gold Chan 2004 at 0.710, while junk "hypoxic transcriptomes" (0.716)
  scores ABOVE gold. max(query, anchor) aggregation inherits the same leak.

  A short LLM anchor ("Unpublished negative trial results") compresses the
  distribution until gold and junk overlap (+0.012 margins, unstable).

  The separating scheme (probe_bipolar_anchor.py — CLEAN PASS, 8/8 gold
  kept, 5/5 junk rejected) is BIPOLAR:
    PHENOMENON anchor — the evidence problem phrased in the field's OWN
    terminology ("Publication bias in clinical trials, specifically the
    underrepresentation of negative results in the literature"). Gold
    literature self-describes with these terms; junk does not.
    DISTRACTOR anchor — sources that share the subject vocabulary but are
    not evidence about the phenomenon ("studies on trial registries and
    data sharing practices..."). Junk is CLOSER to the distractor than to
    the phenomenon; gold is not.

DESIGN, therefore:
  1. ANCHORS: one cheap LLM call rewrites the query into
     PHENOMENON + DISTRACTOR + KIND (one call per run; pass 2 reuses it).
     Fallback on failure: phenomenon=query, distractor=none, kind
     "query_fallback" (weaker; recorded honestly).
  2. SCORE: cos(phenomenon, source); the raw query vector is NEVER a
     scorer, so junk cannot ride it.
  3. GATES: keep iff score_p >= max(ABS_FLOOR, RATIO * best_score_in_batch)
     AND (no distractor OR score_p > score_d). The ratio term is
     distribution-free (junk max = 0.77-0.81 x top across anchor classes;
     gold min = 0.83-0.85 x top; RATIO=0.82 sits in the gap). The contrast
     gate is insurance against weak-anchor compression.
  4. HONEST DEGRADATION: if embeddings OR the anchor call fail, nothing is
     dropped and topical_mode="unavailable" — the metrics layer must then
     REFUSE to print a topical precision number (a lexical-only "1.0" is
     exactly the lie run9 printed next to its chronic-pain section).
  5. STARVATION VALVE: if every source is rejected, the top-3 by score are
     re-admitted flagged borderline — and they count against relevance
     precision and render an explicit provenance warning.

Every kept source carries a one-line justification (auditable screening);
every rejection carries its score.
"""

import asyncio
import logging
import math
import os
import re
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Calibrated 2026-10 (see module docstring + scripts/probe_topical_floor*.py)
ABS_FLOOR_DEFAULT = 0.45
RATIO_DEFAULT = 0.82
_STARVE_KEEP = 3
_TEXT_CAP = 1500

# Fixed topic-neutral distractor for "phenomenon"-kind questions: junk that
# shares the subject vocabulary (an ordinary trial/study reporting its own
# findings) is closer to THIS than to the phenomenon anchor. Live-calibrated
# on the run9 poison: the chronic-pain trial scores p=0.662 (above the
# 0.660 floor — the floor alone keeps it) but d_generic=0.668 > p — the
# contrast gate rejects it. Methodological gold sources stay further from
# the distractor than from the phenomenon (probe_generic_distractor.py).
GENERIC_PHENOMENON_DISTRACTOR = (
    "An ordinary primary study in the subject area that applies the subject's "
    "methods and reports its own substantive findings for its participants or "
    "subjects, rather than evidence about how the research system itself behaves."
)

_ANCHOR_PROMPT = """A researcher asks: "{query}"

Classify what evidence this question needs and return EXACTLY three lines:

PHENOMENON: the underlying evidence problem in 25-45 words, using the standard terminology the field's own literature uses in titles.
DISTRACTOR: one sentence naming the most tempting NON-evidence source class for this question — the sources that share the question's vocabulary but are NOT evidence about the phenomenon. Think FIRST of ordinary primary studies that merely belong to the question's subject area and report their own findings (e.g. a trial of a treatment measuring outcomes for a condition), then other tempting classes. Write "none" only if the question is purely about a subject domain.
KIND: either "phenomenon" (question is about how research/evidence itself behaves) or "domain" (question is about a subject)."""


def abs_floor() -> float:
    try:
        return float(os.getenv("APEX_TOPICAL_ABS", "") or ABS_FLOOR_DEFAULT)
    except ValueError:
        return ABS_FLOOR_DEFAULT


def ratio() -> float:
    try:
        return float(os.getenv("APEX_TOPICAL_RATIO", "") or RATIO_DEFAULT)
    except ValueError:
        return RATIO_DEFAULT


def _sanitize_anchor(text: str) -> str:
    """Strip router citation artifacts ([P1], [Source, S2] ...) and padding."""
    t = re.sub(r"\s*\[[^\]]*\]\s*", " ", text or "")
    return re.sub(r"\s+", " ", t).strip(" \"'\u201c\u201d.")


def _cosine(a: List[float], b: List[float]) -> float:
    """Pure-python cosine — no numpy dependency for a 768-dim dot product."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return dot / math.sqrt(na * nb)


def _source_text(s: Dict) -> str:
    """Highest-signal text for topical judgment: title + snippet.

    Registry records carry BriefSummary inside the snippet (after the
    Status/Phase/Results boilerplate) — the summary is the meaning-bearing
    part and the boilerplate is only a few tokens, so title+snippet is
    sufficient without special-casing.
    """
    parts = [str(s.get("title") or ""), str(s.get("snippet") or "")]
    extra = s.get("extra") or {}
    for k in ("brief_summary", "conditions"):
        if extra.get(k):
            parts.append(str(extra[k]))
    return re.sub(r"\s+", " ", " ".join(p for p in parts if p)).strip()[:_TEXT_CAP]


def _anchor_ok(text: str) -> bool:
    t = (text or "").strip().strip('"').strip()
    if not (20 <= len(t) <= 300):
        return False
    low = t.lower()
    if "[all_llm_failed]" in low or "as an ai" in low:
        return False
    return True


async def _bipolar_anchor(query: str) -> Optional[Dict]:
    """One cheap LLM call: PHENOMENON + DISTRACTOR + KIND for the question.

    Returns {"phenomenon", "distractor" (or None), "kind"} or None on any
    failure — the caller falls back to the raw query and records
    kind="query_fallback".
    """
    try:
        from agent.llm_router import synthesize_with_router
        res = await asyncio.wait_for(
            synthesize_with_router(
                query="",
                context=_ANCHOR_PROMPT.format(query=query.strip()[:600]),
                max_tokens=300,
                task_type="classification",
            ),
            timeout=60.0,
        )
        phen = dist = kind = None
        for line in (res.content or "").splitlines():
            m = re.match(r"\s*PHENOMENON\s*:\s*(.+)", line, re.I)
            if m and phen is None:
                phen = _sanitize_anchor(m.group(1))
            m = re.match(r"\s*DISTRACTOR\s*:\s*(.+)", line, re.I)
            if m and dist is None:
                d = m.group(1).strip()
                dist = None if d.lower().strip(". ") == "none" else _sanitize_anchor(d)
            m = re.match(r"\s*KIND\s*:\s*(.+)", line, re.I)
            if m and kind is None:
                kind = m.group(1).strip().strip(". ").lower()
        if not _anchor_ok(phen or ""):
            return None
        return {
            "phenomenon": phen,
            "distractor": dist if _anchor_ok(dist or "") else None,
            "kind": kind if kind in ("phenomenon", "domain") else "phenomenon",
        }
    except Exception as e:
        logger.debug(f"Bipolar anchor generation failed: {e}")
        return None


def _justification(
    score_p: float, score_d: Optional[float], threshold: float, ok: bool, borderline: bool,
) -> str:
    contrast = f" vs distractor {score_d:.2f}" if score_d is not None else ""
    if ok and not borderline:
        return f"phenomenon cosine {score_p:.2f} >= floor {threshold:.2f}{contrast}"
    if borderline:
        return (
            f"STARVATION-RESCUED: phenomenon cosine {score_p:.2f} < floor {threshold:.2f}"
            f"{contrast} — pool would otherwise be empty; treat as unverified topical fit"
        )
    reason = f"phenomenon cosine {score_p:.2f} < floor {threshold:.2f}"
    if score_d is not None and score_p <= score_d:
        reason = (
            f"closer to the distractor ({score_d:.2f}) than to the phenomenon "
            f"({score_p:.2f}) — shares the subject vocabulary, not the evidence problem"
        )
    return reason


# ── LLM adjudication of the borderline band ──
# (constants documented at the call site)
_BAND_BELOW = 0.05      # rejected but within this of the floor → adjudicate
_BAND_ABOVE = 0.02      # kept but within this of the floor → adjudicate
_BAND_CONTRAST = 0.03   # |p - d| this close → contrast gate is a coin flip
_ADJUDICATE_MAX = 14    # cap tiny-LLM calls per pass (overflow counted in scr)

_ADJUDICATE_PROMPT = """A report is being written for this research question:
{query}

The question needs evidence ABOUT this phenomenon: {phenomenon}
Sources that are NOT evidence about it look like this: {distractor}

SOURCE UNDER REVIEW:
Title: {title}
Text: {snippet}

Is this source EVIDENCE ABOUT the phenomenon itself? Answer YES if the source provides data, counts, audits, empirical analyses, or methods whose subject is the phenomenon — whatever kind of study produced them. Answer NO if the source is merely an instance or example of the phenomenon, merely belongs to the subject area and reports its own substantive findings, or is about something else entirely.

Answer with exactly one word: YES or NO."""


async def _adjudicate(
    query: str, phenomenon: str, distractor: Optional[str], source: Dict,
) -> Optional[bool]:
    """Small LLM judgment for one borderline source. None on any failure
    (the vector verdict stands).

    Routed to the NLI chain (report-grade models), NOT the classification
    tier: live A/B showed Granite-4.0-Micro answers NO to every case
    including the canonical Nordic audit — a capability limit, not a prompt
    one. GLM-4.7-Flash (nli chain) judged all five probe cases correctly,
    including rejecting the run9 chronic-pain poison.
    """
    try:
        from agent.llm_router import synthesize_with_router
        res = await asyncio.wait_for(
            synthesize_with_router(
                query="",
                context=_ADJUDICATE_PROMPT.format(
                    query=query.strip()[:400],
                    phenomenon=phenomenon,
                    distractor=distractor or "sources that share the subject vocabulary but are not about the phenomenon",
                    title=(source.get("title") or "Untitled")[:200],
                    snippet=_source_text(source)[:900],
                ),
                max_tokens=16,
                task_type="nli",
            ),
            timeout=60.0,
        )
        answer = (res.content or "").strip().upper()
        if answer.startswith("YES"):
            return True
        if answer.startswith("NO"):
            return False
        return None
    except Exception as e:
        logger.debug(f"Topical adjudication failed (non-fatal): {e}")
        return None


async def apply_topical_filter(
    query: str,
    sources: List[Dict],
    screening: Optional[Dict] = None,
    embed_fn=None,
    anchor_fn=None,
    anchors: Optional[Dict] = None,
    adjudicate_fn=None,
    select_new_only: bool = False,
) -> List[Dict]:
    """Embedding topical gate over the candidate pool. Returns filtered list.

    Every kept source is annotated:
        extra["topical_score"]         float (phenomenon cosine)
        extra["topical_ok"]            bool (False only for borderline rescues)
        extra["topical_justification"] one-line audit string
    Rejections (with scores + justifications) go to screening["topical_rejected"].

    embed_fn:  injectable async Callable[[List[str]], List[List[float]]] (tests)
    anchor_fn: injectable async Callable[[str], Optional[Dict]] (tests);
               default = LLM bipolar rewrite (_bipolar_anchor)
    anchors:   pre-computed {"phenomenon","distractor","kind"} dict (pass 2
               reuses pass 1's anchors — one LLM call per run). When None and
               select_new_only, anchors are read from screening["topical_anchors"].

    select_new_only: second-pass mode — sources already carrying a
    topical_score (pass 1) are passed through untouched; canonical-backfilled
    sources may be exempted by the caller (registry-attested in-domain).
    """
    scr = screening if screening is not None else {}
    af = abs_floor()
    rt = ratio()

    if select_new_only:
        pending = [s for s in sources if "topical_score" not in (s.get("extra") or {})]
        if anchors is None:
            anchors = scr.get("topical_anchors")
    else:
        pending = list(sources)

    if not query or not query.strip() or not pending:
        scr.setdefault("topical_mode", "unavailable" if not query else "embedding")
        return sources

    # ── Anchors: phenomenon + distractor(s) (one cheap LLM call per run) ──
    if not anchors:
        gen = anchor_fn(query) if anchor_fn else _bipolar_anchor(query)
        anchors = await _maybe_await(gen)
    if not anchors or not _anchor_ok(anchors.get("phenomenon") or ""):
        anchors = {"phenomenon": query.strip(), "distractor": None, "kind": "query_fallback"}
        logger.info("Topical anchor: LLM rewrite unusable — falling back to raw query (weaker)")
    phen = anchors["phenomenon"]
    kind = anchors.get("kind") or "phenomenon"
    # Distractor candidates: the LLM's topic-specific one plus, for
    # phenomenon-kind questions, the fixed generic one (the ordinary-study
    # class that poisoned run9). d-score = max over candidates.
    dist_candidates: List[str] = []
    if anchors.get("distractor") and _anchor_ok(anchors["distractor"]):
        dist_candidates.append(anchors["distractor"])
    if kind == "phenomenon":
        dist_candidates.append(GENERIC_PHENOMENON_DISTRACTOR)
    dist_candidates = list(dict.fromkeys(dist_candidates))
    scr["topical_anchors"] = anchors
    scr["topical_anchor"] = phen

    # ── Embed (batched; [phenomenon, distractors...] + N sources, one call) ──
    try:
        if embed_fn is None:
            from ingest.embedder import embed_texts
            embed_fn = embed_texts
        ref_texts = [phen] + dist_candidates
        texts = ref_texts + [_source_text(s) for s in pending]
        vecs = await asyncio.wait_for(embed_fn(texts), timeout=90.0)
        if not vecs or len(vecs) < len(texts) or not vecs[0]:
            raise RuntimeError(f"embedder returned {len(vecs or [])} vectors for {len(texts)} texts")
        pvec = vecs[0]
        dvecs = [v for v in vecs[1:len(ref_texts)]] if dist_candidates else []
        src_vecs = vecs[len(ref_texts):]
        scores_p = [_cosine(pvec, v) for v in src_vecs]
        scores_d = (
            [max(_cosine(dv, v) for dv in dvecs) for v in src_vecs]
            if dvecs else [None] * len(src_vecs)
        )
    except Exception as e:
        # HONEST DEGRADATION: never drop on missing evidence, never fake a
        # score. The metrics layer reads topical_mode and refuses to print a
        # topical precision when this branch ran.
        scr["topical_mode"] = "unavailable"
        scr["topical_error"] = f"{type(e).__name__}: {e}"[:200]
        logger.warning(f"Topical filter unavailable (non-fatal, nothing dropped): {e}")
        return sources

    scr["topical_mode"] = "embedding"
    scr["topical_distractors"] = dist_candidates

    best = max(scores_p) if scores_p else 0.0
    threshold = max(af, rt * best)
    scr.setdefault("topical_floor", round(threshold, 3))

    def _passes(sp: float, sd: Optional[float]) -> bool:
        if sp < threshold:
            return False
        if sd is not None and sp <= sd:
            return False
        return True

    kept: List[Dict] = []
    rejected: List[Dict] = []
    attested_kept: List[Dict] = []
    for s, sp, sd in zip(pending, scores_p, scores_d):
        extra = s.setdefault("extra", {})
        extra["topical_score"] = round(sp, 3)
        if sd is not None:
            extra["topical_distractor_score"] = round(sd, 3)
        # ── Operator-attested anchors (PRISMA hand-search analogue) ──
        # run11 finding: the topic's canonical source (the Nordic 475/83,903
        # page) scored 0.643 vs floor 0.687 — the mid-band noise the bipolar
        # design was built for, but the adjudication cap exhausted first.
        # Operator-attested seed URLs are hand-searched anchors: they are
        # SCORED and AUDITED like everything else, but the automated gates
        # do not reject them — the attestation is the disclosure, rendered
        # in the screening audit verbatim.
        if extra.get("attested"):
            extra["topical_ok"] = True
            extra["topical_attested"] = True
            extra["topical_borderline"] = False
            extra["topical_justification"] = (
                f"operator-attested anchor (vector score {sp:.2f} vs floor "
                f"{threshold:.2f}) — automated screening bypassed, retention "
                f"disclosed by attestation"
            )
            attested_kept.append({
                "title": (s.get("title") or "")[:120],
                "score": round(sp, 3),
                "justification": extra["topical_justification"],
            })
            kept.append(s)
            continue
        ok = _passes(sp, sd)
        if ok:
            extra["topical_ok"] = True
            extra["topical_borderline"] = False
            extra["topical_justification"] = _justification(sp, sd, threshold, True, False)
            kept.append(s)
        else:
            extra["topical_ok"] = False
            extra["topical_borderline"] = False
            extra["topical_justification"] = _justification(sp, sd, threshold, False, False)
            rejected.append((s, sp, sd))
    if attested_kept:
        scr.setdefault("topical_attested", []).extend(attested_kept)

    # ── LLM adjudication of the BORDERLINE BAND ──
    # Live calibration showed single-vector cosine reorders the middle band
    # (nordic-audit vs chronic-pain trial: 0.019 apart) as the anchor wording
    # varies — no threshold on the score separates them robustly. The vector
    # gates decide the clear cases; sources in the band around the decision
    # boundary are arbitrated by a small LLM judgment (the component that
    # actually understands the topic), and every arbitration is logged.
    band: List[tuple] = []
    for s, sp, sd in rejected:
        below_floor_ok = sp >= threshold - _BAND_BELOW
        contrast_ambiguous = sd is not None and abs(sp - sd) <= _BAND_CONTRAST
        if below_floor_ok or contrast_ambiguous:
            band.append((s, sp, sd))
    for s in list(kept):
        ex = s.get("extra") or {}
        # attested anchors: retention is by attestation, NOT by the vector
        # verdict — the band must not be able to LLM-remove them
        if ex.get("topical_attested"):
            continue
        sp_k = float(ex.get("topical_score") or 0.0)
        sd_k = ex.get("topical_distractor_score")
        if sp_k <= threshold + _BAND_ABOVE:
            band.append((s, sp_k, sd_k))
    band.sort(key=lambda t: abs(t[1] - threshold))
    if len(band) > _ADJUDICATE_MAX:
        scr["topical_band_overflow"] = len(band) - _ADJUDICATE_MAX
    band = band[:_ADJUDICATE_MAX]

    adjudicated: List[Dict] = []
    if band and os.getenv("APEX_TOPICAL_ADJUDICATE", "true").lower() != "false":
        for s, sp, sd in band:
            adj_call = adjudicate_fn or _adjudicate
            verdict = await _maybe_await(adj_call(query, phen, anchors.get("distractor"), s))
            if verdict is None:
                continue  # LLM unavailable → vector verdict stands
            extra = s.setdefault("extra", {})
            ok = verdict
            extra["topical_adjudicated"] = True
            extra["topical_ok"] = ok
            extra["topical_borderline"] = False
            if ok:
                extra["topical_justification"] = (
                    f"LLM adjudicated RELEVANT at the vector borderline "
                    f"(phenomenon cosine {sp:.2f}, floor {threshold:.2f})"
                )
                if s not in kept:
                    kept.append(s)
                    rejected = [(r, rp, rd) for r, rp, rd in rejected if r is not s]
            else:
                extra["topical_justification"] = (
                    f"LLM adjudicated NOT evidence about the phenomenon at the "
                    f"vector borderline (phenomenon cosine {sp:.2f}, floor {threshold:.2f})"
                )
                if s in kept:
                    kept.remove(s)
                    rejected.append((s, sp, sd))
            adjudicated.append({
                "title": (s.get("title") or "")[:120],
                "score": round(sp, 3),
                "distractor_score": round(sd, 3) if sd is not None else None,
                "verdict": "relevant" if ok else "off_topic",
            })
    if adjudicated:
        scr["topical_adjudicated"] = adjudicated
        kept = [s for s in kept if (s.get("extra") or {}).get("topical_ok") is not False]
        rejected = [(s, sp, sd) for s, sp, sd in rejected
                    if (s.get("extra") or {}).get("topical_ok") is not True]
        logger.info(f"Topical filter: LLM adjudicated {len(adjudicated)} borderline sources")

    # ── Starvation valve: an empty pool helps nobody; sub-floor survival
    # must be VISIBLE (borderline flag + precision penalty + warning line).
    # Pass 2 skips the valve when pass-1 survivors already exist — the pool
    # is not empty, sub-floor rejects stay rejected. ──
    pool_has_annotated = (
        select_new_only
        and any("topical_score" in (s.get("extra") or {}) for s in sources)
    )
    borderline: List[Dict] = []
    if not kept and rejected and not pool_has_annotated:
        by_score = sorted(rejected, key=lambda t: t[1], reverse=True)
        for s, sp, sd in by_score[:_STARVE_KEEP]:
            extra = s.setdefault("extra", {})
            extra["topical_score"] = round(sp, 3)
            extra["topical_ok"] = False
            extra["topical_borderline"] = True
            extra["topical_justification"] = _justification(sp, sd, threshold, False, True)
            borderline.append(s)
            kept.append(s)
        rejected = [(s, sp, sd) for s, sp, sd in rejected
                    if not (s.get("extra") or {}).get("topical_borderline")]

    if rejected:
        scr["topical_rejected"] = [
            {
                "title": (s.get("title") or "")[:120],
                "url": (s.get("url") or "")[:120],
                "score": round(sp, 3),
                "justification": (s.get("extra") or {}).get("topical_justification", ""),
            }
            for s, sp, sd in sorted(rejected, key=lambda t: t[1])
        ][:15]
    if borderline:
        scr["topical_borderline"] = [
            {"title": (s.get("title") or "")[:120], "score": (s.get("extra") or {}).get("topical_score")}
            for s in borderline
        ]
    if rejected or borderline:
        logger.info(
            f"Topical filter ({'pass2' if select_new_only else 'pass1'}): kept "
            f"{len(kept)}/{len(pending)} at floor {threshold:.2f} "
            f"(anchor_kind={anchors.get('kind')}), rejected {len(rejected)}"
            f"{f', starvation-rescued {len(borderline)} borderline' if borderline else ''}"
        )

    if select_new_only:
        # Pass 2: only the new sources are judged; pre-annotated pass-1
        # survivors flow through untouched, and pass-2's own rejects stay
        # rejected (id-set discipline — annotations alone can't distinguish
        # "scored in an earlier pass" from "scored in this one").
        pre_ids = {id(s) for s in sources} - {id(s) for s in pending}
        kept_ids = {id(s) for s in kept}
        return kept + [s for s in sources
                       if id(s) in pre_ids and id(s) not in kept_ids]
    return kept


async def _maybe_await(x):
    return await x if hasattr(x, "__await__") else x
