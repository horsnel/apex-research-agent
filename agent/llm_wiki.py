"""
APEX 2.0 — LLM Wiki Engine

The core of APEX 2.0: a compounding knowledge base that turns research
into persistent, verified, lifecycle-managed wiki pages.

Architecture:
  Raw Sources (immutable) → Wiki (LLM-compiled) → Schema (config)

Key features:
  - LLM Wiki pattern: search results are compiled into wiki pages
  - Page Lifecycle (Synthadoc): draft → active → stale → contradicted → archived
  - Hot Cache (Claude-Obsidian): session continuity across research sessions
  - SciMem: scientific memory with provenance tracking
  - Provenance/Conflict Detection: track where info came from, detect contradictions
  - Dialogic Wiki: interactive dialogue around wiki topics
  - Secure Wiki Layer + Concurrency Safety: optimistic locking, edit conflicts

Design principle: "Every query compounds. Every claim is traceable."
"""

import asyncio
import hashlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import httpx

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════
# SUPABASE CLIENT
# ═══════════════════════════════════════════════════════════════

_supabase_client = None

def get_supabase():
    """Lazy-initialize Supabase client."""
    global _supabase_client
    if _supabase_client is not None:
        return _supabase_client

    url = os.getenv("NEXT_PUBLIC_SUPABASE_URL") or os.getenv("SUPABASE_URL", "")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_SERVICE_KEY", "")

    if not url or not key:
        logger.debug("Supabase not configured — wiki features disabled")
        return None

    try:
        from supabase import create_client
        _supabase_client = create_client(url, key)
        return _supabase_client
    except ImportError:
        logger.warning("supabase-py not installed — pip install supabase")
        return None
    except Exception as e:
        logger.warning(f"Supabase client init failed: {e}")
        return None


def reset_supabase():
    """Reset client (for testing)."""
    global _supabase_client
    _supabase_client = None


# ═══════════════════════════════════════════════════════════════
# PAGE LIFECYCLE (SYNTHADOC)
# ═══════════════════════════════════════════════════════════════

LIFECYCLE_DRAFT = "draft"
LIFECYCLE_ACTIVE = "active"
LIFECYCLE_STALE = "stale"
LIFECYCLE_CONTRADICTED = "contradicted"
LIFECYCLE_ARCHIVED = "archived"

# Valid transitions
LIFECYCLE_TRANSITIONS = {
    LIFECYCLE_DRAFT: [LIFECYCLE_ACTIVE, LIFECYCLE_ARCHIVED],
    LIFECYCLE_ACTIVE: [LIFECYCLE_STALE, LIFECYCLE_CONTRADICTED, LIFECYCLE_ARCHIVED],
    LIFECYCLE_STALE: [LIFECYCLE_ACTIVE, LIFECYCLE_CONTRADICTED, LIFECYCLE_ARCHIVED],
    LIFECYCLE_CONTRADICTED: [LIFECYCLE_ACTIVE, LIFECYCLE_STALE, LIFECYCLE_ARCHIVED],
    LIFECYCLE_ARCHIVED: [LIFECYCLE_DRAFT],  # Can be re-drafted
}


def validate_lifecycle_transition(current: str, target: str) -> bool:
    """Check if a lifecycle transition is valid."""
    allowed = LIFECYCLE_TRANSITIONS.get(current, [])
    return target in allowed


# ═══════════════════════════════════════════════════════════════
# DATA CLASSES
# ═══════════════════════════════════════════════════════════════

@dataclass
class WikiPage:
    """A wiki page with lifecycle management."""
    slug: str
    title: str
    content: str = ""
    lifecycle: str = LIFECYCLE_DRAFT
    topic: str = ""
    category: str = "general"
    depth: str = "quick"
    source_count: int = 0
    p1_count: int = 0
    p2_count: int = 0
    p3_count: int = 0
    epistemic_summary: Dict = field(default_factory=dict)
    confidence_score: float = 0.0
    earliest_source_date: Optional[str] = None
    latest_source_date: Optional[str] = None
    stale_after_days: int = 90
    version: int = 1
    parent_version_id: Optional[str] = None
    compiled_by: str = "apex"
    compilation_model: str = ""
    id: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    last_verified_at: Optional[str] = None


@dataclass
class WikiSource:
    """An immutable raw source linked to a wiki page."""
    page_id: str
    url: str
    title: str = ""
    snippet: str = ""
    full_content: Optional[str] = None
    source_name: str = ""
    source_tier: str = "UNV"
    source_category: str = "web"
    authors: List[str] = field(default_factory=list)
    published_date: Optional[str] = None
    doi: Optional[str] = None
    relevance_score: float = 0.0
    adjusted_score: float = 0.0
    is_immutable: bool = True
    id: Optional[str] = None


@dataclass
class ResearchCache:
    """A cached research result for session continuity."""
    query_text: str
    normalized_query: str
    query_hash: str
    report: str = ""
    sources: List[Dict] = field(default_factory=list)
    followups: List[str] = field(default_factory=list)
    verification_summary: Dict = field(default_factory=dict)
    mode: str = "quick"
    apex_tier: str = "apex-free"
    original_latency_ms: int = 0
    cache_hit_count: int = 0
    is_hot: bool = False
    wiki_page_id: Optional[str] = None
    user_id: Optional[str] = None
    id: Optional[str] = None


@dataclass
class ClaimVerification:
    """A persisted claim verification result."""
    claim_text: str
    claim_hash: str
    epistemic_status: str  # ESTABLISHED, TENTATIVE, ACTIVE_DEBATE, SPECULATIVE, UNVERIFIED
    confidence: float = 0.0
    evidence_type: str = ""
    supporting_sources: List[Dict] = field(default_factory=list)
    conflicting_sources: List[Dict] = field(default_factory=list)
    sample_size: Optional[int] = None
    study_year: Optional[int] = None
    page_id: Optional[str] = None
    id: Optional[str] = None


@dataclass
class SourceProvenance:
    """Provenance tracking for a source."""
    source_url: str
    source_title: str = ""
    source_tier: str = "UNV"
    contributed_to_type: str = "wiki_page"  # wiki_page, claim, cache_entry
    contributed_to_id: str = ""
    found_by_query: str = ""
    found_by_source: str = ""
    found_at_depth: int = 0
    contradicts_source_url: Optional[str] = None
    conflict_type: Optional[str] = None  # direct, methodological, temporal, scope
    conflict_notes: Optional[str] = None
    is_retracted: bool = False
    retraction_notice_url: Optional[str] = None
    id: Optional[str] = None


@dataclass
class WikiDialogue:
    """A dialogue entry for the Dialogic Wiki."""
    page_id: str
    message: str
    role: str = "user"  # user, apex, reviewer
    message_type: str = "question"  # question, answer, challenge, correction, clarification, debate, summary
    referenced_claims: List[Dict] = field(default_factory=list)
    intent: str = ""  # verify, explore, challenge, deepen
    user_id: Optional[str] = None
    id: Optional[str] = None


@dataclass
class WikiEditLog:
    """An edit log entry for concurrency safety."""
    page_id: str
    editor: str = "apex"
    edit_type: str = "update"
    field_changed: str = ""
    old_value: Optional[str] = None
    new_value: Optional[str] = None
    base_version: int = 1
    result_version: int = 1
    id: Optional[str] = None


# ═══════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def slugify(text: str) -> str:
    """Convert text to URL-safe slug."""
    slug = text.lower().strip()
    slug = re.sub(r'[^\w\s-]', '', slug)
    slug = re.sub(r'[\s_]+', '-', slug)
    slug = re.sub(r'-+', '-', slug)
    return slug[:80].rstrip('-')


def compute_query_hash(query: str, mode: str = "quick", tier: str = "apex-free") -> str:
    """Compute a deterministic hash for a query+mode+tier combination."""
    normalized = query.lower().strip()
    key = f"{normalized}|{mode}|{tier}"
    return hashlib.sha256(key.encode()).hexdigest()


def compute_claim_hash(claim: str) -> str:
    """Compute a deterministic hash for a claim."""
    normalized = claim.lower().strip()
    return hashlib.sha256(normalized.encode()).hexdigest()


# ═══════════════════════════════════════════════════════════════
# LLM WIKI ENGINE
# ═══════════════════════════════════════════════════════════════

class LLMWikiEngine:
    """
    The core LLM Wiki engine — manages the full lifecycle of wiki pages.

    Workflow:
    1. Research query comes in
    2. Check cache for existing result
    3. If not cached, perform research
    4. Compile research into a wiki page (or update existing)
    5. Store raw sources (immutable)
    6. Run claim verification
    7. Record provenance
    8. Update cache
    9. Transition page lifecycle
    """

    def __init__(self):
        self.db = get_supabase()

    # ─── Cache Operations ─────────────────────────────────────

    async def check_cache(
        self,
        query: str,
        mode: str = "quick",
        apex_tier: str = "apex-free",
        user_id: Optional[str] = None,
    ) -> Optional[ResearchCache]:
        """
        Check if a research result is already cached.

        Returns the cached result if found and not expired, None otherwise.
        """
        if not self.db:
            return None

        query_hash = compute_query_hash(query, mode, apex_tier)

        try:
            result = self.db.table("apex_research_cache") \
                .select("*") \
                .eq("query_hash", query_hash) \
                .gt("expires_at", datetime.utcnow().isoformat()) \
                .execute()

            if result.data and len(result.data) > 0:
                row = result.data[0]

                # Update cache hit count and last accessed
                self.db.table("apex_research_cache") \
                    .update({
                        "cache_hit_count": row.get("cache_hit_count", 0) + 1,
                        "last_accessed_at": datetime.utcnow().isoformat(),
                    }) \
                    .eq("id", row["id"]) \
                    .execute()

                return ResearchCache(
                    id=row["id"],
                    query_text=row["query_text"],
                    normalized_query=row["normalized_query"],
                    query_hash=row["query_hash"],
                    report=row.get("report", ""),
                    sources=row.get("sources", []),
                    followups=row.get("followups", []),
                    verification_summary=row.get("verification_summary", {}),
                    mode=row.get("mode", "quick"),
                    apex_tier=row.get("apex_tier", "apex-free"),
                    original_latency_ms=row.get("original_latency_ms", 0),
                    cache_hit_count=row.get("cache_hit_count", 0) + 1,
                    is_hot=row.get("is_hot", False),
                    wiki_page_id=row.get("wiki_page_id"),
                    user_id=row.get("user_id"),
                )

        except Exception as e:
            logger.debug(f"Cache check failed: {e}")

        return None

    async def store_cache(
        self,
        query: str,
        report: str,
        sources: List[Dict],
        followups: List[str],
        verification_summary: Dict,
        mode: str = "quick",
        apex_tier: str = "apex-free",
        original_latency_ms: int = 0,
        wiki_page_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> Optional[str]:
        """Store a research result in the cache."""
        if not self.db:
            return None

        normalized = query.lower().strip()
        query_hash = compute_query_hash(query, mode, apex_tier)

        try:
            # Upsert by hash
            existing = self.db.table("apex_research_cache") \
                .select("id") \
                .eq("query_hash", query_hash) \
                .execute()

            data = {
                "query_text": query,
                "normalized_query": normalized,
                "query_hash": query_hash,
                "report": report,
                "sources": json.dumps(sources),
                "followups": json.dumps(followups),
                "verification_summary": json.dumps(verification_summary),
                "mode": mode,
                "apex_tier": apex_tier,
                "original_latency_ms": original_latency_ms,
                "wiki_page_id": wiki_page_id,
                "user_id": user_id,
                "expires_at": (datetime.utcnow() + timedelta(days=7)).isoformat(),
                "last_accessed_at": datetime.utcnow().isoformat(),
            }

            if existing.data and len(existing.data) > 0:
                # Update existing
                self.db.table("apex_research_cache") \
                    .update(data) \
                    .eq("id", existing.data[0]["id"]) \
                    .execute()
                return existing.data[0]["id"]
            else:
                # Insert new
                result = self.db.table("apex_research_cache") \
                    .insert(data) \
                    .execute()
                return result.data[0]["id"] if result.data else None

        except Exception as e:
            logger.warning(f"Cache store failed: {e}")
            return None

    async def get_hot_cache(self, user_id: Optional[str] = None) -> List[ResearchCache]:
        """
        Get hot cache items for session continuity.
        These are loaded on session start for instant recall.
        """
        if not self.db:
            return []

        try:
            query = self.db.table("apex_research_cache") \
                .select("*") \
                .eq("is_hot", True) \
                .gt("expires_at", datetime.utcnow().isoformat()) \
                .order("last_accessed_at", desc=True) \
                .limit(10)

            if user_id:
                query = query.eq("user_id", user_id)

            result = query.execute()

            return [
                ResearchCache(
                    id=row["id"],
                    query_text=row["query_text"],
                    normalized_query=row["normalized_query"],
                    query_hash=row["query_hash"],
                    report=row.get("report", ""),
                    sources=row.get("sources", []),
                    followups=row.get("followups", []),
                    verification_summary=row.get("verification_summary", {}),
                    mode=row.get("mode", "quick"),
                    apex_tier=row.get("apex_tier", "apex-free"),
                    is_hot=row.get("is_hot", False),
                    wiki_page_id=row.get("wiki_page_id"),
                )
                for row in result.data
            ]

        except Exception as e:
            logger.debug(f"Hot cache fetch failed: {e}")
            return []

    async def mark_hot(self, cache_id: str, is_hot: bool = True) -> bool:
        """Mark a cache entry as hot for session continuity."""
        if not self.db:
            return False

        try:
            self.db.table("apex_research_cache") \
                .update({"is_hot": is_hot}) \
                .eq("id", cache_id) \
                .execute()
            return True
        except Exception as e:
            logger.debug(f"Mark hot failed: {e}")
            return False

    # ─── Wiki Page Operations ──────────────────────────────────

    async def get_or_create_wiki_page(
        self,
        query: str,
        category: str = "general",
        depth: str = "quick",
    ) -> Tuple[WikiPage, bool]:
        """
        Get an existing wiki page by topic/slug, or create a draft.

        Returns (WikiPage, was_created).
        """
        if not self.db:
            # Return in-memory page
            slug = slugify(query)
            page = WikiPage(slug=slug, title=query, topic=query, category=category, depth=depth)
            return page, True

        slug = slugify(query)

        try:
            # Try to find existing page
            result = self.db.table("apex_wiki_pages") \
                .select("*") \
                .eq("slug", slug) \
                .execute()

            if result.data and len(result.data) > 0:
                row = result.data[0]
                page = WikiPage(
                    id=row["id"],
                    slug=row["slug"],
                    title=row["title"],
                    content=row.get("content", ""),
                    lifecycle=row.get("lifecycle", "draft"),
                    topic=row.get("topic", ""),
                    category=row.get("category", "general"),
                    depth=row.get("depth", "quick"),
                    source_count=row.get("source_count", 0),
                    p1_count=row.get("p1_count", 0),
                    p2_count=row.get("p2_count", 0),
                    p3_count=row.get("p3_count", 0),
                    epistemic_summary=row.get("epistemic_summary", {}),
                    confidence_score=row.get("confidence_score", 0.0),
                    version=row.get("version", 1),
                    compiled_by=row.get("compiled_by", "apex"),
                    compilation_model=row.get("compilation_model", ""),
                    created_at=row.get("created_at"),
                    updated_at=row.get("updated_at"),
                    last_verified_at=row.get("last_verified_at"),
                )
                return page, False

        except Exception as e:
            logger.debug(f"Wiki page lookup failed: {e}")

        # Create new draft page
        page = WikiPage(slug=slug, title=query, topic=query, category=category, depth=depth)
        created_id = await self._insert_wiki_page(page)
        if created_id:
            page.id = created_id
        return page, True

    async def _insert_wiki_page(self, page: WikiPage) -> Optional[str]:
        """Insert a new wiki page into Supabase."""
        if not self.db:
            return None

        data = {
            "slug": page.slug,
            "title": page.title,
            "content": page.content,
            "lifecycle": page.lifecycle,
            "topic": page.topic,
            "category": page.category,
            "depth": page.depth,
            "source_count": page.source_count,
            "p1_count": page.p1_count,
            "p2_count": page.p2_count,
            "p3_count": page.p3_count,
            "epistemic_summary": json.dumps(page.epistemic_summary),
            "confidence_score": page.confidence_score,
            "stale_after_days": page.stale_after_days,
            "version": page.version,
            "compiled_by": page.compiled_by,
            "compilation_model": page.compilation_model,
        }

        try:
            result = self.db.table("apex_wiki_pages").insert(data).execute()
            return result.data[0]["id"] if result.data else None
        except Exception as e:
            logger.warning(f"Wiki page insert failed: {e}")
            return None

    async def update_wiki_page(
        self,
        page_id: str,
        content: str,
        sources: List[Dict],
        verification_summary: Dict,
        confidence_score: float,
        compilation_model: str = "",
        base_version: Optional[int] = None,
    ) -> bool:
        """
        Update a wiki page with new content (optimistic locking).

        Uses version-based concurrency: if base_version doesn't match
        current version, the update fails (conflict detected).
        """
        if not self.db:
            return False

        # Count tier distribution
        p1 = sum(1 for s in sources if s.get("tier") == "P1")
        p2 = sum(1 for s in sources if s.get("tier") == "P2")
        p3 = sum(1 for s in sources if s.get("tier") == "P3")

        # Determine lifecycle transition
        # If confidence is high enough and we have P1 sources, promote to active
        new_lifecycle = None
        if confidence_score >= 0.5 and p1 > 0:
            new_lifecycle = LIFECYCLE_ACTIVE

        try:
            # Get current version for optimistic locking
            current = self.db.table("apex_wiki_pages") \
                .select("version, lifecycle") \
                .eq("id", page_id) \
                .execute()

            if not current.data:
                return False

            current_version = current.data[0]["version"]
            current_lifecycle = current.data[0]["lifecycle"]

            # Check optimistic lock
            if base_version is not None and base_version != current_version:
                logger.warning(f"Version conflict: expected {base_version}, got {current_version}")
                return False  # Conflict — caller must resolve

            new_version = current_version + 1

            # Determine new lifecycle
            if new_lifecycle and validate_lifecycle_transition(current_lifecycle, new_lifecycle):
                lifecycle_update = new_lifecycle
            elif current_lifecycle == LIFECYCLE_STALE and confidence_score >= 0.5:
                lifecycle_update = LIFECYCLE_ACTIVE  # Refresh stale page
            elif current_lifecycle == LIFECYCLE_CONTRADICTED and confidence_score >= 0.7:
                lifecycle_update = LIFECYCLE_ACTIVE  # Resolve contradiction
            else:
                lifecycle_update = current_lifecycle

            update_data = {
                "content": content,
                "source_count": len(sources),
                "p1_count": p1,
                "p2_count": p2,
                "p3_count": p3,
                "epistemic_summary": json.dumps(verification_summary),
                "confidence_score": confidence_score,
                "version": new_version,
                "compilation_model": compilation_model,
                "lifecycle": lifecycle_update,
                "updated_at": datetime.utcnow().isoformat(),
                "last_verified_at": datetime.utcnow().isoformat(),
            }

            self.db.table("apex_wiki_pages") \
                .update(update_data) \
                .eq("id", page_id) \
                .execute()

            # Log the edit
            await self._log_edit(
                page_id=page_id,
                edit_type="update",
                field_changed="content",
                base_version=current_version,
                result_version=new_version,
            )

            return True

        except Exception as e:
            logger.warning(f"Wiki page update failed: {e}")
            return False

    async def transition_lifecycle(
        self,
        page_id: str,
        target_lifecycle: str,
        reason: str = "",
    ) -> bool:
        """Transition a wiki page's lifecycle state."""
        if not self.db:
            return False

        try:
            current = self.db.table("apex_wiki_pages") \
                .select("lifecycle, version") \
                .eq("id", page_id) \
                .execute()

            if not current.data:
                return False

            current_lifecycle = current.data[0]["lifecycle"]
            current_version = current.data[0]["version"]

            if not validate_lifecycle_transition(current_lifecycle, target_lifecycle):
                logger.warning(f"Invalid lifecycle transition: {current_lifecycle} → {target_lifecycle}")
                return False

            new_version = current_version + 1

            self.db.table("apex_wiki_pages") \
                .update({
                    "lifecycle": target_lifecycle,
                    "version": new_version,
                    "updated_at": datetime.utcnow().isoformat(),
                }) \
                .eq("id", page_id) \
                .execute()

            await self._log_edit(
                page_id=page_id,
                edit_type="lifecycle_change",
                field_changed="lifecycle",
                old_value=current_lifecycle,
                new_value=target_lifecycle,
                base_version=current_version,
                result_version=new_version,
            )

            return True

        except Exception as e:
            logger.warning(f"Lifecycle transition failed: {e}")
            return False

    # ─── Source Operations ─────────────────────────────────────

    async def store_sources(
        self,
        page_id: str,
        sources: List[Dict],
        query: str = "",
    ) -> int:
        """
        Store raw sources (immutable) linked to a wiki page.

        Returns the number of sources stored (deduplicates by URL).
        """
        if not self.db:
            return 0

        stored = 0

        try:
            # Get existing source URLs for dedup
            existing = self.db.table("apex_wiki_sources") \
                .select("url") \
                .eq("page_id", page_id) \
                .execute()

            existing_urls = {row["url"] for row in existing.data}

            for source in sources:
                url = source.get("url", "")
                if not url or url in existing_urls:
                    continue

                data = {
                    "page_id": page_id,
                    "url": url,
                    "title": source.get("title", ""),
                    "snippet": source.get("snippet", "")[:2000],  # Limit snippet size
                    "full_content": source.get("full_content"),
                    "source_name": source.get("source_name", ""),
                    "source_tier": source.get("tier", "UNV"),
                    "source_category": source.get("category", "web"),
                    "authors": source.get("authors", []),
                    "published_date": source.get("published_date"),
                    "doi": source.get("doi"),
                    "relevance_score": source.get("adjusted_score", source.get("relevance_score", 0.0)),
                    "adjusted_score": source.get("adjusted_score", 0.0),
                }

                try:
                    self.db.table("apex_wiki_sources").insert(data).execute()
                    stored += 1
                except Exception as e:
                    logger.debug(f"Source insert failed for {url}: {e}")

        except Exception as e:
            logger.warning(f"Store sources failed: {e}")

        return stored

    # ─── Claim Verification Persistence ────────────────────────

    async def persist_claim_verifications(
        self,
        page_id: str,
        claims: List[Any],  # List of VerifiedClaim from research_engine
    ) -> int:
        """
        Persist claim verification results.

        Returns number of claims persisted.
        """
        if not self.db:
            return 0

        stored = 0

        for claim in claims:
            claim_text = claim.statement if hasattr(claim, 'statement') else str(claim)
            claim_hash = compute_claim_hash(claim_text)

            data = {
                "page_id": page_id,
                "claim_text": claim_text,
                "claim_hash": claim_hash,
                "epistemic_status": claim.epistemic_status if hasattr(claim, 'epistemic_status') else "UNVERIFIED",
                "confidence": claim.confidence if hasattr(claim, 'confidence') else 0.0,
                "evidence_type": claim.evidence_type if hasattr(claim, 'evidence_type') else "",
                "supporting_sources": json.dumps(
                    claim.supporting_sources if hasattr(claim, 'supporting_sources') else []
                ),
                "conflicting_sources": json.dumps(
                    claim.conflicting_sources if hasattr(claim, 'conflicting_sources') else []
                ),
                "sample_size": claim.sample_size if hasattr(claim, 'sample_size') else None,
                "study_year": claim.year if hasattr(claim, 'year') else None,
            }

            try:
                # Upsert by hash
                existing = self.db.table("apex_claim_verifications") \
                    .select("id") \
                    .eq("claim_hash", claim_hash) \
                    .execute()

                if existing.data:
                    self.db.table("apex_claim_verifications") \
                        .update(data) \
                        .eq("id", existing.data[0]["id"]) \
                        .execute()
                else:
                    self.db.table("apex_claim_verifications").insert(data).execute()

                stored += 1

            except Exception as e:
                logger.debug(f"Claim persist failed: {e}")

        return stored

    # ─── Provenance Tracking ───────────────────────────────────

    async def record_provenance(
        self,
        source_url: str,
        source_title: str,
        source_tier: str,
        contributed_to_type: str,
        contributed_to_id: str,
        found_by_query: str = "",
        found_by_source: str = "",
        found_at_depth: int = 0,
        contradicts_source_url: Optional[str] = None,
        conflict_type: Optional[str] = None,
        conflict_notes: Optional[str] = None,
    ) -> bool:
        """Record provenance for a source."""
        if not self.db:
            return False

        data = {
            "source_url": source_url,
            "source_title": source_title,
            "source_tier": source_tier,
            "contributed_to_type": contributed_to_type,
            "contributed_to_id": contributed_to_id,
            "found_by_query": found_by_query,
            "found_by_source": found_by_source,
            "found_at_depth": found_at_depth,
            "contradicts_source_url": contradicts_source_url,
            "conflict_type": conflict_type,
            "conflict_notes": conflict_notes,
        }

        try:
            self.db.table("apex_source_provenance").insert(data).execute()
            return True
        except Exception as e:
            logger.debug(f"Provenance record failed: {e}")
            return False

    async def detect_conflicts(self, page_id: str) -> List[Dict]:
        """
        Detect conflicting sources for a wiki page.

        Returns list of conflict records.
        """
        if not self.db:
            return []

        try:
            result = self.db.table("apex_source_provenance") \
                .select("*") \
                .eq("contributed_to_type", "wiki_page") \
                .eq("contributed_to_id", page_id) \
                .not_.is_("contradicts_source_url", "null") \
                .execute()

            return result.data

        except Exception as e:
            logger.debug(f"Conflict detection failed: {e}")
            return []

    async def mark_retracted(
        self,
        source_url: str,
        retraction_notice_url: str,
    ) -> bool:
        """Mark a source as retracted."""
        if not self.db:
            return False

        try:
            self.db.table("apex_source_provenance") \
                .update({
                    "is_retracted": True,
                    "retraction_notice_url": retraction_notice_url,
                    "retracted_at": datetime.utcnow().isoformat(),
                }) \
                .eq("source_url", source_url) \
                .execute()

            # Check if any wiki pages should be transitioned to contradicted
            affected = self.db.table("apex_source_provenance") \
                .select("contributed_to_id") \
                .eq("source_url", source_url) \
                .eq("contributed_to_type", "wiki_page") \
                .execute()

            for row in affected.data:
                await self.transition_lifecycle(
                    row["contributed_to_id"],
                    LIFECYCLE_CONTRADICTED,
                    reason=f"Source retracted: {source_url}",
                )

            return True

        except Exception as e:
            logger.debug(f"Retraction mark failed: {e}")
            return False

    # ─── Dialogic Wiki ─────────────────────────────────────────

    async def add_dialogue(
        self,
        page_id: str,
        message: str,
        role: str = "user",
        message_type: str = "question",
        referenced_claims: List[Dict] = None,
        intent: str = "",
        user_id: Optional[str] = None,
    ) -> Optional[str]:
        """Add a dialogue entry to a wiki page."""
        if not self.db:
            return None

        data = {
            "page_id": page_id,
            "message": message,
            "role": role,
            "message_type": message_type,
            "referenced_claims": json.dumps(referenced_claims or []),
            "intent": intent,
            "user_id": user_id,
        }

        try:
            result = self.db.table("apex_wiki_dialogue").insert(data).execute()
            return result.data[0]["id"] if result.data else None
        except Exception as e:
            logger.debug(f"Dialogue insert failed: {e}")
            return None

    async def get_dialogue(
        self,
        page_id: str,
        limit: int = 20,
    ) -> List[WikiDialogue]:
        """Get dialogue history for a wiki page."""
        if not self.db:
            return []

        try:
            result = self.db.table("apex_wiki_dialogue") \
                .select("*") \
                .eq("page_id", page_id) \
                .order("created_at", desc=False) \
                .limit(limit) \
                .execute()

            return [
                WikiDialogue(
                    id=row["id"],
                    page_id=row["page_id"],
                    message=row["message"],
                    role=row.get("role", "user"),
                    message_type=row.get("message_type", "question"),
                    referenced_claims=row.get("referenced_claims", []),
                    intent=row.get("intent", ""),
                    user_id=row.get("user_id"),
                )
                for row in result.data
            ]

        except Exception as e:
            logger.debug(f"Dialogue fetch failed: {e}")
            return []

    # ─── Edit Log (Concurrency Safety) ─────────────────────────

    async def _log_edit(
        self,
        page_id: str,
        edit_type: str,
        field_changed: str = "",
        old_value: Optional[str] = None,
        new_value: Optional[str] = None,
        base_version: int = 1,
        result_version: int = 1,
        editor: str = "apex",
    ) -> None:
        """Log a wiki page edit for audit and concurrency."""
        if not self.db:
            return

        data = {
            "page_id": page_id,
            "editor": editor,
            "edit_type": edit_type,
            "field_changed": field_changed,
            "old_value": old_value,
            "new_value": new_value,
            "base_version": base_version,
            "result_version": result_version,
        }

        try:
            self.db.table("apex_wiki_edit_log").insert(data).execute()
        except Exception as e:
            logger.debug(f"Edit log failed: {e}")

    # ─── Stale Detection ───────────────────────────────────────

    async def detect_stale_pages(self) -> int:
        """
        Find and mark pages that are stale based on their stale_after_days setting.

        Returns number of pages marked as stale.
        """
        if not self.db:
            return 0

        try:
            # Call the Supabase function
            result = self.db.rpc("mark_stale_wiki_pages").execute()
            return result.data if result.data else 0
        except Exception as e:
            logger.debug(f"Stale detection failed: {e}")
            return 0

    # ─── Full Research → Wiki Pipeline ─────────────────────────

    async def research_to_wiki(
        self,
        query: str,
        report: str,
        sources: List[Dict],
        verification: Any,  # VerificationResult from research_engine
        mode: str = "quick",
        apex_tier: str = "apex-free",
        depth: str = "quick",
        category: str = "general",
        user_id: Optional[str] = None,
        original_latency_ms: int = 0,
    ) -> Tuple[WikiPage, Optional[str]]:
        """
        Full pipeline: Research result → Wiki page + Cache + Provenance.

        This is the main entry point for the APEX 2.0 LLM Wiki pattern.

        Steps:
        1. Check cache → return if hit
        2. Get or create wiki page
        3. Store raw sources (immutable)
        4. Update wiki page content
        5. Persist claim verifications
        6. Record provenance for each source
        7. Store in cache
        8. Return wiki page + cache ID

        Args:
            query: Research query
            report: Generated report text
            sources: List of source dicts
            verification: VerificationResult from research_engine
            mode: quick/thorough
            apex_tier: apex-free/apex-premium
            depth: quick/thorough
            category: Topic category
            user_id: Optional user ID
            original_latency_ms: How long the research took

        Returns:
            (WikiPage, cache_id) tuple
        """
        # Step 1: Check cache
        cached = await self.check_cache(query, mode, apex_tier, user_id)
        if cached:
            logger.info(f"Cache hit for: {query[:50]}")
            # Get the linked wiki page
            if cached.wiki_page_id and self.db:
                page_result = self.db.table("apex_wiki_pages") \
                    .select("*") \
                    .eq("id", cached.wiki_page_id) \
                    .execute()
                if page_result.data:
                    row = page_result.data[0]
                    page = WikiPage(
                        id=row["id"], slug=row["slug"], title=row["title"],
                        content=row.get("content", ""), lifecycle=row.get("lifecycle", "draft"),
                    )
                    return page, cached.id
            return WikiPage(slug=slugify(query), title=query), cached.id

        # Step 2: Get or create wiki page
        page, was_created = await self.get_or_create_wiki_page(query, category, depth)

        # Step 3: Store raw sources
        if page.id and sources:
            stored = await self.store_sources(page.id, sources, query)
            logger.info(f"Stored {stored} sources for wiki page: {page.slug}")

        # Step 4: Update wiki page content
        if page.id:
            # Compute verification summary
            ver_summary = {}
            if verification:
                ver_summary = {
                    "established": verification.established_count if hasattr(verification, 'established_count') else 0,
                    "tentative": verification.tentative_count if hasattr(verification, 'tentative_count') else 0,
                    "contested": verification.contested_count if hasattr(verification, 'contested_count') else 0,
                    "unverifiable": verification.unverifiable_count if hasattr(verification, 'unverifiable_count') else 0,
                }

            # Compute confidence score
            total_claims = sum(ver_summary.values()) or 1
            confidence = ver_summary.get("established", 0) / total_claims

            await self.update_wiki_page(
                page_id=page.id,
                content=report,
                sources=sources,
                verification_summary=ver_summary,
                confidence_score=round(confidence, 2),
                compilation_model="apex-2.0",
                base_version=page.version if not was_created else None,
            )
            page.content = report

        # Step 5: Persist claim verifications
        if page.id and verification and hasattr(verification, 'claims'):
            await self.persist_claim_verifications(page.id, verification.claims)

        # Step 6: Record provenance
        if page.id:
            for source in sources:
                await self.record_provenance(
                    source_url=source.get("url", ""),
                    source_title=source.get("title", ""),
                    source_tier=source.get("tier", "UNV"),
                    contributed_to_type="wiki_page",
                    contributed_to_id=page.id,
                    found_by_query=query,
                    found_by_source=source.get("source_name", ""),
                )

        # Step 7: Store in cache
        ver_summary = {}
        if verification:
            ver_summary = {
                "established": verification.established_count if hasattr(verification, 'established_count') else 0,
                "tentative": verification.tentative_count if hasattr(verification, 'tentative_count') else 0,
                "contested": verification.contested_count if hasattr(verification, 'contested_count') else 0,
                "unverifiable": verification.unverifiable_count if hasattr(verification, 'unverifiable_count') else 0,
            }

        cache_id = await self.store_cache(
            query=query,
            report=report,
            sources=sources,
            followups=[],  # Will be populated by caller
            verification_summary=ver_summary,
            mode=mode,
            apex_tier=apex_tier,
            original_latency_ms=original_latency_ms,
            wiki_page_id=page.id,
            user_id=user_id,
        )

        return page, cache_id

    # ─── Wiki Search ───────────────────────────────────────────

    async def search_wiki(self, query: str, limit: int = 5) -> List[WikiPage]:
        """Search wiki pages by text matching."""
        if not self.db:
            return []

        try:
            # Use Supabase full-text search
            result = self.db.table("apex_wiki_pages") \
                .select("*") \
                .text_search("title", query) \
                .limit(limit) \
                .execute()

            pages = []
            for row in result.data:
                pages.append(WikiPage(
                    id=row["id"],
                    slug=row["slug"],
                    title=row["title"],
                    content=row.get("content", ""),
                    lifecycle=row.get("lifecycle", "draft"),
                    confidence_score=row.get("confidence_score", 0.0),
                    source_count=row.get("source_count", 0),
                ))

            return pages

        except Exception as e:
            logger.debug(f"Wiki search failed: {e}")
            return []

    async def get_wiki_page(self, slug: str) -> Optional[Dict]:
        """Get a full wiki page with all sources and verifications."""
        if not self.db:
            return None

        try:
            # Use the Supabase function for joined data
            result = self.db.rpc("get_wiki_page_full", {"p_slug": slug}).execute()
            return result.data if result.data else None
        except Exception as e:
            logger.debug(f"Wiki page fetch failed: {e}")
            return None

    # ─── Cache Maintenance ─────────────────────────────────────

    async def clean_expired_cache(self) -> int:
        """Remove expired cache entries."""
        if not self.db:
            return 0

        try:
            result = self.db.rpc("clean_expired_cache").execute()
            return result.data if result.data else 0
        except Exception as e:
            logger.debug(f"Cache cleanup failed: {e}")
            return 0


# ═══════════════════════════════════════════════════════════════
# MODULE-LEVEL CONVENIENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════

_engine = None

def get_wiki_engine() -> LLMWikiEngine:
    """Get or create the singleton LLM Wiki engine."""
    global _engine
    if _engine is None:
        _engine = LLMWikiEngine()
    return _engine
