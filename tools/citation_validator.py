"""
Citation Validator — verifies inline citations in synthesis output.

Ensures:
1. Every factual claim has a citation
2. Citations match actual sources used
3. Tier information is present
4. Marks unsupported claims as [UNVERIFIED]
"""

import logging
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Citation patterns ──
# Matches: [Author Year, P1], [domain.com, P2], [Source URL, P3], [UNVERIFIED]
CITATION_PATTERN = re.compile(
    r"\[([^\]]+?),\s*(P[123]|UNV)\]|\[UNVERIFIED\]"
)

# Pattern to detect factual claims (sentences with specific data)
CLAIM_PATTERNS = [
    re.compile(r"\d+\.?\d*%"),  # Percentages
    re.compile(r"\$[\d,]+\.?\d*"),  # Dollar amounts
    re.compile(r"\d{4}"),  # Years
    re.compile(r"(?:increased|decreased|improved|reduced|achieved|found|showed|demonstrated)\s", re.I),
    re.compile(r"\d+x\b"),  # Multipliers
]


@dataclass
class ValidationResult:
    """Result of citation validation."""
    is_valid: bool
    total_claims: int
    cited_claims: int
    uncited_claims: int
    warnings: List[str]
    corrected_text: str


def extract_citations(text: str) -> List[Tuple[str, str]]:
    """
    Extract all citations from text.

    Returns:
        List of (source_string, tier_string) tuples
    """
    citations = []
    for match in CITATION_PATTERN.finditer(text):
        if match.group(0) == "[UNVERIFIED]":
            citations.append(("UNVERIFIED", "UNV"))
        else:
            source = match.group(1).strip()
            tier = match.group(2).strip()
            citations.append((source, tier))
    return citations


def detect_claims(text: str) -> List[str]:
    """
    Detect sentences that contain factual claims requiring citation.

    Returns:
        List of claim sentences
    """
    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)

    claims = []
    for sentence in sentences:
        for pattern in CLAIM_PATTERNS:
            if pattern.search(sentence):
                claims.append(sentence)
                break

    return claims


def validate_citations(
    text: str,
    available_sources: Optional[List[dict]] = None,
) -> ValidationResult:
    """
    Validate that all factual claims have proper citations.

    Checks:
    1. Every claim has at least one citation
    2. Citations reference sources that were actually used
    3. Tier information is present in citations

    Args:
        text: Synthesis output text
        available_sources: List of dicts with 'url', 'tier', 'title' keys

    Returns:
        ValidationResult with validation status and corrected text
    """
    warnings = []
    corrected_text = text

    # Extract existing citations
    citations = extract_citations(text)

    # Detect claims
    claims = detect_claims(text)

    # Check: claims without citations
    uncited_claims = []
    for claim in claims:
        # Check if the claim sentence contains a citation
        claim_has_citation = bool(CITATION_PATTERN.search(claim))
        if not claim_has_citation:
            uncited_claims.append(claim)

    # Correct uncited claims by adding [UNVERIFIED]
    for claim in uncited_claims:
        # Add [UNVERIFIED] at the end of the sentence
        if claim.rstrip().endswith("."):
            corrected_claim = claim.rstrip()[:-1] + " [UNVERIFIED]."
        else:
            corrected_claim = claim + " [UNVERIFIED]"
        corrected_text = corrected_text.replace(claim, corrected_claim, 1)

    # Validate citations against available sources
    if available_sources:
        available_urls = {s.get("url", "") for s in available_sources}
        available_tiers = {s.get("tier", "") for s in available_sources}

        for source, tier in citations:
            if source == "UNVERIFIED":
                continue

            # Check if the source domain appears in available sources
            source_found = False
            for avail_url in available_urls:
                if source in avail_url or any(part in source for part in avail_url.split("/")):
                    source_found = True
                    break

            if not source_found and available_urls:
                # Source not in available list — flag but don't remove
                warnings.append(f"Citation [{source}, {tier}] not found in available sources")

    # Compile result
    total_claims = len(claims)
    cited_claims = total_claims - len(uncited_claims)
    is_valid = len(uncited_claims) == 0 and len(warnings) == 0

    if uncited_claims:
        warnings.append(f"{len(uncited_claims)} claim(s) lack citations — marked [UNVERIFIED]")

    return ValidationResult(
        is_valid=is_valid,
        total_claims=total_claims,
        cited_claims=cited_claims,
        uncited_claims=len(uncited_claims),
        warnings=warnings,
        corrected_text=corrected_text,
    )


def format_source_citation(url: str, tier: str, title: str = "", authors: List[str] = None) -> str:
    """
    Format a proper inline citation.

    Args:
        url: Source URL
        tier: Source tier (P1/P2/P3/UNV)
        title: Document title
        authors: List of author names

    Returns:
        Formatted citation string like "[Author Year, P1]"
    """
    if authors:
        first_author = authors[0].split(",")[0].strip()
        # Extract year from URL if possible
        year_match = re.search(r"(20\d{2})", url)
        year = year_match.group(1) if year_match else ""
        return f"[{first_author} {year}, {tier}]".strip()

    # Use domain
    from urllib.parse import urlparse
    domain = urlparse(url).netloc
    return f"[{domain}, {tier}]"
