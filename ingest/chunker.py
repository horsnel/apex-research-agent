"""
Semantic Chunker — splits documents into chunks respecting token budgets.

Strategies:
1. Fixed-size with overlap (default)
2. Sentence-boundary aware
3. Markdown header-aware for structured documents

All strategies target a configurable token count with overlap.
"""

import re
import logging
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)

# Approximate token count: 1 token ≈ 4 characters for English
CHARS_PER_TOKEN = 4


@dataclass
class Chunk:
    """A document chunk with metadata."""
    text: str
    chunk_index: int
    total_chunks: int
    start_char: int = 0
    end_char: int = 0
    token_count: int = 0
    metadata: dict = field(default_factory=dict)


def _estimate_tokens(text: str) -> int:
    """Estimate token count from text length."""
    return max(1, len(text) // CHARS_PER_TOKEN)


def _split_sentences(text: str) -> List[str]:
    """Split text into sentences, preserving whitespace."""
    # Split on sentence-ending punctuation followed by space or end
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]


def _split_by_headers(text: str) -> List[tuple]:
    """Split markdown text by headers, returning (header, content) pairs."""
    header_pattern = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)
    sections = []
    last_end = 0
    last_header = ""

    for match in header_pattern.finditer(text):
        if last_end > 0 or match.start() > 0:
            content = text[last_end:match.start()].strip()
            if content:
                sections.append((last_header, content))
        last_header = f"{match.group(1)} {match.group(2)}"
        last_end = match.end()

    # Remaining content
    if last_end < len(text):
        content = text[last_end:].strip()
        if content:
            sections.append((last_header, content))

    # If no headers found, return entire text as one section
    if not sections:
        sections.append(("", text.strip()))

    return sections


def chunk_fixed_size(
    text: str,
    chunk_size_tokens: int = 512,
    overlap_pct: float = 0.20,
) -> List[Chunk]:
    """
    Split text into fixed-size chunks with overlap.

    Args:
        text: Input text to chunk
        chunk_size_tokens: Target tokens per chunk
        overlap_pct: Fraction of overlap between consecutive chunks (0.0-1.0)

    Returns:
        List of Chunk objects
    """
    if not text or not text.strip():
        return []

    chunk_size_chars = chunk_size_tokens * CHARS_PER_TOKEN
    overlap_chars = int(chunk_size_chars * overlap_pct)
    step_chars = chunk_size_chars - overlap_chars

    chunks = []
    start = 0
    idx = 0

    while start < len(text):
        end = start + chunk_size_chars

        # If not the last chunk, try to break at a sentence boundary
        if end < len(text):
            # Look for sentence boundary within the last 20% of the chunk
            boundary_zone_start = start + int(chunk_size_chars * 0.8)
            boundary_zone = text[boundary_zone_start:end]
            sentence_end = boundary_zone.rfind('. ')
            if sentence_end != -1:
                end = boundary_zone_start + sentence_end + 2
            else:
                # Try newline
                newline_pos = boundary_zone.rfind('\n')
                if newline_pos != -1:
                    end = boundary_zone_start + newline_pos + 1

        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append(Chunk(
                text=chunk_text,
                chunk_index=idx,
                total_chunks=0,  # Will be set after all chunks created
                start_char=start,
                end_char=end,
                token_count=_estimate_tokens(chunk_text),
            ))
            idx += 1

        start += step_chars

    # Set total_chunks
    for chunk in chunks:
        chunk.total_chunks = len(chunks)

    logger.debug(f"Chunked text ({len(text)} chars) into {len(chunks)} chunks")
    return chunks


def chunk_semantic(
    text: str,
    chunk_size_tokens: int = 512,
    overlap_pct: float = 0.20,
) -> List[Chunk]:
    """
    Split text using sentence-boundary-aware chunking.

    Builds chunks by accumulating sentences until the token budget is reached,
    then starts a new chunk with overlap from the last few sentences.

    Args:
        text: Input text to chunk
        chunk_size_tokens: Target tokens per chunk
        overlap_pct: Fraction of overlap (implemented as sentence carry-over)

    Returns:
        List of Chunk objects
    """
    if not text or not text.strip():
        return []

    sentences = _split_sentences(text)
    if not sentences:
        return []

    chunk_size_chars = chunk_size_tokens * CHARS_PER_TOKEN
    overlap_sentences = max(1, int(len(sentences) * overlap_pct * 0.1))

    chunks = []
    current_sentences: List[str] = []
    current_len = 0
    idx = 0

    for i, sentence in enumerate(sentences):
        sent_len = len(sentence)

        if current_len + sent_len > chunk_size_chars and current_sentences:
            # Finalize current chunk
            chunk_text = " ".join(current_sentences)
            chunks.append(Chunk(
                text=chunk_text,
                chunk_index=idx,
                total_chunks=0,
                start_char=0,
                end_char=0,
                token_count=_estimate_tokens(chunk_text),
            ))
            idx += 1

            # Overlap: carry over last N sentences
            overlap_start = max(0, len(current_sentences) - overlap_sentences)
            current_sentences = current_sentences[overlap_start:]
            current_len = sum(len(s) for s in current_sentences)

        current_sentences.append(sentence)
        current_len += sent_len

    # Final chunk
    if current_sentences:
        chunk_text = " ".join(current_sentences)
        chunks.append(Chunk(
            text=chunk_text,
            chunk_index=idx,
            total_chunks=0,
            start_char=0,
            end_char=0,
            token_count=_estimate_tokens(chunk_text),
        ))

    # Set total_chunks
    for chunk in chunks:
        chunk.total_chunks = len(chunks)

    logger.debug(f"Semantic-chunked text into {len(chunks)} chunks")
    return chunks


def chunk_markdown(
    text: str,
    chunk_size_tokens: int = 512,
    overlap_pct: float = 0.20,
) -> List[Chunk]:
    """
    Split markdown text respecting header structure.

    Splits by headers first, then applies fixed-size chunking
    within each section if it exceeds the token budget.

    Args:
        text: Markdown text to chunk
        chunk_size_tokens: Target tokens per chunk
        overlap_pct: Fraction of overlap between chunks

    Returns:
        List of Chunk objects
    """
    if not text or not text.strip():
        return []

    sections = _split_by_headers(text)
    all_chunks = []
    idx = 0

    for header, content in sections:
        section_text = f"{header}\n{content}".strip() if header else content

        if _estimate_tokens(section_text) <= chunk_size_tokens:
            # Section fits in one chunk
            all_chunks.append(Chunk(
                text=section_text,
                chunk_index=idx,
                total_chunks=0,
                token_count=_estimate_tokens(section_text),
                metadata={"header": header} if header else {},
            ))
            idx += 1
        else:
            # Section too large, sub-chunk
            sub_chunks = chunk_fixed_size(section_text, chunk_size_tokens, overlap_pct)
            for sc in sub_chunks:
                sc.chunk_index = idx
                sc.metadata = {"header": header} if header else {}
                all_chunks.append(sc)
                idx += 1

    # Set total_chunks
    for chunk in all_chunks:
        chunk.total_chunks = len(all_chunks)

    logger.debug(f"Markdown-chunked text into {len(all_chunks)} chunks")
    return all_chunks


def chunk_text(
    text: str,
    strategy: str = "fixed",
    chunk_size_tokens: int = 512,
    overlap_pct: float = 0.20,
) -> List[Chunk]:
    """
    Chunk text using the specified strategy.

    Args:
        text: Input text
        strategy: One of "fixed", "semantic", "markdown"
        chunk_size_tokens: Target tokens per chunk
        overlap_pct: Overlap fraction between chunks

    Returns:
        List of Chunk objects
    """
    strategies = {
        "fixed": chunk_fixed_size,
        "semantic": chunk_semantic,
        "markdown": chunk_markdown,
    }

    fn = strategies.get(strategy)
    if fn is None:
        raise ValueError(f"Unknown chunking strategy: {strategy}. Use one of: {list(strategies.keys())}")

    return fn(text, chunk_size_tokens, overlap_pct)
