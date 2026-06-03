"""
HTML Cleaner — strips boilerplate from HTML content.
Produces clean text suitable for chunking and embedding.

Strategy:
1. Parse with BeautifulSoup
2. Remove nav, footer, script, style, and ad tags
3. Extract main content via <main>, <article>, or largest <div>
4. Convert to markdown-like clean text
"""

import re
import logging
from typing import Optional

from bs4 import BeautifulSoup, Tag

logger = logging.getLogger(__name__)

# Tags to completely remove (including content)
REMOVE_TAGS = {
    "script", "style", "nav", "footer", "header",
    "aside", "iframe", "noscript", "svg", "form",
    "button", "input", "select", "textarea",
}

# Tags that are navigation/chrome (remove tag but keep text if substantive)
CHROME_TAGS = {
    "nav", "header", "footer", "aside",
}

# Attributes that indicate ads or tracking
AD_CLASS_PATTERNS = [
    re.compile(r"ad[-_]?", re.I),
    re.compile(r"sidebar", re.I),
    re.compile(r"cookie", re.I),
    re.compile(r"popup|modal|overlay", re.I),
    re.compile(r"social|share|comment", re.I),
    re.compile(r"related|recommend", re.I),
    re.compile(r"newsletter|subscribe", re.I),
]


def _is_ad_element(tag: Tag) -> bool:
    """Check if a tag looks like an ad or boilerplate element."""
    if not hasattr(tag, 'attrs') or tag.attrs is None:
        return False
    classes = " ".join(tag.get("class", []))
    tag_id = tag.get("id", "")
    combined = f"{classes} {tag_id}"

    for pattern in AD_CLASS_PATTERNS:
        if pattern.search(combined):
            return True
    return False


def _find_main_content(soup: BeautifulSoup) -> Tag:
    """Find the main content container using semantic HTML or heuristics."""
    # 1. Try semantic main tag
    main = soup.find("main")
    if main:
        return main

    # 2. Try article tag
    article = soup.find("article")
    if article:
        return article

    # 3. Try role="main"
    role_main = soup.find(attrs={"role": "main"})
    if role_main:
        return role_main

    # 4. Heuristic: find the div with the most text content
    best_div = soup.find("body") or soup
    best_len = 0

    for div in best_div.find_all("div"):
        if _is_ad_element(div):
            continue
        text_len = len(div.get_text(strip=True))
        if text_len > best_len:
            best_len = text_len
            best_div = div

    return best_div


def _tag_to_text(tag: Tag) -> str:
    """Convert a BeautifulSoup tag to clean text with markdown-like formatting."""
    lines = []

    for element in tag.descendants:
        if not isinstance(element, Tag):
            if isinstance(element, str) and element.strip():
                lines.append(element.strip())
            continue

        name = element.name

        if name in ("h1", "h2", "h3", "h4", "h5", "h6"):
            level = int(name[1])
            text = element.get_text(strip=True)
            if text:
                lines.append(f"\n{'#' * level} {text}\n")

        elif name == "p":
            text = element.get_text(strip=True)
            if text:
                lines.append(f"\n{text}\n")

        elif name == "li":
            text = element.get_text(strip=True)
            if text:
                lines.append(f"- {text}")

        elif name in ("strong", "b"):
            text = element.get_text(strip=True)
            if text:
                lines.append(f"**{text}**")

        elif name in ("em", "i"):
            text = element.get_text(strip=True)
            if text:
                lines.append(f"*{text}*")

        elif name == "code":
            text = element.get_text(strip=True)
            if text:
                parent = element.parent
                if parent and parent.name == "pre":
                    lines.append(f"\n```\n{text}\n```\n")
                else:
                    lines.append(f"`{text}`")

        elif name == "a":
            text = element.get_text(strip=True)
            href = element.get("href", "")
            if text and href:
                lines.append(f"[{text}]({href})")

        elif name == "br":
            lines.append("\n")

        elif name in ("table",):
            # Simple table extraction
            rows = []
            for tr in element.find_all("tr"):
                cells = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
                if cells:
                    rows.append(" | ".join(cells))
            if rows:
                lines.append("\n" + "\n".join(rows) + "\n")

    return "\n".join(lines)


def clean_html(html: str, url: Optional[str] = None) -> str:
    """
    Clean HTML content, removing boilerplate and extracting main content.

    Args:
        html: Raw HTML string
        url: Source URL for logging purposes

    Returns:
        Clean text with markdown-like formatting
    """
    if not html or not html.strip():
        return ""

    try:
        soup = BeautifulSoup(html, "html.parser")
    except Exception as e:
        logger.warning(f"Failed to parse HTML from {url}: {e}")
        return html

    # Remove unwanted tags entirely
    for tag_name in REMOVE_TAGS:
        for tag in soup.find_all(tag_name):
            tag.decompose()

    # Remove ad-like elements
    for tag in soup.find_all(True):
        if _is_ad_element(tag):
            tag.decompose()

    # Find main content
    main_content = _find_main_content(soup)

    # Convert to clean text
    clean_text = _tag_to_text(main_content)

    # Post-processing: collapse excessive whitespace
    clean_text = re.sub(r"\n{3,}", "\n\n", clean_text)
    clean_text = re.sub(r" {2,}", " ", clean_text)
    clean_text = clean_text.strip()

    logger.debug(f"Cleaned HTML from {url}: {len(html)} -> {len(clean_text)} chars")
    return clean_text


def clean_markdown(md: str) -> str:
    """
    Clean markdown content from scrapers, removing residual boilerplate.

    Args:
        md: Raw markdown string

    Returns:
        Cleaned markdown
    """
    if not md or not md.strip():
        return ""

    # Remove common boilerplate patterns
    patterns_to_remove = [
        re.compile(r"Cookie\s+Policy.*?\n", re.I | re.S),
        re.compile(r"Accept\s+all\s+cookies.*?\n", re.I | re.S),
        re.compile(r"Sign\s+up\s+for.*?newsletter.*?\n", re.I | re.S),
        re.compile(r"Subscribe\s+to.*?\n", re.I | re.S),
        re.compile(r"Share\s+this\s+article.*?\n", re.I | re.S),
        re.compile(r"Related\s+articles?:.*?\n", re.I | re.S),
        re.compile(r"Comments?\s*\(\d+\).*?\n", re.I),
    ]

    for pattern in patterns_to_remove:
        md = pattern.sub("", md)

    # Collapse whitespace
    md = re.sub(r"\n{3,}", "\n\n", md)
    md = md.strip()

    return md
