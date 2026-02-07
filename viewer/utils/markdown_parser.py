"""Markdown parsing utilities with syntax highlighting."""
import re
import markdown
from markdown.extensions.codehilite import CodeHiliteExtension
from markdown.extensions.fenced_code import FencedCodeExtension
from markdown.extensions.tables import TableExtension
from markdown.extensions.toc import TocExtension


def parse_markdown(content: str) -> dict:
    """
    Parse Markdown content to HTML with syntax highlighting.

    Returns:
        dict: {
            "html": rendered HTML,
            "toc": table of contents HTML,
            "title": extracted title (first H1)
        }
    """
    md = markdown.Markdown(
        extensions=[
            "fenced_code",
            "tables",
            "toc",
            CodeHiliteExtension(
                css_class="highlight",
                linenums=False,
                guess_lang=True,
            ),
            TableExtension(),
            TocExtension(
                permalink=True,
                permalink_class="header-link",
                toc_depth="2-4",
            ),
        ]
    )

    html = md.convert(content)
    toc = md.toc

    # Extract title from first H1
    title = extract_title(content)

    return {
        "html": html,
        "toc": toc,
        "title": title,
    }


def extract_title(content: str) -> str:
    """Extract the first H1 heading from Markdown content."""
    # Match # heading at the start of a line
    match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
    if match:
        return match.group(1).strip()
    return ""


def extract_excerpt(content: str, max_length: int = 200) -> str:
    """Extract a plain text excerpt from Markdown content."""
    # Remove code blocks
    text = re.sub(r"```[\s\S]*?```", "", content)
    # Remove inline code
    text = re.sub(r"`[^`]+`", "", text)
    # Remove headers
    text = re.sub(r"^#+\s+.+$", "", text, flags=re.MULTILINE)
    # Remove links but keep text
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    # Remove images
    text = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", text)
    # Remove bold/italic markers
    text = re.sub(r"[*_]{1,2}([^*_]+)[*_]{1,2}", r"\1", text)
    # Remove extra whitespace
    text = " ".join(text.split())

    if len(text) > max_length:
        return text[:max_length].rsplit(" ", 1)[0] + "..."
    return text
