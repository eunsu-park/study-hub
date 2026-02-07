"""Full-text search utilities using SQLite FTS5 with language support."""
import sqlite3
from pathlib import Path
from typing import Optional

from .markdown_parser import extract_title, extract_excerpt


def create_fts_table(db_path: Path):
    """Create FTS5 virtual table for full-text search."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Drop existing table to recreate with language column
    cursor.execute("DROP TABLE IF EXISTS search_fts")

    # Create FTS5 virtual table with language support
    cursor.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS search_fts USING fts5(
            language,
            topic,
            filename,
            title,
            content,
            tokenize='unicode61'
        )
    """)

    conn.commit()
    conn.close()


def build_search_index(content_dir: Path, db_path: Path, lang: str = "ko"):
    """Build or rebuild the search index for a specific language."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Delete existing entries for this language
    cursor.execute("DELETE FROM search_fts WHERE language = ?", (lang,))

    # Index all markdown files for this language
    for topic_dir in content_dir.iterdir():
        if not topic_dir.is_dir() or topic_dir.name.startswith("."):
            continue

        topic = topic_dir.name
        for md_file in topic_dir.glob("*.md"):
            try:
                content = md_file.read_text(encoding="utf-8")
                title = extract_title(content) or md_file.stem
                cursor.execute(
                    "INSERT INTO search_fts (language, topic, filename, title, content) VALUES (?, ?, ?, ?, ?)",
                    (lang, topic, md_file.name, title, content),
                )
            except Exception as e:
                print(f"Error indexing {md_file}: {e}")

    conn.commit()
    conn.close()
    print(f"Search index built successfully for {lang}.")


def search(db_path: Path, query: str, lang: str = "ko", limit: int = 50) -> list[dict]:
    """
    Search the index for matching documents in a specific language.

    Args:
        db_path: Path to the SQLite database
        query: Search query string
        lang: Language code to search in
        limit: Maximum number of results

    Returns:
        list of dicts with topic, filename, title, and snippet
    """
    if not query.strip():
        return []

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Escape special FTS5 characters
    safe_query = query.replace('"', '""')

    try:
        # Search with language filter
        cursor.execute(
            """
            SELECT topic, filename, title, snippet(search_fts, 4, '<mark>', '</mark>', '...', 50) as snippet
            FROM search_fts
            WHERE search_fts MATCH ? AND language = ?
            ORDER BY rank
            LIMIT ?
            """,
            (f'"{safe_query}"', lang, limit),
        )
        results = cursor.fetchall()
    except sqlite3.OperationalError:
        # Table might not exist yet
        return []
    finally:
        conn.close()

    return [
        {
            "topic": row[0],
            "filename": row[1],
            "title": row[2],
            "snippet": row[3],
        }
        for row in results
    ]
