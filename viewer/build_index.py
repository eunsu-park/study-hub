#!/usr/bin/env python3
"""Build the full-text search index for the Study Viewer."""

from pathlib import Path
from utils.search import build_search_index, create_fts_table

# Paths
BASE_DIR = Path(__file__).parent
CONTENT_DIR = BASE_DIR.parent / "content"
DB_PATH = BASE_DIR / "data.db"

if __name__ == "__main__":
    print(f"Building search index...")
    print(f"  Content directory: {CONTENT_DIR}")
    print(f"  Database: {DB_PATH}")

    # Create FTS table and build index
    create_fts_table(DB_PATH)
    build_search_index(CONTENT_DIR, DB_PATH)

    print("Done!")
