#!/usr/bin/env python3
"""Example: Pagination Patterns

Demonstrates three common API pagination strategies:
1. Offset-based pagination (simple, skippable, but slow on large datasets)
2. Cursor-based pagination (consistent, efficient, forward/backward)
3. Keyset pagination (high performance, database-friendly)

Related lesson: 05_Pagination_and_Filtering.md

Run:
    pip install "fastapi[standard]"
    uvicorn 02_pagination_patterns:app --reload --port 8000
"""

import base64
import json
from datetime import datetime, timezone, timedelta
from typing import Optional
from uuid import uuid4

from fastapi import FastAPI, Query, Request
from pydantic import BaseModel, Field

app = FastAPI(title="Pagination Patterns API", version="1.0.0")


# =============================================================================
# SAMPLE DATA — 100 articles for demonstration
# =============================================================================

articles_db: list[dict] = []
base_time = datetime(2025, 1, 1, tzinfo=timezone.utc)
for i in range(1, 101):
    articles_db.append({
        "id": i,
        "title": f"Article {i}: Understanding API Pagination",
        "author": f"Author {(i % 5) + 1}",
        "published_at": (base_time + timedelta(hours=i)).isoformat(),
        "views": i * 17 % 200,
    })


# =============================================================================
# SCHEMAS
# =============================================================================

class Article(BaseModel):
    id: int
    title: str
    author: str
    published_at: str
    views: int


class PaginationMeta(BaseModel):
    """Standard pagination metadata — always inform the client about the dataset."""
    total: Optional[int] = None
    page: Optional[int] = None
    per_page: int
    has_next: bool
    has_prev: bool


class PaginationLinks(BaseModel):
    """Navigation links following HATEOAS principles."""
    self_link: str = Field(alias="self")
    next: Optional[str] = None
    prev: Optional[str] = None
    first: Optional[str] = None
    last: Optional[str] = None

    model_config = {"populate_by_name": True}


class PaginatedResponse(BaseModel):
    data: list[Article]
    meta: PaginationMeta
    links: PaginationLinks


# =============================================================================
# 1. OFFSET-BASED PAGINATION
# =============================================================================
# Pros: Simple to implement, supports jumping to any page, easy to understand.
# Cons: Inconsistent results when data changes (inserts/deletes shift offsets),
#        performance degrades on large datasets (DB must skip N rows).
#
# Best for: Small datasets, admin dashboards, static content.

@app.get("/api/v1/articles/offset", response_model=PaginatedResponse, tags=["Offset"])
def list_articles_offset(
    request: Request,
    page: int = Query(1, ge=1, description="Page number (1-indexed)"),
    per_page: int = Query(10, ge=1, le=100, description="Items per page"),
):
    """Offset pagination — the most common (and simplest) pattern.

    Uses `page` and `per_page` query parameters. The server calculates
    the offset as `(page - 1) * per_page`.

    Example: GET /api/v1/articles/offset?page=3&per_page=10
    """
    total = len(articles_db)
    offset = (page - 1) * per_page
    items = articles_db[offset : offset + per_page]

    total_pages = (total + per_page - 1) // per_page
    base = str(request.base_url).rstrip("/")
    path = "/api/v1/articles/offset"

    return PaginatedResponse(
        data=[Article(**a) for a in items],
        meta=PaginationMeta(
            total=total,
            page=page,
            per_page=per_page,
            has_next=page < total_pages,
            has_prev=page > 1,
        ),
        links=PaginationLinks(
            **{
                "self": f"{base}{path}?page={page}&per_page={per_page}",
                "next": f"{base}{path}?page={page + 1}&per_page={per_page}" if page < total_pages else None,
                "prev": f"{base}{path}?page={page - 1}&per_page={per_page}" if page > 1 else None,
                "first": f"{base}{path}?page=1&per_page={per_page}",
                "last": f"{base}{path}?page={total_pages}&per_page={per_page}",
            }
        ),
    )


# =============================================================================
# 2. CURSOR-BASED PAGINATION
# =============================================================================
# Pros: Consistent results even when data changes, efficient for feeds/timelines.
# Cons: Cannot jump to arbitrary page, cursor is opaque to clients.
#
# Best for: Social feeds, event logs, real-time data, mobile infinite scroll.
# Used by: Stripe, Slack, GitHub (GraphQL), Facebook.

def encode_cursor(article_id: int) -> str:
    """Encode cursor as base64 JSON. Opaque to the client."""
    payload = json.dumps({"id": article_id})
    return base64.urlsafe_b64encode(payload.encode()).decode()


def decode_cursor(cursor: str) -> int:
    """Decode an opaque cursor back to the article ID."""
    payload = base64.urlsafe_b64decode(cursor.encode()).decode()
    return json.loads(payload)["id"]


@app.get("/api/v1/articles/cursor", response_model=PaginatedResponse, tags=["Cursor"])
def list_articles_cursor(
    request: Request,
    cursor: Optional[str] = Query(None, description="Opaque cursor from previous response"),
    limit: int = Query(10, ge=1, le=100, description="Items to return"),
):
    """Cursor-based pagination — stable ordering even with real-time inserts.

    The cursor is an opaque string that the client passes back verbatim.
    Internally it encodes the position, but clients must not parse or
    construct cursors themselves.

    First request:  GET /api/v1/articles/cursor?limit=10
    Next request:   GET /api/v1/articles/cursor?cursor=<next_cursor>&limit=10
    """
    # Find starting position from cursor
    start_index = 0
    if cursor:
        target_id = decode_cursor(cursor)
        for idx, article in enumerate(articles_db):
            if article["id"] == target_id:
                start_index = idx + 1  # Start AFTER the cursor position
                break

    items = articles_db[start_index : start_index + limit]
    has_next = start_index + limit < len(articles_db)
    has_prev = start_index > 0

    base = str(request.base_url).rstrip("/")
    path = "/api/v1/articles/cursor"

    next_cursor = encode_cursor(items[-1]["id"]) if items and has_next else None
    # For prev cursor, we encode the first item of current page
    prev_cursor = encode_cursor(articles_db[max(0, start_index - 1)]["id"]) if has_prev else None

    return PaginatedResponse(
        data=[Article(**a) for a in items],
        meta=PaginationMeta(
            per_page=limit,
            has_next=has_next,
            has_prev=has_prev,
        ),
        links=PaginationLinks(
            **{
                "self": f"{base}{path}?limit={limit}" + (f"&cursor={cursor}" if cursor else ""),
                "next": f"{base}{path}?cursor={next_cursor}&limit={limit}" if next_cursor else None,
                "prev": None,  # Prev in cursor pagination requires bidirectional cursors
                "first": f"{base}{path}?limit={limit}",
                "last": None,  # Cursor pagination typically does not support last
            }
        ),
    )


# =============================================================================
# 3. KEYSET PAGINATION (aka Seek Method)
# =============================================================================
# Pros: Best database performance (uses indexed WHERE clause, no OFFSET),
#        consistent results, works well with billions of rows.
# Cons: Requires a unique, sequential column; cannot jump to arbitrary page.
#
# Best for: Large datasets, database-backed APIs, high-traffic endpoints.
# SQL pattern: SELECT * FROM articles WHERE id > :last_id ORDER BY id LIMIT :n

@app.get("/api/v1/articles/keyset", response_model=PaginatedResponse, tags=["Keyset"])
def list_articles_keyset(
    request: Request,
    after_id: Optional[int] = Query(None, description="Return items after this ID"),
    limit: int = Query(10, ge=1, le=100, description="Items to return"),
    sort_by: str = Query("id", description="Sort field (id or views)"),
    order: str = Query("asc", pattern="^(asc|desc)$", description="Sort order"),
):
    """Keyset pagination — the most performant approach for large datasets.

    Instead of OFFSET (which requires scanning and discarding rows), keyset
    pagination uses a WHERE clause on an indexed column:

        SELECT * FROM articles WHERE id > 42 ORDER BY id ASC LIMIT 10

    This is O(limit) regardless of how deep into the dataset you are.

    Example: GET /api/v1/articles/keyset?after_id=42&limit=10&sort_by=id&order=asc
    """
    # Sort data
    reverse = order == "desc"
    sorted_articles = sorted(articles_db, key=lambda a: a[sort_by], reverse=reverse)

    # Apply keyset filter
    if after_id is not None:
        if not reverse:
            sorted_articles = [a for a in sorted_articles if a["id"] > after_id]
        else:
            sorted_articles = [a for a in sorted_articles if a["id"] < after_id]

    items = sorted_articles[:limit]
    has_next = len(sorted_articles) > limit

    base = str(request.base_url).rstrip("/")
    path = "/api/v1/articles/keyset"

    last_id = items[-1]["id"] if items else None
    next_link = (
        f"{base}{path}?after_id={last_id}&limit={limit}&sort_by={sort_by}&order={order}"
        if has_next and last_id
        else None
    )

    return PaginatedResponse(
        data=[Article(**a) for a in items],
        meta=PaginationMeta(
            per_page=limit,
            has_next=has_next,
            has_prev=after_id is not None,
        ),
        links=PaginationLinks(
            **{
                "self": f"{base}{path}?limit={limit}&sort_by={sort_by}&order={order}"
                + (f"&after_id={after_id}" if after_id else ""),
                "next": next_link,
                "prev": None,
                "first": f"{base}{path}?limit={limit}&sort_by={sort_by}&order={order}",
                "last": None,
            }
        ),
    )


# =============================================================================
# COMPARISON ENDPOINT — Side-by-side summary
# =============================================================================

@app.get("/api/v1/articles/comparison", tags=["Info"])
def pagination_comparison():
    """Return a summary of when to use each pagination strategy."""
    return {
        "offset": {
            "endpoint": "/api/v1/articles/offset",
            "pros": ["Simple", "Jump to any page", "Easy total count"],
            "cons": ["Inconsistent with mutations", "Slow on large datasets"],
            "use_when": "Small datasets, admin panels, static content",
        },
        "cursor": {
            "endpoint": "/api/v1/articles/cursor",
            "pros": ["Stable ordering", "Efficient", "Works with real-time data"],
            "cons": ["No page jumping", "Opaque cursors"],
            "use_when": "Social feeds, timelines, event logs, mobile apps",
        },
        "keyset": {
            "endpoint": "/api/v1/articles/keyset",
            "pros": ["Best DB performance", "Consistent", "Scales to billions"],
            "cons": ["Needs sequential column", "No page jumping"],
            "use_when": "Large datasets, high-traffic APIs, analytics",
        },
    }


# =============================================================================
# ENTRYPOINT
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("02_pagination_patterns:app", host="127.0.0.1", port=8000, reload=True)
