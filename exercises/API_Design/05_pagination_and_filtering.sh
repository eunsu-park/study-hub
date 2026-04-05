#!/bin/bash
# Exercises for Lesson 05: Pagination and Filtering
# Topic: API_Design
# Solutions to practice problems from the lesson.

# === Exercise 1: Implement Cursor Pagination ===
# Problem: Build a cursor-based pagination endpoint for a timeline feed.
# The cursor should be opaque and encode the position.
exercise_1() {
    echo "=== Exercise 1: Implement Cursor Pagination ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
import base64
import json
from datetime import datetime, timezone, timedelta
from typing import Optional

from fastapi import FastAPI, Query
from pydantic import BaseModel

app = FastAPI()

# Simulated timeline entries
timeline = [
    {"id": i, "text": f"Post #{i}", "created_at": (
        datetime(2025, 6, 1, tzinfo=timezone.utc) + timedelta(minutes=i)
    ).isoformat()}
    for i in range(1, 201)  # 200 posts
]


def encode_cursor(post_id: int, created_at: str) -> str:
    """Encode position as base64 JSON — opaque to the client."""
    return base64.urlsafe_b64encode(
        json.dumps({"id": post_id, "ts": created_at}).encode()
    ).decode()


def decode_cursor(cursor: str) -> dict:
    """Decode cursor back to position data."""
    return json.loads(base64.urlsafe_b64decode(cursor.encode()).decode())


class CursorPage(BaseModel):
    data: list[dict]
    cursors: dict  # {"before": "...", "after": "..."}
    has_next: bool
    has_prev: bool


@app.get("/api/v1/timeline", response_model=CursorPage)
def get_timeline(
    after: Optional[str] = Query(None, description="Cursor: return items after this"),
    before: Optional[str] = Query(None, description="Cursor: return items before this"),
    limit: int = Query(20, ge=1, le=100),
):
    """Cursor-based pagination for a timeline feed.

    - First page:   GET /api/v1/timeline?limit=20
    - Next page:    GET /api/v1/timeline?after=<cursor>&limit=20
    - Prev page:    GET /api/v1/timeline?before=<cursor>&limit=20
    """
    items = timeline  # Sorted by creation time

    if after:
        pos = decode_cursor(after)
        items = [p for p in items if p["id"] > pos["id"]]
    elif before:
        pos = decode_cursor(before)
        items = [p for p in items if p["id"] < pos["id"]]
        items = items[-limit:]  # Take last N for "before" direction

    page = items[:limit]
    has_next = len(items) > limit
    has_prev = page[0]["id"] > timeline[0]["id"] if page else False

    cursors = {}
    if page:
        cursors["after"] = encode_cursor(page[-1]["id"], page[-1]["created_at"])
        cursors["before"] = encode_cursor(page[0]["id"], page[0]["created_at"])

    return CursorPage(
        data=page,
        cursors=cursors,
        has_next=has_next,
        has_prev=has_prev,
    )
SOLUTION
}

# === Exercise 2: Filtering with Multiple Criteria ===
# Problem: Build a filtering system that supports exact match, range,
# and multi-value filters.
exercise_2() {
    echo "=== Exercise 2: Filtering with Multiple Criteria ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
from fastapi import FastAPI, Query
from typing import Optional

app = FastAPI()

products = [
    {"id": 1, "name": "Laptop", "price": 999, "brand": "Dell", "rating": 4.5},
    {"id": 2, "name": "Mouse", "price": 29, "brand": "Logitech", "rating": 4.2},
    {"id": 3, "name": "Keyboard", "price": 79, "brand": "Logitech", "rating": 4.8},
    {"id": 4, "name": "Monitor", "price": 349, "brand": "Dell", "rating": 4.0},
    {"id": 5, "name": "Webcam", "price": 59, "brand": "Logitech", "rating": 3.9},
]


@app.get("/api/v1/products")
def list_products(
    # Exact match filter
    brand: Optional[str] = Query(None, description="Exact brand match"),

    # Multi-value filter (comma-separated)
    brands: Optional[str] = Query(
        None,
        description="Multiple brands: brands=Dell,Logitech",
    ),

    # Range filters
    min_price: Optional[float] = Query(None, ge=0),
    max_price: Optional[float] = Query(None, ge=0),
    min_rating: Optional[float] = Query(None, ge=0, le=5),

    # Sorting
    sort: str = Query("name", description="Sort field: name, price, rating, -price"),

    # Pagination
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
):
    """Filter products with exact, range, and multi-value criteria."""
    result = products.copy()

    # Exact match
    if brand:
        result = [p for p in result if p["brand"].lower() == brand.lower()]

    # Multi-value filter
    if brands:
        brand_list = [b.strip().lower() for b in brands.split(",")]
        result = [p for p in result if p["brand"].lower() in brand_list]

    # Range filters
    if min_price is not None:
        result = [p for p in result if p["price"] >= min_price]
    if max_price is not None:
        result = [p for p in result if p["price"] <= max_price]
    if min_rating is not None:
        result = [p for p in result if p["rating"] >= min_rating]

    # Sorting
    desc = sort.startswith("-")
    sort_key = sort.lstrip("-")
    if sort_key in ("name", "price", "rating"):
        result.sort(key=lambda p: p[sort_key], reverse=desc)

    # Pagination
    total = len(result)
    offset = (page - 1) * per_page
    result = result[offset:offset + per_page]

    return {
        "data": result,
        "meta": {"total": total, "page": page, "per_page": per_page},
    }

# Examples:
# GET /api/v1/products?brand=Dell
# GET /api/v1/products?brands=Dell,Logitech&min_price=50&max_price=500
# GET /api/v1/products?min_rating=4.0&sort=-price
SOLUTION
}

# === Exercise 3: Sparse Fieldsets ===
# Problem: Implement sparse fieldsets that let clients request only
# specific fields, reducing bandwidth usage.
exercise_3() {
    echo "=== Exercise 3: Sparse Fieldsets ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
from fastapi import FastAPI, Query
from typing import Optional

app = FastAPI()

users = [
    {
        "id": "1",
        "name": "Alice",
        "email": "alice@example.com",
        "phone": "+1-555-0101",
        "address": "123 Main St",
        "department": "Engineering",
        "manager_id": "10",
        "created_at": "2025-01-15T10:00:00Z",
    },
]


def apply_sparse_fields(data: dict, fields: Optional[str]) -> dict:
    """Filter response to include only requested fields.

    The `fields` parameter accepts a comma-separated list of field names.
    The `id` field is always included (clients need it for subsequent requests).
    """
    if not fields:
        return data  # No filter — return all fields

    requested = set(f.strip() for f in fields.split(","))
    requested.add("id")  # Always include ID

    return {k: v for k, v in data.items() if k in requested}


@app.get("/api/v1/users")
def list_users(
    fields: Optional[str] = Query(
        None,
        description="Comma-separated fields to include: fields=id,name,email",
        examples=["id,name,email"],
    ),
):
    """List users with optional sparse fieldsets.

    Without fields: returns all fields (full representation)
    With fields=id,name,email: returns only those fields

    Benefits:
    - Reduces payload size (important for mobile clients)
    - Reduces database query scope (if backend optimizes)
    - Client requests only what it displays
    """
    result = [apply_sparse_fields(u, fields) for u in users]
    return {"data": result}


# Full response:
# GET /api/v1/users
# {"data": [{"id":"1","name":"Alice","email":"alice@...", ... all 8 fields}]}

# Sparse response:
# GET /api/v1/users?fields=id,name,email
# {"data": [{"id":"1","name":"Alice","email":"alice@example.com"}]}
SOLUTION
}

# === Exercise 4: Keyset Pagination with Composite Key ===
# Problem: Implement keyset pagination that works with a composite sort key
# (created_at + id) to handle duplicate timestamps.
exercise_4() {
    echo "=== Exercise 4: Keyset Pagination with Composite Key ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
from fastapi import FastAPI, Query
from typing import Optional

app = FastAPI()

# Events with potentially duplicate timestamps
events = [
    {"id": "a1", "type": "click", "created_at": "2025-06-01T10:00:00Z"},
    {"id": "a2", "type": "view",  "created_at": "2025-06-01T10:00:00Z"},  # Same timestamp!
    {"id": "a3", "type": "click", "created_at": "2025-06-01T10:00:00Z"},  # Same timestamp!
    {"id": "a4", "type": "purchase", "created_at": "2025-06-01T10:00:01Z"},
    {"id": "a5", "type": "view",  "created_at": "2025-06-01T10:00:02Z"},
]


@app.get("/api/v1/events")
def list_events(
    after_time: Optional[str] = Query(None, description="Keyset: created_at value"),
    after_id: Optional[str] = Query(None, description="Keyset: id tiebreaker"),
    limit: int = Query(2, ge=1, le=100),
):
    """Keyset pagination with composite key (created_at, id).

    Problem: Simple keyset on created_at fails when multiple events
    share the same timestamp — some events get skipped or duplicated.

    Solution: Use (created_at, id) as a composite keyset. The id
    serves as a tiebreaker when timestamps are equal.

    SQL equivalent:
        SELECT * FROM events
        WHERE (created_at, id) > (:after_time, :after_id)
        ORDER BY created_at ASC, id ASC
        LIMIT :limit
    """
    result = events.copy()

    if after_time and after_id:
        result = [
            e for e in result
            if (e["created_at"], e["id"]) > (after_time, after_id)
        ]

    page = result[:limit]
    has_more = len(result) > limit

    next_params = {}
    if page and has_more:
        last = page[-1]
        next_params = {
            "after_time": last["created_at"],
            "after_id": last["id"],
        }

    return {
        "data": page,
        "has_more": has_more,
        "next": next_params,
    }

# Page 1: GET /api/v1/events?limit=2
# → [a1, a2], next: {after_time: "...10:00:00Z", after_id: "a2"}

# Page 2: GET /api/v1/events?limit=2&after_time=2025-06-01T10:00:00Z&after_id=a2
# → [a3, a4], next: {after_time: "...10:00:01Z", after_id: "a4"}
# Notice: a3 has the same timestamp as a2, but is correctly included!
SOLUTION
}

# Run all exercises
echo "Exercise solutions for Lesson 05: Pagination and Filtering"
echo "============================================================"
exercise_1
echo ""
exercise_2
echo ""
exercise_3
echo ""
exercise_4
