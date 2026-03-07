# Lesson 5: Pagination and Filtering

**Previous**: [Request and Response Design](04_Request_and_Response_Design.md) | [Overview](00_Overview.md) | **Next**: [Authentication and Authorization](06_Authentication_and_Authorization.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Compare offset-based, cursor-based, and keyset pagination and choose the right strategy
2. Implement pagination with HATEOAS navigation links
3. Design filtering syntax for equality, range, and multi-value queries
4. Add sorting with multi-field support and direction control
5. Implement field selection (sparse fieldsets) to reduce response size
6. Handle edge cases: empty results, invalid parameters, and large datasets

---

Any collection endpoint that could return more than a handful of items needs pagination. Without it, a simple `GET /api/users` could try to return millions of records, crashing the server or overwhelming the client. Pagination, filtering, sorting, and field selection work together to give consumers precise control over what data they receive and how much of it.

> **Analogy:** Pagination is like a library catalog. You do not dump every book on the floor and search through them -- you browse page by page, filter by genre, sort by author, and only check out the ones you need. A well-designed API gives consumers the same level of control.

## Table of Contents
1. [Offset Pagination](#offset-pagination)
2. [Cursor Pagination](#cursor-pagination)
3. [Keyset Pagination](#keyset-pagination)
4. [Pagination Comparison](#pagination-comparison)
5. [HATEOAS Pagination Links](#hateoas-pagination-links)
6. [Filtering](#filtering)
7. [Sorting](#sorting)
8. [Field Selection](#field-selection)
9. [Exercises](#exercises)

---

## Offset Pagination

Offset pagination uses `page` and `per_page` (or `offset` and `limit`) parameters to slice the result set.

### Basic Implementation

```python
from fastapi import FastAPI, Query
from pydantic import BaseModel

app = FastAPI()

# Simulated database
ALL_PRODUCTS = [
    {"id": i, "name": f"Product {i}", "price": round(9.99 + i * 0.5, 2), "category": "electronics"}
    for i in range(1, 201)
]

@app.get("/api/products")
async def list_products(
    page: int = Query(default=1, ge=1, description="Page number (1-indexed)"),
    per_page: int = Query(default=20, ge=1, le=100, description="Items per page"),
):
    """
    Offset pagination with page/per_page.

    Example: GET /api/products?page=3&per_page=10
    Returns items 21-30 of the total set.
    """
    total = len(ALL_PRODUCTS)
    total_pages = (total + per_page - 1) // per_page

    # Clamp page to valid range
    if page > total_pages and total > 0:
        page = total_pages

    start = (page - 1) * per_page
    end = start + per_page
    items = ALL_PRODUCTS[start:end]

    return {
        "data": items,
        "meta": {
            "page": page,
            "per_page": per_page,
            "total": total,
            "total_pages": total_pages,
        }
    }
```

### Offset/Limit Variant

```python
@app.get("/api/logs")
async def list_logs(
    offset: int = Query(default=0, ge=0, description="Number of items to skip"),
    limit: int = Query(default=50, ge=1, le=200, description="Max items to return"),
):
    """
    Offset/limit pagination (0-indexed).

    Example: GET /api/logs?offset=100&limit=50
    Returns items 101-150.
    """
    total = 5000
    # In practice: SELECT * FROM logs ORDER BY created_at DESC OFFSET {offset} LIMIT {limit}
    items = [{"id": i, "message": f"Log entry {i}"} for i in range(offset + 1, min(offset + limit + 1, total + 1))]

    return {
        "data": items,
        "meta": {
            "offset": offset,
            "limit": limit,
            "total": total,
            "has_more": offset + limit < total,
        }
    }
```

### Offset Pagination Drawbacks

```
Problem 1: Page drift (data inconsistency)
─────────────────────────────────────────
Request 1: GET /api/users?page=1&per_page=10  (gets users 1-10)
            -- Someone deletes user 5 --
Request 2: GET /api/users?page=2&per_page=10  (gets users 11-20)
            But user 11 shifted to position 10, so it was already in page 1!
            Result: user 11 is skipped entirely.

Problem 2: Performance degradation
─────────────────────────────────
Page 1:    SELECT * FROM users LIMIT 10 OFFSET 0      -- fast
Page 100:  SELECT * FROM users LIMIT 10 OFFSET 990    -- still OK
Page 10000: SELECT * FROM users LIMIT 10 OFFSET 99990 -- SLOW!
            The database must scan and discard 99,990 rows.
```

---

## Cursor Pagination

Cursor pagination uses an opaque token (cursor) that points to a specific position in the result set. The client sends the cursor to get the next page.

### Implementation

```python
from fastapi import FastAPI, Query
import base64
import json

app = FastAPI()

# Simulated database
ALL_USERS = [
    {"id": i, "name": f"User {i}", "created_at": f"2025-01-{i:02d}T00:00:00Z"}
    for i in range(1, 101)
]

def encode_cursor(data: dict) -> str:
    """Encode pagination state as an opaque base64 cursor."""
    return base64.urlsafe_b64encode(json.dumps(data).encode()).decode()

def decode_cursor(cursor: str) -> dict:
    """Decode an opaque cursor back to pagination state."""
    return json.loads(base64.urlsafe_b64decode(cursor.encode()).decode())

@app.get("/api/users")
async def list_users(
    limit: int = Query(default=20, ge=1, le=100),
    after: str | None = Query(default=None, description="Cursor for next page"),
    before: str | None = Query(default=None, description="Cursor for previous page"),
):
    """
    Cursor-based pagination.

    Example:
        GET /api/users?limit=10             -> first page
        GET /api/users?limit=10&after=eyJ.. -> next page (using cursor from previous response)
    """
    # Determine starting position from cursor
    start_index = 0
    if after:
        cursor_data = decode_cursor(after)
        start_index = cursor_data["index"]
    elif before:
        cursor_data = decode_cursor(before)
        start_index = max(0, cursor_data["index"] - limit)

    # Fetch one extra item to determine if there are more
    items = ALL_USERS[start_index:start_index + limit + 1]
    has_next = len(items) > limit
    items = items[:limit]  # trim the extra item
    has_prev = start_index > 0

    # Build cursors
    response = {
        "data": items,
        "meta": {
            "has_next": has_next,
            "has_prev": has_prev,
            "count": len(items),
        },
        "cursors": {}
    }

    if has_next:
        response["cursors"]["after"] = encode_cursor({"index": start_index + limit})
    if has_prev:
        response["cursors"]["before"] = encode_cursor({"index": start_index})

    return response
```

### Cursor Pagination Advantages

```
1. No page drift: Cursors point to a specific item, not a position.
   Even if items are inserted or deleted, the cursor remains valid.

2. Consistent performance: The database uses an index scan from the
   cursor position. No matter how deep you paginate, performance
   is constant (O(1) seek vs O(n) offset scan).

3. Opaque tokens: Clients cannot manipulate cursors to jump to
   arbitrary pages. This prevents abuse and simplifies the API contract.
```

---

## Keyset Pagination

Keyset pagination (also called "seek method") uses the **sort key value** of the last item on the current page to fetch the next page. It is a transparent version of cursor pagination.

### Implementation

```python
from fastapi import FastAPI, Query
from datetime import datetime

app = FastAPI()

@app.get("/api/events")
async def list_events(
    limit: int = Query(default=20, ge=1, le=100),
    after_id: int | None = Query(default=None, description="Return events after this ID"),
    after_date: str | None = Query(default=None, description="Return events after this date"),
):
    """
    Keyset pagination using the sort key directly.

    Example:
        GET /api/events?limit=20
        -> Returns first 20 events, last one has id=20, date=2025-01-20

        GET /api/events?limit=20&after_id=20&after_date=2025-01-20
        -> Returns next 20 events starting after id=20
    """
    # SQL equivalent:
    # SELECT * FROM events
    # WHERE (date, id) > (:after_date, :after_id)
    # ORDER BY date ASC, id ASC
    # LIMIT :limit + 1

    # Simulated data
    all_events = [
        {"id": i, "title": f"Event {i}", "date": f"2025-01-{i:02d}"}
        for i in range(1, 101)
    ]

    # Apply keyset filter
    if after_id is not None:
        all_events = [e for e in all_events if e["id"] > after_id]

    # Fetch limit + 1 to detect if more exist
    items = all_events[:limit + 1]
    has_more = len(items) > limit
    items = items[:limit]

    response = {
        "data": items,
        "meta": {
            "count": len(items),
            "has_more": has_more,
        }
    }

    if has_more and items:
        last = items[-1]
        response["meta"]["next_params"] = {
            "after_id": last["id"],
            "after_date": last["date"],
        }

    return response
```

### Keyset vs Cursor

```
Keyset Pagination:
  - Sort key values are visible (e.g., after_id=42)
  - Client can construct their own starting point
  - Transparent and debuggable
  - Client must know the sort fields

Cursor Pagination:
  - Opaque token (e.g., after=eyJpZCI6NDJ9)
  - Server controls the format entirely
  - More flexible (can change internal representation without breaking clients)
  - Client cannot jump to arbitrary positions

Most APIs use cursor pagination (opaque) for public APIs
and keyset pagination (transparent) for internal APIs.
```

---

## Pagination Comparison

| Feature | Offset | Cursor | Keyset |
|---------|--------|--------|--------|
| Jump to page N | Yes | No | No |
| Total count | Yes (at cost) | Optional | Optional |
| Performance at depth | Degrades (O(n)) | Constant (O(1)) | Constant (O(1)) |
| Data consistency | Page drift possible | Stable | Stable |
| Implementation | Simple | Moderate | Moderate |
| Client complexity | Low | Low | Low-Medium |
| Use case | Small datasets, admin UIs | Social feeds, timelines | Log entries, audit trails |
| Real-world examples | GitHub (repos), Shopify | Twitter, Facebook, Slack | Stripe (events), Datadog |

### When to Use Each

```python
def choose_pagination(
    dataset_size: str,      # "small" | "medium" | "large"
    needs_page_jump: bool,  # Does the UI need "go to page 42"?
    real_time_inserts: bool, # Are new items inserted frequently?
) -> str:
    if dataset_size == "small" and needs_page_jump:
        return "offset"  # simple, page jumps supported
    if real_time_inserts:
        return "cursor"  # no page drift
    if dataset_size == "large":
        return "keyset"  # constant performance
    return "cursor"  # safe default
```

---

## HATEOAS Pagination Links

Pagination links allow clients to navigate without constructing URLs themselves.

### Standard Link Relations

```python
from fastapi import FastAPI, Query

app = FastAPI()

@app.get("/api/articles")
async def list_articles(
    page: int = Query(default=1, ge=1),
    per_page: int = Query(default=20, ge=1, le=100),
):
    total = 250
    total_pages = (total + per_page - 1) // per_page

    items = [
        {"id": i, "title": f"Article {i}"}
        for i in range((page - 1) * per_page + 1, min(page * per_page, total) + 1)
    ]

    # Build HATEOAS pagination links
    base = "/api/articles"
    links = {
        "self": {"href": f"{base}?page={page}&per_page={per_page}"},
        "first": {"href": f"{base}?page=1&per_page={per_page}"},
        "last": {"href": f"{base}?page={total_pages}&per_page={per_page}"},
    }
    if page > 1:
        links["prev"] = {"href": f"{base}?page={page - 1}&per_page={per_page}"}
    if page < total_pages:
        links["next"] = {"href": f"{base}?page={page + 1}&per_page={per_page}"}

    return {
        "data": items,
        "meta": {
            "page": page,
            "per_page": per_page,
            "total": total,
            "total_pages": total_pages,
        },
        "_links": links,
    }
```

### Cursor-Based Navigation Links

```python
@app.get("/api/feed")
async def list_feed(
    limit: int = Query(default=20, ge=1, le=100),
    after: str | None = None,
):
    """Cursor pagination with HATEOAS links."""
    # ... fetch items ...

    items = [{"id": i, "text": f"Post {i}"} for i in range(1, limit + 1)]
    next_cursor = encode_cursor({"index": 20})  # from cursor helper

    return {
        "data": items,
        "meta": {
            "count": len(items),
            "has_more": True,
        },
        "_links": {
            "self": {"href": f"/api/feed?limit={limit}" + (f"&after={after}" if after else "")},
            "next": {"href": f"/api/feed?limit={limit}&after={next_cursor}"},
        }
    }
```

### Link Header (RFC 8288)

Some APIs use the HTTP `Link` header instead of or in addition to the response body:

```python
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse

app = FastAPI()

@app.get("/api/repos")
async def list_repos(
    page: int = Query(default=1, ge=1),
    per_page: int = Query(default=30, ge=1, le=100),
):
    """GitHub-style pagination using Link header."""
    total_pages = 10
    base = "/api/repos"

    # Build Link header (RFC 8288)
    links = []
    if page < total_pages:
        links.append(f'<{base}?page={page + 1}&per_page={per_page}>; rel="next"')
    if page > 1:
        links.append(f'<{base}?page={page - 1}&per_page={per_page}>; rel="prev"')
    links.append(f'<{base}?page=1&per_page={per_page}>; rel="first"')
    links.append(f'<{base}?page={total_pages}&per_page={per_page}>; rel="last"')

    return JSONResponse(
        content={"data": [{"id": 1, "name": "repo-1"}]},
        headers={"Link": ", ".join(links)},
    )
```

---

## Filtering

### Equality Filters

```python
from fastapi import FastAPI, Query

app = FastAPI()

@app.get("/api/products")
async def list_products(
    category: str | None = Query(default=None, description="Filter by category"),
    brand: str | None = Query(default=None, description="Filter by brand"),
    in_stock: bool | None = Query(default=None, description="Filter by availability"),
    color: str | None = Query(default=None, description="Filter by color (comma-separated)"),
):
    """
    Simple equality filters.

    Examples:
        GET /api/products?category=electronics
        GET /api/products?brand=acme&in_stock=true
        GET /api/products?color=red,blue,green
    """
    # Build query conditions
    conditions = {}
    if category:
        conditions["category"] = category
    if brand:
        conditions["brand"] = brand
    if in_stock is not None:
        conditions["in_stock"] = in_stock
    if color:
        conditions["color__in"] = color.split(",")

    return {"data": [], "filters_applied": conditions}
```

### Range Filters

```python
@app.get("/api/orders")
async def list_orders(
    min_total: float | None = Query(default=None, ge=0, description="Minimum order total"),
    max_total: float | None = Query(default=None, ge=0, description="Maximum order total"),
    created_after: str | None = Query(default=None, description="Orders created after (ISO 8601)"),
    created_before: str | None = Query(default=None, description="Orders created before (ISO 8601)"),
    status: str | None = Query(default=None, description="Comma-separated statuses"),
):
    """
    Range and multi-value filters.

    Examples:
        GET /api/orders?min_total=50&max_total=200
        GET /api/orders?created_after=2025-01-01&created_before=2025-06-30
        GET /api/orders?status=pending,processing
    """
    filters = {}
    if min_total is not None:
        filters["total__gte"] = min_total
    if max_total is not None:
        filters["total__lte"] = max_total
    if created_after:
        filters["created_at__gt"] = created_after
    if created_before:
        filters["created_at__lt"] = created_before
    if status:
        filters["status__in"] = status.split(",")

    return {"data": [], "filters_applied": filters}
```

### Full-Text Search

```python
@app.get("/api/search")
async def search(
    q: str = Query(..., min_length=2, max_length=200, description="Search query"),
    type: str | None = Query(default=None, description="Resource type: products, users, orders"),
    page: int = Query(default=1, ge=1),
    per_page: int = Query(default=20, ge=1, le=100),
):
    """
    Cross-resource search endpoint.

    Example: GET /api/search?q=wireless+headphones&type=products
    """
    # In practice: use PostgreSQL FTS, Elasticsearch, or similar
    return {
        "data": [
            {"type": "product", "id": 1, "title": "Wireless Headphones", "score": 0.95},
            {"type": "product", "id": 7, "title": "Wireless Earbuds", "score": 0.82},
        ],
        "meta": {
            "query": q,
            "type": type,
            "page": page,
            "per_page": per_page,
            "total": 2,
        }
    }
```

---

## Sorting

### Single-Field Sorting

```python
@app.get("/api/products")
async def list_products_sorted(
    sort: str = Query(
        default="created_at",
        description="Field to sort by",
        regex="^(name|price|created_at|rating|popularity)$",
    ),
    order: str = Query(
        default="desc",
        description="Sort direction",
        regex="^(asc|desc)$",
    ),
):
    """
    Single-field sorting.

    Examples:
        GET /api/products?sort=price&order=asc      -- cheapest first
        GET /api/products?sort=rating&order=desc     -- highest rated first
        GET /api/products?sort=name&order=asc        -- alphabetical
    """
    return {
        "data": [{"id": 1, "name": "Widget", "price": 9.99}],
        "meta": {"sort": sort, "order": order},
    }
```

### Multi-Field Sorting

```python
@app.get("/api/employees")
async def list_employees(
    sort: str = Query(
        default="-created_at",
        description="Comma-separated sort fields. Prefix with - for descending.",
    ),
):
    """
    Multi-field sorting with direction prefix.

    Convention: "-" prefix means descending, no prefix means ascending.

    Examples:
        GET /api/employees?sort=department,-salary    -- by dept asc, then salary desc
        GET /api/employees?sort=-hire_date             -- newest hires first
        GET /api/employees?sort=last_name,first_name   -- alphabetical by full name
    """
    # Parse sort fields
    allowed_fields = {"first_name", "last_name", "department", "salary",
                      "hire_date", "created_at"}
    sort_specs = []
    for field_spec in sort.split(","):
        field_spec = field_spec.strip()
        if field_spec.startswith("-"):
            field = field_spec[1:]
            direction = "desc"
        else:
            field = field_spec
            direction = "asc"

        if field not in allowed_fields:
            continue  # or raise 400
        sort_specs.append({"field": field, "direction": direction})

    return {
        "data": [{"id": 1, "first_name": "Alice", "department": "Engineering"}],
        "meta": {"sort": sort_specs},
    }
```

---

## Field Selection

Field selection (sparse fieldsets) lets clients request only the fields they need, reducing payload size.

### Implementation

```python
@app.get("/api/users")
async def list_users_with_fields(
    fields: str | None = Query(
        default=None,
        description="Comma-separated list of fields to include",
    ),
    page: int = Query(default=1, ge=1),
    per_page: int = Query(default=20, ge=1, le=100),
):
    """
    Sparse fieldsets: return only requested fields.

    Examples:
        GET /api/users?fields=id,name,email         -- only id, name, email
        GET /api/users?fields=id,name                -- minimal response
        GET /api/users                                -- all fields (default)
    """
    # Full user objects
    users = [
        {
            "id": 1,
            "name": "Alice",
            "email": "alice@example.com",
            "role": "admin",
            "department": "Engineering",
            "phone": "+1234567890",
            "avatar_url": "/avatars/alice.jpg",
            "created_at": "2025-01-15T10:00:00Z",
            "updated_at": "2025-03-01T14:30:00Z",
        },
    ]

    # Apply field selection
    if fields:
        requested = set(fields.split(","))
        # Always include "id" for resource identity
        requested.add("id")
        allowed = {"id", "name", "email", "role", "department", "phone",
                    "avatar_url", "created_at", "updated_at"}
        valid_fields = requested & allowed

        users = [
            {k: v for k, v in user.items() if k in valid_fields}
            for user in users
        ]

    return {
        "data": users,
        "meta": {
            "page": page,
            "per_page": per_page,
            "total": len(users),
            "fields": fields.split(",") if fields else "all",
        }
    }
```

### Resource Expansion

The inverse of field selection -- embed related resources in the response:

```python
@app.get("/api/orders/{order_id}")
async def get_order(
    order_id: int,
    expand: str | None = Query(
        default=None,
        description="Comma-separated resources to embed: items, user, shipping",
    ),
):
    """
    Resource expansion: embed related resources to reduce round trips.

    Examples:
        GET /api/orders/1                              -- basic order
        GET /api/orders/1?expand=items                 -- order + line items
        GET /api/orders/1?expand=items,user,shipping   -- fully expanded
    """
    order = {
        "id": order_id,
        "status": "shipped",
        "total": 149.99,
        "user_id": 1,
    }

    expansions = set(expand.split(",")) if expand else set()

    if "items" in expansions:
        order["items"] = [
            {"product_id": 10, "name": "Widget", "quantity": 2, "price": 9.99},
            {"product_id": 20, "name": "Gadget", "quantity": 1, "price": 129.99},
        ]

    if "user" in expansions:
        order["user"] = {"id": 1, "name": "Alice", "email": "alice@example.com"}

    if "shipping" in expansions:
        order["shipping"] = {
            "carrier": "FedEx",
            "tracking_number": "FX123456789",
            "estimated_delivery": "2025-02-01",
        }

    return {"data": order}
```

---

## Exercises

### Exercise 1: Implement Cursor Pagination

Build a FastAPI endpoint for `/api/messages` with cursor-based pagination:
- Accept `limit` (default 20, max 100) and `after` (cursor) parameters
- Return messages sorted by `created_at` descending (newest first)
- Include `cursors.after` for the next page
- Include `meta.has_more` boolean
- Use base64-encoded JSON cursors
- Test with a dataset of 500 messages

### Exercise 2: Advanced Filtering

Create a `/api/products` endpoint that supports:
- Equality: `?category=electronics`
- Multi-value: `?color=red,blue,green`
- Range: `?min_price=10&max_price=100`
- Date range: `?created_after=2025-01-01`
- Boolean: `?in_stock=true`
- Full-text search: `?q=wireless`

Return the applied filters in the `meta` object so consumers can verify what was applied.

### Exercise 3: Multi-Field Sorting

Implement sorting that supports:
- Single field: `?sort=price`
- Direction: `?sort=-price` (descending)
- Multiple fields: `?sort=category,-price` (category asc, price desc)
- Validation: reject unknown fields with a 400 error
- Default sort: `-created_at` (newest first)

### Exercise 4: Sparse Fieldsets with Expansion

Build a `/api/users/{id}` endpoint that supports:
- `?fields=id,name,email` -- return only specified fields
- `?expand=orders,profile` -- embed related resources
- Both combined: `?fields=id,name&expand=orders`
- Validate field names and expansion targets

### Exercise 5: Pagination Strategy Comparison

Create three versions of a `/api/logs` endpoint:
1. Offset pagination (`page`, `per_page`)
2. Cursor pagination (`after`, `limit`)
3. Keyset pagination (`after_id`, `after_timestamp`, `limit`)

Populate with 10,000 log entries. Compare performance and behavior when:
- Paginating to page 500
- Items are inserted during pagination
- The sort order is changed

---

## Summary

This lesson covered:
1. Offset pagination: simple but suffers from page drift and performance degradation at depth
2. Cursor pagination: opaque tokens for stable, performant deep pagination
3. Keyset pagination: transparent key-based navigation with constant performance
4. HATEOAS pagination links: self, first, last, prev, next navigation
5. Filtering patterns: equality, range, multi-value, and full-text search
6. Sorting: single and multi-field with direction prefixes
7. Field selection: sparse fieldsets and resource expansion for response optimization

---

**Previous**: [Request and Response Design](04_Request_and_Response_Design.md) | [Overview](00_Overview.md) | **Next**: [Authentication and Authorization](06_Authentication_and_Authorization.md)

**License**: CC BY-NC 4.0
