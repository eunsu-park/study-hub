# Lesson 3: URL Design and Naming

**Previous**: [REST Architecture](02_REST_Architecture.md) | [Overview](00_Overview.md) | **Next**: [Request and Response Design](04_Request_and_Response_Design.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Apply resource naming conventions that are intuitive and consistent
2. Design hierarchical URLs that reflect resource relationships
3. Choose correct pluralization and casing strategies
4. Implement query parameters for filtering, sorting, and field selection
5. Recognize and avoid common URL anti-patterns
6. Structure URLs that remain stable across API versions

---

URLs are the most visible part of your API. Before a developer reads your documentation or inspects a response body, they see the URL. A well-designed URL structure communicates intent instantly; a poorly designed one creates confusion that no amount of documentation can fully repair. This lesson establishes the conventions that make APIs predictable and pleasant to work with.

> **Analogy:** URLs are like street addresses. A good addressing system (123 Main St, Apt 4) lets anyone find a location without a map. A bad one ("the blue house past the old oak tree") requires local knowledge. Consistent, hierarchical URLs are the street grid of your API.

## Table of Contents
1. [Resource Naming Conventions](#resource-naming-conventions)
2. [Hierarchical URLs](#hierarchical-urls)
3. [Pluralization Rules](#pluralization-rules)
4. [Query Parameters](#query-parameters)
5. [Filtering Patterns](#filtering-patterns)
6. [URL Anti-Patterns](#url-anti-patterns)
7. [Practical Guidelines](#practical-guidelines)
8. [Exercises](#exercises)

---

## Resource Naming Conventions

### Use Nouns, Not Verbs

Resources represent **things**, not **actions**. The HTTP method carries the verb.

```python
# GOOD: Nouns as resources, HTTP methods as verbs
# GET    /api/users          -- list users
# POST   /api/users          -- create a user
# GET    /api/users/42       -- get user 42
# PATCH  /api/users/42       -- update user 42
# DELETE /api/users/42       -- delete user 42

# BAD: Verbs in URLs (RPC-style, not RESTful)
# GET    /api/getUsers
# POST   /api/createUser
# POST   /api/deleteUser/42
# GET    /api/fetchUserById/42
```

### Use Lowercase with Hyphens

URLs should be lowercase. Use hyphens (`-`) to separate words, not underscores or camelCase.

```python
# GOOD: Lowercase with hyphens
# /api/order-items
# /api/user-profiles
# /api/shipping-addresses

# BAD: Mixed casing or underscores in URLs
# /api/orderItems          -- camelCase
# /api/OrderItems          -- PascalCase
# /api/order_items         -- underscores (acceptable but less standard)
# /api/Order-Items         -- mixed case
```

> **Note:** While hyphens are the most common convention for URLs, some major APIs (like Google's) use underscores. The critical rule is **pick one and be consistent**.

### Use Concrete Names

Prefer specific, domain-relevant names over generic ones.

```python
from fastapi import FastAPI

app = FastAPI()

# GOOD: Specific, meaningful resource names
@app.get("/api/invoices")
async def list_invoices():
    pass

@app.get("/api/shipments")
async def list_shipments():
    pass

@app.get("/api/subscriptions")
async def list_subscriptions():
    pass

# BAD: Generic, meaningless names
# /api/items       -- items of what?
# /api/objects      -- too abstract
# /api/entities     -- conveys no domain meaning
# /api/records      -- what kind of records?
```

---

## Hierarchical URLs

### Parent-Child Relationships

Use nesting to express **ownership** or **containment** relationships.

```python
from fastapi import FastAPI

app = FastAPI()

# A user HAS many orders -- orders belong to a user
@app.get("/api/users/{user_id}/orders")
async def get_user_orders(user_id: int):
    """List orders belonging to a specific user."""
    return {
        "data": [
            {"id": 101, "total": 49.99, "status": "shipped"},
            {"id": 102, "total": 129.99, "status": "pending"},
        ]
    }

# An order HAS many items -- items belong to an order
@app.get("/api/orders/{order_id}/items")
async def get_order_items(order_id: int):
    """List line items within an order."""
    return {
        "data": [
            {"product_id": 1, "name": "Widget", "quantity": 2},
            {"product_id": 5, "name": "Gadget", "quantity": 1},
        ]
    }

# A project HAS many members
@app.get("/api/projects/{project_id}/members")
async def get_project_members(project_id: int):
    """List members of a project."""
    return {
        "data": [
            {"user_id": 1, "role": "owner"},
            {"user_id": 2, "role": "contributor"},
        ]
    }
```

### Nesting Depth Limit

Keep nesting to **two levels maximum**. Deeper nesting makes URLs long and brittle.

```python
# GOOD: Maximum two levels of nesting
# /api/users/{user_id}/orders
# /api/orders/{order_id}/items
# /api/projects/{project_id}/tasks

# BAD: Three or more levels -- too deep
# /api/users/{user_id}/orders/{order_id}/items/{item_id}/reviews
# Problem: Long URL, tightly coupled, hard to cache

# SOLUTION: Flatten deep relationships
# Instead of /api/users/1/orders/101/items/5/reviews
# Use:
# GET /api/reviews?order_item_id=5
# or
# GET /api/order-items/5/reviews
```

### When to Nest vs. Flatten

```python
# NEST when:
# - The child resource does not make sense without the parent
# - You always need the parent context to access the child
# - The relationship is strictly one-to-many ownership

# FLATTEN when:
# - The child resource can be accessed independently
# - You need to query across parents (e.g., "all reviews by any user")
# - The nesting would exceed two levels

# Example: Both valid approaches for different use cases
@app.get("/api/users/{user_id}/reviews")
async def get_reviews_by_user(user_id: int):
    """Nested: reviews in context of a specific user."""
    return {"data": [{"id": 1, "rating": 5}]}

@app.get("/api/reviews")
async def list_all_reviews(user_id: int | None = None, product_id: int | None = None):
    """Flat: query reviews across all users/products."""
    return {"data": [{"id": 1, "user_id": 1, "product_id": 3, "rating": 5}]}
```

---

## Pluralization Rules

### Always Use Plurals for Collections

```python
# GOOD: Plural nouns for collections
# /api/users              -- collection of users
# /api/users/42           -- single user from the collection
# /api/orders             -- collection of orders
# /api/orders/101         -- single order

# BAD: Mixing singular and plural
# /api/user               -- singular for collection?
# /api/user/42            -- inconsistent
# /api/order              -- singular feels like a singleton
```

### Singleton Resources

Some resources are inherently singular (there is exactly one per parent context). Use the singular form for these.

```python
# Singleton: Each user has exactly one profile
@app.get("/api/users/{user_id}/profile")
async def get_user_profile(user_id: int):
    """Singleton sub-resource -- no ID needed."""
    return {"bio": "Developer", "avatar_url": "/avatars/42.jpg"}

# Singleton: Current user's settings
@app.get("/api/settings")
async def get_settings():
    """Global settings -- only one instance exists."""
    return {"theme": "dark", "language": "en", "timezone": "UTC"}

# Singleton: Application health check
@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "uptime": "72h"}
```

### Handling Irregular Plurals

```python
# Standard plurals -- straightforward
# /api/users, /api/orders, /api/products, /api/categories

# Irregular plurals -- use the correct English plural
# /api/people (not /api/persons)
# /api/analyses (not /api/analysiss)
# /api/criteria (not /api/criterias)

# Uncountable nouns -- use plural form anyway for consistency
# /api/feedback (acceptable as-is, since "feedbacks" is awkward)
# /api/media (acceptable as-is)
# /api/metadata (acceptable as-is)

# When in doubt, pick the conventional plural and document it.
```

---

## Query Parameters

### Standard Query Parameter Patterns

```python
from fastapi import FastAPI, Query

app = FastAPI()

@app.get("/api/products")
async def list_products(
    # Pagination
    page: int = Query(default=1, ge=1, description="Page number"),
    per_page: int = Query(default=20, ge=1, le=100, description="Items per page"),

    # Filtering
    category: str | None = Query(default=None, description="Filter by category"),
    min_price: float | None = Query(default=None, ge=0, description="Minimum price"),
    max_price: float | None = Query(default=None, ge=0, description="Maximum price"),
    in_stock: bool | None = Query(default=None, description="Filter by stock status"),

    # Sorting
    sort: str = Query(default="created_at", description="Sort field"),
    order: str = Query(default="desc", regex="^(asc|desc)$", description="Sort order"),

    # Search
    q: str | None = Query(default=None, min_length=2, description="Search query"),

    # Field selection
    fields: str | None = Query(default=None, description="Comma-separated fields to include"),
):
    """
    List products with filtering, sorting, pagination, and field selection.

    Example: GET /api/products?category=electronics&min_price=10&sort=price&order=asc&page=2
    """
    return {
        "data": [{"id": 1, "name": "Widget", "price": 9.99, "category": "electronics"}],
        "meta": {
            "page": page,
            "per_page": per_page,
            "total": 150,
            "filters_applied": {
                "category": category,
                "min_price": min_price,
                "max_price": max_price,
            }
        }
    }
```

### Path Parameters vs Query Parameters

```python
# PATH PARAMETERS: Identify a specific resource
# /api/users/42              -- "user 42" is the identity
# /api/orders/101/items/5    -- "item 5 in order 101"

# QUERY PARAMETERS: Modify the response (filter, sort, paginate, search)
# /api/users?role=admin            -- filter users
# /api/users?sort=name&order=asc   -- sort users
# /api/users?page=3&per_page=10    -- paginate users
# /api/users?fields=id,name,email  -- select fields

# RULE OF THUMB:
# - Use path params for resource identification (required)
# - Use query params for everything else (optional)

# BAD: Query params for resource identity
# /api/users?id=42                 -- should be /api/users/42

# BAD: Path params for optional filters
# /api/users/admin                 -- is "admin" a user ID or a role filter?
# Use: /api/users?role=admin
```

---

## Filtering Patterns

### Simple Equality Filters

```python
# Direct field equality
# GET /api/products?color=red
# GET /api/products?status=active
# GET /api/users?role=admin&department=engineering
```

### Range Filters

```python
# Prefix convention for ranges
# GET /api/products?min_price=10&max_price=50
# GET /api/events?start_after=2025-01-01&start_before=2025-12-31
# GET /api/logs?since=2025-01-01T00:00:00Z

# Alternative: operator suffix
# GET /api/products?price_gte=10&price_lte=50
# GET /api/orders?created_at_gt=2025-01-01
```

### Multi-Value Filters

```python
from fastapi import FastAPI, Query

app = FastAPI()

@app.get("/api/products")
async def list_products(
    # Comma-separated values (most common)
    # GET /api/products?category=electronics,books,toys
    category: str | None = Query(default=None),

    # Repeated parameters (also valid)
    # GET /api/products?tag=sale&tag=new
    tag: list[str] = Query(default=[]),

    # Comma-separated IDs
    # GET /api/products?ids=1,2,3,4,5
    ids: str | None = Query(default=None),
):
    categories = category.split(",") if category else []
    id_list = [int(x) for x in ids.split(",")] if ids else []

    return {"filters": {"categories": categories, "tags": tag, "ids": id_list}}
```

### Complex Filter Syntax

For APIs that need advanced filtering, consider a structured approach:

```python
# LHS bracket syntax (used by Stripe, many others)
# GET /api/products?price[gte]=10&price[lte]=50
# GET /api/orders?created[after]=2025-01-01&status[in]=pending,processing

# Filter parameter with JSON (more flexible but less readable)
# GET /api/products?filter={"price":{"$gte":10,"$lte":50},"status":"active"}

# RSQL / FIQL syntax (standardized, powerful)
# GET /api/products?filter=price>=10;price<=50;status==active

# For most APIs, simple equality + range prefixes are sufficient.
# Only adopt complex syntax when your consumers genuinely need it.
```

### Implementing Bracket Filters in FastAPI

```python
from fastapi import FastAPI, Request

app = FastAPI()

def parse_bracket_filters(query_params: dict) -> dict:
    """Parse LHS bracket filter syntax: price[gte]=10 -> {"price": {"gte": 10}}."""
    import re
    filters = {}
    pattern = re.compile(r"^(\w+)\[(\w+)\]$")
    for key, value in query_params.items():
        match = pattern.match(key)
        if match:
            field, operator = match.groups()
            if field not in filters:
                filters[field] = {}
            filters[field][operator] = value
        else:
            filters[key] = value
    return filters

@app.get("/api/products")
async def list_products(request: Request):
    """
    Supports bracket filter syntax:
    GET /api/products?price[gte]=10&price[lte]=50&category=electronics
    """
    filters = parse_bracket_filters(dict(request.query_params))
    # filters = {"price": {"gte": "10", "lte": "50"}, "category": "electronics"}
    return {"applied_filters": filters, "data": []}
```

---

## URL Anti-Patterns

### 1. Verbs in URLs

```python
# BAD: Verbs as URL segments
# POST /api/users/create
# GET  /api/users/list
# PUT  /api/users/42/update
# POST /api/users/42/delete

# GOOD: HTTP methods carry the verb
# POST   /api/users              -- create
# GET    /api/users              -- list
# PATCH  /api/users/42           -- update
# DELETE /api/users/42           -- delete

# EXCEPTION: Action endpoints for non-CRUD operations
# POST /api/orders/42/cancel     -- acceptable for domain actions
# POST /api/users/42/deactivate  -- acceptable for state transitions
```

### 2. File Extensions in URLs

```python
# BAD: File extensions
# /api/users.json
# /api/users.xml
# /api/reports/monthly.pdf

# GOOD: Content negotiation via Accept header
# GET /api/users
# Accept: application/json        -> returns JSON
# Accept: application/xml         -> returns XML
# Accept: text/csv                -> returns CSV

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
import csv
import io

app = FastAPI()

@app.get("/api/users")
async def list_users(request: Request):
    users = [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]
    accept = request.headers.get("accept", "application/json")

    if "text/csv" in accept:
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=["id", "name"])
        writer.writeheader()
        writer.writerows(users)
        return Response(content=output.getvalue(), media_type="text/csv")

    return JSONResponse(content={"data": users})
```

### 3. Unnecessary Wrappers

```python
# BAD: Redundant path segments
# /api/v1/service/resource/users
# /api/rest/v2/data/users

# GOOD: Clean, direct paths
# /api/v1/users
# /api/users
```

### 4. Inconsistent Naming

```python
# BAD: Mixed conventions across endpoints
# /api/users              -- plural
# /api/order              -- singular
# /api/productList        -- camelCase with "List" suffix
# /api/get-categories     -- verb + hyphen
# /api/User_Profiles      -- PascalCase + underscore

# GOOD: Uniform conventions
# /api/users
# /api/orders
# /api/products
# /api/categories
# /api/user-profiles
```

### 5. Exposing Implementation Details

```python
# BAD: Leaking database/implementation details
# /api/tbl_users                  -- table prefix
# /api/mysql/users                -- database engine
# /api/users?sql=SELECT * FROM... -- raw SQL exposure
# /api/v2/legacy/users            -- "legacy" is internal context

# GOOD: Abstract over implementation
# /api/users
# The client does not need to know about your database schema.
```

### 6. Trailing Slashes Inconsistency

```python
from fastapi import FastAPI
from fastapi.middleware import Middleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import RedirectResponse

app = FastAPI()

# Choose ONE convention and enforce it.
# Most common: No trailing slash.

# FastAPI strips trailing slashes by default.
# /api/users and /api/users/ both route to the same handler.

# If you need strict enforcement:
class TrailingSlashMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        if request.url.path != "/" and request.url.path.endswith("/"):
            url = str(request.url).rstrip("/")
            return RedirectResponse(url=url, status_code=301)
        return await call_next(request)

# app.add_middleware(TrailingSlashMiddleware)
```

---

## Practical Guidelines

### URL Design Checklist

```
1. Resources are plural nouns:              /api/users, /api/orders
2. IDs in path for specific resources:      /api/users/42
3. Sub-resources for ownership:             /api/users/42/orders
4. Max two levels of nesting:               /api/users/42/orders (not deeper)
5. Query params for filtering:              /api/users?role=admin
6. Query params for pagination:             /api/users?page=2&per_page=20
7. Query params for sorting:                /api/users?sort=name&order=asc
8. Lowercase with hyphens:                  /api/order-items (not orderItems)
9. No verbs in URLs:                        POST /api/users (not /api/createUser)
10. No file extensions:                     /api/users (not /api/users.json)
11. Consistent trailing slash policy:       /api/users (no trailing slash)
12. Meaningful, domain-specific names:      /api/invoices (not /api/documents)
```

### A Complete URL Design Example

```python
from fastapi import FastAPI, Query
from pydantic import BaseModel

app = FastAPI()

# --- E-commerce API URL Design ---

# Products (collection)
# GET    /api/products                         -- list products
# POST   /api/products                         -- create product
# GET    /api/products/{id}                    -- get product
# PATCH  /api/products/{id}                    -- update product
# DELETE /api/products/{id}                    -- delete product

# Product reviews (sub-resource)
# GET    /api/products/{id}/reviews            -- list product reviews
# POST   /api/products/{id}/reviews            -- add review

# Product variants (sub-resource)
# GET    /api/products/{id}/variants           -- list variants
# POST   /api/products/{id}/variants           -- create variant

# Categories (independent resource)
# GET    /api/categories                       -- list categories
# GET    /api/categories/{id}                  -- get category
# GET    /api/categories/{id}/products         -- products in category

# Orders (independent resource)
# GET    /api/orders                           -- list orders
# POST   /api/orders                           -- create order
# GET    /api/orders/{id}                      -- get order
# POST   /api/orders/{id}/cancel               -- cancel order (action)

# Order items (sub-resource)
# GET    /api/orders/{id}/items                -- list items in order

# Users (independent resource)
# GET    /api/users/{id}                       -- get user
# GET    /api/users/{id}/orders                -- user's orders
# GET    /api/users/{id}/profile               -- user's profile (singleton)

# Search (cross-resource)
# GET    /api/search?q=widget&type=products    -- search across resources

# Example implementation
@app.get("/api/products")
async def list_products(
    category: str | None = None,
    min_price: float | None = Query(default=None, ge=0),
    max_price: float | None = Query(default=None, ge=0),
    sort: str = Query(default="created_at", regex="^(name|price|created_at|rating)$"),
    order: str = Query(default="desc", regex="^(asc|desc)$"),
    page: int = Query(default=1, ge=1),
    per_page: int = Query(default=20, ge=1, le=100),
    q: str | None = Query(default=None, min_length=2),
):
    """Full-featured product listing with clean URL design."""
    return {
        "data": [
            {"id": 1, "name": "Widget", "price": 9.99, "category": "tools"}
        ],
        "meta": {
            "page": page,
            "per_page": per_page,
            "total": 1,
        },
        "_links": {
            "self": {"href": f"/api/products?page={page}&per_page={per_page}"},
        }
    }
```

### Flask Comparison

```python
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.get("/api/products")
def list_products():
    """Flask version of the product listing endpoint."""
    page = request.args.get("page", 1, type=int)
    per_page = request.args.get("per_page", 20, type=int)
    category = request.args.get("category")
    sort = request.args.get("sort", "created_at")
    order = request.args.get("order", "desc")
    q = request.args.get("q")

    products = [{"id": 1, "name": "Widget", "price": 9.99}]

    if category:
        products = [p for p in products if p.get("category") == category]

    return jsonify({
        "data": products,
        "meta": {"page": page, "per_page": per_page, "total": len(products)},
    })

@app.get("/api/products/<int:product_id>")
def get_product(product_id):
    """Flask: path parameter with type converter."""
    return jsonify({"id": product_id, "name": "Widget", "price": 9.99})

@app.get("/api/products/<int:product_id>/reviews")
def get_product_reviews(product_id):
    """Flask: nested sub-resource."""
    return jsonify({
        "data": [{"id": 1, "product_id": product_id, "rating": 5, "text": "Great!"}]
    })
```

---

## Exercises

### Exercise 1: Fix the URLs

Rewrite each URL to follow REST naming conventions:

1. `GET /api/getAllUsers`
2. `POST /api/user/create`
3. `PUT /api/updateProduct/42`
4. `DELETE /api/remove-order?id=101`
5. `GET /api/Users/42/OrderList`
6. `GET /api/tbl_categories`
7. `POST /api/users/42/orders/101/items/5/reviews/create`

### Exercise 2: Design a URL Hierarchy

Design the complete URL hierarchy for a **project management** application with:
- Organizations, projects, tasks, comments, attachments
- Task assignments (user-to-task relationship)
- Project milestones

For each resource, define:
- The collection URL and singleton URL
- Which resources are sub-resources and which are top-level
- Query parameters for filtering and sorting

### Exercise 3: Implement Filtering

Build a FastAPI endpoint for `/api/events` that supports:
- Date range filtering (`start_after`, `start_before`)
- Category filtering (multiple categories via comma-separated values)
- Location filtering (`city`, `country`)
- Free-text search (`q`)
- Sorting by `date`, `name`, or `popularity`
- Pagination with `page` and `per_page`

### Exercise 4: URL Design Review

Take an existing API you work with and audit its URL design:
- Are resources named consistently (plural nouns, lowercase)?
- Is nesting depth appropriate?
- Are query parameters used correctly for filtering vs path parameters for identity?
- Are there any anti-patterns (verbs in URLs, file extensions, inconsistent casing)?

Write a brief report with specific recommendations.

---

## Summary

This lesson covered:
1. Resource naming conventions: nouns, lowercase, hyphens
2. Hierarchical URL design with parent-child nesting (max two levels)
3. Pluralization rules for collections and singletons
4. Query parameter patterns for filtering, sorting, pagination, and field selection
5. Filtering approaches from simple equality to bracket syntax
6. Common URL anti-patterns and how to avoid them

---

**Previous**: [REST Architecture](02_REST_Architecture.md) | [Overview](00_Overview.md) | **Next**: [Request and Response Design](04_Request_and_Response_Design.md)

**License**: CC BY-NC 4.0
