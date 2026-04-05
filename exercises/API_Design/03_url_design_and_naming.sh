#!/bin/bash
# Exercises for Lesson 03: URL Design and Naming
# Topic: API_Design
# Solutions to practice problems from the lesson.

# === Exercise 1: Resource Naming Conventions ===
# Problem: Fix the following poorly designed URLs to follow REST naming
# best practices (plural nouns, lowercase, hyphens, no verbs in URLs).
exercise_1() {
    echo "=== Exercise 1: Resource Naming Conventions ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
# BAD URL → GOOD URL (with explanation)

# 1. GET /getUserById?id=123
#    → GET /api/v1/users/123
#    Fix: Use path parameter for ID, plural noun, no verb in URL

# 2. POST /api/createNewOrder
#    → POST /api/v1/orders
#    Fix: HTTP method (POST) already means "create", no verb needed

# 3. GET /api/User_Profiles
#    → GET /api/v1/user-profiles
#    Fix: Lowercase, hyphens instead of underscores, plural

# 4. DELETE /api/v1/removeProduct/456
#    → DELETE /api/v1/products/456
#    Fix: DELETE method already means "remove", no verb needed

# 5. PUT /api/v1/product/update/456
#    → PUT /api/v1/products/456
#    Fix: PUT method means "update", use plural noun, no action in URL

# 6. GET /api/v1/getOrdersByUser/789
#    → GET /api/v1/users/789/orders
#    Fix: Hierarchical relationship, no verb

# 7. POST /api/v1/search_Products?q=widget
#    → GET /api/v1/products?q=widget
#    Fix: Search is a read operation (GET), lowercase with hyphens

# RULES SUMMARY:
# 1. Use plural nouns: /users, /orders, /products
# 2. No verbs in URLs: HTTP methods are the verbs
# 3. Lowercase with hyphens: /user-profiles (not /User_Profiles)
# 4. Hierarchical relationships: /users/{id}/orders
# 5. Version prefix: /api/v1/...
# 6. Query params for filtering: ?status=active&sort=name
SOLUTION
}

# === Exercise 2: Hierarchical Resource URLs ===
# Problem: Design a URL hierarchy for a blog platform with users, posts,
# comments, and tags. Show all CRUD endpoints.
exercise_2() {
    echo "=== Exercise 2: Hierarchical Resource URLs ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
from fastapi import FastAPI, Query
from typing import Optional

app = FastAPI()

# --- URL Hierarchy for a Blog Platform ---
#
# /api/v1/users
# /api/v1/users/{user_id}
# /api/v1/users/{user_id}/posts              (user's posts — sub-resource)
# /api/v1/posts                              (all posts — top-level collection)
# /api/v1/posts/{post_id}
# /api/v1/posts/{post_id}/comments           (post's comments — sub-resource)
# /api/v1/posts/{post_id}/comments/{comment_id}
# /api/v1/posts/{post_id}/tags               (post's tags — sub-resource)
# /api/v1/tags                               (all tags — top-level collection)

# Users
@app.get("/api/v1/users")
def list_users(): ...

@app.post("/api/v1/users", status_code=201)
def create_user(): ...

@app.get("/api/v1/users/{user_id}")
def get_user(user_id: str): ...

@app.patch("/api/v1/users/{user_id}")
def update_user(user_id: str): ...

@app.delete("/api/v1/users/{user_id}", status_code=204)
def delete_user(user_id: str): ...

# Posts (both top-level and as user sub-resource)
@app.get("/api/v1/posts")
def list_all_posts(
    author: Optional[str] = None,
    tag: Optional[str] = None,
    sort: str = Query("-created_at", description="Sort field, prefix - for desc"),
): ...

@app.get("/api/v1/users/{user_id}/posts")
def list_user_posts(user_id: str):
    """Sub-resource: posts belonging to a specific user."""
    ...

@app.post("/api/v1/posts", status_code=201)
def create_post(): ...

@app.get("/api/v1/posts/{post_id}")
def get_post(post_id: str): ...

# Comments (sub-resource of posts — max 2 levels deep)
@app.get("/api/v1/posts/{post_id}/comments")
def list_comments(post_id: str): ...

@app.post("/api/v1/posts/{post_id}/comments", status_code=201)
def create_comment(post_id: str): ...

@app.delete("/api/v1/posts/{post_id}/comments/{comment_id}", status_code=204)
def delete_comment(post_id: str, comment_id: str): ...

# Tags
@app.get("/api/v1/tags")
def list_tags(): ...

@app.get("/api/v1/posts/{post_id}/tags")
def list_post_tags(post_id: str): ...

# DESIGN RULES:
# 1. Max 2 levels of nesting: /posts/{id}/comments (not /users/{id}/posts/{id}/comments)
# 2. Provide top-level access when useful: /posts (not just /users/{id}/posts)
# 3. Use query params for cross-cutting filters: /posts?author=alice&tag=python
SOLUTION
}

# === Exercise 3: Query Parameter Design ===
# Problem: Design query parameters for a product search API that supports
# filtering, sorting, pagination, and sparse fieldsets.
exercise_3() {
    echo "=== Exercise 3: Query Parameter Design ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
from fastapi import FastAPI, Query
from typing import Optional
from pydantic import BaseModel

app = FastAPI()

@app.get("/api/v1/products")
def search_products(
    # --- Filtering ---
    category: Optional[str] = Query(None, description="Filter by category"),
    brand: Optional[str] = Query(None, description="Filter by brand"),
    min_price: Optional[float] = Query(None, ge=0, description="Minimum price"),
    max_price: Optional[float] = Query(None, ge=0, description="Maximum price"),
    in_stock: Optional[bool] = Query(None, description="Filter by availability"),
    tags: Optional[str] = Query(
        None,
        description="Comma-separated tags: tags=electronics,sale",
    ),

    # --- Sorting ---
    sort: str = Query(
        "-created_at",
        description="Sort field. Prefix with - for descending. "
        "Examples: price, -price, name, -created_at",
    ),

    # --- Pagination ---
    page: int = Query(1, ge=1, description="Page number"),
    per_page: int = Query(20, ge=1, le=100, description="Items per page"),

    # --- Sparse Fieldsets ---
    fields: Optional[str] = Query(
        None,
        description="Comma-separated fields to include: fields=id,name,price",
    ),

    # --- Search ---
    q: Optional[str] = Query(None, description="Full-text search query"),
):
    """Product search with filtering, sorting, pagination, and sparse fieldsets.

    Example requests:
    - GET /api/v1/products?category=electronics&min_price=10&max_price=100
    - GET /api/v1/products?sort=-price&page=2&per_page=10
    - GET /api/v1/products?q=wireless&fields=id,name,price&in_stock=true
    - GET /api/v1/products?tags=sale,clearance&sort=price
    """
    # Parse sort parameter
    descending = sort.startswith("-")
    sort_field = sort.lstrip("-")

    # Parse sparse fieldsets
    requested_fields = fields.split(",") if fields else None

    # Parse tags
    tag_list = tags.split(",") if tags else None

    # Build response
    return {
        "data": [],
        "meta": {
            "page": page,
            "per_page": per_page,
            "total": 0,
            "sort": sort,
            "filters_applied": {
                "category": category,
                "brand": brand,
                "price_range": [min_price, max_price],
                "in_stock": in_stock,
                "tags": tag_list,
            },
        },
    }
SOLUTION
}

# === Exercise 4: URL Anti-Patterns ===
# Problem: Identify the anti-pattern in each URL and explain the fix.
exercise_4() {
    echo "=== Exercise 4: URL Anti-Patterns ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
# Anti-Pattern 1: File extensions in URLs
# BAD:  GET /api/v1/users.json
# GOOD: GET /api/v1/users  (with Accept: application/json header)
# Why: Content negotiation should use headers, not URL extensions.

# Anti-Pattern 2: Verbs in resource URLs
# BAD:  POST /api/v1/users/123/activate
# OK:   POST /api/v1/users/123/activation  (noun form of action)
# BEST: PATCH /api/v1/users/123  body: {"status": "active"}
# Why: State changes should be modeled as resource updates when possible.
# Exception: truly non-CRUD actions (e.g., /api/v1/orders/123/refund)

# Anti-Pattern 3: Deeply nested resources
# BAD:  GET /api/v1/companies/1/departments/5/teams/3/members/42/tasks
# GOOD: GET /api/v1/tasks?team_id=3&assignee_id=42
# Why: Deep nesting is hard to navigate. Use query params for filtering.

# Anti-Pattern 4: Sensitive data in URLs
# BAD:  GET /api/v1/auth/login?password=secret123
# GOOD: POST /api/v1/auth/login  body: {"password": "secret123"}
# Why: URLs appear in server logs, browser history, and proxy caches.

# Anti-Pattern 5: Inconsistent pluralization
# BAD:  GET /api/v1/user/123/order  (singular)
# GOOD: GET /api/v1/users/123/orders  (always plural for collections)
# Why: Consistency reduces cognitive load for API consumers.

# Anti-Pattern 6: Trailing slashes
# BAD:  GET /api/v1/users/ (inconsistent with GET /api/v1/users)
# GOOD: Pick one and redirect the other. Most APIs use no trailing slash.
SOLUTION
}

# Run all exercises
echo "Exercise solutions for Lesson 03: URL Design and Naming"
echo "========================================================="
exercise_1
echo ""
exercise_2
echo ""
exercise_3
echo ""
exercise_4
