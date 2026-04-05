#!/bin/bash
# Exercises for Lesson 07: API Versioning
# Topic: API_Design
# Solutions to practice problems from the lesson.

# === Exercise 1: URL Path Versioning ===
# Problem: Implement URL path versioning with v1 and v2 of a users endpoint,
# where v2 adds a new field and changes the response format.
exercise_1() {
    echo "=== Exercise 1: URL Path Versioning ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
from fastapi import FastAPI, APIRouter

app = FastAPI()

# --- Version 1 ---
v1_router = APIRouter(prefix="/api/v1", tags=["v1"])

@v1_router.get("/users/{user_id}")
def get_user_v1(user_id: str):
    """V1: Simple flat response with 'name' as a single string."""
    return {
        "id": user_id,
        "name": "Alice Smith",           # Single name field
        "email": "alice@example.com",
    }


# --- Version 2 ---
# Breaking changes from v1:
# 1. "name" split into "first_name" and "last_name"
# 2. Response wrapped in "data" envelope
# 3. Added "created_at" timestamp
v2_router = APIRouter(prefix="/api/v2", tags=["v2"])

@v2_router.get("/users/{user_id}")
def get_user_v2(user_id: str):
    """V2: Structured name, envelope, timestamps."""
    return {
        "data": {
            "id": user_id,
            "first_name": "Alice",        # Split name (breaking change)
            "last_name": "Smith",
            "email": "alice@example.com",
            "created_at": "2025-01-15T10:00:00Z",  # New field
        },
        "meta": {"api_version": "2"},
    }


app.include_router(v1_router)
app.include_router(v2_router)

# URL path versioning:
# Pros: Explicit, easy to understand, easy to route
# Cons: URL changes for every version, harder to test both versions
#
# GET /api/v1/users/123 → flat response with "name"
# GET /api/v2/users/123 → envelope with "first_name" + "last_name"
SOLUTION
}

# === Exercise 2: Header-Based Versioning ===
# Problem: Implement versioning using a custom Accept header
# (Accept: application/vnd.myapp.v2+json).
exercise_2() {
    echo "=== Exercise 2: Header-Based Versioning ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
from fastapi import FastAPI, Request, HTTPException
import re

app = FastAPI()


def extract_version(request: Request) -> int:
    """Extract API version from Accept header.

    Pattern: application/vnd.myapp.v{N}+json
    Default: v1 if no version specified.

    This is the approach used by GitHub's API.
    """
    accept = request.headers.get("accept", "")

    # Parse version from vendor media type
    match = re.search(r"application/vnd\.myapp\.v(\d+)\+json", accept)
    if match:
        return int(match.group(1))

    return 1  # Default to v1


@app.get("/api/users/{user_id}")
def get_user(user_id: str, request: Request):
    """Single endpoint, version determined by Accept header."""
    version = extract_version(request)

    if version == 1:
        return {
            "id": user_id,
            "name": "Alice Smith",
            "email": "alice@example.com",
        }
    elif version == 2:
        return {
            "data": {
                "id": user_id,
                "first_name": "Alice",
                "last_name": "Smith",
                "email": "alice@example.com",
            },
        }
    else:
        raise HTTPException(
            status_code=400,
            detail=f"API version {version} is not supported. Use v1 or v2.",
        )

# Usage:
# curl -H "Accept: application/vnd.myapp.v1+json" :8000/api/users/1
# curl -H "Accept: application/vnd.myapp.v2+json" :8000/api/users/1
#
# Header versioning:
# Pros: Clean URLs (no version in path), follows HTTP content negotiation
# Cons: Harder to test (need custom headers), less discoverable
SOLUTION
}

# === Exercise 3: Deprecation Strategy ===
# Problem: Implement API deprecation with sunset headers, deprecation
# warnings, and a migration guide endpoint.
exercise_3() {
    echo "=== Exercise 3: Deprecation Strategy ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
from fastapi import FastAPI, Request, Response
from datetime import datetime, timezone

app = FastAPI()

# Deprecation schedule
DEPRECATIONS = {
    "/api/v1": {
        "deprecated": True,
        "sunset_date": "2025-12-31T23:59:59Z",
        "successor": "/api/v2",
        "migration_guide": "https://docs.example.com/migration/v1-to-v2",
    },
}


@app.middleware("http")
async def deprecation_middleware(request: Request, call_next):
    """Add deprecation headers to responses for deprecated API versions.

    Standard headers:
    - Deprecation: true (IETF draft-ietf-httpapi-deprecation-header)
    - Sunset: date (RFC 8594 — when the API will be removed)
    - Link: migration guide URL
    """
    response = await call_next(request)

    # Check if the request path matches a deprecated version
    for prefix, info in DEPRECATIONS.items():
        if request.url.path.startswith(prefix) and info["deprecated"]:
            response.headers["Deprecation"] = "true"
            response.headers["Sunset"] = info["sunset_date"]
            response.headers["Link"] = (
                f'<{info["migration_guide"]}>; rel="deprecation"'
            )
            # Add warning header (RFC 7234)
            response.headers["Warning"] = (
                '299 - "This API version is deprecated. '
                f'Migrate to {info["successor"]} before {info["sunset_date"]}"'
            )
    return response


# V1 (deprecated)
@app.get("/api/v1/users")
def list_users_v1():
    """Deprecated endpoint — still works but includes sunset headers."""
    return {"users": [{"name": "Alice"}]}


# V2 (current)
@app.get("/api/v2/users")
def list_users_v2():
    """Current version — no deprecation headers."""
    return {"data": [{"first_name": "Alice", "last_name": "Smith"}]}


# Migration guide endpoint
@app.get("/api/migration/v1-to-v2")
def migration_guide():
    """Machine-readable migration guide for automated tooling."""
    return {
        "from": "v1",
        "to": "v2",
        "breaking_changes": [
            {
                "endpoint": "GET /users/{id}",
                "change": "name → first_name + last_name",
                "type": "field_split",
            },
            {
                "endpoint": "ALL",
                "change": "Responses wrapped in {data: ...} envelope",
                "type": "response_format",
            },
        ],
        "sunset_date": "2025-12-31T23:59:59Z",
    }
SOLUTION
}

# === Exercise 4: Backward Compatible Changes ===
# Problem: Add new features to an API without breaking existing clients.
# Demonstrate additive changes that are backward compatible.
exercise_4() {
    echo "=== Exercise 4: Backward Compatible Changes ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

app = FastAPI()

# BACKWARD COMPATIBLE CHANGES (safe — no version bump needed):
# 1. Adding new optional fields to responses
# 2. Adding new optional query parameters
# 3. Adding new endpoints
# 4. Adding new enum values (if clients use switch/case, warn them)
# 5. Relaxing validation (accept wider input range)

# BREAKING CHANGES (require version bump):
# 1. Removing or renaming fields
# 2. Changing field types (string → int)
# 3. Changing URL structure
# 4. Making optional fields required
# 5. Tightening validation (reject previously valid input)
# 6. Changing error response format
# 7. Changing authentication scheme


# Example: Evolving a product endpoint without breaking clients

# --- Original response (v1 launch) ---
class ProductV1(BaseModel):
    id: str
    name: str
    price: float


# --- After 3 months: add optional fields (backward compatible!) ---
class ProductV1_1(BaseModel):
    id: str
    name: str
    price: float
    # NEW fields — old clients ignore these (Postel's Law)
    currency: str = "USD"           # Default preserves old behavior
    discount_percent: Optional[float] = None   # Optional, not required
    tags: list[str] = []            # Default empty list


# --- After 6 months: add new endpoint (backward compatible!) ---
@app.get("/api/v1/products/{product_id}")
def get_product(product_id: str):
    """Original endpoint — unchanged for existing clients."""
    return ProductV1_1(
        id=product_id,
        name="Widget",
        price=29.99,
        currency="USD",
        discount_percent=10.0,
        tags=["electronics", "sale"],
    )


@app.get("/api/v1/products/{product_id}/reviews")
def get_reviews(product_id: str):
    """NEW endpoint — does not affect existing clients."""
    return {"data": [{"rating": 5, "text": "Great product!"}]}


# --- After 6 months: add optional query parameter (backward compatible!) ---
@app.get("/api/v1/products")
def list_products(
    category: Optional[str] = None,    # Existing
    min_rating: Optional[float] = None, # NEW — old clients do not send this
    currency: Optional[str] = None,     # NEW — old clients do not send this
):
    """New optional params do not break existing client requests."""
    return {"data": []}


# RULE: If you need to make a breaking change, bump the version.
# Otherwise, use additive changes and defaults to stay backward compatible.
SOLUTION
}

# Run all exercises
echo "Exercise solutions for Lesson 07: API Versioning"
echo "=================================================="
exercise_1
echo ""
exercise_2
echo ""
exercise_3
echo ""
exercise_4
