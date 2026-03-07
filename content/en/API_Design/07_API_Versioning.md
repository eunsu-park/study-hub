# Lesson 7: API Versioning

**Previous**: [Authentication and Authorization](06_Authentication_and_Authorization.md) | [Overview](00_Overview.md) | **Next**: [Error Handling](08_Error_Handling.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Compare URL, header, and query parameter versioning strategies with trade-offs
2. Design APIs that maintain backward compatibility through additive changes
3. Implement a deprecation strategy with sunset headers and migration guides
4. Define a version lifecycle from beta through end-of-life
5. Apply practical techniques to avoid breaking changes in production APIs
6. Choose the right versioning strategy for your API's audience and scale

---

Versioning is the art of evolving your API without breaking existing consumers. Every API will change -- new fields, renamed endpoints, restructured responses. The question is not whether you will need versioning, but how gracefully you will handle it. A good versioning strategy lets you innovate freely while honoring the contract you made with your consumers.

> **Analogy:** API versioning is like a restaurant menu. You can add new dishes (additive change) without confusing regulars. But if you rename "Spaghetti Carbonara" to "Pasta #7" and change the recipe, loyal customers will be upset. A versioned menu ("Spring 2025 Menu") lets you make bigger changes while giving customers time to adjust.

## Table of Contents
1. [Why Version?](#why-version)
2. [URL Path Versioning](#url-path-versioning)
3. [Header Versioning](#header-versioning)
4. [Query Parameter Versioning](#query-parameter-versioning)
5. [Versioning Strategy Comparison](#versioning-strategy-comparison)
6. [Backward Compatibility](#backward-compatibility)
7. [Deprecation Strategy](#deprecation-strategy)
8. [Version Lifecycle](#version-lifecycle)
9. [Exercises](#exercises)

---

## Why Version?

### Breaking vs Non-Breaking Changes

```python
# NON-BREAKING changes (safe to deploy without versioning):
# - Adding a new optional field to a response
# - Adding a new endpoint
# - Adding a new optional query parameter
# - Adding a new HTTP method to an existing resource
# - Making a required field optional
# - Adding a new enum value (if clients are tolerant)

# BREAKING changes (require versioning):
# - Removing a field from a response
# - Renaming a field
# - Changing a field's type (string -> integer)
# - Removing an endpoint
# - Changing the URL structure
# - Making an optional field required
# - Changing the meaning of a status code
# - Changing authentication mechanism
# - Changing pagination format
```

### The Cost of Breaking Changes

```
Without versioning:
  1. You deploy a breaking change
  2. All clients break simultaneously
  3. Emergency rollback or hotfix
  4. Loss of trust, support tickets, downtime

With versioning:
  1. You deploy v2 alongside v1
  2. v1 clients continue working unchanged
  3. Clients migrate to v2 at their own pace
  4. v1 is deprecated, then sunset after migration
```

---

## URL Path Versioning

The version number is part of the URL path. This is the most common approach.

### Implementation

```python
from fastapi import FastAPI, APIRouter

app = FastAPI()

# Version 1
v1_router = APIRouter(prefix="/api/v1")

@v1_router.get("/users")
async def list_users_v1():
    """V1: Returns users with 'name' as a single field."""
    return {
        "data": [
            {"id": 1, "name": "Alice Smith", "email": "alice@example.com"},
            {"id": 2, "name": "Bob Jones", "email": "bob@example.com"},
        ]
    }

@v1_router.get("/users/{user_id}")
async def get_user_v1(user_id: int):
    return {"id": user_id, "name": "Alice Smith", "email": "alice@example.com"}


# Version 2: Breaking change -- split 'name' into 'first_name' and 'last_name'
v2_router = APIRouter(prefix="/api/v2")

@v2_router.get("/users")
async def list_users_v2():
    """V2: Breaking change -- 'name' split into 'first_name' and 'last_name'."""
    return {
        "data": [
            {"id": 1, "first_name": "Alice", "last_name": "Smith", "email": "alice@example.com"},
            {"id": 2, "first_name": "Bob", "last_name": "Jones", "email": "bob@example.com"},
        ]
    }

@v2_router.get("/users/{user_id}")
async def get_user_v2(user_id: int):
    return {"id": user_id, "first_name": "Alice", "last_name": "Smith", "email": "alice@example.com"}


# Mount both versions
app.include_router(v1_router)
app.include_router(v2_router)

# Both work simultaneously:
# GET /api/v1/users  -> {"name": "Alice Smith"}
# GET /api/v2/users  -> {"first_name": "Alice", "last_name": "Smith"}
```

### Organized Project Structure

```
app/
├── main.py
├── v1/
│   ├── __init__.py
│   ├── router.py
│   ├── schemas.py      # Pydantic models for v1
│   └── services.py     # Business logic (may be shared)
├── v2/
│   ├── __init__.py
│   ├── router.py
│   ├── schemas.py      # Pydantic models for v2
│   └── services.py
└── shared/
    ├── __init__.py
    ├── database.py      # Shared database access
    ├── auth.py          # Shared authentication
    └── models.py        # Shared ORM models
```

```python
# main.py
from fastapi import FastAPI
from v1.router import router as v1_router
from v2.router import router as v2_router

app = FastAPI(title="My API", version="2.0.0")

app.include_router(v1_router, prefix="/api/v1", tags=["v1"])
app.include_router(v2_router, prefix="/api/v2", tags=["v2"])
```

### URL Versioning in Flask

```python
from flask import Flask, Blueprint, jsonify

app = Flask(__name__)

# Version 1
v1 = Blueprint("v1", __name__, url_prefix="/api/v1")

@v1.get("/users")
def list_users_v1():
    return jsonify({
        "data": [{"id": 1, "name": "Alice Smith"}]
    })

# Version 2
v2 = Blueprint("v2", __name__, url_prefix="/api/v2")

@v2.get("/users")
def list_users_v2():
    return jsonify({
        "data": [{"id": 1, "first_name": "Alice", "last_name": "Smith"}]
    })

app.register_blueprint(v1)
app.register_blueprint(v2)
```

---

## Header Versioning

The version is specified in a custom request header. The URL remains clean and stable.

### Implementation

```python
from fastapi import FastAPI, Header, HTTPException

app = FastAPI()

@app.get("/api/users")
async def list_users(
    api_version: str = Header(default="1", alias="X-API-Version"),
):
    """
    Header versioning: client specifies version in X-API-Version header.

    Usage:
        GET /api/users
        X-API-Version: 1     -> v1 response format

        GET /api/users
        X-API-Version: 2     -> v2 response format

        GET /api/users        -> defaults to v1 (backward compatible)
    """
    if api_version == "1":
        return {
            "data": [{"id": 1, "name": "Alice Smith", "email": "alice@example.com"}]
        }
    elif api_version == "2":
        return {
            "data": [{"id": 1, "first_name": "Alice", "last_name": "Smith", "email": "alice@example.com"}]
        }
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported API version: {api_version}. Supported: 1, 2"
        )
```

### Accept Header Versioning (Content Negotiation)

```python
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
import re

app = FastAPI()

def parse_api_version(accept: str) -> int:
    """
    Parse version from Accept header.
    Example: application/vnd.myapi.v2+json -> 2
    """
    match = re.search(r"application/vnd\.myapi\.v(\d+)\+json", accept)
    if match:
        return int(match.group(1))
    return 1  # default version

@app.get("/api/users")
async def list_users(request: Request):
    """
    Content negotiation versioning via Accept header.

    Usage:
        GET /api/users
        Accept: application/vnd.myapi.v1+json

        GET /api/users
        Accept: application/vnd.myapi.v2+json
    """
    accept = request.headers.get("accept", "")
    version = parse_api_version(accept)

    if version == 1:
        data = [{"id": 1, "name": "Alice Smith"}]
    elif version == 2:
        data = [{"id": 1, "first_name": "Alice", "last_name": "Smith"}]
    else:
        raise HTTPException(status_code=406, detail=f"Unsupported version: v{version}")

    return JSONResponse(
        content={"data": data},
        media_type=f"application/vnd.myapi.v{version}+json",
    )
```

---

## Query Parameter Versioning

The version is passed as a query parameter.

### Implementation

```python
from fastapi import FastAPI, Query

app = FastAPI()

@app.get("/api/users")
async def list_users(
    version: int = Query(default=1, alias="v", ge=1, le=2, description="API version"),
):
    """
    Query parameter versioning.

    Usage:
        GET /api/users?v=1    -> v1 format
        GET /api/users?v=2    -> v2 format
        GET /api/users        -> defaults to v1
    """
    if version == 1:
        return {"data": [{"id": 1, "name": "Alice Smith"}]}
    else:
        return {"data": [{"id": 1, "first_name": "Alice", "last_name": "Smith"}]}
```

---

## Versioning Strategy Comparison

| Aspect | URL Path | Header | Query Parameter |
|--------|----------|--------|-----------------|
| Visibility | `/api/v2/users` -- highly visible | Hidden in headers | `?v=2` -- visible |
| Cacheability | Excellent (different URL = different cache) | Requires `Vary` header | Good (query string included in cache key) |
| Simplicity | Very simple | Moderate | Simple |
| Discoverability | Easy (browse URLs) | Requires documentation | Easy |
| Link sharing | Links include version | Links do not include version | Links include version |
| Client complexity | Low | Medium (must set headers) | Low |
| URL cleanliness | Cluttered with version | Clean URLs | Slightly cluttered |
| Routing | Framework-native | Custom middleware | Framework-native |
| Used by | GitHub, Stripe, Twilio | Azure, Google Cloud | Netflix, Amazon |

### Recommendation

```python
# For most APIs:
# 1. Use URL path versioning (/api/v1/) -- simplest, most common, best tooling support
# 2. Only bump the major version for breaking changes
# 3. Add new features without versioning (additive changes)

# Decision matrix:
def choose_versioning(
    audience: str,       # "public" | "internal" | "partner"
    breaking_frequency: str,  # "rare" | "frequent"
) -> str:
    if audience == "public":
        return "URL path"  # most discoverable, simplest for external devs
    if audience == "internal" and breaking_frequency == "frequent":
        return "header"  # clean URLs, easy to change
    return "URL path"  # safe default
```

---

## Backward Compatibility

### Additive Changes (Non-Breaking)

```python
from pydantic import BaseModel

# Original response (v1):
class UserV1(BaseModel):
    id: int
    name: str
    email: str

# Enhanced response (still v1 -- backward compatible):
class UserV1Enhanced(BaseModel):
    id: int
    name: str
    email: str
    # New fields added as optional -- existing clients ignore them
    avatar_url: str | None = None
    created_at: str | None = None
    department: str | None = None

# Clients that only read id/name/email continue to work.
# Clients that want the new fields can start using them immediately.
# No version bump needed.
```

### Tolerant Reader Pattern

Design clients to be tolerant of unknown fields and missing optional fields:

```python
# Server adds a new field "department" to the user response.
# A well-designed client ignores fields it does not recognize.

import httpx

async def get_user(user_id: int) -> dict:
    """Tolerant reader: only extract the fields we need."""
    async with httpx.AsyncClient() as client:
        response = await client.get(f"https://api.example.com/api/v1/users/{user_id}")
        data = response.json()

    # Extract only what we need -- ignore unknown fields
    return {
        "id": data["id"],
        "name": data["name"],
        "email": data.get("email", ""),  # use .get() for optional fields
    }
    # If the server adds "department", "avatar_url", etc., this code still works.
```

### Avoiding Breaking Changes

```python
# Strategy 1: Add fields, never remove them
# Before: {"id": 1, "name": "Alice"}
# After:  {"id": 1, "name": "Alice", "display_name": "Alice S."}

# Strategy 2: Deprecate fields before removing
# Phase 1: Add new field, keep old field, add deprecation notice
# {"id": 1, "name": "Alice", "first_name": "Alice", "last_name": "Smith"}
# Phase 2: Remove old field in next major version

# Strategy 3: Use nullable types for new required concepts
# Instead of making a new field required (breaking):
# Add it as optional first, then require it in the next version

# Strategy 4: Response evolution with envelope
class UserResponse(BaseModel):
    id: int
    name: str                      # keep for backward compatibility
    first_name: str | None = None  # new field, optional
    last_name: str | None = None   # new field, optional
    email: str

    # Both old and new clients are satisfied:
    # Old client reads "name" -> works
    # New client reads "first_name" + "last_name" -> works
```

---

## Deprecation Strategy

### Sunset Header (RFC 8594)

```python
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from datetime import datetime
import warnings

app = FastAPI()

DEPRECATED_ENDPOINTS = {
    "/api/v1/users": {
        "sunset_date": "2025-06-01",
        "successor": "/api/v2/users",
        "migration_guide": "https://docs.example.com/migration/v1-to-v2",
    }
}

@app.middleware("http")
async def deprecation_middleware(request: Request, call_next):
    """Add deprecation headers to sunset endpoints."""
    response = await call_next(request)
    path = request.url.path

    deprecation_info = DEPRECATED_ENDPOINTS.get(path)
    if deprecation_info:
        response.headers["Deprecation"] = "true"
        response.headers["Sunset"] = deprecation_info["sunset_date"]
        response.headers["Link"] = (
            f'<{deprecation_info["successor"]}>; rel="successor-version", '
            f'<{deprecation_info["migration_guide"]}>; rel="deprecation"'
        )

    return response

@app.get("/api/v1/users")
async def list_users_v1():
    """DEPRECATED: Use /api/v2/users instead. Sunset: 2025-06-01."""
    return JSONResponse(
        content={
            "data": [{"id": 1, "name": "Alice Smith"}],
            "_deprecation": {
                "message": "This endpoint is deprecated. Use /api/v2/users instead.",
                "sunset_date": "2025-06-01",
                "migration_guide": "https://docs.example.com/migration/v1-to-v2",
            }
        },
        headers={
            "Deprecation": "true",
            "Sunset": "Sat, 01 Jun 2025 00:00:00 GMT",
        }
    )
```

### Deprecation Communication Plan

```
Timeline for deprecating an API version:

Month 0:  Announce deprecation
          - Blog post, changelog, email to registered developers
          - Add Deprecation: true header to all v1 responses
          - Set Sunset date (usually 6-12 months out)

Month 1:  v2 is stable
          - Update all documentation to show v2 as primary
          - Add migration guide with code examples
          - Provide automated migration tools if possible

Month 3:  Usage monitoring
          - Track v1 usage metrics
          - Contact high-volume v1 consumers directly
          - Offer migration support

Month 6:  Sunset warning
          - Return 299 Warning header on v1 responses
          - Log all v1 requests for final outreach
          - Ensure <5% of traffic is still on v1

Month 9:  Hard sunset
          - v1 returns 410 Gone
          - Response body includes migration guide link
          - Keep this 410 response for 6+ months
```

### Sunset Response

```python
from fastapi import FastAPI
from fastapi.responses import JSONResponse

app = FastAPI()

@app.get("/api/v1/{path:path}")
async def sunset_v1(path: str):
    """All v1 endpoints return 410 Gone after sunset date."""
    return JSONResponse(
        content={
            "error": {
                "code": "VERSION_SUNSET",
                "message": "API v1 has been discontinued as of 2025-06-01.",
                "migration_guide": "https://docs.example.com/migration/v1-to-v2",
                "successor": f"/api/v2/{path}",
            }
        },
        status_code=410,
    )
```

---

## Version Lifecycle

### Lifecycle Stages

```
┌─────────┐    ┌──────────┐    ┌─────────┐    ┌────────────┐    ┌──────────┐
│  Beta    │───►│  Stable  │───►│  Deprecated  │───►│  Sunset  │───►│  Removed │
│ (preview)│    │ (current)│    │ (maintenance)│    │  (410)   │    │  (gone)  │
└─────────┘    └──────────┘    └────────────┘    └──────────┘    └──────────┘
  0-3 months    12+ months      3-6 months        3-6 months      permanent
```

### Lifecycle Implementation

```python
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from enum import Enum
from datetime import date

app = FastAPI()

class VersionStatus(str, Enum):
    BETA = "beta"
    STABLE = "stable"
    DEPRECATED = "deprecated"
    SUNSET = "sunset"

VERSION_REGISTRY = {
    "v1": {
        "status": VersionStatus.DEPRECATED,
        "released": "2023-01-01",
        "deprecated": "2024-06-01",
        "sunset": "2025-06-01",
        "successor": "v2",
    },
    "v2": {
        "status": VersionStatus.STABLE,
        "released": "2024-06-01",
        "deprecated": None,
        "sunset": None,
        "successor": None,
    },
    "v3": {
        "status": VersionStatus.BETA,
        "released": None,
        "deprecated": None,
        "sunset": None,
        "successor": None,
    },
}

@app.get("/api/versions")
async def list_versions():
    """Discovery endpoint: list all API versions and their status."""
    return {
        "versions": [
            {
                "version": version,
                "status": info["status"].value,
                "released": info["released"],
                "deprecated": info["deprecated"],
                "sunset": info["sunset"],
                "base_url": f"/api/{version}",
            }
            for version, info in VERSION_REGISTRY.items()
        ],
        "current": "v2",
        "latest_beta": "v3",
    }

@app.middleware("http")
async def version_lifecycle_middleware(request: Request, call_next):
    """Enforce version lifecycle policies."""
    path = request.url.path

    # Extract version from path
    for version, info in VERSION_REGISTRY.items():
        if path.startswith(f"/api/{version}/"):
            status = info["status"]

            # Beta: add warning header
            if status == VersionStatus.BETA:
                response = await call_next(request)
                response.headers["X-API-Status"] = "beta"
                response.headers["Warning"] = '199 - "This API version is in beta and may change"'
                return response

            # Deprecated: add sunset headers
            if status == VersionStatus.DEPRECATED:
                response = await call_next(request)
                response.headers["Deprecation"] = "true"
                response.headers["Sunset"] = info["sunset"]
                response.headers["X-API-Status"] = "deprecated"
                successor = info["successor"]
                response.headers["Link"] = f'</api/{successor}/>; rel="successor-version"'
                return response

            # Sunset: return 410
            if status == VersionStatus.SUNSET:
                successor = info["successor"]
                return JSONResponse(
                    status_code=410,
                    content={
                        "error": {
                            "code": "VERSION_SUNSET",
                            "message": f"API {version} was sunset on {info['sunset']}.",
                            "successor": f"/api/{successor}/",
                        }
                    },
                )

            break  # stable version, no special handling

    return await call_next(request)
```

### Semantic Versioning for APIs

```python
# Semantic Versioning: MAJOR.MINOR.PATCH
#
# MAJOR: Breaking changes (bump the URL version: /v1/ -> /v2/)
# MINOR: New features, backward compatible (no URL change)
# PATCH: Bug fixes, backward compatible (no URL change)
#
# In the URL, only the MAJOR version appears:
# /api/v1/users   -- covers v1.0.0 through v1.99.99
# /api/v2/users   -- starts with v2.0.0
#
# The full version can be exposed via header or metadata:

@app.get("/api/v2/version")
async def get_version_info():
    return {
        "api_version": "v2",
        "full_version": "2.3.1",
        "release_date": "2025-01-15",
        "changelog": "https://docs.example.com/changelog",
    }
```

---

## Exercises

### Exercise 1: URL-Versioned API

Build a FastAPI application with two API versions:
- v1: `GET /api/v1/products` returns `{"name": "...", "price": "..."}`  (price as string)
- v2: `GET /api/v2/products` returns `{"name": "...", "price_cents": 999}` (price as integer in cents)
- Both versions share the same database/service layer
- Use separate routers with a shared service module

### Exercise 2: Header Versioning Middleware

Implement header-based versioning:
- Accept `X-API-Version` header (default to latest stable)
- Accept `Accept: application/vnd.myapp.v2+json` header
- Route to the correct handler based on version
- Return appropriate `Content-Type` in response
- Return 400 for unsupported versions

### Exercise 3: Deprecation System

Build a deprecation management system:
- Register endpoints as deprecated with a sunset date
- Automatically add `Deprecation`, `Sunset`, and `Link` headers
- Log warnings when deprecated endpoints are called
- Return 410 Gone after the sunset date
- Provide a `/api/versions` discovery endpoint

### Exercise 4: Backward-Compatible Evolution

Start with a v1 user API and evolve it through three stages without breaking clients:
1. v1.0: `{"name": "Alice Smith", "email": "alice@example.com"}`
2. v1.1: Add `avatar_url` (optional, no version bump)
3. v1.2: Add `first_name` and `last_name` alongside `name` (no version bump)
4. v2.0: Remove `name`, keep only `first_name` and `last_name` (version bump)

Demonstrate that a v1.0 client works unchanged through stages 1-3.

### Exercise 5: Version Lifecycle Dashboard

Create endpoints that expose:
- `GET /api/versions` -- list all versions with status
- `GET /api/versions/v1/changelog` -- changes in v1
- `GET /api/versions/v2/breaking-changes` -- breaking changes from v1 to v2
- `GET /api/health` -- includes current version information

---

## Summary

This lesson covered:
1. URL path versioning: simple, discoverable, cache-friendly (most recommended)
2. Header versioning: clean URLs, requires documentation, used for content negotiation
3. Query parameter versioning: simple but less conventional
4. Backward compatibility: additive changes, tolerant readers, field deprecation
5. Deprecation strategy: Sunset headers, migration guides, communication timeline
6. Version lifecycle: beta, stable, deprecated, sunset, removed stages

---

**Previous**: [Authentication and Authorization](06_Authentication_and_Authorization.md) | [Overview](00_Overview.md) | **Next**: [Error Handling](08_Error_Handling.md)

**License**: CC BY-NC 4.0
