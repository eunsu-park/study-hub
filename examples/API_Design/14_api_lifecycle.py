#!/usr/bin/env python3
"""Example: API Lifecycle Management

Demonstrates API lifecycle patterns:
- Version deprecation with Sunset headers
- Migration endpoint that serves both old and new formats
- Changelog generation from a structured log
- Feature toggles during migration periods

Related lesson: 14_API_Lifecycle.md

Run:
    pip install "fastapi[standard]"
    uvicorn 14_api_lifecycle:app --reload --port 8000
"""

import json
from datetime import date, datetime, timezone
from typing import Optional

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# =============================================================================
# CHANGELOG — Structured version history
# =============================================================================
# Keep a machine-readable changelog alongside the human-readable one.
# This enables automated migration tooling and client SDK generators.

CHANGELOG = [
    {
        "version": "3.0.0",
        "date": "2026-06-01",
        "status": "current",
        "changes": [
            {"type": "breaking", "description": "Renamed 'fullName' to 'full_name' (snake_case)."},
            {"type": "added", "description": "Added 'metadata' object to user response."},
            {"type": "removed", "description": "Removed deprecated 'legacy_id' field."},
        ],
    },
    {
        "version": "2.0.0",
        "date": "2025-01-15",
        "status": "deprecated",
        "sunset": "2026-12-31",
        "changes": [
            {"type": "breaking", "description": "Changed user ID from integer to UUID."},
            {"type": "added", "description": "Added pagination to list endpoints."},
        ],
    },
    {
        "version": "1.0.0",
        "date": "2024-03-01",
        "status": "retired",
        "sunset": "2025-06-30",
        "changes": [
            {"type": "added", "description": "Initial API release."},
        ],
    },
]


# =============================================================================
# SCHEMAS — Version-specific models
# =============================================================================

class UserV2(BaseModel):
    """V2 response — integer-style ID (deprecated)."""
    id: int
    fullName: str       # camelCase (old convention)
    email: str


class UserMetadata(BaseModel):
    created_at: datetime
    login_count: int = 0


class UserV3(BaseModel):
    """V3 response — UUID ID, snake_case, metadata object."""
    id: str
    full_name: str      # snake_case (new convention)
    email: str
    metadata: UserMetadata


# =============================================================================
# APPLICATION
# =============================================================================

app = FastAPI(title="API Lifecycle Demo", version="3.0.0")


# =============================================================================
# DEPRECATION MIDDLEWARE — Adds Sunset headers to old versions
# =============================================================================

DEPRECATED_PREFIXES = {
    "/api/v2/": {"sunset": "Mon, 31 Dec 2026 00:00:00 GMT", "successor": "/api/v3/"},
}

RETIRED_PREFIXES = {"/api/v1/"}


@app.middleware("http")
async def lifecycle_middleware(request: Request, call_next):
    path = request.url.path

    # Retired versions return 410 Gone immediately
    for prefix in RETIRED_PREFIXES:
        if path.startswith(prefix):
            return JSONResponse(
                status_code=410,
                content={"detail": "This API version has been retired. Use /api/v3/."},
                headers={"Link": '</api/v3/>; rel="successor-version"'},
            )

    response = await call_next(request)

    # Deprecated versions get Sunset and Deprecation headers
    for prefix, info in DEPRECATED_PREFIXES.items():
        if path.startswith(prefix):
            response.headers["Sunset"] = info["sunset"]
            response.headers["Deprecation"] = "true"
            response.headers["Link"] = f'<{info["successor"]}>; rel="successor-version"'

    return response


# =============================================================================
# MIGRATION ENDPOINT — Dual-format response
# =============================================================================
# During migration, an endpoint can serve both formats based on a header.

SAMPLE_USER = {
    "id_int": 42,
    "id_uuid": "550e8400-e29b-41d4-a716-446655440000",
    "full_name": "Alice Smith",
    "email": "alice@example.com",
    "created_at": datetime(2025, 1, 1, tzinfo=timezone.utc),
    "login_count": 128,
}


@app.get("/api/v2/users/42", tags=["V2 (Deprecated)"])
def get_user_v2():
    """Deprecated V2 endpoint — still functional but with Sunset header."""
    return UserV2(id=SAMPLE_USER["id_int"], fullName=SAMPLE_USER["full_name"],
                  email=SAMPLE_USER["email"])


@app.get("/api/v3/users/{user_id}", tags=["V3 (Current)"])
def get_user_v3(user_id: str):
    """Current V3 endpoint with metadata object."""
    return UserV3(
        id=SAMPLE_USER["id_uuid"],
        full_name=SAMPLE_USER["full_name"],
        email=SAMPLE_USER["email"],
        metadata=UserMetadata(
            created_at=SAMPLE_USER["created_at"],
            login_count=SAMPLE_USER["login_count"],
        ),
    )


# =============================================================================
# CHANGELOG ENDPOINT — Machine-readable version history
# =============================================================================

@app.get("/api/changelog", tags=["Lifecycle"])
def get_changelog(status_filter: Optional[str] = None):
    """Return the structured changelog, optionally filtered by status."""
    entries = CHANGELOG
    if status_filter:
        entries = [e for e in entries if e["status"] == status_filter]
    return {"changelog": entries}


# =============================================================================
# FEATURE FLAGS — Gradual rollout during migration
# =============================================================================

FEATURE_FLAGS = {
    "new_search_algorithm": {"enabled": True, "rollout_pct": 50},
    "v3_response_format": {"enabled": True, "rollout_pct": 100},
    "experimental_ai_summary": {"enabled": False, "rollout_pct": 0},
}


@app.get("/api/v3/features", tags=["Lifecycle"])
def list_features():
    """Expose active feature flags (useful for client SDKs)."""
    return {
        name: {"enabled": f["enabled"], "rollout_pct": f["rollout_pct"]}
        for name, f in FEATURE_FLAGS.items()
    }


def is_feature_enabled(flag: str, user_id_hash: int = 0) -> bool:
    """Check if a feature is enabled for a given user (percentage rollout)."""
    f = FEATURE_FLAGS.get(flag, {})
    if not f.get("enabled"):
        return False
    return (user_id_hash % 100) < f.get("rollout_pct", 0)


# =============================================================================
# ENTRYPOINT
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    print("Changelog:")
    print(json.dumps(CHANGELOG, indent=2))
    uvicorn.run("14_api_lifecycle:app", host="127.0.0.1", port=8000, reload=True)
