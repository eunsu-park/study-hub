#!/usr/bin/env python3
"""Example: API Evolution

Demonstrates backward-compatible API evolution strategies:
- Additive changes (safe: new fields, new endpoints)
- Feature flags for gradual rollout
- Response envelope evolution (old clients still work)
- Tolerant reader pattern
- Schema migration without breaking clients

Related lesson: 24_API_Evolution.md

Run:
    pip install "fastapi[standard]"
    uvicorn 24_api_evolution:app --reload --port 8000
"""

import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import FastAPI, Header, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("evolution")

# =============================================================================
# FEATURE FLAGS — Control new behavior without redeployment
# =============================================================================
# Feature flags allow you to merge code to main that is not yet active,
# then enable it for a percentage of users or specific clients.

_FEATURE_FLAGS: dict[str, dict] = {
    "enhanced_search": {"enabled": True, "rollout_pct": 100},
    "user_metadata_v2": {"enabled": True, "rollout_pct": 50},
    "experimental_recs": {"enabled": False, "rollout_pct": 0},
}


def is_flag_enabled(flag: str, client_id: str = "") -> bool:
    """Check if a feature flag is active for a given client.

    Uses consistent hashing so the same client always gets the same result.
    """
    f = _FEATURE_FLAGS.get(flag, {})
    if not f.get("enabled"):
        return False
    if f.get("rollout_pct", 0) >= 100:
        return True
    # Deterministic bucketing by client ID
    bucket = int(hashlib.md5(f"{flag}:{client_id}".encode()).hexdigest()[:8], 16) % 100
    return bucket < f["rollout_pct"]


# =============================================================================
# SCHEMAS — Evolved response (backward compatible)
# =============================================================================
# Rule: NEVER remove or rename existing fields. Only ADD new ones.
# Old clients ignore unknown fields (tolerant reader pattern).

class AddressV1(BaseModel):
    """Original address — a single string field."""
    address: str = ""


class AddressV2(BaseModel):
    """Structured address — added alongside the flat string for compatibility."""
    street: str = ""
    city: str = ""
    country: str = ""
    postal_code: str = ""


class UserResponse(BaseModel):
    """Evolved user response demonstrating additive-only changes.

    Evolution history:
    - v1.0: id, name, email
    - v1.1: added 'address' (string)
    - v1.2: added 'address_structured' (object) — old field kept
    - v1.3: added 'metadata' (object) — behind feature flag
    """
    id: str
    name: str
    email: str
    # v1.1 — original flat address (kept for backward compat)
    address: str = ""
    # v1.2 — structured address (new clients use this)
    address_structured: Optional[AddressV2] = None
    # v1.3 — metadata (behind feature flag)
    metadata: Optional[dict[str, Any]] = None
    # Always include API version so clients can adapt
    _api_version: str = Field("1.3", alias="api_version")


# =============================================================================
# DATA
# =============================================================================

USERS = {
    "u1": {
        "id": "u1", "name": "Alice", "email": "alice@example.com",
        "address": "123 Main St, NYC, US 10001",
        "address_structured": {"street": "123 Main St", "city": "NYC",
                                "country": "US", "postal_code": "10001"},
        "metadata": {"signup_source": "web", "tier": "premium",
                      "last_login": "2026-04-01T10:00:00Z"},
    },
    "u2": {
        "id": "u2", "name": "Bob", "email": "bob@example.com",
        "address": "456 Oak Ave, London, UK SW1A 1AA",
        "address_structured": {"street": "456 Oak Ave", "city": "London",
                                "country": "UK", "postal_code": "SW1A 1AA"},
        "metadata": {"signup_source": "mobile", "tier": "free",
                      "last_login": "2026-03-28T15:30:00Z"},
    },
}


# =============================================================================
# APPLICATION
# =============================================================================

app = FastAPI(title="API Evolution Demo", version="1.3.0")


# =============================================================================
# EVOLVED ENDPOINT — Same URL, richer response over time
# =============================================================================

@app.get("/api/v1/users/{user_id}", response_model=UserResponse, tags=["Users"])
def get_user(user_id: str, request: Request, x_client_id: str = Header("default")):
    """Get a user. Response shape evolves over time but remains backward compatible.

    - Old clients: only read 'id', 'name', 'email', 'address' (ignore new fields)
    - New clients: also use 'address_structured' and 'metadata'
    """
    user = USERS.get(user_id)
    if not user:
        return JSONResponse(status_code=404, content={"detail": "User not found"})

    response = {
        "id": user["id"],
        "name": user["name"],
        "email": user["email"],
        "address": user["address"],  # Always present (backward compat)
        "address_structured": AddressV2(**user["address_structured"]),
        "api_version": "1.3",
    }

    # Conditionally include metadata based on feature flag
    if is_flag_enabled("user_metadata_v2", x_client_id):
        response["metadata"] = user["metadata"]

    return response


# =============================================================================
# ADDITIVE ENDPOINT — New capability, zero impact on existing clients
# =============================================================================

@app.get("/api/v1/users/{user_id}/recommendations", tags=["Users"])
def get_recommendations(user_id: str, x_client_id: str = Header("default")):
    """New endpoint added in v1.3 — old clients never call it, so no breakage.

    Behind a feature flag for gradual rollout.
    """
    if not is_flag_enabled("experimental_recs", x_client_id):
        return JSONResponse(
            status_code=404,
            content={"detail": "Feature not available for this client."},
        )
    return {"user_id": user_id, "recommendations": ["item-a", "item-b"]}


# =============================================================================
# SEARCH — Enhanced with feature flag
# =============================================================================

@app.get("/api/v1/search", tags=["Search"])
def search(
    q: str = Query(..., min_length=1),
    x_client_id: str = Header("default"),
):
    """Search endpoint. Enhanced search returns extra fields when flag is on."""
    results = [
        {"id": uid, "name": u["name"]}
        for uid, u in USERS.items()
        if q.lower() in u["name"].lower() or q.lower() in u["email"].lower()
    ]

    if is_flag_enabled("enhanced_search", x_client_id):
        # Add relevance score (new field — old clients ignore it)
        for r in results:
            r["relevance_score"] = 0.95

    return {"query": q, "results": results, "count": len(results)}


# =============================================================================
# FEATURE FLAGS ADMIN — Manage flags at runtime
# =============================================================================

@app.get("/admin/features", tags=["Admin"])
def list_features():
    return _FEATURE_FLAGS


@app.post("/admin/features/{flag}/toggle", tags=["Admin"])
def toggle_feature(flag: str):
    if flag not in _FEATURE_FLAGS:
        return JSONResponse(status_code=404, content={"detail": "Flag not found"})
    _FEATURE_FLAGS[flag]["enabled"] = not _FEATURE_FLAGS[flag]["enabled"]
    return _FEATURE_FLAGS[flag]


# =============================================================================
# EVOLUTION GUIDELINES
# =============================================================================

GUIDELINES = """
API Evolution Rules (Backward Compatible)
==========================================
SAFE (non-breaking):
  + Add new optional fields to responses
  + Add new endpoints
  + Add new optional query parameters
  + Add new enum values (if client uses tolerant reader)
  + Add new HTTP methods to existing resources

UNSAFE (breaking — requires new version):
  - Remove or rename existing fields
  - Change field types (string -> int)
  - Change required/optional status of request fields
  - Change URL structure
  - Change authentication mechanism
  - Change error response format
"""

if __name__ == "__main__":
    import uvicorn
    print(GUIDELINES)
    uvicorn.run("24_api_evolution:app", host="127.0.0.1", port=8000, reload=True)
