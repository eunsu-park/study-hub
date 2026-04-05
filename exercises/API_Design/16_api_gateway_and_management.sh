#!/bin/bash
# Exercises for Lesson 16: API Gateway and Management
# Topic: API_Design
# Solutions to practice problems from the lesson.

# === Exercise 1: API Gateway Routing ===
# Problem: Build a simple API gateway that routes requests to different
# backend services based on the URL path prefix.
exercise_1() {
    echo "=== Exercise 1: API Gateway Routing ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

app = FastAPI(title="API Gateway")

# Service registry: path prefix → backend URL
SERVICES = {
    "/api/v1/users":    "http://user-service:8001",
    "/api/v1/orders":   "http://order-service:8002",
    "/api/v1/products": "http://product-service:8003",
    "/api/v1/payments": "http://payment-service:8004",
}


def find_service(path: str) -> tuple[str, str]:
    """Match request path to a backend service.

    Returns (backend_url, remaining_path) or raises if no match.
    """
    for prefix, backend_url in SERVICES.items():
        if path.startswith(prefix):
            remaining = path[len(prefix):] or "/"
            return backend_url, remaining
    return None, None


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
async def gateway_proxy(request: Request, path: str):
    """Reverse proxy: forward requests to the appropriate backend service.

    The gateway handles:
    1. Service discovery (URL prefix → backend)
    2. Request forwarding (preserves headers, body, query params)
    3. Response passthrough (preserves status code, headers)
    4. Error handling (503 if backend is unreachable)
    """
    full_path = f"/{path}"
    backend_url, remaining_path = find_service(full_path)

    if not backend_url:
        return JSONResponse(
            status_code=404,
            content={"error": f"No service registered for path: {full_path}"},
        )

    # Build target URL
    target = f"{backend_url}{remaining_path}"
    if request.url.query:
        target += f"?{request.url.query}"

    # Forward headers (strip hop-by-hop headers)
    headers = dict(request.headers)
    headers.pop("host", None)
    headers["X-Forwarded-For"] = request.client.host if request.client else "unknown"
    headers["X-Forwarded-Proto"] = request.url.scheme

    try:
        async with httpx.AsyncClient() as client:
            body = await request.body()
            response = await client.request(
                method=request.method,
                url=target,
                headers=headers,
                content=body,
                timeout=30.0,
            )

        # Pass through response
        return Response(
            content=response.content,
            status_code=response.status_code,
            headers=dict(response.headers),
        )

    except httpx.ConnectError:
        return JSONResponse(
            status_code=503,
            content={"error": f"Service unavailable: {backend_url}"},
        )
    except httpx.TimeoutException:
        return JSONResponse(
            status_code=504,
            content={"error": "Backend service timed out"},
        )
SOLUTION
}

# === Exercise 2: API Key Management ===
# Problem: Build an API key management system for a developer portal
# with key creation, rotation, and usage tracking.
exercise_2() {
    echo "=== Exercise 2: API Key Management ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
import hashlib
import secrets
from datetime import datetime, timezone, timedelta
from fastapi import FastAPI, Depends, HTTPException, Header
from pydantic import BaseModel, Field
from typing import Optional

app = FastAPI(title="API Key Management")

# Storage for API keys (in production, use a database)
api_keys_db: dict[str, dict] = {}
usage_log: list[dict] = []


def generate_api_key() -> tuple[str, str]:
    """Generate an API key and its hash.

    The full key is shown to the developer ONCE at creation time.
    We store only the hash (like passwords — never store plaintext keys).
    """
    key = f"sk_live_{secrets.token_urlsafe(32)}"
    key_hash = hashlib.sha256(key.encode()).hexdigest()
    # Return the prefix for display and the full key
    return key, key_hash


class CreateKeyRequest(BaseModel):
    name: str = Field(..., description="Friendly name for this key")
    scopes: list[str] = Field(default=["read"], description="Allowed scopes")
    expires_in_days: Optional[int] = Field(None, ge=1, le=365)


class APIKeyResponse(BaseModel):
    id: str
    name: str
    prefix: str          # "sk_live_abc..." (first 12 chars for identification)
    scopes: list[str]
    created_at: str
    expires_at: Optional[str] = None
    last_used_at: Optional[str] = None
    request_count: int = 0


@app.post("/api/v1/developer/keys", status_code=201)
def create_api_key(body: CreateKeyRequest):
    """Create a new API key.

    IMPORTANT: The full key is returned ONLY in this response.
    We store only the hash — the key cannot be retrieved later.
    """
    key, key_hash = generate_api_key()
    key_id = key_hash[:16]

    expires_at = None
    if body.expires_in_days:
        expires_at = (
            datetime.now(timezone.utc) + timedelta(days=body.expires_in_days)
        ).isoformat()

    record = {
        "id": key_id,
        "name": body.name,
        "key_hash": key_hash,
        "prefix": key[:15] + "...",
        "scopes": body.scopes,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "expires_at": expires_at,
        "last_used_at": None,
        "request_count": 0,
        "active": True,
    }
    api_keys_db[key_hash] = record

    return {
        "key": key,  # SHOWN ONLY ONCE — developer must save this
        "warning": "Save this key securely. It will not be shown again.",
        **APIKeyResponse(**record).model_dump(),
    }


@app.get("/api/v1/developer/keys")
def list_api_keys():
    """List all API keys (without the actual key values)."""
    return {
        "data": [
            APIKeyResponse(**{k: v for k, v in record.items() if k != "key_hash"})
            for record in api_keys_db.values()
            if record["active"]
        ]
    }


@app.post("/api/v1/developer/keys/{key_id}/rotate")
def rotate_api_key(key_id: str):
    """Rotate an API key — generates a new key, deactivates the old one.

    Key rotation workflow:
    1. Generate new key
    2. Return new key to developer
    3. Old key remains active for a grace period (24h)
    4. After grace period, old key is deactivated
    """
    # Find the old key
    old_record = None
    for record in api_keys_db.values():
        if record["id"] == key_id and record["active"]:
            old_record = record
            break

    if not old_record:
        raise HTTPException(status_code=404, detail="API key not found")

    # Generate new key
    new_key, new_hash = generate_api_key()
    new_record = {
        **old_record,
        "id": new_hash[:16],
        "key_hash": new_hash,
        "prefix": new_key[:15] + "...",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "request_count": 0,
    }
    api_keys_db[new_hash] = new_record

    # Mark old key for deactivation (grace period)
    old_record["active"] = False

    return {
        "new_key": new_key,
        "warning": "Save this key securely. Old key has been deactivated.",
        "old_key_prefix": old_record["prefix"],
    }


@app.delete("/api/v1/developer/keys/{key_id}", status_code=204)
def revoke_api_key(key_id: str):
    """Revoke an API key immediately."""
    for record in api_keys_db.values():
        if record["id"] == key_id:
            record["active"] = False
            return
    raise HTTPException(status_code=404, detail="API key not found")
SOLUTION
}

# === Exercise 3: Request/Response Logging ===
# Problem: Implement API gateway middleware that logs all requests and
# responses for monitoring, debugging, and analytics.
exercise_3() {
    echo "=== Exercise 3: Request/Response Logging ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
import time
import json
import logging
from uuid import uuid4
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
)
logger = logging.getLogger("api_gateway")

app = FastAPI()


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next) -> Response:
    """Log every API request and response for observability.

    Logged fields:
    - Request: method, path, client IP, user agent, request ID
    - Response: status code, latency, response size
    - Correlation: request_id links request to response in logs
    """
    request_id = str(uuid4())[:8]
    start = time.monotonic()

    # Extract request info
    request_info = {
        "request_id": request_id,
        "method": request.method,
        "path": str(request.url.path),
        "query": str(request.url.query) if request.url.query else None,
        "client_ip": request.client.host if request.client else "unknown",
        "user_agent": request.headers.get("user-agent", "unknown"),
    }

    # Add request ID to response headers for client-side correlation
    try:
        response = await call_next(request)
    except Exception as exc:
        # Log unhandled errors
        elapsed = (time.monotonic() - start) * 1000
        error_log = {
            **request_info,
            "status": 500,
            "latency_ms": round(elapsed, 2),
            "error": str(exc),
        }
        logger.error(json.dumps(error_log))
        raise

    elapsed = (time.monotonic() - start) * 1000

    # Read response body for size calculation
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Response-Time"] = f"{elapsed:.0f}ms"

    # Log the complete request/response
    log_entry = {
        **request_info,
        "status": response.status_code,
        "latency_ms": round(elapsed, 2),
        "content_type": response.headers.get("content-type", "unknown"),
    }

    # Color-code by status for readability
    if response.status_code >= 500:
        logger.error(json.dumps(log_entry))
    elif response.status_code >= 400:
        logger.warning(json.dumps(log_entry))
    else:
        logger.info(json.dumps(log_entry))

    return response


# --- Analytics aggregation ---
from collections import defaultdict, Counter

analytics = {
    "requests_per_path": Counter(),
    "status_codes": Counter(),
    "avg_latency_per_path": defaultdict(list),
}


@app.get("/api/v1/gateway/analytics")
def get_analytics():
    """Return API usage analytics."""
    avg_latencies = {}
    for path, latencies in analytics["avg_latency_per_path"].items():
        avg_latencies[path] = {
            "avg_ms": round(sum(latencies) / len(latencies), 2),
            "p99_ms": round(sorted(latencies)[int(len(latencies) * 0.99)], 2) if latencies else 0,
            "count": len(latencies),
        }

    return {
        "top_endpoints": analytics["requests_per_path"].most_common(10),
        "status_distribution": dict(analytics["status_codes"]),
        "latency_by_path": avg_latencies,
    }
SOLUTION
}

# === Exercise 4: API Lifecycle Management ===
# Problem: Design an API lifecycle workflow from design to deprecation,
# implementing version negotiation and sunset scheduling.
exercise_4() {
    echo "=== Exercise 4: API Lifecycle Management ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
from fastapi import FastAPI, Request, Response
from datetime import datetime, timezone
from enum import Enum


class APIVersion(str, Enum):
    V1 = "v1"       # Sunset scheduled
    V2 = "v2"       # Current (stable)
    V3_BETA = "v3"  # Beta (unstable, may change)


# API lifecycle states
VERSIONS = {
    "v1": {
        "status": "deprecated",
        "released": "2024-01-01",
        "deprecated": "2025-01-01",
        "sunset": "2025-12-31",
        "supported": True,  # Still works, but deprecated
    },
    "v2": {
        "status": "stable",
        "released": "2025-01-01",
        "deprecated": None,
        "sunset": None,
        "supported": True,
    },
    "v3": {
        "status": "beta",
        "released": "2025-06-01",
        "deprecated": None,
        "sunset": None,
        "supported": True,
        "stability_warning": "Beta API — may change without notice",
    },
}

app = FastAPI()


@app.middleware("http")
async def lifecycle_middleware(request: Request, call_next) -> Response:
    """Add lifecycle headers to every response."""
    response = await call_next(request)

    # Extract version from URL
    path = request.url.path
    for ver, info in VERSIONS.items():
        if f"/api/{ver}/" in path:
            response.headers["X-API-Version"] = ver
            response.headers["X-API-Status"] = info["status"]

            if info["status"] == "deprecated":
                response.headers["Deprecation"] = "true"
                response.headers["Sunset"] = info["sunset"]
                response.headers["Link"] = (
                    f'</api/v2{path.split(ver, 1)[1]}>; rel="successor-version"'
                )

            if info["status"] == "beta":
                response.headers["X-API-Stability"] = "beta"
                response.headers["Warning"] = (
                    '199 - "This is a beta API. It may change without notice."'
                )
            break

    return response


# API version discovery endpoint
@app.get("/api/versions")
def list_versions():
    """Discover available API versions and their lifecycle status.

    Clients should use this to:
    1. Find the current stable version
    2. Check if their version is deprecated
    3. Plan migration before sunset
    """
    return {
        "versions": VERSIONS,
        "recommended": "v2",
        "lifecycle_stages": {
            "beta": "Unstable, may change. Use for testing only.",
            "stable": "Production-ready. Breaking changes require new version.",
            "deprecated": "Still functional but scheduled for removal.",
            "sunset": "Permanently removed. Returns 410 Gone.",
        },
    }


@app.get("/api/v1/data")
def v1_data():
    return {"message": "V1 response (deprecated — migrate to v2)"}


@app.get("/api/v2/data")
def v2_data():
    return {"data": {"message": "V2 response (stable)"}}


@app.get("/api/v3/data")
def v3_data():
    return {"data": {"message": "V3 response (beta)"}, "meta": {"format": "new"}}


# API lifecycle timeline:
# Design → Beta → Stable → Deprecated → Sunset → Removed
#
# Key rules:
# 1. Never sunset without 6+ months notice
# 2. Always provide migration guides
# 3. Support at least 2 versions simultaneously
# 4. Monitor deprecated version usage before sunset
SOLUTION
}

# Run all exercises
echo "Exercise solutions for Lesson 16: API Gateway and Management"
echo "=============================================================="
exercise_1
echo ""
exercise_2
echo ""
exercise_3
echo ""
exercise_4
