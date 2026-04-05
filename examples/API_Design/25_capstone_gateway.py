#!/usr/bin/env python3
"""Example: Capstone — Unified API Gateway

Demonstrates a unified gateway that combines REST, GraphQL, and gRPC backends:
- REST proxy with path-based routing
- GraphQL endpoint with Strawberry
- Simulated gRPC backend integration
- Shared authentication middleware
- Unified error handling and response envelopes
- Health aggregation across all backends
- Request correlation and structured logging

Related lesson: 25_Capstone_Unified_API_Gateway.md

Run:
    pip install strawberry-graphql[fastapi] "fastapi[standard]"
    uvicorn 25_capstone_gateway:app --reload --port 8000

    # REST:    GET /api/rest/products
    # GraphQL: POST /api/graphql
    # gRPC:    GET /api/grpc/products/1  (simulated)
    # Health:  GET /gateway/health
"""

import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

import strawberry
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from strawberry.fastapi import GraphQLRouter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("gateway")

# =============================================================================
# SHARED DATA — Simulated backend databases
# =============================================================================

PRODUCTS = {
    "1": {"id": "1", "name": "Laptop", "price": 999.99, "stock": 50,
           "category": "electronics"},
    "2": {"id": "2", "name": "Keyboard", "price": 79.99, "stock": 200,
           "category": "electronics"},
    "3": {"id": "3", "name": "Notebook", "price": 12.50, "stock": 500,
           "category": "stationery"},
}

USERS = {
    "u1": {"id": "u1", "name": "Alice", "email": "alice@example.com", "role": "admin"},
    "u2": {"id": "u2", "name": "Bob", "email": "bob@example.com", "role": "user"},
}

ORDERS = {
    "o1": {"id": "o1", "user_id": "u1", "product_id": "1", "quantity": 1,
            "total": 999.99, "status": "shipped"},
    "o2": {"id": "o2", "user_id": "u2", "product_id": "3", "quantity": 10,
            "total": 125.00, "status": "processing"},
}

# =============================================================================
# APPLICATION
# =============================================================================

app = FastAPI(
    title="Unified API Gateway",
    version="1.0.0",
    description="A capstone gateway combining REST, GraphQL, and gRPC backends.",
)

# =============================================================================
# MIDDLEWARE — Shared authentication, correlation ID, timing
# =============================================================================

VALID_TOKENS = {"admin-token": "u1", "user-token": "u2"}
PUBLIC_PATHS = {"/gateway/health", "/docs", "/openapi.json", "/api/graphql"}


@app.middleware("http")
async def gateway_middleware(request: Request, call_next):
    start = time.monotonic()

    # Correlation ID for distributed tracing
    correlation_id = request.headers.get("X-Correlation-ID", str(uuid.uuid4())[:12])
    request.state.correlation_id = correlation_id

    # Authentication (skip for public paths)
    path = request.url.path
    if not any(path.startswith(p) for p in PUBLIC_PATHS):
        auth = request.headers.get("Authorization", "")
        token = auth.replace("Bearer ", "") if auth.startswith("Bearer ") else ""
        user_id = VALID_TOKENS.get(token)
        request.state.user_id = user_id
        if not user_id:
            return JSONResponse(
                status_code=401,
                content={"error": "unauthorized", "message": "Invalid or missing token"},
                headers={"X-Correlation-ID": correlation_id},
            )
    else:
        request.state.user_id = None

    response = await call_next(request)

    elapsed_ms = (time.monotonic() - start) * 1000
    response.headers["X-Correlation-ID"] = correlation_id
    response.headers["X-Response-Time-Ms"] = f"{elapsed_ms:.1f}"
    response.headers["X-Gateway-Version"] = "1.0.0"

    logger.info(
        f"correlation_id={correlation_id} method={request.method} "
        f"path={path} status={response.status_code} latency_ms={elapsed_ms:.1f}"
    )
    return response


# =============================================================================
# REST BACKEND — Traditional CRUD endpoints
# =============================================================================

class ProductResponse(BaseModel):
    id: str
    name: str
    price: float
    stock: int
    category: str


@app.get("/api/rest/products", response_model=list[ProductResponse], tags=["REST"])
def rest_list_products(category: Optional[str] = None):
    """REST endpoint: list products with optional category filter."""
    results = list(PRODUCTS.values())
    if category:
        results = [p for p in results if p["category"] == category]
    return results


@app.get("/api/rest/products/{product_id}", response_model=ProductResponse, tags=["REST"])
def rest_get_product(product_id: str):
    """REST endpoint: get a single product."""
    product = PRODUCTS.get(product_id)
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")
    return product


@app.get("/api/rest/orders", tags=["REST"])
def rest_list_orders(request: Request):
    """REST endpoint: list orders for the authenticated user."""
    user_id = request.state.user_id
    user_orders = [o for o in ORDERS.values() if o["user_id"] == user_id]
    return {"orders": user_orders, "count": len(user_orders)}


# =============================================================================
# GRAPHQL BACKEND — Flexible querying
# =============================================================================

@strawberry.type
class GQLProduct:
    id: str
    name: str
    price: float
    stock: int
    category: str


@strawberry.type
class GQLUser:
    id: str
    name: str
    email: str

    @strawberry.field
    def orders(self) -> list["GQLOrder"]:
        return [
            GQLOrder(id=o["id"], quantity=o["quantity"], total=o["total"],
                     status=o["status"], _product_id=o["product_id"])
            for o in ORDERS.values() if o["user_id"] == self.id
        ]


@strawberry.type
class GQLOrder:
    id: str
    quantity: int
    total: float
    status: str
    _product_id: strawberry.Private[str]

    @strawberry.field
    def product(self) -> Optional[GQLProduct]:
        p = PRODUCTS.get(self._product_id)
        return GQLProduct(**p) if p else None


@strawberry.type
class GQLQuery:
    @strawberry.field
    def products(self, category: Optional[str] = None) -> list[GQLProduct]:
        results = list(PRODUCTS.values())
        if category:
            results = [p for p in results if p["category"] == category]
        return [GQLProduct(**p) for p in results]

    @strawberry.field
    def product(self, id: str) -> Optional[GQLProduct]:
        p = PRODUCTS.get(id)
        return GQLProduct(**p) if p else None

    @strawberry.field
    def user(self, id: str) -> Optional[GQLUser]:
        u = USERS.get(id)
        return GQLUser(id=u["id"], name=u["name"], email=u["email"]) if u else None


gql_schema = strawberry.Schema(query=GQLQuery)
graphql_router = GraphQLRouter(gql_schema)
app.include_router(graphql_router, prefix="/api/graphql", tags=["GraphQL"])


# =============================================================================
# GRPC BACKEND (SIMULATED) — High-performance binary protocol
# =============================================================================
# In production, the gateway would use grpcio to call real gRPC services.
# Here we simulate the gRPC response format.

@app.get("/api/grpc/products/{product_id}", tags=["gRPC (simulated)"])
def grpc_get_product(product_id: str):
    """Simulated gRPC call — returns a product via the gRPC backend.

    In a real gateway, this would:
    1. Serialize the request to protobuf
    2. Call the gRPC service via a stub
    3. Deserialize the protobuf response
    4. Convert to JSON for the HTTP client
    """
    product = PRODUCTS.get(product_id)
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")
    return {
        "backend": "grpc",
        "protocol": "HTTP/2 + protobuf (simulated as JSON)",
        "data": product,
    }


@app.get("/api/grpc/users/{user_id}/orders", tags=["gRPC (simulated)"])
def grpc_user_orders(user_id: str):
    """Simulated gRPC streaming — returns orders for a user."""
    user_orders = [o for o in ORDERS.values() if o["user_id"] == user_id]
    return {
        "backend": "grpc",
        "stream_type": "server_streaming",
        "data": user_orders,
        "count": len(user_orders),
    }


# =============================================================================
# HEALTH AGGREGATION — Check all backends
# =============================================================================

@app.get("/gateway/health", tags=["Gateway"])
def health():
    """Aggregate health status from all backends."""
    backends = {
        "rest": {"status": "healthy", "latency_ms": 1.2},
        "graphql": {"status": "healthy", "latency_ms": 2.5},
        "grpc": {"status": "healthy", "latency_ms": 0.8},
    }
    all_healthy = all(b["status"] == "healthy" for b in backends.values())
    return {
        "gateway": "healthy" if all_healthy else "degraded",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "backends": backends,
    }


# =============================================================================
# GATEWAY INFO
# =============================================================================

@app.get("/gateway/info", tags=["Gateway"])
def gateway_info():
    """Overview of available API backends and their capabilities."""
    return {
        "name": "Unified API Gateway",
        "version": "1.0.0",
        "backends": {
            "rest": {
                "base_path": "/api/rest/",
                "protocol": "HTTP/1.1 + JSON",
                "best_for": "Simple CRUD, public APIs, browser clients",
            },
            "graphql": {
                "base_path": "/api/graphql",
                "protocol": "HTTP + JSON (GraphQL)",
                "best_for": "Flexible queries, mobile clients, aggregation",
            },
            "grpc": {
                "base_path": "/api/grpc/",
                "protocol": "HTTP/2 + Protobuf (simulated)",
                "best_for": "Microservice-to-microservice, low latency, streaming",
            },
        },
        "auth": "Bearer token in Authorization header",
        "tokens": {"admin": "admin-token", "user": "user-token"},
    }


# =============================================================================
# ENTRYPOINT
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    print("=" * 60)
    print("Unified API Gateway — Capstone Example")
    print("=" * 60)
    print()
    print("REST:    GET http://localhost:8000/api/rest/products")
    print("GraphQL: POST http://localhost:8000/api/graphql")
    print("gRPC:    GET http://localhost:8000/api/grpc/products/1")
    print("Health:  GET http://localhost:8000/gateway/health")
    print("Info:    GET http://localhost:8000/gateway/info")
    print()
    print("Auth:    Authorization: Bearer admin-token")
    print()
    uvicorn.run("25_capstone_gateway:app", host="127.0.0.1", port=8000, reload=True)
