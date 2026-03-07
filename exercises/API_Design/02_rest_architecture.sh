#!/bin/bash
# Exercises for Lesson 02: REST Architecture
# Topic: API_Design
# Solutions to practice problems from the lesson.

# === Exercise 1: Richardson Maturity Model Classification ===
# Problem: Classify the following API examples by Richardson Maturity Level
# (Level 0 through Level 3).
exercise_1() {
    echo "=== Exercise 1: Richardson Maturity Model Classification ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
# Richardson Maturity Model Levels:
# Level 0 — The Swamp of POX (single endpoint, single verb)
# Level 1 — Resources (multiple endpoints, but one verb)
# Level 2 — HTTP Verbs (proper use of GET, POST, PUT, DELETE + status codes)
# Level 3 — Hypermedia Controls (HATEOAS — responses contain navigation links)

# --- Level 0: Single endpoint, RPC-style ---
# POST /api
# Body: {"action": "getUser", "userId": 123}
# POST /api
# Body: {"action": "createOrder", "product": "Widget"}
# Everything goes through one URL, one HTTP method.

# --- Level 1: Resources, but only POST ---
# POST /api/users         (to get users)
# POST /api/users/123     (to get one user)
# POST /api/orders        (to create an order)
# Different URLs but always POST — does not leverage HTTP semantics.

# --- Level 2: Proper HTTP methods + status codes ---
# GET    /api/users        → 200 OK
# GET    /api/users/123    → 200 OK  or  404 Not Found
# POST   /api/users        → 201 Created
# PUT    /api/users/123    → 200 OK
# DELETE /api/users/123    → 204 No Content
# This is where most production APIs land.

# --- Level 3: HATEOAS (Hypermedia) ---
# GET /api/users/123 → 200 OK
# {
#   "id": 123,
#   "name": "Alice",
#   "_links": {
#     "self":   {"href": "/api/users/123"},
#     "orders": {"href": "/api/users/123/orders"},
#     "update": {"href": "/api/users/123", "method": "PUT"},
#     "delete": {"href": "/api/users/123", "method": "DELETE"}
#   }
# }
# The response tells the client what actions are available.
SOLUTION
}

# === Exercise 2: REST Constraints Checklist ===
# Problem: Audit a sample API against the 6 REST architectural constraints.
exercise_2() {
    echo "=== Exercise 2: REST Constraints Checklist ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
from fastapi import FastAPI, Request, Response
from pydantic import BaseModel

app = FastAPI()

# Constraint 1: Client-Server Separation
# The API server has no knowledge of the client's UI or state.
# The client has no knowledge of the server's database.
# They communicate ONLY through the HTTP interface.

# Constraint 2: Statelessness
# Each request contains ALL information needed to process it.
# The server does not store client session state between requests.
@app.get("/api/v1/orders")
def list_orders(
    # Auth token in header — no server-side session
    # Pagination in query — no server-side cursor state
    authorization: str = "",  # via Header
    page: int = 1,
    per_page: int = 20,
):
    """Every request is self-contained. The server stores no session."""
    return {"data": [], "page": page}


# Constraint 3: Cacheability
# Responses must indicate whether they are cacheable.
@app.get("/api/v1/products/{product_id}")
def get_product(product_id: str, response: Response):
    """Cache-Control header tells clients and proxies how to cache."""
    response.headers["Cache-Control"] = "public, max-age=3600"  # Cache 1 hour
    response.headers["ETag"] = '"v1-abc123"'  # Enables conditional requests
    return {"id": product_id, "name": "Widget", "price": 29.99}


# Constraint 4: Uniform Interface
# - Resource identification via URIs: /api/v1/products/123
# - Manipulation through representations: JSON request/response bodies
# - Self-descriptive messages: Content-Type, Accept headers
# - HATEOAS: Links in responses (see Exercise 1, Level 3)


# Constraint 5: Layered System
# The client does not know if it is talking to the origin server or
# an intermediary (load balancer, CDN, API gateway).
# This is transparent — the API works the same through any proxy.

# Constraint 6: Code on Demand (Optional)
# The server can extend client functionality by sending executable code.
# Example: returning a JavaScript snippet for form validation.
# This constraint is rarely used in modern REST APIs.

# AUDIT RESULT for a sample API:
# | Constraint       | Status  | Notes                        |
# |------------------|---------|------------------------------|
# | Client-Server    | PASS    | Clear separation             |
# | Statelessness    | PASS    | JWT auth, no sessions        |
# | Cacheability     | PARTIAL | Missing Cache-Control on GET |
# | Uniform Interface| PASS    | Consistent JSON, proper URLs |
# | Layered System   | PASS    | Works behind API gateway     |
# | Code on Demand   | N/A     | Optional constraint          |
SOLUTION
}

# === Exercise 3: Implementing HATEOAS Links ===
# Problem: Add HATEOAS links to an order API that shows available
# state transitions based on the current order status.
exercise_3() {
    echo "=== Exercise 3: Implementing HATEOAS Links ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import Optional

app = FastAPI()

orders_db = {
    "1": {"id": "1", "product": "Laptop", "status": "pending", "total": 999.99},
    "2": {"id": "2", "product": "Mouse", "status": "shipped", "total": 29.99},
    "3": {"id": "3", "product": "Keyboard", "status": "delivered", "total": 79.99},
}

# State machine: which transitions are allowed from each status
STATE_TRANSITIONS = {
    "pending":   ["confirm", "cancel"],
    "confirmed": ["ship", "cancel"],
    "shipped":   ["deliver"],
    "delivered": ["return"],
    "cancelled": [],  # Terminal state — no further transitions
    "returned":  [],  # Terminal state
}


def build_order_links(request: Request, order: dict) -> list[dict]:
    """Generate HATEOAS links based on current order state.

    The links tell the client what actions are available RIGHT NOW.
    A 'pending' order can be confirmed or cancelled, but not shipped.
    A 'delivered' order can be returned, but not cancelled.
    """
    base = str(request.base_url).rstrip("/")
    order_id = order["id"]

    links = [
        {"rel": "self", "href": f"{base}/api/v1/orders/{order_id}", "method": "GET"},
    ]

    # Add state transition links based on current status
    allowed = STATE_TRANSITIONS.get(order["status"], [])

    if "confirm" in allowed:
        links.append({
            "rel": "confirm",
            "href": f"{base}/api/v1/orders/{order_id}/confirm",
            "method": "POST",
        })
    if "cancel" in allowed:
        links.append({
            "rel": "cancel",
            "href": f"{base}/api/v1/orders/{order_id}/cancel",
            "method": "POST",
        })
    if "ship" in allowed:
        links.append({
            "rel": "ship",
            "href": f"{base}/api/v1/orders/{order_id}/ship",
            "method": "POST",
        })
    if "deliver" in allowed:
        links.append({
            "rel": "deliver",
            "href": f"{base}/api/v1/orders/{order_id}/deliver",
            "method": "POST",
        })
    if "return" in allowed:
        links.append({
            "rel": "return",
            "href": f"{base}/api/v1/orders/{order_id}/return",
            "method": "POST",
        })

    return links


@app.get("/api/v1/orders/{order_id}")
def get_order(request: Request, order_id: str):
    order = orders_db.get(order_id)
    if not order:
        raise HTTPException(status_code=404, detail="Order not found")

    return {
        "data": order,
        "_links": build_order_links(request, order),
    }

# Example response for a "pending" order:
# {
#   "data": {"id": "1", "product": "Laptop", "status": "pending", "total": 999.99},
#   "_links": [
#     {"rel": "self",    "href": "/api/v1/orders/1",         "method": "GET"},
#     {"rel": "confirm", "href": "/api/v1/orders/1/confirm", "method": "POST"},
#     {"rel": "cancel",  "href": "/api/v1/orders/1/cancel",  "method": "POST"}
#   ]
# }
SOLUTION
}

# === Exercise 4: Statelessness vs Stateful Design ===
# Problem: Refactor a stateful API (server-side sessions) to be stateless.
exercise_4() {
    echo "=== Exercise 4: Statelessness vs Stateful Design ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
# BAD: Stateful — server stores pagination cursor in session
# Problems: does not scale horizontally, session lost on restart

# class StatefulAPI:
#     sessions = {}
#     def list_items(self, session_id):
#         cursor = self.sessions[session_id]["cursor"]  # Server stores state
#         items = db.query(offset=cursor, limit=10)
#         self.sessions[session_id]["cursor"] += 10
#         return items

# GOOD: Stateless — all state is in the request
from fastapi import FastAPI, Query
import jwt
from datetime import datetime, timezone

app = FastAPI()

# Pagination state is in query parameters, not server sessions
@app.get("/api/v1/items")
def list_items(
    cursor: str = Query(None, description="Opaque cursor from previous response"),
    limit: int = Query(20, ge=1, le=100),
):
    """Stateless pagination — cursor encodes the position."""
    # Decode cursor to get offset (client passes it back verbatim)
    offset = 0
    if cursor:
        import base64, json
        offset = json.loads(base64.b64decode(cursor))["offset"]

    # items = db.query(offset=offset, limit=limit)
    items = [{"id": i} for i in range(offset, offset + limit)]

    # Encode next cursor for client
    import base64, json
    next_cursor = base64.b64encode(
        json.dumps({"offset": offset + limit}).encode()
    ).decode()

    return {
        "data": items,
        "next_cursor": next_cursor,
        "has_more": True,
    }


# Authentication state is in the JWT token, not server sessions
@app.get("/api/v1/profile")
def get_profile(authorization: str = ""):
    """Stateless auth — user identity is in the JWT, not a session cookie."""
    # token = authorization.replace("Bearer ", "")
    # payload = jwt.decode(token, SECRET, algorithms=["HS256"])
    # user_id = payload["sub"]   # User ID embedded in token
    # role = payload["role"]     # Permissions embedded in token
    # No server-side session lookup needed!
    return {"user_id": "from_jwt", "role": "from_jwt"}

# Benefits of statelessness:
# 1. Horizontal scaling: any server can handle any request
# 2. Reliability: no session state to lose on server crash
# 3. Cacheability: identical requests always produce same results
# 4. Simplicity: no session store infrastructure (Redis, etc.)
SOLUTION
}

# Run all exercises
echo "Exercise solutions for Lesson 02: REST Architecture"
echo "===================================================="
exercise_1
echo ""
exercise_2
echo ""
exercise_3
echo ""
exercise_4
