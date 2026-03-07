#!/bin/bash
# Exercises for Lesson 04: Request and Response Design
# Topic: API_Design
# Solutions to practice problems from the lesson.

# === Exercise 1: HTTP Method Selection ===
# Problem: For each operation, choose the correct HTTP method and explain
# the idempotency and safety characteristics.
exercise_1() {
    echo "=== Exercise 1: HTTP Method Selection ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
# HTTP Method Properties:
# | Method | Safe? | Idempotent? | Request Body? | Typical Use        |
# |--------|-------|-------------|---------------|--------------------|
# | GET    | Yes   | Yes         | No            | Read resource      |
# | POST   | No    | No          | Yes           | Create resource    |
# | PUT    | No    | Yes         | Yes           | Full replace       |
# | PATCH  | No    | No*         | Yes           | Partial update     |
# | DELETE | No    | Yes         | Optional      | Remove resource    |
# | HEAD   | Yes   | Yes         | No            | Check existence    |
# | OPTIONS| Yes   | Yes         | No            | Discover methods   |

# Operations and correct method choices:

# 1. "Get user profile"
#    Method: GET /api/v1/users/123
#    Safe: Yes (no side effects), Idempotent: Yes

# 2. "Create a new order"
#    Method: POST /api/v1/orders
#    Safe: No, Idempotent: No (each call creates a new order)
#    TIP: Use Idempotency-Key header to make POST idempotent

# 3. "Replace user profile entirely"
#    Method: PUT /api/v1/users/123
#    Safe: No, Idempotent: Yes (same body = same result)

# 4. "Update only the user's email"
#    Method: PATCH /api/v1/users/123  body: {"email": "new@example.com"}
#    Safe: No, Idempotent: Depends on patch format

# 5. "Delete an order"
#    Method: DELETE /api/v1/orders/456
#    Safe: No, Idempotent: Yes (deleting twice = same result)

# 6. "Check if a resource exists without downloading it"
#    Method: HEAD /api/v1/documents/789
#    Returns headers only (Content-Length, ETag) — no body

# 7. "Transfer money between accounts"
#    Method: POST /api/v1/transfers
#    Why POST not PUT: transfers are not idempotent by nature
#    Use Idempotency-Key: "txn-abc123" to prevent double-charging
SOLUTION
}

# === Exercise 2: Status Code Mapping ===
# Problem: Implement a FastAPI endpoint that returns the correct status code
# for each possible outcome.
exercise_2() {
    echo "=== Exercise 2: Status Code Mapping ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
from fastapi import FastAPI, HTTPException, status, Response
from pydantic import BaseModel
from typing import Optional

app = FastAPI()

# In-memory store
items_db = {"1": {"id": "1", "name": "Widget", "version": 1}}


class ItemCreate(BaseModel):
    name: str
    category: Optional[str] = None


class ItemUpdate(BaseModel):
    name: Optional[str] = None
    category: Optional[str] = None


# 200 OK — Successful read or update
@app.get("/api/v1/items/{item_id}", status_code=status.HTTP_200_OK)
def get_item(item_id: str):
    if item_id not in items_db:
        # 404 Not Found — Resource does not exist
        raise HTTPException(status_code=404, detail="Item not found")
    return items_db[item_id]


# 201 Created — New resource successfully created
@app.post("/api/v1/items", status_code=status.HTTP_201_CREATED)
def create_item(body: ItemCreate, response: Response):
    import uuid
    item_id = str(uuid.uuid4())
    items_db[item_id] = {"id": item_id, **body.model_dump()}
    # Location header points to the new resource
    response.headers["Location"] = f"/api/v1/items/{item_id}"
    return items_db[item_id]


# 204 No Content — Successful operation with no response body
@app.delete("/api/v1/items/{item_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_item(item_id: str):
    if item_id not in items_db:
        raise HTTPException(status_code=404, detail="Item not found")
    del items_db[item_id]
    # No return value — 204 means empty body


# 400 Bad Request — Client sent malformed data
@app.post("/api/v1/items/import")
def import_items(data: list[dict]):
    if not data:
        raise HTTPException(
            status_code=400,
            detail="Request body must be a non-empty array",
        )
    return {"imported": len(data)}


# 409 Conflict — State conflict (e.g., duplicate, version mismatch)
@app.put("/api/v1/items/{item_id}")
def replace_item(item_id: str, body: ItemCreate, if_match: str = ""):
    item = items_db.get(item_id)
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")
    # Optimistic concurrency with ETag
    if if_match and if_match != f'"{item.get("version", 1)}"':
        raise HTTPException(
            status_code=409,
            detail="Version conflict — resource was modified by another client",
        )
    item.update(body.model_dump())
    return item


# Common status codes summary:
# 2xx Success: 200 OK, 201 Created, 204 No Content, 202 Accepted
# 3xx Redirect: 301 Moved Permanently, 304 Not Modified
# 4xx Client Error: 400 Bad Request, 401 Unauthorized, 403 Forbidden,
#                   404 Not Found, 409 Conflict, 422 Unprocessable, 429 Too Many
# 5xx Server Error: 500 Internal, 502 Bad Gateway, 503 Unavailable
SOLUTION
}

# === Exercise 3: Content Negotiation ===
# Problem: Implement an endpoint that returns different formats based on
# the Accept header (JSON and CSV).
exercise_3() {
    echo "=== Exercise 3: Content Negotiation ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
import csv
import io

app = FastAPI()

PRODUCTS = [
    {"id": 1, "name": "Laptop", "price": 999.99, "category": "electronics"},
    {"id": 2, "name": "Headphones", "price": 79.99, "category": "electronics"},
    {"id": 3, "name": "Python Book", "price": 39.99, "category": "books"},
]


@app.get("/api/v1/products")
def list_products(request: Request):
    """Return products in the format requested by the Accept header.

    Content negotiation lets a single endpoint serve multiple formats:
    - Accept: application/json → JSON (default)
    - Accept: text/csv → CSV export
    """
    accept = request.headers.get("accept", "application/json")

    if "text/csv" in accept:
        # CSV format
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=["id", "name", "price", "category"])
        writer.writeheader()
        writer.writerows(PRODUCTS)
        return Response(
            content=output.getvalue(),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=products.csv"},
        )

    # Default: JSON
    return JSONResponse(
        content={"data": PRODUCTS, "total": len(PRODUCTS)},
        media_type="application/json",
    )


# Test with:
# curl -H "Accept: application/json" http://localhost:8000/api/v1/products
# curl -H "Accept: text/csv" http://localhost:8000/api/v1/products

# 406 Not Acceptable — server cannot produce the requested format
@app.get("/api/v1/reports")
def get_report(request: Request):
    accept = request.headers.get("accept", "")
    supported = ["application/json", "text/csv"]
    if accept != "*/*" and not any(fmt in accept for fmt in supported):
        return JSONResponse(
            status_code=406,
            content={"error": f"Supported formats: {', '.join(supported)}"},
        )
    return {"data": "report contents"}
SOLUTION
}

# === Exercise 4: Idempotent POST with Idempotency Key ===
# Problem: Implement idempotent POST using an Idempotency-Key header
# to prevent duplicate payment processing.
exercise_4() {
    echo "=== Exercise 4: Idempotent POST with Idempotency Key ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
from fastapi import FastAPI, Header, HTTPException, status
from pydantic import BaseModel
from typing import Optional

app = FastAPI()

# Store processed idempotency keys and their results
idempotency_store: dict[str, dict] = {}
payments_db: dict[str, dict] = {}


class PaymentRequest(BaseModel):
    amount: float
    currency: str = "USD"
    recipient: str


@app.post("/api/v1/payments", status_code=201)
def create_payment(
    body: PaymentRequest,
    idempotency_key: str = Header(..., alias="Idempotency-Key"),
):
    """Create a payment with idempotency guarantee.

    The Idempotency-Key header ensures that retrying the same request
    (e.g., due to network timeout) does not create duplicate payments.

    Flow:
    1. Client generates a unique key (e.g., UUID) before the first attempt
    2. Client sends the request with the key
    3. If the key was already processed, return the cached response
    4. Otherwise, process the payment and cache the response

    This pattern is used by Stripe, PayPal, and most payment APIs.
    """
    # Check if we already processed this key
    if idempotency_key in idempotency_store:
        cached = idempotency_store[idempotency_key]
        # Return the same response — no duplicate processing
        return cached["response"]

    # Process the payment (first time for this key)
    import uuid
    payment_id = str(uuid.uuid4())
    payment = {
        "id": payment_id,
        "amount": body.amount,
        "currency": body.currency,
        "recipient": body.recipient,
        "status": "completed",
    }
    payments_db[payment_id] = payment

    # Cache the response for this idempotency key
    idempotency_store[idempotency_key] = {
        "response": payment,
        "status_code": 201,
    }

    return payment

# Usage:
# First request:
#   POST /api/v1/payments
#   Idempotency-Key: pay_abc123
#   {"amount": 100, "currency": "USD", "recipient": "merchant_1"}
#   → 201 Created (payment processed)

# Retry (same key):
#   POST /api/v1/payments
#   Idempotency-Key: pay_abc123
#   {"amount": 100, "currency": "USD", "recipient": "merchant_1"}
#   → 201 Created (cached response, NOT a duplicate payment)
SOLUTION
}

# Run all exercises
echo "Exercise solutions for Lesson 04: Request and Response Design"
echo "=============================================================="
exercise_1
echo ""
exercise_2
echo ""
exercise_3
echo ""
exercise_4
