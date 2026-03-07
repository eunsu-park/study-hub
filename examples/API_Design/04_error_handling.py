#!/usr/bin/env python3
"""Example: Error Handling — RFC 7807 Problem Details

Demonstrates standardized API error responses using the Problem Details
specification (RFC 7807 / RFC 9457). Covers:
- Structured error responses with type, title, status, detail, instance
- Validation error formatting
- Custom exception hierarchy
- Error response middleware

Related lesson: 08_Error_Handling.md

Run:
    pip install "fastapi[standard]"
    uvicorn 04_error_handling:app --reload --port 8000

Test:
    http GET :8000/api/v1/products/999
    http POST :8000/api/v1/products title="" price=-5
"""

from datetime import datetime, timezone
from typing import Any, Optional
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# =============================================================================
# RFC 7807 PROBLEM DETAILS
# =============================================================================
# RFC 7807 (superseded by RFC 9457) defines a standard format for HTTP API
# error responses. Instead of each API inventing its own error format, this
# standard provides consistent fields:
#
#   type     — URI reference identifying the error type (machine-readable)
#   title    — Short human-readable summary (same for all instances of this type)
#   status   — HTTP status code
#   detail   — Human-readable explanation specific to this occurrence
#   instance — URI identifying this specific occurrence (for log correlation)
#
# Content-Type: application/problem+json


class ProblemDetail(BaseModel):
    """RFC 7807 Problem Details response body."""
    type: str = "about:blank"
    title: str
    status: int
    detail: str
    instance: Optional[str] = None
    # Extension fields (RFC 7807 allows additional members)
    timestamp: Optional[str] = None
    errors: Optional[list[dict[str, Any]]] = None


# =============================================================================
# CUSTOM EXCEPTION HIERARCHY
# =============================================================================
# Define domain-specific exceptions that map cleanly to HTTP responses.
# This separates business logic from HTTP concerns.

class APIError(Exception):
    """Base exception for all API errors."""
    def __init__(
        self,
        status_code: int,
        error_type: str,
        title: str,
        detail: str,
    ):
        self.status_code = status_code
        self.error_type = error_type
        self.title = title
        self.detail = detail


class NotFoundError(APIError):
    """Resource not found."""
    def __init__(self, resource: str, resource_id: str):
        super().__init__(
            status_code=404,
            error_type="https://api.example.com/problems/not-found",
            title="Resource Not Found",
            detail=f"{resource} with id '{resource_id}' does not exist",
        )
        self.resource = resource
        self.resource_id = resource_id


class ConflictError(APIError):
    """Resource already exists or state conflict."""
    def __init__(self, detail: str):
        super().__init__(
            status_code=409,
            error_type="https://api.example.com/problems/conflict",
            title="Resource Conflict",
            detail=detail,
        )


class BusinessRuleError(APIError):
    """A business rule was violated."""
    def __init__(self, detail: str):
        super().__init__(
            status_code=422,
            error_type="https://api.example.com/problems/business-rule-violation",
            title="Business Rule Violation",
            detail=detail,
        )


class RateLimitError(APIError):
    """Client has exceeded the rate limit."""
    def __init__(self, retry_after: int = 60):
        super().__init__(
            status_code=429,
            error_type="https://api.example.com/problems/rate-limited",
            title="Rate Limit Exceeded",
            detail=f"Too many requests. Retry after {retry_after} seconds.",
        )
        self.retry_after = retry_after


# =============================================================================
# APPLICATION SETUP WITH EXCEPTION HANDLERS
# =============================================================================

app = FastAPI(title="Error Handling API", version="1.0.0")


@app.exception_handler(APIError)
async def api_error_handler(request: Request, exc: APIError) -> JSONResponse:
    """Convert APIError exceptions to RFC 7807 Problem Details responses.

    This handler catches all APIError subclasses and formats them consistently.
    The `instance` field uses the request URL for correlation with server logs.
    """
    problem = ProblemDetail(
        type=exc.error_type,
        title=exc.title,
        status=exc.status_code,
        detail=exc.detail,
        instance=str(request.url),
        timestamp=datetime.now(timezone.utc).isoformat(),
    )

    headers = {}
    if isinstance(exc, RateLimitError):
        headers["Retry-After"] = str(exc.retry_after)

    return JSONResponse(
        status_code=exc.status_code,
        content=problem.model_dump(exclude_none=True),
        headers=headers,
        media_type="application/problem+json",
    )


@app.exception_handler(RequestValidationError)
async def validation_error_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Convert Pydantic validation errors to RFC 7807 format.

    FastAPI's default validation error response is useful but non-standard.
    This handler reformats it as Problem Details with field-level errors
    in the `errors` extension field.
    """
    field_errors = []
    for error in exc.errors():
        field_errors.append({
            "field": ".".join(str(loc) for loc in error["loc"] if loc != "body"),
            "message": error["msg"],
            "type": error["type"],
        })

    problem = ProblemDetail(
        type="https://api.example.com/problems/validation-error",
        title="Validation Error",
        status=422,
        detail=f"{len(field_errors)} validation error(s) in request",
        instance=str(request.url),
        timestamp=datetime.now(timezone.utc).isoformat(),
        errors=field_errors,
    )

    return JSONResponse(
        status_code=422,
        content=problem.model_dump(exclude_none=True),
        media_type="application/problem+json",
    )


@app.exception_handler(Exception)
async def generic_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all for unhandled exceptions.

    IMPORTANT: Never expose internal error details to clients in production.
    Log the full traceback server-side, but return a generic message to the client.
    """
    # In production, log exc here (e.g., Sentry, structured logging)
    problem = ProblemDetail(
        type="https://api.example.com/problems/internal-error",
        title="Internal Server Error",
        status=500,
        detail="An unexpected error occurred. Please try again later.",
        instance=str(request.url),
        timestamp=datetime.now(timezone.utc).isoformat(),
    )

    return JSONResponse(
        status_code=500,
        content=problem.model_dump(exclude_none=True),
        media_type="application/problem+json",
    )


# =============================================================================
# SCHEMAS
# =============================================================================

class ProductCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)
    price: float = Field(..., gt=0, description="Price in USD, must be positive")
    sku: Optional[str] = Field(None, pattern=r"^[A-Z]{2,4}-\d{4,8}$")
    stock: int = Field(0, ge=0)


class ProductResponse(BaseModel):
    id: str
    title: str
    price: float
    sku: Optional[str] = None
    stock: int


# =============================================================================
# IN-MEMORY STORE
# =============================================================================

products_db: dict[str, dict] = {}


# =============================================================================
# ROUTES — Demonstrating error responses
# =============================================================================

@app.post(
    "/api/v1/products",
    response_model=ProductResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Products"],
)
def create_product(body: ProductCreate):
    """Create a product — demonstrates validation errors.

    Try: POST with {"title": "", "price": -5} to see RFC 7807 validation errors.
    Try: POST with duplicate SKU to see ConflictError.
    """
    # Business rule: SKU must be unique
    if body.sku:
        for p in products_db.values():
            if p.get("sku") == body.sku:
                raise ConflictError(f"Product with SKU '{body.sku}' already exists")

    product_id = str(uuid4())
    product = {
        "id": product_id,
        "title": body.title,
        "price": body.price,
        "sku": body.sku,
        "stock": body.stock,
    }
    products_db[product_id] = product
    return ProductResponse(**product)


@app.get("/api/v1/products/{product_id}", response_model=ProductResponse, tags=["Products"])
def get_product(product_id: str):
    """Get a product — demonstrates 404 Not Found with Problem Details."""
    product = products_db.get(product_id)
    if not product:
        raise NotFoundError("Product", product_id)
    return ProductResponse(**product)


@app.post("/api/v1/products/{product_id}/purchase", tags=["Products"])
def purchase_product(product_id: str, quantity: int = 1):
    """Purchase a product — demonstrates business rule violations.

    Try purchasing with quantity > stock to see BusinessRuleError.
    """
    product = products_db.get(product_id)
    if not product:
        raise NotFoundError("Product", product_id)

    if quantity > product["stock"]:
        raise BusinessRuleError(
            f"Insufficient stock for product '{product['title']}'. "
            f"Requested: {quantity}, available: {product['stock']}"
        )

    product["stock"] -= quantity
    return {"message": f"Purchased {quantity} unit(s)", "remaining_stock": product["stock"]}


@app.get("/api/v1/demo/rate-limit", tags=["Demo"])
def demo_rate_limit():
    """Demonstrates a rate limit error response with Retry-After header."""
    raise RateLimitError(retry_after=30)


@app.get("/api/v1/demo/internal-error", tags=["Demo"])
def demo_internal_error():
    """Demonstrates the catch-all error handler for unhandled exceptions."""
    raise RuntimeError("Something broke unexpectedly")


# =============================================================================
# ENTRYPOINT
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("04_error_handling:app", host="127.0.0.1", port=8000, reload=True)
