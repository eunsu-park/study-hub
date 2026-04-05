#!/bin/bash
# Exercises for Lesson 08: Error Handling
# Topic: API_Design
# Solutions to practice problems from the lesson.

# === Exercise 1: RFC 7807 Problem Details Implementation ===
# Problem: Create a complete error handling system using RFC 7807 Problem Details
# with custom error types and consistent formatting.
exercise_1() {
    echo "=== Exercise 1: RFC 7807 Problem Details Implementation ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, Any
from datetime import datetime, timezone

app = FastAPI()


class ProblemDetail(BaseModel):
    """RFC 7807 Problem Details for HTTP APIs.

    Required fields: type, title, status
    Optional fields: detail, instance
    Extension fields: any additional members
    """
    type: str = "about:blank"     # URI identifying the problem type
    title: str                     # Short, human-readable summary
    status: int                    # HTTP status code
    detail: Optional[str] = None   # Explanation specific to this occurrence
    instance: Optional[str] = None # URI identifying this specific occurrence
    # Extension members
    timestamp: Optional[str] = None
    trace_id: Optional[str] = None
    errors: Optional[list[dict[str, Any]]] = None


def problem_response(
    status_code: int,
    error_type: str,
    title: str,
    detail: str,
    request: Request,
    errors: list[dict] = None,
) -> JSONResponse:
    """Build a Problem Details JSON response."""
    import uuid
    problem = ProblemDetail(
        type=f"https://api.example.com/problems/{error_type}",
        title=title,
        status=status_code,
        detail=detail,
        instance=str(request.url),
        timestamp=datetime.now(timezone.utc).isoformat(),
        trace_id=str(uuid.uuid4())[:8],
        errors=errors,
    )
    return JSONResponse(
        status_code=status_code,
        content=problem.model_dump(exclude_none=True),
        media_type="application/problem+json",
    )


# Example error responses:

# 404 Not Found:
# {
#   "type": "https://api.example.com/problems/not-found",
#   "title": "Resource Not Found",
#   "status": 404,
#   "detail": "User with id 'abc123' does not exist",
#   "instance": "/api/v1/users/abc123",
#   "timestamp": "2025-06-15T10:30:00Z",
#   "trace_id": "a1b2c3d4"
# }

# 422 Validation Error:
# {
#   "type": "https://api.example.com/problems/validation-error",
#   "title": "Validation Error",
#   "status": 422,
#   "detail": "2 validation error(s) in request",
#   "errors": [
#     {"field": "email", "message": "Not a valid email address"},
#     {"field": "age", "message": "Must be at least 18"}
#   ]
# }
SOLUTION
}

# === Exercise 2: Custom Exception Hierarchy ===
# Problem: Build a domain exception hierarchy that maps cleanly to HTTP
# error responses with automatic conversion.
exercise_2() {
    echo "=== Exercise 2: Custom Exception Hierarchy ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from datetime import datetime, timezone


# --- Exception Hierarchy ---

class APIError(Exception):
    """Base for all API errors. Maps to HTTP Problem Details."""
    status_code: int = 500
    error_type: str = "internal-error"
    title: str = "Internal Server Error"

    def __init__(self, detail: str, **extra):
        self.detail = detail
        self.extra = extra
        super().__init__(detail)


class NotFoundError(APIError):
    status_code = 404
    error_type = "not-found"
    title = "Resource Not Found"

    def __init__(self, resource: str, resource_id: str):
        super().__init__(
            detail=f"{resource} with id '{resource_id}' does not exist",
            resource=resource,
            resource_id=resource_id,
        )


class ValidationError(APIError):
    status_code = 422
    error_type = "validation-error"
    title = "Validation Error"

    def __init__(self, errors: list[dict]):
        super().__init__(
            detail=f"{len(errors)} validation error(s)",
            errors=errors,
        )


class ConflictError(APIError):
    status_code = 409
    error_type = "conflict"
    title = "Resource Conflict"


class ForbiddenError(APIError):
    status_code = 403
    error_type = "forbidden"
    title = "Access Denied"


class BusinessRuleError(APIError):
    status_code = 422
    error_type = "business-rule-violation"
    title = "Business Rule Violation"


# --- FastAPI Integration ---

app = FastAPI()


@app.exception_handler(APIError)
async def api_error_handler(request: Request, exc: APIError) -> JSONResponse:
    """Automatically convert any APIError to RFC 7807 Problem Details."""
    body = {
        "type": f"https://api.example.com/problems/{exc.error_type}",
        "title": exc.title,
        "status": exc.status_code,
        "detail": exc.detail,
        "instance": str(request.url),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    # Add extension fields from exc.extra
    body.update({k: v for k, v in exc.extra.items() if v is not None})
    return JSONResponse(
        status_code=exc.status_code,
        content=body,
        media_type="application/problem+json",
    )


# --- Usage in routes ---

@app.get("/api/v1/users/{user_id}")
def get_user(user_id: str):
    raise NotFoundError("User", user_id)


@app.post("/api/v1/orders/{order_id}/cancel")
def cancel_order(order_id: str):
    raise BusinessRuleError("Order has already been shipped and cannot be cancelled")
SOLUTION
}

# === Exercise 3: Validation Error Formatting ===
# Problem: Format Pydantic validation errors into a user-friendly structure
# with field paths and human-readable messages.
exercise_3() {
    echo "=== Exercise 3: Validation Error Formatting ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from typing import Optional
from datetime import datetime, timezone

app = FastAPI()


@app.exception_handler(RequestValidationError)
async def validation_handler(request: Request, exc: RequestValidationError):
    """Convert Pydantic errors to a clean, user-friendly format.

    Default FastAPI format:
    {"detail": [{"loc": ["body", "email"], "msg": "...", "type": "..."}]}

    Our format (RFC 7807 with errors extension):
    {"type": "...", "title": "Validation Error", "errors": [
        {"field": "email", "message": "Not a valid email", "code": "value_error"}
    ]}
    """
    errors = []
    for error in exc.errors():
        # Build field path: ["body", "address", "zip"] → "address.zip"
        loc = error["loc"]
        field_parts = [str(part) for part in loc if part != "body"]
        field_path = ".".join(field_parts) if field_parts else "request"

        errors.append({
            "field": field_path,
            "message": error["msg"],
            "code": error["type"],
            "input": error.get("input"),
        })

    return JSONResponse(
        status_code=422,
        content={
            "type": "https://api.example.com/problems/validation-error",
            "title": "Validation Error",
            "status": 422,
            "detail": f"Request contains {len(errors)} invalid field(s)",
            "instance": str(request.url),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "errors": errors,
        },
        media_type="application/problem+json",
    )


class Address(BaseModel):
    street: str = Field(..., min_length=1)
    city: str = Field(..., min_length=1)
    zip_code: str = Field(..., pattern=r"^\d{5}$")


class UserCreate(BaseModel):
    name: str = Field(..., min_length=2, max_length=100)
    email: str = Field(..., pattern=r"^[\w.-]+@[\w.-]+\.\w+$")
    age: int = Field(..., ge=18, le=150)
    address: Address

    @field_validator("name")
    @classmethod
    def name_not_numeric(cls, v):
        if v.isdigit():
            raise ValueError("Name cannot be purely numeric")
        return v


@app.post("/api/v1/users")
def create_user(body: UserCreate):
    return {"id": "1", **body.model_dump()}

# POST /api/v1/users with invalid data:
# {"name": "A", "email": "not-email", "age": 15, "address": {"zip_code": "abc"}}
#
# Response:
# {
#   "type": ".../validation-error",
#   "title": "Validation Error",
#   "status": 422,
#   "detail": "Request contains 4 invalid field(s)",
#   "errors": [
#     {"field": "name", "message": "String should have at least 2 characters", ...},
#     {"field": "email", "message": "String should match pattern ...", ...},
#     {"field": "age", "message": "Input should be >= 18", ...},
#     {"field": "address.zip_code", "message": "String should match pattern ...", ...}
#   ]
# }
SOLUTION
}

# === Exercise 4: Error Response Testing ===
# Problem: Write tests that verify error responses have the correct format,
# status codes, and error details.
exercise_4() {
    echo "=== Exercise 4: Error Response Testing ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
import pytest
from fastapi.testclient import TestClient
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI()


class ItemCreate(BaseModel):
    name: str = Field(..., min_length=1)
    price: float = Field(..., gt=0)


items = {"1": {"id": "1", "name": "Widget", "price": 9.99}}


@app.get("/api/v1/items/{item_id}")
def get_item(item_id: str):
    if item_id not in items:
        raise HTTPException(status_code=404, detail=f"Item '{item_id}' not found")
    return items[item_id]


@app.post("/api/v1/items", status_code=201)
def create_item(body: ItemCreate):
    return {"id": "2", **body.model_dump()}


client = TestClient(app)


class TestErrorResponses:
    def test_404_has_correct_status(self):
        response = client.get("/api/v1/items/nonexistent")
        assert response.status_code == 404

    def test_404_has_detail_message(self):
        response = client.get("/api/v1/items/nonexistent")
        data = response.json()
        assert "detail" in data
        assert "nonexistent" in data["detail"]

    def test_422_on_invalid_body(self):
        response = client.post("/api/v1/items", json={"name": "", "price": -5})
        assert response.status_code == 422

    def test_422_lists_all_field_errors(self):
        response = client.post("/api/v1/items", json={"name": "", "price": -5})
        data = response.json()
        # Should report errors for both fields
        assert "detail" in data
        error_fields = [e["loc"][-1] for e in data["detail"]]
        assert "name" in error_fields
        assert "price" in error_fields

    def test_error_response_is_json(self):
        response = client.get("/api/v1/items/nonexistent")
        assert response.headers["content-type"].startswith("application/json")

    def test_successful_request_has_no_error_fields(self):
        response = client.get("/api/v1/items/1")
        assert response.status_code == 200
        data = response.json()
        assert "error" not in data
        assert "detail" not in data
SOLUTION
}

# Run all exercises
echo "Exercise solutions for Lesson 08: Error Handling"
echo "=================================================="
exercise_1
echo ""
exercise_2
echo ""
exercise_3
echo ""
exercise_4
