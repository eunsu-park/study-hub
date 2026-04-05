#!/bin/bash
# Exercises for Lesson 10: API Documentation
# Topic: API_Design
# Solutions to practice problems from the lesson.

# === Exercise 1: OpenAPI Schema with Rich Examples ===
# Problem: Write a FastAPI endpoint with comprehensive OpenAPI documentation
# including examples, descriptions, and response schemas.
exercise_1() {
    echo "=== Exercise 1: OpenAPI Schema with Rich Examples ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
from fastapi import FastAPI, Path, Query, Body, status
from pydantic import BaseModel, Field
from typing import Optional

app = FastAPI(
    title="Invoice API",
    version="1.0.0",
    description="API for managing invoices with detailed documentation.",
)


class LineItem(BaseModel):
    """A single line item on an invoice."""
    description: str = Field(..., examples=["Web development services"])
    quantity: int = Field(..., ge=1, examples=[10])
    unit_price: float = Field(..., gt=0, examples=[150.0])
    amount: float = Field(..., examples=[1500.0])


class InvoiceCreate(BaseModel):
    """Request body for creating a new invoice."""
    customer_name: str = Field(
        ...,
        min_length=1,
        description="Full legal name of the customer",
        examples=["Acme Corp"],
    )
    customer_email: str = Field(
        ...,
        description="Email for sending the invoice",
        examples=["billing@acme.com"],
    )
    line_items: list[LineItem] = Field(
        ...,
        min_length=1,
        description="At least one line item is required",
    )
    notes: Optional[str] = Field(
        None,
        description="Optional notes visible to the customer",
        examples=["Payment due within 30 days"],
    )
    due_days: int = Field(
        30,
        ge=1,
        le=365,
        description="Number of days until invoice is due",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "customer_name": "Acme Corp",
                    "customer_email": "billing@acme.com",
                    "line_items": [
                        {"description": "Web development", "quantity": 10,
                         "unit_price": 150.0, "amount": 1500.0},
                        {"description": "Code review", "quantity": 5,
                         "unit_price": 100.0, "amount": 500.0},
                    ],
                    "notes": "Payment due within 30 days",
                    "due_days": 30,
                }
            ]
        }
    }


class InvoiceResponse(BaseModel):
    id: str = Field(..., description="Unique invoice identifier")
    status: str = Field(..., description="Invoice status: draft, sent, paid, overdue")
    customer_name: str
    total: float = Field(..., description="Total amount in USD")
    created_at: str
    due_date: str


class ErrorResponse(BaseModel):
    type: str
    title: str
    status: int
    detail: str


@app.post(
    "/api/v1/invoices",
    response_model=InvoiceResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new invoice",
    description="Creates an invoice in **draft** status. Send it using the "
    "`POST /api/v1/invoices/{id}/send` endpoint.",
    response_description="The created invoice with server-generated fields",
    tags=["Invoices"],
    responses={
        201: {
            "description": "Invoice created successfully",
            "content": {
                "application/json": {
                    "example": {
                        "id": "inv_abc123",
                        "status": "draft",
                        "customer_name": "Acme Corp",
                        "total": 2000.0,
                        "created_at": "2025-06-15T10:00:00Z",
                        "due_date": "2025-07-15T10:00:00Z",
                    }
                }
            },
        },
        422: {
            "model": ErrorResponse,
            "description": "Validation error — check the `errors` field for details",
        },
    },
)
def create_invoice(body: InvoiceCreate):
    """Create a new invoice from line items.

    The invoice is created in **draft** status. To send it to the customer,
    use the send endpoint. The total is calculated server-side from line items.
    """
    total = sum(item.amount for item in body.line_items)
    return InvoiceResponse(
        id="inv_abc123",
        status="draft",
        customer_name=body.customer_name,
        total=total,
        created_at="2025-06-15T10:00:00Z",
        due_date="2025-07-15T10:00:00Z",
    )
SOLUTION
}

# === Exercise 2: API Documentation Tags and Grouping ===
# Problem: Organize a large API into logical groups using tags with
# descriptions and external documentation links.
exercise_2() {
    echo "=== Exercise 2: API Documentation Tags and Grouping ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
from fastapi import FastAPI

tags_metadata = [
    {
        "name": "Authentication",
        "description": "Register, login, refresh tokens, and manage sessions. "
        "All endpoints return JWT tokens.",
        "externalDocs": {
            "description": "Authentication guide",
            "url": "https://docs.example.com/auth",
        },
    },
    {
        "name": "Users",
        "description": "User profile management. Requires authentication.",
    },
    {
        "name": "Projects",
        "description": "CRUD operations for projects. Projects contain tasks "
        "and have team members with role-based access.",
        "externalDocs": {
            "description": "Project permissions model",
            "url": "https://docs.example.com/projects/permissions",
        },
    },
    {
        "name": "Tasks",
        "description": "Task management within projects. Tasks support "
        "assignment, status transitions, and comments.",
    },
    {
        "name": "Webhooks",
        "description": "Register webhook endpoints to receive real-time "
        "notifications about events (task.created, project.updated, etc.).",
    },
    {
        "name": "Admin",
        "description": "Administrative operations. Requires `admin` role.",
    },
]

app = FastAPI(
    title="Project Management API",
    openapi_tags=tags_metadata,
)

# Endpoints are grouped by their tag
@app.post("/api/v1/auth/login", tags=["Authentication"])
def login(): ...

@app.get("/api/v1/users/me", tags=["Users"])
def get_profile(): ...

@app.get("/api/v1/projects", tags=["Projects"])
def list_projects(): ...

@app.get("/api/v1/projects/{id}/tasks", tags=["Tasks"])
def list_tasks(id: str): ...

@app.post("/api/v1/webhooks", tags=["Webhooks"])
def register_webhook(): ...

@app.get("/api/v1/admin/users", tags=["Admin"])
def admin_list_users(): ...

# Visit /docs to see endpoints grouped by these tags with descriptions
SOLUTION
}

# === Exercise 3: Documentation-Driven Design ===
# Problem: Write an OpenAPI spec first, then implement the FastAPI endpoints
# to match the spec exactly.
exercise_3() {
    echo "=== Exercise 3: Documentation-Driven Design ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
# Step 1: Write the OpenAPI spec first (YAML)
OPENAPI_SPEC = """
openapi: "3.1.0"
info:
  title: Notification API
  version: "1.0.0"
  description: API for sending and managing notifications.
paths:
  /api/v1/notifications:
    get:
      summary: List notifications
      parameters:
        - name: status
          in: query
          schema:
            type: string
            enum: [unread, read, all]
          required: false
        - name: limit
          in: query
          schema:
            type: integer
            minimum: 1
            maximum: 100
            default: 20
      responses:
        "200":
          description: List of notifications
    post:
      summary: Send a notification
      requestBody:
        required: true
        content:
          application/json:
            schema:
              type: object
              required: [recipient_id, title, message]
              properties:
                recipient_id:
                  type: string
                title:
                  type: string
                  maxLength: 200
                message:
                  type: string
                  maxLength: 5000
                channel:
                  type: string
                  enum: [email, push, sms]
                  default: push
      responses:
        "201":
          description: Notification sent
  /api/v1/notifications/{id}/read:
    post:
      summary: Mark notification as read
      responses:
        "200":
          description: Notification marked as read
"""

# Step 2: Implement to match the spec exactly
from fastapi import FastAPI, Query
from pydantic import BaseModel, Field
from typing import Optional

app = FastAPI(title="Notification API", version="1.0.0")


class NotificationCreate(BaseModel):
    recipient_id: str
    title: str = Field(..., max_length=200)
    message: str = Field(..., max_length=5000)
    channel: str = Field("push", pattern="^(email|push|sms)$")


@app.get("/api/v1/notifications")
def list_notifications(
    status: Optional[str] = Query(None, pattern="^(unread|read|all)$"),
    limit: int = Query(20, ge=1, le=100),
):
    return {"data": [], "total": 0}


@app.post("/api/v1/notifications", status_code=201)
def send_notification(body: NotificationCreate):
    return {"id": "notif_1", **body.model_dump(), "status": "sent"}


@app.post("/api/v1/notifications/{notif_id}/read")
def mark_as_read(notif_id: str):
    return {"id": notif_id, "status": "read"}

# Step 3: Validate implementation matches spec (automated)
# Tools: openapi-spec-validator, schemathesis
# schemathesis run --url http://localhost:8000/openapi.json
SOLUTION
}

# === Exercise 4: Generating Client SDKs from OpenAPI ===
# Problem: Use the OpenAPI spec to generate a typed Python client SDK
# and demonstrate how it simplifies API consumption.
exercise_4() {
    echo "=== Exercise 4: Generating Client SDKs from OpenAPI ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
# SDK generation workflow:
# 1. Export OpenAPI spec: curl http://localhost:8000/openapi.json > spec.json
# 2. Generate SDK: openapi-python-client generate --path spec.json
# 3. Install SDK: pip install ./generated-client

# The generated SDK provides typed methods for every endpoint:

# --- Without SDK (raw HTTP) ---
import httpx

# Manual: construct URL, headers, parse JSON, handle errors
response = httpx.get(
    "https://api.example.com/api/v1/notifications",
    params={"status": "unread", "limit": 10},
    headers={"Authorization": "Bearer token123"},
)
data = response.json()  # Untyped dict
notifications = data["data"]  # No autocomplete, no type hints

# --- With Generated SDK ---
# from notification_client import Client
# from notification_client.models import NotificationCreate
#
# client = Client(base_url="https://api.example.com", token="token123")
#
# # Typed method with autocomplete
# notifications = client.list_notifications(status="unread", limit=10)
# # Returns List[Notification] — full type hints!
#
# # Create notification with type-checked input
# result = client.send_notification(
#     NotificationCreate(
#         recipient_id="user_1",
#         title="New message",
#         message="You have a new message from Alice",
#         channel="push",
#     )
# )
# # result.id is str, result.status is str — all typed

# SDK Generation Tools:
# | Tool                    | Languages           | Spec Support  |
# |-------------------------|---------------------|---------------|
# | openapi-python-client   | Python              | OpenAPI 3.x   |
# | openapi-generator       | 50+ languages       | OpenAPI 2/3   |
# | openapi-typescript-fetch| TypeScript (fetch)  | OpenAPI 3.x   |
# | Kiota (Microsoft)       | C#, Go, Java, etc.  | OpenAPI 3.x   |
# | Stainless               | Python, TS, Go, etc.| OpenAPI 3.x   |

# BEST PRACTICE: Publish your OpenAPI spec alongside your API.
# This enables consumers to generate their own SDKs in any language.
SOLUTION
}

# Run all exercises
echo "Exercise solutions for Lesson 10: API Documentation"
echo "====================================================="
exercise_1
echo ""
exercise_2
echo ""
exercise_3
echo ""
exercise_4
