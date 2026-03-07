#!/usr/bin/env python3
"""Example: OpenAPI Specification and Documentation

Demonstrates FastAPI's OpenAPI integration with:
- Custom OpenAPI schema customization
- Rich request/response examples
- Tags and grouping
- Schema descriptions and documentation annotations
- Custom Swagger UI and ReDoc configuration

Related lesson: 10_API_Documentation.md

Run:
    pip install "fastapi[standard]"
    uvicorn 06_openapi_spec:app --reload --port 8000
    # Visit http://localhost:8000/docs for Swagger UI
    # Visit http://localhost:8000/redoc for ReDoc
    # Visit http://localhost:8000/openapi.json for raw spec
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Optional
from uuid import uuid4

from fastapi import FastAPI, Path, Query, status
from pydantic import BaseModel, Field

# =============================================================================
# TAG METADATA — Group and document your endpoints
# =============================================================================
# Tags organize endpoints in the documentation. Add descriptions and external
# links to help consumers understand each section of your API.

tags_metadata = [
    {
        "name": "Tasks",
        "description": "CRUD operations for task management. Tasks represent work "
        "items that can be assigned, prioritized, and tracked through "
        "a lifecycle from **open** to **closed**.",
        "externalDocs": {
            "description": "Task lifecycle documentation",
            "url": "https://example.com/docs/tasks",
        },
    },
    {
        "name": "Health",
        "description": "Service health and readiness probes for orchestration platforms.",
    },
]


# =============================================================================
# APPLICATION SETUP — Custom metadata enriches the generated spec
# =============================================================================

app = FastAPI(
    title="Task Manager API",
    version="2.1.0",
    summary="A production-grade task management API demonstrating OpenAPI best practices.",
    description="""
## Overview

The Task Manager API provides full CRUD operations for managing tasks with
priority levels, status tracking, and assignee management.

## Authentication

All endpoints require a Bearer token in the `Authorization` header:

```
Authorization: Bearer <your-token>
```

## Rate Limits

| Plan    | Requests/min |
|---------|-------------|
| Free    | 60          |
| Pro     | 600         |
| Enterprise | 6000    |

## Changelog

- **v2.1.0** — Added task filtering by status and priority
- **v2.0.0** — Breaking: changed task ID from integer to UUID
- **v1.0.0** — Initial release
""",
    terms_of_service="https://example.com/terms",
    contact={
        "name": "API Support",
        "url": "https://example.com/support",
        "email": "api-support@example.com",
    },
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT",
    },
    openapi_tags=tags_metadata,
    servers=[
        {"url": "http://localhost:8000", "description": "Development"},
        {"url": "https://staging-api.example.com", "description": "Staging"},
        {"url": "https://api.example.com", "description": "Production"},
    ],
)


# =============================================================================
# ENUMS — Documented choices in the OpenAPI spec
# =============================================================================

class TaskStatus(str, Enum):
    """Task lifecycle states."""
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    REVIEW = "review"
    CLOSED = "closed"


class TaskPriority(str, Enum):
    """Task urgency levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# =============================================================================
# SCHEMAS — Rich Pydantic models with OpenAPI examples
# =============================================================================

class TaskCreate(BaseModel):
    """Schema for creating a new task."""
    title: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Short, descriptive title for the task",
        examples=["Implement user authentication"],
    )
    description: Optional[str] = Field(
        None,
        max_length=5000,
        description="Detailed description in Markdown format",
        examples=["Add JWT-based auth with refresh tokens and RBAC"],
    )
    priority: TaskPriority = Field(
        TaskPriority.MEDIUM,
        description="Urgency level determining processing order",
    )
    assignee: Optional[str] = Field(
        None,
        description="Username of the assigned team member",
        examples=["alice"],
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Categorization labels",
        examples=[["backend", "auth", "security"]],
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "title": "Implement user authentication",
                    "description": "Add JWT-based auth with refresh tokens and RBAC",
                    "priority": "high",
                    "assignee": "alice",
                    "tags": ["backend", "auth", "security"],
                },
                {
                    "title": "Fix pagination bug",
                    "description": "Cursor pagination returns duplicates on page boundary",
                    "priority": "critical",
                    "assignee": "bob",
                    "tags": ["backend", "bug"],
                },
            ]
        }
    }


class TaskResponse(BaseModel):
    """Schema for task responses — includes server-generated fields."""
    id: str = Field(..., description="Unique task identifier (UUID)")
    title: str
    description: Optional[str] = None
    status: TaskStatus = Field(..., description="Current lifecycle state")
    priority: TaskPriority
    assignee: Optional[str] = None
    tags: list[str] = []
    created_at: str = Field(..., description="ISO 8601 creation timestamp")
    updated_at: str = Field(..., description="ISO 8601 last-update timestamp")


class TaskListResponse(BaseModel):
    """Paginated list of tasks with metadata."""
    data: list[TaskResponse]
    total: int = Field(..., description="Total number of matching tasks")
    page: int = Field(..., description="Current page number")
    per_page: int = Field(..., description="Number of items per page")


class ErrorResponse(BaseModel):
    """RFC 7807 Problem Details error format."""
    type: str = Field(..., examples=["https://api.example.com/problems/not-found"])
    title: str = Field(..., examples=["Resource Not Found"])
    status: int = Field(..., examples=[404])
    detail: str = Field(..., examples=["Task with id '...' does not exist"])


# =============================================================================
# IN-MEMORY STORE
# =============================================================================

tasks_db: dict[str, dict] = {}


# =============================================================================
# ROUTES — Fully documented endpoints
# =============================================================================

@app.get(
    "/api/v1/tasks",
    response_model=TaskListResponse,
    tags=["Tasks"],
    summary="List tasks with filtering",
    response_description="Paginated list of tasks matching the filter criteria",
)
def list_tasks(
    status_filter: Optional[TaskStatus] = Query(
        None,
        alias="status",
        description="Filter by task status",
    ),
    priority: Optional[TaskPriority] = Query(
        None,
        description="Filter by priority level",
    ),
    assignee: Optional[str] = Query(
        None,
        description="Filter by assignee username",
    ),
    page: int = Query(1, ge=1, description="Page number (1-indexed)"),
    per_page: int = Query(20, ge=1, le=100, description="Items per page"),
):
    """Retrieve a paginated, filterable list of tasks.

    Supports filtering by **status**, **priority**, and **assignee**. All filters
    are optional and can be combined. Results are ordered by creation date
    (newest first).

    ### Examples

    - All open tasks: `GET /api/v1/tasks?status=open`
    - Critical tasks assigned to Alice: `GET /api/v1/tasks?priority=critical&assignee=alice`
    - Page 2 with 10 items: `GET /api/v1/tasks?page=2&per_page=10`
    """
    tasks = list(tasks_db.values())

    # Apply filters
    if status_filter:
        tasks = [t for t in tasks if t["status"] == status_filter.value]
    if priority:
        tasks = [t for t in tasks if t["priority"] == priority.value]
    if assignee:
        tasks = [t for t in tasks if t.get("assignee") == assignee]

    total = len(tasks)
    offset = (page - 1) * per_page
    page_items = tasks[offset : offset + per_page]

    return TaskListResponse(
        data=[TaskResponse(**t) for t in page_items],
        total=total,
        page=page,
        per_page=per_page,
    )


@app.post(
    "/api/v1/tasks",
    response_model=TaskResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Tasks"],
    summary="Create a new task",
    response_description="The newly created task with server-generated fields",
    responses={
        201: {
            "description": "Task created successfully",
            "content": {
                "application/json": {
                    "example": {
                        "id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
                        "title": "Implement user authentication",
                        "description": "Add JWT-based auth",
                        "status": "open",
                        "priority": "high",
                        "assignee": "alice",
                        "tags": ["backend", "auth"],
                        "created_at": "2025-06-15T10:30:00Z",
                        "updated_at": "2025-06-15T10:30:00Z",
                    }
                }
            },
        },
        422: {
            "model": ErrorResponse,
            "description": "Validation error in request body",
        },
    },
)
def create_task(body: TaskCreate):
    """Create a new task with the specified properties.

    The task is created in the **open** status. Use the update endpoint
    to transition it through the lifecycle: `open -> in_progress -> review -> closed`.
    """
    task_id = str(uuid4())
    now = datetime.now(timezone.utc).isoformat()
    task = {
        "id": task_id,
        "title": body.title,
        "description": body.description,
        "status": TaskStatus.OPEN.value,
        "priority": body.priority.value,
        "assignee": body.assignee,
        "tags": body.tags,
        "created_at": now,
        "updated_at": now,
    }
    tasks_db[task_id] = task
    return TaskResponse(**task)


@app.get(
    "/api/v1/tasks/{task_id}",
    response_model=TaskResponse,
    tags=["Tasks"],
    summary="Get a task by ID",
    responses={
        200: {"description": "Task found"},
        404: {
            "model": ErrorResponse,
            "description": "Task not found",
            "content": {
                "application/json": {
                    "example": {
                        "type": "https://api.example.com/problems/not-found",
                        "title": "Resource Not Found",
                        "status": 404,
                        "detail": "Task with id 'abc123' does not exist",
                    }
                }
            },
        },
    },
)
def get_task(
    task_id: str = Path(
        ...,
        description="Unique task identifier (UUID format)",
        examples=["a1b2c3d4-e5f6-7890-abcd-ef1234567890"],
    ),
):
    """Retrieve a single task by its unique identifier."""
    task = tasks_db.get(task_id)
    if not task:
        return {"type": "not-found", "title": "Not Found", "status": 404, "detail": f"Task {task_id} not found"}
    return TaskResponse(**task)


# =============================================================================
# HEALTH ENDPOINTS
# =============================================================================

@app.get(
    "/health",
    tags=["Health"],
    summary="Liveness probe",
    response_description="Service is running",
)
def health():
    """Basic health check — returns 200 if the service is running.

    Use this for Kubernetes liveness probes or load balancer health checks.
    """
    return {"status": "healthy", "timestamp": datetime.now(timezone.utc).isoformat()}


@app.get(
    "/ready",
    tags=["Health"],
    summary="Readiness probe",
    response_description="Service is ready to accept traffic",
)
def readiness():
    """Readiness check — verifies the service can handle requests.

    In production, this should also verify database connectivity,
    cache availability, and other critical dependencies.
    """
    return {
        "status": "ready",
        "checks": {
            "database": "connected",
            "cache": "connected",
        },
    }


# =============================================================================
# CUSTOM OPENAPI SCHEMA
# =============================================================================

def custom_openapi():
    """Customize the generated OpenAPI schema.

    This function is called once and cached. Use it to add global security
    schemes, modify the info section, or add custom extensions.
    """
    if app.openapi_schema:
        return app.openapi_schema

    from fastapi.openapi.utils import get_openapi

    schema = get_openapi(
        title=app.title,
        version=app.version,
        summary=app.summary,
        description=app.description,
        routes=app.routes,
        tags=tags_metadata,
        servers=app.servers,
        terms_of_service=app.terms_of_service,
        contact=app.contact,
        license_info=app.license_info,
    )

    # Add security scheme
    schema["components"]["securitySchemes"] = {
        "BearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT",
            "description": "Enter your JWT token",
        },
        "ApiKeyAuth": {
            "type": "apiKey",
            "in": "header",
            "name": "X-API-Key",
            "description": "API key for server-to-server communication",
        },
    }

    # Apply security globally
    schema["security"] = [{"BearerAuth": []}, {"ApiKeyAuth": []}]

    # Add custom extension for SDK generation hints
    schema["x-sdk-languages"] = ["python", "typescript", "go"]

    app.openapi_schema = schema
    return schema


app.openapi = custom_openapi


# =============================================================================
# ENTRYPOINT
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("06_openapi_spec:app", host="127.0.0.1", port=8000, reload=True)
