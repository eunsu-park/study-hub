# 02. FastAPI Basics

**Previous**: [Backend Web Fundamentals](./01_Backend_Web_Fundamentals.md) | **Next**: [FastAPI Advanced](./03_FastAPI_Advanced.md)

**Difficulty**: ⭐⭐

---

## Learning Objectives

- Build a minimal FastAPI application with typed path and query parameters
- Define request and response schemas using Pydantic v2 models with validation
- Explain how FastAPI generates OpenAPI documentation automatically from type hints
- Configure CORS middleware to allow cross-origin requests from frontend applications
- Implement proper HTTP status codes and response models for CRUD endpoints

---

## Table of Contents

1. [What is FastAPI](#1-what-is-fastapi)
2. [Installation and First App](#2-installation-and-first-app)
3. [Path Parameters and Query Parameters](#3-path-parameters-and-query-parameters)
4. [Request Body with Pydantic v2](#4-request-body-with-pydantic-v2)
5. [Response Models and Status Codes](#5-response-models-and-status-codes)
6. [Automatic OpenAPI Documentation](#6-automatic-openapi-documentation)
7. [CORS Middleware](#7-cors-middleware)
8. [Practice Problems](#8-practice-problems)
9. [References](#9-references)

---

## 1. What is FastAPI

FastAPI is a modern Python web framework built on three pillars:

1. **Type hints** (Python 3.10+): Parameters are typed, enabling automatic validation and documentation
2. **Starlette**: The ASGI framework underneath, handling HTTP and WebSocket connections
3. **Pydantic v2**: Data validation and serialization using Python type annotations

```
┌─────────────────────────────────┐
│         Your Application         │
│   (endpoints, business logic)    │
├─────────────────────────────────┤
│           FastAPI                │
│   (routing, DI, OpenAPI gen)     │
├─────────────────────────────────┤
│          Starlette               │
│   (ASGI, middleware, responses)  │
├─────────────────────────────────┤
│      Pydantic v2                 │
│   (validation, serialization)    │
├─────────────────────────────────┤
│    Uvicorn (ASGI server)         │
│   (event loop, HTTP parsing)     │
└─────────────────────────────────┘
```

### Why FastAPI?

| Feature | Flask | Django REST | FastAPI |
|---------|-------|-------------|---------|
| Async support | Limited (with ext.) | Limited | Native |
| Auto validation | No | Serializers | Type hints |
| Auto docs (OpenAPI) | No (ext.) | Yes (ext.) | Built-in |
| Performance | ~1x | ~1x | ~3-5x |
| Learning curve | Low | Medium | Low-Medium |

---

## 2. Installation and First App

### Installation

```bash
# Create a virtual environment first -- isolates project dependencies
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install FastAPI with all optional dependencies (uvicorn, etc.)
pip install "fastapi[standard]"
```

### Minimal Application

```python
# main.py
from fastapi import FastAPI

# Create the application instance. title and version appear in the auto-generated docs.
app = FastAPI(
    title="My First API",
    version="0.1.0",
    description="A simple API to learn FastAPI basics"
)

@app.get("/")
async def root():
    """Root endpoint. FastAPI uses the docstring as the
    endpoint description in the OpenAPI docs."""
    return {"message": "Hello, World!"}

@app.get("/health")
async def health_check():
    """Health check endpoint for load balancers and monitoring.
    Returns 200 if the service is running."""
    return {"status": "healthy"}
```

### Running the Server

```bash
# --reload watches for file changes and restarts automatically
# Only use --reload in development -- it adds overhead
uvicorn main:app --reload --port 8000

# Production: use multiple workers, no reload
# uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

After starting, visit:
- `http://localhost:8000` -- your API
- `http://localhost:8000/docs` -- Swagger UI (interactive documentation)
- `http://localhost:8000/redoc` -- ReDoc (alternative documentation)

---

## 3. Path Parameters and Query Parameters

### Path Parameters

Path parameters are part of the URL and are **required**. FastAPI converts them to the declared type automatically.

```python
from fastapi import FastAPI, Path

app = FastAPI()

@app.get("/users/{user_id}")
async def get_user(
    # Path() adds validation constraints and metadata for docs
    user_id: int = Path(
        ...,  # ... means required (Ellipsis)
        title="User ID",
        description="The unique identifier of the user",
        gt=0,  # greater than 0
        examples=[42]
    )
):
    """Retrieve a user by their ID.
    FastAPI automatically returns 422 if user_id is not a valid int."""
    return {"user_id": user_id, "name": f"User {user_id}"}


# Multiple path parameters with type enforcement
@app.get("/users/{user_id}/posts/{post_id}")
async def get_user_post(user_id: int, post_id: int):
    return {"user_id": user_id, "post_id": post_id}
```

### Predefined Values with Enum

```python
from enum import Enum

class UserRole(str, Enum):
    """Inheriting from str makes the values JSON-serializable
    and allows string comparison in path matching."""
    admin = "admin"
    editor = "editor"
    viewer = "viewer"

@app.get("/users/role/{role}")
async def get_users_by_role(role: UserRole):
    # FastAPI validates that role is one of the enum values
    # Invalid values automatically return 422 with clear error message
    return {"role": role, "message": f"Listing {role.value} users"}
```

### Query Parameters

Query parameters come after `?` in the URL. Parameters not declared in the path are automatically treated as query parameters.

```python
from fastapi import FastAPI, Query

app = FastAPI()

@app.get("/users")
async def list_users(
    # Default values make query parameters optional
    skip: int = Query(
        default=0,
        ge=0,  # must be >= 0
        description="Number of records to skip"
    ),
    limit: int = Query(
        default=10,
        ge=1,
        le=100,  # prevent clients from requesting too many records
        description="Maximum number of records to return"
    ),
    # Optional parameters use None as default
    role: str | None = Query(
        default=None,
        min_length=2,
        max_length=20,
        description="Filter by user role"
    ),
    # List query parameters: /users?tag=python&tag=api
    tags: list[str] = Query(default=[]),
):
    """List users with pagination and optional filtering.
    Example: /users?skip=0&limit=20&role=admin"""
    result = {"skip": skip, "limit": limit}
    if role:
        result["role_filter"] = role
    if tags:
        result["tag_filter"] = tags
    return result
```

### Path vs Query Parameters Summary

```
GET /users/42/posts?page=2&sort=date
     ├──────┘       ├────┘  ├───────┘
     Path param     Query    Query
     (required)     param    param
                    (opt.)   (opt.)
```

---

## 4. Request Body with Pydantic v2

For `POST`, `PUT`, and `PATCH` requests, clients send data in the request body. FastAPI uses Pydantic models to validate and parse this data.

### Basic Pydantic Model

```python
from pydantic import BaseModel, Field, field_validator
from datetime import datetime

class UserCreate(BaseModel):
    """Schema for creating a new user.
    Pydantic v2 validates all fields on instantiation and raises
    clear errors if the data doesn't match the schema."""

    name: str = Field(
        ...,
        min_length=1,
        max_length=100,
        examples=["Alice Johnson"],
        description="User's full name"
    )
    email: str = Field(
        ...,
        pattern=r"^[\w\.-]+@[\w\.-]+\.\w+$",  # Regex validation
        examples=["alice@example.com"]
    )
    age: int | None = Field(
        default=None,
        ge=0,
        le=150,
        description="User's age (optional)"
    )

    # Pydantic v2 uses @field_validator instead of v1's @validator
    @field_validator("name")
    @classmethod
    def name_must_not_be_empty(cls, v: str) -> str:
        """Strip whitespace and ensure the name isn't blank.
        Validators run after type checking but before the model is created."""
        stripped = v.strip()
        if not stripped:
            raise ValueError("Name cannot be empty or whitespace-only")
        return stripped


class UserResponse(BaseModel):
    """Schema for returning user data to clients.
    Separate from UserCreate because the response includes
    server-generated fields like id and created_at."""
    id: int
    name: str
    email: str
    age: int | None = None
    created_at: datetime

    # Pydantic v2 uses model_config instead of class Config
    model_config = {
        "from_attributes": True  # Allows creating from ORM objects
    }
```

### Using Models in Endpoints

```python
from fastapi import FastAPI, status

app = FastAPI()

# In-memory storage for demonstration
users_db: dict[int, dict] = {}
next_id = 1

@app.post(
    "/users",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED
)
async def create_user(user: UserCreate):
    """Create a new user.
    FastAPI automatically:
    1. Parses the JSON body into a UserCreate instance
    2. Validates all fields (returns 422 on failure)
    3. Filters the response through UserResponse"""
    global next_id
    now = datetime.now()
    user_data = {
        "id": next_id,
        **user.model_dump(),  # Pydantic v2: model_dump() replaces dict()
        "created_at": now
    }
    users_db[next_id] = user_data
    next_id += 1
    return user_data
```

### Nested Models

```python
from pydantic import BaseModel

class Address(BaseModel):
    street: str
    city: str
    country: str = "US"
    zip_code: str = Field(pattern=r"^\d{5}(-\d{4})?$")

class UserWithAddress(BaseModel):
    """Pydantic validates nested models recursively.
    If address.zip_code is invalid, the error message includes
    the full path: body -> address -> zip_code."""
    name: str
    email: str
    address: Address  # Nested model
    tags: list[str] = []  # List of strings with default

@app.post("/users-with-address")
async def create_user_with_address(user: UserWithAddress):
    return user
```

### Request Body Example

```json
{
    "name": "Alice Johnson",
    "email": "alice@example.com",
    "address": {
        "street": "123 Main St",
        "city": "Springfield",
        "country": "US",
        "zip_code": "62704"
    },
    "tags": ["admin", "premium"]
}
```

---

## 5. Response Models and Status Codes

### Response Model Filtering

Response models control what data is sent to the client. This is critical for security -- you never want to accidentally expose password hashes or internal fields.

```python
from pydantic import BaseModel, EmailStr

class UserInDB(BaseModel):
    """Internal representation with sensitive fields."""
    id: int
    name: str
    email: str
    hashed_password: str  # Never expose this!
    is_active: bool
    internal_notes: str  # Admin-only field

class UserPublic(BaseModel):
    """Public representation -- only safe fields."""
    id: int
    name: str
    email: str
    is_active: bool

@app.get(
    "/users/{user_id}",
    response_model=UserPublic,  # Filters out hashed_password and internal_notes
    response_model_exclude_none=True  # Omit fields with None values
)
async def get_user(user_id: int):
    # Even if the function returns all fields, only UserPublic fields
    # appear in the response. This is a safety net.
    user = get_user_from_db(user_id)
    return user
```

### Multiple Response Status Codes

```python
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse

app = FastAPI()

@app.post(
    "/users",
    status_code=status.HTTP_201_CREATED,
    responses={
        201: {"description": "User created successfully"},
        409: {"description": "Email already registered"},
        422: {"description": "Validation error in request body"},
    }
)
async def create_user(user: UserCreate):
    """Using the responses parameter documents all possible status codes
    in the OpenAPI spec, helping API consumers understand error cases."""
    existing = find_user_by_email(user.email)
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Email {user.email} is already registered"
        )
    return save_user(user)

@app.delete(
    "/users/{user_id}",
    status_code=status.HTTP_204_NO_CONTENT  # No body in response
)
async def delete_user(user_id: int):
    """204 No Content is the standard for successful DELETE.
    The response has no body -- just the status code."""
    user = find_user(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    remove_user(user_id)
    # Return nothing -- FastAPI sends 204 automatically
```

---

## 6. Automatic OpenAPI Documentation

FastAPI generates an OpenAPI 3.1 schema from your type hints, docstrings, and metadata. This schema powers two interactive documentation UIs.

### Swagger UI (`/docs`)

```
┌──────────────────────────────────────────────┐
│  My First API v0.1.0                          │
│                                               │
│  ▼ users                                      │
│    GET  /users         List users             │
│    POST /users         Create a new user      │
│    GET  /users/{id}    Get user by ID         │
│    PUT  /users/{id}    Update user            │
│    DEL  /users/{id}    Delete user            │
│                                               │
│  [Try it out] button lets you send real       │
│  requests and see responses inline.           │
└──────────────────────────────────────────────┘
```

### Enriching the Documentation

```python
from fastapi import FastAPI, status

app = FastAPI(
    title="User Management API",
    version="1.0.0",
    description="""
    ## Overview
    This API manages users for the application.

    ## Authentication
    Most endpoints require a Bearer token in the Authorization header.
    """,
    # Group endpoints by tags in the docs UI
    openapi_tags=[
        {"name": "users", "description": "User CRUD operations"},
        {"name": "admin", "description": "Administrative endpoints"},
    ]
)

@app.post(
    "/users",
    tags=["users"],  # Groups this endpoint under "users" in docs
    summary="Create a new user",  # Short description in endpoint list
    description="Creates a new user account with the provided information.",
    response_description="The newly created user object",
    status_code=status.HTTP_201_CREATED,
)
async def create_user(user: UserCreate):
    """If both `summary` and a docstring are provided,
    the summary is used for the endpoint list and the
    docstring appears in the expanded detail view."""
    ...
```

### Exporting the OpenAPI Schema

```python
# Access the schema programmatically
@app.get("/openapi-custom")
async def get_custom_schema():
    """The schema is a plain dict that can be serialized to JSON or YAML.
    Useful for generating client SDKs or importing into Postman."""
    return app.openapi()
```

```bash
# Or fetch it directly from the running server
curl http://localhost:8000/openapi.json | python -m json.tool
```

---

## 7. CORS Middleware

**CORS** (Cross-Origin Resource Sharing) controls which frontend domains can call your API. Without CORS configuration, browsers block requests from different origins.

### The Problem

```
Frontend: https://myapp.com        Backend API: https://api.myapp.com
        │                                      │
        │  fetch("/api/users")                 │
        │ ──────────────────────────────────▶  │
        │                                      │
        │  ✗ Blocked by browser!               │
        │  "No 'Access-Control-Allow-Origin'"  │
        │ ◀─────────────────────────────────── │
```

### The Solution

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Define which origins are allowed to make cross-origin requests
# In development, you might allow all (*); in production, be specific
origins = [
    "http://localhost:3000",     # React dev server
    "http://localhost:5173",     # Vite dev server
    "https://myapp.com",         # Production frontend
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,       # Which origins can access the API
    allow_credentials=True,      # Allow cookies/auth headers
    allow_methods=["*"],         # Allow all HTTP methods
    allow_headers=["*"],         # Allow all headers
    max_age=600,                 # Cache preflight response for 10 minutes
)
```

### How CORS Works (Preflight)

For non-simple requests (e.g., `POST` with JSON body), the browser sends a **preflight** `OPTIONS` request first:

```
Browser                              Server
  │                                     │
  │  OPTIONS /api/users HTTP/1.1        │  1. Preflight request
  │  Origin: https://myapp.com          │
  │  Access-Control-Request-Method: POST│
  │ ──────────────────────────────────▶ │
  │                                     │
  │  HTTP/1.1 204 No Content            │  2. Server approves
  │  Access-Control-Allow-Origin: *     │
  │  Access-Control-Allow-Methods: POST │
  │ ◀────────────────────────────────── │
  │                                     │
  │  POST /api/users HTTP/1.1           │  3. Actual request
  │  Origin: https://myapp.com          │
  │  Content-Type: application/json     │
  │ ──────────────────────────────────▶ │
  │                                     │
  │  HTTP/1.1 201 Created               │  4. Response with CORS headers
  │  Access-Control-Allow-Origin: *     │
  │ ◀────────────────────────────────── │
```

### Common CORS Pitfalls

| Issue | Cause | Fix |
|-------|-------|-----|
| `allow_origins=["*"]` with credentials | Wildcard origin and credentials are incompatible | List specific origins |
| Missing `OPTIONS` handler | Framework doesn't handle preflight | Use CORS middleware (handles it automatically) |
| `http` vs `https` mismatch | `http://localhost` is not `https://localhost` | Match the exact origin |

---

## 8. Practice Problems

### Problem 1: Build a Todo API

Create a complete FastAPI application with the following endpoints:
- `POST /todos` -- Create a new todo item (title, description, is_completed)
- `GET /todos` -- List all todos with optional query params: `completed` (bool filter), `skip`, `limit`
- `GET /todos/{todo_id}` -- Get a specific todo
- `PUT /todos/{todo_id}` -- Update a todo
- `DELETE /todos/{todo_id}` -- Delete a todo

Requirements:
- Use Pydantic v2 models for request/response
- Use proper HTTP status codes (201, 200, 204, 404)
- Store data in an in-memory dictionary
- Add type validation (title must be 1-200 chars)

### Problem 2: Pydantic Model Design

Design Pydantic v2 models for an e-commerce product catalog:
- `ProductCreate`: name, price (positive float), category (enum), description (optional), tags (list)
- `ProductResponse`: includes id, created_at, and an `is_on_sale` computed field
- `ProductUpdate`: all fields optional (for PATCH requests)

Use `Field()` with proper validation constraints and `model_config` for ORM compatibility.

### Problem 3: Query Parameter Validation

Create a `GET /search` endpoint that accepts:
- `q`: required search query (min 2 chars)
- `category`: optional, must be one of ["books", "electronics", "clothing"]
- `min_price` and `max_price`: optional floats, min_price must be less than max_price
- `sort_by`: optional, default "relevance", choices: ["relevance", "price_asc", "price_desc", "newest"]
- `page` and `page_size`: pagination with sensible defaults and limits

Return a mock response showing all applied filters.

### Problem 4: Error Response Standardization

Design a standardized error response format and implement a custom exception handler:

```json
{
    "error": {
        "code": "USER_NOT_FOUND",
        "message": "User with ID 42 was not found",
        "details": null,
        "timestamp": "2025-01-15T14:30:00Z"
    }
}
```

Implement this for at least three different error types (not found, validation error, duplicate resource).

### Problem 5: CORS Configuration

You have the following setup:
- API server: `https://api.example.com`
- Web app: `https://app.example.com`
- Mobile app: makes requests from any origin
- Admin panel: `https://admin.example.com` (needs cookies for auth)

Write the CORS middleware configuration that:
1. Allows all three frontends
2. Supports credential-based auth for the admin panel
3. Restricts methods to only those your API uses
4. Allows custom headers: `X-API-Key`, `X-Request-Id`

Explain why `allow_origins=["*"]` would not work here.

---

## 9. References

- [FastAPI Official Documentation](https://fastapi.tiangolo.com/)
- [Pydantic v2 Documentation](https://docs.pydantic.dev/latest/)
- [Starlette - The ASGI Toolkit](https://www.starlette.io/)
- [Uvicorn - ASGI Server](https://www.uvicorn.org/)
- [OpenAPI 3.1 Specification](https://spec.openapis.org/oas/v3.1.0)
- [MDN CORS Guide](https://developer.mozilla.org/en-US/docs/Web/HTTP/CORS)

---

**Previous**: [Backend Web Fundamentals](./01_Backend_Web_Fundamentals.md) | **Next**: [FastAPI Advanced](./03_FastAPI_Advanced.md)
