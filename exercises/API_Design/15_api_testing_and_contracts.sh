#!/bin/bash
# Exercises for Lesson 15: API Testing and Contracts
# Topic: API_Design
# Solutions to practice problems from the lesson.

# === Exercise 1: API Integration Testing ===
# Problem: Write comprehensive integration tests for a FastAPI CRUD endpoint
# using the TestClient. Cover happy paths, error cases, and edge cases.
exercise_1() {
    echo "=== Exercise 1: API Integration Testing ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
import pytest
from fastapi import FastAPI, HTTPException, status
from fastapi.testclient import TestClient
from pydantic import BaseModel, Field
from typing import Optional
from uuid import uuid4

# --- Application ---
app = FastAPI()
todos_db = {}


class TodoCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)
    description: Optional[str] = None
    priority: int = Field(1, ge=1, le=5)


@app.post("/api/v1/todos", status_code=201)
def create_todo(body: TodoCreate):
    todo_id = str(uuid4())
    todo = {"id": todo_id, **body.model_dump(), "completed": False}
    todos_db[todo_id] = todo
    return todo


@app.get("/api/v1/todos/{todo_id}")
def get_todo(todo_id: str):
    if todo_id not in todos_db:
        raise HTTPException(status_code=404, detail="Todo not found")
    return todos_db[todo_id]


@app.get("/api/v1/todos")
def list_todos(completed: Optional[bool] = None):
    items = list(todos_db.values())
    if completed is not None:
        items = [t for t in items if t["completed"] == completed]
    return {"data": items, "total": len(items)}


@app.patch("/api/v1/todos/{todo_id}")
def update_todo(todo_id: str, body: dict):
    if todo_id not in todos_db:
        raise HTTPException(status_code=404, detail="Todo not found")
    todos_db[todo_id].update(body)
    return todos_db[todo_id]


@app.delete("/api/v1/todos/{todo_id}", status_code=204)
def delete_todo(todo_id: str):
    if todo_id not in todos_db:
        raise HTTPException(status_code=404, detail="Todo not found")
    del todos_db[todo_id]


# --- Tests ---
@pytest.fixture(autouse=True)
def clear_db():
    todos_db.clear()
    yield
    todos_db.clear()


client = TestClient(app)


class TestCreateTodo:
    def test_create_success(self):
        response = client.post("/api/v1/todos", json={
            "title": "Write tests", "priority": 3,
        })
        assert response.status_code == 201
        data = response.json()
        assert data["title"] == "Write tests"
        assert data["priority"] == 3
        assert data["completed"] is False
        assert "id" in data

    def test_create_minimal(self):
        """Only required field (title)."""
        response = client.post("/api/v1/todos", json={"title": "Minimal"})
        assert response.status_code == 201
        assert response.json()["priority"] == 1  # Default

    def test_create_empty_title_rejected(self):
        response = client.post("/api/v1/todos", json={"title": ""})
        assert response.status_code == 422

    def test_create_priority_out_of_range(self):
        response = client.post("/api/v1/todos", json={"title": "X", "priority": 10})
        assert response.status_code == 422


class TestGetTodo:
    def test_get_existing(self):
        create = client.post("/api/v1/todos", json={"title": "Test"})
        todo_id = create.json()["id"]
        response = client.get(f"/api/v1/todos/{todo_id}")
        assert response.status_code == 200
        assert response.json()["title"] == "Test"

    def test_get_nonexistent_returns_404(self):
        response = client.get("/api/v1/todos/nonexistent")
        assert response.status_code == 404


class TestFullLifecycle:
    def test_crud_lifecycle(self):
        """Create → Read → Update → Delete → Verify gone."""
        # Create
        r = client.post("/api/v1/todos", json={"title": "Lifecycle"})
        assert r.status_code == 201
        todo_id = r.json()["id"]

        # Read
        r = client.get(f"/api/v1/todos/{todo_id}")
        assert r.status_code == 200

        # Update
        r = client.patch(f"/api/v1/todos/{todo_id}", json={"completed": True})
        assert r.status_code == 200
        assert r.json()["completed"] is True

        # Delete
        r = client.delete(f"/api/v1/todos/{todo_id}")
        assert r.status_code == 204

        # Verify gone
        r = client.get(f"/api/v1/todos/{todo_id}")
        assert r.status_code == 404
SOLUTION
}

# === Exercise 2: Contract Testing with Schemathesis ===
# Problem: Set up property-based API testing using Schemathesis to
# automatically find edge cases from the OpenAPI spec.
exercise_2() {
    echo "=== Exercise 2: Contract Testing with Schemathesis ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
# Schemathesis generates test cases from your OpenAPI spec automatically.
# It finds bugs by exploring edge cases you might not think of:
# - Unicode strings, empty strings, very long strings
# - Negative numbers, zero, very large numbers
# - Missing required fields, extra fields
# - Invalid Content-Type headers

# Installation:
# pip install schemathesis

# Method 1: Command-line (quickest)
# schemathesis run http://localhost:8000/openapi.json

# Method 2: Python test integration
import schemathesis

# Load schema from the running app
schema = schemathesis.from_url("http://localhost:8000/openapi.json")

# Or from a FastAPI app directly (no server needed)
# from your_app import app
# schema = schemathesis.from_asgi("/openapi.json", app=app)


@schema.parametrize()
def test_api_contract(case):
    """Schemathesis generates hundreds of test cases automatically.

    For each endpoint, it tests:
    - All valid parameter combinations
    - Boundary values
    - Invalid inputs (should get 4xx, not 5xx)
    - Response schema validation
    """
    response = case.call()

    # Basic contract assertions:
    # 1. No 500 errors (server should handle all inputs gracefully)
    assert response.status_code < 500, (
        f"Server error on {case.method} {case.path}: {response.text}"
    )

    # 2. Response matches the declared schema
    case.validate_response(response)


# Method 3: Targeted testing for specific endpoints
@schema.parametrize(endpoint="/api/v1/todos")
def test_todos_endpoint(case):
    response = case.call()
    assert response.status_code < 500

    if response.status_code == 200:
        data = response.json()
        assert "data" in data
        assert "total" in data


# Method 4: Stateful testing (sequences of API calls)
# Schemathesis can chain requests: POST → GET → PATCH → DELETE
@schema.parametrize()
def test_stateful_api(case):
    """Tests that sequences of API operations are consistent."""
    response = case.call()
    case.validate_response(response)


# Running schemathesis from CLI with options:
# schemathesis run http://localhost:8000/openapi.json \
#   --checks all \                    # Run all built-in checks
#   --hypothesis-max-examples 100 \   # Generate 100 cases per endpoint
#   --stateful links \                # Follow HATEOAS links
#   --base-url http://localhost:8000
SOLUTION
}

# === Exercise 3: API Response Schema Validation ===
# Problem: Build reusable schema validators that verify API responses
# match the expected contract across all endpoints.
exercise_3() {
    echo "=== Exercise 3: API Response Schema Validation ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
import pytest
from fastapi.testclient import TestClient


def assert_json_structure(data: dict, schema: dict, path: str = "root"):
    """Recursively validate JSON response structure.

    schema format:
    {
        "field_name": type,           # Required field with type
        "?field_name": type,          # Optional field (prefix ?)
        "nested": {"sub": type},      # Nested object
        "items": [type],              # Array of type
    }
    """
    for key, expected_type in schema.items():
        optional = key.startswith("?")
        field_name = key.lstrip("?")
        field_path = f"{path}.{field_name}"

        if optional and field_name not in data:
            continue

        assert field_name in data, f"Missing required field: {field_path}"
        value = data[field_name]

        if isinstance(expected_type, dict):
            assert isinstance(value, dict), f"{field_path} should be object"
            assert_json_structure(value, expected_type, field_path)
        elif isinstance(expected_type, list) and len(expected_type) == 1:
            assert isinstance(value, list), f"{field_path} should be array"
            for i, item in enumerate(value):
                if isinstance(expected_type[0], dict):
                    assert_json_structure(item, expected_type[0], f"{field_path}[{i}]")
                else:
                    assert isinstance(item, expected_type[0]), \
                        f"{field_path}[{i}] should be {expected_type[0].__name__}"
        else:
            assert isinstance(value, expected_type), \
                f"{field_path} should be {expected_type.__name__}, got {type(value).__name__}"


# --- Schema definitions for our API ---

USER_SCHEMA = {
    "id": str,
    "username": str,
    "email": str,
    "?bio": str,
    "created_at": str,
}

TODO_SCHEMA = {
    "id": str,
    "title": str,
    "?description": str,
    "priority": int,
    "completed": bool,
}

LIST_SCHEMA = {
    "data": [TODO_SCHEMA],
    "total": int,
}

ERROR_SCHEMA = {
    "detail": str,
}


# --- Tests using the schema validator ---

class TestResponseSchemas:
    def test_create_todo_matches_schema(self, client):
        response = client.post("/api/v1/todos", json={"title": "Test"})
        assert_json_structure(response.json(), TODO_SCHEMA)

    def test_list_todos_matches_schema(self, client):
        response = client.get("/api/v1/todos")
        assert_json_structure(response.json(), LIST_SCHEMA)

    def test_error_matches_schema(self, client):
        response = client.get("/api/v1/todos/nonexistent")
        assert_json_structure(response.json(), ERROR_SCHEMA)

    def test_all_list_items_match_schema(self, client):
        """Verify every item in the list matches the schema."""
        # Create some todos first
        for i in range(5):
            client.post("/api/v1/todos", json={"title": f"Todo {i}"})

        response = client.get("/api/v1/todos")
        data = response.json()

        for todo in data["data"]:
            assert_json_structure(todo, TODO_SCHEMA)
SOLUTION
}

# === Exercise 4: API Mocking for Frontend Development ===
# Problem: Create a mock API server that returns realistic fake data
# for frontend development while the real API is being built.
exercise_4() {
    echo "=== Exercise 4: API Mocking for Frontend Development ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
from fastapi import FastAPI, Query
from typing import Optional
from uuid import uuid4
from datetime import datetime, timezone, timedelta
import random

app = FastAPI(title="Mock API Server", version="0.1.0-mock")

# --- Realistic fake data generators ---

FIRST_NAMES = ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank"]
LAST_NAMES = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia"]
DOMAINS = ["example.com", "test.org", "demo.io"]
PRODUCT_ADJECTIVES = ["Premium", "Basic", "Pro", "Enterprise", "Lite"]
PRODUCT_NOUNS = ["Widget", "Dashboard", "Analytics", "Connector", "Platform"]


def fake_user(user_id: str = None) -> dict:
    first = random.choice(FIRST_NAMES)
    last = random.choice(LAST_NAMES)
    return {
        "id": user_id or str(uuid4()),
        "first_name": first,
        "last_name": last,
        "email": f"{first.lower()}.{last.lower()}@{random.choice(DOMAINS)}",
        "role": random.choice(["admin", "user", "viewer"]),
        "created_at": (
            datetime.now(timezone.utc) - timedelta(days=random.randint(1, 365))
        ).isoformat(),
    }


def fake_product(product_id: str = None) -> dict:
    return {
        "id": product_id or str(uuid4()),
        "name": f"{random.choice(PRODUCT_ADJECTIVES)} {random.choice(PRODUCT_NOUNS)}",
        "price": round(random.uniform(9.99, 999.99), 2),
        "category": random.choice(["electronics", "software", "services"]),
        "in_stock": random.choice([True, True, True, False]),  # 75% in stock
    }


# --- Mock endpoints matching the real API contract ---

@app.get("/api/v1/users")
def list_users(
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
):
    """Returns fake users matching the real API response format."""
    users = [fake_user() for _ in range(per_page)]
    return {
        "data": users,
        "meta": {"total": 150, "page": page, "per_page": per_page},
    }


@app.get("/api/v1/users/{user_id}")
def get_user(user_id: str):
    return {"data": fake_user(user_id)}


@app.get("/api/v1/products")
def list_products(
    category: Optional[str] = None,
    page: int = 1,
    per_page: int = 20,
):
    products = [fake_product() for _ in range(per_page)]
    if category:
        products = [p for p in products if p["category"] == category]
    return {"data": products, "meta": {"total": 50, "page": page}}


@app.post("/api/v1/users", status_code=201)
def create_user(body: dict):
    """Echo back the input with server-generated fields."""
    return {"data": {**body, "id": str(uuid4()), "created_at": datetime.now(timezone.utc).isoformat()}}


# Run: uvicorn mock_server:app --port 8000
# Frontend team uses http://localhost:8000 during development
# Switch to real API URL in production config
SOLUTION
}

# Run all exercises
echo "Exercise solutions for Lesson 15: API Testing and Contracts"
echo "============================================================="
exercise_1
echo ""
exercise_2
echo ""
exercise_3
echo ""
exercise_4
