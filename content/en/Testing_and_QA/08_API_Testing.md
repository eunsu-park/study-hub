# API Testing

**Previous**: [Integration Testing](./07_Integration_Testing.md) | **Next**: [End-to-End Testing](./09_End_to_End_Testing.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Test REST APIs using framework-provided test clients
2. Write request/response tests for Flask and FastAPI applications
3. Test authentication, error handling, and edge cases in API endpoints
4. Mock external HTTP calls with `pytest-httpx` and `responses`
5. Apply contract testing principles to verify API compatibility

---

## Why Test APIs?

APIs are the contract between your backend and its consumers (frontends, mobile apps, other services). API tests verify:

- **Correct status codes** — 200 for success, 404 for missing resources, 422 for validation errors
- **Response structure** — JSON shape, required fields, correct types
- **Business logic** — Authorization rules, data transformations, side effects
- **Error handling** — Graceful failures, informative error messages

API tests sit between unit tests and E2E tests in the test pyramid. They are faster than browser-based E2E tests but more realistic than pure unit tests because they exercise routing, serialization, middleware, and validation.

---

## Testing Flask Applications

Flask provides a built-in test client that simulates HTTP requests without starting a server.

### Application Setup

```python
# app.py
from flask import Flask, jsonify, request, abort

app = Flask(__name__)

# In-memory store for simplicity
BOOKS = {}
_next_id = 1


@app.post("/api/books")
def create_book():
    global _next_id
    data = request.get_json()
    if not data or "title" not in data:
        abort(400, description="Title is required")
    if not data.get("author"):
        abort(400, description="Author is required")

    book = {
        "id": _next_id,
        "title": data["title"],
        "author": data["author"],
        "year": data.get("year"),
    }
    BOOKS[_next_id] = book
    _next_id += 1
    return jsonify(book), 201


@app.get("/api/books")
def list_books():
    return jsonify(list(BOOKS.values()))


@app.get("/api/books/<int:book_id>")
def get_book(book_id):
    book = BOOKS.get(book_id)
    if book is None:
        abort(404, description="Book not found")
    return jsonify(book)


@app.delete("/api/books/<int:book_id>")
def delete_book(book_id):
    if book_id not in BOOKS:
        abort(404, description="Book not found")
    del BOOKS[book_id]
    return "", 204


@app.put("/api/books/<int:book_id>")
def update_book(book_id):
    if book_id not in BOOKS:
        abort(404, description="Book not found")
    data = request.get_json()
    if not data:
        abort(400, description="Request body is required")

    book = BOOKS[book_id]
    book["title"] = data.get("title", book["title"])
    book["author"] = data.get("author", book["author"])
    book["year"] = data.get("year", book["year"])
    return jsonify(book)
```

### Test Configuration

```python
# conftest.py
import pytest
from app import app, BOOKS


@pytest.fixture
def client():
    """Create a Flask test client."""
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


@pytest.fixture(autouse=True)
def reset_store():
    """Reset the in-memory store before each test."""
    global _next_id
    BOOKS.clear()
    import app as app_module
    app_module._next_id = 1
```

### Writing API Tests

```python
# test_books_api.py
import json


class TestCreateBook:
    def test_create_book_success(self, client):
        response = client.post(
            "/api/books",
            json={"title": "Clean Code", "author": "Robert Martin", "year": 2008},
        )
        assert response.status_code == 201
        data = response.get_json()
        assert data["title"] == "Clean Code"
        assert data["author"] == "Robert Martin"
        assert data["year"] == 2008
        assert "id" in data

    def test_create_book_without_title(self, client):
        response = client.post(
            "/api/books",
            json={"author": "Someone"},
        )
        assert response.status_code == 400

    def test_create_book_without_author(self, client):
        response = client.post(
            "/api/books",
            json={"title": "Some Book"},
        )
        assert response.status_code == 400

    def test_create_book_empty_body(self, client):
        response = client.post(
            "/api/books",
            data="",
            content_type="application/json",
        )
        assert response.status_code == 400

    def test_create_book_assigns_sequential_ids(self, client):
        r1 = client.post("/api/books", json={"title": "Book 1", "author": "A"})
        r2 = client.post("/api/books", json={"title": "Book 2", "author": "B"})
        assert r1.get_json()["id"] == 1
        assert r2.get_json()["id"] == 2


class TestListBooks:
    def test_list_empty(self, client):
        response = client.get("/api/books")
        assert response.status_code == 200
        assert response.get_json() == []

    def test_list_after_adding(self, client):
        client.post("/api/books", json={"title": "Book A", "author": "Auth A"})
        client.post("/api/books", json={"title": "Book B", "author": "Auth B"})
        response = client.get("/api/books")
        data = response.get_json()
        assert len(data) == 2


class TestGetBook:
    def test_get_existing_book(self, client):
        create_resp = client.post(
            "/api/books", json={"title": "Refactoring", "author": "Martin Fowler"}
        )
        book_id = create_resp.get_json()["id"]

        response = client.get(f"/api/books/{book_id}")
        assert response.status_code == 200
        assert response.get_json()["title"] == "Refactoring"

    def test_get_nonexistent_book(self, client):
        response = client.get("/api/books/9999")
        assert response.status_code == 404


class TestUpdateBook:
    def test_update_title(self, client):
        client.post("/api/books", json={"title": "Old Title", "author": "Author"})
        response = client.put("/api/books/1", json={"title": "New Title"})
        assert response.status_code == 200
        assert response.get_json()["title"] == "New Title"
        assert response.get_json()["author"] == "Author"  # Unchanged

    def test_update_nonexistent(self, client):
        response = client.put("/api/books/9999", json={"title": "X"})
        assert response.status_code == 404


class TestDeleteBook:
    def test_delete_existing(self, client):
        client.post("/api/books", json={"title": "To Delete", "author": "Author"})
        response = client.delete("/api/books/1")
        assert response.status_code == 204

        # Verify it is gone
        get_response = client.get("/api/books/1")
        assert get_response.status_code == 404

    def test_delete_nonexistent(self, client):
        response = client.delete("/api/books/9999")
        assert response.status_code == 404
```

---

## Testing FastAPI Applications

FastAPI provides `TestClient` from Starlette, which supports both sync and async endpoints.

```python
# fastapi_app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()


class TodoCreate(BaseModel):
    title: str
    description: str = ""
    priority: int = 1


class Todo(BaseModel):
    id: int
    title: str
    description: str
    priority: int
    completed: bool = False


TODOS: dict[int, dict] = {}
_counter = 0


@app.post("/api/todos", response_model=Todo, status_code=201)
def create_todo(todo: TodoCreate):
    global _counter
    _counter += 1
    item = {
        "id": _counter,
        "title": todo.title,
        "description": todo.description,
        "priority": todo.priority,
        "completed": False,
    }
    TODOS[_counter] = item
    return item


@app.get("/api/todos", response_model=list[Todo])
def list_todos(completed: bool | None = None):
    items = list(TODOS.values())
    if completed is not None:
        items = [t for t in items if t["completed"] == completed]
    return items


@app.patch("/api/todos/{todo_id}")
def toggle_complete(todo_id: int):
    if todo_id not in TODOS:
        raise HTTPException(status_code=404, detail="Todo not found")
    TODOS[todo_id]["completed"] = not TODOS[todo_id]["completed"]
    return TODOS[todo_id]
```

```python
# test_fastapi_app.py
import pytest
from fastapi.testclient import TestClient
from fastapi_app import app, TODOS


@pytest.fixture
def client():
    TODOS.clear()
    import fastapi_app
    fastapi_app._counter = 0
    return TestClient(app)


class TestCreateTodo:
    def test_create_success(self, client):
        response = client.post(
            "/api/todos",
            json={"title": "Buy groceries", "priority": 2},
        )
        assert response.status_code == 201
        data = response.json()
        assert data["title"] == "Buy groceries"
        assert data["priority"] == 2
        assert data["completed"] is False

    def test_create_with_defaults(self, client):
        response = client.post("/api/todos", json={"title": "Simple task"})
        data = response.json()
        assert data["description"] == ""
        assert data["priority"] == 1

    def test_create_missing_title(self, client):
        """FastAPI validates automatically via Pydantic."""
        response = client.post("/api/todos", json={"priority": 1})
        assert response.status_code == 422  # Validation error
        detail = response.json()["detail"]
        assert any("title" in str(e).lower() for e in detail)


class TestListTodos:
    def test_filter_completed(self, client):
        client.post("/api/todos", json={"title": "Task 1"})
        client.post("/api/todos", json={"title": "Task 2"})
        client.patch("/api/todos/1")  # Complete task 1

        response = client.get("/api/todos?completed=true")
        data = response.json()
        assert len(data) == 1
        assert data[0]["title"] == "Task 1"

    def test_filter_incomplete(self, client):
        client.post("/api/todos", json={"title": "Task 1"})
        client.post("/api/todos", json={"title": "Task 2"})
        client.patch("/api/todos/1")  # Complete task 1

        response = client.get("/api/todos?completed=false")
        data = response.json()
        assert len(data) == 1
        assert data[0]["title"] == "Task 2"
```

---

## Testing Authentication

Many APIs require authentication. Test both authenticated and unauthenticated access.

```python
# Auth helper for tests
@pytest.fixture
def auth_headers():
    """Generate valid authentication headers."""
    token = create_test_token(user_id=1, role="admin")
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def viewer_headers():
    """Generate headers for a read-only user."""
    token = create_test_token(user_id=2, role="viewer")
    return {"Authorization": f"Bearer {token}"}


class TestAuthenticatedEndpoints:
    def test_create_requires_auth(self, client):
        response = client.post("/api/books", json={"title": "X", "author": "Y"})
        assert response.status_code == 401

    def test_create_with_valid_token(self, client, auth_headers):
        response = client.post(
            "/api/books",
            json={"title": "X", "author": "Y"},
            headers=auth_headers,
        )
        assert response.status_code == 201

    def test_create_with_expired_token(self, client):
        expired_token = create_test_token(user_id=1, expired=True)
        headers = {"Authorization": f"Bearer {expired_token}"}
        response = client.post(
            "/api/books",
            json={"title": "X", "author": "Y"},
            headers=headers,
        )
        assert response.status_code == 401

    def test_viewer_cannot_delete(self, client, viewer_headers):
        # First create as admin
        admin_token = create_test_token(user_id=1, role="admin")
        client.post(
            "/api/books",
            json={"title": "X", "author": "Y"},
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        # Viewer tries to delete
        response = client.delete("/api/books/1", headers=viewer_headers)
        assert response.status_code == 403
```

---

## Testing Error Responses

Always test that your API returns useful error information:

```python
class TestErrorResponses:
    def test_404_includes_message(self, client):
        response = client.get("/api/books/9999")
        assert response.status_code == 404
        data = response.get_json()
        assert "not found" in data.get("description", "").lower()

    def test_400_includes_field_info(self, client):
        response = client.post("/api/books", json={})
        assert response.status_code == 400

    def test_405_method_not_allowed(self, client):
        response = client.patch("/api/books")  # PATCH not supported on collection
        assert response.status_code == 405

    def test_content_type_is_json(self, client):
        response = client.get("/api/books")
        assert response.content_type == "application/json"
```

---

## Mocking External HTTP Calls

When your API calls external services, mock those calls to keep tests fast and deterministic.

### Using responses Library

```bash
pip install responses
```

```python
import responses
import requests


def get_weather(city: str) -> dict:
    """Call external weather API."""
    resp = requests.get(
        "https://api.weather.com/current",
        params={"city": city},
    )
    resp.raise_for_status()
    return resp.json()


@responses.activate
def test_get_weather_success():
    responses.add(
        responses.GET,
        "https://api.weather.com/current",
        json={"city": "Seoul", "temp": 22.5},
        status=200,
    )

    result = get_weather("Seoul")
    assert result["temp"] == 22.5


@responses.activate
def test_get_weather_api_error():
    responses.add(
        responses.GET,
        "https://api.weather.com/current",
        json={"error": "Service unavailable"},
        status=503,
    )

    with pytest.raises(requests.HTTPError):
        get_weather("Seoul")
```

### Using pytest-httpx (for httpx-based apps)

```bash
pip install pytest-httpx
```

```python
import httpx
import pytest


async def fetch_user(user_id: int) -> dict:
    async with httpx.AsyncClient() as client:
        response = await client.get(f"https://api.example.com/users/{user_id}")
        response.raise_for_status()
        return response.json()


@pytest.mark.asyncio
async def test_fetch_user(httpx_mock):
    httpx_mock.add_response(
        url="https://api.example.com/users/1",
        json={"id": 1, "name": "Alice"},
    )

    user = await fetch_user(1)
    assert user["name"] == "Alice"


@pytest.mark.asyncio
async def test_fetch_user_not_found(httpx_mock):
    httpx_mock.add_response(
        url="https://api.example.com/users/999",
        status_code=404,
    )

    with pytest.raises(httpx.HTTPStatusError):
        await fetch_user(999)
```

---

## Response Validation

Beyond status codes, validate response structure rigorously:

```python
def test_book_response_structure(client):
    """Verify the response contains all expected fields with correct types."""
    client.post("/api/books", json={"title": "Test", "author": "Author", "year": 2024})
    response = client.get("/api/books/1")
    data = response.get_json()

    # Required fields exist
    assert "id" in data
    assert "title" in data
    assert "author" in data

    # Correct types
    assert isinstance(data["id"], int)
    assert isinstance(data["title"], str)
    assert isinstance(data["author"], str)
    assert data["year"] is None or isinstance(data["year"], int)

    # No unexpected fields leaked (e.g., internal IDs, passwords)
    allowed_fields = {"id", "title", "author", "year"}
    assert set(data.keys()) <= allowed_fields
```

### Using jsonschema for Validation

```bash
pip install jsonschema
```

```python
from jsonschema import validate

BOOK_SCHEMA = {
    "type": "object",
    "required": ["id", "title", "author"],
    "properties": {
        "id": {"type": "integer", "minimum": 1},
        "title": {"type": "string", "minLength": 1},
        "author": {"type": "string", "minLength": 1},
        "year": {"type": ["integer", "null"]},
    },
    "additionalProperties": False,
}


def test_response_matches_schema(client):
    client.post("/api/books", json={"title": "Schema Test", "author": "Tester"})
    response = client.get("/api/books/1")
    validate(instance=response.get_json(), schema=BOOK_SCHEMA)
```

---

## Contract Testing Introduction

Contract testing verifies that a provider (API) and consumer (client) agree on the API's structure. Unlike integration tests that test both sides together, contract tests verify each side independently against a shared contract.

```
Consumer Test:             Provider Test:
"I expect GET /users/1     "When called GET /users/1
 to return {id, name}"      I return {id, name}"
         │                           │
         └──── Contract File ────────┘
              (shared JSON)
```

Tools like **Pact** formalize this. The consumer generates a contract (pact file), and the provider verifies it:

```python
# Consumer side: generate the contract
# Provider side: verify against the contract

# This ensures API changes do not break consumers,
# even without running both services together.
```

Contract testing is most valuable in microservice architectures where teams own different services and need to coordinate API changes safely.

---

## Exercises

1. **Full CRUD test suite**: Build a simple Flask API for managing "notes" (create, list, get, update, delete). Write a complete test suite covering all endpoints, including validation errors, 404s, and empty state.

2. **Authentication tests**: Add token-based authentication to the notes API. Write tests verifying: unauthenticated requests are rejected, expired tokens are rejected, valid tokens succeed, and role-based access is enforced.

3. **External API mock**: Write a service that calls an external API (e.g., currency conversion). Use the `responses` library to mock the external API. Test both success and error scenarios (timeout, 500 error, malformed JSON).

---

**License**: CC BY-NC 4.0
