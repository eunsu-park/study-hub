# Exercise: FastAPI Basics
# Practice with path params, query params, Pydantic models.
#
# Run: pip install fastapi uvicorn httpx pytest
#      pytest 02_fastapi_basics.py -v

from fastapi import FastAPI, Query, Path, HTTPException
from fastapi.testclient import TestClient
from pydantic import BaseModel, Field

app = FastAPI()

# --- Exercise 1: Todo API ---
# Create a complete CRUD API for a Todo list.
# Model: id (int), title (str), completed (bool), priority (1-5)

class TodoCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)
    priority: int = Field(default=3, ge=1, le=5)


class TodoResponse(BaseModel):
    id: int
    title: str
    completed: bool
    priority: int


todos: dict[int, dict] = {}
next_id = 1

# TODO: Implement these endpoints:

# POST /todos — Create a todo (status 201)
# GET /todos — List all todos (with optional ?completed=true filter)
# GET /todos/{id} — Get single todo (404 if missing)
# PATCH /todos/{id} — Update todo fields (partial update)
# DELETE /todos/{id} — Delete todo (204, 404 if missing)


# --- Exercise 2: Search with Pagination ---
# GET /search?q=...&skip=0&limit=10&sort_by=priority

# TODO: Implement search endpoint that filters todos by title (case-insensitive)
# and returns paginated results with total count.


# --- Exercise 3: Bulk Operations ---
# POST /todos/bulk — Create multiple todos at once
# DELETE /todos/bulk — Delete multiple todos by IDs

# TODO: Implement bulk endpoints


# --- Tests ---
client = TestClient(app)


def test_create_todo():
    response = client.post("/todos", json={"title": "Buy milk", "priority": 2})
    assert response.status_code == 201
    data = response.json()
    assert data["title"] == "Buy milk"
    assert data["priority"] == 2
    assert data["completed"] is False


def test_list_todos():
    response = client.get("/todos")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_get_missing_todo():
    response = client.get("/todos/9999")
    assert response.status_code == 404


def test_validation():
    response = client.post("/todos", json={"title": "", "priority": 10})
    assert response.status_code == 422  # Validation error


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
