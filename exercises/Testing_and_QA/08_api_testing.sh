#!/bin/bash
# Exercises for Lesson 08: API Testing
# Topic: Testing_and_QA
# Solutions to practice problems from the lesson.

# === Exercise 1: REST API Contract Testing ===
# Problem: Write tests that verify the API contract (response schema,
# status codes, headers) for a REST endpoint.
exercise_1() {
    echo "=== Exercise 1: REST API Contract Testing ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
import pytest
from flask import Flask, jsonify, request

def create_app():
    app = Flask(__name__)
    _todos = {}
    _next_id = [1]

    @app.route("/api/todos", methods=["GET"])
    def list_todos():
        return jsonify({
            "items": list(_todos.values()),
            "total": len(_todos)
        })

    @app.route("/api/todos", methods=["POST"])
    def create_todo():
        data = request.get_json()
        if not data or "title" not in data:
            return jsonify({"error": "title is required"}), 400
        todo = {"id": _next_id[0], "title": data["title"], "done": False}
        _todos[_next_id[0]] = todo
        _next_id[0] += 1
        return jsonify(todo), 201

    return app

@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    return app.test_client()

# --- Contract: Status Codes ---
def test_list_returns_200(client):
    response = client.get("/api/todos")
    assert response.status_code == 200

def test_create_returns_201(client):
    response = client.post("/api/todos", json={"title": "Buy milk"})
    assert response.status_code == 201

def test_create_without_title_returns_400(client):
    response = client.post("/api/todos", json={})
    assert response.status_code == 400

# --- Contract: Response Schema ---
def test_list_response_schema(client):
    """Every list response must have 'items' array and 'total' count."""
    response = client.get("/api/todos")
    data = response.get_json()
    assert "items" in data
    assert "total" in data
    assert isinstance(data["items"], list)
    assert isinstance(data["total"], int)

def test_todo_item_schema(client):
    """Every todo item must have id, title, and done fields."""
    client.post("/api/todos", json={"title": "Test"})
    response = client.get("/api/todos")
    item = response.get_json()["items"][0]
    assert set(item.keys()) == {"id", "title", "done"}
    assert isinstance(item["id"], int)
    assert isinstance(item["title"], str)
    assert isinstance(item["done"], bool)

# --- Contract: Headers ---
def test_response_content_type(client):
    response = client.get("/api/todos")
    assert response.content_type == "application/json"

# --- Contract: Error Response Schema ---
def test_error_response_has_message(client):
    response = client.post("/api/todos", json={})
    data = response.get_json()
    assert "error" in data
    assert isinstance(data["error"], str)
SOLUTION
}

# === Exercise 2: Testing HTTP Methods ===
# Problem: Write comprehensive tests for all CRUD operations
# of a REST resource, covering success and error cases.
exercise_2() {
    echo "=== Exercise 2: Testing HTTP Methods ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
import pytest

# Assuming the create_app from Exercise 1, extended with PUT and DELETE

@pytest.fixture
def populated_client(client):
    """Client with pre-existing todos."""
    client.post("/api/todos", json={"title": "Task 1"})
    client.post("/api/todos", json={"title": "Task 2"})
    client.post("/api/todos", json={"title": "Task 3"})
    return client

# GET collection
def test_get_empty_list(client):
    resp = client.get("/api/todos")
    assert resp.get_json()["total"] == 0

def test_get_populated_list(populated_client):
    resp = populated_client.get("/api/todos")
    assert resp.get_json()["total"] == 3

# POST (create)
def test_post_valid_todo(client):
    resp = client.post("/api/todos", json={"title": "New task"})
    assert resp.status_code == 201
    data = resp.get_json()
    assert data["title"] == "New task"
    assert data["done"] is False

def test_post_empty_body(client):
    resp = client.post("/api/todos", data="", content_type="text/plain")
    assert resp.status_code == 400

# End-to-end workflow
def test_create_then_verify_in_list(client):
    """POST creates the item, GET confirms it appears in the list."""
    client.post("/api/todos", json={"title": "Verify me"})
    resp = client.get("/api/todos")
    titles = [item["title"] for item in resp.get_json()["items"]]
    assert "Verify me" in titles

# Testing idempotency
def test_get_is_idempotent(populated_client):
    """Multiple GET requests return the same result (no side effects)."""
    resp1 = populated_client.get("/api/todos").get_json()
    resp2 = populated_client.get("/api/todos").get_json()
    assert resp1 == resp2
SOLUTION
}

# === Exercise 3: Authentication Testing ===
# Problem: Test API endpoints that require authentication tokens.
exercise_3() {
    echo "=== Exercise 3: Authentication Testing ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
import pytest
from flask import Flask, jsonify, request, g

def create_auth_app():
    app = Flask(__name__)
    VALID_TOKEN = "valid-api-token-123"

    @app.before_request
    def check_auth():
        if request.endpoint == "health":
            return  # Health check is public
        token = request.headers.get("Authorization", "").replace("Bearer ", "")
        if token != VALID_TOKEN:
            return jsonify({"error": "Unauthorized"}), 401

    @app.route("/health")
    def health():
        return jsonify({"status": "ok"})

    @app.route("/api/secret")
    def secret():
        return jsonify({"data": "classified"})

    return app

@pytest.fixture
def auth_client():
    app = create_auth_app()
    app.config["TESTING"] = True
    return app.test_client()

# --- Public endpoint ---
def test_health_no_auth_required(auth_client):
    resp = auth_client.get("/health")
    assert resp.status_code == 200

# --- Protected endpoint without token ---
def test_no_token_returns_401(auth_client):
    resp = auth_client.get("/api/secret")
    assert resp.status_code == 401

# --- Protected endpoint with invalid token ---
def test_invalid_token_returns_401(auth_client):
    resp = auth_client.get("/api/secret", headers={
        "Authorization": "Bearer wrong-token"
    })
    assert resp.status_code == 401

# --- Protected endpoint with valid token ---
def test_valid_token_returns_200(auth_client):
    resp = auth_client.get("/api/secret", headers={
        "Authorization": "Bearer valid-api-token-123"
    })
    assert resp.status_code == 200
    assert resp.get_json()["data"] == "classified"

# --- Helper fixture for authenticated requests ---
@pytest.fixture
def authed_client(auth_client):
    """Client pre-configured with valid auth token."""
    class AuthClient:
        def __init__(self, client, token):
            self.client = client
            self.headers = {"Authorization": f"Bearer {token}"}

        def get(self, url, **kwargs):
            kwargs.setdefault("headers", {}).update(self.headers)
            return self.client.get(url, **kwargs)

        def post(self, url, **kwargs):
            kwargs.setdefault("headers", {}).update(self.headers)
            return self.client.post(url, **kwargs)

    return AuthClient(auth_client, "valid-api-token-123")

def test_with_authed_client(authed_client):
    resp = authed_client.get("/api/secret")
    assert resp.status_code == 200
SOLUTION
}

# === Exercise 4: Testing Error Responses ===
# Problem: Systematically test all error conditions an API can return.
exercise_4() {
    echo "=== Exercise 4: Testing Error Responses ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
import pytest

# Error testing checklist for any API endpoint:

# 1. Missing required fields
def test_missing_required_field(client):
    resp = client.post("/api/todos", json={"description": "no title"})
    assert resp.status_code == 400
    assert "title" in resp.get_json()["error"].lower()

# 2. Invalid field types
def test_invalid_field_type(client):
    resp = client.post("/api/todos", json={"title": 12345})
    # Depending on API design: 400 or accepts it
    assert resp.status_code in (200, 201, 400)

# 3. Invalid JSON body
def test_invalid_json(client):
    resp = client.post(
        "/api/todos",
        data="{invalid json",
        content_type="application/json"
    )
    assert resp.status_code == 400

# 4. Resource not found
def test_not_found(client):
    resp = client.get("/api/todos/99999")
    assert resp.status_code == 404

# 5. Method not allowed
def test_method_not_allowed(client):
    resp = client.patch("/api/todos")
    assert resp.status_code == 405

# 6. Wrong content type
def test_wrong_content_type(client):
    resp = client.post(
        "/api/todos",
        data="title=test",
        content_type="application/x-www-form-urlencoded"
    )
    # API expecting JSON should reject form-encoded data
    assert resp.status_code in (400, 415)

# Best practice: All error responses should have the same structure
# {"error": "Human-readable message"} or
# {"error": {"code": "VALIDATION_ERROR", "message": "...", "details": [...]}}
SOLUTION
}

# Run all exercises
echo "Exercise solutions for Lesson 08: API Testing"
echo "================================================"
exercise_1
echo ""
exercise_2
echo ""
exercise_3
echo ""
exercise_4
