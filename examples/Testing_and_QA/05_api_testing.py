#!/usr/bin/env python3
"""Example: API Testing

Demonstrates testing web APIs using Flask's test client and FastAPI's TestClient.
Covers JSON APIs, authentication, error handling, and file uploads.
Related lesson: 07_Integration_Testing.md, 08_API_Testing.md
"""

# =============================================================================
# WHY TEST APIs WITH A TEST CLIENT?
# Test clients send HTTP requests to your application IN-PROCESS, without
# starting a real server. This makes tests:
#   - Fast: no network overhead
#   - Isolated: no port conflicts
#   - Deterministic: no real external services
#   - Easy to debug: full access to app internals
# =============================================================================

import pytest
import json

# We will use Flask for this example since it is the viewer's stack.
# The same principles apply to FastAPI, Django, etc.

try:
    from flask import Flask, jsonify, request, abort
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not FLASK_AVAILABLE,
    reason="Flask not installed (pip install flask)"
)


# =============================================================================
# APPLICATION UNDER TEST
# =============================================================================
# A minimal REST API for managing books. In production, this would use
# a real database; here we use an in-memory dict for simplicity.

def create_app(testing=False):
    """Application factory pattern — essential for testability.
    Creating the app inside a function means each test gets a fresh app."""
    app = Flask(__name__)
    app.config["TESTING"] = testing

    # In-memory "database"
    books_db = {}
    next_id = [1]  # Mutable container for closure

    @app.route("/api/books", methods=["GET"])
    def list_books():
        """GET /api/books — list all books with optional filtering."""
        genre = request.args.get("genre")
        books = list(books_db.values())
        if genre:
            books = [b for b in books if b.get("genre") == genre]
        return jsonify({"books": books, "count": len(books)})

    @app.route("/api/books/<int:book_id>", methods=["GET"])
    def get_book(book_id):
        """GET /api/books/<id> — get a single book by ID."""
        book = books_db.get(book_id)
        if not book:
            abort(404)
        return jsonify(book)

    @app.route("/api/books", methods=["POST"])
    def create_book():
        """POST /api/books — create a new book."""
        data = request.get_json()
        if not data:
            return jsonify({"error": "Request body must be JSON"}), 400
        if "title" not in data or "author" not in data:
            return jsonify({"error": "title and author are required"}), 400

        book_id = next_id[0]
        next_id[0] += 1
        book = {
            "id": book_id,
            "title": data["title"],
            "author": data["author"],
            "genre": data.get("genre", "unknown"),
        }
        books_db[book_id] = book
        return jsonify(book), 201

    @app.route("/api/books/<int:book_id>", methods=["PUT"])
    def update_book(book_id):
        """PUT /api/books/<id> — update an existing book."""
        book = books_db.get(book_id)
        if not book:
            abort(404)
        data = request.get_json()
        if not data:
            return jsonify({"error": "Request body must be JSON"}), 400
        book.update({
            "title": data.get("title", book["title"]),
            "author": data.get("author", book["author"]),
            "genre": data.get("genre", book["genre"]),
        })
        return jsonify(book)

    @app.route("/api/books/<int:book_id>", methods=["DELETE"])
    def delete_book(book_id):
        """DELETE /api/books/<id> — delete a book."""
        if book_id not in books_db:
            abort(404)
        del books_db[book_id]
        return "", 204

    @app.route("/api/health", methods=["GET"])
    def health():
        """Health check endpoint."""
        return jsonify({"status": "healthy"})

    @app.errorhandler(404)
    def not_found(e):
        return jsonify({"error": "Resource not found"}), 404

    return app


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def app():
    """Create a fresh app instance for each test module."""
    app = create_app(testing=True)
    return app


@pytest.fixture
def client(app):
    """Flask test client — sends HTTP requests without a real server.
    This is the primary tool for API testing in Flask."""
    return app.test_client()


@pytest.fixture
def sample_book():
    """Reusable test data — keeps tests DRY."""
    return {"title": "Clean Code", "author": "Robert C. Martin", "genre": "programming"}


@pytest.fixture
def populated_client(client, sample_book):
    """Client with pre-populated data for tests that need existing records."""
    books = [
        {"title": "Clean Code", "author": "Robert Martin", "genre": "programming"},
        {"title": "Dune", "author": "Frank Herbert", "genre": "sci-fi"},
        {"title": "SICP", "author": "Abelson & Sussman", "genre": "programming"},
    ]
    for book in books:
        client.post("/api/books", json=book)
    return client


# =============================================================================
# 1. BASIC CRUD TESTS
# =============================================================================

class TestHealthCheck:
    def test_health_endpoint(self, client):
        """Smoke test: verify the app starts and responds."""
        response = client.get("/api/health")
        assert response.status_code == 200
        data = response.get_json()
        assert data["status"] == "healthy"


class TestCreateBook:
    def test_create_book_success(self, client, sample_book):
        """POST with valid JSON should return 201 and the created resource."""
        response = client.post("/api/books", json=sample_book)

        assert response.status_code == 201
        data = response.get_json()
        assert data["title"] == sample_book["title"]
        assert data["author"] == sample_book["author"]
        assert "id" in data  # Server assigns the ID

    def test_create_book_missing_title(self, client):
        """POST without required fields should return 400 with error message.
        Testing error responses is as important as testing success paths."""
        response = client.post("/api/books", json={"author": "Someone"})

        assert response.status_code == 400
        data = response.get_json()
        assert "error" in data
        assert "title" in data["error"].lower() or "required" in data["error"].lower()

    def test_create_book_no_json(self, client):
        """POST without JSON body should return 400."""
        response = client.post(
            "/api/books",
            data="not json",
            content_type="text/plain"
        )
        assert response.status_code == 400

    def test_create_book_default_genre(self, client):
        """Verify default values are applied correctly."""
        response = client.post(
            "/api/books",
            json={"title": "Test", "author": "Author"}
        )
        data = response.get_json()
        assert data["genre"] == "unknown"


class TestReadBooks:
    def test_list_empty(self, client):
        """GET on empty database should return empty list, not error."""
        response = client.get("/api/books")
        assert response.status_code == 200
        data = response.get_json()
        assert data["books"] == []
        assert data["count"] == 0

    def test_list_books(self, populated_client):
        """GET should return all created books."""
        response = populated_client.get("/api/books")
        data = response.get_json()
        assert data["count"] == 3

    def test_filter_by_genre(self, populated_client):
        """Query parameter filtering — a common API pattern."""
        response = populated_client.get("/api/books?genre=programming")
        data = response.get_json()
        assert data["count"] == 2
        assert all(b["genre"] == "programming" for b in data["books"])

    def test_get_single_book(self, populated_client):
        """GET by ID should return the specific book."""
        response = populated_client.get("/api/books/1")
        assert response.status_code == 200
        data = response.get_json()
        assert data["id"] == 1

    def test_get_nonexistent_book(self, client):
        """GET for missing ID should return 404, not 500."""
        response = client.get("/api/books/999")
        assert response.status_code == 404


class TestUpdateBook:
    def test_update_book(self, populated_client):
        """PUT should update specified fields and preserve others."""
        response = populated_client.put(
            "/api/books/1",
            json={"title": "Clean Code (2nd Edition)"}
        )
        assert response.status_code == 200
        data = response.get_json()
        assert data["title"] == "Clean Code (2nd Edition)"
        # Author should be preserved (not overwritten)
        assert data["author"] == "Robert Martin"

    def test_update_nonexistent(self, client):
        response = client.put("/api/books/999", json={"title": "X"})
        assert response.status_code == 404


class TestDeleteBook:
    def test_delete_book(self, populated_client):
        """DELETE should return 204 No Content and remove the resource."""
        response = populated_client.delete("/api/books/1")
        assert response.status_code == 204

        # Verify the book is actually gone
        response = populated_client.get("/api/books/1")
        assert response.status_code == 404

    def test_delete_nonexistent(self, client):
        response = client.delete("/api/books/999")
        assert response.status_code == 404


# =============================================================================
# 2. INTEGRATION SCENARIOS
# =============================================================================

class TestCRUDWorkflow:
    """End-to-end CRUD workflow — tests the full lifecycle of a resource."""

    def test_full_lifecycle(self, client):
        """Create -> Read -> Update -> Delete -> Verify deletion.
        This tests that operations compose correctly."""
        # Create
        response = client.post("/api/books", json={
            "title": "Test Book", "author": "Test Author"
        })
        assert response.status_code == 201
        book_id = response.get_json()["id"]

        # Read
        response = client.get(f"/api/books/{book_id}")
        assert response.status_code == 200
        assert response.get_json()["title"] == "Test Book"

        # Update
        response = client.put(
            f"/api/books/{book_id}",
            json={"title": "Updated Book"}
        )
        assert response.status_code == 200
        assert response.get_json()["title"] == "Updated Book"

        # Delete
        response = client.delete(f"/api/books/{book_id}")
        assert response.status_code == 204

        # Verify deletion
        response = client.get(f"/api/books/{book_id}")
        assert response.status_code == 404


# =============================================================================
# 3. RESPONSE FORMAT VALIDATION
# =============================================================================

class TestResponseFormat:
    """Verify the API contract — the structure of JSON responses."""

    def test_book_response_schema(self, client):
        """Every book response must have these exact fields.
        Schema validation prevents breaking API consumers."""
        response = client.post("/api/books", json={
            "title": "Schema Test", "author": "Author"
        })
        data = response.get_json()

        required_fields = {"id", "title", "author", "genre"}
        assert set(data.keys()) == required_fields

    def test_list_response_schema(self, client):
        """List endpoints should have a consistent envelope."""
        response = client.get("/api/books")
        data = response.get_json()

        assert "books" in data
        assert "count" in data
        assert isinstance(data["books"], list)
        assert isinstance(data["count"], int)

    def test_error_response_schema(self, client):
        """Error responses should have a consistent format too."""
        response = client.get("/api/books/999")
        data = response.get_json()
        assert "error" in data
        assert isinstance(data["error"], str)

    def test_content_type_is_json(self, client):
        """Verify Content-Type header — important for API consumers."""
        response = client.get("/api/health")
        assert response.content_type == "application/json"


# =============================================================================
# RUNNING THIS FILE
# =============================================================================
# pip install flask pytest
# pytest 05_api_testing.py -v

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
