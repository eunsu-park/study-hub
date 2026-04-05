#!/usr/bin/env python3
"""Example: API Testing

Demonstrates API testing patterns with pytest and FastAPI's TestClient:
- Unit testing individual endpoints
- Contract testing (schema validation)
- Integration testing (multi-step workflows)
- Fixture patterns for test isolation
- Parametrized tests for edge cases

Related lesson: 11_API_Testing.md

Run:
    pip install "fastapi[standard]" pytest httpx
    pytest 11_api_testing.py -v
"""

from datetime import datetime, timezone
from typing import Optional
from uuid import uuid4

from fastapi import FastAPI, HTTPException, status
from fastapi.testclient import TestClient
from pydantic import BaseModel, Field

# =============================================================================
# MINIMAL API UNDER TEST
# =============================================================================

app = FastAPI(title="Testable API")

_db: dict[str, dict] = {}


class ItemCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    price: float = Field(..., gt=0)


class ItemResponse(BaseModel):
    id: str
    name: str
    price: float
    created_at: datetime


@app.post("/api/v1/items", response_model=ItemResponse, status_code=201)
def create_item(body: ItemCreate):
    item_id = str(uuid4())
    item = {"id": item_id, "name": body.name, "price": body.price,
            "created_at": datetime.now(timezone.utc)}
    _db[item_id] = item
    return item


@app.get("/api/v1/items/{item_id}", response_model=ItemResponse)
def get_item(item_id: str):
    item = _db.get(item_id)
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")
    return item


@app.get("/api/v1/items", response_model=list[ItemResponse])
def list_items():
    return list(_db.values())


@app.delete("/api/v1/items/{item_id}", status_code=204)
def delete_item(item_id: str):
    if item_id not in _db:
        raise HTTPException(status_code=404, detail="Item not found")
    del _db[item_id]


# =============================================================================
# TEST FIXTURES — Isolation is key to reliable tests
# =============================================================================

import pytest


@pytest.fixture(autouse=True)
def reset_db():
    """Clear the database before each test for isolation."""
    _db.clear()
    yield
    _db.clear()


@pytest.fixture
def client():
    """Provide a fresh TestClient instance."""
    return TestClient(app)


@pytest.fixture
def sample_item(client):
    """Create a sample item and return its data."""
    resp = client.post("/api/v1/items", json={"name": "Widget", "price": 9.99})
    assert resp.status_code == 201
    return resp.json()


# =============================================================================
# UNIT TESTS — One endpoint, one behavior per test
# =============================================================================

class TestCreateItem:
    def test_success(self, client):
        resp = client.post("/api/v1/items", json={"name": "Bolt", "price": 1.50})
        assert resp.status_code == 201
        data = resp.json()
        assert data["name"] == "Bolt"
        assert data["price"] == 1.50
        assert "id" in data

    def test_missing_name(self, client):
        resp = client.post("/api/v1/items", json={"price": 1.50})
        assert resp.status_code == 422  # Validation error

    def test_negative_price(self, client):
        resp = client.post("/api/v1/items", json={"name": "Bad", "price": -5})
        assert resp.status_code == 422


class TestGetItem:
    def test_found(self, client, sample_item):
        resp = client.get(f"/api/v1/items/{sample_item['id']}")
        assert resp.status_code == 200
        assert resp.json()["name"] == "Widget"

    def test_not_found(self, client):
        resp = client.get("/api/v1/items/nonexistent")
        assert resp.status_code == 404


# =============================================================================
# CONTRACT TESTS — Validate response schema matches the Pydantic model
# =============================================================================
# Contract tests ensure the API does not accidentally change its shape.

class TestContract:
    REQUIRED_FIELDS = {"id", "name", "price", "created_at"}

    def test_create_response_schema(self, client):
        resp = client.post("/api/v1/items", json={"name": "X", "price": 1.0})
        assert resp.status_code == 201
        assert self.REQUIRED_FIELDS.issubset(resp.json().keys())

    def test_list_returns_array(self, client, sample_item):
        resp = client.get("/api/v1/items")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)
        assert len(resp.json()) == 1


# =============================================================================
# INTEGRATION TEST — Multi-step workflow
# =============================================================================

class TestItemLifecycle:
    def test_create_read_delete(self, client):
        """Full CRUD lifecycle: create -> read -> delete -> confirm gone."""
        # Create
        create_resp = client.post("/api/v1/items", json={"name": "Temp", "price": 5.0})
        assert create_resp.status_code == 201
        item_id = create_resp.json()["id"]

        # Read
        get_resp = client.get(f"/api/v1/items/{item_id}")
        assert get_resp.status_code == 200

        # Delete
        del_resp = client.delete(f"/api/v1/items/{item_id}")
        assert del_resp.status_code == 204

        # Confirm gone
        get_resp2 = client.get(f"/api/v1/items/{item_id}")
        assert get_resp2.status_code == 404


# =============================================================================
# PARAMETRIZED TESTS — Cover many edge cases concisely
# =============================================================================

@pytest.mark.parametrize("name,price,expected_status", [
    ("Valid", 10.0, 201),
    ("", 10.0, 422),       # empty name
    ("X" * 101, 10.0, 422),  # name too long
    ("OK", 0, 422),         # zero price
    ("OK", -1, 422),        # negative price
])
def test_create_validation(client, name, price, expected_status):
    resp = client.post("/api/v1/items", json={"name": name, "price": price})
    assert resp.status_code == expected_status


# =============================================================================
# ENTRYPOINT
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
