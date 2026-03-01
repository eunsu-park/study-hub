"""
FastAPI Testing — TestClient, pytest fixtures, async tests
Demonstrates: synchronous TestClient, httpx AsyncClient, fixture patterns.

Run: pip install fastapi httpx pytest pytest-asyncio
     pytest 04_testing.py -v
"""

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from pydantic import BaseModel


# --- App Under Test ---

app = FastAPI()

items: dict[int, dict] = {}


class Item(BaseModel):
    name: str
    price: float


@app.get("/")
async def root():
    return {"message": "Hello"}


@app.post("/items/{item_id}", status_code=201)
async def create_item(item_id: int, item: Item):
    if item_id in items:
        raise HTTPException(status_code=409, detail="Item exists")
    items[item_id] = item.model_dump()
    return {"item_id": item_id, **item.model_dump()}


@app.get("/items/{item_id}")
async def get_item(item_id: int):
    if item_id not in items:
        raise HTTPException(status_code=404, detail="Not found")
    return items[item_id]


# --- Synchronous Tests with TestClient ---

client = TestClient(app)


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello"}


def test_create_item():
    items.clear()
    response = client.post("/items/1", json={"name": "Widget", "price": 9.99})
    assert response.status_code == 201
    data = response.json()
    assert data["name"] == "Widget"
    assert data["price"] == 9.99


def test_create_duplicate():
    items.clear()
    client.post("/items/1", json={"name": "A", "price": 1.0})
    response = client.post("/items/1", json={"name": "B", "price": 2.0})
    assert response.status_code == 409


def test_get_missing():
    items.clear()
    response = client.get("/items/999")
    assert response.status_code == 404


def test_validation_error():
    """Pydantic rejects invalid data automatically."""
    response = client.post("/items/1", json={"name": "X"})
    assert response.status_code == 422  # Validation error


# --- Async Tests with httpx ---

@pytest.fixture
def clean_items():
    items.clear()
    yield
    items.clear()


@pytest.mark.asyncio
async def test_async_roundtrip(clean_items):
    """Use httpx.AsyncClient for truly async tests."""
    from httpx import AsyncClient, ASGITransport

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        # Create
        resp = await ac.post("/items/42", json={"name": "Gadget", "price": 19.99})
        assert resp.status_code == 201

        # Read back
        resp = await ac.get("/items/42")
        assert resp.status_code == 200
        assert resp.json()["name"] == "Gadget"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
