"""
FastAPI Basics — Hello API
Demonstrates: app creation, path/query params, Pydantic models, OpenAPI docs.

Run: pip install fastapi uvicorn && uvicorn 01_hello_api:app --reload
Docs: http://127.0.0.1:8000/docs
"""

from fastapi import FastAPI, Query, Path, HTTPException
from pydantic import BaseModel, Field
from enum import Enum

app = FastAPI(title="Hello API", version="1.0.0")


# --- 1. Basic Routes ---

@app.get("/")
async def root():
    return {"message": "Hello, FastAPI!"}


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


# --- 2. Path Parameters ---

class ItemCategory(str, Enum):
    tools = "tools"
    electronics = "electronics"
    food = "food"


@app.get("/items/{item_id}")
async def get_item(
    item_id: int = Path(..., title="Item ID", ge=1),
):
    return {"item_id": item_id}


@app.get("/categories/{category}")
async def get_category(category: ItemCategory):
    return {"category": category, "message": f"Browsing {category.value}"}


# --- 3. Query Parameters ---

@app.get("/search")
async def search(
    q: str = Query(..., min_length=1, max_length=100),
    skip: int = Query(0, ge=0),
    limit: int = Query(10, ge=1, le=100),
):
    return {"query": q, "skip": skip, "limit": limit}


# --- 4. Request Body with Pydantic ---

class ItemCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    price: float = Field(..., gt=0)
    description: str | None = None
    tags: list[str] = []

    model_config = {"json_schema_extra": {
        "examples": [{"name": "Widget", "price": 9.99, "tags": ["new"]}]
    }}


class ItemResponse(BaseModel):
    id: int
    name: str
    price: float
    description: str | None
    tags: list[str]


# In-memory store
items_db: dict[int, dict] = {}
next_id = 1


@app.post("/items", response_model=ItemResponse, status_code=201)
async def create_item(item: ItemCreate):
    global next_id
    item_data = {"id": next_id, **item.model_dump()}
    items_db[next_id] = item_data
    next_id += 1
    return item_data


@app.get("/items", response_model=list[ItemResponse])
async def list_items():
    return list(items_db.values())


@app.delete("/items/{item_id}", status_code=204)
async def delete_item(item_id: int):
    if item_id not in items_db:
        raise HTTPException(status_code=404, detail="Item not found")
    del items_db[item_id]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
