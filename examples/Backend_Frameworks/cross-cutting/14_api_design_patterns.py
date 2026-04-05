"""
API Design Patterns — REST, HATEOAS, Pagination
Demonstrates: RESTful resource design, hypermedia links, cursor pagination,
              content negotiation, and idempotency keys.

Run: pip install fastapi uvicorn && uvicorn 14_api_design_patterns:app --reload
Docs: http://127.0.0.1:8000/docs
"""

from fastapi import FastAPI, Query, Request, Response, Header
from pydantic import BaseModel, Field
from typing import Optional
import base64
import uuid

app = FastAPI(title="API Design Patterns", version="1.0.0")

# --- In-memory store ---
BOOKS = [{"id": i, "title": f"Book {i}", "author": f"Author {i % 5}"} for i in range(1, 101)]
IDEMPOTENCY_STORE: dict[str, dict] = {}


# --- 1. Cursor-Based Pagination ---

class PaginatedResponse(BaseModel):
    data: list[dict]
    next_cursor: Optional[str] = None
    prev_cursor: Optional[str] = None
    has_more: bool


def encode_cursor(book_id: int) -> str:
    return base64.urlsafe_b64encode(str(book_id).encode()).decode()


def decode_cursor(cursor: str) -> int:
    return int(base64.urlsafe_b64decode(cursor.encode()).decode())


@app.get("/api/books", response_model=PaginatedResponse)
async def list_books(
    cursor: Optional[str] = Query(None, description="Opaque pagination cursor"),
    limit: int = Query(10, ge=1, le=50),
):
    """Cursor-based pagination avoids offset drift on mutable datasets."""
    start_id = decode_cursor(cursor) if cursor else 0
    filtered = [b for b in BOOKS if b["id"] > start_id]
    page = filtered[:limit]
    has_more = len(filtered) > limit
    return PaginatedResponse(
        data=page,
        next_cursor=encode_cursor(page[-1]["id"]) if page and has_more else None,
        has_more=has_more,
    )


# --- 2. HATEOAS Links ---

def add_links(book: dict, request: Request) -> dict:
    """Attach hypermedia _links so clients discover related actions."""
    base = str(request.base_url).rstrip("/")
    return {
        **book,
        "_links": {
            "self": {"href": f"{base}/api/books/{book['id']}"},
            "collection": {"href": f"{base}/api/books"},
            "reviews": {"href": f"{base}/api/books/{book['id']}/reviews"},
        },
    }


@app.get("/api/books/{book_id}")
async def get_book(book_id: int, request: Request):
    book = next((b for b in BOOKS if b["id"] == book_id), None)
    if not book:
        return Response(status_code=404, content='{"detail":"Not found"}')
    return add_links(book, request)


# --- 3. Idempotency Key ---

class OrderCreate(BaseModel):
    book_id: int
    quantity: int = Field(ge=1)


@app.post("/api/orders", status_code=201)
async def create_order(
    order: OrderCreate,
    idempotency_key: Optional[str] = Header(None, alias="Idempotency-Key"),
):
    """Idempotency keys prevent duplicate order creation on retries."""
    if idempotency_key and idempotency_key in IDEMPOTENCY_STORE:
        return IDEMPOTENCY_STORE[idempotency_key]

    result = {
        "order_id": str(uuid.uuid4()),
        "book_id": order.book_id,
        "quantity": order.quantity,
        "status": "created",
    }
    if idempotency_key:
        IDEMPOTENCY_STORE[idempotency_key] = result
    return result


# --- 4. Content Negotiation ---

@app.get("/api/books/{book_id}/summary")
async def book_summary(book_id: int, accept: str = Header("application/json")):
    """Return JSON or plain text depending on Accept header."""
    book = next((b for b in BOOKS if b["id"] == book_id), None)
    if not book:
        return Response(status_code=404)
    if "text/plain" in accept:
        return Response(
            content=f"{book['title']} by {book['author']}",
            media_type="text/plain",
        )
    return book


# --- 5. Bulk Operation Endpoint ---

class BulkDelete(BaseModel):
    ids: list[int] = Field(max_length=100)


@app.post("/api/books/bulk-delete")
async def bulk_delete(payload: BulkDelete):
    """Bulk operations use POST (not DELETE) because of the request body."""
    deleted, not_found = [], []
    for bid in payload.ids:
        book = next((b for b in BOOKS if b["id"] == bid), None)
        if book:
            deleted.append(bid)
        else:
            not_found.append(bid)
    return {"deleted": deleted, "not_found": not_found}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
