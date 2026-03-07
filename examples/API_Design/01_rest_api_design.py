#!/usr/bin/env python3
"""Example: REST API Design

Demonstrates a well-structured REST API using FastAPI with:
- Proper resource modeling (Pydantic schemas)
- Correct HTTP status codes for each operation
- HATEOAS links for discoverability
- Content negotiation and consistent response envelopes

Related lessons: 01_API_Design_Fundamentals.md, 02_REST_Architecture.md

Run:
    pip install "fastapi[standard]"
    uvicorn 01_rest_api_design:app --reload --port 8000
"""

from datetime import datetime, timezone
from typing import Optional
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Query, Request, status
from pydantic import BaseModel, Field

# =============================================================================
# SCHEMAS — Define the API contract with Pydantic models
# =============================================================================
# Separate input (Create/Update) models from output (Response) models.
# This prevents clients from setting server-managed fields like `id` or
# `created_at`, and lets you evolve inputs and outputs independently.

class BookCreate(BaseModel):
    """Input schema for creating a book."""
    title: str = Field(..., min_length=1, max_length=200, examples=["Clean Code"])
    author: str = Field(..., min_length=1, max_length=100, examples=["Robert C. Martin"])
    isbn: Optional[str] = Field(None, pattern=r"^\d{13}$", examples=["9780132350884"])
    genre: Optional[str] = Field(None, examples=["programming"])


class BookUpdate(BaseModel):
    """Input schema for partial updates (PATCH semantics)."""
    title: Optional[str] = Field(None, min_length=1, max_length=200)
    author: Optional[str] = Field(None, min_length=1, max_length=100)
    isbn: Optional[str] = Field(None, pattern=r"^\d{13}$")
    genre: Optional[str] = None


class Link(BaseModel):
    """HATEOAS link — tells clients what actions are available."""
    href: str
    rel: str
    method: str = "GET"


class BookResponse(BaseModel):
    """Output schema — includes server-generated fields and HATEOAS links."""
    id: str
    title: str
    author: str
    isbn: Optional[str] = None
    genre: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    links: list[Link] = []


class BookListResponse(BaseModel):
    """Envelope for collection responses — always include metadata."""
    data: list[BookResponse]
    total: int
    links: list[Link] = []


# =============================================================================
# APPLICATION SETUP
# =============================================================================

app = FastAPI(
    title="Bookstore API",
    version="1.0.0",
    description="A demonstration REST API following best practices.",
)

# In-memory store (replace with a real database in production)
books_db: dict[str, dict] = {}


# =============================================================================
# HELPER — Build HATEOAS links for a book resource
# =============================================================================

def build_book_links(request: Request, book_id: str) -> list[Link]:
    """Generate HATEOAS links so clients can discover available actions.

    HATEOAS (Hypermedia As The Engine Of Application State) is the highest
    level of the Richardson Maturity Model (Level 3). Each response tells
    the client what it can do next, reducing coupling to hardcoded URLs.
    """
    base = str(request.base_url).rstrip("/")
    return [
        Link(href=f"{base}/api/v1/books/{book_id}", rel="self", method="GET"),
        Link(href=f"{base}/api/v1/books/{book_id}", rel="update", method="PATCH"),
        Link(href=f"{base}/api/v1/books/{book_id}", rel="delete", method="DELETE"),
        Link(href=f"{base}/api/v1/books", rel="collection", method="GET"),
    ]


def to_response(request: Request, book: dict) -> BookResponse:
    """Convert internal book dict to API response with HATEOAS links."""
    return BookResponse(
        **book,
        links=build_book_links(request, book["id"]),
    )


# =============================================================================
# ROUTES — RESTful resource endpoints
# =============================================================================

@app.get(
    "/api/v1/books",
    response_model=BookListResponse,
    summary="List all books",
    tags=["Books"],
)
def list_books(
    request: Request,
    genre: Optional[str] = Query(None, description="Filter by genre"),
    limit: int = Query(20, ge=1, le=100, description="Items per page"),
    offset: int = Query(0, ge=0, description="Number of items to skip"),
):
    """GET /api/v1/books — Retrieve a paginated list of books.

    Returns 200 OK with a JSON envelope containing `data`, `total`, and
    navigation `links`. An empty collection returns 200 with an empty list,
    never 404.
    """
    books = list(books_db.values())

    # Optional filtering
    if genre:
        books = [b for b in books if b.get("genre") == genre]

    total = len(books)
    page = books[offset : offset + limit]

    base = str(request.base_url).rstrip("/")
    links = [Link(href=f"{base}/api/v1/books", rel="self")]
    if offset + limit < total:
        links.append(
            Link(href=f"{base}/api/v1/books?offset={offset + limit}&limit={limit}", rel="next")
        )
    if offset > 0:
        prev_offset = max(0, offset - limit)
        links.append(
            Link(href=f"{base}/api/v1/books?offset={prev_offset}&limit={limit}", rel="prev")
        )

    return BookListResponse(
        data=[to_response(request, b) for b in page],
        total=total,
        links=links,
    )


@app.post(
    "/api/v1/books",
    response_model=BookResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new book",
    tags=["Books"],
)
def create_book(request: Request, body: BookCreate):
    """POST /api/v1/books — Create a new book resource.

    Returns 201 Created with the full resource representation and a Location
    header pointing to the new resource. The 201 status code tells clients
    that a new resource was successfully created.
    """
    book_id = str(uuid4())
    now = datetime.now(timezone.utc)
    book = {
        "id": book_id,
        "title": body.title,
        "author": body.author,
        "isbn": body.isbn,
        "genre": body.genre,
        "created_at": now,
        "updated_at": now,
    }
    books_db[book_id] = book
    return to_response(request, book)


@app.get(
    "/api/v1/books/{book_id}",
    response_model=BookResponse,
    summary="Get a single book",
    tags=["Books"],
)
def get_book(request: Request, book_id: str):
    """GET /api/v1/books/{book_id} — Retrieve a specific book.

    Returns 200 OK if found, 404 Not Found otherwise. The response includes
    HATEOAS links to update, delete, and list operations.
    """
    book = books_db.get(book_id)
    if not book:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Book {book_id} not found",
        )
    return to_response(request, book)


@app.patch(
    "/api/v1/books/{book_id}",
    response_model=BookResponse,
    summary="Partially update a book",
    tags=["Books"],
)
def update_book(request: Request, book_id: str, body: BookUpdate):
    """PATCH /api/v1/books/{book_id} — Partial update.

    PATCH is preferred over PUT for partial updates because it only requires
    the fields being changed. PUT implies replacing the entire resource.
    Returns 200 OK with the updated resource.
    """
    book = books_db.get(book_id)
    if not book:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Book {book_id} not found",
        )

    update_data = body.model_dump(exclude_unset=True)
    book.update(update_data)
    book["updated_at"] = datetime.now(timezone.utc)

    return to_response(request, book)


@app.delete(
    "/api/v1/books/{book_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a book",
    tags=["Books"],
)
def delete_book(book_id: str):
    """DELETE /api/v1/books/{book_id} — Remove a book.

    Returns 204 No Content on success (no response body needed).
    Returns 404 if the book does not exist.
    """
    if book_id not in books_db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Book {book_id} not found",
        )
    del books_db[book_id]


# =============================================================================
# ENTRYPOINT
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("01_rest_api_design:app", host="127.0.0.1", port=8000, reload=True)
