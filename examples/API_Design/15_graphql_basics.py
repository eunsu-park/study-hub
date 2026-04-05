#!/usr/bin/env python3
"""Example: GraphQL Basics

Demonstrates fundamental GraphQL concepts in Python with Strawberry:
- Schema definition with types
- Queries (read data)
- Mutations (write data)
- Variables and arguments
- Running a GraphQL server with FastAPI

Related lesson: 15_GraphQL_Basics.md

Run:
    pip install strawberry-graphql[fastapi] "fastapi[standard]"
    uvicorn 15_graphql_basics:app --reload --port 8000

    # GraphQL Playground: http://localhost:8000/graphql
"""

from datetime import datetime, timezone
from typing import Optional
from uuid import uuid4

import strawberry
from fastapi import FastAPI

# =============================================================================
# DATA STORE — Simple in-memory storage
# =============================================================================

_books_db: dict[str, dict] = {
    "1": {"id": "1", "title": "Clean Code", "author": "Robert C. Martin",
           "year": 2008, "rating": 4.5},
    "2": {"id": "2", "title": "Designing Data-Intensive Applications",
           "author": "Martin Kleppmann", "year": 2017, "rating": 4.8},
    "3": {"id": "3", "title": "The Pragmatic Programmer",
           "author": "David Thomas", "year": 2019, "rating": 4.6},
}


# =============================================================================
# TYPES — GraphQL object types (equivalent to REST response schemas)
# =============================================================================
# Unlike REST where the server decides the shape, in GraphQL the CLIENT
# chooses which fields to fetch. The type defines what is available.

@strawberry.type
class Book:
    """A book in the catalog. Clients can request any subset of these fields."""
    id: str
    title: str
    author: str
    year: int
    rating: float


@strawberry.type
class DeleteResult:
    success: bool
    message: str


# =============================================================================
# INPUT TYPES — For mutations (equivalent to REST request body schemas)
# =============================================================================

@strawberry.input
class BookInput:
    """Input for creating a book. Separate from the Book type because
    the client should not set 'id' (server-generated)."""
    title: str
    author: str
    year: int
    rating: float = 0.0


@strawberry.input
class BookUpdateInput:
    """Partial update input — all fields optional."""
    title: Optional[str] = None
    author: Optional[str] = None
    year: Optional[int] = None
    rating: Optional[float] = None


# =============================================================================
# QUERY — Read operations
# =============================================================================
# In GraphQL, all reads go through the Query root type.
# Unlike REST where each endpoint returns a fixed shape, the client
# specifies exactly which fields it needs.

@strawberry.type
class Query:
    @strawberry.field(description="Fetch a single book by ID.")
    def book(self, id: str) -> Optional[Book]:
        data = _books_db.get(id)
        if not data:
            return None
        return Book(**data)

    @strawberry.field(description="List all books, optionally filtered by minimum rating.")
    def books(self, min_rating: Optional[float] = None) -> list[Book]:
        results = list(_books_db.values())
        if min_rating is not None:
            results = [b for b in results if b["rating"] >= min_rating]
        return [Book(**b) for b in results]

    @strawberry.field(description="Count total books in the catalog.")
    def book_count(self) -> int:
        return len(_books_db)


# =============================================================================
# MUTATION — Write operations
# =============================================================================
# Mutations are GraphQL's equivalent of POST/PUT/DELETE in REST.
# They can also return data, so the client gets the updated state
# in a single round-trip.

@strawberry.type
class Mutation:
    @strawberry.mutation(description="Add a new book to the catalog.")
    def create_book(self, input: BookInput) -> Book:
        book_id = str(uuid4())[:8]
        book_data = {
            "id": book_id,
            "title": input.title,
            "author": input.author,
            "year": input.year,
            "rating": input.rating,
        }
        _books_db[book_id] = book_data
        return Book(**book_data)

    @strawberry.mutation(description="Update an existing book (partial update).")
    def update_book(self, id: str, input: BookUpdateInput) -> Optional[Book]:
        book = _books_db.get(id)
        if not book:
            return None
        if input.title is not None:
            book["title"] = input.title
        if input.author is not None:
            book["author"] = input.author
        if input.year is not None:
            book["year"] = input.year
        if input.rating is not None:
            book["rating"] = input.rating
        return Book(**book)

    @strawberry.mutation(description="Delete a book by ID.")
    def delete_book(self, id: str) -> DeleteResult:
        if id in _books_db:
            del _books_db[id]
            return DeleteResult(success=True, message=f"Book {id} deleted")
        return DeleteResult(success=False, message=f"Book {id} not found")


# =============================================================================
# SCHEMA & APP
# =============================================================================

schema = strawberry.Schema(query=Query, mutation=Mutation)

# Integrate Strawberry with FastAPI
from strawberry.fastapi import GraphQLRouter

graphql_app = GraphQLRouter(schema)

app = FastAPI(title="GraphQL Basics Demo")
app.include_router(graphql_app, prefix="/graphql")


# =============================================================================
# EXAMPLE QUERIES — Try these in the GraphQL Playground
# =============================================================================

EXAMPLE_QUERIES = """
# === Query: Get all books (only title and author) ===
query {
  books {
    title
    author
  }
}

# === Query: Get a specific book with all fields ===
query {
  book(id: "1") {
    id
    title
    author
    year
    rating
  }
}

# === Query: Filter books by rating ===
query {
  books(minRating: 4.6) {
    title
    rating
  }
}

# === Mutation: Create a new book ===
mutation {
  createBook(input: {title: "Refactoring", author: "Martin Fowler", year: 2018, rating: 4.7}) {
    id
    title
  }
}

# === Mutation: Update a book ===
mutation {
  updateBook(id: "1", input: {rating: 4.9}) {
    title
    rating
  }
}

# === Mutation: Delete a book ===
mutation {
  deleteBook(id: "2") {
    success
    message
  }
}
"""

# =============================================================================
# ENTRYPOINT
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    print("Example GraphQL queries:\n")
    print(EXAMPLE_QUERIES)
    uvicorn.run("15_graphql_basics:app", host="127.0.0.1", port=8000, reload=True)
