# Exercise: FastAPI Database — SQLAlchemy 2.0 Async
# Practice with async database operations.
#
# Run: pip install fastapi uvicorn sqlalchemy[asyncio] aiosqlite pytest

# Exercise 1: Define models
# Create SQLAlchemy models for a bookstore:
# - Author: id, name, bio, created_at
# - Book: id, title, isbn (unique), price, author_id (FK), published (bool)
# Use Mapped[] annotations (SQLAlchemy 2.0 style).

# TODO: Define Base, Author, Book models


# Exercise 2: Create Pydantic schemas
# AuthorCreate, AuthorResponse, BookCreate, BookResponse
# BookResponse should include author name (nested).

# TODO: Define schemas with model_config = {"from_attributes": True}


# Exercise 3: CRUD operations
# Implement repository functions:
# - create_author(session, data) → Author
# - get_authors(session, skip, limit) → list[Author]
# - create_book(session, data) → Book
# - get_books_by_author(session, author_id) → list[Book]
# - search_books(session, query, min_price, max_price) → list[Book]

# TODO: Implement repository functions


# Exercise 4: API endpoints
# POST /authors — create author
# GET /authors — list with pagination
# POST /books — create book (validate author exists)
# GET /books?q=...&min_price=...&max_price=... — search books
# GET /authors/{id}/books — books by author

# TODO: Implement FastAPI endpoints with async session dependency


# Exercise 5: Transaction
# POST /authors/{id}/books/bulk — create multiple books for an author
# All books should be created in a single transaction.
# If any book fails validation, none should be saved.

# TODO: Implement bulk create with transaction


if __name__ == "__main__":
    print("Implement the exercises and test with:")
    print("  pytest 04_fastapi_database.py -v")
