"""
Exercise: Code-First GraphQL with Python
Practice defining GraphQL schemas using Python type annotations (Strawberry pattern).

Run: python 09_code_first_python.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ============================================================
# Exercise 1: Type Definitions with Decorators
# Implement a type system that maps Python classes to GraphQL types.
# ============================================================

class GraphQLType:
    """Simulates @strawberry.type decorator."""
    _registry: dict[str, type] = {}

    @classmethod
    def register(cls, python_class):
        cls._registry[python_class.__name__] = python_class
        return python_class

    @classmethod
    def to_sdl(cls, python_class) -> str:
        """Convert a Python dataclass to GraphQL SDL."""
        lines = [f"type {python_class.__name__} {{"]
        for fname, ftype in python_class.__annotations__.items():
            gql_type = cls._map_type(ftype)
            lines.append(f"  {fname}: {gql_type}")
        lines.append("}")
        return "\n".join(lines)

    @classmethod
    def _map_type(cls, python_type) -> str:
        """Map Python type annotations to GraphQL types."""
        type_str = str(python_type)

        # Handle Optional
        if "Optional" in type_str:
            inner = type_str.replace("typing.Optional[", "").replace("Optional[", "").rstrip("]")
            return cls._map_simple(inner)

        # Handle list
        if "list[" in type_str.lower():
            inner = type_str.split("[")[1].rstrip("]")
            return f"[{cls._map_simple(inner)}!]!"

        return cls._map_simple(type_str) + "!"

    @classmethod
    def _map_simple(cls, type_name: str) -> str:
        mapping = {"str": "String", "int": "Int", "float": "Float", "bool": "Boolean"}
        clean = type_name.strip().split(".")[-1]
        return mapping.get(clean, clean)


@GraphQLType.register
@dataclass
class Author:
    id: str
    name: str
    bio: Optional[str] = None


@GraphQLType.register
@dataclass
class Book:
    id: str
    title: str
    price: float
    author_id: str
    tags: list[str] = field(default_factory=list)


# ============================================================
# Exercise 2: Resolver Registry
# Implement a resolver system that maps fields to Python functions.
# ============================================================

class ResolverRegistry:
    def __init__(self):
        self.resolvers: dict[str, dict[str, callable]] = {}

    def query(self, type_name: str, field_name: str):
        """Decorator to register a query resolver."""
        def decorator(func):
            if type_name not in self.resolvers:
                self.resolvers[type_name] = {}
            self.resolvers[type_name][field_name] = func
            return func
        return decorator

    def resolve(self, type_name: str, field_name: str, parent=None, **kwargs):
        """Execute a resolver."""
        resolver_map = self.resolvers.get(type_name, {})
        resolver = resolver_map.get(field_name)
        if not resolver:
            # Default resolver: attribute access on parent
            if parent and hasattr(parent, field_name):
                return getattr(parent, field_name)
            raise ValueError(f"No resolver for {type_name}.{field_name}")
        return resolver(parent, **kwargs)


# Mock data
authors_db = [
    Author(id="1", name="Tolkien", bio="Fantasy author"),
    Author(id="2", name="Orwell", bio="Dystopian author"),
]

books_db = [
    Book(id="b1", title="The Hobbit", price=12.99, author_id="1", tags=["fantasy"]),
    Book(id="b2", title="1984", price=9.99, author_id="2", tags=["dystopian", "classic"]),
    Book(id="b3", title="LOTR", price=24.99, author_id="1", tags=["fantasy", "epic"]),
]

registry = ResolverRegistry()


@registry.query("Query", "books")
def resolve_books(parent, genre: Optional[str] = None):
    if genre:
        return [b for b in books_db if genre in b.tags]
    return books_db


@registry.query("Query", "book")
def resolve_book(parent, id: str):
    return next((b for b in books_db if b.id == id), None)


@registry.query("Query", "authors")
def resolve_authors(parent):
    return authors_db


@registry.query("Book", "author")
def resolve_book_author(book):
    return next((a for a in authors_db if a.id == book.author_id), None)


@registry.query("Author", "books")
def resolve_author_books(author):
    return [b for b in books_db if b.author_id == author.id]


# ============================================================
# Exercise 3: Input Types and Mutations
# Implement input type validation and mutation execution.
# ============================================================

@dataclass
class CreateBookInput:
    title: str
    price: float
    author_id: str
    tags: list[str] = field(default_factory=list)


def validate_input(input_obj: CreateBookInput) -> list[str]:
    """Validate mutation input, return list of errors."""
    errors = []
    if not input_obj.title or len(input_obj.title.strip()) == 0:
        errors.append("Title is required")
    if input_obj.price <= 0:
        errors.append("Price must be positive")
    if not any(a.id == input_obj.author_id for a in authors_db):
        errors.append(f"Author {input_obj.author_id} not found")
    if len(input_obj.tags) > 5:
        errors.append("Maximum 5 tags allowed")
    return errors


@registry.query("Mutation", "createBook")
def resolve_create_book(parent, input: CreateBookInput):
    errors = validate_input(input)
    if errors:
        return {"success": False, "errors": errors, "book": None}
    book = Book(
        id=f"b{len(books_db) + 1}",
        title=input.title,
        price=input.price,
        author_id=input.author_id,
        tags=input.tags,
    )
    books_db.append(book)
    return {"success": True, "errors": [], "book": book}


# ============================================================
# Exercise 4: Async DataLoader Pattern
# Implement a batch loading pattern for resolving related entities.
# ============================================================

class DataLoader:
    def __init__(self, batch_fn):
        self.batch_fn = batch_fn
        self.cache: dict = {}
        self.queue: list = []
        self.batch_count = 0

    def load(self, key):
        """Queue a key for batch loading."""
        if key in self.cache:
            return self.cache[key]
        self.queue.append(key)
        return None  # Will be resolved after dispatch

    def dispatch(self):
        """Execute batch function for all queued keys."""
        if not self.queue:
            return {}
        keys = list(set(self.queue))
        self.batch_count += 1
        results = self.batch_fn(keys)
        for key, value in zip(keys, results):
            self.cache[key] = value
        self.queue.clear()
        return {k: self.cache.get(k) for k in keys}


def batch_load_authors(ids: list[str]) -> list[Optional[Author]]:
    """Batch load authors by IDs."""
    print(f"  DataLoader: batch loading authors [{', '.join(ids)}]")
    return [next((a for a in authors_db if a.id == id), None) for id in ids]


# ============================================================
# Test all exercises
# ============================================================

print("=== Exercise 1: Type Definitions ===\n")

author_sdl = GraphQLType.to_sdl(Author)
print(author_sdl)
has_name = "name: String!" in author_sdl
has_bio = "bio: String" in author_sdl
print(f"\nAuthor SDL has name:String! — {'PASS' if has_name else 'FAIL'}")
print(f"Author SDL has bio:String (nullable) — {'PASS' if has_bio else 'FAIL'}")

print(f"\nRegistered types: {list(GraphQLType._registry.keys())}")

print("\n=== Exercise 2: Resolver Registry ===\n")

all_books = registry.resolve("Query", "books")
print(f"All books: {len(all_books)} (expected 3): {'PASS' if len(all_books) == 3 else 'FAIL'}")

fantasy = registry.resolve("Query", "books", genre="fantasy")
print(f"Fantasy books: {len(fantasy)} (expected 2): {'PASS' if len(fantasy) == 2 else 'FAIL'}")

book = registry.resolve("Query", "book", id="b1")
print(f"Book b1: {book.title} (expected The Hobbit): {'PASS' if book.title == 'The Hobbit' else 'FAIL'}")

author = registry.resolve("Book", "author", parent=book)
print(f"Book author: {author.name} (expected Tolkien): {'PASS' if author.name == 'Tolkien' else 'FAIL'}")

author_books = registry.resolve("Author", "books", parent=author)
print(f"Author books: {len(author_books)} (expected 2): {'PASS' if len(author_books) == 2 else 'FAIL'}")

print("\n=== Exercise 3: Input Validation & Mutations ===\n")

# Valid input
good = CreateBookInput(title="New Book", price=15.99, author_id="1", tags=["fiction"])
result = registry.resolve("Mutation", "createBook", input=good)
print(f"Valid mutation: success={result['success']} (expected True): {'PASS' if result['success'] else 'FAIL'}")

# Invalid: missing author
bad1 = CreateBookInput(title="Bad", price=10.0, author_id="99")
result1 = registry.resolve("Mutation", "createBook", input=bad1)
print(f"Bad author: success={result1['success']} (expected False): {'PASS' if not result1['success'] else 'FAIL'}")
print(f"  Errors: {result1['errors']}")

# Invalid: negative price
bad2 = CreateBookInput(title="Bad", price=-5.0, author_id="1")
result2 = registry.resolve("Mutation", "createBook", input=bad2)
print(f"Bad price: success={result2['success']} (expected False): {'PASS' if not result2['success'] else 'FAIL'}")

print("\n=== Exercise 4: DataLoader Pattern ===\n")

loader = DataLoader(batch_load_authors)

# Queue multiple loads
loader.load("1")
loader.load("2")
loader.load("1")  # duplicate, should deduplicate

results = loader.dispatch()
print(f"Batch count: {loader.batch_count} (expected 1): {'PASS' if loader.batch_count == 1 else 'FAIL'}")
print(f"Loaded authors: {len(results)} (expected 2): {'PASS' if len(results) == 2 else 'FAIL'}")

# Second load should use cache
cached = loader.load("1")
print(f"Cache hit: {cached.name if cached else 'MISS'} (expected Tolkien): {'PASS' if cached and cached.name == 'Tolkien' else 'FAIL'}")

print("\nAll exercises completed!")
