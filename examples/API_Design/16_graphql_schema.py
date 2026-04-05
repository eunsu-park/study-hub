#!/usr/bin/env python3
"""Example: GraphQL Schema Design

Demonstrates advanced GraphQL schema concepts with Strawberry:
- Object types with relationships
- Interfaces (shared fields across types)
- Union types (polymorphic returns)
- Input types for mutations
- Enum types
- Custom scalars (DateTime)
- Nullable vs non-nullable fields

Related lesson: 16_GraphQL_Schema_Design.md

Run:
    pip install strawberry-graphql[fastapi] "fastapi[standard]"
    uvicorn 16_graphql_schema:app --reload --port 8000
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Annotated, Optional, Union

import strawberry
from fastapi import FastAPI
from strawberry.fastapi import GraphQLRouter

# =============================================================================
# ENUMS — Constrained value sets
# =============================================================================

@strawberry.enum
class ContentStatus(Enum):
    DRAFT = "DRAFT"
    PUBLISHED = "PUBLISHED"
    ARCHIVED = "ARCHIVED"


@strawberry.enum
class MediaType(Enum):
    IMAGE = "IMAGE"
    VIDEO = "VIDEO"
    AUDIO = "AUDIO"


# =============================================================================
# INTERFACES — Shared fields that multiple types implement
# =============================================================================
# Interfaces define a contract. Any type implementing the interface
# MUST provide all its fields. Useful for polymorphic queries.

@strawberry.interface
class Node:
    """All entities have a globally unique ID (Relay convention)."""
    id: str


@strawberry.interface
class Timestamped:
    """Entities with audit timestamps."""
    created_at: datetime
    updated_at: datetime


# =============================================================================
# OBJECT TYPES — The core schema entities
# =============================================================================

@strawberry.type
class Author:
    id: str
    name: str
    bio: Optional[str] = None

    @strawberry.field(description="Articles written by this author.")
    def articles(self) -> list["Article"]:
        return [Article(**a) for a in _articles_db.values()
                if a["author_id"] == self.id]


@strawberry.type
class Tag:
    name: str
    slug: str


@strawberry.type
class Article:
    """A published article with rich metadata."""
    id: str
    title: str
    body: str
    status: ContentStatus
    tags: list[Tag]
    author_id: str
    created_at: datetime
    updated_at: datetime

    @strawberry.field(description="Resolve the author relationship.")
    def author(self) -> Optional[Author]:
        data = _authors_db.get(self.author_id)
        return Author(**data) if data else None


@strawberry.type
class MediaItem:
    """A media attachment (image, video, audio)."""
    id: str
    url: str
    media_type: MediaType
    alt_text: Optional[str] = None
    duration_seconds: Optional[int] = None  # Only for video/audio


# =============================================================================
# UNION TYPES — Polymorphic search results
# =============================================================================
# Unions let a single field return different types. The client uses
# inline fragments (... on Article) to access type-specific fields.

SearchResult = Annotated[
    Union[Article, Author, MediaItem],
    strawberry.union("SearchResult"),
]


# =============================================================================
# INPUT TYPES — Structured mutation arguments
# =============================================================================

@strawberry.input
class ArticleInput:
    title: str
    body: str
    author_id: str
    status: ContentStatus = ContentStatus.DRAFT
    tag_names: list[str] = strawberry.field(default_factory=list)


@strawberry.input
class ArticleFilterInput:
    """Filter criteria for listing articles."""
    status: Optional[ContentStatus] = None
    author_id: Optional[str] = None
    tag: Optional[str] = None


# =============================================================================
# DATA STORE
# =============================================================================

_authors_db: dict[str, dict] = {
    "a1": {"id": "a1", "name": "Alice Chen", "bio": "Backend engineer and writer."},
    "a2": {"id": "a2", "name": "Bob Park", "bio": None},
}

_now = datetime.now(timezone.utc)
_articles_db: dict[str, dict] = {
    "art1": {
        "id": "art1", "title": "GraphQL Schema Design", "body": "Schemas are the backbone...",
        "status": ContentStatus.PUBLISHED, "author_id": "a1",
        "tags": [{"name": "GraphQL", "slug": "graphql"}, {"name": "API", "slug": "api"}],
        "created_at": _now, "updated_at": _now,
    },
    "art2": {
        "id": "art2", "title": "REST vs GraphQL", "body": "Comparing two paradigms...",
        "status": ContentStatus.DRAFT, "author_id": "a2",
        "tags": [{"name": "REST", "slug": "rest"}],
        "created_at": _now, "updated_at": _now,
    },
}

_media_db: dict[str, dict] = {
    "m1": {"id": "m1", "url": "/media/diagram.png", "media_type": MediaType.IMAGE,
            "alt_text": "Architecture diagram", "duration_seconds": None},
}


# =============================================================================
# QUERY ROOT
# =============================================================================

@strawberry.type
class Query:
    @strawberry.field
    def article(self, id: str) -> Optional[Article]:
        data = _articles_db.get(id)
        if not data:
            return None
        return Article(**{**data, "tags": [Tag(**t) for t in data["tags"]]})

    @strawberry.field
    def articles(self, filter: Optional[ArticleFilterInput] = None) -> list[Article]:
        results = list(_articles_db.values())
        if filter:
            if filter.status:
                results = [a for a in results if a["status"] == filter.status]
            if filter.author_id:
                results = [a for a in results if a["author_id"] == filter.author_id]
            if filter.tag:
                results = [a for a in results
                           if any(t["slug"] == filter.tag for t in a["tags"])]
        return [Article(**{**a, "tags": [Tag(**t) for t in a["tags"]]}) for a in results]

    @strawberry.field
    def author(self, id: str) -> Optional[Author]:
        data = _authors_db.get(id)
        return Author(**data) if data else None

    @strawberry.field(description="Polymorphic search across all content types.")
    def search(self, query: str) -> list[SearchResult]:
        results: list[SearchResult] = []
        q = query.lower()
        for a in _articles_db.values():
            if q in a["title"].lower():
                results.append(Article(**{**a, "tags": [Tag(**t) for t in a["tags"]]}))
        for a in _authors_db.values():
            if q in a["name"].lower():
                results.append(Author(**a))
        for m in _media_db.values():
            if q in m.get("alt_text", "").lower():
                results.append(MediaItem(**m))
        return results


# =============================================================================
# SCHEMA & APP
# =============================================================================

schema = strawberry.Schema(query=Query)
graphql_app = GraphQLRouter(schema)

app = FastAPI(title="GraphQL Schema Design Demo")
app.include_router(graphql_app, prefix="/graphql")

# =============================================================================
# EXAMPLE QUERIES
# =============================================================================

EXAMPLE_QUERIES = """
# === Union type query with inline fragments ===
query {
  search(query: "graphql") {
    ... on Article { id title status }
    ... on Author  { id name }
    ... on MediaItem { id url mediaType }
  }
}

# === Filtered articles with nested author ===
query {
  articles(filter: {status: PUBLISHED}) {
    title
    author { name bio }
    tags { name }
  }
}
"""

if __name__ == "__main__":
    import uvicorn
    print(EXAMPLE_QUERIES)
    uvicorn.run("16_graphql_schema:app", host="127.0.0.1", port=8000, reload=True)
