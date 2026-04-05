#!/usr/bin/env python3
"""Example: GraphQL Performance

Demonstrates GraphQL performance optimization techniques:
- Query complexity analysis and limiting
- Depth limiting to prevent deeply nested attacks
- Persisted queries (allowlisting)
- Response caching with TTL
- Field-level cost directives

Related lesson: 21_GraphQL_Performance.md

Run:
    pip install strawberry-graphql[fastapi] "fastapi[standard]"
    uvicorn 21_graphql_performance:app --reload --port 8000
"""

import hashlib
import json
import logging
import time
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any, Optional

import strawberry
from fastapi import FastAPI, HTTPException, Request
from strawberry.extensions import SchemaExtension
from strawberry.fastapi import GraphQLRouter
from strawberry.types import Info

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("performance")

# =============================================================================
# DEPTH LIMITER — Prevent deeply nested queries
# =============================================================================
# Malicious clients can craft deeply nested queries to exhaust server resources:
#   { user { friends { friends { friends { ... } } } } }

MAX_DEPTH = 5


def calculate_depth(selection_set, current_depth: int = 1) -> int:
    """Recursively measure the deepest nesting level of a parsed query."""
    if not selection_set:
        return current_depth
    max_d = current_depth
    for field in selection_set.selections:
        if hasattr(field, "selection_set") and field.selection_set:
            d = calculate_depth(field.selection_set, current_depth + 1)
            max_d = max(max_d, d)
    return max_d


class DepthLimitExtension(SchemaExtension):
    """Reject queries that exceed the maximum allowed nesting depth."""

    def on_operation(self):
        # Access the parsed document from the execution context
        document = self.execution_context.graphql_document
        if document:
            for definition in document.definitions:
                if hasattr(definition, "selection_set"):
                    depth = calculate_depth(definition.selection_set)
                    if depth > MAX_DEPTH:
                        raise ValueError(
                            f"Query depth {depth} exceeds maximum allowed depth {MAX_DEPTH}"
                        )
                    logger.info(f"Query depth: {depth}/{MAX_DEPTH}")
        yield


# =============================================================================
# QUERY COMPLEXITY ANALYZER
# =============================================================================
# Assign costs to fields. Lists multiply cost by expected size.

FIELD_COSTS = {
    "posts": 10,       # list query — higher cost
    "comments": 5,     # nested list
    "author": 1,       # single object lookup
    "post": 1,         # single lookup
}

MAX_COMPLEXITY = 100


def estimate_complexity(selection_set, parent_multiplier: int = 1) -> int:
    """Estimate query cost by summing weighted field costs."""
    if not selection_set:
        return 0
    total = 0
    for field in selection_set.selections:
        name = field.name.value if hasattr(field, "name") else ""
        cost = FIELD_COSTS.get(name, 1) * parent_multiplier
        total += cost
        if hasattr(field, "selection_set") and field.selection_set:
            # Lists multiply nested costs
            multiplier = FIELD_COSTS.get(name, 1) if name in ("posts", "comments") else 1
            total += estimate_complexity(field.selection_set, multiplier)
    return total


class ComplexityExtension(SchemaExtension):
    """Reject queries that exceed the maximum complexity budget."""

    def on_operation(self):
        document = self.execution_context.graphql_document
        if document:
            for definition in document.definitions:
                if hasattr(definition, "selection_set"):
                    complexity = estimate_complexity(definition.selection_set)
                    if complexity > MAX_COMPLEXITY:
                        raise ValueError(
                            f"Query complexity {complexity} exceeds limit {MAX_COMPLEXITY}"
                        )
                    logger.info(f"Query complexity: {complexity}/{MAX_COMPLEXITY}")
        yield


# =============================================================================
# RESPONSE CACHE — TTL-based caching for read-heavy fields
# =============================================================================

_cache: dict[str, tuple[float, Any]] = {}
CACHE_TTL_SECONDS = 60


def cache_key(query: str, variables: Optional[dict]) -> str:
    raw = json.dumps({"q": query, "v": variables}, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def get_cached(key: str) -> Optional[Any]:
    entry = _cache.get(key)
    if entry and (time.monotonic() - entry[0]) < CACHE_TTL_SECONDS:
        logger.info(f"Cache HIT: {key}")
        return entry[1]
    return None


def set_cached(key: str, value: Any):
    _cache[key] = (time.monotonic(), value)


# =============================================================================
# PERSISTED QUERIES — Allowlisted query hashes
# =============================================================================
# Clients send a hash instead of the full query string. The server looks up
# the real query from a pre-registered map. This prevents arbitrary queries
# and reduces bandwidth.

PERSISTED_QUERIES: dict[str, str] = {
    "abc123": "{ posts { id title } }",
    "def456": '{ post(id: "1") { title author { name } comments { body } } }',
}


# =============================================================================
# DATA
# =============================================================================

@strawberry.type
class Comment:
    id: str
    body: str
    author_name: str


@strawberry.type
class Author:
    id: str
    name: str


@strawberry.type
class Post:
    id: str
    title: str
    body: str
    _author_id: strawberry.Private[str]

    @strawberry.field
    def author(self) -> Author:
        return _AUTHORS.get(self._author_id, Author(id="?", name="Unknown"))

    @strawberry.field
    def comments(self) -> list[Comment]:
        return [Comment(**c) for c in _COMMENTS_DB.get(self.id, [])]


_AUTHORS = {
    "a1": Author(id="a1", name="Alice"),
    "a2": Author(id="a2", name="Bob"),
}

_POSTS = {
    "1": {"id": "1", "title": "Performance Tips", "body": "Cache aggressively...", "_author_id": "a1"},
    "2": {"id": "2", "title": "Depth Limiting", "body": "Always limit depth...", "_author_id": "a2"},
}

_COMMENTS_DB = {
    "1": [
        {"id": "c1", "body": "Great post!", "author_name": "Carol"},
        {"id": "c2", "body": "Very useful.", "author_name": "Dave"},
    ],
    "2": [
        {"id": "c3", "body": "Thanks for sharing.", "author_name": "Eve"},
    ],
}


# =============================================================================
# QUERY ROOT
# =============================================================================

@strawberry.type
class Query:
    @strawberry.field
    def posts(self) -> list[Post]:
        return [Post(id=p["id"], title=p["title"], body=p["body"], _author_id=p["_author_id"])
                for p in _POSTS.values()]

    @strawberry.field
    def post(self, id: str) -> Optional[Post]:
        p = _POSTS.get(id)
        if not p:
            return None
        return Post(id=p["id"], title=p["title"], body=p["body"], _author_id=p["_author_id"])

    @strawberry.field(description="Resolve a persisted query by its hash.")
    def persisted_query(self, hash: str) -> Optional[str]:
        return PERSISTED_QUERIES.get(hash)


# =============================================================================
# SCHEMA & APP
# =============================================================================

schema = strawberry.Schema(
    query=Query,
    extensions=[DepthLimitExtension, ComplexityExtension],
)
graphql_app = GraphQLRouter(schema)

app = FastAPI(title="GraphQL Performance Demo")
app.include_router(graphql_app, prefix="/graphql")


EXAMPLES = """
# === This query has depth 3 (within limit) ===
query {
  posts {
    title
    author { name }
    comments { body }
  }
}

# === Persisted query lookup ===
query {
  persistedQuery(hash: "abc123")
}
"""

if __name__ == "__main__":
    import uvicorn
    print(EXAMPLES)
    uvicorn.run("21_graphql_performance:app", host="127.0.0.1", port=8000, reload=True)
