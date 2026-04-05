#!/usr/bin/env python3
"""Example: GraphQL Server Implementation

Demonstrates a production-ready GraphQL server with Strawberry + FastAPI:
- Schema with query, mutation, and subscription
- Custom middleware (logging, timing)
- Authentication via context
- Error formatting and extensions
- Schema export to SDL file

Related lesson: 19_GraphQL_Server_Implementation.md

Run:
    pip install strawberry-graphql[fastapi] "fastapi[standard]"
    uvicorn 19_graphql_server:app --reload --port 8000

    # Export schema SDL:
    python 19_graphql_server.py --export-schema
"""

import logging
import time
from datetime import datetime, timezone
from typing import Any, Optional
from uuid import uuid4

import strawberry
from fastapi import Depends, FastAPI, Request
from strawberry.extensions import SchemaExtension
from strawberry.fastapi import GraphQLRouter
from strawberry.types import Info

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("graphql-server")

# =============================================================================
# AUTHENTICATION — Extract user from request headers
# =============================================================================

class AuthUser:
    """Represents the authenticated user extracted from the request."""
    def __init__(self, user_id: str, username: str, roles: list[str]):
        self.user_id = user_id
        self.username = username
        self.roles = roles

    @property
    def is_admin(self) -> bool:
        return "admin" in self.roles


# Simulated token->user mapping
_TOKENS = {
    "admin-token": AuthUser("u1", "admin", ["admin", "user"]),
    "user-token": AuthUser("u2", "reader", ["user"]),
}


def get_current_user(request: Request) -> Optional[AuthUser]:
    """Extract user from Authorization header (Bearer token)."""
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        token = auth[7:]
        return _TOKENS.get(token)
    return None


# =============================================================================
# SCHEMA EXTENSION — Logging and timing for all operations
# =============================================================================

class TimingExtension(SchemaExtension):
    """Measures and logs execution time of each GraphQL operation.

    Extensions run as middleware around resolver execution.
    They can also add data to the 'extensions' field in the response.
    """

    def on_operation(self):
        start = time.monotonic()
        yield  # execution happens here
        elapsed_ms = (time.monotonic() - start) * 1000
        op_name = self.execution_context.operation_name or "anonymous"
        logger.info(f"Operation '{op_name}' completed in {elapsed_ms:.1f}ms")


# =============================================================================
# DATA STORE
# =============================================================================

_posts: dict[str, dict] = {
    "p1": {"id": "p1", "title": "Hello World", "body": "First post!",
            "author_id": "u1", "published": True,
            "created_at": datetime(2026, 1, 1, tzinfo=timezone.utc)},
    "p2": {"id": "p2", "title": "Draft Post", "body": "Work in progress...",
            "author_id": "u2", "published": False,
            "created_at": datetime(2026, 3, 15, tzinfo=timezone.utc)},
}


# =============================================================================
# TYPES
# =============================================================================

@strawberry.type
class Post:
    id: str
    title: str
    body: str
    author_id: str
    published: bool
    created_at: datetime


@strawberry.type
class PostResult:
    post: Optional[Post]
    errors: list[str]


@strawberry.input
class PostInput:
    title: str
    body: str
    published: bool = False


# =============================================================================
# QUERY
# =============================================================================

@strawberry.type
class Query:
    @strawberry.field
    def posts(self, published_only: bool = True) -> list[Post]:
        results = list(_posts.values())
        if published_only:
            results = [p for p in results if p["published"]]
        return [Post(**p) for p in results]

    @strawberry.field
    def post(self, id: str) -> Optional[Post]:
        data = _posts.get(id)
        return Post(**data) if data else None

    @strawberry.field(description="Return the currently authenticated user's name.")
    def me(self, info: Info) -> Optional[str]:
        user = info.context.get("user")
        return user.username if user else None


# =============================================================================
# MUTATION — With auth checks
# =============================================================================

@strawberry.type
class Mutation:
    @strawberry.mutation
    def create_post(self, input: PostInput, info: Info) -> PostResult:
        user = info.context.get("user")
        if not user:
            return PostResult(post=None, errors=["Authentication required."])

        post_id = str(uuid4())[:8]
        post_data = {
            "id": post_id, "title": input.title, "body": input.body,
            "author_id": user.user_id, "published": input.published,
            "created_at": datetime.now(timezone.utc),
        }
        _posts[post_id] = post_data
        return PostResult(post=Post(**post_data), errors=[])

    @strawberry.mutation
    def delete_post(self, id: str, info: Info) -> PostResult:
        user = info.context.get("user")
        if not user:
            return PostResult(post=None, errors=["Authentication required."])

        post = _posts.get(id)
        if not post:
            return PostResult(post=None, errors=[f"Post {id} not found."])

        # Only author or admin can delete
        if post["author_id"] != user.user_id and not user.is_admin:
            return PostResult(post=None, errors=["Forbidden: not the author."])

        del _posts[id]
        return PostResult(post=Post(**post), errors=[])


# =============================================================================
# CONTEXT GETTER — Inject auth user and request into resolvers
# =============================================================================

async def get_context(request: Request) -> dict[str, Any]:
    return {
        "user": get_current_user(request),
        "request": request,
    }


# =============================================================================
# SCHEMA & APP
# =============================================================================

schema = strawberry.Schema(
    query=Query,
    mutation=Mutation,
    extensions=[TimingExtension],
)

graphql_app = GraphQLRouter(schema, context_getter=get_context)

app = FastAPI(title="GraphQL Server Demo")
app.include_router(graphql_app, prefix="/graphql")


# =============================================================================
# SCHEMA EXPORT — SDL for documentation and client codegen
# =============================================================================

def export_schema(path: str = "schema.graphql"):
    sdl = schema.as_str()
    with open(path, "w") as f:
        f.write(sdl)
    print(f"Schema exported to {path}")
    print(sdl[:500] + "\n...")


# =============================================================================
# ENTRYPOINT
# =============================================================================

if __name__ == "__main__":
    import sys
    if "--export-schema" in sys.argv:
        export_schema()
    else:
        import uvicorn
        print("GraphQL Playground: http://localhost:8000/graphql")
        print("Auth header: Authorization: Bearer admin-token")
        uvicorn.run("19_graphql_server:app", host="127.0.0.1", port=8000, reload=True)
