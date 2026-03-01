# 10. Code-First with Python (Strawberry)

**Previous**: [GraphQL Clients](./09_GraphQL_Clients.md) | **Next**: [Persisted Queries and Caching](./11_Persisted_Queries_Caching.md)

---

So far, every lesson has used JavaScript/TypeScript with schema-first SDL. But GraphQL is language-agnostic, and Python has excellent GraphQL libraries. Strawberry GraphQL takes the code-first approach to its logical conclusion: you define Python classes with type annotations, and Strawberry generates the GraphQL schema automatically. Combined with FastAPI, this gives you a fully typed, async-capable GraphQL server with minimal boilerplate. If you have ever wished your GraphQL types and your Python types were the same thing, Strawberry is the answer.

**Difficulty**: ⭐⭐⭐

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain the trade-offs between code-first and schema-first approaches for Python
2. Define GraphQL types, queries, mutations, and subscriptions using Strawberry decorators
3. Integrate Strawberry with FastAPI for a production-ready async GraphQL API
4. Implement DataLoaders in Strawberry to solve the N+1 problem
5. Write unit and integration tests for Strawberry schemas

---

## Table of Contents

1. [Code-First vs Schema-First in Python](#1-code-first-vs-schema-first-in-python)
2. [Strawberry Basics](#2-strawberry-basics)
3. [Queries and Input Types](#3-queries-and-input-types)
4. [Mutations](#4-mutations)
5. [FastAPI Integration](#5-fastapi-integration)
6. [Async Resolvers](#6-async-resolvers)
7. [DataLoader in Strawberry](#7-dataloader-in-strawberry)
8. [Subscriptions](#8-subscriptions)
9. [Testing Strawberry Schemas](#9-testing-strawberry-schemas)
10. [Practice Problems](#10-practice-problems)
11. [References](#11-references)

---

## 1. Code-First vs Schema-First in Python

In the Python ecosystem, two approaches dominate:

| Library | Approach | Style |
|---------|----------|-------|
| **Strawberry** | Code-first | Python dataclasses + type hints |
| **Ariadne** | Schema-first | SDL + resolver binding |
| **Graphene** | Code-first | Class-based (older API) |

### Why Code-First?

In schema-first, you write the SDL and then write Python resolvers that must match it. The two can drift apart — your schema says a field returns `String!` but your resolver returns `None`, and you only discover the bug at runtime.

Code-first eliminates this category of bug. Your Python types *are* your GraphQL types:

```python
# This Python class IS the GraphQL type.
# No separate SDL to maintain. No drift possible.
@strawberry.type
class User:
    id: strawberry.ID
    name: str
    email: str
    is_active: bool
```

generates:

```graphql
type User {
  id: ID!
  name: String!
  email: String!
  isActive: Boolean!
}
```

Notice that Strawberry automatically converts `snake_case` Python names to `camelCase` GraphQL names. This follows the conventions of each language.

### When to Prefer Schema-First (Ariadne)

- When the frontend team defines the API contract and hands you an SDL
- When migrating an existing SDL-first API from another language to Python
- When you prefer reading the schema as documentation

---

## 2. Strawberry Basics

### Installation

```bash
pip install strawberry-graphql[debug-server]
```

### Defining Types

Every Strawberry type is a Python dataclass under the hood. The `@strawberry.type` decorator converts it to both a dataclass and a GraphQL type.

```python
# types.py
import strawberry
from datetime import datetime
from enum import Enum

# Enum types
@strawberry.enum
class Role(Enum):
    USER = "user"
    ADMIN = "admin"
    MODERATOR = "moderator"

# Object types
@strawberry.type
class User:
    id: strawberry.ID
    name: str
    email: str
    role: Role
    created_at: datetime

@strawberry.type
class Post:
    id: strawberry.ID
    title: str
    content: str
    published: bool
    author_id: strawberry.ID
    created_at: datetime

    # Field with a custom resolver.
    # The @strawberry.field decorator lets you compute fields
    # that are not stored in the database.
    @strawberry.field
    def excerpt(self) -> str:
        """First 200 characters of the content."""
        return self.content[:200] + "..." if len(self.content) > 200 else self.content

    @strawberry.field
    def reading_time_minutes(self) -> int:
        """Estimated reading time based on word count."""
        word_count = len(self.content.split())
        return max(1, word_count // 200)
```

### Scalar Types Mapping

| Python Type | GraphQL Type |
|-------------|-------------|
| `str` | `String!` |
| `int` | `Int!` |
| `float` | `Float!` |
| `bool` | `Boolean!` |
| `strawberry.ID` | `ID!` |
| `Optional[str]` | `String` (nullable) |
| `list[str]` | `[String!]!` |
| `Optional[list[str]]` | `[String!]` |
| `datetime` | `DateTime!` (custom scalar) |
| `Decimal` | `Decimal!` (custom scalar) |

### Optional and List Fields

```python
from typing import Optional

@strawberry.type
class Post:
    id: strawberry.ID
    title: str
    subtitle: Optional[str] = None           # String (nullable)
    tags: list[str] = strawberry.field(default_factory=list)  # [String!]!
    related_posts: Optional[list["Post"]] = None  # [Post!] (nullable list)
```

---

## 3. Queries and Input Types

### Defining Queries

```python
# schema.py
import strawberry
from typing import Optional

@strawberry.type
class Query:
    @strawberry.field
    def hello(self, name: str = "World") -> str:
        return f"Hello, {name}!"

    @strawberry.field
    def posts(
        self,
        limit: int = 10,
        offset: int = 0,
        published_only: bool = True,
    ) -> list[Post]:
        # In a real app, this would query a database
        query = db.query(PostModel)
        if published_only:
            query = query.filter(PostModel.published == True)
        return [
            Post.from_model(p)
            for p in query.offset(offset).limit(limit).all()
        ]

    @strawberry.field
    def post(self, id: strawberry.ID) -> Optional[Post]:
        model = db.query(PostModel).get(id)
        return Post.from_model(model) if model else None

    @strawberry.field
    def search_posts(self, term: str, limit: int = 10) -> list[Post]:
        models = db.query(PostModel).filter(
            PostModel.title.ilike(f"%{term}%")
        ).limit(limit).all()
        return [Post.from_model(m) for m in models]


schema = strawberry.Schema(query=Query)
```

### Input Types

Input types are used for mutation arguments. They are separate from output types to maintain the GraphQL convention of distinguishing inputs from outputs.

```python
@strawberry.input
class CreatePostInput:
    title: str
    content: str
    published: bool = False
    tags: list[str] = strawberry.field(default_factory=list)

@strawberry.input
class UpdatePostInput:
    title: Optional[str] = None
    content: Optional[str] = None
    published: Optional[bool] = None
    tags: Optional[list[str]] = None
```

The `@strawberry.input` decorator generates a GraphQL `input` type:

```graphql
input CreatePostInput {
  title: String!
  content: String!
  published: Boolean! = false
  tags: [String!]! = []
}
```

---

## 4. Mutations

```python
@strawberry.type
class MutationResponse:
    success: bool
    message: str

@strawberry.type
class PostMutationResponse(MutationResponse):
    post: Optional[Post] = None

@strawberry.type
class Mutation:
    @strawberry.mutation
    def create_post(self, input: CreatePostInput, info: strawberry.types.Info) -> PostMutationResponse:
        # Access context (e.g., current user, database)
        current_user = info.context["current_user"]
        if not current_user:
            return PostMutationResponse(
                success=False,
                message="Authentication required",
                post=None,
            )

        db = info.context["db"]
        post_model = PostModel(
            title=input.title,
            content=input.content,
            published=input.published,
            author_id=current_user.id,
        )
        db.add(post_model)
        db.commit()
        db.refresh(post_model)

        return PostMutationResponse(
            success=True,
            message="Post created successfully",
            post=Post.from_model(post_model),
        )

    @strawberry.mutation
    def update_post(
        self,
        id: strawberry.ID,
        input: UpdatePostInput,
        info: strawberry.types.Info,
    ) -> PostMutationResponse:
        db = info.context["db"]
        current_user = info.context["current_user"]

        post_model = db.query(PostModel).get(id)
        if not post_model:
            return PostMutationResponse(
                success=False, message="Post not found"
            )

        if post_model.author_id != current_user.id:
            return PostMutationResponse(
                success=False, message="Not authorized"
            )

        # Only update fields that were provided
        if input.title is not None:
            post_model.title = input.title
        if input.content is not None:
            post_model.content = input.content
        if input.published is not None:
            post_model.published = input.published
        if input.tags is not None:
            post_model.tags = input.tags

        db.commit()
        db.refresh(post_model)

        return PostMutationResponse(
            success=True,
            message="Post updated",
            post=Post.from_model(post_model),
        )

    @strawberry.mutation
    def delete_post(
        self,
        id: strawberry.ID,
        info: strawberry.types.Info,
    ) -> MutationResponse:
        db = info.context["db"]
        current_user = info.context["current_user"]

        post_model = db.query(PostModel).get(id)
        if not post_model:
            return MutationResponse(success=False, message="Post not found")

        if post_model.author_id != current_user.id and current_user.role != Role.ADMIN:
            return MutationResponse(success=False, message="Not authorized")

        db.delete(post_model)
        db.commit()

        return MutationResponse(success=True, message="Post deleted")


# Complete schema with query and mutation
schema = strawberry.Schema(query=Query, mutation=Mutation)
```

---

## 5. FastAPI Integration

FastAPI and Strawberry are a natural fit — both are async-first, both use Python type hints, and Strawberry provides a first-class FastAPI integration.

```bash
pip install strawberry-graphql[fastapi] uvicorn
```

```python
# main.py
from fastapi import FastAPI, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from strawberry.fastapi import GraphQLRouter
from contextlib import asynccontextmanager

from schema import schema
from database import get_db, engine, Base
from auth import get_current_user

# Create tables on startup
@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)
    yield

app = FastAPI(title="GraphQL API", lifespan=lifespan)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Context function — this is called for every GraphQL request.
# It receives the FastAPI request and can inject dependencies.
async def get_context(
    request: Request,
    db=Depends(get_db),
):
    # Extract and verify the JWT from the Authorization header
    current_user = await get_current_user(request, db)

    return {
        "request": request,
        "db": db,
        "current_user": current_user,
    }

# Mount Strawberry as a FastAPI router
graphql_app = GraphQLRouter(
    schema,
    context_getter=get_context,
)

app.include_router(graphql_app, prefix="/graphql")

# Health check (FastAPI route, not GraphQL)
@app.get("/health")
async def health():
    return {"status": "ok"}
```

### Running the Server

```bash
uvicorn main:app --reload --port 8000
```

Visit `http://localhost:8000/graphql` to access the GraphQL Playground (Strawberry includes one by default).

### Accessing Context in Resolvers

```python
@strawberry.type
class Query:
    @strawberry.field
    def me(self, info: strawberry.types.Info) -> Optional[User]:
        # info.context is the dict returned by get_context
        current_user = info.context["current_user"]
        if not current_user:
            return None
        return User.from_model(current_user)
```

---

## 6. Async Resolvers

Strawberry fully supports async resolvers, which is essential for non-blocking database queries and external API calls.

```python
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.future import select

# Async database setup
engine = create_async_engine("postgresql+asyncpg://user:pass@localhost/mydb")

@strawberry.type
class Query:
    @strawberry.field
    async def posts(
        self,
        info: strawberry.types.Info,
        limit: int = 10,
        offset: int = 0,
    ) -> list[Post]:
        db: AsyncSession = info.context["db"]

        result = await db.execute(
            select(PostModel)
            .where(PostModel.published == True)
            .offset(offset)
            .limit(limit)
            .order_by(PostModel.created_at.desc())
        )
        models = result.scalars().all()
        return [Post.from_model(m) for m in models]

    @strawberry.field
    async def user(
        self,
        info: strawberry.types.Info,
        id: strawberry.ID,
    ) -> Optional[User]:
        db: AsyncSession = info.context["db"]
        result = await db.execute(
            select(UserModel).where(UserModel.id == id)
        )
        model = result.scalar_one_or_none()
        return User.from_model(model) if model else None
```

### Parallel Resolution

When a query requests multiple independent fields, GraphQL resolves them concurrently if the resolvers are async:

```graphql
query Dashboard {
  recentPosts(limit: 5) { title }     # These three resolve
  topAuthors(limit: 3) { name }       # concurrently because
  stats { totalPosts totalUsers }      # they are independent.
}
```

```python
@strawberry.type
class Query:
    @strawberry.field
    async def recent_posts(self, info, limit: int = 5) -> list[Post]:
        # This runs concurrently with top_authors and stats
        ...

    @strawberry.field
    async def top_authors(self, info, limit: int = 3) -> list[User]:
        ...

    @strawberry.field
    async def stats(self, info) -> Stats:
        ...
```

---

## 7. DataLoader in Strawberry

Strawberry provides a built-in DataLoader that integrates with its context system, solving the N+1 problem just like the JavaScript DataLoader library.

### The Problem

```python
# Without DataLoader: N+1 queries
@strawberry.type
class Post:
    author_id: strawberry.ID

    @strawberry.field
    async def author(self, info: strawberry.types.Info) -> User:
        db = info.context["db"]
        # This runs once PER POST. 10 posts = 10 database queries.
        result = await db.execute(
            select(UserModel).where(UserModel.id == self.author_id)
        )
        return User.from_model(result.scalar_one())
```

### The Solution

```python
# dataloaders.py
from strawberry.dataloader import DataLoader
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

async def load_users(keys: list[str]) -> list[User]:
    """Batch load users by their IDs.

    DataLoader guarantees that:
    1. This function is called at most once per execution phase
    2. The returned list must be the same length as `keys`
    3. The returned list must be in the same order as `keys`
    """
    # One query for all requested users, regardless of how many
    # Post.author resolvers triggered this
    async with AsyncSession(engine) as db:
        result = await db.execute(
            select(UserModel).where(UserModel.id.in_(keys))
        )
        user_models = {str(u.id): u for u in result.scalars().all()}

    # Return in the same order as keys, with None for missing IDs
    return [
        User.from_model(user_models[key]) if key in user_models else None
        for key in keys
    ]

async def load_posts_by_author(keys: list[str]) -> list[list[Post]]:
    """Batch load posts grouped by author ID.

    For one-to-many relations, each key maps to a list of results.
    """
    async with AsyncSession(engine) as db:
        result = await db.execute(
            select(PostModel).where(PostModel.author_id.in_(keys))
        )
        all_posts = result.scalars().all()

    # Group posts by author_id
    posts_by_author: dict[str, list[Post]] = {key: [] for key in keys}
    for post in all_posts:
        author_id = str(post.author_id)
        if author_id in posts_by_author:
            posts_by_author[author_id].append(Post.from_model(post))

    return [posts_by_author[key] for key in keys]
```

### Registering DataLoaders in Context

```python
# main.py
from dataloaders import load_users, load_posts_by_author

async def get_context(request: Request, db=Depends(get_db)):
    return {
        "db": db,
        "current_user": await get_current_user(request, db),
        # Create DataLoader instances per request.
        # Each DataLoader caches within a single request.
        "user_loader": DataLoader(load_fn=load_users),
        "posts_by_author_loader": DataLoader(load_fn=load_posts_by_author),
    }
```

### Using DataLoaders in Resolvers

```python
@strawberry.type
class Post:
    id: strawberry.ID
    title: str
    content: str
    author_id: strawberry.ID

    @strawberry.field
    async def author(self, info: strawberry.types.Info) -> User:
        # This call is batched with all other author() calls
        # in the same execution phase.
        return await info.context["user_loader"].load(self.author_id)

@strawberry.type
class User:
    id: strawberry.ID
    name: str

    @strawberry.field
    async def posts(self, info: strawberry.types.Info) -> list[Post]:
        return await info.context["posts_by_author_loader"].load(self.id)
```

With 10 posts, instead of 10 individual `SELECT * FROM users WHERE id = ?` queries, the DataLoader batches them into one `SELECT * FROM users WHERE id IN (?, ?, ?, ...)`.

---

## 8. Subscriptions

Strawberry supports subscriptions using async generators.

```python
import asyncio
from typing import AsyncGenerator

@strawberry.type
class Subscription:
    @strawberry.subscription
    async def message_created(
        self,
        info: strawberry.types.Info,
        channel_id: strawberry.ID,
    ) -> AsyncGenerator[Message, None]:
        """Subscribe to new messages in a channel.

        The async generator yields values; each yield sends
        a result to the client over WebSocket.
        """
        pubsub = info.context["pubsub"]

        async for event in pubsub.subscribe(f"messages:{channel_id}"):
            yield Message(
                id=event["id"],
                content=event["content"],
                author_id=event["author_id"],
                created_at=event["created_at"],
            )

    @strawberry.subscription
    async def countdown(self, seconds: int = 10) -> AsyncGenerator[int, None]:
        """Simple countdown subscription for testing."""
        for i in range(seconds, 0, -1):
            yield i
            await asyncio.sleep(1)
        yield 0


schema = strawberry.Schema(
    query=Query,
    mutation=Mutation,
    subscription=Subscription,
)
```

### FastAPI WebSocket Setup

```python
# main.py
from strawberry.fastapi import GraphQLRouter
from strawberry.subscriptions import GRAPHQL_TRANSPORT_WS_PROTOCOL, GRAPHQL_WS_PROTOCOL

graphql_app = GraphQLRouter(
    schema,
    context_getter=get_context,
    subscription_protocols=[
        GRAPHQL_TRANSPORT_WS_PROTOCOL,  # Modern protocol (graphql-ws)
        GRAPHQL_WS_PROTOCOL,            # Legacy protocol
    ],
)

app.include_router(graphql_app, prefix="/graphql")
```

### Redis-Backed PubSub

For production with multiple server instances, use Redis:

```python
# pubsub.py
import aioredis
import json
from typing import AsyncGenerator

class RedisPubSub:
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_url = redis_url

    async def publish(self, channel: str, data: dict) -> None:
        redis = await aioredis.from_url(self.redis_url)
        await redis.publish(channel, json.dumps(data))
        await redis.close()

    async def subscribe(self, channel: str) -> AsyncGenerator[dict, None]:
        redis = await aioredis.from_url(self.redis_url)
        pubsub = redis.pubsub()
        await pubsub.subscribe(channel)

        try:
            async for message in pubsub.listen():
                if message["type"] == "message":
                    yield json.loads(message["data"])
        finally:
            await pubsub.unsubscribe(channel)
            await redis.close()
```

---

## 9. Testing Strawberry Schemas

Strawberry schemas can be tested without running a server by executing queries directly against the schema object.

### Unit Testing Resolvers

```python
# tests/test_schema.py
import pytest
import strawberry
from schema import schema

@pytest.mark.asyncio
async def test_hello():
    """Test a simple query resolver."""
    result = await schema.execute("{ hello }")
    assert result.errors is None
    assert result.data["hello"] == "Hello, World!"

@pytest.mark.asyncio
async def test_hello_with_name():
    result = await schema.execute('{ hello(name: "Alice") }')
    assert result.errors is None
    assert result.data["hello"] == "Hello, Alice!"

@pytest.mark.asyncio
async def test_posts_query():
    """Test query with context (mock database)."""
    mock_posts = [
        PostModel(id="1", title="Test Post", content="Content", published=True),
    ]

    # Provide a mock context
    context = {
        "db": MockDB(posts=mock_posts),
        "current_user": None,
    }

    result = await schema.execute(
        """
        query {
            posts(limit: 10) {
                id
                title
            }
        }
        """,
        context_value=context,
    )

    assert result.errors is None
    assert len(result.data["posts"]) == 1
    assert result.data["posts"][0]["title"] == "Test Post"
```

### Testing Mutations

```python
@pytest.mark.asyncio
async def test_create_post_authenticated():
    """Test that authenticated users can create posts."""
    context = {
        "db": MockDB(),
        "current_user": MockUser(id="42", role="user"),
    }

    result = await schema.execute(
        """
        mutation {
            createPost(input: { title: "New Post", content: "Body text" }) {
                success
                message
                post {
                    title
                    readingTimeMinutes
                }
            }
        }
        """,
        context_value=context,
    )

    assert result.errors is None
    assert result.data["createPost"]["success"] is True
    assert result.data["createPost"]["post"]["title"] == "New Post"

@pytest.mark.asyncio
async def test_create_post_unauthenticated():
    """Test that unauthenticated users cannot create posts."""
    context = {
        "db": MockDB(),
        "current_user": None,
    }

    result = await schema.execute(
        """
        mutation {
            createPost(input: { title: "New Post", content: "Body" }) {
                success
                message
            }
        }
        """,
        context_value=context,
    )

    assert result.errors is None
    assert result.data["createPost"]["success"] is False
    assert "Authentication required" in result.data["createPost"]["message"]
```

### Integration Testing with FastAPI TestClient

```python
# tests/test_api.py
import pytest
from httpx import AsyncClient, ASGITransport
from main import app

@pytest.mark.asyncio
async def test_graphql_endpoint():
    """Test the full HTTP request path."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/graphql",
            json={
                "query": "{ hello }",
            },
        )

    assert response.status_code == 200
    data = response.json()
    assert data["data"]["hello"] == "Hello, World!"

@pytest.mark.asyncio
async def test_graphql_with_auth():
    """Test authenticated request."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/graphql",
            json={
                "query": "{ me { name email } }",
            },
            headers={
                "Authorization": "Bearer valid-test-token",
            },
        )

    assert response.status_code == 200
    data = response.json()
    assert data["data"]["me"]["name"] == "Test User"
```

### Testing Subscriptions

```python
@pytest.mark.asyncio
async def test_countdown_subscription():
    """Test subscription by collecting all yielded values."""
    result = await schema.execute(
        """
        subscription {
            countdown(seconds: 3)
        }
        """
    )

    # Subscriptions return an async iterator
    values = []
    async for event in result:
        values.append(event.data["countdown"])

    assert values == [3, 2, 1, 0]
```

---

## 10. Practice Problems

### Exercise 1: Blog API Types (Beginner)

Define Strawberry types for a blog application with the following requirements:

- `User`: id, username, email, bio (optional), avatar_url (optional), role (enum: READER, AUTHOR, ADMIN), created_at
- `Post`: id, title, slug (computed from title), content, published, view_count, author_id, tags (list of strings), created_at, updated_at
- `Comment`: id, content, author_id, post_id, parent_id (optional, for nested comments), created_at
- Input types: `CreatePostInput`, `UpdatePostInput`, `CreateCommentInput`

Include computed fields: `Post.slug` (lowercase, hyphenated title), `Post.reading_time_minutes`, and `Post.is_new` (published within the last 7 days).

### Exercise 2: FastAPI Integration with Auth (Intermediate)

Build a complete Strawberry + FastAPI setup with:

1. JWT-based authentication (extract token from `Authorization` header in context)
2. A `me` query that returns the current user (or error if unauthenticated)
3. A `createPost` mutation that requires authentication
4. A `deletePost` mutation that requires the current user to be the author or an admin
5. CORS configured for `http://localhost:3000`

Provide the complete `main.py`, `auth.py`, and `schema.py` files.

### Exercise 3: DataLoader Implementation (Intermediate)

Given these types and a PostgreSQL database:

```python
@strawberry.type
class Author:
    id: strawberry.ID
    name: str
    books: list["Book"]  # One-to-many

@strawberry.type
class Book:
    id: strawberry.ID
    title: str
    author: Author        # Many-to-one
    categories: list["Category"]  # Many-to-many

@strawberry.type
class Category:
    id: strawberry.ID
    name: str
    books: list["Book"]  # Many-to-many
```

Implement DataLoaders for:

1. Loading an author by book's `author_id` (many-to-one)
2. Loading books by author's `id` (one-to-many)
3. Loading categories for a book (many-to-many via `book_categories` join table)
4. Loading books for a category (many-to-many)

Write the batch functions, context setup, and resolver integration.

### Exercise 4: Subscription with PubSub (Advanced)

Build a real-time notification system using Strawberry subscriptions:

1. Define a `Notification` type with id, type (enum), message, recipient_id, read, created_at
2. Implement a `notificationReceived` subscription that filters by the authenticated user's ID
3. Implement a `sendNotification` mutation that publishes to the subscription
4. Use an in-memory PubSub for development and explain how to swap to Redis for production
5. Write a test that verifies a mutation triggers the subscription

### Exercise 5: Complete Test Suite (Advanced)

Write a comprehensive test suite for a Strawberry-based task management API:

- Types: `Project`, `Task`, `User`, `Label`
- Operations: CRUD for all types, plus `assignTask`, `moveTask` (change status), `addLabel`

Your test suite should include:

1. Unit tests for computed fields (e.g., `Project.completion_percentage`)
2. Query tests with mock context (happy path + error cases)
3. Mutation tests verifying authorization (owner vs non-owner vs admin)
4. Integration tests using FastAPI's `AsyncClient`
5. At least one subscription test

Use `pytest`, `pytest-asyncio`, and mock objects. Aim for at least 15 test cases.

---

## 11. References

- Strawberry GraphQL documentation — https://strawberry.rocks/docs
- Strawberry + FastAPI — https://strawberry.rocks/docs/integrations/fastapi
- Strawberry DataLoader — https://strawberry.rocks/docs/guides/dataloaders
- FastAPI documentation — https://fastapi.tiangolo.com
- Ariadne (schema-first alternative) — https://ariadnegraphql.org
- graphql-core (Python reference implementation) — https://github.com/graphql-python/graphql-core

---

**Previous**: [GraphQL Clients](./09_GraphQL_Clients.md) | **Next**: [Persisted Queries and Caching](./11_Persisted_Queries_Caching.md)
