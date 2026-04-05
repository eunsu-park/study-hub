# 19. GraphQL Resolvers

**Previous**: [GraphQL Schema Design](./18_GraphQL_Schema_Design.md) | **Next**: [GraphQL Subscriptions](./20_GraphQL_Subscriptions.md)

**Difficulty**: ⭐⭐⭐

---

## Learning Objectives

- Explain the resolver execution model and how GraphQL traverses the type graph
- Implement resolvers in Python using Strawberry with proper context and info usage
- Identify and solve the N+1 problem using DataLoader for batched data fetching
- Apply resolver-level authentication and authorization patterns
- Structure resolver code for maintainability using service layers and dependency injection
- Implement field-level error handling within resolvers

---

## Table of Contents

1. [How Resolvers Work](#1-how-resolvers-work)
2. [Resolver Anatomy](#2-resolver-anatomy)
3. [Context and Info Objects](#3-context-and-info-objects)
4. [The N+1 Problem](#4-the-n-plus-1-problem)
5. [DataLoader Pattern](#5-dataloader-pattern)
6. [Resolver Authentication and Authorization](#6-resolver-authentication-and-authorization)
7. [Error Handling in Resolvers](#7-error-handling-in-resolvers)
8. [Resolver Patterns and Architecture](#8-resolver-patterns-and-architecture)
9. [Performance Considerations](#9-performance-considerations)
10. [Exercises](#10-exercises)
11. [References](#11-references)

---

## 1. How Resolvers Work

A resolver is a function that populates data for a single field in your schema. When a query arrives, the GraphQL execution engine calls resolvers for each field in a depth-first, breadth-first hybrid traversal.

### Execution Model

```
Query:
{
  user(id: "1") {      # 1. Calls Query.user resolver
    username            # 2. Calls User.username resolver (trivial)
    posts(first: 3) {   # 3. Calls User.posts resolver
      title             # 4. Calls Post.title resolver (trivial)
      author {          # 5. Calls Post.author resolver
        username        # 6. Calls User.username resolver (trivial)
      }
    }
  }
}
```

### Default Resolvers

For simple fields that map directly to attributes, GraphQL uses **default resolvers** (also called trivial resolvers). These return `obj.field_name` or `obj["field_name"]` automatically.

```python
# You do NOT need to write a resolver for this:
@strawberry.type
class User:
    username: str  # Default resolver returns self.username
    email: str     # Default resolver returns self.email
```

You only write custom resolvers for fields that require:
- Database queries
- Computed values
- Data transformation
- Authorization checks

---

## 2. Resolver Anatomy

### Basic Resolver in Strawberry

```python
import strawberry
from typing import Optional


@strawberry.type
class Query:
    @strawberry.field
    def user(self, id: strawberry.ID) -> Optional["User"]:
        """Fetch a user by ID."""
        return user_repository.find_by_id(id)

    @strawberry.field
    def users(
        self,
        limit: int = 10,
        offset: int = 0,
    ) -> list["User"]:
        """Fetch a paginated list of users."""
        return user_repository.find_all(limit=limit, offset=offset)
```

### Resolvers on Object Types

```python
@strawberry.type
class User:
    id: strawberry.ID
    username: str
    email: str

    @strawberry.field
    def posts(
        self,
        info: strawberry.types.Info,
        first: int = 10,
        status: Optional[PostStatus] = None,
    ) -> list["Post"]:
        """Resolve the user's posts."""
        filters = {"author_id": self.id}
        if status:
            filters["status"] = status.value
        return post_repository.find_by(filters, limit=first)

    @strawberry.field
    def post_count(self) -> int:
        """Computed field: total number of posts."""
        return post_repository.count_by_author(self.id)

    @strawberry.field
    def full_name(self) -> str:
        """Computed field: concatenate first and last name."""
        return f"{self.first_name} {self.last_name}"
```

### Resolver Return Types

| Return | GraphQL Effect |
|--------|---------------|
| Value | Field resolves to that value |
| `None` | Field resolves to `null` (must be nullable) |
| Exception | Error added to `errors` array; field resolves to `null` |
| List | Field resolves to a list |
| Awaitable | Async resolver; execution awaits the result |

---

## 3. Context and Info Objects

### The Context Object

Context is shared across all resolvers in a single request. It typically contains:

```python
from dataclasses import dataclass
from strawberry.fastapi import GraphQLRouter


@dataclass
class RequestContext:
    db: Database
    current_user: Optional[User]
    request: Request
    dataloaders: "DataLoaders"


async def get_context(request: Request) -> RequestContext:
    """Build context for each GraphQL request."""
    db = get_database()
    token = request.headers.get("Authorization", "").replace("Bearer ", "")
    current_user = await authenticate(token, db) if token else None

    return RequestContext(
        db=db,
        current_user=current_user,
        request=request,
        dataloaders=DataLoaders(db),
    )


graphql_router = GraphQLRouter(
    schema,
    context_getter=get_context,
)
```

### Using Context in Resolvers

```python
@strawberry.type
class Query:
    @strawberry.field
    def me(self, info: strawberry.types.Info) -> Optional["User"]:
        """Return the currently authenticated user."""
        ctx: RequestContext = info.context
        return ctx.current_user

    @strawberry.field
    def user(self, info: strawberry.types.Info, id: strawberry.ID) -> Optional["User"]:
        ctx: RequestContext = info.context
        return ctx.db.users.find_by_id(id)
```

### The Info Object

The `info` parameter provides metadata about the current execution:

```python
@strawberry.field
def user(self, info: strawberry.types.Info, id: strawberry.ID) -> Optional["User"]:
    # Access context
    ctx = info.context

    # Check which fields are requested (for optimization)
    selected_fields = [
        selection.name
        for selection in info.selected_fields[0].selections
    ]

    # Only join posts table if posts field is requested
    include_posts = "posts" in selected_fields
    return user_repository.find_by_id(id, include_posts=include_posts)
```

### Lookahead Optimization

Inspect requested fields to optimize database queries:

```python
def _get_requested_fields(info: strawberry.types.Info) -> set[str]:
    """Extract top-level requested field names."""
    fields = set()
    for selection in info.selected_fields:
        for sub in selection.selections:
            fields.add(sub.name)
    return fields


@strawberry.type
class Query:
    @strawberry.field
    def posts(self, info: strawberry.types.Info, first: int = 10) -> list["Post"]:
        requested = _get_requested_fields(info)
        query = post_repository.query()

        # Only join author if requested
        if "author" in requested:
            query = query.join(User)

        # Only join comments if requested
        if "comments" in requested:
            query = query.join(Comment)

        return query.limit(first).all()
```

---

## 4. The N+1 Problem

The N+1 problem is the most common performance pitfall in GraphQL.

### How It Happens

```graphql
query {
  posts(first: 10) {     # 1 query: SELECT * FROM posts LIMIT 10
    title
    author {              # 10 queries: SELECT * FROM users WHERE id = ?
      username            #   (one per post)
    }
  }
}
```

Total: **11 queries** instead of 2.

### Why It Happens

Each `Post.author` resolver executes independently:

```python
@strawberry.type
class Post:
    author_id: strawberry.ID

    @strawberry.field
    def author(self) -> "User":
        # This runs once PER POST
        return db.query(User).filter(User.id == self.author_id).first()
```

### The Problem at Scale

| Posts | Without Batching | With Batching |
|-------|-----------------|---------------|
| 10 | 11 queries | 2 queries |
| 100 | 101 queries | 2 queries |
| 1,000 | 1,001 queries | 2 queries |

Nested queries make it worse:

```graphql
query {
  posts(first: 10) {           # 1 query
    author {                    # 10 queries
      posts(first: 5) {        # 10 queries
        comments(first: 3) {   # 50 queries
          author { username }   # 150 queries
        }
      }
    }
  }
}
# Total: 221 queries!
```

---

## 5. DataLoader Pattern

DataLoader solves the N+1 problem by batching and caching database requests.

### How DataLoader Works

1. **Collect**: During a single execution tick, DataLoader collects all requested keys
2. **Batch**: At the end of the tick, it calls the batch function once with all keys
3. **Cache**: Results are cached for the duration of the request (per-request cache)

### Implementation with Strawberry

```python
from strawberry.dataloader import DataLoader
from typing import Any


# Batch loading function
async def load_users(keys: list[str]) -> list[User]:
    """Load multiple users in a single query."""
    users = await db.query(User).filter(User.id.in_(keys)).all()
    # IMPORTANT: Return results in the same order as keys
    user_map = {str(u.id): u for u in users}
    return [user_map.get(key) for key in keys]


async def load_posts_by_author(keys: list[str]) -> list[list[Post]]:
    """Load posts grouped by author ID."""
    posts = await db.query(Post).filter(Post.author_id.in_(keys)).all()
    posts_by_author: dict[str, list[Post]] = {}
    for post in posts:
        posts_by_author.setdefault(str(post.author_id), []).append(post)
    return [posts_by_author.get(key, []) for key in keys]


# DataLoader container
@dataclass
class DataLoaders:
    def __init__(self, db: Database):
        self.user_loader = DataLoader(load_fn=load_users)
        self.posts_by_author_loader = DataLoader(load_fn=load_posts_by_author)
```

### Using DataLoaders in Resolvers

```python
@strawberry.type
class Post:
    id: strawberry.ID
    title: str
    author_id: strawberry.ID

    @strawberry.field
    async def author(self, info: strawberry.types.Info) -> "User":
        ctx: RequestContext = info.context
        return await ctx.dataloaders.user_loader.load(self.author_id)


@strawberry.type
class User:
    id: strawberry.ID
    username: str

    @strawberry.field
    async def posts(self, info: strawberry.types.Info) -> list["Post"]:
        ctx: RequestContext = info.context
        return await ctx.dataloaders.posts_by_author_loader.load(self.id)
```

### Before and After

```
# Without DataLoader (N+1)
SELECT * FROM posts LIMIT 10;
SELECT * FROM users WHERE id = 1;
SELECT * FROM users WHERE id = 2;
SELECT * FROM users WHERE id = 3;
...
SELECT * FROM users WHERE id = 10;
# 11 queries

# With DataLoader (batched)
SELECT * FROM posts LIMIT 10;
SELECT * FROM users WHERE id IN (1, 2, 3, ..., 10);
# 2 queries
```

### DataLoader Rules

| Rule | Description |
|------|-------------|
| **Per-request instances** | Create new DataLoader instances for each request |
| **Key ordering** | Return results in the same order as input keys |
| **Key count** | Return exactly one result per key (use `None` for missing) |
| **No side effects** | Batch functions should be pure reads |
| **Error handling** | Return `Error` objects for individual key failures |

### Nested DataLoaders

```python
async def load_comment_counts(keys: list[str]) -> list[int]:
    """Batch load comment counts for posts."""
    rows = await db.execute(
        "SELECT post_id, COUNT(*) as cnt FROM comments "
        "WHERE post_id = ANY($1) GROUP BY post_id",
        keys,
    )
    count_map = {str(r["post_id"]): r["cnt"] for r in rows}
    return [count_map.get(key, 0) for key in keys]


@strawberry.type
class Post:
    id: strawberry.ID

    @strawberry.field
    async def comment_count(self, info: strawberry.types.Info) -> int:
        ctx: RequestContext = info.context
        return await ctx.dataloaders.comment_count_loader.load(self.id)
```

---

## 6. Resolver Authentication and Authorization

### Authentication via Context

```python
def require_auth(info: strawberry.types.Info) -> User:
    """Helper to enforce authentication."""
    ctx: RequestContext = info.context
    if ctx.current_user is None:
        raise PermissionError("Authentication required")
    return ctx.current_user


@strawberry.type
class Query:
    @strawberry.field
    def me(self, info: strawberry.types.Info) -> "User":
        return require_auth(info)
```

### Field-Level Authorization

```python
import functools


def require_role(*roles: str):
    """Decorator to enforce role-based access on resolvers."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, info: strawberry.types.Info, *args, **kwargs):
            user = require_auth(info)
            if user.role not in roles:
                raise PermissionError(
                    f"Requires one of: {', '.join(roles)}"
                )
            return func(self, info, *args, **kwargs)
        return wrapper
    return decorator


@strawberry.type
class Query:
    @strawberry.field
    @require_role("ADMIN", "SUPER_ADMIN")
    def admin_dashboard(self, info: strawberry.types.Info) -> "Dashboard":
        return dashboard_service.get_dashboard()
```

### Strawberry Permissions

```python
from strawberry.permission import BasePermission
from strawberry.types import Info


class IsAuthenticated(BasePermission):
    message = "User is not authenticated"

    def has_permission(self, source: Any, info: Info, **kwargs) -> bool:
        ctx: RequestContext = info.context
        return ctx.current_user is not None


class IsAdmin(BasePermission):
    message = "Admin access required"

    def has_permission(self, source: Any, info: Info, **kwargs) -> bool:
        ctx: RequestContext = info.context
        return (
            ctx.current_user is not None
            and ctx.current_user.role == "ADMIN"
        )


@strawberry.type
class Query:
    @strawberry.field(permission_classes=[IsAuthenticated])
    def me(self, info: strawberry.types.Info) -> "User":
        return info.context.current_user

    @strawberry.field(permission_classes=[IsAdmin])
    def all_users(self, info: strawberry.types.Info) -> list["User"]:
        return info.context.db.users.all()
```

### Data-Level Authorization

Filter results based on the viewer's permissions:

```python
@strawberry.type
class User:
    @strawberry.field
    def email(self, info: strawberry.types.Info) -> Optional[str]:
        """Email is only visible to the user themselves or admins."""
        viewer = info.context.current_user
        if viewer and (viewer.id == self.id or viewer.role == "ADMIN"):
            return self._email
        return None  # Field resolves to null for unauthorized viewers
```

---

## 7. Error Handling in Resolvers

### Top-Level Errors vs. User Errors

| Type | When | How |
|------|------|-----|
| **GraphQL errors** | Unexpected failures (DB down, bug) | Raised as exceptions; appear in `errors` array |
| **User errors** | Expected failures (validation, not found) | Returned in payload `userErrors` field |

### Exception-Based Error Handling

```python
class NotFoundError(Exception):
    def __init__(self, entity: str, id: str):
        self.message = f"{entity} with id '{id}' not found"
        self.extensions = {"code": "NOT_FOUND", "entity": entity, "id": id}


@strawberry.type
class Query:
    @strawberry.field
    def post(self, id: strawberry.ID) -> "Post":
        post = post_repository.find_by_id(id)
        if post is None:
            raise NotFoundError("Post", id)
        return post
```

### Structured Error Payloads (Preferred for Mutations)

```python
@strawberry.type
class UserError:
    field: list[str]
    message: str
    code: str


@strawberry.type
class CreatePostPayload:
    post: Optional["Post"] = None
    user_errors: list[UserError] = strawberry.field(default_factory=list)


@strawberry.type
class Mutation:
    @strawberry.mutation
    def create_post(
        self,
        info: strawberry.types.Info,
        input: CreatePostInput,
    ) -> CreatePostPayload:
        errors = []

        if not input.title.strip():
            errors.append(UserError(
                field=["input", "title"],
                message="Title cannot be blank",
                code="BLANK",
            ))

        if len(input.content) < 10:
            errors.append(UserError(
                field=["input", "content"],
                message="Content must be at least 10 characters",
                code="TOO_SHORT",
            ))

        if errors:
            return CreatePostPayload(user_errors=errors)

        post = post_repository.create(
            title=input.title,
            content=input.content,
            author_id=info.context.current_user.id,
        )
        return CreatePostPayload(post=post)
```

---

## 8. Resolver Patterns and Architecture

### Service Layer Pattern

Keep resolvers thin by delegating business logic to services:

```python
# services/post_service.py
class PostService:
    def __init__(self, db: Database, current_user: User):
        self.db = db
        self.current_user = current_user

    def create_post(self, input: CreatePostInput) -> CreatePostPayload:
        errors = self._validate(input)
        if errors:
            return CreatePostPayload(user_errors=errors)

        post = Post(
            title=input.title,
            content=input.content,
            author_id=self.current_user.id,
        )
        self.db.posts.save(post)
        return CreatePostPayload(post=post)

    def _validate(self, input: CreatePostInput) -> list[UserError]:
        errors = []
        if not input.title.strip():
            errors.append(UserError(["input", "title"], "Required", "BLANK"))
        return errors


# Resolver: thin wrapper
@strawberry.type
class Mutation:
    @strawberry.mutation
    def create_post(
        self, info: strawberry.types.Info, input: CreatePostInput
    ) -> CreatePostPayload:
        service = PostService(info.context.db, info.context.current_user)
        return service.create_post(input)
```

### Repository Pattern

Abstract data access behind repositories:

```python
class UserRepository:
    def __init__(self, db: Database):
        self.db = db

    async def find_by_id(self, id: str) -> Optional[User]:
        return await self.db.query(User).filter(User.id == id).first()

    async def find_by_ids(self, ids: list[str]) -> list[User]:
        return await self.db.query(User).filter(User.id.in_(ids)).all()

    async def find_by_email(self, email: str) -> Optional[User]:
        return await self.db.query(User).filter(User.email == email).first()

    async def save(self, user: User) -> User:
        self.db.add(user)
        await self.db.commit()
        return user
```

### Middleware Pattern

Apply cross-cutting concerns (logging, timing, error handling):

```python
from strawberry.extensions import SchemaExtension
import time
import logging

logger = logging.getLogger("graphql")


class QueryLoggingExtension(SchemaExtension):
    def on_operation(self):
        start = time.monotonic()
        yield
        duration = time.monotonic() - start
        logger.info(
            "GraphQL operation completed",
            extra={
                "operation_name": self.execution_context.operation_name,
                "duration_ms": round(duration * 1000, 2),
            },
        )


class QueryDepthExtension(SchemaExtension):
    MAX_DEPTH = 10

    def on_operation(self):
        depth = self._calculate_depth(self.execution_context.query)
        if depth > self.MAX_DEPTH:
            raise ValueError(
                f"Query depth {depth} exceeds maximum of {self.MAX_DEPTH}"
            )
        yield


schema = strawberry.Schema(
    query=Query,
    mutation=Mutation,
    extensions=[QueryLoggingExtension, QueryDepthExtension],
)
```

---

## 9. Performance Considerations

### Resolver Execution Order

```
Query.posts        →  [Post1, Post2, Post3]
Post1.author       ┐
Post2.author       ├→ Batched by DataLoader → 1 SQL query
Post3.author       ┘
Post1.comments     ┐
Post2.comments     ├→ Batched by DataLoader → 1 SQL query
Post3.comments     ┘
```

### Async Resolvers

Use async resolvers for I/O-bound operations:

```python
@strawberry.type
class Query:
    @strawberry.field
    async def user(self, info: strawberry.types.Info, id: strawberry.ID) -> Optional["User"]:
        return await info.context.db.users.find_by_id(id)
```

### Caching Strategies

| Level | Strategy | Implementation |
|-------|----------|----------------|
| **Request** | DataLoader cache | Built into DataLoader |
| **Application** | LRU / Redis cache | Cache popular queries |
| **HTTP** | CDN / reverse proxy | Cache persisted queries by hash |

### Database Query Optimization

```python
# Use JOINs based on selected fields
@strawberry.field
async def posts(self, info: strawberry.types.Info, first: int = 10) -> list["Post"]:
    query = select(PostModel).limit(first)

    # Check if author is requested
    requested = get_requested_fields(info)
    if "author" in requested:
        query = query.options(selectinload(PostModel.author))
    if "comments" in requested:
        query = query.options(selectinload(PostModel.comments))

    return await info.context.db.execute(query)
```

---

## 10. Exercises

### Exercise 1: DataLoader Implementation

Given the following schema, implement DataLoaders to eliminate N+1 queries:

```graphql
type Query {
  orders(first: Int): [Order!]!
}

type Order {
  id: ID!
  customer: Customer!
  items: [OrderItem!]!
}

type OrderItem {
  product: Product!
  quantity: Int!
}
```

Write batch functions for: `load_customers`, `load_items_by_order`, `load_products`.

### Exercise 2: Authorization Resolver

Implement a resolver for `User.email` that returns the email only if:
- The viewer is the user themselves
- The viewer has ADMIN role
- The user's `emailPublic` setting is true

Otherwise, return `null`.

### Exercise 3: Error Handling

Implement a `transferFunds` mutation with comprehensive error handling:
- Validate source and destination account existence
- Check sufficient balance
- Detect self-transfer attempts
- Return structured `UserError` objects for each case

### Exercise 4: Service Layer Refactoring

Refactor the following "fat resolver" into a clean service layer:

```python
@strawberry.mutation
def create_order(self, info, input):
    user = info.context.current_user
    if not user:
        raise Exception("Not authenticated")
    items = []
    total = 0
    for item_input in input.items:
        product = db.query(Product).get(item_input.product_id)
        if not product:
            raise Exception(f"Product {item_input.product_id} not found")
        if product.stock < item_input.quantity:
            raise Exception("Insufficient stock")
        items.append(OrderItem(product=product, quantity=item_input.quantity))
        total += product.price * item_input.quantity
    order = Order(user_id=user.id, items=items, total=total)
    db.add(order)
    db.commit()
    return order
```

---

## 11. References

### Libraries
- [Strawberry DataLoader Documentation](https://strawberry.rocks/docs/guides/dataloaders)
- [graphql-core — Python GraphQL Implementation](https://github.com/graphql-python/graphql-core)
- [Facebook DataLoader (Original JS Implementation)](https://github.com/graphql/dataloader)

### Articles
- "DataLoader — Source Code Walkthrough" by Lee Byron
- "Solving the N+1 Problem for GraphQL through Batching" — Apollo Blog
- "Authentication and Authorization in GraphQL" — Escape.tech

### Videos
- "GraphQL Resolvers: Best Practices" — GraphQL Conf
- Lee Byron, "Exploring GraphQL" (2015 React.js Conf)

---

**License**: CC BY-NC 4.0
