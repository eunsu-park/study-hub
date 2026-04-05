# 21. GraphQL Server Implementation

**Previous**: [GraphQL Subscriptions](./20_GraphQL_Subscriptions.md) | **Next**: [GraphQL Federation](./22_GraphQL_Federation.md)

**Difficulty**: ⭐⭐⭐

---

## Learning Objectives

- Compare code-first and schema-first development approaches and choose appropriately
- Build a production-ready GraphQL server with Strawberry (Python) and FastAPI
- Implement Apollo Server in Node.js/TypeScript for polyglot comparison
- Apply middleware patterns for logging, error handling, and performance tracing
- Configure file uploads, CORS, and request batching in a GraphQL server
- Structure a GraphQL server project for maintainability at scale

---

## Table of Contents

1. [Code-First vs. Schema-First](#1-code-first-vs-schema-first)
2. [Strawberry (Python) Deep Dive](#2-strawberry-python-deep-dive)
3. [Apollo Server (Node.js)](#3-apollo-server-nodejs)
4. [Middleware and Extensions](#4-middleware-and-extensions)
5. [File Uploads](#5-file-uploads)
6. [Request Batching](#6-request-batching)
7. [CORS and Security Headers](#7-cors-and-security-headers)
8. [Project Structure](#8-project-structure)
9. [Database Integration](#9-database-integration)
10. [Deployment](#10-deployment)
11. [Exercises](#11-exercises)
12. [References](#12-references)

---

## 1. Code-First vs. Schema-First

### Schema-First Approach

Write the SDL file first, then implement resolvers to match:

```graphql
# schema.graphql
type User {
  id: ID!
  username: String!
  email: String!
  posts: [Post!]!
}

type Query {
  user(id: ID!): User
}
```

```python
# resolvers.py (Ariadne example)
from ariadne import QueryType

query = QueryType()

@query.field("user")
def resolve_user(_, info, id):
    return db.users.find_by_id(id)
```

### Code-First Approach

Define types in your programming language; the schema is generated:

```python
# types.py (Strawberry)
import strawberry

@strawberry.type
class User:
    id: strawberry.ID
    username: str
    email: str

    @strawberry.field
    def posts(self) -> list["Post"]:
        return post_repo.find_by_author(self.id)
```

### Comparison

| Aspect | Schema-First | Code-First |
|--------|-------------|------------|
| Source of truth | `.graphql` files | Python/TS code |
| Type safety | External validation | Compile-time / type-checker |
| Refactoring | Manual sync | IDE-assisted |
| Onboarding | Easy (readable SDL) | Requires language knowledge |
| Tooling | GraphQL-specific | Language ecosystem |
| Libraries | Ariadne, Apollo Server | Strawberry, TypeGraphQL |

### Recommendation

| Team / Project | Approach |
|---------------|----------|
| Python backend with type hints | Code-first (Strawberry) |
| TypeScript fullstack | Code-first (TypeGraphQL) or schema-first (Apollo) |
| Multi-language teams | Schema-first (shared SDL) |
| API-first organizations | Schema-first |
| Rapid prototyping | Code-first |

---

## 2. Strawberry (Python) Deep Dive

### Why Strawberry?

- Built on Python dataclasses and type hints
- First-class async/await support
- FastAPI and Django integration
- Active development and community
- Full subscription support

### Complete Server Setup

```python
# main.py
import strawberry
from fastapi import FastAPI
from strawberry.fastapi import GraphQLRouter
from contextlib import asynccontextmanager

from schema import schema
from context import get_context
from database import engine, create_tables


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle management."""
    await create_tables(engine)
    yield
    await engine.dispose()


app = FastAPI(
    title="GraphQL API",
    version="1.0.0",
    lifespan=lifespan,
)

graphql_router = GraphQLRouter(
    schema,
    context_getter=get_context,
    graphql_ide="graphiql",  # or "apollo-sandbox"
)

app.include_router(graphql_router, prefix="/graphql")


@app.get("/health")
async def health():
    return {"status": "healthy"}
```

### Schema Assembly

```python
# schema.py
import strawberry
from queries.user_queries import UserQuery
from queries.post_queries import PostQuery
from mutations.user_mutations import UserMutation
from mutations.post_mutations import PostMutation
from subscriptions.post_subscriptions import PostSubscription


@strawberry.type
class Query(UserQuery, PostQuery):
    """Root query type — combines all query resolvers."""
    pass


@strawberry.type
class Mutation(UserMutation, PostMutation):
    """Root mutation type — combines all mutation resolvers."""
    pass


@strawberry.type
class Subscription(PostSubscription):
    """Root subscription type."""
    pass


schema = strawberry.Schema(
    query=Query,
    mutation=Mutation,
    subscription=Subscription,
)
```

### Type Definitions with Resolvers

```python
# types/user.py
import strawberry
from typing import Optional, TYPE_CHECKING
from datetime import datetime

if TYPE_CHECKING:
    from types.post import Post


@strawberry.type
class User:
    id: strawberry.ID
    username: str
    email: str
    bio: Optional[str] = None
    created_at: datetime

    @strawberry.field
    async def posts(
        self,
        info: strawberry.types.Info,
        first: int = 10,
        after: Optional[str] = None,
    ) -> "PostConnection":
        loader = info.context.dataloaders.posts_by_author
        posts = await loader.load(self.id)
        # Apply pagination
        return paginate(posts, first=first, after=after)

    @strawberry.field
    async def post_count(self, info: strawberry.types.Info) -> int:
        loader = info.context.dataloaders.post_count_by_author
        return await loader.load(self.id)
```

### Enum and Input Types

```python
# types/enums.py
import strawberry
from enum import Enum


@strawberry.enum
class PostStatus(Enum):
    DRAFT = "draft"
    PUBLISHED = "published"
    ARCHIVED = "archived"


@strawberry.enum
class SortDirection(Enum):
    ASC = "asc"
    DESC = "desc"


# types/inputs.py
@strawberry.input
class CreatePostInput:
    title: str
    content: str
    status: PostStatus = PostStatus.DRAFT
    tags: list[str] = strawberry.field(default_factory=list)


@strawberry.input
class UpdatePostInput:
    title: Optional[str] = None
    content: Optional[str] = None
    status: Optional[PostStatus] = None
    tags: Optional[list[str]] = None


@strawberry.input
class PostFilterInput:
    status: Optional[PostStatus] = None
    author_id: Optional[strawberry.ID] = None
    tag: Optional[str] = None
    search: Optional[str] = None
```

---

## 3. Apollo Server (Node.js)

For comparison, here is the equivalent setup in Apollo Server v4:

### Setup

```bash
npm install @apollo/server graphql
npm install -D typescript @types/node ts-node
```

### Schema-First with Apollo

```typescript
// src/schema.ts
export const typeDefs = `#graphql
  type User {
    id: ID!
    username: String!
    email: String!
    posts(first: Int = 10): [Post!]!
  }

  type Post {
    id: ID!
    title: String!
    content: String!
    status: PostStatus!
    author: User!
    createdAt: DateTime!
  }

  enum PostStatus {
    DRAFT
    PUBLISHED
    ARCHIVED
  }

  scalar DateTime

  type Query {
    user(id: ID!): User
    users(limit: Int = 10, offset: Int = 0): [User!]!
    post(id: ID!): Post
  }

  type Mutation {
    createPost(input: CreatePostInput!): CreatePostPayload!
  }

  input CreatePostInput {
    title: String!
    content: String!
    status: PostStatus = DRAFT
  }

  type CreatePostPayload {
    post: Post
    userErrors: [UserError!]!
  }

  type UserError {
    field: [String!]!
    message: String!
  }
`;
```

### Resolvers

```typescript
// src/resolvers.ts
import { Resolvers } from './generated/types';
import DataLoader from 'dataloader';

export const resolvers: Resolvers = {
  Query: {
    user: async (_, { id }, { dataSources }) => {
      return dataSources.userAPI.getUser(id);
    },
    users: async (_, { limit, offset }, { dataSources }) => {
      return dataSources.userAPI.getUsers(limit, offset);
    },
  },

  User: {
    posts: async (user, { first }, { dataSources }) => {
      return dataSources.postAPI.getPostsByAuthor(user.id, first);
    },
  },

  Post: {
    author: async (post, _, { loaders }) => {
      return loaders.userLoader.load(post.authorId);
    },
  },

  Mutation: {
    createPost: async (_, { input }, { dataSources, currentUser }) => {
      if (!currentUser) {
        return {
          post: null,
          userErrors: [{ field: [], message: 'Authentication required' }],
        };
      }
      const post = await dataSources.postAPI.create({
        ...input,
        authorId: currentUser.id,
      });
      return { post, userErrors: [] };
    },
  },
};
```

### Server Entry Point

```typescript
// src/index.ts
import { ApolloServer } from '@apollo/server';
import { startStandaloneServer } from '@apollo/server/standalone';
import { typeDefs } from './schema';
import { resolvers } from './resolvers';

const server = new ApolloServer({ typeDefs, resolvers });

const { url } = await startStandaloneServer(server, {
  listen: { port: 4000 },
  context: async ({ req }) => {
    const token = req.headers.authorization?.replace('Bearer ', '');
    const currentUser = token ? await authenticate(token) : null;
    return {
      currentUser,
      dataSources: createDataSources(),
      loaders: createLoaders(),
    };
  },
});

console.log(`Server ready at ${url}`);
```

---

## 4. Middleware and Extensions

### Strawberry Extensions

```python
from strawberry.extensions import SchemaExtension
from strawberry.extensions.tracing import ApolloTracingExtension
import time
import logging

logger = logging.getLogger("graphql")


class TimingExtension(SchemaExtension):
    """Log execution time for each operation."""

    def on_operation(self):
        start = time.monotonic()
        yield
        duration_ms = (time.monotonic() - start) * 1000
        op_name = self.execution_context.operation_name or "anonymous"
        logger.info(f"Operation '{op_name}' completed in {duration_ms:.1f}ms")


class ErrorLoggingExtension(SchemaExtension):
    """Log GraphQL errors with context."""

    def on_operation(self):
        yield
        result = self.execution_context.result
        if result and result.errors:
            for error in result.errors:
                logger.error(
                    f"GraphQL error: {error.message}",
                    extra={
                        "path": error.path,
                        "operation": self.execution_context.operation_name,
                    },
                )


class QueryComplexityExtension(SchemaExtension):
    """Reject queries that exceed a complexity threshold."""
    MAX_COMPLEXITY = 1000

    def on_operation(self):
        # Calculate complexity before execution
        complexity = self._estimate_complexity()
        if complexity > self.MAX_COMPLEXITY:
            raise ValueError(
                f"Query complexity {complexity} exceeds limit {self.MAX_COMPLEXITY}"
            )
        yield

    def _estimate_complexity(self) -> int:
        # Simplified: count total selected fields
        query = self.execution_context.query
        return query.count("{")  # Rough heuristic


# Apply extensions to schema
schema = strawberry.Schema(
    query=Query,
    mutation=Mutation,
    extensions=[
        TimingExtension,
        ErrorLoggingExtension,
        QueryComplexityExtension,
    ],
)
```

### Apollo Server Plugins

```typescript
const loggingPlugin = {
  async requestDidStart(requestContext) {
    const start = Date.now();
    console.log(`Query: ${requestContext.request.operationName}`);

    return {
      async willSendResponse() {
        const duration = Date.now() - start;
        console.log(`Response sent in ${duration}ms`);
      },
      async didEncounterErrors(ctx) {
        for (const err of ctx.errors) {
          console.error(`GraphQL Error: ${err.message}`);
        }
      },
    };
  },
};

const server = new ApolloServer({
  typeDefs,
  resolvers,
  plugins: [loggingPlugin],
});
```

---

## 5. File Uploads

GraphQL does not natively support file uploads. Two common solutions exist.

### Approach 1: Separate REST Endpoint (Recommended)

```python
# Use REST for upload, GraphQL for metadata
@app.post("/api/upload")
async def upload_file(file: UploadFile) -> dict:
    """Upload a file and return its URL."""
    path = await storage.save(file)
    return {"url": path, "filename": file.filename, "size": file.size}


# Then reference in GraphQL mutation
@strawberry.mutation
async def create_post_with_image(
    self, info: strawberry.types.Info, input: CreatePostInput
) -> CreatePostPayload:
    """Create a post referencing an already-uploaded image URL."""
    post = await post_repo.create(
        title=input.title,
        content=input.content,
        image_url=input.image_url,  # URL from upload endpoint
    )
    return CreatePostPayload(post=post)
```

### Approach 2: GraphQL Multipart Request (graphql-upload)

```python
# Using strawberry-graphql-django or custom scalar
from strawberry.file_uploads import Upload

@strawberry.type
class Mutation:
    @strawberry.mutation
    async def upload_avatar(
        self, info: strawberry.types.Info, file: Upload
    ) -> str:
        data = await file.read()
        path = f"avatars/{info.context.current_user.id}.jpg"
        await storage.write(path, data)
        return storage.url(path)
```

### Recommendation

| Approach | Pros | Cons |
|----------|------|------|
| REST endpoint | Simple, cacheable, CDN-friendly | Two endpoints |
| Multipart upload | Single endpoint | Complex, non-standard |

> **Best Practice**: Use REST for file upload and reference the uploaded file URL in GraphQL mutations.

---

## 6. Request Batching

GraphQL supports sending multiple operations in a single HTTP request.

### Batched Request

```json
[
  {
    "query": "query { user(id: \"1\") { username } }",
    "operationName": "GetUser1"
  },
  {
    "query": "query { user(id: \"2\") { username } }",
    "operationName": "GetUser2"
  }
]
```

### Batched Response

```json
[
  { "data": { "user": { "username": "alice" } } },
  { "data": { "user": { "username": "bob" } } }
]
```

### Configuration

```python
# Strawberry + FastAPI: batching is supported by default
# Limit batch size for security
from strawberry.fastapi import GraphQLRouter

graphql_router = GraphQLRouter(
    schema,
    allow_queries_via_get=False,  # POST only for mutations
)

# Add middleware to limit batch size
@app.middleware("http")
async def limit_batch_size(request, call_next):
    if request.url.path == "/graphql" and request.method == "POST":
        body = await request.json()
        if isinstance(body, list) and len(body) > 10:
            return JSONResponse(
                {"error": "Batch size exceeds maximum of 10"},
                status_code=400,
            )
    return await call_next(request)
```

---

## 7. CORS and Security Headers

### CORS Configuration

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://app.example.com",
        "http://localhost:3000",  # Development
    ],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
    allow_credentials=True,
    max_age=3600,
)
```

### Security Headers

```python
from starlette.middleware.base import BaseHTTPMiddleware


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Strict-Transport-Security"] = (
            "max-age=31536000; includeSubDomains"
        )
        response.headers["Content-Security-Policy"] = "default-src 'self'"
        return response


app.add_middleware(SecurityHeadersMiddleware)
```

---

## 8. Project Structure

### Recommended Layout

```
graphql_api/
├── main.py                 # FastAPI app entry point
├── schema.py               # Schema assembly (Query, Mutation, Subscription)
├── context.py              # Request context builder
├── database.py             # Database engine and session
├── types/                  # GraphQL type definitions
│   ├── __init__.py
│   ├── user.py
│   ├── post.py
│   ├── comment.py
│   ├── enums.py
│   ├── inputs.py
│   ├── payloads.py
│   └── scalars.py
├── queries/                # Query resolvers
│   ├── __init__.py
│   ├── user_queries.py
│   └── post_queries.py
├── mutations/              # Mutation resolvers
│   ├── __init__.py
│   ├── user_mutations.py
│   └── post_mutations.py
├── subscriptions/          # Subscription resolvers
│   ├── __init__.py
│   └── post_subscriptions.py
├── dataloaders/            # DataLoader definitions
│   ├── __init__.py
│   ├── user_loaders.py
│   └── post_loaders.py
├── services/               # Business logic layer
│   ├── __init__.py
│   ├── user_service.py
│   └── post_service.py
├── repositories/           # Data access layer
│   ├── __init__.py
│   ├── user_repository.py
│   └── post_repository.py
├── models/                 # SQLAlchemy/ORM models
│   ├── __init__.py
│   ├── user.py
│   └── post.py
├── middleware/              # Custom middleware
│   └── __init__.py
├── tests/
│   ├── test_queries.py
│   ├── test_mutations.py
│   └── conftest.py
├── requirements.txt
└── Dockerfile
```

### Layer Responsibilities

| Layer | Responsibility | Depends On |
|-------|---------------|------------|
| Types | GraphQL type definitions | None |
| Resolvers | Map queries to services | Services, DataLoaders |
| DataLoaders | Batch and cache data fetching | Repositories |
| Services | Business logic, validation | Repositories |
| Repositories | Data access, SQL queries | Models, Database |
| Models | ORM entity definitions | Database |

---

## 9. Database Integration

### SQLAlchemy Async Setup

```python
# database.py
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    create_async_engine,
    async_sessionmaker,
)

DATABASE_URL = "postgresql+asyncpg://user:pass@localhost/graphql_db"

engine = create_async_engine(DATABASE_URL, echo=False, pool_size=20)
SessionLocal = async_sessionmaker(engine, class_=AsyncSession)


async def get_session() -> AsyncSession:
    async with SessionLocal() as session:
        yield session
```

### ORM Models

```python
# models/user.py
from sqlalchemy import Column, String, DateTime, func
from sqlalchemy.orm import relationship
from database import Base


class UserModel(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True)
    username = Column(String, unique=True, nullable=False)
    email = Column(String, unique=True, nullable=False)
    bio = Column(String, nullable=True)
    created_at = Column(DateTime, server_default=func.now())

    posts = relationship("PostModel", back_populates="author")
```

### Converting ORM to GraphQL Types

```python
# types/user.py
@strawberry.type
class User:
    id: strawberry.ID
    username: str
    email: str
    bio: str | None
    created_at: datetime

    @classmethod
    def from_model(cls, model: "UserModel") -> "User":
        return cls(
            id=strawberry.ID(model.id),
            username=model.username,
            email=model.email,
            bio=model.bio,
            created_at=model.created_at,
        )
```

---

## 10. Deployment

### Docker

```dockerfile
FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

### Docker Compose with PostgreSQL

```yaml
version: "3.9"
services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql+asyncpg://graphql:secret@db/graphql_db
      - REDIS_URL=redis://redis:6379
    depends_on:
      - db
      - redis

  db:
    image: postgres:16
    environment:
      POSTGRES_DB: graphql_db
      POSTGRES_USER: graphql
      POSTGRES_PASSWORD: secret
    volumes:
      - pgdata:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

volumes:
  pgdata:
```

### Production Checklist

| Item | Action |
|------|--------|
| Introspection | Disable in production |
| Query depth | Limit to 10-15 |
| Query complexity | Set maximum threshold |
| Batch size | Limit to 10-20 operations |
| CORS | Restrict to known origins |
| Rate limiting | Per-client limits |
| Logging | Structured JSON logs |
| Monitoring | Prometheus metrics |
| Error handling | Never leak stack traces |
| HTTPS | Always in production |

---

## 11. Exercises

### Exercise 1: Full Server Setup

Build a complete Strawberry + FastAPI server for a task management app with:
- Users, Projects, Tasks (with status workflow)
- CRUD mutations with input/payload pattern
- DataLoaders for all relationships
- Authentication via JWT in context

### Exercise 2: Apollo Server Comparison

Reimplement Exercise 1 in Apollo Server v4 with TypeScript. Compare:
- Lines of code
- Type safety
- Development speed
- Error messages

### Exercise 3: Middleware Pipeline

Create three Strawberry extensions:
- `QueryCostExtension` — calculates and enforces query cost limits
- `CacheExtension` — caches query results by hash for 60 seconds
- `RateLimitExtension` — limits each user to 100 operations per minute

### Exercise 4: Database Integration

Extend the server from Exercise 1 to use SQLAlchemy async with PostgreSQL:
- Define ORM models
- Implement repository classes
- Write DataLoader batch functions using `IN` queries
- Add database migrations with Alembic

---

## 12. References

### Libraries
- [Strawberry Documentation](https://strawberry.rocks/docs)
- [Apollo Server v4](https://www.apollographql.com/docs/apollo-server/)
- [Ariadne — Schema-First Python](https://ariadnegraphql.org/)
- [TypeGraphQL — Code-First TypeScript](https://typegraphql.com/)

### Deployment
- [Uvicorn — ASGI Server](https://www.uvicorn.org/)
- [Gunicorn + Uvicorn Workers](https://www.uvicorn.org/deployment/)

### Best Practices
- "Principled GraphQL" — Apollo Team
- "GraphQL Server Best Practices" — The Guild
- "Production Ready GraphQL" by Marc-Andre Giroux

---

**License**: CC BY-NC 4.0
