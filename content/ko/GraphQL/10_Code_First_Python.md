# 10. Python 코드 우선 방식 (Strawberry)

**이전**: [GraphQL 클라이언트](./09_GraphQL_Clients.md) | **다음**: [영속 쿼리와 캐싱](./11_Persisted_Queries_Caching.md)

---

지금까지 모든 레슨에서는 스키마 우선 SDL 방식의 JavaScript/TypeScript를 사용했습니다. 하지만 GraphQL은 언어에 독립적이며 Python에는 훌륭한 GraphQL 라이브러리가 있습니다. Strawberry GraphQL은 코드 우선 방식을 논리적인 결론까지 밀어붙입니다: Python 클래스에 타입 어노테이션을 정의하면 Strawberry가 GraphQL 스키마를 자동으로 생성합니다. FastAPI와 결합하면 최소한의 보일러플레이트(boilerplate)로 완전히 타입 지정된 비동기(async) 가능 GraphQL 서버를 얻을 수 있습니다. GraphQL 타입과 Python 타입이 동일한 것이길 바란 적이 있다면, Strawberry가 그 답입니다.

**난이도**: ⭐⭐⭐

## 학습 목표

이 레슨을 완료하면 다음을 수행할 수 있습니다:

1. Python에서 코드 우선과 스키마 우선 접근 방식의 트레이드오프를 설명할 수 있다
2. Strawberry 데코레이터를 사용하여 GraphQL 타입, 쿼리, 뮤테이션, 구독을 정의할 수 있다
3. 프로덕션 준비 비동기 GraphQL API를 위해 Strawberry를 FastAPI와 통합할 수 있다
4. Strawberry에서 N+1 문제를 해결하기 위한 DataLoader를 구현할 수 있다
5. Strawberry 스키마에 대한 단위 및 통합 테스트를 작성할 수 있다

---

## 목차

1. [Python에서 코드 우선 vs 스키마 우선](#1-python에서-코드-우선-vs-스키마-우선)
2. [Strawberry 기초](#2-strawberry-기초)
3. [쿼리와 입력 타입](#3-쿼리와-입력-타입)
4. [뮤테이션](#4-뮤테이션)
5. [FastAPI 통합](#5-fastapi-통합)
6. [비동기 리졸버](#6-비동기-리졸버)
7. [Strawberry에서 DataLoader](#7-strawberry에서-dataloader)
8. [구독](#8-구독)
9. [Strawberry 스키마 테스트](#9-strawberry-스키마-테스트)
10. [연습 문제](#10-연습-문제)
11. [참고 자료](#11-참고-자료)

---

## 1. Python에서 코드 우선 vs 스키마 우선

Python 생태계에서는 두 가지 접근 방식이 지배적입니다:

| 라이브러리 | 접근 방식 | 스타일 |
|---------|--------|------|
| **Strawberry** | 코드 우선 | Python 데이터클래스 + 타입 힌트 |
| **Ariadne** | 스키마 우선 | SDL + 리졸버 바인딩 |
| **Graphene** | 코드 우선 | 클래스 기반 (구형 API) |

### 코드 우선을 선택하는 이유?

스키마 우선에서는 SDL을 작성한 다음 이와 일치해야 하는 Python 리졸버를 작성합니다. 두 가지가 서로 달라질 수 있습니다 — 스키마에는 필드가 `String!`을 반환한다고 하지만 리졸버는 `None`을 반환하고, 이 버그를 런타임에서야 발견하게 됩니다.

코드 우선은 이러한 카테고리의 버그를 제거합니다. Python 타입이 *곧* GraphQL 타입입니다:

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

다음을 생성합니다:

```graphql
type User {
  id: ID!
  name: String!
  email: String!
  isActive: Boolean!
}
```

Strawberry는 Python의 `snake_case` 이름을 GraphQL의 `camelCase` 이름으로 자동 변환합니다. 이는 각 언어의 관례를 따릅니다.

### 스키마 우선 (Ariadne)을 선호할 때

- 프론트엔드 팀이 API 계약을 정의하고 SDL을 건네줄 때
- 다른 언어에서 Python으로 기존 SDL 우선 API를 마이그레이션할 때
- 스키마를 문서로 읽는 것을 선호할 때

---

## 2. Strawberry 기초

### 설치

```bash
pip install strawberry-graphql[debug-server]
```

### 타입 정의

모든 Strawberry 타입은 내부적으로 Python 데이터클래스(dataclass)입니다. `@strawberry.type` 데코레이터는 이를 데이터클래스와 GraphQL 타입 모두로 변환합니다.

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

### 스칼라 타입 매핑

| Python 타입 | GraphQL 타입 |
|-----------|------------|
| `str` | `String!` |
| `int` | `Int!` |
| `float` | `Float!` |
| `bool` | `Boolean!` |
| `strawberry.ID` | `ID!` |
| `Optional[str]` | `String` (null 허용) |
| `list[str]` | `[String!]!` |
| `Optional[list[str]]` | `[String!]` |
| `datetime` | `DateTime!` (커스텀 스칼라) |
| `Decimal` | `Decimal!` (커스텀 스칼라) |

### 옵셔널과 리스트 필드

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

## 3. 쿼리와 입력 타입

### 쿼리 정의

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

### 입력 타입

입력 타입은 뮤테이션 인수에 사용됩니다. GraphQL 관례인 입력과 출력을 구분하기 위해 출력 타입과 별도로 유지합니다.

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

`@strawberry.input` 데코레이터는 GraphQL `input` 타입을 생성합니다:

```graphql
input CreatePostInput {
  title: String!
  content: String!
  published: Boolean! = false
  tags: [String!]! = []
}
```

---

## 4. 뮤테이션

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

## 5. FastAPI 통합

FastAPI와 Strawberry는 자연스러운 조합입니다 — 둘 다 비동기 우선이고, 둘 다 Python 타입 힌트를 사용하며, Strawberry는 일급(first-class) FastAPI 통합을 제공합니다.

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

### 서버 실행

```bash
uvicorn main:app --reload --port 8000
```

`http://localhost:8000/graphql`을 방문하면 GraphQL Playground에 접근할 수 있습니다 (Strawberry가 기본으로 포함).

### 리졸버에서 컨텍스트 접근

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

## 6. 비동기 리졸버

Strawberry는 비동기 리졸버를 완전히 지원하며, 이는 비차단(non-blocking) 데이터베이스 쿼리와 외부 API 호출에 필수적입니다.

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

### 병렬 해석

쿼리가 여러 독립적인 필드를 요청하면, GraphQL은 리졸버가 비동기인 경우 이들을 동시에 해석합니다:

```graphql
query Dashboard {
  recentPosts(limit: 5) { title }     # 이 세 개는
  topAuthors(limit: 3) { name }       # 독립적이므로
  stats { totalPosts totalUsers }      # 동시에 해석됩니다.
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

## 7. Strawberry에서 DataLoader

Strawberry는 컨텍스트 시스템과 통합되는 내장 DataLoader를 제공하여, JavaScript DataLoader 라이브러리와 마찬가지로 N+1 문제를 해결합니다.

### 문제

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

### 해결책

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

### 컨텍스트에 DataLoader 등록

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

### 리졸버에서 DataLoader 사용

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

게시물이 10개일 때, 10개의 개별 `SELECT * FROM users WHERE id = ?` 쿼리 대신 DataLoader가 이들을 하나의 `SELECT * FROM users WHERE id IN (?, ?, ?, ...)`로 배치합니다.

---

## 8. 구독

Strawberry는 비동기 제너레이터(async generators)를 사용하여 구독을 지원합니다.

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

### FastAPI WebSocket 설정

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

### Redis 기반 PubSub

여러 서버 인스턴스가 있는 프로덕션에서는 Redis를 사용하세요:

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

## 9. Strawberry 스키마 테스트

Strawberry 스키마는 서버를 실행하지 않고도 스키마 객체에 대해 직접 쿼리를 실행하여 테스트할 수 있습니다.

### 리졸버 단위 테스트

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

### 뮤테이션 테스트

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

### FastAPI TestClient로 통합 테스트

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

### 구독 테스트

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

## 10. 연습 문제

### 연습 1: 블로그 API 타입 (입문)

다음 요구사항으로 블로그 애플리케이션을 위한 Strawberry 타입을 정의하세요:

- `User`: id, username, email, bio (선택), avatar_url (선택), role (열거형: READER, AUTHOR, ADMIN), created_at
- `Post`: id, title, slug (제목에서 계산됨), content, published, view_count, author_id, tags (문자열 목록), created_at, updated_at
- `Comment`: id, content, author_id, post_id, parent_id (선택, 중첩 댓글용), created_at
- 입력 타입: `CreatePostInput`, `UpdatePostInput`, `CreateCommentInput`

계산 필드 포함: `Post.slug` (소문자, 하이픈 처리된 제목), `Post.reading_time_minutes`, `Post.is_new` (최근 7일 내 게시).

### 연습 2: 인증이 있는 FastAPI 통합 (중급)

다음을 갖춘 완전한 Strawberry + FastAPI 설정을 구축하세요:

1. JWT 기반 인증 (컨텍스트의 `Authorization` 헤더에서 토큰 추출)
2. 현재 사용자를 반환하는 `me` 쿼리 (인증되지 않은 경우 오류)
3. 인증이 필요한 `createPost` 뮤테이션
4. 현재 사용자가 작성자 또는 관리자이어야 하는 `deletePost` 뮤테이션
5. `http://localhost:3000`에 구성된 CORS

완전한 `main.py`, `auth.py`, `schema.py` 파일을 제공하세요.

### 연습 3: DataLoader 구현 (중급)

다음 타입과 PostgreSQL 데이터베이스가 주어졌을 때:

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

다음에 대한 DataLoader를 구현하세요:

1. 책의 `author_id`로 저자 로딩 (다대일)
2. 저자의 `id`로 책 로딩 (일대다)
3. 책의 카테고리 로딩 (`book_categories` 조인 테이블을 통한 다대다)
4. 카테고리의 책 로딩 (다대다)

배치 함수, 컨텍스트 설정, 리졸버 통합을 작성하세요.

### 연습 4: PubSub을 사용한 구독 (고급)

Strawberry 구독을 사용하여 실시간 알림 시스템을 구축하세요:

1. id, type (열거형), message, recipient_id, read, created_at가 있는 `Notification` 타입 정의
2. 인증된 사용자의 ID로 필터링하는 `notificationReceived` 구독 구현
3. 구독에 게시하는 `sendNotification` 뮤테이션 구현
4. 개발용 인메모리 PubSub 사용 및 프로덕션에서 Redis로 교체하는 방법 설명
5. 뮤테이션이 구독을 트리거하는지 확인하는 테스트 작성

### 연습 5: 완전한 테스트 스위트 (고급)

Strawberry 기반 작업 관리 API에 대한 포괄적인 테스트 스위트를 작성하세요:

- 타입: `Project`, `Task`, `User`, `Label`
- 작업: 모든 타입에 대한 CRUD, 더하기 `assignTask`, `moveTask` (상태 변경), `addLabel`

테스트 스위트에 포함해야 할 것:

1. 계산 필드에 대한 단위 테스트 (예: `Project.completion_percentage`)
2. 목(mock) 컨텍스트를 사용한 쿼리 테스트 (정상 경로 + 오류 경우)
3. 인가 검증 뮤테이션 테스트 (소유자 vs 비소유자 vs 관리자)
4. FastAPI의 `AsyncClient`를 사용한 통합 테스트
5. 최소 하나의 구독 테스트

`pytest`, `pytest-asyncio`, 목 객체를 사용하세요. 최소 15개의 테스트 케이스를 목표로 하세요.

---

## 11. 참고 자료

- Strawberry GraphQL 문서 — https://strawberry.rocks/docs
- Strawberry + FastAPI — https://strawberry.rocks/docs/integrations/fastapi
- Strawberry DataLoader — https://strawberry.rocks/docs/guides/dataloaders
- FastAPI 문서 — https://fastapi.tiangolo.com
- Ariadne (스키마 우선 대안) — https://ariadnegraphql.org
- graphql-core (Python 참조 구현) — https://github.com/graphql-python/graphql-core

---

**이전**: [GraphQL 클라이언트](./09_GraphQL_Clients.md) | **다음**: [영속 쿼리와 캐싱](./11_Persisted_Queries_Caching.md)
