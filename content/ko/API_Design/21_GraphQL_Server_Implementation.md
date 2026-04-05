# 21. GraphQL 서버 구현(GraphQL Server Implementation)

**이전**: [GraphQL 서브스크립션](./20_GraphQL_Subscriptions.md) | **다음**: [GraphQL 페더레이션](./22_GraphQL_Federation.md)

**난이도**: ⭐⭐⭐

---

## 학습 목표

- 코드 우선(code-first)과 스키마 우선(schema-first) 개발 접근법을 비교하고 적절히 선택할 수 있다
- Strawberry(Python)와 FastAPI로 프로덕션 수준의 GraphQL 서버를 구축할 수 있다
- 다국어 비교를 위해 Node.js/TypeScript에서 Apollo Server를 구현할 수 있다
- 로깅, 오류 처리, 성능 추적을 위한 미들웨어 패턴을 적용할 수 있다
- GraphQL 서버에서 파일 업로드, CORS, 요청 배칭을 구성할 수 있다
- 대규모에서 유지보수 가능한 GraphQL 서버 프로젝트를 구조화할 수 있다

---

## 목차

1. [코드 우선 vs. 스키마 우선](#1-코드-우선-vs-스키마-우선)
2. [Strawberry (Python) 심화](#2-strawberry-python-심화)
3. [Apollo Server (Node.js)](#3-apollo-server-nodejs)
4. [미들웨어와 확장](#4-미들웨어와-확장)
5. [파일 업로드](#5-파일-업로드)
6. [요청 배칭](#6-요청-배칭)
7. [CORS 및 보안 헤더](#7-cors-및-보안-헤더)
8. [프로젝트 구조](#8-프로젝트-구조)
9. [데이터베이스 통합](#9-데이터베이스-통합)
10. [배포](#10-배포)
11. [연습 문제](#11-연습-문제)
12. [참고 자료](#12-참고-자료)

---

## 1. 코드 우선 vs. 스키마 우선

### 스키마 우선 접근법

SDL 파일을 먼저 작성한 후 이에 맞는 리졸버를 구현합니다:

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

### 코드 우선 접근법

프로그래밍 언어에서 타입을 정의하고 스키마가 생성됩니다:

```python
@strawberry.type
class User:
    id: strawberry.ID
    username: str
    email: str

    @strawberry.field
    def posts(self) -> list["Post"]:
        return post_repo.find_by_author(self.id)
```

### 비교

| 관점 | 스키마 우선 | 코드 우선 |
|------|-----------|----------|
| 진실의 원천 | `.graphql` 파일 | Python/TS 코드 |
| 타입 안전성 | 외부 검증 | 컴파일 타임 / 타입 체커 |
| 리팩토링 | 수동 동기화 | IDE 지원 |
| 온보딩 | 쉬움 (읽기 쉬운 SDL) | 언어 지식 필요 |
| 도구 | GraphQL 전용 | 언어 생태계 |
| 라이브러리 | Ariadne, Apollo Server | Strawberry, TypeGraphQL |

### 권장 사항

| 팀 / 프로젝트 | 접근법 |
|---------------|--------|
| 타입 힌트 있는 Python 백엔드 | 코드 우선 (Strawberry) |
| TypeScript 풀스택 | 코드 우선 (TypeGraphQL) 또는 스키마 우선 (Apollo) |
| 다국어 팀 | 스키마 우선 (공유 SDL) |
| API 우선 조직 | 스키마 우선 |
| 빠른 프로토타이핑 | 코드 우선 |

---

## 2. Strawberry (Python) 심화

### Strawberry를 선택하는 이유

- Python 데이터클래스와 타입 힌트 기반
- 일급 async/await 지원
- FastAPI 및 Django 통합
- 활발한 개발과 커뮤니티
- 완전한 서브스크립션 지원

### 완전한 서버 설정

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
    await create_tables(engine)
    yield
    await engine.dispose()


app = FastAPI(title="GraphQL API", version="1.0.0", lifespan=lifespan)

graphql_router = GraphQLRouter(
    schema,
    context_getter=get_context,
    graphql_ide="graphiql",
)

app.include_router(graphql_router, prefix="/graphql")
```

### 스키마 조립

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
    """루트 쿼리 타입 — 모든 쿼리 리졸버를 결합합니다."""
    pass


@strawberry.type
class Mutation(UserMutation, PostMutation):
    """루트 뮤테이션 타입 — 모든 뮤테이션 리졸버를 결합합니다."""
    pass


@strawberry.type
class Subscription(PostSubscription):
    """루트 서브스크립션 타입."""
    pass


schema = strawberry.Schema(
    query=Query,
    mutation=Mutation,
    subscription=Subscription,
)
```

### 타입 정의와 리졸버(Type Definitions with Resolvers)

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

### 열거형과 입력 타입(Enum and Input Types)

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

비교를 위해 Apollo Server v4에서의 동등한 설정:

### 설치

```bash
npm install @apollo/server graphql
npm install -D typescript @types/node ts-node
```

### 스키마 우선 Apollo

```typescript
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
  }

  enum PostStatus { DRAFT PUBLISHED ARCHIVED }

  type Query {
    user(id: ID!): User
    users(limit: Int = 10): [User!]!
  }

  type Mutation {
    createPost(input: CreatePostInput!): CreatePostPayload!
  }

  input CreatePostInput {
    title: String!
    content: String!
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

### 리졸버

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

### 서버 진입점

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

## 4. 미들웨어와 확장

### Strawberry 확장(Strawberry Extensions)

```python
from strawberry.extensions import SchemaExtension
from strawberry.extensions.tracing import ApolloTracingExtension
import time
import logging

logger = logging.getLogger("graphql")


class TimingExtension(SchemaExtension):
    """각 작업의 실행 시간을 로깅합니다."""
    def on_operation(self):
        start = time.monotonic()
        yield
        duration_ms = (time.monotonic() - start) * 1000
        op_name = self.execution_context.operation_name or "anonymous"
        logger.info(f"작업 '{op_name}' 완료: {duration_ms:.1f}ms")


class ErrorLoggingExtension(SchemaExtension):
    """컨텍스트와 함께 GraphQL 오류를 로깅합니다."""
    def on_operation(self):
        yield
        result = self.execution_context.result
        if result and result.errors:
            for error in result.errors:
                logger.error(f"GraphQL 오류: {error.message}")


class QueryComplexityExtension(SchemaExtension):
    """복잡도 임계값을 초과하는 쿼리를 거부합니다."""
    MAX_COMPLEXITY = 1000

    def on_operation(self):
        complexity = self._estimate_complexity()
        if complexity > self.MAX_COMPLEXITY:
            raise ValueError(
                f"쿼리 복잡도 {complexity}가 제한 {self.MAX_COMPLEXITY}을 초과합니다"
            )
        yield


schema = strawberry.Schema(
    query=Query,
    mutation=Mutation,
    extensions=[TimingExtension, ErrorLoggingExtension, QueryComplexityExtension],
)
```

### Apollo Server 플러그인

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

## 5. 파일 업로드

GraphQL은 기본적으로 파일 업로드를 지원하지 않습니다. 두 가지 일반적인 해결책이 있습니다.

### 접근법 1: 별도 REST 엔드포인트 (권장)

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

### 접근법 2: GraphQL 멀티파트 요청 (graphql-upload)

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

### 권장 사항

| 접근법 | 장점 | 단점 |
|--------|------|------|
| REST 엔드포인트 | 단순, 캐시 가능, CDN 친화적 | 두 개의 엔드포인트 |
| 멀티파트 업로드 | 단일 엔드포인트 | 복잡, 비표준 |

> **모범 사례**: 파일 업로드에는 REST를 사용하고 GraphQL 뮤테이션에서 업로드된 파일 URL을 참조하세요.

---

## 6. 요청 배칭

GraphQL은 단일 HTTP 요청으로 여러 작업을 보내는 것을 지원합니다.

### 배칭된 요청(Batched Request)

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

### 배칭된 응답(Batched Response)

```json
[
  { "data": { "user": { "username": "alice" } } },
  { "data": { "user": { "username": "bob" } } }
]
```

### 설정

```python
# Strawberry + FastAPI: batching is supported by default
# Limit batch size for security
from strawberry.fastapi import GraphQLRouter

graphql_router = GraphQLRouter(
    schema,
    allow_queries_via_get=False,  # POST only for mutations
)
```

### 배치 크기 제한

보안을 위해 배치 크기를 제한합니다:

```python
@app.middleware("http")
async def limit_batch_size(request, call_next):
    if request.url.path == "/graphql" and request.method == "POST":
        body = await request.json()
        if isinstance(body, list) and len(body) > 10:
            return JSONResponse(
                {"error": "배치 크기가 최대 10을 초과합니다"},
                status_code=400,
            )
    return await call_next(request)
```

---

## 7. CORS 및 보안 헤더

### CORS 설정

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

### 보안 헤더

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

## 8. 프로젝트 구조

### 권장 레이아웃

```
graphql_api/
├── main.py                 # FastAPI 앱 진입점
├── schema.py               # 스키마 조립 (Query, Mutation, Subscription)
├── context.py              # 요청 컨텍스트 빌더
├── database.py             # 데이터베이스 엔진 및 세션
├── types/                  # GraphQL 타입 정의
│   ├── user.py
│   ├── post.py
│   ├── enums.py
│   ├── inputs.py
│   └── payloads.py
├── queries/                # 쿼리 리졸버
├── mutations/              # 뮤테이션 리졸버
├── subscriptions/          # 서브스크립션 리졸버
├── dataloaders/            # DataLoader 정의
├── services/               # 비즈니스 로직 레이어
├── repositories/           # 데이터 접근 레이어
├── models/                 # ORM 모델
├── tests/
├── requirements.txt
└── Dockerfile
```

### 레이어 책임

| 레이어 | 책임 | 의존 대상 |
|--------|------|----------|
| Types | GraphQL 타입 정의 | 없음 |
| Resolvers | 쿼리를 서비스에 매핑 | Services, DataLoaders |
| DataLoaders | 배치 및 캐시 데이터 페칭 | Repositories |
| Services | 비즈니스 로직, 유효성 검사 | Repositories |
| Repositories | 데이터 접근, SQL 쿼리 | Models, Database |

---

## 9. 데이터베이스 통합

### SQLAlchemy 비동기 설정

```python
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

DATABASE_URL = "postgresql+asyncpg://user:pass@localhost/graphql_db"
engine = create_async_engine(DATABASE_URL, echo=False, pool_size=20)
SessionLocal = async_sessionmaker(engine)
```

### ORM 모델

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

### ORM을 GraphQL 타입으로 변환

```python
@strawberry.type
class User:
    id: strawberry.ID
    username: str
    email: str

    @classmethod
    def from_model(cls, model: "UserModel") -> "User":
        return cls(
            id=strawberry.ID(model.id),
            username=model.username,
            email=model.email,
        )
```

---

## 10. 배포

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

### PostgreSQL을 포함한 Docker Compose

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

### 프로덕션 체크리스트

| 항목 | 조치 |
|------|------|
| 인트로스펙션 | 프로덕션에서 비활성화 |
| 쿼리 깊이 | 10-15로 제한 |
| 쿼리 복잡도 | 최대 임계값 설정 |
| 배치 크기 | 10-20개 작업으로 제한 |
| CORS | 알려진 출처로 제한 |
| 속도 제한 | 클라이언트별 제한 |
| 로깅 | 구조화된 JSON 로그 |
| 모니터링 | Prometheus 메트릭 |
| 오류 처리 | 스택 트레이스 절대 노출 금지 |
| HTTPS | 프로덕션에서 항상 사용 |

---

## 11. 연습 문제

### 연습 1: 완전한 서버 설정

작업 관리 앱을 위한 완전한 Strawberry + FastAPI 서버를 구축하세요:
- Users, Projects, Tasks (상태 워크플로우 포함)
- 입력/페이로드 패턴의 CRUD 뮤테이션
- 모든 관계에 대한 DataLoaders
- 컨텍스트에서 JWT 인증

### 연습 2: Apollo Server 비교

연습 1을 Apollo Server v4와 TypeScript로 재구현하세요. 비교 항목:
- 코드 줄 수
- 타입 안전성
- 개발 속도
- 오류 메시지

### 연습 3: 미들웨어 파이프라인

세 가지 Strawberry 확장을 만드세요:
- `QueryCostExtension` — 쿼리 비용 제한 계산 및 적용
- `CacheExtension` — 해시별로 쿼리 결과를 60초간 캐시
- `RateLimitExtension` — 사용자당 분당 100개 작업 제한

### 연습 4: 데이터베이스 통합

연습 1의 서버를 SQLAlchemy 비동기와 PostgreSQL로 확장하세요.

---

## 12. 참고 자료

### 라이브러리
- [Strawberry Documentation](https://strawberry.rocks/docs)
- [Apollo Server v4](https://www.apollographql.com/docs/apollo-server/)
- [Ariadne — Schema-First Python](https://ariadnegraphql.org/)
- [TypeGraphQL — Code-First TypeScript](https://typegraphql.com/)

### 배포
- [Uvicorn — ASGI Server](https://www.uvicorn.org/)
- [Gunicorn + Uvicorn Workers](https://www.uvicorn.org/deployment/)

---

**License**: CC BY-NC 4.0
