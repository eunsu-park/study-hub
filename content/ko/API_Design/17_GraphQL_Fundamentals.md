# 17. GraphQL 기초(GraphQL Fundamentals)

**이전**: [API 생명주기 관리](./16_API_Lifecycle_Management.md) | **다음**: [GraphQL 스키마 설계](./18_GraphQL_Schema_Design.md)

**난이도**: ⭐⭐⭐

---

## 학습 목표

- GraphQL의 동기와 REST 대비 해결하는 문제점을 설명할 수 있다
- GraphQL 쿼리 언어를 사용하여 쿼리, 뮤테이션, 프래그먼트를 작성할 수 있다
- Schema Definition Language(SDL)에서 타입, 필드, 관계를 정의할 수 있다
- GraphQL의 타입 시스템 기본 요소(스칼라, 객체, 열거형, 리스트)를 구분할 수 있다
- 다양한 API 시나리오에서 REST와 GraphQL의 트레이드오프를 비교할 수 있다
- Python에서 Strawberry를 사용하여 최소한의 GraphQL 서버를 구축할 수 있다

---

## 목차

1. [GraphQL이란?](#1-graphql이란)
2. [GraphQL vs. REST](#2-graphql-vs-rest)
3. [Schema Definition Language (SDL)](#3-schema-definition-language-sdl)
4. [타입 시스템](#4-타입-시스템)
5. [쿼리](#5-쿼리)
6. [뮤테이션](#6-뮤테이션)
7. [변수와 프래그먼트](#7-변수와-프래그먼트)
8. [인트로스펙션](#8-인트로스펙션)
9. [최소 Python 서버 구축](#9-최소-python-서버-구축)
10. [연습 문제](#10-연습-문제)
11. [참고 자료](#11-참고-자료)

---

## 1. GraphQL이란?

GraphQL은 API를 위한 쿼리 언어이자 기존 데이터로 해당 쿼리를 실행하기 위한 런타임입니다. 2012년 Facebook에서 개발되어 2015년에 오픈소스로 공개되었으며, API의 데이터에 대한 완전하고 이해하기 쉬운 설명을 제공하고, 클라이언트가 정확히 필요한 데이터만 요청할 수 있도록 하며, 시간이 지남에 따라 API를 더 쉽게 발전시킬 수 있도록 합니다.

### 핵심 원칙

| 원칙 | 설명 |
|------|------|
| **선언적 데이터 페칭** | 클라이언트가 필요한 데이터를 정확히 명시 |
| **단일 엔드포인트** | 모든 작업이 하나의 URL(일반적으로 `/graphql`)을 통해 이루어짐 |
| **강타입** | 모든 필드, 인자, 반환값에 명시적 타입이 있음 |
| **계층적** | 쿼리가 응답의 형태를 반영 |
| **자기 기술적** | 스키마 자체를 쿼리하여 문서화 및 도구화 가능 |

### GraphQL이 해결하는 문제

전통적인 REST API는 두 가지 근본적인 비효율성을 겪습니다:

1. **과다 페칭(Over-fetching)**: `GET /users/42`가 클라이언트가 `name`과 `email`만 필요한 경우에도 모든 필드를 반환합니다.
2. **과소 페칭(Under-fetching)**: 사용자 프로필을 게시글 및 팔로워와 함께 표시하려면 여러 번의 왕복이 필요합니다(`/users/42`, `/users/42/posts`, `/users/42/followers`).

GraphQL은 클라이언트가 단일 요청으로 응답의 정확한 형태를 기술할 수 있게 하여 두 문제를 모두 해결합니다.

```graphql
# 한 번의 요청으로 정확한 데이터
query {
  user(id: "42") {
    name
    email
    posts(first: 5) {
      title
      createdAt
    }
    followers {
      totalCount
    }
  }
}
```

### 간략한 역사

| 연도 | 이정표 |
|------|--------|
| 2012 | Facebook에서 모바일 앱용으로 내부 개발 |
| 2015 | JavaScript 참조 구현과 함께 오픈소스 공개 |
| 2016 | GitHub가 GraphQL API(v4) 출시 |
| 2018 | Linux Foundation 산하에 GraphQL Foundation 설립 |
| 2021 | 2021년 10월 사양 릴리스 (최신 안정 버전) |
| 2023 | Shopify, Stripe, Yelp, Airbnb 등 광범위한 채택 |

---

## 2. GraphQL vs. REST

GraphQL을 사용할 시점과 REST를 사용할 시점을 아는 것은 API 설계자에게 필수적입니다.

### 기능 비교

| 관점 | REST | GraphQL |
|------|------|---------|
| 엔드포인트 | 다중 (리소스별 하나) | 단일 (`/graphql`) |
| 데이터 형태 | 서버 결정 | 클라이언트 결정 |
| 과다 페칭 | 일반적 | 설계적으로 제거 |
| 과소 페칭 | 일반적 (여러 호출 필요) | 제거 (중첩 쿼리) |
| 버전 관리 | URL 또는 헤더 버전 관리 | 스키마 진화 (추가적 변경) |
| 캐싱 | HTTP 캐싱 (GET, ETag) | 커스텀 캐싱 전략 필요 |
| 파일 업로드 | 네이티브 멀티파트 지원 | 확장 필요 |
| 오류 처리 | HTTP 상태 코드 | 항상 200에 `errors` 배열 반환 |
| 도구 | 성숙 (Swagger, Postman) | 성장 중 (GraphiQL, Apollo Studio) |
| 학습 곡선 | 낮음 | 보통 |

| 차원 | REST | GraphQL |
|------|------|---------|
| 데이터 페칭 | 다중 엔드포인트, 고정 구조 | 단일 엔드포인트, 유연한 쿼리 |
| 과다/과소 페칭 | 일반적 (고정 응답) | 제거 (클라이언트가 필드 지정) |
| 타입 시스템 | 선택적 (OpenAPI) | 내장 (SDL) |
| 실시간 | 폴링 또는 WebSocket (별도) | Subscriptions (네이티브) |
| 캐싱 | HTTP 캐싱 (단순) | 복잡 (정규화된 클라이언트 캐시) |
| 학습 곡선 | 낮음 | 높음 |
| 도구 성숙도 | 매우 성숙 | 빠르게 성장 중 |
| 최적 용도 | CRUD, 공개 API, 단순 클라이언트 | 복잡한 UI, 모바일, 집계 |

### GraphQL 선택 시점

- 제한된 대역폭에서 정확한 데이터 페칭이 필요한 **모바일 애플리케이션**
- 다른 UI 컴포넌트가 다른 데이터 조각을 필요로 하는 **마이크로 프론트엔드**
- 백엔드와 독립적으로 변경되는 **빠르게 진화하는 UI**
- 여러 마이크로서비스의 데이터를 결합하는 **집계 레이어**

### REST 유지 시점

- 예측 가능한 접근 패턴을 가진 **단순 CRUD** API
- **파일 중심** API (업로드/다운로드)
- HTTP 캐싱이 중요한 **공개 API**
- **Webhook/이벤트 기반** 통합
- **팀 기술 세트**가 GraphQL 경험 없이 REST 중심인 경우

### 하이브리드 접근법

많은 프로덕션 시스템이 둘 다 사용합니다:

```
Client → API Gateway → REST 엔드포인트 (CRUD, 파일 작업)
                     → GraphQL 엔드포인트 (복잡한 쿼리, 집계)
                     → gRPC (내부 마이크로서비스 호출)
```

---

## 3. Schema Definition Language (SDL)

Schema Definition Language는 GraphQL에서 API의 형태를 정의하는 방법입니다. 클라이언트와 서버 간의 "계약"이라고 생각할 수 있습니다.

### 객체 타입

객체 타입은 가장 기본적인 구성 요소입니다:

```graphql
type User {
  id: ID!
  username: String!
  email: String!
  bio: String
  createdAt: DateTime!
}
```

- `!`는 비널(non-nullable)을 의미합니다 (필드가 항상 값을 반환해야 함)
- `!` 없이는 필드가 기본적으로 널러블입니다

### 스칼라 타입

GraphQL은 5개의 내장 스칼라를 제공합니다:

| 타입 | 설명 | 예시 |
|------|------|------|
| `Int` | 부호 있는 32비트 정수 | `42` |
| `Float` | 배정밀도 부동소수점 | `3.14` |
| `String` | UTF-8 문자열 | `"hello"` |
| `Boolean` | `true` 또는 `false` | `true` |
| `ID` | 고유 식별자 (문자열로 직렬화) | `"abc-123"` |

### 리스트와 Non-Null 수정자

```graphql
type Post {
  tags: [String!]!    # 비널 리스트의 비널 문자열
  comments: [Comment] # 널러블 리스트의 널러블 댓글
}
```

| 선언 | 리스트가 null? | 요소가 null? |
|------|--------------|-------------|
| `[String]` | 예 | 예 |
| `[String!]` | 예 | 아니오 |
| `[String]!` | 아니오 | 예 |
| `[String!]!` | 아니오 | 아니오 |

### 관계

타입은 서로 참조하여 그래프를 형성합니다:

```graphql
type User {
  id: ID!
  username: String!
  posts: [Post!]!
}

type Post {
  id: ID!
  title: String!
  author: User!
  comments: [Comment!]!
}

type Comment {
  id: ID!
  body: String!
  author: User!
  post: Post!
}
```

### 진입점: Query와 Mutation 타입

```graphql
type Query {
  user(id: ID!): User
  users(limit: Int = 10, offset: Int = 0): [User!]!
  post(id: ID!): Post
  searchPosts(query: String!): [Post!]!
}

type Mutation {
  createUser(username: String!, email: String!): User!
  createPost(title: String!, content: String!, authorId: ID!): Post!
  deletePost(id: ID!): Boolean!
}
```

`Query` 타입은 읽기 작업을 정의하고, `Mutation` 타입은 쓰기 작업을 정의합니다.

---

## 4. 타입 시스템

GraphQL의 타입 시스템은 계약 보장의 근간입니다.

### 열거형(Enum)

열거형은 필드를 유한한 값 집합으로 제한합니다:

```graphql
enum PostStatus {
  DRAFT
  PUBLISHED
  ARCHIVED
}

type Post {
  id: ID!
  title: String!
  status: PostStatus!
}
```

### 입력 타입(Input Type)

입력 타입은 구조화된 뮤테이션 인자에 사용됩니다:

```graphql
input CreatePostInput {
  title: String!
  content: String!
  status: PostStatus = DRAFT
  tags: [String!]
}

type Mutation {
  createPost(input: CreatePostInput!): Post!
}
```

> **관례**: 입력 타입은 `Input` 접미사로 명명합니다.

### 인터페이스

인터페이스는 타입이 구현해야 하는 계약을 정의합니다:

```graphql
interface Node {
  id: ID!
}

interface Timestamped {
  createdAt: DateTime!
  updatedAt: DateTime!
}

type User implements Node & Timestamped {
  id: ID!
  username: String!
  createdAt: DateTime!
  updatedAt: DateTime!
}

type Post implements Node & Timestamped {
  id: ID!
  title: String!
  createdAt: DateTime!
  updatedAt: DateTime!
}
```

### 유니온 타입

유니온은 여러 타입 중 하나가 될 수 있는 값을 나타냅니다:

```graphql
union SearchResult = User | Post | Comment

type Query {
  search(query: String!): [SearchResult!]!
}
```

클라이언트는 인라인 프래그먼트를 사용하여 각 경우를 처리합니다:

```graphql
query {
  search(query: "graphql") {
    ... on User {
      username
    }
    ... on Post {
      title
      author { username }
    }
    ... on Comment {
      body
    }
  }
}
```

### 커스텀 스칼라

도메인 특화 스칼라를 정의할 수 있습니다:

```graphql
scalar DateTime
scalar Email
scalar URL
scalar JSON
```

서버는 각 커스텀 스칼라에 대한 파싱 및 직렬화 로직을 제공합니다.

---

## 5. 쿼리

쿼리는 읽기 작업입니다. 가장 일반적인 GraphQL 연산입니다.

### 기본 쿼리

```graphql
query GetUser {
  user(id: "42") {
    username
    email
  }
}
```

응답:

```json
{
  "data": {
    "user": {
      "username": "alice",
      "email": "alice@example.com"
    }
  }
}
```

### 중첩 쿼리

관련 데이터를 가져올 때 GraphQL이 빛을 발합니다:

```graphql
query UserWithPosts {
  user(id: "42") {
    username
    posts(first: 3) {
      title
      commentCount
      comments(first: 2) {
        body
        author {
          username
        }
      }
    }
  }
}
```

### 별칭(Alias)

같은 필드를 다른 인자로 쿼리할 때 별칭을 사용합니다:

```graphql
query {
  admin: user(id: "1") {
    username
    email
  }
  viewer: user(id: "42") {
    username
    email
  }
}
```

### 인자(Arguments)

필드는 모든 레벨에서 인자를 받을 수 있습니다:

```graphql
query {
  users(limit: 10, sortBy: CREATED_AT, order: DESC) {
    username
    posts(status: PUBLISHED, first: 5) {
      title
    }
  }
}
```

### `__typename` 메타 필드

모든 타입에는 유니온/인터페이스 구분에 유용한 내장 `__typename` 필드가 있습니다:

```graphql
query {
  search(query: "graphql") {
    __typename
    ... on User { username }
    ... on Post { title }
  }
}
```

---

## 6. 뮤테이션

뮤테이션은 쓰기 작업입니다. 관례적으로 동사형으로 명명합니다.

### 기본 뮤테이션

```graphql
mutation CreateUser {
  createUser(input: {
    username: "bob"
    email: "bob@example.com"
  }) {
    id
    username
    createdAt
  }
}
```

### 뮤테이션 응답 패턴

결과와 잠재적 오류를 모두 포함하는 페이로드 타입을 반환하는 것이 모범 사례입니다:

```graphql
type CreatePostPayload {
  post: Post
  userErrors: [UserError!]!
}

type UserError {
  field: [String!]
  message: String!
  code: ErrorCode!
}

enum ErrorCode {
  INVALID_INPUT
  NOT_FOUND
  UNAUTHORIZED
  CONFLICT
}

type Mutation {
  createPost(input: CreatePostInput!): CreatePostPayload!
}
```

이 접근법(Shopify에서 대중화)은 최상위 `errors` 배열에 의존하지 않고 오류를 구조화하고 타입 안전하게 유지합니다.

### 다중 뮤테이션

단일 요청의 여러 뮤테이션은 순차적으로 실행됩니다 (병렬로 실행될 수 있는 쿼리와 달리):

```graphql
mutation {
  createPost(input: { title: "First", content: "..." }) {
    post { id }
  }
  updateUser(input: { bio: "Updated bio" }) {
    user { bio }
  }
}
```

### 입력 유효성 검사

입력 타입을 자기 문서화되도록 구조화합니다:

```graphql
input UpdatePostInput {
  title: String                       # 선택적 (nullable = 생략 가능)
  content: String
  status: PostStatus
  publishAt: DateTime                 # 미래 게시 예약
}

type Mutation {
  updatePost(id: ID!, input: UpdatePostInput!): UpdatePostPayload!
}
```

---

## 7. 변수와 프래그먼트

### 변수

변수는 정적 쿼리 구조와 동적 런타임 값을 분리합니다:

```graphql
query GetUser($userId: ID!, $postLimit: Int = 5) {
  user(id: $userId) {
    username
    posts(first: $postLimit) {
      title
    }
  }
}
```

변수 JSON:

```json
{
  "userId": "42",
  "postLimit": 3
}
```

> **모범 사례**: 동적 값에는 항상 변수를 사용하세요. 사용자 입력을 쿼리 문자열에 직접 삽입하지 마세요 (인젝션 공격 방지).

### 프래그먼트

프래그먼트는 재사용 가능한 필드 선택을 가능하게 합니다:

```graphql
fragment UserBasic on User {
  id
  username
  email
  avatarUrl
}

fragment PostSummary on Post {
  id
  title
  status
  createdAt
  author {
    ...UserBasic
  }
}

query Dashboard {
  me {
    ...UserBasic
    posts(first: 10) {
      ...PostSummary
    }
  }
  trending: posts(first: 5, sortBy: TRENDING) {
    ...PostSummary
  }
}
```

### 인라인 프래그먼트

유니온과 인터페이스에서 사용됩니다:

```graphql
query {
  node(id: "abc") {
    id
    ... on User {
      username
      email
    }
    ... on Post {
      title
      content
    }
  }
}
```

### 디렉티브

내장 디렉티브는 필드 포함을 제어합니다:

```graphql
query GetUser($userId: ID!, $includeEmail: Boolean!) {
  user(id: $userId) {
    username
    email @include(if: $includeEmail)
    internalNotes @skip(if: true)
  }
}
```

| 디렉티브 | 동작 |
|----------|------|
| `@include(if: Boolean!)` | `true`인 경우에만 필드 포함 |
| `@skip(if: Boolean!)` | `true`인 경우 필드 건너뛰기 |
| `@deprecated(reason: String)` | 스키마 필드를 더 이상 사용되지 않음으로 표시 |

---

## 8. 인트로스펙션

GraphQL 스키마는 자기 문서화됩니다. 클라이언트가 스키마 자체를 쿼리할 수 있습니다.

### 스키마 인트로스펙션 쿼리

```graphql
query IntrospectSchema {
  __schema {
    types {
      name
      kind
      description
    }
    queryType { name }
    mutationType { name }
    subscriptionType { name }
  }
}
```

### 타입 인트로스펙션

```graphql
query IntrospectType {
  __type(name: "User") {
    name
    kind
    fields {
      name
      type {
        name
        kind
        ofType {
          name
        }
      }
      args {
        name
        type { name }
        defaultValue
      }
    }
  }
}
```

### 인트로스펙션의 용도

- **GraphiQL / GraphQL Playground**: 자동 완성 및 문서화
- **코드 생성**: 스키마에서 타입이 지정된 클라이언트 생성
- **스키마 검증**: 호환성 깨지는 변경 사항을 확인하는 CI 파이프라인

> **보안 참고**: 스키마 유출을 방지하기 위해 프로덕션 공개 API에서 인트로스펙션을 비활성화하세요. 내부 API 및 개발 환경에서는 활성화된 상태로 유지합니다.

---

## 9. 최소 Python 서버 구축

데이터클래스와 타입 힌트를 사용하는 현대적인 Python GraphQL 라이브러리인 [Strawberry](https://strawberry.rocks/)를 사용하여 작동하는 GraphQL 서버를 구축해 봅시다.

### 설치

```bash
pip install strawberry-graphql[fastapi] uvicorn
```

### 스키마 정의 (코드 우선)

```python
# schema.py
import strawberry
from datetime import datetime
from typing import Optional


@strawberry.enum
class PostStatus(str):
    DRAFT = "DRAFT"
    PUBLISHED = "PUBLISHED"
    ARCHIVED = "ARCHIVED"


@strawberry.type
class User:
    id: strawberry.ID
    username: str
    email: str
    bio: Optional[str] = None
    created_at: datetime = strawberry.field(name="createdAt")


@strawberry.type
class Post:
    id: strawberry.ID
    title: str
    content: str
    status: PostStatus
    author_id: strawberry.ID
    created_at: datetime = strawberry.field(name="createdAt")


# 데모용 인메모리 저장소
USERS = {
    "1": User(id="1", username="alice", email="alice@example.com",
              bio="Engineer", created_at=datetime(2024, 1, 1)),
    "2": User(id="2", username="bob", email="bob@example.com",
              bio=None, created_at=datetime(2024, 2, 1)),
}

POSTS = {
    "101": Post(id="101", title="Intro to GraphQL", content="...",
                status=PostStatus.PUBLISHED, author_id="1",
                created_at=datetime(2024, 3, 1)),
}


@strawberry.input
class CreatePostInput:
    title: str
    content: str
    author_id: strawberry.ID
    status: PostStatus = PostStatus.DRAFT


@strawberry.type
class Query:
    @strawberry.field
    def user(self, id: strawberry.ID) -> Optional[User]:
        return USERS.get(id)

    @strawberry.field
    def users(self, limit: int = 10) -> list[User]:
        return list(USERS.values())[:limit]

    @strawberry.field
    def post(self, id: strawberry.ID) -> Optional[Post]:
        return POSTS.get(id)


@strawberry.type
class Mutation:
    @strawberry.mutation
    def create_post(self, input: CreatePostInput) -> Post:
        post_id = str(len(POSTS) + 200)
        post = Post(
            id=post_id,
            title=input.title,
            content=input.content,
            status=input.status,
            author_id=input.author_id,
            created_at=datetime.now(),
        )
        POSTS[post_id] = post
        return post


schema = strawberry.Schema(query=Query, mutation=Mutation)
```

### FastAPI 통합

```python
# main.py
from fastapi import FastAPI
from strawberry.fastapi import GraphQLRouter
from schema import schema

app = FastAPI(title="GraphQL Demo")
graphql_router = GraphQLRouter(schema)
app.include_router(graphql_router, prefix="/graphql")


@app.get("/health")
def health():
    return {"status": "ok"}
```

### 서버 실행

```bash
uvicorn main:app --reload --port 8000
# http://localhost:8000/graphql 에서 GraphiQL 인터페이스 열기
```

### curl로 테스트

```bash
# 쿼리
curl -X POST http://localhost:8000/graphql \
  -H "Content-Type: application/json" \
  -d '{"query": "{ user(id: \"1\") { username email } }"}'

# 뮤테이션
curl -X POST http://localhost:8000/graphql \
  -H "Content-Type: application/json" \
  -d '{
    "query": "mutation($input: CreatePostInput!) { createPost(input: $input) { id title } }",
    "variables": {
      "input": {
        "title": "New Post",
        "content": "Hello World",
        "authorId": "1"
      }
    }
  }'
```

### 응답 형식

모든 GraphQL 응답은 다음 구조를 갖습니다:

```json
{
  "data": { ... },
  "errors": [ ... ],
  "extensions": { ... }
}
```

- `data`: 결과 (요청이 유효하면 항상 존재)
- `errors`: 오류 객체 배열 (오류가 발생한 경우 존재)
- `extensions`: 선택적 메타데이터 (타이밍, 추적 등)

---

## 10. 연습 문제

### 연습 1: 스키마 설계

전자상거래 플랫폼을 위한 GraphQL 스키마(SDL 사용)를 설계하세요:
- Products (id, name, price, category, inStock)
- Categories (id, name, products)
- Cart (id, items, total)
- CartItem (product, quantity)
- 쿼리: product, products (필터링 포함), cart
- 뮤테이션: addToCart, removeFromCart, checkout

### 연습 2: 쿼리 작성

다음 스키마가 주어졌을 때, 다음 시나리오에 대한 쿼리를 작성하세요:

```graphql
type Query {
  movie(id: ID!): Movie
  movies(genre: Genre, year: Int, limit: Int): [Movie!]!
  search(query: String!): [SearchResult!]!
}
union SearchResult = Movie | Actor | Director
```

a) id "tt0111161"인 영화를 제목, 연도, 감독 이름, 처음 5명의 출연진과 함께 가져오기
b) "Nolan"을 검색하고 세 가지 결과 타입을 모두 처리하기
c) 별칭을 사용하여 한 요청에서 두 개의 다른 영화 가져오기

### 연습 3: 뮤테이션 페이로드

이 뮤테이션을 구조화된 오류가 포함된 Shopify 스타일 페이로드 패턴으로 재설계하세요:

```graphql
# Before
type Mutation {
  registerUser(email: String!, password: String!, username: String!): User!
}
```

### 연습 4: Python 서버

9절의 최소 Strawberry 서버를 확장하여 다음을 추가하세요:
- `Post`와 `User` 모두에 연결된 `Comment` 타입
- `createComment` 뮤테이션
- 게시글의 모든 댓글을 가져오는 쿼리

---

## 11. 참고 자료

### 사양
- [GraphQL Specification (October 2021)](https://spec.graphql.org/October2021/)
- [GraphQL over HTTP Specification](https://graphql.github.io/graphql-over-http/)

### 공식 자료
- [GraphQL.org — 공식 문서](https://graphql.org/learn/)
- [GraphQL Foundation](https://graphql.org/foundation/)

### 라이브러리
- [Strawberry — Python GraphQL Library](https://strawberry.rocks/)
- [Apollo Server — JavaScript/TypeScript](https://www.apollographql.com/docs/apollo-server/)
- [graphql-core — Python 참조 구현](https://github.com/graphql-python/graphql-core)

### 도구
- [GraphiQL — 브라우저 내 IDE](https://github.com/graphql/graphiql)
- [Apollo Studio — 스키마 레지스트리 및 탐색기](https://studio.apollographql.com/)
- [GraphQL Voyager — 스키마 시각화](https://graphql-kit.com/graphql-voyager/)

---

**License**: CC BY-NC 4.0
