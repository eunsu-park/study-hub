# 22. GraphQL 페더레이션(GraphQL Federation)

**이전**: [GraphQL 서버 구현](./21_GraphQL_Server_Implementation.md) | **다음**: [GraphQL 성능과 보안](./23_GraphQL_Performance_Security.md)

**난이도**: ⭐⭐⭐⭐

---

## 학습 목표

- 스키마 페더레이션의 개념과 모놀리식 스키마가 문제가 되는 이유를 설명할 수 있다
- 엔티티 타입, 키, 서비스 간 참조를 가진 서브그래프 스키마를 설계할 수 있다
- `__resolveReference` 패턴을 사용하여 엔티티 해석을 구현할 수 있다
- Apollo Federation 2 라우터를 구성하여 서브그래프 스키마를 합성할 수 있다
- 페더레이션 디렉티브(`@key`, `@shareable`, `@external`, `@requires`, `@provides`)를 적용할 수 있다
- 모놀리식 GraphQL 스키마에서 페더레이션 아키텍처로의 마이그레이션을 계획할 수 있다

---

## 목차

1. [페더레이션이 필요한 이유](#1-페더레이션이-필요한-이유)
2. [페더레이션 아키텍처](#2-페더레이션-아키텍처)
3. [서브그래프 설계](#3-서브그래프-설계)
4. [엔티티 해석](#4-엔티티-해석)
5. [페더레이션 디렉티브](#5-페더레이션-디렉티브)
6. [Apollo Federation 2](#6-apollo-federation-2)
7. [Python에서 서브그래프 구현](#7-python에서-서브그래프-구현)
8. [라우터 구성](#8-라우터-구성)
9. [페더레이션 스키마 테스트](#9-페더레이션-스키마-테스트)
10. [모놀리스에서 마이그레이션](#10-모놀리스에서-마이그레이션)
11. [연습 문제](#11-연습-문제)
12. [참고 자료](#12-참고-자료)

---

## 1. 페더레이션이 필요한 이유

조직이 성장함에 따라 단일 모놀리식 GraphQL 스키마는 병목이 됩니다.

### 모놀리식 스키마의 문제점

| 문제 | 영향 |
|------|------|
| **단일 코드베이스** | 모든 팀이 같은 스키마 파일을 수정 |
| **배포 결합** | 한 타입의 변경이 전체 재배포 필요 |
| **소유권 혼란** | `User` 같은 공유 타입의 명확한 소유자 없음 |
| **확장 한계** | 하나의 서버가 모든 리졸버 처리 |
| **개발 속도** | 머지 충돌, 조정 오버헤드 |

### 페더레이션이 해결하는 것

```
이전 (모놀리스):
  Client → 단일 GraphQL 서버 (모든 타입, 모든 리졸버)

이후 (페더레이션):
  Client → Router → Users 서브그래프 (User 타입)
                   → Posts 서브그래프 (Post 타입)
                   → Comments 서브그래프 (Comment 타입)
```

### 장점

- **독립적 배포**: 각 서브그래프가 독립적으로 배포
- **팀 소유권**: 서브그래프별 명확한 소유권 경계
- **점진적 채택**: 한 번에 하나의 도메인씩 마이그레이션
- **이종 구현**: 서브그래프가 다른 언어 사용 가능
- **집중 확장**: 핫 서브그래프를 독립적으로 확장

---

## 2. 페더레이션 아키텍처

### 컴포넌트

```
┌──────────┐     ┌──────────────────────────────────────────┐
│  Client   │────▶│              Router (Gateway)             │
└──────────┘     │   - 스키마 합성                           │
                 │   - 쿼리 계획                             │
                 │   - 결과 병합                             │
                 └──┬──────────┬──────────┬────────────────┘
                    │          │          │
              ┌─────▼──┐ ┌────▼───┐ ┌────▼───┐
              │ Users   │ │ Posts  │ │Comments│
              │서브그래프│ │서브그래프│ │서브그래프│
              └────┬────┘ └────┬───┘ └────┬───┘
              ┌────▼────┐ ┌───▼────┐ ┌───▼────┐
              │Users DB │ │Posts DB│ │Cmts DB │
              └─────────┘ └────────┘ └────────┘
```

### 쿼리 실행 흐름(Query Execution Flow)

```graphql
# 클라이언트가 라우터에 이 쿼리를 전송합니다:
query {
  user(id: "1") {        # Users 서브그래프가 해석
    username
    posts(first: 5) {    # Posts 서브그래프가 해석
      title
      comments {          # Comments 서브그래프가 해석
        body
        author {          # Users 서브그래프가 해석 (엔티티 참조)
          username
        }
      }
    }
  }
}
```

라우터는 **쿼리 계획(query plan)**을 작성합니다:

```
1. Users 서브그래프에서 user 조회
2. Posts 서브그래프에서 해당 사용자의 posts 조회
3. Comments 서브그래프에서 각 post의 comments 조회
4. Users 서브그래프에서 댓글 작성자 해석 (엔티티 참조)
5. 결과를 단일 응답으로 병합
```

---

## 3. 서브그래프 설계

### 엔티티 정의

**엔티티**는 서브그래프 간에 해석될 수 있는 타입입니다. `@key` 디렉티브를 사용합니다:

```graphql
# Users 서브그래프
type User @key(fields: "id") {
  id: ID!
  username: String!
  email: String!
}

# Posts 서브그래프
type User @key(fields: "id") {
  id: ID!
  posts(first: Int = 10): [Post!]!   # Posts 서브그래프가 추가
}

type Post @key(fields: "id") {
  id: ID!
  title: String!
  content: String!
  author: User!
}
```

### 소유권 규칙

| 규칙 | 설명 |
|------|------|
| **단일 소유자** | 각 타입 필드에 정확히 하나의 소유 서브그래프 |
| **엔티티 확장** | 다른 서브그래프가 엔티티에 필드 추가 가능 |
| **스텁 타입** | 비소유 서브그래프는 `@key` 필드만 선언 |
| **값 타입** | 공유 타입(열거형, 스칼라)은 `@shareable` 가능 |

### 도메인 경계 가이드라인(Domain Boundary Guidelines)

```
Users 서브그래프:
  소유: User, Profile, Address
  확장: 없음 (원본 서브그래프)

Posts 서브그래프:
  소유: Post, Tag, PostConnection
  확장: User (posts 필드 추가)

Comments 서브그래프:
  소유: Comment
  확장: User (comments 필드 추가), Post (comments 필드 추가)

Orders 서브그래프:
  소유: Order, OrderItem, Payment
  확장: User (orders 필드 추가), Product (orders 필드 추가)
```

---

## 4. 엔티티 해석

엔티티 해석은 라우터가 서브그래프 경계를 넘어 타입 데이터를 가져오는 방법입니다.

### `__resolveReference` 함수

```python
@strawberry.federation.type(keys=["id"])
class User:
    id: strawberry.ID
    username: str
    email: str

    @classmethod
    async def resolve_reference(cls, info, id: strawberry.ID) -> "User":
        """라우터가 User 엔티티를 해석하기 위해 호출합니다."""
        user = await info.context.db.users.find_by_id(id)
        return cls(id=user.id, username=user.username, email=user.email)
```

### 라우터의 참조 사용 방식

```
1. Comments 서브그래프 반환:
   { comment: { body: "좋아요!", author: { __typename: "User", id: "42" } } }

2. 라우터가 User 엔티티 참조 감지

3. 라우터가 Users 서브그래프 호출:
   query {
     _entities(representations: [{ __typename: "User", id: "42" }]) {
       ... on User { username email }
     }
   }

4. Users 서브그래프가 참조를 해석하여 반환

5. 라우터가 최종 응답으로 병합
```

---

## 5. 페더레이션 디렉티브

### Apollo Federation 2 디렉티브

| 디렉티브 | 목적 | 예시 |
|----------|------|------|
| `@key` | 고유 키 필드로 엔티티 타입 표시 | `type User @key(fields: "id")` |
| `@shareable` | 여러 서브그래프가 필드를 해석할 수 있음 | `type Money @shareable` |
| `@external` | 다른 서브그래프에서 정의된 필드 표시 | `id: ID! @external` |
| `@requires` | 다른 서브그래프에서 필요한 필드 선언 | `shippingCost: Float @requires(fields: "weight")` |
| `@provides` | 해석 시 제공되는 추가 필드 선언 | `author: User @provides(fields: "email")` |
| `@override` | 한 서브그래프에서 다른 서브그래프로 필드 마이그레이션 | `username: String! @override(from: "users")` |
| `@inaccessible` | 합성된 스키마에서 필드 숨기기 | `internalNote: String @inaccessible` |

### `@requires` 예시

```graphql
# Products 서브그래프
type Product @key(fields: "id") {
  id: ID!
  weight: Float!
  size: String!
}

# Shipping 서브그래프
type Product @key(fields: "id") {
  id: ID!
  weight: Float! @external
  size: String! @external
  shippingCost: Float! @requires(fields: "weight size")
}
```

라우터는 Shipping 서브그래프의 `shippingCost` 리졸버를 호출하기 전에 Products 서브그래프에서 `weight`와 `size`를 먼저 가져옵니다.

### `@provides` 예시

```graphql
# Reviews 서브그래프 — reviews 해석 시 author.email을 함께 제공
type Review @key(fields: "id") {
  id: ID!
  body: String!
  author: User @provides(fields: "email")
}

type User @key(fields: "id") {
  id: ID!
  email: String! @external
}
```

### `@shareable` 예시

여러 서브그래프에서 동일하게 정의되는 값 타입:

```graphql
# 두 서브그래프 모두 이 타입을 정의하고 해석할 수 있음
type Money @shareable {
  amount: Float!
  currency: CurrencyCode!
}

enum CurrencyCode @shareable {
  USD
  EUR
  GBP
  JPY
}
```

---

## 6. Apollo Federation 2

### 슈퍼그래프 스키마

합성된 스키마(슈퍼그래프)는 모든 서브그래프 스키마의 조합입니다.

### 합성

```bash
# Rover CLI 설치
curl -sSL https://rover.apollo.dev/nix/latest | sh

# 서브그래프 스키마에서 슈퍼그래프 합성
rover supergraph compose --config supergraph.yaml > supergraph.graphql
```

```yaml
# supergraph.yaml
federation_version: =2.7.0
subgraphs:
  users:
    routing_url: http://users-service:4001/graphql
    schema:
      file: ./subgraphs/users/schema.graphql
  posts:
    routing_url: http://posts-service:4002/graphql
    schema:
      file: ./subgraphs/posts/schema.graphql
  comments:
    routing_url: http://comments-service:4003/graphql
    schema:
      file: ./subgraphs/comments/schema.graphql
```

### 합성 유효성 검사(Composition Validation)

합성 단계에서 다음을 검증합니다:
- 타입 정의 충돌 없음
- 모든 엔티티 참조가 해석 가능
- 필수 필드 접근 가능
- `@requires`에 순환 의존성 없음

```bash
# 합성 오류 확인
rover supergraph compose --config supergraph.yaml 2>&1

# 주요 오류 예시:
# - EXTERNAL_MISSING_ON_BASE: 원본 서브그래프에 없는 @external 필드
# - KEY_FIELDS_SELECT_INVALID_TYPE: 존재하지 않는 필드의 @key
# - REQUIRES_FIELDS_MISSING_EXTERNAL: @external로 표시되지 않은 @requires 필드
```

---

## 7. Python에서 서브그래프 구현

### Strawberry 페더레이션 지원

```python
# users_subgraph/schema.py
import strawberry
from strawberry.federation import Schema


@strawberry.federation.type(keys=["id"])
class User:
    id: strawberry.ID
    username: str
    email: str
    bio: str | None = None

    @classmethod
    async def resolve_reference(cls, info, id: strawberry.ID) -> "User":
        user = await info.context.db.users.find_by_id(id)
        return cls(
            id=user.id,
            username=user.username,
            email=user.email,
            bio=user.bio,
        )


@strawberry.type
class Query:
    @strawberry.field
    async def user(self, info, id: strawberry.ID) -> User | None:
        return await info.context.db.users.find_by_id(id)

    @strawberry.field
    async def users(self, info, limit: int = 10) -> list[User]:
        return await info.context.db.users.find_all(limit=limit)


schema = Schema(query=Query, enable_federation_2=True)
```

### User를 확장하는 Posts 서브그래프

```python
# posts_subgraph/schema.py
import strawberry
from strawberry.federation import Schema


@strawberry.federation.type(keys=["id"])
class User:
    id: strawberry.ID

    @strawberry.field
    async def posts(self, info, first: int = 10) -> list["Post"]:
        return await info.context.db.posts.find_by_author(
            self.id, limit=first
        )


@strawberry.federation.type(keys=["id"])
class Post:
    id: strawberry.ID
    title: str
    content: str
    author_id: strawberry.ID

    @strawberry.field
    async def author(self) -> User:
        return User(id=self.author_id)

    @classmethod
    async def resolve_reference(cls, info, id: strawberry.ID) -> "Post":
        post = await info.context.db.posts.find_by_id(id)
        return cls(
            id=post.id,
            title=post.title,
            content=post.content,
            author_id=post.author_id,
        )


@strawberry.type
class Query:
    @strawberry.field
    async def post(self, info, id: strawberry.ID) -> Post | None:
        return await info.context.db.posts.find_by_id(id)


schema = Schema(query=Query, enable_federation_2=True)
```

---

## 8. 라우터 구성

### Apollo Router

```bash
curl -sSL https://router.apollo.dev/download/nix/latest | sh
./router --supergraph supergraph.graphql --config router.yaml
```

```yaml
# router.yaml
supergraph:
  listen: 0.0.0.0:4000

cors:
  origins:
    - https://app.example.com
  methods:
    - GET
    - POST

headers:
  all:
    request:
      - propagate:
          named: Authorization
      - propagate:
          named: X-Request-ID

limits:
  max_depth: 15
  max_height: 200

telemetry:
  instrumentation:
    spans:
      mode: spec_compliant
```

### 전체 스택 Docker Compose

```yaml
version: "3.9"
services:
  router:
    image: ghcr.io/apollographql/router:v1.40.0
    ports:
      - "4000:4000"
    volumes:
      - ./supergraph.graphql:/etc/config/supergraph.graphql
      - ./router.yaml:/etc/config/router.yaml
    command: --supergraph /etc/config/supergraph.graphql --config /etc/config/router.yaml
    depends_on:
      - users
      - posts
      - comments

  users:
    build: ./subgraphs/users
    ports:
      - "4001:8000"

  posts:
    build: ./subgraphs/posts
    ports:
      - "4002:8000"

  comments:
    build: ./subgraphs/comments
    ports:
      - "4003:8000"
```

---

## 9. 페더레이션 스키마 테스트

### 합성 테스트

```bash
# 오류 없이 스키마가 합성되는지 검증
rover supergraph compose --config supergraph.yaml

# 프로덕션 대비 호환성 깨지는 변경 확인
rover subgraph check my-graph@production \
  --name users \
  --schema subgraphs/users/schema.graphql
```

### 서브그래프 단위 테스트

```python
# tests/test_users_subgraph.py
import pytest
from strawberry.test import GraphQLTestClient
from schema import schema


@pytest.fixture
def client():
    return GraphQLTestClient(schema)


def test_user_query(client):
    result = client.query("""
        query {
            user(id: "1") {
                id
                username
                email
            }
        }
    """)
    assert result.errors is None
    assert result.data["user"]["username"] == "alice"


def test_entity_resolution(client):
    """라우터가 사용하는 _entities 쿼리를 테스트합니다."""
    result = client.query("""
        query {
            _entities(representations: [
                { __typename: "User", id: "1" }
            ]) {
                ... on User { username email }
            }
        }
    """)
    assert result.errors is None
    assert result.data["_entities"][0]["username"] == "alice"
```

### 통합 테스트

```python
# tests/test_federation_integration.py
import httpx
import pytest

ROUTER_URL = "http://localhost:4000/graphql"


@pytest.mark.integration
async def test_cross_subgraph_query():
    """여러 서브그래프에 걸친 쿼리를 테스트합니다."""
    async with httpx.AsyncClient() as client:
        response = await client.post(ROUTER_URL, json={
            "query": """
                query {
                    user(id: "1") {
                        username
                        posts(first: 3) {
                            title
                            comments(first: 2) {
                                body
                                author { username }
                            }
                        }
                    }
                }
            """
        })
    data = response.json()
    assert "errors" not in data
    assert data["data"]["user"]["username"] is not None
    assert len(data["data"]["user"]["posts"]) <= 3
```

---

## 10. 모놀리스에서 마이그레이션

### 단계적 마이그레이션 전략

```
1단계: Strangler Fig 패턴
  - 모놀리스 앞에 라우터 배포
  - 모놀리스가 첫 번째 서브그래프가 됨
  - 클라이언트 쿼리가 라우터를 통과 (투명)

2단계: 첫 번째 서브그래프 추출
  - 바운디드 컨텍스트 선택 (예: Users)
  - Users 서브그래프 구축
  - 모놀리스에서 User 리졸버 마이그레이션

3단계: 반복
  - 다음 서브그래프 추출 (Posts, Comments 등)
  - @override를 사용하여 점진적으로 필드 마이그레이션

4단계: 모놀리스 해체
  - 모든 타입이 서브그래프로 추출됨
  - 모놀리스 종료
```

### `@override`를 사용한 점진적 마이그레이션

```graphql
# 2단계: Posts 서브그래프가 모놀리스에서 `posts` 필드를 인수
type User @key(fields: "id") {
  id: ID!
  posts: [Post!]! @override(from: "monolith")
}
```

### 검증 체크리스트

| 단계 | 검증 |
|------|------|
| 스키마 합성 | `rover supergraph compose` 성공 |
| 엔티티 해석 | `_entities` 쿼리가 올바른 데이터 반환 |
| 성능 | 지연시간이 모놀리스의 10% 이내 |
| 오류 처리 | 오류가 라우터를 통해 올바르게 전파 |
| 인증 | 인증 헤더가 서브그래프로 전달 |
| 모니터링 | 각 서브그래프의 메트릭 가시성 확인 |

---

## 11. 연습 문제

### 연습 1: 서브그래프 경계 설계

Users, Products, Orders, Reviews, Inventory, Payments, Shipping이 있는 전자상거래 도메인에서 서브그래프 경계를 설계하세요.

### 연습 2: 두 개의 서브그래프 구현

Strawberry 페더레이션으로 Users와 Posts 서브그래프를 구축하세요.

### 연습 3: 라우터 합성

Rover CLI를 사용하여 두 서브그래프의 슈퍼그래프 스키마를 합성하세요.

### 연습 4: 마이그레이션 계획

50개 타입과 200개 필드를 가진 모놀리식 GraphQL 서버의 마이그레이션 계획을 작성하세요.

---

## 12. 참고 자료

### 공식 문서
- [Apollo Federation 2 Docs](https://www.apollographql.com/docs/federation/)
- [Apollo Router Documentation](https://www.apollographql.com/docs/router/)
- [Rover CLI Documentation](https://www.apollographql.com/docs/rover/)

### 페더레이션 구현체
- [Strawberry Federation](https://strawberry.rocks/docs/guides/federation)
- [Apollo Server Federation](https://www.apollographql.com/docs/apollo-server/using-federation/apollo-subgraph-setup/)
- [GraphQL Mesh — 대안 게이트웨이](https://the-guild.dev/graphql/mesh)

### 아티클
- "Apollo Federation: A Revolution in GraphQL Architecture" — Apollo Blog
- "Federated GraphQL at Netflix" — Netflix Tech Blog

---

**License**: CC BY-NC 4.0
