# 18. GraphQL 스키마 설계(GraphQL Schema Design)

**이전**: [GraphQL 기초](./17_GraphQL_Fundamentals.md) | **다음**: [GraphQL 리졸버](./19_GraphQL_Resolvers.md)

**난이도**: ⭐⭐⭐

---

## 학습 목표

- 직관적이고 진화 가능한 GraphQL API를 생성하는 스키마 설계 원칙을 적용할 수 있다
- 유효성 검사 제약 조건과 기본값을 가진 효과적인 입력 타입을 설계할 수 있다
- 열거형, 인터페이스, 유니온 타입을 사용하여 도메인 개념을 정확하게 모델링할 수 있다
- 도메인 특화 데이터 타입을 위한 커스텀 스칼라를 구현할 수 있다
- 커서 기반 페이지네이션을 위한 Relay Connection 사양을 적용할 수 있다
- 구조화된 오류 처리가 포함된 입력/페이로드 패턴으로 뮤테이션을 설계할 수 있다

---

## 목차

1. [스키마 설계 원칙](#1-스키마-설계-원칙)
2. [네이밍 컨벤션](#2-네이밍-컨벤션)
3. [객체 타입 설계](#3-객체-타입-설계)
4. [입력 타입](#4-입력-타입)
5. [열거형](#5-열거형)
6. [인터페이스와 추상 타입](#6-인터페이스와-추상-타입)
7. [유니온 타입](#7-유니온-타입)
8. [커스텀 스칼라](#8-커스텀-스칼라)
9. [페이지네이션: Relay Connection](#9-페이지네이션-relay-connection)
10. [뮤테이션 설계 패턴](#10-뮤테이션-설계-패턴)
11. [스키마 진화](#11-스키마-진화)
12. [연습 문제](#12-연습-문제)
13. [참고 자료](#13-참고-자료)

---

## 1. 스키마 설계 원칙

잘 설계된 GraphQL 스키마는 성공적인 GraphQL API에서 가장 중요한 요소입니다. 스키마는 문서화이자 계약입니다.

### 핵심 원칙

| 원칙 | 설명 |
|------|------|
| **클라이언트 중심** | 데이터 저장 방식이 아닌 클라이언트의 데이터 소비 방식에 맞게 설계 |
| **표현력** | 스키마가 도메인 문서처럼 읽혀야 함 |
| **진화 가능** | 클라이언트를 깨뜨리지 않고 새 필드와 타입 추가 가능 |
| **일관성** | 네이밍 패턴을 스키마 전체에 걸쳐 균일하게 따름 |
| **최소성** | 클라이언트가 필요한 것만 노출; 데이터베이스 테이블 미러링 금지 |

### 엔드포인트가 아닌 그래프로 사고하기

```
REST 사고방식:
  GET /users/42
  GET /users/42/posts
  GET /posts/101/comments

GraphQL 사고방식:
  User --[has many]--> Post --[has many]--> Comment
       --[has many]--> Follower
       --[has one]---> Profile
```

API 연산이 아닌 도메인 관계를 중심으로 타입을 설계하세요.

### 도메인 주도 설계 정렬

GraphQL 타입을 데이터베이스 테이블이 아닌 도메인 엔티티에 매핑하세요:

```graphql
# 나쁨: DB 테이블 미러링
type user_accounts {
  user_id: Int!
  user_name: String
  fk_profile_id: Int
}

# 좋음: 도메인 모델링
type User {
  id: ID!
  username: String!
  profile: Profile!
  posts(first: Int): PostConnection!
}
```

---

## 2. 네이밍 컨벤션

일관된 네이밍은 스키마를 자기 문서화합니다.

### 표준 컨벤션

| 요소 | 컨벤션 | 예시 |
|------|--------|------|
| 타입 | PascalCase | `User`, `BlogPost` |
| 필드 | camelCase | `firstName`, `createdAt` |
| 열거형 | SCREAMING_SNAKE_CASE | `POST_STATUS`, `PUBLISHED` |
| 입력 타입 | PascalCase + `Input` 접미사 | `CreateUserInput` |
| 페이로드 타입 | PascalCase + `Payload` 접미사 | `CreateUserPayload` |
| 뮤테이션 | camelCase 동사 | `createUser`, `deletePost` |
| 쿼리 | camelCase 명사 | `user`, `posts`, `searchResults` |
| 인자 | camelCase | `firstName`, `sortBy` |

### CRUD 네이밍 패턴

```graphql
type Query {
  # 단수: ID로 조회
  user(id: ID!): User
  post(id: ID!): Post

  # 복수: 목록 조회
  users(first: Int, after: String): UserConnection!
  posts(filter: PostFilter, first: Int, after: String): PostConnection!
}

type Mutation {
  # 동사 + 명사
  createUser(input: CreateUserInput!): CreateUserPayload!
  updateUser(input: UpdateUserInput!): UpdateUserPayload!
  deleteUser(id: ID!): DeleteUserPayload!
}
```

### 불리언 필드

불리언 필드에는 `is`, `has`, `can` 접두사를 사용합니다:

```graphql
type User {
  isVerified: Boolean!
  hasAvatar: Boolean!
  canEdit: Boolean!        # 뷰어 권한에 따라 계산됨
}
```

---

## 3. 객체 타입 설계

### 단일 책임

각 타입은 하나의 명확한 도메인 개념을 나타내야 합니다:

```graphql
# 나쁨: 혼합된 관심사
type User {
  id: ID!
  username: String!
  orderTotal: Float!       # Order 집계에 속함
  shippingAddress: String! # Address 타입에 속함
}

# 좋음: 분리된 관심사
type User {
  id: ID!
  username: String!
  orders: OrderConnection!
  addresses: [Address!]!
}
```

### 계산 필드

다른 데이터에서 파생되는 필드를 추가합니다:

```graphql
type Post {
  id: ID!
  title: String!
  content: String!
  wordCount: Int!           # content에서 계산
  readingTimeMinutes: Int!  # wordCount에서 계산
  isPublished: Boolean!     # status에서 계산
  excerpt(length: Int = 200): String! # 매개변수화된 계산 필드
}
```

### 필드 인자(Field Arguments)

필드는 응답을 커스터마이즈하기 위한 인자를 받을 수 있습니다:

```graphql
type User {
  id: ID!
  username: String!

  # 기본값이 있는 인자
  posts(
    status: PostStatus
    first: Int = 10
    after: String
    orderBy: PostOrderField = CREATED_AT
    orderDirection: OrderDirection = DESC
  ): PostConnection!

  # 포맷이 지정된 필드
  createdAt(format: DateFormat = ISO8601): String!
}
```

### 널러빌리티 가이드라인

| 시나리오 | 널러블? | 근거 |
|----------|---------|------|
| ID 필드 | Non-null (`!`) | 항상 존재 |
| 필수 비즈니스 데이터 | Non-null (`!`) | 반드시 존재해야 함 |
| 선택적 프로필 필드 | Nullable | 설정되지 않을 수 있음 |
| 부모와의 관계 | Non-null (`!`) | Post에는 항상 Author가 있음 |
| 선택적 관계 | Nullable | User에 Profile이 없을 수 있음 |
| 리스트 | Non-null 리스트 (`[T!]!`) | null 대신 빈 리스트 반환 |
| 계산 필드 | Non-null (`!`) | 항상 계산 가능 |

> **경험 법칙**: 기본적으로 non-null을 사용하세요. 부재에 대한 유효한 비즈니스 이유가 있을 때만 필드를 nullable로 만드세요.

---

## 4. 입력 타입

입력 타입은 뮤테이션 인자를 구조화합니다.

### 설계 지침

```graphql
# Create: 모든 필수 필드가 non-null
input CreateProductInput {
  name: String!
  description: String!
  price: Float!
  categoryId: ID!
  tags: [String!] = []      # 기본 빈 리스트
  isPublished: Boolean = false
}

# Update: 모든 필드가 nullable (부분 업데이트)
input UpdateProductInput {
  name: String
  description: String
  price: Float
  categoryId: ID
  tags: [String!]
  isPublished: Boolean
}
```

### 재사용 vs. 특수화(Reuse vs. Specificity)

```graphql
# 나쁨: 모든 작업에 하나의 입력 타입 사용
input ProductInput {
  id: ID           # update에만 필요
  name: String!    # create에서는 필수, update에서는 선택
  price: Float!
}

# 좋음: 작업별 별도 입력 타입
input CreateProductInput {
  name: String!
  price: Float!
}

input UpdateProductInput {
  name: String
  price: Float
}
```

### 중첩 입력 타입

```graphql
input CreateOrderInput {
  items: [OrderItemInput!]!
  shippingAddress: AddressInput!
  billingAddress: AddressInput
  couponCode: String
}

input OrderItemInput {
  productId: ID!
  quantity: Int!
}

input AddressInput {
  street: String!
  city: String!
  state: String
  postalCode: String!
  country: String!
}
```

---

## 5. 열거형

열거형은 유효한 값의 고정 집합을 정의합니다.

### 열거형 사용 시점

```graphql
# 상태 워크플로우
enum OrderStatus {
  PENDING
  CONFIRMED
  SHIPPED
  DELIVERED
  CANCELLED
  REFUNDED
}

# 정렬 옵션
enum ProductSortField {
  NAME
  PRICE
  CREATED_AT
  POPULARITY
}

enum SortDirection {
  ASC
  DESC
}
```

### 열거형 네이밍 규칙(Enum Naming Rules)

- 값에는 SCREAMING_SNAKE_CASE 사용
- 관련 값은 함께 그룹화
- 새 값은 끝에 추가 (순서 기대치 유지)
- 프로덕션에서는 값을 제거하거나 이름 변경 금지 (대신 deprecated 처리)

### 열거형 폐기

```graphql
enum PostStatus {
  DRAFT
  PUBLISHED
  ARCHIVED
  DELETED @deprecated(reason: "ARCHIVED를 대신 사용하세요. 소프트 삭제는 isDeleted 필드로 처리됩니다.")
}
```

### 필터 인자로서의 열거형(Enums as Filter Arguments)

```graphql
type Query {
  products(
    category: ProductCategory
    sortBy: ProductSortField = CREATED_AT
    sortDirection: SortDirection = DESC
    status: [ProductStatus!]  # 여러 상태로 필터링
  ): ProductConnection!
}
```

---

## 6. 인터페이스와 추상 타입

인터페이스는 여러 타입에 걸쳐 공유 필드를 정의합니다.

### Node 인터페이스

GraphQL에서 가장 일반적인 인터페이스 (Relay 사양):

```graphql
interface Node {
  id: ID!
}

type User implements Node {
  id: ID!
  username: String!
}

type Post implements Node {
  id: ID!
  title: String!
}

type Query {
  node(id: ID!): Node  # 글로벌 ID로 모든 엔티티 조회
}
```

### 도메인 인터페이스

```graphql
interface Auditable {
  createdAt: DateTime!
  updatedAt: DateTime!
  createdBy: User!
  updatedBy: User
}

interface Publishable {
  status: PublishStatus!
  publishedAt: DateTime
  author: User!
}

type Article implements Node & Auditable & Publishable {
  id: ID!
  title: String!
  content: String!
  createdAt: DateTime!
  updatedAt: DateTime!
  createdBy: User!
  updatedBy: User
  status: PublishStatus!
  publishedAt: DateTime
  author: User!
}
```

### 인터페이스 쿼리하기(Querying Interfaces)

```graphql
query {
  node(id: "abc123") {
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

### 인터페이스 vs. 유니온

| 관점 | 인터페이스 | 유니온 |
|------|-----------|--------|
| 공유 필드 | 필수 | 없음 |
| 공통 동작 | 예 | 아니오 |
| 사용 사례 | 구조를 공유하는 타입 | 개념적으로 관련되지만 구조적으로 다른 타입 |
| 예시 | `Node { id }` | `SearchResult = User \| Post \| Comment` |

---

## 7. 유니온 타입

유니온은 필드가 여러 무관한 타입 중 하나를 반환할 수 있는 상황을 모델링합니다.

### 검색 결과

```graphql
union SearchResult = User | Post | Comment | Tag

type Query {
  search(query: String!, types: [SearchableType!]): [SearchResult!]!
}

enum SearchableType {
  USER
  POST
  COMMENT
  TAG
}
```

### 다형성 피드(Polymorphic Feeds)

```graphql
union FeedItem = TextPost | ImagePost | VideoPost | SharedLink | Poll

type Feed {
  items(first: Int, after: String): FeedItemConnection!
}
```

### 오류 유니온 (Result 패턴)

페이로드 기반 오류의 대안:

```graphql
union CreateUserResult = User | ValidationError | DuplicateEmailError

type ValidationError {
  field: String!
  message: String!
}

type DuplicateEmailError {
  email: String!
  message: String!
}

type Mutation {
  createUser(input: CreateUserInput!): CreateUserResult!
}
```

클라이언트 사용법:

```graphql
mutation {
  createUser(input: { username: "alice", email: "alice@example.com" }) {
    ... on User {
      id
      username
    }
    ... on ValidationError {
      field
      message
    }
    ... on DuplicateEmailError {
      email
      message
    }
  }
}
```

---

## 8. 커스텀 스칼라

커스텀 스칼라는 도메인 특화 타입 안전성을 추가합니다.

### 일반적인 커스텀 스칼라

```graphql
scalar DateTime    # ISO 8601 날짜-시간 문자열
scalar Date        # ISO 8601 날짜 (시간 없음)
scalar Email       # RFC 5322 이메일 주소
scalar URL         # RFC 3986 URI
scalar JSON        # 임의의 JSON 블롭
scalar UUID        # RFC 4122 UUID
scalar Decimal     # 정밀한 십진수 (금액)
```

### Strawberry에서 구현

```python
import strawberry
from datetime import datetime, date
from decimal import Decimal
import re


EmailScalar = strawberry.scalar(
    str,
    name="Email",
    description="RFC 5322 email address",
    serialize=lambda v: str(v),
    parse_value=lambda v: _validate_email(v),
)


def _validate_email(value: str) -> str:
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    if not re.match(pattern, value):
        raise ValueError(f"Invalid email address: {value}")
    return value


URLScalar = strawberry.scalar(
    str,
    name="URL",
    description="RFC 3986 URI",
    serialize=lambda v: str(v),
    parse_value=lambda v: _validate_url(v),
)


def _validate_url(value: str) -> str:
    from urllib.parse import urlparse
    result = urlparse(value)
    if not all([result.scheme, result.netloc]):
        raise ValueError(f"Invalid URL: {value}")
    return value


# Using custom scalars in types
@strawberry.type
class User:
    id: strawberry.ID
    email: EmailScalar          # type: ignore
    website: URLScalar | None   # type: ignore
```

### 커스텀 스칼라를 만들어야 할 때(When to Create Custom Scalars)

| 생성 | 생성하지 않음 |
|------|---------------|
| 유효성 검사가 있는 도메인 값 (Email, URL) | 단순 별칭 (String의 Name) |
| 정밀도 타입 (금액을 위한 Decimal) | 구조가 있는 타입 (객체 타입 사용) |
| 표준 형식 (DateTime, UUID) | 복합 값 (입력 타입 사용) |

---

## 9. 페이지네이션: Relay Connection

Relay Connection 사양은 페이지네이션된 리스트를 위한 표준 패턴을 제공합니다.

### Connection 타입

```graphql
type PostConnection {
  edges: [PostEdge!]!
  pageInfo: PageInfo!
  totalCount: Int!
}

type PostEdge {
  node: Post!
  cursor: String!
}

type PageInfo {
  hasNextPage: Boolean!
  hasPreviousPage: Boolean!
  startCursor: String
  endCursor: String
}
```

### 페이지네이션 인자(Pagination Arguments)

```graphql
type Query {
  # 순방향 페이지네이션
  posts(first: Int!, after: String): PostConnection!

  # 역방향 페이지네이션
  posts(last: Int!, before: String): PostConnection!

  # 필터링 포함
  posts(
    first: Int
    after: String
    filter: PostFilterInput
    orderBy: PostOrderInput
  ): PostConnection!
}
```

### 사용법

```graphql
query {
  posts(first: 10, after: "cursor_abc") {
    edges {
      cursor
      node {
        id
        title
        author { username }
      }
    }
    pageInfo {
      hasNextPage
      endCursor
    }
    totalCount
  }
}
```

### 커서 구현(Cursor Implementation)

```python
import base64
from typing import Any


def encode_cursor(type_name: str, id_value: Any) -> str:
    """Encode a cursor as a base64 string."""
    raw = f"{type_name}:{id_value}"
    return base64.b64encode(raw.encode()).decode()


def decode_cursor(cursor: str) -> tuple[str, str]:
    """Decode a base64 cursor into (type_name, id_value)."""
    raw = base64.b64decode(cursor.encode()).decode()
    type_name, id_value = raw.split(":", 1)
    return type_name, id_value


# Example
cursor = encode_cursor("Post", 42)   # "UG9zdDo0Mg=="
type_name, id_val = decode_cursor(cursor)  # ("Post", "42")
```

### 오프셋 vs. 커서 페이지네이션

| 관점 | 오프셋 | 커서 |
|------|--------|------|
| 단순성 | 단순 | 보통 |
| 성능 | 높은 오프셋에서 저하 | 일관적 |
| 안정성 | 불안정 (삽입이 페이지 이동) | 안정적 |
| 랜덤 접근 | 가능 (`page=5`) | 불가 (순차적만) |
| 적합 사례 | 소규모 데이터셋, 관리자 패널 | 대규모 데이터셋, 피드, 무한 스크롤 |

---

## 10. 뮤테이션 설계 패턴

### 입력/페이로드 패턴

Shopify, GitHub, Stripe에서 사용하는 권장 패턴:

```graphql
input CreateOrderInput {
  items: [OrderItemInput!]!
  shippingAddressId: ID!
  couponCode: String
}

type CreateOrderPayload {
  order: Order
  userErrors: [UserError!]!
}

type UserError {
  field: [String!]!
  message: String!
  code: UserErrorCode!
}

type Mutation {
  createOrder(input: CreateOrderInput!): CreateOrderPayload!
}
```

### 대량 작업(Bulk Operations)

```graphql
input BulkUpdateProductStatusInput {
  ids: [ID!]!
  status: ProductStatus!
}

type BulkUpdateProductStatusPayload {
  products: [Product!]!
  failedIds: [ID!]!
  userErrors: [UserError!]!
}

type Mutation {
  bulkUpdateProductStatus(
    input: BulkUpdateProductStatusInput!
  ): BulkUpdateProductStatusPayload!
}
```

### 멱등 뮤테이션

클라이언트 생성 멱등성 키를 사용합니다:

```graphql
input CreatePaymentInput {
  orderId: ID!
  amount: Decimal!
  currency: CurrencyCode!
  idempotencyKey: String!  # 클라이언트 생성 UUID
}
```

---

## 11. 스키마 진화

### 추가적 변경 (호환성 유지)

- 새 타입 추가
- 기존 타입에 새 필드 추가
- 새 열거형 값 추가
- 필드에 선택적 인자 추가
- 새 쿼리 또는 뮤테이션 추가

### 호환성 깨지는 변경 (피하기)

- 타입 또는 필드 제거
- 필드 타입 변경
- nullable 필드를 non-null로 변경
- 열거형 값 제거
- 필드 이름 변경

### 폐기 전략

```graphql
type User {
  name: String! @deprecated(reason: "`firstName`과 `lastName`을 대신 사용하세요")
  firstName: String!
  lastName: String!
  fullName: String!  # 새 계산 필드
}
```

### 스키마 레지스트리(Schema Registry)

CI에서 스키마 변경을 추적합니다:

```bash
# Apollo Rover CLI
rover graph check my-graph@production --schema schema.graphql
rover graph publish my-graph@production --schema schema.graphql
```

---

## 12. 연습 문제

### 연습 1: 전자상거래 스키마

전자상거래 플랫폼을 위한 완전한 GraphQL 스키마를 설계하세요:
- 변형이 있는 제품(사이즈, 색상)과 가격
- 계층 구조가 있는 카테고리(부모/자식)
- 라인 아이템이 있는 장바구니
- 상태 워크플로우가 있는 주문
- 고객 리뷰 및 평점
- 모든 목록 필드에 커서 기반 페이지네이션

### 연습 2: 입력 타입 설계

다음 뮤테이션에 대해 적절한 입력 및 페이로드 타입을 설계하세요:
- `createReview` — 사용자가 제품을 리뷰 (1-5 별, 선택적 댓글)
- `updateReview` — 사용자가 리뷰를 수정
- `flagReview` — 관리자가 이유와 함께 리뷰를 신고

### 연습 3: 커스텀 스칼라

Strawberry에서 다음 커스텀 스칼라를 구현하세요:
- `Currency` — ISO 4217 통화 코드 (USD, EUR, JPY)
- `Latitude` — -90에서 90 사이의 부동소수점
- `Longitude` — -180에서 180 사이의 부동소수점

### 연습 4: 스키마 리팩토링

이 스키마에는 문제가 있습니다. 문제를 식별하고 리팩토링하세요:

```graphql
type Product {
  id: Int
  name: String
  desc: String
  price: Float
  price_currency: String
  cat_id: Int
  cat_name: String
  created: String
  tags: String  # 쉼표 구분
}
```

---

## 13. 참고 자료

### 사양
- [Relay Connection Specification](https://relay.dev/graphql/connections.htm)
- [Relay Global Object Identification](https://relay.dev/graphql/objectidentification.htm)

### 설계 가이드
- [Shopify GraphQL Design Tutorial](https://github.com/Shopify/graphql-design-tutorial)
- [GitHub GraphQL API Design Guidelines](https://docs.github.com/en/graphql/overview/about-the-graphql-api)
- [Apollo GraphQL Best Practices](https://www.apollographql.com/docs/technotes/)

### 도서
- "Production Ready GraphQL" by Marc-Andre Giroux
- "Learning GraphQL" by Eve Porcello and Alex Banks (O'Reilly)

---

**License**: CC BY-NC 4.0
