# 12. 연합(Federation)

**이전**: [지속 쿼리와 캐싱](./11_Persisted_Queries_Caching.md) | **다음**: [테스팅](./13_Testing.md)

## 학습 목표

이 레슨을 마치면 다음을 할 수 있습니다:

1. 모놀리식(monolithic)과 연합(federated) GraphQL 아키텍처를 비교하고 연합이 적합한 시점을 결정한다
2. Apollo Federation 2 지시어(@key, @external, @requires, @provides, @shareable)를 사용하여 서브그래프(subgraph)를 구축한다
3. `__resolveReference`를 사용한 엔티티 해석(entity resolution)을 구현하여 서브그래프 간에 데이터를 연결한다
4. 팀 구조가 아닌 비즈니스 도메인을 중심으로 서브그래프 경계를 설계한다
5. Rover CLI를 사용하여 슈퍼그래프(supergraph)를 구성하고 Apollo Router로 배포한다

## 목차

1. [모놀리식 vs. 연합 GraphQL](#1-모놀리식-vs-연합-graphql)
2. [Apollo Federation 2 아키텍처](#2-apollo-federation-2-아키텍처)
3. [Federation 지시어](#3-federation-지시어)
4. [엔티티 해석](#4-엔티티-해석)
5. [서브그래프 경계 설계](#5-서브그래프-경계-설계)
6. [스키마 구성](#6-스키마-구성)
7. [Apollo Router와 쿼리 플래닝](#7-apollo-router와-쿼리-플래닝)
8. [마이그레이션: 모놀리식에서 연합으로](#8-마이그레이션-모놀리식에서-연합으로)
9. [연습 문제](#9-연습-문제)

**난이도**: ⭐⭐⭐⭐

---

조직이 성장함에 따라 단일 GraphQL 서버는 병목이 됩니다 — 성능 때문이 아니라 팀 조율 때문입니다. 10개 팀이 모두 같은 스키마 파일에 커밋하면, 병합 충돌과 배포 결합이 모두의 속도를 늦춥니다. Apollo Federation은 각 팀이 그래프의 자신의 부분을 소유하면서 클라이언트에게 단일하고 통합된 API를 제공할 수 있게 해줍니다. 이 레슨은 Federation 2의 기본 원리부터 프로덕션 배포까지 다룹니다.

---

## 1. 모놀리식 vs. 연합 GraphQL

### 모놀리식 아키텍처

```
┌────────────────────────────────────────────┐
│             모놀리식 GraphQL                │
│                                              │
│  스키마: User, Product, Order, Review...    │
│  리졸버: 모두 하나의 코드베이스에           │
│  팀: 모두가 여기에 커밋                     │
│                                              │
│      ┌──────┐  ┌──────┐  ┌──────┐          │
│      │ DB 1 │  │ DB 2 │  │ DB 3 │          │
│      └──────┘  └──────┘  └──────┘          │
└────────────────────────────────────────────┘
```

**적합한 경우**: 소규모 팀(개발자 5명 미만), 단일 도메인, 단순한 배포 파이프라인.

### 연합 아키텍처

```
                    ┌──────────────┐
                    │  Apollo       │
      클라이언트 ──▶ │  Router      │
                    │  (게이트웨이) │
                    └──────┬───────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
       ┌──────────┐ ┌──────────┐ ┌──────────┐
       │ Users    │ │ Products │ │ Orders   │
       │ Subgraph │ │ Subgraph │ │ Subgraph │
       │  팀 A   │ │  팀 B   │ │  팀 C   │
       └────┬─────┘ └────┬─────┘ └────┬─────┘
            │            │            │
            ▼            ▼            ▼
       ┌────────┐  ┌────────┐  ┌────────┐
       │ UserDB │  │ ProdDB │  │OrderDB │
       └────────┘  └────────┘  └────────┘
```

**연합을 선택하는 경우**: 여러 팀, 독립적인 배포 필요, 도메인별 다른 데이터 저장소, 스키마가 큰 경우(타입 100개 이상).

### 결정 매트릭스

| 요소 | 모놀리식 | 연합 |
|------|---------|------|
| 팀 규모 | 개발자 1-5명 | 팀에 걸쳐 5명 이상 |
| 스키마 크기 | 타입 100개 미만 | 타입 100개 이상 |
| 배포 | 단일 단위 | 서브그래프별 독립 |
| 데이터 소스 | 소수, 공유 | 다수, 도메인별 |
| 운영 복잡성 | 낮음 | 높음(라우터, 구성) |
| 타입 소유권 | 공유 | 팀별 명확 |

---

## 2. Apollo Federation 2 아키텍처

Federation 2는 **슈퍼그래프(supergraph)** 개념을 도입합니다: 각각 독립적인 GraphQL 서비스로 실행되는 여러 **서브그래프(subgraph)**로 구성된 통합 스키마입니다.

### 핵심 구성 요소

```
┌─────────────────────────────────────────────────────────┐
│                      슈퍼그래프                           │
│                                                          │
│  ┌─────────────────────────────────────────────────┐    │
│  │                Apollo Router                     │    │
│  │  - 클라이언트 쿼리 수신                          │    │
│  │  - 서브그래프에 걸쳐 실행 계획                   │    │
│  │  - 최종 응답 조립                               │    │
│  └──────────────────────┬──────────────────────────┘    │
│                         │                                │
│          ┌──────────────┼──────────────┐                │
│          ▼              ▼              ▼                │
│   ┌────────────┐ ┌────────────┐ ┌────────────┐        │
│   │ 서브그래프 A│ │ 서브그래프 B│ │ 서브그래프 C│        │
│   │            │ │            │ │            │        │
│   │ 소유:      │ │ 소유:      │ │ 소유:      │        │
│   │  User      │ │  Product   │ │  Order     │        │
│   │  Account   │ │  Category  │ │  Payment   │        │
│   └────────────┘ └────────────┘ └────────────┘        │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### 서브그래프 설정

각 서브그래프는 `@apollo/subgraph` 패키지를 사용하는 독립적인 Apollo Server입니다:

```typescript
// users-subgraph/index.ts
import { ApolloServer } from '@apollo/server';
import { startStandaloneServer } from '@apollo/server/standalone';
import { buildSubgraphSchema } from '@apollo/subgraph';
import gql from 'graphql-tag';

const typeDefs = gql`
  extend schema
    @link(url: "https://specs.apollo.dev/federation/v2.0",
          import: ["@key", "@shareable"])

  type Query {
    me: User
    user(id: ID!): User
  }

  type User @key(fields: "id") {
    id: ID!
    name: String!
    email: String!
  }
`;

const resolvers = {
  Query: {
    me: (_, __, { userId }) => users.findById(userId),
    user: (_, { id }) => users.findById(id),
  },
  User: {
    __resolveReference: (ref) => users.findById(ref.id),
  },
};

const server = new ApolloServer({
  schema: buildSubgraphSchema({ typeDefs, resolvers }),
});

startStandaloneServer(server, { listen: { port: 4001 } });
```

---

## 3. Federation 지시어

Federation 2는 서브그래프 간에 타입과 필드가 공유되는 방식을 정의하는 지시어를 제공합니다.

### @key — 엔티티 식별자

타입을 엔티티로 표시하고 기본 키를 지정합니다. 엔티티는 다른 서브그래프에서 참조하고 확장할 수 있습니다.

```graphql
# Users 서브그래프 — User 엔티티 정의
type User @key(fields: "id") {
  id: ID!
  name: String!
  email: String!
}

# 복합 키
type ProductVariant @key(fields: "productId sku") {
  productId: ID!
  sku: String!
  price: Float!
}

# 다중 키 (해석에 어느 것이든 사용 가능)
type Product @key(fields: "id") @key(fields: "slug") {
  id: ID!
  slug: String!
  name: String!
}
```

### @external — 외부 필드 참조

필드가 다른 서브그래프에서 정의되었음을 선언합니다. 현재 서브그래프는 이를 해석하지 않지만 `@requires`에서 참조할 수 있습니다.

```graphql
# Orders 서브그래프 — users 서브그래프의 User 확장
type User @key(fields: "id") {
  id: ID!
  orders: [Order!]!
}

type Order @key(fields: "id") {
  id: ID!
  total: Float!
  user: User!
}
```

### @requires — 의존 필드

필드를 해석하기 위해 다른 서브그래프의 데이터가 필요함을 지정합니다.

```graphql
# Shipping 서브그래프
type Product @key(fields: "id") {
  id: ID!
  weight: Float! @external    # products 서브그래프에서 정의됨
  size: String! @external     # products 서브그래프에서 정의됨
  shippingCost: Float! @requires(fields: "weight size")
}
```

라우터는 먼저 products 서브그래프에서 `weight`와 `size`를 가져온 다음, shipping 서브그래프의 리졸버에 전달합니다:

```typescript
const resolvers = {
  Product: {
    shippingCost: (product) => {
      // product.weight와 product.size는 라우터에 의해 제공됨
      return calculateShipping(product.weight, product.size);
    },
  },
};
```

### @provides — 낙관적 해석(Optimistic Resolution)

리졸버가 자식 엔티티의 필드를 제공할 수 있음을 선언하여 추가적인 서브그래프 호출을 방지합니다.

```graphql
# Reviews 서브그래프
type Review @key(fields: "id") {
  id: ID!
  body: String!
  author: User! @provides(fields: "name")
}

type User @key(fields: "id") {
  id: ID!
  name: String! @external
}
```

리뷰에 이미 작성자의 이름이 있는 경우(예: 리뷰 레코드에 비정규화됨), 라우터는 users 서브그래프 호출을 건너뛸 수 있습니다.

### @shareable — 공유 필드

여러 서브그래프가 같은 필드를 해석할 수 있게 합니다. `@shareable` 없이는 두 서브그래프가 같은 필드를 정의하면 구성이 실패합니다.

```graphql
# 두 서브그래프 모두 Product.name을 해석할 수 있음
# Products 서브그래프
type Product @key(fields: "id") {
  id: ID!
  name: String! @shareable
  description: String!
}

# Inventory 서브그래프
type Product @key(fields: "id") {
  id: ID!
  name: String! @shareable
  inStock: Boolean!
}
```

### 지시어 요약

| 지시어 | 목적 | 사용 대상 |
|--------|------|----------|
| `@key` | 엔티티 기본 키 정의 | 타입 |
| `@external` | 다른 서브그래프의 필드 참조 | 필드 |
| `@requires` | 외부 필드에 대한 의존성 선언 | 필드 |
| `@provides` | 자식 엔티티 필드 해석 능력 선언 | 필드 |
| `@shareable` | 여러 서브그래프가 필드를 해석하도록 허용 | 필드 |
| `@override` | 서브그래프 간 필드 소유권 이전 | 필드 |
| `@inaccessible` | 공개 API에서 필드 숨김 | 필드/타입 |

---

## 4. 엔티티 해석

엔티티 해석(Entity Resolution)은 라우터가 여러 서브그래프의 데이터를 단일 응답으로 "연결"하는 메커니즘입니다.

### __resolveReference 함수

모든 엔티티 타입은 `__resolveReference`를 구현해야 합니다. 라우터는 엔티티의 키 필드로 이 함수를 호출하여 전체 객체를 가져옵니다.

```typescript
// Users 서브그래프
const resolvers = {
  User: {
    __resolveReference: async (reference, { dataSources }) => {
      // reference = { __typename: "User", id: "42" }
      // 라우터가 @key의 키 필드를 제공
      return dataSources.users.findById(reference.id);
    },
  },
};
```

### 라우터가 서브그래프 간 쿼리를 해석하는 방법

다음 클라이언트 쿼리를 생각해 보세요:

```graphql
query {
  order(id: "100") {
    id
    total
    user {       # User 엔티티는 users 서브그래프에 있음
      name
      email
    }
  }
}
```

라우터는 쿼리 계획을 생성합니다:

```
1단계: orders 서브그래프 쿼리
  → { order(id: "100") { id total user { __typename id } } }
  → 반환: { id: "100", total: 59.99, user: { __typename: "User", id: "42" } }

2단계: 엔티티 참조로 users 서브그래프 쿼리
  → _entities(representations: [{ __typename: "User", id: "42" }]) { ... on User { name email } }
  → 반환: { name: "Alice", email: "alice@example.com" }

3단계: 결과 병합
  → { order: { id: "100", total: 59.99, user: { name: "Alice", email: "alice@example.com" } } }
```

### 배치 엔티티 해석

`_entities` 쿼리는 표현(representations) 배열을 받아 서브그래프가 엔티티를 일괄 로드할 수 있게 합니다:

```typescript
const resolvers = {
  User: {
    __resolveReference: async (refs, { dataSources }) => {
      // DataLoader를 사용하면 여러 참조가 자동으로 배치 처리됨
      return dataSources.users.findById(refs.id);
    },
  },
};
```

---

## 5. 서브그래프 경계 설계

연합에서 가장 중요한 결정은 스키마를 서브그래프로 어떻게 분할하느냐입니다. 올바른 경계는 결합도를 줄이고 팀이 독립적으로 작업할 수 있게 합니다.

### 원칙: 팀 중심이 아닌 도메인 중심

```
✗ 나쁜 예: 팀 기반 경계
  "프론트엔드 팀" 서브그래프, "백엔드 팀" 서브그래프
  → 팀이 변경되고, 타입이 임의로 분할됨

✓ 좋은 예: 도메인 기반 경계
  Users 서브그래프, Products 서브그래프, Orders 서브그래프
  → 비즈니스 역량과 정렬됨
```

### 경계 가이드라인

1. **비즈니스 역량별 그룹화**: 사용자/인증, 상품/카탈로그, 주문/결제, 배송/이행
2. **서브그래프 간 의존성 최소화**: 두 타입이 항상 함께 나타나면 같은 서브그래프에 유지
3. **데이터 소유권 고려**: 데이터베이스를 소유한 팀이 서브그래프를 소유해야 함
4. **순환 의존성 방지**: 서브그래프 A가 B에 의존하고, B가 A에 의존하지 않도록

### 예시: 전자상거래 경계

```
Users 서브그래프         Products 서브그래프      Orders 서브그래프
─────────────          ──────────────────       ────────────────
User @key(id)          Product @key(id)         Order @key(id)
Account                Category                 OrderItem
Address                Review                   Payment
                       ProductVariant           Shipment

                       # 리뷰를 위해 User 확장
                       User @key(id) {          # 주문을 위해 User 확장
                         reviews: [Review!]!     User @key(id) {
                       }                           orders: [Order!]!
                                                 }
```

---

## 6. 스키마 구성

구성(Composition)은 여러 서브그래프 스키마를 단일 슈퍼그래프 스키마로 결합하는 과정입니다.

### Rover CLI

Rover는 연합 스키마를 관리하기 위한 Apollo의 공식 CLI입니다.

```bash
# Rover 설치
npm install -g @apollo/rover

# 서브그래프 스키마 검증
rover subgraph check my-graph@production \
  --name users \
  --schema ./users/schema.graphql

# 로컬에서 슈퍼그래프 구성
rover supergraph compose --config ./supergraph-config.yaml \
  --output supergraph.graphql
```

### 슈퍼그래프 구성

```yaml
# supergraph-config.yaml
federation_version: =2.0.0
subgraphs:
  users:
    routing_url: http://users-service:4001/graphql
    schema:
      file: ./users/schema.graphql
  products:
    routing_url: http://products-service:4002/graphql
    schema:
      file: ./products/schema.graphql
  orders:
    routing_url: http://orders-service:4003/graphql
    schema:
      file: ./orders/schema.graphql
```

### CI에서의 스키마 검사

```yaml
# .github/workflows/schema-check.yml
name: Schema Check
on:
  pull_request:
    paths:
      - 'subgraphs/**/*.graphql'

jobs:
  check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Install Rover
        run: |
          curl -sSL https://rover.apollo.dev/nix/latest | sh
          echo "$HOME/.rover/bin" >> $GITHUB_PATH
      - name: Check schema
        run: |
          rover subgraph check my-graph@production \
            --name ${{ matrix.subgraph }} \
            --schema ./subgraphs/${{ matrix.subgraph }}/schema.graphql
        env:
          APOLLO_KEY: ${{ secrets.APOLLO_KEY }}
    strategy:
      matrix:
        subgraph: [users, products, orders]
```

### 구성 오류

일반적인 구성 실패와 해결 방법:

```
오류: "Product.name" 필드가 여러 서브그래프에 정의되어 있으나
  @shareable로 표시되지 않음
해결: 두 서브그래프의 필드에 @shareable 추가

오류: "User" 엔티티에 "orders" 서브그래프의 @key 지시어가 없음
해결: orders의 User 타입 스텁에 @key(fields: "id") 추가

오류: "Product.weight" 필드가 @external이지만 @requires에서 사용되지 않음
해결: 이를 사용하는 @requires를 추가하거나 @external 주석 제거
```

---

## 7. Apollo Router와 쿼리 플래닝

Apollo Router는 구형 `@apollo/gateway` Node.js 패키지를 대체하는 프로덕션 게이트웨이입니다. Rust로 작성되어 훨씬 낮은 지연 시간과 리소스 사용으로 쿼리 플래닝과 실행을 처리합니다.

### 라우터 구성

```yaml
# router.yaml
supergraph:
  path: ./supergraph.graphql
  # 또는 Apollo GraphOS에서 가져오기:
  # apollo_graph_ref: my-graph@production

cors:
  origins:
    - https://myapp.com
  allow_headers:
    - Content-Type
    - Authorization

headers:
  all:
    request:
      - propagate:
          named: Authorization

traffic_shaping:
  router:
    timeout: 30s
  all:
    timeout: 15s

telemetry:
  exporters:
    tracing:
      jaeger:
        endpoint: http://jaeger:14268/api/traces
```

### 라우터 실행

```bash
# Apollo Router 다운로드
curl -sSL https://router.apollo.dev/download/nix/latest | sh

# 로컬 슈퍼그래프로 실행
./router --config router.yaml --supergraph supergraph.graphql

# 또는 Apollo GraphOS로 실행
APOLLO_KEY=your-api-key APOLLO_GRAPH_REF=my-graph@production ./router
```

### 쿼리 플래닝

라우터는 각 수신 쿼리를 분석하고 서브그래프 호출을 최소화하는 실행 계획을 생성합니다.

```graphql
# 클라이언트 쿼리
query GetOrderWithDetails {
  order(id: "100") {
    id
    total
    items {
      product {
        name
        category { name }
        reviews { body author { name } }
      }
    }
    user { name email }
  }
}
```

```
쿼리 계획:
─────────────
Fetch(orders) ──▶ { order(id: "100") { id total items { product { __typename id } } user { __typename id } } }
    │
    ├─▶ Fetch(products) ──▶ _entities[Product] { name category { name } reviews { body author { __typename id } } }
    │       │
    │       └─▶ Fetch(users) ──▶ _entities[User] { name }  (리뷰 작성자)
    │
    └─▶ Fetch(users) ──▶ _entities[User] { name email }   (주문 사용자)
```

라우터는 독립적인 페치를 병렬화하고 엔티티 참조를 배치 처리합니다.

---

## 8. 마이그레이션: 모놀리식에서 연합으로

### 1단계: 도메인 경계 식별

모놀리식 스키마를 분석하고 타입을 도메인별로 그룹화합니다:

```graphql
# 모놀리식 스키마 — 자연스러운 그룹화 식별
type Query {
  user(id: ID!): User          # → Users 서브그래프
  products: [Product!]!        # → Products 서브그래프
  order(id: ID!): Order        # → Orders 서브그래프
}
```

### 2단계: 한 번에 하나의 서브그래프 추출

가장 결합도가 낮은 도메인부터 시작합니다:

```typescript
// 단계 1: 모놀리식이 "기본" 서브그래프가 됨
// 모든 것을 제공하지만 스키마는 이제 federation 인식

// 단계 2: users 서브그래프 추출
// - User 타입과 관련 리졸버를 users 서비스로 이동
// - 모놀리식에는 스텁 유지: type User @key(fields: "id") { id: ID! }
// - User를 반환하는 모놀리식 리졸버는 이제 { id }만 반환

// 단계 3: products 서브그래프 추출
// 동일한 과정, 한 번에 하나의 도메인
```

### 3단계: 점진적 롤아웃

```
단계 1:  [─── 모놀리식(연합) ───]                  ← 여기서 시작
단계 2:  [Users] [─── 모놀리식 ───────]             ← users 추출
단계 3:  [Users] [Products] [─ 모놀리식 ─]         ← products 추출
단계 4:  [Users] [Products] [Orders]                ← 완전히 연합됨
```

핵심 원칙:
- "빅뱅" 마이그레이션은 절대 하지 않음
- 첫날부터 모놀리식 앞에 라우터 배치
- 한 번에 하나의 서브그래프 추출, 검증 후 진행
- `@override`를 사용하여 점진적으로 필드 소유권 이전

```graphql
# 새 서브그래프로 필드를 일시적으로 재정의
# 구형 모놀리식에 여전히 필드가 있지만 라우터는 새 서브그래프를 선호
type User @key(fields: "id") {
  id: ID!
  name: String! @override(from: "monolith")
}
```

---

## 9. 연습 문제

### 문제 1: 서브그래프 설계
소셜 미디어 플랫폼의 다음 타입들이 주어졌을 때, 명확한 도메인 경계를 가진 3개의 서브그래프를 설계하세요. 어떤 타입이 엔티티인지, 어떤 필드에 `@external`이 필요한지, `@requires`가 필요한 곳이 어디인지 식별하세요:

```
User, Profile, Post, Comment, Like, Follow,
Notification, Message, Group, Event, Photo, Video
```

### 문제 2: 엔티티 해석
`Products` 서브그래프의 `Product` 엔티티를 확장하는 `Reviews` 서브그래프를 구현하세요. Reviews 서브그래프는 Product에 `reviews` 필드와 `averageRating` 필드를 추가해야 합니다. typeDefs, 리졸버, `__resolveReference`를 포함한 전체 서브그래프 코드를 작성하세요.

### 문제 3: 구성 디버깅
다음 두 서브그래프 스키마가 구성에 실패합니다. 모든 오류를 식별하고 수정하세요:

```graphql
# 서브그래프 A
type Product @key(fields: "id") {
  id: ID!
  name: String!
  price: Float!
}

# 서브그래프 B
type Product @key(fields: "id") {
  id: ID!
  name: String!
  inventory: Int!
}
```

### 문제 4: 마이그레이션 계획
50개의 타입과 200개의 리졸버를 가진 모놀리식 GraphQL 서버가 있습니다. 세 팀(Users, Commerce, Content)이 독립적으로 작업해야 합니다. 단계별 마이그레이션 계획을 작성하세요: 먼저 추출할 서브그래프, 모놀리식에 남길 스텁, Rover 스키마 검사를 사용한 각 단계 검증 방법.

### 문제 5: 쿼리 계획 분석
`Users`, `Products`, `Orders` 서브그래프를 가진 슈퍼그래프가 주어졌을 때, 다음 클라이언트 쿼리에 대한 쿼리 계획(어떤 서브그래프가 어떤 순서로, 어떤 쿼리와 함께 호출되는지)을 작성하세요:

```graphql
query {
  me {
    name
    orders {
      id
      items { product { name price reviews { body } } }
    }
  }
}
```

---

## 참고 자료

- [Apollo Federation 2 문서](https://www.apollographql.com/docs/federation/)
- [Apollo Router 구성](https://www.apollographql.com/docs/router/configuration/overview)
- [Rover CLI 레퍼런스](https://www.apollographql.com/docs/rover/)
- [Federation 사양](https://www.apollographql.com/docs/federation/federation-spec/)
- [원칙적인 GraphQL - Federation](https://principledgraphql.com/agility#4-abstract-demand-oriented-schema)
- [Netflix DGS Federation](https://netflix.github.io/dgs/federation/)

---

**이전**: [지속 쿼리와 캐싱](./11_Persisted_Queries_Caching.md) | **다음**: [테스팅](./13_Testing.md)
