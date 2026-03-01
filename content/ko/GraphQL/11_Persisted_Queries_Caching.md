# 11. 지속 쿼리와 캐싱(Persisted Queries and Caching)

**이전**: [코드 우선 방식(Python)](./10_Code_First_Python.md) | **다음**: [연합(Federation)](./12_Federation.md)

## 학습 목표

이 레슨을 마치면 다음을 할 수 있습니다:

1. GraphQL의 POST 기반 모델이 전통적인 HTTP 캐싱을 우회하는 이유와 이를 해결하는 방법을 설명한다
2. CDN 친화적인 쿼리 캐싱을 가능하게 하는 자동 지속 쿼리(APQ, Automatic Persisted Queries)를 구현한다
3. 빌드 시점 추출과 작업 허용 목록(allowlist)을 사용한 완전한 지속 쿼리(Full Persisted Queries)를 구성한다
4. CDN, 서버 측, 클라이언트 측 캐시를 결합한 다층 캐싱(multi-layer caching) 전략을 설계한다
5. 캐시 정책(cache policies), 제거(eviction), 가비지 컬렉션(garbage collection)을 사용하여 Apollo Client의 InMemoryCache를 관리한다

## 목차

1. [GraphQL 캐싱이 필요한 이유](#1-graphql-캐싱이-필요한-이유)
2. [자동 지속 쿼리(APQ)](#2-자동-지속-쿼리apq)
3. [완전한 지속 쿼리](#3-완전한-지속-쿼리)
4. [HTTP 캐싱 전략](#4-http-캐싱-전략)
5. [Apollo Client 캐시](#5-apollo-client-캐시)
6. [서버 측 캐싱](#6-서버-측-캐싱)
7. [다층 캐싱 아키텍처](#7-다층-캐싱-아키텍처)
8. [연습 문제](#8-연습-문제)

**난이도**: ⭐⭐⭐

---

GraphQL의 유연성은 가장 큰 장점인 동시에 캐싱의 가장 큰 걸림돌입니다. 모든 쿼리는 서로 다른 형태의 데이터를 요청할 수 있으며, 표준 전송 방식은 POST 요청을 사용합니다 — 이는 HTTP 캐시가 기본적으로 무시하는 방식입니다. 이 레슨에서는 GraphQL을 REST만큼 캐시 가능하게 만드는 기술들을 살펴봅니다. POST를 GET으로 전환하는 지속 쿼리부터, 서버 부하를 크게 줄이는 다층 캐싱 아키텍처까지 다룹니다.

---

## 1. GraphQL 캐싱이 필요한 이유

### POST의 문제점

REST API는 HTTP 캐싱에 자연스럽게 대응합니다: `GET /api/users/42`는 항상 같은 리소스를 반환하며, CDN, 브라우저, 리버스 프록시 모두 GET 요청을 캐시하는 방법을 이해합니다. GraphQL은 이 모델을 깨뜨립니다.

```
REST (기본적으로 캐시 가능):
  GET /api/users/42
  → CDN이 응답을 캐시
  → 브라우저가 응답을 캐시
  → 리버스 프록시가 응답을 캐시

GraphQL (기본적으로 캐시 불가):
  POST /graphql
  Body: { "query": "{ user(id: 42) { name email } }" }
  → CDN: "POST 요청이므로 캐시 건너뜀"
  → 브라우저: "POST 요청이므로 캐시 건너뜀"
  → 리버스 프록시: "POST 요청이므로 캐시 건너뜀"
```

### 세 가지 과제

1. **전송 계층**: POST 요청이 HTTP 캐싱 인프라를 우회함
2. **고유성**: 각 클라이언트가 같은 데이터에 대해 다른 쿼리 형태를 보낼 수 있음
3. **세분성**: 단일 GraphQL 응답에 캐시 유효 기간이 서로 다른 데이터가 포함될 수 있음

### 캐싱 계층

```
┌─────────────────────────────────────────────────┐
│                  클라이언트 계층                    │
│  Apollo InMemoryCache / urql의 Graphcache         │
├─────────────────────────────────────────────────┤
│                   CDN 계층                         │
│  CloudFront / Fastly / Cloudflare                 │
│  (GET 요청 또는 지속 쿼리 필요)                    │
├─────────────────────────────────────────────────┤
│                 애플리케이션 계층                   │
│  응답 캐시 플러그인 / Redis / Memcached            │
├─────────────────────────────────────────────────┤
│                   데이터 계층                       │
│  DataLoader 요청 범위 캐시 / ORM 캐시             │
└─────────────────────────────────────────────────┘
```

---

## 2. 자동 지속 쿼리(APQ)

APQ(Automatic Persisted Queries)는 클라이언트와 서버 사이의 프로토콜로, 전체 쿼리 문자열을 짧은 해시로 대체합니다. 처음 해시를 받으면 클라이언트가 전체 쿼리를 전송하고, 이후 요청은 해시만 사용합니다 — GET 요청에 충분히 작은 크기입니다.

### APQ 작동 방식

```
1단계: 클라이언트가 해시만 전송
  GET /graphql?extensions={"persistedQuery":{"sha256Hash":"abc123..."}}
  → 서버: "해당 해시가 없음" → PersistedQueryNotFound 반환

2단계: 클라이언트가 해시 + 전체 쿼리 전송
  POST /graphql
  Body: { "query": "{ user(id: 42) { name } }",
          "extensions": { "persistedQuery": { "sha256Hash": "abc123..." } } }
  → 서버: 해시→쿼리 매핑 저장, 데이터 반환

3단계: 이후 모든 요청은 해시만 사용
  GET /graphql?extensions={"persistedQuery":{"sha256Hash":"abc123..."}}
  → 서버: 해시로 쿼리 찾아 실행 후 데이터 반환
  → CDN이 이 GET 요청을 캐시할 수 있음!
```

### 서버 설정 (Apollo Server 4)

```typescript
import { ApolloServer } from '@apollo/server';
import { expressMiddleware } from '@apollo/server/express4';
import { ApolloServerPluginCacheControl } from '@apollo/server/plugin/cacheControl';
import { KeyvAdapter } from '@apollo/utils.keyvadapter';
import Keyv from 'keyv';

const server = new ApolloServer({
  typeDefs,
  resolvers,
  // APQ는 Apollo Server 4에서 기본적으로 활성화됨
  // 캐시 백엔드 구성:
  cache: new KeyvAdapter(new Keyv('redis://localhost:6379')),
  plugins: [
    ApolloServerPluginCacheControl({
      defaultMaxAge: 60, // 기본 60초
    }),
  ],
});
```

### 클라이언트 설정 (Apollo Client)

```typescript
import { ApolloClient, InMemoryCache, HttpLink } from '@apollo/client';
import { createPersistedQueryLink } from '@apollo/client/link/persisted-queries';
import { sha256 } from 'crypto-hash';

const httpLink = new HttpLink({ uri: '/graphql' });

// APQ 링크는 체인에서 HTTP 링크 앞에 위치
const persistedQueryLink = createPersistedQueryLink({
  sha256,
  useGETForHashedQueries: true, // 캐시 친화적 요청에 GET 사용
});

const client = new ApolloClient({
  link: persistedQueryLink.concat(httpLink),
  cache: new InMemoryCache(),
});
```

### GET 요청과 CDN을 활용한 APQ

`useGETForHashedQueries`가 활성화되면 요청이 단순한 GET이 됩니다:

```
GET /graphql
  ?operationName=GetUser
  &variables={"id":"42"}
  &extensions={"persistedQuery":{"version":1,"sha256Hash":"ecf4edb..."}}
```

이 URL은 어떤 CDN에서도 캐시할 수 있습니다. 서버에서 `Cache-Control` 헤더를 추가해 TTL(유효 기간)을 제어하세요.

---

## 3. 완전한 지속 쿼리

완전한 지속 쿼리(Full Persisted Queries)는 APQ보다 한 단계 더 나아갑니다: 쿼리-해시 매핑이 **빌드 시점**에 확립되며, 서버는 알려진 해시만 허용합니다. 이는 성능과 보안 두 가지 이점을 제공합니다.

### 빌드 시점 쿼리 추출

```typescript
// queries/GetUser.graphql
query GetUser($id: ID!) {
  user(id: $id) {
    id
    name
    email
    avatar
  }
}
```

`graphql-codegen`과 같은 도구를 사용해 빌드 시점에 모든 쿼리를 추출합니다:

```yaml
# codegen.yml
schema: "http://localhost:4000/graphql"
documents: "src/**/*.graphql"
generates:
  ./src/generated/persisted-queries.json:
    plugins:
      - graphql-codegen-persisted-query-ids
```

이는 매니페스트(manifest)를 생성합니다:

```json
{
  "ecf4edb46db40b5132295c0291d62fb65d6759a9eedfa4d5d612dd5ec54a6b38": "query GetUser($id: ID!) { user(id: $id) { id name email avatar } }",
  "a1b2c3d4e5f6...": "query ListProducts($first: Int) { products(first: $first) { edges { node { id name price } } } }"
}
```

### 서버 측 허용 목록(Allowlist)

```typescript
import persistedQueries from './persisted-queries.json';

const server = new ApolloServer({
  typeDefs,
  resolvers,
  persistedQueries: {
    cache: new KeyvAdapter(new Keyv()),
  },
  // 허용 목록에 있는 쿼리만 허용
  validationRules: [
    (context) => ({
      Document(node) {
        // 사용자 정의 규칙: 매니페스트에 없는 쿼리 거부
      },
    }),
  ],
});
```

### 완전한 지속 쿼리를 사용하는 이유

| 이점 | 설명 |
|------|------|
| **보안** | 사전 승인된 쿼리만 실행 가능 — 임의 쿼리 인젝션 차단 |
| **성능** | 알려진 쿼리에 대한 파싱/검증 불필요 |
| **대역폭** | 해시는 ~64바이트 대 수 킬로바이트의 쿼리 텍스트 |
| **CDN 친화적** | 짧은 GET URL이 CDN 캐싱에 이상적 |

---

## 4. HTTP 캐싱 전략

### GraphQL과 Cache-Control

Apollo Server의 캐시 제어 플러그인은 스키마의 `@cacheControl` 지시어를 읽습니다:

```graphql
type Query {
  # 공개 데이터, 5분간 캐시 가능
  products: [Product!]! @cacheControl(maxAge: 300, scope: PUBLIC)

  # 사용자별 데이터, 개인 캐시만 허용
  me: User @cacheControl(maxAge: 60, scope: PRIVATE)
}

type Product {
  id: ID!
  name: String!
  price: Float!
  # 재고는 자주 변경됨
  stockCount: Int! @cacheControl(maxAge: 30)
}
```

서버는 응답의 모든 필드에 걸쳐 **최솟값** maxAge로 전체 `Cache-Control` 헤더를 계산합니다:

```
# 쿼리가 products(300초)와 stockCount(30초)를 포함하는 경우:
Cache-Control: public, max-age=30
```

### HTTP 캐싱을 위한 GET 요청

```typescript
// GET 쿼리를 지원하는 Express 미들웨어
import express from 'express';

const app = express();

// Apollo Server는 GET과 POST를 자동으로 처리함
// GET: 쿼리와 변수가 쿼리 문자열에 포함
// POST: 쿼리와 변수가 요청 본문에 포함
app.use('/graphql', expressMiddleware(server));
```

### CDN 구성 (CloudFront 예시)

```yaml
# CloudFront의 /graphql에 대한 동작 설정
CacheBehavior:
  PathPattern: /graphql
  AllowedMethods:
    - GET
    - HEAD
    - OPTIONS
  CachedMethods:
    - GET
    - HEAD
  ForwardedValues:
    QueryString: true  # 중요: 쿼리 문자열에 따라 캐시 변형
    Headers:
      - Authorization  # 인증된 응답을 함께 캐시하지 않음
  DefaultTTL: 60
  MaxTTL: 300
```

---

## 5. Apollo Client 캐시

Apollo Client의 `InMemoryCache`는 타입과 ID로 데이터를 저장하는 정규화된 반응형 캐시입니다. 이 동작 방식을 이해하는 것은 반응형 GraphQL 애플리케이션을 구축하는 데 필수적입니다.

### 정규화 캐시 구조

```typescript
// 다음 쿼리가 반환될 때:
// { user: { __typename: "User", id: "42", name: "Alice", posts: [{ __typename: "Post", id: "1", title: "Hello" }] } }

// 캐시는 이를 다음과 같이 저장:
{
  "ROOT_QUERY": {
    "user({\"id\":\"42\"})": { "__ref": "User:42" }
  },
  "User:42": {
    "__typename": "User",
    "id": "42",
    "name": "Alice",
    "posts": [{ "__ref": "Post:1" }]
  },
  "Post:1": {
    "__typename": "Post",
    "id": "1",
    "title": "Hello"
  }
}
```

### 캐시 정책(fetchPolicy)

```typescript
const { data, loading } = useQuery(GET_USER, {
  variables: { id: '42' },
  // 페치 정책 선택:
  fetchPolicy: 'cache-first',       // 기본값: 캐시 사용, 없을 때만 페치
  // fetchPolicy: 'cache-and-network', // 즉시 캐시 반환 후 네트워크에서 업데이트
  // fetchPolicy: 'network-only',      // 항상 페치하되 캐시 업데이트
  // fetchPolicy: 'cache-only',        // 페치하지 않고 캐시만 사용
  // fetchPolicy: 'no-cache',          // 항상 페치, 캐시에 저장하지 않음
});
```

### 각 정책 사용 시점

| 정책 | 사용 사례 |
|------|----------|
| `cache-first` | 정적 데이터(카테고리, 설정). 대부분의 쿼리 기본값. |
| `cache-and-network` | 변경되지만 오래된 데이터도 허용되는 경우(소셜 피드, 대시보드) |
| `network-only` | 항상 최신이어야 하는 데이터(결제 정보, 실시간 재고) |
| `cache-only` | 오프라인 모드 또는 이미 캐시된 데이터를 읽는 경우 |
| `no-cache` | 저장하지 않아야 하는 민감한 데이터(토큰, 비밀) |

### 타입 정책과 필드 병합(Field Merging)

```typescript
const cache = new InMemoryCache({
  typePolicies: {
    Query: {
      fields: {
        // 페이지네이션 결과를 교체하지 않고 병합
        products: {
          keyArgs: ['category'],  // 카테고리별로 별도 캐시 항목
          merge(existing = { edges: [] }, incoming) {
            return {
              ...incoming,
              edges: [...existing.edges, ...incoming.edges],
            };
          },
        },
      },
    },
    Product: {
      fields: {
        // 파생 데이터를 계산하는 읽기 함수 제공
        displayPrice: {
          read(_, { readField }) {
            const price = readField('price');
            const currency = readField('currency');
            return `${currency} ${price.toFixed(2)}`;
          },
        },
      },
    },
  },
});
```

### 캐시 제거(Eviction)와 가비지 컬렉션

```typescript
// 특정 엔티티 제거
cache.evict({ id: 'User:42' });

// 엔티티의 특정 필드 제거
cache.evict({ id: 'User:42', fieldName: 'posts' });

// 도달 불가능한 객체를 제거하는 가비지 컬렉션 실행
cache.gc();

// 캐시된 데이터 직접 수정
cache.modify({
  id: 'User:42',
  fields: {
    name(existingName) {
      return 'Updated Name';
    },
    posts(existingPosts, { toReference }) {
      // 새 게시물 참조 추가
      return [...existingPosts, toReference({ __typename: 'Post', id: '99' })];
    },
  },
});
```

---

## 6. 서버 측 캐싱

### 응답 캐시 플러그인

`@apollo/server-plugin-response-cache` 플러그인은 쿼리 + 변수를 키로 하여 전체 응답을 캐시합니다:

```typescript
import responseCachePlugin from '@apollo/server-plugin-response-cache';

const server = new ApolloServer({
  typeDefs,
  resolvers,
  plugins: [
    responseCachePlugin({
      // 사용자 ID를 사용하여 인증된 사용자의 캐시된 응답을 분리
      sessionId: (requestContext) =>
        requestContext.request.http?.headers.get('authorization') || null,
    }),
  ],
});
```

### Redis 기반 캐시 백엔드

```typescript
import { KeyvAdapter } from '@apollo/utils.keyvadapter';
import Keyv from 'keyv';
import KeyvRedis from '@keyv/redis';

const server = new ApolloServer({
  typeDefs,
  resolvers,
  // APQ와 응답 캐싱 모두에 Redis 저장소 사용
  cache: new KeyvAdapter(
    new Keyv({
      store: new KeyvRedis('redis://localhost:6379'),
      namespace: 'gql',
      ttl: 300_000, // 밀리초 단위 5분
    })
  ),
});
```

### @cacheControl을 활용한 부분 쿼리 캐싱

진정한 강점은 필드 수준의 캐시 힌트에서 옵니다. 각 타입과 필드가 자체 캐시 정책을 선언할 수 있으며, 서버는 전체 응답의 캐시 가능성을 계산합니다:

```graphql
type Query {
  topProducts: [Product!]! @cacheControl(maxAge: 3600, scope: PUBLIC)
  cart: Cart @cacheControl(maxAge: 0, scope: PRIVATE)
}

type Product @cacheControl(maxAge: 600) {
  id: ID!
  name: String!
  reviews: [Review!]! @cacheControl(maxAge: 120)
}
```

```typescript
// 리졸버에서 동적으로 캐시 힌트를 설정할 수 있음
const resolvers = {
  Query: {
    topProducts: (_, __, { cacheControl }) => {
      // 런타임에 스키마 힌트 재정의
      cacheControl.setCacheHint({ maxAge: 1800, scope: 'PUBLIC' });
      return db.products.findTop();
    },
  },
};
```

---

## 7. 다층 캐싱 아키텍처

프로덕션 GraphQL 배포는 여러 캐싱 계층을 결합합니다:

```
클라이언트 요청
     │
     ▼
┌─────────────────────┐
│   Apollo Client      │  계층 1: 정규화 클라이언트 캐시
│   InMemoryCache      │  - 불필요한 네트워크 요청 제거
│                      │  - 캐시된 데이터로 즉각적인 UI 업데이트
└──────────┬──────────┘
           │ 캐시 미스
           ▼
┌─────────────────────┐
│   CDN / 엣지 캐시   │  계층 2: HTTP 캐시(APQ + GET)
│   CloudFront 등     │  - 지리적 분산
│                      │  - 읽기 위주의 공개 데이터 처리
└──────────┬──────────┘
           │ 캐시 미스
           ▼
┌─────────────────────┐
│   응답 캐시         │  계층 3: 서버 응답 캐시
│   (Redis)           │  - 전체 응답 캐싱
│                      │  - 인증 데이터를 위한 세션 인식
└──────────┬──────────┘
           │ 캐시 미스
           ▼
┌─────────────────────┐
│   DataLoader         │  계층 4: 요청 범위 배치 캐시
│   (요청별)          │  - 단일 요청 내 중복 제거
│                      │  - N+1 쿼리 문제 해결
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   데이터베이스/API   │  진실의 원천(Source of truth)
└─────────────────────┘
```

### 캐시 무효화 전략

```typescript
// 전략 1: TTL 기반 (가장 단순)
// @cacheControl에 maxAge를 설정하고 항목이 만료되도록 놔둠

// 전략 2: 뮤테이션 기반 무효화
const resolvers = {
  Mutation: {
    updateProduct: async (_, { id, input }, { cache, dataSources }) => {
      const product = await dataSources.products.update(id, input);

      // 이 상품에 대한 CDN 캐시 무효화
      await fetch(`https://api.fastly.com/purge/product-${id}`, {
        method: 'POST',
        headers: { 'Fastly-Key': process.env.FASTLY_KEY },
      });

      return product;
    },
  },
};

// 전략 3: 이벤트 기반 무효화
// 데이터베이스 변경 이벤트를 수신하여 관련 캐시 항목 제거
import { PubSub } from 'graphql-subscriptions';
const pubsub = new PubSub();

pubsub.subscribe('PRODUCT_UPDATED', ({ productId }) => {
  cache.evict({ id: `Product:${productId}` });
  cache.gc();
});
```

---

## 8. 연습 문제

### 문제 1: APQ 구성
Redis 캐시를 백엔드로 APQ가 활성화된 Apollo Server를 설정하세요. GET 요청으로 APQ를 사용하여 쿼리를 전송하는 클라이언트를 작성하세요. 첫 번째 요청이 전체 쿼리(POST)를 전송하고 이후 요청은 해시만(GET) 사용하는지 확인하세요.

### 문제 2: 캐시 정책 설계
뉴스 사이트를 위한 다음 스키마에 적절한 `@cacheControl` 지시어를 추가하세요. 어떤 데이터가 공개(public)이고 어떤 것이 개인(private)인지, 어떤 필드가 자주 변경되고 어떤 것이 드물게 변경되는지 고려하세요:

```graphql
type Query {
  headlines: [Article!]!
  article(slug: String!): Article
  me: User
  myBookmarks: [Article!]!
}

type Article {
  id: ID!
  title: String!
  body: String!
  author: User!
  publishedAt: DateTime!
  viewCount: Int!
  comments: [Comment!]!
}

type User {
  id: ID!
  name: String!
  email: String!
}
```

### 문제 3: 클라이언트 캐시 병합
무한 스크롤 피드를 위해 페이지네이션된 결과를 올바르게 병합하는 타입 정책(type policies)을 가진 Apollo Client `InMemoryCache`를 구현하세요. `feed` 쿼리는 `after`와 `first` 인수를 사용한 커서 기반 페이지네이션을 사용합니다.

### 문제 4: 캐시 무효화
전자상거래 상품 카탈로그를 위한 캐시 무효화 전략을 설계하세요. 뮤테이션을 통해 상품 가격이 변경될 때 다음 캐시들이 업데이트되어야 합니다: (a) 서버의 응답 캐시, (b) CDN, (c) 구독을 통한 연결된 클라이언트들. 뮤테이션 리졸버와 지원 인프라 코드를 작성하세요.

### 문제 5: 다층 벤치마크
서로 다른 캐싱 시나리오에서 응답 시간을 측정하는 테스트 스크립트를 작성하세요: (a) 캐시 없음, (b) DataLoader만, (c) DataLoader + 응답 캐시, (d) CDN 시뮬레이션이 포함된 전체 스택. 각 계층에서의 지연 시간 감소를 문서화하세요.

---

## 참고 자료

- [Apollo Client 캐싱](https://www.apollographql.com/docs/react/caching/overview)
- [자동 지속 쿼리](https://www.apollographql.com/docs/apollo-server/performance/apq/)
- [Apollo 응답 캐시 플러그인](https://www.apollographql.com/docs/apollo-server/performance/caching/)
- [대규모 GraphQL 캐싱 (Netflix)](https://netflixtechblog.com/)
- [지속 쿼리 모범 사례](https://www.apollographql.com/docs/graphos/operations/persisted-queries)

---

**이전**: [코드 우선 방식(Python)](./10_Code_First_Python.md) | **다음**: [연합(Federation)](./12_Federation.md)
