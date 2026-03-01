# 15. REST에서 GraphQL로 마이그레이션(REST to GraphQL Migration)

**이전**: [성능과 보안](./14_Performance_Security.md) | **다음**: [프로젝트: API 게이트웨이](./16_Project_API_Gateway.md)

## 학습 목표

이 레슨을 마치면 다음을 할 수 있습니다:

1. 세 가지 마이그레이션 전략(래핑, 확장, 교체)을 평가하고 주어진 코드베이스에 적합한 방법을 선택한다
2. 캐싱과 중복 제거를 갖춘 RESTDataSource를 사용하여 기존 REST API를 GraphQL 리졸버로 래핑한다
3. 스키마 스티칭(schema stitching)과 타입 병합(type merging)을 사용하여 여러 GraphQL 스키마를 결합한다
4. graphql-mesh를 사용하여 REST/OpenAPI 사양에서 GraphQL 계층을 자동 생성한다
5. GraphQL을 집계 계층(aggregation layer)으로 사용하는 BFF(Backend for Frontend) 패턴을 구현한다

## 목차

1. [마이그레이션 전략 개요](#1-마이그레이션-전략-개요)
2. [REST API 래핑](#2-rest-api-래핑)
3. [Apollo RESTDataSource](#3-apollo-restdatasource)
4. [스키마 스티칭](#4-스키마-스티칭)
5. [GraphQL Mesh](#5-graphql-mesh)
6. [점진적 마이그레이션](#6-점진적-마이그레이션)
7. [BFF 패턴](#7-bff-패턴)
8. [연습 문제](#8-연습-문제)

**난이도**: ⭐⭐⭐

---

대부분의 팀은 GraphQL로 시작하지 않습니다. 모바일 앱, 웹 프론트엔드, 파트너 통합을 제공하는 REST API를 보유하고 있습니다 — 때로는 수십 개씩이요. 모든 것을 GraphQL로 다시 작성하는 것은 위험하고 불필요합니다. 대신, 가장 성공적인 마이그레이션은 기존 REST 서비스를 GraphQL 계층 뒤에 래핑하고, 엔드포인트를 점진적으로 교체하며, 전환이 완료될 때까지 두 시스템을 병렬로 실행합니다. 이 레슨은 그 과정을 원활하게 만드는 도구와 패턴을 다룹니다.

---

## 1. 마이그레이션 전략 개요

### 세 가지 접근법

```
┌───────────────────────────────────────────────────────────────┐
│                                                                │
│   전략 1: 래핑(WRAP)                                           │
│   ─────────────────                                           │
│   GraphQL ──▶ REST APIs (변경 없음)                           │
│   • 구현이 가장 빠름                                           │
│   • REST가 진실의 원천으로 유지됨                              │
│   • 적합한 경우: "GraphQL 클라이언트는 원하지만 BE를 바꿀 수 없을 때" │
│                                                                │
│   전략 2: 확장(EXTEND)                                        │
│   ──────────────────                                          │
│   GraphQL ──▶ REST + 새 GraphQL 리졸버                        │
│   • REST 엔드포인트를 점진적으로 교체                          │
│   • 새 기능은 GraphQL로 직접 구축                             │
│   • 적합한 경우: 활발한 개발, 점진적 마이그레이션             │
│                                                                │
│   전략 3: 교체(REPLACE)                                       │
│   ────────────────────                                        │
│   GraphQL ──▶ 데이터베이스/서비스 직접                        │
│   • 전체 재작성, REST 의존성 없음                             │
│   • 최고의 노력, 최고의 보상                                  │
│   • 적합한 경우: 그린필드(greenfield) 또는 REST 폐기 시        │
│                                                                │
└───────────────────────────────────────────────────────────────┘
```

### 결정 매트릭스

| 요소 | 래핑 | 확장 | 교체 |
|------|------|------|------|
| 첫 가치까지의 시간 | 일 단위 | 주 단위 | 월 단위 |
| 위험 | 낮음 | 중간 | 높음 |
| REST 의존성 | 영구적 | 일시적 | 없음 |
| 새 기능 속도 | REST와 동일 | 증가 | 최대 |
| 권장 시기 | 읽기 중심, 안정적인 REST | 활발한 개발 | 그린필드 / 기술 부채 |

---

## 2. REST API 래핑

가장 단순한 마이그레이션: REST API 앞에 GraphQL 서버를 놓습니다. 각 리졸버가 REST 엔드포인트를 호출하고 응답을 변환합니다.

### 기본 REST 래퍼

```typescript
import { ApolloServer } from '@apollo/server';
import { startStandaloneServer } from '@apollo/server/standalone';

const typeDefs = `#graphql
  type User {
    id: ID!
    name: String!
    email: String!
    posts: [Post!]!
  }

  type Post {
    id: ID!
    title: String!
    body: String!
    userId: ID!
  }

  type Query {
    user(id: ID!): User
    users: [User!]!
    post(id: ID!): Post
  }
`;

const resolvers = {
  Query: {
    // 각 리졸버가 REST 엔드포인트를 래핑
    user: async (_, { id }) => {
      const res = await fetch(`https://api.example.com/users/${id}`);
      return res.json();
    },
    users: async () => {
      const res = await fetch('https://api.example.com/users');
      return res.json();
    },
    post: async (_, { id }) => {
      const res = await fetch(`https://api.example.com/posts/${id}`);
      return res.json();
    },
  },

  User: {
    // User와 Posts 간의 관계 해석
    posts: async (user) => {
      const res = await fetch(
        `https://api.example.com/users/${user.id}/posts`
      );
      return res.json();
    },
  },
};

const server = new ApolloServer({ typeDefs, resolvers });
startStandaloneServer(server, { listen: { port: 4000 } });
```

### 필드 매핑

REST와 GraphQL은 종종 다른 명명 규칙을 사용합니다. 리졸버에서 매핑하세요:

```typescript
const resolvers = {
  User: {
    // REST는 snake_case, GraphQL은 camelCase 사용
    firstName: (user) => user.first_name,
    lastName: (user) => user.last_name,
    createdAt: (user) => user.created_at,
    avatarUrl: (user) => user.avatar_url || user.profile_image,
  },
};
```

### 오류 매핑

```typescript
const resolvers = {
  Query: {
    user: async (_, { id }) => {
      const res = await fetch(`https://api.example.com/users/${id}`);

      if (res.status === 404) {
        return null; // GraphQL 관례: 찾을 수 없으면 null 반환
      }

      if (res.status === 401) {
        throw new GraphQLError('Authentication required', {
          extensions: { code: 'UNAUTHENTICATED' },
        });
      }

      if (!res.ok) {
        throw new GraphQLError(`REST API error: ${res.statusText}`, {
          extensions: {
            code: 'UPSTREAM_ERROR',
            statusCode: res.status,
          },
        });
      }

      return res.json();
    },
  },
};
```

---

## 3. Apollo RESTDataSource

`@apollo/datasource-rest`는 내장 캐싱, 요청 중복 제거, 오류 처리를 갖춘 REST API 래핑의 구조화된 방법을 제공합니다.

### 설정

```typescript
import { RESTDataSource } from '@apollo/datasource-rest';

class UsersAPI extends RESTDataSource {
  override baseURL = 'https://api.example.com/';

  // GET /users/:id — HTTP 캐시 헤더를 통해 자동으로 캐시됨
  async getUser(id: string) {
    return this.get(`users/${id}`);
  }

  // GET /users
  async getUsers() {
    return this.get('users');
  }

  // GET /users/:id/posts
  async getUserPosts(userId: string) {
    return this.get(`users/${userId}/posts`);
  }

  // POST /users — 뮤테이션은 절대 캐시되지 않음
  async createUser(input: CreateUserInput) {
    return this.post('users', { body: input });
  }

  // PUT /users/:id
  async updateUser(id: string, input: UpdateUserInput) {
    return this.put(`users/${id}`, { body: input });
  }

  // DELETE /users/:id
  async deleteUser(id: string) {
    return this.delete(`users/${id}`);
  }
}

class ProductsAPI extends RESTDataSource {
  override baseURL = 'https://products-api.example.com/v2/';

  async getProduct(id: string) {
    return this.get(`products/${id}`);
  }

  async searchProducts(query: string, page: number = 1) {
    return this.get('products/search', {
      params: { q: query, page: String(page) },
    });
  }
}
```

### 데이터 소스를 컨텍스트에 연결

```typescript
import { ApolloServer } from '@apollo/server';
import { startStandaloneServer } from '@apollo/server/standalone';

interface ContextValue {
  dataSources: {
    usersAPI: UsersAPI;
    productsAPI: ProductsAPI;
  };
}

const server = new ApolloServer<ContextValue>({
  typeDefs,
  resolvers,
});

const { url } = await startStandaloneServer(server, {
  context: async () => {
    return {
      dataSources: {
        usersAPI: new UsersAPI(),
        productsAPI: new ProductsAPI(),
      },
    };
  },
});
```

### 요청 중복 제거

RESTDataSource는 단일 GraphQL 작업 내의 GET 요청을 자동으로 중복 제거합니다:

```graphql
query {
  user1: user(id: "42") { name }
  user2: user(id: "42") { email }
  # 두 별칭 모두 getUser("42")를 호출 → REST API에 HTTP 요청이 단 한 번만 발생
}
```

### 캐싱 구성

```typescript
class UsersAPI extends RESTDataSource {
  override baseURL = 'https://api.example.com/';

  // 요청별 캐싱 동작 재정의
  async getUser(id: string) {
    return this.get(`users/${id}`, {
      cacheOptions: {
        ttl: 300, // REST 헤더와 무관하게 5분간 캐시
      },
    });
  }

  // 컨텍스트에서 인증 헤더 전달
  override willSendRequest(_path: string, request: AugmentedRequest) {
    request.headers['Authorization'] = this.context.token;
  }
}
```

---

## 4. 스키마 스티칭

스키마 스티칭(Schema Stitching)은 여러 GraphQL 스키마를 하나로 결합합니다. 여러 기존 GraphQL 서비스를 단일 게이트웨이를 통해 노출하고 싶을 때 유용합니다.

### @graphql-tools/stitch를 사용한 기본 스티칭

```typescript
import { stitchSchemas } from '@graphql-tools/stitch';
import { schemaFromExecutor } from '@graphql-tools/wrap';
import { buildHTTPExecutor } from '@graphql-tools/executor-http';

// 원격 스키마를 위한 실행기(executor) 생성
const usersExecutor = buildHTTPExecutor({
  endpoint: 'http://users-service:4001/graphql',
});

const productsExecutor = buildHTTPExecutor({
  endpoint: 'http://products-service:4002/graphql',
});

// 원격 스키마 가져오기
const usersSubschema = {
  schema: await schemaFromExecutor(usersExecutor),
  executor: usersExecutor,
};

const productsSubschema = {
  schema: await schemaFromExecutor(productsExecutor),
  executor: productsExecutor,
};

// 스키마 스티칭
const gatewaySchema = stitchSchemas({
  subschemas: [usersSubschema, productsSubschema],
});
```

### 타입 병합(Type Merging)

같은 엔티티가 여러 서비스에 존재할 때, 타입 병합은 이를 통합된 타입으로 결합합니다:

```typescript
const gatewaySchema = stitchSchemas({
  subschemas: [
    {
      schema: usersSchema,
      executor: usersExecutor,
      merge: {
        User: {
          // 이 서비스에서 User를 가져오는 방법
          selectionSet: '{ id }',
          fieldName: 'user',
          args: (originalObject) => ({ id: originalObject.id }),
        },
      },
    },
    {
      schema: reviewsSchema,
      executor: reviewsExecutor,
      merge: {
        User: {
          // Reviews 서비스도 User(reviews 필드 포함)를 해석할 수 있음
          selectionSet: '{ id }',
          fieldName: 'userById',
          args: (originalObject) => ({ id: originalObject.id }),
        },
      },
    },
  ],
});

// 이제 쿼리가 두 서비스에 걸칠 수 있음:
// query { user(id: "1") { name reviews { body } } }
//   name → users 서비스에서
//   reviews → reviews 서비스에서
```

### 스키마 스티칭 vs. Federation

| 기능 | 스키마 스티칭 | Apollo Federation |
|------|------------|------------------|
| 게이트웨이 로직 | 게이트웨이에서 | 라우터에서 (사전 컴파일) |
| 서브그래프 인식 | 서브그래프가 인식하지 못함 | 서브그래프가 federation 지시어 사용 |
| 타입 병합 | 게이트웨이가 병합 구성 | `@key` + `__resolveReference` |
| 벤더 종속 | 없음 (@graphql-tools) | Apollo 생태계 |
| 성능 | 단순한 그래프에 양호 | 복잡한 그래프에 더 우수 |
| 사용 사례 | 소수의 서비스, 빠른 설정 | 다수의 서비스, 팀 소유권 |

---

## 5. GraphQL Mesh

GraphQL Mesh는 GraphQL이 아닌 소스(REST/OpenAPI, gRPC, 데이터베이스 등)에서 GraphQL 스키마를 자동 생성합니다.

### OpenAPI를 사용한 설정

```yaml
# .meshrc.yaml
sources:
  - name: PetStore
    handler:
      openapi:
        source: https://petstore.swagger.io/v2/swagger.json
        # 또는 로컬 파일:
        # source: ./openapi-spec.yaml

  - name: WeatherAPI
    handler:
      openapi:
        source: ./weather-openapi.yaml
        operationHeaders:
          'x-api-key': '{env.WEATHER_API_KEY}'

serve:
  port: 4000
```

```bash
# 의존성 설치
npm install @graphql-mesh/cli @graphql-mesh/openapi

# 메시 게이트웨이 시작
npx mesh dev
```

이는 OpenAPI 사양에서 GraphQL 타입과 리졸버를 자동으로 생성합니다:

```graphql
# PetStore OpenAPI 사양에서 자동 생성됨
type Query {
  findPetsByStatus(status: [Status!]): [Pet!]
  getPetById(petId: Int!): Pet
  getInventory: JSON
}

type Mutation {
  addPet(input: PetInput!): Pet
  updatePet(input: PetInput!): Pet
  deletePet(petId: Int!): JSON
}

type Pet {
  id: Int
  name: String!
  category: Category
  photoUrls: [String!]!
  tags: [Tag!]
  status: Status
}
```

### 변환(Transforms)

GraphQL Mesh는 생성된 스키마를 커스터마이징하기 위한 변환을 지원합니다:

```yaml
# .meshrc.yaml
sources:
  - name: PetStore
    handler:
      openapi:
        source: https://petstore.swagger.io/v2/swagger.json
    transforms:
      # 타입 이름 변경
      - rename:
          renames:
            - from:
                type: Pet
              to:
                type: Animal

      # 노출하고 싶지 않은 작업 필터링
      - filterSchema:
          filters:
            - Query.!getInventory   # getInventory 제외
            - Mutation.!deletePet   # deletePet 제외

      # 이름 충돌 방지를 위한 접두사 추가
      - prefix:
          value: PetStore_
          includeRootOperations: true
```

### 여러 소스 결합

```yaml
# .meshrc.yaml
sources:
  - name: Users
    handler:
      openapi:
        source: ./users-api.yaml

  - name: Products
    handler:
      graphql:
        endpoint: http://products-service:4002/graphql

  - name: Orders
    handler:
      grpc:
        endpoint: orders-service:50051
        protoFilePath: ./orders.proto

# 소스 간 관계 정의
additionalTypeDefs: |
  extend type User {
    orders: [Order!]!
  }

additionalResolvers:
  - targetTypeName: User
    targetFieldName: orders
    requiredSelectionSet: '{ id }'
    sourceName: Orders
    sourceTypeName: Query
    sourceFieldName: ordersByUserId
    sourceArgs:
      userId: '{root.id}'
```

---

## 6. 점진적 마이그레이션

가장 안전한 마이그레이션 경로는 REST와 GraphQL을 병렬로 실행하면서 한 번에 하나의 엔드포인트를 마이그레이션합니다.

### 단계별 마이그레이션

```
단계 1: GraphQL 게이트웨이 (래핑)
─────────────────────────────────
┌──────────┐     ┌───────────────┐     ┌──────────────┐
│ 클라이언트 │ ──▶ │  GraphQL GW   │ ──▶ │  REST APIs   │
│  (신규)   │     │  (REST 래핑)  │     │  (변경 없음) │
└──────────┘     └───────────────┘     └──────────────┘
┌──────────┐                           ┌──────────────┐
│ 클라이언트 │ ─────────────────────── ▶ │  REST APIs   │
│  (기존)   │                           │  (변경 없음) │
└──────────┘                           └──────────────┘

단계 2: 점진적 교체
─────────────────────────────────
┌──────────┐     ┌───────────────┐     ┌──────────────┐
│ 클라이언트 │ ──▶ │  GraphQL GW   │ ──▶ │  REST APIs   │
│  (전체)   │     │               │     │  (축소 중)   │
└──────────┘     │ 새 리졸버     │     └──────────────┘
                 │  (증가 중)    │──▶  데이터베이스
                 └───────────────┘

단계 3: REST 폐기
─────────────────────────────────
┌──────────┐     ┌───────────────┐
│ 클라이언트 │ ──▶ │  GraphQL      │──▶  데이터베이스
│  (전체)   │     │  서버         │
└──────────┘     └───────────────┘
```

### 라우터 수준 마이그레이션

리버스 프록시를 사용하여 REST에서 GraphQL로 트래픽을 점진적으로 라우팅합니다:

```nginx
# nginx.conf — 엔드포인트별 라우팅
server {
    listen 80;

    # 마이그레이션된 엔드포인트는 GraphQL로
    location /api/users {
        proxy_pass http://graphql-gateway:4000/graphql;
        # REST 호출을 GraphQL 쿼리로 재작성 (미들웨어를 통해)
    }

    # 아직 마이그레이션되지 않은 엔드포인트는 REST에 유지
    location /api/ {
        proxy_pass http://rest-api:3000;
    }
}
```

### REST-to-GraphQL 프록시 미들웨어

```typescript
// REST 요청을 GraphQL 쿼리로 변환하는 미들웨어
import express from 'express';
import { graphqlHTTP } from 'express-graphql';

const app = express();

// GET /api/users/:id → GraphQL 쿼리로 변환
app.get('/api/users/:id', async (req, res) => {
  const query = `
    query GetUser($id: ID!) {
      user(id: $id) {
        id
        name
        email
        createdAt
      }
    }
  `;

  const response = await fetch('http://localhost:4000/graphql', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: req.headers.authorization,
    },
    body: JSON.stringify({
      query,
      variables: { id: req.params.id },
    }),
  });

  const { data, errors } = await response.json();

  if (errors) {
    return res.status(400).json({ errors });
  }

  // REST 스타일 응답으로 다시 변환
  if (!data.user) {
    return res.status(404).json({ error: 'User not found' });
  }

  res.json(data.user);
});
```

---

## 7. BFF 패턴

BFF(Backend for Frontend) 패턴은 GraphQL을 프론트엔드와 여러 백엔드 서비스 사이의 집계 계층으로 사용합니다.

### 아키텍처

```
┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│   웹 앱     │  │  모바일 앱  │  │  관리자 앱  │
└──────┬──────┘  └──────┬──────┘  └──────┬──────┘
       │                │                │
       ▼                ▼                ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│   웹 BFF    │ │  모바일 BFF │ │  관리자 BFF │
│  (GraphQL)  │ │  (GraphQL)  │ │  (GraphQL)  │
│              │ │              │ │              │
│  웹에 최적화 │ │ 모바일 최적화│ │  전체 데이터 │
│              │ │              │ │    접근     │
└──────┬───────┘ └──────┬───────┘ └──────┬───────┘
       │                │                │
       └────────────────┼────────────────┘
                        │
              ┌─────────┼─────────┐
              ▼         ▼         ▼
         ┌────────┐ ┌────────┐ ┌────────┐
         │ Users  │ │Products│ │ Orders │
         │  API   │ │  API   │ │  API   │
         └────────┘ └────────┘ └────────┘
```

### GraphQL과 BFF를 사용하는 이유

1. **클라이언트별 최적화**: 모바일은 최소 필드, 웹은 전체 객체 수신
2. **API 집계**: 하나의 GraphQL 쿼리가 3-4개의 REST 호출을 대체
3. **타입 안전성**: 스키마가 프론트엔드와 백엔드 간의 계약 역할
4. **분리**: 프론트엔드 변경이 백엔드 변경을 필요로 하지 않음

### 구현

```typescript
// 모바일 BFF — 대역폭 최적화
const mobileTypeDefs = `#graphql
  type Query {
    feed(first: Int = 20): FeedConnection!
    profile: UserProfile!
  }

  # 모바일을 위한 최소 타입
  type UserProfile {
    id: ID!
    name: String!
    avatarUrl: String
    unreadCount: Int!
  }

  type FeedItem {
    id: ID!
    title: String!
    thumbnailUrl: String
    # 본문 없음 — 모바일은 탭할 때 로드
  }
`;

const mobileResolvers = {
  Query: {
    feed: async (_, { first }, { dataSources }) => {
      // 여러 서비스에서 집계
      const posts = await dataSources.postsAPI.getRecent(first);
      const enriched = await Promise.all(
        posts.map(async (post) => ({
          ...post,
          thumbnailUrl: await dataSources.mediaAPI.getThumbnail(post.imageId),
        }))
      );
      return enriched;
    },
    profile: async (_, __, { dataSources, userId }) => {
      // 여러 서비스에 병렬 요청
      const [user, notifications] = await Promise.all([
        dataSources.usersAPI.getUser(userId),
        dataSources.notificationsAPI.getUnreadCount(userId),
      ]);
      return { ...user, unreadCount: notifications.count };
    },
  },
};
```

### 클라이언트 인식 리졸버를 가진 단일 BFF

별도의 BFF 대신, 클라이언트 인식 동작을 가진 단일 GraphQL 서버를 사용합니다:

```typescript
const resolvers = {
  Query: {
    products: async (_, args, { dataSources, clientType }) => {
      const products = await dataSources.productsAPI.list(args);

      // 모바일은 압축된 이미지, 웹은 전체 해상도
      if (clientType === 'mobile') {
        return products.map((p) => ({
          ...p,
          imageUrl: p.imageUrl.replace('/full/', '/thumb/'),
        }));
      }

      return products;
    },
  },
};

// 헤더에서 클라이언트 유형 감지
const context = async ({ req }) => ({
  clientType: req.headers['x-client-type'] || 'web',
  dataSources: { /* ... */ },
});
```

---

## 8. 연습 문제

### 문제 1: REST 래퍼
다음 REST API 엔드포인트들이 주어졌을 때, GraphQL 스키마를 설계하고 각 엔드포인트를 래핑하는 리졸버를 작성하세요. 필드 이름 매핑(snake_case에서 camelCase로)과 오류 응답을 처리하세요:

```
GET    /api/v1/users                → 모든 사용자 목록
GET    /api/v1/users/:id            → ID로 사용자 조회
POST   /api/v1/users                → 사용자 생성
GET    /api/v1/users/:id/orders     → 사용자의 주문 조회
GET    /api/v1/products?category=X  → 상품 검색
```

### 문제 2: RESTDataSource
상품 카탈로그 REST API를 래핑하는 RESTDataSource를 확장하는 `ProductsAPI` 클래스를 구현하세요. 포함 사항: (a) 5분 TTL의 응답 캐싱, (b) 요청 중복 제거, (c) 인증을 위한 헤더 전달, (d) HTTP 상태 코드를 GraphQL 오류 코드로 매핑하는 오류 처리.

### 문제 3: GraphQL Mesh 구성
다음을 결합하는 `.meshrc.yaml` 구성을 작성하세요: (a) PetStore OpenAPI 사양, (b) 날씨 REST API, (c) 기존 GraphQL 인벤토리 서비스. 타입 이름 변경, 작업 필터링, 소스 간 관계(pets는 위치 기반 `weather` 필드를 가져야 함) 생성을 위한 변환을 추가하세요.

### 문제 4: 마이그레이션 계획
3개의 프론트엔드 애플리케이션(웹, iOS, Android)을 제공하는 25개 엔드포인트를 가진 REST API가 있습니다. 점진적 마이그레이션 계획을 설계하세요: (a) 먼저 마이그레이션할 엔드포인트 식별(최고 가치, 최저 위험), (b) GraphQL 게이트웨이를 REST 래퍼로 설정, (c) 각 단계별 성공 기준이 있는 4단계 마이그레이션 정의.

### 문제 5: BFF 설계
세 가지 클라이언트를 가진 스트리밍 플랫폼을 위한 BFF 아키텍처를 설계하세요: 웹(전체 기능), 모바일(오프라인 가능), 스마트 TV(최소 UI). "카탈로그 탐색" 기능에 대해 동일한 백엔드 데이터가 클라이언트별로 다른 GraphQL 스키마와 리졸버 로직을 어떻게 갖는지 보여주세요. 각기 다른 응답 형태를 보여주는 쿼리 예시를 포함하세요.

---

## 참고 자료

- [Apollo RESTDataSource](https://www.apollographql.com/docs/apollo-server/data/fetching-rest/)
- [GraphQL Mesh 문서](https://the-guild.dev/graphql/mesh/docs)
- [@graphql-tools를 사용한 스키마 스티칭](https://the-guild.dev/graphql/stitching)
- [REST에서 GraphQL로 마이그레이션 (Apollo 가이드)](https://www.apollographql.com/docs/technotes/TN0032-sdui-overview/)
- [Backend for Frontend 패턴](https://samnewman.io/patterns/architectural/bff/)
- [Netflix GraphQL Federation](https://netflixtechblog.com/how-netflix-scales-its-api-with-graphql-federation-part-1-ae3557c187e2)

---

**이전**: [성능과 보안](./14_Performance_Security.md) | **다음**: [프로젝트: API 게이트웨이](./16_Project_API_Gateway.md)
