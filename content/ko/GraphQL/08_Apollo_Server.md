# 08. Apollo Server

**이전**: [인증과 인가](./07_Authentication_Authorization.md) | **다음**: [GraphQL 클라이언트](./09_GraphQL_Clients.md)

---

Apollo Server는 JavaScript/TypeScript 생태계에서 가장 널리 채택된 GraphQL 서버입니다. 버전 4 — 완전히 새로 작성된 — 는 프레임워크에 독립적입니다: 단독(standalone)으로 실행하거나, Express 미들웨어로, Fastify 내에서, 또는 서버리스 함수 안에서도 실행할 수 있습니다. 이 레슨에서는 Apollo Server 4를 초기 설정부터 프로덕션 준비 구성까지 살펴보며, 스키마 우선(schema-first)과 코드 우선(code-first) 접근 방식, 플러그인 시스템, Apollo Studio 통합을 다룹니다.

**난이도**: ⭐⭐⭐

## 학습 목표

이 레슨을 완료하면 다음을 수행할 수 있습니다:

1. 단독 모드와 Express/Fastify 미들웨어로 Apollo Server 4를 설정할 수 있다
2. 스키마 우선과 코드 우선 개발 접근 방식을 비교하고 적절히 선택할 수 있다
3. 데이터베이스, 인증, DataLoader를 주입하도록 컨텍스트 함수를 구성할 수 있다
4. 로깅, 캐싱, 오류 형식화를 위한 Apollo Server 플러그인을 사용할 수 있다
5. CSRF 방지, 인트로스펙션(introspection) 제어, 모니터링을 갖춘 프로덕션 준비 Apollo Server를 준비할 수 있다

---

## 목차

1. [Apollo Server 4 아키텍처](#1-apollo-server-4-아키텍처)
2. [단독 설정](#2-단독-설정)
3. [Express 미들웨어 설정](#3-express-미들웨어-설정)
4. [Fastify 통합](#4-fastify-통합)
5. [스키마 우선 vs 코드 우선](#5-스키마-우선-vs-코드-우선)
6. [컨텍스트 함수](#6-컨텍스트-함수)
7. [플러그인](#7-플러그인)
8. [오류 처리와 형식화](#8-오류-처리와-형식화)
9. [Apollo Studio 통합](#9-apollo-studio-통합)
10. [프로덕션 구성](#10-프로덕션-구성)
11. [연습 문제](#11-연습-문제)
12. [참고 자료](#12-참고-자료)

---

## 1. Apollo Server 4 아키텍처

Apollo Server 4는 GraphQL 실행 엔진과 HTTP 레이어를 분리합니다. 핵심 `ApolloServer` 클래스는 스키마 검증, 쿼리 파싱, 실행을 처리합니다. HTTP 통합(Express, Fastify, 단독)은 요청/응답 배관(plumbing)을 처리합니다.

```
┌─────────────────────────────────────────────────┐
│                HTTP 프레임워크                    │
│  (Express / Fastify / Standalone / Lambda)       │
├─────────────────────────────────────────────────┤
│           expressMiddleware / startStandalone     │
│           (HTTP 통합 레이어)                       │
├─────────────────────────────────────────────────┤
│              ApolloServer 코어                    │
│  ┌───────────┬──────────┬──────────────────┐    │
│  │  스키마    │ 플러그인  │  요청 파이프라인   │    │
│  │  파싱      │          │  파싱 → 검증       │    │
│  │           │          │  → 실행            │    │
│  └───────────┴──────────┴──────────────────┘    │
├─────────────────────────────────────────────────┤
│              리졸버 + 데이터 소스                  │
└─────────────────────────────────────────────────┘
```

Apollo Server 3에서의 주요 변경 사항:

- `apollo-server-express`, `apollo-server-koa` 등이 없어짐 — 모든 통합에 하나의 패키지(`@apollo/server`) 사용
- 프레임워크 통합이 명시적 (직접 `expressMiddleware()`를 호출하며, `server.applyMiddleware()`가 아님)
- TypeScript 우선 API 설계
- 내장 `ApolloServerPluginDrainHttpServer`가 명시적 드레인(drain) 패턴으로 교체됨

---

## 2. 단독 설정

Apollo Server를 실행하는 가장 간단한 방법 — Express 없음, 프레임워크 없음, GraphQL만 있음.

```bash
npm install @apollo/server graphql
```

```typescript
// src/index.ts
import { ApolloServer } from '@apollo/server';
import { startStandaloneServer } from '@apollo/server/standalone';

// Type definitions (SDL)
const typeDefs = `#graphql
  type Book {
    id: ID!
    title: String!
    author: String!
    year: Int
  }

  type Query {
    books: [Book!]!
    book(id: ID!): Book
  }

  type Mutation {
    addBook(title: String!, author: String!, year: Int): Book!
  }
`;

// In-memory data store (replace with database in production)
let books = [
  { id: '1', title: 'The GraphQL Guide', author: 'John Resig', year: 2022 },
  { id: '2', title: 'Learning GraphQL', author: "Eve Porcello", year: 2018 },
];

const resolvers = {
  Query: {
    books: () => books,
    book: (_, { id }) => books.find((b) => b.id === id),
  },
  Mutation: {
    addBook: (_, { title, author, year }) => {
      const book = {
        id: String(books.length + 1),
        title,
        author,
        year,
      };
      books.push(book);
      return book;
    },
  },
};

const server = new ApolloServer({ typeDefs, resolvers });

const { url } = await startStandaloneServer(server, {
  listen: { port: 4000 },
  context: async ({ req }) => ({
    // Context is available in every resolver
    token: req.headers.authorization,
  }),
});

console.log(`Server ready at ${url}`);
```

`startStandaloneServer`는 내부적으로 Express 앱을 생성합니다. 프로토타이핑에 편리하지만 제한적입니다 — 커스텀 미들웨어나 라우트를 추가할 수 없습니다.

---

## 3. Express 미들웨어 설정

실제 애플리케이션에서는 일반적으로 추가 라우트(헬스 체크, 웹훅), 미들웨어(CORS, 바디 파싱, 속도 제한), 또는 WebSocket 지원(구독)을 위해 Express가 필요합니다.

```bash
npm install @apollo/server express graphql cors
npm install -D @types/cors @types/express
```

```typescript
// src/index.ts
import { ApolloServer } from '@apollo/server';
import { expressMiddleware } from '@apollo/server/express4';
import {
  ApolloServerPluginDrainHttpServer,
} from '@apollo/server/plugin/drainHttpServer';
import express from 'express';
import http from 'http';
import cors from 'cors';
import { typeDefs, resolvers } from './schema';
import { createContext } from './context';

async function startServer() {
  const app = express();
  const httpServer = http.createServer(app);

  const server = new ApolloServer({
    typeDefs,
    resolvers,
    plugins: [
      // Ensures Apollo Server shuts down gracefully alongside
      // the Express HTTP server — no dangling connections.
      ApolloServerPluginDrainHttpServer({ httpServer }),
    ],
  });

  // Must call start() before applying middleware
  await server.start();

  // Health check endpoint (useful for load balancers)
  app.get('/health', (_, res) => res.json({ status: 'ok' }));

  // GraphQL endpoint
  app.use(
    '/graphql',
    cors<cors.CorsRequest>({
      origin: ['http://localhost:3000', 'https://myapp.com'],
    }),
    express.json(),
    expressMiddleware(server, {
      context: createContext,
    })
  );

  httpServer.listen(4000, () => {
    console.log('Server ready at http://localhost:4000/graphql');
  });
}

startServer();
```

### `ApolloServerPluginDrainHttpServer`가 필요한 이유?

이 플러그인 없이 `server.stop()`을 호출하면 Apollo는 새로운 요청 수락을 중지하지만 기존 HTTP 연결을 닫지 않습니다. 이 플러그인은 Apollo의 생명주기를 HTTP 서버의 생명주기와 연결합니다: Apollo가 중지되면 HTTP 서버에 새 연결 수락을 중지하도록 신호를 보내고 처리 중인 요청이 완료될 때까지 기다립니다(드레인).

---

## 4. Fastify 통합

Fastify는 Express의 고성능 대안입니다. Apollo Server 4는 `@as-integrations/fastify` 패키지를 통해 지원합니다.

```bash
npm install @apollo/server graphql fastify @as-integrations/fastify
```

```typescript
// src/index.ts
import Fastify from 'fastify';
import { ApolloServer } from '@apollo/server';
import fastifyApollo, {
  fastifyApolloDrainPlugin,
} from '@as-integrations/fastify';
import { typeDefs, resolvers } from './schema';

async function startServer() {
  const fastify = Fastify({ logger: true });

  const server = new ApolloServer({
    typeDefs,
    resolvers,
    plugins: [fastifyApolloDrainPlugin(fastify)],
  });

  await server.start();

  // Register Apollo as a Fastify plugin
  await fastify.register(fastifyApollo(server), {
    context: async (request) => ({
      token: request.headers.authorization,
    }),
  });

  await fastify.listen({ port: 4000 });
  console.log('Fastify + Apollo Server running on port 4000');
}

startServer();
```

Fastify의 스키마 기반 유효성 검사와 직렬화는 JSON 중심 워크로드에서 Express보다 빠릅니다. 순수 처리량이 중요할 때 Fastify를 선택하세요.

---

## 5. 스키마 우선 vs 코드 우선

GraphQL 스키마를 정의하는 두 가지 근본적으로 다른 접근 방식이 있습니다.

### 스키마 우선 (SDL 우선)

GraphQL SDL(Schema Definition Language)로 스키마를 작성한 다음 스키마에 맞는 리졸버를 구현합니다.

```typescript
// Schema (SDL)
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
    content: String!
    author: User!
  }

  type Query {
    user(id: ID!): User
    posts(limit: Int = 10): [Post!]!
  }
`;

// Resolvers (must match the schema exactly)
const resolvers = {
  Query: {
    user: (_, { id }, { db }) => db.user.findUnique({ where: { id } }),
    posts: (_, { limit }, { db }) => db.post.findMany({ take: limit }),
  },
  User: {
    posts: (parent, _, { db }) =>
      db.post.findMany({ where: { authorId: parent.id } }),
  },
  Post: {
    author: (parent, _, { db }) =>
      db.user.findUnique({ where: { id: parent.authorId } }),
  },
};
```

**장점**: 비개발자도 스키마를 읽을 수 있고, PR에서 검토하기 쉬우며, 명확한 계약.
**단점**: codegen 없이는 스키마와 리졸버 간의 타입 안전성 없이 분리(drift)될 수 있음.

### 코드 우선

라이브러리를 사용하여 프로그래밍 방식으로 스키마를 정의합니다. SDL은 자동으로 생성됩니다.

```typescript
// Using Nexus (code-first library for Node.js)
import { makeSchema, objectType, queryType, intArg, idArg } from 'nexus';

const User = objectType({
  name: 'User',
  definition(t) {
    t.nonNull.id('id');
    t.nonNull.string('name');
    t.nonNull.string('email');
    t.nonNull.list.nonNull.field('posts', {
      type: 'Post',
      resolve: (parent, _, { db }) =>
        db.post.findMany({ where: { authorId: parent.id } }),
    });
  },
});

const Post = objectType({
  name: 'Post',
  definition(t) {
    t.nonNull.id('id');
    t.nonNull.string('title');
    t.nonNull.string('content');
    t.nonNull.field('author', {
      type: 'User',
      resolve: (parent, _, { db }) =>
        db.user.findUnique({ where: { id: parent.authorId } }),
    });
  },
});

const Query = queryType({
  definition(t) {
    t.field('user', {
      type: 'User',
      args: { id: idArg() },
      resolve: (_, { id }, { db }) => db.user.findUnique({ where: { id } }),
    });
    t.nonNull.list.nonNull.field('posts', {
      type: 'Post',
      args: { limit: intArg({ default: 10 }) },
      resolve: (_, { limit }, { db }) => db.post.findMany({ take: limit }),
    });
  },
});

export const schema = makeSchema({
  types: [User, Post, Query],
  outputs: {
    schema: './generated/schema.graphql',  // Auto-generated SDL
    typegen: './generated/nexus-typegen.ts', // Auto-generated TS types
  },
});
```

**장점**: 완전한 타입 안전성, 스키마와 리졸버가 분리될 수 없음, IDE 자동 완성.
**단점**: 전체적으로 스키마를 읽기 어려움; 더 가파른 학습 곡선.

### 비교

| 측면 | 스키마 우선 | 코드 우선 |
|------|-----------|---------|
| 가독성 | SDL이 사람이 읽기 좋음 | 코드가 장황할 수 있음 |
| 타입 안전성 | codegen 필요 (예: GraphQL Code Generator) | 내장 |
| 팀 워크플로 | 프론트엔드/백엔드가 SDL 먼저 합의 | 주로 백엔드 주도 |
| 도구 | graphql-codegen, graphql-tools | Nexus, TypeGraphQL, Pothos |
| 스키마 분리 | 가능 (스키마가 한 가지를 말하고 리졸버가 다른 것을 함) | 불가능 |
| 학습 곡선 | 낮음 (SDL이 단순) | 높음 (라이브러리별 API) |

**권장 사항**: 프론트엔드와 백엔드가 API 계약에 협력하는 팀의 경우, GraphQL Code Generator를 사용한 스키마 우선이 잘 작동합니다. 최대 타입 안전성을 원하는 백엔드 중심 팀의 경우 코드 우선이 바람직합니다. 둘 다 프로덕션에서 검증된 방식입니다.

---

## 6. 컨텍스트 함수

컨텍스트 함수는 HTTP 요청과 리졸버 사이의 브릿지입니다. 요청당 한 번 실행되며, 해당 요청의 모든 리졸버에서 공유하는 객체를 생성합니다.

```typescript
// src/context.ts
import { PrismaClient } from '@prisma/client';
import { verifyToken } from './auth';
import { createLoaders } from './dataloaders';

// Initialize Prisma once (connection pooling)
const prisma = new PrismaClient();

export interface Context {
  db: PrismaClient;
  currentUser: User | null;
  loaders: ReturnType<typeof createLoaders>;
  requestId: string;
}

export async function createContext({ req }): Promise<Context> {
  // 1. Authentication: extract user from token
  const token = req.headers.authorization?.replace('Bearer ', '');
  const currentUser = token ? await verifyToken(token, prisma) : null;

  // 2. DataLoaders: create per-request instances.
  //    DataLoaders cache results within a single request.
  //    Sharing them across requests would leak data between users.
  const loaders = createLoaders(prisma);

  // 3. Request ID for logging and tracing
  const requestId =
    (req.headers['x-request-id'] as string) || crypto.randomUUID();

  return {
    db: prisma,
    currentUser,
    loaders,
    requestId,
  };
}
```

### 컨텍스트에 포함해야 할 것

| 포함 | 제외 |
|------|------|
| 데이터베이스 클라이언트 | 요청 본문 (리졸버가 args를 받음) |
| 인증된 사용자 | 응답 객체 (리졸버가 데이터를 반환) |
| DataLoader (요청당) | 비즈니스 로직 (서비스에 속함) |
| 요청 ID | 대규모 캐시된 데이터셋 |
| 로거 인스턴스 | |

### DataLoader가 요청당 생성되어야 하는 이유

DataLoader는 N+1 문제를 해결하기 위해 단일 요청 내에서 키별로 캐시합니다. DataLoader를 여러 요청에서 공유하면 사용자 A의 요청이 사용자 B의 캐시된 데이터를 반환할 수 있습니다. 항상 컨텍스트 함수에서 새로운 DataLoader를 생성하세요.

---

## 7. 플러그인

Apollo Server의 플러그인 시스템을 사용하면 리졸버를 수정하지 않고도 요청 생명주기의 모든 단계에 연결할 수 있습니다.

### 생명주기 훅

```
serverWillStart
  └── requestDidStart
       ├── didResolveSource
       ├── parsingDidStart
       ├── validationDidStart
       ├── didResolveOperation
       ├── responseForOperation
       ├── executionDidStart
       │    └── willResolveField (필드마다)
       ├── willSendResponse
       └── didEncounterErrors
```

### 로깅 플러그인

```typescript
// src/plugins/logging.ts
import { ApolloServerPlugin } from '@apollo/server';

export const loggingPlugin: ApolloServerPlugin = {
  async requestDidStart({ request, contextValue }) {
    const start = Date.now();
    const { operationName } = request;

    console.log(`[${contextValue.requestId}] ${operationName || 'anonymous'} started`);

    return {
      async willSendResponse({ response }) {
        const duration = Date.now() - start;
        console.log(
          `[${contextValue.requestId}] ${operationName || 'anonymous'} ` +
          `completed in ${duration}ms`
        );
      },

      async didEncounterErrors({ errors }) {
        for (const error of errors) {
          console.error(
            `[${contextValue.requestId}] Error: ${error.message}`,
            error.extensions
          );
        }
      },
    };
  },
};
```

### 응답 캐시 플러그인

Apollo는 내장 전체 응답 캐시 플러그인을 제공합니다. 쿼리 + 변수를 키로 사용하여 전체 JSON 응답을 캐싱합니다.

```bash
npm install @apollo/server-plugin-response-cache
```

```typescript
import responseCachePlugin from '@apollo/server-plugin-response-cache';

const server = new ApolloServer({
  typeDefs,
  resolvers,
  plugins: [
    responseCachePlugin({
      // Custom session ID for per-user caching
      sessionId: async ({ request }) =>
        request.http?.headers.get('authorization') || null,
    }),
  ],
});
```

캐시를 사용하려면 스키마에 `@cacheControl`을 주석으로 달아야 합니다:

```graphql
type Query {
  posts: [Post!]! @cacheControl(maxAge: 60)       # Cache 60 seconds
  currentUser: User @cacheControl(maxAge: 0)        # Never cache
}

type Post @cacheControl(maxAge: 120) {
  id: ID!
  title: String!
  viewCount: Int! @cacheControl(maxAge: 10)          # Field-level override
}
```

### 커스텀 플러그인: 쿼리 복잡도

```typescript
// src/plugins/complexity.ts
import { ApolloServerPlugin } from '@apollo/server';

const MAX_DEPTH = 10;

export const depthLimitPlugin: ApolloServerPlugin = {
  async requestDidStart() {
    return {
      async didResolveOperation({ document }) {
        const depth = calculateDepth(document);
        if (depth > MAX_DEPTH) {
          throw new GraphQLError(
            `Query depth ${depth} exceeds maximum of ${MAX_DEPTH}`,
            { extensions: { code: 'QUERY_TOO_DEEP' } }
          );
        }
      },
    };
  },
};

function calculateDepth(document: DocumentNode): number {
  // Recursive traversal of the AST to find max nesting depth
  let maxDepth = 0;

  function traverse(selections: readonly SelectionNode[], depth: number) {
    for (const selection of selections) {
      if (selection.kind === 'Field' && selection.selectionSet) {
        traverse(selection.selectionSet.selections, depth + 1);
      }
      if (selection.kind === 'InlineFragment' && selection.selectionSet) {
        traverse(selection.selectionSet.selections, depth);
      }
    }
    maxDepth = Math.max(maxDepth, depth);
  }

  for (const def of document.definitions) {
    if (def.kind === 'OperationDefinition' && def.selectionSet) {
      traverse(def.selectionSet.selections, 1);
    }
  }

  return maxDepth;
}
```

---

## 8. 오류 처리와 형식화

Apollo Server는 모든 오류를 표준 형식으로 래핑합니다. `formatError` 옵션을 사용하여 클라이언트에 오류가 표시되는 방식을 커스터마이징할 수 있습니다.

```typescript
import { ApolloServer } from '@apollo/server';
import { GraphQLError, GraphQLFormattedError } from 'graphql';

const server = new ApolloServer({
  typeDefs,
  resolvers,
  formatError: (
    formattedError: GraphQLFormattedError,
    error: unknown
  ) => {
    // Log the full error server-side
    console.error('GraphQL Error:', error);

    // In production, hide internal error details from clients
    if (process.env.NODE_ENV === 'production') {
      // Known errors: pass through with their code
      if (
        formattedError.extensions?.code === 'UNAUTHENTICATED' ||
        formattedError.extensions?.code === 'FORBIDDEN' ||
        formattedError.extensions?.code === 'BAD_USER_INPUT'
      ) {
        return formattedError;
      }

      // Unknown errors: mask the message
      return {
        message: 'Internal server error',
        extensions: { code: 'INTERNAL_SERVER_ERROR' },
      };
    }

    // In development, return full details
    return formattedError;
  },
});
```

### 구조화된 오류 던지기

```typescript
import { GraphQLError } from 'graphql';

// Validation error
throw new GraphQLError('Email address is invalid', {
  extensions: {
    code: 'BAD_USER_INPUT',
    field: 'email',
    value: input.email,
  },
});

// Not found
throw new GraphQLError('Post not found', {
  extensions: {
    code: 'NOT_FOUND',
    resourceType: 'Post',
    resourceId: id,
  },
});

// Rate limited
throw new GraphQLError('Too many requests', {
  extensions: {
    code: 'RATE_LIMITED',
    retryAfter: 30,
  },
});
```

---

## 9. Apollo Studio 통합

Apollo Studio(이전 Apollo Engine)는 스키마 관리, 메트릭, 작업 추적을 위한 클라우드 플랫폼을 제공합니다.

### Apollo Studio 연결

```bash
# Set your Apollo API key
export APOLLO_KEY=service:my-graph:abc123
export APOLLO_GRAPH_REF=my-graph@production
```

```typescript
const server = new ApolloServer({
  typeDefs,
  resolvers,
  plugins: [
    // Usage reporting sends anonymous operation metrics to Apollo Studio
    ApolloServerPluginUsageReporting({
      sendVariableValues: { none: true }, // Never send variable values
      sendHeaders: { none: true },         // Never send headers
    }),

    // Schema reporting automatically publishes your schema on server start
    ApolloServerPluginSchemaReporting(),
  ],
});
```

### Apollo Studio가 제공하는 것

| 기능 | 설명 |
|------|------|
| **스키마 레지스트리** | 스키마의 버전 이력; 배포 간 변경 사항 비교 |
| **작업 메트릭** | 요청 속도, 지연시간 (p50/p95/p99), 작업별 오류율 |
| **필드 사용량** | 클라이언트가 실제로 사용하는 필드 — 안전한 deprecation 인사이트 |
| **스키마 검사** | 실제 트래픽에 대해 스키마 변경을 검증하는 CI 통합 |
| **Explorer** | 스키마와 실제 데이터가 있는 브라우저 내 GraphQL IDE |

Apollo Studio는 소규모 팀(월 최대 1,000만 작업)에 무료이며 가치 있는 관측 가능성을 제공합니다. Apollo Federation을 사용하지 않더라도 메트릭과 스키마 레지스트리를 설정할 가치가 있습니다.

---

## 10. 프로덕션 구성

프로덕션 Apollo Server는 개발에서 사용하는 것 이외에 여러 추가 구성이 필요합니다.

### 완전한 프로덕션 설정

```typescript
// src/index.ts
import { ApolloServer } from '@apollo/server';
import { expressMiddleware } from '@apollo/server/express4';
import {
  ApolloServerPluginDrainHttpServer,
} from '@apollo/server/plugin/drainHttpServer';
import {
  ApolloServerPluginLandingPageDisabled,
} from '@apollo/server/plugin/disabled';
import {
  ApolloServerPluginLandingPageLocalDefault,
} from '@apollo/server/plugin/landingPage/default';
import express from 'express';
import http from 'http';
import cors from 'cors';
import helmet from 'helmet';

const IS_PRODUCTION = process.env.NODE_ENV === 'production';

const app = express();
const httpServer = http.createServer(app);

const server = new ApolloServer({
  typeDefs,
  resolvers,

  // Disable introspection in production to hide your schema
  // from unauthorized users. Enable for staging/development.
  introspection: !IS_PRODUCTION,

  // Plugin configuration
  plugins: [
    ApolloServerPluginDrainHttpServer({ httpServer }),

    // Landing page: show Apollo Sandbox in dev, disable in prod
    IS_PRODUCTION
      ? ApolloServerPluginLandingPageDisabled()
      : ApolloServerPluginLandingPageLocalDefault(),
  ],

  // CSRF prevention: requires a specific header on every request.
  // Apollo Client and most GraphQL clients send this automatically.
  csrfPrevention: true,

  // Error formatting (see Section 8)
  formatError: productionErrorFormatter,

  // Cache control
  cacheControl: {
    defaultMaxAge: 0, // Do not cache by default
  },
});

await server.start();

// Security headers
app.use(helmet());

// Trust proxy (if behind a load balancer)
app.set('trust proxy', 1);

app.use(
  '/graphql',
  cors({
    origin: IS_PRODUCTION
      ? ['https://myapp.com']
      : ['http://localhost:3000'],
    credentials: true,
  }),
  express.json({ limit: '1mb' }), // Limit request body size
  expressMiddleware(server, { context: createContext })
);

httpServer.listen(4000);
```

### 프로덕션 체크리스트

| 카테고리 | 항목 | 구현 |
|---------|------|------|
| **보안** | CSRF 방지 | `csrfPrevention: true` |
| | 인트로스펙션(introspection) 제어 | 프로덕션에서 `introspection: false` |
| | CORS | 명시적 출처 허용 목록 |
| | 본문 크기 제한 | `express.json({ limit: '1mb' })` |
| | 쿼리 깊이 제한 | 깊이 제한 플러그인 |
| | 쿼리 복잡도 제한 | 복잡도 분석 플러그인 |
| **성능** | DataLoader | 요청당 DataLoader |
| | 응답 캐싱 | `@cacheControl` + 캐시 플러그인 |
| | 영속 쿼리(Persisted queries) | APQ 또는 등록된 작업 |
| **관측 가능성** | 로깅 | 요청/오류 로깅 플러그인 |
| | 메트릭 | Apollo Studio 또는 Prometheus |
| | 추적 | OpenTelemetry 통합 |
| **신뢰성** | 정상 종료 | `ApolloServerPluginDrainHttpServer` |
| | 헬스 체크 | `/health` 엔드포인트 |
| | 속도 제한 | Express rate-limit 미들웨어 |

---

## 11. 연습 문제

### 연습 1: 단독에서 Express 마이그레이션 (입문)

`startStandaloneServer`를 사용하는 작동하는 Apollo Server가 있습니다. 다음을 추가할 수 있도록 Express 미들웨어로 마이그레이션하세요:

1. `{ status: "ok", uptime: process.uptime() }`을 반환하는 `GET /health` 엔드포인트
2. Stripe 웹훅 처리를 위한 `POST /webhooks/stripe` 엔드포인트
3. `http://localhost:3000`과 `https://yourapp.com`만 허용하도록 구성된 CORS
4. 정상 종료를 위한 `ApolloServerPluginDrainHttpServer`

완전한 마이그레이션 코드를 작성하세요.

### 연습 2: 커스텀 로깅 플러그인 (중급)

다음을 로깅하는 Apollo Server 플러그인을 구축하세요:

1. 작업 이름, 타입 (query/mutation/subscription), 변수(`password`라는 이름의 필드는 삭제)
2. 밀리초 단위의 실행 시간
3. 어떤 리졸버가 호출되었고 각각 얼마나 걸렸는지 (`willResolveField` 사용)
4. 마지막에 가장 느린 리졸버를 보여주는 요약

로그 집계 시스템에 의해 수집될 수 있는 구조화된 JSON으로 출력을 형식화하세요.

### 연습 3: codegen을 사용한 스키마 우선 (중급)

타입 안전성을 갖춘 스키마 우선 Apollo Server 프로젝트를 설정하세요:

1. `User`, `Post`, `Comment` 타입, 쿼리와 뮤테이션이 있는 스키마 작성
2. SDL에서 TypeScript 타입을 생성하도록 GraphQL Code Generator 구성
3. 생성된 타입을 사용하여 완전히 타입 지정된 리졸버 작성
4. 리졸버의 타입 오류가 컴파일 시간에 잡히는 것을 시연

`codegen.yml` 구성과 생성된 리졸버 타입 시그니처를 제공하세요.

### 연습 4: 멀티 스키마 스티칭(Stitching) (고급)

두 개의 Apollo Server 인스턴스가 있습니다:

- **Users Service**: `user(id)` 및 `users` 쿼리가 있는 `User` 타입
- **Posts Service**: `post(id)` 및 `posts` 쿼리가 있는 `Post` 타입 (Post에는 `authorId` 필드가 있음)

다음을 수행하는 게이트웨이 서버를 구축하세요:

1. 두 스키마를 함께 스티칭
2. Posts Service를 쿼리하여 해석하는 `posts` 필드를 `User`에 추가
3. Users Service를 쿼리하여 해석하는 `author` 필드를 `Post`에 추가
4. DataLoader를 사용하여 서비스 간 호출을 배치

### 연습 5: 프로덕션 강화 감사 (고급)

다음 Apollo Server 구성을 보고 모든 프로덕션 문제를 식별하고 수정된 버전을 제공하세요:

```typescript
const server = new ApolloServer({
  typeDefs,
  resolvers,
});

const { url } = await startStandaloneServer(server, {
  listen: { port: 4000 },
});
```

강화된 버전은 보안, 성능, 관측 가능성, 신뢰성 전반에 걸쳐 최소 8가지 구체적인 사항을 다루어야 합니다. 각 변경이 중요한 이유를 설명하세요.

---

## 12. 참고 자료

- Apollo Server 4 문서 — https://www.apollographql.com/docs/apollo-server/
- Apollo Server 마이그레이션 가이드 (v3에서 v4로) — https://www.apollographql.com/docs/apollo-server/migration
- Express 통합 — https://www.apollographql.com/docs/apollo-server/api/express-middleware
- Apollo Server 플러그인 — https://www.apollographql.com/docs/apollo-server/integrations/plugins
- Apollo Studio — https://www.apollographql.com/docs/studio/
- GraphQL Code Generator — https://the-guild.dev/graphql/codegen

---

**이전**: [인증과 인가](./07_Authentication_Authorization.md) | **다음**: [GraphQL 클라이언트](./09_GraphQL_Clients.md)
