# 14. 성능과 보안(Performance and Security)

**이전**: [테스팅](./13_Testing.md) | **다음**: [REST에서 GraphQL로 마이그레이션](./15_REST_to_GraphQL_Migration.md)

## 학습 목표

이 레슨을 마치면 다음을 할 수 있습니다:

1. 리소스 고갈 공격을 방지하기 위한 쿼리 깊이 제한(query depth limiting)과 비용 분석(cost analysis)을 구현한다
2. 필드 수준 비용 지시어와 작업별 복잡도 예산을 구성한다
3. 배치 남용, 인트로스펙션 유출, 인젝션을 포함한 일반적인 공격 벡터로부터 GraphQL API를 보호한다
4. GraphQL 작업의 가변적 비용을 고려한 속도 제한(rate limiting) 전략을 설정한다
5. 느린 쿼리 로깅, 작업 메트릭, 알림을 포함한 모니터링 파이프라인을 구축한다

## 목차

1. [GraphQL에 특별한 보호가 필요한 이유](#1-graphql에-특별한-보호가-필요한-이유)
2. [쿼리 깊이 제한](#2-쿼리-깊이-제한)
3. [쿼리 복잡도 분석](#3-쿼리-복잡도-분석)
4. [필드 수준 비용 지시어](#4-필드-수준-비용-지시어)
5. [속도 제한](#5-속도-제한)
6. [인트로스펙션 제어](#6-인트로스펙션-제어)
7. [배치 공격 방지](#7-배치-공격-방지)
8. [입력 유효성 검사와 인젝션 방지](#8-입력-유효성-검사와-인젝션-방지)
9. [타임아웃과 리소스 제한](#9-타임아웃과-리소스-제한)
10. [모니터링과 가관측성](#10-모니터링과-가관측성)
11. [연습 문제](#11-연습-문제)

**난이도**: ⭐⭐⭐⭐

---

GraphQL의 강력함은 동시에 취약점이기도 합니다. 단일 쿼리가 전체 그래프를 순회하며, 수천 개의 데이터베이스 호출을 유발하는 깊게 중첩된 데이터를 요청할 수 있습니다. 각 엔드포인트가 예측 가능한 비용을 가진 REST와 달리, GraphQL 엔드포인트의 비용은 전적으로 클라이언트가 무엇을 요청하느냐에 달려 있습니다. 이 레슨은 의도적이든 우발적이든 서버를 남용으로부터 안전하게 유지하면서 유연한 쿼리를 제공할 수 있는 기술들을 다룹니다.

---

## 1. GraphQL에 특별한 보호가 필요한 이유

### REST와의 차이점

```
REST:
  GET /api/users       → 고정 비용: 1번의 DB 쿼리, ~50개 행
  GET /api/users/42    → 고정 비용: 1번의 DB 쿼리, 1개 행
  비용 예측 가능. 엔드포인트별 속도 제한이 효과적.

GraphQL:
  query {
    users {               → 1번의 쿼리
      posts {             → N번의 쿼리 (사용자당 1번)
        comments {        → N*M번의 쿼리 (게시물당 1번)
          author {        → N*M*K번의 쿼리 (댓글당 1번)
            posts { ... } → 재귀적 폭발
          }
        }
      }
    }
  }
  비용 예측 불가. 단일 요청이 1번이 될 수도 10,000번이 될 수도 있음.
```

### 공격 표면

```
┌────────────────────────────────────────────────────┐
│              GraphQL 공격 표면                       │
├────────────────────────────────────────────────────┤
│                                                      │
│  ┌─────────────┐  ┌─────────────┐  ┌────────────┐  │
│  │  깊은 쿼리  │  │   배치      │  │ 인트로스펙 │  │
│  │    중첩     │  │   공격      │  │  션 유출   │  │
│  └─────────────┘  └─────────────┘  └────────────┘  │
│                                                      │
│  ┌─────────────┐  ┌─────────────┐  ┌────────────┐  │
│  │   별칭      │  │  인젝션     │  │  리소스    │  │
│  │  플러딩     │  │  (입력 통해)│  │   고갈     │  │
│  └─────────────┘  └─────────────┘  └────────────┘  │
│                                                      │
└────────────────────────────────────────────────────┘
```

---

## 2. 쿼리 깊이 제한

가장 단순한 방어: 임계값보다 더 깊이 중첩된 쿼리를 거부합니다.

### 깊이가 중요한 이유

```graphql
# 깊이 1 — 안전
query { users { name } }

# 깊이 3 — 일반적
query {
  user(id: "1") {
    posts {
      comments {
        body
      }
    }
  }
}

# 깊이 10+ — 공격 또는 실수 가능성
query {
  user(id: "1") {
    friends {
      friends {
        friends {
          friends {
            friends {
              friends {
                friends { name }
              }
            }
          }
        }
      }
    }
  }
}
```

### graphql-depth-limit을 사용한 구현

```typescript
import depthLimit from 'graphql-depth-limit';
import { ApolloServer } from '@apollo/server';

const server = new ApolloServer({
  typeDefs,
  resolvers,
  validationRules: [
    depthLimit(10), // 10레벨보다 깊은 쿼리 거부
  ],
});
```

### 커스텀 깊이 제한기

```typescript
import { ValidationContext, ASTVisitor } from 'graphql';

function createDepthLimiter(maxDepth: number): (context: ValidationContext) => ASTVisitor {
  return (context: ValidationContext): ASTVisitor => {
    return {
      // 선택 집합에 진입/이탈 시 깊이 추적
      Field: {
        enter(node, _key, _parent, path) {
          // AST 경로에서 깊이 계산
          const depth = path.filter(
            (segment) => typeof segment === 'number'
          ).length;

          if (depth > maxDepth) {
            context.reportError(
              new GraphQLError(
                `Query depth ${depth} exceeds maximum allowed depth of ${maxDepth}`,
                { nodes: [node] }
              )
            );
          }
        },
      },
    };
  };
}

const server = new ApolloServer({
  typeDefs,
  resolvers,
  validationRules: [createDepthLimiter(10)],
});
```

### 깊이 제한 권장 사항

| API 유형 | 권장 최대 깊이 |
|---------|--------------|
| 단순 CRUD | 5-7 |
| 소셜 그래프 | 7-10 |
| 전자상거래 | 8-12 |
| 내부/신뢰된 | 15-20 |

---

## 3. 쿼리 복잡도 분석

깊이 제한은 거친 방법입니다 — 1,000개의 별칭을 가진 얕은 쿼리도 위험할 수 있습니다. 복잡도 분석(Complexity Analysis)은 각 필드에 비용을 할당하고 예산을 초과하는 쿼리를 거부합니다.

### 작동 방식

```graphql
# 각 필드의 비용:
# 스칼라 필드: 0 (무료)
# 객체 필드: 1
# 목록 필드: 비용 * 예상 크기

query {
  users(first: 100) {        # 비용: 1 + (100 * 자식 비용)
    name                     # 비용: 0
    posts(first: 50) {       # 비용: 100 * (1 + 50 * 자식 비용) = 100 * 51
      title                  # 비용: 0
      comments(first: 20) {  # 비용: 100 * 50 * (1 + 20 * 1) = 100 * 50 * 21
        body                 # 비용: 0
      }
    }
  }
}

# 합계: 1 + 100 * (51 + 50 * 21) = 1 + 100 * 1101 = 110,101
# 예산이 10,000이라면 → 거부됨
```

### graphql-query-complexity를 사용한 구현

```typescript
import {
  getComplexity,
  simpleEstimator,
  fieldExtensionsEstimator,
} from 'graphql-query-complexity';
import { ApolloServer } from '@apollo/server';

const server = new ApolloServer({
  typeDefs,
  resolvers,
  plugins: [
    {
      async requestDidStart() {
        return {
          async didResolveOperation({ request, document, schema }) {
            const complexity = getComplexity({
              schema,
              operationName: request.operationName,
              query: document,
              variables: request.variables,
              estimators: [
                // 스키마의 @complexity 지시어 사용 (먼저 확인)
                fieldExtensionsEstimator(),
                // 대체: 필드당 1
                simpleEstimator({ defaultComplexity: 1 }),
              ],
            });

            const MAX_COMPLEXITY = 1000;

            if (complexity > MAX_COMPLEXITY) {
              throw new GraphQLError(
                `Query complexity ${complexity} exceeds maximum of ${MAX_COMPLEXITY}`,
                {
                  extensions: {
                    code: 'QUERY_TOO_COMPLEX',
                    complexity,
                    maxComplexity: MAX_COMPLEXITY,
                  },
                }
              );
            }

            // 모니터링을 위한 복잡도 로깅
            console.log(`Query complexity: ${complexity}`);
          },
        };
      },
    },
  ],
});
```

---

## 4. 필드 수준 비용 지시어

일률적인 추정기 대신, 비용이 많이 드는 것으로 알려진 필드에 특정 비용을 할당합니다.

### 스키마 수준 비용 주석

```graphql
directive @complexity(
  value: Int!
  multipliers: [String!]
) on FIELD_DEFINITION

type Query {
  # 단순 조회: 낮은 비용
  user(id: ID!): User @complexity(value: 1)

  # 전체 텍스트 검색: 비용 높음
  searchProducts(query: String!, first: Int = 10): [Product!]!
    @complexity(value: 10, multipliers: ["first"])

  # 분석 쿼리: 매우 높은 비용
  salesReport(from: DateTime!, to: DateTime!): SalesReport
    @complexity(value: 50)
}

type User {
  id: ID!
  name: String!      # 스칼라: 기본 비용 0

  # 조인 쿼리가 필요한 목록
  orders(first: Int = 10): [Order!]!
    @complexity(value: 5, multipliers: ["first"])

  # 외부 API 호출이 있는 계산 필드
  creditScore: Int @complexity(value: 20)
}
```

### 커스텀 추정기

```typescript
import { ComplexityEstimator } from 'graphql-query-complexity';

const customEstimator: ComplexityEstimator = ({
  type,
  field,
  args,
  childComplexity,
}) => {
  // 스키마 확장에서 비용 읽기(@complexity 지시어로 설정됨)
  const complexity = field.extensions?.complexity;

  if (complexity) {
    const { value, multipliers } = complexity;

    // 인수에서 배수 적용 (예: "first", "limit")
    let multiplier = 1;
    if (multipliers) {
      for (const argName of multipliers) {
        if (args[argName]) {
          multiplier *= args[argName];
        }
      }
    }

    // 총 비용 = 기본 비용 + (배수 * 자식 복잡도)
    return value + multiplier * childComplexity;
  }

  // 주석 없음: undefined를 반환하여 다음 추정기가 처리하게 함
  return undefined;
};
```

---

## 5. 속도 제한

전통적인 속도 제한(분당 X 요청)은 GraphQL에 잘 맞지 않습니다. 한 요청의 비용이 엄청나게 다를 수 있기 때문입니다. 대신 **복잡도 예산(complexity budget)**으로 속도를 제한하세요.

### 복잡도 기반 속도 제한

```typescript
import { RateLimiterRedis } from 'rate-limiter-flexible';
import Redis from 'ioredis';

const redis = new Redis();

// 각 사용자는 시간 창당 복잡도 예산을 받음
const complexityLimiter = new RateLimiterRedis({
  storeClient: redis,
  keyPrefix: 'gql_complexity',
  points: 10_000,   // 복잡도 포인트 10,000
  duration: 60,     // 60초당
});

// 작업 수도 제한
const operationLimiter = new RateLimiterRedis({
  storeClient: redis,
  keyPrefix: 'gql_operations',
  points: 100,      // 100번의 작업
  duration: 60,     // 60초당
});

const rateLimitPlugin = {
  async requestDidStart() {
    return {
      async didResolveOperation({ request, document, schema, contextValue }) {
        const userId = contextValue.userId || contextValue.ip;

        // 작업 수 제한 확인
        try {
          await operationLimiter.consume(userId);
        } catch {
          throw new GraphQLError('Rate limit exceeded: too many requests', {
            extensions: { code: 'RATE_LIMITED' },
          });
        }

        // 복잡도 예산 확인
        const complexity = calculateComplexity(schema, document, request.variables);
        try {
          await complexityLimiter.consume(userId, complexity);
        } catch (rateLimiterRes) {
          throw new GraphQLError(
            `Complexity budget exceeded. Retry after ${Math.ceil(
              rateLimiterRes.msBeforeNext / 1000
            )}s`,
            {
              extensions: {
                code: 'COMPLEXITY_BUDGET_EXCEEDED',
                retryAfter: Math.ceil(rateLimiterRes.msBeforeNext / 1000),
              },
            }
          );
        }
      },
    };
  },
};
```

### 속도 제한 응답 헤더

```typescript
// 클라이언트에게 남은 예산 알림
const rateLimitHeadersPlugin = {
  async requestDidStart() {
    return {
      async willSendResponse({ response, contextValue }) {
        const userId = contextValue.userId || contextValue.ip;
        const limiterRes = await complexityLimiter.get(userId);

        response.http.headers.set(
          'X-RateLimit-Complexity-Limit', '10000'
        );
        response.http.headers.set(
          'X-RateLimit-Complexity-Remaining',
          String(limiterRes ? limiterRes.remainingPoints : 10000)
        );
        response.http.headers.set(
          'X-RateLimit-Complexity-Reset',
          String(limiterRes ? Math.ceil(limiterRes.msBeforeNext / 1000) : 60)
        );
      },
    };
  },
};
```

---

## 6. 인트로스펙션 제어

인트로스펙션(Introspection)은 타입, 필드, 인수, 지시어를 포함한 전체 스키마를 노출합니다. 개발 중에는 유용하지만 프로덕션에서는 위험합니다.

### 프로덕션에서 인트로스펙션 비활성화

```typescript
import { ApolloServer } from '@apollo/server';
import {
  ApolloServerPluginInlineTraceDisabled,
} from '@apollo/server/plugin/disabled';

const server = new ApolloServer({
  typeDefs,
  resolvers,
  introspection: process.env.NODE_ENV !== 'production',
  plugins: [
    // 프로덕션에서 인라인 추적도 비활성화
    ...(process.env.NODE_ENV === 'production'
      ? [ApolloServerPluginInlineTraceDisabled()]
      : []),
  ],
});
```

### 선택적 인트로스펙션 (내부 허용, 외부 차단)

```typescript
// 커스텀 플러그인: 내부 IP에서만 인트로스펙션 허용
const selectiveIntrospectionPlugin = {
  async requestDidStart({ request, contextValue }) {
    return {
      async didResolveOperation({ document }) {
        const isIntrospection = document.definitions.some(
          (def) =>
            def.kind === 'OperationDefinition' &&
            def.selectionSet.selections.some(
              (sel) =>
                sel.kind === 'Field' &&
                (sel.name.value === '__schema' || sel.name.value === '__type')
            )
        );

        if (isIntrospection && !contextValue.isInternalRequest) {
          throw new GraphQLError('Introspection is not allowed', {
            extensions: { code: 'INTROSPECTION_DISABLED' },
          });
        }
      },
    };
  },
};
```

---

## 7. 배치 공격 방지

GraphQL은 단일 HTTP 요청에 여러 작업을 보내는 것(쿼리 배치)을 지원합니다. 공격자는 이를 악용하여 속도 제한을 우회할 수 있습니다.

### 공격 방식

```json
// 1,000번의 로그인 시도가 담긴 단일 HTTP 요청
[
  { "query": "mutation { login(email: \"a@b.com\", password: \"pass1\") { token } }" },
  { "query": "mutation { login(email: \"a@b.com\", password: \"pass2\") { token } }" },
  { "query": "mutation { login(email: \"a@b.com\", password: \"pass3\") { token } }" },
  // ... 997개 더
]
```

### 별칭 기반 공격 (단일 쿼리)

```graphql
# 1,000개의 별칭이 있는 단일 쿼리
query {
  attempt1: login(email: "a@b.com", password: "pass1") { token }
  attempt2: login(email: "a@b.com", password: "pass2") { token }
  attempt3: login(email: "a@b.com", password: "pass3") { token }
  # ... 997개 더
}
```

### 방지책

```typescript
// 1. 배치 크기 제한
const server = new ApolloServer({
  typeDefs,
  resolvers,
  allowBatchedHttpRequests: true,  // 배치 허용하되...
});

// 배치 크기를 제한하는 미들웨어
app.use('/graphql', (req, res, next) => {
  if (Array.isArray(req.body) && req.body.length > 5) {
    return res.status(400).json({
      error: 'Batch size exceeds maximum of 5 operations',
    });
  }
  next();
});

// 2. 쿼리당 별칭 제한
import { ASTVisitor, ValidationContext } from 'graphql';

function aliasLimit(maxAliases: number) {
  return (context: ValidationContext): ASTVisitor => {
    let aliasCount = 0;
    return {
      Field(node) {
        if (node.alias) {
          aliasCount++;
          if (aliasCount > maxAliases) {
            context.reportError(
              new GraphQLError(
                `Query uses ${aliasCount} aliases, exceeding the maximum of ${maxAliases}`
              )
            );
          }
        }
      },
    };
  };
}

const server = new ApolloServer({
  typeDefs,
  resolvers,
  validationRules: [aliasLimit(50)],
});
```

---

## 8. 입력 유효성 검사와 인젝션 방지

### 유효성 검사를 위한 커스텀 스칼라

```typescript
import { GraphQLScalarType, Kind } from 'graphql';

// 유효성 검사가 있는 Email 스칼라
const EmailScalar = new GraphQLScalarType({
  name: 'Email',
  description: 'A valid email address',
  serialize(value: string) {
    return value;
  },
  parseValue(value: string) {
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailRegex.test(value)) {
      throw new GraphQLError(`Invalid email address: ${value}`);
    }
    return value.toLowerCase();
  },
  parseLiteral(ast) {
    if (ast.kind !== Kind.STRING) {
      throw new GraphQLError('Email must be a string');
    }
    return this.parseValue(ast.value);
  },
});

// HTML/스크립트를 제거하는 정제된 문자열
const SanitizedString = new GraphQLScalarType({
  name: 'SanitizedString',
  description: 'A string with HTML tags stripped',
  parseValue(value: string) {
    // 저장된 XSS를 방지하기 위해 HTML 태그 제거
    return value.replace(/<[^>]*>/g, '').trim();
  },
  serialize(value: string) {
    return value;
  },
  parseLiteral(ast) {
    if (ast.kind !== Kind.STRING) {
      throw new GraphQLError('SanitizedString must be a string');
    }
    return this.parseValue(ast.value);
  },
});
```

### SQL 인젝션 방지

GraphQL 자체는 SQL 인젝션을 방지하지 않습니다 — 그 책임은 리졸버 계층에 있습니다.

```typescript
// 나쁜 예: 문자열 보간 → SQL 인젝션 취약
const resolvers = {
  Query: {
    user: async (_, { id }, { db }) => {
      // 절대 이렇게 하지 마세요
      return db.query(`SELECT * FROM users WHERE id = '${id}'`);
    },
  },
};

// 좋은 예: 매개변수화된 쿼리
const resolvers = {
  Query: {
    user: async (_, { id }, { db }) => {
      return db.query('SELECT * FROM users WHERE id = $1', [id]);
    },
  },
};

// 좋은 예: 내장 매개변수화가 있는 ORM
const resolvers = {
  Query: {
    user: async (_, { id }, { prisma }) => {
      return prisma.user.findUnique({ where: { id } });
    },
  },
};
```

### 입력 객체 유효성 검사

```typescript
import { z } from 'zod';

// Zod로 유효성 검사 스키마 정의
const CreateUserSchema = z.object({
  name: z.string().min(1).max(100),
  email: z.string().email(),
  password: z.string().min(8).max(72),
  bio: z.string().max(500).optional(),
});

const resolvers = {
  Mutation: {
    createUser: async (_, { input }, { dataSources }) => {
      // 처리 전 입력 유효성 검사
      const validatedInput = CreateUserSchema.parse(input);
      return dataSources.users.create(validatedInput);
    },
  },
};
```

---

## 9. 타임아웃과 리소스 제한

### 쿼리 실행 타임아웃

```typescript
// 리졸버당 타임아웃
const withTimeout = (resolver, timeoutMs = 5000) => {
  return async (...args) => {
    const timeoutPromise = new Promise((_, reject) =>
      setTimeout(
        () => reject(new GraphQLError(`Resolver timed out after ${timeoutMs}ms`)),
        timeoutMs
      )
    );

    return Promise.race([resolver(...args), timeoutPromise]);
  };
};

const resolvers = {
  Query: {
    searchProducts: withTimeout(
      async (_, { query }, { dataSources }) => {
        return dataSources.products.search(query);
      },
      3000 // 검색에 3초 타임아웃
    ),

    analyticsReport: withTimeout(
      async (_, { dateRange }, { dataSources }) => {
        return dataSources.analytics.generate(dateRange);
      },
      10000 // 무거운 보고서에 10초 타임아웃
    ),
  },
};
```

### 요청 크기 제한

```typescript
import express from 'express';

const app = express();

// 거대한 쿼리를 방지하기 위해 요청 본문 크기 제한
app.use(express.json({ limit: '100kb' }));

// 추가 검사를 위한 커스텀 미들웨어
app.use('/graphql', (req, res, next) => {
  const queryLength = req.body?.query?.length || 0;

  if (queryLength > 10_000) {
    return res.status(413).json({
      errors: [{
        message: 'Query too large',
        extensions: {
          code: 'QUERY_TOO_LARGE',
          maxLength: 10_000,
          actualLength: queryLength,
        },
      }],
    });
  }

  next();
});
```

### 커넥션 풀 제한

```typescript
// 단일 쿼리가 모든 데이터베이스 연결을 소비하지 않도록 방지
import { Pool } from 'pg';

const pool = new Pool({
  max: 20,                    // 최대 20개 연결
  connectionTimeoutMillis: 5000, // 연결 대기 최대 5초
  idleTimeoutMillis: 30000,     // 30초 후 유휴 연결 종료
  statement_timeout: 10000,     // 10초 이상 실행되는 쿼리 종료
});
```

---

## 10. 모니터링과 가관측성

### 느린 쿼리 로깅

```typescript
const slowQueryPlugin = {
  async requestDidStart({ request }) {
    const startTime = Date.now();

    return {
      async willSendResponse({ response }) {
        const duration = Date.now() - startTime;
        const SLOW_THRESHOLD = 1000; // 1초

        if (duration > SLOW_THRESHOLD) {
          console.warn({
            message: 'Slow GraphQL query detected',
            operationName: request.operationName,
            duration: `${duration}ms`,
            query: request.query?.substring(0, 500), // 로깅용 잘라내기
            variables: request.variables,
            timestamp: new Date().toISOString(),
          });
        }
      },
    };
  },
};
```

### Prometheus를 사용한 작업 메트릭

```typescript
import { Counter, Histogram, register } from 'prom-client';

const operationDuration = new Histogram({
  name: 'graphql_operation_duration_seconds',
  help: 'Duration of GraphQL operations',
  labelNames: ['operationName', 'operationType', 'status'],
  buckets: [0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10],
});

const operationErrors = new Counter({
  name: 'graphql_operation_errors_total',
  help: 'Total GraphQL operation errors',
  labelNames: ['operationName', 'errorCode'],
});

const metricsPlugin = {
  async requestDidStart({ request }) {
    const timer = operationDuration.startTimer();

    return {
      async willSendResponse({ response }) {
        const hasErrors = response.body?.singleResult?.errors?.length > 0;

        timer({
          operationName: request.operationName || 'anonymous',
          operationType: 'query', // 문서에서 결정
          status: hasErrors ? 'error' : 'success',
        });

        if (hasErrors) {
          for (const error of response.body.singleResult.errors) {
            operationErrors.inc({
              operationName: request.operationName || 'anonymous',
              errorCode: error.extensions?.code || 'UNKNOWN',
            });
          }
        }
      },
    };
  },
};

// 메트릭 엔드포인트 노출
app.get('/metrics', async (req, res) => {
  res.set('Content-Type', register.contentType);
  res.end(await register.metrics());
});
```

### OpenTelemetry를 사용한 추적

```typescript
import { NodeTracerProvider } from '@opentelemetry/sdk-trace-node';
import { SimpleSpanProcessor } from '@opentelemetry/sdk-trace-base';
import { JaegerExporter } from '@opentelemetry/exporter-jaeger';

const provider = new NodeTracerProvider();
provider.addSpanProcessor(
  new SimpleSpanProcessor(
    new JaegerExporter({ endpoint: 'http://jaeger:14268/api/traces' })
  )
);
provider.register();

// 리졸버에서 비용이 많은 작업에 스팬 생성
const tracer = provider.getTracer('graphql-server');

const resolvers = {
  Query: {
    searchProducts: async (_, { query }, { dataSources }) => {
      const span = tracer.startSpan('searchProducts');
      span.setAttribute('search.query', query);

      try {
        const results = await dataSources.products.search(query);
        span.setAttribute('search.resultCount', results.length);
        return results;
      } catch (error) {
        span.recordException(error);
        throw error;
      } finally {
        span.end();
      }
    },
  },
};
```

---

## 11. 연습 문제

### 문제 1: 깊이와 복잡도 제한
다음 스키마에 대한 깊이 + 복잡도 결합 제한기를 구현하세요. 적절한 제한을 설정하고 다양한 깊이와 비용의 쿼리로 테스트하세요:

```graphql
type Query {
  users(first: Int = 10): [User!]!
  product(id: ID!): Product
}

type User {
  id: ID!
  name: String!
  friends: [User!]!
  orders: [Order!]!
}

type Order {
  id: ID!
  items: [OrderItem!]!
}

type OrderItem {
  product: Product!
  quantity: Int!
}

type Product {
  id: ID!
  name: String!
  reviews: [Review!]!
}

type Review {
  author: User!
  body: String!
}
```

### 문제 2: 속도 제한기
Redis에서 사용자별 예산을 추적하는 복잡도 인식 속도 제한기를 구축하세요. 각 사용자는 분당 5,000 복잡도 포인트를 받습니다. 제한기를 Apollo Server 플러그인으로 구현하고 남은 예산을 보여주는 응답 헤더를 포함하세요.

### 문제 3: 보안 감사
보안 조치가 없는 GraphQL 서버가 주어졌을 때, 필요한 모든 보호 조치의 체크리스트를 작성하고 가장 중요한 세 가지를 구현하세요: (a) 쿼리 깊이 제한, (b) 프로덕션에서 인트로스펙션 비활성화, (c) 모든 뮤테이션 인수의 입력 유효성 검사.

### 문제 4: 모니터링 대시보드
GraphQL 서버를 위해 Prometheus와 Grafana를 사용한 모니터링 설정을 설계하세요. 추적할 최소 5가지 메트릭을 정의하세요(작업 시간, 오류율, 복잡도 분포, 캐시 적중률, 리졸버 수준 지연 시간). Prometheus 메트릭 정의와 샘플 Grafana 대시보드 JSON을 작성하세요.

### 문제 5: 별칭 플러드 방어
공격자가 귀하의 전자상거래 API에 이 쿼리를 보냅니다:

```graphql
query {
  a1: product(id: "1") { name price }
  a2: product(id: "2") { name price }
  # ... 998개 더
}
```

세 가지 방어 계층을 구현하세요: (a) 별칭 수 제한, (b) 별칭 수를 계산하는 복잡도 분석, (c) 별칭 수를 고려하는 IP별 속도 제한. 각 계층을 독립적으로 테스트하세요.

---

## 참고 자료

- [graphql-depth-limit](https://github.com/stems/graphql-depth-limit)
- [graphql-query-complexity](https://github.com/slicknode/graphql-query-complexity)
- [Apollo Server 보안](https://www.apollographql.com/docs/apollo-server/security/authentication/)
- [OWASP GraphQL 치트 시트](https://cheatsheetseries.owasp.org/cheatsheets/GraphQL_Cheat_Sheet.html)
- [GraphQL 보안 모범 사례 (The Guild)](https://the-guild.dev/blog/graphql-security)
- [GraphQL API 보안 (Apollo Blog)](https://www.apollographql.com/blog/graphql/security/)

---

**이전**: [테스팅](./13_Testing.md) | **다음**: [REST에서 GraphQL로 마이그레이션](./15_REST_to_GraphQL_Migration.md)
