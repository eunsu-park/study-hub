# 13. 테스팅(Testing)

**이전**: [연합(Federation)](./12_Federation.md) | **다음**: [성능과 보안](./14_Performance_Security.md)

## 학습 목표

이 레슨을 마치면 다음을 할 수 있습니다:

1. 단위(unit), 통합(integration), 엔드-투-엔드(end-to-end) 계층에 걸친 GraphQL API의 종합적인 테스트 전략을 설계한다
2. 모의(mocked) 컨텍스트와 데이터 소스로 리졸버 단위 테스트를 작성한다
3. `executeOperation`과 HTTP 테스트 클라이언트를 사용한 통합 테스트를 수행한다
4. 파괴적 변경(breaking-change) 감지와 스키마 린팅(linting)으로 스키마 진화를 검증한다
5. MockedProvider와 요청 모킹(request mocking) 라이브러리를 사용하여 GraphQL 클라이언트를 테스트한다

## 목차

1. [GraphQL 테스트 전략](#1-graphql-테스트-전략)
2. [리졸버 단위 테스트](#2-리졸버-단위-테스트)
3. [통합 테스트](#3-통합-테스트)
4. [스키마 테스트](#4-스키마-테스트)
5. [클라이언트 측 테스트](#5-클라이언트-측-테스트)
6. [스냅샷 테스트](#6-스냅샷-테스트)
7. [CI 통합](#7-ci-통합)
8. [연습 문제](#8-연습-문제)

**난이도**: ⭐⭐⭐

---

GraphQL 테스팅은 REST 테스팅과 다릅니다. 단일 엔드포인트가 무한한 쿼리 형태를 제공하고, 리졸버는 하나의 버그가 연쇄적으로 영향을 미칠 수 있는 체인을 형성하며, 스키마 자체가 예상치 못하게 깨지면 안 되는 계약입니다. 이 레슨은 격리된 리졸버 단위 테스트부터 시작하여, 전체 서버 통합 테스트를 거쳐, 클라이언트 측 모킹과 CI 파이프라인 통합으로 마무리하는 테스트 전략을 처음부터 구축합니다.

---

## 1. GraphQL 테스트 전략

### GraphQL을 위한 테스팅 피라미드

```
           ┌───────┐
           │  E2E  │   Cypress / Playwright
           │ 테스트 │   전체 스택, 느림, 소수
           ├───────┤
           │       │
        ┌──┤ 통합  ├──┐   executeOperation, HTTP
        │  │ 테스트 │  │   서버 + 리졸버 + 데이터
        │  ├───────┤  │
        │  │       │  │
     ┌──┤  │ 단위  │  ├──┐   리졸버 함수
     │  │  │ 테스트 │  │  │   모의 컨텍스트/데이터
     │  │  ├───────┤  │  │
     │  │  │       │  │  │
  ┌──┤  │  │스키마 │  │  ├──┐   린팅, 파괴적 변경
  │  │  │  │ 테스트 │  │  │  │   구성 검증
  └──┴──┴──┴───────┴──┴──┴──┘
```

### 각 계층에서 테스트할 내용

| 계층 | 테스트 대상 | 도구 | 속도 |
|------|-----------|------|------|
| 스키마 | 파괴적 변경, 린팅 규칙, 구성 | graphql-inspector, Rover | 빠름 |
| 단위 | 개별 리졸버 로직, 비즈니스 규칙 | Jest, Vitest | 빠름 |
| 통합 | 전체 쿼리 실행, 리졸버 체인, 인증 | executeOperation, supertest | 보통 |
| E2E | 클라이언트-서버 흐름, 부수 효과가 있는 뮤테이션 | Cypress, Playwright | 느림 |

---

## 2. 리졸버 단위 테스트

리졸버는 순수 함수입니다. 테스트는 모의(mocked) 인수, 컨텍스트, 데이터 소스로 함수를 직접 호출하는 것을 의미합니다.

### 기본 리졸버 테스트

```typescript
// resolvers/user.ts
export const userResolvers = {
  Query: {
    user: async (_, { id }, { dataSources }) => {
      return dataSources.users.findById(id);
    },
    users: async (_, { limit = 10 }, { dataSources }) => {
      return dataSources.users.findAll({ limit });
    },
  },
  User: {
    posts: async (parent, _, { dataSources }) => {
      return dataSources.posts.findByAuthorId(parent.id);
    },
    fullName: (parent) => `${parent.firstName} ${parent.lastName}`,
  },
};
```

```typescript
// resolvers/user.test.ts
import { describe, it, expect, vi } from 'vitest';
import { userResolvers } from './user';

describe('User Resolvers', () => {
  // 데이터 소스 모킹 — 리졸버 로직을 데이터베이스에서 격리
  const mockDataSources = {
    users: {
      findById: vi.fn(),
      findAll: vi.fn(),
    },
    posts: {
      findByAuthorId: vi.fn(),
    },
  };

  const mockContext = {
    dataSources: mockDataSources,
    userId: 'current-user-42',
  };

  afterEach(() => {
    vi.clearAllMocks();
  });

  describe('Query.user', () => {
    it('ID로 사용자를 반환한다', async () => {
      const expectedUser = { id: '1', firstName: 'Alice', lastName: 'Smith' };
      mockDataSources.users.findById.mockResolvedValue(expectedUser);

      const result = await userResolvers.Query.user(
        {},           // parent (루트)
        { id: '1' },  // args
        mockContext,   // context
      );

      expect(result).toEqual(expectedUser);
      expect(mockDataSources.users.findById).toHaveBeenCalledWith('1');
    });

    it('존재하지 않는 사용자에 대해 null을 반환한다', async () => {
      mockDataSources.users.findById.mockResolvedValue(null);

      const result = await userResolvers.Query.user(
        {},
        { id: 'nonexistent' },
        mockContext,
      );

      expect(result).toBeNull();
    });
  });

  describe('User.fullName', () => {
    it('성과 이름을 결합한다', () => {
      const parent = { firstName: 'Alice', lastName: 'Smith' };
      const result = userResolvers.User.fullName(parent);
      expect(result).toBe('Alice Smith');
    });
  });

  describe('User.posts', () => {
    it('부모 사용자의 게시물을 로드한다', async () => {
      const posts = [
        { id: 'p1', title: 'First Post' },
        { id: 'p2', title: 'Second Post' },
      ];
      mockDataSources.posts.findByAuthorId.mockResolvedValue(posts);

      const parent = { id: 'user-1' };
      const result = await userResolvers.User.posts(parent, {}, mockContext);

      expect(result).toEqual(posts);
      expect(mockDataSources.posts.findByAuthorId).toHaveBeenCalledWith('user-1');
    });
  });
});
```

### 인증 로직 테스트

```typescript
// resolvers/admin.ts
export const adminResolvers = {
  Query: {
    adminDashboard: async (_, __, { userId, dataSources }) => {
      if (!userId) {
        throw new GraphQLError('Authentication required', {
          extensions: { code: 'UNAUTHENTICATED' },
        });
      }

      const user = await dataSources.users.findById(userId);
      if (user.role !== 'ADMIN') {
        throw new GraphQLError('Admin access required', {
          extensions: { code: 'FORBIDDEN' },
        });
      }

      return dataSources.analytics.getDashboard();
    },
  },
};
```

```typescript
// resolvers/admin.test.ts
describe('Query.adminDashboard', () => {
  it('인증되지 않은 요청을 거부한다', async () => {
    const context = { userId: null, dataSources: mockDataSources };

    await expect(
      adminResolvers.Query.adminDashboard({}, {}, context)
    ).rejects.toThrow('Authentication required');
  });

  it('관리자가 아닌 사용자를 거부한다', async () => {
    mockDataSources.users.findById.mockResolvedValue({ role: 'USER' });
    const context = { userId: 'user-1', dataSources: mockDataSources };

    await expect(
      adminResolvers.Query.adminDashboard({}, {}, context)
    ).rejects.toThrow('Admin access required');
  });

  it('관리자 사용자에게 대시보드를 반환한다', async () => {
    const dashboardData = { totalUsers: 100, revenue: 5000 };
    mockDataSources.users.findById.mockResolvedValue({ role: 'ADMIN' });
    mockDataSources.analytics.getDashboard.mockResolvedValue(dashboardData);

    const context = { userId: 'admin-1', dataSources: mockDataSources };
    const result = await adminResolvers.Query.adminDashboard({}, {}, context);

    expect(result).toEqual(dashboardData);
  });
});
```

---

## 3. 통합 테스트

통합 테스트는 서버에 대해 실제 GraphQL 작업을 실행하여, 완전한 리졸버 체인, 미들웨어, 오류 처리를 검증합니다.

### executeOperation 사용 (Apollo Server 4)

```typescript
// __tests__/integration/user.test.ts
import { ApolloServer } from '@apollo/server';
import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { typeDefs } from '../schema';
import { resolvers } from '../resolvers';

describe('User Integration Tests', () => {
  let server: ApolloServer;

  beforeAll(() => {
    server = new ApolloServer({ typeDefs, resolvers });
  });

  it('ID로 사용자를 쿼리한다', async () => {
    const response = await server.executeOperation({
      query: `
        query GetUser($id: ID!) {
          user(id: $id) {
            id
            name
            email
          }
        }
      `,
      variables: { id: '1' },
    }, {
      // 작업을 위한 컨텍스트 값 제공
      contextValue: {
        dataSources: createTestDataSources(),
        userId: 'test-user',
      },
    });

    // 응답 확인
    expect(response.body.kind).toBe('single');
    if (response.body.kind === 'single') {
      expect(response.body.singleResult.errors).toBeUndefined();
      expect(response.body.singleResult.data?.user).toEqual({
        id: '1',
        name: 'Alice',
        email: 'alice@example.com',
      });
    }
  });

  it('잘못된 쿼리에 대해 유효성 검사 오류를 반환한다', async () => {
    const response = await server.executeOperation({
      query: `
        query {
          user(id: "1") {
            nonExistentField
          }
        }
      `,
    });

    if (response.body.kind === 'single') {
      expect(response.body.singleResult.errors).toBeDefined();
      expect(response.body.singleResult.errors?.[0].message).toContain(
        'Cannot query field "nonExistentField"'
      );
    }
  });
});
```

### supertest를 사용한 HTTP 통합 테스트

```typescript
// __tests__/integration/http.test.ts
import request from 'supertest';
import { createApp } from '../app';

describe('GraphQL HTTP 엔드포인트', () => {
  const app = createApp();

  it('POST 요청을 처리한다', async () => {
    const response = await request(app)
      .post('/graphql')
      .send({
        query: '{ users { id name } }',
      })
      .expect(200);

    expect(response.body.data.users).toBeInstanceOf(Array);
    expect(response.body.errors).toBeUndefined();
  });

  it('인증 헤더 없는 쿼리를 거부한다', async () => {
    const response = await request(app)
      .post('/graphql')
      .send({
        query: '{ me { id name email } }',
      })
      .expect(200); // GraphQL은 인증 오류에도 200을 반환

    expect(response.body.errors[0].extensions.code).toBe('UNAUTHENTICATED');
  });

  it('변수가 있는 뮤테이션을 처리한다', async () => {
    const response = await request(app)
      .post('/graphql')
      .set('Authorization', 'Bearer test-token')
      .send({
        query: `
          mutation CreatePost($input: CreatePostInput!) {
            createPost(input: $input) {
              id
              title
            }
          }
        `,
        variables: {
          input: { title: 'Test Post', body: 'Content here' },
        },
      })
      .expect(200);

    expect(response.body.data.createPost.title).toBe('Test Post');
  });
});
```

### 실제 데이터베이스를 사용한 테스트

```typescript
// __tests__/integration/db.test.ts
import { beforeAll, afterAll, afterEach } from 'vitest';
import { PrismaClient } from '@prisma/client';

const prisma = new PrismaClient();

beforeAll(async () => {
  // 테스트 데이터베이스 사용 — .env.test에 DATABASE_URL 설정
  await prisma.$connect();
});

afterEach(async () => {
  // 각 테스트 후 테스트 데이터 정리
  await prisma.post.deleteMany();
  await prisma.user.deleteMany();
});

afterAll(async () => {
  await prisma.$disconnect();
});

it('사용자를 생성하고 쿼리한다', async () => {
  // 테스트 데이터 시딩
  await prisma.user.create({
    data: { id: 'test-1', name: 'Test User', email: 'test@example.com' },
  });

  const response = await server.executeOperation({
    query: 'query { user(id: "test-1") { id name email } }',
  }, {
    contextValue: { dataSources: { db: prisma } },
  });

  if (response.body.kind === 'single') {
    expect(response.body.singleResult.data?.user.name).toBe('Test User');
  }
});
```

---

## 4. 스키마 테스트

스키마 테스트는 파괴적 변경이 프로덕션에 도달하기 전에 이를 감지합니다. 스키마가 안전하게 진화하는지, 팀 규약을 따르는지 검증합니다.

### graphql-inspector를 사용한 파괴적 변경 감지

```bash
# graphql-inspector 설치
npm install -D @graphql-inspector/cli

# 현재 스키마를 프로덕션 버전과 비교
npx graphql-inspector diff \
  'https://api.example.com/graphql' \
  './schema.graphql'
```

출력 예시:

```
✖ 필드 'User.email'이 제거됨   [파괴적 변경]
✖ 'Query.users'의 'limit' 인수 타입이 'Int'에서 'Int!'로 변경됨  [파괴적 변경]
✔ 필드 'User.avatar'가 추가됨    [비파괴적 변경]
✔ 타입 'Comment'이 추가됨         [비파괴적 변경]
```

### graphql-eslint를 사용한 스키마 린팅

```javascript
// .eslintrc.js
module.exports = {
  overrides: [
    {
      files: ['*.graphql'],
      parser: '@graphql-eslint/eslint-plugin',
      plugins: ['@graphql-eslint'],
      rules: {
        // 명명 규칙
        '@graphql-eslint/naming-convention': ['error', {
          types: 'PascalCase',
          FieldDefinition: 'camelCase',
          EnumValueDefinition: 'UPPER_CASE',
          InputValueDefinition: 'camelCase',
        }],
        // 타입과 필드에 설명 필수
        '@graphql-eslint/require-description': ['error', {
          types: true,
          FieldDefinition: true,
        }],
        // 뮤테이션에 입력 타입 강제
        '@graphql-eslint/input-name': ['error', {
          checkInputType: true,
        }],
        // 스키마 수준에서 과도하게 깊은 쿼리 방지
        '@graphql-eslint/no-unreachable-types': 'error',
        '@graphql-eslint/no-duplicate-fields': 'error',
      },
    },
  ],
};
```

### 프로그래밍 방식 스키마 검증

```typescript
// __tests__/schema.test.ts
import { buildSchema, validateSchema, parse, validate } from 'graphql';
import { readFileSync } from 'fs';

describe('Schema Validation', () => {
  const schemaSDL = readFileSync('./schema.graphql', 'utf-8');
  const schema = buildSchema(schemaSDL);

  it('유효성 검사 오류가 없다', () => {
    const errors = validateSchema(schema);
    expect(errors).toHaveLength(0);
  });

  it('Query 타입이 있다', () => {
    const queryType = schema.getQueryType();
    expect(queryType).toBeDefined();
  });

  it('모든 타입에 설명이 있다', () => {
    const typeMap = schema.getTypeMap();
    const userTypes = Object.entries(typeMap)
      .filter(([name]) => !name.startsWith('__')); // 인트로스펙션 타입 건너뜀

    for (const [name, type] of userTypes) {
      if (type.description === undefined || type.description === '') {
        console.warn(`타입 "${name}"에 설명이 없음`);
      }
    }
  });

  it('샘플 쿼리를 스키마에 대해 검증한다', () => {
    const query = parse(`
      query {
        user(id: "1") {
          id
          name
          email
        }
      }
    `);

    const errors = validate(schema, query);
    expect(errors).toHaveLength(0);
  });
});
```

---

## 5. 클라이언트 측 테스트

### MockedProvider (Apollo Client)

`MockedProvider`는 실제 Apollo Client를 미리 정해진 응답으로 교체하여 React 컴포넌트를 격리해서 테스트할 수 있게 합니다.

```tsx
// components/UserProfile.tsx
import { useQuery, gql } from '@apollo/client';

export const GET_USER = gql`
  query GetUser($id: ID!) {
    user(id: $id) {
      id
      name
      email
      avatar
    }
  }
`;

export function UserProfile({ userId }: { userId: string }) {
  const { data, loading, error } = useQuery(GET_USER, {
    variables: { id: userId },
  });

  if (loading) return <div data-testid="loading">Loading...</div>;
  if (error) return <div data-testid="error">Error: {error.message}</div>;

  return (
    <div data-testid="user-profile">
      <img src={data.user.avatar} alt={data.user.name} />
      <h2>{data.user.name}</h2>
      <p>{data.user.email}</p>
    </div>
  );
}
```

```tsx
// components/UserProfile.test.tsx
import { render, screen, waitFor } from '@testing-library/react';
import { MockedProvider } from '@apollo/client/testing';
import { UserProfile, GET_USER } from './UserProfile';

const mocks = [
  {
    request: {
      query: GET_USER,
      variables: { id: '42' },
    },
    result: {
      data: {
        user: {
          id: '42',
          name: 'Alice',
          email: 'alice@example.com',
          avatar: 'https://example.com/alice.jpg',
        },
      },
    },
  },
];

const errorMocks = [
  {
    request: {
      query: GET_USER,
      variables: { id: '999' },
    },
    error: new Error('User not found'),
  },
];

describe('UserProfile', () => {
  it('초기에 로딩 상태를 렌더링한다', () => {
    render(
      <MockedProvider mocks={mocks}>
        <UserProfile userId="42" />
      </MockedProvider>
    );

    expect(screen.getByTestId('loading')).toBeInTheDocument();
  });

  it('로딩 후 사용자 데이터를 렌더링한다', async () => {
    render(
      <MockedProvider mocks={mocks}>
        <UserProfile userId="42" />
      </MockedProvider>
    );

    await waitFor(() => {
      expect(screen.getByTestId('user-profile')).toBeInTheDocument();
    });

    expect(screen.getByText('Alice')).toBeInTheDocument();
    expect(screen.getByText('alice@example.com')).toBeInTheDocument();
  });

  it('오류 상태를 렌더링한다', async () => {
    render(
      <MockedProvider mocks={errorMocks}>
        <UserProfile userId="999" />
      </MockedProvider>
    );

    await waitFor(() => {
      expect(screen.getByTestId('error')).toBeInTheDocument();
    });

    expect(screen.getByText(/User not found/)).toBeInTheDocument();
  });
});
```

### MSW (Mock Service Worker)

MSW는 서비스 워커 수준에서 네트워크 요청을 가로채며, 어떤 GraphQL 클라이언트와도 동작합니다.

```typescript
// mocks/handlers.ts
import { graphql, HttpResponse } from 'msw';

export const handlers = [
  // 작업 이름으로 GraphQL 쿼리 모킹
  graphql.query('GetUser', ({ variables }) => {
    const { id } = variables;

    if (id === '42') {
      return HttpResponse.json({
        data: {
          user: { id: '42', name: 'Alice', email: 'alice@example.com' },
        },
      });
    }

    return HttpResponse.json({
      errors: [{ message: 'User not found' }],
    });
  }),

  // GraphQL 뮤테이션 모킹
  graphql.mutation('CreatePost', ({ variables }) => {
    return HttpResponse.json({
      data: {
        createPost: {
          id: 'new-post-1',
          title: variables.input.title,
        },
      },
    });
  }),
];
```

```typescript
// mocks/server.ts
import { setupServer } from 'msw/node';
import { handlers } from './handlers';

export const server = setupServer(...handlers);
```

```typescript
// vitest.setup.ts
import { beforeAll, afterEach, afterAll } from 'vitest';
import { server } from './mocks/server';

beforeAll(() => server.listen());
afterEach(() => server.resetHandlers());
afterAll(() => server.close());
```

---

## 6. 스냅샷 테스트

스냅샷 테스트는 GraphQL 응답의 형태를 캡처하고 예상치 않게 변경될 때 알려줍니다.

### 쿼리 결과 스냅샷

```typescript
// __tests__/snapshots/queries.test.ts
import { describe, it, expect } from 'vitest';

describe('Query Snapshots', () => {
  it('GetUser가 예상된 형태를 반환한다', async () => {
    const response = await server.executeOperation({
      query: `
        query GetUser($id: ID!) {
          user(id: $id) {
            id
            name
            email
            posts {
              id
              title
              createdAt
            }
          }
        }
      `,
      variables: { id: 'test-user-1' },
    }, {
      contextValue: createTestContext(),
    });

    if (response.body.kind === 'single') {
      // 스냅샷이 전체 응답 구조를 캡처
      // 형태가 변경되면 테스트가 실패하고 차이점을 보여줌
      expect(response.body.singleResult.data).toMatchSnapshot();
    }
  });
});
```

### 스키마 스냅샷

```typescript
// __tests__/snapshots/schema.test.ts
import { printSchema } from 'graphql';
import { schema } from '../schema';

describe('Schema Snapshot', () => {
  it('현재 스키마와 일치한다', () => {
    // 의도하지 않은 스키마 변경을 감지
    expect(printSchema(schema)).toMatchSnapshot();
  });
});
```

### 인트로스펙션(Introspection) 스냅샷

```typescript
import { introspectionQuery, graphqlSync } from 'graphql';

it('인트로스펙션 결과가 안정적이다', () => {
  const result = graphqlSync({ schema, source: introspectionQuery });
  expect(result).toMatchSnapshot();
});
```

---

## 7. CI 통합

### GitHub Actions 파이프라인

```yaml
# .github/workflows/graphql-tests.yml
name: GraphQL Tests
on:
  pull_request:
    paths:
      - 'src/**'
      - 'schema.graphql'

jobs:
  test:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: postgres:16
        env:
          POSTGRES_DB: test
          POSTGRES_PASSWORD: test
        ports:
          - 5432:5432
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5

    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-node@v4
        with:
          node-version: '20'
          cache: 'npm'

      - run: npm ci

      # 스키마 린팅
      - name: Lint Schema
        run: npx eslint '**/*.graphql'

      # 파괴적 변경 감지
      - name: Check Schema Changes
        run: |
          npx graphql-inspector diff \
            'https://api.production.example.com/graphql' \
            './schema.graphql'

      # 단위 및 통합 테스트
      - name: Run Tests
        run: npm test -- --coverage
        env:
          DATABASE_URL: postgresql://postgres:test@localhost:5432/test

      # 커버리지 업로드
      - uses: codecov/codecov-action@v4
        with:
          files: coverage/lcov.info

  schema-check:
    runs-on: ubuntu-latest
    if: contains(github.event.pull_request.changed_files, 'schema.graphql')
    steps:
      - uses: actions/checkout@v4
      - name: Install Rover
        run: |
          curl -sSL https://rover.apollo.dev/nix/latest | sh
          echo "$HOME/.rover/bin" >> $GITHUB_PATH
      - name: Check Schema
        run: |
          rover subgraph check my-graph@production \
            --name my-service \
            --schema ./schema.graphql
        env:
          APOLLO_KEY: ${{ secrets.APOLLO_KEY }}
```

### 테스트 구성

```
__tests__/
├── unit/
│   ├── resolvers/
│   │   ├── user.test.ts
│   │   ├── post.test.ts
│   │   └── admin.test.ts
│   └── utils/
│       └── auth.test.ts
├── integration/
│   ├── queries.test.ts
│   ├── mutations.test.ts
│   └── subscriptions.test.ts
├── schema/
│   ├── breaking-changes.test.ts
│   ├── lint.test.ts
│   └── snapshot.test.ts
├── e2e/
│   └── user-flow.test.ts
└── mocks/
    ├── handlers.ts
    ├── server.ts
    └── fixtures/
        ├── users.json
        └── posts.json
```

---

## 8. 연습 문제

### 문제 1: 리졸버 단위 테스트
다음 리졸버에 대한 종합적인 단위 테스트를 작성하세요. 성공 케이스, 오류 케이스, 엣지 케이스(빈 결과, 누락된 필드)를 모두 다루세요:

```typescript
const resolvers = {
  Query: {
    searchProducts: async (_, { query, minPrice, maxPrice }, { dataSources }) => {
      if (!query || query.length < 2) {
        throw new GraphQLError('Search query must be at least 2 characters');
      }
      return dataSources.products.search({ query, minPrice, maxPrice });
    },
  },
  Product: {
    isOnSale: (product) => product.salePrice < product.originalPrice,
    discount: (product) => {
      if (product.salePrice >= product.originalPrice) return 0;
      return Math.round((1 - product.salePrice / product.originalPrice) * 100);
    },
  },
};
```

### 문제 2: 통합 테스트 스위트
다음 작업들을 갖춘 블로그 API에 대한 통합 테스트 스위트를 작성하세요: `createPost`(뮤테이션), `getPost`(쿼리), `updatePost`(뮤테이션), `deletePost`(뮤테이션). `executeOperation`을 사용하고 CRUD 작업이 엔드-투-엔드로 동작하는지 검증하세요. 인증 테스트도 포함하세요(작성자만 자신의 게시물을 수정/삭제할 수 있음).

### 문제 3: MockedProvider 컴포넌트 테스트
`useQuery`로 상품을 가져오고 `useMutation`으로 장바구니에 항목을 추가하는 `ProductList` React 컴포넌트의 테스트를 작성하세요. 테스트 항목: (a) 로딩 상태, (b) 오류 상태, (c) 상품 렌더링, (d) 장바구니에 추가하면 UI가 낙관적으로 업데이트됨.

### 문제 4: 스키마 진화
`graphql-inspector`를 사용하여 두 스키마 파일(`schema-v1.graphql`과 `schema-v2.graphql`)을 읽고 모든 파괴적 변경 대 비파괴적 변경을 보고하는 테스트를 만드세요. 파괴적 변경이 감지되면 실패하는 테스트를 작성하세요.

### 문제 5: CI 파이프라인 설계
다음을 실행하는 완전한 CI 파이프라인 구성(GitHub Actions YAML)을 설계하세요: (a) 스키마 린팅, (b) 프로덕션 대비 파괴적 변경 감지, (c) 단위 테스트, (d) PostgreSQL 서비스 컨테이너를 사용한 통합 테스트, (e) 연합 그래프에 대한 스키마 구성 검사. node_modules 캐싱과 병렬 잡 실행을 포함하세요.

---

## 참고 자료

- [Apollo Server 테스팅](https://www.apollographql.com/docs/apollo-server/testing/testing/)
- [Apollo Client MockedProvider](https://www.apollographql.com/docs/react/development-testing/testing/)
- [Mock Service Worker (MSW)](https://mswjs.io/docs/network-behavior/graphql)
- [graphql-inspector](https://the-guild.dev/graphql/inspector)
- [graphql-eslint](https://the-guild.dev/graphql/eslint/docs)
- [GraphQL API 테스팅 (The Guild)](https://the-guild.dev/blog/testing-graphql-apis)

---

**이전**: [연합(Federation)](./12_Federation.md) | **다음**: [성능과 보안](./14_Performance_Security.md)
