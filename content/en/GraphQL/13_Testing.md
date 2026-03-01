# 13. Testing

**Previous**: [Federation](./12_Federation.md) | **Next**: [Performance and Security](./14_Performance_Security.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Design a comprehensive testing strategy for GraphQL APIs spanning unit, integration, and end-to-end layers
2. Write unit tests for resolvers with mocked context and data sources
3. Perform integration tests using `executeOperation` and HTTP test clients
4. Validate schema evolution with breaking-change detection and schema linting
5. Test GraphQL clients using MockedProvider and request mocking libraries

## Table of Contents

1. [GraphQL Testing Strategy](#1-graphql-testing-strategy)
2. [Unit Testing Resolvers](#2-unit-testing-resolvers)
3. [Integration Testing](#3-integration-testing)
4. [Schema Testing](#4-schema-testing)
5. [Client-Side Testing](#5-client-side-testing)
6. [Snapshot Testing](#6-snapshot-testing)
7. [CI Integration](#7-ci-integration)
8. [Practice Problems](#8-practice-problems)

**Difficulty**: ⭐⭐⭐

---

Testing GraphQL is different from testing REST. A single endpoint serves infinite query shapes, resolvers form a chain where one bug can cascade, and the schema itself is a contract that must not break unexpectedly. This lesson builds a testing strategy from the ground up -- starting with isolated resolver unit tests, moving through full server integration tests, and finishing with client-side mocking and CI pipeline integration.

---

## 1. GraphQL Testing Strategy

### The Testing Pyramid for GraphQL

```
           ┌───────┐
           │  E2E  │   Cypress / Playwright
           │ Tests │   Full stack, slow, few
           ├───────┤
           │       │
        ┌──┤ Integ.├──┐   executeOperation, HTTP
        │  │ Tests │  │   Server + resolvers + data
        │  ├───────┤  │
        │  │       │  │
     ┌──┤  │ Unit  │  ├──┐   Resolver functions
     │  │  │ Tests │  │  │   Mocked context/data
     │  │  ├───────┤  │  │
     │  │  │       │  │  │
  ┌──┤  │  │Schema │  │  ├──┐   Linting, breaking changes
  │  │  │  │ Tests │  │  │  │   Composition validation
  └──┴──┴──┴───────┴──┴──┴──┘
```

### What to Test at Each Layer

| Layer | What to Test | Tools | Speed |
|-------|-------------|-------|-------|
| Schema | Breaking changes, linting rules, composition | graphql-inspector, Rover | Fast |
| Unit | Individual resolver logic, business rules | Jest, Vitest | Fast |
| Integration | Full query execution, resolver chain, auth | executeOperation, supertest | Medium |
| E2E | Client-to-server flow, mutations with side effects | Cypress, Playwright | Slow |

---

## 2. Unit Testing Resolvers

Resolvers are plain functions. Testing them means calling the function directly with mocked arguments, context, and data sources.

### Basic Resolver Test

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
  // Mock data sources -- isolate resolver logic from database
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
    it('returns a user by ID', async () => {
      const expectedUser = { id: '1', firstName: 'Alice', lastName: 'Smith' };
      mockDataSources.users.findById.mockResolvedValue(expectedUser);

      const result = await userResolvers.Query.user(
        {},           // parent (root)
        { id: '1' },  // args
        mockContext,   // context
      );

      expect(result).toEqual(expectedUser);
      expect(mockDataSources.users.findById).toHaveBeenCalledWith('1');
    });

    it('returns null for non-existent user', async () => {
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
    it('combines first and last name', () => {
      const parent = { firstName: 'Alice', lastName: 'Smith' };
      const result = userResolvers.User.fullName(parent);
      expect(result).toBe('Alice Smith');
    });
  });

  describe('User.posts', () => {
    it('loads posts for the parent user', async () => {
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

### Testing Authorization Logic

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
  it('rejects unauthenticated requests', async () => {
    const context = { userId: null, dataSources: mockDataSources };

    await expect(
      adminResolvers.Query.adminDashboard({}, {}, context)
    ).rejects.toThrow('Authentication required');
  });

  it('rejects non-admin users', async () => {
    mockDataSources.users.findById.mockResolvedValue({ role: 'USER' });
    const context = { userId: 'user-1', dataSources: mockDataSources };

    await expect(
      adminResolvers.Query.adminDashboard({}, {}, context)
    ).rejects.toThrow('Admin access required');
  });

  it('returns dashboard for admin users', async () => {
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

## 3. Integration Testing

Integration tests execute real GraphQL operations against your server, verifying the complete resolver chain, middleware, and error handling.

### Using executeOperation (Apollo Server 4)

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

  it('queries a user by ID', async () => {
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
      // Provide context value for the operation
      contextValue: {
        dataSources: createTestDataSources(),
        userId: 'test-user',
      },
    });

    // Assert on the response
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

  it('returns validation error for invalid query', async () => {
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

### HTTP Integration Tests with supertest

```typescript
// __tests__/integration/http.test.ts
import request from 'supertest';
import { createApp } from '../app';

describe('GraphQL HTTP Endpoint', () => {
  const app = createApp();

  it('handles POST requests', async () => {
    const response = await request(app)
      .post('/graphql')
      .send({
        query: '{ users { id name } }',
      })
      .expect(200);

    expect(response.body.data.users).toBeInstanceOf(Array);
    expect(response.body.errors).toBeUndefined();
  });

  it('rejects queries without authentication header', async () => {
    const response = await request(app)
      .post('/graphql')
      .send({
        query: '{ me { id name email } }',
      })
      .expect(200); // GraphQL returns 200 even for auth errors

    expect(response.body.errors[0].extensions.code).toBe('UNAUTHENTICATED');
  });

  it('handles mutations with variables', async () => {
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

### Testing with a Real Database

```typescript
// __tests__/integration/db.test.ts
import { beforeAll, afterAll, afterEach } from 'vitest';
import { PrismaClient } from '@prisma/client';

const prisma = new PrismaClient();

beforeAll(async () => {
  // Use a test database -- set DATABASE_URL in .env.test
  await prisma.$connect();
});

afterEach(async () => {
  // Clean up test data after each test
  await prisma.post.deleteMany();
  await prisma.user.deleteMany();
});

afterAll(async () => {
  await prisma.$disconnect();
});

it('creates a user and queries it', async () => {
  // Seed test data
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

## 4. Schema Testing

Schema tests catch breaking changes before they reach production. They validate that the schema evolves safely and follows team conventions.

### Breaking Change Detection with graphql-inspector

```bash
# Install graphql-inspector
npm install -D @graphql-inspector/cli

# Compare current schema against the production version
npx graphql-inspector diff \
  'https://api.example.com/graphql' \
  './schema.graphql'
```

Output example:

```
✖ Field 'User.email' was removed   [BREAKING]
✖ Argument 'limit' on 'Query.users' changed type from 'Int' to 'Int!'  [BREAKING]
✔ Field 'User.avatar' was added    [NON-BREAKING]
✔ Type 'Comment' was added         [NON-BREAKING]
```

### Schema Linting with graphql-eslint

```javascript
// .eslintrc.js
module.exports = {
  overrides: [
    {
      files: ['*.graphql'],
      parser: '@graphql-eslint/eslint-plugin',
      plugins: ['@graphql-eslint'],
      rules: {
        // Naming conventions
        '@graphql-eslint/naming-convention': ['error', {
          types: 'PascalCase',
          FieldDefinition: 'camelCase',
          EnumValueDefinition: 'UPPER_CASE',
          InputValueDefinition: 'camelCase',
        }],
        // Require descriptions on types and fields
        '@graphql-eslint/require-description': ['error', {
          types: true,
          FieldDefinition: true,
        }],
        // Enforce input types for mutations
        '@graphql-eslint/input-name': ['error', {
          checkInputType: true,
        }],
        // Prevent overly deep queries at the schema level
        '@graphql-eslint/no-unreachable-types': 'error',
        '@graphql-eslint/no-duplicate-fields': 'error',
      },
    },
  ],
};
```

### Programmatic Schema Validation

```typescript
// __tests__/schema.test.ts
import { buildSchema, validateSchema, parse, validate } from 'graphql';
import { readFileSync } from 'fs';

describe('Schema Validation', () => {
  const schemaSDL = readFileSync('./schema.graphql', 'utf-8');
  const schema = buildSchema(schemaSDL);

  it('has no validation errors', () => {
    const errors = validateSchema(schema);
    expect(errors).toHaveLength(0);
  });

  it('has a Query type', () => {
    const queryType = schema.getQueryType();
    expect(queryType).toBeDefined();
  });

  it('all types have descriptions', () => {
    const typeMap = schema.getTypeMap();
    const userTypes = Object.entries(typeMap)
      .filter(([name]) => !name.startsWith('__')); // Skip introspection types

    for (const [name, type] of userTypes) {
      if (type.description === undefined || type.description === '') {
        console.warn(`Type "${name}" is missing a description`);
      }
    }
  });

  it('validates a sample query against the schema', () => {
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

## 5. Client-Side Testing

### MockedProvider (Apollo Client)

`MockedProvider` replaces the real Apollo Client with predetermined responses, letting you test React components in isolation.

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
  it('renders loading state initially', () => {
    render(
      <MockedProvider mocks={mocks}>
        <UserProfile userId="42" />
      </MockedProvider>
    );

    expect(screen.getByTestId('loading')).toBeInTheDocument();
  });

  it('renders user data after loading', async () => {
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

  it('renders error state', async () => {
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

MSW intercepts network requests at the service worker level, working with any GraphQL client.

```typescript
// mocks/handlers.ts
import { graphql, HttpResponse } from 'msw';

export const handlers = [
  // Mock a GraphQL query by operation name
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

  // Mock a GraphQL mutation
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

## 6. Snapshot Testing

Snapshot testing captures the shape of GraphQL responses and alerts you when they change unexpectedly.

### Query Result Snapshots

```typescript
// __tests__/snapshots/queries.test.ts
import { describe, it, expect } from 'vitest';

describe('Query Snapshots', () => {
  it('GetUser returns expected shape', async () => {
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
      // Snapshot captures the full response structure
      // If the shape changes, the test fails and shows a diff
      expect(response.body.singleResult.data).toMatchSnapshot();
    }
  });
});
```

### Schema Snapshot

```typescript
// __tests__/snapshots/schema.test.ts
import { printSchema } from 'graphql';
import { schema } from '../schema';

describe('Schema Snapshot', () => {
  it('matches the current schema', () => {
    // Catches unintentional schema changes
    expect(printSchema(schema)).toMatchSnapshot();
  });
});
```

### Introspection Snapshot

```typescript
import { introspectionQuery, graphqlSync } from 'graphql';

it('introspection result is stable', () => {
  const result = graphqlSync({ schema, source: introspectionQuery });
  expect(result).toMatchSnapshot();
});
```

---

## 7. CI Integration

### GitHub Actions Pipeline

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

      # Schema linting
      - name: Lint Schema
        run: npx eslint '**/*.graphql'

      # Breaking change detection
      - name: Check Schema Changes
        run: |
          npx graphql-inspector diff \
            'https://api.production.example.com/graphql' \
            './schema.graphql'

      # Unit and integration tests
      - name: Run Tests
        run: npm test -- --coverage
        env:
          DATABASE_URL: postgresql://postgres:test@localhost:5432/test

      # Upload coverage
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

### Test Organization

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

## 8. Practice Problems

### Problem 1: Resolver Unit Tests
Write comprehensive unit tests for the following resolver. Cover success cases, error cases, and edge cases (empty results, missing fields):

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

### Problem 2: Integration Test Suite
Create an integration test suite for a blog API with these operations: `createPost` (mutation), `getPost` (query), `updatePost` (mutation), `deletePost` (mutation). Use `executeOperation` and verify that CRUD operations work end-to-end. Include tests for authorization (only the author can update/delete their posts).

### Problem 3: MockedProvider Component Test
Write tests for a `ProductList` React component that uses `useQuery` to fetch products and `useMutation` to add items to a cart. Test: (a) loading state, (b) error state, (c) rendering products, (d) adding to cart updates the UI optimistically.

### Problem 4: Schema Evolution
Using `graphql-inspector`, create a test that reads two schema files (`schema-v1.graphql` and `schema-v2.graphql`) and reports all breaking vs. non-breaking changes. Write a test that fails if any breaking changes are detected.

### Problem 5: CI Pipeline Design
Design a complete CI pipeline configuration (GitHub Actions YAML) that runs: (a) schema linting, (b) breaking change detection against production, (c) unit tests, (d) integration tests with a PostgreSQL service container, (e) schema composition check for a federated graph. Include caching for node_modules and parallel job execution.

---

## References

- [Apollo Server Testing](https://www.apollographql.com/docs/apollo-server/testing/testing/)
- [Apollo Client MockedProvider](https://www.apollographql.com/docs/react/development-testing/testing/)
- [Mock Service Worker (MSW)](https://mswjs.io/docs/network-behavior/graphql)
- [graphql-inspector](https://the-guild.dev/graphql/inspector)
- [graphql-eslint](https://the-guild.dev/graphql/eslint/docs)
- [Testing GraphQL APIs (The Guild)](https://the-guild.dev/blog/testing-graphql-apis)

---

**Previous**: [Federation](./12_Federation.md) | **Next**: [Performance and Security](./14_Performance_Security.md)
