# 09. Express Testing

**Previous**: [Express Database](./08_Express_Database.md) | **Next**: [Django Basics](./10_Django_Basics.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Configure Jest and Supertest for testing Express applications
2. Write unit tests for route handlers and middleware in isolation
3. Mock Prisma database calls to test business logic without a real database
4. Design integration tests that exercise the full request-response cycle against a test database
5. Measure and interpret code coverage reports using Jest's built-in tooling

---

Tests are the safety net that lets you refactor, add features, and fix bugs with confidence. Without tests, every change risks breaking something silently. This lesson covers the full testing spectrum for Express applications: unit tests for individual functions, integration tests for HTTP endpoints, mocking strategies for database calls, and code coverage analysis. We use Jest as the test runner and Supertest for making HTTP assertions.

## Table of Contents

Before the framework reference, read [**Theory & Principles**](#theory--principles) — how Supertest mounts an Express app on an ephemeral port (and why that beats real servers), what makes Jest and Mocha philosophically different, and the test-double taxonomy (mocks vs stubs vs spies vs fakes) that explains every mocking decision.

1. [Testing with Jest and Supertest](#1-testing-with-jest-and-supertest)
2. [Setting Up the Test Environment](#2-setting-up-the-test-environment)
3. [Testing Routes](#3-testing-routes)
4. [Testing Middleware](#4-testing-middleware)
5. [Mocking Database Calls](#5-mocking-database-calls)
6. [Integration Tests with Test Database](#6-integration-tests-with-test-database)
7. [Testing Authentication](#7-testing-authentication)
8. [Code Coverage with Jest](#8-code-coverage-with-jest)
9. [Practice Problems](#9-practice-problems)

---

## Theory & Principles

A test suite for an Express application is built from three independent pieces, each with a deeper mechanism than its API suggests.

- **(A) HTTP test transports** — what Supertest does at the socket level, and the alternative of skipping the network entirely.
- **(B) Jest vs Mocha — different philosophies** — Jest's built-in batteries vs Mocha's à la carte composition, and why the choice colors every other testing decision.
- **(C) Test doubles taxonomy** — mocks, stubs, spies, fakes are not synonyms; each addresses a specific design question.

### A. HTTP Test Transports

Lesson 05 §A established that FastAPI's `TestClient` skips the network entirely by speaking ASGI in-process. Express has no equivalent of ASGI — Node's `http` module is intrinsically socket-based. Supertest works around this in a clever way.

#### A.1 What Supertest actually does

Supertest takes your `app` (an Express instance, which is a function `(req, res) => void`) and:

1. Creates an HTTP server with `http.createServer(app)`.
2. Calls `server.listen(0)` — the `0` tells the OS to assign any free port.
3. Makes the request to `http://127.0.0.1:<assigned-port>/...`.
4. Closes the server when the test promise resolves.

The whole exchange runs through the real Node HTTP stack. Headers are parsed by `http_parser`, the request goes through `net.Socket`, the response chunked encoding is handled by Node. From the application's perspective, the test is *indistinguishable* from a real client.

#### A.2 Why a real socket — and what it costs

The cost: one syscall per test for `listen()`, one per response for `close()`, plus actual TCP socket I/O. Per-test overhead is around 1-2 ms on Linux. That is fine for hundreds of tests; for tens of thousands you may notice.

The benefit: you exercise the full Node HTTP stack. Anything that depends on header parsing, timeouts, request body chunking, or kernel-level back-pressure works naturally. There is no "in-process" gotcha that breaks under load.

#### A.3 The alternative: directly invoking the app

Express `app` is callable: `app(req, res)`. With manually constructed `req`/`res` mocks (e.g., `node-mocks-http`), you can invoke the app without a server:

```javascript
const req = httpMocks.createRequest({ method: "GET", url: "/" });
const res = httpMocks.createResponse();
app(req, res);
```

Faster and synchronous, but `req`/`res` are mocks — they do not parse headers from raw bytes, do not stream, do not back-pressure. Acceptable for unit-testing a single middleware in isolation; not acceptable for integration tests of the full request pipeline. Supertest is the right default.

### B. Jest vs Mocha: Two Philosophies

The two dominant Node test frameworks differ in foundational design. Picking one shapes every other testing decision.

#### B.1 Jest: batteries included

Jest is a Meta project that bundles the test runner, assertion library, mocking utilities, snapshot testing, code coverage, and parallel test execution. One install, one config, one CLI. Defaults are opinionated:

- Tests run in *isolated worker processes* — each test file gets its own Node process, so module state cannot leak between files.
- Mocking is built in: `jest.mock(...)` rewires module imports, `jest.fn()` creates spies.
- Coverage is one CLI flag away: `jest --coverage` produces a full report.
- Snapshot testing — capture an output once, fail if it ever changes.

The opinionated defaults are also Jest's downsides: the worker-isolation strategy doubles startup cost, the module mocking happens at *module-load* time (not call time), which can confuse intuition.

#### B.2 Mocha: bring your own

Mocha is just a test runner: `describe/it`, lifecycle hooks, parallel mode. Everything else — assertions (`chai`, `expect`, native `assert`), mocking (`sinon`), coverage (`c8`, `nyc`) — is a separate library you bolt on. The cost is more setup; the benefit is exact control over each piece.

Mocha tests share a process by default, which is faster for small suites but means modules with side effects can leak across files unless you clean up.

#### B.3 The native alternative

Modern Node 20+ ships a built-in test runner (`node --test`) and assertion module (`node:test`). It is API-compatible with neither Jest nor Mocha but is fast and zero-dependency. For greenfield projects without complex mocking needs, it is increasingly viable.

The decision matrix: **Jest** for most apps (Express included), **Mocha** when you need fine-grained control or a smaller dependency footprint, **node:test** for libraries and small services.

### C. The Test Doubles Taxonomy

"Mock the database" is shorthand for several different patterns. Gerard Meszaros's classification (xUnit Test Patterns) names them precisely:

| Double | Used to | Verifies |
|--------|---------|----------|
| **Dummy** | Fill a parameter that won't be used | Nothing |
| **Stub** | Return canned data when called | Nothing |
| **Spy** | Record calls so you can inspect them later | Calls (after the fact) |
| **Mock** | Pre-program expected calls; fail if calls deviate | Calls (during the test) |
| **Fake** | Lightweight working implementation | Behavior |

Each addresses a different design question: a stub answers "what does the code under test do when X returns Y?"; a mock answers "does the code under test call X with Z?".

#### C.1 Stubs are usually what you want

Most database tests are stubs in disguise:

```javascript
prisma.user.findUnique = jest.fn().mockResolvedValue({ id: 42, name: "Alice" });
```

This is a stub: it returns canned data, no expectations on whether or how often it is called. The test focuses on the handler's behavior given that data. Simple, robust, mostly immune to internal refactors.

#### C.2 Mocks are coupling

A mock asserts the exact API contract:

```javascript
expect(prisma.user.findUnique).toHaveBeenCalledWith({ where: { id: 42 } });
```

That assertion fails if the handler is refactored to call `findFirst` or to use a different `where` shape — even if the *behavior* is identical. Mocks couple your tests to implementation details. Use them sparingly, mostly for cross-boundary calls (analytics events, queued jobs, external API calls) where the call is itself the side effect under test.

#### C.3 Fakes are the gold standard for databases

The richest substitute is a fake: a real working implementation, just lighter. For databases, this means a real SQLite/PostgreSQL instance with the test schema — populated with deterministic seed data. Tests run against real SQL, real transactions, real constraints. The bug-finding power of fakes far exceeds stubs.

The tradeoff is setup cost. The pattern from Lesson 05 §C.1 — transactional rollback per test against a session-scoped database — is the modern compromise: real database fidelity with stub-tier per-test cost.

#### C.4 The right combination

Most production test suites blend levels:

- **Unit tests** — stubs for everything not under test. Fast, focused.
- **Integration tests** — real database (fake-style), real HTTP via Supertest, mocks only for genuinely external services (Stripe, S3, third-party APIs).
- **End-to-end** — a real running stack, no doubles. Slow and rare.

The classification matters because each layer answers a different question: unit answers "does this function compute the right thing?", integration answers "does this endpoint talk to its dependencies correctly?", end-to-end answers "does the whole system work?"

### From Theory to the Code Below

Each section that follows operationalizes one piece of this framework:

- §1 (Jest and Supertest) introduces the §B.1 batteries-included runner and the §A.1 ephemeral-port HTTP transport.
- §2 (Setting up) configures Jest's worker isolation (§B.1) and the database from Lesson 08 for the test environment.
- §3 (Testing routes) is integration testing via Supertest — §A.1 in concrete code.
- §4 (Testing middleware) often uses the §A.3 direct-invoke pattern with mocked `req`/`res` for unit isolation.
- §5 (Mocking database calls) is mostly §C.1 stubs (sometimes §C.2 mocks) on Prisma methods.
- §6 (Integration tests with test database) is the §C.3 fake pattern: a real database with the §C.3 transactional-rollback isolation from Lesson 05 §C.1.
- §7 (Testing authentication) chains "login → use token" via Supertest — pure §A flow.
- §8 (Coverage) is the §B.1 built-in coverage report measuring which code the §A–§C suite actually exercises.

---

## 1. Testing with Jest and Supertest

### Why Jest?

Jest is a batteries-included test framework from Meta that provides a test runner, assertion library, mocking utilities, and code coverage — all in one package.

### Why Supertest?

Supertest lets you make HTTP requests against an Express app **without starting the server**. It binds to an ephemeral port internally, which means tests run fast and do not conflict with other services.

```
┌────────────────────────────────────────────────────────┐
│                    Test Architecture                   │
│                                                        │
│  Test File                                             │
│  ┌──────────────────────┐                              │
│  │ describe('GET /api') │                              │
│  │   it('returns 200')  │                              │
│  │     request(app)     │───▶ Express App (no listen)  │
│  │       .get('/api')   │                              │
│  │       .expect(200)   │◀── Response assertions       │
│  └──────────────────────┘                              │
│                                                        │
│  Supertest creates an internal HTTP server for each    │
│  test — no port conflicts, no server lifecycle mgmt    │
└────────────────────────────────────────────────────────┘
```

### Installation

```bash
npm install -D jest @jest/globals supertest

# For ES modules support — Jest needs experimental VM modules
# because Node's native ESM and Jest's module system interact poorly
```

---

## 2. Setting Up the Test Environment

### Jest Configuration

```javascript
// jest.config.js
export default {
  // Use the experimental ESM loader — required for import/export syntax
  transform: {},
  testEnvironment: 'node',
  // Match test files by convention: *.test.js or files in __tests__/
  testMatch: ['**/__tests__/**/*.js', '**/*.test.js'],
  // Ignore build artifacts and dependencies
  testPathIgnorePatterns: ['/node_modules/', '/dist/'],
  // Collect coverage from source files, excluding test utilities
  collectCoverageFrom: [
    'src/**/*.js',
    '!src/server.js',       // Exclude server startup (side effects)
    '!src/lib/prisma.js',   // Exclude Prisma singleton
  ],
};
```

### Package.json Scripts

```json
{
  "scripts": {
    "test": "NODE_OPTIONS='--experimental-vm-modules' jest",
    "test:watch": "NODE_OPTIONS='--experimental-vm-modules' jest --watch",
    "test:coverage": "NODE_OPTIONS='--experimental-vm-modules' jest --coverage"
  }
}
```

### App/Server Separation

Separating the Express app from the server startup is critical for testing. Supertest needs the app object, not a running server.

```javascript
// src/app.js — exports the configured Express app (no listen call)
import express from 'express';
import usersRouter from './routes/users.js';

const app = express();
app.use(express.json());
app.use('/api/users', usersRouter);

app.use((req, res) => {
  res.status(404).json({ error: 'Not found' });
});

app.use((err, req, res, next) => {
  res.status(err.statusCode || 500).json({ error: { message: err.message } });
});

export default app;
```

```javascript
// src/server.js — starts the server; only run directly, never imported by tests
import app from './app.js';
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`Listening on port ${PORT}`));
```

---

## 3. Testing Routes

### Basic Route Tests

```javascript
// __tests__/routes/users.test.js
import { jest } from '@jest/globals';
import request from 'supertest';
import app from '../../src/app.js';

describe('GET /api/users', () => {
  it('should return 200 and an array of users', async () => {
    const response = await request(app)
      .get('/api/users')
      .expect('Content-Type', /json/)
      .expect(200);

    // Verify response structure — tests the contract, not implementation details
    expect(response.body).toHaveProperty('data');
    expect(Array.isArray(response.body.data)).toBe(true);
  });

  it('should support pagination query parameters', async () => {
    const response = await request(app)
      .get('/api/users?page=1&limit=5')
      .expect(200);

    expect(response.body.pagination).toEqual(
      expect.objectContaining({
        page: 1,
        limit: 5,
      })
    );
  });
});

describe('POST /api/users', () => {
  it('should create a user and return 201', async () => {
    const newUser = { name: 'Alice', email: 'alice@example.com' };

    const response = await request(app)
      .post('/api/users')
      .send(newUser)      // .send() sets Content-Type to application/json
      .expect(201);

    expect(response.body).toMatchObject({
      name: 'Alice',
      email: 'alice@example.com',
    });
    // Use toMatchObject for partial matching — the response may include
    // fields like id and createdAt that we do not want to hard-code
    expect(response.body).toHaveProperty('id');
  });

  it('should return 400 for missing required fields', async () => {
    const response = await request(app)
      .post('/api/users')
      .send({}) // Missing name and email
      .expect(400);

    expect(response.body).toHaveProperty('error');
  });
});
```

### Testing Route Parameters

```javascript
describe('GET /api/users/:id', () => {
  it('should return a single user by ID', async () => {
    const response = await request(app)
      .get('/api/users/1')
      .expect(200);

    expect(response.body).toHaveProperty('id', 1);
  });

  it('should return 404 for non-existent user', async () => {
    await request(app)
      .get('/api/users/99999')
      .expect(404);
  });
});

describe('DELETE /api/users/:id', () => {
  it('should return 204 for successful deletion', async () => {
    await request(app)
      .delete('/api/users/1')
      .expect(204);
  });
});
```

---

## 4. Testing Middleware

Middleware can be tested in two ways: in isolation (unit) or through routes (integration).

### Unit Testing Middleware

```javascript
// __tests__/middleware/auth.test.js
import { jest } from '@jest/globals';

// Test the middleware function directly — no HTTP requests needed
describe('requireAuth middleware', () => {
  let req, res, next;

  beforeEach(() => {
    // Create mock request, response, and next objects
    // This isolates the middleware from Express internals
    req = {
      headers: {},
      get: jest.fn((header) => req.headers[header.toLowerCase()]),
    };
    res = {
      status: jest.fn().mockReturnThis(), // Chainable — status().json() pattern
      json: jest.fn().mockReturnThis(),
    };
    next = jest.fn();
  });

  it('should call next() when valid token is provided', async () => {
    req.headers.authorization = 'Bearer valid-token-here';

    // Import after mocks are set up to ensure mocks take effect
    const { requireAuth } = await import('../../src/middleware/auth.js');
    await requireAuth(req, res, next);

    expect(next).toHaveBeenCalled();
    expect(res.status).not.toHaveBeenCalled();
  });

  it('should return 401 when no token is provided', async () => {
    const { requireAuth } = await import('../../src/middleware/auth.js');
    await requireAuth(req, res, next);

    expect(res.status).toHaveBeenCalledWith(401);
    expect(res.json).toHaveBeenCalledWith(
      expect.objectContaining({ error: expect.any(String) })
    );
    // next() should NOT be called — the request chain stops here
    expect(next).not.toHaveBeenCalled();
  });
});
```

### Integration Testing Middleware via Routes

```javascript
// Test middleware behavior through the routes it protects
describe('Rate limiting middleware', () => {
  it('should return 429 after exceeding limit', async () => {
    // Send requests up to the limit
    for (let i = 0; i < 5; i++) {
      await request(app).post('/api/auth/login').send({
        email: 'test@example.com',
        password: 'wrong',
      });
    }

    // The 6th request should be rate-limited
    const response = await request(app)
      .post('/api/auth/login')
      .send({ email: 'test@example.com', password: 'wrong' })
      .expect(429);

    expect(response.body.error).toMatch(/too many/i);
  });
});
```

---

## 5. Mocking Database Calls

Mocking the database lets you test route logic without a running database. This makes tests faster, more deterministic, and independent of external state.

### Mocking Prisma Client

```javascript
// __tests__/mocks/prisma.js
// Create a mock that replaces every Prisma model method with a Jest mock function
import { jest } from '@jest/globals';

const prismaMock = {
  user: {
    findMany: jest.fn(),
    findUnique: jest.fn(),
    create: jest.fn(),
    update: jest.fn(),
    delete: jest.fn(),
    count: jest.fn(),
  },
  post: {
    findMany: jest.fn(),
    findUnique: jest.fn(),
    create: jest.fn(),
    update: jest.fn(),
    delete: jest.fn(),
  },
  $transaction: jest.fn(),
  $disconnect: jest.fn(),
};

export default prismaMock;
```

### Using Mocks in Tests

```javascript
// __tests__/routes/users.test.js
import { jest, beforeEach, describe, it, expect } from '@jest/globals';

// Mock the Prisma module — all imports of prisma.js will get the mock instead
jest.unstable_mockModule('../../src/lib/prisma.js', () => ({
  default: (await import('../mocks/prisma.js')).default,
}));

// Import AFTER mocking — ensures the route module receives the mocked prisma
const { default: app } = await import('../../src/app.js');
const { default: prisma } = await import('../../src/lib/prisma.js');

import request from 'supertest';

describe('GET /api/users', () => {
  beforeEach(() => {
    // Reset all mocks between tests — prevents state leaking between tests
    // which is one of the most common causes of flaky tests
    jest.clearAllMocks();
  });

  it('should return users from database', async () => {
    const mockUsers = [
      { id: 1, name: 'Alice', email: 'alice@example.com', role: 'USER' },
      { id: 2, name: 'Bob', email: 'bob@example.com', role: 'ADMIN' },
    ];

    prisma.user.findMany.mockResolvedValue(mockUsers);
    prisma.user.count.mockResolvedValue(2);

    const response = await request(app)
      .get('/api/users')
      .expect(200);

    expect(response.body.data).toHaveLength(2);
    expect(response.body.data[0].name).toBe('Alice');

    // Verify that Prisma was called with expected arguments
    expect(prisma.user.findMany).toHaveBeenCalledTimes(1);
  });

  it('should handle database errors gracefully', async () => {
    // Simulate a database connection failure
    prisma.user.findMany.mockRejectedValue(new Error('Connection refused'));
    prisma.user.count.mockRejectedValue(new Error('Connection refused'));

    const response = await request(app)
      .get('/api/users')
      .expect(500);

    expect(response.body.error).toBeDefined();
  });
});

describe('POST /api/users', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('should create a user', async () => {
    const newUser = { name: 'Charlie', email: 'charlie@example.com' };
    const createdUser = { id: 3, ...newUser, role: 'USER', createdAt: new Date().toISOString() };

    prisma.user.create.mockResolvedValue(createdUser);

    const response = await request(app)
      .post('/api/users')
      .send(newUser)
      .expect(201);

    expect(response.body).toMatchObject(newUser);
    // Verify the data passed to Prisma matches what the route received
    expect(prisma.user.create).toHaveBeenCalledWith({
      data: expect.objectContaining(newUser),
    });
  });

  it('should return 409 for duplicate email', async () => {
    // Simulate Prisma's unique constraint violation error
    const prismaError = new Error('Unique constraint failed');
    prismaError.code = 'P2002';
    prisma.user.create.mockRejectedValue(prismaError);

    await request(app)
      .post('/api/users')
      .send({ name: 'Duplicate', email: 'existing@example.com' })
      .expect(409);
  });
});
```

---

## 6. Integration Tests with Test Database

Integration tests verify that all layers (routes, middleware, database) work together correctly. They use a real database — typically a separate test database that is reset between runs.

### Test Database Setup

```bash
# .env.test — separate database for tests to avoid corrupting development data
DATABASE_URL="postgresql://myuser:mypassword@localhost:5432/mydb_test?schema=public"
```

```javascript
// __tests__/helpers/setup.js
import { PrismaClient } from '@prisma/client';

const prisma = new PrismaClient();

// Reset database before each test suite — ensures tests start from a clean state
// deleteMany in reverse dependency order avoids foreign key constraint errors
export async function resetDatabase() {
  await prisma.post.deleteMany();
  await prisma.user.deleteMany();
}

// Seed minimal test data — shared fixtures that most tests need
export async function seedTestData() {
  const user = await prisma.user.create({
    data: {
      email: 'test@example.com',
      name: 'Test User',
      role: 'USER',
    },
  });

  const adminUser = await prisma.user.create({
    data: {
      email: 'admin@example.com',
      name: 'Admin User',
      role: 'ADMIN',
    },
  });

  return { user, adminUser };
}

export async function disconnectDatabase() {
  await prisma.$disconnect();
}

export { prisma };
```

### Writing Integration Tests

```javascript
// __tests__/integration/users.integration.test.js
import request from 'supertest';
import app from '../../src/app.js';
import { resetDatabase, seedTestData, disconnectDatabase } from '../helpers/setup.js';

describe('Users API (Integration)', () => {
  let testUser;

  beforeAll(async () => {
    // Run once before all tests in this describe block
    await resetDatabase();
  });

  beforeEach(async () => {
    await resetDatabase();
    const data = await seedTestData();
    testUser = data.user;
  });

  afterAll(async () => {
    await resetDatabase();
    await disconnectDatabase();
  });

  describe('GET /api/users', () => {
    it('should return seeded users from the test database', async () => {
      const response = await request(app)
        .get('/api/users')
        .expect(200);

      // Integration test verifies end-to-end: HTTP → route → Prisma → PostgreSQL → response
      expect(response.body.data).toHaveLength(2);
    });
  });

  describe('POST /api/users', () => {
    it('should persist a new user to the database', async () => {
      const newUser = { name: 'New User', email: 'new@example.com' };

      const createResponse = await request(app)
        .post('/api/users')
        .send(newUser)
        .expect(201);

      // Verify the user was actually persisted by reading it back
      const getResponse = await request(app)
        .get(`/api/users/${createResponse.body.id}`)
        .expect(200);

      expect(getResponse.body.email).toBe('new@example.com');
    });

    it('should reject duplicate emails at the database level', async () => {
      await request(app)
        .post('/api/users')
        .send({ name: 'Duplicate', email: testUser.email })
        .expect(409);
    });
  });

  describe('PUT /api/users/:id', () => {
    it('should update and persist changes', async () => {
      await request(app)
        .put(`/api/users/${testUser.id}`)
        .send({ name: 'Updated Name', email: testUser.email })
        .expect(200);

      const getResponse = await request(app)
        .get(`/api/users/${testUser.id}`)
        .expect(200);

      expect(getResponse.body.name).toBe('Updated Name');
    });
  });

  describe('DELETE /api/users/:id', () => {
    it('should remove the user from the database', async () => {
      await request(app)
        .delete(`/api/users/${testUser.id}`)
        .expect(204);

      // Confirm deletion — the user should no longer exist
      await request(app)
        .get(`/api/users/${testUser.id}`)
        .expect(404);
    });
  });
});
```

### Running Integration Tests Separately

```json
{
  "scripts": {
    "test": "NODE_OPTIONS='--experimental-vm-modules' jest --testPathPattern='__tests__/(?!integration)'",
    "test:integration": "dotenv -e .env.test -- npx jest --testPathPattern='integration'",
    "test:all": "npm test && npm run test:integration"
  }
}
```

---

## 7. Testing Authentication

### Helper to Generate Test Tokens

```javascript
// __tests__/helpers/auth.js
import jwt from 'jsonwebtoken';

// Generate a valid JWT for test requests — avoids duplicating token logic
// across every test that needs an authenticated user
export function generateTestToken(user = { id: 1, email: 'test@example.com', role: 'USER' }) {
  return jwt.sign(
    { sub: user.id, email: user.email, role: user.role },
    process.env.JWT_SECRET || 'test-secret',
    { expiresIn: '1h' }
  );
}

export function generateExpiredToken() {
  return jwt.sign(
    { sub: 1, email: 'test@example.com' },
    process.env.JWT_SECRET || 'test-secret',
    { expiresIn: '0s' } // Expires immediately
  );
}
```

### Testing Protected Routes

```javascript
// __tests__/routes/protected.test.js
import request from 'supertest';
import app from '../../src/app.js';
import { generateTestToken, generateExpiredToken } from '../helpers/auth.js';

describe('Protected routes', () => {
  const validToken = generateTestToken({ id: 1, email: 'alice@test.com', role: 'USER' });
  const adminToken = generateTestToken({ id: 2, email: 'admin@test.com', role: 'ADMIN' });

  describe('GET /api/profile', () => {
    it('should return user profile with valid token', async () => {
      const response = await request(app)
        .get('/api/profile')
        .set('Authorization', `Bearer ${validToken}`)  // .set() adds request headers
        .expect(200);

      expect(response.body.user).toHaveProperty('email', 'alice@test.com');
    });

    it('should return 401 without a token', async () => {
      await request(app)
        .get('/api/profile')
        .expect(401);
    });

    it('should return 401 with an expired token', async () => {
      const expiredToken = generateExpiredToken();

      await request(app)
        .get('/api/profile')
        .set('Authorization', `Bearer ${expiredToken}`)
        .expect(401);
    });

    it('should return 401 with a malformed token', async () => {
      await request(app)
        .get('/api/profile')
        .set('Authorization', 'Bearer not-a-real-token')
        .expect(401);
    });
  });

  describe('DELETE /api/users/:id (admin only)', () => {
    it('should allow admin to delete a user', async () => {
      await request(app)
        .delete('/api/users/1')
        .set('Authorization', `Bearer ${adminToken}`)
        .expect(204);
    });

    it('should return 403 for non-admin users', async () => {
      await request(app)
        .delete('/api/users/1')
        .set('Authorization', `Bearer ${validToken}`)
        .expect(403);
    });
  });
});
```

### Testing Login Endpoint

```javascript
describe('POST /api/auth/login', () => {
  it('should return a JWT for valid credentials', async () => {
    const response = await request(app)
      .post('/api/auth/login')
      .send({ email: 'test@example.com', password: 'correct-password' })
      .expect(200);

    expect(response.body).toHaveProperty('token');
    expect(response.body).toHaveProperty('expiresIn');

    // Verify the returned token is valid by using it immediately
    await request(app)
      .get('/api/profile')
      .set('Authorization', `Bearer ${response.body.token}`)
      .expect(200);
  });

  it('should return 401 for wrong password', async () => {
    await request(app)
      .post('/api/auth/login')
      .send({ email: 'test@example.com', password: 'wrong' })
      .expect(401);
  });

  it('should return 401 for non-existent user', async () => {
    await request(app)
      .post('/api/auth/login')
      .send({ email: 'nobody@example.com', password: 'anything' })
      .expect(401);
  });
});
```

---

## 8. Code Coverage with Jest

Code coverage measures how much of your source code is executed during tests. It helps identify untested code paths.

### Running Coverage

```bash
# Generate coverage report
npm run test:coverage

# Jest outputs a summary table and detailed HTML report:
# coverage/lcov-report/index.html
```

### Coverage Report

```
--------------------|---------|----------|---------|---------|
File                | % Stmts | % Branch | % Funcs | % Lines |
--------------------|---------|----------|---------|---------|
All files           |   87.5  |    72.3  |   91.2  |   88.1  |
 src/app.js         |   100   |    100   |   100   |   100   |
 src/routes/users.js|    92   |     78   |   100   |    93   |
 src/middleware/     |    75   |     60   |    80   |    76   |
--------------------|---------|----------|---------|---------|
```

### Coverage Metrics Explained

| Metric | What It Measures |
|--------|-----------------|
| **Statements** | Percentage of code statements executed |
| **Branches** | Percentage of if/else, switch, ternary branches taken |
| **Functions** | Percentage of functions called at least once |
| **Lines** | Percentage of source lines executed |

### Setting Coverage Thresholds

```javascript
// jest.config.js — fail the test run if coverage drops below thresholds
export default {
  // ... other config
  coverageThreshold: {
    global: {
      branches: 70,
      functions: 80,
      lines: 80,
      statements: 80,
    },
    // Per-file thresholds for critical modules
    './src/routes/': {
      branches: 75,
      lines: 85,
    },
  },
};
```

### What to Aim For

- **80%+ line coverage** is a reasonable target for most projects
- **100% is not the goal** -- diminishing returns on trivial code (getters, configuration)
- Focus on **branch coverage** for complex logic (error handling, edge cases)
- High coverage does not guarantee correctness -- you also need meaningful assertions

### Coverage-Guided Test Writing

```javascript
// If coverage shows that the error branch in this route is untested:
router.get('/:id', async (req, res, next) => {
  try {
    const user = await prisma.user.findUnique({ where: { id: parseInt(req.params.id) } });
    if (!user) return res.status(404).json({ error: 'Not found' }); // ← Untested branch?
    res.json(user);
  } catch (err) {
    next(err); // ← Untested error path?
  }
});

// Add tests specifically for those branches:
it('should return 404 for non-existent user', async () => {
  prisma.user.findUnique.mockResolvedValue(null);
  await request(app).get('/api/users/999').expect(404);
});

it('should pass database errors to error handler', async () => {
  prisma.user.findUnique.mockRejectedValue(new Error('DB down'));
  await request(app).get('/api/users/1').expect(500);
});
```

---

## 9. Practice Problems

### Problem 1: CRUD Test Suite

Write a complete test suite for a "products" API with these endpoints:
- `GET /api/products` -- list all products
- `GET /api/products/:id` -- get one product (test 200 and 404 cases)
- `POST /api/products` -- create (test 201 for valid data, 400 for missing `name` or `price`)
- `PUT /api/products/:id` -- update (test 200 and 404)
- `DELETE /api/products/:id` -- delete (test 204 and 404)

Mock all Prisma calls. Aim for 100% branch coverage on the route file.

### Problem 2: Middleware Test

Write unit tests for a `validateApiKey` middleware that:
- Reads the `X-API-Key` header
- Returns 401 if the header is missing
- Returns 403 if the key does not match `process.env.API_KEY`
- Calls `next()` if the key is valid

Test all three branches using mock `req`, `res`, and `next` objects.

### Problem 3: Integration Test with Seeding

Write integration tests for a "comments" system:
- Seed the test database with a user and a post before tests
- Test creating a comment on the post (`POST /api/posts/:postId/comments`)
- Test listing comments for a post (`GET /api/posts/:postId/comments`)
- Test that deleting a post cascades to delete its comments
- Clean up the database after tests

### Problem 4: Authentication Flow Test

Write end-to-end tests for this flow:
1. Register a new user (`POST /api/auth/register`)
2. Log in with the registered credentials (`POST /api/auth/login`)
3. Use the returned token to access `GET /api/profile`
4. Verify that the profile matches the registered user
5. Test that an expired token returns 401

Each step should depend on the result of the previous step.

### Problem 5: Coverage Analysis

Given a route handler with the following branches:
- Success path (200)
- Not found (404)
- Validation error (400)
- Duplicate key (409)
- Database error (500)

Write the minimum number of tests needed to achieve 100% branch coverage. For each test, document which branch it exercises. Then add one negative test that verifies the error response body structure.

---

## References

- [Jest Documentation](https://jestjs.io/docs/getting-started)
- [Supertest Documentation](https://github.com/ladjs/supertest)
- [Jest Mock Functions](https://jestjs.io/docs/mock-functions)
- [Jest Code Coverage](https://jestjs.io/docs/configuration#collectcoverage-boolean)
- [Prisma Testing Guide](https://www.prisma.io/docs/guides/testing)
- [Node.js Testing Best Practices](https://github.com/goldbergyoni/nodebestpractices#4-testing-and-overall-quality-practices)

---

**Previous**: [Express Database](./08_Express_Database.md) | **Next**: [Django Basics](./10_Django_Basics.md)
