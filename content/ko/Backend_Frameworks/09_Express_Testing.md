# 09. Express 테스트

**이전**: [Express 데이터베이스](./08_Express_Database.md) | **다음**: [Django 기초](./10_Django_Basics.md)

## 학습 목표

이 레슨을 마치면 다음을 수행할 수 있습니다:

1. Express 애플리케이션 테스트를 위한 Jest와 Supertest 설정하기
2. 라우트 핸들러와 미들웨어에 대한 단위 테스트를 격리하여 작성하기
3. 실제 데이터베이스 없이 비즈니스 로직을 테스트하기 위해 Prisma 데이터베이스 호출 목킹하기
4. 테스트 데이터베이스에 대해 전체 요청-응답 사이클을 검증하는 통합 테스트 설계하기
5. Jest 내장 도구를 사용하여 코드 커버리지 리포트 측정 및 해석하기

---

테스트는 리팩토링, 기능 추가, 버그 수정을 자신 있게 할 수 있게 해주는 안전망입니다. 테스트 없이는 모든 변경이 조용히 무언가를 망가뜨릴 위험이 있습니다. 이 레슨은 Express 애플리케이션의 전체 테스트 스펙트럼을 다룹니다: 개별 함수에 대한 단위 테스트, HTTP 엔드포인트에 대한 통합 테스트, 데이터베이스 호출 목킹 전략, 코드 커버리지 분석. 테스트 러너(test runner)로 Jest, HTTP 검증에 Supertest를 사용합니다.

## 목차

1. [Jest와 Supertest를 사용한 테스트](#1-jest와-supertest를-사용한-테스트)
2. [테스트 환경 설정](#2-테스트-환경-설정)
3. [라우트 테스트](#3-라우트-테스트)
4. [미들웨어 테스트](#4-미들웨어-테스트)
5. [데이터베이스 호출 목킹](#5-데이터베이스-호출-목킹)
6. [테스트 데이터베이스를 사용한 통합 테스트](#6-테스트-데이터베이스를-사용한-통합-테스트)
7. [인증 테스트](#7-인증-테스트)
8. [Jest를 사용한 코드 커버리지](#8-jest를-사용한-코드-커버리지)
9. [연습 문제](#9-연습-문제)

---

## 1. Jest와 Supertest를 사용한 테스트

### Jest란?

Jest는 Meta에서 만든 완전한 기능을 갖춘 테스트 프레임워크로, 테스트 러너, 검증 라이브러리, 목킹 유틸리티, 코드 커버리지를 하나의 패키지에 모두 제공합니다.

### Supertest란?

Supertest는 **서버를 시작하지 않고** Express 앱에 HTTP 요청을 보낼 수 있게 합니다. 내부적으로 임시 포트에 바인딩하므로 테스트가 빠르고 다른 서비스와 충돌하지 않습니다.

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
│  Supertest는 각 테스트마다 내부 HTTP 서버를 생성합니다 —  │
│  포트 충돌 없음, 서버 생명 주기 관리 불필요              │
└────────────────────────────────────────────────────────┘
```

### 설치

```bash
npm install -D jest @jest/globals supertest

# ES 모듈 지원을 위해 — Node의 네이티브 ESM과 Jest의 모듈 시스템이
# 잘 호환되지 않기 때문에 Jest에 실험적 VM 모듈이 필요합니다
```

---

## 2. 테스트 환경 설정

### Jest 설정

```javascript
// jest.config.js
export default {
  // 실험적 ESM 로더 사용 — import/export 문법에 필요합니다
  transform: {},
  testEnvironment: 'node',
  // 컨벤션으로 테스트 파일 탐색: *.test.js 또는 __tests__/ 내의 파일
  testMatch: ['**/__tests__/**/*.js', '**/*.test.js'],
  // 빌드 산출물과 의존성 무시
  testPathIgnorePatterns: ['/node_modules/', '/dist/'],
  // 소스 파일에서 커버리지 수집, 테스트 유틸리티 제외
  collectCoverageFrom: [
    'src/**/*.js',
    '!src/server.js',       // 서버 시작 파일 제외 (사이드 이펙트)
    '!src/lib/prisma.js',   // Prisma 싱글톤 제외
  ],
};
```

### Package.json 스크립트

```json
{
  "scripts": {
    "test": "NODE_OPTIONS='--experimental-vm-modules' jest",
    "test:watch": "NODE_OPTIONS='--experimental-vm-modules' jest --watch",
    "test:coverage": "NODE_OPTIONS='--experimental-vm-modules' jest --coverage"
  }
}
```

### 앱/서버 분리

Express 앱을 서버 시작에서 분리하는 것이 테스트에 필수적입니다. Supertest는 실행 중인 서버가 아닌 앱 객체가 필요합니다.

```javascript
// src/app.js — 설정된 Express 앱을 내보냅니다 (listen 호출 없음)
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
// src/server.js — 서버를 시작합니다; 직접 실행만 하고 테스트에서 임포트하지 않습니다
import app from './app.js';
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`Listening on port ${PORT}`));
```

---

## 3. 라우트 테스트

### 기본 라우트 테스트

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

    // 응답 구조 검증 — 구현 세부사항이 아닌 계약(contract)을 테스트합니다
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
      .send(newUser)      // .send()는 Content-Type을 application/json으로 설정합니다
      .expect(201);

    expect(response.body).toMatchObject({
      name: 'Alice',
      email: 'alice@example.com',
    });
    // 부분 일치를 위해 toMatchObject를 사용합니다 — 응답에는
    // 하드코딩하고 싶지 않은 id와 createdAt 같은 필드가 포함될 수 있습니다
    expect(response.body).toHaveProperty('id');
  });

  it('should return 400 for missing required fields', async () => {
    const response = await request(app)
      .post('/api/users')
      .send({}) // name과 email 누락
      .expect(400);

    expect(response.body).toHaveProperty('error');
  });
});
```

### 라우트 파라미터 테스트

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

## 4. 미들웨어 테스트

미들웨어는 두 가지 방법으로 테스트할 수 있습니다: 격리하여 (단위 테스트) 또는 라우트를 통해 (통합 테스트).

### 미들웨어 단위 테스트

```javascript
// __tests__/middleware/auth.test.js
import { jest } from '@jest/globals';

// 미들웨어 함수를 직접 테스트합니다 — HTTP 요청이 필요 없습니다
describe('requireAuth middleware', () => {
  let req, res, next;

  beforeEach(() => {
    // 모의 요청, 응답, next 객체를 생성합니다
    // 이것은 미들웨어를 Express 내부에서 격리시킵니다
    req = {
      headers: {},
      get: jest.fn((header) => req.headers[header.toLowerCase()]),
    };
    res = {
      status: jest.fn().mockReturnThis(), // 체이닝 가능 — status().json() 패턴
      json: jest.fn().mockReturnThis(),
    };
    next = jest.fn();
  });

  it('should call next() when valid token is provided', async () => {
    req.headers.authorization = 'Bearer valid-token-here';

    // 모의 객체가 적용되도록 설정 후에 임포트합니다
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
    // next()는 호출되지 않아야 합니다 — 요청 체인이 여기서 중단됩니다
    expect(next).not.toHaveBeenCalled();
  });
});
```

### 라우트를 통한 미들웨어 통합 테스트

```javascript
// 미들웨어가 보호하는 라우트를 통해 미들웨어 동작을 테스트합니다
describe('Rate limiting middleware', () => {
  it('should return 429 after exceeding limit', async () => {
    // 제한까지 요청을 보냅니다
    for (let i = 0; i < 5; i++) {
      await request(app).post('/api/auth/login').send({
        email: 'test@example.com',
        password: 'wrong',
      });
    }

    // 6번째 요청은 속도 제한에 걸려야 합니다
    const response = await request(app)
      .post('/api/auth/login')
      .send({ email: 'test@example.com', password: 'wrong' })
      .expect(429);

    expect(response.body.error).toMatch(/too many/i);
  });
});
```

---

## 5. 데이터베이스 호출 목킹

데이터베이스를 목킹(mocking)하면 실행 중인 데이터베이스 없이 라우트 로직을 테스트할 수 있습니다. 이를 통해 테스트가 더 빠르고, 결정론적이며, 외부 상태에 독립적입니다.

### Prisma Client 목킹

```javascript
// __tests__/mocks/prisma.js
// 모든 Prisma 모델 메서드를 Jest 목 함수로 교체하는 모의 객체를 생성합니다
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

### 테스트에서 목 사용하기

```javascript
// __tests__/routes/users.test.js
import { jest, beforeEach, describe, it, expect } from '@jest/globals';

// Prisma 모듈을 목킹합니다 — prisma.js의 모든 임포트가 목을 받게 됩니다
jest.unstable_mockModule('../../src/lib/prisma.js', () => ({
  default: (await import('../mocks/prisma.js')).default,
}));

// 목킹 후에 임포트합니다 — 라우트 모듈이 목킹된 prisma를 받도록 보장합니다
const { default: app } = await import('../../src/app.js');
const { default: prisma } = await import('../../src/lib/prisma.js');

import request from 'supertest';

describe('GET /api/users', () => {
  beforeEach(() => {
    // 테스트 간 모든 목을 초기화합니다 — 테스트 간 상태 누수를 방지합니다
    // 이것은 불안정한 테스트(flaky test)의 가장 일반적인 원인 중 하나입니다
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

    // Prisma가 예상한 인수로 호출되었는지 확인합니다
    expect(prisma.user.findMany).toHaveBeenCalledTimes(1);
  });

  it('should handle database errors gracefully', async () => {
    // 데이터베이스 연결 실패를 시뮬레이션합니다
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
    // Prisma에 전달된 데이터가 라우트가 받은 데이터와 일치하는지 확인합니다
    expect(prisma.user.create).toHaveBeenCalledWith({
      data: expect.objectContaining(newUser),
    });
  });

  it('should return 409 for duplicate email', async () => {
    // Prisma의 유니크 제약 조건 위반 오류를 시뮬레이션합니다
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

## 6. 테스트 데이터베이스를 사용한 통합 테스트

통합 테스트는 모든 레이어(라우트, 미들웨어, 데이터베이스)가 올바르게 함께 작동하는지 확인합니다. 실제 데이터베이스를 사용합니다 — 보통 실행 간 초기화되는 별도의 테스트 데이터베이스입니다.

### 테스트 데이터베이스 설정

```bash
# .env.test — 개발 데이터를 손상시키지 않기 위한 테스트용 별도 데이터베이스
DATABASE_URL="postgresql://myuser:mypassword@localhost:5432/mydb_test?schema=public"
```

```javascript
// __tests__/helpers/setup.js
import { PrismaClient } from '@prisma/client';

const prisma = new PrismaClient();

// 각 테스트 스위트 전에 데이터베이스를 초기화합니다 — 테스트가 깨끗한 상태에서 시작하도록 보장합니다
// 외래 키 제약 조건 오류를 피하기 위해 역 의존성 순서로 deleteMany를 실행합니다
export async function resetDatabase() {
  await prisma.post.deleteMany();
  await prisma.user.deleteMany();
}

// 최소한의 테스트 데이터 시딩 — 대부분의 테스트에 필요한 공유 픽스처
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

### 통합 테스트 작성하기

```javascript
// __tests__/integration/users.integration.test.js
import request from 'supertest';
import app from '../../src/app.js';
import { resetDatabase, seedTestData, disconnectDatabase } from '../helpers/setup.js';

describe('Users API (Integration)', () => {
  let testUser;

  beforeAll(async () => {
    // 이 describe 블록의 모든 테스트 전에 한 번 실행합니다
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

      // 통합 테스트는 엔드-투-엔드를 검증합니다: HTTP → 라우트 → Prisma → PostgreSQL → 응답
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

      // 다시 읽어와서 실제로 저장되었는지 확인합니다
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

      // 삭제 확인 — 사용자가 더 이상 존재하지 않아야 합니다
      await request(app)
        .get(`/api/users/${testUser.id}`)
        .expect(404);
    });
  });
});
```

### 통합 테스트를 별도로 실행하기

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

## 7. 인증 테스트

### 테스트 토큰 생성 헬퍼

```javascript
// __tests__/helpers/auth.js
import jwt from 'jsonwebtoken';

// 테스트 요청을 위한 유효한 JWT를 생성합니다 — 인증된 사용자가 필요한
// 모든 테스트에서 토큰 로직을 중복하는 것을 방지합니다
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
    { expiresIn: '0s' } // 즉시 만료
  );
}
```

### 보호된 라우트 테스트

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
        .set('Authorization', `Bearer ${validToken}`)  // .set()은 요청 헤더를 추가합니다
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

### 로그인 엔드포인트 테스트

```javascript
describe('POST /api/auth/login', () => {
  it('should return a JWT for valid credentials', async () => {
    const response = await request(app)
      .post('/api/auth/login')
      .send({ email: 'test@example.com', password: 'correct-password' })
      .expect(200);

    expect(response.body).toHaveProperty('token');
    expect(response.body).toHaveProperty('expiresIn');

    // 반환된 토큰을 즉시 사용하여 유효한지 확인합니다
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

## 8. Jest를 사용한 코드 커버리지

코드 커버리지는 테스트 중 소스 코드의 얼마나 많은 부분이 실행되는지를 측정합니다. 테스트되지 않은 코드 경로를 파악하는 데 도움이 됩니다.

### 커버리지 실행

```bash
# 커버리지 리포트 생성
npm run test:coverage

# Jest가 요약 테이블과 상세 HTML 리포트를 출력합니다:
# coverage/lcov-report/index.html
```

### 커버리지 리포트

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

### 커버리지 메트릭 설명

| 메트릭 | 측정 대상 |
|--------|-----------------|
| **Statements (구문)** | 실행된 코드 구문의 비율 |
| **Branches (분기)** | 취해진 if/else, switch, 삼항 분기의 비율 |
| **Functions (함수)** | 최소 한 번 호출된 함수의 비율 |
| **Lines (라인)** | 실행된 소스 라인의 비율 |

### 커버리지 임계값 설정

```javascript
// jest.config.js — 커버리지가 임계값 아래로 떨어지면 테스트 실행을 실패시킵니다
export default {
  // ... 다른 설정
  coverageThreshold: {
    global: {
      branches: 70,
      functions: 80,
      lines: 80,
      statements: 80,
    },
    // 중요한 모듈에 대한 파일별 임계값
    './src/routes/': {
      branches: 75,
      lines: 85,
    },
  },
};
```

### 목표치

- **라인 커버리지 80%+**가 대부분의 프로젝트에서 합리적인 목표입니다
- **100%는 목표가 아닙니다** -- 사소한 코드(getter, 설정)에서 수익 감소
- 복잡한 로직(오류 처리, 엣지 케이스)에 대해 **분기 커버리지**에 집중하세요
- 높은 커버리지가 정확성을 보장하지 않습니다 -- 의미 있는 검증도 필요합니다

### 커버리지 기반 테스트 작성

```javascript
// 커버리지가 이 라우트의 오류 분기가 테스트되지 않았음을 보여준다면:
router.get('/:id', async (req, res, next) => {
  try {
    const user = await prisma.user.findUnique({ where: { id: parseInt(req.params.id) } });
    if (!user) return res.status(404).json({ error: 'Not found' }); // ← 테스트되지 않은 분기?
    res.json(user);
  } catch (err) {
    next(err); // ← 테스트되지 않은 오류 경로?
  }
});

// 해당 분기에 대한 테스트를 추가합니다:
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

## 9. 연습 문제

### 문제 1: CRUD 테스트 스위트

다음 엔드포인트가 있는 "products" API에 대한 완전한 테스트 스위트를 작성하세요:
- `GET /api/products` -- 모든 제품 목록 조회
- `GET /api/products/:id` -- 단일 제품 조회 (200과 404 케이스 테스트)
- `POST /api/products` -- 생성 (유효한 데이터는 201, `name`이나 `price` 누락 시 400)
- `PUT /api/products/:id` -- 업데이트 (200과 404 테스트)
- `DELETE /api/products/:id` -- 삭제 (204와 404 테스트)

모든 Prisma 호출을 목킹하세요. 라우트 파일에서 100% 분기 커버리지를 목표로 하세요.

### 문제 2: 미들웨어 테스트

다음과 같은 `validateApiKey` 미들웨어에 대한 단위 테스트를 작성하세요:
- `X-API-Key` 헤더를 읽습니다
- 헤더가 없으면 401을 반환합니다
- 키가 `process.env.API_KEY`와 일치하지 않으면 403을 반환합니다
- 키가 유효하면 `next()`를 호출합니다

모의 `req`, `res`, `next` 객체를 사용하여 세 가지 분기 모두를 테스트하세요.

### 문제 3: 시딩이 있는 통합 테스트

"comments" 시스템에 대한 통합 테스트를 작성하세요:
- 테스트 전에 사용자와 게시물로 테스트 데이터베이스를 시딩합니다
- 게시물에 댓글 생성 테스트 (`POST /api/posts/:postId/comments`)
- 게시물의 댓글 목록 조회 테스트 (`GET /api/posts/:postId/comments`)
- 게시물 삭제 시 댓글도 함께 삭제되는지 테스트 (cascade 삭제)
- 테스트 후 데이터베이스를 정리합니다

### 문제 4: 인증 흐름 테스트

다음 흐름에 대한 엔드-투-엔드 테스트를 작성하세요:
1. 새 사용자 등록 (`POST /api/auth/register`)
2. 등록된 자격 증명으로 로그인 (`POST /api/auth/login`)
3. 반환된 토큰으로 `GET /api/profile`에 접근
4. 프로필이 등록된 사용자와 일치하는지 확인
5. 만료된 토큰이 401을 반환하는지 테스트

각 단계는 이전 단계의 결과에 의존해야 합니다.

### 문제 5: 커버리지 분석

다음 분기가 있는 라우트 핸들러가 주어졌을 때:
- 성공 경로 (200)
- 찾을 수 없음 (404)
- 검증 오류 (400)
- 중복 키 (409)
- 데이터베이스 오류 (500)

100% 분기 커버리지를 달성하는 데 필요한 최소한의 테스트 수를 작성하세요. 각 테스트에 대해 어떤 분기를 검증하는지 문서화하세요. 그런 다음 오류 응답 본문 구조를 확인하는 부정 테스트(negative test) 하나를 추가하세요.

---

## 참고 자료

- [Jest 공식 문서](https://jestjs.io/docs/getting-started)
- [Supertest 공식 문서](https://github.com/ladjs/supertest)
- [Jest 목 함수](https://jestjs.io/docs/mock-functions)
- [Jest 코드 커버리지](https://jestjs.io/docs/configuration#collectcoverage-boolean)
- [Prisma 테스트 가이드](https://www.prisma.io/docs/guides/testing)
- [Node.js 테스트 모범 사례](https://github.com/goldbergyoni/nodebestpractices#4-testing-and-overall-quality-practices)

---

**이전**: [Express 데이터베이스](./08_Express_Database.md) | **다음**: [Django 기초](./10_Django_Basics.md)
