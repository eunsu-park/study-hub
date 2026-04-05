# 08. Express 데이터베이스

**이전**: [Express 심화](./07_Express_Advanced.md) | **다음**: [Express 테스트](./09_Express_Testing.md)

## 학습 목표

이 레슨을 마치면 다음을 수행할 수 있습니다:

1. Express 프로젝트에 Prisma ORM을 설정하고 PostgreSQL에 연결하기
2. 관계(relation)를 포함한 Prisma의 선언적 스키마 언어로 데이터베이스 스키마 설계하기
3. Prisma Client의 타입 안전 쿼리 API로 CRUD 작업 수행하기
4. Prisma Migrate로 데이터베이스 스키마 변경 이력 관리하기
5. 데이터 무결성을 위한 `select`, `include`, 트랜잭션으로 쿼리 최적화하기

---

모든 진지한 백엔드 애플리케이션에는 영속적인 데이터 저장소가 필요합니다. 원시 SQL 쿼리를 작성할 수도 있지만, ORM(Object-Relational Mapper, 객체-관계 매퍼)은 보일러플레이트(boilerplate)를 줄이고 SQL 인젝션 같은 일반적인 실수를 방지하는 고수준의 타입 안전 인터페이스를 제공합니다. 이 레슨은 Node.js에서 가장 인기 있는 현대적 ORM인 Prisma를 다루며, Express와 통합하여 데이터 기반 API를 구축하는 방법을 보여줍니다.

> **왜 Sequelize나 TypeORM이 아닌 Prisma인가?** Prisma는 스키마 우선(schema-first) 접근 방식을 취합니다. `.prisma` 파일에 데이터 모델을 선언하면 완전한 타입 안전 클라이언트를 생성합니다. 이를 통해 런타임이 아닌 빌드 시점에 오류를 잡습니다. 마이그레이션 시스템도 더 단순하고 예측 가능합니다.

## 목차

1. [Prisma ORM 개요](#1-prisma-orm-개요)
2. [프로젝트 설정](#2-프로젝트-설정)
3. [스키마 정의](#3-스키마-정의)
4. [Prisma Client: CRUD 작업](#4-prisma-client-crud-작업)
5. [관계(Relations)](#5-관계relations)
6. [마이그레이션(Migrations)](#6-마이그레이션migrations)
7. [쿼리 최적화](#7-쿼리-최적화)
8. [트랜잭션 처리](#8-트랜잭션-처리)
9. [PostgreSQL 연결](#9-postgresql-연결)
10. [연습 문제](#10-연습-문제)

---

## 1. Prisma ORM 개요

Prisma는 세 가지 핵심 컴포넌트로 구성됩니다:

```
┌──────────────────────────────────────────────────────┐
│                  Prisma Ecosystem                    │
│                                                      │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────┐ │
│  │ Prisma Schema│  │ Prisma Client│  │  Prisma    │ │
│  │ (.prisma)    │  │ (Generated)  │  │  Migrate   │ │
│  │              │  │              │  │            │ │
│  │ Defines your │  │ Type-safe    │  │ Version-   │ │
│  │ data model   │  │ query API    │  │ controlled │ │
│  │              │  │              │  │ schema     │ │
│  │              │  │              │  │ changes    │ │
│  └──────────────┘  └──────────────┘  └────────────┘ │
└──────────────────────────────────────────────────────┘
```

| 컴포넌트 | 목적 |
|-----------|---------|
| **Prisma Schema** | 데이터베이스 구조의 단일 진실 공급원(Single Source of Truth) |
| **Prisma Client** | 자동 생성된 타입 안전 쿼리 빌더 |
| **Prisma Migrate** | 스키마 diff를 기반으로 한 선언적 마이그레이션 시스템 |
| **Prisma Studio** | 데이터 조회 및 편집을 위한 GUI (개발 도구) |

---

## 2. 프로젝트 설정

### 설치

```bash
# 새 Express 프로젝트 초기화 (아직 안 한 경우)
mkdir express-prisma && cd express-prisma
npm init -y
npm install express
npm install -D prisma

# Prisma 초기화 — prisma/schema.prisma와 .env 생성
npx prisma init

# Prisma Client 런타임 설치
npm install @prisma/client
```

### 프로젝트 구조

```
express-prisma/
├── prisma/
│   ├── schema.prisma     # 데이터 모델 정의
│   └── migrations/       # 생성된 마이그레이션 SQL 파일
├── src/
│   ├── app.js
│   ├── server.js
│   ├── lib/
│   │   └── prisma.js     # 싱글톤 Prisma Client 인스턴스
│   └── routes/
│       └── users.js
├── .env                  # DATABASE_URL
└── package.json
```

### Prisma Client 싱글톤

```javascript
// src/lib/prisma.js
// 단일 PrismaClient 인스턴스를 생성하고 애플리케이션 전체에서 재사용합니다.
// 여러 인스턴스 생성은 데이터베이스 연결을 낭비하고
// 부하 시 커넥션 풀(connection pool)을 고갈시킬 수 있습니다.
import { PrismaClient } from '@prisma/client';

const prisma = new PrismaClient({
  // 개발 중 쿼리 로깅 — N+1 문제와 느린 쿼리 디버깅에 도움이 됩니다
  log: process.env.NODE_ENV === 'development' ? ['query', 'warn', 'error'] : ['error'],
});

export default prisma;
```

---

## 3. 스키마 정의

Prisma 스키마 파일(`prisma/schema.prisma`)은 데이터 모델, 데이터베이스 연결, 제너레이터 설정을 정의합니다.

### 기본 스키마

```prisma
// prisma/schema.prisma

// 제너레이터는 Prisma가 무엇을 생성할지 지정합니다 — 여기서는 JavaScript 클라이언트
generator client {
  provider = "prisma-client-js"
}

// 데이터소스(datasource)는 데이터베이스 연결을 설정합니다
datasource db {
  provider = "postgresql"
  url      = env("DATABASE_URL")  // .env 파일에서 읽습니다
}

// 모델은 데이터베이스 테이블에 매핑됩니다 — 각 필드는 컬럼이 됩니다
model User {
  id        Int      @id @default(autoincrement())
  email     String   @unique
  name      String
  role      Role     @default(USER)
  createdAt DateTime @default(now())  // 생성 시 자동 설정
  updatedAt DateTime @updatedAt       // 매 수정 시 자동 업데이트

  // 관계: 한 사용자가 여러 게시물을 가집니다
  posts     Post[]

  @@map("users")  // 데이터베이스의 테이블명 재정의
}

model Post {
  id          Int      @id @default(autoincrement())
  title       String   @db.VarChar(200)  // 특정 SQL 타입에 매핑
  content     String?                     // ?는 필드를 nullable로 만듭니다
  published   Boolean  @default(false)
  authorId    Int
  createdAt   DateTime @default(now())
  updatedAt   DateTime @updatedAt

  // 외래 키(Foreign key) — 게시물을 작성자와 연결합니다
  author      User     @relation(fields: [authorId], references: [id])
  tags        Tag[]    // 암묵적 조인 테이블을 통한 다대다(Many-to-many)

  // 복합 인덱스 — 작성자 + 발행 상태로 필터링하는 쿼리 속도를 높입니다
  @@index([authorId, published])
  @@map("posts")
}

model Tag {
  id    Int    @id @default(autoincrement())
  name  String @unique
  posts Post[] // 다대다 (Prisma가 조인 테이블을 자동으로 생성합니다)

  @@map("tags")
}

enum Role {
  USER
  ADMIN
  MODERATOR
}
```

### 필드 타입 참고

| Prisma 타입 | PostgreSQL 타입 | 비고 |
|-------------|----------------|-------|
| `String` | `text` | 길이 제한은 `@db.VarChar(n)` 사용 |
| `Int` | `integer` | 32비트 정수 |
| `BigInt` | `bigint` | 64비트 정수 |
| `Float` | `double precision` | 부동 소수점 |
| `Decimal` | `decimal(65,30)` | 정밀 소수 (금액 등) |
| `Boolean` | `boolean` | true/false |
| `DateTime` | `timestamp(3)` | 밀리초 정밀도 |
| `Json` | `jsonb` | 구조화된 JSON 데이터 |

---

## 4. Prisma Client: CRUD 작업

스키마를 정의한 후 클라이언트를 생성하고 라우트에서 사용합니다:

```bash
# Prisma Client 생성 — 스키마 변경 후 반드시 실행해야 합니다
npx prisma generate
```

### 생성 (Create)

```javascript
// src/routes/users.js
import { Router } from 'express';
import prisma from '../lib/prisma.js';

const router = Router();

router.post('/', async (req, res, next) => {
  try {
    const { email, name, role } = req.body;

    const user = await prisma.user.create({
      data: { email, name, role },
    });

    res.status(201).json(user);
  } catch (err) {
    // Prisma는 유니크 제약 조건 위반 시 P2002를 발생시킵니다 —
    // 여기서 처리하여 500 대신 친근한 오류를 반환합니다
    if (err.code === 'P2002') {
      return res.status(409).json({ error: `Email already exists` });
    }
    next(err);
  }
});

// 중첩 관계를 포함한 생성 — 한 번의 쿼리로 사용자와 게시물을 함께 생성합니다
router.post('/with-posts', async (req, res, next) => {
  try {
    const user = await prisma.user.create({
      data: {
        email: req.body.email,
        name: req.body.name,
        posts: {
          create: [
            { title: 'First Post', content: 'Hello world!' },
            { title: 'Second Post', content: 'Another post' },
          ],
        },
      },
      include: { posts: true }, // 응답에 생성된 게시물을 포함합니다
    });

    res.status(201).json(user);
  } catch (err) {
    next(err);
  }
});
```

### 읽기 (Read)

```javascript
// 필터링, 페이지네이션, 정렬로 여러 항목 조회
router.get('/', async (req, res, next) => {
  try {
    const { page = 1, limit = 10, role, search } = req.query;
    const skip = (parseInt(page) - 1) * parseInt(limit);

    // where 절을 동적으로 구성합니다 — 제공된 필터만 포함합니다
    const where = {};
    if (role) where.role = role;
    if (search) {
      where.OR = [
        { name: { contains: search, mode: 'insensitive' } },
        { email: { contains: search, mode: 'insensitive' } },
      ];
    }

    // count와 findMany를 병렬로 실행합니다 — 두 번의 순차적인 DB 왕복을 방지합니다
    const [users, total] = await Promise.all([
      prisma.user.findMany({
        where,
        skip,
        take: parseInt(limit),
        orderBy: { createdAt: 'desc' },
        select: { id: true, email: true, name: true, role: true, createdAt: true },
      }),
      prisma.user.count({ where }),
    ]);

    res.json({
      data: users,
      pagination: {
        page: parseInt(page),
        limit: parseInt(limit),
        total,
        totalPages: Math.ceil(total / parseInt(limit)),
      },
    });
  } catch (err) {
    next(err);
  }
});

// ID로 단일 항목 조회
router.get('/:id', async (req, res, next) => {
  try {
    const user = await prisma.user.findUnique({
      where: { id: parseInt(req.params.id) },
      include: {
        posts: {
          where: { published: true },
          orderBy: { createdAt: 'desc' },
        },
      },
    });

    if (!user) return res.status(404).json({ error: 'User not found' });
    res.json(user);
  } catch (err) {
    next(err);
  }
});
```

### 업데이트 (Update)

```javascript
router.put('/:id', async (req, res, next) => {
  try {
    const { name, email, role } = req.body;

    const user = await prisma.user.update({
      where: { id: parseInt(req.params.id) },
      data: { name, email, role },
    });

    res.json(user);
  } catch (err) {
    // P2025: 업데이트할 레코드를 찾을 수 없음
    if (err.code === 'P2025') {
      return res.status(404).json({ error: 'User not found' });
    }
    next(err);
  }
});

// Upsert — 없으면 생성, 있으면 업데이트
// 클라이언트가 레코드 존재 여부를 알 수 없는 "저장" 작업에 유용합니다
router.put('/by-email/:email', async (req, res, next) => {
  try {
    const user = await prisma.user.upsert({
      where: { email: req.params.email },
      update: { name: req.body.name },
      create: { email: req.params.email, name: req.body.name },
    });

    res.json(user);
  } catch (err) {
    next(err);
  }
});
```

### 삭제 (Delete)

```javascript
router.delete('/:id', async (req, res, next) => {
  try {
    await prisma.user.delete({
      where: { id: parseInt(req.params.id) },
    });

    res.status(204).send();
  } catch (err) {
    if (err.code === 'P2025') {
      return res.status(404).json({ error: 'User not found' });
    }
    next(err);
  }
});

export default router;
```

---

## 5. 관계(Relations)

### 일대다(One-to-Many)

```prisma
// 한 사용자가 여러 게시물을 가집니다 (위 스키마에서 이미 보여드린 내용)
model User {
  id    Int    @id @default(autoincrement())
  posts Post[]
}

model Post {
  id       Int  @id @default(autoincrement())
  authorId Int
  author   User @relation(fields: [authorId], references: [id])
}
```

```javascript
// 관계와 함께 쿼리
const userWithPosts = await prisma.user.findUnique({
  where: { id: 1 },
  include: { posts: true },
});

// 기존 사용자에 대한 게시물 생성 — connect는 외래 키 컬럼명을
// 알 필요 없이 기존 레코드에 연결합니다
const post = await prisma.post.create({
  data: {
    title: 'New Post',
    author: { connect: { id: 1 } },
  },
});
```

### 다대다(Many-to-Many)

```prisma
// Prisma가 조인 테이블을 자동으로 관리합니다 — 직접 조작할 필요가 없습니다
model Post {
  id   Int   @id @default(autoincrement())
  tags Tag[]
}

model Tag {
  id    Int    @id @default(autoincrement())
  name  String @unique
  posts Post[]
}
```

```javascript
// 새 태그와 기존 태그로 게시물 생성
const post = await prisma.post.create({
  data: {
    title: 'Prisma Guide',
    author: { connect: { id: 1 } },
    tags: {
      // connectOrCreate는 중복을 방지합니다 — 태그가 존재하지 않으면 생성하고,
      // 이미 존재하면 기존 태그에 연결합니다
      connectOrCreate: [
        { where: { name: 'prisma' }, create: { name: 'prisma' } },
        { where: { name: 'orm' }, create: { name: 'orm' } },
      ],
    },
  },
  include: { tags: true },
});

// 태그로 게시물 찾기
const postsWithTag = await prisma.post.findMany({
  where: {
    tags: { some: { name: 'prisma' } }, // "some" = 최소 하나의 관련 태그가 일치
  },
  include: { tags: true },
});
```

### 자기 참조 관계(Self-Relations)

```prisma
// 댓글에는 답글이 달릴 수 있습니다 — 부모와 자식 모두 Comment 레코드입니다
model Comment {
  id       Int       @id @default(autoincrement())
  text     String
  parentId Int?
  parent   Comment?  @relation("CommentReplies", fields: [parentId], references: [id])
  replies  Comment[] @relation("CommentReplies")
}
```

---

## 6. 마이그레이션(Migrations)

Prisma Migrate는 스키마 변경을 버전이 관리되는 SQL 마이그레이션 파일로 추적합니다.

### 개발 워크플로우

```bash
# 스키마 변경에서 마이그레이션 생성 — SQL을 생성하고 적용합니다
# 이름은 마이그레이션 이력을 쉽게 읽을 수 있도록 변경 내용을 설명해야 합니다
npx prisma migrate dev --name add_user_model

# 이 명령은 다음을 수행합니다:
# 1. 스키마 diff 감지 (마지막 마이그레이션 이후 변경된 내용)
# 2. prisma/migrations/에 SQL 마이그레이션 파일 생성
# 3. 데이터베이스에 마이그레이션 적용
# 4. Prisma Client 재생성
```

### 마이그레이션 파일

```
prisma/migrations/
├── 20250601120000_add_user_model/
│   └── migration.sql
├── 20250602150000_add_post_model/
│   └── migration.sql
└── migration_lock.toml    # 데이터베이스 제공자를 잠급니다
```

```sql
-- prisma/migrations/20250601120000_add_user_model/migration.sql
-- Prisma Migrate에 의해 자동 생성됨 — 필요한 경우를 제외하고 수동으로 편집하지 마세요
CREATE TABLE "users" (
    "id" SERIAL NOT NULL,
    "email" TEXT NOT NULL,
    "name" TEXT NOT NULL,
    "role" "Role" NOT NULL DEFAULT 'USER',
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,
    CONSTRAINT "users_pkey" PRIMARY KEY ("id")
);

CREATE UNIQUE INDEX "users_email_key" ON "users"("email");
```

### 프로덕션 배포

```bash
# 프로덕션에서는 `dev` 대신 `deploy`를 사용합니다 — 새 마이그레이션을 생성하거나
# 클라이언트를 재생성하지 않고 보류 중인 마이그레이션만 적용합니다
npx prisma migrate deploy

# 데이터베이스 초기화 (파괴적 — 모든 데이터 삭제)
# 마이그레이션 이력이 불일치할 때만 개발 환경에서 사용합니다
npx prisma migrate reset
```

### 시딩(Seeding)

```javascript
// prisma/seed.js — 초기 또는 테스트 데이터로 데이터베이스를 채웁니다
import { PrismaClient } from '@prisma/client';

const prisma = new PrismaClient();

async function main() {
  // Upsert는 시드가 여러 번 실행될 때 오류를 방지합니다
  const alice = await prisma.user.upsert({
    where: { email: 'alice@example.com' },
    update: {},
    create: {
      email: 'alice@example.com',
      name: 'Alice',
      role: 'ADMIN',
      posts: {
        create: [
          { title: 'Hello World', content: 'My first post', published: true },
        ],
      },
    },
  });

  console.log('Seeded:', { alice });
}

main()
  .catch(console.error)
  .finally(() => prisma.$disconnect());
```

```json
// package.json에 추가 — `prisma migrate reset` 시 Prisma가 이 스크립트를 자동으로 호출합니다
{
  "prisma": {
    "seed": "node prisma/seed.js"
  }
}
```

```bash
# 수동으로 시드 실행
npx prisma db seed
```

---

## 7. 쿼리 최적화

### Select vs Include

```javascript
// select — 특정 필드만 가져옵니다; 데이터 전송을 줄입니다
// 컬럼의 일부만 필요할 때 사용합니다 (드롭다운 목록, 검색 결과 등)
const userNames = await prisma.user.findMany({
  select: {
    id: true,
    name: true,
    email: true,
    // posts: false — posts를 선택하지 않으면 JOIN이 수행되지 않습니다
  },
});

// include — 전체 모델과 관련 레코드를 가져옵니다
// JOIN 또는 보조 쿼리를 트리거합니다; 관련 데이터가 필요할 때만 사용합니다
const usersWithPosts = await prisma.user.findMany({
  include: {
    posts: {
      select: { id: true, title: true },  // 관련 필드도 제한합니다
      where: { published: true },
      take: 5,                              // 최신 5개 게시물만 로드합니다
      orderBy: { createdAt: 'desc' },
    },
  },
});

// 경고: 같은 레벨에서 select와 include를 동시에 사용할 수 없습니다.
// 쿼리당 하나의 접근 방식을 선택하세요.
```

### N+1 쿼리 방지하기

```javascript
// 나쁜 예: N+1 문제 — 사용자에 대한 1개의 쿼리, 그 다음 사용자당 1개의 게시물 쿼리
const users = await prisma.user.findMany();
for (const user of users) {
  // 각 반복마다 별도의 데이터베이스 쿼리를 발생시킵니다
  const posts = await prisma.post.findMany({ where: { authorId: user.id } });
  user.posts = posts;
}

// 좋은 예: include와 함께 단일 쿼리 — Prisma가 효율적인 JOIN 또는 IN 절을 생성합니다
const usersWithPosts = await prisma.user.findMany({
  include: { posts: true },
});
```

### 원시(Raw) 쿼리

```javascript
// Prisma Client로 표현할 수 없는 복잡한 쿼리를 위해
// Prisma의 커넥션 풀링 이점을 유지하면서 원시 SQL을 사용합니다

// 태그 템플릿 — 파라미터는 SQL 인젝션을 방지하기 위해 자동으로 이스케이프됩니다
const result = await prisma.$queryRaw`
  SELECT u.name, COUNT(p.id) as post_count
  FROM users u
  LEFT JOIN posts p ON p."authorId" = u.id
  WHERE p.published = true
  GROUP BY u.id
  HAVING COUNT(p.id) > ${minPosts}
  ORDER BY post_count DESC
`;
```

---

## 8. 트랜잭션 처리

트랜잭션(transaction)은 연산 그룹이 모두 성공하거나 모두 실패하도록 보장합니다. 데이터 일관성 유지에 필수적입니다.

### 순차적 트랜잭션

```javascript
// prisma.$transaction()은 데이터베이스 트랜잭션으로 연산을 감쌉니다 —
// 어떤 연산이 실패하면 이전의 모든 연산이 롤백됩니다
const transferCredits = async (fromId, toId, amount) => {
  const [sender, receiver] = await prisma.$transaction([
    prisma.user.update({
      where: { id: fromId },
      data: { credits: { decrement: amount } },
    }),
    prisma.user.update({
      where: { id: toId },
      data: { credits: { increment: amount } },
    }),
  ]);

  return { sender, receiver };
};
```

### 인터랙티브 트랜잭션

```javascript
// 인터랙티브 트랜잭션은 전체 Prisma Client 기능을 갖춘 트랜잭션 클라이언트(tx)를 제공합니다
// 트랜잭션 내 조건부 로직에 유용합니다
const createOrder = async (userId, items) => {
  return prisma.$transaction(async (tx) => {
    // 트랜잭션 내에서 재고 수준을 확인합니다
    for (const item of items) {
      const product = await tx.product.findUnique({
        where: { id: item.productId },
      });

      if (!product || product.stock < item.quantity) {
        // 콜백 내에서 예외를 발생시키면 전체 트랜잭션이 롤백됩니다
        throw new Error(`Insufficient stock for ${product?.name ?? item.productId}`);
      }

      // 재고 감소
      await tx.product.update({
        where: { id: item.productId },
        data: { stock: { decrement: item.quantity } },
      });
    }

    // 라인 아이템으로 주문 생성
    const order = await tx.order.create({
      data: {
        userId,
        items: {
          create: items.map(item => ({
            productId: item.productId,
            quantity: item.quantity,
            price: item.price,
          })),
        },
        total: items.reduce((sum, i) => sum + i.price * i.quantity, 0),
      },
      include: { items: true },
    });

    return order;
  }, {
    maxWait: 5000,  // 트랜잭션 슬롯 대기 최대 시간 (ms)
    timeout: 10000, // 최대 트랜잭션 지속 시간 (ms) — 장시간 실행 잠금 방지
  });
};
```

---

## 9. PostgreSQL 연결

### 환경 설정

```bash
# .env
# 형식: postgresql://USER:PASSWORD@HOST:PORT/DATABASE?schema=SCHEMA
DATABASE_URL="postgresql://myuser:mypassword@localhost:5432/mydb?schema=public"

# 커넥션 풀링 (프로덕션) — PgBouncer 또는 Prisma Accelerate
# DATABASE_URL="postgresql://myuser:mypassword@pgbouncer:6432/mydb?pgbouncer=true"
```

### 로컬 개발을 위한 Docker 설정

```yaml
# docker-compose.yml — PostgreSQL을 설치하지 않고 로컬에서 실행합니다
version: '3.8'
services:
  postgres:
    image: postgres:16
    environment:
      POSTGRES_USER: myuser
      POSTGRES_PASSWORD: mypassword
      POSTGRES_DB: mydb
    ports:
      - '5432:5432'
    volumes:
      - pgdata:/var/lib/postgresql/data  # 컨테이너 재시작 시에도 데이터 유지

volumes:
  pgdata:
```

```bash
# PostgreSQL 시작
docker compose up -d

# 마이그레이션 적용
npx prisma migrate dev

# Prisma Studio 열기 — 브라우저 기반 데이터 뷰어
npx prisma studio
```

### 그레이스풀 셧다운(Graceful Shutdown)

```javascript
// src/server.js — 프로세스 종료 시 Prisma 연결을 끊습니다
// 없으면 재시작이나 배포 중에 데이터베이스 연결이 누수될 수 있습니다
import app from './app.js';
import prisma from './lib/prisma.js';

const PORT = process.env.PORT || 3000;

const server = app.listen(PORT, () => {
  console.log(`Server running on http://localhost:${PORT}`);
});

// 컨테이너 오케스트레이터(Docker, K8s)의 셧다운 신호 처리
const shutdown = async () => {
  console.log('Shutting down gracefully...');
  server.close();
  await prisma.$disconnect();
  process.exit(0);
};

process.on('SIGTERM', shutdown);
process.on('SIGINT', shutdown);
```

---

## 10. 연습 문제

### 문제 1: 블로그 API

다음 모델과 엔드포인트로 완전한 블로그 API를 만드세요:
- **User**: id, email (unique), name, bio (optional)
- **Post**: id, title, content, published (default false), authorId
- 엔드포인트: 사용자와 게시물에 대한 CRUD, `GET /api/users/:id/posts`, 발행 상태 토글

### 문제 2: 페이지네이션과 필터링

게시물 엔드포인트에 커서 기반(cursor-based) 페이지네이션을 추가하세요:
- `GET /api/posts?cursor=42&take=20`은 ID 42 이후에 20개의 게시물을 반환합니다
- `published` 상태로 필터링하고 `title`로 검색하는 기능을 지원합니다
- 클라이언트가 다음 요청에 사용할 수 있도록 응답에 `nextCursor`를 반환합니다

### 문제 3: 태그를 사용한 다대다

블로그에 Tag 모델을 추가하여 확장하세요:
- `POST /api/posts`는 문자열 배열 `tags`를 받습니다
- `connectOrCreate`를 사용하여 기존 태그가 재사용되도록 합니다
- `GET /api/tags/:name/posts`는 해당 태그의 모든 게시물을 반환합니다
- `DELETE /api/tags/:id`는 게시물이 그 태그를 사용하지 않는 경우에만 작동해야 합니다

### 문제 4: 트랜잭션 이체

"크레딧 이체" 시스템을 구현하세요:
- 사용자에게 `balance` 필드(Decimal)가 있습니다
- `POST /api/transfers`는 `{ fromId, toId, amount }`를 받습니다
- 인터랙티브 트랜잭션으로 발신자의 잔액이 충분한지 검증합니다
- 감사 로그로 `Transfer` 레코드를 생성합니다
- 잔액이 부족하면 400을 반환합니다 (데이터를 수정하지 않고)

### 문제 5: 스키마 마이그레이션

블로그 스키마에서 시작하여 Prisma Migrate로 다음 변경사항을 적용하세요:
1. `id`와 `name` 필드가 있는 `Category` 모델 추가
2. `Post`에 `categoryId` 외래 키 추가 (optional, nullable)
3. `Post(title, authorId)`에 복합 유니크 제약 조건 추가 -- 작성자당 중복 제목 방지
4. 마이그레이션을 실행하고 Prisma Studio로 확인합니다

---

## 참고 자료

- [Prisma 공식 문서](https://www.prisma.io/docs)
- [Prisma Schema 레퍼런스](https://www.prisma.io/docs/reference/api-reference/prisma-schema-reference)
- [Prisma Client API](https://www.prisma.io/docs/reference/api-reference/prisma-client-reference)
- [Prisma Migrate 가이드](https://www.prisma.io/docs/guides/migrate)
- [PostgreSQL 공식 문서](https://www.postgresql.org/docs/)

---

**이전**: [Express 심화](./07_Express_Advanced.md) | **다음**: [Express 테스트](./09_Express_Testing.md)
