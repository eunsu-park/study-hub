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

프레임워크 참조에 들어가기 전에, [**이론과 원리**](#이론과-원리) 섹션을 먼저 읽어보세요. ORM이 query builder와 비교해 실제로 무엇을 하는지, 왜 prepared statement가 SQL 인젝션의 유일한 진정한 방어인지, 그리고 트랜잭션·격리 수준·연결 풀이 어떻게 맞물리는지를 다룹니다.

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

## 이론과 원리

Node 앱의 데이터베이스 계층은 세 가지 직교 메커니즘 위에 있습니다. Prisma의 API는 이를 유려한 타입 안전 표면 뒤에 숨기지만, 모든 프로덕션 디버깅 세션은 결국 표면 아래를 들여다보게 만듭니다.

- **(A) ORM vs query builder vs raw SQL** — 다른 트레이드오프를 가진 세 가지 추상화 수준.
- **(B) 인젝션 방어로서의 prepared statement** — 왜 `${userInput}` 보간이 안전하지 않은지, 플레이스홀더가 프로토콜 수준에서 무엇을 하는지.
- **(C) 트랜잭션과 격리** — `BEGIN`/`COMMIT`이 실제로 사 주는 것, 그리고 ANSI 4가지 격리 수준이 어떤 특정 이상 현상을 막는지.

### A. ORM, query builder, raw SQL: 스펙트럼

데이터베이스 접근의 세 계층이 추상화 vs 제어 축의 다른 지점에 있습니다. 실제 앱은 보통 이들을 섞어 씁니다.

#### A.1 세 계층

| 수준 | 예시 | 작성하는 것 | 실행되는 것 |
|-------|---------|----------------|-----------|
| Raw SQL | `pg.query("SELECT * FROM users WHERE id = $1", [42])` | SQL 문자열 + bind 값 | 정확히 입력한 그대로 |
| Query builder | `knex("users").where("id", 42).first()` | 체이닝 가능한 JS 호출 | 대부분 투명한 생성 SQL |
| ORM | `prisma.user.findUnique({ where: { id: 42 } })` | 모델 객체를 반환하는 객체 모양 쿼리 | 생성 SQL + 객체 hydration + 관계 그래프 |

표를 내려갈수록 제어를 사용성으로 교환합니다. ORM은 타입이 있는 모델 객체, eager 로딩, 마이그레이션 생성을 줍니다. raw SQL은 쿼리 플랜에 대한 정확한 제어를 줍니다.

#### A.2 ORM이 query builder 너머로 더하는 것

Query builder는 SQL을 생성하지만 평범한 행 객체를 반환합니다. ORM은 더 많은 것을 합니다.

1. **객체 hydration.** 행이 타입이 있는 모델 인스턴스(또는 Prisma의 경우 타입이 있는 평범한 객체)가 됩니다. 모양은 스키마 선언과 일치합니다.
2. **관계 그래프.** `include: { posts: true }`가 join(또는 selectinload 스타일 후속 쿼리, 레슨 04 §C.2의 SQLAlchemy처럼)을 수행하고 중첩 객체 그래프를 조립합니다.
3. **진실 원천으로서의 스키마.** 스키마 파일이 마이그레이션, 타입 클라이언트, 검증을 구동합니다 — 한 번 바꾸면 전체 스택이 갱신됩니다.
4. **Identity / 변경 추적** — Prisma는 이를 생략합니다(identity map 없음). SQLAlchemy와 다릅니다. 이는 의도된 단순화입니다. 각 쿼리가 신선한 객체를 반환합니다.

비용: ORM은 어떤 패턴(window 함수, CTE, 복잡한 집계)을 어색하게 만듭니다. 대부분의 ORM은 "탈출구"를 제공합니다 — Prisma의 `prisma.$queryRaw` — SQL이 올바른 도구인 경우를 위해.

#### A.3 N+1 문제가 돌아온다

레슨 04 §C.1의 같은 N+1 함정이 Prisma에도 있습니다.

```javascript
const users = await prisma.user.findMany();
for (const user of users) {
    const posts = await prisma.post.findMany({ where: { userId: user.id } });
}
```

이는 N+1입니다. 사용자 쿼리 1개, posts 쿼리 N개. 해결책은 SQLAlchemy의 `selectinload`와 같습니다.

```javascript
const users = await prisma.user.findMany({ include: { posts: true } });
```

Prisma는 `WHERE userId IN (...)`이라는 단일 후속 쿼리를 발행하고 그래프를 조립합니다. N과 무관하게 총 두 라운드트립.

### B. Prepared statement와 SQL 인젝션

데이터베이스 계층의 가장 큰 결과를 가진 보안 성질은 **사용자 입력이 SQL 문법이 될 수 있는가**입니다. 방어책 — 모든 평판 좋은 데이터베이스 드라이버가 사용 — 은 *prepared statement*입니다.

#### B.1 취약점 모양

어느 언어든 안전하지 않은 패턴:

```javascript
const query = `SELECT * FROM users WHERE name = '${req.body.name}'`;
db.query(query);
```

`req.body.name`이 `' OR '1'='1`이라면, 결과 SQL은 `... WHERE name = '' OR '1'='1'`이 됩니다 — 모든 사용자를 선택합니다. `'; DROP TABLE users; --`라면 테이블을 잃습니다. 근본 원인은 사용자 입력이 SQL 문법으로 연결되었다는 것입니다. 데이터베이스 파서는 어떤 문자가 "데이터"이고 어떤 문자가 "코드"인지 구분할 수 없습니다.

#### B.2 Prepared statement가 하는 일

Prepared statement는 와이어 프로토콜 수준에서 SQL 템플릿과 그 파라미터를 분리합니다.

```
1. 드라이버 송신: "SELECT * FROM users WHERE name = $1"   (PARSE)
2. 데이터베이스가 파싱하고 plan id 반환                    (PARSE COMPLETE)
3. 드라이버 송신: ["alice"]                                (BIND)
4. 데이터베이스가 캐시된 플랜을 bind 값으로 실행            (EXECUTE)
```

데이터베이스는 파라미터를 보기 *전에* 템플릿을 파싱합니다. 파라미터는 플레이스홀더 슬롯에 바인딩됩니다 — SQL이 될 수 있는 문법적 컨텍스트가 없습니다. `' OR '1'='1`은 코드 조각이 아니라 `name`의 리터럴 문자열 값입니다.

이것이 방어입니다. 모든 매개변수화된 쿼리 — `pg.query(sql, params)`, `mysql2`의 `?` 플레이스홀더, Prisma의 `where: { name }` — 가 내부적으로 prepared statement를 사용합니다.

#### B.3 ORM이 "기본적으로 안전"한 이유

Prisma의 API에 사용자 입력을 우연히 보간할 수 없습니다. `prisma.user.findMany({ where: { name } })`은 항상 매개변수화된 쿼리를 만듭니다. 탈출구 `$queryRawUnsafe`가 존재하며, 그 이름이 정확히 그렇게 되어 있어 움찔하게 만듭니다 — 안전한 형제 `$queryRaw`는 매개변수화를 강제하는 tagged-template literal 문법을 사용합니다.

레슨은 일반화됩니다: **문자열만 노출하는 쿼리 인터페이스는 모두 위험하고, 템플릿과 값을 구분하는 인터페이스는 모두 안전합니다**. 후자를 선호하세요.

### C. 트랜잭션과 격리

트랜잭션은 여러 SQL 문을 하나의 원자 단위로 묶습니다. 모두 성공해 함께 보이게 되거나(`COMMIT`), 아무것도 보이지 않습니다(`ROLLBACK`). 원자성 위에는 격리 문제가 있습니다 — 두 트랜잭션이 같은 데이터를 동시에 건드릴 때 무엇이 일어나야 하는가?

#### C.1 ACID, 짧게

- **원자성(Atomicity)** — 모두 아니면 무.
- **일관성(Consistency)** — 제약(`UNIQUE`, `FOREIGN KEY`, `CHECK`)이 모든 commit 경계에서 유지됨.
- **격리(Isolation)** — 동시 트랜잭션이 어떤 직렬 순서로 실행되는 것처럼 보임(선택한 수준에 따라).
- **지속성(Durability)** — commit된 트랜잭션이 충돌에서 살아남음.

원자성은 래퍼 관심사입니다: `BEGIN`, 문 실행, `COMMIT` 또는 `ROLLBACK`. 격리가 깊은 문제입니다.

#### C.2 ANSI 4가지 격리 수준

각 수준은 그 다음 약한 수준이 허용하는 특정 부류의 이상 현상을 막습니다.

| 수준 | Dirty read | Non-repeatable read | Phantom read |
|-------|------------|---------------------|--------------|
| READ UNCOMMITTED | 가능 | 가능 | 가능 |
| READ COMMITTED (PostgreSQL 기본) | 방지 | 가능 | 가능 |
| REPEATABLE READ (MySQL 기본) | 방지 | 방지 | 가능 (PG: 또한 방지) |
| SERIALIZABLE | 방지 | 방지 | 방지 |

- **Dirty read.** 트랜잭션 A가 B의 commit되지 않은 쓰기를 봅니다. B가 롤백하면 A가 쓰레기를 읽은 것이 됩니다.
- **Non-repeatable read.** A가 같은 행을 두 번 읽었는데 그 사이에 B가 commit해서 다른 값을 얻습니다.
- **Phantom read.** A가 같은 쿼리를 다시 돌렸는데 B가 INSERT해서 새 행이 보입니다.

높은 격리는 더 안전하지만 더 느립니다(잠금 증가, 낙관적 동시성 제어에서 abort 증가). 대부분의 앱은 READ COMMITTED에서 돌리고 남은 이상 현상은 명시적으로 처리합니다. 행 잠금을 위한 `SELECT FOR UPDATE`, 낙관적 버전 검사(`UPDATE ... WHERE version = ?`), 또는 알려진 임계 경로에 대해 SERIALIZABLE로 격상.

#### C.3 Prisma의 트랜잭션

```javascript
await prisma.$transaction(async (tx) => {
    const order = await tx.order.create({ data: { ... } });
    await tx.inventory.update({ where: { sku }, data: { qty: { decrement: 1 } } });
});
```

콜백이 단일 트랜잭션에서 실행됩니다. 무엇이든 throw하면 전체가 롤백됩니다. `tx` 파라미터는 그 트랜잭션에 한정된 Prisma 클라이언트입니다 — 콜백 안에서 바깥의 `prisma`를 쓰면 트랜잭션 *바깥*에서 실행되어 의미가 사라집니다.

Prisma는 또한 콜백 없이 모든 쿼리를 한 트랜잭션에서 실행하는 "interactive batch" 형태(`prisma.$transaction([query1, query2])`)도 지원합니다. 쿼리들이 서로의 결과에 의존하지 않을 때 유용합니다.

#### C.4 연결 풀이 중요한 이유

트랜잭션은 commit이나 롤백까지 연결을 잡고 있습니다. `pool_size`가 10이면 동시에 10개의 열린 트랜잭션을 가질 수 있습니다. 그 이상이면 호출자가 기다립니다. 오래 걸리는 트랜잭션은 다른 모든 요청을 굶깁니다 — 레슨 04 §A.1과 같은 풀 고갈 패턴이지만, 긴 트랜잭션은 행 잠금까지 잡고 있어 더 위험합니다.

규율: 트랜잭션을 짧게 유지하세요. 트랜잭션 안에서 외부 HTTP 서비스를 호출하거나 사용자 입력을 기다리지 마세요. 꼭 그래야 한다면, 트랜잭션이 데이터베이스 작업만 감싸도록 재설계하세요.

### 이론에서 아래 코드로

뒤에 나오는 각 절은 이 틀의 한 조각을 구체화합니다.

- §1 (Prisma 개요)는 §A.2의 진실 원천으로서의 스키마 모델을 도입합니다.
- §2 (설정)은 Prisma를 §A 연결 모델에 배선하고 타입 안전 클라이언트를 부트스트랩합니다.
- §3 (스키마 정의)는 §A.2 생성을 구동하는 선언적 원천입니다: 타입, 마이그레이션, 검증.
- §4 (CRUD 작업)은 §B.2 prepared statement를 자동으로 만들어 내는 타입 안전 API입니다.
- §5 (관계)는 §A.3을 구체화합니다 — `include`로 N+1 함정을 무찌릅니다.
- §6 (마이그레이션)은 스키마 진화 도구입니다. 각 스키마 변경이 버전 관리된 마이그레이션 스크립트가 됩니다.
- §7 (쿼리 최적화)는 §A.3을 정련합니다: 필요한 컬럼만 가져오는 `select`와 인덱싱 전략.
- §8 (트랜잭션)은 §C 원자성 래퍼이며, §C.4의 풀 고갈 주의사항이 따라옵니다.
- §9 (PostgreSQL 연결)은 레슨 04 §A.1에서 자세히 논의된 §A 연결 풀 구성입니다.

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
