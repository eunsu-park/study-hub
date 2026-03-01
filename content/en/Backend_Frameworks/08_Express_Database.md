# 08. Express Database

**Previous**: [Express Advanced](./07_Express_Advanced.md) | **Next**: [Express Testing](./09_Express_Testing.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Set up Prisma ORM in an Express project and connect it to PostgreSQL
2. Design database schemas using Prisma's declarative schema language with relations
3. Perform CRUD operations using Prisma Client's type-safe query API
4. Manage database schema evolution with Prisma Migrate
5. Optimize queries using `select`, `include`, and transactions for data integrity

---

Every serious backend application needs persistent data storage. While you could write raw SQL queries, an ORM (Object-Relational Mapper) provides a higher-level, type-safe interface that reduces boilerplate and prevents common mistakes like SQL injection. This lesson covers Prisma, the most popular modern ORM for Node.js, and shows how to integrate it with Express to build data-driven APIs.

> **Why Prisma over Sequelize or TypeORM?** Prisma takes a schema-first approach: you declare your data model in a `.prisma` file, and it generates a fully type-safe client. This catches errors at build time rather than at runtime. Its migration system is also simpler and more predictable.

## Table of Contents

1. [Prisma ORM Overview](#1-prisma-orm-overview)
2. [Project Setup](#2-project-setup)
3. [Schema Definition](#3-schema-definition)
4. [Prisma Client: CRUD Operations](#4-prisma-client-crud-operations)
5. [Relations](#5-relations)
6. [Migrations](#6-migrations)
7. [Query Optimization](#7-query-optimization)
8. [Transaction Handling](#8-transaction-handling)
9. [Connecting to PostgreSQL](#9-connecting-to-postgresql)
10. [Practice Problems](#10-practice-problems)

---

## 1. Prisma ORM Overview

Prisma consists of three core components:

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

| Component | Purpose |
|-----------|---------|
| **Prisma Schema** | Single source of truth for your database structure |
| **Prisma Client** | Auto-generated, type-safe query builder |
| **Prisma Migrate** | Declarative migration system based on schema diffs |
| **Prisma Studio** | GUI for viewing and editing data (development tool) |

---

## 2. Project Setup

### Installation

```bash
# Initialize a new Express project (if not already done)
mkdir express-prisma && cd express-prisma
npm init -y
npm install express
npm install -D prisma

# Initialize Prisma — creates prisma/schema.prisma and .env
npx prisma init

# Install the Prisma Client runtime
npm install @prisma/client
```

### Project Structure

```
express-prisma/
├── prisma/
│   ├── schema.prisma     # Data model definition
│   └── migrations/       # Generated migration SQL files
├── src/
│   ├── app.js
│   ├── server.js
│   ├── lib/
│   │   └── prisma.js     # Singleton Prisma Client instance
│   └── routes/
│       └── users.js
├── .env                  # DATABASE_URL
└── package.json
```

### Prisma Client Singleton

```javascript
// src/lib/prisma.js
// Create a single PrismaClient instance and reuse it across the application.
// Creating multiple instances wastes database connections and can exhaust
// the connection pool under load.
import { PrismaClient } from '@prisma/client';

const prisma = new PrismaClient({
  // Log queries in development — helps debug N+1 problems and slow queries
  log: process.env.NODE_ENV === 'development' ? ['query', 'warn', 'error'] : ['error'],
});

export default prisma;
```

---

## 3. Schema Definition

The Prisma schema file (`prisma/schema.prisma`) defines your data model, database connection, and generator configuration.

### Basic Schema

```prisma
// prisma/schema.prisma

// Generator tells Prisma what to produce — here, the JavaScript client
generator client {
  provider = "prisma-client-js"
}

// Datasource configures the database connection
datasource db {
  provider = "postgresql"
  url      = env("DATABASE_URL")  // Read from .env file
}

// Model maps to a database table — each field becomes a column
model User {
  id        Int      @id @default(autoincrement())
  email     String   @unique
  name      String
  role      Role     @default(USER)
  createdAt DateTime @default(now())  // Auto-set on creation
  updatedAt DateTime @updatedAt       // Auto-updated on every modification

  // Relation: one user has many posts
  posts     Post[]

  @@map("users")  // Override table name in the database
}

model Post {
  id          Int      @id @default(autoincrement())
  title       String   @db.VarChar(200)  // Map to specific SQL type
  content     String?                     // ? makes the field nullable
  published   Boolean  @default(false)
  authorId    Int
  createdAt   DateTime @default(now())
  updatedAt   DateTime @updatedAt

  // Foreign key — links post to its author
  author      User     @relation(fields: [authorId], references: [id])
  tags        Tag[]    // Many-to-many via implicit join table

  // Composite index — speeds up queries that filter by author + published status
  @@index([authorId, published])
  @@map("posts")
}

model Tag {
  id    Int    @id @default(autoincrement())
  name  String @unique
  posts Post[] // Many-to-many (Prisma creates join table automatically)

  @@map("tags")
}

enum Role {
  USER
  ADMIN
  MODERATOR
}
```

### Field Types Reference

| Prisma Type | PostgreSQL Type | Notes |
|-------------|----------------|-------|
| `String` | `text` | Use `@db.VarChar(n)` for length limit |
| `Int` | `integer` | 32-bit integer |
| `BigInt` | `bigint` | 64-bit integer |
| `Float` | `double precision` | Floating point |
| `Decimal` | `decimal(65,30)` | Exact precision (money, etc.) |
| `Boolean` | `boolean` | true/false |
| `DateTime` | `timestamp(3)` | Millisecond precision |
| `Json` | `jsonb` | Structured JSON data |

---

## 4. Prisma Client: CRUD Operations

After defining the schema, generate the client and use it in your routes:

```bash
# Generate Prisma Client — must be run after every schema change
npx prisma generate
```

### Create

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
    // Prisma throws P2002 for unique constraint violations —
    // catch it here to return a friendly error instead of a 500
    if (err.code === 'P2002') {
      return res.status(409).json({ error: `Email already exists` });
    }
    next(err);
  }
});

// Create with nested relation — creates a user and their posts in one query
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
      include: { posts: true }, // Return the created posts in the response
    });

    res.status(201).json(user);
  } catch (err) {
    next(err);
  }
});
```

### Read

```javascript
// Find many with filtering, pagination, and sorting
router.get('/', async (req, res, next) => {
  try {
    const { page = 1, limit = 10, role, search } = req.query;
    const skip = (parseInt(page) - 1) * parseInt(limit);

    // Build where clause dynamically — only include filters that are provided
    const where = {};
    if (role) where.role = role;
    if (search) {
      where.OR = [
        { name: { contains: search, mode: 'insensitive' } },
        { email: { contains: search, mode: 'insensitive' } },
      ];
    }

    // Execute count and findMany in parallel — avoids two sequential DB round trips
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

// Find one by ID
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

### Update

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
    // P2025: Record to update not found
    if (err.code === 'P2025') {
      return res.status(404).json({ error: 'User not found' });
    }
    next(err);
  }
});

// Upsert — create if not exists, update if exists
// Useful for "save" operations where the client does not know if the record exists
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

### Delete

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

## 5. Relations

### One-to-Many

```prisma
// One user has many posts (already shown in schema above)
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
// Query with relation
const userWithPosts = await prisma.user.findUnique({
  where: { id: 1 },
  include: { posts: true },
});

// Create post for existing user — connect links to an existing record
// without needing to know the foreign key column name
const post = await prisma.post.create({
  data: {
    title: 'New Post',
    author: { connect: { id: 1 } },
  },
});
```

### Many-to-Many

```prisma
// Prisma manages the join table automatically — you never interact with it directly
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
// Create post with new and existing tags
const post = await prisma.post.create({
  data: {
    title: 'Prisma Guide',
    author: { connect: { id: 1 } },
    tags: {
      // connectOrCreate avoids duplicates — creates the tag only if it
      // does not already exist, otherwise connects to the existing one
      connectOrCreate: [
        { where: { name: 'prisma' }, create: { name: 'prisma' } },
        { where: { name: 'orm' }, create: { name: 'orm' } },
      ],
    },
  },
  include: { tags: true },
});

// Find posts by tag
const postsWithTag = await prisma.post.findMany({
  where: {
    tags: { some: { name: 'prisma' } }, // "some" = at least one related tag matches
  },
  include: { tags: true },
});
```

### Self-Relations

```prisma
// A comment can have replies — the parent and children are both Comment records
model Comment {
  id       Int       @id @default(autoincrement())
  text     String
  parentId Int?
  parent   Comment?  @relation("CommentReplies", fields: [parentId], references: [id])
  replies  Comment[] @relation("CommentReplies")
}
```

---

## 6. Migrations

Prisma Migrate tracks schema changes as versioned SQL migration files.

### Development Workflow

```bash
# Create a migration from schema changes — generates SQL and applies it
# The name should describe what changed for easy reading of migration history
npx prisma migrate dev --name add_user_model

# This command:
# 1. Detects schema diff (what changed since last migration)
# 2. Generates a SQL migration file in prisma/migrations/
# 3. Applies the migration to the database
# 4. Re-generates Prisma Client
```

### Migration Files

```
prisma/migrations/
├── 20250601120000_add_user_model/
│   └── migration.sql
├── 20250602150000_add_post_model/
│   └── migration.sql
└── migration_lock.toml    # Locks the database provider
```

```sql
-- prisma/migrations/20250601120000_add_user_model/migration.sql
-- Auto-generated by Prisma Migrate — do not edit manually unless necessary
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

### Production Deployment

```bash
# In production, use `deploy` instead of `dev` — it only applies pending
# migrations without creating new ones or re-generating the client
npx prisma migrate deploy

# Reset database (DESTRUCTIVE — drops all data)
# Only use in development when migration history becomes inconsistent
npx prisma migrate reset
```

### Seeding

```javascript
// prisma/seed.js — populate database with initial or test data
import { PrismaClient } from '@prisma/client';

const prisma = new PrismaClient();

async function main() {
  // Upsert prevents errors when seed runs multiple times
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
// Add to package.json — Prisma calls this script automatically on `prisma migrate reset`
{
  "prisma": {
    "seed": "node prisma/seed.js"
  }
}
```

```bash
# Run seed manually
npx prisma db seed
```

---

## 7. Query Optimization

### Select vs Include

```javascript
// select — fetch only specific fields; reduces data transfer
// Use when you need a subset of columns (e.g., dropdown lists, search results)
const userNames = await prisma.user.findMany({
  select: {
    id: true,
    name: true,
    email: true,
    // posts: false — not selecting posts means no JOIN is performed
  },
});

// include — fetch the full model plus related records
// Triggers a JOIN or secondary query; use only when you need the related data
const usersWithPosts = await prisma.user.findMany({
  include: {
    posts: {
      select: { id: true, title: true },  // Limit related fields too
      where: { published: true },
      take: 5,                              // Only load latest 5 posts
      orderBy: { createdAt: 'desc' },
    },
  },
});

// WARNING: You cannot use select and include at the same level.
// Choose one approach per query.
```

### Avoiding N+1 Queries

```javascript
// BAD: N+1 problem — 1 query for users, then 1 query per user for posts
const users = await prisma.user.findMany();
for (const user of users) {
  // Each iteration fires a separate database query
  const posts = await prisma.post.findMany({ where: { authorId: user.id } });
  user.posts = posts;
}

// GOOD: Single query with include — Prisma generates an efficient JOIN or IN clause
const usersWithPosts = await prisma.user.findMany({
  include: { posts: true },
});
```

### Raw Queries

```javascript
// For complex queries that Prisma Client cannot express,
// use raw SQL while still benefiting from Prisma's connection pooling

// Tagged template — parameters are automatically escaped to prevent SQL injection
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

## 8. Transaction Handling

Transactions ensure that a group of operations either all succeed or all fail. This is essential for maintaining data consistency.

### Sequential Transactions

```javascript
// prisma.$transaction() wraps operations in a database transaction —
// if any operation fails, all preceding operations are rolled back
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

### Interactive Transactions

```javascript
// Interactive transactions give you a transaction client (tx) with full
// Prisma Client capabilities — useful for conditional logic within a transaction
const createOrder = async (userId, items) => {
  return prisma.$transaction(async (tx) => {
    // Check stock levels within the transaction
    for (const item of items) {
      const product = await tx.product.findUnique({
        where: { id: item.productId },
      });

      if (!product || product.stock < item.quantity) {
        // Throwing inside the callback rolls back the entire transaction
        throw new Error(`Insufficient stock for ${product?.name ?? item.productId}`);
      }

      // Decrement stock
      await tx.product.update({
        where: { id: item.productId },
        data: { stock: { decrement: item.quantity } },
      });
    }

    // Create the order with line items
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
    maxWait: 5000,  // Max time to wait for a transaction slot (ms)
    timeout: 10000, // Max transaction duration (ms) — prevents long-running locks
  });
};
```

---

## 9. Connecting to PostgreSQL

### Environment Configuration

```bash
# .env
# Format: postgresql://USER:PASSWORD@HOST:PORT/DATABASE?schema=SCHEMA
DATABASE_URL="postgresql://myuser:mypassword@localhost:5432/mydb?schema=public"

# For connection pooling (production) — PgBouncer or Prisma Accelerate
# DATABASE_URL="postgresql://myuser:mypassword@pgbouncer:6432/mydb?pgbouncer=true"
```

### Docker Setup for Local Development

```yaml
# docker-compose.yml — run PostgreSQL locally without installing it
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
      - pgdata:/var/lib/postgresql/data  # Persist data across container restarts

volumes:
  pgdata:
```

```bash
# Start PostgreSQL
docker compose up -d

# Apply migrations
npx prisma migrate dev

# Open Prisma Studio — browser-based data viewer
npx prisma studio
```

### Graceful Shutdown

```javascript
// src/server.js — disconnect Prisma when the process exits
// Without this, database connections may leak during restarts or deployments
import app from './app.js';
import prisma from './lib/prisma.js';

const PORT = process.env.PORT || 3000;

const server = app.listen(PORT, () => {
  console.log(`Server running on http://localhost:${PORT}`);
});

// Handle shutdown signals from container orchestrators (Docker, K8s)
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

## 10. Practice Problems

### Problem 1: Blog API

Build a complete blog API with these models and endpoints:
- **User**: id, email (unique), name, bio (optional)
- **Post**: id, title, content, published (default false), authorId
- Endpoints: CRUD for users and posts, `GET /api/users/:id/posts`, toggle publish status

### Problem 2: Pagination and Filtering

Add cursor-based pagination to the posts endpoint:
- `GET /api/posts?cursor=42&take=20` returns 20 posts starting after ID 42
- Support filtering by `published` status and searching by `title`
- Return `nextCursor` in the response for the client to use in the next request

### Problem 3: Many-to-Many with Tags

Extend the blog with a Tag model:
- `POST /api/posts` accepts a `tags` array of strings
- Use `connectOrCreate` so existing tags are reused
- `GET /api/tags/:name/posts` returns all posts with that tag
- `DELETE /api/tags/:id` should only work if no posts use the tag

### Problem 4: Transactional Transfer

Implement a "credit transfer" system:
- Users have a `balance` field (Decimal)
- `POST /api/transfers` accepts `{ fromId, toId, amount }`
- Use an interactive transaction to validate the sender has sufficient balance
- Create a `Transfer` record as an audit log
- Return 400 if the balance is insufficient (without modifying any data)

### Problem 5: Schema Migration

Starting from the blog schema, make these changes using Prisma Migrate:
1. Add a `Category` model with `id` and `name` fields
2. Add a `categoryId` foreign key to `Post` (optional, nullable)
3. Add a composite unique constraint on `Post(title, authorId)` -- no duplicate titles per author
4. Run the migration and verify with Prisma Studio

---

## References

- [Prisma Documentation](https://www.prisma.io/docs)
- [Prisma Schema Reference](https://www.prisma.io/docs/reference/api-reference/prisma-schema-reference)
- [Prisma Client API](https://www.prisma.io/docs/reference/api-reference/prisma-client-reference)
- [Prisma Migrate Guide](https://www.prisma.io/docs/guides/migrate)
- [PostgreSQL Documentation](https://www.postgresql.org/docs/)

---

**Previous**: [Express Advanced](./07_Express_Advanced.md) | **Next**: [Express Testing](./09_Express_Testing.md)
