/**
 * Express Database — Prisma ORM with PostgreSQL
 * Demonstrates: Prisma client, CRUD operations, transactions, pagination.
 *
 * Setup:
 *   npm install express @prisma/client
 *   npx prisma init --datasource-provider postgresql
 *
 * schema.prisma example:
 *   model User {
 *     id        Int      @id @default(autoincrement())
 *     email     String   @unique
 *     name      String
 *     posts     Post[]
 *     createdAt DateTime @default(now())
 *   }
 *
 *   model Post {
 *     id        Int      @id @default(autoincrement())
 *     title     String
 *     content   String?
 *     published Boolean  @default(false)
 *     author    User     @relation(fields: [authorId], references: [id])
 *     authorId  Int
 *     createdAt DateTime @default(now())
 *   }
 *
 *   npx prisma migrate dev --name init
 */

const express = require('express');
// const { PrismaClient } = require('@prisma/client');

const app = express();
app.use(express.json());

// Simulated Prisma client for demonstration
// In production: const prisma = new PrismaClient();
const prisma = {
  user: {
    async findMany({ skip, take, include, orderBy } = {}) {
      return [{ id: 1, email: 'alice@example.com', name: 'Alice', posts: [] }];
    },
    async findUnique({ where, include } = {}) {
      if (where.id === 1) return { id: 1, email: 'alice@example.com', name: 'Alice' };
      return null;
    },
    async create({ data }) {
      return { id: 2, ...data, createdAt: new Date() };
    },
    async update({ where, data }) {
      return { id: where.id, ...data };
    },
    async delete({ where }) {
      return { id: where.id };
    },
    async count() {
      return 1;
    },
  },
  post: {
    async findMany({ where, include, skip, take } = {}) {
      return [];
    },
    async create({ data }) {
      return { id: 1, ...data, createdAt: new Date() };
    },
  },
  $transaction: async (fn) => fn(prisma),
};

// --- User CRUD ---

// List users with pagination
app.get('/api/users', async (req, res, next) => {
  try {
    const page = Math.max(1, Number(req.query.page) || 1);
    const limit = Math.min(50, Math.max(1, Number(req.query.limit) || 10));
    const skip = (page - 1) * limit;

    const [users, total] = await Promise.all([
      prisma.user.findMany({
        skip,
        take: limit,
        include: { posts: true },
        orderBy: { createdAt: 'desc' },
      }),
      prisma.user.count(),
    ]);

    res.json({
      data: users,
      meta: {
        total,
        page,
        limit,
        pages: Math.ceil(total / limit),
      },
    });
  } catch (err) {
    next(err);
  }
});

// Get single user
app.get('/api/users/:id', async (req, res, next) => {
  try {
    const user = await prisma.user.findUnique({
      where: { id: Number(req.params.id) },
      include: { posts: true },
    });
    if (!user) return res.status(404).json({ error: 'User not found' });
    res.json(user);
  } catch (err) {
    next(err);
  }
});

// Create user
app.post('/api/users', async (req, res, next) => {
  try {
    const { email, name } = req.body;
    if (!email || !name) {
      return res.status(400).json({ error: 'email and name are required' });
    }
    const user = await prisma.user.create({
      data: { email, name },
    });
    res.status(201).json(user);
  } catch (err) {
    // Handle unique constraint violation
    if (err.code === 'P2002') {
      return res.status(409).json({ error: 'Email already exists' });
    }
    next(err);
  }
});

// Create user with posts (transaction)
app.post('/api/users/with-posts', async (req, res, next) => {
  try {
    const { email, name, posts } = req.body;

    const result = await prisma.$transaction(async (tx) => {
      const user = await tx.user.create({ data: { email, name } });
      const createdPosts = await Promise.all(
        (posts || []).map((p) =>
          tx.post.create({ data: { ...p, authorId: user.id } })
        )
      );
      return { ...user, posts: createdPosts };
    });

    res.status(201).json(result);
  } catch (err) {
    next(err);
  }
});

// --- Error Handler ---
app.use((err, req, res, next) => {
  console.error(err);
  res.status(500).json({ error: 'Internal server error' });
});

app.listen(3000, () => console.log('Prisma demo on http://localhost:3000'));
