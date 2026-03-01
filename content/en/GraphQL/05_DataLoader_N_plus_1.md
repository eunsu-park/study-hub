# 05. DataLoader and the N+1 Problem

**Previous**: [Resolvers](./04_Resolvers.md) | **Next**: [Subscriptions](./06_Subscriptions.md)

---

The N+1 problem is the most common performance pitfall in GraphQL. Because each field resolver is independent, fetching a list of items and then resolving a relationship for each item creates an explosion of database queries. DataLoader solves this by batching and caching individual loads within a single request tick. This lesson explains why the N+1 problem occurs, how DataLoader fixes it, and how to implement DataLoader correctly in production.

**Difficulty**: ⭐⭐⭐⭐

## Learning Objectives

After completing this lesson, you will be able to:

1. Identify N+1 query patterns in GraphQL resolvers and calculate the total query count
2. Explain how DataLoader's batching mechanism coalesces individual loads into a single batch call
3. Implement DataLoader batch functions for different data relationships (one-to-one, one-to-many)
4. Configure DataLoader's caching behavior and understand per-request scoping
5. Monitor and detect N+1 problems in production using query logging and tracing

---

## Table of Contents

1. [The N+1 Problem Explained](#1-the-n1-problem-explained)
2. [Why GraphQL Is Especially Prone to N+1](#2-why-graphql-is-especially-prone-to-n1)
3. [Introducing DataLoader](#3-introducing-dataloader)
4. [Batching: How It Works](#4-batching-how-it-works)
5. [Implementing DataLoader](#5-implementing-dataloader)
6. [DataLoader for One-to-Many Relationships](#6-dataloader-for-one-to-many-relationships)
7. [Caching Behavior](#7-caching-behavior)
8. [Per-Request DataLoader Instances](#8-per-request-dataloader-instances)
9. [DataLoader with Different Data Sources](#9-dataloader-with-different-data-sources)
10. [Monitoring and Detecting N+1](#10-monitoring-and-detecting-n1)
11. [Common Pitfalls](#11-common-pitfalls)
12. [Practice Problems](#12-practice-problems)
13. [References](#13-references)

---

## 1. The N+1 Problem Explained

The N+1 problem occurs when fetching a list of N items requires 1 query for the list plus N additional queries for a related entity on each item.

### 1.1 A Concrete Example

Consider this schema and query:

```graphql
type Query {
  posts(first: Int): [Post!]!
}

type Post {
  id: ID!
  title: String!
  author: User!        # Each post has one author
}

type User {
  id: ID!
  name: String!
}
```

```graphql
query {
  posts(first: 10) {
    title
    author {
      name
    }
  }
}
```

And these naive resolvers:

```javascript
const resolvers = {
  Query: {
    posts: (_, { first }, ctx) => {
      return ctx.db.query('SELECT * FROM posts LIMIT $1', [first]);
      // Query 1: SELECT * FROM posts LIMIT 10
    },
  },
  Post: {
    author: (parent, _, ctx) => {
      return ctx.db.query('SELECT * FROM users WHERE id = $1', [parent.author_id]);
      // Query 2:  SELECT * FROM users WHERE id = 'user_1'
      // Query 3:  SELECT * FROM users WHERE id = 'user_2'
      // Query 4:  SELECT * FROM users WHERE id = 'user_1'  ← duplicate!
      // Query 5:  SELECT * FROM users WHERE id = 'user_3'
      // ...
      // Query 11: SELECT * FROM users WHERE id = 'user_5'
    },
  },
};
```

**Result: 1 + 10 = 11 queries** for a simple page of 10 posts. If several posts share the same author, we are even fetching the same user multiple times.

### 1.2 Scaling the Problem

The N+1 problem compounds with nesting:

```graphql
query {
  posts(first: 10) {           # 1 query
    title
    author {                    # 10 queries
      name
      followers(first: 5) {     # 10 queries (one per author)
        name                    # 50 queries (one per follower)
      }
    }
    comments(first: 5) {       # 10 queries
      body
      author {                  # 50 queries
        name
      }
    }
  }
}
```

**Total: 1 + 10 + 10 + 50 + 10 + 50 = 131 queries** for a single GraphQL request. In a REST API, the server controls the data fetching strategy. In GraphQL, the client controls the query shape, so the server must defend against arbitrary nesting.

## 2. Why GraphQL Is Especially Prone to N+1

### 2.1 Resolver Independence

Each resolver is a standalone function that does not know about other resolvers. `Post.author` does not know that it will be called 10 times for 10 different posts. It simply receives one parent object and returns one result.

```
Query.posts → [post1, post2, ..., post10]
  ├── Post.author(post1)  → db.query("SELECT ... WHERE id = 1")
  ├── Post.author(post2)  → db.query("SELECT ... WHERE id = 2")
  ├── Post.author(post3)  → db.query("SELECT ... WHERE id = 1")  ← duplicate
  ├── Post.author(post4)  → db.query("SELECT ... WHERE id = 3")
  └── ... (10 separate queries)
```

### 2.2 Client-controlled Depth

REST APIs have fixed endpoints with predictable query patterns. GraphQL lets clients nest arbitrarily deep:

```graphql
# A client could send this
query DeeplyNested {
  user(id: "1") {
    posts {
      comments {
        author {
          posts {
            comments {
              author { name }
            }
          }
        }
      }
    }
  }
}
```

Without protections, this creates a cascade of N+1 problems at every level.

### 2.3 Why Not Just JOIN?

You might think: "Just use SQL JOINs!" The problem is that resolvers do not know the full query shape at the time they execute. `Query.posts` does not know that `Post.author` will also be requested --- it just returns the posts. And even if it did, eager-loading everything "just in case" wastes resources when the client does not request those fields.

DataLoader provides a better solution: **deferred batching**.

## 3. Introducing DataLoader

DataLoader is a utility library (originally created by Facebook) that provides two key features:

1. **Batching**: Collects individual loads within a single event loop tick and executes them as a batch
2. **Caching**: Deduplicates requests for the same key within a single request

```
Without DataLoader                With DataLoader
──────────────────               ─────────────────
Post.author(id: 1) → SQL query   Post.author(id: 1) → loader.load(1)
Post.author(id: 2) → SQL query   Post.author(id: 2) → loader.load(2)
Post.author(id: 1) → SQL query   Post.author(id: 1) → loader.load(1) ← cached!
Post.author(id: 3) → SQL query   Post.author(id: 3) → loader.load(3)
                                  ──── tick boundary ────
4 SQL queries                     1 SQL query: WHERE id IN (1, 2, 3)
                                  + cache hit for id: 1
```

### 3.1 Installation

```bash
npm install dataloader
```

### 3.2 Basic Usage

```javascript
import DataLoader from 'dataloader';

// 1. Define a batch function
//    Input: array of keys [1, 2, 3]
//    Output: Promise of array of values, in the same order
const userLoader = new DataLoader(async (userIds) => {
  // Single query that fetches all requested users at once
  const users = await db.query(
    'SELECT * FROM users WHERE id = ANY($1)',
    [userIds]
  );

  // CRITICAL: Return results in the same order as the input keys
  const userMap = new Map(users.map(u => [u.id, u]));
  return userIds.map(id => userMap.get(id) || null);
});

// 2. Use it in resolvers (instead of direct DB queries)
const author = await userLoader.load('user_1');   // Queued
const author2 = await userLoader.load('user_2');  // Queued
const author3 = await userLoader.load('user_1');  // Cache hit!
// At the next tick: SELECT * FROM users WHERE id IN ('user_1', 'user_2')
```

## 4. Batching: How It Works

DataLoader's batching relies on Node.js's event loop. Here is the mechanism step by step:

### 4.1 The Event Loop Trick

```
Time  │  Resolver Execution                   DataLoader Internal State
──────┼──────────────────────────────────     ─────────────────────────
 t0   │  Post.author(post1) → load("u1")     queue: ["u1"]
 t0   │  Post.author(post2) → load("u2")     queue: ["u1", "u2"]
 t0   │  Post.author(post3) → load("u1")     cache hit → return cached Promise
 t0   │  Post.author(post4) → load("u3")     queue: ["u1", "u2", "u3"]
      │
 t1   │  ── process.nextTick / microtask ──   Batch function called!
      │                                        batchFn(["u1", "u2", "u3"])
      │                                        → SELECT ... WHERE id IN ("u1","u2","u3")
      │
 t2   │  All Promises resolve                 Results distributed to callers
```

The key insight: GraphQL resolves sibling fields in the same tick (using `Promise.all`). DataLoader schedules the batch function on `process.nextTick`, so all loads from the current tick are collected before the batch executes.

### 4.2 Batch Scheduling Visualization

```
┌─────────────────────────────────────────────┐
│            JavaScript Event Loop             │
│                                              │
│  ┌──────── Microtask Queue ───────────┐     │
│  │ load("u1") → add to batch          │     │
│  │ load("u2") → add to batch          │     │
│  │ load("u1") → cache hit             │     │
│  │ load("u3") → add to batch          │     │
│  └────────────────────────────────────┘     │
│                    │                         │
│                    ▼                         │
│  ┌──────── Next Tick ─────────────────┐     │
│  │ batchFn(["u1", "u2", "u3"])        │     │
│  │ → 1 database query                  │     │
│  │ → resolve all pending Promises      │     │
│  └────────────────────────────────────┘     │
└─────────────────────────────────────────────┘
```

## 5. Implementing DataLoader

### 5.1 The Batch Function Contract

The batch function must follow strict rules:

```javascript
// The batch function receives an array of keys
// and must return a Promise of an array of values
// where values[i] corresponds to keys[i]
async function batchUsers(userIds) {
  // ✅ Rule 1: Return the same number of elements as keys
  // ✅ Rule 2: Return elements in the same order as keys
  // ✅ Rule 3: Return an Error instance for failed lookups (not null, not throw)

  const users = await db.query(
    'SELECT * FROM users WHERE id = ANY($1)',
    [userIds]
  );

  // Create a map for O(1) lookups
  const userMap = new Map(users.map(u => [u.id, u]));

  // Map each input key to its result, preserving order
  return userIds.map(id =>
    userMap.get(id) || new Error(`User ${id} not found`)
  );
}

const userLoader = new DataLoader(batchUsers);
```

### 5.2 One-to-One Relationship: Post → Author

```javascript
// Schema: Post.author resolves to a single User
function createUserByIdLoader(db) {
  return new DataLoader(async (userIds) => {
    console.log(`Batch loading users: [${userIds.join(', ')}]`);

    const users = await db.query(
      'SELECT * FROM users WHERE id = ANY($1)',
      [userIds]
    );

    const userMap = new Map(users.map(u => [u.id, u]));
    return userIds.map(id => userMap.get(id) || null);
  });
}

// Resolver
const resolvers = {
  Post: {
    author: (parent, _, ctx) => ctx.loaders.userById.load(parent.authorId),
  },
};
```

### 5.3 Complete DataLoader Factory

```javascript
// dataloaders.js
import DataLoader from 'dataloader';

export function createDataLoaders(db) {
  return {
    // One-to-one: load a single entity by ID
    userById: new DataLoader(async (ids) => {
      const users = await db.user.findMany({ where: { id: { in: [...ids] } } });
      const map = new Map(users.map(u => [u.id, u]));
      return ids.map(id => map.get(id) || null);
    }),

    postById: new DataLoader(async (ids) => {
      const posts = await db.post.findMany({ where: { id: { in: [...ids] } } });
      const map = new Map(posts.map(p => [p.id, p]));
      return ids.map(id => map.get(id) || null);
    }),

    // One-to-many: load arrays of related entities
    postsByAuthorId: new DataLoader(async (authorIds) => {
      const posts = await db.post.findMany({
        where: { authorId: { in: [...authorIds] } },
        orderBy: { createdAt: 'desc' },
      });
      const grouped = new Map();
      for (const post of posts) {
        if (!grouped.has(post.authorId)) grouped.set(post.authorId, []);
        grouped.get(post.authorId).push(post);
      }
      return authorIds.map(id => grouped.get(id) || []);
    }),

    commentsByPostId: new DataLoader(async (postIds) => {
      const comments = await db.comment.findMany({
        where: { postId: { in: [...postIds] } },
        orderBy: { createdAt: 'asc' },
      });
      const grouped = new Map();
      for (const comment of comments) {
        if (!grouped.has(comment.postId)) grouped.set(comment.postId, []);
        grouped.get(comment.postId).push(comment);
      }
      return postIds.map(id => grouped.get(id) || []);
    }),

    // Aggregation: load counts
    commentCountByPostId: new DataLoader(async (postIds) => {
      const results = await db.$queryRaw`
        SELECT post_id, COUNT(*)::int as count
        FROM comments
        WHERE post_id = ANY(${postIds})
        GROUP BY post_id
      `;
      const map = new Map(results.map(r => [r.post_id, r.count]));
      return postIds.map(id => map.get(id) || 0);
    }),
  };
}
```

## 6. DataLoader for One-to-Many Relationships

One-to-many relationships require special handling because each key maps to an array of results.

### 6.1 The Pattern

```javascript
// User.posts: one user → many posts
const postsByAuthorIdLoader = new DataLoader(async (authorIds) => {
  // 1. Fetch all posts for all requested authors in one query
  const posts = await db.query(
    'SELECT * FROM posts WHERE author_id = ANY($1) ORDER BY created_at DESC',
    [authorIds]
  );

  // 2. Group posts by author_id
  const postsByAuthor = new Map();
  for (const post of posts) {
    if (!postsByAuthor.has(post.author_id)) {
      postsByAuthor.set(post.author_id, []);
    }
    postsByAuthor.get(post.author_id).push(post);
  }

  // 3. Return arrays in the same order as input keys
  //    Missing keys get empty arrays (not null, not Error)
  return authorIds.map(id => postsByAuthor.get(id) || []);
});
```

### 6.2 One-to-Many with Limits

What if the schema allows pagination?

```graphql
type User {
  posts(first: Int = 10): [Post!]!
}
```

DataLoader batches by key only, so you cannot easily batch `posts(first: 5)` and `posts(first: 10)` together. Two approaches:

**Approach A: Fetch all, slice in resolver**

```javascript
// Fetch all posts per author, let resolver handle limits
const postsByAuthorIdLoader = new DataLoader(async (authorIds) => {
  const posts = await db.query(
    'SELECT * FROM posts WHERE author_id = ANY($1) ORDER BY created_at DESC',
    [authorIds]
  );
  const grouped = groupBy(posts, 'author_id');
  return authorIds.map(id => grouped.get(id) || []);
});

// Resolver slices the result
const resolvers = {
  User: {
    posts: (parent, { first = 10 }, ctx) => {
      const allPosts = ctx.loaders.postsByAuthorId.load(parent.id);
      return allPosts.then(posts => posts.slice(0, first));
    },
  },
};
```

**Approach B: Composite key with custom cache key**

```javascript
const postsByAuthorIdLoader = new DataLoader(
  async (keys) => {
    // keys = [{ authorId: "1", first: 5 }, { authorId: "2", first: 10 }]
    // This is harder to batch efficiently into a single SQL query
    // Often Approach A is simpler and sufficient
    return Promise.all(keys.map(({ authorId, first }) =>
      db.query(
        'SELECT * FROM posts WHERE author_id = $1 ORDER BY created_at DESC LIMIT $2',
        [authorId, first]
      )
    ));
  },
  {
    cacheKeyFn: (key) => `${key.authorId}:${key.first}`,
  }
);
```

Approach A is usually preferred because it still batches into a single query.

## 7. Caching Behavior

DataLoader provides two levels of caching, and understanding the difference is critical.

### 7.1 Request-level Cache (DataLoader's Built-in)

DataLoader caches results by key for the lifetime of the DataLoader instance. Since we create instances per request, this is a per-request cache.

```javascript
// Within the same request:
const user1 = await userLoader.load('user_1');  // DB query
const user2 = await userLoader.load('user_2');  // DB query (batched with above)
const user3 = await userLoader.load('user_1');  // Cache hit! No DB query.
```

This cache solves the deduplication problem (same author referenced by multiple posts).

### 7.2 Disabling Caching

Sometimes you want batching without caching:

```javascript
const freshUserLoader = new DataLoader(batchUsers, {
  cache: false,  // Disable the memoization cache
});
```

### 7.3 Manual Cache Operations

```javascript
// Prime the cache (useful when you already have the data)
userLoader.prime('user_1', { id: 'user_1', name: 'Alice' });

// Clear a specific key (after a mutation, for example)
userLoader.clear('user_1');

// Clear the entire cache
userLoader.clearAll();
```

### 7.4 Application-level Cache (External)

DataLoader's cache is per-request only. For cross-request caching, use an external cache:

```javascript
const userByIdLoader = new DataLoader(async (ids) => {
  // Check external cache first (Redis, Memcached)
  const cached = await redis.mget(ids.map(id => `user:${id}`));
  const missingIds = ids.filter((_, i) => !cached[i]);

  // Fetch only missing entries from DB
  let freshUsers = [];
  if (missingIds.length > 0) {
    freshUsers = await db.query(
      'SELECT * FROM users WHERE id = ANY($1)',
      [missingIds]
    );

    // Populate external cache
    const pipeline = redis.pipeline();
    for (const user of freshUsers) {
      pipeline.set(`user:${user.id}`, JSON.stringify(user), 'EX', 300);
    }
    await pipeline.exec();
  }

  // Merge cached and fresh results
  const userMap = new Map(freshUsers.map(u => [u.id, u]));
  return ids.map((id, i) => {
    if (cached[i]) return JSON.parse(cached[i]);
    return userMap.get(id) || null;
  });
});
```

## 8. Per-Request DataLoader Instances

This is the most important DataLoader rule: **create new DataLoader instances for each request**.

### 8.1 Why Per-Request?

```javascript
// ❌ WRONG: Shared across requests
const globalLoader = new DataLoader(batchUsers);

// Request 1: loads user_1 → cached
// Request 2: loads user_1 → gets stale data from Request 1's cache!
// Request 3: user_1 was updated between requests → still stale!
```

Shared DataLoader instances cause:
- **Stale data**: Mutations in one request are not reflected in another
- **Memory leaks**: The cache grows indefinitely
- **Authorization leaks**: User A sees data cached from User B's request

### 8.2 Correct Pattern

```javascript
// dataloaders.js
import DataLoader from 'dataloader';

export function createDataLoaders(db) {
  // Called once per request — fresh instances, empty caches
  return {
    userById: new DataLoader(async (ids) => {
      const users = await db.user.findMany({ where: { id: { in: [...ids] } } });
      const map = new Map(users.map(u => [u.id, u]));
      return ids.map(id => map.get(id) || null);
    }),
    postsByAuthorId: new DataLoader(async (authorIds) => {
      const posts = await db.post.findMany({
        where: { authorId: { in: [...authorIds] } },
      });
      const grouped = new Map();
      for (const p of posts) {
        if (!grouped.has(p.authorId)) grouped.set(p.authorId, []);
        grouped.get(p.authorId).push(p);
      }
      return authorIds.map(id => grouped.get(id) || []);
    }),
  };
}

// server.js
const server = new ApolloServer({ typeDefs, resolvers });

const { url } = await startStandaloneServer(server, {
  context: async ({ req }) => ({
    db: prisma,
    user: await authenticate(req),
    loaders: createDataLoaders(prisma),  // Fresh per request
  }),
});
```

## 9. DataLoader with Different Data Sources

DataLoader is not limited to SQL databases. It works with any data source that supports batch fetching.

### 9.1 REST API

```javascript
const userByIdFromAPI = new DataLoader(async (ids) => {
  // Batch: single request with multiple IDs
  const response = await fetch(
    `https://api.example.com/users?ids=${ids.join(',')}`
  );
  const users = await response.json();
  const map = new Map(users.map(u => [u.id, u]));
  return ids.map(id => map.get(id) || null);
});
```

### 9.2 Redis

```javascript
const cachedDataLoader = new DataLoader(async (keys) => {
  // Redis MGET: batch fetch multiple keys in one call
  const values = await redis.mget(keys.map(k => `cache:${k}`));
  return values.map(v => v ? JSON.parse(v) : null);
});
```

### 9.3 MongoDB

```javascript
const userByIdFromMongo = new DataLoader(async (ids) => {
  const users = await db.collection('users').find({
    _id: { $in: ids.map(id => new ObjectId(id)) },
  }).toArray();
  const map = new Map(users.map(u => [u._id.toString(), u]));
  return ids.map(id => map.get(id) || null);
});
```

### 9.4 Elasticsearch

```javascript
const searchResultsLoader = new DataLoader(async (queries) => {
  // Multi-search: batch multiple queries into one request
  const body = queries.flatMap(q => [
    { index: 'posts' },
    { query: { match: { title: q } } },
  ]);
  const { responses } = await client.msearch({ body });
  return responses.map(r => r.hits.hits.map(h => h._source));
});
```

## 10. Monitoring and Detecting N+1

### 10.1 Query Logging

The simplest detection method is counting queries per GraphQL request:

```javascript
// Middleware that counts database queries per request
function createQueryCounter() {
  let count = 0;
  return {
    increment() { count++; },
    getCount() { return count; },
    reset() { count = 0; },
  };
}

// In context creation
context: async ({ req }) => {
  const queryCounter = createQueryCounter();

  // Wrap the DB client to count queries
  const countedDb = new Proxy(prisma, {
    get(target, prop) {
      return new Proxy(target[prop], {
        get(model, method) {
          if (typeof model[method] === 'function') {
            return (...args) => {
              queryCounter.increment();
              return model[method](...args);
            };
          }
          return model[method];
        },
      });
    },
  });

  return {
    db: countedDb,
    queryCounter,
    loaders: createDataLoaders(countedDb),
  };
},

// Plugin to log query count after each request
const queryCountPlugin = {
  async requestDidStart() {
    return {
      async willSendResponse({ contextValue }) {
        const count = contextValue.queryCounter.getCount();
        if (count > 20) {
          console.warn(`[N+1 WARNING] ${count} DB queries in single request`);
        }
      },
    };
  },
};
```

### 10.2 Apollo Studio Tracing

Apollo Server can report per-field timing data:

```javascript
const server = new ApolloServer({
  typeDefs,
  resolvers,
  plugins: [
    ApolloServerPluginUsageReporting({
      sendErrors: { unmodified: true },
    }),
  ],
});
```

In Apollo Studio, look for:
- Fields with high execution count (e.g., `Post.author` called 100 times)
- Fields with individually low latency but high total time (many sequential queries)
- Resolver execution count that exceeds the item count of the parent list

### 10.3 Before/After Comparison

```
                          Without DataLoader    With DataLoader
────────────────────────  ────────────────────  ──────────────────
posts(first: 50)          1 query               1 query
  Post.author (×50)       50 queries            1 query (batch)
  Post.comments (×50)     50 queries            1 query (batch)
    Comment.author (×200) 200 queries           1 query (batch)*
────────────────────────  ────────────────────  ──────────────────
Total                     301 queries           4 queries

* Deduplicated by DataLoader cache (200 loads → ~40 unique authors)
```

## 11. Common Pitfalls

### 11.1 Forgetting to Create Per-Request Instances

```javascript
// ❌ DataLoader created once at startup
const userLoader = new DataLoader(batchUsers);
// Stale cache, memory leaks, authorization issues

// ✅ DataLoader created in context (per request)
context: async () => ({
  loaders: { userById: new DataLoader(batchUsers) },
}),
```

### 11.2 Wrong Return Order

```javascript
// ❌ Returns users in database order (not input order)
const batchUsers = async (ids) => {
  return db.query('SELECT * FROM users WHERE id = ANY($1)', [ids]);
  // DB returns [user_3, user_1, user_2] but keys were [1, 2, 3]
};

// ✅ Maps results back to input key order
const batchUsers = async (ids) => {
  const users = await db.query('SELECT * FROM users WHERE id = ANY($1)', [ids]);
  const map = new Map(users.map(u => [u.id, u]));
  return ids.map(id => map.get(id) || null);  // Preserves input order
};
```

### 11.3 Wrong Return Count

```javascript
// ❌ Filters out nulls (array length !== key length)
const batchUsers = async (ids) => {
  const users = await db.query('SELECT * FROM users WHERE id = ANY($1)', [ids]);
  return users;  // If 2 of 5 IDs are missing, returns 3 instead of 5
};

// ✅ Returns one result per key (null for missing)
const batchUsers = async (ids) => {
  const users = await db.query('SELECT * FROM users WHERE id = ANY($1)', [ids]);
  const map = new Map(users.map(u => [u.id, u]));
  return ids.map(id => map.get(id) || null);  // Always 5 results for 5 keys
};
```

### 11.4 Mutating After Load

After a mutation, the DataLoader cache may be stale:

```javascript
const resolvers = {
  Mutation: {
    updateUser: async (_, { id, input }, ctx) => {
      const user = await ctx.db.user.update({ where: { id }, data: input });

      // Clear the cached value so subsequent loads get fresh data
      ctx.loaders.userById.clear(id);
      // Optionally prime with the updated value
      ctx.loaders.userById.prime(id, user);

      return { user, errors: [] };
    },
  },
};
```

### 11.5 Using DataLoader for Non-batchable Sources

If your data source does not support batch fetching (e.g., a REST API that only accepts one ID at a time), DataLoader still helps with deduplication but the performance gain from batching is lost:

```javascript
// This "batch" function still makes N requests
const userLoader = new DataLoader(async (ids) => {
  return Promise.all(ids.map(id =>
    fetch(`https://api.example.com/users/${id}`).then(r => r.json())
  ));
});
// Benefit: deduplication (same ID requested twice → one fetch)
// No benefit: batching (still N fetches, just deduplicated)
```

---

## 12. Practice Problems

### Exercise 1: Count the Queries (Beginner)

Given these resolvers (no DataLoader):

```javascript
const resolvers = {
  Query: {
    users: (_, { first }) => db.query('SELECT * FROM users LIMIT $1', [first]),
  },
  User: {
    posts: (parent) => db.query('SELECT * FROM posts WHERE author_id = $1', [parent.id]),
    followerCount: (parent) => db.query('SELECT COUNT(*) FROM follows WHERE followed_id = $1', [parent.id]),
  },
  Post: {
    commentCount: (parent) => db.query('SELECT COUNT(*) FROM comments WHERE post_id = $1', [parent.id]),
  },
};
```

How many database queries are executed for this request?

```graphql
query {
  users(first: 20) {
    posts {
      commentCount
    }
    followerCount
  }
}
```

Assume each user has exactly 3 posts.

### Exercise 2: Implement a Batch Function (Intermediate)

Write a batch function for loading comments by post ID (one-to-many). Given:

- Database table: `comments (id, post_id, body, author_id, created_at)`
- A function `db.query(sql, params)` that returns rows
- Input: array of post IDs `["post_1", "post_2", "post_3"]`
- Output: array of comment arrays, one per post ID, in order

Handle the case where a post has no comments (return empty array, not null).

### Exercise 3: Fix the DataLoader Bug (Intermediate)

This code has a bug. Find and fix it.

```javascript
const tagLoader = new DataLoader(async (tagNames) => {
  const tags = await db.query(
    'SELECT * FROM tags WHERE name = ANY($1)',
    [tagNames]
  );
  return tags;
});
```

Hint: What happens when `tagNames = ['graphql', 'rest', 'api']` and the database only has `graphql` and `api` tags?

### Exercise 4: Full DataLoader Integration (Advanced)

Given this schema, implement the complete DataLoader factory and resolver map:

```graphql
type Query {
  courses(first: Int = 10): [Course!]!
}

type Course {
  id: ID!
  title: String!
  instructor: User!
  students: [User!]!
  lessonCount: Int!
}

type User {
  id: ID!
  name: String!
  enrolledCourses: [Course!]!
}
```

Database tables:
- `courses (id, title, instructor_id)`
- `users (id, name)`
- `enrollments (user_id, course_id)`
- `lessons (id, course_id, title)`

Requirements:
1. Create DataLoader instances for all relationships
2. Use aggregation queries for `lessonCount` (do not load all lessons)
3. Handle the many-to-many relationship (enrollments) correctly
4. Create per-request DataLoader instances

### Exercise 5: Monitoring Implementation (Advanced)

Build a simple N+1 detection system that:

1. Wraps a database client to count queries per request
2. Logs a warning if a single GraphQL request exceeds 10 database queries
3. Reports the top 5 most-called resolvers by field name
4. Integrates with Apollo Server as a plugin

Write the plugin code and the monitoring middleware.

---

## 13. References

- DataLoader GitHub Repository - https://github.com/graphql/dataloader
- DataLoader Source Code (reference implementation) - https://github.com/graphql/dataloader/blob/main/src/index.js
- Lee Byron, "DataLoader — Source Code Walkthrough" (YouTube, 2016)
- Apollo Server DataLoader Guide - https://www.apollographql.com/docs/apollo-server/data/fetching-data/#batching-and-caching
- Marc-Andre Giroux, "Production Ready GraphQL" (2020) - Chapter 5: Performance
- Node.js Event Loop - https://nodejs.org/en/docs/guides/event-loop-timers-and-nexttick

---

**Previous**: [Resolvers](./04_Resolvers.md) | **Next**: [Subscriptions](./06_Subscriptions.md)
