# 04. Resolvers

**Previous**: [Queries and Mutations](./03_Queries_and_Mutations.md) | **Next**: [DataLoader and N+1](./05_DataLoader_N_plus_1.md)

---

If the schema is the contract, resolvers are the implementation. Every field in a GraphQL schema is backed by a resolver function that knows how to fetch or compute that field's value. Understanding how resolvers work --- their signature, execution order, and relationship to each other --- is essential for building correct, performant GraphQL servers. This lesson dissects the resolver mechanism from the ground up.

**Difficulty**: ⭐⭐⭐

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain the four arguments of a resolver function (parent, args, context, info)
2. Describe how the resolver chain works for nested fields and identify the execution order
3. Use the context object to share per-request resources (database connections, auth tokens)
4. Apply resolver design patterns: thin resolvers, service layer delegation, and field-level resolvers
5. Implement structured error handling within resolvers

---

## Table of Contents

1. [What Is a Resolver?](#1-what-is-a-resolver)
2. [The Resolver Signature](#2-the-resolver-signature)
3. [Default Resolvers](#3-default-resolvers)
4. [The Resolver Chain](#4-the-resolver-chain)
5. [The Context Object](#5-the-context-object)
6. [The Info Object](#6-the-info-object)
7. [Async Resolvers](#7-async-resolvers)
8. [Resolver Design Patterns](#8-resolver-design-patterns)
9. [Error Handling in Resolvers](#9-error-handling-in-resolvers)
10. [Practice Problems](#10-practice-problems)
11. [References](#11-references)

---

## 1. What Is a Resolver?

A resolver is a function that populates the data for a single field in your schema. When a GraphQL query is executed, the server calls a resolver for each field in the selection set. The result of all these resolver calls is assembled into the response JSON.

```
Schema field                Resolver function
───────────                 ─────────────────
Query.user(id: ID!)    →    (parent, args, ctx, info) => db.users.findById(args.id)
User.name              →    (parent) => parent.name
User.posts             →    (parent, args, ctx) => db.posts.findByAuthorId(parent.id)
Post.title             →    (parent) => parent.title
Post.author            →    (parent, args, ctx) => db.users.findById(parent.authorId)
```

The resolver tree mirrors the query tree. For a query like:

```graphql
query {
  user(id: "1") {
    name
    posts {
      title
      author {
        name
      }
    }
  }
}
```

The execution order is:

```
1. Query.user(id: "1")           → returns User object
2.   User.name                   → returns "Alice"
3.   User.posts                  → returns [Post, Post, ...]
4.     Post.title (for each)     → returns "GraphQL 101"
5.     Post.author (for each)    → returns User object
6.       User.name (for each)    → returns "Alice"
```

## 2. The Resolver Signature

Every resolver receives four arguments:

```javascript
function resolver(parent, args, context, info) {
  // ...
}
```

### 2.1 parent (also called root, obj, or source)

The return value of the parent field's resolver. For root-level resolvers (Query, Mutation), this is the `rootValue` passed to the server (often `undefined`).

```javascript
const resolvers = {
  Query: {
    // parent is rootValue (usually undefined)
    user: (parent, args, context) => {
      return context.db.users.findById(args.id);
      // Returns: { id: '1', name: 'Alice', email: 'alice@example.com' }
    },
  },
  User: {
    // parent is the User object returned by Query.user
    name: (parent) => {
      // parent = { id: '1', name: 'Alice', email: 'alice@example.com' }
      return parent.name; // "Alice"
    },
    posts: (parent, args, context) => {
      // parent.id gives us the user's ID to find their posts
      return context.db.posts.findByAuthorId(parent.id);
    },
  },
};
```

### 2.2 args

An object containing the arguments passed to the field:

```graphql
type Query {
  posts(first: Int = 10, offset: Int = 0, status: PostStatus): [Post!]!
}
```

```javascript
const resolvers = {
  Query: {
    posts: (parent, args, context) => {
      // args = { first: 10, offset: 0, status: 'PUBLISHED' }
      const { first, offset, status } = args;
      return context.db.posts.findAll({
        where: status ? { status } : {},
        limit: first,
        offset,
      });
    },
  },
};
```

### 2.3 context

A shared object available to all resolvers in a single request. It is created fresh for each request and typically contains:

- Database connections or ORM instances
- Authentication/authorization information
- DataLoader instances (Lesson 05)
- Request-specific utilities (logging, tracing)

```javascript
// Server setup: create context per request
const server = new ApolloServer({ typeDefs, resolvers });

const { url } = await startStandaloneServer(server, {
  context: async ({ req }) => ({
    // Fresh context for every request
    db: database,
    user: await authenticateUser(req.headers.authorization),
    loaders: createDataLoaders(database),
  }),
});
```

### 2.4 info

The `GraphQLResolveInfo` object containing metadata about the current execution state. This is the most complex argument and is rarely needed for basic resolvers. It contains:

- `fieldName`: the name of the current field
- `returnType`: the GraphQL type of the field
- `parentType`: the parent type
- `path`: the path from root to this field
- `fieldNodes`: the AST nodes for this field's selection set
- `schema`: the full GraphQL schema object

We cover `info` in more detail in Section 6.

## 3. Default Resolvers

GraphQL has a built-in default resolver that handles the most common case: extracting a property from the parent object.

```javascript
// This is essentially what the default resolver does:
function defaultResolver(parent, args, context, info) {
  return parent[info.fieldName];
}
```

This means you do not need to write resolvers for simple property access:

```javascript
// Schema
// type User {
//   id: ID!
//   name: String!
//   email: String!
// }

const resolvers = {
  Query: {
    user: (_, { id }, ctx) => ctx.db.users.findById(id),
    // Returns: { id: '1', name: 'Alice', email: 'alice@example.com' }
  },
  // No User resolvers needed!
  // Default resolver handles User.id, User.name, User.email
  // because the database object has matching property names
};
```

**When you DO need explicit resolvers:**

```javascript
const resolvers = {
  User: {
    // 1. Computed fields (not a direct property of the data object)
    fullName: (parent) => `${parent.firstName} ${parent.lastName}`,

    // 2. Property name mismatch (DB column differs from schema field)
    avatarUrl: (parent) => parent.avatar_url,

    // 3. Relationships (require additional data fetching)
    posts: (parent, args, ctx) => ctx.db.posts.findByAuthorId(parent.id),

    // 4. Authorization checks
    email: (parent, args, ctx) => {
      if (ctx.user?.id === parent.id || ctx.user?.role === 'ADMIN') {
        return parent.email;
      }
      return null; // Hide email from other users
    },
  },
};
```

## 4. The Resolver Chain

The resolver chain is how GraphQL resolves nested fields. Each resolver returns data, and that data becomes the `parent` argument for the next level of resolvers.

### 4.1 Step-by-step Execution

Consider this query:

```graphql
query {
  user(id: "1") {
    name
    posts(first: 2) {
      title
      comments(first: 3) {
        body
        author {
          name
        }
      }
    }
  }
}
```

Execution trace:

```
Step 1: Query.user({ id: "1" })
  → Returns: { id: "1", name: "Alice", ... }

Step 2: User.name(parent={ id: "1", name: "Alice", ... })
  → Returns: "Alice" (default resolver: parent.name)

Step 3: User.posts(parent={ id: "1", ... }, args={ first: 2 })
  → Returns: [
      { id: "10", title: "Post A", authorId: "1" },
      { id: "11", title: "Post B", authorId: "1" },
    ]

Step 4a: Post.title(parent={ id: "10", title: "Post A", ... })
  → Returns: "Post A"

Step 4b: Post.title(parent={ id: "11", title: "Post B", ... })
  → Returns: "Post B"

Step 5a: Post.comments(parent={ id: "10", ... }, args={ first: 3 })
  → Returns: [
      { id: "100", body: "Great!", authorId: "2" },
      { id: "101", body: "Thanks!", authorId: "3" },
    ]

Step 5b: Post.comments(parent={ id: "11", ... }, args={ first: 3 })
  → Returns: [{ id: "102", body: "Nice post", authorId: "2" }]

Step 6: Comment.author(parent={ id: "100", ..., authorId: "2" })
  → Returns: { id: "2", name: "Bob", ... }

... and so on for each comment
```

### 4.2 The Tree Structure

```
Query.user ─────────────────────────────────┐
  ├── User.name          → "Alice"          │
  └── User.posts         → [Post, Post]     │
        ├── Post.title   → "Post A"         │
        ├── Post.comments → [Comment, ...]  │
        │     ├── Comment.body → "Great!"   │
        │     └── Comment.author ──────┐    │
        │           └── User.name      │    │
        ├── Post.title   → "Post B"    │    │
        └── Post.comments → [Comment]  │    │
              └── ...                   │    │
                                       │    │
Each arrow calls a resolver function.  │    │
Each resolver receives the parent's    │    │
return value.                          │    │
```

### 4.3 Parallel Execution

Sibling fields at the same level may execute in parallel:

```javascript
// User.name and User.posts can execute simultaneously
// because they are siblings in the selection set
const resolvers = {
  User: {
    name: async (parent) => parent.name,     // These two can run
    posts: async (parent, _, ctx) => {       // at the same time
      return ctx.db.posts.findByAuthorId(parent.id);
    },
  },
};
```

The GraphQL execution engine uses `Promise.all` for sibling async resolvers. Child resolvers wait for their parent to complete.

## 5. The Context Object

The context is the dependency injection mechanism for GraphQL resolvers. It carries per-request state that every resolver can access.

### 5.1 Building Context

```javascript
import { ApolloServer } from '@apollo/server';
import { startStandaloneServer } from '@apollo/server/standalone';
import { PrismaClient } from '@prisma/client';
import { createDataLoaders } from './dataloaders.js';
import { verifyToken } from './auth.js';

const prisma = new PrismaClient();

const server = new ApolloServer({ typeDefs, resolvers });

const { url } = await startStandaloneServer(server, {
  context: async ({ req }) => {
    // 1. Authenticate the user from the request header
    const token = req.headers.authorization?.replace('Bearer ', '');
    let user = null;
    if (token) {
      try {
        user = await verifyToken(token);
      } catch (err) {
        // Token is invalid; user remains null (unauthenticated)
      }
    }

    // 2. Create per-request DataLoaders (Lesson 05)
    const loaders = createDataLoaders(prisma);

    // 3. Return the context object
    return {
      db: prisma,
      user,           // Authenticated user (or null)
      loaders,        // DataLoader instances
      requestId: crypto.randomUUID(),  // For tracing/logging
    };
  },
});
```

### 5.2 Using Context in Resolvers

```javascript
const resolvers = {
  Query: {
    me: (_, __, context) => {
      // Require authentication
      if (!context.user) {
        throw new GraphQLError('Not authenticated', {
          extensions: { code: 'UNAUTHENTICATED' },
        });
      }
      return context.db.user.findUnique({ where: { id: context.user.id } });
    },
    posts: (_, args, context) => {
      return context.db.post.findMany({
        take: args.first,
        skip: args.offset,
        orderBy: { createdAt: 'desc' },
      });
    },
  },
  User: {
    posts: (parent, _, context) => {
      // Use DataLoader from context (prevents N+1)
      return context.loaders.postsByAuthorId.load(parent.id);
    },
    email: (parent, _, context) => {
      // Authorization: only show email to the user themselves or admins
      if (context.user?.id === parent.id || context.user?.role === 'ADMIN') {
        return parent.email;
      }
      return null;
    },
  },
  Post: {
    author: (parent, _, context) => {
      return context.loaders.userById.load(parent.authorId);
    },
  },
};
```

### 5.3 Context Design Principles

1. **Create fresh per request**: Never share mutable state across requests
2. **Keep it flat**: Avoid deeply nested context structures
3. **Include only what resolvers need**: Database, auth, loaders, logger
4. **Type it**: In TypeScript, define a `Context` interface

```typescript
// types.ts
interface Context {
  db: PrismaClient;
  user: AuthenticatedUser | null;
  loaders: DataLoaders;
  requestId: string;
}
```

## 6. The Info Object

The `info` parameter (`GraphQLResolveInfo`) is the least commonly used but most powerful resolver argument. It provides the full execution context.

### 6.1 Key Properties

```javascript
const resolvers = {
  Query: {
    user: (parent, args, context, info) => {
      console.log(info.fieldName);     // "user"
      console.log(info.returnType);    // User (GraphQL type object)
      console.log(info.parentType);    // Query (GraphQL type object)
      console.log(info.path);          // { key: 'user', prev: undefined }

      // info.fieldNodes contains the AST of the selection set
      // This tells you exactly which fields the client requested
      return context.db.users.findById(args.id);
    },
  },
};
```

### 6.2 Practical Use: Field-level Optimization

The `info` object lets you inspect what fields were requested, enabling query optimization:

```javascript
import { parseResolveInfo, simplifyParsedResolveInfoFragmentType } from 'graphql-parse-resolve-info';

const resolvers = {
  Query: {
    user: (parent, args, context, info) => {
      // Parse the selection set to determine which fields are requested
      const parsedInfo = parseResolveInfo(info);
      const fields = simplifyParsedResolveInfoFragmentType(parsedInfo, info.returnType);

      // Only join 'posts' table if 'posts' field is requested
      const includePosts = 'posts' in (fields.fields || {});

      return context.db.user.findUnique({
        where: { id: args.id },
        include: includePosts ? { posts: true } : {},
      });
    },
  },
};
```

### 6.3 When to Use Info

| Use Case | Example |
|----------|---------|
| SQL SELECT optimization | Only SELECT columns that were queried |
| JOIN optimization | Only JOIN related tables if relationships are queried |
| DataLoader decisions | Skip loading if related field was not requested |
| Query complexity analysis | Count depth/breadth from the AST |
| Caching key generation | Use the selection set as part of the cache key |

For most applications, you will not need `info`. It becomes relevant when optimizing database queries or building framework-level tools.

## 7. Async Resolvers

In practice, almost all resolvers interact with databases, external APIs, or other async operations. GraphQL natively supports async resolvers.

### 7.1 Returning Promises

```javascript
const resolvers = {
  Query: {
    // Resolver returns a Promise — GraphQL awaits it automatically
    user: (_, { id }, ctx) => {
      return ctx.db.user.findUnique({ where: { id } });
    },

    // async/await syntax (equivalent)
    posts: async (_, args, ctx) => {
      const posts = await ctx.db.post.findMany({
        take: args.first,
        orderBy: { createdAt: 'desc' },
      });
      return posts;
    },

    // Multiple async operations
    dashboard: async (_, __, ctx) => {
      const [userCount, postCount, commentCount] = await Promise.all([
        ctx.db.user.count(),
        ctx.db.post.count(),
        ctx.db.comment.count(),
      ]);
      return { userCount, postCount, commentCount };
    },
  },
};
```

### 7.2 Resolver Return Types

A resolver can return:

| Return Value | Behavior |
|-------------|----------|
| Scalar value | Used directly |
| Object | Passed as `parent` to child resolvers |
| Array | Each element is resolved individually |
| Promise | Awaited, then processed as above |
| `null` / `undefined` | Field is null (error if non-null type) |
| Error (thrown) | Field is null, error added to errors array |

## 8. Resolver Design Patterns

### 8.1 Thin Resolvers

Keep resolvers thin. They should delegate to a service/data layer, not contain business logic:

```javascript
// ❌ Fat resolver: business logic mixed with data access
const resolvers = {
  Mutation: {
    createPost: async (_, { input }, ctx) => {
      // Validation
      if (input.title.length < 3) throw new Error('Title too short');
      if (input.title.length > 200) throw new Error('Title too long');
      if (!input.body.trim()) throw new Error('Body is required');

      // Business logic
      const slug = input.title.toLowerCase().replace(/\s+/g, '-');
      const existingSlug = await ctx.db.post.findFirst({ where: { slug } });
      const finalSlug = existingSlug ? `${slug}-${Date.now()}` : slug;

      // Data access
      const post = await ctx.db.post.create({
        data: {
          ...input,
          slug: finalSlug,
          authorId: ctx.user.id,
          publishedAt: input.publishNow ? new Date() : null,
        },
      });

      // Side effects
      await ctx.notificationService.notifyFollowers(ctx.user.id, post.id);
      await ctx.searchIndex.indexPost(post);

      return { post, errors: [] };
    },
  },
};

// ✅ Thin resolver: delegates to service layer
const resolvers = {
  Mutation: {
    createPost: async (_, { input }, ctx) => {
      return ctx.services.posts.create(input, ctx.user);
    },
  },
};
```

The service layer (`PostService`) handles validation, slug generation, database access, and notifications. This separation makes the service independently testable and reusable.

### 8.2 Field-level Resolvers for Computed Data

```javascript
const resolvers = {
  Post: {
    // Computed field: reading time
    readingTime: (parent) => {
      const wordsPerMinute = 200;
      const wordCount = parent.body.split(/\s+/).length;
      return Math.ceil(wordCount / wordsPerMinute);
    },

    // Computed field: excerpt (first 200 characters)
    excerpt: (parent, { length = 200 }) => {
      if (parent.body.length <= length) return parent.body;
      return parent.body.substring(0, length).trimEnd() + '...';
    },

    // Conditional field: requires authorization check
    viewCount: (parent, _, ctx) => {
      // Only authors and admins can see view counts
      if (ctx.user?.id === parent.authorId || ctx.user?.role === 'ADMIN') {
        return parent.viewCount;
      }
      return null;
    },
  },
};
```

### 8.3 Resolver Map Structure

```javascript
// Complete resolver map example
const resolvers = {
  // Root types
  Query: {
    me: (_, __, ctx) => ctx.user ? ctx.services.users.findById(ctx.user.id) : null,
    user: (_, { id }, ctx) => ctx.services.users.findById(id),
    posts: (_, args, ctx) => ctx.services.posts.findMany(args),
    search: (_, { query }, ctx) => ctx.services.search.execute(query),
  },

  Mutation: {
    createUser: (_, { input }, ctx) => ctx.services.users.create(input),
    updateUser: (_, { id, input }, ctx) => ctx.services.users.update(id, input, ctx.user),
    createPost: (_, { input }, ctx) => ctx.services.posts.create(input, ctx.user),
    deletePost: (_, { id }, ctx) => ctx.services.posts.delete(id, ctx.user),
  },

  // Type resolvers
  User: {
    posts: (parent, _, ctx) => ctx.loaders.postsByAuthorId.load(parent.id),
    followerCount: (parent, _, ctx) => ctx.loaders.followerCount.load(parent.id),
  },

  Post: {
    author: (parent, _, ctx) => ctx.loaders.userById.load(parent.authorId),
    comments: (parent, _, ctx) => ctx.loaders.commentsByPostId.load(parent.id),
    readingTime: (parent) => Math.ceil(parent.body.split(/\s+/).length / 200),
  },

  Comment: {
    author: (parent, _, ctx) => ctx.loaders.userById.load(parent.authorId),
  },

  // Abstract type resolvers
  SearchResult: {
    __resolveType(obj) {
      if (obj.username) return 'User';
      if (obj.title) return 'Post';
      if (obj.body && !obj.title) return 'Comment';
      return null;
    },
  },
};
```

## 9. Error Handling in Resolvers

### 9.1 Throwing Errors

When a resolver throws, GraphQL catches the error, sets the field to `null`, and adds the error to the response's `errors` array:

```javascript
import { GraphQLError } from 'graphql';

const resolvers = {
  Query: {
    user: async (_, { id }, ctx) => {
      const user = await ctx.db.user.findUnique({ where: { id } });
      if (!user) {
        throw new GraphQLError(`User with ID ${id} not found`, {
          extensions: {
            code: 'NOT_FOUND',
            argumentName: 'id',
          },
        });
      }
      return user;
    },
  },
  Mutation: {
    deletePost: async (_, { id }, ctx) => {
      if (!ctx.user) {
        throw new GraphQLError('You must be logged in', {
          extensions: { code: 'UNAUTHENTICATED' },
        });
      }
      const post = await ctx.db.post.findUnique({ where: { id } });
      if (!post) {
        throw new GraphQLError('Post not found', {
          extensions: { code: 'NOT_FOUND' },
        });
      }
      if (post.authorId !== ctx.user.id && ctx.user.role !== 'ADMIN') {
        throw new GraphQLError('Not authorized to delete this post', {
          extensions: { code: 'FORBIDDEN' },
        });
      }
      await ctx.db.post.delete({ where: { id } });
      return { success: true };
    },
  },
};
```

### 9.2 Error Propagation and Nullability

Remember from Lesson 02: non-null fields propagate null upward. This affects error handling:

```graphql
type Query {
  user(id: ID!): User       # Nullable → error stays here
}

type User {
  name: String!              # Non-null → if this errors, User becomes null
  riskyField: String!        # Non-null → if this errors, User becomes null
  safeField: String          # Nullable → error stays here
}
```

```json
// If riskyField resolver throws:
{
  "data": {
    "user": null          // Entire user is nullified because riskyField is String!
  },
  "errors": [{
    "message": "Something went wrong",
    "path": ["user", "riskyField"]
  }]
}
```

### 9.3 Structured Error Handling Pattern

For mutations, return errors as data rather than thrown exceptions:

```javascript
const resolvers = {
  Mutation: {
    createUser: async (_, { input }, ctx) => {
      // Validate
      const errors = [];

      if (input.username.length < 3) {
        errors.push({ field: 'username', message: 'Must be at least 3 characters' });
      }
      const existing = await ctx.db.user.findUnique({
        where: { email: input.email },
      });
      if (existing) {
        errors.push({ field: 'email', message: 'Email already in use' });
      }

      if (errors.length > 0) {
        return { user: null, errors };
      }

      // Create
      const user = await ctx.db.user.create({ data: input });
      return { user, errors: [] };
    },
  },
};
```

This approach:
- Returns a 200 HTTP status (errors are data, not transport failures)
- Allows multiple field-level errors in a single response
- Keeps the `errors` array at the top level for actual execution errors

---

## 10. Practice Problems

### Exercise 1: Resolver Tracing (Beginner)

Given this schema and query, list every resolver call in execution order. Indicate which argument each resolver receives as `parent`.

```graphql
type Query {
  book(id: ID!): Book
}

type Book {
  id: ID!
  title: String!
  author: Author!
  reviews: [Review!]!
}

type Author {
  id: ID!
  name: String!
}

type Review {
  id: ID!
  rating: Int!
  reviewer: String!
}
```

```graphql
query {
  book(id: "1") {
    title
    author {
      name
    }
    reviews {
      rating
      reviewer
    }
  }
}
```

Assume the database has:
- Book: `{ id: "1", title: "Clean Code", authorId: "a1" }`
- Author: `{ id: "a1", name: "Robert Martin" }`
- Reviews: `[{ id: "r1", rating: 5, reviewer: "Alice" }, { id: "r2", rating: 4, reviewer: "Bob" }]`

### Exercise 2: Write Resolvers (Intermediate)

Given this schema, write the complete resolver map. Use `context.db` for data access.

```graphql
type Query {
  product(id: ID!): Product
  products(category: String, minPrice: Float, maxPrice: Float): [Product!]!
}

type Product {
  id: ID!
  name: String!
  price: Float!
  category: String!
  inStock: Boolean!
  reviews: [Review!]!
  averageRating: Float       # Computed from reviews
}

type Review {
  id: ID!
  rating: Int!
  comment: String
  author: User!
}

type User {
  id: ID!
  username: String!
}
```

Assume `context.db` has these methods:
- `db.products.findById(id)`
- `db.products.findAll({ category, minPrice, maxPrice })`
- `db.reviews.findByProductId(productId)`
- `db.users.findById(userId)`

### Exercise 3: Context Design (Intermediate)

Design the context factory function for a blogging platform that requires:

1. PostgreSQL database access (via Prisma)
2. JWT-based authentication
3. Redis cache for frequently accessed data
4. A request ID for distributed tracing
5. Rate limiting state (requests per user per minute)

Write the `context` function for Apollo Server's `startStandaloneServer`.

### Exercise 4: Error Handling (Intermediate)

Write a `updateProfile` mutation resolver that:

1. Requires authentication (throw UNAUTHENTICATED if no user)
2. Validates that `displayName` is between 2 and 50 characters
3. Validates that `bio` is at most 500 characters
4. Returns field-level errors for validation failures (not thrown errors)
5. Returns the updated user on success

Use the structured error handling pattern (payload with user + errors array).

### Exercise 5: Resolver Optimization (Advanced)

The following resolver has performance problems. Identify all issues and rewrite it:

```javascript
const resolvers = {
  Query: {
    posts: async (_, { first = 10 }, ctx) => {
      const posts = await ctx.db.post.findMany({ take: first });
      return posts;
    },
  },
  Post: {
    author: async (parent, _, ctx) => {
      const author = await ctx.db.user.findUnique({
        where: { id: parent.authorId },
      });
      return author;
    },
    comments: async (parent, _, ctx) => {
      const comments = await ctx.db.comment.findMany({
        where: { postId: parent.id },
      });
      return comments;
    },
    commentCount: async (parent, _, ctx) => {
      const comments = await ctx.db.comment.findMany({
        where: { postId: parent.id },
      });
      return comments.length;
    },
    likeCount: async (parent, _, ctx) => {
      const likes = await ctx.db.like.findMany({
        where: { postId: parent.id },
      });
      return likes.length;
    },
  },
  Comment: {
    author: async (parent, _, ctx) => {
      const author = await ctx.db.user.findUnique({
        where: { id: parent.authorId },
      });
      return author;
    },
  },
};
```

Hint: This resolver has N+1 problems, redundant queries, and inefficient counting. Lesson 05 covers DataLoader, but you can already identify the issues and suggest solutions.

---

## 11. References

- GraphQL Execution Specification - https://spec.graphql.org/October2021/#sec-Execution
- Apollo Server Resolvers - https://www.apollographql.com/docs/apollo-server/data/resolvers/
- graphql-parse-resolve-info - https://github.com/graphile/graphile-engine/tree/master/packages/graphql-parse-resolve-info
- Marc-Andre Giroux, "Production Ready GraphQL" (2020) - Chapter 4: Resolver Design
- GraphQL Error Handling - https://www.apollographql.com/docs/apollo-server/data/errors/

---

**Previous**: [Queries and Mutations](./03_Queries_and_Mutations.md) | **Next**: [DataLoader and N+1](./05_DataLoader_N_plus_1.md)
