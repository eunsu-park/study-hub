# 08. Apollo Server

**Previous**: [Authentication and Authorization](./07_Authentication_Authorization.md) | **Next**: [GraphQL Clients](./09_GraphQL_Clients.md)

---

Apollo Server is the most widely adopted GraphQL server for the JavaScript/TypeScript ecosystem. Version 4 — a ground-up rewrite — is framework-agnostic: it can run standalone, as Express middleware, inside Fastify, or even in a serverless function. This lesson walks through Apollo Server 4 from initial setup to production-ready configuration, covering both schema-first and code-first approaches, the plugin system, and Apollo Studio integration.

**Difficulty**: ⭐⭐⭐

## Learning Objectives

After completing this lesson, you will be able to:

1. Set up Apollo Server 4 in standalone mode and as Express/Fastify middleware
2. Compare schema-first and code-first development approaches and choose appropriately
3. Configure the context function to inject databases, auth, and DataLoaders
4. Use Apollo Server plugins for logging, caching, and error formatting
5. Prepare a production-ready Apollo Server with CSRF prevention, introspection control, and monitoring

---

## Table of Contents

1. [Apollo Server 4 Architecture](#1-apollo-server-4-architecture)
2. [Standalone Setup](#2-standalone-setup)
3. [Express Middleware Setup](#3-express-middleware-setup)
4. [Fastify Integration](#4-fastify-integration)
5. [Schema-First vs Code-First](#5-schema-first-vs-code-first)
6. [The Context Function](#6-the-context-function)
7. [Plugins](#7-plugins)
8. [Error Handling and Formatting](#8-error-handling-and-formatting)
9. [Apollo Studio Integration](#9-apollo-studio-integration)
10. [Production Configuration](#10-production-configuration)
11. [Practice Problems](#11-practice-problems)
12. [References](#12-references)

---

## 1. Apollo Server 4 Architecture

Apollo Server 4 separates the GraphQL execution engine from the HTTP layer. The core `ApolloServer` class handles schema validation, query parsing, and execution. The HTTP integration (Express, Fastify, standalone) handles request/response plumbing.

```
┌─────────────────────────────────────────────────┐
│                HTTP Framework                    │
│  (Express / Fastify / Standalone / Lambda)       │
├─────────────────────────────────────────────────┤
│           expressMiddleware / startStandalone     │
│           (HTTP integration layer)                │
├─────────────────────────────────────────────────┤
│              ApolloServer Core                    │
│  ┌───────────┬──────────┬──────────────────┐    │
│  │  Schema   │ Plugins  │  Request Pipeline │    │
│  │  Parsing  │          │  parse → validate │    │
│  │           │          │  → execute         │    │
│  └───────────┴──────────┴──────────────────┘    │
├─────────────────────────────────────────────────┤
│              Your Resolvers + DataSources         │
└─────────────────────────────────────────────────┘
```

Key changes from Apollo Server 3:

- No more `apollo-server-express`, `apollo-server-koa`, etc. — one package (`@apollo/server`) for all integrations
- Framework integration is explicit (you call `expressMiddleware()`, not `server.applyMiddleware()`)
- TypeScript-first API design
- Built-in `ApolloServerPluginDrainHttpServer` replaced by explicit drain patterns

---

## 2. Standalone Setup

The simplest way to run Apollo Server — no Express, no framework, just GraphQL.

```bash
npm install @apollo/server graphql
```

```typescript
// src/index.ts
import { ApolloServer } from '@apollo/server';
import { startStandaloneServer } from '@apollo/server/standalone';

// Type definitions (SDL)
const typeDefs = `#graphql
  type Book {
    id: ID!
    title: String!
    author: String!
    year: Int
  }

  type Query {
    books: [Book!]!
    book(id: ID!): Book
  }

  type Mutation {
    addBook(title: String!, author: String!, year: Int): Book!
  }
`;

// In-memory data store (replace with database in production)
let books = [
  { id: '1', title: 'The GraphQL Guide', author: 'John Resig', year: 2022 },
  { id: '2', title: 'Learning GraphQL', author: "Eve Porcello", year: 2018 },
];

const resolvers = {
  Query: {
    books: () => books,
    book: (_, { id }) => books.find((b) => b.id === id),
  },
  Mutation: {
    addBook: (_, { title, author, year }) => {
      const book = {
        id: String(books.length + 1),
        title,
        author,
        year,
      };
      books.push(book);
      return book;
    },
  },
};

const server = new ApolloServer({ typeDefs, resolvers });

const { url } = await startStandaloneServer(server, {
  listen: { port: 4000 },
  context: async ({ req }) => ({
    // Context is available in every resolver
    token: req.headers.authorization,
  }),
});

console.log(`Server ready at ${url}`);
```

`startStandaloneServer` creates an Express app internally. It is convenient for prototyping but limited — you cannot add custom middleware or routes.

---

## 3. Express Middleware Setup

For real applications, you typically need Express for additional routes (health checks, webhooks), middleware (CORS, body parsing, rate limiting), or WebSocket support (subscriptions).

```bash
npm install @apollo/server express graphql cors
npm install -D @types/cors @types/express
```

```typescript
// src/index.ts
import { ApolloServer } from '@apollo/server';
import { expressMiddleware } from '@apollo/server/express4';
import {
  ApolloServerPluginDrainHttpServer,
} from '@apollo/server/plugin/drainHttpServer';
import express from 'express';
import http from 'http';
import cors from 'cors';
import { typeDefs, resolvers } from './schema';
import { createContext } from './context';

async function startServer() {
  const app = express();
  const httpServer = http.createServer(app);

  const server = new ApolloServer({
    typeDefs,
    resolvers,
    plugins: [
      // Ensures Apollo Server shuts down gracefully alongside
      // the Express HTTP server — no dangling connections.
      ApolloServerPluginDrainHttpServer({ httpServer }),
    ],
  });

  // Must call start() before applying middleware
  await server.start();

  // Health check endpoint (useful for load balancers)
  app.get('/health', (_, res) => res.json({ status: 'ok' }));

  // GraphQL endpoint
  app.use(
    '/graphql',
    cors<cors.CorsRequest>({
      origin: ['http://localhost:3000', 'https://myapp.com'],
    }),
    express.json(),
    expressMiddleware(server, {
      context: createContext,
    })
  );

  httpServer.listen(4000, () => {
    console.log('Server ready at http://localhost:4000/graphql');
  });
}

startServer();
```

### Why `ApolloServerPluginDrainHttpServer`?

Without this plugin, calling `server.stop()` would stop Apollo from accepting new requests but would not close existing HTTP connections. The plugin ties Apollo's lifecycle to the HTTP server's lifecycle: when Apollo stops, it signals the HTTP server to stop accepting new connections and waits for in-flight requests to complete (draining).

---

## 4. Fastify Integration

Fastify is a high-performance alternative to Express. Apollo Server 4 supports it through the `@as-integrations/fastify` package.

```bash
npm install @apollo/server graphql fastify @as-integrations/fastify
```

```typescript
// src/index.ts
import Fastify from 'fastify';
import { ApolloServer } from '@apollo/server';
import fastifyApollo, {
  fastifyApolloDrainPlugin,
} from '@as-integrations/fastify';
import { typeDefs, resolvers } from './schema';

async function startServer() {
  const fastify = Fastify({ logger: true });

  const server = new ApolloServer({
    typeDefs,
    resolvers,
    plugins: [fastifyApolloDrainPlugin(fastify)],
  });

  await server.start();

  // Register Apollo as a Fastify plugin
  await fastify.register(fastifyApollo(server), {
    context: async (request) => ({
      token: request.headers.authorization,
    }),
  });

  await fastify.listen({ port: 4000 });
  console.log('Fastify + Apollo Server running on port 4000');
}

startServer();
```

Fastify's schema-based validation and serialization make it faster than Express for JSON-heavy workloads. Choose Fastify when raw throughput matters.

---

## 5. Schema-First vs Code-First

There are two fundamentally different approaches to defining your GraphQL schema.

### Schema-First (SDL-First)

You write the schema in GraphQL SDL (Schema Definition Language) and then implement resolvers that match the schema.

```typescript
// Schema (SDL)
const typeDefs = `#graphql
  type User {
    id: ID!
    name: String!
    email: String!
    posts: [Post!]!
  }

  type Post {
    id: ID!
    title: String!
    content: String!
    author: User!
  }

  type Query {
    user(id: ID!): User
    posts(limit: Int = 10): [Post!]!
  }
`;

// Resolvers (must match the schema exactly)
const resolvers = {
  Query: {
    user: (_, { id }, { db }) => db.user.findUnique({ where: { id } }),
    posts: (_, { limit }, { db }) => db.post.findMany({ take: limit }),
  },
  User: {
    posts: (parent, _, { db }) =>
      db.post.findMany({ where: { authorId: parent.id } }),
  },
  Post: {
    author: (parent, _, { db }) =>
      db.user.findUnique({ where: { id: parent.authorId } }),
  },
};
```

**Pros**: Schema is readable by non-developers, easy to review in PRs, clear contract.
**Cons**: Schema and resolvers can drift apart; no type safety between them without codegen.

### Code-First

You define the schema programmatically using a library. The SDL is generated automatically.

```typescript
// Using Nexus (code-first library for Node.js)
import { makeSchema, objectType, queryType, intArg, idArg } from 'nexus';

const User = objectType({
  name: 'User',
  definition(t) {
    t.nonNull.id('id');
    t.nonNull.string('name');
    t.nonNull.string('email');
    t.nonNull.list.nonNull.field('posts', {
      type: 'Post',
      resolve: (parent, _, { db }) =>
        db.post.findMany({ where: { authorId: parent.id } }),
    });
  },
});

const Post = objectType({
  name: 'Post',
  definition(t) {
    t.nonNull.id('id');
    t.nonNull.string('title');
    t.nonNull.string('content');
    t.nonNull.field('author', {
      type: 'User',
      resolve: (parent, _, { db }) =>
        db.user.findUnique({ where: { id: parent.authorId } }),
    });
  },
});

const Query = queryType({
  definition(t) {
    t.field('user', {
      type: 'User',
      args: { id: idArg() },
      resolve: (_, { id }, { db }) => db.user.findUnique({ where: { id } }),
    });
    t.nonNull.list.nonNull.field('posts', {
      type: 'Post',
      args: { limit: intArg({ default: 10 }) },
      resolve: (_, { limit }, { db }) => db.post.findMany({ take: limit }),
    });
  },
});

export const schema = makeSchema({
  types: [User, Post, Query],
  outputs: {
    schema: './generated/schema.graphql',  // Auto-generated SDL
    typegen: './generated/nexus-typegen.ts', // Auto-generated TS types
  },
});
```

**Pros**: Full type safety, schema and resolvers cannot drift, IDE autocompletion.
**Cons**: Schema is harder to read as a whole; steeper learning curve.

### Comparison

| Aspect | Schema-First | Code-First |
|--------|-------------|-----------|
| Readability | SDL is human-readable | Code can be verbose |
| Type Safety | Requires codegen (e.g., GraphQL Code Generator) | Built-in |
| Team Workflow | Frontend/backend can agree on SDL first | Primarily backend-driven |
| Tooling | graphql-codegen, graphql-tools | Nexus, TypeGraphQL, Pothos |
| Schema Drift | Possible (schema says one thing, resolver does another) | Impossible |
| Learning Curve | Lower (SDL is simple) | Higher (library-specific API) |

**Recommendation**: For teams where frontend and backend collaborate on the API contract, schema-first with GraphQL Code Generator works well. For backend-heavy teams who want maximum type safety, code-first is preferable. Both are production-ready.

---

## 6. The Context Function

The context function is the bridge between the HTTP request and your resolvers. It runs once per request and creates an object shared by every resolver in that request's execution.

```typescript
// src/context.ts
import { PrismaClient } from '@prisma/client';
import { verifyToken } from './auth';
import { createLoaders } from './dataloaders';

// Initialize Prisma once (connection pooling)
const prisma = new PrismaClient();

export interface Context {
  db: PrismaClient;
  currentUser: User | null;
  loaders: ReturnType<typeof createLoaders>;
  requestId: string;
}

export async function createContext({ req }): Promise<Context> {
  // 1. Authentication: extract user from token
  const token = req.headers.authorization?.replace('Bearer ', '');
  const currentUser = token ? await verifyToken(token, prisma) : null;

  // 2. DataLoaders: create per-request instances.
  //    DataLoaders cache results within a single request.
  //    Sharing them across requests would leak data between users.
  const loaders = createLoaders(prisma);

  // 3. Request ID for logging and tracing
  const requestId =
    (req.headers['x-request-id'] as string) || crypto.randomUUID();

  return {
    db: prisma,
    currentUser,
    loaders,
    requestId,
  };
}
```

### What Should Go in Context?

| Include | Exclude |
|---------|---------|
| Database client | Request body (resolvers get args) |
| Authenticated user | Response object (resolvers return data) |
| DataLoaders (per-request) | Business logic (belongs in services) |
| Request ID | Large cached datasets |
| Logger instance | |

### Why DataLoaders Must Be Per-Request

DataLoader caches by key within a single request to solve the N+1 problem. If you share a DataLoader across requests, User A's request might return User B's cached data. Always create fresh DataLoaders in the context function.

---

## 7. Plugins

Apollo Server's plugin system lets you hook into every phase of the request lifecycle without modifying resolvers.

### Lifecycle Hooks

```
serverWillStart
  └── requestDidStart
       ├── didResolveSource
       ├── parsingDidStart
       ├── validationDidStart
       ├── didResolveOperation
       ├── responseForOperation
       ├── executionDidStart
       │    └── willResolveField (per field)
       ├── willSendResponse
       └── didEncounterErrors
```

### Logging Plugin

```typescript
// src/plugins/logging.ts
import { ApolloServerPlugin } from '@apollo/server';

export const loggingPlugin: ApolloServerPlugin = {
  async requestDidStart({ request, contextValue }) {
    const start = Date.now();
    const { operationName } = request;

    console.log(`[${contextValue.requestId}] ${operationName || 'anonymous'} started`);

    return {
      async willSendResponse({ response }) {
        const duration = Date.now() - start;
        console.log(
          `[${contextValue.requestId}] ${operationName || 'anonymous'} ` +
          `completed in ${duration}ms`
        );
      },

      async didEncounterErrors({ errors }) {
        for (const error of errors) {
          console.error(
            `[${contextValue.requestId}] Error: ${error.message}`,
            error.extensions
          );
        }
      },
    };
  },
};
```

### Response Cache Plugin

Apollo provides a built-in full-response cache plugin. It caches the entire JSON response keyed by query + variables.

```bash
npm install @apollo/server-plugin-response-cache
```

```typescript
import responseCachePlugin from '@apollo/server-plugin-response-cache';

const server = new ApolloServer({
  typeDefs,
  resolvers,
  plugins: [
    responseCachePlugin({
      // Custom session ID for per-user caching
      sessionId: async ({ request }) =>
        request.http?.headers.get('authorization') || null,
    }),
  ],
});
```

To use the cache, annotate your schema with `@cacheControl`:

```graphql
type Query {
  posts: [Post!]! @cacheControl(maxAge: 60)       # Cache 60 seconds
  currentUser: User @cacheControl(maxAge: 0)        # Never cache
}

type Post @cacheControl(maxAge: 120) {
  id: ID!
  title: String!
  viewCount: Int! @cacheControl(maxAge: 10)          # Field-level override
}
```

### Custom Plugin: Query Complexity

```typescript
// src/plugins/complexity.ts
import { ApolloServerPlugin } from '@apollo/server';

const MAX_DEPTH = 10;

export const depthLimitPlugin: ApolloServerPlugin = {
  async requestDidStart() {
    return {
      async didResolveOperation({ document }) {
        const depth = calculateDepth(document);
        if (depth > MAX_DEPTH) {
          throw new GraphQLError(
            `Query depth ${depth} exceeds maximum of ${MAX_DEPTH}`,
            { extensions: { code: 'QUERY_TOO_DEEP' } }
          );
        }
      },
    };
  },
};

function calculateDepth(document: DocumentNode): number {
  // Recursive traversal of the AST to find max nesting depth
  let maxDepth = 0;

  function traverse(selections: readonly SelectionNode[], depth: number) {
    for (const selection of selections) {
      if (selection.kind === 'Field' && selection.selectionSet) {
        traverse(selection.selectionSet.selections, depth + 1);
      }
      if (selection.kind === 'InlineFragment' && selection.selectionSet) {
        traverse(selection.selectionSet.selections, depth);
      }
    }
    maxDepth = Math.max(maxDepth, depth);
  }

  for (const def of document.definitions) {
    if (def.kind === 'OperationDefinition' && def.selectionSet) {
      traverse(def.selectionSet.selections, 1);
    }
  }

  return maxDepth;
}
```

---

## 8. Error Handling and Formatting

Apollo Server wraps all errors in a standard format. You can customize how errors appear to clients using the `formatError` option.

```typescript
import { ApolloServer } from '@apollo/server';
import { GraphQLError, GraphQLFormattedError } from 'graphql';

const server = new ApolloServer({
  typeDefs,
  resolvers,
  formatError: (
    formattedError: GraphQLFormattedError,
    error: unknown
  ) => {
    // Log the full error server-side
    console.error('GraphQL Error:', error);

    // In production, hide internal error details from clients
    if (process.env.NODE_ENV === 'production') {
      // Known errors: pass through with their code
      if (
        formattedError.extensions?.code === 'UNAUTHENTICATED' ||
        formattedError.extensions?.code === 'FORBIDDEN' ||
        formattedError.extensions?.code === 'BAD_USER_INPUT'
      ) {
        return formattedError;
      }

      // Unknown errors: mask the message
      return {
        message: 'Internal server error',
        extensions: { code: 'INTERNAL_SERVER_ERROR' },
      };
    }

    // In development, return full details
    return formattedError;
  },
});
```

### Throwing Structured Errors

```typescript
import { GraphQLError } from 'graphql';

// Validation error
throw new GraphQLError('Email address is invalid', {
  extensions: {
    code: 'BAD_USER_INPUT',
    field: 'email',
    value: input.email,
  },
});

// Not found
throw new GraphQLError('Post not found', {
  extensions: {
    code: 'NOT_FOUND',
    resourceType: 'Post',
    resourceId: id,
  },
});

// Rate limited
throw new GraphQLError('Too many requests', {
  extensions: {
    code: 'RATE_LIMITED',
    retryAfter: 30,
  },
});
```

---

## 9. Apollo Studio Integration

Apollo Studio (formerly Apollo Engine) provides a cloud platform for schema management, metrics, and operation tracing.

### Connecting to Apollo Studio

```bash
# Set your Apollo API key
export APOLLO_KEY=service:my-graph:abc123
export APOLLO_GRAPH_REF=my-graph@production
```

```typescript
const server = new ApolloServer({
  typeDefs,
  resolvers,
  plugins: [
    // Usage reporting sends anonymous operation metrics to Apollo Studio
    ApolloServerPluginUsageReporting({
      sendVariableValues: { none: true }, // Never send variable values
      sendHeaders: { none: true },         // Never send headers
    }),

    // Schema reporting automatically publishes your schema on server start
    ApolloServerPluginSchemaReporting(),
  ],
});
```

### What Apollo Studio Provides

| Feature | Description |
|---------|-------------|
| **Schema Registry** | Version history of your schema; diff changes across deploys |
| **Operation Metrics** | Request rate, latency (p50/p95/p99), error rate per operation |
| **Field Usage** | Which fields are actually used by clients — safe deprecation insights |
| **Schema Checks** | CI integration that validates schema changes against real traffic |
| **Explorer** | In-browser GraphQL IDE with your schema and real data |

Apollo Studio is free for small teams (up to 10M operations/month) and provides valuable observability. Even if you do not use Apollo Federation, the metrics and schema registry are worth setting up.

---

## 10. Production Configuration

A production Apollo Server needs several additional configurations beyond what you use in development.

### Complete Production Setup

```typescript
// src/index.ts
import { ApolloServer } from '@apollo/server';
import { expressMiddleware } from '@apollo/server/express4';
import {
  ApolloServerPluginDrainHttpServer,
} from '@apollo/server/plugin/drainHttpServer';
import {
  ApolloServerPluginLandingPageDisabled,
} from '@apollo/server/plugin/disabled';
import {
  ApolloServerPluginLandingPageLocalDefault,
} from '@apollo/server/plugin/landingPage/default';
import express from 'express';
import http from 'http';
import cors from 'cors';
import helmet from 'helmet';

const IS_PRODUCTION = process.env.NODE_ENV === 'production';

const app = express();
const httpServer = http.createServer(app);

const server = new ApolloServer({
  typeDefs,
  resolvers,

  // Disable introspection in production to hide your schema
  // from unauthorized users. Enable for staging/development.
  introspection: !IS_PRODUCTION,

  // Plugin configuration
  plugins: [
    ApolloServerPluginDrainHttpServer({ httpServer }),

    // Landing page: show Apollo Sandbox in dev, disable in prod
    IS_PRODUCTION
      ? ApolloServerPluginLandingPageDisabled()
      : ApolloServerPluginLandingPageLocalDefault(),
  ],

  // CSRF prevention: requires a specific header on every request.
  // Apollo Client and most GraphQL clients send this automatically.
  csrfPrevention: true,

  // Error formatting (see Section 8)
  formatError: productionErrorFormatter,

  // Cache control
  cacheControl: {
    defaultMaxAge: 0, // Do not cache by default
  },
});

await server.start();

// Security headers
app.use(helmet());

// Trust proxy (if behind a load balancer)
app.set('trust proxy', 1);

app.use(
  '/graphql',
  cors({
    origin: IS_PRODUCTION
      ? ['https://myapp.com']
      : ['http://localhost:3000'],
    credentials: true,
  }),
  express.json({ limit: '1mb' }), // Limit request body size
  expressMiddleware(server, { context: createContext })
);

httpServer.listen(4000);
```

### Production Checklist

| Category | Item | Implementation |
|----------|------|----------------|
| **Security** | CSRF prevention | `csrfPrevention: true` |
| | Introspection control | `introspection: false` in prod |
| | CORS | Explicit origin whitelist |
| | Body size limit | `express.json({ limit: '1mb' })` |
| | Query depth limit | Depth limit plugin |
| | Query complexity limit | Complexity analysis plugin |
| **Performance** | DataLoader | Per-request DataLoaders |
| | Response caching | `@cacheControl` + cache plugin |
| | Persisted queries | APQ or registered operations |
| **Observability** | Logging | Request/error logging plugin |
| | Metrics | Apollo Studio or Prometheus |
| | Tracing | OpenTelemetry integration |
| **Reliability** | Graceful shutdown | `ApolloServerPluginDrainHttpServer` |
| | Health checks | `/health` endpoint |
| | Rate limiting | Express rate-limit middleware |

---

## 11. Practice Problems

### Exercise 1: Standalone to Express Migration (Beginner)

You have a working Apollo Server using `startStandaloneServer`. Migrate it to Express middleware so you can add:

1. A `GET /health` endpoint that returns `{ status: "ok", uptime: process.uptime() }`
2. A `POST /webhooks/stripe` endpoint for Stripe webhook processing
3. CORS configured to allow only `http://localhost:3000` and `https://yourapp.com`
4. The `ApolloServerPluginDrainHttpServer` for graceful shutdown

Write the complete migration code.

### Exercise 2: Custom Logging Plugin (Intermediate)

Build an Apollo Server plugin that logs:

1. Operation name, type (query/mutation/subscription), and variables (redacting any field named `password`)
2. Execution duration in milliseconds
3. Which resolvers were called and how long each took (using `willResolveField`)
4. A summary at the end showing the slowest resolver

Format the output as structured JSON suitable for ingestion by a log aggregation system.

### Exercise 3: Schema-First with Codegen (Intermediate)

Set up a schema-first Apollo Server project with type safety:

1. Write a schema with `User`, `Post`, and `Comment` types, plus queries and mutations
2. Configure GraphQL Code Generator to generate TypeScript types from the SDL
3. Write resolvers that are fully typed using the generated types
4. Demonstrate that a type error in a resolver is caught at compile time

Provide the `codegen.yml` configuration and the generated resolver type signatures.

### Exercise 4: Multi-Schema Stitching (Advanced)

You have two Apollo Server instances:

- **Users Service**: `User` type with queries `user(id)` and `users`
- **Posts Service**: `Post` type with queries `post(id)` and `posts`, where `Post` has an `authorId` field

Build a gateway server that:

1. Stitches both schemas together
2. Adds a `posts` field to `User` that resolves by querying the Posts Service
3. Adds an `author` field to `Post` that resolves by querying the Users Service
4. Uses DataLoader to batch cross-service calls

### Exercise 5: Production Hardening Audit (Advanced)

Given the following Apollo Server configuration, identify all production issues and provide the corrected version:

```typescript
const server = new ApolloServer({
  typeDefs,
  resolvers,
});

const { url } = await startStandaloneServer(server, {
  listen: { port: 4000 },
});
```

Your hardened version should address at least 8 specific concerns across security, performance, observability, and reliability. Explain why each change matters.

---

## 12. References

- Apollo Server 4 documentation — https://www.apollographql.com/docs/apollo-server/
- Apollo Server migration guide (v3 to v4) — https://www.apollographql.com/docs/apollo-server/migration
- Express integration — https://www.apollographql.com/docs/apollo-server/api/express-middleware
- Apollo Server plugins — https://www.apollographql.com/docs/apollo-server/integrations/plugins
- Apollo Studio — https://www.apollographql.com/docs/studio/
- GraphQL Code Generator — https://the-guild.dev/graphql/codegen

---

**Previous**: [Authentication and Authorization](./07_Authentication_Authorization.md) | **Next**: [GraphQL Clients](./09_GraphQL_Clients.md)
