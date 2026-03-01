# 11. Persisted Queries and Caching

**Previous**: [Code-First with Python](./10_Code_First_Python.md) | **Next**: [Federation](./12_Federation.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain why GraphQL's POST-based model bypasses traditional HTTP caching and how to work around it
2. Implement Automatic Persisted Queries (APQ) to enable CDN-friendly query caching
3. Configure full persisted queries with build-time extraction and operation allowlists
4. Design a multi-layer caching strategy combining CDN, server-side, and client-side caches
5. Manage Apollo Client's InMemoryCache with cache policies, eviction, and garbage collection

## Table of Contents

1. [Why Cache GraphQL?](#1-why-cache-graphql)
2. [Automatic Persisted Queries (APQ)](#2-automatic-persisted-queries-apq)
3. [Full Persisted Queries](#3-full-persisted-queries)
4. [HTTP Caching Strategies](#4-http-caching-strategies)
5. [Apollo Client Cache](#5-apollo-client-cache)
6. [Server-Side Caching](#6-server-side-caching)
7. [Multi-Layer Caching Architecture](#7-multi-layer-caching-architecture)
8. [Practice Problems](#8-practice-problems)

**Difficulty**: ⭐⭐⭐

---

GraphQL's flexibility is its greatest strength and its biggest caching challenge. Every query can request a different shape of data, and the standard transport uses POST requests -- which HTTP caches ignore by default. This lesson explores the techniques that make GraphQL just as cacheable as REST, from persisted queries that turn POST into GET, to multi-layer caching architectures that dramatically reduce server load.

---

## 1. Why Cache GraphQL?

### The Problem with POST

REST APIs map naturally to HTTP caching: `GET /api/users/42` always returns the same resource, and CDNs, browsers, and reverse proxies all understand how to cache GET requests. GraphQL breaks this model.

```
REST (Cacheable by default):
  GET /api/users/42
  → CDN caches response
  → Browser caches response
  → Reverse proxy caches response

GraphQL (Not cacheable by default):
  POST /graphql
  Body: { "query": "{ user(id: 42) { name email } }" }
  → CDN: "It's a POST, skip cache"
  → Browser: "It's a POST, skip cache"
  → Reverse proxy: "It's a POST, skip cache"
```

### The Three Challenges

1. **Transport-level**: POST requests bypass HTTP caching infrastructure
2. **Uniqueness**: Each client can send different query shapes for the same data
3. **Granularity**: A single GraphQL response may contain data with different cache lifetimes

### Caching Layers

```
┌─────────────────────────────────────────────────┐
│                  Client Layer                     │
│  Apollo InMemoryCache / urql's Graphcache         │
├─────────────────────────────────────────────────┤
│                  CDN Layer                         │
│  CloudFront / Fastly / Cloudflare                 │
│  (Requires GET requests or persisted queries)     │
├─────────────────────────────────────────────────┤
│               Application Layer                   │
│  Response cache plugin / Redis / Memcached        │
├─────────────────────────────────────────────────┤
│                Data Layer                          │
│  DataLoader per-request cache / ORM cache         │
└─────────────────────────────────────────────────┘
```

---

## 2. Automatic Persisted Queries (APQ)

APQ is a protocol between client and server that replaces full query strings with short hashes. The first time a hash is seen, the client sends the full query; subsequent requests use only the hash -- small enough for GET requests.

### How APQ Works

```
Step 1: Client sends hash only
  GET /graphql?extensions={"persistedQuery":{"sha256Hash":"abc123..."}}
  → Server: "I don't have that hash" → returns PersistedQueryNotFound

Step 2: Client sends hash + full query
  POST /graphql
  Body: { "query": "{ user(id: 42) { name } }",
          "extensions": { "persistedQuery": { "sha256Hash": "abc123..." } } }
  → Server: stores hash→query mapping, returns data

Step 3: All future requests use hash only
  GET /graphql?extensions={"persistedQuery":{"sha256Hash":"abc123..."}}
  → Server: finds query by hash, executes, returns data
  → CDN can cache this GET request!
```

### Server Setup (Apollo Server 4)

```typescript
import { ApolloServer } from '@apollo/server';
import { expressMiddleware } from '@apollo/server/express4';
import { ApolloServerPluginCacheControl } from '@apollo/server/plugin/cacheControl';
import { KeyvAdapter } from '@apollo/utils.keyvadapter';
import Keyv from 'keyv';

const server = new ApolloServer({
  typeDefs,
  resolvers,
  // APQ is enabled by default in Apollo Server 4
  // To configure the cache backend:
  cache: new KeyvAdapter(new Keyv('redis://localhost:6379')),
  plugins: [
    ApolloServerPluginCacheControl({
      defaultMaxAge: 60, // 60 seconds default
    }),
  ],
});
```

### Client Setup (Apollo Client)

```typescript
import { ApolloClient, InMemoryCache, HttpLink } from '@apollo/client';
import { createPersistedQueryLink } from '@apollo/client/link/persisted-queries';
import { sha256 } from 'crypto-hash';

const httpLink = new HttpLink({ uri: '/graphql' });

// APQ link sits before the HTTP link in the chain
const persistedQueryLink = createPersistedQueryLink({
  sha256,
  useGETForHashedQueries: true, // Use GET for cache-friendly requests
});

const client = new ApolloClient({
  link: persistedQueryLink.concat(httpLink),
  cache: new InMemoryCache(),
});
```

### APQ with GET Requests and CDN

When `useGETForHashedQueries` is enabled, the request becomes a simple GET:

```
GET /graphql
  ?operationName=GetUser
  &variables={"id":"42"}
  &extensions={"persistedQuery":{"version":1,"sha256Hash":"ecf4edb..."}}
```

This URL is cacheable by any CDN. Add `Cache-Control` headers on the server to control TTL.

---

## 3. Full Persisted Queries

Full persisted queries go further than APQ: the query-to-hash mapping is established at **build time**, and the server only accepts known hashes. This provides both performance and security benefits.

### Build-Time Query Extraction

```typescript
// queries/GetUser.graphql
query GetUser($id: ID!) {
  user(id: $id) {
    id
    name
    email
    avatar
  }
}
```

Use a tool like `graphql-codegen` to extract all queries at build time:

```yaml
# codegen.yml
schema: "http://localhost:4000/graphql"
documents: "src/**/*.graphql"
generates:
  ./src/generated/persisted-queries.json:
    plugins:
      - graphql-codegen-persisted-query-ids
```

This produces a manifest:

```json
{
  "ecf4edb46db40b5132295c0291d62fb65d6759a9eedfa4d5d612dd5ec54a6b38": "query GetUser($id: ID!) { user(id: $id) { id name email avatar } }",
  "a1b2c3d4e5f6...": "query ListProducts($first: Int) { products(first: $first) { edges { node { id name price } } } }"
}
```

### Server-Side Allowlist

```typescript
import persistedQueries from './persisted-queries.json';

const server = new ApolloServer({
  typeDefs,
  resolvers,
  persistedQueries: {
    cache: new KeyvAdapter(new Keyv()),
  },
  // Only accept queries in the allowlist
  validationRules: [
    (context) => ({
      Document(node) {
        // Custom rule: reject if query isn't in the manifest
      },
    }),
  ],
});
```

### Why Use Full Persisted Queries?

| Benefit | Description |
|---------|-------------|
| **Security** | Only pre-approved queries can execute -- blocks arbitrary query injection |
| **Performance** | No parsing/validation needed for known queries |
| **Bandwidth** | Hash is ~64 bytes vs. potentially kilobytes of query text |
| **CDN-friendly** | Small GET URLs are ideal for CDN caching |

---

## 4. HTTP Caching Strategies

### Cache-Control with GraphQL

Apollo Server's cache control plugin reads `@cacheControl` directives from your schema:

```graphql
type Query {
  # Public data, cacheable for 5 minutes
  products: [Product!]! @cacheControl(maxAge: 300, scope: PUBLIC)

  # Per-user data, private cache only
  me: User @cacheControl(maxAge: 60, scope: PRIVATE)
}

type Product {
  id: ID!
  name: String!
  price: Float!
  # Inventory changes frequently
  stockCount: Int! @cacheControl(maxAge: 30)
}
```

The server calculates the overall `Cache-Control` header as the **minimum** maxAge across all fields in the response:

```
# If a query touches products (300s) and stockCount (30s):
Cache-Control: public, max-age=30
```

### GET Requests for HTTP Caching

```typescript
// Express middleware to support GET queries
import express from 'express';

const app = express();

// Apollo Server handles both GET and POST automatically
// GET: query and variables in query string
// POST: query and variables in request body
app.use('/graphql', expressMiddleware(server));
```

### CDN Configuration (CloudFront Example)

```yaml
# CloudFront behavior for /graphql
CacheBehavior:
  PathPattern: /graphql
  AllowedMethods:
    - GET
    - HEAD
    - OPTIONS
  CachedMethods:
    - GET
    - HEAD
  ForwardedValues:
    QueryString: true  # Important: cache varies by query string
    Headers:
      - Authorization  # Don't cache authenticated responses together
  DefaultTTL: 60
  MaxTTL: 300
```

---

## 5. Apollo Client Cache

Apollo Client's `InMemoryCache` is a normalized, reactive cache that stores data by type and ID. Understanding how it works is essential for building responsive GraphQL applications.

### Normalized Cache Structure

```typescript
// When this query returns:
// { user: { __typename: "User", id: "42", name: "Alice", posts: [{ __typename: "Post", id: "1", title: "Hello" }] } }

// The cache stores it as:
{
  "ROOT_QUERY": {
    "user({\"id\":\"42\"})": { "__ref": "User:42" }
  },
  "User:42": {
    "__typename": "User",
    "id": "42",
    "name": "Alice",
    "posts": [{ "__ref": "Post:1" }]
  },
  "Post:1": {
    "__typename": "Post",
    "id": "1",
    "title": "Hello"
  }
}
```

### Cache Policies (fetchPolicy)

```typescript
const { data, loading } = useQuery(GET_USER, {
  variables: { id: '42' },
  // Choose a fetch policy:
  fetchPolicy: 'cache-first',       // Default: use cache, fetch only if missing
  // fetchPolicy: 'cache-and-network', // Return cache immediately, then update from network
  // fetchPolicy: 'network-only',      // Always fetch, but update cache
  // fetchPolicy: 'cache-only',        // Never fetch, only use cache
  // fetchPolicy: 'no-cache',          // Always fetch, don't store in cache
});
```

### When to Use Each Policy

| Policy | Use Case |
|--------|----------|
| `cache-first` | Static data (categories, config). Default for most queries. |
| `cache-and-network` | Data that changes but stale is acceptable (social feed, dashboards) |
| `network-only` | Data that must be fresh (payment info, real-time inventory) |
| `cache-only` | Offline mode or reading data you know is already cached |
| `no-cache` | Sensitive data you don't want stored (tokens, secrets) |

### Type Policies and Field Merging

```typescript
const cache = new InMemoryCache({
  typePolicies: {
    Query: {
      fields: {
        // Merge paginated results instead of replacing them
        products: {
          keyArgs: ['category'],  // Separate cache entries per category
          merge(existing = { edges: [] }, incoming) {
            return {
              ...incoming,
              edges: [...existing.edges, ...incoming.edges],
            };
          },
        },
      },
    },
    Product: {
      fields: {
        // Provide a read function that computes derived data
        displayPrice: {
          read(_, { readField }) {
            const price = readField('price');
            const currency = readField('currency');
            return `${currency} ${price.toFixed(2)}`;
          },
        },
      },
    },
  },
});
```

### Cache Eviction and Garbage Collection

```typescript
// Evict a specific entity
cache.evict({ id: 'User:42' });

// Evict a specific field from an entity
cache.evict({ id: 'User:42', fieldName: 'posts' });

// Run garbage collection to remove unreachable objects
cache.gc();

// Modify cached data directly
cache.modify({
  id: 'User:42',
  fields: {
    name(existingName) {
      return 'Updated Name';
    },
    posts(existingPosts, { toReference }) {
      // Add a new post reference
      return [...existingPosts, toReference({ __typename: 'Post', id: '99' })];
    },
  },
});
```

---

## 6. Server-Side Caching

### Response Cache Plugin

The `@apollo/server-plugin-response-cache` plugin caches entire responses keyed by query + variables:

```typescript
import responseCachePlugin from '@apollo/server-plugin-response-cache';

const server = new ApolloServer({
  typeDefs,
  resolvers,
  plugins: [
    responseCachePlugin({
      // Use the user's ID to separate cached responses for authenticated users
      sessionId: (requestContext) =>
        requestContext.request.http?.headers.get('authorization') || null,
    }),
  ],
});
```

### Redis-Based Cache Backend

```typescript
import { KeyvAdapter } from '@apollo/utils.keyvadapter';
import Keyv from 'keyv';
import KeyvRedis from '@keyv/redis';

const server = new ApolloServer({
  typeDefs,
  resolvers,
  // Redis store for both APQ and response caching
  cache: new KeyvAdapter(
    new Keyv({
      store: new KeyvRedis('redis://localhost:6379'),
      namespace: 'gql',
      ttl: 300_000, // 5 minutes in milliseconds
    })
  ),
});
```

### Partial Query Caching with @cacheControl

The real power comes from field-level cache hints. Each type and field can declare its own cache policy, and the server computes the overall response cacheability:

```graphql
type Query {
  topProducts: [Product!]! @cacheControl(maxAge: 3600, scope: PUBLIC)
  cart: Cart @cacheControl(maxAge: 0, scope: PRIVATE)
}

type Product @cacheControl(maxAge: 600) {
  id: ID!
  name: String!
  reviews: [Review!]! @cacheControl(maxAge: 120)
}
```

```typescript
// In resolvers, you can set cache hints dynamically
const resolvers = {
  Query: {
    topProducts: (_, __, { cacheControl }) => {
      // Override the schema hint at runtime
      cacheControl.setCacheHint({ maxAge: 1800, scope: 'PUBLIC' });
      return db.products.findTop();
    },
  },
};
```

---

## 7. Multi-Layer Caching Architecture

A production GraphQL deployment combines multiple caching layers:

```
Client Request
     │
     ▼
┌─────────────────────┐
│   Apollo Client      │  Layer 1: Normalized client cache
│   InMemoryCache      │  - Eliminates redundant network requests
│                      │  - Instant UI updates for cached data
└──────────┬──────────┘
           │ Cache miss
           ▼
┌─────────────────────┐
│   CDN / Edge Cache   │  Layer 2: HTTP cache (APQ + GET)
│   CloudFront, etc.   │  - Geographic distribution
│                      │  - Handles read-heavy public data
└──────────┬──────────┘
           │ Cache miss
           ▼
┌─────────────────────┐
│   Response Cache     │  Layer 3: Server response cache
│   (Redis)            │  - Full response caching
│                      │  - Session-aware for auth data
└──────────┬──────────┘
           │ Cache miss
           ▼
┌─────────────────────┐
│   DataLoader         │  Layer 4: Request-scoped batch cache
│   (per-request)      │  - Deduplicates within a single request
│                      │  - Solves N+1 queries
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   Database / API     │  Source of truth
└─────────────────────┘
```

### Cache Invalidation Strategies

```typescript
// Strategy 1: TTL-based (simplest)
// Set maxAge in @cacheControl and let entries expire

// Strategy 2: Mutation-based invalidation
const resolvers = {
  Mutation: {
    updateProduct: async (_, { id, input }, { cache, dataSources }) => {
      const product = await dataSources.products.update(id, input);

      // Invalidate CDN cache for this product
      await fetch(`https://api.fastly.com/purge/product-${id}`, {
        method: 'POST',
        headers: { 'Fastly-Key': process.env.FASTLY_KEY },
      });

      return product;
    },
  },
};

// Strategy 3: Event-driven invalidation
// Listen for database change events and purge relevant cache entries
import { PubSub } from 'graphql-subscriptions';
const pubsub = new PubSub();

pubsub.subscribe('PRODUCT_UPDATED', ({ productId }) => {
  cache.evict({ id: `Product:${productId}` });
  cache.gc();
});
```

---

## 8. Practice Problems

### Problem 1: APQ Configuration
Set up an Apollo Server with APQ enabled, backed by a Redis cache. Write a client that sends a query using APQ with GET requests. Verify that the first request sends the full query (POST) and subsequent requests use only the hash (GET).

### Problem 2: Cache Policy Design
Given the following schema for a news site, add appropriate `@cacheControl` directives. Consider which data is public vs. private, and which fields change frequently vs. rarely:

```graphql
type Query {
  headlines: [Article!]!
  article(slug: String!): Article
  me: User
  myBookmarks: [Article!]!
}

type Article {
  id: ID!
  title: String!
  body: String!
  author: User!
  publishedAt: DateTime!
  viewCount: Int!
  comments: [Comment!]!
}

type User {
  id: ID!
  name: String!
  email: String!
}
```

### Problem 3: Client Cache Merging
Implement an Apollo Client `InMemoryCache` with type policies that correctly merge paginated results for an infinite scroll feed. The `feed` query uses cursor-based pagination with `after` and `first` arguments.

### Problem 4: Cache Invalidation
Design a cache invalidation strategy for an e-commerce product catalog. When a product's price changes via a mutation, the following caches must be updated: (a) the response cache on the server, (b) the CDN, and (c) any connected clients via subscriptions. Write the mutation resolver and supporting infrastructure code.

### Problem 5: Multi-Layer Benchmark
Create a test script that measures response times across different caching scenarios: (a) no cache, (b) DataLoader only, (c) DataLoader + response cache, (d) full stack with CDN simulation. Document the latency reduction at each layer.

---

## References

- [Apollo Client Caching](https://www.apollographql.com/docs/react/caching/overview)
- [Automatic Persisted Queries](https://www.apollographql.com/docs/apollo-server/performance/apq/)
- [Apollo Response Cache Plugin](https://www.apollographql.com/docs/apollo-server/performance/caching/)
- [GraphQL Caching at Scale (Netflix)](https://netflixtechblog.com/)
- [Persisted Queries Best Practices](https://www.apollographql.com/docs/graphos/operations/persisted-queries)

---

**Previous**: [Code-First with Python](./10_Code_First_Python.md) | **Next**: [Federation](./12_Federation.md)
