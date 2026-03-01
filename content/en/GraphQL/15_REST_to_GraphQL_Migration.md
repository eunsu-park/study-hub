# 15. REST to GraphQL Migration

**Previous**: [Performance and Security](./14_Performance_Security.md) | **Next**: [Project: API Gateway](./16_Project_API_Gateway.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Evaluate the three migration strategies (wrap, extend, replace) and choose the right one for a given codebase
2. Wrap existing REST APIs as GraphQL resolvers using RESTDataSource with caching and deduplication
3. Combine multiple GraphQL schemas using schema stitching and type merging
4. Auto-generate a GraphQL layer from REST/OpenAPI specifications using graphql-mesh
5. Implement the Backend for Frontend (BFF) pattern with GraphQL as an aggregation layer

## Table of Contents

1. [Migration Strategies Overview](#1-migration-strategies-overview)
2. [Wrapping REST APIs](#2-wrapping-rest-apis)
3. [Apollo RESTDataSource](#3-apollo-restdatasource)
4. [Schema Stitching](#4-schema-stitching)
5. [GraphQL Mesh](#5-graphql-mesh)
6. [Incremental Migration](#6-incremental-migration)
7. [BFF Pattern](#7-bff-pattern)
8. [Practice Problems](#8-practice-problems)

**Difficulty**: ⭐⭐⭐

---

Most teams do not start with GraphQL. They have REST APIs -- sometimes dozens of them -- serving mobile apps, web frontends, and partner integrations. Rewriting everything in GraphQL is risky and unnecessary. Instead, the most successful migrations wrap existing REST services behind a GraphQL layer, replace endpoints incrementally, and run both systems side by side until the transition is complete. This lesson walks through the tools and patterns that make that process smooth.

---

## 1. Migration Strategies Overview

### The Three Approaches

```
┌───────────────────────────────────────────────────────────────┐
│                                                                │
│   Strategy 1: WRAP                                            │
│   ─────────────────                                           │
│   GraphQL ──▶ REST APIs (unchanged)                           │
│   • Fastest to implement                                      │
│   • REST stays as the source of truth                         │
│   • Best for: "we want GraphQL clients, but can't change BE"  │
│                                                                │
│   Strategy 2: EXTEND                                          │
│   ──────────────────                                          │
│   GraphQL ──▶ REST + new GraphQL resolvers                    │
│   • Gradually replace REST endpoints                          │
│   • New features built directly in GraphQL                    │
│   • Best for: active development, incremental migration       │
│                                                                │
│   Strategy 3: REPLACE                                         │
│   ────────────────────                                        │
│   GraphQL ──▶ Database/Services directly                      │
│   • Full rewrite, no REST dependency                          │
│   • Highest effort, highest reward                            │
│   • Best for: greenfield or when REST is being decommissioned │
│                                                                │
└───────────────────────────────────────────────────────────────┘
```

### Decision Matrix

| Factor | Wrap | Extend | Replace |
|--------|------|--------|---------|
| Time to first value | Days | Weeks | Months |
| Risk | Low | Medium | High |
| REST dependency | Permanent | Temporary | None |
| New feature velocity | Same as REST | Increasing | Maximum |
| Recommended when | Read-heavy, stable REST | Active development | Greenfield / tech debt |

---

## 2. Wrapping REST APIs

The simplest migration: put a GraphQL server in front of your REST APIs. Each resolver calls a REST endpoint and transforms the response.

### Basic REST Wrapper

```typescript
import { ApolloServer } from '@apollo/server';
import { startStandaloneServer } from '@apollo/server/standalone';

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
    body: String!
    userId: ID!
  }

  type Query {
    user(id: ID!): User
    users: [User!]!
    post(id: ID!): Post
  }
`;

const resolvers = {
  Query: {
    // Each resolver wraps a REST endpoint
    user: async (_, { id }) => {
      const res = await fetch(`https://api.example.com/users/${id}`);
      return res.json();
    },
    users: async () => {
      const res = await fetch('https://api.example.com/users');
      return res.json();
    },
    post: async (_, { id }) => {
      const res = await fetch(`https://api.example.com/posts/${id}`);
      return res.json();
    },
  },

  User: {
    // Resolve the relationship between User and Posts
    posts: async (user) => {
      const res = await fetch(
        `https://api.example.com/users/${user.id}/posts`
      );
      return res.json();
    },
  },
};

const server = new ApolloServer({ typeDefs, resolvers });
startStandaloneServer(server, { listen: { port: 4000 } });
```

### Field Mapping

REST and GraphQL often use different naming conventions. Map them in the resolver:

```typescript
const resolvers = {
  User: {
    // REST returns snake_case, GraphQL uses camelCase
    firstName: (user) => user.first_name,
    lastName: (user) => user.last_name,
    createdAt: (user) => user.created_at,
    avatarUrl: (user) => user.avatar_url || user.profile_image,
  },
};
```

### Error Mapping

```typescript
const resolvers = {
  Query: {
    user: async (_, { id }) => {
      const res = await fetch(`https://api.example.com/users/${id}`);

      if (res.status === 404) {
        return null; // GraphQL convention: return null for not found
      }

      if (res.status === 401) {
        throw new GraphQLError('Authentication required', {
          extensions: { code: 'UNAUTHENTICATED' },
        });
      }

      if (!res.ok) {
        throw new GraphQLError(`REST API error: ${res.statusText}`, {
          extensions: {
            code: 'UPSTREAM_ERROR',
            statusCode: res.status,
          },
        });
      }

      return res.json();
    },
  },
};
```

---

## 3. Apollo RESTDataSource

`@apollo/datasource-rest` provides a structured way to wrap REST APIs with built-in caching, request deduplication, and error handling.

### Setup

```typescript
import { RESTDataSource } from '@apollo/datasource-rest';

class UsersAPI extends RESTDataSource {
  override baseURL = 'https://api.example.com/';

  // GET /users/:id -- cached automatically via HTTP cache headers
  async getUser(id: string) {
    return this.get(`users/${id}`);
  }

  // GET /users
  async getUsers() {
    return this.get('users');
  }

  // GET /users/:id/posts
  async getUserPosts(userId: string) {
    return this.get(`users/${userId}/posts`);
  }

  // POST /users -- mutations are never cached
  async createUser(input: CreateUserInput) {
    return this.post('users', { body: input });
  }

  // PUT /users/:id
  async updateUser(id: string, input: UpdateUserInput) {
    return this.put(`users/${id}`, { body: input });
  }

  // DELETE /users/:id
  async deleteUser(id: string) {
    return this.delete(`users/${id}`);
  }
}

class ProductsAPI extends RESTDataSource {
  override baseURL = 'https://products-api.example.com/v2/';

  async getProduct(id: string) {
    return this.get(`products/${id}`);
  }

  async searchProducts(query: string, page: number = 1) {
    return this.get('products/search', {
      params: { q: query, page: String(page) },
    });
  }
}
```

### Wiring Data Sources into Context

```typescript
import { ApolloServer } from '@apollo/server';
import { startStandaloneServer } from '@apollo/server/standalone';

interface ContextValue {
  dataSources: {
    usersAPI: UsersAPI;
    productsAPI: ProductsAPI;
  };
}

const server = new ApolloServer<ContextValue>({
  typeDefs,
  resolvers,
});

const { url } = await startStandaloneServer(server, {
  context: async () => {
    return {
      dataSources: {
        usersAPI: new UsersAPI(),
        productsAPI: new ProductsAPI(),
      },
    };
  },
});
```

### Request Deduplication

RESTDataSource automatically deduplicates GET requests within a single GraphQL operation:

```graphql
query {
  user1: user(id: "42") { name }
  user2: user(id: "42") { email }
  # Both aliases call getUser("42") → only ONE HTTP request to the REST API
}
```

### Caching Configuration

```typescript
class UsersAPI extends RESTDataSource {
  override baseURL = 'https://api.example.com/';

  // Override caching behavior per request
  async getUser(id: string) {
    return this.get(`users/${id}`, {
      cacheOptions: {
        ttl: 300, // Cache for 5 minutes regardless of REST headers
      },
    });
  }

  // Pass auth headers from context
  override willSendRequest(_path: string, request: AugmentedRequest) {
    request.headers['Authorization'] = this.context.token;
  }
}
```

---

## 4. Schema Stitching

Schema stitching combines multiple GraphQL schemas into one. Useful when you have several existing GraphQL services that you want to expose through a single gateway.

### Basic Stitching with @graphql-tools/stitch

```typescript
import { stitchSchemas } from '@graphql-tools/stitch';
import { schemaFromExecutor } from '@graphql-tools/wrap';
import { buildHTTPExecutor } from '@graphql-tools/executor-http';

// Create executors for remote schemas
const usersExecutor = buildHTTPExecutor({
  endpoint: 'http://users-service:4001/graphql',
});

const productsExecutor = buildHTTPExecutor({
  endpoint: 'http://products-service:4002/graphql',
});

// Fetch remote schemas
const usersSubschema = {
  schema: await schemaFromExecutor(usersExecutor),
  executor: usersExecutor,
};

const productsSubschema = {
  schema: await schemaFromExecutor(productsExecutor),
  executor: productsExecutor,
};

// Stitch them together
const gatewaySchema = stitchSchemas({
  subschemas: [usersSubschema, productsSubschema],
});
```

### Type Merging

When the same entity exists in multiple services, type merging combines them into a unified type:

```typescript
const gatewaySchema = stitchSchemas({
  subschemas: [
    {
      schema: usersSchema,
      executor: usersExecutor,
      merge: {
        User: {
          // How to fetch a User from this service
          selectionSet: '{ id }',
          fieldName: 'user',
          args: (originalObject) => ({ id: originalObject.id }),
        },
      },
    },
    {
      schema: reviewsSchema,
      executor: reviewsExecutor,
      merge: {
        User: {
          // Reviews service can also resolve User (with reviews field)
          selectionSet: '{ id }',
          fieldName: 'userById',
          args: (originalObject) => ({ id: originalObject.id }),
        },
      },
    },
  ],
});

// Now queries can span both services:
// query { user(id: "1") { name reviews { body } } }
//   name → from users service
//   reviews → from reviews service
```

### Schema Stitching vs. Federation

| Feature | Schema Stitching | Apollo Federation |
|---------|-----------------|-------------------|
| Gateway logic | In the gateway | In the router (precompiled) |
| Subgraph awareness | Subgraphs are unaware | Subgraphs use federation directives |
| Type merging | Gateway configures merging | `@key` + `__resolveReference` |
| Vendor lock-in | None (@graphql-tools) | Apollo ecosystem |
| Performance | Good for simple graphs | Better for complex graphs |
| Use case | Few services, quick setup | Many services, team ownership |

---

## 5. GraphQL Mesh

GraphQL Mesh auto-generates a GraphQL schema from non-GraphQL sources: REST/OpenAPI, gRPC, databases, and more.

### Setup with OpenAPI

```yaml
# .meshrc.yaml
sources:
  - name: PetStore
    handler:
      openapi:
        source: https://petstore.swagger.io/v2/swagger.json
        # Or a local file:
        # source: ./openapi-spec.yaml

  - name: WeatherAPI
    handler:
      openapi:
        source: ./weather-openapi.yaml
        operationHeaders:
          'x-api-key': '{env.WEATHER_API_KEY}'

serve:
  port: 4000
```

```bash
# Install dependencies
npm install @graphql-mesh/cli @graphql-mesh/openapi

# Start the mesh gateway
npx mesh dev
```

This automatically generates GraphQL types and resolvers from the OpenAPI specification:

```graphql
# Auto-generated from PetStore OpenAPI spec
type Query {
  findPetsByStatus(status: [Status!]): [Pet!]
  getPetById(petId: Int!): Pet
  getInventory: JSON
}

type Mutation {
  addPet(input: PetInput!): Pet
  updatePet(input: PetInput!): Pet
  deletePet(petId: Int!): JSON
}

type Pet {
  id: Int
  name: String!
  category: Category
  photoUrls: [String!]!
  tags: [Tag!]
  status: Status
}
```

### Transforms

GraphQL Mesh supports transforms to customize the generated schema:

```yaml
# .meshrc.yaml
sources:
  - name: PetStore
    handler:
      openapi:
        source: https://petstore.swagger.io/v2/swagger.json
    transforms:
      # Rename types
      - rename:
          renames:
            - from:
                type: Pet
              to:
                type: Animal

      # Filter out operations you don't want to expose
      - filterSchema:
          filters:
            - Query.!getInventory   # Exclude getInventory
            - Mutation.!deletePet   # Exclude deletePet

      # Add prefix to avoid name collisions
      - prefix:
          value: PetStore_
          includeRootOperations: true
```

### Combining Multiple Sources

```yaml
# .meshrc.yaml
sources:
  - name: Users
    handler:
      openapi:
        source: ./users-api.yaml

  - name: Products
    handler:
      graphql:
        endpoint: http://products-service:4002/graphql

  - name: Orders
    handler:
      grpc:
        endpoint: orders-service:50051
        protoFilePath: ./orders.proto

# Define cross-source relationships
additionalTypeDefs: |
  extend type User {
    orders: [Order!]!
  }

additionalResolvers:
  - targetTypeName: User
    targetFieldName: orders
    requiredSelectionSet: '{ id }'
    sourceName: Orders
    sourceTypeName: Query
    sourceFieldName: ordersByUserId
    sourceArgs:
      userId: '{root.id}'
```

---

## 6. Incremental Migration

The safest migration path runs REST and GraphQL side by side, migrating one endpoint at a time.

### Phase-by-Phase Migration

```
Phase 1: GraphQL Gateway (Wrap)
─────────────────────────────────
┌──────────┐     ┌───────────────┐     ┌──────────────┐
│  Client   │ ──▶ │  GraphQL GW   │ ──▶ │  REST APIs   │
│  (new)    │     │  (wraps REST) │     │  (unchanged) │
└──────────┘     └───────────────┘     └──────────────┘
┌──────────┐                           ┌──────────────┐
│  Client   │ ─────────────────────── ▶ │  REST APIs   │
│  (old)    │                           │  (unchanged) │
└──────────┘                           └──────────────┘

Phase 2: Gradual Replacement
─────────────────────────────────
┌──────────┐     ┌───────────────┐     ┌──────────────┐
│  Client   │ ──▶ │  GraphQL GW   │ ──▶ │  REST APIs   │
│  (all)    │     │               │     │  (shrinking) │
└──────────┘     │  New resolvers │     └──────────────┘
                 │  (growing)     │──▶  Database
                 └───────────────┘

Phase 3: REST Decommissioned
─────────────────────────────────
┌──────────┐     ┌───────────────┐
│  Client   │ ──▶ │  GraphQL      │──▶  Database
│  (all)    │     │  Server       │
└──────────┘     └───────────────┘
```

### Router-Level Migration

Use a reverse proxy to gradually route traffic from REST to GraphQL:

```nginx
# nginx.conf -- route by endpoint
server {
    listen 80;

    # Migrated endpoints go to GraphQL
    location /api/users {
        proxy_pass http://graphql-gateway:4000/graphql;
        # Rewrite REST calls to GraphQL queries (via a middleware)
    }

    # Not-yet-migrated endpoints stay on REST
    location /api/ {
        proxy_pass http://rest-api:3000;
    }
}
```

### REST-to-GraphQL Proxy Middleware

```typescript
// Middleware that translates REST requests to GraphQL queries
import express from 'express';
import { graphqlHTTP } from 'express-graphql';

const app = express();

// Translate GET /api/users/:id → GraphQL query
app.get('/api/users/:id', async (req, res) => {
  const query = `
    query GetUser($id: ID!) {
      user(id: $id) {
        id
        name
        email
        createdAt
      }
    }
  `;

  const response = await fetch('http://localhost:4000/graphql', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: req.headers.authorization,
    },
    body: JSON.stringify({
      query,
      variables: { id: req.params.id },
    }),
  });

  const { data, errors } = await response.json();

  if (errors) {
    return res.status(400).json({ errors });
  }

  // Transform back to REST-style response
  if (!data.user) {
    return res.status(404).json({ error: 'User not found' });
  }

  res.json(data.user);
});
```

---

## 7. BFF Pattern

The Backend for Frontend (BFF) pattern uses GraphQL as an aggregation layer that sits between the frontend and multiple backend services.

### Architecture

```
┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│  Web App    │  │  Mobile App │  │  Admin App  │
└──────┬──────┘  └──────┬──────┘  └──────┬──────┘
       │                │                │
       ▼                ▼                ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│  Web BFF     │ │  Mobile BFF  │ │  Admin BFF   │
│  (GraphQL)   │ │  (GraphQL)   │ │  (GraphQL)   │
│              │ │              │ │              │
│  Optimized   │ │  Optimized   │ │  Full access │
│  for web     │ │  for mobile  │ │  to all data │
└──────┬───────┘ └──────┬───────┘ └──────┬───────┘
       │                │                │
       └────────────────┼────────────────┘
                        │
              ┌─────────┼─────────┐
              ▼         ▼         ▼
         ┌────────┐ ┌────────┐ ┌────────┐
         │ Users  │ │Products│ │ Orders │
         │  API   │ │  API   │ │  API   │
         └────────┘ └────────┘ └────────┘
```

### Why BFF with GraphQL?

1. **Client-specific optimization**: Mobile gets minimal fields, web gets full objects
2. **API aggregation**: One GraphQL query replaces 3-4 REST calls
3. **Type safety**: Schema acts as a contract between frontend and backend
4. **Decoupling**: Frontend changes don't require backend changes

### Implementation

```typescript
// Mobile BFF -- optimized for bandwidth
const mobileTypeDefs = `#graphql
  type Query {
    feed(first: Int = 20): FeedConnection!
    profile: UserProfile!
  }

  # Minimal types for mobile
  type UserProfile {
    id: ID!
    name: String!
    avatarUrl: String
    unreadCount: Int!
  }

  type FeedItem {
    id: ID!
    title: String!
    thumbnailUrl: String
    # No full body -- mobile loads it on tap
  }
`;

const mobileResolvers = {
  Query: {
    feed: async (_, { first }, { dataSources }) => {
      // Aggregate from multiple services
      const posts = await dataSources.postsAPI.getRecent(first);
      const enriched = await Promise.all(
        posts.map(async (post) => ({
          ...post,
          thumbnailUrl: await dataSources.mediaAPI.getThumbnail(post.imageId),
        }))
      );
      return enriched;
    },
    profile: async (_, __, { dataSources, userId }) => {
      // Parallel requests to multiple services
      const [user, notifications] = await Promise.all([
        dataSources.usersAPI.getUser(userId),
        dataSources.notificationsAPI.getUnreadCount(userId),
      ]);
      return { ...user, unreadCount: notifications.count };
    },
  },
};
```

### Single BFF with Client-Aware Resolvers

Instead of separate BFFs, use a single GraphQL server with client-aware behavior:

```typescript
const resolvers = {
  Query: {
    products: async (_, args, { dataSources, clientType }) => {
      const products = await dataSources.productsAPI.list(args);

      // Mobile gets compressed images, web gets full resolution
      if (clientType === 'mobile') {
        return products.map((p) => ({
          ...p,
          imageUrl: p.imageUrl.replace('/full/', '/thumb/'),
        }));
      }

      return products;
    },
  },
};

// Detect client type from headers
const context = async ({ req }) => ({
  clientType: req.headers['x-client-type'] || 'web',
  dataSources: { /* ... */ },
});
```

---

## 8. Practice Problems

### Problem 1: REST Wrapper
Given the following REST API endpoints, design a GraphQL schema and write resolvers that wrap each endpoint. Handle field name mapping (snake_case to camelCase) and error responses:

```
GET    /api/v1/users                → List all users
GET    /api/v1/users/:id            → Get user by ID
POST   /api/v1/users                → Create user
GET    /api/v1/users/:id/orders     → Get user's orders
GET    /api/v1/products?category=X  → Search products
```

### Problem 2: RESTDataSource
Implement a `ProductsAPI` class extending RESTDataSource that wraps a product catalog REST API. Include: (a) response caching with 5-minute TTL, (b) request deduplication, (c) header forwarding for authentication, (d) error handling that maps HTTP status codes to GraphQL error codes.

### Problem 3: GraphQL Mesh Configuration
Write a `.meshrc.yaml` configuration that combines: (a) a PetStore OpenAPI spec, (b) a weather REST API, (c) an existing GraphQL inventory service. Add transforms to rename types, filter operations, and create a cross-source relationship (pets should have a `weather` field based on their location).

### Problem 4: Migration Plan
You have a REST API with 25 endpoints serving 3 frontend applications (web, iOS, Android). Design an incremental migration plan that: (a) identifies which endpoints to migrate first (highest-value, lowest-risk), (b) sets up the GraphQL gateway as a REST wrapper, (c) defines 4 migration phases with success criteria for each.

### Problem 5: BFF Design
Design a BFF architecture for a streaming platform with three clients: web (full feature), mobile (offline-capable), and smart TV (minimal UI). For the "browse catalog" feature, show how the same backend data gets different GraphQL schemas and resolver logic per client. Include query examples showing the different response shapes.

---

## References

- [Apollo RESTDataSource](https://www.apollographql.com/docs/apollo-server/data/fetching-rest/)
- [GraphQL Mesh Documentation](https://the-guild.dev/graphql/mesh/docs)
- [Schema Stitching with @graphql-tools](https://the-guild.dev/graphql/stitching)
- [Migrating from REST to GraphQL (Apollo Guide)](https://www.apollographql.com/docs/technotes/TN0032-sdui-overview/)
- [Backend for Frontend Pattern](https://samnewman.io/patterns/architectural/bff/)
- [Netflix GraphQL Federation](https://netflixtechblog.com/how-netflix-scales-its-api-with-graphql-federation-part-1-ae3557c187e2)

---

**Previous**: [Performance and Security](./14_Performance_Security.md) | **Next**: [Project: API Gateway](./16_Project_API_Gateway.md)
