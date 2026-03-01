# 12. Federation

**Previous**: [Persisted Queries and Caching](./11_Persisted_Queries_Caching.md) | **Next**: [Testing](./13_Testing.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Compare monolithic and federated GraphQL architectures and determine when federation is appropriate
2. Build subgraphs using Apollo Federation 2 directives (@key, @external, @requires, @provides, @shareable)
3. Implement entity resolution with `__resolveReference` to connect data across subgraphs
4. Design subgraph boundaries around business domains rather than team structures
5. Compose a supergraph using Rover CLI and deploy it with Apollo Router

## Table of Contents

1. [Monolithic vs. Federated GraphQL](#1-monolithic-vs-federated-graphql)
2. [Apollo Federation 2 Architecture](#2-apollo-federation-2-architecture)
3. [Federation Directives](#3-federation-directives)
4. [Entity Resolution](#4-entity-resolution)
5. [Designing Subgraph Boundaries](#5-designing-subgraph-boundaries)
6. [Schema Composition](#6-schema-composition)
7. [Apollo Router and Query Planning](#7-apollo-router-and-query-planning)
8. [Migration: Monolith to Federation](#8-migration-monolith-to-federation)
9. [Practice Problems](#9-practice-problems)

**Difficulty**: ⭐⭐⭐⭐

---

As organizations grow, a single GraphQL server becomes a bottleneck -- not because of performance, but because of team coordination. When 10 teams all commit to the same schema file, merge conflicts and deployment coupling slow everyone down. Apollo Federation solves this by letting each team own their part of the graph while presenting a single, unified API to clients. This lesson covers Federation 2 from first principles through production deployment.

---

## 1. Monolithic vs. Federated GraphQL

### Monolithic Architecture

```
┌────────────────────────────────────────────┐
│              Monolith GraphQL               │
│                                              │
│  Schema:  User, Product, Order, Review...    │
│  Resolvers: all in one codebase              │
│  Team: everyone commits here                 │
│                                              │
│      ┌──────┐  ┌──────┐  ┌──────┐          │
│      │ DB 1 │  │ DB 2 │  │ DB 3 │          │
│      └──────┘  └──────┘  └──────┘          │
└────────────────────────────────────────────┘
```

**Works well when**: Small team (< 5 developers), single domain, simple deployment pipeline.

### Federated Architecture

```
                    ┌──────────────┐
                    │  Apollo       │
      Clients ────▶ │  Router      │
                    │  (Gateway)   │
                    └──────┬───────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
       ┌──────────┐ ┌──────────┐ ┌──────────┐
       │ Users    │ │ Products │ │ Orders   │
       │ Subgraph │ │ Subgraph │ │ Subgraph │
       │  Team A  │ │  Team B  │ │  Team C  │
       └────┬─────┘ └────┬─────┘ └────┬─────┘
            │            │            │
            ▼            ▼            ▼
       ┌────────┐  ┌────────┐  ┌────────┐
       │ UserDB │  │ ProdDB │  │OrderDB │
       └────────┘  └────────┘  └────────┘
```

**Choose federation when**: Multiple teams, independent deployment needed, different data stores per domain, schema is large (> 100 types).

### Decision Matrix

| Factor | Monolith | Federation |
|--------|----------|------------|
| Team size | 1-5 developers | 5+ developers across teams |
| Schema size | < 100 types | 100+ types |
| Deployment | Single unit | Independent per subgraph |
| Data sources | Few, shared | Many, domain-specific |
| Operational complexity | Low | Higher (router, composition) |
| Type ownership | Shared | Clear per-team |

---

## 2. Apollo Federation 2 Architecture

Federation 2 introduces the concept of a **supergraph**: a unified schema composed from multiple **subgraphs**, each running as an independent GraphQL service.

### Core Components

```
┌─────────────────────────────────────────────────────────┐
│                      Supergraph                          │
│                                                          │
│  ┌─────────────────────────────────────────────────┐    │
│  │                Apollo Router                     │    │
│  │  - Receives client queries                      │    │
│  │  - Plans execution across subgraphs             │    │
│  │  - Assembles final response                     │    │
│  └──────────────────────┬──────────────────────────┘    │
│                         │                                │
│          ┌──────────────┼──────────────┐                │
│          ▼              ▼              ▼                │
│   ┌────────────┐ ┌────────────┐ ┌────────────┐        │
│   │ Subgraph A │ │ Subgraph B │ │ Subgraph C │        │
│   │            │ │            │ │            │        │
│   │ Owns:      │ │ Owns:      │ │ Owns:      │        │
│   │  User      │ │  Product   │ │  Order     │        │
│   │  Account   │ │  Category  │ │  Payment   │        │
│   └────────────┘ └────────────┘ └────────────┘        │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### Subgraph Setup

Each subgraph is a standalone Apollo Server that uses the `@apollo/subgraph` package:

```typescript
// users-subgraph/index.ts
import { ApolloServer } from '@apollo/server';
import { startStandaloneServer } from '@apollo/server/standalone';
import { buildSubgraphSchema } from '@apollo/subgraph';
import gql from 'graphql-tag';

const typeDefs = gql`
  extend schema
    @link(url: "https://specs.apollo.dev/federation/v2.0",
          import: ["@key", "@shareable"])

  type Query {
    me: User
    user(id: ID!): User
  }

  type User @key(fields: "id") {
    id: ID!
    name: String!
    email: String!
  }
`;

const resolvers = {
  Query: {
    me: (_, __, { userId }) => users.findById(userId),
    user: (_, { id }) => users.findById(id),
  },
  User: {
    __resolveReference: (ref) => users.findById(ref.id),
  },
};

const server = new ApolloServer({
  schema: buildSubgraphSchema({ typeDefs, resolvers }),
});

startStandaloneServer(server, { listen: { port: 4001 } });
```

---

## 3. Federation Directives

Federation 2 provides directives that define how types and fields are shared across subgraphs.

### @key -- Entity Identifier

Marks a type as an entity and specifies its primary key. Entities can be referenced and extended by other subgraphs.

```graphql
# Users subgraph -- defines the User entity
type User @key(fields: "id") {
  id: ID!
  name: String!
  email: String!
}

# Compound keys
type ProductVariant @key(fields: "productId sku") {
  productId: ID!
  sku: String!
  price: Float!
}

# Multiple keys (any can be used for resolution)
type Product @key(fields: "id") @key(fields: "slug") {
  id: ID!
  slug: String!
  name: String!
}
```

### @external -- Referencing Foreign Fields

Declares that a field is defined in another subgraph. The current subgraph doesn't resolve it but can reference it in `@requires`.

```graphql
# Orders subgraph -- extends User from users subgraph
type User @key(fields: "id") {
  id: ID!
  orders: [Order!]!
}

type Order @key(fields: "id") {
  id: ID!
  total: Float!
  user: User!
}
```

### @requires -- Dependent Fields

Specifies that a field requires data from another subgraph to resolve.

```graphql
# Shipping subgraph
type Product @key(fields: "id") {
  id: ID!
  weight: Float! @external    # Defined in products subgraph
  size: String! @external     # Defined in products subgraph
  shippingCost: Float! @requires(fields: "weight size")
}
```

The router fetches `weight` and `size` from the products subgraph first, then passes them to the shipping subgraph's resolver:

```typescript
const resolvers = {
  Product: {
    shippingCost: (product) => {
      // product.weight and product.size are provided by the router
      return calculateShipping(product.weight, product.size);
    },
  },
};
```

### @provides -- Optimistic Resolution

Declares that a resolver can provide fields of a child entity, avoiding an extra subgraph call.

```graphql
# Reviews subgraph
type Review @key(fields: "id") {
  id: ID!
  body: String!
  author: User! @provides(fields: "name")
}

type User @key(fields: "id") {
  id: ID!
  name: String! @external
}
```

When the review already has the author's name (e.g., denormalized in the review record), the router can skip calling the users subgraph.

### @shareable -- Shared Fields

Allows multiple subgraphs to resolve the same field. Without `@shareable`, composition fails if two subgraphs define the same field.

```graphql
# Both subgraphs can resolve Product.name
# Products subgraph
type Product @key(fields: "id") {
  id: ID!
  name: String! @shareable
  description: String!
}

# Inventory subgraph
type Product @key(fields: "id") {
  id: ID!
  name: String! @shareable
  inStock: Boolean!
}
```

### Directive Summary

| Directive | Purpose | Used On |
|-----------|---------|---------|
| `@key` | Define entity primary key | Type |
| `@external` | Reference field from another subgraph | Field |
| `@requires` | Declare dependency on external fields | Field |
| `@provides` | Declare ability to resolve child entity fields | Field |
| `@shareable` | Allow multiple subgraphs to resolve a field | Field |
| `@override` | Move field ownership between subgraphs | Field |
| `@inaccessible` | Hide field from the public API | Field/Type |

---

## 4. Entity Resolution

Entity resolution is the mechanism that lets the router "stitch" data from multiple subgraphs into a single response.

### The __resolveReference Function

Every entity type must implement `__resolveReference`. The router calls this function with the entity's key fields to fetch the full object.

```typescript
// Users subgraph
const resolvers = {
  User: {
    __resolveReference: async (reference, { dataSources }) => {
      // reference = { __typename: "User", id: "42" }
      // The router provides the key fields from @key
      return dataSources.users.findById(reference.id);
    },
  },
};
```

### How the Router Resolves a Cross-Subgraph Query

Consider this client query:

```graphql
query {
  order(id: "100") {
    id
    total
    user {       # User entity lives in users subgraph
      name
      email
    }
  }
}
```

The router creates a query plan:

```
Step 1: Query orders subgraph
  → { order(id: "100") { id total user { __typename id } } }
  → Returns: { id: "100", total: 59.99, user: { __typename: "User", id: "42" } }

Step 2: Query users subgraph with entity reference
  → _entities(representations: [{ __typename: "User", id: "42" }]) { ... on User { name email } }
  → Returns: { name: "Alice", email: "alice@example.com" }

Step 3: Merge results
  → { order: { id: "100", total: 59.99, user: { name: "Alice", email: "alice@example.com" } } }
```

### Batch Entity Resolution

The `_entities` query receives an array of representations, allowing the subgraph to batch-load entities:

```typescript
const resolvers = {
  User: {
    __resolveReference: async (refs, { dataSources }) => {
      // With DataLoader, multiple references are batched automatically
      return dataSources.users.findById(refs.id);
    },
  },
};
```

---

## 5. Designing Subgraph Boundaries

The most important decision in federation is how to split your schema into subgraphs. The right boundaries reduce coupling and let teams work independently.

### Principle: Domain-Driven, Not Team-Driven

```
✗ Bad: Team-based boundaries
  "Frontend team" subgraph, "Backend team" subgraph
  → Teams change, types get split arbitrarily

✓ Good: Domain-based boundaries
  Users subgraph, Products subgraph, Orders subgraph
  → Aligned with business capabilities
```

### Boundary Guidelines

1. **Group by business capability**: Users/Auth, Products/Catalog, Orders/Payments, Shipping/Fulfillment
2. **Minimize cross-subgraph dependencies**: If two types always appear together, keep them in the same subgraph
3. **Consider data ownership**: The team that owns the database should own the subgraph
4. **Avoid circular dependencies**: Subgraph A should not depend on B which depends on A

### Example: E-Commerce Boundaries

```
Users Subgraph          Products Subgraph        Orders Subgraph
─────────────          ──────────────────        ────────────────
User @key(id)          Product @key(id)          Order @key(id)
Account                Category                  OrderItem
Address                Review                    Payment
                       ProductVariant            Shipment

                       # Extends User for reviews
                       User @key(id) {           # Extends User for orders
                         reviews: [Review!]!      User @key(id) {
                       }                            orders: [Order!]!
                                                  }
```

---

## 6. Schema Composition

Composition is the process of combining multiple subgraph schemas into a single supergraph schema.

### Rover CLI

Rover is Apollo's official CLI for managing federated schemas.

```bash
# Install Rover
npm install -g @apollo/rover

# Validate a subgraph schema
rover subgraph check my-graph@production \
  --name users \
  --schema ./users/schema.graphql

# Compose a supergraph locally
rover supergraph compose --config ./supergraph-config.yaml \
  --output supergraph.graphql
```

### Supergraph Configuration

```yaml
# supergraph-config.yaml
federation_version: =2.0.0
subgraphs:
  users:
    routing_url: http://users-service:4001/graphql
    schema:
      file: ./users/schema.graphql
  products:
    routing_url: http://products-service:4002/graphql
    schema:
      file: ./products/schema.graphql
  orders:
    routing_url: http://orders-service:4003/graphql
    schema:
      file: ./orders/schema.graphql
```

### Schema Checks in CI

```yaml
# .github/workflows/schema-check.yml
name: Schema Check
on:
  pull_request:
    paths:
      - 'subgraphs/**/*.graphql'

jobs:
  check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Install Rover
        run: |
          curl -sSL https://rover.apollo.dev/nix/latest | sh
          echo "$HOME/.rover/bin" >> $GITHUB_PATH
      - name: Check schema
        run: |
          rover subgraph check my-graph@production \
            --name ${{ matrix.subgraph }} \
            --schema ./subgraphs/${{ matrix.subgraph }}/schema.graphql
        env:
          APOLLO_KEY: ${{ secrets.APOLLO_KEY }}
    strategy:
      matrix:
        subgraph: [users, products, orders]
```

### Composition Errors

Common composition failures and how to fix them:

```
Error: Field "Product.name" is defined in multiple subgraphs
  but is not marked as @shareable
Fix: Add @shareable to the field in both subgraphs

Error: Entity "User" has no @key directive in subgraph "orders"
Fix: Add @key(fields: "id") to the User type stub in orders

Error: Field "Product.weight" is @external but not used in @requires
Fix: Either add a @requires that uses it or remove the @external annotation
```

---

## 7. Apollo Router and Query Planning

The Apollo Router is the production gateway that replaces the older `@apollo/gateway` Node.js package. Written in Rust, it handles query planning and execution with significantly lower latency and resource usage.

### Router Configuration

```yaml
# router.yaml
supergraph:
  path: ./supergraph.graphql
  # Or fetch from Apollo GraphOS:
  # apollo_graph_ref: my-graph@production

cors:
  origins:
    - https://myapp.com
  allow_headers:
    - Content-Type
    - Authorization

headers:
  all:
    request:
      - propagate:
          named: Authorization

traffic_shaping:
  router:
    timeout: 30s
  all:
    timeout: 15s

telemetry:
  exporters:
    tracing:
      jaeger:
        endpoint: http://jaeger:14268/api/traces
```

### Running the Router

```bash
# Download Apollo Router
curl -sSL https://router.apollo.dev/download/nix/latest | sh

# Run with local supergraph
./router --config router.yaml --supergraph supergraph.graphql

# Or with Apollo GraphOS
APOLLO_KEY=your-api-key APOLLO_GRAPH_REF=my-graph@production ./router
```

### Query Planning

The router analyzes each incoming query and creates an execution plan that minimizes subgraph calls.

```graphql
# Client query
query GetOrderWithDetails {
  order(id: "100") {
    id
    total
    items {
      product {
        name
        category { name }
        reviews { body author { name } }
      }
    }
    user { name email }
  }
}
```

```
Query Plan:
─────────────
Fetch(orders) ──▶ { order(id: "100") { id total items { product { __typename id } } user { __typename id } } }
    │
    ├─▶ Fetch(products) ──▶ _entities[Product] { name category { name } reviews { body author { __typename id } } }
    │       │
    │       └─▶ Fetch(users) ──▶ _entities[User] { name }  (review authors)
    │
    └─▶ Fetch(users) ──▶ _entities[User] { name email }   (order user)
```

The router parallelizes independent fetches and batches entity references.

---

## 8. Migration: Monolith to Federation

### Step 1: Identify Domain Boundaries

Analyze your monolith schema and group types by domain:

```graphql
# Monolith schema -- identify natural groupings
type Query {
  user(id: ID!): User          # → Users subgraph
  products: [Product!]!        # → Products subgraph
  order(id: ID!): Order        # → Orders subgraph
}
```

### Step 2: Extract One Subgraph at a Time

Start with the least-coupled domain:

```typescript
// Phase 1: Monolith becomes the "default" subgraph
// It serves everything, but its schema is now federation-aware

// Phase 2: Extract users subgraph
// - Move User type and related resolvers to users service
// - Monolith keeps a stub: type User @key(fields: "id") { id: ID! }
// - Monolith resolvers that return User now return just { id }

// Phase 3: Extract products subgraph
// Same process, one domain at a time
```

### Step 3: Incremental Rollout

```
Phase 1:  [─── Monolith (federated) ───]                  ← Start here
Phase 2:  [Users] [─── Monolith ───────]                  ← Extract users
Phase 3:  [Users] [Products] [─ Monolith ─]              ← Extract products
Phase 4:  [Users] [Products] [Orders]                     ← Fully federated
```

Key principles:
- Never do a "big bang" migration
- Run the router in front of the monolith from day one
- Extract one subgraph at a time, validate, then proceed
- Use `@override` to gradually move field ownership

```graphql
# Temporarily override a field to the new subgraph
# Old monolith still has the field, but router prefers the new subgraph
type User @key(fields: "id") {
  id: ID!
  name: String! @override(from: "monolith")
}
```

---

## 9. Practice Problems

### Problem 1: Subgraph Design
Given the following types for a social media platform, design 3 subgraphs with clear domain boundaries. Identify which types are entities, which fields need `@external`, and where `@requires` is needed:

```
User, Profile, Post, Comment, Like, Follow,
Notification, Message, Group, Event, Photo, Video
```

### Problem 2: Entity Resolution
Implement a `Reviews` subgraph that extends a `Product` entity from a `Products` subgraph. The Reviews subgraph should add a `reviews` field and an `averageRating` field to Product. Write the full subgraph code including typeDefs, resolvers, and `__resolveReference`.

### Problem 3: Composition Debugging
The following two subgraph schemas fail composition. Identify all errors and fix them:

```graphql
# Subgraph A
type Product @key(fields: "id") {
  id: ID!
  name: String!
  price: Float!
}

# Subgraph B
type Product @key(fields: "id") {
  id: ID!
  name: String!
  inventory: Int!
}
```

### Problem 4: Migration Plan
You have a monolithic GraphQL server with 50 types and 200 resolvers. Three teams (Users, Commerce, Content) need to work independently. Write a phased migration plan: which subgraph to extract first, what stubs to leave in the monolith, and how to verify each phase using Rover schema checks.

### Problem 5: Query Plan Analysis
Given a supergraph with `Users`, `Products`, and `Orders` subgraphs, write the query plan (which subgraph is called in what order, with what query) for the following client query:

```graphql
query {
  me {
    name
    orders {
      id
      items { product { name price reviews { body } } }
    }
  }
}
```

---

## References

- [Apollo Federation 2 Documentation](https://www.apollographql.com/docs/federation/)
- [Apollo Router Configuration](https://www.apollographql.com/docs/router/configuration/overview)
- [Rover CLI Reference](https://www.apollographql.com/docs/rover/)
- [Federation Specification](https://www.apollographql.com/docs/federation/federation-spec/)
- [Principled GraphQL - Federation](https://principledgraphql.com/agility#4-abstract-demand-oriented-schema)
- [Netflix DGS Federation](https://netflix.github.io/dgs/federation/)

---

**Previous**: [Persisted Queries and Caching](./11_Persisted_Queries_Caching.md) | **Next**: [Testing](./13_Testing.md)
