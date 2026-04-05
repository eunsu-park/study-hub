# 22. GraphQL Federation

**Previous**: [GraphQL Server Implementation](./21_GraphQL_Server_Implementation.md) | **Next**: [GraphQL Performance and Security](./23_GraphQL_Performance_Security.md)

**Difficulty**: ⭐⭐⭐⭐

---

## Learning Objectives

- Explain the concept of schema federation and why monolithic schemas become problematic
- Design subgraph schemas with entity types, keys, and cross-service references
- Implement entity resolution using the `__resolveReference` pattern
- Configure an Apollo Federation 2 router to compose subgraph schemas
- Apply federation directives (`@key`, `@shareable`, `@external`, `@requires`, `@provides`)
- Plan a migration from a monolithic GraphQL schema to a federated architecture

---

## Table of Contents

1. [Why Federation?](#1-why-federation)
2. [Federation Architecture](#2-federation-architecture)
3. [Subgraph Design](#3-subgraph-design)
4. [Entity Resolution](#4-entity-resolution)
5. [Federation Directives](#5-federation-directives)
6. [Apollo Federation 2](#6-apollo-federation-2)
7. [Implementing Subgraphs in Python](#7-implementing-subgraphs-in-python)
8. [Router Configuration](#8-router-configuration)
9. [Testing Federated Schemas](#9-testing-federated-schemas)
10. [Migration from Monolith](#10-migration-from-monolith)
11. [Exercises](#11-exercises)
12. [References](#12-references)

---

## 1. Why Federation?

As organizations grow, a single monolithic GraphQL schema becomes a bottleneck.

### Problems with Monolithic Schemas

| Problem | Impact |
|---------|--------|
| **Single codebase** | All teams modify the same schema file |
| **Deployment coupling** | A change in one type requires full redeployment |
| **Ownership confusion** | No clear owner for shared types like `User` |
| **Scaling limits** | One server handles all resolvers |
| **Development velocity** | Merge conflicts, coordination overhead |

### What Federation Solves

Federation allows multiple teams to own and deploy independent GraphQL services (subgraphs) that compose into a single unified schema at the gateway (router).

```
Before (Monolith):
  Client → Single GraphQL Server (all types, all resolvers)

After (Federation):
  Client → Router → Users Subgraph (User type)
                   → Posts Subgraph (Post type)
                   → Comments Subgraph (Comment type)
                   → Search Subgraph (search queries)
```

### Benefits

- **Independent deployments**: Each subgraph deploys independently
- **Team ownership**: Clear ownership boundaries per subgraph
- **Incremental adoption**: Migrate one domain at a time
- **Heterogeneous implementations**: Subgraphs can use different languages
- **Focused scaling**: Scale hot subgraphs independently

---

## 2. Federation Architecture

### Components

```
┌──────────┐     ┌──────────────────────────────────────────┐
│  Client   │────▶│              Router (Gateway)             │
└──────────┘     │   - Schema composition                    │
                 │   - Query planning                        │
                 │   - Result merging                        │
                 └──┬──────────┬──────────┬────────────────┘
                    │          │          │
              ┌─────▼──┐ ┌────▼───┐ ┌────▼───┐
              │ Users   │ │ Posts  │ │Comments│
              │Subgraph │ │Subgraph│ │Subgraph│
              │         │ │        │ │        │
              │ User    │ │ Post   │ │Comment │
              │ Profile │ │ Tag    │ │        │
              └────┬────┘ └────┬───┘ └────┬───┘
                   │          │          │
              ┌────▼────┐ ┌───▼────┐ ┌───▼────┐
              │Users DB │ │Posts DB│ │Cmts DB │
              └─────────┘ └────────┘ └────────┘
```

### Query Execution Flow

```graphql
# Client sends this query to the Router:
query {
  user(id: "1") {        # Resolved by Users subgraph
    username
    posts(first: 5) {    # Resolved by Posts subgraph
      title
      comments {          # Resolved by Comments subgraph
        body
        author {          # Resolved by Users subgraph (entity reference)
          username
        }
      }
    }
  }
}
```

The router creates a **query plan**:

```
1. Fetch user from Users subgraph
2. Fetch posts for user from Posts subgraph
3. Fetch comments for each post from Comments subgraph
4. Resolve comment authors from Users subgraph (entity references)
5. Merge results into single response
```

---

## 3. Subgraph Design

### Defining Entities

An **entity** is a type that can be resolved across subgraphs. It uses the `@key` directive:

```graphql
# Users subgraph
type User @key(fields: "id") {
  id: ID!
  username: String!
  email: String!
  bio: String
  createdAt: DateTime!
}

type Query {
  user(id: ID!): User
  users(limit: Int): [User!]!
}
```

```graphql
# Posts subgraph
# Extend User from Users subgraph
type User @key(fields: "id") {
  id: ID!
  posts(first: Int = 10): [Post!]!   # Added by Posts subgraph
}

type Post @key(fields: "id") {
  id: ID!
  title: String!
  content: String!
  authorId: ID!
  author: User!
  createdAt: DateTime!
}

type Query {
  post(id: ID!): Post
  posts(first: Int, after: String): PostConnection!
}
```

```graphql
# Comments subgraph
type User @key(fields: "id") {
  id: ID!
}

type Post @key(fields: "id") {
  id: ID!
  comments(first: Int = 10): [Comment!]!  # Added by Comments subgraph
}

type Comment @key(fields: "id") {
  id: ID!
  body: String!
  authorId: ID!
  author: User!
  post: Post!
  createdAt: DateTime!
}
```

### Ownership Rules

| Rule | Description |
|------|-------------|
| **One owner** | Each type field has exactly one owning subgraph |
| **Entity extension** | Other subgraphs can add fields to entities |
| **Stub types** | Non-owning subgraphs declare only the `@key` fields |
| **Value types** | Shared types (enums, scalars) can be `@shareable` |

### Domain Boundary Guidelines

```
Users Subgraph:
  Owns: User, Profile, Address
  Extends: (none — origin subgraph)

Posts Subgraph:
  Owns: Post, Tag, PostConnection
  Extends: User (adds posts field)

Comments Subgraph:
  Owns: Comment
  Extends: User (adds comments field), Post (adds comments field)

Orders Subgraph:
  Owns: Order, OrderItem, Payment
  Extends: User (adds orders field), Product (adds orders field)
```

---

## 4. Entity Resolution

Entity resolution is how the router fetches type data across subgraph boundaries.

### The `__resolveReference` Function

When the router needs to resolve a `User` entity in the Posts subgraph, it calls the Users subgraph's `__resolveReference`:

```python
# Users subgraph — entity resolver
@strawberry.type
class User:
    id: strawberry.ID
    username: str
    email: str

    @classmethod
    async def resolve_reference(cls, info, id: strawberry.ID) -> "User":
        """Called by the router to resolve User entities."""
        user = await info.context.db.users.find_by_id(id)
        if user is None:
            return None
        return cls(id=user.id, username=user.username, email=user.email)
```

### How the Router Uses References

```
1. Comments subgraph returns:
   { comment: { body: "Great!", author: { __typename: "User", id: "42" } } }

2. Router sees User entity reference { __typename: "User", id: "42" }

3. Router calls Users subgraph:
   query {
     _entities(representations: [{ __typename: "User", id: "42" }]) {
       ... on User { username email }
     }
   }

4. Users subgraph resolves the reference and returns:
   { username: "alice", email: "alice@example.com" }

5. Router merges into final response:
   { comment: { body: "Great!", author: { username: "alice", email: "alice@example.com" } } }
```

### Batch Entity Resolution

The router batches entity references:

```graphql
# Instead of 10 separate calls, the router sends:
query {
  _entities(representations: [
    { __typename: "User", id: "1" },
    { __typename: "User", id: "2" },
    { __typename: "User", id: "3" },
    ...
  ]) {
    ... on User { username email }
  }
}
```

### Compound Keys

Entities can use multiple key fields:

```graphql
type ProductVariant @key(fields: "productId sku") {
  productId: ID!
  sku: String!
  name: String!
  price: Float!
  inStock: Boolean!
}
```

---

## 5. Federation Directives

### Apollo Federation 2 Directives

| Directive | Purpose | Example |
|-----------|---------|---------|
| `@key` | Marks entity types with unique key fields | `type User @key(fields: "id")` |
| `@shareable` | Allows multiple subgraphs to resolve a field | `type Position @shareable { x: Int! y: Int! }` |
| `@external` | Marks a field as defined in another subgraph | `id: ID! @external` |
| `@requires` | Declares fields needed from other subgraphs | `shippingCost: Float @requires(fields: "weight size")` |
| `@provides` | Declares additional fields provided when resolving | `author: User @provides(fields: "email")` |
| `@override` | Migrates a field from one subgraph to another | `username: String! @override(from: "users")` |
| `@inaccessible` | Hides a field from the composed schema | `internalNote: String @inaccessible` |
| `@tag` | Labels schema elements for tooling | `type User @tag(name: "internal")` |

### `@requires` Example

```graphql
# Products subgraph
type Product @key(fields: "id") {
  id: ID!
  name: String!
  weight: Float!
  size: String!
}

# Shipping subgraph
type Product @key(fields: "id") {
  id: ID!
  weight: Float! @external
  size: String! @external
  shippingCost: Float! @requires(fields: "weight size")
}
```

The router fetches `weight` and `size` from the Products subgraph before calling the Shipping subgraph's `shippingCost` resolver.

### `@provides` Example

```graphql
# Reviews subgraph — provides author.email when resolving reviews
type Review @key(fields: "id") {
  id: ID!
  body: String!
  author: User @provides(fields: "email")
}

type User @key(fields: "id") {
  id: ID!
  email: String! @external
}
```

### `@shareable` Example

Value types that are identical across subgraphs:

```graphql
# Both subgraphs can define and resolve this type
type Money @shareable {
  amount: Float!
  currency: CurrencyCode!
}

enum CurrencyCode @shareable {
  USD
  EUR
  GBP
  JPY
}
```

---

## 6. Apollo Federation 2

### Supergraph Schema

The composed schema (supergraph) is the combination of all subgraph schemas:

```
Subgraph A schema + Subgraph B schema + ... → Supergraph schema
```

### Composition

```bash
# Install Rover CLI
curl -sSL https://rover.apollo.dev/nix/latest | sh

# Compose supergraph from subgraph schemas
rover supergraph compose --config supergraph.yaml > supergraph.graphql
```

```yaml
# supergraph.yaml
federation_version: =2.7.0
subgraphs:
  users:
    routing_url: http://users-service:4001/graphql
    schema:
      file: ./subgraphs/users/schema.graphql
  posts:
    routing_url: http://posts-service:4002/graphql
    schema:
      file: ./subgraphs/posts/schema.graphql
  comments:
    routing_url: http://comments-service:4003/graphql
    schema:
      file: ./subgraphs/comments/schema.graphql
```

### Composition Validation

The composition step validates:
- No conflicting type definitions
- All entity references are resolvable
- Required fields are accessible
- No circular dependencies in `@requires`

```bash
# Check for composition errors
rover supergraph compose --config supergraph.yaml 2>&1

# Common errors:
# - EXTERNAL_MISSING_ON_BASE: @external field not in origin subgraph
# - KEY_FIELDS_SELECT_INVALID_TYPE: @key on non-existent field
# - REQUIRES_FIELDS_MISSING_EXTERNAL: @requires field not marked @external
```

---

## 7. Implementing Subgraphs in Python

### Strawberry Federation Support

```python
# users_subgraph/schema.py
import strawberry
from strawberry.federation import Schema


@strawberry.federation.type(keys=["id"])
class User:
    id: strawberry.ID
    username: str
    email: str
    bio: str | None = None

    @classmethod
    async def resolve_reference(cls, info, id: strawberry.ID) -> "User":
        user = await info.context.db.users.find_by_id(id)
        return cls(
            id=user.id,
            username=user.username,
            email=user.email,
            bio=user.bio,
        )


@strawberry.type
class Query:
    @strawberry.field
    async def user(self, info, id: strawberry.ID) -> User | None:
        return await info.context.db.users.find_by_id(id)

    @strawberry.field
    async def users(self, info, limit: int = 10) -> list[User]:
        return await info.context.db.users.find_all(limit=limit)


schema = Schema(query=Query, enable_federation_2=True)
```

### Posts Subgraph Extending User

```python
# posts_subgraph/schema.py
import strawberry
from strawberry.federation import Schema


@strawberry.federation.type(keys=["id"])
class User:
    id: strawberry.ID

    @strawberry.field
    async def posts(self, info, first: int = 10) -> list["Post"]:
        return await info.context.db.posts.find_by_author(
            self.id, limit=first
        )


@strawberry.federation.type(keys=["id"])
class Post:
    id: strawberry.ID
    title: str
    content: str
    author_id: strawberry.ID

    @strawberry.field
    async def author(self) -> User:
        return User(id=self.author_id)

    @classmethod
    async def resolve_reference(cls, info, id: strawberry.ID) -> "Post":
        post = await info.context.db.posts.find_by_id(id)
        return cls(
            id=post.id,
            title=post.title,
            content=post.content,
            author_id=post.author_id,
        )


@strawberry.type
class Query:
    @strawberry.field
    async def post(self, info, id: strawberry.ID) -> Post | None:
        return await info.context.db.posts.find_by_id(id)


schema = Schema(query=Query, enable_federation_2=True)
```

---

## 8. Router Configuration

### Apollo Router

```bash
# Download and run
curl -sSL https://router.apollo.dev/download/nix/latest | sh
./router --supergraph supergraph.graphql --config router.yaml
```

```yaml
# router.yaml
supergraph:
  listen: 0.0.0.0:4000

cors:
  origins:
    - https://app.example.com
  methods:
    - GET
    - POST

headers:
  all:
    request:
      - propagate:
          named: Authorization
      - propagate:
          named: X-Request-ID

limits:
  max_depth: 15
  max_height: 200

telemetry:
  instrumentation:
    spans:
      mode: spec_compliant
```

### Docker Compose for Full Stack

```yaml
version: "3.9"
services:
  router:
    image: ghcr.io/apollographql/router:v1.40.0
    ports:
      - "4000:4000"
    volumes:
      - ./supergraph.graphql:/etc/config/supergraph.graphql
      - ./router.yaml:/etc/config/router.yaml
    command: --supergraph /etc/config/supergraph.graphql --config /etc/config/router.yaml
    depends_on:
      - users
      - posts
      - comments

  users:
    build: ./subgraphs/users
    ports:
      - "4001:8000"

  posts:
    build: ./subgraphs/posts
    ports:
      - "4002:8000"

  comments:
    build: ./subgraphs/comments
    ports:
      - "4003:8000"
```

---

## 9. Testing Federated Schemas

### Composition Testing

```bash
# Verify schema composes without errors
rover supergraph compose --config supergraph.yaml

# Check for breaking changes against production
rover subgraph check my-graph@production \
  --name users \
  --schema subgraphs/users/schema.graphql
```

### Subgraph Unit Testing

```python
# tests/test_users_subgraph.py
import pytest
from strawberry.test import GraphQLTestClient
from schema import schema


@pytest.fixture
def client():
    return GraphQLTestClient(schema)


def test_user_query(client):
    result = client.query("""
        query {
            user(id: "1") {
                id
                username
                email
            }
        }
    """)
    assert result.errors is None
    assert result.data["user"]["username"] == "alice"


def test_entity_resolution(client):
    """Test _entities query used by the router."""
    result = client.query("""
        query {
            _entities(representations: [
                { __typename: "User", id: "1" }
            ]) {
                ... on User {
                    username
                    email
                }
            }
        }
    """)
    assert result.errors is None
    assert result.data["_entities"][0]["username"] == "alice"
```

### Integration Testing

```python
# tests/test_federation_integration.py
import httpx
import pytest

ROUTER_URL = "http://localhost:4000/graphql"


@pytest.mark.integration
async def test_cross_subgraph_query():
    """Test a query that spans multiple subgraphs."""
    async with httpx.AsyncClient() as client:
        response = await client.post(ROUTER_URL, json={
            "query": """
                query {
                    user(id: "1") {
                        username
                        posts(first: 3) {
                            title
                            comments(first: 2) {
                                body
                                author { username }
                            }
                        }
                    }
                }
            """
        })
    data = response.json()
    assert "errors" not in data
    assert data["data"]["user"]["username"] is not None
    assert len(data["data"]["user"]["posts"]) <= 3
```

---

## 10. Migration from Monolith

### Phased Migration Strategy

```
Phase 1: Strangler Fig Pattern
  - Deploy router in front of monolith
  - Monolith becomes first subgraph
  - Client queries go through router (transparent)

Phase 2: Extract First Subgraph
  - Choose a bounded context (e.g., Users)
  - Build Users subgraph
  - Migrate User resolvers from monolith
  - Verify via shadow testing

Phase 3: Iterate
  - Extract next subgraph (Posts, Comments, etc.)
  - Use @override to migrate fields gradually
  - Remove from monolith once verified

Phase 4: Decommission Monolith
  - All types extracted to subgraphs
  - Monolith shutdown
```

### Using `@override` for Gradual Migration

```graphql
# Phase 2: Posts subgraph takes over the `posts` field from monolith
type User @key(fields: "id") {
  id: ID!
  posts: [Post!]! @override(from: "monolith")
}
```

### Validation Checklist

| Step | Verification |
|------|-------------|
| Schema composition | `rover supergraph compose` succeeds |
| Entity resolution | `_entities` queries return correct data |
| Performance | Latency within 10% of monolith |
| Error handling | Errors propagate correctly through router |
| Auth | Authentication headers forwarded to subgraphs |
| Monitoring | Metrics visible for each subgraph |

---

## 11. Exercises

### Exercise 1: Design Subgraph Boundaries

Given an e-commerce domain with: Users, Products, Orders, Reviews, Inventory, Payments, and Shipping, design the subgraph boundaries:
- Which types belong to which subgraph?
- Which types are entities?
- Which fields use `@requires` or `@provides`?

### Exercise 2: Implement Two Subgraphs

Build Users and Posts subgraphs with Strawberry federation:
- Users subgraph: User type with `@key(fields: "id")`
- Posts subgraph: Post type with author reference to User
- Test entity resolution for both subgraphs

### Exercise 3: Router Composition

Using Rover CLI:
- Write supergraph.yaml referencing your two subgraphs
- Compose the supergraph schema
- Verify cross-subgraph queries work through the router

### Exercise 4: Migration Plan

You have a monolithic GraphQL server with 50 types and 200 fields. Create a migration plan:
- Identify 4-5 bounded contexts
- Define subgraph boundaries
- Write the `@override` directives for the first migration
- Estimate timeline and risk areas

---

## 12. References

### Official Documentation
- [Apollo Federation 2 Docs](https://www.apollographql.com/docs/federation/)
- [Apollo Router Documentation](https://www.apollographql.com/docs/router/)
- [Rover CLI Documentation](https://www.apollographql.com/docs/rover/)

### Federation Implementations
- [Strawberry Federation](https://strawberry.rocks/docs/guides/federation)
- [Apollo Server Federation](https://www.apollographql.com/docs/apollo-server/using-federation/apollo-subgraph-setup/)
- [GraphQL Mesh — Alternative Gateway](https://the-guild.dev/graphql/mesh)

### Articles
- "Apollo Federation: A Revolution in GraphQL Architecture" — Apollo Blog
- "Federated GraphQL at Netflix" — Netflix Tech Blog
- "Schema Federation at Expedia" — Expedia Engineering

---

**License**: CC BY-NC 4.0
