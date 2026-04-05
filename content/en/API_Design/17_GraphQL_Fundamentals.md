# 17. GraphQL Fundamentals

**Previous**: [API Lifecycle Management](./16_API_Lifecycle_Management.md) | **Next**: [GraphQL Schema Design](./18_GraphQL_Schema_Design.md)

**Difficulty**: ⭐⭐⭐

---

## Learning Objectives

- Explain the motivation behind GraphQL and the problems it solves compared to REST
- Write GraphQL queries, mutations, and fragments using the GraphQL query language
- Define types, fields, and relationships in the Schema Definition Language (SDL)
- Distinguish between GraphQL's type system primitives: scalars, objects, enums, and lists
- Compare REST and GraphQL trade-offs for different API scenarios
- Set up a minimal GraphQL server in Python with Strawberry

---

## Table of Contents

1. [What Is GraphQL?](#1-what-is-graphql)
2. [GraphQL vs. REST](#2-graphql-vs-rest)
3. [Schema Definition Language (SDL)](#3-schema-definition-language-sdl)
4. [The Type System](#4-the-type-system)
5. [Queries](#5-queries)
6. [Mutations](#6-mutations)
7. [Variables and Fragments](#7-variables-and-fragments)
8. [Introspection](#8-introspection)
9. [A Minimal Python Server](#9-a-minimal-python-server)
10. [Exercises](#10-exercises)
11. [References](#11-references)

---

## 1. What Is GraphQL?

GraphQL is a query language for APIs and a runtime for fulfilling those queries with your existing data. Created at Facebook in 2012 and open-sourced in 2015, it provides a complete and understandable description of the data in your API, gives clients the power to ask for exactly what they need, and makes it easier to evolve APIs over time.

### Core Principles

| Principle | Description |
|-----------|-------------|
| **Declarative data fetching** | The client specifies exactly what data it needs |
| **Single endpoint** | All operations go through one URL (typically `/graphql`) |
| **Strongly typed** | Every field, argument, and return value has an explicit type |
| **Hierarchical** | Queries mirror the shape of the response |
| **Introspective** | The schema itself can be queried for documentation and tooling |

### The Problem GraphQL Solves

Traditional REST APIs suffer from two fundamental inefficiencies:

1. **Over-fetching**: A `GET /users/42` returns all fields even when the client only needs `name` and `email`.
2. **Under-fetching**: To display a user profile with their posts and followers, the client must make multiple round trips (`/users/42`, `/users/42/posts`, `/users/42/followers`).

GraphQL solves both by letting the client describe the exact shape of the response in a single request.

```graphql
# One request, exact data
query {
  user(id: "42") {
    name
    email
    posts(first: 5) {
      title
      createdAt
    }
    followers {
      totalCount
    }
  }
}
```

### Brief History

| Year | Milestone |
|------|-----------|
| 2012 | Created internally at Facebook for mobile apps |
| 2015 | Open-sourced with reference JavaScript implementation |
| 2016 | GitHub launches GraphQL API (v4) |
| 2018 | GraphQL Foundation established under the Linux Foundation |
| 2021 | October 2021 spec release (latest stable) |
| 2023 | Widespread adoption: Shopify, Stripe, Yelp, Airbnb, and others |

---

## 2. GraphQL vs. REST

Understanding when to use GraphQL versus REST is essential for API designers.

### Feature Comparison

| Aspect | REST | GraphQL |
|--------|------|---------|
| Endpoints | Multiple (one per resource) | Single (`/graphql`) |
| Data shape | Server-determined | Client-determined |
| Over-fetching | Common | Eliminated by design |
| Under-fetching | Common (requires multiple calls) | Eliminated (nested queries) |
| Versioning | URL or header versioning | Schema evolution (additive changes) |
| Caching | HTTP caching (GET, ETag) | Requires custom caching strategies |
| File upload | Native multipart support | Requires extensions |
| Error handling | HTTP status codes | Always returns 200 with `errors` array |
| Tooling | Mature (Swagger, Postman) | Growing (GraphiQL, Apollo Studio) |
| Learning curve | Low | Moderate |

| Dimension | REST | GraphQL |
|-----------|------|---------|
| Data fetching | Multiple endpoints, fixed structure | Single endpoint, flexible queries |
| Over/under-fetching | Common (fixed responses) | Eliminated (client specifies fields) |
| Type system | Optional (OpenAPI) | Built-in (SDL) |
| Real-time | Polling or WebSocket (separate) | Subscriptions (native) |
| Caching | HTTP caching (simple) | Complex (normalized client cache) |
| Learning curve | Lower | Higher |
| Tooling maturity | Very mature | Rapidly growing |
| Best for | CRUD, public APIs, simple clients | Complex UIs, mobile, aggregation |

### When to Choose GraphQL

- **Mobile applications** with constrained bandwidth that need precise data fetching
- **Micro-frontends** where different UI components need different data slices
- **Rapidly evolving UIs** that change independently of the backend
- **Aggregation layers** that combine data from multiple microservices

### When to Stick with REST

- **Simple CRUD** APIs with predictable access patterns
- **File-heavy** APIs (upload/download)
- **Public APIs** where HTTP caching is critical
- **Webhook/event-driven** integrations
- **Team skill set** is REST-native with no GraphQL experience

### Hybrid Approach

Many production systems use both:

```
Client → API Gateway → REST endpoints (CRUD, file operations)
                     → GraphQL endpoint (complex queries, aggregation)
                     → gRPC (internal microservice calls)
```

---

## 3. Schema Definition Language (SDL)

The Schema Definition Language is GraphQL's way of defining the shape of your API. Think of it as the "contract" between client and server.

### Object Types

Object types are the most basic building blocks:

```graphql
type User {
  id: ID!
  username: String!
  email: String!
  bio: String
  createdAt: DateTime!
}
```

- `!` means non-nullable (the field must always return a value)
- Without `!`, the field is nullable by default

### Scalar Types

GraphQL comes with five built-in scalars:

| Type | Description | Example |
|------|-------------|---------|
| `Int` | Signed 32-bit integer | `42` |
| `Float` | Double-precision floating point | `3.14` |
| `String` | UTF-8 character sequence | `"hello"` |
| `Boolean` | `true` or `false` | `true` |
| `ID` | Unique identifier (serialized as String) | `"abc-123"` |

### Lists and Non-Null Modifiers

```graphql
type Post {
  tags: [String!]!    # Non-null list of non-null strings
  comments: [Comment] # Nullable list of nullable comments
}
```

| Declaration | Null list? | Null elements? |
|-------------|-----------|----------------|
| `[String]` | Yes | Yes |
| `[String!]` | Yes | No |
| `[String]!` | No | Yes |
| `[String!]!` | No | No |

### Relationships

Types reference each other to form a graph:

```graphql
type User {
  id: ID!
  username: String!
  posts: [Post!]!
}

type Post {
  id: ID!
  title: String!
  author: User!
  comments: [Comment!]!
}

type Comment {
  id: ID!
  body: String!
  author: User!
  post: Post!
}
```

### Entry Points: Query and Mutation Types

```graphql
type Query {
  user(id: ID!): User
  users(limit: Int = 10, offset: Int = 0): [User!]!
  post(id: ID!): Post
  searchPosts(query: String!): [Post!]!
}

type Mutation {
  createUser(username: String!, email: String!): User!
  createPost(title: String!, content: String!, authorId: ID!): Post!
  deletePost(id: ID!): Boolean!
}
```

The `Query` type defines read operations; the `Mutation` type defines write operations.

---

## 4. The Type System

GraphQL's type system is the backbone of its contract guarantees.

### Enums

Enums restrict a field to a finite set of values:

```graphql
enum PostStatus {
  DRAFT
  PUBLISHED
  ARCHIVED
}

type Post {
  id: ID!
  title: String!
  status: PostStatus!
}
```

### Input Types

Input types are used for structured mutation arguments:

```graphql
input CreatePostInput {
  title: String!
  content: String!
  status: PostStatus = DRAFT
  tags: [String!]
}

type Mutation {
  createPost(input: CreatePostInput!): Post!
}
```

> **Convention**: Name input types with an `Input` suffix.

### Interfaces

Interfaces define a contract that types must implement:

```graphql
interface Node {
  id: ID!
}

interface Timestamped {
  createdAt: DateTime!
  updatedAt: DateTime!
}

type User implements Node & Timestamped {
  id: ID!
  username: String!
  createdAt: DateTime!
  updatedAt: DateTime!
}

type Post implements Node & Timestamped {
  id: ID!
  title: String!
  createdAt: DateTime!
  updatedAt: DateTime!
}
```

### Union Types

Unions represent a value that could be one of several types:

```graphql
union SearchResult = User | Post | Comment

type Query {
  search(query: String!): [SearchResult!]!
}
```

Clients use inline fragments to handle each case:

```graphql
query {
  search(query: "graphql") {
    ... on User {
      username
    }
    ... on Post {
      title
      author { username }
    }
    ... on Comment {
      body
    }
  }
}
```

### Custom Scalars

You can define domain-specific scalars:

```graphql
scalar DateTime
scalar Email
scalar URL
scalar JSON
```

The server provides parsing and serialization logic for each custom scalar.

---

## 5. Queries

Queries are read operations. They are the most common GraphQL operation.

### Basic Query

```graphql
query GetUser {
  user(id: "42") {
    username
    email
  }
}
```

Response:

```json
{
  "data": {
    "user": {
      "username": "alice",
      "email": "alice@example.com"
    }
  }
}
```

### Nested Queries

GraphQL shines when fetching related data:

```graphql
query UserWithPosts {
  user(id: "42") {
    username
    posts(first: 3) {
      title
      commentCount
      comments(first: 2) {
        body
        author {
          username
        }
      }
    }
  }
}
```

### Aliases

When querying the same field with different arguments, use aliases:

```graphql
query {
  admin: user(id: "1") {
    username
    email
  }
  viewer: user(id: "42") {
    username
    email
  }
}
```

### Arguments

Fields can accept arguments at any level:

```graphql
query {
  users(limit: 10, sortBy: CREATED_AT, order: DESC) {
    username
    posts(status: PUBLISHED, first: 5) {
      title
    }
  }
}
```

### The `__typename` Meta-Field

Every type has a built-in `__typename` field useful for union/interface disambiguation:

```graphql
query {
  search(query: "graphql") {
    __typename
    ... on User { username }
    ... on Post { title }
  }
}
```

---

## 6. Mutations

Mutations are write operations. By convention they should be named as verbs.

### Basic Mutation

```graphql
mutation CreateUser {
  createUser(input: {
    username: "bob"
    email: "bob@example.com"
  }) {
    id
    username
    createdAt
  }
}
```

### Mutation Response Pattern

A best practice is to return a payload type with both the result and potential errors:

```graphql
type CreatePostPayload {
  post: Post
  userErrors: [UserError!]!
}

type UserError {
  field: [String!]
  message: String!
  code: ErrorCode!
}

enum ErrorCode {
  INVALID_INPUT
  NOT_FOUND
  UNAUTHORIZED
  CONFLICT
}

type Mutation {
  createPost(input: CreatePostInput!): CreatePostPayload!
}
```

This approach (popularized by Shopify) keeps errors structured and type-safe rather than relying on the top-level `errors` array.

### Multiple Mutations

Multiple mutations in a single request execute sequentially (unlike queries which may execute in parallel):

```graphql
mutation {
  createPost(input: { title: "First", content: "..." }) {
    post { id }
  }
  updateUser(input: { bio: "Updated bio" }) {
    user { bio }
  }
}
```

### Input Validation

Structure your input types to be self-documenting:

```graphql
input UpdatePostInput {
  title: String                       # Optional (nullable = can skip)
  content: String
  status: PostStatus
  publishAt: DateTime                 # Schedule future publication
}

type Mutation {
  updatePost(id: ID!, input: UpdatePostInput!): UpdatePostPayload!
}
```

---

## 7. Variables and Fragments

### Variables

Variables separate static query structure from dynamic runtime values:

```graphql
query GetUser($userId: ID!, $postLimit: Int = 5) {
  user(id: $userId) {
    username
    posts(first: $postLimit) {
      title
    }
  }
}
```

Variables JSON:

```json
{
  "userId": "42",
  "postLimit": 3
}
```

> **Best Practice**: Always use variables for dynamic values. Never interpolate user input into query strings (prevents injection attacks).

### Fragments

Fragments enable reusable field selections:

```graphql
fragment UserBasic on User {
  id
  username
  email
  avatarUrl
}

fragment PostSummary on Post {
  id
  title
  status
  createdAt
  author {
    ...UserBasic
  }
}

query Dashboard {
  me {
    ...UserBasic
    posts(first: 10) {
      ...PostSummary
    }
  }
  trending: posts(first: 5, sortBy: TRENDING) {
    ...PostSummary
  }
}
```

### Inline Fragments

Used with unions and interfaces:

```graphql
query {
  node(id: "abc") {
    id
    ... on User {
      username
      email
    }
    ... on Post {
      title
      content
    }
  }
}
```

### Directives

Built-in directives control field inclusion:

```graphql
query GetUser($userId: ID!, $includeEmail: Boolean!) {
  user(id: $userId) {
    username
    email @include(if: $includeEmail)
    internalNotes @skip(if: true)
  }
}
```

| Directive | Behavior |
|-----------|----------|
| `@include(if: Boolean!)` | Include field only if `true` |
| `@skip(if: Boolean!)` | Skip field if `true` |
| `@deprecated(reason: String)` | Mark schema field as deprecated |

---

## 8. Introspection

GraphQL schemas are self-documenting. Clients can query the schema itself.

### Schema Introspection Query

```graphql
query IntrospectSchema {
  __schema {
    types {
      name
      kind
      description
    }
    queryType { name }
    mutationType { name }
    subscriptionType { name }
  }
}
```

### Type Introspection

```graphql
query IntrospectType {
  __type(name: "User") {
    name
    kind
    fields {
      name
      type {
        name
        kind
        ofType {
          name
        }
      }
      args {
        name
        type { name }
        defaultValue
      }
    }
  }
}
```

### Uses of Introspection

- **GraphiQL / GraphQL Playground**: Auto-complete and documentation
- **Code generation**: Generate typed clients from the schema
- **Schema validation**: CI pipelines that check for breaking changes

> **Security Note**: Disable introspection in production public APIs to prevent schema leaking. Keep it enabled for internal APIs and development environments.

```python
# Strawberry: disable introspection in production
import strawberry

schema = strawberry.Schema(
    query=Query,
    mutation=Mutation,
    extensions=[],
)

# In production configuration
if not DEBUG:
    # Use middleware to block introspection queries
    pass
```

---

## 9. A Minimal Python Server

Let us build a working GraphQL server using [Strawberry](https://strawberry.rocks/), a modern Python GraphQL library that uses dataclasses and type hints.

### Installation

```bash
pip install strawberry-graphql[fastapi] uvicorn
```

### Schema Definition (Code-First)

```python
# schema.py
import strawberry
from datetime import datetime
from typing import Optional


@strawberry.enum
class PostStatus(str):
    DRAFT = "DRAFT"
    PUBLISHED = "PUBLISHED"
    ARCHIVED = "ARCHIVED"


@strawberry.type
class User:
    id: strawberry.ID
    username: str
    email: str
    bio: Optional[str] = None
    created_at: datetime = strawberry.field(name="createdAt")


@strawberry.type
class Post:
    id: strawberry.ID
    title: str
    content: str
    status: PostStatus
    author_id: strawberry.ID
    created_at: datetime = strawberry.field(name="createdAt")


# In-memory store for demo purposes
USERS = {
    "1": User(id="1", username="alice", email="alice@example.com",
              bio="Engineer", created_at=datetime(2024, 1, 1)),
    "2": User(id="2", username="bob", email="bob@example.com",
              bio=None, created_at=datetime(2024, 2, 1)),
}

POSTS = {
    "101": Post(id="101", title="Intro to GraphQL", content="...",
                status=PostStatus.PUBLISHED, author_id="1",
                created_at=datetime(2024, 3, 1)),
}


@strawberry.input
class CreatePostInput:
    title: str
    content: str
    author_id: strawberry.ID
    status: PostStatus = PostStatus.DRAFT


@strawberry.type
class Query:
    @strawberry.field
    def user(self, id: strawberry.ID) -> Optional[User]:
        return USERS.get(id)

    @strawberry.field
    def users(self, limit: int = 10) -> list[User]:
        return list(USERS.values())[:limit]

    @strawberry.field
    def post(self, id: strawberry.ID) -> Optional[Post]:
        return POSTS.get(id)


@strawberry.type
class Mutation:
    @strawberry.mutation
    def create_post(self, input: CreatePostInput) -> Post:
        post_id = str(len(POSTS) + 200)
        post = Post(
            id=post_id,
            title=input.title,
            content=input.content,
            status=input.status,
            author_id=input.author_id,
            created_at=datetime.now(),
        )
        POSTS[post_id] = post
        return post


schema = strawberry.Schema(query=Query, mutation=Mutation)
```

### FastAPI Integration

```python
# main.py
from fastapi import FastAPI
from strawberry.fastapi import GraphQLRouter
from schema import schema

app = FastAPI(title="GraphQL Demo")
graphql_router = GraphQLRouter(schema)
app.include_router(graphql_router, prefix="/graphql")


@app.get("/health")
def health():
    return {"status": "ok"}
```

### Running the Server

```bash
uvicorn main:app --reload --port 8000
# Open http://localhost:8000/graphql for GraphiQL interface
```

### Testing with curl

```bash
# Query
curl -X POST http://localhost:8000/graphql \
  -H "Content-Type: application/json" \
  -d '{"query": "{ user(id: \"1\") { username email } }"}'

# Mutation
curl -X POST http://localhost:8000/graphql \
  -H "Content-Type: application/json" \
  -d '{
    "query": "mutation($input: CreatePostInput!) { createPost(input: $input) { id title } }",
    "variables": {
      "input": {
        "title": "New Post",
        "content": "Hello World",
        "authorId": "1"
      }
    }
  }'
```

### Response Format

Every GraphQL response has this structure:

```json
{
  "data": { ... },
  "errors": [ ... ],
  "extensions": { ... }
}
```

- `data`: The result (always present if the request is valid)
- `errors`: Array of error objects (present if any errors occurred)
- `extensions`: Optional metadata (timing, tracing, etc.)

Error objects follow this format:

```json
{
  "message": "User not found",
  "locations": [{ "line": 2, "column": 3 }],
  "path": ["user"],
  "extensions": {
    "code": "NOT_FOUND"
  }
}
```

---

## 10. Exercises

### Exercise 1: Schema Design

Design a GraphQL schema (using SDL) for an e-commerce platform with:
- Products (id, name, price, category, inStock)
- Categories (id, name, products)
- Cart (id, items, total)
- CartItem (product, quantity)
- Queries: product, products (with filtering), cart
- Mutations: addToCart, removeFromCart, checkout

### Exercise 2: Query Writing

Given this schema, write queries for the following scenarios:

```graphql
type Query {
  movie(id: ID!): Movie
  movies(genre: Genre, year: Int, limit: Int): [Movie!]!
  search(query: String!): [SearchResult!]!
}
union SearchResult = Movie | Actor | Director
```

a) Fetch a movie with id "tt0111161" including its title, year, director name, and first 5 cast members.
b) Search for "Nolan" and handle all three result types.
c) Fetch two different movies in one request using aliases.

### Exercise 3: Mutation Payloads

Redesign this mutation to use the Shopify-style payload pattern with structured errors:

```graphql
# Before
type Mutation {
  registerUser(email: String!, password: String!, username: String!): User!
}
```

### Exercise 4: Python Server

Extend the minimal Strawberry server from Section 9 to add:
- A `Comment` type linked to both `Post` and `User`
- A `createComment` mutation
- A query to fetch all comments for a post

---

## 11. References

### Specifications
- [GraphQL Specification (October 2021)](https://spec.graphql.org/October2021/)
- [GraphQL over HTTP Specification](https://graphql.github.io/graphql-over-http/)

### Official Resources
- [GraphQL.org — Official Documentation](https://graphql.org/learn/)
- [GraphQL Foundation](https://graphql.org/foundation/)

### Libraries
- [Strawberry — Python GraphQL Library](https://strawberry.rocks/)
- [Apollo Server — JavaScript/TypeScript](https://www.apollographql.com/docs/apollo-server/)
- [graphql-core — Python Reference Implementation](https://github.com/graphql-python/graphql-core)

### Articles
- Lee Byron, "GraphQL: A Data Query Language" (2015)
- "Lessons from 4 Years of GraphQL" — Netflix Engineering
- "GraphQL at GitHub" — GitHub Engineering Blog

### Tools
- [GraphiQL — In-Browser IDE](https://github.com/graphql/graphiql)
- [Apollo Studio — Schema Registry and Explorer](https://studio.apollographql.com/)
- [GraphQL Voyager — Schema Visualization](https://graphql-kit.com/graphql-voyager/)

---

**License**: CC BY-NC 4.0
