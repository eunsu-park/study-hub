# 02. Schema Design

**Previous**: [GraphQL Fundamentals](./01_GraphQL_Fundamentals.md) | **Next**: [Queries and Mutations](./03_Queries_and_Mutations.md)

---

The schema is the contract between your GraphQL server and every client that consumes it. A well-designed schema is intuitive, evolvable, and hard to misuse. A poorly designed schema leads to awkward queries, N+1 problems, and painful migrations. This lesson covers the full GraphQL type system --- scalars, objects, enums, interfaces, unions, and input types --- along with design principles that keep schemas maintainable as they grow.

**Difficulty**: ⭐⭐⭐

## Learning Objectives

After completing this lesson, you will be able to:

1. Use all GraphQL type system features: scalars, objects, enums, interfaces, unions, and input types
2. Apply non-null (`!`) and list (`[]`) modifiers correctly and explain their four combinations
3. Design custom scalar types for domain-specific data (DateTime, JSON, URL)
4. Choose between interfaces and unions for polymorphic schema design
5. Apply schema design best practices: naming conventions, nullable-by-default, and Relay-style connections

---

## Table of Contents

1. [The Type System at a Glance](#1-the-type-system-at-a-glance)
2. [Scalar Types](#2-scalar-types)
3. [Object Types](#3-object-types)
4. [Enum Types](#4-enum-types)
5. [Interface Types](#5-interface-types)
6. [Union Types](#6-union-types)
7. [Input Types](#7-input-types)
8. [Non-null and List Modifiers](#8-non-null-and-list-modifiers)
9. [Schema Design Best Practices](#9-schema-design-best-practices)
10. [Practice Problems](#10-practice-problems)
11. [References](#11-references)

---

## 1. The Type System at a Glance

GraphQL's type system has six named type kinds:

```
                    ┌─────────────────────────────────────┐
                    │         GraphQL Type System          │
                    └─────────────────────────────────────┘
                                    │
          ┌─────────────┬───────────┼───────────┬──────────────┐
          │             │           │           │              │
      ┌───▼───┐   ┌────▼────┐ ┌───▼───┐ ┌────▼────┐  ┌─────▼─────┐
      │Scalar │   │ Object  │ │ Enum  │ │Interface│  │   Union   │
      └───────┘   └─────────┘ └───────┘ └─────────┘  └───────────┘
       Int,String   User,Post   Status    Node         SearchResult
       Float,Bool   Comment     Role      Timestamped  Feed
       ID,custom    ...         ...       ...          ...

                           ┌─────▼─────┐
                           │   Input   │
                           └───────────┘
                           CreateUserInput
                           FilterInput
```

Plus two **wrapping types** that modify other types:

- **Non-Null** (`!`): the value must not be null
- **List** (`[]`): the value is an array of items

## 2. Scalar Types

Scalars are the leaf nodes of a GraphQL query --- they resolve to concrete values with no sub-fields.

### 2.1 Built-in Scalars

```graphql
type Example {
  id: ID!           # Unique identifier, serialized as String
  name: String!     # UTF-8 string
  age: Int          # 32-bit signed integer (-2^31 to 2^31-1)
  rating: Float     # IEEE 754 double-precision float
  isActive: Boolean # true or false
}
```

### 2.2 Custom Scalars

The five built-in scalars are not enough for real applications. Custom scalars add domain-specific semantics and validation.

**Schema definition:**

```graphql
scalar DateTime
scalar JSON
scalar URL
scalar EmailAddress
scalar PositiveInt

type Event {
  id: ID!
  title: String!
  startTime: DateTime!
  endTime: DateTime!
  metadata: JSON
  website: URL
  contactEmail: EmailAddress!
  maxAttendees: PositiveInt!
}
```

**Server implementation (Apollo Server):**

```javascript
import { GraphQLScalarType, Kind } from 'graphql';

const DateTimeScalar = new GraphQLScalarType({
  name: 'DateTime',
  description: 'ISO 8601 date-time string',

  // Server → Client: serialize internal value to output
  serialize(value) {
    if (value instanceof Date) {
      return value.toISOString();
    }
    throw new Error('DateTime must be a Date object');
  },

  // Client → Server: parse value from JSON (variables)
  parseValue(value) {
    if (typeof value === 'string') {
      const date = new Date(value);
      if (isNaN(date.getTime())) {
        throw new Error(`Invalid DateTime: ${value}`);
      }
      return date;
    }
    throw new Error('DateTime must be a string');
  },

  // Client → Server: parse value from inline literal in query
  parseLiteral(ast) {
    if (ast.kind === Kind.STRING) {
      const date = new Date(ast.value);
      if (isNaN(date.getTime())) {
        throw new Error(`Invalid DateTime: ${ast.value}`);
      }
      return date;
    }
    throw new Error('DateTime must be a string');
  },
});

const resolvers = {
  DateTime: DateTimeScalar,
  // ... other resolvers
};
```

**Community scalar library:**

Rather than implementing common scalars yourself, use `graphql-scalars`:

```bash
npm install graphql-scalars
```

```javascript
import { DateTimeResolver, JSONResolver, URLResolver } from 'graphql-scalars';

const resolvers = {
  DateTime: DateTimeResolver,
  JSON: JSONResolver,
  URL: URLResolver,
  // ... other resolvers
};
```

This library provides 40+ validated scalar types out of the box.

## 3. Object Types

Object types are the backbone of a GraphQL schema. They represent entities with fields.

### 3.1 Basic Object Types

```graphql
type User {
  id: ID!
  username: String!
  email: String!
  displayName: String
  bio: String
  avatarUrl: URL
  createdAt: DateTime!
}

type Post {
  id: ID!
  title: String!
  body: String!
  slug: String!
  publishedAt: DateTime
  author: User!          # Relationship: Post → User
  comments: [Comment!]!  # Relationship: Post → [Comment]
  tags: [Tag!]!
}
```

### 3.2 Field Arguments

Any field on an object type can accept arguments:

```graphql
type User {
  id: ID!
  username: String!
  posts(
    first: Int = 10       # Default value
    after: String         # Cursor for pagination
    status: PostStatus    # Filter by status
  ): PostConnection!

  # Computed field with argument
  fullName(format: NameFormat = FIRST_LAST): String!
}

enum PostStatus {
  DRAFT
  PUBLISHED
  ARCHIVED
}

enum NameFormat {
  FIRST_LAST
  LAST_FIRST
  USERNAME
}
```

Field arguments provide powerful filtering and customization without creating separate query fields.

### 3.3 The Root Query Type

The `Query` type is the entry point for all read operations. Think of it as the "table of contents" for your API:

```graphql
type Query {
  # Single-entity lookups
  user(id: ID!): User
  post(id: ID!): Post
  post_by_slug(slug: String!): Post

  # Collection queries
  users(first: Int = 20, after: String): UserConnection!
  posts(
    first: Int = 20
    after: String
    authorId: ID
    status: PostStatus
    tag: String
  ): PostConnection!

  # Search
  search(query: String!, types: [SearchType!]): [SearchResult!]!

  # Current user (authenticated)
  me: User
}
```

## 4. Enum Types

Enums represent a fixed set of allowed values. They are self-documenting and type-safe.

```graphql
enum Role {
  ADMIN
  MODERATOR
  USER
  GUEST
}

enum OrderStatus {
  PENDING
  CONFIRMED
  SHIPPED
  DELIVERED
  CANCELLED
  REFUNDED
}

enum SortOrder {
  ASC
  DESC
}

type User {
  id: ID!
  username: String!
  role: Role!
}

type Query {
  users(
    role: Role
    sortBy: String = "createdAt"
    sortOrder: SortOrder = DESC
  ): [User!]!
}
```

**Resolver mapping (when database values differ from enum names):**

```javascript
const resolvers = {
  // Map internal database values to GraphQL enum values
  Role: {
    ADMIN: 'admin',
    MODERATOR: 'mod',
    USER: 'user',
    GUEST: 'guest',
  },
  // In the resolver, the argument value will be the mapped value
  Query: {
    users: (_, { role }) => {
      // role is 'admin', 'mod', 'user', or 'guest' (the mapped value)
      return db.users.findAll({ where: role ? { role } : {} });
    },
  },
};
```

## 5. Interface Types

Interfaces define a set of fields that multiple types must implement. They enable polymorphism --- querying a collection of different types through a shared contract.

### 5.1 Defining Interfaces

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
  createdAt: DateTime!
  updatedAt: DateTime!
  username: String!
  email: String!
}

type Post implements Node & Timestamped {
  id: ID!
  createdAt: DateTime!
  updatedAt: DateTime!
  title: String!
  body: String!
  author: User!
}

type Comment implements Node & Timestamped {
  id: ID!
  createdAt: DateTime!
  updatedAt: DateTime!
  body: String!
  author: User!
}
```

### 5.2 Querying Interfaces

When a field returns an interface type, you can query the shared fields directly and use inline fragments for type-specific fields:

```graphql
type Query {
  node(id: ID!): Node
  recentActivity(limit: Int = 10): [Timestamped!]!
}
```

```graphql
query {
  node(id: "abc123") {
    id
    ... on User {
      username
      email
    }
    ... on Post {
      title
      body
    }
    ... on Comment {
      body
    }
  }
}
```

### 5.3 Resolver for Interface: __resolveType

The server needs to know which concrete type an object is:

```javascript
const resolvers = {
  Node: {
    __resolveType(obj) {
      // Determine the type based on the object's properties
      if (obj.username) return 'User';
      if (obj.title) return 'Post';
      if (obj.body && !obj.title) return 'Comment';
      return null;
    },
  },
};
```

### 5.4 When to Use Interfaces

Use interfaces when types share a **common contract** and you want to query them uniformly:

- `Node` interface for global object identification (Relay pattern)
- `Timestamped` for audit fields
- `Searchable` for full-text search results
- `Error` interface for typed error handling

## 6. Union Types

Unions are similar to interfaces, but without shared fields. They represent "one of these types."

### 6.1 Defining Unions

```graphql
union SearchResult = User | Post | Comment

union Feed = Post | SharedPost | Ad

type SharedPost {
  id: ID!
  originalPost: Post!
  sharedBy: User!
  comment: String
}

type Ad {
  id: ID!
  title: String!
  imageUrl: URL!
  targetUrl: URL!
  sponsor: String!
}

type Query {
  search(query: String!): [SearchResult!]!
  feed(first: Int = 20, after: String): [Feed!]!
}
```

### 6.2 Querying Unions

Since union types have no guaranteed shared fields, you must use inline fragments for all fields:

```graphql
query SearchQuery {
  search(query: "graphql") {
    ... on User {
      id
      username
      avatarUrl
    }
    ... on Post {
      id
      title
      publishedAt
    }
    ... on Comment {
      id
      body
      author { username }
    }
  }
}
```

You can also use the `__typename` meta-field to determine the concrete type:

```graphql
query {
  search(query: "graphql") {
    __typename
    ... on User { username }
    ... on Post { title }
    ... on Comment { body }
  }
}
```

### 6.3 Interfaces vs Unions: Decision Guide

| Criterion | Interface | Union |
|-----------|-----------|-------|
| Shared fields? | Yes, enforced | No |
| Types implement it? | Yes (`implements`) | No |
| Can add fields later? | Must update all implementors | No shared fields to add |
| Use case | Common contract (Node, Error) | Heterogeneous results (Search, Feed) |
| Multiple inheritance? | Yes (`implements A & B`) | N/A |

**Rule of thumb**: If the types share meaningful fields, use an interface. If they are fundamentally different types that happen to appear in the same context, use a union.

## 7. Input Types

Input types define the shape of arguments for mutations and complex queries. They are the "write-side" counterpart of object types.

### 7.1 Why Input Types Exist

Object types can have fields that return other object types (circular references, computed fields, etc.), which makes them unsuitable for input. Input types are strictly tree-shaped --- no cycles, no computed fields.

```graphql
# ❌ Cannot use object types as input
type Mutation {
  createUser(user: User!): User!  # Error: User is an object type
}

# ✅ Use input types for arguments
input CreateUserInput {
  username: String!
  email: String!
  password: String!
  role: Role = USER
}

type Mutation {
  createUser(input: CreateUserInput!): CreateUserPayload!
}
```

### 7.2 Input Type Patterns

```graphql
# Creation input: all required fields
input CreatePostInput {
  title: String!
  body: String!
  tags: [String!]
  publishNow: Boolean = false
}

# Update input: all fields optional (partial update)
input UpdatePostInput {
  title: String
  body: String
  tags: [String!]
}

# Filter input: flexible querying
input PostFilterInput {
  authorId: ID
  status: PostStatus
  tag: String
  createdAfter: DateTime
  createdBefore: DateTime
  search: String
}

# Pagination input
input PaginationInput {
  first: Int = 20
  after: String
}

type Query {
  posts(filter: PostFilterInput, pagination: PaginationInput): PostConnection!
}
```

### 7.3 The Payload Pattern

Mutations should return dedicated payload types, not raw entities. This pattern allows returning the entity, errors, and metadata:

```graphql
type CreateUserPayload {
  user: User
  errors: [UserError!]!
}

type UserError {
  field: String!
  message: String!
}

type Mutation {
  createUser(input: CreateUserInput!): CreateUserPayload!
}
```

```json
// Success response
{
  "data": {
    "createUser": {
      "user": { "id": "1", "username": "alice" },
      "errors": []
    }
  }
}

// Validation error response
{
  "data": {
    "createUser": {
      "user": null,
      "errors": [
        { "field": "email", "message": "Email already in use" }
      ]
    }
  }
}
```

## 8. Non-null and List Modifiers

The `!` (non-null) and `[]` (list) modifiers control nullability and cardinality. Understanding their four combinations is critical for schema design.

### 8.1 The Four Combinations

```graphql
type Example {
  a: String         # Nullable string: can be null
  b: String!        # Non-null string: never null
  c: [String]       # Nullable list of nullable strings: null, [], ["a", null]
  d: [String]!      # Non-null list of nullable strings: [], ["a", null]
  e: [String!]      # Nullable list of non-null strings: null, [], ["a", "b"]
  f: [String!]!     # Non-null list of non-null strings: [], ["a", "b"]
}
```

**Detailed breakdown:**

| Type | Can be `null`? | Can contain `null` items? | Valid values |
|------|---------------|--------------------------|--------------|
| `[String]` | Yes | Yes | `null`, `[]`, `["a"]`, `["a", null]` |
| `[String]!` | No | Yes | `[]`, `["a"]`, `["a", null]` |
| `[String!]` | Yes | No | `null`, `[]`, `["a", "b"]` |
| `[String!]!` | No | No | `[]`, `["a", "b"]` |

### 8.2 Nullability Best Practices

**Nullable by default.** This is the GraphQL specification's recommendation and for good reason.

Why make fields nullable?

1. **Error resilience**: If a resolver fails for one field, only that field becomes `null` --- the rest of the response is still valid. A non-null field that errors propagates the null upward to the nearest nullable parent, potentially destroying the entire response.

2. **Evolution**: Changing a non-null field to nullable is a breaking change for clients that rely on it being present.

```graphql
# The null propagation problem
type Query {
  user(id: ID!): User!   # Non-null
}

type User {
  name: String!           # Non-null
  avatar: String!         # Non-null — if this resolver fails...
  posts: [Post!]!         # Non-null
}

# If avatar resolver throws an error:
# 1. avatar → null (but it's String!, can't be null)
# 2. User object → null (but Query.user is User!, can't be null)
# 3. Entire data → null
# Result: { "data": null, "errors": [...] }
```

**When to use non-null:**

- `id: ID!` --- entities always have an ID
- Input fields that are truly required
- Boolean fields with clear defaults
- Fields backed by reliable sources (database columns with NOT NULL)

## 9. Schema Design Best Practices

### 9.1 Naming Conventions

```graphql
# Types: PascalCase
type UserProfile { ... }

# Fields: camelCase
type User {
  firstName: String!
  lastName: String!
  createdAt: DateTime!
}

# Enums: SCREAMING_SNAKE_CASE values
enum OrderStatus {
  IN_PROGRESS
  COMPLETED
  CANCELLED
}

# Input types: PascalCase with Input suffix
input CreateUserInput { ... }
input UpdateUserInput { ... }
input UserFilterInput { ... }

# Mutations: verb + noun
type Mutation {
  createUser(input: CreateUserInput!): CreateUserPayload!
  updateUser(input: UpdateUserInput!): UpdateUserPayload!
  deleteUser(id: ID!): DeleteUserPayload!
  publishPost(id: ID!): PublishPostPayload!  # Not: setPostPublished
}
```

### 9.2 Use Descriptions

SDL supports string descriptions that appear in introspection and documentation:

```graphql
"""
A registered user of the platform.
Users can create posts, leave comments, and follow other users.
"""
type User {
  "Globally unique identifier"
  id: ID!

  "Unique login name (3-30 chars, alphanumeric + underscores)"
  username: String!

  "Display name shown in the UI. Falls back to username if null."
  displayName: String

  "Number of followers. Cached, may be slightly stale."
  followerCount: Int!
}
```

### 9.3 Pagination: Relay-style Connections

For paginated lists, the Relay connection specification is the industry standard:

```graphql
type Query {
  posts(first: Int, after: String, last: Int, before: String): PostConnection!
}

type PostConnection {
  edges: [PostEdge!]!
  pageInfo: PageInfo!
  totalCount: Int!
}

type PostEdge {
  node: Post!
  cursor: String!
}

type PageInfo {
  hasNextPage: Boolean!
  hasPreviousPage: Boolean!
  startCursor: String
  endCursor: String
}
```

```graphql
# Usage: fetch first 10 posts after a cursor
query {
  posts(first: 10, after: "cursor_abc") {
    edges {
      cursor
      node {
        title
        author { username }
      }
    }
    pageInfo {
      hasNextPage
      endCursor
    }
    totalCount
  }
}
```

### 9.4 Think in Graphs, Not Endpoints

REST thinks in resources and endpoints. GraphQL thinks in nodes and edges.

```graphql
# ❌ REST-style thinking (flat, disconnected)
type Query {
  user(id: ID!): User
  userPosts(userId: ID!): [Post!]!
  userFollowers(userId: ID!): [User!]!
  postComments(postId: ID!): [Comment!]!
}

# ✅ Graph thinking (connected)
type Query {
  user(id: ID!): User
  post(id: ID!): Post
}

type User {
  id: ID!
  posts(first: Int = 10): PostConnection!
  followers(first: Int = 10): UserConnection!
  following(first: Int = 10): UserConnection!
}

type Post {
  id: ID!
  author: User!
  comments(first: Int = 10): CommentConnection!
}
```

### 9.5 Schema Evolution

Schemas evolve. Here is how to make changes safely:

```graphql
# ✅ Safe changes (non-breaking)
# - Adding a new nullable field
# - Adding a new type
# - Adding a new enum value (end of list)
# - Adding a new query/mutation field
# - Deprecating a field

type User {
  id: ID!
  name: String!
  email: String! @deprecated(reason: "Use emailAddress instead")
  emailAddress: String!   # New field
  phone: String           # New nullable field
}

# ❌ Breaking changes
# - Removing a field
# - Making a nullable field non-null
# - Removing an enum value
# - Changing a field's type
# - Adding a required argument to an existing field
```

---

## 10. Practice Problems

### Exercise 1: Nullability Analysis (Beginner)

For each field, state whether the given value is valid or would cause an error:

```graphql
type Article {
  id: ID!
  title: String!
  subtitle: String
  tags: [String!]!
  relatedArticles: [Article]
}
```

1. `id: null`
2. `subtitle: null`
3. `tags: null`
4. `tags: []`
5. `tags: ["graphql", null]`
6. `tags: ["graphql", "api"]`
7. `relatedArticles: null`
8. `relatedArticles: [null]`

### Exercise 2: Interface vs Union (Intermediate)

A notification system has three notification types:

- **FollowNotification**: someone followed you (includes follower user)
- **CommentNotification**: someone commented on your post (includes comment and post)
- **MentionNotification**: someone mentioned you (includes mentioner user and context text)

All notifications share: id, createdAt, isRead, recipient.

Design this using (a) an interface and (b) a union. Which approach is better and why?

### Exercise 3: Schema Design (Intermediate)

Design a GraphQL schema for an e-commerce platform with:

- Products with name, price, description, images, category, inventory count
- Categories that can be nested (e.g., Electronics > Phones > Smartphones)
- Shopping cart with items (product + quantity)
- Orders with items, shipping address, status tracking
- Users with addresses, order history, and wishlists

Include appropriate input types for creating/updating entities. Use Relay-style connections for paginated lists.

### Exercise 4: Custom Scalar (Advanced)

Implement a `Currency` custom scalar that:

- Accepts values like `"USD 29.99"` or `"EUR 15.00"`
- Validates the currency code against a whitelist (USD, EUR, GBP, JPY, KRW)
- Stores internally as `{ code: string, amount: number }`
- Serializes back to the string format

Write the `GraphQLScalarType` implementation in JavaScript.

### Exercise 5: Schema Review (Advanced)

Review the following schema and identify all design issues. Propose improvements.

```graphql
type Query {
  getUser(userId: String!): User!
  getAllUsers: [User]
  getUserPosts(userId: String!): [Post]
  getPost(postId: String!): Post
  search(q: String!): [SearchResult]
}

type User {
  userId: String!
  name: String!
  email: String!
  age: String
  role: String
  Posts: [Post]
}

type Post {
  postId: String!
  Title: String!
  content: String!
  date: String!
  user_id: String!
  comments: [Comment]!
}

type Comment {
  id: Int!
  text: String!
  userId: String!
}

union SearchResult = User | Post

type Mutation {
  createUser(name: String!, email: String!, age: Int, role: String): User!
  updateUser(userId: String!, name: String, email: String, age: Int, role: String): User
  createPost(userId: String!, title: String!, content: String!): Post!
}
```

---

## 11. References

- GraphQL Type System Specification - https://spec.graphql.org/October2021/#sec-Type-System
- Relay Connection Specification - https://relay.dev/graphql/connections.htm
- GraphQL Scalars Library - https://the-guild.dev/graphql/scalars
- Principled GraphQL: Schema Design - https://principledgraphql.com/agility#4-abstract-demand-oriented-schema
- Marc-Andre Giroux, "Production Ready GraphQL" (2020)
- Lee Byron, "Designing a GraphQL Schema" (GraphQL Summit 2018)

---

**Previous**: [GraphQL Fundamentals](./01_GraphQL_Fundamentals.md) | **Next**: [Queries and Mutations](./03_Queries_and_Mutations.md)
