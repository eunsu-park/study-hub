# 18. GraphQL Schema Design

**Previous**: [GraphQL Fundamentals](./17_GraphQL_Fundamentals.md) | **Next**: [GraphQL Resolvers](./19_GraphQL_Resolvers.md)

**Difficulty**: ⭐⭐⭐

---

## Learning Objectives

- Apply schema design principles that produce intuitive, evolvable GraphQL APIs
- Design effective input types with validation constraints and default values
- Use enums, interfaces, and union types to model domain concepts precisely
- Implement custom scalars for domain-specific data types
- Apply the Relay Connection specification for cursor-based pagination
- Design mutations following the input/payload pattern with structured error handling

---

## Table of Contents

1. [Schema Design Principles](#1-schema-design-principles)
2. [Naming Conventions](#2-naming-conventions)
3. [Designing Object Types](#3-designing-object-types)
4. [Input Types](#4-input-types)
5. [Enums](#5-enums)
6. [Interfaces and Abstract Types](#6-interfaces-and-abstract-types)
7. [Union Types](#7-union-types)
8. [Custom Scalars](#8-custom-scalars)
9. [Pagination: Relay Connections](#9-pagination-relay-connections)
10. [Mutation Design Patterns](#10-mutation-design-patterns)
11. [Schema Evolution](#11-schema-evolution)
12. [Exercises](#12-exercises)
13. [References](#13-references)

---

## 1. Schema Design Principles

A well-designed GraphQL schema is the most important factor in a successful GraphQL API. The schema is both documentation and contract.

### Guiding Principles

| Principle | Description |
|-----------|-------------|
| **Client-centric** | Design for how clients consume data, not how it is stored |
| **Expressive** | The schema should read like domain documentation |
| **Evolvable** | New fields and types can be added without breaking clients |
| **Consistent** | Follow naming patterns uniformly across the schema |
| **Minimal** | Expose only what clients need; avoid mirroring database tables |

### Think in Graphs, Not Endpoints

```
REST mindset:
  GET /users/42
  GET /users/42/posts
  GET /posts/101/comments

GraphQL mindset:
  User --[has many]--> Post --[has many]--> Comment
       --[has many]--> Follower
       --[has one]---> Profile
```

Design your types around domain relationships, not around API operations.

### Domain-Driven Design Alignment

Map your GraphQL types to domain entities, not database tables:

```graphql
# Bad: mirrors DB tables
type user_accounts {
  user_id: Int!
  user_name: String
  fk_profile_id: Int
}

# Good: models the domain
type User {
  id: ID!
  username: String!
  profile: Profile!
  posts(first: Int): PostConnection!
}
```

---

## 2. Naming Conventions

Consistent naming makes schemas self-documenting.

### Standard Conventions

| Element | Convention | Example |
|---------|-----------|---------|
| Types | PascalCase | `User`, `BlogPost` |
| Fields | camelCase | `firstName`, `createdAt` |
| Enums | SCREAMING_SNAKE_CASE | `POST_STATUS`, `PUBLISHED` |
| Input types | PascalCase + `Input` suffix | `CreateUserInput` |
| Payload types | PascalCase + `Payload` suffix | `CreateUserPayload` |
| Mutations | camelCase verbs | `createUser`, `deletePost` |
| Queries | camelCase nouns | `user`, `posts`, `searchResults` |
| Arguments | camelCase | `firstName`, `sortBy` |

### Naming Patterns for CRUD

```graphql
type Query {
  # Singular: fetch by ID
  user(id: ID!): User
  post(id: ID!): Post

  # Plural: fetch lists
  users(first: Int, after: String): UserConnection!
  posts(filter: PostFilter, first: Int, after: String): PostConnection!
}

type Mutation {
  # verb + noun
  createUser(input: CreateUserInput!): CreateUserPayload!
  updateUser(input: UpdateUserInput!): UpdateUserPayload!
  deleteUser(id: ID!): DeleteUserPayload!
}
```

### Boolean Fields

Prefix boolean fields with `is`, `has`, or `can`:

```graphql
type User {
  isVerified: Boolean!
  hasAvatar: Boolean!
  canEdit: Boolean!        # Computed based on viewer permissions
}
```

---

## 3. Designing Object Types

### Single Responsibility

Each type should represent one clear domain concept:

```graphql
# Bad: mixed concerns
type User {
  id: ID!
  username: String!
  orderTotal: Float!       # Belongs on Order aggregate
  shippingAddress: String! # Belongs on Address type
}

# Good: separated concerns
type User {
  id: ID!
  username: String!
  orders: OrderConnection!
  addresses: [Address!]!
}

type Address {
  id: ID!
  street: String!
  city: String!
  country: String!
  isDefault: Boolean!
}
```

### Computed Fields

Add fields that derive from other data:

```graphql
type Post {
  id: ID!
  title: String!
  content: String!
  wordCount: Int!           # Computed from content
  readingTimeMinutes: Int!  # Computed from wordCount
  isPublished: Boolean!     # Computed from status
  excerpt(length: Int = 200): String! # Parameterized computed field
}
```

### Field Arguments

Fields can accept arguments to customize the response:

```graphql
type User {
  id: ID!
  username: String!

  # Argument with default
  posts(
    status: PostStatus
    first: Int = 10
    after: String
    orderBy: PostOrderField = CREATED_AT
    orderDirection: OrderDirection = DESC
  ): PostConnection!

  # Formatted field
  createdAt(format: DateFormat = ISO8601): String!
}
```

### Nullability Guidelines

| Scenario | Nullable? | Rationale |
|----------|-----------|-----------|
| ID fields | Non-null (`!`) | Always present |
| Required business data | Non-null (`!`) | Must exist |
| Optional profile fields | Nullable | May not be set |
| Relationship to parent | Non-null (`!`) | Post always has Author |
| Relationship to optional | Nullable | User may not have Profile |
| Lists | Non-null list (`[T!]!`) | Return empty list, not null |
| Computed fields | Non-null (`!`) | Always computable |

> **Rule of Thumb**: Default to non-null. Make a field nullable only when there is a valid business reason for absence.

---

## 4. Input Types

Input types structure mutation arguments.

### Design Guidelines

```graphql
# Create: all required fields are non-null
input CreateProductInput {
  name: String!
  description: String!
  price: Float!
  categoryId: ID!
  tags: [String!] = []      # Default empty list
  isPublished: Boolean = false
}

# Update: all fields are nullable (partial update)
input UpdateProductInput {
  name: String
  description: String
  price: Float
  categoryId: ID
  tags: [String!]
  isPublished: Boolean
}
```

### Nested Input Types

```graphql
input CreateOrderInput {
  items: [OrderItemInput!]!
  shippingAddress: AddressInput!
  billingAddress: AddressInput
  couponCode: String
}

input OrderItemInput {
  productId: ID!
  quantity: Int!
}

input AddressInput {
  street: String!
  city: String!
  state: String
  postalCode: String!
  country: String!
}
```

### Reuse vs. Specificity

```graphql
# Bad: one input for all operations
input ProductInput {
  id: ID           # Only for update
  name: String!    # Required for create, optional for update
  price: Float!
}

# Good: separate inputs per operation
input CreateProductInput {
  name: String!
  price: Float!
}

input UpdateProductInput {
  name: String
  price: Float
}
```

---

## 5. Enums

Enums define a fixed set of valid values.

### When to Use Enums

```graphql
# Status workflows
enum OrderStatus {
  PENDING
  CONFIRMED
  SHIPPED
  DELIVERED
  CANCELLED
  REFUNDED
}

# Sorting options
enum ProductSortField {
  NAME
  PRICE
  CREATED_AT
  POPULARITY
}

enum SortDirection {
  ASC
  DESC
}

# Role-based access
enum UserRole {
  VIEWER
  EDITOR
  ADMIN
  SUPER_ADMIN
}
```

### Enum Naming Rules

- Use SCREAMING_SNAKE_CASE for values
- Group related values together
- Add new values at the end (to maintain ordering expectations)
- Never remove or rename values in production (deprecate instead)

### Deprecating Enum Values

```graphql
enum PostStatus {
  DRAFT
  PUBLISHED
  ARCHIVED
  DELETED @deprecated(reason: "Use ARCHIVED instead. Soft-delete is handled by isDeleted field.")
}
```

### Enums as Filter Arguments

```graphql
type Query {
  products(
    category: ProductCategory
    sortBy: ProductSortField = CREATED_AT
    sortDirection: SortDirection = DESC
    status: [ProductStatus!]  # Filter by multiple statuses
  ): ProductConnection!
}
```

---

## 6. Interfaces and Abstract Types

Interfaces define shared fields across multiple types.

### The Node Interface

The most common interface in GraphQL (from the Relay specification):

```graphql
interface Node {
  id: ID!
}

type User implements Node {
  id: ID!
  username: String!
}

type Post implements Node {
  id: ID!
  title: String!
}

type Query {
  node(id: ID!): Node  # Fetch any entity by global ID
}
```

### Domain Interfaces

```graphql
interface Auditable {
  createdAt: DateTime!
  updatedAt: DateTime!
  createdBy: User!
  updatedBy: User
}

interface Publishable {
  status: PublishStatus!
  publishedAt: DateTime
  author: User!
}

type Article implements Node & Auditable & Publishable {
  id: ID!
  title: String!
  content: String!
  createdAt: DateTime!
  updatedAt: DateTime!
  createdBy: User!
  updatedBy: User
  status: PublishStatus!
  publishedAt: DateTime
  author: User!
}
```

### Querying Interfaces

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
      content
    }
  }
}
```

### Interface vs. Union

| Aspect | Interface | Union |
|--------|-----------|-------|
| Shared fields | Required | None |
| Common behavior | Yes | No |
| Use case | Types that share structure | Types that are conceptually related but structurally different |
| Example | `Node { id }` | `SearchResult = User \| Post \| Comment` |

---

## 7. Union Types

Unions model situations where a field can return one of several unrelated types.

### Search Results

```graphql
union SearchResult = User | Post | Comment | Tag

type Query {
  search(query: String!, types: [SearchableType!]): [SearchResult!]!
}

enum SearchableType {
  USER
  POST
  COMMENT
  TAG
}
```

### Polymorphic Feeds

```graphql
union FeedItem = TextPost | ImagePost | VideoPost | SharedLink | Poll

type Feed {
  items(first: Int, after: String): FeedItemConnection!
}
```

### Error Unions (Result Pattern)

An alternative to payload-based errors:

```graphql
union CreateUserResult = User | ValidationError | DuplicateEmailError

type ValidationError {
  field: String!
  message: String!
}

type DuplicateEmailError {
  email: String!
  message: String!
}

type Mutation {
  createUser(input: CreateUserInput!): CreateUserResult!
}
```

Client usage:

```graphql
mutation {
  createUser(input: { username: "alice", email: "alice@example.com" }) {
    ... on User {
      id
      username
    }
    ... on ValidationError {
      field
      message
    }
    ... on DuplicateEmailError {
      email
      message
    }
  }
}
```

---

## 8. Custom Scalars

Custom scalars add domain-specific type safety.

### Common Custom Scalars

```graphql
scalar DateTime    # ISO 8601 date-time string
scalar Date        # ISO 8601 date (no time)
scalar Email       # RFC 5322 email address
scalar URL         # RFC 3986 URI
scalar JSON        # Arbitrary JSON blob
scalar UUID        # RFC 4122 UUID
scalar BigInt      # Integers beyond 32-bit
scalar Decimal     # Precise decimal numbers (money)
scalar PhoneNumber # E.164 phone number
```

### Implementation in Strawberry

```python
import strawberry
from datetime import datetime, date
from decimal import Decimal
import re


EmailScalar = strawberry.scalar(
    str,
    name="Email",
    description="RFC 5322 email address",
    serialize=lambda v: str(v),
    parse_value=lambda v: _validate_email(v),
)


def _validate_email(value: str) -> str:
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    if not re.match(pattern, value):
        raise ValueError(f"Invalid email address: {value}")
    return value


URLScalar = strawberry.scalar(
    str,
    name="URL",
    description="RFC 3986 URI",
    serialize=lambda v: str(v),
    parse_value=lambda v: _validate_url(v),
)


def _validate_url(value: str) -> str:
    from urllib.parse import urlparse
    result = urlparse(value)
    if not all([result.scheme, result.netloc]):
        raise ValueError(f"Invalid URL: {value}")
    return value


# Using custom scalars in types
@strawberry.type
class User:
    id: strawberry.ID
    email: EmailScalar          # type: ignore
    website: URLScalar | None   # type: ignore
```

### When to Create Custom Scalars

| Create | Do Not Create |
|--------|---------------|
| Domain values with validation (Email, URL) | Simple aliases (Name for String) |
| Precision types (Decimal for money) | Types with structure (use object types) |
| Standard formats (DateTime, UUID) | Composite values (use input types) |

---

## 9. Pagination: Relay Connections

The Relay Connection specification provides a standard pattern for paginated lists.

### Connection Types

```graphql
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

### Usage

```graphql
query {
  posts(first: 10, after: "cursor_abc") {
    edges {
      cursor
      node {
        id
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

### Pagination Arguments

```graphql
type Query {
  # Forward pagination
  posts(first: Int!, after: String): PostConnection!

  # Backward pagination
  posts(last: Int!, before: String): PostConnection!

  # With filtering
  posts(
    first: Int
    after: String
    filter: PostFilterInput
    orderBy: PostOrderInput
  ): PostConnection!
}
```

### Cursor Implementation

```python
import base64
from typing import Any


def encode_cursor(type_name: str, id_value: Any) -> str:
    """Encode a cursor as a base64 string."""
    raw = f"{type_name}:{id_value}"
    return base64.b64encode(raw.encode()).decode()


def decode_cursor(cursor: str) -> tuple[str, str]:
    """Decode a base64 cursor into (type_name, id_value)."""
    raw = base64.b64decode(cursor.encode()).decode()
    type_name, id_value = raw.split(":", 1)
    return type_name, id_value


# Example
cursor = encode_cursor("Post", 42)   # "UG9zdDo0Mg=="
type_name, id_val = decode_cursor(cursor)  # ("Post", "42")
```

### Offset vs. Cursor Pagination

| Aspect | Offset | Cursor |
|--------|--------|--------|
| Simplicity | Simple | Moderate |
| Performance | Degrades at high offsets | Consistent |
| Stability | Unstable (inserts shift pages) | Stable |
| Random access | Yes (`page=5`) | No (sequential only) |
| Best for | Small datasets, admin panels | Large datasets, feeds, infinite scroll |

---

## 10. Mutation Design Patterns

### Input/Payload Pattern

The recommended pattern used by Shopify, GitHub, and Stripe:

```graphql
# Input: what the client sends
input CreateOrderInput {
  items: [OrderItemInput!]!
  shippingAddressId: ID!
  couponCode: String
}

# Payload: what the server returns
type CreateOrderPayload {
  order: Order
  userErrors: [UserError!]!
}

type UserError {
  field: [String!]!
  message: String!
  code: UserErrorCode!
}

enum UserErrorCode {
  BLANK
  INVALID
  TOO_SHORT
  TOO_LONG
  NOT_FOUND
  TAKEN
  INSUFFICIENT_STOCK
}

type Mutation {
  createOrder(input: CreateOrderInput!): CreateOrderPayload!
}
```

### Bulk Operations

```graphql
input BulkUpdateProductStatusInput {
  ids: [ID!]!
  status: ProductStatus!
}

type BulkUpdateProductStatusPayload {
  products: [Product!]!
  failedIds: [ID!]!
  userErrors: [UserError!]!
}

type Mutation {
  bulkUpdateProductStatus(
    input: BulkUpdateProductStatusInput!
  ): BulkUpdateProductStatusPayload!
}
```

### Idempotent Mutations

Use a client-generated idempotency key:

```graphql
input CreatePaymentInput {
  orderId: ID!
  amount: Decimal!
  currency: CurrencyCode!
  idempotencyKey: String!  # Client-generated UUID
}
```

---

## 11. Schema Evolution

### Additive Changes (Non-Breaking)

- Adding new types
- Adding new fields to existing types
- Adding new enum values
- Adding optional arguments to fields
- Adding new queries or mutations

### Breaking Changes (Avoid)

- Removing types or fields
- Changing field types
- Making nullable fields non-null
- Removing enum values
- Renaming fields

### Deprecation Strategy

```graphql
type User {
  name: String! @deprecated(reason: "Use `firstName` and `lastName` instead")
  firstName: String!
  lastName: String!

  fullName: String!  # New computed field
}
```

### Schema Registry

Track schema changes in CI:

```bash
# Apollo Rover CLI
rover graph check my-graph@production --schema schema.graphql
rover graph publish my-graph@production --schema schema.graphql
```

---

## 12. Exercises

### Exercise 1: E-Commerce Schema

Design a complete GraphQL schema for an e-commerce platform with:
- Products with variants (size, color) and pricing
- Categories with hierarchy (parent/child)
- Shopping cart with line items
- Orders with status workflow
- Customer reviews and ratings
- Cursor-based pagination on all list fields

### Exercise 2: Input Type Design

Given these mutations, design the appropriate input and payload types:
- `createReview` — user reviews a product (1-5 stars, optional comment)
- `updateReview` — user updates their review
- `flagReview` — moderator flags a review with a reason

### Exercise 3: Custom Scalars

Implement custom scalars in Strawberry for:
- `Currency` — ISO 4217 currency code (USD, EUR, JPY)
- `Latitude` — floating point between -90 and 90
- `Longitude` — floating point between -180 and 180

### Exercise 4: Schema Refactoring

This schema has problems. Identify them and refactor:

```graphql
type Product {
  id: Int
  name: String
  desc: String
  price: Float
  price_currency: String
  cat_id: Int
  cat_name: String
  created: String
  tags: String  # Comma-separated
}
```

---

## 13. References

### Specifications
- [Relay Connection Specification](https://relay.dev/graphql/connections.htm)
- [Relay Global Object Identification](https://relay.dev/graphql/objectidentification.htm)

### Design Guides
- [Shopify GraphQL Design Tutorial](https://github.com/Shopify/graphql-design-tutorial)
- [GitHub GraphQL API Design Guidelines](https://docs.github.com/en/graphql/overview/about-the-graphql-api)
- [Apollo GraphQL Best Practices](https://www.apollographql.com/docs/technotes/)

### Books
- "Production Ready GraphQL" by Marc-Andre Giroux
- "Learning GraphQL" by Eve Porcello and Alex Banks (O'Reilly)

### Tools
- [GraphQL Inspector — Schema Diffing](https://graphql-inspector.com/)
- [GraphQL Code Generator](https://the-guild.dev/graphql/codegen)

---

**License**: CC BY-NC 4.0
