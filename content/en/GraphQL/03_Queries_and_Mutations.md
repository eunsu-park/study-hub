# 03. Queries and Mutations

**Previous**: [Schema Design](./02_Schema_Design.md) | **Next**: [Resolvers](./04_Resolvers.md)

---

Lesson 01 introduced queries and mutations conceptually. This lesson takes you deep into the client-side query language: how to structure queries, reuse field selections with fragments, rename fields with aliases, control execution with directives, and design mutations that handle errors gracefully. Mastering these patterns is essential for building efficient, maintainable GraphQL clients.

**Difficulty**: ⭐⭐

## Learning Objectives

After completing this lesson, you will be able to:

1. Write queries with nested selection sets, arguments, and variables
2. Use fragments (named and inline) to eliminate duplication in queries
3. Apply aliases to rename fields and query the same field with different arguments
4. Use built-in directives (`@include`, `@skip`, `@deprecated`) to control query behavior
5. Design mutation operations with input types, payload types, and structured error handling

---

## Table of Contents

1. [Query Structure](#1-query-structure)
2. [Arguments and Variables](#2-arguments-and-variables)
3. [Fragments](#3-fragments)
4. [Inline Fragments](#4-inline-fragments)
5. [Aliases](#5-aliases)
6. [Directives](#6-directives)
7. [Mutations in Depth](#7-mutations-in-depth)
8. [Error Handling Patterns](#8-error-handling-patterns)
9. [Query Organization Tips](#9-query-organization-tips)
10. [Practice Problems](#10-practice-problems)
11. [References](#11-references)

---

## 1. Query Structure

A GraphQL query is a tree of **selection sets**. Each selection set specifies which fields to retrieve from a given type.

### 1.1 Basic Query Anatomy

```graphql
# Operation type    Operation name (optional but recommended)
query                GetUserProfile {
  # Root field with argument
  user(id: "1") {
    # Selection set: fields to retrieve
    id
    username
    email
    # Nested selection set (navigating a relationship)
    posts {
      id
      title
      publishedAt
      # Deeper nesting
      comments {
        id
        body
        author {
          username
        }
      }
    }
  }
}
```

Key rules:

- **Scalar fields** (String, Int, etc.) are leaf nodes --- they cannot have sub-selections
- **Object fields** (User, Post, etc.) must have a sub-selection set
- The depth of nesting is unlimited (but see Lesson 14 on depth limiting)

### 1.2 Multiple Root Fields

A single query can request multiple root fields. They execute in parallel:

```graphql
query DashboardData {
  me {
    username
    notificationCount
  }
  trending: posts(sortBy: TRENDING, first: 5) {
    title
    author { username }
  }
  recentPosts: posts(sortBy: RECENT, first: 10) {
    title
    publishedAt
  }
}
```

This is one of GraphQL's strengths: the client can compose exactly the data it needs for a view in a single request.

### 1.3 Shorthand Syntax

If a query has no name and no variables, you can omit the `query` keyword:

```graphql
# Shorthand (anonymous query)
{
  user(id: "1") {
    name
  }
}

# Equivalent to:
query {
  user(id: "1") {
    name
  }
}
```

The shorthand is fine for exploration in GraphiQL but should be avoided in application code. Always name your operations --- it helps with debugging, logging, and server-side metrics.

## 2. Arguments and Variables

### 2.1 Inline Arguments

Arguments can be hardcoded directly in the query:

```graphql
query {
  user(id: "1") { name }
  posts(first: 5, status: PUBLISHED) { title }
}
```

This is useful for quick exploration but impractical for real applications where values come from user input, state, or route parameters.

### 2.2 Variables

Variables separate the query structure from the dynamic values:

```graphql
# Query definition with variable declarations
query GetUser($userId: ID!, $postLimit: Int = 10) {
  user(id: $userId) {
    name
    email
    posts(first: $postLimit) {
      title
    }
  }
}
```

```json
// Variables (sent as a separate JSON object)
{
  "userId": "1",
  "postLimit": 5
}
```

**Variable rules:**

- Declared with `$` prefix and a type: `$userId: ID!`
- Can have default values: `$postLimit: Int = 10`
- Must be scalars, enums, or input types (not object types)
- Referenced by name in arguments: `id: $userId`

### 2.3 Why Variables Matter

```javascript
// ❌ String interpolation (dangerous, defeats caching)
const query = `
  query {
    user(id: "${userId}") { name }
  }
`;

// ✅ Variables (safe, cacheable, reusable)
const query = `
  query GetUser($userId: ID!) {
    user(id: $userId) { name }
  }
`;
const variables = { userId };
```

Using variables:

1. **Prevents injection attacks** --- variable values are type-checked and never interpolated into the query string
2. **Enables query caching** --- the same query string can be reused with different variables
3. **Supports persisted queries** --- the server stores the query string and the client sends only a hash + variables

### 2.4 Variable Types and Nullability

The type of a variable must match the argument it is used for, with some flexibility:

```graphql
# Schema
type Query {
  user(id: ID!): User            # id is non-null
  posts(status: PostStatus): [Post!]!  # status is nullable
}

# Query: variable type must match or be more specific
query($userId: ID!, $status: PostStatus) {
  user(id: $userId) { name }      # ID! matches ID!  ✅
  posts(status: $status) { title } # PostStatus matches PostStatus  ✅
}
```

A non-null variable (`$x: String!`) can be used where a nullable argument is expected (`arg: String`), but not vice versa.

## 3. Fragments

Fragments are reusable units of field selections. They solve the duplication problem when the same fields are needed in multiple places.

### 3.1 Defining and Using Fragments

```graphql
# Fragment definition
fragment UserBasicInfo on User {
  id
  username
  displayName
  avatarUrl
}

fragment PostSummary on Post {
  id
  title
  publishedAt
  author {
    ...UserBasicInfo
  }
}

# Using fragments with the spread operator (...)
query FeedPage {
  trending: posts(sortBy: TRENDING, first: 5) {
    ...PostSummary
    commentCount
  }
  recent: posts(sortBy: RECENT, first: 10) {
    ...PostSummary
    tags { name }
  }
  me {
    ...UserBasicInfo
    email
    notificationCount
  }
}
```

Without fragments, `UserBasicInfo` would be duplicated in three places. Fragments keep queries DRY and make them easier to maintain.

### 3.2 Fragment Composition

Fragments can reference other fragments, forming a composition tree:

```graphql
fragment CommentInfo on Comment {
  id
  body
  createdAt
  author {
    ...UserBasicInfo
  }
}

fragment PostDetail on Post {
  ...PostSummary
  body
  tags { name }
  comments(first: 20) {
    ...CommentInfo
  }
}

query PostPage($postId: ID!) {
  post(id: $postId) {
    ...PostDetail
  }
}
```

This creates a hierarchy: `PostDetail` includes `PostSummary`, which includes `UserBasicInfo`. Each fragment is defined once and reused everywhere.

### 3.3 Colocated Fragments (Client Pattern)

In component-based frontends (React, Vue), fragments are colocated with the component that uses them:

```javascript
// UserAvatar.jsx
import { gql } from '@apollo/client';

// This component declares exactly what data it needs
export const USER_AVATAR_FRAGMENT = gql`
  fragment UserAvatar on User {
    id
    username
    avatarUrl
  }
`;

export function UserAvatar({ user }) {
  return <img src={user.avatarUrl} alt={user.username} />;
}

// PostCard.jsx
import { gql } from '@apollo/client';
import { USER_AVATAR_FRAGMENT } from './UserAvatar';

export const POST_CARD_FRAGMENT = gql`
  fragment PostCard on Post {
    id
    title
    publishedAt
    author {
      ...UserAvatar
    }
  }
  ${USER_AVATAR_FRAGMENT}
`;

// FeedPage.jsx — composes the fragments into a query
import { gql, useQuery } from '@apollo/client';
import { POST_CARD_FRAGMENT } from './PostCard';

const FEED_QUERY = gql`
  query FeedPage($first: Int!) {
    posts(first: $first) {
      ...PostCard
    }
  }
  ${POST_CARD_FRAGMENT}
`;
```

This pattern ensures each component declares its data requirements, and the page query composes them all. Changing a component's data needs only requires updating its fragment.

## 4. Inline Fragments

Inline fragments are used in two situations: querying polymorphic types (interfaces/unions) and applying directives to a group of fields.

### 4.1 Type Conditions

When a field returns an interface or union, inline fragments select type-specific fields:

```graphql
query SearchResults($query: String!) {
  search(query: $query) {
    __typename
    ... on User {
      id
      username
      avatarUrl
    }
    ... on Post {
      id
      title
      publishedAt
      author { username }
    }
    ... on Comment {
      id
      body
      post { title }
    }
  }
}
```

The `__typename` meta-field returns the concrete type name (`"User"`, `"Post"`, or `"Comment"`), which is useful for client-side type discrimination.

### 4.2 Mixing Named and Inline Fragments

```graphql
fragment SearchUser on User {
  id
  username
  avatarUrl
  followerCount
}

query Search($query: String!) {
  search(query: $query) {
    ... on User {
      ...SearchUser
    }
    ... on Post {
      id
      title
      body
    }
  }
}
```

### 4.3 Inline Fragments Without Type Conditions

Inline fragments can also group fields to apply a directive:

```graphql
query Profile($userId: ID!, $includeStats: Boolean!) {
  user(id: $userId) {
    username
    bio
    ... @include(if: $includeStats) {
      followerCount
      followingCount
      postCount
    }
  }
}
```

This pattern conditionally includes a group of fields without creating a named fragment.

## 5. Aliases

Aliases let you rename a field in the response. They are essential when querying the same field multiple times with different arguments.

### 5.1 Basic Aliases

```graphql
query {
  # Without aliases, this would be a conflict:
  # two "user" fields in the same selection set
  alice: user(id: "1") {
    name
    email
  }
  bob: user(id: "2") {
    name
    email
  }
}
```

Response:

```json
{
  "data": {
    "alice": { "name": "Alice", "email": "alice@example.com" },
    "bob": { "name": "Bob", "email": "bob@example.com" }
  }
}
```

### 5.2 Aliases for Same-field Different-arguments

```graphql
query PostsByCategory {
  techPosts: posts(category: "technology", first: 5) {
    title
    publishedAt
  }
  sciencePosts: posts(category: "science", first: 5) {
    title
    publishedAt
  }
  sportsPosts: posts(category: "sports", first: 5) {
    title
    publishedAt
  }
}
```

### 5.3 Aliases for Client-friendly Names

Sometimes the schema field name does not match what the client expects:

```graphql
query {
  user(id: "1") {
    userId: id
    userName: username
    profilePic: avatarUrl
  }
}
```

This is useful when adapting GraphQL data to match an existing client data model or component props.

## 6. Directives

Directives modify how a field or fragment is executed. GraphQL specifies three built-in directives, and servers can define custom ones.

### 6.1 @include

Includes the field only if the condition is `true`:

```graphql
query GetUser($userId: ID!, $withPosts: Boolean!) {
  user(id: $userId) {
    name
    email
    posts @include(if: $withPosts) {
      title
    }
  }
}
```

```json
// Variables
{ "userId": "1", "withPosts": true }   // → posts included
{ "userId": "1", "withPosts": false }  // → posts omitted
```

### 6.2 @skip

The inverse of `@include` --- skips the field if the condition is `true`:

```graphql
query GetUser($userId: ID!, $skipEmail: Boolean!) {
  user(id: $userId) {
    name
    email @skip(if: $skipEmail)
  }
}
```

`@include(if: $x)` and `@skip(if: $x)` are logically opposite. Use whichever reads more naturally.

### 6.3 @deprecated (Schema Directive)

`@deprecated` is a schema directive (not a query directive). It marks a field as deprecated in the schema:

```graphql
type User {
  id: ID!
  name: String!

  # Deprecated field with migration guidance
  email: String! @deprecated(reason: "Use 'emailAddress' instead. Will be removed in v3.")
  emailAddress: String!

  # Deprecated enum value
  role: Role!
}

enum Role {
  ADMIN
  USER
  MODERATOR
  SUPER_ADMIN @deprecated(reason: "Use ADMIN with elevated permissions instead")
}
```

Deprecated fields remain functional but:
- Show warnings in GraphiQL/Apollo Explorer
- Appear with strikethrough in documentation
- Can be detected via introspection (the `isDeprecated` field)

### 6.4 Custom Directives (Preview)

Servers can define custom directives for cross-cutting concerns:

```graphql
# Schema definition
directive @auth(requires: Role!) on FIELD_DEFINITION
directive @cacheControl(maxAge: Int!) on FIELD_DEFINITION
directive @rateLimit(max: Int!, window: String!) on FIELD_DEFINITION

type Query {
  publicPosts: [Post!]! @cacheControl(maxAge: 300)

  me: User! @auth(requires: USER)

  adminStats: Stats! @auth(requires: ADMIN) @rateLimit(max: 10, window: "1m")
}
```

Custom directives are covered in more detail in Lesson 07 (Authentication) and Lesson 14 (Performance).

## 7. Mutations in Depth

Mutations are the write operations of GraphQL. While syntactically similar to queries, they have distinct semantics and design patterns.

### 7.1 Mutation Structure

```graphql
mutation CreatePost($input: CreatePostInput!) {
  createPost(input: $input) {
    post {
      id
      title
      slug
      publishedAt
    }
    errors {
      field
      message
    }
  }
}
```

```json
{
  "input": {
    "title": "GraphQL Best Practices",
    "body": "Here are some tips for designing GraphQL APIs...",
    "tags": ["graphql", "api-design"],
    "publishNow": true
  }
}
```

### 7.2 Sequential Execution

Unlike query fields which may execute in parallel, mutation fields execute **sequentially** in the order they appear:

```graphql
mutation {
  # Step 1: Execute first
  createUser(input: { username: "alice", email: "alice@example.com" }) {
    user { id }
  }
  # Step 2: Execute after step 1 completes
  createPost(input: { title: "Hello", body: "World", authorId: "new-user-id" }) {
    post { id }
  }
}
```

This ordering guarantee is important when later mutations depend on the side effects of earlier ones. However, note that you cannot reference the result of the first mutation in the second --- GraphQL does not support variable references between fields. For dependent operations, use separate requests or a single mutation that handles the chain server-side.

### 7.3 The Input Pattern

Following Relay conventions, mutations take a single `input` argument:

```graphql
# ❌ Many arguments (harder to extend, verbose)
type Mutation {
  createUser(
    username: String!
    email: String!
    password: String!
    displayName: String
    bio: String
    avatarUrl: String
  ): User!
}

# ✅ Single input argument (clean, extensible)
input CreateUserInput {
  username: String!
  email: String!
  password: String!
  displayName: String
  bio: String
  avatarUrl: String
}

type Mutation {
  createUser(input: CreateUserInput!): CreateUserPayload!
}
```

Benefits of the input pattern:
- **Extensible**: Adding fields to the input type is non-breaking
- **Reusable**: Input types can be shared or composed
- **Client-friendly**: Variables map cleanly to the input

### 7.4 Common Mutation Patterns

```graphql
# CRUD operations
type Mutation {
  # Create
  createPost(input: CreatePostInput!): CreatePostPayload!

  # Update (partial)
  updatePost(id: ID!, input: UpdatePostInput!): UpdatePostPayload!

  # Delete
  deletePost(id: ID!): DeletePostPayload!

  # Domain-specific actions (not CRUD)
  publishPost(id: ID!): PublishPostPayload!
  archivePost(id: ID!): ArchivePostPayload!
  likePost(id: ID!): LikePostPayload!
  addComment(postId: ID!, input: AddCommentInput!): AddCommentPayload!
}
```

Name mutations with `verb + noun` to clearly communicate intent. Avoid generic names like `updateEntity` or `modifyData`.

## 8. Error Handling Patterns

GraphQL has two layers of errors, and choosing the right pattern is crucial for client-developer experience.

### 8.1 Top-level Errors (Transport/Execution)

These appear in the `errors` array at the top level of the response:

```json
{
  "data": null,
  "errors": [
    {
      "message": "Cannot query field 'nonexistent' on type 'User'",
      "locations": [{ "line": 3, "column": 5 }],
      "extensions": {
        "code": "GRAPHQL_VALIDATION_FAILED"
      }
    }
  ]
}
```

Top-level errors include:
- Syntax errors (parse failure)
- Validation errors (invalid query)
- Authorization errors (not logged in)
- Internal server errors (unhandled exceptions)

### 8.2 Application Errors in Payloads

For business logic errors (validation, not found, permission denied), use the payload pattern:

```graphql
type CreateUserPayload {
  user: User
  errors: [CreateUserError!]!
}

type CreateUserError {
  field: String
  message: String!
  code: CreateUserErrorCode!
}

enum CreateUserErrorCode {
  USERNAME_TAKEN
  EMAIL_TAKEN
  INVALID_EMAIL
  PASSWORD_TOO_WEAK
  RATE_LIMITED
}
```

```json
{
  "data": {
    "createUser": {
      "user": null,
      "errors": [
        {
          "field": "username",
          "message": "Username 'alice' is already taken",
          "code": "USERNAME_TAKEN"
        }
      ]
    }
  }
}
```

### 8.3 Union-based Error Handling

A more type-safe approach uses unions:

```graphql
union CreateUserResult = CreateUserSuccess | ValidationError | NotFoundError

type CreateUserSuccess {
  user: User!
}

type ValidationError {
  field: String!
  message: String!
}

type NotFoundError {
  message: String!
  resourceType: String!
  resourceId: ID!
}

type Mutation {
  createUser(input: CreateUserInput!): CreateUserResult!
}
```

```graphql
mutation {
  createUser(input: { username: "alice", email: "alice@example.com" }) {
    ... on CreateUserSuccess {
      user { id username }
    }
    ... on ValidationError {
      field
      message
    }
  }
}
```

This pattern forces clients to handle errors explicitly. The tradeoff is more verbose queries.

## 9. Query Organization Tips

### 9.1 One Operation per File

In production codebases, store each operation in its own file:

```
src/
  graphql/
    queries/
      GetUser.graphql
      GetPosts.graphql
      SearchResults.graphql
    mutations/
      CreatePost.graphql
      UpdateUser.graphql
    fragments/
      UserBasicInfo.graphql
      PostSummary.graphql
```

GraphQL code generators (like `graphql-codegen`) can process these files to generate typed client code.

### 9.2 Naming Conventions

```graphql
# Queries: Get + Resource
query GetUser($id: ID!) { ... }
query GetPosts($filter: PostFilter) { ... }
query SearchContent($query: String!) { ... }

# Mutations: Verb + Resource
mutation CreatePost($input: CreatePostInput!) { ... }
mutation UpdateUser($id: ID!, $input: UpdateUserInput!) { ... }
mutation DeleteComment($id: ID!) { ... }
mutation PublishPost($id: ID!) { ... }

# Subscriptions: On + Event
subscription OnCommentAdded($postId: ID!) { ... }
subscription OnUserStatusChanged { ... }
```

### 9.3 Avoid Over-fetching in Queries

Even with GraphQL, it is possible to over-fetch:

```graphql
# ❌ Fetching everything "just in case"
query {
  user(id: "1") {
    id username email displayName bio avatarUrl
    createdAt updatedAt lastLoginAt
    posts { id title body publishedAt tags { name } }
    followers { id username avatarUrl }
    following { id username avatarUrl }
  }
}

# ✅ Fetch only what the current view needs
query UserProfile($userId: ID!) {
  user(id: $userId) {
    username
    displayName
    bio
    avatarUrl
    followerCount
  }
}
```

---

## 10. Practice Problems

### Exercise 1: Write Queries (Beginner)

Given this schema:

```graphql
type Query {
  user(id: ID!): User
  posts(first: Int, after: String, authorId: ID): PostConnection!
  search(query: String!, types: [SearchType!]): [SearchResult!]!
}

enum SearchType { USER POST COMMENT }
union SearchResult = User | Post | Comment

type User { id: ID!, username: String!, email: String!, posts: [Post!]! }
type Post { id: ID!, title: String!, body: String!, author: User!, comments: [Comment!]! }
type Comment { id: ID!, body: String!, author: User! }

type PostConnection {
  edges: [PostEdge!]!
  pageInfo: PageInfo!
}
type PostEdge { node: Post!, cursor: String! }
type PageInfo { hasNextPage: Boolean!, endCursor: String }
```

Write these queries using proper variables:

1. Fetch a user's username and email by ID
2. Fetch the first 5 posts with title, author username, and cursor-based pagination info
3. Search for "graphql" across all types, returning appropriate fields for each type

### Exercise 2: Fragment Refactoring (Intermediate)

The following query has duplicated fields. Refactor it using fragments:

```graphql
query {
  recentPosts: posts(sortBy: RECENT, first: 5) {
    id
    title
    publishedAt
    author {
      id
      username
      avatarUrl
    }
    commentCount
  }
  popularPosts: posts(sortBy: POPULAR, first: 5) {
    id
    title
    publishedAt
    author {
      id
      username
      avatarUrl
    }
    likeCount
  }
  me {
    id
    username
    avatarUrl
    drafts: posts(status: DRAFT) {
      id
      title
      publishedAt
      author {
        id
        username
        avatarUrl
      }
    }
  }
}
```

### Exercise 3: Mutation Design (Intermediate)

Design a complete mutation for "updating a user's profile." Include:

1. An input type with all updatable fields (display name, bio, avatar URL, timezone)
2. A payload type with the updated user and potential errors
3. An error type with field-level error information
4. The mutation field definition
5. A sample mutation query with variables

### Exercise 4: Aliases and Directives (Intermediate)

Write a single query that:

1. Fetches the current user's profile
2. Fetches the top 3 posts in each of these categories: technology, science, sports (use aliases)
3. Conditionally includes the current user's notification count (use `@include`)
4. Conditionally skips post comments (use `@skip`)

Use proper variables for all dynamic values.

### Exercise 5: Error Handling Comparison (Advanced)

A social media app needs a "follow user" mutation. Design it using:

**Approach A**: Payload pattern with errors array
**Approach B**: Union-based result type

For each approach:
- Define the schema types
- Write the mutation query
- Show the JSON response for: (a) success, (b) user not found, (c) already following, (d) cannot follow yourself

Compare the two approaches and explain which you would choose and why.

---

## 11. References

- GraphQL Query Language Specification - https://spec.graphql.org/October2021/#sec-Language
- GraphQL Variables - https://spec.graphql.org/October2021/#sec-Language.Variables
- GraphQL Fragments - https://spec.graphql.org/October2021/#sec-Language.Fragments
- GraphQL Directives - https://spec.graphql.org/October2021/#sec-Language.Directives
- Relay Mutation Convention - https://relay.dev/docs/guided-tour/updating-data/graphql-mutations/
- Apollo Client Fragments Guide - https://www.apollographql.com/docs/react/data/fragments/

---

**Previous**: [Schema Design](./02_Schema_Design.md) | **Next**: [Resolvers](./04_Resolvers.md)
