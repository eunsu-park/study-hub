# 01. GraphQL Fundamentals

**Previous**: [GraphQL Overview](./00_Overview.md) | **Next**: [Schema Design](./02_Schema_Design.md)

---

GraphQL is a query language for APIs and a runtime for fulfilling those queries with your existing data. Unlike REST, where the server decides what data each endpoint returns, GraphQL puts the client in control. Clients ask for exactly what they need, and the server responds with precisely that shape of data. This lesson introduces the core concepts that every GraphQL developer must understand before building schemas, writing resolvers, or optimizing performance.

**Difficulty**: ⭐⭐

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain what GraphQL is and articulate its advantages over REST APIs
2. Identify the three root operation types: Query, Mutation, and Subscription
3. Read and write basic schemas using the Schema Definition Language (SDL)
4. Use introspection queries to explore a GraphQL API
5. Describe the GraphQL execution model (parse, validate, execute)

---

## Table of Contents

1. [What Is GraphQL?](#1-what-is-graphql)
2. [GraphQL vs REST](#2-graphql-vs-rest)
3. [Core Concepts](#3-core-concepts)
4. [Schema Definition Language (SDL)](#4-schema-definition-language-sdl)
5. [Introspection](#5-introspection)
6. [The Execution Model](#6-the-execution-model)
7. [GraphiQL and Apollo Explorer](#7-graphiql-and-apollo-explorer)
8. [Your First GraphQL Server](#8-your-first-graphql-server)
9. [Practice Problems](#9-practice-problems)
10. [References](#10-references)

---

## 1. What Is GraphQL?

GraphQL was created at Facebook in 2012 to solve problems their mobile teams faced with REST APIs. The News Feed, for example, requires deeply nested data: posts, authors, comments, likes, and friend connections. With REST, fetching this data required either multiple round trips or bloated endpoints that returned far more data than needed.

GraphQL provides three key innovations:

1. **A type system** that describes your data (the schema)
2. **A query language** that lets clients specify exactly what they need
3. **A runtime** that validates queries against the schema and executes them

```
Traditional REST                        GraphQL
──────────────                         ─────────
GET /users/1           ──→  1 req     query {
GET /users/1/posts     ──→  2 req       user(id: 1) {
GET /posts/5/comments  ──→  3 req         name
                                          posts {
3 requests, lots of                        title
unused fields                              comments {
                                             body
                                           }
                                         }
                                       }
                                     }
                                     ──→  1 request, exact data
```

The GraphQL specification (maintained by the GraphQL Foundation under the Linux Foundation) defines the language, type system, and execution semantics. It is transport-agnostic --- most implementations use HTTP, but GraphQL can work over WebSockets, gRPC, or any other protocol.

## 2. GraphQL vs REST

Understanding when GraphQL shines (and when it does not) requires a fair comparison with REST.

### Where GraphQL Wins

| Problem | REST | GraphQL |
|---------|------|---------|
| **Over-fetching** | `/users/1` returns all 30 fields | Client selects only `name` and `email` |
| **Under-fetching** | Need 3 endpoints for a single view | One query, one request |
| **Versioning** | `/api/v1/`, `/api/v2/` | Schema evolves with `@deprecated` |
| **Documentation** | Separate (Swagger/OpenAPI) | Built-in via introspection |
| **Type safety** | Optional (JSON Schema) | Required (SDL) |

### Where REST Wins

| Aspect | REST | GraphQL |
|--------|------|---------|
| **Caching** | HTTP caching works naturally (GET + URL = cache key) | Requires special handling (POST body varies) |
| **File uploads** | Multipart form data is standard | Requires extensions (graphql-upload) |
| **Simplicity** | Trivial for CRUD resources | Schema design requires upfront investment |
| **Browser support** | Native fetch with URLs | Needs a client library for best experience |
| **Rate limiting** | Per-endpoint is straightforward | Query complexity varies per request |

### The Pragmatic View

Most production systems use both. REST handles simple CRUD and file operations; GraphQL serves complex, relational data needs. Facebook, GitHub, Shopify, and Airbnb all mix REST and GraphQL.

## 3. Core Concepts

GraphQL has three root operation types. Every GraphQL API must define at least a `Query` type.

### 3.1 Queries (Read)

Queries fetch data. They are analogous to REST GET requests.

```graphql
# Client sends this query
query GetUser {
  user(id: "1") {
    name
    email
    posts {
      title
      publishedAt
    }
  }
}
```

```json
// Server responds with this shape
{
  "data": {
    "user": {
      "name": "Alice",
      "email": "alice@example.com",
      "posts": [
        {
          "title": "Intro to GraphQL",
          "publishedAt": "2025-01-15"
        }
      ]
    }
  }
}
```

The response shape mirrors the query shape exactly. This is the fundamental principle of GraphQL: **you get what you ask for**.

### 3.2 Mutations (Write)

Mutations modify data. They are analogous to REST POST, PUT, PATCH, and DELETE.

```graphql
mutation CreatePost {
  createPost(input: {
    title: "GraphQL Basics"
    body: "GraphQL is a query language..."
    authorId: "1"
  }) {
    id
    title
    createdAt
  }
}
```

Mutations follow the same request/response pattern as queries, but by convention they represent side effects. The server executes mutation fields sequentially (unlike queries, which may execute in parallel).

### 3.3 Subscriptions (Real-time)

Subscriptions maintain a persistent connection for real-time updates.

```graphql
subscription OnNewComment {
  commentAdded(postId: "5") {
    id
    body
    author {
      name
    }
  }
}
```

When a new comment is added to post 5, the server pushes the data to all subscribed clients. Subscriptions typically use WebSockets.

### 3.4 The Response Format

Every GraphQL response has a consistent shape:

```json
{
  "data": { ... },      // The result (null if all fields errored)
  "errors": [ ... ],    // Optional: array of error objects
  "extensions": { ... } // Optional: metadata (timing, tracing)
}
```

This predictable structure means clients always know where to find data and errors.

## 4. Schema Definition Language (SDL)

The Schema Definition Language (SDL) is the syntax for defining GraphQL schemas. It is human-readable and serves as both documentation and a contract between client and server.

### 4.1 Type Definitions

```graphql
# Object type: represents an entity
type User {
  id: ID!                # Non-null scalar
  name: String!
  email: String!
  age: Int
  score: Float
  isActive: Boolean!
  posts: [Post!]!        # Non-null list of non-null Posts
  createdAt: String
}

type Post {
  id: ID!
  title: String!
  body: String!
  author: User!
  comments: [Comment!]!
  tags: [String!]
}

type Comment {
  id: ID!
  body: String!
  author: User!
  post: Post!
}
```

### 4.2 The Five Built-in Scalar Types

| Scalar | Description | Example |
|--------|-------------|---------|
| `Int` | 32-bit signed integer | `42` |
| `Float` | Double-precision float | `3.14` |
| `String` | UTF-8 character sequence | `"hello"` |
| `Boolean` | `true` or `false` | `true` |
| `ID` | Unique identifier (serialized as String) | `"abc123"` |

### 4.3 Root Types

```graphql
type Query {
  user(id: ID!): User
  users(limit: Int = 10, offset: Int = 0): [User!]!
  post(id: ID!): Post
  searchPosts(term: String!): [Post!]!
}

type Mutation {
  createUser(input: CreateUserInput!): User!
  updateUser(id: ID!, input: UpdateUserInput!): User!
  deleteUser(id: ID!): Boolean!
  createPost(input: CreatePostInput!): Post!
}

type Subscription {
  commentAdded(postId: ID!): Comment!
  userStatusChanged: User!
}
```

### 4.4 Input Types

Input types define the shape of mutation arguments:

```graphql
input CreateUserInput {
  name: String!
  email: String!
  age: Int
}

input UpdateUserInput {
  name: String
  email: String
  age: Int
}
```

Input types use `input` instead of `type` and cannot have fields that return object types.

### 4.5 The schema Keyword

The `schema` keyword explicitly maps root types:

```graphql
schema {
  query: Query
  mutation: Mutation
  subscription: Subscription
}
```

Most frameworks infer this from the type names, so you rarely write it explicitly.

## 5. Introspection

One of GraphQL's most powerful features is introspection: the ability to query the schema itself. This is how tools like GraphiQL and Apollo Explorer auto-generate documentation and provide autocomplete.

### 5.1 The __schema Query

```graphql
# List all types in the schema
query IntrospectSchema {
  __schema {
    types {
      name
      kind
      description
    }
    queryType {
      name
    }
    mutationType {
      name
    }
  }
}
```

### 5.2 The __type Query

```graphql
# Inspect a specific type
query IntrospectUser {
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
          kind
        }
      }
    }
  }
}
```

The response reveals every field, its type, nullability, and arguments. This is introspection in action.

### 5.3 Security Note

In production, introspection should be disabled for public APIs. It exposes your entire schema to potential attackers:

```javascript
// Apollo Server: disable introspection in production
const server = new ApolloServer({
  typeDefs,
  resolvers,
  introspection: process.env.NODE_ENV !== 'production',
});
```

## 6. The Execution Model

Every GraphQL request goes through three phases:

```
                 ┌─────────────┐
   Query String  │    Parse    │  → Abstract Syntax Tree (AST)
                 └──────┬──────┘
                        │
                 ┌──────▼──────┐
                 │   Validate  │  → Check against schema
                 └──────┬──────┘
                        │
                 ┌──────▼──────┐
                 │   Execute   │  → Call resolvers, build response
                 └─────────────┘
```

### 6.1 Parse

The query string is parsed into an Abstract Syntax Tree (AST). Syntax errors are caught here.

```
"{ user(id: 1) { name } }"

  → Document
    └─ OperationDefinition (query)
       └─ SelectionSet
          └─ Field: user
             ├─ Arguments: [id: 1]
             └─ SelectionSet
                └─ Field: name
```

### 6.2 Validate

The AST is validated against the schema. This phase catches errors like:

- Querying a field that does not exist on the type
- Missing required arguments
- Type mismatches (passing a String where an Int is expected)
- Fragment type conditions that are impossible

```graphql
# This would fail validation: 'age' might exist, but 'nonExistentField' does not
query {
  user(id: "1") {
    name
    nonExistentField   # ← Validation error
  }
}
```

### 6.3 Execute

The executor walks the AST and calls resolver functions for each field. Resolvers are covered in depth in Lesson 04, but the key insight is:

1. Start at the root Query type
2. Call the resolver for each requested field
3. For object-typed fields, recurse into their selection set
4. Assemble the final JSON response

```
Query.user(id: "1")          → { id: "1", name: "Alice", ... }
  └─ User.name               → "Alice"
  └─ User.posts               → [{ id: "5", title: "...", ... }]
       └─ Post.title          → "Intro to GraphQL"
       └─ Post.comments       → [{ id: "10", body: "...", ... }]
            └─ Comment.body   → "Great post!"
```

## 7. GraphiQL and Apollo Explorer

GraphQL APIs are best explored through interactive tools rather than curl commands.

### 7.1 GraphiQL

GraphiQL (pronounced "graphical") is the reference IDE for GraphQL. It provides:

- Schema documentation browser (via introspection)
- Query editor with syntax highlighting and autocomplete
- Variable editor panel
- Response viewer
- Query history

Most GraphQL servers serve GraphiQL at a dedicated endpoint:

```javascript
// Express + graphql-http with GraphiQL
import express from 'express';
import { createHandler } from 'graphql-http/lib/use/express';
import { buildSchema } from 'graphql';

const app = express();

const schema = buildSchema(`
  type Query {
    hello: String
  }
`);

app.all('/graphql', createHandler({
  schema,
  rootValue: {
    hello: () => 'Hello, World!',
  },
}));

// Serve GraphiQL HTML at /graphiql
app.get('/graphiql', (req, res) => {
  res.send(`
    <!DOCTYPE html>
    <html>
      <head>
        <link rel="stylesheet"
          href="https://unpkg.com/graphiql/graphiql.min.css" />
      </head>
      <body style="margin:0">
        <div id="graphiql" style="height:100vh"></div>
        <script src="https://unpkg.com/graphiql/graphiql.min.js"></script>
        <script>
          GraphiQL.createRoot(document.getElementById('graphiql'))
            .render(React.createElement(GraphiQL, {
              fetcher: GraphiQL.createFetcher({ url: '/graphql' }),
            }));
        </script>
      </body>
    </html>
  `);
});

app.listen(4000);
```

### 7.2 Apollo Explorer

Apollo Explorer is a cloud-hosted GraphQL IDE with additional features:

- Schema change tracking and diff
- Operation collection and sharing
- Performance tracing
- Mock responses

You can connect it to any GraphQL endpoint at [studio.apollographql.com/sandbox](https://studio.apollographql.com/sandbox).

## 8. Your First GraphQL Server

Let us build a minimal GraphQL server with Node.js to see all the concepts in action.

```bash
# Initialize project
mkdir graphql-intro && cd graphql-intro
npm init -y
npm install @apollo/server graphql
```

```javascript
// server.js
import { ApolloServer } from '@apollo/server';
import { startStandaloneServer } from '@apollo/server/standalone';

// 1. Define the schema (SDL)
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

// 2. In-memory data store
const books = [
  { id: '1', title: 'The Pragmatic Programmer', author: 'David Thomas', year: 1999 },
  { id: '2', title: 'Clean Code', author: 'Robert C. Martin', year: 2008 },
  { id: '3', title: 'Design Patterns', author: 'Gang of Four', year: 1994 },
];

let nextId = 4;

// 3. Define resolvers
const resolvers = {
  Query: {
    books: () => books,
    book: (_, { id }) => books.find(b => b.id === id),
  },
  Mutation: {
    addBook: (_, { title, author, year }) => {
      const book = { id: String(nextId++), title, author, year };
      books.push(book);
      return book;
    },
  },
};

// 4. Create and start the server
const server = new ApolloServer({ typeDefs, resolvers });
const { url } = await startStandaloneServer(server, { listen: { port: 4000 } });
console.log(`Server ready at ${url}`);
```

Run it:

```bash
node server.js
# Server ready at http://localhost:4000/
```

Now open `http://localhost:4000/` in a browser. Apollo Server 4 serves Apollo Sandbox by default. Try these queries:

```graphql
# Fetch all books
query {
  books {
    title
    author
  }
}

# Fetch a single book
query {
  book(id: "2") {
    title
    year
  }
}

# Add a new book
mutation {
  addBook(title: "Refactoring", author: "Martin Fowler", year: 1999) {
    id
    title
  }
}
```

This minimal example demonstrates the entire GraphQL flow: schema definition, resolver implementation, query execution, and mutation handling.

---

## 9. Practice Problems

### Exercise 1: Schema Reading (Beginner)

Given the following schema, answer the questions below:

```graphql
type Query {
  product(id: ID!): Product
  products(category: String): [Product!]!
}

type Product {
  id: ID!
  name: String!
  price: Float!
  category: String!
  reviews: [Review!]
}

type Review {
  id: ID!
  rating: Int!
  body: String
  reviewer: String!
}
```

1. What happens if you query `product(id: "999")` and no product exists with that ID?
2. Can `products(category: "electronics")` return an empty array? Can it return `null`?
3. Can a `Product` have `reviews` be `null`? Can a product have an empty array `[]` for reviews?
4. Can a single `Review` have `body` be `null`? Can `rating` be `null`?

### Exercise 2: Write a Schema (Intermediate)

Design a GraphQL schema for a music streaming service. Include:

- `Song` with title, artist, duration (in seconds), album, and genre
- `Playlist` with name, description, creator, and songs
- `User` with username, email, and playlists
- A query to search songs by title
- A mutation to create a playlist

Use appropriate scalar types, non-null modifiers, and list modifiers.

### Exercise 3: Identify Over-fetching (Beginner)

A mobile app needs to display a list of user names and avatars. The REST API returns:

```json
{
  "id": 1, "name": "Alice", "avatar": "/img/alice.jpg",
  "email": "alice@example.com", "phone": "555-0100",
  "address": { "street": "123 Main", "city": "Springfield", "zip": "62704" },
  "preferences": { "theme": "dark", "language": "en", "notifications": true },
  "lastLogin": "2025-01-15T10:30:00Z",
  "createdAt": "2023-06-01T00:00:00Z"
}
```

Write a GraphQL query that fetches only the data the mobile app needs. Estimate the bandwidth savings as a percentage.

### Exercise 4: Introspection Query (Intermediate)

Write an introspection query that retrieves:

1. All the fields of the `Product` type (from Exercise 1)
2. Each field's type name and whether it is non-null
3. Any arguments on the root `Query` type's fields

### Exercise 5: Build a GraphQL Server (Advanced)

Extend the book server from Section 8 with the following:

1. Add a `Genre` enum type with values: `FICTION`, `NON_FICTION`, `TECHNICAL`, `BIOGRAPHY`
2. Add a `genre` field to the `Book` type
3. Add a `booksByGenre(genre: Genre!): [Book!]!` query
4. Add an `updateBook(id: ID!, title: String, author: String, year: Int): Book` mutation
5. Add a `deleteBook(id: ID!): Boolean!` mutation

Test all operations using Apollo Sandbox.

---

## 10. References

- GraphQL Specification (October 2021) - https://spec.graphql.org/October2021/
- GraphQL Foundation - https://graphql.org/
- Apollo Server Documentation - https://www.apollographql.com/docs/apollo-server/
- "GraphQL: A Query Language for APIs" (original Facebook blog post) - https://engineering.fb.com/2015/09/14/core-infra/graphql-a-data-query-language/
- Lee Byron, "Lessons from 4 Years of GraphQL" (GraphQL Summit 2019)
- Principled GraphQL - https://principledgraphql.com/

---

**Previous**: [GraphQL Overview](./00_Overview.md) | **Next**: [Schema Design](./02_Schema_Design.md)
