# GraphQL

## Topic Overview

GraphQL is a query language for APIs that lets clients request exactly the data they need — no more, no less. Created at Facebook in 2012 and open-sourced in 2015, it has become the standard API layer for complex frontend applications, mobile apps, and microservice architectures. Unlike REST where the server defines the response shape, GraphQL puts the client in control.

This topic covers GraphQL from schema design through production deployment, touching both server-side (Apollo Server, Strawberry) and client-side (Apollo Client, urql) implementations. The final lessons cover Federation for microservices and a capstone project.

## Learning Path

```
Fundamentals              Server-Side                    Client & Operations
──────────────           ──────────────                 ──────────────────
01 Fundamentals          04 Resolvers                    09 GraphQL Clients
02 Schema Design         05 DataLoader (N+1)            11 Persisted Queries
03 Queries & Mutations   06 Subscriptions               12 Federation
                         07 Auth                         13 Testing
                         08 Apollo Server                14 Performance & Security
                         10 Code-First (Python)          15 REST to GraphQL

                                                         Project
                                                         ──────────────────
                                                         16 API Gateway Project
```

## Lesson List

| # | Lesson | Difficulty | Key Concepts |
|---|--------|------------|--------------|
| 01 | [GraphQL Fundamentals](./01_GraphQL_Fundamentals.md) | ⭐⭐ | Query/Mutation/Subscription, SDL, introspection |
| 02 | [Schema Design](./02_Schema_Design.md) | ⭐⭐⭐ | Types: Scalar, Object, Interface, Union, Enum |
| 03 | [Queries and Mutations](./03_Queries_and_Mutations.md) | ⭐⭐ | Variables, fragments, aliases, directives |
| 04 | [Resolvers](./04_Resolvers.md) | ⭐⭐⭐ | Resolver chain, context, info object |
| 05 | [DataLoader and N+1](./05_DataLoader_N_plus_1.md) | ⭐⭐⭐⭐ | N+1 problem, DataLoader batching/caching |
| 06 | [Subscriptions](./06_Subscriptions.md) | ⭐⭐⭐ | WebSocket, graphql-ws, Redis pub/sub |
| 07 | [Authentication and Authorization](./07_Authentication_Authorization.md) | ⭐⭐⭐ | Context-based auth, @auth directive |
| 08 | [Apollo Server](./08_Apollo_Server.md) | ⭐⭐⭐ | Apollo Server 4, schema-first vs code-first |
| 09 | [GraphQL Clients](./09_GraphQL_Clients.md) | ⭐⭐⭐ | Apollo Client 3, urql, TanStack Query |
| 10 | [Code-First with Python](./10_Code_First_Python.md) | ⭐⭐⭐ | Strawberry + FastAPI integration |
| 11 | [Persisted Queries and Caching](./11_Persisted_Queries_Caching.md) | ⭐⭐⭐ | APQ, HTTP caching, CDN |
| 12 | [Federation](./12_Federation.md) | ⭐⭐⭐⭐ | Apollo Federation 2, supergraph, @key/@external |
| 13 | [Testing](./13_Testing.md) | ⭐⭐⭐ | Resolver unit tests, integration tests |
| 14 | [Performance and Security](./14_Performance_Security.md) | ⭐⭐⭐⭐ | Query depth/complexity limits, rate limiting |
| 15 | [REST to GraphQL Migration](./15_REST_to_GraphQL_Migration.md) | ⭐⭐⭐ | REST wrapping, schema stitching, graphql-mesh |
| 16 | [Project: API Gateway](./16_Project_API_Gateway.md) | ⭐⭐⭐⭐ | Federation-based API gateway project |

## Prerequisites

- HTTP and REST API concepts
- JavaScript/TypeScript basics
- Node.js and npm
- Python basics (for Lesson 10)
- Basic database knowledge (SQL)

## Example Code

Runnable examples are in [`examples/GraphQL/`](../../../examples/GraphQL/).
