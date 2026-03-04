# GraphQL Examples

This directory contains example files demonstrating key GraphQL concepts, from schema design to federation and API gateways.

## Files Overview

| # | File | Topic | Key Demos |
|---|------|-------|-----------|
| 01 | `01_schema_resolvers.js` | Schema and Resolvers | SDL, type definitions, resolver functions |
| 02 | `02_dataloader.js` | DataLoader | N+1 problem, batching, per-request caching |
| 03 | `03_auth_context.js` | Authentication and Authorization | Context-based auth, field-level permissions, directives |
| 04 | `04_subscriptions.js` | Subscriptions | PubSub, real-time WebSocket, graphql-ws |
| 05 | `05_strawberry_python.py` | Code-First Python | Strawberry + FastAPI, type definitions, mutations |
| 06 | `06_apollo_client.tsx` | Apollo Client (React) | useQuery, useMutation, cache, optimistic updates |
| 07 | `07_testing.js` | Testing | Resolver unit tests, executeOperation, mocking |
| 08 | `08_persisted_queries.js` | Persisted Queries and Caching | APQ protocol, cache control directives, multi-layer caching |
| 09 | `09_federation.js` | Federation | Apollo Federation 2, entity references, cross-subgraph resolution |
| 10 | `10_performance_security.js` | Performance and Security | Query depth limiting, complexity analysis, rate limiting |
| 11 | `11_rest_migration.js` | REST-to-GraphQL Migration | REST wrapper resolvers, data source pattern, field mapping |
| 12 | `12_api_gateway.js` | API Gateway | Federated supergraph, gateway routing, entity resolution |

## Running Examples

JavaScript examples require Node.js 18+ and relevant npm packages (see file headers for install commands):

```bash
npm install @apollo/server graphql
node 01_schema_resolvers.js
```

The Python example requires `strawberry-graphql` and `fastapi`:

```bash
pip install strawberry-graphql[fastapi] fastapi uvicorn
python 05_strawberry_python.py
```
