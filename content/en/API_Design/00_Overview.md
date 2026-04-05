# API Design

## Table of Contents
1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Lessons](#lessons)
4. [Learning Path](#learning-path)
5. [Related Topics](#related-topics)
6. [Additional Resources](#additional-resources)

---

## Overview

API Design is the discipline of crafting application programming interfaces that are intuitive, consistent, and durable. A well-designed API reduces integration friction, lowers the learning curve for consumers, and becomes the cornerstone of a scalable software ecosystem. Whether you are building internal microservice boundaries or a public developer platform, the principles covered in this guide will help you ship interfaces that developers enjoy using and that evolve gracefully over time.

This guide takes a hands-on, code-first approach grounded in Python (FastAPI and Flask). You will move from foundational concepts such as REST constraints and URL modeling through advanced topics like rate limiting, SDK generation, and API gateway architecture. Each lesson contains runnable examples, real-world patterns, and exercises designed to build production-ready skills.

By the end of this guide you will be able to design, document, version, secure, and operate HTTP APIs that meet industry best practices and stand the test of time.

**Learning Goals:**
- Understand API paradigms (REST, RPC, GraphQL, event-driven) and choose the right one
- Design clean, consistent, and discoverable resource URLs
- Implement authentication, authorization, pagination, filtering, and error handling
- Version APIs and manage backward compatibility
- Document APIs with OpenAPI/Swagger
- Apply rate limiting, caching, and performance optimization
- Design and operate API gateways and developer portals

**Characteristics:**
- Progressive learning from fundamentals to production operations
- Practical focus with Python/FastAPI and Flask code examples
- Coverage of latest standards (OpenAPI 3.1, RFC 7807, OAuth 2.1)
- Real-world patterns drawn from public APIs (Stripe, GitHub, Twilio)

---

## Prerequisites

### Required Knowledge
- [Programming](../Programming/00_Overview.md) -- variables, control flow, functions, data structures
- [Web Development](../Web_Development/00_Overview.md) -- HTTP basics, client-server model, HTML/CSS
- [Python](../Python/00_Overview.md) -- intermediate Python including decorators, type hints, and virtual environments

### Recommended Tools
- **Python 3.11+** with `pip` and `venv`
- **FastAPI** (`pip install "fastapi[standard]"`)
- **Flask** (`pip install flask`)
- **HTTPie or curl** for testing endpoints
- **Postman** or **Bruno** for interactive API exploration
- **Docker** (optional, for containerized examples)

### Installation Guide

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows

# Install core dependencies
pip install "fastapi[standard]" flask httpie pydantic

# Verify installations
python -c "import fastapi; print(f'FastAPI {fastapi.__version__}')"
python -c "import flask; print(f'Flask {flask.__version__}')"
```

---

## Lessons

| # | File Name | Topic | Difficulty | Time |
|---|-----------|-------|------------|------|
| 00 | 00_Overview.md | Learning guide and roadmap | - | 10min |
| 01 | 01_API_Design_Fundamentals.md | API types, design principles, contract-first vs code-first | :star: | 2h |
| 02 | 02_REST_Architecture.md | REST constraints, Richardson Maturity Model, HATEOAS | :star: | 2h |
| 03 | 03_URL_Design_and_Naming.md | Resource naming, hierarchical URLs, query parameters | :star: | 2h |
| 04 | 04_Request_and_Response_Design.md | HTTP methods, status codes, content negotiation | :star::star: | 3h |
| 05 | 05_Pagination_and_Filtering.md | Cursor/offset pagination, filtering, sorting, sparse fieldsets | :star::star: | 2h |
| 06 | 06_Authentication_and_Authorization.md | API keys, OAuth 2.0, JWT, scopes and permissions | :star::star: | 3h |
| 07 | 07_API_Versioning.md | URL/header/query versioning, deprecation, backward compatibility | :star::star: | 2h |
| 08 | 08_Error_Handling.md | RFC 7807 Problem Details, error hierarchy, validation errors | :star::star: | 2h |
| 09 | 09_Rate_Limiting_and_Throttling.md | Token bucket, sliding window, rate limit headers, retry logic | :star::star: | 2h |
| 10 | 10_API_Documentation.md | OpenAPI 3.1, Swagger UI, Redoc, documentation-driven design | :star::star: | 2h |
| 11 | 11_API_Testing_and_Validation.md | API testing pyramid, schema validation, contract testing, fuzz testing | :star::star::star: | 3h |
| 12 | 12_Webhooks_and_Callbacks.md | Webhook design, HMAC signatures, retry policies, dead letter queues | :star::star::star: | 2h |
| 13 | 13_API_Gateway_Patterns.md | API gateway, BFF pattern, API composition, Kong, AWS API Gateway | :star::star::star: | 2h |
| 14 | 14_gRPC_and_Protocol_Buffers.md | Protocol Buffers, gRPC services, streaming patterns, transcoding | :star::star::star: | 3h |
| 15 | 15_API_Security.md | OWASP API Top 10, input validation, CORS, CSRF, BOLA prevention | :star::star::star: | 3h |
| 16 | 16_API_Lifecycle_Management.md | API lifecycle, changelog, deprecation, migration guides, developer portal | :star::star::star: | 3h |
| 17 | 17_GraphQL_Fundamentals.md | GraphQL overview, SDL, type system, queries vs mutations | :star::star::star: | 3h |
| 18 | 18_GraphQL_Schema_Design.md | Schema design, input types, enums, interfaces, unions, custom scalars | :star::star::star: | 3h |
| 19 | 19_GraphQL_Resolvers.md | Resolver patterns, context, DataLoader for N+1, batching | :star::star::star: | 3h |
| 20 | 20_GraphQL_Subscriptions.md | Real-time subscriptions, WebSocket transport, pub/sub patterns | :star::star::star: | 3h |
| 21 | 21_GraphQL_Server_Implementation.md | Strawberry, Apollo Server, code-first vs schema-first, middleware | :star::star::star: | 3h |
| 22 | 22_GraphQL_Federation.md | Schema federation, subgraph design, entity resolution, Apollo Federation 2 | :star::star::star::star: | 3h |
| 23 | 23_GraphQL_Performance_Security.md | Query complexity, depth limiting, persisted queries, caching, rate limiting | :star::star::star::star: | 3h |
| 24 | 24_GraphQL_Testing_and_Tooling.md | Testing strategies, mocking, introspection, code generation | :star::star::star: | 3h |
| 25 | 25_API_Capstone_Unified_Gateway.md | Unified API gateway: REST + GraphQL + gRPC, migration strategies | :star::star::star::star: | 4h |

**Total estimated learning time: ~65 hours**

---

## Learning Path

### Beginner Track (Week 1)
Build a solid foundation in REST principles and URL design.

```
Day 1-2: 01_API_Design_Fundamentals.md
Day 3-4: 02_REST_Architecture.md
Day 5-7: 03_URL_Design_and_Naming.md
```

### Intermediate Track (Weeks 2-3)
Master the core mechanics of request/response handling, security, and versioning.

```
Week 2:
├── Day 1-2: 04_Request_and_Response_Design.md
├── Day 3-4: 05_Pagination_and_Filtering.md
└── Day 5-7: 06_Authentication_and_Authorization.md

Week 3:
├── Day 1-2: 07_API_Versioning.md
├── Day 3-4: 08_Error_Handling.md
└── Day 5-7: 09_Rate_Limiting_and_Throttling.md
```

### Advanced Track (Weeks 4-5)
Dive into documentation, advanced patterns, and operations.

```
Week 4:
├── Day 1-2: 10_API_Documentation.md
├── Day 3-4: 11_API_Testing_and_Validation.md
└── Day 5-7: 12_Webhooks_and_Callbacks.md

Week 5:
├── Day 1-2: 13_API_Gateway_Patterns.md
├── Day 3-4: 14_gRPC_and_Protocol_Buffers.md
├── Day 5-6: 15_API_Security.md
└── Day 7:   16_API_Lifecycle_Management.md

Week 6-7 (GraphQL Deep Dive):
├── Day 1-2: 17_GraphQL_Fundamentals.md
├── Day 3-4: 18_GraphQL_Schema_Design.md
├── Day 5-6: 19_GraphQL_Resolvers.md
├── Day 7-8: 20_GraphQL_Subscriptions.md
├── Day 9-10: 21_GraphQL_Server_Implementation.md
├── Day 11-12: 22_GraphQL_Federation.md
├── Day 13: 23_GraphQL_Performance_Security.md
└── Day 14: 24_GraphQL_Testing_and_Tooling.md

Week 8 (Capstone):
└── Day 1-7: 25_API_Capstone_Unified_Gateway.md
```

### Recommended Learning Order

```
Minimal Path:
01 → 02 → 03 → 04 → 08

Standard Path (Recommended):
01 → 02 → 03 → 04 → 05 → 06 → 07 → 08 → 10

Complete Path:
01 → 02 → ... → 16 → 17 → 18 → 19 → 20 → 21 → 22 → 23 → 24 → 25
```

---

## Related Topics

- [Web Development](../Web_Development/00_Overview.md) -- HTTP fundamentals, client-server architecture
- [Backend Frameworks](../Backend_Frameworks/00_Overview.md) -- FastAPI, Flask, Django framework internals
- [Security](../Security/00_Overview.md) -- authentication protocols, TLS, OWASP API Top 10
- [Software Engineering](../Software_Engineering/00_Overview.md) -- design patterns, testing strategies, CI/CD

---

## Additional Resources

### Specifications and Standards
- [OpenAPI Specification 3.1](https://spec.openapis.org/oas/v3.1.0)
- [RFC 7807 -- Problem Details for HTTP APIs](https://datatracker.ietf.org/doc/html/rfc7807)
- [RFC 6749 -- OAuth 2.0 Authorization Framework](https://datatracker.ietf.org/doc/html/rfc6749)
- [JSON:API Specification](https://jsonapi.org/)

### Books
- "Designing Web APIs" by Brenda Jin, Saurabh Sahni, Amir Shevat
- "API Design Patterns" by JJ Geewax (Manning)
- "RESTful Web APIs" by Leonard Richardson, Mike Amundsen
- "The Design of Web APIs" by Arnaud Lauret

### Online References
- [Microsoft REST API Guidelines](https://github.com/microsoft/api-guidelines)
- [Google API Design Guide](https://cloud.google.com/apis/design)
- [Stripe API Reference](https://stripe.com/docs/api) -- widely regarded as best-in-class
- [GitHub REST API Docs](https://docs.github.com/en/rest)

### Tools
- [Swagger Editor](https://editor.swagger.io/) -- OpenAPI specification editor
- [Postman](https://www.postman.com/) -- API development and testing platform
- [Stoplight](https://stoplight.io/) -- API design-first platform
- [Hoppscotch](https://hoppscotch.io/) -- Open-source API testing

---

**License**: CC BY-NC 4.0
