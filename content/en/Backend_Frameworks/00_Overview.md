# Backend Frameworks

## Topic Overview

Backend frameworks provide the structure for building server-side applications — routing HTTP requests, connecting to databases, authenticating users, and serving APIs. This topic covers three of the most popular frameworks across two ecosystems: **FastAPI** (Python), **Express** (Node.js), and **Django** (Python). Rather than teaching each in isolation, we compare their approaches to common problems, helping you choose the right tool and transfer knowledge between frameworks.

The course follows a deliberate progression: framework-specific deep dives (Lessons 02–13), then cross-cutting patterns that apply everywhere (Lessons 14–17), a capstone project (Lesson 18), and three extension lessons (Lessons 19–21) covering Go web development, Redis caching, and job queues.

## Learning Path

```
Fundamentals            Framework Deep Dives              Cross-Cutting Patterns
──────────────         ──────────────────────             ─────────────────────
01 Web Fundamentals    02-05 FastAPI (Python)             14 API Design Patterns
                       06-09 Express (Node.js)            15 Authentication Patterns
                       10-13 Django (Python)              16 Production Deployment
                                                          17 Observability

                                                          Project
                                                          ─────────────────────
                                                          18 REST API Project

                                                          Extensions
                                                          ─────────────────────
                                                          19 Go Web Basics
                                                          20 Redis Caching Patterns
                                                          21 Job Queues
```

## Lesson List

| # | Lesson | Difficulty | Key Concepts |
|---|--------|------------|--------------|
| 01 | [Backend Web Fundamentals](./01_Backend_Web_Fundamentals.md) | ⭐⭐ | HTTP, WSGI/ASGI, REST principles |
| 02 | [FastAPI Basics](./02_FastAPI_Basics.md) | ⭐⭐ | Path/query params, Pydantic v2, OpenAPI |
| 03 | [FastAPI Advanced](./03_FastAPI_Advanced.md) | ⭐⭐⭐ | DI, OAuth2/JWT, background tasks |
| 04 | [FastAPI Database](./04_FastAPI_Database.md) | ⭐⭐⭐ | SQLAlchemy 2.0 async, Alembic |
| 05 | [FastAPI Testing](./05_FastAPI_Testing.md) | ⭐⭐⭐ | TestClient, pytest-asyncio |
| 06 | [Express Basics](./06_Express_Basics.md) | ⭐⭐ | Routing, middleware chain |
| 07 | [Express Advanced](./07_Express_Advanced.md) | ⭐⭐⭐ | Error middleware, Passport.js, rate limiting |
| 08 | [Express Database](./08_Express_Database.md) | ⭐⭐⭐ | Prisma ORM + PostgreSQL |
| 09 | [Express Testing](./09_Express_Testing.md) | ⭐⭐⭐ | Supertest, Jest |
| 10 | [Django Basics](./10_Django_Basics.md) | ⭐⭐ | MTV pattern, models, views, admin |
| 11 | [Django ORM](./11_Django_ORM.md) | ⭐⭐⭐ | QuerySet, F/Q objects, select_related |
| 12 | [Django REST Framework](./12_Django_REST_Framework.md) | ⭐⭐⭐ | Serializers, ViewSets |
| 13 | [Django Advanced](./13_Django_Advanced.md) | ⭐⭐⭐ | Channels, Celery, Redis |
| 14 | [API Design Patterns](./14_API_Design_Patterns.md) | ⭐⭐⭐ | RESTful design, versioning, pagination |
| 15 | [Authentication Patterns](./15_Authentication_Patterns.md) | ⭐⭐⭐ | JWT vs sessions vs API keys, OAuth2 |
| 16 | [Production Deployment](./16_Production_Deployment.md) | ⭐⭐⭐⭐ | Gunicorn/uvicorn, PM2, nginx, Docker |
| 17 | [Observability](./17_Observability.md) | ⭐⭐⭐ | Structured logging, Prometheus, OpenTelemetry |
| 18 | [Project: REST API](./18_Project_REST_API.md) | ⭐⭐⭐⭐ | FastAPI blog API — full implementation |
| 19 | [Go Web Basics](./19_Go_Web_Basics.md) | ⭐⭐⭐ | Go net/http, Gin framework, middleware, goroutine concurrency |
| 20 | [Redis Caching Patterns](./20_Redis_Caching_Patterns.md) | ⭐⭐⭐ | Redis data types, caching strategies, pub/sub, session store |
| 21 | [Job Queues](./21_Job_Queues.md) | ⭐⭐⭐ | Celery, BullMQ, task scheduling, retry policies, dead letter queues |

## Prerequisites

- Python basics (variables, functions, classes, async/await)
- JavaScript/Node.js basics (ES6+, promises, async/await)
- HTTP fundamentals (methods, status codes, headers)
- SQL basics (SELECT, INSERT, JOIN)
- Command line proficiency

## Example Code

Runnable examples for this topic are in [`examples/Backend_Frameworks/`](../../../examples/Backend_Frameworks/).

```
examples/Backend_Frameworks/
├── fastapi/     # FastAPI examples
├── express/     # Express.js examples
└── django/      # Django examples
```
