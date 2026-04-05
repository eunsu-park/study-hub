# Go Advanced

Building on Go fundamentals, this topic covers practical application development and advanced language features. You will build HTTP servers, REST APIs, CLI tools, and a complete microservice — while learning generics, reflection, profiling, and cloud-native patterns.

## What You'll Learn

- **Web Development**: HTTP servers, REST APIs, middleware, and routing
- **Database Access**: database/sql, connection pooling, and migrations
- **CLI Tools**: cobra, flag, interactive TUI
- **Advanced Types**: Generics, type constraints, and generic data structures
- **Reflection**: reflect package, struct tags, and code generation
- **Performance**: pprof, trace, benchmarking, and memory optimization
- **Build and Deploy**: Cross-compilation, Docker, CI/CD, release
- **Networking**: TCP/UDP, WebSocket, gRPC
- **Cloud Native**: Context, health checks, graceful shutdown, 12-factor apps

## Prerequisites

- [Go Basics](../Go_Basics/00_Overview.md) — Go fundamentals, concurrency, and standard library

## Learning Roadmap

```
                         Go Advanced — Learning Path
  ┌─────────────────────────────────────────────────────────────────────────┐
  │                                                                         │
  │  Application Building                Advanced Language                  │
  │  ─────────────────                   ─────────────────                  │
  │  ┌──────────────┐                    ┌──────────────────┐              │
  │  │ 01 HTTP       │──▶ 02 REST API    │ 05 Advanced Types │              │
  │  │    Server     │   ──▶ 03 Database │    (Generics)     │              │
  │  └──────────────┘       ──▶ 04 CLI  └────────┬─────────┘              │
  │                                               │                        │
  │                                               ▼                        │
  │  Operations                          ┌──────────────────┐              │
  │  ─────────────────                   │ 06 Reflection &   │              │
  │  07 Profiling ──▶ 08 Build & Deploy  │    Code Gen       │              │
  │                                      └──────────────────┘              │
  │  Network & Cloud                                                       │
  │  ─────────────────                   Project                           │
  │  09 Network ──▶ 10 Cloud Native      ─────────────────                 │
  │                  ──▶ 11 Capstone     11 Capstone Microservice          │
  │                                                                         │
  └─────────────────────────────────────────────────────────────────────────┘
```

## Lessons

| # | Title | Difficulty | Key Content |
|---|-------|-----------|-------------|
| 01 | [HTTP Server](01_HTTP_Server.md) | ⭐⭐ | net/http server, routing, middleware, handlers |
| 02 | [REST API](02_REST_API.md) | ⭐⭐⭐ | Building REST APIs, JSON handling, validation |
| 03 | [Database Access](03_Database_Access.md) | ⭐⭐⭐ | database/sql, connection pooling, migrations |
| 04 | [CLI Tools](04_CLI_Tools.md) | ⭐⭐ | cobra, flag, stdin/stdout, interactive TUI |
| 05 | [Advanced Types](05_Advanced_Types.md) | ⭐⭐⭐ | Generics, type constraints, generic data structures |
| 06 | [Reflection and Code Generation](06_Reflection_and_Codegen.md) | ⭐⭐⭐⭐ | reflect package, struct tags, code generation |
| 07 | [Performance Profiling](07_Performance_Profiling.md) | ⭐⭐⭐ | pprof, trace, benchmarking, memory optimization |
| 08 | [Build and Deploy](08_Build_and_Deploy.md) | ⭐⭐ | Cross-compilation, Docker, CI/CD, release |
| 09 | [Network Programming](09_Network_Programming.md) | ⭐⭐⭐ | TCP/UDP, WebSocket, gRPC |
| 10 | [Cloud Native Patterns](10_Cloud_Native_Patterns.md) | ⭐⭐⭐ | Context, health checks, graceful shutdown, 12-factor |
| 11 | [Capstone: Microservice](11_Capstone_Microservice.md) | ⭐⭐⭐⭐ | Build a complete microservice with all patterns |

## Environment Setup

```bash
go version   # Go 1.22+ recommended
```

Example code for each lesson is available in `examples/Go_Advanced/`.

## Related Materials

- [Go Basics](../Go_Basics/00_Overview.md) — Go fundamentals, concurrency primitives, and standard library
- [System Design](../System_Design/00_Overview.md) — Architecture patterns for distributed systems
- [Docker](../Docker/00_Overview.md) — Containerization for Go services
- [Kubernetes](../Kubernetes/00_Overview.md) — Orchestration for cloud-native Go services

---

**License**: Content licensed under CC BY-NC 4.0
