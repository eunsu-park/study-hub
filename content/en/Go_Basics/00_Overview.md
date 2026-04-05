# Go Basics

Go (Golang) is a statically typed, compiled programming language designed at Google. Its design philosophy emphasizes **simplicity**, **readability**, and **practical productivity**. This topic covers Go fundamentals — from variables and types through concurrency primitives (goroutines, channels) to testing and the standard library.

## What You'll Learn

- **Language Fundamentals**: Variables, types, control flow, and basic syntax
- **Composite Types**: Arrays, slices, maps, and structs
- **Functions and Methods**: Function types, receivers, closures, and method sets
- **Interfaces**: Duck typing, type assertions, and interface design patterns
- **Error Handling**: Error type, custom errors, wrapping, and sentinel errors
- **Packages and Modules**: Go modules, dependency management, and project structure
- **Concurrency**: Goroutines, channels, and concurrency patterns (fan-in/out, worker pools)
- **Testing**: Table-driven tests, benchmarks, and fuzzing
- **Standard Library**: io, net/http, encoding/json, os, filepath

## Prerequisites

- [Programming](../Programming/00_Overview.md) — Familiarity with variables, functions, and control flow in any language

## Learning Roadmap

```
                          Go Basics — Learning Path
  ┌─────────────────────────────────────────────────────────────────────────┐
  │                                                                         │
  │  ┌──────────────┐   ┌──────────────────┐   ┌────────────────────────┐  │
  │  │ 01 Go         │──▶│ 02 Composite     │──▶│ 03 Functions &         │  │
  │  │    Fundamentals│   │    Types         │   │    Methods             │  │
  │  └──────────────┘   └──────────────────┘   └────────────┬───────────┘  │
  │                                                          │              │
  │                                                          ▼              │
  │  ┌──────────────┐   ┌──────────────────┐   ┌────────────────────────┐  │
  │  │ 06 Packages & │◀──│ 05 Error         │◀──│ 04 Interfaces          │  │
  │  │    Modules    │   │    Handling      │   │                        │  │
  │  └──────┬───────┘   └──────────────────┘   └────────────────────────┘  │
  │         │                                                               │
  │         ▼                                                               │
  │  ┌──────────────┐   ┌──────────────────┐   ┌────────────────────────┐  │
  │  │ 07 Goroutines │──▶│ 08 Channels      │──▶│ 09 Concurrency         │  │
  │  │              │   │                  │   │    Patterns             │  │
  │  └──────────────┘   └──────────────────┘   └────────────┬───────────┘  │
  │                                                          │              │
  │                                                          ▼              │
  │  ┌──────────────┐   ┌──────────────────┐                               │
  │  │ 10 Testing   │──▶│ 11 Standard      │                               │
  │  │              │   │    Library       │                               │
  │  └──────────────┘   └──────────────────┘                               │
  │                                                                         │
  └─────────────────────────────────────────────────────────────────────────┘
```

## Lessons

| # | Title | Difficulty | Key Content |
|---|-------|-----------|-------------|
| 01 | [Go Fundamentals](01_Go_Fundamentals.md) | ⭐ | Variables, types, control flow, functions |
| 02 | [Composite Types](02_Composite_Types.md) | ⭐⭐ | Arrays, slices, maps, structs |
| 03 | [Functions and Methods](03_Functions_and_Methods.md) | ⭐⭐ | Function types, methods, receivers, closures |
| 04 | [Interfaces](04_Interfaces.md) | ⭐⭐⭐ | Interface design, duck typing, type assertions |
| 05 | [Error Handling](05_Error_Handling.md) | ⭐⭐ | Error type, custom errors, wrapping, sentinel errors |
| 06 | [Packages and Modules](06_Packages_and_Modules.md) | ⭐⭐ | Go modules, package design, dependency management |
| 07 | [Concurrency: Goroutines](07_Concurrency_Goroutines.md) | ⭐⭐⭐ | Goroutines, WaitGroup, sync primitives |
| 08 | [Channels](08_Channels.md) | ⭐⭐⭐ | Channel types, buffered/unbuffered, select |
| 09 | [Concurrency Patterns](09_Concurrency_Patterns.md) | ⭐⭐⭐⭐ | Fan-in/out, pipeline, worker pool, context |
| 10 | [Testing](10_Testing.md) | ⭐⭐ | testing package, table-driven tests, benchmarks, fuzzing |
| 11 | [Standard Library](11_Standard_Library.md) | ⭐⭐ | io, net/http, encoding/json, os, filepath |

## Environment Setup

```bash
# Install Go (macOS with Homebrew)
brew install go

# Install Go (Linux)
wget https://go.dev/dl/go1.22.0.linux-amd64.tar.gz
sudo tar -C /usr/local -xzf go1.22.0.linux-amd64.tar.gz
export PATH=$PATH:/usr/local/go/bin

# Verify installation
go version
go env GOPATH
```

Example code for each lesson is available in `examples/Go_Basics/`.

## Related Materials

- [Go (Advanced)](../Go_Advanced/00_Overview.md) — HTTP servers, REST APIs, generics, reflection, cloud-native patterns
- [Programming](../Programming/00_Overview.md) — Language-independent programming concepts

---

**License**: Content licensed under CC BY-NC 4.0
