# CSharp Advanced

C# Advanced is a comprehensive course that takes you beyond the fundamentals of C# programming into the powerful, expressive features that make C# one of the most productive languages for building modern software. This course covers advanced type system features, asynchronous programming, performance optimization, and real-world architectural patterns used in professional .NET development. By the end of this course, you will be equipped to write idiomatic, high-performance, and maintainable C# code for any domain — from cloud services to systems programming.

---

## What You'll Learn

- **Functional programming patterns** — delegates, events, lambdas, closures, and LINQ for declarative data processing
- **Modern type system features** — pattern matching, nullable reference types, records, and immutability
- **Asynchronous programming** — async/await, Task-based patterns, cancellation, and ValueTask
- **Concurrency and parallelism** — threads, Parallel, concurrent collections, channels, and synchronization
- **Memory-efficient programming** — Span\<T\>, Memory\<T\>, stackalloc, and allocation reduction techniques
- **Software engineering practices** — dependency injection, serialization, testing, and NuGet packaging
- **Low-level capabilities** — reflection, attributes, P/Invoke, unsafe code, and source generators
- **Performance profiling** — BenchmarkDotNet, dotnet-counters, and allocation analysis
- **Full-stack integration** — building a production-ready Minimal Web API with EF Core, auth, and testing

---

## Prerequisites

- [CSharp Basics](../CSharp_Basics/00_Overview.md) — You should be comfortable with C# syntax, control flow, OOP (classes, interfaces, inheritance), generics, collections, and basic exception handling.

---

## Learning Roadmap

```
                        ┌─────────────────────┐
                        │   00  Overview       │
                        └─────────┬───────────┘
                                  │
              ┌───────────────────┼───────────────────┐
              ▼                   ▼                   ▼
     ┌────────────────┐  ┌────────────────┐  ┌────────────────┐
     │ 01 Delegates & │  │ 04 Pattern     │  │ 05 Nullable    │
     │    Events      │  │    Matching    │  │    Reference   │
     └───────┬────────┘  └───────┬────────┘  └───────┬────────┘
             │                   │                   │
             ▼                   │                   ▼
     ┌────────────────┐          │           ┌────────────────┐
     │ 02 Lambda &    │          │           │ 06 Records &   │
     │    Closures    │          │           │    Immutability│
     └───────┬────────┘          │           └───────┬────────┘
             │                   │                   │
             ▼                   │                   │
     ┌────────────────┐          │                   │
     │ 03 LINQ        │◄─────────┘                   │
     └───────┬────────┘                              │
             │                                       │
             └───────────────┬───────────────────────┘
                             ▼
                    ┌────────────────┐
                    │ 07 Async/Await │
                    └───────┬────────┘
                            │
                            ▼
                    ┌────────────────┐
                    │ 08 Concurrency │
                    │ & Parallelism  │
                    └───────┬────────┘
                            │
                            ▼
                    ┌────────────────┐
                    │ 09 Spans &     │
                    │    Memory      │
                    └───────┬────────┘
                            │
         ┌──────────────────┼──────────────────┐
         ▼                  ▼                  ▼
┌────────────────┐ ┌────────────────┐ ┌────────────────┐
│ 10 Dependency  │ │ 11 Serializa-  │ │ 12 Testing     │
│    Injection   │ │    tion        │ │                │
└───────┬────────┘ └───────┬────────┘ └───────┬────────┘
        │                  │                  │
        └──────────────────┼──────────────────┘
                           ▼
                  ┌────────────────┐
                  │ 13 NuGet &     │
                  │ Project System │
                  └───────┬────────┘
                          │
              ┌───────────┼───────────┐
              ▼                       ▼
     ┌────────────────┐      ┌────────────────┐
     │ 14 Reflection  │      │ 15 Interop &   │
     │ & Attributes   │      │    Unsafe      │
     └───────┬────────┘      └───────┬────────┘
             │                       │
             └───────────┬───────────┘
                         ▼
                ┌────────────────┐
                │ 16 Performance │
                │ & Profiling    │
                └───────┬────────┘
                        │
                        ▼
                ┌────────────────┐
                │ 17 Capstone:   │
                │ Minimal WebAPI │
                └────────────────┘
```

---

## Lessons

| # | Title | Difficulty | Key Content |
|---|-------|-----------|-------------|
| [01](./01_Delegates_and_Events.md) | Delegates and Events | ⭐⭐⭐ | Delegates, Action/Func, multicast, events, EventHandler |
| [02](./02_Lambda_and_Closures.md) | Lambda Expressions and Closures | ⭐⭐⭐ | Lambda syntax, closures, expression-bodied, local functions |
| [03](./03_LINQ.md) | LINQ | ⭐⭐⭐ | Query/method syntax, deferred execution, custom operators |
| [04](./04_Pattern_Matching.md) | Pattern Matching | ⭐⭐⭐ | Type/property/positional/relational/list patterns |
| [05](./05_Nullable_Reference_Types.md) | Nullable Reference Types | ⭐⭐⭐ | Nullable context, flow analysis, null guards |
| [06](./06_Records_and_Immutability.md) | Records and Immutability | ⭐⭐⭐ | record class, record struct, with-expressions, immutability |
| [07](./07_Async_Await.md) | Async and Await | ⭐⭐⭐⭐ | Task, async/await, cancellation, ValueTask |
| [08](./08_Concurrency_and_Parallelism.md) | Concurrency and Parallelism | ⭐⭐⭐⭐ | Thread, Parallel, ConcurrentCollections, lock, channels |
| [09](./09_Spans_and_Memory.md) | Spans and Memory | ⭐⭐⭐⭐ | Span\<T\>, Memory\<T\>, stackalloc, reducing allocations |
| [10](./10_Dependency_Injection.md) | Dependency Injection | ⭐⭐⭐ | DI container, service lifetimes, Microsoft.Extensions.DI |
| [11](./11_Serialization.md) | Serialization | ⭐⭐ | System.Text.Json, attributes, source generators |
| [12](./12_Testing.md) | Testing | ⭐⭐ | xUnit, Arrange-Act-Assert, mocking, integration tests |
| [13](./13_NuGet_and_Project_System.md) | NuGet and Project System | ⭐⭐ | .csproj, NuGet, multi-targeting, central package management |
| [14](./14_Reflection_and_Attributes.md) | Reflection and Attributes | ⭐⭐⭐ | Custom attributes, reflection API, source generators |
| [15](./15_Interop_and_Unsafe.md) | Interop and Unsafe | ⭐⭐⭐⭐ | P/Invoke, unsafe, pointers, LibraryImport |
| [16](./16_Performance_Profiling.md) | Performance and Profiling | ⭐⭐⭐ | BenchmarkDotNet, dotnet-counters, allocation analysis |
| [17](./17_Capstone_Web_API.md) | Capstone: Minimal Web API | ⭐⭐⭐⭐ | ASP.NET Core minimal API, EF Core, auth, testing |
