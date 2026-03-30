# C# Basics

This topic covers fundamental C# programming from environment setup through generics, exception handling, and file I/O. Whether you are learning your first object-oriented language or transitioning from Python, Java, or C++, these lessons will give you a solid understanding of the C# type system, the .NET runtime, and modern C# idioms. C# is a versatile, strongly-typed language used for desktop applications, web services, cloud computing, game development with Unity, and cross-platform mobile apps with .NET MAUI.

## What You'll Learn

This topic provides hands-on coverage of:

- **Getting Started**: .NET SDK installation, `dotnet` CLI, Hello World, project structure, and the compilation pipeline
- **Variables and Types**: Value types, reference types, `var`, `const`, nullable types, and type conversions
- **Operators and Expressions**: Arithmetic, comparison, logical, bitwise, null-coalescing, and checked arithmetic
- **Control Flow**: `if`/`else`, `switch` statements and expressions, `for`, `foreach`, `while`, pattern matching basics
- **Methods**: Parameter passing (`ref`/`out`/`in`/`params`), overloading, local functions, recursion, and tuple returns
- **Arrays and Strings**: Single and multidimensional arrays, jagged arrays, string interpolation, `StringBuilder`
- **Enums and Structs**: Enumerations, flags, structs, and value vs reference semantics
- **Collections**: `List<T>`, `Dictionary<TKey,TValue>`, `HashSet<T>`, `Queue<T>`, `Stack<T>`, and LINQ basics
- **Classes and Objects**: Constructors, fields, properties, access modifiers, and static members
- **Properties and Indexers**: Auto-properties, init-only setters, computed properties, and indexers
- **Inheritance**: `virtual`/`override`/`abstract`, `sealed`, `base`, type checking with `is`/`as`
- **Interfaces**: Interface design, default interface methods, and multiple implementation
- **Generics**: Generic classes and methods, constraints, covariance and contravariance
- **Exception Handling**: `try`/`catch`/`finally`, custom exceptions, exception filters
- **File I/O**: `File` and `Stream` APIs, text and binary I/O, the `using` statement

## Prerequisites

- [Programming](../Programming/00_Overview.md) — Familiarity with general programming concepts (variables, control flow, functions)

No prior C# experience is required. If you understand what a variable, loop, and function are in any language, you are ready.

## Learning Roadmap

```
                         C# Basics — Learning Path
  ┌─────────────────────────────────────────────────────────────────────────────┐
  │                                                                             │
  │  ┌──────────────┐   ┌──────────────────┐   ┌────────────────────────────┐  │
  │  │ 01 Getting   │──▶│ 02 Variables &   │──▶│ 03 Operators &             │  │
  │  │    Started   │   │    Types         │   │    Expressions             │  │
  │  └──────────────┘   └──────────────────┘   └────────────┬───────────────┘  │
  │                                                          │                  │
  │                                                          ▼                  │
  │  ┌──────────────┐   ┌──────────────────┐   ┌────────────────────────────┐  │
  │  │ 06 Arrays &  │◀──│ 05 Methods       │◀──│ 04 Control Flow            │  │
  │  │    Strings   │   │                  │   │                            │  │
  │  └──────┬───────┘   └──────────────────┘   └────────────────────────────┘  │
  │         │                                                                   │
  │         ▼                                                                   │
  │  ┌──────────────┐   ┌──────────────────┐   ┌────────────────────────────┐  │
  │  │ 07 Enums &   │──▶│ 08 Collections   │──▶│ 09 Classes &               │  │
  │  │    Structs   │   │                  │   │    Objects                 │  │
  │  └──────────────┘   └──────────────────┘   └────────────┬───────────────┘  │
  │                                                          │                  │
  │                                                          ▼                  │
  │  ┌──────────────┐   ┌──────────────────┐   ┌────────────────────────────┐  │
  │  │ 12 Interfaces│◀──│ 11 Inheritance   │◀──│ 10 Properties &            │  │
  │  │              │   │                  │   │    Indexers                │  │
  │  └──────┬───────┘   └──────────────────┘   └────────────────────────────┘  │
  │         │                                                                   │
  │         ▼                                                                   │
  │  ┌──────────────┐   ┌──────────────────┐   ┌────────────────────────────┐  │
  │  │ 13 Generics  │──▶│ 14 Exception     │──▶│ 15 File I/O                │  │
  │  │              │   │    Handling      │   │                            │  │
  │  └──────────────┘   └──────────────────┘   └────────────────────────────┘  │
  │                                                                             │
  └─────────────────────────────────────────────────────────────────────────────┘
```

## Lessons

| # | Title | Difficulty | Key Content |
|---|-------|-----------|-------------|
| 01 | [Getting Started](01_Getting_Started.md) | ⭐ | .NET SDK, dotnet CLI, Hello World, project structure |
| 02 | [Variables and Types](02_Variables_and_Types.md) | ⭐ | Value types, reference types, var, const, type conversions |
| 03 | [Operators and Expressions](03_Operators_and_Expressions.md) | ⭐ | Arithmetic, comparison, logical, bitwise, precedence |
| 04 | [Control Flow](04_Control_Flow.md) | ⭐ | if/else, switch, for, foreach, while, break/continue |
| 05 | [Methods](05_Methods.md) | ⭐ | Parameters (ref/out/in/params), overloading, recursion |
| 06 | [Arrays and Strings](06_Arrays_and_Strings.md) | ⭐⭐ | Arrays, multidimensional, jagged, string interpolation, StringBuilder |
| 07 | [Enums and Structs](07_Enums_and_Structs.md) | ⭐⭐ | Enums, flags, structs, value vs reference semantics |
| 08 | [Collections](08_Collections.md) | ⭐⭐ | List, Dictionary, HashSet, Queue, Stack, LINQ basics |
| 09 | [Classes and Objects](09_Classes_and_Objects.md) | ⭐⭐ | Constructors, fields, access modifiers, static members |
| 10 | [Properties and Indexers](10_Properties_and_Indexers.md) | ⭐⭐ | Auto-properties, init-only, computed, indexers |
| 11 | [Inheritance](11_Inheritance.md) | ⭐⭐⭐ | virtual/override/abstract, sealed, base, is/as |
| 12 | [Interfaces](12_Interfaces.md) | ⭐⭐⭐ | Interface design, default methods, multiple implementation |
| 13 | [Generics](13_Generics.md) | ⭐⭐⭐ | Generic classes/methods, constraints, covariance/contravariance |
| 14 | [Exception Handling](14_Exception_Handling.md) | ⭐⭐ | try/catch/finally, custom exceptions, exception filters |
| 15 | [File I/O](15_File_IO.md) | ⭐⭐ | File/Stream APIs, text and binary I/O, using statement |

## Recommended Learning Order

Follow the lessons sequentially from 01 through 15. Each lesson builds on concepts introduced in the previous one:

1. **Getting Started (Lesson 1)**: Install the .NET SDK and run your first C# program
2. **Language Fundamentals (Lessons 2-4)**: Variables, operators, and control flow form the backbone of every C# program
3. **Methods (Lesson 5)**: Organize code into reusable functions with flexible parameter passing
4. **Data Structures (Lessons 6-8)**: Work with arrays, strings, enums, structs, and collection classes
5. **Object-Oriented Programming (Lessons 9-12)**: Classes, properties, inheritance, and interfaces
6. **Generics (Lesson 13)**: Write type-safe, reusable code with generic classes and methods
7. **Error Handling and I/O (Lessons 14-15)**: Handle exceptions gracefully and work with the file system

## Practice Environment

Verify your .NET SDK installation:

```bash
dotnet --version
# 8.0.x (or newer)

# Quick test
dotnet new console -n test_app && cd test_app && dotnet run
# Output: Hello, World!
cd .. && rm -rf test_app
```

Example code for each lesson is available in `examples/CSharp_Basics/`.

## Related Materials

- [C# Advanced](../CSharp_Advanced/00_Overview.md) — Async/await, LINQ in depth, delegates, events, and advanced patterns
- [Programming](../Programming/00_Overview.md) — Language-independent programming concepts
- [Software Engineering](../Software_Engineering/00_Overview.md) — Design principles and patterns applicable to C#
- [Web Development](../Web_Development/00_Overview.md) — Building web applications with ASP.NET Core

---

**License**: Content licensed under CC BY-NC 4.0
