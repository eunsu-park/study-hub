# Rust Basics

Rust is a systems programming language focused on **safety**, **concurrency**, and **performance**. Its unique ownership system eliminates entire classes of bugs — data races, null pointer dereferences, dangling references — at compile time, with zero runtime cost. This topic covers Rust from first principles through concurrency, async programming, and project organization with Cargo.

## What You'll Learn

- **Getting Started**: rustup, cargo, toolchain setup
- **Variables and Types**: let/mut, shadowing, scalar and compound types
- **Ownership Model**: Stack/heap, move semantics, borrowing, references, slices
- **Data Modeling**: Structs, enums, pattern matching, collections
- **Error Handling**: Result, ?, thiserror/anyhow patterns
- **Traits and Generics**: Trait design, impl Trait, generic programming
- **Lifetimes**: Lifetime annotations, elision rules, 'static
- **Closures and Iterators**: Fn traits, map/filter/fold chains
- **Smart Pointers**: Box, Rc, RefCell, Arc
- **Concurrency**: Threads, channels, Mutex, Send/Sync
- **Async/Await**: async fn, Future, Tokio runtime
- **Modules and Cargo**: mod/use, workspaces, Cargo.toml

## Prerequisites

- [Programming](../Programming/00_Overview.md) — Familiarity with variables, functions, and control flow in any language

## Learning Roadmap

```
                          Rust Basics — Learning Path
  ┌─────────────────────────────────────────────────────────────────────────┐
  │                                                                         │
  │  ┌──────────────┐   ┌──────────────────┐   ┌────────────────────────┐  │
  │  │ 01 Getting    │──▶│ 02 Variables &   │──▶│ 03 Ownership           │  │
  │  │    Started    │   │    Types         │   │                        │  │
  │  └──────────────┘   └──────────────────┘   └────────────┬───────────┘  │
  │                                                          │              │
  │                                                          ▼              │
  │  ┌──────────────┐   ┌──────────────────┐   ┌────────────────────────┐  │
  │  │ 06 Structs & │◀──│ 05 Slices        │◀──│ 04 Borrowing &         │  │
  │  │    Methods   │   │                  │   │    References          │  │
  │  └──────┬───────┘   └──────────────────┘   └────────────────────────┘  │
  │         │                                                               │
  │         ▼                                                               │
  │  ┌──────────────┐   ┌──────────────────┐   ┌────────────────────────┐  │
  │  │ 07 Enums &   │──▶│ 08 Collections   │──▶│ 09 Error Handling      │  │
  │  │   Patterns   │   │                  │   │                        │  │
  │  └──────────────┘   └──────────────────┘   └────────────┬───────────┘  │
  │                                                          │              │
  │                                                          ▼              │
  │  ┌──────────────┐   ┌──────────────────┐   ┌────────────────────────┐  │
  │  │ 10 Traits &  │──▶│ 11 Lifetimes     │──▶│ 12 Closures &          │  │
  │  │   Generics   │   │                  │   │    Iterators           │  │
  │  └──────────────┘   └──────────────────┘   └────────────┬───────────┘  │
  │                                                          │              │
  │                                                          ▼              │
  │  ┌──────────────┐   ┌──────────────────┐   ┌────────────────────────┐  │
  │  │ 13 Smart     │──▶│ 14 Concurrency   │──▶│ 15 Async/Await         │  │
  │  │   Pointers   │   │                  │   │    ──▶ 16 Modules      │  │
  │  └──────────────┘   └──────────────────┘   └────────────────────────┘  │
  │                                                                         │
  └─────────────────────────────────────────────────────────────────────────┘
```

## Lessons

| # | Title | Difficulty | Key Content |
|---|-------|-----------|-------------|
| 01 | [Getting Started](01_Getting_Started.md) | ⭐ | rustup, cargo, Hello World |
| 02 | [Variables and Types](02_Variables_and_Types.md) | ⭐ | let/mut, shadowing, scalar/compound types |
| 03 | [Ownership](03_Ownership.md) | ⭐⭐⭐ | Stack/heap, move semantics, Copy/Clone |
| 04 | [Borrowing and References](04_Borrowing_and_References.md) | ⭐⭐⭐ | &T, &mut T, borrowing rules |
| 05 | [Slices](05_Slices.md) | ⭐⭐ | &str vs String, array slices |
| 06 | [Structs and Methods](06_Structs_and_Methods.md) | ⭐⭐ | struct, impl, #[derive] |
| 07 | [Enums and Pattern Matching](07_Enums_and_Pattern_Matching.md) | ⭐⭐⭐ | enum, Option, match, if let |
| 08 | [Collections](08_Collections.md) | ⭐⭐ | Vec, HashMap, Iterator chaining |
| 09 | [Error Handling](09_Error_Handling.md) | ⭐⭐⭐ | Result, ?, thiserror/anyhow |
| 10 | [Traits and Generics](10_Traits_and_Generics.md) | ⭐⭐⭐ | trait, impl Trait, generics, where clauses |
| 11 | [Lifetimes](11_Lifetimes.md) | ⭐⭐⭐⭐ | Lifetime annotations, elision rules, 'static |
| 12 | [Closures and Iterators](12_Closures_and_Iterators.md) | ⭐⭐⭐ | Fn/FnMut/FnOnce, map/filter/fold |
| 13 | [Smart Pointers](13_Smart_Pointers.md) | ⭐⭐⭐ | Box, Rc, RefCell, Arc |
| 14 | [Concurrency](14_Concurrency.md) | ⭐⭐⭐⭐ | thread::spawn, channels, Mutex, Send/Sync |
| 15 | [Async and Await](15_Async_Await.md) | ⭐⭐⭐⭐ | async fn, Future, Tokio runtime |
| 16 | [Modules and Cargo](16_Modules_and_Cargo.md) | ⭐⭐ | mod/use, Cargo.toml, workspaces |

## Development Environment

```bash
# Install Rust via rustup
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Verify installation
rustc --version
cargo --version

# Useful components
rustup component add clippy        # Linter
rustup component add rustfmt       # Formatter
rustup component add rust-analyzer # LSP (IDE support)
```

Example code for each lesson is available in `examples/Rust_Basics/`.

## Related Materials

- [Rust (Advanced)](../Rust_Advanced/00_Overview.md) — Unsafe, macros, FFI, WebAssembly, embedded, networking, and performance
- [Programming](../Programming/00_Overview.md) — Language-independent programming concepts

---

**License**: Content licensed under CC BY-NC 4.0
