# Rust Programming

## Topic Overview

Rust is a systems programming language focused on **safety**, **concurrency**, and **performance**. Its unique ownership system eliminates entire classes of bugs — data races, null pointer dereferences, dangling references — at compile time, with zero runtime cost. Rust consistently ranks as the most loved programming language in developer surveys and powers critical infrastructure at Mozilla, AWS, Google, Microsoft, and Cloudflare.

This topic assumes basic programming knowledge and covers Rust from first principles through advanced async programming. The ownership model (Lessons 03–05) is the conceptual core — everything else builds on it.

## Learning Path

```
Foundations                     Core Concepts                    Advanced
─────────────────              ─────────────────                ─────────────────
01 Getting Started             03 Ownership          ★★★       11 Lifetimes        ★★★★
02 Variables & Types           04 Borrowing          ★★★       13 Smart Pointers   ★★★
                               05 Slices             ★★        14 Concurrency      ★★★★
Data Modeling                  06 Structs & Methods  ★★        15 Async/Await      ★★★★
─────────────────              07 Enums & Patterns   ★★★       17 Unsafe Rust      ★★★★
08 Collections                 09 Error Handling     ★★★
10 Traits & Generics           12 Closures & Iters   ★★★       Project
16 Modules & Cargo                                              ─────────────────
                                                                18 CLI Tool Project ★★★
```

## Lesson List

| # | Lesson | Difficulty | Key Concepts |
|---|--------|------------|--------------|
| 01 | [Getting Started](./01_Getting_Started.md) | ⭐ | rustup, cargo, Hello World |
| 02 | [Variables and Types](./02_Variables_and_Types.md) | ⭐ | let/mut, shadowing, scalar/compound types |
| 03 | [Ownership](./03_Ownership.md) | ⭐⭐⭐ | Stack/heap, move semantics, Copy/Clone |
| 04 | [Borrowing and References](./04_Borrowing_and_References.md) | ⭐⭐⭐ | &T, &mut T, borrowing rules |
| 05 | [Slices](./05_Slices.md) | ⭐⭐ | &str vs String, array slices |
| 06 | [Structs and Methods](./06_Structs_and_Methods.md) | ⭐⭐ | struct, impl, #[derive] |
| 07 | [Enums and Pattern Matching](./07_Enums_and_Pattern_Matching.md) | ⭐⭐⭐ | enum, Option, match, if let |
| 08 | [Collections](./08_Collections.md) | ⭐⭐ | Vec, HashMap, Iterator chaining |
| 09 | [Error Handling](./09_Error_Handling.md) | ⭐⭐⭐ | Result, ?, thiserror/anyhow |
| 10 | [Traits and Generics](./10_Traits_and_Generics.md) | ⭐⭐⭐ | trait, impl Trait, generics, where clauses |
| 11 | [Lifetimes](./11_Lifetimes.md) | ⭐⭐⭐⭐ | Lifetime annotations, elision rules, 'static |
| 12 | [Closures and Iterators](./12_Closures_and_Iterators.md) | ⭐⭐⭐ | Fn/FnMut/FnOnce, map/filter/fold |
| 13 | [Smart Pointers](./13_Smart_Pointers.md) | ⭐⭐⭐ | Box, Rc, RefCell, Arc |
| 14 | [Concurrency](./14_Concurrency.md) | ⭐⭐⭐⭐ | thread::spawn, channels, Mutex, Send/Sync |
| 15 | [Async and Await](./15_Async_Await.md) | ⭐⭐⭐⭐ | async fn, Future, Tokio runtime |
| 16 | [Modules and Cargo](./16_Modules_and_Cargo.md) | ⭐⭐ | mod/use, Cargo.toml, workspaces |
| 17 | [Unsafe Rust](./17_Unsafe_Rust.md) | ⭐⭐⭐⭐ | unsafe blocks, raw pointers, FFI |
| 18 | [Project: CLI Tool](./18_Project_CLI_Tool.md) | ⭐⭐⭐ | clap + serde + tokio CLI project |

## Prerequisites

- Basic programming experience (variables, functions, control flow) in any language
- Familiarity with the command line
- Optional: C/C++ experience helps with understanding systems concepts

## Development Environment

```bash
# Install Rust via rustup
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Verify installation
rustc --version
cargo --version

# Useful components
rustup component add clippy      # Linter
rustup component add rustfmt     # Formatter
rustup component add rust-analyzer  # LSP (IDE support)
```

## Example Code

Runnable examples for this topic are in [`examples/Rust/`](../../../examples/Rust/).
