# Rust Advanced

Building on Rust fundamentals, this topic covers advanced language features, systems programming, and the broader Rust ecosystem. You will work with unsafe code, macros, FFI, WebAssembly, embedded Rust, and build a production HTTP server as the capstone project.

## What You'll Learn

- **Unsafe Rust**: Raw pointers, unsafe blocks, and safety invariants
- **Macros**: Declarative (macro_rules!) and procedural (derive, attribute) macros
- **Advanced Traits**: GATs, trait objects, blanket impls, sealed traits
- **Advanced Async**: Tokio internals, select!, streams, Tower middleware
- **FFI and Interop**: C interop, bindgen/cbindgen, PyO3
- **WebAssembly**: wasm-pack, wasm-bindgen, WASI, Yew
- **Embedded Rust**: no_std, embedded-hal, RTIC
- **Networking**: TCP/UDP, Axum, WebSocket, TLS
- **Error Handling**: Advanced patterns with thiserror, anyhow, and recovery strategies
- **Performance**: criterion, flamegraph, SIMD, data-oriented design

## Prerequisites

- [Rust Basics](../Rust_Basics/00_Overview.md) — Rust ownership, traits, concurrency, async, and Cargo

## Learning Roadmap

```
                         Rust Advanced — Learning Path
  ┌─────────────────────────────────────────────────────────────────────────┐
  │                                                                         │
  │  Foundation                           Macros & Traits                   │
  │  ─────────────────                    ─────────────────                 │
  │  01 Unsafe Rust                       04 Declarative Macros             │
  │    ──▶ 02 CLI Tool (Project)          05 Procedural Macros              │
  │         ──▶ 03 Build System           06 Advanced Traits                │
  │                                       07 Advanced Async                 │
  │                                                                         │
  │  Ecosystem                            Operations                        │
  │  ─────────────────                    ─────────────────                 │
  │  08 FFI & Interop                     12 Advanced Error Handling        │
  │  09 WebAssembly                       13 Performance & Profiling        │
  │  10 Embedded Rust                                                       │
  │  11 Network Programming              Project                           │
  │                                       ─────────────────                 │
  │                                       14 Capstone: HTTP Server          │
  │                                                                         │
  └─────────────────────────────────────────────────────────────────────────┘
```

## Lessons

| # | Title | Difficulty | Key Content |
|---|-------|-----------|-------------|
| 01 | [Unsafe Rust](01_Unsafe_Rust.md) | ⭐⭐⭐⭐ | unsafe blocks, raw pointers, FFI |
| 02 | [Project: CLI Tool](02_Project_CLI_Tool.md) | ⭐⭐⭐ | clap + serde + tokio CLI project |
| 03 | [Build System Deep Dive](03_Build_System.md) | ⭐⭐⭐ | Workspaces, feature flags, build.rs, cross-compilation |
| 04 | [Declarative Macros](04_Declarative_Macros.md) | ⭐⭐⭐ | macro_rules!, repetition, fragment specifiers |
| 05 | [Procedural Macros](05_Procedural_Macros.md) | ⭐⭐⭐⭐ | derive macros, syn/quote, attribute macros |
| 06 | [Advanced Traits](06_Advanced_Traits.md) | ⭐⭐⭐⭐ | GATs, trait objects, blanket impls, sealed traits |
| 07 | [Advanced Async](07_Advanced_Async.md) | ⭐⭐⭐⭐ | Tokio internals, select!, streams, Tower |
| 08 | [FFI and Interop](08_FFI_and_Interop.md) | ⭐⭐⭐⭐ | C interop, bindgen/cbindgen, PyO3 |
| 09 | [WebAssembly](09_WebAssembly.md) | ⭐⭐⭐ | wasm-pack, wasm-bindgen, WASI, Yew |
| 10 | [Embedded Rust](10_Embedded_Rust.md) | ⭐⭐⭐⭐ | no_std, embedded-hal, RTIC, probe-rs |
| 11 | [Network Programming](11_Network_Programming.md) | ⭐⭐⭐ | TCP/UDP, Axum, WebSocket, TLS |
| 12 | [Advanced Error Handling](12_Advanced_Error_Handling.md) | ⭐⭐⭐ | thiserror, anyhow, recovery patterns |
| 13 | [Performance and Profiling](13_Performance_Profiling.md) | ⭐⭐⭐⭐ | criterion, flamegraph, SIMD, data-oriented design |
| 14 | [Capstone: HTTP Server](14_Capstone_HTTP_Server.md) | ⭐⭐⭐⭐ | Axum + SQLx + JWT + middleware project |

## Development Environment

```bash
rustc --version   # Rust 1.75+ recommended
cargo --version
```

Example code for each lesson is available in `examples/Rust_Advanced/`.

## Related Materials

- [Rust Basics](../Rust_Basics/00_Overview.md) — Rust fundamentals, ownership, traits, concurrency
- [C Advanced](../C_Advanced/00_Overview.md) — Systems programming in C for comparison
- [Linux](../Linux/00_Overview.md) — Linux systems knowledge for embedded and FFI work

---

**License**: Content licensed under CC BY-NC 4.0
