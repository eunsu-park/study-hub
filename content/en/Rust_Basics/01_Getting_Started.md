# 01. Getting Started

**Previous**: [Overview](./00_Overview.md) | **Next**: [Variables and Types](./02_Variables_and_Types.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Install and manage Rust toolchains using `rustup`
2. Create, build, and run projects with `cargo`
3. Explain the compile-then-run execution model and how it differs from interpreted languages
4. Write a basic Rust program with `fn main()` and `println!`

---

Rust's toolchain is refreshingly unified. Where C/C++ projects juggle Makefiles, CMake, and system-specific compilers, Rust provides a single tool — **Cargo** — that handles project creation, dependency management, building, testing, and publishing.

## Table of Contents
1. [Installing Rust](#1-installing-rust)
2. [Hello, World!](#2-hello-world)
3. [Cargo: The Rust Build System](#3-cargo-the-rust-build-system)
4. [Anatomy of a Cargo Project](#4-anatomy-of-a-cargo-project)
5. [Practice Problems](#5-practice-problems)

---

## 1. Installing Rust

### 1.1 rustup

`rustup` is the official installer and version manager. It manages compiler versions, standard libraries, and cross-compilation targets.

```bash
# Install Rust (Unix/macOS)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install Rust (Windows) — download rustup-init.exe from https://rustup.rs

# Check installed versions
rustc --version      # Compiler
cargo --version      # Build tool / package manager
rustup --version     # Toolchain manager
```

### 1.2 Toolchain Channels

Rust uses three release channels:

| Channel | Release Cycle | Use Case |
|---------|--------------|----------|
| **stable** | Every 6 weeks | Production code |
| **beta** | 6 weeks before stable | Testing upcoming features |
| **nightly** | Daily | Experimental features (`#![feature(...)]`) |

```bash
# Switch default toolchain
rustup default stable
rustup default nightly

# Install a specific version
rustup install 1.82.0

# Use nightly for one command
rustup run nightly cargo build
```

### 1.3 Essential Components

```bash
# Linter — catches common mistakes and anti-patterns
rustup component add clippy
cargo clippy

# Formatter — enforces consistent style
rustup component add rustfmt
cargo fmt

# Language Server — IDE support (VS Code, Neovim, etc.)
rustup component add rust-analyzer

# Documentation — generate and open docs for your project + dependencies
cargo doc --open
```

---

## 2. Hello, World!

### 2.1 Without Cargo

```rust
// hello.rs
fn main() {
    println!("Hello, world!");
}
```

```bash
# Compile and run
rustc hello.rs     # Produces ./hello (or hello.exe on Windows)
./hello            # Hello, world!
```

Key observations:
- `fn main()` is the entry point — every Rust binary has exactly one
- `println!` is a **macro** (note the `!`), not a function — macros generate code at compile time
- Semicolons terminate statements (Rust is expression-based, but statement-ending `;` is required)

### 2.2 Compiled vs Interpreted

```
Rust (compiled):
  source.rs  →  rustc  →  native binary  →  OS executes directly
                          (no runtime needed)

Python (interpreted):
  script.py  →  python interpreter  →  bytecode  →  VM executes
                (runtime always needed)
```

Rust produces standalone executables linked against the system's C library (by default). There is no garbage collector, no VM, and no runtime — just machine code.

---

## 3. Cargo: The Rust Build System

### 3.1 Creating a Project

```bash
# Create a new binary project
cargo new my_project
# Creates:
# my_project/
# ├── Cargo.toml    # Manifest (metadata + dependencies)
# └── src/
#     └── main.rs   # Entry point

# Create a library project
cargo new my_lib --lib
# Creates src/lib.rs instead of src/main.rs
```

### 3.2 Essential Commands

```bash
cargo build            # Compile (debug mode, target/debug/)
cargo build --release  # Compile optimized (target/release/)
cargo run              # Build + run
cargo run --release    # Build optimized + run
cargo check            # Type-check without producing binary (fast!)
cargo test             # Run tests
cargo doc --open       # Generate and open documentation
cargo clippy           # Run linter
cargo fmt              # Format code
cargo update           # Update dependencies to latest compatible
cargo clean            # Remove target/ directory
```

> **Tip**: `cargo check` is much faster than `cargo build` because it skips code generation. Use it frequently during development to catch type errors early.

### 3.3 Debug vs Release Builds

| Aspect | `cargo build` (debug) | `cargo build --release` |
|--------|----------------------|------------------------|
| Optimization | None (`opt-level = 0`) | Full (`opt-level = 3`) |
| Compile speed | Fast | Slow |
| Runtime speed | Slow (no optimizations) | Fast |
| Debug info | Included | Stripped |
| Output directory | `target/debug/` | `target/release/` |
| Use case | Development | Production / benchmarking |

---

## 4. Anatomy of a Cargo Project

### 4.1 Cargo.toml

```toml
[package]
name = "my_project"
version = "0.1.0"
edition = "2021"       # Rust edition (2015, 2018, 2021, 2024)
authors = ["Your Name <you@example.com>"]
description = "A brief description"
license = "MIT"

[dependencies]
serde = { version = "1.0", features = ["derive"] }
tokio = { version = "1", features = ["full"] }
clap = "4"

[dev-dependencies]
assert_cmd = "2"

[profile.release]
opt-level = 3
lto = true             # Link-Time Optimization
```

### 4.2 Editions

Rust **editions** (2015, 2018, 2021, 2024) let the language evolve without breaking existing code. Each edition can introduce new keywords or syntax changes, but crates using different editions interoperate seamlessly.

```
Edition 2015: Initial syntax
Edition 2018: async/await keywords, module path changes
Edition 2021: Closure capture improvements, IntoIterator for arrays
Edition 2024: gen blocks, unsafe_op_in_unsafe_fn lint
```

### 4.3 Project Layout Conventions

```
my_project/
├── Cargo.toml
├── Cargo.lock          # Exact dependency versions (commit for binaries)
├── src/
│   ├── main.rs         # Binary entry point
│   ├── lib.rs          # Library root (optional)
│   └── bin/
│       └── other.rs    # Additional binaries
├── tests/
│   └── integration.rs  # Integration tests
├── benches/
│   └── bench.rs        # Benchmarks
└── examples/
    └── demo.rs         # Example programs (cargo run --example demo)
```

---

## 5. Practice Problems

### Exercise 1: Project Setup
Create a new Cargo project, add `rand = "0.8"` as a dependency, and write a program that prints a random number between 1 and 100.

<details>
<summary>Hint</summary>

Use `rand::Rng` trait and `thread_rng().gen_range(1..=100)`.
</details>

### Exercise 2: Multiple Binaries
Create a project with two binaries: `src/main.rs` that prints "main binary" and `src/bin/greet.rs` that prints "Hello from greet!". Run each with `cargo run` and `cargo run --bin greet`.

### Exercise 3: Cargo Features
Add `serde = { version = "1.0", features = ["derive"] }` to your project. Explain the difference between `cargo build` and `cargo build --release` in terms of compilation speed and runtime performance.

### Exercise 4: Toolchain Exploration
Run `cargo clippy` on your project. Intentionally introduce a warning (e.g., unused variable) and observe clippy's suggestion. Then run `cargo fmt` and compare the formatting changes.

---

## References
- [The Rust Programming Language (The Book)](https://doc.rust-lang.org/book/)
- [Cargo Book](https://doc.rust-lang.org/cargo/)
- [rustup Documentation](https://rust-lang.github.io/rustup/)

---

**Previous**: [Overview](./00_Overview.md) | **Next**: [Variables and Types](./02_Variables_and_Types.md)
