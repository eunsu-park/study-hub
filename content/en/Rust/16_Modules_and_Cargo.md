# 16. Modules and Cargo

**Previous**: [Async and Await](./15_Async_Await.md) | **Next**: [Unsafe Rust](./17_Unsafe_Rust.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Organize code into modules using `mod`, `pub`, `use`, `self`, `super`, and `crate`
2. Structure multi-file projects using both the `mod.rs` and named-file conventions
3. Control visibility with `pub`, `pub(crate)`, and `pub(super)` to enforce encapsulation
4. Configure `Cargo.toml` for dependencies, features, and workspaces
5. Use Cargo commands (`cargo add`, `cargo tree`, `cargo audit`) for dependency management

---

As your Rust programs grow beyond a single file, you need a way to split code into logical units and manage external libraries. Rust's **module system** handles the first concern — think of it as a filing cabinet where every drawer (module) has its own lock (visibility). **Cargo** handles the second — it is your build system, package manager, and project orchestrator all in one.

## Table of Contents
1. [The Module System](#1-the-module-system)
2. [File-Based Module Structure](#2-file-based-module-structure)
3. [Visibility Rules](#3-visibility-rules)
4. [Re-exports with pub use](#4-re-exports-with-pub-use)
5. [Cargo.toml in Depth](#5-cargotoml-in-depth)
6. [Cargo Workspaces](#6-cargo-workspaces)
7. [Publishing to crates.io](#7-publishing-to-cratesio)
8. [Useful Cargo Commands](#8-useful-cargo-commands)
9. [Feature Flags](#9-feature-flags)
10. [Practice Problems](#10-practice-problems)

---

## 1. The Module System

### 1.1 Defining Inline Modules

Modules group related items — functions, structs, enums, constants, traits, and even other modules:

```rust
// Modules create namespaces, similar to folders on a filesystem.
// Items inside are private by default — you must opt in to exposure.
mod math {
    pub fn add(a: i32, b: i32) -> i32 {
        a + b
    }

    pub fn multiply(a: i32, b: i32) -> i32 {
        a * b
    }

    // Private helper — only accessible within this module
    fn _validate(n: i32) -> bool {
        n >= 0
    }
}

fn main() {
    // Access public items through the module path
    let sum = math::add(3, 4);
    let product = math::multiply(3, 4);
    println!("{sum}, {product}"); // 7, 12
}
```

### 1.2 use, self, super, crate

These four keywords navigate the module tree:

```
crate (root)
├── main.rs          ← crate root
├── network          ← mod network
│   ├── client       ← mod client (child of network)
│   └── server       ← mod server (child of network)
└── utils            ← mod utils
```

```rust
mod network {
    pub mod client {
        pub fn connect() {
            println!("Client connecting...");
        }

        pub fn status() {
            // `self` refers to the current module (client)
            self::connect();
        }
    }

    pub mod server {
        pub fn listen() {
            // `super` goes up one level (to network)
            // then down into client
            super::client::connect();
            println!("Server listening...");
        }
    }
}

mod utils {
    pub fn log(msg: &str) {
        println!("[LOG] {msg}");
    }
}

fn main() {
    // `crate` refers to the root of the current crate
    crate::network::client::connect();
    crate::network::server::listen();

    // `use` brings items into scope to avoid long paths
    use network::client::connect;
    connect();

    // Rename on import to avoid conflicts
    use utils::log as write_log;
    write_log("startup complete");
}
```

### 1.3 Grouping Imports

```rust
// Import multiple items from the same module
use std::collections::{HashMap, HashSet, BTreeMap};

// Import a module and specific items from it
use std::io::{self, Read, Write};
// Now you can use both `io::Result` and `Read` directly

// Glob import — pulls everything public into scope (use sparingly)
use std::collections::*;
```

---

## 2. File-Based Module Structure

For real projects you split modules into separate files. Rust offers two conventions:

### 2.1 The mod.rs Convention (Edition 2015 style)

```
src/
├── main.rs
└── network/
    ├── mod.rs       ← declares the network module's contents
    ├── client.rs
    └── server.rs
```

```rust
// src/network/mod.rs
pub mod client;  // loads from network/client.rs
pub mod server;  // loads from network/server.rs
```

### 2.2 The Named-File Convention (Edition 2018+ preferred)

```
src/
├── main.rs
├── network.rs       ← declares the network module's contents
└── network/
    ├── client.rs
    └── server.rs
```

```rust
// src/network.rs — same role as mod.rs but lives beside the folder
pub mod client;
pub mod server;
```

Both conventions are equivalent. The 2018+ style avoids having many files all named `mod.rs` open in your editor tabs.

### 2.3 Declaring Modules from the Crate Root

```rust
// src/main.rs (or src/lib.rs for a library crate)
mod network;  // Rust looks for network.rs or network/mod.rs
mod utils;    // Rust looks for utils.rs

fn main() {
    network::client::connect();
    utils::log("started");
}
```

A key mental model: **`mod` is a declaration, not an import.** When you write `mod network;`, you are telling the compiler "there is a module called `network` — find its source file and compile it as part of this crate."

---

## 3. Visibility Rules

Rust defaults to **private**. You explicitly grant access:

| Keyword | Visible to |
|---------|-----------|
| (none) | Current module and its children only |
| `pub` | Everyone |
| `pub(crate)` | Anything within the same crate |
| `pub(super)` | The parent module |
| `pub(in path)` | A specific ancestor module |

```rust
mod database {
    // Public — any code that can see `database` can call this
    pub fn connect() -> Connection {
        Connection { pool: create_pool() }
    }

    // Crate-visible — other modules in this crate can use it,
    // but external crates cannot
    pub(crate) fn diagnostics() -> String {
        format!("pool size: {}", POOL_SIZE)
    }

    // Private — only code inside `database` (and its submodules)
    fn create_pool() -> Vec<()> {
        vec![(); POOL_SIZE]
    }

    const POOL_SIZE: usize = 10;

    pub struct Connection {
        // The field is private even though the struct is public.
        // External code can obtain a Connection via connect()
        // but cannot construct one directly.
        pool: Vec<()>,
    }

    impl Connection {
        pub fn query(&self, sql: &str) -> String {
            format!("Executing: {sql}")
        }
    }

    mod internal {
        pub(super) fn migrate() {
            // pub(super) makes this visible to `database` but
            // not to anything outside `database`
            println!("Running migrations...");
        }
    }
}

fn main() {
    let conn = database::connect();
    println!("{}", conn.query("SELECT 1"));

    // This works — pub(crate) is visible within the same crate
    println!("{}", database::diagnostics());

    // These would NOT compile:
    // database::create_pool();           // private
    // database::internal::migrate();     // pub(super), only visible to database
    // let c = database::Connection { pool: vec![] };  // field is private
}
```

---

## 4. Re-exports with pub use

Re-exports let you present a clean public API that hides internal structure:

```rust
// Without re-exports, users must navigate your internal structure:
//   mycrate::parsers::json::JsonParser
//   mycrate::parsers::csv::CsvParser

// With re-exports, you flatten the API surface:
//   mycrate::JsonParser
//   mycrate::CsvParser

mod parsers {
    pub mod json {
        pub struct JsonParser;
        impl JsonParser {
            pub fn parse(input: &str) -> String {
                format!("Parsed JSON: {input}")
            }
        }
    }

    pub mod csv {
        pub struct CsvParser;
        impl CsvParser {
            pub fn parse(input: &str) -> String {
                format!("Parsed CSV: {input}")
            }
        }
    }
}

// Re-export at the crate root so users get a clean API
pub use parsers::json::JsonParser;
pub use parsers::csv::CsvParser;

fn main() {
    // Users can now access directly without knowing internal paths
    let result = JsonParser::parse(r#"{"key": "value"}"#);
    println!("{result}");
}
```

The `prelude` pattern is a common application: library crates often have a `prelude` module that re-exports the most commonly used types.

```rust
// In a library crate:
// pub mod prelude {
//     pub use crate::JsonParser;
//     pub use crate::CsvParser;
//     pub use crate::Error;
// }
//
// Users then write:
//   use mycrate::prelude::*;
```

---

## 5. Cargo.toml in Depth

`Cargo.toml` is the manifest file for every Rust project. Here is an annotated example:

```toml
[package]
name = "my-app"
version = "0.1.0"
edition = "2021"           # Rust edition (2015, 2018, 2021, 2024)
authors = ["Alice <alice@example.com>"]
description = "A sample application"
license = "MIT"
repository = "https://github.com/alice/my-app"
rust-version = "1.75"     # Minimum supported Rust version

[dependencies]
serde = { version = "1.0", features = ["derive"] }   # with feature flag
tokio = { version = "1", features = ["full"] }
log = "0.4"                 # shorthand: version only

[dev-dependencies]
# Only used for tests, examples, and benchmarks
criterion = "0.5"
tempfile = "3"

[build-dependencies]
# Only used by build.rs (build script)
cc = "1.0"

[profile.release]
opt-level = 3        # Maximum optimization
lto = true           # Link-time optimization — slower build, faster binary
strip = true         # Strip debug symbols — smaller binary
```

### Version Requirements

```
"1.0"      →  >=1.0.0, <2.0.0   (caret, default)
"~1.0"     →  >=1.0.0, <1.1.0   (tilde — patch-level only)
"=1.0.3"   →  exactly 1.0.3
">=1.0, <1.5"  → range
"*"        →  any version (avoid in published crates)
```

---

## 6. Cargo Workspaces

Workspaces let you manage multiple related crates in a single repository. They share a single `Cargo.lock` and output directory, which speeds up builds and keeps dependency versions consistent.

```
my-project/
├── Cargo.toml          ← workspace root
├── crates/
│   ├── core/
│   │   ├── Cargo.toml
│   │   └── src/lib.rs
│   ├── cli/
│   │   ├── Cargo.toml
│   │   └── src/main.rs
│   └── web/
│       ├── Cargo.toml
│       └── src/main.rs
```

```toml
# Root Cargo.toml — workspace definition
[workspace]
members = [
    "crates/core",
    "crates/cli",
    "crates/web",
]
resolver = "2"   # Use the v2 dependency resolver (recommended)

# Shared dependencies — member crates inherit these versions
[workspace.dependencies]
serde = { version = "1.0", features = ["derive"] }
tokio = { version = "1", features = ["full"] }
```

```toml
# crates/cli/Cargo.toml
[package]
name = "my-cli"
version = "0.1.0"
edition = "2021"

[dependencies]
# Reference a sibling crate by path
my-core = { path = "../core" }
# Inherit version from workspace — keeps all crates in sync
serde.workspace = true
```

```bash
# Build a specific crate
cargo build -p my-cli

# Run tests across the entire workspace
cargo test --workspace

# Run a specific binary
cargo run -p my-cli
```

---

## 7. Publishing to crates.io

```bash
# 1. Create an account at https://crates.io and get an API token
cargo login <your-token>

# 2. Verify your package is ready
cargo package        # creates a .crate archive — review what gets included
cargo package --list # list files that will be published

# 3. Publish
cargo publish

# 4. Yank a version (if you publish a broken release)
cargo yank --version 0.1.0    # discourages new projects from using it
cargo yank --undo --version 0.1.0  # undo the yank
```

Important rules for crates.io:
- Package names are globally unique and first-come-first-served
- Published versions are **permanent** — you cannot overwrite or delete them
- Yanking prevents new `Cargo.lock` files from selecting the version, but existing projects that already depend on it are unaffected
- You need `description` and `license` fields in `Cargo.toml` to publish

---

## 8. Useful Cargo Commands

```bash
# --- Dependency management ---
cargo add serde --features derive    # add a dependency (edits Cargo.toml)
cargo add tokio -F full              # -F is short for --features
cargo add --dev criterion            # add to [dev-dependencies]
cargo remove serde                   # remove a dependency

# --- Inspection ---
cargo tree                    # show dependency tree
cargo tree -d                 # show only duplicate dependencies
cargo tree -i serde           # invert: who depends on serde?

# --- Security ---
cargo audit                   # check for known vulnerabilities (install: cargo install cargo-audit)
cargo deny check              # policy-based dependency linting (install: cargo install cargo-deny)

# --- Quality ---
cargo clippy                  # lint: catches common mistakes and suggests improvements
cargo fmt                     # format code according to rustfmt rules
cargo doc --open              # generate and open HTML documentation

# --- Build & run ---
cargo build --release         # optimized build
cargo run -- arg1 arg2        # run with arguments (-- separates cargo args from program args)
cargo test                    # run all tests
cargo bench                   # run benchmarks
cargo check                   # type-check without producing a binary (fastest feedback)
```

---

## 9. Feature Flags

Features let users opt in to functionality at compile time. This keeps the default build lightweight while offering optional capabilities.

```toml
# Cargo.toml for a library crate
[package]
name = "my-logger"
version = "0.1.0"
edition = "2021"

[features]
# No features are enabled by default
default = []

# Individual features
color = ["dep:colored"]        # enables the `colored` dependency
json = ["dep:serde_json", "dep:serde"]
file = []                      # a feature with no extra dependencies
full = ["color", "json", "file"]  # a convenience feature that enables everything

[dependencies]
colored = { version = "2", optional = true }      # only compiled when `color` feature is active
serde = { version = "1", features = ["derive"], optional = true }
serde_json = { version = "1", optional = true }
```

```rust
// src/lib.rs — use cfg attributes to conditionally compile code

pub fn log(message: &str) {
    // Base functionality always available
    let timestamp = "2025-01-15T10:30:00";

    #[cfg(feature = "color")]
    {
        use colored::Colorize;
        println!("{} {}", timestamp.dimmed(), message.green());
        return;
    }

    #[cfg(not(feature = "color"))]
    println!("{timestamp} {message}");
}

#[cfg(feature = "json")]
pub fn log_json(message: &str) {
    // This entire function only exists when the `json` feature is enabled
    use serde_json::json;
    let entry = json!({
        "timestamp": "2025-01-15T10:30:00",
        "message": message,
    });
    println!("{}", entry);
}
```

```bash
# Users choose features when adding the dependency:
cargo add my-logger --features color,json

# Or in their Cargo.toml:
# [dependencies]
# my-logger = { version = "0.1", features = ["color", "json"] }
```

---

## 10. Practice Problems

### Exercise 1: Module Hierarchy
Create a project with the following module structure using files (not inline modules):

```
src/
├── main.rs
├── shapes.rs
└── shapes/
    ├── circle.rs
    └── rectangle.rs
```

Each shape module should define a struct with an `area(&self) -> f64` method. The `shapes.rs` file should re-export both structs. `main.rs` should use them via `shapes::Circle` and `shapes::Rectangle`.

### Exercise 2: Visibility Challenge
Create a `bank` module with an `Account` struct. The `balance` field must be private. Provide `pub fn new(owner: &str, initial: f64) -> Account`, `pub fn deposit(&mut self, amount: f64)`, `pub fn balance(&self) -> f64`, and a `pub(crate)` method `audit_log(&self) -> String`. Verify that code outside the `bank` module cannot directly access `balance` but can call `audit_log`.

### Exercise 3: Workspace Setup
Design (on paper or on disk) a Cargo workspace with three crates: `math-core` (library with pure math functions), `math-cli` (binary that uses `math-core` and `clap` for argument parsing), and `math-web` (binary that uses `math-core` and `axum` for a REST API). Write the three `Cargo.toml` files and the root workspace `Cargo.toml`. Use `workspace.dependencies` for shared versions.

### Exercise 4: Feature Flags
Create a library crate `greeter` with two optional features: `formal` and `loud`. Without features, `greet(name)` returns `"Hello, {name}"`. With `formal`, it returns `"Good day, {name}. How do you do?"`. With `loud`, the output is uppercased. Both features can be active simultaneously. Write the `Cargo.toml`, `lib.rs`, and a test for each combination.

### Exercise 5: Dependency Audit
Take any Rust project (your own or clone one from GitHub). Run `cargo tree` and `cargo tree -d` to find duplicate dependencies. Write a short report: how many total dependencies? How many duplicates? What is the deepest dependency chain? Then run `cargo audit` (install it first if needed) and note any advisories.

---

## References
- [The Rust Book — Modules](https://doc.rust-lang.org/book/ch07-00-managing-growing-projects-with-packages-crates-and-modules.html)
- [Cargo Book — Specifying Dependencies](https://doc.rust-lang.org/cargo/reference/specifying-dependencies.html)
- [Cargo Book — Workspaces](https://doc.rust-lang.org/cargo/reference/workspaces.html)
- [Cargo Book — Features](https://doc.rust-lang.org/cargo/reference/features.html)
- [crates.io publishing guide](https://doc.rust-lang.org/cargo/reference/publishing.html)
- [cargo-audit](https://github.com/rustsec/rustsec/tree/main/cargo-audit)

---

**Previous**: [Async and Await](./15_Async_Await.md) | **Next**: [Unsafe Rust](./17_Unsafe_Rust.md)
