# 19. Build System Deep Dive

**Previous**: [Project: CLI Tool](./02_Project_CLI_Tool.md) | **Next**: [Declarative Macros](./04_Declarative_Macros.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Organize large Rust projects with Cargo workspaces and multi-crate architectures
2. Use feature flags to enable conditional compilation and optional dependencies
3. Write build scripts (`build.rs`) that generate code and link native libraries
4. Set up cross-compilation for different target platforms
5. Define custom Cargo profiles to tune optimization, debug info, and other compiler settings
6. Manage complex dependency graphs using patch, replace, and alternative registries
7. Prepare and publish a crate to crates.io with proper metadata and versioning

---

## Table of Contents

1. [Cargo Workspaces and Multi-Crate Projects](#1-cargo-workspaces-and-multi-crate-projects)
2. [Feature Flags and Conditional Compilation](#2-feature-flags-and-conditional-compilation)
3. [Build Scripts (build.rs)](#3-build-scripts-buildrs)
4. [Cross-Compilation Targets](#4-cross-compilation-targets)
5. [Custom Profiles](#5-custom-profiles)
6. [Dependencies Management](#6-dependencies-management)
7. [Publishing to crates.io](#7-publishing-to-cratesio)

---

## 1. Cargo Workspaces and Multi-Crate Projects

As a Rust project grows, splitting it into multiple crates improves compilation times, enforces API boundaries, and makes each piece independently testable. A **workspace** is a set of crates that share a common `Cargo.lock` and output directory.

### 1.1 Creating a Workspace

A workspace is defined by a root `Cargo.toml` that contains a `[workspace]` table:

```toml
# workspace root: Cargo.toml
[workspace]
members = [
    "core",
    "cli",
    "server",
    "shared-utils",
]
resolver = "2"   # Recommended for edition 2021+
```

The root `Cargo.toml` does **not** have a `[package]` section (unless the root itself is also a crate). Each member directory has its own `Cargo.toml`:

```
my-project/
├── Cargo.toml          # workspace root
├── Cargo.lock          # shared lock file
├── core/
│   ├── Cargo.toml      # [package] name = "my-core"
│   └── src/lib.rs
├── cli/
│   ├── Cargo.toml      # [package] name = "my-cli"
│   └── src/main.rs
├── server/
│   ├── Cargo.toml      # [package] name = "my-server"
│   └── src/main.rs
└── shared-utils/
    ├── Cargo.toml      # [package] name = "shared-utils"
    └── src/lib.rs
```

### 1.2 Inter-Crate Dependencies

Members reference each other with **path dependencies**:

```toml
# cli/Cargo.toml
[package]
name = "my-cli"
version = "0.1.0"
edition = "2021"

[dependencies]
my-core = { path = "../core" }
shared-utils = { path = "../shared-utils" }
```

Path dependencies are resolved at compile time. When you publish, you replace the path with a version requirement (or use both):

```toml
[dependencies]
my-core = { version = "0.1", path = "../core" }
```

### 1.3 Workspace-Level Dependencies

Since Rust 1.64, you can declare shared dependency versions at the workspace level to avoid duplication:

```toml
# workspace root: Cargo.toml
[workspace]
members = ["core", "cli", "server"]

[workspace.dependencies]
serde = { version = "1.0", features = ["derive"] }
tokio = { version = "1", features = ["full"] }
anyhow = "1.0"
```

Each member then inherits:

```toml
# cli/Cargo.toml
[dependencies]
serde = { workspace = true }
tokio = { workspace = true }
anyhow = { workspace = true }
my-core = { path = "../core" }
```

This guarantees every crate uses exactly the same version of `serde`, `tokio`, etc.

### 1.4 Running Commands in a Workspace

```bash
# Build everything
cargo build

# Build a specific member
cargo build -p my-cli

# Run tests for a specific member
cargo test -p my-core

# Run tests for the entire workspace
cargo test --workspace

# Run a specific binary
cargo run -p my-cli -- --help
```

### 1.5 Virtual vs. Non-Virtual Workspaces

A **virtual workspace** has no `[package]` in the root — it exists only to group members. A **non-virtual workspace** is a regular package whose `Cargo.toml` also contains `[workspace]`:

```toml
# Non-virtual: root is itself a crate
[package]
name = "my-app"
version = "0.1.0"
edition = "2021"

[workspace]
members = ["plugins/*"]
```

Virtual workspaces are more common in large projects because the root acts purely as an organizer.

### 1.6 Glob Patterns for Members

You can use globs to include multiple crates:

```toml
[workspace]
members = [
    "crates/*",
    "tools/*",
    "examples/*",
]
exclude = [
    "crates/experimental-unstable",
]
```

---

## 2. Feature Flags and Conditional Compilation

Feature flags allow crate authors to expose optional functionality. Consumers choose which features to enable, keeping binary size and compile times minimal.

### 2.1 Declaring Features

```toml
# core/Cargo.toml
[package]
name = "my-core"
version = "0.1.0"
edition = "2021"

[features]
default = ["json"]          # Enabled unless the consumer opts out
json = ["dep:serde_json"]   # Enables serde_json dependency
xml = ["dep:quick-xml"]     # Enables quick-xml dependency
async = ["dep:tokio"]       # Enables async support
full = ["json", "xml", "async"]  # Convenience umbrella feature

[dependencies]
serde = { version = "1.0", features = ["derive"] }
serde_json = { version = "1.0", optional = true }
quick-xml = { version = "0.31", optional = true }
tokio = { version = "1", features = ["full"], optional = true }
```

The `dep:` prefix (stabilized in Rust 1.60) prevents an optional dependency from implicitly creating a feature with the same name.

### 2.2 Using `cfg` Attributes

Inside your code, conditionally compile based on features:

```rust
// core/src/lib.rs

pub struct Config {
    pub name: String,
    pub value: String,
}

#[cfg(feature = "json")]
pub mod json_support {
    use super::Config;
    use serde::{Deserialize, Serialize};

    #[derive(Serialize, Deserialize)]
    struct ConfigJson {
        name: String,
        value: String,
    }

    pub fn to_json(config: &Config) -> Result<String, serde_json::Error> {
        let cj = ConfigJson {
            name: config.name.clone(),
            value: config.value.clone(),
        };
        serde_json::to_string_pretty(&cj)
    }

    pub fn from_json(data: &str) -> Result<Config, serde_json::Error> {
        let cj: ConfigJson = serde_json::from_str(data)?;
        Ok(Config {
            name: cj.name,
            value: cj.value,
        })
    }
}

#[cfg(feature = "xml")]
pub mod xml_support {
    use super::Config;

    pub fn to_xml(config: &Config) -> String {
        format!(
            "<config><name>{}</name><value>{}</value></config>",
            config.name, config.value
        )
    }
}
```

### 2.3 Conditional Compilation Beyond Features

The `cfg` system goes beyond feature flags:

```rust
// Target OS
#[cfg(target_os = "linux")]
fn platform_specific() {
    println!("Running on Linux");
}

#[cfg(target_os = "windows")]
fn platform_specific() {
    println!("Running on Windows");
}

#[cfg(target_os = "macos")]
fn platform_specific() {
    println!("Running on macOS");
}

// Target architecture
#[cfg(target_arch = "x86_64")]
fn simd_optimized() {
    println!("Using x86_64 SIMD instructions");
}

#[cfg(target_arch = "aarch64")]
fn simd_optimized() {
    println!("Using ARM NEON instructions");
}

// Combining conditions
#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
fn linux_amd64_only() {
    println!("Linux x86_64 specific code");
}

#[cfg(any(target_os = "linux", target_os = "macos"))]
fn unix_like() {
    println!("Running on a Unix-like system");
}

#[cfg(not(target_os = "windows"))]
fn non_windows() {
    println!("Not Windows");
}
```

### 2.4 `cfg_attr` for Conditional Attributes

Apply attributes only when a condition is met:

```rust
// Only derive Serialize when the json feature is on
#[cfg_attr(feature = "json", derive(serde::Serialize, serde::Deserialize))]
pub struct Metadata {
    pub author: String,
    pub version: u32,
}

// Conditional lint suppression
#[cfg_attr(test, allow(dead_code))]
fn internal_helper() -> u32 {
    42
}
```

### 2.5 Enabling Features from Consumers

```toml
# In a consumer Cargo.toml
[dependencies]
my-core = { version = "0.1", features = ["json", "async"] }

# Or disable default features and pick explicitly
my-core = { version = "0.1", default-features = false, features = ["xml"] }
```

### 2.6 Feature Unification

Cargo **unifies** features across the dependency graph. If crate A enables `serde/derive` and crate B enables `serde/alloc`, the final build includes both. This means features must be **additive** — enabling a feature should never break existing functionality.

```rust
// BAD: Feature that removes functionality
#[cfg(not(feature = "logging"))]
pub fn process(data: &[u8]) -> Vec<u8> {
    // fast path without logging
    data.to_vec()
}

#[cfg(feature = "logging")]
pub fn process(data: &[u8]) -> Vec<u8> {
    // This changes the function signature/behavior — violates additivity
    log::info!("Processing {} bytes", data.len());
    data.to_vec()
}

// GOOD: Feature that adds functionality
pub fn process(data: &[u8]) -> Vec<u8> {
    #[cfg(feature = "logging")]
    log::info!("Processing {} bytes", data.len());
    data.to_vec()
}
```

---

## 3. Build Scripts (build.rs)

A **build script** is a Rust file named `build.rs` placed at the crate root (next to `Cargo.toml`). Cargo compiles and runs it **before** building the crate. Build scripts can generate code, link native libraries, set environment variables, and more.

### 3.1 Basic Build Script

```rust
// build.rs
fn main() {
    // Tell Cargo to re-run this script if build.rs itself changes
    println!("cargo::rerun-if-changed=build.rs");

    // Set an environment variable accessible via env!() in the crate
    println!("cargo::rustc-env=BUILD_TIMESTAMP={}", chrono_like_now());
}

fn chrono_like_now() -> String {
    // In practice you would use the `chrono` crate or std::time
    "2026-03-16T00:00:00Z".to_string()
}
```

Access it in your code:

```rust
fn main() {
    println!("Built at: {}", env!("BUILD_TIMESTAMP"));
}
```

### 3.2 Cargo Instruction Reference

Build scripts communicate with Cargo via `println!("cargo::...")` directives:

```rust
fn main() {
    // Re-run triggers
    println!("cargo::rerun-if-changed=src/data.json");
    println!("cargo::rerun-if-changed=wrapper.h");
    println!("cargo::rerun-if-env-changed=MY_CONFIG_VAR");

    // Link a native library
    println!("cargo::rustc-link-lib=sqlite3");
    println!("cargo::rustc-link-search=native=/usr/local/lib");

    // Pass cfg flags to the compiler
    println!("cargo::rustc-cfg=has_avx2");

    // Set environment variables for downstream code
    println!("cargo::rustc-env=GIT_HASH=abc123");

    // Warning message (shows during build)
    println!("cargo::warning=Using fallback configuration");
}
```

> **Note**: Older versions used `cargo:` with a single colon. The `cargo::` double-colon syntax is preferred since Rust 1.77.

### 3.3 Generating Code

A common use case is generating Rust source files at build time:

```rust
// build.rs
use std::env;
use std::fs;
use std::path::Path;

fn main() {
    println!("cargo::rerun-if-changed=build.rs");
    println!("cargo::rerun-if-changed=data/commands.txt");

    let out_dir = env::var("OUT_DIR").unwrap();
    let dest_path = Path::new(&out_dir).join("generated_commands.rs");

    // Read a list of commands from a data file
    let commands = fs::read_to_string("data/commands.txt")
        .unwrap_or_else(|_| "help\nversion\nquit".to_string());

    let mut code = String::from("pub const COMMANDS: &[&str] = &[\n");
    for cmd in commands.lines() {
        let trimmed = cmd.trim();
        if !trimmed.is_empty() {
            code.push_str(&format!("    \"{trimmed}\",\n"));
        }
    }
    code.push_str("];\n");

    fs::write(&dest_path, code).unwrap();
}
```

Include the generated file in your crate:

```rust
// src/lib.rs
include!(concat!(env!("OUT_DIR"), "/generated_commands.rs"));

pub fn is_valid_command(input: &str) -> bool {
    COMMANDS.contains(&input)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_commands_exist() {
        assert!(is_valid_command("help"));
        assert!(is_valid_command("version"));
        assert!(!is_valid_command("unknown"));
    }
}
```

### 3.4 Linking Native C Libraries

Build scripts are essential for FFI. Here is a pattern using the `cc` crate:

```toml
# Cargo.toml
[build-dependencies]
cc = "1.0"
```

```c
// csrc/fast_math.c
#include <math.h>

double fast_hypot(double x, double y) {
    return sqrt(x * x + y * y);
}
```

```rust
// build.rs
fn main() {
    println!("cargo::rerun-if-changed=csrc/fast_math.c");

    cc::Build::new()
        .file("csrc/fast_math.c")
        .opt_level(3)
        .warnings(true)
        .compile("fast_math");
}
```

```rust
// src/lib.rs
extern "C" {
    fn fast_hypot(x: f64, y: f64) -> f64;
}

pub fn hypot(x: f64, y: f64) -> f64 {
    unsafe { fast_hypot(x, y) }
}
```

### 3.5 Using `bindgen` for Automatic FFI Bindings

For complex C headers, the `bindgen` crate generates Rust FFI bindings automatically:

```toml
# Cargo.toml
[build-dependencies]
bindgen = "0.70"
```

```rust
// build.rs
use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo::rerun-if-changed=wrapper.h");

    let bindings = bindgen::Builder::default()
        .header("wrapper.h")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("Unable to generate bindings");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings");
}
```

```rust
// src/lib.rs
#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
```

### 3.6 Build Script Dependencies

Build scripts can have their own dependencies, separate from the crate's runtime dependencies:

```toml
[build-dependencies]
cc = "1.0"
bindgen = "0.70"
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
```

These are only compiled and linked for the build script — they do not affect the final binary.

---

## 4. Cross-Compilation Targets

Rust's cross-compilation story is one of its major strengths. The compiler can produce binaries for a wide range of platforms.

### 4.1 Listing Available Targets

```bash
# List all supported targets
rustup target list

# List only installed targets
rustup target list --installed

# Add a new target
rustup target add aarch64-unknown-linux-gnu
rustup target add x86_64-unknown-linux-musl
rustup target add wasm32-unknown-unknown
rustup target add x86_64-pc-windows-gnu
```

### 4.2 Building for a Different Target

```bash
# Build for ARM Linux (e.g., Raspberry Pi)
cargo build --target aarch64-unknown-linux-gnu

# Build a statically linked Linux binary (using musl)
cargo build --target x86_64-unknown-linux-musl --release

# Build for WebAssembly
cargo build --target wasm32-unknown-unknown --release
```

### 4.3 Configuring Linkers

Cross-compilation often requires a cross-linker. Configure it in `.cargo/config.toml`:

```toml
# .cargo/config.toml

[target.aarch64-unknown-linux-gnu]
linker = "aarch64-linux-gnu-gcc"

[target.x86_64-unknown-linux-musl]
linker = "x86_64-linux-musl-gcc"

[target.x86_64-pc-windows-gnu]
linker = "x86_64-w64-mingw32-gcc"

# You can also set a default target for this project
[build]
target = "x86_64-unknown-linux-gnu"
```

### 4.4 Target-Specific Dependencies

```toml
# Cargo.toml

# Dependency only for Linux
[target.'cfg(target_os = "linux")'.dependencies]
inotify = "0.10"

# Dependency only for Windows
[target.'cfg(target_os = "windows")'.dependencies]
winapi = { version = "0.3", features = ["winuser"] }

# Dependency only for macOS
[target.'cfg(target_os = "macos")'.dependencies]
cocoa = "0.25"

# Dependency for any Unix-like system
[target.'cfg(unix)'.dependencies]
nix = "0.28"
```

### 4.5 Platform-Specific Code Patterns

```rust
pub fn get_config_dir() -> std::path::PathBuf {
    #[cfg(target_os = "linux")]
    {
        let home = std::env::var("HOME").unwrap_or_else(|_| "/root".into());
        std::path::PathBuf::from(home).join(".config").join("myapp")
    }

    #[cfg(target_os = "macos")]
    {
        let home = std::env::var("HOME").unwrap_or_else(|_| "/Users/default".into());
        std::path::PathBuf::from(home)
            .join("Library")
            .join("Application Support")
            .join("myapp")
    }

    #[cfg(target_os = "windows")]
    {
        let appdata = std::env::var("APPDATA").unwrap_or_else(|_| "C:\\Users".into());
        std::path::PathBuf::from(appdata).join("myapp")
    }
}

// Use the `dirs` crate in production for cross-platform paths.
```

### 4.6 Cross-Compilation with Docker

For reproducible cross-compilation environments, a multi-stage Dockerfile works well:

```dockerfile
# Example: Build a statically linked binary for Linux
FROM rust:1.77-slim AS builder
RUN rustup target add x86_64-unknown-linux-musl
RUN apt-get update && apt-get install -y musl-tools
WORKDIR /app
COPY . .
RUN cargo build --release --target x86_64-unknown-linux-musl

FROM alpine:3.19
COPY --from=builder /app/target/x86_64-unknown-linux-musl/release/myapp /usr/local/bin/
ENTRYPOINT ["myapp"]
```

### 4.7 Checking Multiple Targets in CI

```yaml
# .github/workflows/cross.yml
name: Cross-platform CI
on: [push, pull_request]

jobs:
  build:
    strategy:
      matrix:
        include:
          - target: x86_64-unknown-linux-gnu
            os: ubuntu-latest
          - target: x86_64-apple-darwin
            os: macos-latest
          - target: x86_64-pc-windows-msvc
            os: windows-latest
          - target: aarch64-unknown-linux-gnu
            os: ubuntu-latest
    runs-on: ${{ matrix.os }}
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
        with:
          targets: ${{ matrix.target }}
      - name: Install cross-compiler (ARM Linux)
        if: matrix.target == 'aarch64-unknown-linux-gnu'
        run: |
          sudo apt-get update
          sudo apt-get install -y gcc-aarch64-linux-gnu
      - run: cargo build --target ${{ matrix.target }} --release
```

---

## 5. Custom Profiles

Cargo comes with two built-in profiles: `dev` (for development) and `release` (for production). You can customize both and create entirely new profiles.

### 5.1 Built-in Profile Defaults

| Setting | `dev` | `release` |
|---|---|---|
| `opt-level` | 0 | 3 |
| `debug` | true | false |
| `debug-assertions` | true | false |
| `overflow-checks` | true | false |
| `lto` | false | false |
| `codegen-units` | 256 | 16 |
| `incremental` | true | false |
| `strip` | none | none |

### 5.2 Customizing Built-in Profiles

```toml
# Cargo.toml

# Faster dev builds: use opt-level 1 for dependencies
[profile.dev]
opt-level = 0

[profile.dev.package."*"]
opt-level = 1       # Optimize dependencies but not your code

# Maximize release performance
[profile.release]
opt-level = 3
lto = "fat"          # Full link-time optimization
codegen-units = 1    # Single codegen unit for maximum optimization
strip = "symbols"    # Strip debug symbols from the binary
panic = "abort"      # Smaller binary, no unwinding
```

### 5.3 Optimization Levels

| Level | Meaning |
|---|---|
| `0` | No optimization (fastest compile) |
| `1` | Basic optimizations |
| `2` | Most optimizations |
| `3` | All optimizations including vectorization |
| `"s"` | Optimize for binary size |
| `"z"` | Optimize for binary size aggressively (may be slower) |

```toml
# Optimize for size — useful for embedded or WASM
[profile.release]
opt-level = "z"
lto = true
codegen-units = 1
strip = true
```

### 5.4 Link-Time Optimization (LTO)

LTO enables the compiler to optimize across crate boundaries:

```toml
[profile.release]
# "fat" — full cross-crate LTO (slowest compile, best optimization)
lto = "fat"

# "thin" — parallel LTO with most of the benefit of "fat"
# lto = "thin"

# false — no LTO
# lto = false
```

Typical speedup from `thin` LTO is 10–20%, with `fat` LTO adding a few more percent at the cost of significantly longer compile times.

### 5.5 Custom Named Profiles

Create profiles that inherit from built-in ones:

```toml
# Cargo.toml

# A profile for CI testing: release-like but with debug assertions
[profile.ci]
inherits = "release"
debug-assertions = true
overflow-checks = true
debug = 1               # Line-number info for backtraces

# A profile for profiling: release speed with debug info
[profile.profiling]
inherits = "release"
debug = 2               # Full debug info for perf/flamegraph
strip = "none"

# A profile for benchmarks with maximum speed
[profile.bench-max]
inherits = "release"
opt-level = 3
lto = "fat"
codegen-units = 1
```

Build with a custom profile:

```bash
cargo build --profile ci
cargo build --profile profiling
cargo build --profile bench-max

# Output goes to target/<profile-name>/
# e.g., target/ci/myapp, target/profiling/myapp
```

### 5.6 Per-Package Profile Overrides

Override settings for specific dependencies:

```toml
[profile.dev.package.image]
opt-level = 3        # Image processing is too slow at opt-level 0

[profile.dev.package.regex]
opt-level = 2

# Optimize all dependencies but keep your own code unoptimized
[profile.dev.package."*"]
opt-level = 2
```

### 5.7 Comparing Binary Sizes

A practical example of profile impact:

```bash
# Standard dev build
$ cargo build
$ ls -lh target/debug/myapp
-rwxr-xr-x  1 user  staff   42M  myapp

# Standard release build
$ cargo build --release
$ ls -lh target/release/myapp
-rwxr-xr-x  1 user  staff   8.3M  myapp

# Optimized-for-size release build
$ cargo build --profile release-small
$ ls -lh target/release-small/myapp
-rwxr-xr-x  1 user  staff   2.1M  myapp
```

---

## 6. Dependencies Management

Cargo provides powerful tools for managing dependencies beyond simple version requirements.

### 6.1 Version Requirements

Cargo uses semantic versioning (SemVer):

```toml
[dependencies]
# Caret requirement (default) — compatible updates
serde = "1.0"         # >=1.0.0, <2.0.0
serde = "1.0.193"     # >=1.0.193, <2.0.0
serde = "0.9"         # >=0.9.0, <0.10.0 (pre-1.0: patch is breaking)

# Tilde requirement — more restrictive
serde = "~1.0.193"    # >=1.0.193, <1.1.0

# Wildcard requirement
serde = "1.*"          # >=1.0.0, <2.0.0

# Exact version
serde = "=1.0.193"    # Exactly 1.0.193

# Comparison requirements
serde = ">=1.0, <1.5" # Range
```

### 6.2 The Cargo.lock File

`Cargo.lock` records the exact version of every dependency used. This ensures reproducible builds.

- **Libraries** (lib crates): Do **not** commit `Cargo.lock` — consumers should resolve their own versions.
- **Applications** (bin crates): **Do** commit `Cargo.lock` — ensures everyone builds with the same versions.

```bash
# Update all dependencies to their latest allowed versions
cargo update

# Update a specific dependency
cargo update serde

# Update a specific dependency to a specific version
cargo update serde --precise 1.0.193

# Check for outdated dependencies (requires cargo-outdated)
cargo outdated
```

### 6.3 Patching Dependencies

The `[patch]` section lets you temporarily replace a dependency — useful for testing bug fixes or local forks:

```toml
# Use a local fork of serde
[patch.crates-io]
serde = { path = "../my-serde-fork" }

# Use a Git branch
[patch.crates-io]
serde = { git = "https://github.com/myuser/serde.git", branch = "fix-issue-123" }

# Patch a transitive dependency (one your dependency depends on)
[patch.crates-io]
hyper = { git = "https://github.com/myuser/hyper.git", rev = "abc1234" }
```

Patches apply transitively: if your dependency depends on `serde`, the patch replaces it everywhere.

### 6.4 Alternative Registries

For private crates, you can use alternative registries:

```toml
# .cargo/config.toml
[registries]
my-company = { index = "https://cargo.mycompany.com/git/index" }
```

```toml
# Cargo.toml
[dependencies]
internal-auth = { version = "2.0", registry = "my-company" }
public-crate = "1.0"   # defaults to crates.io
```

### 6.5 Git Dependencies

```toml
[dependencies]
# Use the default branch
my-lib = { git = "https://github.com/user/my-lib.git" }

# Pin to a specific branch
my-lib = { git = "https://github.com/user/my-lib.git", branch = "develop" }

# Pin to a specific tag
my-lib = { git = "https://github.com/user/my-lib.git", tag = "v1.2.3" }

# Pin to a specific commit
my-lib = { git = "https://github.com/user/my-lib.git", rev = "8a3ed4c" }
```

### 6.6 Dependency Types

Cargo distinguishes three kinds of dependencies:

```toml
# Normal dependencies — used when your crate is compiled
[dependencies]
serde = "1.0"

# Dev dependencies — only for tests, examples, benchmarks
[dev-dependencies]
tempfile = "3.10"
criterion = "0.5"
proptest = "1.4"

# Build dependencies — only for build.rs
[build-dependencies]
cc = "1.0"
bindgen = "0.70"
```

### 6.7 Auditing Dependencies

Cargo supports supply-chain auditing:

```bash
# Install cargo-audit
cargo install cargo-audit

# Check for known vulnerabilities
cargo audit

# Install cargo-deny for comprehensive checks
cargo install cargo-deny

# Check licenses, bans, advisories, and sources
cargo deny check
```

A `deny.toml` configuration file lets you set policies:

```toml
# deny.toml
[licenses]
allow = ["MIT", "Apache-2.0", "BSD-2-Clause", "BSD-3-Clause"]

[bans]
multiple-versions = "warn"
deny = ["openssl"]    # Prefer rustls
```

---

## 7. Publishing to crates.io

Sharing your crate with the Rust community requires a few preparation steps.

### 7.1 Preparing Your Crate

Ensure your `Cargo.toml` has all required metadata:

```toml
[package]
name = "my-awesome-crate"
version = "0.1.0"
edition = "2021"
authors = ["Your Name <you@example.com>"]
description = "A short description of what this crate does"
license = "MIT OR Apache-2.0"
repository = "https://github.com/youruser/my-awesome-crate"
homepage = "https://github.com/youruser/my-awesome-crate"
documentation = "https://docs.rs/my-awesome-crate"
readme = "README.md"
keywords = ["keyword1", "keyword2"]   # Max 5
categories = ["development-tools"]     # From crates.io list
exclude = [
    "tests/fixtures/*",
    ".github/*",
    "benches/*",
]
```

### 7.2 Pre-Publish Checks

```bash
# Verify the package builds and tests pass
cargo test
cargo clippy -- -D warnings
cargo fmt --check

# Inspect what will be published
cargo package --list

# Dry run — build the package without uploading
cargo publish --dry-run
```

### 7.3 API Documentation

Write comprehensive doc comments and verify them:

```rust
//! # My Awesome Crate
//!
//! `my_awesome_crate` provides utilities for doing amazing things.
//!
//! ## Quick Start
//!
//! ```
//! use my_awesome_crate::Widget;
//!
//! let w = Widget::new("example");
//! assert_eq!(w.name(), "example");
//! ```

/// A widget that does amazing things.
///
/// # Examples
///
/// ```
/// use my_awesome_crate::Widget;
///
/// let widget = Widget::new("test");
/// assert_eq!(widget.name(), "test");
/// ```
///
/// # Panics
///
/// Panics if `name` is empty.
pub struct Widget {
    name: String,
}

impl Widget {
    /// Creates a new `Widget` with the given name.
    ///
    /// # Arguments
    ///
    /// * `name` - The name for this widget. Must not be empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use my_awesome_crate::Widget;
    /// let w = Widget::new("hello");
    /// ```
    pub fn new(name: &str) -> Self {
        assert!(!name.is_empty(), "Widget name must not be empty");
        Self {
            name: name.to_string(),
        }
    }

    /// Returns the widget's name.
    pub fn name(&self) -> &str {
        &self.name
    }
}
```

```bash
# Build and open documentation locally
cargo doc --open

# Test all doc examples
cargo test --doc
```

### 7.4 Versioning Strategy

Follow SemVer rigorously:

| Change Type | Version Bump | Example |
|---|---|---|
| Bug fix, no API change | Patch | 0.1.0 → 0.1.1 |
| New feature, backward compatible | Minor | 0.1.1 → 0.2.0 |
| Breaking API change | Major | 0.2.0 → 1.0.0 |
| Pre-1.0 breaking change | Minor | 0.2.0 → 0.3.0 |

```bash
# Use cargo-release for automated versioning
cargo install cargo-release
cargo release patch   # 0.1.0 -> 0.1.1
cargo release minor   # 0.1.1 -> 0.2.0
cargo release major   # 0.2.0 -> 1.0.0
```

### 7.5 Publishing

```bash
# Login (one-time — stores token in ~/.cargo/credentials.toml)
cargo login <your-api-token>

# Publish!
cargo publish

# Publish a specific workspace member
cargo publish -p my-core
```

### 7.6 Yanking Versions

If you discover a critical bug after publishing:

```bash
# Yank a version — prevents new projects from depending on it
# Existing Cargo.lock files are unaffected
cargo yank --version 0.1.5

# Undo a yank
cargo yank --version 0.1.5 --undo
```

### 7.7 Publishing Workspace Members

When you have a workspace, publish in dependency order:

```bash
# 1. Publish the leaf dependency first
cargo publish -p shared-utils

# 2. Then crates that depend on it
cargo publish -p my-core

# 3. Finally the top-level crates
cargo publish -p my-cli
cargo publish -p my-server
```

Each member's `Cargo.toml` must have path **and** version for inter-workspace deps:

```toml
# cli/Cargo.toml
[dependencies]
my-core = { version = "0.1.0", path = "../core" }
shared-utils = { version = "0.1.0", path = "../shared-utils" }
```

When building locally, Cargo uses the path. When published to crates.io, the version is used.

---

## Summary

| Topic | Key Takeaway |
|---|---|
| **Workspaces** | Share `Cargo.lock` and `target/` across crates; use `workspace.dependencies` for consistent versions |
| **Feature flags** | Must be additive; use `dep:` prefix for optional dependencies; `cfg` and `cfg_attr` for conditional compilation |
| **Build scripts** | `build.rs` runs before compilation; generates code, links C libraries, sets env vars |
| **Cross-compilation** | `rustup target add` + linker config in `.cargo/config.toml`; target-specific deps via `[target.'cfg(...)'.dependencies]` |
| **Profiles** | Custom named profiles inherit from `dev` or `release`; per-package overrides for hot dependencies |
| **Dependencies** | `[patch]` for local overrides; `cargo audit` and `cargo deny` for supply-chain security |
| **Publishing** | Complete metadata, doc tests, SemVer discipline; publish workspace members in dependency order |

---

## Exercises

1. **Workspace setup**: Create a workspace with three crates: `math-core` (library), `math-cli` (binary), and `math-web` (binary). Both binaries depend on `math-core`.
2. **Feature flags**: Add a `simd` feature to `math-core` that uses SIMD intrinsics for vector addition when enabled, and falls back to a scalar loop when disabled.
3. **Build script**: Write a `build.rs` that reads a CSV file of constants and generates a Rust source file with those constants as `const` items.
4. **Cross-compile**: Configure `.cargo/config.toml` to cross-compile your workspace for `x86_64-unknown-linux-musl` and verify the resulting binary is statically linked.
5. **Custom profile**: Create a `profiling` profile that inherits from `release` but keeps full debug info. Use it with `perf` or `flamegraph` on a sample workload.
6. **Dependency patching**: Fork a dependency, make a change, and use `[patch.crates-io]` to test it locally before submitting a pull request.
7. **Publish dry run**: Prepare one of your workspace members for publishing. Run `cargo publish --dry-run` and fix any warnings or errors.

---

## Further Reading

- [The Cargo Book](https://doc.rust-lang.org/cargo/) — official reference for everything Cargo
- [Cargo Reference: Workspaces](https://doc.rust-lang.org/cargo/reference/workspaces.html)
- [Cargo Reference: Features](https://doc.rust-lang.org/cargo/reference/features.html)
- [Cargo Reference: Build Scripts](https://doc.rust-lang.org/cargo/reference/build-scripts.html)
- [Cargo Reference: Profiles](https://doc.rust-lang.org/cargo/reference/profiles.html)
- [The Rustup Book: Cross-compilation](https://rust-lang.github.io/rustup/cross-compilation.html)
- [crates.io Publishing Guide](https://doc.rust-lang.org/cargo/reference/publishing.html)

---

**Previous**: [Project: CLI Tool](./02_Project_CLI_Tool.md) | **Next**: [Declarative Macros](./04_Declarative_Macros.md)
