# 01. 시작하기

**이전**: [개요](./00_Overview.md) | **다음**: [변수와 타입](./02_Variables_and_Types.md)

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. `rustup`을 사용하여 Rust 툴체인(toolchain)을 설치하고 관리할 수 있다
2. `cargo`로 프로젝트를 생성, 빌드, 실행할 수 있다
3. 컴파일 후 실행(compile-then-run) 실행 모델을 설명하고, 인터프리터 언어와의 차이점을 설명할 수 있다
4. `fn main()`과 `println!`을 사용하여 기본 Rust 프로그램을 작성할 수 있다

---

Rust의 툴체인은 놀랍도록 통일되어 있습니다. C/C++ 프로젝트가 Makefile, CMake, 플랫폼별 컴파일러를 조합해야 하는 것과 달리, Rust는 **Cargo**라는 단 하나의 도구가 프로젝트 생성, 의존성 관리, 빌드, 테스트, 배포까지 모두 처리합니다.

## 목차
1. [Rust 설치](#1-rust-설치)
2. [Hello, World!](#2-hello-world)
3. [Cargo: Rust 빌드 시스템](#3-cargo-rust-빌드-시스템)
4. [Cargo 프로젝트의 구조](#4-cargo-프로젝트의-구조)
5. [연습 문제](#5-연습-문제)

---

## 1. Rust 설치

### 1.1 rustup

`rustup`은 공식 설치 관리자이자 버전 관리자입니다. 컴파일러 버전, 표준 라이브러리, 크로스 컴파일 타겟을 관리합니다.

```bash
# Install Rust (Unix/macOS)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install Rust (Windows) — download rustup-init.exe from https://rustup.rs

# Check installed versions
rustc --version      # Compiler
cargo --version      # Build tool / package manager
rustup --version     # Toolchain manager
```

### 1.2 툴체인 채널(Toolchain Channels)

Rust는 세 가지 릴리즈 채널을 사용합니다:

| 채널 | 릴리즈 주기 | 용도 |
|------|------------|------|
| **stable** | 6주마다 | 프로덕션 코드 |
| **beta** | stable 출시 6주 전 | 예정 기능 테스트 |
| **nightly** | 매일 | 실험적 기능 (`#![feature(...)]`) |

```bash
# Switch default toolchain
rustup default stable
rustup default nightly

# Install a specific version
rustup install 1.82.0

# Use nightly for one command
rustup run nightly cargo build
```

### 1.3 필수 컴포넌트

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

### 2.1 Cargo 없이 실행하기

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

핵심 관찰:
- `fn main()`은 진입점(entry point)입니다 — 모든 Rust 바이너리는 정확히 하나의 main을 가집니다
- `println!`은 **매크로(macro)**입니다 (`!` 주목) — 함수가 아니며, 매크로는 컴파일 타임에 코드를 생성합니다
- 세미콜론은 구문(statement)을 종료합니다 (Rust는 표현식(expression) 기반이지만, 구문 종료 `;`는 필수입니다)

### 2.2 컴파일 언어 vs 인터프리터 언어

```
Rust (compiled):
  source.rs  →  rustc  →  native binary  →  OS executes directly
                          (no runtime needed)

Python (interpreted):
  script.py  →  python interpreter  →  bytecode  →  VM executes
                (runtime always needed)
```

Rust는 시스템의 C 라이브러리에 링크된 독립 실행 파일을 생성합니다 (기본값). 가비지 컬렉터(Garbage Collector), VM, 런타임(runtime)이 없으며 — 오직 기계어 코드만 존재합니다.

---

## 3. Cargo: Rust 빌드 시스템

### 3.1 프로젝트 생성

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

### 3.2 필수 명령어

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

> **팁**: `cargo check`는 코드 생성 단계를 건너뛰기 때문에 `cargo build`보다 훨씬 빠릅니다. 개발 중에 타입 오류를 조기에 발견하기 위해 자주 사용하세요.

### 3.3 디버그 빌드 vs 릴리즈 빌드

| 항목 | `cargo build` (디버그) | `cargo build --release` |
|------|----------------------|------------------------|
| 최적화 | 없음 (`opt-level = 0`) | 최대 (`opt-level = 3`) |
| 컴파일 속도 | 빠름 | 느림 |
| 런타임 속도 | 느림 (최적화 없음) | 빠름 |
| 디버그 정보 | 포함 | 제거됨 |
| 출력 디렉토리 | `target/debug/` | `target/release/` |
| 용도 | 개발 | 프로덕션 / 벤치마킹 |

---

## 4. Cargo 프로젝트의 구조

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

### 4.2 에디션(Editions)

Rust **에디션(Edition)** (2015, 2018, 2021, 2024)은 기존 코드를 깨뜨리지 않고 언어를 발전시킬 수 있게 해줍니다. 각 에디션은 새 키워드나 구문 변경을 도입할 수 있지만, 서로 다른 에디션을 사용하는 크레이트(crate)들은 원활하게 상호 운용됩니다.

```
Edition 2015: Initial syntax
Edition 2018: async/await keywords, module path changes
Edition 2021: Closure capture improvements, IntoIterator for arrays
Edition 2024: gen blocks, unsafe_op_in_unsafe_fn lint
```

### 4.3 프로젝트 레이아웃 관례

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

## 5. 연습 문제

### 연습 1: 프로젝트 설정
새 Cargo 프로젝트를 생성하고, `rand = "0.8"`을 의존성으로 추가한 뒤, 1부터 100 사이의 난수를 출력하는 프로그램을 작성하세요.

<details>
<summary>힌트</summary>

`rand::Rng` 트레이트와 `thread_rng().gen_range(1..=100)`을 사용하세요.
</details>

### 연습 2: 다중 바이너리
두 개의 바이너리를 가진 프로젝트를 만드세요: "main binary"를 출력하는 `src/main.rs`와 "Hello from greet!"를 출력하는 `src/bin/greet.rs`. `cargo run`과 `cargo run --bin greet`로 각각 실행해 보세요.

### 연습 3: Cargo 기능 탐색
프로젝트에 `serde = { version = "1.0", features = ["derive"] }`를 추가하세요. `cargo build`와 `cargo build --release`의 컴파일 속도와 런타임 성능 측면에서의 차이를 설명하세요.

### 연습 4: 툴체인 탐색
프로젝트에서 `cargo clippy`를 실행해 보세요. 의도적으로 경고를 유발하는 코드(예: 미사용 변수)를 작성하고 clippy의 제안을 살펴보세요. 이후 `cargo fmt`를 실행하고 포맷팅 변화를 비교해 보세요.

---

## 참고 자료
- [The Rust Programming Language (The Book)](https://doc.rust-lang.org/book/)
- [Cargo Book](https://doc.rust-lang.org/cargo/)
- [rustup Documentation](https://rust-lang.github.io/rustup/)

---

**이전**: [개요](./00_Overview.md) | **다음**: [변수와 타입](./02_Variables_and_Types.md)
