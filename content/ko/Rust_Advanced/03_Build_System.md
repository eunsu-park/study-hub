# 19. 빌드 시스템 심층 분석

**이전**: [프로젝트: CLI 도구](./02_Project_CLI_Tool.md) | **다음**: [선언적 매크로](./04_Declarative_Macros.md)

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. Cargo 워크스페이스와 다중 크레이트 아키텍처로 대규모 Rust 프로젝트를 구성하기
2. 기능 플래그(Feature Flag)를 사용하여 조건부 컴파일과 선택적 의존성 활성화하기
3. 빌드 스크립트(`build.rs`)를 작성하여 코드를 생성하고 네이티브 라이브러리를 연결하기
4. 다양한 대상 플랫폼에 대한 크로스 컴파일(Cross-Compilation) 설정하기
5. 사용자 정의 Cargo 프로필을 정의하여 최적화, 디버그 정보 및 기타 컴파일러 설정 조정하기
6. 패치(patch), 대체(replace), 대안 레지스트리를 사용하여 복잡한 의존성 그래프 관리하기
7. 적절한 메타데이터와 버전 관리로 크레이트를 crates.io에 준비하고 게시하기

---

## 목차

1. [Cargo 워크스페이스와 다중 크레이트 프로젝트](#1-cargo-워크스페이스와-다중-크레이트-프로젝트)
2. [기능 플래그와 조건부 컴파일](#2-기능-플래그와-조건부-컴파일)
3. [빌드 스크립트 (build.rs)](#3-빌드-스크립트-buildrs)
4. [크로스 컴파일 대상](#4-크로스-컴파일-대상)
5. [사용자 정의 프로필](#5-사용자-정의-프로필)
6. [의존성 관리](#6-의존성-관리)
7. [crates.io에 게시하기](#7-cratesio에-게시하기)

---

## 1. Cargo 워크스페이스와 다중 크레이트 프로젝트

Rust 프로젝트가 커지면, 여러 크레이트로 분할하면 컴파일 시간이 개선되고, API 경계가 강제되며, 각 부분을 독립적으로 테스트할 수 있게 됩니다. **워크스페이스(Workspace)**는 공통 `Cargo.lock`과 출력 디렉토리를 공유하는 크레이트의 집합입니다.

### 1.1 워크스페이스 생성

워크스페이스는 `[workspace]` 테이블을 포함하는 루트 `Cargo.toml`로 정의됩니다:

```toml
# 워크스페이스 루트: Cargo.toml
[workspace]
members = [
    "core",
    "cli",
    "server",
    "shared-utils",
]
resolver = "2"   # edition 2021+ 에서 권장
```

루트 `Cargo.toml`에는 `[package]` 섹션이 **없습니다** (루트 자체가 크레이트인 경우 제외). 각 멤버 디렉토리에는 자체 `Cargo.toml`이 있습니다:

```
my-project/
├── Cargo.toml          # 워크스페이스 루트
├── Cargo.lock          # 공유 락 파일
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

### 1.2 크레이트 간 의존성

멤버들은 **경로 의존성(Path Dependency)**으로 서로를 참조합니다:

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

경로 의존성은 컴파일 시에 해석됩니다. 게시할 때는 경로를 버전 요구사항으로 대체합니다 (또는 둘 다 사용):

```toml
[dependencies]
my-core = { version = "0.1", path = "../core" }
```

### 1.3 워크스페이스 수준의 의존성

Rust 1.64부터, 중복을 피하기 위해 워크스페이스 수준에서 공유 의존성 버전을 선언할 수 있습니다:

```toml
# 워크스페이스 루트: Cargo.toml
[workspace]
members = ["core", "cli", "server"]

[workspace.dependencies]
serde = { version = "1.0", features = ["derive"] }
tokio = { version = "1", features = ["full"] }
anyhow = "1.0"
```

각 멤버는 이를 상속받습니다:

```toml
# cli/Cargo.toml
[dependencies]
serde = { workspace = true }
tokio = { workspace = true }
anyhow = { workspace = true }
my-core = { path = "../core" }
```

이렇게 하면 모든 크레이트가 정확히 동일한 버전의 `serde`, `tokio` 등을 사용하는 것이 보장됩니다.

### 1.4 워크스페이스에서 명령 실행

```bash
# 모든 것을 빌드
cargo build

# 특정 멤버를 빌드
cargo build -p my-cli

# 특정 멤버의 테스트 실행
cargo test -p my-core

# 전체 워크스페이스의 테스트 실행
cargo test --workspace

# 특정 바이너리 실행
cargo run -p my-cli -- --help
```

### 1.5 가상 vs. 비가상 워크스페이스

**가상 워크스페이스(Virtual Workspace)**는 루트에 `[package]`가 없습니다 — 멤버를 그룹화하기 위해서만 존재합니다. **비가상 워크스페이스(Non-Virtual Workspace)**는 `Cargo.toml`에 `[workspace]`도 포함하는 일반 패키지입니다:

```toml
# 비가상: 루트 자체가 크레이트
[package]
name = "my-app"
version = "0.1.0"
edition = "2021"

[workspace]
members = ["plugins/*"]
```

대규모 프로젝트에서는 루트가 순수하게 조직자(Organizer) 역할을 하므로 가상 워크스페이스가 더 일반적입니다.

### 1.6 멤버를 위한 글롭(Glob) 패턴

글롭을 사용하여 여러 크레이트를 포함할 수 있습니다:

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

## 2. 기능 플래그와 조건부 컴파일

기능 플래그(Feature Flag)를 사용하면 크레이트 작성자가 선택적 기능을 노출할 수 있습니다. 소비자(Consumer)는 어떤 기능을 활성화할지 선택하여 바이너리 크기와 컴파일 시간을 최소화합니다.

### 2.1 기능 선언

```toml
# core/Cargo.toml
[package]
name = "my-core"
version = "0.1.0"
edition = "2021"

[features]
default = ["json"]          # 소비자가 비활성화하지 않는 한 활성화됨
json = ["dep:serde_json"]   # serde_json 의존성 활성화
xml = ["dep:quick-xml"]     # quick-xml 의존성 활성화
async = ["dep:tokio"]       # 비동기 지원 활성화
full = ["json", "xml", "async"]  # 편의를 위한 우산(Umbrella) 기능

[dependencies]
serde = { version = "1.0", features = ["derive"] }
serde_json = { version = "1.0", optional = true }
quick-xml = { version = "0.31", optional = true }
tokio = { version = "1", features = ["full"], optional = true }
```

`dep:` 접두사(Rust 1.60에서 안정화)는 선택적 의존성이 같은 이름의 기능을 암묵적으로 생성하는 것을 방지합니다.

### 2.2 `cfg` 속성 사용

기능에 따라 코드를 조건부로 컴파일합니다:

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

### 2.3 기능 플래그 이상의 조건부 컴파일

`cfg` 시스템은 기능 플래그 이상의 것을 지원합니다:

```rust
// 대상 OS
#[cfg(target_os = "linux")]
fn platform_specific() {
    println!("Linux에서 실행 중");
}

#[cfg(target_os = "windows")]
fn platform_specific() {
    println!("Windows에서 실행 중");
}

#[cfg(target_os = "macos")]
fn platform_specific() {
    println!("macOS에서 실행 중");
}

// 대상 아키텍처
#[cfg(target_arch = "x86_64")]
fn simd_optimized() {
    println!("x86_64 SIMD 명령어 사용");
}

#[cfg(target_arch = "aarch64")]
fn simd_optimized() {
    println!("ARM NEON 명령어 사용");
}

// 조건 결합
#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
fn linux_amd64_only() {
    println!("Linux x86_64 전용 코드");
}

#[cfg(any(target_os = "linux", target_os = "macos"))]
fn unix_like() {
    println!("유닉스 계열 시스템에서 실행 중");
}

#[cfg(not(target_os = "windows"))]
fn non_windows() {
    println!("Windows가 아닙니다");
}
```

### 2.4 조건부 속성을 위한 `cfg_attr`

조건이 충족될 때만 속성을 적용합니다:

```rust
// json 기능이 활성화된 경우에만 Serialize를 파생(Derive)
#[cfg_attr(feature = "json", derive(serde::Serialize, serde::Deserialize))]
pub struct Metadata {
    pub author: String,
    pub version: u32,
}

// 조건부 린트(Lint) 억제
#[cfg_attr(test, allow(dead_code))]
fn internal_helper() -> u32 {
    42
}
```

### 2.5 소비자에서 기능 활성화

```toml
# 소비자의 Cargo.toml에서
[dependencies]
my-core = { version = "0.1", features = ["json", "async"] }

# 또는 기본 기능을 비활성화하고 명시적으로 선택
my-core = { version = "0.1", default-features = false, features = ["xml"] }
```

### 2.6 기능 통합(Feature Unification)

Cargo는 의존성 그래프 전체에서 기능을 **통합**합니다. 크레이트 A가 `serde/derive`를 활성화하고 크레이트 B가 `serde/alloc`을 활성화하면, 최종 빌드에는 둘 다 포함됩니다. 이는 기능이 **추가적(Additive)**이어야 함을 의미합니다 — 기능을 활성화해도 기존 기능이 깨져서는 안 됩니다.

```rust
// 나쁜 예: 기능을 제거하는 기능
#[cfg(not(feature = "logging"))]
pub fn process(data: &[u8]) -> Vec<u8> {
    // 로깅 없는 빠른 경로
    data.to_vec()
}

#[cfg(feature = "logging")]
pub fn process(data: &[u8]) -> Vec<u8> {
    // 함수 시그니처/동작이 변경됨 — 추가성 위반
    log::info!("Processing {} bytes", data.len());
    data.to_vec()
}

// 좋은 예: 기능을 추가하는 기능
pub fn process(data: &[u8]) -> Vec<u8> {
    #[cfg(feature = "logging")]
    log::info!("Processing {} bytes", data.len());
    data.to_vec()
}
```

---

## 3. 빌드 스크립트 (build.rs)

**빌드 스크립트(Build Script)**는 크레이트 루트(`Cargo.toml` 옆)에 위치하는 `build.rs`라는 이름의 Rust 파일입니다. Cargo는 크레이트를 빌드하기 **전에** 이 파일을 컴파일하고 실행합니다. 빌드 스크립트는 코드를 생성하고, 네이티브 라이브러리를 연결하고, 환경 변수를 설정하는 등의 작업을 수행할 수 있습니다.

### 3.1 기본 빌드 스크립트

```rust
// build.rs
fn main() {
    // build.rs 자체가 변경되면 이 스크립트를 다시 실행하도록 Cargo에 알림
    println!("cargo::rerun-if-changed=build.rs");

    // 크레이트에서 env!()로 접근 가능한 환경 변수 설정
    println!("cargo::rustc-env=BUILD_TIMESTAMP={}", chrono_like_now());
}

fn chrono_like_now() -> String {
    // 실제로는 `chrono` 크레이트나 std::time을 사용합니다
    "2026-03-16T00:00:00Z".to_string()
}
```

코드에서 접근:

```rust
fn main() {
    println!("빌드 시각: {}", env!("BUILD_TIMESTAMP"));
}
```

### 3.2 Cargo 지시문(Instruction) 참조

빌드 스크립트는 `println!("cargo::...")` 지시문을 통해 Cargo와 통신합니다:

```rust
fn main() {
    // 재실행 트리거
    println!("cargo::rerun-if-changed=src/data.json");
    println!("cargo::rerun-if-changed=wrapper.h");
    println!("cargo::rerun-if-env-changed=MY_CONFIG_VAR");

    // 네이티브 라이브러리 연결
    println!("cargo::rustc-link-lib=sqlite3");
    println!("cargo::rustc-link-search=native=/usr/local/lib");

    // 컴파일러에 cfg 플래그 전달
    println!("cargo::rustc-cfg=has_avx2");

    // 다운스트림 코드를 위한 환경 변수 설정
    println!("cargo::rustc-env=GIT_HASH=abc123");

    // 경고 메시지 (빌드 중 표시)
    println!("cargo::warning=폴백 구성을 사용합니다");
}
```

> **참고**: 이전 버전에서는 콜론 하나(`cargo:`)를 사용했습니다. Rust 1.77부터는 이중 콜론(`cargo::`) 구문이 권장됩니다.

### 3.3 코드 생성

빌드 시에 Rust 소스 파일을 생성하는 것은 일반적인 사용 사례입니다:

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

    // 데이터 파일에서 명령어 목록 읽기
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

생성된 파일을 크레이트에 포함:

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

### 3.4 네이티브 C 라이브러리 연결

빌드 스크립트는 FFI에 필수적입니다. 다음은 `cc` 크레이트를 사용하는 패턴입니다:

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

### 3.5 자동 FFI 바인딩을 위한 `bindgen` 사용

복잡한 C 헤더의 경우, `bindgen` 크레이트가 자동으로 Rust FFI 바인딩을 생성합니다:

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
        .expect("바인딩을 생성할 수 없습니다");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("바인딩을 쓸 수 없습니다");
}
```

```rust
// src/lib.rs
#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
```

### 3.6 빌드 스크립트 의존성

빌드 스크립트는 크레이트의 런타임 의존성과 별도로 자체 의존성을 가질 수 있습니다:

```toml
[build-dependencies]
cc = "1.0"
bindgen = "0.70"
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
```

이들은 빌드 스크립트에 대해서만 컴파일되고 연결됩니다 — 최종 바이너리에는 영향을 주지 않습니다.

---

## 4. 크로스 컴파일 대상

Rust의 크로스 컴파일 지원은 주요 강점 중 하나입니다. 컴파일러는 다양한 플랫폼용 바이너리를 생성할 수 있습니다.

### 4.1 사용 가능한 대상 나열

```bash
# 지원되는 모든 대상 나열
rustup target list

# 설치된 대상만 나열
rustup target list --installed

# 새 대상 추가
rustup target add aarch64-unknown-linux-gnu
rustup target add x86_64-unknown-linux-musl
rustup target add wasm32-unknown-unknown
rustup target add x86_64-pc-windows-gnu
```

### 4.2 다른 대상을 위한 빌드

```bash
# ARM Linux용 빌드 (예: 라즈베리 파이)
cargo build --target aarch64-unknown-linux-gnu

# 정적 링크된 Linux 바이너리 빌드 (musl 사용)
cargo build --target x86_64-unknown-linux-musl --release

# WebAssembly용 빌드
cargo build --target wasm32-unknown-unknown --release
```

### 4.3 링커 설정

크로스 컴파일에는 종종 크로스 링커(Cross-Linker)가 필요합니다. `.cargo/config.toml`에서 설정합니다:

```toml
# .cargo/config.toml

[target.aarch64-unknown-linux-gnu]
linker = "aarch64-linux-gnu-gcc"

[target.x86_64-unknown-linux-musl]
linker = "x86_64-linux-musl-gcc"

[target.x86_64-pc-windows-gnu]
linker = "x86_64-w64-mingw32-gcc"

# 이 프로젝트의 기본 대상을 설정할 수도 있습니다
[build]
target = "x86_64-unknown-linux-gnu"
```

### 4.4 대상별 의존성

```toml
# Cargo.toml

# Linux 전용 의존성
[target.'cfg(target_os = "linux")'.dependencies]
inotify = "0.10"

# Windows 전용 의존성
[target.'cfg(target_os = "windows")'.dependencies]
winapi = { version = "0.3", features = ["winuser"] }

# macOS 전용 의존성
[target.'cfg(target_os = "macos")'.dependencies]
cocoa = "0.25"

# 모든 유닉스 계열 시스템 의존성
[target.'cfg(unix)'.dependencies]
nix = "0.28"
```

### 4.5 플랫폼별 코드 패턴

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

// 프로덕션에서는 크로스 플랫폼 경로를 위해 `dirs` 크레이트를 사용하세요.
```

### 4.6 Docker를 이용한 크로스 컴파일

재현 가능한 크로스 컴파일 환경을 위해 멀티 스테이지 Dockerfile이 잘 작동합니다:

```dockerfile
# 예시: Linux용 정적 링크 바이너리 빌드
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

### 4.7 CI에서 여러 대상 확인

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

## 5. 사용자 정의 프로필

Cargo에는 두 가지 기본 프로필이 있습니다: `dev` (개발용)와 `release` (프로덕션용). 둘 다 커스터마이즈할 수 있으며 완전히 새로운 프로필을 만들 수도 있습니다.

### 5.1 기본 프로필 기본값

| 설정 | `dev` | `release` |
|---|---|---|
| `opt-level` | 0 | 3 |
| `debug` | true | false |
| `debug-assertions` | true | false |
| `overflow-checks` | true | false |
| `lto` | false | false |
| `codegen-units` | 256 | 16 |
| `incremental` | true | false |
| `strip` | none | none |

### 5.2 기본 프로필 커스터마이즈

```toml
# Cargo.toml

# 더 빠른 dev 빌드: 의존성에 opt-level 1 사용
[profile.dev]
opt-level = 0

[profile.dev.package."*"]
opt-level = 1       # 의존성은 최적화하되 자신의 코드는 최적화하지 않음

# 릴리스 성능 최대화
[profile.release]
opt-level = 3
lto = "fat"          # 전체 링크 타임 최적화(LTO)
codegen-units = 1    # 최대 최적화를 위한 단일 코드 생성 유닛
strip = "symbols"    # 바이너리에서 디버그 심볼 제거
panic = "abort"      # 더 작은 바이너리, 언와인딩(Unwinding) 없음
```

### 5.3 최적화 수준

| 수준 | 의미 |
|---|---|
| `0` | 최적화 없음 (가장 빠른 컴파일) |
| `1` | 기본 최적화 |
| `2` | 대부분의 최적화 |
| `3` | 벡터화(Vectorization)를 포함한 모든 최적화 |
| `"s"` | 바이너리 크기 최적화 |
| `"z"` | 바이너리 크기 공격적 최적화 (더 느릴 수 있음) |

```toml
# 크기 최적화 — 임베디드나 WASM에 유용
[profile.release]
opt-level = "z"
lto = true
codegen-units = 1
strip = true
```

### 5.4 링크 타임 최적화 (LTO)

LTO는 크레이트 경계를 넘어 컴파일러가 최적화할 수 있게 합니다:

```toml
[profile.release]
# "fat" — 전체 크로스 크레이트 LTO (가장 느린 컴파일, 최고의 최적화)
lto = "fat"

# "thin" — 병렬 LTO, "fat"의 대부분의 이점을 가짐
# lto = "thin"

# false — LTO 없음
# lto = false
```

`thin` LTO의 일반적인 속도 향상은 10~20%이며, `fat` LTO는 상당히 긴 컴파일 시간의 대가로 몇 퍼센트 더 추가됩니다.

### 5.5 사용자 정의 이름 프로필

기본 프로필에서 상속하는 프로필을 만듭니다:

```toml
# Cargo.toml

# CI 테스트용 프로필: 릴리스와 유사하지만 디버그 어설션 포함
[profile.ci]
inherits = "release"
debug-assertions = true
overflow-checks = true
debug = 1               # 백트레이스를 위한 줄 번호 정보

# 프로파일링용 프로필: 릴리스 속도 + 디버그 정보
[profile.profiling]
inherits = "release"
debug = 2               # perf/flamegraph를 위한 전체 디버그 정보
strip = "none"

# 최대 속도의 벤치마크 프로필
[profile.bench-max]
inherits = "release"
opt-level = 3
lto = "fat"
codegen-units = 1
```

사용자 정의 프로필로 빌드:

```bash
cargo build --profile ci
cargo build --profile profiling
cargo build --profile bench-max

# 출력은 target/<프로필-이름>/에 저장됩니다
# 예: target/ci/myapp, target/profiling/myapp
```

### 5.6 패키지별 프로필 오버라이드

특정 의존성에 대한 설정 재정의:

```toml
[profile.dev.package.image]
opt-level = 3        # 이미지 처리는 opt-level 0에서 너무 느림

[profile.dev.package.regex]
opt-level = 2

# 모든 의존성을 최적화하되 자신의 코드는 최적화하지 않음
[profile.dev.package."*"]
opt-level = 2
```

### 5.7 바이너리 크기 비교

프로필 영향의 실제 예시:

```bash
# 표준 dev 빌드
$ cargo build
$ ls -lh target/debug/myapp
-rwxr-xr-x  1 user  staff   42M  myapp

# 표준 release 빌드
$ cargo build --release
$ ls -lh target/release/myapp
-rwxr-xr-x  1 user  staff   8.3M  myapp

# 크기 최적화된 release 빌드
$ cargo build --profile release-small
$ ls -lh target/release-small/myapp
-rwxr-xr-x  1 user  staff   2.1M  myapp
```

---

## 6. 의존성 관리

Cargo는 단순한 버전 요구사항 이상의 강력한 의존성 관리 도구를 제공합니다.

### 6.1 버전 요구사항

Cargo는 시맨틱 버저닝(SemVer)을 사용합니다:

```toml
[dependencies]
# 캐럿(Caret) 요구사항 (기본) — 호환되는 업데이트
serde = "1.0"         # >=1.0.0, <2.0.0
serde = "1.0.193"     # >=1.0.193, <2.0.0
serde = "0.9"         # >=0.9.0, <0.10.0 (1.0 이전: 패치가 호환성 깨짐)

# 틸데(Tilde) 요구사항 — 더 제한적
serde = "~1.0.193"    # >=1.0.193, <1.1.0

# 와일드카드(Wildcard) 요구사항
serde = "1.*"          # >=1.0.0, <2.0.0

# 정확한 버전
serde = "=1.0.193"    # 정확히 1.0.193

# 비교 요구사항
serde = ">=1.0, <1.5" # 범위
```

### 6.2 Cargo.lock 파일

`Cargo.lock`은 사용된 모든 의존성의 정확한 버전을 기록합니다. 이는 재현 가능한 빌드를 보장합니다.

- **라이브러리** (lib 크레이트): `Cargo.lock`을 커밋하지 **마세요** — 소비자가 자체적으로 버전을 해석해야 합니다.
- **애플리케이션** (bin 크레이트): `Cargo.lock`을 **커밋하세요** — 모든 사람이 같은 버전으로 빌드하는 것을 보장합니다.

```bash
# 모든 의존성을 허용된 최신 버전으로 업데이트
cargo update

# 특정 의존성 업데이트
cargo update serde

# 특정 의존성을 특정 버전으로 업데이트
cargo update serde --precise 1.0.193

# 오래된 의존성 확인 (cargo-outdated 필요)
cargo outdated
```

### 6.3 의존성 패칭

`[patch]` 섹션을 사용하면 의존성을 임시로 대체할 수 있습니다 — 버그 수정 테스트나 로컬 포크에 유용합니다:

```toml
# serde의 로컬 포크 사용
[patch.crates-io]
serde = { path = "../my-serde-fork" }

# Git 브랜치 사용
[patch.crates-io]
serde = { git = "https://github.com/myuser/serde.git", branch = "fix-issue-123" }

# 전이적 의존성 패치 (의존성이 의존하는 것)
[patch.crates-io]
hyper = { git = "https://github.com/myuser/hyper.git", rev = "abc1234" }
```

패치는 전이적으로 적용됩니다: 의존성이 `serde`에 의존하면, 패치가 모든 곳에서 이를 대체합니다.

### 6.4 대안 레지스트리

비공개 크레이트를 위해 대안 레지스트리를 사용할 수 있습니다:

```toml
# .cargo/config.toml
[registries]
my-company = { index = "https://cargo.mycompany.com/git/index" }
```

```toml
# Cargo.toml
[dependencies]
internal-auth = { version = "2.0", registry = "my-company" }
public-crate = "1.0"   # 기본값은 crates.io
```

### 6.5 Git 의존성

```toml
[dependencies]
# 기본 브랜치 사용
my-lib = { git = "https://github.com/user/my-lib.git" }

# 특정 브랜치에 고정
my-lib = { git = "https://github.com/user/my-lib.git", branch = "develop" }

# 특정 태그에 고정
my-lib = { git = "https://github.com/user/my-lib.git", tag = "v1.2.3" }

# 특정 커밋에 고정
my-lib = { git = "https://github.com/user/my-lib.git", rev = "8a3ed4c" }
```

### 6.6 의존성 유형

Cargo는 세 가지 종류의 의존성을 구분합니다:

```toml
# 일반 의존성 — 크레이트가 컴파일될 때 사용
[dependencies]
serde = "1.0"

# 개발 의존성 — 테스트, 예제, 벤치마크에만 사용
[dev-dependencies]
tempfile = "3.10"
criterion = "0.5"
proptest = "1.4"

# 빌드 의존성 — build.rs에만 사용
[build-dependencies]
cc = "1.0"
bindgen = "0.70"
```

### 6.7 의존성 감사(Auditing)

Cargo는 공급망 감사를 지원합니다:

```bash
# cargo-audit 설치
cargo install cargo-audit

# 알려진 취약점 확인
cargo audit

# 포괄적 검사를 위한 cargo-deny 설치
cargo install cargo-deny

# 라이선스, 금지 목록, 보안 권고, 소스 확인
cargo deny check
```

`deny.toml` 구성 파일로 정책을 설정할 수 있습니다:

```toml
# deny.toml
[licenses]
allow = ["MIT", "Apache-2.0", "BSD-2-Clause", "BSD-3-Clause"]

[bans]
multiple-versions = "warn"
deny = ["openssl"]    # rustls 선호
```

---

## 7. crates.io에 게시하기

Rust 커뮤니티와 크레이트를 공유하려면 몇 가지 준비 단계가 필요합니다.

### 7.1 크레이트 준비

`Cargo.toml`에 필수 메타데이터가 모두 있는지 확인합니다:

```toml
[package]
name = "my-awesome-crate"
version = "0.1.0"
edition = "2021"
authors = ["Your Name <you@example.com>"]
description = "이 크레이트가 무엇을 하는지에 대한 간단한 설명"
license = "MIT OR Apache-2.0"
repository = "https://github.com/youruser/my-awesome-crate"
homepage = "https://github.com/youruser/my-awesome-crate"
documentation = "https://docs.rs/my-awesome-crate"
readme = "README.md"
keywords = ["keyword1", "keyword2"]   # 최대 5개
categories = ["development-tools"]     # crates.io 목록에서 선택
exclude = [
    "tests/fixtures/*",
    ".github/*",
    "benches/*",
]
```

### 7.2 게시 전 검사

```bash
# 패키지가 빌드되고 테스트가 통과하는지 확인
cargo test
cargo clippy -- -D warnings
cargo fmt --check

# 게시될 내용 검사
cargo package --list

# 드라이 런(Dry Run) — 업로드 없이 패키지 빌드
cargo publish --dry-run
```

### 7.3 API 문서

포괄적인 문서 주석을 작성하고 검증합니다:

```rust
//! # My Awesome Crate
//!
//! `my_awesome_crate`는 놀라운 작업을 수행하는 유틸리티를 제공합니다.
//!
//! ## 빠른 시작
//!
//! ```
//! use my_awesome_crate::Widget;
//!
//! let w = Widget::new("example");
//! assert_eq!(w.name(), "example");
//! ```

/// 놀라운 작업을 수행하는 위젯.
///
/// # 예시
///
/// ```
/// use my_awesome_crate::Widget;
///
/// let widget = Widget::new("test");
/// assert_eq!(widget.name(), "test");
/// ```
///
/// # 패닉
///
/// `name`이 비어 있으면 패닉합니다.
pub struct Widget {
    name: String,
}

impl Widget {
    /// 주어진 이름으로 새 `Widget`을 생성합니다.
    ///
    /// # 인수
    ///
    /// * `name` - 이 위젯의 이름. 비어 있으면 안 됩니다.
    ///
    /// # 예시
    ///
    /// ```
    /// use my_awesome_crate::Widget;
    /// let w = Widget::new("hello");
    /// ```
    pub fn new(name: &str) -> Self {
        assert!(!name.is_empty(), "위젯 이름은 비어 있으면 안 됩니다");
        Self {
            name: name.to_string(),
        }
    }

    /// 위젯의 이름을 반환합니다.
    pub fn name(&self) -> &str {
        &self.name
    }
}
```

```bash
# 로컬에서 문서 빌드 및 열기
cargo doc --open

# 모든 문서 예제 테스트
cargo test --doc
```

### 7.4 버전 관리 전략

SemVer를 엄격히 따릅니다:

| 변경 유형 | 버전 범프 | 예시 |
|---|---|---|
| 버그 수정, API 변경 없음 | 패치(Patch) | 0.1.0 → 0.1.1 |
| 새 기능, 하위 호환 | 마이너(Minor) | 0.1.1 → 0.2.0 |
| 호환성 깨지는 API 변경 | 메이저(Major) | 0.2.0 → 1.0.0 |
| 1.0 이전 호환성 깨지는 변경 | 마이너 | 0.2.0 → 0.3.0 |

```bash
# 자동화된 버전 관리를 위한 cargo-release 사용
cargo install cargo-release
cargo release patch   # 0.1.0 -> 0.1.1
cargo release minor   # 0.1.1 -> 0.2.0
cargo release major   # 0.2.0 -> 1.0.0
```

### 7.5 게시

```bash
# 로그인 (한 번만 — ~/.cargo/credentials.toml에 토큰 저장)
cargo login <your-api-token>

# 게시!
cargo publish

# 특정 워크스페이스 멤버 게시
cargo publish -p my-core
```

### 7.6 버전 양크(Yank)

게시 후 치명적인 버그를 발견한 경우:

```bash
# 버전 양크 — 새 프로젝트가 이 버전에 의존하는 것을 방지
# 기존 Cargo.lock 파일에는 영향 없음
cargo yank --version 0.1.5

# 양크 취소
cargo yank --version 0.1.5 --undo
```

### 7.7 워크스페이스 멤버 게시

워크스페이스가 있는 경우, 의존성 순서대로 게시합니다:

```bash
# 1. 먼저 리프(Leaf) 의존성 게시
cargo publish -p shared-utils

# 2. 그 다음 이에 의존하는 크레이트
cargo publish -p my-core

# 3. 마지막으로 최상위 크레이트
cargo publish -p my-cli
cargo publish -p my-server
```

각 멤버의 `Cargo.toml`에는 워크스페이스 간 의존성에 경로 **및** 버전이 있어야 합니다:

```toml
# cli/Cargo.toml
[dependencies]
my-core = { version = "0.1.0", path = "../core" }
shared-utils = { version = "0.1.0", path = "../shared-utils" }
```

로컬에서 빌드할 때 Cargo는 경로를 사용합니다. crates.io에 게시되면 버전이 사용됩니다.

---

## 요약

| 주제 | 핵심 요약 |
|---|---|
| **워크스페이스** | 크레이트 간 `Cargo.lock`과 `target/` 공유; 일관된 버전을 위해 `workspace.dependencies` 사용 |
| **기능 플래그** | 반드시 추가적이어야 함; 선택적 의존성에 `dep:` 접두사 사용; 조건부 컴파일에 `cfg`와 `cfg_attr` |
| **빌드 스크립트** | `build.rs`는 컴파일 전에 실행; 코드 생성, C 라이브러리 연결, 환경 변수 설정 |
| **크로스 컴파일** | `rustup target add` + `.cargo/config.toml`에서 링커 설정; `[target.'cfg(...)'.dependencies]`로 대상별 의존성 |
| **프로필** | 사용자 정의 이름 프로필은 `dev` 또는 `release`에서 상속; 핫(Hot) 의존성에 패키지별 오버라이드 |
| **의존성** | 로컬 오버라이드에 `[patch]`; 공급망 보안에 `cargo audit`과 `cargo deny` |
| **게시** | 완전한 메타데이터, 문서 테스트, SemVer 준수; 워크스페이스 멤버는 의존성 순서대로 게시 |

---

## 연습 문제

1. **워크스페이스 설정**: `math-core` (라이브러리), `math-cli` (바이너리), `math-web` (바이너리) 세 크레이트로 워크스페이스를 만드세요. 두 바이너리 모두 `math-core`에 의존합니다.
2. **기능 플래그**: `math-core`에 `simd` 기능을 추가하여 활성화되면 벡터 덧셈에 SIMD 인트린식(Intrinsic)을 사용하고, 비활성화되면 스칼라 루프로 폴백하도록 하세요.
3. **빌드 스크립트**: CSV 파일에서 상수를 읽고 해당 상수를 `const` 항목으로 갖는 Rust 소스 파일을 생성하는 `build.rs`를 작성하세요.
4. **크로스 컴파일**: `.cargo/config.toml`을 설정하여 워크스페이스를 `x86_64-unknown-linux-musl`로 크로스 컴파일하고 결과 바이너리가 정적으로 링크되었는지 확인하세요.
5. **사용자 정의 프로필**: `release`에서 상속받되 전체 디버그 정보를 유지하는 `profiling` 프로필을 만드세요. 샘플 워크로드에서 `perf`나 `flamegraph`와 함께 사용하세요.
6. **의존성 패칭**: 의존성을 포크하고, 변경한 다음, `[patch.crates-io]`를 사용하여 풀 리퀘스트를 제출하기 전에 로컬에서 테스트하세요.
7. **게시 드라이 런**: 워크스페이스 멤버 중 하나를 게시할 준비를 하세요. `cargo publish --dry-run`을 실행하고 경고나 오류를 수정하세요.

---

## 추가 자료

- [The Cargo Book](https://doc.rust-lang.org/cargo/) — Cargo 관련 모든 것의 공식 레퍼런스
- [Cargo 레퍼런스: 워크스페이스](https://doc.rust-lang.org/cargo/reference/workspaces.html)
- [Cargo 레퍼런스: 기능](https://doc.rust-lang.org/cargo/reference/features.html)
- [Cargo 레퍼런스: 빌드 스크립트](https://doc.rust-lang.org/cargo/reference/build-scripts.html)
- [Cargo 레퍼런스: 프로필](https://doc.rust-lang.org/cargo/reference/profiles.html)
- [Rustup 북: 크로스 컴파일](https://rust-lang.github.io/rustup/cross-compilation.html)
- [crates.io 게시 가이드](https://doc.rust-lang.org/cargo/reference/publishing.html)

---

**이전**: [프로젝트: CLI 도구](./02_Project_CLI_Tool.md) | **다음**: [선언적 매크로](./04_Declarative_Macros.md)
