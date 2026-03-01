# 16. 모듈(Module)과 카고(Cargo)

**이전**: [비동기 및 Await](./15_Async_Await.md) | **다음**: [안전하지 않은 Rust](./17_Unsafe_Rust.md)

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. `mod`, `pub`, `use`, `self`, `super`, `crate`를 사용하여 코드를 모듈(Module)로 구성하기
2. `mod.rs` 방식과 파일명 방식 모두를 사용하여 다중 파일 프로젝트 구조화하기
3. `pub`, `pub(crate)`, `pub(super)`로 가시성을 제어하여 캡슐화(Encapsulation) 강제하기
4. 의존성(Dependency), 피처(Feature), 워크스페이스(Workspace)를 위한 `Cargo.toml` 설정하기
5. 의존성 관리를 위해 카고(Cargo) 명령어(`cargo add`, `cargo tree`, `cargo audit`) 사용하기

---

Rust 프로그램이 단일 파일을 넘어 커지면, 코드를 논리적 단위로 분리하고 외부 라이브러리를 관리할 방법이 필요합니다. Rust의 **모듈 시스템(Module System)**은 첫 번째 문제를 해결합니다 — 모든 서랍(모듈)에 자체 잠금장치(가시성)가 달린 서류 캐비닛처럼 생각하세요. **카고(Cargo)**는 두 번째 문제를 해결합니다 — 빌드 시스템, 패키지 관리자, 프로젝트 오케스트레이터(Orchestrator)를 하나로 합친 것입니다.

## 목차
1. [모듈 시스템](#1-모듈-시스템)
2. [파일 기반 모듈 구조](#2-파일-기반-모듈-구조)
3. [가시성 규칙](#3-가시성-규칙)
4. [pub use로 재내보내기](#4-pub-use로-재내보내기)
5. [Cargo.toml 심층 분석](#5-cargotoml-심층-분석)
6. [카고 워크스페이스](#6-카고-워크스페이스)
7. [crates.io에 배포하기](#7-cratesio에-배포하기)
8. [유용한 카고 명령어](#8-유용한-카고-명령어)
9. [피처 플래그](#9-피처-플래그)
10. [연습 문제](#10-연습-문제)

---

## 1. 모듈 시스템

### 1.1 인라인 모듈 정의하기

모듈은 관련 항목들 — 함수, 구조체(Struct), 열거형(Enum), 상수, 트레이트(Trait), 심지어 다른 모듈까지 — 을 묶어줍니다:

```rust
// 모듈은 네임스페이스(Namespace)를 생성합니다. 파일 시스템의 폴더와 유사합니다.
// 내부 항목은 기본적으로 비공개입니다 — 외부에 공개하려면 명시적으로 지정해야 합니다.
mod math {
    pub fn add(a: i32, b: i32) -> i32 {
        a + b
    }

    pub fn multiply(a: i32, b: i32) -> i32 {
        a * b
    }

    // 비공개 헬퍼(Helper) — 이 모듈 내부에서만 접근 가능
    fn _validate(n: i32) -> bool {
        n >= 0
    }
}

fn main() {
    // 모듈 경로를 통해 공개 항목에 접근
    let sum = math::add(3, 4);
    let product = math::multiply(3, 4);
    println!("{sum}, {product}"); // 7, 12
}
```

### 1.2 use, self, super, crate

이 네 가지 키워드로 모듈 트리(Tree)를 탐색합니다:

```
crate (루트)
├── main.rs          ← 크레이트(Crate) 루트
├── network          ← mod network
│   ├── client       ← mod client (network의 자식)
│   └── server       ← mod server (network의 자식)
└── utils            ← mod utils
```

```rust
mod network {
    pub mod client {
        pub fn connect() {
            println!("Client connecting...");
        }

        pub fn status() {
            // `self`는 현재 모듈(client)을 가리킵니다
            self::connect();
        }
    }

    pub mod server {
        pub fn listen() {
            // `super`는 한 단계 위(network)로 올라갔다가
            // client로 내려갑니다
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
    // `crate`는 현재 크레이트(Crate)의 루트를 가리킵니다
    crate::network::client::connect();
    crate::network::server::listen();

    // `use`로 항목을 스코프(Scope)에 가져와 긴 경로 입력을 피합니다
    use network::client::connect;
    connect();

    // 충돌을 피하기 위해 가져올 때 이름 변경
    use utils::log as write_log;
    write_log("startup complete");
}
```

### 1.3 임포트 묶기

```rust
// 같은 모듈에서 여러 항목 임포트(Import)
use std::collections::{HashMap, HashSet, BTreeMap};

// 모듈 자체와 그 모듈의 특정 항목 임포트
use std::io::{self, Read, Write};
// 이제 `io::Result`와 `Read`를 직접 사용할 수 있습니다

// 글로브(Glob) 임포트 — 공개된 모든 항목을 스코프로 가져옵니다 (신중하게 사용하세요)
use std::collections::*;
```

---

## 2. 파일 기반 모듈 구조

실제 프로젝트에서는 모듈을 별도 파일로 분리합니다. Rust는 두 가지 관례를 제공합니다:

### 2.1 mod.rs 관례 (에디션(Edition) 2015 스타일)

```
src/
├── main.rs
└── network/
    ├── mod.rs       ← network 모듈의 내용을 선언합니다
    ├── client.rs
    └── server.rs
```

```rust
// src/network/mod.rs
pub mod client;  // network/client.rs에서 로드
pub mod server;  // network/server.rs에서 로드
```

### 2.2 파일명 관례 (에디션 2018+ 권장)

```
src/
├── main.rs
├── network.rs       ← network 모듈의 내용을 선언합니다
└── network/
    ├── client.rs
    └── server.rs
```

```rust
// src/network.rs — mod.rs와 같은 역할이지만 폴더 옆에 위치합니다
pub mod client;
pub mod server;
```

두 관례는 동일합니다. 2018+ 스타일은 에디터 탭에 `mod.rs`라는 이름의 파일이 여러 개 열리는 것을 피할 수 있습니다.

### 2.3 크레이트 루트에서 모듈 선언하기

```rust
// src/main.rs (또는 라이브러리 크레이트의 경우 src/lib.rs)
mod network;  // Rust는 network.rs 또는 network/mod.rs를 찾습니다
mod utils;    // Rust는 utils.rs를 찾습니다

fn main() {
    network::client::connect();
    utils::log("started");
}
```

핵심 멘탈 모델(Mental Model): **`mod`는 선언이지 임포트가 아닙니다.** `mod network;`를 작성할 때, 컴파일러에게 "`network`라는 모듈이 있으니 — 소스 파일을 찾아 이 크레이트의 일부로 컴파일하라"고 알려주는 것입니다.

---

## 3. 가시성 규칙

Rust는 기본적으로 **비공개**입니다. 접근 권한을 명시적으로 부여해야 합니다:

| 키워드 | 접근 가능 범위 |
|--------|--------------|
| (없음) | 현재 모듈과 그 자식 모듈만 |
| `pub` | 모든 곳 |
| `pub(crate)` | 동일한 크레이트 내 어디서든 |
| `pub(super)` | 부모 모듈 |
| `pub(in path)` | 특정 상위 모듈 |

```rust
mod database {
    // 공개 — `database`를 볼 수 있는 모든 코드에서 호출 가능
    pub fn connect() -> Connection {
        Connection { pool: create_pool() }
    }

    // 크레이트 범위 공개 — 이 크레이트의 다른 모듈에서 사용 가능하지만,
    // 외부 크레이트에서는 사용 불가
    pub(crate) fn diagnostics() -> String {
        format!("pool size: {}", POOL_SIZE)
    }

    // 비공개 — `database`(와 그 서브모듈) 내부 코드만 접근 가능
    fn create_pool() -> Vec<()> {
        vec![(); POOL_SIZE]
    }

    const POOL_SIZE: usize = 10;

    pub struct Connection {
        // 구조체는 공개이지만 필드는 비공개입니다.
        // 외부 코드는 connect()를 통해 Connection을 얻을 수 있지만
        // 직접 생성할 수는 없습니다.
        pool: Vec<()>,
    }

    impl Connection {
        pub fn query(&self, sql: &str) -> String {
            format!("Executing: {sql}")
        }
    }

    mod internal {
        pub(super) fn migrate() {
            // pub(super)는 이것을 `database`에는 보이게 하지만
            // `database` 외부에는 보이지 않게 합니다
            println!("Running migrations...");
        }
    }
}

fn main() {
    let conn = database::connect();
    println!("{}", conn.query("SELECT 1"));

    // 작동합니다 — pub(crate)는 동일한 크레이트 내에서 보입니다
    println!("{}", database::diagnostics());

    // 아래는 컴파일되지 않습니다:
    // database::create_pool();           // 비공개
    // database::internal::migrate();     // pub(super), database에서만 보임
    // let c = database::Connection { pool: vec![] };  // 필드가 비공개
}
```

---

## 4. pub use로 재내보내기

재내보내기(Re-export)를 통해 내부 구조를 숨기면서 깔끔한 공개 API를 제공할 수 있습니다:

```rust
// 재내보내기 없이는, 사용자가 내부 구조를 직접 탐색해야 합니다:
//   mycrate::parsers::json::JsonParser
//   mycrate::parsers::csv::CsvParser

// 재내보내기를 사용하면, API 표면을 평탄화할 수 있습니다:
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

// 크레이트 루트에서 재내보내기하여 사용자에게 깔끔한 API 제공
pub use parsers::json::JsonParser;
pub use parsers::csv::CsvParser;

fn main() {
    // 사용자는 이제 내부 경로를 몰라도 직접 접근 가능
    let result = JsonParser::parse(r#"{"key": "value"}"#);
    println!("{result}");
}
```

`prelude` 패턴은 이것의 일반적인 활용입니다: 라이브러리 크레이트는 흔히 가장 많이 사용되는 타입을 재내보내는 `prelude` 모듈을 갖습니다.

```rust
// 라이브러리 크레이트 내에서:
// pub mod prelude {
//     pub use crate::JsonParser;
//     pub use crate::CsvParser;
//     pub use crate::Error;
// }
//
// 사용자는 이렇게 씁니다:
//   use mycrate::prelude::*;
```

---

## 5. Cargo.toml 심층 분석

`Cargo.toml`은 모든 Rust 프로젝트의 매니페스트(Manifest) 파일입니다. 다음은 주석이 달린 예시입니다:

```toml
[package]
name = "my-app"
version = "0.1.0"
edition = "2021"           # Rust 에디션 (2015, 2018, 2021, 2024)
authors = ["Alice <alice@example.com>"]
description = "A sample application"
license = "MIT"
repository = "https://github.com/alice/my-app"
rust-version = "1.75"     # 최소 지원 Rust 버전(MSRV)

[dependencies]
serde = { version = "1.0", features = ["derive"] }   # 피처 플래그 포함
tokio = { version = "1", features = ["full"] }
log = "0.4"                 # 축약형: 버전만 지정

[dev-dependencies]
# 테스트, 예제, 벤치마크에서만 사용
criterion = "0.5"
tempfile = "3"

[build-dependencies]
# build.rs (빌드 스크립트)에서만 사용
cc = "1.0"

[profile.release]
opt-level = 3        # 최대 최적화
lto = true           # 링크 타임 최적화(LTO) — 빌드는 느려지지만 바이너리는 빠름
strip = true         # 디버그 심볼 제거 — 더 작은 바이너리
```

### 버전 요구사항

```
"1.0"      →  >=1.0.0, <2.0.0   (캐럿(Caret), 기본값)
"~1.0"     →  >=1.0.0, <1.1.0   (틸드(Tilde) — 패치 수준만)
"=1.0.3"   →  정확히 1.0.3
">=1.0, <1.5"  → 범위
"*"        →  모든 버전 (배포된 크레이트에서는 사용하지 마세요)
```

---

## 6. 카고 워크스페이스

워크스페이스(Workspace)를 사용하면 단일 저장소(Repository)에서 여러 관련 크레이트를 관리할 수 있습니다. 하나의 `Cargo.lock`과 출력 디렉토리를 공유하므로 빌드 속도가 빨라지고 의존성 버전이 일관되게 유지됩니다.

```
my-project/
├── Cargo.toml          ← 워크스페이스 루트
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
# 루트 Cargo.toml — 워크스페이스 정의
[workspace]
members = [
    "crates/core",
    "crates/cli",
    "crates/web",
]
resolver = "2"   # v2 의존성 리졸버(Resolver) 사용 (권장)

# 공유 의존성 — 멤버 크레이트들이 이 버전을 상속받습니다
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
# 경로로 형제 크레이트 참조
my-core = { path = "../core" }
# 워크스페이스에서 버전 상속 — 모든 크레이트를 동기화 상태로 유지
serde.workspace = true
```

```bash
# 특정 크레이트 빌드
cargo build -p my-cli

# 전체 워크스페이스에서 테스트 실행
cargo test --workspace

# 특정 바이너리 실행
cargo run -p my-cli
```

---

## 7. crates.io에 배포하기

```bash
# 1. https://crates.io에서 계정 생성 후 API 토큰 획득
cargo login <your-token>

# 2. 패키지 준비 상태 확인
cargo package        # .crate 아카이브 생성 — 포함될 내용 검토
cargo package --list # 배포될 파일 목록 확인

# 3. 배포
cargo publish

# 4. 버전 얀크(Yank, 오류 릴리즈를 배포한 경우)
cargo yank --version 0.1.0    # 새 프로젝트에서 이 버전 사용을 권장하지 않음
cargo yank --undo --version 0.1.0  # 얀크 취소
```

crates.io 중요 규칙:
- 패키지 이름은 전역적으로 유일하며 선착순입니다
- 배포된 버전은 **영구적**입니다 — 덮어쓰거나 삭제할 수 없습니다
- 얀크는 새 `Cargo.lock` 파일이 해당 버전을 선택하지 않도록 하지만, 이미 의존하는 기존 프로젝트는 영향을 받지 않습니다
- 배포하려면 `Cargo.toml`에 `description`과 `license` 필드가 필요합니다

---

## 8. 유용한 카고 명령어

```bash
# --- 의존성 관리 ---
cargo add serde --features derive    # 의존성 추가 (Cargo.toml 편집)
cargo add tokio -F full              # -F는 --features의 축약형
cargo add --dev criterion            # [dev-dependencies]에 추가
cargo remove serde                   # 의존성 제거

# --- 검사 ---
cargo tree                    # 의존성 트리 표시
cargo tree -d                 # 중복 의존성만 표시
cargo tree -i serde           # 역방향: serde에 의존하는 것은?

# --- 보안 ---
cargo audit                   # 알려진 취약점 확인 (설치: cargo install cargo-audit)
cargo deny check              # 정책 기반 의존성 린팅(Linting) (설치: cargo install cargo-deny)

# --- 품질 ---
cargo clippy                  # 린트: 일반적인 실수를 잡고 개선 사항 제안
cargo fmt                     # rustfmt 규칙에 따라 코드 포맷
cargo doc --open              # HTML 문서 생성 및 열기

# --- 빌드 & 실행 ---
cargo build --release         # 최적화 빌드
cargo run -- arg1 arg2        # 인수와 함께 실행 (--는 cargo 인수와 프로그램 인수를 구분)
cargo test                    # 모든 테스트 실행
cargo bench                   # 벤치마크 실행
cargo check                   # 바이너리 생성 없이 타입 검사 (가장 빠른 피드백)
```

---

## 9. 피처 플래그

피처(Feature)를 사용하면 사용자가 컴파일 시점에 기능을 선택적으로 활성화할 수 있습니다. 이를 통해 기본 빌드를 가볍게 유지하면서 선택적 기능을 제공합니다.

```toml
# 라이브러리 크레이트의 Cargo.toml
[package]
name = "my-logger"
version = "0.1.0"
edition = "2021"

[features]
# 기본적으로 활성화된 피처 없음
default = []

# 개별 피처
color = ["dep:colored"]        # `colored` 의존성을 활성화
json = ["dep:serde_json", "dep:serde"]
file = []                      # 추가 의존성 없는 피처
full = ["color", "json", "file"]  # 모든 것을 활성화하는 편의 피처

[dependencies]
colored = { version = "2", optional = true }      # `color` 피처가 활성화될 때만 컴파일
serde = { version = "1", features = ["derive"], optional = true }
serde_json = { version = "1", optional = true }
```

```rust
// src/lib.rs — cfg 속성(Attribute)으로 조건부 컴파일

pub fn log(message: &str) {
    // 기본 기능은 항상 사용 가능
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
    // 이 함수 전체는 `json` 피처가 활성화될 때만 존재합니다
    use serde_json::json;
    let entry = json!({
        "timestamp": "2025-01-15T10:30:00",
        "message": message,
    });
    println!("{}", entry);
}
```

```bash
# 사용자는 의존성을 추가할 때 피처를 선택합니다:
cargo add my-logger --features color,json

# 또는 Cargo.toml에서:
# [dependencies]
# my-logger = { version = "0.1", features = ["color", "json"] }
```

---

## 10. 연습 문제

### 연습 1: 모듈 계층 구조
다음 모듈 구조를 파일을 사용하여 프로젝트로 만드세요 (인라인 모듈이 아닌):

```
src/
├── main.rs
├── shapes.rs
└── shapes/
    ├── circle.rs
    └── rectangle.rs
```

각 도형 모듈은 `area(&self) -> f64` 메서드를 가진 구조체를 정의해야 합니다. `shapes.rs` 파일은 두 구조체를 모두 재내보내야 합니다. `main.rs`는 `shapes::Circle`과 `shapes::Rectangle`을 통해 사용해야 합니다.

### 연습 2: 가시성 도전
`bank` 모듈에 `Account` 구조체를 만드세요. `balance` 필드는 비공개여야 합니다. `pub fn new(owner: &str, initial: f64) -> Account`, `pub fn deposit(&mut self, amount: f64)`, `pub fn balance(&self) -> f64`, 그리고 `pub(crate)` 메서드 `audit_log(&self) -> String`을 제공하세요. `bank` 모듈 외부의 코드가 `balance`에 직접 접근할 수 없지만 `audit_log`는 호출할 수 있음을 확인하세요.

### 연습 3: 워크스페이스 설정
세 개의 크레이트로 구성된 카고 워크스페이스를 설계하세요 (종이 위 또는 실제로): `math-core` (순수 수학 함수를 가진 라이브러리), `math-cli` (`math-core`와 인수 파싱을 위한 `clap`을 사용하는 바이너리), `math-web` (`math-core`와 REST API를 위한 `axum`을 사용하는 바이너리). 세 개의 `Cargo.toml`과 루트 워크스페이스 `Cargo.toml`을 작성하세요. 공유 버전에는 `workspace.dependencies`를 사용하세요.

### 연습 4: 피처 플래그
두 개의 선택적 피처 `formal`과 `loud`를 가진 라이브러리 크레이트 `greeter`를 만드세요. 피처 없이는 `greet(name)`이 `"Hello, {name}"`을 반환합니다. `formal`을 사용하면 `"Good day, {name}. How do you do?"`를 반환합니다. `loud`를 사용하면 출력이 대문자가 됩니다. 두 피처가 동시에 활성화될 수 있습니다. `Cargo.toml`, `lib.rs`, 그리고 각 조합에 대한 테스트를 작성하세요.

### 연습 5: 의존성 감사
임의의 Rust 프로젝트를 사용하세요 (자신의 프로젝트 또는 GitHub에서 클론한 것). `cargo tree`와 `cargo tree -d`를 실행하여 중복 의존성을 찾으세요. 짧은 보고서를 작성하세요: 총 의존성 수는? 중복 수는? 가장 깊은 의존성 체인은? 그런 다음 `cargo audit`을 실행하고 (필요하다면 먼저 설치) 어드바이저리(Advisory)를 기록하세요.

---

## 참고 자료
- [The Rust Book — 모듈](https://doc.rust-lang.org/book/ch07-00-managing-growing-projects-with-packages-crates-and-modules.html)
- [Cargo Book — 의존성 지정](https://doc.rust-lang.org/cargo/reference/specifying-dependencies.html)
- [Cargo Book — 워크스페이스](https://doc.rust-lang.org/cargo/reference/workspaces.html)
- [Cargo Book — 피처](https://doc.rust-lang.org/cargo/reference/features.html)
- [crates.io 배포 가이드](https://doc.rust-lang.org/cargo/reference/publishing.html)
- [cargo-audit](https://github.com/rustsec/rustsec/tree/main/cargo-audit)

---

**이전**: [비동기 및 Await](./15_Async_Await.md) | **다음**: [안전하지 않은 Rust](./17_Unsafe_Rust.md)
