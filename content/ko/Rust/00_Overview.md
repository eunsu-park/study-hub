# Rust 프로그래밍

## 토픽 개요

Rust는 **안전성(Safety)**, **동시성(Concurrency)**, **성능(Performance)**에 초점을 맞춘 시스템 프로그래밍 언어입니다. Rust의 독창적인 소유권(Ownership) 시스템은 데이터 경쟁(Data Race), 널 포인터 역참조(Null Pointer Dereference), 댕글링 참조(Dangling Reference) 등 다양한 버그를 런타임 비용 없이 컴파일 타임에 원천 차단합니다. Rust는 개발자 설문에서 가장 사랑받는 프로그래밍 언어로 지속적으로 선정되며, Mozilla, AWS, Google, Microsoft, Cloudflare의 핵심 인프라를 구동합니다.

이 토픽은 기본적인 프로그래밍 지식을 전제로 하며, Rust의 기초부터 고급 비동기 프로그래밍까지 다룹니다. 소유권 모델(레슨 03–05)이 개념적 핵심이며, 나머지 모든 내용은 이를 기반으로 합니다.

## 학습 경로

```
기초(Foundations)               핵심 개념(Core Concepts)         고급(Advanced)
─────────────────              ─────────────────                ─────────────────
01 Getting Started             03 Ownership          ★★★       11 Lifetimes        ★★★★
02 Variables & Types           04 Borrowing          ★★★       13 Smart Pointers   ★★★
                               05 Slices             ★★        14 Concurrency      ★★★★
데이터 모델링                   06 Structs & Methods  ★★        15 Async/Await      ★★★★
─────────────────              07 Enums & Patterns   ★★★       17 Unsafe Rust      ★★★★
08 Collections                 09 Error Handling     ★★★
10 Traits & Generics           12 Closures & Iters   ★★★       프로젝트(Project)
16 Modules & Cargo                                              ─────────────────
                                                                18 CLI Tool Project ★★★
```

## 레슨 목록

| # | 레슨 | 난이도 | 핵심 개념 |
|---|------|--------|-----------|
| 01 | [시작하기](./01_Getting_Started.md) | ⭐ | rustup, cargo, Hello World |
| 02 | [변수와 타입](./02_Variables_and_Types.md) | ⭐ | let/mut, 섀도잉(shadowing), 스칼라/복합 타입 |
| 03 | [소유권](./03_Ownership.md) | ⭐⭐⭐ | 스택(Stack)/힙(Heap), 이동 의미론(Move Semantics), Copy/Clone |
| 04 | [빌림과 참조](./04_Borrowing_and_References.md) | ⭐⭐⭐ | &T, &mut T, 빌림 규칙 |
| 05 | [슬라이스](./05_Slices.md) | ⭐⭐ | &str vs String, 배열 슬라이스 |
| 06 | [구조체와 메서드](./06_Structs_and_Methods.md) | ⭐⭐ | struct, impl, #[derive] |
| 07 | [열거형과 패턴 매칭](./07_Enums_and_Pattern_Matching.md) | ⭐⭐⭐ | enum, Option, match, if let |
| 08 | [컬렉션](./08_Collections.md) | ⭐⭐ | Vec, HashMap, 이터레이터 체이닝 |
| 09 | [에러 처리](./09_Error_Handling.md) | ⭐⭐⭐ | Result, ?, thiserror/anyhow |
| 10 | [트레이트와 제네릭](./10_Traits_and_Generics.md) | ⭐⭐⭐ | trait, impl Trait, 제네릭(generics), where 절 |
| 11 | [라이프타임](./11_Lifetimes.md) | ⭐⭐⭐⭐ | 라이프타임 어노테이션, 생략 규칙, 'static |
| 12 | [클로저와 이터레이터](./12_Closures_and_Iterators.md) | ⭐⭐⭐ | Fn/FnMut/FnOnce, map/filter/fold |
| 13 | [스마트 포인터](./13_Smart_Pointers.md) | ⭐⭐⭐ | Box, Rc, RefCell, Arc |
| 14 | [동시성](./14_Concurrency.md) | ⭐⭐⭐⭐ | thread::spawn, 채널(channels), Mutex, Send/Sync |
| 15 | [비동기와 Await](./15_Async_Await.md) | ⭐⭐⭐⭐ | async fn, Future, Tokio 런타임 |
| 16 | [모듈과 Cargo](./16_Modules_and_Cargo.md) | ⭐⭐ | mod/use, Cargo.toml, 워크스페이스(workspaces) |
| 17 | [Unsafe Rust](./17_Unsafe_Rust.md) | ⭐⭐⭐⭐ | unsafe 블록, 원시 포인터(raw pointers), FFI |
| 18 | [프로젝트: CLI 도구](./18_Project_CLI_Tool.md) | ⭐⭐⭐ | clap + serde + tokio CLI 프로젝트 |

## 사전 요구사항

- 임의의 언어로 기본 프로그래밍 경험 (변수, 함수, 제어 흐름)
- 커맨드 라인 사용에 익숙할 것
- 선택 사항: C/C++ 경험이 있으면 시스템 개념 이해에 도움이 됨

## 개발 환경

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

## 예제 코드

이 토픽의 실행 가능한 예제는 [`examples/Rust/`](../../../examples/Rust/)에 있습니다.
