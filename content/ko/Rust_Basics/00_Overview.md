# Rust 기초

Rust는 **안전성**, **동시성**, **성능**에 초점을 맞춘 시스템 프로그래밍 언어입니다. 독창적인 소유권 시스템이 데이터 경쟁, 널 포인터 역참조, 댕글링 참조 등 다양한 버그를 런타임 비용 없이 컴파일 타임에 원천 차단합니다. 이 토픽은 Rust의 기초부터 동시성, 비동기 프로그래밍, Cargo를 활용한 프로젝트 구조화까지 다룹니다.

## 학습 내용

- **시작하기**: rustup, cargo, 툴체인 설정
- **변수와 타입**: let/mut, 섀도잉, 스칼라/복합 타입
- **소유권 모델**: 스택/힙, 이동 의미론, 빌림, 참조, 슬라이스
- **데이터 모델링**: 구조체, 열거형, 패턴 매칭, 컬렉션
- **에러 처리**: Result, ?, thiserror/anyhow 패턴
- **트레이트와 제네릭**: 트레이트 설계, impl Trait, 제네릭 프로그래밍
- **라이프타임**: 라이프타임 어노테이션, 생략 규칙, 'static
- **클로저와 이터레이터**: Fn 트레이트, map/filter/fold 체인
- **스마트 포인터**: Box, Rc, RefCell, Arc
- **동시성**: 스레드, 채널, Mutex, Send/Sync
- **비동기/Await**: async fn, Future, Tokio 런타임
- **모듈과 Cargo**: mod/use, 워크스페이스, Cargo.toml

## 사전 요구 사항

- [Programming](../Programming/00_Overview.md) — 변수, 함수, 제어 흐름에 대한 기본 이해 (어떤 언어든 가능)

## 학습 경로

```
                          Rust 기초 — 학습 경로
  ┌─────────────────────────────────────────────────────────────────────────┐
  │                                                                         │
  │  ┌──────────────┐   ┌──────────────────┐   ┌────────────────────────┐  │
  │  │ 01 시작하기    │──▶│ 02 변수와 타입    │──▶│ 03 소유권              │  │
  │  │              │   │                  │   │                        │  │
  │  └──────────────┘   └──────────────────┘   └────────────┬───────────┘  │
  │                                                          │              │
  │                                                          ▼              │
  │  ┌──────────────┐   ┌──────────────────┐   ┌────────────────────────┐  │
  │  │ 06 구조체와   │◀──│ 05 슬라이스       │◀──│ 04 빌림과 참조          │  │
  │  │    메서드     │   │                  │   │                        │  │
  │  └──────┬───────┘   └──────────────────┘   └────────────────────────┘  │
  │         │                                                               │
  │         ▼                                                               │
  │  ┌──────────────┐   ┌──────────────────┐   ┌────────────────────────┐  │
  │  │ 07 열거형과   │──▶│ 08 컬렉션         │──▶│ 09 에러 처리           │  │
  │  │    패턴 매칭  │   │                  │   │                        │  │
  │  └──────────────┘   └──────────────────┘   └────────────┬───────────┘  │
  │                                                          │              │
  │                                                          ▼              │
  │  ┌──────────────┐   ┌──────────────────┐   ┌────────────────────────┐  │
  │  │ 10 트레이트와 │──▶│ 11 라이프타임     │──▶│ 12 클로저와             │  │
  │  │    제네릭     │   │                  │   │    이터레이터           │  │
  │  └──────────────┘   └──────────────────┘   └────────────┬───────────┘  │
  │                                                          │              │
  │                                                          ▼              │
  │  ┌──────────────┐   ┌──────────────────┐   ┌────────────────────────┐  │
  │  │ 13 스마트     │──▶│ 14 동시성         │──▶│ 15 비동기/Await         │  │
  │  │    포인터     │   │                  │   │    ──▶ 16 모듈과 Cargo │  │
  │  └──────────────┘   └──────────────────┘   └────────────────────────┘  │
  │                                                                         │
  └─────────────────────────────────────────────────────────────────────────┘
```

## 레슨 목록

| # | 레슨 | 난이도 | 핵심 개념 |
|---|------|--------|----------|
| 01 | [시작하기](01_Getting_Started.md) | ⭐ | rustup, cargo, Hello World |
| 02 | [변수와 타입](02_Variables_and_Types.md) | ⭐ | let/mut, 섀도잉, 스칼라/복합 타입 |
| 03 | [소유권](03_Ownership.md) | ⭐⭐⭐ | 스택/힙, 이동 의미론, Copy/Clone |
| 04 | [빌림과 참조](04_Borrowing_and_References.md) | ⭐⭐⭐ | &T, &mut T, 빌림 규칙 |
| 05 | [슬라이스](05_Slices.md) | ⭐⭐ | &str vs String, 배열 슬라이스 |
| 06 | [구조체와 메서드](06_Structs_and_Methods.md) | ⭐⭐ | struct, impl, #[derive] |
| 07 | [열거형과 패턴 매칭](07_Enums_and_Pattern_Matching.md) | ⭐⭐⭐ | enum, Option, match, if let |
| 08 | [컬렉션](08_Collections.md) | ⭐⭐ | Vec, HashMap, 이터레이터 체이닝 |
| 09 | [에러 처리](09_Error_Handling.md) | ⭐⭐⭐ | Result, ?, thiserror/anyhow |
| 10 | [트레이트와 제네릭](10_Traits_and_Generics.md) | ⭐⭐⭐ | trait, impl Trait, 제네릭, where 절 |
| 11 | [라이프타임](11_Lifetimes.md) | ⭐⭐⭐⭐ | 라이프타임 어노테이션, 생략 규칙, 'static |
| 12 | [클로저와 이터레이터](12_Closures_and_Iterators.md) | ⭐⭐⭐ | Fn/FnMut/FnOnce, map/filter/fold |
| 13 | [스마트 포인터](13_Smart_Pointers.md) | ⭐⭐⭐ | Box, Rc, RefCell, Arc |
| 14 | [동시성](14_Concurrency.md) | ⭐⭐⭐⭐ | thread::spawn, 채널, Mutex, Send/Sync |
| 15 | [비동기와 Await](15_Async_Await.md) | ⭐⭐⭐⭐ | async fn, Future, Tokio 런타임 |
| 16 | [모듈과 Cargo](16_Modules_and_Cargo.md) | ⭐⭐ | mod/use, Cargo.toml, 워크스페이스 |

## 개발 환경

```bash
# rustup으로 Rust 설치
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# 설치 확인
rustc --version
cargo --version

# 유용한 컴포넌트
rustup component add clippy        # 린터
rustup component add rustfmt       # 포매터
rustup component add rust-analyzer # LSP (IDE 지원)
```

예제 코드는 `examples/Rust_Basics/`에서 확인할 수 있습니다.

## 관련 자료

- [Rust (고급)](../Rust_Advanced/00_Overview.md) — Unsafe, 매크로, FFI, WebAssembly, 임베디드, 네트워킹, 성능
- [Programming](../Programming/00_Overview.md) — 언어 독립적 프로그래밍 개념

---

**License**: Content licensed under CC BY-NC 4.0
