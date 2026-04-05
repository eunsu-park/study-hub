# Rust 고급

Rust 기초를 바탕으로, 고급 언어 기능, 시스템 프로그래밍, 그리고 Rust 생태계를 다룹니다. unsafe 코드, 매크로, FFI, WebAssembly, 임베디드 Rust를 다루며, 캡스톤 프로젝트로 프로덕션 HTTP 서버를 구축합니다.

## 학습 내용

- **Unsafe Rust**: 원시 포인터, unsafe 블록, 안전성 불변량
- **매크로**: 선언적(macro_rules!) 및 절차적(derive, attribute) 매크로
- **고급 트레이트**: GATs, 트레이트 객체, 블랭킷 구현, 봉인된 트레이트
- **고급 비동기**: Tokio 내부, select!, 스트림, Tower 미들웨어
- **FFI와 상호운용**: C 상호운용, bindgen/cbindgen, PyO3
- **WebAssembly**: wasm-pack, wasm-bindgen, WASI, Yew
- **임베디드 Rust**: no_std, embedded-hal, RTIC
- **네트워킹**: TCP/UDP, Axum, WebSocket, TLS
- **에러 처리**: thiserror, anyhow, 복구 전략의 고급 패턴
- **성능**: criterion, 플레임그래프, SIMD, 데이터 지향 설계

## 사전 요구 사항

- [Rust 기초](../Rust_Basics/00_Overview.md) — Rust 소유권, 트레이트, 동시성, 비동기, Cargo

## 학습 경로

```
                         Rust 고급 — 학습 경로
  ┌─────────────────────────────────────────────────────────────────────────┐
  │                                                                         │
  │  기초                                 매크로와 트레이트                  │
  │  ─────────────────                    ─────────────────                 │
  │  01 Unsafe Rust                       04 선언적 매크로                   │
  │    ──▶ 02 CLI 도구 (프로젝트)          05 절차적 매크로                   │
  │         ──▶ 03 빌드 시스템             06 고급 트레이트                   │
  │                                       07 고급 비동기                     │
  │                                                                         │
  │  에코시스템                            운영                              │
  │  ─────────────────                    ─────────────────                 │
  │  08 FFI와 상호운용                     12 고급 에러 처리                  │
  │  09 WebAssembly                       13 성능과 프로파일링               │
  │  10 임베디드 Rust                                                       │
  │  11 네트워크 프로그래밍                프로젝트                          │
  │                                       ─────────────────                 │
  │                                       14 캡스톤: HTTP 서버               │
  │                                                                         │
  └─────────────────────────────────────────────────────────────────────────┘
```

## 레슨 목록

| # | 레슨 | 난이도 | 핵심 개념 |
|---|------|--------|----------|
| 01 | [Unsafe Rust](01_Unsafe_Rust.md) | ⭐⭐⭐⭐ | unsafe 블록, 원시 포인터, FFI |
| 02 | [프로젝트: CLI 도구](02_Project_CLI_Tool.md) | ⭐⭐⭐ | clap + serde + tokio CLI 프로젝트 |
| 03 | [빌드 시스템 심층 분석](03_Build_System.md) | ⭐⭐⭐ | 워크스페이스, 피처 플래그, build.rs, 크로스 컴파일 |
| 04 | [선언적 매크로](04_Declarative_Macros.md) | ⭐⭐⭐ | macro_rules!, 반복, 프래그먼트 지정자 |
| 05 | [절차적 매크로](05_Procedural_Macros.md) | ⭐⭐⭐⭐ | 디라이브 매크로, syn/quote, 속성 매크로 |
| 06 | [고급 트레이트](06_Advanced_Traits.md) | ⭐⭐⭐⭐ | GATs, 트레이트 객체, 블랭킷 구현, 봉인된 트레이트 |
| 07 | [고급 비동기](07_Advanced_Async.md) | ⭐⭐⭐⭐ | Tokio 내부, select!, 스트림, Tower |
| 08 | [FFI와 상호운용](08_FFI_and_Interop.md) | ⭐⭐⭐⭐ | C 상호운용, bindgen/cbindgen, PyO3 |
| 09 | [WebAssembly](09_WebAssembly.md) | ⭐⭐⭐ | wasm-pack, wasm-bindgen, WASI, Yew |
| 10 | [임베디드 Rust](10_Embedded_Rust.md) | ⭐⭐⭐⭐ | no_std, embedded-hal, RTIC, probe-rs |
| 11 | [네트워크 프로그래밍](11_Network_Programming.md) | ⭐⭐⭐ | TCP/UDP, Axum, WebSocket, TLS |
| 12 | [고급 에러 처리](12_Advanced_Error_Handling.md) | ⭐⭐⭐ | thiserror, anyhow, 복구 패턴 |
| 13 | [성능과 프로파일링](13_Performance_Profiling.md) | ⭐⭐⭐⭐ | criterion, 플레임그래프, SIMD, 데이터 지향 설계 |
| 14 | [캡스톤: HTTP 서버](14_Capstone_HTTP_Server.md) | ⭐⭐⭐⭐ | Axum + SQLx + JWT + 미들웨어 프로젝트 |

## 개발 환경

```bash
rustc --version   # Rust 1.75+ 권장
cargo --version
```

예제 코드는 `examples/Rust_Advanced/`에서 확인할 수 있습니다.

## 관련 자료

- [Rust 기초](../Rust_Basics/00_Overview.md) — Rust 기초, 소유권, 트레이트, 동시성
- [C 고급](../C_Advanced/00_Overview.md) — 비교를 위한 C 시스템 프로그래밍
- [Linux](../Linux/00_Overview.md) — 임베디드 및 FFI 작업을 위한 Linux 시스템 지식

---

**License**: Content licensed under CC BY-NC 4.0
