# 29. 성능과 프로파일링

**이전**: [고급 에러 처리](./12_Advanced_Error_Handling.md) | **다음**: [캡스톤: HTTP 서버](./14_Capstone_HTTP_Server.md)

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. `criterion`으로 신뢰할 수 있는 벤치마크를 작성하고 결과 해석하기
2. 플레임 그래프를 생성하고 읽어 성능 병목 지점 식별하기
3. `perf`, `Instruments` 등의 시스템 프로파일러를 Rust 코드와 함께 사용하기
4. 메모리 사용량 프로파일링, 누수 탐지, 할당 줄이기
5. SIMD 힌트와 데이터 지향 설계(Data-Oriented Design) 적용하기

---

Rust는 성능을 위한 도구를 제공하지만, 측정 방법을 아는 것이 최적화 방법을 아는 것만큼 중요합니다. 이 레슨은 완전한 프로파일링 워크플로우를 다룹니다: 먼저 벤치마크, 병목 지점 찾기 위해 프로파일링, 증거 기반 최적화, 그리고 개선 검증.

## 목차
1. [최적화 워크플로우](#1-최적화-워크플로우)
2. [Criterion으로 벤치마킹](#2-criterion으로-벤치마킹)
3. [플레임 그래프](#3-플레임-그래프)
4. [perf와 시스템 프로파일러](#4-perf와-시스템-프로파일러)
5. [메모리 프로파일링](#5-메모리-프로파일링)
6. [할당 프로파일링](#6-할당-프로파일링)
7. [컴파일러 최적화](#7-컴파일러-최적화)
8. [데이터 지향 설계](#8-데이터-지향-설계)
9. [SIMD와 벡터화](#9-simd와-벡터화)
10. [비동기 성능](#10-비동기-성능)
11. [프로파일링 체크리스트](#11-프로파일링-체크리스트)
12. [연습문제](#12-연습문제)

---

## 1. 최적화 워크플로우

```
1. 성능 요구사항 정의 (지연 시간, 처리량, 메모리)
2. 핵심 경로에 대한 벤치마크 작성
3. 실제 병목 지점을 찾기 위해 프로파일링
4. 병목 지점 최적화 (느리다고 생각하는 곳이 아닌)
5. 개선을 검증하기 위해 다시 벤치마크
6. 3단계부터 반복
```

**규칙 #1**: 측정 없이 절대 최적화하지 마세요. 인간은 병목 지점을 추측하는 데 매우 서투릅니다.

---

## 2. Criterion으로 벤치마킹

Criterion은 회귀 감지 기능이 있는 통계적으로 엄밀한 벤치마크를 제공합니다:

```toml
[dev-dependencies]
criterion = { version = "0.5", features = ["html_reports"] }

[[bench]]
name = "my_benchmarks"
harness = false
```

### 기본 벤치마크

```rust
// benches/my_benchmarks.rs
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn fibonacci_recursive(n: u64) -> u64 {
    match n {
        0 => 0,
        1 => 1,
        _ => fibonacci_recursive(n - 1) + fibonacci_recursive(n - 2),
    }
}

fn fibonacci_iterative(n: u64) -> u64 {
    let (mut a, mut b) = (0u64, 1u64);
    for _ in 0..n {
        let temp = a + b;
        a = b;
        b = temp;
    }
    a
}

fn bench_fibonacci(c: &mut Criterion) {
    let mut group = c.benchmark_group("fibonacci");

    group.bench_function("recursive_20", |b| {
        b.iter(|| fibonacci_recursive(black_box(20)))
    });

    group.bench_function("iterative_20", |b| {
        b.iter(|| fibonacci_iterative(black_box(20)))
    });

    group.finish();
}

criterion_group!(benches, bench_fibonacci);
criterion_main!(benches);
```

```bash
cargo bench
# target/criterion/report/index.html에 HTML 보고서 생성
```

### 매개변수화된 벤치마크

```rust
use criterion::{BenchmarkId, Criterion, Throughput};

fn bench_sorting(c: &mut Criterion) {
    let mut group = c.benchmark_group("sorting");

    for size in [100, 1_000, 10_000, 100_000] {
        let data: Vec<u64> = (0..size).rev().collect();

        group.throughput(Throughput::Elements(size));

        group.bench_with_input(
            BenchmarkId::new("std_sort", size),
            &data,
            |b, data| {
                b.iter(|| {
                    let mut v = data.clone();
                    v.sort();
                    v
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("std_sort_unstable", size),
            &data,
            |b, data| {
                b.iter(|| {
                    let mut v = data.clone();
                    v.sort_unstable();
                    v
                })
            },
        );
    }

    group.finish();
}
```

### 구현 비교

```rust
fn bench_string_building(c: &mut Criterion) {
    let mut group = c.benchmark_group("string_building");
    let words: Vec<&str> = "the quick brown fox jumps over the lazy dog"
        .split_whitespace()
        .collect();

    group.bench_function("push_str", |b| {
        b.iter(|| {
            let mut s = String::new();
            for word in &words {
                s.push_str(word);
                s.push(' ');
            }
            s
        })
    });

    group.bench_function("format", |b| {
        b.iter(|| {
            words.iter().map(|w| *w).collect::<Vec<_>>().join(" ")
        })
    });

    group.bench_function("with_capacity", |b| {
        b.iter(|| {
            let cap: usize = words.iter().map(|w| w.len() + 1).sum();
            let mut s = String::with_capacity(cap);
            for word in &words {
                s.push_str(word);
                s.push(' ');
            }
            s
        })
    });

    group.finish();
}
```

### black_box

`black_box`는 컴파일러가 벤치마크 코드를 최적화하는 것을 방지합니다:

```rust
use criterion::black_box;

// 나쁜 예: 컴파일러가 전체 계산을 최적화할 수 있음
let result = fibonacci(20);

// 좋은 예: black_box가 데드코드 제거를 방지
let result = fibonacci(black_box(20));
black_box(result);  // 결과도 최적화되지 않도록 방지
```

---

## 3. 플레임 그래프

플레임 그래프는 프로그램이 CPU 시간을 어디서 소비하는지 시각화합니다:

```bash
# flamegraph 도구 설치
cargo install flamegraph

# 플레임 그래프 생성 (Linux에서는 perf, macOS에서는 DTrace 필요)
cargo flamegraph --bin my-app

# 특정 인수와 함께
cargo flamegraph --bin my-app -- --input large_file.dat

# 벤치마크용
cargo flamegraph --bench my_benchmarks -- --bench
```

### 플레임 그래프 읽는 법

```
            ┌─────────────────────────────────────┐
            │          program::main               │
            ├──────────────┬──────────────────────┤
            │ process_data │   serialize_output    │
            ├─────┬────────┤                      │
            │parse│ sort   │                      │
            └─────┴────────┴──────────────────────┘
너비 = 소비된 시간 (자식 포함)
```

- **넓은 막대** = 많은 시간 소비 → 최적화 대상
- **높은 스택** = 깊은 호출 체인
- **평평한 꼭대기** = 실제 작업 (리프 함수)
- 예상치 못하게 넓은 막대를 찾으세요 — 그것이 병목입니다

---

## 4. perf와 시스템 프로파일러

### perf (Linux)

```bash
# 프로파일 기록
perf record -g --call-graph dwarf target/release/my-app

# 프로파일 조회
perf report

# 탑다운 뷰
perf stat target/release/my-app
# 보여주는 것: 명령어, 사이클, 캐시 미스, 브랜치 미스 등
```

### Instruments (macOS)

```bash
# 커맨드 라인에서 Instruments로 프로파일
xcrun xctrace record --template "Time Profiler" --launch target/release/my-app

# 또는 Instruments GUI 열기
open -a Instruments
```

### 프로파일링을 위한 컴파일

```toml
# Cargo.toml — 디버그 정보가 있는 릴리스 프로파일
[profile.release]
debug = true       # 프로파일링을 위한 디버그 심볼 유지
opt-level = 3      # 완전한 최적화

# 또는 전용 프로파일링 프로파일
[profile.profiling]
inherits = "release"
debug = true
```

```bash
cargo build --profile profiling
```

---

## 5. 메모리 프로파일링

### DHAT로 할당 추적

```toml
[dev-dependencies]
dhat = "0.3"
```

```rust
#[cfg(feature = "dhat-heap")]
#[global_allocator]
static ALLOC: dhat::Alloc = dhat::Alloc;

fn main() {
    #[cfg(feature = "dhat-heap")]
    let _profiler = dhat::Profiler::new_heap();

    // 프로그램 코드
    let data: Vec<String> = (0..10000)
        .map(|i| format!("item_{i}"))
        .collect();

    process(&data);
}

fn process(data: &[String]) {
    // 처리 중...
    let _filtered: Vec<&String> = data.iter()
        .filter(|s| s.len() > 8)
        .collect();
}
```

```bash
cargo run --features dhat-heap
# dhat-heap.json 생성 — https://nnethercote.github.io/dh_view/dh_view.html에서 열기
```

### 피크 메모리 측정

```rust
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

struct TrackingAllocator;

static ALLOCATED: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for TrackingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = System.alloc(layout);
        if !ptr.is_null() {
            let current = ALLOCATED.fetch_add(layout.size(), Ordering::Relaxed) + layout.size();
            PEAK.fetch_max(current, Ordering::Relaxed);
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout);
        ALLOCATED.fetch_sub(layout.size(), Ordering::Relaxed);
    }
}

#[global_allocator]
static GLOBAL: TrackingAllocator = TrackingAllocator;

fn main() {
    // 코드 실행
    let v: Vec<u8> = vec![0; 1_000_000];
    drop(v);

    println!("현재: {} 바이트", ALLOCATED.load(Ordering::Relaxed));
    println!("피크: {} 바이트", PEAK.load(Ordering::Relaxed));
}
```

---

## 6. 할당 프로파일링

### 할당 줄이기

```rust
// 나쁜 예: 반복마다 새 String 할당
fn process_bad(items: &[&str]) -> Vec<String> {
    items.iter()
        .map(|s| format!("processed: {s}"))
        .collect()
}

// 더 좋은 예: 용량 사전 할당
fn process_better(items: &[&str]) -> Vec<String> {
    let mut result = Vec::with_capacity(items.len());
    for s in items {
        result.push(format!("processed: {s}"));
    }
    result
}

// 최선: Cow로 할당 완전 회피
use std::borrow::Cow;

fn process_best<'a>(items: &[&'a str]) -> Vec<Cow<'a, str>> {
    items.iter()
        .map(|&s| {
            if s.starts_with("processed:") {
                Cow::Borrowed(s)  // 할당 불필요
            } else {
                Cow::Owned(format!("processed: {s}"))
            }
        })
        .collect()
}
```

### SmallVec과 ArrayVec

```rust
use smallvec::SmallVec;

// SmallVec은 N개 요소를 인라인(스택)에 저장
// 인라인 용량을 초과할 때만 힙 할당
fn collect_small_results() -> SmallVec<[u32; 8]> {
    let mut results = SmallVec::new();
    for i in 0..5 {
        results.push(i * 2);
    }
    results  // 힙 할당 없음! 5개 요소 모두 인라인 버퍼에 들어감
}

// arrayvec::ArrayVec은 완전히 스택 기반 (오버플로우 시 패닉)
use arrayvec::ArrayVec;

fn fixed_buffer() -> ArrayVec<u32, 16> {
    let mut buf = ArrayVec::new();
    buf.push(1);
    buf.push(2);
    buf.push(3);
    buf
}
```

### 문자열 인터닝 (String Interning)

```rust
use std::collections::HashMap;

struct StringInterner {
    strings: Vec<String>,
    lookup: HashMap<String, usize>,
}

impl StringInterner {
    fn new() -> Self {
        Self {
            strings: Vec::new(),
            lookup: HashMap::new(),
        }
    }

    fn intern(&mut self, s: &str) -> usize {
        if let Some(&id) = self.lookup.get(s) {
            return id;
        }
        let id = self.strings.len();
        self.strings.push(s.to_string());
        self.lookup.insert(s.to_string(), id);
        id
    }

    fn resolve(&self, id: usize) -> &str {
        &self.strings[id]
    }
}

fn main() {
    let mut interner = StringInterner::new();

    // 같은 문자열은 한 번만 저장
    let a = interner.intern("hello");
    let b = interner.intern("world");
    let c = interner.intern("hello");  // `a`와 같은 ID 반환

    assert_eq!(a, c);
    println!("{} == {} : {}", a, c, a == c);  // 0 == 0 : true
}
```

---

## 7. 컴파일러 최적화

### 어셈블리 출력 검사

```bash
# 생성된 어셈블리 보기
cargo rustc --release -- --emit asm
# target/release/deps/*.s에 출력

# 또는 cargo-show-asm 사용
cargo install cargo-show-asm
cargo asm my_crate::my_function
```

### Godbolt Compiler Explorer

[godbolt.org](https://godbolt.org/)에서 Rust로 어셈블리 출력을 인터랙티브하게 확인하세요.

### 최적화 힌트

```rust
// 브랜치가 드물다는 힌트 (nightly)
// #[cold]
fn error_handler() {
    // 이 함수는 드물게 호출됨
}

// 인라인 힌트
#[inline]          // 인라인 제안
fn hot_function(x: u32) -> u32 { x * 2 }

#[inline(always)]  // 인라인 강제
fn critical_path(x: u32) -> u32 { x + 1 }

#[inline(never)]   // 인라인 방지 (프로파일링 가시성용)
fn cold_function() { /* ... */ }

// 경계 검사 제거
fn sum_slice(data: &[u32]) -> u32 {
    // 컴파일러가 이것을 자동 벡터화할 수 있음
    data.iter().sum()
}

// unreachable 힌트로 컴파일러 도움
fn safe_divide(a: u32, b: u32) -> u32 {
    if b == 0 {
        unreachable!("0으로 나누기는 이전에 처리됐어야 함");
    }
    a / b
}

// 경계가 증명된 경우 get_unchecked 사용 (unsafe)
fn sum_range(data: &[u32], start: usize, end: usize) -> u32 {
    assert!(end <= data.len() && start <= end);
    let mut sum = 0;
    for i in start..end {
        // 컴파일러는 위의 assert 덕분에 경계가 유효함을 알고 있음
        sum += data[i];
    }
    sum
}
```

---

## 8. 데이터 지향 설계

### 구조체 배열 (AoS) vs 배열 구조체 (SoA)

```rust
// 구조체 배열 (AoS) — 필드별 연산에 캐시 지역성 나쁨
struct ParticleAoS {
    x: f64,
    y: f64,
    z: f64,
    mass: f64,
    vx: f64,
    vy: f64,
    vz: f64,
    charge: f64,
}

fn sum_masses_aos(particles: &[ParticleAoS]) -> f64 {
    // 캐시 라인이 x, y, z, mass를 로드하지만 mass만 필요
    // 로드된 캐시 데이터의 75%가 낭비됨
    particles.iter().map(|p| p.mass).sum()
}

// 배열 구조체 (SoA) — 캐시 지역성 우수
struct ParticlesSoA {
    x: Vec<f64>,
    y: Vec<f64>,
    z: Vec<f64>,
    mass: Vec<f64>,
    vx: Vec<f64>,
    vy: Vec<f64>,
    vz: Vec<f64>,
    charge: Vec<f64>,
}

fn sum_masses_soa(particles: &ParticlesSoA) -> f64 {
    // 캐시 라인이 100% 유용 — mass 값만 로드됨
    // 자동 벡터화도 간단하게 됨
    particles.mass.iter().sum()
}
```

### 캐시 친화적 데이터 접근

```rust
// 나쁜 예: 랜덤 접근 패턴 — 캐시 미스
fn random_access(data: &[u64], indices: &[usize]) -> u64 {
    indices.iter().map(|&i| data[i]).sum()
}

// 좋은 예: 순차 접근 패턴 — 캐시 친화적
fn sequential_access(data: &[u64]) -> u64 {
    data.iter().sum()
}

// 더 나은 캐시 활용을 위한 청크 처리
fn process_chunked(data: &mut [f64], chunk_size: usize) {
    for chunk in data.chunks_mut(chunk_size) {
        for value in chunk.iter_mut() {
            *value = (*value * 2.0).sqrt();
        }
    }
}
```

---

## 9. SIMD와 벡터화

### 자동 벡터화 (Auto-Vectorization)

컴파일러는 많은 간단한 루프를 자동으로 벡터화합니다:

```rust
// 이것은 자동 벡터화됨 (cargo asm으로 확인)
fn add_vectors(a: &[f32], b: &[f32], result: &mut [f32]) {
    assert_eq!(a.len(), b.len());
    assert_eq!(a.len(), result.len());

    for i in 0..a.len() {
        result[i] = a[i] + b[i];
    }
}

// 정확한 청크로 자동 벡터화 지원
fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| x * y)
        .sum()
}
```

### std::simd를 이용한 명시적 SIMD (Nightly)

```rust
// Nightly 전용
#![feature(portable_simd)]
use std::simd::f32x8;

fn simd_add(a: &[f32], b: &[f32], result: &mut [f32]) {
    let chunks = a.len() / 8;

    for i in 0..chunks {
        let va = f32x8::from_slice(&a[i * 8..]);
        let vb = f32x8::from_slice(&b[i * 8..]);
        let vr = va + vb;
        vr.copy_to_slice(&mut result[i * 8..]);
    }

    // 나머지 처리
    for i in (chunks * 8)..a.len() {
        result[i] = a[i] + b[i];
    }
}
```

### 대상 기능 (Target Features)

```rust
// AVX2 지원으로 컴파일
// RUSTFLAGS="-C target-feature=+avx2" cargo build --release

// 또는 함수별 대상 지정
#[target_feature(enable = "avx2")]
unsafe fn fast_sum(data: &[f32]) -> f32 {
    data.iter().sum()
}

// 런타임 감지
fn sum_dispatch(data: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { fast_sum(data) };
        }
    }
    data.iter().sum()  // 폴백
}
```

---

## 10. 비동기 성능

### 비동기 태스크 오버헤드 측정

```rust
use tokio::time::Instant;

#[tokio::main]
async fn main() {
    // 태스크 스폰 오버헤드 측정
    let start = Instant::now();
    let mut handles = Vec::new();

    for _ in 0..10_000 {
        handles.push(tokio::spawn(async {}));
    }

    for h in handles {
        h.await.unwrap();
    }

    println!("10k 태스크 스폰: {:?}", start.elapsed());

    // 채널 처리량 측정
    let (tx, mut rx) = tokio::sync::mpsc::channel::<u64>(1000);

    let start = Instant::now();
    let count = 100_000u64;

    tokio::spawn(async move {
        for i in 0..count {
            tx.send(i).await.unwrap();
        }
    });

    let mut received = 0;
    while let Some(_) = rx.recv().await {
        received += 1;
        if received == count {
            break;
        }
    }

    let elapsed = start.elapsed();
    println!(
        "{count}개 메시지를 {:?}에 ({:.0} msg/s)",
        elapsed,
        count as f64 / elapsed.as_secs_f64()
    );
}
```

### Tokio Console

```toml
[dependencies]
console-subscriber = "0.4"
```

```rust
#[tokio::main]
async fn main() {
    console_subscriber::init();
    // 비동기 애플리케이션...
}
```

```bash
# 다른 터미널에서
tokio-console
# 실시간 태스크 메트릭, waker 수, 폴링 시간 표시
```

---

## 11. 프로파일링 체크리스트

```
최적화 전:
□ 핵심 경로에 대한 벤치마크 존재
□ 성능 요구사항 정의됨 (지연 시간, 처리량)
□ 릴리스 빌드로 프로파일링 수행 (디버그 빌드는 10-100배 느림)

흔한 병목:
□ 불필요한 할당 (String, Vec, Box)
□ 과도한 클로닝 (빌림으로 충분한 곳에서 clone())
□ 캐시 비친화적 데이터 레이아웃
□ 비동기 코드에서의 블로킹
□ 락 경합
□ 직렬화/역직렬화
□ 버퍼링 없는 I/O

최적화 기법:
□ 컬렉션 사전 할당 (Vec::with_capacity)
□ 클로닝 대신 참조/Cow 사용
□ 배치 작업 (시스템 콜 감소, 락 획득 감소)
□ 적절한 자료구조 사용 (HashMap vs BTreeMap vs Vec)
□ I/O 버퍼링 (BufReader, BufWriter)
□ 연결 및 비용이 많이 드는 리소스 풀링
□ 비동기 코드에서 CPU 집약적 작업에 spawn_blocking 사용
```

---

## 12. 연습문제

1. **벤치마크 비교**: N=10, 100, 1000, 10000 요소에 대해 `HashMap` vs `BTreeMap` vs `Vec`(선형 탐색) 조회를 비교하는 Criterion 벤치마크를 작성하세요. 결과를 그래프로 그리고 교차점을 설명하세요.

2. **할당 감사**: 대규모 텍스트 파일을 처리하는 Rust 프로그램(예: 단어 빈도 카운터)의 할당을 DHAT으로 프로파일링하세요. `Cow`, 사전 할당, 문자열 인터닝을 사용하여 할당을 50% 이상 줄이세요.

3. **SoA 변환**: 구조체 배열 레이아웃을 사용하는 입자 시뮬레이션을 가져와 배열 구조체로 변환하고 개선을 벤치마크하세요. `perf stat`으로 캐시 미스율을 측정하세요.

4. **플레임 그래프 분석**: 대규모 JSON 파일을 파싱하고, 데이터를 변환하고, CSV 출력을 쓰는 프로그램을 만드세요. 플레임 그래프를 생성하고, 가장 뜨거운 함수를 식별하고, 최적화하세요. 전후 성능을 문서화하세요.

5. **SIMD 내적**: 내적 함수를 세 가지 방식으로 구현하세요: (1) 나이브 루프, (2) 이터레이터 체인, (3) 명시적 SIMD (`packed_simd2` 또는 `std::simd` 크레이트 사용). 세 가지를 모두 벤치마크하고 컴파일러의 자동 벡터화 출력과 비교하세요.

---

## 참고 자료

- [Criterion documentation](https://bheisler.github.io/criterion.rs/book/)
- [The Rust Performance Book](https://nnethercote.github.io/perf-book/)
- [cargo-flamegraph](https://github.com/flamegraph-rs/flamegraph)
- [DHAT documentation](https://docs.rs/dhat/latest/dhat/)
- [Tokio Console](https://github.com/tokio-rs/console)
- [Data-Oriented Design (Andrew Kelley talk)](https://www.youtube.com/watch?v=yOyaJXpAYZQ)

---

**이전**: [고급 에러 처리](./12_Advanced_Error_Handling.md) | **다음**: [캡스톤: HTTP 서버](./14_Capstone_HTTP_Server.md)
