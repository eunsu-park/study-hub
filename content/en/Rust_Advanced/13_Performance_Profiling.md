# 29. Performance and Profiling

**Previous**: [Advanced Error Handling](./12_Advanced_Error_Handling.md) | **Next**: [Capstone: HTTP Server](./14_Capstone_HTTP_Server.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Write reliable benchmarks with `criterion` and interpret their results
2. Generate and read flame graphs to identify performance bottlenecks
3. Use `perf`, `Instruments`, and other system profilers with Rust code
4. Profile memory usage, detect leaks, and reduce allocations
5. Apply SIMD hints and data-oriented design for maximum throughput

---

Rust gives you the tools for performance, but knowing how to measure is just as important as knowing how to optimize. This lesson covers the complete profiling workflow: benchmark first, profile to find bottlenecks, optimize with evidence, then verify the improvement.

## Table of Contents
1. [The Optimization Workflow](#1-the-optimization-workflow)
2. [Benchmarking with Criterion](#2-benchmarking-with-criterion)
3. [Flame Graphs](#3-flame-graphs)
4. [perf and System Profilers](#4-perf-and-system-profilers)
5. [Memory Profiling](#5-memory-profiling)
6. [Allocation Profiling](#6-allocation-profiling)
7. [Compiler Optimizations](#7-compiler-optimizations)
8. [Data-Oriented Design](#8-data-oriented-design)
9. [SIMD and Vectorization](#9-simd-and-vectorization)
10. [Async Performance](#10-async-performance)
11. [Profiling Checklist](#11-profiling-checklist)
12. [Exercises](#12-exercises)

---

## 1. The Optimization Workflow

```
1. Define performance requirements (latency, throughput, memory)
2. Write benchmarks for the critical path
3. Profile to find the actual bottleneck
4. Optimize the bottleneck (not what you think is slow)
5. Benchmark again to verify improvement
6. Repeat from step 3
```

**Rule #1**: Never optimize without measuring first. Humans are terrible at guessing where bottlenecks are.

---

## 2. Benchmarking with Criterion

Criterion provides statistically rigorous benchmarks with regression detection:

```toml
[dev-dependencies]
criterion = { version = "0.5", features = ["html_reports"] }

[[bench]]
name = "my_benchmarks"
harness = false
```

### Basic Benchmark

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
# Opens HTML report in target/criterion/report/index.html
```

### Parameterized Benchmarks

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

### Comparing Implementations

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

`black_box` prevents the compiler from optimizing away your benchmarked code:

```rust
use criterion::black_box;

// BAD: compiler might optimize away the entire computation
let result = fibonacci(20);

// GOOD: black_box prevents dead-code elimination
let result = fibonacci(black_box(20));
black_box(result);  // Also prevent optimizing away the result
```

---

## 3. Flame Graphs

Flame graphs visualize where your program spends CPU time:

```bash
# Install flamegraph tool
cargo install flamegraph

# Generate a flame graph (requires perf on Linux, DTrace on macOS)
cargo flamegraph --bin my-app

# With specific arguments
cargo flamegraph --bin my-app -- --input large_file.dat

# For benchmarks
cargo flamegraph --bench my_benchmarks -- --bench
```

### Reading Flame Graphs

```
            ┌─────────────────────────────────────┐
            │          program::main               │
            ├──────────────┬──────────────────────┤
            │ process_data │   serialize_output    │
            ├─────┬────────┤                      │
            │parse│ sort   │                      │
            └─────┴────────┴──────────────────────┘
Width = time spent (including children)
```

- **Wide bars** = lots of time spent → optimization target
- **Tall stacks** = deep call chains
- **Flat tops** = actual work (leaf functions)
- Look for unexpectedly wide bars — those are your bottlenecks

---

## 4. perf and System Profilers

### perf (Linux)

```bash
# Record a profile
perf record -g --call-graph dwarf target/release/my-app

# View the profile
perf report

# Top-down view
perf stat target/release/my-app
# Shows: instructions, cycles, cache misses, branch misses, etc.
```

### Instruments (macOS)

```bash
# Profile with Instruments from command line
xcrun xctrace record --template "Time Profiler" --launch target/release/my-app

# Or open Instruments GUI
open -a Instruments
```

### Compile for Profiling

```toml
# Cargo.toml — release profile with debug info
[profile.release]
debug = true       # Keep debug symbols for profiling
opt-level = 3      # Full optimization

# Or a dedicated profiling profile
[profile.profiling]
inherits = "release"
debug = true
```

```bash
cargo build --profile profiling
```

---

## 5. Memory Profiling

### Tracking Allocations with DHAT

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

    // Your program here
    let data: Vec<String> = (0..10000)
        .map(|i| format!("item_{i}"))
        .collect();

    process(&data);
}

fn process(data: &[String]) {
    // Processing...
    let _filtered: Vec<&String> = data.iter()
        .filter(|s| s.len() > 8)
        .collect();
}
```

```bash
cargo run --features dhat-heap
# Creates dhat-heap.json — open at https://nnethercote.github.io/dh_view/dh_view.html
```

### Measuring Peak Memory

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
    // Your code here
    let v: Vec<u8> = vec![0; 1_000_000];
    drop(v);

    println!("Current: {} bytes", ALLOCATED.load(Ordering::Relaxed));
    println!("Peak: {} bytes", PEAK.load(Ordering::Relaxed));
}
```

---

## 6. Allocation Profiling

### Reducing Allocations

```rust
// BAD: Allocates a new String for every iteration
fn process_bad(items: &[&str]) -> Vec<String> {
    items.iter()
        .map(|s| format!("processed: {s}"))
        .collect()
}

// BETTER: Pre-allocate with capacity
fn process_better(items: &[&str]) -> Vec<String> {
    let mut result = Vec::with_capacity(items.len());
    for s in items {
        result.push(format!("processed: {s}"));
    }
    result
}

// BEST: Avoid allocation entirely by returning references or using Cow
use std::borrow::Cow;

fn process_best<'a>(items: &[&'a str]) -> Vec<Cow<'a, str>> {
    items.iter()
        .map(|&s| {
            if s.starts_with("processed:") {
                Cow::Borrowed(s)  // No allocation needed
            } else {
                Cow::Owned(format!("processed: {s}"))
            }
        })
        .collect()
}
```

### SmallVec and ArrayVec

```rust
use smallvec::SmallVec;

// SmallVec stores up to N elements inline (on the stack)
// Only heap-allocates when it exceeds the inline capacity
fn collect_small_results() -> SmallVec<[u32; 8]> {
    let mut results = SmallVec::new();
    for i in 0..5 {
        results.push(i * 2);
    }
    results  // No heap allocation! All 5 elements fit in the inline buffer
}

// arrayvec::ArrayVec is fully stack-based (panics on overflow)
use arrayvec::ArrayVec;

fn fixed_buffer() -> ArrayVec<u32, 16> {
    let mut buf = ArrayVec::new();
    buf.push(1);
    buf.push(2);
    buf.push(3);
    buf
}
```

### String Interning

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

    // Same string is stored only once
    let a = interner.intern("hello");
    let b = interner.intern("world");
    let c = interner.intern("hello");  // Returns same ID as `a`

    assert_eq!(a, c);
    println!("{} == {} : {}", a, c, a == c);  // 0 == 0 : true
}
```

---

## 7. Compiler Optimizations

### Inspecting Assembly Output

```bash
# View generated assembly
cargo rustc --release -- --emit asm
# Output in target/release/deps/*.s

# Or use cargo-show-asm
cargo install cargo-show-asm
cargo asm my_crate::my_function
```

### Godbolt Compiler Explorer

Use [godbolt.org](https://godbolt.org/) with Rust to see assembly output interactively.

### Optimization Hints

```rust
// Hint that a branch is unlikely (nightly)
// #[cold]
fn error_handler() {
    // This function is rarely called
}

// Inline hints
#[inline]          // Suggest inlining
fn hot_function(x: u32) -> u32 { x * 2 }

#[inline(always)]  // Force inlining
fn critical_path(x: u32) -> u32 { x + 1 }

#[inline(never)]   // Prevent inlining (for profiling visibility)
fn cold_function() { /* ... */ }

// Bounds checking elimination
fn sum_slice(data: &[u32]) -> u32 {
    // The compiler can auto-vectorize this
    data.iter().sum()
}

// Help the compiler with unreachable hints
fn safe_divide(a: u32, b: u32) -> u32 {
    if b == 0 {
        unreachable!("Division by zero should be caught earlier");
    }
    a / b
}

// Use get_unchecked when bounds are proven (unsafe)
fn sum_range(data: &[u32], start: usize, end: usize) -> u32 {
    assert!(end <= data.len() && start <= end);
    let mut sum = 0;
    for i in start..end {
        // Compiler knows bounds are valid due to the assert above
        sum += data[i];
    }
    sum
}
```

---

## 8. Data-Oriented Design

### Struct of Arrays vs Array of Structs

```rust
// Array of Structs (AoS) — poor cache locality for field-specific operations
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
    // Cache line loads x, y, z, mass — but we only need mass
    // 75% of loaded cache data is wasted
    particles.iter().map(|p| p.mass).sum()
}

// Struct of Arrays (SoA) — excellent cache locality
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
    // Cache lines are 100% useful — only mass values are loaded
    // Also auto-vectorizes trivially
    particles.mass.iter().sum()
}
```

### Cache-Friendly Data Access

```rust
// BAD: Random access pattern — cache misses
fn random_access(data: &[u64], indices: &[usize]) -> u64 {
    indices.iter().map(|&i| data[i]).sum()
}

// GOOD: Sequential access pattern — cache-friendly
fn sequential_access(data: &[u64]) -> u64 {
    data.iter().sum()
}

// Chunked processing for better cache utilization
fn process_chunked(data: &mut [f64], chunk_size: usize) {
    for chunk in data.chunks_mut(chunk_size) {
        for value in chunk.iter_mut() {
            *value = (*value * 2.0).sqrt();
        }
    }
}
```

---

## 9. SIMD and Vectorization

### Auto-Vectorization

The compiler automatically vectorizes many simple loops:

```rust
// This auto-vectorizes (check with cargo asm)
fn add_vectors(a: &[f32], b: &[f32], result: &mut [f32]) {
    assert_eq!(a.len(), b.len());
    assert_eq!(a.len(), result.len());

    for i in 0..a.len() {
        result[i] = a[i] + b[i];
    }
}

// Help auto-vectorization with exact chunks
fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| x * y)
        .sum()
}
```

### Explicit SIMD with std::simd (Nightly)

```rust
// Nightly only
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

    // Handle remainder
    for i in (chunks * 8)..a.len() {
        result[i] = a[i] + b[i];
    }
}
```

### Target Features

```rust
// Compile with AVX2 support
// RUSTFLAGS="-C target-feature=+avx2" cargo build --release

// Or per-function targeting
#[target_feature(enable = "avx2")]
unsafe fn fast_sum(data: &[f32]) -> f32 {
    data.iter().sum()
}

// Runtime detection
fn sum_dispatch(data: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { fast_sum(data) };
        }
    }
    data.iter().sum()  // Fallback
}
```

---

## 10. Async Performance

### Measuring Async Task Overhead

```rust
use tokio::time::Instant;

#[tokio::main]
async fn main() {
    // Measure task spawn overhead
    let start = Instant::now();
    let mut handles = Vec::new();

    for _ in 0..10_000 {
        handles.push(tokio::spawn(async {}));
    }

    for h in handles {
        h.await.unwrap();
    }

    println!("10k task spawns: {:?}", start.elapsed());

    // Measure channel throughput
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
        "{count} messages in {:?} ({:.0} msg/s)",
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
    // Your async application...
}
```

```bash
# In another terminal
tokio-console
# Shows real-time task metrics, waker counts, poll times
```

---

## 11. Profiling Checklist

```
Before Optimizing:
□ Benchmarks exist for the critical path
□ Performance requirements are defined (latency, throughput)
□ Profiling done with release build (debug builds are 10-100x slower)

Common Bottlenecks:
□ Unnecessary allocations (String, Vec, Box)
□ Excessive cloning (clone() where borrow works)
□ Cache-unfriendly data layout
□ Blocking in async code
□ Lock contention
□ Serialization/deserialization
□ I/O without buffering

Optimization Techniques:
□ Pre-allocate collections (Vec::with_capacity)
□ Use references/Cow instead of cloning
□ Batch operations (reduce syscalls, reduce lock acquisitions)
□ Use appropriate data structures (HashMap vs BTreeMap vs Vec)
□ Buffer I/O (BufReader, BufWriter)
□ Pool connections and expensive resources
□ Use spawn_blocking for CPU-intensive work in async code
```

---

## 12. Exercises

1. **Benchmark comparison**: Write Criterion benchmarks comparing: `HashMap` vs `BTreeMap` vs `Vec` (linear search) for lookups with N=10, 100, 1000, 10000 elements. Graph the results and explain the crossover points.

2. **Allocation audit**: Take a Rust program that processes a large text file (e.g., word frequency counter) and profile its allocations with DHAT. Reduce allocations by 50%+ by using `Cow`, pre-allocation, and string interning.

3. **SoA transform**: Take a particle simulation using Array-of-Structs layout. Convert it to Struct-of-Arrays and benchmark the improvement. Measure cache miss rates with `perf stat`.

4. **Flame graph analysis**: Build a program that parses a large JSON file, transforms the data, and writes CSV output. Generate a flame graph, identify the hottest function, and optimize it. Document the before/after performance.

5. **SIMD dot product**: Implement a dot product function three ways: (1) naive loop, (2) iterator chain, (3) explicit SIMD (use the `packed_simd2` or `std::simd` crate). Benchmark all three and compare with the compiler's auto-vectorized output.

---

## References

- [Criterion documentation](https://bheisler.github.io/criterion.rs/book/)
- [The Rust Performance Book](https://nnethercote.github.io/perf-book/)
- [cargo-flamegraph](https://github.com/flamegraph-rs/flamegraph)
- [DHAT documentation](https://docs.rs/dhat/latest/dhat/)
- [Tokio Console](https://github.com/tokio-rs/console)
- [Data-Oriented Design (Andrew Kelley talk)](https://www.youtube.com/watch?v=yOyaJXpAYZQ)

---

**Previous**: [Advanced Error Handling](./12_Advanced_Error_Handling.md) | **Next**: [Capstone: HTTP Server](./14_Capstone_HTTP_Server.md)
