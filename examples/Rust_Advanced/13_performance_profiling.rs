// 13_performance_profiling.rs — Benchmarking, optimization, and data layout
//
// Run: rustc -O 13_performance_profiling.rs && ./13_performance_profiling

use std::time::Instant;

fn main() {
    println!("=== Manual Benchmarking ===");
    benchmark_demo();

    println!("\n=== Data Layout and Size ===");
    data_layout();

    println!("\n=== Iterator vs Loop ===");
    iterator_vs_loop();

    println!("\n=== Allocation Strategies ===");
    allocation_demo();

    println!("\n=== Cache-friendly Access ===");
    cache_demo();
}

// --- Simple benchmark harness ---

fn bench<F: FnMut()>(name: &str, iterations: u32, mut f: F) {
    // Warm-up
    for _ in 0..iterations / 10 {
        f();
    }

    let start = Instant::now();
    for _ in 0..iterations {
        f();
    }
    let elapsed = start.elapsed();
    let per_iter = elapsed / iterations;
    println!("  {name}: {per_iter:?}/iter ({iterations} iterations, {elapsed:?} total)");
}

fn benchmark_demo() {
    let data: Vec<i64> = (0..10_000).collect();

    bench("sum_iter", 1000, || {
        let _: i64 = data.iter().sum();
    });

    bench("sum_loop", 1000, || {
        let mut sum: i64 = 0;
        for &x in &data {
            sum += x;
        }
        std::hint::black_box(sum);
    });

    bench("sum_fold", 1000, || {
        let _: i64 = data.iter().fold(0, |acc, &x| acc + x);
    });
}

// --- Data layout ---

fn data_layout() {
    // Size and alignment
    println!("  Type sizes:");
    println!("    bool:   {} byte", std::mem::size_of::<bool>());
    println!("    u8:     {} byte", std::mem::size_of::<u8>());
    println!("    u32:    {} bytes", std::mem::size_of::<u32>());
    println!("    u64:    {} bytes", std::mem::size_of::<u64>());
    println!("    f64:    {} bytes", std::mem::size_of::<f64>());
    println!("    usize:  {} bytes", std::mem::size_of::<usize>());
    println!("    &str:   {} bytes (ptr + len)", std::mem::size_of::<&str>());
    println!("    String: {} bytes (ptr + len + cap)", std::mem::size_of::<String>());

    // Struct padding
    #[repr(C)]
    struct Padded { a: u8, b: u64, c: u8 } // 8 + 8 + 8 = 24 bytes with C layout

    struct Optimized { a: u8, c: u8, b: u64 } // Rust may reorder to 16 bytes

    println!("\n  Struct layout:");
    println!("    Padded (repr(C)): {} bytes", std::mem::size_of::<Padded>());
    println!("    Optimized (Rust): {} bytes", std::mem::size_of::<Optimized>());

    // Enum size (tag + largest variant)
    enum Small { A, B, C }
    enum WithData { None, Int(u64), Pair(u64, u64) }
    println!("    Small enum: {} byte", std::mem::size_of::<Small>());
    println!("    WithData enum: {} bytes", std::mem::size_of::<WithData>());

    // Option<NonNull> is zero-cost (niche optimization)
    println!("    Option<Box<u8>>: {} bytes (niche!)", std::mem::size_of::<Option<Box<u8>>>());
    println!("    Box<u8>:         {} bytes", std::mem::size_of::<Box<u8>>());
}

// --- Iterator chains vs manual loops ---

fn iterator_vs_loop() {
    let data: Vec<f64> = (0..100_000).map(|i| i as f64).collect();

    // Iterator chain — compiler can auto-vectorize
    bench("iter_chain", 100, || {
        let _: f64 = data.iter()
            .filter(|&&x| x > 50_000.0)
            .map(|x| x * 2.0)
            .sum();
    });

    // Manual loop — equivalent but more verbose
    bench("manual_loop", 100, || {
        let mut sum: f64 = 0.0;
        for &x in &data {
            if x > 50_000.0 {
                sum += x * 2.0;
            }
        }
        std::hint::black_box(sum);
    });
}

// --- Allocation strategies ---

fn allocation_demo() {
    let n = 100_000;

    // Growing Vec (many reallocations)
    bench("vec_grow", 100, || {
        let mut v = Vec::new();
        for i in 0..n {
            v.push(i);
        }
        std::hint::black_box(&v);
    });

    // Pre-allocated Vec (one allocation)
    bench("vec_preallocated", 100, || {
        let mut v = Vec::with_capacity(n);
        for i in 0..n {
            v.push(i);
        }
        std::hint::black_box(&v);
    });

    // collect (pre-allocates via size_hint)
    bench("vec_collect", 100, || {
        let v: Vec<usize> = (0..n).collect();
        std::hint::black_box(&v);
    });

    // String concatenation strategies
    bench("string_push", 1000, || {
        let mut s = String::with_capacity(1000);
        for i in 0..100 {
            s.push_str(&i.to_string());
            s.push(' ');
        }
        std::hint::black_box(&s);
    });
}

// --- Cache-friendly access ---

fn cache_demo() {
    const SIZE: usize = 1024;

    // Row-major (cache-friendly)
    let mut matrix = vec![vec![0u64; SIZE]; SIZE];

    bench("row_major", 10, || {
        for i in 0..SIZE {
            for j in 0..SIZE {
                matrix[i][j] = (i + j) as u64;
            }
        }
    });

    // Column-major (cache-unfriendly)
    bench("col_major", 10, || {
        for j in 0..SIZE {
            for i in 0..SIZE {
                matrix[i][j] = (i + j) as u64;
            }
        }
    });

    std::hint::black_box(&matrix);
}
