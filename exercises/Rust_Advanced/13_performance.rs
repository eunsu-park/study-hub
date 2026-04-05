// Exercise: Performance and Profiling
// Practice optimization techniques and data-oriented design.
//
// Run: rustc -O 29_performance.rs && ./29_performance

use std::collections::HashMap;
use std::time::Instant;

// Exercise 1: Reduce allocations
// Optimize word_count to minimize allocations.
fn word_count_naive(text: &str) -> HashMap<String, usize> {
    let mut counts = HashMap::new();
    for word in text.split_whitespace() {
        let word = word.to_lowercase();  // Allocates every time!
        *counts.entry(word).or_insert(0) += 1;
    }
    counts
}

fn word_count_optimized(text: &str) -> HashMap<&str, usize> {
    // TODO: Use &str keys to avoid allocation
    let mut counts = HashMap::new();
    for word in text.split_whitespace() {
        *counts.entry(word).or_insert(0) += 1;
    }
    counts
}

// Exercise 2: SoA vs AoS
// Compare Array-of-Structs vs Struct-of-Arrays performance.

#[derive(Clone)]
struct ParticleAoS {
    x: f64, y: f64, z: f64,
    mass: f64,
}

struct ParticlesSoA {
    x: Vec<f64>, y: Vec<f64>, z: Vec<f64>,
    mass: Vec<f64>,
}

fn sum_mass_aos(particles: &[ParticleAoS]) -> f64 {
    particles.iter().map(|p| p.mass).sum()
}

fn sum_mass_soa(particles: &ParticlesSoA) -> f64 {
    particles.mass.iter().sum()
}

// Exercise 3: Pre-allocation
fn build_string_naive(n: usize) -> String {
    let mut s = String::new();
    for i in 0..n {
        s += &i.to_string();
        s += " ";
    }
    s
}

fn build_string_optimized(n: usize) -> String {
    // Estimate capacity: avg 3 chars per number + space
    let mut s = String::with_capacity(n * 4);
    for i in 0..n {
        use std::fmt::Write;
        write!(s, "{i} ").unwrap();
    }
    s
}

fn benchmark<F: Fn() -> T, T>(name: &str, iterations: u32, f: F) -> T {
    let start = Instant::now();
    let mut result = f();
    for _ in 1..iterations {
        result = f();
    }
    let elapsed = start.elapsed();
    println!("  {name}: {:?} ({iterations} iterations)", elapsed);
    result
}

fn main() {
    let text = "the quick brown fox jumps over the lazy dog the fox the dog";

    // Exercise 1
    println!("=== Word Count ===");
    let naive = word_count_naive(text);
    let optimized = word_count_optimized(text);
    println!("  Naive result: {naive:?}");
    println!("  Optimized result: {optimized:?}");
    // Note: optimized doesn't lowercase, but avoids allocation

    // Exercise 2
    println!("\n=== AoS vs SoA ===");
    let n = 1_000_000;

    let aos: Vec<ParticleAoS> = (0..n).map(|i| ParticleAoS {
        x: i as f64, y: 0.0, z: 0.0, mass: 1.0,
    }).collect();

    let soa = ParticlesSoA {
        x: (0..n).map(|i| i as f64).collect(),
        y: vec![0.0; n],
        z: vec![0.0; n],
        mass: vec![1.0; n],
    };

    let sum1 = benchmark("AoS sum_mass", 100, || sum_mass_aos(&aos));
    let sum2 = benchmark("SoA sum_mass", 100, || sum_mass_soa(&soa));
    assert_eq!(sum1, sum2);
    println!("  Both sums: {sum1}");

    // Exercise 3
    println!("\n=== String Building ===");
    let n = 10_000;
    benchmark("naive", 100, || build_string_naive(n));
    benchmark("optimized", 100, || build_string_optimized(n));

    // Exercise 4: Cache-friendly access
    println!("\n=== Sequential vs Random Access ===");
    let data: Vec<u64> = (0..1_000_000).collect();
    let indices: Vec<usize> = {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        (0..1_000_000).map(|i| {
            let mut h = DefaultHasher::new();
            i.hash(&mut h);
            (h.finish() as usize) % data.len()
        }).collect()
    };

    benchmark("sequential", 10, || -> u64 { data.iter().sum() });
    benchmark("random", 10, || -> u64 { indices.iter().map(|&i| data[i]).sum() });

    println!("\nAll exercises complete!");
}
