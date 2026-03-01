// 12_iterators.rs — Closures and iterator adaptors
//
// Run: rustc 12_iterators.rs && ./12_iterators

fn main() {
    println!("=== Closures ===");
    closure_demo();

    println!("\n=== Fn Traits ===");
    fn_traits_demo();

    println!("\n=== Iterator Adaptors ===");
    adaptor_demo();

    println!("\n=== Custom Iterator ===");
    custom_iterator_demo();
}

fn closure_demo() {
    // Basic closure syntax
    let add = |a: i32, b: i32| a + b;
    println!("add(3, 4) = {}", add(3, 4));

    // Closures capture variables from their environment
    let multiplier = 3;
    let multiply = |x| x * multiplier; // Captures multiplier by reference
    println!("multiply(5) = {}", multiply(5));

    // Multi-line closure
    let classify = |n: i32| -> &str {
        if n < 0 {
            "negative"
        } else if n == 0 {
            "zero"
        } else {
            "positive"
        }
    };
    println!("classify(-3) = {}", classify(-3));

    // move closure — takes ownership of captured variables
    let name = String::from("Rust");
    let greet = move || println!("Hello, {name}!"); // name is moved into closure
    greet();
    // println!("{name}"); // ERROR: name was moved
}

fn fn_traits_demo() {
    // Fn — borrows captured values (can be called multiple times)
    fn apply_fn(f: &dyn Fn(i32) -> i32, x: i32) -> i32 {
        f(x)
    }

    let offset = 10;
    let add_offset = |x| x + offset; // Captures offset by &ref → Fn
    println!("Fn: {}", apply_fn(&add_offset, 5));
    println!("Fn: {}", apply_fn(&add_offset, 10)); // Can call again

    // FnMut — mutably borrows captured values
    let mut count = 0;
    let mut counter = || {
        count += 1; // Mutably captures count → FnMut
        count
    };
    println!("FnMut: {}", counter());
    println!("FnMut: {}", counter());

    // FnOnce — consumes captured values (can only be called once)
    let name = String::from("Alice");
    let consume = move || {
        println!("FnOnce consuming: {name}");
        name // Returns the moved value
    };
    let returned = consume();
    // consume(); // ERROR: already consumed
    println!("Returned: {returned}");
}

fn adaptor_demo() {
    let numbers = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];

    // map — transform each element
    let squares: Vec<i32> = numbers.iter().map(|&n| n * n).collect();
    println!("Squares: {squares:?}");

    // filter — keep elements matching predicate
    let evens: Vec<&i32> = numbers.iter().filter(|&&n| n % 2 == 0).collect();
    println!("Evens: {evens:?}");

    // filter_map — filter and transform in one step
    let parsed: Vec<i32> = ["1", "two", "3", "four", "5"]
        .iter()
        .filter_map(|s| s.parse().ok())
        .collect();
    println!("Parsed: {parsed:?}");

    // enumerate — add index
    for (i, n) in numbers.iter().enumerate().take(3) {
        println!("  [{i}] = {n}");
    }

    // zip — pair two iterators
    let names = vec!["Alice", "Bob", "Charlie"];
    let scores = vec![95, 87, 92];
    let roster: Vec<_> = names.iter().zip(scores.iter()).collect();
    println!("Roster: {roster:?}");

    // chain — concatenate iterators
    let a = vec![1, 2, 3];
    let b = vec![4, 5, 6];
    let all: Vec<_> = a.iter().chain(b.iter()).collect();
    println!("Chained: {all:?}");

    // fold — accumulate into a single value
    let sum = numbers.iter().fold(0, |acc, &n| acc + n);
    let product = numbers.iter().fold(1, |acc, &n| acc * n);
    println!("Sum: {sum}, Product: {product}");

    // Chaining multiple adaptors (lazy — only computed when consumed)
    let result: i32 = (1..=100)
        .filter(|n| n % 3 == 0) // Divisible by 3
        .map(|n| n * n) // Square
        .take(5) // First 5
        .sum(); // Consume
    println!("Sum of first 5 squares divisible by 3: {result}");
}

// Custom iterator
struct Fibonacci {
    a: u64,
    b: u64,
}

impl Fibonacci {
    fn new() -> Self {
        Self { a: 0, b: 1 }
    }
}

impl Iterator for Fibonacci {
    type Item = u64;

    fn next(&mut self) -> Option<Self::Item> {
        let result = self.a;
        let next = self.a.checked_add(self.b)?; // Return None on overflow
        self.a = self.b;
        self.b = next;
        Some(result)
    }
}

fn custom_iterator_demo() {
    // Take first 15 Fibonacci numbers
    let fibs: Vec<u64> = Fibonacci::new().take(15).collect();
    println!("Fibonacci: {fibs:?}");

    // Use iterator adaptors on custom iterator
    let even_fibs: Vec<u64> = Fibonacci::new()
        .filter(|&n| n % 2 == 0)
        .take(8)
        .collect();
    println!("Even Fibonacci: {even_fibs:?}");

    // Sum Fibonacci numbers below 1000
    let sum: u64 = Fibonacci::new().take_while(|&n| n < 1000).sum();
    println!("Sum of Fibonacci < 1000: {sum}");
}
