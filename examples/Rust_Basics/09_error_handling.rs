// 09_error_handling.rs — Result, ?, and custom error types
//
// Run: rustc 09_error_handling.rs && ./09_error_handling

use std::fmt;
use std::fs;
use std::io;
use std::num::ParseIntError;

// Custom error type
#[derive(Debug)]
enum AppError {
    Io(io::Error),
    Parse(ParseIntError),
    Validation(String),
}

impl fmt::Display for AppError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AppError::Io(e) => write!(f, "I/O error: {e}"),
            AppError::Parse(e) => write!(f, "Parse error: {e}"),
            AppError::Validation(msg) => write!(f, "Validation error: {msg}"),
        }
    }
}

// From conversions enable the ? operator to convert errors automatically
impl From<io::Error> for AppError {
    fn from(e: io::Error) -> Self {
        AppError::Io(e)
    }
}

impl From<ParseIntError> for AppError {
    fn from(e: ParseIntError) -> Self {
        AppError::Parse(e)
    }
}

// Function using ? for error propagation
fn read_age_from_file(path: &str) -> Result<u32, AppError> {
    let contents = fs::read_to_string(path)?; // io::Error → AppError via From
    let age: u32 = contents.trim().parse()?; // ParseIntError → AppError via From

    if age > 150 {
        return Err(AppError::Validation("Age cannot exceed 150".to_string()));
    }

    Ok(age)
}

// Chaining Results with and_then
fn parse_and_validate(input: &str) -> Result<i32, String> {
    input
        .parse::<i32>()
        .map_err(|e| format!("Parse failed: {e}"))
        .and_then(|n| {
            if n > 0 {
                Ok(n)
            } else {
                Err("Number must be positive".to_string())
            }
        })
}

fn main() {
    println!("=== Basic Result ===");
    basic_result();

    println!("\n=== The ? Operator ===");
    match read_age_from_file("/nonexistent") {
        Ok(age) => println!("Age: {age}"),
        Err(e) => println!("Error: {e}"),
    }

    println!("\n=== Result Chaining ===");
    for input in ["42", "-5", "abc"] {
        match parse_and_validate(input) {
            Ok(n) => println!("\"{input}\" → Ok({n})"),
            Err(e) => println!("\"{input}\" → Err({e})"),
        }
    }

    println!("\n=== Collecting Results ===");
    collecting_results();

    println!("\n=== unwrap Variants ===");
    unwrap_variants();
}

fn basic_result() {
    // parse() returns Result<T, E>
    let good: Result<i32, _> = "42".parse();
    let bad: Result<i32, _> = "xyz".parse();

    // Pattern matching
    match good {
        Ok(n) => println!("Parsed: {n}"),
        Err(e) => println!("Error: {e}"),
    }

    // map and unwrap_or
    let doubled = good.map(|n| n * 2).unwrap_or(0);
    println!("Doubled: {doubled}");

    let fallback = bad.unwrap_or_else(|_| {
        println!("Parse failed, using default");
        0
    });
    println!("Fallback: {fallback}");
}

fn collecting_results() {
    let inputs = vec!["1", "2", "three", "4"];

    // Collect into Result<Vec<i32>, _> — fails on first error
    let result: Result<Vec<i32>, _> = inputs.iter().map(|s| s.parse::<i32>()).collect();
    println!("Collect all: {result:?}");

    // Filter out errors, keep successes
    let successes: Vec<i32> = inputs
        .iter()
        .filter_map(|s| s.parse::<i32>().ok())
        .collect();
    println!("Successes only: {successes:?}");

    // Partition into successes and failures
    let (oks, errs): (Vec<_>, Vec<_>) = inputs
        .iter()
        .map(|s| s.parse::<i32>())
        .partition(Result::is_ok);
    let oks: Vec<i32> = oks.into_iter().map(Result::unwrap).collect();
    let errs: Vec<_> = errs.into_iter().map(Result::unwrap_err).collect();
    println!("Oks: {oks:?}, Errors: {errs:?}");
}

fn unwrap_variants() {
    let x: Result<i32, &str> = Ok(42);
    let y: Result<i32, &str> = Err("oops");

    // unwrap — panics on Err (use in tests/prototyping only)
    println!("unwrap: {}", x.unwrap());

    // expect — panics with a custom message
    println!("expect: {}", x.expect("should have a value"));

    // unwrap_or — provide a default
    println!("unwrap_or: {}", y.unwrap_or(0));

    // unwrap_or_default — uses Default trait
    println!("unwrap_or_default: {}", y.unwrap_or_default());

    // is_ok / is_err
    println!("x.is_ok()={}, y.is_err()={}", x.is_ok(), y.is_err());
}
