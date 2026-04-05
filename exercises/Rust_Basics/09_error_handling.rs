// Exercise: Error Handling
// Practice with Result, ?, and custom error types.
//
// Run: rustc 09_error_handling.rs && ./09_error_handling

use std::fmt;
use std::num::ParseIntError;

fn main() {
    // Exercise 1: Parse and validate
    assert_eq!(parse_positive("42"), Ok(42));
    assert!(parse_positive("-5").is_err());
    assert!(parse_positive("abc").is_err());
    println!("Exercise 1 passed!");

    // Exercise 2: Chain operations with ?
    assert_eq!(parse_add("10", "20"), Ok(30));
    assert!(parse_add("abc", "20").is_err());
    println!("Exercise 2 passed!");

    // Exercise 3: Custom error type
    assert!(validate_username("alice").is_ok());
    assert!(validate_username("ab").is_err()); // Too short
    assert!(validate_username("alice bob").is_err()); // Contains space
    assert!(validate_username("").is_err()); // Empty
    println!("Exercise 3 passed!");

    // Exercise 4: Collect Results
    let inputs = vec!["1", "2", "three", "4", "5"];
    let (successes, failures) = partition_results(&inputs);
    assert_eq!(successes, vec![1, 2, 4, 5]);
    assert_eq!(failures.len(), 1);
    println!("Exercise 4 passed!");

    println!("\nAll exercises passed!");
}

fn parse_positive(s: &str) -> Result<i32, String> {
    // TODO: Parse s as i32, then validate it's positive
    // Return Err with a descriptive message for invalid input or non-positive numbers
    todo!()
}

fn parse_add(a: &str, b: &str) -> Result<i32, ParseIntError> {
    // TODO: Parse both strings as i32 and return their sum
    // Use the ? operator for error propagation
    todo!()
}

// TODO: Define a ValidationError enum with variants: Empty, TooShort(usize), ContainsSpace
#[derive(Debug)]
enum ValidationError {
    Empty,
    TooShort(usize),
    ContainsSpace,
}

impl fmt::Display for ValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // TODO: Implement Display
        todo!()
    }
}

fn validate_username(name: &str) -> Result<&str, ValidationError> {
    // TODO: Validate: non-empty, at least 3 chars, no spaces
    todo!()
}

fn partition_results(inputs: &[&str]) -> (Vec<i32>, Vec<ParseIntError>) {
    // TODO: Parse each input, partition into successes and failures
    todo!()
}
