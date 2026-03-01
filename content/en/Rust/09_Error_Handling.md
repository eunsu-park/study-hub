# 09. Error Handling

**Previous**: [Collections](./08_Collections.md) | **Next**: [Traits and Generics](./10_Traits_and_Generics.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Distinguish between recoverable (`Result`) and unrecoverable (`panic!`) errors and choose the appropriate one
2. Propagate errors concisely using the `?` operator and explain how it desugars
3. Define custom error types that implement `std::error::Error`
4. Use `thiserror` and `anyhow` crates for ergonomic error handling in libraries and applications
5. Implement error conversion using the `From` trait

---

Error handling is where Rust diverges most sharply from languages you may already know. There are no exceptions, no try/catch blocks, and no null pointer surprises. Instead, Rust encodes the possibility of failure directly into the type system — if a function can fail, its return type says so. This makes errors impossible to accidentally ignore and forces you to think about failure paths up front.

Think of it like a restaurant kitchen: rather than letting a fire spread and hoping someone notices (exceptions), Rust hands you a fire extinguisher at every station and requires you to acknowledge you have it (Result).

## Table of Contents
1. [Unrecoverable Errors: panic!](#1-unrecoverable-errors-panic)
2. [Result<T, E> — Recoverable Errors](#2-resultt-e--recoverable-errors)
3. [The ? Operator](#3-the--operator)
4. [Custom Error Types](#4-custom-error-types)
5. [thiserror — Deriving Error Types](#5-thiserror--deriving-error-types)
6. [anyhow — Application-Level Errors](#6-anyhow--application-level-errors)
7. [When to panic vs When to Result](#7-when-to-panic-vs-when-to-result)
8. [Error Conversion with From](#8-error-conversion-with-from)
9. [Practice Problems](#9-practice-problems)

---

## 1. Unrecoverable Errors: panic!

When something goes so wrong that the program cannot reasonably continue, you `panic!`. This immediately stops execution, unwinds the stack (by default), and prints an error message.

```rust
fn main() {
    // Explicit panic
    // panic!("something went terribly wrong");

    // Implicit panics from the standard library
    let v = vec![1, 2, 3];
    // v[99]; // panics: index out of bounds

    // Panics are for bugs, not for expected failures
    // Examples: violated invariants, unreachable code, failed assertions
    let x = 5;
    assert!(x > 0, "x must be positive, got {}", x);
    assert_eq!(2 + 2, 4);
}
```

### 1.1 Unwinding vs Abort

When a panic occurs, Rust has two strategies:

```
Unwinding (default):                  Abort:
┌──────────────────────┐              ┌──────────────────────┐
│ Walks back up the    │              │ Immediately kills    │
│ call stack, running  │              │ the process.         │
│ Drop for each frame. │              │ No cleanup, no Drop. │
│                      │              │                      │
│ Pro: memory cleaned  │              │ Pro: smaller binary  │
│ Con: larger binary   │              │ Con: no cleanup      │
│      slower panic    │              │      OS reclaims mem │
└──────────────────────┘              └──────────────────────┘
```

To use abort mode, add this to `Cargo.toml`:
```toml
[profile.release]
panic = "abort"
```

### 1.2 Backtrace

Set `RUST_BACKTRACE=1` to see where a panic originated:

```bash
$ RUST_BACKTRACE=1 cargo run
thread 'main' panicked at 'index out of bounds: the len is 3 but the index is 99'
stack backtrace:
   0: std::panicking::begin_panic_handler
   ...
   8: playground::main
             at ./src/main.rs:3:5
```

---

## 2. Result<T, E> — Recoverable Errors

For errors you **expect** might happen (file not found, invalid input, network timeout), Rust uses `Result<T, E>`:

```rust
enum Result<T, E> {
    Ok(T),   // Success, contains the value
    Err(E),  // Failure, contains the error
}
```

```rust
use std::fs;

fn main() {
    // fs::read_to_string returns Result<String, io::Error>
    let result = fs::read_to_string("hello.txt");

    match result {
        Ok(contents) => println!("File contents:\n{}", contents),
        Err(error) => println!("Failed to read file: {}", error),
    }
}
```

### 2.1 Working with Result

```rust
use std::num::ParseIntError;

fn parse_age(input: &str) -> Result<u32, ParseIntError> {
    input.trim().parse::<u32>()
}

fn main() {
    // Pattern matching — the most explicit approach
    match parse_age("25") {
        Ok(age) => println!("Age: {}", age),
        Err(e) => println!("Parse error: {}", e),
    }

    // unwrap() — panics on Err (use only in prototypes or tests)
    let age = parse_age("30").unwrap();
    println!("age = {}", age);

    // expect() — panics with a custom message (better than unwrap)
    let age = parse_age("30").expect("failed to parse age");
    println!("age = {}", age);

    // unwrap_or() — provide a fallback value
    let age = parse_age("not_a_number").unwrap_or(0);
    println!("age with fallback = {}", age);

    // unwrap_or_else() — compute fallback lazily
    let age = parse_age("not_a_number").unwrap_or_else(|_| {
        println!("Using default age");
        18
    });
    println!("age = {}", age);
}
```

### 2.2 Mapping and Chaining

```rust
fn double_parse(input: &str) -> Result<i32, std::num::ParseIntError> {
    input
        .trim()
        .parse::<i32>()
        .map(|n| n * 2) // transform the Ok value
}

fn main() {
    println!("{:?}", double_parse("21")); // Ok(42)
    println!("{:?}", double_parse("xx")); // Err(ParseIntError)

    // map_err transforms the Err variant
    let result: Result<i32, String> = "abc"
        .parse::<i32>()
        .map_err(|e| format!("Custom error: {}", e));
    println!("{:?}", result);

    // and_then chains operations that each return Result
    let result = "42"
        .parse::<i32>()
        .and_then(|n| {
            if n > 0 {
                Ok(n)
            } else {
                Err("must be positive".parse::<i32>().unwrap_err())
            }
        });
    println!("{:?}", result);
}
```

---

## 3. The ? Operator

The `?` operator is Rust's secret weapon for clean error handling. It replaces verbose `match` boilerplate with a single character:

```rust
use std::fs;
use std::io;

// WITHOUT ? — verbose and deeply nested
fn read_username_verbose() -> Result<String, io::Error> {
    let result = fs::read_to_string("username.txt");
    match result {
        Ok(contents) => Ok(contents.trim().to_string()),
        Err(e) => Err(e),
    }
}

// WITH ? — concise and readable
fn read_username() -> Result<String, io::Error> {
    let contents = fs::read_to_string("username.txt")?;
    // If read_to_string returns Err, the function returns that Err immediately
    // If it returns Ok, the inner value is extracted and assigned to contents
    Ok(contents.trim().to_string())
}

fn main() {
    match read_username() {
        Ok(name) => println!("Username: {}", name),
        Err(e) => println!("Error: {}", e),
    }
}
```

### 3.1 How ? Desugars

The `?` operator is roughly equivalent to:

```
expression?

// desugars to approximately:

match expression {
    Ok(val)  => val,
    Err(err) => return Err(From::from(err)),
                       ^^^^^^^^^^^^^^^^^^^^
                       Note: calls From::from() to convert
                       the error type if needed!
}
```

This means `?` does **two** things:
1. Early-returns on `Err`
2. Converts the error type using `From`

### 3.2 Chaining Multiple ? Calls

```rust
use std::fs::File;
use std::io::{self, Read};

fn read_first_line(path: &str) -> Result<String, io::Error> {
    let mut file = File::open(path)?;       // might fail
    let mut contents = String::new();
    file.read_to_string(&mut contents)?;    // might fail
    let first_line = contents
        .lines()
        .next()
        .unwrap_or("")
        .to_string();
    Ok(first_line)
}

// Even more concise with method chaining
fn read_first_line_short(path: &str) -> Result<String, io::Error> {
    let contents = std::fs::read_to_string(path)?;
    Ok(contents.lines().next().unwrap_or("").to_string())
}

fn main() {
    match read_first_line("/etc/hostname") {
        Ok(line) => println!("First line: {}", line),
        Err(e) => println!("Error: {}", e),
    }
}
```

### 3.3 Using ? in main()

```rust
use std::fs;
use std::io;

// main() can return Result — errors are printed automatically
fn main() -> Result<(), io::Error> {
    let contents = fs::read_to_string("config.txt")?;
    println!("Config: {}", contents);
    Ok(())
}
```

---

## 4. Custom Error Types

Real applications have multiple failure modes. A custom error type gives callers a structured way to handle each one:

```rust
use std::fmt;
use std::num::ParseIntError;
use std::io;

#[derive(Debug)]
enum AppError {
    Io(io::Error),
    Parse(ParseIntError),
    Validation(String),
}

// Display: human-readable error messages
impl fmt::Display for AppError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AppError::Io(e) => write!(f, "I/O error: {}", e),
            AppError::Parse(e) => write!(f, "Parse error: {}", e),
            AppError::Validation(msg) => write!(f, "Validation error: {}", msg),
        }
    }
}

// Error trait: connects to Rust's error ecosystem
impl std::error::Error for AppError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        // source() returns the underlying cause, enabling error chains
        match self {
            AppError::Io(e) => Some(e),
            AppError::Parse(e) => Some(e),
            AppError::Validation(_) => None,
        }
    }
}

// From impls enable automatic conversion with ?
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

fn load_config(path: &str) -> Result<u32, AppError> {
    let contents = std::fs::read_to_string(path)?; // io::Error → AppError via From
    let port: u32 = contents.trim().parse()?;       // ParseIntError → AppError via From
    if port < 1024 {
        return Err(AppError::Validation(
            format!("Port {} is reserved (must be >= 1024)", port),
        ));
    }
    Ok(port)
}

fn main() {
    match load_config("port.txt") {
        Ok(port) => println!("Listening on port {}", port),
        Err(AppError::Io(e)) => eprintln!("Could not read config: {}", e),
        Err(AppError::Parse(e)) => eprintln!("Invalid port number: {}", e),
        Err(AppError::Validation(msg)) => eprintln!("{}", msg),
    }
}
```

That works, but the boilerplate is tedious. Enter `thiserror`.

---

## 5. thiserror — Deriving Error Types

The `thiserror` crate generates `Display`, `Error`, and `From` implementations via derive macros. It is the standard choice for **library** error types:

```toml
# Cargo.toml
[dependencies]
thiserror = "2"
```

```rust
use thiserror::Error;

#[derive(Debug, Error)]
enum AppError {
    // #[error(...)] generates the Display impl
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),  // #[from] generates From<io::Error>

    #[error("Parse error: {0}")]
    Parse(#[from] std::num::ParseIntError),

    #[error("Validation: {0}")]
    Validation(String),
}

// That's it! ~10 lines replaces ~40 lines of manual impls.

fn load_config(path: &str) -> Result<u32, AppError> {
    let contents = std::fs::read_to_string(path)?;
    let port: u32 = contents.trim().parse()?;
    if port < 1024 {
        return Err(AppError::Validation(
            format!("Port {} is reserved", port),
        ));
    }
    Ok(port)
}

fn main() {
    if let Err(e) = load_config("port.txt") {
        eprintln!("Error: {}", e);
        // Can inspect the source chain
        if let Some(source) = std::error::Error::source(&e) {
            eprintln!("  caused by: {}", source);
        }
    }
}
```

---

## 6. anyhow — Application-Level Errors

While `thiserror` is for libraries (structured, typed errors), `anyhow` is for **applications** where you just want to propagate any error with context:

```toml
# Cargo.toml
[dependencies]
anyhow = "1"
```

```rust
use anyhow::{Context, Result};

// anyhow::Result<T> is an alias for Result<T, anyhow::Error>
// anyhow::Error can hold any error type that implements std::error::Error

fn load_config(path: &str) -> Result<u32> {
    let contents = std::fs::read_to_string(path)
        .context(format!("Failed to read config from '{}'", path))?;
        //       ^^^^^^^ adds human-readable context to the error chain

    let port: u32 = contents
        .trim()
        .parse()
        .context("Config file does not contain a valid port number")?;

    anyhow::ensure!(port >= 1024, "Port {} is reserved (must be >= 1024)", port);
    //      ^^^^^^^ like assert!, but returns Err instead of panicking

    Ok(port)
}

fn main() -> Result<()> {
    let port = load_config("port.txt")?;
    println!("Server running on port {}", port);
    Ok(())
    // If an error occurs, anyhow prints the full error chain:
    //   Error: Failed to read config from 'port.txt'
    //
    //   Caused by:
    //       No such file or directory (os error 2)
}
```

### thiserror vs anyhow — When to Use Which

```
                 thiserror                    anyhow
              ┌─────────────────┐         ┌─────────────────┐
  Use for:    │ Libraries       │         │ Applications    │
              │ Public APIs     │         │ CLI tools       │
              │                 │         │ Scripts         │
              ├─────────────────┤         ├─────────────────┤
  Errors:     │ Enum variants   │         │ Opaque, boxed   │
              │ Typed, matchable│         │ Context strings  │
              ├─────────────────┤         ├─────────────────┤
  Callers:    │ Can match on    │         │ Just print or   │
              │ specific errors │         │ propagate       │
              └─────────────────┘         └─────────────────┘

  Rule of thumb: if your code is consumed by others → thiserror
                  if your code is the final binary  → anyhow
```

---

## 7. When to panic vs When to Result

| Situation | Use | Reason |
|-----------|-----|--------|
| Array index out of bounds (bug) | `panic!` | Programming error — should not happen |
| File not found | `Result` | Expected failure the user can fix |
| Invalid user input | `Result` | Normal runtime condition |
| Violated invariant (e.g., negative array size) | `panic!` | Bug in calling code |
| Network timeout | `Result` | Expected, retryable |
| Prototype / example code | `unwrap()` | Acceptable for quick experiments |
| Tests | `unwrap()` / `?` | Test failures should panic |

**Guidelines:**
1. Use `Result` as the default for any function that can fail
2. Use `panic!` only for truly unrecoverable situations or programming bugs
3. In library code, almost never `panic!` — let the caller decide what to do
4. `unwrap()` in production code is a code smell — use `expect()` at minimum

---

## 8. Error Conversion with From

The `From` trait is the glue that makes `?` work across error types. When you write `result?` in a function returning `Result<T, MyError>`, Rust calls `MyError::from(original_error)` automatically:

```rust
use std::num::ParseIntError;

#[derive(Debug)]
struct AgeError {
    message: String,
}

// This From impl lets ? convert ParseIntError → AgeError
impl From<ParseIntError> for AgeError {
    fn from(e: ParseIntError) -> Self {
        AgeError {
            message: format!("Invalid age: {}", e),
        }
    }
}

fn parse_age(input: &str) -> Result<u32, AgeError> {
    let age: u32 = input.trim().parse()?;
    // Without the From impl above, this ? would fail to compile
    // because parse() returns ParseIntError, not AgeError

    if age > 150 {
        return Err(AgeError {
            message: format!("Age {} is unrealistic", age),
        });
    }
    Ok(age)
}

fn main() {
    match parse_age("abc") {
        Ok(age) => println!("Age: {}", age),
        Err(e) => println!("Error: {}", e.message),
    }

    match parse_age("200") {
        Ok(age) => println!("Age: {}", age),
        Err(e) => println!("Error: {}", e.message),
    }
}
```

### 8.1 The Error Conversion Flow

```
Your function returns: Result<T, YourError>

Code inside the function:
    let val = some_call()?;
              ^^^^^^^^^^^
              Returns Result<V, OtherError>

Compiler checks:
    Is there an impl From<OtherError> for YourError?
    ├── Yes → calls YourError::from(other_error), returns Err(converted)
    └── No  → compile error!

This is why thiserror's #[from] attribute is so useful:
    #[error("...")]
    Io(#[from] std::io::Error)
    // generates: impl From<io::Error> for YourError
```

---

## 9. Practice Problems

### Problem 1: Safe Division
Write a function `safe_divide(a: f64, b: f64) -> Result<f64, String>` that returns an error when dividing by zero instead of producing `inf` or `NaN`. Then write a `calculate` function that chains multiple divisions using `?`.

### Problem 2: Config Parser
Write a function `parse_config(input: &str) -> Result<Config, ConfigError>` where:
- `Config` has fields `host: String` and `port: u16`
- Input format is `host:port` (e.g., `"localhost:8080"`)
- `ConfigError` has variants: `MissingPort`, `InvalidPort(ParseIntError)`, `EmptyHost`
- Implement `Display` and `Error` for `ConfigError` manually (without `thiserror`)

### Problem 3: File Processing Pipeline
Write a function that reads a file of numbers (one per line), parses each line as `f64`, computes the average, and returns it. Use a custom error type (with `thiserror`) that can represent both I/O errors and parse errors. The error should include the line number where parsing failed.

### Problem 4: Error Context Chain
Using `anyhow`, write a function `process_user_data(path: &str) -> anyhow::Result<UserSummary>` that:
1. Reads a JSON-like file (you can simulate with a simple format)
2. Parses user name and age
3. Validates the data (age must be 0-150)
4. Each step adds `.context(...)` so the final error message shows the full chain

### Problem 5: Retry with Backoff
Write a generic retry function:
```rust
fn retry<T, E, F>(max_attempts: u32, mut operation: F) -> Result<T, E>
where F: FnMut() -> Result<T, E>
```
It should call `operation()` up to `max_attempts` times, returning the first `Ok` or the last `Err`. Test it with a function that fails randomly.

---

## References

- [The Rust Programming Language, Ch. 9: Error Handling](https://doc.rust-lang.org/book/ch09-00-error-handling.html)
- [Rust by Example: Error Handling](https://doc.rust-lang.org/rust-by-example/error.html)
- [thiserror crate documentation](https://docs.rs/thiserror)
- [anyhow crate documentation](https://docs.rs/anyhow)
- [Error Handling in a Correctness-Critical Rust Project (BurntSushi)](https://blog.burntsushi.net/rust-error-handling/)

---

**Previous**: [Collections](./08_Collections.md) | **Next**: [Traits and Generics](./10_Traits_and_Generics.md)
