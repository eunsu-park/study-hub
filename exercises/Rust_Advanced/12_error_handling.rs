// Exercise: Advanced Error Handling
// Practice designing error types and error handling patterns.
//
// Run: rustc 28_error_handling.rs && ./28_error_handling

use std::fmt;
use std::num::ParseIntError;

// Exercise 1: Design a custom error type hierarchy
#[derive(Debug)]
enum FileProcessError {
    Io(std::io::Error),
    Parse { line: usize, column: usize, message: String },
    Validation(String),
    Encoding(String),
}

impl fmt::Display for FileProcessError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => write!(f, "I/O error: {e}"),
            Self::Parse { line, column, message } =>
                write!(f, "Parse error at {line}:{column}: {message}"),
            Self::Validation(msg) => write!(f, "Validation: {msg}"),
            Self::Encoding(msg) => write!(f, "Encoding: {msg}"),
        }
    }
}

impl std::error::Error for FileProcessError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for FileProcessError {
    fn from(e: std::io::Error) -> Self { Self::Io(e) }
}

// Exercise 2: Retry function
fn retry<F, T, E: fmt::Display>(max: u32, mut f: F) -> Result<T, E>
where
    F: FnMut() -> Result<T, E>,
{
    let mut last_err = None;
    for attempt in 1..=max {
        match f() {
            Ok(val) => return Ok(val),
            Err(e) => {
                println!("  Attempt {attempt}/{max} failed: {e}");
                last_err = Some(e);
            }
        }
    }
    Err(last_err.unwrap())
}

// Exercise 3: Error context builder
#[derive(Debug)]
struct ContextError {
    message: String,
    contexts: Vec<String>,
    source: Option<Box<dyn std::error::Error>>,
}

impl ContextError {
    fn new(msg: impl Into<String>) -> Self {
        ContextError {
            message: msg.into(),
            contexts: Vec::new(),
            source: None,
        }
    }

    fn with_context(mut self, ctx: impl Into<String>) -> Self {
        self.contexts.push(ctx.into());
        self
    }

    fn with_source(mut self, err: impl std::error::Error + 'static) -> Self {
        self.source = Some(Box::new(err));
        self
    }
}

impl fmt::Display for ContextError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.message)?;
        for ctx in &self.contexts {
            write!(f, "\n  Context: {ctx}")?;
        }
        if let Some(ref src) = self.source {
            write!(f, "\n  Caused by: {src}")?;
        }
        Ok(())
    }
}

// Exercise 4: Fallback chain
fn get_value(primary: Result<i32, &str>, secondary: Result<i32, &str>, default: i32) -> i32 {
    primary
        .or(secondary)
        .unwrap_or(default)
}

fn main() {
    // Test Exercise 1
    let err = FileProcessError::Parse {
        line: 42, column: 10,
        message: "unexpected token".into(),
    };
    println!("Error: {err}");
    assert!(err.to_string().contains("42:10"));

    let io_err = FileProcessError::from(
        std::io::Error::new(std::io::ErrorKind::NotFound, "file missing")
    );
    println!("IO Error: {io_err}");
    assert!(io_err.source().is_some());

    // Test Exercise 2
    let mut counter = 0;
    let result = retry(3, || {
        counter += 1;
        if counter < 3 {
            Err(format!("not ready (attempt {counter})"))
        } else {
            Ok(42)
        }
    });
    assert_eq!(result.unwrap(), 42);
    println!("Retry succeeded after {counter} attempts");

    // Test Exercise 3
    let err = ContextError::new("database query failed")
        .with_context("while loading user profile")
        .with_context("in request handler /api/users/42");
    println!("\n{err}");

    // Test Exercise 4
    assert_eq!(get_value(Ok(1), Ok(2), 0), 1);
    assert_eq!(get_value(Err("fail"), Ok(2), 0), 2);
    assert_eq!(get_value(Err("fail"), Err("fail"), 99), 99);
    println!("\nFallback chain works correctly");

    println!("\nAll exercises passed!");
}
