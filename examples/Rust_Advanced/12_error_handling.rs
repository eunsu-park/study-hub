// 28_error_handling.rs — Custom error types, composition, and recovery
//
// Run: rustc 28_error_handling.rs && ./28_error_handling

use std::fmt;
use std::num::ParseIntError;

// === Custom Error Type ===

#[derive(Debug)]
enum AppError {
    Io(std::io::Error),
    Parse(ParseIntError),
    Config { key: String, message: String },
    NotFound(String),
}

impl fmt::Display for AppError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AppError::Io(e) => write!(f, "I/O error: {e}"),
            AppError::Parse(e) => write!(f, "Parse error: {e}"),
            AppError::Config { key, message } => write!(f, "Config '{key}': {message}"),
            AppError::NotFound(item) => write!(f, "Not found: {item}"),
        }
    }
}

impl std::error::Error for AppError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            AppError::Io(e) => Some(e),
            AppError::Parse(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for AppError {
    fn from(e: std::io::Error) -> Self { AppError::Io(e) }
}

impl From<ParseIntError> for AppError {
    fn from(e: ParseIntError) -> Self { AppError::Parse(e) }
}

// === Error with Context ===

struct ContextError {
    message: String,
    contexts: Vec<String>,
}

impl ContextError {
    fn new(msg: impl Into<String>) -> Self {
        ContextError { message: msg.into(), contexts: Vec::new() }
    }

    fn context(mut self, ctx: impl Into<String>) -> Self {
        self.contexts.push(ctx.into());
        self
    }
}

impl fmt::Display for ContextError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Error: {}", self.message)?;
        for (i, ctx) in self.contexts.iter().enumerate() {
            writeln!(f, "  {}: {ctx}", i + 1)?;
        }
        Ok(())
    }
}

// === Retry Pattern ===

fn retry<F, T, E: fmt::Display>(max: u32, mut f: F) -> Result<T, E>
where
    F: FnMut(u32) -> Result<T, E>,
{
    let mut last_err = None;
    for attempt in 1..=max {
        match f(attempt) {
            Ok(val) => return Ok(val),
            Err(e) => {
                println!("    Attempt {attempt}/{max}: {e}");
                last_err = Some(e);
            }
        }
    }
    Err(last_err.unwrap())
}

// === Fallback Chain ===

fn get_config_value(key: &str) -> String {
    // Try environment variable first
    if let Ok(val) = std::env::var(key) {
        return val;
    }
    // Then try a "config file" (simulated)
    if key == "PORT" {
        return "8080".to_string();
    }
    // Default
    "unknown".to_string()
}

fn main() {
    println!("=== Custom Error Type ===");
    let errors: Vec<AppError> = vec![
        AppError::NotFound("user:42".into()),
        AppError::Config {
            key: "database_url".into(),
            message: "missing required value".into(),
        },
    ];

    for err in &errors {
        println!("  {err}");
        if let Some(source) = err.source() {
            println!("    Caused by: {source}");
        }
    }

    println!("\n=== Error Context ===");
    let err = ContextError::new("connection refused")
        .context("while connecting to database")
        .context("in user service initialization")
        .context("during application startup");
    print!("{err}");

    println!("=== Retry Pattern ===");
    let result = retry(3, |attempt| {
        if attempt < 3 {
            Err(format!("service unavailable"))
        } else {
            Ok("connected!")
        }
    });
    println!("  Result: {:?}", result);

    println!("\n=== Fallback Chain ===");
    println!("  PORT = {}", get_config_value("PORT"));
    println!("  UNKNOWN = {}", get_config_value("UNKNOWN_KEY"));

    println!("\n=== Pattern Matching Errors ===");
    fn process(input: &str) -> Result<i32, AppError> {
        let value: i32 = input.parse()?;  // ParseIntError -> AppError
        if value < 0 {
            return Err(AppError::Config {
                key: "input".into(),
                message: "must be non-negative".into(),
            });
        }
        Ok(value * 2)
    }

    match process("42") {
        Ok(v) => println!("  process(\"42\") = {v}"),
        Err(e) => println!("  Error: {e}"),
    }
    match process("abc") {
        Ok(v) => println!("  process(\"abc\") = {v}"),
        Err(e) => println!("  Error: {e}"),
    }
    match process("-5") {
        Ok(v) => println!("  process(\"-5\") = {v}"),
        Err(e) => println!("  Error: {e}"),
    }
}

use std::error::Error;
