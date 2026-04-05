# 28. Advanced Error Handling

**Previous**: [Network Programming](./11_Network_Programming.md) | **Next**: [Performance and Profiling](./13_Performance_Profiling.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Design custom error type hierarchies for libraries and applications
2. Use `thiserror` for library error types and `anyhow` for application error handling
3. Compose errors across module boundaries with conversion traits
4. Apply recovery patterns including retry, fallback, and circuit breaker
5. Provide rich error context with backtraces, source chains, and span information

---

Lesson 09 covered Rust's `Result`/`Option` basics and the `?` operator. This lesson tackles error handling at scale: designing error types for real-world applications, choosing the right crate for your context, composing errors across layers, and handling errors gracefully in production.

## Table of Contents
1. [Error Handling Philosophy](#1-error-handling-philosophy)
2. [Custom Error Types from Scratch](#2-custom-error-types-from-scratch)
3. [thiserror: Library Error Types](#3-thiserror-library-error-types)
4. [anyhow: Application Error Handling](#4-anyhow-application-error-handling)
5. [Error Composition Across Layers](#5-error-composition-across-layers)
6. [Error Context and Backtraces](#6-error-context-and-backtraces)
7. [Recovery Patterns](#7-recovery-patterns)
8. [Error Handling in Async Code](#8-error-handling-in-async-code)
9. [Logging and Reporting](#9-logging-and-reporting)
10. [Testing Error Paths](#10-testing-error-paths)
11. [Design Guidelines](#11-design-guidelines)
12. [Exercises](#12-exercises)

---

## 1. Error Handling Philosophy

### Library vs Application Code

| Context | Approach | Crate | Why |
|---------|----------|-------|-----|
| **Library** | Typed errors | `thiserror` | Callers need to match on specific error variants |
| **Application** | Contextual errors | `anyhow` | Top-level code mostly logs/reports errors |
| **Both** | Typed where it matters | Mix | Public API uses types, internal uses `anyhow` |

### The Error Trait

All Rust errors implement `std::error::Error`:

```rust
pub trait Error: Display + Debug {
    fn source(&self) -> Option<&(dyn Error + 'static)> { None }
    // Deprecated: fn description() and fn cause()
}
```

The chain: `Display` for human-readable messages, `Debug` for developer diagnostics, `source()` for error chaining.

---

## 2. Custom Error Types from Scratch

### Enum-Based Error Type

```rust
use std::fmt;
use std::io;
use std::num::ParseIntError;

#[derive(Debug)]
pub enum AppError {
    Io(io::Error),
    Parse(ParseIntError),
    Config { key: String, message: String },
    NotFound(String),
    Unauthorized,
}

impl fmt::Display for AppError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AppError::Io(e) => write!(f, "I/O error: {e}"),
            AppError::Parse(e) => write!(f, "Parse error: {e}"),
            AppError::Config { key, message } => {
                write!(f, "Config error for '{key}': {message}")
            }
            AppError::NotFound(item) => write!(f, "Not found: {item}"),
            AppError::Unauthorized => write!(f, "Unauthorized"),
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

// From impls for automatic conversion with ?
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

// Usage
fn load_config(path: &str) -> Result<u32, AppError> {
    let content = std::fs::read_to_string(path)?;  // io::Error → AppError::Io
    let value: u32 = content.trim().parse()?;       // ParseIntError → AppError::Parse
    Ok(value)
}
```

That's a lot of boilerplate. `thiserror` eliminates most of it.

---

## 3. thiserror: Library Error Types

`thiserror` derives `Display`, `Error`, and `From` implementations:

```rust
use thiserror::Error;

#[derive(Debug, Error)]
pub enum DataError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Parse error at line {line}: {message}")]
    Parse { line: usize, message: String },

    #[error("Validation failed: {0}")]
    Validation(String),

    #[error("Record not found: {id}")]
    NotFound { id: u64 },

    #[error("Permission denied for user {user} on resource {resource}")]
    Forbidden { user: String, resource: String },

    #[error(transparent)]  // Delegate Display to the inner error
    Other(#[from] anyhow::Error),
}

// The #[from] attribute generates From<io::Error> for DataError
// The #[error("...")] attribute generates the Display impl

fn read_record(path: &str, id: u64) -> Result<String, DataError> {
    let content = std::fs::read_to_string(path)?;  // auto-converted

    let record = content
        .lines()
        .find(|line| line.starts_with(&id.to_string()))
        .ok_or(DataError::NotFound { id })?;

    if record.contains("RESTRICTED") {
        return Err(DataError::Forbidden {
            user: "anonymous".into(),
            resource: format!("record:{id}"),
        });
    }

    Ok(record.to_string())
}
```

### Struct Error Types

```rust
use thiserror::Error;

#[derive(Debug, Error)]
#[error("Connection failed to {host}:{port} after {attempts} attempts")]
pub struct ConnectionError {
    pub host: String,
    pub port: u16,
    pub attempts: u32,
    #[source]
    pub cause: Option<std::io::Error>,
}

fn connect(host: &str, port: u16) -> Result<(), ConnectionError> {
    Err(ConnectionError {
        host: host.to_string(),
        port,
        attempts: 3,
        cause: Some(std::io::Error::new(
            std::io::ErrorKind::ConnectionRefused,
            "connection refused",
        )),
    })
}
```

---

## 4. anyhow: Application Error Handling

`anyhow::Error` wraps any error and adds context. Perfect for application code:

```rust
use anyhow::{Context, Result, bail, ensure};

fn load_config(path: &str) -> Result<Config> {
    let content = std::fs::read_to_string(path)
        .with_context(|| format!("Failed to read config file: {path}"))?;

    let config: Config = toml::from_str(&content)
        .with_context(|| format!("Failed to parse config file: {path}"))?;

    ensure!(config.port > 0, "Port must be positive, got {}", config.port);

    if config.host.is_empty() {
        bail!("Host cannot be empty");
    }

    Ok(config)
}

#[derive(serde::Deserialize)]
struct Config {
    host: String,
    port: u16,
}

fn main() -> Result<()> {
    let config = load_config("config.toml")?;
    println!("Loaded config: {}:{}", config.host, config.port);
    Ok(())
}
```

### anyhow Features

```rust
use anyhow::{anyhow, Context, Result};

fn process() -> Result<()> {
    // Create ad-hoc errors
    let _ = Err(anyhow!("Something went wrong"))?;

    // Attach context to existing errors
    let data = std::fs::read("data.bin")
        .context("Failed to load data file")?;

    // Chain multiple contexts
    let parsed = parse_data(&data)
        .context("Data parsing failed")
        .context("While processing input")?;

    // Downcast to check for specific errors
    if let Err(e) = might_fail() {
        if let Some(io_err) = e.downcast_ref::<std::io::Error>() {
            if io_err.kind() == std::io::ErrorKind::NotFound {
                println!("File not found, using defaults");
                return Ok(());
            }
        }
        return Err(e);
    }

    Ok(())
}

fn parse_data(data: &[u8]) -> Result<Vec<u8>> {
    Ok(data.to_vec())
}

fn might_fail() -> Result<()> {
    Ok(())
}
```

---

## 5. Error Composition Across Layers

### Layer Architecture

```rust
// Domain layer — typed errors
mod domain {
    use thiserror::Error;

    #[derive(Debug, Error)]
    pub enum DomainError {
        #[error("User not found: {0}")]
        UserNotFound(u64),
        #[error("Invalid email: {0}")]
        InvalidEmail(String),
        #[error("Duplicate username: {0}")]
        DuplicateUsername(String),
    }
}

// Infrastructure layer — wraps external errors
mod infra {
    use thiserror::Error;

    #[derive(Debug, Error)]
    pub enum InfraError {
        #[error("Database error: {0}")]
        Database(#[from] sqlx::Error),
        #[error("Cache error: {0}")]
        Cache(String),
        #[error("External API error: {0}")]
        ExternalApi(#[from] reqwest::Error),
    }
}

// Service layer — composes domain and infra errors
mod service {
    use thiserror::Error;
    use super::{domain, infra};

    #[derive(Debug, Error)]
    pub enum ServiceError {
        #[error(transparent)]
        Domain(#[from] domain::DomainError),
        #[error(transparent)]
        Infra(#[from] infra::InfraError),
        #[error("Service unavailable: {0}")]
        Unavailable(String),
    }
}

// HTTP layer — converts service errors to status codes
mod http {
    use super::service::ServiceError;
    use super::domain::DomainError;
    use axum::http::StatusCode;
    use axum::response::{IntoResponse, Response};

    impl IntoResponse for ServiceError {
        fn into_response(self) -> Response {
            let (status, message) = match &self {
                ServiceError::Domain(DomainError::UserNotFound(_)) =>
                    (StatusCode::NOT_FOUND, self.to_string()),
                ServiceError::Domain(DomainError::InvalidEmail(_)) =>
                    (StatusCode::BAD_REQUEST, self.to_string()),
                ServiceError::Domain(DomainError::DuplicateUsername(_)) =>
                    (StatusCode::CONFLICT, self.to_string()),
                ServiceError::Infra(_) =>
                    (StatusCode::INTERNAL_SERVER_ERROR, "Internal error".into()),
                ServiceError::Unavailable(_) =>
                    (StatusCode::SERVICE_UNAVAILABLE, self.to_string()),
            };

            (status, message).into_response()
        }
    }
}
```

---

## 6. Error Context and Backtraces

### Building Error Chains

```rust
use anyhow::{Context, Result};

fn read_user_settings(user_id: u64) -> Result<Settings> {
    let path = format!("/users/{user_id}/settings.json");

    let content = std::fs::read_to_string(&path)
        .with_context(|| format!("Failed to read settings for user {user_id}"))?;

    let settings: Settings = serde_json::from_str(&content)
        .with_context(|| format!("Failed to parse settings at {path}"))?;

    validate_settings(&settings)
        .with_context(|| format!("Invalid settings for user {user_id}"))?;

    Ok(settings)
}

// Error output with chain:
// Error: Invalid settings for user 42
//
// Caused by:
//     0: Failed to parse settings at /users/42/settings.json
//     1: expected value at line 5 column 3

#[derive(serde::Deserialize)]
struct Settings {
    theme: String,
}

fn validate_settings(s: &Settings) -> Result<()> {
    Ok(())
}
```

### Backtraces

```rust
use std::backtrace::Backtrace;
use thiserror::Error;

#[derive(Debug, Error)]
#[error("Critical failure: {message}")]
pub struct CriticalError {
    pub message: String,
    pub backtrace: Backtrace,
}

impl CriticalError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            backtrace: Backtrace::capture(),
        }
    }
}

// anyhow captures backtraces automatically when RUST_BACKTRACE=1
// anyhow::Error has a .backtrace() method
```

---

## 7. Recovery Patterns

### Retry with Exponential Backoff

```rust
use std::time::Duration;
use tokio::time::sleep;

pub async fn retry<F, Fut, T, E>(
    max_attempts: u32,
    initial_delay: Duration,
    mut operation: F,
) -> Result<T, E>
where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = Result<T, E>>,
    E: std::fmt::Display,
{
    let mut delay = initial_delay;
    let mut last_error = None;

    for attempt in 1..=max_attempts {
        match operation().await {
            Ok(value) => return Ok(value),
            Err(e) => {
                eprintln!("Attempt {attempt}/{max_attempts} failed: {e}");
                last_error = Some(e);

                if attempt < max_attempts {
                    sleep(delay).await;
                    delay *= 2;  // Exponential backoff
                }
            }
        }
    }

    Err(last_error.unwrap())
}

// Usage:
// let result = retry(3, Duration::from_millis(100), || async {
//     connect_to_database().await
// }).await?;
```

### Fallback Chain

```rust
async fn get_user_data(user_id: u64) -> Result<UserData, anyhow::Error> {
    // Try cache first
    if let Ok(data) = cache_lookup(user_id).await {
        return Ok(data);
    }

    // Fall back to database
    if let Ok(data) = db_lookup(user_id).await {
        // Populate cache for next time
        let _ = cache_store(user_id, &data).await;
        return Ok(data);
    }

    // Fall back to default
    Ok(UserData::default_for(user_id))
}

struct UserData {
    id: u64,
    name: String,
}

impl UserData {
    fn default_for(id: u64) -> Self {
        UserData { id, name: format!("User {id}") }
    }
}

async fn cache_lookup(id: u64) -> Result<UserData, anyhow::Error> {
    anyhow::bail!("cache miss")
}

async fn db_lookup(id: u64) -> Result<UserData, anyhow::Error> {
    Ok(UserData { id, name: "Alice".into() })
}

async fn cache_store(id: u64, data: &UserData) -> Result<(), anyhow::Error> {
    Ok(())
}
```

### Circuit Breaker

```rust
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::time::{Duration, Instant};

pub struct CircuitBreaker {
    failure_count: AtomicU32,
    threshold: u32,
    reset_timeout: Duration,
    last_failure: std::sync::Mutex<Option<Instant>>,
}

#[derive(Debug, PartialEq)]
pub enum CircuitState {
    Closed,     // Normal operation
    Open,       // Failing, reject requests
    HalfOpen,   // Testing if service recovered
}

impl CircuitBreaker {
    pub fn new(threshold: u32, reset_timeout: Duration) -> Self {
        Self {
            failure_count: AtomicU32::new(0),
            threshold,
            reset_timeout,
            last_failure: std::sync::Mutex::new(None),
        }
    }

    pub fn state(&self) -> CircuitState {
        let failures = self.failure_count.load(Ordering::Relaxed);
        if failures < self.threshold {
            return CircuitState::Closed;
        }

        let last = self.last_failure.lock().unwrap();
        if let Some(last_time) = *last {
            if last_time.elapsed() > self.reset_timeout {
                CircuitState::HalfOpen
            } else {
                CircuitState::Open
            }
        } else {
            CircuitState::Closed
        }
    }

    pub fn record_success(&self) {
        self.failure_count.store(0, Ordering::Relaxed);
        *self.last_failure.lock().unwrap() = None;
    }

    pub fn record_failure(&self) {
        self.failure_count.fetch_add(1, Ordering::Relaxed);
        *self.last_failure.lock().unwrap() = Some(Instant::now());
    }

    pub async fn call<F, Fut, T, E>(&self, operation: F) -> Result<T, CircuitError<E>>
    where
        F: FnOnce() -> Fut,
        Fut: std::future::Future<Output = Result<T, E>>,
    {
        match self.state() {
            CircuitState::Open => Err(CircuitError::Open),
            _ => {
                match operation().await {
                    Ok(value) => {
                        self.record_success();
                        Ok(value)
                    }
                    Err(e) => {
                        self.record_failure();
                        Err(CircuitError::Inner(e))
                    }
                }
            }
        }
    }
}

#[derive(Debug)]
pub enum CircuitError<E> {
    Open,
    Inner(E),
}
```

---

## 8. Error Handling in Async Code

### Handling Errors from Spawned Tasks

```rust
use tokio::task::JoinError;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // JoinHandle returns Result<T, JoinError>
    // where JoinError means the task panicked or was cancelled
    let handle = tokio::spawn(async {
        might_fail().await
    });

    match handle.await {
        Ok(Ok(value)) => println!("Success: {value}"),
        Ok(Err(e)) => println!("Task returned error: {e}"),
        Err(join_err) => {
            if join_err.is_panic() {
                println!("Task panicked!");
            } else {
                println!("Task was cancelled");
            }
        }
    }

    Ok(())
}

async fn might_fail() -> anyhow::Result<String> {
    Ok("done".into())
}
```

### Collecting Results from Multiple Tasks

```rust
use futures_util::future::join_all;

async fn fetch_all(urls: &[&str]) -> Vec<anyhow::Result<String>> {
    let futures: Vec<_> = urls.iter().map(|url| async move {
        let resp = reqwest::get(*url).await?;
        let body = resp.text().await?;
        Ok(body)
    }).collect();

    join_all(futures).await
}

// Partition into successes and failures
async fn fetch_with_report(urls: &[&str]) {
    let results = fetch_all(urls).await;

    let (successes, failures): (Vec<_>, Vec<_>) = results
        .into_iter()
        .enumerate()
        .partition(|(_, r)| r.is_ok());

    println!("{} succeeded, {} failed", successes.len(), failures.len());

    for (i, result) in failures {
        if let Err(e) = result {
            eprintln!("URL[{i}] failed: {e:#}");
        }
    }
}
```

---

## 9. Logging and Reporting

### Structured Error Logging with tracing

```rust
use tracing::{error, warn, info, instrument};
use anyhow::{Context, Result};

#[instrument(skip(password))]
async fn authenticate(username: &str, password: &str) -> Result<User> {
    let user = find_user(username).await
        .with_context(|| format!("Authentication failed for user '{username}'"))?;

    if !verify_password(&user, password) {
        warn!(username, "Invalid password attempt");
        anyhow::bail!("Invalid credentials");
    }

    info!(username, user_id = user.id, "User authenticated");
    Ok(user)
}

struct User { id: u64 }

async fn find_user(username: &str) -> Result<User> {
    Ok(User { id: 1 })
}

fn verify_password(user: &User, password: &str) -> bool {
    true
}
```

### Error Reporting for End Users

```rust
fn report_error(error: &anyhow::Error) {
    // For end users: just the top-level message
    eprintln!("Error: {error}");

    // For developers: full error chain
    if std::env::var("RUST_LOG").is_ok() {
        eprintln!("\nError chain:");
        for (i, cause) in error.chain().enumerate() {
            eprintln!("  {i}: {cause}");
        }

        // Backtrace if available
        let bt = error.backtrace();
        if bt.status() == std::backtrace::BacktraceStatus::Captured {
            eprintln!("\nBacktrace:\n{bt}");
        }
    }
}
```

---

## 10. Testing Error Paths

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_invalid_email() {
        let result = validate_email("not-an-email");
        assert!(result.is_err());

        let err = result.unwrap_err();
        assert!(matches!(err, DomainError::InvalidEmail(_)));
        assert!(err.to_string().contains("not-an-email"));
    }

    #[test]
    fn test_error_chain() {
        let result = load_config("/nonexistent/path");
        let err = result.unwrap_err();

        // Check the error chain
        let mut chain = err.chain();
        assert!(chain.next().unwrap().to_string().contains("config"));
        assert!(chain.next().unwrap().to_string().contains("No such file"));
    }

    #[test]
    fn test_error_downcast() {
        let result: Result<(), anyhow::Error> = Err(
            DomainError::UserNotFound(42).into()
        );

        let err = result.unwrap_err();
        assert!(err.downcast_ref::<DomainError>().is_some());

        match err.downcast_ref::<DomainError>() {
            Some(DomainError::UserNotFound(id)) => assert_eq!(*id, 42),
            _ => panic!("Expected UserNotFound"),
        }
    }

    // Test that error messages are helpful
    #[test]
    fn test_error_messages_are_descriptive() {
        let err = DomainError::InvalidEmail("bad".into());
        let msg = err.to_string();
        assert!(msg.contains("bad"), "Error should mention the invalid value");
        assert!(msg.to_lowercase().contains("email"), "Error should mention what's invalid");
    }
}

use thiserror::Error;

#[derive(Debug, Error)]
enum DomainError {
    #[error("Invalid email: {0}")]
    InvalidEmail(String),
    #[error("User not found: {0}")]
    UserNotFound(u64),
}

fn validate_email(email: &str) -> Result<(), DomainError> {
    if !email.contains('@') {
        return Err(DomainError::InvalidEmail(email.to_string()));
    }
    Ok(())
}

fn load_config(path: &str) -> anyhow::Result<()> {
    std::fs::read_to_string(path)
        .map_err(|e| anyhow::anyhow!(e))
        .context("Failed to load config")?;
    Ok(())
}
```

---

## 11. Design Guidelines

### Do's

1. **Be specific in library errors**: Use enums with descriptive variants
2. **Add context when propagating**: Use `.context()` or `.with_context()`
3. **Include relevant data in errors**: IDs, paths, values that caused the failure
4. **Implement `source()`**: Enable error chain traversal
5. **Test error paths**: They are as important as happy paths

### Don'ts

1. **Don't use `String` as your error type**: No type safety, no composition
2. **Don't `.unwrap()` in library code**: Return `Result` instead
3. **Don't discard error context**: `map_err(|_| MyError)` loses the cause
4. **Don't panic for recoverable errors**: Reserve `panic!` for programmer bugs
5. **Don't log AND return the same error**: Choose one, or the caller logs it twice

### Decision Tree

```
Is this library code or application code?
├── Library → Use thiserror, return typed errors
├── Application → Use anyhow, add context
└── Both → Public API uses thiserror, internals use anyhow
```

---

## 12. Exercises

1. **Error type hierarchy**: Design a complete error type hierarchy for a file processing library with errors for: I/O, parsing (with line/column info), validation, and encoding. Include proper `source()` chains.

2. **Retry middleware**: Build a generic async retry function that supports: configurable max attempts, exponential backoff with jitter, and a predicate for which errors are retryable.

3. **Circuit breaker**: Extend the circuit breaker pattern to support: configurable success threshold for closing, sliding window failure rate, and integration with async functions.

4. **Error aggregation**: Write a function that runs N async operations concurrently and returns a combined result: all successes OR a report of which operations failed with their individual errors.

5. **User-friendly errors**: Build an error reporter that takes an `anyhow::Error` and produces both a user-friendly message (no internal details) and a developer-friendly report (full chain + backtrace).

---

## References

- [thiserror documentation](https://docs.rs/thiserror/latest/thiserror/)
- [anyhow documentation](https://docs.rs/anyhow/latest/anyhow/)
- [Error Handling in Rust (Rust Book)](https://doc.rust-lang.org/book/ch09-00-error-handling.html)
- [Jane Lusby: Error Handling in Rust](https://www.youtube.com/watch?v=rAF8mLI0naQ)

---

**Previous**: [Network Programming](./11_Network_Programming.md) | **Next**: [Performance and Profiling](./13_Performance_Profiling.md)
