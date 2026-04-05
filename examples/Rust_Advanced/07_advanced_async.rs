// 07_advanced_async.rs — Advanced async patterns (no external runtime)
//
// Run: rustc 07_advanced_async.rs && ./07_advanced_async
//
// Demonstrates: async combinators, cancellation patterns, and async streams
// using a minimal executor. In production, use Tokio.

use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll, RawWaker, RawWakerVTable, Waker};

fn main() {
    println!("=== Race / Select Pattern ===");
    race_demo();

    println!("\n=== Timeout Pattern ===");
    timeout_demo();

    println!("\n=== Async Stream Pattern ===");
    stream_demo();

    println!("\n=== Cancellation Pattern ===");
    cancellation_demo();

    println!("\n=== Retry with Backoff ===");
    retry_demo();
}

// --- Minimal executor ---

fn block_on<F: Future>(mut future: F) -> F::Output {
    fn dummy_raw_waker() -> RawWaker {
        fn no_op(_: *const ()) {}
        fn clone(p: *const ()) -> RawWaker { RawWaker::new(p, &VTABLE) }
        const VTABLE: RawWakerVTable = RawWakerVTable::new(clone, no_op, no_op, no_op);
        RawWaker::new(std::ptr::null(), &VTABLE)
    }
    let waker = unsafe { Waker::from_raw(dummy_raw_waker()) };
    let mut cx = Context::from_waker(&waker);
    let mut future = unsafe { Pin::new_unchecked(&mut future) };
    loop {
        match future.as_mut().poll(&mut cx) {
            Poll::Ready(output) => return output,
            Poll::Pending => panic!("Blocked in minimal executor"),
        }
    }
}

// --- Race: return the first completed result ---

async fn fast_service() -> (&'static str, u32) {
    ("fast", 42)
}

async fn slow_service() -> (&'static str, u32) {
    ("slow", 99)
}

/// Simulates select! — in this sync executor both resolve immediately,
/// so we demonstrate the pattern structure
async fn race_two() -> String {
    let (name, value) = fast_service().await;
    // In real code with tokio::select!, slow_service would be cancelled
    format!("Winner: {name} with value {value}")
}

fn race_demo() {
    let result = block_on(race_two());
    println!("  {result}");
}

// --- Timeout pattern ---

#[derive(Debug)]
enum TimeoutError<E> {
    TimedOut,
    Inner(E),
}

impl<E: std::fmt::Display> std::fmt::Display for TimeoutError<E> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TimeoutError::TimedOut => write!(f, "operation timed out"),
            TimeoutError::Inner(e) => write!(f, "{e}"),
        }
    }
}

async fn with_timeout<F, T, E>(future: F, _deadline_ms: u64) -> Result<T, TimeoutError<E>>
where
    F: Future<Output = Result<T, E>>,
{
    // In real code, this would race against tokio::time::sleep
    future.await.map_err(TimeoutError::Inner)
}

async fn database_query() -> Result<String, String> {
    Ok("42 rows".to_string())
}

fn timeout_demo() {
    let result = block_on(with_timeout(database_query(), 5000));
    match result {
        Ok(data) => println!("  Query result: {data}"),
        Err(e) => println!("  Error: {e}"),
    }
}

// --- Async stream pattern ---

struct AsyncRange {
    current: u32,
    end: u32,
}

impl AsyncRange {
    fn new(start: u32, end: u32) -> Self {
        AsyncRange { current: start, end }
    }

    async fn next(&mut self) -> Option<u32> {
        if self.current < self.end {
            let val = self.current;
            self.current += 1;
            Some(val)
        } else {
            None
        }
    }
}

async fn consume_stream() -> Vec<u32> {
    let mut stream = AsyncRange::new(1, 6);
    let mut results = Vec::new();

    while let Some(value) = stream.next().await {
        results.push(value * value);
    }

    results
}

fn stream_demo() {
    let squares = block_on(consume_stream());
    println!("  Squares: {squares:?}");
}

// --- Cancellation token pattern ---

struct CancellationToken {
    cancelled: bool,
}

impl CancellationToken {
    fn new() -> Self {
        CancellationToken { cancelled: false }
    }

    fn cancel(&mut self) {
        self.cancelled = true;
    }

    fn is_cancelled(&self) -> bool {
        self.cancelled
    }
}

async fn long_running_task(token: &CancellationToken) -> Result<String, &'static str> {
    for i in 1..=5 {
        if token.is_cancelled() {
            return Err("Task was cancelled");
        }
        // Simulate work
        println!("  Step {i}/5...");
    }
    Ok("Completed all steps".to_string())
}

fn cancellation_demo() {
    // Run to completion
    let token = CancellationToken::new();
    let result = block_on(long_running_task(&token));
    println!("  Result: {result:?}");

    // Cancelled before start
    let mut token = CancellationToken::new();
    token.cancel();
    let result = block_on(long_running_task(&token));
    println!("  Cancelled result: {result:?}");
}

// --- Retry with backoff ---

async fn unreliable_service(attempt: u32) -> Result<String, String> {
    if attempt < 3 {
        Err(format!("Attempt {attempt}: connection refused"))
    } else {
        Ok("Success!".to_string())
    }
}

async fn retry_with_backoff(max_attempts: u32) -> Result<String, String> {
    let mut last_error = String::new();

    for attempt in 1..=max_attempts {
        match unreliable_service(attempt).await {
            Ok(result) => {
                println!("  Attempt {attempt}: succeeded");
                return Ok(result);
            }
            Err(e) => {
                let backoff = 100 * 2u64.pow(attempt - 1); // Exponential backoff
                println!("  {e} (would wait {backoff}ms)");
                last_error = e;
            }
        }
    }

    Err(format!("All {max_attempts} attempts failed. Last: {last_error}"))
}

fn retry_demo() {
    let result = block_on(retry_with_backoff(5));
    println!("  Final: {result:?}");
}
