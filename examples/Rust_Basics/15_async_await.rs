// 15_async_await.rs — Async/await fundamentals (no external runtime)
//
// This example demonstrates the core concepts of async Rust using
// a minimal manual executor. In real projects, use Tokio or async-std.
//
// Run: rustc 15_async_await.rs && ./15_async_await

use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll, RawWaker, RawWakerVTable, Waker};

fn main() {
    println!("=== Manual Future ===");
    manual_future_demo();

    println!("\n=== Async/Await Basics ===");
    async_basics();

    println!("\n=== Chaining Async Functions ===");
    chaining_demo();

    println!("\n=== Async Control Flow ===");
    async_control_flow();
}

// --- Minimal executor: polls a future to completion ---

fn block_on<F: Future>(mut future: F) -> F::Output {
    // Create a no-op waker (sufficient for synchronous futures)
    fn dummy_raw_waker() -> RawWaker {
        fn no_op(_: *const ()) {}
        fn clone(p: *const ()) -> RawWaker { RawWaker::new(p, &VTABLE) }
        const VTABLE: RawWakerVTable = RawWakerVTable::new(clone, no_op, no_op, no_op);
        RawWaker::new(std::ptr::null(), &VTABLE)
    }
    let waker = unsafe { Waker::from_raw(dummy_raw_waker()) };
    let mut cx = Context::from_waker(&waker);

    // SAFETY: we never move `future` after pinning
    let mut future = unsafe { Pin::new_unchecked(&mut future) };

    loop {
        match future.as_mut().poll(&mut cx) {
            Poll::Ready(output) => return output,
            Poll::Pending => panic!("Future returned Pending in this simple executor"),
        }
    }
}

// --- Manual Future implementation ---

struct Countdown {
    remaining: u32,
}

impl Future for Countdown {
    type Output = String;

    fn poll(mut self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Self::Output> {
        if self.remaining == 0 {
            Poll::Ready("Liftoff!".to_string())
        } else {
            println!("  {}...", self.remaining);
            self.remaining = 0; // Resolve immediately for demo
            Poll::Ready("Liftoff!".to_string())
        }
    }
}

fn manual_future_demo() {
    let countdown = Countdown { remaining: 3 };
    let result = block_on(countdown);
    println!("  {result}");
}

// --- Async functions ---

async fn fetch_data(id: u32) -> String {
    // In real code this would be an async HTTP call
    format!("Data for item #{id}")
}

async fn compute(value: i32) -> i32 {
    value * value + 1
}

fn async_basics() {
    // Calling an async fn returns a Future — it doesn't run yet
    let future = fetch_data(42);
    let result = block_on(future);
    println!("  {result}");

    let result = block_on(compute(7));
    println!("  compute(7) = {result}");
}

// --- Chaining ---

async fn get_user_id(name: &str) -> u32 {
    match name {
        "alice" => 1,
        "bob" => 2,
        _ => 0,
    }
}

async fn get_user_email(id: u32) -> String {
    format!("user{}@example.com", id)
}

async fn lookup_email(name: &str) -> String {
    let id = get_user_id(name).await;
    let email = get_user_email(id).await;
    email
}

fn chaining_demo() {
    let email = block_on(lookup_email("alice"));
    println!("  alice's email: {email}");

    let email = block_on(lookup_email("bob"));
    println!("  bob's email: {email}");
}

// --- Async control flow ---

async fn classify(n: i32) -> &'static str {
    if n < 0 {
        "negative"
    } else if n == 0 {
        "zero"
    } else {
        "positive"
    }
}

async fn process_batch(items: &[i32]) -> Vec<String> {
    let mut results = Vec::new();
    for &item in items {
        let label = classify(item).await;
        results.push(format!("{item}: {label}"));
    }
    results
}

fn async_control_flow() {
    let items = [3, -1, 0, 42, -7];
    let results = block_on(process_batch(&items));
    for r in &results {
        println!("  {r}");
    }
}
