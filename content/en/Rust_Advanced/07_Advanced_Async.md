# 23. Advanced Async

**Previous**: [Advanced Traits](./06_Advanced_Traits.md) | **Next**: [FFI and Interop](./08_FFI_and_Interop.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Understand Tokio runtime internals: task scheduling, work-stealing, and thread pools
2. Use `tokio::select!` for concurrent branch execution with cancel safety
3. Work with async streams for processing sequences of asynchronous values
4. Build middleware stacks with the Tower framework
5. Handle cancellation, timeouts, and graceful shutdown in async applications

---

Lesson 15 introduced `async`/`await` fundamentals. This lesson dives into production async Rust: the Tokio runtime's architecture, composing futures with `select!`, stream processing, the Tower middleware ecosystem, and the critical topic of cancel safety.

## Table of Contents
1. [Tokio Runtime Internals](#1-tokio-runtime-internals)
2. [Task Spawning and JoinHandle](#2-task-spawning-and-joinhandle)
3. [tokio::select!](#3-tokioselect)
4. [Cancel Safety](#4-cancel-safety)
5. [Async Streams](#5-async-streams)
6. [Channels in Async Code](#6-channels-in-async-code)
7. [Timeouts and Deadlines](#7-timeouts-and-deadlines)
8. [Graceful Shutdown](#8-graceful-shutdown)
9. [Tower Middleware](#9-tower-middleware)
10. [Async Patterns](#10-async-patterns)
11. [Performance Considerations](#11-performance-considerations)
12. [Exercises](#12-exercises)

---

## 1. Tokio Runtime Internals

Tokio is the most widely-used async runtime for Rust. Understanding its architecture helps you write correct and performant async code.

### Runtime Flavors

```rust
// Multi-threaded runtime (default) — work-stealing scheduler
#[tokio::main]
async fn main() {
    // Uses num_cpus threads by default
    println!("Running on multi-threaded runtime");
}

// Customize thread count
#[tokio::main(worker_threads = 4)]
async fn main() {
    println!("Running on 4 worker threads");
}

// Current-thread runtime — single thread, cooperative scheduling
#[tokio::main(flavor = "current_thread")]
async fn main() {
    // All tasks run on one thread — good for lightweight apps
    println!("Running on current-thread runtime");
}

// Manual runtime construction
fn main() {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(4)
        .thread_name("my-worker")
        .enable_all()
        .build()
        .unwrap();

    rt.block_on(async {
        println!("Running on custom runtime");
    });
}
```

### Work-Stealing Scheduler

The multi-threaded runtime uses a **work-stealing** scheduler:

```
Thread 1: [Task A] [Task C] [Task E]     ← local queue
Thread 2: [Task B] [Task D]              ← local queue
Thread 3: []                             ← idle, steals from Thread 1
Thread 4: [Task F]                       ← local queue

Global inject queue: [Task G, Task H]    ← new tasks land here first
```

- Each worker thread has a **local task queue** (256-slot ring buffer)
- New tasks go to the **global inject queue** first
- Idle threads **steal** from other threads' local queues
- This minimizes contention while keeping all cores busy

### Cooperative Scheduling

Tokio tasks are cooperatively scheduled — they must yield control by hitting an `.await` point:

```rust
#[tokio::main]
async fn main() {
    // BAD: This blocks the entire thread — no other tasks can run
    tokio::spawn(async {
        loop {
            // CPU-intensive work without yielding
            std::thread::sleep(std::time::Duration::from_secs(1));
        }
    });

    // GOOD: Use tokio::task::yield_now() for CPU-bound work
    tokio::spawn(async {
        for i in 0..1_000_000 {
            if i % 1000 == 0 {
                tokio::task::yield_now().await;  // Give other tasks a chance
            }
            // ... work ...
        }
    });

    // BEST: Use spawn_blocking for truly CPU-bound work
    let result = tokio::task::spawn_blocking(|| {
        // This runs on a separate thread pool
        (0..1_000_000).sum::<u64>()
    }).await.unwrap();

    println!("Result: {result}");
}
```

---

## 2. Task Spawning and JoinHandle

### Spawning Tasks

```rust
use tokio::task::JoinHandle;

#[tokio::main]
async fn main() {
    // Spawn returns a JoinHandle
    let handle: JoinHandle<String> = tokio::spawn(async {
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        "Task completed".to_string()
    });

    // Await the result
    let result = handle.await.unwrap();
    println!("{result}");

    // Spawn multiple tasks and collect results
    let mut handles = Vec::new();
    for i in 0..5 {
        handles.push(tokio::spawn(async move {
            tokio::time::sleep(std::time::Duration::from_millis(50 * i)).await;
            i * 10
        }));
    }

    let mut results = Vec::new();
    for handle in handles {
        results.push(handle.await.unwrap());
    }
    println!("Results: {results:?}");  // [0, 10, 20, 30, 40]
}
```

### JoinSet for Managing Task Groups

```rust
use tokio::task::JoinSet;

#[tokio::main]
async fn main() {
    let mut set = JoinSet::new();

    for i in 0..5 {
        set.spawn(async move {
            tokio::time::sleep(std::time::Duration::from_millis(100 - i * 10)).await;
            format!("Task {i} done")
        });
    }

    // Collect results as they complete (not in spawn order!)
    while let Some(result) = set.join_next().await {
        match result {
            Ok(msg) => println!("{msg}"),
            Err(e) => eprintln!("Task failed: {e}"),
        }
    }

    // Abort all remaining tasks
    // set.abort_all();
}
```

---

## 3. tokio::select!

`select!` waits for multiple futures concurrently and acts on the first one to complete:

```rust
use tokio::time::{sleep, Duration};

#[tokio::main]
async fn main() {
    let mut interval = tokio::time::interval(Duration::from_secs(1));

    tokio::select! {
        _ = sleep(Duration::from_secs(3)) => {
            println!("3-second timer fired");
        }
        _ = sleep(Duration::from_secs(5)) => {
            println!("5-second timer fired");  // Never reached
        }
    }
    // Only the 3-second branch runs; the 5-second future is DROPPED
}
```

### select! with Pattern Matching

```rust
use tokio::sync::mpsc;

#[tokio::main]
async fn main() {
    let (tx1, mut rx1) = mpsc::channel::<String>(32);
    let (tx2, mut rx2) = mpsc::channel::<i32>(32);

    // Simulate producers
    tokio::spawn(async move {
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        tx1.send("hello".into()).await.unwrap();
    });
    tokio::spawn(async move {
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        tx2.send(42).await.unwrap();
    });

    tokio::select! {
        Some(msg) = rx1.recv() => {
            println!("Got string: {msg}");
        }
        Some(num) = rx2.recv() => {
            println!("Got number: {num}");
        }
    }
}
```

### select! in a Loop

```rust
use tokio::sync::mpsc;
use tokio::signal;

#[tokio::main]
async fn main() {
    let (tx, mut rx) = mpsc::channel::<String>(32);

    // Producer
    tokio::spawn(async move {
        for i in 0..10 {
            tokio::time::sleep(std::time::Duration::from_millis(200)).await;
            if tx.send(format!("Message {i}")).await.is_err() {
                break;
            }
        }
    });

    let mut count = 0;
    loop {
        tokio::select! {
            Some(msg) = rx.recv() => {
                println!("Received: {msg}");
                count += 1;
            }
            _ = signal::ctrl_c() => {
                println!("\nShutting down after {count} messages");
                break;
            }
            else => {
                println!("All channels closed");
                break;
            }
        }
    }
}
```

---

## 4. Cancel Safety

When a branch in `select!` is not chosen, its future is **dropped** (cancelled). This can leave state inconsistent if the future was in the middle of an operation:

```rust
use tokio::sync::mpsc;

// NOT cancel-safe: if cancelled between recv() completing and processing,
// the message is lost
async fn process_messages(rx: &mut mpsc::Receiver<String>) {
    // If this future is dropped right after recv() returns but before
    // we finish processing, the message is consumed but not processed!
    if let Some(msg) = rx.recv().await {
        println!("Processing: {msg}");
        // ... expensive work ...
    }
}

// Cancel-safe alternative
async fn process_messages_safe(
    rx: &mut mpsc::Receiver<String>,
    buffer: &mut Option<String>,
) {
    // Check if we have a buffered message from a previous cancellation
    let msg = if let Some(msg) = buffer.take() {
        msg
    } else {
        match rx.recv().await {
            Some(msg) => msg,
            None => return,
        }
    };

    // Process the message
    println!("Processing: {msg}");
    // If cancelled here, msg is dropped, but that's OK —
    // we already consumed it from the channel intentionally
}
```

### Cancel-Safe Operations

| Operation | Cancel-Safe? | Notes |
|-----------|-------------|-------|
| `mpsc::Receiver::recv()` | Yes | Message stays in channel if cancelled before completion |
| `oneshot::Receiver::recv()` | Yes | Value stays in channel |
| `TcpStream::read()` | No | Partial reads may be lost |
| `tokio::io::AsyncReadExt::read_exact()` | No | Partial progress lost |
| `tokio::time::sleep()` | Yes | No state to corrupt |
| `JoinHandle::await` | No | Task continues running but result is lost |

### Making Operations Cancel-Safe

```rust
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

struct SafeReader {
    stream: TcpStream,
    buffer: Vec<u8>,
    bytes_read: usize,
    target_len: usize,
}

impl SafeReader {
    fn new(stream: TcpStream, target_len: usize) -> Self {
        Self {
            stream,
            buffer: vec![0u8; target_len],
            bytes_read: 0,
            target_len,
        }
    }

    /// Cancel-safe read: progress is stored in self, so cancellation
    /// doesn't lose partial data
    async fn read_exact(&mut self) -> std::io::Result<&[u8]> {
        while self.bytes_read < self.target_len {
            let n = self.stream
                .read(&mut self.buffer[self.bytes_read..])
                .await?;
            if n == 0 {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    "connection closed",
                ));
            }
            self.bytes_read += n;
        }
        Ok(&self.buffer[..self.target_len])
    }
}
```

---

## 5. Async Streams

An async stream is like an async iterator — it yields a sequence of values over time:

```rust
use tokio_stream::{self as stream, StreamExt};

#[tokio::main]
async fn main() {
    // Create a stream from an iterator
    let mut s = stream::iter(vec![1, 2, 3, 4, 5]);

    while let Some(value) = s.next().await {
        println!("Got: {value}");
    }

    // Stream combinators (like Iterator but async)
    let doubled: Vec<_> = stream::iter(1..=5)
        .map(|x| x * 2)
        .collect()
        .await;
    println!("Doubled: {doubled:?}");

    // Filter and take
    let result: Vec<_> = stream::iter(1..=100)
        .filter(|x| x % 7 == 0)
        .take(5)
        .collect()
        .await;
    println!("First 5 multiples of 7: {result:?}");
}
```

### Creating Custom Streams

```rust
use std::pin::Pin;
use std::task::{Context, Poll};
use tokio_stream::Stream;

struct Counter {
    current: u64,
    max: u64,
}

impl Stream for Counter {
    type Item = u64;

    fn poll_next(mut self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if self.current < self.max {
            let val = self.current;
            self.current += 1;
            Poll::Ready(Some(val))
        } else {
            Poll::Ready(None)
        }
    }
}

// Using async_stream crate for easier stream creation
use async_stream::stream;

fn countdown(from: u32) -> impl Stream<Item = u32> {
    stream! {
        for i in (0..=from).rev() {
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
            yield i;
        }
    }
}

#[tokio::main]
async fn main() {
    use tokio_stream::StreamExt;

    let mut s = countdown(5);
    while let Some(n) = s.next().await {
        println!("Countdown: {n}");
    }
    println!("Liftoff!");
}
```

### Stream Concurrency

```rust
use tokio_stream::StreamExt;

#[tokio::main]
async fn main() {
    // Process stream items concurrently with buffer_unordered
    let urls = vec![
        "https://example.com/1",
        "https://example.com/2",
        "https://example.com/3",
    ];

    let results: Vec<_> = tokio_stream::iter(urls)
        .map(|url| async move {
            // Simulate HTTP request
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
            format!("Response from {url}")
        })
        .buffer_unordered(3)  // Process up to 3 concurrently
        .collect()
        .await;

    for r in results {
        println!("{r}");
    }
}
```

---

## 6. Channels in Async Code

### mpsc — Multiple Producer, Single Consumer

```rust
use tokio::sync::mpsc;

#[tokio::main]
async fn main() {
    // Bounded channel — backpressure when full
    let (tx, mut rx) = mpsc::channel::<String>(100);

    for i in 0..5 {
        let tx = tx.clone();
        tokio::spawn(async move {
            tx.send(format!("Message from task {i}")).await.unwrap();
        });
    }

    // Drop the original sender so the receiver knows when all senders are done
    drop(tx);

    while let Some(msg) = rx.recv().await {
        println!("{msg}");
    }
}
```

### broadcast — Multiple Producer, Multiple Consumer

```rust
use tokio::sync::broadcast;

#[tokio::main]
async fn main() {
    let (tx, _) = broadcast::channel::<String>(16);

    let mut rx1 = tx.subscribe();
    let mut rx2 = tx.subscribe();

    tokio::spawn(async move {
        tx.send("Hello everyone!".into()).unwrap();
        tx.send("Goodbye!".into()).unwrap();
    });

    tokio::spawn(async move {
        while let Ok(msg) = rx1.recv().await {
            println!("[Subscriber 1] {msg}");
        }
    });

    tokio::spawn(async move {
        while let Ok(msg) = rx2.recv().await {
            println!("[Subscriber 2] {msg}");
        }
    });

    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
}
```

### watch — Single Producer, Multiple Consumer (Latest Value)

```rust
use tokio::sync::watch;

#[tokio::main]
async fn main() {
    let (tx, mut rx) = watch::channel("initial".to_string());

    // Consumer — watches for changes
    let mut rx2 = rx.clone();
    tokio::spawn(async move {
        while rx2.changed().await.is_ok() {
            println!("[Watcher] Value changed to: {}", *rx2.borrow());
        }
    });

    // Producer — update the value
    tx.send("update 1".into()).unwrap();
    tokio::time::sleep(std::time::Duration::from_millis(10)).await;

    tx.send("update 2".into()).unwrap();
    tokio::time::sleep(std::time::Duration::from_millis(10)).await;

    // Current value is always available
    println!("Current: {}", *rx.borrow());
}
```

---

## 7. Timeouts and Deadlines

```rust
use tokio::time::{timeout, Duration, Instant};

async fn slow_operation() -> String {
    tokio::time::sleep(Duration::from_secs(5)).await;
    "Done".into()
}

#[tokio::main]
async fn main() {
    // Simple timeout
    match timeout(Duration::from_secs(1), slow_operation()).await {
        Ok(result) => println!("Got: {result}"),
        Err(_) => println!("Operation timed out"),
    }

    // Deadline-based timeout
    let deadline = Instant::now() + Duration::from_secs(2);
    match tokio::time::timeout_at(deadline, slow_operation()).await {
        Ok(result) => println!("Got: {result}"),
        Err(_) => println!("Deadline exceeded"),
    }

    // Retry with timeout
    let result = retry_with_timeout(3, Duration::from_millis(500)).await;
    println!("Retry result: {result:?}");
}

async fn retry_with_timeout(
    max_retries: u32,
    per_attempt_timeout: Duration,
) -> Result<String, String> {
    for attempt in 1..=max_retries {
        match timeout(per_attempt_timeout, slow_operation()).await {
            Ok(result) => return Ok(result),
            Err(_) => {
                eprintln!("Attempt {attempt}/{max_retries} timed out");
            }
        }
    }
    Err("All attempts timed out".into())
}
```

---

## 8. Graceful Shutdown

```rust
use tokio::sync::{broadcast, mpsc};
use tokio::signal;

#[tokio::main]
async fn main() {
    let (shutdown_tx, _) = broadcast::channel::<()>(1);
    let (done_tx, mut done_rx) = mpsc::channel::<()>(10);

    // Spawn worker tasks
    for id in 0..3 {
        let mut shutdown_rx = shutdown_tx.subscribe();
        let done_tx = done_tx.clone();

        tokio::spawn(async move {
            println!("[Worker {id}] Started");

            loop {
                tokio::select! {
                    _ = shutdown_rx.recv() => {
                        println!("[Worker {id}] Shutting down...");
                        // Cleanup work here
                        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
                        println!("[Worker {id}] Cleanup complete");
                        drop(done_tx);  // Signal that we're done
                        return;
                    }
                    _ = tokio::time::sleep(std::time::Duration::from_secs(1)) => {
                        println!("[Worker {id}] Working...");
                    }
                }
            }
        });
    }

    // Drop our copy of done_tx
    drop(done_tx);

    // Wait for shutdown signal
    signal::ctrl_c().await.expect("Failed to listen for Ctrl+C");
    println!("\nReceived Ctrl+C, initiating shutdown...");

    // Send shutdown signal to all workers
    let _ = shutdown_tx.send(());

    // Wait for all workers to finish
    let _ = done_rx.recv().await;
    println!("All workers shut down. Goodbye!");
}
```

---

## 9. Tower Middleware

Tower is a middleware framework for async services. It's used by axum, tonic, hyper, and other Rust networking libraries:

```rust
use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};
use std::time::Instant;

// The core Tower trait (simplified)
trait Service<Request> {
    type Response;
    type Error;
    type Future: Future<Output = Result<Self::Response, Self::Error>>;

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>>;
    fn call(&mut self, req: Request) -> Self::Future;
}

// A simple echo service
struct EchoService;

impl Service<String> for EchoService {
    type Response = String;
    type Error = std::convert::Infallible;
    type Future = Pin<Box<dyn Future<Output = Result<String, Self::Error>> + Send>>;

    fn poll_ready(&mut self, _cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        Poll::Ready(Ok(()))
    }

    fn call(&mut self, req: String) -> Self::Future {
        Box::pin(async move {
            Ok(format!("Echo: {req}"))
        })
    }
}

// A timing middleware (Layer)
struct TimingLayer;

struct TimingService<S> {
    inner: S,
}

impl<S, Req> Service<Req> for TimingService<S>
where
    S: Service<Req>,
    S::Future: Send + 'static,
    S::Response: std::fmt::Debug + Send + 'static,
    S::Error: Send + 'static,
    Req: Send + 'static,
{
    type Response = S::Response;
    type Error = S::Error;
    type Future = Pin<Box<dyn Future<Output = Result<S::Response, S::Error>> + Send>>;

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        self.inner.poll_ready(cx)
    }

    fn call(&mut self, req: Req) -> Self::Future {
        let start = Instant::now();
        let future = self.inner.call(req);
        Box::pin(async move {
            let result = future.await;
            println!("Request took {:?}", start.elapsed());
            result
        })
    }
}
```

### Using Tower with Axum (Practical)

```rust
use axum::{
    Router,
    routing::get,
    middleware::{self, Next},
    extract::Request,
    response::Response,
};
use std::time::Instant;
use tower::ServiceBuilder;
use tower_http::{
    compression::CompressionLayer,
    cors::CorsLayer,
    timeout::TimeoutLayer,
};

async fn timing_middleware(req: Request, next: Next) -> Response {
    let start = Instant::now();
    let path = req.uri().path().to_string();
    let response = next.run(req).await;
    println!("{path} took {:?}", start.elapsed());
    response
}

async fn hello() -> &'static str {
    "Hello, World!"
}

fn app() -> Router {
    Router::new()
        .route("/", get(hello))
        .layer(
            ServiceBuilder::new()
                .layer(TimeoutLayer::new(std::time::Duration::from_secs(30)))
                .layer(CompressionLayer::new())
                .layer(CorsLayer::permissive())
                .layer(middleware::from_fn(timing_middleware))
        )
}
```

---

## 10. Async Patterns

### Fan-Out / Fan-In

```rust
use tokio::task::JoinSet;

async fn fetch_url(url: &str) -> Result<String, String> {
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    Ok(format!("Content from {url}"))
}

#[tokio::main]
async fn main() {
    let urls = vec![
        "https://api.example.com/users",
        "https://api.example.com/posts",
        "https://api.example.com/comments",
    ];

    // Fan-out: spawn concurrent requests
    let mut set = JoinSet::new();
    for url in &urls {
        let url = url.to_string();
        set.spawn(async move { fetch_url(&url).await });
    }

    // Fan-in: collect results
    let mut results = Vec::new();
    while let Some(result) = set.join_next().await {
        match result {
            Ok(Ok(content)) => results.push(content),
            Ok(Err(e)) => eprintln!("Request error: {e}"),
            Err(e) => eprintln!("Task panic: {e}"),
        }
    }

    println!("Collected {} results", results.len());
}
```

### Rate Limiting

```rust
use tokio::sync::Semaphore;
use std::sync::Arc;

#[tokio::main]
async fn main() {
    let semaphore = Arc::new(Semaphore::new(3));  // Max 3 concurrent operations
    let mut handles = Vec::new();

    for i in 0..10 {
        let sem = semaphore.clone();
        handles.push(tokio::spawn(async move {
            let _permit = sem.acquire().await.unwrap();
            println!("[{i}] Start (active permits: {})", 3 - sem.available_permits());
            tokio::time::sleep(std::time::Duration::from_millis(200)).await;
            println!("[{i}] Done");
            // _permit is dropped here, releasing the semaphore
        }));
    }

    for h in handles {
        h.await.unwrap();
    }
}
```

---

## 11. Performance Considerations

### Avoid Blocking the Runtime

```rust
// BAD: Blocks the entire worker thread
async fn bad_hash(data: &[u8]) -> Vec<u8> {
    // This is CPU-intensive and blocks the async runtime
    expensive_hash_function(data)
}

// GOOD: Move CPU-intensive work to blocking thread pool
async fn good_hash(data: Vec<u8>) -> Vec<u8> {
    tokio::task::spawn_blocking(move || {
        expensive_hash_function(&data)
    }).await.unwrap()
}

fn expensive_hash_function(data: &[u8]) -> Vec<u8> {
    // Simulated CPU-intensive work
    std::thread::sleep(std::time::Duration::from_millis(100));
    data.to_vec()
}
```

### Task Size and Allocation

```rust
// BAD: Large future — stored on the heap when spawned
async fn large_task() {
    let buf = [0u8; 1_000_000];  // 1MB on the future's stack!
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;
    println!("Buffer len: {}", buf.len());
}

// GOOD: Heap-allocate large data
async fn small_task() {
    let buf = vec![0u8; 1_000_000];  // Heap allocated, future is small
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;
    println!("Buffer len: {}", buf.len());
}
```

### Reducing Contention

```rust
use std::sync::Arc;
use tokio::sync::RwLock;

// For read-heavy workloads, use RwLock instead of Mutex
struct Cache {
    data: Arc<RwLock<std::collections::HashMap<String, String>>>,
}

impl Cache {
    async fn get(&self, key: &str) -> Option<String> {
        let data = self.data.read().await;  // Multiple readers OK
        data.get(key).cloned()
    }

    async fn set(&self, key: String, value: String) {
        let mut data = self.data.write().await;  // Exclusive access
        data.insert(key, value);
    }
}
```

---

## 12. Exercises

1. **Fan-out fetcher**: Write an async function that takes a list of URLs and fetches them concurrently with a configurable concurrency limit (using a semaphore). Return results in the original order.

2. **Async pipeline**: Build a stream processing pipeline: generate numbers → filter evens → map to squares → buffer into chunks of 10 → print chunks. Use `tokio_stream`.

3. **Graceful shutdown server**: Write a simple TCP echo server that handles Ctrl+C by: (1) stopping new connections, (2) waiting for existing connections to finish (with a 5-second deadline), (3) force-closing any remaining connections.

4. **Rate limiter middleware**: Implement a Tower-compatible rate limiting layer that allows N requests per second per client IP, using a token bucket algorithm.

5. **Cancel-safe state machine**: Implement a cancel-safe message processor that reads messages from a channel and writes processed results to a file. Ensure no messages are lost even if the task is cancelled mid-operation.

---

## References

- [Tokio Tutorial](https://tokio.rs/tokio/tutorial)
- [Tokio: select!](https://tokio.rs/tokio/tutorial/select)
- [Tower documentation](https://docs.rs/tower/latest/tower/)
- [async-stream crate](https://docs.rs/async-stream/latest/async_stream/)
- [Alice Ryhl: Actors with Tokio](https://ryhl.io/blog/actors-with-tokio/)
- [Jon Gjengset: Decrusting Tokio](https://www.youtube.com/watch?v=o2ob8zkeq2s)

---

**Previous**: [Advanced Traits](./06_Advanced_Traits.md) | **Next**: [FFI and Interop](./08_FFI_and_Interop.md)
