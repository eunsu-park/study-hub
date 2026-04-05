# 15. Async and Await

**Previous**: [Concurrency](./14_Concurrency.md) | **Next**: [Modules and Cargo](./16_Modules_and_Cargo.md)

**Difficulty**: ⭐⭐⭐⭐

## Learning Objectives

- Compare async programming with OS threads and explain when async is preferable for I/O-bound workloads
- Describe the `Future` trait's poll-based execution model and the role of `Pin` in self-referential futures
- Write async functions with `async fn` / `.await` and execute them on the Tokio runtime
- Use `tokio::select!`, async channels, and async I/O to build concurrent network applications
- Identify common pitfalls such as holding locks across `.await` points and blocking inside async contexts

## Table of Contents

1. [Async vs Threading](#1-async-vs-threading)
2. [The Future Trait](#2-the-future-trait)
3. [Async Fn and Await Syntax](#3-async-fn-and-await-syntax)
4. [The Tokio Runtime](#4-the-tokio-runtime)
5. [Racing Futures with select!](#5-racing-futures-with-select)
6. [Async Channels](#6-async-channels)
7. [Async I/O](#7-async-io)
8. [Error Handling in Async](#8-error-handling-in-async)
9. [Streams: Async Iterators](#9-streams-async-iterators)
10. [Common Pitfalls](#10-common-pitfalls)
11. [Practice Problems](#11-practice-problems)
12. [References](#12-references)

---

## 1. Async vs Threading

OS threads and async tasks solve the same problem (running things concurrently) but with different trade-offs:

```
OS Threads:                          Async Tasks:
┌───────┐ ┌───────┐ ┌───────┐      ┌──────────────────────────────┐
│Thread1│ │Thread2│ │Thread3│      │        Single Thread          │
│ stack │ │ stack │ │ stack │      │  ┌──┐  ┌──┐  ┌──┐  ┌──┐     │
│ ~8MB  │ │ ~8MB  │ │ ~8MB  │      │  │T1│  │T2│  │T1│  │T3│ ... │
└───────┘ └───────┘ └───────┘      │  └──┘  └──┘  └──┘  └──┘     │
OS manages scheduling               └──────────────────────────────┘
High memory per thread               Runtime manages scheduling
Good for CPU-bound work               Low memory per task (~few KB)
                                      Great for I/O-bound work
```

| Aspect | OS Threads | Async Tasks |
|--------|-----------|-------------|
| Memory per unit | ~8 MB stack | ~few KB |
| Scalability | Hundreds to low thousands | Hundreds of thousands |
| Context switch | OS kernel (slower) | User-space (faster) |
| Best for | CPU-bound parallelism | I/O-bound concurrency (network, disk) |
| Preemption | Yes (OS can interrupt) | No (cooperative, yields at `.await`) |

**Rule of thumb**: if your tasks spend most of their time waiting (network requests, database queries, file I/O), use async. If they spend most of their time computing, use threads.

---

## 2. The Future Trait

In Rust, an async operation is represented by the `Future` trait:

```rust
use std::pin::Pin;
use std::task::{Context, Poll};

trait Future {
    type Output;
    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output>;
}

enum Poll<T> {
    Ready(T),   // the future has completed with value T
    Pending,    // not ready yet — will wake the executor when ready
}
```

Think of a `Future` as a recipe card. The recipe is not executed when you create it. Instead, an **executor** (like Tokio) repeatedly checks ("polls") the future until it completes:

```
Executor polling a future:

poll() → Pending  (not ready, I/O in progress)
  ... executor does other work ...
poll() → Pending  (still waiting)
  ... executor does other work ...
poll() → Ready(value)  (done! here's the result)
```

### Why Pin?

Some futures contain self-referential data (a field that points to another field within the same struct). Moving such a struct in memory would invalidate the internal pointer. `Pin<&mut Self>` guarantees the future will not be moved after it is first polled, making self-references safe.

You rarely need to use `Pin` directly — `async fn` and `.await` handle it automatically. Just know that `Pin` exists to make the poll-based model sound.

---

## 3. Async Fn and Await Syntax

`async fn` is syntactic sugar that transforms a function body into a state machine implementing `Future`:

```rust
// This async function...
async fn fetch_data(url: &str) -> String {
    format!("Data from {url}")
}

// ...is roughly equivalent to:
// fn fetch_data(url: &str) -> impl Future<Output = String> { ... }

// .await suspends execution until the future completes
async fn process() {
    let data = fetch_data("https://example.com").await;
    println!("Got: {data}");
}

// async blocks create anonymous futures
async fn demo() {
    let future = async {
        let x = 1 + 2;
        x * 10
    };
    let result = future.await; // drives the future to completion
    println!("result = {result}"); // 30
}
```

**Key point**: calling an `async fn` does **not** execute it. It returns a `Future`. The future only runs when polled, which is what `.await` triggers:

```rust
async fn greet() -> String {
    println!("Inside greet");
    String::from("Hello!")
}

async fn example() {
    let future = greet(); // nothing printed yet — future is just created
    println!("Future created but not started");

    let result = future.await; // NOW "Inside greet" is printed
    println!("{result}");
}
```

---

## 4. The Tokio Runtime

Rust's standard library does not include an async runtime. **Tokio** is the most widely used runtime, providing a multi-threaded task scheduler, async I/O, timers, and channels.

```rust
// Add to Cargo.toml:
// [dependencies]
// tokio = { version = "1", features = ["full"] }

use tokio::time::{sleep, Duration};

// #[tokio::main] sets up the runtime and blocks on the async main function
#[tokio::main]
async fn main() {
    println!("Starting...");

    // Spawn concurrent tasks — like thread::spawn but for async
    let task1 = tokio::spawn(async {
        sleep(Duration::from_millis(200)).await;
        println!("Task 1 complete");
        1
    });

    let task2 = tokio::spawn(async {
        sleep(Duration::from_millis(100)).await;
        println!("Task 2 complete");
        2
    });

    // Await both tasks — they run concurrently
    let (r1, r2) = (task1.await.unwrap(), task2.await.unwrap());
    println!("Results: {r1} + {r2} = {}", r1 + r2);
}
```

```
Timeline with tokio::spawn:

Task 1: ──[spawn]──────────[sleep 200ms]──[done]──►
Task 2: ──[spawn]──[sleep 100ms]──[done]──────────►
Main:   ──[spawn tasks]──[await t1]──[await t2]──[print]──►

Both tasks run concurrently on the Tokio thread pool.
Task 2 finishes first, but main awaits t1 first.
```

### Spawning vs Awaiting

- **`task.await`**: runs the future inline, one at a time
- **`tokio::spawn(task)`**: runs the future on the thread pool, truly concurrent

```rust
use tokio::time::{sleep, Duration, Instant};

#[tokio::main]
async fn main() {
    let start = Instant::now();

    // Sequential: total ~300ms
    sleep(Duration::from_millis(100)).await;
    sleep(Duration::from_millis(200)).await;
    println!("Sequential: {:?}", start.elapsed());

    let start = Instant::now();

    // Concurrent with tokio::join!: total ~200ms (max of both)
    let ((), ()) = tokio::join!(
        sleep(Duration::from_millis(100)),
        sleep(Duration::from_millis(200)),
    );
    println!("Concurrent: {:?}", start.elapsed());
}
```

---

## 5. Racing Futures with select!

`tokio::select!` waits on multiple futures and executes the branch for **whichever completes first**. The remaining futures are dropped (cancelled):

```rust
use tokio::time::{sleep, Duration};
use tokio::sync::oneshot;

#[tokio::main]
async fn main() {
    let (tx, rx) = oneshot::channel::<String>();

    // Simulate a response arriving after 150ms
    tokio::spawn(async move {
        sleep(Duration::from_millis(150)).await;
        let _ = tx.send(String::from("Response received"));
    });

    // Race the response against a 200ms timeout
    tokio::select! {
        result = rx => {
            match result {
                Ok(msg) => println!("Got message: {msg}"),
                Err(_) => println!("Sender dropped"),
            }
        }
        _ = sleep(Duration::from_millis(200)) => {
            println!("Timeout! No response within 200ms");
        }
    }
    // Output: "Got message: Response received" (150ms < 200ms)
}
```

```
select! picks the first to complete:

rx future:      ──────[150ms]──[Ready("Response")]  ← wins
timeout future: ──────[200ms]──[Ready(())]          ← cancelled

Result: the rx branch executes
```

`select!` is essential for implementing timeouts, cancellation, graceful shutdown, and multiplexing I/O sources.

---

## 6. Async Channels

Tokio provides async-aware channels that `.await` when sending or receiving:

```rust
use tokio::sync::mpsc;

#[tokio::main]
async fn main() {
    // Bounded channel: buffer holds up to 32 messages
    // Sender blocks (awaits) when buffer is full — backpressure
    let (tx, mut rx) = mpsc::channel::<String>(32);

    // Spawn 3 producer tasks
    for id in 0..3 {
        let tx = tx.clone();
        tokio::spawn(async move {
            for i in 0..3 {
                let msg = format!("Producer {id}, msg {i}");
                // send().await yields if buffer is full
                tx.send(msg).await.unwrap();
            }
        });
    }

    // Drop the original sender so the receiver knows when all are done
    drop(tx);

    // Consume messages as they arrive
    while let Some(msg) = rx.recv().await {
        println!("Received: {msg}");
    }
    println!("All producers finished");
}
```

Tokio also provides:
- **`oneshot`**: single value, single sender, single receiver
- **`broadcast`**: multiple consumers, each gets every message
- **`watch`**: single value that can be observed by many receivers (latest value only)

```rust
use tokio::sync::broadcast;

#[tokio::main]
async fn main() {
    let (tx, _) = broadcast::channel::<String>(16);

    let mut rx1 = tx.subscribe();
    let mut rx2 = tx.subscribe();

    tx.send("Hello everyone!".to_string()).unwrap();

    // Both receivers get the same message
    println!("rx1: {}", rx1.recv().await.unwrap());
    println!("rx2: {}", rx2.recv().await.unwrap());
}
```

---

## 7. Async I/O

Tokio provides async versions of standard I/O operations. These yield control while waiting for the OS, allowing other tasks to run:

```rust
use tokio::fs;
use tokio::io::{self, AsyncBufReadExt, AsyncWriteExt, BufReader};

#[tokio::main]
async fn main() -> io::Result<()> {
    // Async file write
    let mut file = fs::File::create("/tmp/async_demo.txt").await?;
    file.write_all(b"Hello from async Rust!\n").await?;
    file.write_all(b"Second line\n").await?;
    // Ensure all data is flushed to disk
    file.flush().await?;

    // Async file read — line by line
    let file = fs::File::open("/tmp/async_demo.txt").await?;
    let reader = BufReader::new(file);
    let mut lines = reader.lines();

    while let Some(line) = lines.next_line().await? {
        println!("Read: {line}");
    }

    // Read entire file at once
    let contents = fs::read_to_string("/tmp/async_demo.txt").await?;
    println!("Full contents:\n{contents}");

    // Clean up
    fs::remove_file("/tmp/async_demo.txt").await?;

    Ok(())
}
```

### Async TCP Server Example

```rust
use tokio::net::TcpListener;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let listener = TcpListener::bind("127.0.0.1:8080").await?;
    println!("Server listening on 127.0.0.1:8080");

    loop {
        // Accept a new connection
        let (mut socket, addr) = listener.accept().await?;
        println!("New connection from {addr}");

        // Spawn a task for each connection — concurrent handling
        tokio::spawn(async move {
            let mut buf = [0u8; 1024];

            loop {
                // Read data from the client
                let n = match socket.read(&mut buf).await {
                    Ok(0) => return, // connection closed
                    Ok(n) => n,
                    Err(e) => {
                        eprintln!("Read error: {e}");
                        return;
                    }
                };

                // Echo data back to the client
                if let Err(e) = socket.write_all(&buf[..n]).await {
                    eprintln!("Write error: {e}");
                    return;
                }
            }
        });
    }
}
```

```
TCP Echo Server:

Client A ──[connect]──[send "Hi"]──[recv "Hi"]──►
Client B ──[connect]──[send "Hey"]──[recv "Hey"]──►
                          │
              ┌───────────┴───────────┐
              │ Tokio Runtime         │
              │ ┌─────┐  ┌─────┐     │
              │ │TaskA│  │TaskB│     │
              │ └─────┘  └─────┘     │
              │ Each connection has   │
              │ its own spawned task  │
              └───────────────────────┘
```

---

## 8. Error Handling in Async

Async functions work seamlessly with Rust's `Result` type. Use `?` for propagation just like in synchronous code:

```rust
use std::num::ParseIntError;

#[derive(Debug)]
enum AppError {
    Io(tokio::io::Error),
    Parse(ParseIntError),
    NotFound(String),
}

impl From<tokio::io::Error> for AppError {
    fn from(e: tokio::io::Error) -> Self { AppError::Io(e) }
}

impl From<ParseIntError> for AppError {
    fn from(e: ParseIntError) -> Self { AppError::Parse(e) }
}

impl std::fmt::Display for AppError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AppError::Io(e) => write!(f, "I/O error: {e}"),
            AppError::Parse(e) => write!(f, "Parse error: {e}"),
            AppError::NotFound(key) => write!(f, "Not found: {key}"),
        }
    }
}

// Async function returning Result with custom error
async fn load_config(path: &str) -> Result<u32, AppError> {
    let contents = tokio::fs::read_to_string(path).await?; // ? converts io::Error
    let port: u32 = contents.trim().parse()?; // ? converts ParseIntError
    if port == 0 {
        return Err(AppError::NotFound("valid port".to_string()));
    }
    Ok(port)
}

#[tokio::main]
async fn main() {
    match load_config("/tmp/nonexistent.conf").await {
        Ok(port) => println!("Port: {port}"),
        Err(e) => println!("Error: {e}"),
    }
}
```

For simpler cases, use `Box<dyn std::error::Error>` or the `anyhow` crate:

```rust
// With anyhow (add: anyhow = "1" to Cargo.toml):
// use anyhow::{Context, Result};
//
// async fn load_config(path: &str) -> Result<u32> {
//     let contents = tokio::fs::read_to_string(path).await
//         .context("failed to read config file")?;
//     let port: u32 = contents.trim().parse()
//         .context("config file does not contain a valid port number")?;
//     Ok(port)
// }
```

---

## 9. Streams: Async Iterators

A `Stream` is the async counterpart of `Iterator`. While `Iterator::next()` returns `Option<T>`, a stream's `next()` returns a future that resolves to `Option<T>`:

```rust
// Add to Cargo.toml:
// tokio-stream = "0.1"

use tokio_stream::StreamExt;
use tokio::time::{interval, Duration};

#[tokio::main]
async fn main() {
    // Create a stream from an interval timer
    let mut tick_stream = tokio_stream::wrappers::IntervalStream::new(
        interval(Duration::from_millis(200))
    );

    // Take the first 5 ticks
    let mut count = 0;
    while let Some(instant) = tick_stream.next().await {
        count += 1;
        println!("Tick {count} at {instant:?}");
        if count >= 5 {
            break;
        }
    }

    // Streams support adaptor methods similar to iterators
    let numbers = tokio_stream::iter(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    let sum: i32 = numbers
        .filter(|&x| x % 2 == 0)     // keep even numbers
        .map(|x| x * x)              // square them
        .fold(0, |acc, x| acc + x)   // sum the results
        .await;
    println!("Sum of squares of even numbers: {sum}"); // 4+16+36+64+100 = 220
}
```

```
Iterator vs Stream:

Iterator:   next() → Option<T>             (synchronous, blocking)
Stream:     next() → Future<Option<T>>      (async, non-blocking)

Stream:  ──[poll]──Pending──[poll]──Ready(Some(1))──[poll]──Ready(Some(2))──...──Ready(None)
```

---

## 10. Common Pitfalls

### Pitfall 1: Holding a Mutex Lock Across an `.await`

```rust
use tokio::sync::Mutex;
use std::sync::Arc;

async fn bad_example(data: Arc<Mutex<Vec<i32>>>) {
    let mut guard = data.lock().await;
    guard.push(1);

    // BAD: holding the MutexGuard across an await point
    // The task might be suspended here, keeping the lock held
    // while other tasks try to acquire it — potential deadlock or starvation
    tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;

    guard.push(2); // lock still held during the sleep above
}

async fn good_example(data: Arc<Mutex<Vec<i32>>>) {
    // GOOD: limit the lock scope
    {
        let mut guard = data.lock().await;
        guard.push(1);
    } // lock released here, before the await

    tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;

    {
        let mut guard = data.lock().await;
        guard.push(2);
    } // lock released immediately
}

#[tokio::main]
async fn main() {
    let data = Arc::new(Mutex::new(Vec::new()));
    good_example(data).await;
}
```

### Pitfall 2: Blocking in Async Context

Synchronous blocking calls (heavy computation, blocking I/O, `std::thread::sleep`) inside an async task block the entire executor thread, starving other tasks:

```rust
#[tokio::main]
async fn main() {
    // BAD: std::thread::sleep blocks the runtime thread
    // std::thread::sleep(std::time::Duration::from_secs(5));

    // GOOD: use tokio's async sleep
    tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;

    // For CPU-intensive work, move it to a blocking thread pool
    let result = tokio::task::spawn_blocking(|| {
        // This runs on a dedicated thread, not the async executor
        let mut sum: u64 = 0;
        for i in 0..10_000_000 {
            sum += i;
        }
        sum
    }).await.unwrap();

    println!("CPU result: {result}");
}
```

### Pitfall 3: Forgetting That Futures Are Lazy

```rust
async fn important_work() -> i32 {
    println!("Doing important work");
    42
}

#[tokio::main]
async fn main() {
    // This creates the future but does NOT run it
    let future = important_work(); // nothing printed yet

    // The future must be awaited or spawned to execute
    println!("Before await");
    let result = future.await; // NOW "Doing important work" prints
    println!("Result: {result}");
}
```

### Summary of Pitfalls

```
Pitfall                        Fix
───────────────────────────────────────────────────────────
Lock held across .await        Scope the lock, drop before .await
Blocking call in async         Use spawn_blocking() or async equivalents
Forgetting futures are lazy    Always .await or tokio::spawn()
Spawning !Send futures         Use tokio::task::LocalSet for non-Send futures
```

---

## 11. Practice Problems

### Problem 1: Concurrent HTTP Fetcher

Using `reqwest` (async HTTP client) and `tokio`, write a program that takes a list of URLs, fetches them concurrently (using `tokio::spawn`), and prints each URL with its response status code and content length. Add a 5-second timeout per request using `tokio::time::timeout`.

### Problem 2: Chat Server

Build a simple TCP chat server using `tokio::net::TcpListener`. When a client connects, it sends messages as lines. The server broadcasts each message to all other connected clients. Use `tokio::sync::broadcast` for the message distribution. Handle client disconnections gracefully.

```
Client A ──"Hi"──►┐
                  │ Server ──"[A]: Hi"──► Client B
Client B ──"Hey"──►│        ──"[B]: Hey"──► Client A
```

### Problem 3: Rate Limiter

Implement an async rate limiter that allows at most N requests per second. Use `tokio::time::Interval` and `tokio::sync::Semaphore`. Spawn 50 tasks that each want to "make a request," and verify that no more than N execute within any 1-second window.

### Problem 4: Async File Processor Pipeline

Build a pipeline that:
1. Reads filenames from an async channel
2. For each file, reads its contents asynchronously
3. Processes the content (e.g., counts lines, finds a pattern)
4. Writes results to an output file

Use `tokio::sync::mpsc` channels between stages and `tokio::fs` for file operations. Process multiple files concurrently with a configurable concurrency limit (hint: use `tokio::sync::Semaphore`).

### Problem 5: Graceful Shutdown

Write a TCP echo server (similar to Section 7) that handles `Ctrl+C` gracefully. When the signal is received:
1. Stop accepting new connections
2. Wait for all active connections to finish (with a 10-second timeout)
3. Print a summary (total connections served, bytes transferred)

Use `tokio::signal::ctrl_c()` and `tokio::select!` to implement the shutdown logic.

---

## 12. References

- [The Rust Async Book](https://rust-lang.github.io/async-book/)
- [Tokio Tutorial](https://tokio.rs/tokio/tutorial)
- [Tokio API Documentation](https://docs.rs/tokio/latest/tokio/)
- [tokio-stream Documentation](https://docs.rs/tokio-stream/latest/tokio_stream/)
- [Pin and Unpin Explained](https://doc.rust-lang.org/std/pin/index.html)
- [Asynchronous Programming in Rust (Jon Gjengset)](https://www.youtube.com/watch?v=ThjvMReOXYM)

---

**Previous**: [Concurrency](./14_Concurrency.md) | **Next**: [Modules and Cargo](./16_Modules_and_Cargo.md)
