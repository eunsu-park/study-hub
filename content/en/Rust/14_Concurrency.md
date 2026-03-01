# 14. Concurrency

**Previous**: [Smart Pointers](./13_Smart_Pointers.md) | **Next**: [Async and Await](./15_Async_Await.md)

**Difficulty**: ⭐⭐⭐⭐

## Learning Objectives

- Spawn threads with `thread::spawn` and use `JoinHandle` to collect results safely
- Transfer data into threads using `move` closures and communicate between threads via `mpsc` channels
- Protect shared mutable state with `Mutex<T>` and `Arc<Mutex<T>>` and explain the trade-offs vs `RwLock<T>`
- Describe how the `Send` and `Sync` marker traits enable Rust's compile-time thread safety guarantees
- Apply scoped threads and data parallelism libraries to write safe concurrent code without lifetime issues

## Table of Contents

1. [Why Concurrency Matters](#1-why-concurrency-matters)
2. [Thread Basics](#2-thread-basics)
3. [Move Closures with Threads](#3-move-closures-with-threads)
4. [Message Passing with Channels](#4-message-passing-with-channels)
5. [Multiple Producers](#5-multiple-producers)
6. [Shared State with Mutex](#6-shared-state-with-mutex)
7. [RwLock vs Mutex](#7-rwlock-vs-mutex)
8. [Send and Sync Traits](#8-send-and-sync-traits)
9. [Avoiding Deadlocks](#9-avoiding-deadlocks)
10. [Scoped Threads](#10-scoped-threads)
11. [Data Parallelism with Rayon](#11-data-parallelism-with-rayon)
12. [Practice Problems](#12-practice-problems)
13. [References](#13-references)

---

## 1. Why Concurrency Matters

Modern CPUs have multiple cores, and programs that use only one core leave performance on the table. Concurrency lets you run tasks simultaneously, but it introduces hazards: data races, deadlocks, and race conditions.

Most languages rely on the programmer to avoid these bugs at runtime. Rust takes a different approach: the **type system** prevents data races at **compile time**. If your concurrent Rust program compiles, it is free from data races. This is Rust's "fearless concurrency" guarantee.

```
Single-threaded:             Multi-threaded:
┌──────────────┐             ┌──────────┐  ┌──────────┐
│ Task A       │             │ Task A   │  │ Task B   │
│ Task B       │             │          │  │          │
│ Task C       │             └──────────┘  └──────────┘
└──────────────┘             ┌──────────┐
Total time: A + B + C        │ Task C   │
                             └──────────┘
                             Total time: max(A+C, B) (overlapping)
```

---

## 2. Thread Basics

`std::thread::spawn` creates a new OS thread. It returns a `JoinHandle` that you can use to wait for the thread to finish and retrieve its return value:

```rust
use std::thread;
use std::time::Duration;

fn main() {
    // spawn() takes a closure and runs it in a new thread
    let handle = thread::spawn(|| {
        for i in 1..=5 {
            println!("  spawned thread: count {i}");
            thread::sleep(Duration::from_millis(100));
        }
        42 // return value from the thread
    });

    // Main thread continues running concurrently
    for i in 1..=3 {
        println!("main thread: count {i}");
        thread::sleep(Duration::from_millis(150));
    }

    // join() blocks until the spawned thread finishes
    // It returns Result<T, Box<dyn Any>> — Ok(T) or Err if thread panicked
    let result = handle.join().unwrap();
    println!("Spawned thread returned: {result}");
}
```

```
Execution timeline:
Main:     ──[1]────[2]────[3]────[join/wait]──►
Spawned:  ──[1]──[2]──[3]──[4]──[5]──────────►
                                    ▲
                                    └─ main resumes after join
```

---

## 3. Move Closures with Threads

Spawned threads may outlive the scope that created them. Therefore, closures passed to `thread::spawn` must own all the data they use. The `move` keyword transfers ownership:

```rust
use std::thread;

fn main() {
    let name = String::from("Alice");
    let age = 30; // Copy type — gets copied into the closure

    let handle = thread::spawn(move || {
        // `name` is moved (String doesn't implement Copy)
        // `age` is copied (i32 implements Copy)
        println!("{name} is {age} years old");
    });

    // println!("{name}"); // ERROR: `name` was moved into the thread
    println!("age is still accessible: {age}"); // OK: i32 was copied

    handle.join().unwrap();
}
```

Without `move`, the compiler rejects the code because it cannot guarantee the borrowed data will live long enough:

```
error[E0373]: closure may outlive the current function, but it borrows `name`
  --> help: to force the closure to take ownership, use the `move` keyword
```

---

## 4. Message Passing with Channels

Channels implement the "share by communicating" philosophy. Rust provides `mpsc` (multiple producer, single consumer) channels:

```
Producer (tx)           Channel              Consumer (rx)
┌──────────┐       ┌──────────────┐       ┌──────────┐
│ thread 1 │──tx──►│  [ A, B, C ] │──rx──►│  main    │
└──────────┘       └──────────────┘       └──────────┘
   send(A)            buffered queue         recv() → A
   send(B)                                   recv() → B
   send(C)                                   recv() → C
```

```rust
use std::sync::mpsc;
use std::thread;
use std::time::Duration;

fn main() {
    // Create a channel: tx (transmitter/sender), rx (receiver)
    let (tx, rx) = mpsc::channel();

    thread::spawn(move || {
        let messages = vec![
            String::from("hello"),
            String::from("from"),
            String::from("the"),
            String::from("thread"),
        ];

        for msg in messages {
            tx.send(msg).unwrap(); // send() transfers ownership of `msg`
            thread::sleep(Duration::from_millis(200));
        }
        // tx is dropped here — signals end of stream
    });

    // recv() blocks until a message arrives or the channel closes
    // Using rx as an iterator automatically calls recv() in a loop
    for received in rx {
        println!("Got: {received}");
    }
    // Loop ends when all senders are dropped
}
```

**Important**: `send()` transfers ownership. After sending a value, the sender can no longer use it. This prevents data races by design.

---

## 5. Multiple Producers

Clone the sender to allow multiple threads to send messages into the same channel:

```rust
use std::sync::mpsc;
use std::thread;
use std::time::Duration;

fn main() {
    let (tx, rx) = mpsc::channel();

    // Clone the sender for each producer thread
    for id in 0..3 {
        let tx_clone = tx.clone();
        thread::spawn(move || {
            for i in 0..3 {
                let msg = format!("Thread {id}, message {i}");
                tx_clone.send(msg).unwrap();
                thread::sleep(Duration::from_millis(100));
            }
        });
    }

    // Drop the original sender so rx knows when all producers are done
    drop(tx);

    // Collect all messages
    for msg in rx {
        println!("{msg}");
    }
    println!("All producers finished");
}
```

```
Thread 0 ──tx0──►┐
Thread 1 ──tx1──►├──[ channel ]──rx──► main
Thread 2 ──tx2──►┘
                  Messages arrive in send order (per-thread),
                  but interleaved between threads
```

---

## 6. Shared State with Mutex

When multiple threads need to read and write the same data, use a `Mutex` (mutual exclusion lock). `Mutex<T>` provides interior mutability by requiring callers to `lock()` before accessing the data:

```rust
use std::sync::{Arc, Mutex};
use std::thread;

fn main() {
    // Arc allows shared ownership across threads
    // Mutex provides safe mutable access
    let counter = Arc::new(Mutex::new(0));

    let mut handles = vec![];

    for _ in 0..10 {
        let counter = Arc::clone(&counter);
        let handle = thread::spawn(move || {
            // lock() returns a MutexGuard — unlocks automatically when dropped
            let mut num = counter.lock().unwrap();
            *num += 1;
            // MutexGuard dropped here → lock released
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.join().unwrap();
    }

    println!("Final counter: {}", *counter.lock().unwrap()); // 10
}
```

```
Thread 1: ──[lock]──[*num += 1]──[unlock]──────────────────►
Thread 2: ──────────[wait...]──[lock]──[*num += 1]──[unlock]►
Thread 3: ──────────────────────────────[wait..]──[lock]──..►

Mutex ensures only one thread accesses data at a time
```

**Why `Arc` and not `Rc`?** `Rc` is not thread-safe (it uses non-atomic reference counting). `Arc` uses atomic operations, which adds a small overhead but guarantees correctness across threads.

---

## 7. RwLock vs Mutex

`RwLock<T>` allows **multiple simultaneous readers** OR **one writer**, which can improve throughput for read-heavy workloads:

```rust
use std::sync::{Arc, RwLock};
use std::thread;

fn main() {
    let config = Arc::new(RwLock::new(String::from("initial")));

    let mut handles = vec![];

    // Spawn 5 reader threads
    for i in 0..5 {
        let config = Arc::clone(&config);
        handles.push(thread::spawn(move || {
            // read() allows multiple concurrent readers
            let val = config.read().unwrap();
            println!("Reader {i}: {val}");
        }));
    }

    // Spawn 1 writer thread
    {
        let config = Arc::clone(&config);
        handles.push(thread::spawn(move || {
            // write() requires exclusive access
            let mut val = config.write().unwrap();
            *val = String::from("updated");
            println!("Writer: updated config");
        }));
    }

    for handle in handles {
        handle.join().unwrap();
    }

    println!("Final: {}", config.read().unwrap());
}
```

| Feature | `Mutex<T>` | `RwLock<T>` |
|---------|------------|-------------|
| Concurrent reads | No | Yes |
| Write access | Exclusive | Exclusive |
| Performance (read-heavy) | Lower (serialized) | Higher (parallel reads) |
| Performance (write-heavy) | Similar | Slightly worse (overhead) |
| Risk of writer starvation | No | Possible |

**Guideline**: use `Mutex` by default. Switch to `RwLock` only when profiling shows contention from serialized reads.

---

## 8. Send and Sync Traits

Rust guarantees thread safety through two **marker traits** (they have no methods):

- **`Send`**: a type can be **transferred** to another thread (ownership can move across thread boundaries)
- **`Sync`**: a type can be **referenced** from multiple threads (`&T` is safe to share)

```
Send: "I can be moved to another thread"
  - Most types are Send
  - NOT Send: Rc<T> (non-atomic ref count), raw pointers

Sync: "Multiple threads can hold &T simultaneously"
  - A type T is Sync if &T is Send
  - NOT Sync: RefCell<T> (runtime borrow checking is not thread-safe),
              Cell<T>, Rc<T>
```

```rust
use std::rc::Rc;
use std::sync::Arc;

fn must_be_send<T: Send>(_val: T) {}
fn must_be_sync<T: Sync>(_val: &T) {}

fn main() {
    let arc = Arc::new(42);
    must_be_send(arc.clone()); // Arc<i32> is Send
    must_be_sync(&arc);        // Arc<i32> is Sync

    // let rc = Rc::new(42);
    // must_be_send(rc);       // ERROR: Rc<i32> is not Send
    // must_be_sync(&rc);      // ERROR: Rc<i32> is not Sync

    println!("Arc is both Send and Sync!");
}
```

These traits are automatically implemented by the compiler. If all fields of a struct are `Send`, the struct is `Send`. This composability is why Rust can prevent data races at compile time without runtime overhead.

---

## 9. Avoiding Deadlocks

A **deadlock** occurs when two or more threads each hold a lock and wait for the other's lock, creating a circular dependency:

```
DEADLOCK:
Thread A: holds Lock 1, waiting for Lock 2
Thread B: holds Lock 2, waiting for Lock 1

Thread A ──[lock1]──────────[wait lock2...]──► STUCK
Thread B ──────────[lock2]──[wait lock1...]──► STUCK
```

Prevention strategies:

```rust
use std::sync::{Arc, Mutex};
use std::thread;

fn main() {
    let resource_a = Arc::new(Mutex::new("Resource A"));
    let resource_b = Arc::new(Mutex::new("Resource B"));

    // STRATEGY 1: Always acquire locks in the same global order
    // Both threads lock A first, then B — no circular dependency

    let (ra, rb) = (Arc::clone(&resource_a), Arc::clone(&resource_b));
    let t1 = thread::spawn(move || {
        let _a = ra.lock().unwrap(); // always lock A first
        let _b = rb.lock().unwrap(); // then lock B
        println!("Thread 1: got both locks");
    });

    let (ra, rb) = (Arc::clone(&resource_a), Arc::clone(&resource_b));
    let t2 = thread::spawn(move || {
        let _a = ra.lock().unwrap(); // same order: A first
        let _b = rb.lock().unwrap(); // then B
        println!("Thread 2: got both locks");
    });

    t1.join().unwrap();
    t2.join().unwrap();

    // STRATEGY 2: Use try_lock() to avoid blocking
    let ra = Arc::clone(&resource_a);
    let lock_result = ra.try_lock();
    match lock_result {
        Ok(guard) => println!("Got lock: {}", *guard),
        Err(_) => println!("Could not acquire lock, doing something else"),
    }

    // STRATEGY 3: Minimize lock scope — hold locks for the shortest time
    {
        let data = resource_a.lock().unwrap();
        let result = data.len(); // do minimal work under lock
        drop(data); // release lock explicitly before doing more work
        println!("Length was: {result}");
    }
}
```

---

## 10. Scoped Threads

`std::thread::scope` (stabilized in Rust 1.63) lets you spawn threads that can borrow local variables, because the scope guarantees all threads finish before it returns:

```rust
use std::thread;

fn main() {
    let mut data = vec![1, 2, 3, 4, 5, 6, 7, 8];
    let chunk_size = data.len() / 2;

    // Scoped threads can borrow `data` without `move`
    thread::scope(|s| {
        let (left, right) = data.split_at_mut(chunk_size);

        // Thread borrows `left` — guaranteed to finish before scope ends
        s.spawn(|| {
            for val in left.iter_mut() {
                *val *= 2;
            }
            println!("Left doubled: {left:?}");
        });

        // Thread borrows `right` — no data race because slices don't overlap
        s.spawn(|| {
            for val in right.iter_mut() {
                *val *= 3;
            }
            println!("Right tripled: {right:?}");
        });
    }); // all scoped threads are joined here automatically

    println!("Final data: {data:?}");
    // [2, 4, 6, 8, 15, 18, 21, 24]
}
```

```
Scope boundary:
┌──────────────────────────────────┐
│  s.spawn(|| { ... left ... });   │
│  s.spawn(|| { ... right ... });  │
│                                  │
│  ← all threads joined here      │
└──────────────────────────────────┘
data is accessible again after the scope
```

Scoped threads are ideal when you want to parallelize work without the overhead of `Arc` and channels.

---

## 11. Data Parallelism with Rayon

The `rayon` crate provides effortless data parallelism through parallel iterators. It manages a thread pool and distributes work automatically:

```rust
// Add to Cargo.toml: rayon = "1.10"

use rayon::prelude::*;

fn is_prime(n: u64) -> bool {
    if n < 2 { return false; }
    if n == 2 { return true; }
    if n % 2 == 0 { return false; }
    let mut i = 3;
    while i * i <= n {
        if n % i == 0 { return false; }
        i += 2;
    }
    true
}

fn main() {
    let numbers: Vec<u64> = (1..100_000).collect();

    // Sequential
    let count_seq = numbers.iter().filter(|&&n| is_prime(n)).count();

    // Parallel — just change iter() to par_iter()
    let count_par = numbers.par_iter().filter(|&&n| is_prime(n)).count();

    assert_eq!(count_seq, count_par);
    println!("Primes below 100,000: {count_par}");

    // Parallel sort
    let mut data = vec![5, 3, 8, 1, 9, 2, 7, 4, 6];
    data.par_sort();
    println!("Sorted: {data:?}");

    // Parallel map-reduce
    let sum_of_squares: u64 = (1..=1_000_000u64)
        .into_par_iter()
        .map(|n| n * n)
        .sum();
    println!("Sum of squares 1..=1M: {sum_of_squares}");
}
```

Rayon uses **work stealing**: idle threads take tasks from busy threads' queues, balancing the load automatically. For most data-parallel workloads, Rayon is simpler and often faster than manual thread management.

---

## 12. Practice Problems

### Problem 1: Parallel Word Counter

Write a program that reads multiple text files concurrently (one thread per file), counts the words in each file, and uses a channel to send results back to the main thread. Print the word count per file and the total.

### Problem 2: Producer-Consumer Pipeline

Build a three-stage pipeline using channels:
1. **Stage 1**: Generate numbers 1..=100
2. **Stage 2**: Filter out non-primes
3. **Stage 3**: Collect and print the results

Each stage runs in its own thread. Use `mpsc::channel` to connect them.

```
[Generator] ──ch1──► [Filter] ──ch2──► [Collector]
```

### Problem 3: Thread-Safe Cache

Implement a concurrent cache using `Arc<RwLock<HashMap<K, V>>>`. Support `get()` (read lock) and `insert()` (write lock) operations. Spawn multiple reader and writer threads and verify correctness. Compare performance with an `Arc<Mutex<HashMap>>` version.

### Problem 4: Dining Philosophers

Implement the classic Dining Philosophers problem with 5 philosophers and 5 forks. Use `Arc<Mutex<()>>` for each fork. Implement a deadlock-free solution using ordered lock acquisition. Print when each philosopher is thinking, eating, and done.

### Problem 5: Parallel Matrix Multiplication

Given two matrices (represented as `Vec<Vec<f64>>`), compute their product using scoped threads. Divide the result matrix rows among available threads. Compare the execution time with a sequential implementation for large matrices (e.g., 500x500).

---

## 13. References

- [The Rust Programming Language, Ch. 16: Fearless Concurrency](https://doc.rust-lang.org/book/ch16-00-concurrency.html)
- [std::thread Module Documentation](https://doc.rust-lang.org/std/thread/index.html)
- [std::sync Module Documentation](https://doc.rust-lang.org/std/sync/index.html)
- [Rayon: Data Parallelism in Rust](https://docs.rs/rayon/latest/rayon/)
- [Rust Atomics and Locks (Mara Bos)](https://marabos.nl/atomics/)

---

**Previous**: [Smart Pointers](./13_Smart_Pointers.md) | **Next**: [Async and Await](./15_Async_Await.md)
