# Concurrency & Parallelism

**Previous**: [Debugging & Profiling](./11_Debugging_and_Profiling.md) | **Next**: [API Design Principles](./13_API_Design.md)

> **Topic**: Programming
> **Lesson**: 12 of 16
> **Prerequisites**: Functions and Methods, Error Handling, Debugging and Profiling

## Learning Objectives

After completing this lesson, you will be able to:

1. Distinguish between concurrency (structuring programs to handle multiple tasks) and parallelism (executing multiple tasks simultaneously on multiple cores)
2. Compare processes, threads, and green threads/goroutines in terms of memory, overhead, and communication
3. Identify and fix race conditions using mutexes, locks, and semaphores across Python, Java, and C++
4. Diagnose deadlock scenarios using the four necessary conditions and apply prevention strategies such as lock ordering
5. Implement asynchronous I/O using async/await in Python, JavaScript, Rust, and C#
6. Apply message-passing concurrency patterns including channels (Go, Rust) and the actor model (Erlang)
7. Implement parallel computation patterns: map-reduce, fork-join, pipeline, and thread pool
8. Use immutability and lock-free data structures as strategies for safe concurrent programming

---

## Introduction

Modern applications must handle many tasks simultaneously: responsive user interfaces, network I/O, database queries, background processing. Concurrency and parallelism are essential tools for building such systems.

However, concurrent programming is notoriously difficult. Race conditions, deadlocks, and data corruption are common pitfalls. This lesson gives you the mental models, patterns, and practical techniques to write correct concurrent code.

---

## Concurrency vs Parallelism

> **Analogy: Highway Lanes** -- Think of concurrency as a single-lane road where cars (tasks) take turns at intersections; one car moves while another waits at a red light. Parallelism is a multi-lane highway where cars travel simultaneously in separate lanes. Adding lanes (CPU cores) helps parallel workloads move faster, but a traffic cop (scheduler) is still needed to prevent collisions at merge points.

These terms are often confused, but they represent different concepts:

### Concurrency: Dealing with Many Things at Once

**Concurrency is about structure** – how you organize your program to handle multiple tasks.

**Example:** A single chef (one CPU core) preparing multiple dishes:
```
Chef switches between tasks:
1. Chop vegetables (pause to let water boil)
2. Stir sauce (pause while pasta cooks)
3. Plate first dish (pause while second dish cooks)

One chef, many tasks, context switching between them
```

**In code:**
```python
# Concurrent: Single thread handles multiple tasks by switching
async def make_coffee():
    print("Grinding beans...")
    await asyncio.sleep(2)  # Wait for grinding (yield control)
    print("Brewing...")
    await asyncio.sleep(3)  # Wait for brewing (yield control)
    return "Coffee ready"

async def make_toast():
    print("Toasting bread...")
    await asyncio.sleep(3)  # Wait for toasting (yield control)
    return "Toast ready"

# Run concurrently: single thread switches between tasks during waits
await asyncio.gather(make_coffee(), make_toast())
```

### Parallelism: Doing Many Things at Once

**Parallelism is about execution** – actually running multiple computations simultaneously on multiple CPU cores.

**Example:** Multiple chefs (multiple CPU cores) preparing dishes simultaneously:
```
Chef 1: Chops vegetables
Chef 2: Stirs sauce          } All at the same time
Chef 3: Plates dishes
```

**In code:**
```python
# Parallel: Multiple processes run on multiple CPU cores
from multiprocessing import Pool

def expensive_computation(n):
    return sum(i * i for i in range(n))

# Run in parallel: multiple CPU cores work simultaneously
with Pool(4) as pool:
    results = pool.map(expensive_computation, [10**7, 10**7, 10**7, 10**7])
```

### Rob Pike: "Concurrency is Not Parallelism"

[Rob Pike's famous talk](https://go.dev/blog/waza-talk) explains:
- **Concurrency:** A way to structure your program (design)
- **Parallelism:** Simultaneous execution (runtime)

You can have:
- **Concurrency without parallelism:** Single core, context switching
- **Parallelism without concurrency:** SIMD operations (same instruction, multiple data)
- **Both:** Multi-threaded program on multi-core CPU

---

## Why Concurrency?

### 1. Responsive UIs

Without concurrency, long-running operations freeze the UI:

```javascript
// BAD: Blocks UI thread
button.addEventListener('click', () => {
    const result = expensiveComputation();  // UI freezes!
    displayResult(result);
});

// GOOD: Offload to background
button.addEventListener('click', async () => {
    const result = await runInBackground(expensiveComputation);  // UI stays responsive
    displayResult(result);
});
```

### 2. Efficient I/O

While waiting for I/O (network, disk, database), the CPU can do other work:

```python
# Sequential: Waits for each request (slow)
def fetch_all(urls):
    results = []
    for url in urls:
        results.append(fetch(url))  # Wait for response
    return results
# Total time: sum of all requests

# Concurrent: Overlaps I/O waits (fast)
async def fetch_all(urls):
    tasks = [fetch(url) for url in urls]
    return await asyncio.gather(*tasks)  # All requests in parallel
# Total time: max of all requests (not sum!)
```

### 3. Utilizing Multi-Core CPUs

Modern CPUs have multiple cores. Sequential code uses only one core:

```python
# Uses 1 core
def process_data(data):
    return [expensive_function(item) for item in data]

# Uses all cores
from multiprocessing import Pool
def process_data_parallel(data):
    with Pool() as pool:
        return pool.map(expensive_function, data)
```

---

## Processes vs Threads

### Process

- **Independent memory space**: Each process has its own memory
- **Heavier**: Creating/destroying is expensive
- **Safer**: Crash in one process doesn't affect others
- **Communication**: Must use IPC (pipes, sockets, shared memory)

**Example: Python multiprocessing**
```python
from multiprocessing import Process

def worker(name):
    print(f"Worker {name} starting")
    # Do work
    print(f"Worker {name} done")

if __name__ == "__main__":
    p1 = Process(target=worker, args=("A",))
    p2 = Process(target=worker, args=("B",))

    p1.start()
    p2.start()

    p1.join()  # Wait for completion
    p2.join()
```

### Thread

- **Shared memory space**: All threads see the same memory
- **Lighter**: Creating/destroying is cheap
- **Dangerous**: Shared state requires synchronization
- **Communication**: Direct memory access (but requires locking)

**Example: Python threading**
```python
from threading import Thread

counter = 0  # Shared between threads!

def worker(name):
    global counter
    print(f"Worker {name} starting")
    # Do work
    counter += 1  # DANGER: Race condition!
    print(f"Worker {name} done")

t1 = Thread(target=worker, args=("A",))
t2 = Thread(target=worker, args=("B",))

t1.start()
t2.start()

t1.join()
t2.join()

print(f"Counter: {counter}")  # May be 1 instead of 2!
```

**Java threads:**
```java
class Worker extends Thread {
    private String name;

    public Worker(String name) {
        this.name = name;
    }

    @Override
    public void run() {
        System.out.println("Worker " + name + " starting");
        // Do work
        System.out.println("Worker " + name + " done");
    }
}

// Usage
Worker w1 = new Worker("A");
Worker w2 = new Worker("B");
w1.start();
w2.start();
w1.join();
w2.join();
```

**C++ threads (C++11):**
```cpp
#include <iostream>
#include <thread>

void worker(std::string name) {
    std::cout << "Worker " << name << " starting\n";
    // Do work
    std::cout << "Worker " << name << " done\n";
}

int main() {
    std::thread t1(worker, "A");
    std::thread t2(worker, "B");

    t1.join();
    t2.join();

    return 0;
}
```

### Green Threads / Goroutines / Virtual Threads

Some languages provide lightweight threads scheduled by the runtime, not the OS:

- **Go:** Goroutines (thousands of goroutines on a few OS threads)
- **Erlang:** Processes (millions of lightweight processes)
- **Java 21+:** Virtual threads (lightweight threads)

**Go example:**
```go
package main

import (
    "fmt"
    "time"
)

func worker(name string) {
    fmt.Printf("Worker %s starting\n", name)
    time.Sleep(1 * time.Second)
    fmt.Printf("Worker %s done\n", name)
}

func main() {
    go worker("A")  // Launch goroutine
    go worker("B")  // Launch goroutine

    time.Sleep(2 * time.Second)  // Wait for goroutines
}
```

---

## Thread-Based Concurrency

### Shared State Problems

**Race condition:** Multiple threads access shared data without synchronization, leading to unpredictable results.

**Example:**
```python
counter = 0

def increment():
    global counter
    for _ in range(100000):
        counter += 1  # Three operations: read, increment, write

# Run two threads
from threading import Thread
t1 = Thread(target=increment)
t2 = Thread(target=increment)
t1.start()
t2.start()
t1.join()
t2.join()

print(f"Counter: {counter}")  # Expected: 200000, Actual: varies (e.g., 153421)
```

**Why?**
```
Thread 1: read counter (0)
Thread 2: read counter (0)
Thread 1: increment (0 + 1 = 1)
Thread 2: increment (0 + 1 = 1)
Thread 1: write counter (1)
Thread 2: write counter (1)  # Overwrites Thread 1's write!
# Both increments happened, but counter is only 1
```

### Synchronization: Mutexes/Locks

**Mutex (Mutual Exclusion):** Only one thread can hold the lock at a time.

**Python:**
```python
from threading import Thread, Lock

counter = 0
lock = Lock()

def increment():
    global counter
    for _ in range(100000):
        with lock:  # Acquire lock
            counter += 1
        # Lock released

t1 = Thread(target=increment)
t2 = Thread(target=increment)
t1.start()
t2.start()
t1.join()
t2.join()

print(f"Counter: {counter}")  # Always 200000
```

**Java:**
```java
class Counter {
    private int count = 0;
    private final Object lock = new Object();

    public void increment() {
        synchronized(lock) {  // Acquire lock
            count++;
        }  // Release lock
    }

    public int getCount() {
        synchronized(lock) {
            return count;
        }
    }
}
```

**C++:**
```cpp
#include <mutex>

std::mutex mtx;
int counter = 0;

void increment() {
    for (int i = 0; i < 100000; i++) {
        std::lock_guard<std::mutex> lock(mtx);  // RAII: acquires lock
        counter++;
    }  // Lock released when lock_guard goes out of scope
}
```

### Semaphores

**Semaphore:** Allows N threads to access a resource simultaneously.

```python
from threading import Semaphore

# Only 3 threads can access the resource at once
semaphore = Semaphore(3)

def worker(name):
    semaphore.acquire()  # Wait if 3 threads are already inside
    print(f"{name}: Accessing resource")
    time.sleep(1)
    print(f"{name}: Done")
    semaphore.release()  # Allow another thread to enter

# Launch 10 threads, but only 3 run concurrently
threads = [Thread(target=worker, args=(f"T{i}",)) for i in range(10)]
for t in threads:
    t.start()
for t in threads:
    t.join()
```

### Deadlock

**Deadlock:** Two or more threads wait for each other, and none can proceed.

**Example:**
```python
lock1 = Lock()
lock2 = Lock()

def thread1():
    with lock1:
        time.sleep(0.1)  # Give thread2 time to acquire lock2
        with lock2:
            print("Thread 1 got both locks")

def thread2():
    with lock2:
        time.sleep(0.1)  # Give thread1 time to acquire lock1
        with lock1:
            print("Thread 2 got both locks")

# Deadlock!
# Thread 1: holds lock1, waits for lock2
# Thread 2: holds lock2, waits for lock1
# Neither can proceed
```

**Four conditions for deadlock (all must be true):**
1. **Mutual exclusion:** Resource can't be shared
2. **Hold and wait:** Thread holds resources while waiting for others
3. **No preemption:** Resources can't be forcibly taken
4. **Circular wait:** T1 waits for T2, T2 waits for T1

**Prevention strategies:**
- **Lock ordering:** Always acquire locks in the same order
- **Timeouts:** Use `try_lock` with timeout
- **Avoid holding multiple locks:** Redesign to need only one lock at a time

**Fix with lock ordering:**
```python
def thread1():
    with lock1:  # Acquire lock1 first
        with lock2:  # Then lock2
            print("Thread 1 got both locks")

def thread2():
    with lock1:  # Acquire lock1 first (same order!)
        with lock2:  # Then lock2
            print("Thread 2 got both locks")
```

### Producer-Consumer Problem

**Problem:** Producers generate data, consumers process it. Need thread-safe queue.

**Python:**
```python
from threading import Thread
from queue import Queue
import time

queue = Queue(maxsize=10)

def producer(name):
    for i in range(5):
        item = f"{name}-{i}"
        queue.put(item)  # Thread-safe: blocks if queue is full
        print(f"{name} produced {item}")
        time.sleep(0.1)

def consumer(name):
    while True:
        item = queue.get()  # Thread-safe: blocks if queue is empty
        if item is None:  # Poison pill to stop
            break
        print(f"{name} consumed {item}")
        time.sleep(0.2)
        queue.task_done()

# Start producers and consumers
producers = [Thread(target=producer, args=(f"P{i}",)) for i in range(2)]
consumers = [Thread(target=consumer, args=(f"C{i}",)) for i in range(3)]

for p in producers:
    p.start()
for c in consumers:
    c.start()

# Wait for producers
for p in producers:
    p.join()

# Send poison pills to stop consumers
for _ in consumers:
    queue.put(None)

# Wait for consumers
for c in consumers:
    c.join()
```

### Reader-Writer Problem

**Problem:** Multiple readers can read simultaneously, but writers need exclusive access.

**Python (using threading.RLock):**
```python
from threading import Thread, RLock

class ReadWriteLock:
    def __init__(self):
        self.readers = 0
        self.lock = RLock()
        self.write_lock = RLock()

    def acquire_read(self):
        with self.lock:
            self.readers += 1
            if self.readers == 1:
                self.write_lock.acquire()  # First reader blocks writers

    def release_read(self):
        with self.lock:
            self.readers -= 1
            if self.readers == 0:
                self.write_lock.release()  # Last reader unblocks writers

    def acquire_write(self):
        self.write_lock.acquire()

    def release_write(self):
        self.write_lock.release()

# Usage
data = 0
rw_lock = ReadWriteLock()

def reader(name):
    rw_lock.acquire_read()
    print(f"{name} reading: {data}")
    time.sleep(0.1)
    rw_lock.release_read()

def writer(name, value):
    rw_lock.acquire_write()
    global data
    print(f"{name} writing: {value}")
    data = value
    time.sleep(0.1)
    rw_lock.release_write()
```

---

## Async/Await Pattern

Async/await provides concurrency without threads: a single thread handles multiple tasks by yielding during I/O waits.

### Event Loop: Single-Threaded Concurrency

**Event loop:** Runs one task at a time, but switches between tasks when they're waiting.

```python
# Conceptual event loop
tasks = [task1(), task2(), task3()]
while tasks:
    for task in tasks:
        if task.is_waiting():
            continue  # Skip this task, it's waiting for I/O
        task.run_until_wait()  # Run until it waits again
        if task.is_done():
            tasks.remove(task)
```

### Promises/Futures: Representing Eventual Values

**Promise (JavaScript) / Future (Python):** Represents a value that will be available in the future.

**States:**
- **Pending:** Not yet resolved
- **Fulfilled:** Successfully resolved with a value
- **Rejected:** Failed with an error

### Async/Await Syntax

**Python:**
```python
import asyncio

async def fetch_data(url):
    print(f"Fetching {url}...")
    await asyncio.sleep(2)  # Simulate network I/O (yields control)
    print(f"Fetched {url}")
    return f"Data from {url}"

async def main():
    # Sequential: 6 seconds
    data1 = await fetch_data("http://example.com/1")
    data2 = await fetch_data("http://example.com/2")
    data3 = await fetch_data("http://example.com/3")

    # Concurrent: 2 seconds (all run together)
    results = await asyncio.gather(
        fetch_data("http://example.com/1"),
        fetch_data("http://example.com/2"),
        fetch_data("http://example.com/3")
    )
    print(results)

asyncio.run(main())
```

**JavaScript:**
```javascript
async function fetchData(url) {
    console.log(`Fetching ${url}...`);
    await new Promise(resolve => setTimeout(resolve, 2000));  // Simulate delay
    console.log(`Fetched ${url}`);
    return `Data from ${url}`;
}

async function main() {
    // Sequential: 6 seconds
    const data1 = await fetchData('http://example.com/1');
    const data2 = await fetchData('http://example.com/2');
    const data3 = await fetchData('http://example.com/3');

    // Concurrent: 2 seconds
    const results = await Promise.all([
        fetchData('http://example.com/1'),
        fetchData('http://example.com/2'),
        fetchData('http://example.com/3')
    ]);
    console.log(results);
}

main();
```

**Rust:**
```rust
use tokio::time::{sleep, Duration};

async fn fetch_data(url: &str) -> String {
    println!("Fetching {}...", url);
    sleep(Duration::from_secs(2)).await;  // Yields control
    println!("Fetched {}", url);
    format!("Data from {}", url)
}

#[tokio::main]
async fn main() {
    // Concurrent
    let (data1, data2, data3) = tokio::join!(
        fetch_data("http://example.com/1"),
        fetch_data("http://example.com/2"),
        fetch_data("http://example.com/3")
    );

    println!("{}, {}, {}", data1, data2, data3);
}
```

**C# (.NET):**
```csharp
using System;
using System.Threading.Tasks;

async Task<string> FetchData(string url) {
    Console.WriteLine($"Fetching {url}...");
    await Task.Delay(2000);  // Simulate delay
    Console.WriteLine($"Fetched {url}");
    return $"Data from {url}";
}

async Task Main() {
    // Concurrent
    var results = await Task.WhenAll(
        FetchData("http://example.com/1"),
        FetchData("http://example.com/2"),
        FetchData("http://example.com/3")
    );

    foreach (var result in results) {
        Console.WriteLine(result);
    }
}
```

### When to Use Async vs Threads

**Use async/await when:**
- I/O-bound tasks (network, disk, database)
- You need to handle many concurrent operations (thousands of connections)
- You want to avoid threading overhead

**Use threads when:**
- CPU-bound tasks (computation-heavy)
- You need true parallelism
- You're interfacing with thread-based APIs

**Example: I/O-bound (async wins)**
```python
# Fetching 1000 URLs
# Threads: 1000 threads = high memory, context-switching overhead
# Async: Single thread, 1000 concurrent tasks = low overhead
```

**Example: CPU-bound (threads/processes win)**
```python
# Computing 1000 expensive calculations
# Async: Single core, sequential execution
# Threads/processes: Multiple cores, parallel execution
```

---

## Message Passing

Instead of sharing memory (and dealing with locks), threads/processes communicate by sending messages.

### Channels (Go, Rust)

**Go:**
```go
package main

import "fmt"

func producer(ch chan int) {
    for i := 0; i < 5; i++ {
        ch <- i  // Send to channel
    }
    close(ch)  // Signal no more data
}

func consumer(ch chan int) {
    for value := range ch {  // Receive from channel
        fmt.Println("Received:", value)
    }
}

func main() {
    ch := make(chan int)
    go producer(ch)
    consumer(ch)
}
```

**Rust:**
```rust
use std::thread;
use std::sync::mpsc;  // Multiple Producer, Single Consumer

fn main() {
    let (tx, rx) = mpsc::channel();

    thread::spawn(move || {
        for i in 0..5 {
            tx.send(i).unwrap();  // Send to channel
        }
    });

    for received in rx {  // Receive from channel
        println!("Received: {}", received);
    }
}
```

**Benefits:**
- No shared state → no locks needed
- Clear ownership (in Rust, channel owns the data)
- Easier to reason about

### Actor Model (Erlang, Akka)

**Actor:** Isolated entity that:
- Has private state
- Communicates only via messages
- Processes messages sequentially

**Erlang example:**
```erlang
% Define an actor
counter_actor() ->
    counter_loop(0).

counter_loop(Count) ->
    receive
        {increment, From} ->
            NewCount = Count + 1,
            From ! {count, NewCount},
            counter_loop(NewCount);
        {get, From} ->
            From ! {count, Count},
            counter_loop(Count)
    end.

% Start the actor
Pid = spawn(fun counter_actor/0),

% Send messages
Pid ! {increment, self()},
Pid ! {get, self()}.
```

### CSP (Communicating Sequential Processes)

**Go's concurrency model:** Goroutines communicate via channels (CSP).

**Slogan:** "Don't communicate by sharing memory; share memory by communicating."

---

## Parallel Patterns

### Map-Reduce

**Map:** Apply function to each element (in parallel)
**Reduce:** Aggregate results

```python
from multiprocessing import Pool

def square(x):
    return x * x

def sum_squares(numbers):
    with Pool() as pool:
        # Map: square each number in parallel
        squared = pool.map(square, numbers)

    # Reduce: sum the results
    return sum(squared)

numbers = list(range(1000000))
result = sum_squares(numbers)
```

### Fork-Join

**Fork:** Divide task into subtasks
**Execute:** Run subtasks in parallel
**Join:** Combine results

```python
def merge_sort_parallel(arr):
    if len(arr) <= 1:
        return arr

    mid = len(arr) // 2
    left = arr[:mid]
    right = arr[mid:]

    # Fork: Create parallel tasks
    with Pool(2) as pool:
        sorted_left, sorted_right = pool.map(merge_sort_parallel, [left, right])

    # Join: Merge results
    return merge(sorted_left, sorted_right)
```

### Pipeline

**Pipeline:** Data flows through stages, each stage processes in parallel.

```python
# Stage 1: Read files
# Stage 2: Parse data
# Stage 3: Process data
# Stage 4: Write results

from queue import Queue
from threading import Thread

def stage1_read(output_queue):
    for filename in filenames:
        data = read_file(filename)
        output_queue.put(data)
    output_queue.put(None)  # Signal end

def stage2_parse(input_queue, output_queue):
    while True:
        data = input_queue.get()
        if data is None:
            break
        parsed = parse(data)
        output_queue.put(parsed)
    output_queue.put(None)

def stage3_process(input_queue, output_queue):
    while True:
        data = input_queue.get()
        if data is None:
            break
        processed = process(data)
        output_queue.put(processed)
    output_queue.put(None)

def stage4_write(input_queue):
    while True:
        data = input_queue.get()
        if data is None:
            break
        write_result(data)

# Connect stages with queues
q1 = Queue()
q2 = Queue()
q3 = Queue()

Thread(target=stage1_read, args=(q1,)).start()
Thread(target=stage2_parse, args=(q1, q2)).start()
Thread(target=stage3_process, args=(q2, q3)).start()
Thread(target=stage4_write, args=(q3,)).start()
```

### Thread Pool / Worker Pool

**Idea:** Pre-create threads, assign tasks from a queue.

```python
from concurrent.futures import ThreadPoolExecutor

def task(n):
    return n * n

# Create thread pool with 4 workers
with ThreadPoolExecutor(max_workers=4) as executor:
    # Submit tasks
    futures = [executor.submit(task, i) for i in range(100)]

    # Get results
    results = [future.result() for future in futures]

print(results)
```

---

## Common Pitfalls

### 1. Race Conditions (Check-Then-Act)

```python
# BAD: Race condition
if not file_exists("output.txt"):  # Thread 1 checks
    # Thread 2 also checks (file still doesn't exist)
    create_file("output.txt")  # Both threads create the file!

# GOOD: Atomic operation
try:
    create_file_exclusive("output.txt")  # Fails if file exists
except FileExistsError:
    pass
```

### 2. Deadlock

(See earlier section)

### 3. Livelock

**Livelock:** Threads keep changing state in response to each other, but make no progress.

**Example:**
```
Person A and Person B meet in a narrow hallway.
A steps left to let B pass.
B steps right to let A pass.
Now they're blocking each other again.
A steps right.
B steps left.
Repeat forever...
```

**Prevention:** Add randomness to retry logic.

### 4. Starvation

**Starvation:** A thread never gets a chance to run because others keep taking priority.

**Example:**
```python
# High-priority threads keep getting the lock
# Low-priority thread waits forever
```

**Prevention:** Use fair locks that guarantee eventual access.

### 5. False Sharing (Cache Lines)

**Problem:** Two threads access different variables, but they're on the same cache line, causing cache invalidation.

```c
// BAD: Both on same cache line (typically 64 bytes)
struct {
    int counter1;  // Thread 1 modifies
    int counter2;  // Thread 2 modifies
} shared_data;

// Every write by Thread 1 invalidates Thread 2's cache (and vice versa)

// GOOD: Pad to separate cache lines
struct {
    int counter1;
    char padding[60];  // Ensure counter2 is on different cache line
    int counter2;
} shared_data;
```

---

## Immutability as a Concurrency Strategy

**Immutable data can be safely shared between threads without locks.**

```python
# Mutable (requires locking)
class Counter:
    def __init__(self):
        self.count = 0
        self.lock = Lock()

    def increment(self):
        with self.lock:
            self.count += 1

# Immutable (no locking needed)
class ImmutableCounter:
    def __init__(self, count):
        self._count = count

    def increment(self):
        return ImmutableCounter(self._count + 1)  # Returns new instance

# Usage
counter = ImmutableCounter(0)
counter = counter.increment()  # New object, old one unchanged
```

**Functional programming languages** (Haskell, Clojure, Erlang) use immutability extensively for safe concurrency.

---

## Lock-Free Data Structures

**Lock-free:** Use atomic operations (compare-and-swap) instead of locks.

**Example: Atomic counter**
```python
from threading import Thread
import ctypes

class AtomicCounter:
    def __init__(self):
        self._value = ctypes.c_int(0)

    def increment(self):
        # This is conceptual; Python doesn't have true CAS
        # In C++: std::atomic<int>
        while True:
            old_value = self._value.value
            new_value = old_value + 1
            # Compare-and-swap: update only if value hasn't changed
            if compare_and_swap(self._value, old_value, new_value):
                break

    def get(self):
        return self._value.value
```

**C++ atomic example:**
```cpp
#include <atomic>

std::atomic<int> counter(0);

void increment() {
    counter.fetch_add(1);  // Atomic increment
}
```

**Benefits:**
- No locks → no deadlocks
- Better performance in low-contention scenarios

**Drawbacks:**
- Complex to implement correctly
- ABA problem (value changes A→B→A, CAS thinks nothing changed)

---

## Summary

**Key Principles:**

1. **Concurrency ≠ Parallelism** – Structure vs execution
2. **Shared state is dangerous** – Locks or immutability required
3. **Locks introduce complexity** – Deadlocks, ordering, performance
4. **Message passing > shared memory** – In many cases
5. **Async for I/O, threads for CPU** – Choose the right tool
6. **Immutability helps** – No locks needed
7. **Test thoroughly** – Concurrency bugs are rare and hard to reproduce

**Mental Model:**
- **Threads:** Shared memory, requires synchronization, can run in parallel
- **Async/await:** Single thread, cooperative multitasking, I/O-bound
- **Message passing:** No shared state, isolated actors/processes

---

## Exercises

### Exercise 1: Identify Race Conditions

Find and fix the race condition:

```python
balance = 1000

def withdraw(amount):
    global balance
    if balance >= amount:
        time.sleep(0.01)  # Simulate processing time
        balance -= amount
        return True
    return False

# Two threads withdraw simultaneously
t1 = Thread(target=lambda: withdraw(600))
t2 = Thread(target=lambda: withdraw(600))
t1.start()
t2.start()
t1.join()
t2.join()

print(f"Balance: {balance}")  # Should be 1000 or 400, never -200!
```

### Exercise 2: Implement Producer-Consumer

Implement a thread-safe producer-consumer system where:
- 3 producers generate random numbers (1-100)
- 2 consumers compute the sum
- Print the final sum after all numbers are processed

Use a queue and appropriate synchronization.

### Exercise 3: Async/Await for I/O

Rewrite this sequential code to use async/await:

```python
def fetch_user(user_id):
    time.sleep(1)  # Simulate network delay
    return {"id": user_id, "name": f"User{user_id}"}

def fetch_orders(user_id):
    time.sleep(1)  # Simulate network delay
    return [{"id": 1, "total": 100}, {"id": 2, "total": 200}]

def get_user_with_orders(user_id):
    user = fetch_user(user_id)
    orders = fetch_orders(user_id)
    return {"user": user, "orders": orders}

# Takes 2 seconds (sequential)
result = get_user_with_orders(123)
```

Optimize to run in 1 second using asyncio.

### Exercise 4: Parallel Map-Reduce

Implement a parallel word count:
- Input: List of text files
- Output: Dictionary of word frequencies
- Use multiprocessing to process files in parallel

```python
def word_count(files):
    # Map: Count words in each file (parallel)
    # Reduce: Combine counts
    pass
```

### Exercise 5: Deadlock Scenario

This code has a potential deadlock. Fix it:

```python
lock_a = Lock()
lock_b = Lock()

def transfer_a_to_b(amount):
    with lock_a:
        with lock_b:
            # Transfer from A to B
            pass

def transfer_b_to_a(amount):
    with lock_b:
        with lock_a:
            # Transfer from B to A
            pass

# If both run simultaneously, deadlock occurs
```

---

---

**Previous**: [Debugging & Profiling](./11_Debugging_and_Profiling.md) | **Next**: [API Design Principles](./13_API_Design.md)
