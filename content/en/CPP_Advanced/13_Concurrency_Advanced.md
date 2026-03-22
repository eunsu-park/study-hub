# Advanced Concurrency

**Previous**: [Multithreading](./12_Multithreading.md) | **Next**: [Design Patterns: Creational and Structural](./14_Design_Patterns_Creational_Structural.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Synchronize threads using C++20 primitives: `std::latch`, `std::barrier`, and `std::counting_semaphore`
2. Perform atomic operations with `std::atomic` and explain `compare_exchange_strong` vs `compare_exchange_weak`
3. Select the correct memory ordering (`relaxed`, `acquire`, `release`, `seq_cst`) for a given synchronization scenario
4. Identify the ABA problem and apply techniques to build lock-free data structures
5. Accelerate STL algorithms using execution policies (`std::execution::par`, `par_unseq`)
6. Implement a reusable thread pool with task submission and future-based result retrieval
7. Apply the producer-consumer pattern with bounded buffers and condition variables

---

The previous lesson introduced threads, mutexes, and futures -- the fundamental building blocks. This lesson goes deeper: C++20 added synchronization primitives that simplify coordination patterns that were previously error-prone. Understanding memory ordering lets you write correct lock-free code. Parallel algorithms let the standard library handle threading for you. These techniques are what separate programs that "work on my machine" from programs that are provably correct under all scheduling interleavings.

---

## Table of Contents

1. [C++20 Synchronization](#1-c20-synchronization)
2. [Atomic Operations](#2-atomic-operations)
3. [Memory Ordering](#3-memory-ordering)
4. [Lock-Free Programming](#4-lock-free-programming)
5. [Parallel Algorithms](#5-parallel-algorithms)
6. [Thread Pools](#6-thread-pools)
7. [Producer-Consumer Patterns](#7-producer-consumer-patterns)
8. [Coroutine-Based Concurrency](#8-coroutine-based-concurrency)

---

## 1. C++20 Synchronization

### std::latch

A latch is a single-use countdown barrier. Threads decrement the counter; when it reaches zero, all waiting threads are released.

```cpp
#include <latch>
#include <thread>
#include <iostream>
#include <vector>

void worker(int id, std::latch& start_signal, std::latch& done_signal) {
    // Wait for all workers to be created
    start_signal.wait();

    std::cout << "Worker " << id << " is running\n";

    // Signal completion
    done_signal.count_down();
}

int main() {
    constexpr int N = 5;
    std::latch start_signal(1);    // Main thread releases everyone
    std::latch done_signal(N);     // Wait for N workers

    std::vector<std::jthread> threads;
    for (int i = 0; i < N; ++i) {
        threads.emplace_back(worker, i, std::ref(start_signal),
                             std::ref(done_signal));
    }

    std::cout << "All workers created. Starting...\n";
    start_signal.count_down();  // Release all workers

    done_signal.wait();  // Wait for all to finish
    std::cout << "All workers done.\n";

    return 0;
}
```

### std::barrier

A barrier is a reusable synchronization point. Threads arrive, wait for all to arrive, then a completion function runs, and the barrier resets.

```cpp
#include <barrier>
#include <thread>
#include <iostream>
#include <vector>

int main() {
    constexpr int N = 4;
    int phase = 0;

    // Completion function runs after all threads arrive
    auto on_completion = [&phase]() noexcept {
        ++phase;
        std::cout << "--- Phase " << phase << " complete ---\n";
    };

    std::barrier sync_point(N, on_completion);

    auto task = [&](int id) {
        for (int i = 0; i < 3; ++i) {
            std::cout << "Thread " << id << " doing phase work\n";
            sync_point.arrive_and_wait();  // Synchronize
        }
    };

    std::vector<std::jthread> threads;
    for (int i = 0; i < N; ++i) {
        threads.emplace_back(task, i);
    }

    return 0;
}
```

### std::counting_semaphore

A semaphore limits concurrent access to a resource to at most N threads.

```cpp
#include <semaphore>
#include <thread>
#include <iostream>
#include <vector>
#include <chrono>

// Allow at most 3 concurrent accesses
std::counting_semaphore<3> sem(3);

void access_resource(int id) {
    sem.acquire();  // Decrement (blocks if count == 0)

    std::cout << "Thread " << id << " entered critical section\n";
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    std::cout << "Thread " << id << " leaving critical section\n";

    sem.release();  // Increment
}

int main() {
    std::vector<std::jthread> threads;
    for (int i = 0; i < 10; ++i) {
        threads.emplace_back(access_resource, i);
    }
    return 0;
}
```

### std::binary_semaphore

A special case of `counting_semaphore<1>`, useful as a lightweight mutex or signaling mechanism:

```cpp
#include <semaphore>
#include <thread>

std::binary_semaphore signal(0);  // Initially unavailable

void waiter() {
    signal.acquire();  // Block until signaled
    std::cout << "Signal received!\n";
}

void signaler() {
    std::this_thread::sleep_for(std::chrono::seconds(1));
    signal.release();  // Wake up the waiter
}
```

### Comparison Table

| Primitive | Reusable? | Max Threads | Use Case |
|-----------|-----------|-------------|----------|
| `std::latch` | No (single-use) | Unlimited | "Wait for N tasks to finish" |
| `std::barrier` | Yes (phases) | Fixed N | "Synchronize N threads repeatedly" |
| `std::counting_semaphore` | Yes | Configurable | "Limit concurrency to N" |
| `std::binary_semaphore` | Yes | 1 | "Signal between threads" |

---

## 2. Atomic Operations

### std::atomic in Depth

```cpp
#include <atomic>
#include <iostream>

std::atomic<int> counter{0};

// fetch_add: atomically add and return the old value
void increment_n(int n) {
    for (int i = 0; i < n; ++i) {
        int old = counter.fetch_add(1, std::memory_order_relaxed);
        // old is the value before incrementing
    }
}
```

### compare_exchange_strong vs compare_exchange_weak

```cpp
#include <atomic>

std::atomic<int> value{0};

void cas_demo() {
    int expected = 0;

    // Strong: fails only if value != expected
    bool success = value.compare_exchange_strong(expected, 42);
    // If value was 0: sets to 42, returns true
    // If value was not 0: sets expected to current value, returns false

    // Weak: may spuriously fail even if value == expected
    // More efficient on some architectures (ARM LL/SC)
    expected = 42;
    while (!value.compare_exchange_weak(expected, 100)) {
        // Retry loop — weak CAS is meant to be used in loops
        expected = 42;
    }
}
```

### Atomic Flag (Lock-Free Boolean)

```cpp
#include <atomic>

// The only type guaranteed to be lock-free on all platforms
std::atomic_flag lock = ATOMIC_FLAG_INIT;

void spinlock_acquire() {
    while (lock.test_and_set(std::memory_order_acquire)) {
        // Spin — optionally use yield or pause
    }
}

void spinlock_release() {
    lock.clear(std::memory_order_release);
}
```

### Atomic with User-Defined Types

```cpp
#include <atomic>

struct Point {
    int x, y;
};

// Works if Point is trivially copyable and fits in a machine word (or two)
std::atomic<Point> pos{{0, 0}};

void update() {
    Point expected = pos.load();
    Point desired = {expected.x + 1, expected.y + 1};
    while (!pos.compare_exchange_weak(expected, desired)) {
        desired = {expected.x + 1, expected.y + 1};
    }
}
```

---

## 3. Memory Ordering

### The Problem

Modern CPUs and compilers reorder memory operations for performance. Without proper ordering constraints, one thread's writes may appear in a different order to another thread.

### Memory Order Options

| Order | Guarantees | Performance |
|-------|-----------|-------------|
| `memory_order_relaxed` | Atomicity only. No ordering guarantees. | Fastest |
| `memory_order_acquire` | Reads after this cannot be reordered before it. | Medium |
| `memory_order_release` | Writes before this cannot be reordered after it. | Medium |
| `memory_order_acq_rel` | Both acquire and release. | Medium |
| `memory_order_seq_cst` | Total order across all threads. Default. | Slowest |

### Acquire-Release Example

```cpp
#include <atomic>
#include <thread>
#include <cassert>

std::atomic<bool> ready{false};
int data = 0;

void producer() {
    data = 42;                                    // (1) ordinary write
    ready.store(true, std::memory_order_release); // (2) release store
    // Guarantee: (1) is visible before (2) to any acquire-loading thread
}

void consumer() {
    while (!ready.load(std::memory_order_acquire)) {} // (3) acquire load
    assert(data == 42);  // (4) guaranteed to see data == 42
    // Because (3) acquires what (2) released, (1) is visible at (4)
}
```

### Relaxed Ordering

```cpp
#include <atomic>
#include <thread>

std::atomic<int> counter{0};

// Relaxed is fine for a simple counter where we only care about
// the final value, not ordering relative to other variables
void count() {
    for (int i = 0; i < 10000; ++i) {
        counter.fetch_add(1, std::memory_order_relaxed);
    }
}
```

### Sequential Consistency

```cpp
#include <atomic>
#include <thread>
#include <cassert>

std::atomic<bool> x{false}, y{false};
std::atomic<int> z{0};

void write_x() { x.store(true, std::memory_order_seq_cst); }
void write_y() { y.store(true, std::memory_order_seq_cst); }

void read_x_then_y() {
    while (!x.load(std::memory_order_seq_cst)) {}
    if (y.load(std::memory_order_seq_cst)) ++z;
}

void read_y_then_x() {
    while (!y.load(std::memory_order_seq_cst)) {}
    if (x.load(std::memory_order_seq_cst)) ++z;
}

// With seq_cst, z can never be 0 after all threads complete.
// At least one of the read functions will see both flags as true.
```

---

## 4. Lock-Free Programming

### Lock-Free Stack

```cpp
#include <atomic>
#include <memory>

template<typename T>
class LockFreeStack {
    struct Node {
        T data;
        Node* next;
        Node(T val) : data(std::move(val)), next(nullptr) {}
    };

    std::atomic<Node*> head{nullptr};

public:
    void push(T value) {
        Node* new_node = new Node(std::move(value));
        new_node->next = head.load(std::memory_order_relaxed);
        // CAS loop: keep trying until we successfully link the new node
        while (!head.compare_exchange_weak(
                   new_node->next, new_node,
                   std::memory_order_release,
                   std::memory_order_relaxed)) {
            // new_node->next is updated to current head on failure
        }
    }

    bool pop(T& result) {
        Node* old_head = head.load(std::memory_order_acquire);
        while (old_head &&
               !head.compare_exchange_weak(
                   old_head, old_head->next,
                   std::memory_order_release,
                   std::memory_order_acquire)) {
            // old_head updated on failure
        }
        if (!old_head) return false;
        result = std::move(old_head->data);
        delete old_head;  // Caution: see ABA problem below
        return true;
    }
};
```

### The ABA Problem

```
Thread 1: reads head = A, prepares to CAS(A, B)
Thread 2: pops A, pops B, pushes A back (A is recycled)
Thread 1: CAS succeeds (head is still A), but the stack is corrupted
```

**Solutions:**
- **Tagged pointers**: pair the pointer with a version counter
- **Hazard pointers**: defer deletion until no thread holds a reference
- **Epoch-based reclamation**: batch memory reclamation by epoch

```cpp
// Tagged pointer approach (simplified)
#include <atomic>
#include <cstdint>

struct TaggedPtr {
    void* ptr;
    uint64_t tag;  // Incremented on every push/pop

    bool operator==(const TaggedPtr& other) const {
        return ptr == other.ptr && tag == other.tag;
    }
};

// Use std::atomic<TaggedPtr> if it's lock-free on your platform
// Otherwise, use a double-width CAS or platform-specific intrinsics
```

---

## 5. Parallel Algorithms

C++17 added execution policies to standard algorithms. The compiler and runtime handle the threading.

### Execution Policies

```cpp
#include <algorithm>
#include <execution>
#include <vector>
#include <numeric>

int main() {
    std::vector<int> v(10'000'000);
    std::iota(v.begin(), v.end(), 0);

    // Sequential (default)
    std::sort(std::execution::seq, v.begin(), v.end());

    // Parallel
    std::sort(std::execution::par, v.begin(), v.end());

    // Parallel + vectorized (SIMD)
    std::sort(std::execution::par_unseq, v.begin(), v.end());

    return 0;
}
```

### Common Parallel Algorithms

```cpp
#include <algorithm>
#include <execution>
#include <numeric>
#include <vector>

std::vector<int> v(1'000'000, 1);

// Parallel reduce (sum)
long long sum = std::reduce(std::execution::par, v.begin(), v.end(), 0LL);

// Parallel transform
std::transform(std::execution::par, v.begin(), v.end(), v.begin(),
               [](int x) { return x * 2; });

// Parallel for_each
std::for_each(std::execution::par, v.begin(), v.end(),
              [](int& x) { x += 1; });

// Parallel find
auto it = std::find(std::execution::par, v.begin(), v.end(), 42);

// Parallel count
auto cnt = std::count_if(std::execution::par, v.begin(), v.end(),
                         [](int x) { return x > 100; });
```

### Performance Considerations

| Factor | Guidance |
|--------|----------|
| Data size | Parallel pays off only for large datasets (>10K elements) |
| Work per element | Trivial lambdas may not offset thread overhead |
| Memory access | Contiguous data (vector) parallelizes better than linked structures |
| Compilation | Requires linking TBB on GCC (`-ltbb`) |

```bash
# Compile with TBB for parallel execution (GCC)
g++ -std=c++17 -O2 -ltbb program.cpp -o program
```

---

## 6. Thread Pools

### Improved Thread Pool with jthread

```cpp
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <future>
#include <vector>
#include <iostream>

class ThreadPool {
    std::vector<std::jthread> workers_;
    std::queue<std::function<void()>> tasks_;
    std::mutex mtx_;
    std::condition_variable cv_;
    bool shutdown_ = false;

public:
    explicit ThreadPool(size_t num_threads = std::thread::hardware_concurrency()) {
        for (size_t i = 0; i < num_threads; ++i) {
            workers_.emplace_back([this](std::stop_token stoken) {
                while (true) {
                    std::function<void()> task;
                    {
                        std::unique_lock lock(mtx_);
                        cv_.wait(lock, [this, &stoken] {
                            return shutdown_ || !tasks_.empty()
                                   || stoken.stop_requested();
                        });
                        if ((shutdown_ || stoken.stop_requested())
                            && tasks_.empty()) {
                            return;
                        }
                        task = std::move(tasks_.front());
                        tasks_.pop();
                    }
                    task();
                }
            });
        }
    }

    template<typename F, typename... Args>
    auto submit(F&& f, Args&&... args)
        -> std::future<std::invoke_result_t<F, Args...>>
    {
        using R = std::invoke_result_t<F, Args...>;
        auto task_ptr = std::make_shared<std::packaged_task<R()>>(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...)
        );
        auto future = task_ptr->get_future();
        {
            std::unique_lock lock(mtx_);
            tasks_.emplace([task_ptr] { (*task_ptr)(); });
        }
        cv_.notify_one();
        return future;
    }

    ~ThreadPool() {
        {
            std::unique_lock lock(mtx_);
            shutdown_ = true;
        }
        cv_.notify_all();
        // jthread destructors request_stop + join automatically
    }
};

int main() {
    ThreadPool pool(4);

    std::vector<std::future<int>> results;
    for (int i = 0; i < 20; ++i) {
        results.push_back(pool.submit([i] {
            return i * i;
        }));
    }

    for (auto& f : results) {
        std::cout << f.get() << " ";
    }
    std::cout << "\n";

    return 0;
}
```

---

## 7. Producer-Consumer Patterns

### Bounded Buffer

```cpp
#include <queue>
#include <mutex>
#include <condition_variable>
#include <optional>
#include <semaphore>

template<typename T, size_t Capacity>
class BoundedBuffer {
    std::queue<T> buffer_;
    std::mutex mtx_;
    std::counting_semaphore<Capacity> empty_slots_{Capacity};
    std::counting_semaphore<Capacity> full_slots_{0};

public:
    void produce(T item) {
        empty_slots_.acquire();   // Wait for an empty slot
        {
            std::lock_guard lock(mtx_);
            buffer_.push(std::move(item));
        }
        full_slots_.release();    // Signal a filled slot
    }

    T consume() {
        full_slots_.acquire();    // Wait for a filled slot
        T item;
        {
            std::lock_guard lock(mtx_);
            item = std::move(buffer_.front());
            buffer_.pop();
        }
        empty_slots_.release();   // Signal an empty slot
        return item;
    }
};
```

### Multi-Producer Multi-Consumer

```cpp
#include <thread>
#include <iostream>
#include <vector>

int main() {
    BoundedBuffer<int, 10> buffer;
    std::atomic<bool> done{false};

    // 3 Producers
    std::vector<std::jthread> producers;
    for (int p = 0; p < 3; ++p) {
        producers.emplace_back([&buffer, p] {
            for (int i = 0; i < 10; ++i) {
                buffer.produce(p * 100 + i);
            }
        });
    }

    // 2 Consumers
    std::atomic<int> consumed_count{0};
    std::vector<std::jthread> consumers;
    for (int c = 0; c < 2; ++c) {
        consumers.emplace_back([&buffer, &consumed_count] {
            while (consumed_count.fetch_add(1) < 30) {
                int item = buffer.consume();
                std::cout << "Consumed: " << item << "\n";
            }
        });
    }

    return 0;
}
```

---

## 8. Coroutine-Based Concurrency

Coroutines can replace callbacks in async I/O patterns. Here is a sketch of how a coroutine-based task integrates with a simple scheduler.

```cpp
#include <coroutine>
#include <queue>
#include <iostream>
#include <functional>

class Scheduler {
    std::queue<std::coroutine_handle<>> ready_;

public:
    void schedule(std::coroutine_handle<> h) {
        ready_.push(h);
    }

    void run() {
        while (!ready_.empty()) {
            auto h = ready_.front();
            ready_.pop();
            if (!h.done()) h.resume();
        }
    }

    // Awaitable: yield control back to the scheduler
    auto yield() {
        struct YieldAwaiter {
            Scheduler& sched;
            bool await_ready() { return false; }
            void await_suspend(std::coroutine_handle<> h) {
                sched.schedule(h);
            }
            void await_resume() {}
        };
        return YieldAwaiter{*this};
    }
};

// Simple coroutine task for the scheduler
struct ScheduledTask {
    struct promise_type {
        ScheduledTask get_return_object() { return {}; }
        std::suspend_never initial_suspend() { return {}; }
        std::suspend_never final_suspend() noexcept { return {}; }
        void return_void() {}
        void unhandled_exception() { std::terminate(); }
    };
};

Scheduler global_sched;

ScheduledTask task_a() {
    std::cout << "Task A: step 1\n";
    co_await global_sched.yield();
    std::cout << "Task A: step 2\n";
    co_await global_sched.yield();
    std::cout << "Task A: step 3\n";
}

ScheduledTask task_b() {
    std::cout << "Task B: step 1\n";
    co_await global_sched.yield();
    std::cout << "Task B: step 2\n";
}

int main() {
    task_a();  // Starts, runs step 1, yields
    task_b();  // Starts, runs step 1, yields

    global_sched.run();
    // Output interleaves: A1, B1, A2, B2, A3
    return 0;
}
```

---

## Exercises

### Exercise 1: Barrier-Based Matrix Computation

Use `std::barrier` to implement a parallel Jacobi iteration solver. N threads each update one row of a matrix, then synchronize before the next iteration. Run for 100 iterations on a 1000x1000 grid.

### Exercise 2: Lock-Free Queue

Implement a single-producer, single-consumer lock-free queue using `std::atomic` and a fixed-size ring buffer. Test with one producer thread generating 1 million items and one consumer thread reading them.

### Exercise 3: Memory Ordering Experiment

Write a program that demonstrates the difference between `memory_order_relaxed` and `memory_order_seq_cst`. Use two atomic variables and two threads. Show that relaxed ordering can produce results impossible under sequential consistency.

### Exercise 4: Parallel Word Count

Given a large text file (>100 MB), use `std::execution::par` with `std::transform_reduce` to count word frequencies. Compare performance against a single-threaded implementation.

### Exercise 5: Async Pipeline with Coroutines

Build a coroutine-based pipeline with three stages: `read` (produces strings), `transform` (converts to uppercase), `write` (prints). Use a scheduler to interleave execution. Each stage should process 10 items.

---

## Next Steps

Concurrency gives your programs speed; design patterns give them structure. The next two lessons cover the Gang of Four patterns and C++ idioms that produce maintainable, extensible architectures.

- [Design Patterns: Creational and Structural](./14_Design_Patterns_Creational_Structural.md)
