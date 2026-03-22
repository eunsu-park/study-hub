# C++20 Coroutines

**Previous**: [C++20 Ranges](./09_CPP20_Ranges.md) | **Next**: [C++20 Modules and Utilities](./11_Modules_and_CPP20_Utilities.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain how stackless coroutines differ from regular functions and threads, including suspend/resume semantics
2. Use `co_await` to suspend a coroutine until an awaitable object is ready
3. Implement the generator pattern with `co_yield` to produce lazy sequences
4. Return final values from coroutines using `co_return`
5. Design a `promise_type` that controls coroutine creation, suspension, and completion
6. Manipulate `coroutine_handle` objects to resume, destroy, and inspect coroutine state
7. Build a complete Generator class and an async Task class from scratch

---

Coroutines are functions that can **suspend** their execution and **resume** later without blocking a thread. Unlike threads, coroutines are cooperative: they yield control explicitly, so there is no context-switch overhead and no need for locks around shared state. C++20 provides the low-level machinery -- `co_await`, `co_yield`, `co_return`, and the promise/handle protocol -- from which you build high-level abstractions like generators, async tasks, and event loops. Understanding this machinery is essential because the standard library provides almost no ready-made coroutine types until C++23's `std::generator`.

---

## Table of Contents

1. [What Are Coroutines?](#1-what-are-coroutines)
2. [co_await](#2-co_await)
3. [co_yield](#3-co_yield)
4. [co_return](#4-co_return)
5. [Promise Type](#5-promise-type)
6. [Coroutine Handle](#6-coroutine-handle)
7. [Implementing a Generator](#7-implementing-a-generator)
8. [Async Tasks](#8-async-tasks)

---

## 1. What Are Coroutines?

### Stackless Coroutines

C++20 coroutines are **stackless**: the compiler transforms the function body into a state machine stored on the heap. Each `co_await`, `co_yield`, or `co_return` becomes a suspension point.

```
Regular function:     call  ────────────────────>  return
                                 (runs to completion)

Coroutine:            call  ──> suspend ──> resume ──> suspend ──> resume ──> return
                                   |           ^          |           ^
                                   v           |          v           |
                                 (caller)   (caller)   (caller)   (caller)
```

### Coroutines vs Threads

| Property | Coroutines | Threads |
|----------|-----------|---------|
| Scheduling | Cooperative (explicit yield) | Preemptive (OS scheduler) |
| Stack | Heap-allocated frame | Full stack (~1-8 MB) |
| Context switch | Nanoseconds | Microseconds |
| Concurrency | Single-threaded by default | Truly parallel |
| Synchronization | Usually not needed | Mutexes, atomics required |

### Use Cases

- **Generators**: lazily produce sequences (e.g., Fibonacci, file lines)
- **Async I/O**: suspend while waiting for network/disk, resume when ready
- **State machines**: natural encoding of multi-step protocols
- **Cooperative multitasking**: run thousands of tasks on a single thread

### The Three Keywords

Any function containing `co_await`, `co_yield`, or `co_return` is a coroutine:

```cpp
#include <coroutine>

// This is a coroutine because it uses co_return
Task<int> example() {
    co_return 42;
}

// This is a coroutine because it uses co_yield
Generator<int> sequence() {
    co_yield 1;
    co_yield 2;
}

// This is a coroutine because it uses co_await
Task<void> async_work() {
    co_await some_async_operation();
}
```

---

## 2. co_await

### Awaitable Objects

`co_await expr` suspends the coroutine if the expression is not yet ready. The expression must produce an **awaitable** -- an object with three methods:

```cpp
struct MyAwaitable {
    // Called first: should we suspend at all?
    bool await_ready() const noexcept {
        return false;  // false = yes, suspend
    }

    // Called when suspending: receives the coroutine handle
    void await_suspend(std::coroutine_handle<> h) const noexcept {
        // Option 1: store h for later resumption
        // Option 2: resume immediately (symmetric transfer)
        // Option 3: return void (always suspend)
    }

    // Called when resumed: produces the co_await result
    int await_resume() const noexcept {
        return 42;  // This becomes the value of `co_await expr`
    }
};
```

### Built-in Awaitables

```cpp
#include <coroutine>

// Always suspends
std::suspend_always{};
// await_ready() returns false
// await_suspend() does nothing
// await_resume() returns void

// Never suspends
std::suspend_never{};
// await_ready() returns true
// (await_suspend and await_resume are never called)
```

### await_suspend Return Types

The return type of `await_suspend` controls behavior:

| Return Type | Behavior |
|-------------|----------|
| `void` | Always suspend |
| `bool` | `true` = suspend, `false` = resume immediately |
| `coroutine_handle<>` | Symmetric transfer: resume the returned handle |

```cpp
struct TransferAwaitable {
    std::coroutine_handle<> target;

    bool await_ready() { return false; }

    // Symmetric transfer: suspend this coroutine, resume target
    std::coroutine_handle<> await_suspend(std::coroutine_handle<>) {
        return target;
    }

    void await_resume() {}
};
```

---

## 3. co_yield

### Generator Pattern

`co_yield value` is syntactic sugar for `co_await promise.yield_value(value)`. It suspends the coroutine and makes `value` available to the caller.

```cpp
#include <coroutine>
#include <iostream>

// Forward declaration — full implementation in Section 7
template<typename T>
struct Generator;

Generator<int> fibonacci() {
    int a = 0, b = 1;
    while (true) {
        co_yield a;
        int next = a + b;
        a = b;
        b = next;
    }
}

// Usage (assuming Generator is defined):
int main() {
    auto gen = fibonacci();
    for (int i = 0; i < 10; ++i) {
        gen.next();
        std::cout << gen.value() << " ";
    }
    // Output: 0 1 1 2 3 5 8 13 21 34
    return 0;
}
```

### Yielding Different Types

```cpp
Generator<std::string> greetings() {
    co_yield "Hello";
    co_yield "Bonjour";
    co_yield "Hola";
    co_yield "Hallo";
}

Generator<int> range(int start, int end) {
    for (int i = start; i < end; ++i) {
        co_yield i;
    }
    // Coroutine ends — generator reports no more values
}
```

---

## 4. co_return

### Returning Final Values

`co_return value` terminates the coroutine and stores a final result. `co_return` (without a value) terminates a void-returning coroutine.

```cpp
// co_return with a value
Task<int> compute(int x) {
    int result = x * x + 2 * x + 1;
    co_return result;
}

// co_return void (implicit at end of coroutine body)
Task<void> log_message(const std::string& msg) {
    std::cout << msg << "\n";
    co_return;  // explicit, or implicit at '}'
}
```

### Promise Interaction

When the compiler sees `co_return value`, it calls `promise.return_value(value)`. For `co_return` without a value, it calls `promise.return_void()`. A promise type must define exactly one of these.

```cpp
struct promise_type {
    int result;

    // For co_return <value>
    void return_value(int v) { result = v; }

    // OR for co_return (void)
    // void return_void() {}

    // Never both!
};
```

---

## 5. Promise Type

### The promise_type Interface

Every coroutine return type `R` must have a nested `R::promise_type` (or a specialization of `std::coroutine_traits`). The promise controls the entire coroutine lifecycle.

```cpp
template<typename T>
struct Task {
    struct promise_type {
        T result;

        // 1. Called to create the return object
        Task get_return_object() {
            return Task{
                std::coroutine_handle<promise_type>::from_promise(*this)
            };
        }

        // 2. Should the coroutine suspend before running the body?
        std::suspend_never initial_suspend() { return {}; }
        //   suspend_never = start eagerly
        //   suspend_always = start lazily (caller must resume)

        // 3. Should the coroutine suspend after completing?
        std::suspend_always final_suspend() noexcept { return {}; }
        //   suspend_always = coroutine frame stays alive for the caller to read
        //   suspend_never = frame destroyed immediately (dangerous if handle is used)

        // 4. Store the co_return value
        void return_value(T value) { result = std::move(value); }

        // 5. Handle uncaught exceptions
        void unhandled_exception() { std::terminate(); }
    };

    std::coroutine_handle<promise_type> handle;

    T get() const { return handle.promise().result; }

    ~Task() { if (handle) handle.destroy(); }
};
```

### Lifecycle Summary

```
1. Allocate coroutine frame (heap)
2. Construct promise_type in the frame
3. Call promise.get_return_object() → returned to caller
4. Call co_await promise.initial_suspend()
5. Execute coroutine body (suspend/resume at co_await/co_yield points)
6. Call promise.return_value(x) or promise.return_void()
7. Call co_await promise.final_suspend()
8. Destroy promise, deallocate frame (when handle.destroy() is called)
```

### Heap Allocation and HALO

Coroutine frames are **heap-allocated** by default because the compiler does not know at call time how long the coroutine will live. The compiler may eliminate this allocation through **HALO** (Heap Allocation eLision Optimization) when it can prove that the coroutine's lifetime is strictly bounded by its caller's stack frame — for example, when the coroutine is created and fully consumed within a single scope. HALO is an optional compiler optimization, not a language guarantee; do not rely on it for performance-critical code without profiling.

### Symmetric Transfer

Without symmetric transfer, chaining coroutines via `co_await` grows the call stack by one frame per level. A deeply nested chain of `co_await` calls can overflow the stack just as badly as deep recursion.

**Symmetric transfer** solves this: when `await_suspend` returns a `coroutine_handle<>` (instead of `void`), the compiler performs a **tail-call** — it suspends the current coroutine and resumes the returned one without adding a new stack frame. The `AwaitableTask` in Section 8 already uses symmetric transfer in its `final_suspend`:

```cpp
// In FinalAwaiter::await_suspend:
std::coroutine_handle<> await_suspend(
    std::coroutine_handle<promise_type> h) noexcept {
    if (h.promise().continuation)
        return h.promise().continuation;  // <-- symmetric transfer
    return std::noop_coroutine();         // no continuation: stop
}
// The runtime resumes `continuation` directly, without growing the stack.
```

Use symmetric transfer whenever building coroutine chains that could reach arbitrary depth (e.g., recursive algorithms, deeply nested async calls).

---

## 6. Coroutine Handle

### std::coroutine_handle

The handle is a non-owning pointer to the coroutine frame. It provides the interface to resume, destroy, and inspect the coroutine.

```cpp
#include <coroutine>

void demo(std::coroutine_handle<> h) {
    // Check if the coroutine has finished
    if (h.done()) {
        std::cout << "Coroutine is done\n";
    }

    // Resume the coroutine (undefined behavior if done)
    h.resume();
    // Equivalent: h();

    // Destroy the coroutine frame (frees memory)
    h.destroy();

    // Get the raw address (for storage in C callbacks)
    void* addr = h.address();
    auto h2 = std::coroutine_handle<>::from_address(addr);
}
```

### Typed vs Untyped Handles

```cpp
// Untyped: cannot access the promise
std::coroutine_handle<> generic;

// Typed: can access the promise via .promise()
std::coroutine_handle<MyPromise> typed;
MyPromise& p = typed.promise();

// Typed handle converts implicitly to untyped
generic = typed;

// Create from promise
auto h = std::coroutine_handle<MyPromise>::from_promise(my_promise);
```

### Ownership Rules

- `coroutine_handle` does **not** own the coroutine frame
- You must call `handle.destroy()` exactly once when done
- Resuming a destroyed or completed coroutine is undefined behavior
- RAII wrappers (like Generator, Task) should destroy in their destructor

---

## 7. Implementing a Generator

### Complete Generator Class

```cpp
#include <coroutine>
#include <iostream>
#include <optional>
#include <utility>

template<typename T>
class Generator {
public:
    struct promise_type {
        T current_value;
        std::exception_ptr exception = nullptr;

        Generator get_return_object() {
            return Generator{
                std::coroutine_handle<promise_type>::from_promise(*this)
            };
        }

        // Start lazily — caller must call next() first
        std::suspend_always initial_suspend() { return {}; }

        // Keep frame alive so caller can read the last value
        std::suspend_always final_suspend() noexcept { return {}; }

        std::suspend_always yield_value(T value) {
            current_value = std::move(value);
            return {};
        }

        void return_void() {}

        void unhandled_exception() {
            exception = std::current_exception();
        }
    };

    using handle_type = std::coroutine_handle<promise_type>;

    // RAII: destroy the coroutine frame
    Generator(handle_type h) : handle_(h) {}
    ~Generator() { if (handle_) handle_.destroy(); }

    // Move-only
    Generator(Generator&& other) noexcept : handle_(other.handle_) {
        other.handle_ = nullptr;
    }
    Generator& operator=(Generator&& other) noexcept {
        if (this != &other) {
            if (handle_) handle_.destroy();
            handle_ = other.handle_;
            other.handle_ = nullptr;
        }
        return *this;
    }
    Generator(const Generator&) = delete;
    Generator& operator=(const Generator&) = delete;

    // Advance to the next value
    bool next() {
        if (handle_ && !handle_.done()) {
            handle_.resume();
            if (handle_.promise().exception) {
                std::rethrow_exception(handle_.promise().exception);
            }
            return !handle_.done();
        }
        return false;
    }

    // Get the current value
    const T& value() const {
        return handle_.promise().current_value;
    }

    // Range-based for loop support
    struct sentinel {};
    struct iterator {
        handle_type handle;

        iterator& operator++() {
            handle.resume();
            return *this;
        }
        const T& operator*() const { return handle.promise().current_value; }
        bool operator==(sentinel) const { return handle.done(); }
    };

    iterator begin() {
        handle_.resume();  // Advance past initial_suspend
        return {handle_};
    }
    sentinel end() { return {}; }

private:
    handle_type handle_;
};
```

### Using the Generator

```cpp
Generator<int> count_up(int start, int end) {
    for (int i = start; i < end; ++i) {
        co_yield i;
    }
}

Generator<int> fibonacci() {
    int a = 0, b = 1;
    while (true) {
        co_yield a;
        int next = a + b;
        a = b;
        b = next;
    }
}

int main() {
    // Manual iteration
    auto fib = fibonacci();
    for (int i = 0; i < 10 && fib.next(); ++i) {
        std::cout << fib.value() << " ";
    }
    std::cout << "\n";

    // Range-based for
    for (int n : count_up(1, 6)) {
        std::cout << n << " ";  // 1 2 3 4 5
    }
    std::cout << "\n";

    return 0;
}
```

---

## 8. Async Tasks

### Basic Task with co_return

```cpp
#include <coroutine>
#include <iostream>
#include <optional>

template<typename T>
class Task {
public:
    struct promise_type {
        std::optional<T> result;
        std::exception_ptr exception = nullptr;

        Task get_return_object() {
            return Task{handle_type::from_promise(*this)};
        }

        std::suspend_never initial_suspend() { return {}; }
        std::suspend_always final_suspend() noexcept { return {}; }

        void return_value(T value) {
            result = std::move(value);
        }

        void unhandled_exception() {
            exception = std::current_exception();
        }
    };

    using handle_type = std::coroutine_handle<promise_type>;

    Task(handle_type h) : handle_(h) {}
    ~Task() { if (handle_) handle_.destroy(); }

    Task(Task&& other) noexcept : handle_(other.handle_) {
        other.handle_ = nullptr;
    }

    T get() {
        if (handle_.promise().exception) {
            std::rethrow_exception(handle_.promise().exception);
        }
        return std::move(*handle_.promise().result);
    }

private:
    handle_type handle_;
};

Task<int> async_add(int a, int b) {
    co_return a + b;
}

int main() {
    auto task = async_add(10, 20);
    std::cout << "Result: " << task.get() << "\n";  // 30
    return 0;
}
```

### Making Task Awaitable

To chain coroutines with `co_await`, the Task itself must be awaitable:

```cpp
template<typename T>
class AwaitableTask {
public:
    struct promise_type {
        std::optional<T> result;
        std::coroutine_handle<> continuation = nullptr;

        AwaitableTask get_return_object() {
            return AwaitableTask{handle_type::from_promise(*this)};
        }

        std::suspend_always initial_suspend() { return {}; }

        auto final_suspend() noexcept {
            struct FinalAwaiter {
                bool await_ready() noexcept { return false; }
                std::coroutine_handle<> await_suspend(
                    std::coroutine_handle<promise_type> h) noexcept {
                    if (h.promise().continuation)
                        return h.promise().continuation;
                    return std::noop_coroutine();
                }
                void await_resume() noexcept {}
            };
            return FinalAwaiter{};
        }

        void return_value(T value) { result = std::move(value); }
        void unhandled_exception() { std::terminate(); }
    };

    using handle_type = std::coroutine_handle<promise_type>;

    // Make Task awaitable
    bool await_ready() const { return handle_.done(); }

    std::coroutine_handle<> await_suspend(std::coroutine_handle<> caller) {
        handle_.promise().continuation = caller;
        return handle_;  // Symmetric transfer: resume this task
    }

    T await_resume() {
        return std::move(*handle_.promise().result);
    }

    // RAII
    AwaitableTask(handle_type h) : handle_(h) {}
    ~AwaitableTask() { if (handle_) handle_.destroy(); }

private:
    handle_type handle_;
};

// Now coroutines can co_await each other:
AwaitableTask<int> inner() {
    co_return 42;
}

AwaitableTask<int> outer() {
    int val = co_await inner();  // Suspend, run inner, resume with result
    co_return val * 2;
}
```

### Simple Event Loop

```cpp
#include <coroutine>
#include <queue>
#include <functional>
#include <iostream>

// Minimal event loop for scheduling coroutines
class EventLoop {
    std::queue<std::coroutine_handle<>> ready_queue;

public:
    void schedule(std::coroutine_handle<> h) {
        ready_queue.push(h);
    }

    void run() {
        while (!ready_queue.empty()) {
            auto h = ready_queue.front();
            ready_queue.pop();
            if (!h.done()) {
                h.resume();
            }
        }
    }

    // Awaitable that reschedules the coroutine
    auto suspend() {
        struct ScheduleAwaiter {
            EventLoop& loop;
            bool await_ready() { return false; }
            void await_suspend(std::coroutine_handle<> h) {
                loop.schedule(h);
            }
            void await_resume() {}
        };
        return ScheduleAwaiter{*this};
    }
};
```

---

## Exercises

### Exercise 1: Line Generator

Write a coroutine `Generator<std::string> read_lines(const std::string& text)` that yields one line at a time from a multi-line string (split on `'\n'`).

### Exercise 2: Filtered Generator

Write a coroutine `Generator<int> filter_gen(Generator<int> source, std::function<bool(int)> pred)` that consumes values from `source` and yields only those matching `pred`.

### Exercise 3: Task Chain

Using the `AwaitableTask` pattern, write three coroutines: `fetch_data()` returns a string, `parse_data(string)` returns an int, and `process()` chains them with `co_await`. Verify the final result.

### Exercise 4: Interleaved Generators

Write a coroutine `Generator<T> interleave(Generator<T> a, Generator<T> b)` that alternates values from two generators until both are exhausted.

### Exercise 5: Async Timer

Create a `TimerAwaitable` struct whose `await_suspend` schedules resumption after a given number of "ticks" on a simple event loop. Write a coroutine that co_awaits two timers sequentially and prints a message after each.

---

## Next Steps

Coroutines provide the suspension mechanism; Modules and C++20 utilities provide the organizational and formatting tools that complete the C++20 picture.

- [C++20 Modules and Utilities](./11_Modules_and_CPP20_Utilities.md)
