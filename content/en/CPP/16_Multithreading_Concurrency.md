# Multithreading and Concurrency

**Previous**: [Modern C++ (C++11/14/17/20)](./15_Modern_CPP.md) | **Next**: [Advanced C++20](./17_CPP20_Advanced.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Create and manage threads using `std::thread`, `join()`, and `detach()`
2. Identify data races and protect shared state with `std::mutex` and lock guards
3. Coordinate thread execution using `std::condition_variable` for producer-consumer patterns
4. Use `std::atomic` operations to perform lock-free updates on shared variables
5. Write asynchronous code with `std::async`, `std::future`, and `std::promise`
6. Implement a basic thread pool that distributes tasks across worker threads
7. Diagnose and prevent common concurrency bugs: deadlocks, livelocks, and starvation

---

Modern processors deliver performance through parallelism, not faster clock speeds. A single-threaded program leaves most of that hardware idle. Multithreading lets you exploit every core, but it introduces an entirely new class of bugs--data races, deadlocks, and ordering issues--that cannot be found by reading code sequentially. Learning the C++ threading primitives and their correct usage patterns is essential for writing software that is both fast and correct on today's multicore machines.

**Difficulty**: ****

**Prerequisites**: Functions, lambdas, smart pointers

---

## Table of Contents

1. [Thread Basics](#thread-basics)
2. [Mutex and Locks](#mutex-and-locks)
3. [Condition Variables](#condition-variables)
4. [Atomic Operations](#atomic-operations)
5. [Asynchronous Programming](#asynchronous-programming)
6. [Thread Pool](#thread-pool)
7. [Common Problems and Solutions](#common-problems-and-solutions)

---

## Thread Basics

### std::thread

```cpp
#include <iostream>
#include <thread>

void hello() {
    std::cout << "Hello from thread!\n";
}

int main() {
    std::thread t(hello);  // Create and start thread
    t.join();              // Wait for thread completion
    return 0;
}
```

### Compilation

```bash
# Linux/macOS
g++ -std=c++17 -pthread program.cpp -o program

# Windows (MSVC)
cl /std:c++17 program.cpp
```

### Creating Thread with Lambda

```cpp
#include <iostream>
#include <thread>

int main() {
    int value = 42;

    // Capture by value
    std::thread t1([value]() {
        std::cout << "Value: " << value << "\n";
    });

    // Capture by reference
    std::thread t2([&value]() {
        value = 100;
    });

    t1.join();
    t2.join();

    std::cout << "After: " << value << "\n";
    return 0;
}
```

### Passing Arguments to Thread

```cpp
#include <iostream>
#include <thread>
#include <string>

void print_message(const std::string& msg, int count) {
    for (int i = 0; i < count; ++i) {
        std::cout << msg << "\n";
    }
}

void modify_value(int& x) {
    x *= 2;
}

int main() {
    // Pass by value
    std::thread t1(print_message, "Hello", 3);

    // Pass by reference (std::ref required)
    int num = 10;
    std::thread t2(modify_value, std::ref(num));

    t1.join();
    t2.join();

    std::cout << "num: " << num << "\n";  // 20
    return 0;
}
```

### join vs detach

```cpp
#include <iostream>
#include <thread>
#include <chrono>

void task() {
    std::this_thread::sleep_for(std::chrono::seconds(2));
    std::cout << "Task completed\n";
}

int main() {
    std::thread t(task);

    // join: Wait until thread completes
    // t.join();

    // detach: Separate thread (independent execution)
    t.detach();

    std::cout << "Main continues\n";

    // Cannot join after detach
    // Note: If main exits first, thread is forcefully terminated

    std::this_thread::sleep_for(std::chrono::seconds(3));
    return 0;
}
```

### Thread ID and Hardware Concurrency

```cpp
#include <iostream>
#include <thread>

void show_id() {
    std::cout << "Thread ID: " << std::this_thread::get_id() << "\n";
}

int main() {
    std::cout << "Main thread ID: " << std::this_thread::get_id() << "\n";
    std::cout << "Hardware concurrency: "
              << std::thread::hardware_concurrency() << "\n";

    std::thread t(show_id);
    t.join();

    return 0;
}
```

### RAII Thread Wrapper

```cpp
#include <thread>

class ThreadGuard {
    std::thread& t;

public:
    explicit ThreadGuard(std::thread& t_) : t(t_) {}

    ~ThreadGuard() {
        if (t.joinable()) {
            t.join();
        }
    }

    // Disable copy
    ThreadGuard(const ThreadGuard&) = delete;
    ThreadGuard& operator=(const ThreadGuard&) = delete;
};

// C++20: std::jthread (auto join)
#include <thread>

void task() { /* ... */ }

int main() {
    std::jthread t(task);  // Automatically joins in destructor
    // No need to call join()
    return 0;
}
```

---

## Mutex and Locks

### Data Race Problem

```cpp
#include <iostream>
#include <thread>
#include <vector>

int counter = 0;

void increment() {
    for (int i = 0; i < 100000; ++i) {
        ++counter;  // Data race!
    }
}

int main() {
    std::thread t1(increment);
    std::thread t2(increment);

    t1.join();
    t2.join();

    // Expected: 200000, Actual: varies each time
    std::cout << "Counter: " << counter << "\n";
    return 0;
}
```

### std::mutex

```cpp
#include <iostream>
#include <thread>
#include <mutex>

int counter = 0;
std::mutex mtx;

void increment() {
    for (int i = 0; i < 100000; ++i) {
        mtx.lock();
        ++counter;
        mtx.unlock();
    }
}

int main() {
    std::thread t1(increment);
    std::thread t2(increment);

    t1.join();
    t2.join();

    std::cout << "Counter: " << counter << "\n";  // Always 200000
    return 0;
}
```

### std::lock_guard (RAII)

```cpp
#include <iostream>
#include <thread>
#include <mutex>

int counter = 0;
std::mutex mtx;

void increment() {
    for (int i = 0; i < 100000; ++i) {
        std::lock_guard<std::mutex> lock(mtx);
        ++counter;
        // lock is automatically released at end of scope
    }
}
```

### std::unique_lock (Flexible Lock)

```cpp
#include <mutex>

std::mutex mtx;

void flexible_locking() {
    std::unique_lock<std::mutex> lock(mtx);

    // Perform work...

    lock.unlock();  // Manual unlock

    // Other work...

    lock.lock();    // Re-acquire

    // lock is automatically released in destructor (if locked)
}

// Deferred lock
void deferred_locking() {
    std::unique_lock<std::mutex> lock(mtx, std::defer_lock);
    // Lock not acquired yet

    // ... preparation work ...

    lock.lock();  // Now acquire lock
}
```

### std::scoped_lock (C++17, Multiple Mutexes)

```cpp
#include <mutex>

std::mutex mtx1, mtx2;

void transfer() {
    // Acquire multiple mutexes without deadlock
    std::scoped_lock lock(mtx1, mtx2);

    // Perform work...
}
```

### std::shared_mutex (Read-Write Lock, C++17)

```cpp
#include <shared_mutex>
#include <mutex>

class ThreadSafeCounter {
    int value = 0;
    mutable std::shared_mutex mtx;

public:
    // Read: Multiple threads can access simultaneously
    int get() const {
        std::shared_lock lock(mtx);
        return value;
    }

    // Write: Exclusive access
    void increment() {
        std::unique_lock lock(mtx);
        ++value;
    }
};
```

---

## Condition Variables

### std::condition_variable

```cpp
#include <iostream>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>

std::queue<int> dataQueue;
std::mutex mtx;
std::condition_variable cv;
bool finished = false;

void producer() {
    for (int i = 0; i < 10; ++i) {
        {
            std::lock_guard<std::mutex> lock(mtx);
            dataQueue.push(i);
            std::cout << "Produced: " << i << "\n";
        }
        cv.notify_one();  // Wake up waiting thread
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    {
        std::lock_guard<std::mutex> lock(mtx);
        finished = true;
    }
    cv.notify_all();  // Wake up all waiting threads
}

void consumer() {
    while (true) {
        std::unique_lock<std::mutex> lock(mtx);

        // Wait with predicate (prevents spurious wakeup)
        cv.wait(lock, [] {
            return !dataQueue.empty() || finished;
        });

        if (dataQueue.empty() && finished) {
            break;
        }

        int value = dataQueue.front();
        dataQueue.pop();
        std::cout << "Consumed: " << value << "\n";
    }
}

int main() {
    std::thread prod(producer);
    std::thread cons(consumer);

    prod.join();
    cons.join();

    return 0;
}
```

### wait_for and wait_until

```cpp
#include <condition_variable>
#include <chrono>

std::condition_variable cv;
std::mutex mtx;
bool ready = false;

void waiter() {
    std::unique_lock<std::mutex> lock(mtx);

    // Wait with timeout
    if (cv.wait_for(lock, std::chrono::seconds(5), [] { return ready; })) {
        std::cout << "Ready!\n";
    } else {
        std::cout << "Timeout!\n";
    }
}
```

---

## Atomic Operations

### std::atomic

```cpp
#include <iostream>
#include <thread>
#include <atomic>

std::atomic<int> counter(0);

void increment() {
    for (int i = 0; i < 100000; ++i) {
        ++counter;  // Atomic increment
    }
}

int main() {
    std::thread t1(increment);
    std::thread t2(increment);

    t1.join();
    t2.join();

    std::cout << "Counter: " << counter << "\n";  // Always 200000
    return 0;
}
```

### Atomic Operations

```cpp
#include <atomic>

std::atomic<int> value(0);

void atomic_operations() {
    // Basic operations
    value.store(10);         // Store
    int v = value.load();    // Load
    int old = value.exchange(20);  // Exchange

    // Arithmetic operations
    value++;
    value--;
    value += 5;
    value.fetch_add(3);      // Add and return previous value
    value.fetch_sub(2);

    // Compare and swap (CAS)
    int expected = 20;
    value.compare_exchange_strong(expected, 30);
    // If value equals expected, change to 30
    // Otherwise, store current value in expected
}
```

### Memory Order

```cpp
#include <atomic>

std::atomic<int> x(0);
std::atomic<int> y(0);

void thread1() {
    x.store(1, std::memory_order_release);
}

void thread2() {
    while (x.load(std::memory_order_acquire) == 0);
    // If x is 1, all previous writes from thread1 are visible
}
```

| Memory Order | Description |
|--------------|-------------|
| `relaxed` | Only atomicity guaranteed, no ordering |
| `acquire` | Reads/writes before this operation won't be reordered |
| `release` | Reads/writes after this operation won't be reordered |
| `acq_rel` | acquire + release |
| `seq_cst` | Sequential consistency (default, strongest) |

---

## Asynchronous Programming

### std::async

```cpp
#include <iostream>
#include <future>
#include <chrono>

int compute(int x) {
    std::this_thread::sleep_for(std::chrono::seconds(2));
    return x * x;
}

int main() {
    // Asynchronous execution
    std::future<int> result = std::async(std::launch::async, compute, 10);

    std::cout << "Computing...\n";

    // Can perform other work

    // Wait and get result
    int value = result.get();
    std::cout << "Result: " << value << "\n";

    return 0;
}
```

### Launch Policies

```cpp
#include <future>

// async: Execute immediately in new thread
auto f1 = std::async(std::launch::async, task);

// deferred: Execute in current thread when get() is called
auto f2 = std::async(std::launch::deferred, task);

// Default: System decides
auto f3 = std::async(task);
```

### std::future and std::promise

```cpp
#include <iostream>
#include <thread>
#include <future>

void producer(std::promise<int>& prom) {
    std::this_thread::sleep_for(std::chrono::seconds(1));
    prom.set_value(42);  // Set value
}

void consumer(std::future<int>& fut) {
    std::cout << "Waiting for value...\n";
    int value = fut.get();  // Wait for value
    std::cout << "Received: " << value << "\n";
}

int main() {
    std::promise<int> prom;
    std::future<int> fut = prom.get_future();

    std::thread t1(producer, std::ref(prom));
    std::thread t2(consumer, std::ref(fut));

    t1.join();
    t2.join();

    return 0;
}
```

### std::packaged_task

```cpp
#include <iostream>
#include <thread>
#include <future>

int add(int a, int b) {
    return a + b;
}

int main() {
    std::packaged_task<int(int, int)> task(add);
    std::future<int> result = task.get_future();

    std::thread t(std::move(task), 10, 20);

    std::cout << "Result: " << result.get() << "\n";

    t.join();
    return 0;
}
```

### Checking Future Status

```cpp
#include <future>
#include <chrono>

auto fut = std::async(std::launch::async, task);

// Wait with timeout
auto status = fut.wait_for(std::chrono::seconds(1));

if (status == std::future_status::ready) {
    std::cout << "Ready!\n";
} else if (status == std::future_status::timeout) {
    std::cout << "Timeout\n";
} else if (status == std::future_status::deferred) {
    std::cout << "Deferred\n";
}
```

---

## Thread Pool

### Simple Thread Pool Implementation

```cpp
#include <iostream>
#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <future>

class ThreadPool {
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;
    std::mutex mtx;
    std::condition_variable cv;
    bool stop = false;

public:
    explicit ThreadPool(size_t numThreads) {
        for (size_t i = 0; i < numThreads; ++i) {
            workers.emplace_back([this] {
                while (true) {
                    std::function<void()> task;

                    {
                        std::unique_lock<std::mutex> lock(mtx);
                        cv.wait(lock, [this] {
                            return stop || !tasks.empty();
                        });

                        if (stop && tasks.empty()) {
                            return;
                        }

                        task = std::move(tasks.front());
                        tasks.pop();
                    }

                    task();
                }
            });
        }
    }

    template<class F, class... Args>
    auto enqueue(F&& f, Args&&... args)
        -> std::future<typename std::invoke_result<F, Args...>::type>
    {
        using return_type = typename std::invoke_result<F, Args...>::type;

        auto task = std::make_shared<std::packaged_task<return_type()>>(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...)
        );

        std::future<return_type> result = task->get_future();

        {
            std::unique_lock<std::mutex> lock(mtx);
            if (stop) {
                throw std::runtime_error("enqueue on stopped ThreadPool");
            }
            tasks.emplace([task]() { (*task)(); });
        }

        cv.notify_one();
        return result;
    }

    ~ThreadPool() {
        {
            std::unique_lock<std::mutex> lock(mtx);
            stop = true;
        }
        cv.notify_all();

        for (std::thread& worker : workers) {
            worker.join();
        }
    }
};

// Usage example
int main() {
    ThreadPool pool(4);

    std::vector<std::future<int>> results;

    for (int i = 0; i < 8; ++i) {
        results.emplace_back(
            pool.enqueue([i] {
                std::this_thread::sleep_for(std::chrono::seconds(1));
                return i * i;
            })
        );
    }

    for (auto& result : results) {
        std::cout << result.get() << " ";
    }
    std::cout << "\n";

    return 0;
}
```

---

## Common Problems and Solutions

### Deadlock

```cpp
// Problem: Circular wait
std::mutex m1, m2;

void thread1() {
    std::lock_guard<std::mutex> l1(m1);
    std::lock_guard<std::mutex> l2(m2);  // Waiting for m2
}

void thread2() {
    std::lock_guard<std::mutex> l2(m2);
    std::lock_guard<std::mutex> l1(m1);  // Waiting for m1
}

// Solution: Use std::scoped_lock
void thread1_fixed() {
    std::scoped_lock lock(m1, m2);  // Acquire simultaneously
}

void thread2_fixed() {
    std::scoped_lock lock(m1, m2);
}
```

### Livelock

```cpp
// Situation where two threads keep yielding to each other
// Solution: Random backoff, priority setting
```

### Starvation

```cpp
// Situation where a specific thread never acquires resources
// Solution: Use fair locks, priority queues
```

### Thread-Safe Singleton

```cpp
#include <mutex>

class Singleton {
    static Singleton* instance;
    static std::once_flag initFlag;

    Singleton() = default;

public:
    static Singleton& getInstance() {
        std::call_once(initFlag, [] {
            instance = new Singleton();
        });
        return *instance;
    }
};

Singleton* Singleton::instance = nullptr;
std::once_flag Singleton::initFlag;

// Or use C++11 static local variable (thread-safe)
class Singleton2 {
    Singleton2() = default;

public:
    static Singleton2& getInstance() {
        static Singleton2 instance;
        return instance;
    }
};
```

---

## Exercises

### Problem 1: Parallel Sum

Calculate the sum of a vector in parallel using multiple threads.

<details>
<summary>Show Answer</summary>

```cpp
#include <iostream>
#include <vector>
#include <thread>
#include <numeric>

long long parallelSum(const std::vector<int>& data, int numThreads) {
    std::vector<long long> partialSums(numThreads);
    std::vector<std::thread> threads;

    size_t chunkSize = data.size() / numThreads;

    for (int i = 0; i < numThreads; ++i) {
        size_t start = i * chunkSize;
        size_t end = (i == numThreads - 1) ? data.size() : start + chunkSize;

        threads.emplace_back([&data, &partialSums, i, start, end] {
            partialSums[i] = std::accumulate(
                data.begin() + start, data.begin() + end, 0LL
            );
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    return std::accumulate(partialSums.begin(), partialSums.end(), 0LL);
}

int main() {
    std::vector<int> data(10000000, 1);
    std::cout << "Sum: " << parallelSum(data, 4) << "\n";
    return 0;
}
```

</details>

### Problem 2: Producer-Consumer

Implement a thread-safe queue with multiple producers and consumers.

<details>
<summary>Show Answer</summary>

```cpp
#include <queue>
#include <mutex>
#include <condition_variable>
#include <optional>

template<typename T>
class ThreadSafeQueue {
    std::queue<T> queue;
    mutable std::mutex mtx;
    std::condition_variable cv;

public:
    void push(T value) {
        std::lock_guard<std::mutex> lock(mtx);
        queue.push(std::move(value));
        cv.notify_one();
    }

    std::optional<T> pop() {
        std::unique_lock<std::mutex> lock(mtx);
        cv.wait(lock, [this] { return !queue.empty(); });

        T value = std::move(queue.front());
        queue.pop();
        return value;
    }

    bool tryPop(T& value) {
        std::lock_guard<std::mutex> lock(mtx);
        if (queue.empty()) return false;

        value = std::move(queue.front());
        queue.pop();
        return true;
    }
};
```

</details>

---

## Next Step

- [17_CPP20_Advanced.md](./17_CPP20_Advanced.md) - Concepts, Ranges, Coroutines

---

## References

- [C++ Concurrency in Action (book)](https://www.manning.com/books/c-plus-plus-concurrency-in-action-second-edition)
- [cppreference - Thread support](https://en.cppreference.com/w/cpp/thread)
- [C++17 parallel algorithms](https://en.cppreference.com/w/cpp/algorithm#Execution_policies)
