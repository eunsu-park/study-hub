// Exercise 13: Advanced Concurrency
// Practice latch, barrier, semaphore, atomic operations, lock-free structures.
// Compile: g++ -std=c++20 -Wall -Wextra -pthread -o ex13 13_concurrency_advanced.cpp && ./ex13

#include <iostream>
#include <thread>
#include <vector>
#include <atomic>
#include <latch>
#include <barrier>
#include <semaphore>
#include <mutex>
#include <queue>
#include <functional>
#include <future>
#include <numeric>
#include <cassert>
#include <chrono>

// TODO 1: Implement a thread-safe bounded queue using std::counting_semaphore.
// - put(T) blocks if queue is full
// - take() -> T blocks if queue is empty
// - try_put(T) returns false if full
// - size() returns current size

// template <typename T, size_t MaxSize = 16>
// class BoundedQueue {
//     // TODO: Implement using two semaphores (empty_slots, full_slots) + mutex
// };

// TODO 2: Implement parallel_map using std::latch for synchronization.
// Apply a function to each element of a vector in parallel.
// Use a latch to wait for all workers to finish.

// template <typename T, typename Func>
// std::vector<T> parallel_map(const std::vector<T>& input, Func f, int num_threads = 4) { ... }

// TODO 3: Implement a parallel pipeline with 3 stages using std::barrier.
// Stage 1: Generate data
// Stage 2: Process data
// Stage 3: Aggregate results
// All stages synchronize between iterations.

// void pipeline_demo() { ... }

// TODO 4: Implement a lock-free stack using std::atomic and compare_exchange.

// template <typename T>
// class LockFreeStack {
//     struct Node { T data; Node* next; };
//     std::atomic<Node*> head_{nullptr};
// public:
//     void push(T val) { ... }
//     std::optional<T> pop() { ... }
// };

// TODO 5: Implement a simple thread pool with a fixed number of workers.
// - submit(Func) -> std::future<ReturnType>
// - Workers pick tasks from a shared queue.
// - shutdown() stops all workers.

// class ThreadPool {
//     // TODO: Implement
// };

int main() {
    std::cout << "=== Exercise 13: Advanced Concurrency ===\n\n";

    // Test 1: BoundedQueue
    // BoundedQueue<int, 4> bq;
    // bq.put(1); bq.put(2); bq.put(3);
    // assert(bq.size() == 3);
    // assert(bq.take() == 1);
    // assert(bq.size() == 2);
    // std::cout << "Test 1 passed: BoundedQueue\n";

    // Test 2: parallel_map
    // std::vector<int> input(100);
    // std::iota(input.begin(), input.end(), 0);
    // auto result = parallel_map(input, [](int x) { return x * x; });
    // assert(result[10] == 100);
    // assert(result[50] == 2500);
    // std::cout << "Test 2 passed: parallel_map\n";

    // Test 4: LockFreeStack
    // LockFreeStack<int> lfs;
    // constexpr int N = 1000;
    // std::vector<std::thread> threads;
    // for (int i = 0; i < 4; ++i) {
    //     threads.emplace_back([&lfs, i]() {
    //         for (int j = 0; j < N; ++j) lfs.push(i * N + j);
    //     });
    // }
    // for (auto& t : threads) t.join();
    // int count = 0;
    // while (lfs.pop().has_value()) ++count;
    // assert(count == 4 * N);
    // std::cout << "Test 4 passed: LockFreeStack\n";

    // Test 5: ThreadPool
    // ThreadPool pool(4);
    // std::vector<std::future<int>> futures;
    // for (int i = 0; i < 20; ++i) {
    //     futures.push_back(pool.submit([i]() { return i * i; }));
    // }
    // for (int i = 0; i < 20; ++i) {
    //     assert(futures[i].get() == i * i);
    // }
    // pool.shutdown();
    // std::cout << "Test 5 passed: ThreadPool\n";

    std::cout << "Uncomment tests as you implement each part.\n";
    return 0;
}
