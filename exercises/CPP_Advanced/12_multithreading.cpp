/*
 * Exercises for Lesson 16: Multithreading and Concurrency
 * Topic: CPP
 * Compile: g++ -std=c++17 -Wall -Wextra -pthread -o ex16 16_multithreading_concurrency.cpp
 */
#include <iostream>
#include <vector>
#include <string>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <numeric>
#include <optional>
#include <functional>
#include <atomic>
#include <chrono>
using namespace std;

// === Exercise 1: Parallel Sum ===
// Problem: Calculate the sum of a vector in parallel using multiple threads.
//          Split the work into chunks, compute partial sums in parallel,
//          then combine the results.

long long parallelSum(const vector<int>& data, int numThreads) {
    // Each thread computes its chunk's partial sum into this array.
    // No synchronization needed because each thread writes to its own index.
    vector<long long> partialSums(numThreads, 0);
    vector<thread> threads;

    size_t chunkSize = data.size() / numThreads;

    for (int i = 0; i < numThreads; ++i) {
        size_t start = i * chunkSize;
        // Last thread handles any remainder elements
        size_t end = (i == numThreads - 1) ? data.size() : start + chunkSize;

        threads.emplace_back([&data, &partialSums, i, start, end] {
            partialSums[i] = accumulate(
                data.begin() + start,
                data.begin() + end,
                0LL
            );
        });
    }

    // Wait for all threads to complete
    for (auto& t : threads) {
        t.join();
    }

    // Combine partial sums (done in the main thread)
    return accumulate(partialSums.begin(), partialSums.end(), 0LL);
}

void exercise_1() {
    cout << "=== Exercise 1: Parallel Sum ===" << endl;

    // Create a large vector
    const int SIZE = 10'000'000;
    vector<int> data(SIZE, 1);  // 10 million 1's -> sum should be 10,000,000

    // Sequential sum for comparison
    auto seqStart = chrono::high_resolution_clock::now();
    long long seqSum = accumulate(data.begin(), data.end(), 0LL);
    auto seqEnd = chrono::high_resolution_clock::now();
    auto seqMs = chrono::duration_cast<chrono::microseconds>(seqEnd - seqStart).count();

    // Parallel sum
    auto parStart = chrono::high_resolution_clock::now();
    long long parSum = parallelSum(data, 4);
    auto parEnd = chrono::high_resolution_clock::now();
    auto parMs = chrono::duration_cast<chrono::microseconds>(parEnd - parStart).count();

    cout << "  Sequential sum: " << seqSum << " (" << seqMs << " us)" << endl;
    cout << "  Parallel sum:   " << parSum << " (" << parMs << " us)" << endl;
    cout << "  Results match:  " << boolalpha << (seqSum == parSum) << endl;

    // Test with different values
    vector<int> data2(1'000'000);
    iota(data2.begin(), data2.end(), 1);  // 1, 2, ..., 1000000
    long long expected = static_cast<long long>(1'000'000) * 1'000'001 / 2;
    long long actual = parallelSum(data2, 4);
    cout << "  Sum 1..1M: expected=" << expected << ", got=" << actual
         << ", match=" << (expected == actual) << endl;
}

// === Exercise 2: Producer-Consumer ===
// Problem: Implement a thread-safe queue with multiple producers and consumers.

template<typename T>
class ThreadSafeQueue {
    queue<T> queue_;
    mutable mutex mtx_;
    condition_variable cv_;
    bool shutdown_ = false;

public:
    // Push an item and notify one waiting consumer
    void push(T value) {
        {
            lock_guard<mutex> lock(mtx_);
            queue_.push(std::move(value));
        }
        cv_.notify_one();
    }

    // Blocking pop: waits until an item is available or shutdown is signaled.
    // Returns nullopt on shutdown with empty queue.
    optional<T> pop() {
        unique_lock<mutex> lock(mtx_);
        cv_.wait(lock, [this] { return !queue_.empty() || shutdown_; });

        if (queue_.empty()) {
            return nullopt;  // Shutdown signaled and queue is empty
        }

        T value = std::move(queue_.front());
        queue_.pop();
        return value;
    }

    // Non-blocking try_pop
    bool tryPop(T& value) {
        lock_guard<mutex> lock(mtx_);
        if (queue_.empty()) return false;

        value = std::move(queue_.front());
        queue_.pop();
        return true;
    }

    // Signal all waiting threads to wake up and exit
    void requestShutdown() {
        {
            lock_guard<mutex> lock(mtx_);
            shutdown_ = true;
        }
        cv_.notify_all();
    }

    size_t size() const {
        lock_guard<mutex> lock(mtx_);
        return queue_.size();
    }
};

void exercise_2() {
    cout << "\n=== Exercise 2: Producer-Consumer ===" << endl;

    ThreadSafeQueue<int> queue;
    atomic<int> totalProduced{0};
    atomic<int> totalConsumed{0};

    const int NUM_PRODUCERS = 3;
    const int NUM_CONSUMERS = 2;
    const int ITEMS_PER_PRODUCER = 10;

    vector<thread> producers;
    vector<thread> consumers;

    // Producers: each pushes ITEMS_PER_PRODUCER items
    for (int p = 0; p < NUM_PRODUCERS; p++) {
        producers.emplace_back([&queue, &totalProduced, p] {
            for (int i = 0; i < ITEMS_PER_PRODUCER; i++) {
                int value = p * 100 + i;
                queue.push(value);
                totalProduced++;
            }
        });
    }

    // Consumers: pop items until shutdown is signaled
    for (int c = 0; c < NUM_CONSUMERS; c++) {
        consumers.emplace_back([&queue, &totalConsumed, c] {
            while (true) {
                auto item = queue.pop();
                if (!item.has_value()) break;  // Shutdown
                totalConsumed++;
            }
        });
    }

    // Wait for producers to finish
    for (auto& t : producers) {
        t.join();
    }

    // Give consumers time to drain the queue, then signal shutdown
    // In a real system, we'd use a more sophisticated shutdown protocol
    while (queue.size() > 0) {
        this_thread::yield();
    }
    queue.requestShutdown();

    // Wait for consumers to finish
    for (auto& t : consumers) {
        t.join();
    }

    cout << "  Producers: " << NUM_PRODUCERS
         << " x " << ITEMS_PER_PRODUCER << " items" << endl;
    cout << "  Consumers: " << NUM_CONSUMERS << endl;
    cout << "  Total produced: " << totalProduced.load() << endl;
    cout << "  Total consumed: " << totalConsumed.load() << endl;
    cout << "  All consumed: " << boolalpha
         << (totalProduced.load() == totalConsumed.load()) << endl;
}

int main() {
    exercise_1();
    exercise_2();
    cout << "\nAll exercises completed!" << endl;
    return 0;
}
