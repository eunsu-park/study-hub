// concurrency_demo.cpp — Latch, barrier, atomic, parallel algorithms (C++20)
// Compile: g++ -std=c++20 -Wall -Wextra -pthread -o concurrency_demo concurrency_demo.cpp

#include <iostream>
#include <thread>
#include <vector>
#include <atomic>
#include <latch>
#include <barrier>
#include <semaphore>
#include <numeric>
#include <algorithm>
#include <chrono>
#include <functional>
#include <mutex>

// --- std::atomic with fetch operations ---
void atomic_demo() {
    std::cout << "=== Atomic Operations ===\n";

    std::atomic<int> counter{0};
    constexpr int N = 10;
    constexpr int ITERS = 10000;

    std::vector<std::thread> threads;
    for (int i = 0; i < N; ++i) {
        threads.emplace_back([&counter]() {
            for (int j = 0; j < ITERS; ++j) {
                counter.fetch_add(1, std::memory_order_relaxed);
            }
        });
    }
    for (auto& t : threads) t.join();

    std::cout << "Expected: " << N * ITERS << '\n';
    std::cout << "Actual:   " << counter.load() << '\n';

    // Compare-and-swap (CAS)
    std::atomic<int> val{10};
    int expected = 10;
    bool swapped = val.compare_exchange_strong(expected, 20);
    std::cout << "CAS: swapped=" << std::boolalpha << swapped
              << " val=" << val.load() << '\n';
}

// --- std::latch (single-use barrier) ---
void latch_demo() {
    std::cout << "\n=== std::latch ===\n";

    constexpr int WORKERS = 4;
    std::latch ready(WORKERS);
    std::latch done(WORKERS);

    auto worker = [&](int id) {
        // Simulate initialization
        std::this_thread::sleep_for(std::chrono::milliseconds(id * 50));
        std::cout << "  Worker " << id << " ready\n";
        ready.count_down();
        ready.wait();  // all workers wait until everyone is ready

        // Do work
        std::cout << "  Worker " << id << " working\n";
        done.count_down();
    };

    std::vector<std::thread> threads;
    for (int i = 0; i < WORKERS; ++i) {
        threads.emplace_back(worker, i);
    }

    done.wait();  // main waits for all workers
    std::cout << "All workers completed\n";

    for (auto& t : threads) t.join();
}

// --- std::barrier (reusable synchronization) ---
void barrier_demo() {
    std::cout << "\n=== std::barrier ===\n";

    constexpr int WORKERS = 3;
    constexpr int PHASES = 3;
    int phase = 0;

    std::barrier sync_point(WORKERS, [&]() noexcept {
        ++phase;
        std::cout << "  --- Phase " << phase << " complete ---\n";
    });

    auto worker = [&](int id) {
        for (int p = 0; p < PHASES; ++p) {
            std::cout << "  Worker " << id << " phase " << (p + 1) << '\n';
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
            sync_point.arrive_and_wait();
        }
    };

    std::vector<std::thread> threads;
    for (int i = 0; i < WORKERS; ++i) {
        threads.emplace_back(worker, i);
    }
    for (auto& t : threads) t.join();
}

// --- std::counting_semaphore ---
void semaphore_demo() {
    std::cout << "\n=== std::counting_semaphore ===\n";

    std::counting_semaphore<3> sem(3);  // max 3 concurrent
    std::mutex print_mtx;

    auto worker = [&](int id) {
        sem.acquire();
        {
            std::lock_guard lock(print_mtx);
            std::cout << "  Worker " << id << " acquired slot\n";
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        {
            std::lock_guard lock(print_mtx);
            std::cout << "  Worker " << id << " releasing slot\n";
        }
        sem.release();
    };

    std::vector<std::thread> threads;
    for (int i = 0; i < 6; ++i) {
        threads.emplace_back(worker, i);
    }
    for (auto& t : threads) t.join();
}

// --- atomic_ref (C++20) ---
void atomic_ref_demo() {
    std::cout << "\n=== std::atomic_ref ===\n";
    int value = 0;

    std::vector<std::thread> threads;
    for (int i = 0; i < 4; ++i) {
        threads.emplace_back([&value]() {
            std::atomic_ref<int> ref(value);
            for (int j = 0; j < 1000; ++j) {
                ref.fetch_add(1, std::memory_order_relaxed);
            }
        });
    }
    for (auto& t : threads) t.join();
    std::cout << "Expected 4000, got " << value << '\n';
}

int main() {
    atomic_demo();
    latch_demo();
    barrier_demo();
    semaphore_demo();
    atomic_ref_demo();
    return 0;
}
