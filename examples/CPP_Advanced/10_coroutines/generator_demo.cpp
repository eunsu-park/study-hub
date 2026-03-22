// generator_demo.cpp — Simple generator using co_yield (C++20 coroutines)
// Compile: g++ -std=c++20 -Wall -Wextra -fcoroutines -o generator_demo generator_demo.cpp

#include <iostream>
#include <coroutine>
#include <optional>
#include <vector>
#include <cmath>

// --- Minimal Generator<T> coroutine type ---
template <typename T>
class Generator {
public:
    struct promise_type {
        T current_value;

        Generator get_return_object() {
            return Generator{std::coroutine_handle<promise_type>::from_promise(*this)};
        }

        std::suspend_always initial_suspend() { return {}; }
        std::suspend_always final_suspend() noexcept { return {}; }

        std::suspend_always yield_value(T value) {
            current_value = std::move(value);
            return {};
        }

        void return_void() {}
        void unhandled_exception() { std::terminate(); }
    };

    using handle_type = std::coroutine_handle<promise_type>;

    explicit Generator(handle_type h) : handle_(h) {}
    ~Generator() { if (handle_) handle_.destroy(); }

    // Non-copyable, movable
    Generator(const Generator&) = delete;
    Generator& operator=(const Generator&) = delete;
    Generator(Generator&& other) noexcept : handle_(other.handle_) {
        other.handle_ = nullptr;
    }

    // Iterator-like interface
    bool next() {
        if (handle_ && !handle_.done()) {
            handle_.resume();
            return !handle_.done();
        }
        return false;
    }

    T value() const { return handle_.promise().current_value; }

private:
    handle_type handle_;
};

// --- Generator functions ---

// Fibonacci sequence
Generator<long long> fibonacci(int count) {
    long long a = 0, b = 1;
    for (int i = 0; i < count; ++i) {
        co_yield a;
        auto next = a + b;
        a = b;
        b = next;
    }
}

// Range generator
Generator<int> range(int start, int end, int step = 1) {
    for (int i = start; i < end; i += step) {
        co_yield i;
    }
}

// Filtered generator: prime numbers
Generator<int> primes(int limit) {
    auto is_prime = [](int n) {
        if (n < 2) return false;
        for (int i = 2; i <= static_cast<int>(std::sqrt(n)); ++i) {
            if (n % i == 0) return false;
        }
        return true;
    };

    for (int n = 2; n <= limit; ++n) {
        if (is_prime(n)) {
            co_yield n;
        }
    }
}

// Infinite counter
Generator<int> counter(int start = 0) {
    int i = start;
    while (true) {
        co_yield i++;
    }
}

int main() {
    // Fibonacci
    std::cout << "=== Fibonacci (first 15) ===\n";
    auto fib = fibonacci(15);
    while (fib.next()) {
        std::cout << fib.value() << ' ';
    }
    std::cout << '\n';

    // Range
    std::cout << "\n=== Range(0, 20, 3) ===\n";
    auto r = range(0, 20, 3);
    while (r.next()) {
        std::cout << r.value() << ' ';
    }
    std::cout << '\n';

    // Primes
    std::cout << "\n=== Primes up to 50 ===\n";
    auto p = primes(50);
    while (p.next()) {
        std::cout << p.value() << ' ';
    }
    std::cout << '\n';

    // Infinite counter (take first 10)
    std::cout << "\n=== Infinite Counter (take 10) ===\n";
    auto c = counter(100);
    for (int i = 0; i < 10 && c.next(); ++i) {
        std::cout << c.value() << ' ';
    }
    std::cout << '\n';

    // Collecting into vector
    std::cout << "\n=== Collect into vector ===\n";
    std::vector<long long> fib_vec;
    auto fib2 = fibonacci(10);
    while (fib2.next()) {
        fib_vec.push_back(fib2.value());
    }
    std::cout << "Collected " << fib_vec.size() << " values: ";
    for (auto v : fib_vec) std::cout << v << ' ';
    std::cout << '\n';

    return 0;
}
