// Exercise 10: C++20 Coroutines
// Practice generators, lazy sequences, and async coroutines.
// Compile: g++ -std=c++20 -Wall -Wextra -fcoroutines -o ex10 10_cpp20_coroutines.cpp && ./ex10

#include <iostream>
#include <coroutine>
#include <vector>
#include <string>
#include <cassert>
#include <cmath>

// --- Provided: Generator<T> type ---
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
    Generator(const Generator&) = delete;
    Generator(Generator&& o) noexcept : handle_(o.handle_) { o.handle_ = nullptr; }

    bool next() {
        if (handle_ && !handle_.done()) { handle_.resume(); return !handle_.done(); }
        return false;
    }
    T value() const { return handle_.promise().current_value; }

private:
    handle_type handle_;
};

// Helper: collect all values from a generator
template <typename T>
std::vector<T> collect(Generator<T> gen) {
    std::vector<T> result;
    while (gen.next()) result.push_back(gen.value());
    return result;
}

// TODO 1: Implement a generator that yields prime numbers up to a limit.

// Generator<int> primes(int limit) { ... co_yield ... }

// TODO 2: Implement a generator that yields Collatz sequence for a starting number.
// Collatz: if even -> n/2, if odd -> 3n+1. Stop when n reaches 1.

// Generator<long long> collatz(long long start) { ... co_yield ... }

// TODO 3: Implement a generator that flattens a 2D vector.
// flatten({{1,2},{3},{4,5,6}}) -> yields 1, 2, 3, 4, 5, 6

// Generator<int> flatten(const std::vector<std::vector<int>>& v) { ... co_yield ... }

// TODO 4: Implement a "zip" generator that yields pairs from two generators.
// Stops when either generator is exhausted.

// template <typename T, typename U>
// Generator<std::pair<T, U>> zip(Generator<T> a, Generator<U> b) { ... }

// TODO 5: Implement a "filter" generator adaptor that only yields values
// matching a predicate.

// template <typename T, typename Pred>
// Generator<T> filter(Generator<T> gen, Pred pred) { ... }

int main() {
    std::cout << "=== Exercise 10: C++20 Coroutines ===\n\n";

    // Test 1: Primes
    // auto prime_list = collect(primes(30));
    // assert(prime_list == std::vector<int>({2, 3, 5, 7, 11, 13, 17, 19, 23, 29}));
    // std::cout << "Test 1 passed: primes generator\n";

    // Test 2: Collatz
    // auto seq = collect(collatz(6));
    // assert(seq.front() == 6 && seq.back() == 1);
    // std::cout << "Test 2 passed: Collatz sequence, length=" << seq.size() << '\n';

    // Test 3: Flatten
    // auto flat = collect(flatten({{1,2},{3},{4,5,6}}));
    // assert((flat == std::vector<int>{1,2,3,4,5,6}));
    // std::cout << "Test 3 passed: flatten\n";

    // Test 5: Filter
    // auto evens = collect(filter(primes(50), [](int n) { return n % 2 == 0; }));
    // assert(evens.size() == 1 && evens[0] == 2);
    // std::cout << "Test 5 passed: filter adaptor\n";

    std::cout << "Uncomment tests as you implement each generator.\n";
    return 0;
}
