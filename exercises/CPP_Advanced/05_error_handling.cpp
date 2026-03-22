// Exercise 05: Error Handling
// Practice exception safety, noexcept, optional, expected, error strategies.
// Compile: g++ -std=c++20 -Wall -Wextra -o ex05 05_error_handling.cpp && ./ex05

#include <iostream>
#include <string>
#include <vector>
#include <optional>
#include <variant>
#include <stdexcept>
#include <cassert>
#include <fstream>
#include <cmath>

// TODO 1: Implement a Result<T, E> type using std::variant.
// Support: is_ok(), is_err(), value(), error(), map(), and_then()

// template <typename T, typename E>
// class Result {
// public:
//     static Result ok(T val) { ... }
//     static Result err(E e) { ... }
//     bool is_ok() const { ... }
//     bool is_err() const { ... }
//     T value() const { ... }     // throws if err
//     E error() const { ... }     // throws if ok
//     template <typename F>
//     auto map(F f) const -> Result<decltype(f(std::declval<T>())), E> { ... }
// };

// TODO 2: Rewrite these functions to return Result instead of throwing:
//   - parse_int(string) -> Result<int, string>
//   - safe_divide(double, double) -> Result<double, string>
//   - read_file(string path) -> Result<string, string>

// TODO 3: Implement a function that chains multiple Result operations:
// parse_int(s) -> safe_divide(result, 2.0) -> format as string
// Use map() and and_then() for clean chaining.

// std::string process_input(const std::string& s) { ... }

// TODO 4: Implement exception-safe push_back for a custom container.
// Use the copy-and-swap idiom to provide strong exception guarantee.

class SafeVector {
    int* data_ = nullptr;
    size_t size_ = 0;
    size_t capacity_ = 0;

public:
    SafeVector() = default;
    // TODO: Implement destructor, copy ctor, move ctor, assignments

    // TODO: push_back with strong exception guarantee
    // void push_back(int val) { ... }

    size_t size() const { return size_; }
    int operator[](size_t i) const { return data_[i]; }
};

// TODO 5: Write a noexcept-safe swap and demonstrate conditional noexcept.

// template <typename T>
// void safe_swap(T& a, T& b) noexcept(noexcept(T(std::move(a)))) { ... }

int main() {
    std::cout << "=== Exercise 05: Error Handling ===\n\n";

    // Test 1: Result type
    // auto ok = Result<int, std::string>::ok(42);
    // assert(ok.is_ok() && ok.value() == 42);
    // auto err = Result<int, std::string>::err("bad input");
    // assert(err.is_err() && err.error() == "bad input");
    // std::cout << "Test 1 passed: Result type\n";

    // Test 2: Result-returning functions
    // auto r1 = parse_int("42");
    // assert(r1.is_ok() && r1.value() == 42);
    // auto r2 = parse_int("abc");
    // assert(r2.is_err());
    // auto r3 = safe_divide(10.0, 0.0);
    // assert(r3.is_err());
    // std::cout << "Test 2 passed: Result functions\n";

    // Test 4: SafeVector
    // SafeVector sv;
    // for (int i = 0; i < 100; ++i) sv.push_back(i);
    // assert(sv.size() == 100);
    // assert(sv[50] == 50);
    // std::cout << "Test 4 passed: SafeVector\n";

    std::cout << "Uncomment tests as you implement each part.\n";
    return 0;
}
