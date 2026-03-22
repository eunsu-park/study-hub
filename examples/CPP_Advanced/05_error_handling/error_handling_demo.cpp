// error_handling_demo.cpp — Exception safety, noexcept, error strategies
// Compile: g++ -std=c++20 -Wall -Wextra -o error_handling_demo error_handling_demo.cpp

#include <iostream>
#include <string>
#include <vector>
#include <optional>
#include <variant>
#include <expected>   // C++23, use std::variant fallback if unavailable
#include <stdexcept>
#include <system_error>
#include <cassert>

// --- noexcept specification ---
int safe_add(int a, int b) noexcept {
    return a + b;
}

void might_throw(bool do_throw) {
    if (do_throw) throw std::runtime_error("boom");
}

// --- std::optional for "no value" ---
std::optional<int> find_index(const std::vector<int>& v, int target) {
    for (size_t i = 0; i < v.size(); ++i) {
        if (v[i] == target) return static_cast<int>(i);
    }
    return std::nullopt;
}

// --- Error code pattern (like errno / std::error_code) ---
enum class ParseError {
    None = 0,
    EmptyInput,
    InvalidFormat,
    OutOfRange
};

struct ParseResult {
    int value;
    ParseError error;
};

ParseResult parse_int(const std::string& s) {
    if (s.empty()) return {0, ParseError::EmptyInput};
    try {
        size_t pos = 0;
        int val = std::stoi(s, &pos);
        if (pos != s.size()) return {0, ParseError::InvalidFormat};
        return {val, ParseError::None};
    } catch (const std::out_of_range&) {
        return {0, ParseError::OutOfRange};
    } catch (...) {
        return {0, ParseError::InvalidFormat};
    }
}

// --- std::variant as Result<T, E> (pre-C++23) ---
template <typename T, typename E>
using Result = std::variant<T, E>;

Result<double, std::string> safe_sqrt(double x) {
    if (x < 0.0) return std::string("Cannot sqrt negative number");
    return std::sqrt(x);
}

// --- Strong exception guarantee (copy-and-swap) ---
class SafeBuffer {
    std::vector<int> data_;

public:
    explicit SafeBuffer(std::initializer_list<int> init) : data_(init) {}

    // Strong guarantee: either succeeds completely or no change
    void append(const std::vector<int>& other) {
        std::vector<int> tmp = data_;  // copy
        tmp.insert(tmp.end(), other.begin(), other.end());  // may throw
        data_.swap(tmp);  // noexcept — commit
    }

    void print() const {
        for (auto x : data_) std::cout << x << ' ';
        std::cout << '\n';
    }
};

int main() {
    // noexcept
    std::cout << "=== noexcept ===\n";
    std::cout << "safe_add noexcept? " << std::boolalpha
              << noexcept(safe_add(1, 2)) << '\n';
    std::cout << "might_throw noexcept? "
              << noexcept(might_throw(false)) << '\n';

    // std::optional
    std::cout << "\n=== std::optional ===\n";
    std::vector<int> nums = {10, 20, 30, 40, 50};
    if (auto idx = find_index(nums, 30)) {
        std::cout << "Found 30 at index " << *idx << '\n';
    }
    if (auto idx = find_index(nums, 99)) {
        std::cout << "Found 99\n";
    } else {
        std::cout << "99 not found (nullopt)\n";
    }

    // Error code pattern
    std::cout << "\n=== Error Code Pattern ===\n";
    for (const auto& input : {"42", "", "abc", "99999999999"}) {
        auto [val, err] = parse_int(input);
        if (err == ParseError::None) {
            std::cout << "\"" << input << "\" -> " << val << '\n';
        } else {
            std::cout << "\"" << input << "\" -> error code "
                      << static_cast<int>(err) << '\n';
        }
    }

    // variant-based Result
    std::cout << "\n=== variant<T, E> as Result ===\n";
    for (double x : {25.0, -4.0, 0.0}) {
        auto result = safe_sqrt(x);
        if (auto* val = std::get_if<double>(&result)) {
            std::cout << "sqrt(" << x << ") = " << *val << '\n';
        } else {
            std::cout << "sqrt(" << x << "): "
                      << std::get<std::string>(result) << '\n';
        }
    }

    // Strong exception guarantee
    std::cout << "\n=== Strong Exception Guarantee ===\n";
    SafeBuffer buf = {1, 2, 3};
    buf.append({4, 5, 6});
    std::cout << "Buffer: ";
    buf.print();

    // std::error_code
    std::cout << "\n=== std::error_code ===\n";
    std::error_code ec = std::make_error_code(std::errc::no_such_file_or_directory);
    std::cout << "Category: " << ec.category().name() << '\n';
    std::cout << "Message: " << ec.message() << '\n';
    std::cout << "Value: " << ec.value() << '\n';

    return 0;
}
