/**
 * Example: C++23 Features
 * Topic: CPP – Lesson 21
 *
 * Demonstrates key C++23 features:
 *   1. std::expected for error handling
 *   2. Deducing this (explicit object parameter)
 *   3. std::print / std::println
 *   4. std::mdspan for multidimensional views
 *   5. std::generator for lazy sequences
 *   6. New range adaptors (zip, enumerate, chunk)
 *
 * Compile: g++ -std=c++23 -o cpp23 09_cpp23.cpp
 * Note: Requires GCC 14+ or Clang 18+ for full C++23 support.
 *       Individual features may require different compiler versions.
 */

#include <iostream>
#include <vector>
#include <string>
#include <numeric>
#include <algorithm>
#include <cassert>

// ============================================================
// Feature guards: only compile sections the compiler supports
// ============================================================

#if __has_include(<expected>) && __cplusplus >= 202302L
#include <expected>
#define HAS_EXPECTED 1
#else
#define HAS_EXPECTED 0
#endif

#if __has_include(<print>) && __cplusplus >= 202302L
#include <print>
#define HAS_PRINT 1
#else
#define HAS_PRINT 0
#endif

#if __has_include(<mdspan>) && __cplusplus >= 202302L
#include <mdspan>
#define HAS_MDSPAN 1
#else
#define HAS_MDSPAN 0
#endif

#if __has_include(<generator>) && __cplusplus >= 202302L
#include <generator>
#define HAS_GENERATOR 1
#else
#define HAS_GENERATOR 0
#endif


// ============================================================
// Demo 1: std::expected
// ============================================================
#if HAS_EXPECTED

enum class ParseError {
    empty_input,
    invalid_format,
    overflow
};

std::string to_string(ParseError e) {
    switch (e) {
        case ParseError::empty_input:    return "empty input";
        case ParseError::invalid_format: return "invalid format";
        case ParseError::overflow:       return "overflow";
    }
    return "unknown";
}

std::expected<int, ParseError> parse_int(std::string_view sv) {
    if (sv.empty())
        return std::unexpected(ParseError::empty_input);

    int result = 0;
    bool negative = false;
    size_t i = 0;

    if (sv[0] == '-') { negative = true; i = 1; }
    if (i >= sv.size()) return std::unexpected(ParseError::invalid_format);

    for (; i < sv.size(); ++i) {
        if (sv[i] < '0' || sv[i] > '9')
            return std::unexpected(ParseError::invalid_format);
        result = result * 10 + (sv[i] - '0');
    }

    return negative ? -result : result;
}

// Monadic chaining: validate that value is positive
std::expected<int, ParseError> validate_positive(int val) {
    if (val <= 0)
        return std::unexpected(ParseError::invalid_format);
    return val;
}

void demo_expected() {
    std::cout << "=== Demo 1: std::expected ===\n\n";

    // Success case
    auto r1 = parse_int("42");
    if (r1) {
        std::cout << "  parse_int(\"42\") = " << *r1 << "\n";
    }

    // Error case
    auto r2 = parse_int("abc");
    if (!r2) {
        std::cout << "  parse_int(\"abc\") error: "
                  << to_string(r2.error()) << "\n";
    }

    // Monadic chaining with and_then
    auto r3 = parse_int("42")
        .and_then(validate_positive)
        .transform([](int v) { return v * 2; });

    if (r3) {
        std::cout << "  Chained result: " << *r3 << "\n";
    }

    // value_or for default
    int val = parse_int("bad").value_or(-1);
    std::cout << "  parse_int(\"bad\").value_or(-1) = " << val << "\n\n";
}

#else
void demo_expected() {
    std::cout << "=== Demo 1: std::expected (not available) ===\n";
    std::cout << "  Requires C++23 compiler with <expected> support.\n\n";
}
#endif


// ============================================================
// Demo 2: Deducing this
// ============================================================
#if __cplusplus >= 202302L

// Simplified CRTP replacement
struct Printable {
    template<typename Self>
    void print(this const Self& self) {
        std::cout << "  " << self.to_string() << "\n";
    }
};

struct Point : Printable {
    double x, y;
    Point(double x, double y) : x(x), y(y) {}
    std::string to_string() const {
        return "Point(" + std::to_string(x) + ", " + std::to_string(y) + ")";
    }
};

struct Color : Printable {
    int r, g, b;
    Color(int r, int g, int b) : r(r), g(g), b(b) {}
    std::string to_string() const {
        return "Color(" + std::to_string(r) + ", "
               + std::to_string(g) + ", " + std::to_string(b) + ")";
    }
};

// Recursive lambda with deducing this
void demo_deducing_this() {
    std::cout << "=== Demo 2: Deducing this ===\n\n";

    // CRTP replacement
    Point p{1.5, 2.7};
    Color c{255, 128, 0};
    p.print();
    c.print();

    // Recursive lambda
    auto fib = [](this auto self, int n) -> int {
        if (n <= 1) return n;
        return self(n - 1) + self(n - 2);
    };

    std::cout << "  fib(10) = " << fib(10) << "\n\n";
}

#else
void demo_deducing_this() {
    std::cout << "=== Demo 2: Deducing this (not available) ===\n";
    std::cout << "  Requires C++23 compiler.\n\n";
}
#endif


// ============================================================
// Demo 3: std::print
// ============================================================
#if HAS_PRINT

void demo_print() {
    std::cout << "=== Demo 3: std::print / std::println ===\n\n";

    std::println("  Hello, {}!", "C++23");
    std::println("  int={}, double={:.3f}, bool={}", 42, 3.14159, true);
    std::println("  Hex: {:#x}, Oct: {:#o}, Bin: {:#b}", 255, 255, 255);
    std::println("  Right-aligned: {:>10}", "hello");
    std::println("  Center-padded: {:*^15}", "C++23");
    std::println();
}

#else
void demo_print() {
    std::cout << "=== Demo 3: std::print (not available) ===\n";
    std::cout << "  Requires C++23 compiler with <print> support.\n\n";
}
#endif


// ============================================================
// Demo 4: std::mdspan
// ============================================================
#if HAS_MDSPAN

void demo_mdspan() {
    std::cout << "=== Demo 4: std::mdspan ===\n\n";

    // Create a 3×4 matrix view over flat storage
    std::vector<double> data(12);
    std::iota(data.begin(), data.end(), 1.0);

    std::mdspan mat(data.data(), 3, 4);

    std::cout << "  3x4 matrix:\n";
    for (std::size_t i = 0; i < mat.extent(0); ++i) {
        std::cout << "    ";
        for (std::size_t j = 0; j < mat.extent(1); ++j) {
            std::cout << mat[i, j] << "\t";
        }
        std::cout << "\n";
    }

    // Matrix-vector multiply using mdspan
    std::vector<double> vec = {1, 2, 3, 4};
    std::vector<double> result(3, 0.0);

    for (std::size_t i = 0; i < mat.extent(0); ++i) {
        for (std::size_t j = 0; j < mat.extent(1); ++j) {
            result[i] += mat[i, j] * vec[j];
        }
    }

    std::cout << "\n  Matrix × [1,2,3,4] = [";
    for (size_t i = 0; i < result.size(); ++i) {
        std::cout << result[i];
        if (i + 1 < result.size()) std::cout << ", ";
    }
    std::cout << "]\n\n";
}

#else
void demo_mdspan() {
    std::cout << "=== Demo 4: std::mdspan (not available) ===\n";
    std::cout << "  Requires C++23 compiler with <mdspan> support.\n\n";
}
#endif


// ============================================================
// Demo 5: New Range Adaptors (available in some C++23 compilers)
// ============================================================
void demo_ranges() {
    std::cout << "=== Demo 5: C++23 Range Adaptors (conceptual) ===\n\n";

    // These require full C++23 ranges support, which varies by compiler.
    // Shown as conceptual code with manual equivalents.

    std::vector<int> v = {1, 2, 3, 4, 5, 6, 7, 8};

    // Manual "enumerate" equivalent
    std::cout << "  Enumerate:\n";
    for (size_t i = 0; i < v.size(); ++i) {
        std::cout << "    [" << i << "] = " << v[i] << "\n";
    }

    // Manual "chunk" equivalent (groups of 3)
    std::cout << "\n  Chunk(3):\n";
    for (size_t i = 0; i < v.size(); i += 3) {
        std::cout << "    {";
        for (size_t j = i; j < std::min(i + 3, v.size()); ++j) {
            if (j > i) std::cout << ", ";
            std::cout << v[j];
        }
        std::cout << "}\n";
    }

    // Manual "slide" equivalent (window of 3)
    std::cout << "\n  Slide(3):\n";
    for (size_t i = 0; i + 2 < v.size(); ++i) {
        std::cout << "    {" << v[i] << ", " << v[i+1] << ", " << v[i+2] << "}\n";
    }

    std::cout << "\n  With C++23 ranges:\n";
    std::cout << "    for (auto [i, val] : v | std::views::enumerate) ...\n";
    std::cout << "    for (auto chunk : v | std::views::chunk(3)) ...\n";
    std::cout << "    for (auto win : v | std::views::slide(3)) ...\n";
    std::cout << "    for (auto [a,b] : std::views::zip(v1, v2)) ...\n\n";
}


// ============================================================
// Main
// ============================================================
int main() {
    demo_expected();
    demo_deducing_this();
    demo_print();
    demo_mdspan();
    demo_ranges();

    std::cout << "=== Compiler Info ===\n";
    std::cout << "  __cplusplus = " << __cplusplus << "\n";
#ifdef __GNUC__
    std::cout << "  GCC " << __GNUC__ << "." << __GNUC_MINOR__ << "\n";
#endif
#ifdef __clang__
    std::cout << "  Clang " << __clang_major__ << "." << __clang_minor__ << "\n";
#endif

    return 0;
}
