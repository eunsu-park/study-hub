# Lesson 21: C++23 Features

**Previous**: [CMake and Build Systems](./20_CMake_and_Build_Systems.md) | **Next**: [External Libraries](./22_External_Libraries.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Use `std::expected` for error handling without exceptions, replacing ad-hoc error codes and `std::optional` workarounds
2. Apply deducing `this` to simplify CRTP patterns and write value-category-aware member functions
3. Write formatted output with `std::print` and `std::println` as a type-safe, efficient replacement for `printf` and `iostream`
4. Use `std::mdspan` for multidimensional array views without owning data, enabling zero-copy interop with C arrays and libraries
5. Create lazy generators with `std::generator` to produce sequences on demand using coroutine syntax

---

## Table of Contents

1. [C++23 at a Glance](#1-c23-at-a-glance)
2. [`std::expected` — Error Handling Done Right](#2-stdexpected--error-handling-done-right)
3. [Deducing `this`](#3-deducing-this)
4. [`std::print` and `std::println`](#4-stdprint-and-stdprintln)
5. [`std::mdspan` — Multidimensional Views](#5-stdmdspan--multidimensional-views)
6. [`std::generator` — Lazy Coroutine Sequences](#6-stdgenerator--lazy-coroutine-sequences)
7. [Other Notable C++23 Features](#7-other-notable-c23-features)
8. [Exercises](#8-exercises)

---

## 1. C++23 at a Glance

C++23 is the latest published ISO standard (ISO/IEC 14882:2024). While C++20 introduced the "big four" (concepts, ranges, coroutines, modules), C++23 polishes rough edges and adds practical utilities.

| Category | Key Additions |
|----------|---------------|
| Error handling | `std::expected` |
| Language | Deducing `this`, `if consteval`, `static operator()` |
| I/O | `std::print`, `std::println` |
| Containers | `std::flat_map`, `std::flat_set`, `std::mdspan` |
| Ranges | `std::views::zip`, `chunk`, `slide`, `cartesian_product`, `enumerate` |
| Coroutines | `std::generator` |
| Utilities | `std::stacktrace`, `std::move_only_function` |

**Compiler support**: GCC 14+, Clang 18+, MSVC 19.38+ cover most features. Check [cppreference compiler support](https://en.cppreference.com/w/cpp/compiler_support) for details.

---

## 2. `std::expected` — Error Handling Done Right

### The Problem

C++ has several error-handling approaches, each with trade-offs:

| Approach | Downside |
|----------|----------|
| Exceptions | Runtime overhead, hard to reason about control flow |
| Error codes | Easy to ignore, no type-safe error payload |
| `std::optional` | No error information — only "value or nothing" |
| `std::variant` | Verbose, index-based access |

### `std::expected<T, E>`

`std::expected` holds either a value of type `T` or an error of type `E`:

```cpp
#include <expected>
#include <string>
#include <charconv>

enum class ParseError { empty_input, invalid_format, overflow };

std::expected<int, ParseError> parse_int(std::string_view sv) {
    if (sv.empty())
        return std::unexpected(ParseError::empty_input);

    int result{};
    auto [ptr, ec] = std::from_chars(sv.data(), sv.data() + sv.size(), result);

    if (ec == std::errc::result_out_of_range)
        return std::unexpected(ParseError::overflow);
    if (ec != std::errc{} || ptr != sv.data() + sv.size())
        return std::unexpected(ParseError::invalid_format);

    return result;  // implicitly wraps in expected
}

void demo() {
    auto result = parse_int("42");
    if (result) {
        // Access value with * or .value()
        std::println("Parsed: {}", *result);
    }

    auto err = parse_int("abc");
    if (!err) {
        // Access error with .error()
        std::println("Error code: {}", static_cast<int>(err.error()));
    }
}
```

### Monadic Operations

`std::expected` supports `and_then`, `or_else`, and `transform` for chaining:

```cpp
auto read_config(std::string_view path)
    -> std::expected<Config, Error>;

auto validate(Config cfg)
    -> std::expected<Config, Error>;

auto apply(Config cfg)
    -> std::expected<void, Error>;

// Chain operations — each step propagates errors automatically
auto result = read_config("/etc/app.conf")
    .and_then(validate)
    .and_then(apply);

// Equivalent to nested if-else without the indentation
```

---

## 3. Deducing `this`

### The Problem

Before C++23, writing member functions that behave differently based on value category (lvalue vs rvalue) required duplicating code:

```cpp
// Pre-C++23: two overloads for const/non-const
class Widget {
    std::string name_;
public:
    const std::string& name() const& { return name_; }
    std::string name() && { return std::move(name_); }
};
```

### Explicit Object Parameter

C++23 allows the first parameter to be an explicit `this`:

```cpp
class Widget {
    std::string name_;
public:
    // Single function handles all value categories
    template<typename Self>
    auto&& name(this Self&& self) {
        return std::forward<Self>(self).name_;
    }
};

// Usage:
Widget w{"hello"};
auto& ref = w.name();           // lvalue: returns const string&
auto val = std::move(w).name(); // rvalue: returns string&&
```

### Simplified CRTP

The Curiously Recurring Template Pattern (CRTP) becomes much simpler:

```cpp
// Pre-C++23 CRTP
template<typename Derived>
class Addable {
public:
    Derived operator+(const Derived& other) const {
        Derived result = static_cast<const Derived&>(*this);
        result += other;
        return result;
    }
};

class Vec2 : public Addable<Vec2> { /* ... */ };

// C++23: no template parameter needed
class Addable23 {
public:
    template<typename Self>
    Self operator+(this Self self, const Self& other) {
        self += other;
        return self;
    }
};

class Vec2 : public Addable23 { /* ... */ };
```

### Recursive Lambdas

Deducing `this` makes recursive lambdas natural:

```cpp
auto fibonacci = [](this auto self, int n) -> int {
    if (n <= 1) return n;
    return self(n - 1) + self(n - 2);
};

std::println("{}", fibonacci(10));  // 55
```

---

## 4. `std::print` and `std::println`

`std::print` brings `std::format` directly to stdout, replacing both `printf` (type-unsafe) and `iostream` (verbose):

```cpp
#include <print>

void demo() {
    int x = 42;
    double pi = 3.14159;
    std::string name = "C++23";

    // Type-safe, compile-time checked format strings
    std::println("Hello, {}!", name);
    std::println("x = {}, pi = {:.2f}", x, pi);

    // Alignment and fill
    std::println("{:>10}", "right");    //      right
    std::println("{:*^10}", "center");  //  **center**

    // Print to any FILE* or ostream
    std::print(stderr, "Error: {}\n", "something went wrong");
}
```

**Why not just `std::format` + `std::cout`?**

`std::print` is more efficient — it writes directly to the output stream without creating an intermediate `std::string`. It also handles Unicode correctly and flushes properly.

---

## 5. `std::mdspan` — Multidimensional Views

`std::mdspan` provides a non-owning multidimensional view over contiguous memory. Think of it as a multi-dimensional `std::span`.

```cpp
#include <mdspan>
#include <vector>

void demo() {
    std::vector<double> data(12);
    std::iota(data.begin(), data.end(), 1.0);

    // View as 3×4 matrix (row-major by default)
    std::mdspan mat(data.data(), 3, 4);

    // Access with multidimensional indexing
    for (std::size_t i = 0; i < mat.extent(0); ++i) {
        for (std::size_t j = 0; j < mat.extent(1); ++j) {
            std::print("{:4.0f}", mat[i, j]);  // C++23 multi-subscript
        }
        std::println();
    }
    // Output:
    //    1   2   3   4
    //    5   6   7   8
    //    9  10  11  12
}
```

### Layout Policies

```cpp
// Column-major (Fortran style) — for BLAS/LAPACK interop
std::mdspan<double, std::dextents<size_t, 2>,
            std::layout_left> col_major(data.data(), 3, 4);

// Custom stride
std::mdspan<double, std::dextents<size_t, 2>,
            std::layout_stride> strided(
    data.data(),
    std::layout_stride::mapping(
        std::dextents<size_t, 2>(3, 4),
        std::array<size_t, 2>{4, 1}  // row stride=4, col stride=1
    )
);
```

### Zero-Copy Interop with C

```cpp
// Wrap a C array without copying
extern "C" void legacy_compute(double* matrix, int rows, int cols);

void modern_wrapper(std::mdspan<double, std::dextents<size_t, 2>> mat) {
    // Pass underlying pointer to C code
    legacy_compute(mat.data_handle(),
                   static_cast<int>(mat.extent(0)),
                   static_cast<int>(mat.extent(1)));
}
```

---

## 6. `std::generator` — Lazy Coroutine Sequences

`std::generator<T>` is the standard library's coroutine-based lazy sequence generator:

```cpp
#include <generator>
#include <ranges>

// Infinite sequence of Fibonacci numbers
std::generator<long long> fibonacci() {
    long long a = 0, b = 1;
    while (true) {
        co_yield a;
        auto next = a + b;
        a = b;
        b = next;
    }
}

void demo() {
    // Take first 10 Fibonacci numbers
    for (auto n : fibonacci() | std::views::take(10)) {
        std::print("{} ", n);
    }
    // Output: 0 1 1 2 3 5 8 13 21 34
}
```

### Tree Traversal

```cpp
struct TreeNode {
    int value;
    TreeNode* left = nullptr;
    TreeNode* right = nullptr;
};

std::generator<int> inorder(TreeNode* node) {
    if (!node) co_return;
    co_yield std::ranges::elements_of(inorder(node->left));
    co_yield node->value;
    co_yield std::ranges::elements_of(inorder(node->right));
}
```

### Compared to Iterators

```cpp
// Traditional iterator: ~50 lines of boilerplate
// std::generator: 5 lines of clear, sequential logic
// Both produce the same lazy, on-demand sequence
```

---

## 7. Other Notable C++23 Features

### `std::flat_map` and `std::flat_set`

Cache-friendly sorted containers backed by contiguous arrays:

```cpp
#include <flat_map>
std::flat_map<std::string, int> scores;
scores["Alice"] = 95;
// Internally: sorted vector<pair<string,int>>
// Better cache locality than std::map (red-black tree)
```

### New Range Adaptors

```cpp
#include <ranges>

std::vector v = {1, 2, 3, 4, 5};

// zip: combine multiple ranges
for (auto [a, b] : std::views::zip(v, v | std::views::reverse)) {
    std::println("{} {}", a, b);  // 1 5, 2 4, 3 3, ...
}

// enumerate: index + value (like Python's enumerate)
for (auto [i, val] : std::views::enumerate(v)) {
    std::println("[{}] = {}", i, val);
}

// chunk: split into groups of N
for (auto chunk : v | std::views::chunk(2)) {
    // chunk = {1,2}, {3,4}, {5}
}

// slide: sliding window
for (auto window : v | std::views::slide(3)) {
    // window = {1,2,3}, {2,3,4}, {3,4,5}
}
```

### `if consteval`

```cpp
consteval int compile_time_only(int x) { return x * 2; }

constexpr int flexible(int x) {
    if consteval {
        return compile_time_only(x);  // only at compile time
    } else {
        return x * 2;  // fallback at runtime
    }
}
```

### `static operator()` and `static operator[]`

```cpp
struct Multiply {
    static int operator()(int a, int b) { return a * b; }
};
// No implicit 'this' pointer → potentially more efficient
```

---

## 8. Exercises

### Exercise 1: Error Pipeline with `std::expected`

Build a data processing pipeline using `std::expected`:
1. `read_file(path) → expected<string, Error>` — read a "file" (mock)
2. `parse_json(str) → expected<Config, Error>` — parse "JSON" (mock)
3. `validate(Config) → expected<Config, Error>` — check required fields

Chain them with `and_then`. Handle all error cases.

### Exercise 2: CRTP Replacement with Deducing `this`

Refactor a CRTP-based `Printable<Derived>` mixin to use deducing `this`. The mixin should provide a `print()` method that calls `to_string()` on the derived class. Test with two different derived classes.

### Exercise 3: Matrix Operations with `std::mdspan`

Write a function that multiplies two matrices using `std::mdspan`:
- `multiply(mdspan<double, dextents<size_t,2>> A, mdspan<double, dextents<size_t,2>> B, mdspan<double, dextents<size_t,2>> C)`
- Handle both row-major and column-major layouts

### Exercise 4: Lazy Sequence Generator

Implement these generators using `std::generator`:
1. `primes()` — infinite prime number sequence
2. `flatten(vector<vector<int>>)` — flatten nested containers
3. `interleave(gen1, gen2)` — alternate between two generators

### Exercise 5: Range Pipeline

Using C++23 range adaptors, solve this in a single pipeline:
- Given a vector of strings, enumerate them, filter by length > 3, chunk into groups of 2, and format each chunk as `"[idx] word, [idx] word"`.

---

## Navigation

**Previous**: [CMake and Build Systems](./20_CMake_and_Build_Systems.md) | **Next**: [External Libraries](./22_External_Libraries.md)
