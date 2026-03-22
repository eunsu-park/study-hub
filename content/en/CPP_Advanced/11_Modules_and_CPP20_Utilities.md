# C++20 Modules and Utilities

**Previous**: [C++20 Coroutines](./10_CPP20_Coroutines.md) | **Next**: [Multithreading in C++](./12_Multithreading.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain why modules replace header files and describe the difference in compilation models
2. Write module interface units with `export module`, `import`, and module partitions
3. Use header units (`import <header>`) to transition incrementally from headers to modules
4. Format strings with `std::format` including positional arguments and custom formatters
5. Apply `std::span` as a non-owning view over contiguous data and extract subspans safely
6. Use the three-way comparison (spaceship) operator to auto-generate all relational operators
7. Create auto-joining threads with `std::jthread` and implement cooperative cancellation with `stop_token`

---

C++20 introduced several features beyond Concepts, Ranges, and Coroutines that modernize everyday coding. Modules eliminate the preprocessor-based inclusion model that has caused slow builds and macro pollution for decades. `std::format` brings Python-style formatting to C++. `std::span` provides a safe, non-owning view over contiguous memory. The spaceship operator removes boilerplate comparison code. `std::jthread` fixes the footgun of forgetting to join threads. Together, these utilities make C++20 code shorter, safer, and faster to compile.

---

## Table of Contents

1. [Modules Introduction](#1-modules-introduction)
2. [Module Syntax](#2-module-syntax)
3. [Header Units](#3-header-units)
4. [std::format](#4-stdformat)
5. [std::span](#5-stdspan)
6. [Three-Way Comparison](#6-three-way-comparison)
7. [std::jthread](#7-stdjthread)
8. [Other C++20 Features](#8-other-c20-features)

---

## 1. Modules Introduction

### Why Modules?

The `#include` model has fundamental problems:

| Problem | Description |
|---------|-------------|
| Repeated parsing | `<vector>` is parsed in **every** translation unit that includes it |
| Macro leakage | `#define`s from one header affect all subsequent headers |
| Include order | Different orders can produce different results |
| No encapsulation | Everything in a header is visible (no "private" section) |
| Slow builds | Large projects spend most compile time re-parsing headers |

Modules solve all of these: each module is compiled **once** into a binary module interface (BMI), macros do not leak across module boundaries, and import order does not matter.

### Headers vs Modules

| Traditional Headers | C++20 Modules |
|--------------------|---------------|
| Textual inclusion (`#include`) | Semantic import (`import`) |
| Parsed every TU | Compiled once, cached |
| Macro pollution | Macros do not leak |
| Order-dependent | Order-independent |
| No visibility control | `export` controls API |
| Slow incremental builds | Fast incremental builds |

---

## 2. Module Syntax

### Module Interface Unit

```cpp
// math.cppm (or math.ixx on MSVC)
export module math;  // Declares this file as the interface of module "math"

// Exported: visible to importers
export int add(int a, int b) {
    return a + b;
}

export int multiply(int a, int b) {
    return a * b;
}

// Not exported: internal to the module
int helper_function() {
    return 42;
}
```

### Module Implementation Unit

```cpp
// math_impl.cpp
module math;  // Implements the "math" module (no 'export' keyword)

// Can access all names in the module, including non-exported ones
int internal_compute(int x) {
    return helper_function() + x;
}
```

### Importing a Module

```cpp
// main.cpp
import math;
import <iostream>;  // Header unit (see Section 3)

int main() {
    std::cout << add(1, 2) << "\n";       // OK: exported
    std::cout << multiply(3, 4) << "\n";  // OK: exported
    // helper_function();                  // Error: not exported
    return 0;
}
```

### Module Partitions

Large modules can be split into partitions:

```cpp
// math-arithmetic.cppm
export module math:arithmetic;

export int add(int a, int b) { return a + b; }
export int sub(int a, int b) { return a - b; }

// math-trig.cppm
export module math:trig;

import <cmath>;

export double sine(double x) { return std::sin(x); }
export double cosine(double x) { return std::cos(x); }

// math.cppm (primary module interface)
export module math;

export import :arithmetic;  // Re-export partition
export import :trig;
```

### Compilation

```bash
# GCC
g++ -std=c++20 -fmodules-ts -c math.cppm -o math.o
g++ -std=c++20 -fmodules-ts main.cpp math.o -o main

# Clang
clang++ -std=c++20 --precompile math.cppm -o math.pcm
clang++ -std=c++20 -fmodule-file=math=math.pcm main.cpp math.o -o main

# MSVC
cl /std:c++20 /c math.ixx
cl /std:c++20 main.cpp math.obj
```

---

## 3. Header Units

Header units let you `import` traditional headers as if they were modules, gaining some compilation speedup without rewriting code.

```cpp
// Instead of:
#include <iostream>
#include <vector>
#include <string>

// Write:
import <iostream>;
import <vector>;
import <string>;
```

### Transition Strategy

1. **Phase 1**: Replace `#include` with `import` for standard headers
2. **Phase 2**: Convert your own utility headers to modules
3. **Phase 3**: Convert application code to modules
4. **Keep** third-party headers as `#include` until they provide modules

### Importable Headers

Not all headers are importable. The standard guarantees that all C++ standard library headers are importable. C headers (`<cstdio>`, `<cmath>`) may or may not be importable depending on the compiler.

```cpp
import <vector>;     // Always importable
import <iostream>;   // Always importable
import <cmath>;      // Implementation-defined
// import "mylib.h"; // Importable only if the build system supports it
```

---

## 4. std::format

### Basic Formatting

`std::format` brings Python-style string formatting to C++:

```cpp
#include <format>
#include <iostream>
#include <string>

int main() {
    // Basic replacement
    std::string s = std::format("Hello, {}!", "World");
    std::cout << s << "\n";  // Hello, World!

    // Multiple arguments
    std::cout << std::format("{} + {} = {}", 1, 2, 3) << "\n";

    // Type is inferred automatically
    std::cout << std::format("int={}, double={}, bool={}, str={}",
                             42, 3.14, true, "hello") << "\n";

    return 0;
}
```

### Format Specifiers

```cpp
#include <format>
#include <iostream>

int main() {
    // Width and alignment
    std::cout << std::format("{:>10}", "right") << "\n";    //      right
    std::cout << std::format("{:<10}", "left") << "\n";     // left
    std::cout << std::format("{:^10}", "center") << "\n";   //   center

    // Fill character
    std::cout << std::format("{:*>10}", 42) << "\n";        // ********42
    std::cout << std::format("{:0>8}", 42) << "\n";         // 00000042

    // Number formatting
    std::cout << std::format("{:d}", 255) << "\n";          // 255 (decimal)
    std::cout << std::format("{:x}", 255) << "\n";          // ff (hex)
    std::cout << std::format("{:o}", 255) << "\n";          // 377 (octal)
    std::cout << std::format("{:b}", 255) << "\n";          // 11111111 (binary)
    std::cout << std::format("{:#x}", 255) << "\n";         // 0xff (with prefix)

    // Floating point
    std::cout << std::format("{:.2f}", 3.14159) << "\n";    // 3.14
    std::cout << std::format("{:.4e}", 12345.6) << "\n";    // 1.2346e+04

    return 0;
}
```

### Positional Arguments

```cpp
#include <format>

// Refer to arguments by position
auto s1 = std::format("{0} scored {1} points. {0} wins!", "Alice", 95);
// "Alice scored 95 points. Alice wins!"

// Reuse arguments
auto s2 = std::format("{0}{1}{0}", "abra", "cad");
// "abracadabra"
```

### Custom Formatters

```cpp
#include <format>
#include <iostream>

struct Point {
    double x, y;
};

template<>
struct std::formatter<Point> {
    // Parse format spec (e.g., {:f} for fixed)
    constexpr auto parse(std::format_parse_context& ctx) {
        return ctx.begin();  // No custom spec
    }

    // Format the Point
    auto format(const Point& p, std::format_context& ctx) const {
        return std::format_to(ctx.out(), "({:.2f}, {:.2f})", p.x, p.y);
    }
};

int main() {
    Point p{1.5, 2.7};
    std::cout << std::format("Point: {}", p) << "\n";
    // Point: (1.50, 2.70)
    return 0;
}
```

---

## 5. std::span

### Non-Owning View over Contiguous Data

`std::span` is a lightweight, non-owning reference to a contiguous sequence of elements. It replaces the `(pointer, size)` pattern and unifies arrays, vectors, and C arrays under one type.

```cpp
#include <span>
#include <vector>
#include <array>
#include <iostream>

void print(std::span<const int> data) {
    for (int n : data) {
        std::cout << n << " ";
    }
    std::cout << "\n";
}

void double_values(std::span<int> data) {
    for (int& n : data) {
        n *= 2;
    }
}

int main() {
    int c_arr[] = {1, 2, 3, 4, 5};
    std::vector<int> vec = {10, 20, 30};
    std::array<int, 4> std_arr = {100, 200, 300, 400};

    print(c_arr);     // 1 2 3 4 5
    print(vec);       // 10 20 30
    print(std_arr);   // 100 200 300 400

    double_values(c_arr);
    print(c_arr);     // 2 4 6 8 10

    return 0;
}
```

### Subspan

```cpp
#include <span>
#include <vector>
#include <iostream>

int main() {
    std::vector<int> v = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::span<int> s(v);

    // First N elements
    auto first3 = s.first(3);   // 0 1 2

    // Last N elements
    auto last3 = s.last(3);     // 7 8 9

    // Subspan: offset, count
    auto mid = s.subspan(3, 4); // 3 4 5 6

    // Size and empty
    std::cout << "Size: " << s.size() << "\n";      // 10
    std::cout << "Empty: " << s.empty() << "\n";     // 0 (false)

    // Element access
    std::cout << "Front: " << s.front() << "\n";     // 0
    std::cout << "Back: " << s.back() << "\n";       // 9
    std::cout << "s[4]: " << s[4] << "\n";           // 4

    return 0;
}
```

### Static vs Dynamic Extent

```cpp
#include <span>

// Dynamic extent: size known at runtime (default)
void dynamic_span(std::span<int> s);          // s.size() varies

// Static extent: size known at compile time
void fixed_span(std::span<int, 4> s);         // s.size() == 4 always

// Static extent enables compile-time checks
std::array<int, 4> arr = {1, 2, 3, 4};
fixed_span(arr);          // OK
// std::vector<int> v(4);
// fixed_span(v);          // Error: vector has dynamic extent
```

---

## 6. Three-Way Comparison

### The Spaceship Operator (`<=>`)

The three-way comparison operator generates all six relational operators from a single declaration.

```cpp
#include <compare>
#include <iostream>

struct Point {
    int x, y;

    // Default: lexicographic comparison of members in declaration order
    auto operator<=>(const Point&) const = default;
};

int main() {
    Point a{1, 2}, b{1, 3}, c{1, 2};

    // All six operators work:
    std::cout << (a < b) << "\n";   // 1 (true)
    std::cout << (a > b) << "\n";   // 0 (false)
    std::cout << (a <= c) << "\n";  // 1 (true)
    std::cout << (a >= c) << "\n";  // 1 (true)
    std::cout << (a == c) << "\n";  // 1 (true)
    std::cout << (a != b) << "\n";  // 1 (true)

    return 0;
}
```

### Comparison Categories

```cpp
#include <compare>

// strong_ordering: exactly one of <, ==, > holds; equal means identical
struct IntWrapper {
    int value;
    std::strong_ordering operator<=>(const IntWrapper&) const = default;
};

// weak_ordering: equivalent objects may not be identical
struct CaseInsensitiveString {
    std::string str;
    std::weak_ordering operator<=>(const CaseInsensitiveString& other) const {
        // Case-insensitive comparison
        auto to_lower = [](std::string s) {
            std::transform(s.begin(), s.end(), s.begin(), ::tolower);
            return s;
        };
        return to_lower(str) <=> to_lower(other.str);
    }
    bool operator==(const CaseInsensitiveString& other) const {
        return (*this <=> other) == 0;
    }
};

// partial_ordering: some values may be unordered (e.g., NaN)
struct FloatWrapper {
    float value;
    std::partial_ordering operator<=>(const FloatWrapper&) const = default;
};
```

### Custom Spaceship

```cpp
#include <compare>
#include <string>

struct Student {
    std::string name;
    double gpa;

    // Compare by GPA first (descending), then name (ascending)
    std::strong_ordering operator<=>(const Student& other) const {
        // Higher GPA first
        if (auto cmp = other.gpa <=> gpa; cmp != 0) return cmp;
        // Then alphabetical name
        return name <=> other.name;
    }

    bool operator==(const Student& other) const {
        return name == other.name && gpa == other.gpa;
    }
};
```

---

## 7. std::jthread

### Auto-Joining Thread

`std::jthread` is a `std::thread` that **automatically joins** in its destructor, eliminating the common bug of a `std::terminate` call from a joinable thread going out of scope.

```cpp
#include <thread>
#include <iostream>

void work() {
    std::cout << "Working...\n";
}

int main() {
    {
        std::jthread t(work);
        // No need to call t.join()!
    }
    // t's destructor calls join automatically

    // Compare with std::thread:
    // {
    //     std::thread t(work);
    //     // Forgetting t.join() here causes std::terminate!
    // }

    return 0;
}
```

### Cooperative Cancellation with stop_token

`std::jthread` provides a built-in cancellation mechanism via `stop_token`:

```cpp
#include <thread>
#include <iostream>
#include <chrono>

void worker(std::stop_token stoken) {
    int counter = 0;
    while (!stoken.stop_requested()) {
        std::cout << "Working... " << ++counter << "\n";
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
    std::cout << "Worker received stop request. Cleaning up.\n";
}

int main() {
    std::jthread t(worker);

    std::this_thread::sleep_for(std::chrono::seconds(1));

    // Request cooperative stop
    t.request_stop();

    // Destructor joins automatically
    return 0;
}
```

### Stop Callback

```cpp
#include <thread>
#include <iostream>

void demo() {
    std::jthread t([](std::stop_token stoken) {
        // Register a callback that runs when stop is requested
        std::stop_callback cb(stoken, [] {
            std::cout << "Stop callback invoked!\n";
        });

        while (!stoken.stop_requested()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    t.request_stop();  // Triggers the callback + loop exit
}
```

---

## 8. Other C++20 Features

### consteval

`consteval` functions **must** be evaluated at compile time (unlike `constexpr`, which *can* be):

```cpp
consteval int square(int n) {
    return n * n;
}

constexpr int a = square(5);   // OK: compile-time
// int x = 5;
// int b = square(x);          // Error: x is not a constant expression
```

### constinit

`constinit` ensures a variable is initialized at compile time, avoiding the "static initialization order fiasco":

```cpp
constinit int global_counter = 0;       // OK: zero-initialized at compile time
// constinit int bad = some_function();  // Error if some_function() isn't constexpr
```

### Designated Initializers

```cpp
struct Config {
    int width = 800;
    int height = 600;
    bool fullscreen = false;
    const char* title = "App";
};

int main() {
    Config cfg{
        .width = 1920,
        .height = 1080,
        .fullscreen = true
        // .title uses default
    };
    return 0;
}
```

### [[likely]] and [[unlikely]]

Branch prediction hints for the optimizer:

```cpp
int process(int value) {
    if (value > 0) [[likely]] {
        return value * 2;
    } else [[unlikely]] {
        throw std::runtime_error("Negative value");
    }
}
```

### std::source_location

Replaces `__FILE__`, `__LINE__`, `__func__` macros:

```cpp
#include <source_location>
#include <iostream>

void log(const std::string& msg,
         const std::source_location& loc = std::source_location::current()) {
    std::cout << loc.file_name() << ":"
              << loc.line() << " ["
              << loc.function_name() << "] "
              << msg << "\n";
}

int main() {
    log("Application started");
    // main.cpp:42 [main] Application started
    return 0;
}
```

---

## Exercises

### Exercise 1: Module Library

Create a `geometry` module with two partitions: `:shapes` (Circle, Rectangle classes) and `:algorithms` (area, perimeter functions). Write a `main.cpp` that imports the module and uses both partitions.

### Exercise 2: Custom Formatter

Implement a `std::formatter` specialization for a `Duration` struct (hours, minutes, seconds). Support two format specs: `{:short}` for "2h30m15s" and `{:long}` for "2 hours, 30 minutes, 15 seconds".

### Exercise 3: Span Utilities

Write a function `split_at(std::span<int>, size_t pos)` that returns a `std::pair<std::span<int>, std::span<int>>`. Write a second function `sliding_window(std::span<const int>, size_t window_size)` that returns a `vector<span<const int>>` of overlapping windows.

### Exercise 4: Spaceship Operator

Define a `Version` struct (major, minor, patch) with a defaulted `<=>`. Then define a `SemanticVersion` that ignores the patch field in ordering but includes it in equality. Write tests for both.

### Exercise 5: Jthread Worker Pool

Create a simple worker pool using `std::jthread` and `stop_token`. The pool should accept tasks via a thread-safe queue and shut down gracefully when `request_stop()` is called. Test with 4 workers and 20 tasks.

---

## Next Steps

With modules, formatting, spans, and jthread under your belt, you are ready to tackle multithreading in depth -- the foundation for writing high-performance, concurrent C++ applications.

- [Multithreading in C++](./12_Multithreading.md)
