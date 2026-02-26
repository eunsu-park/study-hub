# Advanced C++20

**Previous**: [Multithreading and Concurrency](./16_Multithreading_Concurrency.md) | **Next**: [C++ Design Patterns](./18_Design_Patterns.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Define custom concepts and apply standard library concepts to constrain template parameters
2. Compose range pipelines using views (`filter`, `transform`, `take`, `drop`) for lazy data processing
3. Implement coroutines with `co_yield` and `co_return` to build generators and async tasks
4. Organize code into modules as an alternative to the traditional header/source model
5. Use the three-way comparison (spaceship) operator to auto-generate all relational operators
6. Apply C++20 utilities including `std::span`, `std::format`, designated initializers, and `consteval`
7. Identify C++23 features (`std::expected`, `std::print`, `std::generator`) and their intended use cases

---

C++20 is the most transformative standard since C++11, fundamentally changing how templates, iteration, and code organization work. Concepts replace cryptic SFINAE errors with readable constraints. Ranges eliminate iterator-pair boilerplate in favor of composable pipelines. Coroutines unlock cooperative multitasking without thread overhead. Understanding these features is not just about using the latest syntax--it is about writing code that is clearer, safer, and easier to maintain.

**Difficulty**: *****

**Prerequisites**: Templates, Lambdas, Smart Pointers

---

## Table of Contents

1. [Concepts](#concepts)
2. [Ranges](#ranges)
3. [Coroutines](#coroutines)
4. [Modules](#modules)
5. [Other C++20 Features](#other-c20-features)
6. [C++23 Preview](#c23-preview)

---

## Concepts

### What are Concepts?

A feature that defines **constraints** on template parameters. Much more readable than the previous SFINAE approach.

### Basic Usage

```cpp
#include <concepts>
#include <iostream>

// Define a concept
template<typename T>
concept Numeric = std::integral<T> || std::floating_point<T>;

// Use concept
template<Numeric T>
T add(T a, T b) {
    return a + b;
}

// Or use requires clause
template<typename T>
    requires Numeric<T>
T multiply(T a, T b) {
    return a * b;
}

// Or trailing requires
template<typename T>
T divide(T a, T b) requires Numeric<T> {
    return a / b;
}

int main() {
    std::cout << add(1, 2) << "\n";        // OK
    std::cout << add(1.5, 2.5) << "\n";    // OK
    // add("hello", "world");              // Compile error!
    return 0;
}
```

### Standard Concepts

```cpp
#include <concepts>

// Type-related
std::same_as<T, U>           // T and U are the same type
std::derived_from<D, B>      // D derives from B
std::convertible_to<From, To>// From is convertible to To

// Arithmetic-related
std::integral<T>             // Integer type
std::floating_point<T>       // Floating-point type
std::signed_integral<T>      // Signed integer
std::unsigned_integral<T>    // Unsigned integer

// Comparison-related
std::equality_comparable<T>  // == operation possible
std::totally_ordered<T>      // <, >, <=, >= operations possible

// Callable-related
std::invocable<F, Args...>   // F(Args...) is callable
std::predicate<F, Args...>   // F(Args...) returns bool
```

### Custom Concept Definition

```cpp
#include <concepts>
#include <string>

// Type that behaves like a string
template<typename T>
concept StringLike = requires(T t) {
    { t.length() } -> std::convertible_to<std::size_t>;
    { t.c_str() } -> std::same_as<const char*>;
    { t[0] } -> std::convertible_to<char>;
};

// Container concept
template<typename T>
concept Container = requires(T t) {
    typename T::value_type;
    typename T::iterator;
    { t.begin() } -> std::same_as<typename T::iterator>;
    { t.end() } -> std::same_as<typename T::iterator>;
    { t.size() } -> std::convertible_to<std::size_t>;
};

// Usage
template<Container C>
void printContainer(const C& container) {
    for (const auto& item : container) {
        std::cout << item << " ";
    }
    std::cout << "\n";
}
```

### Requires Expressions

```cpp
// Simple requirement
template<typename T>
concept Addable = requires(T a, T b) {
    a + b;  // This expression must be valid
};

// Type requirement
template<typename T>
concept HasValueType = requires {
    typename T::value_type;
};

// Compound requirement
template<typename T>
concept Hashable = requires(T t) {
    { std::hash<T>{}(t) } -> std::convertible_to<std::size_t>;
};

// Nested requirement
template<typename T>
concept Sortable = requires(T t) {
    requires std::totally_ordered<typename T::value_type>;
    { t.begin() } -> std::random_access_iterator;
};
```

### Overloading with Concepts

```cpp
#include <concepts>
#include <iostream>

template<std::integral T>
void print(T value) {
    std::cout << "Integer: " << value << "\n";
}

template<std::floating_point T>
void print(T value) {
    std::cout << "Float: " << value << "\n";
}

template<typename T>
void print(T value) {
    std::cout << "Other: " << value << "\n";
}

int main() {
    print(42);       // Integer: 42
    print(3.14);     // Float: 3.14
    print("hello");  // Other: hello
    return 0;
}
```

---

## Ranges

### What are Ranges?

A library for handling containers and algorithms more elegantly. Supports pipeline-style operations.

### Basic Usage

```cpp
#include <ranges>
#include <vector>
#include <iostream>

int main() {
    std::vector<int> nums = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

    // Traditional way
    // for (auto it = nums.begin(); it != nums.end(); ++it) { ... }

    // Ranges way
    for (int n : nums | std::views::filter([](int x) { return x % 2 == 0; })
                      | std::views::transform([](int x) { return x * x; })) {
        std::cout << n << " ";  // 4 16 36 64 100
    }

    return 0;
}
```

### Views

Views are lazily evaluated and don't copy the original data.

```cpp
#include <ranges>
#include <vector>
#include <iostream>

int main() {
    std::vector<int> v = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

    // filter: only elements matching condition
    auto evens = v | std::views::filter([](int x) { return x % 2 == 0; });

    // transform: transformation
    auto squared = v | std::views::transform([](int x) { return x * x; });

    // take: first n elements
    auto first3 = v | std::views::take(3);

    // drop: skip first n elements
    auto afterFirst3 = v | std::views::drop(3);

    // reverse: reverse order
    auto reversed = v | std::views::reverse;

    // Combination
    auto result = v | std::views::filter([](int x) { return x > 3; })
                    | std::views::transform([](int x) { return x * 2; })
                    | std::views::take(3);

    for (int n : result) {
        std::cout << n << " ";  // 8 10 12
    }

    return 0;
}
```

### Main Views

```cpp
#include <ranges>
namespace views = std::views;

// Generator views
auto r1 = views::iota(1, 10);        // 1, 2, ..., 9
auto r2 = views::iota(1) | views::take(10);  // 10 from infinite sequence

// Transform views
auto r3 = v | views::transform(func);
auto r4 = v | views::filter(pred);
auto r5 = v | views::take(n);
auto r6 = v | views::drop(n);
auto r7 = v | views::take_while(pred);
auto r8 = v | views::drop_while(pred);
auto r9 = v | views::reverse;

// Split views
auto r10 = str | views::split(' ');  // Split by space

// Join views
auto r11 = nested | views::join;     // Flatten nested range

// Element views
auto r12 = pairs | views::elements<0>;  // First element of tuple
auto r13 = pairs | views::keys;         // Keys of map
auto r14 = pairs | views::values;       // Values of map
```

### Range Algorithms

```cpp
#include <ranges>
#include <algorithm>
#include <vector>

int main() {
    std::vector<int> v = {3, 1, 4, 1, 5, 9, 2, 6};

    // Range-based algorithms
    std::ranges::sort(v);
    std::ranges::reverse(v);

    auto it = std::ranges::find(v, 5);
    bool found = std::ranges::contains(v, 5);

    int count = std::ranges::count_if(v, [](int x) { return x > 3; });

    auto [min, max] = std::ranges::minmax(v);

    // Projection
    struct Person {
        std::string name;
        int age;
    };

    std::vector<Person> people = {{"Alice", 30}, {"Bob", 25}};
    std::ranges::sort(people, {}, &Person::age);  // Sort by age

    return 0;
}
```

---

## Coroutines

### What are Coroutines?

Functions that can **suspend** execution and **resume** later.

### Basic Structure

```cpp
#include <coroutine>
#include <iostream>

// Coroutine return type
template<typename T>
struct Generator {
    struct promise_type {
        T current_value;

        Generator get_return_object() {
            return Generator{std::coroutine_handle<promise_type>::from_promise(*this)};
        }

        std::suspend_always initial_suspend() { return {}; }
        std::suspend_always final_suspend() noexcept { return {}; }
        void unhandled_exception() { std::terminate(); }

        std::suspend_always yield_value(T value) {
            current_value = value;
            return {};
        }

        void return_void() {}
    };

    std::coroutine_handle<promise_type> handle;

    explicit Generator(std::coroutine_handle<promise_type> h) : handle(h) {}
    ~Generator() { if (handle) handle.destroy(); }

    Generator(Generator&& other) noexcept : handle(other.handle) {
        other.handle = nullptr;
    }

    bool next() {
        if (!handle.done()) {
            handle.resume();
        }
        return !handle.done();
    }

    T value() const {
        return handle.promise().current_value;
    }
};

// Coroutine function
Generator<int> range(int start, int end) {
    for (int i = start; i < end; ++i) {
        co_yield i;  // Yield value and suspend
    }
}

int main() {
    auto gen = range(1, 5);

    while (gen.next()) {
        std::cout << gen.value() << " ";  // 1 2 3 4
    }

    return 0;
}
```

### co_await, co_yield, co_return

```cpp
// co_yield: yield a value and suspend
Generator<int> numbers() {
    co_yield 1;
    co_yield 2;
    co_yield 3;
}

// co_return: terminate the coroutine
Task<int> compute() {
    // Async work...
    co_return 42;
}

// co_await: wait on an awaitable object
Task<void> asyncWork() {
    auto result = co_await asyncOperation();
    // Use result...
}
```

### Practical Example: Simple Task

```cpp
#include <coroutine>
#include <optional>
#include <iostream>

template<typename T>
struct Task {
    struct promise_type {
        std::optional<T> result;

        Task get_return_object() {
            return Task{std::coroutine_handle<promise_type>::from_promise(*this)};
        }

        std::suspend_never initial_suspend() { return {}; }
        std::suspend_always final_suspend() noexcept { return {}; }
        void unhandled_exception() { std::terminate(); }

        void return_value(T value) {
            result = value;
        }
    };

    std::coroutine_handle<promise_type> handle;

    Task(std::coroutine_handle<promise_type> h) : handle(h) {}
    ~Task() { if (handle) handle.destroy(); }

    T get() {
        return *handle.promise().result;
    }
};

Task<int> asyncAdd(int a, int b) {
    co_return a + b;
}

int main() {
    auto task = asyncAdd(10, 20);
    std::cout << "Result: " << task.get() << "\n";
    return 0;
}
```

---

## Modules

### What are Modules?

A new code organization method that solves the drawbacks of header files.

### Module Definition

```cpp
// math.cppm (module interface)
export module math;

export int add(int a, int b) {
    return a + b;
}

export int multiply(int a, int b) {
    return a * b;
}

// Internal implementation (not exported)
int helper() {
    return 42;
}
```

### Using Modules

```cpp
// main.cpp
import math;
import <iostream>;

int main() {
    std::cout << add(1, 2) << "\n";
    std::cout << multiply(3, 4) << "\n";
    return 0;
}
```

### Compilation (GCC Example)

```bash
# Compile module
g++ -std=c++20 -fmodules-ts -c math.cppm

# Compile and link main
g++ -std=c++20 -fmodules-ts main.cpp math.o -o main
```

### Module Advantages

| Traditional Headers | Modules |
|-----------|------|
| Parsed every time | Compiled once |
| Macro pollution | Isolated |
| Include order matters | Order independent |
| Slow builds | Fast builds |

---

## Other C++20 Features

### Three-way Comparison Operator (Spaceship Operator)

```cpp
#include <compare>

struct Point {
    int x, y;

    auto operator<=>(const Point&) const = default;
    // ==, !=, <, >, <=, >= automatically generated
};

int main() {
    Point p1{1, 2}, p2{1, 3};

    if (p1 < p2) { /* ... */ }
    if (p1 == p2) { /* ... */ }

    auto result = p1 <=> p2;
    if (result < 0) { /* p1 < p2 */ }

    return 0;
}
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
    // C++20 designated initializers
    Config cfg{
        .width = 1920,
        .height = 1080,
        .fullscreen = true
        // title uses default value
    };

    return 0;
}
```

### consteval and constinit

```cpp
// consteval: must be evaluated at compile time
consteval int square(int n) {
    return n * n;
}

constexpr int a = square(5);  // OK
// int b = square(x);         // Error! x is not a constant

// constinit: forces static initialization
constinit int global = 42;
// constinit int bad = foo();  // Error! foo() is not constexpr
```

### std::span

```cpp
#include <span>
#include <vector>
#include <array>

void process(std::span<int> data) {
    for (int& n : data) {
        n *= 2;
    }
}

int main() {
    int arr[] = {1, 2, 3, 4, 5};
    std::vector<int> vec = {1, 2, 3, 4, 5};
    std::array<int, 5> stdArr = {1, 2, 3, 4, 5};

    process(arr);      // OK
    process(vec);      // OK
    process(stdArr);   // OK

    return 0;
}
```

### std::format

```cpp
#include <format>
#include <iostream>

int main() {
    std::string s = std::format("Hello, {}!", "World");
    std::cout << s << "\n";

    std::cout << std::format("{:>10}", 42) << "\n";      // Right align
    std::cout << std::format("{:08x}", 255) << "\n";     // Hex, zero-padded
    std::cout << std::format("{:.2f}", 3.14159) << "\n"; // 2 decimal places

    return 0;
}
```

### std::source_location

```cpp
#include <source_location>
#include <iostream>

void log(const std::string& msg,
         const std::source_location& loc = std::source_location::current()) {
    std::cout << loc.file_name() << ":"
              << loc.line() << " "
              << loc.function_name() << ": "
              << msg << "\n";
}

int main() {
    log("Hello!");  // main.cpp:15 main: Hello!
    return 0;
}
```

---

## C++23 Preview

### std::expected

```cpp
#include <expected>
#include <string>

std::expected<int, std::string> divide(int a, int b) {
    if (b == 0) {
        return std::unexpected("Division by zero");
    }
    return a / b;
}

int main() {
    auto result = divide(10, 2);
    if (result) {
        std::cout << "Result: " << *result << "\n";
    } else {
        std::cout << "Error: " << result.error() << "\n";
    }
    return 0;
}
```

### std::print

```cpp
#include <print>

int main() {
    std::print("Hello, {}!\n", "World");
    std::println("Value: {}", 42);  // Automatic newline
    return 0;
}
```

### std::generator (C++23)

```cpp
#include <generator>

std::generator<int> range(int start, int end) {
    for (int i = start; i < end; ++i) {
        co_yield i;
    }
}

int main() {
    for (int n : range(1, 10)) {
        std::cout << n << " ";
    }
    return 0;
}
```

---

## Practice Problems

### Problem 1: Define a Concept

Define a Concept representing a "printable" type (supports operator<<).

<details>
<summary>Show Answer</summary>

```cpp
template<typename T>
concept Printable = requires(std::ostream& os, T t) {
    { os << t } -> std::same_as<std::ostream&>;
};

template<Printable T>
void print(const T& value) {
    std::cout << value << "\n";
}
```

</details>

### Problem 2: Range Pipeline

Find the sum of squares of numbers from 1 to 100 that are multiples of 3 but not multiples of 5.

<details>
<summary>Show Answer</summary>

```cpp
#include <ranges>
#include <numeric>
#include <iostream>

int main() {
    auto result = std::views::iota(1, 101)
        | std::views::filter([](int x) { return x % 3 == 0 && x % 5 != 0; })
        | std::views::transform([](int x) { return x * x; });

    int sum = std::accumulate(result.begin(), result.end(), 0);
    std::cout << "Sum: " << sum << "\n";

    return 0;
}
```

</details>

---

## Next Steps

- [18_Design_Patterns.md](./18_Design_Patterns.md) - C++ Design Patterns

---

## References

- [cppreference C++20](https://en.cppreference.com/w/cpp/20)
- [C++20 Complete Guide](https://leanpub.com/cpp20)
- [Ranges Library](https://en.cppreference.com/w/cpp/ranges)
