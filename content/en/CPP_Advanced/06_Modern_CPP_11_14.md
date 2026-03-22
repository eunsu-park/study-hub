# Modern C++ -- C++11 and C++14

**Previous**: [Error Handling Patterns](./05_Error_Handling_Patterns.md) | **Next**: [Modern C++ (C++17)](./07_Modern_CPP_17.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Use `auto` and `decltype` for type deduction and trailing return types
2. Apply uniform initialization with braces to prevent narrowing conversions
3. Write lambda expressions with various capture modes including generic lambdas (C++14)
4. Use `constexpr` for compile-time evaluation and understand relaxed `constexpr` in C++14
5. Distinguish `nullptr` from `NULL` and apply scoped enumerations with `enum class`
6. Use `static_assert` for compile-time assertions with descriptive messages

---

C++11 was the most transformative update in the language's history, turning C++ from a language burdened by manual memory management and verbose syntax into one capable of expressive, safe, and efficient modern code. C++14 refined these features with quality-of-life improvements. Together, they form the baseline that every modern C++ programmer must master. This lesson extracts and expands the C++11/14 features, covering them in the depth they deserve.

## 1. auto and decltype

### auto Type Deduction

```cpp
#include <iostream>
#include <vector>
#include <map>
#include <memory>

int main() {
    // Basic type deduction
    auto x = 42;          // int
    auto y = 3.14;        // double
    auto s = "Hello";     // const char*
    auto b = true;        // bool

    // Complex types simplified
    std::vector<int> vec = {1, 2, 3, 4, 5};
    auto it = vec.begin();  // std::vector<int>::iterator

    std::map<std::string, std::vector<int>> data;
    auto& ref = data;  // std::map<std::string, std::vector<int>>&

    // auto with const and references
    const auto cx = 42;     // const int
    auto& ry = y;           // double&
    const auto& cry = y;    // const double&
    auto&& urx = 42;        // int&& (rvalue reference)
    auto&& ury = y;         // double& (lvalue reference, reference collapsing)

    // auto in return types (C++14)
    auto makeVec = []() {
        return std::vector<int>{1, 2, 3};
    };
    auto v = makeVec();

    return 0;
}
```

### decltype

```cpp
#include <iostream>

int x = 10;
decltype(x) y = 20;        // int (same type as x)
decltype(x + 0.5) z = 1.5; // double

// decltype preserves references and const
int& rx = x;
decltype(rx) ry = x;       // int& (preserves reference)

const int cx = 42;
decltype(cx) cy = 10;      // const int

// decltype on expressions
// decltype(x) is int (variable)
// decltype((x)) is int& (expression yielding lvalue)

// Trailing return type
template<typename T, typename U>
auto add(T a, U b) -> decltype(a + b) {
    return a + b;
}

// C++14: no trailing return type needed
template<typename T, typename U>
auto addModern(T a, U b) {
    return a + b;
}
```

### decltype(auto) (C++14)

```cpp
#include <iostream>
#include <string>

// decltype(auto) preserves exact type including references
int x = 10;

auto getValue1() { return x; }          // int (decays)
decltype(auto) getValue2() { return x; }  // int (same)
decltype(auto) getRef() { return (x); }  // int& (expression is lvalue)

// Useful for perfect return type forwarding
template<typename F, typename... Args>
decltype(auto) callAndReturn(F&& f, Args&&... args) {
    return f(std::forward<Args>(args)...);
}
```

---

## 2. Uniform Initialization

Braced initialization (`{}`) provides a uniform syntax that works everywhere and prevents narrowing conversions.

```cpp
#include <iostream>
#include <vector>
#include <map>
#include <initializer_list>

// Custom class with initializer_list
class Matrix {
    std::vector<std::vector<int>> data_;

public:
    Matrix(std::initializer_list<std::initializer_list<int>> init) {
        for (auto& row : init) {
            data_.emplace_back(row);
        }
    }

    void print() const {
        for (auto& row : data_) {
            for (int val : row) std::cout << val << " ";
            std::cout << "\n";
        }
    }
};

int main() {
    // Direct initialization
    int a{42};
    double b{3.14};
    std::string c{"Hello"};

    // Narrowing prevention
    // int narrow{3.14};     // ERROR: narrowing conversion
    // char small{1000};     // ERROR: narrowing conversion
    int ok{static_cast<int>(3.14)};  // OK: explicit cast

    // Container initialization
    std::vector<int> vec = {1, 2, 3, 4, 5};
    std::map<std::string, int> ages = {
        {"Alice", 25},
        {"Bob", 30}
    };

    // Struct initialization
    struct Point { int x, y; };
    Point p{10, 20};

    // Custom class
    Matrix m = {
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9}
    };
    m.print();

    // Most vexing parse avoided
    // Widget w();   // Declares a function! (most vexing parse)
    // Widget w{};   // Creates a Widget object

    return 0;
}
```

### initializer_list Gotcha

```cpp
#include <iostream>
#include <vector>

class Widget {
public:
    Widget(int size, double value) {
        std::cout << "size=" << size << ", value=" << value << "\n";
    }
    Widget(std::initializer_list<double> list) {
        std::cout << "initializer_list with " << list.size() << " elements\n";
    }
};

int main() {
    Widget w1(10, 5.0);    // Calls (int, double) constructor
    Widget w2{10, 5.0};    // Calls initializer_list constructor!
    Widget w3(10, 5.0);    // Use () when you want non-list constructor

    // std::vector has the same issue
    std::vector<int> v1(5, 10);   // 5 elements, each is 10
    std::vector<int> v2{5, 10};   // 2 elements: 5 and 10

    return 0;
}
```

---

## 3. Range-Based For Loop

```cpp
#include <iostream>
#include <vector>
#include <map>

int main() {
    std::vector<int> vec = {1, 2, 3, 4, 5};

    // Copy by value (safe but copies)
    for (int x : vec) {
        std::cout << x << " ";
    }
    std::cout << "\n";

    // Reference (modifiable)
    for (int& x : vec) {
        x *= 2;
    }

    // Const reference (read-only, no copy)
    for (const int& x : vec) {
        std::cout << x << " ";
    }
    std::cout << "\n";

    // auto deduction (recommended)
    for (const auto& x : vec) {
        std::cout << x << " ";
    }
    std::cout << "\n";

    // Works with maps
    std::map<std::string, int> scores = {{"Alice", 95}, {"Bob", 87}};
    for (const auto& [name, score] : scores) {  // C++17 structured bindings
        std::cout << name << ": " << score << "\n";
    }

    // Works with C arrays
    int arr[] = {10, 20, 30};
    for (int x : arr) {
        std::cout << x << " ";
    }
    std::cout << "\n";

    // Works with initializer list
    for (int x : {100, 200, 300}) {
        std::cout << x << " ";
    }
    std::cout << "\n";

    return 0;
}
```

---

## 4. nullptr

```cpp
#include <iostream>

void foo(int n) {
    std::cout << "int: " << n << std::endl;
}

void foo(int* p) {
    std::cout << "pointer: " << (p ? "non-null" : "null") << std::endl;
}

int main() {
    // NULL is defined as 0, causing ambiguity
    // foo(NULL);  // Ambiguous: int or pointer?

    // nullptr is type-safe
    foo(nullptr);  // Calls pointer overload
    foo(0);        // Calls int overload

    // Type: std::nullptr_t
    std::nullptr_t np = nullptr;

    // Works in boolean context
    int* p = nullptr;
    if (!p) {
        std::cout << "p is null" << std::endl;
    }

    // Template context
    auto lambda = [](auto* ptr) {
        if (ptr) std::cout << *ptr << "\n";
    };

    int x = 42;
    lambda(&x);     // OK
    // lambda(NULL);  // Error with templates
    // lambda(nullptr);  // OK: deduces std::nullptr_t

    return 0;
}
```

---

## 5. Lambda Expressions

### Basic Syntax

```cpp
#include <iostream>
#include <vector>
#include <algorithm>
#include <functional>

int main() {
    // [capture](params) -> return_type { body }

    // Basic lambda
    auto hello = []() { std::cout << "Hello!\n"; };
    hello();

    // With parameters and return
    auto add = [](int a, int b) -> int { return a + b; };
    std::cout << add(3, 4) << "\n";  // 7

    // Return type deduction (usually not needed)
    auto multiply = [](double a, double b) { return a * b; };

    // Immediately invoked lambda (IIFE)
    int result = [](int x) { return x * x; }(5);
    std::cout << "IIFE: " << result << "\n";  // 25

    return 0;
}
```

### Capture Modes

```cpp
#include <iostream>
#include <string>

int main() {
    int x = 10;
    int y = 20;
    std::string name = "Alice";

    // Capture by value (copy at lambda creation time)
    auto byValue = [x, y]() {
        std::cout << x + y << "\n";
        // x = 100;  // ERROR: captured by value is const
    };

    // Capture by reference
    auto byRef = [&x, &y]() {
        x += 10;
        y += 10;
    };
    byRef();
    std::cout << x << ", " << y << "\n";  // 20, 30

    // Capture all by value
    auto allVal = [=]() {
        std::cout << x << " " << name << "\n";
    };

    // Capture all by reference
    auto allRef = [&]() {
        x = 100;
        name = "Bob";
    };

    // Mixed capture
    auto mixed = [=, &x]() {  // y,name by value; x by reference
        x = 50;
        std::cout << y << " " << name << "\n";
    };

    // Mutable lambda (modify value-captured variables)
    int counter = 0;
    auto increment = [counter]() mutable {
        return ++counter;  // Modifies the lambda's internal copy
    };
    std::cout << increment() << "\n";  // 1
    std::cout << increment() << "\n";  // 2
    std::cout << counter << "\n";      // 0 (original unchanged)

    // Capture this pointer (in member functions)
    // auto lambda = [this]() { memberFunc(); };
    // auto lambda = [*this]() { /* captures copy of *this */ };  // C++17

    return 0;
}
```

### Lambdas with STL

```cpp
#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>

int main() {
    std::vector<int> vec = {3, 1, 4, 1, 5, 9, 2, 6};

    // Sort descending
    std::sort(vec.begin(), vec.end(),
        [](int a, int b) { return a > b; });

    // Find first even
    auto it = std::find_if(vec.begin(), vec.end(),
        [](int x) { return x % 2 == 0; });
    if (it != vec.end()) {
        std::cout << "First even: " << *it << "\n";
    }

    // Count elements > 3
    int count = std::count_if(vec.begin(), vec.end(),
        [](int x) { return x > 3; });
    std::cout << "Count > 3: " << count << "\n";

    // Transform
    std::vector<int> squared(vec.size());
    std::transform(vec.begin(), vec.end(), squared.begin(),
        [](int x) { return x * x; });

    // for_each
    std::for_each(vec.begin(), vec.end(),
        [](int x) { std::cout << x << " "; });
    std::cout << "\n";

    // Accumulate with lambda
    int sum = std::accumulate(vec.begin(), vec.end(), 0,
        [](int acc, int x) { return acc + x; });
    std::cout << "Sum: " << sum << "\n";

    return 0;
}
```

### Generic Lambdas (C++14)

```cpp
#include <iostream>
#include <string>
#include <vector>

int main() {
    // C++14: auto parameters make lambdas generic (template-like)
    auto print = [](const auto& x) {
        std::cout << x << "\n";
    };

    print(42);          // int
    print(3.14);        // double
    print("Hello");     // const char*

    // Multiple auto parameters
    auto add = [](auto a, auto b) {
        return a + b;
    };
    std::cout << add(1, 2) << "\n";            // 3
    std::cout << add(1.5, 2.5) << "\n";        // 4.0
    std::cout << add(std::string("Hello, "),
                     std::string("World!")) << "\n";

    // Generic lambda with perfect forwarding
    auto wrapper = [](auto&& func, auto&&... args) {
        return func(std::forward<decltype(args)>(args)...);
    };

    auto result = wrapper([](int a, int b) { return a + b; }, 3, 4);
    std::cout << "Wrapped: " << result << "\n";  // 7

    return 0;
}
```

---

## 6. constexpr

### C++11 constexpr

```cpp
#include <iostream>
#include <array>

// C++11: constexpr functions must be a single return statement
constexpr int square(int x) {
    return x * x;
}

constexpr int factorial(int n) {
    return (n <= 1) ? 1 : n * factorial(n - 1);
}

// constexpr variables
constexpr double PI = 3.14159265358979;
constexpr int MAX_SIZE = 1024;

// constexpr constructor
class Point {
public:
    int x, y;
    constexpr Point(int x, int y) : x(x), y(y) {}
    constexpr int manhattanDist() const { return x + y; }
};

int main() {
    // Evaluated at compile time
    constexpr int s = square(5);
    static_assert(s == 25, "square(5) should be 25");

    constexpr int f = factorial(5);
    static_assert(f == 120, "5! should be 120");

    // Used as template argument or array size
    std::array<int, factorial(4)> arr;  // Size 24

    constexpr Point p(3, 4);
    static_assert(p.manhattanDist() == 7);

    // Also works at runtime
    int n;
    std::cin >> n;
    std::cout << "square(" << n << ") = " << square(n) << "\n";

    return 0;
}
```

### Relaxed constexpr (C++14)

```cpp
#include <iostream>
#include <array>

// C++14: constexpr functions can have multiple statements
constexpr int fibonacci(int n) {
    if (n <= 1) return n;
    int a = 0, b = 1;
    for (int i = 2; i <= n; ++i) {
        int temp = a + b;
        a = b;
        b = temp;
    }
    return b;
}

// constexpr with loops and local variables
constexpr int sumOfSquares(int n) {
    int sum = 0;
    for (int i = 1; i <= n; ++i) {
        sum += i * i;
    }
    return sum;
}

// constexpr array generation
template<std::size_t N>
constexpr std::array<int, N> generateFibArray() {
    std::array<int, N> arr{};
    for (std::size_t i = 0; i < N; ++i) {
        arr[i] = fibonacci(i);
    }
    return arr;
}

int main() {
    constexpr int fib10 = fibonacci(10);
    static_assert(fib10 == 55);

    constexpr int ss = sumOfSquares(5);
    static_assert(ss == 55);  // 1+4+9+16+25

    constexpr auto fibs = generateFibArray<10>();
    for (int f : fibs) {
        std::cout << f << " ";  // 0 1 1 2 3 5 8 13 21 34
    }
    std::cout << "\n";

    return 0;
}
```

---

## 7. enum class

Scoped enumerations prevent name collisions and implicit conversions.

```cpp
#include <iostream>

// Old-style enum (C++03): pollutes namespace, implicit conversions
enum OldColor { RED, GREEN, BLUE };
// enum TrafficLight { RED, YELLOW, GREEN };  // ERROR: RED and GREEN conflict

// Scoped enum (C++11)
enum class Color { Red, Green, Blue };
enum class TrafficLight { Red, Yellow, Green };  // No conflict!

// With underlying type
enum class ErrorCode : uint8_t {
    Success = 0,
    NotFound = 1,
    Timeout = 2,
    Internal = 255
};

// Forward declaration (only possible with scoped enums or explicit type)
enum class Direction : int;

void handleError(ErrorCode code) {
    switch (code) {
        case ErrorCode::Success:
            std::cout << "OK\n";
            break;
        case ErrorCode::NotFound:
            std::cout << "Not found\n";
            break;
        default:
            std::cout << "Error: " << static_cast<int>(code) << "\n";
    }
}

int main() {
    Color c = Color::Red;
    TrafficLight t = TrafficLight::Red;

    // No implicit conversion to int
    // int x = c;              // ERROR
    int x = static_cast<int>(c);  // OK: explicit cast

    // Type-safe comparison
    // if (c == t) {}          // ERROR: different types
    if (c == Color::Red) {
        std::cout << "It's red\n";
    }

    handleError(ErrorCode::NotFound);

    // Underlying type
    std::cout << "Size of ErrorCode: " << sizeof(ErrorCode) << "\n";  // 1

    return 0;
}
```

---

## 8. static_assert

Compile-time assertions that produce clear error messages.

```cpp
#include <iostream>
#include <type_traits>

// Check sizes at compile time
static_assert(sizeof(int) >= 4, "int must be at least 4 bytes");
static_assert(sizeof(void*) == 8, "64-bit platform required");

// Check type properties
template<typename T>
class NumericContainer {
    static_assert(std::is_arithmetic_v<T>,
                  "NumericContainer requires an arithmetic type");

    T value_;

public:
    NumericContainer(T v) : value_(v) {}
    T get() const { return value_; }
};

// Check alignment
template<typename T>
class AlignedStorage {
    static_assert(alignof(T) <= 64,
                  "Type alignment must not exceed 64 bytes");
    alignas(T) char storage[sizeof(T)];
};

// C++17: static_assert without message
// static_assert(sizeof(int) == 4);

int main() {
    NumericContainer<int> ic(42);      // OK
    NumericContainer<double> dc(3.14); // OK
    // NumericContainer<std::string> sc("hi"); // ERROR at compile time

    return 0;
}
```

---

## 9. Move Semantics Integration

A brief recap of how move semantics integrate with C++11/14 features.

```cpp
#include <iostream>
#include <string>
#include <vector>
#include <utility>

int main() {
    // std::move in practice
    std::string s = "Hello, World!";
    std::vector<std::string> vec;

    vec.push_back(s);              // Copy
    vec.push_back(std::move(s));   // Move (s is now empty)

    // emplace_back: construct in-place (avoids even the move)
    vec.emplace_back("Constructed in-place");

    // Move semantics with unique_ptr
    auto ptr = std::make_unique<int>(42);
    auto ptr2 = std::move(ptr);   // Ownership transfer

    // Returning by value: compiler applies RVO
    auto makeVec = []() -> std::vector<int> {
        std::vector<int> v = {1, 2, 3};
        return v;  // NRVO or move
    };
    auto result = makeVec();

    return 0;
}
```

---

## 10. C++14 Additions

### std::make_unique

```cpp
#include <memory>
#include <iostream>

int main() {
    // C++11: no make_unique
    std::unique_ptr<int> p1(new int(42));

    // C++14: make_unique (preferred)
    auto p2 = std::make_unique<int>(42);
    auto arr = std::make_unique<int[]>(10);

    std::cout << *p2 << "\n";  // 42

    return 0;
}
```

### Variable Templates

```cpp
#include <iostream>

// Variable template (C++14)
template<typename T>
constexpr T pi = T(3.14159265358979323846L);

template<typename T>
constexpr T e = T(2.71828182845904523536L);

// Used by standard library: std::is_integral_v<T> is a variable template
// template<typename T>
// inline constexpr bool is_integral_v = is_integral<T>::value;

int main() {
    std::cout << "float pi:  " << pi<float> << "\n";
    std::cout << "double pi: " << pi<double> << "\n";
    std::cout << "double e:  " << e<double> << "\n";

    return 0;
}
```

### Return Type Deduction

```cpp
#include <iostream>
#include <vector>

// C++14: compiler deduces return type
auto multiply(int a, int b) {
    return a * b;  // Deduced as int
}

auto getString() {
    return std::string("Hello");
}

// Note: all return paths must return the same type
auto conditional(bool flag) {
    if (flag) return 1;
    return 2;
    // return 1.0;  // ERROR: inconsistent return types
}
```

### Binary Literals and Digit Separators

```cpp
#include <iostream>

int main() {
    // Binary literal (C++14)
    int bits = 0b1010'1010;       // 170
    int mask = 0b1111'0000;       // 240

    // Digit separators (any numeric literal)
    int million = 1'000'000;
    double pi = 3.141'592'653;
    int hex = 0xFF'FF;
    long long big = 1'000'000'000'000LL;

    std::cout << "bits: " << bits << "\n";
    std::cout << "million: " << million << "\n";
    std::cout << "pi: " << pi << "\n";

    return 0;
}
```

---

## Summary

| Feature | Version | Key Benefit |
|---------|---------|-------------|
| `auto` / `decltype` | C++11 | Reduce type verbosity |
| Uniform initialization | C++11 | Prevent narrowing, consistent syntax |
| Range-based for | C++11 | Cleaner iteration |
| `nullptr` | C++11 | Type-safe null pointer |
| Lambda expressions | C++11 | Inline functions, closures |
| `constexpr` | C++11 | Compile-time evaluation |
| `enum class` | C++11 | Scoped, type-safe enumerations |
| `static_assert` | C++11 | Compile-time assertions |
| Move semantics | C++11 | Efficient resource transfer |
| Generic lambdas | C++14 | Template-like lambdas |
| `decltype(auto)` | C++14 | Exact type deduction |
| Relaxed `constexpr` | C++14 | Multi-statement compile-time functions |
| `std::make_unique` | C++14 | Safe unique_ptr creation |
| Variable templates | C++14 | Type-parameterized constants |
| Binary literals | C++14 | `0b1010` notation |
| Digit separators | C++14 | Readable numeric literals |

---

## Exercises

### Exercise 1: Lambda Accumulator

Write a function that returns a lambda which accumulates values across calls (stateful lambda using mutable capture).

### Exercise 2: constexpr Lookup Table

Create a `constexpr` function that generates a lookup table (e.g., sine values at integer degrees) at compile time and stores it in a `std::array`.

### Exercise 3: Generic Print Container

Write a generic lambda that prints any container using range-based for, with a configurable delimiter and prefix/suffix.

### Exercise 4: Type-Safe Builder Pattern

Using `auto` return types and move semantics, implement a builder pattern where each `.set_X()` call returns the builder by move and the final `.build()` returns the constructed object.

### Exercise 5: Enum-Based State Machine

Implement a simple state machine using `enum class` for states and events, with `static_assert` to verify state transition tables at compile time.

---

## Next Steps

C++17 brought structured bindings, `std::optional`, `std::variant`, and `std::filesystem` among many other features. Let's explore them in [07_Modern_CPP_17.md](./07_Modern_CPP_17.md).
