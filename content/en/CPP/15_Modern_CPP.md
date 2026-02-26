# Modern C++ (C++11/14/17/20)

**Previous**: [Smart Pointers and Memory Management](./14_Smart_Pointers_Memory.md) | **Next**: [Multithreading and Concurrency](./16_Multithreading_Concurrency.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Use C++11 core features: `auto`, range-based for loops, `nullptr`, initializer lists, and lambdas
2. Implement move semantics with move constructors and move assignment operators
3. Apply `constexpr` to evaluate expressions at compile time
4. Write generic lambdas and use return-type deduction introduced in C++14
5. Decompose return values with C++17 structured bindings and use `if constexpr` for compile-time branching
6. Leverage C++17 vocabulary types (`std::optional`, `std::variant`, `std::string_view`)
7. Use C++20 features including concepts, ranges pipelines, the spaceship operator, and `std::format`

---

C++ has evolved more in the last decade than in the preceding twenty years combined. Each standard--C++11, 14, 17, and 20--has introduced features that make the language safer, more expressive, and often faster without changing a line of runtime code. Understanding these features is not optional: modern libraries and codebases expect them, and ignoring them means writing more code that is harder to read and more prone to bugs.

## 1. C++ Version Evolution

```
┌─────────────────────────────────────────────────────────────┐
│                    C++ Standard History                      │
├─────────────────────────────────────────────────────────────┤
│  1998   2003   2011        2014    2017    2020    2023     │
│   │      │      │           │       │       │       │       │
│   ▼      ▼      ▼           ▼       ▼       ▼       ▼       │
│ C++98  C++03  C++11       C++14   C++17   C++20   C++23     │
│        (bug   (major      (improve)(major  (major           │
│        fix)   update)      ments)  update) update)          │
└─────────────────────────────────────────────────────────────┘
```

### Compilation Options

```bash
# C++11
g++ -std=c++11 main.cpp -o main

# C++14
g++ -std=c++14 main.cpp -o main

# C++17
g++ -std=c++17 main.cpp -o main

# C++20
g++ -std=c++20 main.cpp -o main
```

---

## 2. C++11 Key Features

### auto Keyword

```cpp
#include <iostream>
#include <vector>
#include <map>

int main() {
    // Automatic type inference
    auto x = 42;          // int
    auto y = 3.14;        // double
    auto s = "Hello";     // const char*

    std::vector<int> vec = {1, 2, 3, 4, 5};

    // Use auto instead of long type names
    auto it = vec.begin();  // std::vector<int>::iterator

    std::map<std::string, int> ages = {
        {"Alice", 25},
        {"Bob", 30}
    };

    // Complex types simplified with auto
    for (auto& pair : ages) {
        std::cout << pair.first << ": " << pair.second << std::endl;
    }

    // Function return type inference
    auto add = [](int a, int b) { return a + b; };
    std::cout << add(3, 4) << std::endl;  // 7

    return 0;
}
```

### Range-Based For Loop

```cpp
#include <iostream>
#include <vector>
#include <array>

int main() {
    std::vector<int> vec = {1, 2, 3, 4, 5};

    // Copy by value
    for (int x : vec) {
        std::cout << x << " ";
    }
    std::cout << std::endl;

    // Reference (modifiable)
    for (int& x : vec) {
        x *= 2;
    }

    // Const reference (read-only)
    for (const int& x : vec) {
        std::cout << x << " ";
    }
    std::cout << std::endl;

    // Works with arrays too
    int arr[] = {10, 20, 30};
    for (int x : arr) {
        std::cout << x << " ";
    }
    std::cout << std::endl;

    // With initializer list
    for (int x : {100, 200, 300}) {
        std::cout << x << " ";
    }
    std::cout << std::endl;

    return 0;
}
```

### nullptr

```cpp
#include <iostream>

void foo(int n) {
    std::cout << "int: " << n << std::endl;
}

void foo(int* p) {
    std::cout << "pointer" << std::endl;
}

int main() {
    // Before C++11: NULL is defined as 0
    // foo(NULL);  // Ambiguous!

    // C++11: Use nullptr
    foo(nullptr);  // pointer
    foo(0);        // int: 0

    int* p = nullptr;
    if (p == nullptr) {
        std::cout << "p is null" << std::endl;
    }

    return 0;
}
```

### Initializer List

```cpp
#include <iostream>
#include <vector>
#include <map>
#include <initializer_list>

class MyContainer {
private:
    std::vector<int> data;

public:
    MyContainer(std::initializer_list<int> list)
        : data(list) {
        std::cout << "Constructor called" << std::endl;
    }

    void print() const {
        for (int x : data) {
            std::cout << x << " ";
        }
        std::cout << std::endl;
    }
};

int main() {
    // Uniform Initialization
    int a{42};
    double b{3.14};
    std::string c{"Hello"};

    // Container initialization
    std::vector<int> vec = {1, 2, 3, 4, 5};
    std::map<std::string, int> ages = {
        {"Alice", 25},
        {"Bob", 30}
    };

    // Custom class
    MyContainer mc = {10, 20, 30, 40};
    mc.print();  // 10 20 30 40

    // Array
    int arr[] = {1, 2, 3};

    // Braces prevent narrowing
    // int x{3.14};  // Error! narrowing conversion

    return 0;
}
```

### Lambda Expressions

```cpp
#include <iostream>
#include <vector>
#include <algorithm>
#include <functional>

int main() {
    // Basic lambda
    auto hello = []() {
        std::cout << "Hello, Lambda!" << std::endl;
    };
    hello();

    // Parameters and return
    auto add = [](int a, int b) -> int {
        return a + b;
    };
    std::cout << add(3, 4) << std::endl;  // 7

    // Capture (access external variables)
    int x = 10;
    int y = 20;

    // Capture by value
    auto byValue = [x, y]() {
        std::cout << x + y << std::endl;
    };

    // Capture by reference
    auto byRef = [&x, &y]() {
        x++;
        y++;
    };

    // Capture all by value
    auto allByValue = [=]() {
        std::cout << x + y << std::endl;
    };

    // Capture all by reference
    auto allByRef = [&]() {
        x = 100;
        y = 200;
    };

    // Mixed capture
    auto mixed = [=, &x]() {  // y by value, x by reference
        x = 50;
        std::cout << y << std::endl;
    };

    // Use with STL algorithms
    std::vector<int> vec = {3, 1, 4, 1, 5, 9, 2, 6};

    // Sort
    std::sort(vec.begin(), vec.end(),
        [](int a, int b) { return a > b; });  // Descending

    // Print
    std::for_each(vec.begin(), vec.end(),
        [](int x) { std::cout << x << " "; });
    std::cout << std::endl;

    return 0;
}
```

### Move Semantics

```cpp
#include <iostream>
#include <string>
#include <vector>
#include <utility>

class Buffer {
private:
    int* data;
    size_t size;

public:
    // Constructor
    Buffer(size_t n) : data(new int[n]), size(n) {
        std::cout << "Constructor" << std::endl;
    }

    // Destructor
    ~Buffer() {
        delete[] data;
        std::cout << "Destructor" << std::endl;
    }

    // Copy constructor
    Buffer(const Buffer& other)
        : data(new int[other.size]), size(other.size) {
        std::copy(other.data, other.data + size, data);
        std::cout << "Copy constructor" << std::endl;
    }

    // Move constructor (C++11)
    Buffer(Buffer&& other) noexcept
        : data(other.data), size(other.size) {
        other.data = nullptr;
        other.size = 0;
        std::cout << "Move constructor" << std::endl;
    }

    // Copy assignment operator
    Buffer& operator=(const Buffer& other) {
        if (this != &other) {
            delete[] data;
            size = other.size;
            data = new int[size];
            std::copy(other.data, other.data + size, data);
        }
        std::cout << "Copy assignment" << std::endl;
        return *this;
    }

    // Move assignment operator (C++11)
    Buffer& operator=(Buffer&& other) noexcept {
        if (this != &other) {
            delete[] data;
            data = other.data;
            size = other.size;
            other.data = nullptr;
            other.size = 0;
        }
        std::cout << "Move assignment" << std::endl;
        return *this;
    }
};

Buffer createBuffer() {
    return Buffer(100);  // Move optimization
}

int main() {
    Buffer b1(10);

    Buffer b2 = b1;              // Copy constructor
    Buffer b3 = std::move(b1);   // Move constructor
    // b1 is now empty

    Buffer b4 = createBuffer();  // Move constructor (may be elided by RVO)

    return 0;
}
```

### Smart Pointers

```cpp
#include <iostream>
#include <memory>

class Resource {
public:
    Resource() { std::cout << "Created" << std::endl; }
    ~Resource() { std::cout << "Destroyed" << std::endl; }
};

int main() {
    // unique_ptr: Exclusive ownership
    std::unique_ptr<Resource> p1(new Resource());
    auto p2 = std::make_unique<Resource>();  // C++14

    // shared_ptr: Shared ownership
    auto p3 = std::make_shared<Resource>();
    auto p4 = p3;  // Reference count incremented
    std::cout << "Reference count: " << p3.use_count() << std::endl;

    // weak_ptr: Weak reference
    std::weak_ptr<Resource> weak = p3;
    if (auto sp = weak.lock()) {
        std::cout << "Object accessible" << std::endl;
    }

    return 0;
}
```

### constexpr

```cpp
#include <iostream>
#include <array>

// Compile-time constant function
constexpr int factorial(int n) {
    return (n <= 1) ? 1 : n * factorial(n - 1);
}

constexpr int square(int x) {
    return x * x;
}

// Compile-time constant class
class Point {
public:
    int x, y;
    constexpr Point(int x, int y) : x(x), y(y) {}
    constexpr int getX() const { return x; }
    constexpr int getY() const { return y; }
};

int main() {
    // Computed at compile time
    constexpr int fact5 = factorial(5);  // 120
    constexpr int sq10 = square(10);     // 100

    // Can be used as array size
    std::array<int, factorial(4)> arr;  // Size 24

    // Compile-time constant object
    constexpr Point p(3, 4);
    static_assert(p.getX() == 3, "X should be 3");

    std::cout << "5! = " << fact5 << std::endl;
    std::cout << "10^2 = " << sq10 << std::endl;

    return 0;
}
```

---

## 3. C++14 Key Features

### Generic Lambdas

```cpp
#include <iostream>
#include <vector>
#include <string>

int main() {
    // C++14: auto parameters
    auto print = [](auto x) {
        std::cout << x << std::endl;
    };

    print(42);          // int
    print(3.14);        // double
    print("Hello");     // const char*

    // Multiple type parameters
    auto add = [](auto a, auto b) {
        return a + b;
    };

    std::cout << add(1, 2) << std::endl;       // 3
    std::cout << add(1.5, 2.5) << std::endl;   // 4
    std::cout << add(std::string("Hello, "), std::string("World!")) << std::endl;

    return 0;
}
```

### Variable Templates

```cpp
#include <iostream>

// Variable template (C++14)
template<typename T>
constexpr T pi = T(3.141592653589793238462643383);

template<typename T>
constexpr T e = T(2.718281828459045235360287471);

int main() {
    std::cout << "float pi: " << pi<float> << std::endl;
    std::cout << "double pi: " << pi<double> << std::endl;
    std::cout << "double e: " << e<double> << std::endl;

    return 0;
}
```

### [[deprecated]] Attribute

```cpp
#include <iostream>

// Function deprecation warning
[[deprecated("Use newFunction() instead")]]
void oldFunction() {
    std::cout << "Old function" << std::endl;
}

void newFunction() {
    std::cout << "New function" << std::endl;
}

// Classes too
class [[deprecated("Use NewClass instead")]] OldClass {};

int main() {
    // oldFunction();  // Compiler warning generated
    newFunction();

    return 0;
}
```

### Binary Literals and Digit Separators

```cpp
#include <iostream>

int main() {
    // Binary literal (C++14)
    int binary = 0b1010'1010;  // 170
    int hex = 0xFF'FF;         // 65535

    // Digit separators
    int million = 1'000'000;
    double pi = 3.141'592'653;
    int binary2 = 0b1111'0000'1111'0000;

    std::cout << "binary: " << binary << std::endl;
    std::cout << "million: " << million << std::endl;
    std::cout << "pi: " << pi << std::endl;

    return 0;
}
```

### Return Type Deduction

```cpp
#include <iostream>
#include <vector>

// C++14: Automatic return type deduction
auto multiply(int a, int b) {
    return a * b;  // Deduced as int
}

auto getString() {
    return std::string("Hello");  // Deduced as std::string
}

// Recursive also possible
auto factorial(int n) -> int {  // Recursion needs explicit type
    return (n <= 1) ? 1 : n * factorial(n - 1);
}

int main() {
    auto result = multiply(3, 4);
    std::cout << result << std::endl;  // 12

    auto str = getString();
    std::cout << str << std::endl;  // Hello

    return 0;
}
```

---

## 4. C++17 Key Features

### Structured Bindings

```cpp
#include <iostream>
#include <tuple>
#include <map>
#include <array>

std::tuple<int, double, std::string> getData() {
    return {42, 3.14, "Hello"};
}

int main() {
    // Tuple decomposition
    auto [num, pi, str] = getData();
    std::cout << num << ", " << pi << ", " << str << std::endl;

    // Pair decomposition
    std::pair<int, std::string> p = {1, "Alice"};
    auto [id, name] = p;
    std::cout << id << ": " << name << std::endl;

    // Array decomposition
    int arr[] = {1, 2, 3};
    auto [a, b, c] = arr;
    std::cout << a << ", " << b << ", " << c << std::endl;

    // Map iteration
    std::map<std::string, int> ages = {
        {"Alice", 25},
        {"Bob", 30}
    };

    for (auto& [name, age] : ages) {
        std::cout << name << ": " << age << std::endl;
    }

    // Struct decomposition
    struct Point { int x, y; };
    Point pt = {10, 20};
    auto [x, y] = pt;
    std::cout << "Point: " << x << ", " << y << std::endl;

    return 0;
}
```

### if/switch with Initializer

```cpp
#include <iostream>
#include <map>
#include <mutex>

std::map<int, std::string> database = {
    {1, "Alice"},
    {2, "Bob"}
};

std::mutex mtx;

int main() {
    // if with initializer
    if (auto it = database.find(1); it != database.end()) {
        std::cout << "Found: " << it->second << std::endl;
    }

    // switch with initializer
    switch (auto x = 2 * 3; x) {
        case 6:
            std::cout << "x is 6" << std::endl;
            break;
        default:
            std::cout << "x is " << x << std::endl;
    }

    // Use with lock
    if (std::lock_guard<std::mutex> lock(mtx); true) {
        // Work while mutex is locked
        std::cout << "Protected section" << std::endl;
    }

    return 0;
}
```

### if constexpr

```cpp
#include <iostream>
#include <type_traits>
#include <string>

template<typename T>
void process(T value) {
    if constexpr (std::is_integral_v<T>) {
        std::cout << "Integer: " << value * 2 << std::endl;
    } else if constexpr (std::is_floating_point_v<T>) {
        std::cout << "Float: " << value / 2 << std::endl;
    } else if constexpr (std::is_same_v<T, std::string>) {
        std::cout << "String: " << value.length() << " characters" << std::endl;
    } else {
        std::cout << "Other: " << value << std::endl;
    }
}

int main() {
    process(10);                    // Integer: 20
    process(3.14);                  // Float: 1.57
    process(std::string("Hello"));  // String: 5 characters
    process("Hello");               // Other: Hello

    return 0;
}
```

### std::optional

```cpp
#include <iostream>
#include <optional>
#include <string>

std::optional<int> divide(int a, int b) {
    if (b == 0) {
        return std::nullopt;
    }
    return a / b;
}

std::optional<std::string> findUser(int id) {
    if (id == 1) return "Alice";
    if (id == 2) return "Bob";
    return std::nullopt;
}

int main() {
    // Basic usage
    auto result = divide(10, 2);
    if (result) {
        std::cout << "Result: " << *result << std::endl;
    }

    auto result2 = divide(10, 0);
    std::cout << "has_value: " << result2.has_value() << std::endl;  // 0

    // Provide default value with value_or
    std::cout << divide(10, 3).value_or(-1) << std::endl;  // 3
    std::cout << divide(10, 0).value_or(-1) << std::endl;  // -1

    // String
    auto user = findUser(1);
    if (user) {
        std::cout << "User: " << *user << std::endl;
    }

    std::cout << findUser(3).value_or("Unknown") << std::endl;  // Unknown

    return 0;
}
```

### std::variant

```cpp
#include <iostream>
#include <variant>
#include <string>

int main() {
    // Store one of multiple types
    std::variant<int, double, std::string> v;

    v = 42;
    std::cout << std::get<int>(v) << std::endl;

    v = 3.14;
    std::cout << std::get<double>(v) << std::endl;

    v = "Hello";
    std::cout << std::get<std::string>(v) << std::endl;

    // Check current type
    if (std::holds_alternative<std::string>(v)) {
        std::cout << "It's a string" << std::endl;
    }

    // Access by index
    std::cout << "index: " << v.index() << std::endl;  // 2

    // Visit pattern
    std::variant<int, double, std::string> values[] = {
        42, 3.14, std::string("Hello")
    };

    for (auto& val : values) {
        std::visit([](auto&& arg) {
            std::cout << arg << std::endl;
        }, val);
    }

    return 0;
}
```

### std::filesystem

```cpp
#include <iostream>
#include <filesystem>

namespace fs = std::filesystem;

int main() {
    // Working with paths
    fs::path p = "/home/user/documents/file.txt";

    std::cout << "filename: " << p.filename() << std::endl;
    std::cout << "stem: " << p.stem() << std::endl;
    std::cout << "extension: " << p.extension() << std::endl;
    std::cout << "parent_path: " << p.parent_path() << std::endl;

    // Current directory
    std::cout << "Current path: " << fs::current_path() << std::endl;

    // Check file/directory existence
    fs::path testPath = ".";
    if (fs::exists(testPath)) {
        std::cout << testPath << " exists" << std::endl;
    }

    // Directory traversal
    std::cout << "\n=== Current directory contents ===" << std::endl;
    for (const auto& entry : fs::directory_iterator(".")) {
        std::cout << entry.path().filename();
        if (fs::is_directory(entry)) {
            std::cout << " [DIR]";
        } else {
            std::cout << " [" << fs::file_size(entry) << " bytes]";
        }
        std::cout << std::endl;
    }

    // Path concatenation
    fs::path dir = "/home/user";
    fs::path file = "document.txt";
    fs::path full = dir / file;
    std::cout << "Combined path: " << full << std::endl;

    return 0;
}
```

### std::string_view

```cpp
#include <iostream>
#include <string>
#include <string_view>

// Reference string without copying
void printView(std::string_view sv) {
    std::cout << "View: " << sv << std::endl;
    std::cout << "Length: " << sv.length() << std::endl;
}

int main() {
    // Create from various string types
    std::string str = "Hello, World!";
    const char* cstr = "Hello from C!";
    char arr[] = "Hello from array!";

    printView(str);
    printView(cstr);
    printView(arr);
    printView("Literal string");

    // Substring (no copy)
    std::string_view sv = "Hello, World!";
    std::string_view sub = sv.substr(0, 5);
    std::cout << "Substring: " << sub << std::endl;  // Hello

    // Warning: Dangling if original disappears!
    // std::string_view bad;
    // {
    //     std::string temp = "temporary";
    //     bad = temp;
    // }
    // std::cout << bad << std::endl;  // Undefined behavior!

    return 0;
}
```

---

## 5. C++20 Key Features

### Concepts

```cpp
#include <iostream>
#include <concepts>
#include <vector>

// Define concept
template<typename T>
concept Numeric = std::is_arithmetic_v<T>;

template<typename T>
concept Printable = requires(T t) {
    { std::cout << t };
};

template<typename T>
concept Container = requires(T t) {
    t.begin();
    t.end();
    t.size();
};

// Use concept
template<Numeric T>
T add(T a, T b) {
    return a + b;
}

// requires clause
template<typename T>
requires Printable<T>
void print(const T& value) {
    std::cout << value << std::endl;
}

// Abbreviated form
void printSize(Container auto& c) {
    std::cout << "Size: " << c.size() << std::endl;
}

int main() {
    std::cout << add(1, 2) << std::endl;       // 3
    std::cout << add(1.5, 2.5) << std::endl;   // 4
    // add("a", "b");  // Error: Numeric constraint not satisfied

    print(42);
    print("Hello");

    std::vector<int> vec = {1, 2, 3};
    printSize(vec);  // Size: 3

    return 0;
}
```

### Ranges

```cpp
#include <iostream>
#include <vector>
#include <ranges>
#include <algorithm>

int main() {
    std::vector<int> numbers = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

    // Pipeline style
    auto result = numbers
        | std::views::filter([](int n) { return n % 2 == 0; })  // Even
        | std::views::transform([](int n) { return n * n; })     // Square
        | std::views::take(3);                                    // Only 3

    std::cout << "Result: ";
    for (int n : result) {
        std::cout << n << " ";  // 4 16 36
    }
    std::cout << std::endl;

    // Various views
    // iota: Generate number range
    for (int n : std::views::iota(1, 6)) {
        std::cout << n << " ";  // 1 2 3 4 5
    }
    std::cout << std::endl;

    // reverse
    for (int n : std::views::reverse(numbers)) {
        std::cout << n << " ";  // 10 9 8 7 6 5 4 3 2 1
    }
    std::cout << std::endl;

    // drop: Skip first n
    for (int n : numbers | std::views::drop(5)) {
        std::cout << n << " ";  // 6 7 8 9 10
    }
    std::cout << std::endl;

    return 0;
}
```

### Three-Way Comparison (Spaceship Operator)

```cpp
#include <iostream>
#include <compare>
#include <string>

class Version {
public:
    int major, minor, patch;

    Version(int ma, int mi, int pa)
        : major(ma), minor(mi), patch(pa) {}

    // Three-way comparison operator (C++20)
    auto operator<=>(const Version& other) const = default;

    // == operator also auto-generated
};

int main() {
    // Basic types
    int a = 5, b = 10;
    auto result = a <=> b;

    if (result < 0) {
        std::cout << "a < b" << std::endl;
    } else if (result > 0) {
        std::cout << "a > b" << std::endl;
    } else {
        std::cout << "a == b" << std::endl;
    }

    // Custom class
    Version v1{1, 2, 3};
    Version v2{1, 3, 0};

    if (v1 < v2) {
        std::cout << "v1 < v2" << std::endl;
    }

    if (v1 == Version{1, 2, 3}) {
        std::cout << "v1 == 1.2.3" << std::endl;
    }

    return 0;
}
```

### Modules

```cpp
// math.cppm (module interface)
export module math;

export int add(int a, int b) {
    return a + b;
}

export int multiply(int a, int b) {
    return a * b;
}

// main.cpp
import math;
import <iostream>;

int main() {
    std::cout << add(3, 4) << std::endl;       // 7
    std::cout << multiply(3, 4) << std::endl;  // 12
    return 0;
}

// Compilation (varies by compiler)
// g++ -std=c++20 -fmodules-ts -c math.cppm
// g++ -std=c++20 -fmodules-ts main.cpp math.o -o main
```

### Coroutines

```cpp
#include <iostream>
#include <coroutine>

// Simple Generator example
template<typename T>
struct Generator {
    struct promise_type {
        T value;

        Generator get_return_object() {
            return Generator{
                std::coroutine_handle<promise_type>::from_promise(*this)
            };
        }
        std::suspend_always initial_suspend() { return {}; }
        std::suspend_always final_suspend() noexcept { return {}; }
        std::suspend_always yield_value(T v) {
            value = v;
            return {};
        }
        void return_void() {}
        void unhandled_exception() { std::terminate(); }
    };

    std::coroutine_handle<promise_type> handle;

    Generator(std::coroutine_handle<promise_type> h) : handle(h) {}
    ~Generator() { if (handle) handle.destroy(); }

    bool next() {
        if (!handle.done()) {
            handle.resume();
        }
        return !handle.done();
    }

    T value() { return handle.promise().value; }
};

Generator<int> range(int start, int end) {
    for (int i = start; i < end; i++) {
        co_yield i;
    }
}

int main() {
    auto gen = range(1, 5);

    while (gen.next()) {
        std::cout << gen.value() << " ";  // 1 2 3 4
    }
    std::cout << std::endl;

    return 0;
}
```

### std::format (C++20)

```cpp
#include <iostream>
#include <format>
#include <string>

int main() {
    // Basic usage
    std::string s1 = std::format("Hello, {}!", "World");
    std::cout << s1 << std::endl;  // Hello, World!

    // Index specification
    std::string s2 = std::format("{1} + {0} = {2}", 10, 20, 30);
    std::cout << s2 << std::endl;  // 20 + 10 = 30

    // Alignment and width
    std::string s3 = std::format("|{:>10}|", 42);    // Right align
    std::string s4 = std::format("|{:<10}|", 42);    // Left align
    std::string s5 = std::format("|{:^10}|", 42);    // Center align
    std::cout << s3 << std::endl;  // |        42|
    std::cout << s4 << std::endl;  // |42        |
    std::cout << s5 << std::endl;  // |    42    |

    // Number formats
    std::string s6 = std::format("{:b}", 42);   // Binary
    std::string s7 = std::format("{:x}", 255);  // Hexadecimal
    std::string s8 = std::format("{:.2f}", 3.14159);  // 2 decimal places
    std::cout << s6 << std::endl;  // 101010
    std::cout << s7 << std::endl;  // ff
    std::cout << s8 << std::endl;  // 3.14

    return 0;
}
```

---

## 6. Best Practices

### Code Style Recommendations

```cpp
#include <iostream>
#include <memory>
#include <vector>
#include <string>

// 1. Use auto appropriately
auto calculate() {
    return 42;  // Use auto when clear
}

// 2. Use const actively
void printVector(const std::vector<int>& vec) {
    for (const auto& x : vec) {
        std::cout << x << " ";
    }
}

// 3. Use smart pointers
class Resource {
public:
    void use() { std::cout << "Using" << std::endl; }
};

void goodMemoryManagement() {
    auto ptr = std::make_unique<Resource>();
    ptr->use();
    // Auto-freed
}

// 4. Use range-based for
void iterateModern(const std::vector<int>& vec) {
    for (const auto& item : vec) {
        std::cout << item << std::endl;
    }
}

// 5. Use initializer lists
class Person {
private:
    std::string name;
    int age;

public:
    Person(std::string n, int a)
        : name(std::move(n)), age(a) {}  // Use move
};

// 6. Use noexcept appropriately
void safeFunction() noexcept {
    // Function that doesn't throw exceptions
}

// 7. Use constexpr
constexpr int maxSize = 100;
constexpr int square(int x) { return x * x; }

int main() {
    std::vector<int> vec = {1, 2, 3, 4, 5};
    printVector(vec);
    std::cout << std::endl;

    goodMemoryManagement();

    iterateModern(vec);

    constexpr int result = square(10);
    std::cout << "10^2 = " << result << std::endl;

    return 0;
}
```

---

## 7. Summary

| Version | Key Features |
|---------|--------------|
| C++11 | auto, lambda, move semantics, smart pointers, nullptr, constexpr |
| C++14 | Generic lambdas, variable templates, return type deduction |
| C++17 | Structured bindings, if constexpr, optional, variant, filesystem |
| C++20 | Concepts, Ranges, three-way comparison, modules, coroutines, std::format |

---

## 8. Exercises

### Exercise 1: Modern C++ Refactoring

Refactor existing C++98/03 style code to C++17 or later.

### Exercise 2: Type-Safe Configuration System

Implement a type-safe configuration management system using `std::variant` and `std::optional`.

### Exercise 3: Pipeline Processor

Implement a data processing pipeline using C++20 Ranges.

---

## Learning Complete

You have completed learning C++ from basics to modern C++!

### Recommended Review Order

1. Basics review: 01-06 (variables, functions, pointers)
2. OOP review: 07-09 (classes, inheritance)
3. STL review: 10-11 (containers, algorithms)
4. Advanced review: 12-15 (templates, smart pointers, modern C++)

### Next Learning Recommendations

- Design patterns
- Multithreading
- Network programming
- Actual project work
