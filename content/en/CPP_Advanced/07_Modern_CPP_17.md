# Modern C++ -- C++17

**Previous**: [Modern C++ (C++11/14)](./06_Modern_CPP_11_14.md) | **Next**: [C++20 Concepts](./08_CPP20_Concepts.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Decompose structured data using structured bindings with structs, pairs, tuples, arrays, and maps
2. Apply `if constexpr` for compile-time branching that replaces SFINAE patterns
3. Use vocabulary types `std::optional`, `std::variant`, and `std::any` for expressive data modeling
4. Leverage `std::string_view` for zero-copy string references
5. Perform file system operations using `std::filesystem`
6. Apply CTAD, fold expressions, inline variables, and other C++17 features

---

C++17 delivered a collection of features that dramatically improved everyday C++ coding. Structured bindings eliminated the tedium of `std::get` and `std::tie`. Vocabulary types (`optional`, `variant`, `any`) replaced ad-hoc patterns with standard, well-tested alternatives. `std::filesystem` brought portable file operations into the standard library. And `if constexpr` made template code readable again. This lesson covers each feature in depth with practical examples.

## 1. Structured Bindings

Structured bindings let you decompose aggregates, pairs, tuples, and arrays into named variables.

```cpp
#include <iostream>
#include <tuple>
#include <map>
#include <array>

std::tuple<int, double, std::string> getData() {
    return {42, 3.14, "Hello"};
}

struct Point { double x, y, z; };

int main() {
    // Tuple decomposition
    auto [num, pi, str] = getData();
    std::cout << num << ", " << pi << ", " << str << "\n";

    // Pair decomposition
    std::pair<int, std::string> p = {1, "Alice"};
    auto [id, name] = p;
    std::cout << id << ": " << name << "\n";

    // Array decomposition
    int arr[] = {10, 20, 30};
    auto [a, b, c] = arr;
    std::cout << a << ", " << b << ", " << c << "\n";

    // Struct decomposition
    Point pt{1.0, 2.5, 3.7};
    auto [x, y, z] = pt;
    std::cout << "Point: " << x << ", " << y << ", " << z << "\n";

    // Map iteration (most common use)
    std::map<std::string, int> ages = {
        {"Alice", 25}, {"Bob", 30}, {"Carol", 28}
    };
    for (const auto& [name, age] : ages) {
        std::cout << name << " is " << age << "\n";
    }

    // With references (modifiable)
    auto& [rx, ry, rz] = pt;
    rx = 100.0;  // Modifies pt.x
    std::cout << "Modified: " << pt.x << "\n";  // 100

    // In if-statement
    std::map<int, std::string> lookup = {{1, "one"}, {2, "two"}};
    if (auto [it, inserted] = lookup.insert({3, "three"}); inserted) {
        std::cout << "Inserted: " << it->second << "\n";
    }

    return 0;
}
```

---

## 2. if constexpr

Compile-time branching that discards the false branch entirely, preventing instantiation errors.

```cpp
#include <iostream>
#include <type_traits>
#include <string>
#include <vector>

// Replaces complex SFINAE patterns
template<typename T>
std::string stringify(const T& value) {
    if constexpr (std::is_integral_v<T>) {
        return "int:" + std::to_string(value);
    } else if constexpr (std::is_floating_point_v<T>) {
        return "float:" + std::to_string(value);
    } else if constexpr (std::is_same_v<T, std::string>) {
        return "string:" + value;
    } else if constexpr (std::is_same_v<T, const char*>) {
        return "cstr:" + std::string(value);
    } else {
        // This branch only compiled if reached
        static_assert(sizeof(T) == 0, "Unsupported type");
    }
}

// Compile-time recursive tuple processing
template<typename Tuple, std::size_t I = 0>
void printTuple(const Tuple& t) {
    if constexpr (I < std::tuple_size_v<Tuple>) {
        if constexpr (I > 0) std::cout << ", ";
        std::cout << std::get<I>(t);
        printTuple<Tuple, I + 1>(t);
    }
}

// Heterogeneous container operations
template<typename T>
auto getSize(const T& container) {
    if constexpr (requires { container.size(); }) {
        return container.size();
    } else if constexpr (std::is_array_v<T>) {
        return sizeof(T) / sizeof(T[0]);
    } else {
        return 1;  // Scalar
    }
}

int main() {
    std::cout << stringify(42) << "\n";
    std::cout << stringify(3.14) << "\n";
    std::cout << stringify(std::string("hello")) << "\n";
    std::cout << stringify("world") << "\n";

    auto t = std::make_tuple(1, "hello", 3.14);
    printTuple(t);  // 1, hello, 3.14
    std::cout << "\n";

    return 0;
}
```

---

## 3. std::optional

Represents a value that may or may not be present. Replaces sentinel values and pointer-based patterns.

```cpp
#include <iostream>
#include <optional>
#include <string>
#include <vector>
#include <algorithm>

// Function that may not return a value
std::optional<int> divide(int a, int b) {
    if (b == 0) return std::nullopt;
    return a / b;
}

std::optional<std::string> findUser(int id) {
    if (id == 1) return "Alice";
    if (id == 2) return "Bob";
    return std::nullopt;
}

// Optional as class member
class Config {
    std::optional<int> port_;
    std::optional<std::string> host_;

public:
    void setPort(int p) { port_ = p; }
    void setHost(const std::string& h) { host_ = h; }

    int port() const { return port_.value_or(8080); }
    std::string host() const { return host_.value_or("localhost"); }
};

int main() {
    // Basic usage
    auto result = divide(10, 3);
    if (result) {
        std::cout << "Result: " << *result << "\n";  // 3
    }

    auto bad = divide(10, 0);
    std::cout << "has_value: " << bad.has_value() << "\n";  // false

    // value_or provides default
    std::cout << divide(10, 3).value_or(-1) << "\n";  // 3
    std::cout << divide(10, 0).value_or(-1) << "\n";  // -1

    // value() throws std::bad_optional_access if empty
    try {
        auto v = bad.value();
    } catch (const std::bad_optional_access& e) {
        std::cout << "No value: " << e.what() << "\n";
    }

    // Optional with strings
    auto user = findUser(1);
    std::cout << "User: " << user.value_or("Unknown") << "\n";

    // In-place construction
    std::optional<std::vector<int>> ov(std::in_place, {1, 2, 3});
    std::cout << "Size: " << ov->size() << "\n";

    // Config usage
    Config cfg;
    std::cout << cfg.host() << ":" << cfg.port() << "\n";  // localhost:8080
    cfg.setPort(3000);
    std::cout << cfg.host() << ":" << cfg.port() << "\n";  // localhost:3000

    return 0;
}
```

---

## 4. std::variant

A type-safe union that holds exactly one of its alternative types at any time.

```cpp
#include <iostream>
#include <variant>
#include <string>
#include <vector>

// Type-safe union
using Value = std::variant<int, double, std::string>;

// Visitor pattern using std::visit
struct ValuePrinter {
    void operator()(int i) const { std::cout << "int: " << i; }
    void operator()(double d) const { std::cout << "double: " << d; }
    void operator()(const std::string& s) const { std::cout << "string: " << s; }
};

// Overload pattern (C++17 idiom)
template<class... Ts> struct overloaded : Ts... { using Ts::operator()...; };
template<class... Ts> overloaded(Ts...) -> overloaded<Ts...>;

// JSON-like value type
using JsonValue = std::variant<
    std::nullptr_t, bool, int, double, std::string,
    std::vector<int>  // Simplified
>;

int main() {
    Value v = 42;
    std::cout << std::get<int>(v) << "\n";  // 42

    v = 3.14;
    std::cout << std::get<double>(v) << "\n";  // 3.14

    v = std::string("Hello");

    // Check current type
    if (std::holds_alternative<std::string>(v)) {
        std::cout << "It's a string: " << std::get<std::string>(v) << "\n";
    }

    // get_if returns pointer (nullptr if wrong type)
    if (auto* sp = std::get_if<std::string>(&v)) {
        std::cout << "String value: " << *sp << "\n";
    }

    // index() returns the zero-based type index
    std::cout << "Active index: " << v.index() << "\n";  // 2

    // std::visit with visitor struct
    Value values[] = {42, 3.14, std::string("Hello")};
    for (const auto& val : values) {
        std::visit(ValuePrinter{}, val);
        std::cout << "\n";
    }

    // std::visit with overloaded lambda pattern
    for (const auto& val : values) {
        std::visit(overloaded{
            [](int i) { std::cout << "int: " << i << "\n"; },
            [](double d) { std::cout << "double: " << d << "\n"; },
            [](const std::string& s) { std::cout << "str: " << s << "\n"; }
        }, val);
    }

    return 0;
}
```

---

## 5. std::any

A type-erased container that can hold any single value.

```cpp
#include <iostream>
#include <any>
#include <string>
#include <vector>

int main() {
    std::any a = 42;
    std::cout << std::any_cast<int>(a) << "\n";  // 42

    a = 3.14;
    std::cout << std::any_cast<double>(a) << "\n";

    a = std::string("Hello");
    std::cout << std::any_cast<std::string>(a) << "\n";

    // Type checking
    std::cout << "type: " << a.type().name() << "\n";
    std::cout << "has_value: " << a.has_value() << "\n";

    // Safe cast with pointer (returns nullptr on type mismatch)
    if (auto* p = std::any_cast<std::string>(&a)) {
        std::cout << "String: " << *p << "\n";
    }

    // Wrong type throws std::bad_any_cast
    try {
        auto val = std::any_cast<int>(a);  // a holds string!
    } catch (const std::bad_any_cast& e) {
        std::cout << "Bad cast: " << e.what() << "\n";
    }

    // Reset
    a.reset();
    std::cout << "has_value after reset: " << a.has_value() << "\n";

    // Practical use: heterogeneous container
    std::vector<std::any> config = {
        42,
        std::string("localhost"),
        true,
        3.14
    };

    return 0;
}
```

### When to Use optional vs variant vs any

| Type | Use When |
|------|----------|
| `std::optional<T>` | Value may be absent (nullable single type) |
| `std::variant<T, U, ...>` | Value is one of known types (type-safe union) |
| `std::any` | Value type completely unknown (type-erased, last resort) |

---

## 6. std::string_view

A non-owning reference to a string. Zero-copy, lightweight, and compatible with both `std::string` and `const char*`.

```cpp
#include <iostream>
#include <string>
#include <string_view>

// Accepts any string type without copying
void printView(std::string_view sv) {
    std::cout << "View: " << sv
              << " (length: " << sv.length() << ")\n";
}

// Efficient substring
std::string_view getExtension(std::string_view filename) {
    auto pos = filename.rfind('.');
    if (pos == std::string_view::npos) return "";
    return filename.substr(pos + 1);
}

// Parse tokens
void parseCSV(std::string_view line) {
    while (!line.empty()) {
        auto pos = line.find(',');
        auto token = line.substr(0, pos);
        std::cout << "[" << token << "] ";
        if (pos == std::string_view::npos) break;
        line.remove_prefix(pos + 1);
    }
    std::cout << "\n";
}

int main() {
    // Works with all string types
    std::string str = "Hello, World!";
    const char* cstr = "Hello from C!";

    printView(str);
    printView(cstr);
    printView("Literal string");

    // Substring (no copy!)
    std::string_view sv = "Hello, World!";
    auto sub = sv.substr(0, 5);
    std::cout << "Substring: " << sub << "\n";  // Hello

    // Extension parsing
    std::cout << "Extension: " << getExtension("main.cpp") << "\n";
    std::cout << "Extension: " << getExtension("archive.tar.gz") << "\n";

    // CSV parsing
    parseCSV("Alice,25,New York");

    // WARNING: dangling string_view
    // std::string_view bad;
    // {
    //     std::string temp = "temporary";
    //     bad = temp;  // bad points to temp's buffer
    // }
    // std::cout << bad;  // UNDEFINED BEHAVIOR: temp is destroyed

    return 0;
}
```

---

## 7. std::filesystem

Portable file system operations standardized from Boost.Filesystem.

```cpp
#include <iostream>
#include <filesystem>
#include <fstream>

namespace fs = std::filesystem;

int main() {
    // Path operations
    fs::path p = "/home/user/documents/file.txt";
    std::cout << "filename:   " << p.filename() << "\n";
    std::cout << "stem:       " << p.stem() << "\n";
    std::cout << "extension:  " << p.extension() << "\n";
    std::cout << "parent:     " << p.parent_path() << "\n";
    std::cout << "root:       " << p.root_path() << "\n";

    // Path concatenation
    fs::path dir = "/home/user";
    fs::path file = "document.txt";
    fs::path full = dir / file;
    std::cout << "Combined: " << full << "\n";

    // Current directory
    std::cout << "CWD: " << fs::current_path() << "\n";

    // Check existence and type
    fs::path testPath = ".";
    std::cout << "exists: " << fs::exists(testPath) << "\n";
    std::cout << "is_dir: " << fs::is_directory(testPath) << "\n";

    // Directory iteration
    std::cout << "\n=== Current directory ===\n";
    for (const auto& entry : fs::directory_iterator(".")) {
        std::cout << entry.path().filename();
        if (entry.is_directory()) {
            std::cout << " [DIR]";
        } else {
            std::cout << " [" << entry.file_size() << " bytes]";
        }
        std::cout << "\n";
    }

    // Recursive directory iteration
    // for (const auto& entry : fs::recursive_directory_iterator(".")) { ... }

    // File operations (with error code for expected failures)
    std::error_code ec;
    fs::create_directories("/tmp/test/subdir", ec);
    if (!ec) {
        std::cout << "Directories created\n";
    }

    // Copy, rename, remove
    // fs::copy("source.txt", "dest.txt", ec);
    // fs::rename("old.txt", "new.txt", ec);
    // fs::remove("file.txt", ec);
    // auto removed = fs::remove_all("/tmp/test", ec);  // Recursive

    // File size and last write time
    // auto size = fs::file_size("file.txt");
    // auto time = fs::last_write_time("file.txt");

    return 0;
}
```

---

## 8. Fold Expressions

C++17 fold expressions apply a binary operator across a parameter pack.

```cpp
#include <iostream>

// Sum all arguments
template<typename... Args>
auto sum(Args... args) {
    return (args + ...);
}

// Print with spaces
template<typename... Args>
void print(Args... args) {
    ((std::cout << args << " "), ...);
    std::cout << "\n";
}

// Check all conditions
template<typename... Args>
bool all(Args... args) {
    return (args && ...);
}

// Push all to vector
#include <vector>
template<typename T, typename... Args>
void pushAll(std::vector<T>& vec, Args&&... args) {
    (vec.push_back(std::forward<Args>(args)), ...);
}

int main() {
    std::cout << sum(1, 2, 3, 4, 5) << "\n";  // 15
    print(1, "hello", 3.14);  // 1 hello 3.14

    std::vector<int> v;
    pushAll(v, 1, 2, 3, 4, 5);
    for (int x : v) std::cout << x << " ";
    std::cout << "\n";

    return 0;
}
```

---

## 9. Class Template Argument Deduction (CTAD)

C++17 allows the compiler to deduce class template arguments from constructor arguments.

```cpp
#include <iostream>
#include <vector>
#include <tuple>
#include <mutex>
#include <optional>

int main() {
    // Before C++17: explicit template arguments
    std::pair<int, double> p1(1, 3.14);
    std::tuple<int, double, std::string> t1(1, 3.14, "hello");

    // C++17: CTAD deduces types
    std::pair p2(1, 3.14);                  // pair<int, double>
    std::tuple t2(1, 3.14, "hello");        // tuple<int, double, const char*>
    std::optional o(42);                    // optional<int>
    std::vector v{1, 2, 3, 4};             // vector<int>

    // Deduction guides for custom classes
    // template<typename T>
    // class MyContainer {
    //     T value;
    // public:
    //     MyContainer(T v) : value(v) {}
    // };
    // // Implicit deduction guide from constructor
    // MyContainer mc(42);  // MyContainer<int>

    // lock_guard CTAD
    std::mutex mtx;
    std::lock_guard lock(mtx);  // lock_guard<std::mutex>

    std::cout << p2.first << ", " << p2.second << "\n";

    return 0;
}
```

---

## 10. Other C++17 Features

### Inline Variables

```cpp
// header.h
// C++17: inline variables can be defined in headers
struct Config {
    static inline int maxRetries = 3;
    static inline std::string defaultHost = "localhost";
};

// Also works at namespace scope
inline constexpr int VERSION = 17;
```

### Nested Namespaces

```cpp
// Before C++17
namespace A { namespace B { namespace C {
    void func() {}
}}}

// C++17
namespace A::B::C {
    void func() {}
}
```

### [[nodiscard]]

```cpp
#include <iostream>

[[nodiscard]] int computeValue() {
    return 42;
}

[[nodiscard("Error codes must not be ignored")]]
int openFile(const char* path) {
    return 0;  // Success
}

class [[nodiscard]] ErrorCode {
    int code_;
public:
    ErrorCode(int c) : code_(c) {}
};

int main() {
    // computeValue();  // Warning: discarding return value
    int v = computeValue();  // OK

    // openFile("test.txt");  // Warning with custom message
    int err = openFile("test.txt");  // OK

    std::cout << v << "\n";
    return 0;
}
```

### [[maybe_unused]] and [[fallthrough]]

```cpp
#include <iostream>

void example([[maybe_unused]] int debugValue) {
    // No warning even if debugValue is unused in release builds
    #ifdef DEBUG
    std::cout << debugValue << "\n";
    #endif
}

void handleStatus(int status) {
    switch (status) {
        case 0:
            std::cout << "Success\n";
            break;
        case 1:
            std::cout << "Warning: ";
            [[fallthrough]];  // Intentional fall-through
        case 2:
            std::cout << "Continuing...\n";
            break;
    }
}
```

### if/switch with Initializer

```cpp
#include <iostream>
#include <map>

int main() {
    std::map<int, std::string> db = {{1, "Alice"}, {2, "Bob"}};

    // if with initializer
    if (auto it = db.find(1); it != db.end()) {
        std::cout << "Found: " << it->second << "\n";
    }
    // 'it' is not visible here

    // switch with initializer
    switch (auto val = 2 * 3; val) {
        case 6: std::cout << "Six\n"; break;
        default: std::cout << "Other: " << val << "\n";
    }

    return 0;
}
```

---

## Summary

| Feature | Category | Key Benefit |
|---------|----------|-------------|
| Structured bindings | Syntax | Clean decomposition |
| `if constexpr` | Templates | Readable compile-time branching |
| `std::optional` | Vocabulary | Nullable values |
| `std::variant` | Vocabulary | Type-safe union |
| `std::any` | Vocabulary | Type-erased container |
| `std::string_view` | Performance | Zero-copy string reference |
| `std::filesystem` | Library | Portable file operations |
| Fold expressions | Templates | Simplified pack operations |
| CTAD | Templates | Less template argument boilerplate |
| Inline variables | Linkage | Header-defined variables |
| `[[nodiscard]]` | Safety | Prevent ignored return values |

---

## Exercises

### Exercise 1: Config Parser

Use `std::variant`, `std::optional`, and `std::string_view` to implement a configuration parser that handles string, integer, float, and boolean values.

### Exercise 2: File Search

Using `std::filesystem`, write a function that recursively searches a directory for files matching a pattern (by extension or name substring).

### Exercise 3: Variant Calculator

Implement a calculator where operands are `std::variant<int, double>` and use `std::visit` to implement operations that produce the correct result type.

### Exercise 4: String Tokenizer

Write a tokenizer using `std::string_view` that splits a string by a delimiter without any memory allocation.

### Exercise 5: Type-Safe Settings

Create a `Settings` class where each setting is a `std::variant` of allowed types. Use `std::visit` with the overloaded pattern to serialize settings to a string format.

---

## Next Steps

C++20 introduced Concepts, a revolutionary approach to constraining templates. Let's explore them in [08_CPP20_Concepts.md](./08_CPP20_Concepts.md).
