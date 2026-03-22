# Error Handling Patterns

**Previous**: [Smart Pointers and RAII](./04_Smart_Pointers_and_RAII.md) | **Next**: [Modern C++ (C++11/14)](./06_Modern_CPP_11_14.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Apply the three levels of exception safety guarantees (basic, strong, nothrow)
2. Use `noexcept` specifications to enable compiler optimizations
3. Design error handling strategies using error codes, exceptions, and `std::expected`
4. Implement the scope guard pattern for exception-safe cleanup
5. Handle errors in constructors, destructors, and move operations

---

Error handling is one of the most consequential design decisions in any C++ codebase. A poorly chosen strategy leads to resource leaks, silent failures, or unreadable code littered with error checks. C++ offers multiple mechanisms--exceptions, error codes, and the newer `std::expected`--each with distinct trade-offs. This lesson establishes a framework for choosing among them and shows how to write code that remains correct even when things go wrong.

## 1. Exception Safety Guarantees

The C++ community defines three levels of exception safety. Every function you write should provide at least the basic guarantee.

| Guarantee | Promise | Example |
|-----------|---------|---------|
| **Nothrow** | Operation never throws | `std::swap`, destructors, move ops |
| **Strong** | If exception thrown, state is unchanged (commit-or-rollback) | `std::vector::push_back` |
| **Basic** | If exception thrown, invariants preserved, no leaks | Most standard library operations |

```cpp
#include <iostream>
#include <vector>
#include <stdexcept>

class Account {
    double balance_;
    std::string owner_;

public:
    Account(std::string owner, double balance)
        : balance_(balance), owner_(std::move(owner)) {}

    // Nothrow guarantee
    double balance() const noexcept { return balance_; }

    // Strong guarantee: either transfer succeeds or nothing changes
    void transfer(Account& to, double amount) {
        if (amount > balance_) {
            throw std::runtime_error("Insufficient funds");
        }
        // Both operations are noexcept (double subtraction/addition)
        balance_ -= amount;
        to.balance_ += amount;
    }

    // Basic guarantee: object remains valid but state may change
    void addTransaction(std::vector<std::string>& log, double amount) {
        balance_ += amount;  // noexcept
        log.push_back(owner_ + ": " + std::to_string(amount));
        // If push_back throws, balance_ already changed
        // Object is still valid, but state is partially updated
    }
};

int main() {
    Account alice("Alice", 1000.0);
    Account bob("Bob", 500.0);

    try {
        alice.transfer(bob, 2000.0);  // Throws: insufficient funds
    } catch (const std::runtime_error& e) {
        std::cout << "Error: " << e.what() << "\n";
        // Strong guarantee: balances unchanged
        std::cout << "Alice: " << alice.balance() << "\n";  // 1000
        std::cout << "Bob: " << bob.balance() << "\n";      // 500
    }

    return 0;
}
```

---

## 2. noexcept

The `noexcept` specifier declares that a function will not throw exceptions. This enables important compiler optimizations and is required for certain STL operations.

### Basic noexcept

```cpp
#include <iostream>
#include <vector>
#include <type_traits>

class Widget {
    int* data_;
    size_t size_;

public:
    Widget(size_t n) : data_(new int[n]()), size_(n) {}
    ~Widget() noexcept { delete[] data_; }

    // Move operations MUST be noexcept for vector reallocation optimization
    Widget(Widget&& other) noexcept
        : data_(other.data_), size_(other.size_) {
        other.data_ = nullptr;
        other.size_ = 0;
    }

    Widget& operator=(Widget&& other) noexcept {
        if (this != &other) {
            delete[] data_;
            data_ = other.data_;
            size_ = other.size_;
            other.data_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    // Copy (may throw due to allocation)
    Widget(const Widget& other)
        : data_(new int[other.size_]), size_(other.size_) {
        std::copy(other.data_, other.data_ + size_, data_);
    }
};

int main() {
    // noexcept operator: query at compile time
    std::cout << std::boolalpha;
    std::cout << "Widget move is noexcept: "
              << std::is_nothrow_move_constructible_v<Widget> << "\n";  // true
    std::cout << "Widget copy is noexcept: "
              << std::is_nothrow_copy_constructible_v<Widget> << "\n";  // false

    // vector uses move when noexcept, copy otherwise
    std::vector<Widget> vec;
    vec.reserve(1);
    vec.emplace_back(100);
    vec.emplace_back(200);  // Reallocation: uses move (noexcept)

    return 0;
}
```

### Conditional noexcept

```cpp
#include <type_traits>

// noexcept depends on whether the contained operation can throw
template<typename T>
void swapValues(T& a, T& b)
    noexcept(std::is_nothrow_move_constructible_v<T> &&
             std::is_nothrow_move_assignable_v<T>) {
    T temp = std::move(a);
    a = std::move(b);
    b = std::move(temp);
}

// Propagate noexcept from called function
template<typename F, typename... Args>
decltype(auto) callNoexcept(F&& f, Args&&... args)
    noexcept(noexcept(f(std::forward<Args>(args)...))) {
    return f(std::forward<Args>(args)...);
}
```

---

## 3. Exception-Safe Code

### Strong Guarantee via Copy-and-Swap

```cpp
#include <iostream>
#include <algorithm>

class StrongSafe {
    int* data_;
    size_t size_;

public:
    StrongSafe(size_t n) : data_(new int[n]()), size_(n) {}
    ~StrongSafe() { delete[] data_; }

    friend void swap(StrongSafe& a, StrongSafe& b) noexcept {
        using std::swap;
        swap(a.data_, b.data_);
        swap(a.size_, b.size_);
    }

    // Copy constructor (may throw)
    StrongSafe(const StrongSafe& other)
        : data_(new int[other.size_]), size_(other.size_) {
        std::copy(other.data_, other.data_ + size_, data_);
    }

    // Strong guarantee: if copy-construction of temp fails,
    // *this is completely unchanged
    StrongSafe& operator=(StrongSafe other) noexcept {
        swap(*this, other);
        return *this;
    }

    // Move constructor (noexcept)
    StrongSafe(StrongSafe&& other) noexcept
        : data_(other.data_), size_(other.size_) {
        other.data_ = nullptr;
        other.size_ = 0;
    }

    size_t size() const noexcept { return size_; }
};
```

### RAII + Exceptions

```cpp
#include <iostream>
#include <memory>
#include <fstream>
#include <stdexcept>

// Bad: manual cleanup with exceptions
void badExample() {
    int* data = new int[1000];
    // If readFile throws, data is leaked!
    // readFile(data);
    delete[] data;
}

// Good: RAII handles cleanup automatically
void goodExample() {
    auto data = std::make_unique<int[]>(1000);
    // If readFile throws, unique_ptr destructor frees data
    // readFile(data.get());
}

// Multiple resources: all protected by RAII
void multiResource() {
    auto file = std::fstream("data.txt", std::ios::out);
    auto buffer = std::make_unique<char[]>(4096);
    auto connection = std::make_unique<int>(42);  // simulated
    // If any operation throws, ALL resources are cleaned up
    // in reverse order of construction
}
```

---

## 4. Error Codes vs Exceptions

### When to Use Each

| Criterion | Error Codes | Exceptions |
|-----------|-------------|------------|
| Expected failures | Preferred | Overkill |
| Unexpected failures | Easily ignored | Preferred |
| Performance-critical hot path | Preferred | Overhead on throw |
| Constructors | Cannot return code | Preferred |
| Deep call chains | Tedious propagation | Automatic propagation |

### std::error_code and std::system_error

```cpp
#include <iostream>
#include <system_error>
#include <fstream>
#include <cerrno>
#include <cstring>

// Using error_code for expected failures
std::error_code openFile(const std::string& path, std::fstream& out) {
    out.open(path, std::ios::in);
    if (!out.is_open()) {
        return std::make_error_code(std::errc::no_such_file_or_directory);
    }
    return {};  // No error
}

// Using exceptions for unexpected failures
void processFile(const std::string& path) {
    std::fstream file;
    if (auto ec = openFile(path, file)) {
        // Convert to exception when failure is not expected
        throw std::system_error(ec, "Cannot process " + path);
    }
    // Process file...
}

int main() {
    // Error code: caller decides how to handle
    std::fstream file;
    if (auto ec = openFile("missing.txt", file)) {
        std::cout << "Error: " << ec.message() << "\n";
        // Can check category and code
        if (ec == std::errc::no_such_file_or_directory) {
            std::cout << "File not found, using defaults\n";
        }
    }

    // Exception: propagates automatically
    try {
        processFile("missing.txt");
    } catch (const std::system_error& e) {
        std::cout << "System error: " << e.what() << "\n";
        std::cout << "Code: " << e.code() << "\n";
    }

    return 0;
}
```

---

## 5. std::expected (C++23)

`std::expected<T, E>` provides monadic error handling: a value that is either the expected result or an error. It combines the explicitness of error codes with the composability of exceptions.

```cpp
#include <iostream>
#include <string>
#include <cmath>

// Simulating std::expected for pre-C++23
// In C++23, use #include <expected>
#if __cplusplus >= 202302L
#include <expected>
using std::expected;
using std::unexpected;
#else
// Simplified polyfill for demonstration
template<typename T, typename E>
class expected {
    bool has_val_;
    union { T val_; E err_; };
public:
    expected(T val) : has_val_(true), val_(std::move(val)) {}
    expected(E err, bool) : has_val_(false), err_(std::move(err)) {}
    bool has_value() const { return has_val_; }
    T& value() { return val_; }
    E& error() { return err_; }
    T value_or(T default_val) { return has_val_ ? val_ : default_val; }
    explicit operator bool() const { return has_val_; }
    T& operator*() { return val_; }
    ~expected() { if (has_val_) val_.~T(); else err_.~E(); }
};

template<typename E>
auto make_unexpected(E e) { return expected<int, E>(std::move(e), false); }
#endif

// Error type
enum class MathError {
    DivisionByZero,
    NegativeSqrt,
    Overflow
};

std::string to_string(MathError e) {
    switch (e) {
        case MathError::DivisionByZero: return "division by zero";
        case MathError::NegativeSqrt: return "negative sqrt";
        case MathError::Overflow: return "overflow";
    }
    return "unknown";
}

// Functions returning expected
expected<double, MathError> safeDivide(double a, double b) {
    if (b == 0.0) return expected<double, MathError>(MathError::DivisionByZero, false);
    return a / b;
}

expected<double, MathError> safeSqrt(double x) {
    if (x < 0.0) return expected<double, MathError>(MathError::NegativeSqrt, false);
    return std::sqrt(x);
}

int main() {
    auto result = safeDivide(10.0, 3.0);
    if (result) {
        std::cout << "10 / 3 = " << *result << "\n";
    }

    auto bad = safeDivide(10.0, 0.0);
    if (!bad) {
        std::cout << "Error: " << to_string(bad.error()) << "\n";
    }

    // value_or provides a default
    std::cout << "Result: " << safeDivide(10.0, 0.0).value_or(-1.0) << "\n";

    return 0;
}
```

### Monadic Operations (C++23)

```cpp
// C++23 std::expected supports monadic chaining:
// auto result = getData(id)
//     .and_then(validate)      // Chain if value
//     .transform(serialize)    // Map the value
//     .or_else(handleError);   // Handle error

// Example (C++23):
// std::expected<double, MathError> compute(double x) {
//     return safeSqrt(x)
//         .and_then([](double v) { return safeDivide(1.0, v); })
//         .transform([](double v) { return v * 100; });
// }
```

---

## 6. Scope Guards

Scope guards execute cleanup code when a scope exits, providing exception-safe cleanup without RAII wrapper classes.

```cpp
#include <iostream>
#include <functional>
#include <exception>

// Simple scope guard
class ScopeGuard {
    std::function<void()> cleanup_;
    bool dismissed_ = false;

public:
    explicit ScopeGuard(std::function<void()> cleanup)
        : cleanup_(std::move(cleanup)) {}

    ~ScopeGuard() {
        if (!dismissed_ && cleanup_) {
            cleanup_();
        }
    }

    void dismiss() { dismissed_ = true; }

    ScopeGuard(const ScopeGuard&) = delete;
    ScopeGuard& operator=(const ScopeGuard&) = delete;
};

// Convenience macro
#define CONCAT_IMPL(a, b) a##b
#define CONCAT(a, b) CONCAT_IMPL(a, b)
#define SCOPE_EXIT auto CONCAT(scope_guard_, __LINE__) = ScopeGuard

// Usage
void processTransaction() {
    std::cout << "Begin transaction\n";

    ScopeGuard rollback([&]() {
        std::cout << "Rolling back transaction\n";
    });

    // Do work that might throw...
    std::cout << "Doing work...\n";

    // If we get here, commit succeeded
    rollback.dismiss();
    std::cout << "Transaction committed\n";
}

void fileOperation() {
    FILE* f = fopen("/tmp/test.txt", "w");
    if (!f) return;

    // Ensure file is closed on any exit
    ScopeGuard closeFile([&]() {
        std::cout << "Closing file\n";
        fclose(f);
    });

    fprintf(f, "Hello\n");
    // If any exception occurs, file is still closed
}

int main() {
    processTransaction();
    std::cout << "---\n";
    fileOperation();

    return 0;
}
```

---

## 7. Error Handling in Special Members

### Constructors

Constructors are the only reliable way to signal construction failure in C++.

```cpp
#include <iostream>
#include <memory>
#include <stdexcept>

class Connection {
    int fd_;

public:
    // Constructors CAN and SHOULD throw on failure
    Connection(const std::string& host, int port) {
        fd_ = -1;  // Simulated connection
        if (host.empty()) {
            throw std::invalid_argument("Empty host");
        }
        // If this throws, no destructor runs (object never fully constructed)
        // But member destructors DO run for already-constructed members
        fd_ = 42;  // Simulated successful connection
        std::cout << "Connected to " << host << ":" << port << "\n";
    }

    ~Connection() {
        if (fd_ >= 0) {
            std::cout << "Disconnecting\n";
            // close(fd_);
        }
    }
};

// Multi-resource constructor: use smart pointers
class Server {
    std::unique_ptr<Connection> db_;
    std::unique_ptr<Connection> cache_;

public:
    Server() {
        db_ = std::make_unique<Connection>("db.local", 5432);
        // If this throws, db_ is automatically cleaned up
        cache_ = std::make_unique<Connection>("cache.local", 6379);
    }
};
```

### Destructors

Destructors should **never** throw. If they do, and another exception is already in flight, `std::terminate` is called.

```cpp
class SafeCleanup {
public:
    ~SafeCleanup() noexcept {
        try {
            // risky cleanup operation
        } catch (...) {
            // Log but do NOT rethrow
            // std::cerr << "Cleanup failed\n";
        }
    }
};
```

### Move Operations

Move operations should be `noexcept` whenever possible. A throwing move defeats the purpose of move semantics in many STL contexts.

```cpp
class Buffer {
    int* data_;
    size_t size_;

public:
    // Move: noexcept, just pointer swap
    Buffer(Buffer&& other) noexcept
        : data_(other.data_), size_(other.size_) {
        other.data_ = nullptr;
        other.size_ = 0;
    }

    // If move MUST do something that could throw,
    // document it clearly and accept the consequences
};
```

---

## 8. Best Practices

### Error Handling Decision Matrix

```
Is the error expected and common?
  ├── YES → Use error codes or std::expected
  └── NO → Is it a programming error (bug)?
        ├── YES → Use assert() / std::terminate
        └── NO → Use exceptions
```

### Summary Table

| Pattern | Use When | Example |
|---------|----------|---------|
| RAII | Always for resource management | Smart pointers, lock_guard |
| Exceptions | Unexpected errors, constructors | File not found, out of memory |
| Error codes | Expected failures, hot paths | Network timeout, parse failure |
| `std::expected` | Composable error handling | Data pipeline steps |
| `noexcept` | Move ops, destructors, swap | Move constructor |
| Scope guard | Ad-hoc cleanup without RAII class | Transaction rollback |
| `assert` | Programming errors (debug only) | Precondition violations |

### Guidelines

1. **Use RAII everywhere** -- Every resource should be owned by an RAII object
2. **Mark move operations noexcept** -- Enables STL optimizations
3. **Never throw from destructors** -- Can cause `std::terminate`
4. **Prefer `std::expected` for composable errors** (C++23)
5. **Use exceptions for exceptional conditions**, not control flow
6. **Document the exception safety guarantee** of every function

---

## Exercises

### Exercise 1: Exception-Safe Stack

Implement a stack that provides the strong exception safety guarantee for `push`. If the internal allocation fails, the stack must remain unchanged.

### Exercise 2: Scope Guard Implementation

Implement `ScopeExit`, `ScopeSuccess` (runs only on normal exit), and `ScopeFail` (runs only on exception) using `std::uncaught_exceptions()`.

### Exercise 3: Expected Pipeline

Using `std::expected` (or a polyfill), implement a data processing pipeline: `readFile -> parseCSV -> validate -> transform`. Each step can fail with a descriptive error.

### Exercise 4: Transaction Class

Write a `Transaction` class that collects a series of operations. On `commit()`, all operations execute. If any throws, all previously executed operations are rolled back.

### Exercise 5: noexcept Audit

Take an existing class with move operations and analyze whether they can be marked `noexcept`. Fix any operations that unnecessarily allocate or throw during moves.

---

## Next Steps

Modern C++11 and C++14 introduced a wealth of features that changed how C++ is written daily. Let's explore `auto`, lambdas, `constexpr`, and more in [06_Modern_CPP_11_14.md](./06_Modern_CPP_11_14.md).
