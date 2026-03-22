# Move Semantics Deep Dive

**Previous**: [C++ Advanced](./00_Overview.md) | **Next**: [Templates](./02_Templates.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Distinguish lvalues, rvalues, xvalues, prvalues, and glvalues in expression contexts
2. Implement move constructors and move assignment operators correctly
3. Apply `std::move` and `std::forward` for efficient resource transfer
4. Design classes following the Rule of Five and Rule of Zero
5. Explain copy elision, RVO, and NRVO optimizations

---

Move semantics, introduced in C++11, fundamentally changed how C++ handles resource ownership. Before move semantics, returning a large object from a function meant copying every byte--even when the source was about to be destroyed. By distinguishing objects that can be safely "stolen from" (rvalues) from those that must be preserved (lvalues), the compiler and the programmer can collaborate to eliminate unnecessary copies. Mastering value categories and move operations is essential for writing high-performance modern C++.

## 1. Value Categories

Every C++ expression has two independent properties: a **type** and a **value category**. C++11 introduced a refined taxonomy of value categories.

```
          expression
          /        \
       glvalue    rvalue
       /    \     /    \
    lvalue  xvalue  prvalue
```

| Category | Has Identity? | Can Be Moved From? | Examples |
|----------|:---:|:---:|---------|
| **lvalue** | Yes | No | variables, string literals, `*ptr`, `arr[i]` |
| **xvalue** | Yes | Yes | `std::move(x)`, cast to `T&&` |
| **prvalue** | No | Yes | literals (`42`, `true`), `x + y`, function returning by value |
| **glvalue** | Yes | -- | lvalue or xvalue |
| **rvalue** | -- | Yes | xvalue or prvalue |

```cpp
#include <iostream>
#include <string>

int getValue() { return 42; }
int& getRef(int& x) { return x; }

int main() {
    int x = 10;

    // lvalue examples
    int& ref = x;            // x is an lvalue
    int* ptr = &x;           // Can take address of lvalue

    // prvalue examples
    // int& bad = 42;        // Error: cannot bind lvalue ref to prvalue
    const int& ok = 42;      // OK: const lvalue ref extends lifetime
    int&& rref = 42;         // OK: rvalue ref binds to prvalue

    // xvalue examples
    int&& moved = std::move(x);  // std::move(x) is an xvalue

    // Distinguishing categories
    // &getValue();           // Error: prvalue has no address
    &getRef(x);              // OK: lvalue has an address

    return 0;
}
```

---

## 2. Rvalue References

An rvalue reference (`T&&`) is a reference that binds to rvalues. It enables the compiler to distinguish between copies (from lvalues) and moves (from rvalues).

```cpp
#include <iostream>

void process(int& x)  { std::cout << "lvalue: " << x << "\n"; }
void process(int&& x) { std::cout << "rvalue: " << x << "\n"; }

int main() {
    int a = 10;

    process(a);             // lvalue overload
    process(20);            // rvalue overload
    process(std::move(a));  // rvalue overload (a is cast to xvalue)

    // Rvalue reference extends lifetime of temporary
    int&& rref = 42;       // Temporary lives as long as rref
    rref = 100;             // Can modify through rvalue reference
    std::cout << rref << "\n";  // 100

    // Important: a named rvalue reference is itself an lvalue!
    int&& r = 5;
    // process(r);          // Calls lvalue overload! r is a named variable
    process(std::move(r));  // Calls rvalue overload

    return 0;
}
```

### Binding Rules

| Reference Type | Binds to lvalue? | Binds to rvalue? |
|---------------|:---:|:---:|
| `T&` | Yes | No |
| `const T&` | Yes | Yes |
| `T&&` | No | Yes |
| `const T&&` | No | Yes (rarely used) |

---

## 3. Move Constructor

A move constructor transfers ownership of resources from a source object, leaving the source in a valid but unspecified state.

```cpp
#include <iostream>
#include <cstring>
#include <utility>

class String {
private:
    char* data_;
    size_t size_;

public:
    // Constructor
    String(const char* str = "") {
        size_ = std::strlen(str);
        data_ = new char[size_ + 1];
        std::strcpy(data_, str);
        std::cout << "Constructed: \"" << data_ << "\"\n";
    }

    // Copy constructor
    String(const String& other)
        : size_(other.size_), data_(new char[other.size_ + 1]) {
        std::strcpy(data_, other.data_);
        std::cout << "Copied: \"" << data_ << "\"\n";
    }

    // Move constructor
    String(String&& other) noexcept
        : data_(other.data_), size_(other.size_) {
        // Steal resources
        other.data_ = nullptr;
        other.size_ = 0;
        std::cout << "Moved: \"" << data_ << "\"\n";
    }

    // Destructor
    ~String() {
        std::cout << "Destroyed: \""
                  << (data_ ? data_ : "null") << "\"\n";
        delete[] data_;
    }

    const char* c_str() const { return data_ ? data_ : ""; }
    size_t size() const { return size_; }
};

int main() {
    String s1("Hello");
    String s2 = s1;              // Copy constructor
    String s3 = std::move(s1);   // Move constructor
    // s1 is now in a moved-from state (data_ == nullptr)

    std::cout << "s2: " << s2.c_str() << "\n";
    std::cout << "s3: " << s3.c_str() << "\n";
    std::cout << "s1: " << s1.c_str() << " (moved-from)\n";

    return 0;
}
```

### Why noexcept Matters

The `noexcept` specifier on move operations is critical. STL containers like `std::vector` will only use move constructors during reallocation if they are marked `noexcept`. Otherwise, the container falls back to copying for exception safety.

```cpp
#include <vector>
#include <iostream>

class Widget {
public:
    // Without noexcept, vector::push_back will COPY instead of move
    Widget(Widget&& other) noexcept { /* ... */ }
    Widget& operator=(Widget&& other) noexcept { /* ... */ }
};
```

---

## 4. Move Assignment

The move assignment operator transfers resources from one existing object to another.

```cpp
#include <iostream>
#include <algorithm>
#include <cstring>

class Buffer {
private:
    int* data_;
    size_t size_;

public:
    Buffer(size_t n) : data_(new int[n]()), size_(n) {}

    ~Buffer() { delete[] data_; }

    // Copy assignment
    Buffer& operator=(const Buffer& other) {
        if (this != &other) {
            delete[] data_;
            size_ = other.size_;
            data_ = new int[size_];
            std::copy(other.data_, other.data_ + size_, data_);
        }
        return *this;
    }

    // Move assignment
    Buffer& operator=(Buffer&& other) noexcept {
        if (this != &other) {
            delete[] data_;       // Release current resources
            data_ = other.data_;  // Steal resources
            size_ = other.size_;
            other.data_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    size_t size() const { return size_; }
};

int main() {
    Buffer b1(100);
    Buffer b2(50);

    b2 = std::move(b1);  // Move assignment
    std::cout << "b2 size: " << b2.size() << "\n";  // 100

    return 0;
}
```

### Moved-From State

After `std::move`, the source object is left in a **valid but unspecified** state. The C++ standard guarantees only that you can safely reassign or destroy it — but reading any value from it is undefined behavior for most types.

```cpp
#include <string>

int main() {
    std::string s = "hello";
    std::string t = std::move(s);

    // s is in a valid but unspecified state — do NOT read it
    // std::cout << s;  // undefined behavior: may print "", may crash

    s = "world";       // OK: reassigning a moved-from object is always safe
    std::cout << s;    // "world" -- fully usable again
}
```

### Self-Move

Assigning an object to itself via `std::move` — `x = std::move(x)` — is technically undefined behavior for standard library types. The standard requires only that self-move leaves the object in a "valid" (but unspecified) state; in practice it often silently corrupts data.

```cpp
// Defensive move assignment: guard against self-assignment
Buffer& operator=(Buffer&& other) noexcept {
    if (this != &other) {          // <-- critical check
        delete[] data_;
        data_ = other.data_;
        size_ = other.size_;
        other.data_ = nullptr;
        other.size_ = 0;
    }
    return *this;
}
// Alternatively, the copy-and-swap idiom handles self-move safely by design.
```

---

### Copy-and-Swap Idiom

An alternative approach that provides both copy and move assignment through a single function:

```cpp
class String {
    char* data_;
    size_t size_;

public:
    // ... constructors, destructor ...

    friend void swap(String& a, String& b) noexcept {
        using std::swap;
        swap(a.data_, b.data_);
        swap(a.size_, b.size_);
    }

    // Unified assignment: handles both copy and move
    // When called with lvalue: param is copy-constructed
    // When called with rvalue: param is move-constructed
    String& operator=(String other) noexcept {
        swap(*this, other);
        return *this;
    }
};
```

---

## 5. std::move

`std::move` does **not** move anything. It is simply an unconditional cast to an rvalue reference, signaling that the object may be moved from.

```cpp
#include <iostream>
#include <utility>
#include <string>
#include <vector>

int main() {
    std::string s = "Hello, World!";

    // std::move is just static_cast<std::string&&>(s)
    std::string&& rref = std::move(s);
    // Nothing has been moved yet! s is still intact.

    // The move happens when the rvalue reference is used
    // to initialize or assign to another object:
    std::string s2 = std::move(s);  // NOW the move happens
    std::cout << "s2: " << s2 << "\n";   // "Hello, World!"
    std::cout << "s: \"" << s << "\"\n";  // "" (moved-from)

    // Practical use: moving into containers
    std::vector<std::string> vec;
    std::string name = "Alice";
    vec.push_back(std::move(name));  // Move instead of copy
    // name is now empty

    // When NOT to use std::move:
    // 1. On const objects (results in a copy, not a move)
    const std::string cs = "constant";
    std::string s3 = std::move(cs);  // Copies! const prevents move

    // 2. On return values (prevents RVO)
    // return std::move(local);  // BAD: prevents copy elision

    return 0;
}
```

---

## 6. Perfect Forwarding

In template code, a **forwarding reference** (`T&&` where `T` is a deduced template parameter) can bind to both lvalues and rvalues. `std::forward` preserves the original value category.

```cpp
#include <iostream>
#include <utility>
#include <string>

void process(const std::string& s) {
    std::cout << "lvalue: " << s << "\n";
}

void process(std::string&& s) {
    std::cout << "rvalue: " << s << "\n";
}

// Without perfect forwarding: always calls lvalue overload
template<typename T>
void wrapperBad(T&& arg) {
    process(arg);  // arg is always an lvalue (it has a name)
}

// With perfect forwarding: preserves value category
template<typename T>
void wrapperGood(T&& arg) {
    process(std::forward<T>(arg));
}

// How reference collapsing works:
// T = std::string&   -> T&& = std::string& && = std::string&  (lvalue)
// T = std::string    -> T&& = std::string&&                    (rvalue)

// Factory function using perfect forwarding
template<typename T, typename... Args>
std::unique_ptr<T> make(Args&&... args) {
    return std::make_unique<T>(std::forward<Args>(args)...);
}

int main() {
    std::string s = "Hello";

    wrapperBad(s);              // lvalue (correct)
    wrapperBad(std::move(s));   // lvalue (WRONG! lost rvalue-ness)

    std::string s2 = "World";
    wrapperGood(s2);            // lvalue (correct)
    wrapperGood(std::move(s2)); // rvalue (correct!)

    return 0;
}
```

### Reference Collapsing Rules

| Template Parameter `T` | `T&&` Becomes | Forwarded As |
|------------------------|---------------|-------------|
| `X&` | `X& &&` = `X&` | lvalue |
| `X&&` | `X&& &&` = `X&&` | rvalue |
| `X` | `X&&` | rvalue |

---

## 7. Rule of Five and Rule of Zero

### Rule of Five

If a class manages resources and defines **any** of these five special member functions, it should define **all** of them:

```cpp
class ResourceOwner {
    int* data_;
    size_t size_;

public:
    // 1. Destructor
    ~ResourceOwner() {
        delete[] data_;
    }

    // 2. Copy constructor
    ResourceOwner(const ResourceOwner& other)
        : data_(new int[other.size_]), size_(other.size_) {
        std::copy(other.data_, other.data_ + size_, data_);
    }

    // 3. Copy assignment operator
    ResourceOwner& operator=(const ResourceOwner& other) {
        if (this != &other) {
            delete[] data_;
            size_ = other.size_;
            data_ = new int[size_];
            std::copy(other.data_, other.data_ + size_, data_);
        }
        return *this;
    }

    // 4. Move constructor
    ResourceOwner(ResourceOwner&& other) noexcept
        : data_(other.data_), size_(other.size_) {
        other.data_ = nullptr;
        other.size_ = 0;
    }

    // 5. Move assignment operator
    ResourceOwner& operator=(ResourceOwner&& other) noexcept {
        if (this != &other) {
            delete[] data_;
            data_ = other.data_;
            size_ = other.size_;
            other.data_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    // Constructor
    ResourceOwner(size_t n) : data_(new int[n]()), size_(n) {}
};
```

### Rule of Zero

If a class does **not** directly manage resources (using smart pointers or standard containers instead), it should define **none** of the five special members. The compiler-generated defaults will do the right thing.

```cpp
#include <memory>
#include <vector>
#include <string>

// Rule of Zero: no special members needed
class Employee {
    std::string name_;
    int id_;
    std::vector<std::string> projects_;
    std::unique_ptr<int[]> scores_;

public:
    Employee(std::string name, int id)
        : name_(std::move(name)), id_(id),
          scores_(std::make_unique<int[]>(10)) {}

    // Compiler generates:
    // - Destructor (unique_ptr handles cleanup)
    // - Move constructor and move assignment (unique_ptr is move-only)
    // - Copy is implicitly deleted (unique_ptr is non-copyable)
};
```

### When to Use Which

| Scenario | Rule |
|----------|------|
| Class owns raw resources (raw pointers, file handles) | Rule of Five |
| Class uses only smart pointers and STL containers | Rule of Zero |
| Class needs custom copy but default move is fine | `= default` for move ops |

---

## 8. Copy Elision

Copy elision is a compiler optimization that eliminates unnecessary copy/move operations.

### Return Value Optimization (RVO)

```cpp
#include <iostream>

class Heavy {
public:
    Heavy() { std::cout << "Constructed\n"; }
    Heavy(const Heavy&) { std::cout << "Copied\n"; }
    Heavy(Heavy&&) noexcept { std::cout << "Moved\n"; }
};

// Named Return Value Optimization (NRVO)
Heavy createNamed() {
    Heavy h;       // Constructed directly in caller's memory
    return h;      // NRVO: no copy or move
}

// Return Value Optimization (RVO) - guaranteed since C++17
Heavy createUnnamed() {
    return Heavy();  // Guaranteed: no copy or move in C++17
}

int main() {
    std::cout << "--- NRVO ---\n";
    Heavy h1 = createNamed();    // Typically: "Constructed" only

    std::cout << "--- RVO (guaranteed C++17) ---\n";
    Heavy h2 = createUnnamed();  // Always: "Constructed" only

    return 0;
}
```

### Guaranteed Copy Elision (C++17)

C++17 mandates copy elision for prvalue initialization. This means:

```cpp
// These are guaranteed to NOT invoke copy/move constructors in C++17:
Heavy h = Heavy();                 // Direct initialization
Heavy h = Heavy(Heavy(Heavy()));   // Even nested temporaries
auto h = createUnnamed();          // From prvalue return

// NRVO is NOT guaranteed (but usually applied):
auto h = createNamed();            // May or may not elide
```

### When Copy Elision Does NOT Apply

```cpp
Heavy selectOne(bool flag) {
    Heavy a, b;
    if (flag) return a;  // NRVO cannot apply (multiple return paths)
    return b;            // Compiler may use move instead
}

Heavy passThrough(Heavy h) {
    return h;  // Parameter: move, not elision
}
```

| Optimization | Guaranteed (C++17)? | Condition |
|-------------|:---:|---------|
| RVO (unnamed return) | Yes | Returning a prvalue |
| NRVO (named return) | No | Single local variable returned |
| Parameter passing | No | Function parameters |

---

## Exercises

### Exercise 1: Value Category Quiz

For each expression, identify its value category (lvalue, xvalue, or prvalue):
- `x` (a local variable)
- `std::move(x)`
- `42`
- `x + y`
- `*ptr`

### Exercise 2: Implement a Move-Aware String Class

Write a `MyString` class that manages a dynamically allocated character array. Implement all five special member functions following the Rule of Five. Verify that moves are used instead of copies when appropriate.

### Exercise 3: Perfect Forwarding Factory

Write a `create<T>(args...)` function template that uses perfect forwarding to construct any type `T` with arbitrary arguments. Test it with types that have different constructor signatures.

### Exercise 4: Benchmark Copy vs Move

Create a `LargeObject` class containing a `std::vector<int>` with 1 million elements. Measure the time difference between copying and moving 1000 such objects.

### Exercise 5: Detecting Elision

Write a program that demonstrates when RVO, NRVO, and move construction occur. Use a class that prints messages in all special member functions. Test with `-fno-elide-constructors` to see the difference.

---

## Next Steps

Templates are the foundation of generic programming in C++. Let's explore function templates, class templates, and specialization in [02_Templates.md](./02_Templates.md).
