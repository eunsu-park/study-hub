# Smart Pointers and RAII

**Previous**: [Template Metaprogramming](./03_Template_Metaprogramming.md) | **Next**: [Error Handling Patterns](./05_Error_Handling_Patterns.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Identify common manual memory management pitfalls (leaks, double free, dangling pointers)
2. Apply the RAII principle to tie resource lifetimes to object scope
3. Use `unique_ptr` for exclusive ownership and transfer ownership with `std::move`
4. Use `shared_ptr` for shared ownership and explain how reference counting works
5. Break circular references with `weak_ptr` and safely promote them using `lock()`
6. Choose the correct smart pointer type for a given ownership scenario
7. Pass smart pointers to and from functions following modern C++ best practices

---

Manual `new`/`delete` is the single largest source of bugs in traditional C++ code: memory leaks, double frees, and dangling pointers have caused countless production outages and security vulnerabilities. Smart pointers eliminate these entire classes of bugs by encoding ownership semantics directly in the type system. Once you internalize when to reach for `unique_ptr`, `shared_ptr`, or `weak_ptr`, you can write code that is both safer and easier to reason about than anything raw pointers allow.

## 1. Challenges of Memory Management

Manual memory management in C++ can cause several problems.

```cpp
#include <iostream>

// Memory leak example
void memoryLeak() {
    int* p = new int(42);
    // Forgot delete - memory leak!
}

// Double free example
void doubleFree() {
    int* p = new int(42);
    delete p;
    // delete p;  // Double free - undefined behavior!
}

// Dangling pointer example
int* danglingPointer() {
    int* p = new int(42);
    delete p;
    return p;  // Points to freed memory - dangerous!
}

// Memory leak on exception
void exceptionLeak() {
    int* p = new int(42);
    // throw std::runtime_error("Error!");  // delete won't execute
    delete p;
}
```

### Problem Summary

| Problem | Description |
|---------|-------------|
| Memory leak | Forgetting to call delete |
| Double free | Freeing the same memory twice |
| Dangling pointer | Accessing freed memory |
| Exception safety | Memory leak when exception occurs |

---

## 2. RAII (Resource Acquisition Is Initialization)

Resource Acquisition Is Initialization: Acquire resources at object creation, automatically release at destruction.

```cpp
#include <iostream>

// Class applying RAII principle
class IntPtr {
private:
    int* ptr;

public:
    // Acquire resource in constructor
    explicit IntPtr(int value) : ptr(new int(value)) {
        std::cout << "Memory allocated" << std::endl;
    }

    // Release resource in destructor
    ~IntPtr() {
        delete ptr;
        std::cout << "Memory freed" << std::endl;
    }

    int& operator*() { return *ptr; }
    int* get() { return ptr; }

    // Disable copy (simplified)
    IntPtr(const IntPtr&) = delete;
    IntPtr& operator=(const IntPtr&) = delete;
};

void useRAII() {
    IntPtr p(42);
    std::cout << "Value: " << *p << std::endl;
    // Memory automatically freed when function ends
}

int main() {
    std::cout << "=== RAII Start ===" << std::endl;
    useRAII();
    std::cout << "=== RAII End ===" << std::endl;
    return 0;
}
```

Output:
```
=== RAII Start ===
Memory allocated
Value: 42
Memory freed
=== RAII End ===
```

---

## 3. unique_ptr

A smart pointer with exclusive ownership. Only one `unique_ptr` can own an object.

### Basic Usage

```cpp
#include <iostream>
#include <memory>

class Resource {
public:
    Resource() { std::cout << "Resource created" << std::endl; }
    ~Resource() { std::cout << "Resource destroyed" << std::endl; }
    void use() { std::cout << "Resource used" << std::endl; }
};

int main() {
    // Create unique_ptr
    std::unique_ptr<Resource> p1(new Resource());
    p1->use();

    // Using make_unique (C++14, recommended)
    auto p2 = std::make_unique<Resource>();
    p2->use();

    // Basic type
    auto num = std::make_unique<int>(42);
    std::cout << "Value: " << *num << std::endl;

    // Array
    auto arr = std::make_unique<int[]>(5);
    for (int i = 0; i < 5; i++) {
        arr[i] = i * 10;
    }

    std::cout << "Array: ";
    for (int i = 0; i < 5; i++) {
        std::cout << arr[i] << " ";
    }
    std::cout << std::endl;

    return 0;  // All memory automatically freed
}
```

### Ownership Transfer (move)

```cpp
#include <iostream>
#include <memory>

void takeOwnership(std::unique_ptr<int> p) {
    std::cout << "Inside function: " << *p << std::endl;
}  // p is destroyed here

std::unique_ptr<int> createResource() {
    return std::make_unique<int>(100);
}

int main() {
    auto p1 = std::make_unique<int>(42);

    // Cannot copy
    // auto p2 = p1;  // Compile error!

    // Move is allowed
    auto p2 = std::move(p1);
    std::cout << "p2: " << *p2 << std::endl;

    // p1 is now nullptr
    if (p1 == nullptr) {
        std::cout << "p1 is empty" << std::endl;
    }

    // Pass to function (ownership transfer)
    auto p3 = std::make_unique<int>(200);
    takeOwnership(std::move(p3));
    // p3 is now nullptr

    // Return from function (ownership transfer)
    auto p4 = createResource();
    std::cout << "p4: " << *p4 << std::endl;

    return 0;
}
```

### unique_ptr Methods

```cpp
#include <iostream>
#include <memory>

int main() {
    auto p = std::make_unique<int>(42);

    // get(): Get raw pointer (ownership retained)
    int* raw = p.get();
    std::cout << "raw: " << *raw << std::endl;

    // release(): Give up ownership and return raw pointer
    int* released = p.release();
    if (p == nullptr) {
        std::cout << "p is empty" << std::endl;
    }
    delete released;  // Manual deletion needed

    // reset(): Release existing object and set new one
    auto p2 = std::make_unique<int>(100);
    std::cout << "Before reset: " << *p2 << std::endl;
    p2.reset(new int(200));
    std::cout << "After reset: " << *p2 << std::endl;
    p2.reset();  // Set to nullptr

    // swap(): Exchange two pointers
    auto a = std::make_unique<int>(1);
    auto b = std::make_unique<int>(2);
    a.swap(b);
    std::cout << "After swap: a=" << *a << ", b=" << *b << std::endl;

    return 0;
}
```

### Custom Deleter

```cpp
#include <iostream>
#include <memory>
#include <cstdio>

// Function deleter
void customDeleter(int* p) {
    std::cout << "Custom deleter called" << std::endl;
    delete p;
}

// FILE* management with lambda deleter
auto fileDeleter = [](FILE* f) {
    if (f) {
        std::cout << "Closing file" << std::endl;
        fclose(f);
    }
};

// C API wrapper pattern
auto make_file(const char* path, const char* mode) {
    return std::unique_ptr<FILE, decltype(fileDeleter)>(
        fopen(path, mode), fileDeleter
    );
}

int main() {
    // Lambda deleter
    auto deleter = [](int* p) {
        std::cout << "Lambda deleter" << std::endl;
        delete p;
    };
    std::unique_ptr<int, decltype(deleter)> p(new int(100), deleter);

    // shared_ptr has simpler syntax for custom deleters
    auto sp = std::shared_ptr<FILE>(
        fopen("/dev/null", "w"),
        [](FILE* f) { if (f) fclose(f); }
    );

    return 0;
}
```

---

## 4. shared_ptr

A smart pointer with shared ownership. Multiple `shared_ptr`s can share the same object.

> **Analogy -- The Shared Library Book**: A `shared_ptr` works like a library book checkout system. Multiple readers (owners) can check out the same book. A hidden counter tracks how many readers still have it. Only when the last reader returns the book (counter drops to zero) does the library free the memory.

### Basic Usage

```cpp
#include <iostream>
#include <memory>

class Resource {
public:
    Resource() { std::cout << "Resource created" << std::endl; }
    ~Resource() { std::cout << "Resource destroyed" << std::endl; }
};

int main() {
    std::shared_ptr<Resource> p1 = std::make_shared<Resource>();
    std::cout << "Reference count: " << p1.use_count() << std::endl;  // 1

    {
        std::shared_ptr<Resource> p2 = p1;
        std::cout << "Reference count: " << p1.use_count() << std::endl;  // 2

        std::shared_ptr<Resource> p3 = p1;
        std::cout << "Reference count: " << p1.use_count() << std::endl;  // 3
    }
    // p2, p3 destroyed
    std::cout << "Reference count: " << p1.use_count() << std::endl;  // 1

    return 0;  // Resource destroyed when reference count becomes 0
}
```

### Advantages of make_shared

```cpp
#include <iostream>
#include <memory>

class Widget {
public:
    int data[100];
};

int main() {
    // Method 1: Using new (2 memory allocations)
    std::shared_ptr<Widget> p1(new Widget());

    // Method 2: Using make_shared (1 memory allocation, recommended)
    auto p2 = std::make_shared<Widget>();

    /*
    Advantages of make_shared:
    1. Single memory allocation (object + control block)
    2. Exception safety
    3. Cleaner code
    */

    return 0;
}
```

### make_shared Control Block and weak_ptr Lifetime

`make_shared` performs a single allocation that holds both the managed object and the control block (reference count + weak count). This is efficient, but it has a subtle lifetime implication: the **entire allocation** stays alive until the last `weak_ptr` is destroyed, even after the object itself has been destroyed.

```cpp
auto sp = std::make_shared<Widget>();  // 1 allocation: [Widget | control block]
std::weak_ptr<Widget> wp = sp;

sp.reset();  // Widget is destroyed (strong ref count = 0)
             // BUT: the memory is NOT freed yet — weak_ptr keeps the
             // allocation alive to read the control block's weak count.
wp.reset();  // NOW the memory is freed (weak count = 0)
```

If `Widget` is large and you hold many `weak_ptr`s that outlive the object, prefer `std::shared_ptr<Widget>(new Widget())` so the object memory is freed independently of the control block.

### shared_ptr and Containers

```cpp
#include <iostream>
#include <memory>
#include <vector>

class Person {
public:
    std::string name;
    Person(const std::string& n) : name(n) {
        std::cout << name << " created" << std::endl;
    }
    ~Person() {
        std::cout << name << " destroyed" << std::endl;
    }
};

int main() {
    std::vector<std::shared_ptr<Person>> people;

    auto alice = std::make_shared<Person>("Alice");
    auto bob = std::make_shared<Person>("Bob");

    people.push_back(alice);
    people.push_back(bob);
    people.push_back(alice);  // Alice shared

    std::cout << "Alice reference count: " << alice.use_count() << std::endl;  // 3

    people.clear();
    std::cout << "Alice reference count: " << alice.use_count() << std::endl;  // 1

    return 0;
}
```

---

## 5. weak_ptr

Solves the circular reference problem of `shared_ptr`. Does not increment the reference count.

### Circular Reference Problem

```cpp
#include <iostream>
#include <memory>

class B;  // Forward declaration

class A {
public:
    std::shared_ptr<B> b_ptr;
    ~A() { std::cout << "A destroyed" << std::endl; }
};

class B {
public:
    std::shared_ptr<A> a_ptr;  // Circular reference!
    ~B() { std::cout << "B destroyed" << std::endl; }
};

int main() {
    {
        auto a = std::make_shared<A>();
        auto b = std::make_shared<B>();

        a->b_ptr = b;
        b->a_ptr = a;  // Circular reference occurs

        std::cout << "a ref count: " << a.use_count() << std::endl;  // 2
        std::cout << "b ref count: " << b.use_count() << std::endl;  // 2
    }
    // Memory leak! Neither A nor B is destroyed
    std::cout << "Block ended" << std::endl;

    return 0;
}
```

### Solution with weak_ptr

```cpp
#include <iostream>
#include <memory>

class B;

class A {
public:
    std::shared_ptr<B> b_ptr;
    ~A() { std::cout << "A destroyed" << std::endl; }
};

class B {
public:
    std::weak_ptr<A> a_ptr;  // Using weak_ptr!
    ~B() { std::cout << "B destroyed" << std::endl; }
};

int main() {
    {
        auto a = std::make_shared<A>();
        auto b = std::make_shared<B>();

        a->b_ptr = b;
        b->a_ptr = a;  // weak_ptr doesn't increment reference count

        std::cout << "a ref count: " << a.use_count() << std::endl;  // 1
        std::cout << "b ref count: " << b.use_count() << std::endl;  // 2
    }
    // Properly destroyed!
    std::cout << "Block ended" << std::endl;

    return 0;
}
```

### weak_ptr Usage

```cpp
#include <iostream>
#include <memory>

int main() {
    std::weak_ptr<int> weak;

    {
        auto shared = std::make_shared<int>(42);
        weak = shared;

        std::cout << "Inside block:" << std::endl;
        std::cout << "  expired: " << weak.expired() << std::endl;  // false
        std::cout << "  use_count: " << weak.use_count() << std::endl;  // 1

        // Accessing weak_ptr: Get shared_ptr with lock()
        if (auto sp = weak.lock()) {
            std::cout << "  Value: " << *sp << std::endl;
        }
    }
    // shared is destroyed

    std::cout << "Outside block:" << std::endl;
    std::cout << "  expired: " << weak.expired() << std::endl;  // true

    if (auto sp = weak.lock()) {
        std::cout << "  Value: " << *sp << std::endl;
    } else {
        std::cout << "  Object is destroyed" << std::endl;
    }

    return 0;
}
```

### Cache Implementation Example

```cpp
#include <iostream>
#include <memory>
#include <map>
#include <string>

class Image {
public:
    std::string filename;

    Image(const std::string& fn) : filename(fn) {
        std::cout << "Loading image: " << filename << std::endl;
    }
    ~Image() {
        std::cout << "Releasing image: " << filename << std::endl;
    }
};

class ImageCache {
private:
    std::map<std::string, std::weak_ptr<Image>> cache;

public:
    std::shared_ptr<Image> getImage(const std::string& filename) {
        auto it = cache.find(filename);

        if (it != cache.end()) {
            if (auto sp = it->second.lock()) {
                std::cout << "Cache hit: " << filename << std::endl;
                return sp;
            }
        }

        std::cout << "Cache miss: " << filename << std::endl;
        auto image = std::make_shared<Image>(filename);
        cache[filename] = image;
        return image;
    }
};

int main() {
    ImageCache cache;

    {
        auto img1 = cache.getImage("photo.jpg");
        auto img2 = cache.getImage("photo.jpg");  // Cache hit
        auto img3 = cache.getImage("icon.png");
    }
    // All images released

    auto img = cache.getImage("photo.jpg");  // Load again

    return 0;
}
```

---

## 6. enable_shared_from_this

Safely get a `shared_ptr` of yourself from within a class.

> **Pitfall — Never call `shared_from_this()` in a constructor.** When the constructor runs, no `shared_ptr` owns the object yet. Calling `shared_from_this()` at this point throws `std::bad_weak_ptr` (or causes undefined behavior with older implementations). Always call it from ordinary member functions after the object is managed by a `shared_ptr`.
>
> ```cpp
> class Bad : public std::enable_shared_from_this<Bad> {
> public:
>     Bad() {
>         auto self = shared_from_this();  // THROWS: no shared_ptr exists yet
>     }
> };
>
> // Correct pattern: use a factory function
> class Good : public std::enable_shared_from_this<Good> {
>     Good() = default;  // private constructor
> public:
>     static std::shared_ptr<Good> create() {
>         return std::shared_ptr<Good>(new Good());  // shared_ptr created first
>     }
>     std::shared_ptr<Good> getPtr() {
>         return shared_from_this();  // safe: called after creation
>     }
> };
> ```

```cpp
#include <iostream>
#include <memory>
#include <vector>

class Task : public std::enable_shared_from_this<Task> {
public:
    std::string name;

    Task(const std::string& n) : name(n) {}

    // Safely return shared_ptr to self
    std::shared_ptr<Task> getPtr() {
        return shared_from_this();
    }

    void addToQueue(std::vector<std::shared_ptr<Task>>& queue) {
        queue.push_back(shared_from_this());
    }
};

int main() {
    std::vector<std::shared_ptr<Task>> taskQueue;

    {
        auto task = std::make_shared<Task>("Task1");
        std::cout << "Ref count: " << task.use_count() << std::endl;  // 1

        task->addToQueue(taskQueue);
        std::cout << "Ref count: " << task.use_count() << std::endl;  // 2
    }
    // task variable destroyed, but remains in taskQueue

    for (const auto& t : taskQueue) {
        std::cout << t->name << std::endl;
    }

    return 0;
}
```

---

## 7. Smart Pointer Selection Guide

| Situation | Choice |
|-----------|--------|
| Single owner | `unique_ptr` |
| Multiple owners | `shared_ptr` |
| Prevent circular reference | `weak_ptr` |
| Cache, Observer | `weak_ptr` |
| Factory function return | `unique_ptr` |
| Container storage | `shared_ptr` or `unique_ptr` |

---

## 8. Smart Pointers and Functions

### Function Parameters

```cpp
#include <iostream>
#include <memory>

class Widget {
public:
    int value;
    Widget(int v) : value(v) {}
};

// Transfer ownership
void takeOwnership(std::unique_ptr<Widget> w) {
    std::cout << "Ownership received: " << w->value << std::endl;
}

// Share ownership
void shareOwnership(std::shared_ptr<Widget> w) {
    std::cout << "Shared: " << w->value
              << " (count: " << w.use_count() << ")" << std::endl;
}

// Use without ownership (preferred for non-owning access)
void useOnly(Widget& w) {
    std::cout << "Use only: " << w.value << std::endl;
}

// Use without ownership (nullable)
void useOnlyPtr(Widget* w) {
    if (w) {
        std::cout << "Pointer use: " << w->value << std::endl;
    }
}

int main() {
    auto up = std::make_unique<Widget>(1);
    useOnly(*up);
    useOnlyPtr(up.get());
    takeOwnership(std::move(up));

    auto sp = std::make_shared<Widget>(2);
    useOnly(*sp);
    shareOwnership(sp);

    return 0;
}
```

### Function Return

```cpp
#include <iostream>
#include <memory>

class Product {
public:
    std::string name;
    Product(const std::string& n) : name(n) {}
};

// Factory function: Return unique_ptr
std::unique_ptr<Product> createProduct(const std::string& name) {
    return std::make_unique<Product>(name);
}

// Cached object: Return shared_ptr
std::shared_ptr<Product> getCachedProduct() {
    static auto cached = std::make_shared<Product>("Cached");
    return cached;
}

int main() {
    auto p1 = createProduct("Widget");
    std::cout << p1->name << std::endl;

    auto p2 = getCachedProduct();
    auto p3 = getCachedProduct();
    std::cout << "Cache count: " << p2.use_count() << std::endl;  // 3

    return 0;
}
```

---

## 9. Pimpl Idiom with unique_ptr

The Pimpl (Pointer to Implementation) idiom hides implementation details behind a pointer, reducing compile-time dependencies.

**widget.h** (header):
```cpp
#ifndef WIDGET_H
#define WIDGET_H

#include <memory>
#include <string>

class Widget {
public:
    Widget(const std::string& name, int value);
    ~Widget();  // Must be declared here, defined in .cpp

    Widget(Widget&& other) noexcept;
    Widget& operator=(Widget&& other) noexcept;

    Widget(const Widget&) = delete;
    Widget& operator=(const Widget&) = delete;

    void doWork();
    std::string getName() const;

private:
    struct Impl;                      // Forward declaration only
    std::unique_ptr<Impl> pImpl_;     // The "compilation firewall"
};

#endif
```

**widget.cpp** (implementation):
```cpp
#include "widget.h"
#include <iostream>
#include <vector>

struct Widget::Impl {
    std::string name;
    int value;
    std::vector<int> history;

    Impl(const std::string& n, int v) : name(n), value(v) {}
};

Widget::Widget(const std::string& name, int value)
    : pImpl_(std::make_unique<Impl>(name, value)) {}

Widget::~Widget() = default;

Widget::Widget(Widget&& other) noexcept = default;
Widget& Widget::operator=(Widget&& other) noexcept = default;

void Widget::doWork() {
    pImpl_->history.push_back(pImpl_->value);
    std::cout << "Widget '" << pImpl_->name << "' doing work\n";
}

std::string Widget::getName() const {
    return pImpl_->name;
}
```

---

## 10. Performance Considerations

### unique_ptr vs shared_ptr

| | `unique_ptr` | `shared_ptr` |
|---|---|---|
| **Size** | Same as raw pointer | 2 pointers (object + control block) |
| **Overhead** | Zero | Ref counting (atomic operations) |
| **Allocation** | 1 (object only) | 1 with `make_shared`, 2 with `new` |
| **Thread safety** | None (single owner) | Ref count is atomic |

> **Memory Structure**
>
> unique_ptr: `ptr --> Object` (only one pointer)
>
> shared_ptr: `ptr --> Object`, `control --> Control Block` (ref count, weak count, deleter)

### Core Principles

1. **Avoid direct new/delete** - Use `make_unique`, `make_shared`
2. **Default to unique_ptr** - Only use shared_ptr when needed
3. **Beware of circular references** - Solve with weak_ptr
4. **Follow RAII principle** - Automate resource management
5. **Pass by reference for non-owning access** - Don't pass smart pointers unnecessarily

---

## Exercises

### Exercise 1: Resource Manager

Implement a class that manages various resources (file handles, network connections) using `unique_ptr` with custom deleters.

### Exercise 2: Graph Data Structure

Implement a graph where nodes are connected to each other using `shared_ptr` and `weak_ptr` to avoid circular references.

### Exercise 3: Object Pool

Implement a reusable object pool using smart pointers. Objects should be returned to the pool when no longer in use (hint: custom deleter on `shared_ptr`).

### Exercise 4: Pimpl Refactoring

Take a class with several heavy headers in its header file and refactor it to use the Pimpl idiom with `unique_ptr`.

### Exercise 5: Observer Pattern

Implement the Observer pattern where observers hold `weak_ptr` to the subject, and the subject holds `shared_ptr` to observer callbacks. Demonstrate that expired observers are automatically cleaned up.

---

## Next Steps

Error handling is a critical aspect of robust C++ programs. Let's explore exception safety guarantees, `noexcept`, and modern error handling patterns in [05_Error_Handling_Patterns.md](./05_Error_Handling_Patterns.md).
