# Smart Pointers and Memory Management

**Previous**: [Exception Handling and File I/O](./13_Exceptions_and_File_IO.md) | **Next**: [Modern C++ (C++11/14/17/20)](./15_Modern_CPP.md)

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
    if (!p2) {
        std::cout << "p2 is empty" << std::endl;
    }

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

// Deleter for FILE*
auto fileDeleter = [](FILE* f) {
    if (f) {
        std::cout << "Closing file" << std::endl;
        fclose(f);
    }
};

int main() {
    // Function pointer deleter
    std::unique_ptr<int, void(*)(int*)> p1(
        new int(42), customDeleter
    );

    // Lambda deleter
    auto deleter = [](int* p) {
        std::cout << "Lambda deleter" << std::endl;
        delete p;
    };
    std::unique_ptr<int, decltype(deleter)> p2(
        new int(100), deleter
    );

    // FILE management
    std::unique_ptr<FILE, decltype(fileDeleter)> file(
        fopen("test.txt", "w"), fileDeleter
    );
    if (file) {
        fprintf(file.get(), "Hello, World!\n");
    }

    return 0;
}
```

---

## 4. shared_ptr

A smart pointer with shared ownership. Multiple `shared_ptr`s can share the same object.

> **Analogy -- The Shared Library Book**: A `shared_ptr` works like a library book checkout system. Multiple readers (owners) can check out the same book. A hidden counter tracks how many readers still have it. Only when the last reader returns the book (counter drops to zero) does the library put it back on the shelf (free the memory). A `weak_ptr` is like peeking at the catalog to see if the book still exists without actually checking it out.

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
    // Create shared_ptr
    std::shared_ptr<Resource> p1 = std::make_shared<Resource>();
    std::cout << "Reference count: " << p1.use_count() << std::endl;  // 1

    {
        // Share
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

    std::cout << "p1 use_count: " << p1.use_count() << std::endl;
    std::cout << "p2 use_count: " << p2.use_count() << std::endl;

    return 0;
}
```

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

    std::cout << "\n=== List ===" << std::endl;
    for (const auto& p : people) {
        std::cout << p->name << std::endl;
    }

    people.clear();
    std::cout << "\n=== After clear ===" << std::endl;
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

        std::cout << "a reference count: " << a.use_count() << std::endl;  // 2
        std::cout << "b reference count: " << b.use_count() << std::endl;  // 2
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

        std::cout << "a reference count: " << a.use_count() << std::endl;  // 1
        std::cout << "b reference count: " << b.use_count() << std::endl;  // 2
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
    std::cout << "  use_count: " << weak.use_count() << std::endl;  // 0

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
            // If in cache, try to get shared_ptr from weak_ptr
            if (auto sp = it->second.lock()) {
                std::cout << "Cache hit: " << filename << std::endl;
                return sp;
            }
        }

        // Cache miss: Load new
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

        std::cout << "img1 use_count: " << img1.use_count() << std::endl;
    }
    // All images released

    std::cout << "\n=== Request again ===" << std::endl;
    auto img = cache.getImage("photo.jpg");  // Load again

    return 0;
}
```

---

## 6. enable_shared_from_this

Safely get a `shared_ptr` of yourself from within a class.

```cpp
#include <iostream>
#include <memory>
#include <vector>

class Task : public std::enable_shared_from_this<Task> {
public:
    std::string name;

    Task(const std::string& n) : name(n) {
        std::cout << name << " created" << std::endl;
    }

    ~Task() {
        std::cout << name << " destroyed" << std::endl;
    }

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
        std::cout << "Reference count: " << task.use_count() << std::endl;  // 1

        task->addToQueue(taskQueue);
        std::cout << "Reference count: " << task.use_count() << std::endl;  // 2
    }
    // task variable destroyed, but remains in taskQueue

    std::cout << "\n=== Queue contents ===" << std::endl;
    for (const auto& t : taskQueue) {
        std::cout << t->name << std::endl;
    }

    return 0;
}
```

Caution:
```cpp
// Wrong usage - must be managed by shared_ptr
// Task t("Direct");
// t.getPtr();  // Runtime error!
```

---

## 7. Smart Pointer Selection Guide

```
┌─────────────────────────────────────────────────────┐
│              Smart Pointer Selection                │
└─────────────────────────────────────────────────────┘
                        │
          ┌─────────────┼─────────────┐
          ▼             ▼             ▼
    Exclusive        Shared         Weak
    Ownership?      Needed?       Reference?
          │             │             │
          ▼             ▼             ▼
    unique_ptr    shared_ptr     weak_ptr
```

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

// Transfer ownership (unique_ptr)
void takeOwnership(std::unique_ptr<Widget> w) {
    std::cout << "Ownership received: " << w->value << std::endl;
}

// Share ownership (shared_ptr copy)
void shareOwnership(std::shared_ptr<Widget> w) {
    std::cout << "Shared: " << w->value
              << " (count: " << w.use_count() << ")" << std::endl;
}

// Use without ownership (reference)
void useOnly(Widget& w) {
    std::cout << "Use only: " << w.value << std::endl;
}

// Use without ownership (raw pointer)
void useOnlyPtr(Widget* w) {
    if (w) {
        std::cout << "Pointer use: " << w->value << std::endl;
    }
}

int main() {
    // unique_ptr
    auto up = std::make_unique<Widget>(1);
    useOnly(*up);
    useOnlyPtr(up.get());
    takeOwnership(std::move(up));  // Transfer ownership

    // shared_ptr
    auto sp = std::make_shared<Widget>(2);
    useOnly(*sp);
    useOnlyPtr(sp.get());
    shareOwnership(sp);  // Share
    std::cout << "Original count: " << sp.use_count() << std::endl;

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

## 9. Common Mistakes and Solutions

### Mistake 1: Creating Multiple Smart Pointers from Same Raw Pointer

```cpp
#include <iostream>
#include <memory>

int main() {
    int* raw = new int(42);

    // Wrong code - never do this!
    // std::shared_ptr<int> p1(raw);
    // std::shared_ptr<int> p2(raw);  // Double free!

    // Correct code
    auto p1 = std::make_shared<int>(42);
    auto p2 = p1;  // Share

    return 0;
}
```

### Mistake 2: Converting this to shared_ptr

```cpp
#include <iostream>
#include <memory>

class Bad {
public:
    // Wrong method
    std::shared_ptr<Bad> getShared() {
        // return std::shared_ptr<Bad>(this);  // Dangerous!
        return nullptr;
    }
};

class Good : public std::enable_shared_from_this<Good> {
public:
    // Correct method
    std::shared_ptr<Good> getShared() {
        return shared_from_this();
    }
};
```

### Mistake 3: Circular Reference

```cpp
// See weak_ptr section above
// Using only shared_ptr causes memory leak due to circular reference
// Change one connection to weak reference using weak_ptr
```

### Mistake 4: Attempting to Copy unique_ptr

```cpp
#include <memory>

void processWidget(std::unique_ptr<int> p) {}

int main() {
    auto p = std::make_unique<int>(42);

    // Wrong code
    // processWidget(p);  // Compile error

    // Correct code (ownership transfer)
    processWidget(std::move(p));

    return 0;
}
```

---

## 10. Performance Considerations

### unique_ptr vs shared_ptr

```cpp
#include <iostream>
#include <memory>
#include <chrono>
#include <vector>

int main() {
    const int N = 1000000;

    // unique_ptr (almost no overhead)
    auto start1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; i++) {
        auto p = std::make_unique<int>(i);
    }
    auto end1 = std::chrono::high_resolution_clock::now();

    // shared_ptr (reference counting overhead)
    auto start2 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; i++) {
        auto p = std::make_shared<int>(i);
    }
    auto end2 = std::chrono::high_resolution_clock::now();

    auto dur1 = std::chrono::duration_cast<std::chrono::microseconds>(end1 - start1);
    auto dur2 = std::chrono::duration_cast<std::chrono::microseconds>(end2 - start2);

    std::cout << "unique_ptr: " << dur1.count() << " us" << std::endl;
    std::cout << "shared_ptr: " << dur2.count() << " us" << std::endl;

    return 0;
}
```

### Memory Structure

```
unique_ptr:
┌─────────────────┐
│  ptr → Object   │  (only one pointer)
└─────────────────┘

shared_ptr:
┌─────────────────┐     ┌─────────────────┐
│  ptr ─────────────┬──▶│     Object      │
│  control ───┐    │    └─────────────────┘
└─────────────│───┘
              ▼
        ┌─────────────────┐
        │  Reference count│
        │  Weak count     │
        │  Deleter        │
        └─────────────────┘
```

---

## 11. Summary

| Smart Pointer | Ownership | Copy | Ref Count | Use Case |
|---------------|-----------|------|-----------|----------|
| `unique_ptr` | Exclusive | X | X | Single owner |
| `shared_ptr` | Shared | O | O | Shared ownership |
| `weak_ptr` | None | O | X | Prevent circular reference |

### Core Principles

1. **Avoid direct new/delete** - Use `make_unique`, `make_shared`
2. **Default to unique_ptr** - Only use shared_ptr when needed
3. **Beware of circular references** - Solve with weak_ptr
4. **Follow RAII principle** - Automate resource management

---

## 12. Advanced Smart Pointer Patterns

### weak_ptr Use Cases Beyond Cycle Breaking

While `weak_ptr` is most commonly introduced for breaking circular references, it has several other valuable applications:

```cpp
#include <iostream>
#include <memory>
#include <vector>
#include <string>
#include <unordered_map>

/* Use case 1: Observer pattern
 * Why weak_ptr: Observers should not keep the subject alive. If the subject
 * is destroyed, observers should gracefully discover this rather than
 * holding a dangling reference or preventing destruction. */
class EventEmitter {
    std::vector<std::weak_ptr<std::function<void(const std::string&)>>> listeners;

public:
    void subscribe(std::shared_ptr<std::function<void(const std::string&)>> listener) {
        listeners.push_back(listener);
    }

    void emit(const std::string& event) {
        /* Iterate and clean up expired listeners in one pass */
        auto it = listeners.begin();
        while (it != listeners.end()) {
            if (auto sp = it->lock()) {
                (*sp)(event);     /* Listener still alive: invoke */
                ++it;
            } else {
                it = listeners.erase(it);  /* Listener gone: remove */
            }
        }
    }
};

/* Use case 2: Cache with automatic expiration
 * Why weak_ptr: The cache should not prevent objects from being destroyed.
 * When no one else holds a reference, the cached entry naturally expires. */
template <typename Key, typename Value>
class WeakCache {
    std::unordered_map<Key, std::weak_ptr<Value>> cache_;

public:
    std::shared_ptr<Value> get(const Key& key) {
        auto it = cache_.find(key);
        if (it != cache_.end()) {
            if (auto sp = it->second.lock()) {
                return sp;   /* Cache hit: object still alive */
            }
            cache_.erase(it);  /* Expired: clean up stale entry */
        }
        return nullptr;  /* Cache miss */
    }

    void put(const Key& key, std::shared_ptr<Value> value) {
        cache_[key] = value;
    }
};
```

### Custom Deleters for Resource Management

Custom deleters transform `unique_ptr` and `shared_ptr` into RAII wrappers for **any** resource, not just heap memory:

```cpp
#include <iostream>
#include <memory>
#include <cstdio>

/* Pattern 1: FILE* wrapper
 * Why custom deleter: fopen/fclose follow acquire/release semantics,
 * which maps perfectly to RAII. A custom deleter calls fclose. */
auto make_file(const char* path, const char* mode) {
    /* Lambda deleter: called automatically when unique_ptr goes out of scope */
    auto deleter = [](FILE* f) {
        if (f) {
            std::cout << "  Closing file\n";
            fclose(f);
        }
    };
    return std::unique_ptr<FILE, decltype(deleter)>(fopen(path, mode), deleter);
}

/* Pattern 2: C library cleanup (e.g., OpenSSL, SQLite, SDL)
 * Why: Many C libraries return opaque pointers with matching free functions.
 * Wrapping them in unique_ptr prevents resource leaks on exceptions. */
struct SDL_Window;
void SDL_DestroyWindow(SDL_Window*);  /* Forward declaration for illustration */

// auto window = std::unique_ptr<SDL_Window, decltype(&SDL_DestroyWindow)>(
//     SDL_CreateWindow(...), SDL_DestroyWindow
// );

/* Pattern 3: shared_ptr with custom deleter (simpler syntax)
 * Why shared_ptr is easier: The deleter is type-erased, so it doesn't
 * affect the shared_ptr type. No need for decltype gymnastics. */
void demo_shared_deleter() {
    auto sp = std::shared_ptr<FILE>(
        fopen("/dev/null", "w"),
        [](FILE* f) { if (f) fclose(f); }
    );
    /* sp is just std::shared_ptr<FILE> -- the deleter is hidden inside */
    if (sp) {
        fprintf(sp.get(), "Hello from shared_ptr!\n");
    }
}

int main() {
    {
        auto f = make_file("/tmp/test_smart.txt", "w");
        if (f) {
            fprintf(f.get(), "Written via unique_ptr with custom deleter\n");
            std::cout << "File is open\n";
        }
    }   /* f destroyed here → fclose called automatically */
    std::cout << "File has been closed\n";

    demo_shared_deleter();

    return 0;
}
```

### make_unique/make_shared vs Raw new: Exception Safety

```cpp
#include <iostream>
#include <memory>
#include <stdexcept>

class Widget {
public:
    Widget(int v) : value(v) { std::cout << "Widget(" << v << ")\n"; }
    ~Widget() { std::cout << "~Widget(" << value << ")\n"; }
    int value;
};

void might_throw() {
    throw std::runtime_error("oops");
}

/* DANGER: With raw new, this can leak memory!
 * Pre-C++17, the compiler could evaluate arguments in any order:
 *   1. new Widget(1)         -- allocates Widget
 *   2. might_throw()         -- throws! Widget from step 1 is LEAKED
 *   3. std::shared_ptr(...)  -- never reached
 * C++17 fixed the interleaving, but make_shared is still preferred. */
void unsafe_call(std::shared_ptr<Widget> w, int x) {
    std::cout << "Widget value: " << w->value << ", x: " << x << "\n";
}

/* SAFE: make_shared performs allocation + construction atomically.
 * Even if other arguments throw, no memory is leaked. */
void safe_call(std::shared_ptr<Widget> w, int x) {
    std::cout << "Widget value: " << w->value << ", x: " << x << "\n";
}

int main() {
    /* Prefer this: */
    auto w = std::make_shared<Widget>(42);

    /* Also preferred for unique_ptr: */
    auto u = std::make_unique<Widget>(99);

    /* Additional benefit of make_shared:
     * Single allocation for both the object and the control block.
     * With new, you get two allocations:
     *   std::shared_ptr<Widget>(new Widget(42))
     *     → allocation 1: new Widget
     *     → allocation 2: control block (ref count, weak count, deleter) */
    std::cout << "Widget: " << w->value << "\n";

    return 0;
}
```

### Pimpl Idiom with unique_ptr (Compilation Firewall)

The Pimpl (Pointer to Implementation) idiom hides implementation details behind a pointer, reducing compile-time dependencies. `unique_ptr` is the natural choice for Pimpl.

**widget.h** (header -- exposed to users):
```cpp
#ifndef WIDGET_H
#define WIDGET_H

#include <memory>
#include <string>

/* Why Pimpl: Changing the implementation (adding fields, changing types)
 * only requires recompiling widget.cpp, NOT every file that includes widget.h.
 * This dramatically speeds up builds in large projects. */
class Widget {
public:
    Widget(const std::string& name, int value);
    ~Widget();  /* Must be declared in header, defined in .cpp */

    /* Move operations must also be declared here, defined in .cpp
     * Why: The compiler needs to see ~Impl to generate move operations,
     * but Impl is only defined in the .cpp file. */
    Widget(Widget&& other) noexcept;
    Widget& operator=(Widget&& other) noexcept;

    /* Non-copyable (optional design choice) */
    Widget(const Widget&) = delete;
    Widget& operator=(const Widget&) = delete;

    void doWork();
    std::string getName() const;

private:
    struct Impl;                      /* Forward declaration only */
    std::unique_ptr<Impl> pImpl_;     /* The "compilation firewall" */
};

#endif
```

**widget.cpp** (implementation -- hidden from users):
```cpp
#include "widget.h"
#include <iostream>
#include <vector>
#include <algorithm>
// Can include heavy headers here without affecting widget.h users

/* The actual implementation struct -- can be changed freely
 * without recompiling files that include widget.h */
struct Widget::Impl {
    std::string name;
    int value;
    std::vector<int> history;   /* Adding this field doesn't change the header */

    Impl(const std::string& n, int v) : name(n), value(v) {}
};

/* Destructor must be defined where Impl is complete.
 * Why: unique_ptr needs to call delete on Impl, which requires
 * knowing Impl's size and destructor. */
Widget::Widget(const std::string& name, int value)
    : pImpl_(std::make_unique<Impl>(name, value)) {}

Widget::~Widget() = default;  /* Now the compiler can see ~Impl */

Widget::Widget(Widget&& other) noexcept = default;
Widget& Widget::operator=(Widget&& other) noexcept = default;

void Widget::doWork() {
    pImpl_->history.push_back(pImpl_->value);
    std::cout << "Widget '" << pImpl_->name << "' doing work (value="
              << pImpl_->value << ", history size="
              << pImpl_->history.size() << ")\n";
}

std::string Widget::getName() const {
    return pImpl_->name;
}
```

### unique_ptr for C API Wrappers

```cpp
#include <iostream>
#include <memory>
#include <cstdlib>
#include <cstring>

/* Simulating a C library API */
typedef struct {
    char* data;
    size_t size;
} CBuffer;

CBuffer* cbuffer_create(size_t size) {
    CBuffer* buf = (CBuffer*)malloc(sizeof(CBuffer));
    buf->data = (char*)calloc(size, 1);
    buf->size = size;
    return buf;
}

void cbuffer_destroy(CBuffer* buf) {
    if (buf) {
        free(buf->data);
        free(buf);
    }
}

/* C++ RAII wrapper using unique_ptr with custom deleter.
 * Why: This pattern works for any C API that follows create/destroy pairs.
 * The unique_ptr ensures cbuffer_destroy is called even on exceptions. */
using CBufferPtr = std::unique_ptr<CBuffer, decltype(&cbuffer_destroy)>;

CBufferPtr make_cbuffer(size_t size) {
    return CBufferPtr(cbuffer_create(size), cbuffer_destroy);
}

int main() {
    {
        auto buf = make_cbuffer(256);
        std::strncpy(buf->data, "Hello from C API wrapper!", buf->size - 1);
        std::cout << "Buffer: " << buf->data << "\n";
        std::cout << "Size: " << buf->size << "\n";
    }   /* cbuffer_destroy called automatically */

    std::cout << "Buffer has been destroyed\n";

    return 0;
}
```

---

## 13. Exercises

### Exercise 1: Resource Manager

Implement a class that manages various resources (file, network connection, etc.) using `unique_ptr`.

### Exercise 2: Graph Data Structure

Implement a graph where nodes are connected to each other using `shared_ptr` and `weak_ptr`.

### Exercise 3: Object Pool

Implement a reusable object pool using smart pointers.

---

## Next Step

Let's learn about C++11/14/17/20 major features in [15_Modern_CPP.md](./15_Modern_CPP.md)!
