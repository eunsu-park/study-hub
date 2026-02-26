# STL Containers

**Previous**: [Inheritance and Polymorphism](./09_Inheritance_and_Polymorphism.md) | **Next**: [STL Algorithms and Iterators](./11_STL_Algorithms_Iterators.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain what the STL is and identify its four main components: containers, iterators, algorithms, and function objects
2. Implement dynamic arrays with `std::vector` and apply element access, insertion, deletion, and iterator traversal
3. Compare sequence containers (`vector`, `array`, `deque`, `list`) and select the appropriate one for a given use case
4. Apply associative containers (`set`, `map`) and their unordered counterparts for sorted and hash-based storage
5. Implement LIFO, FIFO, and priority-based logic using `stack`, `queue`, and `priority_queue` adapters
6. Distinguish between ordered and unordered containers in terms of internal structure, time complexity, and iteration order
7. Design composite data with `std::pair` and `std::tuple`, and destructure them with C++17 structured bindings

---

The Standard Template Library is where C++ truly shines. Instead of reinventing linked lists, hash maps, or sorting algorithms for every project, you can rely on battle-tested, highly optimized containers and algorithms that have been refined over decades. Learning which container to reach for -- and understanding the Big-O trade-offs behind each choice -- is one of the most impactful skills for writing performant, idiomatic C++ code.

## 1. What is STL?

STL (Standard Template Library) is the core of the C++ standard library, providing data structures and algorithms.

### STL Components

| Component | Description |
|-----------|-------------|
| Containers | Data structures for storing data |
| Iterators | Traversing container elements |
| Algorithms | General-purpose functions like sorting, searching |
| Function Objects | Objects that behave like functions |

---

## 2. vector

A dynamic-sized array. Most commonly used.

> **Analogy -- The Expanding Row of Seats**: Think of `std::vector` as a row of seats in a theater. When the row is full and a new guest arrives, the theater doesn't add one chair -- it moves everyone to a bigger row (typically double the size). This is why `push_back` is amortized O(1): most additions are instant, but occasionally the entire audience must relocate.

### Basic Usage

```cpp
#include <iostream>
#include <vector>

int main() {
    // Creation
    std::vector<int> v1;                  // Empty vector
    std::vector<int> v2(5);               // Size 5, initialized to 0
    std::vector<int> v3(5, 10);           // Size 5, initialized to 10
    std::vector<int> v4 = {1, 2, 3, 4, 5}; // Initializer list

    // Adding elements
    v1.push_back(10);
    v1.push_back(20);
    v1.push_back(30);

    // Output
    for (int num : v1) {
        std::cout << num << " ";
    }
    std::cout << std::endl;  // 10 20 30

    return 0;
}
```

### Element Access

```cpp
#include <iostream>
#include <vector>

int main() {
    std::vector<int> v = {10, 20, 30, 40, 50};

    // Index access
    std::cout << v[0] << std::endl;      // 10
    std::cout << v.at(2) << std::endl;   // 30 (range checked)

    // First/last
    std::cout << v.front() << std::endl;  // 10
    std::cout << v.back() << std::endl;   // 50

    // Size
    std::cout << "Size: " << v.size() << std::endl;
    std::cout << "Empty: " << v.empty() << std::endl;

    return 0;
}
```

### Insertion and Deletion

```cpp
#include <iostream>
#include <vector>

int main() {
    std::vector<int> v = {1, 2, 3, 4, 5};

    // Add/remove at end
    v.push_back(6);   // {1, 2, 3, 4, 5, 6}
    v.pop_back();     // {1, 2, 3, 4, 5}

    // Insert in middle
    v.insert(v.begin() + 2, 100);  // {1, 2, 100, 3, 4, 5}

    // Delete in middle
    v.erase(v.begin() + 2);  // {1, 2, 3, 4, 5}

    // Range deletion
    v.erase(v.begin(), v.begin() + 2);  // {3, 4, 5}

    // Clear all
    v.clear();

    return 0;
}
```

### Iterators

```cpp
#include <iostream>
#include <vector>

int main() {
    std::vector<int> v = {1, 2, 3, 4, 5};

    // Iterate with iterator
    for (std::vector<int>::iterator it = v.begin(); it != v.end(); ++it) {
        std::cout << *it << " ";
    }
    std::cout << std::endl;

    // Using auto (recommended)
    for (auto it = v.begin(); it != v.end(); ++it) {
        std::cout << *it << " ";
    }
    std::cout << std::endl;

    // Reverse iterator
    for (auto it = v.rbegin(); it != v.rend(); ++it) {
        std::cout << *it << " ";
    }
    std::cout << std::endl;  // 5 4 3 2 1

    return 0;
}
```

---

## 3. array

A fixed-size array.

```cpp
#include <iostream>
#include <array>

int main() {
    std::array<int, 5> arr = {1, 2, 3, 4, 5};

    // Access
    std::cout << arr[0] << std::endl;
    std::cout << arr.at(2) << std::endl;
    std::cout << arr.front() << std::endl;
    std::cout << arr.back() << std::endl;

    // Size
    std::cout << "Size: " << arr.size() << std::endl;

    // Iterate
    for (int num : arr) {
        std::cout << num << " ";
    }
    std::cout << std::endl;

    // Fill
    arr.fill(0);

    return 0;
}
```

---

## 4. deque

A container with fast insertion/deletion at both ends.

```cpp
#include <iostream>
#include <deque>

int main() {
    std::deque<int> dq;

    // Add to front/back
    dq.push_back(1);
    dq.push_back(2);
    dq.push_front(0);
    dq.push_front(-1);

    // {-1, 0, 1, 2}
    for (int num : dq) {
        std::cout << num << " ";
    }
    std::cout << std::endl;

    // Remove from front/back
    dq.pop_front();
    dq.pop_back();

    // {0, 1}
    for (int num : dq) {
        std::cout << num << " ";
    }
    std::cout << std::endl;

    return 0;
}
```

---

## 5. list

A doubly-linked list.

> **Analogy -- The Sticky-Note Chain**: A `std::list` is like a chain of sticky notes, each one saying "the next note is on page X." You can insert or remove a note anywhere instantly (just rewrite one pointer), but finding the 50th note means following the chain from the beginning -- there's no shortcut.

```cpp
#include <iostream>
#include <list>

int main() {
    std::list<int> lst = {3, 1, 4, 1, 5};

    // Add to front/back
    lst.push_front(0);
    lst.push_back(9);

    // Sort (member method)
    lst.sort();

    for (int num : lst) {
        std::cout << num << " ";
    }
    std::cout << std::endl;  // 0 1 1 3 4 5 9

    // Remove duplicates (consecutive only)
    lst.unique();

    for (int num : lst) {
        std::cout << num << " ";
    }
    std::cout << std::endl;  // 0 1 3 4 5 9

    // Insert
    auto it = lst.begin();
    std::advance(it, 2);  // Move 2 positions
    lst.insert(it, 100);  // Insert at that position

    return 0;
}
```

---

## 6. set

A sorted collection of unique elements.

```cpp
#include <iostream>
#include <set>

int main() {
    std::set<int> s;

    // Insert
    s.insert(30);
    s.insert(10);
    s.insert(20);
    s.insert(10);  // Duplicate, ignored

    // Auto-sorted
    for (int num : s) {
        std::cout << num << " ";
    }
    std::cout << std::endl;  // 10 20 30

    // Size
    std::cout << "Size: " << s.size() << std::endl;  // 3

    // Search
    if (s.find(20) != s.end()) {
        std::cout << "20 found" << std::endl;
    }

    // count (0 or 1)
    std::cout << "Count of 30: " << s.count(30) << std::endl;

    // Delete
    s.erase(20);

    return 0;
}
```

### multiset

A set that allows duplicates.

```cpp
#include <iostream>
#include <set>

int main() {
    std::multiset<int> ms;

    ms.insert(10);
    ms.insert(10);
    ms.insert(20);
    ms.insert(10);

    for (int num : ms) {
        std::cout << num << " ";
    }
    std::cout << std::endl;  // 10 10 10 20

    std::cout << "Count of 10: " << ms.count(10) << std::endl;  // 3

    return 0;
}
```

---

## 7. map

A sorted container of key-value pairs.

```cpp
#include <iostream>
#include <map>
#include <string>

int main() {
    std::map<std::string, int> ages;

    // Insert
    ages["Alice"] = 25;
    ages["Bob"] = 30;
    ages.insert({"Charlie", 35});
    ages.insert(std::make_pair("David", 40));

    // Access
    std::cout << "Alice: " << ages["Alice"] << std::endl;

    // Iterate (sorted by key)
    for (const auto& pair : ages) {
        std::cout << pair.first << ": " << pair.second << std::endl;
    }

    // Structured binding (C++17)
    for (const auto& [name, age] : ages) {
        std::cout << name << ": " << age << std::endl;
    }

    // Search
    if (ages.find("Alice") != ages.end()) {
        std::cout << "Alice found" << std::endl;
    }

    // Delete
    ages.erase("Bob");

    return 0;
}
```

### Note: operator[]

```cpp
std::map<std::string, int> m;

// Accessing non-existent key → inserts with default value (0)!
std::cout << m["unknown"] << std::endl;  // 0 (and gets inserted)
std::cout << m.size() << std::endl;      // 1

// Safe access
if (m.count("key") > 0) {
    std::cout << m["key"] << std::endl;
}

// Or use find
auto it = m.find("key");
if (it != m.end()) {
    std::cout << it->second << std::endl;
}
```

---

## 8. unordered_set / unordered_map

Hash table-based containers with average O(1) access.

### unordered_set

```cpp
#include <iostream>
#include <unordered_set>

int main() {
    std::unordered_set<int> us;

    us.insert(30);
    us.insert(10);
    us.insert(20);

    // Order not guaranteed
    for (int num : us) {
        std::cout << num << " ";
    }
    std::cout << std::endl;  // Order undefined

    // Search (O(1) average)
    if (us.count(20)) {
        std::cout << "20 found" << std::endl;
    }

    return 0;
}
```

### unordered_map

```cpp
#include <iostream>
#include <unordered_map>
#include <string>

int main() {
    std::unordered_map<std::string, int> umap;

    umap["apple"] = 100;
    umap["banana"] = 200;
    umap["cherry"] = 300;

    // Access (O(1) average)
    std::cout << "apple: " << umap["apple"] << std::endl;

    // Iterate (order undefined)
    for (const auto& [key, value] : umap) {
        std::cout << key << ": " << value << std::endl;
    }

    return 0;
}
```

### set vs unordered_set

| Feature | set | unordered_set |
|---------|-----|---------------|
| Internal structure | Red-black tree | Hash table |
| Sorted | Yes | No |
| Insert/search | O(log n) | O(1) average |
| Iteration order | Sorted | Undefined |

---

## 9. stack and queue

Container adapters.

### stack (LIFO)

```cpp
#include <iostream>
#include <stack>

int main() {
    std::stack<int> s;

    // push
    s.push(10);
    s.push(20);
    s.push(30);

    // pop (LIFO)
    while (!s.empty()) {
        std::cout << s.top() << " ";  // Top element
        s.pop();
    }
    std::cout << std::endl;  // 30 20 10

    return 0;
}
```

### queue (FIFO)

```cpp
#include <iostream>
#include <queue>

int main() {
    std::queue<int> q;

    // push
    q.push(10);
    q.push(20);
    q.push(30);

    // pop (FIFO)
    while (!q.empty()) {
        std::cout << q.front() << " ";  // Front element
        q.pop();
    }
    std::cout << std::endl;  // 10 20 30

    return 0;
}
```

### priority_queue

```cpp
#include <iostream>
#include <queue>

int main() {
    // Default: max heap (larger values first)
    std::priority_queue<int> pq;

    pq.push(30);
    pq.push(10);
    pq.push(20);

    while (!pq.empty()) {
        std::cout << pq.top() << " ";
        pq.pop();
    }
    std::cout << std::endl;  // 30 20 10

    // Min heap
    std::priority_queue<int, std::vector<int>, std::greater<int>> minPq;

    minPq.push(30);
    minPq.push(10);
    minPq.push(20);

    while (!minPq.empty()) {
        std::cout << minPq.top() << " ";
        minPq.pop();
    }
    std::cout << std::endl;  // 10 20 30

    return 0;
}
```

---

## 10. pair and tuple

### pair

```cpp
#include <iostream>
#include <utility>

int main() {
    // Creation
    std::pair<std::string, int> p1("Alice", 25);
    auto p2 = std::make_pair("Bob", 30);

    // Access
    std::cout << p1.first << ": " << p1.second << std::endl;

    // Comparison
    if (p1 < p2) {  // Compares first, then second
        std::cout << p1.first << " < " << p2.first << std::endl;
    }

    return 0;
}
```

### tuple

```cpp
#include <iostream>
#include <tuple>
#include <string>

int main() {
    // Creation
    std::tuple<std::string, int, double> t("Alice", 25, 165.5);

    // Access
    std::cout << std::get<0>(t) << std::endl;  // Alice
    std::cout << std::get<1>(t) << std::endl;  // 25
    std::cout << std::get<2>(t) << std::endl;  // 165.5

    // Structured binding (C++17)
    auto [name, age, height] = t;
    std::cout << name << ", " << age << ", " << height << std::endl;

    return 0;
}
```

---

## 11. Container Selection Guide

| Requirement | Recommended Container |
|-------------|----------------------|
| Sequential access + end insertion/deletion | `vector` |
| Both ends insertion/deletion | `deque` |
| Frequent middle insertion/deletion | `list` |
| Unique elements + sorted | `set` |
| Unique elements + fast search | `unordered_set` |
| Key-value + sorted | `map` |
| Key-value + fast search | `unordered_map` |
| LIFO | `stack` |
| FIFO | `queue` |
| Priority | `priority_queue` |

---

## 12. Custom Allocators and Hash Customization

### Why Custom Allocators?

Every STL container accepts an optional allocator template parameter. By default, `std::allocator<T>` uses `new`/`delete`, but custom allocators let you:

- **Memory pools**: Pre-allocate a large block and hand out fixed-size chunks (eliminates per-allocation system call overhead)
- **Arena allocation**: Allocate many objects in a contiguous region, then free them all at once (useful in game engines, compilers, request-scoped servers)
- **Tracking**: Count allocations, detect leaks, log memory usage
- **Alignment**: Guarantee specific alignment for SIMD or hardware requirements

### Minimal Custom Allocator (C++17)

C++17 drastically simplified the allocator requirements. You only need `allocate`, `deallocate`, and a few type aliases:

```cpp
#include <iostream>
#include <vector>
#include <cstdlib>
#include <memory>

/* A tracking allocator that counts how many bytes have been allocated.
 * Why: Useful for debugging, profiling, or enforcing memory budgets. */
template <typename T>
struct TrackingAllocator {
    using value_type = T;

    /* Shared counter across all rebound copies of this allocator.
     * Why shared_ptr: When a container rebound allocator (e.g., for internal nodes),
     * we still want to track total memory through one counter. */
    std::shared_ptr<std::size_t> total_allocated;

    TrackingAllocator()
        : total_allocated(std::make_shared<std::size_t>(0)) {}

    /* Rebinding constructor: allows the allocator to be used for a different type.
     * Why needed: std::vector<T, Alloc> internally needs Alloc<SomeInternalType>,
     * so the container converts your allocator via this constructor. */
    template <typename U>
    TrackingAllocator(const TrackingAllocator<U>& other)
        : total_allocated(other.total_allocated) {}

    T* allocate(std::size_t n) {
        std::size_t bytes = n * sizeof(T);
        *total_allocated += bytes;
        std::cout << "[alloc] " << bytes << " bytes (total: "
                  << *total_allocated << ")\n";
        return static_cast<T*>(std::malloc(bytes));
    }

    void deallocate(T* ptr, std::size_t n) {
        std::size_t bytes = n * sizeof(T);
        *total_allocated -= bytes;
        std::cout << "[dealloc] " << bytes << " bytes (total: "
                  << *total_allocated << ")\n";
        std::free(ptr);
    }

    /* Required for container equality checks. Two allocators are "equal"
     * if memory from one can be freed by the other. */
    template <typename U>
    bool operator==(const TrackingAllocator<U>&) const { return true; }
    template <typename U>
    bool operator!=(const TrackingAllocator<U>&) const { return false; }
};

int main() {
    /* Use the tracking allocator with std::vector */
    std::vector<int, TrackingAllocator<int>> v;

    v.push_back(1);   // allocates initial buffer
    v.push_back(2);
    v.push_back(3);
    v.push_back(4);
    v.push_back(5);   // may trigger reallocation (capacity doubling)

    std::cout << "Vector contents: ";
    for (int x : v) std::cout << x << " ";
    std::cout << std::endl;

    return 0;
}
```

### Arena Allocator Concept

```cpp
#include <iostream>
#include <vector>
#include <cstdint>

/* A simple arena (bump) allocator: allocates from a fixed-size buffer.
 * Why: Extremely fast allocation (just increment a pointer), and all memory
 * is freed at once when the arena is destroyed. No per-object deallocation. */
template <typename T>
struct ArenaAllocator {
    using value_type = T;

    /* Shared arena state */
    struct Arena {
        std::uint8_t* buffer;
        std::size_t   capacity;
        std::size_t   offset;

        Arena(std::size_t cap)
            : buffer(new std::uint8_t[cap]), capacity(cap), offset(0) {}
        ~Arena() { delete[] buffer; }
    };

    std::shared_ptr<Arena> arena;

    explicit ArenaAllocator(std::size_t capacity)
        : arena(std::make_shared<Arena>(capacity)) {}

    template <typename U>
    ArenaAllocator(const ArenaAllocator<U>& other)
        : arena(other.arena) {}

    T* allocate(std::size_t n) {
        std::size_t bytes = n * sizeof(T);
        /* Align to alignof(T) */
        std::size_t aligned = (arena->offset + alignof(T) - 1) & ~(alignof(T) - 1);
        if (aligned + bytes > arena->capacity) {
            throw std::bad_alloc();
        }
        T* result = reinterpret_cast<T*>(arena->buffer + aligned);
        arena->offset = aligned + bytes;
        return result;
    }

    /* Arena allocator: deallocation is a no-op. Memory is freed all at once. */
    void deallocate(T*, std::size_t) { /* intentionally empty */ }

    template <typename U> bool operator==(const ArenaAllocator<U>&) const { return true; }
    template <typename U> bool operator!=(const ArenaAllocator<U>&) const { return false; }
};

int main() {
    ArenaAllocator<int> alloc(4096);  // 4KB arena
    std::vector<int, ArenaAllocator<int>> v(alloc);

    for (int i = 0; i < 100; i++) {
        v.push_back(i);
    }
    std::cout << "Arena used: " << alloc.arena->offset << " bytes\n";

    return 0;  // All arena memory freed at once in Arena destructor
}
```

### Custom Hash for `unordered_map`

By default, `std::unordered_map` and `std::unordered_set` use `std::hash<Key>`, which only works for built-in types and `std::string`. For custom types, you must provide a hash function.

```cpp
#include <iostream>
#include <unordered_map>
#include <string>
#include <functional>

struct Point {
    int x, y;

    /* operator== is REQUIRED for unordered containers.
     * Why: After hashing, the container needs equality to handle collisions. */
    bool operator==(const Point& other) const {
        return x == other.x && y == other.y;
    }
};

/* Method 1: Specialize std::hash (preferred for widely-used types) */
template <>
struct std::hash<Point> {
    std::size_t operator()(const Point& p) const {
        /* Hash combine pattern: mix the hashes of individual fields.
         * Why this formula? Multiplying by a prime and XORing prevents
         * (1,2) and (2,1) from producing the same hash. The shift and
         * golden ratio constant (0x9e3779b9) spread bits evenly. */
        std::size_t h1 = std::hash<int>{}(p.x);
        std::size_t h2 = std::hash<int>{}(p.y);
        return h1 ^ (h2 * 0x9e3779b9 + (h1 << 6) + (h1 >> 2));
    }
};

int main() {
    /* After specializing std::hash, Point works directly as a key */
    std::unordered_map<Point, std::string> labels;
    labels[{0, 0}] = "origin";
    labels[{1, 2}] = "point A";
    labels[{3, 4}] = "point B";

    for (const auto& [pt, label] : labels) {
        std::cout << "(" << pt.x << ", " << pt.y << "): "
                  << label << "\n";
    }

    return 0;
}
```

### Hash Combine for Composite Keys

```cpp
#include <iostream>
#include <unordered_map>
#include <string>
#include <functional>

/* A reusable hash_combine utility.
 * Why: Combining multiple field hashes into one is a recurring need.
 * This is modeled after boost::hash_combine. */
inline void hash_combine(std::size_t& seed, std::size_t value) {
    seed ^= value + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

struct Employee {
    std::string department;
    std::string name;
    int id;

    bool operator==(const Employee& o) const {
        return department == o.department && name == o.name && id == o.id;
    }
};

/* Method 2: Functor passed as template argument (for local/specific use) */
struct EmployeeHash {
    std::size_t operator()(const Employee& e) const {
        std::size_t seed = 0;
        hash_combine(seed, std::hash<std::string>{}(e.department));
        hash_combine(seed, std::hash<std::string>{}(e.name));
        hash_combine(seed, std::hash<int>{}(e.id));
        return seed;
    }
};

int main() {
    /* Pass the hash functor as the third template argument */
    std::unordered_map<Employee, double, EmployeeHash> salaries;

    salaries[{"Engineering", "Alice", 1001}] = 95000.0;
    salaries[{"Marketing",   "Bob",   2001}] = 85000.0;

    for (const auto& [emp, salary] : salaries) {
        std::cout << emp.name << " (" << emp.department << "): $"
                  << salary << "\n";
    }

    return 0;
}
```

### When to Customize

| Scenario | Solution |
|----------|----------|
| Custom type as `unordered_map` key | Specialize `std::hash` or pass hash functor |
| Composite key (multiple fields) | Use `hash_combine` pattern |
| Need deterministic memory allocation | Custom allocator with arena/pool |
| Memory usage tracking or budgets | Tracking allocator |
| High-frequency, same-size allocations | Pool allocator |

---

## 13. Summary

| Container | Characteristics |
|-----------|----------------|
| `vector` | Dynamic array, O(1) at end |
| `array` | Fixed array |
| `deque` | O(1) at both ends |
| `list` | Doubly-linked list |
| `set` | Sorted + unique |
| `map` | Key-value + sorted |
| `unordered_set` | Hash + unique |
| `unordered_map` | Hash + key-value |
| `stack` | LIFO |
| `queue` | FIFO |
| `priority_queue` | Heap |

---

## Exercises

### Exercise 1: Container Selection Justification

For each of the following scenarios, choose the most appropriate STL container and justify your choice in terms of time complexity and use-case fit:

1. A browser history where the most recently visited page is always retrieved first (LIFO).
2. A dictionary that maps English words to Korean translations, looked up by the English word frequently but iteration order doesn't matter.
3. A ranked leaderboard where you always need the player with the highest score instantly.
4. A to-do list where items are processed in the order they were added (FIFO) and new tasks are frequently added at both ends.
5. A set of unique IP addresses that need to be checked for membership millions of times per second.

Write one or two sentences per scenario explaining the reasoning, including the Big-O complexity of the key operation.

### Exercise 2: Vector Growth Observation

Write a program that pushes 20 integers into an empty `std::vector<int>` one by one and prints the `size()` and `capacity()` after each `push_back`. Observe where `capacity()` jumps and confirm it doubles. Then use `reserve(20)` before the loop and repeat — note how the number of reallocations changes.

```cpp
#include <iostream>
#include <vector>

int main() {
    std::vector<int> v;
    // Optional: v.reserve(20);
    for (int i = 1; i <= 20; i++) {
        v.push_back(i);
        std::cout << "size=" << v.size()
                  << " capacity=" << v.capacity() << "\n";
    }
    return 0;
}
```

### Exercise 3: Word Frequency Counter

Write a program that reads words from a `std::vector<std::string>` (hardcoded or from `std::cin`) and uses a `std::map<std::string, int>` to count how many times each word appears. After counting, iterate the map to print each word and its count in alphabetical order. Then repeat using `std::unordered_map` and compare the output ordering.

### Exercise 4: Bracket Matching with stack

Use `std::stack<char>` to verify whether a string of brackets is balanced. The function `bool isBalanced(const std::string& s)` should return `true` if every opening bracket (`(`, `[`, `{`) has a matching closing bracket in the correct order, and `false` otherwise. Test with at least five inputs: `"([]{})"`, `"([)]"`, `""`, `"((("`, and `"{[()]}"`.

### Exercise 5: Custom Hash for Composite Key

Extend the hash customization example from the lesson. Define a struct `Point3D { int x, y, z; }` and specialize `std::hash<Point3D>` using the `hash_combine` technique. Store several `Point3D → std::string` mappings in an `std::unordered_map<Point3D, std::string>` and verify that lookup works correctly. Add `operator==` to `Point3D` and explain in a comment why it is required alongside the hash specialization.

---

## Next Steps

Let's learn about STL algorithms in [11_STL_Algorithms_Iterators.md](./11_STL_Algorithms_Iterators.md)!
