# C++20 Ranges

**Previous**: [C++20 Concepts](./08_CPP20_Concepts.md) | **Next**: [C++20 Coroutines](./10_CPP20_Coroutines.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain the range concept and how `begin()`/`end()` pairs define a range in C++20
2. Distinguish views from containers and describe the lazy, non-owning, composable nature of views
3. Compose range pipelines using adaptors such as `views::filter`, `views::transform`, `views::take`, and `views::drop`
4. Apply the pipe operator (`|`) to chain multiple range adaptors into readable data-processing pipelines
5. Use range factories like `views::iota`, `views::empty`, and `views::single` to generate sequences
6. Leverage projections in range algorithms to operate on specific members without writing custom comparators
7. Implement a simple custom view adaptor that integrates with the standard range machinery

---

C++20 Ranges replace the iterator-pair convention that has dominated the STL since its inception. Instead of passing two iterators to every algorithm, you pass a single range object. Views add a composable, lazy layer on top: you describe *what* to compute, and evaluation happens only when elements are actually consumed. The result is code that reads like a data-flow description rather than a loop-centric procedure. Mastering ranges is the key to writing expressive, efficient C++20 pipelines.

---

## Table of Contents

1. [Ranges Overview](#1-ranges-overview)
2. [Views](#2-views)
3. [Range Adaptors](#3-range-adaptors)
4. [Pipe Operator](#4-pipe-operator)
5. [Range Factories](#5-range-factories)
6. [Projections](#6-projections)
7. [Range Algorithms](#7-range-algorithms)
8. [Custom Views](#8-custom-views)

---

## 1. Ranges Overview

### What Is a Range?

A range is any type that provides `begin()` and `end()`. All standard containers are ranges, but so is any user-defined type that satisfies the `std::ranges::range` concept.

```cpp
#include <ranges>
#include <vector>
#include <iostream>

// The range concept (simplified):
// template<typename R>
// concept range = requires(R& r) {
//     std::ranges::begin(r);
//     std::ranges::end(r);
// };

void print_range(std::ranges::range auto&& r) {
    for (const auto& elem : r) {
        std::cout << elem << " ";
    }
    std::cout << "\n";
}

int main() {
    std::vector<int> v = {1, 2, 3, 4, 5};
    int arr[] = {10, 20, 30};

    print_range(v);    // 1 2 3 4 5
    print_range(arr);  // 10 20 30
    return 0;
}
```

### Why Ranges Improve C++

| Traditional STL | C++20 Ranges |
|-----------------|--------------|
| `std::sort(v.begin(), v.end())` | `std::ranges::sort(v)` |
| Two iterators per call | Single range object |
| Error-prone iterator mismatches | Type-safe range passing |
| Manual loop composition | Composable view pipelines |
| Eager evaluation only | Lazy views + eager algorithms |

### Range Categories

```cpp
// Ranges refine into categories, mirroring iterator categories:
// input_range        — single-pass read
// forward_range      — multi-pass read
// bidirectional_range — forward + backward
// random_access_range — O(1) element access
// contiguous_range   — elements in contiguous memory (vector, array, span)

#include <ranges>
#include <vector>
#include <list>

static_assert(std::ranges::random_access_range<std::vector<int>>);
static_assert(std::ranges::bidirectional_range<std::list<int>>);
static_assert(std::ranges::contiguous_range<std::vector<int>>);
```

---

## 2. Views

### Lazy, Non-Owning, Composable

A view is a lightweight range that does **not** own its elements. Views are:

- **Lazy**: computation happens only when you iterate
- **Non-owning**: they reference existing data
- **Composable**: you can stack views on top of each other
- **O(1) copy/move**: they store only a reference plus small state

```cpp
#include <ranges>
#include <vector>
#include <iostream>

int main() {
    std::vector<int> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

    // This does NOT compute anything yet — it's lazy
    auto view = data
        | std::views::filter([](int x) { return x % 2 == 0; })
        | std::views::transform([](int x) { return x * x; });

    // Computation happens here, during iteration
    for (int val : view) {
        std::cout << val << " ";  // 4 16 36 64 100
    }
    std::cout << "\n";

    return 0;
}
```

### Views vs Containers

| Property | Container (`vector`, `list`) | View (`filter_view`, `transform_view`) |
|----------|------------------------------|----------------------------------------|
| Owns data | Yes | No |
| Copy cost | O(n) | O(1) |
| Evaluation | Eager | Lazy |
| Mutation | Elements modifiable | Depends on underlying range |
| Storage | Allocates memory | Stores reference + state |

---

## 3. Range Adaptors

Range adaptors are factory functions in `std::views` that create views from existing ranges.

### views::filter

```cpp
#include <ranges>
#include <vector>
#include <iostream>

int main() {
    std::vector<int> v = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

    for (int n : v | std::views::filter([](int x) { return x % 3 == 0; })) {
        std::cout << n << " ";  // 3 6 9
    }
    std::cout << "\n";
    return 0;
}
```

### views::transform

```cpp
#include <ranges>
#include <vector>
#include <string>
#include <iostream>

int main() {
    std::vector<std::string> names = {"alice", "bob", "charlie"};

    auto upper_first = names | std::views::transform([](std::string s) {
        if (!s.empty()) s[0] = static_cast<char>(std::toupper(s[0]));
        return s;
    });

    for (const auto& name : upper_first) {
        std::cout << name << " ";  // Alice Bob Charlie
    }
    std::cout << "\n";
    return 0;
}
```

### views::take and views::drop

```cpp
#include <ranges>
#include <vector>
#include <iostream>

int main() {
    std::vector<int> v = {10, 20, 30, 40, 50, 60};

    // Take first 3
    for (int n : v | std::views::take(3)) {
        std::cout << n << " ";  // 10 20 30
    }
    std::cout << "\n";

    // Drop first 3
    for (int n : v | std::views::drop(3)) {
        std::cout << n << " ";  // 40 50 60
    }
    std::cout << "\n";

    return 0;
}
```

### Other Useful Adaptors

```cpp
#include <ranges>
#include <vector>
#include <string>
namespace views = std::views;

std::vector<int> v = {5, 3, 1, 4, 2};

auto r1 = v | views::reverse;                    // 2 4 1 3 5
auto r2 = v | views::take_while([](int x) { return x > 2; });  // 5 3
auto r3 = v | views::drop_while([](int x) { return x > 2; });  // 1 4 2

// Split a string
std::string csv = "one,two,three";
auto r4 = csv | views::split(',');  // ["one", "two", "three"]

// Flatten nested ranges
std::vector<std::vector<int>> nested = {{1,2}, {3,4}, {5}};
auto r5 = nested | views::join;  // 1 2 3 4 5

// Access tuple/pair elements
std::vector<std::pair<std::string, int>> pairs = {{"a", 1}, {"b", 2}};
auto r6 = pairs | views::keys;    // "a" "b"
auto r7 = pairs | views::values;  // 1 2
```

---

## 4. Pipe Operator

The pipe operator (`|`) chains adaptors left to right, mimicking Unix pipes. Each adaptor receives the result of the previous one.

```cpp
#include <ranges>
#include <vector>
#include <iostream>
#include <numeric>

int main() {
    std::vector<int> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

    // Pipeline: keep evens -> square them -> take first 3
    auto pipeline = data
        | std::views::filter([](int x) { return x % 2 == 0; })
        | std::views::transform([](int x) { return x * x; })
        | std::views::take(3);

    for (int n : pipeline) {
        std::cout << n << " ";  // 4 16 36
    }
    std::cout << "\n";

    // Sum via accumulate (ranges to eager)
    auto squares = data
        | std::views::transform([](int x) { return x * x; });

    int total = 0;
    for (int n : squares) total += n;
    std::cout << "Sum of squares: " << total << "\n";  // 385

    return 0;
}
```

### Storing Pipelines

```cpp
#include <ranges>
#include <vector>

// You can store an adaptor for reuse
auto even_squares = std::views::filter([](int x) { return x % 2 == 0; })
                  | std::views::transform([](int x) { return x * x; });

int main() {
    std::vector<int> v1 = {1, 2, 3, 4, 5};
    std::vector<int> v2 = {10, 11, 12, 13, 14};

    // Apply the same pipeline to different data
    for (int n : v1 | even_squares) { /* 4 16 */ }
    for (int n : v2 | even_squares) { /* 100 144 196 */ }
    return 0;
}
```

---

## 5. Range Factories

Range factories create ranges from scratch, without an underlying container.

### views::iota

```cpp
#include <ranges>
#include <iostream>

int main() {
    // Bounded iota: [1, 10)
    for (int n : std::views::iota(1, 10)) {
        std::cout << n << " ";  // 1 2 3 4 5 6 7 8 9
    }
    std::cout << "\n";

    // Unbounded (infinite) iota — must limit with take
    for (int n : std::views::iota(100) | std::views::take(5)) {
        std::cout << n << " ";  // 100 101 102 103 104
    }
    std::cout << "\n";

    return 0;
}
```

### views::empty and views::single

```cpp
#include <ranges>
#include <iostream>

int main() {
    // Empty range of ints
    auto empty = std::views::empty<int>;
    // Useful as a default/sentinel in generic code

    // Single-element range
    for (int n : std::views::single(42)) {
        std::cout << n << "\n";  // 42
    }

    return 0;
}
```

### views::repeat (C++23)

```cpp
#include <ranges>

// Repeat a value N times (C++23)
auto fives = std::views::repeat(5, 3);  // 5 5 5

// Infinite repeat — limit with take
auto ones = std::views::repeat(1) | std::views::take(10);
```

---

## 6. Projections

Projections let you tell a range algorithm *which part* of each element to operate on, without writing a custom comparator or transforming the data first.

```cpp
#include <ranges>
#include <algorithm>
#include <vector>
#include <string>
#include <iostream>

struct Employee {
    std::string name;
    int age;
    double salary;
};

int main() {
    std::vector<Employee> staff = {
        {"Alice", 35, 90000},
        {"Bob", 28, 75000},
        {"Charlie", 42, 110000},
        {"Diana", 31, 85000},
    };

    // Sort by age using projection (no custom comparator needed)
    std::ranges::sort(staff, {}, &Employee::age);
    // Bob(28), Diana(31), Alice(35), Charlie(42)

    // Sort by salary descending
    std::ranges::sort(staff, std::ranges::greater{}, &Employee::salary);
    // Charlie(110k), Alice(90k), Diana(85k), Bob(75k)

    // Find by name
    auto it = std::ranges::find(staff, "Bob", &Employee::name);
    if (it != staff.end()) {
        std::cout << it->name << " earns $" << it->salary << "\n";
    }

    // Min/max by age
    auto youngest = std::ranges::min(staff, {}, &Employee::age);
    std::cout << "Youngest: " << youngest.name << " (" << youngest.age << ")\n";

    return 0;
}
```

### Projection with Lambda

```cpp
#include <ranges>
#include <algorithm>
#include <vector>
#include <string>

int main() {
    std::vector<std::string> words = {"Banana", "apple", "Cherry"};

    // Case-insensitive sort using projection
    std::ranges::sort(words, {}, [](const std::string& s) {
        std::string lower = s;
        std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
        return lower;
    });
    // apple, Banana, Cherry

    return 0;
}
```

---

## 7. Range Algorithms

C++20 provides range-based versions of most `<algorithm>` functions in the `std::ranges` namespace. They accept ranges directly and support projections.

### Common Range Algorithms

```cpp
#include <ranges>
#include <algorithm>
#include <vector>
#include <iostream>

int main() {
    std::vector<int> v = {3, 1, 4, 1, 5, 9, 2, 6, 5};

    // Sort
    std::ranges::sort(v);

    // Find
    auto it = std::ranges::find(v, 5);

    // Count
    auto cnt = std::ranges::count(v, 5);

    // For each
    std::ranges::for_each(v, [](int x) { std::cout << x << " "; });
    std::cout << "\n";

    // Any/All/None
    bool has_neg = std::ranges::any_of(v, [](int x) { return x < 0; });
    bool all_pos = std::ranges::all_of(v, [](int x) { return x > 0; });

    // Min/Max
    auto [lo, hi] = std::ranges::minmax(v);
    std::cout << "Min: " << lo << ", Max: " << hi << "\n";

    // Contains (C++23, but widely available)
    // bool found = std::ranges::contains(v, 5);

    // Copy to output
    std::vector<int> dest;
    std::ranges::copy(v, std::back_inserter(dest));

    // Remove-erase idiom simplified
    auto [rem_begin, rem_end] = std::ranges::remove(v, 1);
    v.erase(rem_begin, rem_end);

    return 0;
}
```

### Algorithm Comparison

| Traditional | Range-based |
|-------------|-------------|
| `std::sort(v.begin(), v.end())` | `std::ranges::sort(v)` |
| `std::find(v.begin(), v.end(), x)` | `std::ranges::find(v, x)` |
| `std::count_if(v.begin(), v.end(), pred)` | `std::ranges::count_if(v, pred)` |
| `std::transform(v.begin(), v.end(), out, f)` | `std::ranges::transform(v, out, f)` |
| No projection support | `std::ranges::sort(v, {}, &T::member)` |

---

## 8. Custom Views

You can implement your own view adaptor to plug into the range pipeline. Here is a simplified `stride_view` that takes every Nth element.

```cpp
#include <ranges>
#include <vector>
#include <iostream>
#include <iterator>

template<std::ranges::input_range R>
class stride_view : public std::ranges::view_interface<stride_view<R>> {
    R base_;
    std::size_t stride_;

public:
    struct iterator {
        using iterator_category = std::input_iterator_tag;
        using value_type = std::ranges::range_value_t<R>;
        using difference_type = std::ranges::range_difference_t<R>;

        std::ranges::iterator_t<R> current_;
        std::ranges::sentinel_t<R> end_;
        std::size_t stride_;

        iterator& operator++() {
            for (std::size_t i = 0; i < stride_ && current_ != end_; ++i) {
                ++current_;
            }
            return *this;
        }

        iterator operator++(int) {
            auto tmp = *this;
            ++(*this);
            return tmp;
        }

        decltype(auto) operator*() const { return *current_; }

        bool operator==(std::default_sentinel_t) const {
            return current_ == end_;
        }
    };

    stride_view() = default;
    stride_view(R base, std::size_t stride)
        : base_(std::move(base)), stride_(stride) {}

    auto begin() {
        return iterator{std::ranges::begin(base_),
                        std::ranges::end(base_), stride_};
    }

    auto end() { return std::default_sentinel; }
};

// Deduction guide
template<typename R>
stride_view(R&&, std::size_t) -> stride_view<std::views::all_t<R>>;

int main() {
    std::vector<int> v = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

    for (int n : stride_view(v, 3)) {
        std::cout << n << " ";  // 1 4 7 10
    }
    std::cout << "\n";

    return 0;
}
```

### Making It Pipe-able

```cpp
// Range adaptor closure object for pipe support
struct stride_adaptor {
    std::size_t stride;

    template<std::ranges::viewable_range R>
    auto operator()(R&& r) const {
        return stride_view(std::forward<R>(r), stride);
    }

    // Enable pipe syntax: range | stride(3)
    template<std::ranges::viewable_range R>
    friend auto operator|(R&& r, const stride_adaptor& a) {
        return a(std::forward<R>(r));
    }
};

auto stride(std::size_t n) { return stride_adaptor{n}; }

// Usage:
// auto result = v | stride(2) | std::views::transform(f);
```

---

## Exercises

### Exercise 1: FizzBuzz Pipeline

Using `views::iota` and `views::transform`, create a pipeline that generates FizzBuzz output for numbers 1 through 30. Filter out plain numbers and keep only the entries that are "Fizz", "Buzz", or "FizzBuzz".

### Exercise 2: Top-N by Field

Given a `std::vector<Student>` with fields `name`, `gpa`, and `year`, write a function that returns a view of the top N students by GPA using range adaptors and projections. Do not modify the original vector.

### Exercise 3: CSV Field Extraction

Given a `std::string` containing comma-separated values, use `views::split` and `views::transform` to extract the third field from each line. Handle the case where a line has fewer than three fields.

### Exercise 4: Infinite Sequence

Use `views::iota` to create an infinite sequence of natural numbers. Build a pipeline that filters for prime numbers and takes the first 20 primes. Print them.

### Exercise 5: Custom enumerate View

Implement an `enumerate_view` that pairs each element with its zero-based index (similar to Python's `enumerate`). It should work with the pipe operator: `vec | enumerate()`.

---

## Next Steps

The Ranges library handles synchronous, pull-based iteration. The next lesson explores C++20 Coroutines, which add cooperative, push-based control flow -- the foundation for generators and async tasks.

- [C++20 Coroutines](./10_CPP20_Coroutines.md)
