# STL Algorithms and Iterators

**Previous**: [STL Containers](./10_STL_Containers.md) | **Next**: [Templates](./12_Templates.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Classify the five iterator categories and identify which containers provide each kind
2. Write lambda expressions with various capture modes (by value, by reference, mixed)
3. Apply STL search algorithms (`find`, `find_if`, `binary_search`) to locate elements in containers
4. Use sorting algorithms (`sort`, `partial_sort`, `nth_element`) with custom comparators
5. Combine modifying algorithms (`copy`, `transform`, `remove`/`erase`) to reshape container data
6. Perform numeric reductions with `accumulate` and generate sequences with `iota`
7. Execute set operations (`set_union`, `set_intersection`, `set_difference`) on sorted ranges

---

The STL algorithms library turns C++ from a language where you hand-roll every loop into one where common data operations are a single function call. Rather than reinventing searching, sorting, and transforming logic for every project, you compose well-tested, highly optimized building blocks that the standard library already provides. Mastering iterators and algorithms is what separates code that merely compiles from code that is concise, correct, and performant.

## 1. Iterator

Iterators are pointer-like objects that point to container elements.

### Iterator Types

| Type | Description | Example Container |
|------|-------------|------------------|
| Input Iterator | Read-only, one direction | istream_iterator |
| Output Iterator | Write-only, one direction | ostream_iterator |
| Forward Iterator | Read/write, one direction | forward_list |
| Bidirectional Iterator | Read/write, both directions | list, set, map |
| Random Access Iterator | All operations, random access | vector, deque, array |

### Basic Iterator Usage

```cpp
#include <iostream>
#include <vector>

int main() {
    std::vector<int> v = {1, 2, 3, 4, 5};

    // begin(), end()
    std::vector<int>::iterator it = v.begin();
    std::cout << *it << std::endl;  // 1

    ++it;
    std::cout << *it << std::endl;  // 2

    // Iteration
    for (auto it = v.begin(); it != v.end(); ++it) {
        std::cout << *it << " ";
    }
    std::cout << std::endl;

    return 0;
}
```

### const Iterator

```cpp
#include <iostream>
#include <vector>

int main() {
    std::vector<int> v = {1, 2, 3, 4, 5};

    // const_iterator: read-only
    for (std::vector<int>::const_iterator it = v.cbegin();
         it != v.cend(); ++it) {
        std::cout << *it << " ";
        // *it = 10;  // Error! Cannot modify
    }

    return 0;
}
```

### Reverse Iterator

```cpp
#include <iostream>
#include <vector>

int main() {
    std::vector<int> v = {1, 2, 3, 4, 5};

    // rbegin(), rend()
    for (auto it = v.rbegin(); it != v.rend(); ++it) {
        std::cout << *it << " ";
    }
    std::cout << std::endl;  // 5 4 3 2 1

    return 0;
}
```

---

## 2. Lambda Expressions

Defines anonymous functions concisely.

### Basic Syntax

```cpp
[capture](parameters) -> return_type { body }
```

### Examples

```cpp
#include <iostream>
#include <vector>
#include <algorithm>

int main() {
    // Basic lambda
    auto add = [](int a, int b) {
        return a + b;
    };
    std::cout << add(3, 5) << std::endl;  // 8

    // Explicit return type
    auto divide = [](double a, double b) -> double {
        return a / b;
    };

    // With algorithms
    std::vector<int> v = {3, 1, 4, 1, 5, 9};
    std::sort(v.begin(), v.end(), [](int a, int b) {
        return a > b;  // Descending order
    });

    return 0;
}
```

### Capture

```cpp
#include <iostream>

int main() {
    int x = 10;
    int y = 20;

    // Capture by value (copy)
    auto f1 = [x]() { return x; };

    // Capture by reference
    auto f2 = [&x]() { x++; };

    // Capture all by value
    auto f3 = [=]() { return x + y; };

    // Capture all by reference
    auto f4 = [&]() { x++; y++; };

    // Mixed
    auto f5 = [=, &x]() {  // y by value, x by reference
        x++;
        return y;
    };

    f2();
    std::cout << x << std::endl;  // 11

    return 0;
}
```

### mutable Lambda

```cpp
#include <iostream>

int main() {
    int x = 10;

    // Value capture is const by default
    auto f = [x]() mutable {  // mutable allows modification
        x++;
        return x;
    };

    std::cout << f() << std::endl;  // 11
    std::cout << x << std::endl;    // 10 (original unchanged)

    return 0;
}
```

---

## 3. Basic Algorithms

Include the `<algorithm>` header.

### for_each

```cpp
#include <iostream>
#include <vector>
#include <algorithm>

int main() {
    std::vector<int> v = {1, 2, 3, 4, 5};

    std::for_each(v.begin(), v.end(), [](int n) {
        std::cout << n * 2 << " ";
    });
    std::cout << std::endl;  // 2 4 6 8 10

    return 0;
}
```

### transform

```cpp
#include <iostream>
#include <vector>
#include <algorithm>

int main() {
    std::vector<int> v = {1, 2, 3, 4, 5};
    std::vector<int> result(v.size());

    // Transform each element
    std::transform(v.begin(), v.end(), result.begin(),
                   [](int n) { return n * n; });

    for (int n : result) {
        std::cout << n << " ";
    }
    std::cout << std::endl;  // 1 4 9 16 25

    return 0;
}
```

---

## 4. Search Algorithms

### find

```cpp
#include <iostream>
#include <vector>
#include <algorithm>

int main() {
    std::vector<int> v = {1, 2, 3, 4, 5};

    auto it = std::find(v.begin(), v.end(), 3);
    if (it != v.end()) {
        std::cout << "Found: " << *it << std::endl;
        std::cout << "Index: " << std::distance(v.begin(), it) << std::endl;
    }

    return 0;
}
```

### find_if

```cpp
#include <iostream>
#include <vector>
#include <algorithm>

int main() {
    std::vector<int> v = {1, 2, 3, 4, 5};

    // First element satisfying condition
    auto it = std::find_if(v.begin(), v.end(),
                           [](int n) { return n > 3; });

    if (it != v.end()) {
        std::cout << "First > 3: " << *it << std::endl;  // 4
    }

    return 0;
}
```

### count / count_if

```cpp
#include <iostream>
#include <vector>
#include <algorithm>

int main() {
    std::vector<int> v = {1, 2, 2, 3, 2, 4, 5};

    // Count specific value
    int c1 = std::count(v.begin(), v.end(), 2);
    std::cout << "Count of 2: " << c1 << std::endl;  // 3

    // Count satisfying condition
    int c2 = std::count_if(v.begin(), v.end(),
                           [](int n) { return n % 2 == 0; });
    std::cout << "Even count: " << c2 << std::endl;  // 4

    return 0;
}
```

### binary_search

Use only on sorted ranges.

```cpp
#include <iostream>
#include <vector>
#include <algorithm>

int main() {
    std::vector<int> v = {1, 2, 3, 4, 5};  // Sorted

    bool found = std::binary_search(v.begin(), v.end(), 3);
    std::cout << "3 found: " << found << std::endl;  // 1

    return 0;
}
```

---

## 5. Sorting Algorithms

### sort

```cpp
#include <iostream>
#include <vector>
#include <algorithm>

int main() {
    std::vector<int> v = {3, 1, 4, 1, 5, 9, 2, 6};

    // Ascending (default)
    std::sort(v.begin(), v.end());

    // Descending
    std::sort(v.begin(), v.end(), std::greater<int>());

    // Custom comparison
    std::sort(v.begin(), v.end(), [](int a, int b) {
        return a > b;  // Descending
    });

    return 0;
}
```

### partial_sort

```cpp
#include <iostream>
#include <vector>
#include <algorithm>

int main() {
    std::vector<int> v = {3, 1, 4, 1, 5, 9, 2, 6};

    // Sort only top 3
    std::partial_sort(v.begin(), v.begin() + 3, v.end());

    for (int n : v) {
        std::cout << n << " ";
    }
    // 1 1 2 ... (only first 3 sorted)

    return 0;
}
```

### nth_element

```cpp
#include <iostream>
#include <vector>
#include <algorithm>

int main() {
    std::vector<int> v = {3, 1, 4, 1, 5, 9, 2, 6};

    // Place 3rd element in its sorted position
    std::nth_element(v.begin(), v.begin() + 3, v.end());

    std::cout << "3rd element: " << v[3] << std::endl;

    return 0;
}
```

---

## 6. Modifying Algorithms

### copy

```cpp
#include <iostream>
#include <vector>
#include <algorithm>

int main() {
    std::vector<int> src = {1, 2, 3, 4, 5};
    std::vector<int> dest(5);

    std::copy(src.begin(), src.end(), dest.begin());

    return 0;
}
```

### fill

```cpp
#include <iostream>
#include <vector>
#include <algorithm>

int main() {
    std::vector<int> v(5);

    std::fill(v.begin(), v.end(), 42);

    for (int n : v) {
        std::cout << n << " ";
    }
    std::cout << std::endl;  // 42 42 42 42 42

    return 0;
}
```

### replace

```cpp
#include <iostream>
#include <vector>
#include <algorithm>

int main() {
    std::vector<int> v = {1, 2, 3, 2, 4, 2, 5};

    // Replace 2 with 100
    std::replace(v.begin(), v.end(), 2, 100);

    for (int n : v) {
        std::cout << n << " ";
    }
    std::cout << std::endl;  // 1 100 3 100 4 100 5

    return 0;
}
```

### remove / erase

```cpp
#include <iostream>
#include <vector>
#include <algorithm>

int main() {
    std::vector<int> v = {1, 2, 3, 2, 4, 2, 5};

    // remove doesn't actually delete
    auto newEnd = std::remove(v.begin(), v.end(), 2);

    // Use with erase (erase-remove idiom)
    v.erase(newEnd, v.end());

    for (int n : v) {
        std::cout << n << " ";
    }
    std::cout << std::endl;  // 1 3 4 5

    return 0;
}
```

### reverse

```cpp
#include <iostream>
#include <vector>
#include <algorithm>

int main() {
    std::vector<int> v = {1, 2, 3, 4, 5};

    std::reverse(v.begin(), v.end());

    for (int n : v) {
        std::cout << n << " ";
    }
    std::cout << std::endl;  // 5 4 3 2 1

    return 0;
}
```

### unique

```cpp
#include <iostream>
#include <vector>
#include <algorithm>

int main() {
    std::vector<int> v = {1, 1, 2, 2, 2, 3, 3, 4};

    // Remove consecutive duplicates (requires sorted)
    auto newEnd = std::unique(v.begin(), v.end());
    v.erase(newEnd, v.end());

    for (int n : v) {
        std::cout << n << " ";
    }
    std::cout << std::endl;  // 1 2 3 4

    return 0;
}
```

---

## 7. Numeric Algorithms

Include the `<numeric>` header.

### accumulate

```cpp
#include <iostream>
#include <vector>
#include <numeric>

int main() {
    std::vector<int> v = {1, 2, 3, 4, 5};

    // Sum
    int sum = std::accumulate(v.begin(), v.end(), 0);
    std::cout << "Sum: " << sum << std::endl;  // 15

    // Product
    int product = std::accumulate(v.begin(), v.end(), 1,
                                  std::multiplies<int>());
    std::cout << "Product: " << product << std::endl;  // 120

    // Custom
    int sumSquares = std::accumulate(v.begin(), v.end(), 0,
        [](int acc, int n) { return acc + n * n; });
    std::cout << "Sum of squares: " << sumSquares << std::endl;  // 55

    return 0;
}
```

### iota

```cpp
#include <iostream>
#include <vector>
#include <numeric>

int main() {
    std::vector<int> v(10);

    // Fill with consecutive values
    std::iota(v.begin(), v.end(), 1);

    for (int n : v) {
        std::cout << n << " ";
    }
    std::cout << std::endl;  // 1 2 3 4 5 6 7 8 9 10

    return 0;
}
```

---

## 8. Set Algorithms

Works only on sorted ranges.

```cpp
#include <iostream>
#include <vector>
#include <algorithm>

int main() {
    std::vector<int> a = {1, 2, 3, 4, 5};
    std::vector<int> b = {3, 4, 5, 6, 7};
    std::vector<int> result;

    // Union
    std::set_union(a.begin(), a.end(), b.begin(), b.end(),
                   std::back_inserter(result));
    // result: 1 2 3 4 5 6 7

    result.clear();

    // Intersection
    std::set_intersection(a.begin(), a.end(), b.begin(), b.end(),
                          std::back_inserter(result));
    // result: 3 4 5

    result.clear();

    // Difference
    std::set_difference(a.begin(), a.end(), b.begin(), b.end(),
                        std::back_inserter(result));
    // result: 1 2

    return 0;
}
```

---

## 9. min/max Algorithms

```cpp
#include <iostream>
#include <vector>
#include <algorithm>

int main() {
    std::vector<int> v = {3, 1, 4, 1, 5, 9, 2, 6};

    // Min/max element
    auto minIt = std::min_element(v.begin(), v.end());
    auto maxIt = std::max_element(v.begin(), v.end());

    std::cout << "Min: " << *minIt << std::endl;
    std::cout << "Max: " << *maxIt << std::endl;

    // Both
    auto [minEl, maxEl] = std::minmax_element(v.begin(), v.end());
    std::cout << *minEl << " ~ " << *maxEl << std::endl;

    // Value comparison
    std::cout << std::min(3, 5) << std::endl;  // 3
    std::cout << std::max(3, 5) << std::endl;  // 5

    return 0;
}
```

---

## 10. all_of / any_of / none_of

```cpp
#include <iostream>
#include <vector>
#include <algorithm>

int main() {
    std::vector<int> v = {2, 4, 6, 8, 10};

    // All satisfy?
    bool all = std::all_of(v.begin(), v.end(),
                           [](int n) { return n % 2 == 0; });
    std::cout << "All even: " << all << std::endl;  // 1

    // Any satisfy?
    bool any = std::any_of(v.begin(), v.end(),
                           [](int n) { return n > 5; });
    std::cout << "Any > 5: " << any << std::endl;  // 1

    // None satisfy?
    bool none = std::none_of(v.begin(), v.end(),
                             [](int n) { return n < 0; });
    std::cout << "No negatives: " << none << std::endl;  // 1

    return 0;
}
```

---

## 11. Summary

| Algorithm | Purpose |
|-----------|---------|
| `find`, `find_if` | Search |
| `count`, `count_if` | Count |
| `sort`, `partial_sort` | Sort |
| `binary_search` | Binary search |
| `transform` | Transform |
| `for_each` | Apply function to each element |
| `copy`, `fill`, `replace` | Modify |
| `remove`, `unique` | Remove |
| `reverse` | Reverse |
| `accumulate` | Accumulate |
| `min_element`, `max_element` | Min/max |

---

## Exercises

### Exercise 1: Lambda Capture Modes

Write a program that demonstrates all four lambda capture modes. Create two local variables `int base = 10` and `int multiplier = 3`. Write four lambdas:
- Captures `base` by value and returns `base + n` for a given `n`.
- Captures `multiplier` by reference and doubles it inside the lambda (verify the original changed).
- Captures everything by value (`[=]`) and computes `base * multiplier + n`.
- Captures everything by reference (`[&]`) and increments both variables.

Print the variables after each lambda call and explain which captures can observe the change.

### Exercise 2: Pipeline with transform and accumulate

Given a `std::vector<std::string> words = {"hello", "world", "cpp", "algorithms"}`:

1. Use `std::transform` to create a new vector where each string is replaced by its length (`words.size()`).
2. Use `std::accumulate` with a lambda to compute the total character count across all words.
3. Use `std::find_if` to locate the first word longer than 5 characters.
4. Use `std::count_if` to count how many words have an even number of characters.

Write all steps without raw loops — only STL algorithms and lambdas.

### Exercise 3: Sort with Custom Comparator

Create a `struct Person { std::string name; int age; };` and a `std::vector<Person>`. Populate it with at least five people. Then:
1. Sort by age ascending.
2. Sort by name alphabetically (case-insensitive if you can).
3. Sort by name length, then by name alphabetically as a tie-breaker.

Use lambdas as comparators for `std::sort` in each step and print the vector after each sort.

### Exercise 4: Erase-Remove Idiom

Start with `std::vector<int> v = {1, 5, 2, 8, 3, 7, 4, 6, 9, 10}`.

1. Remove all even numbers using the erase-remove idiom (`std::remove_if` + `v.erase`).
2. From the remaining odd numbers, keep only those greater than 3 (apply the idiom again).
3. Confirm the final vector contains exactly `{5, 7, 9}`.

Explain in a comment why `std::remove_if` alone is not sufficient to shrink the vector.

### Exercise 5: Set Operations on Sorted Ranges

Create two sorted `std::vector<int>` collections representing the members of two clubs:
- Club A: `{1, 3, 5, 7, 9, 11}`
- Club B: `{3, 6, 9, 12, 15}`

Use STL set algorithms to compute and print:
1. All members who belong to either club (union).
2. Members who belong to both clubs (intersection).
3. Members who are in Club A but not Club B (difference).
4. Members who are in exactly one club but not both (symmetric difference — use `std::set_symmetric_difference`).

---

## Next Steps

Let's learn about templates in [12_Templates.md](./12_Templates.md)!
