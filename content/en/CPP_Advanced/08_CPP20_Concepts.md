# C++20 Concepts

**Previous**: [Modern C++ (C++17)](./07_Modern_CPP_17.md) | **Next**: [C++20 Ranges](./09_CPP20_Ranges.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain why concepts were introduced and what problems they solve
2. Use standard concepts from the `<concepts>` header to constrain templates
3. Write `requires` clauses with simple, compound, type, and nested requirements
4. Define custom concepts using the `concept` keyword
5. Apply constrained `auto` for abbreviated function templates
6. Leverage concept-based overloading and subsumption rules

---

Concepts are arguably the most impactful C++20 feature for everyday template programming. Before concepts, a misuse of a template produced error messages that could span hundreds of lines, referencing internal implementation details. SFINAE-based constraints were powerful but cryptic. Concepts replace both problems with a single, readable mechanism: you declare what a type must be able to do, and the compiler enforces it with clear diagnostics. This lesson covers concepts from first principles through advanced patterns.

## 1. Why Concepts?

### The Problem: Template Error Messages

```cpp
#include <algorithm>
#include <list>

int main() {
    std::list<int> lst = {3, 1, 4, 1, 5};
    std::sort(lst.begin(), lst.end());
    // Without concepts: pages of incomprehensible errors about
    // RandomAccessIterator requirements buried deep in <algorithm>

    // With concepts (C++20): clear error message:
    // "error: std::list<int>::iterator does not satisfy random_access_iterator"

    return 0;
}
```

### The Problem: SFINAE Complexity

```cpp
// Pre-C++20: SFINAE to constrain templates
template<typename T,
         typename = std::enable_if_t<std::is_integral_v<T>>>
T doubleValue(T x) { return x * 2; }

// C++20: concepts make this readable
template<std::integral T>
T doubleValue(T x) { return x * 2; }
```

---

## 2. Using Standard Concepts

The `<concepts>` header provides a rich set of ready-to-use concepts.

### Core Language Concepts

```cpp
#include <concepts>
#include <iostream>

// same_as: types are identical
template<std::same_as<int> T>
void intOnly(T x) {
    std::cout << "int: " << x << "\n";
}

// derived_from: inheritance check
struct Base {};
struct Derived : Base {};

template<std::derived_from<Base> T>
void acceptBase(T& obj) {
    std::cout << "Derived from Base\n";
}

// convertible_to: implicit conversion exists
template<std::convertible_to<double> T>
double toDouble(T x) {
    return static_cast<double>(x);
}

int main() {
    intOnly(42);       // OK
    // intOnly(3.14);  // Error: double is not same_as<int>

    Derived d;
    acceptBase(d);     // OK

    std::cout << toDouble(42) << "\n";    // OK: int -> double
    std::cout << toDouble(3.14f) << "\n"; // OK: float -> double

    return 0;
}
```

### Arithmetic Concepts

```cpp
#include <concepts>
#include <iostream>

template<std::integral T>
T bitwiseOr(T a, T b) {
    return a | b;  // Bitwise operations only make sense for integers
}

template<std::floating_point T>
T interpolate(T a, T b, T t) {
    return a + t * (b - a);
}

template<std::signed_integral T>
T absolute(T x) {
    return x < 0 ? -x : x;
}

template<std::unsigned_integral T>
T safeDivide(T a, T b) {
    return b == 0 ? 0 : a / b;
}

int main() {
    std::cout << bitwiseOr(0b1010, 0b0101) << "\n";  // 15
    std::cout << interpolate(0.0, 10.0, 0.5) << "\n"; // 5.0
    std::cout << absolute(-42) << "\n";                // 42
    std::cout << safeDivide(10u, 3u) << "\n";          // 3

    return 0;
}
```

### Comparison Concepts

```cpp
#include <concepts>
#include <iostream>
#include <string>

template<std::equality_comparable T>
bool areSame(const T& a, const T& b) {
    return a == b;
}

template<std::totally_ordered T>
const T& clamp(const T& val, const T& lo, const T& hi) {
    return val < lo ? lo : val > hi ? hi : val;
}

int main() {
    std::cout << std::boolalpha;
    std::cout << areSame(42, 42) << "\n";             // true
    std::cout << areSame(std::string("a"), std::string("b")) << "\n"; // false

    std::cout << clamp(15, 0, 10) << "\n";  // 10
    std::cout << clamp(5, 0, 10) << "\n";   // 5

    return 0;
}
```

### Callable Concepts

```cpp
#include <concepts>
#include <iostream>
#include <functional>

// invocable: F can be called with Args
template<std::invocable<int, int> F>
int apply(F func, int a, int b) {
    return func(a, b);
}

// predicate: F(Args...) returns bool-like
template<typename T, std::predicate<T> F>
int countMatching(const std::vector<T>& vec, F pred) {
    int count = 0;
    for (const auto& elem : vec) {
        if (pred(elem)) ++count;
    }
    return count;
}

int main() {
    std::cout << apply([](int a, int b) { return a + b; }, 3, 4) << "\n";
    std::cout << apply(std::plus<>{}, 3, 4) << "\n";

    std::vector<int> v = {1, 2, 3, 4, 5, 6};
    auto isEven = [](int x) { return x % 2 == 0; };
    std::cout << "Even count: " << countMatching(v, isEven) << "\n";

    return 0;
}
```

### Complete Standard Concepts Reference

| Concept | Header | Description |
|---------|--------|-------------|
| `same_as<T, U>` | `<concepts>` | T and U are the same type |
| `derived_from<D, B>` | `<concepts>` | D derives from B |
| `convertible_to<From, To>` | `<concepts>` | Implicit conversion exists |
| `integral<T>` | `<concepts>` | Integer type (int, long, etc.) |
| `signed_integral<T>` | `<concepts>` | Signed integer |
| `unsigned_integral<T>` | `<concepts>` | Unsigned integer |
| `floating_point<T>` | `<concepts>` | float, double, long double |
| `equality_comparable<T>` | `<concepts>` | Supports `==` and `!=` |
| `totally_ordered<T>` | `<concepts>` | Supports `<`, `>`, `<=`, `>=` |
| `movable<T>` | `<concepts>` | Move constructible and assignable |
| `copyable<T>` | `<concepts>` | Copy constructible and assignable |
| `semiregular<T>` | `<concepts>` | Copyable + default constructible |
| `regular<T>` | `<concepts>` | Semiregular + equality comparable |
| `invocable<F, Args...>` | `<concepts>` | F(Args...) is valid |
| `predicate<F, Args...>` | `<concepts>` | F(Args...) returns bool |

---

## 3. requires Clause

A `requires` clause specifies constraints on template parameters. There are several syntactic forms.

### Simple Constraints

```cpp
#include <concepts>
#include <iostream>

// Trailing requires clause
template<typename T>
T add(T a, T b) requires std::integral<T> {
    return a + b;
}

// requires before body (same meaning, different position)
template<typename T>
    requires std::integral<T>
T multiply(T a, T b) {
    return a * b;
}

// Conjunction (AND)
template<typename T>
    requires std::integral<T> && std::signed_integral<T>
T negate(T x) {
    return -x;
}

// Disjunction (OR)
template<typename T>
    requires std::integral<T> || std::floating_point<T>
T square(T x) {
    return x * x;
}

int main() {
    std::cout << add(1, 2) << "\n";       // OK
    // add(1.0, 2.0);                      // Error: not integral

    std::cout << square(3) << "\n";        // OK: integral
    std::cout << square(3.14) << "\n";     // OK: floating_point

    return 0;
}
```

---

## 4. requires Expression

A `requires` expression is a compile-time predicate that checks whether certain expressions are valid.

### Simple Requirements

```cpp
#include <iostream>
#include <concepts>

// Check that expressions are valid
template<typename T>
concept Addable = requires(T a, T b) {
    a + b;      // Expression must be valid
    a - b;      // This too
    a += b;     // And this
};

template<Addable T>
T combine(T a, T b) {
    return a + b;
}

int main() {
    std::cout << combine(1, 2) << "\n";      // OK
    std::cout << combine(1.5, 2.5) << "\n";  // OK

    return 0;
}
```

### Type Requirements

```cpp
#include <concepts>
#include <vector>
#include <iostream>

// Check that associated types exist
template<typename T>
concept HasValueType = requires {
    typename T::value_type;     // Must have value_type
    typename T::iterator;       // Must have iterator
    typename T::size_type;      // Must have size_type
};

template<HasValueType C>
void printInfo(const C& container) {
    std::cout << "Container with " << container.size() << " elements\n";
}

int main() {
    std::vector<int> v = {1, 2, 3};
    printInfo(v);  // OK: vector has value_type, iterator, size_type

    // printInfo(42);  // Error: int doesn't have these types

    return 0;
}
```

### Compound Requirements

```cpp
#include <concepts>
#include <string>
#include <iostream>

// Check expression validity AND return type
template<typename T>
concept Hashable = requires(T t) {
    { std::hash<T>{}(t) } -> std::convertible_to<std::size_t>;
};

// StringLike: must have specific methods with specific return types
template<typename T>
concept StringLike = requires(T t, std::size_t i) {
    { t.length() } -> std::convertible_to<std::size_t>;
    { t[i] } -> std::convertible_to<char>;
    { t.substr(i, i) } -> std::same_as<T>;
};

template<typename T>
concept Printable = requires(std::ostream& os, T t) {
    { os << t } -> std::same_as<std::ostream&>;
};

template<Printable T>
void println(const T& value) {
    std::cout << value << "\n";
}

int main() {
    println(42);            // OK
    println("Hello");       // OK
    println(3.14);          // OK

    static_assert(Hashable<int>);
    static_assert(Hashable<std::string>);
    static_assert(StringLike<std::string>);

    return 0;
}
```

### Nested Requirements

```cpp
#include <concepts>
#include <vector>

// Nested requirement: use requires inside requires
template<typename T>
concept Container = requires(T t) {
    typename T::value_type;
    { t.begin() } -> std::input_or_output_iterator;
    { t.end() } -> std::input_or_output_iterator;
    { t.size() } -> std::convertible_to<std::size_t>;
    // Nested: value_type must be equality_comparable
    requires std::equality_comparable<typename T::value_type>;
};

// SortableContainer: values must be ordered
template<typename T>
concept SortableContainer = Container<T> &&
    requires {
        requires std::totally_ordered<typename T::value_type>;
    };

template<SortableContainer C>
void sortContainer(C& c) {
    std::sort(c.begin(), c.end());
}

int main() {
    std::vector<int> v = {3, 1, 4};
    sortContainer(v);  // OK

    return 0;
}
```

---

## 5. Defining Custom Concepts

### Basic Custom Concepts

```cpp
#include <concepts>
#include <iostream>
#include <string>
#include <cmath>

// Arithmetic concept (integers or floats)
template<typename T>
concept Arithmetic = std::is_arithmetic_v<T>;

// Numeric with specific operations
template<typename T>
concept Number = requires(T a, T b) {
    { a + b } -> std::convertible_to<T>;
    { a - b } -> std::convertible_to<T>;
    { a * b } -> std::convertible_to<T>;
    { a / b } -> std::convertible_to<T>;
};

// Composing concepts
template<typename T>
concept OrderedNumber = Number<T> && std::totally_ordered<T>;

// Iterator concept
template<typename I>
concept ForwardIterable = requires(I it) {
    { *it };         // Dereferenceable
    { ++it } -> std::same_as<I&>;  // Incrementable
    { it != it } -> std::convertible_to<bool>;  // Comparable
};

// Practical: serializable concept
template<typename T>
concept Serializable = requires(T t, std::ostream& os, std::istream& is) {
    { os << t } -> std::same_as<std::ostream&>;
    { is >> t } -> std::same_as<std::istream&>;
};

template<OrderedNumber T>
T median(T a, T b, T c) {
    if (a > b) std::swap(a, b);
    if (b > c) std::swap(b, c);
    if (a > b) std::swap(a, b);
    return b;
}

int main() {
    std::cout << median(3, 1, 2) << "\n";      // 2
    std::cout << median(3.0, 1.0, 2.0) << "\n"; // 2.0

    static_assert(Number<int>);
    static_assert(Number<double>);
    static_assert(!Number<std::string>);  // string doesn't have /

    return 0;
}
```

---

## 6. Constrained auto

The `auto` keyword can be constrained with a concept, creating abbreviated function templates.

```cpp
#include <concepts>
#include <iostream>
#include <string>

// Constrained auto parameters (abbreviated template)
void printNumber(std::integral auto n) {
    std::cout << "Integer: " << n << "\n";
}

void printNumber(std::floating_point auto n) {
    std::cout << "Float: " << n << "\n";
}

// Each auto is independently constrained
auto multiply(std::integral auto a, std::floating_point auto b) {
    return a * b;
}

// Constrained auto return type
std::integral auto computeSize(int width, int height) {
    return width * height;
}

// Constrained auto in variable declarations
void example() {
    std::integral auto x = 42;        // OK: int is integral
    // std::integral auto y = 3.14;   // Error: double is not integral
}

// With custom concepts
template<typename T>
concept Printable = requires(std::ostream& os, T t) {
    { os << t } -> std::same_as<std::ostream&>;
};

void display(Printable auto const& value) {
    std::cout << value << "\n";
}

int main() {
    printNumber(42);      // Integer: 42
    printNumber(3.14);    // Float: 3.14

    std::cout << multiply(3, 2.5) << "\n";  // 7.5
    std::cout << computeSize(10, 20) << "\n";  // 200

    display(42);
    display("Hello");
    display(std::string("World"));

    return 0;
}
```

---

## 7. Concept-Based Overloading

When multiple overloads satisfy a call, the compiler uses **subsumption** to select the most constrained overload.

```cpp
#include <concepts>
#include <iostream>
#include <string>

// Unconstrained (least specific)
template<typename T>
void describe(const T& value) {
    std::cout << "Unknown type\n";
}

// integral constraint
template<std::integral T>
void describe(const T& value) {
    std::cout << "Integral: " << value << "\n";
}

// signed_integral subsumes integral (more specific)
template<std::signed_integral T>
void describe(const T& value) {
    std::cout << "Signed integral: " << value << "\n";
}

// Custom concept hierarchy
template<typename T>
concept Animal = requires(T t) {
    { t.name() } -> std::convertible_to<std::string>;
};

template<typename T>
concept Pet = Animal<T> && requires(T t) {
    { t.owner() } -> std::convertible_to<std::string>;
};

// Pet subsumes Animal (more constrained)
template<Animal T>
void greet(const T& a) {
    std::cout << "Hello, animal " << a.name() << "\n";
}

template<Pet T>
void greet(const T& p) {
    std::cout << "Hello, " << p.name() << " (owned by " << p.owner() << ")\n";
}

int main() {
    describe("hello");     // Unknown type (const char*)
    describe(42u);         // Integral (unsigned int)
    describe(42);          // Signed integral (int) -- most constrained wins

    // Subsumption: signed_integral => integral => unconstrained
    // Compiler picks the most specific match

    return 0;
}
```

### Subsumption Rules

```
More constrained (preferred)
         |
  signed_integral<T>      -- implies integral<T>
         |
     integral<T>           -- implies basic type check
         |
   (unconstrained)
         |
Less constrained (fallback)
```

The compiler prefers the **most constrained** viable overload. A concept C1 **subsumes** C2 if C1's constraints logically imply C2's constraints.

---

## 8. Practical Patterns

### Constraining Container Types

```cpp
#include <concepts>
#include <ranges>
#include <vector>
#include <list>
#include <iostream>

template<typename C>
concept RandomAccessContainer = requires(C c, typename C::size_type i) {
    typename C::value_type;
    { c[i] } -> std::same_as<typename C::reference>;
    { c.size() } -> std::convertible_to<std::size_t>;
    requires std::random_access_iterator<typename C::iterator>;
};

template<RandomAccessContainer C>
auto binarySearch(const C& container, const typename C::value_type& target)
    -> std::optional<typename C::size_type> {
    typename C::size_type lo = 0;
    typename C::size_type hi = container.size();
    while (lo < hi) {
        auto mid = lo + (hi - lo) / 2;
        if (container[mid] == target) return mid;
        if (container[mid] < target) lo = mid + 1;
        else hi = mid;
    }
    return std::nullopt;
}

int main() {
    std::vector<int> v = {1, 3, 5, 7, 9};
    if (auto idx = binarySearch(v, 5)) {
        std::cout << "Found at index " << *idx << "\n";
    }

    // std::list<int> l = {1, 2, 3};
    // binarySearch(l, 2);  // Error: list is not RandomAccessContainer

    return 0;
}
```

### Arithmetic Concept with Operations

```cpp
#include <concepts>
#include <iostream>

template<typename T>
concept ArithmeticLike = requires(T a, T b) {
    { a + b } -> std::same_as<T>;
    { a - b } -> std::same_as<T>;
    { a * b } -> std::same_as<T>;
    { -a } -> std::same_as<T>;
} && std::totally_ordered<T> && std::regular<T>;

template<ArithmeticLike T>
class Matrix2x2 {
    T data_[2][2];

public:
    Matrix2x2(T a, T b, T c, T d)
        : data_{{a, b}, {c, d}} {}

    Matrix2x2 operator+(const Matrix2x2& other) const {
        return {
            data_[0][0] + other.data_[0][0],
            data_[0][1] + other.data_[0][1],
            data_[1][0] + other.data_[1][0],
            data_[1][1] + other.data_[1][1]
        };
    }

    T determinant() const {
        return data_[0][0] * data_[1][1] - data_[0][1] * data_[1][0];
    }

    void print() const {
        std::cout << "[" << data_[0][0] << " " << data_[0][1] << "]\n"
                  << "[" << data_[1][0] << " " << data_[1][1] << "]\n";
    }
};

int main() {
    Matrix2x2<int> m1(1, 2, 3, 4);
    Matrix2x2<double> m2(1.0, 0.0, 0.0, 1.0);

    m1.print();
    std::cout << "det = " << m1.determinant() << "\n";

    auto m3 = m2 + Matrix2x2<double>(0.5, 0.5, 0.5, 0.5);
    m3.print();

    return 0;
}
```

### Constraining with Multiple Concepts

```cpp
#include <concepts>
#include <iostream>
#include <string>

// Combine multiple requirements
template<typename T>
concept Displayable =
    std::copyable<T> &&
    requires(std::ostream& os, const T& t) {
        { os << t } -> std::same_as<std::ostream&>;
    };

template<typename T>
concept Parseable =
    std::default_initializable<T> &&
    requires(std::istream& is, T& t) {
        { is >> t } -> std::same_as<std::istream&>;
    };

template<typename T>
concept Serializable = Displayable<T> && Parseable<T>;

template<Serializable T>
class ConfigValue {
    std::string key_;
    T value_;

public:
    ConfigValue(std::string key, T value)
        : key_(std::move(key)), value_(std::move(value)) {}

    friend std::ostream& operator<<(std::ostream& os, const ConfigValue& cv) {
        return os << cv.key_ << "=" << cv.value_;
    }
};

int main() {
    ConfigValue<int> port("port", 8080);
    ConfigValue<std::string> host("host", std::string("localhost"));

    std::cout << port << "\n";
    std::cout << host << "\n";

    return 0;
}
```

---

## Summary

| Feature | Syntax | Description |
|---------|--------|-------------|
| Standard concept | `std::integral<T>` | Pre-defined type constraint |
| requires clause | `requires C<T>` | Constrain template parameter |
| requires expression | `requires(T t) { ... }` | Check expression validity |
| Custom concept | `concept C = ...` | Define reusable constraint |
| Constrained auto | `std::integral auto x` | Abbreviated template syntax |
| Subsumption | More constrained wins | Overload resolution rule |

### Concept Syntax Cheat Sheet

```cpp
// Four ways to apply concepts:
template<std::integral T>             // 1. Concept as template parameter
void f1(T x);

template<typename T>
    requires std::integral<T>         // 2. requires clause
void f2(T x);

template<typename T>
void f3(T x) requires std::integral<T>;  // 3. Trailing requires

void f4(std::integral auto x);       // 4. Constrained auto (abbreviated)
```

---

## Exercises

### Exercise 1: Printable Concept

Define a `Printable` concept that checks if a type supports `operator<<` to `std::ostream`. Write a `print` function that accepts only `Printable` types.

### Exercise 2: Numeric Tower

Create a concept hierarchy: `Number` -> `Integral` -> `SignedIntegral`, where each level adds more constraints. Write overloaded `describe()` functions that demonstrate subsumption.

### Exercise 3: Container Concept

Define a `Container` concept that requires `begin()`, `end()`, `size()`, `value_type`, and `iterator`. Then define a `SortableContainer` that additionally requires `random_access_iterator` and `totally_ordered` values.

### Exercise 4: Constrained Generic Algorithm

Write a `myAccumulate` function that requires the value type to satisfy a custom `Addable` concept. Test with integers, doubles, and `std::string`.

### Exercise 5: Concept-Based Dispatch

Create a `serialize` function that uses concept overloading to handle:
- Arithmetic types (convert to string)
- String-like types (quote and escape)
- Container types (serialize each element recursively)
- Everything else (static_assert failure)

---

## Next Steps

C++20 Ranges provide a composable, lazy pipeline for data processing. Let's explore views, adaptors, and range algorithms in [09_CPP20_Ranges.md](./09_CPP20_Ranges.md).
