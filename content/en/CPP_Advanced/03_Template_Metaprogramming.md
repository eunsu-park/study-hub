# Template Metaprogramming

**Previous**: [Templates](./02_Templates.md) | **Next**: [Smart Pointers and RAII](./04_Smart_Pointers_and_RAII.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Write variadic templates using parameter packs and fold expressions
2. Apply SFINAE and `std::enable_if` for compile-time overload resolution
3. Use type traits from `<type_traits>` to query and transform types
4. Implement compile-time computation with `constexpr` and `consteval`
5. Design template policies and tag dispatch patterns

---

Template metaprogramming (TMP) is the art of using the C++ template system to perform computation at compile time. What began as an accidental discovery--that C++ templates form a Turing-complete language--has evolved into a disciplined set of techniques for writing code that is both more generic and more efficient. Modern C++ (C++17 and C++20) has tamed much of TMP's historical complexity with `if constexpr`, fold expressions, and concepts, but understanding the underlying mechanisms remains essential for library design and performance-critical code.

## 1. Variadic Templates

Variadic templates accept an arbitrary number of template arguments using **parameter packs**.

### Parameter Packs and Expansion

```cpp
#include <iostream>
#include <string>

// sizeof... returns the number of elements in a pack
template<typename... Args>
void countArgs(Args... args) {
    std::cout << "Type pack size: " << sizeof...(Args) << "\n";
    std::cout << "Value pack size: " << sizeof...(args) << "\n";
}

// Recursive unpacking (pre-C++17 style)
// Base case
void printAll() {
    std::cout << "\n";
}

// Recursive case
template<typename T, typename... Rest>
void printAll(T first, Rest... rest) {
    std::cout << first;
    if constexpr (sizeof...(rest) > 0) {
        std::cout << ", ";
    }
    printAll(rest...);
}

// Pack expansion in different contexts
template<typename... Ts>
void expandDemo(Ts... args) {
    // Expansion in function call
    // Each element of args is passed through a function
    auto dummy = {(std::cout << args << " ", 0)...};
    (void)dummy;
    std::cout << "\n";
}

int main() {
    countArgs(1, 2.0, "three", 'f');  // 4, 4
    printAll(1, "hello", 3.14, true); // 1, hello, 3.14, 1
    expandDemo(10, 20, 30);           // 10 20 30

    return 0;
}
```

### Compile-Time Index Sequences

```cpp
#include <iostream>
#include <tuple>
#include <utility>

// Print all elements of a tuple
template<typename Tuple, std::size_t... Is>
void printTupleImpl(const Tuple& t, std::index_sequence<Is...>) {
    ((std::cout << (Is == 0 ? "" : ", ") << std::get<Is>(t)), ...);
    std::cout << "\n";
}

template<typename... Args>
void printTuple(const std::tuple<Args...>& t) {
    printTupleImpl(t, std::index_sequence_for<Args...>{});
}

int main() {
    auto t = std::make_tuple(1, "hello", 3.14);
    printTuple(t);  // 1, hello, 3.14

    return 0;
}
```

---

## 2. Fold Expressions

C++17 fold expressions simplify operations over parameter packs, eliminating the need for recursive templates in many cases.

### Four Forms of Fold

```cpp
#include <iostream>
#include <string>

// Unary right fold: (args op ...)
// Expands to: a1 op (a2 op (a3 op a4))
template<typename... Args>
auto sumRight(Args... args) {
    return (args + ...);
}

// Unary left fold: (... op args)
// Expands to: ((a1 op a2) op a3) op a4
template<typename... Args>
auto sumLeft(Args... args) {
    return (... + args);
}

// Binary right fold: (args op ... op init)
// Expands to: a1 op (a2 op (a3 op init))
template<typename... Args>
auto sumWithInit(Args... args) {
    return (args + ... + 0);  // Handles empty pack
}

// Binary left fold: (init op ... op args)
// Expands to: ((init op a1) op a2) op a3
template<typename... Args>
auto concatStrings(Args... args) {
    return (std::string{} + ... + args);
}

int main() {
    std::cout << sumRight(1, 2, 3, 4) << "\n";     // 10
    std::cout << sumLeft(1, 2, 3, 4) << "\n";       // 10
    std::cout << sumWithInit() << "\n";               // 0 (empty pack)

    std::cout << concatStrings("Hello", " ", "World") << "\n";

    return 0;
}
```

### Common Fold Patterns

```cpp
#include <iostream>

// Print all with comma separation
template<typename... Args>
void printComma(Args... args) {
    std::size_t n = 0;
    ((std::cout << (n++ ? ", " : "") << args), ...);
    std::cout << "\n";
}

// Check all conditions
template<typename... Args>
bool allPositive(Args... args) {
    return ((args > 0) && ...);
}

// Check any condition
template<typename... Args>
bool anyNegative(Args... args) {
    return ((args < 0) || ...);
}

// Apply function to all arguments
template<typename F, typename... Args>
void forEachArg(F func, Args&&... args) {
    (func(std::forward<Args>(args)), ...);
}

int main() {
    printComma(1, "two", 3.0, '4');   // 1, two, 3, 4
    std::cout << std::boolalpha;
    std::cout << allPositive(1, 2, 3) << "\n";   // true
    std::cout << allPositive(1, -2, 3) << "\n";  // false
    std::cout << anyNegative(1, -2, 3) << "\n";  // true

    forEachArg([](auto x) { std::cout << x << " "; },
               1, "hello", 3.14);  // 1 hello 3.14
    std::cout << "\n";

    return 0;
}
```

---

## 3. SFINAE

SFINAE (Substitution Failure Is Not An Error) allows the compiler to silently discard template overloads that produce invalid types during substitution.

### std::enable_if

```cpp
#include <iostream>
#include <type_traits>
#include <string>

// Method 1: Return type SFINAE
template<typename T>
typename std::enable_if_t<std::is_integral_v<T>, std::string>
describe(T value) {
    return "integer: " + std::to_string(value);
}

template<typename T>
typename std::enable_if_t<std::is_floating_point_v<T>, std::string>
describe(T value) {
    return "float: " + std::to_string(value);
}

// Method 2: Template parameter SFINAE (cleaner)
template<typename T, std::enable_if_t<std::is_pointer_v<T>, int> = 0>
void check(T ptr) {
    std::cout << "Pointer: " << (ptr ? "non-null" : "null") << "\n";
}

template<typename T, std::enable_if_t<!std::is_pointer_v<T>, int> = 0>
void check(T value) {
    std::cout << "Value: " << value << "\n";
}

int main() {
    std::cout << describe(42) << "\n";      // integer: 42
    std::cout << describe(3.14) << "\n";    // float: 3.140000

    int x = 10;
    check(&x);    // Pointer: non-null
    check(42);    // Value: 42

    return 0;
}
```

### Detecting Member Functions with SFINAE

```cpp
#include <iostream>
#include <type_traits>

// Detect if type has a .size() method
template<typename T, typename = void>
struct has_size : std::false_type {};

template<typename T>
struct has_size<T, std::void_t<decltype(std::declval<T>().size())>>
    : std::true_type {};

// Detect if type has operator<<
template<typename T, typename = void>
struct is_printable : std::false_type {};

template<typename T>
struct is_printable<T, std::void_t<
    decltype(std::declval<std::ostream&>() << std::declval<T>())
>> : std::true_type {};

// Usage
template<typename T>
void smartPrint(const T& obj) {
    if constexpr (has_size<T>::value) {
        std::cout << "Container with " << obj.size() << " elements\n";
    } else if constexpr (is_printable<T>::value) {
        std::cout << obj << "\n";
    } else {
        std::cout << "(unprintable type)\n";
    }
}

#include <vector>
#include <string>

int main() {
    std::cout << std::boolalpha;
    std::cout << "vector has size: " << has_size<std::vector<int>>::value << "\n";  // true
    std::cout << "int has size: " << has_size<int>::value << "\n";                   // false

    smartPrint(std::vector<int>{1, 2, 3});  // Container with 3 elements
    smartPrint(42);                           // 42
    smartPrint(std::string("hello"));        // Container with 5 elements

    return 0;
}
```

---

## 4. Type Traits

The `<type_traits>` header provides a rich set of compile-time type queries and transformations.

### Primary Type Categories

```cpp
#include <iostream>
#include <type_traits>
#include <vector>

template<typename T>
void analyzeType() {
    std::cout << std::boolalpha;
    std::cout << "  is_void:            " << std::is_void_v<T> << "\n";
    std::cout << "  is_integral:        " << std::is_integral_v<T> << "\n";
    std::cout << "  is_floating_point:  " << std::is_floating_point_v<T> << "\n";
    std::cout << "  is_array:           " << std::is_array_v<T> << "\n";
    std::cout << "  is_pointer:         " << std::is_pointer_v<T> << "\n";
    std::cout << "  is_reference:       " << std::is_reference_v<T> << "\n";
    std::cout << "  is_class:           " << std::is_class_v<T> << "\n";
    std::cout << "  is_enum:            " << std::is_enum_v<T> << "\n";
}

int main() {
    std::cout << "--- int ---\n";
    analyzeType<int>();

    std::cout << "--- double* ---\n";
    analyzeType<double*>();

    std::cout << "--- std::vector<int> ---\n";
    analyzeType<std::vector<int>>();

    return 0;
}
```

### Type Transformations

```cpp
#include <iostream>
#include <type_traits>

int main() {
    std::cout << std::boolalpha;

    // Remove qualifiers
    using A = std::remove_const_t<const int>;        // int
    using B = std::remove_reference_t<int&>;         // int
    using C = std::remove_pointer_t<int*>;           // int
    using D = std::decay_t<const int&>;              // int
    using E = std::decay_t<int[10]>;                 // int*
    using F = std::decay_t<int(double)>;             // int(*)(double)

    std::cout << std::is_same_v<A, int> << "\n";  // true
    std::cout << std::is_same_v<B, int> << "\n";  // true
    std::cout << std::is_same_v<C, int> << "\n";  // true
    std::cout << std::is_same_v<D, int> << "\n";  // true

    // Add qualifiers
    using G = std::add_const_t<int>;                 // const int
    using H = std::add_lvalue_reference_t<int>;      // int&
    using I = std::add_pointer_t<int>;               // int*

    // Conditional type selection
    using J = std::conditional_t<(sizeof(int) > 4), long, int>;

    // Common type
    using K = std::common_type_t<int, double>;       // double
    std::cout << std::is_same_v<K, double> << "\n";  // true

    return 0;
}
```

### Type Relationships

```cpp
#include <iostream>
#include <type_traits>

class Base {};
class Derived : public Base {};
class Unrelated {};

int main() {
    std::cout << std::boolalpha;

    // is_same
    std::cout << std::is_same_v<int, int> << "\n";          // true
    std::cout << std::is_same_v<int, unsigned> << "\n";      // false

    // is_base_of
    std::cout << std::is_base_of_v<Base, Derived> << "\n";   // true
    std::cout << std::is_base_of_v<Base, Unrelated> << "\n"; // false

    // is_convertible
    std::cout << std::is_convertible_v<Derived*, Base*> << "\n";  // true
    std::cout << std::is_convertible_v<int, double> << "\n";      // true

    // is_constructible
    std::cout << std::is_constructible_v<std::string, const char*> << "\n"; // true

    // is_assignable
    std::cout << std::is_assignable_v<int&, double> << "\n";  // true

    return 0;
}
```

---

## 5. if constexpr

C++17's `if constexpr` evaluates conditions at compile time and discards the false branch entirely, avoiding instantiation errors.

### Replacing SFINAE

```cpp
#include <iostream>
#include <type_traits>
#include <string>
#include <vector>

// Before: SFINAE (complex, hard to read)
template<typename T>
typename std::enable_if_t<std::is_integral_v<T>, std::string>
toStringOld(T value) { return std::to_string(value); }

template<typename T>
typename std::enable_if_t<std::is_floating_point_v<T>, std::string>
toStringOld(T value) { return std::to_string(value); }

// After: if constexpr (clean, readable)
template<typename T>
std::string toString(T value) {
    if constexpr (std::is_integral_v<T>) {
        return "int:" + std::to_string(value);
    } else if constexpr (std::is_floating_point_v<T>) {
        return "float:" + std::to_string(value);
    } else if constexpr (std::is_same_v<T, std::string>) {
        return "string:" + value;
    } else {
        return "(unknown type)";
    }
}

int main() {
    std::cout << toString(42) << "\n";
    std::cout << toString(3.14) << "\n";
    std::cout << toString(std::string("hello")) << "\n";

    return 0;
}
```

### Compile-Time Recursion with if constexpr

```cpp
#include <iostream>
#include <tuple>

// Print tuple elements using if constexpr
template<std::size_t I = 0, typename... Ts>
void printTuple(const std::tuple<Ts...>& t) {
    if constexpr (I < sizeof...(Ts)) {
        if constexpr (I > 0) std::cout << ", ";
        std::cout << std::get<I>(t);
        printTuple<I + 1>(t);
    }
}

// Compile-time factorial
template<int N>
constexpr int factorial() {
    if constexpr (N <= 1) {
        return 1;
    } else {
        return N * factorial<N - 1>();
    }
}

int main() {
    auto t = std::make_tuple(1, "hello", 3.14);
    printTuple(t);  // 1, hello, 3.14
    std::cout << "\n";

    constexpr int f5 = factorial<5>();
    std::cout << "5! = " << f5 << "\n";  // 120

    return 0;
}
```

---

## 6. constexpr and consteval

### constexpr Functions

`constexpr` functions can be evaluated at compile time when given constant arguments, or at runtime otherwise.

```cpp
#include <iostream>
#include <array>

constexpr int fibonacci(int n) {
    if (n <= 1) return n;
    return fibonacci(n - 1) + fibonacci(n - 2);
}

constexpr int power(int base, int exp) {
    int result = 1;
    for (int i = 0; i < exp; ++i) {
        result *= base;
    }
    return result;
}

// constexpr class
class Point {
public:
    int x, y;
    constexpr Point(int x, int y) : x(x), y(y) {}
    constexpr int manhattanDistance() const { return x + y; }
    constexpr Point operator+(const Point& other) const {
        return {x + other.x, y + other.y};
    }
};

int main() {
    // Compile-time evaluation
    constexpr int fib10 = fibonacci(10);
    static_assert(fib10 == 55, "fibonacci(10) should be 55");

    constexpr int p = power(2, 10);
    static_assert(p == 1024);

    // Used as array size
    std::array<int, fibonacci(6)> arr;  // Size 8

    constexpr Point a(3, 4);
    constexpr Point b(1, 2);
    constexpr Point c = a + b;
    static_assert(c.x == 4 && c.y == 6);

    // Runtime evaluation also works
    int n;
    std::cin >> n;
    std::cout << "fib(" << n << ") = " << fibonacci(n) << "\n";

    return 0;
}
```

### consteval (C++20)

`consteval` functions **must** be evaluated at compile time. They are called "immediate functions."

```cpp
#include <iostream>

// Must be evaluated at compile time
consteval int square(int n) {
    return n * n;
}

// consteval can call constexpr, but not vice versa
consteval int cube(int n) {
    return n * n * n;
}

int main() {
    constexpr int s = square(5);  // OK: compile-time
    std::cout << s << "\n";       // 25

    // int x = 5;
    // int bad = square(x);  // ERROR: x is not a constant expression

    // Useful for compile-time assertions
    constexpr int result = cube(3);
    static_assert(result == 27);

    return 0;
}
```

---

## 7. Tag Dispatch

Tag dispatch uses empty types (tags) to select overloads based on type properties, providing a clean alternative to SFINAE.

```cpp
#include <iostream>
#include <type_traits>
#include <iterator>
#include <vector>
#include <list>

// Tag dispatch for iterator categories
namespace detail {

template<typename Iter>
void advanceImpl(Iter& it, int n, std::random_access_iterator_tag) {
    std::cout << "Random access advance (O(1))\n";
    it += n;
}

template<typename Iter>
void advanceImpl(Iter& it, int n, std::bidirectional_iterator_tag) {
    std::cout << "Bidirectional advance (O(n))\n";
    while (n > 0) { ++it; --n; }
    while (n < 0) { --it; ++n; }
}

template<typename Iter>
void advanceImpl(Iter& it, int n, std::input_iterator_tag) {
    std::cout << "Input advance (O(n), forward only)\n";
    while (n > 0) { ++it; --n; }
}

} // namespace detail

template<typename Iter>
void myAdvance(Iter& it, int n) {
    // Dispatch based on iterator category tag
    detail::advanceImpl(it, n,
        typename std::iterator_traits<Iter>::iterator_category{});
}

// Tag dispatch with true_type/false_type
template<typename T>
void processImpl(T value, std::true_type /* is_integral */) {
    std::cout << "Processing integer: " << value << "\n";
}

template<typename T>
void processImpl(T value, std::false_type /* is_integral */) {
    std::cout << "Processing non-integer: " << value << "\n";
}

template<typename T>
void process(T value) {
    processImpl(value, std::is_integral<T>{});
}

int main() {
    std::vector<int> v = {1, 2, 3, 4, 5};
    auto vit = v.begin();
    myAdvance(vit, 3);  // Random access advance (O(1))

    std::list<int> l = {1, 2, 3, 4, 5};
    auto lit = l.begin();
    myAdvance(lit, 3);  // Bidirectional advance (O(n))

    process(42);    // Processing integer: 42
    process(3.14);  // Processing non-integer: 3.14

    return 0;
}
```

---

## 8. Policy-Based Design

Policy-based design uses template parameters to inject behavior at compile time, creating flexible yet efficient classes.

```cpp
#include <iostream>
#include <string>
#include <mutex>

// Logging policies
struct ConsoleLog {
    static void log(const std::string& msg) {
        std::cout << "[Console] " << msg << "\n";
    }
};

struct NullLog {
    static void log(const std::string&) {
        // Do nothing
    }
};

// Threading policies
struct SingleThreaded {
    struct Lock {
        Lock() {}  // No-op
    };
};

struct MultiThreaded {
    struct Lock {
        Lock() { /* mutex.lock(); */ }
        ~Lock() { /* mutex.unlock(); */ }
    };
};

// Class using policies
template<typename LogPolicy = ConsoleLog,
         typename ThreadPolicy = SingleThreaded>
class DataStore {
    int data_ = 0;

public:
    void set(int value) {
        typename ThreadPolicy::Lock lock;
        LogPolicy::log("Setting value to " + std::to_string(value));
        data_ = value;
    }

    int get() const {
        return data_;
    }
};

int main() {
    // Verbose, single-threaded store
    DataStore<ConsoleLog, SingleThreaded> verbose;
    verbose.set(42);  // [Console] Setting value to 42

    // Silent store (logging compiled out entirely)
    DataStore<NullLog, SingleThreaded> silent;
    silent.set(42);   // No output, no overhead

    // Thread-safe store
    DataStore<ConsoleLog, MultiThreaded> safe;
    safe.set(42);     // [Console] Setting value to 42

    return 0;
}
```

### CRTP (Curiously Recurring Template Pattern)

```cpp
#include <iostream>

// Static polymorphism via CRTP
template<typename Derived>
class Shape {
public:
    double area() const {
        return static_cast<const Derived*>(this)->areaImpl();
    }

    void describe() const {
        std::cout << "Shape with area: " << area() << "\n";
    }
};

class Circle : public Shape<Circle> {
    double radius_;
public:
    Circle(double r) : radius_(r) {}
    double areaImpl() const { return 3.14159 * radius_ * radius_; }
};

class Rectangle : public Shape<Rectangle> {
    double w_, h_;
public:
    Rectangle(double w, double h) : w_(w), h_(h) {}
    double areaImpl() const { return w_ * h_; }
};

// Works with any Shape<Derived> -- no virtual dispatch
template<typename T>
void printArea(const Shape<T>& shape) {
    shape.describe();
}

int main() {
    Circle c(5.0);
    Rectangle r(3.0, 4.0);

    printArea(c);  // Shape with area: 78.5397
    printArea(r);  // Shape with area: 12

    return 0;
}
```

---

## 9. Summary

| Technique | Purpose | Era |
|-----------|---------|-----|
| Variadic templates | Accept arbitrary arguments | C++11 |
| Fold expressions | Simplify pack operations | C++17 |
| SFINAE / enable_if | Conditional overloads | C++11 |
| Type traits | Compile-time type queries | C++11 |
| if constexpr | Compile-time branching | C++17 |
| constexpr / consteval | Compile-time computation | C++11/C++20 |
| Tag dispatch | Clean overload selection | C++98+ |
| Policy-based design | Compile-time strategy | C++98+ |
| CRTP | Static polymorphism | C++98+ |

---

## Exercises

### Exercise 1: Type-Safe printf

Implement a `safePrintf(format, args...)` function using variadic templates that replaces `%` placeholders with type-safe arguments. Throw an exception for mismatched argument count.

### Exercise 2: Compile-Time String Hash

Write a `constexpr` function that computes a hash of a string literal at compile time. Use it in a `switch` statement.

### Exercise 3: has_method Detector

Create a generic `has_method` detector using SFINAE and `std::void_t` that can detect any named method (e.g., `has_push_back`, `has_reserve`).

### Exercise 4: Compile-Time Fibonacci Sequence

Implement a `constexpr std::array` that contains the first N Fibonacci numbers, computed entirely at compile time.

### Exercise 5: Policy-Based Logger

Design a Logger class using policy-based design with interchangeable output policies (console, file, null) and formatting policies (plain, timestamped, JSON).

---

## Next Steps

Smart pointers and RAII form the backbone of resource management in modern C++. Let's explore `unique_ptr`, `shared_ptr`, and `weak_ptr` in depth in [04_Smart_Pointers_and_RAII.md](./04_Smart_Pointers_and_RAII.md).
