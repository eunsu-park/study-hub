// concepts_demo.cpp — Custom concepts, requires, constrained templates (C++20)
// Compile: g++ -std=c++20 -Wall -Wextra -o concepts_demo concepts_demo.cpp

#include <iostream>
#include <string>
#include <vector>
#include <concepts>
#include <type_traits>
#include <numeric>

// --- Basic concept ---
template <typename T>
concept Numeric = std::is_arithmetic_v<T>;

template <Numeric T>
T square(T x) {
    return x * x;
}

// --- Concept with requires clause ---
template <typename T>
concept Printable = requires(T t) {
    { std::cout << t } -> std::same_as<std::ostream&>;
};

template <Printable T>
void display(const T& val) {
    std::cout << "  -> " << val << '\n';
}

// --- Compound requires expression ---
template <typename T>
concept Container = requires(T c) {
    { c.begin() } -> std::input_or_output_iterator;
    { c.end() } -> std::input_or_output_iterator;
    { c.size() } -> std::convertible_to<size_t>;
    typename T::value_type;
};

template <Container C>
void print_container(const C& c) {
    std::cout << "[";
    bool first = true;
    for (const auto& x : c) {
        if (!first) std::cout << ", ";
        std::cout << x;
        first = false;
    }
    std::cout << "]";
}

// --- Concept composition ---
template <typename T>
concept Sortable = Container<T> && requires(T c) {
    { *c.begin() < *c.begin() } -> std::convertible_to<bool>;
};

template <Sortable C>
typename C::value_type find_median(C c) {  // by value — we sort a copy
    std::sort(c.begin(), c.end());
    return c[c.size() / 2];
}

// --- Constrained auto ---
void constrained_auto_demo() {
    std::cout << "\n=== Constrained auto ===\n";
    auto add = [](std::integral auto a, std::integral auto b) {
        return a + b;
    };
    std::cout << "add(3, 4) = " << add(3, 4) << '\n';
    // add(3.5, 4.5);  // ERROR: not integral
}

// --- requires clause on member function ---
template <typename T>
class Wrapper {
    T value_;
public:
    explicit Wrapper(T val) : value_(std::move(val)) {}

    T get() const { return value_; }

    // Only available for arithmetic types
    T doubled() const requires std::is_arithmetic_v<T> {
        return value_ * 2;
    }

    // Only available for string-like types
    size_t length() const requires requires { value_.length(); } {
        return value_.length();
    }
};

// --- Subsumption: more specific concept wins ---
template <typename T>
concept Addable = requires(T a, T b) { { a + b } -> std::same_as<T>; };

template <typename T>
concept NumericAddable = Addable<T> && std::is_arithmetic_v<T>;

template <Addable T>
std::string describe_add(T, T) { return "generic addable"; }

template <NumericAddable T>
std::string describe_add(T, T) { return "numeric addable (more specific)"; }

int main() {
    // Basic concept
    std::cout << "=== Numeric Concept ===\n";
    std::cout << "square(5) = " << square(5) << '\n';
    std::cout << "square(2.5) = " << square(2.5) << '\n';
    // square("hi");  // ERROR: not Numeric

    // Printable
    std::cout << "\n=== Printable Concept ===\n";
    display(42);
    display(3.14);
    display(std::string("hello"));

    // Container concept
    std::cout << "\n=== Container Concept ===\n";
    std::vector<int> v = {5, 2, 8, 1, 9};
    std::cout << "vector: ";
    print_container(v);
    std::cout << '\n';

    // Sortable
    std::cout << "median: " << find_median(v) << '\n';

    // Constrained auto
    constrained_auto_demo();

    // Wrapper with conditional members
    std::cout << "\n=== Conditional Members ===\n";
    Wrapper<int> wi(21);
    std::cout << "doubled: " << wi.doubled() << '\n';

    Wrapper<std::string> ws(std::string("hello"));
    std::cout << "length: " << ws.length() << '\n';
    // ws.doubled();  // ERROR: not available for string

    // Subsumption
    std::cout << "\n=== Concept Subsumption ===\n";
    std::cout << describe_add(1, 2) << '\n';
    std::cout << describe_add(std::string("a"), std::string("b")) << '\n';

    return 0;
}
