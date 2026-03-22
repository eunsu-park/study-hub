// metaprogramming_demo.cpp — Variadic templates, type_traits, if constexpr
// Compile: g++ -std=c++20 -Wall -Wextra -o metaprogramming_demo metaprogramming_demo.cpp

#include <iostream>
#include <string>
#include <type_traits>
#include <array>
#include <tuple>

// --- Compile-time factorial ---
template <int N>
struct Factorial {
    static constexpr int value = N * Factorial<N - 1>::value;
};
template <>
struct Factorial<0> {
    static constexpr int value = 1;
};

// Modern alternative: constexpr function
constexpr int factorial(int n) { return (n <= 1) ? 1 : n * factorial(n - 1); }

// --- Variadic templates: type-safe print ---
void print() {
    std::cout << '\n';  // base case
}

template <typename T, typename... Args>
void print(const T& first, const Args&... rest) {
    std::cout << first;
    if constexpr (sizeof...(rest) > 0) {
        std::cout << ", ";
    }
    print(rest...);
}

// --- Fold expressions (C++17) ---
template <typename... Args>
auto sum(Args... args) {
    return (args + ...);  // unary right fold
}

template <typename... Args>
void print_all(const Args&... args) {
    ((std::cout << args << ' '), ...);  // comma fold
    std::cout << '\n';
}

// --- if constexpr for type-dependent behavior ---
template <typename T>
std::string type_info(const T& val) {
    if constexpr (std::is_integral_v<T>) {
        return "integer: " + std::to_string(val);
    } else if constexpr (std::is_floating_point_v<T>) {
        return "float: " + std::to_string(val);
    } else if constexpr (std::is_same_v<T, std::string>) {
        return "string: " + val;
    } else {
        return "unknown type";
    }
}

// --- SFINAE with enable_if ---
template <typename T>
std::enable_if_t<std::is_arithmetic_v<T>, T>
safe_abs(T val) {
    return val < 0 ? -val : val;
}

// --- Compile-time array generation ---
template <size_t N>
constexpr std::array<int, N> make_squares() {
    std::array<int, N> arr{};
    for (size_t i = 0; i < N; ++i) {
        arr[i] = static_cast<int>(i * i);
    }
    return arr;
}

// --- Type list operations ---
template <typename... Ts>
struct TypeList {
    static constexpr size_t size = sizeof...(Ts);
};

template <typename T, typename List>
struct Contains;

template <typename T, typename... Ts>
struct Contains<T, TypeList<Ts...>> {
    static constexpr bool value = (std::is_same_v<T, Ts> || ...);
};

int main() {
    // Compile-time factorial
    std::cout << "=== Compile-time Factorial ===\n";
    std::cout << "Factorial<5> = " << Factorial<5>::value << '\n';
    constexpr int f10 = factorial(10);
    std::cout << "factorial(10) = " << f10 << '\n';

    // Variadic print
    std::cout << "\n=== Variadic Templates ===\n";
    print(1, 2.5, "hello", 'A', std::string("world"));

    // Fold expressions
    std::cout << "\n=== Fold Expressions ===\n";
    std::cout << "sum(1,2,3,4,5) = " << sum(1, 2, 3, 4, 5) << '\n';
    std::cout << "sum(1.1, 2.2, 3.3) = " << sum(1.1, 2.2, 3.3) << '\n';
    print_all("a", 42, 3.14, "z");

    // if constexpr
    std::cout << "\n=== if constexpr ===\n";
    std::cout << type_info(42) << '\n';
    std::cout << type_info(3.14) << '\n';
    std::cout << type_info(std::string("hello")) << '\n';

    // SFINAE
    std::cout << "\n=== SFINAE (safe_abs) ===\n";
    std::cout << "safe_abs(-7) = " << safe_abs(-7) << '\n';
    std::cout << "safe_abs(-3.5) = " << safe_abs(-3.5) << '\n';

    // Compile-time array
    std::cout << "\n=== Compile-time Array ===\n";
    constexpr auto squares = make_squares<8>();
    for (auto x : squares) std::cout << x << ' ';
    std::cout << '\n';

    // Type list
    std::cout << "\n=== Type List ===\n";
    using MyTypes = TypeList<int, double, std::string>;
    std::cout << "TypeList size: " << MyTypes::size << '\n';
    std::cout << "Contains<int>: " << Contains<int, MyTypes>::value << '\n';
    std::cout << "Contains<char>: " << Contains<char, MyTypes>::value << '\n';

    return 0;
}
