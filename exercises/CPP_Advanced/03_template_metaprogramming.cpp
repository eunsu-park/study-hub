// Exercise 03: Template Metaprogramming
// Practice variadic templates, type traits, fold expressions, if constexpr.
// Compile: g++ -std=c++20 -Wall -Wextra -o ex03 03_template_metaprogramming.cpp && ./ex03

#include <iostream>
#include <string>
#include <vector>
#include <type_traits>
#include <tuple>
#include <array>
#include <cassert>

// TODO 1: Implement a compile-time power function using template recursion.
// Power<2, 10>::value should be 1024.

// template <int Base, int Exp>
// struct Power { ... };

// TODO 2: Implement a variadic "tuple_print" that prints all elements of a tuple.
// tuple_print(std::make_tuple(1, 3.14, "hello")) -> "1 3.14 hello\n"

// template <typename Tuple, size_t... Is>
// void tuple_print_impl(const Tuple& t, std::index_sequence<Is...>) { ... }
//
// template <typename... Args>
// void tuple_print(const std::tuple<Args...>& t) { ... }

// TODO 3: Implement a type-safe "Variant" that can hold one of several types.
// Use if constexpr and type_traits to implement a get<T>() method.
// Simplified: store a union-like buffer + type index.

// (This is challenging — start with a simpler approach using std::variant)

// TODO 4: Implement a compile-time string hash using constexpr.
// constexpr uint32_t fnv1a(const char* s) that works at compile time.

// constexpr uint32_t fnv1a(const char* s) { ... }

// TODO 5: Implement a fold-expression based "all_same" that checks
// if all arguments are equal.
// all_same(1, 1, 1) -> true
// all_same(1, 2, 1) -> false

// template <typename T, typename... Args>
// constexpr bool all_same(T first, Args... rest) { ... }

// TODO 6: Implement a type list with "contains" and "index_of" operations.
// TypeList<int, double, string>::contains<int> -> true
// TypeList<int, double, string>::index_of<double> -> 1

// template <typename... Ts>
// struct TypeList { ... };

int main() {
    std::cout << "=== Exercise 03: Template Metaprogramming ===\n\n";

    // Test 1: Compile-time power
    // static_assert(Power<2, 10>::value == 1024);
    // static_assert(Power<3, 4>::value == 81);
    // std::cout << "Test 1 passed: Power<2,10> = 1024\n";

    // Test 2: tuple_print
    // auto t = std::make_tuple(42, 3.14, std::string("hello"), 'A');
    // tuple_print(t);
    // std::cout << "Test 2 passed: tuple_print\n";

    // Test 4: Compile-time hash
    // constexpr auto h1 = fnv1a("hello");
    // constexpr auto h2 = fnv1a("world");
    // static_assert(h1 != h2);
    // std::cout << "Test 4 passed: fnv1a(\"hello\") = " << h1 << '\n';

    // Test 5: all_same
    // static_assert(all_same(1, 1, 1) == true);
    // static_assert(all_same(1, 2, 1) == false);
    // std::cout << "Test 5 passed: all_same\n";

    // Test 6: TypeList
    // using TL = TypeList<int, double, std::string>;
    // static_assert(TL::contains<int> == true);
    // static_assert(TL::contains<char> == false);
    // static_assert(TL::index_of<double> == 1);
    // std::cout << "Test 6 passed: TypeList\n";

    std::cout << "Uncomment tests as you implement each part.\n";
    return 0;
}
