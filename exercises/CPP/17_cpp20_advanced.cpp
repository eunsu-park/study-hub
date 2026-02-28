/*
 * Exercises for Lesson 17: C++20 Advanced Features
 * Topic: CPP
 * Compile: g++ -std=c++20 -Wall -Wextra -o ex17 17_cpp20_advanced.cpp
 *
 * Note: Requires a C++20-capable compiler (GCC 10+, Clang 13+, MSVC 19.29+).
 *       If C++20 is unavailable, this file demonstrates the concepts with
 *       C++17 fallbacks where noted.
 */
#include <iostream>
#include <vector>
#include <string>
#include <numeric>
#include <algorithm>
#include <type_traits>
using namespace std;

// Check for C++20 support
#if __cplusplus >= 202002L
#define HAS_CPP20 1
#include <concepts>
#include <ranges>
#else
#define HAS_CPP20 0
#endif

// === Exercise 1: Define a Concept ===
// Problem: Define a Concept representing a "printable" type (supports operator<<).

#if HAS_CPP20

// Concept: T is Printable if os << t is valid and returns ostream&
template<typename T>
concept Printable = requires(ostream& os, T t) {
    { os << t } -> same_as<ostream&>;
};

// Additional concepts for demonstration
template<typename T>
concept Numeric = is_arithmetic_v<T>;

template<typename T>
concept Addable = requires(T a, T b) {
    { a + b } -> convertible_to<T>;
};

// Constrained function: only accepts Printable types
template<Printable T>
void printValue(const T& value) {
    cout << "    Printable: " << value << endl;
}

// Constrained function with multiple concepts
template<typename T>
    requires Printable<T> && Numeric<T>
void printNumeric(const T& value) {
    cout << "    Numeric+Printable: " << value << " (x2 = " << value * 2 << ")" << endl;
}

#else

// C++17 fallback using SFINAE
template<typename T,
         typename = decltype(declval<ostream&>() << declval<T>())>
void printValue(const T& value) {
    cout << "    Printable (C++17 SFINAE): " << value << endl;
}

template<typename T,
         enable_if_t<is_arithmetic_v<T>, int> = 0>
void printNumeric(const T& value) {
    cout << "    Numeric (C++17 SFINAE): " << value << " (x2 = " << value * 2 << ")" << endl;
}

#endif

// A non-printable type for testing concept constraints
struct NotPrintable {
    int data;
};

void exercise_1() {
    cout << "=== Exercise 1: Define a Concept ===" << endl;

#if HAS_CPP20
    cout << "  Using C++20 concepts:" << endl;
#else
    cout << "  Using C++17 SFINAE fallback:" << endl;
#endif

    // Printable types
    printValue(42);
    printValue(3.14);
    printValue(string("Hello Concepts"));

    // Numeric + Printable
    printNumeric(100);
    printNumeric(2.718);

    // This would fail to compile (correctly):
    // printValue(NotPrintable{42});  // No operator<< for NotPrintable

#if HAS_CPP20
    // Compile-time concept checks
    static_assert(Printable<int>, "int should be Printable");
    static_assert(Printable<string>, "string should be Printable");
    static_assert(!Printable<NotPrintable>, "NotPrintable should not be Printable");
    static_assert(Numeric<double>, "double should be Numeric");
    static_assert(!Numeric<string>, "string should not be Numeric");
    cout << "\n  All static_assert concept checks passed!" << endl;
#endif
}

// === Exercise 2: Range Pipeline ===
// Problem: Find the sum of squares of numbers from 1 to 100 that are
//          multiples of 3 but not multiples of 5.

void exercise_2() {
    cout << "\n=== Exercise 2: Range Pipeline ===" << endl;

#if HAS_CPP20
    cout << "  Using C++20 ranges:" << endl;

    // Declarative pipeline: generate -> filter -> transform -> accumulate
    auto range = views::iota(1, 101)
        | views::filter([](int x) { return x % 3 == 0 && x % 5 != 0; })
        | views::transform([](int x) { return x * x; });

    int sum = 0;
    for (int val : range) {
        sum += val;
    }

    cout << "    Sum of squares of multiples of 3 (not 5) in [1,100]: "
         << sum << endl;

    // Show which numbers contribute
    cout << "    Contributing numbers: ";
    for (int x : views::iota(1, 101)
                 | views::filter([](int x) { return x % 3 == 0 && x % 5 != 0; })
                 | views::take(10)) {
        cout << x << " ";
    }
    cout << "..." << endl;

#else
    cout << "  Using C++17 STL algorithms (ranges not available):" << endl;

    // Generate numbers 1..100
    vector<int> nums(100);
    iota(nums.begin(), nums.end(), 1);

    // Filter: multiples of 3 but not 5
    vector<int> filtered;
    copy_if(nums.begin(), nums.end(), back_inserter(filtered),
            [](int x) { return x % 3 == 0 && x % 5 != 0; });

    // Transform: square each
    vector<int> squared(filtered.size());
    transform(filtered.begin(), filtered.end(), squared.begin(),
              [](int x) { return x * x; });

    // Accumulate
    int sum = accumulate(squared.begin(), squared.end(), 0);

    cout << "    Sum of squares of multiples of 3 (not 5) in [1,100]: "
         << sum << endl;

    // Show first 10 contributing numbers
    cout << "    Contributing numbers: ";
    for (size_t i = 0; i < min(filtered.size(), size_t(10)); i++) {
        cout << filtered[i] << " ";
    }
    cout << "..." << endl;

#endif

    // Manual verification: multiples of 3 not div by 5 in [1,100]:
    // 3, 6, 9, 12, 18, 21, 24, 27, 33, 36, ...
    // Their squares summed
    int manualSum = 0;
    for (int i = 1; i <= 100; i++) {
        if (i % 3 == 0 && i % 5 != 0) {
            manualSum += i * i;
        }
    }
    cout << "    Manual verification: " << manualSum << endl;
    cout << "    Match: " << boolalpha << (sum == manualSum) << endl;
}

int main() {
    exercise_1();
    exercise_2();
    cout << "\nAll exercises completed!" << endl;
    return 0;
}
