// variables_demo.cpp — C++ fundamental types, auto, const, constexpr
// Compile: g++ -std=c++20 -Wall -Wextra -o variables_demo variables_demo.cpp

#include <iostream>
#include <string>
#include <limits>
#include <typeinfo>

constexpr int DAYS_IN_WEEK = 7;
constexpr double PI = 3.14159265358979;

constexpr int factorial(int n) {
    return (n <= 1) ? 1 : n * factorial(n - 1);
}

int main() {
    // --- Fundamental types ---
    bool flag = true;
    char letter = 'A';
    int count = 42;
    unsigned int positive_only = 100u;
    long long big = 9'000'000'000LL;   // digit separator (C++14)
    float  f = 3.14f;
    double d = 3.141592653589793;

    std::cout << "=== Fundamental Types ===\n";
    std::cout << "bool  : " << flag << "  sizeof=" << sizeof(bool) << '\n';
    std::cout << "char  : " << letter << "  sizeof=" << sizeof(char) << '\n';
    std::cout << "int   : " << count << "  sizeof=" << sizeof(int) << '\n';
    std::cout << "uint  : " << positive_only << '\n';
    std::cout << "llong : " << big << "  sizeof=" << sizeof(long long) << '\n';
    std::cout << "float : " << f << "  sizeof=" << sizeof(float) << '\n';
    std::cout << "double: " << d << "  sizeof=" << sizeof(double) << '\n';

    // --- auto type deduction ---
    std::cout << "\n=== auto deduction ===\n";
    auto x = 10;          // int
    auto y = 3.14;        // double
    auto name = std::string("C++");  // std::string
    std::cout << "x=" << x << " type=" << typeid(x).name() << '\n';
    std::cout << "y=" << y << " type=" << typeid(y).name() << '\n';
    std::cout << "name=" << name << '\n';

    // --- const vs constexpr ---
    std::cout << "\n=== const / constexpr ===\n";
    const int max_score = 100;           // runtime constant
    constexpr int compile_val = factorial(5);  // compile-time
    std::cout << "DAYS_IN_WEEK = " << DAYS_IN_WEEK << '\n';
    std::cout << "PI           = " << PI << '\n';
    std::cout << "max_score    = " << max_score << '\n';
    std::cout << "factorial(5) = " << compile_val << '\n';

    // --- Numeric limits ---
    std::cout << "\n=== Numeric Limits ===\n";
    std::cout << "int  min=" << std::numeric_limits<int>::min()
              << "  max=" << std::numeric_limits<int>::max() << '\n';
    std::cout << "double epsilon=" << std::numeric_limits<double>::epsilon() << '\n';

    // --- Type conversions ---
    std::cout << "\n=== Type Conversions ===\n";
    int    i = 65;
    char   c = static_cast<char>(i);
    double narrowed = static_cast<double>(i) / 4;
    std::cout << "int 65 -> char: " << c << '\n';
    std::cout << "65 / 4 (double): " << narrowed << '\n';

    return 0;
}
