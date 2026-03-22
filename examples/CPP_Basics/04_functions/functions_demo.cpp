// functions_demo.cpp — Overloading, default params, pass-by-reference
// Compile: g++ -std=c++20 -Wall -Wextra -o functions_demo functions_demo.cpp

#include <iostream>
#include <string>
#include <vector>
#include <numeric>

// --- Function overloading ---
int add(int a, int b) { return a + b; }
double add(double a, double b) { return a + b; }
std::string add(const std::string& a, const std::string& b) { return a + b; }

// --- Default parameters ---
void greet(const std::string& name, const std::string& greeting = "Hello") {
    std::cout << greeting << ", " << name << "!\n";
}

// --- Pass by value vs reference vs const reference ---
void pass_by_value(int x) {
    x = 999;  // does NOT affect caller
}

void pass_by_reference(int& x) {
    x = 999;  // DOES affect caller
}

void pass_by_const_ref(const std::vector<int>& v) {
    // efficient: no copy, but cannot modify
    std::cout << "Vector size: " << v.size() << '\n';
}

// --- Returning multiple values (structured bindings) ---
struct MinMax {
    int min_val;
    int max_val;
};

MinMax find_minmax(const std::vector<int>& v) {
    int lo = v[0], hi = v[0];
    for (int x : v) {
        if (x < lo) lo = x;
        if (x > hi) hi = x;
    }
    return {lo, hi};
}

// --- Inline function ---
inline int square(int x) { return x * x; }

// --- Lambda functions ---
void lambda_demo() {
    std::cout << "\n=== Lambda ===\n";

    auto multiply = [](int a, int b) { return a * b; };
    std::cout << "5 * 3 = " << multiply(5, 3) << '\n';

    int factor = 10;
    auto scale = [factor](int x) { return x * factor; };
    std::cout << "scale(7) = " << scale(7) << '\n';

    // Generic lambda (C++14)
    auto print = [](const auto& val) { std::cout << val << '\n'; };
    print(42);
    print("hello");
}

int main() {
    // Overloading
    std::cout << "=== Overloading ===\n";
    std::cout << "add(3, 4)       = " << add(3, 4) << '\n';
    std::cout << "add(1.5, 2.7)   = " << add(1.5, 2.7) << '\n';
    std::cout << "add(\"Hi\",\" C++\") = " << add(std::string("Hi"), std::string(" C++")) << '\n';

    // Default params
    std::cout << "\n=== Default Parameters ===\n";
    greet("Alice");
    greet("Bob", "Bonjour");

    // Pass by value vs reference
    std::cout << "\n=== Pass by Value vs Reference ===\n";
    int val = 10;
    pass_by_value(val);
    std::cout << "After pass_by_value: " << val << '\n';     // still 10
    pass_by_reference(val);
    std::cout << "After pass_by_reference: " << val << '\n';  // now 999

    // Const reference
    std::vector<int> nums = {5, 2, 8, 1, 9, 3};
    pass_by_const_ref(nums);

    // Structured binding with return struct
    std::cout << "\n=== Multiple Return Values ===\n";
    auto [lo, hi] = find_minmax(nums);
    std::cout << "min=" << lo << " max=" << hi << '\n';

    // Inline
    std::cout << "\n=== Inline ===\n";
    std::cout << "square(7) = " << square(7) << '\n';

    // Lambda
    lambda_demo();

    return 0;
}
