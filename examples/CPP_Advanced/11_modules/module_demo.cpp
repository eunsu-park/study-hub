// module_demo.cpp — C++20 Modules: interface and implementation demo
// NOTE: Module support varies by compiler. This file demonstrates the syntax.
//       Actual compilation requires specific compiler flags and build system support.
//
// With GCC:
//   g++ -std=c++20 -fmodules-ts -x c++-system-header iostream
//   g++ -std=c++20 -fmodules-ts -c math_utils.cppm     (module interface)
//   g++ -std=c++20 -fmodules-ts -c module_demo.cpp      (consumer)
//   g++ -std=c++20 -o module_demo module_demo.o math_utils.o
//
// With Clang:
//   clang++ -std=c++20 -fmodule-file=math_utils.pcm ...
//
// This single file demonstrates the CONCEPTS behind modules.
// For an actual modular build, split into separate files as shown below.

#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <numeric>

// ============================================================
// SIMULATED MODULE INTERFACE: math_utils
// In a real module, this would be in math_utils.cppm:
//
//   export module math_utils;
//
//   export namespace math_utils {
//       double circle_area(double r);
//       double factorial(int n);
//       ...
//   }
// ============================================================

namespace math_utils {

double circle_area(double radius) {
    return M_PI * radius * radius;
}

double factorial(int n) {
    if (n <= 1) return 1.0;
    double result = 1.0;
    for (int i = 2; i <= n; ++i) result *= i;
    return result;
}

double mean(const std::vector<double>& data) {
    if (data.empty()) return 0.0;
    return std::accumulate(data.begin(), data.end(), 0.0) / data.size();
}

double stddev(const std::vector<double>& data) {
    if (data.size() < 2) return 0.0;
    double m = mean(data);
    double sum_sq = 0.0;
    for (double x : data) sum_sq += (x - m) * (x - m);
    return std::sqrt(sum_sq / (data.size() - 1));
}

}  // namespace math_utils

// ============================================================
// SIMULATED MODULE INTERFACE: string_utils
// In a real module: export module string_utils;
// ============================================================

namespace string_utils {

std::string to_upper(std::string s) {
    for (char& c : s) c = static_cast<char>(std::toupper(c));
    return s;
}

std::string to_lower(std::string s) {
    for (char& c : s) c = static_cast<char>(std::tolower(c));
    return s;
}

std::vector<std::string> split(const std::string& s, char delim) {
    std::vector<std::string> tokens;
    std::string token;
    for (char c : s) {
        if (c == delim) {
            if (!token.empty()) tokens.push_back(token);
            token.clear();
        } else {
            token += c;
        }
    }
    if (!token.empty()) tokens.push_back(token);
    return tokens;
}

std::string join(const std::vector<std::string>& parts, const std::string& sep) {
    std::string result;
    for (size_t i = 0; i < parts.size(); ++i) {
        if (i > 0) result += sep;
        result += parts[i];
    }
    return result;
}

}  // namespace string_utils

// ============================================================
// Consumer: import math_utils; import string_utils;
// (In real modules, replace #include with import)
// ============================================================

int main() {
    std::cout << "=== Module Demo (simulated) ===\n\n";

    // Using math_utils
    std::cout << "--- math_utils ---\n";
    std::cout << "circle_area(5) = " << math_utils::circle_area(5.0) << '\n';
    std::cout << "factorial(10)  = " << math_utils::factorial(10) << '\n';

    std::vector<double> data = {2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0};
    std::cout << "mean   = " << math_utils::mean(data) << '\n';
    std::cout << "stddev = " << math_utils::stddev(data) << '\n';

    // Using string_utils
    std::cout << "\n--- string_utils ---\n";
    std::cout << "to_upper(\"hello\") = " << string_utils::to_upper("hello") << '\n';
    std::cout << "to_lower(\"WORLD\") = " << string_utils::to_lower("WORLD") << '\n';

    auto parts = string_utils::split("one-two-three-four", '-');
    std::cout << "split: ";
    for (const auto& p : parts) std::cout << '[' << p << "] ";
    std::cout << '\n';

    std::cout << "join:  " << string_utils::join(parts, " | ") << '\n';

    // Module benefits summary
    std::cout << "\n=== C++20 Module Benefits ===\n";
    std::cout << "1. Faster compilation (no header re-parsing)\n";
    std::cout << "2. No include-order dependencies\n";
    std::cout << "3. No macro leakage across modules\n";
    std::cout << "4. Explicit export control\n";
    std::cout << "5. Module partitions for large libraries\n";

    return 0;
}
