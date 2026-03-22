// Exercise 14: CMake and Build Basics
// This exercise tests your understanding of build systems.
// It is a standalone program — but the real exercise is writing
// CMakeLists.txt files to build a multi-file project.
//
// Compile (standalone): g++ -std=c++20 -Wall -Wextra -o ex14 14_cmake_and_build_basics.cpp && ./ex14

#include <iostream>
#include <string>
#include <cassert>

// TODO 1: Split this code into a proper project structure:
//   project/
//   ├── CMakeLists.txt           (top-level)
//   ├── src/
//   │   ├── main.cpp
//   │   └── calculator.cpp
//   ├── include/
//   │   └── calculator.hpp
//   └── tests/
//       ├── CMakeLists.txt
//       └── test_calculator.cpp
//
// Write the CMakeLists.txt that:
//   - Sets minimum cmake version 3.20
//   - Sets C++ standard to 20
//   - Creates a library target "calculator_lib" from calculator.cpp
//   - Creates an executable "calculator" linking calculator_lib
//   - Creates a test executable "test_calculator"
//   - Uses enable_testing() and add_test()

// --- Calculator (inline for standalone compilation) ---

namespace calc {

double add(double a, double b) { return a + b; }
double subtract(double a, double b) { return a - b; }
double multiply(double a, double b) { return a * b; }

double divide(double a, double b) {
    if (b == 0.0) throw std::invalid_argument("Division by zero");
    return a / b;
}

double power(double base, int exp) {
    double result = 1.0;
    bool neg = exp < 0;
    if (neg) exp = -exp;
    for (int i = 0; i < exp; ++i) result *= base;
    return neg ? 1.0 / result : result;
}

}  // namespace calc

// TODO 2: Write a CMakeLists.txt (as a comment or separate file) for this project.
// Paste your CMakeLists.txt content as comments below:

/*
cmake_minimum_required(VERSION 3.20)
project(calculator LANGUAGES CXX)

# TODO: Complete the CMakeLists.txt
*/

// TODO 3: Add an install() rule to your CMakeLists.txt that:
//   - Installs the executable to bin/
//   - Installs the header to include/

// TODO 4: Add a find_package() for a common library (e.g., fmt or nlohmann_json)
//   and link it to your target. (Describe in comments what you would add.)

int main() {
    std::cout << "=== Exercise 14: CMake and Build Basics ===\n\n";

    // Test calculator functions
    assert(calc::add(3, 4) == 7);
    assert(calc::subtract(10, 3) == 7);
    assert(calc::multiply(3, 4) == 12);
    assert(calc::divide(10, 4) == 2.5);
    assert(calc::power(2, 10) == 1024);

    try {
        calc::divide(1, 0);
        assert(false);
    } catch (const std::invalid_argument&) {
        // expected
    }

    std::cout << "All calculator tests passed.\n";
    std::cout << "\nThe real exercise: create the project structure\n";
    std::cout << "and CMakeLists.txt as described in the TODOs above.\n";

    return 0;
}
