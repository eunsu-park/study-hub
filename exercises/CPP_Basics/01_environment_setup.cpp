// Exercise 01: Environment Setup
// Verify your C++ toolchain is working correctly.
// Compile: g++ -std=c++20 -Wall -Wextra -o ex01 01_environment_setup.cpp && ./ex01

#include <iostream>
#include <string>

// TODO 1: Write a function that returns your compiler version as a string.
// Hint: Use __cplusplus, __GNUC__, __clang_major__ macros.
std::string compiler_info() {
    // TODO: Implement
    return "";
}

// TODO 2: Write a function that prints the sizes (in bytes) of fundamental types:
// bool, char, short, int, long, long long, float, double, long double
void print_type_sizes() {
    // TODO: Implement
}

// TODO 3: Write a function that checks if the system is little-endian or big-endian.
// Hint: Store a multi-byte integer and check the first byte.
bool is_little_endian() {
    // TODO: Implement
    return true;
}

// TODO 4: Write a constexpr function that computes the sum 1 + 2 + ... + n.
// Verify it works at compile time with static_assert.
constexpr int sum_to(int n) {
    // TODO: Implement
    return 0;
}

int main() {
    std::cout << "=== Exercise 01: Environment Setup ===\n\n";

    // Test 1
    std::cout << "Compiler: " << compiler_info() << '\n';

    // Test 2
    std::cout << "\nType sizes:\n";
    print_type_sizes();

    // Test 3
    std::cout << "\nEndianness: "
              << (is_little_endian() ? "little-endian" : "big-endian") << '\n';

    // Test 4
    // static_assert(sum_to(10) == 55, "sum_to(10) should be 55");
    std::cout << "\nsum_to(100) = " << sum_to(100) << '\n';

    return 0;
}
