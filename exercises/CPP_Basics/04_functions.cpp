// Exercise 04: Functions
// Practice function overloading, default parameters, and pass-by-reference.
// Compile: g++ -std=c++20 -Wall -Wextra -o ex04 04_functions.cpp && ./ex04

#include <iostream>
#include <string>
#include <vector>
#include <cassert>

// TODO 1: Write overloaded "max_val" functions for int, double, and std::string.
// For strings, return the lexicographically greater one.

// int max_val(int a, int b) { ... }
// double max_val(double a, double b) { ... }
// std::string max_val(const std::string& a, const std::string& b) { ... }

// TODO 2: Write a function "repeat" with a default parameter.
// repeat("Hi", 3) -> "HiHiHi"
// repeat("Hi")     -> "Hi" (default count = 1)

// std::string repeat(const std::string& s, int count = 1) { ... }

// TODO 3: Write a function that takes a vector by reference and removes
// all negative numbers. Return the count of removed elements.

// int remove_negatives(std::vector<int>& v) { ... }

// TODO 4: Write a function template that works with any numeric type.
// It should return the average of a vector<T>.

// template <typename T>
// double average(const std::vector<T>& v) { ... }

// TODO 5: Write a lambda that captures a multiplier and returns
// a function that multiplies its argument by that multiplier.

// auto make_multiplier(int factor) { ... }

int main() {
    std::cout << "=== Exercise 04: Functions ===\n\n";

    // Test 1: Overloading
    // assert(max_val(3, 7) == 7);
    // assert(max_val(3.14, 2.72) == 3.14);
    // assert(max_val(std::string("apple"), std::string("banana")) == "banana");
    // std::cout << "Test 1 passed: max_val overloading\n";

    // Test 2: Default parameters
    // assert(repeat("Ha", 3) == "HaHaHa");
    // assert(repeat("X") == "X");
    // std::cout << "Test 2 passed: repeat with default\n";

    // Test 3: Pass by reference
    // std::vector<int> v = {3, -1, 4, -5, 2, -3};
    // int removed = remove_negatives(v);
    // assert(removed == 3);
    // assert(v.size() == 3);
    // std::cout << "Test 3 passed: remove_negatives\n";

    // Test 4: Template function
    // std::vector<double> dv = {1.0, 2.0, 3.0, 4.0};
    // assert(average(dv) == 2.5);
    // std::cout << "Test 4 passed: average template\n";

    // Test 5: Lambda factory
    // auto times3 = make_multiplier(3);
    // assert(times3(5) == 15);
    // assert(times3(10) == 30);
    // std::cout << "Test 5 passed: make_multiplier lambda\n";

    std::cout << "\nUncomment tests as you implement each function.\n";
    return 0;
}
