// Exercise 09: Classes Advanced
// Practice operator overloading, copy/move semantics, and Rule of Five.
// Compile: g++ -std=c++20 -Wall -Wextra -o ex09 09_classes_advanced.cpp && ./ex09

#include <iostream>
#include <string>
#include <cassert>
#include <cstring>
#include <cmath>

// TODO 1: Implement a "Fraction" class with:
// - Numerator and denominator (always keep in reduced form using GCD)
// - Constructor Fraction(int num, int den) — throw if den == 0
// - Operators: +, -, *, /, ==, !=, <
// - Stream insertion operator<<
// - to_double() method

class Fraction {
    // TODO: Implement
};

// TODO 2: Implement a "String" class (simplified std::string) with:
// - Internal char* buffer (dynamically allocated)
// - Rule of Five: destructor, copy ctor, copy assign, move ctor, move assign
// - operator+ for concatenation
// - operator[] for character access
// - operator== for comparison
// - operator<< for printing
// - length(), c_str() methods

class String {
    // TODO: Implement
};

// TODO 3: Implement a "Matrix2x2" class with:
// - 2x2 double array
// - Constructor from 4 doubles (row-major)
// - operator+ (matrix addition)
// - operator* (matrix multiplication)
// - determinant() method
// - transpose() method returning a new Matrix2x2
// - operator<< for pretty printing

class Matrix2x2 {
    // TODO: Implement
};

int main() {
    std::cout << "=== Exercise 09: Classes Advanced ===\n\n";

    // Test 1: Fraction
    // Fraction a(1, 2), b(1, 3);
    // Fraction sum = a + b;  // should be 5/6
    // assert(sum == Fraction(5, 6));
    // Fraction prod = a * b; // should be 1/6
    // assert(prod == Fraction(1, 6));
    // assert(b < a);
    // std::cout << a << " + " << b << " = " << sum << '\n';
    // std::cout << "Test 1 passed: Fraction\n";

    // Test 2: String (Rule of Five)
    // String s1("Hello");
    // String s2(" World");
    // String s3 = s1 + s2;
    // assert(s3.length() == 11);
    // assert(s3[0] == 'H');
    // String s4 = s3;             // copy ctor
    // assert(s4 == s3);
    // String s5 = std::move(s4);  // move ctor
    // assert(s5 == s3);
    // std::cout << s5 << '\n';
    // std::cout << "Test 2 passed: String (Rule of Five)\n";

    // Test 3: Matrix2x2
    // Matrix2x2 m1(1, 2, 3, 4);
    // Matrix2x2 m2(5, 6, 7, 8);
    // Matrix2x2 sum_m = m1 + m2;  // {6,8,10,12}
    // Matrix2x2 prod_m = m1 * m2; // {19,22,43,50}
    // assert(std::abs(m1.determinant() - (-2.0)) < 1e-9);
    // std::cout << "det(m1) = " << m1.determinant() << '\n';
    // std::cout << "Test 3 passed: Matrix2x2\n";

    std::cout << "Uncomment tests as you implement each class.\n";
    return 0;
}
