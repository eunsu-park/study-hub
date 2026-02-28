/*
 * Exercises for Lesson 02: Variables and Types
 * Topic: CPP
 * Compile: g++ -std=c++17 -Wall -Wextra -o ex02 02_variables_and_types.cpp
 */
#include <iostream>
#include <iomanip>
#include <string>
#include <cstdint>
using namespace std;

// === Exercise 1: Variable Output ===
// Problem: Declare variables of various types (int, double, char, bool,
//          string, long long) and print their values with descriptive labels.
void exercise_1() {
    cout << "=== Exercise 1: Variable Output ===" << endl;

    int age = 30;
    double height = 175.5;
    char grade = 'A';
    bool isStudent = true;
    string name = "Alice";
    long long population = 8'000'000'000LL;

    cout << "Name: " << name << endl;
    cout << "Age: " << age << endl;
    cout << "Height: " << height << " cm" << endl;
    cout << "Grade: " << grade << endl;
    cout << boolalpha;
    cout << "Is student: " << isStudent << endl;
    cout << "World population: " << population << endl;

    // Demonstrate brace initialization (C++11) to catch narrowing
    int safe{42};           // OK
    // int bad{3.14};       // Would cause compile error: narrowing conversion
    double precise{3.14};   // OK

    cout << "\nBrace-initialized int: " << safe << endl;
    cout << "Brace-initialized double: " << precise << endl;

    // Using auto for type deduction
    auto inferredInt = 100;
    auto inferredDouble = 99.9;
    auto inferredChar = 'Z';
    auto inferredBool = false;

    cout << "\nauto int: " << inferredInt << endl;
    cout << "auto double: " << inferredDouble << endl;
    cout << "auto char: " << inferredChar << endl;
    cout << "auto bool: " << inferredBool << endl;
}

// === Exercise 2: Celsius to Fahrenheit Conversion ===
// Problem: Write a program that converts Celsius to Fahrenheit.
//          Formula: F = C * 9/5 + 32
//          Use proper type casting to avoid integer division.
void exercise_2() {
    cout << "\n=== Exercise 2: Celsius to Fahrenheit ===" << endl;

    // Test with several temperatures to show type conversion behavior
    double temperatures[] = {0.0, 100.0, 37.0, -40.0, 22.5};

    cout << fixed << setprecision(2);
    cout << left << setw(15) << "Celsius" << setw(15) << "Fahrenheit" << endl;
    cout << string(30, '-') << endl;

    for (double celsius : temperatures) {
        // The key insight: 9/5 would give 1 (integer division).
        // We use 9.0/5.0 or static_cast to ensure floating-point division.
        double fahrenheit = celsius * static_cast<double>(9) / 5 + 32;
        cout << left << setw(15) << celsius << setw(15) << fahrenheit << endl;
    }

    // Demonstrate the integer division pitfall
    int c = 37;
    int wrongF = c * 9 / 5 + 32;                    // Integer arithmetic: 37*9=333, 333/5=66, +32=98
    double correctF = static_cast<double>(c) * 9 / 5 + 32;  // 37.0*9=333.0, /5=66.6, +32=98.6

    cout << "\nInteger division pitfall:" << endl;
    cout << "  Wrong (int math):  " << c << " C = " << wrongF << " F" << endl;
    cout << "  Correct (cast):    " << c << " C = " << correctF << " F" << endl;
}

// === Exercise 3: sizeof for All Basic Types ===
// Problem: Print the size in bytes of all basic C++ types, including
//          fixed-width types from <cstdint>.
void exercise_3() {
    cout << "\n=== Exercise 3: sizeof All Basic Types ===" << endl;

    cout << left << setw(20) << "Type" << setw(10) << "Size (bytes)" << endl;
    cout << string(30, '-') << endl;

    // Fundamental types
    cout << setw(20) << "bool" << setw(10) << sizeof(bool) << endl;
    cout << setw(20) << "char" << setw(10) << sizeof(char) << endl;
    cout << setw(20) << "wchar_t" << setw(10) << sizeof(wchar_t) << endl;
    cout << setw(20) << "char16_t" << setw(10) << sizeof(char16_t) << endl;
    cout << setw(20) << "char32_t" << setw(10) << sizeof(char32_t) << endl;
    cout << setw(20) << "short" << setw(10) << sizeof(short) << endl;
    cout << setw(20) << "int" << setw(10) << sizeof(int) << endl;
    cout << setw(20) << "long" << setw(10) << sizeof(long) << endl;
    cout << setw(20) << "long long" << setw(10) << sizeof(long long) << endl;
    cout << setw(20) << "float" << setw(10) << sizeof(float) << endl;
    cout << setw(20) << "double" << setw(10) << sizeof(double) << endl;
    cout << setw(20) << "long double" << setw(10) << sizeof(long double) << endl;

    cout << "\n--- Fixed-width types (<cstdint>) ---" << endl;
    cout << setw(20) << "int8_t" << setw(10) << sizeof(int8_t) << endl;
    cout << setw(20) << "int16_t" << setw(10) << sizeof(int16_t) << endl;
    cout << setw(20) << "int32_t" << setw(10) << sizeof(int32_t) << endl;
    cout << setw(20) << "int64_t" << setw(10) << sizeof(int64_t) << endl;
    cout << setw(20) << "uint8_t" << setw(10) << sizeof(uint8_t) << endl;
    cout << setw(20) << "uint16_t" << setw(10) << sizeof(uint16_t) << endl;
    cout << setw(20) << "uint32_t" << setw(10) << sizeof(uint32_t) << endl;
    cout << setw(20) << "uint64_t" << setw(10) << sizeof(uint64_t) << endl;

    // Pointer and array sizes
    cout << "\n--- Pointer and array sizes ---" << endl;
    cout << setw(20) << "int*" << setw(10) << sizeof(int*) << endl;
    cout << setw(20) << "double*" << setw(10) << sizeof(double*) << endl;
    cout << setw(20) << "void*" << setw(10) << sizeof(void*) << endl;

    int arr[10];
    cout << setw(20) << "int[10]" << setw(10) << sizeof(arr) << endl;
    cout << setw(20) << "  (elements)" << setw(10)
         << sizeof(arr) / sizeof(arr[0]) << endl;

    // constexpr verification
    constexpr size_t intSize = sizeof(int);
    static_assert(intSize >= 2, "int must be at least 2 bytes per the standard");
    cout << "\nstatic_assert passed: sizeof(int) >= 2" << endl;
}

int main() {
    exercise_1();
    exercise_2();
    exercise_3();
    cout << "\nAll exercises completed!" << endl;
    return 0;
}
