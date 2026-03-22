// operators_demo.cpp — Operators, if/switch, loops
// Compile: g++ -std=c++20 -Wall -Wextra -o operators_demo operators_demo.cpp

#include <iostream>
#include <string>

int main() {
    // --- Arithmetic operators ---
    std::cout << "=== Arithmetic ===\n";
    int a = 17, b = 5;
    std::cout << a << " + " << b << " = " << (a + b) << '\n';
    std::cout << a << " - " << b << " = " << (a - b) << '\n';
    std::cout << a << " * " << b << " = " << (a * b) << '\n';
    std::cout << a << " / " << b << " = " << (a / b) << " (integer)\n";
    std::cout << a << " % " << b << " = " << (a % b) << '\n';

    // --- Comparison & logical ---
    std::cout << "\n=== Comparison & Logical ===\n";
    bool p = true, q = false;
    std::cout << std::boolalpha;
    std::cout << "p && q = " << (p && q) << '\n';
    std::cout << "p || q = " << (p || q) << '\n';
    std::cout << "!p     = " << (!p) << '\n';
    std::cout << "(10 > 5) = " << (10 > 5) << '\n';

    // --- Bitwise ---
    std::cout << "\n=== Bitwise ===\n";
    unsigned x = 0b1010, y = 0b1100;
    std::cout << "x & y  = " << (x & y) << '\n';
    std::cout << "x | y  = " << (x | y) << '\n';
    std::cout << "x ^ y  = " << (x ^ y) << '\n';
    std::cout << "x << 1 = " << (x << 1) << '\n';

    // --- Ternary ---
    int score = 85;
    std::string grade = (score >= 90) ? "A" : (score >= 80) ? "B" : "C";
    std::cout << "\nScore " << score << " -> Grade " << grade << '\n';

    // --- if / else if / else ---
    std::cout << "\n=== if-else ===\n";
    int temp = 22;
    if (temp > 30) {
        std::cout << "Hot day\n";
    } else if (temp > 20) {
        std::cout << "Pleasant day\n";
    } else {
        std::cout << "Cool day\n";
    }

    // --- switch ---
    std::cout << "\n=== switch ===\n";
    int day = 3;
    switch (day) {
        case 1: std::cout << "Monday\n";    break;
        case 2: std::cout << "Tuesday\n";   break;
        case 3: std::cout << "Wednesday\n"; break;
        case 4: std::cout << "Thursday\n";  break;
        case 5: std::cout << "Friday\n";    break;
        default: std::cout << "Weekend\n";  break;
    }

    // --- Loops ---
    std::cout << "\n=== for loop ===\n";
    for (int i = 0; i < 5; ++i) {
        std::cout << i << ' ';
    }
    std::cout << '\n';

    std::cout << "\n=== range-based for ===\n";
    int arr[] = {10, 20, 30, 40, 50};
    for (auto val : arr) {
        std::cout << val << ' ';
    }
    std::cout << '\n';

    std::cout << "\n=== while loop ===\n";
    int n = 1;
    while (n <= 16) {
        std::cout << n << ' ';
        n *= 2;
    }
    std::cout << '\n';

    return 0;
}
