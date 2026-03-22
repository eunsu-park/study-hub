#include "calculator.hpp"
#include <iostream>
#include <iomanip>

int main() {
    calc::Calculator c;

    std::cout << "=== Calculator Demo ===" << std::endl;

    // Basic operations
    std::cout << "\nBasic operations:" << std::endl;
    std::cout << "  3 + 5 = " << c.add(3, 5) << std::endl;
    std::cout << "  10 - 4 = " << c.subtract(10, 4) << std::endl;
    std::cout << "  6 * 7 = " << c.multiply(6, 7) << std::endl;
    std::cout << "  15 / 4 = " << c.divide(15, 4) << std::endl;

    // Accumulated operations
    std::cout << "\nAccumulated operations:" << std::endl;
    c.set_result(100);
    std::cout << "  Start: " << c.result() << std::endl;
    c.apply_add(50);
    std::cout << "  +50:   " << c.result() << std::endl;
    c.apply_multiply(2);
    std::cout << "  x2:    " << c.result() << std::endl;

    // Free functions
    std::cout << "\nFactorial:" << std::endl;
    for (unsigned int n : {0u, 1u, 5u, 10u, 20u}) {
        std::cout << "  " << n << "! = " << calc::factorial(n) << std::endl;
    }

    std::cout << "\nPower:" << std::endl;
    std::cout << "  2^10 = " << calc::power(2, 10) << std::endl;
    std::cout << "  3^-2 = " << std::fixed << std::setprecision(6)
              << calc::power(3, -2) << std::endl;

    // History
    std::cout << "\nHistory (" << c.history().size() << " entries):" << std::endl;
    // Why: const auto& avoids copying each Entry struct during iteration — without const,
    // the compiler could allow accidental modification of the history
    for (const auto& entry : c.history()) {
        std::cout << "  " << entry.op << "(" << entry.a << ", " << entry.b
                  << ") = " << entry.result << std::endl;
    }

    // Why: demonstrating exception handling at the call site — catching by const reference
    // avoids slicing and the overhead of copying the exception object
    std::cout << "\nDivision by zero:" << std::endl;
    try {
        c.divide(1, 0);
    } catch (const std::invalid_argument& e) {
        std::cout << "  Caught: " << e.what() << std::endl;
    }

    return 0;
}
