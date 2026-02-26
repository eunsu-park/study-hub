/**
 * Unit tests for Calculator library.
 *
 * Uses <cassert> for simplicity (no external test framework needed).
 * In production, prefer Google Test with gtest_discover_tests().
 *
 * Build & Run:
 *   cmake -B build && cmake --build build && cd build && ctest -V
 */

#include "calculator.hpp"
#include <cassert>
#include <cmath>
#include <iostream>
#include <string>

static int tests_run = 0;
static int tests_passed = 0;

#define TEST(name) do { \
    tests_run++; \
    std::cout << "  " << std::left << std::setw(45) << #name; \
    name(); \
    tests_passed++; \
    std::cout << "PASS" << std::endl; \
} while(0)

// Why: floating-point comparison with a tolerance is essential because binary representation
// of decimals introduces rounding errors — direct == comparison would cause false failures
#define ASSERT_NEAR(a, b, tol) \
    assert(std::fabs((a) - (b)) < (tol))

// ── Basic Arithmetic Tests ──────────────────────────

void test_add() {
    calc::Calculator c;
    ASSERT_NEAR(c.add(2, 3), 5.0, 1e-9);
    ASSERT_NEAR(c.add(-1, 1), 0.0, 1e-9);
    ASSERT_NEAR(c.add(0, 0), 0.0, 1e-9);
    ASSERT_NEAR(c.add(-3, -7), -10.0, 1e-9);
}

void test_subtract() {
    calc::Calculator c;
    ASSERT_NEAR(c.subtract(10, 4), 6.0, 1e-9);
    ASSERT_NEAR(c.subtract(0, 0), 0.0, 1e-9);
    ASSERT_NEAR(c.subtract(3, 5), -2.0, 1e-9);
}

void test_multiply() {
    calc::Calculator c;
    ASSERT_NEAR(c.multiply(6, 7), 42.0, 1e-9);
    ASSERT_NEAR(c.multiply(0, 100), 0.0, 1e-9);
    ASSERT_NEAR(c.multiply(-3, 4), -12.0, 1e-9);
}

void test_divide() {
    calc::Calculator c;
    ASSERT_NEAR(c.divide(15, 4), 3.75, 1e-9);
    ASSERT_NEAR(c.divide(10, 3), 3.333333, 1e-5);
    ASSERT_NEAR(c.divide(-12, 4), -3.0, 1e-9);
}

// Why: testing that exceptions ARE thrown is as important as testing return values —
// this pattern verifies the error path works correctly, not just the happy path
void test_divide_by_zero() {
    calc::Calculator c;
    bool caught = false;
    try {
        c.divide(1, 0);
    } catch (const std::invalid_argument&) {
        caught = true;
    }
    assert(caught);
}

// ── Accumulated Operations ──────────────────────────

void test_accumulate() {
    calc::Calculator c;
    c.set_result(100);
    assert(c.result() == 100.0);

    c.apply_add(50);
    ASSERT_NEAR(c.result(), 150.0, 1e-9);

    c.apply_multiply(2);
    ASSERT_NEAR(c.result(), 300.0, 1e-9);

    c.clear();
    ASSERT_NEAR(c.result(), 0.0, 1e-9);
}

// ── History ─────────────────────────────────────────

void test_history() {
    calc::Calculator c;
    c.add(1, 2);
    c.multiply(3, 4);
    assert(c.history().size() == 2);

    assert(c.history()[0].op == "add");
    ASSERT_NEAR(c.history()[0].result, 3.0, 1e-9);

    assert(c.history()[1].op == "mul");
    ASSERT_NEAR(c.history()[1].result, 12.0, 1e-9);

    c.clear_history();
    assert(c.history().empty());
}

// ── Free Functions ──────────────────────────────────

void test_factorial() {
    assert(calc::factorial(0) == 1);
    assert(calc::factorial(1) == 1);
    assert(calc::factorial(5) == 120);
    assert(calc::factorial(10) == 3628800);
    assert(calc::factorial(20) == 2432902008176640000ULL);
}

void test_factorial_overflow() {
    bool caught = false;
    try {
        calc::factorial(21);
    } catch (const std::overflow_error&) {
        caught = true;
    }
    assert(caught);
}

void test_power() {
    ASSERT_NEAR(calc::power(2, 10), 1024.0, 1e-9);
    ASSERT_NEAR(calc::power(3, 0), 1.0, 1e-9);
    ASSERT_NEAR(calc::power(5, 1), 5.0, 1e-9);
    ASSERT_NEAR(calc::power(2, -3), 0.125, 1e-9);
}

// ── Main ────────────────────────────────────────────

#include <iomanip>

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "  Calculator Library Tests" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;

    std::cout << "Arithmetic:" << std::endl;
    TEST(test_add);
    TEST(test_subtract);
    TEST(test_multiply);
    TEST(test_divide);
    TEST(test_divide_by_zero);

    std::cout << std::endl << "Accumulation:" << std::endl;
    TEST(test_accumulate);

    std::cout << std::endl << "History:" << std::endl;
    TEST(test_history);

    std::cout << std::endl << "Free Functions:" << std::endl;
    TEST(test_factorial);
    TEST(test_factorial_overflow);
    TEST(test_power);

    std::cout << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "  Results: " << tests_passed << "/" << tests_run
              << " tests passed" << std::endl;
    std::cout << "========================================" << std::endl;

    // Why: returning non-zero on failure lets CTest (and CI systems) automatically detect
    // test failures — CTest checks the exit code of each test executable
    return (tests_passed == tests_run) ? 0 : 1;
}
