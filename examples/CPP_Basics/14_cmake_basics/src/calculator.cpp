#include "calculator.hpp"
#include <cmath>

namespace calc {

double Calculator::add(double a, double b) {
    double r = a + b;
    record("add", a, b, r);
    return r;
}

double Calculator::subtract(double a, double b) {
    double r = a - b;
    record("sub", a, b, r);
    return r;
}

double Calculator::multiply(double a, double b) {
    double r = a * b;
    record("mul", a, b, r);
    return r;
}

// Why: throwing an exception for division by zero makes the error impossible to ignore —
// returning a sentinel value (like NaN) could silently propagate through calculations
double Calculator::divide(double a, double b) {
    if (b == 0.0) {
        throw std::invalid_argument("Division by zero");
    }
    double r = a / b;
    record("div", a, b, r);
    return r;
}

void Calculator::clear() {
    result_ = 0.0;
}

double Calculator::apply_add(double val) {
    result_ += val;
    record("apply_add", result_ - val, val, result_);
    return result_;
}

double Calculator::apply_multiply(double val) {
    result_ *= val;
    record("apply_mul", result_ / val, val, result_);
    return result_;
}

void Calculator::record(const std::string& op, double a, double b, double r) {
    history_.push_back({op, a, b, r});
}

// Why: unsigned long long max is ~1.8×10^19 while 21! = ~5.1×10^19, so capping at 20
// prevents silent overflow that would produce silently wrong results
unsigned long long factorial(unsigned int n) {
    if (n > 20) {
        throw std::overflow_error("Factorial overflow for n > 20");
    }
    unsigned long long result = 1;
    for (unsigned int i = 2; i <= n; ++i) {
        result *= i;
    }
    return result;
}

double power(double base, int exponent) {
    if (exponent < 0) {
        return 1.0 / power(base, -exponent);
    }
    double result = 1.0;
    for (int i = 0; i < exponent; ++i) {
        result *= base;
    }
    return result;
}

} // namespace calc
