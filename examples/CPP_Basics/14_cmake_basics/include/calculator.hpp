// Why: #pragma once is simpler than traditional include guards and avoids name collisions —
// supported by all major compilers, though not part of the C++ standard
#pragma once

#include <string>
#include <stdexcept>
#include <vector>

// Why: wrapping the library in a namespace prevents symbol collisions when this code
// is linked with other libraries that might also define common names like "add" or "power"
namespace calc {

/**
 * Basic arithmetic calculator with history tracking.
 *
 * Demonstrates:
 *   - Header-only public API
 *   - Separate implementation (.cpp)
 *   - Testable design
 */
class Calculator {
public:
    Calculator() = default;

    double add(double a, double b);
    double subtract(double a, double b);
    double multiply(double a, double b);
    double divide(double a, double b);

    // Accumulated result operations
    void clear();
    double result() const { return result_; }
    void set_result(double val) { result_ = val; }

    // Apply operation to accumulated result
    double apply_add(double val);
    double apply_multiply(double val);

    // History
    struct Entry {
        std::string op;
        double a;
        double b;
        double result;
    };

    const std::vector<Entry>& history() const { return history_; }
    void clear_history() { history_.clear(); }

private:
    double result_ = 0.0;
    std::vector<Entry> history_;
    // Why: private helper keeps the recording logic in one place — if history format
    // changes, only this method needs updating, not every arithmetic operation
    void record(const std::string& op, double a, double b, double r);
};

// Free function: factorial
unsigned long long factorial(unsigned int n);

// Free function: power (integer exponent)
double power(double base, int exponent);

} // namespace calc
