// Exercise 08: Classes Basics
// Practice class design, constructors, destructors, and encapsulation.
// Compile: g++ -std=c++20 -Wall -Wextra -o ex08 08_classes_basics.cpp && ./ex08

#include <iostream>
#include <string>
#include <vector>
#include <cassert>

// TODO 1: Implement a "Date" class with:
// - Private members: year_, month_, day_
// - Constructor with validation (throw invalid_argument for bad dates)
// - Getters for year, month, day
// - to_string() returning "YYYY-MM-DD" format
// - is_leap_year() method
// - A static method days_in_month(int year, int month)

class Date {
    // TODO: Implement
};

// TODO 2: Implement a "Stack" class (int stack) with:
// - Private std::vector<int> as underlying storage
// - push(int), pop() -> int, top() -> int, empty() -> bool, size() -> size_t
// - pop() and top() should throw std::runtime_error if empty

class Stack {
    // TODO: Implement
};

// TODO 3: Implement a "Counter" class with:
// - Private count (default 0)
// - increment(), decrement() (don't go below 0), reset()
// - get() const
// - Static member tracking total increments across ALL Counter instances
// - Static method total_increments()

class Counter {
    // TODO: Implement
};

int main() {
    std::cout << "=== Exercise 08: Classes Basics ===\n\n";

    // Test 1: Date
    // Date d(2024, 2, 29);
    // assert(d.to_string() == "2024-02-29");
    // assert(d.is_leap_year() == true);
    // assert(Date::days_in_month(2024, 2) == 29);
    // assert(Date::days_in_month(2023, 2) == 28);
    // try { Date bad(2023, 2, 29); assert(false); }
    // catch (const std::invalid_argument&) {}
    // std::cout << "Test 1 passed: Date class\n";

    // Test 2: Stack
    // Stack s;
    // assert(s.empty());
    // s.push(10); s.push(20); s.push(30);
    // assert(s.size() == 3);
    // assert(s.top() == 30);
    // assert(s.pop() == 30);
    // assert(s.size() == 2);
    // try { Stack empty_s; empty_s.pop(); assert(false); }
    // catch (const std::runtime_error&) {}
    // std::cout << "Test 2 passed: Stack class\n";

    // Test 3: Counter
    // Counter c1, c2;
    // c1.increment(); c1.increment(); c1.increment();
    // c2.increment(); c2.increment();
    // assert(c1.get() == 3);
    // assert(c2.get() == 2);
    // assert(Counter::total_increments() == 5);
    // c1.decrement();
    // assert(c1.get() == 2);
    // c1.reset();
    // assert(c1.get() == 0);
    // std::cout << "Test 3 passed: Counter class\n";

    std::cout << "Uncomment tests as you implement each class.\n";
    return 0;
}
