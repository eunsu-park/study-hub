// Exercise 09: C++20 Ranges
// Practice views, adaptors, range algorithms, and custom views.
// Compile: g++ -std=c++20 -Wall -Wextra -o ex09 09_cpp20_ranges.cpp && ./ex09

#include <iostream>
#include <vector>
#include <string>
#include <ranges>
#include <algorithm>
#include <numeric>
#include <cassert>

namespace rv = std::ranges::views;

// TODO 1: Write a function that uses ranges to find the sum of squares
// of all even numbers in a vector.
// sum_even_squares({1,2,3,4,5,6}) -> 4+16+36 = 56

// int sum_even_squares(const std::vector<int>& v) { ... }

// TODO 2: Write a function that generates the first N Fibonacci numbers using ranges.
// Hint: Use a custom view or iota with transform (accumulate state via closure).

// std::vector<long long> fibonacci_range(int n) { ... }

// TODO 3: Write a pipeline that processes strings:
// - Split words (already split in vector)
// - Filter words longer than 3 characters
// - Transform to uppercase
// - Take first 5
// Return the result as a vector<string>.

// std::vector<std::string> process_words(const std::vector<std::string>& words) { ... }

// TODO 4: Implement a ranges-based "group_by" that groups consecutive equal elements.
// group_by({1,1,2,2,2,3,1,1}) -> {{1,1},{2,2,2},{3},{1,1}}

// std::vector<std::vector<int>> group_consecutive(const std::vector<int>& v) { ... }

// TODO 5: Use ranges to implement a "sliding window" average.
// sliding_avg({1,2,3,4,5}, 3) -> {2.0, 3.0, 4.0}  (averages of [1,2,3], [2,3,4], [3,4,5])

// std::vector<double> sliding_avg(const std::vector<int>& v, int window) { ... }

int main() {
    std::cout << "=== Exercise 09: C++20 Ranges ===\n\n";

    // Test 1: sum_even_squares
    // assert(sum_even_squares({1, 2, 3, 4, 5, 6}) == 56);
    // assert(sum_even_squares({1, 3, 5}) == 0);
    // std::cout << "Test 1 passed: sum_even_squares\n";

    // Test 2: fibonacci_range
    // auto fibs = fibonacci_range(10);
    // assert(fibs.size() == 10);
    // assert(fibs[0] == 0 && fibs[1] == 1);
    // assert(fibs[9] == 34);
    // std::cout << "Test 2 passed: fibonacci_range\n";

    // Test 3: process_words
    // std::vector<std::string> words = {"the", "quick", "brown", "fox", "jumps", "over", "a", "lazy", "dog"};
    // auto result = process_words(words);
    // assert(result.size() == 5);
    // assert(result[0] == "QUICK");
    // std::cout << "Test 3 passed: process_words\n";

    // Test 4: group_consecutive
    // auto groups = group_consecutive({1, 1, 2, 2, 2, 3, 1, 1});
    // assert(groups.size() == 4);
    // assert(groups[0] == std::vector<int>{1, 1});
    // assert(groups[1] == std::vector<int>{2, 2, 2});
    // std::cout << "Test 4 passed: group_consecutive\n";

    // Test 5: sliding_avg
    // auto avgs = sliding_avg({1, 2, 3, 4, 5}, 3);
    // assert(avgs.size() == 3);
    // assert(std::abs(avgs[0] - 2.0) < 0.01);
    // assert(std::abs(avgs[2] - 4.0) < 0.01);
    // std::cout << "Test 5 passed: sliding_avg\n";

    std::cout << "Uncomment tests as you implement each function.\n";
    return 0;
}
