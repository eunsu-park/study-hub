// ranges_demo.cpp — Views, filter, transform, pipe operator (C++20)
// Compile: g++ -std=c++20 -Wall -Wextra -o ranges_demo ranges_demo.cpp

#include <iostream>
#include <vector>
#include <ranges>
#include <algorithm>
#include <string>
#include <numeric>

namespace rv = std::ranges::views;

int main() {
    std::vector<int> nums = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

    // --- Basic views with pipe operator ---
    std::cout << "=== Filter + Transform (pipe) ===\n";
    auto even_squared = nums
        | rv::filter([](int n) { return n % 2 == 0; })
        | rv::transform([](int n) { return n * n; });

    for (int x : even_squared) {
        std::cout << x << ' ';   // 4 16 36 64 100
    }
    std::cout << '\n';

    // --- views::iota (range generator) ---
    std::cout << "\n=== views::iota ===\n";
    for (int x : rv::iota(1, 11)) {
        std::cout << x << ' ';
    }
    std::cout << '\n';

    // Infinite iota with take
    std::cout << "First 5 from iota(100): ";
    for (int x : rv::iota(100) | rv::take(5)) {
        std::cout << x << ' ';
    }
    std::cout << '\n';

    // --- views::take and views::drop ---
    std::cout << "\n=== take / drop ===\n";
    std::cout << "take(3): ";
    for (int x : nums | rv::take(3)) std::cout << x << ' ';
    std::cout << '\n';

    std::cout << "drop(7): ";
    for (int x : nums | rv::drop(7)) std::cout << x << ' ';
    std::cout << '\n';

    // --- views::reverse ---
    std::cout << "\n=== reverse ===\n";
    for (int x : nums | rv::reverse | rv::take(5)) {
        std::cout << x << ' ';  // 10 9 8 7 6
    }
    std::cout << '\n';

    // --- Chained pipeline ---
    std::cout << "\n=== Complex Pipeline ===\n";
    // Squares of odd numbers > 3, take first 4
    auto result = rv::iota(1)
        | rv::filter([](int n) { return n % 2 != 0; })
        | rv::filter([](int n) { return n > 3; })
        | rv::transform([](int n) { return n * n; })
        | rv::take(4);

    for (int x : result) {
        std::cout << x << ' ';  // 25 49 81 121
    }
    std::cout << '\n';

    // --- ranges::sort (works directly on range) ---
    std::cout << "\n=== ranges::sort ===\n";
    std::vector<int> v = {5, 2, 8, 1, 9, 3};
    std::ranges::sort(v);
    for (int x : v) std::cout << x << ' ';
    std::cout << '\n';

    // Sort with projection
    std::vector<std::string> words = {"banana", "apple", "cherry", "date"};
    std::ranges::sort(words, {}, &std::string::size);
    std::cout << "By length: ";
    for (const auto& w : words) std::cout << w << ' ';
    std::cout << '\n';

    // --- ranges algorithms ---
    std::cout << "\n=== ranges algorithms ===\n";
    auto it = std::ranges::find(v, 5);
    if (it != v.end()) std::cout << "Found 5\n";

    bool all_pos = std::ranges::all_of(v, [](int x) { return x > 0; });
    std::cout << "All positive? " << std::boolalpha << all_pos << '\n';

    auto [mn, mx] = std::ranges::minmax(v);
    std::cout << "min=" << mn << " max=" << mx << '\n';

    // --- views::enumerate (C++23, emulated) ---
    std::cout << "\n=== Enumerate (index + value) ===\n";
    for (auto [i, val] : rv::iota(0) | rv::transform([&v](int i) {
        return std::pair{i, v[static_cast<size_t>(i)]};
    }) | rv::take(static_cast<int>(v.size()))) {
        std::cout << "[" << i << "]=" << val << ' ';
    }
    std::cout << '\n';

    // --- views::zip (C++23 preview — manual) ---
    std::cout << "\n=== Lazy evaluation proof ===\n";
    int eval_count = 0;
    auto lazy = rv::iota(1)
        | rv::transform([&eval_count](int n) {
            ++eval_count;
            return n * n;
        })
        | rv::take(3);

    std::cout << "Before iteration: eval_count=" << eval_count << '\n';
    for (int x : lazy) std::cout << x << ' ';
    std::cout << "\nAfter iteration: eval_count=" << eval_count << '\n';

    return 0;
}
