// algorithms_demo.cpp — sort, find, transform with lambda
// Compile: g++ -std=c++20 -Wall -Wextra -o algorithms_demo algorithms_demo.cpp

#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <string>
#include <iterator>

void print(const std::string& label, const std::vector<int>& v) {
    std::cout << label << ": ";
    for (auto x : v) std::cout << x << ' ';
    std::cout << '\n';
}

int main() {
    std::vector<int> nums = {5, 2, 8, 1, 9, 3, 7, 4, 6};
    print("Original", nums);

    // --- Sorting ---
    std::cout << "\n=== Sorting ===\n";
    std::sort(nums.begin(), nums.end());
    print("Ascending", nums);

    std::sort(nums.begin(), nums.end(), std::greater<>());
    print("Descending", nums);

    // Custom comparator
    std::vector<std::string> words = {"banana", "apple", "cherry", "date"};
    std::sort(words.begin(), words.end(),
              [](const auto& a, const auto& b) { return a.length() < b.length(); });
    std::cout << "By length: ";
    for (const auto& w : words) std::cout << w << ' ';
    std::cout << '\n';

    // --- Searching ---
    std::cout << "\n=== Searching ===\n";
    std::sort(nums.begin(), nums.end());
    auto it = std::find(nums.begin(), nums.end(), 7);
    if (it != nums.end()) {
        std::cout << "Found 7 at index " << std::distance(nums.begin(), it) << '\n';
    }

    auto it2 = std::find_if(nums.begin(), nums.end(),
                             [](int x) { return x > 6; });
    std::cout << "First > 6: " << *it2 << '\n';

    bool has_even = std::any_of(nums.begin(), nums.end(),
                                [](int x) { return x % 2 == 0; });
    std::cout << "Has even? " << std::boolalpha << has_even << '\n';

    // binary_search (requires sorted range)
    bool found = std::binary_search(nums.begin(), nums.end(), 5);
    std::cout << "binary_search(5): " << found << '\n';

    // --- Transform ---
    std::cout << "\n=== Transform ===\n";
    std::vector<int> squared(nums.size());
    std::transform(nums.begin(), nums.end(), squared.begin(),
                   [](int x) { return x * x; });
    print("Squared", squared);

    // --- Accumulate (reduce) ---
    std::cout << "\n=== Accumulate ===\n";
    int sum = std::accumulate(nums.begin(), nums.end(), 0);
    int product = std::accumulate(nums.begin(), nums.end(), 1, std::multiplies<>());
    std::cout << "Sum = " << sum << '\n';
    std::cout << "Product = " << product << '\n';

    // --- count / count_if ---
    std::cout << "\n=== Count ===\n";
    int evens = static_cast<int>(
        std::count_if(nums.begin(), nums.end(), [](int x) { return x % 2 == 0; }));
    std::cout << "Even count: " << evens << '\n';

    // --- remove_if + erase (erase-remove idiom) ---
    std::cout << "\n=== Erase-Remove Idiom ===\n";
    std::vector<int> data = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    data.erase(
        std::remove_if(data.begin(), data.end(), [](int x) { return x % 2 == 0; }),
        data.end());
    print("Odds only", data);

    // --- min/max ---
    std::cout << "\n=== Min/Max ===\n";
    auto [mn, mx] = std::minmax_element(nums.begin(), nums.end());
    std::cout << "min=" << *mn << " max=" << *mx << '\n';

    // --- for_each ---
    std::cout << "\n=== for_each ===\n";
    std::for_each(nums.begin(), nums.end(),
                  [](int x) { std::cout << x * 2 << ' '; });
    std::cout << '\n';

    return 0;
}
