// arrays_strings_demo.cpp — C arrays, std::array, std::string operations
// Compile: g++ -std=c++20 -Wall -Wextra -o arrays_strings_demo arrays_strings_demo.cpp

#include <iostream>
#include <array>
#include <vector>
#include <string>
#include <algorithm>
#include <sstream>

int main() {
    // --- C-style array ---
    std::cout << "=== C-style Array ===\n";
    int c_arr[5] = {10, 20, 30, 40, 50};
    for (int i = 0; i < 5; ++i) {
        std::cout << c_arr[i] << ' ';
    }
    std::cout << '\n';

    // --- std::array (fixed-size, stack) ---
    std::cout << "\n=== std::array ===\n";
    std::array<int, 5> arr = {1, 2, 3, 4, 5};
    std::cout << "size=" << arr.size() << " front=" << arr.front()
              << " back=" << arr.back() << '\n';
    std::sort(arr.begin(), arr.end(), std::greater<>());
    for (auto x : arr) std::cout << x << ' ';
    std::cout << '\n';

    // --- std::vector (dynamic) ---
    std::cout << "\n=== std::vector ===\n";
    std::vector<int> v = {3, 1, 4, 1, 5, 9};
    v.push_back(2);
    v.push_back(6);
    std::cout << "size=" << v.size() << " capacity=" << v.capacity() << '\n';
    for (auto x : v) std::cout << x << ' ';
    std::cout << '\n';

    // --- std::string basics ---
    std::cout << "\n=== std::string Basics ===\n";
    std::string s1 = "Hello";
    std::string s2 = " World";
    std::string s3 = s1 + s2;          // concatenation
    std::cout << s3 << "  length=" << s3.length() << '\n';

    // Substring, find, replace
    std::cout << "\n=== String Operations ===\n";
    std::string text = "The quick brown fox jumps over the lazy dog";
    std::cout << "substr(4,5) = \"" << text.substr(4, 5) << "\"\n";
    auto pos = text.find("fox");
    std::cout << "find(\"fox\") at position " << pos << '\n';

    std::string modified = text;
    modified.replace(pos, 3, "cat");
    std::cout << "after replace: " << modified << '\n';

    // Iteration
    std::cout << "\n=== Character Iteration ===\n";
    std::string word = "C++20";
    for (char ch : word) {
        std::cout << '[' << ch << ']';
    }
    std::cout << '\n';

    // String to number and back
    std::cout << "\n=== String Conversions ===\n";
    int num = std::stoi("42");
    double dbl = std::stod("3.14");
    std::string from_num = std::to_string(num) + " and " + std::to_string(dbl);
    std::cout << from_num << '\n';

    // Splitting with stringstream
    std::cout << "\n=== Split by delimiter ===\n";
    std::string csv = "apple,banana,cherry,date";
    std::istringstream iss(csv);
    std::string token;
    while (std::getline(iss, token, ',')) {
        std::cout << "  -> " << token << '\n';
    }

    // Compare
    std::cout << "\n=== String Compare ===\n";
    std::string a = "abc", b = "abd";
    std::cout << "\"abc\" vs \"abd\": " << a.compare(b)
              << "  (negative = a < b)\n";
    std::cout << "equal? " << std::boolalpha << (a == b) << '\n';

    return 0;
}
