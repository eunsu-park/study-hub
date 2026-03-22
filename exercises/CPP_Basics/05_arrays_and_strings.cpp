// Exercise 05: Arrays and Strings
// Practice std::array, std::vector, and std::string operations.
// Compile: g++ -std=c++20 -Wall -Wextra -o ex05 05_arrays_and_strings.cpp && ./ex05

#include <iostream>
#include <string>
#include <vector>
#include <array>
#include <cassert>

// TODO 1: Write a function that reverses a string in-place (without creating a new string).

// void reverse_inplace(std::string& s) { ... }

// TODO 2: Write a function that checks if a string is a palindrome (case-insensitive).
// Ignore spaces. "Race Car" -> true, "hello" -> false.

// bool is_palindrome(const std::string& s) { ... }

// TODO 3: Write a function that rotates a vector left by k positions.
// rotate_left({1,2,3,4,5}, 2) -> {3,4,5,1,2}

// void rotate_left(std::vector<int>& v, int k) { ... }

// TODO 4: Write a function that counts word frequencies in a sentence.
// Return a vector of pairs sorted by frequency (descending).
// "the cat and the dog and the fish" -> {{"the",3}, {"and",2}, {"cat",1}, ...}

// std::vector<std::pair<std::string, int>> word_freq(const std::string& text) { ... }

// TODO 5: Write a function that merges two sorted vectors into one sorted vector.

// std::vector<int> merge_sorted(const std::vector<int>& a, const std::vector<int>& b) { ... }

int main() {
    std::cout << "=== Exercise 05: Arrays and Strings ===\n\n";

    // Test 1: reverse_inplace
    // std::string s1 = "hello";
    // reverse_inplace(s1);
    // assert(s1 == "olleh");
    // std::cout << "Test 1 passed: reverse_inplace\n";

    // Test 2: is_palindrome
    // assert(is_palindrome("Race Car") == true);
    // assert(is_palindrome("hello") == false);
    // assert(is_palindrome("A man a plan a canal Panama") == true);
    // std::cout << "Test 2 passed: is_palindrome\n";

    // Test 3: rotate_left
    // std::vector<int> v = {1, 2, 3, 4, 5};
    // rotate_left(v, 2);
    // assert((v == std::vector<int>{3, 4, 5, 1, 2}));
    // std::cout << "Test 3 passed: rotate_left\n";

    // Test 4: word_freq
    // auto freq = word_freq("the cat and the dog and the fish");
    // assert(freq[0].first == "the" && freq[0].second == 3);
    // std::cout << "Test 4 passed: word_freq\n";

    // Test 5: merge_sorted
    // auto merged = merge_sorted({1, 3, 5}, {2, 4, 6});
    // assert((merged == std::vector<int>{1, 2, 3, 4, 5, 6}));
    // std::cout << "Test 5 passed: merge_sorted\n";

    std::cout << "Uncomment tests as you implement each function.\n";
    return 0;
}
