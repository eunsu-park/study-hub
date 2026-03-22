// Exercise 11: Modules and Utilities
// Practice C++20 module concepts and modern utility features.
// Note: Actual module compilation requires specific toolchain support.
// This exercise focuses on the design patterns modules enable.
// Compile: g++ -std=c++20 -Wall -Wextra -o ex11 11_modules_and_utilities.cpp && ./ex11

#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <functional>
#include <cassert>
#include <cmath>
#include <sstream>
#include <numeric>

// TODO 1: Design a "math_module" namespace that would be a C++20 module.
// Implement these exported functions:
//   - gcd(int, int), lcm(int, int)
//   - is_prime(int) -> bool
//   - prime_factors(int) -> vector<int>
//   - combinations(int n, int r) -> long long

// namespace math_module { ... }

// TODO 2: Design a "string_module" namespace with:
//   - trim(string_view) -> string  (remove leading/trailing whitespace)
//   - split(string_view, char) -> vector<string>
//   - join(vector<string>, string_view) -> string
//   - replace_all(string, string_view, string_view) -> string
//   - starts_with(string_view, string_view) -> bool
//   - ends_with(string_view, string_view) -> bool

// namespace string_module { ... }

// TODO 3: Design a "container_module" with generic utilities:
//   - contains(Container, Value) -> bool
//   - transform_values(map, Func) -> map with transformed values
//   - zip(vector<A>, vector<B>) -> vector<pair<A,B>>
//   - enumerate(Container) -> vector<pair<size_t, Value>>

// namespace container_module { ... }

// TODO 4: Write module interface declarations (as comments).
// Show what the .cppm file would look like for math_module.
// Include: export module, export namespace, import declarations.

/*
// math_module.cppm
export module math_module;

// TODO: Write the module interface
*/

// TODO 5: Design a module partition structure (as comments).
// Split a "network" module into partitions:
//   network:socket, network:http, network:dns

/*
// network-socket.cppm
export module network:socket;
// ...

// network-http.cppm
export module network:http;
import :socket;
// ...
*/

int main() {
    std::cout << "=== Exercise 11: Modules and Utilities ===\n\n";

    // Test 1: math_module
    // assert(math_module::gcd(12, 8) == 4);
    // assert(math_module::lcm(4, 6) == 12);
    // assert(math_module::is_prime(17) == true);
    // assert(math_module::is_prime(15) == false);
    // assert((math_module::prime_factors(12) == std::vector<int>{2, 2, 3}));
    // assert(math_module::combinations(5, 2) == 10);
    // std::cout << "Test 1 passed: math_module\n";

    // Test 2: string_module
    // assert(string_module::trim("  hello  ") == "hello");
    // auto parts = string_module::split("a,b,c", ',');
    // assert(parts.size() == 3 && parts[0] == "a");
    // assert(string_module::join({"x", "y", "z"}, "-") == "x-y-z");
    // assert(string_module::replace_all("aabaa", "aa", "X") == "XbX");
    // assert(string_module::starts_with("hello", "hel"));
    // std::cout << "Test 2 passed: string_module\n";

    // Test 3: container_module
    // std::vector<int> v = {1, 2, 3, 4, 5};
    // assert(container_module::contains(v, 3));
    // assert(!container_module::contains(v, 9));
    // auto zipped = container_module::zip(
    //     std::vector<int>{1,2,3}, std::vector<std::string>{"a","b","c"});
    // assert(zipped[0].first == 1 && zipped[0].second == "a");
    // std::cout << "Test 3 passed: container_module\n";

    std::cout << "Uncomment tests as you implement each module.\n";
    return 0;
}
