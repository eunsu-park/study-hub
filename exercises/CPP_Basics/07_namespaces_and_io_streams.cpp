// Exercise 07: Namespaces and I/O Streams
// Practice namespaces, formatted output, and stringstream parsing.
// Compile: g++ -std=c++20 -Wall -Wextra -o ex07 07_namespaces_and_io_streams.cpp && ./ex07

#include <iostream>
#include <sstream>
#include <iomanip>
#include <string>
#include <vector>
#include <cassert>

// TODO 1: Create a namespace "convert" with functions:
//   - celsius_to_fahrenheit(double c) -> double
//   - fahrenheit_to_celsius(double f) -> double
//   - km_to_miles(double km) -> double

// namespace convert { ... }

// TODO 2: Write a function that formats a price table using iomanip.
// Each row: left-aligned name (20 chars), right-aligned price ($XX.XX).
// Return the formatted table as a string.

struct Product {
    std::string name;
    double price;
};

// std::string format_price_table(const std::vector<Product>& products) { ... }

// TODO 3: Write a function that parses a "key=value" config string.
// Input: "host=localhost\nport=8080\nverbose=true"
// Return: vector of {key, value} pairs.

// std::vector<std::pair<std::string, std::string>>
// parse_config(const std::string& config) { ... }

// TODO 4: Write a function that reads comma-separated integers from a string
// and returns them as a vector.
// Input: "10, 20, 30, 40" -> {10, 20, 30, 40}

// std::vector<int> parse_csv_ints(const std::string& csv) { ... }

// TODO 5: Create a nested namespace geometry::shapes with a function
// describe_circle(double r) that returns a formatted string:
// "Circle(r=5.00): area=78.54, circumference=31.42"

// namespace geometry::shapes { ... }

int main() {
    std::cout << "=== Exercise 07: Namespaces and I/O Streams ===\n\n";

    // Test 1: convert namespace
    // assert(std::abs(convert::celsius_to_fahrenheit(100.0) - 212.0) < 0.01);
    // assert(std::abs(convert::fahrenheit_to_celsius(32.0) - 0.0) < 0.01);
    // assert(std::abs(convert::km_to_miles(1.0) - 0.621371) < 0.001);
    // std::cout << "Test 1 passed: convert namespace\n";

    // Test 2: format_price_table
    // std::vector<Product> products = {{"Widget", 9.99}, {"Gadget", 24.50}, {"Thingamajig", 149.95}};
    // std::string table = format_price_table(products);
    // std::cout << table;
    // std::cout << "Test 2: check formatted output above\n";

    // Test 3: parse_config
    // auto config = parse_config("host=localhost\nport=8080\nverbose=true");
    // assert(config.size() == 3);
    // assert(config[0].first == "host" && config[0].second == "localhost");
    // std::cout << "Test 3 passed: parse_config\n";

    // Test 4: parse_csv_ints
    // auto nums = parse_csv_ints("10, 20, 30, 40");
    // assert(nums.size() == 4 && nums[0] == 10 && nums[3] == 40);
    // std::cout << "Test 4 passed: parse_csv_ints\n";

    // Test 5: geometry::shapes
    // std::string desc = geometry::shapes::describe_circle(5.0);
    // std::cout << desc << '\n';
    // std::cout << "Test 5: check formatted output above\n";

    std::cout << "Uncomment tests as you implement each function.\n";
    return 0;
}
