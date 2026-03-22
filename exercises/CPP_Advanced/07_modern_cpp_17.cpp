// Exercise 07: Modern C++17 Features
// Practice structured bindings, optional, variant, filesystem, string_view.
// Compile: g++ -std=c++17 -Wall -Wextra -o ex07 07_modern_cpp_17.cpp && ./ex07

#include <iostream>
#include <string>
#include <string_view>
#include <map>
#include <vector>
#include <optional>
#include <variant>
#include <any>
#include <filesystem>
#include <cassert>
#include <algorithm>
#include <numeric>

namespace fs = std::filesystem;

// TODO 1: Implement a config parser that returns std::optional values.
// parse_config returns a map; get_value returns optional<string>.

class Config {
    std::map<std::string, std::string> data_;
public:
    // TODO: parse(const std::string& text) — parse "key=value\n..." format
    // void parse(const std::string& text) { ... }

    // TODO: get(key) -> optional<string>
    // std::optional<std::string> get(std::string_view key) const { ... }

    // TODO: get_int(key) -> optional<int>
    // std::optional<int> get_int(std::string_view key) const { ... }
};

// TODO 2: Implement a type-safe JSON-like value using std::variant.
// Support: null, bool, int, double, string, array, object.

// using JsonNull = std::monostate;
// using JsonValue = std::variant<JsonNull, bool, int, double, std::string,
//                                std::vector<???>, std::map<std::string, ???>>;
// (Use a simplified version with forward-declared recursive type or limit depth)

// std::string json_to_string(const JsonValue& v) { ... }

// TODO 3: Write a directory analyzer using std::filesystem.
// Return stats: total files, total dirs, total size, largest file, file type counts.

struct DirStats {
    size_t file_count = 0;
    size_t dir_count = 0;
    uintmax_t total_size = 0;
    fs::path largest_file;
    uintmax_t largest_size = 0;
    std::map<std::string, size_t> extension_counts;
};

// DirStats analyze_directory(const fs::path& dir) { ... }

// TODO 4: Implement a function using std::string_view for zero-copy parsing.
// Split a string_view into tokens without allocating new strings.

// std::vector<std::string_view> split_view(std::string_view sv, char delim) { ... }

// TODO 5: Use if-with-initializer and structured bindings to write
// a clean "find and process" function for a map.

// template <typename Map, typename Key, typename Func>
// bool find_and_apply(const Map& m, const Key& k, Func f) { ... }

int main() {
    std::cout << "=== Exercise 07: Modern C++17 ===\n\n";

    // Test 1: Config parser
    // Config cfg;
    // cfg.parse("host=localhost\nport=8080\ndebug=true\n");
    // assert(cfg.get("host").value() == "localhost");
    // assert(cfg.get_int("port").value() == 8080);
    // assert(!cfg.get("missing").has_value());
    // std::cout << "Test 1 passed: Config with optional\n";

    // Test 3: Directory analyzer
    // auto stats = analyze_directory(fs::temp_directory_path());
    // std::cout << "Files: " << stats.file_count
    //           << " Dirs: " << stats.dir_count
    //           << " Size: " << stats.total_size << '\n';
    // std::cout << "Test 3 passed: Directory analyzer\n";

    // Test 4: string_view split
    // std::string data = "one,two,three,four";
    // auto parts = split_view(data, ',');
    // assert(parts.size() == 4);
    // assert(parts[0] == "one");
    // assert(parts[3] == "four");
    // std::cout << "Test 4 passed: split_view\n";

    // Test 5: find_and_apply
    // std::map<std::string, int> scores = {{"Alice", 95}, {"Bob", 87}};
    // bool found = find_and_apply(scores, std::string("Alice"),
    //     [](const auto& name, int score) {
    //         std::cout << name << " scored " << score << '\n';
    //     });
    // assert(found);
    // std::cout << "Test 5 passed: find_and_apply\n";

    std::cout << "Uncomment tests as you implement each part.\n";
    return 0;
}
