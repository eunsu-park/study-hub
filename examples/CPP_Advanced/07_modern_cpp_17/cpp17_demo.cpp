// cpp17_demo.cpp — Structured bindings, optional, variant, filesystem
// Compile: g++ -std=c++17 -Wall -Wextra -o cpp17_demo cpp17_demo.cpp

#include <iostream>
#include <string>
#include <map>
#include <optional>
#include <variant>
#include <any>
#include <tuple>
#include <filesystem>
#include <algorithm>

namespace fs = std::filesystem;

// --- Structured bindings ---
void structured_bindings_demo() {
    std::cout << "=== Structured Bindings ===\n";

    // With pair
    std::map<std::string, int> scores = {{"Alice", 95}, {"Bob", 87}, {"Carol", 92}};
    for (const auto& [name, score] : scores) {
        std::cout << name << ": " << score << '\n';
    }

    // With tuple
    auto [x, y, z] = std::make_tuple(1, 2.5, "hello");
    std::cout << "tuple: " << x << ", " << y << ", " << z << '\n';

    // With struct
    struct Point { double x, y; };
    auto [px, py] = Point{3.0, 4.0};
    std::cout << "point: (" << px << ", " << py << ")\n";
}

// --- std::optional ---
std::optional<std::string> find_user(int id) {
    std::map<int, std::string> db = {{1, "Alice"}, {2, "Bob"}};
    if (auto it = db.find(id); it != db.end()) {
        return it->second;
    }
    return std::nullopt;
}

void optional_demo() {
    std::cout << "\n=== std::optional ===\n";
    for (int id : {1, 3}) {
        auto user = find_user(id);
        std::cout << "id=" << id << ": "
                  << user.value_or("(not found)") << '\n';
    }
}

// --- std::variant ---
using JsonValue = std::variant<int, double, std::string, bool>;

void print_json(const JsonValue& v) {
    std::visit([](const auto& val) {
        using T = std::decay_t<decltype(val)>;
        if constexpr (std::is_same_v<T, bool>) {
            std::cout << std::boolalpha << val;
        } else if constexpr (std::is_same_v<T, std::string>) {
            std::cout << '"' << val << '"';
        } else {
            std::cout << val;
        }
    }, v);
}

void variant_demo() {
    std::cout << "\n=== std::variant ===\n";
    std::vector<JsonValue> values = {42, 3.14, std::string("hello"), true};
    for (const auto& v : values) {
        std::cout << "  index=" << v.index() << " value=";
        print_json(v);
        std::cout << '\n';
    }
}

// --- std::any ---
void any_demo() {
    std::cout << "\n=== std::any ===\n";
    std::any a = 42;
    std::cout << "int: " << std::any_cast<int>(a) << '\n';
    a = std::string("hello");
    std::cout << "string: " << std::any_cast<std::string>(a) << '\n';
    try {
        std::any_cast<double>(a);
    } catch (const std::bad_any_cast& e) {
        std::cout << "bad_any_cast: " << e.what() << '\n';
    }
}

// --- if-init statement ---
void if_init_demo() {
    std::cout << "\n=== if-init Statement ===\n";
    std::map<std::string, int> m = {{"key", 42}};
    if (auto it = m.find("key"); it != m.end()) {
        std::cout << "Found: " << it->second << '\n';
    }
    if (auto it = m.find("missing"); it == m.end()) {
        std::cout << "Not found (as expected)\n";
    }
}

// --- std::filesystem ---
void filesystem_demo() {
    std::cout << "\n=== std::filesystem ===\n";
    fs::path p = fs::temp_directory_path() / "cpp17_demo";

    // Create directory
    fs::create_directories(p);
    std::cout << "Created: " << p << '\n';
    std::cout << "Exists: " << std::boolalpha << fs::exists(p) << '\n';

    // Path operations
    fs::path file = p / "test.txt";
    std::cout << "stem: " << file.stem() << '\n';
    std::cout << "extension: " << file.extension() << '\n';
    std::cout << "parent: " << file.parent_path() << '\n';

    // Iterate temp dir (first 5 entries)
    std::cout << "\nTemp dir entries (first 5):\n";
    int count = 0;
    for (const auto& entry : fs::directory_iterator(fs::temp_directory_path())) {
        if (++count > 5) break;
        std::cout << "  " << entry.path().filename()
                  << (entry.is_directory() ? " [dir]" : "") << '\n';
    }

    // Cleanup
    fs::remove_all(p);
}

int main() {
    structured_bindings_demo();
    optional_demo();
    variant_demo();
    any_demo();
    if_init_demo();
    filesystem_demo();
    return 0;
}
