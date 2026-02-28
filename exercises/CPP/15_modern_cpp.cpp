/*
 * Exercises for Lesson 15: Modern C++ (C++11/14/17/20)
 * Topic: CPP
 * Compile: g++ -std=c++17 -Wall -Wextra -o ex15 15_modern_cpp.cpp
 */
#include <iostream>
#include <vector>
#include <string>
#include <optional>
#include <variant>
#include <map>
#include <any>
#include <algorithm>
#include <numeric>
#include <sstream>
#include <functional>
#include <cassert>
using namespace std;

// === Exercise 1: Modern C++ Refactoring ===
// Problem: Refactor C++98/03-style code to modern C++17 style.
//          Demonstrate: auto, range-for, structured bindings, optional,
//          if-init, emplace, string_view, and more.

namespace OldStyle {
    // C++98 style: manual iterator loops, explicit types, NULL
    void processData() {
        // Old: explicit type, push_back with copy
        vector<pair<string, int> > scores;
        scores.push_back(make_pair(string("Alice"), 95));
        scores.push_back(make_pair(string("Bob"), 87));
        scores.push_back(make_pair(string("Charlie"), 92));

        // Old: manual iterator loop
        for (vector<pair<string, int> >::const_iterator it = scores.begin();
             it != scores.end(); ++it) {
            cout << "    " << it->first << ": " << it->second << endl;
        }

        // Old: find with manual comparison
        string target = "Bob";
        int* found = NULL;
        for (size_t i = 0; i < scores.size(); i++) {
            if (scores[i].first == target) {
                found = &scores[i].second;
                break;
            }
        }
        if (found != NULL) {
            cout << "    Found " << target << ": " << *found << endl;
        }
    }
}

namespace ModernStyle {
    // Helper: find a score by name, return optional
    optional<int> findScore(const vector<pair<string, int>>& scores,
                            string_view name) {
        // Use find_if with lambda instead of manual loop
        auto it = find_if(scores.begin(), scores.end(),
            [name](const auto& p) { return p.first == name; });

        if (it != scores.end()) {
            return it->second;
        }
        return nullopt;
    }

    void processData() {
        // Modern: brace initialization, emplace_back (no make_pair needed)
        vector<pair<string, int>> scores;
        scores.emplace_back("Alice", 95);
        scores.emplace_back("Bob", 87);
        scores.emplace_back("Charlie", 92);

        // Modern: structured bindings in range-for
        for (const auto& [name, score] : scores) {
            cout << "    " << name << ": " << score << endl;
        }

        // Modern: optional return + if-with-initializer (C++17)
        if (auto result = findScore(scores, "Bob"); result.has_value()) {
            cout << "    Found Bob: " << *result << endl;
        }

        // Demonstrate value_or for missing keys
        auto missing = findScore(scores, "Zara");
        cout << "    Zara's score: " << missing.value_or(-1)
             << " (default)" << endl;
    }
}

void exercise_1() {
    cout << "=== Exercise 1: Modern C++ Refactoring ===" << endl;

    cout << "  Old C++98 style:" << endl;
    OldStyle::processData();

    cout << "\n  Modern C++17 style:" << endl;
    ModernStyle::processData();

    cout << "\n  Key improvements:" << endl;
    cout << "  - auto + structured bindings reduce boilerplate" << endl;
    cout << "  - optional replaces nullable pointers for missing values" << endl;
    cout << "  - emplace_back avoids unnecessary copies" << endl;
    cout << "  - string_view avoids string copies for read-only access" << endl;
    cout << "  - if-with-initializer limits variable scope" << endl;
}

// === Exercise 2: Type-Safe Configuration System ===
// Problem: Implement a configuration manager using std::variant and
//          std::optional. Supports string, int, double, bool value types.

class Config {
public:
    // A config value can be one of these types
    using Value = variant<string, int, double, bool>;

private:
    map<string, Value> store_;

public:
    // Set a configuration value
    void set(const string& key, Value value) {
        store_[key] = std::move(value);
    }

    // Get a value with type safety. Returns optional if key exists
    // and holds the requested type, nullopt otherwise.
    template<typename T>
    optional<T> get(const string& key) const {
        auto it = store_.find(key);
        if (it == store_.end()) return nullopt;

        // Use get_if to safely extract the type
        if (auto* ptr = std::get_if<T>(&it->second)) {
            return *ptr;
        }
        return nullopt;  // Key exists but type mismatch
    }

    // Check if a key exists
    bool has(const string& key) const {
        return store_.count(key) > 0;
    }

    // Get the type name of a stored value
    string typeName(const string& key) const {
        auto it = store_.find(key);
        if (it == store_.end()) return "(not found)";

        // visit with overloaded lambdas to determine type
        return visit([](const auto& val) -> string {
            using T = decay_t<decltype(val)>;
            if constexpr (is_same_v<T, string>) return "string";
            else if constexpr (is_same_v<T, int>) return "int";
            else if constexpr (is_same_v<T, double>) return "double";
            else if constexpr (is_same_v<T, bool>) return "bool";
            else return "unknown";
        }, it->second);
    }

    // Convert any value to string for display
    string toString(const string& key) const {
        auto it = store_.find(key);
        if (it == store_.end()) return "(not set)";

        return visit([](const auto& val) -> string {
            using T = decay_t<decltype(val)>;
            if constexpr (is_same_v<T, string>) return val;
            else if constexpr (is_same_v<T, bool>) return val ? "true" : "false";
            else return to_string(val);
        }, it->second);
    }

    // Print all configuration
    void dump() const {
        for (const auto& [key, value] : store_) {
            cout << "    " << key << " (" << typeName(key)
                 << ") = " << toString(key) << endl;
        }
    }
};

void exercise_2() {
    cout << "\n=== Exercise 2: Type-Safe Configuration System ===" << endl;

    Config config;

    // Set various typed values
    config.set("app.name", string("MyApp"));
    config.set("app.port", 8080);
    config.set("app.rate", 0.75);
    config.set("app.debug", true);

    // Dump all
    cout << "  All config:" << endl;
    config.dump();

    // Type-safe retrieval
    cout << "\n  Type-safe get:" << endl;

    auto name = config.get<string>("app.name");
    cout << "    app.name (string): " << name.value_or("N/A") << endl;

    auto port = config.get<int>("app.port");
    cout << "    app.port (int): " << port.value_or(-1) << endl;

    // Type mismatch: port is int, not string
    auto portAsString = config.get<string>("app.port");
    cout << "    app.port as string: "
         << portAsString.value_or("(type mismatch)") << endl;

    // Missing key
    auto missing = config.get<int>("app.timeout");
    cout << "    app.timeout: " << missing.value_or(30)
         << " (default)" << endl;
}

// === Exercise 3: Pipeline Processor ===
// Problem: Implement a data processing pipeline using function composition.
//          (Using C++17 since C++20 ranges may not be available everywhere)

template<typename T>
class Pipeline {
    vector<T> data_;

public:
    explicit Pipeline(vector<T> data) : data_(std::move(data)) {}

    // Filter: keep elements matching predicate
    Pipeline& filter(function<bool(const T&)> pred) {
        data_.erase(
            remove_if(data_.begin(), data_.end(),
                      [&pred](const T& x) { return !pred(x); }),
            data_.end()
        );
        return *this;
    }

    // Transform: apply function to each element (in-place)
    Pipeline& transform(function<T(const T&)> fn) {
        std::transform(data_.begin(), data_.end(), data_.begin(), fn);
        return *this;
    }

    // Sort with custom comparator
    Pipeline& sort(function<bool(const T&, const T&)> cmp = less<T>{}) {
        std::sort(data_.begin(), data_.end(), cmp);
        return *this;
    }

    // Take first n elements
    Pipeline& take(size_t n) {
        if (n < data_.size()) {
            data_.resize(n);
        }
        return *this;
    }

    // Reduce to a single value
    T reduce(T init, function<T(const T&, const T&)> fn) const {
        return accumulate(data_.begin(), data_.end(), init, fn);
    }

    // Collect results
    vector<T> collect() const { return data_; }

    // For each
    void forEach(function<void(const T&)> fn) const {
        for (const auto& item : data_) {
            fn(item);
        }
    }

    size_t size() const { return data_.size(); }
};

void exercise_3() {
    cout << "\n=== Exercise 3: Pipeline Processor ===" << endl;

    // Pipeline 1: Integer processing
    cout << "  --- Integer pipeline ---" << endl;
    {
        // Generate 1..20, keep odds, square them, take top 5
        vector<int> nums(20);
        iota(nums.begin(), nums.end(), 1);  // Fill with 1..20

        auto result = Pipeline<int>(nums)
            .filter([](const int& x) { return x % 2 != 0; })   // Keep odds
            .transform([](const int& x) { return x * x; })      // Square
            .sort([](const int& a, const int& b) { return a > b; }) // Descending
            .take(5)                                              // Top 5
            .collect();

        cout << "    Top 5 squared odds: ";
        for (int v : result) cout << v << " ";
        cout << endl;

        // Sum using reduce
        int sum = Pipeline<int>(result)
            .reduce(0, [](const int& a, const int& b) { return a + b; });
        cout << "    Sum: " << sum << endl;
    }

    // Pipeline 2: String processing
    cout << "\n  --- String pipeline ---" << endl;
    {
        vector<string> words = {
            "hello", "world", "cpp", "modern", "programming",
            "template", "lambda", "auto", "concept", "ranges"
        };

        Pipeline<string>(words)
            .filter([](const string& s) { return s.size() > 4; })  // Longer than 4
            .transform([](const string& s) {                        // Uppercase
                string upper = s;
                std::transform(upper.begin(), upper.end(), upper.begin(), ::toupper);
                return upper;
            })
            .sort()                                                 // Alphabetical
            .forEach([](const string& s) {
                cout << "    " << s << endl;
            });
    }
}

int main() {
    exercise_1();
    exercise_2();
    exercise_3();
    cout << "\nAll exercises completed!" << endl;
    return 0;
}
