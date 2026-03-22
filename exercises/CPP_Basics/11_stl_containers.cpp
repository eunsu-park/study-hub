/*
 * Exercises for Lesson 10: STL Containers
 * Topic: CPP
 * Compile: g++ -std=c++17 -Wall -Wextra -o ex10 10_stl_containers.cpp
 */
#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <unordered_map>
#include <stack>
#include <deque>
#include <queue>
#include <unordered_set>
#include <functional>
using namespace std;

// === Exercise 1: Container Selection Justification ===
// Problem: For five scenarios, choose the most appropriate STL container
//          and justify based on time complexity and use-case fit.
void exercise_1() {
    cout << "=== Exercise 1: Container Selection Justification ===" << endl;

    cout << R"(
  1. Browser history (LIFO):
     -> std::stack<string>
     Reason: Stack provides O(1) push and pop from one end (LIFO).
     The most recently visited page is always on top.

  2. English-to-Korean dictionary (frequent key lookup, order doesn't matter):
     -> std::unordered_map<string, string>
     Reason: Average O(1) lookup by key. Since iteration order doesn't
     matter, the unordered variant is faster than std::map's O(log n).

  3. Ranked leaderboard (always need highest score):
     -> std::priority_queue<int>
     Reason: Max-heap gives O(1) access to the maximum element,
     with O(log n) insertion and extraction.

  4. To-do list (FIFO, frequent additions at both ends):
     -> std::deque<string>
     Reason: O(1) push/pop at both front and back.
     std::queue (which wraps deque) would work for pure FIFO,
     but deque is better when adding at both ends.

  5. Unique IP addresses (millions of membership checks per second):
     -> std::unordered_set<string>
     Reason: Average O(1) lookup for membership testing.
     With millions of checks, the constant-time average of hash-based
     lookup far outperforms std::set's O(log n).
)" << endl;

    // Demonstrate each briefly:
    // 1. Stack
    stack<string> history;
    history.push("google.com");
    history.push("github.com");
    history.push("cppreference.com");
    cout << "  Browser history top: " << history.top() << endl;

    // 2. Unordered map
    unordered_map<string, string> dict;
    dict["hello"] = "안녕하세요";
    dict["world"] = "세계";
    cout << "  dict[\"hello\"] = " << dict["hello"] << endl;

    // 3. Priority queue
    priority_queue<int> leaderboard;
    leaderboard.push(100);
    leaderboard.push(250);
    leaderboard.push(175);
    cout << "  Top score: " << leaderboard.top() << endl;

    // 4. Deque
    deque<string> todo;
    todo.push_back("Task 1");
    todo.push_front("Urgent Task");
    todo.push_back("Task 2");
    cout << "  First todo: " << todo.front() << endl;

    // 5. Unordered set
    unordered_set<string> ips;
    ips.insert("192.168.1.1");
    ips.insert("10.0.0.1");
    cout << "  Contains 10.0.0.1? " << boolalpha
         << (ips.count("10.0.0.1") > 0) << endl;
}

// === Exercise 2: Vector Growth Observation ===
// Problem: Push 20 integers and observe size() vs capacity() changes.
//          Then repeat with reserve(20) to see zero reallocations.
void exercise_2() {
    cout << "\n=== Exercise 2: Vector Growth Observation ===" << endl;

    // Without reserve
    cout << "--- Without reserve ---" << endl;
    {
        vector<int> v;
        size_t prevCapacity = 0;
        int reallocations = 0;

        for (int i = 1; i <= 20; i++) {
            v.push_back(i);
            if (v.capacity() != prevCapacity) {
                cout << "  [REALLOC] ";
                reallocations++;
            } else {
                cout << "  ";
            }
            cout << "push " << i
                 << "  size=" << v.size()
                 << "  capacity=" << v.capacity() << endl;
            prevCapacity = v.capacity();
        }
        cout << "  Total reallocations: " << reallocations << endl;
    }

    // With reserve
    cout << "\n--- With reserve(20) ---" << endl;
    {
        vector<int> v;
        v.reserve(20);
        size_t prevCapacity = v.capacity();
        int reallocations = 0;

        for (int i = 1; i <= 20; i++) {
            v.push_back(i);
            if (v.capacity() != prevCapacity) {
                reallocations++;
            }
            prevCapacity = v.capacity();
        }
        cout << "  After 20 pushes: size=" << v.size()
             << "  capacity=" << v.capacity() << endl;
        cout << "  Total reallocations: " << reallocations
             << " (should be 0)" << endl;
    }
}

// === Exercise 3: Word Frequency Counter ===
// Problem: Count word occurrences using map (sorted) and unordered_map.
void exercise_3() {
    cout << "\n=== Exercise 3: Word Frequency Counter ===" << endl;

    vector<string> words = {
        "the", "quick", "brown", "fox", "jumps", "over",
        "the", "lazy", "dog", "the", "fox", "the", "dog"
    };

    // Using std::map (sorted alphabetically)
    cout << "--- std::map (alphabetical order) ---" << endl;
    map<string, int> sortedCount;
    for (const auto& w : words) {
        sortedCount[w]++;
    }
    for (const auto& [word, count] : sortedCount) {
        cout << "  " << word << ": " << count << endl;
    }

    // Using std::unordered_map (arbitrary order, faster)
    cout << "\n--- std::unordered_map (insertion/hash order) ---" << endl;
    unordered_map<string, int> hashCount;
    for (const auto& w : words) {
        hashCount[w]++;
    }
    for (const auto& [word, count] : hashCount) {
        cout << "  " << word << ": " << count << endl;
    }

    cout << "\n  Note: map iterates in sorted key order." << endl;
    cout << "  unordered_map order depends on the hash function." << endl;
}

// === Exercise 4: Bracket Matching with stack ===
// Problem: Verify whether a string of brackets is balanced using std::stack.
bool isBalanced(const string& s) {
    stack<char> st;

    for (char c : s) {
        if (c == '(' || c == '[' || c == '{') {
            st.push(c);
        } else if (c == ')' || c == ']' || c == '}') {
            if (st.empty()) return false;

            char top = st.top();
            // Check if the closing bracket matches the most recent opening
            if ((c == ')' && top != '(') ||
                (c == ']' && top != '[') ||
                (c == '}' && top != '{')) {
                return false;
            }
            st.pop();
        }
    }

    // Balanced only if all opening brackets have been matched
    return st.empty();
}

void exercise_4() {
    cout << "\n=== Exercise 4: Bracket Matching ===" << endl;

    vector<pair<string, bool>> testCases = {
        {"([]{})",  true},
        {"([)]",    false},
        {"",        true},
        {"(((",     false},
        {"{[()]}",  true},
        {"{[}]",    false},
        {"()[]{}", true},
        {"((()))", true},
    };

    for (const auto& [input, expected] : testCases) {
        bool result = isBalanced(input);
        string status = (result == expected) ? "PASS" : "FAIL";
        cout << "  " << status << " | \""
             << (input.empty() ? "(empty)" : input)
             << "\" -> " << boolalpha << result << endl;
    }
}

// === Exercise 5: Custom Hash for Point3D ===
// Problem: Specialize std::hash for Point3D using hash_combine technique.
//          operator== is required because unordered_map needs to handle
//          hash collisions by comparing keys for equality.

struct Point3D {
    int x, y, z;

    // operator== is required alongside the hash specialization because
    // when two different keys hash to the same bucket (collision), the
    // container must compare them for equality to distinguish them.
    bool operator==(const Point3D& other) const {
        return x == other.x && y == other.y && z == other.z;
    }
};

// Hash combine technique: mixes multiple hash values into one.
// The magic constant 0x9e3779b9 is derived from the golden ratio
// and provides good bit distribution.
struct Point3DHash {
    size_t operator()(const Point3D& p) const {
        size_t seed = 0;
        auto hash_combine = [&seed](size_t h) {
            seed ^= h + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        };
        hash_combine(hash<int>{}(p.x));
        hash_combine(hash<int>{}(p.y));
        hash_combine(hash<int>{}(p.z));
        return seed;
    }
};

void exercise_5() {
    cout << "\n=== Exercise 5: Custom Hash for Point3D ===" << endl;

    unordered_map<Point3D, string, Point3DHash> pointLabels;

    pointLabels[{0, 0, 0}] = "origin";
    pointLabels[{1, 0, 0}] = "unit-x";
    pointLabels[{0, 1, 0}] = "unit-y";
    pointLabels[{0, 0, 1}] = "unit-z";
    pointLabels[{1, 1, 1}] = "diagonal";

    // Lookup
    Point3D query{1, 1, 1};
    if (auto it = pointLabels.find(query); it != pointLabels.end()) {
        cout << "  Found ({1,1,1}): " << it->second << endl;
    }

    // Iterate all entries
    cout << "  All entries:" << endl;
    for (const auto& [pt, label] : pointLabels) {
        cout << "    (" << pt.x << "," << pt.y << "," << pt.z
             << ") -> " << label << endl;
    }

    // Verify that different points don't collide on lookup
    Point3D missing{9, 9, 9};
    cout << "  Contains (9,9,9)? " << boolalpha
         << (pointLabels.count(missing) > 0) << endl;
}

int main() {
    exercise_1();
    exercise_2();
    exercise_3();
    exercise_4();
    exercise_5();
    cout << "\nAll exercises completed!" << endl;
    return 0;
}
