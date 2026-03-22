/*
 * Exercises for Lesson 12: Templates
 * Topic: CPP
 * Compile: g++ -std=c++17 -Wall -Wextra -o ex12 12_templates.cpp
 */
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <initializer_list>
#include <type_traits>
using namespace std;

// === Exercise 1: Min/Max from Variadic Arguments ===
// Problem: Write a function template that returns the minimum and maximum
//          values from an arbitrary number of arguments using variadic templates.

// Base case: single argument is both min and max
template<typename T>
pair<T, T> minmax_of(T val) {
    return {val, val};
}

// Recursive variadic: compare head with the min/max of the rest
template<typename T, typename... Args>
pair<T, T> minmax_of(T first, Args... rest) {
    auto [rmin, rmax] = minmax_of(rest...);
    T resultMin = (first < rmin) ? first : rmin;
    T resultMax = (first > rmax) ? first : rmax;
    return {resultMin, resultMax};
}

void exercise_1() {
    cout << "=== Exercise 1: Min/Max from Variadic Arguments ===" << endl;

    // Integer test
    auto [imin, imax] = minmax_of(5, 2, 8, 1, 9, 3);
    cout << "  minmax_of(5,2,8,1,9,3) -> min=" << imin
         << ", max=" << imax << endl;

    // Double test
    auto [dmin, dmax] = minmax_of(3.14, 1.41, 2.72, 0.57);
    cout << "  minmax_of(3.14,1.41,2.72,0.57) -> min=" << dmin
         << ", max=" << dmax << endl;

    // String test (lexicographic)
    auto [smin, smax] = minmax_of(string("banana"), string("apple"), string("cherry"));
    cout << "  minmax_of(\"banana\",\"apple\",\"cherry\") -> min=\"" << smin
         << "\", max=\"" << smax << "\"" << endl;

    // Single argument
    auto [single_min, single_max] = minmax_of(42);
    cout << "  minmax_of(42) -> min=" << single_min
         << ", max=" << single_max << endl;
}

// === Exercise 2: Generic Queue ===
// Problem: Implement a Queue class template with enqueue, dequeue, front,
//          empty, and size operations.

template<typename T>
class Queue {
    vector<T> data_;
    size_t frontIdx_ = 0;  // Index of the front element

public:
    // Enqueue: add element to the back
    void enqueue(const T& value) {
        data_.push_back(value);
    }

    // Move version for efficiency
    void enqueue(T&& value) {
        data_.push_back(std::move(value));
    }

    // Dequeue: remove and return front element
    T dequeue() {
        if (empty()) {
            throw runtime_error("Queue is empty");
        }
        T value = std::move(data_[frontIdx_]);
        frontIdx_++;

        // Compact the vector when half the capacity is wasted
        // to prevent unbounded memory growth
        if (frontIdx_ > data_.size() / 2 && frontIdx_ > 16) {
            data_.erase(data_.begin(), data_.begin() + frontIdx_);
            frontIdx_ = 0;
        }

        return value;
    }

    // Peek at front without removing
    const T& front() const {
        if (empty()) {
            throw runtime_error("Queue is empty");
        }
        return data_[frontIdx_];
    }

    bool empty() const {
        return frontIdx_ >= data_.size();
    }

    size_t size() const {
        return data_.size() - frontIdx_;
    }
};

void exercise_2() {
    cout << "\n=== Exercise 2: Generic Queue ===" << endl;

    // Integer queue
    Queue<int> intQ;
    for (int i = 1; i <= 5; i++) {
        intQ.enqueue(i * 10);
    }
    cout << "  Int queue front: " << intQ.front()
         << ", size: " << intQ.size() << endl;

    cout << "  Dequeue order: ";
    while (!intQ.empty()) {
        cout << intQ.dequeue() << " ";
    }
    cout << endl;

    // String queue
    Queue<string> strQ;
    strQ.enqueue("first");
    strQ.enqueue("second");
    strQ.enqueue("third");

    cout << "  String queue: ";
    while (!strQ.empty()) {
        cout << "\"" << strQ.dequeue() << "\" ";
    }
    cout << endl;

    // Exception test
    Queue<int> emptyQ;
    try {
        emptyQ.dequeue();
    } catch (const runtime_error& e) {
        cout << "  Expected error: " << e.what() << endl;
    }
}

// === Exercise 3: Type-Specific Serialization ===
// Problem: Write a serialize function template that converts various types
//          to strings, with specializations for containers.

// Primary template: uses stringstream for basic types
template<typename T>
string serialize(const T& value) {
    ostringstream oss;
    oss << value;
    return oss.str();
}

// Specialization for bool: print "true"/"false" instead of 1/0
template<>
string serialize<bool>(const bool& value) {
    return value ? "true" : "false";
}

// Specialization for string: wrap in quotes
template<>
string serialize<string>(const string& value) {
    return "\"" + value + "\"";
}

// Specialization for const char*: wrap in quotes
template<>
string serialize<const char*>(const char* const& value) {
    return string("\"") + value + "\"";
}

// Overload for vector<T>: serialize each element in brackets
template<typename T>
string serialize(const vector<T>& vec) {
    ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < vec.size(); i++) {
        if (i > 0) oss << ", ";
        oss << serialize(vec[i]);
    }
    oss << "]";
    return oss.str();
}

// Overload for pair<K, V>
template<typename K, typename V>
string serialize(const pair<K, V>& p) {
    return "(" + serialize(p.first) + ", " + serialize(p.second) + ")";
}

void exercise_3() {
    cout << "\n=== Exercise 3: Type-Specific Serialization ===" << endl;

    // Basic types
    cout << "  int:    " << serialize(42) << endl;
    cout << "  double: " << serialize(3.14159) << endl;
    cout << "  bool:   " << serialize(true) << endl;
    cout << "  string: " << serialize(string("hello world")) << endl;
    cout << "  char:   " << serialize('A') << endl;

    // Containers
    vector<int> nums = {1, 2, 3, 4, 5};
    cout << "  vector<int>:    " << serialize(nums) << endl;

    vector<string> words = {"hello", "world"};
    cout << "  vector<string>: " << serialize(words) << endl;

    // Nested container
    vector<vector<int>> matrix = {{1, 2}, {3, 4}, {5, 6}};
    cout << "  vector<vector>: " << serialize(matrix) << endl;

    // Pair
    auto p = make_pair(string("key"), 42);
    cout << "  pair:           " << serialize(p) << endl;
}

int main() {
    exercise_1();
    exercise_2();
    exercise_3();
    cout << "\nAll exercises completed!" << endl;
    return 0;
}
