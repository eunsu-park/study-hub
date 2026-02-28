/*
 * Exercises for Lesson 11: STL Algorithms and Iterators
 * Topic: CPP
 * Compile: g++ -std=c++17 -Wall -Wextra -o ex11 11_stl_algorithms_iterators.cpp
 */
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <numeric>
#include <iterator>
#include <cctype>
using namespace std;

// === Exercise 1: Lambda Capture Modes ===
// Problem: Demonstrate all four capture modes: by-value, by-reference,
//          [=] (all by value), [&] (all by reference).
void exercise_1() {
    cout << "=== Exercise 1: Lambda Capture Modes ===" << endl;

    int base = 10;
    int multiplier = 3;

    cout << "  Initial: base=" << base << ", multiplier=" << multiplier << endl;

    // 1. Capture base by value: lambda gets a copy, original unchanged
    auto addBase = [base](int n) { return base + n; };
    cout << "\n  [base] addBase(5) = " << addBase(5) << endl;
    cout << "  base after: " << base << " (unchanged, captured by value)" << endl;

    // 2. Capture multiplier by reference: lambda modifies the original
    auto doubleMultiplier = [&multiplier]() {
        multiplier *= 2;
    };
    doubleMultiplier();
    cout << "\n  [&multiplier] after doubling: multiplier=" << multiplier << endl;
    cout << "  (original changed because captured by reference)" << endl;

    // 3. Capture everything by value [=]: snapshot of all variables
    auto computeAll = [=](int n) { return base * multiplier + n; };
    cout << "\n  [=] computeAll(7) = " << computeAll(7)
         << " (base=" << base << " * multiplier=" << multiplier << " + 7)" << endl;

    // 4. Capture everything by reference [&]: modifies originals
    auto incrementBoth = [&]() {
        base++;
        multiplier++;
    };
    incrementBoth();
    cout << "\n  [&] after incrementBoth: base=" << base
         << ", multiplier=" << multiplier << endl;
    cout << "  (both changed because captured by reference)" << endl;
}

// === Exercise 2: Pipeline with transform and accumulate ===
// Problem: Process a vector of strings using only STL algorithms and lambdas.
void exercise_2() {
    cout << "\n=== Exercise 2: Pipeline with transform and accumulate ===" << endl;

    vector<string> words = {"hello", "world", "cpp", "algorithms"};

    // 1. Transform: create a vector of lengths
    vector<size_t> lengths(words.size());
    transform(words.begin(), words.end(), lengths.begin(),
              [](const string& w) { return w.size(); });

    cout << "  Word lengths: ";
    for (size_t len : lengths) {
        cout << len << " ";
    }
    cout << endl;

    // 2. Accumulate: total character count across all words
    size_t totalChars = accumulate(words.begin(), words.end(), size_t{0},
        [](size_t sum, const string& w) { return sum + w.size(); });
    cout << "  Total characters: " << totalChars << endl;

    // 3. find_if: first word longer than 5 characters
    auto it = find_if(words.begin(), words.end(),
        [](const string& w) { return w.size() > 5; });
    if (it != words.end()) {
        cout << "  First word > 5 chars: \"" << *it << "\"" << endl;
    } else {
        cout << "  No word > 5 chars found" << endl;
    }

    // 4. count_if: words with even character count
    auto evenCount = count_if(words.begin(), words.end(),
        [](const string& w) { return w.size() % 2 == 0; });
    cout << "  Words with even length: " << evenCount << endl;
}

// === Exercise 3: Sort with Custom Comparator ===
// Problem: Sort a vector of Person structs by age, by name (case-insensitive),
//          and by name length with alphabetical tie-breaking.
void exercise_3() {
    cout << "\n=== Exercise 3: Sort with Custom Comparator ===" << endl;

    struct Person {
        string name;
        int age;
    };

    auto printPeople = [](const string& label, const vector<Person>& people) {
        cout << "  " << label << ": ";
        for (const auto& p : people) {
            cout << p.name << "(" << p.age << ") ";
        }
        cout << endl;
    };

    vector<Person> people = {
        {"Charlie", 30}, {"alice", 25}, {"Bob", 35},
        {"Diana", 28}, {"Eve", 22}
    };

    printPeople("Original", people);

    // 1. Sort by age ascending
    sort(people.begin(), people.end(),
         [](const Person& a, const Person& b) { return a.age < b.age; });
    printPeople("By age", people);

    // 2. Sort by name alphabetically (case-insensitive)
    sort(people.begin(), people.end(),
         [](const Person& a, const Person& b) {
             // Compare lowercase versions of each name
             string aLower = a.name, bLower = b.name;
             transform(aLower.begin(), aLower.end(), aLower.begin(), ::tolower);
             transform(bLower.begin(), bLower.end(), bLower.begin(), ::tolower);
             return aLower < bLower;
         });
    printPeople("By name (case-insensitive)", people);

    // 3. Sort by name length, then alphabetically as tie-breaker
    sort(people.begin(), people.end(),
         [](const Person& a, const Person& b) {
             if (a.name.size() != b.name.size()) {
                 return a.name.size() < b.name.size();
             }
             // Tie-breaker: alphabetical (case-insensitive)
             string aLower = a.name, bLower = b.name;
             transform(aLower.begin(), aLower.end(), aLower.begin(), ::tolower);
             transform(bLower.begin(), bLower.end(), bLower.begin(), ::tolower);
             return aLower < bLower;
         });
    printPeople("By length, then alpha", people);
}

// === Exercise 4: Erase-Remove Idiom ===
// Problem: Remove even numbers, then keep only those > 3, verify result is {5, 7, 9}.
void exercise_4() {
    cout << "\n=== Exercise 4: Erase-Remove Idiom ===" << endl;

    vector<int> v = {1, 5, 2, 8, 3, 7, 4, 6, 9, 10};

    auto printVec = [](const string& label, const vector<int>& vec) {
        cout << "  " << label << ": {";
        for (size_t i = 0; i < vec.size(); i++) {
            if (i > 0) cout << ", ";
            cout << vec[i];
        }
        cout << "}" << endl;
    };

    printVec("Original", v);

    // Step 1: Remove all even numbers
    // std::remove_if moves non-matching elements to the front and returns
    // an iterator to the new "logical end." The vector's actual size doesn't
    // change -- elements past the returned iterator are in an unspecified state.
    // That's why we need .erase() to actually shrink the container.
    v.erase(
        remove_if(v.begin(), v.end(), [](int x) { return x % 2 == 0; }),
        v.end()
    );
    printVec("After removing evens", v);

    // Step 2: Keep only those > 3 (remove those <= 3)
    v.erase(
        remove_if(v.begin(), v.end(), [](int x) { return x <= 3; }),
        v.end()
    );
    printVec("After keeping only > 3", v);

    // Verify
    vector<int> expected = {5, 7, 9};
    cout << "  Matches expected {5, 7, 9}? " << boolalpha
         << (v == expected) << endl;

    // Explanation: std::remove_if alone doesn't shrink the vector because
    // it is an algorithm operating on iterators, not on the container itself.
    // It can only rearrange elements within the range. The container must
    // be told to release the "dead" elements via erase().
}

// === Exercise 5: Set Operations on Sorted Ranges ===
// Problem: Compute union, intersection, difference, and symmetric difference
//          of two sorted vectors using STL set algorithms.
void exercise_5() {
    cout << "\n=== Exercise 5: Set Operations on Sorted Ranges ===" << endl;

    vector<int> clubA = {1, 3, 5, 7, 9, 11};
    vector<int> clubB = {3, 6, 9, 12, 15};

    auto printVec = [](const string& label, const vector<int>& vec) {
        cout << "  " << label << ": {";
        for (size_t i = 0; i < vec.size(); i++) {
            if (i > 0) cout << ", ";
            cout << vec[i];
        }
        cout << "}" << endl;
    };

    printVec("Club A", clubA);
    printVec("Club B", clubB);
    cout << endl;

    // 1. Union: members in either club
    vector<int> unionResult;
    set_union(clubA.begin(), clubA.end(),
              clubB.begin(), clubB.end(),
              back_inserter(unionResult));
    printVec("Union (A | B)", unionResult);

    // 2. Intersection: members in both clubs
    vector<int> interResult;
    set_intersection(clubA.begin(), clubA.end(),
                     clubB.begin(), clubB.end(),
                     back_inserter(interResult));
    printVec("Intersection (A & B)", interResult);

    // 3. Difference: in A but not in B
    vector<int> diffResult;
    set_difference(clubA.begin(), clubA.end(),
                   clubB.begin(), clubB.end(),
                   back_inserter(diffResult));
    printVec("Difference (A - B)", diffResult);

    // 4. Symmetric difference: in exactly one club but not both
    vector<int> symDiffResult;
    set_symmetric_difference(clubA.begin(), clubA.end(),
                             clubB.begin(), clubB.end(),
                             back_inserter(symDiffResult));
    printVec("Symmetric Diff (A ^ B)", symDiffResult);
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
