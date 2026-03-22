/*
 * Exercises for Lesson 03: Operators and Control Flow
 * Topic: CPP
 * Compile: g++ -std=c++17 -Wall -Wextra -o ex03 03_operators_and_control_flow.cpp
 */
#include <iostream>
#include <string>
using namespace std;

// === Exercise 1: Operator Evaluation Prediction ===
// Problem: Predict the output of each statement involving integer division,
//          modulus, pre/post increment, and short-circuit evaluation.
void exercise_1() {
    cout << "=== Exercise 1: Operator Evaluation Prediction ===" << endl;

    int a = 10, b = 3;

    // 10 / 3 = 3 (integer division truncates toward zero)
    cout << a / b << endl;          // 3

    // 10 % 3 = 1 (remainder after integer division)
    cout << a % b << endl;          // 1

    // (double)10 / 3 = 3.33333 (one operand is double, so floating-point division)
    cout << (double)a / b << endl;  // 3.33333

    int x = 5;
    // Postfix: returns current value (5), THEN increments x to 6
    cout << x++ << endl;  // 5
    // x is now 6
    cout << x   << endl;  // 6
    // Prefix: increments x to 7 first, THEN returns 7
    cout << ++x << endl;  // 7

    // Short-circuit evaluation:
    // - (false && expr) : expr is never evaluated because AND needs both true
    // - (true || expr)  : expr is never evaluated because OR only needs one true
    int counter = 0;
    if (false && (++counter > 0)) {}  // ++counter is NOT evaluated
    if (true  || (++counter > 0)) {}  // ++counter is NOT evaluated
    cout << counter << endl;          // 0

    cout << "\nExplanation: counter remains 0 because short-circuit evaluation" << endl;
    cout << "skips the right-hand side of && when the left is false, and" << endl;
    cout << "skips the right-hand side of || when the left is true." << endl;
}

// === Exercise 2: Bitwise Flag Operations ===
// Problem: Implement a permission system using bitwise operators with
//          READ, WRITE, and EXECUTE flags. Grant, revoke, and check permissions.
void exercise_2() {
    cout << "\n=== Exercise 2: Bitwise Flag Operations ===" << endl;

    const int READ    = 1;  // 001
    const int WRITE   = 2;  // 010
    const int EXECUTE = 4;  // 100

    // Helper lambda to print permissions in human-readable format
    auto printPerms = [&](const string& label, int perms) {
        cout << label << ": "
             << ((perms & READ)    ? "R" : "-")
             << ((perms & WRITE)   ? "W" : "-")
             << ((perms & EXECUTE) ? "X" : "-")
             << " (binary: ";
        // Print as 3-bit binary
        for (int i = 2; i >= 0; --i) {
            cout << ((perms >> i) & 1);
        }
        cout << ")" << endl;
    };

    // Step 1: Create permission with READ and WRITE
    int perms = READ | WRITE;
    printPerms("Initial (R+W)", perms);

    // Step 2: Check if EXECUTE is set (it should not be)
    bool hasExecute = (perms & EXECUTE) != 0;
    cout << "Has EXECUTE? " << boolalpha << hasExecute << endl;

    // Step 3: Grant EXECUTE using bitwise OR
    perms |= EXECUTE;
    printPerms("After grant X", perms);

    // Step 4: Revoke WRITE using bitwise AND with NOT
    perms &= ~WRITE;
    printPerms("After revoke W", perms);

    // Final: perms should have READ and EXECUTE only (101 = 5)
    cout << "Final permissions value: " << perms << endl;
}

// === Exercise 3: Grade Calculator with switch ===
// Problem: Convert a numeric score (0-100) to a letter grade using switch on
//          score/10. Test with boundary values including 59, 60, 89, 90, 100.
char letterGrade(int score) {
    // Clamp score to valid range for safety
    if (score < 0 || score > 100) return '?';

    // Divide by 10 to get a single digit for the switch.
    // score=100 gives 10, score=90 gives 9, etc.
    switch (score / 10) {
        case 10:  // 100
        case 9:   // 90-99
            return 'A';
        case 8:   // 80-89
            return 'B';
        case 7:   // 70-79
            return 'C';
        case 6:   // 60-69
            return 'D';
        default:  // 0-59
            return 'F';
    }
}

void exercise_3() {
    cout << "\n=== Exercise 3: Grade Calculator with switch ===" << endl;

    int testScores[] = {59, 60, 89, 90, 100, 0, 75, 85, 95};

    for (int score : testScores) {
        cout << "Score " << score << " -> Grade " << letterGrade(score) << endl;
    }

    // Verify boundary cases explicitly
    cout << "\nBoundary verification:" << endl;
    cout << "  59 -> " << letterGrade(59) << " (should be F)" << endl;
    cout << "  60 -> " << letterGrade(60) << " (should be D)" << endl;
    cout << "  89 -> " << letterGrade(89) << " (should be B)" << endl;
    cout << "  90 -> " << letterGrade(90) << " (should be A)" << endl;
    cout << " 100 -> " << letterGrade(100) << " (should be A)" << endl;
}

// === Exercise 4: Diamond Pattern ===
// Problem: Print a diamond pattern using nested for loops for a given size.
//          The top half has rows of 2*i-1 stars (i from 1 to size), and
//          the bottom half mirrors it.
void exercise_4() {
    cout << "\n=== Exercise 4: Diamond Pattern ===" << endl;

    int size = 4;

    // Top half (including middle row): rows with 1, 3, 5, 7 stars
    for (int i = 1; i <= size; i++) {
        // Print leading spaces: (size - i) spaces
        for (int j = 0; j < size - i; j++) {
            cout << ' ';
        }
        // Print stars: 2*i - 1 stars
        for (int j = 0; j < 2 * i - 1; j++) {
            cout << '*';
        }
        cout << endl;
    }

    // Bottom half: rows with 5, 3, 1 stars (mirror of top, excluding middle)
    for (int i = size - 1; i >= 1; i--) {
        // Print leading spaces
        for (int j = 0; j < size - i; j++) {
            cout << ' ';
        }
        // Print stars
        for (int j = 0; j < 2 * i - 1; j++) {
            cout << '*';
        }
        cout << endl;
    }
}

// === Exercise 5: Input Validation Loop ===
// Problem: Repeatedly prompt for an integer in [1, 100] using do-while.
//          Then classify as "low" (1-33), "medium" (34-66), or "high" (67-100)
//          using the ternary operator.
void exercise_5() {
    cout << "\n=== Exercise 5: Input Validation Loop ===" << endl;

    // Since we cannot use interactive input in an exercise solution file,
    // we simulate the validation logic with test values.
    int testValues[] = {0, -5, 150, 101, 33, 34, 66, 67, 100, 1, 50};

    for (int num : testValues) {
        bool valid = (num >= 1 && num <= 100);

        if (!valid) {
            cout << "  " << num << " -> INVALID (not in [1, 100])" << endl;
            continue;
        }

        // Ternary operator to classify
        string classification = (num <= 33) ? "low"
                              : (num <= 66) ? "medium"
                              : "high";

        cout << "  " << num << " -> " << classification << endl;
    }

    // Show the do-while pattern that would be used with real input:
    cout << "\n--- Do-while pattern (commented pseudocode) ---" << endl;
    cout << R"(
    int num;
    do {
        cout << "Enter a number [1-100]: ";
        cin >> num;
    } while (num < 1 || num > 100);

    string cls = (num <= 33) ? "low"
               : (num <= 66) ? "medium"
               : "high";
    cout << "Classification: " << cls << endl;
    )" << endl;
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
