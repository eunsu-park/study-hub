# Operators and Control Flow

**Previous**: [Variables and Types](./02_Variables_and_Types.md) | **Next**: [Functions](./04_Functions.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Apply arithmetic, assignment, comparison, logical, and bitwise operators in C++ expressions
2. Distinguish between prefix and postfix increment/decrement behavior
3. Explain short-circuit evaluation in logical expressions and identify its side-effect implications
4. Implement branching logic using `if`, `else if`, `else`, and `switch` statements
5. Design loops using `for`, `while`, and `do-while`, including range-based `for` (C++11)
6. Apply `break` and `continue` to control loop execution flow
7. Identify operator precedence rules and use parentheses to clarify intent

---

Operators and control flow are the steering wheel and accelerator of every program. Without operators you cannot compute, compare, or combine values; without control flow you cannot make decisions or repeat work. Together they transform a static list of declarations into dynamic, responsive logic -- and every advanced C++ feature you will encounter later is ultimately built on these primitives.

## 1. Arithmetic Operators

### Basic Arithmetic Operators

| Operator | Meaning | Example |
|----------|---------|---------|
| `+` | Addition | `a + b` |
| `-` | Subtraction | `a - b` |
| `*` | Multiplication | `a * b` |
| `/` | Division | `a / b` |
| `%` | Modulus | `a % b` |

```cpp
#include <iostream>

int main() {
    int a = 17, b = 5;

    std::cout << "a + b = " << a + b << std::endl;  // 22
    std::cout << "a - b = " << a - b << std::endl;  // 12
    std::cout << "a * b = " << a * b << std::endl;  // 85
    std::cout << "a / b = " << a / b << std::endl;  // 3 (integer division)
    std::cout << "a % b = " << a % b << std::endl;  // 2

    return 0;
}
```

### Integer Division vs Floating-Point Division

```cpp
#include <iostream>

int main() {
    int a = 7, b = 2;

    // Integer division (decimal truncated)
    std::cout << "7 / 2 = " << a / b << std::endl;  // 3

    // Floating-point division
    std::cout << "7.0 / 2 = " << 7.0 / 2 << std::endl;  // 3.5
    std::cout << "(double)7 / 2 = " << static_cast<double>(a) / b << std::endl;  // 3.5

    return 0;
}
```

### Increment and Decrement Operators

```cpp
#include <iostream>

int main() {
    int a = 5;

    std::cout << "a = " << a << std::endl;    // 5
    std::cout << "++a = " << ++a << std::endl; // 6 (prefix: increment first)
    std::cout << "a++ = " << a++ << std::endl; // 6 (postfix: increment after)
    std::cout << "a = " << a << std::endl;    // 7

    return 0;
}
```

---

## 2. Assignment Operators

### Compound Assignment Operators

```cpp
#include <iostream>

int main() {
    int a = 10;

    a += 5;   // a = a + 5
    std::cout << "a += 5: " << a << std::endl;  // 15

    a -= 3;   // a = a - 3
    std::cout << "a -= 3: " << a << std::endl;  // 12

    a *= 2;   // a = a * 2
    std::cout << "a *= 2: " << a << std::endl;  // 24

    a /= 4;   // a = a / 4
    std::cout << "a /= 4: " << a << std::endl;  // 6

    a %= 4;   // a = a % 4
    std::cout << "a %= 4: " << a << std::endl;  // 2

    return 0;
}
```

---

## 3. Comparison Operators

| Operator | Meaning | Example |
|----------|---------|---------|
| `==` | Equal to | `a == b` |
| `!=` | Not equal to | `a != b` |
| `<` | Less than | `a < b` |
| `>` | Greater than | `a > b` |
| `<=` | Less than or equal to | `a <= b` |
| `>=` | Greater than or equal to | `a >= b` |

```cpp
#include <iostream>

int main() {
    int a = 5, b = 10;

    std::cout << std::boolalpha;  // Output as true/false
    std::cout << "a == b: " << (a == b) << std::endl;  // false
    std::cout << "a != b: " << (a != b) << std::endl;  // true
    std::cout << "a < b: " << (a < b) << std::endl;    // true
    std::cout << "a > b: " << (a > b) << std::endl;    // false
    std::cout << "a <= b: " << (a <= b) << std::endl;  // true
    std::cout << "a >= b: " << (a >= b) << std::endl;  // false

    return 0;
}
```

---

## 4. Logical Operators

| Operator | Meaning | Example |
|----------|---------|---------|
| `&&` | AND | `a && b` |
| `\|\|` | OR | `a \|\| b` |
| `!` | NOT | `!a` |

```cpp
#include <iostream>

int main() {
    bool a = true, b = false;

    std::cout << std::boolalpha;
    std::cout << "a && b: " << (a && b) << std::endl;  // false
    std::cout << "a || b: " << (a || b) << std::endl;  // true
    std::cout << "!a: " << (!a) << std::endl;          // false
    std::cout << "!b: " << (!b) << std::endl;          // true

    // Compound conditions
    int age = 25;
    bool isStudent = true;

    bool discount = (age < 20) || isStudent;  // Student or under 20
    std::cout << "Discount applied: " << discount << std::endl;  // true

    return 0;
}
```

### Short-circuit Evaluation

```cpp
#include <iostream>

int main() {
    int x = 0;

    // &&: If first is false, second is not evaluated
    if (false && (++x > 0)) {
        // x is not incremented
    }
    std::cout << "x after &&: " << x << std::endl;  // 0

    // ||: If first is true, second is not evaluated
    if (true || (++x > 0)) {
        // x is not incremented
    }
    std::cout << "x after ||: " << x << std::endl;  // 0

    return 0;
}
```

---

## 5. Bitwise Operators

| Operator | Meaning | Example |
|----------|---------|---------|
| `&` | AND | `a & b` |
| `\|` | OR | `a \| b` |
| `^` | XOR | `a ^ b` |
| `~` | NOT | `~a` |
| `<<` | Left shift | `a << n` |
| `>>` | Right shift | `a >> n` |

```cpp
#include <iostream>

int main() {
    int a = 5;  // 0101
    int b = 3;  // 0011

    std::cout << "a & b = " << (a & b) << std::endl;  // 1 (0001)
    std::cout << "a | b = " << (a | b) << std::endl;  // 7 (0111)
    std::cout << "a ^ b = " << (a ^ b) << std::endl;  // 6 (0110)
    std::cout << "~a = " << (~a) << std::endl;        // -6

    std::cout << "a << 1 = " << (a << 1) << std::endl;  // 10 (1010)
    std::cout << "a >> 1 = " << (a >> 1) << std::endl;  // 2 (0010)

    return 0;
}
```

---

## 6. Ternary Operator

```cpp
condition ? value_if_true : value_if_false
```

```cpp
#include <iostream>

int main() {
    int a = 10, b = 20;

    // Alternative to if-else
    int max = (a > b) ? a : b;
    std::cout << "Maximum: " << max << std::endl;  // 20

    // String selection
    int score = 85;
    std::string result = (score >= 60) ? "Pass" : "Fail";
    std::cout << "Result: " << result << std::endl;  // Pass

    // Nested (be careful with readability)
    int num = 0;
    std::string sign = (num > 0) ? "positive" : (num < 0) ? "negative" : "zero";
    std::cout << "Sign: " << sign << std::endl;  // zero

    return 0;
}
```

---

## 7. if Statement

### Basic if Statement

```cpp
#include <iostream>

int main() {
    int age = 18;

    if (age >= 18) {
        std::cout << "You are an adult." << std::endl;
    }

    return 0;
}
```

### if-else Statement

```cpp
#include <iostream>

int main() {
    int score = 75;

    if (score >= 60) {
        std::cout << "Pass" << std::endl;
    } else {
        std::cout << "Fail" << std::endl;
    }

    return 0;
}
```

### if-else if-else Statement

```cpp
#include <iostream>

int main() {
    int score = 85;

    if (score >= 90) {
        std::cout << "A" << std::endl;
    } else if (score >= 80) {
        std::cout << "B" << std::endl;
    } else if (score >= 70) {
        std::cout << "C" << std::endl;
    } else if (score >= 60) {
        std::cout << "D" << std::endl;
    } else {
        std::cout << "F" << std::endl;
    }

    return 0;
}
```

### Variable Declaration in if Statement (C++17)

```cpp
#include <iostream>
#include <map>
#include <string>

int main() {
    std::map<std::string, int> scores = {{"Alice", 90}, {"Bob", 85}};

    // C++17: Variable declaration in if statement
    if (auto it = scores.find("Alice"); it != scores.end()) {
        std::cout << "Alice's score: " << it->second << std::endl;
    }

    return 0;
}
```

---

## 8. switch Statement

### Basic switch Statement

```cpp
#include <iostream>

int main() {
    int day = 3;

    switch (day) {
        case 1:
            std::cout << "Monday" << std::endl;
            break;
        case 2:
            std::cout << "Tuesday" << std::endl;
            break;
        case 3:
            std::cout << "Wednesday" << std::endl;
            break;
        case 4:
            std::cout << "Thursday" << std::endl;
            break;
        case 5:
            std::cout << "Friday" << std::endl;
            break;
        case 6:
        case 7:
            std::cout << "Weekend" << std::endl;
            break;
        default:
            std::cout << "Invalid value" << std::endl;
    }

    return 0;
}
```

### Fall-through (Intentional Omission)

```cpp
#include <iostream>

int main() {
    char grade = 'B';

    switch (grade) {
        case 'A':
        case 'B':
        case 'C':
            std::cout << "Pass" << std::endl;
            break;
        case 'D':
        case 'F':
            std::cout << "Fail" << std::endl;
            break;
        default:
            std::cout << "Invalid grade" << std::endl;
    }

    return 0;
}
```

### switch Statement Cautions

```cpp
// switch only works with integral types, character types, and enums
// Strings are not allowed (in C++)

// Variable declaration requires braces
switch (value) {
    case 1: {
        int x = 10;  // Scope defined with braces
        // ...
        break;
    }
    case 2:
        // ...
        break;
}
```

---

## 9. for Loop

### Basic for Loop

```cpp
#include <iostream>

int main() {
    // Print 1 to 5
    for (int i = 1; i <= 5; i++) {
        std::cout << i << " ";
    }
    std::cout << std::endl;  // 1 2 3 4 5

    return 0;
}
```

### Reverse for Loop

```cpp
#include <iostream>

int main() {
    for (int i = 5; i >= 1; i--) {
        std::cout << i << " ";
    }
    std::cout << std::endl;  // 5 4 3 2 1

    return 0;
}
```

### Nested for Loop

```cpp
#include <iostream>

int main() {
    // Multiplication table of 2
    for (int i = 1; i <= 9; i++) {
        std::cout << "2 x " << i << " = " << 2 * i << std::endl;
    }

    // Star triangle
    for (int i = 1; i <= 5; i++) {
        for (int j = 1; j <= i; j++) {
            std::cout << "*";
        }
        std::cout << std::endl;
    }

    return 0;
}
```

Output:
```
*
**
***
****
*****
```

### Range-based for Loop (C++11)

```cpp
#include <iostream>
#include <vector>

int main() {
    int arr[] = {1, 2, 3, 4, 5};

    // Array traversal
    for (int num : arr) {
        std::cout << num << " ";
    }
    std::cout << std::endl;

    // Modification by reference
    for (int& num : arr) {
        num *= 2;
    }

    // Vector traversal
    std::vector<std::string> names = {"Alice", "Bob", "Charlie"};
    for (const auto& name : names) {
        std::cout << name << std::endl;
    }

    return 0;
}
```

---

## 10. while Loop

### Basic while Loop

```cpp
#include <iostream>

int main() {
    int count = 1;

    while (count <= 5) {
        std::cout << count << " ";
        count++;
    }
    std::cout << std::endl;  // 1 2 3 4 5

    return 0;
}
```

### Infinite Loop and Breaking

```cpp
#include <iostream>

int main() {
    int num;

    while (true) {
        std::cout << "Enter a number (0 to exit): ";
        std::cin >> num;

        if (num == 0) {
            break;  // Exit loop
        }

        std::cout << "Input: " << num << std::endl;
    }

    std::cout << "Exited" << std::endl;

    return 0;
}
```

---

## 11. do-while Loop

Executes at least once.

```cpp
#include <iostream>

int main() {
    int num;

    do {
        std::cout << "Enter a number between 1 and 10: ";
        std::cin >> num;
    } while (num < 1 || num > 10);  // Repeat if condition is true

    std::cout << "You entered: " << num << std::endl;

    return 0;
}
```

### while vs do-while

```cpp
#include <iostream>

int main() {
    int x = 0;

    // while: Condition checked first
    while (x > 0) {
        std::cout << "while executed" << std::endl;
        x--;
    }
    // No output

    // do-while: Executes at least once
    do {
        std::cout << "do-while executed" << std::endl;
        x--;
    } while (x > 0);
    // "do-while executed" is printed

    return 0;
}
```

---

## 12. break and continue

### break

Immediately exits the loop.

```cpp
#include <iostream>

int main() {
    for (int i = 1; i <= 10; i++) {
        if (i == 5) {
            break;  // Exit at 5
        }
        std::cout << i << " ";
    }
    std::cout << std::endl;  // 1 2 3 4

    return 0;
}
```

### continue

Skips the current iteration.

```cpp
#include <iostream>

int main() {
    for (int i = 1; i <= 10; i++) {
        if (i % 2 == 0) {
            continue;  // Skip even numbers
        }
        std::cout << i << " ";
    }
    std::cout << std::endl;  // 1 3 5 7 9

    return 0;
}
```

---

## 13. Operator Precedence

| Precedence | Operators |
|------------|-----------|
| 1 (highest) | `()`, `[]`, `->`, `.` |
| 2 | `!`, `~`, `++`, `--`, `sizeof` |
| 3 | `*`, `/`, `%` |
| 4 | `+`, `-` |
| 5 | `<<`, `>>` |
| 6 | `<`, `<=`, `>`, `>=` |
| 7 | `==`, `!=` |
| 8 | `&` |
| 9 | `^` |
| 10 | `\|` |
| 11 | `&&` |
| 12 | `\|\|` |
| 13 | `?:` |
| 14 (lowest) | `=`, `+=`, `-=`, etc. |

**Tip**: When in doubt, use parentheses!

---

## 14. Summary

| Category | Operators |
|----------|-----------|
| Arithmetic | `+`, `-`, `*`, `/`, `%` |
| Comparison | `==`, `!=`, `<`, `>`, `<=`, `>=` |
| Logical | `&&`, `\|\|`, `!` |
| Bitwise | `&`, `\|`, `^`, `~`, `<<`, `>>` |
| Assignment | `=`, `+=`, `-=`, `*=`, `/=` |

| Control Flow | Purpose |
|--------------|---------|
| `if-else` | Conditional branching |
| `switch` | Multiple branching |
| `for` | Count-based iteration |
| `while` | Condition-based iteration |
| `do-while` | Executes at least once |

---

## Exercises

### Exercise 1: Operator Evaluation Prediction

Without running the code, predict the output of each statement. Then compile and verify your answers.

```cpp
#include <iostream>

int main() {
    int a = 10, b = 3;

    // Predict each output before running
    std::cout << a / b << std::endl;          // ?
    std::cout << a % b << std::endl;          // ?
    std::cout << (double)a / b << std::endl;  // ?

    int x = 5;
    std::cout << x++ << std::endl;  // ?
    std::cout << x   << std::endl;  // ?
    std::cout << ++x << std::endl;  // ?

    // Predict: does the following print 0 or 1?
    int counter = 0;
    if (false && (++counter > 0)) {}
    if (true  || (++counter > 0)) {}
    std::cout << counter << std::endl;  // ?

    return 0;
}
```

After predicting, explain in your own words why `counter` has that value (short-circuit evaluation).

### Exercise 2: Bitwise Flag Operations

Implement a simple permission system using bitwise operators. Define three flags (`READ = 1`, `WRITE = 2`, `EXECUTE = 4`) and write a program that:

1. Creates a permission variable with READ and WRITE set.
2. Checks whether EXECUTE is set (it should not be).
3. Grants EXECUTE permission using bitwise OR.
4. Revokes WRITE permission using bitwise AND with NOT (`&= ~WRITE`).
5. Prints the final permissions in a human-readable format.

```cpp
#include <iostream>

int main() {
    const int READ    = 1;  // 001
    const int WRITE   = 2;  // 010
    const int EXECUTE = 4;  // 100

    int perms = READ | WRITE;  // Start: READ + WRITE

    // Add your steps here ...

    return 0;
}
```

### Exercise 3: Grade Calculator with switch

Write a function `char letterGrade(int score)` that converts a numeric score (0–100) to a letter grade using a `switch` statement on `score / 10`. Map 10 and 9 to `'A'`, 8 to `'B'`, 7 to `'C'`, 6 to `'D'`, and anything else to `'F'`. Test it with at least five different scores including boundary values (59, 60, 89, 90, 100).

### Exercise 4: Loop Pattern Challenge

Using nested `for` loops, print the following diamond pattern for a given `size` (shown here for `size = 4`):

```
   *
  ***
 *****
*******
 *****
  ***
   *
```

The top half (including the middle row) should have rows of `2*i - 1` stars (`i` from 1 to `size`), and the bottom half mirrors it. Use only `std::cout`, loops, and `continue` or `break` if needed — do not use any string manipulation functions.

### Exercise 5: Input Validation Loop

Write a complete program that repeatedly prompts the user to enter an integer in the range [1, 100] using a `do-while` loop. The loop must continue until a valid value is entered. Once a valid number is received, use the ternary operator to classify it as `"low"` (1–33), `"medium"` (34–66), or `"high"` (67–100), then print the classification.

---

## Next Step

Let's learn about functions in [04_Functions.md](./04_Functions.md)!
