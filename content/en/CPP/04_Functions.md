# Functions

**Previous**: [Operators and Control Flow](./03_Operators_and_Control_Flow.md) | **Next**: [Arrays and Strings](./05_Arrays_and_Strings.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain the purpose of functions and describe the relationship between declaration (prototype) and definition
2. Compare pass-by-value, pass-by-reference, pass-by-pointer, and `const` reference parameter passing
3. Implement functions with default parameters and apply the right-to-left default rule
4. Design overloaded functions that differ by parameter type or count
5. Apply recursion to solve problems such as factorial and Fibonacci, and compare recursive vs. iterative solutions
6. Implement callback patterns using function pointers
7. Identify when to use `inline` functions for performance hints
8. Apply C++17 structured bindings to capture multiple return values from a function

---

Functions are the primary tool for managing complexity in any nontrivial program. They let you name a piece of logic, reuse it in many places, and reason about each piece independently. In professional C++ codebases, well-designed function signatures -- choosing the right parameter-passing strategy and return type -- are often what separate maintainable code from fragile code.

## 1. What is a Function?

A function is a block of code that performs a specific task.

```cpp
#include <iostream>

// Function definition
void sayHello() {
    std::cout << "Hello!" << std::endl;
}

int main() {
    sayHello();  // Function call
    sayHello();
    return 0;
}
```

### Function Structure

```cpp
return_type function_name(parameters) {
    // Function body
    return value;  // Can be omitted for void
}
```

---

## 2. Function Declaration and Definition

### Declaration (Prototype)

```cpp
#include <iostream>

// Function declaration (prototype)
int add(int a, int b);

int main() {
    std::cout << add(3, 5) << std::endl;  // 8
    return 0;
}

// Function definition
int add(int a, int b) {
    return a + b;
}
```

### Declaration in Header Files

```cpp
// math_utils.h
#ifndef MATH_UTILS_H
#define MATH_UTILS_H

int add(int a, int b);
int multiply(int a, int b);

#endif
```

```cpp
// math_utils.cpp
#include "math_utils.h"

int add(int a, int b) {
    return a + b;
}

int multiply(int a, int b) {
    return a * b;
}
```

```cpp
// main.cpp
#include <iostream>
#include "math_utils.h"

int main() {
    std::cout << add(3, 5) << std::endl;
    std::cout << multiply(3, 5) << std::endl;
    return 0;
}
```

---

## 3. Parameter Passing Methods

### Pass by Value

```cpp
#include <iostream>

void increment(int n) {  // n is a copy
    n++;
    std::cout << "Inside function: " << n << std::endl;
}

int main() {
    int x = 10;
    increment(x);
    std::cout << "After function: " << x << std::endl;  // 10 (unchanged)
    return 0;
}
```

### Pass by Reference

```cpp
#include <iostream>

void increment(int& n) {  // n is a reference to original
    n++;
    std::cout << "Inside function: " << n << std::endl;
}

int main() {
    int x = 10;
    increment(x);
    std::cout << "After function: " << x << std::endl;  // 11 (changed)
    return 0;
}
```

### Pass by Pointer

```cpp
#include <iostream>

void increment(int* n) {  // n is an address
    (*n)++;
    std::cout << "Inside function: " << *n << std::endl;
}

int main() {
    int x = 10;
    increment(&x);  // Pass address
    std::cout << "After function: " << x << std::endl;  // 11
    return 0;
}
```

### const Reference (Read-only)

```cpp
#include <iostream>
#include <string>

// Read without copy (efficient)
void printLength(const std::string& str) {
    std::cout << "Length: " << str.length() << std::endl;
    // str[0] = 'x';  // Error! Cannot modify const
}

int main() {
    std::string name = "Hello";
    printLength(name);
    return 0;
}
```

### When to Use Which Method?

| Situation | Recommended Method |
|-----------|-------------------|
| Small types (int, double) for reading | Pass by value |
| Large types for reading | `const T&` |
| Need to modify | `T&` |
| Need to allow nullptr | `T*` |

---

## 4. Return Values

### Single Value Return

```cpp
int square(int n) {
    return n * n;
}
```

### Reference Return (Use with Caution)

```cpp
#include <iostream>

int& getElement(int arr[], int index) {
    return arr[index];  // Return reference to array element
}

int main() {
    int arr[] = {1, 2, 3, 4, 5};

    getElement(arr, 2) = 100;  // arr[2] = 100
    std::cout << arr[2] << std::endl;  // 100

    return 0;
}
```

### Multiple Value Return (C++17 Structured Bindings)

```cpp
#include <iostream>
#include <tuple>

std::tuple<int, int, int> getStats(int arr[], int size) {
    int sum = 0, min = arr[0], max = arr[0];
    for (int i = 0; i < size; i++) {
        sum += arr[i];
        if (arr[i] < min) min = arr[i];
        if (arr[i] > max) max = arr[i];
    }
    return {sum, min, max};
}

int main() {
    int arr[] = {5, 2, 8, 1, 9};

    // C++17 structured bindings
    auto [sum, min, max] = getStats(arr, 5);
    std::cout << "Sum: " << sum << ", Min: " << min << ", Max: " << max << std::endl;

    return 0;
}
```

---

## 5. Default Parameters

```cpp
#include <iostream>

void greet(std::string name = "Guest", int times = 1) {
    for (int i = 0; i < times; i++) {
        std::cout << "Hello, " << name << "!" << std::endl;
    }
}

int main() {
    greet();                // Hello, Guest!
    greet("Alice");         // Hello, Alice!
    greet("Bob", 3);        // Hello, Bob! (3 times)
    return 0;
}
```

### Rules

```cpp
// Default values must be from right to left
void func(int a, int b = 10, int c = 20);  // OK
// void func(int a = 5, int b, int c = 20);  // Error!

// Only in declaration or definition, not both
void func(int a, int b = 10);  // Default in declaration
void func(int a, int b) { }    // No default in definition
```

---

## 6. Function Overloading

You can define multiple functions with the same name but different parameters.

```cpp
#include <iostream>

// Integer addition
int add(int a, int b) {
    return a + b;
}

// Floating-point addition
double add(double a, double b) {
    return a + b;
}

// Three-number addition
int add(int a, int b, int c) {
    return a + b + c;
}

int main() {
    std::cout << add(3, 5) << std::endl;        // int version: 8
    std::cout << add(3.5, 2.5) << std::endl;    // double version: 6.0
    std::cout << add(1, 2, 3) << std::endl;     // three-number version: 6
    return 0;
}
```

### Overloading Rules

```cpp
// Different parameter types: OK
void print(int n);
void print(double n);
void print(std::string s);

// Different number of parameters: OK
void print(int a);
void print(int a, int b);

// Different return type only: NOT allowed!
// int func(int a);
// double func(int a);  // Error!
```

---

## 7. inline Functions

Reduces call overhead for short functions.

```cpp
#include <iostream>

inline int square(int n) {
    return n * n;
}

int main() {
    std::cout << square(5) << std::endl;  // Compiler may substitute with 25
    return 0;
}
```

### When to Use

- When function body is short (1-2 lines)
- Frequently called functions
- Compiler makes final decision (inline is a hint)

---

## 8. Recursive Functions

A function that calls itself.

### Factorial

```cpp
#include <iostream>

int factorial(int n) {
    if (n <= 1) return 1;       // Base case
    return n * factorial(n - 1); // Recursive call
}

int main() {
    std::cout << "5! = " << factorial(5) << std::endl;  // 120
    return 0;
}
```

### Fibonacci

```cpp
#include <iostream>

int fibonacci(int n) {
    if (n <= 1) return n;
    return fibonacci(n - 1) + fibonacci(n - 2);
}

int main() {
    for (int i = 0; i < 10; i++) {
        std::cout << fibonacci(i) << " ";
    }
    std::cout << std::endl;  // 0 1 1 2 3 5 8 13 21 34
    return 0;
}
```

### Recursion vs Iteration

```cpp
// Iterative version (efficient)
int factorialLoop(int n) {
    int result = 1;
    for (int i = 2; i <= n; i++) {
        result *= i;
    }
    return result;
}
```

---

## 9. Function Pointers

Functions can be stored and passed like variables.

```cpp
#include <iostream>

int add(int a, int b) { return a + b; }
int subtract(int a, int b) { return a - b; }
int multiply(int a, int b) { return a * b; }

int main() {
    // Function pointer declaration
    int (*operation)(int, int);

    operation = add;
    std::cout << "Add: " << operation(5, 3) << std::endl;  // 8

    operation = subtract;
    std::cout << "Subtract: " << operation(5, 3) << std::endl;  // 2

    operation = multiply;
    std::cout << "Multiply: " << operation(5, 3) << std::endl;  // 15

    return 0;
}
```

### Callback Functions

```cpp
#include <iostream>

void processArray(int arr[], int size, int (*func)(int)) {
    for (int i = 0; i < size; i++) {
        arr[i] = func(arr[i]);
    }
}

int doubleIt(int n) { return n * 2; }
int squareIt(int n) { return n * n; }

int main() {
    int arr[] = {1, 2, 3, 4, 5};

    processArray(arr, 5, doubleIt);
    for (int n : arr) std::cout << n << " ";  // 2 4 6 8 10
    std::cout << std::endl;

    processArray(arr, 5, squareIt);
    for (int n : arr) std::cout << n << " ";  // 4 16 36 64 100
    std::cout << std::endl;

    return 0;
}
```

---

## 10. Lambda Expressions - Preview

From C++11, you can create anonymous functions.

```cpp
#include <iostream>

int main() {
    // Basic lambda
    auto add = [](int a, int b) {
        return a + b;
    };

    std::cout << add(3, 5) << std::endl;  // 8

    // Capture
    int multiplier = 10;
    auto multiply = [multiplier](int n) {
        return n * multiplier;
    };

    std::cout << multiply(5) << std::endl;  // 50

    return 0;
}
```

---

## 11. main Function Parameters

```cpp
#include <iostream>

int main(int argc, char* argv[]) {
    std::cout << "Argument count: " << argc << std::endl;

    for (int i = 0; i < argc; i++) {
        std::cout << "argv[" << i << "]: " << argv[i] << std::endl;
    }

    return 0;
}
```

Execution:
```bash
./program hello world
```

Output:
```
Argument count: 3
argv[0]: ./program
argv[1]: hello
argv[2]: world
```

---

## 12. Practice Examples

### Greatest Common Divisor (Euclidean Algorithm)

```cpp
#include <iostream>

int gcd(int a, int b) {
    if (b == 0) return a;
    return gcd(b, a % b);
}

int main() {
    std::cout << "gcd(48, 18) = " << gcd(48, 18) << std::endl;  // 6
    std::cout << "gcd(56, 98) = " << gcd(56, 98) << std::endl;  // 14
    return 0;
}
```

### Swap Two Values

```cpp
#include <iostream>

void swap(int& a, int& b) {
    int temp = a;
    a = b;
    b = temp;
}

int main() {
    int x = 5, y = 10;
    std::cout << "Before: x=" << x << ", y=" << y << std::endl;

    swap(x, y);
    std::cout << "After: x=" << x << ", y=" << y << std::endl;

    return 0;
}
```

---

## 13. Summary

| Concept | Description |
|---------|-------------|
| Function declaration | Function signature (prototype) |
| Function definition | Function body |
| Pass by value | Copy is passed, original unchanged |
| Pass by reference | Original is passed, can be modified |
| const reference | Read-only original passing |
| Default parameters | Arguments that can be omitted |
| Overloading | Same name, different parameters |
| inline | Code insertion instead of function call |
| Recursion | Function calling itself |

---

## Next Step

Let's learn about arrays and strings in [05_Arrays_and_Strings.md](./05_Arrays_and_Strings.md)!
