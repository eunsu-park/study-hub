# Project 1: Basic Arithmetic Calculator

**Previous**: [C Language Basics Quick Review](./02_C_Basics_Review.md) | **Next**: [Project 2: Number Guessing Game](./04_Project_Number_Guessing.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Read user input using `scanf` with proper format specifiers and the address-of operator
2. Route program flow through multiple branches using `switch-case` statements
3. Define and call functions that accept pointer parameters and return status codes
4. Handle runtime errors gracefully using input validation and return codes
5. Design a program that loops continuously until the user chooses to exit
6. Separate concerns by extracting input, computation, and output into distinct functions

---

A calculator is the "Hello World" of interactive programs. It forces you to handle every messy real-world detail at once -- reading user input that might be garbage, choosing the right operation, and reporting errors like division by zero. By the end of this project you will have a small but complete command-line tool, and more importantly, a reusable pattern for structuring any interactive C program.

## Step 1: Basic Calculator

### Requirements

```
Take two numbers and an operator as input, then output the result
Example: 10 + 5 → Result: 15
```

### Core Syntax: scanf

```c
#include <stdio.h>

int main(void) {
    int num;
    printf("Enter a number: ");
    scanf("%d", &num);        // & required! (pass address)
    printf("You entered: %d\n", num);

    // Multiple values
    int a, b;
    printf("Enter two numbers (space-separated): ");
    scanf("%d %d", &a, &b);
    printf("a=%d, b=%d\n", a, b);

    // Character input
    char op;
    printf("Enter operator: ");
    scanf(" %c", &op);        // Space before %c: ignore previous newline
    printf("Operator: %c\n", op);

    return 0;
}
```

### Core Syntax: switch-case

```c
char grade = 'B';

switch (grade) {
    case 'A':
        printf("Excellent\n");
        break;
    case 'B':
        printf("Good\n");
        break;
    case 'C':
        printf("Average\n");
        break;
    default:
        printf("Other\n");
        break;
}
```

### Implementation

```c
// calculator_v1.c
#include <stdio.h>

int main(void) {
    double num1, num2;
    char operator;

    printf("=== Simple Calculator ===\n");
    printf("Enter expression (e.g., 10 + 5): ");
    scanf("%lf %c %lf", &num1, &operator, &num2);

    double result;

    switch (operator) {
        case '+':
            result = num1 + num2;
            break;
        case '-':
            result = num1 - num2;
            break;
        case '*':
            result = num1 * num2;
            break;
        case '/':
            result = num1 / num2;
            break;
        default:
            printf("Error: Unsupported operator.\n");
            return 1;
    }

    printf("Result: %.2f %c %.2f = %.2f\n", num1, operator, num2, result);

    return 0;
}
```

### Execution Example

```
$ ./calculator_v1
=== Simple Calculator ===
Enter expression (e.g., 10 + 5): 10 + 5
Result: 10.00 + 5.00 = 15.00

$ ./calculator_v1
Enter expression (e.g., 10 + 5): 20 / 4
Result: 20.00 / 4.00 = 5.00
```

---

## Step 2: Add Error Handling

### Problem

```
20 / 0 → Result: inf (infinity) or error
```

### Improved Code

```c
// calculator_v2.c
#include <stdio.h>

int main(void) {
    double num1, num2;
    char operator;

    printf("=== Calculator v2 ===\n");
    printf("Enter expression (e.g., 10 + 5): ");

    // Input validation
    if (scanf("%lf %c %lf", &num1, &operator, &num2) != 3) {
        printf("Error: Invalid input format.\n");
        return 1;
    }

    double result;
    int error = 0;

    switch (operator) {
        case '+':
            result = num1 + num2;
            break;
        case '-':
            result = num1 - num2;
            break;
        case '*':
            result = num1 * num2;
            break;
        case '/':
            if (num2 == 0) {
                printf("Error: Cannot divide by zero.\n");
                error = 1;
            } else {
                result = num1 / num2;
            }
            break;
        default:
            printf("Error: '%c' is not a supported operator.\n", operator);
            error = 1;
            break;
    }

    if (!error) {
        printf("Result: %.2f %c %.2f = %.2f\n", num1, operator, num2, result);
    }

    return error;
}
```

---

## Step 3: Separate into Functions

### Structure

```
main() → get_input() → Get input
       → calculate() → Perform calculation
       → Output result
```

### Complete Code

```c
// calculator_v3.c
#include <stdio.h>

// Function declarations
int get_input(double *num1, char *op, double *num2);
int calculate(double num1, char op, double num2, double *result);
void print_result(double num1, char op, double num2, double result);

int main(void) {
    double num1, num2, result;
    char operator;

    printf("=== Calculator v3 ===\n");

    // Get input
    if (get_input(&num1, &operator, &num2) != 0) {
        printf("Error: Invalid input format.\n");
        return 1;
    }

    // Calculate
    if (calculate(num1, operator, num2, &result) != 0) {
        return 1;
    }

    // Print result
    print_result(num1, operator, num2, result);

    return 0;
}

// Input function
int get_input(double *num1, char *op, double *num2) {
    printf("Enter expression (e.g., 10 + 5): ");
    if (scanf("%lf %c %lf", num1, op, num2) != 3) {
        return -1;  // Error
    }
    return 0;  // Success
}

// Calculate function
int calculate(double num1, char op, double num2, double *result) {
    switch (op) {
        case '+':
            *result = num1 + num2;
            break;
        case '-':
            *result = num1 - num2;
            break;
        case '*':
            *result = num1 * num2;
            break;
        case '/':
            if (num2 == 0) {
                printf("Error: Cannot divide by zero.\n");
                return -1;
            }
            *result = num1 / num2;
            break;
        case '%':
            // Integer modulo operation
            if (num2 == 0) {
                printf("Error: Cannot divide by zero.\n");
                return -1;
            }
            *result = (int)num1 % (int)num2;
            break;
        default:
            printf("Error: '%c' is not a supported operator.\n", op);
            return -1;
    }
    return 0;
}

// Output function
void print_result(double num1, char op, double num2, double result) {
    printf("Result: %.2f %c %.2f = %.2f\n", num1, op, num2, result);
}
```

---

## Step 4: Repeat Calculations (Final Version)

### Complete Code

```c
// calculator.c (final)
#include <stdio.h>
#include <stdlib.h>

// Function declarations
int get_input(double *num1, char *op, double *num2);
int calculate(double num1, char op, double num2, double *result);
void print_result(double num1, char op, double num2, double result);
void print_help(void);
void clear_input_buffer(void);

int main(void) {
    double num1, num2, result;
    char operator;
    char continue_calc;

    printf("=============================\n");
    printf("     Simple Calculator v4    \n");
    printf("=============================\n");
    print_help();

    do {
        // Get input
        if (get_input(&num1, &operator, &num2) != 0) {
            printf("Error: Invalid input format.\n");
            clear_input_buffer();
            continue;
        }

        // Calculate
        if (calculate(num1, operator, num2, &result) == 0) {
            // Print result
            print_result(num1, operator, num2, result);
        }

        // Continue?
        printf("\nContinue? (y/n): ");
        scanf(" %c", &continue_calc);
        clear_input_buffer();
        printf("\n");

    } while (continue_calc == 'y' || continue_calc == 'Y');

    printf("Exiting calculator.\n");
    return 0;
}

int get_input(double *num1, char *op, double *num2) {
    printf("\nEnter expression: ");
    if (scanf("%lf %c %lf", num1, op, num2) != 3) {
        return -1;
    }
    return 0;
}

int calculate(double num1, char op, double num2, double *result) {
    switch (op) {
        case '+':
            *result = num1 + num2;
            break;
        case '-':
            *result = num1 - num2;
            break;
        case '*':
        case 'x':
        case 'X':
            *result = num1 * num2;
            break;
        case '/':
            if (num2 == 0) {
                printf("Error: Cannot divide by zero.\n");
                return -1;
            }
            *result = num1 / num2;
            break;
        case '%':
            if (num2 == 0) {
                printf("Error: Cannot divide by zero.\n");
                return -1;
            }
            *result = (int)num1 % (int)num2;
            break;
        case '^':
            // Simple exponentiation (positive integers only)
            *result = 1;
            for (int i = 0; i < (int)num2; i++) {
                *result *= num1;
            }
            break;
        default:
            printf("Error: '%c' is not a supported operator.\n", op);
            return -1;
    }
    return 0;
}

void print_result(double num1, char op, double num2, double result) {
    printf(">>> %.4g %c %.4g = %.4g\n", num1, op, num2, result);
}

void print_help(void) {
    printf("\nSupported operators: + - * / %% ^\n");
    printf("Input format: number operator number\n");
    printf("Examples: 10 + 5, 20 / 4, 2 ^ 10\n");
}

void clear_input_buffer(void) {
    int c;
    while ((c = getchar()) != '\n' && c != EOF);
}
```

### Execution Example

```
=============================
     Simple Calculator v4
=============================

Supported operators: + - * / % ^
Input format: number operator number
Examples: 10 + 5, 20 / 4, 2 ^ 10

Enter expression: 100 + 250
>>> 100 + 250 = 350

Continue? (y/n): y

Enter expression: 2 ^ 10
>>> 2 ^ 10 = 1024

Continue? (y/n): y

Enter expression: 10 / 0
Error: Cannot divide by zero.

Continue? (y/n): n

Exiting calculator.
```

---

## Compile and Run

```bash
# Compile
gcc -Wall -Wextra -std=c11 calculator.c -o calculator

# Run
./calculator
```

---

## Summary of What You've Learned

| Concept | Description |
|------|------|
| `scanf` | Read input in specified format |
| `switch-case` | Branch based on value |
| Function separation | Code structure, reusability |
| Pointer parameters | Modify values in functions |
| Error handling | Use return values to indicate success/failure |

---

## Practice Problems

1. **Add square root operation**: Add `sqrt` operator (Hint: `#include <math.h>`, `sqrt()`)

2. **Calculation history**: Store last 10 calculation results in array and display

3. **Support parentheses**: Handle expressions like `(10 + 5) * 2` (challenging!)

---

## Next Steps

[Project 2: Number Guessing Game](./04_Project_Number_Guessing.md) → Let's build a game!
