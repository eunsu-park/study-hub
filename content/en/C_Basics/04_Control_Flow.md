# Control Flow

**Previous**: [Operators and Expressions](./03_Operators_and_Expressions.md) | **Next**: [Functions and Scope](./05_Functions_and_Scope.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Write branching logic using `if`, `else if`, and `else` statements
2. Replace multi-way branches with `switch`-`case` and explain fall-through behavior
3. Implement counted loops with `for` and conditional loops with `while` and `do-while`
4. Control loop execution using `break`, `continue`, and nested loop patterns
5. Explain why `goto` exists in C and when (if ever) it is appropriate

---

Programs become useful when they can make decisions and repeat actions. Control flow statements let you branch execution based on conditions and loop over blocks of code until a criterion is met. C provides a compact set of control flow constructs that, once mastered, give you precise control over program execution.

## 1. if / else if / else

The `if` statement is the most fundamental branching construct. It executes a block of code only when a condition is true (non-zero).

### Basic Syntax

```c
if (condition) {
    /* executed when condition is non-zero (true) */
}

if (condition) {
    /* true branch */
} else {
    /* false branch */
}

if (condition1) {
    /* ... */
} else if (condition2) {
    /* ... */
} else if (condition3) {
    /* ... */
} else {
    /* none of the above */
}
```

### Examples

```c
#include <stdio.h>

int main(void) {
    int temperature = 22;

    if (temperature > 30) {
        printf("It's hot outside\n");
    } else if (temperature > 20) {
        printf("It's warm outside\n");    /* This prints */
    } else if (temperature > 10) {
        printf("It's cool outside\n");
    } else {
        printf("It's cold outside\n");
    }

    return 0;
}
```

### Nesting

```c
#include <stdio.h>

int main(void) {
    int age = 25;
    int has_license = 1;

    if (age >= 18) {
        if (has_license) {
            printf("You can drive\n");
        } else {
            printf("Get a license first\n");
        }
    } else {
        printf("Too young to drive\n");
    }

    return 0;
}
```

### Common Pitfalls

```c
#include <stdio.h>

int main(void) {
    int x = 5;

    /* Pitfall 1: = vs == */
    if (x = 10) {              /* BUG: assigns 10 to x, always true */
        printf("x is now %d\n", x);
    }

    /* Pitfall 2: Dangling else */
    int a = 1, b = 0;
    if (a)
        if (b)
            printf("a and b\n");
    else                       /* This else belongs to the INNER if, not the outer */
        printf("This might surprise you\n");  /* Prints when a=1 and b=0 */

    /* Fix: always use braces */
    if (a) {
        if (b) {
            printf("a and b\n");
        }
    } else {
        printf("not a\n");
    }

    /* Pitfall 3: Empty statement */
    if (x > 0);  /* WARNING: semicolon makes this a no-op */
    {
        printf("This always executes regardless of x\n");
    }

    return 0;
}
```

> **Best Practice**: Always use braces `{}` even for single-statement bodies. It prevents the dangling-else problem and makes future modifications safer.

---

## 2. switch-case

The `switch` statement selects among multiple alternatives based on the value of an integer expression. It is often cleaner than a long `if-else if` chain.

### Syntax

```c
switch (expression) {
    case constant1:
        /* statements */
        break;
    case constant2:
        /* statements */
        break;
    default:
        /* none of the above */
        break;
}
```

### Example: Menu Selection

```c
#include <stdio.h>

int main(void) {
    int choice;
    printf("Menu:\n");
    printf("1. New Game\n");
    printf("2. Load Game\n");
    printf("3. Settings\n");
    printf("4. Quit\n");
    printf("Enter choice: ");
    scanf("%d", &choice);

    switch (choice) {
        case 1:
            printf("Starting new game...\n");
            break;
        case 2:
            printf("Loading saved game...\n");
            break;
        case 3:
            printf("Opening settings...\n");
            break;
        case 4:
            printf("Goodbye!\n");
            break;
        default:
            printf("Invalid choice\n");
            break;
    }

    return 0;
}
```

### Fall-Through Behavior

Without `break`, execution **falls through** to the next case. This is sometimes intentional.

```c
#include <stdio.h>

int main(void) {
    char grade = 'B';

    switch (grade) {
        case 'A':
        case 'B':            /* fall-through: A and B both print "Good" */
            printf("Good job!\n");
            break;
        case 'C':
            printf("Average\n");
            break;
        case 'D':
        case 'F':            /* fall-through: D and F both print "Needs improvement" */
            printf("Needs improvement\n");
            break;
        default:
            printf("Invalid grade\n");
            break;
    }

    return 0;
}
```

### Intentional Fall-Through: Days in a Month

```c
#include <stdio.h>

int main(void) {
    int month = 2, year = 2024;
    int days;

    switch (month) {
        case 2:
            days = (year % 4 == 0 && (year % 100 != 0 || year % 400 == 0))
                   ? 29 : 28;
            break;
        case 4: case 6: case 9: case 11:  /* 30-day months */
            days = 30;
            break;
        default:                            /* 31-day months */
            days = 31;
            break;
    }

    printf("Month %d in year %d has %d days\n", month, year, days);
    return 0;
}
```

### Constraints

- The `switch` expression must be an integer type (`int`, `char`, `enum`, etc.) -- **not** `float`, `double`, or `char *`.
- Each `case` label must be a **compile-time constant**.
- Duplicate case values are not allowed.

---

## 3. for Loop

The `for` loop is C's workhorse for counted iteration. It packs initialization, condition, and update into a single line.

### Syntax

```c
for (initialization; condition; update) {
    /* body */
}
```

Execution order:
1. **Initialization** — runs once before the loop starts.
2. **Condition** — checked before each iteration; loop ends when false.
3. **Body** — executed if condition is true.
4. **Update** — runs after each iteration, then back to step 2.

### Examples

```c
#include <stdio.h>

int main(void) {
    /* Count from 0 to 4 */
    for (int i = 0; i < 5; i++) {
        printf("%d ", i);
    }
    printf("\n");  /* 0 1 2 3 4 */

    /* Count down */
    for (int i = 10; i > 0; i--) {
        printf("%d ", i);
    }
    printf("\n");  /* 10 9 8 7 6 5 4 3 2 1 */

    /* Step by 2 */
    for (int i = 0; i <= 20; i += 2) {
        printf("%d ", i);
    }
    printf("\n");  /* 0 2 4 6 8 10 12 14 16 18 20 */

    /* Sum 1 to 100 */
    int sum = 0;
    for (int i = 1; i <= 100; i++) {
        sum += i;
    }
    printf("Sum 1..100 = %d\n", sum);  /* 5050 */

    return 0;
}
```

### Multiple Variables in a for Loop

```c
#include <stdio.h>

int main(void) {
    /* Two loop variables converging */
    for (int lo = 0, hi = 10; lo < hi; lo++, hi--) {
        printf("lo=%d hi=%d\n", lo, hi);
    }
    /* lo=0 hi=10, lo=1 hi=9, ..., lo=4 hi=6 */

    return 0;
}
```

### Infinite Loop

```c
/* Infinite loop — must use break or return to exit */
for (;;) {
    printf("Running forever...\n");
    break;  /* exit immediately for this example */
}
```

---

## 4. while Loop

The `while` loop repeats a block as long as its condition is true. It is an **entry-controlled** loop: if the condition is false initially, the body never executes.

### Syntax

```c
while (condition) {
    /* body */
}
```

### Examples

```c
#include <stdio.h>

int main(void) {
    /* Count up */
    int i = 0;
    while (i < 5) {
        printf("%d ", i);
        i++;
    }
    printf("\n");  /* 0 1 2 3 4 */

    /* Sentinel value: read until -1 */
    int num, total = 0, count = 0;
    printf("Enter numbers (-1 to stop): ");
    scanf("%d", &num);

    while (num != -1) {
        total += num;
        count++;
        scanf("%d", &num);
    }

    if (count > 0) {
        printf("Average: %.2f\n", (double)total / count);
    }

    return 0;
}
```

### Digit Counter

```c
#include <stdio.h>

int main(void) {
    int number = 123456;
    int digits = 0;
    int temp = number;

    if (temp == 0) {
        digits = 1;
    } else {
        while (temp != 0) {
            temp /= 10;
            digits++;
        }
    }

    printf("%d has %d digits\n", number, digits);  /* 123456 has 6 digits */
    return 0;
}
```

---

## 5. do-while Loop

The `do-while` loop is an **exit-controlled** loop: the body always executes at least once before the condition is checked.

### Syntax

```c
do {
    /* body — always executes at least once */
} while (condition);  /* note the semicolon! */
```

### Input Validation Pattern

The most common use of `do-while` is input validation: prompt the user, then repeat if the input is invalid.

```c
#include <stdio.h>

int main(void) {
    int choice;

    do {
        printf("Enter a number between 1 and 10: ");
        scanf("%d", &choice);

        if (choice < 1 || choice > 10) {
            printf("Invalid! Try again.\n");
        }
    } while (choice < 1 || choice > 10);

    printf("You chose: %d\n", choice);
    return 0;
}
```

### Menu Loop

```c
#include <stdio.h>

int main(void) {
    int option;

    do {
        printf("\n--- Menu ---\n");
        printf("1. Say Hello\n");
        printf("2. Say Goodbye\n");
        printf("0. Exit\n");
        printf("Choice: ");
        scanf("%d", &option);

        switch (option) {
            case 1: printf("Hello!\n"); break;
            case 2: printf("Goodbye!\n"); break;
            case 0: printf("Exiting...\n"); break;
            default: printf("Unknown option\n"); break;
        }
    } while (option != 0);

    return 0;
}
```

### Comparison: while vs do-while

| Feature | `while` | `do-while` |
|---------|---------|------------|
| Condition check | Before body | After body |
| Minimum executions | 0 | 1 |
| Use case | General loops | Input validation, menu loops |

---

## 6. break and continue

### break

`break` immediately exits the innermost `for`, `while`, `do-while`, or `switch`.

```c
#include <stdio.h>

int main(void) {
    /* Find first multiple of 7 greater than 50 */
    for (int i = 51; ; i++) {
        if (i % 7 == 0) {
            printf("Found: %d\n", i);  /* 56 */
            break;
        }
    }

    /* Search an array */
    int data[] = {10, 25, 37, 42, 58};
    int target = 37;
    int found = 0;

    for (int i = 0; i < 5; i++) {
        if (data[i] == target) {
            printf("Found %d at index %d\n", target, i);
            found = 1;
            break;
        }
    }
    if (!found) {
        printf("%d not found\n", target);
    }

    return 0;
}
```

### continue

`continue` skips the rest of the current iteration and jumps to the next one.

```c
#include <stdio.h>

int main(void) {
    /* Print only odd numbers */
    for (int i = 0; i < 10; i++) {
        if (i % 2 == 0) {
            continue;  /* skip even numbers */
        }
        printf("%d ", i);
    }
    printf("\n");  /* 1 3 5 7 9 */

    /* Sum positive numbers, skip negatives */
    int values[] = {3, -1, 4, -1, 5, -9, 2, 6};
    int sum = 0;
    for (int i = 0; i < 8; i++) {
        if (values[i] < 0) {
            continue;
        }
        sum += values[i];
    }
    printf("Sum of positives: %d\n", sum);  /* 20 */

    return 0;
}
```

### break and continue in while Loops

```c
#include <stdio.h>

int main(void) {
    int i = 0;
    while (i < 100) {
        i++;
        if (i % 3 != 0) {
            continue;   /* skip non-multiples of 3 */
        }
        if (i > 20) {
            break;       /* stop after 20 */
        }
        printf("%d ", i);
    }
    printf("\n");  /* 3 6 9 12 15 18 */

    return 0;
}
```

---

## 7. Nested Loops

Loops can be placed inside other loops. This is common for working with two-dimensional data, generating patterns, and searching.

### Multiplication Table

```c
#include <stdio.h>

int main(void) {
    printf("    ");
    for (int j = 1; j <= 9; j++) {
        printf("%4d", j);
    }
    printf("\n    ------------------------------------\n");

    for (int i = 1; i <= 9; i++) {
        printf("%2d |", i);
        for (int j = 1; j <= 9; j++) {
            printf("%4d", i * j);
        }
        printf("\n");
    }

    return 0;
}
```

### Triangle Pattern

```c
#include <stdio.h>

int main(void) {
    int rows = 5;

    for (int i = 1; i <= rows; i++) {
        for (int j = 1; j <= i; j++) {
            printf("* ");
        }
        printf("\n");
    }
    /*
    *
    * *
    * * *
    * * * *
    * * * * *
    */

    return 0;
}
```

### Early Exit from Nested Loops

`break` only exits the innermost loop. To exit multiple levels, use a flag variable or `goto`.

```c
#include <stdio.h>

int main(void) {
    /* Method 1: Flag variable */
    int found = 0;
    int matrix[3][3] = {
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9}
    };
    int target = 5;

    for (int i = 0; i < 3 && !found; i++) {
        for (int j = 0; j < 3 && !found; j++) {
            if (matrix[i][j] == target) {
                printf("Found %d at [%d][%d]\n", target, i, j);
                found = 1;
            }
        }
    }

    return 0;
}
```

---

## 8. goto

The `goto` statement performs an unconditional jump to a labeled statement within the same function. While widely discouraged in general programming, it has one well-established use case in C: centralized error cleanup.

### Syntax

```c
goto label;

/* ... */

label:
    /* statements */
```

### Error Cleanup Pattern

When a function acquires multiple resources (memory, files, locks), `goto` provides a clean way to release them in reverse order if an error occurs.

```c
#include <stdio.h>
#include <stdlib.h>

int process_file(const char *path) {
    FILE *fp = NULL;
    char *buffer = NULL;
    int result = -1;  /* assume failure */

    fp = fopen(path, "r");
    if (fp == NULL) {
        fprintf(stderr, "Cannot open file\n");
        goto cleanup;
    }

    buffer = malloc(1024);
    if (buffer == NULL) {
        fprintf(stderr, "malloc failed\n");
        goto cleanup;
    }

    /* ... do work with fp and buffer ... */
    if (fgets(buffer, 1024, fp) == NULL) {
        fprintf(stderr, "Read failed\n");
        goto cleanup;
    }

    printf("Read: %s", buffer);
    result = 0;  /* success */

cleanup:
    free(buffer);       /* free(NULL) is safe */
    if (fp != NULL) {
        fclose(fp);
    }
    return result;
}

int main(void) {
    process_file("test.txt");
    return 0;
}
```

### Why Avoid goto in General

- It makes control flow hard to follow ("spaghetti code").
- It bypasses structured programming constructs.
- It can skip variable initializations, leading to bugs.

> **Rule of thumb**: Use `goto` only for forward jumps to a single cleanup label at the end of a function. Never jump backward (that is what loops are for). If your function has no resource cleanup needs, you almost certainly do not need `goto`.

---

## Exercises

### Exercise 1: Grade Classifier

Write a program that reads a numerical score (0-100) and prints the letter grade:

- 90-100: A
- 80-89: B
- 70-79: C
- 60-69: D
- Below 60: F

Use `if-else if-else`. Handle invalid input (below 0 or above 100) with an error message. Then rewrite it using a `switch` statement on `score / 10`.

### Exercise 2: FizzBuzz

Print numbers from 1 to 100. For multiples of 3, print "Fizz". For multiples of 5, print "Buzz". For multiples of both, print "FizzBuzz". Use a `for` loop. Then implement a second version using only `while`.

### Exercise 3: Number Guessing Loop

Write a program that generates a fixed "secret" number (e.g., 42) and repeatedly prompts the user to guess. After each guess, print "Too high", "Too low", or "Correct!". Use a `do-while` loop. Count the number of guesses and print it when the user wins.

### Exercise 4: Prime Number Finder

Write a program that prints all prime numbers from 2 to 200. Use nested loops: the outer loop iterates candidate numbers, the inner loop checks divisibility from 2 to the square root of the candidate. Use `break` to exit the inner loop early when a factor is found. Use `continue` in the outer loop to skip non-primes.

### Exercise 5: Pattern Printer

Write a program that prints the following diamond pattern for a given odd number `n` (e.g., `n = 7`):

```
   *
  ***
 *****
*******
 *****
  ***
   *
```

Use nested loops for spaces and stars. The program should work for any odd value of `n`.

---

## Next Steps

You can now direct the flow of your programs. Next, let's learn how to organize code into reusable blocks with [Functions and Scope](./05_Functions_and_Scope.md)!
