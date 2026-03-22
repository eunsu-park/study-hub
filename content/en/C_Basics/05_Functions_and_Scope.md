# Functions and Scope

**Previous**: [Control Flow](./04_Control_Flow.md) | **Next**: [Arrays and Strings](./06_Arrays_and_Strings.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Declare function prototypes and define functions with parameters and return values
2. Explain call-by-value semantics and simulate call-by-reference using pointers
3. Apply scope rules (block, function, file) and the `static` keyword for persistence
4. Write recursive functions with proper base cases
5. Use variadic functions basics (brief introduction to `stdarg.h`)

---

Functions are the primary mechanism for organizing C code into manageable, reusable pieces. A well-designed function does one thing, does it well, and communicates its purpose through its name and parameters. Understanding how C passes arguments, how variables are scoped, and where values are stored in memory will make you a far more effective C programmer.

## 1. Function Declaration and Definition

A C function has two parts: a **declaration** (prototype) that announces its existence and a **definition** that provides the implementation.

### Syntax

```c
/* Declaration (prototype) — tells the compiler the function's signature */
return_type function_name(parameter_list);

/* Definition — provides the actual implementation */
return_type function_name(parameter_list) {
    /* body */
    return value;  /* omit for void functions */
}
```

### Complete Example

```c
#include <stdio.h>

/* Declaration (prototype) */
int add(int a, int b);
void greet(const char *name);

int main(void) {
    int sum = add(3, 4);
    printf("3 + 4 = %d\n", sum);  /* 7 */

    greet("Alice");  /* Hello, Alice! */

    return 0;
}

/* Definition */
int add(int a, int b) {
    return a + b;
}

void greet(const char *name) {
    printf("Hello, %s!\n", name);
}
```

### void Functions

A function that returns nothing uses `void` as its return type. A function that takes no parameters uses `void` in the parameter list.

```c
#include <stdio.h>

/* Takes nothing, returns nothing */
void print_separator(void) {
    printf("========================\n");
}

/* Takes parameters, returns nothing */
void print_range(int start, int end) {
    for (int i = start; i <= end; i++) {
        printf("%d ", i);
    }
    printf("\n");
}

int main(void) {
    print_separator();
    print_range(1, 10);
    print_separator();
    return 0;
}
```

### Multiple Return Statements

A function can have multiple `return` statements. Execution ends at the first one reached.

```c
int absolute(int n) {
    if (n >= 0) {
        return n;
    }
    return -n;
}

char grade(int score) {
    if (score >= 90) return 'A';
    if (score >= 80) return 'B';
    if (score >= 70) return 'C';
    if (score >= 60) return 'D';
    return 'F';
}
```

---

## 2. Parameters and Return Values

### Pass by Value

C always passes arguments **by value**. The function receives a copy of each argument, so modifying a parameter inside the function does not affect the original variable.

```c
#include <stdio.h>

void try_to_modify(int x) {
    x = 999;  /* modifies the local copy only */
    printf("Inside function: x = %d\n", x);
}

int main(void) {
    int num = 42;
    try_to_modify(num);
    printf("After function:  num = %d\n", num);  /* still 42 */
    return 0;
}
```

### Simulating Pass by Reference with Pointers

To modify the caller's variable, pass a **pointer** to it. (Pointers are covered in depth in Lesson 07, but the pattern is introduced here.)

```c
#include <stdio.h>

void swap(int *a, int *b) {
    int temp = *a;
    *a = *b;
    *b = temp;
}

int main(void) {
    int x = 10, y = 20;
    printf("Before: x=%d y=%d\n", x, y);

    swap(&x, &y);  /* pass addresses */
    printf("After:  x=%d y=%d\n", x, y);  /* x=20 y=10 */

    return 0;
}
```

### Returning Multiple Values via Pointers

Since a function can return only one value directly, use output parameters (pointers) to return additional results.

```c
#include <stdio.h>

void divide(int dividend, int divisor, int *quotient, int *remainder) {
    *quotient  = dividend / divisor;
    *remainder = dividend % divisor;
}

int main(void) {
    int q, r;
    divide(17, 5, &q, &r);
    printf("17 / 5 = %d remainder %d\n", q, r);  /* 3 remainder 2 */
    return 0;
}
```

### Returning Structs

For grouping related return values, you can return a struct (covered in Lesson 08):

```c
#include <stdio.h>

typedef struct {
    int quot;
    int rem;
} DivResult;

DivResult divide2(int a, int b) {
    DivResult result = { a / b, a % b };
    return result;
}

int main(void) {
    DivResult dr = divide2(17, 5);
    printf("17 / 5 = %d remainder %d\n", dr.quot, dr.rem);
    return 0;
}
```

---

## 3. Function Prototypes

A **prototype** tells the compiler about a function's return type and parameter types before the function is defined. This is necessary when a function is called before its definition appears in the source file.

### Why Prototypes Matter

```c
#include <stdio.h>

/* Without a prototype, the compiler does not know about add()
   when it encounters the call in main(). In C89, this would
   trigger an implicit declaration (now removed in C99+).
   In C99 and later, it is an error. */

/* Prototype */
int add(int a, int b);

int main(void) {
    printf("%d\n", add(3, 4));  /* OK — compiler knows add's signature */
    return 0;
}

int add(int a, int b) {
    return a + b;
}
```

### Header File Convention

In multi-file projects, prototypes go in header files (`.h`) and definitions in source files (`.c`).

```c
/* math_utils.h */
#ifndef MATH_UTILS_H
#define MATH_UTILS_H

int add(int a, int b);
int multiply(int a, int b);
double average(const int *arr, int n);

#endif /* MATH_UTILS_H */
```

```c
/* math_utils.c */
#include "math_utils.h"

int add(int a, int b) {
    return a + b;
}

int multiply(int a, int b) {
    return a * b;
}

double average(const int *arr, int n) {
    int sum = 0;
    for (int i = 0; i < n; i++) {
        sum += arr[i];
    }
    return (double)sum / n;
}
```

```c
/* main.c */
#include <stdio.h>
#include "math_utils.h"

int main(void) {
    printf("3 + 4 = %d\n", add(3, 4));
    printf("3 * 4 = %d\n", multiply(3, 4));

    int data[] = {10, 20, 30, 40, 50};
    printf("average = %.1f\n", average(data, 5));
    return 0;
}
```

---

## 4. Scope Rules

**Scope** determines where a variable is visible and accessible. C has several levels of scope.

### Block Scope

Variables declared inside a block `{}` are visible only within that block.

```c
#include <stdio.h>

int main(void) {
    int x = 10;

    {
        int y = 20;           /* y is visible only in this block */
        printf("x=%d y=%d\n", x, y);  /* OK */
    }

    /* printf("y=%d\n", y);  — ERROR: y is not visible here */

    /* for loop variable has block scope (C99+) */
    for (int i = 0; i < 3; i++) {
        printf("%d ", i);
    }
    /* printf("%d\n", i);  — ERROR: i is out of scope */
    printf("\n");

    return 0;
}
```

### Function Scope

Labels (used with `goto`) have function scope -- they are visible throughout the entire function regardless of block nesting.

### File Scope (Global Variables)

Variables declared outside all functions have **file scope**. They are accessible from the point of declaration to the end of the file.

```c
#include <stdio.h>

int global_count = 0;   /* file scope — accessible everywhere below */

void increment(void) {
    global_count++;
}

int main(void) {
    increment();
    increment();
    printf("count = %d\n", global_count);  /* 2 */
    return 0;
}
```

> **Best Practice**: Minimize use of global variables. They make code harder to reason about, test, and maintain. Prefer passing data through function parameters.

### Shadowing

An inner scope can declare a variable with the same name as an outer scope, **shadowing** it.

```c
#include <stdio.h>

int x = 100;  /* global */

int main(void) {
    int x = 50;  /* shadows global x */
    printf("x = %d\n", x);  /* 50 */

    {
        int x = 10;  /* shadows the main() x */
        printf("x = %d\n", x);  /* 10 */
    }

    printf("x = %d\n", x);  /* 50 — back to main's x */
    return 0;
}
```

> **Warning**: Shadowing is legal but confusing. Compiling with `-Wshadow` enables a warning for this.

---

## 5. Storage Classes

Storage classes control the **lifetime** and **linkage** of variables.

| Keyword | Scope | Lifetime | Default Value | Notes |
|---------|-------|----------|---------------|-------|
| `auto` | Block | Block duration | Undefined (garbage) | Default for local variables; keyword rarely used |
| `static` (local) | Block | Program duration | 0 | Retains value between calls |
| `static` (file) | File | Program duration | 0 | Not visible outside the file |
| `extern` | File+ | Program duration | 0 | Declared in one file, accessible from others |
| `register` | Block | Block duration | Undefined | Hint to store in CPU register (rarely used today) |

### static Local Variables

A `static` local variable is initialized once and persists across function calls.

```c
#include <stdio.h>

int next_id(void) {
    static int id = 0;  /* initialized once; persists between calls */
    id++;
    return id;
}

int main(void) {
    printf("ID: %d\n", next_id());  /* 1 */
    printf("ID: %d\n", next_id());  /* 2 */
    printf("ID: %d\n", next_id());  /* 3 */
    return 0;
}
```

### static at File Scope

A `static` global variable or function is **internal** to its translation unit (source file). Other files cannot access it, even with `extern`.

```c
/* helpers.c */
static int internal_counter = 0;  /* only visible in helpers.c */

static void helper(void) {        /* only callable from helpers.c */
    internal_counter++;
}

void public_function(void) {      /* visible to other files */
    helper();
}
```

### extern

`extern` declares a variable that is **defined** in another file.

```c
/* config.c */
int max_connections = 100;  /* definition */

/* main.c */
#include <stdio.h>

extern int max_connections;  /* declaration — uses config.c's definition */

int main(void) {
    printf("Max connections: %d\n", max_connections);
    return 0;
}
```

Compile both files together: `gcc main.c config.c -o app`

---

## 6. Recursion

A recursive function calls itself. Every recursive function needs:

1. A **base case** that stops the recursion.
2. A **recursive case** that moves toward the base case.

### Factorial

```c
#include <stdio.h>

long long factorial(int n) {
    if (n <= 1) {
        return 1;          /* base case */
    }
    return n * factorial(n - 1);  /* recursive case */
}

int main(void) {
    for (int i = 0; i <= 10; i++) {
        printf("%2d! = %lld\n", i, factorial(i));
    }
    return 0;
}
```

### Fibonacci

```c
#include <stdio.h>

/* Simple recursive Fibonacci — exponential time, for illustration only */
int fib(int n) {
    if (n <= 0) return 0;
    if (n == 1) return 1;
    return fib(n - 1) + fib(n - 2);
}

/* Iterative version — linear time */
int fib_iter(int n) {
    if (n <= 0) return 0;
    int prev = 0, curr = 1;
    for (int i = 2; i <= n; i++) {
        int next = prev + curr;
        prev = curr;
        curr = next;
    }
    return curr;
}

int main(void) {
    printf("Recursive: fib(10) = %d\n", fib(10));       /* 55 */
    printf("Iterative: fib(10) = %d\n", fib_iter(10));  /* 55 */
    return 0;
}
```

### Stack Usage and Limits

Each recursive call adds a **stack frame** to the call stack. Too many nested calls cause a **stack overflow**.

```c
#include <stdio.h>

void count_down(int n) {
    printf("%d\n", n);
    if (n > 0) {
        count_down(n - 1);
    }
}

int main(void) {
    count_down(10);        /* fine */
    /* count_down(1000000);  — stack overflow! */
    return 0;
}
```

### Tail Recursion

When the recursive call is the very last operation, it is called **tail recursion**. Some compilers (with optimization) can convert tail recursion into a loop, eliminating stack growth.

```c
/* Tail-recursive factorial */
long long factorial_tail(int n, long long acc) {
    if (n <= 1) return acc;
    return factorial_tail(n - 1, n * acc);  /* tail position */
}

/* Wrapper */
long long factorial2(int n) {
    return factorial_tail(n, 1);
}
```

> **Note**: The C standard does not require tail-call optimization, but GCC and Clang perform it with `-O2` or higher.

---

## 7. Variadic Functions

Variadic functions accept a variable number of arguments. The most familiar example is `printf`. To write your own, use `<stdarg.h>`.

### The stdarg.h Macros

| Macro | Purpose |
|-------|---------|
| `va_list` | Type to hold variadic argument state |
| `va_start(ap, last_fixed)` | Initialize `ap` after the last fixed parameter |
| `va_arg(ap, type)` | Retrieve the next argument as `type` |
| `va_end(ap)` | Clean up |

### Example: Sum of Variable Arguments

```c
#include <stdio.h>
#include <stdarg.h>

/* count: number of integers that follow */
int sum(int count, ...) {
    va_list ap;
    va_start(ap, count);

    int total = 0;
    for (int i = 0; i < count; i++) {
        total += va_arg(ap, int);
    }

    va_end(ap);
    return total;
}

int main(void) {
    printf("sum(3, 10, 20, 30)  = %d\n", sum(3, 10, 20, 30));   /* 60 */
    printf("sum(5, 1,2,3,4,5)   = %d\n", sum(5, 1, 2, 3, 4, 5)); /* 15 */
    return 0;
}
```

### Example: Custom Logger

```c
#include <stdio.h>
#include <stdarg.h>

void log_message(const char *level, const char *fmt, ...) {
    printf("[%s] ", level);

    va_list ap;
    va_start(ap, fmt);
    vprintf(fmt, ap);   /* vprintf takes a va_list */
    va_end(ap);

    printf("\n");
}

int main(void) {
    log_message("INFO",  "Server started on port %d", 8080);
    log_message("WARN",  "Memory usage at %d%%", 85);
    log_message("ERROR", "Failed to open '%s': code %d", "data.csv", -1);
    return 0;
}
```

> **Caution**: Variadic functions have no type checking for the variable arguments. Passing the wrong type is undefined behavior. Use them sparingly and document the expected types carefully.

---

## 8. Function Pointers

A **function pointer** stores the address of a function and allows it to be called indirectly. This is C's mechanism for callbacks, dispatch tables, and pluggable behavior.

### Declaration and Basic Use

```c
#include <stdio.h>

int add(int a, int b) { return a + b; }
int mul(int a, int b) { return a * b; }

int main(void) {
    /* Declare a function pointer: return type (*name)(param types) */
    int (*fp)(int, int) = add;   /* fp points to add */

    printf("add via pointer: %d\n", fp(3, 4));  /* 7 */

    fp = mul;                    /* reassign to a different function */
    printf("mul via pointer: %d\n", fp(3, 4));  /* 12 */

    return 0;
}
```

### typedef for Readability

Function pointer syntax becomes unwieldy in real code. A `typedef` gives the type a clean name.

```c
typedef int (*operation_t)(int, int);   /* operation_t is now a type */

operation_t op = add;
printf("%d\n", op(10, 5));  /* 15 */
```

### Callback Pattern

Passing a function pointer to another function is the **callback pattern** — the foundation of `qsort`, `bsearch`, signal handlers, and many library APIs.

```c
#include <stdio.h>

typedef int (*operation_t)(int, int);

int compute(int x, int y, operation_t op) {
    return op(x, y);
}

int subtract(int a, int b) { return a - b; }

int main(void) {
    printf("compute(10, 3, add)      = %d\n", compute(10, 3, add));       /* 13 */
    printf("compute(10, 3, subtract) = %d\n", compute(10, 3, subtract));  /* 7  */
    printf("compute(10, 3, mul)      = %d\n", compute(10, 3, mul));       /* 30 */
    return 0;
}
```

> **Connection to the standard library**: `qsort` uses exactly this pattern — it accepts a comparator function pointer `int (*compar)(const void *, const void *)` so you can sort any data type with any ordering rule.

---

## Exercises

### Exercise 1: Math Library

Write the following functions and test them in `main`:

1. `int power(int base, int exp)` — compute base^exp using a loop (assume exp >= 0).
2. `int gcd(int a, int b)` — compute the greatest common divisor using Euclid's algorithm (iterative).
3. `int is_prime(int n)` — return 1 if `n` is prime, 0 otherwise.
4. `void print_primes(int start, int end)` — print all primes in the range [start, end].

Place prototypes at the top of the file or in a separate header.

### Exercise 2: Scope Detective

Predict the output of this program without running it, then verify:

```c
#include <stdio.h>

int x = 1;

void f(void) {
    int x = 10;
    printf("f: x = %d\n", x);
    {
        int x = 20;
        printf("f inner: x = %d\n", x);
    }
    printf("f after block: x = %d\n", x);
}

void g(void) {
    printf("g: x = %d\n", x);
    x = 5;
}

int main(void) {
    printf("main: x = %d\n", x);
    f();
    printf("main after f: x = %d\n", x);
    g();
    printf("main after g: x = %d\n", x);
    return 0;
}
```

Write a comment next to each `printf` with the expected output and explain which `x` is being referenced.

### Exercise 3: Static Counter

Write a function `int unique_id(void)` that returns a different integer each time it is called (1, 2, 3, ...) using a `static` local variable. Then write a second function `void reset_id(void)` that resets the counter to 0. Can `reset_id` access the `static` variable inside `unique_id` directly? If not, how would you restructure the code to support resetting?

### Exercise 4: Recursive Power

Write a recursive version of the power function: `long long power_rec(int base, int exp)`. Handle three cases:

1. `exp == 0` returns 1
2. `exp` is even: `base^exp = (base^(exp/2))^2`
3. `exp` is odd: `base^exp = base * base^(exp-1)`

This is known as **exponentiation by squaring** and runs in O(log n) time. Test it with `power_rec(2, 30)` (expected: 1073741824).

### Exercise 5: Mini printf

Write a simplified `my_printf(const char *fmt, ...)` that supports only three format specifiers:

- `%d` — print an `int`
- `%s` — print a `char *`
- `%c` — print a `char`

Use `stdarg.h` to iterate through the format string character by character. When you encounter `%`, read the next character to determine the type, then use `va_arg` to retrieve and print the value. Test with:

```c
my_printf("Name: %s, Age: %d, Grade: %c\n", "Alice", 25, 'A');
```

---

## Next Steps

Functions let you structure programs into reusable components. Next, let's learn how to work with collections of data in [Arrays and Strings](./06_Arrays_and_Strings.md)!
