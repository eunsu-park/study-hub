# Variables and Data Types

**Previous**: [Environment Setup](./01_Environment_Setup.md) | **Next**: [Operators and Expressions](./03_Operators_and_Expressions.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Declare and initialize variables of every fundamental C type
2. Explain the size and range of `char`, `short`, `int`, `long`, `long long`, `float`, and `double`
3. Apply unsigned modifiers and understand two's complement representation
4. Use `sizeof` to inspect type sizes and `const`/`volatile` qualifiers
5. Perform implicit and explicit type conversions and identify truncation risks
6. Select the correct `printf` format specifier for each data type

---

Every C program manipulates data stored in memory. Unlike dynamically-typed languages where variables can change type at any time, C requires you to declare the type of every variable before you use it. This discipline gives the compiler the information it needs to allocate the right amount of memory and generate efficient machine code. Mastering C's type system is the foundation for understanding pointers, structs, and everything that follows.

## 1. Variables and Declaration

A variable in C is a named region of memory with a specific type. You must declare a variable before using it.

### Declaration Syntax

```c
type name;            /* declaration only (uninitialized) */
type name = value;    /* declaration with initialization  */
```

```c
#include <stdio.h>

int main(void) {
    int age;              /* declared but uninitialized — contains garbage */
    int score = 95;       /* declared and initialized */
    double pi = 3.14159;  /* floating-point variable */
    char grade = 'A';     /* single character */

    age = 25;             /* assignment after declaration */

    printf("age = %d, score = %d\n", age, score);
    printf("pi = %f, grade = %c\n", pi, grade);
    return 0;
}
```

### Naming Rules

| Rule | Example | Valid? |
|------|---------|--------|
| Must start with letter or underscore | `count`, `_temp` | Yes |
| Can contain letters, digits, underscores | `score2`, `max_val` | Yes |
| Cannot start with a digit | `2nd_place` | No |
| Cannot be a C keyword | `int`, `return` | No |
| Case-sensitive | `Count` and `count` are different | -- |

### Naming Conventions

```c
int student_count;    /* snake_case — common in C */
int studentCount;     /* camelCase  — less common in C */
#define MAX_SIZE 100  /* UPPER_CASE — for constants and macros */
```

### Multiple Declarations

```c
int a, b, c;           /* three ints, all uninitialized */
int x = 1, y = 2;     /* two ints, both initialized */
double width = 10.5, height = 20.0;
```

> **Warning**: Uninitialized local variables contain unpredictable garbage values. Always initialize variables before reading them.

---

## 2. Integer Types

C provides several integer types that differ in size and range. The exact sizes are platform-dependent, but the C standard guarantees minimum ranges.

### Integer Type Summary

| Type | Minimum Size | Typical Size (64-bit) | Typical Range |
|------|-------------|----------------------|---------------|
| `char` | 1 byte | 1 byte | -128 to 127 |
| `short` | 2 bytes | 2 bytes | -32,768 to 32,767 |
| `int` | 2 bytes | 4 bytes | -2,147,483,648 to 2,147,483,647 |
| `long` | 4 bytes | 8 bytes (Linux/macOS), 4 bytes (Windows) | Platform-dependent |
| `long long` | 8 bytes | 8 bytes | -9.2 x 10^18 to 9.2 x 10^18 |

### The Guaranteed Ordering

The C standard guarantees:

```
sizeof(char) <= sizeof(short) <= sizeof(int) <= sizeof(long) <= sizeof(long long)
```

### Working with Integers

```c
#include <stdio.h>
#include <limits.h>   /* INT_MIN, INT_MAX, etc. */

int main(void) {
    char   c = 'Z';           /* 1 byte — also stores small integers */
    short  s = 1000;          /* at least 2 bytes */
    int    i = 42;            /* at least 2 bytes, usually 4 */
    long   l = 100000L;       /* at least 4 bytes — note the L suffix */
    long long ll = 9000000000000LL;  /* at least 8 bytes — LL suffix */

    printf("char:      %c  (value: %d, size: %zu bytes)\n", c, c, sizeof(c));
    printf("short:     %hd (size: %zu bytes)\n", s, sizeof(s));
    printf("int:       %d  (size: %zu bytes)\n", i, sizeof(i));
    printf("long:      %ld (size: %zu bytes)\n", l, sizeof(l));
    printf("long long: %lld (size: %zu bytes)\n", ll, sizeof(ll));

    printf("\nint range: %d to %d\n", INT_MIN, INT_MAX);
    return 0;
}
```

### Integer Overflow

When an integer exceeds its range, the behavior depends on whether it is signed or unsigned:

```c
#include <stdio.h>
#include <limits.h>

int main(void) {
    int max = INT_MAX;
    printf("INT_MAX     = %d\n", max);
    printf("INT_MAX + 1 = %d\n", max + 1);  /* undefined behavior for signed! */

    return 0;
}
```

> **Important**: Signed integer overflow is **undefined behavior** in C. The compiler is free to do anything, including optimizing away your overflow check. Never rely on signed overflow wrapping around.

---

## 2a. Fixed-Width Integer Types

The sizes of `int`, `long`, and friends are platform-dependent, which creates portability problems when exact sizes matter. The header `<stdint.h>` (C99+) provides types with guaranteed widths.

### Types and Limits

```c
#include <stdint.h>
#include <inttypes.h>  /* PRId32, PRIu64, etc. — format specifier macros */
#include <stdio.h>

int main(void) {
    int8_t   a = -128;          /* exactly 8 bits, signed  */
    uint8_t  b = 255;           /* exactly 8 bits, unsigned */
    int16_t  c = 32767;
    int32_t  d = INT32_MAX;     /* 2,147,483,647            */
    uint64_t e = UINT64_MAX;    /* 18,446,744,073,709,551,615 */

    printf("int32_t max : %" PRId32 "\n", d);
    printf("uint64_t max: %" PRIu64 "\n", e);

    return 0;
}
```

| Signed | Unsigned | Width | Range (signed) |
|--------|----------|-------|----------------|
| `int8_t` | `uint8_t` | 8 bits | -128 to 127 |
| `int16_t` | `uint16_t` | 16 bits | -32,768 to 32,767 |
| `int32_t` | `uint32_t` | 32 bits | ±2.1 × 10⁹ |
| `int64_t` | `uint64_t` | 64 bits | ±9.2 × 10¹⁸ |

### When to Use Fixed-Width Types

- **Binary file formats**: reading/writing a 4-byte field must use `int32_t`, not `int`.
- **Network protocols**: protocol headers specify exact byte widths; `uint16_t` matches a 2-byte port number.
- **Hardware registers**: a 32-bit memory-mapped register should be accessed as `uint32_t`.
- **Cross-platform code**: anywhere that `int` being 2 bytes vs 4 bytes would change behavior.

> **Note**: For general arithmetic where exact size does not matter, plain `int` is still preferred — the compiler can choose the most efficient native size. Reserve fixed-width types for situations where the exact representation is load-bearing.

---

## 3. Unsigned Integers

The `unsigned` keyword restricts an integer to non-negative values, effectively doubling the positive range.

### Unsigned Type Ranges

| Type | Typical Size | Range |
|------|-------------|-------|
| `unsigned char` | 1 byte | 0 to 255 |
| `unsigned short` | 2 bytes | 0 to 65,535 |
| `unsigned int` | 4 bytes | 0 to 4,294,967,295 |
| `unsigned long` | 4 or 8 bytes | 0 to 2^32-1 or 2^64-1 |
| `unsigned long long` | 8 bytes | 0 to 18,446,744,073,709,551,615 |

### Two's Complement

Modern systems use **two's complement** to represent signed integers:

- The most significant bit (MSB) is the sign bit: 0 = positive, 1 = negative.
- To negate a number: flip all bits, then add 1.
- For an 8-bit `char`: `01111111` = 127, `10000000` = -128.

```c
#include <stdio.h>

int main(void) {
    unsigned int u = 0;
    printf("u     = %u\n", u);
    printf("u - 1 = %u\n", u - 1);  /* wraps to 4294967295 (well-defined!) */

    /* Unsigned overflow is well-defined: it wraps modulo 2^N */
    unsigned char byte = 255;
    byte = byte + 1;
    printf("255 + 1 as unsigned char = %u\n", byte);  /* 0 */

    return 0;
}
```

### When to Use Unsigned

- Bit manipulation and flags
- Array indices (though `size_t` is preferred)
- When values are inherently non-negative (e.g., byte counts)
- Interfacing with APIs that use unsigned types

> **Pitfall**: Mixing signed and unsigned in comparisons can produce surprising results:
>
> ```c
> int a = -1;
> unsigned int b = 1;
> if (a < b) {
>     printf("Expected\n");
> } else {
>     printf("Surprise!\n");  /* This prints! -1 is converted to a large unsigned value */
> }
> ```

---

## 4. Floating-Point Types

C provides three floating-point types for representing real numbers.

| Type | Typical Size | Precision | Range (approximate) |
|------|-------------|-----------|---------------------|
| `float` | 4 bytes | ~7 decimal digits | ±3.4 x 10^38 |
| `double` | 8 bytes | ~15 decimal digits | ±1.7 x 10^308 |
| `long double` | 8-16 bytes | ~18-21 digits | Platform-dependent |

### IEEE 754 Basics

Floating-point numbers are stored in three parts: **sign**, **exponent**, and **mantissa** (significand).

- `float`: 1 sign bit + 8 exponent bits + 23 mantissa bits = 32 bits
- `double`: 1 sign bit + 11 exponent bits + 52 mantissa bits = 64 bits

### Working with Floats

```c
#include <stdio.h>
#include <float.h>   /* FLT_MIN, FLT_MAX, DBL_EPSILON, etc. */

int main(void) {
    float  f = 3.14f;        /* f suffix for float literals */
    double d = 3.141592653589793;  /* default literal type is double */
    long double ld = 3.14159265358979323846L;  /* L suffix */

    printf("float:       %.7f  (size: %zu bytes)\n", f, sizeof(f));
    printf("double:      %.15f (size: %zu bytes)\n", d, sizeof(d));
    printf("long double: %.18Lf (size: %zu bytes)\n", ld, sizeof(ld));

    /* Precision limits */
    printf("\nfloat precision:  %d digits\n", FLT_DIG);
    printf("double precision: %d digits\n", DBL_DIG);
    return 0;
}
```

### Floating-Point Pitfalls

```c
#include <stdio.h>
#include <math.h>

int main(void) {
    /* Equality comparison is unreliable */
    double a = 0.1 + 0.2;
    double b = 0.3;
    printf("0.1 + 0.2 == 0.3? %d\n", a == b);  /* 0 (false!) */

    /* Use an epsilon for comparison */
    double epsilon = 1e-9;
    if (fabs(a - b) < epsilon) {
        printf("Approximately equal\n");  /* This prints */
    }

    /* Integer division trap */
    double ratio = 1 / 3;       /* 0.000000 — integer division! */
    double correct = 1.0 / 3.0; /* 0.333333 — floating-point division */
    printf("1/3   = %f\n", ratio);
    printf("1.0/3 = %f\n", correct);

    return 0;
}
```

---

## 5. Type Qualifiers

Type qualifiers modify how a variable can be accessed or optimized.

### const

The `const` qualifier makes a variable read-only after initialization.

```c
#include <stdio.h>

int main(void) {
    const int MAX_STUDENTS = 100;
    const double PI = 3.14159265358979;

    printf("Max students: %d\n", MAX_STUDENTS);
    /* MAX_STUDENTS = 200;  — compiler error: assignment to const variable */

    /* const with pointers (covered in detail in the Pointers lesson) */
    const char *greeting = "Hello";  /* pointer to const char */
    /* greeting[0] = 'h';  — error: cannot modify const data */

    return 0;
}
```

### volatile

The `volatile` qualifier tells the compiler that a variable may change at any time (e.g., hardware registers, signal handlers), so it must not optimize away reads.

```c
volatile int sensor_value;  /* may be changed by hardware */

/* The compiler will re-read sensor_value every time, never caching it */
while (sensor_value == 0) {
    /* wait for sensor to trigger */
}
```

### static Local Variables

A `static` local variable retains its value between function calls.

```c
#include <stdio.h>

void counter(void) {
    static int count = 0;  /* initialized only once */
    count++;
    printf("Called %d times\n", count);
}

int main(void) {
    counter();  /* Called 1 times */
    counter();  /* Called 2 times */
    counter();  /* Called 3 times */
    return 0;
}
```

---

## 6. Type Conversion

C performs type conversions in two ways: **implicitly** (automatic) and **explicitly** (casts).

### Implicit Conversion (Promotion)

When operands of different types appear in an expression, the compiler promotes the "smaller" type to the "larger" type.

```
char/short → int → unsigned int → long → unsigned long → long long → float → double → long double
```

```c
#include <stdio.h>

int main(void) {
    int    i = 42;
    double d = 3.14;

    /* i is promoted to double before addition */
    double result = i + d;
    printf("%f\n", result);  /* 45.140000 */

    /* char is promoted to int in arithmetic */
    char c = 'A';         /* 65 */
    int  n = c + 1;       /* 66 */
    printf("%c\n", (char)n);  /* 'B' */

    return 0;
}
```

### Explicit Conversion (Casting)

Use a cast when you intentionally want to convert between types.

```c
#include <stdio.h>

int main(void) {
    int a = 7, b = 2;

    /* Without cast: integer division */
    double bad  = a / b;          /* 3.000000 */

    /* With cast: floating-point division */
    double good = (double)a / b;  /* 3.500000 */

    printf("bad  = %f\n", bad);
    printf("good = %f\n", good);

    /* Truncation risk: double to int */
    double pi = 3.99;
    int truncated = (int)pi;  /* 3 — fractional part is discarded */
    printf("truncated = %d\n", truncated);

    return 0;
}
```

### Common Truncation Risks

| Conversion | Risk |
|-----------|------|
| `double` to `float` | Precision loss |
| `double` to `int` | Fractional part discarded |
| `long long` to `int` | Upper bits lost if value exceeds `INT_MAX` |
| `int` to `char` | Only lowest 8 bits preserved |
| `unsigned` to `signed` | Reinterpretation if value > `TYPE_MAX` |

```c
#include <stdio.h>

int main(void) {
    long long big = 5000000000LL;
    int small = (int)big;
    printf("big = %lld, small = %d\n", big, small);
    /* small is garbage — 5 billion exceeds INT_MAX */

    unsigned int u = 3000000000U;
    int s = (int)u;
    printf("unsigned %u -> signed %d\n", u, s);
    /* negative number — reinterpretation of bit pattern */

    return 0;
}
```

---

## 7. Format Specifiers

The `printf` and `scanf` families use format specifiers to match the type of each argument.

### Comprehensive Format Specifier Table

| Specifier | Type | Example |
|-----------|------|---------|
| `%d` or `%i` | `int` (signed decimal) | `printf("%d", 42)` |
| `%u` | `unsigned int` | `printf("%u", 42U)` |
| `%ld` | `long` | `printf("%ld", 100000L)` |
| `%lld` | `long long` | `printf("%lld", 9000000000LL)` |
| `%lu` | `unsigned long` | `printf("%lu", 100000UL)` |
| `%llu` | `unsigned long long` | `printf("%llu", val)` |
| `%hd` | `short` | `printf("%hd", (short)10)` |
| `%f` | `double` (in printf) / `float` (in scanf) | `printf("%f", 3.14)` |
| `%lf` | `double` (in scanf only) | `scanf("%lf", &d)` |
| `%e` / `%E` | Scientific notation | `printf("%e", 0.001)` → `1.000000e-03` |
| `%g` | Shorter of `%f` and `%e` | `printf("%g", 3.14)` |
| `%c` | `char` | `printf("%c", 'A')` |
| `%s` | `char *` (string) | `printf("%s", "hello")` |
| `%p` | Pointer (address) | `printf("%p", (void *)&x)` |
| `%x` / `%X` | Unsigned hexadecimal | `printf("%x", 255)` → `ff` |
| `%o` | Unsigned octal | `printf("%o", 8)` → `10` |
| `%zu` | `size_t` | `printf("%zu", sizeof(int))` |
| `%%` | Literal `%` | `printf("100%%")` |

### Width and Precision

```c
#include <stdio.h>

int main(void) {
    int n = 42;
    double pi = 3.14159265;

    printf("[%10d]\n", n);      /* [        42] — right-aligned, width 10 */
    printf("[%-10d]\n", n);     /* [42        ] — left-aligned */
    printf("[%05d]\n", n);      /* [00042]      — zero-padded */

    printf("[%.2f]\n", pi);     /* [3.14]       — 2 decimal places */
    printf("[%10.4f]\n", pi);   /* [    3.1416] — width 10, 4 decimals */

    printf("[%.5s]\n", "Hello, World");  /* [Hello] — max 5 chars from string */

    return 0;
}
```

> **Warning**: Mismatched format specifiers cause **undefined behavior**. Using `%d` to print a `long long` or `%f` to print an `int` can produce garbage output or crashes.

---

## 8. sizeof Operator

The `sizeof` operator returns the size in bytes of a type or variable. It is evaluated at **compile time** (except for variable-length arrays).

```c
#include <stdio.h>

int main(void) {
    /* sizeof with types */
    printf("char:        %zu bytes\n", sizeof(char));        /* always 1 */
    printf("short:       %zu bytes\n", sizeof(short));
    printf("int:         %zu bytes\n", sizeof(int));
    printf("long:        %zu bytes\n", sizeof(long));
    printf("long long:   %zu bytes\n", sizeof(long long));
    printf("float:       %zu bytes\n", sizeof(float));
    printf("double:      %zu bytes\n", sizeof(double));
    printf("long double: %zu bytes\n", sizeof(long double));
    printf("void *:      %zu bytes\n", sizeof(void *));

    printf("\n");

    /* sizeof with variables */
    int arr[10];
    printf("arr:         %zu bytes\n", sizeof(arr));          /* 40 (10 * 4) */
    printf("arr elements: %zu\n", sizeof(arr) / sizeof(arr[0])); /* 10 */

    /* sizeof with expressions — the expression is NOT evaluated */
    int x = 5;
    printf("sizeof(x++): %zu\n", sizeof(x++));  /* x is still 5 */
    printf("x = %d\n", x);                       /* 5, not 6! */

    return 0;
}
```

### Using sizeof for Portable Code

```c
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    int n = 10;

    /* Allocating memory — sizeof ensures correct size on any platform */
    int *arr = malloc(n * sizeof(*arr));  /* preferred: sizeof(*arr) */
    if (arr == NULL) {
        fprintf(stderr, "malloc failed\n");
        return 1;
    }

    for (int i = 0; i < n; i++) {
        arr[i] = i * i;
    }

    /* Compute array length (only works for true arrays, not pointers) */
    int fixed[5] = {10, 20, 30, 40, 50};
    size_t len = sizeof(fixed) / sizeof(fixed[0]);
    printf("fixed has %zu elements\n", len);  /* 5 */

    free(arr);
    return 0;
}
```

---

## Exercises

### Exercise 1: Type Size Explorer

Write a program that prints the size (in bytes) and the minimum/maximum values of every fundamental integer and floating-point type. Use `<limits.h>` for integer limits and `<float.h>` for floating-point limits. Format the output as a neat table:

```
Type              Size    Min                  Max
char              1       -128                 127
unsigned char     1       0                    255
short             2       ...                  ...
...
```

### Exercise 2: Overflow Detective

Write a program that demonstrates:

1. Signed integer overflow with `INT_MAX + 1` (compile with `-Wall` and note the warning).
2. Unsigned integer wraparound with `0U - 1`.
3. Float precision loss by storing `16777217` (2^24 + 1) in a `float` and printing it.

For each case, print the before and after values and write a comment explaining what happened.

### Exercise 3: Temperature Converter

Write a program that reads a temperature in Fahrenheit (as a `double`) using `scanf` and prints the equivalent in Celsius. The formula is `C = (F - 32) * 5.0 / 9.0`. Print the result with exactly 2 decimal places. Test with: 32.0 (expected: 0.00), 212.0 (expected: 100.00), -40.0 (expected: -40.00).

### Exercise 4: Type Conversion Traps

Predict the output of each line, then run the program to verify:

```c
printf("%d\n", (int)3.9);
printf("%d\n", (int)-3.9);
printf("%u\n", (unsigned int)-1);
printf("%d\n", (char)300);
printf("%f\n", 7 / 2);
printf("%f\n", 7.0 / 2);
```

Write a comment next to each line explaining why it produces that output.

### Exercise 5: Format Specifier Drill

Write a single program that declares one variable of each fundamental type (`char`, `short`, `int`, `long`, `long long`, `unsigned int`, `float`, `double`) and prints each using:

1. The correct format specifier.
2. An intentionally wrong specifier (e.g., `%d` for a `double`).

Compile with `-Wall -Wextra` and note which mismatches the compiler catches. Document your findings in comments.

---

## Next Steps

Now that you understand how C stores data in memory, let's explore how to combine and transform that data with [Operators and Expressions](./03_Operators_and_Expressions.md)!
