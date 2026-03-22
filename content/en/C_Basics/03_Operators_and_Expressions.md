# Operators and Expressions

**Previous**: [Variables and Data Types](./02_Variables_and_Data_Types.md) | **Next**: [Control Flow](./04_Control_Flow.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Apply arithmetic, relational, logical, and assignment operators in expressions
2. Evaluate expressions using C's operator precedence and associativity rules
3. Distinguish pre-increment from post-increment and explain side-effect ordering
4. Use bitwise operators for flag manipulation and masking
5. Apply the ternary operator and comma operator in concise expressions

---

Operators are the verbs of a programming language -- they tell the compiler what to do with your data. C has a rich set of operators that ranges from basic arithmetic to bit-level manipulation. Understanding how these operators work, and in what order they are evaluated, is essential for writing correct and efficient C code.

## 1. Arithmetic Operators

Arithmetic operators perform mathematical calculations on numeric operands.

| Operator | Name | Example | Result |
|----------|------|---------|--------|
| `+` | Addition | `7 + 3` | `10` |
| `-` | Subtraction | `7 - 3` | `4` |
| `*` | Multiplication | `7 * 3` | `21` |
| `/` | Division | `7 / 3` | `2` (integer) |
| `%` | Modulo (remainder) | `7 % 3` | `1` |
| `+` | Unary plus | `+5` | `5` |
| `-` | Unary minus | `-5` | `-5` |

### Integer Division vs Floating-Point Division

```c
#include <stdio.h>

int main(void) {
    /* Integer division truncates toward zero */
    printf("7 / 3   = %d\n", 7 / 3);     /* 2  */
    printf("-7 / 3  = %d\n", -7 / 3);    /* -2 (truncates toward zero in C99+) */

    /* Floating-point division preserves the fractional part */
    printf("7.0 / 3 = %f\n", 7.0 / 3);   /* 2.333333 */

    /* Force float division with a cast */
    int a = 7, b = 3;
    printf("(double)a / b = %f\n", (double)a / b);  /* 2.333333 */

    return 0;
}
```

### Modulo Operator

The `%` operator returns the remainder after integer division. It works only on integer types.

```c
#include <stdio.h>

int main(void) {
    printf("10 %% 3 = %d\n", 10 % 3);   /* 1 */
    printf("10 %% 5 = %d\n", 10 % 5);   /* 0 */
    printf("-7 %% 3 = %d\n", -7 % 3);   /* -1 (sign follows dividend in C99+) */

    /* Common uses */
    int n = 42;
    if (n % 2 == 0) {
        printf("%d is even\n", n);
    }

    /* Extract last digit */
    int last_digit = 12345 % 10;  /* 5 */
    printf("Last digit of 12345: %d\n", last_digit);

    return 0;
}
```

---

## 2. Relational and Equality Operators

Relational operators compare two values and produce an `int` result: `1` (true) or `0` (false). C has no built-in boolean type before C99; `_Bool` (or `bool` via `<stdbool.h>`) was added in C99.

| Operator | Meaning | Example |
|----------|---------|---------|
| `==` | Equal to | `a == b` |
| `!=` | Not equal to | `a != b` |
| `<` | Less than | `a < b` |
| `>` | Greater than | `a > b` |
| `<=` | Less than or equal | `a <= b` |
| `>=` | Greater than or equal | `a >= b` |

```c
#include <stdio.h>
#include <stdbool.h>   /* C99: bool, true, false */

int main(void) {
    int x = 10, y = 20;

    printf("x == y: %d\n", x == y);  /* 0 */
    printf("x != y: %d\n", x != y);  /* 1 */
    printf("x < y:  %d\n", x < y);   /* 1 */
    printf("x >= y: %d\n", x >= y);  /* 0 */

    /* Using bool (C99) */
    bool is_positive = (x > 0);
    printf("is_positive: %d\n", is_positive);  /* 1 */

    return 0;
}
```

> **Common Pitfall**: Using `=` (assignment) instead of `==` (comparison):
>
> ```c
> if (x = 5) {  /* BUG: assigns 5 to x, always true! */
>     printf("This always executes\n");
> }
> ```
>
> Some programmers write `5 == x` (Yoda conditions) to catch this mistake, since `5 = x` would be a compiler error. Compiling with `-Wall` also warns about this pattern.

---

## 3. Logical Operators

Logical operators combine boolean expressions. They treat any non-zero value as true and zero as false.

| Operator | Meaning | Example |
|----------|---------|---------|
| `&&` | Logical AND | `a && b` |
| `\|\|` | Logical OR | `a \|\| b` |
| `!` | Logical NOT | `!a` |

### Short-Circuit Evaluation

C evaluates logical expressions **left to right** and stops as soon as the result is determined:

- `&&` stops if the left operand is false (the whole expression is false).
- `||` stops if the left operand is true (the whole expression is true).

```c
#include <stdio.h>

int main(void) {
    int a = 5, b = 0;

    /* Short-circuit: b is 0 (false), so (b && ...) is false immediately */
    if (b != 0 && a / b > 2) {
        printf("This is safe\n");
    } else {
        printf("Division by zero avoided!\n");  /* This prints */
    }

    /* Without short-circuit, a/b would crash */

    /* Logical NOT */
    int logged_in = 0;
    if (!logged_in) {
        printf("Please log in\n");  /* This prints */
    }

    /* Truth table demonstration */
    printf("\nTruth Table for && and ||\n");
    printf("0 && 0 = %d\n", 0 && 0);  /* 0 */
    printf("0 && 1 = %d\n", 0 && 1);  /* 0 */
    printf("1 && 0 = %d\n", 1 && 0);  /* 0 */
    printf("1 && 1 = %d\n", 1 && 1);  /* 1 */
    printf("0 || 0 = %d\n", 0 || 0);  /* 0 */
    printf("0 || 1 = %d\n", 0 || 1);  /* 1 */
    printf("1 || 0 = %d\n", 1 || 0);  /* 1 */
    printf("1 || 1 = %d\n", 1 || 1);  /* 1 */

    return 0;
}
```

### Practical Example: Input Validation

```c
#include <stdio.h>

int main(void) {
    int age;
    printf("Enter age: ");
    scanf("%d", &age);

    if (age >= 0 && age <= 150) {
        printf("Valid age: %d\n", age);
    } else {
        printf("Invalid age\n");
    }

    /* Range check with logical OR */
    char grade;
    printf("Enter grade (A-F): ");
    scanf(" %c", &grade);

    if (grade < 'A' || grade > 'F') {
        printf("Invalid grade\n");
    }

    return 0;
}
```

---

## 4. Assignment Operators

Assignment operators store a value in a variable. The compound variants combine an operation with assignment.

| Operator | Equivalent | Example |
|----------|-----------|---------|
| `=` | -- | `x = 5` |
| `+=` | `x = x + n` | `x += 3` |
| `-=` | `x = x - n` | `x -= 3` |
| `*=` | `x = x * n` | `x *= 3` |
| `/=` | `x = x / n` | `x /= 3` |
| `%=` | `x = x % n` | `x %= 3` |
| `&=` | `x = x & n` | `x &= 0xFF` |
| `\|=` | `x = x \| n` | `x \|= 0x01` |
| `^=` | `x = x ^ n` | `x ^= mask` |
| `<<=` | `x = x << n` | `x <<= 2` |
| `>>=` | `x = x >> n` | `x >>= 2` |

```c
#include <stdio.h>

int main(void) {
    int x = 10;

    x += 5;   /* x = 15 */
    x -= 3;   /* x = 12 */
    x *= 2;   /* x = 24 */
    x /= 4;   /* x = 6  */
    x %= 5;   /* x = 1  */

    printf("x = %d\n", x);  /* 1 */

    /* Assignment is an expression — it returns the assigned value */
    int a, b, c;
    a = b = c = 0;  /* right-to-left: c=0, b=0, a=0 */
    printf("a=%d b=%d c=%d\n", a, b, c);

    return 0;
}
```

---

## 5. Increment and Decrement

The `++` and `--` operators add or subtract 1. They come in two forms with an important difference.

| Form | Name | Behavior |
|------|------|----------|
| `++x` | Pre-increment | Increment first, then use the new value |
| `x++` | Post-increment | Use the current value, then increment |
| `--x` | Pre-decrement | Decrement first, then use the new value |
| `x--` | Post-decrement | Use the current value, then decrement |

```c
#include <stdio.h>

int main(void) {
    int a = 5;
    int b;

    /* Pre-increment: increment, then assign */
    b = ++a;
    printf("++a: a=%d, b=%d\n", a, b);  /* a=6, b=6 */

    a = 5;  /* reset */

    /* Post-increment: assign, then increment */
    b = a++;
    printf("a++: a=%d, b=%d\n", a, b);  /* a=6, b=5 */

    return 0;
}
```

### Side Effects in Expressions

> **Warning**: Using `++` or `--` on the same variable multiple times in one expression is **undefined behavior**:
>
> ```c
> int i = 5;
> int result = i++ + ++i;  /* UNDEFINED BEHAVIOR — do not do this! */
> ```
>
> The compiler is free to evaluate `i++` and `++i` in any order. Different compilers (or optimization levels) may produce different results.

### When to Use Which

- **Standalone statement** (`i++;` or `++i;`): No difference; both increment `i` by 1.
- **Inside an expression**: Use pre-increment (`++i`) unless you specifically need the old value.
- **In `for` loops**: `for (int i = 0; i < n; i++)` — either form works, but `i++` is conventional in C.

---

## 6. Bitwise Operators

Bitwise operators work on the individual bits of integer values. They are essential for systems programming, embedded development, and performance-critical code.

| Operator | Name | Description |
|----------|------|-------------|
| `&` | Bitwise AND | Sets bit to 1 if both bits are 1 |
| `\|` | Bitwise OR | Sets bit to 1 if either bit is 1 |
| `^` | Bitwise XOR | Sets bit to 1 if bits differ |
| `~` | Bitwise NOT | Inverts all bits |
| `<<` | Left shift | Shifts bits left, fills with 0 |
| `>>` | Right shift | Shifts bits right (fill depends on sign) |

### AND, OR, XOR Truth Table

| A | B | A & B | A \| B | A ^ B |
|---|---|-------|--------|-------|
| 0 | 0 | 0 | 0 | 0 |
| 0 | 1 | 0 | 1 | 1 |
| 1 | 0 | 0 | 1 | 1 |
| 1 | 1 | 1 | 1 | 0 |

### Practical Examples

```c
#include <stdio.h>

int main(void) {
    unsigned char a = 0b11001010;  /* 202 in decimal */
    unsigned char b = 0b10110101;  /* 181 in decimal */

    printf("a & b  = 0x%02X\n", a & b);   /* 0x80 = 10000000 */
    printf("a | b  = 0x%02X\n", a | b);   /* 0xFF = 11111111 */
    printf("a ^ b  = 0x%02X\n", a ^ b);   /* 0x7F = 01111111 */
    printf("~a     = 0x%02X\n", (unsigned char)~a);  /* 0x35 = 00110101 */

    /* Shift operators */
    unsigned int x = 1;
    printf("1 << 3 = %u\n", x << 3);   /* 8  (multiply by 2^3) */
    printf("8 >> 2 = %u\n", 8U >> 2);  /* 2  (divide by 2^2)  */

    return 0;
}
```

### Flag Manipulation

Bitwise operators are commonly used to manage flags -- individual bits that represent on/off states.

```c
#include <stdio.h>

/* Define flags as powers of 2 */
#define FLAG_READ    (1 << 0)   /* 0001 = 1 */
#define FLAG_WRITE   (1 << 1)   /* 0010 = 2 */
#define FLAG_EXECUTE (1 << 2)   /* 0100 = 4 */
#define FLAG_DELETE  (1 << 3)   /* 1000 = 8 */

int main(void) {
    unsigned int permissions = 0;

    /* Set flags */
    permissions |= FLAG_READ;          /* Turn on read */
    permissions |= FLAG_WRITE;         /* Turn on write */
    printf("After set: %u\n", permissions);  /* 3 (0011) */

    /* Check a flag */
    if (permissions & FLAG_READ) {
        printf("Read permission is ON\n");
    }
    if (!(permissions & FLAG_EXECUTE)) {
        printf("Execute permission is OFF\n");
    }

    /* Clear a flag */
    permissions &= ~FLAG_WRITE;        /* Turn off write */
    printf("After clear write: %u\n", permissions);  /* 1 (0001) */

    /* Toggle a flag */
    permissions ^= FLAG_EXECUTE;       /* Toggle execute */
    printf("After toggle execute: %u\n", permissions);  /* 5 (0101) */

    return 0;
}
```

### Bit Masking

```c
#include <stdio.h>

int main(void) {
    /* Extract a specific byte from a 32-bit value */
    unsigned int color = 0xFF8040A0;   /* RGBA: R=FF, G=80, B=40, A=A0 */

    unsigned char r = (color >> 24) & 0xFF;
    unsigned char g = (color >> 16) & 0xFF;
    unsigned char b = (color >>  8) & 0xFF;
    unsigned char a = (color >>  0) & 0xFF;

    printf("R=%u G=%u B=%u A=%u\n", r, g, b, a);
    /* R=255 G=128 B=64 A=160 */

    /* Pack bytes into a 32-bit value */
    unsigned int packed = ((unsigned int)r << 24) |
                          ((unsigned int)g << 16) |
                          ((unsigned int)b <<  8) |
                          ((unsigned int)a);
    printf("Packed: 0x%08X\n", packed);  /* 0xFF8040A0 */

    return 0;
}
```

---

## 7. Ternary and Comma Operators

### Ternary Operator

The ternary operator `condition ? expr_if_true : expr_if_false` is a concise alternative to `if-else` for simple expressions.

```c
#include <stdio.h>

int main(void) {
    int x = 10, y = 20;

    /* Instead of if-else */
    int max = (x > y) ? x : y;
    printf("max = %d\n", max);  /* 20 */

    /* Inline in printf */
    int score = 75;
    printf("Result: %s\n", (score >= 60) ? "Pass" : "Fail");  /* Pass */

    /* Nested ternary (use sparingly — hurts readability) */
    int val = 0;
    const char *sign = (val > 0) ? "positive"
                     : (val < 0) ? "negative"
                     : "zero";
    printf("val is %s\n", sign);  /* zero */

    /* Absolute value */
    int n = -42;
    int abs_n = (n >= 0) ? n : -n;
    printf("|%d| = %d\n", n, abs_n);

    return 0;
}
```

### Comma Operator

The comma operator evaluates two expressions left to right and returns the value of the rightmost expression. It is most commonly seen in `for` loop headers.

```c
#include <stdio.h>

int main(void) {
    /* Comma in for loop — multiple variables */
    for (int i = 0, j = 10; i < j; i++, j--) {
        printf("i=%d j=%d\n", i, j);
    }

    /* Comma as an operator (rarely used outside for loops) */
    int x = (1, 2, 3);  /* x = 3 — the value of the last expression */
    printf("x = %d\n", x);

    return 0;
}
```

---

## 8. Operator Precedence Table

When multiple operators appear in an expression, precedence and associativity determine the evaluation order. Higher precedence operators bind more tightly.

| Precedence | Operator(s) | Description | Associativity |
|-----------|-------------|-------------|---------------|
| 1 (highest) | `()` `[]` `->` `.` | Function call, subscript, member access | Left-to-right |
| 2 | `++` `--` (postfix) | Postfix increment/decrement | Left-to-right |
| 3 | `++` `--` (prefix) `+` `-` `!` `~` `*` `&` `sizeof` `(type)` | Unary operators, cast, sizeof | Right-to-left |
| 4 | `*` `/` `%` | Multiplicative | Left-to-right |
| 5 | `+` `-` | Additive | Left-to-right |
| 6 | `<<` `>>` | Bitwise shift | Left-to-right |
| 7 | `<` `<=` `>` `>=` | Relational | Left-to-right |
| 8 | `==` `!=` | Equality | Left-to-right |
| 9 | `&` | Bitwise AND | Left-to-right |
| 10 | `^` | Bitwise XOR | Left-to-right |
| 11 | `\|` | Bitwise OR | Left-to-right |
| 12 | `&&` | Logical AND | Left-to-right |
| 13 | `\|\|` | Logical OR | Left-to-right |
| 14 | `?:` | Ternary conditional | Right-to-left |
| 15 | `=` `+=` `-=` `*=` `/=` `%=` `<<=` `>>=` `&=` `^=` `\|=` | Assignment | Right-to-left |
| 16 (lowest) | `,` | Comma | Left-to-right |

### Precedence Examples

```c
#include <stdio.h>

int main(void) {
    /* Multiplication before addition */
    int a = 2 + 3 * 4;     /* 2 + (3*4) = 14, not (2+3)*4 = 20 */
    printf("2 + 3 * 4 = %d\n", a);

    /* Relational before logical */
    int x = 5, y = 10;
    int result = x > 3 && y < 20;  /* (x>3) && (y<20) = 1 */
    printf("x > 3 && y < 20 = %d\n", result);

    /* Bitwise AND has lower precedence than equality — a common trap! */
    int flags = 5;        /* 0101 */
    int mask  = 4;        /* 0100 */

    /* WRONG: == binds tighter than & */
    if (flags & mask == 4) {
        printf("Bug: this tests flags & (mask == 4)\n");
    }

    /* CORRECT: use parentheses */
    if ((flags & mask) == 4) {
        printf("Correct: bit 2 is set\n");
    }

    /* When in doubt, use parentheses! */
    int b = (2 + 3) * 4;  /* 20 — explicit grouping */
    printf("(2 + 3) * 4 = %d\n", b);

    return 0;
}
```

> **Best Practice**: When the evaluation order is not immediately obvious, add parentheses. They cost nothing at runtime and make your intent clear to every reader.

---

## Exercises

### Exercise 1: Expression Evaluator

Without running the code, evaluate each expression by hand. Then write a program to verify:

```c
int a = 10, b = 3, c = 7;
printf("%d\n", a + b * c);           /* ? */
printf("%d\n", (a + b) * c);         /* ? */
printf("%d\n", a % b + c / b);       /* ? */
printf("%d\n", a > b && c > a);      /* ? */
printf("%d\n", !0 + !1);            /* ? */
printf("%d\n", a & b | c);          /* ? */
```

### Exercise 2: Swap Without Temporary

Write a program that swaps two integer variables using:

1. XOR (`^`) — three statements, no temporary variable.
2. Addition and subtraction — three statements, no temporary variable.

Print the values before and after each swap. Explain when the XOR method might fail (hint: what if both variables are the same object?).

### Exercise 3: Permission Checker

Define four permission flags (`READ`, `WRITE`, `EXECUTE`, `ADMIN`) using bit shifts. Write a function `void print_permissions(unsigned int perms)` that prints which permissions are active. Then write a main function that:

1. Grants READ and WRITE.
2. Checks and prints all active permissions.
3. Revokes WRITE.
4. Grants ADMIN.
5. Checks and prints again.

### Exercise 4: Bit Counter

Write a function `int count_set_bits(unsigned int n)` that returns the number of 1-bits in `n`. Implement it two ways:

1. Using a loop that checks the least significant bit and right-shifts.
2. Using Brian Kernighan's trick: `n = n & (n - 1)` clears the lowest set bit.

Test both implementations with the values 0, 1, 255, and 0xDEADBEEF.

### Exercise 5: RGBA Color Mixer

Write a program that:

1. Defines two RGBA colors as `unsigned int` values (e.g., `0xFF0000FF` for red, `0x00FF00FF` for green).
2. Extracts the R, G, B, and A components of each color using bitwise operators.
3. Computes a 50% blend of the two colors by averaging each channel.
4. Packs the blended channels back into a single `unsigned int`.
5. Prints all three colors in `0xRRGGBBAA` format.

---

## Next Steps

You now know how to compute with data. Next, let's learn how to make decisions and repeat actions with [Control Flow](./04_Control_Flow.md)!
