/*
 * Exercises for Lesson 02: Variables and Data Types
 * Topic: C_Basics
 * Solutions to practice problems from the lesson.
 * Compile: gcc -Wall -Wextra -std=c11 -o ex02 02_variables_and_data_types.c
 */
#include <stdio.h>
#include <limits.h>
#include <float.h>
#include <stdint.h>

/* === Exercise 1: Print Sizes of All Types === */
/* Problem: Display the size (in bytes) of every fundamental C type. */
void exercise_1(void) {
    printf("=== Exercise 1: Print Sizes of All Types ===\n");

    printf("char        : %zu bytes (range: %d to %d)\n",
           sizeof(char), CHAR_MIN, CHAR_MAX);
    printf("short       : %zu bytes (range: %d to %d)\n",
           sizeof(short), SHRT_MIN, SHRT_MAX);
    printf("int         : %zu bytes (range: %d to %d)\n",
           sizeof(int), INT_MIN, INT_MAX);
    printf("long        : %zu bytes (range: %ld to %ld)\n",
           sizeof(long), LONG_MIN, LONG_MAX);
    printf("long long   : %zu bytes (range: %lld to %lld)\n",
           sizeof(long long), LLONG_MIN, LLONG_MAX);
    printf("unsigned int: %zu bytes (range: 0 to %u)\n",
           sizeof(unsigned int), UINT_MAX);
    printf("float       : %zu bytes (precision: %d digits)\n",
           sizeof(float), FLT_DIG);
    printf("double      : %zu bytes (precision: %d digits)\n",
           sizeof(double), DBL_DIG);
    printf("long double : %zu bytes (precision: %d digits)\n",
           sizeof(long double), LDBL_DIG);
    printf("void *      : %zu bytes\n", sizeof(void *));
    printf("int8_t      : %zu bytes\n", sizeof(int8_t));
    printf("int16_t     : %zu bytes\n", sizeof(int16_t));
    printf("int32_t     : %zu bytes\n", sizeof(int32_t));
    printf("int64_t     : %zu bytes\n", sizeof(int64_t));
}

/* === Exercise 2: Type Conversion Demo === */
/* Problem: Demonstrate implicit and explicit type conversions,
 *          showing when precision is lost. */
void exercise_2(void) {
    printf("\n=== Exercise 2: Type Conversion Demo ===\n");

    /* Integer promotion */
    char a = 100, b = 50;
    int sum = a + b;  /* char promoted to int before addition */
    printf("char 100 + char 50 = int %d\n", sum);

    /* Implicit narrowing: int to char */
    int big = 300;
    char narrow = (char)big;
    printf("int %d -> char %d (truncated)\n", big, narrow);

    /* Float to int truncation */
    double pi = 3.14159;
    int truncated = (int)pi;
    printf("double %.5f -> int %d (truncated, not rounded)\n", pi, truncated);

    /* Integer division vs float division */
    int x = 7, y = 2;
    printf("int 7 / int 2 = %d (integer division)\n", x / y);
    printf("7.0 / 2.0 = %.1f (float division)\n", 7.0 / 2.0);

    /* Signed/unsigned interaction */
    unsigned int u = 10;
    int s = -1;
    /* -1 is converted to a large unsigned value */
    if (s < (int)u) {
        printf("Correct comparison using cast: -1 < 10\n");
    }
    printf("unsigned interpretation of -1: %u\n", (unsigned int)s);
}

/* === Exercise 3: Overflow Detection === */
/* Problem: Detect and demonstrate integer overflow for signed and unsigned. */
void exercise_3(void) {
    printf("\n=== Exercise 3: Overflow Detection ===\n");

    /* Unsigned overflow wraps around (well-defined) */
    unsigned char uc = 255;
    printf("unsigned char 255 + 1 = %u (wraps to 0)\n",
           (unsigned char)(uc + 1));

    /* Safe addition check for signed int */
    int a = INT_MAX - 10;
    int b = 20;
    if (b > 0 && a > INT_MAX - b) {
        printf("Overflow detected: %d + %d would exceed INT_MAX (%d)\n",
               a, b, INT_MAX);
    } else {
        printf("%d + %d = %d\n", a, b, a + b);
    }

    /* Safe multiplication check */
    int m = 100000, n = 100000;
    if (m != 0 && n > INT_MAX / m) {
        printf("Overflow detected: %d * %d would exceed INT_MAX\n", m, n);
    } else {
        printf("%d * %d = %d\n", m, n, m * n);
    }

    /* Using fixed-width types for predictable behavior */
    int32_t val = INT32_MAX;
    printf("INT32_MAX = %d\n", val);
    printf("INT32_MAX + 1 overflows (undefined behavior for signed)\n");
}

int main(void) {
    exercise_1();
    exercise_2();
    exercise_3();

    printf("\nAll exercises completed!\n");
    return 0;
}
