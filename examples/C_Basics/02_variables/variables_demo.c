/*
 * variables_demo.c — Demonstrates C data types, sizeof, and format specifiers.
 *
 * Compile: gcc -Wall -Wextra -std=c11 -o variables_demo variables_demo.c
 * Run:     ./variables_demo
 */

#include <stdio.h>
#include <stdbool.h>
#include <limits.h>
#include <float.h>

int main(void)
{
    /* Integer types */
    char          c  = 'A';
    short         s  = -1024;
    int           i  = 42;
    long          l  = 1000000L;
    long long     ll = 9223372036854775807LL;
    unsigned int  u  = 4294967295U;

    /* Floating-point types */
    float       f = 3.14f;
    double      d = 2.718281828;
    long double ld = 1.6180339887498948482L;

    /* Boolean (C99+) */
    bool flag = true;

    printf("=== Integer Types ===\n");
    printf("char          : '%c'  (value %d, size %zu bytes)\n", c, c, sizeof(c));
    printf("short         : %hd   (size %zu bytes, range %d..%d)\n",
           s, sizeof(s), SHRT_MIN, SHRT_MAX);
    printf("int           : %d    (size %zu bytes)\n", i, sizeof(i));
    printf("long          : %ld   (size %zu bytes)\n", l, sizeof(l));
    printf("long long     : %lld  (size %zu bytes)\n", ll, sizeof(ll));
    printf("unsigned int  : %u    (size %zu bytes)\n", u, sizeof(u));

    printf("\n=== Floating-Point Types ===\n");
    printf("float         : %.2f         (size %zu bytes, precision %d digits)\n",
           f, sizeof(f), FLT_DIG);
    printf("double        : %.9f  (size %zu bytes, precision %d digits)\n",
           d, sizeof(d), DBL_DIG);
    printf("long double   : %.18Lf (size %zu bytes)\n", ld, sizeof(ld));

    printf("\n=== Other Types ===\n");
    printf("bool          : %d (size %zu bytes)\n", flag, sizeof(flag));

    printf("\n=== Format Specifier Summary ===\n");
    printf("  %%d  = int          %%u  = unsigned int\n");
    printf("  %%ld = long         %%lld = long long\n");
    printf("  %%f  = float/double %%Lf  = long double\n");
    printf("  %%c  = char         %%s   = string\n");
    printf("  %%x  = hex          %%o   = octal\n");
    printf("  %%zu = size_t       %%p   = pointer\n");

    return 0;
}
