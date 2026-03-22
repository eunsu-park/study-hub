/*
 * operators_demo.c — Arithmetic, relational, logical, and bitwise operators.
 *
 * Compile: gcc -Wall -Wextra -std=c11 -o operators_demo operators_demo.c
 * Run:     ./operators_demo
 */

#include <stdio.h>

int main(void)
{
    int a = 17, b = 5;

    /* Arithmetic operators */
    printf("=== Arithmetic ===\n");
    printf("%d + %d = %d\n", a, b, a + b);
    printf("%d - %d = %d\n", a, b, a - b);
    printf("%d * %d = %d\n", a, b, a * b);
    printf("%d / %d = %d  (integer division)\n", a, b, a / b);
    printf("%d %% %d = %d (modulo)\n", a, b, a % b);

    /* Increment / decrement */
    int x = 10;
    printf("\nx = %d, ++x = %d, x = %d\n", 10, ++x, x);
    x = 10;
    printf("x = %d, x++ = %d, x = %d\n", 10, x++, x);

    /* Relational operators */
    printf("\n=== Relational ===\n");
    printf("%d == %d : %d\n", a, b, a == b);
    printf("%d != %d : %d\n", a, b, a != b);
    printf("%d >  %d : %d\n", a, b, a > b);
    printf("%d <= %d : %d\n", a, b, a <= b);

    /* Logical operators */
    printf("\n=== Logical ===\n");
    int t = 1, f = 0;
    printf("1 && 0 = %d\n", t && f);
    printf("1 || 0 = %d\n", t || f);
    printf("!1     = %d\n", !t);

    /* Bitwise operators */
    printf("\n=== Bitwise (a=0x%X, b=0x%X) ===\n", a, b);
    printf("a & b  = 0x%X  (AND)\n",  a & b);
    printf("a | b  = 0x%X  (OR)\n",   a | b);
    printf("a ^ b  = 0x%X  (XOR)\n",  a ^ b);
    printf("~a     = 0x%X  (NOT)\n",  ~a);
    printf("a << 1 = %d    (left shift)\n",  a << 1);
    printf("a >> 1 = %d    (right shift)\n", a >> 1);

    /* Ternary operator */
    printf("\n=== Ternary ===\n");
    int max = (a > b) ? a : b;
    printf("max(%d, %d) = %d\n", a, b, max);

    /* Compound assignment */
    printf("\n=== Compound Assignment ===\n");
    int v = 100;
    printf("v = %d -> v += 5 -> %d\n", 100, (v += 5));
    printf("       -> v *= 2 -> %d\n", (v *= 2));
    printf("       -> v %%=7 -> %d\n", (v %= 7));

    return 0;
}
