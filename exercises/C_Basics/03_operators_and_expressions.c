/*
 * Exercises for Lesson 03: Operators and Expressions
 * Topic: C_Basics
 * Solutions to practice problems from the lesson.
 * Compile: gcc -Wall -Wextra -std=c11 -o ex03 03_operators_and_expressions.c
 */
#include <stdio.h>
#include <stdbool.h>

/* === Exercise 1: Expression Evaluation === */
/* Problem: Predict and verify the results of complex expressions. */
void exercise_1(void) {
    printf("=== Exercise 1: Expression Evaluation ===\n");

    int a = 5, b = 3, c = 2;

    /* Arithmetic precedence: * before + */
    int r1 = a + b * c;
    printf("5 + 3 * 2 = %d (expected: 11)\n", r1);

    /* Associativity: left-to-right for - */
    int r2 = a - b - c;
    printf("5 - 3 - 2 = %d (expected: 0)\n", r2);

    /* Mixed: assignment is right-to-left */
    int x, y;
    x = y = 10;
    printf("x = y = 10 -> x=%d, y=%d\n", x, y);

    /* Modulo and division */
    printf("17 / 5 = %d, 17 %% 5 = %d\n", 17 / 5, 17 % 5);
    printf("-17 / 5 = %d, -17 %% 5 = %d\n", -17 / 5, -17 % 5);

    /* Increment in expressions */
    int n = 5;
    int pre = ++n;   /* n becomes 6, pre = 6 */
    printf("++n: n=%d, pre=%d\n", n, pre);
    int post = n++;  /* post = 6, n becomes 7 */
    printf("n++: n=%d, post=%d\n", n, post);

    /* Ternary operator */
    int max = (a > b) ? a : b;
    printf("max(%d, %d) = %d\n", a, b, max);
}

/* === Exercise 2: Bitwise Flag Operations === */
/* Problem: Use bitwise operators to manage a set of permission flags. */

#define PERM_READ    (1 << 0)  /* 0001 */
#define PERM_WRITE   (1 << 1)  /* 0010 */
#define PERM_EXECUTE (1 << 2)  /* 0100 */
#define PERM_ADMIN   (1 << 3)  /* 1000 */

void print_permissions(unsigned int perms) {
    printf("  Permissions [%04b]: ", perms);
    if (perms & PERM_READ)    printf("READ ");
    if (perms & PERM_WRITE)   printf("WRITE ");
    if (perms & PERM_EXECUTE) printf("EXECUTE ");
    if (perms & PERM_ADMIN)   printf("ADMIN ");
    printf("\n");
}

void exercise_2(void) {
    printf("\n=== Exercise 2: Bitwise Flag Operations ===\n");

    unsigned int perms = 0;

    /* Set flags using OR */
    perms |= PERM_READ;
    perms |= PERM_WRITE;
    printf("After setting READ and WRITE:\n");
    print_permissions(perms);

    /* Add EXECUTE */
    perms |= PERM_EXECUTE;
    printf("After adding EXECUTE:\n");
    print_permissions(perms);

    /* Remove WRITE using AND + NOT */
    perms &= ~PERM_WRITE;
    printf("After removing WRITE:\n");
    print_permissions(perms);

    /* Toggle ADMIN using XOR */
    perms ^= PERM_ADMIN;
    printf("After toggling ADMIN on:\n");
    print_permissions(perms);

    perms ^= PERM_ADMIN;
    printf("After toggling ADMIN off:\n");
    print_permissions(perms);

    /* Check specific flag */
    bool has_read = (perms & PERM_READ) != 0;
    printf("Has READ permission: %s\n", has_read ? "yes" : "no");
}

/* === Exercise 3: Precedence Challenges === */
/* Problem: Identify and fix common precedence mistakes. */
void exercise_3(void) {
    printf("\n=== Exercise 3: Precedence Challenges ===\n");

    /* Challenge 1: & has lower precedence than == */
    int flags = 0x0F;
    /* BUG: if (flags & 0x04 == 0x04) -> parsed as flags & (0x04 == 0x04) */
    /* FIX: use parentheses */
    if ((flags & 0x04) == 0x04) {
        printf("Challenge 1: Flag 0x04 is set (correct with parens)\n");
    }

    /* Challenge 2: Shift vs addition */
    int val = 1 << 2 + 3;  /* parsed as 1 << (2+3) = 1 << 5 = 32 */
    int intended = (1 << 2) + 3;  /* = 4 + 3 = 7 */
    printf("Challenge 2: 1 << 2 + 3 = %d (probably meant %d)\n",
           val, intended);

    /* Challenge 3: Logical vs bitwise */
    int a = 1, b = 2;
    printf("Challenge 3: a & b = %d, a && b = %d\n", a & b, a && b);
    printf("  (bitwise AND vs logical AND)\n");

    /* Challenge 4: Comma operator */
    int x = (1, 2, 3);  /* comma operator: evaluates all, returns last */
    printf("Challenge 4: x = (1, 2, 3) -> x = %d\n", x);

    /* Challenge 5: sizeof with expressions */
    int arr[10];
    printf("Challenge 5: sizeof(arr) = %zu, sizeof(arr[0]) = %zu, "
           "element count = %zu\n",
           sizeof(arr), sizeof(arr[0]), sizeof(arr) / sizeof(arr[0]));
}

int main(void) {
    exercise_1();
    exercise_2();
    exercise_3();

    printf("\nAll exercises completed!\n");
    return 0;
}
