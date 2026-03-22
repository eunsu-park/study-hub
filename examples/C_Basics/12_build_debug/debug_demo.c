/*
 * debug_demo.c — Program with intentional bugs for debugging practice.
 *
 * This file contains several common C bugs. Use a debugger (gdb/lldb)
 * or compiler warnings to find and fix them.
 *
 * Compile (debug): gcc -Wall -Wextra -g -std=c11 -o debug_demo debug_demo.c
 * Debug:           gdb ./debug_demo   (or lldb ./debug_demo on macOS)
 *
 * Bugs included:
 *   1. Off-by-one error in array access
 *   2. Uninitialized variable
 *   3. Integer overflow
 *   4. Missing break in switch (fall-through)
 *   5. Printf format mismatch
 */

#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

/* Bug 1: Off-by-one — loop goes one past the array */
void bug_off_by_one(void)
{
    printf("=== Bug 1: Off-by-one ===\n");
    int arr[5] = {10, 20, 30, 40, 50};

    /* BUG: should be i < 5, not i <= 5 */
    for (int i = 0; i <= 5; i++)
        printf("arr[%d] = %d\n", i, arr[i]);

    /* FIX:
     * for (int i = 0; i < 5; i++)
     *     printf("arr[%d] = %d\n", i, arr[i]);
     */
}

/* Bug 2: Uninitialized variable */
void bug_uninitialized(void)
{
    printf("\n=== Bug 2: Uninitialized Variable ===\n");
    int sum;  /* BUG: not initialized to 0 */

    for (int i = 1; i <= 5; i++)
        sum += i;

    printf("Sum of 1..5 = %d (expected 15)\n", sum);

    /* FIX:
     * int sum = 0;
     */
}

/* Bug 3: Integer overflow */
void bug_overflow(void)
{
    printf("\n=== Bug 3: Integer Overflow ===\n");
    int a = INT_MAX;
    int b = a + 1;  /* BUG: signed integer overflow (undefined behavior) */
    printf("INT_MAX     = %d\n", a);
    printf("INT_MAX + 1 = %d (overflow!)\n", b);

    /* FIX: use long or check before adding */
}

/* Bug 4: Missing break in switch */
void bug_switch_fallthrough(void)
{
    printf("\n=== Bug 4: Switch Fall-through ===\n");
    int choice = 2;
    switch (choice) {
        case 1:
            printf("Selected: Option 1\n");
            /* BUG: missing break — falls through to case 2 */
        case 2:
            printf("Selected: Option 2\n");
            /* BUG: missing break — falls through to case 3 */
        case 3:
            printf("Selected: Option 3\n");
            break;
        default:
            printf("Unknown option\n");
    }
    printf("(Expected only 'Option 2')\n");
}

/* Bug 5: Format specifier mismatch */
void bug_format_mismatch(void)
{
    printf("\n=== Bug 5: Format Mismatch ===\n");
    long long big = 9223372036854775807LL;
    double pi = 3.14159;

    /* BUG: %d for long long, %d for double */
    printf("big = %d (wrong specifier for long long)\n", (int)big);
    printf("pi  = %d (wrong specifier for double)\n", (int)pi);

    /* FIX:
     * printf("big = %lld\n", big);
     * printf("pi  = %f\n", pi);
     */
    printf("big = %lld (correct)\n", big);
    printf("pi  = %f (correct)\n", pi);
}

int main(void)
{
    printf("Debug Demo — find and fix the bugs!\n");
    printf("Compile with: gcc -Wall -Wextra -g -std=c11\n\n");

    /* Uncomment each to observe the bug: */
    /* bug_off_by_one(); */
    /* bug_uninitialized(); */
    bug_overflow();
    bug_switch_fallthrough();
    bug_format_mismatch();

    return 0;
}
