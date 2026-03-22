/*
 * functions_demo.c — Function declaration, call-by-value, pointer params, recursion.
 *
 * Compile: gcc -Wall -Wextra -std=c11 -o functions_demo functions_demo.c
 * Run:     ./functions_demo
 */

#include <stdio.h>

/* Forward declarations */
int    add(int a, int b);
void   swap_wrong(int a, int b);
void   swap(int *a, int *b);
long   factorial(int n);
void   print_array(const int *arr, int len);

/* Simple function: call-by-value */
int add(int a, int b)
{
    return a + b;
}

/* This swap does NOT work — demonstrates call-by-value pitfall */
void swap_wrong(int a, int b)
{
    int tmp = a;
    a = b;
    b = tmp;
    /* Changes are local; caller's variables are unaffected */
    (void)a; (void)b;
}

/* Correct swap using pointers (call-by-reference) */
void swap(int *a, int *b)
{
    int tmp = *a;
    *a = *b;
    *b = tmp;
}

/* Recursion: factorial */
long factorial(int n)
{
    if (n <= 1) return 1;
    return n * factorial(n - 1);
}

/* Array parameter (decays to pointer) */
void print_array(const int *arr, int len)
{
    printf("[");
    for (int i = 0; i < len; i++)
        printf("%s%d", i ? ", " : "", arr[i]);
    printf("]\n");
}

int main(void)
{
    /* Basic function call */
    printf("=== Call-by-Value ===\n");
    printf("add(3, 7) = %d\n", add(3, 7));

    /* Swap demo */
    printf("\n=== Swap Demo ===\n");
    int x = 10, y = 20;
    printf("Before swap_wrong: x=%d, y=%d\n", x, y);
    swap_wrong(x, y);
    printf("After  swap_wrong: x=%d, y=%d  (unchanged!)\n", x, y);

    printf("Before swap:       x=%d, y=%d\n", x, y);
    swap(&x, &y);
    printf("After  swap:       x=%d, y=%d  (swapped)\n", x, y);

    /* Recursion */
    printf("\n=== Recursion ===\n");
    for (int n = 0; n <= 10; n++)
        printf("%d! = %ld\n", n, factorial(n));

    /* Array parameter */
    printf("\n=== Array Parameter ===\n");
    int nums[] = {5, 3, 8, 1, 9, 2};
    int len = (int)(sizeof(nums) / sizeof(nums[0]));
    printf("nums = ");
    print_array(nums, len);

    return 0;
}
