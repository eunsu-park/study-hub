/*
 * pointers_demo.c — Pointer declaration, dereferencing, arithmetic, and swap.
 *
 * Compile: gcc -Wall -Wextra -std=c11 -o pointers_demo pointers_demo.c
 * Run:     ./pointers_demo
 */

#include <stdio.h>

void swap(int *a, int *b)
{
    int tmp = *a;
    *a = *b;
    *b = tmp;
}

int main(void)
{
    /* Basic pointer usage */
    printf("=== Pointer Basics ===\n");
    int x = 42;
    int *p = &x;
    printf("x  = %d,  &x = %p\n", x, (void *)&x);
    printf("p  = %p,  *p = %d\n", (void *)p, *p);

    /* Modify value through pointer */
    *p = 100;
    printf("After *p = 100:  x = %d\n", x);

    /* Pointer arithmetic with arrays */
    printf("\n=== Pointer Arithmetic ===\n");
    int arr[] = {10, 20, 30, 40, 50};
    int *ptr = arr;  /* array decays to pointer */

    for (int i = 0; i < 5; i++)
        printf("*(ptr + %d) = %d   (addr %p)\n", i, *(ptr + i), (void *)(ptr + i));

    printf("\nDifference between &arr[4] and &arr[0] = %ld elements\n",
           (long)(&arr[4] - &arr[0]));

    /* Pointer to pointer */
    printf("\n=== Pointer to Pointer ===\n");
    int val = 7;
    int *p1  = &val;
    int **p2 = &p1;
    printf("val = %d, *p1 = %d, **p2 = %d\n", val, *p1, **p2);

    /* Swap using pointers */
    printf("\n=== Swap via Pointers ===\n");
    int a = 10, b = 20;
    printf("Before: a=%d, b=%d\n", a, b);
    swap(&a, &b);
    printf("After:  a=%d, b=%d\n", a, b);

    /* NULL pointer */
    printf("\n=== NULL Pointer ===\n");
    int *np = NULL;
    printf("np = %p\n", (void *)np);
    if (np == NULL)
        printf("Pointer is NULL — safe to check before dereferencing\n");

    /* const pointer vs pointer to const */
    printf("\n=== const Qualifiers ===\n");
    int m = 5, n = 10;
    const int *pc = &m;    /* pointer to const: can't modify *pc */
    int *const cp = &m;    /* const pointer: can't change cp itself */
    printf("*pc = %d (pointer to const int)\n", *pc);
    pc = &n;               /* OK: can reassign pointer */
    printf("*pc = %d (after pc = &n)\n", *pc);
    *cp = 99;              /* OK: can modify value through const pointer */
    printf("*cp = %d, m = %d (modified through const pointer)\n", *cp, m);

    return 0;
}
