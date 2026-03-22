/*
 * advanced_pointers_demo.c
 *
 * Demonstrates function pointers, void pointers, const correctness,
 * and qsort callback usage.
 *
 * Build:  gcc -Wall -Wextra -std=c11 -o advanced_pointers_demo advanced_pointers_demo.c
 * Run:    ./advanced_pointers_demo
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ── Function pointer typedef ─────────────────────────────────── */
typedef int (*compare_fn)(const void *, const void *);

/* ── Comparator callbacks for qsort ───────────────────────────── */
static int cmp_int_asc(const void *a, const void *b)
{
    return *(const int *)a - *(const int *)b;
}

static int cmp_int_desc(const void *a, const void *b)
{
    return *(const int *)b - *(const int *)a;
}

static int cmp_str(const void *a, const void *b)
{
    return strcmp(*(const char *const *)a, *(const char *const *)b);
}

/* ── Void-pointer generic swap ────────────────────────────────── */
static void generic_swap(void *a, void *b, size_t size)
{
    unsigned char tmp[size];  /* VLA — C11 optional, widely supported */
    memcpy(tmp, a, size);
    memcpy(a, b, size);
    memcpy(b, tmp, size);
}

/* ── Const correctness demo ───────────────────────────────────── */
static void print_array(const int *arr, size_t n)
{
    /* arr[i] is read-only here — const int * */
    for (size_t i = 0; i < n; i++)
        printf("%d%s", arr[i], i + 1 < n ? ", " : "\n");
}

/* ── Dispatch table using function pointers ───────────────────── */
static double add(double a, double b) { return a + b; }
static double sub(double a, double b) { return a - b; }
static double mul(double a, double b) { return a * b; }

typedef double (*math_op)(double, double);

static const struct {
    const char *name;
    math_op     fn;
} ops[] = {
    {"add", add},
    {"sub", sub},
    {"mul", mul},
};

int main(void)
{
    /* 1. qsort with function-pointer comparators */
    int nums[] = {42, 7, 19, 3, 88, 15};
    size_t n = sizeof nums / sizeof nums[0];

    printf("Original:   ");
    print_array(nums, n);

    qsort(nums, n, sizeof(int), cmp_int_asc);
    printf("Ascending:  ");
    print_array(nums, n);

    qsort(nums, n, sizeof(int), cmp_int_desc);
    printf("Descending: ");
    print_array(nums, n);

    /* 2. qsort on string array */
    const char *words[] = {"delta", "alpha", "charlie", "bravo"};
    size_t nw = sizeof words / sizeof words[0];
    qsort(words, nw, sizeof(char *), cmp_str);
    printf("Sorted strings: ");
    for (size_t i = 0; i < nw; i++)
        printf("%s ", words[i]);
    putchar('\n');

    /* 3. Void-pointer generic swap */
    int x = 10, y = 20;
    printf("\nBefore swap: x=%d, y=%d\n", x, y);
    generic_swap(&x, &y, sizeof(int));
    printf("After swap:  x=%d, y=%d\n", x, y);

    /* 4. Dispatch table */
    printf("\nDispatch table:\n");
    for (size_t i = 0; i < sizeof ops / sizeof ops[0]; i++)
        printf("  %s(5, 3) = %.1f\n", ops[i].name, ops[i].fn(5, 3));

    return 0;
}
