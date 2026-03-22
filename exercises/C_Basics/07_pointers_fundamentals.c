/*
 * Exercises for Lesson 07: Pointers Fundamentals
 * Topic: C_Basics
 * Solutions to practice problems from the lesson.
 * Compile: gcc -Wall -Wextra -std=c11 -o ex07 07_pointers_fundamentals.c
 */
#include <stdio.h>
#include <string.h>

/* === Exercise 1: Swap via Pointers === */
/* Problem: Implement swap functions for int, double, and generic data. */

void swap_int(int *a, int *b) {
    int tmp = *a;
    *a = *b;
    *b = tmp;
}

void swap_double(double *a, double *b) {
    double tmp = *a;
    *a = *b;
    *b = tmp;
}

/* Generic swap using void pointer and memcpy */
void swap_generic(void *a, void *b, size_t size) {
    unsigned char tmp[size];  /* VLA for temporary storage */
    memcpy(tmp, a, size);
    memcpy(a, b, size);
    memcpy(b, tmp, size);
}

void exercise_1(void) {
    printf("=== Exercise 1: Swap via Pointers ===\n");

    /* Integer swap */
    int x = 10, y = 20;
    printf("Before: x=%d, y=%d\n", x, y);
    swap_int(&x, &y);
    printf("After:  x=%d, y=%d\n", x, y);

    /* Double swap */
    double a = 3.14, b = 2.72;
    printf("Before: a=%.2f, b=%.2f\n", a, b);
    swap_double(&a, &b);
    printf("After:  a=%.2f, b=%.2f\n", a, b);

    /* Generic swap with strings */
    char s1[20] = "hello";
    char s2[20] = "world";
    printf("Before: s1=\"%s\", s2=\"%s\"\n", s1, s2);
    swap_generic(s1, s2, sizeof(s1));
    printf("After:  s1=\"%s\", s2=\"%s\"\n", s1, s2);

    /*
     * Key takeaway: C passes by value, so to modify caller's variables,
     * we must pass pointers. swap_generic shows how void* enables
     * type-agnostic programming (similar to how qsort works).
     */
}

/* === Exercise 2: Array Operations Using Pointer Arithmetic === */
/* Problem: Implement common array operations using only pointer arithmetic
 *          (no [] subscript operator). */

/* Sum array elements using pointer arithmetic */
int array_sum(const int *arr, int size) {
    int sum = 0;
    const int *end = arr + size;
    while (arr < end) {
        sum += *arr;
        arr++;
    }
    return sum;
}

/* Find element in array, return pointer or NULL */
const int *array_find(const int *arr, int size, int target) {
    const int *end = arr + size;
    while (arr < end) {
        if (*arr == target) return arr;
        arr++;
    }
    return NULL;
}

/* Copy array using pointers */
void array_copy(int *dest, const int *src, int size) {
    const int *end = src + size;
    while (src < end) {
        *dest++ = *src++;
    }
}

/* Fill array with a value */
void array_fill(int *arr, int size, int value) {
    int *end = arr + size;
    while (arr < end) {
        *arr++ = value;
    }
}

void exercise_2(void) {
    printf("\n=== Exercise 2: Array Operations via Pointer Arithmetic ===\n");

    int data[] = {10, 25, 3, 47, 12, 36, 8, 51, 19, 42};
    int size = sizeof(data) / sizeof(data[0]);

    /* Sum */
    printf("Sum: %d\n", array_sum(data, size));

    /* Find */
    const int *found = array_find(data, size, 47);
    if (found) {
        printf("Found 47 at offset %td from start\n", found - data);
    }

    found = array_find(data, size, 99);
    printf("Found 99: %s\n", found ? "yes" : "no");

    /* Copy */
    int copy[10];
    array_copy(copy, data, size);
    printf("Copy[0]=%d, Copy[9]=%d\n", copy[0], copy[9]);

    /* Fill */
    array_fill(copy, size, 0);
    printf("After fill(0): copy[0]=%d, copy[9]=%d\n", copy[0], copy[9]);

    /* Pointer arithmetic demo */
    printf("\nPointer arithmetic demo:\n");
    int *p = data;
    printf("  *p       = %d (first element)\n", *p);
    printf("  *(p+3)   = %d (fourth element)\n", *(p + 3));
    printf("  p[3]     = %d (same as *(p+3))\n", p[3]);
    printf("  &data[5] - &data[2] = %td (pointer difference)\n",
           &data[5] - &data[2]);
}

int main(void) {
    exercise_1();
    exercise_2();

    printf("\nAll exercises completed!\n");
    return 0;
}
