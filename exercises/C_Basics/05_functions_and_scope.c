/*
 * Exercises for Lesson 05: Functions and Scope
 * Topic: C_Basics
 * Solutions to practice problems from the lesson.
 * Compile: gcc -Wall -Wextra -std=c11 -o ex05 05_functions_and_scope.c
 */
#include <stdio.h>
#include <float.h>

/* === Exercise 1: Recursive Power Function === */
/* Problem: Implement power(base, exp) using recursion with O(log n) approach. */

/* Naive recursion: O(n) */
double power_naive(double base, int exp) {
    if (exp < 0) return 1.0 / power_naive(base, -exp);
    if (exp == 0) return 1.0;
    return base * power_naive(base, exp - 1);
}

/* Fast exponentiation: O(log n) */
double power_fast(double base, int exp) {
    if (exp < 0) return 1.0 / power_fast(base, -exp);
    if (exp == 0) return 1.0;
    if (exp % 2 == 0) {
        double half = power_fast(base, exp / 2);
        return half * half;
    }
    return base * power_fast(base, exp - 1);
}

void exercise_1(void) {
    printf("=== Exercise 1: Recursive Power Function ===\n");

    double test_cases[][2] = {{2.0, 10}, {3.0, 5}, {2.0, -3}, {5.0, 0}};
    int n = sizeof(test_cases) / sizeof(test_cases[0]);

    for (int i = 0; i < n; i++) {
        double base = test_cases[i][0];
        int exp = (int)test_cases[i][1];
        printf("power(%.1f, %d) = %.6f (naive) = %.6f (fast)\n",
               base, exp, power_naive(base, exp), power_fast(base, exp));
    }

    /*
     * Why O(log n) is better:
     * power_fast(2, 10) makes ~4 recursive calls (10->5->4->2->1->0)
     * power_naive(2, 10) makes 10 recursive calls
     * For large exponents, the difference is significant.
     */
}

/* === Exercise 2: Array Stats via Pointers === */
/* Problem: Compute min, max, and average of an array,
 *          returning results through pointer parameters. */

void array_stats(const int *arr, int size, int *min, int *max, double *avg) {
    if (size <= 0) return;

    *min = arr[0];
    *max = arr[0];
    long sum = arr[0];

    for (int i = 1; i < size; i++) {
        if (arr[i] < *min) *min = arr[i];
        if (arr[i] > *max) *max = arr[i];
        sum += arr[i];
    }
    *avg = (double)sum / size;
}

void exercise_2(void) {
    printf("\n=== Exercise 2: Array Stats (min/max/avg via pointers) ===\n");

    int data1[] = {42, 17, 93, 5, 67, 28, 81, 3, 56, 74};
    int size1 = sizeof(data1) / sizeof(data1[0]);

    int min, max;
    double avg;
    array_stats(data1, size1, &min, &max, &avg);
    printf("Array: {42, 17, 93, 5, 67, 28, 81, 3, 56, 74}\n");
    printf("  Min: %d, Max: %d, Avg: %.2f\n", min, max, avg);

    int data2[] = {-5, -1, -10, -3};
    int size2 = sizeof(data2) / sizeof(data2[0]);
    array_stats(data2, size2, &min, &max, &avg);
    printf("Array: {-5, -1, -10, -3}\n");
    printf("  Min: %d, Max: %d, Avg: %.2f\n", min, max, avg);

    int data3[] = {7};
    array_stats(data3, 1, &min, &max, &avg);
    printf("Array: {7}\n");
    printf("  Min: %d, Max: %d, Avg: %.2f\n", min, max, avg);

    /*
     * Key concepts demonstrated:
     * - Using pointer parameters to return multiple values
     * - const qualifier on input array (read-only contract)
     * - Handling edge case of single-element array
     */
}

/* === Exercise 3: Scope and Lifetime Demonstration === */
/* Problem: Show differences between local, static, and global scope. */

int global_counter = 0;  /* file scope, external linkage */

void increment_static(void) {
    static int call_count = 0;  /* static local: persists between calls */
    call_count++;
    global_counter++;
    printf("  static call_count = %d, global_counter = %d\n",
           call_count, global_counter);
}

void increment_local(void) {
    int call_count = 0;  /* auto local: reset each call */
    call_count++;
    global_counter++;
    printf("  local call_count = %d, global_counter = %d\n",
           call_count, global_counter);
}

void exercise_3(void) {
    printf("\n=== Exercise 3: Scope and Lifetime Demo ===\n");

    printf("Calling increment_static 3 times:\n");
    for (int i = 0; i < 3; i++) increment_static();

    printf("Calling increment_local 3 times:\n");
    for (int i = 0; i < 3; i++) increment_local();

    printf("Final global_counter = %d\n", global_counter);
    /*
     * static call_count retains its value: 1, 2, 3
     * local call_count resets each time: always 1
     * global_counter incremented by all functions: 6
     */
}

int main(void) {
    exercise_1();
    exercise_2();
    exercise_3();

    printf("\nAll exercises completed!\n");
    return 0;
}
