/**
 * Profiling Demo — Sorting Algorithm Comparison
 *
 * Demonstrates:
 *   - Performance measurement with clock()
 *   - Algorithm comparison (bubble sort vs quicksort)
 *   - Cache-friendly vs cache-unfriendly access patterns
 *   - Profiling-guided optimization
 *
 * Build:
 *   gcc -Wall -O2 -o profiling_demo profiling_demo.c -lm
 *
 * Profile with gprof:
 *   gcc -Wall -pg -O2 -o profiling_demo profiling_demo.c -lm
 *   ./profiling_demo
 *   gprof profiling_demo gmon.out
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ── Sorting Algorithms ─────────────────────────────────────────── */

static int compare_count = 0;
static int swap_count = 0;

static void reset_counters(void) {
    compare_count = 0;
    swap_count = 0;
}

static void swap(int *a, int *b) {
    swap_count++;
    int tmp = *a;
    *a = *b;
    *b = tmp;
}

/**
 * Bubble Sort — O(n^2)
 * Simple but slow. Good baseline for comparison.
 */
void bubble_sort(int *arr, int n) {
    for (int i = 0; i < n - 1; i++) {
        int swapped = 0;
        for (int j = 0; j < n - i - 1; j++) {
            compare_count++;
            if (arr[j] > arr[j + 1]) {
                swap(arr + j, arr + j + 1);
                swapped = 1;
            }
        }
        if (!swapped) break;  /* Early termination */
    }
}

/**
 * Quicksort — O(n log n) average
 * Partition-based divide and conquer.
 */
static int partition(int *arr, int lo, int hi) {
    int pivot = arr[hi];
    int i = lo - 1;
    for (int j = lo; j < hi; j++) {
        compare_count++;
        if (arr[j] <= pivot) {
            i++;
            swap(arr + i, arr + j);
        }
    }
    swap(arr + i + 1, arr + hi);
    return i + 1;
}

void quicksort(int *arr, int lo, int hi) {
    if (lo < hi) {
        int p = partition(arr, lo, hi);
        quicksort(arr, lo, p - 1);
        quicksort(arr, p + 1, hi);
    }
}

/* ── Timing Utilities ───────────────────────────────────────────── */

typedef struct {
    clock_t start;
    clock_t end;
} Timer;

static void timer_start(Timer *t) {
    t->start = clock();
}

static double timer_elapsed_ms(Timer *t) {
    t->end = clock();
    return (double)(t->end - t->start) / CLOCKS_PER_SEC * 1000.0;
}

/* ── Helper Functions ───────────────────────────────────────────── */

static int *generate_random_array(int n) {
    int *arr = malloc((size_t)n * sizeof(int));
    for (int i = 0; i < n; i++) {
        arr[i] = rand() % (n * 10);
    }
    return arr;
}

static int *copy_array(const int *src, int n) {
    int *dst = malloc((size_t)n * sizeof(int));
    memcpy(dst, src, (size_t)n * sizeof(int));
    return dst;
}

static int is_sorted(const int *arr, int n) {
    for (int i = 0; i < n - 1; i++) {
        if (arr[i] > arr[i + 1]) return 0;
    }
    return 1;
}

/* ── Demos ──────────────────────────────────────────────────────── */

void demo_sorting_comparison(void) {
    printf("========================================\n");
    printf("  Sorting Algorithm Comparison\n");
    printf("========================================\n\n");

    int sizes[] = {1000, 5000, 10000, 20000};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    Timer t;

    printf("  %-8s  %-12s %-12s  %-12s %-12s\n",
           "Size", "Bubble(ms)", "Quick(ms)", "Bubble Cmp", "Quick Cmp");
    printf("  %-8s  %-12s %-12s  %-12s %-12s\n",
           "----", "----------", "---------", "----------", "---------");

    for (int i = 0; i < num_sizes; i++) {
        int n = sizes[i];
        int *original = generate_random_array(n);

        /* Bubble sort */
        int *arr_bubble = copy_array(original, n);
        reset_counters();
        timer_start(&t);
        bubble_sort(arr_bubble, n);
        double bubble_ms = timer_elapsed_ms(&t);
        int bubble_cmp = compare_count;

        /* Quicksort */
        int *arr_quick = copy_array(original, n);
        reset_counters();
        timer_start(&t);
        quicksort(arr_quick, 0, n - 1);
        double quick_ms = timer_elapsed_ms(&t);
        int quick_cmp = compare_count;

        /* Verify */
        if (!is_sorted(arr_bubble, n) || !is_sorted(arr_quick, n)) {
            printf("  ERROR: Sort verification failed!\n");
        }

        printf("  %-8d  %-12.2f %-12.2f  %-12d %-12d\n",
               n, bubble_ms, quick_ms, bubble_cmp, quick_cmp);

        free(original);
        free(arr_bubble);
        free(arr_quick);
    }

    printf("\n  Bubble Sort: O(n^2) comparisons\n");
    printf("  Quicksort:   O(n log n) comparisons (average)\n");
}

void demo_cache_effects(void) {
    printf("\n========================================\n");
    printf("  Cache Access Pattern Comparison\n");
    printf("========================================\n\n");

    const int SIZE = 1024;
    Timer t;

    /* Allocate 2D matrix */
    int (*matrix)[SIZE] = malloc(sizeof(int[SIZE][SIZE]));
    for (int i = 0; i < SIZE; i++)
        for (int j = 0; j < SIZE; j++)
            matrix[i][j] = i + j;

    /* Row-major access (cache-friendly) */
    long long sum1 = 0;
    timer_start(&t);
    for (int iter = 0; iter < 10; iter++) {
        for (int i = 0; i < SIZE; i++)
            for (int j = 0; j < SIZE; j++)
                sum1 += matrix[i][j];
    }
    double row_ms = timer_elapsed_ms(&t);

    /* Column-major access (cache-unfriendly) */
    long long sum2 = 0;
    timer_start(&t);
    for (int iter = 0; iter < 10; iter++) {
        for (int j = 0; j < SIZE; j++)
            for (int i = 0; i < SIZE; i++)
                sum2 += matrix[i][j];
    }
    double col_ms = timer_elapsed_ms(&t);

    printf("  Matrix size: %d x %d (%lu KB)\n",
           SIZE, SIZE, sizeof(int[SIZE][SIZE]) / 1024);
    printf("  Iterations: 10\n\n");
    printf("  Row-major (cache-friendly):     %8.2f ms\n", row_ms);
    printf("  Column-major (cache-unfriendly): %8.2f ms\n", col_ms);
    printf("  Ratio: %.1fx faster\n", col_ms / row_ms);
    printf("  Checksums: %lld == %lld (%s)\n",
           sum1, sum2, sum1 == sum2 ? "match" : "MISMATCH");

    free(matrix);
}

void demo_optimization_levels(void) {
    printf("\n========================================\n");
    printf("  Optimization Level Guide\n");
    printf("========================================\n\n");

    printf("  Flag     Description              When to Use\n");
    printf("  ------   ---------------------    ----------------\n");
    printf("  -O0      No optimization           Debugging\n");
    printf("  -O1      Basic optimizations        Development\n");
    printf("  -O2      Standard optimizations     Release builds\n");
    printf("  -O3      Aggressive optimization    Performance-critical\n");
    printf("  -Os      Optimize for size           Embedded systems\n");
    printf("  -Ofast   -O3 + fast-math            Scientific computing\n\n");

    printf("  Profiling Tools:\n");
    printf("  %-15s %s\n", "gprof", "Function-level CPU profiling");
    printf("  %-15s %s\n", "callgrind", "Instruction-level profiling");
    printf("  %-15s %s\n", "massif", "Heap memory profiling");
    printf("  %-15s %s\n", "perf stat", "CPU counter statistics");
    printf("  %-15s %s\n", "time", "Wall-clock execution time");
}

int main(void) {
    srand(42);
    demo_sorting_comparison();
    demo_cache_effects();
    demo_optimization_levels();
    return 0;
}
