// parallel_sort.c
// Parallel Merge Sort
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <time.h>

// Why: below this threshold, thread creation overhead exceeds the parallelism
// benefit — spawning a thread costs ~10-50us, which dominates for small arrays
#define THRESHOLD 10000  // Use single thread below this size

typedef struct {
    int* arr;
    int left;
    int right;
} SortTask;

// Merge
void merge(int* arr, int left, int mid, int right) {
    int n1 = mid - left + 1;
    int n2 = right - mid;

    // Why: temporary arrays are needed because merge overwrites the original —
    // in-place merge is possible but complex and has worse cache performance
    int* L = malloc(n1 * sizeof(int));
    int* R = malloc(n2 * sizeof(int));

    memcpy(L, arr + left, n1 * sizeof(int));
    memcpy(R, arr + mid + 1, n2 * sizeof(int));

    int i = 0, j = 0, k = left;
    while (i < n1 && j < n2) {
        arr[k++] = (L[i] <= R[j]) ? L[i++] : R[j++];
    }
    while (i < n1) arr[k++] = L[i++];
    while (j < n2) arr[k++] = R[j++];

    free(L);
    free(R);
}

// Single-threaded merge sort
void merge_sort_single(int* arr, int left, int right) {
    if (left < right) {
        int mid = left + (right - left) / 2;
        merge_sort_single(arr, left, mid);
        merge_sort_single(arr, mid + 1, right);
        merge(arr, left, mid, right);
    }
}

// Multi-threaded merge sort
void* merge_sort_parallel(void* arg) {
    SortTask* task = (SortTask*)arg;
    int* arr = task->arr;
    int left = task->left;
    int right = task->right;

    if (left >= right) return NULL;

    // Small arrays use single thread
    if (right - left < THRESHOLD) {
        merge_sort_single(arr, left, right);
        return NULL;
    }

    int mid = left + (right - left) / 2;

    // Why: only one half spawns a new thread while the other reuses the current
    // thread — this halves thread creation overhead compared to spawning two new
    // threads, and the current thread stays busy instead of idling during join
    SortTask left_task = { arr, left, mid };
    pthread_t left_thread;
    pthread_create(&left_thread, NULL, merge_sort_parallel, &left_task);

    // Right half: current thread
    SortTask right_task = { arr, mid + 1, right };
    merge_sort_parallel(&right_task);

    // Wait for left thread
    pthread_join(left_thread, NULL);

    // Merge
    merge(arr, left, mid, right);

    return NULL;
}

// Print array
void print_array(int* arr, int n) {
    for (int i = 0; i < n && i < 20; i++) {
        printf("%d ", arr[i]);
    }
    if (n > 20) printf("...");
    printf("\n");
}

// Verify array is sorted
int is_sorted(int* arr, int n) {
    for (int i = 1; i < n; i++) {
        if (arr[i] < arr[i - 1]) return 0;
    }
    return 1;
}

int main(void) {
    srand(time(NULL));

    int n = 1000000;  // One million elements
    int* arr1 = malloc(n * sizeof(int));
    int* arr2 = malloc(n * sizeof(int));

    // Generate random array
    for (int i = 0; i < n; i++) {
        arr1[i] = rand();
        arr2[i] = arr1[i];  // Copy
    }

    printf("Array size: %d\n\n", n);

    // Single-threaded sort
    clock_t start = clock();
    merge_sort_single(arr1, 0, n - 1);
    clock_t end = clock();
    double single_time = (double)(end - start) / CLOCKS_PER_SEC;

    printf("Single-threaded: %.3f sec\n", single_time);
    printf("Sort verification: %s\n\n", is_sorted(arr1, n) ? "OK" : "FAIL");

    // Multi-threaded sort
    start = clock();
    SortTask task = { arr2, 0, n - 1 };
    merge_sort_parallel(&task);
    end = clock();
    double parallel_time = (double)(end - start) / CLOCKS_PER_SEC;

    printf("Multi-threaded: %.3f sec\n", parallel_time);
    printf("Sort verification: %s\n\n", is_sorted(arr2, n) ? "OK" : "FAIL");

    printf("Speedup: %.2fx\n", single_time / parallel_time);

    free(arr1);
    free(arr2);

    return 0;
}
