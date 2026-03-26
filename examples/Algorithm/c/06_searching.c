/*
 * Searching Algorithms
 * Linear Search, Binary Search, Parametric Search
 *
 * Various searching techniques and binary search applications.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

/* =============================================================================
 * 1. Linear Search - O(n)
 * ============================================================================= */

int linear_search(int arr[], int n, int target) {
    for (int i = 0; i < n; i++) {
        if (arr[i] == target)
            return i;
    }
    return -1;
}

/* =============================================================================
 * 2. Binary Search - O(log n)
 * ============================================================================= */

int binary_search(int arr[], int n, int target) {
    int left = 0, right = n - 1;

    while (left <= right) {
        int mid = left + (right - left) / 2;

        if (arr[mid] == target)
            return mid;
        else if (arr[mid] < target)
            left = mid + 1;
        else
            right = mid - 1;
    }

    return -1;
}

/* Recursive version */
int binary_search_recursive(int arr[], int left, int right, int target) {
    if (left > right)
        return -1;

    int mid = left + (right - left) / 2;

    if (arr[mid] == target)
        return mid;
    else if (arr[mid] < target)
        return binary_search_recursive(arr, mid + 1, right, target);
    else
        return binary_search_recursive(arr, left, mid - 1, target);
}

/* =============================================================================
 * 3. Lower Bound / Upper Bound
 * ============================================================================= */

/* First position >= target */
int lower_bound(int arr[], int n, int target) {
    int left = 0, right = n;

    while (left < right) {
        int mid = left + (right - left) / 2;
        if (arr[mid] < target)
            left = mid + 1;
        else
            right = mid;
    }

    return left;
}

/* First position > target */
int upper_bound(int arr[], int n, int target) {
    int left = 0, right = n;

    while (left < right) {
        int mid = left + (right - left) / 2;
        if (arr[mid] <= target)
            left = mid + 1;
        else
            right = mid;
    }

    return left;
}

/* Count of target */
int count_occurrences(int arr[], int n, int target) {
    return upper_bound(arr, n, target) - lower_bound(arr, n, target);
}

/* =============================================================================
 * 4. Find Insertion Position
 * ============================================================================= */

int search_insert(int arr[], int n, int target) {
    int left = 0, right = n;

    while (left < right) {
        int mid = left + (right - left) / 2;
        if (arr[mid] < target)
            left = mid + 1;
        else
            right = mid;
    }

    return left;
}

/* =============================================================================
 * 5. Search in Rotated Sorted Array
 * ============================================================================= */

int search_rotated(int arr[], int n, int target) {
    int left = 0, right = n - 1;

    while (left <= right) {
        int mid = left + (right - left) / 2;

        if (arr[mid] == target)
            return mid;

        /* Left half is sorted */
        if (arr[left] <= arr[mid]) {
            if (arr[left] <= target && target < arr[mid])
                right = mid - 1;
            else
                left = mid + 1;
        }
        /* Right half is sorted */
        else {
            if (arr[mid] < target && target <= arr[right])
                left = mid + 1;
            else
                right = mid - 1;
        }
    }

    return -1;
}

/* Find minimum in rotated array */
int find_min_rotated(int arr[], int n) {
    int left = 0, right = n - 1;

    while (left < right) {
        int mid = left + (right - left) / 2;

        if (arr[mid] > arr[right])
            left = mid + 1;
        else
            right = mid;
    }

    return arr[left];
}

/* =============================================================================
 * 6. Parametric Search
 * ============================================================================= */

/* Minimize the maximum sum when splitting array into k parts */
bool can_split(int arr[], int n, int max_sum, int k) {
    int count = 1;
    int current_sum = 0;

    for (int i = 0; i < n; i++) {
        if (arr[i] > max_sum)
            return false;

        if (current_sum + arr[i] > max_sum) {
            count++;
            current_sum = arr[i];
        } else {
            current_sum += arr[i];
        }
    }

    return count <= k;
}

int split_array_min_max(int arr[], int n, int k) {
    int left = 0, right = 0;

    for (int i = 0; i < n; i++) {
        if (arr[i] > left) left = arr[i];
        right += arr[i];
    }

    while (left < right) {
        int mid = left + (right - left) / 2;

        if (can_split(arr, n, mid, k))
            right = mid;
        else
            left = mid + 1;
    }

    return left;
}

/* Tree cutting: find the maximum height H to cut and obtain at least M wood */
long long cut_trees(int trees[], int n, long long target) {
    long long left = 0, right = 0;

    for (int i = 0; i < n; i++) {
        if (trees[i] > right)
            right = trees[i];
    }

    while (left < right) {
        long long mid = left + (right - left + 1) / 2;
        long long total = 0;

        for (int i = 0; i < n; i++) {
            if (trees[i] > mid)
                total += trees[i] - mid;
        }

        if (total >= target)
            left = mid;
        else
            right = mid - 1;
    }

    return left;
}

/* =============================================================================
 * 7. Real-valued Binary Search
 * ============================================================================= */

double sqrt_binary_search(double x) {
    if (x < 0) return -1;
    if (x < 1) {
        double lo = x, hi = 1.0;
        while (hi - lo > 1e-9) {
            double mid = (lo + hi) / 2;
            if (mid * mid < x)
                lo = mid;
            else
                hi = mid;
        }
        return lo;
    }

    double lo = 1.0, hi = x;
    while (hi - lo > 1e-9) {
        double mid = (lo + hi) / 2;
        if (mid * mid < x)
            lo = mid;
        else
            hi = mid;
    }
    return lo;
}

/* =============================================================================
 * 8. Find Peak Element
 * ============================================================================= */

int find_peak(int arr[], int n) {
    int left = 0, right = n - 1;

    while (left < right) {
        int mid = left + (right - left) / 2;

        if (arr[mid] < arr[mid + 1])
            left = mid + 1;
        else
            right = mid;
    }

    return left;
}

/* =============================================================================
 * Test
 * ============================================================================= */

void print_array(int arr[], int n) {
    printf("[");
    for (int i = 0; i < n; i++) {
        printf("%d", arr[i]);
        if (i < n - 1) printf(", ");
    }
    printf("]");
}

int main(void) {
    printf("============================================================\n");
    printf("Searching Algorithms Examples\n");
    printf("============================================================\n");

    /* 1. Binary Search */
    printf("\n[1] Binary Search\n");
    int arr1[] = {1, 3, 5, 7, 9, 11, 13, 15};
    printf("    Array: ");
    print_array(arr1, 8);
    printf("\n");
    printf("    Position of 7: %d\n", binary_search(arr1, 8, 7));
    printf("    Position of 6: %d\n", binary_search(arr1, 8, 6));

    /* 2. Lower/Upper Bound */
    printf("\n[2] Lower/Upper Bound\n");
    int arr2[] = {1, 2, 2, 2, 3, 4, 4, 5};
    printf("    Array: ");
    print_array(arr2, 8);
    printf("\n");
    printf("    lower_bound(2): %d\n", lower_bound(arr2, 8, 2));
    printf("    upper_bound(2): %d\n", upper_bound(arr2, 8, 2));
    printf("    Count of 2: %d\n", count_occurrences(arr2, 8, 2));

    /* 3. Insertion Position */
    printf("\n[3] Insertion Position\n");
    int arr3[] = {1, 3, 5, 7};
    printf("    Array: [1,3,5,7]\n");
    printf("    Insert position of 4: %d\n", search_insert(arr3, 4, 4));
    printf("    Insert position of 6: %d\n", search_insert(arr3, 4, 6));

    /* 4. Rotated Array */
    printf("\n[4] Rotated Array Search\n");
    int arr4[] = {4, 5, 6, 7, 0, 1, 2};
    printf("    Array: ");
    print_array(arr4, 7);
    printf("\n");
    printf("    Position of 0: %d\n", search_rotated(arr4, 7, 0));
    printf("    Minimum: %d\n", find_min_rotated(arr4, 7));

    /* 5. Parametric Search - Array Split */
    printf("\n[5] Parametric Search - Array Split\n");
    int arr5[] = {7, 2, 5, 10, 8};
    printf("    Array: [7,2,5,10,8], k=2\n");
    printf("    Minimum of maximum sum: %d\n", split_array_min_max(arr5, 5, 2));

    /* 6. Tree Cutting */
    printf("\n[6] Tree Cutting\n");
    int trees[] = {20, 15, 10, 17};
    long long target = 7;
    printf("    Tree heights: [20,15,10,17], required: %lld\n", target);
    printf("    Maximum cut height: %lld\n", cut_trees(trees, 4, target));

    /* 7. Real-valued Binary Search */
    printf("\n[7] Square Root Binary Search\n");
    printf("    sqrt(2): %.6f\n", sqrt_binary_search(2));
    printf("    sqrt(10): %.6f\n", sqrt_binary_search(10));

    /* 8. Find Peak */
    printf("\n[8] Find Peak Element\n");
    int arr8[] = {1, 2, 1, 3, 5, 6, 4};
    printf("    Array: [1,2,1,3,5,6,4]\n");
    int peak_idx = find_peak(arr8, 7);
    printf("    Peak index: %d (value: %d)\n", peak_idx, arr8[peak_idx]);

    /* 9. Algorithm Summary */
    printf("\n[9] Binary Search Applications Summary\n");
    printf("    | Problem Type     | Key Idea                  |\n");
    printf("    |------------------|---------------------------|\n");
    printf("    | lower_bound      | arr[mid] < target         |\n");
    printf("    | upper_bound      | arr[mid] <= target        |\n");
    printf("    | Rotated array    | Find the sorted half      |\n");
    printf("    | Parametric       | Convert to decision prob  |\n");
    printf("    | Min of max       | Binary search on answer   |\n");

    printf("\n============================================================\n");

    return 0;
}
