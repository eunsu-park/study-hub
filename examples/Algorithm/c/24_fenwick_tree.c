/*
 * Fenwick Tree (Binary Indexed Tree)
 * Range Sum, Inversion Count, 2D BIT
 *
 * Simpler and more memory-efficient than a segment tree.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_N 100001

/* =============================================================================
 * 1. Basic Fenwick Tree (1-indexed)
 * ============================================================================= */

typedef struct {
    long long* tree;
    int n;
} BIT;

BIT* bit_create(int n) {
    BIT* bit = malloc(sizeof(BIT));
    bit->n = n;
    bit->tree = calloc(n + 1, sizeof(long long));
    return bit;
}

void bit_free(BIT* bit) {
    free(bit->tree);
    free(bit);
}

/* Add delta to i-th element (1-indexed) */
void bit_update(BIT* bit, int i, long long delta) {
    for (; i <= bit->n; i += i & (-i)) {
        bit->tree[i] += delta;
    }
}

/* Prefix sum [1, i] (1-indexed) */
long long bit_query(BIT* bit, int i) {
    long long sum = 0;
    for (; i > 0; i -= i & (-i)) {
        sum += bit->tree[i];
    }
    return sum;
}

/* Range sum [l, r] (1-indexed) */
long long bit_range_query(BIT* bit, int l, int r) {
    return bit_query(bit, r) - bit_query(bit, l - 1);
}

/* Initialize from array */
void bit_build(BIT* bit, int arr[], int n) {
    for (int i = 1; i <= n; i++) {
        bit_update(bit, i, arr[i - 1]);
    }
}

/* =============================================================================
 * 2. Range Update, Point Query (Difference Array)
 * ============================================================================= */

typedef struct {
    long long* tree;
    int n;
} BITDiff;

BITDiff* bitd_create(int n) {
    BITDiff* bitd = malloc(sizeof(BITDiff));
    bitd->n = n;
    bitd->tree = calloc(n + 2, sizeof(long long));
    return bitd;
}

void bitd_free(BITDiff* bitd) {
    free(bitd->tree);
    free(bitd);
}

void bitd_update_internal(BITDiff* bitd, int i, long long delta) {
    for (; i <= bitd->n; i += i & (-i)) {
        bitd->tree[i] += delta;
    }
}

/* Add delta to range [l, r] */
void bitd_range_update(BITDiff* bitd, int l, int r, long long delta) {
    bitd_update_internal(bitd, l, delta);
    bitd_update_internal(bitd, r + 1, -delta);
}

/* Query value of i-th element */
long long bitd_point_query(BITDiff* bitd, int i) {
    long long sum = 0;
    for (; i > 0; i -= i & (-i)) {
        sum += bitd->tree[i];
    }
    return sum;
}

/* =============================================================================
 * 3. Range Update, Range Query
 * ============================================================================= */

typedef struct {
    long long* tree1;
    long long* tree2;
    int n;
} BITRange;

BITRange* bitr_create(int n) {
    BITRange* bitr = malloc(sizeof(BITRange));
    bitr->n = n;
    bitr->tree1 = calloc(n + 2, sizeof(long long));
    bitr->tree2 = calloc(n + 2, sizeof(long long));
    return bitr;
}

void bitr_free(BITRange* bitr) {
    free(bitr->tree1);
    free(bitr->tree2);
    free(bitr);
}

void bitr_update_internal(long long* tree, int n, int i, long long delta) {
    for (; i <= n; i += i & (-i)) {
        tree[i] += delta;
    }
}

long long bitr_query_internal(long long* tree, int i) {
    long long sum = 0;
    for (; i > 0; i -= i & (-i)) {
        sum += tree[i];
    }
    return sum;
}

/* Add delta to range [l, r] */
void bitr_range_update(BITRange* bitr, int l, int r, long long delta) {
    bitr_update_internal(bitr->tree1, bitr->n, l, delta);
    bitr_update_internal(bitr->tree1, bitr->n, r + 1, -delta);
    bitr_update_internal(bitr->tree2, bitr->n, l, delta * (l - 1));
    bitr_update_internal(bitr->tree2, bitr->n, r + 1, -delta * r);
}

/* Prefix sum [1, i] */
long long bitr_prefix_sum(BITRange* bitr, int i) {
    return bitr_query_internal(bitr->tree1, i) * i -
           bitr_query_internal(bitr->tree2, i);
}

/* Range sum [l, r] */
long long bitr_range_query(BITRange* bitr, int l, int r) {
    return bitr_prefix_sum(bitr, r) - bitr_prefix_sum(bitr, l - 1);
}

/* =============================================================================
 * 4. 2D Fenwick Tree
 * ============================================================================= */

typedef struct {
    long long** tree;
    int rows;
    int cols;
} BIT2D;

BIT2D* bit2d_create(int rows, int cols) {
    BIT2D* bit = malloc(sizeof(BIT2D));
    bit->rows = rows;
    bit->cols = cols;
    bit->tree = malloc((rows + 1) * sizeof(long long*));
    for (int i = 0; i <= rows; i++) {
        bit->tree[i] = calloc(cols + 1, sizeof(long long));
    }
    return bit;
}

void bit2d_free(BIT2D* bit) {
    for (int i = 0; i <= bit->rows; i++) {
        free(bit->tree[i]);
    }
    free(bit->tree);
    free(bit);
}

/* Add delta at (x, y) */
void bit2d_update(BIT2D* bit, int x, int y, long long delta) {
    for (int i = x; i <= bit->rows; i += i & (-i)) {
        for (int j = y; j <= bit->cols; j += j & (-j)) {
            bit->tree[i][j] += delta;
        }
    }
}

/* Rectangle sum [(1,1), (x,y)] */
long long bit2d_query(BIT2D* bit, int x, int y) {
    long long sum = 0;
    for (int i = x; i > 0; i -= i & (-i)) {
        for (int j = y; j > 0; j -= j & (-j)) {
            sum += bit->tree[i][j];
        }
    }
    return sum;
}

/* Rectangle sum [(x1,y1), (x2,y2)] */
long long bit2d_range_query(BIT2D* bit, int x1, int y1, int x2, int y2) {
    return bit2d_query(bit, x2, y2) -
           bit2d_query(bit, x1 - 1, y2) -
           bit2d_query(bit, x2, y1 - 1) +
           bit2d_query(bit, x1 - 1, y1 - 1);
}

/* =============================================================================
 * 5. Inversion Count
 * ============================================================================= */

long long count_inversions(int arr[], int n) {
    /* Coordinate compression */
    int* sorted = malloc(n * sizeof(int));
    memcpy(sorted, arr, n * sizeof(int));

    /* Sort */
    for (int i = 0; i < n - 1; i++) {
        for (int j = i + 1; j < n; j++) {
            if (sorted[i] > sorted[j]) {
                int temp = sorted[i];
                sorted[i] = sorted[j];
                sorted[j] = temp;
            }
        }
    }

    /* Remove duplicates and compress */
    int* compressed = malloc(n * sizeof(int));
    for (int i = 0; i < n; i++) {
        int lo = 0, hi = n - 1;
        while (lo < hi) {
            int mid = (lo + hi) / 2;
            if (sorted[mid] < arr[i]) lo = mid + 1;
            else hi = mid;
        }
        compressed[i] = lo + 1;  /* 1-indexed */
    }

    /* Count inversions */
    BIT* bit = bit_create(n);
    long long inversions = 0;

    for (int i = n - 1; i >= 0; i--) {
        inversions += bit_query(bit, compressed[i] - 1);
        bit_update(bit, compressed[i], 1);
    }

    bit_free(bit);
    free(sorted);
    free(compressed);
    return inversions;
}

/* =============================================================================
 * 6. Find K-th Element
 * ============================================================================= */

/* Find the minimum index where prefix sum >= k */
int bit_find_kth(BIT* bit, long long k) {
    int pos = 0;
    int log_n = 0;
    while ((1 << (log_n + 1)) <= bit->n) log_n++;

    for (int i = log_n; i >= 0; i--) {
        int next = pos + (1 << i);
        if (next <= bit->n && bit->tree[next] < k) {
            pos = next;
            k -= bit->tree[pos];
        }
    }

    return pos + 1;
}

/* =============================================================================
 * Test
 * ============================================================================= */

int main(void) {
    printf("============================================================\n");
    printf("Fenwick Tree (BIT) Examples\n");
    printf("============================================================\n");

    /* 1. Basic BIT */
    printf("\n[1] Basic Fenwick Tree\n");
    int arr1[] = {1, 3, 5, 7, 9, 11};
    int n1 = 6;
    BIT* bit = bit_create(n1);
    bit_build(bit, arr1, n1);

    printf("    Array: [1, 3, 5, 7, 9, 11]\n");
    printf("    Range sum [1, 3]: %lld\n", bit_range_query(bit, 1, 3));
    printf("    Range sum [1, 6]: %lld\n", bit_range_query(bit, 1, 6));
    printf("    Range sum [3, 5]: %lld\n", bit_range_query(bit, 3, 5));

    bit_update(bit, 3, 5);  /* Add 5 to arr[3] */
    printf("    After arr[3] += 5, range sum [1, 6]: %lld\n", bit_range_query(bit, 1, 6));
    bit_free(bit);

    /* 2. Range Update, Point Query */
    printf("\n[2] Range Update, Point Query\n");
    BITDiff* bitd = bitd_create(5);

    bitd_range_update(bitd, 1, 3, 10);  /* Add 10 to [1, 3] */
    bitd_range_update(bitd, 2, 4, 5);   /* Add 5 to [2, 4] */

    printf("    After adding 10 to [1, 3], 5 to [2, 4]:\n");
    printf("    ");
    for (int i = 1; i <= 5; i++) {
        printf("arr[%d]=%lld ", i, bitd_point_query(bitd, i));
    }
    printf("\n");
    bitd_free(bitd);

    /* 3. Range Update, Range Query */
    printf("\n[3] Range Update, Range Query\n");
    BITRange* bitr = bitr_create(5);

    bitr_range_update(bitr, 1, 3, 10);
    bitr_range_update(bitr, 2, 5, 5);

    printf("    After adding 10 to [1, 3], 5 to [2, 5]:\n");
    printf("    Range sum [1, 5]: %lld\n", bitr_range_query(bitr, 1, 5));
    printf("    Range sum [2, 4]: %lld\n", bitr_range_query(bitr, 2, 4));
    bitr_free(bitr);

    /* 4. 2D BIT */
    printf("\n[4] 2D Fenwick Tree\n");
    BIT2D* bit2d = bit2d_create(4, 4);

    bit2d_update(bit2d, 1, 1, 1);
    bit2d_update(bit2d, 2, 2, 2);
    bit2d_update(bit2d, 3, 3, 3);
    bit2d_update(bit2d, 2, 3, 4);

    printf("    Set (1,1)=1, (2,2)=2, (3,3)=3, (2,3)=4\n");
    printf("    [(1,1), (3,3)] sum: %lld\n", bit2d_range_query(bit2d, 1, 1, 3, 3));
    printf("    [(2,2), (3,3)] sum: %lld\n", bit2d_range_query(bit2d, 2, 2, 3, 3));
    bit2d_free(bit2d);

    /* 5. Inversion Count */
    printf("\n[5] Inversion Count\n");
    int arr2[] = {8, 4, 2, 1};
    printf("    Array: [8, 4, 2, 1]\n");
    printf("    Inversion count: %lld\n", count_inversions(arr2, 4));

    int arr3[] = {1, 3, 2, 3, 1};
    printf("    Array: [1, 3, 2, 3, 1]\n");
    printf("    Inversion count: %lld\n", count_inversions(arr3, 5));

    /* 6. K-th Element */
    printf("\n[6] Find K-th Element\n");
    BIT* bit_kth = bit_create(10);
    bit_update(bit_kth, 2, 1);  /* Add 2 */
    bit_update(bit_kth, 5, 1);  /* Add 5 */
    bit_update(bit_kth, 3, 1);  /* Add 3 */
    bit_update(bit_kth, 7, 1);  /* Add 7 */

    printf("    Set: {2, 3, 5, 7}\n");
    printf("    1st element: %d\n", bit_find_kth(bit_kth, 1));
    printf("    2nd element: %d\n", bit_find_kth(bit_kth, 2));
    printf("    3rd element: %d\n", bit_find_kth(bit_kth, 3));
    printf("    4th element: %d\n", bit_find_kth(bit_kth, 4));
    bit_free(bit_kth);

    /* 7. Complexity */
    printf("\n[7] Complexity Comparison (BIT vs Segment Tree)\n");
    printf("    | Operation      | BIT        | Segment Tree  |\n");
    printf("    |----------------|------------|---------------|\n");
    printf("    | Point update   | O(log n)   | O(log n)      |\n");
    printf("    | Range sum      | O(log n)   | O(log n)      |\n");
    printf("    | Range update   | O(log n)*  | O(log n)      |\n");
    printf("    | Space          | O(n)       | O(4n)         |\n");
    printf("    | Implementation | Easy       | Medium        |\n");
    printf("    * Range update requires additional array\n");

    printf("\n============================================================\n");

    return 0;
}
