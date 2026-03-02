/*
 * Bitmask DP
 * TSP, Set Partition, Subset DP
 *
 * Using bit operations to represent set states in dynamic programming.
 */

#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

#define INF INT_MAX
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

/* =============================================================================
 * 1. Bit Operation Basics
 * ============================================================================= */

void print_binary(int n, int bits) {
    for (int i = bits - 1; i >= 0; i--) {
        printf("%d", (n >> i) & 1);
    }
}

int count_bits(int n) {
    int count = 0;
    while (n) {
        count += n & 1;
        n >>= 1;
    }
    return count;
}

/* =============================================================================
 * 2. TSP (Traveling Salesman Problem)
 * ============================================================================= */

int tsp(int** dist, int n) {
    int full_mask = (1 << n) - 1;
    int** dp = malloc((1 << n) * sizeof(int*));
    for (int i = 0; i < (1 << n); i++) {
        dp[i] = malloc(n * sizeof(int));
        for (int j = 0; j < n; j++) dp[i][j] = INF;
    }

    dp[1][0] = 0;  /* Visit starting point 0 */

    for (int mask = 1; mask <= full_mask; mask++) {
        for (int last = 0; last < n; last++) {
            if (!(mask & (1 << last))) continue;
            if (dp[mask][last] == INF) continue;

            for (int next = 0; next < n; next++) {
                if (mask & (1 << next)) continue;
                if (dist[last][next] == 0) continue;

                int new_mask = mask | (1 << next);
                int new_cost = dp[mask][last] + dist[last][next];
                if (new_cost < dp[new_mask][next]) {
                    dp[new_mask][next] = new_cost;
                }
            }
        }
    }

    int result = INF;
    for (int last = 1; last < n; last++) {
        if (dp[full_mask][last] != INF && dist[last][0] > 0) {
            int cost = dp[full_mask][last] + dist[last][0];
            if (cost < result) result = cost;
        }
    }

    for (int i = 0; i < (1 << n); i++) free(dp[i]);
    free(dp);
    return result;
}

/* =============================================================================
 * 3. Subset Sum
 * ============================================================================= */

int subset_sum_exists(int arr[], int n, int target) {
    int full = 1 << n;
    for (int mask = 0; mask < full; mask++) {
        int sum = 0;
        for (int i = 0; i < n; i++) {
            if (mask & (1 << i)) sum += arr[i];
        }
        if (sum == target) return 1;
    }
    return 0;
}

/* =============================================================================
 * 4. Set Partition (Minimum Difference Between Two Subsets)
 * ============================================================================= */

int min_partition_diff(int arr[], int n) {
    int total = 0;
    for (int i = 0; i < n; i++) total += arr[i];

    int full = 1 << n;
    int min_diff = total;

    for (int mask = 0; mask < full; mask++) {
        int sum = 0;
        for (int i = 0; i < n; i++) {
            if (mask & (1 << i)) sum += arr[i];
        }
        int diff = abs(total - 2 * sum);
        if (diff < min_diff) min_diff = diff;
    }

    return min_diff;
}

/* =============================================================================
 * 5. Assignment Problem
 * ============================================================================= */

int assignment_problem(int** cost, int n) {
    int* dp = malloc((1 << n) * sizeof(int));
    for (int i = 0; i < (1 << n); i++) dp[i] = INF;
    dp[0] = 0;

    for (int mask = 0; mask < (1 << n); mask++) {
        if (dp[mask] == INF) continue;
        int person = count_bits(mask);
        if (person >= n) continue;

        for (int task = 0; task < n; task++) {
            if (mask & (1 << task)) continue;
            int new_mask = mask | (1 << task);
            int new_cost = dp[mask] + cost[person][task];
            if (new_cost < dp[new_mask]) dp[new_mask] = new_cost;
        }
    }

    int result = dp[(1 << n) - 1];
    free(dp);
    return result;
}

/* =============================================================================
 * 6. Hamilton Path Count
 * ============================================================================= */

int hamilton_paths(int** adj, int n, int start) {
    int** dp = malloc((1 << n) * sizeof(int*));
    for (int i = 0; i < (1 << n); i++) {
        dp[i] = calloc(n, sizeof(int));
    }

    dp[1 << start][start] = 1;

    for (int mask = 0; mask < (1 << n); mask++) {
        for (int last = 0; last < n; last++) {
            if (!(mask & (1 << last))) continue;
            if (dp[mask][last] == 0) continue;

            for (int next = 0; next < n; next++) {
                if (mask & (1 << next)) continue;
                if (!adj[last][next]) continue;

                int new_mask = mask | (1 << next);
                dp[new_mask][next] += dp[mask][last];
            }
        }
    }

    int full = (1 << n) - 1;
    int total = 0;
    for (int i = 0; i < n; i++) total += dp[full][i];

    for (int i = 0; i < (1 << n); i++) free(dp[i]);
    free(dp);
    return total;
}

/* =============================================================================
 * 7. SOS DP (Sum over Subsets)
 * ============================================================================= */

void sos_dp(int f[], int n) {
    int full = 1 << n;
    for (int i = 0; i < n; i++) {
        for (int mask = 0; mask < full; mask++) {
            if (mask & (1 << i)) {
                f[mask] += f[mask ^ (1 << i)];
            }
        }
    }
}

/* =============================================================================
 * Test
 * ============================================================================= */

int main(void) {
    printf("============================================================\n");
    printf("Bitmask DP Examples\n");
    printf("============================================================\n");

    /* 1. Bit Operations */
    printf("\n[1] Bit Operation Basics\n");
    printf("    5 = "); print_binary(5, 4); printf("\n");
    printf("    5 | 2 = %d\n", 5 | 2);
    printf("    5 & 2 = %d\n", 5 & 2);
    printf("    5 ^ 2 = %d\n", 5 ^ 2);
    printf("    count_bits(13) = %d\n", count_bits(13));

    /* 2. TSP */
    printf("\n[2] TSP (Traveling Salesman Problem)\n");
    int** dist = malloc(4 * sizeof(int*));
    for (int i = 0; i < 4; i++) dist[i] = malloc(4 * sizeof(int));
    int d[4][4] = {{0, 10, 15, 20}, {10, 0, 35, 25}, {15, 35, 0, 30}, {20, 25, 30, 0}};
    for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) dist[i][j] = d[i][j];
    printf("    Minimum cost: %d\n", tsp(dist, 4));
    for (int i = 0; i < 4; i++) free(dist[i]);
    free(dist);

    /* 3. Subset Sum */
    printf("\n[3] Subset Sum\n");
    int arr[] = {3, 34, 4, 12, 5, 2};
    printf("    Array: [3,34,4,12,5,2]\n");
    printf("    Sum 9 exists: %s\n", subset_sum_exists(arr, 6, 9) ? "yes" : "no");
    printf("    Sum 30 exists: %s\n", subset_sum_exists(arr, 6, 30) ? "yes" : "no");

    /* 4. Set Partition */
    printf("\n[4] Set Partition Minimum Difference\n");
    int arr2[] = {1, 6, 11, 5};
    printf("    [1,6,11,5]: min difference = %d\n", min_partition_diff(arr2, 4));

    /* 5. Assignment Problem */
    printf("\n[5] Assignment Problem\n");
    int** cost = malloc(3 * sizeof(int*));
    for (int i = 0; i < 3; i++) cost[i] = malloc(3 * sizeof(int));
    int c[3][3] = {{9, 2, 7}, {6, 4, 3}, {5, 8, 1}};
    for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) cost[i][j] = c[i][j];
    printf("    Cost matrix:\n");
    for (int i = 0; i < 3; i++) {
        printf("      ");
        for (int j = 0; j < 3; j++) printf("%d ", c[i][j]);
        printf("\n");
    }
    printf("    Minimum cost: %d\n", assignment_problem(cost, 3));
    for (int i = 0; i < 3; i++) free(cost[i]);
    free(cost);

    /* 6. Complexity */
    printf("\n[6] Bitmask DP Complexity\n");
    printf("    TSP: O(n^2 * 2^n)\n");
    printf("    Subset Sum: O(n * 2^n)\n");
    printf("    Assignment: O(n * 2^n)\n");
    printf("    SOS DP: O(n * 2^n)\n");

    printf("\n============================================================\n");

    return 0;
}
