/*
 * Dynamic Programming (DP)
 * Fibonacci, Knapsack, LCS, LIS, Edit Distance
 *
 * A problem-solving technique using optimal solutions of subproblems.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

/* =============================================================================
 * 1. Fibonacci (Memoization & Tabulation)
 * ============================================================================= */

long long fib_memo[100];
int fib_computed[100];

long long fibonacci_memo(int n) {
    if (n <= 1) return n;
    if (fib_computed[n]) return fib_memo[n];
    fib_computed[n] = 1;
    fib_memo[n] = fibonacci_memo(n - 1) + fibonacci_memo(n - 2);
    return fib_memo[n];
}

long long fibonacci_tab(int n) {
    if (n <= 1) return n;
    long long* dp = malloc((n + 1) * sizeof(long long));
    dp[0] = 0; dp[1] = 1;
    for (int i = 2; i <= n; i++)
        dp[i] = dp[i - 1] + dp[i - 2];
    long long result = dp[n];
    free(dp);
    return result;
}

/* =============================================================================
 * 2. 0/1 Knapsack Problem
 * ============================================================================= */

int knapsack_01(int W, int weights[], int values[], int n) {
    int** dp = malloc((n + 1) * sizeof(int*));
    for (int i = 0; i <= n; i++)
        dp[i] = calloc(W + 1, sizeof(int));

    for (int i = 1; i <= n; i++) {
        for (int w = 0; w <= W; w++) {
            dp[i][w] = dp[i - 1][w];
            if (weights[i - 1] <= w) {
                int take = dp[i - 1][w - weights[i - 1]] + values[i - 1];
                dp[i][w] = MAX(dp[i][w], take);
            }
        }
    }

    int result = dp[n][W];
    for (int i = 0; i <= n; i++) free(dp[i]);
    free(dp);
    return result;
}

/* Space-optimized */
int knapsack_01_optimized(int W, int weights[], int values[], int n) {
    int* dp = calloc(W + 1, sizeof(int));

    for (int i = 0; i < n; i++) {
        for (int w = W; w >= weights[i]; w--) {
            dp[w] = MAX(dp[w], dp[w - weights[i]] + values[i]);
        }
    }

    int result = dp[W];
    free(dp);
    return result;
}

/* =============================================================================
 * 3. Longest Common Subsequence (LCS)
 * ============================================================================= */

int lcs(const char* s1, const char* s2) {
    int m = strlen(s1);
    int n = strlen(s2);

    int** dp = malloc((m + 1) * sizeof(int*));
    for (int i = 0; i <= m; i++)
        dp[i] = calloc(n + 1, sizeof(int));

    for (int i = 1; i <= m; i++) {
        for (int j = 1; j <= n; j++) {
            if (s1[i - 1] == s2[j - 1])
                dp[i][j] = dp[i - 1][j - 1] + 1;
            else
                dp[i][j] = MAX(dp[i - 1][j], dp[i][j - 1]);
        }
    }

    int result = dp[m][n];
    for (int i = 0; i <= m; i++) free(dp[i]);
    free(dp);
    return result;
}

/* =============================================================================
 * 4. Longest Increasing Subsequence (LIS)
 * ============================================================================= */

/* O(n^2) */
int lis_n2(int arr[], int n) {
    int* dp = malloc(n * sizeof(int));
    for (int i = 0; i < n; i++) dp[i] = 1;

    int max_len = 1;
    for (int i = 1; i < n; i++) {
        for (int j = 0; j < i; j++) {
            if (arr[j] < arr[i] && dp[j] + 1 > dp[i])
                dp[i] = dp[j] + 1;
        }
        if (dp[i] > max_len) max_len = dp[i];
    }

    free(dp);
    return max_len;
}

/* O(n log n) */
int lower_bound(int arr[], int n, int val) {
    int lo = 0, hi = n;
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (arr[mid] < val) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

int lis_nlogn(int arr[], int n) {
    int* tails = malloc(n * sizeof(int));
    int len = 0;

    for (int i = 0; i < n; i++) {
        int pos = lower_bound(tails, len, arr[i]);
        tails[pos] = arr[i];
        if (pos == len) len++;
    }

    free(tails);
    return len;
}

/* =============================================================================
 * 5. Edit Distance
 * ============================================================================= */

int edit_distance(const char* s1, const char* s2) {
    int m = strlen(s1);
    int n = strlen(s2);

    int** dp = malloc((m + 1) * sizeof(int*));
    for (int i = 0; i <= m; i++)
        dp[i] = malloc((n + 1) * sizeof(int));

    for (int i = 0; i <= m; i++) dp[i][0] = i;
    for (int j = 0; j <= n; j++) dp[0][j] = j;

    for (int i = 1; i <= m; i++) {
        for (int j = 1; j <= n; j++) {
            if (s1[i - 1] == s2[j - 1]) {
                dp[i][j] = dp[i - 1][j - 1];
            } else {
                dp[i][j] = 1 + MIN(dp[i - 1][j - 1],
                                   MIN(dp[i - 1][j], dp[i][j - 1]));
            }
        }
    }

    int result = dp[m][n];
    for (int i = 0; i <= m; i++) free(dp[i]);
    free(dp);
    return result;
}

/* =============================================================================
 * 6. Coin Change
 * ============================================================================= */

int coin_change(int coins[], int n, int amount) {
    int* dp = malloc((amount + 1) * sizeof(int));
    for (int i = 0; i <= amount; i++) dp[i] = amount + 1;
    dp[0] = 0;

    for (int i = 1; i <= amount; i++) {
        for (int j = 0; j < n; j++) {
            if (coins[j] <= i && dp[i - coins[j]] + 1 < dp[i]) {
                dp[i] = dp[i - coins[j]] + 1;
            }
        }
    }

    int result = (dp[amount] > amount) ? -1 : dp[amount];
    free(dp);
    return result;
}

/* =============================================================================
 * 7. Matrix Chain Multiplication
 * ============================================================================= */

int matrix_chain(int dims[], int n) {
    int** dp = malloc(n * sizeof(int*));
    for (int i = 0; i < n; i++) {
        dp[i] = calloc(n, sizeof(int));
    }

    for (int len = 2; len < n; len++) {
        for (int i = 1; i < n - len + 1; i++) {
            int j = i + len - 1;
            dp[i][j] = 2147483647;
            for (int k = i; k < j; k++) {
                int cost = dp[i][k] + dp[k + 1][j] + dims[i - 1] * dims[k] * dims[j];
                if (cost < dp[i][j]) dp[i][j] = cost;
            }
        }
    }

    int result = dp[1][n - 1];
    for (int i = 0; i < n; i++) free(dp[i]);
    free(dp);
    return result;
}

/* =============================================================================
 * Test
 * ============================================================================= */

int main(void) {
    printf("============================================================\n");
    printf("Dynamic Programming (DP) Examples\n");
    printf("============================================================\n");

    /* 1. Fibonacci */
    printf("\n[1] Fibonacci\n");
    printf("    fib(10) = %lld\n", fibonacci_tab(10));
    printf("    fib(45) = %lld\n", fibonacci_tab(45));

    /* 2. 0/1 Knapsack */
    printf("\n[2] 0/1 Knapsack Problem\n");
    int weights[] = {1, 2, 3, 4, 5};
    int values[] = {1, 6, 10, 16, 20};
    printf("    Weights: [1,2,3,4,5], Values: [1,6,10,16,20]\n");
    printf("    Max value for capacity 8: %d\n", knapsack_01(8, weights, values, 5));

    /* 3. LCS */
    printf("\n[3] Longest Common Subsequence (LCS)\n");
    printf("    'ABCDGH' vs 'AEDFHR': %d\n", lcs("ABCDGH", "AEDFHR"));
    printf("    'AGGTAB' vs 'GXTXAYB': %d\n", lcs("AGGTAB", "GXTXAYB"));

    /* 4. LIS */
    printf("\n[4] Longest Increasing Subsequence (LIS)\n");
    int arr[] = {10, 22, 9, 33, 21, 50, 41, 60, 80};
    printf("    [10,22,9,33,21,50,41,60,80]\n");
    printf("    O(n^2): %d\n", lis_n2(arr, 9));
    printf("    O(n log n): %d\n", lis_nlogn(arr, 9));

    /* 5. Edit Distance */
    printf("\n[5] Edit Distance\n");
    printf("    'kitten' -> 'sitting': %d\n", edit_distance("kitten", "sitting"));
    printf("    'sunday' -> 'saturday': %d\n", edit_distance("sunday", "saturday"));

    /* 6. Coin Change */
    printf("\n[6] Coin Change\n");
    int coins[] = {1, 5, 10, 25};
    printf("    Coins: [1,5,10,25], Amount: 30\n");
    printf("    Minimum coins: %d\n", coin_change(coins, 4, 30));

    /* 7. Matrix Chain */
    printf("\n[7] Matrix Chain Multiplication\n");
    int dims[] = {10, 30, 5, 60};
    printf("    Dimensions: 10x30, 30x5, 5x60\n");
    printf("    Minimum multiplications: %d\n", matrix_chain(dims, 4));

    /* 8. DP Problem Categories */
    printf("\n[8] DP Problem Categories\n");
    printf("    | Category      | Examples                     | Complexity     |\n");
    printf("    |---------------|------------------------------|----------------|\n");
    printf("    | 1D DP         | Fibonacci, Stair climbing    | O(n)           |\n");
    printf("    | 2D DP         | Knapsack, LCS, Edit dist.    | O(n^2) or O(nm)|\n");
    printf("    | Interval DP   | Matrix chain, Palindrome     | O(n^3)         |\n");

    printf("\n============================================================\n");

    return 0;
}
