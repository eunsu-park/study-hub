/*
 * Dynamic Programming (DP)
 * Fibonacci, Knapsack, LCS, LIS, Edit Distance, Matrix Chain
 *
 * Solves complex problems by breaking them into subproblems.
 */

#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <climits>

using namespace std;

// =============================================================================
// 1. Fibonacci (Top-down, Bottom-up)
// =============================================================================

// Top-down (Memoization)
vector<long long> memo;

long long fibTopDown(int n) {
    if (n <= 1) return n;
    if (memo[n] != -1) return memo[n];
    return memo[n] = fibTopDown(n - 1) + fibTopDown(n - 2);
}

// Bottom-up (Tabulation)
long long fibBottomUp(int n) {
    if (n <= 1) return n;

    vector<long long> dp(n + 1);
    dp[0] = 0;
    dp[1] = 1;

    for (int i = 2; i <= n; i++) {
        dp[i] = dp[i - 1] + dp[i - 2];
    }

    return dp[n];
}

// Space-optimized
long long fibOptimized(int n) {
    if (n <= 1) return n;

    long long prev2 = 0, prev1 = 1;
    for (int i = 2; i <= n; i++) {
        long long curr = prev1 + prev2;
        prev2 = prev1;
        prev1 = curr;
    }

    return prev1;
}

// =============================================================================
// 2. 0/1 Knapsack Problem
// =============================================================================

int knapsack01(int W, const vector<int>& weights, const vector<int>& values) {
    int n = weights.size();
    vector<vector<int>> dp(n + 1, vector<int>(W + 1, 0));

    for (int i = 1; i <= n; i++) {
        for (int w = 0; w <= W; w++) {
            dp[i][w] = dp[i-1][w];  // Don't include
            if (weights[i-1] <= w) {
                dp[i][w] = max(dp[i][w], dp[i-1][w - weights[i-1]] + values[i-1]);
            }
        }
    }

    return dp[n][W];
}

// Space-optimized
int knapsack01Optimized(int W, const vector<int>& weights, const vector<int>& values) {
    int n = weights.size();
    vector<int> dp(W + 1, 0);

    for (int i = 0; i < n; i++) {
        for (int w = W; w >= weights[i]; w--) {
            dp[w] = max(dp[w], dp[w - weights[i]] + values[i]);
        }
    }

    return dp[W];
}

// Unbounded Knapsack
int unboundedKnapsack(int W, const vector<int>& weights, const vector<int>& values) {
    int n = weights.size();
    vector<int> dp(W + 1, 0);

    for (int w = 1; w <= W; w++) {
        for (int i = 0; i < n; i++) {
            if (weights[i] <= w) {
                dp[w] = max(dp[w], dp[w - weights[i]] + values[i]);
            }
        }
    }

    return dp[W];
}

// =============================================================================
// 3. Longest Common Subsequence (LCS)
// =============================================================================

int lcs(const string& s1, const string& s2) {
    int m = s1.length(), n = s2.length();
    vector<vector<int>> dp(m + 1, vector<int>(n + 1, 0));

    for (int i = 1; i <= m; i++) {
        for (int j = 1; j <= n; j++) {
            if (s1[i-1] == s2[j-1]) {
                dp[i][j] = dp[i-1][j-1] + 1;
            } else {
                dp[i][j] = max(dp[i-1][j], dp[i][j-1]);
            }
        }
    }

    return dp[m][n];
}

// LCS string reconstruction
string lcsString(const string& s1, const string& s2) {
    int m = s1.length(), n = s2.length();
    vector<vector<int>> dp(m + 1, vector<int>(n + 1, 0));

    for (int i = 1; i <= m; i++) {
        for (int j = 1; j <= n; j++) {
            if (s1[i-1] == s2[j-1]) {
                dp[i][j] = dp[i-1][j-1] + 1;
            } else {
                dp[i][j] = max(dp[i-1][j], dp[i][j-1]);
            }
        }
    }

    // Backtracking
    string result;
    int i = m, j = n;
    while (i > 0 && j > 0) {
        if (s1[i-1] == s2[j-1]) {
            result = s1[i-1] + result;
            i--; j--;
        } else if (dp[i-1][j] > dp[i][j-1]) {
            i--;
        } else {
            j--;
        }
    }

    return result;
}

// =============================================================================
// 4. Longest Increasing Subsequence (LIS)
// =============================================================================

// O(n^2)
int lisQuadratic(const vector<int>& arr) {
    int n = arr.size();
    vector<int> dp(n, 1);

    for (int i = 1; i < n; i++) {
        for (int j = 0; j < i; j++) {
            if (arr[j] < arr[i]) {
                dp[i] = max(dp[i], dp[j] + 1);
            }
        }
    }

    return *max_element(dp.begin(), dp.end());
}

// O(n log n)
int lisNLogN(const vector<int>& arr) {
    vector<int> tails;

    for (int x : arr) {
        auto it = lower_bound(tails.begin(), tails.end(), x);
        if (it == tails.end()) {
            tails.push_back(x);
        } else {
            *it = x;
        }
    }

    return tails.size();
}

// LIS sequence reconstruction
vector<int> lisWithSequence(const vector<int>& arr) {
    int n = arr.size();
    vector<int> tails;
    vector<int> tailIdx;
    vector<int> prev(n, -1);

    for (int i = 0; i < n; i++) {
        auto it = lower_bound(tails.begin(), tails.end(), arr[i]);
        int pos = it - tails.begin();

        if (it == tails.end()) {
            tails.push_back(arr[i]);
            tailIdx.push_back(i);
        } else {
            *it = arr[i];
            tailIdx[pos] = i;
        }

        if (pos > 0) {
            prev[i] = tailIdx[pos - 1];
        }
    }

    // Backtracking
    vector<int> result;
    for (int i = tailIdx.back(); i != -1; i = prev[i]) {
        result.push_back(arr[i]);
    }
    reverse(result.begin(), result.end());

    return result;
}

// =============================================================================
// 5. Edit Distance
// =============================================================================

int editDistance(const string& s1, const string& s2) {
    int m = s1.length(), n = s2.length();
    vector<vector<int>> dp(m + 1, vector<int>(n + 1));

    for (int i = 0; i <= m; i++) dp[i][0] = i;
    for (int j = 0; j <= n; j++) dp[0][j] = j;

    for (int i = 1; i <= m; i++) {
        for (int j = 1; j <= n; j++) {
            if (s1[i-1] == s2[j-1]) {
                dp[i][j] = dp[i-1][j-1];
            } else {
                dp[i][j] = 1 + min({dp[i-1][j],      // Delete
                                    dp[i][j-1],      // Insert
                                    dp[i-1][j-1]});  // Replace
            }
        }
    }

    return dp[m][n];
}

// =============================================================================
// 6. Matrix Chain Multiplication
// =============================================================================

int matrixChainMultiplication(const vector<int>& dims) {
    int n = dims.size() - 1;
    vector<vector<int>> dp(n, vector<int>(n, 0));

    for (int len = 2; len <= n; len++) {
        for (int i = 0; i + len - 1 < n; i++) {
            int j = i + len - 1;
            dp[i][j] = INT_MAX;

            for (int k = i; k < j; k++) {
                int cost = dp[i][k] + dp[k+1][j] +
                           dims[i] * dims[k+1] * dims[j+1];
                dp[i][j] = min(dp[i][j], cost);
            }
        }
    }

    return dp[0][n-1];
}

// =============================================================================
// 7. Coin Change
// =============================================================================

// Minimum number of coins
int coinChange(const vector<int>& coins, int amount) {
    vector<int> dp(amount + 1, INT_MAX);
    dp[0] = 0;

    for (int i = 1; i <= amount; i++) {
        for (int coin : coins) {
            if (coin <= i && dp[i - coin] != INT_MAX) {
                dp[i] = min(dp[i], dp[i - coin] + 1);
            }
        }
    }

    return dp[amount] == INT_MAX ? -1 : dp[amount];
}

// Number of ways
int coinChangeWays(const vector<int>& coins, int amount) {
    vector<int> dp(amount + 1, 0);
    dp[0] = 1;

    for (int coin : coins) {
        for (int i = coin; i <= amount; i++) {
            dp[i] += dp[i - coin];
        }
    }

    return dp[amount];
}

// =============================================================================
// 8. Maximum Subarray Sum (Kadane's Algorithm)
// =============================================================================

int maxSubarraySum(const vector<int>& arr) {
    int maxSum = arr[0];
    int currSum = arr[0];

    for (int i = 1; i < (int)arr.size(); i++) {
        currSum = max(arr[i], currSum + arr[i]);
        maxSum = max(maxSum, currSum);
    }

    return maxSum;
}

// =============================================================================
// 9. Palindrome Substrings
// =============================================================================

// Longest palindromic substring
string longestPalindrome(const string& s) {
    int n = s.length();
    if (n == 0) return "";

    vector<vector<bool>> dp(n, vector<bool>(n, false));
    int start = 0, maxLen = 1;

    // Length 1
    for (int i = 0; i < n; i++) dp[i][i] = true;

    // Length 2
    for (int i = 0; i < n - 1; i++) {
        if (s[i] == s[i+1]) {
            dp[i][i+1] = true;
            start = i;
            maxLen = 2;
        }
    }

    // Length 3 and above
    for (int len = 3; len <= n; len++) {
        for (int i = 0; i + len - 1 < n; i++) {
            int j = i + len - 1;
            if (s[i] == s[j] && dp[i+1][j-1]) {
                dp[i][j] = true;
                start = i;
                maxLen = len;
            }
        }
    }

    return s.substr(start, maxLen);
}

// Minimum palindrome partition cuts
int minPalindromeCuts(const string& s) {
    int n = s.length();

    // isPalin[i][j]: whether s[i..j] is a palindrome
    vector<vector<bool>> isPalin(n, vector<bool>(n, false));
    for (int i = n - 1; i >= 0; i--) {
        for (int j = i; j < n; j++) {
            if (i == j) {
                isPalin[i][j] = true;
            } else if (s[i] == s[j]) {
                isPalin[i][j] = (j - i == 1) || isPalin[i+1][j-1];
            }
        }
    }

    // dp[i]: minimum number of cuts for s[0..i]
    vector<int> dp(n, 0);
    for (int i = 0; i < n; i++) {
        if (isPalin[0][i]) {
            dp[i] = 0;
        } else {
            dp[i] = INT_MAX;
            for (int j = 0; j < i; j++) {
                if (isPalin[j+1][i]) {
                    dp[i] = min(dp[i], dp[j] + 1);
                }
            }
        }
    }

    return dp[n-1];
}

// =============================================================================
// Test
// =============================================================================

void printVector(const vector<int>& v) {
    cout << "[";
    for (size_t i = 0; i < v.size(); i++) {
        cout << v[i];
        if (i < v.size() - 1) cout << ", ";
    }
    cout << "]";
}

int main() {
    cout << "============================================================" << endl;
    cout << "Dynamic Programming Example" << endl;
    cout << "============================================================" << endl;

    // 1. Fibonacci
    cout << "\n[1] Fibonacci" << endl;
    memo.assign(50, -1);
    cout << "    fib(10) Top-down: " << fibTopDown(10) << endl;
    cout << "    fib(10) Bottom-up: " << fibBottomUp(10) << endl;
    cout << "    fib(10) Optimized: " << fibOptimized(10) << endl;

    // 2. 0/1 Knapsack
    cout << "\n[2] 0/1 Knapsack" << endl;
    vector<int> weights = {2, 3, 4, 5};
    vector<int> values = {3, 4, 5, 6};
    int W = 8;
    cout << "    Weights: [2,3,4,5], Values: [3,4,5,6], Capacity: 8" << endl;
    cout << "    Max value: " << knapsack01(W, weights, values) << endl;

    // 3. LCS
    cout << "\n[3] Longest Common Subsequence (LCS)" << endl;
    string s1 = "ABCDGH", s2 = "AEDFHR";
    cout << "    s1: \"ABCDGH\", s2: \"AEDFHR\"" << endl;
    cout << "    LCS length: " << lcs(s1, s2) << endl;
    cout << "    LCS: \"" << lcsString(s1, s2) << "\"" << endl;

    // 4. LIS
    cout << "\n[4] Longest Increasing Subsequence (LIS)" << endl;
    vector<int> arr = {10, 9, 2, 5, 3, 7, 101, 18};
    cout << "    Array: [10,9,2,5,3,7,101,18]" << endl;
    cout << "    LIS length O(n^2): " << lisQuadratic(arr) << endl;
    cout << "    LIS length O(n log n): " << lisNLogN(arr) << endl;
    cout << "    LIS: ";
    printVector(lisWithSequence(arr));
    cout << endl;

    // 5. Edit Distance
    cout << "\n[5] Edit Distance" << endl;
    cout << "    \"horse\" -> \"ros\": " << editDistance("horse", "ros") << endl;

    // 6. Matrix Chain Multiplication
    cout << "\n[6] Matrix Chain Multiplication" << endl;
    vector<int> dims = {10, 30, 5, 60};
    cout << "    Matrix sizes: 10x30, 30x5, 5x60" << endl;
    cout << "    Min multiplications: " << matrixChainMultiplication(dims) << endl;

    // 7. Coin Change
    cout << "\n[7] Coin Change" << endl;
    vector<int> coins = {1, 2, 5};
    cout << "    Coins: [1,2,5], Amount: 11" << endl;
    cout << "    Min coins: " << coinChange(coins, 11) << endl;
    cout << "    Number of ways: " << coinChangeWays(coins, 11) << endl;

    // 8. Maximum Subarray Sum
    cout << "\n[8] Maximum Subarray Sum" << endl;
    vector<int> subArr = {-2, 1, -3, 4, -1, 2, 1, -5, 4};
    cout << "    Array: [-2,1,-3,4,-1,2,1,-5,4]" << endl;
    cout << "    Max sum: " << maxSubarraySum(subArr) << endl;

    // 9. Palindrome
    cout << "\n[9] Palindrome" << endl;
    cout << "    \"babad\" longest palindrome: \"" << longestPalindrome("babad") << "\"" << endl;
    cout << "    \"aab\" min cuts: " << minPalindromeCuts("aab") << endl;

    // 10. Complexity Summary
    cout << "\n[10] Complexity Summary" << endl;
    cout << "    | Problem           | Time          | Space      |" << endl;
    cout << "    |-------------------|---------------|------------|" << endl;
    cout << "    | Fibonacci         | O(n)          | O(1)       |" << endl;
    cout << "    | 0/1 Knapsack      | O(nW)         | O(W)       |" << endl;
    cout << "    | LCS               | O(mn)         | O(mn)      |" << endl;
    cout << "    | LIS               | O(n log n)    | O(n)       |" << endl;
    cout << "    | Edit Distance     | O(mn)         | O(n)       |" << endl;
    cout << "    | Matrix Chain      | O(n^3)        | O(n^2)     |" << endl;

    cout << "\n============================================================" << endl;

    return 0;
}
