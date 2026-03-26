/*
 * Bitmask DP
 * TSP, Subset Enumeration, Assignment Problem
 *
 * Represents set states as bits to perform DP.
 */

#include <iostream>
#include <vector>
#include <algorithm>
#include <climits>
#include <bitset>

using namespace std;

// =============================================================================
// 1. Bit Operation Basics
// =============================================================================

void bitOperations() {
    int n = 4;  // Set size
    int fullSet = (1 << n) - 1;  // {0, 1, 2, 3}

    cout << "[1] Bit Operation Basics" << endl;

    // Check if element i is in the set
    int set = 0b1010;  // {1, 3}
    cout << "    Set 1010 (decimal: " << set << ")" << endl;
    cout << "    Contains 1: " << ((set & (1 << 1)) ? "Yes" : "No") << endl;
    cout << "    Contains 2: " << ((set & (1 << 2)) ? "Yes" : "No") << endl;

    // Add/remove elements
    set |= (1 << 2);   // Add 2
    cout << "    After adding 2: " << bitset<4>(set) << endl;
    set &= ~(1 << 1);  // Remove 1
    cout << "    After removing 1: " << bitset<4>(set) << endl;

    // Toggle element
    set ^= (1 << 3);  // Toggle 3
    cout << "    After toggling 3: " << bitset<4>(set) << endl;

    // Set size (number of 1s)
    cout << "    Set size: " << __builtin_popcount(set) << endl;

    // Lowest set bit
    cout << "    Lowest set bit: " << (set & -set) << endl;
}

// =============================================================================
// 2. Subset Enumeration
// =============================================================================

void enumerateSubsets(int n) {
    cout << "\n[2] Subset Enumeration" << endl;
    cout << "    All subsets of set with n = " << n << ":" << endl;

    // All subsets
    cout << "    All: ";
    for (int mask = 0; mask < (1 << n); mask++) {
        cout << bitset<3>(mask) << " ";
    }
    cout << endl;

    // Subsets of a specific set
    int set = 0b101;  // {0, 2}
    cout << "    Subsets of " << bitset<3>(set) << ": ";
    for (int sub = set; ; sub = (sub - 1) & set) {
        cout << bitset<3>(sub) << " ";
        if (sub == 0) break;
    }
    cout << endl;
}

// =============================================================================
// 3. Traveling Salesman Problem (TSP)
// =============================================================================

const int INF = INT_MAX / 2;

int tsp(int n, const vector<vector<int>>& dist) {
    // dp[mask][i]: min cost when visiting set mask and currently at i
    vector<vector<int>> dp(1 << n, vector<int>(n, INF));

    dp[1][0] = 0;  // Start from city 0

    for (int mask = 1; mask < (1 << n); mask++) {
        for (int u = 0; u < n; u++) {
            if (!(mask & (1 << u))) continue;
            if (dp[mask][u] == INF) continue;

            for (int v = 0; v < n; v++) {
                if (mask & (1 << v)) continue;  // Already visited
                if (dist[u][v] == INF) continue;

                int newMask = mask | (1 << v);
                dp[newMask][v] = min(dp[newMask][v], dp[mask][u] + dist[u][v]);
            }
        }
    }

    // Return to start after visiting all cities
    int fullMask = (1 << n) - 1;
    int result = INF;
    for (int i = 0; i < n; i++) {
        if (dp[fullMask][i] != INF && dist[i][0] != INF) {
            result = min(result, dp[fullMask][i] + dist[i][0]);
        }
    }

    return result;
}

// TSP with path reconstruction
pair<int, vector<int>> tspWithPath(int n, const vector<vector<int>>& dist) {
    vector<vector<int>> dp(1 << n, vector<int>(n, INF));
    vector<vector<int>> parent(1 << n, vector<int>(n, -1));

    dp[1][0] = 0;

    for (int mask = 1; mask < (1 << n); mask++) {
        for (int u = 0; u < n; u++) {
            if (!(mask & (1 << u))) continue;
            if (dp[mask][u] == INF) continue;

            for (int v = 0; v < n; v++) {
                if (mask & (1 << v)) continue;
                if (dist[u][v] == INF) continue;

                int newMask = mask | (1 << v);
                if (dp[mask][u] + dist[u][v] < dp[newMask][v]) {
                    dp[newMask][v] = dp[mask][u] + dist[u][v];
                    parent[newMask][v] = u;
                }
            }
        }
    }

    // Final result
    int fullMask = (1 << n) - 1;
    int result = INF;
    int lastCity = -1;

    for (int i = 0; i < n; i++) {
        if (dp[fullMask][i] != INF && dist[i][0] != INF) {
            if (dp[fullMask][i] + dist[i][0] < result) {
                result = dp[fullMask][i] + dist[i][0];
                lastCity = i;
            }
        }
    }

    // Path reconstruction
    vector<int> path;
    int mask = fullMask;
    int curr = lastCity;

    while (curr != -1) {
        path.push_back(curr);
        int prev = parent[mask][curr];
        mask ^= (1 << curr);
        curr = prev;
    }

    reverse(path.begin(), path.end());
    path.push_back(0);  // Return to start

    return {result, path};
}

// =============================================================================
// 4. Assignment Problem
// =============================================================================

int minCostAssignment(int n, const vector<vector<int>>& cost) {
    // dp[mask]: min cost when jobs in mask have been assigned
    vector<int> dp(1 << n, INF);
    dp[0] = 0;

    for (int mask = 0; mask < (1 << n); mask++) {
        int person = __builtin_popcount(mask);  // Current person to assign
        if (person >= n) continue;

        for (int job = 0; job < n; job++) {
            if (mask & (1 << job)) continue;  // Job already assigned

            int newMask = mask | (1 << job);
            dp[newMask] = min(dp[newMask], dp[mask] + cost[person][job]);
        }
    }

    return dp[(1 << n) - 1];
}

// =============================================================================
// 5. Subset Sum
// =============================================================================

bool subsetSum(const vector<int>& nums, int target) {
    int n = nums.size();

    for (int mask = 0; mask < (1 << n); mask++) {
        int sum = 0;
        for (int i = 0; i < n; i++) {
            if (mask & (1 << i)) {
                sum += nums[i];
            }
        }
        if (sum == target) return true;
    }

    return false;
}

// Optimization: Meet in the Middle
bool subsetSumMITM(const vector<int>& nums, int target) {
    int n = nums.size();
    int half = n / 2;

    // All subset sums of the first half
    vector<int> leftSums;
    for (int mask = 0; mask < (1 << half); mask++) {
        int sum = 0;
        for (int i = 0; i < half; i++) {
            if (mask & (1 << i)) sum += nums[i];
        }
        leftSums.push_back(sum);
    }
    sort(leftSums.begin(), leftSums.end());

    // Check the second half
    int rightHalf = n - half;
    for (int mask = 0; mask < (1 << rightHalf); mask++) {
        int sum = 0;
        for (int i = 0; i < rightHalf; i++) {
            if (mask & (1 << i)) sum += nums[half + i];
        }

        // Check if target - sum exists in leftSums
        if (binary_search(leftSums.begin(), leftSums.end(), target - sum)) {
            return true;
        }
    }

    return false;
}

// =============================================================================
// 6. Hamiltonian Path
// =============================================================================

int countHamiltonianPaths(int n, const vector<vector<int>>& adj) {
    // dp[mask][i]: number of paths visiting set mask and ending at i
    vector<vector<int>> dp(1 << n, vector<int>(n, 0));

    // Initialize starting points
    for (int i = 0; i < n; i++) {
        dp[1 << i][i] = 1;
    }

    for (int mask = 1; mask < (1 << n); mask++) {
        for (int u = 0; u < n; u++) {
            if (!(mask & (1 << u))) continue;
            if (dp[mask][u] == 0) continue;

            for (int v : adj[u]) {
                if (mask & (1 << v)) continue;

                int newMask = mask | (1 << v);
                dp[newMask][v] += dp[mask][u];
            }
        }
    }

    // Total paths visiting all vertices
    int fullMask = (1 << n) - 1;
    int total = 0;
    for (int i = 0; i < n; i++) {
        total += dp[fullMask][i];
    }

    return total;
}

// =============================================================================
// 7. SOS DP (Sum over Subsets)
// =============================================================================

void sosDP(vector<int>& dp, int n) {
    // Store the sum of all subsets of mask in dp[mask]
    for (int i = 0; i < n; i++) {
        for (int mask = 0; mask < (1 << n); mask++) {
            if (mask & (1 << i)) {
                dp[mask] += dp[mask ^ (1 << i)];
            }
        }
    }
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
    cout << "Bitmask DP Example" << endl;
    cout << "============================================================" << endl;

    // 1. Bit Operation Basics
    bitOperations();

    // 2. Subset Enumeration
    enumerateSubsets(3);

    // 3. TSP
    cout << "\n[3] Traveling Salesman Problem (TSP)" << endl;
    vector<vector<int>> dist = {
        {0, 10, 15, 20},
        {10, 0, 35, 25},
        {15, 35, 0, 30},
        {20, 25, 30, 0}
    };
    cout << "    Distance matrix: 4x4" << endl;
    cout << "    Min cost: " << tsp(4, dist) << endl;

    auto [cost, path] = tspWithPath(4, dist);
    cout << "    Path: ";
    printVector(path);
    cout << endl;

    // 4. Assignment Problem
    cout << "\n[4] Assignment Problem" << endl;
    vector<vector<int>> costMatrix = {
        {9, 2, 7, 8},
        {6, 4, 3, 7},
        {5, 8, 1, 8},
        {7, 6, 9, 4}
    };
    cout << "    Cost matrix: 4x4" << endl;
    cout << "    Min cost: " << minCostAssignment(4, costMatrix) << endl;

    // 5. Subset Sum
    cout << "\n[5] Subset Sum" << endl;
    vector<int> nums = {3, 34, 4, 12, 5, 2};
    cout << "    Array: [3, 34, 4, 12, 5, 2]" << endl;
    cout << "    Sum 9 exists: " << (subsetSum(nums, 9) ? "Yes" : "No") << endl;
    cout << "    Sum 30 exists: " << (subsetSum(nums, 30) ? "Yes" : "No") << endl;

    // 6. SOS DP
    cout << "\n[6] SOS DP" << endl;
    vector<int> sos = {1, 2, 3, 4, 5, 6, 7, 8};  // 2^3 = 8
    cout << "    Initial: ";
    printVector(sos);
    cout << endl;
    sosDP(sos, 3);
    cout << "    SOS: ";
    printVector(sos);
    cout << endl;

    // 7. Complexity Summary
    cout << "\n[7] Complexity Summary" << endl;
    cout << "    | Problem            | Time          | Space      |" << endl;
    cout << "    |--------------------|---------------|------------|" << endl;
    cout << "    | TSP                | O(n^2 * 2^n)  | O(n * 2^n) |" << endl;
    cout << "    | Assignment         | O(n * 2^n)    | O(2^n)     |" << endl;
    cout << "    | Subset Sum         | O(2^n)        | O(1)       |" << endl;
    cout << "    | Meet in the Middle | O(n * 2^(n/2))| O(2^(n/2)) |" << endl;
    cout << "    | SOS DP             | O(n * 2^n)    | O(2^n)     |" << endl;

    // 8. Bitmask Tips
    cout << "\n[8] Bitmask Tips" << endl;
    cout << "    - n <= 20: Consider bitmask DP" << endl;
    cout << "    - n <= 25: Consider Meet in the Middle" << endl;
    cout << "    - __builtin_popcount(x): Count of 1-bits (GCC)" << endl;
    cout << "    - x & -x: Lowest set bit" << endl;
    cout << "    - x & (x-1): Remove lowest set bit" << endl;

    cout << "\n============================================================" << endl;

    return 0;
}
