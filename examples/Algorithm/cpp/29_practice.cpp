/*
 * Practice Problems
 * Combined Problems (Various Algorithm Combinations)
 *
 * Common problem types frequently seen in coding tests.
 */

#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <numeric>
#include <climits>

using namespace std;

// =============================================================================
// 1. Subarray Sum (Two Pointers)
// =============================================================================

// Minimum length subarray with sum >= target
int minSubarrayLen(int target, const vector<int>& nums) {
    int n = nums.size();
    int left = 0, sum = 0;
    int minLen = INT_MAX;

    for (int right = 0; right < n; right++) {
        sum += nums[right];

        while (sum >= target) {
            minLen = min(minLen, right - left + 1);
            sum -= nums[left++];
        }
    }

    return minLen == INT_MAX ? 0 : minLen;
}

// =============================================================================
// 2. Job Scheduling (Greedy)
// =============================================================================

struct Job {
    int deadline, profit;
};

int jobScheduling(vector<Job>& jobs) {
    sort(jobs.begin(), jobs.end(), [](const Job& a, const Job& b) {
        return a.profit > b.profit;
    });

    int maxDeadline = 0;
    for (const auto& job : jobs) {
        maxDeadline = max(maxDeadline, job.deadline);
    }

    vector<bool> slots(maxDeadline + 1, false);
    int totalProfit = 0;

    for (const auto& job : jobs) {
        for (int t = job.deadline; t >= 1; t--) {
            if (!slots[t]) {
                slots[t] = true;
                totalProfit += job.profit;
                break;
            }
        }
    }

    return totalProfit;
}

// =============================================================================
// 3. Minimum Meeting Rooms (Event Sorting)
// =============================================================================

int minMeetingRooms(vector<pair<int, int>>& intervals) {
    vector<pair<int, int>> events;

    for (const auto& [start, end] : intervals) {
        events.push_back({start, 1});
        events.push_back({end, -1});
    }

    sort(events.begin(), events.end());

    int rooms = 0, maxRooms = 0;
    for (const auto& [time, type] : events) {
        rooms += type;
        maxRooms = max(maxRooms, rooms);
    }

    return maxRooms;
}

// =============================================================================
// 4. Palindrome Conversion (DP)
// =============================================================================

int minPalindromeInsertions(const string& s) {
    int n = s.length();
    vector<vector<int>> dp(n, vector<int>(n, 0));

    for (int len = 2; len <= n; len++) {
        for (int i = 0; i + len - 1 < n; i++) {
            int j = i + len - 1;
            if (s[i] == s[j]) {
                dp[i][j] = dp[i + 1][j - 1];
            } else {
                dp[i][j] = 1 + min(dp[i + 1][j], dp[i][j - 1]);
            }
        }
    }

    return dp[0][n - 1];
}

// =============================================================================
// 5. Number of Islands (DFS/BFS)
// =============================================================================

void dfsIsland(vector<vector<int>>& grid, int i, int j,
               vector<vector<bool>>& visited) {
    int rows = grid.size(), cols = grid[0].size();
    if (i < 0 || i >= rows || j < 0 || j >= cols) return;
    if (visited[i][j] || grid[i][j] == 0) return;

    visited[i][j] = true;
    int dx[] = {-1, 1, 0, 0};
    int dy[] = {0, 0, -1, 1};

    for (int d = 0; d < 4; d++) {
        dfsIsland(grid, i + dx[d], j + dy[d], visited);
    }
}

int numIslands(vector<vector<int>>& grid) {
    if (grid.empty()) return 0;

    int rows = grid.size(), cols = grid[0].size();
    vector<vector<bool>> visited(rows, vector<bool>(cols, false));
    int count = 0;

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (grid[i][j] == 1 && !visited[i][j]) {
                dfsIsland(grid, i, j, visited);
                count++;
            }
        }
    }

    return count;
}

// =============================================================================
// 6. Union-Find Application
// =============================================================================

class UnionFind {
private:
    vector<int> parent, rank_;

public:
    UnionFind(int n) : parent(n), rank_(n, 0) {
        iota(parent.begin(), parent.end(), 0);
    }

    int find(int x) {
        if (parent[x] != x) {
            parent[x] = find(parent[x]);
        }
        return parent[x];
    }

    bool unite(int x, int y) {
        int px = find(x), py = find(y);
        if (px == py) return false;

        if (rank_[px] < rank_[py]) swap(px, py);
        parent[py] = px;
        if (rank_[px] == rank_[py]) rank_[px]++;

        return true;
    }
};

// Find redundant connection
vector<int> findRedundantConnection(vector<vector<int>>& edges) {
    int n = edges.size();
    UnionFind uf(n + 1);

    for (const auto& edge : edges) {
        if (!uf.unite(edge[0], edge[1])) {
            return edge;
        }
    }

    return {};
}

// =============================================================================
// 7. LIS (Longest Increasing Subsequence)
// =============================================================================

int lengthOfLIS(const vector<int>& nums) {
    vector<int> tails;

    for (int num : nums) {
        auto it = lower_bound(tails.begin(), tails.end(), num);
        if (it == tails.end()) {
            tails.push_back(num);
        } else {
            *it = num;
        }
    }

    return tails.size();
}

// =============================================================================
// 8. Binary Search Application (Parametric Search)
// =============================================================================

bool canShip(const vector<int>& weights, int capacity, int days) {
    int current = 0;
    int dayCount = 1;

    for (int w : weights) {
        if (w > capacity) return false;
        if (current + w > capacity) {
            dayCount++;
            current = w;
        } else {
            current += w;
        }
    }

    return dayCount <= days;
}

int shipWithinDays(const vector<int>& weights, int days) {
    int lo = *max_element(weights.begin(), weights.end());
    int hi = accumulate(weights.begin(), weights.end(), 0);

    while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (canShip(weights, mid, days)) {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }

    return lo;
}

// =============================================================================
// 9. Word Ladder (BFS)
// =============================================================================

int ladderLength(string beginWord, string endWord, vector<string>& wordList) {
    unordered_set<string> wordSet(wordList.begin(), wordList.end());
    if (wordSet.find(endWord) == wordSet.end()) return 0;

    queue<pair<string, int>> q;
    q.push({beginWord, 1});

    while (!q.empty()) {
        auto [word, dist] = q.front();
        q.pop();

        if (word == endWord) return dist;

        for (int i = 0; i < (int)word.length(); i++) {
            char original = word[i];
            for (char c = 'a'; c <= 'z'; c++) {
                word[i] = c;
                if (wordSet.find(word) != wordSet.end()) {
                    q.push({word, dist + 1});
                    wordSet.erase(word);
                }
            }
            word[i] = original;
        }
    }

    return 0;
}

// =============================================================================
// 10. Problem Solving Strategy
// =============================================================================

void printStrategy() {
    cout << "\n[10] Problem Solving Strategy" << endl;
    cout << "    1. Understand the problem: Check input/output, constraints" << endl;
    cout << "    2. Analyze examples: Solve by hand" << endl;
    cout << "    3. Choose algorithm:" << endl;
    cout << "       - N <= 20: Brute force, bitmask" << endl;
    cout << "       - N <= 10^3: O(N^2) DP, brute force" << endl;
    cout << "       - N <= 10^5: O(N log N) sorting, binary search" << endl;
    cout << "       - N <= 10^7: O(N) two pointers, hashing" << endl;
    cout << "    4. Implement and test" << endl;
    cout << "    5. Check edge cases" << endl;
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
    cout << "Practice Problems Example" << endl;
    cout << "============================================================" << endl;

    // 1. Subarray Sum
    cout << "\n[1] Subarray Sum (Two Pointers)" << endl;
    vector<int> arr1 = {2, 3, 1, 2, 4, 3};
    cout << "    Array: [2, 3, 1, 2, 4, 3], target = 7" << endl;
    cout << "    Min length: " << minSubarrayLen(7, arr1) << endl;

    // 2. Job Scheduling
    cout << "\n[2] Job Scheduling (Greedy)" << endl;
    vector<Job> jobs = {{4, 20}, {1, 10}, {1, 40}, {1, 30}};
    cout << "    Jobs: {deadline:4,profit:20}, {1,10}, {1,40}, {1,30}" << endl;
    cout << "    Max profit: " << jobScheduling(jobs) << endl;

    // 3. Minimum Meeting Rooms
    cout << "\n[3] Minimum Meeting Rooms" << endl;
    vector<pair<int, int>> meetings = {{0, 30}, {5, 10}, {15, 20}};
    cout << "    Meetings: [0-30], [5-10], [15-20]" << endl;
    cout << "    Min rooms: " << minMeetingRooms(meetings) << endl;

    // 4. Palindrome Conversion
    cout << "\n[4] Palindrome Conversion" << endl;
    cout << "    String: \"abcde\"" << endl;
    cout << "    Min insertions: " << minPalindromeInsertions("abcde") << endl;

    // 5. Number of Islands
    cout << "\n[5] Number of Islands" << endl;
    vector<vector<int>> grid = {
        {1, 1, 0, 0},
        {1, 0, 0, 0},
        {0, 0, 1, 0},
        {0, 0, 0, 1}
    };
    cout << "    Grid: 4x4" << endl;
    cout << "    Number of islands: " << numIslands(grid) << endl;

    // 6. Redundant Connection
    cout << "\n[6] Find Redundant Connection (Union-Find)" << endl;
    vector<vector<int>> edges = {{1, 2}, {1, 3}, {2, 3}};
    auto redundant = findRedundantConnection(edges);
    cout << "    Edges: (1,2), (1,3), (2,3)" << endl;
    cout << "    Redundant: (" << redundant[0] << ", " << redundant[1] << ")" << endl;

    // 7. LIS
    cout << "\n[7] Longest Increasing Subsequence (LIS)" << endl;
    vector<int> arr2 = {10, 9, 2, 5, 3, 7, 101, 18};
    cout << "    Array: [10, 9, 2, 5, 3, 7, 101, 18]" << endl;
    cout << "    LIS length: " << lengthOfLIS(arr2) << endl;

    // 8. Binary Search Application
    cout << "\n[8] Binary Search Application (Shipping)" << endl;
    vector<int> weights = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    cout << "    Package weights: [1-10]" << endl;
    cout << "    Min capacity for 5 days: " << shipWithinDays(weights, 5) << endl;

    // 9. Word Ladder
    cout << "\n[9] Word Ladder (BFS)" << endl;
    vector<string> wordList = {"hot", "dot", "dog", "lot", "log", "cog"};
    cout << "    hit -> cog" << endl;
    cout << "    Min transformations: " << ladderLength("hit", "cog", wordList) << endl;

    // 10. Problem Solving Strategy
    printStrategy();

    cout << "\n============================================================" << endl;

    return 0;
}
