/*
 * Practice Problems
 * Combined Problems (Various Algorithm Combinations)
 *
 * Common problem types frequently seen in coding tests.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <limits.h>

#define MAX_N 100001
#define INF INT_MAX

typedef long long ll;

/* =============================================================================
 * 1. Subarray Sum (Two Pointers)
 * ============================================================================= */

/* Minimum length subarray with sum >= target */
int min_subarray_sum(int arr[], int n, int target) {
    int left = 0;
    int sum = 0;
    int min_len = n + 1;

    for (int right = 0; right < n; right++) {
        sum += arr[right];

        while (sum >= target) {
            int len = right - left + 1;
            if (len < min_len) min_len = len;
            sum -= arr[left++];
        }
    }

    return (min_len == n + 1) ? 0 : min_len;
}

/* =============================================================================
 * 2. Job Scheduling (Greedy + Heap)
 * ============================================================================= */

typedef struct {
    int deadline;
    int profit;
} Job;

int compare_jobs(const void* a, const void* b) {
    return ((Job*)b)->profit - ((Job*)a)->profit;
}

int job_scheduling(Job jobs[], int n) {
    qsort(jobs, n, sizeof(Job), compare_jobs);

    int max_deadline = 0;
    for (int i = 0; i < n; i++) {
        if (jobs[i].deadline > max_deadline)
            max_deadline = jobs[i].deadline;
    }

    int* slot = malloc((max_deadline + 1) * sizeof(int));
    memset(slot, -1, (max_deadline + 1) * sizeof(int));

    int total_profit = 0;
    int job_count = 0;

    for (int i = 0; i < n; i++) {
        for (int j = jobs[i].deadline; j >= 1; j--) {
            if (slot[j] == -1) {
                slot[j] = i;
                total_profit += jobs[i].profit;
                job_count++;
                break;
            }
        }
    }

    free(slot);
    return total_profit;
}

/* =============================================================================
 * 3. Minimum Meeting Rooms (Event Sorting)
 * ============================================================================= */

typedef struct {
    int time;
    int type;  /* 1: start, -1: end */
} Event;

int compare_events(const void* a, const void* b) {
    Event* e1 = (Event*)a;
    Event* e2 = (Event*)b;
    if (e1->time != e2->time)
        return e1->time - e2->time;
    return e1->type - e2->type;  /* End events first */
}

int min_meeting_rooms(int start[], int end[], int n) {
    Event* events = malloc(2 * n * sizeof(Event));

    for (int i = 0; i < n; i++) {
        events[2 * i] = (Event){start[i], 1};
        events[2 * i + 1] = (Event){end[i], -1};
    }

    qsort(events, 2 * n, sizeof(Event), compare_events);

    int rooms = 0, max_rooms = 0;
    for (int i = 0; i < 2 * n; i++) {
        rooms += events[i].type;
        if (rooms > max_rooms) max_rooms = rooms;
    }

    free(events);
    return max_rooms;
}

/* =============================================================================
 * 4. Palindrome Conversion (DP)
 * ============================================================================= */

int min_palindrome_insertions(char* s) {
    int n = strlen(s);
    int** dp = malloc(n * sizeof(int*));
    for (int i = 0; i < n; i++) {
        dp[i] = calloc(n, sizeof(int));
    }

    for (int len = 2; len <= n; len++) {
        for (int i = 0; i + len - 1 < n; i++) {
            int j = i + len - 1;
            if (s[i] == s[j]) {
                dp[i][j] = dp[i + 1][j - 1];
            } else {
                int v1 = dp[i + 1][j];
                int v2 = dp[i][j - 1];
                dp[i][j] = 1 + (v1 < v2 ? v1 : v2);
            }
        }
    }

    int result = dp[0][n - 1];
    for (int i = 0; i < n; i++) free(dp[i]);
    free(dp);
    return result;
}

/* =============================================================================
 * 5. Number of Islands (DFS/BFS)
 * ============================================================================= */

int dx[] = {-1, 1, 0, 0};
int dy[] = {0, 0, -1, 1};

void dfs_island(int** grid, int rows, int cols, int i, int j, bool** visited) {
    if (i < 0 || i >= rows || j < 0 || j >= cols) return;
    if (visited[i][j] || grid[i][j] == 0) return;

    visited[i][j] = true;
    for (int d = 0; d < 4; d++) {
        dfs_island(grid, rows, cols, i + dx[d], j + dy[d], visited);
    }
}

int count_islands(int** grid, int rows, int cols) {
    bool** visited = malloc(rows * sizeof(bool*));
    for (int i = 0; i < rows; i++) {
        visited[i] = calloc(cols, sizeof(bool));
    }

    int count = 0;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (grid[i][j] == 1 && !visited[i][j]) {
                dfs_island(grid, rows, cols, i, j, visited);
                count++;
            }
        }
    }

    for (int i = 0; i < rows; i++) free(visited[i]);
    free(visited);
    return count;
}

/* =============================================================================
 * 6. Union-Find Application (Redundant Connection)
 * ============================================================================= */

int uf_parent[MAX_N];
int uf_rank_arr[MAX_N];

void uf_init(int n) {
    for (int i = 0; i < n; i++) {
        uf_parent[i] = i;
        uf_rank_arr[i] = 0;
    }
}

int uf_find(int x) {
    if (uf_parent[x] != x)
        uf_parent[x] = uf_find(uf_parent[x]);
    return uf_parent[x];
}

bool uf_union(int x, int y) {
    int px = uf_find(x);
    int py = uf_find(y);
    if (px == py) return false;

    if (uf_rank_arr[px] < uf_rank_arr[py]) {
        uf_parent[px] = py;
    } else if (uf_rank_arr[px] > uf_rank_arr[py]) {
        uf_parent[py] = px;
    } else {
        uf_parent[py] = px;
        uf_rank_arr[px]++;
    }
    return true;
}

/* Find redundant connection */
int* find_redundant_connection(int edges[][2], int n) {
    uf_init(n + 1);
    static int result[2];

    for (int i = 0; i < n; i++) {
        if (!uf_union(edges[i][0], edges[i][1])) {
            result[0] = edges[i][0];
            result[1] = edges[i][1];
            return result;
        }
    }

    result[0] = -1;
    result[1] = -1;
    return result;
}

/* =============================================================================
 * 7. LIS (Longest Increasing Subsequence)
 * ============================================================================= */

int lower_bound(int arr[], int n, int target) {
    int lo = 0, hi = n;
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (arr[mid] < target) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

int lis_length(int arr[], int n) {
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
 * 8. Sudoku (Backtracking)
 * ============================================================================= */

bool is_valid_sudoku(int board[9][9], int row, int col, int num) {
    /* Check row */
    for (int j = 0; j < 9; j++) {
        if (board[row][j] == num) return false;
    }

    /* Check column */
    for (int i = 0; i < 9; i++) {
        if (board[i][col] == num) return false;
    }

    /* Check 3x3 box */
    int box_row = (row / 3) * 3;
    int box_col = (col / 3) * 3;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            if (board[box_row + i][box_col + j] == num) return false;
        }
    }

    return true;
}

bool solve_sudoku(int board[9][9]) {
    for (int i = 0; i < 9; i++) {
        for (int j = 0; j < 9; j++) {
            if (board[i][j] == 0) {
                for (int num = 1; num <= 9; num++) {
                    if (is_valid_sudoku(board, i, j, num)) {
                        board[i][j] = num;
                        if (solve_sudoku(board)) return true;
                        board[i][j] = 0;
                    }
                }
                return false;
            }
        }
    }
    return true;
}

/* =============================================================================
 * 9. Binary Search Application (Parametric Search)
 * ============================================================================= */

/* Find minimum capacity to ship within given days */
bool can_ship(int weights[], int n, int capacity, int days) {
    int current = 0;
    int day_count = 1;

    for (int i = 0; i < n; i++) {
        if (weights[i] > capacity) return false;
        if (current + weights[i] > capacity) {
            day_count++;
            current = weights[i];
        } else {
            current += weights[i];
        }
    }

    return day_count <= days;
}

int ship_within_days(int weights[], int n, int days) {
    int max_weight = 0, total = 0;
    for (int i = 0; i < n; i++) {
        if (weights[i] > max_weight) max_weight = weights[i];
        total += weights[i];
    }

    int lo = max_weight, hi = total;
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (can_ship(weights, n, mid, days)) {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }

    return lo;
}

/* =============================================================================
 * Test
 * ============================================================================= */

int main(void) {
    printf("============================================================\n");
    printf("Practice Problem Examples\n");
    printf("============================================================\n");

    /* 1. Subarray Sum */
    printf("\n[1] Subarray Sum (Two Pointers)\n");
    int arr1[] = {2, 3, 1, 2, 4, 3};
    printf("    Array: [2, 3, 1, 2, 4, 3], target = 7\n");
    printf("    Minimum length: %d\n", min_subarray_sum(arr1, 6, 7));

    /* 2. Job Scheduling */
    printf("\n[2] Job Scheduling (Greedy)\n");
    Job jobs[] = {{4, 20}, {1, 10}, {1, 40}, {1, 30}};
    printf("    Jobs: {deadline:4,profit:20}, {1,10}, {1,40}, {1,30}\n");
    printf("    Maximum profit: %d\n", job_scheduling(jobs, 4));

    /* 3. Minimum Meeting Rooms */
    printf("\n[3] Minimum Meeting Rooms\n");
    int start[] = {0, 5, 15};
    int end[] = {30, 10, 20};
    printf("    Meetings: [0-30], [5-10], [15-20]\n");
    printf("    Minimum rooms: %d\n", min_meeting_rooms(start, end, 3));

    /* 4. Palindrome Conversion */
    printf("\n[4] Palindrome Conversion\n");
    printf("    String: \"abcde\"\n");
    printf("    Minimum insertions: %d\n", min_palindrome_insertions("abcde"));

    /* 5. Number of Islands */
    printf("\n[5] Number of Islands\n");
    int grid_data[4][4] = {
        {1, 1, 0, 0},
        {1, 0, 0, 0},
        {0, 0, 1, 0},
        {0, 0, 0, 1}
    };
    int** grid = malloc(4 * sizeof(int*));
    for (int i = 0; i < 4; i++) {
        grid[i] = malloc(4 * sizeof(int));
        for (int j = 0; j < 4; j++) grid[i][j] = grid_data[i][j];
    }
    printf("    Grid:\n");
    for (int i = 0; i < 4; i++) {
        printf("      ");
        for (int j = 0; j < 4; j++) printf("%d ", grid[i][j]);
        printf("\n");
    }
    printf("    Number of islands: %d\n", count_islands(grid, 4, 4));
    for (int i = 0; i < 4; i++) free(grid[i]);
    free(grid);

    /* 6. Redundant Connection */
    printf("\n[6] Find Redundant Connection (Union-Find)\n");
    int edges[][2] = {{1, 2}, {1, 3}, {2, 3}};
    int* redundant = find_redundant_connection(edges, 3);
    printf("    Edges: (1,2), (1,3), (2,3)\n");
    printf("    Redundant connection: (%d, %d)\n", redundant[0], redundant[1]);

    /* 7. LIS */
    printf("\n[7] Longest Increasing Subsequence (LIS)\n");
    int arr2[] = {10, 9, 2, 5, 3, 7, 101, 18};
    printf("    Array: [10, 9, 2, 5, 3, 7, 101, 18]\n");
    printf("    LIS length: %d\n", lis_length(arr2, 8));

    /* 8. Binary Search Application */
    printf("\n[8] Binary Search Application (Shipping)\n");
    int weights[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    printf("    Item weights: [1-10]\n");
    printf("    Min capacity for 5-day shipping: %d\n", ship_within_days(weights, 10, 5));

    /* 9. Problem-Solving Strategy */
    printf("\n[9] Problem-Solving Strategy\n");
    printf("    1. Understand the problem: check input/output, constraints\n");
    printf("    2. Analyze examples: work through by hand\n");
    printf("    3. Choose algorithm:\n");
    printf("       - N <= 20: brute force, bitmask\n");
    printf("       - N <= 10^3: O(N^2) DP, brute force\n");
    printf("       - N <= 10^5: O(N log N) sorting, binary search\n");
    printf("       - N <= 10^7: O(N) two pointers, hash\n");
    printf("    4. Implement and test\n");
    printf("    5. Check edge cases\n");

    printf("\n============================================================\n");

    return 0;
}
