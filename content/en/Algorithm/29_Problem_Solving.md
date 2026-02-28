# Problem Solving in Practice

## Learning Objectives

After completing this lesson, you will be able to:

1. Apply a structured 5-step problem solving process — understand, analyze examples, choose algorithm, implement, verify — to unfamiliar algorithmic problems
2. Estimate the required time complexity from the input size constraints and use this to eliminate unsuitable algorithms before coding
3. Recognize problem types from keywords and structural patterns (e.g., shortest path, subset enumeration, interval queries) and map them to appropriate algorithms
4. Select and execute problem-type-specific strategies for varying difficulty levels under timed conditions
5. Manage time effectively during a coding test or competition by prioritizing problems and allocating effort appropriately
6. Apply structured communication and coding techniques during coding interviews to demonstrate clear thinking and produce correct solutions

---

## Overview

Covers practical problem solving strategies and type-specific approaches for coding tests and algorithm competitions.

---

## Table of Contents

1. [Problem Solving Process](#1-problem-solving-process)
2. [Type Recognition](#2-type-recognition)
3. [Difficulty-based Strategy](#3-difficulty-based-strategy)
4. [Core Problems by Type](#4-core-problems-by-type)
5. [Time Management Strategy](#5-time-management-strategy)
6. [Coding Interview Tips](#6-coding-interview-tips)

---

## 1. Problem Solving Process

### 1.1 5-Step Approach

```
┌─────────────────────────────────────────────────────────┐
│                5-Step Problem Solving                    │
├─────────────────────────────────────────────────────────┤
│  1. Understand      → Identify input/output/constraints │
│  2. Analyze Examples→ Solve by hand, find patterns      │
│  3. Choose Algorithm→ Identify type, verify complexity  │
│  4. Implement       → Write code, handle edge cases     │
│  5. Verify          → Test cases, debug                 │
└─────────────────────────────────────────────────────────┘
```

### 1.2 Time Complexity Calculation

```
Allowed complexity by input size N (1 second limit):

┌─────────────┬───────────────────┬─────────────────┐
│ Input Size  │ Max Complexity    │ Suitable Algo   │
├─────────────┼───────────────────┼─────────────────┤
│ N ≤ 10      │ O(N!)             │ Brute force, backtracking│
│ N ≤ 20      │ O(2^N)            │ Bitmask, backtracking│
│ N ≤ 500     │ O(N³)             │ Floyd-Warshall  │
│ N ≤ 5,000   │ O(N²)             │ DP, brute force │
│ N ≤ 100,000 │ O(N log N)        │ Sorting, binary search│
│ N ≤ 10^7    │ O(N)              │ Two pointers, hash│
│ N ≤ 10^18   │ O(log N)          │ Binary search, math│
└─────────────┴───────────────────┴─────────────────┘
```

### 1.3 Problem Reading Checklist

```python
# Problem Analysis Template
def analyze_problem():
    """
    Checklist:
    [ ] Check input range (max N, M)
    [ ] Check time limit (usually 1-2 seconds)
    [ ] Check memory limit (usually 256MB)
    [ ] Check special cases (0, 1, negative, empty input)
    [ ] Check output format (decimals, newlines, spaces)
    """
    pass
```

---

## 2. Type Recognition

### 2.1 Keyword-based Recognition

```
┌────────────────────────┬────────────────────────────────┐
│ Keyword                │ Algorithm                       │
├────────────────────────┼────────────────────────────────┤
│ Shortest distance, min cost│ BFS, Dijkstra, Floyd       │
│ Number of paths/ways   │ DP, combinatorics              │
│ Find max/min           │ Binary search, DP, greedy      │
│ Is it possible?        │ Binary search (parametric)     │
│ All cases, order       │ Backtracking, permutation      │
│ Connected, group       │ Union-Find, DFS/BFS            │
│ Range sum, cumulative  │ Prefix sum, segment tree       │
│ Consecutive subarray   │ Sliding window, two pointers   │
│ String matching        │ KMP, hash, trie                │
└────────────────────────┴────────────────────────────────┘
```

### 2.2 Data Structure Selection Guide

```
┌────────────────────────┬────────────────────────────────┐
│ Required Operation     │ Data Structure                  │
├────────────────────────┼────────────────────────────────┤
│ Fast insert/delete (front/back)│ Deque                  │
│ Fast search (key-value)│ HashMap/Dictionary             │
│ Maintain sorted order  │ TreeMap, heap                  │
│ Fast access to max/min │ Heap (Priority Queue)          │
│ Remove duplicates      │ Set, hash set                  │
│ Ordered unique values  │ OrderedDict, TreeSet           │
│ Range queries          │ Segment tree, Fenwick tree     │
└────────────────────────┴────────────────────────────────┘
```

### 2.3 Problem Type Decision Tree

```
                Problem Start
                        │
           ┌────────────┴────────────┐
           │ Optimization problem?   │
           └────────────┬────────────┘
                 ┌──────┴──────┐
                YES            NO
                 │              │
    ┌────────────┴───┐    ┌────┴────┐
    │ Greedy works?  │    │ Search/enumerate│
    └────────────┬───┘    └────┬────┘
         ┌───────┴───────┐     │
        YES              NO    │
         │                │     │
      Greedy            DP    ┌┴─────────┐
                              │ All cases?│
                              └┬─────────┘
                        ┌──────┴──────┐
                       YES            NO
                        │              │
                   Backtracking    Graph search
                   Brute force     (DFS/BFS)
```

---

## 3. Difficulty-based Strategy

### 3.1 Easy (Bronze~Silver)

```
Key Points:
✓ Implement problem as described
✓ Use basic data structures
✓ Time complexity not critical

Main Types:
- Simple implementation/simulation
- Basic sorting/searching
- 1D DP
- Basic graph traversal

Example Approach:
1. Translate problem conditions directly to code
2. Verify with example cases
3. Test edge cases (0, 1, max value)
```

```python
# Easy problem template - Two Sum
def two_sum_easy(nums, target):
    """
    Brute force sufficient (N ≤ 1000)
    O(N²) allowed
    """
    n = len(nums)
    for i in range(n):
        for j in range(i + 1, n):
            if nums[i] + nums[j] == target:
                return [i, j]
    return []
```

### 3.2 Medium (Gold)

```
Key Points:
✓ Algorithm selection important
✓ Time complexity verification required
✓ Apply optimization techniques

Main Types:
- Binary search applications
- Graph algorithms (Dijkstra, MST)
- 2D DP
- Two pointers/sliding window
- Tree DP

Example Approach:
1. Identify type → Choose algorithm
2. Calculate complexity → Check feasibility
3. Implement → Optimize
```

```python
# Medium problem template - Two Sum (optimized)
def two_sum_medium(nums, target):
    """
    Optimize with hash map (N ≤ 100,000)
    O(N) needed
    """
    seen = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    return []
```

### 3.3 Hard (Platinum+)

```
Key Points:
✓ Combine multiple algorithms
✓ Advanced data structures needed
✓ Creative approach required

Main Types:
- Segment tree/Fenwick tree
- Advanced graph (SCC, 2-SAT)
- Bitmask DP
- Convex hull, geometry
- Advanced string (suffix array, Manacher)

Example Approach:
1. Decompose problem → Define subproblems
2. Check if known algorithms apply
3. Observe → Derive optimization ideas
```

---

## 4. Core Problems by Type

### 4.1 Array/String

```python
# Type 1: Sliding window - Maximum subarray sum
def max_subarray_sum(arr, k):
    """
    Maximum sum of subarray of size k
    Time: O(N)
    """
    n = len(arr)
    if n < k:
        return 0

    window_sum = sum(arr[:k])
    max_sum = window_sum

    for i in range(k, n):
        window_sum += arr[i] - arr[i - k]
        max_sum = max(max_sum, window_sum)

    return max_sum

# Type 2: Two pointers - Two sum in sorted array
def two_sum_sorted(arr, target):
    """
    Find two numbers that sum to target in sorted array
    Time: O(N)
    """
    left, right = 0, len(arr) - 1

    while left < right:
        current = arr[left] + arr[right]
        if current == target:
            return [left, right]
        elif current < target:
            left += 1
        else:
            right -= 1

    return []

# Type 3: Prefix sum - Range sum queries
class PrefixSum:
    def __init__(self, arr):
        self.prefix = [0]
        for x in arr:
            self.prefix.append(self.prefix[-1] + x)

    def query(self, l, r):
        """Sum of range [l, r] (0-indexed)"""
        return self.prefix[r + 1] - self.prefix[l]
```

### 4.2 Graph

```python
from collections import deque
import heapq

# Type 1: BFS - Shortest distance (unweighted)
def bfs_shortest(graph, start, end):
    """
    Shortest distance in unweighted graph
    Time: O(V + E)
    BFS is correct for unweighted shortest paths because it expands nodes
    in order of hop count — the first time a node is reached, it is via the shortest path.
    """
    n = len(graph)
    dist = [-1] * n
    dist[start] = 0

    queue = deque([start])

    while queue:
        curr = queue.popleft()

        if curr == end:
            return dist[end]

        for next_node in graph[curr]:
            if dist[next_node] == -1:  # Not yet visited — first reach = shortest path
                dist[next_node] = dist[curr] + 1
                queue.append(next_node)

    return -1  # end is unreachable from start

# Type 2: Dijkstra - Shortest distance (weighted)
def dijkstra(graph, start):
    """
    Shortest distance in weighted graph
    graph: adjacency list [(next, weight), ...]
    Time: O((V + E) log V)
    Dijkstra works by always extending the currently nearest unfinalized node —
    the greedy choice is safe because all edge weights are non-negative.
    """
    n = len(graph)
    dist = [float('inf')] * n
    dist[start] = 0

    pq = [(0, start)]  # Min-heap on (distance, node) — processes nearest node first

    while pq:
        d, curr = heapq.heappop(pq)

        # Skip stale heap entries — a node may be inserted multiple times with
        # different distances; only the smallest distance (first pop) is final
        if d > dist[curr]:
            continue

        for next_node, weight in graph[curr]:
            new_dist = dist[curr] + weight
            if new_dist < dist[next_node]:
                dist[next_node] = new_dist
                heapq.heappush(pq, (new_dist, next_node))

    return dist

# Type 3: Union-Find - Connected components
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))  # Each node starts as its own root
        self.rank = [0] * n           # Rank guides union direction to keep trees flat

    def find(self, x):
        if self.parent[x] != x:
            # Path compression: make every node on the path point directly to the root,
            # so subsequent finds are nearly O(1) — this is the key to amortized efficiency
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px == py:
            return False  # Already in the same component — no merge needed
        # Union by rank: attach shorter tree under taller to keep height O(log n)
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1  # Height increases only when merging equal-rank trees
        return True
```

### 4.3 Dynamic Programming

```python
# Type 1: 1D DP - Climbing stairs
def climb_stairs(n):
    """
    Number of ways to climb n stairs (1 or 2 steps at a time)
    Time: O(N), Space: O(1)
    Same recurrence as Fibonacci: ways(n) = ways(n-1) + ways(n-2)
    because the last step is either 1 or 2 stairs.
    """
    if n <= 2:
        return n

    prev2, prev1 = 1, 2
    for _ in range(3, n + 1):
        curr = prev1 + prev2
        prev2, prev1 = prev1, curr  # Slide the window — O(1) space instead of O(N) array

    return prev1

# Type 2: 2D DP - 0/1 Knapsack
def knapsack_01(weights, values, capacity):
    """
    Maximum value within capacity constraint
    Time: O(N * W), Space: O(W)
    """
    n = len(weights)
    dp = [0] * (capacity + 1)

    for i in range(n):
        # Iterate capacity in reverse so each item is considered at most once.
        # If we went forward, dp[w - weights[i]] might already include item i,
        # incorrectly allowing it to be picked multiple times (unbounded knapsack).
        for w in range(capacity, weights[i] - 1, -1):
            dp[w] = max(dp[w], dp[w - weights[i]] + values[i])

    return dp[capacity]

# Type 3: String DP - LCS
def lcs_length(s1, s2):
    """
    Longest common subsequence length
    Time: O(N * M)
    dp[i][j] = LCS length for s1[0:i] and s2[0:j].
    When characters match, we extend the LCS from the previous pair;
    when they don't, we take the best of dropping one character from either string.
    """
    n, m = len(s1), len(s2)
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1  # Characters match — extend LCS by 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])  # Take the best without one character

    return dp[n][m]

# Type 4: Interval DP - Matrix chain multiplication
def matrix_chain(dims):
    """
    Minimum operations for matrix multiplication
    dims: matrix dimensions [d0, d1, d2, ...] → (d0×d1) × (d1×d2) × ...
    Time: O(N³)
    Fill by increasing interval length so smaller subproblems are ready
    before the larger ones that depend on them.
    """
    n = len(dims) - 1
    dp = [[0] * n for _ in range(n)]

    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            dp[i][j] = float('inf')
            # Try every possible split point k; the cost of multiplying [i..k] × [k+1..j]
            # is the cost of each sub-chain plus dims[i]*dims[k+1]*dims[j+1] for the final multiply
            for k in range(i, j):
                cost = dp[i][k] + dp[k + 1][j] + dims[i] * dims[k + 1] * dims[j + 1]
                dp[i][j] = min(dp[i][j], cost)

    return dp[0][n - 1]
```

### 4.4 Binary Search

```python
# Type 1: Value search - lower_bound / upper_bound
def lower_bound(arr, target):
    """First position >= target"""
    left, right = 0, len(arr)
    while left < right:
        mid = (left + right) // 2
        if arr[mid] < target:
            left = mid + 1
        else:
            right = mid
    return left

def upper_bound(arr, target):
    """First position > target"""
    left, right = 0, len(arr)
    while left < right:
        mid = (left + right) // 2
        if arr[mid] <= target:
            left = mid + 1
        else:
            right = mid
    return left

# Type 2: Parametric search - Cutting trees
def cut_trees(heights, target):
    """
    Set cutter height to get at least target wood, find maximum height
    Time: O(N log max(H))
    """
    def can_get(cut_height):
        total = sum(max(0, h - cut_height) for h in heights)
        return total >= target

    left, right = 0, max(heights)
    result = 0

    while left <= right:
        mid = (left + right) // 2
        if can_get(mid):
            result = mid
            left = mid + 1
        else:
            right = mid - 1

    return result

# Type 3: Binary search + Greedy - Router installation
def install_routers(houses, n):
    """
    Install n routers to maximize minimum distance
    Time: O(N log D)
    """
    houses.sort()

    def can_install(min_dist):
        count = 1
        last = houses[0]
        for h in houses[1:]:
            if h - last >= min_dist:
                count += 1
                last = h
        return count >= n

    left, right = 1, houses[-1] - houses[0]
    result = 0

    while left <= right:
        mid = (left + right) // 2
        if can_install(mid):
            result = mid
            left = mid + 1
        else:
            right = mid - 1

    return result
```

### 4.5 Backtracking

```python
# Type 1: Permutation generation
def permutations(nums):
    """Generate all permutations - O(N! * N)"""
    result = []
    used = [False] * len(nums)  # Track which elements are already in the current path

    def backtrack(path):
        if len(path) == len(nums):
            result.append(path[:])  # Copy — path is mutated in-place, so we need a snapshot
            return

        for i, num in enumerate(nums):
            if used[i]:
                continue
            used[i] = True   # Choose: mark as used before descending
            path.append(num)
            backtrack(path)
            path.pop()       # Unchoose: undo the choice so the next iteration can try a different element
            used[i] = False

    backtrack([])
    return result

# Type 2: Combination generation
def combinations(nums, k):
    """Generate all combinations of size k - O(C(N,K) * K)"""
    result = []

    def backtrack(start, path):
        if len(path) == k:
            result.append(path[:])
            return

        for i in range(start, len(nums)):
            path.append(nums[i])
            backtrack(i + 1, path)  # start from i+1 to avoid reusing the same element
            path.pop()

    backtrack(0, [])
    return result

# Type 3: N-Queens
def solve_n_queens(n):
    """Number of solutions to N-Queens"""
    count = 0
    cols = [False] * n
    diag1 = [False] * (2 * n - 1)  # row - col + n - 1 (unique for each \ diagonal)
    diag2 = [False] * (2 * n - 1)  # row + col          (unique for each / diagonal)

    def backtrack(row):
        nonlocal count
        if row == n:
            count += 1  # All n queens placed without conflict
            return

        for col in range(n):
            d1 = row - col + n - 1
            d2 = row + col

            # Pruning: skip this column if any constraint is violated — this eliminates
            # entire subtrees and is why backtracking is far faster than brute force
            if cols[col] or diag1[d1] or diag2[d2]:
                continue

            cols[col] = diag1[d1] = diag2[d2] = True
            backtrack(row + 1)
            cols[col] = diag1[d1] = diag2[d2] = False  # Undo before trying next column

    backtrack(0)
    return count
```

---

## 5. Time Management Strategy

### 5.1 Problem Allocation Strategy

```
┌─────────────────────────────────────────────────────────┐
│        Coding Test Time Allocation (3 hour limit)       │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  [Survey]        15m   Read all problems, assess difficulty│
│       ↓                                                 │
│  [Easy]          45m   Solve definitely solvable problems│
│       ↓                                                 │
│  [Medium]        90m   Core problems, aim for partial score│
│       ↓                                                 │
│  [Hard]          20m   Implement ideas only             │
│       ↓                                                 │
│  [Review]        10m   Check runtime errors, edge cases │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### 5.2 Problem Priority

```
Priority determination criteria:

1. Score vs difficulty
   - Easy problems first (secure points)
   - Attempt hard problems if partial credit available

2. Type familiarity
   - Practiced types first
   - New types later

3. Time constraint
   - Time-consuming problems later
   - Be careful with heavy implementation
```

### 5.3 When Stuck

```
1. 5-minute rule
   - No progress in 5 minutes → switch problems

2. Simplify
   - Reduce input size and think
   - Solve special cases first

3. Think backwards
   - Work backwards from output
   - "What do I need to compute this?"

4. Find patterns
   - Trace examples by hand
   - Discover regularities

5. Partial credit
   - Solve small cases only
   - Submit even brute force
```

---

## 6. Coding Interview Tips

### 6.1 Interview Process

```
┌─────────────────────────────────────────────────────────┐
│                  Coding Interview Stages                 │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  1. Problem Description (5m)                            │
│     - Interviewer presents problem                     │
│     - Ask questions if unclear                         │
│                                                         │
│  2. Discuss Approach (10m)                             │
│     - Explain thinking out loud                        │
│     - Exchange ideas with interviewer                  │
│     - Mention time/space complexity                    │
│                                                         │
│  3. Coding (20-25m)                                    │
│     - Code while explaining                            │
│     - OK to ask for hints if stuck                     │
│                                                         │
│  4. Testing (5m)                                       │
│     - Hand trace with examples                         │
│     - Discuss edge cases                               │
│                                                         │
│  5. Optimization/Follow-up (5m)                        │
│     - Discuss improvement approaches                   │
│     - Handle variant problems                          │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### 6.2 Communication Strategy

```python
# Good example: Share thought process

"""
Interviewer: Find two numbers in array that sum to target.

Me: Let me understand the problem.
    - Given an array
    - Return two indices that sum to target
    - Is it sorted? → (question)

    First thinking brute force, that's O(N²).
    Check all pairs.

    More efficiently... using a hash map gets O(N).
    For each number, check if target - num is already in hash.

    Does this approach sound good? → (confirmation)
    Then I'll write the code.
"""
```

### 6.3 Frequently Asked Question Types

```
1. Two Sum variants
   - Sorted array → Two pointers
   - Three sum → Sort + two pointers
   - Closest sum → Sort + two pointers

2. Linked List
   - Cycle detection → Floyd's algorithm
   - Middle node → Fast/slow pointers
   - Reverse → Iterative or recursive

3. Tree
   - Traversal → Recursive/stack
   - Max depth → DFS
   - LCA → Recursion

4. Graph
   - Check connectivity → DFS/BFS
   - Shortest path → BFS
   - Cycle → DFS + visit states

5. Dynamic Programming
   - Climbing stairs → Fibonacci
   - Max subarray → Kadane's
   - Coin change → Unbounded knapsack
```

---

## Recommended Problems (By Platform)

### Baekjoon (BOJ)

| Difficulty | Problem | Type |
|-----------|---------|------|
| Silver | [Number Search (1920)](https://www.acmicpc.net/problem/1920) | Binary search |
| Silver | [DFS and BFS (1260)](https://www.acmicpc.net/problem/1260) | Graph traversal |
| Gold | [Shortest Path (1753)](https://www.acmicpc.net/problem/1753) | Dijkstra |
| Gold | [LCS (9251)](https://www.acmicpc.net/problem/9251) | DP |
| Gold | [N-Queen (9663)](https://www.acmicpc.net/problem/9663) | Backtracking |
| Platinum | [Find Minimum (11003)](https://www.acmicpc.net/problem/11003) | Monotonic deque |

### LeetCode

| Difficulty | Problem | Type |
|-----------|---------|------|
| Easy | [Two Sum](https://leetcode.com/problems/two-sum/) | Hash map |
| Easy | [Valid Parentheses](https://leetcode.com/problems/valid-parentheses/) | Stack |
| Medium | [3Sum](https://leetcode.com/problems/3sum/) | Two pointers |
| Medium | [Coin Change](https://leetcode.com/problems/coin-change/) | DP |
| Medium | [Number of Islands](https://leetcode.com/problems/number-of-islands/) | DFS/BFS |
| Hard | [Median of Two Sorted Arrays](https://leetcode.com/problems/median-of-two-sorted-arrays/) | Binary search |

### Programmers

| Level | Problem | Type |
|-------|---------|------|
| Lv2 | Target Number | DFS/BFS |
| Lv2 | Game Map Shortest Distance | BFS |
| Lv3 | Network | Union-Find |
| Lv3 | Way to School | DP |

---

## Learning Roadmap

```
┌─────────────────────────────────────────────────────────┐
│                   Skill Improvement Roadmap              │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  [Month 1] Build Foundation                             │
│    - Array, string, stack, queue                        │
│    - Basic sorting, binary search                       │
│    - 1 Easy problem per day                             │
│                                                         │
│  [Month 2] Core Algorithms                              │
│    - DFS, BFS, graph basics                             │
│    - 1D DP                                              │
│    - 1 Easy/Medium problem per day                      │
│                                                         │
│  [Month 3] Advanced Learning                            │
│    - Dijkstra, Union-Find                               │
│    - 2D DP, backtracking                                │
│    - 1-2 Medium problems per day                        │
│                                                         │
│  [Month 4+] Practical Practice                          │
│    - Mock tests (with time limit)                       │
│    - Challenge Hard problems                            │
│    - Strengthen weak areas                              │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## Practice Problems

### Comprehensive Problems

| No. | Problem | Difficulty | Hint |
|-----|---------|-----------|------|
| 1 | K-th largest number in array | ⭐⭐ | Heap or quickselect |
| 2 | Maze shortest distance | ⭐⭐ | BFS |
| 3 | Maximum consecutive subarray sum | ⭐⭐ | Kadane's Algorithm |
| 4 | Word conversion | ⭐⭐⭐ | BFS |
| 5 | Longest increasing subsequence | ⭐⭐⭐ | DP + binary search |

---

## References

- [Problem Solving Strategies (Korean)](https://book.algospot.com/)
- [LeetCode Patterns](https://seanprashad.com/leetcode-patterns/)
- [Codeforces](https://codeforces.com/) - Live contests
- [AtCoder](https://atcoder.jp/) - Japanese algorithm contests

---

## Checklist: Coding Test Preparation

```
Essential Types (Must be able to solve):
□ Binary search - lower_bound, parametric search
□ BFS - Shortest distance, level traversal
□ DFS - Connected components, cycles
□ DP - 1D, 2D basics
□ Greedy - Sort then select
□ Two pointers - Sum problems
□ Hash map - Frequency, duplicate check

Intermediate Types (Most tests include):
□ Dijkstra - Weighted shortest path
□ Union-Find - Grouping
□ Backtracking - Permutations, combinations
□ Sliding window - Consecutive intervals
□ Tree traversal - Pre/in/post-order

Advanced Types (Difficult tests):
□ Segment tree - Range queries
□ Topological sort - Dependency ordering
□ LCA - Common ancestor
□ Bitmask DP - State compression
```

---

## Previous Step

- [10_Heaps_and_Priority_Queues.md](./10_Heaps_and_Priority_Queues.md) - Heaps and priority queues

## Next Step

- [Heavy-Light Decomposition](./30_Heavy_Light_Decomposition.md) - Path queries on trees in O(log²N)
