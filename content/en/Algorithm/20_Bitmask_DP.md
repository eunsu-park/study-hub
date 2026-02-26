# Bitmask Dynamic Programming

## Learning Objectives

After completing this lesson, you will be able to:

1. Perform fundamental bit operations (AND, OR, XOR, shift, set, clear, toggle) and apply common bit manipulation tricks
2. Represent subsets of a small set as integer bitmasks and enumerate all subsets or submasks efficiently
3. Design Bitmask DP solutions by defining states that encode which elements have been used
4. Implement the Traveling Salesman Problem (TSP) using Bitmask DP with O(2^n × n) time complexity
5. Apply Bitmask DP to assignment and matching problems where subsets of items must be tracked across states

---

## Overview

Bitmask DP is a technique that represents set states as bits to perform dynamic programming. It efficiently solves subset problems when n is small (n <= 20).

---

## Table of Contents

1. [Bit Operation Basics](#1-bit-operation-basics)
2. [Subset Representation](#2-subset-representation)
3. [Bitmask DP Patterns](#3-bitmask-dp-patterns)
4. [Traveling Salesman Problem (TSP)](#4-traveling-salesman-problem-tsp)
5. [Application Problems](#5-application-problems)
6. [Practice Problems](#6-practice-problems)

---

## 1. Bit Operation Basics

### 1.1 Basic Bit Operations

```
AND (&): 1 only when both are 1
  1010 & 1100 = 1000

OR (|): 1 if either is 1
  1010 | 1100 = 1110

XOR (^): 1 if different
  1010 ^ 1100 = 0110

NOT (~): Bit inversion
  ~1010 = 0101 (Note: actually inverts all bits)

Left Shift (<<): Move left (× 2^n)
  1 << 3 = 8 (1000₂)

Right Shift (>>): Move right (÷ 2^n)
  8 >> 2 = 2 (10₂)
```

### 1.2 Useful Bit Tricks

```python
# 1. Check i-th bit (0-indexed)
def check_bit(mask, i):
    return (mask >> i) & 1
    # or: return mask & (1 << i) != 0

# 2. Set i-th bit
def set_bit(mask, i):
    return mask | (1 << i)

# 3. Clear i-th bit
def clear_bit(mask, i):
    return mask & ~(1 << i)

# 4. Toggle i-th bit
def toggle_bit(mask, i):
    return mask ^ (1 << i)

# 5. Count set bits (popcount)
def count_bits(mask):
    count = 0
    while mask:
        count += mask & 1
        mask >>= 1
    return count
# Python: bin(mask).count('1')
# C++: __builtin_popcount(mask)

# 6. Lowest set bit (LSB)
def lowest_bit(mask):
    return mask & -mask  # or mask & (~mask + 1)

# 7. Remove lowest set bit
def remove_lowest_bit(mask):
    return mask & (mask - 1)

# 8. Fill all bits with 1 (n bits)
def all_ones(n):
    return (1 << n) - 1

# Example
mask = 0b1010  # 10
print(check_bit(mask, 1))  # 1 (True)
print(check_bit(mask, 2))  # 0 (False)
print(bin(set_bit(mask, 0)))  # 0b1011
print(bin(clear_bit(mask, 3)))  # 0b10
```

### 1.3 C++ Bit Operations

```cpp
#include <bitset>

// Check i-th bit
bool checkBit(int mask, int i) {
    return (mask >> i) & 1;
}

// Count set bits
int countBits(int mask) {
    return __builtin_popcount(mask);
}
// long long: __builtin_popcountll(mask)

// Position of lowest set bit (0-indexed)
int lowestBitPos(int mask) {
    return __builtin_ctz(mask);  // count trailing zeros
}

// Position of highest set bit
int highestBitPos(int mask) {
    return 31 - __builtin_clz(mask);  // count leading zeros
}
```

---

## 2. Subset Representation

### 2.1 Set as Bitmask

```
Subsets of set S = {0, 1, 2, 3, 4}

Empty set:   00000 = 0
{0}:         00001 = 1
{1}:         00010 = 2
{0, 1}:      00011 = 3
{2}:         00100 = 4
{0, 2}:      00101 = 5
{0, 1, 2}:   00111 = 7
Full set:    11111 = 31

mask = 13 = 01101₂ → {0, 2, 3}
```

### 2.2 Iterate All Subsets

```python
def iterate_subsets(n):
    """Iterate all subsets of set of size n"""
    for mask in range(1 << n):  # 0 ~ 2^n - 1
        subset = []
        for i in range(n):
            if mask & (1 << i):
                subset.append(i)
        print(f"{mask:0{n}b}: {subset}")

iterate_subsets(3)
# 000: []
# 001: [0]
# 010: [1]
# 011: [0, 1]
# 100: [2]
# 101: [0, 2]
# 110: [1, 2]
# 111: [0, 1, 2]
```

### 2.3 Iterate Only Submasks of a Mask

```python
def iterate_submasks(mask):
    """
    Iterate all submasks of mask (excluding empty)
    Example: mask = 5 (101) → 5, 4, 1

    Why (submask - 1) & mask works:
    Subtracting 1 flips the lowest set bit of submask to 0 and turns all
    lower bits on.  ANDing with mask then clears any bits that were never
    set in the original mask, keeping us strictly within its subsets.
    This visits every submask in O(2^popcount(mask)) steps — far cheaper
    than iterating all 2^n masks and filtering.
    """
    submask = mask
    while submask > 0:
        print(bin(submask))
        submask = (submask - 1) & mask
    print("0 (empty)")

iterate_submasks(5)  # 101, 100, 001, 0
```

### 2.4 Set Operations

```python
def set_operations(a, b, n):
    """
    Set operations (n = full set size)
    """
    print(f"A = {bin(a)}, B = {bin(b)}")

    # Union
    union = a | b
    print(f"A ∪ B = {bin(union)}")

    # Intersection
    inter = a & b
    print(f"A ∩ B = {bin(inter)}")

    # Difference (A - B)
    diff = a & ~b
    print(f"A - B = {bin(diff)}")

    # Complement
    full = (1 << n) - 1
    comp_a = full ^ a
    print(f"A' = {bin(comp_a)}")

    # Symmetric difference (XOR)
    sym_diff = a ^ b
    print(f"A △ B = {bin(sym_diff)}")

# Example
set_operations(0b1010, 0b1100, 4)
```

---

## 3. Bitmask DP Patterns

### 3.1 Basic Pattern

```python
def bitmask_dp_template(n, data):
    """
    Bitmask DP basic template
    State: dp[mask] = optimal value at mask state
    """
    INF = float('inf')
    dp = [INF] * (1 << n)
    dp[0] = 0  # Initial state

    for mask in range(1 << n):
        if dp[mask] == INF:
            continue

        for i in range(n):
            if mask & (1 << i):  # i already selected
                continue

            new_mask = mask | (1 << i)
            # State transition
            dp[new_mask] = min(dp[new_mask], dp[mask] + cost(mask, i))

    return dp[(1 << n) - 1]  # All elements selected
```

### 3.2 2D Bitmask DP

```python
def bitmask_dp_2d(n, start, data):
    """
    dp[mask][i] = optimal value at mask state with current position i
    Used for TSP etc.
    """
    INF = float('inf')
    dp = [[INF] * n for _ in range(1 << n)]
    dp[1 << start][start] = 0

    for mask in range(1 << n):
        for last in range(n):
            if dp[mask][last] == INF:
                continue
            if not (mask & (1 << last)):
                continue

            for next_node in range(n):
                if mask & (1 << next_node):
                    continue

                new_mask = mask | (1 << next_node)
                cost = data[last][next_node]
                dp[new_mask][next_node] = min(
                    dp[new_mask][next_node],
                    dp[mask][last] + cost
                )

    return dp
```

---

## 4. Traveling Salesman Problem (TSP)

### 4.1 Problem Definition

```
TSP (Traveling Salesman Problem):
- Visit all n cities and return to starting point with minimum cost
- Brute force: O(n!)
- Bitmask DP: O(n² × 2^n)

State:
dp[mask][i] = minimum cost when cities in mask are visited
              and currently at city i

Transition:
dp[mask | (1<<j)][j] = min(dp[mask][i] + dist[i][j])
(j not in mask, move from i to j)
```

### 4.2 Implementation

```python
def tsp(dist):
    """
    Traveling Salesman Problem - Bitmask DP
    dist[i][j] = cost from city i to j
    Time: O(n² × 2^n)
    Space: O(n × 2^n)

    Why bitmask: We need to track exactly which cities have been visited.
    A bitmask encodes this as a single integer (bit i = city i visited),
    enabling O(1) membership test and set operations.  The full state
    space is 2^n × n — exponential but tractable for n ≤ 20, whereas
    brute-force over all n! permutations is far worse.
    """
    n = len(dist)
    INF = float('inf')

    # dp[mask][i] = min cost to visit exactly the cities in mask, ending at city i
    dp = [[INF] * n for _ in range(1 << n)]
    dp[1][0] = 0  # Only city 0 visited (mask=0b1), cost 0

    for mask in range(1 << n):
        for last in range(n):
            if dp[mask][last] == INF:
                continue
            if not (mask & (1 << last)):
                # last must be in the visited set — skip inconsistent states
                continue

            for next_city in range(n):
                if mask & (1 << next_city):
                    continue  # Already visited — no revisiting allowed
                if dist[last][next_city] == INF:
                    continue

                new_mask = mask | (1 << next_city)
                dp[new_mask][next_city] = min(
                    dp[new_mask][next_city],
                    dp[mask][last] + dist[last][next_city]
                )

    # Return to starting point (0) after visiting all cities.
    # The round-trip cost adds the final leg back to city 0.
    full_mask = (1 << n) - 1
    result = INF
    for last in range(n):
        if dp[full_mask][last] != INF and dist[last][0] != INF:
            result = min(result, dp[full_mask][last] + dist[last][0])

    return result if result != INF else -1


# Example
dist = [
    [0, 10, 15, 20],
    [10, 0, 35, 25],
    [15, 35, 0, 30],
    [20, 25, 30, 0]
]
print(tsp(dist))  # 80: 0→1→3→2→0
```

### 4.3 Path Reconstruction

```python
def tsp_with_path(dist):
    """TSP + Path reconstruction"""
    n = len(dist)
    INF = float('inf')

    dp = [[INF] * n for _ in range(1 << n)]
    parent = [[-1] * n for _ in range(1 << n)]
    dp[1][0] = 0

    for mask in range(1 << n):
        for last in range(n):
            if dp[mask][last] == INF:
                continue

            for next_city in range(n):
                if mask & (1 << next_city):
                    continue
                if dist[last][next_city] == INF:
                    continue

                new_mask = mask | (1 << next_city)
                new_cost = dp[mask][last] + dist[last][next_city]

                if new_cost < dp[new_mask][next_city]:
                    dp[new_mask][next_city] = new_cost
                    parent[new_mask][next_city] = last

    # Find minimum cost
    full_mask = (1 << n) - 1
    min_cost = INF
    last_city = -1

    for i in range(n):
        if dp[full_mask][i] != INF and dist[i][0] != INF:
            total = dp[full_mask][i] + dist[i][0]
            if total < min_cost:
                min_cost = total
                last_city = i

    if last_city == -1:
        return -1, []

    # Reconstruct path
    path = [0]  # Return to 0 at end
    mask = full_mask
    curr = last_city

    while curr != -1:
        path.append(curr)
        prev = parent[mask][curr]
        mask ^= (1 << curr)
        curr = prev

    path.reverse()
    return min_cost, path

# Example
cost, path = tsp_with_path(dist)
print(f"Minimum cost: {cost}")  # 80
print(f"Path: {path}")  # [0, 1, 3, 2, 0]
```

### 4.4 C++ Implementation

```cpp
#include <vector>
#include <algorithm>
using namespace std;

const int INF = 1e9;

int tsp(vector<vector<int>>& dist) {
    int n = dist.size();
    vector<vector<int>> dp(1 << n, vector<int>(n, INF));
    dp[1][0] = 0;

    for (int mask = 1; mask < (1 << n); mask++) {
        for (int last = 0; last < n; last++) {
            if (dp[mask][last] == INF) continue;
            if (!(mask & (1 << last))) continue;

            for (int next = 0; next < n; next++) {
                if (mask & (1 << next)) continue;
                if (dist[last][next] == INF) continue;

                int newMask = mask | (1 << next);
                dp[newMask][next] = min(dp[newMask][next],
                                        dp[mask][last] + dist[last][next]);
            }
        }
    }

    int fullMask = (1 << n) - 1;
    int result = INF;

    for (int last = 0; last < n; last++) {
        if (dp[fullMask][last] != INF && dist[last][0] != INF) {
            result = min(result, dp[fullMask][last] + dist[last][0]);
        }
    }

    return result == INF ? -1 : result;
}
```

---

## 5. Application Problems

### 5.1 Set Cover Problem

```python
def min_set_cover(n, sets):
    """
    Minimum set cover: minimum subsets to cover all elements
    sets[i] = bitmask of elements in i-th set
    """
    m = len(sets)
    full = (1 << n) - 1

    # dp[mask] = minimum sets to cover mask
    INF = float('inf')
    dp = [INF] * (1 << n)
    dp[0] = 0

    for mask in range(1 << n):
        if dp[mask] == INF:
            continue

        for s in sets:
            new_mask = mask | s
            dp[new_mask] = min(dp[new_mask], dp[mask] + 1)

    return dp[full] if dp[full] != INF else -1

# Example
# n=4, sets: {0,1}, {1,2}, {2,3}, {0,3}
sets = [0b0011, 0b0110, 0b1100, 0b1001]
print(min_set_cover(4, sets))  # 2
```

### 5.2 Assignment Problem

```python
def min_assignment(cost):
    """
    Assign n tasks to n people
    cost[i][j] = cost for person i to do task j
    Each person does exactly one task
    """
    n = len(cost)
    INF = float('inf')

    # dp[mask] = min cost when tasks in mask are assigned
    # Person i = popcount of previous people
    dp = [INF] * (1 << n)
    dp[0] = 0

    for mask in range(1 << n):
        if dp[mask] == INF:
            continue

        person = bin(mask).count('1')  # Current person to assign
        if person >= n:
            continue

        for job in range(n):
            if mask & (1 << job):
                continue

            new_mask = mask | (1 << job)
            dp[new_mask] = min(dp[new_mask], dp[mask] + cost[person][job])

    return dp[(1 << n) - 1]

# Example
cost = [
    [9, 2, 7, 8],
    [6, 4, 3, 7],
    [5, 8, 1, 8],
    [7, 6, 9, 4]
]
print(min_assignment(cost))  # 13: (0→1, 1→0, 2→2, 3→3) or other optimal
```

### 5.3 Hamiltonian Path

```python
def count_hamiltonian_paths(n, adj):
    """
    Count Hamiltonian paths: paths visiting all vertices exactly once
    adj[i][j] = True if edge (i, j) exists
    """
    # dp[mask][i] = number of paths visiting mask and ending at i
    dp = [[0] * n for _ in range(1 << n)]

    # Set starting points
    for i in range(n):
        dp[1 << i][i] = 1

    for mask in range(1, 1 << n):
        for last in range(n):
            if not (mask & (1 << last)):
                continue
            if dp[mask][last] == 0:
                continue

            for next_node in range(n):
                if mask & (1 << next_node):
                    continue
                if not adj[last][next_node]:
                    continue

                new_mask = mask | (1 << next_node)
                dp[new_mask][next_node] += dp[mask][last]

    # Count paths visiting all vertices
    full = (1 << n) - 1
    return sum(dp[full][i] for i in range(n))

# Example: Complete graph
n = 4
adj = [[i != j for j in range(n)] for i in range(n)]
print(count_hamiltonian_paths(n, adj))  # 24 = 4!
```

### 5.4 Subset Sum

```python
def subset_sum_bitmask(arr, target):
    """
    Count subsets with sum equal to target
    """
    n = len(arr)
    count = 0

    for mask in range(1 << n):
        total = sum(arr[i] for i in range(n) if mask & (1 << i))
        if total == target:
            count += 1

    return count

# More efficient: Meet in the Middle (for large n)
def subset_sum_mitm(arr, target):
    """
    Meet in the Middle: O(2^(n/2))
    """
    n = len(arr)
    mid = n // 2

    # All subset sums of left half
    left_sums = []
    for mask in range(1 << mid):
        total = sum(arr[i] for i in range(mid) if mask & (1 << i))
        left_sums.append(total)

    # Sort
    left_sums.sort()

    # Right half + binary search
    from bisect import bisect_left, bisect_right
    count = 0

    for mask in range(1 << (n - mid)):
        total = sum(arr[mid + i] for i in range(n - mid) if mask & (1 << i))
        need = target - total
        # Count occurrences of need in left_sums
        count += bisect_right(left_sums, need) - bisect_left(left_sums, need)

    return count

# Example
arr = [1, 2, 3, 4, 5]
print(subset_sum_bitmask(arr, 10))  # 3: {1,4,5}, {2,3,5}, {1,2,3,4}
```

### 5.5 SOS DP (Sum over Subsets)

```python
def sos_dp(arr):
    """
    SOS DP: Calculate subset sum for each mask
    F[mask] = sum(A[i]) for all i that is subset of mask
    Time: O(n × 2^n)
    """
    n = len(arr).bit_length()
    N = 1 << n

    # Extend arr to size N
    F = arr + [0] * (N - len(arr))

    for i in range(n):
        for mask in range(N):
            if mask & (1 << i):
                F[mask] += F[mask ^ (1 << i)]

    return F

# Example
arr = [1, 2, 3, 4]  # Indices: 00, 01, 10, 11
result = sos_dp(arr)
# result[3] = arr[0] + arr[1] + arr[2] + arr[3] = 10 (all subsets of 11)
# result[2] = arr[0] + arr[2] = 4 (subsets of 10)
```

---

## 6. Practice Problems

### Recommended Problems

| Difficulty | Problem | Platform | Type |
|--------|------|--------|------|
| ⭐⭐⭐ | [Traveling Salesman](https://www.acmicpc.net/problem/2098) | BOJ | TSP |
| ⭐⭐⭐ | [Power Plant](https://www.acmicpc.net/problem/1102) | BOJ | Bitmask DP |
| ⭐⭐⭐ | [Shortest Hamilton Path](https://codeforces.com/problemset/problem/8/C) | CF | Hamilton path |
| ⭐⭐⭐ | [Partition Equal Subset Sum](https://leetcode.com/problems/partition-equal-subset-sum/) | LeetCode | Subset |
| ⭐⭐⭐⭐ | [Sticker](https://www.acmicpc.net/problem/1562) | BOJ | Bitmask DP |
| ⭐⭐⭐⭐ | [Can I Win](https://leetcode.com/problems/can-i-win/) | LeetCode | Game theory |

---

## Bitmask DP Checklist

```
□ Is n small enough? (n <= 20)
□ Can state be represented as a set?
□ Can recurrence relation be established?
□ Is memory sufficient? (2^n × factor)
□ Set base state
□ Check transition direction (small → large or large → small)
```

---

## Time/Space Complexity

```
┌──────────────────────┬─────────────────┬─────────────────┐
│ Problem Type         │ Time            │ Space           │
├──────────────────────┼─────────────────┼─────────────────┤
│ Basic Bitmask DP     │ O(n × 2^n)     │ O(2^n)          │
│ TSP                  │ O(n² × 2^n)    │ O(n × 2^n)      │
│ Assignment           │ O(n × 2^n)     │ O(2^n)          │
│ SOS DP               │ O(n × 2^n)     │ O(2^n)          │
│ Meet in the Middle   │ O(2^(n/2))     │ O(2^(n/2))      │
└──────────────────────┴─────────────────┴─────────────────┘
```

---

## Next Steps

- [21_Math_and_Number_Theory.md](./21_Math_and_Number_Theory.md) - Mathematics and Number Theory

---

## References

- [Bitmask DP](https://cp-algorithms.com/algebra/all-submasks.html)
- [SOS DP](https://codeforces.com/blog/entry/45223)
