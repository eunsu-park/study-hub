# Greedy Algorithms

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain the greedy choice property and optimal substructure as conditions for a greedy algorithm to yield a globally optimal solution
2. Identify problem types where greedy approaches succeed versus where they fail, using counterexamples
3. Implement greedy solutions for classic problems including activity selection, interval scheduling, and Huffman coding
4. Apply greedy techniques to solve task scheduling and resource allocation problems
5. Compare greedy algorithms with Dynamic Programming and determine which paradigm is more appropriate for a given problem

---

## Overview

Greedy algorithms make the locally optimal choice at each step. While they don't always guarantee an optimal solution, they can efficiently find optimal solutions for certain problems.

---

## Table of Contents

1. [Greedy Algorithm Concepts](#1-greedy-algorithm-concepts)
2. [Greedy Choice Property](#2-greedy-choice-property)
3. [Classic Problems](#3-classic-problems)
4. [Greedy vs DP](#4-greedy-vs-dp)
5. [Practice Problems](#5-practice-problems)

---

## 1. Greedy Algorithm Concepts

### Basic Principle

```
Greedy Algorithm:
1. Make the best choice in the current situation
2. Never reconsider the choice
3. Local optimum → Global optimum (not always)

Characteristics:
- Simple and intuitive
- Fast execution time
- Need to verify optimal solution conditions
```

### Coin Change Example

```
Coins: [500, 100, 50, 10], Amount: 1260 won

Greedy approach:
1. 500 won × 2 = 1000 won (remaining: 260 won)
2. 100 won × 2 = 200 won (remaining: 60 won)
3. 50 won × 1 = 50 won (remaining: 10 won)
4. 10 won × 1 = 10 won (remaining: 0 won)

Total coins: 6

In this case greedy is optimal!
(But if coins are [1, 3, 4] and amount is 6:
 Greedy: 4+1+1=3 coins, Optimal: 3+3=2 coins)
```

```python
def coin_change_greedy(coins, amount):
    coins.sort(reverse=True)  # Largest first
    count = 0

    for coin in coins:
        count += amount // coin
        amount %= coin

    return count if amount == 0 else -1
```

---

## 2. Greedy Choice Property

### Optimal Substructure

```
Optimal solution to problem contains optimal solutions to subproblems

Example: Shortest path
If A → B → C is shortest path
Then A → B is also shortest path
```

### Greedy Choice Property

```
A locally optimal choice leads to a globally optimal solution

Proof method:
1. Assume there's an optimal solution without the greedy choice
2. Show that swapping with greedy choice maintains optimality
3. Therefore, an optimal solution with greedy choice exists
```

---

## 3. Classic Problems

### 3.1 Activity Selection Problem

```
Problem: Perform maximum activities with one resource
         Each activity has start and end time

Activities: [(1,4), (3,5), (0,6), (5,7), (3,9), (5,9), (6,10), (8,11), (8,12), (2,14), (12,16)]

Strategy: Select by earliest end time

Sorted: [(1,4), (3,5), (0,6), (5,7), (3,9), (5,9), (6,10), (8,11), (8,12), (2,14), (12,16)]

Selection:
1. (1,4) selected ✓
2. (3,5) start(3) < end(4) → skip
3. (0,6) start(0) < end(4) → skip
4. (5,7) start(5) >= end(4) → selected ✓
5. ...
6. (8,11) selected ✓
7. (12,16) selected ✓

Result: 4 activities can be selected
```

```cpp
// C++
struct Activity {
    int start, end;
};

int maxActivities(vector<Activity>& activities) {
    // Sort by END time, not start time.
    // Finishing earliest leaves the most room for subsequent activities —
    // this is the greedy choice whose optimality is proven by the "exchange
    // argument": any solution that picks a later-ending activity first can
    // be swapped for the earliest-ending one without losing feasibility.
    sort(activities.begin(), activities.end(),
         [](const Activity& a, const Activity& b) {
             return a.end < b.end;
         });

    int count = 1;
    int lastEnd = activities[0].end;

    for (int i = 1; i < activities.size(); i++) {
        if (activities[i].start >= lastEnd) {
            // No overlap with the last selected activity: greedily take it
            count++;
            lastEnd = activities[i].end;
        }
    }

    return count;
}
```

```python
def max_activities(activities):
    # Sort by end time
    activities.sort(key=lambda x: x[1])

    count = 1
    last_end = activities[0][1]

    for start, end in activities[1:]:
        if start >= last_end:
            count += 1
            last_end = end

    return count
```

### 3.2 Meeting Room Scheduling

```
Problem: Hold maximum meetings in one meeting room

Meetings: [(1,4), (3,5), (0,6), (5,7), (3,8), (5,9), (6,10), (8,11)]

Sorted (by end time): [(1,4), (3,5), (0,6), (5,7), (3,8), (5,9), (6,10), (8,11)]

Selected: (1,4), (5,7), (8,11) → 3 meetings
```

```python
def max_meetings(meetings):
    meetings.sort(key=lambda x: x[1])

    count = 0
    last_end = 0

    for start, end in meetings:
        if start >= last_end:
            count += 1
            last_end = end

    return count
```

### 3.3 Fractional Knapsack

```
Problem: Maximum value when items can be divided
         (Different from 0/1 knapsack!)

Items: [(weight, value)] = [(10, 60), (20, 100), (30, 120)]
Capacity: W = 50

Value/weight ratio:
- Item 1: 60/10 = 6
- Item 2: 100/20 = 5
- Item 3: 120/30 = 4

Greedy selection (highest ratio first):
1. Item 1 full: 10kg, value 60
2. Item 2 full: 20kg, value 100
3. Item 3 partial: 20kg, value 120×(20/30) = 80

Total: 50kg, value 240
```

```cpp
// C++
struct Item {
    int weight, value;
    double ratio() const { return (double)value / weight; }
};

double fractionalKnapsack(int W, vector<Item>& items) {
    sort(items.begin(), items.end(),
         [](const Item& a, const Item& b) {
             return a.ratio() > b.ratio();
         });

    double totalValue = 0;
    int remainingWeight = W;

    for (const auto& item : items) {
        if (item.weight <= remainingWeight) {
            totalValue += item.value;
            remainingWeight -= item.weight;
        } else {
            totalValue += item.ratio() * remainingWeight;
            break;
        }
    }

    return totalValue;
}
```

```python
def fractional_knapsack(W, items):
    # Sort by value/weight ratio
    items.sort(key=lambda x: x[1]/x[0], reverse=True)

    total_value = 0

    for weight, value in items:
        if W >= weight:
            total_value += value
            W -= weight
        else:
            total_value += (value / weight) * W
            break

    return total_value
```

### 3.4 Minimum Meeting Rooms

```
Problem: Minimum meeting rooms needed for all meetings

Meetings: [(0,30), (5,10), (15,20)]

Timeline:
0----5---10---15---20---25---30
|=========== Meeting 1 ========|
     |===|
              |====|

Rooms needed:
- Time 0~5: 1
- Time 5~10: 2 (maximum!)
- Time 10~15: 1
- Time 15~20: 2
- Time 20~30: 1

Answer: 2
```

```cpp
// C++
int minMeetingRooms(vector<pair<int,int>>& meetings) {
    vector<int> starts, ends;

    for (const auto& m : meetings) {
        starts.push_back(m.first);
        ends.push_back(m.second);
    }

    sort(starts.begin(), starts.end());
    sort(ends.begin(), ends.end());

    int rooms = 0, maxRooms = 0;
    int i = 0, j = 0;

    while (i < starts.size()) {
        if (starts[i] < ends[j]) {
            rooms++;
            i++;
        } else {
            rooms--;
            j++;
        }
        maxRooms = max(maxRooms, rooms);
    }

    return maxRooms;
}
```

```python
def min_meeting_rooms(meetings):
    # Separate and sort starts and ends independently.
    # This lets us simulate a timeline sweep without building explicit events —
    # whenever the next start comes before the next end, a new room is needed;
    # otherwise an existing room becomes free.
    starts = sorted([m[0] for m in meetings])
    ends = sorted([m[1] for m in meetings])

    rooms = 0
    max_rooms = 0
    i = j = 0

    while i < len(starts):
        if starts[i] < ends[j]:
            rooms += 1   # A new meeting starts before the earliest-ending one finishes
            i += 1
        else:
            rooms -= 1   # The earliest-ending meeting has finished; reuse its room
            j += 1
        max_rooms = max(max_rooms, rooms)  # Track peak concurrent usage

    return max_rooms
```

### 3.5 Jump Game

```
Problem: Can jump up to value at each position
         Can we reach the last position?

Array: [2, 3, 1, 1, 4]

Position 0: max reach = 0 + 2 = 2
Position 1: max reach = max(2, 1 + 3) = 4
Position 2: max reach = max(4, 2 + 1) = 4
Position 3: max reach = max(4, 3 + 1) = 4
Position 4: reached! ✓
```

```cpp
// C++
bool canJump(vector<int>& nums) {
    int maxReach = 0;

    for (int i = 0; i < nums.size(); i++) {
        if (i > maxReach) return false;
        maxReach = max(maxReach, i + nums[i]);
        if (maxReach >= nums.size() - 1) return true;
    }

    return true;
}
```

```python
def can_jump(nums):
    max_reach = 0

    for i, jump in enumerate(nums):
        if i > max_reach:
            return False
        max_reach = max(max_reach, i + jump)
        if max_reach >= len(nums) - 1:
            return True

    return True
```

### 3.6 Minimum Jumps (Jump Game II)

```
Problem: Minimum jumps to reach last position

Array: [2, 3, 1, 1, 4]

Greedy approach:
- Within current range, select position that goes farthest on next jump

Jump 1: Position 0 → Position 1 (can reach up to 0+2=2, from 1 can reach 1+3=4)
Jump 2: Position 1 → Position 4 (arrived!)

Answer: 2
```

```python
def min_jumps(nums):
    if len(nums) <= 1:
        return 0

    jumps = 0
    current_end = 0   # Right boundary of the current jump's reachable range
    farthest = 0      # Farthest position reachable from anywhere in the current range

    # We iterate only up to len-2: reaching the last index IS the goal,
    # so we never need to count a jump FROM the last position.
    for i in range(len(nums) - 1):
        farthest = max(farthest, i + nums[i])

        if i == current_end:
            # We've exhausted all positions reachable with `jumps` jumps.
            # Must take one more jump to advance into [current_end+1, farthest].
            # Choosing `farthest` as the new boundary is the greedy insight:
            # we don't commit to a specific landing spot — we just note that
            # the next jump can reach as far as `farthest`.
            jumps += 1
            current_end = farthest

            if current_end >= len(nums) - 1:
                break

    return jumps
```

---

## 4. Greedy vs DP

### Comparison

```
┌─────────────────┬─────────────────┬─────────────────┐
│                 │ Greedy          │ DP              │
├─────────────────┼─────────────────┼─────────────────┤
│ Selection       │ Current optimal │ Consider all    │
│ Time complexity │ Usually O(n log n)│ Usually O(n²)  │
│ Space complexity│ O(1) possible   │ O(n) or more    │
│ Optimal solution│ Conditional     │ Guaranteed      │
│ Implementation  │ Easy            │ Medium          │
└─────────────────┴─────────────────┴─────────────────┘
```

### Problem-based Selection

```
Use Greedy:
- Activity selection problem
- Fractional knapsack
- Minimum spanning tree (Kruskal, Prim)
- Dijkstra shortest path
- Huffman coding

Use DP:
- 0/1 Knapsack problem
- Coin change (except special cases)
- Longest common subsequence
- Edit distance
```

### When Greedy Fails

```
Coin change:
Coins: [1, 3, 4], Amount: 6

Greedy: 4 + 1 + 1 = 3 coins
Optimal: 3 + 3 = 2 coins

→ Greedy fails! DP needed
```

---

## 5. Practice Problems

### Problem 1: Gas Station Circuit

Find starting point for circular gas station route.

```
gas = [1, 2, 3, 4, 5]
cost = [3, 4, 5, 1, 2]

Starting from station 3:
- 3→4: gas 4, cost 1, remaining 3
- 4→0: gas 3+5=8, cost 2, remaining 6
- 0→1: gas 6+1=7, cost 3, remaining 4
- 1→2: gas 4+2=6, cost 4, remaining 2
- 2→3: gas 2+3=5, cost 5, remaining 0

Answer: 3
```

<details>
<summary>Solution Code</summary>

```python
def can_complete_circuit(gas, cost):
    total_tank = 0
    current_tank = 0
    start = 0

    for i in range(len(gas)):
        diff = gas[i] - cost[i]
        total_tank += diff
        current_tank += diff

        if current_tank < 0:
            start = i + 1
            current_tank = 0

    return start if total_tank >= 0 else -1
```

</details>

### Recommended Problems

| Difficulty | Problem | Platform | Type |
|--------|------|--------|------|
| ⭐ | [Change](https://www.acmicpc.net/problem/5585) | BOJ | Coins |
| ⭐⭐ | [Meeting Room](https://www.acmicpc.net/problem/1931) | BOJ | Activity selection |
| ⭐⭐ | [Jump Game](https://leetcode.com/problems/jump-game/) | LeetCode | Jump |
| ⭐⭐ | [Gas Station](https://leetcode.com/problems/gas-station/) | LeetCode | Circuit |
| ⭐⭐⭐ | [Jump Game II](https://leetcode.com/problems/jump-game-ii/) | LeetCode | Min jumps |
| ⭐⭐⭐ | [Non-overlapping Intervals](https://leetcode.com/problems/non-overlapping-intervals/) | LeetCode | Intervals |

---

## Greedy Algorithm Checklist

```
□ Does the problem have optimal substructure?
□ Does greedy choice lead to optimal solution?
□ Are there counterexamples?
□ Is sorting needed? By what criteria?
□ Should DP be used instead?
```

---

## Next Steps

- [20_Bitmask_DP.md](./20_Bitmask_DP.md) - Bitmask DP

---

## References

- [Greedy Algorithms](https://www.geeksforgeeks.org/greedy-algorithms/)
- Introduction to Algorithms (CLRS) - Chapter 16
