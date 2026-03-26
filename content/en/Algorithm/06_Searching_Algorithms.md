# Searching Algorithms

## Learning Objectives

After completing this lesson, you will be able to:

1. Compare linear search, binary search, and hash search by their time complexity, preconditions, and appropriate use cases
2. Implement iterative and recursive binary search and explain why it achieves O(log n) time on sorted arrays
3. Apply binary search variants (lower bound, upper bound) to solve problems involving first/last occurrence of a target value
4. Design parametric search solutions by formulating an optimization problem as a binary-search-on-answer problem
5. Explain why hash-based lookup achieves O(1) average time and identify its limitations compared to sorted-array binary search
6. Select the most efficient search strategy given the data's structure (sorted vs. unsorted, range constraints, duplicates)

---

## Overview

Searching is a fundamental operation to find desired values in data. In this lesson, we learn various searching techniques including linear search, binary search, parametric search, and hash search.

---

## Table of Contents

1. [Comparing Search Algorithms](#1-comparing-search-algorithms)
2. [Linear Search](#2-linear-search)
3. [Binary Search](#3-binary-search)
4. [Binary Search Variants](#4-binary-search-variants)
5. [Parametric Search](#5-parametric-search)
6. [Hash Search](#6-hash-search)
7. [Practice Problems](#7-practice-problems)

---

## 1. Comparing Search Algorithms

```
┌──────────────┬─────────────┬─────────────┬───────────────────┐
│  Algorithm   │ Time        │ Condition   │ Features          │
├──────────────┼─────────────┼─────────────┼───────────────────┤
│ Linear       │ O(n)        │ None        │ Simple, general   │
│ Binary       │ O(log n)    │ Sorted      │ Fast, D&C         │
│ Hash         │ O(1) avg    │ Hash table  │ Fastest           │
│ Interpolation│ O(log log n)│ Even distri │ Special cases     │
└──────────────┴─────────────┴─────────────┴───────────────────┘
```

---

## 2. Linear Search

### Principle

```
Check sequentially from start to end

Array: [5, 3, 8, 4, 2], Target: 8

Index 0: 5 != 8 → Next
Index 1: 3 != 8 → Next
Index 2: 8 == 8 → Found! Return index 2
```

### Implementation

```c
// C
int linearSearch(int arr[], int n, int target) {
    for (int i = 0; i < n; i++) {
        if (arr[i] == target) {
            return i;
        }
    }
    return -1;  // Not found
}
```

```cpp
// C++
int linearSearch(const vector<int>& arr, int target) {
    for (int i = 0; i < arr.size(); i++) {
        if (arr[i] == target) {
            return i;
        }
    }
    return -1;
}

// STL
auto it = find(arr.begin(), arr.end(), target);
if (it != arr.end()) {
    int index = distance(arr.begin(), it);
}
```

```python
# Python
def linear_search(arr, target):
    for i, x in enumerate(arr):
        if x == target:
            return i
    return -1

# Built-in functions
# arr.index(target)  # Raises ValueError if not found
# target in arr      # Only checks existence
```

---

## 3. Binary Search

### Principle

```
In a sorted array, compare with middle value and halve search range

Array: [1, 3, 5, 7, 9, 11, 13], Target: 9

Step 1: left=0, right=6, mid=3
        arr[3]=7 < 9 → left=4

        [1, 3, 5, 7, 9, 11, 13]
                    ↑
                   left

Step 2: left=4, right=6, mid=5
        arr[5]=11 > 9 → right=4

        [1, 3, 5, 7, 9, 11, 13]
                    ↑
                left,right

Step 3: left=4, right=4, mid=4
        arr[4]=9 == 9 → Found! Return index 4
```

### Implementation

```c
// C - Iterative
int binarySearch(int arr[], int n, int target) {
    int left = 0;
    int right = n - 1;

    while (left <= right) {
        // left + (right - left) / 2 instead of (left + right) / 2:
        // (left + right) can overflow a 32-bit int when both are large
        // (e.g., left = right = 1.5 × 10⁹ → sum exceeds INT_MAX).
        // This form subtracts first and stays within bounds — essential in C/C++/Java
        // where integers have fixed width. Python's arbitrary-precision integers
        // make the naive form safe, but this style is considered best practice.
        int mid = left + (right - left) / 2;  // Prevent overflow

        if (arr[mid] == target) {
            return mid;
        } else if (arr[mid] < target) {
            left = mid + 1;   // Target is in the right half
        } else {
            right = mid - 1;  // Target is in the left half
        }
    }

    return -1;  // Exhausted all candidates — not present
}

// C - Recursive
int binarySearchRecursive(int arr[], int left, int right, int target) {
    // Base case: empty search range — the element does not exist
    if (left > right) {
        return -1;
    }

    int mid = left + (right - left) / 2;  // Same overflow-safe midpoint

    if (arr[mid] == target) {
        return mid;
    } else if (arr[mid] < target) {
        return binarySearchRecursive(arr, mid + 1, right, target);
    } else {
        return binarySearchRecursive(arr, left, mid - 1, target);
    }
}
```

```cpp
// C++
int binarySearch(const vector<int>& arr, int target) {
    int left = 0;
    int right = arr.size() - 1;

    while (left <= right) {
        int mid = left + (right - left) / 2;

        if (arr[mid] == target) {
            return mid;
        } else if (arr[mid] < target) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }

    return -1;
}

// STL
#include <algorithm>

// binary_search: returns only existence
bool found = binary_search(arr.begin(), arr.end(), target);

// lower_bound: first position >= target
auto it = lower_bound(arr.begin(), arr.end(), target);

// upper_bound: first position > target
auto it = upper_bound(arr.begin(), arr.end(), target);
```

```python
# Python
def binary_search(arr, target):
    left, right = 0, len(arr) - 1

    while left <= right:
        mid = (left + right) // 2

        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return -1

# bisect module
import bisect

# bisect_left: first position >= target
idx = bisect.bisect_left(arr, target)

# bisect_right: first position > target
idx = bisect.bisect_right(arr, target)
```

---

## 4. Binary Search Variants

### 4.1 Lower Bound (First Position >= Target)

```
Index of first element >= target

Array: [1, 2, 4, 4, 4, 6, 8], target=4

lower_bound(4) = 2  (index of first 4)
lower_bound(5) = 5  (index of 6, which is >= 5)
lower_bound(0) = 0  (all elements > 0)
lower_bound(9) = 7  (not found, end of array)
```

```c
// C
int lowerBound(int arr[], int n, int target) {
    int left = 0;
    // right = n (one past the last valid index), not n - 1:
    // the result may be n when target is greater than every element,
    // meaning "insert at the end" — using n - 1 would miss this case
    int right = n;

    // Loop condition is left < right (not <=): the range [left, right) is
    // half-open, so left == right means the range is empty and we're done
    while (left < right) {
        int mid = left + (right - left) / 2;

        if (arr[mid] < target) {
            left = mid + 1;  // mid is definitely not the answer — exclude it
        } else {
            // arr[mid] >= target: mid could be the answer, keep it in range
            right = mid;
        }
    }

    return left;  // left == right: the insertion point for target
}
```

```cpp
// C++
int lowerBound(const vector<int>& arr, int target) {
    int left = 0;
    int right = arr.size();

    while (left < right) {
        int mid = left + (right - left) / 2;

        if (arr[mid] < target) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }

    return left;
}
```

```python
# Python
def lower_bound(arr, target):
    left, right = 0, len(arr)

    while left < right:
        mid = (left + right) // 2

        if arr[mid] < target:
            left = mid + 1
        else:
            right = mid

    return left
```

### 4.2 Upper Bound (First Position > Target)

```
Index of first element > target

Array: [1, 2, 4, 4, 4, 6, 8], target=4

upper_bound(4) = 5  (index of first element > 4, which is 6)
upper_bound(5) = 5  (index of first element > 5, which is 6)

Count of specific value = upper_bound - lower_bound
Count of 4 = 5 - 2 = 3
```

```c
// C
int upperBound(int arr[], int n, int target) {
    int left = 0;
    int right = n;  // Same half-open [left, right) convention as lowerBound

    while (left < right) {
        int mid = left + (right - left) / 2;

        // The only difference from lowerBound: <= instead of <.
        // This causes equal elements to push left forward, so the result
        // lands one position *after* all occurrences of target
        if (arr[mid] <= target) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }

    return left;
}
```

```cpp
// C++
int upperBound(const vector<int>& arr, int target) {
    int left = 0;
    int right = arr.size();

    while (left < right) {
        int mid = left + (right - left) / 2;

        if (arr[mid] <= target) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }

    return left;
}
```

```python
# Python
def upper_bound(arr, target):
    left, right = 0, len(arr)

    while left < right:
        mid = (left + right) // 2

        if arr[mid] <= target:
            left = mid + 1
        else:
            right = mid

    return left
```

### 4.3 Search in Rotated Sorted Array

```
Search in a rotated sorted array

Original: [0, 1, 2, 4, 5, 6, 7]
Rotated:  [4, 5, 6, 7, 0, 1, 2]

Property: One side is always sorted
```

```cpp
// C++
int searchRotated(const vector<int>& arr, int target) {
    int left = 0;
    int right = arr.size() - 1;

    while (left <= right) {
        int mid = left + (right - left) / 2;  // Overflow-safe midpoint

        if (arr[mid] == target) {
            return mid;
        }

        // Key insight: in a rotated array, at least one of the two halves
        // is always contiguously sorted. Determine which half is sorted,
        // then check if target falls within that sorted range.

        // Left side is sorted: arr[left] <= arr[mid] means no rotation point here
        if (arr[left] <= arr[mid]) {
            if (arr[left] <= target && target < arr[mid]) {
                right = mid - 1;  // Target is within the sorted left half
            } else {
                left = mid + 1;   // Target must be in the (possibly rotated) right half
            }
        }
        // Right side is sorted: the rotation point is in the left half
        else {
            if (arr[mid] < target && target <= arr[right]) {
                left = mid + 1;   // Target is within the sorted right half
            } else {
                right = mid - 1;  // Target must be in the (rotated) left half
            }
        }
    }

    return -1;
}
```

```python
# Python
def search_rotated(arr, target):
    left, right = 0, len(arr) - 1

    while left <= right:
        mid = (left + right) // 2

        if arr[mid] == target:
            return mid

        # Left side is sorted
        if arr[left] <= arr[mid]:
            if arr[left] <= target < arr[mid]:
                right = mid - 1
            else:
                left = mid + 1
        # Right side is sorted
        else:
            if arr[mid] < target <= arr[right]:
                left = mid + 1
            else:
                right = mid - 1

    return -1
```

---

## 5. Parametric Search

### Concept

```
Transform optimization problem into decision problem and solve with binary search

"Find minimum value" → "Is it possible with ≤ x?"
"Find maximum value" → "Is it possible with ≥ x?"

Condition: Answer must have monotonicity
          - Possible → Possible → Possible → Impossible → Impossible (find boundary)
```

### Example 1: Cutting Cables

```
Problem: Given N cables, cut into K pieces of equal length,
         what's the maximum length?

Cable lengths: [802, 743, 457, 539], K=11

Length 100: 8+7+4+5 = 24 pieces ≥ 11 (possible)
Length 200: 4+3+2+2 = 11 pieces ≥ 11 (possible)
Length 201: 3+3+2+2 = 10 pieces < 11 (impossible)

→ Maximum length is 200
```

```cpp
// C++
bool canMake(const vector<int>& cables, int k, long long length) {
    long long count = 0;
    for (int cable : cables) {
        count += cable / length;  // Integer division gives how many pieces of this length
    }
    return count >= k;
}

long long maxCableLength(vector<int>& cables, int k) {
    long long left = 1;
    long long right = *max_element(cables.begin(), cables.end());
    long long answer = 0;  // 0 = "no valid answer found yet"

    while (left <= right) {
        long long mid = (left + right) / 2;

        if (canMake(cables, k, mid)) {
            // mid works — record it and search for a larger (better) answer
            answer = mid;
            left = mid + 1;  // Try longer length
        } else {
            // mid is too large — shrink the search space
            right = mid - 1;
        }
    }

    return answer;
}
```

```python
# Python
def can_make(cables, k, length):
    count = sum(cable // length for cable in cables)
    return count >= k

def max_cable_length(cables, k):
    left, right = 1, max(cables)
    answer = 0

    while left <= right:
        mid = (left + right) // 2

        if can_make(cables, k, mid):
            answer = mid
            left = mid + 1
        else:
            right = mid - 1

    return answer
```

### Example 2: Cutting Trees

```
Problem: Given N trees, cut at height H,
         what's the maximum H to get ≥ M meters?

Tree heights: [20, 15, 10, 17], M=7

H=15: 5+0+0+2 = 7m (possible)
H=16: 4+0+0+1 = 5m < 7 (impossible)

→ Maximum H is 15
```

```python
# Python
def can_get_wood(trees, m, height):
    wood = sum(max(0, tree - height) for tree in trees)
    return wood >= m

def max_cut_height(trees, m):
    left, right = 0, max(trees)
    answer = 0

    while left <= right:
        mid = (left + right) // 2

        if can_get_wood(trees, m, mid):
            answer = mid
            left = mid + 1
        else:
            right = mid - 1

    return answer
```

### Example 3: Installing Routers

```
Problem: Install C routers in N houses,
         what's the maximum minimum distance between adjacent routers?

House positions: [1, 2, 8, 4, 9], C=3 → Sorted: [1, 2, 4, 8, 9]

Distance 3: 1, 4, 8 or 1, 4, 9 (possible)
Distance 4: 1, 8 (only 2 possible, impossible)

→ Maximum distance is 3
```

```cpp
// C++
bool canInstall(const vector<int>& houses, int c, int dist) {
    // Greedy: always install at the first house that is far enough from the last
    // router — a greedy choice is valid here because installing sooner never
    // prevents future installations (it only leaves more room to the right)
    int count = 1;
    int lastPos = houses[0];

    for (int i = 1; i < houses.size(); i++) {
        if (houses[i] - lastPos >= dist) {
            count++;
            lastPos = houses[i];
        }
    }

    return count >= c;
}

int maxMinDistance(vector<int>& houses, int c) {
    // Must sort first: canInstall relies on consecutive positions being adjacent,
    // and the binary search range [1, max_gap] is only meaningful on sorted input
    sort(houses.begin(), houses.end());

    int left = 1;
    int right = houses.back() - houses.front();  // Maximum possible minimum distance
    int answer = 0;

    while (left <= right) {
        int mid = (left + right) / 2;

        if (canInstall(houses, c, mid)) {
            answer = mid;   // mid is feasible; remember it and try for larger
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }

    return answer;
}
```

```python
# Python
def can_install(houses, c, dist):
    count = 1
    last_pos = houses[0]

    for house in houses[1:]:
        if house - last_pos >= dist:
            count += 1
            last_pos = house

    return count >= c

def max_min_distance(houses, c):
    houses.sort()

    left, right = 1, houses[-1] - houses[0]
    answer = 0

    while left <= right:
        mid = (left + right) // 2

        if can_install(houses, c, mid):
            answer = mid
            left = mid + 1
        else:
            right = mid - 1

    return answer
```

---

## 6. Hash Search

### Concept

```
Search in O(1) time using hash function

Key → Hash function → Hash value (index) → Data

Example: "apple" → hash("apple") = 3 → arr[3] = Data
```

### Using Hash Tables

```cpp
// C++
#include <unordered_map>
#include <unordered_set>

// Hash map (key-value pairs)
unordered_map<string, int> map;
map["apple"] = 5;
map["banana"] = 3;

// Search O(1)
if (map.count("apple")) {
    cout << map["apple"] << endl;  // 5
}

// Hash set (keys only)
unordered_set<int> set;
set.insert(1);
set.insert(2);

// Existence check O(1)
if (set.count(1)) {
    cout << "Found" << endl;
}
```

```python
# Python
# Dictionary (hash map)
d = {"apple": 5, "banana": 3}

# Search O(1)
if "apple" in d:
    print(d["apple"])  # 5

# Set (hash set)
s = {1, 2, 3}

# Existence check O(1)
if 1 in s:
    print("Found")
```

### Two Sum Problem (Using Hash)

```
Problem: Find index pair where two numbers sum to target

Array: [2, 7, 11, 15], target=9
Answer: [0, 1] (2 + 7 = 9)
```

```cpp
// C++ - O(n) hash solution
vector<int> twoSum(const vector<int>& nums, int target) {
    unordered_map<int, int> seen;  // value → index

    for (int i = 0; i < nums.size(); i++) {
        int complement = target - nums[i];

        if (seen.count(complement)) {
            return {seen[complement], i};
        }

        seen[nums[i]] = i;
    }

    return {-1, -1};
}
```

```python
# Python
def two_sum(nums, target):
    seen = {}  # value → index

    for i, num in enumerate(nums):
        complement = target - num

        if complement in seen:
            return [seen[complement], i]

        seen[num] = i

    return [-1, -1]
```

---

## 7. Practice Problems

### Problem 1: Square Root

Return the integer square root of x (floor).

```
Input: 8
Output: 2 (floor of 2.828...)
```

<details>
<summary>Hint</summary>

Find maximum mid where mid * mid <= x (binary search)

</details>

<details>
<summary>Solution</summary>

```python
def sqrt(x):
    if x < 2:
        return x

    left, right = 1, x // 2
    answer = 1

    while left <= right:
        mid = (left + right) // 2

        if mid * mid <= x:
            answer = mid
            left = mid + 1
        else:
            right = mid - 1

    return answer
```

</details>

### Problem 2: Find Peak Element

Peak element: element greater than its neighbors

```
Input: [1, 2, 3, 1]
Output: 2 (element 3 at index 2 is a peak)
```

<details>
<summary>Hint</summary>

Can be solved in O(log n) with binary search
- If mid > mid+1, peak exists on left
- If mid < mid+1, peak exists on right

</details>

<details>
<summary>Solution</summary>

```python
def find_peak_element(arr):
    left, right = 0, len(arr) - 1

    while left < right:
        mid = (left + right) // 2

        if arr[mid] > arr[mid + 1]:
            right = mid
        else:
            left = mid + 1

    return left
```

</details>

### Recommended Problems

| Difficulty | Problem | Platform | Type |
|--------|------|--------|------|
| ⭐ | [Binary Search](https://leetcode.com/problems/binary-search/) | LeetCode | Basic binary search |
| ⭐ | [Find Number](https://www.acmicpc.net/problem/1920) | Baekjoon | Binary search |
| ⭐⭐ | [Search Insert Position](https://leetcode.com/problems/search-insert-position/) | LeetCode | Lower Bound |
| ⭐⭐ | [Cutting Cables](https://www.acmicpc.net/problem/1654) | Baekjoon | Parametric search |
| ⭐⭐ | [Cutting Trees](https://www.acmicpc.net/problem/2805) | Baekjoon | Parametric search |
| ⭐⭐⭐ | [Search in Rotated Array](https://leetcode.com/problems/search-in-rotated-sorted-array/) | LeetCode | Rotated array |
| ⭐⭐⭐ | [Installing Routers](https://www.acmicpc.net/problem/2110) | Baekjoon | Parametric search |

---

## Binary Search Templates Summary

### Basic Template

```python
# Find exact target
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1
```

### Lower Bound / Upper Bound

```python
# First position >= target
def lower_bound(arr, target):
    left, right = 0, len(arr)
    while left < right:
        mid = (left + right) // 2
        if arr[mid] < target:
            left = mid + 1
        else:
            right = mid
    return left

# First position > target
def upper_bound(arr, target):
    left, right = 0, len(arr)
    while left < right:
        mid = (left + right) // 2
        if arr[mid] <= target:
            left = mid + 1
        else:
            right = mid
    return left
```

### Parametric Search

```python
# Maximum value satisfying condition
def parametric_max(check, low, high):
    answer = low
    while low <= high:
        mid = (low + high) // 2
        if check(mid):
            answer = mid
            low = mid + 1
        else:
            high = mid - 1
    return answer

# Minimum value satisfying condition
def parametric_min(check, low, high):
    answer = high
    while low <= high:
        mid = (low + high) // 2
        if check(mid):
            answer = mid
            high = mid - 1
        else:
            low = mid + 1
    return answer
```

---

## Next Steps

- [07_Divide_and_Conquer.md](./07_Divide_and_Conquer.md) - Divide and Conquer

---

## References

- [Binary Search Tutorial](https://www.topcoder.com/thrive/articles/Binary%20Search)
- [Parametric Search Guide](https://cp-algorithms.com/num_methods/binary_search.html)
- [Binary Search Summary](https://www.acmicpc.net/blog/view/109)
