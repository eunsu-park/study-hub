# Divide and Conquer

## Learning Objectives

After completing this lesson, you will be able to:

1. Describe the three steps of divide and conquer (divide, conquer, combine) and identify when a problem is suited for this paradigm
2. Distinguish divide and conquer from dynamic programming by explaining the role of overlapping subproblems and memoization
3. Implement Merge Sort using the divide and conquer pattern and trace through its recursion tree to derive the O(n log n) complexity
4. Implement Quick Sort with partition-based divide and conquer, and analyze its best, average, and worst-case behavior
5. Apply fast exponentiation (exponentiation by squaring) to compute large powers in O(log n) time
6. Analyze the recurrence relations of divide and conquer algorithms using the Master Theorem

---

## Overview

Divide and conquer is an algorithm design technique that divides a large problem into smaller subproblems, solves each one, and then combines the solutions.

---

## Table of Contents

1. [Divide and Conquer Concept](#1-divide-and-conquer-concept)
2. [Merge Sort](#2-merge-sort)
3. [Quick Sort](#3-quick-sort)
4. [Binary Search](#4-binary-search)
5. [Exponentiation](#5-exponentiation)
6. [Practice Problems](#6-practice-problems)

---

## 1. Divide and Conquer Concept

### Basic Steps

```
Divide and Conquer 3 Steps:

1. Divide
   - Split problem into smaller subproblems

2. Conquer
   - Solve subproblems recursively
   - Solve directly if small enough

3. Combine
   - Merge solutions of subproblems to construct solution for original problem
```

### Visualization

```
        [Problem]
       /         \
   [Sub1]      [Sub2]
   /    \      /    \
 [a]   [b]   [c]   [d]
   \    /      \    /
   [Sub1]      [Sub2]
       \         /
        [Solution]
```

### Divide and Conquer vs DP

```
┌────────────────┬─────────────────┬─────────────────┐
│                │ Divide & Conquer│ DP              │
├────────────────┼─────────────────┼─────────────────┤
│ Subproblems    │ Independent     │ Overlapping     │
│ Storage        │ Not needed      │ Memoization     │
│ Example        │ Merge sort      │ Fibonacci       │
└────────────────┴─────────────────┴─────────────────┘
```

---

## 2. Merge Sort

### Principle

```
1. Split array in half
2. Recursively sort each half
3. Merge two sorted halves

[38, 27, 43, 3, 9, 82, 10]
           ↓ Divide
[38, 27, 43, 3]  [9, 82, 10]
      ↓              ↓
[38, 27] [43, 3] [9, 82] [10]
    ↓       ↓       ↓      ↓
[38][27] [43][3] [9][82]  [10]
    ↓       ↓       ↓      ↓
[27, 38] [3, 43] [9, 82] [10]
      ↓              ↓
[3, 27, 38, 43]  [9, 10, 82]
           ↓ Combine
[3, 9, 10, 27, 38, 43, 82]
```

### Implementation

```c
// C
void merge(int arr[], int left, int mid, int right) {
    int n1 = mid - left + 1;
    int n2 = right - mid;

    // VLAs (variable-length arrays) used here for simplicity; in production C
    // prefer malloc to avoid stack overflow on large n
    int L[n1], R[n2];

    // Copy both halves before merging — once we start writing back into arr,
    // the original values would be lost if we read from arr directly
    for (int i = 0; i < n1; i++) L[i] = arr[left + i];
    for (int i = 0; i < n2; i++) R[i] = arr[mid + 1 + i];

    int i = 0, j = 0, k = left;

    while (i < n1 && j < n2) {
        // <= ensures stability: equal left elements are placed before right elements
        if (L[i] <= R[j]) arr[k++] = L[i++];
        else arr[k++] = R[j++];
    }

    // Exactly one of these drains any remaining elements from the unfinished half
    while (i < n1) arr[k++] = L[i++];
    while (j < n2) arr[k++] = R[j++];
}

void mergeSort(int arr[], int left, int right) {
    if (left < right) {
        // Overflow-safe midpoint — also avoids the off-by-one that (left+right)/2
        // would cause when left = 0, right = 1 (mid would still be 0, which is correct,
        // but the safe form is a good habit whenever indices can be large)
        int mid = left + (right - left) / 2;

        mergeSort(arr, left, mid);
        mergeSort(arr, mid + 1, right);
        merge(arr, left, mid, right);
    }
}
```

```python
def merge_sort(arr):
    # Base case: zero or one element is trivially sorted — also the recursion
    # termination condition that prevents infinite splitting
    if len(arr) <= 1:
        return arr

    mid = len(arr) // 2
    # arr[:mid] and arr[mid:] create new list objects (O(n) space each level),
    # which is why Python merge sort uses O(n log n) space in naive form;
    # an in-place variant avoids this but is significantly more complex
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])

    return merge(left, right)

def merge(left, right):
    result = []
    i = j = 0

    while i < len(left) and j < len(right):
        # <= preserves stability: among equal elements, the left subarray's
        # element is always preferred, maintaining the original input ordering
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1

    # extend is O(k) where k is the remaining elements — much cheaper than
    # looping with append because it avoids per-element Python overhead
    result.extend(left[i:])
    result.extend(right[j:])
    return result
```

### Complexity Analysis

```
T(n) = 2T(n/2) + O(n)

Divide: O(1)
Conquer: 2 × T(n/2)
Combine: O(n)

Master Theorem:
a = 2, b = 2, f(n) = n
n^(log_b(a)) = n^1 = n
f(n) = Θ(n)

→ T(n) = Θ(n log n)

Space: O(n) - temporary array
```

---

## 3. Quick Sort

### Principle

```
1. Choose pivot
2. Partition: elements < pivot go left, elements > pivot go right
3. Recursively sort each partition

[5, 3, 8, 4, 2, 7, 1, 6], pivot=5
           ↓ Partition
[3, 4, 2, 1] [5] [8, 7, 6]
      ↓              ↓
[1, 2, 3, 4]    [6, 7, 8]
           ↓
[1, 2, 3, 4, 5, 6, 7, 8]
```

### Implementation

```cpp
// C++
int partition(vector<int>& arr, int low, int high) {
    int pivot = arr[high];
    // i starts at low - 1: it's the index of the last element confirmed smaller
    // than pivot; the region [low..i] grows as we find more such elements
    int i = low - 1;

    for (int j = low; j < high; j++) {
        if (arr[j] < pivot) {
            // Extend the smaller-than-pivot region and swap new element into it
            swap(arr[++i], arr[j]);
        }
    }

    // Place pivot at its correct sorted position — all elements to its left are
    // smaller, all to the right are larger (in-place, no extra memory needed)
    swap(arr[i + 1], arr[high]);
    return i + 1;
}

void quickSort(vector<int>& arr, int low, int high) {
    if (low < high) {
        int pi = partition(arr, low, high);

        // Pivot is now in final position; recurse only on the unsorted partitions
        quickSort(arr, low, pi - 1);
        quickSort(arr, pi + 1, high);
    }
}
```

```python
def quick_sort(arr, low, high):
    if low < high:
        pi = partition(arr, low, high)
        quick_sort(arr, low, pi - 1)
        quick_sort(arr, pi + 1, high)

def partition(arr, low, high):
    pivot = arr[high]
    i = low - 1

    for j in range(low, high):
        if arr[j] < pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]

    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return i + 1
```

### Complexity Analysis

```
Average: T(n) = 2T(n/2) + O(n) = O(n log n)
Worst: T(n) = T(n-1) + O(n) = O(n²)
      (already sorted array + first/last pivot)

Space: O(log n) - recursion stack
```

---

## 4. Binary Search

### Divide and Conquer Perspective

```
Problem: Find target in sorted array

Divide: Split by middle value into left/right
Conquer: Search only the side containing target
Combine: Not needed (return when found)

[1, 3, 5, 7, 9, 11, 13], target=9
           ↓
       [7] middle
      7 < 9
           ↓
    Search right: [9, 11, 13]
           ↓
       [11] middle
      11 > 9
           ↓
    Search left: [9]
           ↓
       Found!
```

### Recursive Implementation

```cpp
// C++
int binarySearchRecursive(const vector<int>& arr, int left, int right, int target) {
    if (left > right) return -1;

    int mid = left + (right - left) / 2;

    if (arr[mid] == target) return mid;
    if (arr[mid] > target) return binarySearchRecursive(arr, left, mid - 1, target);
    return binarySearchRecursive(arr, mid + 1, right, target);
}
```

```python
def binary_search_recursive(arr, left, right, target):
    if left > right:
        return -1

    mid = (left + right) // 2

    if arr[mid] == target:
        return mid
    if arr[mid] > target:
        return binary_search_recursive(arr, left, mid - 1, target)
    return binary_search_recursive(arr, mid + 1, right, target)
```

---

## 5. Exponentiation

### Simple Method vs Divide and Conquer

```
Calculate a^n

Simple: a × a × a × ... × a (n times) → O(n)

Divide and Conquer:
a^n = a^(n/2) × a^(n/2)        (n is even)
a^n = a^(n/2) × a^(n/2) × a    (n is odd)

→ O(log n)

Example: 2^10
= 2^5 × 2^5
= (2^2 × 2^2 × 2) × (2^2 × 2^2 × 2)
= ((2 × 2)^2 × 2)^2
```

### Implementation

```c
// C - Recursive
long long power(long long a, int n) {
    // n == 0: any number to the power 0 is 1 (empty product)
    if (n == 0) return 1;
    // n == 1 is an explicit short-circuit, not strictly necessary but avoids
    // one unnecessary squaring step for the common single-step case
    if (n == 1) return a;

    // Compute a^(n/2) once and reuse — this is the divide step that reduces
    // O(n) multiplications to O(log n) by squaring instead of multiplying linearly
    long long half = power(a, n / 2);

    if (n % 2 == 0) {
        return half * half;
    } else {
        // Odd exponent: a^n = a^(n/2) * a^(n/2) * a — the extra 'a' handles
        // the remainder when integer division floors n/2
        return half * half * a;
    }
}

// C - Iterative (bit operations)
long long powerIterative(long long a, int n) {
    long long result = 1;

    while (n > 0) {
        // Process one bit of n at a time: if the current bit is set, multiply
        // the current power of a into the result (bit = 1 means this term contributes)
        if (n & 1) {  // n is odd
            result *= a;
        }
        // Square a at each step: after k iterations, a = original_a^(2^k)
        a *= a;
        // Right-shift to examine the next bit of the exponent
        n >>= 1;
    }

    return result;
}
```

```python
def power(a, n):
    if n == 0:
        return 1
    if n == 1:
        return a

    half = power(a, n // 2)

    if n % 2 == 0:
        return half * half
    else:
        return half * half * a

# Modular exponentiation (large numbers)
def power_mod(a, n, mod):
    result = 1
    a %= mod

    while n > 0:
        if n & 1:
            result = (result * a) % mod
        a = (a * a) % mod
        n >>= 1

    return result
```

### Matrix Exponentiation

```
Calculate Fibonacci in O(log n)

[F(n+1)]   [1 1]^n   [F(1)]
[F(n)  ] = [1 0]   × [F(0)]

Use divide and conquer for matrix exponentiation!
```

```python
def matrix_mult(A, B, mod=10**9+7):
    # Standard 2×2 matrix multiplication; keeping mod applied at every step
    # prevents Python integers from growing to millions of digits (performance)
    return [
        [(A[0][0]*B[0][0] + A[0][1]*B[1][0]) % mod,
         (A[0][0]*B[0][1] + A[0][1]*B[1][1]) % mod],
        [(A[1][0]*B[0][0] + A[1][1]*B[1][0]) % mod,
         (A[1][0]*B[0][1] + A[1][1]*B[1][1]) % mod]
    ]

def matrix_power(M, n, mod=10**9+7):
    # Base case: M^1 = M (identity matrix would be M^0, but we start from n=1)
    if n == 1:
        return M

    if n % 2 == 0:
        # Square the result of the half-power — same divide-and-conquer idea as
        # scalar exponentiation; reduces recursive calls from O(n) to O(log n)
        half = matrix_power(M, n // 2, mod)
        return matrix_mult(half, half, mod)
    else:
        # Odd exponent: peel off one factor of M and handle the even part recursively
        return matrix_mult(M, matrix_power(M, n - 1, mod), mod)

def fibonacci(n):
    if n <= 1:
        return n

    # The recurrence [[F(n+1)], [F(n)]] = [[1,1],[1,0]]^n × [[F(1)],[F(0)]]
    # lets us compute Fibonacci in O(log n) multiplications instead of O(n)
    M = [[1, 1], [1, 0]]
    result = matrix_power(M, n)
    return result[1][0]  # result[1][0] = F(n) after n matrix multiplications
```

---

## 6. Practice Problems

### Problem 1: Count Inversions

Count pairs where i < j and arr[i] > arr[j]

<details>
<summary>Hint</summary>

Count during merge sort process

</details>

<details>
<summary>Solution</summary>

```python
def count_inversions(arr):
    def merge_count(arr, temp, left, mid, right):
        i = left
        j = mid + 1
        k = left
        inv_count = 0

        while i <= mid and j <= right:
            if arr[i] <= arr[j]:
                temp[k] = arr[i]
                i += 1
            else:
                temp[k] = arr[j]
                inv_count += (mid - i + 1)  # Key!
                j += 1
            k += 1

        while i <= mid:
            temp[k] = arr[i]
            i += 1
            k += 1

        while j <= right:
            temp[k] = arr[j]
            j += 1
            k += 1

        for i in range(left, right + 1):
            arr[i] = temp[i]

        return inv_count

    def merge_sort_count(arr, temp, left, right):
        inv_count = 0
        if left < right:
            mid = (left + right) // 2
            inv_count += merge_sort_count(arr, temp, left, mid)
            inv_count += merge_sort_count(arr, temp, mid + 1, right)
            inv_count += merge_count(arr, temp, left, mid, right)
        return inv_count

    n = len(arr)
    temp = [0] * n
    return merge_sort_count(arr, temp, 0, n - 1)
```

</details>

### Recommended Problems

| Difficulty | Problem | Platform | Type |
|--------|------|--------|------|
| ⭐⭐ | [Matrix Exponentiation](https://www.acmicpc.net/problem/10830) | Baekjoon | Matrix power |
| ⭐⭐ | [Fibonacci Number 6](https://www.acmicpc.net/problem/11444) | Baekjoon | Matrix power |
| ⭐⭐ | [Kth Number](https://www.acmicpc.net/problem/11004) | Baekjoon | Quick Select |
| ⭐⭐⭐ | [Bubble Sort](https://www.acmicpc.net/problem/1517) | Baekjoon | Inversions |
| ⭐⭐⭐ | [Closest Pair of Points](https://www.acmicpc.net/problem/2261) | Baekjoon | Divide & Conquer |

---

## Master Theorem

```
Solve recurrence relations of form T(n) = aT(n/b) + f(n)

Case 1: f(n) = O(n^(log_b(a) - ε))
        → T(n) = Θ(n^(log_b(a)))

Case 2: f(n) = Θ(n^(log_b(a)))
        → T(n) = Θ(n^(log_b(a)) log n)

Case 3: f(n) = Ω(n^(log_b(a) + ε))
        → T(n) = Θ(f(n))

Examples:
- Merge sort: T(n) = 2T(n/2) + n → O(n log n) [Case 2]
- Binary search: T(n) = T(n/2) + 1 → O(log n) [Case 2]
- Exponentiation: T(n) = T(n/2) + 1 → O(log n) [Case 2]
```

---

## Next Steps

- [08_Backtracking.md](./08_Backtracking.md) - Backtracking

---

## References

- Introduction to Algorithms (CLRS) - Chapter 4
- [Divide and Conquer](https://www.geeksforgeeks.org/divide-and-conquer/)
