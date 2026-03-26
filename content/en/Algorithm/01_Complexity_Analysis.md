# Complexity Analysis

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain why hardware-independent complexity analysis is necessary for comparing algorithms
2. Apply Big O notation to express the time and space complexity of common algorithms
3. Analyze iterative and recursive code to determine its time complexity class
4. Distinguish between best-case, average-case, and worst-case complexity
5. Compare the practical trade-offs between time complexity and space complexity when selecting an algorithm
6. Evaluate algorithm scalability by identifying how different complexity classes perform as input size grows

---

## Overview

Learn how to measure algorithm efficiency. Understanding Big O notation and the ability to analyze time/space complexity of code is fundamental to algorithm learning.

---

## Table of Contents

1. [Why Do We Need Complexity Analysis?](#1-why-do-we-need-complexity-analysis)
2. [Big O Notation](#2-big-o-notation)
3. [Time Complexity](#3-time-complexity)
4. [Space Complexity](#4-space-complexity)
5. [Complexity Analysis Examples](#5-complexity-analysis-examples)
6. [Practical Analysis Tips](#6-practical-analysis-tips)
7. [Practice Problems](#7-practice-problems)

---

## 1. Why Do We Need Complexity Analysis?

### Execution Time Limitations

```
Actual execution time depends on:
- Hardware performance (CPU, memory)
- Programming language
- Compiler optimization
- Input data characteristics
```

### Advantages of Complexity Analysis

```
1. Hardware independent
2. Understand growth rate with input size
3. Objective algorithm comparison
4. Predict scalability
```

### Example: Same Problem, Different Algorithms

```
Problem: Find maximum value in n numbers

Method 1: Sequential search → O(n)
Method 2: Sort then get last element → O(n log n)

When n = 1,000,000:
- Method 1: ~1,000,000 comparisons
- Method 2: ~20,000,000 operations

→ Method 1 is about 20× faster
```

---

## 2. Big O Notation

### Definition

```
Big O: When input size n increases,
       represents the upper bound of operation count

f(n) = O(g(n))
→ f(n) does not grow faster than a constant multiple of g(n)
```

### Notation Rules

```
1. Ignore constants
   O(2n) = O(n)
   O(100) = O(1)

2. Ignore lower-order terms
   O(n² + n) = O(n²)
   O(n³ + n² + n) = O(n³)

3. Keep only highest-order term
   O(3n² + 2n + 1) = O(n²)
```

### Major Complexity Classes

```
┌─────────────┬───────────────┬─────────────────────┐
│ Complexity  │     Name      │      Example        │
├─────────────┼───────────────┼─────────────────────┤
│ O(1)        │ Constant      │ Array index access  │
│ O(log n)    │ Logarithmic   │ Binary search       │
│ O(n)        │ Linear        │ Sequential search   │
│ O(n log n)  │ Linearithmic  │ Merge sort          │
│ O(n²)       │ Quadratic     │ Bubble sort         │
│ O(n³)       │ Cubic         │ Floyd-Warshall      │
│ O(2ⁿ)       │ Exponential   │ Subset enumeration  │
│ O(n!)       │ Factorial     │ Permutation enum    │
└─────────────┴───────────────┴─────────────────────┘
```

### Complexity Comparison Graph

```
Operations
   │
   │                                    O(n!)
   │                               ╱
   │                          O(2ⁿ)
   │                      ╱
   │                 O(n²)
   │             ╱
   │        O(n log n)
   │      ╱
   │   O(n)
   │ ╱
   │──────O(log n)
   │══════O(1)
   └───────────────────────────────→ n (input size)
```

### Operation Counts by Input Size

```
┌──────┬─────────┬─────────┬──────────┬───────────┬────────────┐
│  n   │ O(log n)│  O(n)   │O(n log n)│   O(n²)   │   O(2ⁿ)    │
├──────┼─────────┼─────────┼──────────┼───────────┼────────────┤
│   10 │       3 │      10 │       33 │       100 │      1,024 │
│  100 │       7 │     100 │      664 │    10,000 │   10³⁰     │
│ 1000 │      10 │   1,000 │    9,966 │ 1,000,000 │   10³⁰⁰    │
│  10⁶ │      20 │     10⁶ │  2×10⁷   │    10¹²   │     ∞      │
└──────┴─────────┴─────────┴──────────┴───────────┴────────────┘
```

---

## 3. Time Complexity

### 3.1 O(1) - Constant Time

```c
// C
int getFirst(int arr[], int n) {
    return arr[0];  // always 1 operation
}

int getElement(int arr[], int index) {
    return arr[index];  // independent of array size
}
```

```cpp
// C++
int getFirst(const vector<int>& arr) {
    return arr[0];
}

void swap(int& a, int& b) {
    int temp = a;
    a = b;
    b = temp;
}
```

```python
# Python
def get_first(arr):
    return arr[0]

def get_element(arr, index):
    return arr[index]
```

### 3.2 O(log n) - Logarithmic Time

```c
// C - Binary search
int binarySearch(int arr[], int n, int target) {
    int left = 0, right = n - 1;

    while (left <= right) {
        // Use left + (right - left) / 2 instead of (left + right) / 2 —
        // the naive form overflows when left + right exceeds INT_MAX (2^31-1)
        int mid = left + (right - left) / 2;

        if (arr[mid] == target)
            return mid;
        else if (arr[mid] < target)
            left = mid + 1;  // Target must be in the right half; discard left
        else
            right = mid - 1; // Target must be in the left half; discard right
    }
    return -1;
}
// Search range halves each iteration
// n → n/2 → n/4 → ... → 1
// Iterations: log₂(n)
```

```cpp
// C++
int binarySearch(const vector<int>& arr, int target) {
    int left = 0, right = arr.size() - 1;

    while (left <= right) {
        int mid = left + (right - left) / 2;

        if (arr[mid] == target)
            return mid;
        else if (arr[mid] < target)
            left = mid + 1;
        else
            right = mid - 1;
    }
    return -1;
}
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
```

### 3.3 O(n) - Linear Time

```c
// C - Find maximum
int findMax(int arr[], int n) {
    int max = arr[0];
    for (int i = 1; i < n; i++) {  // n-1 iterations
        if (arr[i] > max)
            max = arr[i];
    }
    return max;
}
```

```cpp
// C++
int findMax(const vector<int>& arr) {
    int maxVal = arr[0];
    for (int i = 1; i < arr.size(); i++) {
        maxVal = max(maxVal, arr[i]);
    }
    return maxVal;
}

// Using STL
int findMaxSTL(const vector<int>& arr) {
    return *max_element(arr.begin(), arr.end());
}
```

```python
# Python
def find_max(arr):
    max_val = arr[0]
    for x in arr[1:]:
        if x > max_val:
            max_val = x
    return max_val

# Using built-in function
def find_max_builtin(arr):
    return max(arr)
```

### 3.4 O(n log n) - Linearithmic Time

```c
// C - Merge sort (concept)
void mergeSort(int arr[], int left, int right) {
    // left < right, not left != right: handles length-1 subarrays without
    // extra base-case logic — a single element is already sorted
    if (left < right) {
        // Compute mid with overflow-safe arithmetic (same reason as binary search)
        int mid = left + (right - left) / 2;

        mergeSort(arr, left, mid);      // T(n/2)
        mergeSort(arr, mid + 1, right); // T(n/2)
        merge(arr, left, mid, right);   // O(n) — the cost that drives the log factor
    }
}
// T(n) = 2T(n/2) + O(n) = O(n log n)
```

```cpp
// C++ - STL sort
#include <algorithm>

void sortArray(vector<int>& arr) {
    sort(arr.begin(), arr.end());  // O(n log n)
}
```

```python
# Python
def sort_array(arr):
    return sorted(arr)  # O(n log n) - Timsort

arr = [3, 1, 4, 1, 5, 9, 2, 6]
arr.sort()  # in-place sort
```

### 3.5 O(n²) - Quadratic Time

```c
// C - Bubble sort
void bubbleSort(int arr[], int n) {
    for (int i = 0; i < n - 1; i++) {       // n-1 times
        for (int j = 0; j < n - i - 1; j++) { // n-i-1 times
            if (arr[j] > arr[j + 1]) {
                int temp = arr[j];
                arr[j] = arr[j + 1];
                arr[j + 1] = temp;
            }
        }
    }
}
// (n-1) + (n-2) + ... + 1 = n(n-1)/2 = O(n²)
```

```cpp
// C++ - Nested loops
void printPairs(const vector<int>& arr) {
    int n = arr.size();
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            cout << arr[i] << ", " << arr[j] << endl;
        }
    }
}
// n × n = O(n²)
```

```python
# Python
def print_pairs(arr):
    n = len(arr)
    for i in range(n):
        for j in range(n):
            print(arr[i], arr[j])
```

### 3.6 O(2ⁿ) - Exponential Time

```c
// C - Fibonacci (recursive, inefficient)
int fibonacci(int n) {
    if (n <= 1)
        return n;
    return fibonacci(n - 1) + fibonacci(n - 2);
}
// T(n) = T(n-1) + T(n-2) ≈ O(2ⁿ)
```

```cpp
// C++ - Generate subsets
void generateSubsets(vector<int>& arr, int index, vector<int>& current) {
    if (index == arr.size()) {
        // Base case: we've made an include/exclude decision for every element —
        // 'current' now holds one complete subset
        for (int x : current) cout << x << " ";
        cout << endl;
        return;
    }

    // Exclude-before-include ordering is arbitrary; both branches must be explored
    // because each element has exactly two choices, giving 2ⁿ total paths

    // Exclude current element — recurse without modifying 'current'
    generateSubsets(arr, index + 1, current);

    // Include current element — push/pop maintains the invariant that 'current'
    // is the same state before and after this branch (backtracking)
    current.push_back(arr[index]);
    generateSubsets(arr, index + 1, current);
    current.pop_back();
}
// Generates 2ⁿ subsets
```

```python
# Python - Generate subsets
def generate_subsets(arr):
    result = []
    n = len(arr)

    # 1 << n is the same as 2^n; each integer i represents one subset
    # where bit j being set means arr[j] is included — avoids explicit recursion
    for i in range(1 << n):  # 2^n iterations
        subset = []
        for j in range(n):
            # Check if bit j is set in i; each bit corresponds to one element's
            # include/exclude decision — this is the iterative equivalent of the
            # recursive two-branch approach above
            if i & (1 << j):
                subset.append(arr[j])
        result.append(subset)

    return result
```

---

## 4. Space Complexity

### 4.1 Concept

```
Space Complexity = Amount of memory used by algorithm

Components:
1. Input space: Storage for input data
2. Auxiliary space: Additional memory for algorithm execution

Generally only auxiliary space is counted
```

### 4.2 O(1) - Constant Space

```c
// C - In-place swap
void reverseArray(int arr[], int n) {
    int left = 0, right = n - 1;
    while (left < right) {
        int temp = arr[left];
        arr[left] = arr[right];
        arr[right] = temp;
        left++;
        right--;
    }
}
// Only 3 variables used (left, right, temp)
```

```python
# Python
def reverse_array(arr):
    left, right = 0, len(arr) - 1
    while left < right:
        arr[left], arr[right] = arr[right], arr[left]
        left += 1
        right -= 1
```

### 4.3 O(n) - Linear Space

```c
// C - Array copy
int* copyArray(int arr[], int n) {
    int* copy = (int*)malloc(n * sizeof(int));
    for (int i = 0; i < n; i++) {
        copy[i] = arr[i];
    }
    return copy;
}
// Requires additional memory for n elements
```

```cpp
// C++ - Merge process in merge sort
void merge(vector<int>& arr, int left, int mid, int right) {
    // Temporary buffer is necessary because merging in-place would overwrite
    // elements still needed for comparison — we trade O(n) space for correctness
    vector<int> temp(right - left + 1);  // O(n) additional space

    int i = left, j = mid + 1, k = 0;
    while (i <= mid && j <= right) {
        // <= (not <) preserves stability: equal elements from the left half
        // are placed before those from the right half, maintaining original order
        if (arr[i] <= arr[j])
            temp[k++] = arr[i++];
        else
            temp[k++] = arr[j++];
    }
    // Drain whichever half still has elements; at most one of these loops runs
    while (i <= mid) temp[k++] = arr[i++];
    while (j <= right) temp[k++] = arr[j++];

    for (int i = 0; i < k; i++)
        arr[left + i] = temp[i];
}
```

```python
# Python
def copy_array(arr):
    return arr[:]  # O(n) space
```

### 4.4 O(log n) - Logarithmic Space (Recursion Stack)

```c
// C - Binary search (recursive)
int binarySearchRecursive(int arr[], int left, int right, int target) {
    if (left > right)
        return -1;

    int mid = left + (right - left) / 2;

    if (arr[mid] == target)
        return mid;
    else if (arr[mid] < target)
        return binarySearchRecursive(arr, mid + 1, right, target);
    else
        return binarySearchRecursive(arr, left, mid - 1, target);
}
// Recursion depth: O(log n) → Stack space O(log n)
```

### 4.5 Time-Space Tradeoff

```
┌─────────────────────────────────────────────────────────┐
│ Algorithm         │ Time        │ Space       │ Notes  │
├───────────────────┼─────────────┼─────────────┼────────┤
│ Merge sort        │ O(n log n)  │ O(n)        │ Stable │
│ Quick sort        │ O(n log n)  │ O(log n)    │ In-pl. │
│ Heap sort         │ O(n log n)  │ O(1)        │ In-pl. │
├───────────────────┼─────────────┼─────────────┼────────┤
│ Fibonacci (recur) │ O(2ⁿ)       │ O(n)        │ Poor   │
│ Fibonacci (memo)  │ O(n)        │ O(n)        │ Space↑ │
│ Fibonacci (iter)  │ O(n)        │ O(1)        │ Optimal│
└─────────────────────────────────────────────────────────┘
```

---

## 5. Complexity Analysis Examples

### Example 1: Nested Loops

```cpp
// Analysis: What is the time complexity?
int example1(int n) {
    int count = 0;
    for (int i = 0; i < n; i++) {          // n times
        for (int j = 0; j < n; j++) {      // n times
            count++;
        }
    }
    return count;
}
// Answer: O(n²)
// n × n = n²
```

### Example 2: Asymmetric Nested Loops

```cpp
int example2(int n) {
    int count = 0;
    for (int i = 0; i < n; i++) {          // n times
        for (int j = 0; j < i; j++) {      // 0, 1, 2, ..., n-1 times
            count++;
        }
    }
    return count;
}
// Answer: O(n²)
// 0 + 1 + 2 + ... + (n-1) = n(n-1)/2 = O(n²)
```

### Example 3: Logarithmic Iteration

```cpp
int example3(int n) {
    int count = 0;
    for (int i = 1; i < n; i *= 2) {       // log₂(n) times
        count++;
    }
    return count;
}
// Answer: O(log n)
// i: 1, 2, 4, 8, ..., n → log₂(n) iterations
```

### Example 4: Nested Logarithmic Iteration

```cpp
int example4(int n) {
    int count = 0;
    for (int i = 0; i < n; i++) {          // n times
        for (int j = 1; j < n; j *= 2) {   // log n times
            count++;
        }
    }
    return count;
}
// Answer: O(n log n)
// n × log n
```

### Example 5: Sequential Loops

```cpp
int example5(int n) {
    int count = 0;

    for (int i = 0; i < n; i++) {          // O(n)
        count++;
    }

    for (int i = 0; i < n; i++) {          // O(n)
        for (int j = 0; j < n; j++) {      // O(n)
            count++;
        }
    }

    return count;
}
// Answer: O(n²)
// O(n) + O(n²) = O(n²)
// Keep only highest-order term
```

### Example 6: Conditional Execution

```cpp
int example6(int n, bool flag) {
    int count = 0;

    if (flag) {
        for (int i = 0; i < n; i++) {      // O(n)
            count++;
        }
    } else {
        for (int i = 0; i < n; i++) {      // O(n²)
            for (int j = 0; j < n; j++) {
                count++;
            }
        }
    }

    return count;
}
// Answer: O(n²)
// Consider worst case
```

### Example 7: Recursion Analysis

```cpp
int example7(int n) {
    if (n <= 1)
        return 1;
    return example7(n - 1) + example7(n - 1);
}
// Answer: O(2ⁿ)
// T(n) = 2T(n-1) + O(1)
// Call tree:
//           f(4)
//          /    \
//       f(3)    f(3)
//       / \      / \
//    f(2) f(2) f(2) f(2)
//    ...
// Nodes per level: 1, 2, 4, 8, ... = 2ⁿ
```

---

## 6. Practical Analysis Tips

### 6.1 Coding Test Time Limit Guidelines

```
For 1 second time limit (C/C++):
┌───────────────┬─────────────────────┐
│  Complexity   │  Maximum Input Size │
├───────────────┼─────────────────────┤
│ O(n!)         │ n ≤ 10              │
│ O(2ⁿ)         │ n ≤ 20              │
│ O(n³)         │ n ≤ 500             │
│ O(n²)         │ n ≤ 5,000           │
│ O(n log n)    │ n ≤ 1,000,000       │
│ O(n)          │ n ≤ 10,000,000      │
│ O(log n)      │ n ≤ 10¹⁸            │
└───────────────┴─────────────────────┘

Python is ~10-100× slower than C/C++
→ Reduce n by 1/10 from above guidelines
```

### 6.2 Common Operation Complexities

```
┌──────────────────────────────────────────────────────┐
│ Data Structure/Op  │ Average   │ Worst     │ Note   │
├────────────────────┼───────────┼───────────┼────────┤
│ Array access       │ O(1)      │ O(1)      │        │
│ Array search       │ O(n)      │ O(n)      │        │
│ Array insert/del   │ O(n)      │ O(n)      │ Shift  │
├────────────────────┼───────────┼───────────┼────────┤
│ Hash table access  │ O(1)      │ O(n)      │ Coll.  │
│ Hash table insert  │ O(1)      │ O(n)      │ Coll.  │
├────────────────────┼───────────┼───────────┼────────┤
│ Binary search tree │ O(log n)  │ O(n)      │ Skewed │
│ Balanced BST       │ O(log n)  │ O(log n)  │ AVL    │
├────────────────────┼───────────┼───────────┼────────┤
│ Heap insert/delete │ O(log n)  │ O(log n)  │        │
│ Heap min access    │ O(1)      │ O(1)      │        │
└──────────────────────────────────────────────────────┘
```

### 6.3 Analysis Checklist

```
□ Identify iteration counts for all loops
□ Nested loops multiply complexity
□ Sequential loops add (keep highest order)
□ Analyze recursive calls with recurrence
□ Consider worst case for conditionals
□ Check library function complexity
□ Analyze space complexity too
```

---

## 7. Practice Problems

### Problem 1: Calculate Complexity

Find the time complexity of the following code.

```cpp
void mystery(int n) {
    for (int i = n; i >= 1; i /= 2) {
        for (int j = 1; j <= n; j *= 2) {
            // O(1) work
        }
    }
}
```

<details>
<summary>Show Answer</summary>

**O(log²n)**

- Outer loop: i decreases from n to 1 by half → log n times
- Inner loop: j increases from 1 to n by doubling → log n times
- Total: log n × log n = O(log²n)

</details>

### Problem 2: Complexity Comparison

Compare operation counts when n = 1000.

```
A. O(n)
B. O(n log n)
C. O(n²)
D. O(2^(log n))
```

<details>
<summary>Show Answer</summary>

```
A. O(n) = 1,000
B. O(n log n) ≈ 10,000
C. O(n²) = 1,000,000
D. O(2^(log n)) = O(n) = 1,000

Since 2^(log₂ n) = n, A and D are equal
```

</details>

### Problem 3: Recursion Analysis

```cpp
int f(int n) {
    if (n <= 1) return 1;
    return f(n/2) + f(n/2);
}
```

<details>
<summary>Show Answer</summary>

**O(n)**

Recurrence: T(n) = 2T(n/2) + O(1)

Applying Master Theorem:
- a = 2, b = 2, f(n) = O(1)
- n^(log₂2) = n¹ = n
- f(n) = O(1) < n → Case 1
- T(n) = O(n)

</details>

### Recommended Problems

| Difficulty | Problem | Platform |
|-----------|---------|---------|
| ⭐ | [Sort Numbers](https://www.acmicpc.net/problem/2750) | Baekjoon |
| ⭐ | [Number Search](https://www.acmicpc.net/problem/1920) | Baekjoon |
| ⭐⭐ | [Two Sum](https://leetcode.com/problems/two-sum/) | LeetCode |
| ⭐⭐ | [Contains Duplicate](https://leetcode.com/problems/contains-duplicate/) | LeetCode |

---

## Next Steps

- [02_Arrays_and_Strings.md](./02_Arrays_and_Strings.md) - Two pointers, sliding window

---

## References

- [Big-O Cheat Sheet](https://www.bigocheatsheet.com/)
- [VisuAlgo](https://visualgo.net/)
- Introduction to Algorithms (CLRS)
