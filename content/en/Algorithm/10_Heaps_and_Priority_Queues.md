# Heaps and Priority Queues

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain the heap property for both max-heaps and min-heaps, and represent a complete binary tree efficiently as an array using index arithmetic
2. Implement heap insert (sift-up) and extract-max/min (sift-down) operations, tracing each step to confirm the heap property is restored
3. Build a heap from an arbitrary array in O(n) time using the heapify algorithm and explain why it is more efficient than n individual insertions
4. Implement Heap Sort and analyze its O(n log n) time and O(1) space complexity
5. Use a priority queue (Python heapq or C++ priority_queue) to solve problems such as k-th largest element, merging k sorted lists, and task scheduling
6. Design a solution using two heaps (max-heap + min-heap) to maintain a running median efficiently

---

## Overview

A heap is a complete binary tree-based data structure that allows O(1) access to the maximum/minimum value and O(log n) insertion/deletion.

---

## Table of Contents

1. [Heap Concepts](#1-heap-concepts)
2. [Heap Operations](#2-heap-operations)
3. [Heap Sort](#3-heap-sort)
4. [Priority Queue](#4-priority-queue)
5. [Application Problems](#5-application-problems)
6. [Practice Problems](#6-practice-problems)

---

## 1. Heap Concepts

### Heap Property

```
Max Heap:
- Parent node >= Child nodes
- Root is maximum

       (16)
      /    \
    (14)   (10)
    /  \   /  \
  (8) (7)(9) (3)

Min Heap:
- Parent node <= Child nodes
- Root is minimum

        (1)
       /   \
     (3)   (2)
     / \   /
   (6)(5)(4)
```

### Array Representation

```
Heap is a complete binary tree → Efficient array representation

       (16)                 Index:
      /    \                0: 16
    (14)   (10)             1: 14, 2: 10
    /  \   /  \             3: 8, 4: 7, 5: 9, 6: 3
  (8) (7)(9) (3)

Array: [16, 14, 10, 8, 7, 9, 3]

Index relationships (0-based):
- Parent: (i - 1) / 2
- Left child: 2 * i + 1
- Right child: 2 * i + 2
```

---

## 2. Heap Operations

### 2.1 Insert

```
Insertion process (Max Heap, insert 15):

1. Insert at last position
       (16)
      /    \
    (14)   (10)
    /  \   /  \
  (8) (7)(9) (3)
  /
(15)

2. Compare with parent and move up (Bubble Up)
       (16)
      /    \
    (15)   (10)
    /  \   /  \
  (14)(7)(9) (3)
  /
(8)

Time complexity: O(log n)
```

```c
// C
#define MAX_SIZE 1000

int heap[MAX_SIZE];
int heapSize = 0;

void insert(int value) {
    // Place the new element at the end to maintain the complete binary tree shape;
    // a complete tree maps perfectly to an array without wasted indices
    heap[heapSize] = value;
    int i = heapSize;
    heapSize++;

    // Bubble up: repeatedly compare with parent until the max-heap property holds;
    // each swap moves the new value one level closer to its correct position —
    // at most O(log n) swaps because the tree height is log n
    while (i > 0 && heap[(i - 1) / 2] < heap[i]) {
        int parent = (i - 1) / 2;  // Parent index derived from array formula
        int temp = heap[i];
        heap[i] = heap[parent];
        heap[parent] = temp;
        i = parent;
    }
}
```

```python
class MaxHeap:
    def __init__(self):
        self.heap = []

    def insert(self, value):
        self.heap.append(value)
        self._bubble_up(len(self.heap) - 1)

    def _bubble_up(self, i):
        while i > 0:
            parent = (i - 1) // 2
            if self.heap[parent] >= self.heap[i]:
                break
            self.heap[parent], self.heap[i] = self.heap[i], self.heap[parent]
            i = parent
```

### 2.2 Extract

```
Extract max process:

1. Remove root (max), move last element to root
       (3)
      /    \
    (14)   (10)
    /  \   /
  (8) (7)(9)

2. Compare with children and move down (Bubble Down)
       (14)
      /    \
    (8)    (10)
    /  \   /
  (3) (7)(9)

Time complexity: O(log n)
```

```c
// C
int extractMax() {
    if (heapSize <= 0) return -1;

    int max = heap[0];
    heap[0] = heap[heapSize - 1];
    heapSize--;

    // Bubble down
    int i = 0;
    while (1) {
        int left = 2 * i + 1;
        int right = 2 * i + 2;
        int largest = i;

        if (left < heapSize && heap[left] > heap[largest])
            largest = left;
        if (right < heapSize && heap[right] > heap[largest])
            largest = right;

        if (largest == i) break;

        int temp = heap[i];
        heap[i] = heap[largest];
        heap[largest] = temp;
        i = largest;
    }

    return max;
}
```

```python
class MaxHeap:
    def extract_max(self):
        if not self.heap:
            return None

        max_val = self.heap[0]
        # Move the last element to the root to maintain the complete tree shape;
        # removing from the end is O(1) whereas removing from the middle would
        # require shifting all elements
        self.heap[0] = self.heap[-1]
        self.heap.pop()

        if self.heap:
            self._bubble_down(0)

        return max_val

    def _bubble_down(self, i):
        n = len(self.heap)

        while True:
            left = 2 * i + 1
            right = 2 * i + 2
            largest = i  # Assume current node is the largest

            # Find the largest among node and its two children;
            # swapping with the largest child preserves the heap property locally
            if left < n and self.heap[left] > self.heap[largest]:
                largest = left
            if right < n and self.heap[right] > self.heap[largest]:
                largest = right

            # If the current node is already the largest, the heap property is
            # restored — stop early rather than iterating to a leaf every time
            if largest == i:
                break

            self.heap[i], self.heap[largest] = self.heap[largest], self.heap[i]
            i = largest
```

### 2.3 Heapify (Array to Heap)

```
Convert array to heap

Method 1: Repeated insertion - O(n log n)
Method 2: Bottom-up heapify - O(n)

Bottom-up:
- Bubble down from last non-leaf node to root
- Non-leaf node indices: n/2 - 1 down to 0
```

```cpp
// C++
void heapify(vector<int>& arr, int n, int i) {
    int largest = i;
    int left = 2 * i + 1;
    int right = 2 * i + 2;

    if (left < n && arr[left] > arr[largest])
        largest = left;

    if (right < n && arr[right] > arr[largest])
        largest = right;

    if (largest != i) {
        swap(arr[i], arr[largest]);
        // Recursively fix the subtree rooted at the swapped child —
        // a single swap may have violated the heap property one level down
        heapify(arr, n, largest);
    }
}

void buildHeap(vector<int>& arr) {
    int n = arr.size();

    // Start from last non-leaf node (n/2 - 1) and work upward;
    // leaf nodes are trivially valid heaps so we skip them.
    // This bottom-up approach is O(n), unlike n individual insertions which is O(n log n),
    // because most heapify calls operate near the leaves where depth is small
    for (int i = n / 2 - 1; i >= 0; i--) {
        heapify(arr, n, i);
    }
}
```

```python
def heapify(arr, n, i):
    largest = i
    left = 2 * i + 1
    right = 2 * i + 2

    if left < n and arr[left] > arr[largest]:
        largest = left

    if right < n and arr[right] > arr[largest]:
        largest = right

    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]
        heapify(arr, n, largest)

def build_heap(arr):
    n = len(arr)

    for i in range(n // 2 - 1, -1, -1):
        heapify(arr, n, i)
```

---

## 3. Heap Sort

### Algorithm

```
1. Convert array to max heap
2. Swap root (max) with last element
3. Reduce heap size and heapify
4. Repeat

[4, 10, 3, 5, 1]

Max heap: [10, 5, 3, 4, 1]

Steps:
[1, 5, 3, 4, | 10]  ← 10 sorted
[5, 4, 3, 1, | 10]  ← heapify
[1, 4, 3, | 5, 10]  ← 5 sorted
[4, 1, 3, | 5, 10]  ← heapify
...
[1, 3, 4, 5, 10]    ← complete

Time: O(n log n)
Space: O(1) - in-place sorting
```

```cpp
// C++
void heapSort(vector<int>& arr) {
    int n = arr.size();

    // Build max heap
    for (int i = n / 2 - 1; i >= 0; i--)
        heapify(arr, n, i);

    // Extract one by one
    for (int i = n - 1; i > 0; i--) {
        swap(arr[0], arr[i]);
        heapify(arr, i, 0);
    }
}
```

```python
def heap_sort(arr):
    n = len(arr)

    # Phase 1: build a max-heap in-place in O(n);
    # after this the largest element is guaranteed to be at arr[0]
    for i in range(n // 2 - 1, -1, -1):
        heapify(arr, n, i)

    # Phase 2: repeatedly place the current maximum at the sorted tail.
    # Swapping root with arr[i] puts the max in its final position,
    # then heapify on the reduced heap restores the property for the
    # next iteration — achieves O(n log n) with O(1) extra space
    for i in range(n - 1, 0, -1):
        arr[0], arr[i] = arr[i], arr[0]  # Move current max to sorted region
        heapify(arr, i, 0)               # Re-heapify the unsorted prefix of length i

    return arr
```

---

## 4. Priority Queue

### STL/Library

```cpp
// C++ - priority_queue (default: max heap)
#include <queue>

priority_queue<int> maxPQ;
maxPQ.push(3);
maxPQ.push(1);
maxPQ.push(4);
maxPQ.top();  // 4
maxPQ.pop();

// Min heap
priority_queue<int, vector<int>, greater<int>> minPQ;

// Custom comparator
auto cmp = [](int a, int b) { return a > b; };  // min heap
priority_queue<int, vector<int>, decltype(cmp)> pq(cmp);
```

```python
import heapq

# Python heapq only supports min heap

# Min heap
min_heap = []
heapq.heappush(min_heap, 3)
heapq.heappush(min_heap, 1)
heapq.heappush(min_heap, 4)
heapq.heappop(min_heap)  # 1

# Max heap (negate values trick)
max_heap = []
heapq.heappush(max_heap, -3)  # Store as negative
heapq.heappush(max_heap, -1)
heapq.heappush(max_heap, -4)
-heapq.heappop(max_heap)  # 4 (negate back)

# Convert array to heap
arr = [3, 1, 4, 1, 5, 9]
heapq.heapify(arr)  # O(n)
```

---

## 5. Application Problems

### 5.1 Kth Largest/Smallest Element

```python
import heapq

def kth_largest(nums, k):
    # Method 1: Sorting O(n log n)
    return sorted(nums, reverse=True)[k - 1]

    # Method 2: Min heap (maintain size k) O(n log k)
    min_heap = []
    for num in nums:
        heapq.heappush(min_heap, num)
        if len(min_heap) > k:
            heapq.heappop(min_heap)
    return min_heap[0]
```

### 5.2 Top K Frequent Elements

```python
from collections import Counter
import heapq

def top_k_frequent(nums, k):
    count = Counter(nums)

    # Min heap based on frequency (maintain size k)
    min_heap = []
    for num, freq in count.items():
        heapq.heappush(min_heap, (freq, num))
        if len(min_heap) > k:
            heapq.heappop(min_heap)

    return [num for freq, num in min_heap]
```

### 5.3 Merge K Sorted Arrays

```python
import heapq

def merge_k_sorted(lists):
    min_heap = []
    result = []

    # Insert first element from each list
    for i, lst in enumerate(lists):
        if lst:
            heapq.heappush(min_heap, (lst[0], i, 0))

    while min_heap:
        val, list_idx, elem_idx = heapq.heappop(min_heap)
        result.append(val)

        # Insert next element
        if elem_idx + 1 < len(lists[list_idx]):
            next_val = lists[list_idx][elem_idx + 1]
            heapq.heappush(min_heap, (next_val, list_idx, elem_idx + 1))

    return result
```

### 5.4 Median Stream

```python
import heapq

class MedianFinder:
    def __init__(self):
        self.small = []  # Max heap (smaller half) — negated because heapq is min-heap only
        self.large = []  # Min heap (larger half)

    def add_num(self, num):
        # Always push to small first so the new element is evaluated against the
        # current median boundary before deciding which half it belongs to
        heapq.heappush(self.small, -num)
        # Move small's maximum to large — this balances the partition and ensures
        # every element in small is ≤ every element in large
        heapq.heappush(self.large, -heapq.heappop(self.small))

        # Rebalance sizes: small is allowed to hold one extra element (for odd counts),
        # but large must never be larger; the median is always at small's top
        if len(self.large) > len(self.small):
            heapq.heappush(self.small, -heapq.heappop(self.large))

    def find_median(self):
        # Odd total: small holds one more element — its top is the exact median
        if len(self.small) > len(self.large):
            return -self.small[0]
        # Even total: median is the average of the two middle elements
        return (-self.small[0] + self.large[0]) / 2
```

---

## 6. Practice Problems

### Recommended Problems

| Difficulty | Problem | Platform | Type |
|--------|------|--------|------|
| ⭐⭐ | [Max Heap](https://www.acmicpc.net/problem/11279) | BOJ | Heap basics |
| ⭐⭐ | [Kth Largest](https://leetcode.com/problems/kth-largest-element-in-an-array/) | LeetCode | Kth element |
| ⭐⭐ | [Say the Middle](https://www.acmicpc.net/problem/1655) | BOJ | Median |
| ⭐⭐⭐ | [Merge K Sorted Lists](https://leetcode.com/problems/merge-k-sorted-lists/) | LeetCode | Merge |
| ⭐⭐⭐ | [Top K Frequent](https://leetcode.com/problems/top-k-frequent-elements/) | LeetCode | Frequency |

---

## Heap Operation Complexity Summary

```
┌────────────┬─────────────┐
│ Operation  │ Time        │
├────────────┼─────────────┤
│ Max/Min    │ O(1)        │
│ Insert     │ O(log n)    │
│ Delete     │ O(log n)    │
│ Build Heap │ O(n)        │
│ Heap Sort  │ O(n log n)  │
└────────────┴─────────────┘
```

---

## Next Steps

- [11_Trie.md](./11_Trie.md) - Trie

---

## References

- [Heap Visualization](https://visualgo.net/en/heap)
- Introduction to Algorithms (CLRS) - Chapter 6
