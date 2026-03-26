# Sorting Algorithms

## Learning Objectives

After completing this lesson, you will be able to:

1. Compare sorting algorithms (Bubble, Selection, Insertion, Merge, Quick, Heap, Counting) by their best, average, and worst-case time complexity and space complexity
2. Distinguish between stable and unstable sorting algorithms and explain when stability matters
3. Implement Merge Sort and Quick Sort from scratch, articulating the divide-and-conquer strategy behind each
4. Explain why Counting Sort achieves O(n+k) time and identify the constraints that make it applicable
5. Analyze Quick Sort's worst-case behavior and describe pivot-selection strategies that mitigate it
6. Select the most appropriate sorting algorithm for a given problem based on input size, value range, memory constraints, and stability requirements

---

## Overview

Sorting is a fundamental yet critical algorithm for arranging data in a specific order. This lesson covers the principles, implementations, and time/space complexity of various sorting algorithms.

---

## Table of Contents

1. [Sorting Algorithm Comparison](#1-sorting-algorithm-comparison)
2. [Bubble Sort](#2-bubble-sort)
3. [Selection Sort](#3-selection-sort)
4. [Insertion Sort](#4-insertion-sort)
5. [Merge Sort](#5-merge-sort)
6. [Quick Sort](#6-quick-sort)
7. [Heap Sort](#7-heap-sort)
8. [Counting Sort](#8-counting-sort)
9. [Choosing a Sorting Algorithm](#9-choosing-a-sorting-algorithm)
10. [Practice Problems](#10-practice-problems)

---

## 1. Sorting Algorithm Comparison

### Complexity Comparison Table

```
┌─────────────┬───────────────────────────────────┬─────────┬──────────┐
│  Algorithm  │         Time Complexity           │  Space  │ Stable   │
│             │  Best    │  Average │  Worst     │ Compx.  │          │
├─────────────┼──────────┼──────────┼────────────┼─────────┼──────────┤
│ Bubble Sort │ O(n)     │ O(n²)    │ O(n²)      │ O(1)    │ Stable   │
│ Selection   │ O(n²)    │ O(n²)    │ O(n²)      │ O(1)    │ Unstable │
│ Insertion   │ O(n)     │ O(n²)    │ O(n²)      │ O(1)    │ Stable   │
│ Merge Sort  │ O(nlogn) │ O(nlogn) │ O(nlogn)   │ O(n)    │ Stable   │
│ Quick Sort  │ O(nlogn) │ O(nlogn) │ O(n²)      │ O(logn) │ Unstable │
│ Heap Sort   │ O(nlogn) │ O(nlogn) │ O(nlogn)   │ O(1)    │ Unstable │
│ Counting    │ O(n+k)   │ O(n+k)   │ O(n+k)     │ O(k)    │ Stable   │
└─────────────┴──────────┴──────────┴────────────┴─────────┴──────────┘
* k: range of values
```

### Stable Sort

```
Stable Sort: Elements with equal values maintain their relative order after sorting

Example: Sort [(A,3), (B,1), (C,3)] by number

Stable Sort:   [(B,1), (A,3), (C,3)]  ← A before C (original order preserved)
Unstable Sort: [(B,1), (C,3), (A,3)]  ← Order may change
```

---

## 2. Bubble Sort

### Principle

```
Compare adjacent elements and swap; largest element "bubbles" to the end

Array: [5, 3, 8, 4, 2]

Pass 1: Compare/swap 5 and 3 → [3, 5, 8, 4, 2]
        Compare 5 and 8 (keep) → [3, 5, 8, 4, 2]
        Compare/swap 8 and 4 → [3, 5, 4, 8, 2]
        Compare/swap 8 and 2 → [3, 5, 4, 2, 8] ← 8 fixed

Pass 2: [3, 5, 4, 2, 8]
        → [3, 4, 2, 5, 8] ← 5 fixed

Pass 3: → [3, 2, 4, 5, 8] ← 4 fixed

Pass 4: → [2, 3, 4, 5, 8] ← Complete
```

### Implementation

```c
// C
void bubbleSort(int arr[], int n) {
    for (int i = 0; i < n - 1; i++) {
        // Inner loop stops at n - i - 1: after pass i, the last i elements are
        // already in their final sorted position — no need to re-examine them
        for (int j = 0; j < n - i - 1; j++) {
            if (arr[j] > arr[j + 1]) {
                int temp = arr[j];
                arr[j] = arr[j + 1];
                arr[j + 1] = temp;
            }
        }
    }
}

// Optimization: early termination if no swaps
void bubbleSortOptimized(int arr[], int n) {
    for (int i = 0; i < n - 1; i++) {
        int swapped = 0;

        for (int j = 0; j < n - i - 1; j++) {
            if (arr[j] > arr[j + 1]) {
                int temp = arr[j];
                arr[j] = arr[j + 1];
                arr[j + 1] = temp;
                swapped = 1;
            }
        }

        // If no swap occurred, the array is already sorted — this turns
        // best-case complexity from O(n²) to O(n) for nearly-sorted input
        if (!swapped) break;  // Already sorted
    }
}
```

```cpp
// C++
void bubbleSort(vector<int>& arr) {
    int n = arr.size();
    for (int i = 0; i < n - 1; i++) {
        bool swapped = false;

        for (int j = 0; j < n - i - 1; j++) {
            if (arr[j] > arr[j + 1]) {
                swap(arr[j], arr[j + 1]);
                swapped = true;
            }
        }

        if (!swapped) break;
    }
}
```

```python
# Python
def bubble_sort(arr):
    n = len(arr)

    for i in range(n - 1):
        swapped = False

        for j in range(n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True

        if not swapped:
            break

    return arr
```

---

## 3. Selection Sort

### Principle

```
Find minimum value at each step and swap with front position

Array: [5, 3, 8, 4, 2]

Pass 1: Minimum value 2 → swap with front
        [2, 3, 8, 4, 5] ← 2 fixed

Pass 2: Excluding [2], minimum value 3
        [2, 3, 8, 4, 5] ← 3 fixed (already in place)

Pass 3: Excluding [2, 3], minimum value 4
        [2, 3, 4, 8, 5] ← 4 fixed

Pass 4: Excluding [2, 3, 4], minimum value 5
        [2, 3, 4, 5, 8] ← Complete
```

### Implementation

```c
// C
void selectionSort(int arr[], int n) {
    for (int i = 0; i < n - 1; i++) {
        int minIdx = i;

        for (int j = i + 1; j < n; j++) {
            if (arr[j] < arr[minIdx]) {
                minIdx = j;
            }
        }

        if (minIdx != i) {
            int temp = arr[i];
            arr[i] = arr[minIdx];
            arr[minIdx] = temp;
        }
    }
}
```

```cpp
// C++
void selectionSort(vector<int>& arr) {
    int n = arr.size();

    for (int i = 0; i < n - 1; i++) {
        int minIdx = i;

        for (int j = i + 1; j < n; j++) {
            if (arr[j] < arr[minIdx]) {
                minIdx = j;
            }
        }

        if (minIdx != i) {
            swap(arr[i], arr[minIdx]);
        }
    }
}
```

```python
# Python
def selection_sort(arr):
    n = len(arr)

    for i in range(n - 1):
        min_idx = i

        for j in range(i + 1, n):
            if arr[j] < arr[min_idx]:
                min_idx = j

        arr[i], arr[min_idx] = arr[min_idx], arr[i]

    return arr
```

---

## 4. Insertion Sort

### Principle

```
Insert current element into proper position in already sorted portion

Array: [5, 3, 8, 4, 2]

Initial: [5] ← First element is sorted

Pass 1: Insert 3 → [3, 5]
        3 < 5, so shift 5 right and insert 3

Pass 2: Insert 8 → [3, 5, 8]
        8 > 5, so keep as is

Pass 3: Insert 4 → [3, 4, 5, 8]
        4 < 8, 4 < 5, 4 > 3 → shift 5,8 and insert

Pass 4: Insert 2 → [2, 3, 4, 5, 8]
        2 < 8, 2 < 5, 2 < 4, 2 < 3 → shift all and insert
```

### Implementation

```c
// C
void insertionSort(int arr[], int n) {
    for (int i = 1; i < n; i++) {
        int key = arr[i];
        int j = i - 1;

        // Shift elements greater than key to the right
        while (j >= 0 && arr[j] > key) {
            arr[j + 1] = arr[j];
            j--;
        }

        arr[j + 1] = key;
    }
}
```

```cpp
// C++
void insertionSort(vector<int>& arr) {
    int n = arr.size();

    for (int i = 1; i < n; i++) {
        int key = arr[i];
        int j = i - 1;

        while (j >= 0 && arr[j] > key) {
            arr[j + 1] = arr[j];
            j--;
        }

        arr[j + 1] = key;
    }
}
```

```python
# Python
def insertion_sort(arr):
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1

        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1

        arr[j + 1] = key

    return arr
```

### Advantages of Insertion Sort

```
1. O(n) on nearly sorted arrays - very fast
2. Efficient for small datasets
3. Stable sort
4. In-place sort (no extra memory needed)
5. Online algorithm (can sort as data arrives)
```

---

## 5. Merge Sort

### Principle

```
Divide and conquer: split array in half, sort each, then merge

Array: [5, 3, 8, 4, 2, 7, 1, 6]

Divide:
         [5, 3, 8, 4, 2, 7, 1, 6]
              /              \
       [5, 3, 8, 4]      [2, 7, 1, 6]
        /      \          /      \
     [5, 3]  [8, 4]    [2, 7]  [1, 6]
     /   \    /   \    /   \    /   \
   [5]  [3] [8]  [4] [2]  [7] [1]  [6]

Merge:
   [5]  [3] [8]  [4] [2]  [7] [1]  [6]
     \  /     \  /     \  /     \  /
    [3, 5]  [4, 8]   [2, 7]   [1, 6]
        \    /           \    /
      [3, 4, 5, 8]    [1, 2, 6, 7]
            \              /
         [1, 2, 3, 4, 5, 6, 7, 8]
```

### Implementation

```c
// C
void merge(int arr[], int left, int mid, int right) {
    int n1 = mid - left + 1;
    int n2 = right - mid;

    // Temporary arrays are necessary — merging directly into arr would
    // overwrite elements still needed for comparison on the other side
    int* L = (int*)malloc(n1 * sizeof(int));
    int* R = (int*)malloc(n2 * sizeof(int));

    for (int i = 0; i < n1; i++) L[i] = arr[left + i];
    for (int i = 0; i < n2; i++) R[i] = arr[mid + 1 + i];

    int i = 0, j = 0, k = left;

    while (i < n1 && j < n2) {
        // <= (not <) preserves stability: equal elements from the left subarray
        // come first, maintaining their original relative order
        if (L[i] <= R[j]) {
            arr[k++] = L[i++];
        } else {
            arr[k++] = R[j++];
        }
    }

    // Drain any remaining elements — at most one of these loops will execute
    while (i < n1) arr[k++] = L[i++];
    while (j < n2) arr[k++] = R[j++];

    free(L);
    free(R);
}

void mergeSort(int arr[], int left, int right) {
    if (left < right) {
        // Overflow-safe midpoint: avoids integer overflow when left + right > INT_MAX
        int mid = left + (right - left) / 2;

        mergeSort(arr, left, mid);
        mergeSort(arr, mid + 1, right);
        merge(arr, left, mid, right);
    }
}
```

```cpp
// C++
void merge(vector<int>& arr, int left, int mid, int right) {
    vector<int> temp(right - left + 1);

    int i = left, j = mid + 1, k = 0;

    while (i <= mid && j <= right) {
        if (arr[i] <= arr[j]) {
            temp[k++] = arr[i++];
        } else {
            temp[k++] = arr[j++];
        }
    }

    while (i <= mid) temp[k++] = arr[i++];
    while (j <= right) temp[k++] = arr[j++];

    for (int i = 0; i < k; i++) {
        arr[left + i] = temp[i];
    }
}

void mergeSort(vector<int>& arr, int left, int right) {
    if (left < right) {
        int mid = left + (right - left) / 2;

        mergeSort(arr, left, mid);
        mergeSort(arr, mid + 1, right);
        merge(arr, left, mid, right);
    }
}
```

```python
# Python
def merge_sort(arr):
    if len(arr) <= 1:
        return arr

    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])

    return merge(left, right)

def merge(left, right):
    result = []
    i = j = 0

    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1

    result.extend(left[i:])
    result.extend(right[j:])

    return result
```

---

## 6. Quick Sort

### Principle

```
Choose pivot, partition elements smaller to left, larger to right

Array: [5, 3, 8, 4, 2, 7, 1, 6], pivot = 5 (first element)

Partitioning:
  pivot=5
  smaller: [3, 4, 2, 1]
  larger:  [8, 7, 6]

  → [3, 4, 2, 1] + [5] + [8, 7, 6]

Recursively sort left and right:
  [1, 2, 3, 4] + [5] + [6, 7, 8]

Result: [1, 2, 3, 4, 5, 6, 7, 8]
```

### Lomuto Partitioning

```
Use last element as pivot

Array: [5, 3, 8, 4, 2], pivot = 2

i = -1 (end of smaller-than-pivot region)

j=0: 5 > 2 → skip
j=1: 3 > 2 → skip
j=2: 8 > 2 → skip
j=3: 4 > 2 → skip

Move pivot to i+1 position:
[2, 3, 8, 4, 5]
 ↑
pivot position
```

### Implementation

```c
// C - Lomuto partitioning
int partition(int arr[], int low, int high) {
    int pivot = arr[high];
    // i starts one position before the subarray — it marks the boundary of the
    // "elements smaller than pivot" region; i increments only when we extend it
    int i = low - 1;

    for (int j = low; j < high; j++) {
        if (arr[j] < pivot) {
            // Extend the smaller-than-pivot region and swap the new element into it
            i++;
            int temp = arr[i];
            arr[i] = arr[j];
            arr[j] = temp;
        }
    }

    // Place pivot at its final position (i + 1): everything to the left is
    // smaller, everything to the right is larger — this guarantees O(n) partition
    int temp = arr[i + 1];
    arr[i + 1] = arr[high];
    arr[high] = temp;

    return i + 1;
}

void quickSort(int arr[], int low, int high) {
    if (low < high) {
        int pi = partition(arr, low, high);

        // Pivot is already in its final sorted position — exclude it from both
        // recursive calls to avoid an infinite loop and unnecessary comparisons
        quickSort(arr, low, pi - 1);
        quickSort(arr, pi + 1, high);
    }
}
```

```cpp
// C++
int partition(vector<int>& arr, int low, int high) {
    int pivot = arr[high];
    int i = low - 1;

    for (int j = low; j < high; j++) {
        if (arr[j] < pivot) {
            i++;
            swap(arr[i], arr[j]);
        }
    }

    swap(arr[i + 1], arr[high]);
    return i + 1;
}

void quickSort(vector<int>& arr, int low, int high) {
    if (low < high) {
        int pi = partition(arr, low, high);

        quickSort(arr, low, pi - 1);
        quickSort(arr, pi + 1, high);
    }
}
```

```python
# Python
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

# Usage
# arr = [5, 3, 8, 4, 2, 7, 1, 6]
# quick_sort(arr, 0, len(arr) - 1)
```

### Pivot Selection Strategies

```
1. First/last element: Simple but O(n²) on sorted arrays
2. Random pivot: Good average performance
3. Median of three: Median of first, middle, last elements

import random

def partition_random(arr, low, high):
    # Choose random pivot
    rand_idx = random.randint(low, high)
    arr[rand_idx], arr[high] = arr[high], arr[rand_idx]
    return partition(arr, low, high)
```

---

## 7. Heap Sort

### Principle

```
Build max heap, then extract root (maximum value) to sort

Visualize array as heap:
        [16]                  Index:
       /    \                    0
     [14]   [10]              1    2
     /  \   /  \            3  4  5  6
   [8] [7] [9] [3]
   /\
 [2][4]

Array: [16, 14, 10, 8, 7, 9, 3, 2, 4]

Sorting process:
1. Build max heap
2. Swap root (16) with last element → [16] fixed
3. Rebuild heap with remaining elements
4. Repeat
```

### Implementation

```c
// C
void heapify(int arr[], int n, int i) {
    int largest = i;
    // In a 0-indexed binary heap stored as an array, children of node i are at
    // 2*i+1 (left) and 2*i+2 (right); this arithmetic replaces an explicit tree
    int left = 2 * i + 1;
    int right = 2 * i + 2;

    if (left < n && arr[left] > arr[largest])
        largest = left;

    if (right < n && arr[right] > arr[largest])
        largest = right;

    if (largest != i) {
        int temp = arr[i];
        arr[i] = arr[largest];
        arr[largest] = temp;

        // Recursively fix the subtree that was disturbed by the swap —
        // a single swap may violate the heap property further down the tree
        heapify(arr, n, largest);
    }
}

void heapSort(int arr[], int n) {
    // Build max heap by heapifying from the last internal node upward —
    // leaf nodes are trivially valid heaps, so starting at n/2-1 skips them
    // and achieves O(n) build time (not O(n log n))
    for (int i = n / 2 - 1; i >= 0; i--) {
        heapify(arr, n, i);
    }

    // Repeatedly move the heap's root (current maximum) to the sorted suffix,
    // then restore the heap property over the remaining n-i-1 elements
    for (int i = n - 1; i > 0; i--) {
        int temp = arr[0];
        arr[0] = arr[i];
        arr[i] = temp;

        heapify(arr, i, 0);
    }
}
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
        heapify(arr, n, largest);
    }
}

void heapSort(vector<int>& arr) {
    int n = arr.size();

    // Build max heap
    for (int i = n / 2 - 1; i >= 0; i--) {
        heapify(arr, n, i);
    }

    // Extract elements one by one
    for (int i = n - 1; i > 0; i--) {
        swap(arr[0], arr[i]);
        heapify(arr, i, 0);
    }
}
```

```python
# Python
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

def heap_sort(arr):
    n = len(arr)

    # Build max heap
    for i in range(n // 2 - 1, -1, -1):
        heapify(arr, n, i)

    # Extract elements one by one
    for i in range(n - 1, 0, -1):
        arr[0], arr[i] = arr[i], arr[0]
        heapify(arr, i, 0)

    return arr
```

---

## 8. Counting Sort

### Principle

```
Count occurrences to sort (not comparison-based)
Condition: Elements are integers with limited range

Array: [4, 2, 2, 8, 3, 3, 1]
Range: 1~8

Count occurrences:
Value:  1  2  3  4  5  6  7  8
Count: [1, 2, 2, 1, 0, 0, 0, 1]

Cumulative sum:
       [1, 3, 5, 6, 6, 6, 6, 7]

Build result array (from back):
Element 1 → position 1 → result[0] = 1
Element 3 → position 5 → result[4] = 3
...

Result: [1, 2, 2, 3, 3, 4, 8]
```

### Implementation

```c
// C
void countingSort(int arr[], int n) {
    // Find maximum value
    int max = arr[0];
    for (int i = 1; i < n; i++) {
        if (arr[i] > max) max = arr[i];
    }

    // Count array
    int* count = (int*)calloc(max + 1, sizeof(int));

    for (int i = 0; i < n; i++) {
        count[arr[i]]++;
    }

    // Cumulative sum
    for (int i = 1; i <= max; i++) {
        count[i] += count[i - 1];
    }

    // Result array
    int* output = (int*)malloc(n * sizeof(int));

    for (int i = n - 1; i >= 0; i--) {
        output[count[arr[i]] - 1] = arr[i];
        count[arr[i]]--;
    }

    for (int i = 0; i < n; i++) {
        arr[i] = output[i];
    }

    free(count);
    free(output);
}
```

```cpp
// C++
void countingSort(vector<int>& arr) {
    if (arr.empty()) return;

    int maxVal = *max_element(arr.begin(), arr.end());
    int minVal = *min_element(arr.begin(), arr.end());
    int range = maxVal - minVal + 1;

    vector<int> count(range, 0);
    vector<int> output(arr.size());

    for (int x : arr) {
        count[x - minVal]++;
    }

    for (int i = 1; i < range; i++) {
        count[i] += count[i - 1];
    }

    for (int i = arr.size() - 1; i >= 0; i--) {
        output[count[arr[i] - minVal] - 1] = arr[i];
        count[arr[i] - minVal]--;
    }

    arr = output;
}
```

```python
# Python
def counting_sort(arr):
    if not arr:
        return arr

    max_val = max(arr)
    min_val = min(arr)
    # Shift values so min_val maps to index 0 — allows negative or large-offset
    # integers without allocating a huge array from 0 to max_val
    range_val = max_val - min_val + 1

    count = [0] * range_val
    output = [0] * len(arr)

    for x in arr:
        count[x - min_val] += 1

    # Convert counts to cumulative sums: count[i] now gives the 1-based position
    # of the *last* occurrence of value (min_val + i) in the output
    for i in range(1, range_val):
        count[i] += count[i - 1]

    # Traverse the input backward to achieve stability: among equal elements,
    # the one that appeared last in the input gets placed last in the output,
    # preserving their original relative order (same as a stable sort guarantees)
    for i in range(len(arr) - 1, -1, -1):
        output[count[arr[i] - min_val] - 1] = arr[i]
        count[arr[i] - min_val] -= 1  # Decrement so the next equal element goes one slot earlier

    return output
```

---

## 9. Choosing a Sorting Algorithm

### Recommendations by Situation

```
┌────────────────────────────────┬─────────────────────────┐
│ Situation                      │ Recommended Algorithm   │
├────────────────────────────────┼─────────────────────────┤
│ Small data (n < 50)            │ Insertion Sort          │
│ Nearly sorted data             │ Insertion Sort          │
│ Memory constrained             │ Heap Sort, Quick Sort   │
│ Stable sort required           │ Merge Sort              │
│ Worst case O(n log n) guarantee│ Merge Sort, Heap Sort   │
│ Fastest on average             │ Quick Sort              │
│ Integers, limited range        │ Counting Sort           │
│ General purpose library        │ Timsort (Merge+Insert)  │
└────────────────────────────────┴─────────────────────────┘
```

### Practical Usage

```cpp
// C++ STL
#include <algorithm>

vector<int> arr = {5, 3, 8, 4, 2};

// Ascending order
sort(arr.begin(), arr.end());

// Descending order
sort(arr.begin(), arr.end(), greater<int>());

// Custom comparison function
sort(arr.begin(), arr.end(), [](int a, int b) {
    return abs(a) < abs(b);  // Sort by absolute value
});

// Stable sort
stable_sort(arr.begin(), arr.end());
```

```python
# Python
arr = [5, 3, 8, 4, 2]

# Ascending order (Timsort)
sorted_arr = sorted(arr)

# Descending order
sorted_arr = sorted(arr, reverse=True)

# Custom key
sorted_arr = sorted(arr, key=lambda x: abs(x))

# In-place sort
arr.sort()
```

---

## 10. Practice Problems

### Problem 1: Kth Largest Element

Find the Kth largest element in O(n) average time without full sorting.

<details>
<summary>Hint</summary>

Quick Select algorithm: uses partitioning from Quick Sort

</details>

<details>
<summary>Solution Code</summary>

```python
def find_kth_largest(arr, k):
    k = len(arr) - k  # kth largest = (n-k)th smallest

    def quick_select(left, right):
        pivot = arr[right]
        i = left

        for j in range(left, right):
            if arr[j] < pivot:
                arr[i], arr[j] = arr[j], arr[i]
                i += 1

        arr[i], arr[right] = arr[right], arr[i]

        if i == k:
            return arr[i]
        elif i < k:
            return quick_select(i + 1, right)
        else:
            return quick_select(left, i - 1)

    return quick_select(0, len(arr) - 1)
```

</details>

### Problem 2: Color Sort (Dutch National Flag)

Sort an array of 0s, 1s, and 2s in one pass.

```
Input:  [2, 0, 2, 1, 1, 0]
Output: [0, 0, 1, 1, 2, 2]
```

<details>
<summary>Solution Code</summary>

```python
def sort_colors(arr):
    low, mid, high = 0, 0, len(arr) - 1

    while mid <= high:
        if arr[mid] == 0:
            arr[low], arr[mid] = arr[mid], arr[low]
            low += 1
            mid += 1
        elif arr[mid] == 1:
            mid += 1
        else:  # arr[mid] == 2
            arr[mid], arr[high] = arr[high], arr[mid]
            high -= 1

    return arr

# Time: O(n), Space: O(1)
```

</details>

### Recommended Problems

| Difficulty | Problem | Platform | Type |
|--------|------|--------|------|
| ⭐ | [Sort Numbers](https://www.acmicpc.net/problem/2750) | Baekjoon | Basic Sort |
| ⭐ | [Sort Colors](https://leetcode.com/problems/sort-colors/) | LeetCode | 3-way Partition |
| ⭐⭐ | [Sort Numbers 2](https://www.acmicpc.net/problem/2751) | Baekjoon | O(n log n) |
| ⭐⭐ | [Merge Intervals](https://leetcode.com/problems/merge-intervals/) | LeetCode | Sort Application |
| ⭐⭐ | [Kth Largest Element](https://leetcode.com/problems/kth-largest-element-in-an-array/) | LeetCode | Quick Select |
| ⭐⭐⭐ | [Median of Two Sorted Arrays](https://leetcode.com/problems/median-of-two-sorted-arrays/) | LeetCode | Binary Search |

---

## Next Steps

- [06_Searching_Algorithms.md](./06_Searching_Algorithms.md) - Binary Search, Parametric Search

---

## References

- [Sorting Algorithms Visualized](https://www.toptal.com/developers/sorting-algorithms)
- [VisuAlgo - Sorting](https://visualgo.net/en/sorting)
- Introduction to Algorithms (CLRS) - Chapter 7, 8
