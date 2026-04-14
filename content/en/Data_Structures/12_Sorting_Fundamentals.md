# Sorting Fundamentals

**Previous**: [Strings as Data Structures](./11_Strings_as_Data_Structures.md) | **Next**: [Searching Fundamentals](./13_Searching_Fundamentals.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Implement bubble sort, selection sort, and insertion sort
2. Implement merge sort and quick sort with divide-and-conquer
3. Analyze time and space complexity of each sorting algorithm
4. Explain the concept of stability in sorting
5. Understand the O(n log n) lower bound for comparison-based sorting
6. Use Python's built-in `sorted()` and `list.sort()` effectively
7. Choose the right sorting algorithm based on data characteristics

---

**Sorting** is the process of arranging elements in a defined order (typically ascending or descending). It is one of the most studied problems in computer science because sorted data enables efficient searching, merging, and analysis.

## Classification of Sorting Algorithms

| Property | Description |
|----------|-------------|
| **Comparison-based** | Compares elements pairwise (bubble, merge, quick) |
| **Non-comparison** | Uses element properties (counting, radix, bucket) |
| **Stable** | Preserves relative order of equal elements |
| **In-place** | Uses O(1) extra space |
| **Adaptive** | Runs faster on partially sorted input |

## Bubble Sort

Repeatedly swap adjacent elements if they are in the wrong order:

```
Pass 1: [5, 3, 8, 1, 2]
         ^--^               3, 5 swap
        [3, 5, 8, 1, 2]
            ^--^            5, 8 no swap
        [3, 5, 8, 1, 2]
               ^--^         1, 8 swap
        [3, 5, 1, 8, 2]
                  ^--^      2, 8 swap
        [3, 5, 1, 2, 8]    (8 bubbled to end)

Pass 2: [3, 5, 1, 2, 8]
        [3, 1, 2, 5, 8]    (5 bubbled to position)

Pass 3: [1, 2, 3, 5, 8]    Sorted!
```

```python
def bubble_sort(arr):
    """Bubble sort -- O(n^2) time, O(1) space, stable."""
    n = len(arr)
    for i in range(n):
        swapped = False
        for j in range(n - 1 - i):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True
        if not swapped:  # Optimization: already sorted
            break
    return arr
```

## Selection Sort

Find the minimum element and swap it to the front:

```
[5, 3, 8, 1, 2]   min=1 at idx 3, swap with idx 0
[1, 3, 8, 5, 2]   min=2 at idx 4, swap with idx 1
[1, 2, 8, 5, 3]   min=3 at idx 4, swap with idx 2
[1, 2, 3, 5, 8]   min=5 at idx 3, already correct
[1, 2, 3, 5, 8]   Sorted!
```

```python
def selection_sort(arr):
    """Selection sort -- O(n^2) time, O(1) space, NOT stable."""
    n = len(arr)
    for i in range(n):
        min_idx = i
        for j in range(i + 1, n):
            if arr[j] < arr[min_idx]:
                min_idx = j
        arr[i], arr[min_idx] = arr[min_idx], arr[i]
    return arr
```

## Insertion Sort

Build the sorted portion one element at a time, inserting each into its correct position:

```
[5, 3, 8, 1, 2]   Insert 3: shift 5 right
[3, 5, 8, 1, 2]   Insert 8: already in place
[3, 5, 8, 1, 2]   Insert 1: shift 8, 5, 3 right
[1, 3, 5, 8, 2]   Insert 2: shift 8, 5, 3 right
[1, 2, 3, 5, 8]   Sorted!
```

```python
def insertion_sort(arr):
    """Insertion sort -- O(n^2) time, O(1) space, stable, adaptive."""
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
    return arr
```

**Why insertion sort matters**: It is O(n) on nearly sorted data, and is used as the base case in hybrid algorithms like Timsort.

## Merge Sort

Divide the array in half, sort each half, and merge the sorted halves:

```
Split:
[5, 3, 8, 1, 2]
[5, 3, 8]  |  [1, 2]
[5, 3] [8] |  [1] [2]
[5] [3]    |

Merge:
[3, 5] [8] |  [1, 2]
[3, 5, 8]  |  [1, 2]
[1, 2, 3, 5, 8]
```

```python
def merge_sort(arr):
    """Merge sort -- O(n log n) time, O(n) space, stable."""
    if len(arr) <= 1:
        return arr
    
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    
    return merge(left, right)

def merge(left, right):
    """Merge two sorted arrays into one sorted array."""
    result = []
    i = j = 0
    
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:  # <= ensures stability
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    
    result.extend(left[i:])
    result.extend(right[j:])
    return result
```

### In-Place Merge Sort (reduces extra space)

```python
def merge_sort_inplace(arr, left=0, right=None):
    """In-place merge sort variant."""
    if right is None:
        right = len(arr) - 1
    if left >= right:
        return
    
    mid = (left + right) // 2
    merge_sort_inplace(arr, left, mid)
    merge_sort_inplace(arr, mid + 1, right)
    merge_inplace(arr, left, mid, right)

def merge_inplace(arr, left, mid, right):
    temp = arr[left:mid + 1]
    i = 0
    j = mid + 1
    k = left
    
    while i < len(temp) and j <= right:
        if temp[i] <= arr[j]:
            arr[k] = temp[i]
            i += 1
        else:
            arr[k] = arr[j]
            j += 1
        k += 1
    
    while i < len(temp):
        arr[k] = temp[i]
        i += 1
        k += 1
```

## Quick Sort

Choose a pivot, partition elements around it, and recursively sort each partition:

```
Pivot = 5:
[3, 8, 1, 5, 2, 7, 4, 6]
         partition
[3, 1, 2, 4]  [5]  [8, 7, 6]
  less than     =   greater than

Recurse on each partition...
```

```python
def quick_sort(arr):
    """Quick sort -- O(n log n) average, O(n^2) worst, O(log n) space."""
    if len(arr) <= 1:
        return arr
    
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    
    return quick_sort(left) + middle + quick_sort(right)
```

### In-Place Quick Sort (Lomuto Partition)

```python
def quick_sort_inplace(arr, low=0, high=None):
    """In-place quick sort using Lomuto partition."""
    if high is None:
        high = len(arr) - 1
    if low < high:
        pivot_idx = partition(arr, low, high)
        quick_sort_inplace(arr, low, pivot_idx - 1)
        quick_sort_inplace(arr, pivot_idx + 1, high)

def partition(arr, low, high):
    """Lomuto partition scheme."""
    pivot = arr[high]
    i = low - 1
    
    for j in range(low, high):
        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
    
    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return i + 1
```

### Choosing a Good Pivot

```python
def median_of_three(arr, low, high):
    """Choose median of first, middle, last as pivot."""
    mid = (low + high) // 2
    if arr[low] > arr[mid]:
        arr[low], arr[mid] = arr[mid], arr[low]
    if arr[low] > arr[high]:
        arr[low], arr[high] = arr[high], arr[low]
    if arr[mid] > arr[high]:
        arr[mid], arr[high] = arr[high], arr[mid]
    # Median is now at mid
    arr[mid], arr[high - 1] = arr[high - 1], arr[mid]
    return arr[high - 1]
```

## Comparison Summary

| Algorithm | Best | Average | Worst | Space | Stable |
|-----------|------|---------|-------|-------|--------|
| Bubble | O(n) | O(n^2) | O(n^2) | O(1) | Yes |
| Selection | O(n^2) | O(n^2) | O(n^2) | O(1) | No |
| Insertion | **O(n)** | O(n^2) | O(n^2) | O(1) | Yes |
| Merge | O(n log n) | O(n log n) | O(n log n) | O(n) | Yes |
| Quick | O(n log n) | O(n log n) | O(n^2) | O(log n) | No |
| Heap | O(n log n) | O(n log n) | O(n log n) | O(1) | No |
| Timsort | **O(n)** | O(n log n) | O(n log n) | O(n) | Yes |

## The O(n log n) Lower Bound

Any comparison-based sorting algorithm must make at least O(n log n) comparisons in the worst case. This is because:

```
For n elements, there are n! possible orderings.
A comparison tree has at most 2^h leaves for height h.
We need 2^h >= n!, so h >= log2(n!) = O(n log n).
```

Non-comparison sorts (counting sort, radix sort) can break this bound by exploiting element properties.

## Python's `sorted()` and Timsort

Python uses **Timsort**, a hybrid of merge sort and insertion sort:

```python
# sorted() returns a new list
sorted([3, 1, 4, 1, 5])      # [1, 1, 3, 4, 5]
sorted([3, 1, 4], reverse=True)  # [4, 3, 1]

# list.sort() sorts in-place
nums = [3, 1, 4]
nums.sort()  # nums is now [1, 3, 4]

# Custom key function
words = ["banana", "apple", "cherry"]
sorted(words, key=len)  # ['apple', 'banana', 'cherry']

# Sorting objects
students = [("Alice", 88), ("Bob", 95), ("Charlie", 88)]
sorted(students, key=lambda s: (-s[1], s[0]))
# [('Bob', 95), ('Alice', 88), ('Charlie', 88)]
```

## Summary

| Concept | Key Takeaway |
|---------|-------------|
| Bubble sort | Simple but slow; good for teaching |
| Selection sort | Minimal swaps; not stable |
| Insertion sort | Great for small/nearly-sorted data |
| Merge sort | Guaranteed O(n log n); stable; needs O(n) space |
| Quick sort | Fast in practice; O(n^2) worst case; not stable |
| Stability | Preserves relative order of equal elements |
| Lower bound | Comparison sorts cannot beat O(n log n) |
| Timsort | Python's built-in; hybrid merge+insertion |

---

**Next**: [Searching Fundamentals](./13_Searching_Fundamentals.md) -- Learn efficient searching techniques.
