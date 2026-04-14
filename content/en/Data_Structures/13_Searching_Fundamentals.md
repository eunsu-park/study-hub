# Searching Fundamentals

**Previous**: [Sorting Fundamentals](./12_Sorting_Fundamentals.md) | **Next**: [Choosing the Right Data Structure](./14_Choosing_the_Right_Data_Structure.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Implement and analyze linear search
2. Implement binary search and its common variants
3. Avoid off-by-one errors in binary search implementations
4. Apply binary search to problems beyond simple lookup (lower/upper bound)
5. Use hash-based search for O(1) average-case lookups
6. Combine searching with sorting for efficient query processing
7. Choose the appropriate search strategy based on data and query patterns

---

**Searching** is the process of finding a specific element or determining its absence in a collection. The choice of search algorithm depends on the data structure, whether the data is sorted, and how many queries will be performed.

## Linear Search

Examine every element until the target is found or the end is reached:

```
Search for 7 in [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 7]:

  [3] [1] [4] [1] [5] [9] [2] [6] [5] [3] [7]
   ^   ^   ^   ^   ^   ^   ^   ^   ^   ^   ^
   1   2   3   4   5   6   7   8   9  10  11 comparisons
                                          Found at index 10!
```

```python
def linear_search(arr, target):
    """Linear search -- O(n) time, O(1) space.
    
    Returns the index of target, or -1 if not found.
    """
    for i, val in enumerate(arr):
        if val == target:
            return i
    return -1

def linear_search_all(arr, target):
    """Find all occurrences -- O(n)."""
    return [i for i, val in enumerate(arr) if val == target]
```

| Case | Comparisons |
|------|------------|
| Best | 1 (found at beginning) |
| Average | n/2 |
| Worst | n (not found or at end) |

**When to use**: Unsorted data, small collections, linked lists, one-time search.

## Binary Search

Requires **sorted** data. Repeatedly halve the search space:

```
Search for 7 in [1, 2, 3, 4, 5, 6, 7, 8, 9]:

Step 1: [1  2  3  4 |5| 6  7  8  9]   mid=5, 7 > 5, go right
Step 2:              [6 |7| 8  9]      mid=7, found!

Only 2 comparisons vs 7 for linear search!
```

```python
def binary_search(arr, target):
    """Binary search -- O(log n) time, O(1) space.
    
    arr must be sorted in ascending order.
    Returns the index of target, or -1 if not found.
    """
    left, right = 0, len(arr) - 1
    
    while left <= right:
        mid = left + (right - left) // 2  # Avoids overflow
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1
```

### Recursive Binary Search

```python
def binary_search_recursive(arr, target, left=0, right=None):
    """Recursive binary search -- O(log n) time, O(log n) space."""
    if right is None:
        right = len(arr) - 1
    
    if left > right:
        return -1
    
    mid = left + (right - left) // 2
    if arr[mid] == target:
        return mid
    elif arr[mid] < target:
        return binary_search_recursive(arr, target, mid + 1, right)
    else:
        return binary_search_recursive(arr, target, left, mid - 1)
```

### Common Pitfalls

```python
# WRONG: Integer overflow in some languages (not Python)
# mid = (left + right) // 2  # Can overflow if left+right > INT_MAX

# CORRECT:
mid = left + (right - left) // 2

# WRONG: Infinite loop
# while left < right:  # Misses the case when left == right
#     ...

# CORRECT:
while left <= right:
    ...
```

## Binary Search Variants

### Lower Bound (bisect_left)

Find the leftmost position where target could be inserted to maintain sorted order:

```python
def lower_bound(arr, target):
    """Find first position >= target -- O(log n).
    
    >>> lower_bound([1, 2, 4, 4, 4, 6, 8], 4)
    2
    >>> lower_bound([1, 2, 4, 4, 4, 6, 8], 5)
    5
    """
    left, right = 0, len(arr)
    while left < right:
        mid = left + (right - left) // 2
        if arr[mid] < target:
            left = mid + 1
        else:
            right = mid
    return left
```

### Upper Bound (bisect_right)

Find the first position where target could be inserted after all existing occurrences:

```python
def upper_bound(arr, target):
    """Find first position > target -- O(log n).
    
    >>> upper_bound([1, 2, 4, 4, 4, 6, 8], 4)
    5
    """
    left, right = 0, len(arr)
    while left < right:
        mid = left + (right - left) // 2
        if arr[mid] <= target:
            left = mid + 1
        else:
            right = mid
    return left
```

### Count Occurrences

```python
def count_occurrences(arr, target):
    """Count occurrences using binary search -- O(log n).
    
    >>> count_occurrences([1, 2, 4, 4, 4, 6, 8], 4)
    3
    """
    return upper_bound(arr, target) - lower_bound(arr, target)
```

### First and Last Position

```python
def first_and_last(arr, target):
    """Find the first and last occurrence -- O(log n).
    
    >>> first_and_last([1, 2, 4, 4, 4, 6, 8], 4)
    (2, 4)
    """
    lo = lower_bound(arr, target)
    if lo >= len(arr) or arr[lo] != target:
        return (-1, -1)
    hi = upper_bound(arr, target) - 1
    return (lo, hi)
```

## Binary Search on Answer

Binary search is not limited to arrays. You can binary search on the **answer** whenever the problem has a monotonic property:

```python
def sqrt_integer(n):
    """Find floor(sqrt(n)) using binary search.
    
    >>> sqrt_integer(8)
    2
    >>> sqrt_integer(16)
    4
    """
    if n < 2:
        return n
    left, right = 1, n // 2
    while left <= right:
        mid = left + (right - left) // 2
        if mid * mid == n:
            return mid
        elif mid * mid < n:
            left = mid + 1
        else:
            right = mid - 1
    return right

def min_capacity_ship(weights, days):
    """Find minimum ship capacity to deliver all packages in `days` days.
    
    Binary search on the answer (capacity).
    """
    def can_ship(capacity):
        day_count = 1
        current_load = 0
        for w in weights:
            if current_load + w > capacity:
                day_count += 1
                current_load = 0
            current_load += w
        return day_count <= days
    
    left = max(weights)
    right = sum(weights)
    while left < right:
        mid = left + (right - left) // 2
        if can_ship(mid):
            right = mid
        else:
            left = mid + 1
    return left
```

## Hash-Based Search

Using a hash table for O(1) average-case lookups:

```python
# Python set for membership testing
lookup_set = set(large_list)
if target in lookup_set:  # O(1) average
    print("Found!")

# Python dict for key-value lookup
lookup_dict = {item.id: item for item in items}
result = lookup_dict.get(target_id)  # O(1) average
```

### When Hash vs Binary Search

| Criteria | Hash-Based | Binary Search |
|----------|-----------|---------------|
| Preprocessing | O(n) build hash table | O(n log n) sort |
| Single query | O(1) average | O(log n) |
| Many queries | O(1) each | O(log n) each |
| Range queries | Not supported | Efficient |
| Space | O(n) | O(1) extra (if sorted) |
| Ordered results | No | Yes |
| Worst case | O(n) | O(log n) |

## Python's `bisect` Module

```python
import bisect

sorted_list = [1, 3, 5, 7, 9, 11, 13, 15]

# Find insertion point
bisect.bisect_left(sorted_list, 7)   # 3
bisect.bisect_right(sorted_list, 7)  # 4

# Insert while maintaining order
bisect.insort(sorted_list, 8)
# [1, 3, 5, 7, 8, 9, 11, 13, 15]

# Binary search using bisect
def binary_search_bisect(arr, target):
    i = bisect.bisect_left(arr, target)
    if i < len(arr) and arr[i] == target:
        return i
    return -1
```

## Search Comparison

| Algorithm | Time | Space | Sorted Required | Best For |
|-----------|------|-------|----------------|----------|
| Linear | O(n) | O(1) | No | Small/unsorted data |
| Binary | O(log n) | O(1) | **Yes** | Sorted arrays, range queries |
| Hash | O(1)* | O(n) | No | Frequent lookups, exact match |
| BST | O(log n)** | O(n) | Maintained | Dynamic sorted data |
| Interpolation | O(log log n)*** | O(1) | Yes, uniform | Uniformly distributed data |

*Average case | **Balanced tree | ***Uniformly distributed

## Summary

| Concept | Key Takeaway |
|---------|-------------|
| Linear search | O(n), works on any data |
| Binary search | O(log n), requires sorted data |
| Lower/upper bound | Find insertion points, count occurrences |
| Binary search on answer | Apply when problem is monotonic |
| Hash-based search | O(1) average, O(n) space |
| `bisect` module | Python's built-in binary search utilities |
| Trade-off | Hash for exact match; binary for range queries |

---

**Next**: [Choosing the Right Data Structure](./14_Choosing_the_Right_Data_Structure.md) -- Synthesize everything into a practical decision framework.
