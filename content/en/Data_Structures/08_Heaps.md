# Heaps

**Previous**: [Binary Search Trees](./07_Binary_Search_Trees.md) | **Next**: [Graphs Basics](./09_Graphs_Basics.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Define the heap property (min-heap and max-heap)
2. Implement a binary heap using an array
3. Perform heapify-up (sift-up) and heapify-down (sift-down) operations
4. Build a heap from an unordered array in O(n) time
5. Implement a priority queue using a heap
6. Use Python's `heapq` module for heap operations
7. Apply heaps to solve top-K and merge-K-sorted problems

---

A **heap** is a complete binary tree that satisfies the **heap property**. It is the foundation of the **priority queue** abstract data type and is used in heap sort, graph algorithms, and scheduling.

## Heap Property

```
Min-Heap: parent <= children       Max-Heap: parent >= children

       [1]                               [9]
      /   \                             /   \
    [3]   [2]                         [7]   [8]
   / \   /                           / \   /
 [7] [4][5]                        [3] [6][5]

Root is always the minimum!        Root is always the maximum!
```

**Key difference from BST**: A heap does NOT maintain sorted order between left and right children. Only the parent-child relationship matters.

## Array Representation

A complete binary tree maps perfectly to an array (no wasted space):

```
          [1]                    Index:  0  1  2  3  4  5
         /   \                   Array: [1, 3, 2, 7, 4, 5]
       [3]   [2]
      / \   /
    [7] [4][5]

Parent of i:       (i - 1) // 2
Left child of i:   2 * i + 1
Right child of i:  2 * i + 2
```

## Min-Heap Implementation

```python
class MinHeap:
    """Binary min-heap using an array."""
    
    def __init__(self):
        self._data = []
    
    def __len__(self):
        return len(self._data)
    
    def is_empty(self):
        return len(self._data) == 0
    
    def peek(self):
        """Return the minimum element -- O(1)."""
        if self.is_empty():
            raise IndexError("peek at empty heap")
        return self._data[0]
    
    def push(self, val):
        """Insert a value -- O(log n)."""
        self._data.append(val)
        self._sift_up(len(self._data) - 1)
    
    def pop(self):
        """Remove and return the minimum -- O(log n)."""
        if self.is_empty():
            raise IndexError("pop from empty heap")
        self._swap(0, len(self._data) - 1)
        min_val = self._data.pop()
        if self._data:
            self._sift_down(0)
        return min_val
    
    def _sift_up(self, idx):
        """Restore heap property upward."""
        parent = (idx - 1) // 2
        while idx > 0 and self._data[idx] < self._data[parent]:
            self._swap(idx, parent)
            idx = parent
            parent = (idx - 1) // 2
    
    def _sift_down(self, idx):
        """Restore heap property downward."""
        size = len(self._data)
        while True:
            smallest = idx
            left = 2 * idx + 1
            right = 2 * idx + 2
            
            if left < size and self._data[left] < self._data[smallest]:
                smallest = left
            if right < size and self._data[right] < self._data[smallest]:
                smallest = right
            
            if smallest == idx:
                break
            
            self._swap(idx, smallest)
            idx = smallest
    
    def _swap(self, i, j):
        self._data[i], self._data[j] = self._data[j], self._data[i]
```

### Push (Sift-Up) Visualization

```
Push 2 into heap:
Step 1: Append to end     Step 2: Sift up (2 < 7)    Step 3: Sift up (2 < 3)
       [1]                       [1]                         [1]
      /   \                     /   \                       /   \
    [3]   [5]                 [3]   [5]                   [2]   [5]
   / \   /  \               / \   /  \                   / \   /  \
 [7] [4][8]  [2]           [2] [4][8] [7]              [3] [4][8] [7]
                            ^sifted up                   ^done (2 > 1)
```

### Pop (Sift-Down) Visualization

```
Pop minimum (1):
Step 1: Swap root & last   Step 2: Remove last    Step 3: Sift down
       [7]                       [7]                     [2]
      /   \                     /   \                   /   \
    [2]   [5]                 [2]   [5]               [3]   [5]
   / \   /                   / \                     / \
 [3] [4][8]  [1]removed    [3] [4]  [8]           [7] [4]  [8]
                             sift 7 down
```

## Building a Heap: O(n)

Building a heap by inserting elements one by one is O(n log n). But there is an O(n) method:

```python
def heapify(arr):
    """Convert an array into a min-heap in-place -- O(n)."""
    n = len(arr)
    # Start from the last non-leaf node, sift down each
    for i in range(n // 2 - 1, -1, -1):
        sift_down(arr, i, n)

def sift_down(arr, idx, size):
    while True:
        smallest = idx
        left = 2 * idx + 1
        right = 2 * idx + 2
        if left < size and arr[left] < arr[smallest]:
            smallest = left
        if right < size and arr[right] < arr[smallest]:
            smallest = right
        if smallest == idx:
            break
        arr[idx], arr[smallest] = arr[smallest], arr[idx]
        idx = smallest
```

**Why O(n)?** Most nodes are near the bottom and sift down a short distance. The total work is proportional to the sum of heights of all nodes, which is O(n).

## Heap Sort

```python
def heap_sort(arr):
    """Sort using a heap -- O(n log n) time, O(1) extra space."""
    n = len(arr)
    
    # Build max-heap
    for i in range(n // 2 - 1, -1, -1):
        sift_down_max(arr, i, n)
    
    # Extract elements one by one
    for i in range(n - 1, 0, -1):
        arr[0], arr[i] = arr[i], arr[0]  # Move max to end
        sift_down_max(arr, 0, i)         # Restore heap on reduced array

def sift_down_max(arr, idx, size):
    while True:
        largest = idx
        left = 2 * idx + 1
        right = 2 * idx + 2
        if left < size and arr[left] > arr[largest]:
            largest = left
        if right < size and arr[right] > arr[largest]:
            largest = right
        if largest == idx:
            break
        arr[idx], arr[largest] = arr[largest], arr[idx]
        idx = largest
```

## Priority Queue

A priority queue is an ADT where each element has a priority, and the highest-priority element is dequeued first:

```python
class PriorityQueue:
    """Priority queue using a min-heap (lower value = higher priority)."""
    
    def __init__(self):
        self._heap = MinHeap()
        self._counter = 0  # Tie-breaker for equal priorities
    
    def enqueue(self, item, priority):
        """Add item with given priority."""
        self._heap.push((priority, self._counter, item))
        self._counter += 1
    
    def dequeue(self):
        """Remove and return the highest-priority item."""
        priority, _, item = self._heap.pop()
        return item
    
    def peek(self):
        """Return the highest-priority item without removing."""
        priority, _, item = self._heap.peek()
        return item
    
    def is_empty(self):
        return self._heap.is_empty()
```

## Python's `heapq` Module

Python provides a min-heap implementation via `heapq`:

```python
import heapq

# Create a heap from a list
data = [5, 3, 8, 1, 2]
heapq.heapify(data)  # In-place, O(n)
# data is now [1, 2, 8, 5, 3]

# Push and pop
heapq.heappush(data, 0)    # Push 0
smallest = heapq.heappop(data)  # Pop smallest (0)

# Push and pop in one operation
result = heapq.heappushpop(data, 4)  # Push 4, then pop smallest

# N smallest / largest -- O(n + k log n)
heapq.nsmallest(3, data)   # [1, 2, 3]
heapq.nlargest(3, data)    # [8, 5, 4]

# Max-heap trick: negate values
max_heap = []
heapq.heappush(max_heap, -5)
heapq.heappush(max_heap, -3)
heapq.heappush(max_heap, -8)
largest = -heapq.heappop(max_heap)  # 8
```

## Application: Top K Elements

```python
import heapq

def top_k_frequent(nums, k):
    """Find the k most frequent elements.
    
    >>> top_k_frequent([1,1,1,2,2,3], 2)
    [1, 2]
    """
    from collections import Counter
    counts = Counter(nums)
    return heapq.nlargest(k, counts.keys(), key=counts.get)
```

## Application: Merge K Sorted Lists

```python
import heapq

def merge_k_sorted(lists):
    """Merge k sorted lists into one sorted list -- O(n log k).
    
    >>> merge_k_sorted([[1,4,5], [1,3,4], [2,6]])
    [1, 1, 2, 3, 4, 4, 5, 6]
    """
    result = []
    heap = []
    
    for i, lst in enumerate(lists):
        if lst:
            heapq.heappush(heap, (lst[0], i, 0))
    
    while heap:
        val, list_idx, elem_idx = heapq.heappop(heap)
        result.append(val)
        if elem_idx + 1 < len(lists[list_idx]):
            next_val = lists[list_idx][elem_idx + 1]
            heapq.heappush(heap, (next_val, list_idx, elem_idx + 1))
    
    return result
```

## Application: Running Median

```python
import heapq

class MedianFinder:
    """Find the median of a stream of numbers.
    
    Uses two heaps: max-heap for lower half, min-heap for upper half.
    """
    
    def __init__(self):
        self.small = []  # Max-heap (negated values) -- lower half
        self.large = []  # Min-heap -- upper half
    
    def add_num(self, num):
        heapq.heappush(self.small, -num)
        # Ensure max of small <= min of large
        heapq.heappush(self.large, -heapq.heappop(self.small))
        # Balance sizes (small can have at most 1 more)
        if len(self.large) > len(self.small):
            heapq.heappush(self.small, -heapq.heappop(self.large))
    
    def find_median(self):
        if len(self.small) > len(self.large):
            return -self.small[0]
        return (-self.small[0] + self.large[0]) / 2
```

## Time Complexity Summary

| Operation | Time Complexity |
|-----------|----------------|
| `peek` (find min/max) | O(1) |
| `push` (insert) | O(log n) |
| `pop` (extract min/max) | O(log n) |
| `heapify` (build heap) | O(n) |
| Heap sort | O(n log n) |
| Find arbitrary element | O(n) |
| Delete arbitrary element | O(n) |

## Summary

| Concept | Key Takeaway |
|---------|-------------|
| Heap property | Parent <= children (min) or >= (max) |
| Array representation | Complete binary tree maps to array perfectly |
| Sift up | Restore heap after insertion |
| Sift down | Restore heap after extraction |
| Build heap | Bottom-up heapify is O(n) |
| Priority queue | Heap-based, O(log n) enqueue/dequeue |
| `heapq` | Python's built-in min-heap module |
| Max-heap trick | Negate values with `heapq` |

---

**Next**: [Graphs Basics](./09_Graphs_Basics.md) -- Generalize trees to model networks and relationships.
