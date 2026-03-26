"""
Heap and Priority Queue
Heap and Priority Queue

Implements heap data structures and related algorithms.
"""

from typing import List, Optional, Tuple, Any
import heapq


# =============================================================================
# 1. Min Heap (Direct Implementation)
# =============================================================================

class MinHeap:
    """
    Min Heap (Complete Binary Tree)
    - Parent node <= Child nodes
    - Array implementation: parent(i) = (i-1)//2, left(i) = 2i+1, right(i) = 2i+2
    """

    def __init__(self):
        self.heap: List[int] = []

    def __len__(self) -> int:
        return len(self.heap)

    def __bool__(self) -> bool:
        return len(self.heap) > 0

    def _parent(self, i: int) -> int:
        return (i - 1) // 2

    def _left(self, i: int) -> int:
        return 2 * i + 1

    def _right(self, i: int) -> int:
        return 2 * i + 2

    def _swap(self, i: int, j: int) -> None:
        self.heap[i], self.heap[j] = self.heap[j], self.heap[i]

    def _sift_up(self, i: int) -> None:
        """Sift up after insertion - O(log n)"""
        while i > 0:
            parent = self._parent(i)
            if self.heap[i] < self.heap[parent]:
                self._swap(i, parent)
                i = parent
            else:
                break

    def _sift_down(self, i: int) -> None:
        """Sift down after deletion - O(log n)"""
        n = len(self.heap)

        while True:
            smallest = i
            left = self._left(i)
            right = self._right(i)

            if left < n and self.heap[left] < self.heap[smallest]:
                smallest = left
            if right < n and self.heap[right] < self.heap[smallest]:
                smallest = right

            if smallest != i:
                self._swap(i, smallest)
                i = smallest
            else:
                break

    def push(self, val: int) -> None:
        """Insert - O(log n)"""
        self.heap.append(val)
        self._sift_up(len(self.heap) - 1)

    def pop(self) -> int:
        """Remove and return minimum - O(log n)"""
        if not self.heap:
            raise IndexError("pop from empty heap")

        result = self.heap[0]
        last = self.heap.pop()

        if self.heap:
            self.heap[0] = last
            self._sift_down(0)

        return result

    def peek(self) -> int:
        """View minimum - O(1)"""
        if not self.heap:
            raise IndexError("peek from empty heap")
        return self.heap[0]

    def heapify(self, arr: List[int]) -> None:
        """Convert array to heap - O(n)"""
        self.heap = arr[:]
        # Sift down from last non-leaf node
        for i in range(len(self.heap) // 2 - 1, -1, -1):
            self._sift_down(i)


# =============================================================================
# 2. Max Heap
# =============================================================================

class MaxHeap:
    """Max Heap - Parent node >= Child nodes"""

    def __init__(self):
        self.heap: List[int] = []

    def __len__(self) -> int:
        return len(self.heap)

    def push(self, val: int) -> None:
        """Insert - O(log n)"""
        # Store as negative in min heap
        heapq.heappush(self.heap, -val)

    def pop(self) -> int:
        """Remove and return maximum - O(log n)"""
        return -heapq.heappop(self.heap)

    def peek(self) -> int:
        """View maximum - O(1)"""
        return -self.heap[0]


# =============================================================================
# 3. Heap Sort
# =============================================================================

def heap_sort(arr: List[int]) -> List[int]:
    """
    Heap Sort
    Time Complexity: O(n log n)
    Space Complexity: O(1) (in-place)
    """
    n = len(arr)
    result = arr[:]

    def sift_down(arr: List[int], n: int, i: int) -> None:
        while True:
            largest = i
            left = 2 * i + 1
            right = 2 * i + 2

            if left < n and arr[left] > arr[largest]:
                largest = left
            if right < n and arr[right] > arr[largest]:
                largest = right

            if largest != i:
                arr[i], arr[largest] = arr[largest], arr[i]
                i = largest
            else:
                break

    # 1. Build max heap - O(n)
    for i in range(n // 2 - 1, -1, -1):
        sift_down(result, n, i)

    # 2. Extract one by one - O(n log n)
    for i in range(n - 1, 0, -1):
        result[0], result[i] = result[i], result[0]
        sift_down(result, i, 0)

    return result


# =============================================================================
# 4. Find Kth Element
# =============================================================================

def kth_largest(arr: List[int], k: int) -> int:
    """
    Kth largest element - using min heap
    Time Complexity: O(n log k)
    Space Complexity: O(k)
    """
    min_heap = []

    for num in arr:
        heapq.heappush(min_heap, num)
        if len(min_heap) > k:
            heapq.heappop(min_heap)

    return min_heap[0]


def kth_smallest(arr: List[int], k: int) -> int:
    """
    Kth smallest element - using max heap
    Time Complexity: O(n log k)
    Space Complexity: O(k)
    """
    max_heap = []

    for num in arr:
        heapq.heappush(max_heap, -num)
        if len(max_heap) > k:
            heapq.heappop(max_heap)

    return -max_heap[0]


# =============================================================================
# 5. Median Stream (Median Finder)
# =============================================================================

class MedianFinder:
    """
    Median of a data stream
    - Max heap (left half): smaller values
    - Min heap (right half): larger values
    """

    def __init__(self):
        self.max_heap: List[int] = []  # Left (smaller values)
        self.min_heap: List[int] = []  # Right (larger values)

    def add_num(self, num: int) -> None:
        """Add number - O(log n)"""
        # Add to left heap
        heapq.heappush(self.max_heap, -num)

        # Move left max to right
        heapq.heappush(self.min_heap, -heapq.heappop(self.max_heap))

        # Maintain size balance (left >= right)
        if len(self.min_heap) > len(self.max_heap):
            heapq.heappush(self.max_heap, -heapq.heappop(self.min_heap))

    def find_median(self) -> float:
        """Return median - O(1)"""
        if len(self.max_heap) > len(self.min_heap):
            return -self.max_heap[0]
        return (-self.max_heap[0] + self.min_heap[0]) / 2


# =============================================================================
# 6. Priority Queue Application: Task Scheduling
# =============================================================================

def schedule_tasks(tasks: List[Tuple[int, int, str]]) -> List[str]:
    """
    Priority-based task scheduling
    tasks: [(priority, arrival_time, task_name), ...]
    Lower priority number = higher priority
    """
    # Add to heap as (priority, arrival_time, task_name)
    task_heap = []
    for priority, arrival, name in tasks:
        heapq.heappush(task_heap, (priority, arrival, name))

    schedule = []
    while task_heap:
        _, _, name = heapq.heappop(task_heap)
        schedule.append(name)

    return schedule


# =============================================================================
# 7. Merge K Sorted Lists
# =============================================================================

def merge_k_sorted_lists(lists: List[List[int]]) -> List[int]:
    """
    Merge K sorted lists
    Time Complexity: O(N log k), N = total elements, k = number of lists
    """
    result = []
    min_heap = []

    # Add first element of each list to the heap
    for i, lst in enumerate(lists):
        if lst:
            heapq.heappush(min_heap, (lst[0], i, 0))  # (value, list_index, element_index)

    while min_heap:
        val, list_idx, elem_idx = heapq.heappop(min_heap)
        result.append(val)

        # Add next element if available
        if elem_idx + 1 < len(lists[list_idx]):
            next_val = lists[list_idx][elem_idx + 1]
            heapq.heappush(min_heap, (next_val, list_idx, elem_idx + 1))

    return result


# =============================================================================
# 8. Top K Frequent Elements
# =============================================================================

def top_k_frequent(nums: List[int], k: int) -> List[int]:
    """
    K most frequent elements
    Time Complexity: O(n log k)
    """
    from collections import Counter

    freq = Counter(nums)

    # Keep only K elements using min heap
    min_heap = []
    for num, count in freq.items():
        heapq.heappush(min_heap, (count, num))
        if len(min_heap) > k:
            heapq.heappop(min_heap)

    return [num for count, num in min_heap]


# =============================================================================
# 9. K Closest Points
# =============================================================================

def k_closest_points(points: List[Tuple[int, int]], k: int) -> List[Tuple[int, int]]:
    """
    K closest points to the origin
    Time Complexity: O(n log k)
    """
    # Keep K elements using max heap (store negative distances)
    max_heap = []

    for x, y in points:
        dist = x * x + y * y
        heapq.heappush(max_heap, (-dist, x, y))
        if len(max_heap) > k:
            heapq.heappop(max_heap)

    return [(x, y) for _, x, y in max_heap]


# =============================================================================
# 10. Meeting Rooms
# =============================================================================

def min_meeting_rooms(intervals: List[Tuple[int, int]]) -> int:
    """
    Minimum number of meeting rooms needed
    intervals: [(start_time, end_time), ...]
    Time Complexity: O(n log n)
    """
    if not intervals:
        return 0

    # Sort by start time
    intervals.sort(key=lambda x: x[0])

    # Min heap storing end times
    end_times = []
    heapq.heappush(end_times, intervals[0][1])

    for start, end in intervals[1:]:
        # Reuse room if earliest ending meeting finishes before current starts
        if end_times[0] <= start:
            heapq.heappop(end_times)
        heapq.heappush(end_times, end)

    return len(end_times)


# =============================================================================
# Tests
# =============================================================================

def main():
    print("=" * 60)
    print("Heap and Priority Queue Examples")
    print("=" * 60)

    # 1. Min Heap
    print("\n[1] Min Heap (Direct Implementation)")
    min_heap = MinHeap()
    for val in [5, 3, 8, 1, 2, 7]:
        min_heap.push(val)
    print(f"    Insert: [5, 3, 8, 1, 2, 7]")
    print(f"    Heap array: {min_heap.heap}")
    print(f"    Pop order: ", end="")
    result = []
    while min_heap:
        result.append(min_heap.pop())
    print(result)

    # 2. Max Heap
    print("\n[2] Max Heap")
    max_heap = MaxHeap()
    for val in [5, 3, 8, 1, 2, 7]:
        max_heap.push(val)
    print(f"    Insert: [5, 3, 8, 1, 2, 7]")
    print(f"    Pop order: ", end="")
    result = []
    while max_heap:
        result.append(max_heap.pop())
    print(result)

    # 3. Heapify
    print("\n[3] Heapify (Array -> Heap)")
    arr = [9, 5, 6, 2, 3]
    heap = MinHeap()
    heap.heapify(arr)
    print(f"    Original: {arr}")
    print(f"    Heap: {heap.heap}")

    # 4. Heap Sort
    print("\n[4] Heap Sort")
    arr = [64, 34, 25, 12, 22, 11, 90]
    sorted_arr = heap_sort(arr)
    print(f"    Original: {arr}")
    print(f"    Sorted: {sorted_arr}")

    # 5. Kth Element
    print("\n[5] Kth Element")
    arr = [3, 2, 1, 5, 6, 4]
    print(f"    Array: {arr}")
    print(f"    2nd largest: {kth_largest(arr, 2)}")
    print(f"    2nd smallest: {kth_smallest(arr, 2)}")

    # 6. Median Stream
    print("\n[6] Median Stream")
    mf = MedianFinder()
    stream = [2, 3, 4]
    print(f"    Stream: {stream}")
    for num in stream:
        mf.add_num(num)
        print(f"    After adding {num}, median: {mf.find_median()}")

    # 7. Merge K Sorted Lists
    print("\n[7] Merge K Sorted Lists")
    lists = [[1, 4, 5], [1, 3, 4], [2, 6]]
    merged = merge_k_sorted_lists(lists)
    print(f"    Input: {lists}")
    print(f"    Merged: {merged}")

    # 8. Top K Frequent
    print("\n[8] Top K Frequent Elements")
    nums = [1, 1, 1, 2, 2, 3]
    k = 2
    result = top_k_frequent(nums, k)
    print(f"    Array: {nums}, k={k}")
    print(f"    Result: {result}")

    # 9. K Closest Points
    print("\n[9] K Closest Points to Origin")
    points = [(1, 3), (-2, 2), (5, 8), (0, 1)]
    k = 2
    closest = k_closest_points(points, k)
    print(f"    Points: {points}, k={k}")
    print(f"    Result: {closest}")

    # 10. Meeting Rooms
    print("\n[10] Minimum Meeting Rooms")
    meetings = [(0, 30), (5, 10), (15, 20)]
    rooms = min_meeting_rooms(meetings)
    print(f"    Meetings: {meetings}")
    print(f"    Rooms needed: {rooms}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
