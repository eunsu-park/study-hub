"""
08 Heaps
========
Demonstrates min-heap, heapq module, priority queue,
top-K, and merge-K-sorted problems.
"""

import heapq
from collections import Counter


class MinHeap:
    """Binary min-heap using an array."""
    def __init__(self):
        self._data = []

    def push(self, val):
        self._data.append(val)
        i = len(self._data) - 1
        while i > 0:
            p = (i - 1) // 2
            if self._data[i] < self._data[p]:
                self._data[i], self._data[p] = self._data[p], self._data[i]
                i = p
            else: break

    def pop(self):
        self._data[0], self._data[-1] = self._data[-1], self._data[0]
        val = self._data.pop()
        if self._data: self._sift_down(0)
        return val

    def _sift_down(self, i):
        n = len(self._data)
        while True:
            s = i; l, r = 2*i+1, 2*i+2
            if l < n and self._data[l] < self._data[s]: s = l
            if r < n and self._data[r] < self._data[s]: s = r
            if s == i: break
            self._data[i], self._data[s] = self._data[s], self._data[i]; i = s

    def __len__(self): return len(self._data)
    def __repr__(self): return f"MinHeap({self._data})"


if __name__ == "__main__":
    # Custom heap
    h = MinHeap()
    for v in [5, 3, 8, 1, 2, 7]:
        h.push(v)
    print(f"Heap: {h}")
    print("Pop order: ", end="")
    while len(h): print(h.pop(), end=" ")
    print()

    # heapq module
    print("\nheapq module:")
    data = [5, 3, 8, 1, 2]
    heapq.heapify(data)
    print(f"Heapified: {data}")
    heapq.heappush(data, 0)
    print(f"After push(0): {data}")
    print(f"Pop: {heapq.heappop(data)}")
    print(f"3 smallest: {heapq.nsmallest(3, data)}")
    print(f"3 largest:  {heapq.nlargest(3, data)}")

    # Top K frequent
    print("\nTop K frequent:")
    nums = [1,1,1,2,2,3,3,3,3,4]
    counts = Counter(nums)
    print(f"Top 2 frequent in {nums}: {heapq.nlargest(2, counts.keys(), key=counts.get)}")

    # Merge K sorted
    lists = [[1,4,5], [1,3,4], [2,6]]
    print(f"\nMerge {lists}: {list(heapq.merge(*lists))}")
