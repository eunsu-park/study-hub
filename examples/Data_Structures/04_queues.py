"""
04 Queues
=========
Demonstrates queue implementations: circular queue,
deque, BFS, and sliding window maximum.
"""

from collections import deque


class CircularQueue:
    """Fixed-capacity circular queue."""

    def __init__(self, capacity):
        self._data = [None] * capacity
        self._capacity = capacity
        self._front = 0
        self._size = 0

    def enqueue(self, item):
        if self._size == self._capacity:
            raise OverflowError("queue is full")
        rear = (self._front + self._size) % self._capacity
        self._data[rear] = item
        self._size += 1

    def dequeue(self):
        if self.is_empty():
            raise IndexError("dequeue from empty queue")
        item = self._data[self._front]
        self._front = (self._front + 1) % self._capacity
        self._size -= 1
        return item

    def is_empty(self):
        return self._size == 0

    def __len__(self):
        return self._size

    def __repr__(self):
        items = []
        for i in range(self._size):
            idx = (self._front + i) % self._capacity
            items.append(str(self._data[idx]))
        return f"CircularQueue([{', '.join(items)}])"


def demo_circular_queue():
    """Demonstrate circular queue operations."""
    cq = CircularQueue(5)
    for x in [10, 20, 30, 40, 50]:
        cq.enqueue(x)
    print(f"Full queue: {cq}")
    print(f"Dequeue: {cq.dequeue()}")
    print(f"Dequeue: {cq.dequeue()}")
    cq.enqueue(60)
    cq.enqueue(70)
    print(f"After wrap: {cq}")


def demo_deque():
    """Demonstrate collections.deque."""
    d = deque([1, 2, 3])
    d.append(4)
    d.appendleft(0)
    print(f"Deque: {d}")
    d.rotate(1)
    print(f"Rotated right: {d}")
    d.rotate(-2)
    print(f"Rotated left 2: {d}")

    bounded = deque(maxlen=3)
    for x in range(5):
        bounded.append(x)
        print(f"  append({x}): {list(bounded)}")


def demo_bfs():
    """BFS using a queue."""
    graph = {
        'A': ['B', 'C'], 'B': ['A', 'D', 'E'],
        'C': ['A', 'F'], 'D': ['B'],
        'E': ['B', 'F'], 'F': ['C', 'E'],
    }
    visited = {'A'}
    queue = deque(['A'])
    order = []
    while queue:
        node = queue.popleft()
        order.append(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    print(f"BFS order: {order}")


def sliding_window_max(nums, k):
    """Find max in each sliding window of size k."""
    result = []
    dq = deque()
    for i in range(len(nums)):
        while dq and dq[0] < i - k + 1:
            dq.popleft()
        while dq and nums[dq[-1]] < nums[i]:
            dq.pop()
        dq.append(i)
        if i >= k - 1:
            result.append(nums[dq[0]])
    return result


def demo_sliding_window():
    """Demonstrate sliding window maximum."""
    nums = [1, 3, -1, -3, 5, 3, 6, 7]
    k = 3
    result = sliding_window_max(nums, k)
    print(f"nums = {nums}, k = {k}")
    print(f"Window maxima: {result}")


if __name__ == "__main__":
    for title, func in [
        ("Circular Queue", demo_circular_queue),
        ("collections.deque", demo_deque),
        ("BFS with Queue", demo_bfs),
        ("Sliding Window Max", demo_sliding_window),
    ]:
        print(f"\n{'=' * 50}")
        print(f"  {title}")
        print('=' * 50)
        func()
