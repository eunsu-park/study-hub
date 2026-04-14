# Queues

**Previous**: [Stacks](./03_Stacks.md) | **Next**: [Hash Tables](./05_Hash_Tables.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain the FIFO (First-In, First-Out) principle
2. Implement a queue using arrays, linked lists, and circular buffers
3. Understand and implement a circular queue with fixed capacity
4. Use Python's `collections.deque` for efficient double-ended operations
5. Implement a double-ended queue (deque) from scratch
6. Apply queues to BFS, task scheduling, and producer-consumer patterns
7. Compare different queue implementations and their trade-offs

---

A **queue** is a linear data structure that follows the **First-In, First-Out (FIFO)** principle. Elements are added at the **rear** and removed from the **front**, like a line of people waiting.

## The Queue ADT

```
  Dequeue <-- [Front] [  ] [  ] [  ] [Rear] <-- Enqueue

  enqueue(10), enqueue(20), enqueue(30):
  Front                     Rear
    |                         |
    v                         v
  +----+----+----+
  | 10 | 20 | 30 |
  +----+----+----+

  dequeue() returns 10:
       Front          Rear
         |              |
         v              v
       +----+----+
       | 20 | 30 |
       +----+----+
```

### Core Operations

| Operation | Description | Time |
|-----------|-------------|------|
| `enqueue(item)` | Add item to the rear | O(1) |
| `dequeue()` | Remove and return the front item | O(1) |
| `front()` / `peek()` | Return the front item without removing | O(1) |
| `is_empty()` | Check if the queue is empty | O(1) |
| `size()` | Return the number of elements | O(1) |

## Naive Array-Based Queue (Inefficient)

Using a Python list naively leads to O(n) dequeue:

```python
class NaiveQueue:
    """Queue using list -- O(n) dequeue due to shifting!"""
    
    def __init__(self):
        self._data = []
    
    def enqueue(self, item):
        self._data.append(item)  # O(1) amortized
    
    def dequeue(self):
        if self.is_empty():
            raise IndexError("dequeue from empty queue")
        return self._data.pop(0)  # O(n) -- shifts all elements!
    
    def is_empty(self):
        return len(self._data) == 0
```

**Why is this bad?** `pop(0)` shifts every element one position left, making each dequeue O(n).

## Circular Queue (Ring Buffer)

A circular queue uses a fixed-size array with two pointers (`front` and `rear`) that wrap around:

```
Circular buffer (capacity=5):

  Initial state:           After enqueue(10, 20, 30):
  +---+---+---+---+---+   +----+----+----+---+---+
  |   |   |   |   |   |   | 10 | 20 | 30 |   |   |
  +---+---+---+---+---+   +----+----+----+---+---+
    ^                        ^              ^
    front, rear              front          rear

  After dequeue() (removes 10):    After enqueue(40, 50, 60):
  +---+----+----+---+---+         +----+----+----+----+----+
  |   | 20 | 30 |   |   |        | 60 | 20 | 30 | 40 | 50 |
  +---+----+----+---+---+        +----+----+----+----+----+
        ^         ^                 ^    ^
        front     rear              rear front
                                    (wraps around!)
```

```python
class CircularQueue:
    """Fixed-capacity circular queue using an array."""
    
    def __init__(self, capacity):
        self._data = [None] * capacity
        self._capacity = capacity
        self._front = 0
        self._size = 0
    
    def enqueue(self, item):
        """Add item to the rear -- O(1)."""
        if self._size == self._capacity:
            raise OverflowError("queue is full")
        rear = (self._front + self._size) % self._capacity
        self._data[rear] = item
        self._size += 1
    
    def dequeue(self):
        """Remove and return the front item -- O(1)."""
        if self.is_empty():
            raise IndexError("dequeue from empty queue")
        item = self._data[self._front]
        self._data[self._front] = None  # Help garbage collection
        self._front = (self._front + 1) % self._capacity
        self._size -= 1
        return item
    
    def peek(self):
        """Return the front item without removing -- O(1)."""
        if self.is_empty():
            raise IndexError("peek at empty queue")
        return self._data[self._front]
    
    def is_empty(self):
        return self._size == 0
    
    def is_full(self):
        return self._size == self._capacity
    
    def __len__(self):
        return self._size
    
    def __repr__(self):
        items = []
        for i in range(self._size):
            idx = (self._front + i) % self._capacity
            items.append(str(self._data[idx]))
        return f"CircularQueue([{', '.join(items)}])"
```

## Linked-List-Based Queue

Using a singly linked list with both head and tail pointers:

```python
class LinkedQueue:
    """Queue implementation using a singly linked list."""
    
    def __init__(self):
        self._head = None  # Front of queue
        self._tail = None  # Rear of queue
        self._size = 0
    
    def enqueue(self, item):
        """Add item to the rear -- O(1)."""
        new_node = Node(item)
        if self._tail:
            self._tail.next = new_node
        else:
            self._head = new_node
        self._tail = new_node
        self._size += 1
    
    def dequeue(self):
        """Remove and return the front item -- O(1)."""
        if self.is_empty():
            raise IndexError("dequeue from empty queue")
        data = self._head.data
        self._head = self._head.next
        if self._head is None:
            self._tail = None
        self._size -= 1
        return data
    
    def peek(self):
        if self.is_empty():
            raise IndexError("peek at empty queue")
        return self._head.data
    
    def is_empty(self):
        return self._head is None
    
    def __len__(self):
        return self._size
```

## Double-Ended Queue (Deque)

A **deque** (pronounced "deck") allows insertion and removal at both ends in O(1):

```
        +----+----+----+----+----+
  <---> | 10 | 20 | 30 | 40 | 50 | <--->
        +----+----+----+----+----+
          ^                    ^
        front                rear

  append_left / pop_left      append_right / pop_right
```

```python
class Deque:
    """Double-ended queue using a doubly linked list."""
    
    def __init__(self):
        self._head = None
        self._tail = None
        self._size = 0
    
    def append_right(self, item):
        """Add to the right end -- O(1)."""
        new_node = DNode(item, prev_node=self._tail)
        if self._tail:
            self._tail.next = new_node
        else:
            self._head = new_node
        self._tail = new_node
        self._size += 1
    
    def append_left(self, item):
        """Add to the left end -- O(1)."""
        new_node = DNode(item, next_node=self._head)
        if self._head:
            self._head.prev = new_node
        else:
            self._tail = new_node
        self._head = new_node
        self._size += 1
    
    def pop_right(self):
        """Remove from the right end -- O(1)."""
        if self.is_empty():
            raise IndexError("pop from empty deque")
        data = self._tail.data
        self._tail = self._tail.prev
        if self._tail:
            self._tail.next = None
        else:
            self._head = None
        self._size -= 1
        return data
    
    def pop_left(self):
        """Remove from the left end -- O(1)."""
        if self.is_empty():
            raise IndexError("pop from empty deque")
        data = self._head.data
        self._head = self._head.next
        if self._head:
            self._head.prev = None
        else:
            self._tail = None
        self._size -= 1
        return data
    
    def is_empty(self):
        return self._size == 0
    
    def __len__(self):
        return self._size
```

## Python's `collections.deque`

Python provides an optimized deque implemented as a doubly linked list of fixed-size blocks:

```python
from collections import deque

# Create a deque
d = deque([1, 2, 3])

# O(1) operations at both ends
d.append(4)        # [1, 2, 3, 4]
d.appendleft(0)    # [0, 1, 2, 3, 4]
d.pop()            # returns 4, deque is [0, 1, 2, 3]
d.popleft()        # returns 0, deque is [1, 2, 3]

# Rotate
d.rotate(1)        # [3, 1, 2]  (right rotation)
d.rotate(-1)       # [1, 2, 3]  (left rotation)

# Bounded deque (acts as circular buffer)
bounded = deque(maxlen=3)
bounded.extend([1, 2, 3])  # deque([1, 2, 3])
bounded.append(4)           # deque([2, 3, 4]) -- 1 dropped!
```

## Application: BFS (Breadth-First Search)

Queues are essential for BFS traversal:

```python
from collections import deque

def bfs(graph, start):
    """Breadth-first search using a queue.
    
    graph: adjacency list {node: [neighbors]}
    """
    visited = {start}
    queue = deque([start])
    order = []
    
    while queue:
        node = queue.popleft()  # O(1)
        order.append(node)
        
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    
    return order
```

## Application: Hot Potato Simulation

```python
from collections import deque

def hot_potato(names, num_passes):
    """Simulate the hot potato game.
    
    Players stand in a circle. Potato is passed num_passes times.
    The person holding it is eliminated. Repeat until one remains.
    """
    queue = deque(names)
    
    while len(queue) > 1:
        for _ in range(num_passes):
            queue.append(queue.popleft())  # Pass the potato
        eliminated = queue.popleft()
        print(f"{eliminated} is eliminated")
    
    return queue[0]  # Winner
```

## Application: Sliding Window Maximum

Using a deque to find the maximum in each window of size k:

```python
from collections import deque

def sliding_window_max(nums, k):
    """Find the maximum in each sliding window of size k.
    
    Uses a monotonic deque -- O(n) total time.
    
    >>> sliding_window_max([1, 3, -1, -3, 5, 3, 6, 7], 3)
    [3, 3, 5, 5, 6, 7]
    """
    result = []
    dq = deque()  # Stores indices of potentially maximum elements
    
    for i in range(len(nums)):
        # Remove indices outside the window
        while dq and dq[0] < i - k + 1:
            dq.popleft()
        
        # Remove smaller elements (they can never be the max)
        while dq and nums[dq[-1]] < nums[i]:
            dq.pop()
        
        dq.append(i)
        
        # Window is fully formed starting at index k-1
        if i >= k - 1:
            result.append(nums[dq[0]])
    
    return result
```

## Comparison of Queue Implementations

| Implementation | Enqueue | Dequeue | Space | Notes |
|---------------|---------|---------|-------|-------|
| List (naive) | O(1)* | **O(n)** | Dynamic | Avoid for queues |
| Circular array | O(1) | O(1) | Fixed | Best for bounded queues |
| Linked list | O(1) | O(1) | Dynamic | Extra pointer overhead |
| `collections.deque` | O(1) | O(1) | Dynamic | Best general-purpose |

*Amortized

## Priority Queue Preview

A regular queue processes elements in arrival order. A **priority queue** processes elements by priority (covered in [Lesson 8: Heaps](./08_Heaps.md)):

```
Regular Queue:    First come, first served
Priority Queue:   Highest priority first

  enqueue(task_A, priority=3)
  enqueue(task_B, priority=1)  -- highest priority
  enqueue(task_C, priority=2)
  
  dequeue() -> task_B  (priority 1, not task_A which arrived first)
```

## Summary

| Concept | Key Takeaway |
|---------|-------------|
| FIFO | First-In, First-Out principle |
| Core ops | enqueue, dequeue, peek -- all O(1) |
| Circular queue | Wrapping indices avoid shifting |
| Deque | Double-ended, O(1) at both ends |
| `collections.deque` | Go-to queue implementation in Python |
| BFS | Queues enable level-by-level graph traversal |
| Sliding window | Monotonic deque for O(n) window queries |

---

**Next**: [Hash Tables](./05_Hash_Tables.md) -- Explore constant-time key-value lookups.
