# Linked Lists

**Previous**: [Arrays and Lists](./01_Arrays_and_Lists.md) | **Next**: [Stacks](./03_Stacks.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain the structure of singly, doubly, and circular linked lists
2. Implement a linked list from scratch using Python classes
3. Perform insertion, deletion, and traversal operations
4. Compare linked lists to arrays in terms of time and space complexity
5. Understand pointer manipulation and common pitfalls
6. Apply the runner (fast/slow pointer) technique to solve linked list problems
7. Recognize when linked lists are preferable to arrays

---

A **linked list** is a linear data structure where each element (called a **node**) contains data and a reference (pointer) to the next node. Unlike arrays, linked list elements are not stored contiguously in memory -- each node can be anywhere in the heap.

## Singly Linked List

In a singly linked list, each node has two fields: `data` and `next`.

```
  head
   |
   v
+------+------+    +------+------+    +------+------+    +------+------+
| data | next-+--->| data | next-+--->| data | next-+--->| data | next-+--->None
|  10  |      |    |  20  |      |    |  30  |      |    |  40  |      |
+------+------+    +------+------+    +------+------+    +------+------+
```

### Node Class

```python
class Node:
    """A node in a singly linked list."""
    
    __slots__ = ('data', 'next')  # Memory optimization
    
    def __init__(self, data, next_node=None):
        self.data = data
        self.next = next_node
```

### Singly Linked List Implementation

```python
class SinglyLinkedList:
    """A singly linked list with head pointer."""
    
    def __init__(self):
        self.head = None
        self._size = 0
    
    def __len__(self):
        return self._size
    
    def is_empty(self):
        return self.head is None
    
    def prepend(self, data):
        """Insert at the beginning -- O(1)."""
        self.head = Node(data, self.head)
        self._size += 1
    
    def append(self, data):
        """Insert at the end -- O(n)."""
        new_node = Node(data)
        if self.head is None:
            self.head = new_node
        else:
            current = self.head
            while current.next:
                current = current.next
            current.next = new_node
        self._size += 1
    
    def delete(self, data):
        """Delete the first occurrence of data -- O(n)."""
        if self.head is None:
            raise ValueError(f"{data} not found in list")
        
        if self.head.data == data:
            self.head = self.head.next
            self._size -= 1
            return
        
        current = self.head
        while current.next:
            if current.next.data == data:
                current.next = current.next.next
                self._size -= 1
                return
            current = current.next
        
        raise ValueError(f"{data} not found in list")
    
    def search(self, data):
        """Search for data -- O(n)."""
        current = self.head
        while current:
            if current.data == data:
                return True
            current = current.next
        return False
    
    def __iter__(self):
        current = self.head
        while current:
            yield current.data
            current = current.next
    
    def __repr__(self):
        return " -> ".join(str(x) for x in self) + " -> None"
```

### Insertion Operations Visualized

```
Prepend (insert at head) -- O(1):
  Before:  head -> [20] -> [30] -> None
  After:   head -> [10] -> [20] -> [30] -> None
  Steps:   1. new_node.next = head
           2. head = new_node

Insert after a node -- O(1) if you have the reference:
  Before:  ... -> [20] -> [30] -> ...
  After:   ... -> [20] -> [25] -> [30] -> ...
  Steps:   1. new_node.next = current.next
           2. current.next = new_node

Append (insert at tail) -- O(n) without tail pointer:
  Before:  head -> [10] -> [20] -> None
  After:   head -> [10] -> [20] -> [30] -> None
  Steps:   1. Traverse to last node
           2. last.next = new_node
```

### Deletion Operations Visualized

```
Delete head -- O(1):
  Before:  head -> [10] -> [20] -> [30] -> None
  After:   head -> [20] -> [30] -> None
  Steps:   1. head = head.next

Delete middle -- O(n) to find, O(1) to unlink:
  Before:  ... -> [10] -> [20] -> [30] -> ...
  After:   ... -> [10] -> [30] -> ...
  Steps:   1. Find node before target (prev)
           2. prev.next = prev.next.next
```

## The Sentinel (Dummy Head) Technique

A common trick to simplify edge cases is using a **sentinel node** -- a dummy node at the head that never contains real data:

```python
class SinglyLinkedListSentinel:
    """Singly linked list with sentinel node."""
    
    def __init__(self):
        self._sentinel = Node(None)  # Dummy head
        self._size = 0
    
    def prepend(self, data):
        """No special case needed -- sentinel is always there."""
        new_node = Node(data, self._sentinel.next)
        self._sentinel.next = new_node
        self._size += 1
    
    def delete(self, data):
        """No special case for head deletion."""
        prev = self._sentinel
        while prev.next:
            if prev.next.data == data:
                prev.next = prev.next.next
                self._size -= 1
                return
            prev = prev.next
        raise ValueError(f"{data} not found")
```

## Doubly Linked List

A doubly linked list adds a `prev` pointer to each node, enabling O(1) operations in both directions:

```
         head                                              tail
          |                                                  |
          v                                                  v
None <--+------+------+  +------+------+------+  +------+------+--> None
        | prev | data |  | prev | data | next |  | prev | data |
        |      |  10  |  |      |  20  |      |  |      |  30  |
        +------+--+---+  +---+--+------+--+---+  +---+--+------+
                  |           ^            |           ^
                  +-----------+            +-----------+
```

```python
class DNode:
    """A node in a doubly linked list."""
    
    __slots__ = ('data', 'prev', 'next')
    
    def __init__(self, data, prev_node=None, next_node=None):
        self.data = data
        self.prev = prev_node
        self.next = next_node


class DoublyLinkedList:
    """Doubly linked list with head and tail pointers."""
    
    def __init__(self):
        self.head = None
        self.tail = None
        self._size = 0
    
    def append(self, data):
        """Insert at the end -- O(1)."""
        new_node = DNode(data, prev_node=self.tail)
        if self.tail:
            self.tail.next = new_node
        else:
            self.head = new_node
        self.tail = new_node
        self._size += 1
    
    def prepend(self, data):
        """Insert at the beginning -- O(1)."""
        new_node = DNode(data, next_node=self.head)
        if self.head:
            self.head.prev = new_node
        else:
            self.tail = new_node
        self.head = new_node
        self._size += 1
    
    def delete_node(self, node):
        """Delete a specific node -- O(1) given the reference."""
        if node.prev:
            node.prev.next = node.next
        else:
            self.head = node.next
        
        if node.next:
            node.next.prev = node.prev
        else:
            self.tail = node.prev
        
        self._size -= 1
    
    def __iter__(self):
        current = self.head
        while current:
            yield current.data
            current = current.next
    
    def __reversed__(self):
        current = self.tail
        while current:
            yield current.data
            current = current.prev
    
    def __repr__(self):
        return " <-> ".join(str(x) for x in self) + " <-> None"
```

## Circular Linked List

In a circular linked list, the last node points back to the first, forming a ring:

```
Singly Circular:
        +---> [10] ---> [20] ---> [30] ---+
        |                                   |
        +-----------------------------------+

Doubly Circular:
        +--> [10] <--> [20] <--> [30] --+
        |                                |
        +--- prev                next ---+
```

```python
class CircularLinkedList:
    """Circular singly linked list."""
    
    def __init__(self):
        self.tail = None  # Points to last node (last.next = first)
        self._size = 0
    
    def append(self, data):
        new_node = Node(data)
        if self.tail is None:
            new_node.next = new_node  # Points to itself
            self.tail = new_node
        else:
            new_node.next = self.tail.next  # Point to head
            self.tail.next = new_node
            self.tail = new_node
        self._size += 1
    
    def __iter__(self):
        if self.tail is None:
            return
        current = self.tail.next  # Start at head
        for _ in range(self._size):
            yield current.data
            current = current.next
```

## The Fast/Slow Pointer Technique

Also called the "tortoise and hare" technique. One pointer moves one step at a time, the other moves two steps.

### Detect a Cycle

```python
def has_cycle(head):
    """Detect if a linked list has a cycle -- O(n) time, O(1) space."""
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow is fast:
            return True
    return False
```

### Find the Middle Node

```python
def find_middle(head):
    """Find the middle node -- O(n) time, O(1) space."""
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    return slow  # When fast reaches end, slow is at middle
```

## Reversing a Linked List

One of the most fundamental linked list operations:

```python
def reverse_list(head):
    """Reverse a singly linked list in-place -- O(n) time, O(1) space."""
    prev = None
    current = head
    while current:
        next_node = current.next  # Save next
        current.next = prev       # Reverse pointer
        prev = current            # Move prev forward
        current = next_node       # Move current forward
    return prev  # New head

# Visualization:
# Step 0: None    10 -> 20 -> 30 -> None
#          ^prev  ^curr
# Step 1: None <- 10    20 -> 30 -> None
#                 ^prev ^curr
# Step 2: None <- 10 <- 20    30 -> None
#                       ^prev ^curr
# Step 3: None <- 10 <- 20 <- 30
#                             ^prev  curr=None
```

## Comparison: Array vs Linked List

| Operation | Array (list) | Singly LL | Doubly LL |
|-----------|-------------|-----------|-----------|
| Access by index | **O(1)** | O(n) | O(n) |
| Search | O(n) | O(n) | O(n) |
| Insert at head | O(n) | **O(1)** | **O(1)** |
| Insert at tail | O(1)* | O(n)** | **O(1)** |
| Delete at head | O(n) | **O(1)** | **O(1)** |
| Delete at tail | O(1) | O(n) | **O(1)** |
| Delete given node | O(n) | O(n)*** | **O(1)** |
| Memory per element | Lower | Higher (+1 ptr) | Higher (+2 ptrs) |
| Cache performance | **Excellent** | Poor | Poor |

*Amortized | **O(1) with tail pointer | ***O(1) if you have previous node

## When to Use Linked Lists

**Use linked lists when:**
- Frequent insertions/deletions at the beginning
- You need a queue (use doubly linked list)
- You need to splice lists together in O(1)
- Memory allocation should be incremental (no large reallocations)

**Avoid linked lists when:**
- You need random access by index
- Cache performance matters (prefer arrays)
- Memory overhead per element is a concern

## Summary

| Concept | Key Takeaway |
|---------|-------------|
| Singly linked list | Each node has data + next pointer |
| Doubly linked list | Each node has data + next + prev pointers |
| Circular linked list | Last node connects back to first |
| Sentinel node | Dummy head simplifies edge cases |
| Fast/slow pointers | Detect cycles, find midpoints |
| Reversal | Classic O(n) in-place technique |
| vs Arrays | Better for insertions, worse for access |

---

**Next**: [Stacks](./03_Stacks.md) -- Build on linked lists and arrays to create LIFO data structures.
