"""
02 Linked Lists
===============
Demonstrates singly, doubly, and circular linked lists
with common operations and algorithms.
"""


class Node:
    """A node in a singly linked list."""

    __slots__ = ('data', 'next')

    def __init__(self, data, next_node=None):
        self.data = data
        self.next = next_node


class SinglyLinkedList:
    """Singly linked list with head pointer."""

    def __init__(self):
        self.head = None
        self._size = 0

    def prepend(self, data):
        self.head = Node(data, self.head)
        self._size += 1

    def append(self, data):
        new_node = Node(data)
        if self.head is None:
            self.head = new_node
        else:
            cur = self.head
            while cur.next:
                cur = cur.next
            cur.next = new_node
        self._size += 1

    def delete(self, data):
        if self.head and self.head.data == data:
            self.head = self.head.next
            self._size -= 1
            return
        cur = self.head
        while cur and cur.next:
            if cur.next.data == data:
                cur.next = cur.next.next
                self._size -= 1
                return
            cur = cur.next

    def search(self, data):
        cur = self.head
        while cur:
            if cur.data == data:
                return True
            cur = cur.next
        return False

    def __len__(self):
        return self._size

    def __repr__(self):
        items = []
        cur = self.head
        while cur:
            items.append(str(cur.data))
            cur = cur.next
        return " -> ".join(items) + " -> None"


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
        new_node = DNode(data, prev_node=self.tail)
        if self.tail:
            self.tail.next = new_node
        else:
            self.head = new_node
        self.tail = new_node
        self._size += 1

    def prepend(self, data):
        new_node = DNode(data, next_node=self.head)
        if self.head:
            self.head.prev = new_node
        else:
            self.tail = new_node
        self.head = new_node
        self._size += 1

    def __repr__(self):
        items = []
        cur = self.head
        while cur:
            items.append(str(cur.data))
            cur = cur.next
        return " <-> ".join(items)


def reverse_list(head):
    """Reverse a singly linked list in-place."""
    prev = None
    current = head
    while current:
        next_node = current.next
        current.next = prev
        prev = current
        current = next_node
    return prev


def find_middle(head):
    """Find the middle node using fast/slow pointers."""
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    return slow


def has_cycle(head):
    """Detect cycle using Floyd's algorithm."""
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow is fast:
            return True
    return False


def demo_singly_linked_list():
    """Demonstrate singly linked list operations."""
    ll = SinglyLinkedList()
    for x in [10, 20, 30, 40, 50]:
        ll.append(x)
    print(f"List: {ll}")
    print(f"Length: {len(ll)}")
    print(f"Search 30: {ll.search(30)}")
    print(f"Search 99: {ll.search(99)}")

    ll.prepend(5)
    print(f"After prepend(5): {ll}")

    ll.delete(30)
    print(f"After delete(30): {ll}")


def demo_doubly_linked_list():
    """Demonstrate doubly linked list operations."""
    dll = DoublyLinkedList()
    for x in [10, 20, 30]:
        dll.append(x)
    dll.prepend(5)
    print(f"Doubly linked: {dll}")
    print(f"Head: {dll.head.data}, Tail: {dll.tail.data}")

    # Traverse in reverse
    items = []
    cur = dll.tail
    while cur:
        items.append(str(cur.data))
        cur = cur.prev
    print(f"Reversed: {' <-> '.join(items)}")


def demo_reverse():
    """Demonstrate list reversal."""
    ll = SinglyLinkedList()
    for x in [1, 2, 3, 4, 5]:
        ll.append(x)
    print(f"Before reverse: {ll}")
    ll.head = reverse_list(ll.head)
    print(f"After reverse:  {ll}")


def demo_middle():
    """Demonstrate finding the middle node."""
    ll = SinglyLinkedList()
    for x in [1, 2, 3, 4, 5]:
        ll.append(x)
    mid = find_middle(ll.head)
    print(f"List: {ll}")
    print(f"Middle node: {mid.data}")


def demo_cycle_detection():
    """Demonstrate cycle detection."""
    # No cycle
    ll = SinglyLinkedList()
    for x in [1, 2, 3]:
        ll.append(x)
    print(f"List {ll} has cycle: {has_cycle(ll.head)}")

    # Create a cycle manually
    a = Node(1)
    b = Node(2)
    c = Node(3)
    a.next = b
    b.next = c
    c.next = b  # Cycle: c -> b
    print(f"Cyclic list has cycle: {has_cycle(a)}")


if __name__ == "__main__":
    sections = [
        ("Singly Linked List", demo_singly_linked_list),
        ("Doubly Linked List", demo_doubly_linked_list),
        ("Reverse", demo_reverse),
        ("Find Middle", demo_middle),
        ("Cycle Detection", demo_cycle_detection),
    ]

    for title, func in sections:
        print(f"\n{'=' * 50}")
        print(f"  {title}")
        print('=' * 50)
        func()
