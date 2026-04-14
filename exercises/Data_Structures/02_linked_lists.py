"""
Exercise 02: Linked Lists

Practice linked list operations: reversal, cycle detection,
merge, and nth-from-end.
"""


class Node:
    def __init__(self, data, next_node=None):
        self.data = data
        self.next = next_node


def list_to_linked(lst):
    """Helper: convert a Python list to a linked list."""
    if not lst:
        return None
    head = Node(lst[0])
    cur = head
    for val in lst[1:]:
        cur.next = Node(val)
        cur = cur.next
    return head


def linked_to_list(head):
    """Helper: convert a linked list to a Python list."""
    result = []
    while head:
        result.append(head.data)
        head = head.next
    return result


def reverse_linked_list(head):
    """Reverse a singly linked list in-place.

    Args:
        head: Head node of the linked list.

    Returns:
        New head node of the reversed list.

    >>> linked_to_list(reverse_linked_list(list_to_linked([1, 2, 3, 4, 5])))
    [5, 4, 3, 2, 1]
    """
    # TODO: Implement this
    pass


def find_nth_from_end(head, n):
    """Find the nth node from the end of a linked list.

    Use the two-pointer technique (O(1) space).

    Args:
        head: Head node.
        n: Position from end (1-based).

    Returns:
        Data value of the nth node from end, or None if n is invalid.

    >>> find_nth_from_end(list_to_linked([1, 2, 3, 4, 5]), 2)
    4
    """
    # TODO: Implement this
    pass


def merge_sorted_lists(l1, l2):
    """Merge two sorted linked lists into one sorted linked list.

    Args:
        l1: Head of first sorted linked list.
        l2: Head of second sorted linked list.

    Returns:
        Head of merged sorted linked list.

    >>> linked_to_list(merge_sorted_lists(list_to_linked([1, 3, 5]), list_to_linked([2, 4, 6])))
    [1, 2, 3, 4, 5, 6]
    """
    # TODO: Implement this
    pass


def has_cycle(head):
    """Detect if a linked list has a cycle.

    Use Floyd's tortoise and hare algorithm.

    Args:
        head: Head node.

    Returns:
        True if cycle exists, False otherwise.
    """
    # TODO: Implement this
    pass


def remove_duplicates(head):
    """Remove duplicates from a sorted linked list.

    Args:
        head: Head of sorted linked list.

    Returns:
        Head of list with duplicates removed.

    >>> linked_to_list(remove_duplicates(list_to_linked([1, 1, 2, 3, 3])))
    [1, 2, 3]
    """
    # TODO: Implement this
    pass


if __name__ == "__main__":
    # Test reverse
    assert linked_to_list(reverse_linked_list(list_to_linked([1, 2, 3, 4, 5]))) == [5, 4, 3, 2, 1]
    assert reverse_linked_list(None) is None
    print("reverse_linked_list: PASSED")

    # Test nth from end
    assert find_nth_from_end(list_to_linked([1, 2, 3, 4, 5]), 2) == 4
    assert find_nth_from_end(list_to_linked([1, 2, 3, 4, 5]), 1) == 5
    assert find_nth_from_end(list_to_linked([1, 2, 3, 4, 5]), 5) == 1
    print("find_nth_from_end: PASSED")

    # Test merge sorted
    assert linked_to_list(merge_sorted_lists(
        list_to_linked([1, 3, 5]), list_to_linked([2, 4, 6]))) == [1, 2, 3, 4, 5, 6]
    assert linked_to_list(merge_sorted_lists(None, list_to_linked([1, 2]))) == [1, 2]
    print("merge_sorted_lists: PASSED")

    # Test has_cycle
    assert has_cycle(list_to_linked([1, 2, 3])) is False
    a = Node(1); b = Node(2); c = Node(3)
    a.next = b; b.next = c; c.next = b
    assert has_cycle(a) is True
    print("has_cycle: PASSED")

    # Test remove duplicates
    assert linked_to_list(remove_duplicates(list_to_linked([1, 1, 2, 3, 3]))) == [1, 2, 3]
    print("remove_duplicates: PASSED")

    print("\nAll tests passed!")
