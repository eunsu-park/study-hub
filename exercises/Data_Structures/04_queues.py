"""
Exercise 04: Queues

Practice queue implementations and applications.
"""

from collections import deque


class CircularQueue:
    """Implement a circular queue with fixed capacity.

    >>> cq = CircularQueue(3)
    >>> cq.enqueue(1); cq.enqueue(2); cq.enqueue(3)
    >>> cq.is_full()
    True
    >>> cq.dequeue()
    1
    >>> cq.enqueue(4)
    >>> cq.dequeue()
    2
    """

    def __init__(self, capacity):
        # TODO: Initialize the circular queue
        pass

    def enqueue(self, item):
        """Add item to the rear. Raise OverflowError if full."""
        # TODO: Implement this
        pass

    def dequeue(self):
        """Remove and return front item. Raise IndexError if empty."""
        # TODO: Implement this
        pass

    def peek(self):
        """Return front item without removing."""
        # TODO: Implement this
        pass

    def is_empty(self):
        # TODO: Implement this
        pass

    def is_full(self):
        # TODO: Implement this
        pass

    def __len__(self):
        # TODO: Implement this
        pass


def hot_potato(names, num_passes):
    """Simulate the hot potato game.

    Players stand in a circle. The potato is passed num_passes times.
    The person holding it is eliminated. Repeat until one remains.

    Args:
        names: List of player names.
        num_passes: Number of passes per round.

    Returns:
        Name of the winner.

    >>> hot_potato(["A", "B", "C", "D", "E"], 3)
    'D'
    """
    # TODO: Implement this using a deque
    pass


def sliding_window_max(nums, k):
    """Find the maximum in each sliding window of size k.

    Use a monotonic deque for O(n) time.

    >>> sliding_window_max([1, 3, -1, -3, 5, 3, 6, 7], 3)
    [3, 3, 5, 5, 6, 7]
    """
    # TODO: Implement this
    pass


if __name__ == "__main__":
    # Test CircularQueue
    cq = CircularQueue(3)
    cq.enqueue(1); cq.enqueue(2); cq.enqueue(3)
    assert cq.is_full()
    assert cq.dequeue() == 1
    cq.enqueue(4)
    assert cq.dequeue() == 2
    assert cq.dequeue() == 3
    assert cq.dequeue() == 4
    assert cq.is_empty()
    print("CircularQueue: PASSED")

    # Test hot_potato
    assert hot_potato(["A", "B", "C", "D", "E"], 3) == "D"
    print("hot_potato: PASSED")

    # Test sliding_window_max
    assert sliding_window_max([1, 3, -1, -3, 5, 3, 6, 7], 3) == [3, 3, 5, 5, 6, 7]
    assert sliding_window_max([1], 1) == [1]
    print("sliding_window_max: PASSED")

    print("\nAll tests passed!")
