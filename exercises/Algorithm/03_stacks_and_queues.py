"""
Exercises for Lesson 03: Stacks and Queues
Topic: Algorithm

Solutions to practice problems from the lesson.
"""

from collections import deque


# === Exercise 1: Min Stack ===
# Problem: Implement a stack with push, pop, top operations and getMin in O(1).
# Approach: Store (value, current_minimum) pairs so we always know the minimum
#           at every stack height without rescanning.

def exercise_1():
    """Solution: Min Stack with O(1) getMin."""
    class MinStack:
        def __init__(self):
            # Each element is a tuple: (value, min_at_this_point)
            self.stack = []

        def push(self, x):
            current_min = min(x, self.stack[-1][1]) if self.stack else x
            self.stack.append((x, current_min))

        def pop(self):
            return self.stack.pop()[0]

        def top(self):
            return self.stack[-1][0]

        def get_min(self):
            return self.stack[-1][1]

        def is_empty(self):
            return len(self.stack) == 0

    ms = MinStack()

    # Test sequence
    ms.push(5)
    print(f"Push 5, min = {ms.get_min()}")
    assert ms.get_min() == 5

    ms.push(3)
    print(f"Push 3, min = {ms.get_min()}")
    assert ms.get_min() == 3

    ms.push(7)
    print(f"Push 7, min = {ms.get_min()}")
    assert ms.get_min() == 3

    ms.push(1)
    print(f"Push 1, min = {ms.get_min()}")
    assert ms.get_min() == 1

    val = ms.pop()
    print(f"Pop {val}, min = {ms.get_min()}")
    assert val == 1
    assert ms.get_min() == 3

    val = ms.pop()
    print(f"Pop {val}, min = {ms.get_min()}")
    assert val == 7
    assert ms.get_min() == 3

    val = ms.pop()
    print(f"Pop {val}, min = {ms.get_min()}")
    assert val == 3
    assert ms.get_min() == 5

    print("All MinStack tests passed!")


# === Exercise 2: Implement Stack Using Two Queues ===
# Problem: Implement a stack using two queues.
# Approach: On push, put the new element in q2, move all elements from q1 to q2,
#           then swap q1 and q2. This ensures q1 always has the most recent
#           element at the front, giving O(1) pop/top and O(n) push.

def exercise_2():
    """Solution: Stack using two queues."""
    class StackUsingQueues:
        def __init__(self):
            self.q1 = deque()
            self.q2 = deque()

        def push(self, x):
            # Put new element in q2 first, then drain q1 into q2.
            # This places x before all previous elements.
            self.q2.append(x)
            while self.q1:
                self.q2.append(self.q1.popleft())
            # Swap so q1 is always the "active" queue
            self.q1, self.q2 = self.q2, self.q1

        def pop(self):
            return self.q1.popleft()

        def top(self):
            return self.q1[0]

        def empty(self):
            return len(self.q1) == 0

    s = StackUsingQueues()

    # Test LIFO behavior
    s.push(1)
    s.push(2)
    s.push(3)
    print(f"Pushed 1, 2, 3")

    val = s.pop()
    print(f"Pop: {val}")
    assert val == 3, f"Expected 3, got {val}"

    val = s.top()
    print(f"Top: {val}")
    assert val == 2, f"Expected 2, got {val}"

    val = s.pop()
    print(f"Pop: {val}")
    assert val == 2, f"Expected 2, got {val}"

    print(f"Empty: {s.empty()}")
    assert not s.empty()

    val = s.pop()
    print(f"Pop: {val}")
    assert val == 1, f"Expected 1, got {val}"

    print(f"Empty: {s.empty()}")
    assert s.empty()

    print("All Stack-using-Queues tests passed!")


if __name__ == "__main__":
    print("=== Exercise 1: Min Stack ===")
    exercise_1()
    print("\n=== Exercise 2: Stack Using Two Queues ===")
    exercise_2()
    print("\nAll exercises completed!")
