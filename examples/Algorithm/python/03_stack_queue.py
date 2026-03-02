"""
Stack and Queue Applications
Stack and Queue Applications

Examples of algorithms using stack (LIFO) and queue (FIFO) data structures.
"""

from collections import deque
from typing import List, Optional


# =============================================================================
# Stack Implementation (List-based)
# =============================================================================
class Stack:
    """List-based stack implementation"""

    def __init__(self):
        self._items = []

    def push(self, item):
        self._items.append(item)

    def pop(self):
        if not self.is_empty():
            return self._items.pop()
        raise IndexError("Stack is empty")

    def peek(self):
        if not self.is_empty():
            return self._items[-1]
        raise IndexError("Stack is empty")

    def is_empty(self):
        return len(self._items) == 0

    def size(self):
        return len(self._items)


# =============================================================================
# Queue Implementation (deque-based)
# =============================================================================
class Queue:
    """Deque-based queue implementation (O(1) operations)"""

    def __init__(self):
        self._items = deque()

    def enqueue(self, item):
        self._items.append(item)

    def dequeue(self):
        if not self.is_empty():
            return self._items.popleft()
        raise IndexError("Queue is empty")

    def front(self):
        if not self.is_empty():
            return self._items[0]
        raise IndexError("Queue is empty")

    def is_empty(self):
        return len(self._items) == 0

    def size(self):
        return len(self._items)


# =============================================================================
# 1. Valid Parentheses
# =============================================================================
def is_valid_parentheses(s: str) -> bool:
    """
    Validate parentheses string
    Time Complexity: O(n), Space Complexity: O(n)
    """
    stack = []
    mapping = {')': '(', '}': '{', ']': '['}

    for char in s:
        if char in mapping:
            # When encountering a closing bracket
            if not stack or stack[-1] != mapping[char]:
                return False
            stack.pop()
        else:
            # When encountering an opening bracket
            stack.append(char)

    return len(stack) == 0


# =============================================================================
# 2. Postfix Evaluation
# =============================================================================
def evaluate_postfix(expression: str) -> int:
    """
    Evaluate postfix expression
    Example: "2 3 + 4 *" = (2 + 3) * 4 = 20
    """
    stack = []
    operators = {'+', '-', '*', '/'}

    for token in expression.split():
        if token in operators:
            b = stack.pop()
            a = stack.pop()
            if token == '+':
                stack.append(a + b)
            elif token == '-':
                stack.append(a - b)
            elif token == '*':
                stack.append(a * b)
            elif token == '/':
                stack.append(int(a / b))  # Integer division
        else:
            stack.append(int(token))

    return stack[0]


# =============================================================================
# 3. Next Greater Element
# =============================================================================
def next_greater_element(arr: List[int]) -> List[int]:
    """
    Find the first greater element to the right of each element
    Returns -1 if none exists
    Time Complexity: O(n), Space Complexity: O(n)
    """
    n = len(arr)
    result = [-1] * n
    stack = []  # Store indices

    for i in range(n):
        # If current element is greater than stack top
        while stack and arr[i] > arr[stack[-1]]:
            idx = stack.pop()
            result[idx] = arr[i]
        stack.append(i)

    return result


# =============================================================================
# 4. Daily Temperatures
# =============================================================================
def daily_temperatures(temperatures: List[int]) -> List[int]:
    """
    Calculate how many days to wait until a warmer day
    Time Complexity: O(n), Space Complexity: O(n)
    """
    n = len(temperatures)
    result = [0] * n
    stack = []  # Store (index)

    for i in range(n):
        while stack and temperatures[i] > temperatures[stack[-1]]:
            prev_idx = stack.pop()
            result[prev_idx] = i - prev_idx
        stack.append(i)

    return result


# =============================================================================
# 5. Min Stack
# =============================================================================
class MinStack:
    """
    Stack that returns the minimum value in O(1) time
    """

    def __init__(self):
        self.stack = []
        self.min_stack = []

    def push(self, val: int):
        self.stack.append(val)
        if not self.min_stack or val <= self.min_stack[-1]:
            self.min_stack.append(val)

    def pop(self):
        if self.stack:
            val = self.stack.pop()
            if val == self.min_stack[-1]:
                self.min_stack.pop()

    def top(self) -> int:
        return self.stack[-1] if self.stack else None

    def get_min(self) -> int:
        return self.min_stack[-1] if self.min_stack else None


# =============================================================================
# 6. Stack Using Two Queues
# =============================================================================
class StackUsingQueues:
    """Implement a stack using two queues"""

    def __init__(self):
        self.q1 = deque()
        self.q2 = deque()

    def push(self, x: int):
        self.q2.append(x)
        while self.q1:
            self.q2.append(self.q1.popleft())
        self.q1, self.q2 = self.q2, self.q1

    def pop(self) -> int:
        return self.q1.popleft() if self.q1 else None

    def top(self) -> int:
        return self.q1[0] if self.q1 else None

    def empty(self) -> bool:
        return len(self.q1) == 0


# =============================================================================
# 7. Sliding Window Maximum (Monotonic Queue)
# =============================================================================
def max_sliding_window(nums: List[int], k: int) -> List[int]:
    """
    Find the maximum in a sliding window of size k
    Uses Monotonic Decreasing Deque
    Time Complexity: O(n), Space Complexity: O(k)
    """
    result = []
    dq = deque()  # Store indices, values are monotonically decreasing

    for i in range(len(nums)):
        # Remove indices outside the window range
        while dq and dq[0] < i - k + 1:
            dq.popleft()

        # Remove values smaller than current (from the back)
        while dq and nums[dq[-1]] < nums[i]:
            dq.pop()

        dq.append(i)

        # Add maximum once the window is complete
        if i >= k - 1:
            result.append(nums[dq[0]])

    return result


# =============================================================================
# Tests
# =============================================================================
def main():
    print("=" * 60)
    print("Stack and Queue Application Examples")
    print("=" * 60)

    # 1. Valid Parentheses
    print("\n[1] Valid Parentheses")
    test_cases = ["()", "()[]{}", "(]", "([)]", "{[]}"]
    for tc in test_cases:
        result = is_valid_parentheses(tc)
        print(f"    '{tc}' -> {result}")

    # 2. Postfix Evaluation
    print("\n[2] Postfix Evaluation")
    expressions = ["2 3 +", "2 3 + 4 *", "5 1 2 + 4 * + 3 -"]
    for expr in expressions:
        result = evaluate_postfix(expr)
        print(f"    '{expr}' = {result}")

    # 3. Next Greater Element
    print("\n[3] Next Greater Element")
    arr = [4, 5, 2, 25]
    result = next_greater_element(arr)
    print(f"    Array: {arr}")
    print(f"    Result: {result}")

    # 4. Daily Temperatures
    print("\n[4] Daily Temperatures")
    temps = [73, 74, 75, 71, 69, 72, 76, 73]
    result = daily_temperatures(temps)
    print(f"    Temperatures: {temps}")
    print(f"    Days to wait: {result}")

    # 5. Min Stack
    print("\n[5] Min Stack (MinStack)")
    min_stack = MinStack()
    operations = [
        ("push", 3), ("push", 5), ("getMin", None),
        ("push", 2), ("push", 1), ("getMin", None),
        ("pop", None), ("getMin", None)
    ]
    for op, val in operations:
        if op == "push":
            min_stack.push(val)
            print(f"    push({val})")
        elif op == "pop":
            min_stack.pop()
            print(f"    pop()")
        elif op == "getMin":
            print(f"    getMin() = {min_stack.get_min()}")

    # 6. Sliding Window Maximum
    print("\n[6] Sliding Window Maximum")
    nums = [1, 3, -1, -3, 5, 3, 6, 7]
    k = 3
    result = max_sliding_window(nums, k)
    print(f"    Array: {nums}, k={k}")
    print(f"    Window maximums: {result}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
