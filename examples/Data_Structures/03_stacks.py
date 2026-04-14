"""
03 Stacks
=========
Demonstrates stack implementations, balanced parentheses,
postfix evaluation, and monotonic stack pattern.
"""


class ArrayStack:
    """Stack implementation using a Python list."""

    def __init__(self):
        self._data = []

    def push(self, item):
        self._data.append(item)

    def pop(self):
        if self.is_empty():
            raise IndexError("pop from empty stack")
        return self._data.pop()

    def peek(self):
        if self.is_empty():
            raise IndexError("peek at empty stack")
        return self._data[-1]

    def is_empty(self):
        return len(self._data) == 0

    def __len__(self):
        return len(self._data)

    def __repr__(self):
        return f"Stack(top -> {self._data[::-1]})"


def is_balanced(expression):
    """Check if parentheses/brackets/braces are balanced."""
    stack = []
    matching = {')': '(', ']': '[', '}': '{'}
    for char in expression:
        if char in '([{':
            stack.append(char)
        elif char in ')]}':
            if not stack or stack[-1] != matching[char]:
                return False
            stack.pop()
    return len(stack) == 0


def eval_postfix(expression):
    """Evaluate a postfix expression."""
    stack = []
    for token in expression.split():
        if token.lstrip('-').isdigit():
            stack.append(int(token))
        else:
            b = stack.pop()
            a = stack.pop()
            if token == '+':
                stack.append(a + b)
            elif token == '-':
                stack.append(a - b)
            elif token == '*':
                stack.append(a * b)
            elif token == '/':
                stack.append(int(a / b))
    return stack[0]


def infix_to_postfix(expression):
    """Convert infix expression to postfix (Shunting Yard)."""
    precedence = {'+': 1, '-': 1, '*': 2, '/': 2}
    output = []
    op_stack = []
    for token in expression.split():
        if token.isdigit():
            output.append(token)
        elif token in precedence:
            while (op_stack and op_stack[-1] != '(' and
                   op_stack[-1] in precedence and
                   precedence[op_stack[-1]] >= precedence[token]):
                output.append(op_stack.pop())
            op_stack.append(token)
        elif token == '(':
            op_stack.append(token)
        elif token == ')':
            while op_stack and op_stack[-1] != '(':
                output.append(op_stack.pop())
            op_stack.pop()
    while op_stack:
        output.append(op_stack.pop())
    return ' '.join(output)


def next_greater_elements(nums):
    """Find next greater element for each position."""
    n = len(nums)
    result = [-1] * n
    stack = []
    for i in range(n):
        while stack and nums[stack[-1]] < nums[i]:
            result[stack.pop()] = nums[i]
        stack.append(i)
    return result


class MinStack:
    """Stack with O(1) minimum query."""

    def __init__(self):
        self._data = []
        self._mins = []

    def push(self, val):
        self._data.append(val)
        if not self._mins or val <= self._mins[-1]:
            self._mins.append(val)

    def pop(self):
        val = self._data.pop()
        if val == self._mins[-1]:
            self._mins.pop()
        return val

    def get_min(self):
        return self._mins[-1]


def demo_stack():
    """Demonstrate basic stack operations."""
    s = ArrayStack()
    for x in [10, 20, 30]:
        s.push(x)
    print(f"Stack: {s}")
    print(f"Peek: {s.peek()}")
    print(f"Pop: {s.pop()}")
    print(f"After pop: {s}")


def demo_balanced():
    """Demonstrate parentheses balancing."""
    tests = ["((()))", "({[]})", "(()", "([)]", "{}", ""]
    for expr in tests:
        print(f"  '{expr}' -> {is_balanced(expr)}")


def demo_postfix():
    """Demonstrate postfix evaluation."""
    cases = [
        ("3 4 2 * +", 11),
        ("3 4 + 2 *", 14),
        ("5 1 2 + 4 * + 3 -", 14),
    ]
    for expr, expected in cases:
        result = eval_postfix(expr)
        print(f"  {expr} = {result} (expected {expected})")


def demo_infix_to_postfix():
    """Demonstrate infix to postfix conversion."""
    cases = [
        "3 + 4 * 2",
        "( 3 + 4 ) * 2",
    ]
    for expr in cases:
        postfix = infix_to_postfix(expr)
        result = eval_postfix(postfix)
        print(f"  Infix: {expr}")
        print(f"  Postfix: {postfix} = {result}")


def demo_next_greater():
    """Demonstrate monotonic stack."""
    nums = [4, 5, 2, 10, 8]
    result = next_greater_elements(nums)
    print(f"Input:  {nums}")
    print(f"Output: {result}")


def demo_min_stack():
    """Demonstrate min stack."""
    ms = MinStack()
    for val in [5, 3, 7, 2, 8]:
        ms.push(val)
        print(f"  Push {val}, min = {ms.get_min()}")
    for _ in range(3):
        val = ms.pop()
        print(f"  Pop {val}, min = {ms.get_min()}")


if __name__ == "__main__":
    sections = [
        ("Basic Stack", demo_stack),
        ("Balanced Parentheses", demo_balanced),
        ("Postfix Evaluation", demo_postfix),
        ("Infix to Postfix", demo_infix_to_postfix),
        ("Next Greater Element", demo_next_greater),
        ("Min Stack", demo_min_stack),
    ]

    for title, func in sections:
        print(f"\n{'=' * 50}")
        print(f"  {title}")
        print('=' * 50)
        func()
