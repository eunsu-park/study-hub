# Stacks

**Previous**: [Linked Lists](./02_Linked_Lists.md) | **Next**: [Queues](./04_Queues.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain the LIFO (Last-In, First-Out) principle and its significance
2. Implement a stack using both arrays and linked lists
3. Analyze the time complexity of push, pop, and peek operations
4. Apply stacks to solve expression evaluation and parenthesis matching
5. Understand how the call stack works in program execution
6. Use Python's built-in list as a stack and know its limitations
7. Implement and apply the monotonic stack pattern

---

A **stack** is a linear data structure that follows the **Last-In, First-Out (LIFO)** principle. The last element added is the first one removed, like a stack of plates.

## The Stack ADT

```
        +-------+
        |  Top  |  <-- push/pop happen here
        +-------+
        |       |
        +-------+
        |       |
        +-------+
        |       |
        +-------+
        | Bottom|
        +-------+

Push 10, Push 20, Push 30:      Pop:
+----+                          +----+
| 30 | <-- top                  | 20 | <-- top (30 removed)
+----+                          +----+
| 20 |                          | 10 |
+----+                          +----+
| 10 |
+----+
```

### Core Operations

| Operation | Description | Time |
|-----------|-------------|------|
| `push(item)` | Add item to the top | O(1) |
| `pop()` | Remove and return the top item | O(1) |
| `peek()` / `top()` | Return the top item without removing | O(1) |
| `is_empty()` | Check if the stack is empty | O(1) |
| `size()` | Return the number of elements | O(1) |

## Array-Based Stack

The simplest stack implementation uses a Python list, treating the end as the top:

```python
class ArrayStack:
    """Stack implementation using a Python list."""
    
    def __init__(self):
        self._data = []
    
    def push(self, item):
        """Add item to the top -- O(1) amortized."""
        self._data.append(item)
    
    def pop(self):
        """Remove and return the top item -- O(1)."""
        if self.is_empty():
            raise IndexError("pop from empty stack")
        return self._data.pop()
    
    def peek(self):
        """Return the top item without removing -- O(1)."""
        if self.is_empty():
            raise IndexError("peek at empty stack")
        return self._data[-1]
    
    def is_empty(self):
        return len(self._data) == 0
    
    def __len__(self):
        return len(self._data)
    
    def __repr__(self):
        return f"Stack(top -> {self._data[::-1]})"
```

## Linked-List-Based Stack

Using a singly linked list with push/pop at the head:

```python
class LinkedStack:
    """Stack implementation using a singly linked list."""
    
    def __init__(self):
        self._head = None
        self._size = 0
    
    def push(self, item):
        """Add item to the top -- O(1)."""
        self._head = Node(item, self._head)
        self._size += 1
    
    def pop(self):
        """Remove and return the top item -- O(1)."""
        if self.is_empty():
            raise IndexError("pop from empty stack")
        data = self._head.data
        self._head = self._head.next
        self._size -= 1
        return data
    
    def peek(self):
        """Return the top item without removing -- O(1)."""
        if self.is_empty():
            raise IndexError("peek at empty stack")
        return self._head.data
    
    def is_empty(self):
        return self._head is None
    
    def __len__(self):
        return self._size
```

## Application 1: Balanced Parentheses

One of the most classic stack applications:

```python
def is_balanced(expression):
    """Check if parentheses/brackets/braces are balanced.
    
    >>> is_balanced("((()))")
    True
    >>> is_balanced("({[]})")
    True
    >>> is_balanced("(()")
    False
    >>> is_balanced("([)]")
    False
    """
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
```

### How It Works

```
Input: "({[]})"

Step 1: '(' -> push    Stack: ['(']
Step 2: '{' -> push    Stack: ['(', '{']
Step 3: '[' -> push    Stack: ['(', '{', '[']
Step 4: ']' -> match   Stack: ['(', '{']        ('[' matches ']')
Step 5: '}' -> match   Stack: ['(']             ('{' matches '}')
Step 6: ')' -> match   Stack: []                ('(' matches ')')
Result: Stack empty -> Balanced!
```

## Application 2: Expression Evaluation

### Infix to Postfix Conversion (Shunting Yard Algorithm)

```
Infix:    3 + 4 * 2
Postfix:  3 4 2 * +

Infix:    (3 + 4) * 2
Postfix:  3 4 + 2 *
```

```python
def infix_to_postfix(expression):
    """Convert infix expression to postfix using the Shunting Yard algorithm."""
    precedence = {'+': 1, '-': 1, '*': 2, '/': 2, '**': 3}
    right_assoc = {'**'}
    output = []
    operator_stack = []
    
    tokens = expression.split()
    for token in tokens:
        if token.isdigit():
            output.append(token)
        elif token in precedence:
            while (operator_stack and 
                   operator_stack[-1] != '(' and
                   operator_stack[-1] in precedence and
                   (precedence[operator_stack[-1]] > precedence[token] or
                    (precedence[operator_stack[-1]] == precedence[token] 
                     and token not in right_assoc))):
                output.append(operator_stack.pop())
            operator_stack.append(token)
        elif token == '(':
            operator_stack.append(token)
        elif token == ')':
            while operator_stack and operator_stack[-1] != '(':
                output.append(operator_stack.pop())
            operator_stack.pop()  # Remove '('
    
    while operator_stack:
        output.append(operator_stack.pop())
    
    return ' '.join(output)
```

### Postfix Evaluation

```python
def eval_postfix(expression):
    """Evaluate a postfix expression.
    
    >>> eval_postfix("3 4 2 * +")
    11
    >>> eval_postfix("3 4 + 2 *")
    14
    """
    stack = []
    for token in expression.split():
        if token.lstrip('-').isdigit():
            stack.append(int(token))
        else:
            b = stack.pop()
            a = stack.pop()
            if token == '+': stack.append(a + b)
            elif token == '-': stack.append(a - b)
            elif token == '*': stack.append(a * b)
            elif token == '/': stack.append(int(a / b))
    return stack[0]
```

## Application 3: The Call Stack

Every running program uses a stack to manage function calls:

```
def main():           Call Stack:
    a()                +----------+
                       | main()   |
def a():               +----------+
    b()
                       +----------+
def b():               | b()      |  <-- currently executing
    c()                +----------+
                       | a()      |
def c():               +----------+
    pass               | main()   |
                       +----------+
```

When a function returns, its frame is popped from the stack. This is why excessive recursion causes a **stack overflow** -- the call stack runs out of space.

```python
import sys
print(sys.getrecursionlimit())  # Default: 1000

# This will cause RecursionError:
def infinite_recursion():
    return infinite_recursion()
```

## Application 4: Undo/Redo

```python
class TextEditor:
    """Simple text editor with undo/redo using two stacks."""
    
    def __init__(self):
        self.text = ""
        self._undo_stack = []
        self._redo_stack = []
    
    def type_text(self, new_text):
        self._undo_stack.append(self.text)
        self._redo_stack.clear()
        self.text += new_text
    
    def undo(self):
        if self._undo_stack:
            self._redo_stack.append(self.text)
            self.text = self._undo_stack.pop()
    
    def redo(self):
        if self._redo_stack:
            self._undo_stack.append(self.text)
            self.text = self._redo_stack.pop()
```

## Monotonic Stack

A monotonic stack maintains elements in sorted order (increasing or decreasing). Useful for "next greater element" problems:

```python
def next_greater_elements(nums):
    """For each element, find the next element that is greater.
    
    >>> next_greater_elements([4, 5, 2, 10, 8])
    [5, 10, 10, -1, -1]
    """
    n = len(nums)
    result = [-1] * n
    stack = []  # Stack of indices
    
    for i in range(n):
        while stack and nums[stack[-1]] < nums[i]:
            result[stack.pop()] = nums[i]
        stack.append(i)
    
    return result
```

```
Input: [4, 5, 2, 10, 8]

i=0: push 0          Stack: [0]         result: [-1,-1,-1,-1,-1]
i=1: 5>4, pop 0      Stack: []          result: [5,-1,-1,-1,-1]
     push 1          Stack: [1]
i=2: 2<5, push 2     Stack: [1,2]       result: [5,-1,-1,-1,-1]
i=3: 10>2, pop 2     Stack: [1]         result: [5,-1,10,-1,-1]
     10>5, pop 1     Stack: []          result: [5,10,10,-1,-1]
     push 3          Stack: [3]
i=4: 8<10, push 4    Stack: [3,4]       result: [5,10,10,-1,-1]
```

## Min Stack

A stack that supports O(1) minimum query:

```python
class MinStack:
    """Stack with O(1) push, pop, and get_min."""
    
    def __init__(self):
        self._data = []
        self._mins = []  # Parallel stack tracking minimums
    
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
```

## Python's Built-in Stack Options

```python
# Option 1: list (most common)
stack = []
stack.append(1)  # push
stack.pop()      # pop

# Option 2: collections.deque (thread-safe, no reallocation)
from collections import deque
stack = deque()
stack.append(1)  # push
stack.pop()      # pop

# Option 3: queue.LifoQueue (thread-safe, blocking)
from queue import LifoQueue
stack = LifoQueue()
stack.put(1)     # push
stack.get()      # pop (blocks if empty)
```

## Summary

| Concept | Key Takeaway |
|---------|-------------|
| LIFO | Last-In, First-Out principle |
| Core ops | push, pop, peek -- all O(1) |
| Array-based | Simple, cache-friendly, amortized O(1) push |
| Linked-based | No capacity limit, guaranteed O(1) push |
| Parentheses matching | Classic stack application |
| Expression evaluation | Shunting Yard + postfix evaluation |
| Call stack | How programs manage function calls |
| Monotonic stack | "Next greater element" pattern |

---

**Next**: [Queues](./04_Queues.md) -- Explore the FIFO counterpart to stacks.
