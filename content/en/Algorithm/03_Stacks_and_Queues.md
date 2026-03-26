# Stack and Queue Applications

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain the LIFO and FIFO principles of stacks and queues and identify their O(1) core operations
2. Implement parenthesis validation using a stack to detect mismatched or unclosed brackets
3. Evaluate postfix (Reverse Polish Notation) expressions by applying the stack-based evaluation algorithm
4. Apply the monotonic stack pattern to solve next-greater-element and largest-rectangle problems efficiently
5. Use a queue to implement breadth-first search (BFS) for level-order traversal and shortest-path problems
6. Leverage a deque to solve sliding-window maximum problems in O(n) time

---

## Overview

Stacks and queues are fundamental data structures, but they play crucial roles in various algorithm problems. This lesson covers frequently used patterns in practice such as parenthesis checking, postfix notation, and monotonic stacks.

---

## Table of Contents

1. [Stack and Queue Basics](#1-stack-and-queue-basics)
2. [Parenthesis Validation](#2-parenthesis-validation)
3. [Postfix Notation](#3-postfix-notation)
4. [Monotonic Stack](#4-monotonic-stack)
5. [Queue and BFS](#5-queue-and-bfs)
6. [Deque Applications](#6-deque-applications)
7. [Practice Problems](#7-practice-problems)

---

## 1. Stack and Queue Basics

### Stack - LIFO

```
LIFO: Last In, First Out

        push(3)     push(7)     pop()
           │           │          │
           ▼           ▼          ▼
        ┌───┐       ┌───┐      ┌───┐
        │   │       │ 7 │      │   │
        ├───┤       ├───┤      ├───┤
        │ 3 │       │ 3 │      │ 3 │
        └───┘       └───┘      └───┘

Key operations (all O(1)):
- push(x): Insert element
- pop(): Remove and return top element
- top()/peek(): View top element
- empty(): Check if empty
```

```c
// C - Array-based stack
#define MAX_SIZE 1000

typedef struct {
    int data[MAX_SIZE];
    int top;
} Stack;

void init(Stack* s) { s->top = -1; }
int isEmpty(Stack* s) { return s->top == -1; }
int isFull(Stack* s) { return s->top == MAX_SIZE - 1; }

void push(Stack* s, int x) {
    if (!isFull(s)) {
        s->data[++s->top] = x;
    }
}

int pop(Stack* s) {
    if (!isEmpty(s)) {
        return s->data[s->top--];
    }
    return -1;  // Error
}

int top(Stack* s) {
    if (!isEmpty(s)) {
        return s->data[s->top];
    }
    return -1;
}
```

```cpp
// C++ - STL stack
#include <stack>

stack<int> s;
s.push(3);      // Insert
s.push(7);
s.top();        // 7 (view top)
s.pop();        // Remove (no return value!)
s.empty();      // false
s.size();       // 1
```

```python
# Python - List as stack
stack = []
stack.append(3)   # push
stack.append(7)
stack[-1]         # top: 7
stack.pop()       # 7 (return and remove)
len(stack) == 0   # check empty
```

### Queue - FIFO

```
FIFO: First In, First Out

enqueue(1)  enqueue(2)  enqueue(3)  dequeue()
    │           │           │           │
    ▼           ▼           ▼           ▼
  ┌───┐      ┌───┬───┐   ┌───┬───┬───┐  ┌───┬───┐
  │ 1 │      │ 1 │ 2 │   │ 1 │ 2 │ 3 │  │ 2 │ 3 │
  └───┘      └───┴───┘   └───┴───┴───┘  └───┴───┘
  front                   front          front
                                back           back

Key operations (all O(1)):
- enqueue(x)/push(x): Insert at back
- dequeue()/pop(): Remove from front
- front(): View front element
- empty(): Check if empty
```

```cpp
// C++ - STL queue
#include <queue>

queue<int> q;
q.push(1);      // enqueue
q.push(2);
q.push(3);
q.front();      // 1 (front)
q.back();       // 3 (back)
q.pop();        // dequeue (no return value!)
q.empty();      // false
```

```python
# Python - collections.deque (recommended)
from collections import deque

q = deque()
q.append(1)     # enqueue
q.append(2)
q.append(3)
q[0]            # front: 1
q[-1]           # back: 3
q.popleft()     # dequeue: 1 (return and remove)
```

---

## 2. Parenthesis Validation

### Problem

Check if parentheses in a given string are properly matched.

```
Valid: "()", "()[]{}", "{[()]}"
Invalid: "(]", "([)]", "((("
```

### Algorithm

```
String: "{[()]}"

Char  Stack       Action
----  ----        ----
{     [           push '{'
[     [{          push '['
(     [{(         push '('
)     [{          pop '(' ← matches ')'
]     [           pop '[' ← matches ']'
}     []          pop '{' ← matches '}'

Stack is empty → Valid!
```

### Implementation

```c
// C
#include <stdbool.h>
#include <string.h>

bool isValid(const char* s) {
    char stack[10000];
    int top = -1;

    for (int i = 0; s[i] != '\0'; i++) {
        char c = s[i];

        if (c == '(' || c == '[' || c == '{') {
            // Push opening brackets; a stack is perfect here because the most
            // recently opened bracket must be closed first (LIFO matches nesting)
            stack[++top] = c;
        } else {
            // An unmatched closing bracket with nothing on the stack means the
            // string is invalid before we even check the bracket type
            if (top == -1) return false;

            char topChar = stack[top--];

            if (c == ')' && topChar != '(') return false;
            if (c == ']' && topChar != '[') return false;
            if (c == '}' && topChar != '{') return false;
        }
    }

    // A non-empty stack means there are unclosed openers — also invalid
    return top == -1;
}
```

```cpp
// C++
bool isValid(const string& s) {
    stack<char> st;
    unordered_map<char, char> pairs = {
        {')', '('},
        {']', '['},
        {'}', '{'}
    };

    for (char c : s) {
        if (c == '(' || c == '[' || c == '{') {
            st.push(c);
        } else {
            if (st.empty() || st.top() != pairs[c]) {
                return false;
            }
            st.pop();
        }
    }

    return st.empty();
}
```

```python
# Python
def is_valid(s):
    stack = []
    pairs = {')': '(', ']': '[', '}': '{'}

    for c in s:
        if c in '([{':
            stack.append(c)
        else:
            if not stack or stack[-1] != pairs[c]:
                return False
            stack.pop()

    return len(stack) == 0
```

---

## 3. Postfix Notation

### Notation Types

```
Infix:   3 + 4 * 2
Prefix:  + 3 * 4 2
Postfix: 3 4 2 * +

Postfix advantages:
- No parentheses needed
- No operator precedence consideration
- Easy for computers to evaluate
```

### 3.1 Evaluate Postfix Expression

```
Expression: "3 4 2 * +"

Step  Token  Stack       Action
----  ----   ----        ----
1     3      [3]         push 3
2     4      [3,4]       push 4
3     2      [3,4,2]     push 2
4     *      [3,8]       pop 2,4 → push 4*2=8
5     +      [11]        pop 8,3 → push 3+8=11

Result: 11
```

```c
// C - Evaluate postfix (single digits)
int evaluatePostfix(const char* expr) {
    int stack[100];
    int top = -1;

    for (int i = 0; expr[i] != '\0'; i++) {
        char c = expr[i];

        if (c >= '0' && c <= '9') {
            stack[++top] = c - '0';
        } else if (c == '+' || c == '-' || c == '*' || c == '/') {
            int b = stack[top--];
            int a = stack[top--];

            int result;
            switch (c) {
                case '+': result = a + b; break;
                case '-': result = a - b; break;
                case '*': result = a * b; break;
                case '/': result = a / b; break;
            }
            stack[++top] = result;
        }
    }

    return stack[top];
}
```

```cpp
// C++
int evaluatePostfix(const string& expr) {
    stack<int> st;

    for (char c : expr) {
        if (isdigit(c)) {
            st.push(c - '0');
        } else if (c == '+' || c == '-' || c == '*' || c == '/') {
            int b = st.top(); st.pop();
            int a = st.top(); st.pop();

            int result;
            switch (c) {
                case '+': result = a + b; break;
                case '-': result = a - b; break;
                case '*': result = a * b; break;
                case '/': result = a / b; break;
            }
            st.push(result);
        }
    }

    return st.top();
}
```

```python
# Python
def evaluate_postfix(expr):
    stack = []
    operators = {'+', '-', '*', '/'}

    for token in expr.split():
        if token not in operators:
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
                stack.append(int(a / b))  # Integer division

    return stack[-1]
```

### 3.2 Infix to Postfix Conversion (Shunting-yard Algorithm)

```
Expression: "3 + 4 * 2"

Operator precedence: * / > + -

Step  Token  Output      Stack    Action
----  ----   ----        ----     ----
1     3      3           []       Number → output
2     +      3           [+]      Operator → stack
3     4      3 4         [+]      Number → output
4     *      3 4         [+,*]    * has higher precedence → push
5     2      3 4 2       [+,*]    Number → output
End          3 4 2 * +   []       Empty stack

Result: "3 4 2 * +"
```

```cpp
// C++
int precedence(char op) {
    if (op == '+' || op == '-') return 1;
    if (op == '*' || op == '/') return 2;
    return 0;
}

string infixToPostfix(const string& expr) {
    string output;
    stack<char> ops;

    for (char c : expr) {
        if (isalnum(c)) {
            output += c;
            output += ' ';
        } else if (c == '(') {
            ops.push(c);
        } else if (c == ')') {
            while (!ops.empty() && ops.top() != '(') {
                output += ops.top();
                output += ' ';
                ops.pop();
            }
            ops.pop();  // Remove '('
        } else if (c == '+' || c == '-' || c == '*' || c == '/') {
            while (!ops.empty() && precedence(ops.top()) >= precedence(c)) {
                output += ops.top();
                output += ' ';
                ops.pop();
            }
            ops.push(c);
        }
    }

    while (!ops.empty()) {
        output += ops.top();
        output += ' ';
        ops.pop();
    }

    return output;
}
```

```python
# Python
def infix_to_postfix(expr):
    precedence = {'+': 1, '-': 1, '*': 2, '/': 2}
    output = []
    ops = []

    for token in expr.split():
        if token.isalnum():
            output.append(token)
        elif token == '(':
            ops.append(token)
        elif token == ')':
            while ops and ops[-1] != '(':
                output.append(ops.pop())
            ops.pop()  # Remove '('
        elif token in precedence:
            while (ops and ops[-1] != '(' and
                   ops[-1] in precedence and
                   precedence[ops[-1]] >= precedence[token]):
                output.append(ops.pop())
            ops.append(token)

    while ops:
        output.append(ops.pop())

    return ' '.join(output)
```

---

## 4. Monotonic Stack

### Concept

```
Monotonic Stack: Stack with monotonically increasing or decreasing elements
→ Solves "next/previous greater or smaller element" problems in O(n)

Types:
1. Monotonic increasing stack: Keep smaller elements (find next greater)
2. Monotonic decreasing stack: Keep larger elements (find next smaller)
```

### 4.1 Next Greater Element

```
Problem: For each element, find the first element to its right that is greater

Array: [2, 1, 2, 4, 3]

Index  Value  Stack(idx)  NextGreater  Action
------  ----  ----------  -----------  ----
0       2     [0]         -            push
1       1     [0,1]       -            1<2, push
2       2     [0,2]       2            pop 1 (2>1), push
3       4     [3]         4,4,4        pop all (4>2, 4>2), push
4       3     [3,4]       -            3<4, push
End           []          -1,-1        Remaining are -1

Result: [4, 2, 4, -1, -1]
```

```c
// C
void nextGreaterElement(int arr[], int n, int result[]) {
    int stack[n];
    int top = -1;

    for (int i = 0; i < n; i++) {
        result[i] = -1;  // Default
    }

    for (int i = 0; i < n; i++) {
        // If current > stack top, pop and store result
        while (top >= 0 && arr[stack[top]] < arr[i]) {
            result[stack[top--]] = arr[i];
        }
        stack[++top] = i;
    }
}
```

```cpp
// C++
vector<int> nextGreaterElement(const vector<int>& arr) {
    int n = arr.size();
    vector<int> result(n, -1);
    stack<int> st;  // Store indices

    for (int i = 0; i < n; i++) {
        while (!st.empty() && arr[st.top()] < arr[i]) {
            result[st.top()] = arr[i];
            st.pop();
        }
        st.push(i);
    }

    return result;
}
```

```python
# Python
def next_greater_element(arr):
    n = len(arr)
    result = [-1] * n
    stack = []  # Store indices

    for i in range(n):
        while stack and arr[stack[-1]] < arr[i]:
            result[stack.pop()] = arr[i]
        stack.append(i)

    return result
```

### 4.2 Largest Rectangle in Histogram

```
Problem: Find the largest rectangle area in a histogram

Heights: [2, 1, 5, 6, 2, 3]

    ┌───┐
    │   │
┌───┤   │
│   │   │       ┌───┐
│   │   │───┬───┤   │
│   │   │   │   │   │
└───┴───┴───┴───┴───┘
  2   1   5   6   2   3

Largest rectangle: height 5, width 2 → area 10
```

```cpp
// C++
int largestRectangleArea(const vector<int>& heights) {
    int n = heights.size();
    stack<int> st;  // Stores indices of bars in ascending height order
    int maxArea = 0;

    // Appending a sentinel height 0 at i == n forces all remaining bars on the
    // stack to be popped and processed — avoids a separate post-loop flush
    for (int i = 0; i <= n; i++) {
        int h = (i == n) ? 0 : heights[i];

        // Pop every bar taller than h: h is the right boundary that limits those
        // bars, so this is their last chance to compute their maximum rectangle
        while (!st.empty() && heights[st.top()] > h) {
            int height = heights[st.top()];
            st.pop();

            // Width extends to the nearest shorter bar on both sides — the new
            // stack top is the closest shorter bar to the left, and i is the
            // closest shorter bar to the right
            int width = st.empty() ? i : i - st.top() - 1;
            maxArea = max(maxArea, height * width);
        }

        st.push(i);
    }

    return maxArea;
}
```

```python
# Python
def largest_rectangle_area(heights):
    stack = []
    max_area = 0
    n = len(heights)

    for i in range(n + 1):
        h = heights[i] if i < n else 0

        while stack and heights[stack[-1]] > h:
            height = heights[stack.pop()]
            width = i if not stack else i - stack[-1] - 1
            max_area = max(max_area, height * width)

        stack.append(i)

    return max_area
```

### 4.3 Daily Temperatures

```
Problem: For each day, how many days until a warmer temperature

Temperatures: [73, 74, 75, 71, 69, 72, 76, 73]
Result:       [1,  1,  4,  2,  1,  1,  0,  0]

Explanation:
- 73 → 1 day until 74
- 74 → 1 day until 75
- 75 → 4 days until 76
- ...
```

```cpp
// C++
vector<int> dailyTemperatures(const vector<int>& temperatures) {
    int n = temperatures.size();
    vector<int> result(n, 0);
    stack<int> st;

    for (int i = 0; i < n; i++) {
        while (!st.empty() && temperatures[st.top()] < temperatures[i]) {
            int prevIdx = st.top();
            st.pop();
            result[prevIdx] = i - prevIdx;
        }
        st.push(i);
    }

    return result;
}
```

```python
# Python
def daily_temperatures(temperatures):
    n = len(temperatures)
    result = [0] * n
    stack = []

    for i in range(n):
        while stack and temperatures[stack[-1]] < temperatures[i]:
            prev_idx = stack.pop()
            result[prev_idx] = i - prev_idx
        stack.append(i)

    return result
```

---

## 5. Queue and BFS

### BFS Basic Structure

```
BFS (Breadth-First Search): Use queue to explore nearest nodes first

       1
      /|\
     2 3 4
    /|   |
   5 6   7

BFS order: 1 → 2 → 3 → 4 → 5 → 6 → 7
```

```cpp
// C++ - BFS template
void bfs(vector<vector<int>>& graph, int start) {
    int n = graph.size();
    vector<bool> visited(n, false);
    queue<int> q;

    q.push(start);
    // Mark visited *before* enqueuing, not after dequeuing — this prevents the
    // same node from being enqueued multiple times via different neighbors,
    // which would corrupt the level-order guarantee and waste O(n) work
    visited[start] = true;

    while (!q.empty()) {
        int node = q.front();
        q.pop();  // Process in FIFO order — guarantees shortest-path distances

        cout << node << " ";

        for (int neighbor : graph[node]) {
            if (!visited[neighbor]) {
                visited[neighbor] = true;
                q.push(neighbor);
            }
        }
    }
}
```

```python
# Python
from collections import deque

def bfs(graph, start):
    visited = set()
    queue = deque([start])
    visited.add(start)

    while queue:
        node = queue.popleft()
        print(node, end=' ')

        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
```

### Shortest Path BFS (2D Grid)

```
Problem: Find shortest distance from start (S) to end (E) in a maze

Maze:
S . . #
# . # .
. . . E

BFS exploration:
1 2 3 #
# 3 # 5
5 4 5 6

Shortest distance: 6
```

```cpp
// C++
int shortestPath(vector<vector<char>>& grid) {
    int m = grid.size(), n = grid[0].size();
    int dx[] = {0, 0, 1, -1};
    int dy[] = {1, -1, 0, 0};

    // Find start
    int startX, startY;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            if (grid[i][j] == 'S') {
                startX = i;
                startY = j;
            }
        }
    }

    queue<tuple<int, int, int>> q;  // x, y, distance
    vector<vector<bool>> visited(m, vector<bool>(n, false));

    q.push({startX, startY, 0});
    visited[startX][startY] = true;

    while (!q.empty()) {
        auto [x, y, dist] = q.front();
        q.pop();

        if (grid[x][y] == 'E') {
            return dist;
        }

        for (int i = 0; i < 4; i++) {
            int nx = x + dx[i];
            int ny = y + dy[i];

            if (nx >= 0 && nx < m && ny >= 0 && ny < n &&
                !visited[nx][ny] && grid[nx][ny] != '#') {
                visited[nx][ny] = true;
                q.push({nx, ny, dist + 1});
            }
        }
    }

    return -1;  // Unreachable
}
```

```python
# Python
from collections import deque

def shortest_path(grid):
    m, n = len(grid), len(grid[0])
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    # Find start
    start = None
    for i in range(m):
        for j in range(n):
            if grid[i][j] == 'S':
                start = (i, j)
                break

    queue = deque([(start[0], start[1], 0)])  # x, y, distance
    visited = {start}

    while queue:
        x, y, dist = queue.popleft()

        if grid[x][y] == 'E':
            return dist

        for dx, dy in directions:
            nx, ny = x + dx, y + dy

            if (0 <= nx < m and 0 <= ny < n and
                (nx, ny) not in visited and grid[nx][ny] != '#'):
                visited.add((nx, ny))
                queue.append((nx, ny, dist + 1))

    return -1
```

---

## 6. Deque Applications

### Deque Characteristics

```
Deque (Double-Ended Queue): Insert/delete from both ends

    push_front    push_back
         ↓            ↓
      ┌──┬──┬──┬──┬──┐
      │  │  │  │  │  │
      └──┴──┴──┴──┴──┘
         ↑            ↑
    pop_front     pop_back

All operations O(1)
```

```cpp
// C++ - deque
#include <deque>

deque<int> dq;
dq.push_back(1);    // Insert at back
dq.push_front(2);   // Insert at front
dq.front();         // Front element
dq.back();          // Back element
dq.pop_front();     // Remove from front
dq.pop_back();      // Remove from back
```

```python
# Python
from collections import deque

dq = deque()
dq.append(1)        # Insert at back
dq.appendleft(2)    # Insert at front
dq[0]               # Front element
dq[-1]              # Back element
dq.popleft()        # Remove from front
dq.pop()            # Remove from back
```

### Sliding Window Maximum

```
Problem: Maximum values in sliding windows of size k

Array: [1, 3, -1, -3, 5, 3, 6, 7], k = 3

Window              Maximum
[1  3  -1] -3  5  3  6  7    3
 1 [3  -1  -3] 5  3  6  7    3
 1  3 [-1  -3  5] 3  6  7    5
 1  3  -1 [-3  5  3] 6  7    5
 1  3  -1  -3 [5  3  6] 7    6
 1  3  -1  -3  5 [3  6  7]   7

Result: [3, 3, 5, 5, 6, 7]

Deque approach: Keep only maximum candidates (monotonic decreasing deque)
```

```cpp
// C++
vector<int> maxSlidingWindow(const vector<int>& nums, int k) {
    deque<int> dq;  // Stores indices; values in dq are monotonically decreasing
    vector<int> result;

    for (int i = 0; i < nums.size(); i++) {
        // Evict the front if it has slid outside the current window — we check
        // by index so we don't need a separate "window start" variable
        while (!dq.empty() && dq.front() < i - k + 1) {
            dq.pop_front();
        }

        // Discard from the back any index whose value is smaller than nums[i]:
        // those elements can never be a future window maximum because nums[i]
        // is both larger and will outlast them in the window
        while (!dq.empty() && nums[dq.back()] < nums[i]) {
            dq.pop_back();
        }

        dq.push_back(i);

        // The front of dq is always the index of the window's maximum —
        // delay output until the first full window is formed
        if (i >= k - 1) {
            result.push_back(nums[dq.front()]);
        }
    }

    return result;
}
```

```python
# Python
from collections import deque

def max_sliding_window(nums, k):
    dq = deque()  # Store indices
    result = []

    for i in range(len(nums)):
        # Remove elements outside window
        while dq and dq[0] < i - k + 1:
            dq.popleft()

        # Remove smaller elements
        while dq and nums[dq[-1]] < nums[i]:
            dq.pop()

        dq.append(i)

        # Store result when window is complete
        if i >= k - 1:
            result.append(nums[dq[0]])

    return result
```

---

## 7. Practice Problems

### Problem 1: Min Stack

Implement a stack with push, pop, top operations and getMin in O(1).

<details>
<summary>Hint</summary>

Use two stacks or store minimum value at each point with each element

</details>

<details>
<summary>Solution Code</summary>

```python
class MinStack:
    def __init__(self):
        self.stack = []      # (value, min_at_this_point)

    def push(self, x):
        current_min = min(x, self.stack[-1][1]) if self.stack else x
        self.stack.append((x, current_min))

    def pop(self):
        return self.stack.pop()[0]

    def top(self):
        return self.stack[-1][0]

    def get_min(self):
        return self.stack[-1][1]
```

</details>

### Problem 2: Implement Stack Using Two Queues

Implement a stack using two queues.

<details>
<summary>Hint</summary>

Move all elements during push or during pop

</details>

<details>
<summary>Solution Code</summary>

```python
from collections import deque

class StackUsingQueues:
    def __init__(self):
        self.q1 = deque()
        self.q2 = deque()

    def push(self, x):
        # Put new element in q2, move all from q1 to q2
        self.q2.append(x)
        while self.q1:
            self.q2.append(self.q1.popleft())
        # Swap q1 and q2
        self.q1, self.q2 = self.q2, self.q1

    def pop(self):
        return self.q1.popleft()

    def top(self):
        return self.q1[0]

    def empty(self):
        return len(self.q1) == 0
```

</details>

### Recommended Problems

| Difficulty | Problem | Platform | Type |
|-----------|---------|---------|------|
| ⭐ | [Valid Parentheses](https://leetcode.com/problems/valid-parentheses/) | LeetCode | Parenthesis check |
| ⭐ | [Stack](https://www.acmicpc.net/problem/10828) | Baekjoon | Stack basics |
| ⭐⭐ | [Postfix Notation](https://www.acmicpc.net/problem/1918) | Baekjoon | Postfix conversion |
| ⭐⭐ | [Daily Temperatures](https://leetcode.com/problems/daily-temperatures/) | LeetCode | Monotonic stack |
| ⭐⭐⭐ | [Largest Rectangle in Histogram](https://leetcode.com/problems/largest-rectangle-in-histogram/) | LeetCode | Monotonic stack |
| ⭐⭐⭐ | [Sliding Window Maximum](https://leetcode.com/problems/sliding-window-maximum/) | LeetCode | Deque |

---

## Next Steps

- [04_Hash_Tables.md](./04_Hash_Tables.md) - Hash tables

---

## References

- [Monotonic Stack Patterns](https://leetcode.com/tag/monotonic-stack/)
- [BFS/DFS Tutorial](https://www.geeksforgeeks.org/bfs-vs-dfs-binary-tree/)
- [Deque Applications](https://www.geeksforgeeks.org/deque-set-1-introduction-applications/)
