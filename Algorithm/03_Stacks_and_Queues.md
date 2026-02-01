# 스택과 큐 활용 (Stack and Queue Applications)

## 개요

스택(Stack)과 큐(Queue)는 기본적인 자료구조이지만, 다양한 알고리즘 문제에서 핵심적인 역할을 합니다. 이 레슨에서는 괄호 검사, 후위 표기법, 모노토닉 스택 등 실전에서 자주 사용되는 패턴을 학습합니다.

---

## 목차

1. [스택과 큐 기초](#1-스택과-큐-기초)
2. [괄호 유효성 검사](#2-괄호-유효성-검사)
3. [후위 표기법](#3-후위-표기법)
4. [모노토닉 스택](#4-모노토닉-스택)
5. [큐와 BFS](#5-큐와-bfs)
6. [덱(Deque) 활용](#6-덱deque-활용)
7. [연습 문제](#7-연습-문제)

---

## 1. 스택과 큐 기초

### 스택 (Stack) - LIFO

```
LIFO: Last In, First Out (후입선출)

        push(3)     push(7)     pop()
           │           │          │
           ▼           ▼          ▼
        ┌───┐       ┌───┐      ┌───┐
        │   │       │ 7 │      │   │
        ├───┤       ├───┤      ├───┤
        │ 3 │       │ 3 │      │ 3 │
        └───┘       └───┘      └───┘

주요 연산 (모두 O(1)):
- push(x): 원소 삽입
- pop(): 최상단 원소 제거 및 반환
- top()/peek(): 최상단 원소 확인
- empty(): 비어있는지 확인
```

```c
// C - 배열 기반 스택
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
    return -1;  // 에러
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
s.push(3);      // 삽입
s.push(7);
s.top();        // 7 (최상단 확인)
s.pop();        // 제거 (반환값 없음!)
s.empty();      // false
s.size();       // 1
```

```python
# Python - 리스트로 스택 구현
stack = []
stack.append(3)   # push
stack.append(7)
stack[-1]         # top: 7
stack.pop()       # 7 반환 및 제거
len(stack) == 0   # empty 확인
```

### 큐 (Queue) - FIFO

```
FIFO: First In, First Out (선입선출)

enqueue(1)  enqueue(2)  enqueue(3)  dequeue()
    │           │           │           │
    ▼           ▼           ▼           ▼
  ┌───┐      ┌───┬───┐   ┌───┬───┬───┐  ┌───┬───┐
  │ 1 │      │ 1 │ 2 │   │ 1 │ 2 │ 3 │  │ 2 │ 3 │
  └───┘      └───┴───┘   └───┴───┴───┘  └───┴───┘
  front                   front          front
                                back           back

주요 연산 (모두 O(1)):
- enqueue(x)/push(x): 뒤에 삽입
- dequeue()/pop(): 앞에서 제거
- front(): 맨 앞 원소 확인
- empty(): 비어있는지 확인
```

```cpp
// C++ - STL queue
#include <queue>

queue<int> q;
q.push(1);      // enqueue
q.push(2);
q.push(3);
q.front();      // 1 (맨 앞)
q.back();       // 3 (맨 뒤)
q.pop();        // dequeue (반환값 없음!)
q.empty();      // false
```

```python
# Python - collections.deque 사용 (권장)
from collections import deque

q = deque()
q.append(1)     # enqueue
q.append(2)
q.append(3)
q[0]            # front: 1
q[-1]           # back: 3
q.popleft()     # dequeue: 1 반환 및 제거
```

---

## 2. 괄호 유효성 검사

### 문제

주어진 문자열의 괄호가 올바르게 짝지어졌는지 확인합니다.

```
유효: "()", "()[]{}", "{[()]}"
무효: "(]", "([)]", "((("
```

### 알고리즘

```
문자열: "{[()]}"

문자  스택         동작
----  ----        ----
{     [           push '{'
[     [{          push '['
(     [{(         push '('
)     [{          pop '(' ← ')' 와 매칭
]     [           pop '[' ← ']' 와 매칭
}     []          pop '{' ← '}' 와 매칭

스택이 비어있음 → 유효!
```

### 구현

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
            stack[++top] = c;
        } else {
            if (top == -1) return false;

            char topChar = stack[top--];

            if (c == ')' && topChar != '(') return false;
            if (c == ']' && topChar != '[') return false;
            if (c == '}' && topChar != '{') return false;
        }
    }

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

## 3. 후위 표기법

### 표기법 종류

```
중위 표기법 (Infix):   3 + 4 * 2
전위 표기법 (Prefix):  + 3 * 4 2
후위 표기법 (Postfix): 3 4 2 * +

후위 표기법의 장점:
- 괄호가 필요 없음
- 연산자 우선순위 고려 불필요
- 컴퓨터가 계산하기 쉬움
```

### 3.1 후위 표기식 계산

```
수식: "3 4 2 * +"

스텝  토큰  스택        동작
----  ----  ----       ----
1     3     [3]        push 3
2     4     [3,4]      push 4
3     2     [3,4,2]    push 2
4     *     [3,8]      pop 2,4 → push 4*2=8
5     +     [11]       pop 8,3 → push 3+8=11

결과: 11
```

```c
// C - 후위 표기식 계산 (단일 자릿수)
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
                stack.append(int(a / b))  # 정수 나눗셈

    return stack[-1]
```

### 3.2 중위 → 후위 변환 (Shunting-yard 알고리즘)

```
수식: "3 + 4 * 2"

연산자 우선순위: * / > + -

스텝  토큰  출력         스택      동작
----  ----  ----        ----     ----
1     3     3           []        숫자 → 출력
2     +     3           [+]       연산자 → 스택
3     4     3 4         [+]       숫자 → 출력
4     *     3 4         [+,*]     *가 +보다 우선순위 높음 → push
5     2     3 4 2       [+,*]     숫자 → 출력
끝          3 4 2 * +   []        스택 비우기

결과: "3 4 2 * +"
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
            ops.pop();  // '(' 제거
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
            ops.pop()  # '(' 제거
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

## 4. 모노토닉 스택

### 개념

```
모노토닉 스택: 원소가 단조 증가 또는 단조 감소하는 스택
→ O(n)으로 "다음/이전 크거나 작은 원소" 문제 해결

유형:
1. 단조 증가 스택: 작은 원소만 유지 (다음 큰 원소 찾기)
2. 단조 감소 스택: 큰 원소만 유지 (다음 작은 원소 찾기)
```

### 4.1 다음 큰 원소 (Next Greater Element)

```
문제: 각 원소에 대해 오른쪽에서 처음으로 자신보다 큰 원소 찾기

배열: [2, 1, 2, 4, 3]

인덱스  원소  스택(인덱스)  다음큰원소  동작
------  ----  ----------  --------  ----
0       2     [0]         -         push
1       1     [0,1]       -         1<2, push
2       2     [0,2]       2         pop 1 (2>1), push
3       4     [3]         4,4,4     pop all (4>2, 4>2), push
4       3     [3,4]       -         3<4, push
끝            []          -1,-1     남은 것들은 -1

결과: [4, 2, 4, -1, -1]
```

```c
// C
void nextGreaterElement(int arr[], int n, int result[]) {
    int stack[n];
    int top = -1;

    for (int i = 0; i < n; i++) {
        result[i] = -1;  // 기본값
    }

    for (int i = 0; i < n; i++) {
        // 현재 원소가 스택 top보다 크면 pop하면서 결과 저장
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
    stack<int> st;  // 인덱스 저장

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
    stack = []  # 인덱스 저장

    for i in range(n):
        while stack and arr[stack[-1]] < arr[i]:
            result[stack.pop()] = arr[i]
        stack.append(i)

    return result
```

### 4.2 히스토그램에서 가장 큰 직사각형

```
문제: 히스토그램에서 가장 큰 직사각형 넓이

높이: [2, 1, 5, 6, 2, 3]

    ┌───┐
    │   │
┌───┤   │
│   │   │       ┌───┐
│   │   │───┬───┤   │
│   │   │   │   │   │
└───┴───┴───┴───┴───┘
  2   1   5   6   2   3

가장 큰 직사각형: 높이 5, 너비 2 → 넓이 10
```

```cpp
// C++
int largestRectangleArea(const vector<int>& heights) {
    int n = heights.size();
    stack<int> st;
    int maxArea = 0;

    for (int i = 0; i <= n; i++) {
        int h = (i == n) ? 0 : heights[i];

        while (!st.empty() && heights[st.top()] > h) {
            int height = heights[st.top()];
            st.pop();

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

### 4.3 일일 온도 (Daily Temperatures)

```
문제: 각 날짜에 대해 더 따뜻한 날이 오기까지 며칠 기다려야 하는지

온도: [73, 74, 75, 71, 69, 72, 76, 73]
결과: [1,  1,  4,  2,  1,  1,  0,  0]

설명:
- 73 → 1일 후 74
- 74 → 1일 후 75
- 75 → 4일 후 76
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

## 5. 큐와 BFS

### BFS 기본 구조

```
BFS (너비 우선 탐색): 큐를 사용하여 가까운 노드부터 탐색

       1
      /|\
     2 3 4
    /|   |
   5 6   7

BFS 순서: 1 → 2 → 3 → 4 → 5 → 6 → 7
```

```cpp
// C++ - BFS 템플릿
void bfs(vector<vector<int>>& graph, int start) {
    int n = graph.size();
    vector<bool> visited(n, false);
    queue<int> q;

    q.push(start);
    visited[start] = true;

    while (!q.empty()) {
        int node = q.front();
        q.pop();

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

### 최단 거리 BFS (2D 그리드)

```
문제: 미로에서 시작점(S)부터 끝점(E)까지 최단 거리

미로:
S . . #
# . # .
. . . E

BFS로 탐색:
1 2 3 #
# 3 # 5
5 4 5 6

최단 거리: 6
```

```cpp
// C++
int shortestPath(vector<vector<char>>& grid) {
    int m = grid.size(), n = grid[0].size();
    int dx[] = {0, 0, 1, -1};
    int dy[] = {1, -1, 0, 0};

    // 시작점 찾기
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

    return -1;  // 도달 불가
}
```

```python
# Python
from collections import deque

def shortest_path(grid):
    m, n = len(grid), len(grid[0])
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    # 시작점 찾기
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

## 6. 덱(Deque) 활용

### 덱의 특성

```
덱 (Double-Ended Queue): 양쪽에서 삽입/삭제 가능

    push_front    push_back
         ↓            ↓
      ┌──┬──┬──┬──┬──┐
      │  │  │  │  │  │
      └──┴──┴──┴──┴──┘
         ↑            ↑
    pop_front     pop_back

모든 연산 O(1)
```

```cpp
// C++ - deque
#include <deque>

deque<int> dq;
dq.push_back(1);    // 뒤에 삽입
dq.push_front(2);   // 앞에 삽입
dq.front();         // 앞 원소
dq.back();          // 뒤 원소
dq.pop_front();     // 앞에서 제거
dq.pop_back();      // 뒤에서 제거
```

```python
# Python
from collections import deque

dq = deque()
dq.append(1)        # 뒤에 삽입
dq.appendleft(2)    # 앞에 삽입
dq[0]               # 앞 원소
dq[-1]              # 뒤 원소
dq.popleft()        # 앞에서 제거
dq.pop()            # 뒤에서 제거
```

### 슬라이딩 윈도우 최댓값

```
문제: 크기 k인 슬라이딩 윈도우의 최댓값들

배열: [1, 3, -1, -3, 5, 3, 6, 7], k = 3

윈도우              최댓값
[1  3  -1] -3  5  3  6  7    3
 1 [3  -1  -3] 5  3  6  7    3
 1  3 [-1  -3  5] 3  6  7    5
 1  3  -1 [-3  5  3] 6  7    5
 1  3  -1  -3 [5  3  6] 7    6
 1  3  -1  -3  5 [3  6  7]   7

결과: [3, 3, 5, 5, 6, 7]

덱 활용: 최댓값 후보만 유지 (단조 감소 덱)
```

```cpp
// C++
vector<int> maxSlidingWindow(const vector<int>& nums, int k) {
    deque<int> dq;  // 인덱스 저장, 단조 감소
    vector<int> result;

    for (int i = 0; i < nums.size(); i++) {
        // 윈도우 범위 벗어난 원소 제거
        while (!dq.empty() && dq.front() < i - k + 1) {
            dq.pop_front();
        }

        // 현재 원소보다 작은 원소들 제거 (최댓값 후보 유지)
        while (!dq.empty() && nums[dq.back()] < nums[i]) {
            dq.pop_back();
        }

        dq.push_back(i);

        // 윈도우가 완성되면 결과 저장
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
    dq = deque()  # 인덱스 저장
    result = []

    for i in range(len(nums)):
        # 윈도우 범위 벗어난 원소 제거
        while dq and dq[0] < i - k + 1:
            dq.popleft()

        # 현재 원소보다 작은 원소들 제거
        while dq and nums[dq[-1]] < nums[i]:
            dq.pop()

        dq.append(i)

        # 윈도우 완성 시 결과 저장
        if i >= k - 1:
            result.append(nums[dq[0]])

    return result
```

---

## 7. 연습 문제

### 문제 1: 스택 최솟값

push, pop, top과 함께 O(1)에 최솟값을 반환하는 스택을 구현하세요.

<details>
<summary>힌트</summary>

두 개의 스택을 사용하거나, 각 원소와 함께 그 시점의 최솟값을 저장

</details>

<details>
<summary>정답 코드</summary>

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

### 문제 2: 큐 두 개로 스택 구현

두 개의 큐를 사용하여 스택을 구현하세요.

<details>
<summary>힌트</summary>

push할 때 모든 원소를 옮기거나, pop할 때 모든 원소를 옮기는 방법

</details>

<details>
<summary>정답 코드</summary>

```python
from collections import deque

class StackUsingQueues:
    def __init__(self):
        self.q1 = deque()
        self.q2 = deque()

    def push(self, x):
        # 새 원소를 q2에 넣고, q1의 모든 원소를 q2로 이동
        self.q2.append(x)
        while self.q1:
            self.q2.append(self.q1.popleft())
        # q1과 q2 교환
        self.q1, self.q2 = self.q2, self.q1

    def pop(self):
        return self.q1.popleft()

    def top(self):
        return self.q1[0]

    def empty(self):
        return len(self.q1) == 0
```

</details>

### 추천 문제

| 난이도 | 문제 | 플랫폼 | 유형 |
|--------|------|--------|------|
| ⭐ | [Valid Parentheses](https://leetcode.com/problems/valid-parentheses/) | LeetCode | 괄호 검사 |
| ⭐ | [스택](https://www.acmicpc.net/problem/10828) | 백준 | 스택 기초 |
| ⭐⭐ | [후위 표기식](https://www.acmicpc.net/problem/1918) | 백준 | 후위 변환 |
| ⭐⭐ | [Daily Temperatures](https://leetcode.com/problems/daily-temperatures/) | LeetCode | 모노토닉 스택 |
| ⭐⭐⭐ | [Largest Rectangle in Histogram](https://leetcode.com/problems/largest-rectangle-in-histogram/) | LeetCode | 모노토닉 스택 |
| ⭐⭐⭐ | [Sliding Window Maximum](https://leetcode.com/problems/sliding-window-maximum/) | LeetCode | 덱 |

---

## 다음 단계

- [04_Hash_Tables.md](./04_Hash_Tables.md) - 해시 테이블

---

## 참고 자료

- [Monotonic Stack Patterns](https://leetcode.com/tag/monotonic-stack/)
- [BFS/DFS Tutorial](https://www.geeksforgeeks.org/bfs-vs-dfs-binary-tree/)
- [Deque Applications](https://www.geeksforgeeks.org/deque-set-1-introduction-applications/)
