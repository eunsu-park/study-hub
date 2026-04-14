# 스택

**이전**: [연결 리스트](./02_Linked_Lists.md) | **다음**: [큐](./04_Queues.md)

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. LIFO (후입선출) 원리와 그 중요성을 설명할 수 있다
2. 배열과 연결 리스트를 사용하여 스택을 구현할 수 있다
3. push, pop, peek 연산의 시간 복잡도를 분석할 수 있다
4. 스택을 활용하여 수식 계산과 괄호 매칭 문제를 해결할 수 있다
5. 프로그램 실행에서 호출 스택의 작동 방식을 이해할 수 있다
6. Python의 내장 리스트를 스택으로 사용하고 그 한계를 알 수 있다
7. 단조 스택 패턴을 구현하고 적용할 수 있다

---

**스택**은 **후입선출 (LIFO, Last-In, First-Out)** 원리를 따르는 선형 자료구조입니다. 마지막에 추가된 요소가 가장 먼저 제거되며, 접시를 쌓아 올린 것과 같습니다.

## 스택 ADT

```
        +-------+
        |  Top  |  <-- push/pop이 여기서 발생
        +-------+
        |       |
        +-------+
        |       |
        +-------+
        | Bottom|
        +-------+
```

### 핵심 연산

| 연산 | 설명 | 시간 |
|------|------|------|
| `push(item)` | 상단에 항목 추가 | O(1) |
| `pop()` | 상단 항목을 제거하고 반환 | O(1) |
| `peek()` / `top()` | 제거하지 않고 상단 항목 반환 | O(1) |
| `is_empty()` | 스택이 비어있는지 확인 | O(1) |
| `size()` | 요소 수 반환 | O(1) |

## 배열 기반 스택

```python
class ArrayStack:
    """Python 리스트를 사용한 스택 구현."""
    
    def __init__(self):
        self._data = []
    
    def push(self, item):
        """상단에 항목 추가 -- 분할 상환 O(1)."""
        self._data.append(item)
    
    def pop(self):
        """상단 항목을 제거하고 반환 -- O(1)."""
        if self.is_empty():
            raise IndexError("pop from empty stack")
        return self._data.pop()
    
    def peek(self):
        """제거하지 않고 상단 항목 반환 -- O(1)."""
        if self.is_empty():
            raise IndexError("peek at empty stack")
        return self._data[-1]
    
    def is_empty(self):
        return len(self._data) == 0
    
    def __len__(self):
        return len(self._data)
```

## 활용 1: 균형 괄호

가장 고전적인 스택 활용 중 하나:

```python
def is_balanced(expression):
    """괄호/대괄호/중괄호가 균형을 이루는지 확인합니다.
    
    >>> is_balanced("((()))")
    True
    >>> is_balanced("({[]})")
    True
    >>> is_balanced("(()")
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

### 작동 방식

```
입력: "({[]})"

단계 1: '(' -> push    Stack: ['(']
단계 2: '{' -> push    Stack: ['(', '{']
단계 3: '[' -> push    Stack: ['(', '{', '[']
단계 4: ']' -> 매칭    Stack: ['(', '{']        ('[' 가 ']'와 매칭)
단계 5: '}' -> 매칭    Stack: ['(']             ('{' 가 '}'와 매칭)
단계 6: ')' -> 매칭    Stack: []                ('(' 가 ')'와 매칭)
결과: 스택 비어있음 -> 균형!
```

## 활용 2: 수식 계산

### 후위 표기법 계산

```python
def eval_postfix(expression):
    """후위 표기식을 계산합니다.
    
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

## 활용 3: 호출 스택

모든 실행 중인 프로그램은 함수 호출을 관리하기 위해 스택을 사용합니다:

```
def main():           호출 스택:
    a()                +----------+
                       | main()   |
def a():               +----------+
    b()
                       +----------+
def b():               | b()      |  <-- 현재 실행 중
    c()                +----------+
                       | a()      |
def c():               +----------+
    pass               | main()   |
                       +----------+
```

함수가 반환되면 해당 프레임이 스택에서 팝됩니다. 이것이 과도한 재귀가 **스택 오버플로**를 일으키는 이유입니다.

## 활용 4: 실행 취소/다시 실행

```python
class TextEditor:
    """두 개의 스택을 사용한 실행 취소/다시 실행 기능의 간단한 텍스트 편집기."""
    
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

## 단조 스택

단조 스택은 요소를 정렬된 순서 (증가 또는 감소)로 유지합니다. "다음 더 큰 요소" 문제에 유용합니다:

```python
def next_greater_elements(nums):
    """각 요소에 대해 다음으로 큰 요소를 찾습니다.
    
    >>> next_greater_elements([4, 5, 2, 10, 8])
    [5, 10, 10, -1, -1]
    """
    n = len(nums)
    result = [-1] * n
    stack = []
    
    for i in range(n):
        while stack and nums[stack[-1]] < nums[i]:
            result[stack.pop()] = nums[i]
        stack.append(i)
    
    return result
```

## 최소 스택

O(1) 최솟값 조회를 지원하는 스택:

```python
class MinStack:
    """O(1) push, pop, get_min을 지원하는 스택."""
    
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
```

## Python의 내장 스택 옵션

```python
# 옵션 1: list (가장 일반적)
stack = []
stack.append(1)  # push
stack.pop()      # pop

# 옵션 2: collections.deque (스레드 안전, 재할당 없음)
from collections import deque
stack = deque()
stack.append(1)  # push
stack.pop()      # pop

# 옵션 3: queue.LifoQueue (스레드 안전, 블로킹)
from queue import LifoQueue
stack = LifoQueue()
stack.put(1)     # push
stack.get()      # pop (비어있으면 블로킹)
```

## 요약

| 개념 | 핵심 요점 |
|------|----------|
| LIFO | 후입선출 원리 |
| 핵심 연산 | push, pop, peek -- 모두 O(1) |
| 배열 기반 | 간단, 캐시 친화적, 분할 상환 O(1) push |
| 연결리스트 기반 | 용량 제한 없음, 보장된 O(1) push |
| 괄호 매칭 | 고전적인 스택 활용 |
| 수식 계산 | 션팅 야드 + 후위 표기법 계산 |
| 호출 스택 | 프로그램이 함수 호출을 관리하는 방식 |
| 단조 스택 | "다음 더 큰 요소" 패턴 |

---

**다음**: [큐](./04_Queues.md) -- 스택의 FIFO 대응 구조를 탐구합니다.
