# 큐

**이전**: [스택](./03_Stacks.md) | **다음**: [해시 테이블](./05_Hash_Tables.md)

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. FIFO (선입선출) 원리를 설명할 수 있다
2. 배열, 연결 리스트, 원형 버퍼를 사용하여 큐를 구현할 수 있다
3. 고정 용량의 원형 큐를 이해하고 구현할 수 있다
4. Python의 `collections.deque`를 사용하여 효율적인 양방향 연산을 할 수 있다
5. 이중 종단 큐 (덱)를 처음부터 구현할 수 있다
6. BFS, 작업 스케줄링, 생산자-소비자 패턴에 큐를 적용할 수 있다
7. 다양한 큐 구현을 비교하고 트레이드오프를 분석할 수 있다

---

**큐**는 **선입선출 (FIFO, First-In, First-Out)** 원리를 따르는 선형 자료구조입니다. 요소는 **뒤쪽(rear)**에서 추가되고 **앞쪽(front)**에서 제거되며, 줄을 서서 기다리는 사람들과 같습니다.

## 큐 ADT

```
  Dequeue <-- [Front] [  ] [  ] [  ] [Rear] <-- Enqueue

  enqueue(10), enqueue(20), enqueue(30):
  Front                     Rear
    |                         |
    v                         v
  +----+----+----+
  | 10 | 20 | 30 |
  +----+----+----+
```

### 핵심 연산

| 연산 | 설명 | 시간 |
|------|------|------|
| `enqueue(item)` | 뒤쪽에 항목 추가 | O(1) |
| `dequeue()` | 앞쪽 항목을 제거하고 반환 | O(1) |
| `front()` / `peek()` | 제거하지 않고 앞쪽 항목 반환 | O(1) |
| `is_empty()` | 큐가 비어있는지 확인 | O(1) |

## 순진한 배열 기반 큐 (비효율적)

Python 리스트를 순진하게 사용하면 O(n) dequeue가 발생합니다:

```python
class NaiveQueue:
    """리스트를 사용한 큐 -- 이동으로 인해 O(n) dequeue!"""
    
    def __init__(self):
        self._data = []
    
    def enqueue(self, item):
        self._data.append(item)  # O(1) 분할 상환
    
    def dequeue(self):
        if self.is_empty():
            raise IndexError("dequeue from empty queue")
        return self._data.pop(0)  # O(n) -- 모든 요소를 이동!
```

**왜 나쁜가?** `pop(0)`은 모든 요소를 한 위치 왼쪽으로 이동시켜 각 dequeue를 O(n)으로 만듭니다.

## 원형 큐 (링 버퍼)

원형 큐는 고정 크기 배열과 두 포인터 (`front`와 `rear`)를 사용하여 순환합니다:

```
원형 버퍼 (용량=5):

  초기 상태:                enqueue(10, 20, 30) 후:
  +---+---+---+---+---+   +----+----+----+---+---+
  |   |   |   |   |   |   | 10 | 20 | 30 |   |   |
  +---+---+---+---+---+   +----+----+----+---+---+
    ^                        ^              ^
    front, rear              front          rear
```

```python
class CircularQueue:
    """배열을 사용한 고정 용량 원형 큐."""
    
    def __init__(self, capacity):
        self._data = [None] * capacity
        self._capacity = capacity
        self._front = 0
        self._size = 0
    
    def enqueue(self, item):
        """뒤쪽에 항목 추가 -- O(1)."""
        if self._size == self._capacity:
            raise OverflowError("queue is full")
        rear = (self._front + self._size) % self._capacity
        self._data[rear] = item
        self._size += 1
    
    def dequeue(self):
        """앞쪽 항목을 제거하고 반환 -- O(1)."""
        if self.is_empty():
            raise IndexError("dequeue from empty queue")
        item = self._data[self._front]
        self._data[self._front] = None
        self._front = (self._front + 1) % self._capacity
        self._size -= 1
        return item
    
    def is_empty(self):
        return self._size == 0
    
    def is_full(self):
        return self._size == self._capacity
```

## 이중 종단 큐 (덱)

**덱** (deque, 발음: "덱")은 양쪽 끝에서 O(1)으로 삽입과 제거를 허용합니다:

```
        +----+----+----+----+----+
  <---> | 10 | 20 | 30 | 40 | 50 | <--->
        +----+----+----+----+----+
          ^                    ^
        front                rear

  append_left / pop_left      append_right / pop_right
```

## Python의 `collections.deque`

Python은 고정 크기 블록의 이중 연결 리스트로 구현된 최적화된 덱을 제공합니다:

```python
from collections import deque

# 덱 생성
d = deque([1, 2, 3])

# 양쪽 끝에서의 O(1) 연산
d.append(4)        # [1, 2, 3, 4]
d.appendleft(0)    # [0, 1, 2, 3, 4]
d.pop()            # 4 반환, deque는 [0, 1, 2, 3]
d.popleft()        # 0 반환, deque는 [1, 2, 3]

# 회전
d.rotate(1)        # [3, 1, 2]  (오른쪽 회전)
d.rotate(-1)       # [1, 2, 3]  (왼쪽 회전)

# 제한된 덱 (원형 버퍼로 동작)
bounded = deque(maxlen=3)
bounded.extend([1, 2, 3])  # deque([1, 2, 3])
bounded.append(4)           # deque([2, 3, 4]) -- 1 삭제!
```

## 활용: BFS (너비 우선 탐색)

큐는 BFS 순회에 필수적입니다:

```python
from collections import deque

def bfs(graph, start):
    """큐를 사용한 너비 우선 탐색.
    
    graph: 인접 리스트 {노드: [이웃들]}
    """
    visited = {start}
    queue = deque([start])
    order = []
    
    while queue:
        node = queue.popleft()  # O(1)
        order.append(node)
        
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    
    return order
```

## 활용: 슬라이딩 윈도우 최댓값

단조 덱을 사용하여 크기 k인 각 윈도우에서 최댓값을 찾습니다:

```python
from collections import deque

def sliding_window_max(nums, k):
    """크기 k인 각 슬라이딩 윈도우의 최댓값을 찾습니다.
    
    단조 덱 사용 -- 총 O(n) 시간.
    
    >>> sliding_window_max([1, 3, -1, -3, 5, 3, 6, 7], 3)
    [3, 3, 5, 5, 6, 7]
    """
    result = []
    dq = deque()
    
    for i in range(len(nums)):
        while dq and dq[0] < i - k + 1:
            dq.popleft()
        while dq and nums[dq[-1]] < nums[i]:
            dq.pop()
        dq.append(i)
        if i >= k - 1:
            result.append(nums[dq[0]])
    
    return result
```

## 큐 구현 비교

| 구현 | Enqueue | Dequeue | 공간 | 비고 |
|------|---------|---------|------|------|
| List (순진) | O(1)* | **O(n)** | 동적 | 큐에 사용 금지 |
| 원형 배열 | O(1) | O(1) | 고정 | 제한된 큐에 최적 |
| 연결 리스트 | O(1) | O(1) | 동적 | 추가 포인터 오버헤드 |
| `collections.deque` | O(1) | O(1) | 동적 | 가장 좋은 범용 선택 |

*분할 상환

## 요약

| 개념 | 핵심 요점 |
|------|----------|
| FIFO | 선입선출 원리 |
| 핵심 연산 | enqueue, dequeue, peek -- 모두 O(1) |
| 원형 큐 | 인덱스 순환으로 이동 방지 |
| 덱 | 양방향, 양쪽 끝에서 O(1) |
| `collections.deque` | Python에서 큐 구현의 표준 선택 |
| BFS | 큐가 레벨별 그래프 순회를 가능하게 함 |
| 슬라이딩 윈도우 | O(n) 윈도우 쿼리를 위한 단조 덱 |

---

**다음**: [해시 테이블](./05_Hash_Tables.md) -- 상수 시간 키-값 조회를 탐구합니다.
