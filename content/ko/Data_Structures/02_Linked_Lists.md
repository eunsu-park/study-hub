# 연결 리스트

**이전**: [배열과 리스트](./01_Arrays_and_Lists.md) | **다음**: [스택](./03_Stacks.md)

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 단일, 이중, 원형 연결 리스트의 구조를 설명할 수 있다
2. Python 클래스를 사용하여 연결 리스트를 처음부터 구현할 수 있다
3. 삽입, 삭제, 순회 연산을 수행할 수 있다
4. 시간/공간 복잡도 측면에서 연결 리스트와 배열을 비교할 수 있다
5. 포인터 조작과 흔한 실수를 이해할 수 있다
6. 러너 (빠른/느린 포인터) 기법을 적용하여 연결 리스트 문제를 해결할 수 있다
7. 연결 리스트가 배열보다 적합한 경우를 인식할 수 있다

---

**연결 리스트**는 각 요소(노드)가 데이터와 다음 노드에 대한 참조(포인터)를 포함하는 선형 자료구조입니다. 배열과 달리 연결 리스트의 요소들은 메모리에 연속적으로 저장되지 않습니다 -- 각 노드는 힙의 어디에나 있을 수 있습니다.

## 단일 연결 리스트

단일 연결 리스트에서 각 노드는 `data`와 `next` 두 필드를 가집니다.

```
  head
   |
   v
+------+------+    +------+------+    +------+------+    +------+------+
| data | next-+--->| data | next-+--->| data | next-+--->| data | next-+--->None
|  10  |      |    |  20  |      |    |  30  |      |    |  40  |      |
+------+------+    +------+------+    +------+------+    +------+------+
```

### 노드 클래스

```python
class Node:
    """단일 연결 리스트의 노드."""
    
    __slots__ = ('data', 'next')  # 메모리 최적화
    
    def __init__(self, data, next_node=None):
        self.data = data
        self.next = next_node
```

### 삽입 연산 시각화

```
앞에 삽입 (prepend) -- O(1):
  이전:  head -> [20] -> [30] -> None
  이후:  head -> [10] -> [20] -> [30] -> None
  단계:  1. new_node.next = head
         2. head = new_node

노드 뒤에 삽입 -- 참조가 있으면 O(1):
  이전:  ... -> [20] -> [30] -> ...
  이후:  ... -> [20] -> [25] -> [30] -> ...
  단계:  1. new_node.next = current.next
         2. current.next = new_node
```

### 삭제 연산 시각화

```
머리 삭제 -- O(1):
  이전:  head -> [10] -> [20] -> [30] -> None
  이후:  head -> [20] -> [30] -> None
  단계:  1. head = head.next

중간 삭제 -- 찾기 O(n), 해제 O(1):
  이전:  ... -> [10] -> [20] -> [30] -> ...
  이후:  ... -> [10] -> [30] -> ...
  단계:  1. 대상 앞 노드 찾기 (prev)
         2. prev.next = prev.next.next
```

## 센티넬 (더미 헤드) 기법

경계 조건을 단순화하는 일반적인 기법으로, 실제 데이터를 포함하지 않는 **센티넬 노드**를 사용합니다:

```python
class SinglyLinkedListSentinel:
    """센티넬 노드가 있는 단일 연결 리스트."""
    
    def __init__(self):
        self._sentinel = Node(None)  # 더미 헤드
        self._size = 0
    
    def prepend(self, data):
        """특별한 경우가 필요 없음 -- 센티넬이 항상 존재."""
        new_node = Node(data, self._sentinel.next)
        self._sentinel.next = new_node
        self._size += 1
```

## 이중 연결 리스트

이중 연결 리스트는 각 노드에 `prev` 포인터를 추가하여 양방향으로 O(1) 연산을 가능하게 합니다:

```
         head                                              tail
          |                                                  |
          v                                                  v
None <--+------+------+  +------+------+------+  +------+------+--> None
        | prev | data |  | prev | data | next |  | prev | data |
        |      |  10  |  |      |  20  |      |  |      |  30  |
        +------+--+---+  +---+--+------+--+---+  +---+--+------+
                  |           ^            |           ^
                  +-----------+            +-----------+
```

```python
class DNode:
    """이중 연결 리스트의 노드."""
    
    __slots__ = ('data', 'prev', 'next')
    
    def __init__(self, data, prev_node=None, next_node=None):
        self.data = data
        self.prev = prev_node
        self.next = next_node
```

## 원형 연결 리스트

원형 연결 리스트에서는 마지막 노드가 다시 첫 번째 노드를 가리켜 고리를 형성합니다:

```
단일 원형:
        +---> [10] ---> [20] ---> [30] ---+
        |                                   |
        +-----------------------------------+
```

## 빠른/느린 포인터 기법

"거북이와 토끼" 기법이라고도 합니다. 하나의 포인터는 한 칸씩, 다른 하나는 두 칸씩 이동합니다.

### 순환 감지

```python
def has_cycle(head):
    """연결 리스트에 순환이 있는지 감지 -- O(n) 시간, O(1) 공간."""
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow is fast:
            return True
    return False
```

### 중간 노드 찾기

```python
def find_middle(head):
    """중간 노드 찾기 -- O(n) 시간, O(1) 공간."""
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    return slow
```

## 연결 리스트 뒤집기

가장 기본적인 연결 리스트 연산 중 하나:

```python
def reverse_list(head):
    """단일 연결 리스트를 제자리에서 뒤집기 -- O(n) 시간, O(1) 공간."""
    prev = None
    current = head
    while current:
        next_node = current.next
        current.next = prev
        prev = current
        current = next_node
    return prev
```

## 비교: 배열 vs 연결 리스트

| 연산 | 배열 (list) | 단일 LL | 이중 LL |
|------|-----------|---------|---------|
| 인덱스로 접근 | **O(1)** | O(n) | O(n) |
| 탐색 | O(n) | O(n) | O(n) |
| 머리에 삽입 | O(n) | **O(1)** | **O(1)** |
| 꼬리에 삽입 | O(1)* | O(n)** | **O(1)** |
| 머리에서 삭제 | O(n) | **O(1)** | **O(1)** |
| 꼬리에서 삭제 | O(1) | O(n) | **O(1)** |
| 주어진 노드 삭제 | O(n) | O(n)*** | **O(1)** |
| 요소당 메모리 | 낮음 | 높음 (+1 ptr) | 높음 (+2 ptrs) |
| 캐시 성능 | **우수** | 나쁨 | 나쁨 |

*분할 상환 | **꼬리 포인터 시 O(1) | ***이전 노드가 있으면 O(1)

## 요약

| 개념 | 핵심 요점 |
|------|----------|
| 단일 연결 리스트 | 각 노드에 data + next 포인터 |
| 이중 연결 리스트 | 각 노드에 data + next + prev 포인터 |
| 원형 연결 리스트 | 마지막 노드가 첫 번째와 연결 |
| 센티넬 노드 | 더미 헤드로 경계 조건 단순화 |
| 빠른/느린 포인터 | 순환 감지, 중간점 찾기 |
| 뒤집기 | 고전적인 O(n) 제자리 기법 |
| 배열 대비 | 삽입에는 유리, 접근에는 불리 |

---

**다음**: [스택](./03_Stacks.md) -- 연결 리스트와 배열 위에 LIFO 자료구조를 구축합니다.
