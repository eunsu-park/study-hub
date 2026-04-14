# 힙

**이전**: [이진 탐색 트리](./07_Binary_Search_Trees.md) | **다음**: [그래프 기초](./09_Graphs_Basics.md)

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 힙 속성을 정의할 수 있다 (최소 힙과 최대 힙)
2. 배열을 사용하여 이진 힙을 구현할 수 있다
3. 힙 올리기 (sift-up)와 힙 내리기 (sift-down) 연산을 수행할 수 있다
4. 정렬되지 않은 배열에서 O(n) 시간에 힙을 구축할 수 있다
5. 힙을 사용하여 우선순위 큐를 구현할 수 있다
6. Python의 `heapq` 모듈을 힙 연산에 사용할 수 있다
7. 힙을 적용하여 상위-K 및 K개 정렬 리스트 병합 문제를 해결할 수 있다

---

**힙**은 **힙 속성**을 만족하는 완전 이진 트리입니다. **우선순위 큐** 추상 데이터 타입의 기반이며, 힙 정렬, 그래프 알고리즘, 스케줄링에 사용됩니다.

## 힙 속성

```
최소 힙: 부모 <= 자식           최대 힙: 부모 >= 자식

       [1]                               [9]
      /   \                             /   \
    [3]   [2]                         [7]   [8]
   / \   /                           / \   /
 [7] [4][5]                        [3] [6][5]

루트가 항상 최솟값!               루트가 항상 최댓값!
```

## 배열 표현

완전 이진 트리는 배열에 완벽하게 매핑됩니다 (낭비 공간 없음):

```
          [1]                    인덱스:  0  1  2  3  4  5
         /   \                   배열:   [1, 3, 2, 7, 4, 5]
       [3]   [2]
      / \   /
    [7] [4][5]

i의 부모:        (i - 1) // 2
i의 왼쪽 자식:    2 * i + 1
i의 오른쪽 자식:  2 * i + 2
```

## 최소 힙 구현

```python
class MinHeap:
    """배열을 사용한 이진 최소 힙."""
    
    def __init__(self):
        self._data = []
    
    def peek(self):
        """최솟값 반환 -- O(1)."""
        if not self._data:
            raise IndexError("peek at empty heap")
        return self._data[0]
    
    def push(self, val):
        """값 삽입 -- O(log n)."""
        self._data.append(val)
        self._sift_up(len(self._data) - 1)
    
    def pop(self):
        """최솟값을 제거하고 반환 -- O(log n)."""
        if not self._data:
            raise IndexError("pop from empty heap")
        self._swap(0, len(self._data) - 1)
        min_val = self._data.pop()
        if self._data:
            self._sift_down(0)
        return min_val
    
    def _sift_up(self, idx):
        """위쪽으로 힙 속성 복원."""
        parent = (idx - 1) // 2
        while idx > 0 and self._data[idx] < self._data[parent]:
            self._swap(idx, parent)
            idx = parent
            parent = (idx - 1) // 2
    
    def _sift_down(self, idx):
        """아래쪽으로 힙 속성 복원."""
        size = len(self._data)
        while True:
            smallest = idx
            left = 2 * idx + 1
            right = 2 * idx + 2
            if left < size and self._data[left] < self._data[smallest]:
                smallest = left
            if right < size and self._data[right] < self._data[smallest]:
                smallest = right
            if smallest == idx:
                break
            self._swap(idx, smallest)
            idx = smallest
    
    def _swap(self, i, j):
        self._data[i], self._data[j] = self._data[j], self._data[i]
```

## 힙 구축: O(n)

```python
def heapify(arr):
    """배열을 제자리에서 최소 힙으로 변환 -- O(n)."""
    n = len(arr)
    for i in range(n // 2 - 1, -1, -1):
        sift_down(arr, i, n)
```

**왜 O(n)인가?** 대부분의 노드가 바닥 근처에 있어 짧은 거리만 내려갑니다. 모든 노드의 높이 합이 O(n)입니다.

## Python의 `heapq` 모듈

```python
import heapq

data = [5, 3, 8, 1, 2]
heapq.heapify(data)  # 제자리, O(n)

heapq.heappush(data, 0)
smallest = heapq.heappop(data)

# N개의 최솟값/최댓값
heapq.nsmallest(3, data)
heapq.nlargest(3, data)

# 최대 힙 트릭: 값을 부정
max_heap = []
heapq.heappush(max_heap, -5)
largest = -heapq.heappop(max_heap)  # 5
```

## 활용: 상위 K 요소

```python
import heapq

def top_k_frequent(nums, k):
    """가장 빈번한 k개의 요소를 찾습니다."""
    from collections import Counter
    counts = Counter(nums)
    return heapq.nlargest(k, counts.keys(), key=counts.get)
```

## 활용: K개의 정렬된 리스트 병합

```python
import heapq

def merge_k_sorted(lists):
    """k개의 정렬된 리스트를 하나의 정렬된 리스트로 병합 -- O(n log k)."""
    result = []
    heap = []
    for i, lst in enumerate(lists):
        if lst:
            heapq.heappush(heap, (lst[0], i, 0))
    while heap:
        val, list_idx, elem_idx = heapq.heappop(heap)
        result.append(val)
        if elem_idx + 1 < len(lists[list_idx]):
            next_val = lists[list_idx][elem_idx + 1]
            heapq.heappush(heap, (next_val, list_idx, elem_idx + 1))
    return result
```

## 활용: 실시간 중앙값

```python
import heapq

class MedianFinder:
    """숫자 스트림의 중앙값을 찾습니다.
    
    두 개의 힙 사용: 하위 절반을 위한 최대 힙, 상위 절반을 위한 최소 힙.
    """
    
    def __init__(self):
        self.small = []  # 최대 힙 (부정 값) -- 하위 절반
        self.large = []  # 최소 힙 -- 상위 절반
    
    def add_num(self, num):
        heapq.heappush(self.small, -num)
        heapq.heappush(self.large, -heapq.heappop(self.small))
        if len(self.large) > len(self.small):
            heapq.heappush(self.small, -heapq.heappop(self.large))
    
    def find_median(self):
        if len(self.small) > len(self.large):
            return -self.small[0]
        return (-self.small[0] + self.large[0]) / 2
```

## 시간 복잡도 요약

| 연산 | 시간 복잡도 |
|------|-----------|
| `peek` (최솟값/최댓값 찾기) | O(1) |
| `push` (삽입) | O(log n) |
| `pop` (최솟값/최댓값 추출) | O(log n) |
| `heapify` (힙 구축) | O(n) |
| 힙 정렬 | O(n log n) |

## 요약

| 개념 | 핵심 요점 |
|------|----------|
| 힙 속성 | 부모 <= 자식 (최소) 또는 >= (최대) |
| 배열 표현 | 완전 이진 트리가 배열에 완벽하게 매핑 |
| 힙 올리기 | 삽입 후 힙 복원 |
| 힙 내리기 | 추출 후 힙 복원 |
| 힙 구축 | 상향식 힙화는 O(n) |
| 우선순위 큐 | 힙 기반, O(log n) enqueue/dequeue |
| `heapq` | Python의 내장 최소 힙 모듈 |
| 최대 힙 트릭 | `heapq`에서 값을 부정 |

---

**다음**: [그래프 기초](./09_Graphs_Basics.md) -- 트리를 일반화하여 네트워크와 관계를 모델링합니다.
