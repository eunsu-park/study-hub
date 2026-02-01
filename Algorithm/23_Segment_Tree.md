# 세그먼트 트리 (Segment Tree)

## 개요

세그먼트 트리는 구간 쿼리와 점 업데이트를 O(log n)에 처리하는 자료구조입니다. 구간 합, 최솟값, 최댓값 등 다양한 쿼리에 활용됩니다.

---

## 목차

1. [세그먼트 트리 개념](#1-세그먼트-트리-개념)
2. [기본 구현](#2-기본-구현)
3. [구간 합 쿼리](#3-구간-합-쿼리)
4. [구간 최솟값/최댓값](#4-구간-최솟값최댓값)
5. [Lazy Propagation](#5-lazy-propagation)
6. [활용 문제](#6-활용-문제)
7. [연습 문제](#7-연습-문제)

---

## 1. 세그먼트 트리 개념

### 1.1 기본 아이디어

```
배열 [2, 4, 1, 3, 5, 2, 7, 6]의 구간 합 세그먼트 트리

                    [30]              (0-7: 전체 합)
                  /      \
             [10]          [20]       (0-3, 4-7)
            /    \        /    \
         [6]     [4]   [7]     [13]   (0-1, 2-3, 4-5, 6-7)
        /  \    /  \   /  \    /  \
       [2] [4] [1] [3][5] [2] [7] [6] (각 원소)

특징:
- 리프 노드: 원본 배열의 각 원소
- 내부 노드: 자식 노드들의 합 (또는 최소/최대)
- 높이: O(log n)
- 노드 개수: 2n - 1 (최대 4n으로 여유있게 할당)
```

### 1.2 시간 복잡도

```
┌─────────────────┬─────────────┬──────────────┐
│ 연산             │ 시간        │ 설명          │
├─────────────────┼─────────────┼──────────────┤
│ 트리 구성        │ O(n)        │ 1회 전처리    │
│ 점 업데이트      │ O(log n)    │ 단일 값 변경  │
│ 구간 쿼리        │ O(log n)    │ 합/최소/최대  │
│ 구간 업데이트    │ O(log n)    │ Lazy 필요     │
└─────────────────┴─────────────┴──────────────┘
```

### 1.3 인덱스 규칙

```
1-indexed 트리 (권장):
- 루트: tree[1]
- 왼쪽 자식: tree[2*i]
- 오른쪽 자식: tree[2*i + 1]
- 부모: tree[i // 2]

0-indexed 트리:
- 루트: tree[0]
- 왼쪽 자식: tree[2*i + 1]
- 오른쪽 자식: tree[2*i + 2]
- 부모: tree[(i - 1) // 2]
```

---

## 2. 기본 구현

### 2.1 구간 합 세그먼트 트리 (재귀)

```python
class SegmentTree:
    def __init__(self, arr):
        self.n = len(arr)
        self.tree = [0] * (4 * self.n)  # 여유있게 4n
        self._build(arr, 1, 0, self.n - 1)

    def _build(self, arr, node, start, end):
        """트리 구성 - O(n)"""
        if start == end:
            self.tree[node] = arr[start]
        else:
            mid = (start + end) // 2
            self._build(arr, 2 * node, start, mid)
            self._build(arr, 2 * node + 1, mid + 1, end)
            self.tree[node] = self.tree[2 * node] + self.tree[2 * node + 1]

    def update(self, idx, val):
        """점 업데이트 - O(log n)"""
        self._update(1, 0, self.n - 1, idx, val)

    def _update(self, node, start, end, idx, val):
        if start == end:
            self.tree[node] = val
        else:
            mid = (start + end) // 2
            if idx <= mid:
                self._update(2 * node, start, mid, idx, val)
            else:
                self._update(2 * node + 1, mid + 1, end, idx, val)
            self.tree[node] = self.tree[2 * node] + self.tree[2 * node + 1]

    def query(self, left, right):
        """구간 합 쿼리 - O(log n)"""
        return self._query(1, 0, self.n - 1, left, right)

    def _query(self, node, start, end, left, right):
        # 범위 밖
        if right < start or end < left:
            return 0

        # 완전히 포함
        if left <= start and end <= right:
            return self.tree[node]

        # 부분 포함
        mid = (start + end) // 2
        left_sum = self._query(2 * node, start, mid, left, right)
        right_sum = self._query(2 * node + 1, mid + 1, end, left, right)
        return left_sum + right_sum


# 사용 예시
arr = [2, 4, 1, 3, 5, 2, 7, 6]
st = SegmentTree(arr)

print(st.query(0, 7))  # 30 (전체 합)
print(st.query(2, 5))  # 11 (1+3+5+2)

st.update(3, 10)  # arr[3] = 10으로 변경
print(st.query(2, 5))  # 18 (1+10+5+2)
```

### 2.2 반복 구현 (Bottom-up)

```python
class SegmentTreeIterative:
    def __init__(self, arr):
        self.n = len(arr)
        self.tree = [0] * (2 * self.n)

        # 리프 노드 초기화
        for i in range(self.n):
            self.tree[self.n + i] = arr[i]

        # 내부 노드 구성
        for i in range(self.n - 1, 0, -1):
            self.tree[i] = self.tree[2 * i] + self.tree[2 * i + 1]

    def update(self, idx, val):
        """점 업데이트"""
        idx += self.n
        self.tree[idx] = val

        while idx > 1:
            idx //= 2
            self.tree[idx] = self.tree[2 * idx] + self.tree[2 * idx + 1]

    def query(self, left, right):
        """구간 합 [left, right]"""
        left += self.n
        right += self.n
        result = 0

        while left <= right:
            if left % 2 == 1:
                result += self.tree[left]
                left += 1
            if right % 2 == 0:
                result += self.tree[right]
                right -= 1
            left //= 2
            right //= 2

        return result
```

### 2.3 C++ 구현

```cpp
#include <vector>
using namespace std;

class SegmentTree {
private:
    vector<long long> tree;
    int n;

    void build(const vector<int>& arr, int node, int start, int end) {
        if (start == end) {
            tree[node] = arr[start];
        } else {
            int mid = (start + end) / 2;
            build(arr, 2 * node, start, mid);
            build(arr, 2 * node + 1, mid + 1, end);
            tree[node] = tree[2 * node] + tree[2 * node + 1];
        }
    }

    void update(int node, int start, int end, int idx, long long val) {
        if (start == end) {
            tree[node] = val;
        } else {
            int mid = (start + end) / 2;
            if (idx <= mid) {
                update(2 * node, start, mid, idx, val);
            } else {
                update(2 * node + 1, mid + 1, end, idx, val);
            }
            tree[node] = tree[2 * node] + tree[2 * node + 1];
        }
    }

    long long query(int node, int start, int end, int left, int right) {
        if (right < start || end < left) return 0;
        if (left <= start && end <= right) return tree[node];

        int mid = (start + end) / 2;
        return query(2 * node, start, mid, left, right) +
               query(2 * node + 1, mid + 1, end, left, right);
    }

public:
    SegmentTree(const vector<int>& arr) {
        n = arr.size();
        tree.resize(4 * n);
        build(arr, 1, 0, n - 1);
    }

    void update(int idx, long long val) {
        update(1, 0, n - 1, idx, val);
    }

    long long query(int left, int right) {
        return query(1, 0, n - 1, left, right);
    }
};
```

---

## 3. 구간 합 쿼리

### 3.1 쿼리 과정 시각화

```
배열: [2, 4, 1, 3, 5, 2, 7, 6]
쿼리: query(2, 5) = 1 + 3 + 5 + 2 = 11

                    [30]
                  /      \
             [10]          [20]
            /    \        /    \
         [6]     [4]   [7]     [13]
        /  \    /  \   /  \    /  \
       [2] [4] [1] [3][5] [2] [7] [6]
            ↑    ↑   ↑   ↑
           범위: 2 ~ 5

쿼리 분해:
[2-3]: 완전 포함 → tree = 4 (1+3)
[4-5]: 완전 포함 → tree = 7 (5+2)
결과: 4 + 7 = 11
```

### 3.2 차이 업데이트 (Difference Update)

```python
class SegmentTreeDiff:
    """값 변경이 아닌 차이 업데이트"""

    def __init__(self, arr):
        self.n = len(arr)
        self.tree = [0] * (4 * self.n)
        self._build(arr, 1, 0, self.n - 1)

    def _build(self, arr, node, start, end):
        if start == end:
            self.tree[node] = arr[start]
        else:
            mid = (start + end) // 2
            self._build(arr, 2 * node, start, mid)
            self._build(arr, 2 * node + 1, mid + 1, end)
            self.tree[node] = self.tree[2 * node] + self.tree[2 * node + 1]

    def add(self, idx, diff):
        """arr[idx] += diff"""
        self._add(1, 0, self.n - 1, idx, diff)

    def _add(self, node, start, end, idx, diff):
        if start == end:
            self.tree[node] += diff
        else:
            mid = (start + end) // 2
            if idx <= mid:
                self._add(2 * node, start, mid, idx, diff)
            else:
                self._add(2 * node + 1, mid + 1, end, idx, diff)
            self.tree[node] = self.tree[2 * node] + self.tree[2 * node + 1]

    def query(self, left, right):
        return self._query(1, 0, self.n - 1, left, right)

    def _query(self, node, start, end, left, right):
        if right < start or end < left:
            return 0
        if left <= start and end <= right:
            return self.tree[node]
        mid = (start + end) // 2
        return (self._query(2 * node, start, mid, left, right) +
                self._query(2 * node + 1, mid + 1, end, left, right))
```

---

## 4. 구간 최솟값/최댓값

### 4.1 최솟값 세그먼트 트리

```python
class MinSegmentTree:
    def __init__(self, arr):
        self.n = len(arr)
        self.tree = [float('inf')] * (4 * self.n)
        self._build(arr, 1, 0, self.n - 1)

    def _build(self, arr, node, start, end):
        if start == end:
            self.tree[node] = arr[start]
        else:
            mid = (start + end) // 2
            self._build(arr, 2 * node, start, mid)
            self._build(arr, 2 * node + 1, mid + 1, end)
            self.tree[node] = min(self.tree[2 * node], self.tree[2 * node + 1])

    def update(self, idx, val):
        self._update(1, 0, self.n - 1, idx, val)

    def _update(self, node, start, end, idx, val):
        if start == end:
            self.tree[node] = val
        else:
            mid = (start + end) // 2
            if idx <= mid:
                self._update(2 * node, start, mid, idx, val)
            else:
                self._update(2 * node + 1, mid + 1, end, idx, val)
            self.tree[node] = min(self.tree[2 * node], self.tree[2 * node + 1])

    def query(self, left, right):
        """구간 [left, right]의 최솟값"""
        return self._query(1, 0, self.n - 1, left, right)

    def _query(self, node, start, end, left, right):
        if right < start or end < left:
            return float('inf')
        if left <= start and end <= right:
            return self.tree[node]
        mid = (start + end) // 2
        return min(self._query(2 * node, start, mid, left, right),
                   self._query(2 * node + 1, mid + 1, end, left, right))


# 예시
arr = [5, 2, 8, 1, 9, 3, 7, 4]
st = MinSegmentTree(arr)
print(st.query(0, 7))  # 1 (전체 최솟값)
print(st.query(2, 5))  # 1 (8, 1, 9, 3 중 최솟값)
print(st.query(4, 7))  # 3 (9, 3, 7, 4 중 최솟값)
```

### 4.2 최솟값 + 인덱스

```python
class MinIndexSegmentTree:
    """최솟값과 해당 인덱스 반환"""

    def __init__(self, arr):
        self.n = len(arr)
        self.tree = [(float('inf'), -1)] * (4 * self.n)
        self._build(arr, 1, 0, self.n - 1)

    def _build(self, arr, node, start, end):
        if start == end:
            self.tree[node] = (arr[start], start)
        else:
            mid = (start + end) // 2
            self._build(arr, 2 * node, start, mid)
            self._build(arr, 2 * node + 1, mid + 1, end)
            self.tree[node] = min(self.tree[2 * node], self.tree[2 * node + 1])

    def query(self, left, right):
        """(최솟값, 인덱스) 반환"""
        return self._query(1, 0, self.n - 1, left, right)

    def _query(self, node, start, end, left, right):
        if right < start or end < left:
            return (float('inf'), -1)
        if left <= start and end <= right:
            return self.tree[node]
        mid = (start + end) // 2
        return min(self._query(2 * node, start, mid, left, right),
                   self._query(2 * node + 1, mid + 1, end, left, right))
```

---

## 5. Lazy Propagation

### 5.1 필요성

```
문제: 구간 [l, r]의 모든 원소에 v를 더하는 연산

일반 세그먼트 트리: O(n) per update (모든 원소 방문)
Lazy Propagation: O(log n) per update

아이디어:
- 업데이트를 바로 적용하지 않고 "나중에" 처리
- lazy[node]에 미처리된 업데이트 저장
- 필요할 때만 자식에게 전파
```

### 5.2 구간 덧셈 + 구간 합

```python
class LazySegmentTree:
    def __init__(self, arr):
        self.n = len(arr)
        self.tree = [0] * (4 * self.n)
        self.lazy = [0] * (4 * self.n)
        self._build(arr, 1, 0, self.n - 1)

    def _build(self, arr, node, start, end):
        if start == end:
            self.tree[node] = arr[start]
        else:
            mid = (start + end) // 2
            self._build(arr, 2 * node, start, mid)
            self._build(arr, 2 * node + 1, mid + 1, end)
            self.tree[node] = self.tree[2 * node] + self.tree[2 * node + 1]

    def _propagate(self, node, start, end):
        """lazy 값을 자식에게 전파"""
        if self.lazy[node] != 0:
            # 현재 노드에 lazy 적용
            self.tree[node] += self.lazy[node] * (end - start + 1)

            # 자식이 있으면 lazy 전파
            if start != end:
                self.lazy[2 * node] += self.lazy[node]
                self.lazy[2 * node + 1] += self.lazy[node]

            self.lazy[node] = 0

    def update_range(self, left, right, val):
        """구간 [left, right]에 val 더하기"""
        self._update_range(1, 0, self.n - 1, left, right, val)

    def _update_range(self, node, start, end, left, right, val):
        self._propagate(node, start, end)

        # 범위 밖
        if right < start or end < left:
            return

        # 완전히 포함
        if left <= start and end <= right:
            self.lazy[node] += val
            self._propagate(node, start, end)
            return

        # 부분 포함
        mid = (start + end) // 2
        self._update_range(2 * node, start, mid, left, right, val)
        self._update_range(2 * node + 1, mid + 1, end, left, right, val)
        self.tree[node] = self.tree[2 * node] + self.tree[2 * node + 1]

    def query(self, left, right):
        """구간 합 쿼리"""
        return self._query(1, 0, self.n - 1, left, right)

    def _query(self, node, start, end, left, right):
        self._propagate(node, start, end)

        if right < start or end < left:
            return 0

        if left <= start and end <= right:
            return self.tree[node]

        mid = (start + end) // 2
        return (self._query(2 * node, start, mid, left, right) +
                self._query(2 * node + 1, mid + 1, end, left, right))


# 사용 예시
arr = [1, 2, 3, 4, 5]
st = LazySegmentTree(arr)

print(st.query(0, 4))  # 15

st.update_range(1, 3, 10)  # [1, 12, 13, 14, 5]
print(st.query(0, 4))  # 45
print(st.query(1, 3))  # 39
```

### 5.3 구간 덧셈 + 구간 최솟값 (Lazy)

```python
class LazyMinSegmentTree:
    def __init__(self, arr):
        self.n = len(arr)
        self.tree = [float('inf')] * (4 * self.n)
        self.lazy = [0] * (4 * self.n)
        self._build(arr, 1, 0, self.n - 1)

    def _build(self, arr, node, start, end):
        if start == end:
            self.tree[node] = arr[start]
        else:
            mid = (start + end) // 2
            self._build(arr, 2 * node, start, mid)
            self._build(arr, 2 * node + 1, mid + 1, end)
            self.tree[node] = min(self.tree[2 * node], self.tree[2 * node + 1])

    def _propagate(self, node):
        if self.lazy[node] != 0:
            self.tree[node] += self.lazy[node]
            if 2 * node < len(self.lazy):
                self.lazy[2 * node] += self.lazy[node]
                self.lazy[2 * node + 1] += self.lazy[node]
            self.lazy[node] = 0

    def update_range(self, left, right, val):
        self._update_range(1, 0, self.n - 1, left, right, val)

    def _update_range(self, node, start, end, left, right, val):
        self._propagate(node)

        if right < start or end < left:
            return

        if left <= start and end <= right:
            self.lazy[node] += val
            self._propagate(node)
            return

        mid = (start + end) // 2
        self._update_range(2 * node, start, mid, left, right, val)
        self._update_range(2 * node + 1, mid + 1, end, left, right, val)
        self.tree[node] = min(self.tree[2 * node], self.tree[2 * node + 1])

    def query(self, left, right):
        return self._query(1, 0, self.n - 1, left, right)

    def _query(self, node, start, end, left, right):
        self._propagate(node)

        if right < start or end < left:
            return float('inf')

        if left <= start and end <= right:
            return self.tree[node]

        mid = (start + end) // 2
        return min(self._query(2 * node, start, mid, left, right),
                   self._query(2 * node + 1, mid + 1, end, left, right))
```

---

## 6. 활용 문제

### 6.1 역순 쌍 개수 (Inversion Count)

```python
def count_inversions(arr):
    """
    역순 쌍: i < j 이고 arr[i] > arr[j]인 쌍의 개수
    세그먼트 트리 활용
    """
    # 좌표 압축
    sorted_arr = sorted(set(arr))
    rank = {v: i for i, v in enumerate(sorted_arr)}
    n = len(sorted_arr)

    st = SegmentTree([0] * n)
    count = 0

    for val in arr:
        r = rank[val]
        # r보다 큰 값의 개수 (이미 처리된 것 중)
        count += st.query(r + 1, n - 1)
        # 현재 값 추가
        st.add(r, 1)

    return count
```

### 6.2 K번째 원소 찾기

```python
def find_kth(st, k):
    """
    세그먼트 트리에서 k번째로 작은 원소의 인덱스 찾기
    st[i] = 1 if i번째 원소 존재, 0 otherwise
    """
    node = 1
    start, end = 0, st.n - 1

    while start != end:
        mid = (start + end) // 2
        left_count = st.tree[2 * node]

        if k <= left_count:
            node = 2 * node
            end = mid
        else:
            k -= left_count
            node = 2 * node + 1
            start = mid + 1

    return start
```

### 6.3 2D 세그먼트 트리

```python
class SegmentTree2D:
    """2차원 구간 합 세그먼트 트리"""

    def __init__(self, matrix):
        self.n = len(matrix)
        self.m = len(matrix[0]) if self.n > 0 else 0
        self.tree = [[0] * (4 * self.m) for _ in range(4 * self.n)]
        if self.n > 0 and self.m > 0:
            self._build_x(matrix, 1, 0, self.n - 1)

    def _build_x(self, matrix, node_x, start_x, end_x):
        if start_x == end_x:
            self._build_y(matrix, node_x, start_x, end_x, 1, 0, self.m - 1, True)
        else:
            mid_x = (start_x + end_x) // 2
            self._build_x(matrix, 2 * node_x, start_x, mid_x)
            self._build_x(matrix, 2 * node_x + 1, mid_x + 1, end_x)
            self._build_y(matrix, node_x, start_x, end_x, 1, 0, self.m - 1, False)

    def _build_y(self, matrix, node_x, start_x, end_x, node_y, start_y, end_y, leaf_x):
        if start_y == end_y:
            if leaf_x:
                self.tree[node_x][node_y] = matrix[start_x][start_y]
            else:
                self.tree[node_x][node_y] = (self.tree[2 * node_x][node_y] +
                                              self.tree[2 * node_x + 1][node_y])
        else:
            mid_y = (start_y + end_y) // 2
            self._build_y(matrix, node_x, start_x, end_x, 2 * node_y, start_y, mid_y, leaf_x)
            self._build_y(matrix, node_x, start_x, end_x, 2 * node_y + 1, mid_y + 1, end_y, leaf_x)
            self.tree[node_x][node_y] = (self.tree[node_x][2 * node_y] +
                                          self.tree[node_x][2 * node_y + 1])

    def query(self, x1, y1, x2, y2):
        """[(x1,y1), (x2,y2)] 사각형 구간 합"""
        return self._query_x(1, 0, self.n - 1, x1, x2, y1, y2)

    def _query_x(self, node_x, start_x, end_x, x1, x2, y1, y2):
        if x2 < start_x or end_x < x1:
            return 0
        if x1 <= start_x and end_x <= x2:
            return self._query_y(node_x, 1, 0, self.m - 1, y1, y2)

        mid_x = (start_x + end_x) // 2
        return (self._query_x(2 * node_x, start_x, mid_x, x1, x2, y1, y2) +
                self._query_x(2 * node_x + 1, mid_x + 1, end_x, x1, x2, y1, y2))

    def _query_y(self, node_x, node_y, start_y, end_y, y1, y2):
        if y2 < start_y or end_y < y1:
            return 0
        if y1 <= start_y and end_y <= y2:
            return self.tree[node_x][node_y]

        mid_y = (start_y + end_y) // 2
        return (self._query_y(node_x, 2 * node_y, start_y, mid_y, y1, y2) +
                self._query_y(node_x, 2 * node_y + 1, mid_y + 1, end_y, y1, y2))
```

---

## 7. 연습 문제

### 추천 문제

| 난이도 | 문제 | 플랫폼 | 유형 |
|--------|------|--------|------|
| ⭐⭐⭐ | [구간 합 구하기](https://www.acmicpc.net/problem/2042) | 백준 | 기본 |
| ⭐⭐⭐ | [최솟값](https://www.acmicpc.net/problem/10868) | 백준 | 최소 쿼리 |
| ⭐⭐⭐ | [Range Sum Query - Mutable](https://leetcode.com/problems/range-sum-query-mutable/) | LeetCode | 기본 |
| ⭐⭐⭐⭐ | [구간 합 구하기 2](https://www.acmicpc.net/problem/10999) | 백준 | Lazy |
| ⭐⭐⭐⭐ | [수열과 쿼리 17](https://www.acmicpc.net/problem/14438) | 백준 | 최소 쿼리 |
| ⭐⭐⭐⭐⭐ | [Count of Smaller Numbers After Self](https://leetcode.com/problems/count-of-smaller-numbers-after-self/) | LeetCode | 응용 |

---

## 다음 단계

- [24_Fenwick_Tree.md](./24_Fenwick_Tree.md) - 펜윅 트리

---

## 참고 자료

- [Segment Tree](https://cp-algorithms.com/data_structures/segment_tree.html)
- [Lazy Propagation](https://cp-algorithms.com/data_structures/segment_tree.html#lazy-propagation)
