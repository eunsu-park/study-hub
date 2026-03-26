# 펜윅 트리 (Fenwick Tree / Binary Indexed Tree)

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 최하위 비트(lowbit, `i & -i`) 트릭을 설명하고 각 펜윅 트리(Fenwick tree) 노드가 담당하는 구간이 결정되는 원리를 서술할 수 있다
2. 펜윅 트리에서 O(log n)에 접두사 합(prefix sum) 쿼리와 점 업데이트(point update)를 구현할 수 있다
3. 차분 배열(difference array) 기법을 이용하여 구간 업데이트(range update)와 구간 쿼리(range query)를 지원하도록 펜윅 트리를 확장할 수 있다
4. 펜윅 트리를 적용하여 역전 수(inversion count) 및 순서 통계(order-statistics) 문제를 효율적으로 풀 수 있다
5. 점 업데이트가 있는 2D 구간 합 쿼리를 처리하기 위한 2D 펜윅 트리를 구현할 수 있다
6. 펜윅 트리와 세그먼트 트리를 구현 복잡도, 메모리 사용량, 기능 지원 범위 기준으로 비교할 수 있다

---

## 개요

펜윅 트리(BIT: Binary Indexed Tree)는 구간 합 쿼리와 점 업데이트를 O(log n)에 처리하는 자료구조입니다. 세그먼트 트리보다 구현이 간단하고 메모리 효율적입니다.

---

## 목차

1. [펜윅 트리 개념](#1-펜윅-트리-개념)
2. [기본 구현](#2-기본-구현)
3. [연산](#3-연산)
4. [응용](#4-응용)
5. [2D 펜윅 트리](#5-2d-펜윅-트리)
6. [연습 문제](#6-연습-문제)

---

## 1. 펜윅 트리 개념

### 1.1 구조

```
펜윅 트리: 1-indexed 배열 기반

핵심 아이디어: i번째 원소가 담당하는 구간은
              i의 최하위 비트(lowbit)에 의해 결정

lowbit(i) = i & (-i)

예시 (n=8):
인덱스:    1    2    3    4    5    6    7    8
lowbit:    1    2    1    4    1    2    1    8
담당구간: [1,1][1,2][3,3][1,4][5,5][5,6][7,7][1,8]

tree[1] = arr[1]
tree[2] = arr[1] + arr[2]
tree[3] = arr[3]
tree[4] = arr[1] + arr[2] + arr[3] + arr[4]
tree[5] = arr[5]
tree[6] = arr[5] + arr[6]
tree[7] = arr[7]
tree[8] = arr[1] + ... + arr[8]
```

### 1.2 lowbit 시각화

```
인덱스를 이진수로 표현했을 때:

1  = 0001 → lowbit = 1   → tree[1] = A[1]
2  = 0010 → lowbit = 2   → tree[2] = A[1..2]
3  = 0011 → lowbit = 1   → tree[3] = A[3]
4  = 0100 → lowbit = 4   → tree[4] = A[1..4]
5  = 0101 → lowbit = 1   → tree[5] = A[5]
6  = 0110 → lowbit = 2   → tree[6] = A[5..6]
7  = 0111 → lowbit = 1   → tree[7] = A[7]
8  = 1000 → lowbit = 8   → tree[8] = A[1..8]

패턴:
- 홀수 인덱스: lowbit = 1 (자기 자신만)
- 2의 거듭제곱: lowbit = 인덱스 (1부터 자신까지)
```

### 1.3 시간/공간 복잡도

```
┌─────────────────┬─────────────┬─────────────┐
│ 연산             │ 시간        │ 설명         │
├─────────────────┼─────────────┼─────────────┤
│ 구성             │ O(n)        │ 또는 O(nlogn)│
│ 점 업데이트      │ O(log n)    │ 값 변경      │
│ 프리픽스 합      │ O(log n)    │ [1, i] 합    │
│ 구간 합          │ O(log n)    │ [l, r] 합    │
└─────────────────┴─────────────┴─────────────┘

공간: O(n) - 세그먼트 트리의 1/2~1/4
```

---

## 2. 기본 구현

### 2.1 Python 구현

```python
class FenwickTree:
    def __init__(self, n):
        """크기 n의 펜윅 트리 (1-indexed)"""
        self.n = n
        self.tree = [0] * (n + 1)  # 인덱스 0은 사용하지 않음; 트리는 1-indexed

    def update(self, i, delta):
        """arr[i] += delta — 변경을 모든 조상 노드에 전파"""
        while i <= self.n:
            self.tree[i] += delta
            # i & (-i)는 최하위 설정 비트("lowbit")만 남긴다.
            # lowbit(i)를 더하면 i를 포함하는 구간을 담당하는 다음 노드로
            # 점프한다 — 정확히 O(log n)개의 조상을 방문한다.
            i += i & (-i)

    def prefix_sum(self, i):
        """arr[1] + arr[2] + ... + arr[i] — 접두사를 O(log n)개의 서로소 구간으로 분해"""
        total = 0
        while i > 0:
            total += self.tree[i]
            # lowbit(i)를 빼면 tree[i]의 구간과 서로소이며
            # 왼쪽에 있는 이전 노드로 이동한다 — 이들이 합쳐져 [1, i]를 정확히 타일링한다.
            i -= i & (-i)
        return total

    def range_sum(self, l, r):
        """arr[l] + arr[l+1] + ... + arr[r]
        포함-배제 원리: prefix[r] - prefix[l-1]이 [1, l-1]을 상쇄한다
        """
        return self.prefix_sum(r) - self.prefix_sum(l - 1)


# 사용 예시
n = 8
bit = FenwickTree(n)

# 배열 [0, 1, 2, 3, 4, 5, 6, 7, 8] 구성 (1-indexed)
for i in range(1, n + 1):
    bit.update(i, i)

print(bit.prefix_sum(4))   # 10 (1+2+3+4)
print(bit.range_sum(2, 5)) # 14 (2+3+4+5)

bit.update(3, 5)  # arr[3] += 5
print(bit.range_sum(2, 5)) # 19 (2+8+4+5)
```

### 2.2 배열로 초기화

```python
class FenwickTreeFromArray:
    def __init__(self, arr):
        """배열로 펜윅 트리 구성 - O(n)
        update()를 n번 호출하는 것(O(n log n))보다 빠르다.
        각 노드가 단일 전진 패스에서 자식의 기여를 축적한다.
        """
        self.n = len(arr)
        self.tree = [0] * (self.n + 1)

        # O(n) 초기화: 각 노드의 값을 직계 부모에게 전파한다.
        # 이 상향식 구성으로 모든 조상 노드가 중복 계산 없이
        # 올바른 부분합을 받게 된다.
        for i in range(1, self.n + 1):
            self.tree[i] += arr[i - 1]  # arr은 0-indexed
            j = i + (i & (-i))          # j는 BIT 구조에서 i의 직계 부모
            if j <= self.n:
                self.tree[j] += self.tree[i]  # 현재 노드의 총합을 부모에게 전달

    def update(self, i, delta):
        while i <= self.n:
            self.tree[i] += delta
            i += i & (-i)

    def prefix_sum(self, i):
        total = 0
        while i > 0:
            total += self.tree[i]
            i -= i & (-i)
        return total

    def range_sum(self, l, r):
        return self.prefix_sum(r) - self.prefix_sum(l - 1)


# 사용 예시
arr = [1, 2, 3, 4, 5, 6, 7, 8]
bit = FenwickTreeFromArray(arr)
print(bit.range_sum(1, 4))  # 10
```

### 2.3 C++ 구현

```cpp
#include <vector>
using namespace std;

class FenwickTree {
private:
    vector<long long> tree;
    int n;

public:
    FenwickTree(int n) : n(n), tree(n + 1, 0) {}

    FenwickTree(const vector<int>& arr) : n(arr.size()), tree(arr.size() + 1, 0) {
        for (int i = 1; i <= n; i++) {
            tree[i] += arr[i - 1];
            int j = i + (i & (-i));
            if (j <= n) tree[j] += tree[i];
        }
    }

    void update(int i, long long delta) {
        while (i <= n) {
            tree[i] += delta;
            i += i & (-i);
        }
    }

    long long prefixSum(int i) {
        long long total = 0;
        while (i > 0) {
            total += tree[i];
            i -= i & (-i);
        }
        return total;
    }

    long long rangeSum(int l, int r) {
        return prefixSum(r) - prefixSum(l - 1);
    }
};
```

---

## 3. 연산

### 3.1 업데이트 과정 시각화

```
update(3, 5): arr[3] += 5

인덱스 이동: 3 → 4 → 8 (→ 16 초과)

3  = 0011 → tree[3] += 5
     0011 + 0001 = 0100
4  = 0100 → tree[4] += 5
     0100 + 0100 = 1000
8  = 1000 → tree[8] += 5
     1000 + 1000 = 10000 (> n, 종료)

영향받는 노드: tree[3], tree[4], tree[8]
```

### 3.2 쿼리 과정 시각화

```
prefix_sum(7): arr[1] + ... + arr[7]

인덱스 이동: 7 → 6 → 4 → 0

7  = 0111 → total += tree[7]   (arr[7])
     0111 - 0001 = 0110
6  = 0110 → total += tree[6]   (arr[5..6])
     0110 - 0010 = 0100
4  = 0100 → total += tree[4]   (arr[1..4])
     0100 - 0100 = 0000
0  = 0000 (종료)

결과: tree[7] + tree[6] + tree[4] = arr[1..7]
```

### 3.3 점 쿼리 (Point Query)

```python
class FenwickTreePointQuery:
    """구간 업데이트 + 점 쿼리"""

    def __init__(self, n):
        self.n = n
        self.tree = [0] * (n + 1)

    def range_update(self, l, r, delta):
        """arr[l..r] += delta"""
        self._update(l, delta)
        if r + 1 <= self.n:
            self._update(r + 1, -delta)

    def _update(self, i, delta):
        while i <= self.n:
            self.tree[i] += delta
            i += i & (-i)

    def point_query(self, i):
        """arr[i] 값"""
        total = 0
        while i > 0:
            total += self.tree[i]
            i -= i & (-i)
        return total


# 예시
bit = FenwickTreePointQuery(8)
bit.range_update(2, 5, 3)  # arr[2..5] += 3
print(bit.point_query(3))  # 3
print(bit.point_query(6))  # 0
```

### 3.4 k번째 원소 찾기

```python
def find_kth(bit, k):
    """
    프리픽스 합이 k 이상인 최소 인덱스 찾기
    (bit[i] = 1 if 원소 존재)
    시간: O(log n) — BIT 구조에서의 이진 리프팅(binary lifting)
    prefix_sum() 호출에 이분 탐색을 하는 것(O(log²n))보다 빠르다.
    BIT를 직접 내려가며 한 비트씩 처리하기 때문이다.
    """
    n = bit.n
    pos = 0
    total = 0

    # 높은 비트부터 낮은 비트까지 각 2의 거듭제곱을 더해 pos를 확장 시도한다.
    # 2^i를 더해도 k번째 원소를 넘지 않으면 그 단계를 확정한다.
    # 이 탐욕적 비트별 하강이 O(log n) 단계로 답을 복원한다.
    log_n = n.bit_length()
    for i in range(log_n - 1, -1, -1):
        next_pos = pos + (1 << i)
        if next_pos <= n and total + bit.tree[next_pos] < k:
            total += bit.tree[next_pos]
            pos = next_pos  # 안전하게 확장 — k번째 원소는 더 오른쪽에 있음

    return pos + 1  # pos는 마지막 "안전한" 위치; pos+1이 k번째 원소


# 예시: 동적 k번째 원소
class DynamicKth:
    def __init__(self, max_val):
        self.bit = FenwickTree(max_val)

    def add(self, x):
        """원소 x 추가"""
        self.bit.update(x, 1)

    def remove(self, x):
        """원소 x 제거"""
        self.bit.update(x, -1)

    def kth(self, k):
        """k번째로 작은 원소"""
        return find_kth(self.bit, k)

    def count_less(self, x):
        """x보다 작은 원소 개수"""
        return self.bit.prefix_sum(x - 1)


# 사용
dk = DynamicKth(100)
dk.add(5)
dk.add(10)
dk.add(3)
dk.add(7)
print(dk.kth(2))  # 5 (정렬: 3, 5, 7, 10)
print(dk.count_less(7))  # 2 (3, 5)
```

---

## 4. 응용

### 4.1 역순 쌍 개수 (Inversion Count)

```python
def count_inversions(arr):
    """
    역순 쌍: i < j이고 arr[i] > arr[j]인 (i, j) 개수
    시간: O(n log n)
    """
    # 좌표 압축
    sorted_arr = sorted(set(arr))
    rank = {v: i + 1 for i, v in enumerate(sorted_arr)}
    max_rank = len(sorted_arr)

    bit = FenwickTree(max_rank)
    count = 0

    # 오른쪽에서 왼쪽으로 처리
    for val in reversed(arr):
        r = rank[val]
        # r보다 작은 값의 개수 (이미 처리된 = 오른쪽에 있던)
        count += bit.prefix_sum(r - 1)
        bit.update(r, 1)

    return count


# 예시
arr = [7, 5, 6, 4]
print(count_inversions(arr))  # 5: (7,5), (7,6), (7,4), (5,4), (6,4)
```

### 4.2 구간 업데이트 + 구간 쿼리

```python
class FenwickTreeRURQ:
    """
    Range Update, Range Query — 두 개의 BIT 사용.
    핵심 통찰: prefix_sum(i) = bit1[i] * i - bit2[i].
    이 대수적 항등식을 통해 구간 [l, r]에 delta를 더하는 연산을
    두 BIT에 대한 점 업데이트로 인코딩하여, 업데이트와 쿼리 모두 O(log n)을 달성한다.
    """
    def __init__(self, n):
        self.n = n
        self.bit1 = [0] * (n + 2)
        self.bit2 = [0] * (n + 2)

    def _update(self, bit, i, delta):
        while i <= self.n:
            bit[i] += delta
            i += i & (-i)

    def _query(self, bit, i):
        total = 0
        while i > 0:
            total += bit[i]
            i -= i & (-i)
        return total

    def range_update(self, l, r, delta):
        """arr[l..r] += delta — 두 개의 차분 배열 기법으로 동시에 인코딩"""
        # bit1은 표준 차분 배열 BIT: l에 delta, r+1에 -delta를 더하면
        # query(bit1, i) = i가 [l, r]에 있을 때 delta, 아닐 때 0
        self._update(self.bit1, l, delta)
        self._update(self.bit1, r + 1, -delta)
        # bit2 보정으로 prefix_sum 공식 prefix_sum(i) = bit1*i - bit2가
        # 올바르게 성립하도록 한다 — l-1과 r 항이 [l, r] 외부의 기여를 상쇄
        self._update(self.bit2, l, delta * (l - 1))
        self._update(self.bit2, r + 1, -delta * r)

    def prefix_sum(self, i):
        """arr[1] + ... + arr[i] — 두 BIT 항등식을 사용"""
        return self._query(self.bit1, i) * i - self._query(self.bit2, i)

    def range_sum(self, l, r):
        """arr[l] + ... + arr[r]"""
        return self.prefix_sum(r) - self.prefix_sum(l - 1)


# 예시
bit = FenwickTreeRURQ(8)
bit.range_update(2, 5, 3)  # arr[2..5] += 3
print(bit.range_sum(1, 4))  # 9 (0+3+3+3)
print(bit.range_sum(3, 6))  # 12 (3+3+3+3)
```

### 4.3 오프라인 쿼리 처리

```python
def offline_range_sum(arr, queries):
    """
    쿼리: (l, r, type)
    type 1: arr[l] += r
    type 2: arr[l..r] 합 반환
    """
    n = len(arr)
    bit = FenwickTreeFromArray(arr)
    results = []

    for query in queries:
        if query[0] == 1:
            _, idx, val = query
            bit.update(idx, val)
        else:
            _, l, r = query
            results.append(bit.range_sum(l, r))

    return results
```

---

## 5. 2D 펜윅 트리

### 5.1 구현

```python
class FenwickTree2D:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.tree = [[0] * (cols + 1) for _ in range(rows + 1)]

    def update(self, x, y, delta):
        """arr[x][y] += delta"""
        i = x
        while i <= self.rows:
            j = y
            while j <= self.cols:
                self.tree[i][j] += delta
                j += j & (-j)
            i += i & (-i)

    def prefix_sum(self, x, y):
        """arr[1..x][1..y] 합"""
        total = 0
        i = x
        while i > 0:
            j = y
            while j > 0:
                total += self.tree[i][j]
                j -= j & (-j)
            i -= i & (-i)
        return total

    def range_sum(self, x1, y1, x2, y2):
        """arr[x1..x2][y1..y2] 합"""
        return (self.prefix_sum(x2, y2)
                - self.prefix_sum(x1 - 1, y2)
                - self.prefix_sum(x2, y1 - 1)
                + self.prefix_sum(x1 - 1, y1 - 1))


# 예시
bit2d = FenwickTree2D(4, 4)
matrix = [
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16]
]

# 초기화
for i in range(4):
    for j in range(4):
        bit2d.update(i + 1, j + 1, matrix[i][j])

print(bit2d.range_sum(2, 2, 3, 3))  # 34 (6+7+10+11)
```

### 5.2 C++ 2D 구현

```cpp
class FenwickTree2D {
private:
    vector<vector<long long>> tree;
    int rows, cols;

public:
    FenwickTree2D(int r, int c) : rows(r), cols(c) {
        tree.assign(r + 1, vector<long long>(c + 1, 0));
    }

    void update(int x, int y, long long delta) {
        for (int i = x; i <= rows; i += i & (-i)) {
            for (int j = y; j <= cols; j += j & (-j)) {
                tree[i][j] += delta;
            }
        }
    }

    long long prefixSum(int x, int y) {
        long long total = 0;
        for (int i = x; i > 0; i -= i & (-i)) {
            for (int j = y; j > 0; j -= j & (-j)) {
                total += tree[i][j];
            }
        }
        return total;
    }

    long long rangeSum(int x1, int y1, int x2, int y2) {
        return prefixSum(x2, y2) - prefixSum(x1 - 1, y2)
               - prefixSum(x2, y1 - 1) + prefixSum(x1 - 1, y1 - 1);
    }
};
```

---

## 6. 연습 문제

### 추천 문제

| 난이도 | 문제 | 플랫폼 | 유형 |
|--------|------|--------|------|
| ⭐⭐⭐ | [구간 합 구하기](https://www.acmicpc.net/problem/2042) | 백준 | 기본 |
| ⭐⭐⭐ | [Range Sum Query - Mutable](https://leetcode.com/problems/range-sum-query-mutable/) | LeetCode | 기본 |
| ⭐⭐⭐ | [버블 소트](https://www.acmicpc.net/problem/1517) | 백준 | 역순 쌍 |
| ⭐⭐⭐⭐ | [Count of Smaller Numbers](https://leetcode.com/problems/count-of-smaller-numbers-after-self/) | LeetCode | 역순 쌍 |
| ⭐⭐⭐⭐ | [직사각형 합](https://www.acmicpc.net/problem/11658) | 백준 | 2D BIT |

---

## 세그먼트 트리 vs 펜윅 트리

```
┌────────────────┬──────────────┬──────────────┐
│ 기준           │ 세그먼트 트리 │ 펜윅 트리     │
├────────────────┼──────────────┼──────────────┤
│ 공간           │ O(4n)        │ O(n)         │
│ 구현 복잡도    │ 중간         │ 간단 ✓       │
│ 상수 계수      │ 큼           │ 작음 ✓       │
│ 점 쿼리        │ ✓            │ ✓            │
│ 구간 쿼리      │ ✓            │ ✓            │
│ 구간 업데이트  │ Lazy 필요    │ 2개 BIT 필요 │
│ 범용성         │ 높음 ✓       │ 낮음 (합만)   │
│ 최소/최대      │ ✓            │ ✗            │
└────────────────┴──────────────┴──────────────┘

결론:
- 구간 합만 필요 → 펜윅 트리
- 최소/최대/복잡한 연산 → 세그먼트 트리
```

---

## 다음 단계

- [25_Network_Flow.md](./25_Network_Flow.md) - 네트워크 플로우

---

## 참고 자료

- [Fenwick Tree](https://cp-algorithms.com/data_structures/fenwick.html)
- [Binary Indexed Trees - TopCoder](https://www.topcoder.com/thrive/articles/Binary%20Indexed%20Trees)
