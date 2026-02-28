# 영속 세그먼트 트리(Persistent Segment Tree)

**이전**: [링크-컷 트리(Link-Cut Tree)](./31_Link_Cut_Tree.md)

---

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 자료구조에서 영속성(persistence)의 개념을 설명하고, 부분 영속성(partial persistence)과 완전 영속성(full persistence)을 구분한다
2. 경로 복사(path copying)를 사용하여 영속 세그먼트 트리를 구현하고, O(log N) 시간 및 공간으로 새로운 버전을 생성한다
3. 영속 접두사 세그먼트 트리(persistent prefix segment tree)를 활용하여 "범위 내 K번째 최솟값" 문제를 해결한다
4. 버전 관리 배열 문제와 오프라인 쿼리 처리에 영속 세그먼트 트리를 적용한다
5. 영속 자료구조의 공간 복잡도를 분석하고 메모리 최적화 기법을 적용한다

---

## 목차

1. [영속성이란?](#1-영속성이란)
2. [경로 복사 기법](#2-경로-복사-기법)
3. [영속 세그먼트 트리: 구축과 업데이트](#3-영속-세그먼트-트리-구축과-업데이트)
4. [범위 내 K번째 최솟값](#4-범위-내-k번째-최솟값)
5. [버전 쿼리](#5-버전-쿼리)
6. [구현](#6-구현)
7. [고급 응용](#7-고급-응용)
8. [연습문제](#8-연습문제)

---

## 1. 영속성이란?

**영속 자료구조(persistent data structure)**는 수정이 발생해도 이전의 모든 버전을 보존합니다. 데이터를 덮어쓰는 대신 새로운 버전을 생성하면서 이전 버전에 계속 접근할 수 있습니다.

| 유형 | 이전 접근 | 이전 수정 |
|------|-----------|------------|
| 임시(Ephemeral, 일반) | 불가 | 불가 |
| 부분 영속(Partially persistent) | 가능 (읽기) | 불가 |
| 완전 영속(Fully persistent) | 가능 (읽기) | 가능 (브랜치) |

**핵심 통찰**: 업데이트가 O(log N)개의 노드만 건드린다면(세그먼트 트리처럼), 해당 O(log N)개의 노드만 복사하면 됩니다 — 나머지는 이전 버전과 공유됩니다.

### 동기: 그냥 복사하면 안 될까?

매 버전마다 전체 세그먼트 트리를 복사하면 업데이트당 O(N)의 비용이 발생합니다. Q번의 업데이트로 전체 공간은 O(NQ)가 됩니다. 경로 복사를 사용하면 각 업데이트가 O(log N)만 소요되어 전체 공간이 O(N + Q log N)이 됩니다.

---

## 2. 경로 복사 기법

세그먼트 트리에서 단일 리프를 업데이트할 때, 루트에서 해당 리프까지의 경로만 변경됩니다(O(log N)개 노드). 해당 노드만 새로 복사합니다:

```
Version 0:          Version 1 (update leaf 3):

      [1,4]              [1,4]'
      /    \             /    \
   [1,2]  [3,4]     [1,2]  [3,4]'    ← shared [1,2]
   /  \   /  \      /  \   /  \
  1   2  3   4     1   2  3'  4      ← shared 1, 2, 4
                              ↑
                            new value
```

`'`가 붙은 노드는 새로 생성된 것입니다. `'`가 없는 노드는 버전 0과 공유됩니다.

**업데이트당 공간**: O(log N)개의 새 노드.
**업데이트당 시간**: O(log N) (일반 세그먼트 트리와 동일).

---

## 3. 영속 세그먼트 트리: 구축과 업데이트

### 노드 구조

```python
class Node:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
```

배열 기반 세그먼트 트리와 달리, 영속 세그먼트 트리는 **포인터 기반(pointer-based)** 표현을 사용합니다(노드가 버전 간에 공유되기 때문입니다).

### 구축 (버전 0)

```python
def build(arr, lo, hi):
    if lo == hi:
        return Node(val=arr[lo])
    mid = (lo + hi) // 2
    left = build(arr, lo, mid)
    right = build(arr, mid + 1, hi)
    return Node(val=left.val + right.val, left=left, right=right)
```

### 업데이트 (새 버전 생성)

```python
def update(node, lo, hi, idx, val):
    if lo == hi:
        return Node(val=val)
    mid = (lo + hi) // 2
    if idx <= mid:
        new_left = update(node.left, lo, mid, idx, val)
        return Node(val=new_left.val + node.right.val,
                    left=new_left, right=node.right)  # share right
    else:
        new_right = update(node.right, mid + 1, hi, idx, val)
        return Node(val=node.left.val + new_right.val,
                    left=node.left, right=new_right)  # share left
```

### 쿼리 (일반과 동일)

```python
def query(node, lo, hi, ql, qr):
    if node is None or qr < lo or hi < ql:
        return 0
    if ql <= lo and hi <= qr:
        return node.val
    mid = (lo + hi) // 2
    return (query(node.left, lo, mid, ql, qr) +
            query(node.right, mid + 1, hi, ql, qr))
```

### 버전 관리

```python
roots = []  # roots[i] = root of version i

# Build initial version
roots.append(build(arr, 0, N - 1))

# Create new version by updating index 3 to value 10
roots.append(update(roots[-1], 0, N - 1, 3, 10))

# Query version 0 (original) or version 1 (updated)
query(roots[0], 0, N - 1, 0, 4)  # uses old values
query(roots[1], 0, N - 1, 0, 4)  # uses new values
```

---

## 4. 범위 내 K번째 최솟값

영속 세그먼트 트리의 고전적 응용: 부분 배열 `arr[l..r]`에서 K번째 최솟값을 구합니다.

### 접근법: 영속 접두사 세그먼트 트리(Persistent Prefix Segment Trees)

1. **좌표 압축(coordinate compression)**: 값들을 [0, M-1] 범위의 순위(rank)로 매핑
2. **영속 접두사 트리 구축**: `roots[i]`는 `arr[0..i-1]`에서의 순위 빈도를 세는 세그먼트 트리를 나타냄
3. **범위 쿼리**: `roots[r+1] - roots[l]`의 "차이"가 `arr[l..r]`에서의 빈도를 제공
4. **트리 탐색**: 누적 카운트로 이진 탐색하여 K번째 원소를 찾음

```python
def kth_smallest(root_l, root_r, lo, hi, k):
    """Find k-th smallest in the range represented by root_r - root_l."""
    if lo == hi:
        return lo  # this rank is the answer
    mid = (lo + hi) // 2
    left_count = root_r.left.val - root_l.left.val
    if k <= left_count:
        return kth_smallest(root_l.left, root_r.left, lo, mid, k)
    else:
        return kth_smallest(root_l.right, root_r.right,
                            mid + 1, hi, k - left_count)
```

### 완전한 알고리즘

```python
def solve_kth_range(arr, queries):
    N = len(arr)

    # Step 1: coordinate compression
    sorted_vals = sorted(set(arr))
    rank = {v: i for i, v in enumerate(sorted_vals)}
    M = len(sorted_vals)

    # Step 2: build persistent prefix trees
    roots = [build_empty(0, M - 1)]  # roots[0] = empty tree
    for i in range(N):
        # Insert rank of arr[i] into the next version
        roots.append(insert(roots[-1], 0, M - 1, rank[arr[i]]))

    # Step 3: answer queries
    results = []
    for l, r, k in queries:
        rank_idx = kth_smallest(roots[l], roots[r + 1], 0, M - 1, k)
        results.append(sorted_vals[rank_idx])

    return results
```

### 복잡도

| 연산 | 시간 | 공간 |
|-----------|------|-------|
| 모든 버전 구축 | O(N log M) | O(N log M) |
| 쿼리당 | O(log M) | — |
| 전체 | O((N + Q) log M) | O(N log M) |

---

## 5. 버전 쿼리

영속 세그먼트 트리는 버전 관리 연산을 자연스럽게 지원합니다:

```python
# Version history:
# v0: initial array [1, 2, 3, 4, 5]
# v1: set index 2 to 10 → [1, 2, 10, 4, 5]
# v2: set index 0 to 7 → [7, 2, 10, 4, 5]
# v3: based on v1, set index 4 to 9 → [1, 2, 10, 4, 9] (branching!)

roots = [build([1, 2, 3, 4, 5], 0, 4)]
roots.append(update(roots[0], 0, 4, 2, 10))  # v1
roots.append(update(roots[1], 0, 4, 0, 7))   # v2
roots.append(update(roots[1], 0, 4, 4, 9))   # v3 (branches from v1!)

# All four versions are accessible simultaneously
for i, root in enumerate(roots):
    print(f"v{i}: sum = {query(root, 0, 4, 0, 4)}")
```

이것이 **완전 영속성(full persistence)**입니다 — 어떤 버전에서도 브랜치를 만들 수 있습니다.

---

## 6. 구현

### 완전한 Python 구현

```python
class PersistentSegTree:
    """Persistent Segment Tree with point update and range query."""

    class Node:
        __slots__ = ('val', 'left', 'right')
        def __init__(self, val=0, left=None, right=None):
            self.val = val
            self.left = left
            self.right = right

    def __init__(self, n):
        self.n = n
        self.EMPTY = self.Node()  # sentinel

    def _build(self, arr, lo, hi):
        if lo == hi:
            return self.Node(val=arr[lo] if lo < len(arr) else 0)
        mid = (lo + hi) // 2
        left = self._build(arr, lo, mid)
        right = self._build(arr, mid + 1, hi)
        return self.Node(val=left.val + right.val, left=left, right=right)

    def build(self, arr):
        """Build initial version from array. Returns root."""
        return self._build(arr, 0, self.n - 1)

    def build_empty(self):
        """Build an empty tree (all zeros). Returns root."""
        return self._build_empty(0, self.n - 1)

    def _build_empty(self, lo, hi):
        if lo == hi:
            return self.Node(val=0)
        mid = (lo + hi) // 2
        left = self._build_empty(lo, mid)
        right = self._build_empty(mid + 1, hi)
        return self.Node(val=0, left=left, right=right)

    def update(self, root, idx, val):
        """Create a new version with arr[idx] = val. Returns new root."""
        return self._update(root, 0, self.n - 1, idx, val)

    def _update(self, node, lo, hi, idx, val):
        if lo == hi:
            return self.Node(val=val)
        mid = (lo + hi) // 2
        if idx <= mid:
            new_left = self._update(
                node.left or self.EMPTY, lo, mid, idx, val)
            right = node.right or self.EMPTY
            return self.Node(val=new_left.val + right.val,
                             left=new_left, right=right)
        else:
            left = node.left or self.EMPTY
            new_right = self._update(
                node.right or self.EMPTY, mid + 1, hi, idx, val)
            return self.Node(val=left.val + new_right.val,
                             left=left, right=new_right)

    def insert(self, root, idx):
        """Increment count at idx (for frequency trees). Returns new root."""
        return self._insert(root, 0, self.n - 1, idx)

    def _insert(self, node, lo, hi, idx):
        if lo == hi:
            return self.Node(val=(node.val if node else 0) + 1)
        mid = (lo + hi) // 2
        left = node.left if node else None
        right = node.right if node else None
        if idx <= mid:
            new_left = self._insert(left, lo, mid, idx)
            r_val = right.val if right else 0
            return self.Node(val=new_left.val + r_val,
                             left=new_left, right=right)
        else:
            new_right = self._insert(right, mid + 1, hi, idx)
            l_val = left.val if left else 0
            return self.Node(val=l_val + new_right.val,
                             left=left, right=new_right)

    def query(self, root, ql, qr):
        """Range sum query on [ql, qr]."""
        return self._query(root, 0, self.n - 1, ql, qr)

    def _query(self, node, lo, hi, ql, qr):
        if node is None or qr < lo or hi < ql:
            return 0
        if ql <= lo and hi <= qr:
            return node.val
        mid = (lo + hi) // 2
        return (self._query(node.left, lo, mid, ql, qr) +
                self._query(node.right, mid + 1, hi, ql, qr))

    def kth(self, root_l, root_r, k):
        """Find k-th smallest in range [l, r] using prefix trees."""
        return self._kth(root_l, root_r, 0, self.n - 1, k)

    def _kth(self, nl, nr, lo, hi, k):
        if lo == hi:
            return lo
        mid = (lo + hi) // 2
        left_count = ((nr.left.val if nr.left else 0) -
                      (nl.left.val if nl.left else 0))
        if k <= left_count:
            return self._kth(nl.left or self.EMPTY, nr.left or self.EMPTY,
                             lo, mid, k)
        else:
            return self._kth(nl.right or self.EMPTY, nr.right or self.EMPTY,
                             mid + 1, hi, k - left_count)


# Demo: K-th smallest in range
def demo_kth_range():
    arr = [5, 1, 3, 2, 4, 7, 6, 8]
    N = len(arr)

    # Coordinate compression
    sorted_vals = sorted(set(arr))
    rank = {v: i for i, v in enumerate(sorted_vals)}
    M = len(sorted_vals)

    pst = PersistentSegTree(M)

    # Build prefix trees
    roots = [pst.build_empty()]
    for val in arr:
        roots.append(pst.insert(roots[-1], rank[val]))

    # Query: 2nd smallest in arr[1..5] = [1,3,2,4,7]
    # Sorted: [1,2,3,4,7], 2nd = 2
    l, r, k = 1, 5, 2
    rank_idx = pst.kth(roots[l], roots[r + 1], k)
    print(f"  {k}-th smallest in arr[{l}..{r}] = {sorted_vals[rank_idx]}")

    # Query: 4th smallest in arr[0..7] = [5,1,3,2,4,7,6,8]
    # Sorted: [1,2,3,4,5,6,7,8], 4th = 4
    l, r, k = 0, 7, 4
    rank_idx = pst.kth(roots[l], roots[r + 1], k)
    print(f"  {k}-th smallest in arr[{l}..{r}] = {sorted_vals[rank_idx]}")
```

### 복잡도 요약

| 연산 | 시간 | 공간 |
|-----------|------|-------|
| 초기 버전 구축 | O(N) | O(N) |
| 업데이트 (새 버전) | O(log N) | O(log N) 새 노드 |
| 임의 버전 쿼리 | O(log N) | — |
| K개 버전 전체 | — | O(N + K log N) |
| K번째 최솟값 전처리 | O(N log M) | O(N log M) |
| K번째 최솟값 쿼리 | O(log M) | — |

---

## 7. 고급 응용

### 영속 배열(Persistent Array)

영속 세그먼트 트리를 영속 배열로 사용:
- `get(version, index)`: 단일 지점 쿼리
- `set(version, index, value)`: 업데이트 시 새 버전 반환

임의 개수의 브랜치와 롤백(rollback)을 지원합니다.

### 범위 내 고유 원소 수

영속 세그먼트 트리와 오프라인 스윕라인(sweepline)을 결합:
1. 배열을 왼쪽에서 오른쪽으로 처리
2. 위치 i에서 값 v를 만날 때, v가 마지막으로 위치 j에서 나타났다면 j를 제거하고 i를 추가
3. `distinct[l..r] = query(roots[r+1], l, N-1)` (영속 트리에서 쿼리)

### 2차원 범위 쿼리 (오프라인)

영속 세그먼트 트리로 2차원 쿼리를 시뮬레이션할 수 있습니다:
- 한 차원 기준으로 이벤트를 정렬
- 다른 차원을 따라 영속 트리를 구축
- 버전에 대한 이진 탐색으로 범위 쿼리 처리

---

## 8. 연습문제(Exercises)

### 연습문제 1: 영속 배열

다음을 지원하는 영속 배열을 구현하세요:
1. `create(arr)`: 초기 버전
2. `get(version, idx)`: 원소 읽기
3. `set(version, idx, val)`: 수정된 원소로 새 버전 생성
4. 버전 분기 시연: 버전 0을 두 번 수정하여 독립적인 두 개의 브랜치 생성

### 연습문제 2: 범위 내 K번째 최솟값

완전한 K번째 최솟값 알고리즘을 구현하세요:
1. 좌표 압축
2. 영속 접두사 트리 구축
3. "arr[l..r]에서 K번째 최솟값"형태의 Q개 쿼리 처리
4. 크기 10⁵까지의 배열에서 테스트

### 연습문제 3: 버전 관리 시뮬레이터

간단한 버전 관리 시스템을 구축하세요:
1. `commit(changes)`: 변경 사항으로 새 버전 생성
2. `checkout(version)`: 과거 임의 버전 읽기
3. `branch(version)`: 과거 버전에서 새 브랜치 생성
4. `diff(v1, v2)`: 두 버전 사이의 다른 위치 찾기

### 연습문제 4: 범위 빈도 쿼리

배열이 주어질 때, "arr[l..r]에서 값 x가 몇 번 등장하는가?"라는 쿼리에 답하세요:
1. 좌표 압축 사용
2. 영속 접두사 트리 구축
3. `roots[r+1] - roots[l]`에서 x의 순위에 해당하는 카운트 쿼리

### 연습문제 5: 영속 유니온-파인드(Persistent Union-Find)

부분 영속 유니온-파인드(Union-Find)를 구현하세요:
1. 각 합집합(union) 연산 후 버전 저장
2. 과거 임의 버전에서 연결성 쿼리
3. 영속 세그먼트 트리를 사용하여 부모 배열 저장

*힌트*: 이것은 도전적인 문제입니다. 트리 높이를 O(log N)으로 유지하기 위해 (경로 압축 없이) 랭크 기반 유니온(union by rank)을 사용하세요.

---

## 네비게이션

**이전**: [링크-컷 트리(Link-Cut Tree)](./31_Link_Cut_Tree.md)
