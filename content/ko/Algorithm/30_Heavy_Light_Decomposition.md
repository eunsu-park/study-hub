# 중간 경로 분해(Heavy-Light Decomposition)

**이전**: [실전 문제 풀이](./29_Problem_Solving.md) | **다음**: [링크-컷 트리](./31_Link_Cut_Tree.md)

---

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 트리에서 경로 쿼리를 단순하게 처리하면 O(N)이 걸리는 이유와 중간 경로 분해(Heavy-Light Decomposition, HLD)로 O(log²N)으로 줄이는 방법을 설명한다
2. 트리 간선을 중간 간선(heavy edge)과 경간선(light edge)으로 분류하고, 루트에서 잎까지의 경로가 최대 O(log N)개의 경간선을 지남을 증명한다
3. HLD를 구현하여 트리를 선형 배열로 펼치고, 세그먼트 트리를 이용해 경로 쿼리/업데이트를 수행한다
4. HLD의 부산물로 최소 공통 조상(LCA, Lowest Common Ancestor) 쿼리를 O(log N)에 해결한다
5. 경로 합, 경로 최댓값, 서브트리 쿼리를 포함하는 알고리즘 대회 문제에 HLD를 적용한다

---

## 목차

1. [문제 정의: 트리에서의 경로 쿼리](#1-문제-정의-트리에서의-경로-쿼리)
2. [중간 간선과 경간선](#2-중간-간선과-경간선)
3. [분해 알고리즘](#3-분해-알고리즘)
4. [세그먼트 트리로의 매핑](#4-세그먼트-트리로의-매핑)
5. [경로 쿼리와 업데이트](#5-경로-쿼리와-업데이트)
6. [서브트리 쿼리](#6-서브트리-쿼리)
7. [구현](#7-구현)
8. [연습문제(Exercises)](#8-연습문제exercises)

---

## 1. 문제 정의: 트리에서의 경로 쿼리

N개의 노드로 이루어진 트리에서 각 노드에 값이 있다고 할 때 다음 연산을 지원해야 합니다:
- **쿼리**: 노드 u에서 v까지의 경로 위 값들의 합(또는 최댓값, 최솟값)을 구한다
- **업데이트**: 단일 노드의 값을 변경한다(또는 경로 위 모든 노드를 변경한다)

**단순 접근**: u→LCA→v 경로를 직접 탐색하면 최악의 경우 O(N)개 노드를 방문한다. Q개의 쿼리가 있다면 총 O(NQ)가 된다.

**목표**: HLD + 세그먼트 트리를 이용해 쿼리당 O(log²N).

---

## 2. 중간 간선과 경간선

각 비잎(non-leaf) 노드에 대해 간선을 다음과 같이 분류합니다:

- **중간 간선(heavy edge)**: **가장 큰 서브트리**를 가진 자식으로 향하는 간선 (동률은 임의로 처리)
- **경간선(light edge)**: 나머지 자식으로 향하는 모든 간선

```
        1 (size=10)
       / \
      2   3 (size=6)    ← 1→3 간선이 중간 간선 (더 큰 서브트리)
     /   / \             ← 1→2 간선은 경간선
    4   5   6 (size=3)   ← 3→6 간선이 중간 간선
       / \
      7   8              ← 6은 자식이 없음
```

**핵심 성질**: 루트에서 임의의 잎까지 경로는 **최대 O(log N)개의 경간선**을 지납니다.

**증명 개요**: 경간선을 통해 부모에서 자식으로 내려갈 때, 자식의 서브트리 크기는 부모 서브트리 크기의 절반 이하입니다(그렇지 않으면 중간 간선이 됩니다). 따라서 경간선을 지날 때마다 서브트리 크기가 절반 이상 줄어들므로, 경간선의 수는 최대 log₂N개입니다.

**중간 경로(체인, chain)**: 연속된 중간 간선들이 체인을 형성합니다. 이 체인들이 트리를 최대 O(N)개의 체인으로 분할하지만, 루트에서 임의의 잎까지 경로는 최대 O(log N)개의 체인만 방문합니다.

---

## 3. 분해 알고리즘

### 1단계: 서브트리 크기 계산을 위한 DFS

```python
def dfs_size(node, parent):
    size[node] = 1
    for child in adj[node]:
        if child != parent:
            depth[child] = depth[node] + 1
            par[child] = node
            dfs_size(child, node)
            size[node] += size[child]
```

### 2단계: 체인 배정을 위한 DFS

**중간 자식을 먼저** 방문하여 각 체인이 연속된 위치를 차지하도록 합니다:

```python
timer = 0

def dfs_hld(node, parent, chain_head):
    pos[node] = timer  # position in the flat array
    timer += 1
    head[node] = chain_head

    # Find heavy child (largest subtree)
    heavy = -1
    max_size = 0
    for child in adj[node]:
        if child != parent and size[child] > max_size:
            max_size = size[child]
            heavy = child

    # Visit heavy child first (continues the chain)
    if heavy != -1:
        dfs_hld(heavy, node, chain_head)

    # Visit light children (each starts a new chain)
    for child in adj[node]:
        if child != parent and child != heavy:
            dfs_hld(child, node, child)  # new chain
```

이 DFS 이후, 각 체인은 `pos[]`에서 연속된 구간을 차지합니다.

---

## 4. 세그먼트 트리로의 매핑

펼쳐진 배열 `A[0..N-1]`에 대해 세그먼트 트리를 구축합니다. 여기서 `A[pos[v]] = value[v]`입니다.

각 체인이 연속적이므로:
- **체인 쿼리**: 세그먼트 트리에서 `[pos[head[v]], pos[v]]` 구간 쿼리
- **체인 업데이트**: 세그먼트 트리에서 `[pos[head[v]], pos[v]]` 구간 업데이트

체인당 O(log N)이 걸립니다.

---

## 5. 경로 쿼리와 업데이트

u → v 경로를 쿼리하려면:

```python
def path_query(u, v):
    result = 0  # identity for sum; -inf for max
    while head[u] != head[v]:
        # Move the deeper node's chain head up
        if depth[head[u]] < depth[head[v]]:
            u, v = v, u
        # Query the segment [head[u]..u] on the segment tree
        result = combine(result, seg_query(pos[head[u]], pos[u]))
        u = par[head[u]]  # jump to parent of chain head

    # Now u and v are on the same chain
    if depth[u] > depth[v]:
        u, v = v, u
    result = combine(result, seg_query(pos[u], pos[v]))
    return result
```

**시간 복잡도**: O(log N)개의 체인 × 세그먼트 트리 쿼리 O(log N) = **O(log²N)**.

---

## 6. 서브트리 쿼리

오일러 투어(Euler-tour) 순서의 보너스: 노드 v의 서브트리는 펼쳐진 배열에서 `pos[v]`부터 `pos[v] + size[v] - 1`까지 연속적으로 위치합니다.

```python
def subtree_query(v):
    return seg_query(pos[v], pos[v] + size[v] - 1)

def subtree_update(v, delta):
    seg_range_update(pos[v], pos[v] + size[v] - 1, delta)
```

이를 통해 O(log N) 서브트리 쿼리/업데이트를 추가 비용 없이 지원합니다.

---

## 7. 구현

### Python 전체 구현

```python
import sys
from sys import setrecursionlimit

def solve():
    # Build tree (1-indexed)
    N = int(input())
    adj = [[] for _ in range(N + 1)]
    values = [0] + list(map(int, input().split()))

    for _ in range(N - 1):
        u, v = map(int, input().split())
        adj[u].append(v)
        adj[v].append(u)

    # Step 1: compute sizes, parents, depths (iterative)
    size = [0] * (N + 1)
    par = [0] * (N + 1)
    depth = [0] * (N + 1)
    order = []

    stack = [(1, 0, False)]
    while stack:
        node, parent, visited = stack.pop()
        if visited:
            size[node] = 1
            for child in adj[node]:
                if child != parent:
                    size[node] += size[child]
            order.append(node)
            continue
        par[node] = parent
        stack.append((node, parent, True))
        for child in adj[node]:
            if child != parent:
                depth[child] = depth[node] + 1
                stack.append((child, node, False))

    # Step 2: HLD (iterative)
    pos = [0] * (N + 1)
    head = [0] * (N + 1)
    timer = 0

    stack = [(1, 1)]  # (node, chain_head)
    while stack:
        node, chain_head = stack.pop()
        pos[node] = timer
        timer += 1
        head[node] = chain_head

        # Find heavy child
        heavy = -1
        max_sz = 0
        for child in adj[node]:
            if child != par[node] and size[child] > max_sz:
                max_sz = size[child]
                heavy = child

        # Push light children first (processed last = DFS order)
        light_children = []
        for child in adj[node]:
            if child != par[node] and child != heavy:
                light_children.append(child)

        for child in reversed(light_children):
            stack.append((child, child))

        # Push heavy child last (processed next)
        if heavy != -1:
            stack.append((heavy, chain_head))

    # Build segment tree on flattened array
    seg = [0] * (4 * N)
    flat = [0] * N
    for v in range(1, N + 1):
        flat[pos[v]] = values[v]

    def build(node, lo, hi):
        if lo == hi:
            seg[node] = flat[lo]
            return
        mid = (lo + hi) // 2
        build(2 * node, lo, mid)
        build(2 * node + 1, mid + 1, hi)
        seg[node] = seg[2 * node] + seg[2 * node + 1]

    def query(node, lo, hi, ql, qr):
        if qr < lo or hi < ql:
            return 0
        if ql <= lo and hi <= qr:
            return seg[node]
        mid = (lo + hi) // 2
        return (query(2 * node, lo, mid, ql, qr) +
                query(2 * node + 1, mid + 1, hi, ql, qr))

    def update(node, lo, hi, idx, val):
        if lo == hi:
            seg[node] = val
            return
        mid = (lo + hi) // 2
        if idx <= mid:
            update(2 * node, lo, mid, idx, val)
        else:
            update(2 * node + 1, mid + 1, hi, idx, val)
        seg[node] = seg[2 * node] + seg[2 * node + 1]

    build(1, 0, N - 1)

    def path_sum(u, v):
        result = 0
        while head[u] != head[v]:
            if depth[head[u]] < depth[head[v]]:
                u, v = v, u
            result += query(1, 0, N - 1, pos[head[u]], pos[u])
            u = par[head[u]]
        if depth[u] > depth[v]:
            u, v = v, u
        result += query(1, 0, N - 1, pos[u], pos[v])
        return result

    # Process queries
    Q = int(input())
    for _ in range(Q):
        parts = input().split()
        if parts[0] == 'Q':
            u, v = int(parts[1]), int(parts[2])
            print(path_sum(u, v))
        else:  # Update
            v, val = int(parts[1]), int(parts[2])
            update(1, 0, N - 1, pos[v], val)
```

### 복잡도 요약

| 연산 | 시간 | 공간 |
|------|------|------|
| 전처리 (DFS + HLD) | O(N) | O(N) |
| 세그먼트 트리 구축 | O(N) | O(N) |
| 경로 쿼리 | O(log²N) | — |
| 경로 업데이트 | O(log²N) | — |
| 단일 업데이트 | O(log N) | — |
| 서브트리 쿼리 | O(log N) | — |

---

## 8. 연습문제(Exercises)

### 연습문제 1: 기본 HLD 구성

N개의 노드로 이루어진 트리에서 HLD를 구현하고 다음을 검증하세요:
1. 각 체인이 펼쳐진 배열에서 연속적으로 위치하는지
2. 임의의 루트-잎 경로의 경간선 수가 ≤ log₂N인지
3. 임의 노드의 서브트리가 연속된 구간을 차지하는지

### 연습문제 2: 경로 합 쿼리

경로 합 쿼리를 위한 HLD + 세그먼트 트리 전체를 구현하세요. 다음 케이스로 테스트합니다:
- 선형 체인(단순 접근의 최악의 경우, HLD는 여전히 O(log²N))
- 균형 이진 트리
- 별 그래프(star graph)

### 연습문제 3: 경로 최댓값 쿼리

경로 합 대신 경로 최댓값을 구하도록 구현을 수정하세요. 이를 이용해 다음 문제를 해결하세요: "u에서 v까지의 경로에서 최대 간선 가중치는?"

*힌트*: 각 간선의 가중치를 자식 노드(더 깊은 끝점)에 할당합니다.

### 연습문제 4: HLD를 이용한 LCA

별도의 LCA 자료구조 없이 HLD만으로 LCA를 구현하세요. 경로 쿼리 루프가 종료되는 노드가 u와 v의 LCA임을 보이세요.

### 연습문제 5: 업데이트가 있는 서브트리 합

두 가지 연산을 지원하세요:
1. v의 서브트리에 속하는 모든 노드에 값을 더하기
2. v의 서브트리에 속하는 모든 노드의 합 쿼리

HLD의 연속적인 서브트리 성질과 지연 세그먼트 트리(lazy segment tree)를 활용하세요.

---

## 탐색

**이전**: [실전 문제 풀이](./29_Problem_Solving.md) | **다음**: [링크-컷 트리](./31_Link_Cut_Tree.md)
