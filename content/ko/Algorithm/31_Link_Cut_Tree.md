# 링크-컷 트리(Link-Cut Tree)

**이전**: [중간 경로 분해](./30_Heavy_Light_Decomposition.md) | **다음**: [영속 세그먼트 트리](./32_Persistent_Segment_Tree.md)

---

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 정적(static) HLD가 동적 트리에 적합하지 않은 이유와 링크-컷 트리(Link-Cut Tree)가 링크/컷 연산을 균등 O(log N)에 지원하는 방법을 설명한다
2. 선호 경로 분해(preferred-path decomposition)와 각 선호 경로를 스플레이 트리(splay tree)로 표현하는 방법을 설명한다
3. `access`, `link`, `cut` 연산을 구현하고 균등 O(log N) 복잡도를 분석한다
4. 링크-컷 트리를 이용해 포레스트(forest)에서의 동적 연결성(dynamic connectivity) 쿼리를 해결한다
5. 동적 LCA(Lowest Common Ancestor), 경로 집계(path aggregate), 증분 트리 구성 문제에 링크-컷 트리를 적용한다

---

## 목차

1. [동기: 동적 트리](#1-동기-동적-트리)
2. [선호 경로와 보조 트리](#2-선호-경로와-보조-트리)
3. [스플레이 트리 복습](#3-스플레이-트리-복습)
4. [핵심 연산](#4-핵심-연산)
5. [동적 트리에서의 경로 쿼리](#5-동적-트리에서의-경로-쿼리)
6. [구현](#6-구현)
7. [응용](#7-응용)
8. [연습문제(Exercises)](#8-연습문제exercises)

---

## 1. 동기: 동적 트리

HLD(30강)는 강력하지만 **정적(static)**입니다 — 트리 구조가 변경될 수 없습니다. 많은 문제에서 다음 연산이 필요합니다:

- **Link(u, v)**: 포레스트 내 두 트리 사이에 간선을 추가한다
- **Cut(u, v)**: 간선을 제거하여 하나의 트리를 둘로 분리한다
- **경로 쿼리(path query)**: u→v 경로 위의 값들을 집계한다
- **Connected(u, v)**: u와 v가 같은 트리에 있는지 확인한다

**예시**: 동적 연결성, 네트워크 설계, 증분 MST(최소 신장 트리).

| 기능 | HLD | 링크-컷 트리 |
|------|-----|-------------|
| 경로 쿼리 | O(log²N) | 균등 O(log N) |
| 단일 업데이트 | O(log N) | 균등 O(log N) |
| 링크/컷 | 미지원 | 균등 O(log N) |
| 구성 | O(N) 일회성 | O(N log N) |

---

## 2. 선호 경로와 보조 트리

### 선호 자식(Preferred Child)

언제든지 각 노드는 최대 하나의 **선호 자식(preferred child)**을 가집니다 — 루트-노드 경로에서 가장 최근에 접근된 자식입니다. 선호 자식으로 향하는 간선을 **선호 간선(preferred edge)**이라 합니다.

**선호 경로(preferred path)**: 선호 간선들이 이루는 최대 체인(maximal chain)이 어떤 노드에서 잎을 향해 내려가는 경로를 형성합니다. 이 경로들은 트리를 서로 소인 선호 경로들로 분할합니다.

```
         1
        / \
       2   3        선호 간선: 1→3, 3→5
      /   / \       선호 경로: [1,3,5], [2], [4], [6]
     4   5   6
```

### 보조 트리(Auxiliary Trees, 스플레이 트리)

각 선호 경로는 스플레이 트리(splay tree)에 저장됩니다. 키(key)는 **깊이(depth)**로, 원래 트리에서 더 깊은 노드가 스플레이 트리에서 더 큰 키를 가집니다.

스플레이 트리들은 서로 연결됩니다: 각 스플레이 트리의 루트는 표현된 트리에서 해당 선호 경로가 연결되는 노드를 가리키는 **경로-부모 포인터(path-parent pointer)**를 가집니다. 이는 단방향 포인터입니다(부모는 이를 알지 못합니다).

---

## 3. 스플레이 트리 복습

스플레이 트리(splay tree)는 자기 균형 이진 탐색 트리(self-balancing BST)로, 접근된 노드를 회전(rotation)을 통해 루트로 이동시킵니다:

- **지그(Zig)**: 단일 회전(부모가 루트일 때)
- **지그-지그(Zig-zig)**: 같은 방향으로 두 번 회전(노드와 부모가 같은 쪽에 있을 때)
- **지그-재그(Zig-zag)**: 반대 방향으로 두 번 회전

연산당 **균등 O(log N)** (포텐셜 함수로 증명됨).

링크-컷 트리에서의 핵심:
```
splay(x): rotate x to the root of its auxiliary tree
```

---

## 4. 핵심 연산

### `access(v)` — 기본 연산

`access(v)`는 v를 루트에서의 선호 경로에서 가장 깊은 노드로 만듭니다. 이 과정에서 경로를 따라 선호 간선들이 변경됩니다.

**알고리즘**:
1. v를 보조 트리에서 스플레이한다
2. v의 오른쪽 자식 연결을 끊는다 (더 깊은 노드들이 더 이상 선호되지 않음)
3. 경로-부모 포인터를 통해 위로 올라간다:
   - 각 단계에서 경로-부모 노드를 스플레이한다
   - v의 트리를 오른쪽 자식으로 연결한다
   - v를 다시 스플레이한다
4. v가 루트의 보조 트리 안에 있을 때까지 반복한다

```
access(v):
    splay(v)
    v.right = null   // cut preferred path below v
    while v.path_parent != null:
        w = v.path_parent
        splay(w)
        w.right = v   // make v's path preferred
        splay(v)       // v becomes root
```

`access(v)` 이후, 루트에서 v까지의 경로는 완전히 선호 간선으로 구성되며 v의 스플레이 트리에 저장됩니다.

### `link(u, v)` — 두 트리 연결

u를 v의 자식으로 만든다(u는 루트여야 함):

```
link(u, v):
    access(u)    // u is now root of its splay tree, no right child
    access(v)    // v is now the deepest on root's preferred path
    u.left = v   // v becomes u's left child in the splay tree
    v.parent = u // (but v's tree might need path-parent adjustment)
```

실제 표준 구현:
```
link(u, v):
    make_root(u)  // reroot u's tree at u
    access(v)
    u.path_parent = v
```

### `cut(u, v)` — 간선 제거

u와 v 사이의 간선을 제거합니다:

```
cut(u, v):
    make_root(u)
    access(v)
    // Now u is v's left child in the splay tree
    v.left = null
    u.parent = null
```

### `make_root(v)` — 트리 재루팅

v에서 현재 루트까지의 경로를 반전시켜 v를 표현된 트리의 루트로 만듭니다:

```
make_root(v):
    access(v)
    reverse(v's splay tree)  // flip left/right subtrees
    // This reverses the depth ordering, making v the shallowest
```

### `find_root(v)` — 루트 탐색

```
find_root(v):
    access(v)
    // Go to leftmost node (shallowest = root)
    while v.left != null:
        v = v.left
    splay(v)
    return v
```

### `connected(u, v)`

```
connected(u, v):
    return find_root(u) == find_root(v)
```

---

## 5. 동적 트리에서의 경로 쿼리

스플레이 트리 노드에 집계값(합, 최댓값, 최솟값)을 유지합니다:

```
path_aggregate(u, v):
    make_root(u)
    access(v)
    return v.aggregate  // aggregate of entire splay tree = path u→v
```

각 스플레이 트리 노드는 다음을 저장합니다:
- `val`: 노드 자체의 값
- `agg`: 스플레이 트리에서 해당 노드의 서브트리 집계값

회전 중 집계값 갱신:
```
pull(v):
    v.agg = combine(v.left.agg, v.val, v.right.agg)
```

---

## 6. 구현

### Python 구현 (간략화)

```python
class Node:
    __slots__ = ('ch', 'p', 'rev', 'val', 'agg', 'sz')

    def __init__(self, val=0):
        self.ch = [None, None]  # left, right children
        self.p = None            # parent (in splay tree or path-parent)
        self.rev = False         # lazy reverse flag
        self.val = val
        self.agg = val
        self.sz = 1

def is_root(x):
    """Is x the root of its auxiliary (splay) tree?"""
    p = x.p
    return p is None or (p.ch[0] != x and p.ch[1] != x)

def pull(x):
    """Update aggregate from children."""
    if x is None:
        return
    x.agg = x.val
    x.sz = 1
    for c in x.ch:
        if c is not None:
            x.agg = x.agg + c.agg  # sum; change for max/min
            x.sz += c.sz

def push(x):
    """Propagate lazy reverse."""
    if x is not None and x.rev:
        x.ch[0], x.ch[1] = x.ch[1], x.ch[0]
        for c in x.ch:
            if c is not None:
                c.rev = not c.rev
        x.rev = False

def rotate(x):
    """Single rotation."""
    p = x.p
    g = p.p
    d = 0 if p.ch[1] == x else 1  # direction

    # x's opposite child becomes p's child
    p.ch[1 - d] = x.ch[d]
    if x.ch[d] is not None:
        x.ch[d].p = p

    # x becomes parent
    x.ch[d] = p
    p.p = x
    x.p = g

    if g is not None:
        if g.ch[0] == p:
            g.ch[0] = x
        elif g.ch[1] == p:
            g.ch[1] = x
        # else: path-parent pointer (don't modify)

    pull(p)
    pull(x)

def splay(x):
    """Splay x to the root of its auxiliary tree."""
    # Push lazy flags from root down to x
    stack = []
    y = x
    while not is_root(y):
        stack.append(y.p)
        y = y.p
    stack.append(y)
    while stack:
        push(stack.pop())

    while not is_root(x):
        p = x.p
        if not is_root(p):
            g = p.p
            # Zig-zig or zig-zag
            same_dir = (g.ch[0] == p) == (p.ch[0] == x)
            if same_dir:
                rotate(p)  # zig-zig: rotate parent first
            else:
                rotate(x)  # zig-zag: rotate x first
        rotate(x)

def access(x):
    """Make x the deepest node on the preferred path from root."""
    last = None
    y = x
    while y is not None:
        splay(y)
        y.ch[1] = last  # change preferred child
        pull(y)
        last = y
        y = y.p
    splay(x)

def make_root(x):
    """Make x the root of its represented tree."""
    access(x)
    x.rev = not x.rev
    push(x)

def find_root(x):
    """Find the root of x's tree."""
    access(x)
    while x.ch[0] is not None:
        push(x)
        x = x.ch[0]
    splay(x)
    return x

def link(x, y):
    """Add edge between x and y (x and y must be in different trees)."""
    make_root(x)
    x.p = y

def cut(x, y):
    """Remove edge between x and y."""
    make_root(x)
    access(y)
    y.ch[0] = None
    x.p = None
    pull(y)

def connected(x, y):
    """Are x and y in the same tree?"""
    return find_root(x) == find_root(y)

def path_aggregate(x, y):
    """Aggregate values on the path from x to y."""
    make_root(x)
    access(y)
    return y.agg
```

### 복잡도

| 연산 | 균등 시간 |
|------|----------|
| access | O(log N) |
| link | O(log N) |
| cut | O(log N) |
| find_root | O(log N) |
| path_aggregate | O(log N) |
| connected | O(log N) |

---

## 7. 응용

### 동적 연결성(Dynamic Connectivity)

```python
# Maintain a forest with link/cut operations
# Query: are u and v connected?
nodes = [Node(i) for i in range(N)]

link(nodes[0], nodes[1])
link(nodes[1], nodes[2])
print(connected(nodes[0], nodes[2]))  # True

cut(nodes[0], nodes[1])
print(connected(nodes[0], nodes[2]))  # False
```

### 동적 MST(최소 신장 트리)

그래프의 신장 트리를 유지합니다. 새로운 간선 (u, v, w)가 추가될 때:
1. u와 v가 연결되어 있지 않으면 링크한다
2. 연결되어 있으면 u→v 경로에서 최대 가중치 간선을 찾는다
3. 새 가중치 < 최대 가중치이면 기존 간선을 컷하고 새 간선을 링크한다

### 네트워크 플로우(Dinic 알고리즘 + 링크-컷 트리)

Dinic 알고리즘(Dinic's algorithm)은 링크-컷 트리를 사용해 증가 경로(augmenting path)를 유지할 수 있으며, 단위 용량 그래프에서의 복잡도를 개선합니다.

---

## 8. 연습문제(Exercises)

### 연습문제 1: 기본 링크-컷 트리 연산

링크-컷 트리를 구현하고 다음 연산 시퀀스로 검증하세요:
1. 10개의 노드를 생성한다
2. 경로로 링크한다: 1→2→3→...→10
3. 1에서 10까지 경로 합을 쿼리한다
4. 간선 5→6을 컷하고 연결 해제를 확인한다
5. 5를 8에 링크하고 새로운 연결 관계를 확인한다

### 연습문제 2: 동적 연결성

링크와 컷 연산이 발생하는 포레스트에서 연결성 쿼리에 답하세요. 링크만 지원하는 유니온-파인드(Union-Find)와 성능을 비교하세요.

### 연습문제 3: 업데이트가 있는 경로 최댓값

다음 연산을 지원하세요:
1. `update(v, new_val)`: 노드 v의 값을 변경한다
2. `query(u, v)`: u→v 경로에서 최댓값을 구한다
3. `link(u, v)` 및 `cut(u, v)`: 동적 트리 수정

### 연습문제 4: 동적 LCA

링크-컷 트리를 이용해 동적 트리에서 LCA 쿼리를 구현하세요. `make_root(u)` 후 `access(v)`를 수행하면, access 경로에서 v 직전에 스플레이된 마지막 노드가 LCA입니다.

### 연습문제 5: 최소 신장 트리 유지

동적 MST 알고리즘을 구현하세요:
1. 빈 그래프에서 시작한다
2. 간선을 하나씩 추가한다
3. MST를 유지하며 더 좋은 간선이 발견되면 교체한다
4. 링크-컷 트리로 경로 위의 최대 간선을 탐색한다

---

## 탐색

**이전**: [중간 경로 분해](./30_Heavy_Light_Decomposition.md) | **다음**: [영속 세그먼트 트리](./32_Persistent_Segment_Tree.md)
