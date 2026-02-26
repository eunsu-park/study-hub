# 26. LCA와 트리쿼리 (LCA and Tree Queries)

## 학습 목표
- 최소 공통 조상(LCA)의 개념 이해
- Binary Lifting 기법 구현
- Sparse Table 활용
- 오일러 투어와 세그먼트 트리 결합
- 다양한 트리 쿼리 문제 해결

## 1. 최소 공통 조상(LCA)이란?

### 정의

트리에서 두 노드 u, v의 **최소 공통 조상(Lowest Common Ancestor)**은 u와 v의 공통 조상 중 가장 깊은(낮은) 노드입니다.

```
           1 (루트)
          /|\
         2 3 4
        /|   |
       5 6   7
      /|
     8 9

LCA(8, 6) = 2
LCA(8, 7) = 1
LCA(5, 6) = 2
LCA(8, 9) = 5
```

### LCA의 활용

```
┌─────────────────────────────────────────────────┐
│                LCA 활용 분야                     │
├─────────────────────────────────────────────────┤
│  • 두 노드 간 거리 계산                          │
│  • 두 노드 간 경로상의 쿼리 (합, 최대/최소)       │
│  • 트리 위의 DP 최적화                          │
│  • 네트워크 라우팅                              │
│  • 계통도/조직도 분석                           │
└─────────────────────────────────────────────────┘
```

---

## 2. 나이브 접근법

### 깊이 맞추기 + 동시 올라가기

```python
class NaiveLCA:
    def __init__(self, n, edges, root=0):
        self.n = n
        self.parent = [-1] * n
        self.depth = [0] * n

        # 트리 구성
        from collections import defaultdict, deque
        adj = defaultdict(list)
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)

        # BFS로 부모와 깊이 계산
        visited = [False] * n
        queue = deque([root])
        visited[root] = True

        while queue:
            node = queue.popleft()
            for neighbor in adj[node]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    self.parent[neighbor] = node
                    self.depth[neighbor] = self.depth[node] + 1
                    queue.append(neighbor)

    def lca(self, u, v):
        # 깊이 맞추기
        while self.depth[u] > self.depth[v]:
            u = self.parent[u]
        while self.depth[v] > self.depth[u]:
            v = self.parent[v]

        # 같은 깊이에서 동시에 올라가기
        while u != v:
            u = self.parent[u]
            v = self.parent[v]

        return u

# 사용 예시
edges = [(0,1), (0,2), (1,3), (1,4), (2,5)]
lca = NaiveLCA(6, edges, 0)
print(lca.lca(3, 4))  # 1
print(lca.lca(3, 5))  # 0
```

### 시간 복잡도

- **전처리**: O(N)
- **쿼리**: O(N) - 최악의 경우 트리 높이만큼

---

## 3. Binary Lifting

### 핵심 아이디어

각 노드에서 2^k번째 조상을 미리 계산해두면, 2의 거듭제곱 점프로 빠르게 올라갈 수 있습니다.

```
ancestor[node][k] = node의 2^k번째 조상

점프 예시 (13칸 올라가기):
13 = 1101₂ = 8 + 4 + 1
→ 2³점프 + 2²점프 + 2⁰점프 = 3번의 점프
```

### 점화식

```
ancestor[node][0] = parent[node]
ancestor[node][k] = ancestor[ancestor[node][k-1]][k-1]

2^k번째 조상 = (2^(k-1)번째 조상)의 2^(k-1)번째 조상
```

### 구현

```python
from collections import defaultdict, deque
import math

class BinaryLiftingLCA:
    def __init__(self, n, edges, root=0):
        self.n = n
        self.LOG = max(1, math.ceil(math.log2(n)))
        self.depth = [0] * n
        self.ancestor = [[-1] * self.LOG for _ in range(n)]

        # 트리 구성
        adj = defaultdict(list)
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)

        # BFS로 부모와 깊이 계산
        visited = [False] * n
        queue = deque([root])
        visited[root] = True

        while queue:
            node = queue.popleft()
            for neighbor in adj[node]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    self.ancestor[neighbor][0] = node
                    self.depth[neighbor] = self.depth[node] + 1
                    queue.append(neighbor)

        # ancestor 테이블 상향식 채우기: ancestor[node][k] = 2^k번째 조상.
        # 각 레벨을 이전 레벨에서 도출: 2^k 단계 점프는 먼저 2^(k-1) 단계
        # 점프하고, 그 중간점에서 다시 2^(k-1) 점프하면 된다.
        # 이렇게 하면 각 쿼리를 재계산하지 않고 O(N log N)에 테이블을 채운다.
        for k in range(1, self.LOG):
            for node in range(n):
                mid = self.ancestor[node][k-1]
                if mid != -1:
                    self.ancestor[node][k] = self.ancestor[mid][k-1]

    def kth_ancestor(self, node, k):
        """node의 k번째 조상 반환 (없으면 -1)"""
        # k를 이진수로 분해: 각 설정된 비트에 대해 독립적으로 점프.
        # 예) k=13 (1101₂) → 8 + 4 + 1 단계를 3번의 테이블 조회로 처리
        # 13번의 개별 부모 이동 대신.
        for i in range(self.LOG):
            if k & (1 << i):
                node = self.ancestor[node][i]
                if node == -1:
                    return -1
        return node

    def lca(self, u, v):
        # u를 더 깊은 노드로
        if self.depth[u] < self.depth[v]:
            u, v = v, u

        # 깊이 맞추기: 더 깊은 노드를 v와 같은 레벨로 올린다.
        diff = self.depth[u] - self.depth[v]
        u = self.kth_ancestor(u, diff)

        if u == v:
            return u

        # O(log N)에 동시에 올라가기: 가장 큰 점프부터 내림차순으로 순회하여
        # LCA를 절대 초과하지 않는다. 이 루프 이후 u와 v는 LCA 바로
        # 한 단계 아래에 위치하므로, ancestor[u][0]이 답이다.
        for k in range(self.LOG - 1, -1, -1):
            if self.ancestor[u][k] != self.ancestor[v][k]:
                u = self.ancestor[u][k]
                v = self.ancestor[v][k]

        return self.ancestor[u][0]

    def distance(self, u, v):
        """u와 v 사이의 거리"""
        lca_node = self.lca(u, v)
        return self.depth[u] + self.depth[v] - 2 * self.depth[lca_node]

# 사용 예시
edges = [(0,1), (0,2), (1,3), (1,4), (2,5), (3,6), (3,7)]
lca = BinaryLiftingLCA(8, edges, 0)

print(f"LCA(6,7) = {lca.lca(6, 7)}")  # 3
print(f"LCA(6,5) = {lca.lca(6, 5)}")  # 0
print(f"LCA(4,7) = {lca.lca(4, 7)}")  # 1
print(f"Distance(6,7) = {lca.distance(6, 7)}")  # 2
```

### C++ 구현

```cpp
#include <bits/stdc++.h>
using namespace std;

class BinaryLiftingLCA {
private:
    int n, LOG;
    vector<int> depth;
    vector<vector<int>> ancestor;

public:
    BinaryLiftingLCA(int n, vector<pair<int,int>>& edges, int root = 0)
        : n(n), LOG(max(1, (int)ceil(log2(n)))),
          depth(n, 0), ancestor(n, vector<int>(LOG, -1)) {

        vector<vector<int>> adj(n);
        for (auto& [u, v] : edges) {
            adj[u].push_back(v);
            adj[v].push_back(u);
        }

        vector<bool> visited(n, false);
        queue<int> q;
        q.push(root);
        visited[root] = true;

        while (!q.empty()) {
            int node = q.front(); q.pop();
            for (int next : adj[node]) {
                if (!visited[next]) {
                    visited[next] = true;
                    ancestor[next][0] = node;
                    depth[next] = depth[node] + 1;
                    q.push(next);
                }
            }
        }

        for (int k = 1; k < LOG; k++) {
            for (int i = 0; i < n; i++) {
                int mid = ancestor[i][k-1];
                if (mid != -1) ancestor[i][k] = ancestor[mid][k-1];
            }
        }
    }

    int lca(int u, int v) {
        if (depth[u] < depth[v]) swap(u, v);

        int diff = depth[u] - depth[v];
        for (int k = 0; k < LOG; k++) {
            if (diff & (1 << k)) u = ancestor[u][k];
        }

        if (u == v) return u;

        for (int k = LOG - 1; k >= 0; k--) {
            if (ancestor[u][k] != ancestor[v][k]) {
                u = ancestor[u][k];
                v = ancestor[v][k];
            }
        }

        return ancestor[u][0];
    }
};
```

### 시간 복잡도

- **전처리**: O(N log N)
- **쿼리**: O(log N)
- **공간**: O(N log N)

---

## 4. 오일러 투어 + RMQ

### 핵심 아이디어

1. DFS로 오일러 투어 수행 (방문 순서 기록)
2. LCA(u, v) = 오일러 투어에서 u~v 구간의 최소 깊이 노드
3. Sparse Table로 O(1) RMQ

### 오일러 투어

```
        0
       / \
      1   2
     / \
    3   4

DFS 방문 순서 (오일러 투어):
0 → 1 → 3 → 1 → 4 → 1 → 0 → 2 → 0

euler = [0, 1, 3, 1, 4, 1, 0, 2, 0]
depth = [0, 1, 2, 1, 2, 1, 0, 1, 0]
first = {0:0, 1:1, 2:7, 3:2, 4:4}  # 각 노드의 첫 등장 위치

LCA(3, 4):
  first[3] = 2, first[4] = 4
  euler[2:5] = [3, 1, 4]
  depth[2:5] = [2, 1, 2]
  최소 깊이 위치 = 3 → 노드 1
```

### 구현

```python
import math

class EulerTourLCA:
    def __init__(self, n, edges, root=0):
        self.n = n
        self.euler = []
        self.depth_arr = []
        self.first = [-1] * n

        # 트리 구성
        from collections import defaultdict
        adj = defaultdict(list)
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)

        # DFS로 오일러 투어
        visited = [False] * n
        depth = [0] * n

        def dfs(node, d):
            visited[node] = True
            depth[node] = d
            self.first[node] = len(self.euler)
            self.euler.append(node)
            self.depth_arr.append(d)

            for neighbor in adj[node]:
                if not visited[neighbor]:
                    dfs(neighbor, d + 1)
                    self.euler.append(node)
                    self.depth_arr.append(d)

        dfs(root, 0)

        # Sparse Table 구성
        self._build_sparse_table()

    def _build_sparse_table(self):
        m = len(self.euler)
        self.LOG = max(1, math.ceil(math.log2(m)))
        self.sparse = [[0] * m for _ in range(self.LOG)]

        # sparse[0][i] = i 위치의 인덱스 (자기 자신)
        for i in range(m):
            self.sparse[0][i] = i

        # O(N log N)으로 구축: 각 항목은 크기 2^(k-1)인 겹치지 않는 두 반을 병합.
        # 쿼리는 나중에 이 미리 계산된 범위를 재사용하여 겹치는 두 개의
        # 2의 거듭제곱 윈도우로 임의의 [l,r] RMQ를 O(1)에 답한다.
        # sparse[k][i] = 구간 [i, i+2^k) 중 최소 깊이 위치
        for k in range(1, self.LOG):
            for i in range(m - (1 << k) + 1):
                left = self.sparse[k-1][i]
                right = self.sparse[k-1][i + (1 << (k-1))]
                if self.depth_arr[left] <= self.depth_arr[right]:
                    self.sparse[k][i] = left
                else:
                    self.sparse[k][i] = right

    def _rmq(self, l, r):
        """구간 [l, r]에서 최소 깊이 위치 반환"""
        length = r - l + 1
        k = int(math.log2(length))
        # 크기 2^k인 두 겹치는 윈도우가 전체 [l, r]을 커버한다.
        # 최솟값 쿼리에서 겹침은 무해하다 — 중복 원소의 min을 취해도
        # 결과가 변하지 않기 때문 — 이것이 스파스 테이블을 쿼리당
        # O(1)로 만드는 핵심 속성이다.
        left = self.sparse[k][l]
        right = self.sparse[k][r - (1 << k) + 1]
        if self.depth_arr[left] <= self.depth_arr[right]:
            return left
        return right

    def lca(self, u, v):
        l, r = self.first[u], self.first[v]
        if l > r:
            l, r = r, l
        idx = self._rmq(l, r)
        return self.euler[idx]

# 사용 예시
edges = [(0,1), (0,2), (1,3), (1,4)]
lca = EulerTourLCA(5, edges, 0)
print(f"LCA(3,4) = {lca.lca(3, 4)}")  # 1
print(f"LCA(3,2) = {lca.lca(3, 2)}")  # 0
```

### 시간 복잡도

- **전처리**: O(N log N)
- **쿼리**: O(1)
- **공간**: O(N log N)

---

## 5. 트리 경로 쿼리

### 경로 합 쿼리

u에서 v까지의 경로에 있는 노드 값의 합

```python
class TreePathSum:
    def __init__(self, n, edges, values, root=0):
        self.lca_solver = BinaryLiftingLCA(n, edges, root)
        self.prefix = [0] * n  # 루트에서 각 노드까지의 합

        # DFS로 prefix sum 계산
        from collections import defaultdict
        adj = defaultdict(list)
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)

        visited = [False] * n
        self.values = values

        def dfs(node, parent_sum):
            visited[node] = True
            self.prefix[node] = parent_sum + values[node]
            for neighbor in adj[node]:
                if not visited[neighbor]:
                    dfs(neighbor, self.prefix[node])

        dfs(root, 0)

    def path_sum(self, u, v):
        """u에서 v까지의 경로 합"""
        lca_node = self.lca_solver.lca(u, v)
        # prefix[u] + prefix[v] - prefix[lca] - prefix[parent(lca)]
        # = prefix[u] + prefix[v] - 2*prefix[lca] + values[lca]
        return (self.prefix[u] + self.prefix[v]
                - 2 * self.prefix[lca_node] + self.values[lca_node])

# 사용 예시
edges = [(0,1), (0,2), (1,3), (1,4)]
values = [1, 2, 3, 4, 5]  # 각 노드의 값
tree = TreePathSum(5, edges, values, 0)
print(f"경로 합(3,4) = {tree.path_sum(3, 4)}")  # 4+2+5 = 11
print(f"경로 합(3,2) = {tree.path_sum(3, 2)}")  # 4+2+1+3 = 10
```

### 경로 최댓값 쿼리

Binary Lifting에 최댓값 정보 추가

```python
class TreePathMax:
    def __init__(self, n, edges, values, root=0):
        self.n = n
        self.LOG = max(1, math.ceil(math.log2(n)))
        self.depth = [0] * n
        self.ancestor = [[-1] * self.LOG for _ in range(n)]
        self.max_val = [[0] * self.LOG for _ in range(n)]  # 경로상 최댓값
        self.values = values

        # 트리 구성 및 전처리
        from collections import defaultdict, deque
        adj = defaultdict(list)
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)

        visited = [False] * n
        queue = deque([root])
        visited[root] = True

        while queue:
            node = queue.popleft()
            for neighbor in adj[node]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    self.ancestor[neighbor][0] = node
                    self.depth[neighbor] = self.depth[node] + 1
                    self.max_val[neighbor][0] = max(values[neighbor], values[node])
                    queue.append(neighbor)

        # 2^k 조상 및 경로 최댓값 계산
        for k in range(1, self.LOG):
            for node in range(n):
                mid = self.ancestor[node][k-1]
                if mid != -1:
                    self.ancestor[node][k] = self.ancestor[mid][k-1]
                    self.max_val[node][k] = max(
                        self.max_val[node][k-1],
                        self.max_val[mid][k-1]
                    )

    def query(self, u, v):
        """u에서 v까지 경로의 최댓값"""
        result = max(self.values[u], self.values[v])

        if self.depth[u] < self.depth[v]:
            u, v = v, u

        # u를 v의 깊이로 올리면서 최댓값 갱신
        diff = self.depth[u] - self.depth[v]
        for k in range(self.LOG):
            if diff & (1 << k):
                result = max(result, self.max_val[u][k])
                u = self.ancestor[u][k]

        if u == v:
            return result

        # 동시에 올라가면서 최댓값 갱신
        for k in range(self.LOG - 1, -1, -1):
            if self.ancestor[u][k] != self.ancestor[v][k]:
                result = max(result, self.max_val[u][k], self.max_val[v][k])
                u = self.ancestor[u][k]
                v = self.ancestor[v][k]

        # 마지막으로 LCA까지
        result = max(result, self.max_val[u][0], self.max_val[v][0])
        return result
```

---

## 6. Heavy-Light Decomposition (HLD)

### 핵심 아이디어

트리를 **Heavy 경로**와 **Light 경로**로 분해하여, 경로 쿼리를 O(log²N)에 처리

```
Heavy Edge: 자식 중 서브트리 크기가 가장 큰 간선
Light Edge: 나머지 간선

        1
       /|\
     [2] 3 4     []: Heavy edge
     /|
   [5] 6
   /
  7

Heavy 경로: 1-2-5-7
```

### 구현 개요

```python
class HLD:
    def __init__(self, n, edges, root=0):
        self.n = n
        self.parent = [-1] * n
        self.depth = [0] * n
        self.subtree_size = [1] * n
        self.chain_head = [0] * n  # 체인의 시작 노드
        self.chain_pos = [0] * n   # 체인 내 순서 (세그트리 인덱스)
        self.chain_arr = []        # 실제 노드 순서

        # 트리 구성
        from collections import defaultdict
        self.adj = defaultdict(list)
        for u, v in edges:
            self.adj[u].append(v)
            self.adj[v].append(u)

        # DFS 1: 서브트리 크기, 부모, 깊이
        self._dfs_size(root, -1, 0)

        # DFS 2: HLD 분해
        self._dfs_hld(root, root)

    def _dfs_size(self, node, parent, depth):
        self.parent[node] = parent
        self.depth[node] = depth

        for i, child in enumerate(self.adj[node]):
            if child != parent:
                self._dfs_size(child, node, depth + 1)
                self.subtree_size[node] += self.subtree_size[child]

                # Heavy child를 맨 앞으로
                if self.subtree_size[child] > self.subtree_size[self.adj[node][0]]:
                    self.adj[node][0], self.adj[node][i] = self.adj[node][i], self.adj[node][0]

    def _dfs_hld(self, node, head):
        self.chain_head[node] = head
        self.chain_pos[node] = len(self.chain_arr)
        self.chain_arr.append(node)

        for child in self.adj[node]:
            if child != self.parent[node]:
                if child == self.adj[node][0]:
                    # Heavy child: 같은 체인
                    self._dfs_hld(child, head)
                else:
                    # Light child: 새 체인
                    self._dfs_hld(child, child)

    def path_query(self, u, v, seg_tree):
        """u에서 v까지의 경로 쿼리 (세그먼트 트리 사용)"""
        result = 0  # 또는 적절한 항등원

        while self.chain_head[u] != self.chain_head[v]:
            # 더 깊은 체인을 올림
            if self.depth[self.chain_head[u]] < self.depth[self.chain_head[v]]:
                u, v = v, u

            # 현재 체인에서 쿼리
            result = max(result, seg_tree.query(
                self.chain_pos[self.chain_head[u]],
                self.chain_pos[u]
            ))
            u = self.parent[self.chain_head[u]]

        # 같은 체인 내에서 쿼리
        if self.depth[u] > self.depth[v]:
            u, v = v, u
        result = max(result, seg_tree.query(
            self.chain_pos[u],
            self.chain_pos[v]
        ))

        return result
```

### 시간 복잡도

- **전처리**: O(N)
- **경로 쿼리**: O(log²N)
- **경로 업데이트**: O(log²N)

---

## 7. 실전 문제 패턴

### 패턴 1: 두 노드 간 거리

```python
def distance(lca_solver, u, v):
    lca_node = lca_solver.lca(u, v)
    return (lca_solver.depth[u] + lca_solver.depth[v]
            - 2 * lca_solver.depth[lca_node])
```

### 패턴 2: 경로에 특정 노드 포함 여부

```python
def is_on_path(lca_solver, u, v, x):
    """x가 u-v 경로에 있는지 확인"""
    lca_uv = lca_solver.lca(u, v)
    lca_ux = lca_solver.lca(u, x)
    lca_vx = lca_solver.lca(v, x)

    # x가 경로에 있으려면:
    # 1. LCA(u,x) = x이고 LCA(x,v) = LCA(u,v), 또는
    # 2. LCA(v,x) = x이고 LCA(x,u) = LCA(u,v)
    if lca_ux == x and lca_solver.lca(x, v) == lca_uv:
        return True
    if lca_vx == x and lca_solver.lca(x, u) == lca_uv:
        return True
    return False
```

### 패턴 3: 서브트리 쿼리 (오일러 투어)

```python
class SubtreeQuery:
    def __init__(self, n, edges, root=0):
        self.n = n
        self.enter = [0] * n  # 서브트리 시작 인덱스
        self.leave = [0] * n  # 서브트리 끝 인덱스
        self.order = []        # DFS 순서

        from collections import defaultdict
        adj = defaultdict(list)
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)

        visited = [False] * n
        self.timer = 0

        def dfs(node):
            visited[node] = True
            self.enter[node] = self.timer
            self.order.append(node)
            self.timer += 1

            for child in adj[node]:
                if not visited[child]:
                    dfs(child)

            self.leave[node] = self.timer - 1

        dfs(root)

    def subtree_range(self, node):
        """node의 서브트리가 order에서 차지하는 범위 [l, r]"""
        return self.enter[node], self.leave[node]

# 세그먼트 트리와 결합하여 서브트리 합/최대 등 쿼리
```

### 패턴 4: K번째 조상

```python
def kth_ancestor(lca_solver, node, k):
    """Binary Lifting으로 k번째 조상"""
    for i in range(lca_solver.LOG):
        if k & (1 << i):
            node = lca_solver.ancestor[node][i]
            if node == -1:
                return -1
    return node
```

### 패턴 5: 경로의 K번째 노드

```python
def kth_node_on_path(lca_solver, u, v, k):
    """u에서 v로 가는 경로의 k번째 노드 (0-indexed)"""
    lca_node = lca_solver.lca(u, v)
    dist_u_lca = lca_solver.depth[u] - lca_solver.depth[lca_node]
    dist_v_lca = lca_solver.depth[v] - lca_solver.depth[lca_node]
    total = dist_u_lca + dist_v_lca

    if k > total:
        return -1

    if k <= dist_u_lca:
        # u에서 LCA 방향으로 k칸
        return kth_ancestor(lca_solver, u, k)
    else:
        # LCA에서 v 방향으로 (k - dist_u_lca)칸
        # = v에서 (total - k)칸 위
        return kth_ancestor(lca_solver, v, total - k)
```

---

## 8. 시간 복잡도 정리

| 방법 | 전처리 | 쿼리 | 공간 |
|------|--------|------|------|
| 나이브 | O(N) | O(N) | O(N) |
| Binary Lifting | O(N log N) | O(log N) | O(N log N) |
| 오일러 투어 + RMQ | O(N log N) | O(1) | O(N log N) |
| HLD + 세그트리 | O(N) | O(log²N) | O(N) |

---

## 9. 자주 하는 실수

### 실수 1: 루트 설정 오류

```python
# 트리가 0-indexed인지 1-indexed인지 확인
root = 0  # 또는 1
```

### 실수 2: 조상 배열 초기화

```python
# 루트의 조상은 -1 또는 루트 자신
ancestor[root][0] = -1  # 또는 root
```

### 실수 3: 깊이 비교 방향

```python
# 잘못됨: 더 깊은 노드를 먼저 올려야 함
if depth[u] > depth[v]:  # v가 더 깊은 경우 처리 안됨!
    u = kth_ancestor(u, diff)

# 올바름
if depth[u] < depth[v]:
    u, v = v, u
diff = depth[u] - depth[v]
u = kth_ancestor(u, diff)
```

---

## 10. 연습 문제

| 난이도 | 문제 유형 | 핵심 개념 |
|--------|----------|-----------|
| ★★☆ | LCA 기본 | Binary Lifting |
| ★★☆ | 두 노드 거리 | LCA + 깊이 |
| ★★★ | 경로 합 쿼리 | Prefix Sum + LCA |
| ★★★ | 경로 최댓값 | Binary Lifting 확장 |
| ★★★★ | 경로 업데이트 | HLD + 세그트리 |

---

## 다음 단계

- [17_Strongly_Connected_Components.md](./17_Strongly_Connected_Components.md) - SCC, 타잔

---

## 학습 점검

1. Binary Lifting에서 2^k 조상을 미리 계산하는 이유는?
2. 오일러 투어로 LCA를 RMQ로 변환하는 원리는?
3. HLD에서 Heavy Edge를 선택하는 기준은?
4. 경로 쿼리를 처리할 때 LCA를 활용하는 방법은?
