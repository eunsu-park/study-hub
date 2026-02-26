# 26. LCA and Tree Queries

## Learning Objectives
- Understanding the Lowest Common Ancestor (LCA) concept
- Implementing Binary Lifting technique
- Utilizing Sparse Tables
- Combining Euler Tour with Segment Trees
- Solving various tree query problems

## 1. What is Lowest Common Ancestor (LCA)?

### Definition

In a tree, the **Lowest Common Ancestor** of two nodes u and v is the deepest (lowest) node that is an ancestor of both u and v.

```
           1 (root)
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

### Applications of LCA

```
┌─────────────────────────────────────────────────┐
│                LCA Applications                  │
├─────────────────────────────────────────────────┤
│  • Distance calculation between two nodes        │
│  • Queries on paths (sum, max/min)              │
│  • Tree DP optimization                         │
│  • Network routing                              │
│  • Genealogy/organizational chart analysis      │
└─────────────────────────────────────────────────┘
```

---

## 2. Naive Approach

### Align Depth + Climb Together

```python
class NaiveLCA:
    def __init__(self, n, edges, root=0):
        self.n = n
        self.parent = [-1] * n
        self.depth = [0] * n

        # Build tree
        from collections import defaultdict, deque
        adj = defaultdict(list)
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)

        # BFS to calculate parent and depth
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
        # Align depth
        while self.depth[u] > self.depth[v]:
            u = self.parent[u]
        while self.depth[v] > self.depth[u]:
            v = self.parent[v]

        # Climb together at same depth
        while u != v:
            u = self.parent[u]
            v = self.parent[v]

        return u

# Usage example
edges = [(0,1), (0,2), (1,3), (1,4), (2,5)]
lca = NaiveLCA(6, edges, 0)
print(lca.lca(3, 4))  # 1
print(lca.lca(3, 5))  # 0
```

### Time Complexity

- **Preprocessing**: O(N)
- **Query**: O(N) - In worst case, tree height

---

## 3. Binary Lifting

### Key Idea

By precomputing the 2^k-th ancestor of each node, we can quickly jump up using powers of 2.

```
ancestor[node][k] = 2^k-th ancestor of node

Jump example (climb 13 steps):
13 = 1101₂ = 8 + 4 + 1
→ 2³ jump + 2² jump + 2⁰ jump = 3 jumps
```

### Recurrence Relation

```
ancestor[node][0] = parent[node]
ancestor[node][k] = ancestor[ancestor[node][k-1]][k-1]

2^k-th ancestor = (2^(k-1)-th ancestor)'s 2^(k-1)-th ancestor
```

### Implementation

```python
from collections import defaultdict, deque
import math

class BinaryLiftingLCA:
    def __init__(self, n, edges, root=0):
        self.n = n
        self.LOG = max(1, math.ceil(math.log2(n)))
        self.depth = [0] * n
        self.ancestor = [[-1] * self.LOG for _ in range(n)]

        # Build tree
        adj = defaultdict(list)
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)

        # BFS to calculate parent and depth
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

        # Fill ancestor table bottom-up: ancestor[node][k] = 2^k-th ancestor.
        # We derive each level from the previous: to jump 2^k steps, first
        # jump 2^(k-1) steps, then jump another 2^(k-1) from that midpoint.
        # This fills the table in O(N log N) rather than recomputing each query.
        for k in range(1, self.LOG):
            for node in range(n):
                mid = self.ancestor[node][k-1]
                if mid != -1:
                    self.ancestor[node][k] = self.ancestor[mid][k-1]

    def kth_ancestor(self, node, k):
        """Return k-th ancestor of node (or -1 if doesn't exist)"""
        # Decompose k in binary: jump over each set bit independently.
        # E.g., k=13 (1101₂) → jump 8 + 4 + 1 steps using 3 table lookups
        # instead of 13 individual parent hops.
        for i in range(self.LOG):
            if k & (1 << i):
                node = self.ancestor[node][i]
                if node == -1:
                    return -1
        return node

    def lca(self, u, v):
        # Make u the deeper node
        if self.depth[u] < self.depth[v]:
            u, v = v, u

        # Align depth: bring the deeper node up to the same level as v.
        diff = self.depth[u] - self.depth[v]
        u = self.kth_ancestor(u, diff)

        if u == v:
            return u

        # Climb together in O(log N): iterate from the largest jump downward
        # so we never overshoot the LCA.  After this loop u and v are just
        # one step below their LCA, so ancestor[u][0] is the answer.
        for k in range(self.LOG - 1, -1, -1):
            if self.ancestor[u][k] != self.ancestor[v][k]:
                u = self.ancestor[u][k]
                v = self.ancestor[v][k]

        return self.ancestor[u][0]

    def distance(self, u, v):
        """Distance between u and v"""
        lca_node = self.lca(u, v)
        return self.depth[u] + self.depth[v] - 2 * self.depth[lca_node]

# Usage example
edges = [(0,1), (0,2), (1,3), (1,4), (2,5), (3,6), (3,7)]
lca = BinaryLiftingLCA(8, edges, 0)

print(f"LCA(6,7) = {lca.lca(6, 7)}")  # 3
print(f"LCA(6,5) = {lca.lca(6, 5)}")  # 0
print(f"LCA(4,7) = {lca.lca(4, 7)}")  # 1
print(f"Distance(6,7) = {lca.distance(6, 7)}")  # 2
```

### C++ Implementation

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

### Time Complexity

- **Preprocessing**: O(N log N)
- **Query**: O(log N)
- **Space**: O(N log N)

---

## 4. Euler Tour + RMQ

### Key Idea

1. Perform Euler Tour via DFS (record visit order)
2. LCA(u, v) = Node with minimum depth in u~v range of Euler Tour
3. O(1) RMQ using Sparse Table

### Euler Tour

```
        0
       / \
      1   2
     / \
    3   4

DFS visit order (Euler Tour):
0 → 1 → 3 → 1 → 4 → 1 → 0 → 2 → 0

euler = [0, 1, 3, 1, 4, 1, 0, 2, 0]
depth = [0, 1, 2, 1, 2, 1, 0, 1, 0]
first = {0:0, 1:1, 2:7, 3:2, 4:4}  # First appearance position of each node

LCA(3, 4):
  first[3] = 2, first[4] = 4
  euler[2:5] = [3, 1, 4]
  depth[2:5] = [2, 1, 2]
  Minimum depth position = 3 → Node 1
```

### Implementation

```python
import math

class EulerTourLCA:
    def __init__(self, n, edges, root=0):
        self.n = n
        self.euler = []
        self.depth_arr = []
        self.first = [-1] * n

        # Build tree
        from collections import defaultdict
        adj = defaultdict(list)
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)

        # DFS for Euler Tour
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

        # Build Sparse Table
        self._build_sparse_table()

    def _build_sparse_table(self):
        m = len(self.euler)
        self.LOG = max(1, math.ceil(math.log2(m)))
        self.sparse = [[0] * m for _ in range(self.LOG)]

        # sparse[0][i] = index at position i (itself)
        for i in range(m):
            self.sparse[0][i] = i

        # Build in O(N log N): each entry merges two non-overlapping halves
        # of size 2^(k-1).  Queries will later reuse these precomputed ranges
        # to answer any [l,r] RMQ in O(1) via two overlapping power-of-2 windows.
        # sparse[k][i] = position with minimum depth in range [i, i+2^k)
        for k in range(1, self.LOG):
            for i in range(m - (1 << k) + 1):
                left = self.sparse[k-1][i]
                right = self.sparse[k-1][i + (1 << (k-1))]
                if self.depth_arr[left] <= self.depth_arr[right]:
                    self.sparse[k][i] = left
                else:
                    self.sparse[k][i] = right

    def _rmq(self, l, r):
        """Return position with minimum depth in range [l, r]"""
        length = r - l + 1
        k = int(math.log2(length))
        # Two overlapping windows of size 2^k together cover the full [l, r].
        # Overlap is harmless for min-queries because taking the min of a
        # duplicated element doesn't change the result — this is the key
        # property that makes sparse tables O(1) per query.
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

# Usage example
edges = [(0,1), (0,2), (1,3), (1,4)]
lca = EulerTourLCA(5, edges, 0)
print(f"LCA(3,4) = {lca.lca(3, 4)}")  # 1
print(f"LCA(3,2) = {lca.lca(3, 2)}")  # 0
```

### Time Complexity

- **Preprocessing**: O(N log N)
- **Query**: O(1)
- **Space**: O(N log N)

---

## 5. Tree Path Queries

### Path Sum Query

Sum of node values on the path from u to v

```python
class TreePathSum:
    def __init__(self, n, edges, values, root=0):
        self.lca_solver = BinaryLiftingLCA(n, edges, root)
        self.prefix = [0] * n  # Sum from root to each node

        # Calculate prefix sum via DFS
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
        """Sum of path from u to v"""
        lca_node = self.lca_solver.lca(u, v)
        # prefix[u] + prefix[v] - prefix[lca] - prefix[parent(lca)]
        # = prefix[u] + prefix[v] - 2*prefix[lca] + values[lca]
        return (self.prefix[u] + self.prefix[v]
                - 2 * self.prefix[lca_node] + self.values[lca_node])

# Usage example
edges = [(0,1), (0,2), (1,3), (1,4)]
values = [1, 2, 3, 4, 5]  # Value of each node
tree = TreePathSum(5, edges, values, 0)
print(f"Path sum(3,4) = {tree.path_sum(3, 4)}")  # 4+2+5 = 11
print(f"Path sum(3,2) = {tree.path_sum(3, 2)}")  # 4+2+1+3 = 10
```

### Path Maximum Query

Add maximum value information to Binary Lifting

```python
class TreePathMax:
    def __init__(self, n, edges, values, root=0):
        self.n = n
        self.LOG = max(1, math.ceil(math.log2(n)))
        self.depth = [0] * n
        self.ancestor = [[-1] * self.LOG for _ in range(n)]
        self.max_val = [[0] * self.LOG for _ in range(n)]  # Maximum on path
        self.values = values

        # Build tree and preprocess
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

        # Calculate 2^k ancestor and path maximum
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
        """Maximum value on path from u to v"""
        result = max(self.values[u], self.values[v])

        if self.depth[u] < self.depth[v]:
            u, v = v, u

        # Raise u to v's depth while updating maximum
        diff = self.depth[u] - self.depth[v]
        for k in range(self.LOG):
            if diff & (1 << k):
                result = max(result, self.max_val[u][k])
                u = self.ancestor[u][k]

        if u == v:
            return result

        # Climb together while updating maximum
        for k in range(self.LOG - 1, -1, -1):
            if self.ancestor[u][k] != self.ancestor[v][k]:
                result = max(result, self.max_val[u][k], self.max_val[v][k])
                u = self.ancestor[u][k]
                v = self.ancestor[v][k]

        # Finally to LCA
        result = max(result, self.max_val[u][0], self.max_val[v][0])
        return result
```

---

## 6. Heavy-Light Decomposition (HLD)

### Key Idea

Decompose tree into **Heavy paths** and **Light paths** to process path queries in O(log²N)

```
Heavy Edge: Edge to child with largest subtree size
Light Edge: Other edges

        1
       /|\
     [2] 3 4     []: Heavy edge
     /|
   [5] 6
   /
  7

Heavy path: 1-2-5-7
```

### Implementation Overview

```python
class HLD:
    def __init__(self, n, edges, root=0):
        self.n = n
        self.parent = [-1] * n
        self.depth = [0] * n
        self.subtree_size = [1] * n
        self.chain_head = [0] * n  # Head of chain
        self.chain_pos = [0] * n   # Position in chain (segment tree index)
        self.chain_arr = []        # Actual node order

        # Build tree
        from collections import defaultdict
        self.adj = defaultdict(list)
        for u, v in edges:
            self.adj[u].append(v)
            self.adj[v].append(u)

        # DFS 1: Subtree size, parent, depth
        self._dfs_size(root, -1, 0)

        # DFS 2: HLD decomposition
        self._dfs_hld(root, root)

    def _dfs_size(self, node, parent, depth):
        self.parent[node] = parent
        self.depth[node] = depth

        for i, child in enumerate(self.adj[node]):
            if child != parent:
                self._dfs_size(child, node, depth + 1)
                self.subtree_size[node] += self.subtree_size[child]

                # Move heavy child to front
                if self.subtree_size[child] > self.subtree_size[self.adj[node][0]]:
                    self.adj[node][0], self.adj[node][i] = self.adj[node][i], self.adj[node][0]

    def _dfs_hld(self, node, head):
        self.chain_head[node] = head
        self.chain_pos[node] = len(self.chain_arr)
        self.chain_arr.append(node)

        for child in self.adj[node]:
            if child != self.parent[node]:
                if child == self.adj[node][0]:
                    # Heavy child: same chain
                    self._dfs_hld(child, head)
                else:
                    # Light child: new chain
                    self._dfs_hld(child, child)

    def path_query(self, u, v, seg_tree):
        """Path query from u to v (using segment tree)"""
        result = 0  # Or appropriate identity element

        while self.chain_head[u] != self.chain_head[v]:
            # Raise deeper chain
            if self.depth[self.chain_head[u]] < self.depth[self.chain_head[v]]:
                u, v = v, u

            # Query current chain
            result = max(result, seg_tree.query(
                self.chain_pos[self.chain_head[u]],
                self.chain_pos[u]
            ))
            u = self.parent[self.chain_head[u]]

        # Query within same chain
        if self.depth[u] > self.depth[v]:
            u, v = v, u
        result = max(result, seg_tree.query(
            self.chain_pos[u],
            self.chain_pos[v]
        ))

        return result
```

### Time Complexity

- **Preprocessing**: O(N)
- **Path Query**: O(log²N)
- **Path Update**: O(log²N)

---

## 7. Practical Problem Patterns

### Pattern 1: Distance Between Two Nodes

```python
def distance(lca_solver, u, v):
    lca_node = lca_solver.lca(u, v)
    return (lca_solver.depth[u] + lca_solver.depth[v]
            - 2 * lca_solver.depth[lca_node])
```

### Pattern 2: Check if Node is on Path

```python
def is_on_path(lca_solver, u, v, x):
    """Check if x is on u-v path"""
    lca_uv = lca_solver.lca(u, v)
    lca_ux = lca_solver.lca(u, x)
    lca_vx = lca_solver.lca(v, x)

    # For x to be on path:
    # 1. LCA(u,x) = x and LCA(x,v) = LCA(u,v), or
    # 2. LCA(v,x) = x and LCA(x,u) = LCA(u,v)
    if lca_ux == x and lca_solver.lca(x, v) == lca_uv:
        return True
    if lca_vx == x and lca_solver.lca(x, u) == lca_uv:
        return True
    return False
```

### Pattern 3: Subtree Query (Euler Tour)

```python
class SubtreeQuery:
    def __init__(self, n, edges, root=0):
        self.n = n
        self.enter = [0] * n  # Subtree start index
        self.leave = [0] * n  # Subtree end index
        self.order = []        # DFS order

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
        """Range [l, r] that node's subtree occupies in order"""
        return self.enter[node], self.leave[node]

# Combine with segment tree for subtree sum/max queries
```

### Pattern 4: K-th Ancestor

```python
def kth_ancestor(lca_solver, node, k):
    """K-th ancestor using Binary Lifting"""
    for i in range(lca_solver.LOG):
        if k & (1 << i):
            node = lca_solver.ancestor[node][i]
            if node == -1:
                return -1
    return node
```

### Pattern 5: K-th Node on Path

```python
def kth_node_on_path(lca_solver, u, v, k):
    """K-th node (0-indexed) on path from u to v"""
    lca_node = lca_solver.lca(u, v)
    dist_u_lca = lca_solver.depth[u] - lca_solver.depth[lca_node]
    dist_v_lca = lca_solver.depth[v] - lca_solver.depth[lca_node]
    total = dist_u_lca + dist_v_lca

    if k > total:
        return -1

    if k <= dist_u_lca:
        # k steps from u towards LCA
        return kth_ancestor(lca_solver, u, k)
    else:
        # (k - dist_u_lca) steps from LCA towards v
        # = (total - k) steps up from v
        return kth_ancestor(lca_solver, v, total - k)
```

---

## 8. Time Complexity Summary

| Method | Preprocessing | Query | Space |
|------|--------|------|------|
| Naive | O(N) | O(N) | O(N) |
| Binary Lifting | O(N log N) | O(log N) | O(N log N) |
| Euler Tour + RMQ | O(N log N) | O(1) | O(N log N) |
| HLD + SegTree | O(N) | O(log²N) | O(N) |

---

## 9. Common Mistakes

### Mistake 1: Root Setting Error

```python
# Check if tree is 0-indexed or 1-indexed
root = 0  # or 1
```

### Mistake 2: Ancestor Array Initialization

```python
# Root's ancestor is -1 or root itself
ancestor[root][0] = -1  # or root
```

### Mistake 3: Depth Comparison Direction

```python
# Wrong: Must raise deeper node first
if depth[u] > depth[v]:  # Doesn't handle v being deeper!
    u = kth_ancestor(u, diff)

# Correct
if depth[u] < depth[v]:
    u, v = v, u
diff = depth[u] - depth[v]
u = kth_ancestor(u, diff)
```

---

## 10. Practice Problems

| Difficulty | Problem Type | Key Concept |
|--------|----------|-----------|
| ★★☆ | LCA Basic | Binary Lifting |
| ★★☆ | Node Distance | LCA + Depth |
| ★★★ | Path Sum Query | Prefix Sum + LCA |
| ★★★ | Path Maximum | Binary Lifting Extension |
| ★★★★ | Path Update | HLD + SegTree |

---

## Next Steps

- [17_Strongly_Connected_Components.md](./17_Strongly_Connected_Components.md) - SCC, Tarjan

---

## Learning Checkpoints

1. Why precompute 2^k ancestors in Binary Lifting?
2. How does Euler Tour transform LCA to RMQ?
3. What is the criterion for selecting Heavy Edge in HLD?
4. How to use LCA for path queries?
