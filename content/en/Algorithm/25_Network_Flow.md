# 25. Network Flow

## Learning Objectives
- Understanding the concept of maximum flow problems
- Implementing Ford-Fulkerson algorithm
- Implementing Edmonds-Karp algorithm
- Solving bipartite matching problems
- Utilizing the max-flow min-cut theorem

## 1. What is Network Flow?

### Definition

Think of a network flow as water flowing through a system of pipes. Each pipe has a maximum capacity, water must be conserved at junctions (what flows in must flow out), and the goal is to maximize throughput from source to sink.

A **flow network** consists of:
- **Directed graph** G = (V, E)
- **Capacity function** c(u, v): maximum flow an edge can accommodate
- **Source** s: starting point of flow
- **Sink** t: destination point of flow

```
          3
    ┌───→ B ───→┐
    │           │ 2
  S │     4     ↓     T
    │ 2 → C ─→ D │ 3
    └───────────→┘

S: Source, T: Sink
Numbers: Edge capacities
```

### Flow Conditions

1. **Capacity constraint**: 0 ≤ f(u,v) ≤ c(u,v)  *(no pipe carries more water than its diameter allows)*
2. **Flow conservation**: Except for source/sink, incoming flow = outgoing flow  *(water doesn't accumulate at junctions — every drop that enters must exit)*
3. **Skew symmetry**: f(u,v) = -f(v,u)  *(if 3 units flow from u to v, that same flow is represented as -3 from v to u, enabling residual graph arithmetic)*

### Maximum Flow Problem

Find the **maximum flow** that can be sent from source to sink

```
┌─────────────────────────────────────────────────┐
│           Network Flow Applications              │
├─────────────────────────────────────────────────┤
│  • Bipartite matching (job assignment, teams)   │
│  • Max/min cut problems                         │
│  • Project selection problems                   │
│  • Image segmentation (computer vision)         │
│  • Traffic/communication network optimization   │
└─────────────────────────────────────────────────┘
```

---

## 2. Ford-Fulkerson Algorithm

### Key Idea

1. Find **augmenting path**: path from source→sink where additional flow can be sent
2. Increase flow along the path
3. Repeat until no more augmenting paths exist

### Residual Graph

```
Original edge: u →(c)→ v, current flow f

Residual graph:
  u →(c-f)→ v  (forward: remaining capacity)
  u ←(f)← v    (backward: cancellable flow)
```

### Example Operation

```
Initial:                 Step 1: S→A→B→T path
      4                        (add flow 2)
  ┌──→A──→┐
  │   ↓   │2
S │3  1   ↓ T            Update residual graph
  │   ↓   │
  └──→B──→┘
      3

Step 2: S→B→T path       Final max flow: 5
  (add flow 3)
```

### Implementation (DFS-based)

```python
from collections import defaultdict

class MaxFlow:
    def __init__(self, n):
        self.n = n
        self.capacity = defaultdict(lambda: defaultdict(int))
        self.flow = defaultdict(lambda: defaultdict(int))

    def add_edge(self, u, v, cap):
        self.capacity[u][v] += cap
        # No explicit reverse edge needed: self.flow[v][u] starts at 0,
        # so residual capacity (cap - flow) = cap in the forward direction,
        # and flow[neighbor][source] -= result below handles the backward bookkeeping

    def dfs(self, source, sink, visited, min_cap):
        if source == sink:
            return min_cap  # Reached sink — return the bottleneck capacity along this path

        visited.add(source)

        for neighbor in self.capacity[source]:
            # Residual capacity = original capacity minus what's already flowing
            residual = self.capacity[source][neighbor] - self.flow[source][neighbor]

            if neighbor not in visited and residual > 0:
                # Track the tightest bottleneck seen so far along this DFS path
                new_cap = min(min_cap, residual)
                result = self.dfs(neighbor, sink, visited, new_cap)

                if result > 0:
                    self.flow[source][neighbor] += result
                    # Backward edge: decreasing flow[neighbor][source] means we
                    # are "crediting" capacity back, allowing future paths to
                    # reroute flow that was sent along a suboptimal route
                    self.flow[neighbor][source] -= result
                    return result

        return 0

    def max_flow(self, source, sink):
        total_flow = 0

        # Keep finding augmenting paths until none remain —
        # each iteration pushes at least 1 unit of flow, so the loop terminates
        while True:
            visited = set()
            path_flow = self.dfs(source, sink, visited, float('inf'))

            if path_flow == 0:
                break  # No augmenting path found — residual graph is exhausted

            total_flow += path_flow

        return total_flow

# Usage example
mf = MaxFlow(4)
# 0: S, 1: A, 2: B, 3: T
mf.add_edge(0, 1, 4)  # S→A
mf.add_edge(0, 2, 3)  # S→B
mf.add_edge(1, 2, 1)  # A→B
mf.add_edge(1, 3, 2)  # A→T
mf.add_edge(2, 3, 3)  # B→T

print("Max flow:", mf.max_flow(0, 3))  # 5
```

### Time Complexity

- **O(E × max_flow)**: DFS-based (for integer capacities)
- May not converge for irrational capacities

---

## 3. Edmonds-Karp Algorithm

### Key Idea

Ford-Fulkerson + **BFS** to select shortest augmenting path

→ Always guarantees **O(VE²)**

### Implementation

```python
from collections import defaultdict, deque

class EdmondsKarp:
    def __init__(self, n):
        self.n = n
        self.capacity = [[0] * n for _ in range(n)]
        self.graph = defaultdict(list)

    def add_edge(self, u, v, cap):
        self.capacity[u][v] += cap
        self.graph[u].append(v)
        self.graph[v].append(u)  # Add reverse direction too — needed for residual graph traversal

    def bfs(self, source, sink, parent):
        # BFS finds the shortest augmenting path — Edmonds-Karp guarantees O(VE²)
        # by always choosing the path with fewest edges, preventing the pathological
        # O(E × max_flow) behavior of DFS-based Ford-Fulkerson
        visited = [False] * self.n
        visited[source] = True
        queue = deque([source])

        while queue:
            node = queue.popleft()

            for neighbor in self.graph[node]:
                # Only traverse edges with remaining residual capacity
                if not visited[neighbor] and self.capacity[node][neighbor] > 0:
                    visited[neighbor] = True
                    parent[neighbor] = node
                    if neighbor == sink:
                        return True  # Path found — stop BFS early
                    queue.append(neighbor)

        return False  # No path to sink — max flow reached

    def max_flow(self, source, sink):
        parent = [-1] * self.n
        total_flow = 0

        # Each BFS call finds one augmenting path and saturates at least one edge,
        # so there are at most O(VE) augmenting path iterations before termination
        while self.bfs(source, sink, parent):
            # Find minimum residual capacity along the path (the bottleneck edge)
            path_flow = float('inf')
            node = sink
            while node != source:
                prev = parent[node]
                path_flow = min(path_flow, self.capacity[prev][node])
                node = prev

            # Update capacities: forward edges decrease, backward edges increase.
            # Increasing capacity[node][prev] creates a "cancel" option for
            # future paths — this is the key mechanism that lets the algorithm
            # undo a previous suboptimal routing decision
            node = sink
            while node != source:
                prev = parent[node]
                self.capacity[prev][node] -= path_flow
                self.capacity[node][prev] += path_flow
                node = prev

            total_flow += path_flow
            parent = [-1] * self.n  # Reset parent array for next BFS

        return total_flow

# Usage example
ek = EdmondsKarp(6)
# More complex example
edges = [
    (0, 1, 16), (0, 2, 13),
    (1, 2, 10), (1, 3, 12),
    (2, 1, 4), (2, 4, 14),
    (3, 2, 9), (3, 5, 20),
    (4, 3, 7), (4, 5, 4)
]
for u, v, cap in edges:
    ek.add_edge(u, v, cap)

print("Max flow:", ek.max_flow(0, 5))  # 23
```

### C++ Implementation

```cpp
#include <bits/stdc++.h>
using namespace std;

class EdmondsKarp {
private:
    int n;
    vector<vector<int>> capacity, adj;

public:
    EdmondsKarp(int n) : n(n), capacity(n, vector<int>(n, 0)), adj(n) {}

    void addEdge(int u, int v, int cap) {
        capacity[u][v] += cap;
        adj[u].push_back(v);
        adj[v].push_back(u);
    }

    int maxFlow(int source, int sink) {
        int totalFlow = 0;

        while (true) {
            vector<int> parent(n, -1);
            queue<int> q;
            q.push(source);
            parent[source] = source;

            while (!q.empty() && parent[sink] == -1) {
                int node = q.front(); q.pop();
                for (int next : adj[node]) {
                    if (parent[next] == -1 && capacity[node][next] > 0) {
                        parent[next] = node;
                        q.push(next);
                    }
                }
            }

            if (parent[sink] == -1) break;

            int pathFlow = INT_MAX;
            for (int v = sink; v != source; v = parent[v]) {
                pathFlow = min(pathFlow, capacity[parent[v]][v]);
            }

            for (int v = sink; v != source; v = parent[v]) {
                capacity[parent[v]][v] -= pathFlow;
                capacity[v][parent[v]] += pathFlow;
            }

            totalFlow += pathFlow;
        }

        return totalFlow;
    }
};
```

---

## 4. Max-Flow Min-Cut Theorem

### Theorem

**Maximum flow = Capacity of minimum cut**

Intuitively, think of a cut as a bottleneck — a set of edges whose removal disconnects all source-to-sink paths. The minimum cut is the tightest bottleneck: no flow can exceed its total capacity, because every unit of flow must cross the cut. Conversely, the max-flow algorithm saturates this bottleneck exactly — when the algorithm terminates, the saturated edges across the residual graph boundary form the minimum cut. Backward edges in the residual graph are essential to this process: they allow the algorithm to reroute flow that was sent along a suboptimal path, effectively "undoing" earlier decisions so that the final flow configuration crosses the minimum cut at full capacity.

### What is a Cut?

A set of edges that partitions the graph into source side (S) and sink side (T)

```
         Cut
          │
    ┌─────│─────┐
S ──┤  A  │  B  ├── T
    └─────│─────┘
          │
Cut capacity = Sum of capacities of edges from S-side to T-side
```

### Finding Minimum Cut

```python
def find_min_cut(self, source, sink):
    """
    Returns edges forming the minimum cut after computing max flow
    """
    # First compute max flow — this saturates all bottleneck edges across the min cut
    max_flow_value = self.max_flow(source, sink)

    # After max flow, BFS the residual graph from source.
    # Nodes reachable from source form the "S-side" of the min cut.
    # The sink is unreachable because all augmenting paths are exhausted.
    visited = [False] * self.n
    queue = deque([source])
    visited[source] = True

    while queue:
        node = queue.popleft()
        for neighbor in range(self.n):
            # Only traverse edges with remaining residual capacity
            if not visited[neighbor] and self.capacity[node][neighbor] > 0:
                visited[neighbor] = True
                queue.append(neighbor)

    # Min cut edges cross from the reachable (S) side to the unreachable (T) side.
    # We check original_capacity (not current residual capacity) to identify edges
    # that existed in the original graph — some may now have 0 residual because
    # they are fully saturated and form the min cut boundary.
    # Without tracking original_capacity, we cannot distinguish between edges that
    # were never in the graph (capacity 0) and edges that were saturated by flow.
    min_cut_edges = []
    for u in range(self.n):
        if visited[u]:
            for v in range(self.n):
                if not visited[v] and self.original_capacity[u][v] > 0:
                    min_cut_edges.append((u, v))

    return max_flow_value, min_cut_edges
```

---

## 5. Bipartite Matching

### Problem

Find maximum matching in a bipartite graph

```
Left group        Right group
   A ─────────── 1
   B ─────────── 2
   C ─────────── 3

Max matching: A-1, B-2, C-3
```

### Convert to Flow Network

```
              1
           ┌──┤
     ┌──A──┤  │
     │     └──┤
S ───┼──B─────┼─── T
     │     ┌──┤
     └──C──┤  │
           └──┤
              1

All edge capacities = 1
Max flow = Max matching
```

### Implementation

```python
class BipartiteMatching:
    def __init__(self, left_size, right_size):
        self.left = left_size
        self.right = right_size
        self.adj = [[] for _ in range(left_size)]

    def add_edge(self, left_node, right_node):
        self.adj[left_node].append(right_node)

    def dfs(self, node, visited, match_right):
        for right in self.adj[node]:
            if visited[right]:
                continue
            # Mark visited to prevent this DFS from revisiting the same right node,
            # which would cause infinite recursion in the augmenting path search
            visited[right] = True

            # Right node is unmatched, or its current left partner can find another match.
            # This recursive call is the augmenting path step: it tries to "bump"
            # the existing match partner to a different right node, freeing this slot.
            if match_right[right] == -1 or self.dfs(match_right[right], visited, match_right):
                match_right[right] = node  # Reassign (or assign) this right node to current left node
                return True

        return False  # No augmenting path found from this left node

    def max_matching(self):
        match_right = [-1] * self.right  # match_right[r] = which left node is matched to right node r
        result = 0

        for left_node in range(self.left):
            # Fresh visited array per left node — ensures DFS can explore all alternative
            # augmenting paths without being blocked by earlier iterations
            visited = [False] * self.right
            if self.dfs(left_node, visited, match_right):
                result += 1

        return result, match_right

# Usage example: Student-Project assignment
bm = BipartiteMatching(3, 3)
# Student 0: can do projects 0, 1
bm.add_edge(0, 0)
bm.add_edge(0, 1)
# Student 1: can do projects 0, 2
bm.add_edge(1, 0)
bm.add_edge(1, 2)
# Student 2: can do projects 1, 2
bm.add_edge(2, 1)
bm.add_edge(2, 2)

count, matching = bm.max_matching()
print(f"Max matching: {count}")
for right, left in enumerate(matching):
    if left != -1:
        print(f"  Student {left} → Project {right}")
```

### Hungarian Algorithm (Kuhn's Algorithm)

The above DFS-based algorithm is a simplified version of the Hungarian algorithm.

**Time Complexity**: O(V × E)

### C++ Implementation

```cpp
#include <bits/stdc++.h>
using namespace std;

class BipartiteMatching {
private:
    int left_size, right_size;
    vector<vector<int>> adj;
    vector<int> matchRight;
    vector<bool> visited;

    bool dfs(int node) {
        for (int right : adj[node]) {
            if (visited[right]) continue;
            visited[right] = true;

            if (matchRight[right] == -1 || dfs(matchRight[right])) {
                matchRight[right] = node;
                return true;
            }
        }
        return false;
    }

public:
    BipartiteMatching(int l, int r)
        : left_size(l), right_size(r), adj(l), matchRight(r, -1) {}

    void addEdge(int left, int right) {
        adj[left].push_back(right);
    }

    int maxMatching() {
        int result = 0;
        for (int i = 0; i < left_size; i++) {
            visited.assign(right_size, false);
            if (dfs(i)) result++;
        }
        return result;
    }
};
```

---

## 6. Practical Problem Patterns

### Pattern 1: Job Assignment Problem

```python
def job_assignment(workers, jobs, can_do):
    """
    workers: number of workers
    jobs: number of jobs
    can_do[i]: list of jobs worker i can do
    """
    bm = BipartiteMatching(workers, jobs)
    for worker, job_list in enumerate(can_do):
        for job in job_list:
            bm.add_edge(worker, job)

    return bm.max_matching()
```

### Pattern 2: Vertex Splitting

When nodes have capacities, split each node into two

```python
def vertex_capacity_flow(n, edges, vertex_cap, source, sink):
    """
    Node i → i_in (2*i), i_out (2*i + 1)
    i_in → i_out capacity = vertex_cap[i]
    """
    new_n = 2 * n
    mf = EdmondsKarp(new_n)

    # Internal node edges
    for i in range(n):
        mf.add_edge(2*i, 2*i + 1, vertex_cap[i])

    # Original edges: u_out → v_in
    for u, v, cap in edges:
        mf.add_edge(2*u + 1, 2*v, cap)

    return mf.max_flow(2*source + 1, 2*sink)
```

### Pattern 3: Multiple Source/Sink

```python
def multi_source_sink(n, edges, sources, sinks, source_cap, sink_cap):
    """
    Add virtual super source and super sink
    """
    super_source = n
    super_sink = n + 1
    mf = EdmondsKarp(n + 2)

    # Original edges
    for u, v, cap in edges:
        mf.add_edge(u, v, cap)

    # Super source → each source
    for s, cap in zip(sources, source_cap):
        mf.add_edge(super_source, s, cap)

    # Each sink → super sink
    for t, cap in zip(sinks, sink_cap):
        mf.add_edge(t, super_sink, cap)

    return mf.max_flow(super_source, super_sink)
```

### Pattern 4: Path Separation (Edge-disjoint paths)

```python
def count_edge_disjoint_paths(n, edges, source, sink):
    """
    Maximum number of paths with no shared edges
    = Max flow with all edge capacities = 1
    """
    mf = EdmondsKarp(n)
    for u, v in edges:
        mf.add_edge(u, v, 1)
    return mf.max_flow(source, sink)

def count_vertex_disjoint_paths(n, edges, source, sink):
    """
    Maximum number of paths with no shared vertices
    = Vertex splitting + all edge capacities = 1
    """
    # Vertex splitting: node i → 2*i (in), 2*i+1 (out)
    mf = EdmondsKarp(2 * n)

    # Internal edges (except source/sink)
    for i in range(n):
        if i == source or i == sink:
            mf.add_edge(2*i, 2*i + 1, float('inf'))
        else:
            mf.add_edge(2*i, 2*i + 1, 1)

    # Original edges
    for u, v in edges:
        mf.add_edge(2*u + 1, 2*v, 1)

    return mf.max_flow(2*source + 1, 2*sink)
```

### Pattern 5: Minimum Path Cover

Minimum number of paths covering all vertices in a DAG

```python
def min_path_cover(n, edges):
    """
    Minimum path cover = n - maximum matching
    """
    bm = BipartiteMatching(n, n)
    for u, v in edges:
        bm.add_edge(u, v)

    max_match, _ = bm.max_matching()
    return n - max_match
```

---

## 7. Dinic's Algorithm (Advanced)

Faster max flow algorithm: **O(V²E)**

### Key Ideas

1. Build level graph with BFS
2. Find blocking flow with DFS
3. Repeat

```python
from collections import deque

class Dinic:
    def __init__(self, n):
        self.n = n
        self.graph = [[] for _ in range(n)]

    def add_edge(self, u, v, cap):
        # Each forward edge stores a reference index to its backward edge and vice versa.
        # This paired-index trick allows O(1) lookup of the reverse edge during flow updates,
        # avoiding the need to search the adjacency list for the corresponding backward edge.
        self.graph[u].append([v, cap, len(self.graph[v])])
        self.graph[v].append([u, 0, len(self.graph[u]) - 1])  # Reverse edge with capacity 0

    def bfs(self, source, sink):
        # BFS assigns each node a "level" (distance from source in the residual graph).
        # This level graph restricts DFS to only advance toward the sink, preventing cycles
        # and ensuring each blocking flow phase makes progress.
        self.level = [-1] * self.n
        self.level[source] = 0
        queue = deque([source])

        while queue:
            node = queue.popleft()
            for next_node, cap, _ in self.graph[node]:
                if cap > 0 and self.level[next_node] == -1:
                    self.level[next_node] = self.level[node] + 1
                    queue.append(next_node)

        return self.level[sink] != -1  # False means no path exists — max flow reached

    def dfs(self, node, sink, flow):
        if node == sink:
            return flow  # Reached sink — return the bottleneck capacity for this path

        while self.iter[node] < len(self.graph[node]):
            edge = self.graph[node][self.iter[node]]
            next_node, cap, rev = edge

            # Only advance along edges that go exactly one level deeper (level graph constraint)
            if cap > 0 and self.level[next_node] == self.level[node] + 1:
                d = self.dfs(next_node, sink, min(flow, cap))
                if d > 0:
                    edge[1] -= d                        # Decrease forward edge capacity
                    self.graph[next_node][rev][1] += d  # Increase reverse edge capacity (residual)
                    return d

            self.iter[node] += 1  # This edge is exhausted — advance to next neighbor

        return 0  # Dead end in level graph; backtrack

    def max_flow(self, source, sink):
        flow = 0

        # Outer loop: rebuild the level graph with BFS after each blocking flow phase.
        # Each BFS phase increases the shortest augmenting path length by at least 1,
        # so there are at most O(V) BFS phases, giving the O(V²E) total complexity.
        while self.bfs(source, sink):
            self.iter = [0] * self.n  # iter[node] tracks the next edge to try — avoids revisiting
            while True:
                f = self.dfs(source, sink, float('inf'))
                if f == 0:
                    break  # Blocking flow complete — rebuild level graph
                flow += f

        return flow
```

---

## 8. Time Complexity Summary

| Algorithm | Time Complexity | Notes |
|-----------|----------------|-------|
| Ford-Fulkerson (DFS) | O(E × max_flow) | Integer capacities |
| Edmonds-Karp (BFS) | O(VE²) | Always guaranteed |
| Dinic | O(V²E) | General graphs |
| Dinic (bipartite) | O(E√V) | Favorable for bipartite matching |
| Kuhn's (bipartite matching) | O(VE) | Simple direct implementation |

---

## 9. Common Mistakes

### Mistake 1: Missing Reverse Edges

```python
# Incorrect
def add_edge(self, u, v, cap):
    self.capacity[u][v] = cap  # No reverse!

# Correct
def add_edge(self, u, v, cap):
    self.capacity[u][v] += cap
    # Reverse direction handled automatically (negative flow possible)
```

### Mistake 2: Traversing Zero-Capacity Edges in BFS

```python
# Incorrect
if not visited[neighbor]:  # No capacity check!

# Correct
if not visited[neighbor] and self.capacity[node][neighbor] > 0:
```

### Mistake 3: Handling Undirected Edges

```python
# For undirected edge (u-v), add both directions
add_edge(u, v, cap)
add_edge(v, u, cap)  # Add reverse too
```

---

## 10. Practice Problems

| Difficulty | Problem Type | Key Concept |
|-----------|--------------|-------------|
| ★★☆ | Basic bipartite matching | Kuhn's algorithm |
| ★★★ | Basic max flow | Edmonds-Karp |
| ★★★ | Path separation | Edge/vertex splitting |
| ★★★★ | Minimum cut | Max-Flow Min-Cut |
| ★★★★ | Project selection | Flow modeling |

---

## Next Steps

- [26_Computational_Geometry.md](./26_Computational_Geometry.md) - Computational geometry algorithms

---

## Learning Checklist

1. What is a residual graph in Ford-Fulkerson?
2. Why does max flow equal min cut?
3. How to convert bipartite matching to flow network?
4. Why is Edmonds-Karp better than Ford-Fulkerson?
