# 24. Strongly Connected Components (SCC)

## Learning Objectives
- Understand the concept of Strongly Connected Components (SCC)
- Implement Tarjan's algorithm
- Implement Kosaraju's algorithm
- Solve 2-SAT problems
- Utilize condensation graphs (DAG)

## 1. What are Strongly Connected Components?

### Definition

In a **directed graph**, a **Strongly Connected Component (SCC)** is a **maximal** subgraph where for every pair of vertices (u, v), there exist paths both u→v and v→u.

```
    ┌───────────┐
    │  SCC 1    │
    │  ┌→1→─┐   │
    │  │    ↓   │         ┌───────┐
    │  4←──2    │────────→│ SCC 2 │
    │      ↓    │         │  5→6  │
    │      3←───│         │  ↑ ↓  │
    └───────────┘         │  8←7  │
                          └───────┘
```

### Properties

1. **Maximality**: Adding a vertex to an SCC breaks the strong connectivity
2. **Partition**: Every vertex belongs to exactly one SCC
3. **DAG Structure**: Viewing SCCs as single nodes forms a DAG (Directed Acyclic Graph)

### Applications of SCC

```
┌─────────────────────────────────────────────────┐
│              SCC Applications                    │
├─────────────────────────────────────────────────┤
│  • 2-SAT problem solving                        │
│  • Deadlock detection                           │
│  • Social network analysis (community detection)│
│  • Web page clustering                          │
│  • Compiler optimization (circular dependency)  │
└─────────────────────────────────────────────────┘
```

---

## 2. Kosaraju's Algorithm

### Core Idea

1. DFS on original graph, record finish order
2. DFS on reversed graph in reverse finish order
3. Each DFS tree is one SCC

### How It Works

```
Step 1: DFS on original graph (record finish order)
        1 → 2 → 3
        ↑       ↓
        └───────┘

        Finish order: [3, 2, 1]

Step 2: Reverse the graph
        1 ← 2 ← 3
        ↓       ↑
        └───────┘

Step 3: DFS in reverse order (starting from 1)
        1 → 3 → 2 → (back to 1)

        SCC: {1, 2, 3}
```

### Implementation

```python
from collections import defaultdict

class KosarajuSCC:
    def __init__(self, n):
        self.n = n
        self.graph = defaultdict(list)
        self.reverse_graph = defaultdict(list)

    def add_edge(self, u, v):
        self.graph[u].append(v)
        self.reverse_graph[v].append(u)

    def find_sccs(self):
        # Step 1: DFS on original graph, record finish order.
        # A node finishes (is appended) only after all reachable nodes are done,
        # so nodes in a "sink" SCC (no outgoing edges to other SCCs) finish first.
        # Reversing this order means we start step 2 from "source" SCCs —
        # nodes that can reach everything but cannot be reached from outside.
        visited = [False] * self.n
        finish_order = []

        def dfs1(node):
            visited[node] = True
            for neighbor in self.graph[node]:
                if not visited[neighbor]:
                    dfs1(neighbor)
            finish_order.append(node)  # Append AFTER all descendants are visited

        for i in range(self.n):
            if not visited[i]:
                dfs1(i)

        # Step 2: DFS on the REVERSED graph in reverse finish order.
        # Reversing all edges means "reachable from u in reversed graph" equals
        # "can reach u in the original graph."  Starting from the highest-finish
        # node (a source SCC root), a DFS on the reversed graph is trapped within
        # the same SCC — it cannot escape into another SCC because those edges
        # were reversed.  Each DFS tree here is therefore exactly one SCC.
        visited = [False] * self.n
        sccs = []

        def dfs2(node, component):
            visited[node] = True
            component.append(node)
            for neighbor in self.reverse_graph[node]:
                if not visited[neighbor]:
                    dfs2(neighbor, component)

        # Process in reverse finish order
        for node in reversed(finish_order):
            if not visited[node]:
                component = []
                dfs2(node, component)
                sccs.append(component)

        return sccs

# Usage example
scc = KosarajuSCC(8)
edges = [(0,1), (1,2), (2,0), (2,3), (3,4), (4,5), (5,6), (6,4), (6,7)]
for u, v in edges:
    scc.add_edge(u, v)

result = scc.find_sccs()
print("SCCs:", result)
# SCCs: [[7], [4, 6, 5], [3], [0, 2, 1]]
```

### C++ Implementation

```cpp
#include <bits/stdc++.h>
using namespace std;

class KosarajuSCC {
private:
    int n;
    vector<vector<int>> graph, reverseGraph;
    vector<bool> visited;
    vector<int> finishOrder;

    void dfs1(int node) {
        visited[node] = true;
        for (int next : graph[node]) {
            if (!visited[next]) dfs1(next);
        }
        finishOrder.push_back(node);
    }

    void dfs2(int node, vector<int>& component) {
        visited[node] = true;
        component.push_back(node);
        for (int next : reverseGraph[node]) {
            if (!visited[next]) dfs2(next, component);
        }
    }

public:
    KosarajuSCC(int n) : n(n), graph(n), reverseGraph(n) {}

    void addEdge(int u, int v) {
        graph[u].push_back(v);
        reverseGraph[v].push_back(u);
    }

    vector<vector<int>> findSCCs() {
        // Step 1
        visited.assign(n, false);
        for (int i = 0; i < n; i++) {
            if (!visited[i]) dfs1(i);
        }

        // Step 2
        visited.assign(n, false);
        vector<vector<int>> sccs;

        for (int i = n - 1; i >= 0; i--) {
            int node = finishOrder[i];
            if (!visited[node]) {
                vector<int> component;
                dfs2(node, component);
                sccs.push_back(component);
            }
        }

        return sccs;
    }
};
```

### Time Complexity

- **Time**: O(V + E) - Two DFS passes
- **Space**: O(V + E) - Storing reversed graph

---

## 3. Tarjan's Algorithm

### Core Idea

Find SCCs in a single DFS using **discovery time** and **low-link values**.

- **Discovery time (disc)**: When a node was first visited
- **Low-link (low)**: Smallest discovery time reachable from this node

### How It Works

```
When visiting a node:
1. disc[node] = low[node] = current time
2. Push onto stack
3. Explore neighbors:
   - Unvisited: Recurse, then low[node] = min(low[node], low[neighbor])
   - On stack: low[node] = min(low[node], disc[neighbor])
4. If disc[node] == low[node], it's an SCC root
   → Pop stack until node to form SCC
```

### Visualization

```
DFS progress:
Nodes:  1 → 2 → 3 → 1(back edge)
disc: [1,  2,  3]
low:  [1,  1,  1]  ← Updated by back edge from 3 to 1

At node 1: disc[1] == low[1] = 1
→ SCC found: {1, 2, 3}
```

### Implementation

```python
class TarjanSCC:
    def __init__(self, n):
        self.n = n
        self.graph = [[] for _ in range(n)]
        self.disc = [-1] * n      # Discovery time
        self.low = [-1] * n       # Low-link value
        self.on_stack = [False] * n
        self.stack = []
        self.time = 0
        self.sccs = []

    def add_edge(self, u, v):
        self.graph[u].append(v)

    def dfs(self, node):
        self.disc[node] = self.low[node] = self.time
        self.time += 1
        self.stack.append(node)
        self.on_stack[node] = True  # Track stack membership separately from "visited"

        for neighbor in self.graph[node]:
            if self.disc[neighbor] == -1:
                # Tree edge: recurse, then propagate the lowest reachable disc time
                # back up — this is how a back-edge deep in the subtree influences
                # the low value of all ancestors on the path.
                self.dfs(neighbor)
                self.low[node] = min(self.low[node], self.low[neighbor])
            elif self.on_stack[neighbor]:
                # Back edge to an ancestor still on the current DFS stack:
                # this node can "escape" to that ancestor, so update low.
                # We check on_stack (not just visited) because a neighbor that
                # has already been popped into a completed SCC is in a different
                # component — following that cross-edge would incorrectly merge
                # two separate SCCs.
                self.low[node] = min(self.low[node], self.disc[neighbor])

        # disc[node] == low[node] means nothing below this node has a back edge
        # reaching above it — it is the topmost (oldest) node of its SCC.
        # Pop everything down to and including this node to form the SCC.
        if self.disc[node] == self.low[node]:
            component = []
            while True:
                top = self.stack.pop()
                self.on_stack[top] = False
                component.append(top)
                if top == node:
                    break
            self.sccs.append(component)

    def find_sccs(self):
        for i in range(self.n):
            if self.disc[i] == -1:
                self.dfs(i)
        return self.sccs

# Usage example
tarjan = TarjanSCC(8)
edges = [(0,1), (1,2), (2,0), (2,3), (3,4), (4,5), (5,6), (6,4), (6,7)]
for u, v in edges:
    tarjan.add_edge(u, v)

result = tarjan.find_sccs()
print("SCCs:", result)
# SCCs: [[2, 1, 0], [6, 5, 4], [3], [7]]
```

### C++ Implementation

```cpp
#include <bits/stdc++.h>
using namespace std;

class TarjanSCC {
private:
    int n, timer = 0;
    vector<vector<int>> graph;
    vector<int> disc, low;
    vector<bool> onStack;
    stack<int> st;
    vector<vector<int>> sccs;

    void dfs(int node) {
        disc[node] = low[node] = timer++;
        st.push(node);
        onStack[node] = true;

        for (int next : graph[node]) {
            if (disc[next] == -1) {
                dfs(next);
                low[node] = min(low[node], low[next]);
            } else if (onStack[next]) {
                low[node] = min(low[node], disc[next]);
            }
        }

        if (disc[node] == low[node]) {
            vector<int> component;
            while (true) {
                int top = st.top(); st.pop();
                onStack[top] = false;
                component.push_back(top);
                if (top == node) break;
            }
            sccs.push_back(component);
        }
    }

public:
    TarjanSCC(int n) : n(n), graph(n), disc(n, -1), low(n, -1), onStack(n, false) {}

    void addEdge(int u, int v) {
        graph[u].push_back(v);
    }

    vector<vector<int>> findSCCs() {
        for (int i = 0; i < n; i++) {
            if (disc[i] == -1) dfs(i);
        }
        return sccs;
    }
};
```

### Kosaraju vs Tarjan Comparison

| Feature | Kosaraju | Tarjan |
|------|----------|--------|
| DFS passes | 2 | 1 |
| Extra space | Reversed graph O(V+E) | Stack O(V) |
| Implementation | Intuitive | Slightly complex |
| Online processing | Impossible | Possible |

---

## 4. Condensation Graph

### Concept

**Compressing** each SCC into a single node creates a **DAG**.

```
Original graph:                Condensation graph (DAG):
    1 ←→ 2                         SCC0
    ↓     ↓                          ↓
    4 ←→ 3 → 5 → 6                 SCC1 → SCC2
              ↑   ↓
              8 ← 7

SCC0 = {1,2,3,4}
SCC1 = {5,6,7,8}  → One cycle
SCC2 = ... (if exists)
```

### Implementation

```python
def build_condensation_graph(n, edges):
    # 1. Find SCCs
    tarjan = TarjanSCC(n)
    for u, v in edges:
        tarjan.add_edge(u, v)
    sccs = tarjan.find_sccs()

    # 2. Map each node to its SCC ID
    scc_id = [-1] * n
    for i, component in enumerate(sccs):
        for node in component:
            scc_id[node] = i

    # 3. Build condensation graph
    num_sccs = len(sccs)
    condensed = [set() for _ in range(num_sccs)]

    for u, v in edges:
        su, sv = scc_id[u], scc_id[v]
        if su != sv:
            condensed[su].add(sv)

    # Convert set to list
    condensed = [list(neighbors) for neighbors in condensed]

    return sccs, scc_id, condensed

# Usage example
edges = [(0,1), (1,2), (2,0), (2,3), (3,4), (4,5), (5,6), (6,4)]
sccs, scc_id, dag = build_condensation_graph(7, edges)

print("SCCs:", sccs)
print("DAG edges:")
for i, neighbors in enumerate(dag):
    for j in neighbors:
        print(f"  SCC{i} → SCC{j}")
```

### Applications of Condensation Graph

```python
def count_reachable_nodes(n, edges):
    """
    Count reachable nodes from each node
    Efficiently compute using DAG DP on condensation graph
    """
    sccs, scc_id, dag = build_condensation_graph(n, edges)
    num_sccs = len(sccs)

    # Process in topological order
    in_degree = [0] * num_sccs
    for u in range(num_sccs):
        for v in dag[u]:
            in_degree[v] += 1

    # Set of reachable nodes from each SCC
    reachable = [set(sccs[i]) for i in range(num_sccs)]

    from collections import deque
    queue = deque([i for i in range(num_sccs) if in_degree[i] == 0])
    order = []

    while queue:
        node = queue.popleft()
        order.append(node)
        for next_node in dag[node]:
            in_degree[next_node] -= 1
            if in_degree[next_node] == 0:
                queue.append(next_node)

    # Propagate reachable set in reverse order
    for scc in reversed(order):
        for next_scc in dag[scc]:
            reachable[scc] |= reachable[next_scc]

    # Count reachable nodes for each original node
    result = [0] * n
    for i in range(n):
        result[i] = len(reachable[scc_id[i]])

    return result
```

---

## 5. 2-SAT Problem

### Concept

2-SAT determines satisfiability of Boolean formulas where each clause has **exactly 2 literals**.

```
Example: (x₁ ∨ x₂) ∧ (¬x₁ ∨ x₃) ∧ (¬x₂ ∨ ¬x₃)

Literals: x₁, ¬x₁, x₂, ¬x₂, x₃, ¬x₃
Clauses: (a ∨ b) form
```

### Graph Transformation

**Key**: (a ∨ b) = (¬a → b) ∧ (¬b → a)

```
Clause (x₁ ∨ x₂):
  ¬x₁ → x₂  (if x₁ is false, x₂ must be true)
  ¬x₂ → x₁  (if x₂ is false, x₁ must be true)

For variable xi:
  Node 2i: xi
  Node 2i+1: ¬xi
```

### Satisfiability Test

**Theorem**: A 2-SAT formula is satisfiable ⟺ For no variable x do x and ¬x belong to the same SCC

### Implementation

```python
class TwoSAT:
    def __init__(self, n):
        """n variables (0 ~ n-1)"""
        self.n = n
        self.graph = [[] for _ in range(2 * n)]

    def add_clause(self, a, neg_a, b, neg_b):
        """
        Add clause (a ∨ b)
        a, b: variable indices (0 ~ n-1)
        neg_a, neg_b: True if negated
        """
        # Literal → node number
        def to_node(var, negated):
            return 2 * var + (1 if negated else 0)

        node_a = to_node(a, neg_a)
        node_b = to_node(b, neg_b)
        not_a = to_node(a, not neg_a)
        not_b = to_node(b, not neg_b)

        # ¬a → b, ¬b → a
        self.graph[not_a].append(node_b)
        self.graph[not_b].append(node_a)

    def add_or(self, a, b):
        """(a ∨ b) - at least one is true"""
        self.add_clause(a, False, b, False)

    def add_implies(self, a, b):
        """a → b (if a then b)"""
        # a → b = ¬a ∨ b
        self.add_clause(a, True, b, False)

    def add_xor(self, a, b):
        """a XOR b (exactly one is true)"""
        # (a ∨ b) ∧ (¬a ∨ ¬b)
        self.add_clause(a, False, b, False)
        self.add_clause(a, True, b, True)

    def add_equal(self, a, b):
        """a = b (both have same value)"""
        # (a → b) ∧ (b → a)
        self.add_implies(a, b)
        self.add_implies(b, a)

    def set_true(self, a):
        """Force variable a to be true"""
        # a ∨ a
        self.add_clause(a, False, a, False)

    def set_false(self, a):
        """Force variable a to be false"""
        # ¬a ∨ ¬a
        self.add_clause(a, True, a, True)

    def solve(self):
        """
        Return variable assignments if satisfiable, None otherwise
        """
        # Find SCCs using Tarjan's algorithm
        n = 2 * self.n
        disc = [-1] * n
        low = [-1] * n
        on_stack = [False] * n
        stack = []
        timer = [0]
        scc_id = [-1] * n
        scc_count = [0]

        def dfs(node):
            disc[node] = low[node] = timer[0]
            timer[0] += 1
            stack.append(node)
            on_stack[node] = True

            for neighbor in self.graph[node]:
                if disc[neighbor] == -1:
                    dfs(neighbor)
                    low[node] = min(low[node], low[neighbor])
                elif on_stack[neighbor]:
                    low[node] = min(low[node], disc[neighbor])

            if disc[node] == low[node]:
                while True:
                    top = stack.pop()
                    on_stack[top] = False
                    scc_id[top] = scc_count[0]
                    if top == node:
                        break
                scc_count[0] += 1

        for i in range(n):
            if disc[i] == -1:
                dfs(i)

        # Check satisfiability
        for i in range(self.n):
            if scc_id[2 * i] == scc_id[2 * i + 1]:
                return None  # x and ¬x in same SCC

        # Determine values: higher SCC ID is true
        # (Tarjan assigns SCC IDs in reverse topological order)
        result = [False] * self.n
        for i in range(self.n):
            # x's SCC ID > ¬x's SCC ID means x = True
            result[i] = scc_id[2 * i] > scc_id[2 * i + 1]

        return result

# Usage example
sat = TwoSAT(3)  # Variables: x0, x1, x2

# (x0 ∨ x1) ∧ (¬x0 ∨ x2) ∧ (¬x1 ∨ ¬x2)
sat.add_or(0, 1)                    # x0 ∨ x1
sat.add_clause(0, True, 2, False)   # ¬x0 ∨ x2
sat.add_clause(1, True, 2, True)    # ¬x1 ∨ ¬x2

result = sat.solve()
if result:
    print("Satisfiable:", result)
    # Verify
    x0, x1, x2 = result
    print(f"x0={x0}, x1={x1}, x2={x2}")
    clause1 = x0 or x1
    clause2 = (not x0) or x2
    clause3 = (not x1) or (not x2)
    print(f"Verification: {clause1 and clause2 and clause3}")
else:
    print("Unsatisfiable")
```

### 2-SAT Application Problems

```python
def team_assignment(n, conflicts):
    """
    Assign n people to 2 teams
    conflicts[i] = (a, b): a and b cannot be on the same team
    """
    sat = TwoSAT(n)

    for a, b in conflicts:
        # a and b on different teams
        # team[a] XOR team[b] = True
        sat.add_xor(a, b)

    result = sat.solve()
    if result is None:
        return None

    team1 = [i for i in range(n) if result[i]]
    team2 = [i for i in range(n) if not result[i]]
    return team1, team2

# Example: 0-1 conflict, 1-2 conflict
result = team_assignment(3, [(0, 1), (1, 2)])
print(result)  # ([0, 2], [1]) or ([1], [0, 2])
```

---

## 6. Practical Problem Patterns

### Pattern 1: Count SCCs

```python
def count_sccs(n, edges):
    tarjan = TarjanSCC(n)
    for u, v in edges:
        tarjan.add_edge(u, v)
    return len(tarjan.find_sccs())
```

### Pattern 2: Universal Source

```python
def find_universal_source(n, edges):
    """
    Is there a starting point that can reach all nodes?
    Possible if there's exactly 1 SCC with in-degree 0 in condensation graph
    """
    sccs, scc_id, dag = build_condensation_graph(n, edges)
    num_sccs = len(sccs)

    in_degree = [0] * num_sccs
    for u in range(num_sccs):
        for v in dag[u]:
            in_degree[v] += 1

    sources = [i for i in range(num_sccs) if in_degree[i] == 0]

    if len(sources) == 1:
        # Return any node from this SCC
        return sccs[sources[0]][0]
    return -1
```

### Pattern 3: Minimum Edges to Make Strongly Connected

```python
def min_edges_to_strongly_connect(n, edges):
    """
    Minimum edges to add to make entire graph one SCC
    """
    if n <= 1:
        return 0

    sccs, scc_id, dag = build_condensation_graph(n, edges)
    num_sccs = len(sccs)

    if num_sccs == 1:
        return 0

    in_degree = [0] * num_sccs
    out_degree = [0] * num_sccs

    for u in range(num_sccs):
        for v in dag[u]:
            out_degree[u] += 1
            in_degree[v] += 1

    sources = sum(1 for d in in_degree if d == 0)  # In-degree 0
    sinks = sum(1 for d in out_degree if d == 0)   # Out-degree 0

    return max(sources, sinks)
```

### Pattern 4: Nodes in Cycles

```python
def nodes_in_cycles(n, edges):
    """Find all nodes that are part of cycles"""
    tarjan = TarjanSCC(n)
    for u, v in edges:
        tarjan.add_edge(u, v)

    sccs = tarjan.find_sccs()

    # Nodes in SCCs with size ≥ 2
    cycle_nodes = set()
    for component in sccs:
        if len(component) > 1:
            cycle_nodes.update(component)

    # Size-1 SCCs with self-loop also form cycles
    edge_set = set(edges)
    for component in sccs:
        if len(component) == 1:
            node = component[0]
            if (node, node) in edge_set:
                cycle_nodes.add(node)

    return cycle_nodes
```

---

## 7. Time Complexity Summary

| Operation | Kosaraju | Tarjan |
|------|----------|--------|
| Find SCCs | O(V + E) | O(V + E) |
| Condensation graph | O(V + E) | O(V + E) |
| 2-SAT | - | O(V + E) |
| Space | O(V + E) × 2 | O(V + E) |

---

## 8. Common Mistakes

### Mistake 1: Missing on_stack Check

```python
# Wrong code
if disc[neighbor] != -1:
    low[node] = min(low[node], disc[neighbor])

# Correct code
if disc[neighbor] != -1 and on_stack[neighbor]:
    low[node] = min(low[node], disc[neighbor])
```

### Mistake 2: 2-SAT Node Number Calculation Error

```python
# For variable i:
# - i itself: 2*i
# - ¬i: 2*i + 1

# Wrong negation
def negate(node):
    return node + 1  # Wrong!

# Correct negation
def negate(node):
    return node ^ 1  # XOR to toggle 0↔1
```

### Mistake 3: Duplicate Edges in Condensation Graph

```python
# Duplicate edges possible
for u, v in edges:
    if scc_id[u] != scc_id[v]:
        condensed[scc_id[u]].append(scc_id[v])  # Duplicates!

# Fix with set
condensed = [set() for _ in range(num_sccs)]
for u, v in edges:
    if scc_id[u] != scc_id[v]:
        condensed[scc_id[u]].add(scc_id[v])
```

---

## 9. Practice Problems

| Difficulty | Problem Type | Key Concepts |
|--------|----------|-----------|
| ★★☆ | Count SCCs | Basic Tarjan/Kosaraju |
| ★★☆ | Domino | SCC + DAG analysis |
| ★★★ | 2-SAT basic | Graph transformation + SCC |
| ★★★ | Team assignment | 2-SAT application |
| ★★★★ | Minimum edge addition | Condensation graph utilization |

---

## Next Steps

- [18_Dynamic_Programming.md](./18_Dynamic_Programming.md) - Dynamic Programming, Memoization

---

## Learning Checklist

1. What is the meaning of low-link value in Tarjan's algorithm?
2. How do you transform (a ∨ b) into a graph in 2-SAT?
3. Why is the condensation graph always a DAG?
4. Why does Kosaraju use the reversed graph?
