# Graphs Basics

**Previous**: [Heaps](./08_Heaps.md) | **Next**: [Sets and Maps](./10_Sets_and_Maps.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Define graph terminology: vertex, edge, directed/undirected, weighted, degree
2. Represent graphs using adjacency lists and adjacency matrices
3. Implement BFS and DFS traversals
4. Detect cycles in directed and undirected graphs
5. Find connected components and determine connectivity
6. Perform topological sorting on directed acyclic graphs (DAGs)
7. Choose the appropriate graph representation for a given problem

---

A **graph** is a collection of **vertices** (nodes) connected by **edges**. Graphs are the most general data structure for modeling relationships and are used in social networks, maps, dependency systems, and countless other applications.

## Graph Terminology

```
Undirected Graph:              Directed Graph (Digraph):
    A --- B                        A ---> B
    |   / |                        |     /|
    |  /  |                        |    / |
    | /   |                        v   v  v
    C --- D                        C ---> D
```

| Term | Definition |
|------|-----------|
| **Vertex (node)** | A point in the graph |
| **Edge** | A connection between two vertices |
| **Directed edge** | Edge with a direction (A -> B) |
| **Undirected edge** | Edge without direction (A -- B) |
| **Weighted edge** | Edge with an associated cost/value |
| **Degree** | Number of edges connected to a vertex |
| **In-degree** | (Directed) Number of incoming edges |
| **Out-degree** | (Directed) Number of outgoing edges |
| **Path** | Sequence of vertices connected by edges |
| **Cycle** | Path that starts and ends at the same vertex |
| **Connected** | Path exists between every pair of vertices |
| **DAG** | Directed Acyclic Graph (no cycles) |

### Graph Types

```
Simple Graph:      Multigraph:        Self-loop:
  A --- B           A === B              A
  |     |           |     |             / \
  C --- D           C --- D            +---+

Complete Graph K4:  Bipartite:        Tree (connected acyclic):
  A --- B           A     D                A
  |\ /| |           |\ /|               / | \
  | X  |            | X |              B  C  D
  |/ \|             |/ \|             / \
  C --- D           B     E          E   F
```

## Graph Representation

### Adjacency List

Each vertex stores a list of its neighbors. Most common for sparse graphs:

```python
class Graph:
    """Graph using adjacency list (dictionary of lists)."""
    
    def __init__(self, directed=False):
        self._adj = {}  # {vertex: [neighbors]}
        self._directed = directed
    
    def add_vertex(self, v):
        if v not in self._adj:
            self._adj[v] = []
    
    def add_edge(self, u, v, weight=None):
        self.add_vertex(u)
        self.add_vertex(v)
        self._adj[u].append((v, weight) if weight is not None else v)
        if not self._directed:
            self._adj[v].append((u, weight) if weight is not None else u)
    
    def neighbors(self, v):
        return self._adj.get(v, [])
    
    def vertices(self):
        return self._adj.keys()
    
    def __repr__(self):
        lines = []
        for v in sorted(self._adj):
            lines.append(f"  {v}: {self._adj[v]}")
        return "Graph {\n" + "\n".join(lines) + "\n}"
```

```
Adjacency List for undirected graph:
A: [B, C]
B: [A, C, D]
C: [A, B, D]
D: [B, C]

Memory: O(V + E)
```

### Adjacency Matrix

A 2D array where `matrix[i][j] = 1` if edge (i, j) exists:

```python
class GraphMatrix:
    """Graph using adjacency matrix."""
    
    def __init__(self, num_vertices, directed=False):
        self._n = num_vertices
        self._matrix = [[0] * num_vertices for _ in range(num_vertices)]
        self._directed = directed
    
    def add_edge(self, u, v, weight=1):
        self._matrix[u][v] = weight
        if not self._directed:
            self._matrix[v][u] = weight
    
    def has_edge(self, u, v):
        return self._matrix[u][v] != 0
    
    def neighbors(self, v):
        return [u for u in range(self._n) if self._matrix[v][u] != 0]
```

```
Adjacency Matrix (A=0, B=1, C=2, D=3):
     A  B  C  D
A [  0  1  1  0 ]
B [  1  0  1  1 ]
C [  1  1  0  1 ]
D [  0  1  1  0 ]

Memory: O(V^2)
```

### Comparison

| Aspect | Adjacency List | Adjacency Matrix |
|--------|---------------|-----------------|
| Space | O(V + E) | O(V^2) |
| Add edge | O(1) | O(1) |
| Remove edge | O(degree) | O(1) |
| Check edge | O(degree) | **O(1)** |
| Iterate neighbors | O(degree) | O(V) |
| Best for | Sparse graphs | Dense graphs |

## Breadth-First Search (BFS)

BFS explores all neighbors at the current depth before moving deeper. Uses a **queue**.

```
BFS from A:           Visit order: A, B, C, D, E, F
     [A]               
    / | \              Level 0: A
  [B][C][D]            Level 1: B, C, D
  |       |            Level 2: E, F
 [E]     [F]
```

```python
from collections import deque

def bfs(graph, start):
    """Breadth-first search.
    
    Returns list of vertices in BFS order.
    """
    visited = {start}
    queue = deque([start])
    order = []
    
    while queue:
        vertex = queue.popleft()
        order.append(vertex)
        
        for neighbor in graph.neighbors(vertex):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    
    return order

def bfs_shortest_path(graph, start, end):
    """Find shortest path (unweighted) using BFS."""
    if start == end:
        return [start]
    
    visited = {start}
    queue = deque([(start, [start])])
    
    while queue:
        vertex, path = queue.popleft()
        for neighbor in graph.neighbors(vertex):
            if neighbor == end:
                return path + [neighbor]
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))
    
    return None  # No path found
```

## Depth-First Search (DFS)

DFS explores as far as possible along each branch before backtracking. Uses a **stack** (or recursion).

```
DFS from A (one possible order): A, B, E, C, D, F
     [A]               
    / | \              Go deep before exploring siblings
  [B][C][D]            
  |       |            
 [E]     [F]
```

```python
def dfs_recursive(graph, start, visited=None):
    """Depth-first search (recursive)."""
    if visited is None:
        visited = set()
    visited.add(start)
    order = [start]
    
    for neighbor in graph.neighbors(start):
        if neighbor not in visited:
            order.extend(dfs_recursive(graph, neighbor, visited))
    
    return order

def dfs_iterative(graph, start):
    """Depth-first search (iterative with stack)."""
    visited = set()
    stack = [start]
    order = []
    
    while stack:
        vertex = stack.pop()
        if vertex not in visited:
            visited.add(vertex)
            order.append(vertex)
            for neighbor in reversed(graph.neighbors(vertex)):
                if neighbor not in visited:
                    stack.append(neighbor)
    
    return order
```

### BFS vs DFS

| Aspect | BFS | DFS |
|--------|-----|-----|
| Data structure | Queue | Stack (or recursion) |
| Exploration | Level by level | Branch by branch |
| Shortest path | Yes (unweighted) | No |
| Memory | O(V) in worst case | O(V) in worst case |
| Complete | Yes | Yes |
| Best for | Shortest paths, level-order | Cycle detection, topological sort |

## Cycle Detection

### Undirected Graph

```python
def has_cycle_undirected(graph):
    """Detect cycle in undirected graph using DFS."""
    visited = set()
    
    def dfs(vertex, parent):
        visited.add(vertex)
        for neighbor in graph.neighbors(vertex):
            if neighbor not in visited:
                if dfs(neighbor, vertex):
                    return True
            elif neighbor != parent:
                return True  # Back edge found
        return False
    
    for vertex in graph.vertices():
        if vertex not in visited:
            if dfs(vertex, None):
                return True
    return False
```

### Directed Graph

```python
def has_cycle_directed(graph):
    """Detect cycle in directed graph using coloring."""
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {v: WHITE for v in graph.vertices()}
    
    def dfs(vertex):
        color[vertex] = GRAY  # Currently being explored
        for neighbor in graph.neighbors(vertex):
            if color[neighbor] == GRAY:
                return True  # Back edge = cycle
            if color[neighbor] == WHITE:
                if dfs(neighbor):
                    return True
        color[vertex] = BLACK  # Fully explored
        return False
    
    for vertex in graph.vertices():
        if color[vertex] == WHITE:
            if dfs(vertex):
                return True
    return False
```

## Connected Components

```python
def connected_components(graph):
    """Find all connected components in an undirected graph."""
    visited = set()
    components = []
    
    for vertex in graph.vertices():
        if vertex not in visited:
            component = []
            queue = deque([vertex])
            visited.add(vertex)
            while queue:
                v = queue.popleft()
                component.append(v)
                for neighbor in graph.neighbors(v):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            components.append(component)
    
    return components
```

## Topological Sort

A linear ordering of vertices in a DAG such that for every edge (u, v), u appears before v:

```
Build System Dependencies:
  A -> B -> D
  A -> C -> D
  
Topological order: A, B, C, D  (or A, C, B, D)
```

```python
def topological_sort(graph):
    """Topological sort using DFS (Kahn-like)."""
    visited = set()
    result = []
    
    def dfs(vertex):
        visited.add(vertex)
        for neighbor in graph.neighbors(vertex):
            if neighbor not in visited:
                dfs(neighbor)
        result.append(vertex)  # Add after all descendants
    
    for vertex in graph.vertices():
        if vertex not in visited:
            dfs(vertex)
    
    return result[::-1]  # Reverse for correct order

def topological_sort_kahn(graph):
    """Topological sort using Kahn's algorithm (BFS-based)."""
    in_degree = {v: 0 for v in graph.vertices()}
    for v in graph.vertices():
        for neighbor in graph.neighbors(v):
            in_degree[neighbor] += 1
    
    queue = deque([v for v in graph.vertices() if in_degree[v] == 0])
    result = []
    
    while queue:
        vertex = queue.popleft()
        result.append(vertex)
        for neighbor in graph.neighbors(vertex):
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
    
    if len(result) != len(in_degree):
        raise ValueError("Graph has a cycle -- topological sort impossible")
    
    return result
```

## Summary

| Concept | Key Takeaway |
|---------|-------------|
| Graph | Vertices + edges modeling relationships |
| Adjacency list | O(V+E) space, good for sparse graphs |
| Adjacency matrix | O(V^2) space, O(1) edge lookup |
| BFS | Queue-based, finds shortest paths (unweighted) |
| DFS | Stack-based, explores deeply first |
| Cycle detection | Color-based DFS for directed graphs |
| Connected components | BFS/DFS on each unvisited vertex |
| Topological sort | Linear ordering respecting edge directions |

---

**Next**: [Sets and Maps](./10_Sets_and_Maps.md) -- Explore mathematical set operations and map abstractions.
