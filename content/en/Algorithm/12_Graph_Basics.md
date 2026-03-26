# Graph Basics

## Learning Objectives

After completing this lesson, you will be able to:

1. Define core graph terminology (vertex, edge, directed/undirected, weighted, connected) and classify graphs by their properties
2. Implement graph representations using adjacency lists and adjacency matrices, and compare their trade-offs
3. Implement Depth First Search (DFS) both recursively and iteratively to traverse or explore a graph
4. Implement Breadth First Search (BFS) using a queue and use it to find shortest paths in unweighted graphs
5. Compare DFS and BFS and select the appropriate traversal strategy for a given problem
6. Apply graph traversal algorithms to solve practical problems such as cycle detection and connected components

---

## Overview

A graph is a data structure consisting of vertices and edges, used to represent networks, relationships, and paths. This lesson covers basic graph concepts and DFS, BFS traversal.

---

## Table of Contents

1. [Graph Basic Concepts](#1-graph-basic-concepts)
2. [Graph Representation](#2-graph-representation)
3. [Depth First Search (DFS)](#3-depth-first-search-dfs)
4. [Breadth First Search (BFS)](#4-breadth-first-search-bfs)
5. [DFS vs BFS](#5-dfs-vs-bfs)
6. [Graph Applications](#6-graph-applications)
7. [Practice Problems](#7-practice-problems)

---

## 1. Graph Basic Concepts

### Graph Terminology

```
Graph G = (V, E)
- V: Set of vertices
- E: Set of edges

Example:
    (1)───(2)
     │   / │
     │  /  │
    (3)───(4)

V = {1, 2, 3, 4}
E = {(1,2), (1,3), (2,3), (2,4), (3,4)}
```

### Types of Graphs

```
1. Directed/Undirected Graphs

   Undirected:           Directed:
   (1)───(2)            (1)───→(2)
                         ↑       │
                         │       ↓
                        (4)←──(3)

2. Weighted Graph

    (1)──5──(2)
     │     / │
     3   2   4
     │ /     │
    (3)──1──(4)

3. Connected/Disconnected Graph

   Connected:           Disconnected:
   (1)─(2)            (1)─(2)  (4)
    │ / │                 │
   (3)─(4)            (3)─┘
```

### Graph Properties

```
- Degree: Number of edges connected to a vertex
  - Undirected: degree(v)
  - Directed: in-degree(v), out-degree(v)

- Path: Sequence of vertices
- Cycle: Path where start equals end
- Connected Component: Set of connected vertices
```

---

## 2. Graph Representation

### 2.1 Adjacency Matrix

```
Undirected graph:
    (0)───(1)
     │   / │
     │  /  │
    (2)───(3)

Adjacency matrix:
       0  1  2  3
    0 [0, 1, 1, 0]
    1 [1, 0, 1, 1]
    2 [1, 1, 0, 1]
    3 [0, 1, 1, 0]

Properties:
- Space: O(V²)
- Edge existence check: O(1)
- Adjacent vertex iteration: O(V)
- Suitable for dense graphs
```

```c
// C
#define MAX_V 100

int graph[MAX_V][MAX_V];
int V;  // Number of vertices

void addEdge(int u, int v) {
    graph[u][v] = 1;
    graph[v][u] = 1;  // Undirected
}

int hasEdge(int u, int v) {
    return graph[u][v];
}
```

```cpp
// C++
class GraphMatrix {
private:
    vector<vector<int>> adj;
    int V;

public:
    GraphMatrix(int v) : V(v), adj(v, vector<int>(v, 0)) {}

    void addEdge(int u, int v) {
        adj[u][v] = 1;
        adj[v][u] = 1;  // Undirected
    }

    bool hasEdge(int u, int v) {
        return adj[u][v] == 1;
    }
};
```

```python
# Python
class GraphMatrix:
    def __init__(self, v):
        self.V = v
        self.adj = [[0] * v for _ in range(v)]

    def add_edge(self, u, v):
        self.adj[u][v] = 1
        self.adj[v][u] = 1  # Undirected

    def has_edge(self, u, v):
        return self.adj[u][v] == 1
```

### 2.2 Adjacency List

```
Undirected graph:
    (0)───(1)
     │   / │
     │  /  │
    (2)───(3)

Adjacency list:
0: [1, 2]
1: [0, 2, 3]
2: [0, 1, 3]
3: [1, 2]

Properties:
- Space: O(V + E)
- Edge existence check: O(degree)
- Adjacent vertex iteration: O(degree)
- Suitable for sparse graphs
```

```c
// C - Linked list approach
#define MAX_V 100

typedef struct Node {
    int vertex;
    struct Node* next;
} Node;

Node* adj[MAX_V];
int V;

void addEdge(int u, int v) {
    // u -> v
    Node* newNode = (Node*)malloc(sizeof(Node));
    newNode->vertex = v;
    newNode->next = adj[u];
    adj[u] = newNode;

    // v -> u (undirected)
    newNode = (Node*)malloc(sizeof(Node));
    newNode->vertex = u;
    newNode->next = adj[v];
    adj[v] = newNode;
}
```

```cpp
// C++
class GraphList {
private:
    int V;
    vector<vector<int>> adj;

public:
    GraphList(int v) : V(v), adj(v) {}

    void addEdge(int u, int v) {
        adj[u].push_back(v);
        adj[v].push_back(u);  // Undirected
    }

    const vector<int>& neighbors(int u) {
        return adj[u];
    }
};
```

```python
# Python
class GraphList:
    def __init__(self, v):
        self.V = v
        self.adj = [[] for _ in range(v)]

    def add_edge(self, u, v):
        self.adj[u].append(v)
        self.adj[v].append(u)  # Undirected

    def neighbors(self, u):
        return self.adj[u]

# Or using defaultdict
from collections import defaultdict

graph = defaultdict(list)
graph[0].append(1)
graph[1].append(0)
```

### Representation Comparison

```
┌────────────────┬───────────────┬───────────────┐
│                │ Adj Matrix    │ Adj List      │
├────────────────┼───────────────┼───────────────┤
│ Space          │ O(V²)         │ O(V + E)      │
│ Edge check     │ O(1)          │ O(degree)     │
│ Neighbor iter  │ O(V)          │ O(degree)     │
│ Add edge       │ O(1)          │ O(1)          │
│ Best for       │ Dense graphs  │ Sparse graphs │
└────────────────┴───────────────┴───────────────┘
```

---

## 3. Depth First Search (DFS)

### Concept

```
DFS: Go deep in one direction, backtrack when stuck
→ Implemented with stack or recursion

       (0)
      / | \
    (1)(2)(3)
    /     / \
  (4)   (5)(6)

DFS order (starting from 0):
0 → 1 → 4 → (backtrack) → 2 → (backtrack) → 3 → 5 → (backtrack) → 6

Visit order: 0, 1, 4, 2, 3, 5, 6
```

### Recursive Implementation

```c
// C
#define MAX_V 100

int adj[MAX_V][MAX_V];
int visited[MAX_V];
int V;

void dfs(int v) {
    visited[v] = 1;
    printf("%d ", v);

    for (int i = 0; i < V; i++) {
        if (adj[v][i] && !visited[i]) {
            dfs(i);
        }
    }
}
```

```cpp
// C++
class Graph {
private:
    int V;
    vector<vector<int>> adj;

public:
    Graph(int v) : V(v), adj(v) {}

    void addEdge(int u, int v) {
        adj[u].push_back(v);
        adj[v].push_back(u);
    }

    void dfs(int start) {
        vector<bool> visited(V, false);
        dfsUtil(start, visited);
    }

private:
    void dfsUtil(int v, vector<bool>& visited) {
        visited[v] = true;
        cout << v << " ";

        for (int neighbor : adj[v]) {
            if (!visited[neighbor]) {
                dfsUtil(neighbor, visited);
            }
        }
    }
};
```

```python
# Python
def dfs_recursive(graph, start, visited=None):
    if visited is None:
        visited = set()

    visited.add(start)
    print(start, end=' ')

    for neighbor in graph[start]:
        if neighbor not in visited:
            dfs_recursive(graph, neighbor, visited)

    return visited
```

### Iterative Implementation (Stack)

```cpp
// C++
void dfsIterative(int start) {
    vector<bool> visited(V, false);
    stack<int> st;

    st.push(start);

    while (!st.empty()) {
        int v = st.top();
        st.pop();

        // Check visited at dequeue time (not enqueue) because a node may be
        // pushed multiple times before it is processed — each unvisited neighbor
        // eagerly pushes it, so we guard against re-processing here
        if (visited[v]) continue;

        visited[v] = true;
        cout << v << " ";

        // Push neighbors in reverse order so that when they are popped from the
        // stack the first neighbor in the adjacency list is processed first —
        // this matches the recursive DFS visit order (left-to-right), making
        // the two implementations produce identical traversal sequences
        for (auto it = adj[v].rbegin(); it != adj[v].rend(); it++) {
            if (!visited[*it]) {
                st.push(*it);
            }
        }
    }
}
```

```python
# Python
def dfs_iterative(graph, start):
    visited = set()
    stack = [start]

    while stack:
        v = stack.pop()

        # Guard against processing a node that was pushed multiple times;
        # unlike BFS we cannot cheaply mark at push time because different
        # stack paths can reach the same node before it is ever popped
        if v in visited:
            continue

        visited.add(v)
        print(v, end=' ')

        # Append neighbors in reverse so that when popped (LIFO) the first
        # neighbor is visited first — this matches the recursive call order
        # and makes the iterative version produce the same traversal sequence
        for neighbor in reversed(graph[v]):
            if neighbor not in visited:
                stack.append(neighbor)

    return visited
```

---

## 4. Breadth First Search (BFS)

### Concept

```
BFS: Visit all adjacent vertices of current vertex first
→ Implemented with queue

       (0)
      / | \
    (1)(2)(3)
    /     / \
  (4)   (5)(6)

BFS order (starting from 0):
Level 0: 0
Level 1: 1, 2, 3
Level 2: 4, 5, 6

Visit order: 0, 1, 2, 3, 4, 5, 6
```

### Implementation

```c
// C
#define MAX_V 100

int adj[MAX_V][MAX_V];
int visited[MAX_V];
int V;

void bfs(int start) {
    int queue[MAX_V];
    int front = 0, rear = 0;

    visited[start] = 1;
    queue[rear++] = start;

    while (front < rear) {
        int v = queue[front++];
        printf("%d ", v);

        for (int i = 0; i < V; i++) {
            if (adj[v][i] && !visited[i]) {
                visited[i] = 1;
                queue[rear++] = i;
            }
        }
    }
}
```

```cpp
// C++
void bfs(int start) {
    vector<bool> visited(V, false);
    queue<int> q;

    // Mark visited at enqueue time, not at dequeue time — if we waited until
    // dequeue, the same neighbor could be enqueued multiple times before it is
    // first processed, causing O(E) redundant entries and breaking O(V+E) complexity
    visited[start] = true;
    q.push(start);

    while (!q.empty()) {
        int v = q.front();
        q.pop();

        cout << v << " ";

        for (int neighbor : adj[v]) {
            if (!visited[neighbor]) {
                // Mark here (enqueue time) so no other path can enqueue this
                // neighbor again — guarantees each vertex enters the queue exactly once
                visited[neighbor] = true;
                q.push(neighbor);
            }
        }
    }
}
```

```python
# Python
from collections import deque

def bfs(graph, start):
    visited = set()
    queue = deque([start])
    # Mark visited at enqueue time — BFS guarantees shortest path in unweighted
    # graphs because it processes nodes in strict order of distance from the source;
    # marking here (rather than at dequeue) prevents a node being enqueued multiple
    # times by different neighbors and preserves the O(V+E) time bound
    visited.add(start)

    while queue:
        v = queue.popleft()
        print(v, end=' ')

        for neighbor in graph[v]:
            if neighbor not in visited:
                # Add to visited immediately so concurrent paths to the same
                # neighbor don't both enqueue it — the first path wins (shortest)
                visited.add(neighbor)
                queue.append(neighbor)

    return visited
```

### Shortest Path with BFS

```cpp
// C++ - Shortest distance in unweighted graph
vector<int> shortestPath(int start) {
    vector<int> dist(V, -1);
    queue<int> q;

    dist[start] = 0;
    q.push(start);

    while (!q.empty()) {
        int v = q.front();
        q.pop();

        for (int neighbor : adj[v]) {
            if (dist[neighbor] == -1) {
                dist[neighbor] = dist[v] + 1;
                q.push(neighbor);
            }
        }
    }

    return dist;
}
```

```python
# Python
def shortest_path(graph, start):
    dist = {start: 0}
    # BFS processes nodes in non-decreasing distance order from the source;
    # the first time a node is reached its distance is therefore minimal —
    # this property holds only in unweighted graphs (all edges cost 1)
    queue = deque([start])

    while queue:
        v = queue.popleft()

        for neighbor in graph[v]:
            # "not in dist" doubles as the visited check — we only record a
            # distance the first time we see the neighbor (which is the shortest)
            if neighbor not in dist:
                dist[neighbor] = dist[v] + 1
                queue.append(neighbor)

    return dist
```

---

## 5. DFS vs BFS

### Comparison Table

```
┌────────────────┬─────────────────────┬─────────────────────┐
│                │ DFS                 │ BFS                 │
├────────────────┼─────────────────────┼─────────────────────┤
│ Data structure │ Stack/Recursion     │ Queue               │
│ Memory         │ O(h) - height       │ O(w) - width        │
│ Complete search│ Possible            │ Possible            │
│ Shortest path  │ ✗                   │ ✓ (unweighted)      │
│ Cycle detection│ ✓                   │ ✓                   │
│ Path existence │ ✓                   │ ✓                   │
└────────────────┴─────────────────────┴─────────────────────┘
```

### When to Use Which?

```
Use DFS:
- When all nodes must be visited
- When path characteristics need to be stored
- Backtracking problems
- Cycle detection
- Topological sort

Use BFS:
- Shortest path/minimum cost (unweighted)
- Level-order traversal
- Finding nearest first
```

### Visual Comparison

```
       (1)
      / | \
    (2)(3)(4)
    / \   / \
  (5)(6)(7)(8)

DFS (Stack):
Visit order: 1 → 2 → 5 → 6 → 3 → 4 → 7 → 8
Depth first: Explore one branch completely

BFS (Queue):
Visit order: 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8
Breadth first: Explore same level first
```

---

## 6. Graph Applications

### 6.1 Finding Connected Components

```cpp
// C++
int countConnectedComponents() {
    vector<bool> visited(V, false);
    int count = 0;

    for (int i = 0; i < V; i++) {
        if (!visited[i]) {
            dfs(i, visited);  // or bfs
            count++;
        }
    }

    return count;
}
```

```python
# Python
def count_connected_components(graph, n):
    visited = set()
    count = 0

    for i in range(n):
        if i not in visited:
            dfs_recursive(graph, i, visited)
            count += 1

    return count
```

### 6.2 Cycle Detection (Undirected Graph)

```cpp
// C++
bool hasCycleDFS(int v, int parent, vector<bool>& visited) {
    visited[v] = true;

    for (int neighbor : adj[v]) {
        if (!visited[neighbor]) {
            if (hasCycleDFS(neighbor, v, visited)) {
                return true;
            }
        } else if (neighbor != parent) {
            // Visited but not parent = cycle
            return true;
        }
    }

    return false;
}

bool hasCycle() {
    vector<bool> visited(V, false);

    for (int i = 0; i < V; i++) {
        if (!visited[i]) {
            if (hasCycleDFS(i, -1, visited)) {
                return true;
            }
        }
    }

    return false;
}
```

```python
# Python
def has_cycle(graph, n):
    visited = set()

    def dfs(v, parent):
        visited.add(v)

        for neighbor in graph[v]:
            if neighbor not in visited:
                if dfs(neighbor, v):
                    return True
            elif neighbor != parent:
                return True

        return False

    for i in range(n):
        if i not in visited:
            if dfs(i, -1):
                return True

    return False
```

### 6.3 Bipartite Graph Check

```
Bipartite graph: Vertices can be divided into two groups,
                 no edges within the same group

   (1)─(2)             (1)   (2)
    │ × │     →         │     │
   (3)─(4)             (3)   (4)

Coloring method: Adjacent vertices must have different colors
```

```cpp
// C++
bool isBipartite() {
    vector<int> color(V, -1);  // -1: unvisited, 0/1: color

    for (int i = 0; i < V; i++) {
        if (color[i] == -1) {
            queue<int> q;
            q.push(i);
            color[i] = 0;

            while (!q.empty()) {
                int v = q.front();
                q.pop();

                for (int neighbor : adj[v]) {
                    if (color[neighbor] == -1) {
                        color[neighbor] = 1 - color[v];
                        q.push(neighbor);
                    } else if (color[neighbor] == color[v]) {
                        return false;  // Adjacent with same color
                    }
                }
            }
        }
    }

    return true;
}
```

```python
# Python
from collections import deque

def is_bipartite(graph, n):
    color = [-1] * n  # -1 means unvisited; 0 and 1 represent the two groups

    for start in range(n):
        if color[start] == -1:
            queue = deque([start])
            color[start] = 0

            while queue:
                v = queue.popleft()

                for neighbor in graph[v]:
                    if color[neighbor] == -1:
                        # Assign the opposite color — if v is group 0,
                        # neighbor must be group 1, and vice versa
                        color[neighbor] = 1 - color[v]
                        queue.append(neighbor)
                    elif color[neighbor] == color[v]:
                        # Two adjacent nodes share the same color — an odd-length
                        # cycle exists, which makes bipartite coloring impossible
                        return False

    return True
```

### 6.4 2D Grid Traversal

```
Treating grid as a graph:
- Each cell = vertex
- Up/down/left/right = edges

Used for maze solving, counting islands, etc.
```

```cpp
// C++ - Number of Islands
int numIslands(vector<vector<char>>& grid) {
    int m = grid.size();
    int n = grid[0].size();
    int count = 0;

    int dx[] = {0, 0, 1, -1};
    int dy[] = {1, -1, 0, 0};

    function<void(int, int)> dfs = [&](int x, int y) {
        if (x < 0 || x >= m || y < 0 || y >= n || grid[x][y] != '1') {
            return;
        }

        grid[x][y] = '0';  // Mark visited

        for (int i = 0; i < 4; i++) {
            dfs(x + dx[i], y + dy[i]);
        }
    };

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            if (grid[i][j] == '1') {
                dfs(i, j);
                count++;
            }
        }
    }

    return count;
}
```

```python
# Python
def num_islands(grid):
    if not grid:
        return 0

    m, n = len(grid), len(grid[0])
    count = 0

    def dfs(x, y):
        if x < 0 or x >= m or y < 0 or y >= n or grid[x][y] != '1':
            return

        grid[x][y] = '0'  # Mark visited

        dfs(x + 1, y)
        dfs(x - 1, y)
        dfs(x, y + 1)
        dfs(x, y - 1)

    for i in range(m):
        for j in range(n):
            if grid[i][j] == '1':
                dfs(i, j)
                count += 1

    return count
```

---

## 7. Practice Problems

### Problem 1: Path Existence Check

Check if a path exists between two vertices.

<details>
<summary>Solution Code</summary>

```python
def has_path(graph, start, end):
    visited = set()
    stack = [start]

    while stack:
        v = stack.pop()

        if v == end:
            return True

        if v in visited:
            continue

        visited.add(v)

        for neighbor in graph[v]:
            if neighbor not in visited:
                stack.append(neighbor)

    return False
```

</details>

### Problem 2: Find All Paths

Find all paths between two vertices.

<details>
<summary>Solution Code</summary>

```python
def find_all_paths(graph, start, end, path=None):
    if path is None:
        path = []

    path = path + [start]

    if start == end:
        return [path]

    paths = []

    for neighbor in graph[start]:
        if neighbor not in path:  # Prevent cycles
            new_paths = find_all_paths(graph, neighbor, end, path)
            paths.extend(new_paths)

    return paths
```

</details>

### Recommended Problems

| Difficulty | Problem | Platform | Type |
|--------|------|--------|------|
| ⭐ | [DFS and BFS](https://www.acmicpc.net/problem/1260) | BOJ | Basic traversal |
| ⭐ | [Connected Components](https://www.acmicpc.net/problem/11724) | BOJ | Components |
| ⭐⭐ | [Number of Islands](https://leetcode.com/problems/number-of-islands/) | LeetCode | Grid DFS |
| ⭐⭐ | [Maze Exploration](https://www.acmicpc.net/problem/2178) | BOJ | BFS shortest |
| ⭐⭐ | [Is Graph Bipartite?](https://leetcode.com/problems/is-graph-bipartite/) | LeetCode | Bipartite |
| ⭐⭐⭐ | [Complex Numbering](https://www.acmicpc.net/problem/2667) | BOJ | Grid traversal |

---

## Template Summary

### DFS Template

```python
# Recursive
def dfs(graph, v, visited):
    visited.add(v)
    for neighbor in graph[v]:
        if neighbor not in visited:
            dfs(graph, neighbor, visited)

# Iterative
def dfs_iterative(graph, start):
    visited = set()
    stack = [start]
    while stack:
        v = stack.pop()
        if v not in visited:
            visited.add(v)
            for neighbor in graph[v]:
                stack.append(neighbor)
```

### BFS Template

```python
from collections import deque

def bfs(graph, start):
    visited = {start}
    queue = deque([start])
    while queue:
        v = queue.popleft()
        for neighbor in graph[v]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
```

### 2D Grid Traversal

```python
dx = [0, 0, 1, -1]
dy = [1, -1, 0, 0]

def is_valid(x, y, m, n):
    return 0 <= x < m and 0 <= y < n
```

---

## Next Steps

- [13_Topological_Sort.md](./13_Topological_Sort.md) - Topological Sort

---

## References

- [Graph Visualization](https://visualgo.net/en/dfsbfs)
- [BFS/DFS Tutorial](https://www.geeksforgeeks.org/difference-between-bfs-and-dfs/)
- Introduction to Algorithms (CLRS) - Chapter 22
