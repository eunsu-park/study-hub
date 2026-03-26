# Shortest Path

## Learning Objectives

After completing this lesson, you will be able to:

1. Compare Dijkstra, Bellman-Ford, Floyd-Warshall, and 0-1 BFS algorithms by time complexity, negative-weight support, and applicable use cases
2. Implement Dijkstra's algorithm using a priority queue and explain why it requires non-negative weights
3. Implement the Bellman-Ford algorithm and use it to detect negative-weight cycles in a graph
4. Implement Floyd-Warshall for all-pairs shortest paths and trace the dynamic programming recurrence
5. Apply 0-1 BFS with a deque to solve problems where edge weights are restricted to 0 or 1
6. Select the most appropriate shortest-path algorithm for a given problem based on graph properties

---

## Overview

Learn algorithms for finding the shortest path between two vertices in weighted graphs. This covers Dijkstra, Bellman-Ford, and Floyd-Warshall algorithms.

---

## Table of Contents

1. [Shortest Path Algorithm Comparison](#1-shortest-path-algorithm-comparison)
2. [Dijkstra](#2-dijkstra)
3. [Bellman-Ford](#3-bellman-ford)
4. [Floyd-Warshall](#4-floyd-warshall)
5. [0-1 BFS](#5-0-1-bfs)
6. [Practice Problems](#6-practice-problems)

---

## 1. Shortest Path Algorithm Comparison

```
┌─────────────────┬─────────────┬───────────────┬─────────────────┐
│ Algorithm       │ Time        │ Negative Wt   │ Use Case        │
├─────────────────┼─────────────┼───────────────┼─────────────────┤
│ BFS             │ O(V+E)      │ ✗             │ Unweighted      │
│ Dijkstra        │ O(E log V)  │ ✗             │ Single source   │
│ Bellman-Ford    │ O(VE)       │ ✓             │ Single source   │
│ Floyd-Warshall  │ O(V³)       │ ✓             │ All pairs       │
│ 0-1 BFS         │ O(V+E)      │ 0,1 only      │ 0/1 weights     │
└─────────────────┴─────────────┴───────────────┴─────────────────┘
```

---

## 2. Dijkstra

### Concept

```
Shortest distance from single source to all vertices
Requirement: No negative weights

Principle:
1. Start distance = 0, rest = ∞
2. Select unvisited vertex with minimum distance
3. Update distances to adjacent vertices through this vertex
4. Repeat until all vertices visited
```

### Example

```
        (B)
       / 1 \
      4     2
     /       \
   (A)       (D)
     \       /
      2     1
       \ 3 /
        (C)

Start: A

Step  Visited    dist[A] dist[B] dist[C] dist[D]
----  -----      ------  ------  ------  ------
0     {}         0       ∞       ∞       ∞
1     {A}        0       4       2       ∞       ← Visit A, update B,C
2     {A,C}      0       4       2       5       ← Visit C, update D (2+3)
3     {A,C,B}    0       4       2       5       ← Visit B
4     {all}      0       4       2       5       ← Visit D

Result: A→B: 4, A→C: 2, A→D: 5
```

### Implementation (Priority Queue)

```c
// C - Adjacency list + array (simple version)
#define INF 1000000000
#define MAX_V 10001

typedef struct {
    int to, weight;
} Edge;

Edge adj[MAX_V][MAX_V];
int adjSize[MAX_V];
int dist[MAX_V];
int visited[MAX_V];
int V, E;

void dijkstra(int start) {
    for (int i = 0; i < V; i++) {
        dist[i] = INF;
        visited[i] = 0;
    }
    dist[start] = 0;

    for (int i = 0; i < V; i++) {
        // Find minimum distance vertex
        int u = -1;
        int minDist = INF;
        for (int j = 0; j < V; j++) {
            if (!visited[j] && dist[j] < minDist) {
                minDist = dist[j];
                u = j;
            }
        }

        if (u == -1) break;
        visited[u] = 1;

        // Update adjacent vertices
        for (int j = 0; j < adjSize[u]; j++) {
            int v = adj[u][j].to;
            int w = adj[u][j].weight;
            if (dist[u] + w < dist[v]) {
                dist[v] = dist[u] + w;
            }
        }
    }
}
```

```cpp
// C++ - Using priority queue (efficient)
#include <queue>
#include <vector>
using namespace std;

typedef pair<int, int> pii;  // {distance, vertex}

vector<int> dijkstra(const vector<vector<pii>>& adj, int start) {
    int V = adj.size();
    vector<int> dist(V, INT_MAX);
    // Min-heap ordered by distance — the greedy choice is always to process
    // the nearest unfinished node, which guarantees its shortest path is final:
    // any other path to it would be longer because edge weights are non-negative
    priority_queue<pii, vector<pii>, greater<pii>> pq;

    dist[start] = 0;
    pq.push({0, start});

    while (!pq.empty()) {
        auto [d, u] = pq.top();
        pq.pop();

        // Lazy deletion: when we find a shorter path we push a new entry without
        // removing the old one; this check discards stale (outdated) entries —
        // they arise because C++ priority_queue lacks a decrease-key operation
        if (d > dist[u]) continue;

        for (auto [v, weight] : adj[u]) {
            // Relaxation: if going through u improves the known distance to v,
            // update and push v so it is reconsidered at its better distance
            if (dist[u] + weight < dist[v]) {
                dist[v] = dist[u] + weight;
                pq.push({dist[v], v});
            }
        }
    }

    return dist;
}

// Usage
// adj[u].push_back({v, weight});  // u → v, weight
// auto dist = dijkstra(adj, 0);
```

```python
# Python
import heapq
from collections import defaultdict

def dijkstra(graph, start):
    dist = {node: float('inf') for node in graph}
    dist[start] = 0
    # Priority queue stores (distance, vertex); Python's heapq is a min-heap,
    # so the vertex with the smallest known distance is always popped first —
    # this implements the greedy selection that makes Dijkstra correct
    pq = [(0, start)]

    while pq:
        d, u = heapq.heappop(pq)

        # Stale entry check: a cheaper path to u was already finalized after
        # this entry was pushed; skip it to avoid redundant relaxation work
        if d > dist[u]:
            continue

        for v, weight in graph[u]:
            # Edge relaxation: if the path through u is better, record and
            # enqueue; non-negative weights guarantee we never need to revise
            # a vertex's distance once it has been popped from the heap
            if dist[u] + weight < dist[v]:
                dist[v] = dist[u] + weight
                heapq.heappush(pq, (dist[v], v))

    return dist

# Usage
# graph = defaultdict(list)
# graph[0].append((1, 4))  # 0 → 1, weight 4
# graph[0].append((2, 2))  # 0 → 2, weight 2
# dist = dijkstra(graph, 0)
```

### Path Reconstruction

```cpp
// C++
pair<vector<int>, vector<int>> dijkstraWithPath(
    const vector<vector<pii>>& adj, int start) {

    int V = adj.size();
    vector<int> dist(V, INT_MAX);
    vector<int> parent(V, -1);
    priority_queue<pii, vector<pii>, greater<pii>> pq;

    dist[start] = 0;
    pq.push({0, start});

    while (!pq.empty()) {
        auto [d, u] = pq.top();
        pq.pop();

        if (d > dist[u]) continue;

        for (auto [v, weight] : adj[u]) {
            if (dist[u] + weight < dist[v]) {
                dist[v] = dist[u] + weight;
                parent[v] = u;  // Store path
                pq.push({dist[v], v});
            }
        }
    }

    return {dist, parent};
}

// Print path
vector<int> getPath(const vector<int>& parent, int end) {
    vector<int> path;
    for (int v = end; v != -1; v = parent[v]) {
        path.push_back(v);
    }
    reverse(path.begin(), path.end());
    return path;
}
```

```python
# Python
def dijkstra_with_path(graph, start):
    dist = {node: float('inf') for node in graph}
    parent = {node: None for node in graph}
    dist[start] = 0
    pq = [(0, start)]

    while pq:
        d, u = heapq.heappop(pq)

        if d > dist[u]:
            continue

        for v, weight in graph[u]:
            if dist[u] + weight < dist[v]:
                dist[v] = dist[u] + weight
                parent[v] = u
                heapq.heappush(pq, (dist[v], v))

    return dist, parent

def get_path(parent, end):
    path = []
    v = end
    while v is not None:
        path.append(v)
        v = parent[v]
    return path[::-1]
```

---

## 3. Bellman-Ford

### Concept

```
Shortest distance from single source to all vertices
Feature: Allows negative weights, detects negative cycles

Principle:
1. Start distance = 0, rest = ∞
2. Update distances through all edges (V-1 iterations)
3. If updated on Vth iteration, negative cycle exists
```

### Example

```
        (B)
       / 1 \
      4     2
     /       \
   (A)       (D)
     \       /
      2     -5    ← negative weight
       \ 3 /
        (C)

Edges: (A,B,4), (A,C,2), (B,D,2), (C,B,1), (C,D,3)

Iteration 1: dist = [0, 4, 2, 5]
Iteration 2: dist = [0, 3, 2, 5]  ← B updated via C→B (2+1=3)
Iteration 3: dist = [0, 3, 2, 5]  ← no change

Result: A→B: 3, A→C: 2, A→D: 5
```

### Implementation

```c
// C
#define INF 1000000000

typedef struct {
    int from, to, weight;
} Edge;

Edge edges[20001];
int dist[501];
int V, E;

int bellmanFord(int start) {
    for (int i = 0; i < V; i++) {
        dist[i] = INF;
    }
    dist[start] = 0;

    // V-1 iterations
    for (int i = 0; i < V - 1; i++) {
        for (int j = 0; j < E; j++) {
            int u = edges[j].from;
            int v = edges[j].to;
            int w = edges[j].weight;

            if (dist[u] != INF && dist[u] + w < dist[v]) {
                dist[v] = dist[u] + w;
            }
        }
    }

    // Check for negative cycle
    for (int j = 0; j < E; j++) {
        int u = edges[j].from;
        int v = edges[j].to;
        int w = edges[j].weight;

        if (dist[u] != INF && dist[u] + w < dist[v]) {
            return -1;  // Negative cycle exists
        }
    }

    return 0;
}
```

```cpp
// C++
struct Edge {
    int from, to, weight;
};

vector<int> bellmanFord(int V, const vector<Edge>& edges, int start) {
    vector<int> dist(V, INT_MAX);
    dist[start] = 0;

    // V-1 iterations
    for (int i = 0; i < V - 1; i++) {
        for (const auto& e : edges) {
            if (dist[e.from] != INT_MAX &&
                dist[e.from] + e.weight < dist[e.to]) {
                dist[e.to] = dist[e.from] + e.weight;
            }
        }
    }

    // Check for negative cycle
    for (const auto& e : edges) {
        if (dist[e.from] != INT_MAX &&
            dist[e.from] + e.weight < dist[e.to]) {
            return {};  // Negative cycle
        }
    }

    return dist;
}
```

```python
# Python
def bellman_ford(V, edges, start):
    dist = [float('inf')] * V
    dist[start] = 0

    # V-1 relaxation rounds: the shortest path in a graph with V vertices can
    # have at most V-1 edges, so V-1 rounds guarantee all shortest paths are found
    # (unlike Dijkstra, this works even when negative weights are present)
    for _ in range(V - 1):
        for u, v, w in edges:
            if dist[u] != float('inf') and dist[u] + w < dist[v]:
                dist[v] = dist[u] + w

    # A Vth relaxation round that still improves a distance signals a negative cycle:
    # shortest paths would be infinitely short, so no finite answer exists
    for u, v, w in edges:
        if dist[u] != float('inf') and dist[u] + w < dist[v]:
            return None  # Negative cycle

    return dist
```

---

## 4. Floyd-Warshall

### Concept

```
Shortest distance between all pairs of vertices
Feature: Allows negative weights, DP-based

Principle:
Path through k vs direct path
dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])

Repeat for all k
```

### Example

```
     1     2
(0) ─→ (1) ─→ (2)
 └──────3──────→

Initial:
      0     1     2
  0 [ 0,    1,    3]
  1 [INF,   0,    2]
  2 [INF,  INF,   0]

k=1 (via 1):
  dist[0][2] = min(3, dist[0][1]+dist[1][2])
             = min(3, 1+2) = 3

Result: 0→2 shortest distance = 3
```

### Implementation

```c
// C
#define INF 1000000000
#define MAX_V 500

int dist[MAX_V][MAX_V];
int V;

void floydWarshall() {
    // k as intermediate vertex
    for (int k = 0; k < V; k++) {
        for (int i = 0; i < V; i++) {
            for (int j = 0; j < V; j++) {
                if (dist[i][k] != INF && dist[k][j] != INF) {
                    if (dist[i][k] + dist[k][j] < dist[i][j]) {
                        dist[i][j] = dist[i][k] + dist[k][j];
                    }
                }
            }
        }
    }
}

// Initialization
void initGraph() {
    for (int i = 0; i < V; i++) {
        for (int j = 0; j < V; j++) {
            if (i == j) dist[i][j] = 0;
            else dist[i][j] = INF;
        }
    }
}
```

```cpp
// C++
void floydWarshall(vector<vector<int>>& dist) {
    int V = dist.size();

    for (int k = 0; k < V; k++) {
        for (int i = 0; i < V; i++) {
            for (int j = 0; j < V; j++) {
                if (dist[i][k] != INT_MAX && dist[k][j] != INT_MAX) {
                    dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j]);
                }
            }
        }
    }
}

// Initialization
vector<vector<int>> initDist(int V) {
    vector<vector<int>> dist(V, vector<int>(V, INT_MAX));
    for (int i = 0; i < V; i++) {
        dist[i][i] = 0;
    }
    return dist;
}
```

```python
# Python
def floyd_warshall(V, edges):
    INF = float('inf')

    # Initialization: distance from a node to itself is 0;
    # all other pairs start as INF (no known path yet)
    dist = [[INF] * V for _ in range(V)]
    for i in range(V):
        dist[i][i] = 0

    for u, v, w in edges:
        dist[u][v] = w

    # DP recurrence: dist[i][j] via intermediate set {0..k} =
    # min(path not using k, path going through k as a waypoint).
    # The outer loop over k must run outermost so that when we update
    # dist[i][j] the sub-paths dist[i][k] and dist[k][j] are already optimal
    # for intermediate vertices {0..k-1}
    for k in range(V):
        for i in range(V):
            for j in range(V):
                # Relax: does routing i→j through k improve the best known path?
                if dist[i][k] + dist[k][j] < dist[i][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]

    return dist
```

### Path Reconstruction

```cpp
// C++
void floydWarshallWithPath(vector<vector<int>>& dist,
                           vector<vector<int>>& next) {
    int V = dist.size();

    // Initialization
    for (int i = 0; i < V; i++) {
        for (int j = 0; j < V; j++) {
            if (dist[i][j] != INT_MAX && i != j) {
                next[i][j] = j;
            } else {
                next[i][j] = -1;
            }
        }
    }

    for (int k = 0; k < V; k++) {
        for (int i = 0; i < V; i++) {
            for (int j = 0; j < V; j++) {
                if (dist[i][k] != INT_MAX && dist[k][j] != INT_MAX &&
                    dist[i][k] + dist[k][j] < dist[i][j]) {
                    dist[i][j] = dist[i][k] + dist[k][j];
                    next[i][j] = next[i][k];
                }
            }
        }
    }
}

vector<int> getPath(const vector<vector<int>>& next, int from, int to) {
    if (next[from][to] == -1) return {};

    vector<int> path = {from};
    while (from != to) {
        from = next[from][to];
        path.push_back(from);
    }
    return path;
}
```

---

## 5. 0-1 BFS

### Concept

```
Shortest distance in graphs with edge weights 0 or 1
→ Solve in O(V+E) using Deque

Principle:
- Weight 0 edge: Add to front of deque
- Weight 1 edge: Add to back of deque

Effect: Minimum distance vertex always at front
```

### Implementation

```cpp
// C++
vector<int> zeroOneBFS(const vector<vector<pii>>& adj, int start) {
    int V = adj.size();
    vector<int> dist(V, INT_MAX);
    deque<int> dq;

    dist[start] = 0;
    dq.push_front(start);

    while (!dq.empty()) {
        int u = dq.front();
        dq.pop_front();

        for (auto [v, weight] : adj[u]) {
            if (dist[u] + weight < dist[v]) {
                dist[v] = dist[u] + weight;

                if (weight == 0) {
                    dq.push_front(v);
                } else {
                    dq.push_back(v);
                }
            }
        }
    }

    return dist;
}
```

```python
# Python
from collections import deque

def zero_one_bfs(graph, start, n):
    dist = [float('inf')] * n
    dist[start] = 0
    dq = deque([start])

    while dq:
        u = dq.popleft()

        for v, weight in graph[u]:
            if dist[u] + weight < dist[v]:
                dist[v] = dist[u] + weight

                if weight == 0:
                    # Zero-cost edge: v is at the same "distance level" as u;
                    # push to the front so it is processed before any weight-1 nodes,
                    # preserving the invariant that the front always holds the minimum
                    dq.appendleft(v)
                else:
                    # Weight-1 edge: v is one level farther; push to the back
                    # like standard BFS — this deque trick replaces a priority queue
                    # and achieves O(V+E) instead of O(E log V)
                    dq.append(v)

    return dist
```

---

## 6. Practice Problems

### Problem 1: Find Cities at Exact Distance K

Find all cities exactly K distance from start city.

<details>
<summary>Solution Code</summary>

```python
import heapq

def cities_at_distance_k(n, edges, start, k):
    graph = [[] for _ in range(n + 1)]
    for a, b in edges:
        graph[a].append(b)  # Directed

    dist = [float('inf')] * (n + 1)
    dist[start] = 0
    pq = [(0, start)]

    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        for v in graph[u]:
            if dist[u] + 1 < dist[v]:
                dist[v] = dist[u] + 1
                heapq.heappush(pq, (dist[v], v))

    result = [i for i in range(1, n + 1) if dist[i] == k]
    return sorted(result) if result else [-1]
```

</details>

### Problem 2: Negative Cycle Detection

Check if a negative cycle exists.

<details>
<summary>Solution Code</summary>

```python
def has_negative_cycle(V, edges):
    dist = [0] * V  # Can start from any vertex

    for _ in range(V - 1):
        for u, v, w in edges:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w

    for u, v, w in edges:
        if dist[u] + w < dist[v]:
            return True

    return False
```

</details>

### Recommended Problems

| Difficulty | Problem | Platform | Algorithm |
|--------|------|--------|----------|
| ⭐⭐ | [Shortest Path](https://www.acmicpc.net/problem/1753) | BOJ | Dijkstra |
| ⭐⭐ | [Network Delay Time](https://leetcode.com/problems/network-delay-time/) | LeetCode | Dijkstra |
| ⭐⭐⭐ | [Time Machine](https://www.acmicpc.net/problem/11657) | BOJ | Bellman-Ford |
| ⭐⭐⭐ | [Floyd](https://www.acmicpc.net/problem/11404) | BOJ | Floyd-Warshall |
| ⭐⭐⭐ | [Find Path](https://www.acmicpc.net/problem/11403) | BOJ | Floyd-Warshall |
| ⭐⭐⭐⭐ | [Cheapest Flights](https://leetcode.com/problems/cheapest-flights-within-k-stops/) | LeetCode | Modified Dijkstra |

---

## Algorithm Selection Guide

```
Question 1: Are there weights?
├── No → BFS (O(V+E))
└── Yes ↓

Question 2: Are there negative weights?
├── No → Dijkstra (O(E log V))
└── Yes ↓

Question 3: Need negative cycle detection?
├── Yes → Bellman-Ford (O(VE))
└── No ↓

Question 4: Need all-pairs shortest distance?
├── Yes → Floyd-Warshall (O(V³))
└── No → Bellman-Ford (O(VE))

Additional: If weights are only 0 and 1 → 0-1 BFS (O(V+E))
```

---

## Next Steps

- [15_Minimum_Spanning_Tree.md](./15_Minimum_Spanning_Tree.md) - Kruskal, Prim, Union-Find

---

## References

- [Dijkstra Visualization](https://visualgo.net/en/sssp)
- [Shortest Path Problems](https://cp-algorithms.com/graph/shortest_paths.html)
- Introduction to Algorithms (CLRS) - Chapter 24, 25
