# Topological Sort

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain the concept of topological sort and state the conditions under which it is applicable (DAG requirement)
2. Implement Kahn's Algorithm using BFS and in-degree tracking to produce a topological ordering
3. Implement DFS-based topological sort using a post-order stack
4. Detect cycles in a directed graph using both Kahn's algorithm and DFS coloring
5. Apply topological sort to solve dependency-ordering problems such as course scheduling and build systems

---

## Overview

Topological sort is an algorithm for linearly ordering vertices of a Directed Acyclic Graph (DAG). For every edge (u, v), u appears before v in the ordering.

---

## Table of Contents

1. [Topological Sort Concepts](#1-topological-sort-concepts)
2. [Kahn's Algorithm (BFS)](#2-kahns-algorithm-bfs)
3. [DFS-based Topological Sort](#3-dfs-based-topological-sort)
4. [Cycle Detection](#4-cycle-detection)
5. [Application Problems](#5-application-problems)
6. [Practice Problems](#6-practice-problems)

---

## 1. Topological Sort Concepts

### 1.1 DAG (Directed Acyclic Graph)

```
DAG: Directed Acyclic Graph
- Has directed edges
- Has no cycles

Topological sort is possible: Graph must be a DAG

Example: Course registration

CourseA → CourseB → CourseD
  ↓         ↓
CourseC → CourseE

Valid course orders:
A → B → C → D → E  or
A → C → B → D → E  or
A → C → B → E → D
```

### 1.2 In-degree and Out-degree

```
In-degree: Number of edges coming into a vertex
Out-degree: Number of edges going out from a vertex

Example:
    A → B
    ↓   ↓
    C → D

Vertex | In  | Out
-------|-----|-----
  A    |  0  |  2
  B    |  1  |  1
  C    |  1  |  1
  D    |  2  |  0

Vertex with in-degree 0: Starting point (no dependencies)
```

### 1.3 Properties of Topological Order

```
- Multiple topological orders may exist
- Topological sort is only possible on DAGs
- Impossible when cycle exists

Topological sort algorithms:
1. Kahn's Algorithm (BFS) - O(V + E)
2. DFS-based - O(V + E)
```

---

## 2. Kahn's Algorithm (BFS)

### 2.1 Algorithm Principle

```
1. Calculate in-degree for all vertices
2. Insert vertices with in-degree 0 into queue
3. Remove vertex from queue and add to result
4. Remove outgoing edges (decrease in-degree of adjacent vertices)
5. Insert newly zero in-degree vertices into queue
6. Repeat until queue is empty

Visualization:
Initial: in_degree = [0, 1, 1, 2]
      A(0) → B(1) → D(2)
        ↓      ↓
       C(1) ←──┘

Step 1: Select A (in-degree 0)
        Queue: [A], Result: [A]
        Decrease B, C in-degree → B(0), C(0)

Step 2: Select B (or C)
        Queue: [B, C], Result: [A, B]
        Decrease D in-degree → D(1)

Step 3: Select C
        Queue: [C], Result: [A, B, C]
        Decrease D in-degree → D(0)

Step 4: Select D
        Queue: [D], Result: [A, B, C, D]

Final: [A, B, C, D]
```

### 2.2 Implementation

```python
from collections import deque

def topological_sort_kahn(n, edges):
    """
    Kahn's Algorithm (BFS-based topological sort)
    n: Number of vertices (0 to n-1)
    edges: [(u, v), ...] u → v edges
    Time: O(V + E)
    """
    # Build graph and calculate in-degrees
    graph = [[] for _ in range(n)]
    in_degree = [0] * n

    for u, v in edges:
        graph[u].append(v)
        in_degree[v] += 1  # Count how many prerequisites each node has

    # A node with in-degree 0 has no unresolved dependencies —
    # it is safe to schedule immediately
    queue = deque()
    for i in range(n):
        if in_degree[i] == 0:
            queue.append(i)

    result = []

    while queue:
        node = queue.popleft()
        result.append(node)  # This node is now "completed"

        for neighbor in graph[node]:
            # Completing 'node' satisfies one prerequisite for each neighbor;
            # if that brings their count to 0 they become schedulable
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    # If a cycle exists, some nodes will never reach in-degree 0 and won't
    # appear in result — the length check detects this without extra bookkeeping
    if len(result) != n:
        return []  # Cycle exists

    return result


# Example
n = 6
edges = [(5, 2), (5, 0), (4, 0), (4, 1), (2, 3), (3, 1)]
print(topological_sort_kahn(n, edges))  # [4, 5, 2, 0, 3, 1] or other valid order
```

### 2.3 C++ Implementation

```cpp
#include <vector>
#include <queue>
using namespace std;

vector<int> topologicalSortKahn(int n, vector<pair<int, int>>& edges) {
    vector<vector<int>> graph(n);
    vector<int> inDegree(n, 0);

    for (auto& [u, v] : edges) {
        graph[u].push_back(v);
        inDegree[v]++;
    }

    queue<int> q;
    for (int i = 0; i < n; i++) {
        if (inDegree[i] == 0) {
            q.push(i);
        }
    }

    vector<int> result;

    while (!q.empty()) {
        int node = q.front();
        q.pop();
        result.push_back(node);

        for (int neighbor : graph[node]) {
            if (--inDegree[neighbor] == 0) {
                q.push(neighbor);
            }
        }
    }

    if (result.size() != n) {
        return {};  // Cycle exists
    }

    return result;
}
```

---

## 3. DFS-based Topological Sort

### 3.1 Algorithm Principle

```
1. Perform DFS on all vertices
2. Push to stack when DFS finishes
3. Output stack in reverse order

Idea:
- When DFS on vertex v finishes, all reachable vertices are already processed
- Therefore, vertices pushed later to stack come first in topological order

Visualization:
    A → B → D
    ↓
    C

DFS visit order: A → B → D(done) → B(done) → C(done) → A(done)
Stack (finish order): [D, B, C, A]
Result (reverse): [A, C, B, D] or [A, B, D, C]
```

### 3.2 Implementation

```python
def topological_sort_dfs(n, edges):
    """
    DFS-based topological sort
    Time: O(V + E)
    """
    graph = [[] for _ in range(n)]
    for u, v in edges:
        graph[u].append(v)

    # Three-color marking: unvisited (0) → visiting (1) → done (2)
    # Using three states instead of two allows cycle detection —
    # a 'visiting' node reached again means we are still on its DFS path (cycle)
    visited = [0] * n
    result = []
    has_cycle = False

    def dfs(node):
        nonlocal has_cycle
        if has_cycle:
            return  # Short-circuit once a cycle is found

        visited[node] = 1  # Visiting: this node is on the current DFS path

        for neighbor in graph[node]:
            if visited[neighbor] == 1:
                # Reached a node still on the recursion stack — back-edge = cycle
                has_cycle = True
                return
            if visited[neighbor] == 0:
                dfs(neighbor)

        visited[node] = 2  # Done: all descendants finished — safe to append
        # Appending after all descendants ensures that if u→v, v is appended before u;
        # reversing at the end gives the correct topological order
        result.append(node)

    for i in range(n):
        if visited[i] == 0:
            dfs(i)

    if has_cycle:
        return []

    return result[::-1]  # Reverse: nodes finished last come first in topo order


# Example
n = 6
edges = [(5, 2), (5, 0), (4, 0), (4, 1), (2, 3), (3, 1)]
print(topological_sort_dfs(n, edges))  # [5, 4, 2, 3, 1, 0] or other valid order
```

### 3.3 Iterative with Stack

```python
def topological_sort_iterative(n, edges):
    """Iterative DFS topological sort"""
    graph = [[] for _ in range(n)]
    for u, v in edges:
        graph[u].append(v)

    visited = [False] * n
    result = []

    for start in range(n):
        if visited[start]:
            continue

        # Each stack entry carries a 'processed' flag to simulate the two phases
        # of recursive DFS: the first pop triggers exploration of children,
        # the second pop (after children return) triggers the post-order append
        stack = [(start, False)]  # (node, processed)

        while stack:
            node, processed = stack.pop()

            if processed:
                # Post-order: all descendants have been visited — append node
                result.append(node)
                continue

            if visited[node]:
                continue  # Already handled via another path

            visited[node] = True
            # Re-push with processed=True so we record this node after its subtree
            stack.append((node, True))

            # Push children with processed=False; they will be explored before
            # node's post-order marker fires (LIFO order achieves DFS depth-first behavior)
            for neighbor in graph[node]:
                if not visited[neighbor]:
                    stack.append((neighbor, False))

    return result[::-1]  # Post-order is reverse topological order
```

---

## 4. Cycle Detection

### 4.1 Cycle Detection with Kahn's Algorithm

```python
def has_cycle_kahn(n, edges):
    """
    If topological sort result length is less than n, cycle exists
    """
    result = topological_sort_kahn(n, edges)
    return len(result) != n

# Case with cycle
edges_with_cycle = [(0, 1), (1, 2), (2, 0)]  # 0 → 1 → 2 → 0
print(has_cycle_kahn(3, edges_with_cycle))  # True
```

### 4.2 DFS Color-based Cycle Detection

```python
def has_cycle_dfs(n, edges):
    """
    Cycle detection using DFS coloring
    WHITE(0): Unvisited
    GRAY(1): Visiting (in recursion stack)
    BLACK(2): Done
    """
    graph = [[] for _ in range(n)]
    for u, v in edges:
        graph[u].append(v)

    WHITE, GRAY, BLACK = 0, 1, 2
    color = [WHITE] * n

    def dfs(node):
        color[node] = GRAY

        for neighbor in graph[node]:
            if color[neighbor] == GRAY:
                return True  # Cycle found
            if color[neighbor] == WHITE:
                if dfs(neighbor):
                    return True

        color[node] = BLACK
        return False

    for i in range(n):
        if color[i] == WHITE:
            if dfs(i):
                return True

    return False
```

### 4.3 Finding Cycle Path

```python
def find_cycle(n, edges):
    """Return cycle path if exists"""
    graph = [[] for _ in range(n)]
    for u, v in edges:
        graph[u].append(v)

    WHITE, GRAY, BLACK = 0, 1, 2
    color = [WHITE] * n
    parent = [-1] * n

    def dfs(node):
        color[node] = GRAY

        for neighbor in graph[node]:
            if color[neighbor] == GRAY:
                # Cycle found! Restore path
                cycle = [neighbor]
                curr = node
                while curr != neighbor:
                    cycle.append(curr)
                    curr = parent[curr]
                cycle.append(neighbor)
                return cycle[::-1]

            if color[neighbor] == WHITE:
                parent[neighbor] = node
                result = dfs(neighbor)
                if result:
                    return result

        color[node] = BLACK
        return None

    for i in range(n):
        if color[i] == WHITE:
            result = dfs(i)
            if result:
                return result

    return None

# Example
edges = [(0, 1), (1, 2), (2, 3), (3, 1)]  # 1 → 2 → 3 → 1 cycle
print(find_cycle(4, edges))  # [1, 2, 3, 1]
```

---

## 5. Application Problems

### 5.1 Course Schedule (Basic)

```python
def can_finish(num_courses, prerequisites):
    """
    Check if all courses can be taken
    prerequisites[i] = [a, b] : must take b before a
    """
    result = topological_sort_kahn(num_courses,
                                    [(b, a) for a, b in prerequisites])
    return len(result) == num_courses

# LeetCode 207. Course Schedule
print(can_finish(2, [[1, 0]]))  # True: 0 → 1
print(can_finish(2, [[1, 0], [0, 1]]))  # False: cycle
```

### 5.2 Find Course Order

```python
def find_order(num_courses, prerequisites):
    """
    Return possible course order
    Return empty list if impossible
    """
    return topological_sort_kahn(num_courses,
                                  [(b, a) for a, b in prerequisites])

# LeetCode 210. Course Schedule II
print(find_order(4, [[1, 0], [2, 0], [3, 1], [3, 2]]))
# [0, 1, 2, 3] or [0, 2, 1, 3]
```

### 5.3 Build Order

```python
def build_order(projects, dependencies):
    """
    Determine project build order
    dependencies: [(a, b), ...] - build b after a
    """
    # Project name → index mapping
    proj_to_idx = {p: i for i, p in enumerate(projects)}
    n = len(projects)

    edges = [(proj_to_idx[b], proj_to_idx[a]) for a, b in dependencies]
    order = topological_sort_kahn(n, edges)

    if not order:
        return None  # Circular dependency

    return [projects[i] for i in order]

# Example
projects = ['a', 'b', 'c', 'd', 'e', 'f']
dependencies = [('a', 'd'), ('f', 'b'), ('b', 'd'), ('f', 'a'), ('d', 'c')]
print(build_order(projects, dependencies))  # ['e', 'f', 'c', 'b', 'd', 'a'] etc.
```

### 5.4 Task Scheduling (Earliest Completion Time)

```python
def earliest_completion(n, edges, times):
    """
    Calculate earliest completion time for each task
    edges: [(u, v), ...] v can start after u completes
    times[i]: Duration of task i
    """
    from collections import deque

    graph = [[] for _ in range(n)]
    in_degree = [0] * n

    for u, v in edges:
        graph[u].append(v)
        in_degree[v] += 1

    # earliest[i] = earliest time task i can start (0 for tasks with no prerequisites)
    earliest = [0] * n

    queue = deque()
    for i in range(n):
        if in_degree[i] == 0:
            queue.append(i)

    while queue:
        node = queue.popleft()

        for neighbor in graph[node]:
            # A neighbor can only start after ALL its prerequisites finish;
            # take the max across all incoming paths to find the true earliest start
            earliest[neighbor] = max(earliest[neighbor],
                                      earliest[node] + times[node])
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                # All predecessors processed — earliest start time is now finalized
                queue.append(neighbor)

    # Completion = start + duration; the overall maximum is the critical path length
    completion = [earliest[i] + times[i] for i in range(n)]
    return completion, max(completion)

# Example
n = 4
edges = [(0, 2), (1, 2), (2, 3)]
times = [3, 2, 5, 4]
completion, total = earliest_completion(n, edges, times)
print(f"Completion times: {completion}")  # [3, 2, 8, 12]
print(f"Total duration: {total}")  # 12
```

### 5.5 Lexicographically Smallest Topological Order

```python
import heapq

def lexicographic_topological_sort(n, edges):
    """
    Lexicographically smallest topological order
    Using min heap
    """
    graph = [[] for _ in range(n)]
    in_degree = [0] * n

    for u, v in edges:
        graph[u].append(v)
        in_degree[v] += 1

    # Use min heap
    heap = []
    for i in range(n):
        if in_degree[i] == 0:
            heapq.heappush(heap, i)

    result = []

    while heap:
        node = heapq.heappop(heap)
        result.append(node)

        for neighbor in graph[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                heapq.heappush(heap, neighbor)

    return result if len(result) == n else []

# Example
n = 4
edges = [(0, 1), (0, 2), (1, 3), (2, 3)]
print(lexicographic_topological_sort(n, edges))  # [0, 1, 2, 3]
```

### 5.6 Count Topological Sort Orderings

```python
def count_topological_sorts(n, edges):
    """
    Count possible topological orderings (backtracking)
    Warning: Exponential time complexity
    """
    graph = [[] for _ in range(n)]
    in_degree = [0] * n

    for u, v in edges:
        graph[u].append(v)
        in_degree[v] += 1

    count = 0

    def backtrack(path):
        nonlocal count

        if len(path) == n:
            count += 1
            return

        for i in range(n):
            if i not in path and in_degree[i] == 0:
                # Choose
                for neighbor in graph[i]:
                    in_degree[neighbor] -= 1
                path.add(i)

                backtrack(path)

                # Restore
                path.remove(i)
                for neighbor in graph[i]:
                    in_degree[neighbor] += 1

    backtrack(set())
    return count

# Example (use only for small n)
n = 4
edges = [(0, 1), (0, 2), (1, 3), (2, 3)]
print(count_topological_sorts(n, edges))  # 2: [0,1,2,3], [0,2,1,3]
```

### 5.7 Alien Dictionary

```python
def alien_order(words):
    """
    Infer alien alphabet order
    words: List of alien words sorted in dictionary order
    """
    from collections import defaultdict

    # Collect all characters
    chars = set()
    for word in words:
        chars.update(word)

    graph = defaultdict(set)
    in_degree = {c: 0 for c in chars}

    # Compare adjacent words to infer order
    for i in range(len(words) - 1):
        w1, w2 = words[i], words[i + 1]

        # Exception: w1 is prefix of w2 but longer is impossible
        if len(w1) > len(w2) and w1.startswith(w2):
            return ""

        for c1, c2 in zip(w1, w2):
            if c1 != c2:
                if c2 not in graph[c1]:
                    graph[c1].add(c2)
                    in_degree[c2] += 1
                break

    # Topological sort
    from collections import deque
    queue = deque([c for c in chars if in_degree[c] == 0])
    result = []

    while queue:
        c = queue.popleft()
        result.append(c)

        for neighbor in graph[c]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    if len(result) != len(chars):
        return ""  # Cycle (invalid input)

    return "".join(result)

# LeetCode 269. Alien Dictionary
words = ["wrt", "wrf", "er", "ett", "rftt"]
print(alien_order(words))  # "wertf"
```

---

## 6. Practice Problems

### Recommended Problems

| Difficulty | Problem | Platform | Type |
|--------|------|--------|------|
| ⭐⭐ | [Line Up](https://www.acmicpc.net/problem/2252) | BOJ | Basic topo sort |
| ⭐⭐ | [Course Schedule](https://leetcode.com/problems/course-schedule/) | LeetCode | Cycle detection |
| ⭐⭐ | [Course Schedule II](https://leetcode.com/problems/course-schedule-ii/) | LeetCode | Topo sort order |
| ⭐⭐⭐ | [Task](https://www.acmicpc.net/problem/2056) | BOJ | Minimum time |
| ⭐⭐⭐ | [Game Development](https://www.acmicpc.net/problem/1516) | BOJ | Task scheduling |
| ⭐⭐⭐ | [Problem Set](https://www.acmicpc.net/problem/1766) | BOJ | Lexicographic topo |
| ⭐⭐⭐⭐ | [Alien Dictionary](https://leetcode.com/problems/alien-dictionary/) | LeetCode | Order inference |

---

## Algorithm Comparison

```
┌─────────────────┬─────────────┬─────────────┬────────────────────┐
│ Algorithm       │ Time        │ Space       │ Characteristics    │
├─────────────────┼─────────────┼─────────────┼────────────────────┤
│ Kahn (BFS)      │ O(V + E)    │ O(V)        │ In-degree based    │
│ DFS-based       │ O(V + E)    │ O(V)        │ Finish reverse     │
├─────────────────┼─────────────┼─────────────┼────────────────────┤
│ Lexicographic   │ O((V+E)logV)│ O(V)        │ Uses min heap      │
└─────────────────┴─────────────┴─────────────┴────────────────────┘

V = number of vertices, E = number of edges
```

---

## Next Steps

- [14_Shortest_Path.md](./14_Shortest_Path.md) - Dijkstra, Bellman-Ford

---

## References

- [Topological Sorting](https://cp-algorithms.com/graph/topological-sort.html)
- Introduction to Algorithms (CLRS) - Chapter 22.4
