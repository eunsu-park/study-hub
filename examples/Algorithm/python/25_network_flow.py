"""
Network Flow
Maximum Flow and Bipartite Matching

Algorithms for computing maximum flow in graphs.
"""

from typing import List, Tuple, Dict
from collections import defaultdict, deque


# =============================================================================
# 1. Ford-Fulkerson (DFS-Based)
# =============================================================================

def ford_fulkerson(capacity: List[List[int]], source: int, sink: int) -> int:
    """
    Ford-Fulkerson Algorithm (DFS)
    Time Complexity: O(E * max_flow)
    """
    n = len(capacity)
    residual = [row[:] for row in capacity]

    def dfs(node: int, flow: int, visited: List[bool]) -> int:
        if node == sink:
            return flow

        visited[node] = True

        for next_node in range(n):
            if not visited[next_node] and residual[node][next_node] > 0:
                min_flow = min(flow, residual[node][next_node])
                result = dfs(next_node, min_flow, visited)

                if result > 0:
                    residual[node][next_node] -= result
                    residual[next_node][node] += result
                    return result

        return 0

    max_flow = 0
    while True:
        visited = [False] * n
        flow = dfs(source, float('inf'), visited)
        if flow == 0:
            break
        max_flow += flow

    return max_flow


# =============================================================================
# 2. Edmonds-Karp (BFS-Based)
# =============================================================================

def edmonds_karp(capacity: List[List[int]], source: int, sink: int) -> int:
    """
    Edmonds-Karp Algorithm (BFS)
    Time Complexity: O(V * E^2)
    """
    n = len(capacity)
    residual = [row[:] for row in capacity]

    def bfs() -> List[int]:
        """Find augmenting path via BFS"""
        parent = [-1] * n
        visited = [False] * n
        visited[source] = True
        queue = deque([source])

        while queue:
            node = queue.popleft()

            for next_node in range(n):
                if not visited[next_node] and residual[node][next_node] > 0:
                    visited[next_node] = True
                    parent[next_node] = node
                    queue.append(next_node)

                    if next_node == sink:
                        return parent

        return parent

    max_flow = 0

    while True:
        parent = bfs()

        if parent[sink] == -1:
            break

        # Find minimum capacity along the path
        path_flow = float('inf')
        node = sink
        while node != source:
            prev = parent[node]
            path_flow = min(path_flow, residual[prev][node])
            node = prev

        # Update residual graph
        node = sink
        while node != source:
            prev = parent[node]
            residual[prev][node] -= path_flow
            residual[node][prev] += path_flow
            node = prev

        max_flow += path_flow

    return max_flow


# =============================================================================
# 3. Dinic's Algorithm
# =============================================================================

class Dinic:
    """
    Dinic's Algorithm
    Time Complexity: O(V^2 * E)
    For bipartite graphs: O(E * sqrt(V))
    """

    def __init__(self, n: int):
        self.n = n
        self.graph = defaultdict(list)

    def add_edge(self, u: int, v: int, cap: int):
        """Add edge (u -> v, capacity cap)"""
        # (adjacent node, residual capacity, reverse edge index)
        self.graph[u].append([v, cap, len(self.graph[v])])
        self.graph[v].append([u, 0, len(self.graph[u]) - 1])

    def bfs(self, source: int, sink: int) -> bool:
        """Build level graph"""
        self.level = [-1] * self.n
        self.level[source] = 0
        queue = deque([source])

        while queue:
            node = queue.popleft()
            for next_node, cap, _ in self.graph[node]:
                if cap > 0 and self.level[next_node] < 0:
                    self.level[next_node] = self.level[node] + 1
                    queue.append(next_node)

        return self.level[sink] >= 0

    def dfs(self, node: int, sink: int, flow: int) -> int:
        """Find blocking flow"""
        if node == sink:
            return flow

        for i in range(self.iter[node], len(self.graph[node])):
            self.iter[node] = i
            next_node, cap, rev = self.graph[node][i]

            if cap > 0 and self.level[next_node] == self.level[node] + 1:
                d = self.dfs(next_node, sink, min(flow, cap))

                if d > 0:
                    self.graph[node][i][1] -= d
                    self.graph[next_node][rev][1] += d
                    return d

        return 0

    def max_flow(self, source: int, sink: int) -> int:
        """Compute maximum flow"""
        flow = 0

        while self.bfs(source, sink):
            self.iter = [0] * self.n

            while True:
                f = self.dfs(source, sink, float('inf'))
                if f == 0:
                    break
                flow += f

        return flow


# =============================================================================
# 4. Bipartite Matching
# =============================================================================

def bipartite_matching(n: int, m: int, edges: List[Tuple[int, int]]) -> int:
    """
    Maximum bipartite matching (simplified Hopcroft-Karp)
    n: number of left vertices, m: number of right vertices
    edges: [(left, right), ...]
    Time Complexity: O(E * sqrt(V))
    """
    # Build graph
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)

    match_left = [-1] * n   # Left vertex matching
    match_right = [-1] * m  # Right vertex matching

    def dfs(u: int, visited: List[bool]) -> bool:
        for v in adj[u]:
            if visited[v]:
                continue
            visited[v] = True

            # Unmatched or can be rematched
            if match_right[v] == -1 or dfs(match_right[v], visited):
                match_left[u] = v
                match_right[v] = u
                return True

        return False

    matching = 0
    for u in range(n):
        visited = [False] * m
        if dfs(u, visited):
            matching += 1

    return matching


# =============================================================================
# 5. Minimum Cut
# =============================================================================

def min_cut(capacity: List[List[int]], source: int, sink: int) -> Tuple[int, List[int]]:
    """
    Minimum cut = Maximum flow
    Returns: (cut capacity, source-side vertex set)
    """
    n = len(capacity)
    residual = [row[:] for row in capacity]

    # Compute max flow using Edmonds-Karp
    def bfs_flow() -> bool:
        parent = [-1] * n
        visited = [False] * n
        visited[source] = True
        queue = deque([source])

        while queue:
            node = queue.popleft()
            for next_node in range(n):
                if not visited[next_node] and residual[node][next_node] > 0:
                    visited[next_node] = True
                    parent[next_node] = node
                    queue.append(next_node)

        if parent[sink] == -1:
            return False

        path_flow = float('inf')
        node = sink
        while node != source:
            prev = parent[node]
            path_flow = min(path_flow, residual[prev][node])
            node = prev

        node = sink
        while node != source:
            prev = parent[node]
            residual[prev][node] -= path_flow
            residual[node][prev] += path_flow
            node = prev

        return True

    while bfs_flow():
        pass

    # Find vertices reachable from source in residual graph
    visited = [False] * n
    queue = deque([source])
    visited[source] = True

    while queue:
        node = queue.popleft()
        for next_node in range(n):
            if not visited[next_node] and residual[node][next_node] > 0:
                visited[next_node] = True
                queue.append(next_node)

    # Compute cut capacity
    cut_capacity = 0
    source_side = []
    for i in range(n):
        if visited[i]:
            source_side.append(i)
            for j in range(n):
                if not visited[j] and capacity[i][j] > 0:
                    cut_capacity += capacity[i][j]

    return cut_capacity, source_side


# =============================================================================
# 6. Practical Problem: Task Assignment
# =============================================================================

def assign_tasks(workers: int, tasks: int, can_do: List[List[int]]) -> List[int]:
    """
    Assign tasks to workers (bipartite matching)
    can_do[i] = list of tasks worker i can perform
    Returns: assignment[worker] = assigned task (-1 if unassigned)
    """
    edges = []
    for worker, tasks_list in enumerate(can_do):
        for task in tasks_list:
            edges.append((worker, task))

    # Perform bipartite matching
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)

    match_worker = [-1] * workers
    match_task = [-1] * tasks

    def dfs(u: int, visited: List[bool]) -> bool:
        for v in adj[u]:
            if visited[v]:
                continue
            visited[v] = True

            if match_task[v] == -1 or dfs(match_task[v], visited):
                match_worker[u] = v
                match_task[v] = u
                return True

        return False

    for u in range(workers):
        visited = [False] * tasks
        dfs(u, visited)

    return match_worker


# =============================================================================
# 7. Practical Problem: Maximum Edge-Disjoint Paths
# =============================================================================

def max_edge_disjoint_paths(n: int, edges: List[Tuple[int, int]], source: int, sink: int) -> int:
    """
    Maximum number of edge-disjoint paths
    Set each edge capacity = 1 and compute max flow
    """
    capacity = [[0] * n for _ in range(n)]

    for u, v in edges:
        capacity[u][v] = 1
        capacity[v][u] = 1  # For undirected graphs

    return edmonds_karp(capacity, source, sink)


# =============================================================================
# Tests
# =============================================================================

def main():
    print("=" * 60)
    print("Network Flow Examples")
    print("=" * 60)

    # Graph example:
    #     10      10
    #  0 ---> 1 ---> 3
    #  |      |      ^
    #  |10    |2     |10
    #  v      v      |
    #  2 ---> 4 ---> 5
    #     10      10

    # 1. Ford-Fulkerson
    print("\n[1] Ford-Fulkerson")
    capacity = [
        [0, 10, 10, 0, 0, 0],  # 0
        [0, 0, 2, 10, 0, 0],   # 1
        [0, 0, 0, 0, 10, 0],   # 2
        [0, 0, 0, 0, 0, 10],   # 3
        [0, 0, 0, 0, 0, 10],   # 4
        [0, 0, 0, 0, 0, 0]     # 5 (sink)
    ]
    flow = ford_fulkerson(capacity, 0, 5)
    print(f"    source=0, sink=5")
    print(f"    Maximum flow: {flow}")

    # 2. Edmonds-Karp
    print("\n[2] Edmonds-Karp")
    flow = edmonds_karp(capacity, 0, 5)
    print(f"    Maximum flow: {flow}")

    # 3. Dinic
    print("\n[3] Dinic's Algorithm")
    dinic = Dinic(6)
    dinic.add_edge(0, 1, 10)
    dinic.add_edge(0, 2, 10)
    dinic.add_edge(1, 2, 2)
    dinic.add_edge(1, 3, 10)
    dinic.add_edge(2, 4, 10)
    dinic.add_edge(3, 5, 10)
    dinic.add_edge(4, 5, 10)
    flow = dinic.max_flow(0, 5)
    print(f"    Maximum flow: {flow}")

    # 4. Bipartite Matching
    print("\n[4] Bipartite Matching")
    # Left: 0, 1, 2 (workers)
    # Right: 0, 1, 2 (tasks)
    edges = [(0, 0), (0, 1), (1, 0), (1, 2), (2, 1)]
    matching = bipartite_matching(3, 3, edges)
    print(f"    Edges: {edges}")
    print(f"    Maximum matching: {matching}")

    # 5. Minimum Cut
    print("\n[5] Minimum Cut")
    cut_cap, source_side = min_cut(capacity, 0, 5)
    print(f"    Minimum cut capacity: {cut_cap}")
    print(f"    Source-side vertices: {source_side}")

    # 6. Task Assignment
    print("\n[6] Task Assignment")
    can_do = [
        [0, 1],     # Worker 0: can do tasks 0, 1
        [1, 2],     # Worker 1: can do tasks 1, 2
        [0, 2]      # Worker 2: can do tasks 0, 2
    ]
    assignment = assign_tasks(3, 3, can_do)
    print(f"    Capabilities: {can_do}")
    print(f"    Assignment: {assignment}")

    # 7. Algorithm Comparison
    print("\n[7] Algorithm Complexity Comparison")
    print("    | Algorithm      | Time Complexity    | Notes              |")
    print("    |----------------|--------------------|--------------------|")
    print("    | Ford-Fulkerson | O(E * max_flow)    | DFS, integer cap   |")
    print("    | Edmonds-Karp   | O(V * E^2)         | BFS, stable        |")
    print("    | Dinic          | O(V^2 * E)         | Level graph        |")
    print("    | Bipartite      | O(E * sqrt(V))     | Dinic special case |")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
