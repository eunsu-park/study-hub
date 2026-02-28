"""
Exercises for Lesson 25: Network Flow
Topic: Algorithm

Solutions to practice problems from the lesson.
Problems: Bipartite matching (Kuhn), Max flow (Edmonds-Karp), Min cut.
"""

from collections import deque


# === Exercise 1: Bipartite Matching (Kuhn's Algorithm) ===
# Problem: Find the maximum matching in a bipartite graph.
# Approach: Hungarian/Kuhn's algorithm using augmenting paths via DFS.

def exercise_1():
    """Solution: Kuhn's algorithm for maximum bipartite matching."""
    def max_matching(n_left, n_right, adj):
        """
        n_left: number of vertices on left side
        n_right: number of vertices on right side
        adj: adj[u] = list of right-side vertices u can be matched to
        Returns: (max matching count, match array for right side)
        """
        match_right = [-1] * n_right  # match_right[v] = which left node v is matched to

        def try_kuhn(u, visited):
            """Try to find an augmenting path starting from left node u."""
            for v in adj[u]:
                if visited[v]:
                    continue
                visited[v] = True

                # v is free, or v's current match can be rematched
                if match_right[v] == -1 or try_kuhn(match_right[v], visited):
                    match_right[v] = u
                    return True

            return False

        result = 0
        for u in range(n_left):
            visited = [False] * n_right
            if try_kuhn(u, visited):
                result += 1

        return result, match_right

    # Test case 1: Perfect matching possible
    # Left: {0, 1, 2}, Right: {0, 1, 2}
    # Edges: 0-0, 0-1, 1-0, 1-2, 2-1
    adj = [[0, 1], [0, 2], [1]]
    count, match = max_matching(3, 3, adj)
    print(f"Max matching: {count}")
    print(f"Match (right->left): {match}")
    assert count == 3

    # Test case 2: Not all can be matched
    # Left: {0, 1, 2}, Right: {0, 1}
    # 0-0, 1-0, 2-1
    adj = [[0], [0], [1]]
    count, match = max_matching(3, 2, adj)
    print(f"\nMax matching: {count}")
    assert count == 2

    # Test case 3: Job assignment
    # Workers: {0,1,2}, Jobs: {0,1,2}
    # 0 can do jobs 0,1; 1 can do job 2; 2 can do jobs 0,1
    adj = [[0, 1], [2], [0, 1]]
    count, match = max_matching(3, 3, adj)
    print(f"\nJob assignment, max: {count}")
    assert count == 3

    print("All Bipartite Matching tests passed!")


# === Exercise 2: Maximum Flow (Edmonds-Karp / BFS Ford-Fulkerson) ===
# Problem: Find the maximum flow from source to sink in a flow network.
# Approach: BFS to find shortest augmenting paths (Edmonds-Karp variant).

def exercise_2():
    """Solution: Edmonds-Karp algorithm for max flow in O(VE^2)."""
    def max_flow_edmonds_karp(n, adj, capacity, source, sink):
        """
        n: number of vertices
        adj: adjacency list (including reverse edges)
        capacity: capacity[u][v] = capacity of edge u->v
        source, sink: source and sink vertices
        Returns: maximum flow value
        """
        def bfs(source, sink, parent):
            """Find an augmenting path using BFS. Return True if path found."""
            visited = [False] * n
            visited[source] = True
            queue = deque([source])

            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    if not visited[v] and capacity[u][v] > 0:
                        visited[v] = True
                        parent[v] = u
                        if v == sink:
                            return True
                        queue.append(v)

            return False

        total_flow = 0

        while True:
            parent = [-1] * n
            if not bfs(source, sink, parent):
                break

            # Find bottleneck (minimum residual capacity along the path)
            path_flow = float('inf')
            v = sink
            while v != source:
                u = parent[v]
                path_flow = min(path_flow, capacity[u][v])
                v = u

            # Update capacities (forward and reverse)
            v = sink
            while v != source:
                u = parent[v]
                capacity[u][v] -= path_flow
                capacity[v][u] += path_flow
                v = u

            total_flow += path_flow

        return total_flow

    # Test case 1: Simple flow network
    #   0 --10--> 1 --5--> 3
    #   |         |        ^
    #   +--5----> 2 --15--+
    n = 4
    adj = [[] for _ in range(n)]
    cap = [[0] * n for _ in range(n)]

    def add_edge(u, v, c):
        adj[u].append(v)
        adj[v].append(u)  # reverse edge for residual graph
        cap[u][v] += c

    add_edge(0, 1, 10)
    add_edge(0, 2, 5)
    add_edge(1, 2, 4)
    add_edge(1, 3, 5)
    add_edge(2, 3, 15)

    result = max_flow_edmonds_karp(n, adj, cap, 0, 3)
    print(f"Max flow (0 -> 3): {result}")
    assert result == 14  # 0->1->3 (5) + 0->1->2->3 (4) + 0->2->3 (5) = 14

    # Test case 2
    n = 6
    adj = [[] for _ in range(n)]
    cap = [[0] * n for _ in range(n)]

    add_edge(0, 1, 16)
    add_edge(0, 2, 13)
    add_edge(1, 2, 10)
    add_edge(1, 3, 12)
    add_edge(2, 1, 4)
    add_edge(2, 4, 14)
    add_edge(3, 2, 9)
    add_edge(3, 5, 20)
    add_edge(4, 3, 7)
    add_edge(4, 5, 4)

    result = max_flow_edmonds_karp(n, adj, cap, 0, 5)
    print(f"Max flow (0 -> 5): {result}")
    assert result == 23

    print("All Max Flow tests passed!")


# === Exercise 3: Minimum Cut via Max Flow ===
# Problem: Find the minimum cut (min set of edges to disconnect source from sink).
# By max-flow min-cut theorem, min cut = max flow.
# After running max flow, vertices reachable from source in residual graph form one side.

def exercise_3():
    """Solution: Min cut = max flow, identify the cut edges."""
    def min_cut(n, edges, source, sink):
        """
        Returns: (min_cut_value, list of cut edges)
        """
        adj = [[] for _ in range(n)]
        cap = [[0] * n for _ in range(n)]

        for u, v, c in edges:
            adj[u].append(v)
            adj[v].append(u)
            cap[u][v] += c

        # Run Edmonds-Karp to find max flow
        def bfs_path(source, sink, parent):
            visited = [False] * n
            visited[source] = True
            queue = deque([source])
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    if not visited[v] and cap[u][v] > 0:
                        visited[v] = True
                        parent[v] = u
                        if v == sink:
                            return True
                        queue.append(v)
            return False

        total_flow = 0
        while True:
            parent = [-1] * n
            if not bfs_path(source, sink, parent):
                break
            path_flow = float('inf')
            v = sink
            while v != source:
                u = parent[v]
                path_flow = min(path_flow, cap[u][v])
                v = u
            v = sink
            while v != source:
                u = parent[v]
                cap[u][v] -= path_flow
                cap[v][u] += path_flow
                v = u
            total_flow += path_flow

        # Find vertices reachable from source in residual graph
        visited = [False] * n
        queue = deque([source])
        visited[source] = True
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if not visited[v] and cap[u][v] > 0:
                    visited[v] = True
                    queue.append(v)

        # Cut edges: from reachable to non-reachable with original capacity > 0
        cut_edges = []
        for u, v, c in edges:
            if visited[u] and not visited[v] and c > 0:
                cut_edges.append((u, v, c))

        return total_flow, cut_edges

    # Test case
    edges = [
        (0, 1, 3), (0, 2, 2),
        (1, 2, 1), (1, 3, 3),
        (2, 3, 2),
    ]

    flow, cuts = min_cut(4, edges, 0, 3)
    print(f"Min cut value: {flow}")
    print(f"Cut edges: {cuts}")
    assert flow == 5  # max flow = 5
    # The sum of cut edge capacities equals the max flow
    assert sum(c for _, _, c in cuts) == flow

    print("All Min Cut tests passed!")


if __name__ == "__main__":
    print("=== Exercise 1: Bipartite Matching ===")
    exercise_1()
    print("\n=== Exercise 2: Maximum Flow (Edmonds-Karp) ===")
    exercise_2()
    print("\n=== Exercise 3: Minimum Cut ===")
    exercise_3()
    print("\nAll exercises completed!")
