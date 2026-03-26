"""
Exercises for Lesson 14: Shortest Path
Topic: Algorithm

Solutions to practice problems from the lesson.
"""

import heapq


# === Exercise 1: Find Cities at Exact Distance K ===
# Problem: Find all cities exactly K distance from start city.
# Approach: Dijkstra (or BFS for unit weights) to compute distances,
#   then filter for distance == k.

def exercise_1():
    """Solution: Dijkstra for shortest distances, then filter."""
    def cities_at_distance_k(n, edges, start, k):
        """
        n: number of cities (1-indexed)
        edges: list of (a, b) directed edges with weight 1
        start: starting city
        k: target distance
        Returns: sorted list of cities at distance k, or [-1] if none
        """
        graph = [[] for _ in range(n + 1)]
        for a, b in edges:
            graph[a].append(b)

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

    # Test case 1
    # Graph: 1->2, 1->3, 2->3, 2->4, 4->5
    # Distances from 1: d(2)=1, d(3)=1 (direct), d(4)=2, d(5)=3
    edges = [(1, 2), (1, 3), (2, 3), (2, 4), (4, 5)]
    result = cities_at_distance_k(5, edges, 1, 2)
    print(f"Cities at distance 2 from 1: {result}")
    assert result == [4]  # 1->2->4 (d=2)

    # Test case 2
    result = cities_at_distance_k(5, edges, 1, 3)
    print(f"Cities at distance 3 from 1: {result}")
    assert result == [5]  # 1->2->4->5 (d=3)

    # Test case 3: no city at that distance
    result = cities_at_distance_k(5, edges, 1, 10)
    print(f"Cities at distance 10 from 1: {result}")
    assert result == [-1]

    print("All Cities at Distance K tests passed!")


# === Exercise 2: Negative Cycle Detection ===
# Problem: Check if a negative cycle exists using Bellman-Ford.
# Approach: Run Bellman-Ford for V-1 iterations, then check if any edge
#   can still be relaxed (indicating a negative cycle).

def exercise_2():
    """Solution: Bellman-Ford negative cycle detection."""
    def has_negative_cycle(V, edges):
        """
        V: number of vertices (0-indexed)
        edges: list of (u, v, w) directed edges
        Returns: True if a negative cycle exists
        """
        # Initialize distances from all vertices to 0 (this allows detection
        # of negative cycles even if not reachable from a single source)
        dist = [0] * V

        # Relax all edges V-1 times
        for _ in range(V - 1):
            for u, v, w in edges:
                if dist[u] + w < dist[v]:
                    dist[v] = dist[u] + w

        # Check for negative cycle: if any edge can still be relaxed
        for u, v, w in edges:
            if dist[u] + w < dist[v]:
                return True

        return False

    # Test case 1: no negative cycle
    # 0 -> 1 (w=1), 1 -> 2 (w=2), 2 -> 0 (w=3)
    edges1 = [(0, 1, 1), (1, 2, 2), (2, 0, 3)]
    result = has_negative_cycle(3, edges1)
    print(f"Cycle sum=6 (positive): has_negative_cycle = {result}")
    assert result is False

    # Test case 2: negative cycle exists
    # 0 -> 1 (w=1), 1 -> 2 (w=-3), 2 -> 0 (w=1)
    # Cycle: 0->1->2->0 with sum = 1 + (-3) + 1 = -1
    edges2 = [(0, 1, 1), (1, 2, -3), (2, 0, 1)]
    result = has_negative_cycle(3, edges2)
    print(f"Cycle sum=-1 (negative): has_negative_cycle = {result}")
    assert result is True

    # Test case 3: negative weight but no negative cycle
    edges3 = [(0, 1, -2), (1, 2, 3), (0, 2, 5)]
    result = has_negative_cycle(3, edges3)
    print(f"Negative weight, no cycle: has_negative_cycle = {result}")
    assert result is False

    # Test case 4: self-loop with negative weight
    edges4 = [(0, 0, -1)]
    result = has_negative_cycle(1, edges4)
    print(f"Self-loop negative: has_negative_cycle = {result}")
    assert result is True

    print("All Negative Cycle Detection tests passed!")


# === Exercise 3: Dijkstra Single-Source Shortest Path ===
# Problem: Find shortest distances from source to all vertices.
# From the recommended problems section.

def exercise_3():
    """Solution: Dijkstra with priority queue."""
    def dijkstra(n, adj, start):
        """
        n: number of vertices (1-indexed)
        adj: adjacency list, adj[u] = [(v, weight), ...]
        start: source vertex
        Returns: list of shortest distances from start
        """
        dist = [float('inf')] * (n + 1)
        dist[start] = 0
        pq = [(0, start)]

        while pq:
            d, u = heapq.heappop(pq)
            if d > dist[u]:
                continue
            for v, w in adj[u]:
                if dist[u] + w < dist[v]:
                    dist[v] = dist[u] + w
                    heapq.heappush(pq, (dist[v], v))

        return dist

    # Build graph
    # 1 --(2)--> 2 --(3)--> 3
    # |                      ^
    # +-------(10)----------+
    # 1 --(1)--> 4 --(4)--> 3
    n = 4
    adj = [[] for _ in range(n + 1)]
    edges = [(1, 2, 2), (2, 3, 3), (1, 3, 10), (1, 4, 1), (4, 3, 4)]
    for u, v, w in edges:
        adj[u].append((v, w))

    dist = dijkstra(n, adj, 1)
    print(f"Shortest distances from vertex 1:")
    for i in range(1, n + 1):
        d = dist[i] if dist[i] != float('inf') else "INF"
        print(f"  to {i}: {d}")

    assert dist[1] == 0
    assert dist[2] == 2   # 1->2
    assert dist[3] == 5   # 1->2->3 (2+3=5) or 1->4->3 (1+4=5)
    assert dist[4] == 1   # 1->4

    print("All Dijkstra tests passed!")


if __name__ == "__main__":
    print("=== Exercise 1: Cities at Distance K ===")
    exercise_1()
    print("\n=== Exercise 2: Negative Cycle Detection ===")
    exercise_2()
    print("\n=== Exercise 3: Dijkstra Shortest Path ===")
    exercise_3()
    print("\nAll exercises completed!")
