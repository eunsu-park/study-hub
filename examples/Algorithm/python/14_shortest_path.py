"""
Dijkstra's Algorithm
Dijkstra's Shortest Path Algorithm

Finds the single-source shortest path in a weighted graph.
Used for graphs without negative weights.
"""

import heapq
from collections import defaultdict
from typing import List, Dict, Tuple, Optional


# =============================================================================
# Weighted Graph Representation
# =============================================================================
def create_weighted_graph(edges: List[Tuple[int, int, int]], directed: bool = False) -> Dict[int, List[Tuple[int, int]]]:
    """
    Create weighted adjacency list from edge list (u, v, weight)
    graph[u] = [(v, weight), ...]
    """
    graph = defaultdict(list)
    for u, v, w in edges:
        graph[u].append((v, w))
        if not directed:
            graph[v].append((u, w))
    return graph


# =============================================================================
# 1. Basic Dijkstra Implementation
# =============================================================================
def dijkstra(graph: Dict[int, List[Tuple[int, int]]], start: int, n: int) -> List[int]:
    """
    Dijkstra's Algorithm (using priority queue)
    Time Complexity: O((V + E) log V)

    Args:
        graph: Adjacency list (node -> [(neighbor, weight), ...])
        start: Starting node
        n: Total number of nodes

    Returns:
        Array of shortest distances to each node (infinity if unreachable)
    """
    INF = float('inf')
    dist = [INF] * n
    dist[start] = 0

    # Min heap storing (distance, node) tuples
    pq = [(0, start)]

    while pq:
        d, u = heapq.heappop(pq)

        # Skip if already processed with shorter distance
        if d > dist[u]:
            continue

        # Check adjacent nodes
        for v, weight in graph[u]:
            new_dist = dist[u] + weight
            if new_dist < dist[v]:
                dist[v] = new_dist
                heapq.heappush(pq, (new_dist, v))

    return dist


# =============================================================================
# 2. Dijkstra + Path Tracking
# =============================================================================
def dijkstra_with_path(
    graph: Dict[int, List[Tuple[int, int]]],
    start: int,
    end: int,
    n: int
) -> Tuple[int, List[int]]:
    """
    Returns shortest distance along with the actual path
    """
    INF = float('inf')
    dist = [INF] * n
    dist[start] = 0
    parent = [-1] * n

    pq = [(0, start)]

    while pq:
        d, u = heapq.heappop(pq)

        if d > dist[u]:
            continue

        if u == end:
            break

        for v, weight in graph[u]:
            new_dist = dist[u] + weight
            if new_dist < dist[v]:
                dist[v] = new_dist
                parent[v] = u
                heapq.heappush(pq, (new_dist, v))

    # Reconstruct path
    if dist[end] == INF:
        return INF, []

    path = []
    node = end
    while node != -1:
        path.append(node)
        node = parent[node]
    path.reverse()

    return dist[end], path


# =============================================================================
# 3. All-Pairs Shortest Path (Floyd-Warshall)
# =============================================================================
def floyd_warshall(n: int, edges: List[Tuple[int, int, int]]) -> List[List[int]]:
    """
    Floyd-Warshall Algorithm
    Computes shortest distances between all pairs of vertices
    Time Complexity: O(V^3)
    """
    INF = float('inf')
    dist = [[INF] * n for _ in range(n)]

    # Distance to self is 0
    for i in range(n):
        dist[i][i] = 0

    # Apply edge information
    for u, v, w in edges:
        dist[u][v] = w

    # Consider paths through vertex k
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i][k] + dist[k][j] < dist[i][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]

    return dist


# =============================================================================
# 4. Bellman-Ford Algorithm (Allows Negative Weights)
# =============================================================================
def bellman_ford(n: int, edges: List[Tuple[int, int, int]], start: int) -> Tuple[List[int], bool]:
    """
    Bellman-Ford Algorithm
    Allows negative weights and detects negative cycles
    Time Complexity: O(VE)

    Returns:
        (shortest distance array, whether negative cycle exists)
    """
    INF = float('inf')
    dist = [INF] * n
    dist[start] = 0

    # Repeat V-1 times
    for _ in range(n - 1):
        for u, v, w in edges:
            if dist[u] != INF and dist[u] + w < dist[v]:
                dist[v] = dist[u] + w

    # Detect negative cycle (if update occurs on one more iteration, negative cycle exists)
    has_negative_cycle = False
    for u, v, w in edges:
        if dist[u] != INF and dist[u] + w < dist[v]:
            has_negative_cycle = True
            break

    return dist, has_negative_cycle


# =============================================================================
# 5. Network Delay Time Problem
# =============================================================================
def network_delay_time(times: List[List[int]], n: int, k: int) -> int:
    """
    Minimum time for signal to reach all nodes from node k
    times[i] = [source, target, time]
    Returns -1 if not all nodes are reachable
    """
    graph = defaultdict(list)
    for u, v, w in times:
        graph[u].append((v, w))

    dist = dijkstra(graph, k, n + 1)  # Node numbers start from 1

    # Maximum distance among nodes 1~n
    max_time = max(dist[1:n + 1])

    return max_time if max_time != float('inf') else -1


# =============================================================================
# 6. Kth Shortest Path
# =============================================================================
def kth_shortest_path(
    graph: Dict[int, List[Tuple[int, int]]],
    start: int,
    end: int,
    k: int,
    n: int
) -> int:
    """
    Return the length of the kth shortest path
    Returns -1 if not found
    """
    INF = float('inf')
    count = [0] * n  # Number of arrivals at each node
    pq = [(0, start)]

    while pq:
        d, u = heapq.heappop(pq)
        count[u] += 1

        if u == end and count[u] == k:
            return d

        # Don't expand nodes visited more than k times
        if count[u] > k:
            continue

        for v, weight in graph[u]:
            heapq.heappush(pq, (d + weight, v))

    return -1


# =============================================================================
# Tests
# =============================================================================
def main():
    print("=" * 60)
    print("Dijkstra & Shortest Path Algorithms")
    print("=" * 60)

    # Example graph
    #       1
    #    0 ---> 1
    #    |      |
    #  4 |      | 2
    #    v      v
    #    2 ---> 3
    #       3
    edges = [
        (0, 1, 1),
        (0, 2, 4),
        (1, 3, 2),
        (2, 3, 3),
        (1, 2, 2)
    ]

    print("\n[Graph Structure]")
    print("    0 --1--> 1")
    print("    |        |")
    print("    4        2")
    print("    v        v")
    print("    2 --3--> 3")
    print("    (1->2 weight 2)")

    # 1. Basic Dijkstra
    print("\n[1] Basic Dijkstra")
    graph = create_weighted_graph(edges, directed=True)
    dist = dijkstra(graph, 0, 4)
    print(f"    Start: 0")
    for i, d in enumerate(dist):
        print(f"    Distance to node {i}: {d}")

    # 2. Dijkstra + Path
    print("\n[2] Dijkstra + Path Tracking")
    distance, path = dijkstra_with_path(graph, 0, 3, 4)
    print(f"    0 -> 3 shortest distance: {distance}")
    print(f"    Path: {' -> '.join(map(str, path))}")

    # 3. Floyd-Warshall
    print("\n[3] Floyd-Warshall (All-Pairs Shortest Distance)")
    all_dist = floyd_warshall(4, edges)
    print("    Distance matrix:")
    for i, row in enumerate(all_dist):
        row_str = [str(d) if d != float('inf') else 'inf' for d in row]
        print(f"    {i}: {row_str}")

    # 4. Bellman-Ford
    print("\n[4] Bellman-Ford (Allows Negative Weights)")
    edges_negative = [(0, 1, 4), (0, 2, 5), (1, 2, -3), (2, 3, 4)]
    dist, has_neg_cycle = bellman_ford(4, edges_negative, 0)
    print(f"    Edges: {edges_negative}")
    print(f"    Shortest distances: {dist}")
    print(f"    Negative cycle: {has_neg_cycle}")

    # Negative cycle example
    edges_neg_cycle = [(0, 1, 1), (1, 2, -1), (2, 0, -1)]
    dist, has_neg_cycle = bellman_ford(3, edges_neg_cycle, 0)
    print(f"\n    Negative cycle graph: {edges_neg_cycle}")
    print(f"    Negative cycle exists: {has_neg_cycle}")

    # 5. Network Delay Time
    print("\n[5] Network Delay Time")
    times = [[2, 1, 1], [2, 3, 1], [3, 4, 1]]
    n, k = 4, 2
    result = network_delay_time(times, n, k)
    print(f"    times={times}, n={n}, k={k}")
    print(f"    Time to reach all nodes: {result}")

    # 6. Kth Shortest Path
    print("\n[6] Kth Shortest Path")
    edges_k = [(0, 1, 1), (0, 2, 3), (1, 2, 1), (1, 3, 2), (2, 3, 1)]
    graph_k = create_weighted_graph(edges_k, directed=True)
    for k in range(1, 4):
        dist = kth_shortest_path(graph_k, 0, 3, k, 4)
        print(f"    0->3 {k}th shortest path: {dist}")

    print("\n" + "=" * 60)
    print("Algorithm Comparison")
    print("=" * 60)
    print("""
    | Algorithm      | Time Complexity | Neg Weights | Use Case              |
    |---------------|----------------|-------------|------------------------|
    | Dijkstra      | O((V+E)log V)  | No          | Single-source shortest |
    | Bellman-Ford  | O(VE)          | Yes         | Neg weights/cycles     |
    | Floyd-Warshall| O(V^3)         | Yes*        | All-pairs shortest     |

    * Floyd-Warshall is also incorrect with negative cycles
    """)


if __name__ == "__main__":
    main()
