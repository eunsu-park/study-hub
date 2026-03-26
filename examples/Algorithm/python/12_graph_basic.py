"""
DFS (Depth-First Search) & BFS (Breadth-First Search)
Depth-First Search & Breadth-First Search

Two fundamental graph traversal algorithms.
"""

from collections import deque, defaultdict
from typing import List, Dict, Set, Optional


# =============================================================================
# Graph Representation
# =============================================================================
def create_adjacency_list(edges: List[List[int]], directed: bool = False) -> Dict[int, List[int]]:
    """Create adjacency list from edge list"""
    graph = defaultdict(list)
    for u, v in edges:
        graph[u].append(v)
        if not directed:
            graph[v].append(u)
    return graph


# =============================================================================
# 1. DFS (Recursive)
# =============================================================================
def dfs_recursive(graph: Dict[int, List[int]], start: int, visited: Set[int] = None) -> List[int]:
    """
    DFS recursive implementation
    Time Complexity: O(V + E), Space Complexity: O(V)
    """
    if visited is None:
        visited = set()

    result = []
    visited.add(start)
    result.append(start)

    for neighbor in graph[start]:
        if neighbor not in visited:
            result.extend(dfs_recursive(graph, neighbor, visited))

    return result


# =============================================================================
# 2. DFS (Stack)
# =============================================================================
def dfs_iterative(graph: Dict[int, List[int]], start: int) -> List[int]:
    """
    DFS iterative implementation (using stack)
    Time Complexity: O(V + E), Space Complexity: O(V)
    """
    visited = set()
    stack = [start]
    result = []

    while stack:
        node = stack.pop()
        if node not in visited:
            visited.add(node)
            result.append(node)
            # Add in reverse order to visit smaller numbers first
            for neighbor in reversed(graph[node]):
                if neighbor not in visited:
                    stack.append(neighbor)

    return result


# =============================================================================
# 3. BFS
# =============================================================================
def bfs(graph: Dict[int, List[int]], start: int) -> List[int]:
    """
    BFS implementation (using queue)
    Time Complexity: O(V + E), Space Complexity: O(V)
    """
    visited = set([start])
    queue = deque([start])
    result = []

    while queue:
        node = queue.popleft()
        result.append(node)

        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

    return result


# =============================================================================
# 4. Find Connected Components
# =============================================================================
def count_connected_components(n: int, edges: List[List[int]]) -> int:
    """
    Count connected components in an undirected graph
    """
    graph = create_adjacency_list(edges, directed=False)
    visited = set()
    count = 0

    for node in range(n):
        if node not in visited:
            # Visit all connected nodes via DFS
            stack = [node]
            while stack:
                curr = stack.pop()
                if curr not in visited:
                    visited.add(curr)
                    stack.extend(graph[curr])
            count += 1

    return count


# =============================================================================
# 5. Shortest Path (BFS) - Unweighted Graph
# =============================================================================
def shortest_path_bfs(graph: Dict[int, List[int]], start: int, end: int) -> Optional[List[int]]:
    """
    Find shortest path in an unweighted graph
    """
    if start == end:
        return [start]

    visited = set([start])
    queue = deque([(start, [start])])

    while queue:
        node, path = queue.popleft()

        for neighbor in graph[node]:
            if neighbor == end:
                return path + [neighbor]
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))

    return None  # No path found


# =============================================================================
# 6. 2D Grid Traversal
# =============================================================================
def num_islands(grid: List[List[str]]) -> int:
    """
    Count the number of islands (DFS)
    '1' = land, '0' = water
    """
    if not grid:
        return 0

    rows, cols = len(grid), len(grid[0])
    count = 0

    def dfs(r, c):
        if r < 0 or r >= rows or c < 0 or c >= cols or grid[r][c] == '0':
            return
        grid[r][c] = '0'  # Mark as visited
        dfs(r + 1, c)
        dfs(r - 1, c)
        dfs(r, c + 1)
        dfs(r, c - 1)

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == '1':
                count += 1
                dfs(r, c)

    return count


# =============================================================================
# 7. BFS by Level
# =============================================================================
def bfs_by_level(graph: Dict[int, List[int]], start: int) -> List[List[int]]:
    """
    BFS grouping nodes by level (depth)
    """
    visited = set([start])
    queue = deque([start])
    result = []

    while queue:
        level_size = len(queue)
        current_level = []

        for _ in range(level_size):
            node = queue.popleft()
            current_level.append(node)

            for neighbor in graph[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        result.append(current_level)

    return result


# =============================================================================
# 8. Cycle Detection (Undirected Graph)
# =============================================================================
def has_cycle_undirected(n: int, edges: List[List[int]]) -> bool:
    """
    Check for cycle in an undirected graph
    """
    graph = create_adjacency_list(edges, directed=False)
    visited = set()

    def dfs(node, parent):
        visited.add(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                if dfs(neighbor, node):
                    return True
            elif neighbor != parent:
                return True
        return False

    for node in range(n):
        if node not in visited:
            if dfs(node, -1):
                return True

    return False


# =============================================================================
# 9. Cycle Detection (Directed Graph)
# =============================================================================
def has_cycle_directed(n: int, edges: List[List[int]]) -> bool:
    """
    Check for cycle in a directed graph
    States: 0=unvisited, 1=in progress (current path), 2=completed
    """
    graph = create_adjacency_list(edges, directed=True)
    state = [0] * n  # 0: unvisited, 1: in progress, 2: completed

    def dfs(node):
        if state[node] == 1:  # Revisiting a node in progress = cycle
            return True
        if state[node] == 2:  # Already completed node
            return False

        state[node] = 1  # Start visiting

        for neighbor in graph[node]:
            if dfs(neighbor):
                return True

        state[node] = 2  # Complete visiting
        return False

    for node in range(n):
        if state[node] == 0:
            if dfs(node):
                return True

    return False


# =============================================================================
# Tests
# =============================================================================
def main():
    print("=" * 60)
    print("DFS & BFS Examples")
    print("=" * 60)

    # Create graph
    edges = [[0, 1], [0, 2], [1, 3], [1, 4], [2, 5], [2, 6]]
    graph = create_adjacency_list(edges)

    print("\n[Graph Structure]")
    print("       0")
    print("      / \\")
    print("     1   2")
    print("    /|   |\\")
    print("   3 4   5 6")

    # 1. DFS (Recursive)
    print("\n[1] DFS (Recursive)")
    result = dfs_recursive(graph, 0)
    print(f"    Start: 0, Traversal order: {result}")

    # 2. DFS (Iterative)
    print("\n[2] DFS (Iterative/Stack)")
    result = dfs_iterative(graph, 0)
    print(f"    Start: 0, Traversal order: {result}")

    # 3. BFS
    print("\n[3] BFS")
    result = bfs(graph, 0)
    print(f"    Start: 0, Traversal order: {result}")

    # 4. Connected Components
    print("\n[4] Connected Components Count")
    edges2 = [[0, 1], [1, 2], [3, 4]]
    count = count_connected_components(5, edges2)
    print(f"    5 nodes, edges: {edges2}")
    print(f"    Connected components: {count}")

    # 5. Shortest Path
    print("\n[5] Shortest Path (BFS)")
    path = shortest_path_bfs(graph, 0, 6)
    print(f"    0 -> 6 shortest path: {path}")

    # 6. Number of Islands
    print("\n[6] Number of Islands")
    grid = [
        ['1', '1', '0', '0', '0'],
        ['1', '1', '0', '0', '0'],
        ['0', '0', '1', '0', '0'],
        ['0', '0', '0', '1', '1']
    ]
    # Use copy (original gets modified)
    grid_copy = [row[:] for row in grid]
    count = num_islands(grid_copy)
    print(f"    Grid:")
    for row in grid:
        print(f"    {row}")
    print(f"    Number of islands: {count}")

    # 7. BFS by Level
    print("\n[7] BFS by Level")
    levels = bfs_by_level(graph, 0)
    for i, level in enumerate(levels):
        print(f"    Level {i}: {level}")

    # 8. Cycle Detection (Undirected)
    print("\n[8] Cycle Detection (Undirected Graph)")
    edges_no_cycle = [[0, 1], [1, 2], [2, 3]]
    edges_with_cycle = [[0, 1], [1, 2], [2, 0]]
    print(f"    Edges {edges_no_cycle}: cycle = {has_cycle_undirected(4, edges_no_cycle)}")
    print(f"    Edges {edges_with_cycle}: cycle = {has_cycle_undirected(3, edges_with_cycle)}")

    # 9. Cycle Detection (Directed)
    print("\n[9] Cycle Detection (Directed Graph)")
    edges_dag = [[0, 1], [1, 2], [0, 2]]
    edges_cycle = [[0, 1], [1, 2], [2, 0]]
    print(f"    DAG {edges_dag}: cycle = {has_cycle_directed(3, edges_dag)}")
    print(f"    Edges {edges_cycle}: cycle = {has_cycle_directed(3, edges_cycle)}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
