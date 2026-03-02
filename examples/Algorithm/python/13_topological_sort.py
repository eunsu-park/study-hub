"""
Topological Sort
Topological Sorting

Arranges vertices of a Directed Acyclic Graph (DAG) in linear order.
"""

from typing import List, Dict, Set, Optional, Tuple
from collections import deque, defaultdict


# =============================================================================
# 1. Kahn's Algorithm (BFS-based)
# =============================================================================

def topological_sort_kahn(n: int, edges: List[Tuple[int, int]]) -> List[int]:
    """
    Kahn's Algorithm (in-degree based)
    Time Complexity: O(V + E)
    Space Complexity: O(V + E)

    edges: [(from, to), ...] - from -> to dependency
    Returns: topologically sorted order, empty list if cycle exists
    """
    # Build graph and in-degree array
    graph = defaultdict(list)
    in_degree = [0] * n

    for u, v in edges:
        graph[u].append(v)
        in_degree[v] += 1

    # Start with nodes having in-degree 0
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    result = []

    while queue:
        node = queue.popleft()
        result.append(node)

        for neighbor in graph[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    # Check for cycle by verifying all nodes were visited
    return result if len(result) == n else []


# =============================================================================
# 2. DFS-based Topological Sort
# =============================================================================

def topological_sort_dfs(n: int, edges: List[Tuple[int, int]]) -> List[int]:
    """
    DFS-based Topological Sort
    Time Complexity: O(V + E)
    Space Complexity: O(V + E)
    """
    graph = defaultdict(list)
    for u, v in edges:
        graph[u].append(v)

    WHITE, GRAY, BLACK = 0, 1, 2
    color = [WHITE] * n
    result = []
    has_cycle = False

    def dfs(node: int) -> None:
        nonlocal has_cycle

        if has_cycle:
            return

        color[node] = GRAY  # In progress

        for neighbor in graph[node]:
            if color[neighbor] == GRAY:  # Cycle detected
                has_cycle = True
                return
            if color[neighbor] == WHITE:
                dfs(neighbor)

        color[node] = BLACK  # Completed
        result.append(node)

    for i in range(n):
        if color[i] == WHITE:
            dfs(i)

    return result[::-1] if not has_cycle else []


# =============================================================================
# 3. Cycle Detection
# =============================================================================

def has_cycle(n: int, edges: List[Tuple[int, int]]) -> bool:
    """
    Check for cycle in a directed graph
    Time Complexity: O(V + E)
    """
    graph = defaultdict(list)
    for u, v in edges:
        graph[u].append(v)

    WHITE, GRAY, BLACK = 0, 1, 2
    color = [WHITE] * n

    def dfs(node: int) -> bool:
        color[node] = GRAY

        for neighbor in graph[node]:
            if color[neighbor] == GRAY:
                return True  # Cycle
            if color[neighbor] == WHITE and dfs(neighbor):
                return True

        color[node] = BLACK
        return False

    for i in range(n):
        if color[i] == WHITE and dfs(i):
            return True

    return False


# =============================================================================
# 4. Find All Topological Sort Orders
# =============================================================================

def all_topological_sorts(n: int, edges: List[Tuple[int, int]]) -> List[List[int]]:
    """
    Find all possible topological sort orders
    Time Complexity: O(V! * (V + E)) - worst case
    """
    graph = defaultdict(list)
    in_degree = [0] * n

    for u, v in edges:
        graph[u].append(v)
        in_degree[v] += 1

    result = []
    path = []
    visited = [False] * n

    def backtrack():
        # All nodes visited
        if len(path) == n:
            result.append(path.copy())
            return

        for i in range(n):
            if not visited[i] and in_degree[i] == 0:
                # Choose
                visited[i] = True
                path.append(i)
                for neighbor in graph[i]:
                    in_degree[neighbor] -= 1

                backtrack()

                # Restore
                visited[i] = False
                path.pop()
                for neighbor in graph[i]:
                    in_degree[neighbor] += 1

    backtrack()
    return result


# =============================================================================
# 5. Practical Problem: Course Schedule
# =============================================================================

def can_finish(num_courses: int, prerequisites: List[List[int]]) -> bool:
    """
    Check if all courses can be completed (no cycle = possible)
    prerequisites: [course, prereq] - prereq -> course
    """
    edges = [(prereq, course) for course, prereq in prerequisites]
    return not has_cycle(num_courses, edges)


def find_order(num_courses: int, prerequisites: List[List[int]]) -> List[int]:
    """
    Return the course completion order
    """
    edges = [(prereq, course) for course, prereq in prerequisites]
    return topological_sort_kahn(num_courses, edges)


# =============================================================================
# 6. Practical Problem: Alien Dictionary
# =============================================================================

def alien_order(words: List[str]) -> str:
    """
    Determine alien alphabet order
    Assumes the word list is sorted in lexicographic order
    """
    # Collect all characters
    chars = set()
    for word in words:
        chars.update(word)

    # Build graph
    graph = defaultdict(set)
    in_degree = defaultdict(int)
    for char in chars:
        in_degree[char] = 0

    for i in range(len(words) - 1):
        word1, word2 = words[i], words[i + 1]

        # Check for invalid ordering
        if len(word1) > len(word2) and word1.startswith(word2):
            return ""

        # Determine order from first different character
        for c1, c2 in zip(word1, word2):
            if c1 != c2:
                if c2 not in graph[c1]:
                    graph[c1].add(c2)
                    in_degree[c2] += 1
                break

    # Topological sort
    queue = deque([c for c in chars if in_degree[c] == 0])
    result = []

    while queue:
        char = queue.popleft()
        result.append(char)

        for neighbor in graph[char]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    return ''.join(result) if len(result) == len(chars) else ""


# =============================================================================
# 7. Practical Problem: Parallel Courses
# =============================================================================

def minimum_semesters(n: int, relations: List[List[int]]) -> int:
    """
    Minimum number of semesters to complete all courses
    Parallel enrollment allowed, relations: [prev, next]
    """
    graph = defaultdict(list)
    in_degree = [0] * (n + 1)

    for prev_course, next_course in relations:
        graph[prev_course].append(next_course)
        in_degree[next_course] += 1

    # Start with nodes having in-degree 0 (1-indexed)
    queue = deque([i for i in range(1, n + 1) if in_degree[i] == 0])
    semesters = 0
    completed = 0

    while queue:
        semesters += 1
        next_queue = deque()

        while queue:
            course = queue.popleft()
            completed += 1

            for next_course in graph[course]:
                in_degree[next_course] -= 1
                if in_degree[next_course] == 0:
                    next_queue.append(next_course)

        queue = next_queue

    return semesters if completed == n else -1


# =============================================================================
# 8. Practical Problem: Longest Path in DAG
# =============================================================================

def longest_path_dag(n: int, edges: List[Tuple[int, int, int]]) -> List[int]:
    """
    Longest path to each node in a DAG (with weights)
    edges: [(from, to, weight), ...]
    """
    graph = defaultdict(list)
    in_degree = [0] * n

    for u, v, w in edges:
        graph[u].append((v, w))
        in_degree[v] += 1

    # Topological sort
    topo_order = []
    queue = deque([i for i in range(n) if in_degree[i] == 0])

    while queue:
        node = queue.popleft()
        topo_order.append(node)
        for neighbor, _ in graph[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    # Calculate longest path
    dist = [0] * n

    for node in topo_order:
        for neighbor, weight in graph[node]:
            dist[neighbor] = max(dist[neighbor], dist[node] + weight)

    return dist


# =============================================================================
# 9. Practical Problem: Build Order
# =============================================================================

def build_order(projects: List[str], dependencies: List[Tuple[str, str]]) -> List[str]:
    """
    Determine build order
    dependencies: [(proj, depends_on), ...] - proj depends on depends_on
    """
    # Project index mapping
    proj_to_idx = {p: i for i, p in enumerate(projects)}
    n = len(projects)

    # Convert edges (depends_on -> proj)
    edges = [(proj_to_idx[dep], proj_to_idx[proj]) for proj, dep in dependencies]

    # Topological sort
    order = topological_sort_kahn(n, edges)

    if not order:
        return []  # Cycle exists

    return [projects[i] for i in order]


# =============================================================================
# Tests
# =============================================================================

def main():
    print("=" * 60)
    print("Topological Sort Examples")
    print("=" * 60)

    # 1. Kahn's Algorithm
    print("\n[1] Kahn's Algorithm (BFS)")
    n = 6
    edges = [(5, 2), (5, 0), (4, 0), (4, 1), (2, 3), (3, 1)]
    result = topological_sort_kahn(n, edges)
    print(f"    Nodes: 0-5, Edges: {edges}")
    print(f"    Topological sort: {result}")

    # 2. DFS-based
    print("\n[2] DFS-based Topological Sort")
    result = topological_sort_dfs(n, edges)
    print(f"    Topological sort: {result}")

    # 3. Cycle Detection
    print("\n[3] Cycle Detection")
    cyclic_edges = [(0, 1), (1, 2), (2, 0)]
    acyclic_edges = [(0, 1), (1, 2), (0, 2)]
    print(f"    {cyclic_edges}: cycle {has_cycle(3, cyclic_edges)}")
    print(f"    {acyclic_edges}: cycle {has_cycle(3, acyclic_edges)}")

    # 4. All Topological Sorts
    print("\n[4] All Topological Sort Orders")
    n = 4
    edges = [(0, 1), (0, 2), (1, 3), (2, 3)]
    all_orders = all_topological_sorts(n, edges)
    print(f"    Nodes: 0-3, Edges: {edges}")
    print(f"    All orders: {all_orders}")

    # 5. Course Schedule
    print("\n[5] Course Schedule")
    num_courses = 4
    prerequisites = [[1, 0], [2, 0], [3, 1], [3, 2]]
    can = can_finish(num_courses, prerequisites)
    order = find_order(num_courses, prerequisites)
    print(f"    Courses: {num_courses}, Prerequisites: {prerequisites}")
    print(f"    Can finish: {can}")
    print(f"    Order: {order}")

    # 6. Alien Dictionary
    print("\n[6] Alien Dictionary")
    words = ["wrt", "wrf", "er", "ett", "rftt"]
    order = alien_order(words)
    print(f"    Words: {words}")
    print(f"    Alphabet order: {order}")

    # 7. Minimum Semesters
    print("\n[7] Minimum Semesters")
    n = 3
    relations = [[1, 3], [2, 3]]
    semesters = minimum_semesters(n, relations)
    print(f"    Courses: {n}, Relations: {relations}")
    print(f"    Minimum semesters: {semesters}")

    # 8. Longest Path in DAG
    print("\n[8] Longest Path in DAG")
    n = 4
    edges = [(0, 1, 3), (0, 2, 2), (1, 3, 4), (2, 3, 1)]
    dist = longest_path_dag(n, edges)
    print(f"    Edges: {edges}")
    print(f"    Longest distance to each node: {dist}")

    # 9. Build Order
    print("\n[9] Build Order")
    projects = ['a', 'b', 'c', 'd', 'e', 'f']
    dependencies = [('d', 'a'), ('b', 'f'), ('d', 'b'), ('a', 'f'), ('c', 'd')]
    order = build_order(projects, dependencies)
    print(f"    Projects: {projects}")
    print(f"    Dependencies: {dependencies}")
    print(f"    Build order: {order}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
