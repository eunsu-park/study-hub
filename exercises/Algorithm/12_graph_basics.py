"""
Exercises for Lesson 12: Graph Basics
Topic: Algorithm

Solutions to practice problems from the lesson.
"""

from collections import deque


# === Exercise 1: Path Existence Check ===
# Problem: Check if a path exists between two vertices in an undirected graph.
# Approach: DFS with iterative stack to avoid recursion limit issues.

def exercise_1():
    """Solution: DFS-based path existence check."""
    def has_path(graph, start, end):
        """Check if path exists from start to end using iterative DFS."""
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

    # Graph:
    # 0 -- 1 -- 3
    # |         |
    # 2         4
    #
    # 5 -- 6 (separate component)
    graph = {
        0: [1, 2],
        1: [0, 3],
        2: [0],
        3: [1, 4],
        4: [3],
        5: [6],
        6: [5],
    }

    tests = [
        (0, 4, True),    # Path: 0 -> 1 -> 3 -> 4
        (0, 2, True),    # Path: 0 -> 2
        (2, 4, True),    # Path: 2 -> 0 -> 1 -> 3 -> 4
        (0, 5, False),   # Different component
        (5, 6, True),    # Path: 5 -> 6
        (0, 0, True),    # Same node
    ]

    for start, end, expected in tests:
        result = has_path(graph, start, end)
        print(f"has_path({start}, {end}) = {result}")
        assert result == expected

    print("All Path Existence tests passed!")


# === Exercise 2: Find All Paths ===
# Problem: Find all paths between two vertices (no cycles).
# Approach: DFS backtracking, tracking the current path to avoid revisiting nodes.

def exercise_2():
    """Solution: DFS backtracking to enumerate all simple paths."""
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

    # Graph:
    # 0 -- 1 -- 3
    # |    |    |
    # 2 ---+    4
    graph = {
        0: [1, 2],
        1: [0, 2, 3],
        2: [0, 1],
        3: [1, 4],
        4: [3],
    }

    # Find all paths from 0 to 4
    all_paths = find_all_paths(graph, 0, 4)
    print(f"All paths from 0 to 4:")
    for p in all_paths:
        print(f"  {' -> '.join(map(str, p))}")
    assert len(all_paths) >= 2  # At least 0->1->3->4 and 0->2->1->3->4

    # Find all paths from 0 to 2
    all_paths_02 = find_all_paths(graph, 0, 2)
    print(f"\nAll paths from 0 to 2:")
    for p in all_paths_02:
        print(f"  {' -> '.join(map(str, p))}")
    assert len(all_paths_02) >= 2  # 0->2 and 0->1->2

    print("All Find All Paths tests passed!")


# === Exercise 3: Number of Islands (Grid BFS) ===
# Problem: Count the number of islands in a 2D grid.
#   '1' represents land and '0' represents water.
# This is from the recommended problems section.

def exercise_3():
    """Solution: BFS flood fill for connected component counting."""
    def num_islands(grid):
        if not grid:
            return 0

        m, n = len(grid), len(grid[0])
        visited = [[False] * n for _ in range(m)]
        count = 0
        dx = [0, 0, 1, -1]
        dy = [1, -1, 0, 0]

        def bfs(si, sj):
            queue = deque([(si, sj)])
            visited[si][sj] = True

            while queue:
                x, y = queue.popleft()
                for d in range(4):
                    nx, ny = x + dx[d], y + dy[d]
                    if 0 <= nx < m and 0 <= ny < n and not visited[nx][ny] and grid[nx][ny] == '1':
                        visited[nx][ny] = True
                        queue.append((nx, ny))

        for i in range(m):
            for j in range(n):
                if grid[i][j] == '1' and not visited[i][j]:
                    bfs(i, j)
                    count += 1

        return count

    # Test case 1
    grid1 = [
        ['1', '1', '1', '1', '0'],
        ['1', '1', '0', '1', '0'],
        ['1', '1', '0', '0', '0'],
        ['0', '0', '0', '0', '0'],
    ]
    result1 = num_islands(grid1)
    print(f"Grid 1: {result1} island(s)")
    assert result1 == 1

    # Test case 2
    grid2 = [
        ['1', '1', '0', '0', '0'],
        ['1', '1', '0', '0', '0'],
        ['0', '0', '1', '0', '0'],
        ['0', '0', '0', '1', '1'],
    ]
    result2 = num_islands(grid2)
    print(f"Grid 2: {result2} island(s)")
    assert result2 == 3

    # Test case 3: all water
    grid3 = [['0', '0'], ['0', '0']]
    result3 = num_islands(grid3)
    print(f"Grid 3: {result3} island(s)")
    assert result3 == 0

    print("All Number of Islands tests passed!")


if __name__ == "__main__":
    print("=== Exercise 1: Path Existence Check ===")
    exercise_1()
    print("\n=== Exercise 2: Find All Paths ===")
    exercise_2()
    print("\n=== Exercise 3: Number of Islands ===")
    exercise_3()
    print("\nAll exercises completed!")
