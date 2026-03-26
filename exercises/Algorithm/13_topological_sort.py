"""
Exercises for Lesson 13: Topological Sort
Topic: Algorithm

Solutions to practice problems from the lesson.
The lesson provides recommended problems: Course Schedule (cycle detection),
Course Schedule II (topo sort order), Task scheduling (minimum time).
"""

from collections import deque


# === Exercise 1: Course Schedule (Cycle Detection) ===
# Problem: Determine if it's possible to finish all courses given prerequisites.
#   Input: numCourses = 2, prerequisites = [[1, 0]] means 0 must come before 1
#   Output: True (take 0 then 1)
# Approach: Kahn's algorithm (BFS topological sort). If we can process all nodes,
#   no cycle exists.

def exercise_1():
    """Solution: Kahn's algorithm for cycle detection in directed graph."""
    def can_finish(num_courses, prerequisites):
        # Build adjacency list and in-degree array
        adj = [[] for _ in range(num_courses)]
        in_degree = [0] * num_courses

        for course, prereq in prerequisites:
            adj[prereq].append(course)
            in_degree[course] += 1

        # Start with all courses having in-degree 0
        queue = deque()
        for i in range(num_courses):
            if in_degree[i] == 0:
                queue.append(i)

        processed = 0
        while queue:
            node = queue.popleft()
            processed += 1
            for neighbor in adj[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        # If all courses were processed, no cycle exists
        return processed == num_courses

    tests = [
        (2, [[1, 0]], True),                    # 0 -> 1, OK
        (2, [[1, 0], [0, 1]], False),            # cycle: 0 -> 1 -> 0
        (4, [[1, 0], [2, 0], [3, 1], [3, 2]], True),  # diamond DAG
        (1, [], True),                           # single course, no prereqs
        (3, [[0, 1], [1, 2], [2, 0]], False),    # cycle of length 3
    ]

    for n, prereqs, expected in tests:
        result = can_finish(n, prereqs)
        print(f"can_finish({n}, {prereqs}) = {result}")
        assert result == expected

    print("All Course Schedule tests passed!")


# === Exercise 2: Course Schedule II (Topological Order) ===
# Problem: Return the ordering of courses you should take to finish all courses.
#   Return empty if impossible (cycle exists).

def exercise_2():
    """Solution: Kahn's algorithm returning the topological order."""
    def find_order(num_courses, prerequisites):
        adj = [[] for _ in range(num_courses)]
        in_degree = [0] * num_courses

        for course, prereq in prerequisites:
            adj[prereq].append(course)
            in_degree[course] += 1

        queue = deque()
        for i in range(num_courses):
            if in_degree[i] == 0:
                queue.append(i)

        order = []
        while queue:
            node = queue.popleft()
            order.append(node)
            for neighbor in adj[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(order) == num_courses:
            return order
        else:
            return []  # Cycle detected

    # Test case 1
    result = find_order(4, [[1, 0], [2, 0], [3, 1], [3, 2]])
    print(f"Order for 4 courses: {result}")
    assert len(result) == 4
    # Verify: 0 before 1, 0 before 2, 1 before 3, 2 before 3
    idx = {course: i for i, course in enumerate(result)}
    assert idx[0] < idx[1] and idx[0] < idx[2]
    assert idx[1] < idx[3] and idx[2] < idx[3]

    # Test case 2: cycle
    result = find_order(2, [[1, 0], [0, 1]])
    print(f"Order for cycle: {result}")
    assert result == []

    # Test case 3: single course
    result = find_order(1, [])
    print(f"Order for 1 course: {result}")
    assert result == [0]

    print("All Course Schedule II tests passed!")


# === Exercise 3: Minimum Time to Complete All Tasks ===
# Problem: Given tasks with dependencies and durations, find the minimum time
#   to complete all tasks (parallel execution allowed).
# Approach: Topological sort with longest path calculation (critical path method).

def exercise_3():
    """Solution: Critical path method using topological sort."""
    def min_completion_time(n, durations, dependencies):
        """
        n: number of tasks (1-indexed)
        durations: list of task durations (1-indexed, durations[0] unused)
        dependencies: list of (task, prerequisite) pairs
        Returns: minimum time to complete all tasks
        """
        adj = [[] for _ in range(n + 1)]
        in_degree = [0] * (n + 1)

        for task, prereq in dependencies:
            adj[prereq].append(task)
            in_degree[task] += 1

        # earliest[i] = earliest start time for task i
        earliest = [0] * (n + 1)

        queue = deque()
        for i in range(1, n + 1):
            if in_degree[i] == 0:
                queue.append(i)

        while queue:
            node = queue.popleft()
            for neighbor in adj[node]:
                # neighbor can start after node finishes
                earliest[neighbor] = max(
                    earliest[neighbor],
                    earliest[node] + durations[node]
                )
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        # Total time = max(earliest[i] + durations[i]) for all tasks
        return max(earliest[i] + durations[i] for i in range(1, n + 1))

    # Test case 1:
    # Task 1 (3s), Task 2 (2s), Task 3 (4s)
    # 1 -> 3, 2 -> 3
    # Tasks 1 and 2 run in parallel, task 3 starts after both finish
    n = 3
    durations = [0, 3, 2, 4]  # 1-indexed
    deps = [(3, 1), (3, 2)]
    result = min_completion_time(n, durations, deps)
    print(f"3 tasks with deps -> min time: {result}")
    assert result == 7  # max(3,2) + 4 = 7

    # Test case 2: chain
    # 1 -> 2 -> 3, durations [2, 3, 1]
    n = 3
    durations = [0, 2, 3, 1]
    deps = [(2, 1), (3, 2)]
    result = min_completion_time(n, durations, deps)
    print(f"Chain 1->2->3 -> min time: {result}")
    assert result == 6  # 2 + 3 + 1 = 6

    # Test case 3: no dependencies (all parallel)
    n = 4
    durations = [0, 5, 3, 7, 2]
    deps = []
    result = min_completion_time(n, durations, deps)
    print(f"4 independent tasks -> min time: {result}")
    assert result == 7  # max(5, 3, 7, 2) = 7

    print("All Minimum Time tests passed!")


if __name__ == "__main__":
    print("=== Exercise 1: Course Schedule (Cycle Detection) ===")
    exercise_1()
    print("\n=== Exercise 2: Course Schedule II (Topological Order) ===")
    exercise_2()
    print("\n=== Exercise 3: Minimum Time to Complete All Tasks ===")
    exercise_3()
    print("\nAll exercises completed!")
