"""
Exercises for Lesson 16: LCA and Tree Queries
Topic: Algorithm

Solutions to practice problems from the lesson.
Problems: LCA Basic (Binary Lifting), Node Distance, Path Sum Query.
"""

import math


# === Exercise 1: LCA with Binary Lifting ===
# Problem: Find the Lowest Common Ancestor of two nodes in a rooted tree
#   using binary lifting (sparse table) for O(log n) queries after O(n log n) preprocessing.

def exercise_1():
    """Solution: Binary Lifting for LCA queries."""
    class LCA:
        def __init__(self, n, adj, root=0):
            self.n = n
            self.LOG = max(1, math.ceil(math.log2(n + 1)))
            self.depth = [0] * n
            # up[k][v] = 2^k-th ancestor of v
            self.up = [[0] * n for _ in range(self.LOG)]

            # BFS to compute depth and parent
            from collections import deque
            visited = [False] * n
            queue = deque([root])
            visited[root] = True
            self.up[0][root] = root  # root's parent is itself

            while queue:
                v = queue.popleft()
                for u in adj[v]:
                    if not visited[u]:
                        visited[u] = True
                        self.depth[u] = self.depth[v] + 1
                        self.up[0][u] = v  # direct parent
                        queue.append(u)

            # Fill sparse table: up[k][v] = up[k-1][up[k-1][v]]
            for k in range(1, self.LOG):
                for v in range(n):
                    self.up[k][v] = self.up[k - 1][self.up[k - 1][v]]

        def query(self, u, v):
            """Find LCA of nodes u and v."""
            # Ensure u is deeper
            if self.depth[u] < self.depth[v]:
                u, v = v, u

            # Lift u to the same depth as v
            diff = self.depth[u] - self.depth[v]
            for k in range(self.LOG):
                if (diff >> k) & 1:
                    u = self.up[k][u]

            if u == v:
                return u

            # Binary lift both until they diverge
            for k in range(self.LOG - 1, -1, -1):
                if self.up[k][u] != self.up[k][v]:
                    u = self.up[k][u]
                    v = self.up[k][v]

            return self.up[0][u]

    # Build tree:
    #       0
    #      / \
    #     1   2
    #    / \   \
    #   3   4   5
    #  /
    # 6
    n = 7
    adj = [[] for _ in range(n)]
    edges = [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (3, 6)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    lca = LCA(n, adj, root=0)

    tests = [
        (3, 4, 1),   # LCA(3,4) = 1
        (6, 4, 1),   # LCA(6,4) = 1
        (6, 5, 0),   # LCA(6,5) = 0
        (3, 3, 3),   # LCA(3,3) = 3
        (1, 5, 0),   # LCA(1,5) = 0
        (6, 1, 1),   # LCA(6,1) = 1
    ]

    for u, v, expected in tests:
        result = lca.query(u, v)
        print(f"LCA({u}, {v}) = {result}")
        assert result == expected, f"Expected {expected}, got {result}"

    print("All LCA Binary Lifting tests passed!")


# === Exercise 2: Node Distance Using LCA ===
# Problem: Find the distance between two nodes in a tree.
#   dist(u, v) = depth[u] + depth[v] - 2 * depth[LCA(u, v)]

def exercise_2():
    """Solution: Distance = depth[u] + depth[v] - 2*depth[LCA(u,v)]."""
    class LCAWithDistance:
        def __init__(self, n, adj, root=0):
            self.n = n
            self.LOG = max(1, math.ceil(math.log2(n + 1)))
            self.depth = [0] * n
            self.up = [[0] * n for _ in range(self.LOG)]

            from collections import deque
            visited = [False] * n
            queue = deque([root])
            visited[root] = True
            self.up[0][root] = root

            while queue:
                v = queue.popleft()
                for u in adj[v]:
                    if not visited[u]:
                        visited[u] = True
                        self.depth[u] = self.depth[v] + 1
                        self.up[0][u] = v
                        queue.append(u)

            for k in range(1, self.LOG):
                for v in range(n):
                    self.up[k][v] = self.up[k - 1][self.up[k - 1][v]]

        def lca(self, u, v):
            if self.depth[u] < self.depth[v]:
                u, v = v, u
            diff = self.depth[u] - self.depth[v]
            for k in range(self.LOG):
                if (diff >> k) & 1:
                    u = self.up[k][u]
            if u == v:
                return u
            for k in range(self.LOG - 1, -1, -1):
                if self.up[k][u] != self.up[k][v]:
                    u = self.up[k][u]
                    v = self.up[k][v]
            return self.up[0][u]

        def distance(self, u, v):
            ancestor = self.lca(u, v)
            return self.depth[u] + self.depth[v] - 2 * self.depth[ancestor]

    # Same tree as exercise 1
    n = 7
    adj = [[] for _ in range(n)]
    edges = [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (3, 6)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    tree = LCAWithDistance(n, adj, root=0)

    tests = [
        (6, 4, 3),   # 6->3->1->4
        (6, 5, 4),   # 6->3->1->0->2->5 ... actually 6->3->1->0->2->5 = 5 edges
        (3, 4, 2),   # 3->1->4
        (0, 6, 3),   # 0->1->3->6
        (6, 6, 0),   # same node
    ]

    # Recompute expected: depth[0]=0, depth[1]=1, depth[2]=1, depth[3]=2,
    #   depth[4]=2, depth[5]=2, depth[6]=3
    # dist(6,5): LCA=0, depth[6]+depth[5]-2*depth[0] = 3+2-0 = 5
    tests = [
        (6, 4, 3),   # LCA=1, 3+2-2*1=3
        (6, 5, 5),   # LCA=0, 3+2-2*0=5
        (3, 4, 2),   # LCA=1, 2+2-2*1=2
        (0, 6, 3),   # LCA=0, 0+3-2*0=3
        (6, 6, 0),   # same node
    ]

    for u, v, expected in tests:
        result = tree.distance(u, v)
        print(f"dist({u}, {v}) = {result}")
        assert result == expected, f"Expected {expected}, got {result}"

    print("All Node Distance tests passed!")


# === Exercise 3: Path Sum Query ===
# Problem: Given node weights, answer queries about the sum of weights on the
#   path between two nodes.
# Approach: Use prefix sums from root + LCA.
#   path_sum(u,v) = prefix[u] + prefix[v] - 2*prefix[LCA(u,v)] + weight[LCA(u,v)]

def exercise_3():
    """Solution: Path sum = prefix[u] + prefix[v] - 2*prefix[lca] + weight[lca]."""
    class PathSumQuery:
        def __init__(self, n, adj, weights, root=0):
            self.n = n
            self.LOG = max(1, math.ceil(math.log2(n + 1)))
            self.depth = [0] * n
            self.up = [[0] * n for _ in range(self.LOG)]
            self.prefix = [0] * n  # prefix[v] = sum of weights from root to v
            self.weights = weights

            from collections import deque
            visited = [False] * n
            queue = deque([root])
            visited[root] = True
            self.up[0][root] = root
            self.prefix[root] = weights[root]

            while queue:
                v = queue.popleft()
                for u in adj[v]:
                    if not visited[u]:
                        visited[u] = True
                        self.depth[u] = self.depth[v] + 1
                        self.up[0][u] = v
                        self.prefix[u] = self.prefix[v] + weights[u]
                        queue.append(u)

            for k in range(1, self.LOG):
                for v in range(n):
                    self.up[k][v] = self.up[k - 1][self.up[k - 1][v]]

        def lca(self, u, v):
            if self.depth[u] < self.depth[v]:
                u, v = v, u
            diff = self.depth[u] - self.depth[v]
            for k in range(self.LOG):
                if (diff >> k) & 1:
                    u = self.up[k][u]
            if u == v:
                return u
            for k in range(self.LOG - 1, -1, -1):
                if self.up[k][u] != self.up[k][v]:
                    u = self.up[k][u]
                    v = self.up[k][v]
            return self.up[0][u]

        def path_sum(self, u, v):
            ancestor = self.lca(u, v)
            return (self.prefix[u] + self.prefix[v]
                    - 2 * self.prefix[ancestor]
                    + self.weights[ancestor])

    # Tree with weights:
    #       0(1)
    #      / \
    #     1(2) 2(3)
    #    / \     \
    #   3(4) 4(5) 5(6)
    n = 6
    weights = [1, 2, 3, 4, 5, 6]
    adj = [[] for _ in range(n)]
    edges = [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    psq = PathSumQuery(n, adj, weights, root=0)

    tests = [
        (3, 4, 2 + 4 + 5),        # path: 3->1->4, sum = 4+2+5 = 11
        (3, 5, 4 + 2 + 1 + 3 + 6),  # path: 3->1->0->2->5, sum = 16
        (0, 5, 1 + 3 + 6),        # path: 0->2->5, sum = 10
        (3, 3, 4),                 # same node, just its weight
    ]

    for u, v, expected in tests:
        result = psq.path_sum(u, v)
        print(f"path_sum({u}, {v}) = {result}")
        assert result == expected, f"Expected {expected}, got {result}"

    print("All Path Sum Query tests passed!")


if __name__ == "__main__":
    print("=== Exercise 1: LCA with Binary Lifting ===")
    exercise_1()
    print("\n=== Exercise 2: Node Distance Using LCA ===")
    exercise_2()
    print("\n=== Exercise 3: Path Sum Query ===")
    exercise_3()
    print("\nAll exercises completed!")
