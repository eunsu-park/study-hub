"""
Lowest Common Ancestor (LCA)
LCA and Tree Queries

Algorithms for finding the lowest common ancestor of two nodes in a tree.
"""

from typing import List, Tuple, Optional
from collections import defaultdict, deque
import math


# =============================================================================
# 1. Naive LCA
# =============================================================================

def lca_naive(n: int, edges: List[Tuple[int, int]], u: int, v: int) -> int:
    """
    Naive LCA (level alignment)
    Time Complexity: O(n) per query
    Preprocessing: O(n)
    """
    # Build tree
    adj = defaultdict(list)
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)

    # Compute parent and depth
    parent = [-1] * n
    depth = [0] * n

    def dfs(node: int, par: int, d: int):
        parent[node] = par
        depth[node] = d
        for child in adj[node]:
            if child != par:
                dfs(child, node, d + 1)

    dfs(0, -1, 0)

    # Align depths
    while depth[u] > depth[v]:
        u = parent[u]
    while depth[v] > depth[u]:
        v = parent[v]

    # Move up simultaneously
    while u != v:
        u = parent[u]
        v = parent[v]

    return u


# =============================================================================
# 2. Binary Lifting (Sparse Table)
# =============================================================================

class LCABinaryLifting:
    """
    LCA using Binary Lifting
    Preprocessing: O(n log n)
    Query: O(log n)
    """

    def __init__(self, n: int, edges: List[Tuple[int, int]], root: int = 0):
        self.n = n
        self.LOG = max(1, int(math.log2(n)) + 1)

        # Build graph
        self.adj = defaultdict(list)
        for a, b in edges:
            self.adj[a].append(b)
            self.adj[b].append(a)

        # Preprocessing
        self.parent = [[-1] * n for _ in range(self.LOG)]
        self.depth = [0] * n

        self._preprocess(root)

    def _preprocess(self, root: int):
        """Compute parent/depth via DFS + build sparse table"""
        stack = [(root, -1, 0)]

        while stack:
            node, par, d = stack.pop()
            self.parent[0][node] = par
            self.depth[node] = d

            for child in self.adj[node]:
                if child != par:
                    stack.append((child, node, d + 1))

        # Build sparse table: parent[i][v] = 2^i-th ancestor of v
        for i in range(1, self.LOG):
            for v in range(self.n):
                if self.parent[i - 1][v] != -1:
                    self.parent[i][v] = self.parent[i - 1][self.parent[i - 1][v]]

    def query(self, u: int, v: int) -> int:
        """LCA query - O(log n)"""
        # Ensure u is deeper
        if self.depth[u] < self.depth[v]:
            u, v = v, u

        # Align depths
        diff = self.depth[u] - self.depth[v]
        for i in range(self.LOG):
            if (diff >> i) & 1:
                u = self.parent[i][u]

        # If equal, done
        if u == v:
            return u

        # Move up simultaneously
        for i in range(self.LOG - 1, -1, -1):
            if self.parent[i][u] != self.parent[i][v]:
                u = self.parent[i][u]
                v = self.parent[i][v]

        return self.parent[0][u]

    def kth_ancestor(self, node: int, k: int) -> int:
        """Find k-th ancestor - O(log n)"""
        for i in range(self.LOG):
            if node == -1:
                break
            if (k >> i) & 1:
                node = self.parent[i][node]
        return node

    def distance(self, u: int, v: int) -> int:
        """Distance between two nodes - O(log n)"""
        lca = self.query(u, v)
        return self.depth[u] + self.depth[v] - 2 * self.depth[lca]


# =============================================================================
# 3. Euler Tour + RMQ (Sparse Table)
# =============================================================================

class LCAEulerTour:
    """
    LCA using Euler Tour + RMQ
    Preprocessing: O(n log n)
    Query: O(1)
    """

    def __init__(self, n: int, edges: List[Tuple[int, int]], root: int = 0):
        self.n = n
        self.adj = defaultdict(list)
        for a, b in edges:
            self.adj[a].append(b)
            self.adj[b].append(a)

        # Euler tour and first occurrence positions
        self.euler = []  # (depth, node) pairs
        self.first = [-1] * n  # First occurrence index for each node

        self._build_euler_tour(root)
        self._build_sparse_table()

    def _build_euler_tour(self, root: int):
        """Build Euler tour - O(n)"""
        stack = [(root, -1, 0, False)]

        while stack:
            node, parent, depth, visited = stack.pop()

            self.euler.append((depth, node))
            if self.first[node] == -1:
                self.first[node] = len(self.euler) - 1

            if visited:
                continue

            stack.append((node, parent, depth, True))
            for child in self.adj[node]:
                if child != parent:
                    stack.append((child, node, depth + 1, False))

    def _build_sparse_table(self):
        """Build sparse table - O(n log n)"""
        m = len(self.euler)
        self.LOG = max(1, int(math.log2(m)) + 1)

        # sparse[i][j] = index of minimum in euler[j..j+2^i)
        self.sparse = [[0] * m for _ in range(self.LOG)]

        for j in range(m):
            self.sparse[0][j] = j

        for i in range(1, self.LOG):
            length = 1 << i
            for j in range(m - length + 1):
                left = self.sparse[i - 1][j]
                right = self.sparse[i - 1][j + (length >> 1)]
                if self.euler[left][0] <= self.euler[right][0]:
                    self.sparse[i][j] = left
                else:
                    self.sparse[i][j] = right

    def _rmq(self, left: int, right: int) -> int:
        """Range Minimum Query - O(1)"""
        length = right - left + 1
        k = int(math.log2(length))
        left_idx = self.sparse[k][left]
        right_idx = self.sparse[k][right - (1 << k) + 1]
        if self.euler[left_idx][0] <= self.euler[right_idx][0]:
            return left_idx
        return right_idx

    def query(self, u: int, v: int) -> int:
        """LCA query - O(1)"""
        left = self.first[u]
        right = self.first[v]
        if left > right:
            left, right = right, left
        idx = self._rmq(left, right)
        return self.euler[idx][1]


# =============================================================================
# 4. Tree Path Sum/Max/Min
# =============================================================================

class TreePathQuery:
    """Tree path queries (LCA + weights)"""

    def __init__(self, n: int, edges: List[Tuple[int, int, int]], root: int = 0):
        """edges: [(u, v, weight), ...]"""
        self.n = n
        self.LOG = max(1, int(math.log2(n)) + 1)

        self.adj = defaultdict(list)
        for a, b, w in edges:
            self.adj[a].append((b, w))
            self.adj[b].append((a, w))

        self.parent = [[-1] * n for _ in range(self.LOG)]
        self.depth = [0] * n
        self.dist_from_root = [0] * n  # Distance from root
        self.max_edge = [[0] * n for _ in range(self.LOG)]  # Maximum edge on path

        self._preprocess(root)

    def _preprocess(self, root: int):
        stack = [(root, -1, 0, 0)]

        while stack:
            node, par, d, dist = stack.pop()
            self.parent[0][node] = par
            self.depth[node] = d
            self.dist_from_root[node] = dist

            for child, weight in self.adj[node]:
                if child != par:
                    self.max_edge[0][child] = weight
                    stack.append((child, node, d + 1, dist + weight))

        # Sparse table
        for i in range(1, self.LOG):
            for v in range(self.n):
                if self.parent[i - 1][v] != -1:
                    self.parent[i][v] = self.parent[i - 1][self.parent[i - 1][v]]
                    self.max_edge[i][v] = max(
                        self.max_edge[i - 1][v],
                        self.max_edge[i - 1][self.parent[i - 1][v]] if self.parent[i - 1][v] != -1 else 0
                    )

    def lca(self, u: int, v: int) -> int:
        """LCA query"""
        if self.depth[u] < self.depth[v]:
            u, v = v, u

        diff = self.depth[u] - self.depth[v]
        for i in range(self.LOG):
            if (diff >> i) & 1:
                u = self.parent[i][u]

        if u == v:
            return u

        for i in range(self.LOG - 1, -1, -1):
            if self.parent[i][u] != self.parent[i][v]:
                u = self.parent[i][u]
                v = self.parent[i][v]

        return self.parent[0][u]

    def path_distance(self, u: int, v: int) -> int:
        """Sum of distances on path"""
        ancestor = self.lca(u, v)
        return self.dist_from_root[u] + self.dist_from_root[v] - 2 * self.dist_from_root[ancestor]

    def path_max_edge(self, u: int, v: int) -> int:
        """Maximum edge weight on path"""
        ancestor = self.lca(u, v)
        result = 0

        # u -> lca
        curr = u
        diff = self.depth[u] - self.depth[ancestor]
        for i in range(self.LOG):
            if (diff >> i) & 1:
                result = max(result, self.max_edge[i][curr])
                curr = self.parent[i][curr]

        # v -> lca
        curr = v
        diff = self.depth[v] - self.depth[ancestor]
        for i in range(self.LOG):
            if (diff >> i) & 1:
                result = max(result, self.max_edge[i][curr])
                curr = self.parent[i][curr]

        return result


# =============================================================================
# 5. Practical Problem: Path Between Two Nodes in a Tree
# =============================================================================

def find_path(n: int, edges: List[Tuple[int, int]], u: int, v: int) -> List[int]:
    """Find the path between two nodes"""
    lca_solver = LCABinaryLifting(n, edges)
    ancestor = lca_solver.query(u, v)

    # u -> lca
    path_u = []
    curr = u
    while curr != ancestor:
        path_u.append(curr)
        curr = lca_solver.parent[0][curr]
    path_u.append(ancestor)

    # v -> lca (reversed)
    path_v = []
    curr = v
    while curr != ancestor:
        path_v.append(curr)
        curr = lca_solver.parent[0][curr]

    return path_u + path_v[::-1]


# =============================================================================
# Tests
# =============================================================================

def main():
    print("=" * 60)
    print("Lowest Common Ancestor (LCA) Examples")
    print("=" * 60)

    # Tree structure
    #        0
    #      / | \
    #     1  2  3
    #    / \    |
    #   4   5   6
    #  /
    # 7

    n = 8
    edges = [(0, 1), (0, 2), (0, 3), (1, 4), (1, 5), (3, 6), (4, 7)]

    # 1. Naive LCA
    print("\n[1] Naive LCA")
    lca = lca_naive(n, edges, 7, 5)
    print(f"    LCA(7, 5) = {lca}")
    lca = lca_naive(n, edges, 7, 6)
    print(f"    LCA(7, 6) = {lca}")

    # 2. Binary Lifting
    print("\n[2] Binary Lifting")
    lca_bl = LCABinaryLifting(n, edges)
    print(f"    LCA(7, 5) = {lca_bl.query(7, 5)}")
    print(f"    LCA(7, 6) = {lca_bl.query(7, 6)}")
    print(f"    LCA(4, 6) = {lca_bl.query(4, 6)}")
    print(f"    Distance(7, 5) = {lca_bl.distance(7, 5)}")
    print(f"    2nd ancestor of 7 = {lca_bl.kth_ancestor(7, 2)}")

    # 3. Euler Tour + RMQ
    print("\n[3] Euler Tour + RMQ (O(1) query)")
    lca_euler = LCAEulerTour(n, edges)
    print(f"    LCA(7, 5) = {lca_euler.query(7, 5)}")
    print(f"    LCA(7, 6) = {lca_euler.query(7, 6)}")

    # 4. Weighted Tree Path Queries
    print("\n[4] Weighted Tree Path Queries")
    weighted_edges = [
        (0, 1, 3), (0, 2, 5), (0, 3, 4),
        (1, 4, 2), (1, 5, 6), (3, 6, 1), (4, 7, 8)
    ]
    path_query = TreePathQuery(n, weighted_edges)
    print(f"    Path distance(7, 5) = {path_query.path_distance(7, 5)}")
    print(f"    Path max edge(7, 5) = {path_query.path_max_edge(7, 5)}")
    print(f"    Path distance(7, 6) = {path_query.path_distance(7, 6)}")

    # 5. Path Finding
    print("\n[5] Path Between Two Nodes")
    path = find_path(n, edges, 7, 6)
    print(f"    Path(7, 6) = {path}")
    path = find_path(n, edges, 5, 2)
    print(f"    Path(5, 2) = {path}")

    # 6. Complexity Comparison
    print("\n[6] Complexity Comparison")
    print("    | Method         | Preprocessing | Query   |")
    print("    |----------------|---------------|---------|")
    print("    | Naive          | O(n)          | O(n)    |")
    print("    | Binary Lifting | O(n log n)    | O(log n)|")
    print("    | Euler + RMQ    | O(n log n)    | O(1)    |")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
