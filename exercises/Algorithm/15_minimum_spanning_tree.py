"""
Exercises for Lesson 15: Minimum Spanning Tree
Topic: Algorithm

Solutions to practice problems from the lesson.
"""


# === Shared: Union-Find Data Structure ===

class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]

    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px == py:
            return False  # Already in same component
        # Union by rank: attach smaller tree under root of larger tree
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1
        return True


# === Exercise 1: Minimum Spanning Tree (Kruskal) ===
# Problem: Find the total weight of the MST for the given graph.
# Approach: Sort edges by weight, greedily add edges that don't form a cycle.

def exercise_1():
    """Solution: Kruskal's algorithm with Union-Find."""
    def kruskal_mst(V, edges):
        """
        V: number of vertices (0-indexed)
        edges: list of (u, v, weight)
        Returns: total MST weight
        """
        edges.sort(key=lambda x: x[2])
        uf = UnionFind(V)

        total = 0
        count = 0

        for u, v, w in edges:
            if count >= V - 1:
                break
            if uf.union(u, v):
                total += w
                count += 1

        return total

    # Test case 1:
    #     0
    #    / \
    #   1   4
    #  / \ / \
    # 1   2   3
    # Edges: (0,1,1), (0,2,4), (1,2,2), (1,3,6), (2,3,3)
    edges1 = [(0, 1, 1), (0, 2, 4), (1, 2, 2), (1, 3, 6), (2, 3, 3)]
    result = kruskal_mst(4, edges1)
    print(f"MST weight: {result}")
    assert result == 6  # edges: (0,1,1), (1,2,2), (2,3,3)

    # Test case 2: complete graph K4
    edges2 = [
        (0, 1, 10), (0, 2, 6), (0, 3, 5),
        (1, 3, 15), (2, 3, 4),
    ]
    result = kruskal_mst(4, edges2)
    print(f"K4 MST weight: {result}")
    assert result == 19  # edges: (2,3,4), (0,3,5), (0,1,10)

    print("All MST tests passed!")


# === Exercise 2: City Division Plan ===
# Problem: Divide N villages into 2 groups and connect each group with minimum cost.
# Approach: Build MST, then remove the largest edge (creating 2 connected components).
#   The remaining cost is the sum of all MST edges minus the largest.

def exercise_2():
    """Solution: MST minus the heaviest edge."""
    def divide_villages(V, edges):
        """
        V: number of villages (0-indexed)
        edges: list of (u, v, weight)
        Returns: minimum cost to connect 2 groups
        """
        edges.sort(key=lambda x: x[2])
        uf = UnionFind(V)

        mst_edges = []

        for u, v, w in edges:
            if uf.union(u, v):
                mst_edges.append(w)
                if len(mst_edges) == V - 1:
                    break

        # Remove the largest edge to split into 2 groups
        return sum(mst_edges) - max(mst_edges)

    # Test case 1
    # 4 villages
    edges1 = [
        (0, 1, 1), (0, 2, 2), (1, 2, 3),
        (1, 3, 4), (2, 3, 5),
    ]
    result = divide_villages(4, edges1)
    print(f"Min cost for 2 groups (4 villages): {result}")
    # MST: (0,1,1), (0,2,2), (1,3,4) -> total=7
    # Remove largest (4): 1 + 2 = 3
    assert result == 3

    # Test case 2
    edges2 = [
        (0, 1, 10), (1, 2, 1), (2, 3, 1), (3, 4, 1),
        (0, 4, 100),
    ]
    result = divide_villages(5, edges2)
    print(f"Min cost for 2 groups (5 villages): {result}")
    # MST: (1,2,1), (2,3,1), (3,4,1), (0,1,10) -> total=13
    # Remove largest (10): 1 + 1 + 1 = 3
    assert result == 3

    print("All City Division Plan tests passed!")


if __name__ == "__main__":
    print("=== Exercise 1: Minimum Spanning Tree ===")
    exercise_1()
    print("\n=== Exercise 2: City Division Plan ===")
    exercise_2()
    print("\nAll exercises completed!")
