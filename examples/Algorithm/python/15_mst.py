"""
Union-Find (Disjoint Set Union)
Union-Find / Disjoint Set Union Data Structure

A data structure for managing disjoint sets, commonly used for graph connectivity problems.
"""

from typing import List, Tuple


# =============================================================================
# 1. Basic Union-Find
# =============================================================================
class UnionFind:
    """
    Basic Union-Find Implementation
    - Path Compression
    - Union by Rank
    """

    def __init__(self, n: int):
        """
        Initialize with n elements (0 ~ n-1)
        """
        self.parent = list(range(n))  # Each element is its own parent
        self.rank = [0] * n           # Approximate tree height
        self.count = n                # Number of sets

    def find(self, x: int) -> int:
        """
        Find the representative (root) of the set containing x
        Nearly O(1) with path compression
        """
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]

    def union(self, x: int, y: int) -> bool:
        """
        Merge the sets containing x and y
        Returns False if already in the same set
        """
        root_x = self.find(x)
        root_y = self.find(y)

        if root_x == root_y:
            return False  # Already in the same set

        # Union by rank (attach smaller tree to larger tree)
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            self.parent[root_y] = root_x
            self.rank[root_x] += 1

        self.count -= 1
        return True

    def connected(self, x: int, y: int) -> bool:
        """Check if x and y are in the same set"""
        return self.find(x) == self.find(y)

    def get_count(self) -> int:
        """Get the current number of sets"""
        return self.count


# =============================================================================
# 2. Size-Based Union-Find
# =============================================================================
class UnionFindWithSize:
    """
    Union-Find that tracks set sizes
    """

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.size = [1] * n  # Size of each set

    def find(self, x: int) -> int:
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: int, y: int) -> bool:
        root_x = self.find(x)
        root_y = self.find(y)

        if root_x == root_y:
            return False

        # Union by size (attach smaller set to larger set)
        if self.size[root_x] < self.size[root_y]:
            self.parent[root_x] = root_y
            self.size[root_y] += self.size[root_x]
        else:
            self.parent[root_y] = root_x
            self.size[root_x] += self.size[root_y]

        return True

    def get_size(self, x: int) -> int:
        """Get the size of the set containing x"""
        return self.size[self.find(x)]


# =============================================================================
# 3. Connected Components Count
# =============================================================================
def count_components(n: int, edges: List[List[int]]) -> int:
    """
    Count connected components given n nodes and an edge list
    """
    uf = UnionFind(n)
    for u, v in edges:
        uf.union(u, v)
    return uf.get_count()


# =============================================================================
# 4. Cycle Detection in Graph
# =============================================================================
def has_cycle(n: int, edges: List[List[int]]) -> bool:
    """
    Check for cycle in an undirected graph
    If nodes are already in the same set when adding an edge, a cycle exists
    """
    uf = UnionFind(n)
    for u, v in edges:
        if not uf.union(u, v):
            return True  # Already connected = cycle
    return False


# =============================================================================
# 5. Kruskal's MST (Minimum Spanning Tree)
# =============================================================================
def kruskal_mst(n: int, edges: List[Tuple[int, int, int]]) -> Tuple[int, List[Tuple[int, int, int]]]:
    """
    Find the Minimum Spanning Tree using Kruskal's Algorithm
    edges: [(u, v, weight), ...]
    Returns: (total weight, list of MST edges)
    """
    # Sort by weight
    edges = sorted(edges, key=lambda x: x[2])

    uf = UnionFind(n)
    mst_weight = 0
    mst_edges = []

    for u, v, w in edges:
        if uf.union(u, v):
            mst_weight += w
            mst_edges.append((u, v, w))

            # Done when n-1 edges are selected
            if len(mst_edges) == n - 1:
                break

    return mst_weight, mst_edges


# =============================================================================
# 6. Account Merging (Friend Relationships)
# =============================================================================
def merge_accounts(accounts: List[List[str]]) -> List[List[str]]:
    """
    Merge accounts that share the same email
    accounts[i] = [name, email1, email2, ...]
    """
    from collections import defaultdict

    # Email -> account index mapping
    email_to_id = {}
    email_to_name = {}

    for i, account in enumerate(accounts):
        name = account[0]
        for email in account[1:]:
            if email in email_to_id:
                pass  # Will union later
            email_to_id[email] = i
            email_to_name[email] = name

    # Connect accounts belonging to the same person using Union-Find
    n = len(accounts)
    uf = UnionFind(n)

    email_first_account = {}
    for i, account in enumerate(accounts):
        for email in account[1:]:
            if email in email_first_account:
                uf.union(i, email_first_account[email])
            else:
                email_first_account[email] = i

    # Aggregate results
    root_to_emails = defaultdict(set)
    for i, account in enumerate(accounts):
        root = uf.find(i)
        for email in account[1:]:
            root_to_emails[root].add(email)

    # Format results
    result = []
    for root, emails in root_to_emails.items():
        name = accounts[root][0]
        result.append([name] + sorted(emails))

    return result


# =============================================================================
# 7. Number of Islands (2D Grid)
# =============================================================================
def num_islands_union_find(grid: List[List[str]]) -> int:
    """
    '1' is land, '0' is water
    Count connected land masses (islands)
    """
    if not grid or not grid[0]:
        return 0

    rows, cols = len(grid), len(grid[0])

    # 2D -> 1D coordinate conversion
    def get_index(r, c):
        return r * cols + c

    uf = UnionFind(rows * cols)
    land_count = 0

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == '1':
                land_count += 1
                # Only check right and down to avoid duplicates
                for dr, dc in [(0, 1), (1, 0)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == '1':
                        if uf.union(get_index(r, c), get_index(nr, nc)):
                            land_count -= 1

    return land_count


# =============================================================================
# Tests
# =============================================================================
def main():
    print("=" * 60)
    print("Union-Find Examples")
    print("=" * 60)

    # 1. Basic Usage
    print("\n[1] Basic Union-Find")
    uf = UnionFind(10)
    operations = [(0, 1), (2, 3), (4, 5), (1, 2), (6, 7), (8, 9), (0, 9)]
    for u, v in operations:
        uf.union(u, v)
        print(f"    union({u}, {v}) -> set count: {uf.get_count()}")

    print(f"\n    0 and 9 connected? {uf.connected(0, 9)}")
    print(f"    0 and 6 connected? {uf.connected(0, 6)}")

    # 2. Size Tracking
    print("\n[2] Size-Based Union-Find")
    uf_size = UnionFindWithSize(5)
    uf_size.union(0, 1)
    uf_size.union(2, 3)
    uf_size.union(0, 2)
    print(f"    Size of set containing 0: {uf_size.get_size(0)}")
    print(f"    Size of set containing 4: {uf_size.get_size(4)}")

    # 3. Connected Components Count
    print("\n[3] Connected Components Count")
    edges = [[0, 1], [1, 2], [3, 4]]
    count = count_components(5, edges)
    print(f"    5 nodes, edges: {edges}")
    print(f"    Connected components: {count}")

    # 4. Cycle Detection
    print("\n[4] Cycle Detection")
    edges_no_cycle = [[0, 1], [1, 2], [2, 3]]
    edges_with_cycle = [[0, 1], [1, 2], [2, 0]]
    print(f"    Edges {edges_no_cycle}: cycle = {has_cycle(4, edges_no_cycle)}")
    print(f"    Edges {edges_with_cycle}: cycle = {has_cycle(3, edges_with_cycle)}")

    # 5. Kruskal's MST
    print("\n[5] Kruskal's MST")
    #     1
    #   0---1
    #   |\  |
    # 4 | \ |2
    #   |  \|
    #   3---2
    #     3
    edges_mst = [
        (0, 1, 1), (0, 2, 4), (0, 3, 4),
        (1, 2, 2), (2, 3, 3)
    ]
    total_weight, mst_edges = kruskal_mst(4, edges_mst)
    print(f"    Edges: {edges_mst}")
    print(f"    MST total weight: {total_weight}")
    print(f"    MST edges: {mst_edges}")

    # 6. Account Merging
    print("\n[6] Account Merging")
    accounts = [
        ["John", "john@mail.com", "john_work@mail.com"],
        ["John", "john@mail.com", "john2@mail.com"],
        ["Mary", "mary@mail.com"],
        ["John", "john3@mail.com"]
    ]
    result = merge_accounts(accounts)
    print(f"    Input:")
    for acc in accounts:
        print(f"      {acc}")
    print(f"    Merged result:")
    for acc in result:
        print(f"      {acc}")

    # 7. Number of Islands (Union-Find)
    print("\n[7] Number of Islands (Union-Find)")
    grid = [
        ['1', '1', '0', '0', '0'],
        ['1', '1', '0', '0', '0'],
        ['0', '0', '1', '0', '0'],
        ['0', '0', '0', '1', '1']
    ]
    count = num_islands_union_find(grid)
    print(f"    Grid:")
    for row in grid:
        print(f"    {row}")
    print(f"    Number of islands: {count}")

    print("\n" + "=" * 60)
    print("Union-Find Time Complexity")
    print("=" * 60)
    print("""
    With path compression + union by rank/size:
    - find(): Nearly O(1) (precisely O(alpha(n)), alpha is inverse Ackermann)
    - union(): Nearly O(1)
    - Space complexity: O(n)

    Key applications:
    - Connected component management
    - Cycle detection
    - Minimum spanning tree (Kruskal's)
    - Dynamic connectivity problems
    """)


if __name__ == "__main__":
    main()
