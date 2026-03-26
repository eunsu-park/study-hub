"""
Exercises for Lesson 31: Link-Cut Tree
Topic: Algorithm

Solutions to practice problems covering dynamic forest operations,
connectivity queries, path aggregates, and dynamic MST.
"""


# ============================================================
# Shared: Link-Cut Tree Implementation
# ============================================================
class Node:
    __slots__ = ('ch', 'p', 'rev', 'val', 'agg', 'sz', 'id', 'mn')

    def __init__(self, val=0, node_id=-1):
        self.ch = [None, None]
        self.p = None
        self.rev = False
        self.val = val
        self.agg = val
        self.mn = val
        self.sz = 1
        self.id = node_id


def _is_root(x):
    if x.p is None: return True
    return x.p.ch[0] != x and x.p.ch[1] != x

def _pull(x):
    if x is None: return
    x.agg = x.val
    x.mn = x.val
    x.sz = 1
    for c in x.ch:
        if c:
            x.agg += c.agg
            x.mn = min(x.mn, c.mn)
            x.sz += c.sz

def _push(x):
    if x and x.rev:
        x.ch[0], x.ch[1] = x.ch[1], x.ch[0]
        for c in x.ch:
            if c: c.rev = not c.rev
        x.rev = False

def _rotate(x):
    p = x.p; g = p.p
    d = 0 if p.ch[0] == x else 1
    p.ch[d] = x.ch[1-d]
    if x.ch[1-d]: x.ch[1-d].p = p
    x.ch[1-d] = p; p.p = x; x.p = g
    if g:
        if g.ch[0] == p: g.ch[0] = x
        elif g.ch[1] == p: g.ch[1] = x
    _pull(p); _pull(x)

def _splay(x):
    path = []; y = x
    while not _is_root(y): path.append(y.p); y = y.p
    path.append(y)
    while path: _push(path.pop())
    _push(x)
    while not _is_root(x):
        p = x.p
        if not _is_root(p):
            g = p.p
            same = (g.ch[0] == p) == (p.ch[0] == x)
            _rotate(p) if same else _rotate(x)
        _rotate(x)

def access(x):
    last = None; y = x
    while y:
        _splay(y); y.ch[1] = last; _pull(y)
        last = y; y = y.p
    _splay(x); return last

def make_root(x):
    access(x); x.rev = not x.rev; _push(x)

def find_root(x):
    access(x)
    while x.ch[0]: _push(x); x = x.ch[0]
    _splay(x); return x

def link(x, y):
    make_root(x); x.p = y

def cut(x, y):
    make_root(x); access(y)
    if y.ch[0] == x: y.ch[0] = None; x.p = None; _pull(y)

def connected(x, y):
    return find_root(x) is find_root(y)

def path_sum(x, y):
    make_root(x); access(y); return y.agg

def path_min(x, y):
    make_root(x); access(y); return y.mn


# ============================================================
# Exercise 1: Basic Link-Cut Tree Operations
# ============================================================
def exercise_1():
    """
    Test basic operations: create, link, cut, path query, connectivity.
    """
    print("=== Exercise 1: Basic Link-Cut Operations ===\n")

    nodes = [Node(val=i+1, node_id=i) for i in range(10)]

    # Build path: 0-1-2-3-4-5-6-7-8-9
    for i in range(9):
        link(nodes[i], nodes[i+1])
    print(f"  Linked 0-1-2-...-9 (path)")

    # Verify connectivity
    print(f"  connected(0, 9) = {connected(nodes[0], nodes[9])}")
    print(f"  connected(0, 5) = {connected(nodes[0], nodes[5])}")

    # Path sum
    s = path_sum(nodes[0], nodes[9])
    expected = sum(range(1, 11))
    print(f"  path_sum(0, 9) = {s} (expected {expected})")

    s = path_sum(nodes[3], nodes[7])
    expected = sum(range(4, 9))
    print(f"  path_sum(3, 7) = {s} (expected {expected})")

    # Cut in middle
    cut(nodes[4], nodes[5])
    print(f"\n  After cut(4, 5):")
    print(f"  connected(0, 4) = {connected(nodes[0], nodes[4])}")
    print(f"  connected(0, 9) = {connected(nodes[0], nodes[9])}")
    print(f"  connected(5, 9) = {connected(nodes[5], nodes[9])}")
    print(f"  path_sum(0, 4) = {path_sum(nodes[0], nodes[4])}")
    print(f"  path_sum(5, 9) = {path_sum(nodes[5], nodes[9])}")

    # Relink differently
    link(nodes[4], nodes[7])
    print(f"\n  After link(4, 7):")
    print(f"  connected(0, 9) = {connected(nodes[0], nodes[9])}")
    print(f"  path_sum(0, 9) = {path_sum(nodes[0], nodes[9])}")
    print()


# ============================================================
# Exercise 2: Dynamic Connectivity
# ============================================================
def exercise_2():
    """
    Handle a sequence of link/cut/connected operations.
    """
    print("=== Exercise 2: Dynamic Connectivity ===\n")

    n = 8
    nodes = [Node(val=1, node_id=i) for i in range(n)]

    operations = [
        ("link", 0, 1), ("link", 1, 2), ("link", 2, 3),
        ("query", 0, 3, True),
        ("query", 4, 5, False),
        ("link", 4, 5), ("link", 5, 6), ("link", 6, 7),
        ("query", 4, 7, True),
        ("query", 0, 7, False),
        ("link", 3, 4),
        ("query", 0, 7, True),
        ("cut", 2, 3),
        ("query", 0, 7, False),
        ("query", 0, 2, True),
    ]

    correct = 0
    total_queries = 0

    for op in operations:
        if op[0] == "link":
            _, u, v = op
            link(nodes[u], nodes[v])
            print(f"  link({u}, {v})")
        elif op[0] == "cut":
            _, u, v = op
            cut(nodes[u], nodes[v])
            print(f"  cut({u}, {v})")
        elif op[0] == "query":
            _, u, v, expected = op
            result = connected(nodes[u], nodes[v])
            ok = result == expected
            total_queries += 1
            if ok: correct += 1
            print(f"  connected({u}, {v}) = {result} "
                  f"(expected {expected}) {'✓' if ok else '✗'}")

    print(f"\n  {correct}/{total_queries} queries correct")
    print()


# ============================================================
# Exercise 3: Path Maximum with Updates
# ============================================================
def exercise_3():
    """
    Path queries with node value updates on dynamic trees.
    """
    print("=== Exercise 3: Path Queries with Updates ===\n")

    nodes = [Node(val=v, node_id=i) for i, v in enumerate([5, 3, 8, 1, 6])]

    # Build tree: 0-1-2, 0-3, 2-4
    link(nodes[0], nodes[1])
    link(nodes[1], nodes[2])
    link(nodes[0], nodes[3])
    link(nodes[2], nodes[4])

    print(f"  Tree: 3-0-1-2-4, with 0-3")
    print(f"  Values: [5, 3, 8, 1, 6]")

    # Path sums
    print(f"\n  path_sum(3, 4) = {path_sum(nodes[3], nodes[4])}")
    print(f"  path_sum(0, 2) = {path_sum(nodes[0], nodes[2])}")

    # Update node 1 value: 3 → 20
    print(f"\n  Update node 1: 3 → 20")
    access(nodes[1])
    nodes[1].val = 20
    _pull(nodes[1])

    print(f"  path_sum(0, 2) = {path_sum(nodes[0], nodes[2])}")
    print(f"  path_sum(3, 4) = {path_sum(nodes[3], nodes[4])}")
    print()


# ============================================================
# Exercise 4: Dynamic LCA
# ============================================================
def exercise_4():
    """
    Compute LCA on dynamic trees using Link-Cut Trees.
    """
    print("=== Exercise 4: Dynamic LCA ===\n")

    nodes = [Node(val=i, node_id=i) for i in range(8)]

    # Build tree:
    #       0
    #      / \
    #     1   2
    #    / \   \
    #   3   4   5
    #  /
    # 6
    for u, v in [(0,1),(0,2),(1,3),(1,4),(2,5),(3,6)]:
        link(nodes[u], nodes[v])

    def lca(u, v):
        """LCA via access: make_root is NOT called — use fixed root."""
        access(nodes[u])
        return access(nodes[v])

    # Test LCA queries (using node 0 as root)
    test_cases = [
        (6, 4, 1), (3, 5, 0), (6, 5, 0), (3, 4, 1), (4, 5, 0),
    ]

    print(f"  Tree rooted at 0\n")
    for u, v, expected in test_cases:
        result = lca(u, v)
        ok = result.id == expected
        print(f"  LCA({u}, {v}) = {result.id} "
              f"(expected {expected}) {'✓' if ok else '✗'}")

    # Dynamic LCA: add node 7 under node 5
    print(f"\n  Link node 7 under 5:")
    link(nodes[7], nodes[5])

    result = lca(7, 6)
    print(f"  LCA(7, 6) = {result.id} (expected 0)")
    result = lca(7, 5)
    print(f"  LCA(7, 5) = {result.id} (expected 5)")
    print()


# ============================================================
# Exercise 5: Dynamic MST
# ============================================================
def exercise_5():
    """
    Maintain MST as edges are added incrementally.
    """
    print("=== Exercise 5: Dynamic MST ===\n")

    # Edges in arbitrary order
    edges = [
        (4, 0, 1), (8, 1, 2), (3, 0, 2), (7, 2, 3),
        (1, 3, 4), (6, 1, 3), (2, 0, 4), (5, 1, 4),
    ]

    n = 5
    nodes = [Node(val=0, node_id=i) for i in range(n)]
    mst_edges = []
    mst_weight = 0

    print(f"  Edges: {edges}")
    print(f"  Building MST incrementally:\n")

    for w, u, v in edges:
        if not connected(nodes[u], nodes[v]):
            link(nodes[u], nodes[v])
            mst_edges.append((w, u, v))
            mst_weight += w
            print(f"    Add ({u}-{v}, w={w}) → MST weight = {mst_weight}")
        else:
            print(f"    Skip ({u}-{v}, w={w}) — cycle")

    print(f"\n  Final MST: {mst_edges}")
    print(f"  Total weight: {mst_weight}")

    # Verify: sort all edges by weight, Kruskal's result
    sorted_edges = sorted(edges)
    parent = list(range(n))
    def find(x):
        while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
        return x
    def union(a, b):
        a, b = find(a), find(b)
        if a != b: parent[a] = b; return True
        return False

    kruskal_weight = 0
    kruskal_edges = []
    for w, u, v in sorted_edges:
        if union(u, v):
            kruskal_weight += w
            kruskal_edges.append((w, u, v))

    print(f"\n  Kruskal's MST: {kruskal_edges}")
    print(f"  Kruskal weight: {kruskal_weight}")
    print(f"  Note: LCT MST may differ from Kruskal because edges arrive")
    print(f"  in non-sorted order. For true dynamic MST, use edge-weighted")
    print(f"  LCT nodes to swap heavier edges when cycles are found.")
    print()


if __name__ == "__main__":
    exercise_1()
    exercise_2()
    exercise_3()
    exercise_4()
    exercise_5()
