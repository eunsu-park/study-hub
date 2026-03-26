"""
Example: Link-Cut Tree
Topic: Algorithm – Lesson 31

Demonstrates:
  1. Splay tree-based Link-Cut Tree operations
  2. Dynamic link/cut on forests
  3. Path aggregate queries
  4. Dynamic connectivity

Run: python 31_link_cut_tree.py
"""


# ============================================================
# Link-Cut Tree Implementation
# ============================================================
class LCTNode:
    """Node in a Link-Cut Tree."""
    __slots__ = ('ch', 'p', 'rev', 'val', 'agg', 'sz', 'id')

    def __init__(self, val=0, node_id=-1):
        self.ch = [None, None]  # [left, right] in splay tree
        self.p = None            # parent (splay parent or path-parent)
        self.rev = False         # lazy reverse flag
        self.val = val           # node value
        self.agg = val           # aggregate of splay subtree
        self.sz = 1
        self.id = node_id


def _is_root(x):
    """Is x the root of its auxiliary splay tree?"""
    if x.p is None:
        return True
    return x.p.ch[0] != x and x.p.ch[1] != x


def _pull(x):
    """Update aggregate from children."""
    if x is None:
        return
    x.agg = x.val
    x.sz = 1
    if x.ch[0]:
        x.agg += x.ch[0].agg
        x.sz += x.ch[0].sz
    if x.ch[1]:
        x.agg += x.ch[1].agg
        x.sz += x.ch[1].sz


def _push(x):
    """Propagate lazy reverse."""
    if x and x.rev:
        x.ch[0], x.ch[1] = x.ch[1], x.ch[0]
        for c in x.ch:
            if c:
                c.rev = not c.rev
        x.rev = False


def _rotate(x):
    """Rotate x up in its splay tree."""
    p = x.p
    g = p.p
    d = 0 if p.ch[0] == x else 1

    p.ch[d] = x.ch[1 - d]
    if x.ch[1 - d]:
        x.ch[1 - d].p = p

    x.ch[1 - d] = p
    p.p = x
    x.p = g

    if g:
        if g.ch[0] == p:
            g.ch[0] = x
        elif g.ch[1] == p:
            g.ch[1] = x

    _pull(p)
    _pull(x)


def _splay(x):
    """Splay x to root of its auxiliary tree."""
    # Collect ancestors for lazy propagation
    path = []
    y = x
    while not _is_root(y):
        path.append(y.p)
        y = y.p
    path.append(y)

    while path:
        _push(path.pop())
    _push(x)

    while not _is_root(x):
        p = x.p
        if not _is_root(p):
            g = p.p
            same = (g.ch[0] == p) == (p.ch[0] == x)
            if same:
                _rotate(p)
            else:
                _rotate(x)
        _rotate(x)


def access(x):
    """Make x the deepest node on the preferred path from root."""
    last = None
    y = x
    while y:
        _splay(y)
        y.ch[1] = last
        _pull(y)
        last = y
        y = y.p
    _splay(x)
    return last


def make_root(x):
    """Make x the root of its represented tree."""
    access(x)
    x.rev = not x.rev
    _push(x)


def find_root(x):
    """Find the root of x's represented tree."""
    access(x)
    while x.ch[0]:
        _push(x)
        x = x.ch[0]
    _splay(x)
    return x


def link(x, y):
    """Add edge between x and y (must be in different trees)."""
    make_root(x)
    x.p = y


def cut(x, y):
    """Remove edge between x and y."""
    make_root(x)
    access(y)
    if y.ch[0] == x:
        y.ch[0] = None
        x.p = None
        _pull(y)


def connected(x, y):
    """Are x and y in the same tree?"""
    return find_root(x) is find_root(y)


def path_aggregate(x, y):
    """Sum of values on the path from x to y."""
    make_root(x)
    access(y)
    return y.agg


# ============================================================
# Demo 1: Basic Link-Cut Operations
# ============================================================
def demo_basic():
    print("=" * 60)
    print("Demo 1: Basic Link-Cut Tree Operations")
    print("=" * 60)

    # Create 6 nodes with values
    nodes = [LCTNode(val=i + 1, node_id=i) for i in range(6)]

    print(f"\n  Nodes: {[(n.id, n.val) for n in nodes]}")

    # Build tree: 0-1-2-3, 0-4, 2-5
    link(nodes[0], nodes[1])
    link(nodes[1], nodes[2])
    link(nodes[2], nodes[3])
    link(nodes[0], nodes[4])
    link(nodes[2], nodes[5])

    print(f"  Edges: 0-1, 1-2, 2-3, 0-4, 2-5")
    print(f"  Tree shape:")
    print(f"      4(5)")
    print(f"      |")
    print(f"      0(1)--1(2)--2(3)--3(4)")
    print(f"                   |")
    print(f"                  5(6)")

    # Connectivity queries
    print(f"\n  Connectivity:")
    for u, v in [(0, 3), (4, 5), (0, 5)]:
        c = connected(nodes[u], nodes[v])
        print(f"    connected({u}, {v}) = {c}")

    # Path aggregate queries
    print(f"\n  Path sums:")
    for u, v in [(0, 3), (4, 5), (0, 5), (3, 4)]:
        s = path_aggregate(nodes[u], nodes[v])
        print(f"    path_sum({u}, {v}) = {s}")

    # Cut and re-check
    print(f"\n  Cut edge 1-2:")
    cut(nodes[1], nodes[2])
    print(f"    connected(0, 3) = {connected(nodes[0], nodes[3])}")
    print(f"    connected(0, 1) = {connected(nodes[0], nodes[1])}")
    print(f"    connected(2, 5) = {connected(nodes[2], nodes[5])}")

    # Relink
    print(f"\n  Link 1-5:")
    link(nodes[1], nodes[5])
    print(f"    connected(0, 3) = {connected(nodes[0], nodes[3])}")
    print(f"    path_sum(4, 3) = {path_aggregate(nodes[4], nodes[3])}")
    print()


# ============================================================
# Demo 2: Dynamic Forest Operations
# ============================================================
def demo_dynamic_forest():
    print("=" * 60)
    print("Demo 2: Dynamic Forest")
    print("=" * 60)

    n = 10
    nodes = [LCTNode(val=1, node_id=i) for i in range(n)]

    print(f"\n  {n} isolated nodes (all val=1)")

    # Build a path: 0-1-2-3-4
    operations = [
        ("link", 0, 1),
        ("link", 1, 2),
        ("link", 2, 3),
        ("link", 3, 4),
    ]

    for op, u, v in operations:
        link(nodes[u], nodes[v])
    print(f"  Built path: 0-1-2-3-4")
    print(f"  path_sum(0, 4) = {path_aggregate(nodes[0], nodes[4])}")

    # Build another tree: 5-6-7
    link(nodes[5], nodes[6])
    link(nodes[6], nodes[7])
    print(f"  Built path: 5-6-7")

    # Connect the two trees
    print(f"\n  connected(0, 7) = {connected(nodes[0], nodes[7])}")
    link(nodes[4], nodes[5])
    print(f"  After link(4, 5):")
    print(f"    connected(0, 7) = {connected(nodes[0], nodes[7])}")
    print(f"    path_sum(0, 7) = {path_aggregate(nodes[0], nodes[7])}")

    # Cut in the middle
    cut(nodes[2], nodes[3])
    print(f"\n  After cut(2, 3):")
    print(f"    connected(0, 2) = {connected(nodes[0], nodes[2])}")
    print(f"    connected(0, 7) = {connected(nodes[0], nodes[7])}")
    print(f"    path_sum(3, 7) = {path_aggregate(nodes[3], nodes[7])}")
    print()


# ============================================================
# Demo 3: MST Maintenance
# ============================================================
def demo_mst():
    print("=" * 60)
    print("Demo 3: Dynamic MST Maintenance")
    print("=" * 60)

    # Edges sorted by weight
    edges = [
        (1, 0, 1), (2, 1, 2), (3, 0, 2), (4, 2, 3),
        (5, 1, 3), (1, 3, 4), (6, 0, 4),
    ]

    n = 5
    # For MST with LCT, we represent edges as nodes too
    # But for simplicity, we track edges separately
    nodes = [LCTNode(val=0, node_id=i) for i in range(n)]

    mst_weight = 0
    mst_edges = []

    print(f"\n  Adding edges to build MST:")
    for w, u, v in sorted(edges):
        if not connected(nodes[u], nodes[v]):
            link(nodes[u], nodes[v])
            mst_weight += w
            mst_edges.append((w, u, v))
            print(f"    Add edge ({u}-{v}, w={w}) → MST weight = {mst_weight}")
        else:
            print(f"    Skip edge ({u}-{v}, w={w}) — already connected")

    print(f"\n  Final MST weight: {mst_weight}")
    print(f"  MST edges: {mst_edges}")
    print()


if __name__ == "__main__":
    demo_basic()
    demo_dynamic_forest()
    demo_mst()
