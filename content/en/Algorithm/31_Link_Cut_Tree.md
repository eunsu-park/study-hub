# Link-Cut Tree

**Previous**: [Heavy-Light Decomposition](./30_Heavy_Light_Decomposition.md) | **Next**: [Persistent Segment Tree](./32_Persistent_Segment_Tree.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain why static HLD fails for dynamic trees and how Link-Cut Trees support link/cut operations in O(log N) amortized time
2. Describe the preferred-path decomposition and how splay trees represent each preferred path
3. Implement the `access`, `link`, and `cut` operations and analyze their amortized O(log N) complexity
4. Use Link-Cut Trees to solve dynamic connectivity queries on forests
5. Apply Link-Cut Trees to problems involving dynamic LCA, path aggregates, and incremental tree construction

---

## Table of Contents

1. [Motivation: Dynamic Trees](#1-motivation-dynamic-trees)
2. [Preferred Paths and Auxiliary Trees](#2-preferred-paths-and-auxiliary-trees)
3. [Splay Tree Review](#3-splay-tree-review)
4. [Core Operations](#4-core-operations)
5. [Path Queries on Dynamic Trees](#5-path-queries-on-dynamic-trees)
6. [Implementation](#6-implementation)
7. [Applications](#7-applications)
8. [Exercises](#8-exercises)

---

## 1. Motivation: Dynamic Trees

HLD (Lesson 30) is powerful but **static** — the tree structure cannot change. Many problems require:

- **Link(u, v)**: add an edge between two trees in a forest
- **Cut(u, v)**: remove an edge, splitting a tree into two
- **Path query**: aggregate values along u→v path
- **Connected(u, v)**: are u and v in the same tree?

**Examples**: dynamic connectivity, network design, incremental MST.

| Feature | HLD | Link-Cut Tree |
|---------|-----|---------------|
| Path query | O(log²N) | O(log N) amortized |
| Point update | O(log N) | O(log N) amortized |
| Link/Cut | Not supported | O(log N) amortized |
| Construction | O(N) one-time | O(N log N) |

---

## 2. Preferred Paths and Auxiliary Trees

### Preferred Child

At any moment, each node has at most one **preferred child** — the child most recently accessed on a root-to-node path. The edge to the preferred child is a **preferred edge**.

**Preferred paths**: maximal chains of preferred edges form paths from some node down toward leaves. These partition the tree into disjoint preferred paths.

```
         1
        / \
       2   3        Preferred edges: 1→3, 3→5
      /   / \       Preferred paths: [1,3,5], [2], [4], [6]
     4   5   6
```

### Auxiliary Trees (Splay Trees)

Each preferred path is stored in a splay tree, keyed by **depth** (nodes deeper in the original tree have larger keys in the splay tree).

The splay trees are connected: the root of each splay tree has a **path-parent pointer** to the node where its preferred path attaches in the represented tree. This is a one-way pointer (parent doesn't know about it).

---

## 3. Splay Tree Review

A splay tree is a self-balancing BST where every access operation moves the accessed node to the root via rotations:

- **Zig**: single rotation (when parent is root)
- **Zig-zig**: two rotations same direction (node and parent on same side)
- **Zig-zag**: two rotations opposite direction

**Amortized O(log N)** per operation (proven via potential function).

Key for Link-Cut Trees:
```
splay(x): rotate x to the root of its auxiliary tree
```

---

## 4. Core Operations

### `access(v)` — The Fundamental Operation

`access(v)` makes v the deepest node on its preferred path from the root. This changes preferred edges along the way.

**Algorithm**:
1. Splay v in its auxiliary tree
2. Disconnect v's right child (deeper nodes no longer preferred)
3. Walk up via path-parent pointers:
   - At each step, splay the path-parent node
   - Connect v's tree as its right child
   - Splay v again
4. Repeat until v is in the root's auxiliary tree

```
access(v):
    splay(v)
    v.right = null   // cut preferred path below v
    while v.path_parent != null:
        w = v.path_parent
        splay(w)
        w.right = v   // make v's path preferred
        splay(v)       // v becomes root
```

After `access(v)`, the path from the root to v consists entirely of preferred edges and is stored in v's splay tree.

### `link(u, v)` — Connect Two Trees

Make u a child of v (u must be a root):

```
link(u, v):
    access(u)    // u is now root of its splay tree, no right child
    access(v)    // v is now the deepest on root's preferred path
    u.left = v   // v becomes u's left child in the splay tree
    v.parent = u // (but v's tree might need path-parent adjustment)
```

Actually, the standard implementation:
```
link(u, v):
    make_root(u)  // reroot u's tree at u
    access(v)
    u.path_parent = v
```

### `cut(u, v)` — Disconnect an Edge

Remove the edge between u and v:

```
cut(u, v):
    make_root(u)
    access(v)
    // Now u is v's left child in the splay tree
    v.left = null
    u.parent = null
```

### `make_root(v)` — Reroot the Tree

Make v the root of its represented tree by reversing the path from v to the current root:

```
make_root(v):
    access(v)
    reverse(v's splay tree)  // flip left/right subtrees
    // This reverses the depth ordering, making v the shallowest
```

### `find_root(v)` — Find the Root

```
find_root(v):
    access(v)
    // Go to leftmost node (shallowest = root)
    while v.left != null:
        v = v.left
    splay(v)
    return v
```

### `connected(u, v)`

```
connected(u, v):
    return find_root(u) == find_root(v)
```

---

## 5. Path Queries on Dynamic Trees

Maintain aggregates (sum, max, min) in splay tree nodes:

```
path_aggregate(u, v):
    make_root(u)
    access(v)
    return v.aggregate  // aggregate of entire splay tree = path u→v
```

Each splay tree node stores:
- `val`: the node's value
- `agg`: aggregate of the node's subtree in the splay tree

Update aggregate during rotations:
```
pull(v):
    v.agg = combine(v.left.agg, v.val, v.right.agg)
```

---

## 6. Implementation

### Python Implementation (Simplified)

```python
class Node:
    __slots__ = ('ch', 'p', 'rev', 'val', 'agg', 'sz')

    def __init__(self, val=0):
        self.ch = [None, None]  # left, right children
        self.p = None            # parent (in splay tree or path-parent)
        self.rev = False         # lazy reverse flag
        self.val = val
        self.agg = val
        self.sz = 1

def is_root(x):
    """Is x the root of its auxiliary (splay) tree?"""
    p = x.p
    return p is None or (p.ch[0] != x and p.ch[1] != x)

def pull(x):
    """Update aggregate from children."""
    if x is None:
        return
    x.agg = x.val
    x.sz = 1
    for c in x.ch:
        if c is not None:
            x.agg = x.agg + c.agg  # sum; change for max/min
            x.sz += c.sz

def push(x):
    """Propagate lazy reverse."""
    if x is not None and x.rev:
        x.ch[0], x.ch[1] = x.ch[1], x.ch[0]
        for c in x.ch:
            if c is not None:
                c.rev = not c.rev
        x.rev = False

def rotate(x):
    """Single rotation."""
    p = x.p
    g = p.p
    d = 0 if p.ch[1] == x else 1  # direction

    # x's opposite child becomes p's child
    p.ch[1 - d] = x.ch[d]
    if x.ch[d] is not None:
        x.ch[d].p = p

    # x becomes parent
    x.ch[d] = p
    p.p = x
    x.p = g

    if g is not None:
        if g.ch[0] == p:
            g.ch[0] = x
        elif g.ch[1] == p:
            g.ch[1] = x
        # else: path-parent pointer (don't modify)

    pull(p)
    pull(x)

def splay(x):
    """Splay x to the root of its auxiliary tree."""
    # Push lazy flags from root down to x
    stack = []
    y = x
    while not is_root(y):
        stack.append(y.p)
        y = y.p
    stack.append(y)
    while stack:
        push(stack.pop())

    while not is_root(x):
        p = x.p
        if not is_root(p):
            g = p.p
            # Zig-zig or zig-zag
            same_dir = (g.ch[0] == p) == (p.ch[0] == x)
            if same_dir:
                rotate(p)  # zig-zig: rotate parent first
            else:
                rotate(x)  # zig-zag: rotate x first
        rotate(x)

def access(x):
    """Make x the deepest node on the preferred path from root."""
    last = None
    y = x
    while y is not None:
        splay(y)
        y.ch[1] = last  # change preferred child
        pull(y)
        last = y
        y = y.p
    splay(x)

def make_root(x):
    """Make x the root of its represented tree."""
    access(x)
    x.rev = not x.rev
    push(x)

def find_root(x):
    """Find the root of x's tree."""
    access(x)
    while x.ch[0] is not None:
        push(x)
        x = x.ch[0]
    splay(x)
    return x

def link(x, y):
    """Add edge between x and y (x and y must be in different trees)."""
    make_root(x)
    x.p = y

def cut(x, y):
    """Remove edge between x and y."""
    make_root(x)
    access(y)
    y.ch[0] = None
    x.p = None
    pull(y)

def connected(x, y):
    """Are x and y in the same tree?"""
    return find_root(x) == find_root(y)

def path_aggregate(x, y):
    """Aggregate values on the path from x to y."""
    make_root(x)
    access(y)
    return y.agg
```

### Complexity

| Operation | Amortized Time |
|-----------|----------------|
| access | O(log N) |
| link | O(log N) |
| cut | O(log N) |
| find_root | O(log N) |
| path_aggregate | O(log N) |
| connected | O(log N) |

---

## 7. Applications

### Dynamic Connectivity

```python
# Maintain a forest with link/cut operations
# Query: are u and v connected?
nodes = [Node(i) for i in range(N)]

link(nodes[0], nodes[1])
link(nodes[1], nodes[2])
print(connected(nodes[0], nodes[2]))  # True

cut(nodes[0], nodes[1])
print(connected(nodes[0], nodes[2]))  # False
```

### Dynamic MST

Maintain a spanning tree of a graph. When a new edge (u, v, w) is added:
1. If u and v are not connected, link them
2. If connected, find the maximum weight edge on the path u→v
3. If new weight < max weight, cut the old edge and link the new one

### Network Flow (Dinic's with Link-Cut Tree)

Dinic's algorithm can use Link-Cut Trees to maintain augmenting paths, improving the complexity for unit-capacity graphs.

---

## 8. Exercises

### Exercise 1: Basic Link-Cut Tree Operations

Implement a Link-Cut Tree and verify with a sequence of operations:
1. Create 10 nodes
2. Link them into a path: 1→2→3→...→10
3. Query path sum from 1 to 10
4. Cut edge 5→6, verify disconnection
5. Link 5 to 8, verify new connections

### Exercise 2: Dynamic Connectivity

Given a forest that undergoes link and cut operations, answer connectivity queries. Compare the performance with Union-Find (which only supports links, not cuts).

### Exercise 3: Path Maximum with Updates

Support:
1. `update(v, new_val)`: change node v's value
2. `query(u, v)`: find the maximum value on the path u→v
3. `link(u, v)` and `cut(u, v)`: dynamic tree modifications

### Exercise 4: Dynamic LCA

Implement LCA queries on a dynamic tree using Link-Cut Trees. After `make_root(u)` and `access(v)`, the last node splayed before v in the access path is the LCA.

### Exercise 5: Minimum Spanning Tree Maintenance

Implement a dynamic MST algorithm:
1. Start with an empty graph
2. Edges are added one at a time
3. Maintain the MST, replacing edges when a better one is found
4. Use Link-Cut Tree to find the maximum edge on the path

---

## Navigation

**Previous**: [Heavy-Light Decomposition](./30_Heavy_Light_Decomposition.md) | **Next**: [Persistent Segment Tree](./32_Persistent_Segment_Tree.md)
