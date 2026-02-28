# Heavy-Light Decomposition

**Previous**: [Problem Solving in Practice](./29_Problem_Solving.md) | **Next**: [Link-Cut Tree](./31_Link_Cut_Tree.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain why naive path queries on trees are O(N) and how Heavy-Light Decomposition (HLD) reduces them to O(log²N)
2. Classify tree edges as heavy or light and prove that any root-to-leaf path crosses at most O(log N) light edges
3. Implement HLD to flatten a tree into a linear array and perform path queries/updates using a segment tree
4. Solve LCA (Lowest Common Ancestor) queries in O(log N) as a byproduct of HLD
5. Apply HLD to solve competitive programming problems involving path sums, path maximums, and subtree queries

---

## Table of Contents

1. [The Problem: Path Queries on Trees](#1-the-problem-path-queries-on-trees)
2. [Heavy and Light Edges](#2-heavy-and-light-edges)
3. [Decomposition Algorithm](#3-decomposition-algorithm)
4. [Mapping to a Segment Tree](#4-mapping-to-a-segment-tree)
5. [Path Query and Update](#5-path-query-and-update)
6. [Subtree Queries](#6-subtree-queries)
7. [Implementation](#7-implementation)
8. [Exercises](#8-exercises)

---

## 1. The Problem: Path Queries on Trees

Consider a tree with N nodes, each with a value. We need to:
- **Query**: find the sum (or max, min) of values along the path from node u to node v
- **Update**: change the value of a single node (or all nodes on a path)

**Naive approach**: walk the path u→LCA→v, visiting O(N) nodes in the worst case. With Q queries, total is O(NQ).

**Goal**: O(log²N) per query using HLD + segment tree.

---

## 2. Heavy and Light Edges

For each non-leaf node, classify its edges:

- **Heavy edge**: the edge to the child with the **largest subtree** (break ties arbitrarily)
- **Light edge**: all other edges to children

```
        1 (size=10)
       / \
      2   3 (size=6)    ← edge 1→3 is heavy (larger subtree)
     /   / \             ← edge 1→2 is light
    4   5   6 (size=3)   ← edge 3→6 is heavy
       / \
      7   8              ← edge 6 has no children
```

**Key property**: any path from root to a leaf crosses **at most O(log N) light edges**.

**Proof sketch**: when traversing a light edge from parent to child, the child's subtree is at most half the parent's subtree size (otherwise, the edge would be heavy). So the subtree size at least halves at each light edge, giving at most log₂N light edges.

**Heavy paths (chains)**: consecutive heavy edges form chains. These chains partition the tree into at most O(N) chains, but any root-to-leaf path visits at most O(log N) chains.

---

## 3. Decomposition Algorithm

### Step 1: DFS to Compute Subtree Sizes

```python
def dfs_size(node, parent):
    size[node] = 1
    for child in adj[node]:
        if child != parent:
            depth[child] = depth[node] + 1
            par[child] = node
            dfs_size(child, node)
            size[node] += size[child]
```

### Step 2: DFS to Assign Chains

Visit the **heavy child first** so that each chain gets consecutive positions:

```python
timer = 0

def dfs_hld(node, parent, chain_head):
    pos[node] = timer  # position in the flat array
    timer += 1
    head[node] = chain_head

    # Find heavy child (largest subtree)
    heavy = -1
    max_size = 0
    for child in adj[node]:
        if child != parent and size[child] > max_size:
            max_size = size[child]
            heavy = child

    # Visit heavy child first (continues the chain)
    if heavy != -1:
        dfs_hld(heavy, node, chain_head)

    # Visit light children (each starts a new chain)
    for child in adj[node]:
        if child != parent and child != heavy:
            dfs_hld(child, node, child)  # new chain
```

After this DFS, each chain occupies a contiguous range in `pos[]`.

---

## 4. Mapping to a Segment Tree

Build a segment tree over the flattened array `A[0..N-1]` where `A[pos[v]] = value[v]`.

Because each chain is contiguous:
- **Chain query**: segment tree query on `[pos[head[v]], pos[v]]`
- **Chain update**: segment tree update on `[pos[head[v]], pos[v]]`

Both are O(log N) per chain.

---

## 5. Path Query and Update

To query the path u → v:

```python
def path_query(u, v):
    result = 0  # identity for sum; -inf for max
    while head[u] != head[v]:
        # Move the deeper node's chain head up
        if depth[head[u]] < depth[head[v]]:
            u, v = v, u
        # Query the segment [head[u]..u] on the segment tree
        result = combine(result, seg_query(pos[head[u]], pos[u]))
        u = par[head[u]]  # jump to parent of chain head

    # Now u and v are on the same chain
    if depth[u] > depth[v]:
        u, v = v, u
    result = combine(result, seg_query(pos[u], pos[v]))
    return result
```

**Complexity**: O(log N) chains × O(log N) per segment tree query = **O(log²N)**.

---

## 6. Subtree Queries

A bonus of the Euler-tour ordering: the subtree of node v occupies `pos[v]` to `pos[v] + size[v] - 1` in the flat array.

```python
def subtree_query(v):
    return seg_query(pos[v], pos[v] + size[v] - 1)

def subtree_update(v, delta):
    seg_range_update(pos[v], pos[v] + size[v] - 1, delta)
```

This gives O(log N) subtree queries/updates for free.

---

## 7. Implementation

### Complete Python Implementation

```python
import sys
from sys import setrecursionlimit

def solve():
    # Build tree (1-indexed)
    N = int(input())
    adj = [[] for _ in range(N + 1)]
    values = [0] + list(map(int, input().split()))

    for _ in range(N - 1):
        u, v = map(int, input().split())
        adj[u].append(v)
        adj[v].append(u)

    # Step 1: compute sizes, parents, depths (iterative)
    size = [0] * (N + 1)
    par = [0] * (N + 1)
    depth = [0] * (N + 1)
    order = []

    stack = [(1, 0, False)]
    while stack:
        node, parent, visited = stack.pop()
        if visited:
            size[node] = 1
            for child in adj[node]:
                if child != parent:
                    size[node] += size[child]
            order.append(node)
            continue
        par[node] = parent
        stack.append((node, parent, True))
        for child in adj[node]:
            if child != parent:
                depth[child] = depth[node] + 1
                stack.append((child, node, False))

    # Step 2: HLD (iterative)
    pos = [0] * (N + 1)
    head = [0] * (N + 1)
    timer = 0

    stack = [(1, 1)]  # (node, chain_head)
    while stack:
        node, chain_head = stack.pop()
        pos[node] = timer
        timer += 1
        head[node] = chain_head

        # Find heavy child
        heavy = -1
        max_sz = 0
        for child in adj[node]:
            if child != par[node] and size[child] > max_sz:
                max_sz = size[child]
                heavy = child

        # Push light children first (processed last = DFS order)
        light_children = []
        for child in adj[node]:
            if child != par[node] and child != heavy:
                light_children.append(child)

        for child in reversed(light_children):
            stack.append((child, child))

        # Push heavy child last (processed next)
        if heavy != -1:
            stack.append((heavy, chain_head))

    # Build segment tree on flattened array
    seg = [0] * (4 * N)
    flat = [0] * N
    for v in range(1, N + 1):
        flat[pos[v]] = values[v]

    def build(node, lo, hi):
        if lo == hi:
            seg[node] = flat[lo]
            return
        mid = (lo + hi) // 2
        build(2 * node, lo, mid)
        build(2 * node + 1, mid + 1, hi)
        seg[node] = seg[2 * node] + seg[2 * node + 1]

    def query(node, lo, hi, ql, qr):
        if qr < lo or hi < ql:
            return 0
        if ql <= lo and hi <= qr:
            return seg[node]
        mid = (lo + hi) // 2
        return (query(2 * node, lo, mid, ql, qr) +
                query(2 * node + 1, mid + 1, hi, ql, qr))

    def update(node, lo, hi, idx, val):
        if lo == hi:
            seg[node] = val
            return
        mid = (lo + hi) // 2
        if idx <= mid:
            update(2 * node, lo, mid, idx, val)
        else:
            update(2 * node + 1, mid + 1, hi, idx, val)
        seg[node] = seg[2 * node] + seg[2 * node + 1]

    build(1, 0, N - 1)

    def path_sum(u, v):
        result = 0
        while head[u] != head[v]:
            if depth[head[u]] < depth[head[v]]:
                u, v = v, u
            result += query(1, 0, N - 1, pos[head[u]], pos[u])
            u = par[head[u]]
        if depth[u] > depth[v]:
            u, v = v, u
        result += query(1, 0, N - 1, pos[u], pos[v])
        return result

    # Process queries
    Q = int(input())
    for _ in range(Q):
        parts = input().split()
        if parts[0] == 'Q':
            u, v = int(parts[1]), int(parts[2])
            print(path_sum(u, v))
        else:  # Update
            v, val = int(parts[1]), int(parts[2])
            update(1, 0, N - 1, pos[v], val)
```

### Complexity Summary

| Operation | Time | Space |
|-----------|------|-------|
| Preprocessing (DFS + HLD) | O(N) | O(N) |
| Build segment tree | O(N) | O(N) |
| Path query | O(log²N) | — |
| Path update | O(log²N) | — |
| Point update | O(log N) | — |
| Subtree query | O(log N) | — |

---

## 8. Exercises

### Exercise 1: Basic HLD Construction

Given a tree with N nodes, implement HLD and verify:
1. Each chain is contiguous in the flat array
2. The number of light edges on any root-to-leaf path is ≤ log₂N
3. The subtree of any node occupies a contiguous range

### Exercise 2: Path Sum Queries

Implement the full HLD + segment tree for path sum queries. Test on:
- A linear chain (worst case for naive, still O(log²N) for HLD)
- A balanced binary tree
- A star graph

### Exercise 3: Path Maximum Query

Modify the implementation for path maximum instead of path sum. Use this to solve: "What is the maximum edge weight on the path from u to v?"

*Hint*: assign each edge's weight to the child node (the deeper endpoint).

### Exercise 4: LCA via HLD

Implement LCA using HLD (without a separate LCA structure). Show that the LCA of u and v is the node at which the path query loop terminates.

### Exercise 5: Subtree Sum with Updates

Support two operations:
1. Add a value to all nodes in the subtree of v
2. Query the sum of all values in the subtree of v

Use HLD's contiguous subtree property with a lazy segment tree.

---

## Navigation

**Previous**: [Problem Solving in Practice](./29_Problem_Solving.md) | **Next**: [Link-Cut Tree](./31_Link_Cut_Tree.md)
