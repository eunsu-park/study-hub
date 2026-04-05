# Register Allocation

**Previous**: [18. SSA Form](./18_SSA_Form.md) | **Next**: [20. LLVM IR Introduction](./20_LLVM_IR_Introduction.md)

---

Register allocation is the process of mapping an unbounded number of virtual registers (or program variables) to a finite set of physical machine registers. It is one of the most critical phases of compilation -- poor register allocation can multiply execution time by orders of magnitude due to memory spills.

This lesson covers the two dominant approaches: graph coloring and linear scan, along with practical concerns like live range splitting, spilling strategies, and register coalescing.

**Difficulty**: ⭐⭐⭐⭐

**Prerequisites**: [11. Code Generation](./11_Code_Generation.md), [18. SSA Form](./18_SSA_Form.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain why register allocation is NP-complete in general
2. Build interference graphs from live ranges
3. Implement graph coloring register allocation (Chaitin-Briggs)
4. Implement linear scan register allocation
5. Apply live range splitting and spilling strategies
6. Understand register coalescing and its interaction with coloring

---

## Table of Contents

1. [The Register Allocation Problem](#1-the-register-allocation-problem)
2. [Liveness Analysis](#2-liveness-analysis)
3. [Interference Graphs](#3-interference-graphs)
4. [Graph Coloring Allocation](#4-graph-coloring-allocation)
5. [Spilling](#5-spilling)
6. [Coalescing](#6-coalescing)
7. [Linear Scan Allocation](#7-linear-scan-allocation)
8. [SSA-Based Register Allocation](#8-ssa-based-register-allocation)
9. [Summary](#9-summary)
10. [Exercises](#10-exercises)
11. [References](#11-references)

---

## 1. The Register Allocation Problem

### 1.1 Why Register Allocation Matters

Modern processors have 16-32 general-purpose registers (x86-64 has 16, ARM64 has 31). A typical function may use hundreds of virtual registers. The allocator must:

- Map virtual registers to physical registers
- Ensure no two simultaneously live values share a register
- Minimize memory accesses (spills and reloads)
- Respect architectural constraints (e.g., certain instructions require specific registers)

```
Virtual registers: v1, v2, v3, v4, v5, v6, ...
Physical registers: r0, r1, r2 (only 3 available)

Challenge: Map many virtual registers to few physical registers
           without breaking the program semantics.
```

### 1.2 Complexity

Register allocation is equivalent to graph coloring, which is NP-complete for general graphs. However, real programs produce interference graphs with special structure that heuristics handle well in practice.

### 1.3 Live Ranges

A **live range** for a variable is the set of program points where the variable holds a value that may be used later:

```
1: v1 = 10        # v1 live range starts
2: v2 = 20        # v2 live range starts
3: v3 = v1 + v2   # v1, v2 used; v3 starts
4: v4 = v3 * 2    # v3 used; v4 starts
5: print(v4)      # v4 used
```

Live ranges:
- v1: [1, 3]
- v2: [2, 3]
- v3: [3, 4]
- v4: [4, 5]

Maximum simultaneously live = 2 (at point 3: v1 and v2), so 2 registers suffice.

---

## 2. Liveness Analysis

### 2.1 Data-Flow Equations

Liveness is a backward data-flow analysis:

```
LiveOut(B) = Union of LiveIn(S) for all successors S of B
LiveIn(B)  = Use(B) ∪ (LiveOut(B) - Def(B))
```

Where:
- `Use(B)` = variables used in B before any definition in B
- `Def(B)` = variables defined in B

### 2.2 Implementation

```python
def liveness_analysis(cfg):
    """
    Compute live-in and live-out sets for each block.
    Returns: (live_in, live_out) dicts mapping block -> set of variables
    """
    live_in = {b: set() for b in cfg.blocks}
    live_out = {b: set() for b in cfg.blocks}

    changed = True
    while changed:
        changed = False
        for block in reversed(cfg.blocks):  # reverse order for efficiency
            # LiveOut = union of LiveIn of successors
            new_out = set()
            for succ in block.successors:
                new_out |= live_in[succ]

            # LiveIn = Use ∪ (LiveOut - Def)
            new_in = block.use_set | (new_out - block.def_set)

            if new_in != live_in[block] or new_out != live_out[block]:
                live_in[block] = new_in
                live_out[block] = new_out
                changed = True

    return live_in, live_out


def compute_live_intervals(cfg, live_in, live_out):
    """
    Compute live intervals (start, end) for each variable.
    Uses instruction numbering across all blocks.
    """
    intervals = {}  # var -> (start, end)

    for block in cfg.blocks:
        live = set(live_out[block])

        # Walk backward through instructions
        for instr in reversed(block.instructions):
            pos = instr.position

            # Definition: end of live range (or start if first seen)
            if instr.dest:
                if instr.dest in live:
                    live.discard(instr.dest)
                if instr.dest not in intervals:
                    intervals[instr.dest] = [pos, pos]
                else:
                    intervals[instr.dest][0] = min(intervals[instr.dest][0], pos)

            # Uses: extend live range
            for use in instr.uses:
                live.add(use)
                if use not in intervals:
                    intervals[use] = [pos, pos]
                else:
                    intervals[use][0] = min(intervals[use][0], pos)
                    intervals[use][1] = max(intervals[use][1], pos)

    return intervals
```

---

## 3. Interference Graphs

### 3.1 Definition

An **interference graph** G = (V, E) where:
- Each node represents a virtual register (or live range)
- An edge (u, v) exists if u and v are simultaneously live at some program point

Two variables that interfere cannot share a register.

### 3.2 Building the Interference Graph

```python
def build_interference_graph(cfg, live_out):
    """
    Build interference graph by walking each block backward.
    """
    graph = InterferenceGraph()

    for block in cfg.blocks:
        live = set(live_out[block])

        for instr in reversed(block.instructions):
            if instr.dest:
                # The defined variable interferes with everything live
                # (except itself and, for copy instructions, the source)
                for var in live:
                    if var != instr.dest:
                        if instr.is_copy and var == instr.source:
                            continue  # Don't add interference for copies
                        graph.add_edge(instr.dest, var)
                live.discard(instr.dest)

            for use in instr.uses:
                live.add(use)

    return graph


class InterferenceGraph:
    def __init__(self):
        self.nodes = set()
        self.edges = set()
        self.adj = {}  # node -> set of neighbors

    def add_edge(self, u, v):
        self.nodes.add(u)
        self.nodes.add(v)
        if u != v:
            self.edges.add((min(u, v), max(u, v)))
            self.adj.setdefault(u, set()).add(v)
            self.adj.setdefault(v, set()).add(u)

    def degree(self, node):
        return len(self.adj.get(node, set()))

    def remove_node(self, node):
        for neighbor in list(self.adj.get(node, set())):
            self.adj[neighbor].discard(node)
        del self.adj[node]
        self.nodes.discard(node)
```

### 3.3 Example

```
Instructions:        Live sets:
1: a = ...           {a}
2: b = ...           {a, b}
3: c = a + b         {c}         (a,b dead after use)
4: d = ...           {c, d}
5: e = c + d         {e}
6: return e          {}

Interference graph:
  a --- b     (both live at point 2)
  c --- d     (both live at point 4)

  (a, c, d, e have no mutual interference except c-d)

Chromatic number = 2 (two registers suffice)
```

---

## 4. Graph Coloring Allocation

### 4.1 Chaitin's Algorithm (1981)

The basic idea: k-color the interference graph where k = number of physical registers.

```python
def chaitin_allocate(graph, k):
    """
    Chaitin's register allocation via graph coloring.
    k: number of available physical registers
    """
    stack = []

    # Simplify: iteratively remove nodes with degree < k
    work_graph = graph.copy()
    while work_graph.nodes:
        # Find a node with degree < k
        low_degree = None
        for node in work_graph.nodes:
            if work_graph.degree(node) < k:
                low_degree = node
                break

        if low_degree:
            stack.append(low_degree)
            work_graph.remove_node(low_degree)
        else:
            # No low-degree node: must spill
            victim = select_spill(work_graph)
            stack.append(('spill', victim))
            work_graph.remove_node(victim)

    # Select: pop nodes and assign colors
    coloring = {}
    for item in reversed(stack):
        if isinstance(item, tuple) and item[0] == 'spill':
            node = item[1]
            # Try to color; if impossible, actually spill
            used = {coloring[n] for n in graph.adj.get(node, set()) if n in coloring}
            available = set(range(k)) - used
            if available:
                coloring[node] = min(available)  # Optimistic: might color after all
            else:
                coloring[node] = 'SPILL'
        else:
            node = item
            used = {coloring[n] for n in graph.adj.get(node, set()) if n in coloring}
            available = set(range(k)) - used
            coloring[node] = min(available)

    return coloring
```

### 4.2 Briggs's Improvement: Optimistic Coloring

Briggs (1994) improved Chaitin's algorithm by being **optimistic** about spill candidates:

```python
def briggs_allocate(graph, k):
    """
    Briggs's optimistic coloring.
    Difference from Chaitin: push potential spills onto stack
    instead of immediately spilling. They might be colorable
    when popped (neighbors may have been removed).
    """
    stack = []
    work_graph = graph.copy()

    while work_graph.nodes:
        # Phase 1: Remove nodes with degree < k
        found = True
        while found:
            found = False
            for node in list(work_graph.nodes):
                if work_graph.degree(node) < k:
                    stack.append(node)
                    work_graph.remove_node(node)
                    found = True

        if work_graph.nodes:
            # Phase 2: Optimistically push a potential spill
            victim = select_spill(work_graph)
            stack.append(victim)  # Not marked as spill yet!
            work_graph.remove_node(victim)

    # Select phase: try to color everything
    coloring = {}
    for node in reversed(stack):
        used = {coloring[n] for n in graph.adj.get(node, set()) if n in coloring}
        available = set(range(k)) - used
        if available:
            coloring[node] = min(available)
        else:
            coloring[node] = 'SPILL'  # Only spill if truly uncolorable

    return coloring
```

### 4.3 Spill Cost Heuristics

```python
def select_spill(graph):
    """
    Choose which variable to spill.
    Heuristic: minimize (use_count / degree) -- spill the variable
    with the most interference but fewest uses.
    """
    best = None
    best_score = float('inf')

    for node in graph.nodes:
        if node.is_precolored:
            continue  # Never spill physical registers
        score = node.use_count / max(graph.degree(node), 1)
        if score < best_score:
            best_score = score
            best = node

    return best
```

---

## 5. Spilling

### 5.1 What is Spilling?

When a variable cannot be assigned a register, it is **spilled** to a stack slot in memory. The allocator inserts:
- **Store** instructions after each definition (write to stack)
- **Load** instructions before each use (read from stack)

```
# Before spilling (v3 spilled):
v3 = v1 + v2
v4 = v3 * 2

# After spilling:
v3 = v1 + v2
store v3, [sp + offset]     # spill store
v3_reload = load [sp + offset]  # spill load
v4 = v3_reload * 2
```

### 5.2 Spill Everywhere vs. Spill Around

**Spill Everywhere**: Insert load before every use and store after every def. Simple but generates many memory operations.

**Spill Around**: Only spill in regions where register pressure is high. Keep the value in a register where possible.

### 5.3 Rematerialization

Instead of loading a spilled value from memory, **rematerialization** recomputes it if cheaper:

```python
def can_rematerialize(instr):
    """
    A value can be rematerialized if:
    - It's a constant load
    - It's a simple computation with available operands
    - Recomputing is cheaper than a memory load
    """
    if instr.is_constant_load:
        return True
    if instr.is_address_computation and all_operands_available(instr):
        return True
    return False
```

---

## 6. Coalescing

### 6.1 Copy Coalescing

Many programs contain copy instructions (`a = b`). If `a` and `b` don't interfere, we can assign them the same register and eliminate the copy:

```python
def coalesce(graph, copies):
    """
    Aggressive coalescing: merge copy-related variables
    that don't interfere.
    """
    for src, dst in copies:
        if not graph.has_edge(src, dst):
            # Safe to coalesce: merge src into dst
            graph.merge_nodes(src, dst)
            # All references to src become dst
```

### 6.2 Conservative Coalescing (Briggs's Criterion)

Aggressive coalescing can increase the degree of the merged node, making it uncolorable. Briggs's criterion: coalesce only if the merged node has fewer than k neighbors with degree >= k.

```python
def briggs_coalesce(graph, src, dst, k):
    """
    Coalesce src and dst only if the result has fewer than k
    high-degree neighbors.
    """
    # Compute neighbors of merged node
    merged_neighbors = (graph.adj.get(src, set()) | graph.adj.get(dst, set())) - {src, dst}

    high_degree_count = sum(1 for n in merged_neighbors if graph.degree(n) >= k)

    if high_degree_count < k:
        graph.merge_nodes(src, dst)
        return True
    return False
```

### 6.3 George's Criterion

An alternative: coalesce a and b if every neighbor t of a either interferes with b or has degree < k.

```python
def george_coalesce(graph, a, b, k):
    """George's coalescing criterion."""
    for t in graph.adj.get(a, set()):
        if t == b:
            continue
        if not graph.has_edge(t, b) and graph.degree(t) >= k:
            return False  # Unsafe
    graph.merge_nodes(a, b)
    return True
```

---

## 7. Linear Scan Allocation

### 7.1 Motivation

Graph coloring is effective but slow for JIT compilers that need fast compilation. **Linear scan** (Poletto and Sarkar, 1999) runs in O(n log n) time.

### 7.2 Algorithm

```python
def linear_scan_allocate(intervals, k):
    """
    Linear scan register allocation.
    intervals: list of (var, start, end) sorted by start
    k: number of physical registers
    """
    active = []       # currently active intervals, sorted by end
    free_regs = list(range(k))  # available registers
    allocation = {}   # var -> register or 'SPILL'
    spill_slots = {}

    for var, start, end in sorted(intervals, key=lambda x: x[1]):
        # Expire old intervals
        active_new = []
        for a_var, a_start, a_end, a_reg in active:
            if a_end <= start:
                free_regs.append(a_reg)  # Return register
            else:
                active_new.append((a_var, a_start, a_end, a_reg))
        active = active_new

        if free_regs:
            # Allocate a register
            reg = free_regs.pop(0)
            allocation[var] = reg
            active.append((var, start, end, reg))
            active.sort(key=lambda x: x[2])  # Sort by end point
        else:
            # Spill: evict the interval with the farthest end point
            if active and active[-1][2] > end:
                # Spill the active interval that ends latest
                spill = active.pop()
                spill_slots[spill[0]] = allocate_stack_slot()
                allocation[spill[0]] = 'SPILL'
                allocation[var] = spill[3]  # Take its register
                active.append((var, start, end, spill[3]))
                active.sort(key=lambda x: x[2])
            else:
                # Spill current interval
                spill_slots[var] = allocate_stack_slot()
                allocation[var] = 'SPILL'

    return allocation, spill_slots
```

### 7.3 Second-Chance Allocation

An extension where spilled values can be reassigned registers in later parts of their live range, splitting the interval:

```python
def linear_scan_with_splitting(intervals, k):
    """
    Linear scan with live range splitting.
    When spilling, split the interval and try to allocate
    the remaining portion later.
    """
    # Sort by start point
    queue = sorted(intervals, key=lambda x: x[1])
    active = []
    allocation = {}

    while queue:
        var, start, end = queue.pop(0)
        expire_old(active, start)

        if len(active) < k:
            reg = assign_register(active)
            allocation[(var, start, end)] = reg
            active.append((var, start, end, reg))
        else:
            # Split: spill current interval at this point,
            # but re-enqueue the tail for later allocation
            split_point = find_optimal_split(start, end, active)
            if split_point < end:
                allocation[(var, start, split_point)] = 'SPILL'
                queue.append((var, split_point, end))
                queue.sort(key=lambda x: x[1])
            else:
                allocation[(var, start, end)] = 'SPILL'

    return allocation
```

---

## 8. SSA-Based Register Allocation

### 8.1 SSA Simplifies Allocation

In SSA form, live ranges never overlap for the same variable (since each variable is defined once). The interference graph of an SSA program is always **chordal**, which means it can be optimally colored in polynomial time.

### 8.2 SSA Deconstruction During Allocation

Modern allocators operate directly on SSA form:

1. Build interference graph on SSA form (chordal graph)
2. Color optimally using perfect elimination ordering
3. Insert copies for phi functions (SSA destruction)
4. Coalesce copies where possible

```python
def ssa_register_allocate(ssa_program, k):
    """
    Register allocation on SSA form.
    The interference graph is chordal -> optimal coloring in O(V+E).
    """
    graph = build_ssa_interference_graph(ssa_program)

    # Chordal graph: compute perfect elimination ordering
    peo = maximum_cardinality_search(graph)

    # Color greedily in reverse PEO order
    coloring = {}
    for node in reversed(peo):
        used = {coloring[n] for n in graph.adj.get(node, set()) if n in coloring}
        for color in range(k):
            if color not in used:
                coloring[node] = color
                break
        else:
            coloring[node] = 'SPILL'

    return coloring


def maximum_cardinality_search(graph):
    """
    Compute perfect elimination ordering for chordal graph.
    Greedy: always pick the unvisited node with the most visited neighbors.
    """
    visited = set()
    order = []
    weight = {n: 0 for n in graph.nodes}

    for _ in range(len(graph.nodes)):
        # Pick node with maximum weight
        best = max((n for n in graph.nodes if n not in visited),
                   key=lambda n: weight[n])
        order.append(best)
        visited.add(best)
        for neighbor in graph.adj.get(best, set()):
            if neighbor not in visited:
                weight[neighbor] += 1

    return order
```

---

## 9. Summary

- **Register allocation** maps virtual registers to physical registers, minimizing spills
- **Liveness analysis** determines which variables are live at each program point
- **Interference graphs** capture which variables cannot share a register
- **Graph coloring** (Chaitin-Briggs) is the standard approach for optimizing compilers
- **Linear scan** provides fast allocation for JIT compilers
- **Spilling** stores values to memory when registers are exhausted
- **Coalescing** eliminates copy instructions by merging non-interfering variables
- **SSA-based allocation** exploits chordal graph properties for optimal coloring

---

## 10. Exercises

1. **Liveness by hand**: Compute live-in and live-out sets for a 5-block CFG.

2. **Interference graph**: Build an interference graph from given live ranges and determine the chromatic number.

3. **Chaitin-Briggs**: Implement the optimistic coloring algorithm and test it on a small interference graph with k=3.

4. **Linear scan**: Implement linear scan allocation for a set of live intervals and compare the spill count with graph coloring.

5. **Coalescing**: Given a program with copy instructions, apply Briggs's conservative coalescing criterion.

---

## 11. References

1. Chaitin, G. J. (1982). "Register Allocation & Spilling via Graph Coloring." *ACM SIGPLAN Notices*, 17(6), 98-105.
2. Briggs, P., Cooper, K. D., Torczon, L. (1994). "Improvements to Graph Coloring Register Allocation." *ACM Transactions on Programming Languages and Systems*, 16(3), 428-455.
3. Poletto, M., Sarkar, V. (1999). "Linear Scan Register Allocation." *ACM Transactions on Programming Languages and Systems*, 21(5), 895-913.
4. Hack, S., Grund, D., Goos, G. (2006). "Register Allocation for Programs in SSA Form." *Compiler Construction (CC)*.
5. Wimmer, C., Franz, M. (2010). "Linear Scan Register Allocation on SSA Form." *CGO*.

---

**Previous**: [18. SSA Form](./18_SSA_Form.md) | **Next**: [20. LLVM IR Introduction](./20_LLVM_IR_Introduction.md)
