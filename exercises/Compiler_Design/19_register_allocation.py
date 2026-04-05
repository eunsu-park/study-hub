"""
Exercises for Lesson 19: Register Allocation
Topic: Compiler_Design

Solutions to practice problems from the lesson.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Set, List, Tuple, Optional


# === Exercise 1: Liveness Analysis ===

def exercise_1():
    """Compute live-in and live-out sets for a simple CFG."""
    print("Exercise 1: Liveness Analysis")
    print()

    # Block definitions: (uses_before_def, defs, successors)
    blocks = {
        'B1': {'use': set(), 'def': {'a', 'b'}, 'succ': ['B2', 'B3']},
        'B2': {'use': {'a'}, 'def': {'c'}, 'succ': ['B4']},
        'B3': {'use': {'b'}, 'def': {'c'}, 'succ': ['B4']},
        'B4': {'use': {'c'}, 'def': {'d'}, 'succ': ['B5']},
        'B5': {'use': {'d'}, 'def': set(), 'succ': []},
    }

    live_in = {b: set() for b in blocks}
    live_out = {b: set() for b in blocks}

    order = ['B5', 'B4', 'B3', 'B2', 'B1']
    changed = True
    while changed:
        changed = False
        for b in order:
            info = blocks[b]
            new_out = set()
            for s in info['succ']:
                new_out |= live_in[s]
            new_in = info['use'] | (new_out - info['def'])
            if new_in != live_in[b] or new_out != live_out[b]:
                live_in[b] = new_in
                live_out[b] = new_out
                changed = True

    print("Results:")
    for b in ['B1', 'B2', 'B3', 'B4', 'B5']:
        print(f"  {b}: LiveIn={live_in[b]}, LiveOut={live_out[b]}")
    print()


# === Exercise 2: Interference Graph ===

def exercise_2():
    """Build an interference graph from live ranges."""
    print("Exercise 2: Interference Graph")
    print()

    # Live ranges: (variable, start, end)
    live_ranges = [
        ('a', 1, 4),
        ('b', 2, 5),
        ('c', 3, 6),
        ('d', 5, 8),
        ('e', 7, 9),
    ]

    print("Live ranges:")
    for var, start, end in live_ranges:
        print(f"  {var}: [{start}, {end})")

    # Build interference: two ranges interfere if they overlap
    adj = defaultdict(set)
    for i, (v1, s1, e1) in enumerate(live_ranges):
        for j, (v2, s2, e2) in enumerate(live_ranges):
            if i < j and s1 < e2 and s2 < e1:
                adj[v1].add(v2)
                adj[v2].add(v1)

    print("\nInterference edges:")
    printed = set()
    for v, neighbors in sorted(adj.items()):
        for n in sorted(neighbors):
            edge = tuple(sorted([v, n]))
            if edge not in printed:
                print(f"  {edge[0]} -- {edge[1]}")
                printed.add(edge)

    max_clique = max(
        sum(1 for v, s, e in live_ranges if s <= t < e)
        for t in range(1, 10)
    )
    print(f"\nMax simultaneously live: {max_clique}")
    print(f"Chromatic number >= {max_clique}")
    print()


# === Exercise 3: Chaitin-Briggs Graph Coloring ===

def exercise_3():
    """Implement optimistic graph coloring with k=3."""
    print("Exercise 3: Chaitin-Briggs (k=3)")
    print()

    adj = {
        'a': {'b', 'c'},
        'b': {'a', 'c', 'd'},
        'c': {'a', 'b', 'd'},
        'd': {'b', 'c', 'e'},
        'e': {'d'},
    }
    k = 3

    print(f"Graph ({k} registers):")
    for v in sorted(adj):
        print(f"  {v}: degree={len(adj[v])}, neighbors={sorted(adj[v])}")

    # Briggs optimistic coloring
    work = {v: set(n) for v, n in adj.items()}
    stack = []

    while work:
        # Try to find degree < k
        found = None
        for v in list(work):
            if len(work[v]) < k:
                found = v
                break

        if found:
            stack.append(found)
        else:
            # Optimistic: push highest-degree node
            found = max(work, key=lambda v: len(work[v]))
            stack.append(found)

        # Remove from work graph
        for n in list(work[found]):
            work[n].discard(found)
        del work[found]

    print(f"\nSimplify stack: {stack}")

    # Color
    coloring = {}
    colors = ['R0', 'R1', 'R2']
    for v in reversed(stack):
        used = {coloring[n] for n in adj[v] if n in coloring}
        available = [c for c in colors if c not in used]
        if available:
            coloring[v] = available[0]
        else:
            coloring[v] = 'SPILL'

    print("\nColoring:")
    for v in sorted(coloring):
        print(f"  {v} -> {coloring[v]}")
    print()


# === Exercise 4: Linear Scan ===

def exercise_4():
    """Implement linear scan register allocation."""
    print("Exercise 4: Linear Scan (k=3)")
    print()

    intervals = [
        ('a', 1, 5),
        ('b', 2, 7),
        ('c', 3, 8),
        ('d', 6, 10),
        ('e', 9, 12),
    ]
    k = 3

    print("Live intervals:")
    for var, s, e in intervals:
        print(f"  {var}: [{s}, {e})")

    active = []
    free_regs = list(range(k))
    allocation = {}

    for var, start, end in sorted(intervals, key=lambda x: x[1]):
        # Expire
        new_active = []
        for a_var, a_end, a_reg in active:
            if a_end <= start:
                free_regs.append(a_reg)
            else:
                new_active.append((a_var, a_end, a_reg))
        active = new_active

        if free_regs:
            reg = free_regs.pop(0)
            allocation[var] = f'R{reg}'
            active.append((var, end, reg))
            active.sort(key=lambda x: x[1])
        else:
            if active and active[-1][1] > end:
                spill_var, spill_end, spill_reg = active.pop()
                allocation[spill_var] = 'SPILL'
                allocation[var] = f'R{spill_reg}'
                active.append((var, end, spill_reg))
                active.sort(key=lambda x: x[1])
            else:
                allocation[var] = 'SPILL'

    print("\nAllocation:")
    for var, reg in allocation.items():
        print(f"  {var} -> {reg}")
    spills = sum(1 for v in allocation.values() if v == 'SPILL')
    print(f"\nTotal spills: {spills}")
    print()


# === Exercise 5: Coalescing ===

def exercise_5():
    """Apply Briggs's conservative coalescing criterion."""
    print("Exercise 5: Conservative Coalescing")
    print()

    adj = {
        'a': {'c', 'd'},
        'b': {'d'},
        'c': {'a', 'd'},
        'd': {'a', 'b', 'c'},
    }
    k = 3
    copies = [('a', 'b')]

    print(f"Interference graph (k={k}):")
    for v in sorted(adj):
        print(f"  {v}: degree={len(adj[v])}")

    print(f"\nCopy instructions: {copies}")

    for src, dst in copies:
        if dst in adj.get(src, set()):
            print(f"\n  {src} and {dst} interfere -> cannot coalesce")
            continue

        merged_neighbors = (adj.get(src, set()) | adj.get(dst, set())) - {src, dst}
        high_degree = sum(1 for n in merged_neighbors if len(adj.get(n, set())) >= k)

        print(f"\n  Merge {src}+{dst}: neighbors={sorted(merged_neighbors)}")
        print(f"  High-degree (>= {k}) neighbors: {high_degree}")

        if high_degree < k:
            print(f"  -> SAFE to coalesce (Briggs criterion satisfied)")
        else:
            print(f"  -> UNSAFE to coalesce")
    print()


# === Main ===

def main():
    exercises = [exercise_1, exercise_2, exercise_3, exercise_4, exercise_5]
    for i, ex in enumerate(exercises, 1):
        print(f"{'=' * 60}")
        print(f"Exercise {i}")
        print(f"{'=' * 60}")
        ex()


if __name__ == "__main__":
    main()
