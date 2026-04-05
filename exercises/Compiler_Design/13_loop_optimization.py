"""
Exercises for Lesson 13: Loop Optimization
Topic: Compiler_Design

Solutions to practice problems covering dominator computation, natural loop
detection, loop-invariant code motion, induction variable strength reduction,
loop unrolling, and loop tiling.
"""

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, Set, List, Tuple, Optional


# === Exercise 1: Dominator Computation ===
# Problem: Compute the dominator sets and dominator tree for a CFG using
# the iterative algorithm.

def exercise_1():
    """Compute dominators and build a dominator tree."""
    print("CFG:")
    print("  Entry -> B1 -> B2 -> B3 -> B4 -> Exit")
    print("                  |           ^")
    print("                  v           |")
    print("                  B5 -> B6 ---+")
    print("           B1 -> B7 (another branch)")
    print()

    # CFG adjacency
    succs = {
        'Entry': ['B1'],
        'B1': ['B2', 'B7'],
        'B2': ['B3', 'B5'],
        'B3': ['B4'],
        'B4': ['Exit'],
        'B5': ['B6'],
        'B6': ['B3'],
        'B7': ['Exit'],
        'Exit': [],
    }
    preds = defaultdict(list)
    for node, targets in succs.items():
        for t in targets:
            preds[t].append(node)

    all_nodes = list(succs.keys())
    entry = 'Entry'

    # Iterative dominator computation
    # Dom(Entry) = {Entry}
    # Dom(n) = {n} U intersection(Dom(p) for p in preds(n)), for n != Entry
    dom = {}
    dom[entry] = {entry}
    for n in all_nodes:
        if n != entry:
            dom[n] = set(all_nodes)  # Initialize to universal set

    changed = True
    iteration = 0
    while changed:
        changed = False
        iteration += 1
        for n in all_nodes:
            if n == entry:
                continue
            if preds[n]:
                new_dom = set.intersection(*[dom[p] for p in preds[n]])
            else:
                new_dom = set()
            new_dom = {n} | new_dom
            if new_dom != dom[n]:
                dom[n] = new_dom
                changed = True

    print(f"Dominator sets (converged after {iteration} iterations):")
    for n in all_nodes:
        print(f"  Dom({n}) = {dom[n]}")
    print()

    # Build dominator tree: immediate dominator (idom)
    # idom(n) = the dominator of n that is dominated by all other dominators of n
    idom = {}
    for n in all_nodes:
        if n == entry:
            continue
        strict_doms = dom[n] - {n}
        # idom is the one in strict_doms that is dominated by no other strict dom
        for candidate in strict_doms:
            # candidate is idom if every other strict dom dominates candidate
            is_idom = True
            for other in strict_doms:
                if other == candidate:
                    continue
                if candidate not in dom[other]:
                    is_idom = False
                    break
            if is_idom:
                idom[n] = candidate
                break

    print("Immediate dominators (dominator tree edges):")
    for n in all_nodes:
        if n in idom:
            print(f"  idom({n}) = {idom[n]}")
    print()

    # Print tree structure
    children = defaultdict(list)
    for n, parent in idom.items():
        children[parent].append(n)

    def print_tree(node, indent=0):
        print("  " + "  " * indent + node)
        for child in sorted(children[node]):
            print_tree(child, indent + 1)

    print("Dominator tree:")
    print_tree(entry)
    print()

    print("Interpretation:")
    print("  Entry dominates all nodes (every path from Entry must go through it).")
    print("  B2 dominates B3, B4, B5, B6 -- all paths to those nodes go through B2.")
    print("  B3 has two predecessors (B2, B6), but B2 dominates both paths.")


# === Exercise 2: Natural Loop Detection ===
# Problem: Given the dominator information, find all natural loops in a CFG.

def exercise_2():
    """Detect natural loops using back edges and dominators."""
    print("CFG with loops:")
    print("  B0 -> B1 -> B2 -> B3 -> B1 (back edge)")
    print("               |")
    print("               v")
    print("              B4 -> B5 -> B4 (back edge)")
    print("               |")
    print("               v")
    print("              B6")
    print()

    succs = {
        'B0': ['B1'],
        'B1': ['B2'],
        'B2': ['B3', 'B4'],
        'B3': ['B1'],         # Back edge: B3 -> B1
        'B4': ['B5', 'B6'],
        'B5': ['B4'],         # Back edge: B5 -> B4
        'B6': [],
    }
    preds = defaultdict(list)
    for n, targets in succs.items():
        for t in targets:
            preds[t].append(n)

    all_nodes = list(succs.keys())
    entry = 'B0'

    # Compute dominators
    dom = {entry: {entry}}
    for n in all_nodes:
        if n != entry:
            dom[n] = set(all_nodes)

    changed = True
    while changed:
        changed = False
        for n in all_nodes:
            if n == entry:
                continue
            if preds[n]:
                new_dom = set.intersection(*[dom[p] for p in preds[n]])
            else:
                new_dom = set()
            new_dom = {n} | new_dom
            if new_dom != dom[n]:
                dom[n] = new_dom
                changed = True

    # Find back edges: edge n -> h where h dominates n
    back_edges = []
    for n in all_nodes:
        for s in succs[n]:
            if s in dom[n]:
                back_edges.append((n, s))

    print(f"Back edges found: {back_edges}")
    print()

    # For each back edge n -> h, find the natural loop
    # Natural loop = {h} U all nodes that can reach n without going through h
    def find_natural_loop(back_edge_src, header):
        loop = {header}
        if back_edge_src == header:
            return loop  # Self-loop
        stack = [back_edge_src]
        loop.add(back_edge_src)
        while stack:
            node = stack.pop()
            for p in preds[node]:
                if p not in loop:
                    loop.add(p)
                    stack.append(p)
        return loop

    for src, hdr in back_edges:
        loop_body = find_natural_loop(src, hdr)
        print(f"Natural loop for back edge {src} -> {hdr}:")
        print(f"  Header: {hdr}")
        print(f"  Body: {sorted(loop_body)}")
        # Find exits
        exits = set()
        for node in loop_body:
            for s in succs[node]:
                if s not in loop_body:
                    exits.add((node, s))
        print(f"  Exit edges: {exits}")
        # Nesting: check if this loop is inside another
        for src2, hdr2 in back_edges:
            if (src, hdr) != (src2, hdr2):
                other_loop = find_natural_loop(src2, hdr2)
                if loop_body < other_loop:
                    print(f"  Nested inside loop with header {hdr2}")
        print()

    print("Interpretation:")
    print("  Loop 1: B1-B2-B3 (header B1, back edge B3->B1)")
    print("  Loop 2: B4-B5 (header B4, back edge B5->B4)")
    print("  Loop 2 is nested inside Loop 1 (B4, B5 are part of B1's loop body)")


# === Exercise 3: Loop-Invariant Code Motion ===
# Problem: Identify loop-invariant instructions and determine which can be
# safely hoisted to the loop preheader.

def exercise_3():
    """Loop-invariant code motion (LICM) analysis and transformation."""
    print("Loop body (header = B1, preheader = B0):")
    print()
    print("  B1:")
    print("    i = phi(0, i2)       # induction variable")
    print("    t1 = a * b           # invariant? a, b defined outside")
    print("    t2 = t1 + c          # invariant? depends on t1 (invariant) and c (outside)")
    print("    arr[i] = t2 + i      # NOT invariant (depends on i)")
    print()
    print("  B2:")
    print("    t3 = x / y           # invariant? x, y defined outside")
    print("    if (t3 > 0):")
    print("      t4 = t3 * 2        # invariant, but in conditional")
    print("      store(t4)          # side effect!")
    print("    i2 = i + 1")
    print("    if i2 < n goto B1")
    print()

    # Analysis: which instructions are loop-invariant?
    # An instruction is loop-invariant if all its operands are either:
    #   (1) defined outside the loop, or
    #   (2) defined by another loop-invariant instruction

    outside_defs = {'a', 'b', 'c', 'x', 'y', 'n'}
    loop_instrs = [
        {'dest': 'i',   'uses': set(),       'invariant': False, 'reason': 'phi node'},
        {'dest': 't1',  'uses': {'a', 'b'},  'invariant': None,  'reason': ''},
        {'dest': 't2',  'uses': {'t1', 'c'}, 'invariant': None,  'reason': ''},
        {'dest': None,  'uses': {'t2', 'i'}, 'invariant': None,  'reason': 'arr[i]=t2+i'},
        {'dest': 't3',  'uses': {'x', 'y'},  'invariant': None,  'reason': ''},
        {'dest': 't4',  'uses': {'t3'},       'invariant': None,  'reason': 'conditional'},
        {'dest': None,  'uses': {'t4'},       'invariant': None,  'reason': 'store side effect'},
        {'dest': 'i2',  'uses': {'i'},        'invariant': False, 'reason': 'depends on i'},
    ]

    # Mark invariant instructions iteratively
    invariant_defs = set()
    changed = True
    while changed:
        changed = False
        for instr in loop_instrs:
            if instr['invariant'] is not None:
                continue
            all_invariant = all(
                u in outside_defs or u in invariant_defs
                for u in instr['uses']
            )
            if all_invariant:
                instr['invariant'] = True
                if instr['dest']:
                    invariant_defs.add(instr['dest'])
                changed = True

    # Mark remaining as not invariant
    for instr in loop_instrs:
        if instr['invariant'] is None:
            instr['invariant'] = False

    print("Loop-invariant analysis:")
    for instr in loop_instrs:
        dest = instr['dest'] if instr['dest'] else '(no dest)'
        status = 'INVARIANT' if instr['invariant'] else 'NOT invariant'
        reason = f" ({instr['reason']})" if instr['reason'] else ''
        print(f"  {dest}: {status}{reason}")
    print()

    # Hoisting conditions
    print("Hoisting analysis:")
    print("  t1 = a * b:")
    print("    [x] Loop-invariant")
    print("    [x] Dominates all loop exits (B1 dominates everything)")
    print("    [x] No side effects")
    print("    [x] Not a phi node")
    print("    -> CAN HOIST to preheader")
    print()
    print("  t2 = t1 + c:")
    print("    [x] Loop-invariant (after t1 is hoisted)")
    print("    [x] Dominates all loop exits")
    print("    [x] No side effects")
    print("    -> CAN HOIST (after t1)")
    print()
    print("  t3 = x / y:")
    print("    [x] Loop-invariant")
    print("    [x] Dominates all exits")
    print("    [ ] Potential exception (division by zero!)")
    print("    -> HOIST ONLY if y != 0 is guaranteed, or loop executes at least once")
    print()
    print("  t4 = t3 * 2:")
    print("    [x] Loop-invariant")
    print("    [ ] Does NOT dominate all exits (inside conditional)")
    print("    -> CANNOT HOIST (would execute on paths where it originally didn't)")
    print()

    print("After LICM (assuming y != 0 guaranteed):")
    print("  Preheader (B0):")
    print("    t1 = a * b          # hoisted")
    print("    t2 = t1 + c         # hoisted")
    print("    t3 = x / y          # hoisted (guarded)")
    print()
    print("  B1 (loop body):")
    print("    i = phi(0, i2)")
    print("    arr[i] = t2 + i")
    print("    if (t3 > 0):")
    print("      t4 = t3 * 2")
    print("      store(t4)")
    print("    i2 = i + 1")
    print("    if i2 < n goto B1")


# === Exercise 4: Induction Variable Strength Reduction ===
# Problem: Identify induction variables and apply strength reduction.

def exercise_4():
    """Induction variable analysis and strength reduction."""
    print("Original loop:")
    print("  for (i = 0; i < n; i++) {")
    print("      j = 4 * i + 1;")
    print("      k = i * i;")
    print("      arr[i] = j + k;")
    print("  }")
    print()

    # Simulate original computation
    n = 8
    print(f"Original values (n={n}):")
    for i in range(n):
        j = 4 * i + 1
        k = i * i
        print(f"  i={i}: j=4*{i}+1={j}, k={i}*{i}={k}, arr[{i}]={j+k}")
    print()

    # Identify induction variables
    print("Induction Variable Analysis:")
    print("  Basic IV: i (incremented by 1 each iteration)")
    print("  Derived IV: j = 4*i + 1 (linear function of i)")
    print("    Family: {i}, multiplier=4, offset=1")
    print("  NOT an IV: k = i*i (quadratic, not linear)")
    print()

    # Strength reduction: replace multiplication with addition
    print("Strength Reduction (j = 4*i + 1 -> incremental update):")
    print()
    print("  // Before:")
    print("  for (i = 0; i < n; i++)")
    print("      j = 4 * i + 1;   // multiplication each iteration")
    print()
    print("  // After strength reduction:")
    print("  j = 1;               // initial value: 4*0 + 1")
    print("  for (i = 0; i < n; i++) {")
    print("      // j already has the correct value")
    print("      j = j + 4;       // increment by stride (4)")
    print("  }")
    print()

    # Verify strength-reduced version
    print("Verification of strength-reduced j:")
    j = 1
    for i in range(n):
        j_expected = 4 * i + 1
        match = "OK" if j == j_expected else "MISMATCH"
        print(f"  i={i}: j={j} (expected {j_expected}) [{match}]")
        j += 4
    print()

    # Strength reduction for k = i*i using difference method
    print("Strength Reduction for k = i*i (quadratic):")
    print("  Observation: (i+1)^2 - i^2 = 2*i + 1")
    print("  So k_next = k + 2*i + 1")
    print("  And (2*(i+1)+1) - (2*i+1) = 2")
    print("  So the increment itself increases by 2 each step.")
    print()
    print("  // After strength reduction:")
    print("  k = 0;           // 0^2")
    print("  delta = 1;       // 2*0 + 1")
    print("  for (i = 0; i < n; i++) {")
    print("      // k = i*i, delta = 2*i+1")
    print("      k = k + delta;")
    print("      delta = delta + 2;")
    print("  }")
    print()

    print("Verification of strength-reduced k:")
    k = 0
    delta = 1
    for i in range(n):
        k_expected = i * i
        match = "OK" if k == k_expected else "MISMATCH"
        print(f"  i={i}: k={k} (expected {k_expected}) [{match}]")
        k += delta
        delta += 2


# === Exercise 5: Loop Unrolling ===
# Problem: Apply loop unrolling and analyze trade-offs.

def exercise_5():
    """Loop unrolling transformation and analysis."""
    print("Original loop:")
    print("  sum = 0")
    print("  for i in range(100):")
    print("      sum += arr[i]")
    print()

    # Simulate with a simple array
    arr = list(range(100))  # arr[i] = i for simplicity

    # Original
    sum_original = 0
    loop_overhead_original = 0
    for i in range(100):
        sum_original += arr[i]
        loop_overhead_original += 1  # count iterations (branch, increment, compare)

    print(f"Original: sum={sum_original}, iterations={loop_overhead_original}")
    print(f"  Loop overhead: 100 branches, 100 increments, 100 comparisons")
    print()

    # Unroll by factor of 4
    print("Unrolled by factor of 4:")
    print("  sum = 0")
    print("  for i in range(0, 100, 4):")
    print("      sum += arr[i]")
    print("      sum += arr[i+1]")
    print("      sum += arr[i+2]")
    print("      sum += arr[i+3]")
    print()

    sum_unrolled = 0
    loop_overhead_unrolled = 0
    for i in range(0, 100, 4):
        sum_unrolled += arr[i]
        sum_unrolled += arr[i + 1]
        sum_unrolled += arr[i + 2]
        sum_unrolled += arr[i + 3]
        loop_overhead_unrolled += 1

    print(f"Unrolled: sum={sum_unrolled}, iterations={loop_overhead_unrolled}")
    print(f"  Loop overhead: 25 branches, 25 increments, 25 comparisons")
    print(f"  Correct: {sum_unrolled == sum_original}")
    print()

    # Handle non-divisible case
    print("Non-divisible trip count (n=103):")
    arr2 = list(range(103))
    n = 103
    unroll_factor = 4
    main_limit = (n // unroll_factor) * unroll_factor  # 100

    print(f"  Main loop: i = 0 to {main_limit - 1} (step {unroll_factor})")
    print(f"  Epilogue:  i = {main_limit} to {n - 1}")
    print()

    sum_main = 0
    for i in range(0, main_limit, unroll_factor):
        sum_main += arr2[i] + arr2[i + 1] + arr2[i + 2] + arr2[i + 3]
    # Epilogue
    for i in range(main_limit, n):
        sum_main += arr2[i]

    sum_expected = sum(arr2)
    print(f"  Result: sum={sum_main}, expected={sum_expected}, "
          f"correct={sum_main == sum_expected}")
    print()

    # Trade-off analysis
    print("Unrolling trade-offs:")
    print("  Pros:")
    print("    - Reduced loop overhead (fewer branches)")
    print("    - More ILP (instruction-level parallelism)")
    print("    - Better scheduling opportunities")
    print("    - Enables other optimizations (CSE across iterations)")
    print("  Cons:")
    print("    - Increased code size (I-cache pressure)")
    print("    - Register pressure increases")
    print("    - Epilogue code needed for non-divisible counts")
    print("    - Diminishing returns beyond factor ~4-8")


# === Exercise 6: Loop Tiling (Blocking) ===
# Problem: Apply loop tiling to a matrix multiplication to improve cache
# performance.

def exercise_6():
    """Loop tiling for matrix multiplication."""
    print("Matrix multiplication: C[i][j] += A[i][k] * B[k][j]")
    print()

    N = 64  # Small matrix for demonstration
    TILE = 16

    # Initialize matrices
    import random
    random.seed(42)
    A = [[random.randint(0, 9) for _ in range(N)] for _ in range(N)]
    B = [[random.randint(0, 9) for _ in range(N)] for _ in range(N)]

    # Naive (ijk order)
    C_naive = [[0] * N for _ in range(N)]
    naive_accesses = 0
    for i in range(N):
        for j in range(N):
            for k in range(N):
                C_naive[i][j] += A[i][k] * B[k][j]
                naive_accesses += 3  # A, B, C access

    # Tiled version
    C_tiled = [[0] * N for _ in range(N)]
    tiled_accesses = 0
    for ii in range(0, N, TILE):
        for jj in range(0, N, TILE):
            for kk in range(0, N, TILE):
                for i in range(ii, min(ii + TILE, N)):
                    for j in range(jj, min(jj + TILE, N)):
                        for k in range(kk, min(kk + TILE, N)):
                            C_tiled[i][j] += A[i][k] * B[k][j]
                            tiled_accesses += 3

    # Verify correctness
    correct = all(
        C_naive[i][j] == C_tiled[i][j]
        for i in range(N)
        for j in range(N)
    )

    print(f"Matrix size: {N}x{N}, Tile size: {TILE}x{TILE}")
    print(f"Naive accesses:  {naive_accesses}")
    print(f"Tiled accesses:  {tiled_accesses}")
    print(f"Results match: {correct}")
    print()

    # Cache analysis
    print("Cache behavior analysis:")
    print(f"  Naive (ijk):")
    print(f"    A[i][k]: row access (good spatial locality)")
    print(f"    B[k][j]: column access (poor spatial locality)")
    print(f"    Working set per inner loop: entire column of B ({N} elements)")
    print()
    print(f"  Tiled ({TILE}x{TILE} blocks):")
    print(f"    A block: {TILE}x{TILE} = {TILE*TILE} elements")
    print(f"    B block: {TILE}x{TILE} = {TILE*TILE} elements")
    print(f"    Working set per tile: {2*TILE*TILE} elements "
          f"({2*TILE*TILE*8} bytes for double)")
    print(f"    Fits in L1 cache (typically 32KB): "
          f"{'Yes' if 2*TILE*TILE*8 < 32768 else 'No'}")
    print()

    # Reuse analysis
    total_elements = N * N
    print("Data reuse improvement:")
    print(f"  Naive: B is accessed {N} times per element across all i iterations")
    print(f"         but cache lines are evicted between uses")
    print(f"  Tiled: B tile ({TILE}x{TILE}) stays in cache for {TILE} iterations of i")
    print(f"         Reuse factor improved by ~{TILE}x for B accesses")


if __name__ == "__main__":
    print("=" * 60)
    print("=== Exercise 1: Dominator Computation ===")
    print("=" * 60)
    exercise_1()

    print("\n" + "=" * 60)
    print("=== Exercise 2: Natural Loop Detection ===")
    print("=" * 60)
    exercise_2()

    print("\n" + "=" * 60)
    print("=== Exercise 3: Loop-Invariant Code Motion ===")
    print("=" * 60)
    exercise_3()

    print("\n" + "=" * 60)
    print("=== Exercise 4: Induction Variable Strength Reduction ===")
    print("=" * 60)
    exercise_4()

    print("\n" + "=" * 60)
    print("=== Exercise 5: Loop Unrolling ===")
    print("=" * 60)
    exercise_5()

    print("\n" + "=" * 60)
    print("=== Exercise 6: Loop Tiling ===")
    print("=" * 60)
    exercise_6()

    print("\nAll exercises completed!")
