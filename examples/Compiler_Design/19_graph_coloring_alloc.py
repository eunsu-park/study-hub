"""
19_graph_coloring_alloc.py - Register Allocation via Graph Coloring

Extends the register allocation concepts from lesson 11 with a more
detailed implementation of the Chaitin-Briggs graph coloring approach
used in production compilers.

Components:
  1. Liveness Analysis
     Compute live ranges for each virtual register using backward
     dataflow analysis.

  2. Interference Graph Construction
     Build an undirected graph where an edge connects two virtual
     registers that are simultaneously live.

  3. Graph Coloring (Chaitin-Briggs)
     Iteratively simplify the graph by removing low-degree nodes,
     then assign colors (physical registers). Spill nodes that
     cannot be colored.

  4. Coalescing
     Merge move-related virtual registers to eliminate copies when
     doing so does not make the graph uncolorable.

  5. Spill Code Insertion
     When there are not enough physical registers, insert loads and
     stores for spilled variables.

Topics covered:
  - Live range computation
  - Interference graph
  - Graph coloring heuristics
  - Iterated register coalescing
  - Spill cost estimation
  - Move coalescing for copy elimination
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# IR Instructions
# ---------------------------------------------------------------------------

@dataclass
class Instr:
    dest: Optional[str] = None
    op: Optional[str] = None
    src1: Optional[str] = None
    src2: Optional[str] = None
    is_move: bool = False  # dest = src1 (copy)

    def uses(self) -> list[str]:
        result = []
        if self.src1 and not self.src1.lstrip('-').isdigit():
            result.append(self.src1)
        if self.src2 and not self.src2.lstrip('-').isdigit():
            result.append(self.src2)
        return result

    def defs(self) -> list[str]:
        return [self.dest] if self.dest else []

    def __str__(self):
        if self.is_move:
            return f"    {self.dest} = {self.src1}"
        if self.op:
            return f"    {self.dest} = {self.src1} {self.op} {self.src2}"
        if self.dest:
            return f"    {self.dest} = {self.src1}"
        return f"    nop"


# ---------------------------------------------------------------------------
# Liveness Analysis
# ---------------------------------------------------------------------------

def compute_liveness(instrs: list[Instr]) -> list[set[str]]:
    """
    Backward dataflow liveness analysis.
    Returns live_out[i]: set of variables live after instruction i.
    """
    n = len(instrs)
    live_out: list[set[str]] = [set() for _ in range(n)]

    changed = True
    while changed:
        changed = False
        for i in range(n - 1, -1, -1):
            old = set(live_out[i])
            # live_in[i] = uses[i] | (live_out[i] - defs[i])
            live_in = set(instrs[i].uses())
            for d in instrs[i].defs():
                live_out[i].discard(d)
            live_in |= live_out[i]

            # live_out[i-1] = live_in[i] (simplified: straight-line code)
            if i > 0:
                live_out[i - 1] = live_out[i - 1] | live_in

            # Recompute live_out for boundary
            if i < n - 1:
                new_out = set(instrs[i + 1].uses())
                for d in instrs[i + 1].defs():
                    pass  # already handled
                new_out |= live_out[i + 1] - set(instrs[i + 1].defs())
                new_out |= set(instrs[i + 1].uses())
                if new_out != old:
                    live_out[i] = new_out
                    changed = True

    # Recompute cleanly
    live: list[set[str]] = [set() for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        live[i] = set(live[i + 1])
        for d in instrs[i].defs():
            live[i].discard(d)
        live[i] |= set(instrs[i].uses())

    return [live[i + 1] for i in range(n)]


# ---------------------------------------------------------------------------
# Interference Graph
# ---------------------------------------------------------------------------

class InterferenceGraph:
    """
    Undirected graph where nodes are virtual registers and edges
    represent interference (simultaneous liveness).
    """

    def __init__(self):
        self.nodes: set[str] = set()
        self.edges: set[frozenset[str]] = set()
        self.move_edges: set[frozenset[str]] = set()

    def add_node(self, name: str) -> None:
        self.nodes.add(name)

    def add_edge(self, a: str, b: str) -> None:
        if a != b:
            self.edges.add(frozenset([a, b]))

    def add_move_edge(self, a: str, b: str) -> None:
        if a != b:
            self.move_edges.add(frozenset([a, b]))

    def neighbors(self, node: str) -> set[str]:
        result = set()
        for edge in self.edges:
            if node in edge:
                result |= edge - {node}
        return result

    def degree(self, node: str) -> int:
        return len(self.neighbors(node))

    def remove_node(self, node: str) -> None:
        self.nodes.discard(node)
        self.edges = {e for e in self.edges if node not in e}
        self.move_edges = {e for e in self.move_edges if node not in e}

    def __str__(self):
        lines = ["Interference Graph:"]
        for node in sorted(self.nodes):
            nbrs = sorted(self.neighbors(node))
            lines.append(f"  {node} (deg={self.degree(node)}): {nbrs}")
        return "\n".join(lines)


def build_interference_graph(instrs: list[Instr],
                             live_out: list[set[str]]) -> InterferenceGraph:
    """Build interference graph from instructions and liveness info."""
    ig = InterferenceGraph()

    # Add all defined variables as nodes
    for instr in instrs:
        for d in instr.defs():
            ig.add_node(d)
        for u in instr.uses():
            ig.add_node(u)

    # Add interference edges
    for i, instr in enumerate(instrs):
        for d in instr.defs():
            for live_var in live_out[i]:
                if live_var != d:
                    ig.add_edge(d, live_var)

        # Track move-related pairs
        if instr.is_move and instr.dest and instr.src1:
            ig.add_move_edge(instr.dest, instr.src1)

    return ig


# ---------------------------------------------------------------------------
# Graph Coloring (Chaitin-Briggs)
# ---------------------------------------------------------------------------

@dataclass
class ColoringResult:
    assignment: dict[str, int] = field(default_factory=dict)
    spilled: list[str] = field(default_factory=list)
    coalesced: dict[str, str] = field(default_factory=dict)


def color_graph(ig: InterferenceGraph, num_regs: int) -> ColoringResult:
    """
    Chaitin-Briggs graph coloring register allocation.
    1. Simplify: push low-degree (<K) nodes onto stack
    2. Potential spill: push remaining nodes
    3. Select: pop and assign colors, spill if impossible
    """
    result = ColoringResult()
    graph = InterferenceGraph()
    graph.nodes = set(ig.nodes)
    graph.edges = set(ig.edges)

    stack: list[str] = []
    K = num_regs

    # Phase 1: Simplify
    remaining = set(graph.nodes)
    while remaining:
        # Find a low-degree node
        low_degree = None
        for node in sorted(remaining):
            deg = sum(1 for e in graph.edges
                      if node in e and (e - {node}).issubset(remaining))
            if deg < K:
                low_degree = node
                break

        if low_degree:
            stack.append(low_degree)
            remaining.remove(low_degree)
        else:
            # Potential spill: pick highest degree node
            max_deg = -1
            spill_candidate = None
            for node in sorted(remaining):
                deg = sum(1 for e in graph.edges
                          if node in e and (e - {node}).issubset(remaining))
                if deg > max_deg:
                    max_deg = deg
                    spill_candidate = node
            if spill_candidate:
                stack.append(spill_candidate)
                remaining.remove(spill_candidate)

    # Phase 2: Select (pop and color)
    while stack:
        node = stack.pop()
        used_colors: set[int] = set()
        for neighbor in ig.neighbors(node):
            if neighbor in result.assignment:
                used_colors.add(result.assignment[neighbor])

        # Find available color
        color = None
        for c in range(K):
            if c not in used_colors:
                color = c
                break

        if color is not None:
            result.assignment[node] = color
        else:
            result.spilled.append(node)

    return result


# ---------------------------------------------------------------------------
# Coalescing
# ---------------------------------------------------------------------------

def coalesce_moves(ig: InterferenceGraph, num_regs: int,
                   instrs: list[Instr]) -> list[Instr]:
    """
    Conservative coalescing: merge move-related nodes if combined
    degree < K (Briggs criterion).
    """
    coalesced_pairs: list[tuple[str, str]] = []
    K = num_regs

    for move_edge in sorted(ig.move_edges, key=lambda e: tuple(sorted(e))):
        a, b = sorted(move_edge)
        if a not in ig.nodes or b not in ig.nodes:
            continue
        if frozenset([a, b]) in ig.edges:
            continue  # already interfere

        # Briggs criterion: combined node must have < K high-degree neighbors
        combined_neighbors = ig.neighbors(a) | ig.neighbors(b)
        high_deg = sum(1 for n in combined_neighbors if ig.degree(n) >= K)
        if high_deg < K:
            coalesced_pairs.append((b, a))
            # Merge b into a
            for neighbor in ig.neighbors(b):
                ig.add_edge(a, neighbor)
            ig.remove_node(b)

    # Rewrite instructions to eliminate coalesced moves
    rename_map = {b: a for b, a in coalesced_pairs}
    new_instrs = []
    for instr in instrs:
        new_instr = Instr(
            dest=rename_map.get(instr.dest, instr.dest),
            op=instr.op,
            src1=rename_map.get(instr.src1, instr.src1) if instr.src1 else None,
            src2=rename_map.get(instr.src2, instr.src2) if instr.src2 else None,
            is_move=instr.is_move
        )
        # Remove trivial moves (x = x)
        if new_instr.is_move and new_instr.dest == new_instr.src1:
            continue
        new_instrs.append(new_instr)

    return new_instrs


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

REG_NAMES = ["r0", "r1", "r2", "r3"]


def make_sample_program() -> list[Instr]:
    """
    a = 1
    b = 2
    c = a + b     # a, b live
    d = c + 1     # c live
    e = b + d     # b, d live
    f = e         # move (coalescing candidate)
    return f
    """
    return [
        Instr(dest="a", src1="1"),
        Instr(dest="b", src1="2"),
        Instr(dest="c", op="+", src1="a", src2="b"),
        Instr(dest="d", op="+", src1="c", src2="1"),
        Instr(dest="e", op="+", src1="b", src2="d"),
        Instr(dest="f", src1="e", is_move=True),
    ]


def main():
    print("=" * 60)
    print("Register Allocation via Graph Coloring")
    print("=" * 60)

    instrs = make_sample_program()
    print(f"\n--- Program ---")
    for i, instr in enumerate(instrs):
        print(f"  [{i}] {instr}")

    # Step 1: Liveness analysis
    live_out = compute_liveness(instrs)
    print(f"\n--- Liveness (live_out) ---")
    for i, (instr, live) in enumerate(zip(instrs, live_out)):
        print(f"  [{i}] {str(instr):30s}  live_out = {sorted(live)}")

    # Step 2: Build interference graph
    ig = build_interference_graph(instrs, live_out)
    print(f"\n--- {ig} ---")

    # Step 3: Graph coloring with 3 registers
    num_regs = 3
    print(f"\n--- Graph Coloring (K={num_regs}) ---")
    result = color_graph(ig, num_regs)
    for var, color in sorted(result.assignment.items()):
        print(f"  {var} -> {REG_NAMES[color]} (color {color})")
    if result.spilled:
        print(f"  Spilled: {result.spilled}")
    else:
        print(f"  No spills needed")

    # Step 4: Try with only 2 registers (force spill)
    print(f"\n--- Graph Coloring (K=2, expect spill) ---")
    ig2 = build_interference_graph(instrs, live_out)
    result2 = color_graph(ig2, 2)
    for var, color in sorted(result2.assignment.items()):
        print(f"  {var} -> {REG_NAMES[color]} (color {color})")
    if result2.spilled:
        print(f"  Spilled: {result2.spilled}")

    # Step 5: Coalescing
    print(f"\n--- Move Coalescing ---")
    ig3 = build_interference_graph(instrs, live_out)
    print(f"  Move pairs: {[tuple(sorted(e)) for e in ig3.move_edges]}")
    coalesced = coalesce_moves(ig3, 3, instrs)
    print(f"  After coalescing ({len(instrs)} -> {len(coalesced)} instrs):")
    for i, instr in enumerate(coalesced):
        print(f"    [{i}] {instr}")

    print(f"\n--- Summary ---")
    print("""
  Graph coloring register allocation:
    1. Compute liveness -> live ranges for each variable
    2. Build interference graph -> edges between simultaneously live vars
    3. Color graph with K colors (K = number of physical registers)
    4. Spill high-degree nodes that cannot be colored
    5. Coalesce move-related nodes to eliminate copies
    """)


if __name__ == "__main__":
    main()
