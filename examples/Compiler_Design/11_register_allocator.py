"""
11_register_allocator.py - Register Allocation via Graph Coloring

Demonstrates register allocation — mapping an unbounded set of virtual
registers (temporaries) to a finite set of physical machine registers.

Algorithm overview (Chaitin-style graph coloring):
  1. Liveness Analysis
     Compute live-in and live-out sets for each instruction by iterating
     backward through the program until a fixed point is reached.

  2. Interference Graph Construction
     Two temporaries interfere (share an edge) if they are simultaneously
     live at some point.  The graph is undirected.

  3. Graph Coloring (simplification + select)
     Repeatedly remove a node with degree < K (number of physical
     registers) and push it onto a stack.  If no such node exists, choose
     a node to *spill* (we pick the one with highest degree).  Then pop
     nodes from the stack and assign colors, avoiding neighbors' colors.

  4. Spill Handling (simplified)
     If a node cannot be colored during the select phase, it is marked
     for spilling — meaning it would be loaded/stored from memory.  A
     production allocator would rewrite the IR and re-run allocation;
     here we simply report the spill.

Concepts covered:
  - Three-address code / SSA-like IR
  - Live variable analysis (backward dataflow)
  - Interference graph
  - Greedy graph coloring heuristic
  - Register pressure and spilling
"""

from __future__ import annotations
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# IR representation — simple three-address instructions
# ---------------------------------------------------------------------------

@dataclass
class IRInst:
    """A three-address instruction: dest = left op right  (or simpler)."""
    op: str             # "add", "sub", "mul", "load", "mov", "ret", ...
    dest: str | None    # destination register (None for ret/nop)
    srcs: list[str] = field(default_factory=list)  # source operands

    def defs(self) -> set[str]:
        """Registers defined by this instruction."""
        if self.dest and not self.dest.startswith("#"):
            return {self.dest}
        return set()

    def uses(self) -> set[str]:
        """Registers used by this instruction."""
        return {s for s in self.srcs if not s.startswith("#")}

    def __repr__(self) -> str:
        srcs_str = ", ".join(self.srcs)
        if self.dest:
            return f"  {self.dest} = {self.op} {srcs_str}"
        return f"  {self.op} {srcs_str}"


# ---------------------------------------------------------------------------
# Liveness analysis (backward dataflow, single basic block)
# ---------------------------------------------------------------------------

def liveness_analysis(
    program: list[IRInst],
) -> tuple[list[set[str]], list[set[str]]]:
    """Compute live-in and live-out sets for each instruction index.

    Uses the standard backward equations:
        live_out[i] = live_in[i+1]  (for straight-line code)
        live_in[i]  = uses[i] ∪ (live_out[i] - defs[i])

    Returns (live_in, live_out) — lists indexed by instruction position.
    """
    n = len(program)
    live_in: list[set[str]] = [set() for _ in range(n)]
    live_out: list[set[str]] = [set() for _ in range(n)]

    # Iterate until fixed point
    changed = True
    while changed:
        changed = False
        for i in range(n - 1, -1, -1):
            # live_out[i] = live_in[i+1]  (straight-line assumption)
            new_out = live_in[i + 1] if i + 1 < n else set()

            # live_in[i] = uses[i] ∪ (live_out[i] - defs[i])
            new_in = program[i].uses() | (new_out - program[i].defs())

            if new_in != live_in[i] or new_out != live_out[i]:
                changed = True
                live_in[i] = new_in
                live_out[i] = new_out

    return live_in, live_out


# ---------------------------------------------------------------------------
# Interference graph
# ---------------------------------------------------------------------------

@dataclass
class InterferenceGraph:
    """Undirected graph where nodes are virtual registers and edges connect
    registers that are simultaneously live."""
    nodes: set[str] = field(default_factory=set)
    edges: set[frozenset[str]] = field(default_factory=set)
    adj: dict[str, set[str]] = field(default_factory=dict)

    def add_node(self, n: str) -> None:
        self.nodes.add(n)
        self.adj.setdefault(n, set())

    def add_edge(self, u: str, v: str) -> None:
        if u == v:
            return
        self.add_node(u)
        self.add_node(v)
        e = frozenset({u, v})
        if e not in self.edges:
            self.edges.add(e)
            self.adj[u].add(v)
            self.adj[v].add(u)

    def degree(self, n: str) -> int:
        return len(self.adj.get(n, set()))

    def remove_node(self, n: str) -> None:
        for neighbor in list(self.adj.get(n, set())):
            self.adj[neighbor].discard(n)
            self.edges.discard(frozenset({n, neighbor}))
        self.adj.pop(n, None)
        self.nodes.discard(n)

    def copy(self) -> InterferenceGraph:
        g = InterferenceGraph()
        g.nodes = set(self.nodes)
        g.edges = set(self.edges)
        g.adj = {n: set(neighbors) for n, neighbors in self.adj.items()}
        return g


def build_interference_graph(
    program: list[IRInst],
    live_out: list[set[str]],
) -> InterferenceGraph:
    """Build interference graph from liveness information.

    Two variables interfere if:
      - One is defined and the other is live-out at the same point, OR
      - Both are in the same live-out set.
    """
    graph = InterferenceGraph()

    for i, inst in enumerate(program):
        # All live-out variables at each point interfere with each other
        live = live_out[i]
        live_list = list(live)
        for v in live_list:
            graph.add_node(v)
        for j in range(len(live_list)):
            for k in range(j + 1, len(live_list)):
                graph.add_edge(live_list[j], live_list[k])

        # Defined variable interferes with all live-out variables (except itself)
        for d in inst.defs():
            graph.add_node(d)
            for v in live:
                graph.add_edge(d, v)

    return graph


# ---------------------------------------------------------------------------
# Graph coloring register allocator
# ---------------------------------------------------------------------------

def allocate_registers(
    graph: InterferenceGraph,
    num_regs: int,
) -> tuple[dict[str, str], list[str]]:
    """Allocate physical registers using graph coloring.

    Returns:
        (allocation, spills) where allocation maps virtual -> physical reg
        and spills lists virtual registers that couldn't be colored.
    """
    reg_names = [f"R{i}" for i in range(num_regs)]
    work = graph.copy()
    stack: list[tuple[str, set[str]]] = []  # (node, original_neighbors)
    spill_candidates: list[str] = []

    # --- Phase 1: Simplify ---
    # Remove nodes with degree < K, push onto stack
    while work.nodes:
        # Find a node with degree < num_regs
        found = None
        for n in sorted(work.nodes):  # sort for determinism
            if work.degree(n) < num_regs:
                found = n
                break

        if found is not None:
            # Save original neighbors before removal
            neighbors = set(work.adj.get(found, set()))
            stack.append((found, neighbors))
            work.remove_node(found)
        else:
            # No low-degree node — must spill one (choose highest degree)
            victim = max(work.nodes, key=lambda n: work.degree(n))
            neighbors = set(work.adj.get(victim, set()))
            stack.append((victim, neighbors))
            spill_candidates.append(victim)
            work.remove_node(victim)

    # --- Phase 2: Select (color) ---
    allocation: dict[str, str] = {}
    spills: list[str] = []

    while stack:
        node, neighbors = stack.pop()
        # Colors used by already-allocated neighbors
        used_colors = set()
        for nb in neighbors:
            if nb in allocation:
                used_colors.add(allocation[nb])

        # Pick the first available color
        assigned = None
        for reg in reg_names:
            if reg not in used_colors:
                assigned = reg
                break

        if assigned is not None:
            allocation[node] = assigned
        else:
            spills.append(node)

    return allocation, spills


# ---------------------------------------------------------------------------
# Demos
# ---------------------------------------------------------------------------

def demo_basic_allocation():
    print("=" * 60)
    print("BASIC REGISTER ALLOCATION")
    print("=" * 60)

    # Example: compute (a+b) * (c-d) + e
    #   t1 = a + b
    #   t2 = c - d
    #   t3 = t1 * t2
    #   t4 = t3 + e
    #   return t4
    program = [
        IRInst("load", "a", ["#1"]),
        IRInst("load", "b", ["#2"]),
        IRInst("add",  "t1", ["a", "b"]),
        IRInst("load", "c", ["#3"]),
        IRInst("load", "d", ["#4"]),
        IRInst("sub",  "t2", ["c", "d"]),
        IRInst("mul",  "t3", ["t1", "t2"]),
        IRInst("load", "e", ["#5"]),
        IRInst("add",  "t4", ["t3", "e"]),
        IRInst("ret",  None, ["t4"]),
    ]

    print("\n  Program:")
    for i, inst in enumerate(program):
        print(f"    [{i:2d}] {inst}")

    # Liveness analysis
    live_in, live_out = liveness_analysis(program)
    print("\n  Liveness:")
    print(f"    {'Inst':>4}  {'Live-In':<30} {'Live-Out':<30}")
    print(f"    {'─'*4}  {'─'*30} {'─'*30}")
    for i in range(len(program)):
        in_str = ", ".join(sorted(live_in[i])) or "∅"
        out_str = ", ".join(sorted(live_out[i])) or "∅"
        print(f"    [{i:2d}]  {in_str:<30} {out_str:<30}")

    # Interference graph
    graph = build_interference_graph(program, live_out)
    print("\n  Interference graph edges:")
    for e in sorted(graph.edges, key=lambda e: tuple(sorted(e))):
        u, v = sorted(e)
        print(f"    {u} -- {v}")

    # Allocate with 3 registers
    num_regs = 3
    allocation, spills = allocate_registers(graph, num_regs)
    print(f"\n  Register allocation (K={num_regs}):")
    for var in sorted(allocation):
        print(f"    {var:<6} → {allocation[var]}")
    if spills:
        print(f"  Spilled: {spills}")
    else:
        print("  No spills needed!")


def demo_register_pressure():
    print("\n" + "=" * 60)
    print("REGISTER PRESSURE & SPILLING")
    print("=" * 60)

    # Program with high register pressure — many temporaries live at once
    # t1 = load #1
    # t2 = load #2
    # t3 = load #3
    # t4 = load #4
    # t5 = t1 + t2
    # t6 = t3 + t4
    # t7 = t5 * t6
    # ret t7
    program = [
        IRInst("load", "t1", ["#1"]),
        IRInst("load", "t2", ["#2"]),
        IRInst("load", "t3", ["#3"]),
        IRInst("load", "t4", ["#4"]),
        IRInst("add",  "t5", ["t1", "t2"]),
        IRInst("add",  "t6", ["t3", "t4"]),
        IRInst("mul",  "t7", ["t5", "t6"]),
        IRInst("ret",  None, ["t7"]),
    ]

    print("\n  Program (4 values live simultaneously):")
    for i, inst in enumerate(program):
        print(f"    [{i:2d}] {inst}")

    live_in, live_out = liveness_analysis(program)

    # Show maximum register pressure
    max_pressure = 0
    max_pressure_point = 0
    for i in range(len(program)):
        pressure = len(live_out[i])
        if pressure > max_pressure:
            max_pressure = pressure
            max_pressure_point = i
    print(f"\n  Max register pressure: {max_pressure} (at instruction {max_pressure_point})")

    graph = build_interference_graph(program, live_out)

    # Try with varying register counts
    for k in [4, 3, 2]:
        alloc, spills = allocate_registers(graph, k)
        status = "OK" if not spills else f"SPILLS: {spills}"
        print(f"\n  K={k} registers: {status}")
        for var in sorted(alloc):
            print(f"    {var:<6} → {alloc[var]}")


def demo_loop_liveness():
    print("\n" + "=" * 60)
    print("REGISTER ALLOCATION FOR LOOP-LIKE CODE")
    print("=" * 60)

    # Simulated loop: sum = 0; for i in range(n): sum += arr[i]
    # (straight-line unrolled 3 iterations for analysis)
    program = [
        IRInst("load", "sum", ["#0"]),       # sum = 0
        IRInst("load", "i",   ["#0"]),       # i = 0
        IRInst("load", "n",   ["#10"]),      # n = 10
        IRInst("load", "val", ["i"]),        # val = arr[i]
        IRInst("add",  "sum", ["sum", "val"]),  # sum += val
        IRInst("add",  "i",   ["i", "#1"]),  # i += 1
        IRInst("load", "val", ["i"]),        # val = arr[i]
        IRInst("add",  "sum", ["sum", "val"]),  # sum += val
        IRInst("add",  "i",   ["i", "#1"]),  # i += 1
        IRInst("load", "val", ["i"]),        # val = arr[i]
        IRInst("add",  "sum", ["sum", "val"]),  # sum += val
        IRInst("ret",  None,  ["sum"]),
    ]

    print("\n  Loop-like program (unrolled 3 iterations):")
    for i, inst in enumerate(program):
        print(f"    [{i:2d}] {inst}")

    live_in, live_out = liveness_analysis(program)

    print("\n  Liveness at each point:")
    for i in range(len(program)):
        out_str = ", ".join(sorted(live_out[i])) or "∅"
        print(f"    [{i:2d}] live-out: {out_str}")

    graph = build_interference_graph(program, live_out)
    print(f"\n  Interference graph: {len(graph.nodes)} nodes, {len(graph.edges)} edges")
    for n in sorted(graph.nodes):
        neighbors = ", ".join(sorted(graph.adj[n]))
        print(f"    {n:<6} (deg {graph.degree(n)}) -- {neighbors}")

    alloc, spills = allocate_registers(graph, num_regs=3)
    print(f"\n  Allocation (K=3):")
    for var in sorted(alloc):
        print(f"    {var:<6} → {alloc[var]}")
    if spills:
        print(f"  Spilled: {spills}")


def demo_comparison():
    print("\n" + "=" * 60)
    print("ALLOCATION STRATEGIES COMPARISON")
    print("=" * 60)

    print("""
  Register Allocation Approaches:

    Algorithm              Complexity    Quality    Notes
    ─────────────────────  ───────────   ────────   ──────────────────────
    Graph Coloring         O(n²)         Optimal*   Chaitin (1981), NP-hard
    Linear Scan            O(n log n)    Good       Used in JIT compilers
    PBQP                   Varies        Optimal    Partitioned Boolean QP
    SSA-based              O(n)          Good       Exploits SSA properties

    * Optimal within heuristic limits; true optimal is NP-hard.

  Spill Cost Heuristics:
    - Degree-based:    Spill highest-degree node (used here)
    - Frequency-based: Spill least-frequently-used variable
    - Loop-aware:      Avoid spilling loop-carried variables
    - Rematerialization: Recompute cheap values instead of spilling

  Physical Register Constraints (not modeled here):
    - Calling conventions: some regs are caller/callee-saved
    - Special regs: stack pointer, frame pointer, return address
    - Register classes: integer vs. floating-point vs. SIMD
    - Pre-colored nodes: variables tied to specific physical registers
    """)


if __name__ == "__main__":
    demo_basic_allocation()
    demo_register_pressure()
    demo_loop_liveness()
    demo_comparison()
