"""
Exercises for Lesson 18: SSA Form
Topic: Compiler_Design

Solutions to practice problems from the lesson.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Set, List, Tuple, Optional


# === Exercise 1: Manual SSA Conversion ===
# Problem: Convert code with control flow into SSA form by hand.

def exercise_1():
    """Convert a program with if-else into SSA form."""
    print("Original program:")
    original = [
        "B1: a = read()",
        "    b = read()",
        "    if a > b goto B2 else B3",
        "",
        "B2: c = a - b",
        "    goto B4",
        "",
        "B3: c = b - a",
        "    goto B4",
        "",
        "B4: print(c)",
    ]
    for line in original:
        print(f"  {line}")
    print()

    print("SSA form:")
    ssa = [
        "B1: a1 = read()",
        "    b1 = read()",
        "    if a1 > b1 goto B2 else B3",
        "",
        "B2: c1 = a1 - b1",
        "    goto B4",
        "",
        "B3: c2 = b1 - a1",
        "    goto B4",
        "",
        "B4: c3 = phi(c1:B2, c2:B3)",
        "    print(c3)",
    ]
    for line in ssa:
        print(f"  {line}")
    print()

    print("Explanation:")
    print("  - Variables a, b are defined once, so renaming is straightforward.")
    print("  - Variable c is defined in two branches (B2 and B3).")
    print("  - A phi function for c is needed at B4 (the merge point).")
    print("  - DF(B2) = {B4}, DF(B3) = {B4}, so phi is placed at B4.")
    print()


# === Exercise 2: Compute Dominator Tree ===
# Problem: Given a CFG, compute immediate dominators and dominance frontiers.

def exercise_2():
    """Compute dominators for a simple CFG."""
    # CFG: predecessors mapping
    # B1 -> B2, B3
    # B2 -> B4
    # B3 -> B4
    # B4 -> B5
    # B5 -> B2 (back edge), B6
    # B6 -> (exit)

    cfg_preds = {
        'B1': [],
        'B2': ['B1', 'B5'],
        'B3': ['B1'],
        'B4': ['B2', 'B3'],
        'B5': ['B4'],
        'B6': ['B5'],
    }

    cfg_succs = defaultdict(list)
    for node, preds in cfg_preds.items():
        for p in preds:
            cfg_succs[p].append(node)

    # Compute reverse postorder
    visited = set()
    rpo_list = []

    def dfs(node):
        visited.add(node)
        for succ in cfg_succs.get(node, []):
            if succ not in visited:
                dfs(succ)
        rpo_list.append(node)

    dfs('B1')
    rpo = list(reversed(rpo_list))
    rpo_number = {b: i for i, b in enumerate(rpo)}
    print(f"Reverse postorder: {rpo}")

    # Iterative dominator computation (Cooper-Harvey-Kennedy)
    idom = {b: None for b in cfg_preds}
    idom['B1'] = 'B1'

    def intersect(b1, b2):
        while b1 != b2:
            while rpo_number[b1] > rpo_number[b2]:
                b1 = idom[b1]
            while rpo_number[b2] > rpo_number[b1]:
                b2 = idom[b2]
        return b1

    changed = True
    while changed:
        changed = False
        for b in rpo:
            if b == 'B1':
                continue
            preds = [p for p in cfg_preds[b] if idom[p] is not None]
            if not preds:
                continue
            new_idom = preds[0]
            for p in preds[1:]:
                new_idom = intersect(new_idom, p)
            if idom[b] != new_idom:
                idom[b] = new_idom
                changed = True

    print("\nImmediate dominators:")
    for b in rpo:
        if b != 'B1':
            print(f"  idom({b}) = {idom[b]}")

    # Compute dominance frontiers
    df = {b: set() for b in cfg_preds}
    for b in cfg_preds:
        preds = cfg_preds[b]
        if len(preds) >= 2:
            for p in preds:
                runner = p
                while runner != idom[b]:
                    df[runner].add(b)
                    runner = idom[runner]

    print("\nDominance frontiers:")
    for b in rpo:
        print(f"  DF({b}) = {df[b]}")
    print()


# === Exercise 3: SSA Construction ===
# Problem: Implement SSA construction for simple three-address code.

@dataclass
class Instruction:
    dest: Optional[str]
    op: str
    args: List[str]

    def __repr__(self):
        if self.dest:
            return f"{self.dest} = {self.op} {', '.join(self.args)}"
        return f"{self.op} {', '.join(self.args)}"


def exercise_3():
    """Simple SSA construction for a basic program."""
    print("Exercise 3: SSA Construction")
    print()

    # Simple program in blocks:
    # B1: x = 1
    # B2: x = x + 1; if x < 10 goto B2 else B3
    # B3: print x

    blocks = {
        'B1': [Instruction('x', 'const', ['1'])],
        'B2': [
            Instruction('x', 'add', ['x', '1']),
            Instruction(None, 'if_lt', ['x', '10', 'B2', 'B3']),
        ],
        'B3': [Instruction(None, 'print', ['x'])],
    }

    print("Original program:")
    for name, instrs in blocks.items():
        print(f"  {name}:")
        for instr in instrs:
            print(f"    {instr}")
    print()

    # Variable x is defined in B1 and B2.
    # DF(B1) includes B2 (since B2 is a merge of B1 and B2-back-edge).
    # DF(B2) includes B2 (loop header).
    # So phi for x at B2.

    print("SSA form:")
    ssa_blocks = {
        'B1': ["x1 = const 1", "goto B2"],
        'B2': [
            "x2 = phi(x1:B1, x3:B2)",
            "x3 = add x2, 1",
            "if x3 < 10 goto B2 else B3",
        ],
        'B3': ["print x3"],
    }
    for name, instrs in ssa_blocks.items():
        print(f"  {name}:")
        for instr in instrs:
            print(f"    {instr}")
    print()


# === Exercise 4: SCCP on SSA ===
# Problem: Apply Sparse Conditional Constant Propagation.

def exercise_4():
    """Apply SCCP to a simple SSA program."""
    print("Exercise 4: SCCP on SSA")
    print()

    print("SSA program:")
    program = [
        "B1: a1 = 5",
        "    b1 = 10",
        "    c1 = a1 + b1      // c1 = 15",
        "    if c1 > 10 goto B2 else B3",
        "",
        "B2: d1 = c1 * 2       // d1 = 30",
        "    goto B4",
        "",
        "B3: d2 = 0            // dead block (c1=15 > 10 always true)",
        "    goto B4",
        "",
        "B4: d3 = phi(d1:B2, d2:B3)",
        "    print d3",
    ]
    for line in program:
        print(f"  {line}")
    print()

    print("SCCP analysis:")
    lattice = {
        'a1': 5,
        'b1': 10,
        'c1': 15,       # 5 + 10 = 15
        'd1': 30,        # 15 * 2 = 30
        'd2': 0,         # dead code (B3 unreachable)
        'd3': 30,        # only B2->B4 edge is executable
    }
    for var, val in lattice.items():
        print(f"  {var} = {val}")

    print()
    print("Optimized program:")
    print("  B1: print 30")
    print("  (everything else eliminated)")
    print()


# === Exercise 5: SSA Destruction ===
# Problem: Convert SSA form back to conventional form.

def exercise_5():
    """Destroy SSA by replacing phi functions with copies."""
    print("Exercise 5: SSA Destruction")
    print()

    print("SSA program:")
    ssa = [
        "B1: x1 = 0",
        "    goto B2",
        "",
        "B2: x2 = phi(x1:B1, x3:B3)",
        "    y1 = x2 * 2",
        "    if y1 < 100 goto B3 else B4",
        "",
        "B3: x3 = x2 + 1",
        "    goto B2",
        "",
        "B4: print x2",
    ]
    for line in ssa:
        print(f"  {line}")
    print()

    print("After SSA destruction (phi -> copies):")
    result = [
        "B1: x = 0",
        "    goto B2",
        "",
        "B2: y = x * 2         // x2 -> x (all versions merged)",
        "    if y < 100 goto B3 else B4",
        "",
        "B3: x = x + 1         // x3 = x2 + 1 -> x = x + 1",
        "    goto B2",
        "",
        "B4: print x",
    ]
    for line in result:
        print(f"  {line}")
    print()

    print("Note: In this simple case, coalescing merges all versions of x")
    print("back into a single variable. In general, copies may be needed")
    print("at the end of predecessor blocks, and critical edges may need splitting.")
    print()


# === Main ===

def main():
    exercises = [exercise_1, exercise_2, exercise_3, exercise_4, exercise_5]
    for i, ex in enumerate(exercises, 1):
        print(f"{'=' * 60}")
        print(f"Exercise {i}")
        print(f"{'=' * 60}")
        ex()
        print()


if __name__ == "__main__":
    main()
