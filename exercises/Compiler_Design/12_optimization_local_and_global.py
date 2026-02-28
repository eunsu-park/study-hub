"""
Exercises for Lesson 12: Optimization -- Local and Global
Topic: Compiler_Design

Solutions to practice problems from the lesson.
"""

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, Set, List, Tuple, Optional
import copy


# === Exercise 1: Local Optimization ===
# Problem: Apply local optimizations to a basic block in order:
# constant propagation, constant folding, algebraic simplification,
# copy propagation, dead code elimination.

def exercise_1():
    """Apply local optimizations to a basic block."""
    print("Original basic block (result is live-out):")
    original = [
        "t1 = 4",
        "t2 = 8",
        "t3 = t1 * t2",
        "t4 = t3 + 0",
        "t5 = t4",
        "t6 = t5 * 1",
        "t7 = t6 / 2",
        "t8 = t7 + t7",
        "unused = t1 + t2",
        "result = t8",
    ]
    for line in original:
        print(f"  {line}")
    print()

    # Pass 1: Constant Propagation
    print("Pass 1: Constant Propagation")
    pass1 = [
        "t1 = 4",
        "t2 = 8",
        "t3 = 4 * 8",       # t1,t2 replaced with constants
        "t4 = t3 + 0",
        "t5 = t4",
        "t6 = t5 * 1",
        "t7 = t6 / 2",
        "t8 = t7 + t7",
        "unused = 4 + 8",   # t1,t2 replaced
        "result = t8",
    ]
    for line in pass1:
        print(f"  {line}")
    print()

    # Pass 2: Constant Folding
    print("Pass 2: Constant Folding")
    pass2 = [
        "t1 = 4",
        "t2 = 8",
        "t3 = 32",          # 4 * 8 = 32
        "t4 = t3 + 0",
        "t5 = t4",
        "t6 = t5 * 1",
        "t7 = t6 / 2",
        "t8 = t7 + t7",
        "unused = 12",      # 4 + 8 = 12
        "result = t8",
    ]
    for line in pass2:
        print(f"  {line}")
    print()

    # Pass 3: Algebraic Simplification
    print("Pass 3: Algebraic Simplification")
    pass3 = [
        "t1 = 4",
        "t2 = 8",
        "t3 = 32",
        "t4 = t3",          # x + 0 = x
        "t5 = t4",
        "t6 = t5",          # x * 1 = x
        "t7 = t6 / 2",
        "t8 = t7 + t7",
        "unused = 12",
        "result = t8",
    ]
    for line in pass3:
        print(f"  {line}")
    print()

    # Pass 4: Copy Propagation
    print("Pass 4: Copy Propagation")
    pass4 = [
        "t1 = 4",
        "t2 = 8",
        "t3 = 32",
        "t4 = 32",          # t3 -> 32
        "t5 = 32",          # t4 -> 32
        "t6 = 32",          # t5 -> 32
        "t7 = 32 / 2",      # t6 -> 32
        "t8 = t7 + t7",     # t7 stays (not a simple copy)
        "unused = 12",
        "result = t8",
    ]
    for line in pass4:
        print(f"  {line}")
    print()

    # One more fold
    print("Pass 4b: Constant Folding (second round)")
    pass4b = [
        "t1 = 4",
        "t2 = 8",
        "t3 = 32",
        "t4 = 32",
        "t5 = 32",
        "t6 = 32",
        "t7 = 16",          # 32 / 2 = 16
        "t8 = 32",          # 16 + 16 = 32
        "unused = 12",
        "result = 32",      # t8 -> 32
    ]
    for line in pass4b:
        print(f"  {line}")
    print()

    # Pass 5: Dead Code Elimination
    print("Pass 5: Dead Code Elimination (result is live-out)")
    pass5 = [
        "result = 32",      # only this is needed!
    ]
    for line in pass5:
        print(f"  {line}")
    print()
    print("  All intermediate values were constant and folded away.")
    print("  'unused' was eliminated (not live-out).")
    print("  Original: 10 instructions -> Final: 1 instruction")


# === Exercise 2: Reaching Definitions ===
# Problem: Compute reaching definitions for a CFG.

def exercise_2():
    """Reaching definitions analysis."""
    print("CFG:")
    print("  B0: d0: a = 1")
    print("      d1: b = 2")
    print()
    print("  B1: d2: c = a + b")
    print("      if c > 10 goto B3")
    print()
    print("  B2: d3: a = a + 1")
    print("      d4: b = b - 1")
    print("      goto B1")
    print()
    print("  B3: d5: d = a * b")
    print("      return d")
    print()

    # Definitions: d0(a), d1(b), d2(c), d3(a), d4(b), d5(d)
    all_defs = {'d0', 'd1', 'd2', 'd3', 'd4', 'd5'}
    var_defs = {
        'a': {'d0', 'd3'},
        'b': {'d1', 'd4'},
        'c': {'d2'},
        'd': {'d5'},
    }

    # Gen and Kill sets
    gen = {
        'B0': {'d0', 'd1'},
        'B1': {'d2'},
        'B2': {'d3', 'd4'},
        'B3': {'d5'},
    }
    kill = {
        'B0': {'d3', 'd4'},    # d0 kills d3 (both define a), d1 kills d4 (both define b)
        'B1': set(),            # d2 defines c, no other def of c
        'B2': {'d0', 'd1'},    # d3 kills d0 (a), d4 kills d1 (b)
        'B3': set(),            # d5 defines d, no other def of d
    }

    # Predecessors
    preds = {
        'B0': [],
        'B1': ['B0', 'B2'],
        'B2': ['B1'],
        'B3': ['B1'],
    }

    # Initialize
    in_sets = {b: set() for b in gen}
    out_sets = {b: set() for b in gen}

    print("Gen and Kill sets:")
    for b in ['B0', 'B1', 'B2', 'B3']:
        print(f"  {b}: Gen={gen[b]}, Kill={kill[b]}")
    print()

    # Iterative algorithm
    changed = True
    iteration = 0
    while changed:
        changed = False
        iteration += 1
        for b in ['B0', 'B1', 'B2', 'B3']:
            # In[b] = union of Out[p] for all predecessors p
            new_in = set()
            for p in preds[b]:
                new_in |= out_sets[p]

            # Out[b] = Gen[b] U (In[b] - Kill[b])
            new_out = gen[b] | (new_in - kill[b])

            if new_in != in_sets[b] or new_out != out_sets[b]:
                changed = True
                in_sets[b] = new_in
                out_sets[b] = new_out

    print(f"Reaching Definitions (converged after {iteration} iterations):")
    for b in ['B0', 'B1', 'B2', 'B3']:
        print(f"  {b}:")
        print(f"    In  = {in_sets[b] if in_sets[b] else '{}'}")
        print(f"    Out = {out_sets[b] if out_sets[b] else '{}'}")
    print()

    print("Interpretation:")
    print("  At entry of B1, defs d0,d1 (from B0) or d2,d3,d4 (from B2) may reach.")
    print("  At B3, we know that a is defined by d0 or d3, b by d1 or d4.")


# === Exercise 3: Live Variables ===
# Problem: Compute live variables for the same CFG as Exercise 2.

def exercise_3():
    """Live variables analysis (backward)."""
    print("Same CFG as Exercise 2.")
    print("  d is live at exit of B3.")
    print()

    # Use/Def sets per block
    use_sets = {
        'B0': set(),            # a=1, b=2: no uses before def
        'B1': {'a', 'b'},      # c = a+b, uses a and b before defining c
        'B2': {'a', 'b'},      # a=a+1 uses a; b=b-1 uses b
        'B3': {'a', 'b'},      # d=a*b uses a and b
    }
    def_sets = {
        'B0': {'a', 'b'},
        'B1': {'c'},
        'B2': {'a', 'b'},
        'B3': {'d'},
    }

    # Successors
    succs = {
        'B0': ['B1'],
        'B1': ['B2', 'B3'],
        'B2': ['B1'],
        'B3': [],
    }

    # Initialize
    live_in = {b: set() for b in use_sets}
    live_out = {b: set() for b in use_sets}
    # d is live at exit of B3
    live_out['B3'] = {'d'}

    # Iterative backward analysis
    changed = True
    iteration = 0
    while changed:
        changed = False
        iteration += 1
        for b in ['B3', 'B2', 'B1', 'B0']:  # backward order
            # Out[b] = union of In[s] for all successors s
            new_out = set()
            for s in succs[b]:
                new_out |= live_in[s]
            if b == 'B3':
                new_out |= {'d'}  # d live at exit

            # In[b] = Use[b] U (Out[b] - Def[b])
            new_in = use_sets[b] | (new_out - def_sets[b])

            if new_in != live_in[b] or new_out != live_out[b]:
                changed = True
                live_in[b] = new_in
                live_out[b] = new_out

    print(f"Live Variables (converged after {iteration} iterations):")
    for b in ['B0', 'B1', 'B2', 'B3']:
        print(f"  {b}:")
        print(f"    LiveIn  = {live_in[b] if live_in[b] else '{}'}")
        print(f"    LiveOut = {live_out[b] if live_out[b] else '{}'}")
    print()

    print("Interpretation:")
    print("  a and b are live throughout (used in B1, B2, B3).")
    print("  c is not live anywhere after B1 (computed but never used later).")
    print("  -> c = a + b is dead code if c is only used in the condition 'c > 10'")
    print("     (wait, c IS used in the condition, so it's live in B1)")


# === Exercise 4: Available Expressions ===
# Problem: Compute available expressions for a given CFG.

def exercise_4():
    """Available expressions analysis."""
    print("CFG:")
    print("  B0: t1 = a + b")
    print("      t2 = c + d")
    print()
    print("  B1: t3 = a + b")
    print("      if (t3 > 0) goto B2 else goto B3")
    print()
    print("  B2: a = a + 1        (kills a+b)")
    print("      t4 = c + d")
    print("      goto B4")
    print()
    print("  B3: t5 = a + b")
    print("      t6 = c + d")
    print("      goto B4")
    print()
    print("  B4: t7 = a + b")
    print("      t8 = c + d")
    print()

    # Expressions: 'a+b', 'c+d'
    all_exprs = {'a+b', 'c+d'}

    # Gen: expressions computed (and not subsequently killed) in the block
    # Kill: expressions killed by assignments to their operands
    e_gen = {
        'B0': {'a+b', 'c+d'},
        'B1': {'a+b'},           # t3 = a+b
        'B2': {'c+d'},           # a=a+1 kills a+b, then t4=c+d generates c+d
        'B3': {'a+b', 'c+d'},
        'B4': {'a+b', 'c+d'},   # both computed here
    }
    e_kill = {
        'B0': set(),
        'B1': set(),
        'B2': {'a+b'},           # a = a + 1 kills a+b
        'B3': set(),
        'B4': set(),             # no assignments to a,b,c,d
    }

    preds = {
        'B0': [],
        'B1': ['B0'],
        'B2': ['B1'],
        'B3': ['B1'],
        'B4': ['B2', 'B3'],
    }

    # Available expressions is a forward, must (intersection) analysis
    # In[B0] = {} (nothing available at entry)
    # In[b] = intersection of Out[p] for all predecessors p
    # Out[b] = Gen[b] U (In[b] - Kill[b])

    avail_in = {b: set() for b in e_gen}
    avail_out = {b: set() for b in e_gen}
    # Initialize all non-entry blocks to universal set
    for b in e_gen:
        if b != 'B0':
            avail_in[b] = set(all_exprs)
            avail_out[b] = set(all_exprs)

    changed = True
    iteration = 0
    while changed:
        changed = False
        iteration += 1
        for b in ['B0', 'B1', 'B2', 'B3', 'B4']:
            if b == 'B0':
                new_in = set()
            elif preds[b]:
                new_in = set.intersection(*[avail_out[p] for p in preds[b]])
            else:
                new_in = set()

            new_out = e_gen[b] | (new_in - e_kill[b])

            if new_in != avail_in[b] or new_out != avail_out[b]:
                changed = True
                avail_in[b] = new_in
                avail_out[b] = new_out

    print(f"Available Expressions (converged after {iteration} iterations):")
    for b in ['B0', 'B1', 'B2', 'B3', 'B4']:
        print(f"  {b}:")
        print(f"    AvailIn  = {avail_in[b] if avail_in[b] else '{}'}")
        print(f"    AvailOut = {avail_out[b] if avail_out[b] else '{}'}")
    print()

    # Answer the questions
    print("Answers:")
    print(f"  Is a+b available at B1 entry? {'a+b' in avail_in['B1']} (from B0)")
    print(f"  Is a+b available at B4 entry? {'a+b' in avail_in['B4']}")
    print(f"    B4 preds: B2(out={'a+b' in avail_out['B2']}), B3(out={'a+b' in avail_out['B3']})")
    print(f"    a+b killed in B2 (a=a+1), so NOT available via B2 path.")
    print(f"    Intersection: a+b not available at B4 entry.")
    print(f"  Is c+d available at B4 entry? {'c+d' in avail_in['B4']}")
    print(f"    Both B2 and B3 generate c+d -> YES, available at B4.")


# === Exercise 5: Worklist Algorithm Trace ===
# Problem: Trace worklist algorithm for live variables.

def exercise_5():
    """Trace the worklist algorithm for live variables analysis."""
    print("CFG:")
    print("  B0: x = read()")
    print("      y = read()")
    print()
    print("  B1: if (x > 0) goto B2 else goto B3")
    print()
    print("  B2: z = x + y")
    print("      x = x - 1")
    print("      goto B1")
    print()
    print("  B3: print z")
    print()

    use_sets = {'B0': set(), 'B1': {'x'}, 'B2': {'x', 'y'}, 'B3': {'z'}}
    def_sets = {'B0': {'x', 'y'}, 'B1': set(), 'B2': {'z', 'x'}, 'B3': set()}
    succs = {'B0': ['B1'], 'B1': ['B2', 'B3'], 'B2': ['B1'], 'B3': []}
    preds = {'B0': [], 'B1': ['B0', 'B2'], 'B2': ['B1'], 'B3': ['B1']}

    live_in = {b: set() for b in use_sets}
    live_out = {b: set() for b in use_sets}

    # Worklist: start with all blocks
    worklist = deque(['B3', 'B2', 'B1', 'B0'])
    step = 0

    print()
    print("Worklist algorithm trace (backward analysis):")
    print(f"  {'Step':<5} {'Process':<8} {'Worklist':<20} {'In':<30} {'Out':<30} {'Changed?'}")
    print(f"  {'-'*5} {'-'*8} {'-'*20} {'-'*30} {'-'*30} {'-'*10}")

    while worklist:
        step += 1
        b = worklist.popleft()

        # Compute Out[b]
        new_out = set()
        for s in succs[b]:
            new_out |= live_in[s]

        # Compute In[b]
        new_in = use_sets[b] | (new_out - def_sets[b])

        changed = new_in != live_in[b] or new_out != live_out[b]

        in_str = f"In={new_in}" if new_in else "In={}"
        out_str = f"Out={new_out}" if new_out else "Out={}"
        wl_str = str(list(worklist))

        print(f"  {step:<5} {b:<8} {wl_str:<20} {in_str:<30} {out_str:<30} {str(changed)}")

        if changed:
            live_in[b] = new_in
            live_out[b] = new_out
            # Add predecessors to worklist
            for p in preds[b]:
                if p not in worklist:
                    worklist.append(p)

    print()
    print("Final results:")
    for b in ['B0', 'B1', 'B2', 'B3']:
        print(f"  {b}: In={live_in[b]}, Out={live_out[b]}")
    print()
    print("Observations:")
    print("  - x and y are live at B0 exit (needed in loop)")
    print("  - z is live from B2 (where it's defined) through B1 to B3")
    print("  - x is live in the loop (B1<->B2)")


# === Exercise 6: Implementation Challenge ===
# Problem: Implement constant propagation and copy propagation analyses.

@dataclass
class TACInstruction:
    """Three-address code instruction."""
    dest: Optional[str] = None
    op: Optional[str] = None
    arg1: Optional[str] = None
    arg2: Optional[str] = None
    raw: str = ""

    def __repr__(self):
        return self.raw


def parse_tac(line):
    """Parse a simple TAC instruction."""
    line = line.strip()
    if '=' in line and not line.startswith('if'):
        parts = line.split('=', 1)
        dest = parts[0].strip()
        rhs = parts[1].strip()
        tokens = rhs.split()
        if len(tokens) == 1:
            return TACInstruction(dest, 'copy', tokens[0], None, line)
        elif len(tokens) == 3:
            return TACInstruction(dest, tokens[1], tokens[0], tokens[2], line)
    return TACInstruction(raw=line)


class ConstantPropagationAnalysis:
    """
    Forward data flow analysis for constant propagation.
    Lattice per variable: TOP (undefined) > constant c > BOTTOM (not constant)
    Meet: TOP meet x = x, c meet c = c, c meet d = BOTTOM
    """

    TOP = 'TOP'
    BOT = 'BOT'

    def __init__(self, blocks, preds, succs):
        self.blocks = blocks     # block_name -> list of TACInstruction
        self.preds = preds
        self.succs = succs
        self.state_in = {}       # block -> {var -> value}
        self.state_out = {}

    def meet(self, val1, val2):
        if val1 == self.TOP:
            return val2
        if val2 == self.TOP:
            return val1
        if val1 == val2:
            return val1
        return self.BOT

    def meet_states(self, state1, state2):
        result = {}
        all_vars = set(state1.keys()) | set(state2.keys())
        for v in all_vars:
            v1 = state1.get(v, self.TOP)
            v2 = state2.get(v, self.TOP)
            result[v] = self.meet(v1, v2)
        return result

    def transfer(self, block_name, in_state):
        """Apply transfer function for a block."""
        state = dict(in_state)
        for instr in self.blocks[block_name]:
            if instr.dest and instr.op:
                if instr.op == 'copy':
                    # x = y: if y is a constant, x becomes that constant
                    val = self._eval_operand(instr.arg1, state)
                    state[instr.dest] = val
                elif instr.op in ('+', '-', '*', '/'):
                    v1 = self._eval_operand(instr.arg1, state)
                    v2 = self._eval_operand(instr.arg2, state)
                    if isinstance(v1, int) and isinstance(v2, int):
                        ops = {'+': lambda a, b: a+b, '-': lambda a, b: a-b,
                               '*': lambda a, b: a*b, '/': lambda a, b: a//b if b else 0}
                        state[instr.dest] = ops[instr.op](v1, v2)
                    elif v1 == self.BOT or v2 == self.BOT:
                        state[instr.dest] = self.BOT
                    else:
                        state[instr.dest] = self.TOP
        return state

    def _eval_operand(self, operand, state):
        try:
            return int(operand)
        except (ValueError, TypeError):
            return state.get(operand, self.TOP)

    def analyze(self):
        """Run the analysis."""
        # Initialize
        for b in self.blocks:
            self.state_in[b] = {}
            self.state_out[b] = {}

        worklist = deque(self.blocks.keys())
        while worklist:
            b = worklist.popleft()
            # Compute In
            if self.preds[b]:
                new_in = self.state_out[self.preds[b][0]].copy()
                for p in self.preds[b][1:]:
                    new_in = self.meet_states(new_in, self.state_out[p])
            else:
                new_in = {}

            new_out = self.transfer(b, new_in)
            if new_out != self.state_out[b]:
                self.state_out[b] = new_out
                self.state_in[b] = new_in
                for s in self.succs[b]:
                    if s not in worklist:
                        worklist.append(s)
            else:
                self.state_in[b] = new_in


def exercise_6():
    """Constant propagation and copy propagation analyses."""
    # Use the CFG from Exercise 2
    blocks = {
        'B0': [
            parse_tac("a = 1"),
            parse_tac("b = 2"),
        ],
        'B1': [
            parse_tac("c = a + b"),
        ],
        'B2': [
            parse_tac("a = a + 1"),
            parse_tac("b = b - 1"),
        ],
        'B3': [
            parse_tac("d = a * b"),
        ],
    }
    preds = {'B0': [], 'B1': ['B0', 'B2'], 'B2': ['B1'], 'B3': ['B1']}
    succs = {'B0': ['B1'], 'B1': ['B2', 'B3'], 'B2': ['B1'], 'B3': []}

    print("1. Constant Propagation Analysis")
    print("=" * 50)
    analysis = ConstantPropagationAnalysis(blocks, preds, succs)
    analysis.analyze()

    for b in ['B0', 'B1', 'B2', 'B3']:
        print(f"  {b}:")
        print(f"    In:  {analysis.state_in[b]}")
        print(f"    Out: {analysis.state_out[b]}")
    print()

    print("  Interpretation:")
    print("    B0 out: a=1, b=2 (constants)")
    print("    B1 in: a and b may be constants (from B0) or modified (from B2)")
    print("    After merge at B1: a and b become BOT (not constant) because")
    print("    B2 modifies them. So c = a + b cannot be folded after the first iteration.")
    print("    In the first iteration only: a=1, b=2, c=3 -- but once B2 feeds back,")
    print("    a and b become non-constant (BOT).")
    print()

    # 2. Copy Propagation
    print("2. Copy Propagation Analysis")
    print("=" * 50)
    print()

    # Copy propagation tracks which copies x = y are valid at each point
    # A copy x = y is killed when x or y is redefined
    copy_blocks = {
        'B0': [parse_tac("x = a"), parse_tac("y = b")],
        'B1': [parse_tac("z = x + y")],
    }
    copy_preds = {'B0': [], 'B1': ['B0']}
    copy_succs = {'B0': ['B1'], 'B1': []}

    print("  Simple example:")
    print("  B0: x = a; y = b")
    print("  B1: z = x + y")
    print()

    # At entry of B1, copies {x=a, y=b} are valid
    # So z = x + y can be replaced with z = a + b
    print("  Active copies at B1 entry: {x -> a, y -> b}")
    print("  After copy propagation:")
    print("    z = a + b   (replaced x with a, y with b)")
    print()
    print("  This is useful because it may enable further optimizations:")
    print("  - If a and b are constants, constant folding can compute z")
    print("  - x and y may become dead (enabling dead code elimination)")
    print()

    print("  Combined optimization example:")
    print("  Before:")
    print("    a = 5")
    print("    x = a          // copy")
    print("    y = x + 3      // uses x")
    print()
    print("  After constant propagation: a = 5, x = 5")
    print("  After copy propagation: y = 5 + 3")
    print("  After constant folding: y = 8")
    print("  After dead code elimination: (a and x removed if not live-out)")
    print("  Final: y = 8")


if __name__ == "__main__":
    print("=" * 60)
    print("=== Exercise 1: Local Optimization ===")
    print("=" * 60)
    exercise_1()

    print("\n" + "=" * 60)
    print("=== Exercise 2: Reaching Definitions ===")
    print("=" * 60)
    exercise_2()

    print("\n" + "=" * 60)
    print("=== Exercise 3: Live Variables ===")
    print("=" * 60)
    exercise_3()

    print("\n" + "=" * 60)
    print("=== Exercise 4: Available Expressions ===")
    print("=" * 60)
    exercise_4()

    print("\n" + "=" * 60)
    print("=== Exercise 5: Worklist Algorithm Trace ===")
    print("=" * 60)
    exercise_5()

    print("\n" + "=" * 60)
    print("=== Exercise 6: Implementation Challenge ===")
    print("=" * 60)
    exercise_6()

    print("\nAll exercises completed!")
