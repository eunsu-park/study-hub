# Loop Optimization

**Previous**: [12. Optimization -- Local and Global](./12_Optimization_Local_and_Global.md) | **Next**: [14. Garbage Collection](./14_Garbage_Collection.md)

---

Programs spend the vast majority of their execution time inside loops. A rule of thumb (sometimes called the 90/10 rule) states that 90% of execution time is spent in 10% of the code -- and that 10% is almost always inside loops. Because of this, loop optimization is the single most impactful category of compiler optimization. A loop that executes a million times amplifies any improvement by a factor of a million.

This lesson covers the full spectrum of loop optimizations: from detecting loops in the control flow graph, to classical transformations like loop-invariant code motion and strength reduction, all the way to modern techniques like vectorization and the polyhedral model.

**Difficulty**: ⭐⭐⭐⭐

**Prerequisites**: [09. Intermediate Representations](./09_Intermediate_Representations.md), [12. Optimization -- Local and Global](./12_Optimization_Local_and_Global.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Identify natural loops using dominance analysis and back edges
2. Construct dominator trees and compute dominance frontiers
3. Apply loop-invariant code motion (LICM) to hoist computations out of loops
4. Recognize and optimize induction variables through strength reduction
5. Understand loop unrolling and its trade-offs
6. Apply loop transformations: fusion, fission, interchange, tiling
7. Explain the basics of vectorization and the polyhedral model
8. Reason about loop parallelization opportunities

---

## Table of Contents

1. [Why Loop Optimization Matters](#1-why-loop-optimization-matters)
2. [Dominators and Dominator Trees](#2-dominators-and-dominator-trees)
3. [Detecting Natural Loops](#3-detecting-natural-loops)
4. [Loop-Invariant Code Motion (LICM)](#4-loop-invariant-code-motion-licm)
5. [Induction Variable Analysis](#5-induction-variable-analysis)
6. [Strength Reduction](#6-strength-reduction)
7. [Loop Unrolling](#7-loop-unrolling)
8. [Loop Fusion and Fission](#8-loop-fusion-and-fission)
9. [Loop Tiling (Blocking)](#9-loop-tiling-blocking)
10. [Loop Interchange](#10-loop-interchange)
11. [Vectorization](#11-vectorization)
12. [The Polyhedral Model](#12-the-polyhedral-model)
13. [Loop Parallelization](#13-loop-parallelization)
14. [Summary](#14-summary)
15. [Exercises](#15-exercises)
16. [References](#16-references)

---

## 1. Why Loop Optimization Matters

Consider a simple loop:

```python
# Before optimization
total = 0
for i in range(1_000_000):
    x = a * b + c        # Invariant: same result every iteration
    total += arr[i] + x
```

The expression `a * b + c` is recomputed one million times, even though it never changes. Moving it out of the loop gives:

```python
# After optimization
x = a * b + c            # Computed once
total = 0
for i in range(1_000_000):
    total += arr[i] + x
```

This eliminates 999,999 redundant multiplications and additions. Multiply such savings across nested loops, and the impact is enormous.

### The Optimization Landscape

Loop optimizations fall into several categories:

```
┌─────────────────────────────────────────────────────────────────┐
│                    Loop Optimizations                            │
├─────────────────────┬──────────────────────┬────────────────────┤
│  Reduce Work        │  Improve Locality    │  Exploit Hardware  │
│                     │                      │                    │
│  - LICM             │  - Loop tiling       │  - Vectorization   │
│  - Strength         │  - Loop interchange  │  - Parallelization │
│    reduction        │  - Loop fusion       │  - Software        │
│  - Dead code in     │  - Data prefetching  │    pipelining      │
│    loops            │                      │                    │
│  - Loop unrolling   │                      │                    │
│  - Loop unswitching │                      │                    │
└─────────────────────┴──────────────────────┴────────────────────┘
```

Before we can optimize loops, we need to find them. That requires understanding dominators.

---

## 2. Dominators and Dominator Trees

### 2.1 Dominance

Given a control flow graph (CFG) with entry node $\text{entry}$, we say that node $d$ **dominates** node $n$ (written $d \;\text{dom}\; n$) if every path from $\text{entry}$ to $n$ must pass through $d$.

Key properties:

- **Reflexivity**: Every node dominates itself: $n \;\text{dom}\; n$.
- **Transitivity**: If $a \;\text{dom}\; b$ and $b \;\text{dom}\; c$, then $a \;\text{dom}\; c$.
- **Antisymmetry**: If $a \;\text{dom}\; b$ and $b \;\text{dom}\; a$, then $a = b$.

The **immediate dominator** of $n$ (written $\text{idom}(n)$) is the closest strict dominator of $n$: the dominator $d \neq n$ such that every other dominator of $n$ also dominates $d$.

### 2.2 Computing Dominators

The classic iterative algorithm starts by initializing every node's dominator set to contain all nodes, then repeatedly refines by intersection:

```python
def compute_dominators(cfg, entry):
    """
    Compute dominator sets for each node in a CFG.

    cfg: dict mapping node -> list of predecessor nodes
    entry: the entry node of the CFG
    Returns: dict mapping node -> set of dominator nodes
    """
    all_nodes = set(cfg.keys())
    dom = {}

    # Initialize
    dom[entry] = {entry}
    for node in all_nodes:
        if node != entry:
            dom[node] = set(all_nodes)  # Start with all nodes

    # Iterate until fixed point
    changed = True
    while changed:
        changed = False
        for node in all_nodes:
            if node == entry:
                continue

            predecessors = cfg[node]
            if not predecessors:
                continue

            # New dom set = intersection of all predecessors' dom sets, plus self
            new_dom = set.intersection(*(dom[p] for p in predecessors))
            new_dom.add(node)

            if new_dom != dom[node]:
                dom[node] = new_dom
                changed = True

    return dom
```

**Example**:

```
CFG:
    entry -> A -> B -> C -> D
                  ^         |
                  |         |
                  +----E----+

Predecessors:
  entry: []
  A: [entry]
  B: [A, E]
  C: [B]
  D: [C]
  E: [D]
```

```python
cfg = {
    'entry': [],
    'A': ['entry'],
    'B': ['A', 'E'],
    'C': ['B'],
    'D': ['C'],
    'E': ['D'],
}

dom = compute_dominators(cfg, 'entry')

# Result:
# dom['entry'] = {'entry'}
# dom['A']     = {'entry', 'A'}
# dom['B']     = {'entry', 'A', 'B'}
# dom['C']     = {'entry', 'A', 'B', 'C'}
# dom['D']     = {'entry', 'A', 'B', 'C', 'D'}
# dom['E']     = {'entry', 'A', 'B', 'C', 'D', 'E'}
```

### 2.3 The Dominator Tree

The dominator tree represents the immediate dominance relation. Each node's parent in the tree is its immediate dominator.

```python
def build_dominator_tree(dom, entry):
    """
    Build the dominator tree from dominator sets.

    Returns: dict mapping node -> immediate dominator
    """
    idom = {}
    for node in dom:
        if node == entry:
            continue
        # Immediate dominator is the closest strict dominator
        strict_doms = dom[node] - {node}
        # idom is the one that is dominated by all others
        # (i.e., has the largest dominator set among strict dominators)
        idom_node = max(strict_doms, key=lambda d: len(dom[d]))
        idom[node] = idom_node
    return idom


def print_dominator_tree(idom, entry, indent=0):
    """Print the dominator tree."""
    print("  " * indent + entry)
    children = [n for n, d in idom.items() if d == entry]
    for child in sorted(children):
        print_dominator_tree(idom, child, indent + 1)
```

For the example above:

```
Dominator Tree:
entry
  A
    B
      C
        D
          E
```

### 2.4 The Cooper-Harvey-Kennedy Algorithm

The most widely used algorithm in practice is the Cooper-Harvey-Kennedy (CHK) algorithm, which computes dominators in near-linear time using a reverse postorder traversal:

```python
def compute_rpo(cfg_successors, entry):
    """Compute reverse postorder of CFG nodes."""
    visited = set()
    postorder = []

    def dfs(node):
        visited.add(node)
        for succ in cfg_successors.get(node, []):
            if succ not in visited:
                dfs(succ)
        postorder.append(node)

    dfs(entry)
    postorder.reverse()
    return postorder


def intersect(idom, rpo_number, b1, b2):
    """Find common dominator using the finger-moving technique."""
    finger1 = b1
    finger2 = b2
    while finger1 != finger2:
        while rpo_number[finger1] > rpo_number[finger2]:
            finger1 = idom[finger1]
        while rpo_number[finger2] > rpo_number[finger1]:
            finger2 = idom[finger2]
    return finger1


def compute_idom_chk(cfg_preds, cfg_succs, entry):
    """
    Cooper-Harvey-Kennedy dominator algorithm.

    cfg_preds: dict node -> list of predecessors
    cfg_succs: dict node -> list of successors
    entry: entry node
    Returns: dict node -> immediate dominator
    """
    rpo = compute_rpo(cfg_succs, entry)
    rpo_number = {node: i for i, node in enumerate(rpo)}

    idom = {entry: entry}

    changed = True
    while changed:
        changed = False
        for b in rpo:
            if b == entry:
                continue
            preds = [p for p in cfg_preds[b] if p in idom]
            if not preds:
                continue

            new_idom = preds[0]
            for p in preds[1:]:
                new_idom = intersect(idom, rpo_number, p, new_idom)

            if idom.get(b) != new_idom:
                idom[b] = new_idom
                changed = True

    return idom
```

### 2.5 Dominance Frontiers

The **dominance frontier** of a node $n$ is the set of nodes where $n$'s dominance just "stops" -- the first nodes that $n$ does not strictly dominate but that have a predecessor dominated by $n$.

$$DF(n) = \{ y \mid \exists\, x \in \text{pred}(y) : n \;\text{dom}\; x \land n \not\text{sdom}\; y \}$$

where $\text{sdom}$ means "strictly dominates" (dominates and is not equal to).

Dominance frontiers are crucial for placing $\phi$-functions when constructing SSA form (covered in Lesson 9).

```python
def compute_dominance_frontiers(cfg_preds, idom, all_nodes):
    """
    Compute dominance frontiers for all nodes.

    Returns: dict mapping node -> set of nodes in its dominance frontier
    """
    df = {n: set() for n in all_nodes}

    for node in all_nodes:
        preds = cfg_preds.get(node, [])
        if len(preds) >= 2:  # Join point
            for pred in preds:
                runner = pred
                while runner != idom.get(node):
                    df[runner].add(node)
                    runner = idom.get(runner)

    return df
```

---

## 3. Detecting Natural Loops

### 3.1 Back Edges

A **back edge** in a CFG is an edge $n \to h$ where $h$ dominates $n$. Back edges indicate the presence of loops.

```python
def find_back_edges(cfg_succs, idom_sets):
    """
    Find all back edges in the CFG.

    cfg_succs: dict node -> list of successors
    idom_sets: dict node -> set of all dominators of that node
    Returns: list of (tail, head) tuples
    """
    back_edges = []
    for node in cfg_succs:
        for succ in cfg_succs[node]:
            if succ in idom_sets.get(node, set()):
                # succ dominates node, so node -> succ is a back edge
                back_edges.append((node, succ))
    return back_edges
```

### 3.2 Natural Loops

Given a back edge $n \to h$, the **natural loop** consists of the **header** $h$ and all nodes that can reach $n$ without going through $h$.

```python
def find_natural_loop(back_edge_tail, header, cfg_preds):
    """
    Find the natural loop for a back edge tail -> header.

    Returns: set of nodes in the loop (including header)
    """
    loop = {header}
    worklist = []

    if back_edge_tail != header:
        loop.add(back_edge_tail)
        worklist.append(back_edge_tail)

    while worklist:
        node = worklist.pop()
        for pred in cfg_preds.get(node, []):
            if pred not in loop:
                loop.add(pred)
                worklist.append(pred)

    return loop
```

**Example -- Detecting loops in a CFG**:

```python
# Example CFG with two loops
#
#  entry -> A -> B -> C -> D -> exit
#                ^    |
#                |    v
#                +--- E
#           and D -> B (outer loop)

cfg_succs = {
    'entry': ['A'],
    'A': ['B'],
    'B': ['C'],
    'C': ['D', 'E'],
    'D': ['exit', 'B'],  # D->B is a back edge (B dom D)
    'E': ['B'],           # E->B is a back edge (B dom E)
    'exit': [],
}

cfg_preds = {
    'entry': [],
    'A': ['entry'],
    'B': ['A', 'D', 'E'],
    'C': ['B'],
    'D': ['C'],
    'E': ['C'],
    'exit': ['D'],
}

# After computing dominators, we find:
# Back edges: (D, B) and (E, B)
# Natural loop for D->B: {B, C, D, E}
# Natural loop for E->B: {B, C, E}
```

### 3.3 Loop Nesting and the Loop Tree

Loops can be nested. The **loop tree** (or **loop forest**) organizes loops by nesting:

```
Loop Tree:
  Function
  ├── Loop L1 (header: B, body: {B, C, D, E})
  │   └── Loop L2 (header: B, body: {B, C, E})  [inner loop]
  └── (no more top-level loops)
```

When two loops share a header, they may be merged into one loop if their bodies overlap.

```python
class Loop:
    """Represents a natural loop in the CFG."""

    def __init__(self, header, body, back_edges):
        self.header = header
        self.body = body           # Set of nodes
        self.back_edges = back_edges
        self.parent = None         # Enclosing loop
        self.children = []         # Nested loops
        self.depth = 0             # Nesting depth (0 = outermost)

    def __repr__(self):
        return f"Loop(header={self.header}, body={self.body}, depth={self.depth})"


def build_loop_tree(loops):
    """
    Build a loop nesting tree from a list of loops.

    Loops are sorted by body size; smaller loops are children of larger ones.
    """
    sorted_loops = sorted(loops, key=lambda l: len(l.body))

    for i, inner in enumerate(sorted_loops):
        for outer in sorted_loops[i+1:]:
            if inner.body.issubset(outer.body):
                inner.parent = outer
                outer.children.append(inner)
                break

    # Compute depths
    for loop in sorted_loops:
        depth = 0
        current = loop
        while current.parent is not None:
            depth += 1
            current = current.parent
        loop.depth = depth

    # Return top-level loops (those with no parent)
    return [l for l in sorted_loops if l.parent is None]
```

### 3.4 Preheader Insertion

Many loop optimizations need a place to put code that should execute exactly once before the loop. The **preheader** is a dedicated block inserted between the loop's non-back-edge predecessors and the header:

```
Before:                       After:

  A ──┐                        A ──┐
       ├──▶ Header                  ├──▶ Preheader ──▶ Header
  B ──┘     ▲                  B ──┘                   ▲
             │                                          │
         back edge                                  back edge
             │                                          │
           Latch                                      Latch
```

```python
def insert_preheader(cfg_preds, cfg_succs, header, loop_body):
    """
    Insert a preheader block for a loop.

    Modifies cfg_preds and cfg_succs in place.
    Returns the name of the new preheader node.
    """
    preheader = f"pre_{header}"

    # Find non-back-edge predecessors of header
    external_preds = [p for p in cfg_preds[header] if p not in loop_body]

    # Redirect external predecessors to preheader
    cfg_preds[preheader] = external_preds
    cfg_succs[preheader] = [header]

    # Update header's predecessors
    cfg_preds[header] = [p for p in cfg_preds[header] if p in loop_body]
    cfg_preds[header].append(preheader)

    # Update external predecessors' successors
    for pred in external_preds:
        cfg_succs[pred] = [preheader if s == header else s
                           for s in cfg_succs[pred]]

    return preheader
```

---

## 4. Loop-Invariant Code Motion (LICM)

### 4.1 Identifying Loop-Invariant Computations

A computation inside a loop is **loop-invariant** if its result does not change across iterations. Formally, an instruction `x = op(a, b)` is loop-invariant if:

1. All operands are either constants, or
2. All definitions of each operand that reach this instruction are outside the loop, or
3. There is exactly one reaching definition for each operand, and that definition is itself loop-invariant

```python
def find_loop_invariant_instructions(loop_body, instructions, reaching_defs):
    """
    Find all loop-invariant instructions.

    loop_body: set of basic blocks in the loop
    instructions: dict block -> list of (target, op, operands)
    reaching_defs: dict (block, var) -> set of defining blocks

    Returns: set of loop-invariant instructions (block, index)
    """
    # Collect all variables defined inside the loop
    loop_defs = set()
    for block in loop_body:
        for idx, (target, op, operands) in enumerate(instructions.get(block, [])):
            loop_defs.add(target)

    invariant = set()
    changed = True

    while changed:
        changed = False
        for block in loop_body:
            for idx, (target, op, operands) in enumerate(instructions.get(block, [])):
                if (block, idx) in invariant:
                    continue

                # Check if all operands are loop-invariant
                all_invariant = True
                for operand in operands:
                    if operand not in loop_defs:
                        # Defined outside the loop -- invariant
                        continue

                    # Check if operand has a single reaching def that is invariant
                    defs = reaching_defs.get((block, operand), set())
                    loop_local_defs = defs & loop_body

                    if not loop_local_defs:
                        continue  # All defs outside loop

                    # Must have exactly one loop-local def, and it must be invariant
                    if len(loop_local_defs) != 1:
                        all_invariant = False
                        break

                    def_block = loop_local_defs.pop()
                    # Find the instruction index in def_block
                    found_invariant = False
                    for di, (dt, _, _) in enumerate(instructions.get(def_block, [])):
                        if dt == operand and (def_block, di) in invariant:
                            found_invariant = True
                            break

                    if not found_invariant:
                        all_invariant = False
                        break

                if all_invariant:
                    invariant.add((block, idx))
                    changed = True

    return invariant
```

### 4.2 Conditions for Safe Code Motion

Not every loop-invariant instruction can be safely moved to the preheader. The instruction `s: x = op(a, b)` can be hoisted if:

1. **Dominance condition**: The block containing $s$ dominates all exits of the loop (the instruction would have executed anyway on every iteration that reaches an exit).
2. **Uniqueness condition**: No other definition of $x$ exists in the loop.
3. **Liveness condition**: The variable $x$ is not live at any definition of $x$ in the loop (no other use of a different definition of $x$).

Alternatively, if the instruction has no side effects and its result is only used inside the loop, we can always speculatively hoist it (at worst we compute something unused).

```python
def can_hoist(instr_block, instr_target, loop_body, loop_exits, dom_sets,
              all_defs_in_loop, live_at_defs):
    """
    Check if a loop-invariant instruction can be safely hoisted.

    instr_block: the block containing the instruction
    instr_target: the variable being defined
    loop_body: set of blocks in the loop
    loop_exits: set of blocks that are loop exits
    dom_sets: dict block -> set of blocks it dominates
    all_defs_in_loop: dict variable -> set of defining blocks in loop
    live_at_defs: dict (block, variable) -> bool
    """
    # Condition 1: Block dominates all loop exits
    for exit_block in loop_exits:
        if exit_block not in dom_sets.get(instr_block, set()):
            return False

    # Condition 2: Unique definition in loop
    if len(all_defs_in_loop.get(instr_target, set())) > 1:
        return False

    # Condition 3: No interference with other defs
    for def_block in all_defs_in_loop.get(instr_target, set()):
        if def_block != instr_block:
            if live_at_defs.get((def_block, instr_target), False):
                return False

    return True
```

### 4.3 LICM in Practice

Here is a complete example demonstrating LICM on a simple loop:

```python
class SimpleInstruction:
    """Represents a three-address instruction."""
    def __init__(self, target, op, operands, is_side_effect=False):
        self.target = target
        self.op = op
        self.operands = operands
        self.is_side_effect = is_side_effect

    def __repr__(self):
        if self.op == 'copy':
            return f"{self.target} = {self.operands[0]}"
        return f"{self.target} = {self.operands[0]} {self.op} {self.operands[1]}"


def demonstrate_licm():
    """
    Demonstrate LICM on a simple loop.

    Original code:
        i = 0
        while i < n:
            x = a * b        # invariant
            y = x + c        # invariant (depends on x which is invariant)
            arr[i] = y + i
            i = i + 1

    After LICM:
        x = a * b            # hoisted
        y = x + c            # hoisted
        i = 0
        while i < n:
            arr[i] = y + i
            i = i + 1
    """
    print("=== Before LICM ===")
    preheader = [
        SimpleInstruction('i', 'copy', ['0']),
    ]
    loop_body = [
        SimpleInstruction('x', '*', ['a', 'b']),
        SimpleInstruction('y', '+', ['x', 'c']),
        SimpleInstruction('t1', '+', ['y', 'i']),
        SimpleInstruction('arr[i]', 'store', ['t1']),
        SimpleInstruction('i', '+', ['i', '1']),
    ]

    print("Preheader:")
    for instr in preheader:
        print(f"  {instr}")
    print("Loop body:")
    for instr in loop_body:
        print(f"  {instr}")

    # Identify invariant instructions
    # x = a * b  -- a, b not defined in loop -> invariant
    # y = x + c  -- x is invariant, c not defined in loop -> invariant
    # t1, arr[i], i are NOT invariant (depend on i which changes)

    hoisted = [loop_body[0], loop_body[1]]
    remaining = loop_body[2:]

    print("\n=== After LICM ===")
    print("Preheader:")
    for instr in preheader + hoisted:
        print(f"  {instr}")
    print("Loop body:")
    for instr in remaining:
        print(f"  {instr}")

demonstrate_licm()
```

Output:

```
=== Before LICM ===
Preheader:
  i = 0
Loop body:
  x = a * b
  y = x + c
  t1 = y + i
  arr[i] = store
  i = i + 1

=== After LICM ===
Preheader:
  i = 0
  x = a * b
  y = x + c
Loop body:
  t1 = y + i
  arr[i] = store
  i = i + 1
```

---

## 5. Induction Variable Analysis

### 5.1 Basic Induction Variables

A **basic induction variable** (BIV) is a variable $i$ whose only definitions inside the loop have the form:

$$i = i + c \quad \text{or} \quad i = i - c$$

where $c$ is a loop-invariant quantity. The canonical example is a loop counter.

### 5.2 Derived Induction Variables

A **derived induction variable** (DIV) $j$ is a variable defined as a linear function of a basic induction variable:

$$j = a \cdot i + b$$

where $a$ and $b$ are loop-invariant. We write the **induction triple** as $(i, a, b)$, meaning $j = a \cdot i + b$.

```python
class InductionVariable:
    """Represents an induction variable as a triple (base_iv, multiplier, offset)."""

    def __init__(self, name, base_iv=None, multiplier=1, offset=0):
        self.name = name
        self.base_iv = base_iv or name  # Basic IV it derives from
        self.multiplier = multiplier     # a in j = a*i + b
        self.offset = offset             # b in j = a*i + b
        self.is_basic = (base_iv is None or base_iv == name)

    def value_at(self, i_value):
        """Compute the value of this IV when the base IV has value i_value."""
        return self.multiplier * i_value + self.offset

    def __repr__(self):
        if self.is_basic:
            return f"BIV({self.name})"
        return f"DIV({self.name} = {self.multiplier}*{self.base_iv} + {self.offset})"


def detect_induction_variables(loop_instructions, loop_invariants):
    """
    Detect basic and derived induction variables in a loop.

    loop_instructions: list of (target, op, operands)
    loop_invariants: set of variable names that are loop-invariant

    Returns: dict name -> InductionVariable
    """
    # Phase 1: Find basic induction variables
    # A variable i is a BIV if its only definition is i = i +/- c (c invariant)
    definitions = {}  # var -> list of (op, operands)
    for target, op, operands in loop_instructions:
        if target not in definitions:
            definitions[target] = []
        definitions[target].append((op, operands))

    bivs = {}
    for var, defs in definitions.items():
        if len(defs) == 1:
            op, operands = defs[0]
            if op in ('+', '-') and var in operands:
                other = [o for o in operands if o != var]
                if len(other) == 1 and (other[0] in loop_invariants or
                                         other[0].lstrip('-').isdigit()):
                    step = int(other[0]) if op == '+' else -int(other[0])
                    bivs[var] = InductionVariable(var, multiplier=1, offset=0)
                    bivs[var].step = step

    # Phase 2: Find derived induction variables
    # j = a * i + b where i is BIV and a, b are loop-invariant
    divs = {}
    for target, op, operands in loop_instructions:
        if target in bivs:
            continue

        if op == '*':
            # Check if one operand is a BIV and the other is invariant
            for i, o in enumerate(operands):
                other_idx = 1 - i
                if o in bivs and (operands[other_idx] in loop_invariants or
                                   operands[other_idx].lstrip('-').isdigit()):
                    mult = operands[other_idx]
                    divs[target] = InductionVariable(
                        target, base_iv=o,
                        multiplier=int(mult), offset=0
                    )

        elif op == '+':
            # Check if one operand is a DIV/BIV and the other is invariant
            for i, o in enumerate(operands):
                other_idx = 1 - i
                if o in bivs and (operands[other_idx] in loop_invariants or
                                   operands[other_idx].lstrip('-').isdigit()):
                    off = int(operands[other_idx])
                    divs[target] = InductionVariable(
                        target, base_iv=o,
                        multiplier=1, offset=off
                    )

    result = {}
    result.update(bivs)
    result.update(divs)
    return result


# Example
print("=== Induction Variable Detection ===")
instructions = [
    ('i', '+', ['i', '1']),        # i = i + 1  (BIV)
    ('t1', '*', ['i', '4']),       # t1 = 4*i   (DIV: base=i, mult=4, off=0)
    ('t2', '+', ['t1', 'base']),   # t2 = t1 + base (not simple IV -- base is invariant
                                   #   but t1 is a DIV, so t2 = 4*i + base)
    ('arr_t2', 'load', ['t2']),    # memory load
]

invariants = {'base', '4', '1'}

ivs = detect_induction_variables(instructions, invariants)
for name, iv in ivs.items():
    print(f"  {iv}")
```

---

## 6. Strength Reduction

### 6.1 The Idea

Strength reduction replaces expensive operations (like multiplication) with cheaper ones (like addition) by exploiting the regular progression of induction variables.

If $j = a \cdot i + b$ and $i$ increments by $s$ each iteration, then:

$$j_{\text{new}} = j_{\text{old}} + a \cdot s$$

Instead of computing $a \cdot i$ every iteration, we compute $j = j + a \cdot s$ -- replacing a multiplication with an addition.

### 6.2 The Allen-Cocke-Kennedy Algorithm

The classic strength reduction algorithm works as follows:

1. Find all basic induction variables and their steps
2. Find all derived induction variables
3. For each DIV $j = a \cdot i + b$:
   - Create a new variable $j'$
   - Initialize $j' = a \cdot i_0 + b$ in the preheader
   - At each increment of $i$ by $s$, add $j' = j' + a \cdot s$
   - Replace uses of $j$ with $j'$

```python
def strength_reduce(loop_code, preheader_code, bivs, divs):
    """
    Apply strength reduction to derived induction variables.

    Demonstrates the transformation on a simple example.
    """
    print("=== Before Strength Reduction ===")
    print("Preheader:")
    for line in preheader_code:
        print(f"  {line}")
    print("Loop body:")
    for line in loop_code:
        print(f"  {line}")

    new_preheader = list(preheader_code)
    new_loop = []

    # For each derived IV, create strength-reduced version
    reduced = {}  # original var -> new var

    for div_name, div_info in divs.items():
        base = div_info['base_iv']
        mult = div_info['multiplier']
        offset = div_info['offset']
        step = bivs[base]['step']

        new_var = f"{div_name}_sr"
        reduced[div_name] = new_var

        # Initialize in preheader: new_var = mult * base_init + offset
        init_val = mult * bivs[base]['init'] + offset
        new_preheader.append(f"{new_var} = {init_val}")

        # Increment in loop: new_var = new_var + mult * step
        increment = mult * step
        new_loop.append(f"{new_var} = {new_var} + {increment}")

    # Copy non-reduced instructions
    for line in loop_code:
        # Replace references to reduced variables
        replaced = False
        for orig, new in reduced.items():
            if line.startswith(f"{orig} ="):
                replaced = True  # Skip original computation
                break
            if orig in line:
                line = line.replace(orig, new)
        if not replaced:
            new_loop.append(line)

    print("\n=== After Strength Reduction ===")
    print("Preheader:")
    for line in new_preheader:
        print(f"  {line}")
    print("Loop body:")
    for line in new_loop:
        print(f"  {line}")


# Example: Array indexing
#   for i in range(n):
#       addr = base + i * 4    # addr is a DIV of i
#       arr[addr] = 0

strength_reduce(
    loop_code=[
        "t1 = i * 4",          # DIV: t1 = 4*i
        "addr = t1 + base",    # addr = 4*i + base (also a DIV)
        "store 0, addr",
        "i = i + 1",
    ],
    preheader_code=[
        "i = 0",
    ],
    bivs={
        'i': {'init': 0, 'step': 1}
    },
    divs={
        't1': {'base_iv': 'i', 'multiplier': 4, 'offset': 0},
    }
)
```

Output:

```
=== Before Strength Reduction ===
Preheader:
  i = 0
Loop body:
  t1 = i * 4
  addr = t1 + base
  store 0, addr
  i = i + 1

=== After Strength Reduction ===
Preheader:
  i = 0
  t1_sr = 0
Loop body:
  t1_sr = t1_sr + 4
  addr = t1_sr + base
  store 0, addr
  i = i + 1
```

The multiplication `i * 4` has been replaced by the addition `t1_sr + 4`.

### 6.3 Linear Test Replacement

After strength reduction, the original basic induction variable may only be used in the loop's exit test. We can replace the test with one based on the strength-reduced variable:

```
Before:  if i < n goto loop     After:  if t1_sr < 4*n goto loop
```

This may allow dead code elimination to remove the original BIV entirely.

---

## 7. Loop Unrolling

### 7.1 Full Unrolling

When the trip count (number of iterations) is known at compile time and is small, the loop can be completely replaced by sequential code:

```python
def demonstrate_full_unrolling():
    """Full unrolling: replace loop with sequential statements."""

    print("=== Before Full Unrolling ===")
    print("""
    sum = 0
    for i in range(4):
        sum += arr[i]
    """)

    print("=== After Full Unrolling ===")
    print("""
    sum = 0
    sum += arr[0]
    sum += arr[1]
    sum += arr[2]
    sum += arr[3]
    """)

demonstrate_full_unrolling()
```

Benefits:
- Eliminates loop overhead (branch, counter increment, comparison)
- Enables further optimizations (constant folding on indices, instruction scheduling)

Cost:
- Increases code size
- Only practical for small, known trip counts

### 7.2 Partial Unrolling

More commonly, loops are **partially unrolled** by a factor $k$: each iteration of the new loop does $k$ iterations' worth of work.

For a loop with trip count $n$ and unroll factor $k$:

```python
def demonstrate_partial_unrolling(n=100, k=4):
    """
    Partial unrolling by factor k.

    Original: for i in range(n): body(i)
    Unrolled:
        for i in range(0, n - n%k, k):
            body(i)
            body(i+1)
            body(i+2)
            body(i+3)
        # Cleanup loop for remainder
        for i in range(n - n%k, n):
            body(i)
    """
    print(f"=== Partial Unrolling (factor {k}) ===")

    # Original loop
    total_orig = 0
    iterations_orig = 0
    for i in range(n):
        total_orig += i * i
        iterations_orig += 1

    # Unrolled loop
    total_unrolled = 0
    iterations_unrolled = 0
    main_limit = n - (n % k)

    for i in range(0, main_limit, k):
        total_unrolled += i * i
        total_unrolled += (i + 1) * (i + 1)
        total_unrolled += (i + 2) * (i + 2)
        total_unrolled += (i + 3) * (i + 3)
        iterations_unrolled += 1

    # Cleanup
    for i in range(main_limit, n):
        total_unrolled += i * i
        iterations_unrolled += 1

    print(f"Original:  {iterations_orig} loop iterations, sum = {total_orig}")
    print(f"Unrolled:  {iterations_unrolled} loop iterations, sum = {total_unrolled}")
    print(f"Results match: {total_orig == total_unrolled}")
    print(f"Branch reduction: {iterations_orig - iterations_unrolled} fewer branches")

demonstrate_partial_unrolling()
```

### 7.3 Unrolling and Software Pipelining

Unrolling enables **software pipelining** -- overlapping operations from different iterations to hide latency. On a processor with a 3-cycle multiply and 1-cycle add:

```
Iteration:   Body 1      Body 2      Body 3      Body 4
Cycle 1:     MUL a1      ---         ---         ---
Cycle 2:     ---          MUL a2     ---         ---
Cycle 3:     ---          ---         MUL a3     ---
Cycle 4:     ADD r1       ---         ---         MUL a4
Cycle 5:     ---          ADD r2     ---         ---
...
```

Without unrolling, there would be bubbles (stalls) while waiting for the multiply to complete.

### 7.4 Implementation Considerations

```python
def unroll_loop(original_body, trip_count, unroll_factor):
    """
    Generate an unrolled loop.

    original_body: function(i) -> list of instructions
    trip_count: number of iterations (may be symbolic)
    unroll_factor: k

    Returns: (main_loop, cleanup_loop)
    """
    if isinstance(trip_count, int) and trip_count <= unroll_factor:
        # Full unroll
        full_body = []
        for i in range(trip_count):
            full_body.extend(original_body(i))
        return full_body, []

    # Partial unroll
    main_body = []
    for offset in range(unroll_factor):
        main_body.extend(original_body(f"i+{offset}" if offset > 0 else "i"))

    cleanup_body = original_body("i")

    return main_body, cleanup_body


# Decision heuristics
def should_unroll(loop, unroll_factor=4):
    """
    Heuristic: decide whether to unroll a loop.
    """
    # Don't unroll loops with function calls (code bloat)
    if loop.get('has_calls', False):
        return False

    # Don't unroll loops with many instructions (code bloat)
    if loop.get('body_size', 0) > 20:
        return False

    # Don't unroll deeply nested inner loops (already many iterations)
    if loop.get('nesting_depth', 0) > 3:
        return False

    # Unroll small inner loops
    if loop.get('is_innermost', False) and loop.get('body_size', 0) <= 10:
        return True

    return False
```

---

## 8. Loop Fusion and Fission

### 8.1 Loop Fusion (Jamming)

Loop fusion combines two adjacent loops with the same iteration space into one:

```python
def demonstrate_loop_fusion():
    """Show loop fusion combining two loops into one."""
    import time

    n = 1_000_000
    a = list(range(n))
    b = [0] * n
    c = [0] * n

    # Before fusion: two separate loops
    print("=== Before Fusion (Two Loops) ===")
    start = time.perf_counter()

    for i in range(n):
        b[i] = a[i] * 2

    for i in range(n):
        c[i] = a[i] + 1

    time_separate = time.perf_counter() - start
    print(f"Time: {time_separate:.4f}s")

    # After fusion: one loop
    print("\n=== After Fusion (Single Loop) ===")
    b2 = [0] * n
    c2 = [0] * n
    start = time.perf_counter()

    for i in range(n):
        b2[i] = a[i] * 2
        c2[i] = a[i] + 1

    time_fused = time.perf_counter() - start
    print(f"Time: {time_fused:.4f}s")
    print(f"Speedup: {time_separate / time_fused:.2f}x")
    print(f"Results correct: {b == b2 and c == c2}")

# demonstrate_loop_fusion()  # Uncomment to run
```

**Benefits of fusion**:
- Reduces loop overhead (one loop instead of two)
- Improves data locality (array `a` is accessed once instead of twice)
- Enables further optimizations (common subexpression elimination across merged bodies)

**Legality**: Fusion is legal when there are no data dependencies between the two loops that would be violated by combining them. Specifically, there must be no **fusion-preventing dependency** -- a dependency where an element produced in loop 2 iteration $i$ is consumed by loop 1 iteration $j > i$.

### 8.2 Loop Fission (Distribution)

Loop fission is the inverse: splitting one loop into multiple loops.

```python
def demonstrate_loop_fission():
    """Show loop fission splitting one loop into two."""

    print("=== Before Fission (Single Loop) ===")
    print("""
    for i in range(n):
        b[i] = a[i] * 2        # Vectorizable
        c[i] = c[i-1] + b[i]   # Sequential dependency
    """)

    print("=== After Fission (Two Loops) ===")
    print("""
    # Loop 1: vectorizable
    for i in range(n):
        b[i] = a[i] * 2

    # Loop 2: sequential
    for i in range(n):
        c[i] = c[i-1] + b[i]
    """)
    print("Loop 1 can now be vectorized independently!")
    print("Loop 2 carries a dependency that prevents vectorization.")

demonstrate_loop_fission()
```

**Benefits of fission**:
- Enables vectorization of one part even if another has dependencies
- Improves cache behavior when each loop accesses different data
- Simplifies analysis for further optimization

---

## 9. Loop Tiling (Blocking)

### 9.1 The Cache Problem

When processing a large 2D array column by column, adjacent accesses in memory are far apart (stride $= n$, the row size). This causes **cache thrashing**.

### 9.2 Tiling Transformation

Loop tiling breaks iterations into small blocks ("tiles") that fit in cache:

```python
import time

def demonstrate_loop_tiling(n=1000):
    """
    Demonstrate the effect of loop tiling on matrix multiplication.

    Standard: O(n^3) with poor cache behavior
    Tiled: O(n^3) same work, but much better cache locality
    """
    import random
    random.seed(42)

    # Create matrices
    A = [[random.random() for _ in range(n)] for _ in range(n)]
    B = [[random.random() for _ in range(n)] for _ in range(n)]

    # Standard matrix multiply (ijk order)
    print(f"=== Matrix Multiply {n}x{n} ===")
    C1 = [[0.0] * n for _ in range(n)]

    start = time.perf_counter()
    for i in range(n):
        for j in range(n):
            total = 0.0
            for k in range(n):
                total += A[i][k] * B[k][j]
            C1[i][j] = total
    time_standard = time.perf_counter() - start
    print(f"Standard (ijk): {time_standard:.3f}s")

    # Tiled matrix multiply
    TILE = 32  # Tile size -- should fit in L1 cache
    C2 = [[0.0] * n for _ in range(n)]

    start = time.perf_counter()
    for ii in range(0, n, TILE):
        for jj in range(0, n, TILE):
            for kk in range(0, n, TILE):
                # Multiply tile
                for i in range(ii, min(ii + TILE, n)):
                    for j in range(jj, min(jj + TILE, n)):
                        total = C2[i][j]
                        for k in range(kk, min(kk + TILE, n)):
                            total += A[i][k] * B[k][j]
                        C2[i][j] = total
    time_tiled = time.perf_counter() - start
    print(f"Tiled (tile={TILE}): {time_tiled:.3f}s")
    print(f"Speedup: {time_standard / time_tiled:.2f}x")

    # Verify correctness
    max_diff = max(abs(C1[i][j] - C2[i][j]) for i in range(n) for j in range(n))
    print(f"Max difference: {max_diff:.2e}")

# demonstrate_loop_tiling(500)  # Uncomment to run (may take a while)
```

### 9.3 Choosing the Tile Size

The tile size $T$ should be chosen so that the working set fits in cache. For matrix multiplication, the tiles of $A$, $B$, and $C$ accessed are each $T \times T$, so we need:

$$3 \cdot T^2 \cdot \text{sizeof(element)} \leq \text{L1 cache size}$$

For an L1 cache of 32 KB with 8-byte doubles:

$$T \leq \sqrt{\frac{32768}{3 \times 8}} = \sqrt{1365} \approx 36$$

A common choice is $T = 32$.

### 9.4 Multi-Level Tiling

For machines with multiple cache levels (L1, L2, L3), we can apply tiling at multiple levels:

```
Original:
  for i in range(n):
      for j in range(n):
          C[i][j] += A[i][k] * B[k][j]

Two-level tiled:
  for ii in range(0, n, T2):          # L2 tile
      for jj in range(0, n, T2):
          for kk in range(0, n, T2):
              for iii in range(ii, ii+T2, T1):    # L1 tile
                  for jjj in range(jj, jj+T2, T1):
                      for kkk in range(kk, kk+T2, T1):
                          # Micro-kernel on T1 x T1 tile
                          ...
```

---

## 10. Loop Interchange

### 10.1 Motivation

Loop interchange swaps the order of two nested loops to improve memory access patterns.

```python
def demonstrate_loop_interchange():
    """
    Show how loop interchange improves cache behavior.

    Row-major arrays: A[i][j] is stored next to A[i][j+1] in memory.
    Accessing A[i][j] with j as the inner loop gives stride-1 access (good).
    Accessing with i as the inner loop gives stride-n access (bad).
    """
    import time
    n = 2000

    # Create a 2D array (list of lists in Python, simulating row-major)
    A = [[0.0] * n for _ in range(n)]

    # Bad order: column-major traversal (i inner, j outer)
    print(f"=== Loop Interchange Demo ({n}x{n}) ===")
    start = time.perf_counter()
    for j in range(n):          # Outer
        for i in range(n):      # Inner -- stride-n access to A[i][j]
            A[i][j] = i + j
    time_bad = time.perf_counter() - start
    print(f"Column-major (j outer, i inner): {time_bad:.3f}s")

    # Good order: row-major traversal (j inner, i outer)
    A = [[0.0] * n for _ in range(n)]
    start = time.perf_counter()
    for i in range(n):          # Outer
        for j in range(n):      # Inner -- stride-1 access to A[i][j]
            A[i][j] = i + j
    time_good = time.perf_counter() - start
    print(f"Row-major (i outer, j inner): {time_good:.3f}s")
    print(f"Speedup: {time_bad / time_good:.2f}x")

# demonstrate_loop_interchange()  # Uncomment to run
```

### 10.2 Legality of Loop Interchange

Loop interchange is legal only when it does not violate data dependencies. Given a dependency with distance vector $(d_1, d_2)$ for loops $i$ and $j$:

- The interchange is legal if the resulting distance vector $(d_2, d_1)$ is **lexicographically positive** (the leftmost non-zero component is positive).

```python
def can_interchange(distance_vectors):
    """
    Check if loop interchange is legal given dependency distance vectors.

    A distance vector (d1, d2) becomes (d2, d1) after interchange.
    All resulting vectors must be lexicographically positive.
    """
    for d1, d2 in distance_vectors:
        # After interchange: (d2, d1)
        new_vec = (d2, d1)

        # Check lexicographic positivity
        if new_vec[0] < 0:
            return False
        if new_vec[0] == 0 and new_vec[1] < 0:
            return False

    return True


# Examples
print("Dependency (1, 0):", can_interchange([(1, 0)]))   # True: becomes (0, 1)
print("Dependency (0, 1):", can_interchange([(0, 1)]))   # True: becomes (1, 0)
print("Dependency (1, -1):", can_interchange([(1, -1)])) # False: becomes (-1, 1)
print("Dependency (1, 1):", can_interchange([(1, 1)]))   # True: becomes (1, 1)
```

---

## 11. Vectorization

### 11.1 SIMD Overview

**SIMD** (Single Instruction, Multiple Data) instructions process multiple data elements simultaneously. Modern CPUs have SIMD units (SSE, AVX, AVX-512 on x86; NEON on ARM) that can operate on 128, 256, or 512 bits at once.

```
Scalar:                    SIMD (4-wide):
  a[0] + b[0] = c[0]        a[0:4] + b[0:4] = c[0:4]
  a[1] + b[1] = c[1]        (single instruction)
  a[2] + b[2] = c[2]
  a[3] + b[3] = c[3]
  (4 instructions)
```

With 256-bit AVX and 32-bit floats, one instruction processes $256 / 32 = 8$ elements.

### 11.2 Auto-Vectorization

Compilers attempt to automatically vectorize loops. The key requirement is **no loop-carried dependencies** -- each iteration must be independent.

```python
def vectorization_analysis():
    """Analyze loops for vectorizability."""

    examples = [
        {
            'name': 'Simple element-wise',
            'code': 'c[i] = a[i] + b[i]',
            'vectorizable': True,
            'reason': 'No loop-carried dependency'
        },
        {
            'name': 'Reduction',
            'code': 'sum += a[i]',
            'vectorizable': True,
            'reason': 'Reduction pattern -- compiler uses horizontal add'
        },
        {
            'name': 'Prefix sum',
            'code': 'a[i] = a[i-1] + b[i]',
            'vectorizable': False,
            'reason': 'Loop-carried dependency: a[i] depends on a[i-1]'
        },
        {
            'name': 'Conditional',
            'code': 'if a[i] > 0: b[i] = a[i]',
            'vectorizable': True,
            'reason': 'Predicated execution with masking'
        },
        {
            'name': 'Indirect access',
            'code': 'b[idx[i]] = a[i]',
            'vectorizable': False,  # Usually
            'reason': 'Scatter -- possible with AVX-512 but slow'
        },
        {
            'name': 'Function call',
            'code': 'b[i] = expensive_func(a[i])',
            'vectorizable': False,
            'reason': 'Function calls prevent vectorization (unless intrinsic)'
        },
    ]

    print("=== Vectorization Analysis ===")
    for ex in examples:
        status = "YES" if ex['vectorizable'] else "NO"
        print(f"\n  {ex['name']}: {ex['code']}")
        print(f"    Vectorizable: {status}")
        print(f"    Reason: {ex['reason']}")

vectorization_analysis()
```

### 11.3 Vectorization with NumPy

In Python, NumPy provides vectorized operations that leverage SIMD under the hood:

```python
import numpy as np
import time

def numpy_vectorization_demo():
    """Demonstrate NumPy vectorization vs pure Python loops."""
    n = 10_000_000

    # Pure Python
    a = list(range(n))
    b = list(range(n))

    start = time.perf_counter()
    c_python = [a[i] + b[i] for i in range(n)]
    time_python = time.perf_counter() - start

    # NumPy vectorized
    a_np = np.arange(n, dtype=np.float64)
    b_np = np.arange(n, dtype=np.float64)

    start = time.perf_counter()
    c_np = a_np + b_np  # Single vectorized operation
    time_numpy = time.perf_counter() - start

    print(f"Python loop: {time_python:.3f}s")
    print(f"NumPy vectorized: {time_numpy:.3f}s")
    print(f"Speedup: {time_python / time_numpy:.1f}x")

# numpy_vectorization_demo()  # Uncomment to run
```

### 11.4 Strip Mining

Strip mining transforms a loop to make the vector length explicit, preparing it for SIMD:

```python
def strip_mine(n, vector_length=4):
    """
    Strip mine a loop for vectorization.

    Before: for i in range(n): body(i)
    After:  for i in range(0, n, VL):
                for ii in range(i, min(i+VL, n)):
                    body(ii)
    """
    print(f"=== Strip Mining (VL={vector_length}) ===")
    print(f"Original: for i in range({n}): body(i)")
    print(f"Strip-mined:")

    iteration_count = 0
    for i in range(0, n, vector_length):
        end = min(i + vector_length, n)
        vec_indices = list(range(i, end))
        print(f"  Vector op: body({vec_indices})")
        iteration_count += 1

    print(f"Outer iterations: {iteration_count} (was {n})")

strip_mine(14, 4)
```

Output:

```
=== Strip Mining (VL=4) ===
Original: for i in range(14): body(i)
Strip-mined:
  Vector op: body([0, 1, 2, 3])
  Vector op: body([4, 5, 6, 7])
  Vector op: body([8, 9, 10, 11])
  Vector op: body([12, 13])
Outer iterations: 4 (was 14)
```

### 11.5 Alignment and Peeling

SIMD instructions often require data to be aligned to specific boundaries (16-byte for SSE, 32-byte for AVX). **Loop peeling** handles the first few iterations (until alignment is reached) as scalar operations:

```
Peeled iterations:      i = 0, 1  (scalar, until aligned)
Main vectorized loop:   i = 2, 3, 4, 5 | 6, 7, 8, 9 | ...  (SIMD)
Cleanup iterations:     i = n-2, n-1  (scalar, remainder)
```

---

## 12. The Polyhedral Model

### 12.1 Overview

The **polyhedral model** (also called the **polytope model**) is a powerful mathematical framework for loop nest optimization. It represents loop iterations as integer points in a polyhedron and loop transformations as affine functions on these points.

### 12.2 Iteration Domains

Consider a doubly nested loop:

```python
for i in range(0, N):
    for j in range(0, M):
        A[i][j] = B[i][j-1] + B[i-1][j]
```

The **iteration domain** is the set of all $(i, j)$ pairs that execute:

$$\mathcal{D} = \{ (i, j) \in \mathbb{Z}^2 \mid 0 \leq i < N, \; 0 \leq j < M \}$$

This is a 2D rectangular polyhedron (a polytope).

### 12.3 Access Functions

Each memory access is described by an **access function** -- an affine mapping from iteration coordinates to array indices:

- `A[i][j]`: access function $f_A(i, j) = (i, j)$
- `B[i][j-1]`: access function $f_{B1}(i, j) = (i, j-1)$
- `B[i-1][j]`: access function $f_{B2}(i, j) = (i-1, j)$

### 12.4 Dependence Polyhedra

Dependencies between statement instances are captured by **dependence polyhedra**. A dependency from iteration $(i_1, j_1)$ to $(i_2, j_2)$ exists when they access the same memory location and the source executes before the sink.

For the read `B[i][j-1]` and the write that produced it:

$$i_2 = i_1, \quad j_2 - 1 = j_1 \implies j_2 = j_1 + 1$$

The dependency is $(i, j) \to (i, j+1)$ with distance vector $(0, 1)$.

### 12.5 Schedule and Transformation

A **schedule** maps each iteration point to a time step. A valid schedule must respect all dependencies. The schedule is often represented as an affine function:

$$\theta(i, j) = \alpha_1 i + \alpha_2 j + \alpha_0$$

The polyhedral model can find optimal schedules by solving an integer linear program (ILP).

```python
def polyhedral_example():
    """
    Simple demonstration of polyhedral iteration domain and scheduling.
    """
    N, M = 4, 4

    print("=== Polyhedral Model Example ===")
    print(f"Iteration domain: 0 <= i < {N}, 0 <= j < {M}")
    print(f"Statement: A[i][j] = B[i][j-1] + B[i-1][j]")
    print(f"Dependencies: (0,1) and (1,0)")

    # Original schedule: theta(i,j) = (i, j) -- lexicographic order
    print("\nOriginal schedule (row by row):")
    for i in range(N):
        for j in range(M):
            print(f"  t={i*M + j}: S({i},{j})")

    # Skewed schedule: theta(i,j) = (i+j, j) -- enables wavefront parallelism
    print("\nSkewed schedule (wavefront):")
    schedule = []
    for i in range(N):
        for j in range(M):
            time = (i + j, j)
            schedule.append((time, i, j))

    schedule.sort()

    for time, i, j in schedule:
        print(f"  t={time}: S({i},{j})")

    print("\nWavefront parallelism:")
    print("Iterations on the same diagonal (same i+j) can execute in parallel!")
    for wave in range(N + M - 1):
        parallel = [(i, wave - i) for i in range(N) if 0 <= wave - i < M]
        print(f"  Wave {wave}: {parallel}")

polyhedral_example()
```

### 12.6 Tools

Real polyhedral optimizers include:
- **ISL** (Integer Set Library): The mathematical foundation
- **Pluto**: Automatic parallelizer and locality optimizer
- **Polly**: LLVM's polyhedral optimizer pass
- **PPCG**: Polyhedral Parallel Code Generator (for GPUs)

---

## 13. Loop Parallelization

### 13.1 Dependence-Free Loops

A loop is **trivially parallelizable** (also called "embarrassingly parallel") if there are no loop-carried dependencies:

```python
# Parallelizable: each iteration is independent
for i in range(n):
    c[i] = a[i] + b[i]

# NOT parallelizable: iteration i depends on i-1
for i in range(1, n):
    a[i] = a[i-1] + b[i]
```

### 13.2 Types of Dependencies

Three types of loop-carried dependencies:

| Type | Name | Example | Blocks Parallelization? |
|------|------|---------|------------------------|
| RAW | True / Flow | `a[i] = ...; ... = a[i-1]` | Yes |
| WAR | Anti | `... = a[i-1]; a[i] = ...` | Renamable |
| WAW | Output | `a[i] = ...; a[i] = ...` | Renamable |

WAR and WAW dependencies can be eliminated by **privatization** (giving each thread its own copy) or renaming.

### 13.3 Reduction Parallelization

Reductions (sum, product, min, max) have a special pattern that allows parallel execution:

```python
# Sequential reduction
total = 0
for i in range(n):
    total += a[i]  # RAW dependency on total

# Parallel reduction
# Each thread computes a partial sum, then combine
from concurrent.futures import ThreadPoolExecutor

def parallel_sum(arr, num_threads=4):
    """Parallel reduction using thread pool."""
    n = len(arr)
    chunk_size = (n + num_threads - 1) // num_threads

    def partial_sum(start, end):
        return sum(arr[start:end])

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        for t in range(num_threads):
            start = t * chunk_size
            end = min(start + chunk_size, n)
            futures.append(executor.submit(partial_sum, start, end))

        total = sum(f.result() for f in futures)

    return total
```

### 13.4 DOALL, DOACROSS, and DOPIPE

Three parallelization strategies:

**DOALL**: All iterations are independent -- run them all in parallel.

```python
# DOALL parallelism
# for i in range(n):
#     c[i] = f(a[i])
# All iterations can execute simultaneously
```

**DOACROSS**: Iterations have dependencies, but can overlap execution with synchronization.

```python
# DOACROSS parallelism
# for i in range(n):
#     a[i] = a[i-1] + b[i]    # depends on previous iteration
#     c[i] = expensive(a[i])   # expensive independent computation
#
# Thread 1 computes a[0], starts expensive(a[0])
# Thread 2 waits for a[0], then computes a[1], starts expensive(a[1])
# Overlap the expensive computation
```

**DOPIPE**: Pipeline parallelism -- each stage of computation runs on a different thread.

```python
# DOPIPE parallelism
# Stage 1 (Thread 1): for i: a[i] = input[i] * 2
# Stage 2 (Thread 2): for i: b[i] = a[i] + offset  (wait for a[i] from Stage 1)
# Stage 3 (Thread 3): for i: output[i] = transform(b[i])
```

### 13.5 OpenMP-Style Parallelization

Compilers that support OpenMP can parallelize loops with pragmas. The equivalent in Python:

```python
from multiprocessing import Pool

def openmp_style_parallel():
    """Demonstrate parallel loop execution (Python equivalent of OpenMP)."""
    import os

    n = 1_000_000
    a = list(range(n))
    num_workers = os.cpu_count() or 4

    def process_chunk(args):
        start, end, data = args
        return [x * x + 2 * x + 1 for x in data[start:end]]

    chunk_size = (n + num_workers - 1) // num_workers
    chunks = [
        (i * chunk_size, min((i + 1) * chunk_size, n), a)
        for i in range(num_workers)
    ]

    with Pool(num_workers) as pool:
        results = pool.map(process_chunk, chunks)

    # Flatten results
    c = []
    for chunk_result in results:
        c.extend(chunk_result)

    print(f"Processed {len(c)} elements using {num_workers} workers")
    print(f"First 5: {c[:5]}")
    print(f"Last 5: {c[-5:]}")

# openmp_style_parallel()  # Uncomment to run
```

### 13.6 Compiler Analysis for Parallelization

```python
def analyze_parallelizability(loop_deps):
    """
    Analyze whether a loop can be parallelized and suggest strategy.

    loop_deps: list of dependency dicts with 'type', 'distance', 'variable'
    """
    if not loop_deps:
        return "DOALL", "No dependencies -- fully parallel"

    has_true_dep = any(d['type'] == 'RAW' for d in loop_deps)
    has_reduction = any(d.get('is_reduction', False) for d in loop_deps)
    only_renamable = all(d['type'] in ('WAR', 'WAW') for d in loop_deps)

    if has_reduction and not has_true_dep:
        return "DOALL + Reduction", "Reduction pattern detected -- use parallel reduction"

    if only_renamable:
        return "DOALL (after privatization)", "Only anti/output deps -- privatize variables"

    if has_true_dep:
        min_distance = min(d['distance'] for d in loop_deps if d['type'] == 'RAW')
        if min_distance > 1:
            return "DOACROSS", f"True dep with distance {min_distance} -- overlap possible"
        return "Sequential", "True dependency with distance 1 -- cannot parallelize"

    return "DOALL", "Safe to parallelize"


# Examples
examples = [
    ("c[i] = a[i] + b[i]", []),
    ("sum += a[i]", [{'type': 'RAW', 'distance': 1, 'variable': 'sum', 'is_reduction': True}]),
    ("a[i] = a[i-1] + 1", [{'type': 'RAW', 'distance': 1, 'variable': 'a'}]),
    ("a[i] = a[i-4] + 1", [{'type': 'RAW', 'distance': 4, 'variable': 'a'}]),
]

print("=== Parallelizability Analysis ===")
for code, deps in examples:
    strategy, reason = analyze_parallelizability(deps)
    print(f"\n  Code: {code}")
    print(f"  Strategy: {strategy}")
    print(f"  Reason: {reason}")
```

---

## 14. Summary

Loop optimization is the most impactful category of compiler optimization because programs spend the overwhelming majority of their time in loops. We covered:

| Optimization | Effect | Key Requirement |
|-------------|--------|-----------------|
| **Dominator analysis** | Foundation for loop detection | CFG available |
| **Loop detection** | Identifies natural loops via back edges | Dominators computed |
| **LICM** | Hoists invariant code out of loops | Dominance + liveness |
| **Strength reduction** | Replaces multiply with add in IVs | Induction variable analysis |
| **Loop unrolling** | Reduces branch overhead, enables ILP | Trip count knowledge |
| **Loop fusion** | Improves locality, reduces overhead | No fusion-preventing deps |
| **Loop fission** | Enables vectorization of subloops | Dependency partitioning |
| **Loop tiling** | Fits working set in cache | Known bounds |
| **Loop interchange** | Matches access pattern to memory layout | Legal distance vectors |
| **Vectorization** | Exploits SIMD hardware | No loop-carried deps |
| **Polyhedral model** | Unified framework for all transforms | Affine loops |
| **Parallelization** | Exploits multiple cores | Dependency analysis |

The key insight unifying all these optimizations is **dependency analysis**: every transformation must preserve the original program's data dependencies. Understanding what computations depend on what -- and what can be safely reordered, hoisted, or parallelized -- is the foundation of all loop optimization.

---

## 15. Exercises

### Exercise 1: Dominator Computation

Given the following CFG, compute the dominator sets for each node:

```
Entry -> A -> B -> C
         |         |
         v         v
         D -> E -> F -> Exit
              ^
              |
              G <-- F
```

Edges: Entry->A, A->B, A->D, B->C, C->F, D->E, E->F, F->Exit, F->G, G->E

(a) Compute the dominator set for each node.
(b) Draw the dominator tree.
(c) Identify all back edges and natural loops.

### Exercise 2: Loop-Invariant Code Motion

Consider the following loop (in pseudo-code):

```
i = 0
while i < n:
    t1 = a + b
    t2 = t1 * c
    t3 = d[i] + t2
    t4 = i * t1
    e[i] = t3 + t4
    i = i + 1
```

(a) Identify all loop-invariant computations.
(b) Which invariant computations can be safely hoisted? Why or why not?
(c) Write the optimized code after applying LICM.

### Exercise 3: Strength Reduction

Given the following loop:

```python
for i in range(0, N):
    addr1 = base1 + i * 8
    addr2 = base2 + i * 12
    mem[addr1] = mem[addr2]
```

(a) Identify all basic and derived induction variables.
(b) Apply strength reduction. Show the resulting code.
(c) Apply linear test replacement to eliminate the original counter if possible.

### Exercise 4: Loop Tiling

Consider a matrix transposition:

```python
for i in range(N):
    for j in range(N):
        B[j][i] = A[i][j]
```

(a) Analyze the cache behavior of this code (assume row-major storage).
(b) Apply loop tiling with tile size $T = 32$. Write the tiled code.
(c) For an L1 cache of 32 KB and 8-byte elements, what is the optimal tile size?

### Exercise 5: Vectorization Analysis

For each loop below, determine whether it can be vectorized. If not, explain why and suggest a transformation that might help:

```python
# (a)
for i in range(n):
    a[i] = b[i] * c[i] + d[i]

# (b)
for i in range(1, n):
    a[i] = a[i-1] + b[i]

# (c)
for i in range(n):
    if a[i] > 0:
        b[i] = a[i] * 2
    else:
        b[i] = 0

# (d)
for i in range(0, n, 2):
    a[i] = b[i] + c[i]
    a[i+1] = b[i+1] - c[i+1]
```

### Exercise 6: Polyhedral Transformation

Consider the following loop nest:

```python
for i in range(1, N):
    for j in range(1, M):
        A[i][j] = A[i-1][j] + A[i][j-1]
```

(a) Draw the iteration domain as a 2D grid.
(b) Identify all data dependencies and their distance vectors.
(c) Is loop interchange legal? Justify your answer.
(d) Propose a skewing transformation $\theta(i, j) = (i + j, j)$ and verify it respects dependencies.
(e) How does the skewed schedule enable wavefront parallelism?

---

## 16. References

1. Aho, A. V., Lam, M. S., Sethi, R., & Ullman, J. D. (2006). *Compilers: Principles, Techniques, and Tools* (2nd ed.), Chapters 9-10.
2. Cooper, K. D., & Torczon, L. (2011). *Engineering a Compiler* (2nd ed.), Chapters 8-10.
3. Muchnick, S. S. (1997). *Advanced Compiler Design and Implementation*, Chapters 14-18.
4. Allen, R., & Kennedy, K. (2001). *Optimizing Compilers for Modern Architectures*.
5. Wolfe, M. (1996). *High Performance Compilers for Parallel Computing*.
6. Bondhugula, U., et al. (2008). "A Practical Automatic Polyhedral Parallelizer and Locality Optimizer." *PLDI*.
7. Cooper, K. D., Harvey, T. J., & Kennedy, K. (2001). "A Simple, Fast Dominance Algorithm." *Software Practice and Experience*.

---

[Previous: 12. Optimization -- Local and Global](./12_Optimization_Local_and_Global.md) | [Next: 14. Garbage Collection](./14_Garbage_Collection.md) | [Overview](./00_Overview.md)
