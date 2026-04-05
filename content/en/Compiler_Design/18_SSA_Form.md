# Static Single Assignment Form

**Previous**: [16. Modern Compiler Infrastructure](./16_Modern_Compiler_Infrastructure.md) | **Next**: [19. Register Allocation](./19_Register_Allocation.md)

---

Static Single Assignment (SSA) form is the most important intermediate representation in modern optimizing compilers. In SSA form, every variable is assigned exactly once, and every use of a variable refers to exactly one definition. This property dramatically simplifies many compiler analyses and optimizations.

LLVM IR, GCC's GIMPLE, HotSpot's Sea-of-Nodes, and virtually every modern compiler uses SSA as its central IR. This lesson covers the theory behind SSA construction, the role of phi functions, dominance relationships, and how SSA enables powerful optimizations.

**Difficulty**: ⭐⭐⭐⭐

**Prerequisites**: [09. Intermediate Representations](./09_Intermediate_Representations.md), [12. Optimization -- Local and Global](./12_Optimization_Local_and_Global.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain the single-assignment property and why it simplifies optimization
2. Convert conventional code into SSA form using phi functions
3. Compute dominance trees and dominance frontiers
4. Implement the Cytron et al. SSA construction algorithm
5. Perform optimizations on SSA form (constant propagation, dead code elimination)
6. Convert out of SSA form for code generation

---

## Table of Contents

1. [What is SSA Form?](#1-what-is-ssa-form)
2. [Phi Functions](#2-phi-functions)
3. [Dominance](#3-dominance)
4. [Dominance Frontiers](#4-dominance-frontiers)
5. [SSA Construction Algorithm](#5-ssa-construction-algorithm)
6. [Optimizations on SSA](#6-optimizations-on-ssa)
7. [SSA Destruction](#7-ssa-destruction)
8. [Pruned and Semi-Pruned SSA](#8-pruned-and-semi-pruned-ssa)
9. [Summary](#9-summary)
10. [Exercises](#10-exercises)
11. [References](#11-references)

---

## 1. What is SSA Form?

### 1.1 The Single-Assignment Property

In conventional three-address code, a variable can be assigned multiple times:

```
x = 1
x = x + 1
y = x * 2
x = y + x
```

In SSA form, each assignment creates a new version of the variable:

```
x1 = 1
x2 = x1 + 1
y1 = x2 * 2
x3 = y1 + x2
```

Every variable name appears on the left-hand side of exactly one assignment. This means the def-use chain is trivially available: for any use of `x2`, we know immediately that it was defined by `x2 = x1 + 1`.

### 1.2 Why SSA Matters

Without SSA, reaching definitions analysis requires iterative data-flow equations. With SSA, key information is encoded directly in the IR:

| Property | Without SSA | With SSA |
|----------|------------|----------|
| Which definition reaches a use? | Requires reaching-definitions analysis | Immediate from variable name |
| Is a variable live? | Requires liveness analysis | Check if any use exists |
| Constant propagation | Iterative worklist on CFG | Simple worklist on SSA graph |
| Dead code elimination | Requires liveness + reaching defs | Check use-count = 0 |

### 1.3 SSA in a Straight-Line Program

Converting straight-line code is straightforward -- just rename:

```python
# Original
a = 5
b = a + 3
a = b * 2
c = a + b

# SSA
a1 = 5
b1 = a1 + 3
a2 = b1 * 2
c1 = a2 + b1
```

The challenge arises at **control-flow merge points**, where a variable could come from different definitions along different paths.

---

## 2. Phi Functions

### 2.1 The Merge Problem

Consider this control-flow graph:

```
        ┌──── B1 ────┐
        │  x = 1      │
        │  if cond     │
        └──┬───────┬──┘
           │       │
     ┌──── B2 ──┐  ┌──── B3 ──┐
     │  x = 2   │  │  x = 3   │
     └────┬─────┘  └────┬─────┘
           │              │
        ┌──┴──── B4 ──────┴──┐
        │  y = x + 1         │
        └────────────────────┘
```

At block B4, which definition of `x` reaches the use? It depends on the runtime path. SSA introduces a **phi function** (written as the Greek letter phi) to handle this:

```
B4:
  x3 = phi(x1:B2, x2:B3)    // x3 gets x1 if from B2, x2 if from B3
  y1 = x3 + 1
```

### 2.2 Phi Function Semantics

A phi function `x = phi(x1:B1, x2:B2, ..., xn:Bn)` means:

- The value of `x` is `xi` if control flows from block `Bi`
- Phi functions execute "simultaneously" at the beginning of a block
- They are not real instructions -- they guide register allocation and are removed before code generation

```python
# Phi function as a conceptual select
def phi_semantics(predecessors, values):
    """
    phi(x1:B1, x2:B2) means:
    - if we came from B1, the value is x1
    - if we came from B2, the value is x2
    """
    for pred, val in zip(predecessors, values):
        if came_from(pred):
            return val
```

### 2.3 Multiple Phi Functions

A block can have multiple phi functions:

```
B4:
  x3 = phi(x1:B2, x2:B3)
  y3 = phi(y1:B2, y2:B3)
  z1 = x3 + y3
```

All phi functions in a block conceptually execute simultaneously before any other instruction in the block.

### 2.4 Phi Functions in Loops

Loops create phi functions at the loop header:

```
// Original:
// x = 0
// while x < 10:
//     x = x + 1

B1: x1 = 0
    goto B2

B2: x2 = phi(x1:B1, x3:B3)    // merge initial and loop-back
    if x2 < 10 goto B3 else B4

B3: x3 = x2 + 1
    goto B2

B4: // x2 is the final value
```

---

## 3. Dominance

### 3.1 Definition

A node **d dominates** a node **n** (written d dom n) if every path from the entry node to n must pass through d. Every node dominates itself.

A node **d strictly dominates** n (d sdom n) if d dom n and d ≠ n.

The **immediate dominator** (idom) of n is the closest strict dominator -- the unique node d such that d sdom n, and for every other d' that sdom n, we have d' dom d.

### 3.2 Dominator Trees

The immediate dominator relation forms a tree rooted at the entry node:

```
CFG:                    Dominator Tree:
    ┌─ B1 ─┐                B1
    ▼      ▼               / \
   B2     B3              B2  B3
    │      │              |
    ▼      │              B4
   B4 ◀───┘
    │
    ▼
   B5                     B5
```

In this tree:
- B1 is the root (entry)
- B1 immediately dominates B2 and B3
- B2 immediately dominates B4 (assuming B4 requires going through B2 on all paths)
- B4 immediately dominates B5

### 3.3 Computing Dominators

The classic algorithm by Cooper, Harvey, and Kennedy (2001) is simple and efficient:

```python
def compute_dominators(cfg, entry):
    """
    Compute immediate dominators using the iterative algorithm.
    cfg: dict mapping block -> list of predecessors
    entry: entry block name
    Returns: dict mapping block -> immediate dominator
    """
    blocks = list(cfg.keys())
    idom = {b: None for b in blocks}
    idom[entry] = entry

    # Assign reverse postorder numbers
    rpo = reverse_postorder(cfg, entry)
    rpo_number = {b: i for i, b in enumerate(rpo)}

    def intersect(b1, b2):
        """Find common dominator of b1 and b2."""
        finger1, finger2 = b1, b2
        while finger1 != finger2:
            while rpo_number[finger1] > rpo_number[finger2]:
                finger1 = idom[finger1]
            while rpo_number[finger2] > rpo_number[finger1]:
                finger2 = idom[finger2]
        return finger1

    changed = True
    while changed:
        changed = False
        for b in rpo:
            if b == entry:
                continue
            # Pick first processed predecessor as initial idom
            preds = [p for p in cfg[b] if idom[p] is not None]
            if not preds:
                continue
            new_idom = preds[0]
            for p in preds[1:]:
                new_idom = intersect(new_idom, p)
            if idom[b] != new_idom:
                idom[b] = new_idom
                changed = True

    return idom


def reverse_postorder(cfg, entry):
    """Compute reverse postorder traversal of CFG."""
    visited = set()
    order = []

    def dfs(node):
        visited.add(node)
        for succ in get_successors(cfg, node):
            if succ not in visited:
                dfs(succ)
        order.append(node)

    dfs(entry)
    return list(reversed(order))
```

### 3.4 Dominator Tree Properties

Key properties used in SSA construction:

1. **If d dom n, then d appears on every path from entry to n**
2. **The dominator relation is transitive**: if a dom b and b dom c, then a dom c
3. **The dominator tree is unique for a given CFG**
4. **In a loop, the loop header dominates all blocks in the loop body**

---

## 4. Dominance Frontiers

### 4.1 Definition

The **dominance frontier** of a node d, written DF(d), is the set of nodes n such that:
- d dominates a predecessor of n, but
- d does not strictly dominate n

Intuitively, the dominance frontier of d is where d's "dominance ends" -- the first nodes reachable from d that d does not dominate.

```
        B1 (entry)
       / \
      B2   B3
       \  /
        B4
```

- DF(B2) = {B4} -- B2 dominates a predecessor of B4 (namely B2 itself), but B2 does not strictly dominate B4
- DF(B3) = {B4} -- similarly
- DF(B1) = {} -- B1 dominates everything

### 4.2 Why Dominance Frontiers Matter for SSA

**Key insight**: Phi functions are needed exactly at dominance frontier nodes.

If block B defines variable x, then a phi function for x is needed at every node in DF(B). This is because DF(B) contains exactly the merge points where an alternative definition of x might "join" with the one from B.

### 4.3 Computing Dominance Frontiers

```python
def compute_dominance_frontiers(cfg, idom):
    """
    Compute dominance frontier for each block.
    Uses the algorithm from Cytron et al. (1991).
    """
    df = {b: set() for b in cfg}

    for b in cfg:
        preds = cfg[b]  # predecessors of b
        if len(preds) >= 2:  # b is a merge point
            for p in preds:
                runner = p
                while runner != idom[b]:
                    df[runner].add(b)
                    runner = idom[runner]

    return df
```

### 4.4 Iterated Dominance Frontier

When placing phi functions, we need the **iterated dominance frontier** (IDF). A phi function itself is a new definition, which may require additional phi functions at its own dominance frontier:

```python
def iterated_dominance_frontier(df, defs):
    """
    Compute IDF for a set of defining blocks.
    defs: set of blocks that define a variable
    """
    worklist = list(defs)
    idf = set()

    while worklist:
        b = worklist.pop()
        for frontier_node in df[b]:
            if frontier_node not in idf:
                idf.add(frontier_node)
                worklist.append(frontier_node)

    return idf
```

---

## 5. SSA Construction Algorithm

### 5.1 The Cytron et al. Algorithm

The standard SSA construction algorithm (Cytron, Ferrante, Rosen, Wegman, Zadeck, 1991) has two phases:

**Phase 1: Place phi functions** using dominance frontiers.
**Phase 2: Rename variables** using a dominator tree walk.

### 5.2 Phase 1: Phi Function Placement

```python
def place_phi_functions(cfg, idom, df, var_defs):
    """
    Place phi functions for each variable.
    var_defs: dict mapping variable -> set of blocks that define it
    Returns: dict mapping (block, variable) -> phi function
    """
    phi_functions = {}

    for var, def_blocks in var_defs.items():
        # Compute iterated dominance frontier
        idf = iterated_dominance_frontier(df, def_blocks)

        for block in idf:
            preds = cfg[block]
            phi_functions[(block, var)] = {
                'var': var,
                'args': {pred: None for pred in preds}  # filled in Phase 2
            }

    return phi_functions
```

### 5.3 Phase 2: Variable Renaming

```python
def rename_variables(cfg, idom, phi_functions, blocks_instructions):
    """
    Rename variables to create SSA form.
    Walk the dominator tree, maintaining a stack of names for each variable.
    """
    counter = {}       # variable -> next version number
    stack = {}         # variable -> stack of current SSA names

    def new_name(var):
        """Generate a fresh SSA name for var."""
        if var not in counter:
            counter[var] = 0
            stack[var] = []
        i = counter[var]
        counter[var] = i + 1
        name = f"{var}_{i}"
        stack[var].append(name)
        return name

    def current_name(var):
        """Get the current SSA name for var."""
        if var not in stack or not stack[var]:
            return f"{var}_undef"
        return stack[var][-1]

    def rename_block(block):
        # Track how many names we push (to pop later)
        push_count = {var: 0 for var in stack}

        # 1. Rename phi function destinations
        for var in get_phi_vars(phi_functions, block):
            name = new_name(var)
            phi_functions[(block, var)]['dest'] = name
            push_count.setdefault(var, 0)
            push_count[var] = push_count.get(var, 0) + 1

        # 2. Rename instructions
        for instr in blocks_instructions[block]:
            # Rename uses (right-hand side)
            for use_var in instr.uses:
                instr.replace_use(use_var, current_name(use_var))
            # Rename definition (left-hand side)
            if instr.defines:
                old_def = instr.defines
                new_def = new_name(old_def)
                instr.defines = new_def
                push_count.setdefault(old_def, 0)
                push_count[old_def] = push_count.get(old_def, 0) + 1

        # 3. Fill phi function arguments in successors
        for succ in get_successors(cfg, block):
            for var in get_phi_vars(phi_functions, succ):
                phi_functions[(succ, var)]['args'][block] = current_name(var)

        # 4. Recurse into dominator tree children
        for child in dominator_tree_children(idom, block):
            rename_block(child)

        # 5. Pop names
        for var, count in push_count.items():
            for _ in range(count):
                if stack.get(var):
                    stack[var].pop()

    rename_block(entry_block(cfg))
```

### 5.4 Complete Example

```python
# Original program:
#   a = read()
#   b = read()
#   if a > b:
#       c = a - b
#   else:
#       c = b - a
#   print(c)

# CFG:
# B1: a = read(); b = read(); if a > b goto B2 else B3
# B2: c = a - b; goto B4
# B3: c = b - a; goto B4
# B4: print(c)

# Dominance: idom(B2)=B1, idom(B3)=B1, idom(B4)=B1
# DF(B2) = {B4}, DF(B3) = {B4}
# Variable c defined in B2, B3 -> phi for c at B4

# SSA form:
# B1: a1 = read(); b1 = read(); if a1 > b1 goto B2 else B3
# B2: c1 = a1 - b1; goto B4
# B3: c2 = b1 - a1; goto B4
# B4: c3 = phi(c1:B2, c2:B3); print(c3)
```

---

## 6. Optimizations on SSA

### 6.1 Sparse Conditional Constant Propagation (SCCP)

SSA enables a powerful combination of constant propagation and unreachable code detection:

```python
class SCCPSolver:
    """
    Sparse Conditional Constant Propagation on SSA form.
    Uses a lattice: TOP -> constant -> BOTTOM
    """
    TOP = "TOP"         # undefined / not yet analyzed
    BOTTOM = "BOTTOM"   # multiple possible values

    def __init__(self, ssa_program):
        self.program = ssa_program
        self.lattice = {}       # variable -> lattice value
        self.cfg_worklist = []  # edges to process
        self.ssa_worklist = []  # SSA edges (def->use) to process
        self.executable = set() # executable CFG edges

    def meet(self, a, b):
        """Lattice meet operation."""
        if a == self.TOP:
            return b
        if b == self.TOP:
            return a
        if a == b:
            return a
        return self.BOTTOM

    def evaluate(self, instr):
        """Evaluate instruction given current lattice values."""
        if instr.is_phi:
            result = self.TOP
            for pred, var in instr.phi_args.items():
                edge = (pred, instr.block)
                if edge in self.executable:
                    result = self.meet(result, self.lattice.get(var, self.TOP))
            return result
        # ... evaluate arithmetic, comparisons, etc.

    def solve(self):
        """Run SCCP to fixpoint."""
        # Initialize: mark entry edge executable
        self.cfg_worklist.append(("entry", self.program.entry))

        while self.cfg_worklist or self.ssa_worklist:
            while self.cfg_worklist:
                edge = self.cfg_worklist.pop()
                if edge not in self.executable:
                    self.executable.add(edge)
                    target = edge[1]
                    # Process phi functions in target
                    for phi in self.program.get_phis(target):
                        new_val = self.evaluate(phi)
                        if new_val != self.lattice.get(phi.dest, self.TOP):
                            self.lattice[phi.dest] = new_val
                            self.ssa_worklist.extend(phi.dest_uses())

            while self.ssa_worklist:
                use_instr = self.ssa_worklist.pop()
                new_val = self.evaluate(use_instr)
                if new_val != self.lattice.get(use_instr.dest, self.TOP):
                    self.lattice[use_instr.dest] = new_val
                    self.ssa_worklist.extend(use_instr.dest_uses())
```

### 6.2 Global Value Numbering (GVN)

SSA makes it easy to detect redundant computations:

```python
def global_value_numbering(ssa_block):
    """
    Assign value numbers to SSA variables.
    If two expressions have the same value number, they compute the same value.
    """
    value_table = {}  # (op, vn1, vn2) -> value number
    var_to_vn = {}    # variable -> value number
    next_vn = [0]

    def get_vn(var):
        if var in var_to_vn:
            return var_to_vn[var]
        # Constants get unique value numbers
        vn = next_vn[0]
        next_vn[0] += 1
        var_to_vn[var] = vn
        return vn

    for instr in ssa_block:
        if instr.is_binary_op:
            vn1 = get_vn(instr.left)
            vn2 = get_vn(instr.right)
            key = (instr.op, vn1, vn2)

            # Check for commutative operations
            if instr.op in ('+', '*'):
                key = (instr.op, min(vn1, vn2), max(vn1, vn2))

            if key in value_table:
                # Redundant! Reuse existing value
                var_to_vn[instr.dest] = value_table[key]
                instr.replace_with_copy(value_table[key])
            else:
                vn = next_vn[0]
                next_vn[0] += 1
                value_table[key] = vn
                var_to_vn[instr.dest] = vn

    return var_to_vn
```

### 6.3 Dead Code Elimination on SSA

```python
def aggressive_dead_code_elimination(ssa_program):
    """
    Mark-sweep style DCE on SSA form.
    Start from essential instructions (I/O, stores, branches),
    mark everything they depend on as live.
    """
    live = set()
    worklist = []

    # Mark essential instructions
    for block in ssa_program.blocks:
        for instr in block.instructions:
            if instr.has_side_effect or instr.is_branch:
                live.add(instr)
                worklist.append(instr)

    # Propagate liveness along SSA def-use chains
    while worklist:
        instr = worklist.pop()
        for use_var in instr.uses:
            def_instr = ssa_program.get_definition(use_var)
            if def_instr not in live:
                live.add(def_instr)
                worklist.append(def_instr)

    # Remove dead instructions
    for block in ssa_program.blocks:
        block.instructions = [i for i in block.instructions if i in live]
```

---

## 7. SSA Destruction

### 7.1 Why Destroy SSA?

Before generating machine code, phi functions must be eliminated because they have no direct hardware equivalent. SSA destruction converts phi functions into copy instructions placed along CFG edges.

### 7.2 Naive Approach: Parallel Copies

```
# SSA:
B3: x3 = phi(x1:B1, x2:B2)
    y3 = phi(y1:B1, y2:B2)

# After destruction (naive):
B1: ...
    x3 = x1    # copy at end of B1
    y3 = y1
    goto B3

B2: ...
    x3 = x2    # copy at end of B2
    y3 = y2
    goto B3

B3: // no phi functions, x3 and y3 are ready
```

### 7.3 The Lost-Copy Problem

Naive insertion can create incorrect code when a phi source is modified before the copy:

```
# SSA:
B2: x2 = phi(x1:B1, x3:B2)
    x3 = x2 + 1
    if x3 < 10 goto B2 else B3

# Naive destruction (INCORRECT):
B2: x2 = x3     # but x3 hasn't been computed yet on first iteration!
    x3 = x2 + 1
    if x3 < 10 goto B2 else B3
```

### 7.4 Correct SSA Destruction

The solution uses **critical edge splitting** and careful copy sequencing:

```python
def destroy_ssa(ssa_program):
    """
    Convert out of SSA by replacing phi functions with copies.
    Handles the lost-copy and swap problems correctly.
    """
    for block in ssa_program.blocks:
        phis = [i for i in block.instructions if i.is_phi]
        if not phis:
            continue

        for phi in phis:
            for pred, src_var in phi.args.items():
                dest_var = phi.dest
                if src_var != dest_var:
                    # Insert copy at end of predecessor (before branch)
                    # May need to split critical edges first
                    if is_critical_edge(pred, block):
                        new_block = split_edge(pred, block)
                        new_block.append_copy(dest_var, src_var)
                    else:
                        pred.insert_copy_before_branch(dest_var, src_var)

        # Remove phi functions
        block.instructions = [i for i in block.instructions if not i.is_phi]


def is_critical_edge(pred, succ):
    """An edge is critical if pred has multiple successors and succ has multiple predecessors."""
    return len(pred.successors) > 1 and len(succ.predecessors) > 1


def split_edge(pred, succ):
    """Insert a new empty block on the edge from pred to succ."""
    new_block = BasicBlock(f"split_{pred.name}_{succ.name}")
    new_block.add_successor(succ)
    pred.replace_successor(succ, new_block)
    succ.replace_predecessor(pred, new_block)
    return new_block
```

---

## 8. Pruned and Semi-Pruned SSA

### 8.1 Minimal SSA

The Cytron algorithm produces **minimal SSA** -- the fewest phi functions needed for correctness. However, some of these phi functions may be dead (their values are never used).

### 8.2 Pruned SSA

**Pruned SSA** removes dead phi functions by combining SSA construction with liveness analysis:

```python
def build_pruned_ssa(cfg, idom, df, var_defs, live_in):
    """
    Build pruned SSA: only place phi where variable is live-in.
    live_in: dict mapping block -> set of variables live at block entry
    """
    phi_functions = {}

    for var, def_blocks in var_defs.items():
        idf = iterated_dominance_frontier(df, def_blocks)
        for block in idf:
            if var in live_in[block]:  # Only if live!
                preds = cfg[block]
                phi_functions[(block, var)] = {
                    'var': var,
                    'args': {pred: None for pred in preds}
                }

    return phi_functions
```

### 8.3 Semi-Pruned SSA

Semi-pruned SSA is a practical compromise: it eliminates phi functions for variables that are local to a single block (never live across block boundaries), without requiring full liveness analysis.

```python
def build_semi_pruned_ssa(cfg, idom, df, var_defs, block_local_vars):
    """
    Semi-pruned SSA: skip phi placement for block-local variables.
    block_local_vars: set of variables only used in their defining block
    """
    phi_functions = {}

    for var, def_blocks in var_defs.items():
        if var in block_local_vars:
            continue  # No phi needed for block-local vars

        idf = iterated_dominance_frontier(df, def_blocks)
        for block in idf:
            preds = cfg[block]
            phi_functions[(block, var)] = {
                'var': var,
                'args': {pred: None for pred in preds}
            }

    return phi_functions
```

### 8.4 SSA Variants Comparison

| Variant | Phi Count | Extra Analysis Required | Use Case |
|---------|-----------|------------------------|----------|
| Maximal | Many (every join for every var) | None | Theoretical only |
| Minimal | Fewest correct | Dominance frontiers | Standard |
| Semi-Pruned | Fewer | Block-local identification | Quick compilation |
| Pruned | Fewest useful | Full liveness | Aggressive optimization |

---

## 9. Summary

- **SSA form** assigns each variable exactly once, making def-use chains explicit
- **Phi functions** merge values at control-flow join points
- **Dominance** and **dominance frontiers** determine where phi functions are needed
- The **Cytron et al. algorithm** constructs SSA in two phases: phi placement and renaming
- SSA enables powerful optimizations: SCCP, GVN, aggressive DCE
- **SSA destruction** replaces phi functions with copies, handling critical edges correctly
- **Pruned SSA** avoids dead phi functions using liveness information

---

## 10. Exercises

1. **Manual SSA conversion**: Convert the following code into SSA form by hand, showing all phi functions:
   ```
   x = 0
   y = 1
   while x < 10:
       if x % 2 == 0:
           y = y + x
       else:
           y = y * 2
       x = x + 1
   print(y)
   ```

2. **Dominator tree**: Given a CFG with blocks B1-B7, compute the dominator tree and dominance frontiers.

3. **Implement SSA construction**: Write a Python program that takes a simple three-address code program and converts it to SSA form.

4. **SCCP on SSA**: Apply Sparse Conditional Constant Propagation to an SSA program with known constant inputs.

5. **SSA destruction**: Given an SSA program with phi functions, produce correct copy instructions, identifying and splitting critical edges.

---

## 11. References

1. Cytron, R., Ferrante, J., Rosen, B., Wegman, M., Zadeck, F. K. (1991). "Efficiently Computing Static Single Assignment Form and the Control Dependence Graph." *ACM Transactions on Programming Languages and Systems*, 13(4), 451-490.
2. Cooper, K. D., Harvey, T. J., Kennedy, K. (2001). "A Simple, Fast Dominance Algorithm." *Software Practice and Experience*.
3. Appel, A. W. (1998). "SSA is Functional Programming." *ACM SIGPLAN Notices*, 33(4), 17-20.
4. Briggs, P., Cooper, K., Harvey, T., Simpson, T. (1998). "Practical Improvements to the Construction and Destruction of Static Single Assignment Form." *Software Practice and Experience*, 28(8), 859-881.
5. Wegman, M. N., Zadeck, F. K. (1991). "Constant Propagation with Conditional Branches." *ACM Transactions on Programming Languages and Systems*, 13(2), 181-210.

---

**Previous**: [16. Modern Compiler Infrastructure](./16_Modern_Compiler_Infrastructure.md) | **Next**: [19. Register Allocation](./19_Register_Allocation.md)
