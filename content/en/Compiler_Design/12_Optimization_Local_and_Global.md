# Lesson 12: Optimization -- Local and Global

## Learning Objectives

After completing this lesson, you will be able to:

1. **Explain** the principles of compiler optimization: safety, profitability, and opportunity
2. **Apply** local optimizations within a single basic block (constant folding, CSE, copy propagation, dead code elimination, strength reduction)
3. **Formulate** global data flow analyses (reaching definitions, available expressions, live variables, very busy expressions)
4. **Describe** the mathematical framework of data flow analysis using lattices and fixed-point iteration
5. **Implement** the worklist algorithm for iterative data flow analysis
6. **Build** a complete data flow analysis engine in Python

---

## 1. Optimization Overview

### 1.1 What Is an Optimization?

In compiler terminology, an **optimization** is a program transformation that preserves the observable behavior of the program while improving some metric -- typically execution speed, code size, or energy consumption.

The term "optimization" is somewhat misleading: compilers rarely find the truly *optimal* code. More accurately, compilers perform **code improvement**.

### 1.2 The Three Requirements

Every optimization must satisfy three criteria:

1. **Safety (Correctness)**: The transformation must not change the observable behavior of the program. A safe transformation produces the same outputs for all possible inputs.

2. **Profitability**: The transformation should actually improve the code. A transformation that increases register pressure and causes more spills may make the code slower despite reducing instruction count.

3. **Opportunity**: The pattern targeted by the optimization must actually occur in the code. The compiler must **detect** the opportunity before applying the transformation.

### 1.3 Classification of Optimizations

| Scope | Description | Examples |
|-------|-------------|---------|
| **Local** | Within a single basic block | Constant folding, CSE, copy propagation |
| **Global** (intraprocedural) | Across basic blocks within one function | Loop-invariant code motion, global CSE |
| **Interprocedural** | Across function boundaries | Inlining, interprocedural constant propagation |
| **Machine-dependent** | Exploit target machine features | Instruction scheduling, peephole optimization |
| **Machine-independent** | Apply to any target | Most IR-level optimizations |

### 1.4 When Optimizations Are Applied

```
Source → [Front End] → IR
                        │
                        ▼
         ┌──────────────────────────────────┐
         │  Machine-Independent Optimizations │
         │  (operate on IR)                   │
         │  - Constant propagation            │
         │  - Dead code elimination           │
         │  - Loop optimizations              │
         │  - Inlining                        │
         └──────────────────────────────────┘
                        │
                        ▼
         ┌──────────────────────────────────┐
         │  Code Generation                   │
         │  (instruction selection, reg alloc) │
         └──────────────────────────────────┘
                        │
                        ▼
         ┌──────────────────────────────────┐
         │  Machine-Dependent Optimizations   │
         │  (operate on machine code)         │
         │  - Peephole optimization           │
         │  - Instruction scheduling          │
         └──────────────────────────────────┘
                        │
                        ▼
                    Machine Code
```

### 1.5 The Phase-Ordering Problem

Optimizations interact: one optimization may create opportunities for another, or undo the work of a previous one. The order in which optimizations are applied matters, and finding the best order is undecidable in general.

Practical compilers use carefully tuned sequences (e.g., GCC's `-O2` enables about 60 optimization passes in a specific order).

---

## 2. Local Optimizations

Local optimizations operate within a **single basic block** -- a straight-line sequence of instructions with no branches in or out (except at the beginning and end). Because there is only one execution path through a basic block, these optimizations require no data flow analysis.

### 2.1 Constant Folding

**Constant folding** evaluates constant expressions at compile time rather than at runtime.

**Before**:
```
t1 = 3 + 4
t2 = t1 * 2
```

**After**:
```
t1 = 7
t2 = 14
```

**Rules**: The compiler can fold an expression $e_1 \;\text{op}\; e_2$ if both $e_1$ and $e_2$ are constants. This applies recursively.

**Caution**: The compiler must respect the target machine's arithmetic semantics:
- Integer overflow behavior (signed vs unsigned)
- Floating-point rounding modes
- Division by zero handling

### 2.2 Constant Propagation

**Constant propagation** replaces uses of a variable with its constant value when the variable is known to be constant.

**Before**:
```
x = 5
y = x + 3
z = x * y
```

**After** (propagation + folding):
```
x = 5
y = 8       // 5 + 3
z = 40      // 5 * 8
```

The combination of constant propagation and constant folding is very powerful -- each enables the other in a cascade.

### 2.3 Algebraic Simplification

**Algebraic simplification** applies mathematical identities to simplify expressions:

| Original | Simplified | Identity |
|----------|-----------|----------|
| $x + 0$ | $x$ | Additive identity |
| $x - 0$ | $x$ | Additive identity |
| $x \times 1$ | $x$ | Multiplicative identity |
| $x \times 0$ | $0$ | Zero property |
| $x / 1$ | $x$ | Division identity |
| $x - x$ | $0$ | Self-subtraction |
| $x / x$ | $1$ | Self-division (when $x \neq 0$) |
| $-(-x)$ | $x$ | Double negation |
| $x + x$ | $2 \times x$ | (or left shift by 1) |

**For booleans**:

| Original | Simplified |
|----------|-----------|
| $x \land \text{true}$ | $x$ |
| $x \land \text{false}$ | $\text{false}$ |
| $x \lor \text{true}$ | $\text{true}$ |
| $x \lor \text{false}$ | $x$ |
| $\lnot(\lnot x)$ | $x$ |

### 2.4 Strength Reduction (Local)

**Strength reduction** replaces expensive operations with cheaper equivalents:

| Expensive | Cheap | Condition |
|-----------|-------|-----------|
| $x \times 2^n$ | $x \ll n$ | Always valid for integers |
| $x / 2^n$ | $x \gg n$ | Unsigned integers |
| $x \% 2^n$ | $x \;\&\; (2^n - 1)$ | Unsigned integers |
| $x \times 3$ | $(x \ll 1) + x$ | When shift+add is cheaper |
| $x \times 5$ | $(x \ll 2) + x$ | When shift+add is cheaper |
| $x \times 15$ | $(x \ll 4) - x$ | When shift-sub is cheaper |

**Example**:
```
Before: t1 = i * 4
After:  t1 = i << 2
```

### 2.5 Dead Code Elimination (Local)

**Dead code** is code whose results are never used. Eliminating it reduces code size and may improve cache behavior.

**Before**:
```
t1 = a + b      // t1 is used
t2 = c * d      // t2 is NEVER used later
t3 = t1 - 5     // t3 is used
```

**After**:
```
t1 = a + b
t3 = t1 - 5
```

In a basic block, a definition is dead if the variable is not used before it is redefined or before the end of the block (and is not live out of the block).

### 2.6 Common Subexpression Elimination (CSE)

**CSE** identifies expressions that are computed more than once and replaces redundant computations with references to the first computation.

**Before**:
```
t1 = a + b
t2 = a + b      // Same expression as t1
t3 = t1 * t2
```

**After**:
```
t1 = a + b
t2 = t1          // Reuse t1's value
t3 = t1 * t1
```

**Safety condition**: Neither `a` nor `b` must be redefined between the first computation and the redundant one.

CSE within a basic block can be efficiently implemented using **value numbering** (see below) or DAG construction (covered in Lesson 9).

### 2.7 Copy Propagation

**Copy propagation** replaces uses of a variable that was assigned by a copy (`x = y`) with the original variable.

**Before**:
```
t1 = a + b
t2 = t1          // copy
t3 = t2 * c      // uses t2 (which is really t1)
```

**After**:
```
t1 = a + b
t2 = t1           // copy (may become dead)
t3 = t1 * c       // replaced t2 with t1
```

After copy propagation, the copy `t2 = t1` may become dead code and can be eliminated.

### 2.8 Local Value Numbering

**Local value numbering** is a systematic method for performing CSE, constant folding, and algebraic simplification in a single pass over a basic block.

**Idea**: Assign a **value number** to each computed value. Two expressions get the same value number if they compute the same value.

**Algorithm**:

```
hash_table = {}  # Maps (op, vn1, vn2) → value number
var_to_vn = {}   # Maps variable name → value number
next_vn = 0

for each instruction "x = y op z":
    vn_y = var_to_vn[y]
    vn_z = var_to_vn[z]

    key = (op, vn_y, vn_z)
    if key in hash_table:
        vn_x = hash_table[key]   # Same value already computed
        # Replace instruction with copy from the representative
    else:
        vn_x = next_vn++
        hash_table[key] = vn_x

    var_to_vn[x] = vn_x
```

### 2.9 Python Implementation: Local Optimizer

```python
"""
Local optimizations within a single basic block.
Implements constant folding, constant propagation, algebraic
simplification, strength reduction, CSE, copy propagation,
and dead code elimination.
"""

from dataclasses import dataclass
from typing import Optional, Union
import operator


@dataclass
class TAC:
    """Three-address code instruction."""
    result: str
    op: str           # '+', '-', '*', '/', 'copy', 'nop'
    arg1: str = ""
    arg2: str = ""

    def __str__(self):
        if self.op == "copy":
            return f"{self.result} = {self.arg1}"
        elif self.op == "nop":
            return f"// NOP (eliminated)"
        elif self.arg2:
            return f"{self.result} = {self.arg1} {self.op} {self.arg2}"
        elif self.arg1:
            return f"{self.result} = {self.op} {self.arg1}"
        else:
            return f"{self.result} = {self.op}"


def is_constant(s: str) -> bool:
    """Check if a string represents an integer constant."""
    try:
        int(s)
        return True
    except (ValueError, TypeError):
        return False


def const_val(s: str) -> int:
    """Get the integer value of a constant string."""
    return int(s)


class LocalOptimizer:
    """
    Performs local optimizations on a basic block (list of TAC instructions).
    """

    def __init__(self, instructions: list[TAC], live_out: set = None):
        self.instructions = list(instructions)
        self.live_out = live_out or set()  # Variables live at exit
        self.changed = False

    def optimize(self, max_passes: int = 10) -> list[TAC]:
        """Run all local optimizations until convergence."""
        for _ in range(max_passes):
            self.changed = False
            self.constant_folding()
            self.constant_propagation()
            self.algebraic_simplification()
            self.strength_reduction()
            self.copy_propagation()
            self.common_subexpression_elimination()
            self.dead_code_elimination()
            if not self.changed:
                break

        # Remove NOP instructions
        return [i for i in self.instructions if i.op != "nop"]

    def constant_folding(self):
        """Evaluate constant expressions at compile time."""
        ops = {
            '+': operator.add, '-': operator.sub,
            '*': operator.mul, '/': operator.floordiv,
            '%': operator.mod,
        }

        for i, instr in enumerate(self.instructions):
            if instr.op in ops and is_constant(instr.arg1) \
               and is_constant(instr.arg2):
                a = const_val(instr.arg1)
                b = const_val(instr.arg2)
                if instr.op in ('/', '%') and b == 0:
                    continue  # Cannot fold division by zero
                result = ops[instr.op](a, b)
                self.instructions[i] = TAC(instr.result, "copy", str(result))
                self.changed = True

    def constant_propagation(self):
        """Replace variable uses with constant values when known."""
        constants = {}  # var -> constant value

        for i, instr in enumerate(self.instructions):
            # Check if this defines a constant
            if instr.op == "copy" and is_constant(instr.arg1):
                constants[instr.result] = instr.arg1

            # Replace uses with constants
            new_arg1 = instr.arg1
            new_arg2 = instr.arg2

            if instr.arg1 in constants and instr.op != "copy":
                new_arg1 = constants[instr.arg1]
            elif instr.arg1 in constants and instr.op == "copy":
                new_arg1 = constants[instr.arg1]

            if instr.arg2 in constants:
                new_arg2 = constants[instr.arg2]

            if new_arg1 != instr.arg1 or new_arg2 != instr.arg2:
                self.instructions[i] = TAC(instr.result, instr.op,
                                           new_arg1, new_arg2)
                self.changed = True

            # If this instruction redefines a variable, invalidate it
            if instr.result in constants and instr.op != "copy":
                del constants[instr.result]
            elif (instr.result in constants and instr.op == "copy"
                  and not is_constant(instr.arg1)):
                del constants[instr.result]

    def algebraic_simplification(self):
        """Apply algebraic identities."""
        for i, instr in enumerate(self.instructions):
            result = instr.result
            new_instr = None

            # x + 0 = x, 0 + x = x
            if instr.op == '+':
                if instr.arg2 == '0':
                    new_instr = TAC(result, "copy", instr.arg1)
                elif instr.arg1 == '0':
                    new_instr = TAC(result, "copy", instr.arg2)

            # x - 0 = x
            elif instr.op == '-':
                if instr.arg2 == '0':
                    new_instr = TAC(result, "copy", instr.arg1)
                # x - x = 0
                elif instr.arg1 == instr.arg2:
                    new_instr = TAC(result, "copy", "0")

            # x * 1 = x, 1 * x = x, x * 0 = 0
            elif instr.op == '*':
                if instr.arg2 == '1':
                    new_instr = TAC(result, "copy", instr.arg1)
                elif instr.arg1 == '1':
                    new_instr = TAC(result, "copy", instr.arg2)
                elif instr.arg2 == '0' or instr.arg1 == '0':
                    new_instr = TAC(result, "copy", "0")

            # x / 1 = x
            elif instr.op == '/':
                if instr.arg2 == '1':
                    new_instr = TAC(result, "copy", instr.arg1)
                # x / x = 1 (assume x != 0)
                elif instr.arg1 == instr.arg2:
                    new_instr = TAC(result, "copy", "1")

            if new_instr:
                self.instructions[i] = new_instr
                self.changed = True

    def strength_reduction(self):
        """Replace expensive operations with cheaper ones."""
        for i, instr in enumerate(self.instructions):
            if instr.op == '*' and is_constant(instr.arg2):
                val = const_val(instr.arg2)
                if val > 0 and (val & (val - 1)) == 0:
                    shift = val.bit_length() - 1
                    self.instructions[i] = TAC(
                        instr.result, "<<", instr.arg1, str(shift)
                    )
                    self.changed = True

            elif instr.op == '/' and is_constant(instr.arg2):
                val = const_val(instr.arg2)
                if val > 0 and (val & (val - 1)) == 0:
                    shift = val.bit_length() - 1
                    self.instructions[i] = TAC(
                        instr.result, ">>", instr.arg1, str(shift)
                    )
                    self.changed = True

    def copy_propagation(self):
        """Replace uses of copied variables with the source."""
        copies = {}  # target -> source

        for i, instr in enumerate(self.instructions):
            # Record copies
            if instr.op == "copy" and not is_constant(instr.arg1):
                copies[instr.result] = instr.arg1

            # Propagate copies in uses
            new_arg1 = copies.get(instr.arg1, instr.arg1)
            new_arg2 = copies.get(instr.arg2, instr.arg2)

            if new_arg1 != instr.arg1 or new_arg2 != instr.arg2:
                self.instructions[i] = TAC(instr.result, instr.op,
                                           new_arg1, new_arg2)
                self.changed = True

            # Invalidate if source is redefined
            to_remove = []
            for target, source in copies.items():
                if source == instr.result or target == instr.result:
                    to_remove.append(target)
            for t in to_remove:
                if t != instr.result or instr.op != "copy":
                    copies.pop(t, None)

    def common_subexpression_elimination(self):
        """Eliminate redundant computations using value numbering."""
        # (op, arg1, arg2) -> result variable
        computed = {}

        for i, instr in enumerate(self.instructions):
            if instr.op in ('+', '-', '*', '/', '%', '<<', '>>'):
                key = (instr.op, instr.arg1, instr.arg2)

                # Check commutative operators
                if instr.op in ('+', '*'):
                    comm_key = (instr.op, instr.arg2, instr.arg1)
                    if comm_key in computed:
                        key = comm_key

                if key in computed:
                    # Replace with copy from previous computation
                    prev_result = computed[key]
                    self.instructions[i] = TAC(
                        instr.result, "copy", prev_result
                    )
                    self.changed = True
                else:
                    computed[key] = instr.result

                # Invalidate entries if arg is redefined
                to_remove = [k for k, v in computed.items()
                             if instr.result in k[1:]]
                for k in to_remove:
                    del computed[k]

    def dead_code_elimination(self):
        """Remove instructions whose results are never used."""
        # Find all used variables (scan backward)
        used = set(self.live_out)

        # Scan backward to find used variables
        for instr in reversed(self.instructions):
            if instr.op == "nop":
                continue
            if instr.arg1 and not is_constant(instr.arg1):
                used.add(instr.arg1)
            if instr.arg2 and not is_constant(instr.arg2):
                used.add(instr.arg2)

        # Mark instructions as dead if result is not used
        for i, instr in enumerate(self.instructions):
            if instr.op == "nop":
                continue
            if instr.result not in used and instr.result not in self.live_out:
                self.instructions[i] = TAC("", "nop")
                self.changed = True


def demo_local_optimizations():
    """Demonstrate local optimizations on a basic block."""

    instructions = [
        TAC("t1", "copy", "5"),         # t1 = 5
        TAC("t2", "copy", "3"),         # t2 = 3
        TAC("t3", "+", "t1", "t2"),     # t3 = t1 + t2 → fold to 8
        TAC("t4", "*", "t3", "0"),      # t4 = t3 * 0 → simplify to 0
        TAC("t5", "+", "a", "b"),       # t5 = a + b
        TAC("t6", "+", "a", "b"),       # t6 = a + b  (CSE)
        TAC("t7", "*", "t5", "t6"),     # t7 = t5 * t6 → t5 * t5
        TAC("t8", "*", "t7", "8"),      # t8 = t7 * 8 → strength reduce
        TAC("t9", "copy", "t8"),        # t9 = t8 (copy)
        TAC("t10", "+", "t9", "0"),     # t10 = t9 + 0 → simplify
        TAC("unused", "+", "t1", "t2"), # dead code (unused not live-out)
    ]

    live_out = {"t8", "t10"}

    print("=== Before Optimization ===")
    for instr in instructions:
        print(f"  {instr}")

    optimizer = LocalOptimizer(instructions, live_out)
    optimized = optimizer.optimize()

    print("\n=== After Optimization ===")
    for instr in optimized:
        print(f"  {instr}")


if __name__ == "__main__":
    demo_local_optimizations()
```

---

## 3. Global Data Flow Analysis

### 3.1 Why Global Analysis?

Local optimizations are limited to a single basic block. To optimize across block boundaries, we need **global** (intraprocedural) analysis that reasons about the flow of data through the entire control flow graph.

**Example**: To determine if a variable has a constant value at a particular point, we must consider all paths that could reach that point -- this requires analyzing the entire CFG.

### 3.2 The Four Classic Analyses

The four foundational data flow analyses differ in what they track and in which direction information flows:

| Analysis | Direction | Flow Meets At | Question |
|----------|-----------|---------------|----------|
| **Reaching Definitions** | Forward | Join points ($\cup$) | Which definitions of $x$ could reach this point? |
| **Available Expressions** | Forward | Join points ($\cap$) | Which expressions are guaranteed to have been computed? |
| **Live Variables** | Backward | Join points ($\cup$) | Which variables might be used before being redefined? |
| **Very Busy Expressions** | Backward | Join points ($\cap$) | Which expressions will definitely be evaluated on all paths? |

### 3.3 Reaching Definitions

A **definition** is an instruction that assigns a value to a variable (e.g., `d: x = ...`).

Definition $d$ of variable $x$ **reaches** point $p$ if:
- There is a path from $d$ to $p$, and
- $x$ is not redefined along that path

**Use**: Constant propagation (if only one reaching definition exists and it assigns a constant, we can propagate it).

#### Data Flow Equations

For each basic block $B$:

$$\text{Out}(B) = \text{Gen}(B) \cup (\text{In}(B) - \text{Kill}(B))$$

$$\text{In}(B) = \bigcup_{P \in \text{pred}(B)} \text{Out}(P)$$

where:
- $\text{Gen}(B)$ = definitions generated in $B$ that survive to the end of $B$
- $\text{Kill}(B)$ = definitions killed by $B$ (definitions of variables that $B$ redefines)

**Direction**: Forward (information flows from predecessors to successors)

**Meet operator**: Union ($\cup$) -- a definition reaches a point if it reaches via *any* path

**Initialization**: $\text{Out}(B) = \text{Gen}(B)$ for all blocks; $\text{In}(\text{entry}) = \emptyset$

#### Example

```
B1: d1: x = 5
    d2: y = 1
    → B2

B2: d3: z = x + y
    if z < 10 → B3 else → B4

B3: d4: x = x + 1
    d5: y = y * 2
    → B2

B4: print z
```

**Gen and Kill sets**:

| Block | Gen | Kill |
|-------|-----|------|
| B1 | {d1, d2} | {d4, d5} (other defs of x, y) |
| B2 | {d3} | {} (no other defs of z) |
| B3 | {d4, d5} | {d1, d2} (other defs of x, y) |
| B4 | {} | {} |

**Iteration**:

| Iteration | In(B1) | Out(B1) | In(B2) | Out(B2) | In(B3) | Out(B3) | In(B4) | Out(B4) |
|-----------|--------|---------|--------|---------|--------|---------|--------|---------|
| Init | {} | {d1,d2} | {} | {d3} | {} | {d4,d5} | {} | {} |
| 1 | {} | {d1,d2} | {d1,d2,d4,d5} | {d1,d2,d3,d4,d5} | {d1,d2,d3,d4,d5} | {d3,d4,d5} | {d1,d2,d3,d4,d5} | {d1,d2,d3,d4,d5} |
| 2 | {} | {d1,d2} | {d1,d2,d3,d4,d5} | {d1,d2,d3,d4,d5} | {d1,d2,d3,d4,d5} | {d3,d4,d5} | {d1,d2,d3,d4,d5} | {d1,d2,d3,d4,d5} |

At B2, both $d_1$ and $d_4$ define $x$, so we cannot propagate a single constant for $x$.

### 3.4 Available Expressions

An expression $e$ is **available** at point $p$ if:
- On **every** path from the entry to $p$, $e$ is computed, and
- None of $e$'s operands are redefined after the last computation of $e$

**Use**: Global common subexpression elimination. If $e$ is available at $p$ and $p$ computes $e$, we can replace $p$'s computation with the previously computed value.

#### Data Flow Equations

$$\text{Out}(B) = \text{Gen}(B) \cup (\text{In}(B) - \text{Kill}(B))$$

$$\text{In}(B) = \bigcap_{P \in \text{pred}(B)} \text{Out}(P)$$

**Direction**: Forward

**Meet operator**: Intersection ($\cap$) -- an expression is available only if it is available on *every* path

**Initialization**: $\text{Out}(\text{entry}) = \emptyset$; $\text{Out}(B) = U$ (all expressions) for $B \neq \text{entry}$

The use of intersection is crucial: an expression is available only if it has been computed along *all* incoming paths. This is a "must" analysis (must be true on all paths), in contrast to reaching definitions which is a "may" analysis.

### 3.5 Live Variables

A variable $v$ is **live** at point $p$ if:
- There is a path from $p$ to a use of $v$, and
- $v$ is not redefined along that path

**Use**: Register allocation (live variables need registers), dead code elimination (a definition is dead if the variable is not live after it).

#### Data Flow Equations

$$\text{In}(B) = \text{Use}(B) \cup (\text{Out}(B) - \text{Def}(B))$$

$$\text{Out}(B) = \bigcup_{S \in \text{succ}(B)} \text{In}(S)$$

where:
- $\text{Use}(B)$ = variables used in $B$ before being defined in $B$
- $\text{Def}(B)$ = variables defined in $B$

**Direction**: Backward (information flows from successors to predecessors)

**Meet operator**: Union ($\cup$) -- a variable is live if it is needed on *any* path forward

**Initialization**: $\text{In}(B) = \emptyset$ for all blocks; $\text{Out}(\text{exit}) = \emptyset$ (or the set of variables whose values are needed after the function returns)

### 3.6 Very Busy Expressions

An expression $e$ is **very busy** at point $p$ if:
- On **every** path from $p$ to the exit, $e$ is evaluated before any of its operands are redefined

**Use**: Code hoisting -- if $e$ is very busy at $p$, we can safely move the computation of $e$ to $p$ (earlier), potentially reducing code size.

#### Data Flow Equations

$$\text{In}(B) = \text{Gen}(B) \cup (\text{Out}(B) - \text{Kill}(B))$$

$$\text{Out}(B) = \bigcap_{S \in \text{succ}(B)} \text{In}(S)$$

**Direction**: Backward

**Meet operator**: Intersection ($\cap$) -- an expression is very busy only if it is evaluated on *every* path

### 3.7 Summary of the Four Analyses

| | Reaching Defs | Available Exprs | Live Vars | Very Busy Exprs |
|--|---------------|-----------------|-----------|-----------------|
| **Domain** | Sets of definitions | Sets of expressions | Sets of variables | Sets of expressions |
| **Direction** | Forward | Forward | Backward | Backward |
| **Meet** | $\cup$ (may) | $\cap$ (must) | $\cup$ (may) | $\cap$ (must) |
| **Transfer** | $\text{Gen} \cup (\text{In} - \text{Kill})$ | $\text{Gen} \cup (\text{In} - \text{Kill})$ | $\text{Use} \cup (\text{Out} - \text{Def})$ | $\text{Gen} \cup (\text{Out} - \text{Kill})$ |
| **Init (boundary)** | $\text{Out}(\text{entry}) = \emptyset$ | $\text{Out}(\text{entry}) = \emptyset$ | $\text{In}(\text{exit}) = \emptyset$ | $\text{In}(\text{exit}) = \emptyset$ |
| **Init (others)** | $\emptyset$ | $U$ (all exprs) | $\emptyset$ | $U$ (all exprs) |
| **Use** | Constant prop. | Global CSE | Reg. alloc., DCE | Code hoisting |

---

## 4. Data Flow Analysis Framework

### 4.1 Lattice Theory Basics

Data flow analyses can be unified under a mathematical framework based on **lattice theory**.

**Definition**: A **lattice** $(L, \sqsubseteq)$ is a partially ordered set where every pair of elements has a **meet** ($\sqcap$, greatest lower bound) and a **join** ($\sqcup$, least upper bound).

For data flow analysis, we typically use **complete lattices** where every subset has a meet and join. Key elements:

- $\top$ (top): The greatest element -- represents "no information yet" or "all possibilities"
- $\bot$ (bottom): The least element -- represents "unreachable" or "contradiction"

#### Lattice for Reaching Definitions

The lattice is the **powerset** of all definitions, ordered by subset inclusion:

$$L = 2^{\text{Defs}}, \quad A \sqsubseteq B \iff A \subseteq B$$

- $\top = \text{Defs}$ (all definitions -- universal set)
- $\bot = \emptyset$ (no definitions)
- Meet ($\sqcap$) = Union ($\cup$) since we use a may-analysis (any path)

Wait -- in the standard formulation, the convention depends on the direction of the lattice. Let us be careful.

For a **may analysis** (like reaching definitions), the lattice goes from $\bot = \emptyset$ (no information) upward, and the meet is $\cup$ (union). We start with the most optimistic value ($\bot = \emptyset$: "nothing reaches here") and add information until reaching a fixed point.

For a **must analysis** (like available expressions), the lattice goes from $\top = U$ (all expressions available) downward, and the meet is $\cap$ (intersection). We start with the most optimistic value ($\top = U$: "everything is available") and remove information until reaching a fixed point.

#### General Framework

A data flow analysis is defined by:

1. A **lattice** $(L, \sqsubseteq)$ with meet $\sqcap$
2. A **transfer function** $f_B : L \to L$ for each block $B$
3. A **direction** (forward or backward)
4. A **boundary condition** (initial value at entry/exit)
5. An **initial value** for all other blocks

The solution is the **greatest fixed point** of the system of equations.

### 4.2 Transfer Functions

A transfer function $f_B$ describes how information changes as it passes through block $B$. Most data flow analyses use transfer functions of the form:

$$f_B(X) = \text{Gen}(B) \cup (X - \text{Kill}(B))$$

This is a **monotone** function: if $X \sqsubseteq Y$, then $f_B(X) \sqsubseteq f_B(Y)$.

**Monotonicity is essential**: it guarantees that the iterative algorithm converges.

### 4.3 Meet-Over-Paths (MOP) Solution

The **ideal** solution is the **Meet-Over-All-Paths** (MOP) solution. For a forward analysis, the MOP solution at the entry of block $B$ is:

$$\text{MOP}(B) = \bigsqcap_{\text{path } p : \text{entry} \to B} f_p(\text{boundary\_value})$$

where $f_p$ is the composition of transfer functions along path $p$.

Computing MOP directly is generally undecidable (there may be infinitely many paths due to loops). Instead, we compute the **Maximum Fixed Point** (MFP), which is guaranteed to be at least as conservative as MOP (i.e., $\text{MFP} \sqsubseteq \text{MOP}$ in the appropriate order).

For transfer functions of the form $f(X) = \text{Gen} \cup (X - \text{Kill})$ (which are **distributive**), MFP = MOP exactly.

### 4.4 Fixed-Point Iteration

**Theorem (Tarski)**: If $f$ is a monotone function on a complete lattice, then $f$ has a least fixed point, which can be obtained by iterating from $\bot$:

$$\bot, f(\bot), f^2(\bot), \ldots$$

For data flow analysis, we simultaneously iterate over all blocks until no values change:

```
Algorithm: Iterative Data Flow Analysis (Forward)

Initialize:
    Out(entry) = boundary_value
    Out(B) = initial_value for all B ≠ entry

Repeat:
    for each block B (in some order):
        In(B) = Meet over all predecessors P of Out(P)
        Out(B) = f_B(In(B))
until no Out(B) changed
```

**Convergence**: Guaranteed because:
1. The lattice has finite height (finite number of definitions/expressions)
2. Transfer functions are monotone
3. Each iteration moves values in one direction (up or down in the lattice)

**Worst-case iterations**: At most $h \times |V|$ where $h$ is the lattice height and $|V|$ is the number of CFG nodes. In practice, 2--3 iterations usually suffice.

### 4.5 Worklist Algorithm

The naive iterative algorithm repeatedly scans all blocks even when most have not changed. The **worklist algorithm** processes only blocks whose inputs have changed:

```
Algorithm: Worklist Data Flow Analysis (Forward)

Initialize:
    Out(entry) = boundary_value
    Out(B) = initial_value for all B ≠ entry
    worklist = all blocks except entry

While worklist is not empty:
    Remove a block B from the worklist
    In(B) = Meet over predecessors P of Out(P)
    new_out = f_B(In(B))
    if new_out ≠ Out(B):
        Out(B) = new_out
        Add all successors of B to the worklist
```

**Efficiency**: The worklist algorithm avoids redundant work. Using reverse postorder for forward analyses (or postorder for backward analyses) as the initial worklist order improves convergence.

---

## 5. Python Implementation: Data Flow Analysis

```python
"""
Generic Data Flow Analysis Framework.

Implements the worklist algorithm for forward and backward analyses.
Includes implementations of all four classic analyses.
"""

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Callable
from enum import Enum, auto
from abc import ABC, abstractmethod


# ============================================================
# CFG Data Structures
# ============================================================

@dataclass
class Instruction:
    """A three-address code instruction."""
    index: int
    result: str = ""
    op: str = ""
    arg1: str = ""
    arg2: str = ""
    label: str = ""

    @property
    def defined_var(self) -> str:
        """Variable defined by this instruction."""
        if self.op and self.result and self.op not in ("goto", "iffalse",
                                                        "iftrue", "label",
                                                        "return"):
            return self.result
        return ""

    @property
    def used_vars(self) -> set:
        """Variables used by this instruction."""
        used = set()
        for arg in (self.arg1, self.arg2):
            if arg and not arg.isdigit() and not arg.startswith("L") \
               and not arg.startswith("#"):
                used.add(arg)
        return used

    def __str__(self):
        if self.op == "label":
            return f"{self.result}:"
        elif self.op == "copy":
            return f"{self.result} = {self.arg1}"
        elif self.arg2:
            return f"{self.result} = {self.arg1} {self.op} {self.arg2}"
        elif self.op in ("goto", "iffalse", "iftrue"):
            return f"{self.op} {self.arg1} {self.result}"
        elif self.op == "return":
            return f"return {self.arg1}"
        else:
            return f"{self.result} = {self.op} {self.arg1}"


@dataclass
class BasicBlock:
    """A basic block in the CFG."""
    id: int
    instructions: list = field(default_factory=list)
    successors: list = field(default_factory=list)
    predecessors: list = field(default_factory=list)

    def __str__(self):
        lines = [f"B{self.id}:"]
        for instr in self.instructions:
            lines.append(f"  {instr}")
        return "\n".join(lines)


class CFG:
    """Control flow graph."""

    def __init__(self):
        self.blocks: dict[int, BasicBlock] = {}

    def add_block(self, block: BasicBlock):
        self.blocks[block.id] = block

    def add_edge(self, from_id: int, to_id: int):
        if to_id not in self.blocks[from_id].successors:
            self.blocks[from_id].successors.append(to_id)
        if from_id not in self.blocks[to_id].predecessors:
            self.blocks[to_id].predecessors.append(from_id)


# ============================================================
# Data Flow Analysis Framework
# ============================================================

class Direction(Enum):
    FORWARD = auto()
    BACKWARD = auto()


class MeetOp(Enum):
    UNION = auto()        # May analysis
    INTERSECTION = auto()  # Must analysis


class DataFlowAnalysis(ABC):
    """
    Abstract base class for data flow analyses.
    """

    def __init__(self, cfg: CFG, direction: Direction, meet_op: MeetOp):
        self.cfg = cfg
        self.direction = direction
        self.meet_op = meet_op
        self.in_sets: dict[int, set] = {}
        self.out_sets: dict[int, set] = {}

    @abstractmethod
    def gen(self, block: BasicBlock) -> set:
        """Compute the Gen set for a block."""
        pass

    @abstractmethod
    def kill(self, block: BasicBlock) -> set:
        """Compute the Kill set for a block."""
        pass

    @abstractmethod
    def universe(self) -> set:
        """Return the universal set for must-analyses."""
        pass

    @abstractmethod
    def boundary_value(self) -> set:
        """Return the boundary value (for entry/exit block)."""
        pass

    def meet(self, sets: list[set]) -> set:
        """Apply the meet operator to a list of sets."""
        if not sets:
            if self.meet_op == MeetOp.UNION:
                return set()
            else:
                return self.universe()

        result = sets[0].copy()
        for s in sets[1:]:
            if self.meet_op == MeetOp.UNION:
                result = result | s
            else:
                result = result & s
        return result

    def transfer(self, block: BasicBlock, input_set: set) -> set:
        """Apply the transfer function: Gen ∪ (input - Kill)."""
        return self.gen(block) | (input_set - self.kill(block))

    def initial_value(self) -> set:
        """Initial value for non-boundary blocks."""
        if self.meet_op == MeetOp.UNION:
            return set()       # Most optimistic for may-analyses
        else:
            return self.universe()  # Most optimistic for must-analyses

    def solve(self):
        """
        Solve the data flow equations using the worklist algorithm.
        """
        block_ids = sorted(self.cfg.blocks.keys())

        if self.direction == Direction.FORWARD:
            self._solve_forward(block_ids)
        else:
            self._solve_backward(block_ids)

    def _solve_forward(self, block_ids):
        """Forward worklist algorithm."""
        entry_id = block_ids[0]

        # Initialize
        for bid in block_ids:
            self.in_sets[bid] = set()
            if bid == entry_id:
                self.out_sets[bid] = self.boundary_value()
            else:
                self.out_sets[bid] = self.initial_value()

        # Worklist (use reverse postorder for efficiency)
        worklist = deque(block_ids[1:])  # Exclude entry initially
        in_worklist = set(worklist)

        while worklist:
            bid = worklist.popleft()
            in_worklist.discard(bid)
            block = self.cfg.blocks[bid]

            # Compute In
            pred_outs = [self.out_sets[p] for p in block.predecessors]
            new_in = self.meet(pred_outs)
            self.in_sets[bid] = new_in

            # Compute Out via transfer function
            new_out = self.transfer(block, new_in)

            if new_out != self.out_sets[bid]:
                self.out_sets[bid] = new_out
                for succ in block.successors:
                    if succ not in in_worklist:
                        worklist.append(succ)
                        in_worklist.add(succ)

    def _solve_backward(self, block_ids):
        """Backward worklist algorithm."""
        # Find exit blocks (no successors)
        exit_ids = [bid for bid in block_ids
                    if not self.cfg.blocks[bid].successors]

        # Initialize
        for bid in block_ids:
            self.out_sets[bid] = set()
            if bid in exit_ids:
                self.in_sets[bid] = self.boundary_value()
            else:
                self.in_sets[bid] = self.initial_value()

        # Worklist
        worklist = deque(bid for bid in block_ids if bid not in exit_ids)
        in_worklist = set(worklist)

        # Also add exit blocks to process their predecessors
        for eid in exit_ids:
            block = self.cfg.blocks[eid]
            self.out_sets[eid] = set()
            self.in_sets[eid] = self.transfer(block, self.out_sets[eid])

        while worklist:
            bid = worklist.popleft()
            in_worklist.discard(bid)
            block = self.cfg.blocks[bid]

            # Compute Out from successors
            succ_ins = [self.in_sets[s] for s in block.successors]
            new_out = self.meet(succ_ins)
            self.out_sets[bid] = new_out

            # Compute In via transfer function
            new_in = self.transfer(block, new_out)

            if new_in != self.in_sets[bid]:
                self.in_sets[bid] = new_in
                for pred in block.predecessors:
                    if pred not in in_worklist:
                        worklist.append(pred)
                        in_worklist.add(pred)

    def print_results(self, analysis_name: str):
        """Pretty-print the analysis results."""
        print(f"\n=== {analysis_name} ===")
        for bid in sorted(self.cfg.blocks.keys()):
            in_s = sorted(self.in_sets.get(bid, set()))
            out_s = sorted(self.out_sets.get(bid, set()))
            print(f"  B{bid}: In = {{{', '.join(str(x) for x in in_s)}}}")
            print(f"       Out = {{{', '.join(str(x) for x in out_s)}}}")


# ============================================================
# Concrete Analyses
# ============================================================

class ReachingDefinitions(DataFlowAnalysis):
    """Reaching Definitions analysis."""

    def __init__(self, cfg: CFG):
        super().__init__(cfg, Direction.FORWARD, MeetOp.UNION)
        # Pre-compute all definitions
        self.all_defs: dict[str, set] = defaultdict(set)  # var -> {def labels}
        for bid, block in cfg.blocks.items():
            for instr in block.instructions:
                dvar = instr.defined_var
                if dvar:
                    def_label = f"d{instr.index}:{dvar}"
                    self.all_defs[dvar].add(def_label)

    def gen(self, block: BasicBlock) -> set:
        gen_set = set()
        killed_in_block = set()  # Track vars defined in this block
        # Process in order; last def of each var survives
        for instr in block.instructions:
            dvar = instr.defined_var
            if dvar:
                # Kill previous defs of this var in this block
                gen_set = {d for d in gen_set
                           if not d.endswith(f":{dvar}")}
                gen_set.add(f"d{instr.index}:{dvar}")
        return gen_set

    def kill(self, block: BasicBlock) -> set:
        kill_set = set()
        for instr in block.instructions:
            dvar = instr.defined_var
            if dvar:
                # Kill all other defs of this var
                for d in self.all_defs[dvar]:
                    if not d.startswith(f"d{instr.index}:"):
                        kill_set.add(d)
        return kill_set

    def universe(self) -> set:
        return set().union(*self.all_defs.values()) if self.all_defs else set()

    def boundary_value(self) -> set:
        return set()


class LiveVariables(DataFlowAnalysis):
    """Live Variables analysis."""

    def __init__(self, cfg: CFG, exit_live: set = None):
        super().__init__(cfg, Direction.BACKWARD, MeetOp.UNION)
        self.exit_live = exit_live or set()

    def gen(self, block: BasicBlock) -> set:
        """Use set: variables used before being defined in this block."""
        use_set = set()
        defined = set()
        for instr in block.instructions:
            # Uses that haven't been defined yet in this block
            for v in instr.used_vars:
                if v not in defined:
                    use_set.add(v)
            # Definition
            dvar = instr.defined_var
            if dvar:
                defined.add(dvar)
        return use_set

    def kill(self, block: BasicBlock) -> set:
        """Def set: variables defined in this block."""
        def_set = set()
        for instr in block.instructions:
            dvar = instr.defined_var
            if dvar:
                def_set.add(dvar)
        return def_set

    def universe(self) -> set:
        all_vars = set()
        for block in self.cfg.blocks.values():
            for instr in block.instructions:
                all_vars |= instr.used_vars
                if instr.defined_var:
                    all_vars.add(instr.defined_var)
        return all_vars

    def boundary_value(self) -> set:
        return self.exit_live


class AvailableExpressions(DataFlowAnalysis):
    """Available Expressions analysis."""

    def __init__(self, cfg: CFG):
        super().__init__(cfg, Direction.FORWARD, MeetOp.INTERSECTION)
        # Collect all expressions
        self.all_exprs = set()
        for block in cfg.blocks.values():
            for instr in block.instructions:
                if instr.op in ('+', '-', '*', '/') and instr.arg1 and instr.arg2:
                    self.all_exprs.add(f"{instr.arg1} {instr.op} {instr.arg2}")

    def gen(self, block: BasicBlock) -> set:
        gen_set = set()
        killed_vars = set()  # Variables defined so far
        for instr in block.instructions:
            # Add expression if operands haven't been killed
            if instr.op in ('+', '-', '*', '/') and instr.arg1 and instr.arg2:
                expr = f"{instr.arg1} {instr.op} {instr.arg2}"
                if instr.arg1 not in killed_vars and instr.arg2 not in killed_vars:
                    gen_set.add(expr)
            # Kill expressions involving the defined variable
            dvar = instr.defined_var
            if dvar:
                gen_set = {e for e in gen_set
                           if dvar not in e.split()}
                killed_vars.add(dvar)
        return gen_set

    def kill(self, block: BasicBlock) -> set:
        kill_set = set()
        for instr in block.instructions:
            dvar = instr.defined_var
            if dvar:
                for expr in self.all_exprs:
                    if dvar in expr.split():
                        kill_set.add(expr)
        return kill_set

    def universe(self) -> set:
        return set(self.all_exprs)

    def boundary_value(self) -> set:
        return set()


class VeryBusyExpressions(DataFlowAnalysis):
    """Very Busy Expressions analysis."""

    def __init__(self, cfg: CFG):
        super().__init__(cfg, Direction.BACKWARD, MeetOp.INTERSECTION)
        self.all_exprs = set()
        for block in cfg.blocks.values():
            for instr in block.instructions:
                if instr.op in ('+', '-', '*', '/') and instr.arg1 and instr.arg2:
                    self.all_exprs.add(f"{instr.arg1} {instr.op} {instr.arg2}")

    def gen(self, block: BasicBlock) -> set:
        """Expressions computed in this block before any operand is redefined."""
        gen_set = set()
        killed_vars = set()
        for instr in block.instructions:
            # Check if this computes an expression with unkilled operands
            if instr.op in ('+', '-', '*', '/') and instr.arg1 and instr.arg2:
                expr = f"{instr.arg1} {instr.op} {instr.arg2}"
                if instr.arg1 not in killed_vars and instr.arg2 not in killed_vars:
                    gen_set.add(expr)
            # Track defined variables
            dvar = instr.defined_var
            if dvar:
                killed_vars.add(dvar)
        return gen_set

    def kill(self, block: BasicBlock) -> set:
        kill_set = set()
        for instr in block.instructions:
            dvar = instr.defined_var
            if dvar:
                for expr in self.all_exprs:
                    if dvar in expr.split():
                        kill_set.add(expr)
        return kill_set

    def universe(self) -> set:
        return set(self.all_exprs)

    def boundary_value(self) -> set:
        return set()


# ============================================================
# Example
# ============================================================

def build_example_cfg() -> CFG:
    """
    Build example CFG:

    B0: d0: x = 5
        d1: y = 1

    B1: d2: z = x + y
        if z < 10 goto B2

    B2: d3: x = x + 1
        d4: y = y * 2
        goto B1

    B3: d5: w = x + y
        return w
    """
    cfg = CFG()

    b0 = BasicBlock(0, instructions=[
        Instruction(0, "x", "copy", "5"),
        Instruction(1, "y", "copy", "1"),
    ])

    b1 = BasicBlock(1, instructions=[
        Instruction(2, "z", "+", "x", "y"),
        Instruction(3, "", "iffalse", "z < 10", result="B3"),
    ])

    b2 = BasicBlock(2, instructions=[
        Instruction(4, "x", "+", "x", "1"),
        Instruction(5, "y", "*", "y", "2"),
    ])

    b3 = BasicBlock(3, instructions=[
        Instruction(6, "w", "+", "x", "y"),
        Instruction(7, "", "return", "w"),
    ])

    cfg.add_block(b0)
    cfg.add_block(b1)
    cfg.add_block(b2)
    cfg.add_block(b3)

    cfg.add_edge(0, 1)
    cfg.add_edge(1, 2)  # z < 10
    cfg.add_edge(1, 3)  # z >= 10
    cfg.add_edge(2, 1)  # loop back

    return cfg


def demo_all_analyses():
    """Run all four data flow analyses on the example CFG."""
    cfg = build_example_cfg()

    print("=== Control Flow Graph ===")
    for bid in sorted(cfg.blocks.keys()):
        block = cfg.blocks[bid]
        print(f"\nB{bid}: (pred={block.predecessors}, succ={block.successors})")
        for instr in block.instructions:
            print(f"  d{instr.index}: {instr}")

    # 1. Reaching Definitions
    rd = ReachingDefinitions(cfg)
    rd.solve()
    rd.print_results("Reaching Definitions")

    # 2. Live Variables
    lv = LiveVariables(cfg, exit_live={"w"})
    lv.solve()
    lv.print_results("Live Variables")

    # 3. Available Expressions
    ae = AvailableExpressions(cfg)
    ae.solve()
    ae.print_results("Available Expressions")

    # 4. Very Busy Expressions
    vb = VeryBusyExpressions(cfg)
    vb.solve()
    vb.print_results("Very Busy Expressions")

    # Print Gen/Kill sets for each analysis
    print("\n=== Gen/Kill Sets ===")
    for bid in sorted(cfg.blocks.keys()):
        block = cfg.blocks[bid]
        print(f"\nB{bid}:")
        print(f"  RD:  Gen={sorted(rd.gen(block))}, Kill={sorted(rd.kill(block))}")
        print(f"  LV:  Use={sorted(lv.gen(block))}, Def={sorted(lv.kill(block))}")
        print(f"  AE:  Gen={sorted(ae.gen(block))}, Kill={sorted(ae.kill(block))}")
        print(f"  VBE: Gen={sorted(vb.gen(block))}, Kill={sorted(vb.kill(block))}")


if __name__ == "__main__":
    demo_all_analyses()
```

---

## 6. Applying Analysis Results to Optimization

### 6.1 Global Constant Propagation

Using **reaching definitions**, we can perform global constant propagation:

```
For each use of variable v at point p:
    Let D = set of reaching definitions of v at p
    If |D| == 1 and that definition assigns a constant c:
        Replace v with c at p
```

**Example**:
```
B1: x = 5         // d1

B2: y = x + 1     // Reaching defs of x: {d1}
                   // d1 assigns constant 5
                   // → y = 5 + 1 → y = 6
```

### 6.2 Global Dead Code Elimination

Using **live variables**, we eliminate definitions of variables that are not live after the definition:

```
For each definition d: v = expr at point p:
    If v is NOT in LiveOut at point p:
        Remove d (it's dead code)
```

### 6.3 Global Common Subexpression Elimination

Using **available expressions**, we eliminate redundant computations:

```
For each computation of expression e at point p:
    If e is in the Available set at p:
        Replace the computation with a reference to
        the previously computed value

Note: We need to insert a temporary at the original computation
      to store the value for later use.
```

**Before**:
```
B1: t1 = a + b      // computes a + b
    ...

B2: ...              // no redefinition of a or b

B3: t2 = a + b      // a + b is available here (from B1)
```

**After**:
```
B1: t1 = a + b      // a + b stored in t1
    ...

B3: t2 = t1         // reuse previously computed value
```

### 6.4 Code Hoisting

Using **very busy expressions**, we hoist computations to earlier points:

```
If expression e is very busy at the entry of block B:
    Compute e at the entry of B (or at the end of B's predecessors)
    Store the result in a temporary
    Replace all subsequent computations of e with the temporary
```

**Before**:
```
B1:
    if (cond) goto B2 else goto B3

B2: t1 = a + b      // a + b computed on this path
    ...

B3: t2 = a + b      // a + b computed on this path too
    ...
```

If `a + b` is very busy at the exit of B1 (evaluated on all paths), we can hoist:

**After**:
```
B1: t0 = a + b      // hoisted
    if (cond) goto B2 else goto B3

B2: t1 = t0         // reuse
    ...

B3: t2 = t0         // reuse
    ...
```

---

## 7. Advanced Topics

### 7.1 Iterative Analysis Ordering

The order in which blocks are processed affects the number of iterations:

| Direction | Best ordering | Why |
|-----------|--------------|-----|
| Forward | Reverse postorder | Ensures predecessors are processed before successors (except back edges) |
| Backward | Postorder | Ensures successors are processed before predecessors (except back edges) |

Using the optimal ordering, most analyses converge in 2--3 passes (one pass handles the acyclic part, additional passes handle loops).

### 7.2 SSA-Based Analysis

In SSA form, many data flow analyses become simpler:

- **Reaching definitions**: Trivial -- each variable has exactly one definition
- **Def-use chains**: Immediately available from SSA structure
- **Constant propagation**: Sparse conditional constant propagation (SCCP) operates directly on SSA

### 7.3 Interprocedural Analysis

Extending analyses across function boundaries requires:

1. **Call graph construction**: Which functions call which?
2. **Context sensitivity**: Distinguishing different calling contexts
3. **Summary functions**: Computing the effect of a function call without analyzing its body each time

Interprocedural analysis enables:
- Global constant propagation across function calls
- Alias analysis (which pointers might point to the same memory?)
- Escape analysis (does an object escape its creating function?)

### 7.4 Widening and Narrowing

For analyses over infinite lattices (e.g., numerical ranges), iteration may not converge. **Widening** accelerates convergence by jumping to higher lattice elements:

$$X_{n+1} = X_n \nabla f(X_n)$$

where $\nabla$ is the widening operator. After convergence, **narrowing** refines the result:

$$X_{n+1} = X_n \Delta f(X_n)$$

This is used in abstract interpretation frameworks like those behind the Astrée static analyzer.

---

## 8. Combining Optimizations: A Complete Pass

Here is how a compiler might chain local and global optimizations:

```python
"""
Example optimization pipeline combining local and global passes.
"""


def optimization_pipeline(cfg):
    """
    Run a sequence of optimization passes on a CFG.

    Pipeline:
    1. Global constant propagation (using reaching definitions)
    2. Local constant folding (within each block)
    3. Global CSE (using available expressions)
    4. Local copy propagation (within each block)
    5. Global DCE (using live variables)
    6. Local algebraic simplification
    7. Repeat if anything changed
    """
    changed = True
    iteration = 0

    while changed:
        changed = False
        iteration += 1
        print(f"\n--- Optimization Pass {iteration} ---")

        # 1. Global: Reaching definitions → constant propagation
        rd = ReachingDefinitions(cfg)
        rd.solve()
        if apply_global_constant_propagation(cfg, rd):
            changed = True
            print("  Applied global constant propagation")

        # 2. Local: Constant folding in each block
        for block in cfg.blocks.values():
            if apply_local_constant_folding(block):
                changed = True
                print(f"  Applied constant folding in B{block.id}")

        # 3. Global: Available expressions → CSE
        ae = AvailableExpressions(cfg)
        ae.solve()
        if apply_global_cse(cfg, ae):
            changed = True
            print("  Applied global CSE")

        # 4. Local: Copy propagation in each block
        for block in cfg.blocks.values():
            if apply_local_copy_propagation(block):
                changed = True
                print(f"  Applied copy propagation in B{block.id}")

        # 5. Global: Live variables → DCE
        lv = LiveVariables(cfg)
        lv.solve()
        if apply_global_dce(cfg, lv):
            changed = True
            print("  Applied global DCE")

        # 6. Local: Algebraic simplification
        for block in cfg.blocks.values():
            if apply_local_algebraic_simplification(block):
                changed = True
                print(f"  Applied algebraic simplification in B{block.id}")

    print(f"\nOptimization converged after {iteration} iterations")
```

Note: The functions `apply_global_constant_propagation`, etc., would implement the transformations described in Section 6, using the analysis results to determine where to apply changes. The above code shows the overall structure of an optimization pipeline.

---

## 9. Summary

In this lesson, we covered the theory and practice of compiler optimization:

1. **Optimization principles**: Every optimization must be safe (preserves semantics), profitable (improves the code), and must detect opportunities to apply.

2. **Local optimizations** operate within a single basic block and include:
   - **Constant folding**: Evaluate constant expressions at compile time
   - **Constant propagation**: Replace variables with their constant values
   - **Algebraic simplification**: Apply mathematical identities
   - **Strength reduction**: Replace expensive operations with cheaper ones
   - **CSE**: Eliminate redundant computations
   - **Copy propagation**: Replace copies with original values
   - **Dead code elimination**: Remove unused computations

3. **Global data flow analysis** reasons about information flow across the entire CFG:
   - **Reaching definitions** (forward, union): Which definitions reach a point?
   - **Available expressions** (forward, intersection): Which expressions are computed on all paths?
   - **Live variables** (backward, union): Which variables are needed later?
   - **Very busy expressions** (backward, intersection): Which expressions are evaluated on all paths?

4. **The data flow framework** unifies all analyses using lattice theory, monotone transfer functions, and fixed-point iteration. The **worklist algorithm** provides an efficient implementation.

5. Analysis results enable global optimizations: global constant propagation, global CSE, global DCE, and code hoisting.

These techniques form the core of the "middle end" of a compiler, transforming the IR to produce faster, smaller code before the back end generates machine instructions.

---

## Exercises

### Exercise 1: Local Optimization

Apply all local optimizations (in the order: constant propagation, constant folding, algebraic simplification, copy propagation, dead code elimination) to the following basic block. Show the result after each pass.

```
t1 = 4
t2 = 8
t3 = t1 * t2
t4 = t3 + 0
t5 = t4
t6 = t5 * 1
t7 = t6 / 2
t8 = t7 + t7
unused = t1 + t2
result = t8
```

Assume `result` is live out.

### Exercise 2: Reaching Definitions

Compute the reaching definitions for each block in the following CFG. Show the Gen, Kill, In, and Out sets.

```
B0: d0: a = 1
    d1: b = 2

B1: d2: c = a + b
    if c > 10 goto B3

B2: d3: a = a + 1
    d4: b = b - 1
    goto B1

B3: d5: d = a * b
    return d
```

### Exercise 3: Live Variables

Compute the live variables at the entry and exit of each block for the CFG in Exercise 2. Assume `d` is live at the exit of B3.

### Exercise 4: Available Expressions

For the following CFG, compute the available expressions at each block entry:

```
B0: t1 = a + b
    t2 = c + d

B1: t3 = a + b      (is a+b available here?)
    if (t3 > 0) goto B2 else goto B3

B2: a = a + 1        (kills a+b)
    t4 = c + d
    goto B4

B3: t5 = a + b
    t6 = c + d
    goto B4

B4: t7 = a + b       (is a+b available here?)
    t8 = c + d       (is c+d available here?)
```

### Exercise 5: Worklist Algorithm Trace

Trace the execution of the worklist algorithm for the live variables analysis on the following CFG. Show the worklist contents, and the In/Out sets at each step.

```
B0: x = read()
    y = read()

B1: if (x > 0) goto B2 else goto B3

B2: z = x + y
    x = x - 1
    goto B1

B3: print z
```

### Exercise 6: Implementation Challenge

Extend the data flow analysis framework to implement:

1. **Constant propagation analysis**: A forward analysis where the lattice element for each variable is one of {$\top$, constant $c$, $\bot$}. The meet of two different constants is $\bot$.

2. **Copy propagation analysis**: Determine which copies `x = y` are still valid (not killed) at each program point.

Test both analyses on the CFG from Exercise 2 and show how they enable further optimization.

---

[Previous: Code Generation](./11_Code_Generation.md) | [Next: Loop Optimization](./13_Loop_Optimization.md) | [Overview](./00_Overview.md)
