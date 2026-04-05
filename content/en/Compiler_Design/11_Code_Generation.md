# Lesson 11: Code Generation

## Learning Objectives

After completing this lesson, you will be able to:

1. **Describe** the target machine model used for code generation
2. **Explain** instruction selection via tree pattern matching and tiling
3. **Implement** the Maximal Munch algorithm for instruction selection
4. **Apply** register allocation techniques: graph coloring and linear scan
5. **Understand** instruction scheduling (list scheduling, software pipelining overview)
6. **Perform** peephole optimization on generated code
7. **Generate** code for expressions, control flow, and function calls
8. **Implement** a complete code generator for a stack machine in Python

---

## 1. Target Machine Model

### 1.1 Why a Machine Model?

Code generation translates the intermediate representation (IR) into instructions for a specific target machine. To study code generation techniques in a machine-independent way, we define an abstract target machine model that captures the essential features of real architectures.

### 1.2 A Simple Target Machine

Our model machine has the following characteristics:

| Feature | Description |
|---------|-------------|
| Registers | $R_0, R_1, \ldots, R_{k-1}$ (general purpose) |
| Memory | Byte-addressable, word size = 4 bytes |
| Instructions | Three-address form: `op dst, src1, src2` |
| Addressing modes | Register, immediate, register-indirect, indexed |
| Stack | Grows downward, SP register |

### 1.3 Addressing Modes

| Mode | Syntax | Meaning | Use Case |
|------|--------|---------|----------|
| Register | `R0` | Value in register R0 | Local variables in registers |
| Immediate | `#42` | Constant value 42 | Constants, offsets |
| Register-indirect | `[R0]` | Memory at address in R0 | Pointer dereference |
| Indexed | `[R0 + #8]` | Memory at R0 + 8 | Array access, struct fields |
| Direct | `[addr]` | Memory at absolute address | Global variables |

### 1.4 Instruction Set

```
Arithmetic:
    ADD  Rd, Rs1, Rs2    ; Rd = Rs1 + Rs2
    SUB  Rd, Rs1, Rs2    ; Rd = Rs1 - Rs2
    MUL  Rd, Rs1, Rs2    ; Rd = Rs1 * Rs2
    DIV  Rd, Rs1, Rs2    ; Rd = Rs1 / Rs2
    ADDI Rd, Rs, #imm    ; Rd = Rs + imm

Data movement:
    MOV  Rd, Rs          ; Rd = Rs
    MOVI Rd, #imm        ; Rd = imm (load immediate)
    LOAD Rd, [Rs + #off] ; Rd = Mem[Rs + off]
    STORE Rs, [Rd + #off]; Mem[Rd + off] = Rs

Control flow:
    CMP  Rs1, Rs2        ; Set condition flags
    BEQ  label           ; Branch if equal
    BNE  label           ; Branch if not equal
    BLT  label           ; Branch if less than
    BGT  label           ; Branch if greater than
    BLE  label           ; Branch if less or equal
    BGE  label           ; Branch if greater or equal
    JMP  label           ; Unconditional jump
    CALL label           ; Function call (push return address, jump)
    RET                  ; Return (pop return address, jump)

Stack:
    PUSH Rs              ; Push register onto stack
    POP  Rd              ; Pop from stack into register
```

### 1.5 Instruction Costs

Different instructions have different costs (in cycles):

| Instruction | Cost | Notes |
|-------------|------|-------|
| ADD, SUB, MOV | 1 | Register-register operations |
| MUL | 3 | Multiplication is slower |
| DIV | 10-40 | Division is very expensive |
| LOAD | 4 (L1 hit) | Memory access (cache dependent) |
| STORE | 1 (buffered) | Write buffer hides latency |
| Branch (taken) | 2 | Pipeline flush penalty |
| Branch (not taken) | 1 | Predicted correctly |

The code generator should prefer cheaper instructions when possible (e.g., using shifts instead of multiply by powers of 2).

---

## 2. Instruction Selection

### 2.1 The Problem

Instruction selection maps IR operations to target machine instructions. The challenge is that:

1. A single IR operation may correspond to multiple machine instructions
2. A single machine instruction may cover multiple IR operations
3. Different instruction sequences may compute the same result with different costs
4. The optimal choice depends on context (register availability, surrounding instructions)

### 2.2 Tree Pattern Matching

Many architectures have instructions that can perform compound operations (e.g., `LOAD R0, [R1 + R2*4 + #8]` performs an addition, a multiplication, and a memory load in one instruction).

The idea of **tree pattern matching** is to:

1. Represent the IR as expression trees
2. Define **tiles** -- tree patterns that correspond to single machine instructions
3. **Cover** the IR tree with non-overlapping tiles that minimize total cost

### 2.3 Tiles

A **tile** is a tree pattern paired with a machine instruction and a cost.

Example tiles for our target machine:

```
Tile 1: register                    Tile 2: immediate
Pattern:   reg                      Pattern:   #imm
Instr:     (none, already in reg)   Instr:     MOVI Rd, #imm
Cost:      0                        Cost:      1

Tile 3: add                         Tile 4: add immediate
Pattern:     +                      Pattern:     +
            / \                                 / \
          reg  reg                            reg  #imm
Instr:   ADD Rd, Rs1, Rs2          Instr:   ADDI Rd, Rs, #imm
Cost:    1                          Cost:    1

Tile 5: load                        Tile 6: indexed load
Pattern:   MEM                      Pattern:   MEM
            |                                   |
           reg                                  +
                                               / \
Instr:   LOAD Rd, [Rs]                       reg  #imm
Cost:    4                          Instr:   LOAD Rd, [Rs + #imm]
                                    Cost:    4

Tile 7: store                       Tile 8: indexed store
Pattern:  STORE                     Pattern:  STORE
          /   \                               /    \
        addr   reg                           +      reg
                                            / \
Instr:  STORE Rs, [Rd]                    reg  #imm
Cost:   1                           Instr:  STORE Rs, [Rd + #imm]
                                    Cost:   1
```

### 2.4 Optimal Tiling vs Maximal Munch

There are two main strategies for covering the tree with tiles:

**Optimal Tiling**: Find the set of tiles that covers the entire tree with minimum total cost. This can be solved with dynamic programming bottom-up on the tree.

**Maximal Munch** (greedy): At each node, select the largest (most-covering) tile that matches, starting from the root. This is simpler but may not find the globally optimal solution.

In practice, Maximal Munch produces results that are close to optimal and is widely used.

### 2.5 The Maximal Munch Algorithm

```
function maximal_munch(node):
    Find the largest tile that matches at this node
    (matching means the tile pattern matches the subtree rooted here)

    For each leaf of the selected tile that corresponds to a
    subtree (not yet covered):
        Recursively apply maximal_munch to that subtree

    Emit the instruction associated with the selected tile
```

The algorithm works **top-down**: it matches the largest pattern at the root, then recursively processes the remaining subtrees.

**Example**: Generate code for `a[i] = b + 1`

IR tree:
```
        STORE
       /     \
      +       +
     / \     / \
    a   *   b   1
       / \
      i   4
```

Maximal Munch might match:
1. At root: STORE tile covering `STORE(addr, value)` -- emits STORE instruction
2. Left child `+`: ADD tile for `a + (i * 4)` or indexed addressing
3. Right child `+`: ADDI tile for `b + 1`

### 2.6 Python Implementation: Maximal Munch

```python
"""Instruction selection via Maximal Munch algorithm."""

from dataclasses import dataclass, field
from typing import Optional


# ---------- IR Tree Nodes ----------

@dataclass
class IRNode:
    """Base class for IR tree nodes."""
    pass

@dataclass
class Const(IRNode):
    """Integer constant."""
    value: int

@dataclass
class Reg(IRNode):
    """Virtual register reference."""
    name: str

@dataclass
class BinOp(IRNode):
    """Binary operation."""
    op: str        # '+', '-', '*', '/'
    left: IRNode
    right: IRNode

@dataclass
class Mem(IRNode):
    """Memory access (load)."""
    address: IRNode

@dataclass
class Store(IRNode):
    """Memory store."""
    address: IRNode
    value: IRNode

@dataclass
class CJump(IRNode):
    """Conditional jump."""
    op: str           # '<', '>', '==', '!=', '<=', '>='
    left: IRNode
    right: IRNode
    true_label: str
    false_label: str


# ---------- Machine Instructions ----------

@dataclass
class MachineInstr:
    """A target machine instruction."""
    opcode: str
    operands: list = field(default_factory=list)
    comment: str = ""

    def __str__(self):
        ops = ", ".join(str(o) for o in self.operands)
        comment = f"  ; {self.comment}" if self.comment else ""
        return f"    {self.opcode:6s} {ops}{comment}"


# ---------- Maximal Munch Code Generator ----------

class MaximalMunchCodeGen:
    """
    Instruction selection using the Maximal Munch (greedy) algorithm.
    Generates code for a simple RISC-like target.
    """

    def __init__(self):
        self.instructions: list[MachineInstr] = []
        self._reg_counter = 0

    def new_reg(self) -> str:
        """Allocate a new virtual register."""
        self._reg_counter += 1
        return f"v{self._reg_counter}"

    def emit(self, opcode: str, operands: list = None,
             comment: str = "") -> MachineInstr:
        """Emit a machine instruction."""
        instr = MachineInstr(opcode, operands or [], comment)
        self.instructions.append(instr)
        return instr

    def munch_expr(self, node: IRNode) -> str:
        """
        Generate code for an expression node using Maximal Munch.
        Returns the register holding the result.
        """

        # --- Tile: Constant ---
        if isinstance(node, Const):
            rd = self.new_reg()
            self.emit("MOVI", [rd, f"#{node.value}"],
                      f"load constant {node.value}")
            return rd

        # --- Tile: Register ---
        if isinstance(node, Reg):
            return node.name

        # --- Tile: Memory load with indexed addressing ---
        # Pattern: MEM(BinOp(+, e1, Const(n)))
        if isinstance(node, Mem) and isinstance(node.address, BinOp) \
           and node.address.op == '+' and isinstance(node.address.right, Const):
            base_reg = self.munch_expr(node.address.left)
            offset = node.address.right.value
            rd = self.new_reg()
            self.emit("LOAD", [rd, f"[{base_reg} + #{offset}]"],
                      f"indexed load")
            return rd

        # --- Tile: Memory load with register addressing ---
        # Pattern: MEM(e)
        if isinstance(node, Mem):
            addr_reg = self.munch_expr(node.address)
            rd = self.new_reg()
            self.emit("LOAD", [rd, f"[{addr_reg}]"],
                      "register-indirect load")
            return rd

        # --- Tile: Add immediate ---
        # Pattern: BinOp(+, e, Const(n))
        if isinstance(node, BinOp) and node.op == '+' \
           and isinstance(node.right, Const):
            rs = self.munch_expr(node.left)
            rd = self.new_reg()
            self.emit("ADDI", [rd, rs, f"#{node.right.value}"],
                      f"add immediate {node.right.value}")
            return rd

        # --- Tile: Add immediate (commuted) ---
        # Pattern: BinOp(+, Const(n), e)
        if isinstance(node, BinOp) and node.op == '+' \
           and isinstance(node.left, Const):
            rs = self.munch_expr(node.right)
            rd = self.new_reg()
            self.emit("ADDI", [rd, rs, f"#{node.left.value}"],
                      f"add immediate {node.left.value}")
            return rd

        # --- Tile: Multiply by power of 2 → shift ---
        if isinstance(node, BinOp) and node.op == '*' \
           and isinstance(node.right, Const) \
           and node.right.value > 0 \
           and (node.right.value & (node.right.value - 1)) == 0:
            rs = self.munch_expr(node.left)
            shift = node.right.value.bit_length() - 1
            rd = self.new_reg()
            self.emit("SHL", [rd, rs, f"#{shift}"],
                      f"multiply by {node.right.value} via shift")
            return rd

        # --- Tile: General binary operation ---
        if isinstance(node, BinOp):
            rs1 = self.munch_expr(node.left)
            rs2 = self.munch_expr(node.right)
            rd = self.new_reg()
            op_map = {'+': 'ADD', '-': 'SUB', '*': 'MUL', '/': 'DIV'}
            opcode = op_map.get(node.op, 'BINOP')
            self.emit(opcode, [rd, rs1, rs2],
                      f"{node.op} operation")
            return rd

        raise ValueError(f"Cannot generate code for: {type(node)}")

    def munch_stmt(self, node: IRNode):
        """Generate code for a statement node."""

        # --- Tile: Indexed store ---
        # Pattern: Store(BinOp(+, e1, Const(n)), e2)
        if isinstance(node, Store) and isinstance(node.address, BinOp) \
           and node.address.op == '+' \
           and isinstance(node.address.right, Const):
            base_reg = self.munch_expr(node.address.left)
            val_reg = self.munch_expr(node.value)
            offset = node.address.right.value
            self.emit("STORE", [val_reg, f"[{base_reg} + #{offset}]"],
                      "indexed store")
            return

        # --- Tile: General store ---
        if isinstance(node, Store):
            addr_reg = self.munch_expr(node.address)
            val_reg = self.munch_expr(node.value)
            self.emit("STORE", [val_reg, f"[{addr_reg}]"],
                      "register-indirect store")
            return

        # --- Tile: Conditional jump ---
        if isinstance(node, CJump):
            rs1 = self.munch_expr(node.left)
            rs2 = self.munch_expr(node.right)
            self.emit("CMP", [rs1, rs2],
                      f"compare for {node.op}")
            branch_map = {
                '<': 'BLT', '>': 'BGT', '==': 'BEQ',
                '!=': 'BNE', '<=': 'BLE', '>=': 'BGE'
            }
            opcode = branch_map.get(node.op, 'BR')
            self.emit(opcode, [node.true_label],
                      f"branch to {node.true_label}")
            self.emit("JMP", [node.false_label],
                      f"fall-through to {node.false_label}")
            return

        # Expression statement: just evaluate for side effects
        self.munch_expr(node)

    def print_code(self):
        """Print all generated instructions."""
        for instr in self.instructions:
            print(instr)


# ---------- Examples ----------

def demo_expression():
    """Generate code for: result = a[i*4 + 8] + 1"""
    print("=== Expression: a[i*4 + 8] + 1 ===\n")

    # IR tree: BinOp(+, Mem(BinOp(+, BinOp(*, i, 4), 8)), 1)
    # Simplified: Mem(BinOp(+, BinOp(+, a, BinOp(*, i, 4)), 8))
    ir = BinOp("+",
        Mem(
            BinOp("+",
                BinOp("+",
                    Reg("a"),
                    BinOp("*", Reg("i"), Const(4))
                ),
                Const(8)
            )
        ),
        Const(1)
    )

    gen = MaximalMunchCodeGen()
    result_reg = gen.munch_expr(ir)
    print(f"Result in register: {result_reg}\n")
    gen.print_code()


def demo_array_store():
    """Generate code for: a[i] = b + c"""
    print("\n=== Statement: a[i] = b + c ===\n")

    # Store(BinOp(+, a, BinOp(*, i, 4)), BinOp(+, b, c))
    ir = Store(
        address=BinOp("+", Reg("a"), BinOp("*", Reg("i"), Const(4))),
        value=BinOp("+", Reg("b"), Reg("c"))
    )

    gen = MaximalMunchCodeGen()
    gen.munch_stmt(ir)
    gen.print_code()


def demo_conditional():
    """Generate code for: if (x < y) goto L1 else goto L2"""
    print("\n=== Conditional: if (x < y) goto L1 else goto L2 ===\n")

    ir = CJump("<", Reg("x"), Reg("y"), "L1", "L2")

    gen = MaximalMunchCodeGen()
    gen.munch_stmt(ir)
    gen.print_code()


if __name__ == "__main__":
    demo_expression()
    demo_array_store()
    demo_conditional()
```

---

## 3. Register Allocation

### 3.1 The Problem

The IR uses an unlimited number of virtual registers (or temporaries), but real machines have a small, fixed number of physical registers ($k$ registers). Register allocation assigns virtual registers to physical registers, **spilling** some to memory when $k$ registers are insufficient.

### 3.2 Why Register Allocation Matters

Register access is orders of magnitude faster than memory access:

| Access | Latency (cycles) |
|--------|-----------------|
| Register | 0-1 |
| L1 cache | 3-5 |
| L2 cache | 10-15 |
| L3 cache | 30-50 |
| Main memory | 100-300 |

Good register allocation can dramatically improve program performance.

### 3.3 Liveness Analysis

Before allocating registers, we must determine which variables are **live** at each program point. A variable is **live** at a point if it holds a value that may be needed in the future.

**Definitions**:
- A variable $v$ is **defined** at point $p$ if $p$ assigns to $v$
- A variable $v$ is **used** at point $p$ if $p$ reads $v$
- A variable $v$ is **live** at point $p$ if there is a path from $p$ to a use of $v$ that does not pass through a redefinition of $v$

**Data flow equations** (computed backward):

$$\text{LiveIn}(B) = \text{Use}(B) \cup (\text{LiveOut}(B) - \text{Def}(B))$$

$$\text{LiveOut}(B) = \bigcup_{S \in \text{succ}(B)} \text{LiveIn}(S)$$

### 3.4 Interference Graph

Two variables **interfere** if they are simultaneously live at some program point. The **interference graph** $G = (V, E)$ has:

- Vertices $V$: one per variable (virtual register)
- Edges $E$: $(u, v) \in E$ if $u$ and $v$ interfere

Register allocation becomes a **graph coloring** problem: assign $k$ colors (physical registers) to the vertices such that no two adjacent vertices have the same color.

### 3.5 Graph Coloring Register Allocation

The graph coloring approach to register allocation was introduced by Chaitin (1981).

#### Chaitin's Algorithm

```
repeat:
    1. Build the interference graph
    2. Simplify: While there exists a node with degree < k:
       - Remove it from the graph and push it on a stack
       - (It can always be colored since it has fewer than k neighbors)
    3. Spill: If no such node exists:
       - Select a node to spill (heuristic: high degree, low use frequency)
       - Insert load/store instructions for the spilled variable
       - Go back to step 1
    4. Select: Pop nodes from the stack and assign colors
       - Each node has fewer than k colored neighbors, so a color is available
```

#### Example

Consider 4 virtual registers $\{a, b, c, d\}$ with interference graph:

```
a --- b
|  X  |     (a-b, a-c, b-c, b-d interfere)
c --- d
   |
   b
```

Wait, let us be precise:
- $a$ interferes with $b$ and $c$
- $b$ interferes with $a$, $c$, and $d$
- $c$ interferes with $a$ and $b$
- $d$ interferes with $b$

With $k = 3$ registers:

```
Step 1 (Simplify):
  d has degree 1 < 3 → push d, remove from graph
  a has degree 2 < 3 → push a, remove
  c has degree 1 < 3 → push c, remove
  b has degree 0 < 3 → push b, remove

Stack (top to bottom): [b, c, a, d]

Step 2 (Select):
  Pop b → assign R0 (no neighbors colored yet)
  Pop c → neighbors: {b=R0} → assign R1
  Pop a → neighbors: {b=R0, c=R1} → assign R2
  Pop d → neighbors: {b=R0} → assign R1 (or R2)

Result: a=R2, b=R0, c=R1, d=R1
```

### 3.6 Spilling

When the graph cannot be $k$-colored (no node has degree $< k$), we must **spill** a variable to memory. This means:

1. Before each use of the spilled variable, insert a `LOAD` from memory
2. After each definition, insert a `STORE` to memory
3. The spilled variable no longer needs a register (or needs one only briefly)
4. Rebuild the interference graph and try again

**Spill heuristics**:
- Spill the variable with the highest degree (most interference)
- Spill the variable used least frequently (lowest use count)
- Spill the variable with the highest degree/use ratio
- Avoid spilling variables inside loops

### 3.7 Linear Scan Register Allocation

Graph coloring is effective but expensive for large programs. **Linear scan** is a faster alternative used in JIT compilers (e.g., HotSpot JVM, V8).

**Idea**: Process variables in order of their **live intervals** (the range from first definition to last use). Allocate registers greedily:

```
Algorithm: Linear Scan Register Allocation

Input:  Live intervals sorted by start point
        k available registers

1. active = []  (intervals currently occupying a register)
2. free_regs = {R0, R1, ..., R(k-1)}

3. for each interval i in order of increasing start:
    a. Expire old intervals:
       for each j in active (sorted by end point):
           if j.end < i.start:
               remove j from active
               return j.register to free_regs

    b. if free_regs is empty:
       Spill: find the interval in active with the latest end point
              if that end point > i.end:
                  spill that interval, give its register to i
              else:
                  spill i (assign memory location)

    c. else:
       Allocate: assign a register from free_regs to i
       add i to active
```

**Time complexity**: $O(n \log n)$ where $n$ is the number of intervals (sorting dominates).

**Comparison**:

| Aspect | Graph Coloring | Linear Scan |
|--------|---------------|-------------|
| Quality | Better (global view) | Good (not optimal) |
| Compile time | $O(n^2)$ or worse | $O(n \log n)$ |
| Use case | Ahead-of-time compilers | JIT compilers |
| Handles coalescing | Naturally | Additional pass needed |

### 3.8 Python Implementation: Linear Scan

```python
"""Linear Scan Register Allocation."""

from dataclasses import dataclass, field


@dataclass
class LiveInterval:
    """A live interval for a virtual register."""
    vreg: str      # Virtual register name
    start: int     # First definition point
    end: int       # Last use point
    preg: str = "" # Assigned physical register (or "spill")

    def __str__(self):
        alloc = self.preg if self.preg else "unallocated"
        return f"{self.vreg}: [{self.start}, {self.end}] -> {alloc}"


class LinearScanAllocator:
    """
    Linear scan register allocator.
    """

    def __init__(self, num_physical_regs: int):
        self.k = num_physical_regs
        self.physical_regs = [f"R{i}" for i in range(num_physical_regs)]
        self.free_regs: list[str] = list(self.physical_regs)
        self.active: list[LiveInterval] = []  # Sorted by end point
        self.spilled: list[LiveInterval] = []

    def allocate(self, intervals: list[LiveInterval]):
        """
        Perform linear scan allocation.
        Modifies interval.preg in place.
        """
        # Sort by start point
        intervals.sort(key=lambda iv: iv.start)

        for iv in intervals:
            # Expire old intervals
            self._expire_old(iv)

            if not self.free_regs:
                # Must spill
                self._spill_at(iv)
            else:
                # Allocate a register
                reg = self.free_regs.pop(0)
                iv.preg = reg
                self.active.append(iv)
                self.active.sort(key=lambda x: x.end)

    def _expire_old(self, current: LiveInterval):
        """Remove intervals that have ended before the current one starts."""
        still_active = []
        for iv in self.active:
            if iv.end < current.start:
                # This interval has expired
                self.free_regs.append(iv.preg)
            else:
                still_active.append(iv)
        self.active = still_active
        self.active.sort(key=lambda x: x.end)

    def _spill_at(self, current: LiveInterval):
        """Spill either the current interval or the one ending latest."""
        if self.active and self.active[-1].end > current.end:
            # Spill the active interval ending latest
            spill = self.active.pop()
            current.preg = spill.preg
            spill.preg = "SPILL"
            self.spilled.append(spill)
            self.active.append(current)
            self.active.sort(key=lambda x: x.end)
        else:
            # Spill the current interval
            current.preg = "SPILL"
            self.spilled.append(current)


def demo_linear_scan():
    """Demonstrate linear scan register allocation."""
    print("=== Linear Scan Register Allocation ===\n")

    intervals = [
        LiveInterval("t1", start=0,  end=10),
        LiveInterval("t2", start=2,  end=8),
        LiveInterval("t3", start=4,  end=12),
        LiveInterval("t4", start=6,  end=14),
        LiveInterval("t5", start=9,  end=16),
        LiveInterval("t6", start=11, end=18),
        LiveInterval("t7", start=15, end=20),
    ]

    print("Live intervals:")
    for iv in intervals:
        print(f"  {iv.vreg}: [{iv.start}, {iv.end}]")

    # Allocate with 3 physical registers
    allocator = LinearScanAllocator(num_physical_regs=3)
    allocator.allocate(intervals)

    print(f"\nAllocation with {allocator.k} registers:")
    for iv in intervals:
        status = f"  {iv}"
        if iv.preg == "SPILL":
            status += " (SPILLED)"
        print(status)

    if allocator.spilled:
        print(f"\nSpilled variables: "
              f"{[s.vreg for s in allocator.spilled]}")

    # Visualize
    print("\nTimeline visualization:")
    max_time = max(iv.end for iv in intervals)
    print("  Time: " + "".join(f"{t:2d}" for t in range(max_time + 1)))
    for iv in intervals:
        line = "  " + f"{iv.vreg:4s}: "
        for t in range(max_time + 1):
            if iv.start <= t <= iv.end:
                if iv.preg == "SPILL":
                    line += " S"
                else:
                    line += f" {iv.preg[1]}"  # Just the number
            else:
                line += " ."
        line += f"  ({iv.preg})"
        print(line)


if __name__ == "__main__":
    demo_linear_scan()
```

---

## 4. Instruction Scheduling

### 4.1 The Problem

Modern processors are **pipelined**: different stages of instruction execution overlap. However, **data hazards** can cause **pipeline stalls** when an instruction depends on the result of a previous instruction that has not yet completed.

**Example stall**:
```
LOAD R1, [R2]     ; takes 4 cycles to complete
ADD  R3, R1, R4   ; must wait for R1 → pipeline stall (3 wasted cycles)
```

**After scheduling**:
```
LOAD R1, [R2]     ; issue load
ADD  R5, R6, R7   ; independent instruction fills the gap
SUB  R8, R9, R10  ; another independent instruction
MOV  R11, R12     ; yet another
ADD  R3, R1, R4   ; R1 is now ready, no stall
```

### 4.2 List Scheduling

**List scheduling** is the most common instruction scheduling algorithm. It operates on a single basic block.

**Input**: A **dependency DAG** where:
- Nodes are instructions
- Edges represent data dependencies (read-after-write, write-after-read, write-after-write)
- Edge weights represent the latency of the source instruction

**Algorithm**:

```
function list_schedule(DAG, num_functional_units):
    ready = set of instructions with no predecessors
    schedule = []
    cycle = 0

    while ready is not empty or pending instructions exist:
        # Select from ready instructions (priority-based)
        available = instructions in ready whose operands are available
        for each functional unit:
            if available is not empty:
                instr = select highest-priority instruction from available
                issue instr at current cycle
                schedule[cycle] = instr
                remove instr from ready

        cycle += 1

        # Check if any pending instructions have completed
        # Add their successors to ready if all predecessors are done

    return schedule
```

**Priority heuristics** (which instruction to schedule first):
- **Critical path length**: Prefer instructions on the longest path to any sink
- **Number of successors**: Prefer instructions with more dependent successors
- **Latency**: Prefer instructions with higher latency (start them sooner)

### 4.3 Example: List Scheduling

```
Instructions:
  I1: LOAD R1, [addr1]   ; latency 4
  I2: LOAD R2, [addr2]   ; latency 4
  I3: ADD  R3, R1, R2    ; latency 1, depends on I1, I2
  I4: MUL  R4, R3, R1    ; latency 3, depends on I3, I1
  I5: STORE R4, [addr3]  ; latency 1, depends on I4

Dependency DAG:
  I1 ──(4)──▶ I3
  I2 ──(4)──▶ I3
  I3 ──(1)──▶ I4
  I1 ──(4)──▶ I4
  I4 ──(3)──▶ I5

Unscheduled (all instructions sequentially):
  Cycle 0: I1 (LOAD)
  Cycle 1-3: stall (waiting for I1)
  Cycle 4: I2 (LOAD)
  Cycle 5-7: stall (waiting for I2)
  Cycle 8: I3 (ADD)
  Cycle 9: I4 (MUL)
  Cycle 10-11: stall
  Cycle 12: I5 (STORE)
  Total: 13 cycles

Scheduled (list scheduling):
  Cycle 0: I1 (LOAD R1)
  Cycle 1: I2 (LOAD R2)    ← issued during I1's latency
  Cycle 4: I3 (ADD R3)     ← both I1, I2 ready
  Cycle 5: I4 (MUL R4)     ← I3 ready
  Cycle 8: I5 (STORE R4)   ← I4 ready
  Total: 9 cycles (saved 4 cycles)
```

### 4.4 Software Pipelining (Overview)

**Software pipelining** is a loop optimization technique that overlaps iterations of a loop, similar to how hardware pipelining overlaps instruction stages.

**Idea**: Instead of completing iteration $i$ before starting iteration $i+1$, begin iteration $i+1$ while iteration $i$ is still in progress:

```
Without software pipelining:     With software pipelining:
  Iter 1: LOAD-ADD-STORE         Cycle 0: LOAD[1]
  Iter 2: LOAD-ADD-STORE         Cycle 1: LOAD[2], ADD[1]
  Iter 3: LOAD-ADD-STORE         Cycle 2: LOAD[3], ADD[2], STORE[1]
                                  Cycle 3: LOAD[4], ADD[3], STORE[2]
                                  ...
```

The steady-state of the software pipeline executes parts of multiple iterations simultaneously, keeping all functional units busy.

Software pipelining is most commonly implemented using **modulo scheduling**, which finds a schedule for one iteration such that repeating it with a fixed **initiation interval** (II) produces a valid overlapping schedule.

---

## 5. Peephole Optimization

### 5.1 What Is Peephole Optimization?

**Peephole optimization** examines a small window (the "peephole") of consecutive instructions and replaces them with faster or shorter equivalents. It is applied after code generation as a final cleanup pass.

### 5.2 Common Peephole Optimizations

#### Redundant Load/Store Elimination

```
Before:                  After:
  STORE R1, [addr]       STORE R1, [addr]
  LOAD  R1, [addr]       (load eliminated -- R1 already has the value)
```

#### Redundant Moves

```
Before:                  After:
  MOV R1, R2             (eliminated if R1 == R2, or
  MOV R2, R1              second move eliminated if first is sufficient)
```

#### Strength Reduction

```
Before:                  After:
  MUL R1, R2, #2         SHL R1, R2, #1    (shift is cheaper)
  MUL R1, R2, #8         SHL R1, R2, #3
  DIV R1, R2, #4         SHR R1, R2, #2    (for unsigned)
```

#### Algebraic Simplification

```
Before:                  After:
  ADD R1, R2, #0         MOV R1, R2         (or eliminated if R1==R2)
  MUL R1, R2, #1         MOV R1, R2
  MUL R1, R2, #0         MOVI R1, #0
  SUB R1, R2, #0         MOV R1, R2
```

#### Branch Optimization

```
Before:                  After:
  JMP L1                 JMP L2             (jump-to-jump elimination)
  ...
L1: JMP L2

Before:                  After:
  BEQ L1                 BNE L2             (branch-over-jump elimination)
  JMP L2                 ...
L1: ...                  L2: ...
```

#### Unreachable Code Elimination

```
Before:                  After:
  JMP L1                 JMP L1
  ADD R1, R2, R3         (unreachable, eliminated)
  MOV R4, R5             (unreachable, eliminated)
L1: ...                  L1: ...
```

### 5.3 Python Implementation: Peephole Optimizer

```python
"""Peephole optimizer for a simple instruction set."""

from dataclasses import dataclass
import re


@dataclass
class Instruction:
    """A machine instruction for peephole optimization."""
    text: str       # Full instruction text
    opcode: str = ""
    operands: list = None
    label: str = ""  # If this is a label

    def __post_init__(self):
        if self.operands is None:
            self.operands = []
        self._parse()

    def _parse(self):
        text = self.text.strip()
        if text.endswith(":"):
            self.label = text[:-1]
            self.opcode = "LABEL"
            return
        parts = text.split(None, 1)
        if parts:
            self.opcode = parts[0].upper()
            if len(parts) > 1:
                self.operands = [o.strip() for o in parts[1].split(",")]

    def __str__(self):
        if self.label:
            return f"{self.label}:"
        return self.text


class PeepholeOptimizer:
    """
    Apply peephole optimizations to a list of instructions.
    """

    def __init__(self, instructions: list[str]):
        self.instructions = [Instruction(text=instr) for instr in instructions]
        self.changed = True  # Track if any optimization was applied

    def optimize(self, max_passes: int = 10) -> list[str]:
        """Run peephole optimizations until no more changes occur."""
        pass_num = 0
        while self.changed and pass_num < max_passes:
            self.changed = False
            pass_num += 1
            self._redundant_moves()
            self._redundant_load_after_store()
            self._strength_reduction()
            self._algebraic_simplification()
            self._jump_to_jump()
            self._unreachable_code()
            # Remove None entries
            self.instructions = [i for i in self.instructions if i is not None]

        return [str(i) for i in self.instructions]

    def _redundant_moves(self):
        """Remove MOV Rx, Rx (move to self)."""
        for i, instr in enumerate(self.instructions):
            if instr and instr.opcode == "MOV" and len(instr.operands) == 2:
                if instr.operands[0] == instr.operands[1]:
                    self.instructions[i] = None
                    self.changed = True

    def _redundant_load_after_store(self):
        """Remove LOAD Rx, [addr] immediately after STORE Rx, [addr]."""
        for i in range(len(self.instructions) - 1):
            curr = self.instructions[i]
            nxt = self.instructions[i + 1]
            if curr is None or nxt is None:
                continue
            if curr.opcode == "STORE" and nxt.opcode == "LOAD":
                # Check if same register and same address
                if (len(curr.operands) >= 2 and len(nxt.operands) >= 2
                    and curr.operands[0] == nxt.operands[0]
                    and curr.operands[1] == nxt.operands[1]):
                    self.instructions[i + 1] = None
                    self.changed = True

    def _strength_reduction(self):
        """Replace MUL/DIV by power of 2 with shifts."""
        for i, instr in enumerate(self.instructions):
            if instr is None:
                continue
            if instr.opcode == "MUL" and len(instr.operands) == 3:
                imm = instr.operands[2]
                if imm.startswith("#"):
                    val = int(imm[1:])
                    if val > 0 and (val & (val - 1)) == 0:
                        shift = val.bit_length() - 1
                        new_text = (f"  SHL {instr.operands[0]}, "
                                    f"{instr.operands[1]}, #{shift}")
                        self.instructions[i] = Instruction(text=new_text)
                        self.changed = True

            if instr.opcode == "DIV" and len(instr.operands) == 3:
                imm = instr.operands[2]
                if imm.startswith("#"):
                    val = int(imm[1:])
                    if val > 0 and (val & (val - 1)) == 0:
                        shift = val.bit_length() - 1
                        new_text = (f"  SHR {instr.operands[0]}, "
                                    f"{instr.operands[1]}, #{shift}")
                        self.instructions[i] = Instruction(text=new_text)
                        self.changed = True

    def _algebraic_simplification(self):
        """Simplify ADD x, y, #0 → MOV x, y, etc."""
        for i, instr in enumerate(self.instructions):
            if instr is None:
                continue

            if instr.opcode == "ADD" and len(instr.operands) == 3:
                if instr.operands[2] == "#0":
                    new_text = f"  MOV {instr.operands[0]}, {instr.operands[1]}"
                    self.instructions[i] = Instruction(text=new_text)
                    self.changed = True

            if instr.opcode == "MUL" and len(instr.operands) == 3:
                if instr.operands[2] == "#1":
                    new_text = f"  MOV {instr.operands[0]}, {instr.operands[1]}"
                    self.instructions[i] = Instruction(text=new_text)
                    self.changed = True
                elif instr.operands[2] == "#0":
                    new_text = f"  MOVI {instr.operands[0]}, #0"
                    self.instructions[i] = Instruction(text=new_text)
                    self.changed = True

            if instr.opcode == "SUB" and len(instr.operands) == 3:
                if instr.operands[2] == "#0":
                    new_text = f"  MOV {instr.operands[0]}, {instr.operands[1]}"
                    self.instructions[i] = Instruction(text=new_text)
                    self.changed = True

    def _jump_to_jump(self):
        """Replace JMP L1 where L1: JMP L2 with JMP L2."""
        # Build label -> index map
        label_map = {}
        for i, instr in enumerate(self.instructions):
            if instr and instr.opcode == "LABEL":
                label_map[instr.label] = i

        for i, instr in enumerate(self.instructions):
            if instr is None:
                continue
            if instr.opcode == "JMP" and len(instr.operands) == 1:
                target = instr.operands[0]
                if target in label_map:
                    target_idx = label_map[target]
                    # Find the next non-label instruction after the target
                    j = target_idx + 1
                    while (j < len(self.instructions)
                           and self.instructions[j] is not None
                           and self.instructions[j].opcode == "LABEL"):
                        j += 1
                    if (j < len(self.instructions)
                        and self.instructions[j] is not None
                        and self.instructions[j].opcode == "JMP"):
                        new_target = self.instructions[j].operands[0]
                        if new_target != target:
                            new_text = f"  JMP {new_target}"
                            self.instructions[i] = Instruction(text=new_text)
                            self.changed = True

    def _unreachable_code(self):
        """Remove instructions between an unconditional jump and the next label."""
        i = 0
        while i < len(self.instructions):
            instr = self.instructions[i]
            if instr and instr.opcode in ("JMP", "RET"):
                j = i + 1
                while (j < len(self.instructions)
                       and self.instructions[j] is not None
                       and self.instructions[j].opcode != "LABEL"):
                    self.instructions[j] = None
                    self.changed = True
                    j += 1
            i += 1


def demo_peephole():
    """Demonstrate peephole optimization."""
    instructions = [
        "  MOVI R1, #5",
        "  MOVI R2, #10",
        "  ADD  R3, R1, R2",
        "  MOV  R3, R3",           # Redundant self-move
        "  MUL  R4, R3, #8",       # Strength reduction: * 8 → << 3
        "  ADD  R5, R4, #0",       # Algebraic: + 0 → move
        "  STORE R5, [R6]",
        "  LOAD  R5, [R6]",        # Redundant load after store
        "  MUL  R7, R5, #1",       # Algebraic: * 1 → move
        "  MUL  R8, R5, #0",       # Algebraic: * 0 → 0
        "  DIV  R9, R3, #4",       # Strength reduction: / 4 → >> 2
        "  JMP  L1",
        "  ADD  R10, R1, R2",      # Unreachable code
        "  SUB  R11, R3, R4",      # Unreachable code
        "L1:",
        "  JMP  L2",               # Jump-to-jump target
        "L2:",
        "  SUB  R12, R1, #0",      # Algebraic: - 0 → move
        "  RET",
    ]

    print("=== Before Peephole Optimization ===")
    for instr in instructions:
        print(instr)

    optimizer = PeepholeOptimizer(instructions)
    optimized = optimizer.optimize()

    print("\n=== After Peephole Optimization ===")
    for instr in optimized:
        print(instr)


if __name__ == "__main__":
    demo_peephole()
```

---

## 6. Code Generation for Language Constructs

### 6.1 Expressions

Generating code for arithmetic expressions follows the structure of the expression tree. For a binary operation $a \;\text{op}\; b$:

```
generate(a)         → result in R1
generate(b)         → result in R2
OP R3, R1, R2       → R3 = R1 op R2
```

For deeply nested expressions, we need to manage registers carefully. The **Sethi-Ullman numbering** algorithm computes the minimum number of registers needed to evaluate an expression tree.

**Sethi-Ullman numbering**:
- A leaf that is a left child: label = 1
- A leaf that is a right child: label = 0
- An interior node with children labeled $l_1$ and $l_2$:

$$\text{label}(n) = \begin{cases} \max(l_1, l_2) & \text{if } l_1 \neq l_2 \\ l_1 + 1 & \text{if } l_1 = l_2 \end{cases}$$

The label at the root gives the minimum number of registers needed.

### 6.2 Control Flow

#### If-Else

```c
if (condition) {
    then_body;
} else {
    else_body;
}
```

Generated code:
```
    <evaluate condition into R1>
    CMP R1, #0
    BEQ else_label
    <then_body code>
    JMP end_label
else_label:
    <else_body code>
end_label:
```

#### While Loop

```c
while (condition) {
    body;
}
```

Generated code:
```
loop_start:
    <evaluate condition into R1>
    CMP R1, #0
    BEQ loop_end
    <body code>
    JMP loop_start
loop_end:
```

#### For Loop

A for loop is typically desugared into a while loop:

```c
for (init; condition; step) { body; }
```

Becomes:
```
    <init code>
loop_start:
    <evaluate condition>
    BEQ loop_end
    <body code>
    <step code>
    JMP loop_start
loop_end:
```

#### Short-Circuit Boolean Evaluation

For `a && b`:
```
    <evaluate a>
    CMP R1, #0
    BEQ false_label     ; if a is false, skip b
    <evaluate b>
    CMP R2, #0
    BEQ false_label     ; if b is false
    MOVI R3, #1          ; result is true
    JMP end_label
false_label:
    MOVI R3, #0          ; result is false
end_label:
```

For `a || b`:
```
    <evaluate a>
    CMP R1, #0
    BNE true_label       ; if a is true, skip b
    <evaluate b>
    CMP R2, #0
    BNE true_label       ; if b is true
    MOVI R3, #0           ; result is false
    JMP end_label
true_label:
    MOVI R3, #1           ; result is true
end_label:
```

### 6.3 Function Calls

Generating code for a function call involves:

```
; Caller-saved registers
PUSH R_caller_saved1
PUSH R_caller_saved2

; Arguments (System V AMD64: first 6 in registers)
MOV RDI, arg1
MOV RSI, arg2
MOV RDX, arg3
; ... or push on stack if > 6 args

; Call
CALL function_label

; Return value is in RAX
MOV result, RAX

; Restore caller-saved registers
POP R_caller_saved2
POP R_caller_saved1
```

---

## 7. Stack Machine Code Generator

### 7.1 What Is a Stack Machine?

A **stack machine** uses an operand stack instead of registers. Instructions implicitly operate on the top of the stack:

```
PUSH 5        ; stack: [5]
PUSH 3        ; stack: [5, 3]
ADD           ; stack: [8]       (pop two, push sum)
PUSH 2        ; stack: [8, 2]
MUL           ; stack: [16]      (pop two, push product)
```

Stack machines are simpler to generate code for because there is no register allocation problem. Examples: JVM bytecode, .NET CIL, Python bytecode, WebAssembly (partially).

### 7.2 Stack Machine Instruction Set

```
Stack operations:
    PUSH <value>     ; Push constant onto stack
    LOAD <var>       ; Push variable's value onto stack
    STORE <var>      ; Pop value and store in variable
    POP              ; Discard top of stack
    DUP              ; Duplicate top of stack

Arithmetic (pop operands, push result):
    ADD, SUB, MUL, DIV, MOD, NEG

Comparison (pop two, push boolean):
    EQ, NE, LT, GT, LE, GE

Logic:
    AND, OR, NOT

Control flow:
    LABEL <name>     ; Define a label
    JMP <label>      ; Unconditional jump
    JMPF <label>     ; Jump if top of stack is false (0)
    JMPT <label>     ; Jump if top of stack is true (non-0)

Functions:
    CALL <func> <nargs>  ; Call function with n arguments
    RET                  ; Return (top of stack is return value)

I/O:
    PRINT            ; Pop and print top of stack
    READ             ; Read input and push onto stack
```

### 7.3 Complete Stack Machine Implementation

```python
"""
Complete code generator and virtual machine for a stack-based target.

Compiles a simple language to stack machine bytecode and executes it.
"""

from dataclasses import dataclass
from typing import Union
from enum import Enum, auto


# ============================================================
# Part 1: Source Language AST
# ============================================================

@dataclass
class NumLit:
    value: int

@dataclass
class BoolLit:
    value: bool

@dataclass
class VarRef:
    name: str

@dataclass
class BinaryExpr:
    op: str
    left: 'Expression'
    right: 'Expression'

@dataclass
class UnaryExpr:
    op: str
    operand: 'Expression'

@dataclass
class CallExpr:
    func: str
    args: list

Expression = Union[NumLit, BoolLit, VarRef, BinaryExpr, UnaryExpr, CallExpr]


@dataclass
class AssignStmt:
    target: str
    value: Expression

@dataclass
class PrintStmt:
    value: Expression

@dataclass
class IfStmt:
    condition: Expression
    then_body: list
    else_body: list

@dataclass
class WhileStmt:
    condition: Expression
    body: list

@dataclass
class ReturnStmt:
    value: Expression

@dataclass
class FuncDef:
    name: str
    params: list  # list of parameter names
    body: list    # list of statements

Statement = Union[AssignStmt, PrintStmt, IfStmt, WhileStmt, ReturnStmt]


# ============================================================
# Part 2: Stack Machine Bytecode
# ============================================================

class Opcode(Enum):
    PUSH = auto()
    LOAD = auto()
    STORE = auto()
    POP = auto()
    DUP = auto()
    ADD = auto()
    SUB = auto()
    MUL = auto()
    DIV = auto()
    MOD = auto()
    NEG = auto()
    EQ = auto()
    NE = auto()
    LT = auto()
    GT = auto()
    LE = auto()
    GE = auto()
    AND = auto()
    OR = auto()
    NOT = auto()
    JMP = auto()
    JMPF = auto()
    JMPT = auto()
    CALL = auto()
    RET = auto()
    PRINT = auto()
    LABEL = auto()
    HALT = auto()


@dataclass
class BytecodeInstr:
    opcode: Opcode
    operand: Union[int, str, None] = None
    operand2: Union[int, str, None] = None

    def __str__(self):
        parts = [f"{self.opcode.name:8s}"]
        if self.operand is not None:
            parts.append(str(self.operand))
        if self.operand2 is not None:
            parts.append(str(self.operand2))
        return " ".join(parts)


# ============================================================
# Part 3: Code Generator
# ============================================================

class CodeGenerator:
    """
    Generate stack machine bytecode from AST.
    """

    def __init__(self):
        self.code: list[BytecodeInstr] = []
        self._label_counter = 0
        self.functions: dict[str, int] = {}  # func name -> code address

    def new_label(self) -> str:
        self._label_counter += 1
        return f"L{self._label_counter}"

    def emit(self, opcode: Opcode, operand=None, operand2=None):
        self.code.append(BytecodeInstr(opcode, operand, operand2))

    def generate_program(self, program: list):
        """
        Generate code for a complete program.
        Program is a list of function definitions and statements.
        """
        # Separate functions from top-level statements
        functions = [node for node in program if isinstance(node, FuncDef)]
        statements = [node for node in program
                      if not isinstance(node, FuncDef)]

        # Jump over function definitions to main code
        main_label = self.new_label()
        self.emit(Opcode.JMP, main_label)

        # Generate code for functions
        for func in functions:
            self._generate_function(func)

        # Generate main code
        self.emit(Opcode.LABEL, main_label)
        for stmt in statements:
            self._generate_stmt(stmt)
        self.emit(Opcode.HALT)

    def _generate_function(self, func: FuncDef):
        """Generate code for a function definition."""
        self.functions[func.name] = len(self.code)
        self.emit(Opcode.LABEL, f"func_{func.name}")

        # Function body
        for stmt in func.body:
            self._generate_stmt(stmt)

        # Implicit return 0 if no explicit return
        self.emit(Opcode.PUSH, 0)
        self.emit(Opcode.RET)

    def _generate_stmt(self, stmt):
        """Generate code for a statement."""
        if isinstance(stmt, AssignStmt):
            self._generate_expr(stmt.value)
            self.emit(Opcode.STORE, stmt.target)

        elif isinstance(stmt, PrintStmt):
            self._generate_expr(stmt.value)
            self.emit(Opcode.PRINT)

        elif isinstance(stmt, IfStmt):
            self._generate_if(stmt)

        elif isinstance(stmt, WhileStmt):
            self._generate_while(stmt)

        elif isinstance(stmt, ReturnStmt):
            self._generate_expr(stmt.value)
            self.emit(Opcode.RET)

        else:
            raise ValueError(f"Unknown statement type: {type(stmt)}")

    def _generate_expr(self, expr):
        """Generate code for an expression (result pushed on stack)."""
        if isinstance(expr, NumLit):
            self.emit(Opcode.PUSH, expr.value)

        elif isinstance(expr, BoolLit):
            self.emit(Opcode.PUSH, 1 if expr.value else 0)

        elif isinstance(expr, VarRef):
            self.emit(Opcode.LOAD, expr.name)

        elif isinstance(expr, BinaryExpr):
            self._generate_expr(expr.left)
            self._generate_expr(expr.right)
            op_map = {
                '+': Opcode.ADD, '-': Opcode.SUB,
                '*': Opcode.MUL, '/': Opcode.DIV,
                '%': Opcode.MOD,
                '==': Opcode.EQ, '!=': Opcode.NE,
                '<': Opcode.LT, '>': Opcode.GT,
                '<=': Opcode.LE, '>=': Opcode.GE,
                'and': Opcode.AND, 'or': Opcode.OR,
            }
            if expr.op in op_map:
                self.emit(op_map[expr.op])
            else:
                raise ValueError(f"Unknown operator: {expr.op}")

        elif isinstance(expr, UnaryExpr):
            self._generate_expr(expr.operand)
            if expr.op == '-':
                self.emit(Opcode.NEG)
            elif expr.op == 'not':
                self.emit(Opcode.NOT)

        elif isinstance(expr, CallExpr):
            # Push arguments in order
            for arg in expr.args:
                self._generate_expr(arg)
            self.emit(Opcode.CALL, f"func_{expr.func}", len(expr.args))

    def _generate_if(self, stmt: IfStmt):
        else_label = self.new_label()
        end_label = self.new_label()

        self._generate_expr(stmt.condition)
        self.emit(Opcode.JMPF, else_label)

        # Then body
        for s in stmt.then_body:
            self._generate_stmt(s)
        self.emit(Opcode.JMP, end_label)

        # Else body
        self.emit(Opcode.LABEL, else_label)
        for s in stmt.else_body:
            self._generate_stmt(s)

        self.emit(Opcode.LABEL, end_label)

    def _generate_while(self, stmt: WhileStmt):
        loop_label = self.new_label()
        end_label = self.new_label()

        self.emit(Opcode.LABEL, loop_label)
        self._generate_expr(stmt.condition)
        self.emit(Opcode.JMPF, end_label)

        # Body
        for s in stmt.body:
            self._generate_stmt(s)
        self.emit(Opcode.JMP, loop_label)

        self.emit(Opcode.LABEL, end_label)

    def print_code(self):
        """Pretty-print the generated bytecode."""
        for i, instr in enumerate(self.code):
            label_marker = ""
            if instr.opcode == Opcode.LABEL:
                label_marker = f"\n{instr.operand}:"
                print(label_marker)
                continue
            print(f"  [{i:3d}] {instr}")


# ============================================================
# Part 4: Stack Machine Virtual Machine
# ============================================================

class StackMachineVM:
    """
    Execute stack machine bytecode.
    """

    def __init__(self, code: list[BytecodeInstr],
                 functions: dict[str, int]):
        self.code = code
        self.functions = functions
        self.stack: list = []          # Operand stack
        self.call_stack: list = []     # Return addresses + saved state
        self.variables: dict[str, int] = {}
        self.pc = 0                    # Program counter
        self.output: list[str] = []    # Captured output

        # Resolve labels to addresses
        self.label_map: dict[str, int] = {}
        for i, instr in enumerate(self.code):
            if instr.opcode == Opcode.LABEL:
                self.label_map[instr.operand] = i

    def run(self, trace: bool = False):
        """Execute the bytecode program."""
        while self.pc < len(self.code):
            instr = self.code[self.pc]

            if trace and instr.opcode != Opcode.LABEL:
                print(f"  PC={self.pc:3d} {instr}  "
                      f"stack={self.stack[-5:]}")

            if instr.opcode == Opcode.HALT:
                break

            elif instr.opcode == Opcode.LABEL:
                self.pc += 1
                continue

            elif instr.opcode == Opcode.PUSH:
                self.stack.append(instr.operand)

            elif instr.opcode == Opcode.LOAD:
                name = instr.operand
                value = self.variables.get(name, 0)
                self.stack.append(value)

            elif instr.opcode == Opcode.STORE:
                name = instr.operand
                value = self.stack.pop()
                self.variables[name] = value

            elif instr.opcode == Opcode.POP:
                self.stack.pop()

            elif instr.opcode == Opcode.DUP:
                self.stack.append(self.stack[-1])

            # Arithmetic
            elif instr.opcode == Opcode.ADD:
                b, a = self.stack.pop(), self.stack.pop()
                self.stack.append(a + b)

            elif instr.opcode == Opcode.SUB:
                b, a = self.stack.pop(), self.stack.pop()
                self.stack.append(a - b)

            elif instr.opcode == Opcode.MUL:
                b, a = self.stack.pop(), self.stack.pop()
                self.stack.append(a * b)

            elif instr.opcode == Opcode.DIV:
                b, a = self.stack.pop(), self.stack.pop()
                self.stack.append(a // b if b != 0 else 0)

            elif instr.opcode == Opcode.MOD:
                b, a = self.stack.pop(), self.stack.pop()
                self.stack.append(a % b if b != 0 else 0)

            elif instr.opcode == Opcode.NEG:
                self.stack.append(-self.stack.pop())

            # Comparison
            elif instr.opcode == Opcode.EQ:
                b, a = self.stack.pop(), self.stack.pop()
                self.stack.append(1 if a == b else 0)

            elif instr.opcode == Opcode.NE:
                b, a = self.stack.pop(), self.stack.pop()
                self.stack.append(1 if a != b else 0)

            elif instr.opcode == Opcode.LT:
                b, a = self.stack.pop(), self.stack.pop()
                self.stack.append(1 if a < b else 0)

            elif instr.opcode == Opcode.GT:
                b, a = self.stack.pop(), self.stack.pop()
                self.stack.append(1 if a > b else 0)

            elif instr.opcode == Opcode.LE:
                b, a = self.stack.pop(), self.stack.pop()
                self.stack.append(1 if a <= b else 0)

            elif instr.opcode == Opcode.GE:
                b, a = self.stack.pop(), self.stack.pop()
                self.stack.append(1 if a >= b else 0)

            # Logic
            elif instr.opcode == Opcode.AND:
                b, a = self.stack.pop(), self.stack.pop()
                self.stack.append(1 if (a and b) else 0)

            elif instr.opcode == Opcode.OR:
                b, a = self.stack.pop(), self.stack.pop()
                self.stack.append(1 if (a or b) else 0)

            elif instr.opcode == Opcode.NOT:
                self.stack.append(1 if not self.stack.pop() else 0)

            # Control flow
            elif instr.opcode == Opcode.JMP:
                self.pc = self.label_map[instr.operand]
                continue

            elif instr.opcode == Opcode.JMPF:
                cond = self.stack.pop()
                if not cond:
                    self.pc = self.label_map[instr.operand]
                    continue

            elif instr.opcode == Opcode.JMPT:
                cond = self.stack.pop()
                if cond:
                    self.pc = self.label_map[instr.operand]
                    continue

            # Functions
            elif instr.opcode == Opcode.CALL:
                func_label = instr.operand
                nargs = instr.operand2

                # Save state
                self.call_stack.append({
                    'return_pc': self.pc + 1,
                    'saved_vars': dict(self.variables),
                })

                # Pop arguments and bind to params
                args = []
                for _ in range(nargs):
                    args.insert(0, self.stack.pop())

                # Jump to function
                self.pc = self.label_map[func_label]

                # Bind arguments (simplified: use positional names)
                for idx, arg_val in enumerate(args):
                    self.variables[f"__arg{idx}"] = arg_val

                continue

            elif instr.opcode == Opcode.RET:
                ret_val = self.stack.pop() if self.stack else 0
                if self.call_stack:
                    frame = self.call_stack.pop()
                    self.variables = frame['saved_vars']
                    self.pc = frame['return_pc']
                    self.stack.append(ret_val)
                    continue
                else:
                    self.stack.append(ret_val)
                    break

            elif instr.opcode == Opcode.PRINT:
                val = self.stack.pop()
                self.output.append(str(val))
                print(f"  OUTPUT: {val}")

            self.pc += 1


# ============================================================
# Part 5: Examples
# ============================================================

def demo_fibonacci():
    """
    Compile and run:
        n = 10
        a = 0
        b = 1
        i = 0
        while (i < n) {
            print a
            temp = a + b
            a = b
            b = temp
            i = i + 1
        }
    """
    print("=" * 60)
    print("Stack Machine: Fibonacci Sequence (first 10)")
    print("=" * 60)

    program = [
        AssignStmt("n", NumLit(10)),
        AssignStmt("a", NumLit(0)),
        AssignStmt("b", NumLit(1)),
        AssignStmt("i", NumLit(0)),
        WhileStmt(
            condition=BinaryExpr("<", VarRef("i"), VarRef("n")),
            body=[
                PrintStmt(VarRef("a")),
                AssignStmt("temp", BinaryExpr("+", VarRef("a"), VarRef("b"))),
                AssignStmt("a", VarRef("b")),
                AssignStmt("b", VarRef("temp")),
                AssignStmt("i", BinaryExpr("+", VarRef("i"), NumLit(1))),
            ]
        ),
    ]

    # Generate code
    gen = CodeGenerator()
    gen.generate_program(program)
    print("\n--- Generated Bytecode ---")
    gen.print_code()

    # Execute
    print("\n--- Execution ---")
    vm = StackMachineVM(gen.code, gen.functions)
    vm.run()
    print(f"\nFibonacci output: {vm.output}")


def demo_factorial():
    """
    Compile and run:
        result = 1
        n = 7
        while (n > 0) {
            result = result * n
            n = n - 1
        }
        print result
    """
    print("\n" + "=" * 60)
    print("Stack Machine: Factorial of 7")
    print("=" * 60)

    program = [
        AssignStmt("result", NumLit(1)),
        AssignStmt("n", NumLit(7)),
        WhileStmt(
            condition=BinaryExpr(">", VarRef("n"), NumLit(0)),
            body=[
                AssignStmt("result",
                    BinaryExpr("*", VarRef("result"), VarRef("n"))),
                AssignStmt("n",
                    BinaryExpr("-", VarRef("n"), NumLit(1))),
            ]
        ),
        PrintStmt(VarRef("result")),
    ]

    gen = CodeGenerator()
    gen.generate_program(program)
    print("\n--- Generated Bytecode ---")
    gen.print_code()

    print("\n--- Execution ---")
    vm = StackMachineVM(gen.code, gen.functions)
    vm.run()
    print(f"\n7! = {vm.output[0]}")


def demo_conditionals():
    """
    Compile and run:
        x = 15
        if (x > 10) {
            if (x > 20) {
                print 3
            } else {
                print 2
            }
        } else {
            print 1
        }
    """
    print("\n" + "=" * 60)
    print("Stack Machine: Nested If-Else")
    print("=" * 60)

    program = [
        AssignStmt("x", NumLit(15)),
        IfStmt(
            condition=BinaryExpr(">", VarRef("x"), NumLit(10)),
            then_body=[
                IfStmt(
                    condition=BinaryExpr(">", VarRef("x"), NumLit(20)),
                    then_body=[PrintStmt(NumLit(3))],
                    else_body=[PrintStmt(NumLit(2))],
                )
            ],
            else_body=[PrintStmt(NumLit(1))],
        ),
    ]

    gen = CodeGenerator()
    gen.generate_program(program)
    print("\n--- Generated Bytecode ---")
    gen.print_code()

    print("\n--- Execution ---")
    vm = StackMachineVM(gen.code, gen.functions)
    vm.run()
    print(f"\nExpected: 2, Got: {vm.output[0]}")


if __name__ == "__main__":
    demo_fibonacci()
    demo_factorial()
    demo_conditionals()
```

---

## 8. Machine-Dependent Optimization

### 8.1 Utilizing Special Instructions

Many processors have specialized instructions that the code generator can exploit:

| Optimization | Example |
|-------------|---------|
| Multiply-accumulate | `MADD Rd, Rs1, Rs2, Rs3` ($Rd = Rs1 + Rs2 \times Rs3$) |
| Count leading zeros | `CLZ Rd, Rs` (useful for log2) |
| Byte swap | `REV Rd, Rs` (endianness conversion) |
| Conditional move | `CMOV Rd, Rs, cond` (avoids branch) |
| SIMD | `VADD.4S V0, V1, V2` (4 additions in parallel) |

### 8.2 Addressing Mode Selection

Complex addressing modes can reduce instruction count:

```
; Without: 3 instructions
MOV  R1, R_base
ADD  R1, R1, R_index, LSL #2    ; index * 4
LOAD R2, [R1]

; With indexed addressing: 1 instruction
LOAD R2, [R_base, R_index, LSL #2]
```

### 8.3 Conditional Execution (ARM)

ARM architectures support predicated execution, where instructions execute only if a condition flag is set:

```
; Standard if-else:
CMP R0, #0
BEQ else
ADD R1, R2, R3      ; then branch
B   endif
else:
SUB R1, R2, R3      ; else branch
endif:

; Predicated (no branches, no pipeline stalls):
CMP R0, #0
ADDNE R1, R2, R3    ; execute if NE (not equal to 0)
SUBEQ R1, R2, R3    ; execute if EQ (equal to 0)
```

### 8.4 Branch Prediction Hints

Some architectures allow the compiler to hint at likely branch directions:

```
; x86 branch hints (via instruction prefix)
; Prefix 0x3E: branch likely taken
; Prefix 0x2E: branch likely not taken

; GCC built-in:
if (__builtin_expect(error_condition, 0)) {
    // unlikely path
    handle_error();
}
```

---

## 9. Summary

In this lesson, we covered the major phases of code generation:

1. **Target machine model**: We defined a simple RISC-like instruction set with multiple addressing modes and instruction costs that guide code generation decisions.

2. **Instruction selection** maps IR trees to machine instructions. **Tree pattern matching** with tiles provides a systematic approach. The **Maximal Munch** algorithm greedily selects the largest matching tile at each node.

3. **Register allocation** assigns physical registers to virtual registers. **Graph coloring** provides optimal allocation but is expensive. **Linear scan** offers a fast alternative suitable for JIT compilers. Spilling moves overflow variables to memory.

4. **Instruction scheduling** reorders instructions to minimize pipeline stalls. **List scheduling** uses dependency DAGs and priority heuristics. **Software pipelining** overlaps loop iterations.

5. **Peephole optimization** applies local transformations to small windows of instructions: strength reduction, redundant code elimination, algebraic simplification, and branch optimization.

6. **Code generation for language constructs** follows predictable patterns: expressions use post-order traversal, control flow uses conditional branches and labels, and function calls follow calling conventions.

7. A **stack machine** provides a simple code generation target where all operations use an implicit stack, eliminating the need for register allocation.

---

## Exercises

### Exercise 1: Maximal Munch

Given the following IR tree, apply the Maximal Munch algorithm to select instructions. Show each step and the final instruction sequence.

```
        STORE
       /     \
      +       MEM
     / \       |
    FP  #-8   +
             / \
            *   #4
           / \
          i   #4
```

Available tiles: register, constant, ADD, ADDI, MUL, SHL (shift left for power-of-2 multiply), LOAD with indexed addressing, STORE with indexed addressing.

### Exercise 2: Register Allocation

Given the following live intervals and 3 physical registers, perform linear scan register allocation. Which variable(s) get spilled?

```
a: [1, 15]
b: [2, 10]
c: [3, 12]
d: [5, 8]
e: [7, 20]
f: [13, 18]
```

### Exercise 3: Instruction Scheduling

Schedule the following instructions for a machine with 1 ALU unit and 1 load/store unit. Load latency is 3 cycles, ALU latency is 1 cycle.

```
I1: LOAD R1, [addr1]      ; uses load unit, latency 3
I2: LOAD R2, [addr2]      ; uses load unit, latency 3
I3: ADD  R3, R1, R2       ; uses ALU, latency 1, depends on I1, I2
I4: LOAD R4, [addr3]      ; uses load unit, latency 3
I5: MUL  R5, R3, R4       ; uses ALU, latency 3, depends on I3, I4
I6: ADD  R6, R5, #1       ; uses ALU, latency 1, depends on I5
```

What is the minimum number of cycles? Draw the schedule.

### Exercise 4: Peephole Optimization

Apply peephole optimizations to the following code. Show the result after each pass.

```
  MOVI R1, #10
  ADD  R2, R1, #0
  MUL  R3, R2, #16
  MOV  R4, R4
  STORE R3, [R5]
  LOAD  R3, [R5]
  MUL  R6, R3, #1
  DIV  R7, R6, #8
  JMP  L1
  ADD  R8, R1, R2
L1:
  JMP  L2
L2:
  RET
```

### Exercise 5: Stack Machine

Hand-compile the following expression into stack machine bytecode and trace the stack contents after each instruction:

```
result = (3 + 4) * (10 - 2) / (1 + 1)
```

### Exercise 6: Implementation Challenge

Extend the stack machine code generator and VM to support:
1. **Arrays**: `ALLOC n` (allocate array of size n), `ALOAD` (load from array), `ASTORE` (store to array)
2. **For loops**: Implement `for i = start to end { body }` as a source-level construct

Test with a program that allocates an array, fills it with squares, and prints the contents.

---

[Previous: Runtime Environments](./10_Runtime_Environments.md) | [Next: Optimization — Local and Global](./12_Optimization_Local_and_Global.md) | [Overview](./00_Overview.md)
