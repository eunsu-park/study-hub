# Lesson 9: Intermediate Representations

## Learning Objectives

After completing this lesson, you will be able to:

1. **Explain** why intermediate representations (IRs) are essential in modern compiler design
2. **Distinguish** between high-level, medium-level, and low-level IRs and their trade-offs
3. **Generate** three-address code (TAC) from an abstract syntax tree
4. **Construct** control flow graphs (CFGs) and identify basic blocks
5. **Convert** code into Static Single Assignment (SSA) form and explain phi functions
6. **Implement** a TAC generator and CFG builder in Python
7. **Describe** DAG representations and their role in optimization

---

## 1. Why Intermediate Representations Matter

### 1.1 The Compiler's Bridge

A compiler transforms source code written in a high-level language into machine code for a specific target architecture. Without an intermediate representation, you would need a separate compiler front end for every (source language, target machine) pair. If you have $m$ source languages and $n$ target machines, the naive approach requires $m \times n$ translators.

An IR decouples the front end from the back end:

```
          Front Ends                Back Ends
Source 1 ──┐                    ┌── Target A
Source 2 ──┼──▶  IR  ──▶──┼── Target B
Source 3 ──┘                    └── Target C
```

With an IR, you need only $m$ front ends and $n$ back ends, reducing the total effort to $m + n$.

### 1.2 Enabling Optimization

IRs serve as the substrate on which optimizations operate. A well-designed IR makes transformations easier to express, verify, and compose. Different IRs expose different optimization opportunities:

- **High-level IR**: loop transformations, inlining, devirtualization
- **Medium-level IR**: constant propagation, dead code elimination, register promotion
- **Low-level IR**: instruction selection, register allocation, scheduling

### 1.3 Portability and Reuse

Real-world examples demonstrate the power of a good IR:

| Compiler | IR | Purpose |
|----------|----|---------|
| GCC | GIMPLE, RTL | Multi-language, multi-target |
| LLVM | LLVM IR | Language-agnostic optimizer and code generator |
| JVM | Java bytecode | Platform-independent execution |
| .NET | CIL (Common Intermediate Language) | Multi-language runtime |
| WebAssembly | Wasm binary format | Portable web execution |

### 1.4 Design Goals for an IR

A good IR should be:

1. **Easy to produce** from the source language AST
2. **Easy to translate** to target machine code
3. **Amenable to optimization** -- exposing opportunities clearly
4. **Compact** -- not wasting memory for large programs
5. **Well-defined** -- unambiguous semantics for every construct

---

## 2. Spectrum of Intermediate Representations

### 2.1 High-Level IR

A high-level IR retains much of the structure of the source language: loops, conditionals, array accesses with indices, and structured types. It is close to an abstract syntax tree (AST) or an annotated version of it.

**Example**: A high-level IR might represent a `for` loop as a single node:

```
ForLoop(
    init = Assign(i, 0),
    cond = LessThan(i, n),
    step = Assign(i, Add(i, 1)),
    body = [
        Assign(ArrayAccess(a, i), Multiply(i, i))
    ]
)
```

**Advantages**:
- Preserves source-level semantics for high-level optimizations (loop interchange, loop fusion)
- Natural for type checking and semantic analysis

**Disadvantages**:
- Too abstract for low-level optimizations (register allocation, instruction scheduling)

### 2.2 Medium-Level IR

A medium-level IR (sometimes called a "three-address code" level) removes high-level control structures and replaces them with labels and jumps. Variables are explicit, but machine details (registers, addressing modes) are abstracted away.

**Example**: The same loop in medium-level IR:

```
    i = 0
L1: if i >= n goto L2
    t1 = i * i
    a[i] = t1
    i = i + 1
    goto L1
L2:
```

**Advantages**:
- Ideal for most classical optimizations (CSE, constant propagation, DCE)
- Simple enough for straightforward analysis

**Disadvantages**:
- Loses source-level structure (harder to do loop transformations)

### 2.3 Low-Level IR

A low-level IR is close to machine code but still somewhat abstracted. It uses virtual registers (an unlimited supply) and exposes machine-specific operations like addressing modes, condition codes, and specific instruction formats.

**Example**: The same loop in a low-level IR for an x86-like target:

```
    mov v1, 0          ; i = 0
L1: cmp v1, v2         ; compare i with n
    jge L2             ; if i >= n goto L2
    mov v3, v1
    imul v3, v1        ; t1 = i * i
    mov [v4 + v1*4], v3 ; a[i] = t1
    add v1, 1          ; i = i + 1
    jmp L1
L2:
```

**Advantages**:
- Close to final output -- easier instruction selection and register allocation
- Can express machine-specific optimizations

**Disadvantages**:
- Machine-dependent, reducing portability

### 2.4 Multi-Level IR Strategies

Modern compilers often use multiple IR levels. For example:

```
Source ──▶ AST ──▶ High IR ──▶ Medium IR ──▶ Low IR ──▶ Machine Code
             (type check)  (loop opts)  (scalar opts)  (reg alloc)
```

LLVM uses a single IR (LLVM IR) but at multiple levels of abstraction through progressive lowering passes. GCC uses GIMPLE (medium-level) and RTL (low-level).

---

## 3. Three-Address Code (TAC)

### 3.1 Definition

Three-address code is one of the most widely used intermediate representations. Each instruction has at most three operands (addresses): typically one result and one or two arguments.

The general form is:

$$x = y \;\text{op}\; z$$

where $x$, $y$, $z$ are names (variables, temporaries, or constants) and $\text{op}$ is an operator.

### 3.2 Instruction Types

TAC encompasses several instruction forms:

| Category | Form | Example |
|----------|------|---------|
| Assignment (binary) | `x = y op z` | `t1 = a + b` |
| Assignment (unary) | `x = op y` | `t2 = -t1` |
| Copy | `x = y` | `t3 = t1` |
| Unconditional jump | `goto L` | `goto L2` |
| Conditional jump | `if x relop y goto L` | `if t1 < n goto L1` |
| Indexed assign (store) | `x[i] = y` | `a[t2] = t3` |
| Indexed assign (load) | `x = y[i]` | `t4 = a[t2]` |
| Address/pointer | `x = &y`, `x = *y`, `*x = y` | `t5 = &a` |
| Procedure call | `param x`, `call p, n`, `x = call p, n` | `param t1` |
| Return | `return x` | `return t1` |

### 3.3 Temporaries

TAC introduces temporary variables ($t_1, t_2, \ldots$) to hold intermediate results. This decomposition ensures that every complex expression is broken down into simple steps.

**Source expression**: `a + b * c - d / e`

**TAC**:
```
t1 = b * c
t2 = a + t1
t3 = d / e
t4 = t2 - t3
```

### 3.4 Data Structures for TAC

There are three common ways to store TAC instructions:

#### Quadruples

A quadruple has four fields: `(op, arg1, arg2, result)`.

| Index | op | arg1 | arg2 | result |
|-------|----|------|------|--------|
| 0 | `*` | `b` | `c` | `t1` |
| 1 | `+` | `a` | `t1` | `t2` |
| 2 | `/` | `d` | `e` | `t3` |
| 3 | `-` | `t2` | `t3` | `t4` |

**Advantages**: Simple, direct, easy to reorder instructions.

**Disadvantages**: Requires explicit names for all temporaries.

#### Triples

A triple has three fields: `(op, arg1, arg2)`. Results are identified by the triple's index rather than by a named temporary.

| Index | op | arg1 | arg2 |
|-------|----|------|------|
| (0) | `*` | `b` | `c` |
| (1) | `+` | `a` | (0) |
| (2) | `/` | `d` | `e` |
| (3) | `-` | (1) | (2) |

**Advantages**: No explicit temporaries needed -- more compact.

**Disadvantages**: Reordering instructions is difficult because references use indices.

#### Indirect Triples

An indirect triple adds an indirection array that maps instruction positions to triple indices. This allows reordering by permuting the indirection array without modifying the triples themselves.

```
Instruction list: [0, 1, 2, 3]   <-- can be reordered
Triple table:     same as above   <-- unchanged
```

### 3.5 Python Implementation: TAC Generator

Let us build a TAC generator that translates a simple expression AST into three-address code.

```python
"""Three-Address Code (TAC) Generator from AST."""

from dataclasses import dataclass, field
from typing import Union


# ---------- AST Nodes ----------

@dataclass
class Num:
    """Numeric literal."""
    value: int

@dataclass
class Var:
    """Variable reference."""
    name: str

@dataclass
class BinOp:
    """Binary operation."""
    op: str          # '+', '-', '*', '/'
    left: 'Expr'
    right: 'Expr'

@dataclass
class UnaryOp:
    """Unary operation."""
    op: str          # '-', 'not'
    operand: 'Expr'

@dataclass
class Assign:
    """Assignment statement."""
    target: str
    value: 'Expr'

@dataclass
class IfElse:
    """If-else statement."""
    condition: 'Expr'
    then_body: list
    else_body: list

@dataclass
class While:
    """While loop."""
    condition: 'Expr'
    body: list

@dataclass
class RelOp:
    """Relational comparison."""
    op: str          # '<', '>', '<=', '>=', '==', '!='
    left: 'Expr'
    right: 'Expr'

Expr = Union[Num, Var, BinOp, UnaryOp, RelOp]
Stmt = Union[Assign, IfElse, While]


# ---------- TAC Instruction ----------

@dataclass
class TACInstruction:
    """A single TAC instruction."""
    op: str
    arg1: str = ""
    arg2: str = ""
    result: str = ""

    def __str__(self):
        if self.op == "label":
            return f"{self.result}:"
        elif self.op == "goto":
            return f"    goto {self.result}"
        elif self.op == "iffalse":
            return f"    iffalse {self.arg1} goto {self.result}"
        elif self.op == "iftrue":
            return f"    iftrue {self.arg1} goto {self.result}"
        elif self.op == "copy":
            return f"    {self.result} = {self.arg1}"
        elif self.op == "param":
            return f"    param {self.arg1}"
        elif self.op == "call":
            return f"    {self.result} = call {self.arg1}, {self.arg2}"
        elif self.op == "return":
            return f"    return {self.arg1}"
        elif self.arg2:
            return f"    {self.result} = {self.arg1} {self.op} {self.arg2}"
        elif self.arg1:
            return f"    {self.result} = {self.op} {self.arg1}"
        else:
            return f"    {self.op}"


# ---------- TAC Generator ----------

class TACGenerator:
    """Generates three-address code from an AST."""

    def __init__(self):
        self.instructions: list[TACInstruction] = []
        self._temp_counter = 0
        self._label_counter = 0

    def new_temp(self) -> str:
        """Generate a fresh temporary variable name."""
        self._temp_counter += 1
        return f"t{self._temp_counter}"

    def new_label(self) -> str:
        """Generate a fresh label name."""
        self._label_counter += 1
        return f"L{self._label_counter}"

    def emit(self, op: str, arg1: str = "", arg2: str = "",
             result: str = "") -> TACInstruction:
        """Emit a single TAC instruction."""
        instr = TACInstruction(op, arg1, arg2, result)
        self.instructions.append(instr)
        return instr

    def generate(self, node) -> str:
        """Generate TAC for a node. Returns the name holding the result."""
        if isinstance(node, Num):
            return str(node.value)

        elif isinstance(node, Var):
            return node.name

        elif isinstance(node, BinOp):
            left = self.generate(node.left)
            right = self.generate(node.right)
            temp = self.new_temp()
            self.emit(node.op, left, right, temp)
            return temp

        elif isinstance(node, UnaryOp):
            operand = self.generate(node.operand)
            temp = self.new_temp()
            self.emit(node.op, operand, result=temp)
            return temp

        elif isinstance(node, RelOp):
            left = self.generate(node.left)
            right = self.generate(node.right)
            temp = self.new_temp()
            self.emit(node.op, left, right, temp)
            return temp

        elif isinstance(node, Assign):
            value = self.generate(node.value)
            self.emit("copy", value, result=node.target)
            return node.target

        elif isinstance(node, IfElse):
            return self._generate_if_else(node)

        elif isinstance(node, While):
            return self._generate_while(node)

        else:
            raise ValueError(f"Unknown node type: {type(node)}")

    def _generate_if_else(self, node: IfElse) -> str:
        """Generate TAC for an if-else statement."""
        cond = self.generate(node.condition)
        else_label = self.new_label()
        end_label = self.new_label()

        self.emit("iffalse", cond, result=else_label)

        # Then body
        for stmt in node.then_body:
            self.generate(stmt)

        self.emit("goto", result=end_label)

        # Else label
        self.emit("label", result=else_label)

        # Else body
        for stmt in node.else_body:
            self.generate(stmt)

        # End label
        self.emit("label", result=end_label)
        return ""

    def _generate_while(self, node: While) -> str:
        """Generate TAC for a while loop."""
        begin_label = self.new_label()
        end_label = self.new_label()

        # Loop start
        self.emit("label", result=begin_label)

        # Condition
        cond = self.generate(node.condition)
        self.emit("iffalse", cond, result=end_label)

        # Body
        for stmt in node.body:
            self.generate(stmt)

        self.emit("goto", result=begin_label)

        # Loop end
        self.emit("label", result=end_label)
        return ""

    def print_tac(self):
        """Pretty-print all generated TAC instructions."""
        for instr in self.instructions:
            print(instr)


# ---------- Example Usage ----------

def demo_expression():
    """Generate TAC for: result = (a + b) * (c - d)"""
    print("=== Expression: result = (a + b) * (c - d) ===")
    ast = Assign("result",
        BinOp("*",
            BinOp("+", Var("a"), Var("b")),
            BinOp("-", Var("c"), Var("d"))
        )
    )
    gen = TACGenerator()
    gen.generate(ast)
    gen.print_tac()
    print()


def demo_if_else():
    """Generate TAC for: if (x < y) max = y; else max = x;"""
    print("=== If-Else: if (x < y) max = y; else max = x; ===")
    ast = IfElse(
        condition=RelOp("<", Var("x"), Var("y")),
        then_body=[Assign("max", Var("y"))],
        else_body=[Assign("max", Var("x"))]
    )
    gen = TACGenerator()
    gen.generate(ast)
    gen.print_tac()
    print()


def demo_while():
    """Generate TAC for: while (i < n) { s = s + a[i]; i = i + 1; }"""
    print("=== While: while (i < n) { s = s + i; i = i + 1; } ===")
    ast = While(
        condition=RelOp("<", Var("i"), Var("n")),
        body=[
            Assign("s", BinOp("+", Var("s"), Var("i"))),
            Assign("i", BinOp("+", Var("i"), Num(1)))
        ]
    )
    gen = TACGenerator()
    gen.generate(ast)
    gen.print_tac()
    print()


if __name__ == "__main__":
    demo_expression()
    demo_if_else()
    demo_while()
```

**Expected output for `demo_expression()`**:
```
=== Expression: result = (a + b) * (c - d) ===
    t1 = a + b
    t2 = c - d
    t3 = t1 * t2
    result = t3
```

**Expected output for `demo_while()`**:
```
=== While: while (i < n) { s = s + i; i = i + 1; } ===
L1:
    t1 = i < n
    iffalse t1 goto L2
    t2 = s + i
    s = t2
    t3 = i + 1
    i = t3
    goto L1
L2:
```

---

## 4. Control Flow Graphs (CFGs)

### 4.1 Motivation

Once we have three-address code, the linear sequence of instructions does not immediately reveal the structure of the program's execution flow. A control flow graph (CFG) makes this structure explicit by organizing instructions into **basic blocks** connected by **edges** that represent possible transfers of control.

A CFG is a directed graph $G = (V, E)$ where:
- Each node $v \in V$ is a **basic block**
- Each edge $(u, v) \in E$ represents a possible flow of control from block $u$ to block $v$
- There is a distinguished **entry** node and one or more **exit** nodes

### 4.2 Basic Blocks

A **basic block** is a maximal sequence of consecutive instructions such that:

1. **Control enters only at the first instruction** (the leader)
2. **Control leaves only at the last instruction** (no jumps into or out of the middle)
3. **All instructions execute sequentially** if the block is entered

This means that within a basic block, every instruction executes if the first one does, making the block an atomic unit for analysis.

### 4.3 Identifying Leaders

To partition TAC into basic blocks, we identify **leaders** -- instructions that begin a new block:

1. **The first instruction** in the program is a leader
2. **Any instruction that is the target of a jump** (conditional or unconditional) is a leader
3. **Any instruction that immediately follows a jump** is a leader

**Algorithm: Identifying Basic Blocks**

```
Input:  List of TAC instructions
Output: List of basic blocks

1. Determine the set of leaders:
   a. First instruction is a leader
   b. Target of any goto/branch is a leader
   c. Instruction after any goto/branch is a leader

2. For each leader, its basic block consists of the leader and
   all instructions up to (but not including) the next leader
   or the end of the program.
```

### 4.4 Building the CFG

After identifying basic blocks, we connect them with edges:

1. If block $B_i$ ends with a conditional branch `if ... goto L`:
   - Add an edge from $B_i$ to the block starting at label $L$ (true branch)
   - Add a **fall-through** edge from $B_i$ to $B_{i+1}$ (false branch)

2. If block $B_i$ ends with an unconditional `goto L`:
   - Add an edge from $B_i$ to the block starting at label $L$
   - No fall-through edge

3. If block $B_i$ ends with neither a branch nor a goto:
   - Add a fall-through edge from $B_i$ to $B_{i+1}$

### 4.5 Example: CFG Construction

Consider the TAC from our while loop example:

```
(0)  L1:                  <-- Block B1 leader (label target)
(1)  t1 = i < n
(2)  iffalse t1 goto L2
(3)  t2 = s + i           <-- Block B2 leader (follows branch)
(4)  s = t2
(5)  t3 = i + 1
(6)  i = t3
(7)  goto L1
(8)  L2:                  <-- Block B3 leader (label target)
```

**Leaders**: instruction 0 (first + label target), instruction 3 (follows branch), instruction 8 (label target)

**Basic Blocks**:
- $B_1$: instructions 0-2 (condition test)
- $B_2$: instructions 3-7 (loop body)
- $B_3$: instruction 8 (loop exit)

**CFG**:
```
    ┌─────────┐
    │   B1    │ ◀────────┐
    │ L1:     │          │
    │ t1=i<n  │          │
    │ if !t1  │          │
    │ goto L2 │          │
    └────┬────┘          │
    true │  \false       │
         │   \           │
         │  ┌──▼──────┐  │
         │  │   B2    │  │
         │  │ t2=s+i  │  │
         │  │ s=t2    │  │
         │  │ t3=i+1  │  │
         │  │ i=t3    │  │
         │  │ goto L1 │──┘
         │  └─────────┘
         │
    ┌────▼────┐
    │   B3    │
    │ L2:     │
    └─────────┘
```

Note: The edge labels depend on the semantics of `iffalse`. If the condition is false, we go to L2 (B3). If the condition is true (i.e., `iffalse` does *not* jump), we fall through to B2.

### 4.6 Python Implementation: CFG Builder

```python
"""Control Flow Graph (CFG) Builder from TAC instructions."""

from dataclasses import dataclass, field


@dataclass
class TACInstr:
    """A TAC instruction for CFG construction."""
    index: int
    op: str
    arg1: str = ""
    arg2: str = ""
    result: str = ""

    @property
    def is_label(self) -> bool:
        return self.op == "label"

    @property
    def is_jump(self) -> bool:
        return self.op in ("goto", "iffalse", "iftrue")

    @property
    def is_unconditional_jump(self) -> bool:
        return self.op == "goto"

    @property
    def is_conditional_jump(self) -> bool:
        return self.op in ("iffalse", "iftrue")

    @property
    def jump_target(self) -> str:
        """Return the label this instruction jumps to, if any."""
        if self.is_jump:
            return self.result
        return ""

    def __str__(self):
        if self.op == "label":
            return f"{self.result}:"
        elif self.op == "goto":
            return f"  goto {self.result}"
        elif self.op in ("iffalse", "iftrue"):
            return f"  {self.op} {self.arg1} goto {self.result}"
        elif self.op == "copy":
            return f"  {self.result} = {self.arg1}"
        elif self.arg2:
            return f"  {self.result} = {self.arg1} {self.op} {self.arg2}"
        else:
            return f"  {self.result} = {self.op} {self.arg1}"


@dataclass
class BasicBlock:
    """A basic block in the CFG."""
    id: int
    instructions: list = field(default_factory=list)
    successors: list = field(default_factory=list)   # Block IDs
    predecessors: list = field(default_factory=list)  # Block IDs

    @property
    def leader(self):
        """The first instruction in the block."""
        return self.instructions[0] if self.instructions else None

    @property
    def terminator(self):
        """The last instruction in the block."""
        return self.instructions[-1] if self.instructions else None

    @property
    def label(self) -> str:
        """Return the label of this block, if it starts with one."""
        if self.instructions and self.instructions[0].is_label:
            return self.instructions[0].result
        return ""

    def __str__(self):
        lines = [f"Block B{self.id}:"]
        lines.append(f"  Predecessors: {[f'B{p}' for p in self.predecessors]}")
        lines.append(f"  Successors:   {[f'B{s}' for s in self.successors]}")
        lines.append("  Instructions:")
        for instr in self.instructions:
            lines.append(f"    {instr}")
        return "\n".join(lines)


class CFG:
    """Control Flow Graph."""

    def __init__(self):
        self.blocks: dict[int, BasicBlock] = {}
        self.entry_id: int = -1
        self.exit_ids: list[int] = []

    def add_block(self, block: BasicBlock):
        """Add a basic block to the CFG."""
        self.blocks[block.id] = block

    def add_edge(self, from_id: int, to_id: int):
        """Add a directed edge between two blocks."""
        if to_id not in self.blocks[from_id].successors:
            self.blocks[from_id].successors.append(to_id)
        if from_id not in self.blocks[to_id].predecessors:
            self.blocks[to_id].predecessors.append(from_id)

    def __str__(self):
        lines = [f"=== Control Flow Graph ({len(self.blocks)} blocks) ==="]
        lines.append(f"Entry: B{self.entry_id}")
        lines.append(f"Exit:  {[f'B{e}' for e in self.exit_ids]}")
        lines.append("")
        for block_id in sorted(self.blocks.keys()):
            lines.append(str(self.blocks[block_id]))
            lines.append("")
        return "\n".join(lines)


def build_cfg(instructions: list[TACInstr]) -> CFG:
    """
    Build a Control Flow Graph from a list of TAC instructions.

    Algorithm:
    1. Identify leaders (first instruction, jump targets, post-jump instructions)
    2. Partition instructions into basic blocks
    3. Connect blocks with edges based on control flow
    """
    if not instructions:
        return CFG()

    # Step 1: Identify leader indices
    leaders = set()
    leaders.add(0)  # First instruction is always a leader

    # Map labels to instruction indices
    label_to_index = {}
    for instr in instructions:
        if instr.is_label:
            label_to_index[instr.result] = instr.index

    for instr in instructions:
        if instr.is_jump:
            # Target of jump is a leader
            target_label = instr.jump_target
            if target_label in label_to_index:
                leaders.add(label_to_index[target_label])

            # Instruction after jump is a leader (if it exists)
            next_idx = instr.index + 1
            if next_idx < len(instructions):
                leaders.add(next_idx)

    sorted_leaders = sorted(leaders)

    # Step 2: Partition into basic blocks
    cfg = CFG()
    block_id = 0
    leader_to_block_id = {}

    for i, leader_idx in enumerate(sorted_leaders):
        # Determine the end of this block
        if i + 1 < len(sorted_leaders):
            end_idx = sorted_leaders[i + 1]
        else:
            end_idx = len(instructions)

        block = BasicBlock(id=block_id)
        block.instructions = instructions[leader_idx:end_idx]
        cfg.add_block(block)
        leader_to_block_id[leader_idx] = block_id
        block_id += 1

    # Build label -> block_id mapping
    label_to_block_id = {}
    for bid, block in cfg.blocks.items():
        label = block.label
        if label:
            label_to_block_id[label] = bid

    # Step 3: Add edges
    block_ids = sorted(cfg.blocks.keys())
    for i, bid in enumerate(block_ids):
        block = cfg.blocks[bid]
        term = block.terminator

        if term is None:
            continue

        if term.is_unconditional_jump:
            # Edge to jump target only
            target_label = term.jump_target
            if target_label in label_to_block_id:
                cfg.add_edge(bid, label_to_block_id[target_label])

        elif term.is_conditional_jump:
            # Edge to jump target
            target_label = term.jump_target
            if target_label in label_to_block_id:
                cfg.add_edge(bid, label_to_block_id[target_label])
            # Fall-through edge to next block
            if i + 1 < len(block_ids):
                cfg.add_edge(bid, block_ids[i + 1])

        else:
            # No jump: fall through to next block
            if i + 1 < len(block_ids):
                cfg.add_edge(bid, block_ids[i + 1])

    # Set entry and exit
    cfg.entry_id = block_ids[0] if block_ids else -1
    for bid in block_ids:
        block = cfg.blocks[bid]
        if not block.successors:
            cfg.exit_ids.append(bid)

    return cfg


# ---------- Example: Build CFG for a While Loop ----------

def demo_while_loop_cfg():
    """Build CFG for: while (i < n) { s = s + i; i = i + 1; }"""
    instructions = [
        TACInstr(0, "label", result="L1"),
        TACInstr(1, "<", "i", "n", "t1"),
        TACInstr(2, "iffalse", "t1", result="L2"),
        TACInstr(3, "+", "s", "i", "t2"),
        TACInstr(4, "copy", "t2", result="s"),
        TACInstr(5, "+", "i", "1", "t3"),
        TACInstr(6, "copy", "t3", result="i"),
        TACInstr(7, "goto", result="L1"),
        TACInstr(8, "label", result="L2"),
    ]

    cfg = build_cfg(instructions)
    print(cfg)


def demo_if_else_cfg():
    """Build CFG for: if (x < y) max = y; else max = x;"""
    instructions = [
        TACInstr(0, "<", "x", "y", "t1"),
        TACInstr(1, "iffalse", "t1", result="L1"),
        TACInstr(2, "copy", "y", result="max"),
        TACInstr(3, "goto", result="L2"),
        TACInstr(4, "label", result="L1"),
        TACInstr(5, "copy", "x", result="max"),
        TACInstr(6, "label", result="L2"),
    ]

    cfg = build_cfg(instructions)
    print(cfg)


if __name__ == "__main__":
    print("--- While Loop CFG ---")
    demo_while_loop_cfg()
    print()
    print("--- If-Else CFG ---")
    demo_if_else_cfg()
```

---

## 5. Static Single Assignment (SSA) Form

### 5.1 What Is SSA?

Static Single Assignment (SSA) form is an IR property in which every variable is assigned exactly once. If a variable in the original program is assigned in multiple places, SSA creates distinct **versions** of that variable, each with its own unique name.

**Original code**:
```
x = 1
x = x + 1
y = x * 2
```

**SSA form**:
```
x1 = 1
x2 = x1 + 1
y1 = x2 * 2
```

Each definition creates a new version. This property makes many optimizations simpler and more efficient because the definition of every variable is unique and easy to locate.

### 5.2 The Need for Phi Functions

A complication arises at **join points** in the control flow graph -- places where two or more paths converge. Consider:

```
if (cond)
    x = 1       // Path A
else
    x = 2       // Path B
y = x + 3       // Which x?
```

After the if-else, which version of `x` should we use? We cannot statically determine which path was taken at runtime. SSA resolves this with a **phi function** ($\phi$-function):

```
if (cond)
    x1 = 1       // Path A
else
    x2 = 2       // Path B
x3 = phi(x1, x2) // Merge point
y1 = x3 + 3
```

The phi function $x_3 = \phi(x_1, x_2)$ means: "if control came from Path A, use $x_1$; if from Path B, use $x_2$."

**Formally**: A $\phi$-function at the entry of block $B$ with $k$ predecessors $B_1, B_2, \ldots, B_k$ has the form:

$$x_i = \phi(x_{j_1}, x_{j_2}, \ldots, x_{j_k})$$

where $x_{j_m}$ is the version of $x$ reaching from predecessor $B_m$.

### 5.3 Properties of SSA

1. **Unique definitions**: Every variable has exactly one definition point
2. **Use-def chains are trivial**: Finding the definition of a variable at a use site is immediate
3. **Def-use chains are compact**: Each definition has a clear set of uses
4. **Sparse representation**: Analysis operates on variables rather than program points

### 5.4 Dominance and Dominance Frontiers

To efficiently place $\phi$-functions, we need the concept of **dominance**.

**Definition**: A node $d$ **dominates** node $n$ in a CFG (written $d \;\text{dom}\; n$) if every path from the entry node to $n$ must pass through $d$.

**Immediate dominator**: Node $d$ is the **immediate dominator** of $n$ (written $\text{idom}(n) = d$) if $d$ dominates $n$, $d \neq n$, and every other dominator of $n$ also dominates $d$.

**Dominator tree**: The immediate dominator relationship forms a tree rooted at the entry node.

**Dominance frontier**: The **dominance frontier** of a node $d$, written $DF(d)$, is the set of nodes $n$ such that:
- $d$ dominates a predecessor of $n$, but
- $d$ does not strictly dominate $n$

Intuitively, the dominance frontier of $d$ is the set of nodes where $d$'s dominance "ends" -- these are exactly the join points where $\phi$-functions for variables defined in $d$ may be needed.

### 5.5 Algorithm: Computing Dominance Frontiers

The following algorithm by Cooper, Harvey, and Kennedy (2001) efficiently computes dominance frontiers:

```
for each node b:
    if b has multiple predecessors:
        for each predecessor p of b:
            runner = p
            while runner != idom(b):
                DF(runner) = DF(runner) ∪ {b}
                runner = idom(runner)
```

### 5.6 Algorithm: Placing Phi Functions

The classic algorithm for placing $\phi$-functions uses dominance frontiers:

```
Input:  CFG with dominance frontiers, set of variables
Output: Placement of phi-functions

For each variable v:
    worklist = set of blocks that define v
    ever_on_worklist = copy of worklist
    has_phi = empty set

    while worklist is not empty:
        remove a block b from worklist
        for each block d in DF(b):
            if d not in has_phi:
                insert phi-function for v at the start of d
                has_phi = has_phi ∪ {d}
                if d not in ever_on_worklist:
                    ever_on_worklist = ever_on_worklist ∪ {d}
                    add d to worklist
```

### 5.7 Algorithm: Variable Renaming

After placing $\phi$-functions, we rename variables by walking the dominator tree:

```
counter[v] = 0 for all variables v
stack[v] = empty for all variables v

function rename(block b):
    for each phi-function "v = phi(...)" in b:
        i = counter[v]++
        push i onto stack[v]
        replace v in the phi with v_i

    for each instruction in b:
        for each use of variable v:
            replace v with v_{top(stack[v])}
        for each definition of variable v:
            i = counter[v]++
            push i onto stack[v]
            replace v with v_i

    for each successor s of b:
        for each phi-function in s:
            let j be the index of b in s's predecessor list
            replace the j-th operand v with v_{top(stack[v])}

    for each child c of b in the dominator tree:
        rename(c)

    for each phi or instruction in b that defined v_i:
        pop stack[v]
```

### 5.8 Complete SSA Example

Consider this program:

```
    a = 0
    b = 1
L1: if a >= n goto L2
    c = a + b
    a = b
    b = c
    goto L1
L2: return b
```

**CFG**:
```
B0: a=0, b=1
 │
 ▼
B1: if a>=n goto L2  ◀──┐
 │        │              │
 │ false  │ true         │
 ▼        │              │
B2: c=a+b │              │
    a=b   │              │
    b=c   │              │
    goto L1 ─────────────┘
          │
          ▼
B3: return b
```

**SSA Form**:
```
B0: a0 = 0
    b0 = 1
    goto B1

B1: a2 = phi(a0, a3)    // From B0 or B2
    b2 = phi(b0, b3)    // From B0 or B2
    if a2 >= n goto B3

B2: c1 = a2 + b2
    a3 = b2
    b3 = c1
    goto B1

B3: b4 = phi(b2)         // (trivial, single predecessor)
    return b4
```

Note: The $\phi$-functions are placed at B1 because it is a join point (reached from both B0 and B2).

### 5.9 Python Implementation: SSA Construction

```python
"""Simplified SSA construction for a basic CFG."""

from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class SSABlock:
    """A basic block in SSA form."""
    id: int
    phis: dict = field(default_factory=dict)     # var -> phi operands
    instructions: list = field(default_factory=list)
    predecessors: list = field(default_factory=list)
    successors: list = field(default_factory=list)
    idom: int = -1  # Immediate dominator block id
    dom_frontier: set = field(default_factory=set)
    dom_children: list = field(default_factory=list)


class SSABuilder:
    """
    Build SSA form from a simple CFG.

    This is a simplified demonstration that illustrates the key ideas:
    1. Compute dominance (simplified)
    2. Compute dominance frontiers
    3. Place phi functions
    4. Rename variables
    """

    def __init__(self):
        self.blocks: dict[int, SSABlock] = {}
        self.var_defs: dict[str, set] = defaultdict(set)  # var -> blocks defining it
        self.all_vars: set = set()

    def add_block(self, block_id: int, instructions: list,
                  predecessors: list, successors: list):
        """Add a block with its instructions and connectivity."""
        block = SSABlock(
            id=block_id,
            instructions=instructions,
            predecessors=predecessors,
            successors=successors,
        )
        self.blocks[block_id] = block

        # Track which blocks define which variables
        for instr in instructions:
            if "=" in instr and not instr.strip().startswith("if") \
               and not instr.strip().startswith("goto") \
               and not instr.strip().startswith("return"):
                var = instr.split("=")[0].strip()
                self.var_defs[var].add(block_id)
                self.all_vars.add(var)

    def compute_dominators(self, entry: int):
        """
        Compute immediate dominators using a simplified iterative algorithm.
        """
        all_ids = sorted(self.blocks.keys())

        # Initialize: entry dominates itself; others dominated by all
        dom = {b: set(all_ids) for b in all_ids}
        dom[entry] = {entry}

        changed = True
        while changed:
            changed = False
            for b in all_ids:
                if b == entry:
                    continue
                preds = self.blocks[b].predecessors
                if not preds:
                    continue
                new_dom = set.intersection(*(dom[p] for p in preds))
                new_dom.add(b)
                if new_dom != dom[b]:
                    dom[b] = new_dom
                    changed = True

        # Compute immediate dominators from dominator sets
        for b in all_ids:
            if b == entry:
                self.blocks[b].idom = -1
                continue
            strict_doms = dom[b] - {b}
            # idom is the dominator closest to b (dominated by all others)
            idom = None
            for d in strict_doms:
                if all(d in dom[other] or other == d
                       for other in strict_doms):
                    if idom is None or d not in dom.get(idom, set()) or d == idom:
                        # d should be the one dominated by fewest
                        if idom is None or idom in dom[d]:
                            idom = d
            if idom is not None:
                self.blocks[b].idom = idom
                self.blocks[idom].dom_children.append(b)

    def compute_dominance_frontiers(self):
        """Compute dominance frontiers using the Cooper-Harvey-Kennedy algorithm."""
        for b_id, block in self.blocks.items():
            if len(block.predecessors) >= 2:
                for p in block.predecessors:
                    runner = p
                    while runner != block.idom and runner != -1:
                        self.blocks[runner].dom_frontier.add(b_id)
                        runner = self.blocks[runner].idom

    def place_phi_functions(self):
        """Insert phi functions at dominance frontier nodes."""
        for var in self.all_vars:
            worklist = list(self.var_defs[var])
            ever_on_worklist = set(worklist)
            has_phi = set()

            while worklist:
                block_id = worklist.pop()
                for df_id in self.blocks[block_id].dom_frontier:
                    if df_id not in has_phi:
                        # Insert phi: var = phi(var, var, ...)
                        n_preds = len(self.blocks[df_id].predecessors)
                        self.blocks[df_id].phis[var] = [var] * n_preds
                        has_phi.add(df_id)
                        if df_id not in ever_on_worklist:
                            ever_on_worklist.add(df_id)
                            worklist.append(df_id)

    def rename_variables(self, entry: int):
        """Rename variables to achieve SSA form."""
        counter = defaultdict(int)
        stack = defaultdict(list)

        def new_name(var: str) -> str:
            i = counter[var]
            counter[var] += 1
            name = f"{var}{i}"
            stack[var].append(name)
            return name

        def current_name(var: str) -> str:
            if stack[var]:
                return stack[var][-1]
            return var + "0"

        def rename(block_id: int):
            block = self.blocks[block_id]
            push_counts = defaultdict(int)

            # Rename phi definitions
            for var in list(block.phis.keys()):
                new = new_name(var)
                push_counts[var] += 1
                # Store the SSA name for this phi
                block.phis[var] = {
                    "result": new,
                    "operands": block.phis[var]  # Will rename operands from preds
                }

            # Rename instructions
            new_instructions = []
            for instr in block.instructions:
                new_instr = instr

                # Handle assignments: replace uses then define
                if "=" in instr and not instr.strip().startswith("if") \
                   and not instr.strip().startswith("goto") \
                   and not instr.strip().startswith("return"):
                    parts = instr.split("=", 1)
                    lhs_var = parts[0].strip()
                    rhs = parts[1].strip()

                    # Rename uses in RHS
                    for v in self.all_vars:
                        if v in rhs:
                            rhs = rhs.replace(v, current_name(v))

                    # Rename definition in LHS
                    new_lhs = new_name(lhs_var)
                    push_counts[lhs_var] += 1
                    new_instr = f"{new_lhs} = {rhs}"

                elif instr.strip().startswith("return"):
                    parts = instr.strip().split()
                    if len(parts) > 1:
                        v = parts[1]
                        if v in self.all_vars:
                            new_instr = f"return {current_name(v)}"

                elif instr.strip().startswith("if"):
                    new_instr = instr
                    for v in self.all_vars:
                        if v in new_instr:
                            new_instr = new_instr.replace(v, current_name(v))

                new_instructions.append(new_instr)

            block.instructions = new_instructions

            # Fill in phi operands in successors
            for succ_id in block.successors:
                succ = self.blocks[succ_id]
                pred_index = succ.predecessors.index(block_id)
                for var, phi_info in succ.phis.items():
                    if isinstance(phi_info, dict):
                        phi_info["operands"][pred_index] = current_name(var)

            # Recurse on dominator tree children
            for child_id in block.dom_children:
                rename(child_id)

            # Pop stacks
            for var, count in push_counts.items():
                for _ in range(count):
                    if stack[var]:
                        stack[var].pop()

        rename(entry)

    def build(self, entry: int):
        """Run the full SSA construction pipeline."""
        self.compute_dominators(entry)
        self.compute_dominance_frontiers()
        self.place_phi_functions()
        self.rename_variables(entry)

    def print_ssa(self):
        """Pretty-print the SSA form."""
        for bid in sorted(self.blocks.keys()):
            block = self.blocks[bid]
            print(f"B{bid}:")

            # Print phi functions
            for var, phi_info in block.phis.items():
                if isinstance(phi_info, dict):
                    operands = ", ".join(phi_info["operands"])
                    print(f"  {phi_info['result']} = phi({operands})")

            # Print instructions
            for instr in block.instructions:
                print(f"  {instr}")
            print()


# ---------- Example ----------

def demo_fibonacci_ssa():
    """
    Convert a Fibonacci-like loop to SSA.

    Original:
        a = 0
        b = 1
    L1: if a >= n goto L2
        c = a + b
        a = b
        b = c
        goto L1
    L2: return b
    """
    print("=== Fibonacci Loop in SSA Form ===\n")

    builder = SSABuilder()

    builder.add_block(0,
        instructions=["a = 0", "b = 1"],
        predecessors=[],
        successors=[1])

    builder.add_block(1,
        instructions=["if a >= n goto L2"],
        predecessors=[0, 2],
        successors=[2, 3])

    builder.add_block(2,
        instructions=["c = a + b", "a = b", "b = c", "goto L1"],
        predecessors=[1],
        successors=[1])

    builder.add_block(3,
        instructions=["return b"],
        predecessors=[1],
        successors=[])

    builder.build(entry=0)
    builder.print_ssa()


if __name__ == "__main__":
    demo_fibonacci_ssa()
```

---

## 6. Directed Acyclic Graphs (DAGs)

### 6.1 DAGs for Expressions

A **Directed Acyclic Graph** (DAG) is a compact representation of expressions that eliminates redundant computation. Unlike a tree, a DAG allows sharing of common subexpressions.

**Expression**: `(a + b) * (a + b) - c`

**Tree representation** (redundant):
```
        -
       / \
      *   c
     / \
    +   +
   / \ / \
  a  b a  b
```

**DAG representation** (shared):
```
        -
       / \
      *   c
     /|
    +
   / \
  a   b
```

In the DAG, the common subexpression `a + b` is computed once and its result is shared.

### 6.2 Constructing DAGs

The algorithm to build a DAG from an expression processes the expression bottom-up:

```
function build_dag(expr):
    if expr is a leaf (variable or constant):
        if node for expr already exists:
            return existing node
        else:
            create new leaf node
            return it

    if expr is "left op right":
        left_node = build_dag(left)
        right_node = build_dag(right)

        if interior node (op, left_node, right_node) already exists:
            return existing node
        else:
            create new interior node
            return it
```

**Key**: We use a hash table to check for existing nodes with the same (op, left, right) triple.

### 6.3 Uses of DAGs

1. **Common subexpression elimination**: Shared nodes represent computations done once
2. **Instruction ordering**: DAG edges define a partial order; topological sort gives a valid execution order
3. **Dead code detection**: Nodes with no outgoing uses (except the final result) may be dead
4. **Algebraic optimization**: Simplification rules can be applied to DAG nodes

### 6.4 Python Implementation: Expression DAG

```python
"""DAG construction for common subexpression elimination."""

from dataclasses import dataclass, field


@dataclass
class DAGNode:
    """A node in the expression DAG."""
    id: int
    op: str           # Operator or "leaf"
    value: str = ""   # For leaf nodes: variable name or constant
    left: int = -1    # Left child node id
    right: int = -1   # Right child node id
    labels: list = field(default_factory=list)  # Variable names assigned to this node

    def __str__(self):
        if self.op == "leaf":
            label_str = f" [{', '.join(self.labels)}]" if self.labels else ""
            return f"n{self.id}: {self.value}{label_str}"
        else:
            label_str = f" [{', '.join(self.labels)}]" if self.labels else ""
            return f"n{self.id}: n{self.left} {self.op} n{self.right}{label_str}"


class ExpressionDAG:
    """Build and manage an expression DAG."""

    def __init__(self):
        self.nodes: dict[int, DAGNode] = {}
        self._next_id = 0
        self._leaf_map: dict[str, int] = {}         # value -> node_id
        self._interior_map: dict[tuple, int] = {}    # (op, left, right) -> node_id

    def _new_id(self) -> int:
        nid = self._next_id
        self._next_id += 1
        return nid

    def find_or_create_leaf(self, value: str) -> int:
        """Find or create a leaf node for a variable/constant."""
        if value in self._leaf_map:
            return self._leaf_map[value]
        nid = self._new_id()
        node = DAGNode(id=nid, op="leaf", value=value)
        node.labels.append(value)
        self.nodes[nid] = node
        self._leaf_map[value] = nid
        return nid

    def find_or_create_op(self, op: str, left_id: int, right_id: int) -> int:
        """Find or create an interior node for an operation."""
        key = (op, left_id, right_id)

        # Check commutativity for + and *
        if op in ("+", "*"):
            alt_key = (op, right_id, left_id)
            if alt_key in self._interior_map:
                return self._interior_map[alt_key]

        if key in self._interior_map:
            return self._interior_map[key]

        nid = self._new_id()
        node = DAGNode(id=nid, op=op, left=left_id, right=right_id)
        self.nodes[nid] = node
        self._interior_map[key] = nid
        return nid

    def assign_label(self, node_id: int, var_name: str):
        """Assign a variable name to a DAG node (for variable tracking)."""
        node = self.nodes[node_id]
        if var_name not in node.labels:
            node.labels.append(var_name)
        # Update leaf map so future references to this variable use this node
        self._leaf_map[var_name] = node_id

    def process_tac(self, instructions: list[str]):
        """
        Process a sequence of TAC instructions and build the DAG.

        Each instruction should be of the form:
            'x = y op z'  or  'x = y'
        """
        for instr in instructions:
            parts = instr.strip().split()

            if len(parts) == 3:
                # Copy: x = y
                target, _, source = parts
                source_id = self.find_or_create_leaf(source)
                self.assign_label(source_id, target)

            elif len(parts) == 5:
                # Binary: x = y op z
                target, _, left, op, right = parts
                left_id = self.find_or_create_leaf(left)
                right_id = self.find_or_create_leaf(right)
                result_id = self.find_or_create_op(op, left_id, right_id)
                self.assign_label(result_id, target)

    def print_dag(self):
        """Pretty-print the DAG."""
        print("=== Expression DAG ===")
        for nid in sorted(self.nodes.keys()):
            print(f"  {self.nodes[nid]}")

    def generate_optimized_tac(self) -> list[str]:
        """
        Generate optimized TAC from the DAG using topological order.
        Common subexpressions are computed only once.
        """
        # Topological sort (using DFS post-order)
        visited = set()
        order = []

        def dfs(nid):
            if nid in visited or nid == -1:
                return
            visited.add(nid)
            node = self.nodes[nid]
            if node.left != -1:
                dfs(node.left)
            if node.right != -1:
                dfs(node.right)
            order.append(nid)

        # Start DFS from all root-like nodes (those not used as children)
        children = set()
        for n in self.nodes.values():
            if n.left != -1:
                children.add(n.left)
            if n.right != -1:
                children.add(n.right)
        roots = [nid for nid in self.nodes if nid not in children]

        for root in roots:
            dfs(root)

        # Generate TAC
        tac = []
        node_result_name = {}

        for nid in order:
            node = self.nodes[nid]
            if node.op == "leaf":
                # Leaf nodes: the variable/constant itself is the name
                node_result_name[nid] = node.value
            else:
                left_name = node_result_name.get(node.left, "?")
                right_name = node_result_name.get(node.right, "?")
                result_name = node.labels[0] if node.labels else f"t{nid}"
                tac.append(f"{result_name} = {left_name} {node.op} {right_name}")
                node_result_name[nid] = result_name

        return tac


# ---------- Example ----------

def demo_cse_dag():
    """
    Demonstrate CSE via DAG for:
        t1 = a + b
        t2 = a + b    <-- common subexpression
        t3 = t1 * t2
        t4 = t3 - c
    """
    print("Original TAC:")
    instructions = [
        "t1 = a + b",
        "t2 = a + b",
        "t3 = t1 * t2",
        "t4 = t3 - c",
    ]
    for instr in instructions:
        print(f"  {instr}")
    print()

    dag = ExpressionDAG()
    dag.process_tac(instructions)
    dag.print_dag()
    print()

    optimized = dag.generate_optimized_tac()
    print("Optimized TAC (CSE eliminated):")
    for instr in optimized:
        print(f"  {instr}")


if __name__ == "__main__":
    demo_cse_dag()
```

**Expected output**:
```
Original TAC:
  t1 = a + b
  t2 = a + b
  t3 = t1 * t2
  t4 = t3 - c

=== Expression DAG ===
  n0: a [a]
  n1: b [b]
  n2: n0 + n1 [t1, t2]
  n3: n2 * n2 [t3]
  n4: c [c]
  n5: n3 - n4 [t4]

Optimized TAC (CSE eliminated):
  t1 = a + b
  t3 = t1 * t1
  t4 = t3 - c
```

Notice that `t2 = a + b` is eliminated because the DAG recognized that `a + b` was already computed as `t1`.

---

## 7. Linearization: From CFG Back to Linear Code

### 7.1 The Problem

After performing optimizations on a CFG (or its SSA form), we need to convert it back to a linear sequence of instructions for the code generator. This process is called **linearization** or **code layout**.

### 7.2 Block Ordering Strategies

The order in which basic blocks appear in the final output affects:
- **Fall-through efficiency**: Minimizing unconditional jumps
- **Cache behavior**: Keeping hot paths contiguous
- **Branch prediction**: Laying out likely paths sequentially

Common strategies:

1. **Reverse postorder (RPO)**: Perform a DFS on the CFG and visit blocks in reverse postorder. This ensures that a block's dominators appear before it.

2. **Trace-based layout**: Identify frequently executed paths (traces) and lay them out contiguously.

3. **Bottom-up layout**: Start from exit blocks and work backwards.

### 7.3 Reverse Postorder Algorithm

```python
def reverse_postorder(cfg, entry_id):
    """
    Compute reverse postorder of CFG blocks.
    Returns a list of block IDs in RPO.
    """
    visited = set()
    postorder = []

    def dfs(block_id):
        if block_id in visited:
            return
        visited.add(block_id)
        block = cfg.blocks[block_id]
        for succ in block.successors:
            dfs(succ)
        postorder.append(block_id)

    dfs(entry_id)
    return list(reversed(postorder))
```

### 7.4 Eliminating Redundant Jumps

After linearization, some jumps become unnecessary because the target block immediately follows the jumping block (fall-through). A simple pass removes these:

```python
def eliminate_redundant_jumps(block_order, cfg):
    """
    Remove goto instructions when the target is the next block
    in the linear layout.
    """
    for i, block_id in enumerate(block_order):
        block = cfg.blocks[block_id]
        if not block.instructions:
            continue

        last = block.instructions[-1]
        if last.is_unconditional_jump:
            # Check if the next block in layout is the jump target
            if i + 1 < len(block_order):
                next_block = cfg.blocks[block_order[i + 1]]
                if next_block.label == last.jump_target:
                    # Remove redundant goto
                    block.instructions.pop()
```

### 7.5 Deconstructing SSA

Before code generation, $\phi$-functions must be eliminated since they have no hardware equivalent. The standard approach replaces each $\phi$-function with **copy instructions** in predecessor blocks.

Given:
```
B1: ...
    goto B3

B2: ...
    goto B3

B3: x3 = phi(x1, x2)    // x1 from B1, x2 from B2
```

After deconstruction:
```
B1: ...
    x3 = x1              // copy added
    goto B3

B2: ...
    x3 = x2              // copy added
    goto B3

B3: // phi removed
```

This introduces copy instructions that can often be eliminated by later **copy propagation** or **register coalescing** passes.

**Complications**: When multiple $\phi$-functions reference each other's operands (the "swap problem"), the naive approach may produce incorrect results. The solution involves inserting copies in a careful order or introducing additional temporaries.

---

## 8. Summary

In this lesson, we explored the role of intermediate representations in compiler design:

1. **IRs decouple front ends from back ends**, reducing the $m \times n$ problem to $m + n$.

2. **Three-address code (TAC)** is a medium-level IR where each instruction has at most three operands. It can be stored as quadruples, triples, or indirect triples.

3. **Control flow graphs (CFGs)** organize TAC into basic blocks connected by control flow edges. Leaders are identified by jump targets and post-jump instructions.

4. **Static Single Assignment (SSA) form** assigns each variable exactly once. Phi functions handle merge points. Construction requires dominance frontiers, phi placement, and variable renaming.

5. **DAGs** provide compact expression representations that naturally expose common subexpressions.

6. **Linearization** converts a CFG back to sequential code, and SSA deconstruction replaces phi functions with copy instructions.

These representations form the backbone of modern compiler infrastructure, enabling the powerful optimization passes we will study in subsequent lessons.

---

## Exercises

### Exercise 1: TAC Generation

Translate the following code into three-address code:

```
x = 2 * a + b
y = a * a - b * b
if (x > y)
    z = x - y
else
    z = y - x
result = z * z
```

Show the complete TAC with temporaries, labels, and jumps.

### Exercise 2: Basic Block Identification

Given the following TAC, identify the leaders and partition the code into basic blocks:

```
(1)  i = 0
(2)  j = 0
(3)  t1 = i < 10
(4)  iffalse t1 goto L3
(5)  j = 0
(6)  t2 = j < 10
(7)  iffalse t2 goto L2
(8)  t3 = i * 10
(9)  t4 = t3 + j
(10) a[t4] = 0
(11) j = j + 1
(12) goto L1
(13) i = i + 1
(14) goto L0
(15) return
```

Where L0 is instruction 3, L1 is instruction 6, L2 is instruction 13, and L3 is instruction 15.

### Exercise 3: SSA Conversion

Convert the following code to SSA form. Clearly show all phi functions and their operands.

```
B0: x = 1
    y = 2
    goto B1

B1: if (x < 10) goto B2 else goto B3

B2: x = x + 1
    y = y * 2
    goto B1

B3: z = x + y
    return z
```

### Exercise 4: DAG Construction

Build the DAG for the following basic block:

```
a = b + c
d = b + c
e = a - d
f = a * e
g = f + e
```

Identify which subexpressions are shared, and write the optimized TAC.

### Exercise 5: Dominance Frontiers

For the following CFG, compute:
1. The dominator tree
2. The dominance frontier of each node

```
Entry → B1
B1 → B2, B3
B2 → B4
B3 → B4
B4 → B5, B6
B5 → B1
B6 → Exit
```

### Exercise 6: Implementation Challenge

Extend the Python TAC generator to handle:
1. **Function calls**: `param x`, `call f, n`, and `result = call f, n`
2. **Array accesses**: `x = a[i]` and `a[i] = x`

Write test cases that generate TAC for a function that computes the sum of an array.

---

[Previous: Semantic Analysis](./08_Semantic_Analysis.md) | [Next: Runtime Environments](./10_Runtime_Environments.md) | [Overview](./00_Overview.md)
