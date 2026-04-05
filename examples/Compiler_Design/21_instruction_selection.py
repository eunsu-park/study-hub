"""
21_instruction_selection.py - LLVM Backend: Instruction Selection

Demonstrates how a compiler backend translates IR instructions into
target machine instructions, the first major phase of code generation
in an LLVM-style backend.

Components:
  1. Target Machine Description
     A simplified description of a RISC-like target architecture with
     registers, instruction formats, and addressing modes.

  2. Tree-Pattern Matching
     Match IR expression trees against machine instruction patterns
     to select optimal instruction sequences.

  3. Instruction Selection via Tiling
     Cover the IR expression DAG with non-overlapping machine
     instruction tiles, minimizing total cost.

  4. Lowering
     Transform high-level IR operations (e.g., function calls, struct
     access) into sequences of machine-level operations.

  5. Instruction Scheduling
     Reorder instructions within a basic block to minimize pipeline
     stalls and maximize instruction-level parallelism.

Topics covered:
  - DAG-based instruction selection
  - Pattern matching for code generation
  - Instruction cost models
  - Machine instruction representation
  - Basic instruction scheduling (list scheduling)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Target Architecture Description
# ---------------------------------------------------------------------------

@dataclass
class Register:
    name: str
    index: int

    def __str__(self):
        return self.name


# RISC-like target with 8 general-purpose registers
TARGET_REGS = [Register(f"r{i}", i) for i in range(8)]


@dataclass
class MachineInstr:
    """A target machine instruction."""
    opcode: str
    dest: Optional[str] = None
    src1: Optional[str] = None
    src2: Optional[str] = None
    imm: Optional[int] = None
    label: Optional[str] = None
    latency: int = 1  # execution latency in cycles

    def __str__(self):
        if self.label and not self.opcode:
            return f"{self.label}:"
        parts = [f"  {self.opcode}"]
        if self.dest:
            parts.append(self.dest)
        if self.src1:
            parts.append(self.src1)
        if self.src2:
            parts.append(self.src2)
        if self.imm is not None:
            parts.append(f"#{self.imm}")
        if self.label:
            parts.append(self.label)
        return ", ".join(parts[:1]) + " " + ", ".join(parts[1:]) if len(parts) > 1 else parts[0]


# ---------------------------------------------------------------------------
# IR Nodes (expression DAG)
# ---------------------------------------------------------------------------

@dataclass
class IRNode:
    op: str
    ty: str = "i32"
    children: list[IRNode] = field(default_factory=list)
    value: Any = None  # for constants
    name: Optional[str] = None  # for variables

    def __repr__(self):
        if self.value is not None:
            return f"Const({self.value})"
        if self.name:
            return f"Var({self.name})"
        children_str = ", ".join(repr(c) for c in self.children)
        return f"{self.op}({children_str})"


# ---------------------------------------------------------------------------
# Instruction Patterns
# ---------------------------------------------------------------------------

@dataclass
class Pattern:
    """A pattern that matches an IR tree and produces machine instructions."""
    name: str
    ir_op: str
    cost: int
    # Pattern matching predicate
    match_imm: bool = False  # True if one operand should be immediate
    match_shift: bool = False  # True if pattern is shift+add combo

    def __str__(self):
        return f"Pattern({self.name}, cost={self.cost})"


# Available instruction patterns for our target
PATTERNS = [
    Pattern("ADD_RR", "add", cost=1),
    Pattern("ADD_RI", "add", cost=1, match_imm=True),
    Pattern("SUB_RR", "sub", cost=1),
    Pattern("SUB_RI", "sub", cost=1, match_imm=True),
    Pattern("MUL_RR", "mul", cost=3),
    Pattern("MUL_RI", "mul", cost=3, match_imm=True),
    Pattern("SHL_RI", "shl", cost=1, match_imm=True),
    Pattern("CMP_RR", "icmp", cost=1),
    Pattern("CMP_RI", "icmp", cost=1, match_imm=True),
    Pattern("LOAD",   "load", cost=3),
    Pattern("STORE",  "store", cost=3),
    Pattern("MOV_RI", "const", cost=1, match_imm=True),
    Pattern("MOV_RR", "copy", cost=1),
    Pattern("ADD_SHL", "add", cost=1, match_shift=True),  # combined shift+add
]


# ---------------------------------------------------------------------------
# Instruction Selector
# ---------------------------------------------------------------------------

class InstructionSelector:
    """
    Selects machine instructions by matching IR trees against patterns.
    Uses a greedy tiling approach (bottom-up, maximal munch).
    """

    def __init__(self):
        self.output: list[MachineInstr] = []
        self.temp_counter = 0
        self.log: list[str] = []

    def _next_temp(self) -> str:
        name = f"v{self.temp_counter}"
        self.temp_counter += 1
        return name

    def select(self, node: IRNode) -> str:
        """
        Select instructions for an IR node. Returns the virtual register
        holding the result.
        """
        # Leaf: constant
        if node.value is not None:
            dest = self._next_temp()
            self.output.append(MachineInstr("MOV", dest, imm=node.value))
            self.log.append(f"  Const({node.value}) -> MOV {dest}, #{node.value}")
            return dest

        # Leaf: variable
        if node.name is not None:
            return node.name

        # Try combined patterns first (maximal munch)

        # Pattern: add(x, shl(y, const)) -> ADD_SHL (single instruction)
        if node.op == "add" and len(node.children) == 2:
            r_child = node.children[1]
            if r_child.op == "shl" and len(r_child.children) == 2:
                if r_child.children[1].value is not None:
                    src1 = self.select(node.children[0])
                    src2 = self.select(r_child.children[0])
                    dest = self._next_temp()
                    shift = r_child.children[1].value
                    self.output.append(
                        MachineInstr("ADD_SHL", dest, src1, src2, imm=shift))
                    self.log.append(
                        f"  add(_, shl(_, {shift})) -> ADD_SHL {dest}, {src1}, "
                        f"{src2}, #{shift}")
                    return dest

        # Binary operations
        if node.op in ("add", "sub", "mul") and len(node.children) == 2:
            left, right = node.children
            src1 = self.select(left)

            # Try immediate form
            if right.value is not None:
                dest = self._next_temp()
                opcode = node.op.upper() + "I"
                self.output.append(
                    MachineInstr(opcode, dest, src1, imm=right.value))
                self.log.append(
                    f"  {node.op}(_, Const({right.value})) -> "
                    f"{opcode} {dest}, {src1}, #{right.value}")
                return dest

            # Register-register form
            src2 = self.select(right)
            dest = self._next_temp()
            opcode = node.op.upper()
            self.output.append(MachineInstr(opcode, dest, src1, src2))
            self.log.append(
                f"  {node.op}(_, _) -> {opcode} {dest}, {src1}, {src2}")
            return dest

        # Shift
        if node.op == "shl" and len(node.children) == 2:
            src1 = self.select(node.children[0])
            if node.children[1].value is not None:
                dest = self._next_temp()
                self.output.append(
                    MachineInstr("SHL", dest, src1, imm=node.children[1].value))
                return dest

        # Compare
        if node.op == "icmp" and len(node.children) == 2:
            src1 = self.select(node.children[0])
            src2 = self.select(node.children[1])
            dest = self._next_temp()
            self.output.append(MachineInstr("CMP", dest, src1, src2))
            return dest

        # Load
        if node.op == "load" and len(node.children) == 1:
            addr = self.select(node.children[0])
            dest = self._next_temp()
            self.output.append(
                MachineInstr("LDR", dest, addr, latency=3))
            return dest

        # Store
        if node.op == "store" and len(node.children) == 2:
            value = self.select(node.children[0])
            addr = self.select(node.children[1])
            self.output.append(
                MachineInstr("STR", src1=value, src2=addr, latency=3))
            return value

        # Fallback
        dest = self._next_temp()
        self.output.append(MachineInstr(f"??? ({node.op})", dest))
        return dest


# ---------------------------------------------------------------------------
# Instruction Scheduler (list scheduling)
# ---------------------------------------------------------------------------

@dataclass
class SchedNode:
    index: int
    instr: MachineInstr
    preds: list[int] = field(default_factory=list)  # dependencies
    earliest: int = 0

    def __repr__(self):
        return f"SN({self.index}, {self.instr.opcode})"


def schedule_instructions(instrs: list[MachineInstr]) -> list[MachineInstr]:
    """
    List scheduling: order instructions to minimize pipeline stalls.
    Builds a dependency graph and schedules instructions as early as possible.
    """
    nodes = [SchedNode(i, instr) for i, instr in enumerate(instrs)]

    # Build dependency graph: instruction j depends on instruction i
    # if j uses the result of i
    defs_map: dict[str, int] = {}  # register -> defining instruction index
    for i, instr in enumerate(instrs):
        # Check dependencies on src1, src2
        for src in [instr.src1, instr.src2]:
            if src and src in defs_map:
                nodes[i].preds.append(defs_map[src])
        # Record definition
        if instr.dest:
            defs_map[instr.dest] = i

    # Compute earliest start times
    for node in nodes:
        for pred_idx in node.preds:
            pred = nodes[pred_idx]
            earliest = pred.earliest + pred.instr.latency
            node.earliest = max(node.earliest, earliest)

    # Sort by earliest time (stable sort preserves original order for ties)
    scheduled = sorted(nodes, key=lambda n: (n.earliest, n.index))
    return [n.instr for n in scheduled]


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Instruction Selection and Scheduling Demo")
    print("=" * 60)

    # Example 1: Simple expression: (a + 3) * (b - 1)
    print("\n--- Example 1: (a + 3) * (b - 1) ---")
    expr1 = IRNode("mul", children=[
        IRNode("add", children=[
            IRNode("", name="a"),
            IRNode("", value=3)
        ]),
        IRNode("sub", children=[
            IRNode("", name="b"),
            IRNode("", value=1)
        ])
    ])
    print(f"  IR: {expr1}")

    sel1 = InstructionSelector()
    result1 = sel1.select(expr1)
    print(f"\n  Selected instructions:")
    for instr in sel1.output:
        print(f"    {instr}")
    print(f"  Selection log:")
    for entry in sel1.log:
        print(entry)

    # Example 2: Array access: a[i] = a[i] + x  (with shift for indexing)
    print("\n--- Example 2: Combined shift+add pattern ---")
    # addr = base + (index << 2)
    expr2 = IRNode("add", children=[
        IRNode("", name="base"),
        IRNode("shl", children=[
            IRNode("", name="index"),
            IRNode("", value=2)
        ])
    ])
    print(f"  IR: {expr2}")

    sel2 = InstructionSelector()
    result2 = sel2.select(expr2)
    print(f"\n  Selected instructions:")
    for instr in sel2.output:
        print(f"    {instr}")
    print(f"  Selection log:")
    for entry in sel2.log:
        print(entry)

    # Example 3: Instruction scheduling
    print("\n--- Instruction Scheduling ---")
    # Simulated instruction stream with varying latencies
    instrs = [
        MachineInstr("LDR", "v0", "r0", latency=3),    # load (3 cycles)
        MachineInstr("LDR", "v1", "r1", latency=3),    # load (3 cycles)
        MachineInstr("ADD", "v2", "v0", "v1"),          # depends on both loads
        MachineInstr("MULI", "v3", "v2", imm=2),       # depends on add
        MachineInstr("ADDI", "v4", "r2", imm=1),       # independent
        MachineInstr("SUB", "v5", "v4", "v3"),          # depends on v4, v3
    ]

    print(f"  Original order:")
    for i, instr in enumerate(instrs):
        print(f"    [{i}] {instr} (latency={instr.latency})")

    scheduled = schedule_instructions(instrs)
    print(f"\n  Scheduled order (pipeline-optimized):")
    for i, instr in enumerate(scheduled):
        print(f"    [{i}] {instr} (latency={instr.latency})")

    print(f"\n--- Summary ---")
    print("""
  Instruction Selection:
    - Maximal munch: match largest IR tree patterns first
    - Immediate operands: use register-immediate forms when possible
    - Combined patterns: fuse operations (e.g., shift+add)

  Instruction Scheduling:
    - List scheduling: build dependency DAG, schedule by earliest time
    - Hide latencies: interleave independent instructions
    - Critical path: prioritize instructions on the longest dependency chain
    """)


if __name__ == "__main__":
    main()
