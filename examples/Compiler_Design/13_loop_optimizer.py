"""
13_loop_optimizer.py - Loop Optimization Techniques

Demonstrates key loop optimization transformations that compilers
apply to improve performance of loop-heavy code:

  1. Natural Loop Detection
     Identify loops in a control-flow graph via dominance and back edges.

  2. Loop-Invariant Code Motion (LICM)
     Hoist computations whose operands do not change inside the loop
     into the preheader, eliminating redundant recalculations.

  3. Induction Variable Detection and Strength Reduction
     Recognize variables that change by a constant amount each iteration
     and replace expensive operations (e.g., multiply) with cheaper ones
     (e.g., addition).

  4. Loop Unrolling
     Replicate the loop body multiple times to reduce branch overhead
     and expose instruction-level parallelism.

Topics covered:
  - Dominator tree construction (simple iterative algorithm)
  - Back edge detection and natural loop identification
  - Reaching-definition-style analysis for invariant detection
  - Strength reduction of induction-variable-based expressions
  - Unroll factor selection and remainder handling
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Control Flow Graph representation
# ---------------------------------------------------------------------------

@dataclass
class BasicBlock:
    """A basic block in the CFG."""
    label: str
    instrs: list[str] = field(default_factory=list)
    succs: list[str] = field(default_factory=list)
    preds: list[str] = field(default_factory=list)

    def __repr__(self) -> str:
        return f"BB({self.label})"


class CFG:
    """Control Flow Graph built from basic blocks."""

    def __init__(self):
        self.blocks: dict[str, BasicBlock] = {}
        self.entry: str = ""

    def add_block(self, label: str, instrs: list[str],
                  succs: list[str]) -> None:
        bb = BasicBlock(label, instrs, succs)
        self.blocks[label] = bb
        if not self.entry:
            self.entry = label

    def build_preds(self) -> None:
        for bb in self.blocks.values():
            bb.preds = []
        for bb in self.blocks.values():
            for s in bb.succs:
                if s in self.blocks:
                    self.blocks[s].preds.append(bb.label)


# ---------------------------------------------------------------------------
# Dominator computation (iterative dataflow)
# ---------------------------------------------------------------------------

def compute_dominators(cfg: CFG) -> dict[str, set[str]]:
    """
    Compute dominator sets for each block.
    Dom(entry) = {entry}
    Dom(n)     = {n} ∪ ∩{Dom(p) | p ∈ preds(n)}
    """
    all_labels = set(cfg.blocks.keys())
    dom: dict[str, set[str]] = {}
    dom[cfg.entry] = {cfg.entry}
    for label in all_labels:
        if label != cfg.entry:
            dom[label] = set(all_labels)

    changed = True
    while changed:
        changed = False
        for label in all_labels:
            if label == cfg.entry:
                continue
            preds = cfg.blocks[label].preds
            if not preds:
                continue
            new_dom = set.intersection(*(dom[p] for p in preds))
            new_dom = new_dom | {label}
            if new_dom != dom[label]:
                dom[label] = new_dom
                changed = True

    return dom


def find_back_edges(cfg: CFG,
                    dom: dict[str, set[str]]) -> list[tuple[str, str]]:
    """
    A back edge is an edge n -> h where h dominates n.
    """
    back_edges = []
    for bb in cfg.blocks.values():
        for succ in bb.succs:
            if succ in dom.get(bb.label, set()):
                back_edges.append((bb.label, succ))
    return back_edges


def find_natural_loop(cfg: CFG, back_edge: tuple[str, str]) -> set[str]:
    """
    Compute the natural loop for a given back edge (n -> h).
    The loop consists of h plus all nodes that can reach n without
    going through h.
    """
    tail, header = back_edge
    loop = {header}
    worklist = []

    if tail != header:
        loop.add(tail)
        worklist.append(tail)

    while worklist:
        node = worklist.pop()
        for pred in cfg.blocks[node].preds:
            if pred not in loop:
                loop.add(pred)
                worklist.append(pred)

    return loop


# ---------------------------------------------------------------------------
# Loop-Invariant Code Motion (LICM)
# ---------------------------------------------------------------------------

def parse_instr(instr: str) -> tuple[Optional[str], list[str]]:
    """
    Parse a simple instruction 'dest = op1 OP op2' or 'dest = op1'.
    Returns (dest, [operands]).
    """
    if '=' not in instr:
        return None, []
    parts = instr.split('=', 1)
    dest = parts[0].strip()
    rhs = parts[1].strip()
    tokens = rhs.split()
    operands = [t for t in tokens if t.isidentifier()]
    return dest, operands


def find_loop_invariants(cfg: CFG,
                         loop_blocks: set[str]) -> list[tuple[str, str]]:
    """
    An instruction is loop-invariant if all its operands are either:
      - Defined outside the loop, or
      - Themselves loop-invariant.
    Returns list of (block_label, instruction) that are invariant.
    """
    # Collect all definitions inside the loop
    loop_defs: set[str] = set()
    for label in loop_blocks:
        for instr in cfg.blocks[label].instrs:
            dest, _ = parse_instr(instr)
            if dest:
                loop_defs.add(dest)

    invariants: list[tuple[str, str]] = []
    invariant_vars: set[str] = set()
    changed = True

    while changed:
        changed = False
        for label in loop_blocks:
            for instr in cfg.blocks[label].instrs:
                if (label, instr) in [(l, i) for l, i in invariants]:
                    continue
                dest, operands = parse_instr(instr)
                if dest is None:
                    continue
                # All operands must be either outside loop or invariant
                is_inv = True
                for op in operands:
                    if op in loop_defs and op not in invariant_vars:
                        is_inv = False
                        break
                if is_inv:
                    invariants.append((label, instr))
                    invariant_vars.add(dest)
                    changed = True

    return invariants


# ---------------------------------------------------------------------------
# Induction Variable Detection and Strength Reduction
# ---------------------------------------------------------------------------

@dataclass
class InductionVar:
    name: str
    init: int
    step: int

    def __repr__(self):
        return f"IndVar({self.name}, init={self.init}, step={self.step})"


def detect_induction_variables(cfg: CFG,
                               loop_blocks: set[str],
                               header: str) -> list[InductionVar]:
    """
    Detect basic induction variables: variables of the form
      i = i + c  (or i = i - c)
    inside the loop body.
    """
    induction_vars = []
    for label in loop_blocks:
        for instr in cfg.blocks[label].instrs:
            dest, _ = parse_instr(instr)
            if dest is None:
                continue
            parts = instr.split('=', 1)[1].strip().split()
            # Pattern: dest = dest + CONST  or  dest = dest - CONST
            if len(parts) == 3 and parts[0] == dest and parts[1] in ('+', '-'):
                try:
                    step = int(parts[2])
                    if parts[1] == '-':
                        step = -step
                    induction_vars.append(InductionVar(dest, 0, step))
                except ValueError:
                    pass
    return induction_vars


def strength_reduce(instrs: list[str],
                    ind_vars: list[InductionVar]) -> list[str]:
    """
    Replace expressions like 'x = ind_var * const' with additive updates.
    For each 'y = iv * k', introduce 'y_sr' that starts at iv.init * k
    and increments by iv.step * k each iteration.
    """
    iv_names = {iv.name: iv for iv in ind_vars}
    reduced: list[str] = []
    replacements: dict[str, str] = {}

    for instr in instrs:
        dest, _ = parse_instr(instr)
        if dest is None:
            reduced.append(instr)
            continue

        rhs = instr.split('=', 1)[1].strip().split()
        # Pattern: dest = iv * const
        if len(rhs) == 3 and rhs[1] == '*':
            var, const_str = rhs[0], rhs[2]
            if var in iv_names:
                try:
                    k = int(const_str)
                    iv = iv_names[var]
                    sr_name = f"{dest}_sr"
                    inc = iv.step * k
                    reduced.append(f"{sr_name} = {sr_name} + {inc}")
                    reduced.append(f"{dest} = {sr_name}")
                    replacements[dest] = sr_name
                    print(f"  Strength reduction: '{instr}' -> "
                          f"'{sr_name} = {sr_name} + {inc}'")
                    continue
                except ValueError:
                    pass
        reduced.append(instr)

    return reduced


# ---------------------------------------------------------------------------
# Loop Unrolling
# ---------------------------------------------------------------------------

def unroll_loop(body: list[str], factor: int,
                trip_count: Optional[int] = None) -> list[str]:
    """
    Unroll a loop body by the given factor.
    If the trip count is known and divisible by factor, no remainder loop
    is needed. Otherwise, the original body handles the remainder.
    """
    unrolled: list[str] = []
    needs_remainder = trip_count is None or trip_count % factor != 0

    main_iters = 1  # times we replicate in the unrolled body
    for _ in range(factor):
        for instr in body:
            unrolled.append(instr)

    result = [f"// Unrolled {factor}x"]
    result.extend(unrolled)
    if needs_remainder:
        result.append(f"// Remainder loop (original body)")
        result.extend(body)

    return result


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def build_sample_cfg() -> CFG:
    """
    Build a CFG for:
        entry:  x = 10; y = 0; i = 0
        header: if i < n goto body else exit
        body:   t = x * 2;  y = y + t;  a = i * 4;  i = i + 1
                goto header
        exit:   return y
    """
    cfg = CFG()
    cfg.add_block("entry",
                  ["x = 10", "y = 0", "i = 0"],
                  ["header"])
    cfg.add_block("header",
                  ["if i < n goto body else exit"],
                  ["body", "exit"])
    cfg.add_block("body",
                  ["t = x * 2", "y = y + t", "a = i * 4", "i = i + 1"],
                  ["header"])
    cfg.add_block("exit",
                  ["return y"],
                  [])
    cfg.build_preds()
    return cfg


def main():
    print("=" * 60)
    print("Loop Optimization Demo")
    print("=" * 60)

    cfg = build_sample_cfg()

    # 1. Dominator analysis
    dom = compute_dominators(cfg)
    print("\n--- Dominator Sets ---")
    for label in sorted(dom):
        print(f"  Dom({label}) = {sorted(dom[label])}")

    # 2. Back edge detection
    back_edges = find_back_edges(cfg, dom)
    print(f"\n--- Back Edges ---")
    for tail, header in back_edges:
        print(f"  {tail} -> {header}")

    # 3. Natural loop identification
    for be in back_edges:
        loop = find_natural_loop(cfg, be)
        print(f"\n--- Natural Loop (back edge {be[0]} -> {be[1]}) ---")
        print(f"  Blocks: {sorted(loop)}")

        # 4. Loop-invariant code motion
        invariants = find_loop_invariants(cfg, loop)
        print(f"\n--- Loop-Invariant Instructions ---")
        for block, instr in invariants:
            print(f"  [{block}] {instr}  -> hoist to preheader")

        # 5. Induction variable detection
        ivs = detect_induction_variables(cfg, loop, be[1])
        print(f"\n--- Induction Variables ---")
        for iv in ivs:
            print(f"  {iv}")

        # 6. Strength reduction
        body_instrs = cfg.blocks["body"].instrs[:]
        print(f"\n--- Strength Reduction ---")
        reduced = strength_reduce(body_instrs, ivs)
        print(f"  Original body: {cfg.blocks['body'].instrs}")
        print(f"  Reduced body:  {reduced}")

    # 7. Loop unrolling demo
    print(f"\n--- Loop Unrolling (factor=4) ---")
    simple_body = ["sum = sum + a[i]", "i = i + 1"]
    unrolled = unroll_loop(simple_body, factor=4, trip_count=100)
    for line in unrolled:
        print(f"  {line}")
    print(f"\n  Body size: {len(simple_body)} -> {len(unrolled)} instructions")

    # Summary
    print(f"\n--- Optimization Summary ---")
    print(f"  Loop detection:       Natural loops via dominators + back edges")
    print(f"  LICM:                 Hoist invariant computations to preheader")
    print(f"  Strength reduction:   Replace multiply with add for induction vars")
    print(f"  Loop unrolling:       Replicate body to reduce branch overhead")


if __name__ == "__main__":
    main()
