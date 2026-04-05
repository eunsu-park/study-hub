"""
18_ssa_form.py - Static Single Assignment (SSA) Form

Demonstrates the construction and use of SSA form, the dominant
intermediate representation in modern optimizing compilers.

In SSA form, every variable is assigned exactly once. When control
flow merges, phi functions select the correct version of a variable.

Components:
  1. SSA Construction
     Convert normal IR to SSA by renaming variables and inserting
     phi functions at dominance frontiers.

  2. Phi Function Insertion
     Place phi nodes at iterated dominance frontiers where multiple
     definitions of a variable reach the same point.

  3. Variable Renaming
     Walk the dominator tree, pushing new names onto a stack and
     renaming uses to the most recent definition.

  4. SSA-Based Optimization
     Demonstrate how SSA simplifies constant propagation and dead
     code elimination (single def makes reaching-definitions trivial).

  5. SSA Destruction
     Convert out of SSA by replacing phi functions with copies,
     preparing for register allocation.

Topics covered:
  - SSA definition and properties
  - Dominance frontiers and phi placement
  - Renaming algorithm (stack-based)
  - SSA advantages for optimization
  - Phi elimination / SSA destruction
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Basic Block and CFG
# ---------------------------------------------------------------------------

@dataclass
class SSAInstr:
    dest: Optional[str] = None
    op: Optional[str] = None
    left: Any = None
    right: Any = None
    is_phi: bool = False
    phi_args: dict[str, str] = field(default_factory=dict)  # pred_label -> var
    is_branch: bool = False
    is_return: bool = False

    def __str__(self):
        if self.is_phi:
            args = ", ".join(f"{v} from {p}" for p, v in self.phi_args.items())
            return f"    {self.dest} = phi({args})"
        if self.is_return:
            return f"    return {self.left}"
        if self.op:
            return f"    {self.dest} = {self.left} {self.op} {self.right}"
        if self.dest:
            return f"    {self.dest} = {self.left}"
        return f"    nop"


@dataclass
class Block:
    label: str
    instrs: list[SSAInstr] = field(default_factory=list)
    succs: list[str] = field(default_factory=list)
    preds: list[str] = field(default_factory=list)

    def __str__(self):
        lines = [f"{self.label}:"]
        for i in self.instrs:
            lines.append(str(i))
        if self.succs:
            lines.append(f"    -> {', '.join(self.succs)}")
        return "\n".join(lines)


class CFG:
    def __init__(self):
        self.blocks: dict[str, Block] = {}
        self.entry: str = ""

    def add_block(self, label: str, instrs: list[SSAInstr],
                  succs: list[str]) -> None:
        self.blocks[label] = Block(label, instrs, succs)
        if not self.entry:
            self.entry = label

    def build_preds(self) -> None:
        for b in self.blocks.values():
            b.preds = []
        for b in self.blocks.values():
            for s in b.succs:
                if s in self.blocks:
                    self.blocks[s].preds.append(b.label)


# ---------------------------------------------------------------------------
# Dominator and Dominance Frontier computation
# ---------------------------------------------------------------------------

def compute_dominators(cfg: CFG) -> dict[str, set[str]]:
    all_labels = set(cfg.blocks.keys())
    dom = {cfg.entry: {cfg.entry}}
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
            new_dom = set.intersection(*(dom[p] for p in preds)) | {label}
            if new_dom != dom[label]:
                dom[label] = new_dom
                changed = True
    return dom


def immediate_dominator(dom: dict[str, set[str]],
                        node: str) -> Optional[str]:
    """Find the immediate dominator of a node."""
    dominators = dom[node] - {node}
    for candidate in dominators:
        # idom is the dominator that is dominated by all other dominators
        if all(candidate in dom[d] or candidate == d
               for d in dominators):
            return candidate
    return None


def compute_dom_frontiers(cfg: CFG,
                          dom: dict[str, set[str]]) -> dict[str, set[str]]:
    """
    Dominance frontier of a node n:
    DF(n) = {y | exists pred p of y where n dominates p but n does not
             strictly dominate y}
    """
    df: dict[str, set[str]] = {label: set() for label in cfg.blocks}
    for y in cfg.blocks:
        if len(cfg.blocks[y].preds) < 2:
            continue
        for p in cfg.blocks[y].preds:
            runner = p
            while runner and runner != immediate_dominator(dom, y):
                df[runner].add(y)
                idom = immediate_dominator(dom, runner)
                runner = idom
    return df


# ---------------------------------------------------------------------------
# Phi Function Insertion
# ---------------------------------------------------------------------------

def find_definitions(cfg: CFG) -> dict[str, set[str]]:
    """Find which blocks define each variable."""
    defs: dict[str, set[str]] = {}
    for label, block in cfg.blocks.items():
        for instr in block.instrs:
            if instr.dest and not instr.is_phi:
                defs.setdefault(instr.dest, set()).add(label)
    return defs


def insert_phi_functions(cfg: CFG,
                         df: dict[str, set[str]]) -> dict[str, set[str]]:
    """
    Insert phi functions at iterated dominance frontiers.
    Returns map of variable -> set of blocks where phis were inserted.
    """
    var_defs = find_definitions(cfg)
    phi_locations: dict[str, set[str]] = {v: set() for v in var_defs}

    for var in var_defs:
        worklist = list(var_defs[var])
        ever_on_wl = set(worklist)

        while worklist:
            block_label = worklist.pop()
            for frontier_label in df.get(block_label, set()):
                if frontier_label not in phi_locations[var]:
                    phi_locations[var].add(frontier_label)
                    # Insert phi instruction at beginning of block
                    preds = cfg.blocks[frontier_label].preds
                    phi = SSAInstr(
                        dest=var, is_phi=True,
                        phi_args={p: var for p in preds}
                    )
                    cfg.blocks[frontier_label].instrs.insert(0, phi)
                    if frontier_label not in ever_on_wl:
                        worklist.append(frontier_label)
                        ever_on_wl.add(frontier_label)

    return phi_locations


# ---------------------------------------------------------------------------
# SSA Renaming
# ---------------------------------------------------------------------------

def rename_variables(cfg: CFG, dom: dict[str, set[str]]) -> None:
    """
    Rename all variables to SSA form using a stack-based approach.
    Each variable gets a subscript: x -> x_0, x_1, x_2, etc.
    """
    counters: dict[str, int] = {}
    stacks: dict[str, list[str]] = {}

    def new_name(var: str) -> str:
        n = counters.get(var, 0)
        counters[var] = n + 1
        name = f"{var}_{n}"
        stacks.setdefault(var, []).append(name)
        return name

    def current_name(var: str) -> str:
        if var in stacks and stacks[var]:
            return stacks[var][-1]
        return var

    def is_var(v: Any) -> bool:
        return isinstance(v, str) and v.isidentifier() and not v.isdigit()

    # Build dominator tree
    idom: dict[str, Optional[str]] = {}
    for node in cfg.blocks:
        idom[node] = immediate_dominator(dom, node)

    children: dict[str, list[str]] = {n: [] for n in cfg.blocks}
    for node, parent in idom.items():
        if parent:
            children[parent].append(node)

    def rename_block(label: str) -> None:
        block = cfg.blocks[label]
        saved_stacks = {v: list(s) for v, s in stacks.items()}

        for instr in block.instrs:
            # Rename uses (except phi args, handled separately)
            if not instr.is_phi:
                if is_var(instr.left):
                    instr.left = current_name(instr.left)
                if is_var(instr.right):
                    instr.right = current_name(instr.right)

            # Rename definition
            if instr.dest:
                base = instr.dest.split('_')[0] if '_' in instr.dest else instr.dest
                instr.dest = new_name(base)

        # Update phi args in successors
        for succ_label in block.succs:
            succ = cfg.blocks.get(succ_label)
            if not succ:
                continue
            for instr in succ.instrs:
                if instr.is_phi and label in instr.phi_args:
                    base = instr.phi_args[label]
                    instr.phi_args[label] = current_name(base)

        # Recurse into dominated blocks
        for child in children.get(label, []):
            rename_block(child)

        # Restore stacks
        for v in stacks:
            stacks[v] = saved_stacks.get(v, [])

    rename_block(cfg.entry)


# ---------------------------------------------------------------------------
# SSA Destruction (phi elimination)
# ---------------------------------------------------------------------------

def eliminate_phis(cfg: CFG) -> list[tuple[str, str, str]]:
    """
    Replace phi functions with copy instructions in predecessor blocks.
    Returns list of (pred_block, dest, src) copies inserted.
    """
    copies = []
    for block in cfg.blocks.values():
        non_phi = []
        for instr in block.instrs:
            if instr.is_phi:
                for pred, src in instr.phi_args.items():
                    copies.append((pred, instr.dest, src))
                    # Insert copy at end of predecessor
                    pred_block = cfg.blocks.get(pred)
                    if pred_block:
                        copy_instr = SSAInstr(dest=instr.dest, left=src)
                        pred_block.instrs.append(copy_instr)
            else:
                non_phi.append(instr)
        block.instrs = non_phi

    return copies


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def build_sample_cfg() -> CFG:
    """
    Build a CFG for:
        entry: x = 1
        loop:  x = x + 1; if x < 10 goto loop else exit
        exit:  return x
    """
    cfg = CFG()
    cfg.add_block("entry", [
        SSAInstr(dest="x", left=1),
    ], ["loop"])
    cfg.add_block("loop", [
        SSAInstr(dest="x", op="+", left="x", right=1),
        SSAInstr(dest="cond", op="<", left="x", right=10),
    ], ["loop", "exit"])
    cfg.add_block("exit", [
        SSAInstr(left="x", is_return=True),
    ], [])
    cfg.build_preds()
    return cfg


def print_cfg(title: str, cfg: CFG) -> None:
    print(f"\n--- {title} ---")
    for label in sorted(cfg.blocks.keys()):
        print(cfg.blocks[label])


def main():
    print("=" * 60)
    print("SSA Form Construction and Destruction Demo")
    print("=" * 60)

    cfg = build_sample_cfg()
    print_cfg("Original CFG", cfg)

    # Step 1: Compute dominators
    dom = compute_dominators(cfg)
    print(f"\n--- Dominators ---")
    for label in sorted(dom):
        print(f"  Dom({label}) = {sorted(dom[label])}")

    # Step 2: Compute dominance frontiers
    df = compute_dom_frontiers(cfg, dom)
    print(f"\n--- Dominance Frontiers ---")
    for label in sorted(df):
        print(f"  DF({label}) = {sorted(df[label])}")

    # Step 3: Insert phi functions
    phi_locs = insert_phi_functions(cfg, df)
    print(f"\n--- Phi Function Locations ---")
    for var, locs in phi_locs.items():
        if locs:
            print(f"  {var}: {sorted(locs)}")

    print_cfg("After Phi Insertion", cfg)

    # Step 4: Rename variables
    rename_variables(cfg, dom)
    print_cfg("After SSA Renaming", cfg)

    # Step 5: SSA Destruction
    copies = eliminate_phis(cfg)
    print(f"\n--- Phi Elimination Copies ---")
    for pred, dest, src in copies:
        print(f"  In {pred}: {dest} = {src}")
    print_cfg("After SSA Destruction", cfg)

    print(f"\n--- SSA Properties ---")
    print("""
  1. Every variable assigned exactly once
  2. Phi functions merge definitions at join points
  3. Simplifies optimization: single reaching definition per use
  4. Enables sparse conditional constant propagation (SCCP)
  5. Global value numbering becomes straightforward
  6. SSA destruction: replace phis with copies before register allocation
    """)


if __name__ == "__main__":
    main()
