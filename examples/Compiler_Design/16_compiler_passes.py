"""
16_compiler_passes.py - Modern Compiler Infrastructure and Pass Management

Demonstrates the pass-based architecture used in modern compilers
like LLVM, GCC, and Cranelift. Instead of a monolithic pipeline,
the compiler is organized as a sequence of independently testable
analysis and transformation passes over an intermediate representation.

Components:
  1. Pass Manager
     Orchestrates the execution order of passes, handles dependencies,
     and provides an interface for registering and scheduling passes.

  2. Analysis Passes (read-only)
     - Dominance analysis
     - Liveness analysis
     - Use-def chain construction

  3. Transformation Passes (modify IR)
     - Dead code elimination
     - Constant propagation
     - Instruction simplification

  4. Pass Pipeline Builder
     Declarative API for constructing optimization pipelines
     with dependency resolution.

Topics covered:
  - Pass manager design patterns
  - Analysis vs transformation passes
  - Pass dependencies and invalidation
  - Declarative pipeline construction
  - Modern compiler infrastructure concepts (LLVM-style)
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional
from collections import OrderedDict


# ---------------------------------------------------------------------------
# Simple IR representation
# ---------------------------------------------------------------------------

@dataclass
class IRInstr:
    """A simple IR instruction: dest = left op right, or dest = src."""
    dest: Optional[str] = None
    op: Optional[str] = None
    left: Any = None
    right: Any = None
    label: Optional[str] = None
    is_branch: bool = False
    target: Optional[str] = None
    is_return: bool = False

    def __str__(self):
        if self.label:
            return f"{self.label}:"
        if self.is_return:
            return f"    return {self.left}"
        if self.is_branch:
            return f"    br {self.left} -> {self.target}"
        if self.op:
            return f"    {self.dest} = {self.left} {self.op} {self.right}"
        return f"    {self.dest} = {self.left}"

    def uses(self) -> list[str]:
        """Return list of variables used by this instruction."""
        result = []
        if isinstance(self.left, str) and not self.left.isdigit():
            result.append(self.left)
        if isinstance(self.right, str) and not self.right.isdigit():
            result.append(self.right)
        return result

    def defs(self) -> Optional[str]:
        """Return the variable defined by this instruction, if any."""
        return self.dest


@dataclass
class IRFunction:
    """A function in the IR, containing a list of instructions."""
    name: str
    instrs: list[IRInstr] = field(default_factory=list)

    def __str__(self):
        lines = [f"func {self.name}:"]
        for i in self.instrs:
            lines.append(str(i))
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Pass base classes
# ---------------------------------------------------------------------------

class AnalysisResult:
    """Base class for analysis results."""
    pass


class Pass(ABC):
    """Base class for all compiler passes."""
    name: str = "unnamed"
    dependencies: list[str] = []

    @abstractmethod
    def run(self, func: IRFunction, manager: PassManager) -> Any:
        ...


class AnalysisPass(Pass):
    """A read-only pass that computes information about the IR."""
    pass


class TransformPass(Pass):
    """A pass that modifies the IR. May invalidate analyses."""
    invalidates: list[str] = []


# ---------------------------------------------------------------------------
# Pass Manager
# ---------------------------------------------------------------------------

class PassManager:
    """
    Manages registration, scheduling, and execution of compiler passes.
    Caches analysis results and invalidates them when transforms run.
    """

    def __init__(self):
        self.registered: OrderedDict[str, Pass] = OrderedDict()
        self.pipeline: list[str] = []
        self.analysis_cache: dict[str, AnalysisResult] = {}
        self.log: list[str] = []

    def register(self, pass_obj: Pass) -> None:
        self.registered[pass_obj.name] = pass_obj

    def schedule(self, *pass_names: str) -> None:
        """Add passes to the execution pipeline."""
        for name in pass_names:
            if name not in self.registered:
                raise ValueError(f"Unknown pass: {name}")
            self.pipeline.append(name)

    def get_analysis(self, name: str, func: IRFunction) -> AnalysisResult:
        """Get a cached analysis result, computing it if needed."""
        if name not in self.analysis_cache:
            if name not in self.registered:
                raise ValueError(f"Unknown analysis: {name}")
            p = self.registered[name]
            if not isinstance(p, AnalysisPass):
                raise ValueError(f"{name} is not an analysis pass")
            result = p.run(func, self)
            self.analysis_cache[name] = result
        return self.analysis_cache[name]

    def invalidate(self, names: list[str]) -> None:
        for name in names:
            self.analysis_cache.pop(name, None)

    def run(self, func: IRFunction) -> IRFunction:
        """Execute the scheduled pipeline on the function."""
        self.log.append(f"Running pipeline on '{func.name}' "
                        f"({len(self.pipeline)} passes)")

        for pass_name in self.pipeline:
            p = self.registered[pass_name]

            # Ensure dependencies are satisfied
            for dep in p.dependencies:
                self.get_analysis(dep, func)

            result = p.run(func, self)

            if isinstance(p, TransformPass):
                if isinstance(result, IRFunction):
                    func = result
                # Invalidate affected analyses
                if hasattr(p, 'invalidates'):
                    self.invalidate(p.invalidates)
                self.log.append(f"  [transform] {pass_name}")
            else:
                self.log.append(f"  [analysis]  {pass_name}")

        return func


# ---------------------------------------------------------------------------
# Analysis Passes
# ---------------------------------------------------------------------------

@dataclass
class UseDefResult(AnalysisResult):
    """Maps each variable to its defining instruction index."""
    defs: dict[str, int] = field(default_factory=dict)
    uses: dict[str, list[int]] = field(default_factory=dict)


class UseDefAnalysis(AnalysisPass):
    name = "use-def"
    dependencies = []

    def run(self, func: IRFunction, manager: PassManager) -> UseDefResult:
        result = UseDefResult()
        for idx, instr in enumerate(func.instrs):
            d = instr.defs()
            if d:
                result.defs[d] = idx
            for u in instr.uses():
                result.uses.setdefault(u, []).append(idx)
        return result


@dataclass
class LivenessResult(AnalysisResult):
    """Live variables at each instruction index."""
    live_in: dict[int, set[str]] = field(default_factory=dict)
    live_out: dict[int, set[str]] = field(default_factory=dict)


class LivenessAnalysis(AnalysisPass):
    name = "liveness"
    dependencies = []

    def run(self, func: IRFunction, manager: PassManager) -> LivenessResult:
        n = len(func.instrs)
        result = LivenessResult()

        # Initialize
        for i in range(n):
            result.live_in[i] = set()
            result.live_out[i] = set()

        # Backward dataflow iteration
        changed = True
        while changed:
            changed = False
            for i in range(n - 1, -1, -1):
                instr = func.instrs[i]
                old_in = set(result.live_in[i])

                # live_out[i] = live_in[i+1] (simplified: single basic block)
                if i + 1 < n:
                    result.live_out[i] = set(result.live_in[i + 1])

                # live_in[i] = uses[i] | (live_out[i] - defs[i])
                new_in = set(instr.uses())
                d = instr.defs()
                out_minus_def = set(result.live_out[i])
                if d:
                    out_minus_def.discard(d)
                new_in |= out_minus_def
                result.live_in[i] = new_in

                if new_in != old_in:
                    changed = True

        return result


# ---------------------------------------------------------------------------
# Transformation Passes
# ---------------------------------------------------------------------------

class ConstantPropagation(TransformPass):
    name = "const-prop"
    dependencies = ["use-def"]
    invalidates = ["use-def", "liveness"]

    def run(self, func: IRFunction, manager: PassManager) -> IRFunction:
        const_map: dict[str, Any] = {}
        changes = 0
        new_instrs = []

        for instr in func.instrs:
            new_instr = IRInstr(
                dest=instr.dest, op=instr.op,
                left=instr.left, right=instr.right,
                label=instr.label, is_branch=instr.is_branch,
                target=instr.target, is_return=instr.is_return
            )

            # Substitute known constants in operands
            if isinstance(new_instr.left, str) and new_instr.left in const_map:
                new_instr.left = const_map[new_instr.left]
                changes += 1
            if isinstance(new_instr.right, str) and new_instr.right in const_map:
                new_instr.right = const_map[new_instr.right]
                changes += 1

            # Try to fold
            if new_instr.op and _is_num(new_instr.left) and _is_num(new_instr.right):
                folded = _fold(new_instr.op, new_instr.left, new_instr.right)
                if folded is not None:
                    new_instr = IRInstr(dest=new_instr.dest, left=folded)
                    changes += 1

            # Track constant assignments
            if new_instr.dest and new_instr.op is None and _is_num(new_instr.left):
                const_map[new_instr.dest] = new_instr.left
            elif new_instr.dest:
                const_map.pop(new_instr.dest, None)

            new_instrs.append(new_instr)

        manager.log.append(f"    const-prop: {changes} substitution(s)")
        return IRFunction(func.name, new_instrs)


class DeadCodeElimination(TransformPass):
    name = "dce"
    dependencies = ["liveness"]
    invalidates = ["use-def", "liveness"]

    def run(self, func: IRFunction, manager: PassManager) -> IRFunction:
        liveness: LivenessResult = manager.get_analysis("liveness", func)
        new_instrs = []
        removed = 0

        for idx, instr in enumerate(func.instrs):
            d = instr.defs()
            if d and d not in liveness.live_out.get(idx, set()):
                # Dead assignment -- skip if it is a temp variable
                if d.startswith("t"):
                    removed += 1
                    continue
            new_instrs.append(instr)

        manager.log.append(f"    dce: removed {removed} dead instruction(s)")
        return IRFunction(func.name, new_instrs)


class InstructionSimplification(TransformPass):
    name = "instsimplify"
    dependencies = []
    invalidates = ["use-def", "liveness"]

    def run(self, func: IRFunction, manager: PassManager) -> IRFunction:
        new_instrs = []
        changes = 0

        for instr in func.instrs:
            simplified = self._simplify(instr)
            if simplified is not instr:
                changes += 1
            new_instrs.append(simplified)

        manager.log.append(f"    instsimplify: {changes} simplification(s)")
        return IRFunction(func.name, new_instrs)

    def _simplify(self, instr: IRInstr) -> IRInstr:
        if not instr.op or not instr.dest:
            return instr

        l, op, r = instr.left, instr.op, instr.right
        # x + 0 = x
        if op == '+' and r == 0:
            return IRInstr(dest=instr.dest, left=l)
        if op == '+' and l == 0:
            return IRInstr(dest=instr.dest, left=r)
        # x * 1 = x
        if op == '*' and r == 1:
            return IRInstr(dest=instr.dest, left=l)
        if op == '*' and l == 1:
            return IRInstr(dest=instr.dest, left=r)
        # x * 0 = 0
        if op == '*' and (r == 0 or l == 0):
            return IRInstr(dest=instr.dest, left=0)
        # x - 0 = x
        if op == '-' and r == 0:
            return IRInstr(dest=instr.dest, left=l)
        # x - x = 0
        if op == '-' and l == r and isinstance(l, str):
            return IRInstr(dest=instr.dest, left=0)
        return instr


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_num(v: Any) -> bool:
    return isinstance(v, (int, float))


def _fold(op: str, l: Any, r: Any) -> Optional[Any]:
    try:
        ops = {'+': lambda a, b: a + b, '-': lambda a, b: a - b,
               '*': lambda a, b: a * b, '/': lambda a, b: a // b if b else None}
        return ops.get(op, lambda a, b: None)(l, r)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Pipeline Builder (declarative API)
# ---------------------------------------------------------------------------

class PipelineBuilder:
    """Fluent API for building optimization pipelines."""

    def __init__(self):
        self.manager = PassManager()
        self._register_default_passes()

    def _register_default_passes(self):
        self.manager.register(UseDefAnalysis())
        self.manager.register(LivenessAnalysis())
        self.manager.register(ConstantPropagation())
        self.manager.register(DeadCodeElimination())
        self.manager.register(InstructionSimplification())

    def add(self, *pass_names: str) -> PipelineBuilder:
        self.manager.schedule(*pass_names)
        return self

    def O0(self) -> PipelineBuilder:
        """No optimizations."""
        return self

    def O1(self) -> PipelineBuilder:
        """Basic optimizations."""
        return self.add("const-prop", "instsimplify", "dce")

    def O2(self) -> PipelineBuilder:
        """Iterative optimization (two rounds)."""
        return (self.add("const-prop", "instsimplify", "dce")
                    .add("const-prop", "instsimplify", "dce"))

    def run(self, func: IRFunction) -> IRFunction:
        return self.manager.run(func)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def make_sample_function() -> IRFunction:
    """
    Create a sample IR function with optimization opportunities:
      t0 = 3 + 4          (constant fold -> 7)
      t1 = t0 * 1         (simplify -> t0)
      x  = t1 + 0         (simplify -> t1)
      t2 = 10 - 10        (fold -> 0, then x * 0 -> 0 if used)
      t3 = a + b          (dead code if t3 unused)
      return x
    """
    return IRFunction("example", [
        IRInstr(dest="t0", op="+", left=3, right=4),
        IRInstr(dest="t1", op="*", left="t0", right=1),
        IRInstr(dest="x", op="+", left="t1", right=0),
        IRInstr(dest="t2", op="-", left=10, right=10),
        IRInstr(dest="t3", op="+", left="a", right="b"),
        IRInstr(dest=None, left="x", is_return=True),
    ])


def main():
    print("=" * 60)
    print("Modern Compiler Pass Infrastructure Demo")
    print("=" * 60)

    func = make_sample_function()
    print(f"\n--- Original IR ---")
    print(func)

    # Run O1 pipeline
    builder = PipelineBuilder().O1()
    optimized = builder.run(func)

    print(f"\n--- After O1 Pipeline ---")
    print(optimized)

    # Show pass manager log
    print(f"\n--- Pass Manager Log ---")
    for entry in builder.manager.log:
        print(f"  {entry}")

    # Run O2 pipeline on fresh function
    print(f"\n--- O2 Pipeline (two iterations) ---")
    func2 = make_sample_function()
    builder2 = PipelineBuilder().O2()
    optimized2 = builder2.run(func2)
    print(optimized2)
    for entry in builder2.manager.log:
        print(f"  {entry}")

    # Analysis demo
    print(f"\n--- Analysis Results ---")
    func3 = make_sample_function()
    pm = PassManager()
    pm.register(UseDefAnalysis())
    pm.register(LivenessAnalysis())
    ud = UseDefAnalysis().run(func3, pm)
    print(f"  Definitions: {ud.defs}")
    print(f"  Uses: {ud.uses}")

    lv = LivenessAnalysis().run(func3, pm)
    for i in range(len(func3.instrs)):
        print(f"  [{i}] live_in={sorted(lv.live_in[i]):20s} "
              f"live_out={sorted(lv.live_out[i])}")

    print(f"\n--- Architecture Summary ---")
    print("""
  Modern compiler infrastructure:
    - Pass Manager: orchestrates pass execution and caching
    - Analysis Passes: compute read-only information (liveness, dominance)
    - Transform Passes: modify IR, declare which analyses they invalidate
    - Pipeline Builder: declarative API for constructing -O1/-O2/-O3 levels
    - Dependency tracking: analyses are computed on-demand and cached
    """)


if __name__ == "__main__":
    main()
