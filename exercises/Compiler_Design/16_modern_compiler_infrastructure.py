"""
Exercises for Lesson 16: Modern Compiler Infrastructure
Topic: Compiler_Design

Solutions to practice problems covering LLVM IR generation, optimization
passes, MLIR dialect concepts, DSL implementation, incremental compilation,
and profile-guided optimization.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Set, Optional, Any, Tuple
from collections import defaultdict
import hashlib
import time


# === Exercise 1: LLVM IR Generation ===
# Problem: Generate LLVM IR text for simple functions. This exercises
# understanding of SSA form, basic blocks, and LLVM type system.

class LLVMType(Enum):
    I1 = "i1"
    I32 = "i32"
    I64 = "i64"
    FLOAT = "float"
    DOUBLE = "double"
    VOID = "void"
    PTR = "ptr"


@dataclass
class IRValue:
    name: str
    ty: LLVMType

    def __repr__(self):
        return f"{self.ty.value} {self.name}"


class LLVMIRBuilder:
    """Simplified LLVM IR text generator."""

    def __init__(self):
        self.lines: List[str] = []
        self._reg_counter = 0
        self._label_counter = 0

    def new_reg(self) -> str:
        self._reg_counter += 1
        return f"%{self._reg_counter}"

    def new_label(self, hint: str = "bb") -> str:
        self._label_counter += 1
        return f"{hint}{self._label_counter}"

    def function_header(self, name: str, ret_type: LLVMType,
                        params: List[Tuple[LLVMType, str]]) -> str:
        param_str = ", ".join(f"{t.value} {n}" for t, n in params)
        header = f"define {ret_type.value} @{name}({param_str}) {{"
        self.lines.append(header)
        return header

    def label(self, name: str):
        self.lines.append(f"{name}:")

    def add(self, dest: str, ty: LLVMType, a: str, b: str):
        self.lines.append(f"  {dest} = add {ty.value} {a}, {b}")

    def sub(self, dest: str, ty: LLVMType, a: str, b: str):
        self.lines.append(f"  {dest} = sub {ty.value} {a}, {b}")

    def mul(self, dest: str, ty: LLVMType, a: str, b: str):
        self.lines.append(f"  {dest} = mul {ty.value} {a}, {b}")

    def icmp(self, dest: str, cond: str, ty: LLVMType, a: str, b: str):
        self.lines.append(f"  {dest} = icmp {cond} {ty.value} {a}, {b}")

    def br_cond(self, cond: str, true_label: str, false_label: str):
        self.lines.append(f"  br i1 {cond}, label %{true_label}, label %{false_label}")

    def br(self, label: str):
        self.lines.append(f"  br label %{label}")

    def phi(self, dest: str, ty: LLVMType,
            incoming: List[Tuple[str, str]]):
        pairs = ", ".join(f"[ {val}, %{lbl} ]" for val, lbl in incoming)
        self.lines.append(f"  {dest} = phi {ty.value} {pairs}")

    def ret(self, ty: LLVMType, val: str):
        self.lines.append(f"  ret {ty.value} {val}")

    def ret_void(self):
        self.lines.append("  ret void")

    def close(self):
        self.lines.append("}")

    def emit(self) -> str:
        return "\n".join(self.lines)


def exercise_1():
    """Generate LLVM IR for simple functions."""
    print("Exercise 1: LLVM IR Generation")
    print()

    # (a) Simple function: int add(int a, int b) { return a + b; }
    print("(a) Function: int add(int a, int b) { return a + b; }")
    print()

    builder = LLVMIRBuilder()
    builder.function_header("add", LLVMType.I32,
                            [(LLVMType.I32, "%a"), (LLVMType.I32, "%b")])
    builder.label("entry")
    builder.add("%result", LLVMType.I32, "%a", "%b")
    builder.ret(LLVMType.I32, "%result")
    builder.close()

    print(builder.emit())
    print()

    # (b) Factorial with loop (demonstrates phi nodes and SSA)
    print("(b) Factorial with loop (SSA form with phi nodes):")
    print("    int factorial(int n) {")
    print("        int result = 1;")
    print("        for (int i = 1; i <= n; i++)")
    print("            result *= i;")
    print("        return result;")
    print("    }")
    print()

    builder2 = LLVMIRBuilder()
    builder2.function_header("factorial", LLVMType.I32,
                             [(LLVMType.I32, "%n")])
    builder2.label("entry")
    builder2.br("loop.header")

    builder2.label("loop.header")
    builder2.phi("%i", LLVMType.I32, [("1", "entry"), ("%i.next", "loop.body")])
    builder2.phi("%result", LLVMType.I32, [("1", "entry"), ("%result.next", "loop.body")])
    builder2.icmp("%cond", "sle", LLVMType.I32, "%i", "%n")
    builder2.br_cond("%cond", "loop.body", "exit")

    builder2.label("loop.body")
    builder2.mul("%result.next", LLVMType.I32, "%result", "%i")
    builder2.add("%i.next", LLVMType.I32, "%i", "1")
    builder2.br("loop.header")

    builder2.label("exit")
    builder2.ret(LLVMType.I32, "%result")
    builder2.close()

    print(builder2.emit())
    print()

    print("Key LLVM IR features demonstrated:")
    print("  - SSA form: each variable assigned exactly once")
    print("  - phi nodes: merge values from different predecessors")
    print("  - Typed instructions: every value has an explicit type")
    print("  - Basic blocks: each block ends with a terminator (br, ret)")
    print("  - Labels: basic block names used as branch targets")


# === Exercise 2: Optimization Pass Pipeline ===
# Problem: Simulate an optimization pass pipeline, showing how passes
# transform IR and depend on each other.

@dataclass
class IRFunction:
    name: str
    instructions: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def copy(self) -> 'IRFunction':
        return IRFunction(self.name, list(self.instructions),
                          dict(self.metadata))


class OptimizationPass:
    """Base class for optimization passes."""

    def __init__(self, name: str, requires: List[str] = None,
                 invalidates: List[str] = None):
        self.name = name
        self.requires = requires or []
        self.invalidates = invalidates or []

    def run(self, func: IRFunction) -> bool:
        """Run the pass. Return True if IR was modified."""
        raise NotImplementedError


class ConstantFoldingPass(OptimizationPass):
    def __init__(self):
        super().__init__("constant-folding", requires=[], invalidates=[])

    def run(self, func: IRFunction) -> bool:
        modified = False
        new_instrs = []
        for instr in func.instructions:
            # Simple pattern: %x = add i32 C1, C2
            if '= add i32' in instr or '= mul i32' in instr:
                parts = instr.split()
                try:
                    dest = parts[0]
                    op = parts[2]
                    a = int(parts[4].rstrip(','))
                    b = int(parts[5])
                    if op == 'add':
                        result = a + b
                    else:
                        result = a * b
                    new_instrs.append(f"{dest} = i32 {result}")
                    modified = True
                    continue
                except (ValueError, IndexError):
                    pass
            new_instrs.append(instr)
        func.instructions = new_instrs
        return modified


class DeadCodeEliminationPass(OptimizationPass):
    def __init__(self):
        super().__init__("dce", requires=[], invalidates=[])

    def run(self, func: IRFunction) -> bool:
        # Find all used variables
        used = set()
        for instr in func.instructions:
            parts = instr.split()
            for i, p in enumerate(parts):
                if p.startswith('%') and i > 0:  # skip destination
                    used.add(p.rstrip(','))

        # Remove definitions of unused variables
        modified = False
        new_instrs = []
        for instr in func.instructions:
            parts = instr.split()
            if parts and parts[0].startswith('%') and '=' in parts:
                dest = parts[0]
                if dest not in used and 'ret' not in instr:
                    modified = True
                    continue
            new_instrs.append(instr)
        func.instructions = new_instrs
        return modified


class PassManager:
    """Manages a pipeline of optimization passes."""

    def __init__(self):
        self.passes: List[OptimizationPass] = []
        self.log: List[str] = []

    def add_pass(self, p: OptimizationPass):
        self.passes.append(p)

    def run(self, func: IRFunction) -> IRFunction:
        self.log.clear()
        for p in self.passes:
            before = list(func.instructions)
            modified = p.run(func)
            status = "MODIFIED" if modified else "no change"
            self.log.append(f"  {p.name}: {status}")
            if modified:
                for i, (old, new) in enumerate(
                        zip(before, func.instructions + ['(removed)'] * 10)):
                    if i < len(before) and i < len(func.instructions):
                        if old != new:
                            self.log.append(f"    {old}")
                            self.log.append(f"    -> {new}")
        return func


def exercise_2():
    """Optimization pass pipeline simulation."""
    print("Optimization pass pipeline:")
    print()

    func = IRFunction("example", [
        "%1 = add i32 3, 4",
        "%2 = mul i32 %1, 2",
        "%3 = add i32 10, 20",    # result unused -> dead code
        "%4 = add i32 %1, %2",
        "ret i32 %4",
    ])

    print("Before optimization:")
    for instr in func.instructions:
        print(f"  {instr}")
    print()

    pm = PassManager()
    pm.add_pass(ConstantFoldingPass())
    pm.add_pass(DeadCodeEliminationPass())
    pm.add_pass(ConstantFoldingPass())  # Second round after DCE

    pm.run(func)

    print("Pass log:")
    for entry in pm.log:
        print(entry)
    print()

    print("After optimization:")
    for instr in func.instructions:
        print(f"  {instr}")
    print()

    print("Pass ordering matters:")
    print("  1. Constant folding first: simplifies constant expressions")
    print("  2. DCE: removes instructions whose results are never used")
    print("  3. Constant folding again: may find new opportunities after DCE")
    print()
    print("Real LLVM pass pipeline (O2) runs 50+ passes including:")
    print("  instcombine, simplifycfg, mem2reg, gvn, licm, loop-unroll,")
    print("  inline, sroa, adce, reassociate, ...")


# === Exercise 3: MLIR Dialect Concepts ===
# Problem: Demonstrate the multi-level IR concept by showing how the same
# computation is represented at different abstraction levels.

def exercise_3():
    """MLIR multi-level IR concept with dialect lowering."""
    print("MLIR Multi-Level IR: Matrix multiply at different abstraction levels")
    print()

    # Level 1: Linalg dialect (high-level)
    print("Level 1: Linalg dialect (high-level, mathematical)")
    print("-" * 50)
    linalg_ir = [
        '// Matrix multiply C = A * B',
        'func @matmul(%A: tensor<4x8xf32>, %B: tensor<8x4xf32>)',
        '    -> tensor<4x4xf32> {',
        '  %C = linalg.matmul ins(%A, %B : tensor<4x8xf32>, tensor<8x4xf32>)',
        '                      outs(%zero : tensor<4x4xf32>)',
        '                      -> tensor<4x4xf32>',
        '  return %C : tensor<4x4xf32>',
        '}',
    ]
    for line in linalg_ir:
        print(f"  {line}")
    print()

    # Level 2: Affine dialect (loop-level)
    print("Level 2: Affine dialect (explicit loops, polyhedral)")
    print("-" * 50)
    affine_ir = [
        'func @matmul(%A: memref<4x8xf32>, %B: memref<8x4xf32>,',
        '             %C: memref<4x4xf32>) {',
        '  affine.for %i = 0 to 4 {',
        '    affine.for %j = 0 to 4 {',
        '      affine.for %k = 0 to 8 {',
        '        %a = affine.load %A[%i, %k] : memref<4x8xf32>',
        '        %b = affine.load %B[%k, %j] : memref<8x4xf32>',
        '        %c = affine.load %C[%i, %j] : memref<4x4xf32>',
        '        %prod = arith.mulf %a, %b : f32',
        '        %sum = arith.addf %c, %prod : f32',
        '        affine.store %sum, %C[%i, %j] : memref<4x4xf32>',
        '      }',
        '    }',
        '  }',
        '  return',
        '}',
    ]
    for line in affine_ir:
        print(f"  {line}")
    print()

    # Level 3: SCF + memref (structured control flow)
    print("Level 3: SCF dialect (structured loops, buffer-level)")
    print("-" * 50)
    scf_ir = [
        'func @matmul(%A: memref<4x8xf32>, %B: memref<8x4xf32>,',
        '             %C: memref<4x4xf32>) {',
        '  %c0 = arith.constant 0 : index',
        '  %c4 = arith.constant 4 : index',
        '  %c8 = arith.constant 8 : index',
        '  %c1 = arith.constant 1 : index',
        '  scf.for %i = %c0 to %c4 step %c1 {',
        '    scf.for %j = %c0 to %c4 step %c1 {',
        '      scf.for %k = %c0 to %c8 step %c1 {',
        '        // ... load, multiply, accumulate, store ...',
        '      }',
        '    }',
        '  }',
        '  return',
        '}',
    ]
    for line in scf_ir:
        print(f"  {line}")
    print()

    # Level 4: LLVM dialect (low-level)
    print("Level 4: LLVM dialect (maps 1:1 to LLVM IR)")
    print("-" * 50)
    llvm_ir = [
        'llvm.func @matmul(%A: !llvm.ptr, %B: !llvm.ptr, %C: !llvm.ptr) {',
        '  // Nested loops with GEP, load, fmul, fadd, store',
        '  // Index calculations explicit',
        '  // All types are LLVM types',
        '  llvm.br ^loop_i(%c0 : i64)',
        '^loop_i(%i: i64):',
        '  %cond_i = llvm.icmp "slt" %i, %c4 : i64',
        '  llvm.cond_br %cond_i, ^body_i, ^exit',
        '^body_i:',
        '  // ... nested j, k loops with llvm.getelementptr ...',
        '  llvm.br ^loop_i(%i_next : i64)',
        '^exit:',
        '  llvm.return',
        '}',
    ]
    for line in llvm_ir:
        print(f"  {line}")
    print()

    print("MLIR key concepts:")
    print("  - Each level (dialect) captures different information")
    print("  - Linalg: knows about linear algebra semantics (can tile, fuse)")
    print("  - Affine: knows about loop structure (can optimize memory access)")
    print("  - SCF: general structured control flow")
    print("  - LLVM: ready for code generation")
    print("  - Progressive lowering: optimize at the highest level possible,")
    print("    then lower to the next level")


# === Exercise 4: DSL Implementation ===
# Problem: Implement a tiny domain-specific language for describing
# data processing pipelines.

class PipelineOp(Enum):
    FILTER = auto()
    MAP = auto()
    REDUCE = auto()
    SORT = auto()
    TAKE = auto()


@dataclass
class PipelineNode:
    op: PipelineOp
    args: Any = None
    source: Optional['PipelineNode'] = None


class PipelineDSL:
    """A DSL for data processing pipelines with lazy evaluation."""

    def __init__(self, data: List[Any]):
        self._data = data
        self._head: Optional[PipelineNode] = None
        self._ir: List[Tuple[str, Any]] = []

    def filter(self, predicate) -> 'PipelineDSL':
        node = PipelineNode(PipelineOp.FILTER, predicate, self._head)
        new = PipelineDSL(self._data)
        new._head = node
        new._ir = self._ir + [("FILTER", predicate.__doc__ or str(predicate))]
        return new

    def map(self, transform) -> 'PipelineDSL':
        node = PipelineNode(PipelineOp.MAP, transform, self._head)
        new = PipelineDSL(self._data)
        new._head = node
        new._ir = self._ir + [("MAP", transform.__doc__ or str(transform))]
        return new

    def take(self, n: int) -> 'PipelineDSL':
        node = PipelineNode(PipelineOp.TAKE, n, self._head)
        new = PipelineDSL(self._data)
        new._head = node
        new._ir = self._ir + [("TAKE", n)]
        return new

    def sort(self, key=None) -> 'PipelineDSL':
        node = PipelineNode(PipelineOp.SORT, key, self._head)
        new = PipelineDSL(self._data)
        new._head = node
        new._ir = self._ir + [("SORT", key.__doc__ if key else "default")]
        return new

    def reduce(self, func, initial=0):
        """Terminal operation: eagerly evaluate the pipeline."""
        data = self._execute()
        result = initial
        for item in data:
            result = func(result, item)
        return result

    def collect(self) -> List[Any]:
        """Terminal operation: collect results into a list."""
        return self._execute()

    def _execute(self) -> List[Any]:
        """Execute the pipeline by walking the operation chain."""
        # Flatten the chain
        ops = []
        node = self._head
        while node:
            ops.append(node)
            node = node.source
        ops.reverse()

        data = list(self._data)
        for op in ops:
            if op.op == PipelineOp.FILTER:
                data = [x for x in data if op.args(x)]
            elif op.op == PipelineOp.MAP:
                data = [op.args(x) for x in data]
            elif op.op == PipelineOp.TAKE:
                data = data[:op.args]
            elif op.op == PipelineOp.SORT:
                data = sorted(data, key=op.args)
        return data

    def show_plan(self):
        """Display the execution plan (IR)."""
        print("  Execution plan:")
        for i, (op, arg) in enumerate(self._ir):
            print(f"    {i}: {op}({arg})")


def exercise_4():
    """DSL for data processing pipelines."""
    print("Domain-Specific Language: Data Processing Pipeline")
    print()

    data = list(range(1, 101))

    # Build pipeline using DSL
    pipeline = (
        PipelineDSL(data)
        .filter(lambda x: x % 2 == 0)    # even numbers
        .map(lambda x: x * x)             # square them
        .filter(lambda x: x > 100)        # keep those > 100
        .take(5)                           # first 5
    )

    pipeline.show_plan()
    result = pipeline.collect()
    print(f"  Result: {result}")
    print()

    # Reduce example
    total = (
        PipelineDSL(data)
        .filter(lambda x: x % 3 == 0)
        .map(lambda x: x * 2)
        .reduce(lambda acc, x: acc + x, 0)
    )
    print(f"  Sum of doubled multiples of 3 (1..100): {total}")
    expected = sum(x * 2 for x in range(1, 101) if x % 3 == 0)
    print(f"  Expected: {expected}, Correct: {total == expected}")
    print()

    # Optimization opportunity: filter-map fusion
    print("DSL optimization opportunities:")
    print("  1. Filter-map fusion: combine adjacent filter+map into single pass")
    print("  2. Take pushdown: stop processing after taking N items")
    print("  3. Predicate merging: combine adjacent filters")
    print("  4. Sort elimination: remove sort if output is consumed unordered")
    print()
    print("This is why DSLs are powerful for compilers:")
    print("  - Domain knowledge enables optimizations impossible in general code")
    print("  - Lazy evaluation enables whole-pipeline optimization")
    print("  - The DSL serves as both user interface and IR")


# === Exercise 5: Incremental Compilation ===
# Problem: Implement a simple incremental compilation system that only
# recompiles changed modules.

@dataclass
class Module:
    name: str
    source: str
    dependencies: List[str] = field(default_factory=list)
    compiled_hash: Optional[str] = None
    object_code: Optional[str] = None


class IncrementalCompiler:
    """Incremental compilation system with dependency tracking."""

    def __init__(self):
        self.modules: Dict[str, Module] = {}
        self.compile_count = 0
        self.cache_hits = 0

    def add_module(self, name: str, source: str, deps: List[str] = None):
        self.modules[name] = Module(name, source, deps or [])

    def _content_hash(self, module: Module) -> str:
        """Hash source + dependency hashes for cache key."""
        hasher = hashlib.md5()
        hasher.update(module.source.encode())
        for dep in sorted(module.dependencies):
            dep_mod = self.modules.get(dep)
            if dep_mod and dep_mod.compiled_hash:
                hasher.update(dep_mod.compiled_hash.encode())
        return hasher.hexdigest()[:12]

    def _compile_module(self, module: Module) -> str:
        """Simulate compilation."""
        self.compile_count += 1
        # Simulate: object code is just an uppercase transform
        return f"OBJ[{module.name}:{len(module.source)}bytes]"

    def build(self) -> Dict[str, str]:
        """Build all modules, recompiling only what changed."""
        # Topological sort for dependency order
        order = self._topo_sort()
        results = {}
        recompiled = []
        cached = []

        for name in order:
            module = self.modules[name]
            current_hash = self._content_hash(module)

            if module.compiled_hash == current_hash and module.object_code:
                self.cache_hits += 1
                cached.append(name)
            else:
                module.object_code = self._compile_module(module)
                module.compiled_hash = current_hash
                recompiled.append(name)

            results[name] = module.object_code

        return results, recompiled, cached

    def _topo_sort(self) -> List[str]:
        """Topological sort of modules by dependencies."""
        visited = set()
        order = []

        def visit(name):
            if name in visited:
                return
            visited.add(name)
            module = self.modules.get(name)
            if module:
                for dep in module.dependencies:
                    visit(dep)
            order.append(name)

        for name in self.modules:
            visit(name)
        return order


def exercise_5():
    """Incremental compilation with dependency tracking."""
    print("Incremental compilation system:")
    print()

    compiler = IncrementalCompiler()

    # Set up modules with dependencies
    compiler.add_module("utils", "def helper(): pass", [])
    compiler.add_module("math_lib", "import utils; def compute(): pass", ["utils"])
    compiler.add_module("io_lib", "def read_file(): pass", [])
    compiler.add_module("app", "import math_lib, io_lib; def main(): pass",
                        ["math_lib", "io_lib"])

    print("Module dependency graph:")
    print("  utils <- math_lib <- app")
    print("  io_lib <-----------/")
    print()

    # First build: everything compiles
    print("Build 1: Initial (clean build)")
    results, recompiled, cached = compiler.build()
    print(f"  Recompiled: {recompiled}")
    print(f"  Cached: {cached}")
    print(f"  Total compiles: {compiler.compile_count}")
    print()

    # Second build: nothing changed
    print("Build 2: No changes")
    results, recompiled, cached = compiler.build()
    print(f"  Recompiled: {recompiled}")
    print(f"  Cached: {cached}")
    print(f"  Total compiles: {compiler.compile_count}")
    print()

    # Third build: change utils (should cascade to math_lib and app)
    print("Build 3: Change utils (should cascade to dependents)")
    compiler.modules["utils"].source = "def helper(): return 42"
    results, recompiled, cached = compiler.build()
    print(f"  Recompiled: {recompiled}")
    print(f"  Cached: {cached}")
    print(f"  Total compiles: {compiler.compile_count}")
    print()

    # Fourth build: change io_lib (should cascade only to app)
    print("Build 4: Change io_lib (only io_lib and app recompile)")
    compiler.modules["io_lib"].source = "def read_file(): return open('x')"
    results, recompiled, cached = compiler.build()
    print(f"  Recompiled: {recompiled}")
    print(f"  Cached: {cached}")
    print(f"  Total compiles: {compiler.compile_count}")
    print()

    print("Incremental compilation strategies:")
    print("  1. File-level: recompile changed files (Make, most C compilers)")
    print("  2. Function-level: recompile changed functions (Rust incremental)")
    print("  3. Query-based: recompute only affected queries (Salsa/rustc)")
    print("  4. Content-addressed: hash-based caching (Bazel, Nix)")


# === Exercise 6: Profile-Guided Optimization ===
# Problem: Simulate profile-guided optimization by collecting execution
# profiles and using them to optimize branch layout and inlining.

def exercise_6():
    """Profile-guided optimization (PGO) simulation."""
    print("Profile-Guided Optimization (PGO)")
    print()

    # Simulate a function with branches
    def process_request(request_type: str) -> str:
        """Simulate request processing with different paths."""
        if request_type == "GET":
            return "page"
        elif request_type == "POST":
            return "created"
        elif request_type == "PUT":
            return "updated"
        elif request_type == "DELETE":
            return "deleted"
        else:
            return "error"

    # Step 1: Profiling run
    print("Step 1: Profiling run (collect branch frequencies)")
    import random
    random.seed(42)

    # Real-world distribution: GET is most common
    profile_data = {
        "GET": 0, "POST": 0, "PUT": 0, "DELETE": 0, "OTHER": 0
    }
    distribution = (["GET"] * 70 + ["POST"] * 20 + ["PUT"] * 5 +
                    ["DELETE"] * 3 + ["OTHER"] * 2)

    N = 10000
    for _ in range(N):
        req = random.choice(distribution)
        process_request(req)
        profile_data[req] += 1

    print("  Branch frequencies:")
    for req_type, count in sorted(profile_data.items(),
                                  key=lambda x: -x[1]):
        pct = count / N * 100
        bar = "#" * int(pct / 2)
        print(f"    {req_type:8s}: {count:5d} ({pct:5.1f}%) {bar}")
    print()

    # Step 2: Optimize branch ordering
    print("Step 2: PGO-optimized branch layout")
    print()
    print("  Before PGO (alphabetical order):")
    print("    if (type == DELETE) ...     // 3% -- checked first!")
    print("    elif (type == GET) ...      // 70%")
    print("    elif (type == OTHER) ...    // 2%")
    print("    elif (type == POST) ...     // 20%")
    print("    elif (type == PUT) ...      // 5%")
    print()
    print("  After PGO (hot-path first):")
    print("    if (type == GET) ...        // 70% -- most likely, checked first")
    print("    elif (type == POST) ...     // 20%")
    print("    elif (type == PUT) ...      // 5%")
    print("    elif (type == DELETE) ...   // 3%")
    print("    else ...                    // 2%")
    print()

    # Compute average comparisons needed
    sorted_types = sorted(profile_data.items(), key=lambda x: -x[1])
    avg_cmp_before = 0
    for i, (_, count) in enumerate(sorted(profile_data.items())):
        avg_cmp_before += (i + 1) * count
    avg_cmp_before /= N

    avg_cmp_after = 0
    for i, (_, count) in enumerate(sorted_types):
        avg_cmp_after += (i + 1) * count
    avg_cmp_after /= N

    print(f"  Average comparisons (before PGO): {avg_cmp_before:.2f}")
    print(f"  Average comparisons (after PGO):  {avg_cmp_after:.2f}")
    print(f"  Improvement: {(1 - avg_cmp_after/avg_cmp_before)*100:.1f}%")
    print()

    # Step 3: Inlining decisions
    print("Step 3: PGO-guided inlining decisions")
    print()

    call_counts = {
        "validate_input": 10000,
        "process_data": 10000,
        "log_request": 10000,
        "handle_error": 150,
        "send_notification": 50,
        "generate_report": 5,
    }
    func_sizes = {
        "validate_input": 15,     # small, hot -> inline
        "process_data": 200,      # large, hot -> maybe inline
        "log_request": 30,        # medium, hot -> inline
        "handle_error": 80,       # medium, cold -> don't inline
        "send_notification": 50,  # medium, cold -> don't inline
        "generate_report": 500,   # large, cold -> don't inline
    }

    print("  Function       Calls   Size  Decision")
    print("  " + "-" * 55)
    for func in call_counts:
        calls = call_counts[func]
        size = func_sizes[func]
        hot = calls > 1000
        small = size < 50

        if hot and small:
            decision = "INLINE (hot + small)"
        elif hot and size < 100:
            decision = "INLINE (hot, medium)"
        elif hot:
            decision = "PARTIAL INLINE (hot but large)"
        else:
            decision = "NO INLINE (cold)"

        print(f"  {func:22s} {calls:5d}  {size:4d}  {decision}")
    print()

    print("PGO workflow:")
    print("  1. Compile with instrumentation:  clang -fprofile-generate ...")
    print("  2. Run representative workload:   ./app < training_data")
    print("  3. Compile with profile data:     clang -fprofile-use=... ...")
    print()
    print("PGO optimizations:")
    print("  - Branch layout (hot path first)")
    print("  - Function inlining (inline hot callees)")
    print("  - Basic block placement (fall-through for hot paths)")
    print("  - Register allocation (prioritize hot variables)")
    print("  - Loop unrolling (unroll hot loops more aggressively)")
    print("  - Typical improvement: 10-30% for real workloads")


if __name__ == "__main__":
    print("=" * 60)
    print("=== Exercise 1: LLVM IR Generation ===")
    print("=" * 60)
    exercise_1()

    print("\n" + "=" * 60)
    print("=== Exercise 2: Optimization Pass Pipeline ===")
    print("=" * 60)
    exercise_2()

    print("\n" + "=" * 60)
    print("=== Exercise 3: MLIR Dialect Concepts ===")
    print("=" * 60)
    exercise_3()

    print("\n" + "=" * 60)
    print("=== Exercise 4: DSL Implementation ===")
    print("=" * 60)
    exercise_4()

    print("\n" + "=" * 60)
    print("=== Exercise 5: Incremental Compilation ===")
    print("=" * 60)
    exercise_5()

    print("\n" + "=" * 60)
    print("=== Exercise 6: Profile-Guided Optimization ===")
    print("=" * 60)
    exercise_6()

    print("\nAll exercises completed!")
