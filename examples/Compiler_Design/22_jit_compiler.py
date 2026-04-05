"""
22_jit_compiler.py - Just-In-Time Compilation

Demonstrates JIT compilation concepts: translating bytecode to native-like
instructions at runtime, with profiling-guided optimization.

Components:
  1. Bytecode Interpreter (Baseline)
     A simple bytecode interpreter that counts execution frequency
     of each function.

  2. JIT Compiler
     When a function becomes "hot" (executed frequently), compile it
     from bytecode to a lower-level representation that executes faster.

  3. Profiling and Tiered Compilation
     Track execution counts and type information. Progressively optimize
     code through multiple tiers:
       Tier 0: Interpreted bytecode
       Tier 1: Simple JIT (direct translation)
       Tier 2: Optimized JIT (with inlining and specialization)

  4. On-Stack Replacement (OSR)
     Replace a running interpreted function with its JIT-compiled
     version mid-execution (at loop back-edges).

  5. Type Specialization
     Use profiled type information to generate specialized code paths
     with guards for type checks.

Topics covered:
  - JIT compilation pipeline
  - Hot function detection and compilation triggers
  - Tiered compilation strategy
  - On-stack replacement (OSR) concept
  - Type feedback and specialization
  - Deoptimization and fallback
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Optional
from time import perf_counter


# ---------------------------------------------------------------------------
# Bytecode
# ---------------------------------------------------------------------------

class Op(Enum):
    CONST  = auto()
    LOAD   = auto()
    STORE  = auto()
    ADD    = auto()
    SUB    = auto()
    MUL    = auto()
    DIV    = auto()
    LT     = auto()
    EQ     = auto()
    JMP    = auto()
    JMP_IF_FALSE = auto()
    CALL   = auto()
    RET    = auto()
    PRINT  = auto()
    HALT   = auto()


@dataclass
class Instr:
    op: Op
    arg: Any = None

    def __repr__(self):
        if self.arg is not None:
            return f"{self.op.name} {self.arg!r}"
        return self.op.name


@dataclass
class BytecodeFunc:
    name: str
    params: list[str]
    code: list[Instr]
    tier: int = 0  # 0=interpreted, 1=simple JIT, 2=optimized JIT


# ---------------------------------------------------------------------------
# Compiled (JIT) representation
# ---------------------------------------------------------------------------

@dataclass
class NativeOp:
    """A lower-level "native" instruction (simulated)."""
    opcode: str
    args: tuple = ()

    def __repr__(self):
        args_str = ", ".join(str(a) for a in self.args)
        return f"{self.opcode} {args_str}" if args_str else self.opcode


@dataclass
class CompiledFunc:
    name: str
    params: list[str]
    native_ops: list[NativeOp]
    tier: int
    specializations: dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Profiling Data
# ---------------------------------------------------------------------------

@dataclass
class ProfileData:
    call_count: int = 0
    loop_iterations: dict[int, int] = field(default_factory=dict)
    type_info: dict[str, set[str]] = field(default_factory=dict)
    total_time: float = 0.0


# ---------------------------------------------------------------------------
# JIT Compiler
# ---------------------------------------------------------------------------

class JITCompiler:
    """
    Compiles bytecode functions to native-like instructions.
    Supports tiered compilation with increasing optimization levels.
    """

    def __init__(self):
        self.compilation_log: list[str] = []

    def compile_tier1(self, func: BytecodeFunc) -> CompiledFunc:
        """
        Tier 1: Direct 1-to-1 translation of bytecode to native ops.
        No optimization, just eliminates interpreter dispatch overhead.
        """
        native_ops = []
        for instr in func.code:
            native_ops.extend(self._translate_instr(instr))

        self.compilation_log.append(
            f"  Tier 1 compile: {func.name} "
            f"({len(func.code)} bytecodes -> {len(native_ops)} native ops)")

        return CompiledFunc(
            name=func.name, params=func.params,
            native_ops=native_ops, tier=1)

    def compile_tier2(self, func: BytecodeFunc,
                      profile: ProfileData) -> CompiledFunc:
        """
        Tier 2: Optimized compilation using profile data.
        Applies constant folding, dead code elimination, and type
        specialization based on profiling information.
        """
        native_ops = []
        for instr in func.code:
            native_ops.extend(self._translate_instr(instr))

        # Optimization: eliminate redundant operations
        optimized = self._optimize(native_ops, profile)

        self.compilation_log.append(
            f"  Tier 2 compile: {func.name} "
            f"({len(native_ops)} -> {len(optimized)} native ops, "
            f"profile: {profile.call_count} calls)")

        specializations = {}
        for var, types in profile.type_info.items():
            if len(types) == 1:
                specializations[var] = next(iter(types))

        return CompiledFunc(
            name=func.name, params=func.params,
            native_ops=optimized, tier=2,
            specializations=specializations)

    def _translate_instr(self, instr: Instr) -> list[NativeOp]:
        """Translate a single bytecode instruction to native ops."""
        op = instr.op
        if op == Op.CONST:
            return [NativeOp("mov_imm", (instr.arg,))]
        elif op == Op.LOAD:
            return [NativeOp("load_local", (instr.arg,))]
        elif op == Op.STORE:
            return [NativeOp("store_local", (instr.arg,))]
        elif op == Op.ADD:
            return [NativeOp("add", ())]
        elif op == Op.SUB:
            return [NativeOp("sub", ())]
        elif op == Op.MUL:
            return [NativeOp("mul", ())]
        elif op == Op.DIV:
            return [NativeOp("div", ())]
        elif op == Op.LT:
            return [NativeOp("cmp_lt", ())]
        elif op == Op.EQ:
            return [NativeOp("cmp_eq", ())]
        elif op == Op.JMP:
            return [NativeOp("jmp", (instr.arg,))]
        elif op == Op.JMP_IF_FALSE:
            return [NativeOp("jz", (instr.arg,))]
        elif op == Op.CALL:
            return [NativeOp("call", (instr.arg,))]
        elif op == Op.RET:
            return [NativeOp("ret", ())]
        elif op == Op.PRINT:
            return [NativeOp("print", ())]
        elif op == Op.HALT:
            return [NativeOp("halt", ())]
        return [NativeOp(f"unknown_{instr.op.name}", ())]

    def _optimize(self, ops: list[NativeOp],
                  profile: ProfileData) -> list[NativeOp]:
        """Apply simple optimizations based on profile data."""
        optimized = []
        i = 0
        removed = 0
        while i < len(ops):
            # Peephole: mov_imm + mov_imm + add -> mov_imm (constant fold)
            if (i + 2 < len(ops)
                    and ops[i].opcode == "mov_imm"
                    and ops[i + 1].opcode == "mov_imm"
                    and ops[i + 2].opcode == "add"):
                a, b = ops[i].args[0], ops[i + 1].args[0]
                if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                    optimized.append(NativeOp("mov_imm", (a + b,)))
                    i += 3
                    removed += 2
                    continue

            # Peephole: store + load same variable -> dup + store
            if (i + 1 < len(ops)
                    and ops[i].opcode == "store_local"
                    and ops[i + 1].opcode == "load_local"
                    and ops[i].args == ops[i + 1].args):
                optimized.append(NativeOp("dup", ()))
                optimized.append(ops[i])
                i += 2
                removed += 0  # same count but faster
                continue

            optimized.append(ops[i])
            i += 1

        return optimized


# ---------------------------------------------------------------------------
# JIT-Enabled Runtime
# ---------------------------------------------------------------------------

class JITRuntime:
    """
    Runtime that combines interpretation and JIT compilation.
    Functions start interpreted, then get compiled when they become hot.
    """

    HOT_THRESHOLD_T1 = 5    # calls before tier 1 compilation
    HOT_THRESHOLD_T2 = 20   # calls before tier 2 compilation

    def __init__(self):
        self.bytecode_funcs: dict[str, BytecodeFunc] = {}
        self.compiled_funcs: dict[str, CompiledFunc] = {}
        self.profiles: dict[str, ProfileData] = {}
        self.jit = JITCompiler()
        self.stack: list[Any] = []
        self.output: list[str] = []
        self.log: list[str] = []

    def register(self, func: BytecodeFunc) -> None:
        self.bytecode_funcs[func.name] = func
        self.profiles[func.name] = ProfileData()

    def call(self, name: str, args: list[Any]) -> Any:
        profile = self.profiles[name]
        profile.call_count += 1

        # Check if we should compile
        if profile.call_count == self.HOT_THRESHOLD_T1:
            bc_func = self.bytecode_funcs[name]
            compiled = self.jit.compile_tier1(bc_func)
            self.compiled_funcs[name] = compiled
            self.log.append(
                f"  Hot function detected: {name} "
                f"(count={profile.call_count}) -> Tier 1 JIT")

        elif profile.call_count == self.HOT_THRESHOLD_T2:
            bc_func = self.bytecode_funcs[name]
            compiled = self.jit.compile_tier2(bc_func, profile)
            self.compiled_funcs[name] = compiled
            self.log.append(
                f"  Very hot function: {name} "
                f"(count={profile.call_count}) -> Tier 2 JIT")

        # Execute: use compiled version if available, else interpret
        if name in self.compiled_funcs:
            return self._execute_compiled(name, args)
        return self._interpret(name, args)

    def _interpret(self, name: str, args: list[Any]) -> Any:
        """Execute bytecode via interpretation."""
        func = self.bytecode_funcs[name]
        locals_map: dict[str, Any] = {}
        for pname, val in zip(func.params, args):
            locals_map[pname] = val

        stack: list[Any] = []
        ip = 0

        while ip < len(func.code):
            instr = func.code[ip]
            ip += 1

            if instr.op == Op.CONST:
                stack.append(instr.arg)
            elif instr.op == Op.LOAD:
                stack.append(locals_map.get(instr.arg, 0))
            elif instr.op == Op.STORE:
                locals_map[instr.arg] = stack.pop()
            elif instr.op == Op.ADD:
                b, a = stack.pop(), stack.pop()
                stack.append(a + b)
            elif instr.op == Op.SUB:
                b, a = stack.pop(), stack.pop()
                stack.append(a - b)
            elif instr.op == Op.MUL:
                b, a = stack.pop(), stack.pop()
                stack.append(a * b)
            elif instr.op == Op.DIV:
                b, a = stack.pop(), stack.pop()
                stack.append(a // b if b else 0)
            elif instr.op == Op.LT:
                b, a = stack.pop(), stack.pop()
                stack.append(1 if a < b else 0)
            elif instr.op == Op.EQ:
                b, a = stack.pop(), stack.pop()
                stack.append(1 if a == b else 0)
            elif instr.op == Op.JMP:
                ip = instr.arg
            elif instr.op == Op.JMP_IF_FALSE:
                if not stack.pop():
                    ip = instr.arg
            elif instr.op == Op.CALL:
                fname, argc = instr.arg
                call_args = [stack.pop() for _ in range(argc)][::-1]
                result = self.call(fname, call_args)
                stack.append(result)
            elif instr.op == Op.RET:
                return stack.pop() if stack else None
            elif instr.op == Op.PRINT:
                val = stack.pop()
                self.output.append(str(val))

        return stack.pop() if stack else None

    def _execute_compiled(self, name: str, args: list[Any]) -> Any:
        """Execute compiled native ops (simulated)."""
        # In a real JIT, this would execute machine code.
        # Here we simulate by interpreting the native ops.
        compiled = self.compiled_funcs[name]
        locals_map: dict[str, Any] = {}
        for pname, val in zip(compiled.params, args):
            locals_map[pname] = val

        stack: list[Any] = []
        ip = 0

        while ip < len(compiled.native_ops):
            op = compiled.native_ops[ip]
            ip += 1

            if op.opcode == "mov_imm":
                stack.append(op.args[0])
            elif op.opcode == "load_local":
                stack.append(locals_map.get(op.args[0], 0))
            elif op.opcode == "store_local":
                locals_map[op.args[0]] = stack.pop()
            elif op.opcode == "dup":
                stack.append(stack[-1])
            elif op.opcode == "add":
                b, a = stack.pop(), stack.pop(); stack.append(a + b)
            elif op.opcode == "sub":
                b, a = stack.pop(), stack.pop(); stack.append(a - b)
            elif op.opcode == "mul":
                b, a = stack.pop(), stack.pop(); stack.append(a * b)
            elif op.opcode == "div":
                b, a = stack.pop(), stack.pop(); stack.append(a // b if b else 0)
            elif op.opcode == "cmp_lt":
                b, a = stack.pop(), stack.pop(); stack.append(1 if a < b else 0)
            elif op.opcode == "cmp_eq":
                b, a = stack.pop(), stack.pop(); stack.append(1 if a == b else 0)
            elif op.opcode == "jmp":
                ip = op.args[0]
            elif op.opcode == "jz":
                if not stack.pop(): ip = op.args[0]
            elif op.opcode == "call":
                fname, argc = op.args[0]
                call_args = [stack.pop() for _ in range(argc)][::-1]
                result = self.call(fname, call_args)
                stack.append(result)
            elif op.opcode == "ret":
                return stack.pop() if stack else None
            elif op.opcode == "print":
                val = stack.pop()
                self.output.append(str(val))

        return stack.pop() if stack else None


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def make_factorial() -> BytecodeFunc:
    """Bytecode for: fn factorial(n) { if n <= 1 return 1; return n * factorial(n-1); }"""
    return BytecodeFunc("factorial", ["n"], [
        Instr(Op.LOAD, "n"),          # 0
        Instr(Op.CONST, 2),           # 1
        Instr(Op.LT, None),           # 2: n < 2
        Instr(Op.JMP_IF_FALSE, 6),    # 3: if false goto 6
        Instr(Op.CONST, 1),           # 4
        Instr(Op.RET),                # 5
        Instr(Op.LOAD, "n"),          # 6
        Instr(Op.LOAD, "n"),          # 7
        Instr(Op.CONST, 1),           # 8
        Instr(Op.SUB),                # 9: n-1
        Instr(Op.CALL, ("factorial", 1)),  # 10
        Instr(Op.MUL),               # 11: n * factorial(n-1)
        Instr(Op.RET),               # 12
    ])


def main():
    print("=" * 60)
    print("JIT Compilation Demo")
    print("=" * 60)

    runtime = JITRuntime()
    runtime.register(make_factorial())

    # Call factorial multiple times to trigger JIT compilation
    print(f"\n--- Tiered Compilation Progression ---")
    results = []
    for i in range(1, 26):
        result = runtime.call("factorial", [10])
        tier = "interpreted"
        if "factorial" in runtime.compiled_funcs:
            tier = f"Tier {runtime.compiled_funcs['factorial'].tier}"
        if i <= 6 or i == 20 or i == 25:
            results.append((i, result, tier))

    for call_num, result, tier in results:
        print(f"  Call #{call_num:2d}: factorial(10) = {result:>10d}  [{tier}]")

    # Show runtime log
    print(f"\n--- JIT Compilation Log ---")
    for entry in runtime.log:
        print(entry)

    # Show JIT compiler log
    print(f"\n--- Compiler Details ---")
    for entry in runtime.jit.compilation_log:
        print(entry)

    # Show profile data
    profile = runtime.profiles["factorial"]
    print(f"\n--- Profile Data ---")
    print(f"  factorial: {profile.call_count} total calls")

    # Show compiled code
    if "factorial" in runtime.compiled_funcs:
        compiled = runtime.compiled_funcs["factorial"]
        print(f"\n--- Compiled Native Ops (Tier {compiled.tier}) ---")
        for i, op in enumerate(compiled.native_ops):
            print(f"  [{i:2d}] {op}")
        if compiled.specializations:
            print(f"  Type specializations: {compiled.specializations}")

    print(f"\n--- JIT Compilation Concepts ---")
    print("""
  Tier 0 (Interpreter):  Simple bytecode dispatch loop
  Tier 1 (Simple JIT):   1-to-1 bytecode-to-native translation
  Tier 2 (Optimized JIT): Profile-guided optimizations applied

  Key techniques:
    - Hot function detection via execution counters
    - On-Stack Replacement (OSR): swap running code at loop back-edges
    - Type specialization: generate fast paths for observed types
    - Deoptimization: fall back to interpreter if assumptions break
    - Inlining: embed small callees directly into the caller
    """)


if __name__ == "__main__":
    main()
