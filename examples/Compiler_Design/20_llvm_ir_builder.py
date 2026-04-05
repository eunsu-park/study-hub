"""
20_llvm_ir_builder.py - LLVM IR Introduction and Builder

Demonstrates the structure and generation of LLVM-style intermediate
representation. This example builds a simplified LLVM IR text format
from an AST, showing the key concepts of LLVM IR.

Components:
  1. LLVM IR Data Model
     Types, values, basic blocks, and instructions that mirror LLVM's
     type system and instruction set.

  2. IR Builder
     A builder API (similar to LLVM's IRBuilder) that provides methods
     to create instructions with automatic SSA naming and type checking.

  3. Module and Function Construction
     Create functions with typed parameters, basic blocks, and control
     flow using LLVM IR conventions.

  4. Code Generation from AST
     Translate a simple expression language into LLVM IR text format.

Topics covered:
  - LLVM IR type system (i32, i1, void, pointers)
  - SSA form with numbered temporaries (%0, %1, ...)
  - Basic block structure and terminators
  - LLVM instructions: add, sub, mul, icmp, br, ret, phi, alloca, load, store
  - Function definitions and declarations
  - LLVM IR text format syntax
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Optional, Union


# ---------------------------------------------------------------------------
# LLVM IR Type System
# ---------------------------------------------------------------------------

class IRType:
    """Base class for LLVM IR types."""
    pass

class IntType(IRType):
    def __init__(self, bits: int = 32):
        self.bits = bits
    def __str__(self):
        return f"i{self.bits}"
    def __eq__(self, other):
        return isinstance(other, IntType) and self.bits == other.bits
    def __hash__(self):
        return hash(self.bits)

class VoidType(IRType):
    def __str__(self):
        return "void"

class PtrType(IRType):
    def __init__(self, pointee: IRType):
        self.pointee = pointee
    def __str__(self):
        return f"ptr"

class FuncType(IRType):
    def __init__(self, ret: IRType, params: list[IRType]):
        self.ret = ret
        self.params = params
    def __str__(self):
        params_str = ", ".join(str(p) for p in self.params)
        return f"{self.ret} ({params_str})"


# Convenience type constants
i1 = IntType(1)
i32 = IntType(32)
i64 = IntType(64)
void = VoidType()


# ---------------------------------------------------------------------------
# LLVM IR Values
# ---------------------------------------------------------------------------

@dataclass
class Value:
    name: str
    type: IRType

    def __str__(self):
        return self.name

@dataclass
class Constant(Value):
    val: Any = 0

    def __str__(self):
        return str(self.val)


def const_int(val: int, ty: IRType = i32) -> Constant:
    return Constant(name=str(val), type=ty, val=val)


# ---------------------------------------------------------------------------
# LLVM IR Instructions
# ---------------------------------------------------------------------------

@dataclass
class IRInstruction:
    """Base for all LLVM IR instructions."""

    def ir_str(self) -> str:
        raise NotImplementedError


@dataclass
class BinOpInst(IRInstruction):
    dest: Value
    op: str  # add, sub, mul, sdiv
    left: Value
    right: Value

    def ir_str(self) -> str:
        return f"  {self.dest} = {self.op} {self.dest.type} {self.left}, {self.right}"


@dataclass
class ICmpInst(IRInstruction):
    dest: Value
    predicate: str  # eq, ne, slt, sgt, sle, sge
    left: Value
    right: Value

    def ir_str(self) -> str:
        return f"  {self.dest} = icmp {self.predicate} {self.left.type} {self.left}, {self.right}"


@dataclass
class AllocaInst(IRInstruction):
    dest: Value
    alloc_type: IRType

    def ir_str(self) -> str:
        return f"  {self.dest} = alloca {self.alloc_type}"


@dataclass
class LoadInst(IRInstruction):
    dest: Value
    ptr: Value
    load_type: IRType

    def ir_str(self) -> str:
        return f"  {self.dest} = load {self.load_type}, ptr {self.ptr}"


@dataclass
class StoreInst(IRInstruction):
    value: Value
    ptr: Value

    def ir_str(self) -> str:
        return f"  store {self.value.type} {self.value}, ptr {self.ptr}"


@dataclass
class RetInst(IRInstruction):
    value: Optional[Value] = None

    def ir_str(self) -> str:
        if self.value:
            return f"  ret {self.value.type} {self.value}"
        return "  ret void"


@dataclass
class BrInst(IRInstruction):
    """Unconditional branch."""
    target: str

    def ir_str(self) -> str:
        return f"  br label %{self.target}"


@dataclass
class CondBrInst(IRInstruction):
    """Conditional branch."""
    cond: Value
    true_label: str
    false_label: str

    def ir_str(self) -> str:
        return (f"  br i1 {self.cond}, label %{self.true_label}, "
                f"label %{self.false_label}")


@dataclass
class PhiInst(IRInstruction):
    dest: Value
    incoming: list[tuple[Value, str]] = field(default_factory=list)

    def ir_str(self) -> str:
        pairs = ", ".join(
            f"[ {val}, %{label} ]" for val, label in self.incoming
        )
        return f"  {self.dest} = phi {self.dest.type} {pairs}"


@dataclass
class CallInst(IRInstruction):
    dest: Optional[Value]
    func_name: str
    args: list[Value]
    ret_type: IRType

    def ir_str(self) -> str:
        args_str = ", ".join(f"{a.type} {a}" for a in self.args)
        if self.dest:
            return f"  {self.dest} = call {self.ret_type} @{self.func_name}({args_str})"
        return f"  call {self.ret_type} @{self.func_name}({args_str})"


# ---------------------------------------------------------------------------
# Basic Block
# ---------------------------------------------------------------------------

@dataclass
class BasicBlock:
    label: str
    instructions: list[IRInstruction] = field(default_factory=list)

    def ir_str(self) -> str:
        lines = [f"{self.label}:"]
        for inst in self.instructions:
            lines.append(inst.ir_str())
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Function and Module
# ---------------------------------------------------------------------------

@dataclass
class Function:
    name: str
    ret_type: IRType
    params: list[tuple[IRType, str]]
    blocks: list[BasicBlock] = field(default_factory=list)

    def ir_str(self) -> str:
        params_str = ", ".join(f"{ty} %{name}" for ty, name in self.params)
        lines = [f"define {self.ret_type} @{self.name}({params_str}) {{"]
        for block in self.blocks:
            lines.append(block.ir_str())
        lines.append("}")
        return "\n".join(lines)


@dataclass
class Module:
    name: str
    functions: list[Function] = field(default_factory=list)
    declarations: list[str] = field(default_factory=list)

    def ir_str(self) -> str:
        lines = [f"; Module: {self.name}", ""]
        for decl in self.declarations:
            lines.append(decl)
        if self.declarations:
            lines.append("")
        for func in self.functions:
            lines.append(func.ir_str())
            lines.append("")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# IR Builder
# ---------------------------------------------------------------------------

class IRBuilder:
    """
    Simplified IRBuilder (similar to LLVM's IRBuilder API).
    Provides methods to create instructions with automatic SSA naming.
    """

    def __init__(self):
        self.counter = 0
        self.current_block: Optional[BasicBlock] = None
        self.current_func: Optional[Function] = None

    def _next_name(self) -> str:
        name = f"%{self.counter}"
        self.counter += 1
        return name

    def set_insert_point(self, block: BasicBlock) -> None:
        self.current_block = block

    def create_add(self, left: Value, right: Value) -> Value:
        dest = Value(self._next_name(), left.type)
        inst = BinOpInst(dest, "add", left, right)
        self.current_block.instructions.append(inst)
        return dest

    def create_sub(self, left: Value, right: Value) -> Value:
        dest = Value(self._next_name(), left.type)
        inst = BinOpInst(dest, "sub", left, right)
        self.current_block.instructions.append(inst)
        return dest

    def create_mul(self, left: Value, right: Value) -> Value:
        dest = Value(self._next_name(), left.type)
        inst = BinOpInst(dest, "mul", left, right)
        self.current_block.instructions.append(inst)
        return dest

    def create_icmp(self, pred: str, left: Value, right: Value) -> Value:
        dest = Value(self._next_name(), i1)
        inst = ICmpInst(dest, pred, left, right)
        self.current_block.instructions.append(inst)
        return dest

    def create_alloca(self, ty: IRType) -> Value:
        dest = Value(self._next_name(), PtrType(ty))
        inst = AllocaInst(dest, ty)
        self.current_block.instructions.append(inst)
        return dest

    def create_load(self, ty: IRType, ptr: Value) -> Value:
        dest = Value(self._next_name(), ty)
        inst = LoadInst(dest, ptr, ty)
        self.current_block.instructions.append(inst)
        return dest

    def create_store(self, value: Value, ptr: Value) -> None:
        inst = StoreInst(value, ptr)
        self.current_block.instructions.append(inst)

    def create_ret(self, value: Optional[Value] = None) -> None:
        inst = RetInst(value)
        self.current_block.instructions.append(inst)

    def create_br(self, target: str) -> None:
        inst = BrInst(target)
        self.current_block.instructions.append(inst)

    def create_cond_br(self, cond: Value, true_bb: str,
                       false_bb: str) -> None:
        inst = CondBrInst(cond, true_bb, false_bb)
        self.current_block.instructions.append(inst)

    def create_phi(self, ty: IRType,
                   incoming: list[tuple[Value, str]]) -> Value:
        dest = Value(self._next_name(), ty)
        inst = PhiInst(dest, incoming)
        self.current_block.instructions.append(inst)
        return dest

    def create_call(self, func_name: str, args: list[Value],
                    ret_type: IRType = i32) -> Optional[Value]:
        dest = None
        if not isinstance(ret_type, VoidType):
            dest = Value(self._next_name(), ret_type)
        inst = CallInst(dest, func_name, args, ret_type)
        self.current_block.instructions.append(inst)
        return dest


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def build_factorial_module() -> Module:
    """
    Build LLVM IR for:
        define i32 @factorial(i32 %n) {
        entry:
            %cmp = icmp sle i32 %n, 1
            br i1 %cmp, label %base, label %recurse
        base:
            ret i32 1
        recurse:
            %n_minus_1 = sub i32 %n, 1
            %rec_result = call i32 @factorial(i32 %n_minus_1)
            %result = mul i32 %n, %rec_result
            ret i32 %result
        }
    """
    mod = Module("factorial_module")
    builder = IRBuilder()

    # Create function
    func = Function("factorial", i32, [(i32, "n")])
    n_param = Value("%n", i32)

    # Entry block
    entry = BasicBlock("entry")
    builder.set_insert_point(entry)
    cmp = builder.create_icmp("sle", n_param, const_int(1))
    builder.create_cond_br(cmp, "base", "recurse")
    func.blocks.append(entry)

    # Base case
    base = BasicBlock("base")
    builder.set_insert_point(base)
    builder.create_ret(const_int(1))
    func.blocks.append(base)

    # Recursive case
    recurse = BasicBlock("recurse")
    builder.set_insert_point(recurse)
    n_minus_1 = builder.create_sub(n_param, const_int(1))
    rec_result = builder.create_call("factorial", [n_minus_1])
    result = builder.create_mul(n_param, rec_result)
    builder.create_ret(result)
    func.blocks.append(recurse)

    mod.functions.append(func)
    return mod


def build_loop_module() -> Module:
    """
    Build LLVM IR for summing 1 to N:
        define i32 @sum_to_n(i32 %n) {
        entry:
            br label %loop
        loop:
            %i = phi i32 [ 1, %entry ], [ %i_next, %loop ]
            %sum = phi i32 [ 0, %entry ], [ %sum_next, %loop ]
            %sum_next = add i32 %sum, %i
            %i_next = add i32 %i, 1
            %cond = icmp sle i32 %i_next, %n
            br i1 %cond, label %loop, label %exit
        exit:
            ret i32 %sum_next
        }
    """
    mod = Module("loop_module")
    builder = IRBuilder()

    func = Function("sum_to_n", i32, [(i32, "n")])
    n_param = Value("%n", i32)

    # Entry
    entry = BasicBlock("entry")
    builder.set_insert_point(entry)
    builder.create_br("loop")
    func.blocks.append(entry)

    # Loop header with phi nodes
    loop = BasicBlock("loop")
    builder.set_insert_point(loop)
    i_val = builder.create_phi(i32, [
        (const_int(1), "entry"),
        (Value("%i_next_placeholder", i32), "loop")
    ])
    sum_val = builder.create_phi(i32, [
        (const_int(0), "entry"),
        (Value("%sum_next_placeholder", i32), "loop")
    ])
    sum_next = builder.create_add(sum_val, i_val)
    i_next = builder.create_add(i_val, const_int(1))
    cond = builder.create_icmp("sle", i_next, n_param)
    builder.create_cond_br(cond, "loop", "exit")
    func.blocks.append(loop)

    # Exit
    exit_bb = BasicBlock("exit")
    builder.set_insert_point(exit_bb)
    builder.create_ret(sum_next)
    func.blocks.append(exit_bb)

    mod.functions.append(func)
    return mod


def main():
    print("=" * 60)
    print("LLVM IR Builder Demo")
    print("=" * 60)

    # Demo 1: Factorial
    print("\n--- Factorial (recursive) ---")
    mod1 = build_factorial_module()
    print(mod1.ir_str())

    # Demo 2: Loop with phi nodes
    print("\n--- Sum 1..N (loop with phi) ---")
    mod2 = build_loop_module()
    print(mod2.ir_str())

    # Demo 3: Manual IR construction
    print("\n--- Manual IR: max(a, b) ---")
    mod3 = Module("max_module")
    func = Function("max", i32, [(i32, "a"), (i32, "b")])
    a_val = Value("%a", i32)
    b_val = Value("%b", i32)

    builder = IRBuilder()

    entry = BasicBlock("entry")
    builder.set_insert_point(entry)
    cmp = builder.create_icmp("sgt", a_val, b_val)
    builder.create_cond_br(cmp, "ret_a", "ret_b")
    func.blocks.append(entry)

    ret_a = BasicBlock("ret_a")
    builder.set_insert_point(ret_a)
    builder.create_ret(a_val)
    func.blocks.append(ret_a)

    ret_b = BasicBlock("ret_b")
    builder.set_insert_point(ret_b)
    builder.create_ret(b_val)
    func.blocks.append(ret_b)

    mod3.functions.append(func)
    print(mod3.ir_str())

    print("--- Key LLVM IR Concepts ---")
    print("""
  - SSA form: every value assigned once (%0, %1, %2, ...)
  - Typed instructions: operand types explicit (i32, i1, ptr)
  - Basic blocks: sequences ending with terminators (br, ret)
  - Phi nodes: merge values at control flow join points
  - IRBuilder: helper API for constructing instructions
  - Module > Function > BasicBlock > Instruction hierarchy
    """)


if __name__ == "__main__":
    main()
