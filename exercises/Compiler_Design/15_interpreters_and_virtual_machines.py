"""
Exercises for Lesson 15: Interpreters and Virtual Machines
Topic: Compiler_Design

Solutions to practice problems covering tree-walking interpreters, bytecode
compilation, stack-based VMs, register-based VMs, instruction dispatch,
and JIT compilation concepts.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Any, Tuple
import time


# === Exercise 1: Tree-Walking Interpreter ===
# Problem: Build a tree-walking interpreter for a simple expression language
# supporting arithmetic, variables, let bindings, and if-expressions.

class NodeType(Enum):
    NUMBER = auto()
    BINOP = auto()
    UNARYOP = auto()
    VAR = auto()
    LET = auto()
    IF = auto()
    COMPARE = auto()


@dataclass
class ASTNode:
    kind: NodeType
    value: Any = None
    children: List['ASTNode'] = field(default_factory=list)
    name: str = ""
    op: str = ""

    def __repr__(self):
        if self.kind == NodeType.NUMBER:
            return f"Num({self.value})"
        if self.kind == NodeType.VAR:
            return f"Var({self.name})"
        if self.kind == NodeType.BINOP:
            return f"({self.children[0]} {self.op} {self.children[1]})"
        return f"{self.kind.name}(...)"


def num(v):
    return ASTNode(NodeType.NUMBER, value=v)

def var(name):
    return ASTNode(NodeType.VAR, name=name)

def binop(op, left, right):
    return ASTNode(NodeType.BINOP, op=op, children=[left, right])

def let(name, value_expr, body_expr):
    return ASTNode(NodeType.LET, name=name, children=[value_expr, body_expr])

def if_expr(cond, then_expr, else_expr):
    return ASTNode(NodeType.IF, children=[cond, then_expr, else_expr])

def compare(op, left, right):
    return ASTNode(NodeType.COMPARE, op=op, children=[left, right])


class TreeWalkInterpreter:
    """Simple tree-walking interpreter with environment-based scoping."""

    def __init__(self):
        self.eval_count = 0

    def evaluate(self, node: ASTNode, env: Dict[str, Any] = None) -> Any:
        if env is None:
            env = {}
        self.eval_count += 1

        if node.kind == NodeType.NUMBER:
            return node.value

        elif node.kind == NodeType.VAR:
            if node.name not in env:
                raise NameError(f"Undefined variable: {node.name}")
            return env[node.name]

        elif node.kind == NodeType.BINOP:
            left = self.evaluate(node.children[0], env)
            right = self.evaluate(node.children[1], env)
            ops = {
                '+': lambda a, b: a + b,
                '-': lambda a, b: a - b,
                '*': lambda a, b: a * b,
                '/': lambda a, b: a // b if b != 0 else 0,
                '%': lambda a, b: a % b if b != 0 else 0,
            }
            return ops[node.op](left, right)

        elif node.kind == NodeType.COMPARE:
            left = self.evaluate(node.children[0], env)
            right = self.evaluate(node.children[1], env)
            ops = {
                '>': lambda a, b: a > b,
                '<': lambda a, b: a < b,
                '>=': lambda a, b: a >= b,
                '<=': lambda a, b: a <= b,
                '==': lambda a, b: a == b,
                '!=': lambda a, b: a != b,
            }
            return ops[node.op](left, right)

        elif node.kind == NodeType.LET:
            val = self.evaluate(node.children[0], env)
            new_env = dict(env)
            new_env[node.name] = val
            return self.evaluate(node.children[1], new_env)

        elif node.kind == NodeType.IF:
            cond = self.evaluate(node.children[0], env)
            if cond:
                return self.evaluate(node.children[1], env)
            else:
                return self.evaluate(node.children[2], env)

        raise ValueError(f"Unknown node kind: {node.kind}")


def exercise_1():
    """Tree-walking interpreter for expressions."""
    print("Tree-walking interpreter for a simple expression language.")
    print()

    interp = TreeWalkInterpreter()

    # Test 1: Arithmetic
    expr1 = binop('+', binop('*', num(3), num(4)), num(5))
    result1 = interp.evaluate(expr1)
    print(f"  3 * 4 + 5 = {result1}")
    assert result1 == 17

    # Test 2: Let binding
    # let x = 10 in (x + 20)
    expr2 = let("x", num(10), binop('+', var("x"), num(20)))
    result2 = interp.evaluate(expr2)
    print(f"  let x = 10 in (x + 20) = {result2}")
    assert result2 == 30

    # Test 3: Nested let
    # let x = 5 in (let y = x * 2 in (x + y))
    expr3 = let("x", num(5),
                let("y", binop('*', var("x"), num(2)),
                    binop('+', var("x"), var("y"))))
    result3 = interp.evaluate(expr3)
    print(f"  let x = 5 in (let y = x*2 in (x + y)) = {result3}")
    assert result3 == 15

    # Test 4: If expression
    # if (10 > 5) then 1 else 0
    expr4 = if_expr(compare('>', num(10), num(5)), num(1), num(0))
    result4 = interp.evaluate(expr4)
    print(f"  if (10 > 5) then 1 else 0 = {result4}")
    assert result4 == 1

    # Test 5: Combined
    # let a = 100 in (let b = 200 in (if (a < b) then (b - a) else (a - b)))
    expr5 = let("a", num(100),
                let("b", num(200),
                    if_expr(compare('<', var("a"), var("b")),
                            binop('-', var("b"), var("a")),
                            binop('-', var("a"), var("b")))))
    result5 = interp.evaluate(expr5)
    print(f"  let a=100 in let b=200 in if a<b then b-a else a-b = {result5}")
    assert result5 == 100

    print(f"\n  Total AST node evaluations: {interp.eval_count}")
    print("  All tests passed!")


# === Exercise 2: Bytecode Compiler and Stack VM ===
# Problem: Compile the same expression language to bytecode and execute
# on a stack-based virtual machine.

class OpCode(Enum):
    PUSH = auto()      # Push constant
    LOAD = auto()      # Load variable
    STORE = auto()     # Store variable
    ADD = auto()
    SUB = auto()
    MUL = auto()
    DIV = auto()
    CMP_GT = auto()
    CMP_LT = auto()
    CMP_EQ = auto()
    JMP_FALSE = auto() # Conditional jump
    JMP = auto()       # Unconditional jump
    POP = auto()       # Discard top
    HALT = auto()


@dataclass
class Instruction:
    opcode: OpCode
    operand: Any = None

    def __repr__(self):
        if self.operand is not None:
            return f"{self.opcode.name} {self.operand}"
        return self.opcode.name


class BytecodeCompiler:
    """Compile AST to stack-based bytecode."""

    def __init__(self):
        self.code: List[Instruction] = []

    def compile(self, node: ASTNode) -> List[Instruction]:
        self.code = []
        self._emit_node(node)
        self.code.append(Instruction(OpCode.HALT))
        return self.code

    def _emit_node(self, node: ASTNode):
        if node.kind == NodeType.NUMBER:
            self.code.append(Instruction(OpCode.PUSH, node.value))

        elif node.kind == NodeType.VAR:
            self.code.append(Instruction(OpCode.LOAD, node.name))

        elif node.kind == NodeType.BINOP:
            self._emit_node(node.children[0])
            self._emit_node(node.children[1])
            op_map = {'+': OpCode.ADD, '-': OpCode.SUB,
                      '*': OpCode.MUL, '/': OpCode.DIV}
            self.code.append(Instruction(op_map[node.op]))

        elif node.kind == NodeType.COMPARE:
            self._emit_node(node.children[0])
            self._emit_node(node.children[1])
            cmp_map = {'>': OpCode.CMP_GT, '<': OpCode.CMP_LT,
                       '==': OpCode.CMP_EQ}
            self.code.append(Instruction(cmp_map[node.op]))

        elif node.kind == NodeType.LET:
            self._emit_node(node.children[0])
            self.code.append(Instruction(OpCode.STORE, node.name))
            self._emit_node(node.children[1])

        elif node.kind == NodeType.IF:
            self._emit_node(node.children[0])  # condition
            jmp_false_idx = len(self.code)
            self.code.append(Instruction(OpCode.JMP_FALSE, -1))  # placeholder
            self._emit_node(node.children[1])  # then
            jmp_end_idx = len(self.code)
            self.code.append(Instruction(OpCode.JMP, -1))  # placeholder
            # Patch JMP_FALSE to point here (else branch)
            self.code[jmp_false_idx].operand = len(self.code)
            self._emit_node(node.children[2])  # else
            # Patch JMP to point past else
            self.code[jmp_end_idx].operand = len(self.code)


class StackVM:
    """Stack-based virtual machine."""

    def __init__(self):
        self.stack: List[Any] = []
        self.env: Dict[str, Any] = {}
        self.pc = 0
        self.steps = 0

    def execute(self, code: List[Instruction], trace: bool = False) -> Any:
        self.pc = 0
        self.steps = 0

        while self.pc < len(code):
            instr = code[self.pc]
            self.steps += 1

            if trace:
                stack_repr = list(self.stack[-5:]) if self.stack else []
                print(f"    [{self.pc:3d}] {str(instr):<20s} stack={stack_repr}")

            if instr.opcode == OpCode.HALT:
                break
            elif instr.opcode == OpCode.PUSH:
                self.stack.append(instr.operand)
            elif instr.opcode == OpCode.LOAD:
                self.stack.append(self.env[instr.operand])
            elif instr.opcode == OpCode.STORE:
                self.env[instr.operand] = self.stack.pop()
            elif instr.opcode == OpCode.ADD:
                b, a = self.stack.pop(), self.stack.pop()
                self.stack.append(a + b)
            elif instr.opcode == OpCode.SUB:
                b, a = self.stack.pop(), self.stack.pop()
                self.stack.append(a - b)
            elif instr.opcode == OpCode.MUL:
                b, a = self.stack.pop(), self.stack.pop()
                self.stack.append(a * b)
            elif instr.opcode == OpCode.DIV:
                b, a = self.stack.pop(), self.stack.pop()
                self.stack.append(a // b if b != 0 else 0)
            elif instr.opcode == OpCode.CMP_GT:
                b, a = self.stack.pop(), self.stack.pop()
                self.stack.append(a > b)
            elif instr.opcode == OpCode.CMP_LT:
                b, a = self.stack.pop(), self.stack.pop()
                self.stack.append(a < b)
            elif instr.opcode == OpCode.CMP_EQ:
                b, a = self.stack.pop(), self.stack.pop()
                self.stack.append(a == b)
            elif instr.opcode == OpCode.JMP_FALSE:
                cond = self.stack.pop()
                if not cond:
                    self.pc = instr.operand
                    continue
            elif instr.opcode == OpCode.JMP:
                self.pc = instr.operand
                continue
            elif instr.opcode == OpCode.POP:
                self.stack.pop()

            self.pc += 1

        return self.stack[-1] if self.stack else None


def exercise_2():
    """Bytecode compiler and stack VM."""
    print("Compiling and executing expressions on a stack VM.")
    print()

    compiler = BytecodeCompiler()
    vm = StackVM()

    # Test: let x = 5 in (let y = x*2 in (x + y))
    expr = let("x", num(5),
               let("y", binop('*', var("x"), num(2)),
                   binop('+', var("x"), var("y"))))

    code = compiler.compile(expr)

    print("  Expression: let x=5 in (let y=x*2 in (x+y))")
    print()
    print("  Bytecode:")
    for i, instr in enumerate(code):
        print(f"    [{i:3d}] {instr}")
    print()

    print("  Execution trace:")
    result = vm.execute(code, trace=True)
    print()
    print(f"  Result: {result}")
    print(f"  Steps: {vm.steps}")
    assert result == 15
    print("  Correct!")
    print()

    # Test with if-expression
    expr2 = if_expr(compare('>', num(10), num(5)),
                    binop('*', num(10), num(2)),
                    num(0))
    code2 = compiler.compile(expr2)
    vm2 = StackVM()
    result2 = vm2.execute(code2)
    print(f"  if (10 > 5) then 10*2 else 0 = {result2}")
    assert result2 == 20
    print("  Correct!")


# === Exercise 3: Register-Based VM ===
# Problem: Implement a register-based VM and compare with the stack VM.

class RegOpCode(Enum):
    LOAD_CONST = auto()  # LOAD_CONST Rd, const
    ADD = auto()         # ADD Rd, Rs1, Rs2
    SUB = auto()         # SUB Rd, Rs1, Rs2
    MUL = auto()         # MUL Rd, Rs1, Rs2
    DIV = auto()         # DIV Rd, Rs1, Rs2
    MOVE = auto()        # MOVE Rd, Rs
    CMP_GT = auto()      # CMP_GT Rd, Rs1, Rs2
    JMP_FALSE = auto()   # JMP_FALSE Rs, target
    JMP = auto()         # JMP target
    HALT = auto()


@dataclass
class RegInstruction:
    opcode: RegOpCode
    rd: int = 0       # destination register
    rs1: int = 0      # source register 1
    rs2: int = 0      # source register 2
    imm: Any = None   # immediate value

    def __repr__(self):
        if self.opcode == RegOpCode.LOAD_CONST:
            return f"LOAD_CONST R{self.rd}, {self.imm}"
        elif self.opcode in (RegOpCode.ADD, RegOpCode.SUB,
                             RegOpCode.MUL, RegOpCode.DIV, RegOpCode.CMP_GT):
            return f"{self.opcode.name} R{self.rd}, R{self.rs1}, R{self.rs2}"
        elif self.opcode == RegOpCode.MOVE:
            return f"MOVE R{self.rd}, R{self.rs1}"
        elif self.opcode == RegOpCode.JMP_FALSE:
            return f"JMP_FALSE R{self.rs1}, {self.imm}"
        elif self.opcode == RegOpCode.JMP:
            return f"JMP {self.imm}"
        return self.opcode.name


class RegisterVM:
    """Register-based virtual machine."""

    def __init__(self, num_regs=16):
        self.regs = [0] * num_regs
        self.pc = 0
        self.steps = 0

    def execute(self, code: List[RegInstruction], trace: bool = False) -> int:
        self.pc = 0
        self.steps = 0

        while self.pc < len(code):
            instr = code[self.pc]
            self.steps += 1

            if trace:
                active = {f"R{i}={self.regs[i]}" for i in range(8) if self.regs[i] != 0}
                print(f"    [{self.pc:3d}] {str(instr):<30s} {active}")

            if instr.opcode == RegOpCode.HALT:
                break
            elif instr.opcode == RegOpCode.LOAD_CONST:
                self.regs[instr.rd] = instr.imm
            elif instr.opcode == RegOpCode.ADD:
                self.regs[instr.rd] = self.regs[instr.rs1] + self.regs[instr.rs2]
            elif instr.opcode == RegOpCode.SUB:
                self.regs[instr.rd] = self.regs[instr.rs1] - self.regs[instr.rs2]
            elif instr.opcode == RegOpCode.MUL:
                self.regs[instr.rd] = self.regs[instr.rs1] * self.regs[instr.rs2]
            elif instr.opcode == RegOpCode.DIV:
                d = self.regs[instr.rs2]
                self.regs[instr.rd] = self.regs[instr.rs1] // d if d != 0 else 0
            elif instr.opcode == RegOpCode.MOVE:
                self.regs[instr.rd] = self.regs[instr.rs1]
            elif instr.opcode == RegOpCode.CMP_GT:
                self.regs[instr.rd] = int(self.regs[instr.rs1] > self.regs[instr.rs2])
            elif instr.opcode == RegOpCode.JMP_FALSE:
                if not self.regs[instr.rs1]:
                    self.pc = instr.imm
                    continue
            elif instr.opcode == RegOpCode.JMP:
                self.pc = instr.imm
                continue

            self.pc += 1

        return self.regs[0]  # R0 is the result register


def exercise_3():
    """Register-based VM vs stack-based VM comparison."""
    print("Register-based VM executing: (3 * 4) + (10 - 5)")
    print()

    # Hand-compiled register code for (3 * 4) + (10 - 5)
    reg_code = [
        RegInstruction(RegOpCode.LOAD_CONST, rd=1, imm=3),
        RegInstruction(RegOpCode.LOAD_CONST, rd=2, imm=4),
        RegInstruction(RegOpCode.MUL, rd=3, rs1=1, rs2=2),       # R3 = 3*4 = 12
        RegInstruction(RegOpCode.LOAD_CONST, rd=4, imm=10),
        RegInstruction(RegOpCode.LOAD_CONST, rd=5, imm=5),
        RegInstruction(RegOpCode.SUB, rd=6, rs1=4, rs2=5),       # R6 = 10-5 = 5
        RegInstruction(RegOpCode.ADD, rd=0, rs1=3, rs2=6),       # R0 = 12+5 = 17
        RegInstruction(RegOpCode.HALT),
    ]

    print("  Register VM bytecode:")
    for i, instr in enumerate(reg_code):
        print(f"    [{i:3d}] {instr}")
    print()

    reg_vm = RegisterVM()
    print("  Execution trace:")
    result = reg_vm.execute(reg_code, trace=True)
    print()
    print(f"  Result: R0 = {result}")
    print(f"  Steps: {reg_vm.steps}")
    assert result == 17
    print()

    # Compare with stack VM for the same expression
    compiler = BytecodeCompiler()
    stack_code = compiler.compile(binop('+', binop('*', num(3), num(4)),
                                        binop('-', num(10), num(5))))
    stack_vm = StackVM()
    stack_result = stack_vm.execute(stack_code)

    print("  Comparison:")
    print(f"    Stack VM: {len(stack_code)} instructions, {stack_vm.steps} steps")
    print(f"    Register VM: {len(reg_code)} instructions, {reg_vm.steps} steps")
    print()
    print("  Register VM advantages:")
    print("    - Fewer instructions (no push/pop overhead)")
    print("    - Operands are explicit (no stack manipulation)")
    print("    - Easier to apply optimizations (register allocation)")
    print("  Stack VM advantages:")
    print("    - Simpler instruction encoding (no register fields)")
    print("    - Simpler compiler (no register allocation needed)")
    print("    - More compact bytecode")


# === Exercise 4: Instruction Dispatch Techniques ===
# Problem: Compare switch dispatch, direct threading, and computed goto
# (simulated in Python).

def exercise_4():
    """Compare instruction dispatch techniques."""
    print("Instruction dispatch technique comparison:")
    print()

    # Simple program: sum 1 to N using a loop
    # Stack VM program:
    #   PUSH 0        ; sum = 0
    #   PUSH 1        ; i = 1
    #   LOAD i        ; top of loop
    #   PUSH N
    #   CMP_GT        ; i > N?
    #   JMP_FALSE body
    #   JMP end
    # body:
    #   LOAD sum
    #   LOAD i
    #   ADD            ; sum + i
    #   STORE sum
    #   LOAD i
    #   PUSH 1
    #   ADD            ; i + 1
    #   STORE i
    #   JMP loop
    # end:
    #   LOAD sum
    #   HALT

    N = 1000

    # Method 1: Switch dispatch (standard approach)
    print(f"  Computing sum(1..{N}) using different dispatch methods:")
    print()

    # Represent bytecode as tuples for speed
    PUSH, LOAD, STORE, ADD, CMP_LE, JMP_FALSE, JMP, HALT = range(8)
    program = [
        (PUSH, 0),       # 0: sum = 0
        (STORE, 'sum'),  # 1
        (PUSH, 1),       # 2: i = 1
        (STORE, 'i'),    # 3
        # Loop top (4):
        (LOAD, 'i'),     # 4
        (PUSH, N),       # 5
        (CMP_LE,),       # 6: i <= N?
        (JMP_FALSE, 14), # 7: if false, goto end
        (LOAD, 'sum'),   # 8
        (LOAD, 'i'),     # 9
        (ADD,),          # 10: sum + i
        (STORE, 'sum'),  # 11
        (LOAD, 'i'),     # 12
        (PUSH, 1),       # 13
        (ADD,),          # 14: i + 1  -- wait, index shifted
        (STORE, 'i'),    # 15
        (JMP, 4),        # 16: goto loop top
        (LOAD, 'sum'),   # 17: end
        (HALT,),         # 18
    ]
    # Fix indices after the JMP_FALSE
    program = [
        (PUSH, 0),       # 0
        (STORE, 'sum'),  # 1
        (PUSH, 1),       # 2
        (STORE, 'i'),    # 3
        (LOAD, 'i'),     # 4  (loop top)
        (PUSH, N),       # 5
        (CMP_LE,),       # 6
        (JMP_FALSE, 16), # 7
        (LOAD, 'sum'),   # 8
        (LOAD, 'i'),     # 9
        (ADD,),          # 10
        (STORE, 'sum'),  # 11
        (LOAD, 'i'),     # 12
        (PUSH, 1),       # 13
        (ADD,),          # 14
        (STORE, 'i'),    # 15
        (JMP, 4),        # 16 -- wait this is where JMP_FALSE goes
    ]
    # Let me be more careful:
    program = [
        (PUSH, 0),       # 0
        (STORE, 'sum'),  # 1
        (PUSH, 1),       # 2
        (STORE, 'i'),    # 3
        # loop_top = 4
        (LOAD, 'i'),     # 4
        (PUSH, N),       # 5
        (CMP_LE,),       # 6
        (JMP_FALSE, 17), # 7 -> end (index 17)
        (LOAD, 'sum'),   # 8
        (LOAD, 'i'),     # 9
        (ADD,),          # 10
        (STORE, 'sum'),  # 11
        (LOAD, 'i'),     # 12
        (PUSH, 1),       # 13
        (ADD,),          # 14
        (STORE, 'i'),    # 15
        (JMP, 4),        # 16 -> loop_top
        (LOAD, 'sum'),   # 17 (end)
        (HALT,),         # 18
    ]

    # Switch dispatch
    def run_switch(prog):
        stack = []
        env = {}
        pc = 0
        dispatch_count = 0
        while pc < len(prog):
            op = prog[pc]
            dispatch_count += 1
            if op[0] == PUSH:
                stack.append(op[1])
            elif op[0] == LOAD:
                stack.append(env[op[1]])
            elif op[0] == STORE:
                env[op[1]] = stack.pop()
            elif op[0] == ADD:
                b, a = stack.pop(), stack.pop()
                stack.append(a + b)
            elif op[0] == CMP_LE:
                b, a = stack.pop(), stack.pop()
                stack.append(a <= b)
            elif op[0] == JMP_FALSE:
                if not stack.pop():
                    pc = op[1]
                    continue
            elif op[0] == JMP:
                pc = op[1]
                continue
            elif op[0] == HALT:
                break
            pc += 1
        return stack[-1] if stack else env.get('sum', 0), dispatch_count

    t0 = time.perf_counter()
    result, dispatches = run_switch(program)
    t1 = time.perf_counter()

    expected = N * (N + 1) // 2
    print(f"  Switch dispatch:")
    print(f"    Result: {result} (expected {expected}, correct={result == expected})")
    print(f"    Dispatches: {dispatches}")
    print(f"    Time: {(t1 - t0) * 1000:.2f}ms")
    print()

    # Method 2: Token-threaded (simulated with function table)
    def run_threaded(prog):
        stack = []
        env = {}
        pc = [0]  # mutable for closures
        dispatch_count = [0]

        def do_push():
            stack.append(prog[pc[0]][1])
            pc[0] += 1
        def do_load():
            stack.append(env[prog[pc[0]][1]])
            pc[0] += 1
        def do_store():
            env[prog[pc[0]][1]] = stack.pop()
            pc[0] += 1
        def do_add():
            b, a = stack.pop(), stack.pop()
            stack.append(a + b)
            pc[0] += 1
        def do_cmp_le():
            b, a = stack.pop(), stack.pop()
            stack.append(a <= b)
            pc[0] += 1
        def do_jmp_false():
            if not stack.pop():
                pc[0] = prog[pc[0]][1]
            else:
                pc[0] += 1
        def do_jmp():
            pc[0] = prog[pc[0]][1]
        def do_halt():
            pc[0] = len(prog)

        dispatch_table = [do_push, do_load, do_store, do_add,
                          do_cmp_le, do_jmp_false, do_jmp, do_halt]

        while pc[0] < len(prog):
            dispatch_count[0] += 1
            dispatch_table[prog[pc[0]][0]]()

        return stack[-1] if stack else env.get('sum', 0), dispatch_count[0]

    t0 = time.perf_counter()
    result2, dispatches2 = run_threaded(program)
    t1 = time.perf_counter()

    print(f"  Token-threaded dispatch (function table):")
    print(f"    Result: {result2} (correct={result2 == expected})")
    print(f"    Dispatches: {dispatches2}")
    print(f"    Time: {(t1 - t0) * 1000:.2f}ms")
    print()

    print("  Dispatch technique summary:")
    print("    Switch dispatch: Simple, but indirect branch misprediction")
    print("    Direct threading: Each handler jumps to the next (computed goto)")
    print("    Token threading: Index into function pointer table")
    print("    Subroutine threading: CALL/RET for each handler")
    print()
    print("  In native C, direct threading (computed goto) is typically 20-30%")
    print("  faster than switch dispatch due to better branch prediction.")


# === Exercise 5: Simple JIT Concept ===
# Problem: Demonstrate the concept of JIT compilation by "specializing"
# an interpreted function for specific argument types.

def exercise_5():
    """JIT compilation concept: specialization and inline caching."""
    print("JIT compilation concept: function specialization")
    print()

    # Generic (interpreted) version
    def generic_dot_product(a, b):
        """Generic version -- must handle any sequence type."""
        result = 0
        for i in range(len(a)):
            result += a[i] * b[i]
        return result

    # "JIT-compiled" specialized version for list-of-int
    def specialized_dot_product_int(a, b, n):
        """Specialized: assumes list of int, known length."""
        result = 0
        # Unrolled by 4 (a JIT might do this for hot loops)
        main_end = (n // 4) * 4
        i = 0
        while i < main_end:
            result += a[i] * b[i] + a[i+1] * b[i+1] + \
                      a[i+2] * b[i+2] + a[i+3] * b[i+3]
            i += 4
        while i < n:
            result += a[i] * b[i]
            i += 1
        return result

    N = 10000
    a = list(range(N))
    b = list(range(N, 2 * N))

    # Run generic
    t0 = time.perf_counter()
    r1 = generic_dot_product(a, b)
    t1 = time.perf_counter()
    generic_time = (t1 - t0) * 1000

    # Run specialized
    t0 = time.perf_counter()
    r2 = specialized_dot_product_int(a, b, N)
    t1 = time.perf_counter()
    specialized_time = (t1 - t0) * 1000

    print(f"  Dot product of two {N}-element vectors:")
    print(f"    Generic result:     {r1}")
    print(f"    Specialized result: {r2}")
    print(f"    Results match: {r1 == r2}")
    print()
    print(f"    Generic time:     {generic_time:.2f}ms")
    print(f"    Specialized time: {specialized_time:.2f}ms")
    print()

    # Inline caching concept
    print("  Inline caching concept:")
    print()

    class InlineCache:
        """Simulates a monomorphic inline cache for method dispatch."""

        def __init__(self):
            self.cached_type = None
            self.cached_method = None
            self.hits = 0
            self.misses = 0

        def lookup(self, obj, method_name):
            obj_type = type(obj)
            if obj_type == self.cached_type:
                self.hits += 1
                return self.cached_method
            else:
                self.misses += 1
                method = getattr(obj, method_name)
                self.cached_type = obj_type
                self.cached_method = method
                return method

    cache = InlineCache()

    # Simulate calling .upper() on many strings (monomorphic)
    strings = ["hello", "world", "foo", "bar"] * 100
    for s in strings:
        method = cache.lookup(s, "upper")
        method()

    print(f"    Monomorphic call site (.upper() on str):")
    print(f"      Hits: {cache.hits}, Misses: {cache.misses}")
    print(f"      Hit rate: {cache.hits / (cache.hits + cache.misses) * 100:.1f}%")
    print()

    # Polymorphic: mix of types
    cache2 = InlineCache()
    items = [1, "hello", 1.5, 2, "world", 2.5] * 50
    for item in items:
        cache2.lookup(item, "__str__")

    print(f"    Polymorphic call site (__str__ on int/str/float):")
    print(f"      Hits: {cache2.hits}, Misses: {cache2.misses}")
    print(f"      Hit rate: {cache2.hits / (cache2.hits + cache2.misses) * 100:.1f}%")
    print()
    print("  Key JIT ideas:")
    print("    1. Profile-guided specialization (compile for observed types)")
    print("    2. Inline caching (cache method lookups at call sites)")
    print("    3. On-stack replacement (OSR: switch from interp to JIT mid-loop)")
    print("    4. Deoptimization (fall back to interpreter if assumptions break)")


# === Exercise 6: VM Design Comparison ===
# Problem: Compare the design choices of real virtual machines.

def exercise_6():
    """Compare real VM architectures."""
    print("Real VM Architecture Comparison")
    print("=" * 70)
    print()

    vms = [
        {
            "name": "JVM (HotSpot)",
            "type": "Stack-based",
            "dispatch": "Template interpreter + JIT (C1/C2)",
            "gc": "G1, ZGC, Shenandoah",
            "jit": "Method JIT (C1 = quick, C2 = optimizing)",
            "key_feature": "Tiered compilation, mature ecosystem",
        },
        {
            "name": "CPython",
            "type": "Stack-based",
            "dispatch": "Switch dispatch (ceval.c), specializing adaptive",
            "gc": "Reference counting + generational cycle detector",
            "jit": "None (3.13+ experimental copy-and-patch JIT)",
            "key_feature": "GIL, simple implementation, C extension API",
        },
        {
            "name": "V8 (JavaScript)",
            "type": "Register-based (Ignition bytecode)",
            "dispatch": "Bytecode interpreter + TurboFan JIT",
            "gc": "Generational, incremental, concurrent (Orinoco)",
            "jit": "Optimizing JIT with deopt, speculative optimization",
            "key_feature": "Hidden classes, inline caching, OSR",
        },
        {
            "name": "Lua 5",
            "type": "Register-based",
            "dispatch": "Switch dispatch",
            "gc": "Incremental mark-sweep",
            "jit": "None (LuaJIT is separate: tracing JIT)",
            "key_feature": "Tiny footprint (~200KB), embeddable",
        },
        {
            "name": "BEAM (Erlang)",
            "type": "Register-based",
            "dispatch": "Threaded code / JIT (OTP 24+)",
            "gc": "Per-process generational copying",
            "jit": "Added in OTP 24",
            "key_feature": "Per-process heap, no stop-the-world GC",
        },
    ]

    for vm in vms:
        print(f"  {vm['name']}")
        print(f"    Type:        {vm['type']}")
        print(f"    Dispatch:    {vm['dispatch']}")
        print(f"    GC:          {vm['gc']}")
        print(f"    JIT:         {vm['jit']}")
        print(f"    Key feature: {vm['key_feature']}")
        print()

    print("Stack vs Register VM trade-offs:")
    print()
    print("  Stack-based (JVM, CPython):")
    print("    + Simpler compiler (no register allocation)")
    print("    + More compact bytecode (implicit operands)")
    print("    + Easy to verify (type checking is simpler)")
    print("    - More instructions needed (push/pop overhead)")
    print("    - Operand stack traffic is redundant work")
    print()
    print("  Register-based (Lua, V8/Ignition, Dalvik):")
    print("    + Fewer instructions (explicit operands)")
    print("    + Better maps to real hardware registers")
    print("    + Research shows ~25% fewer bytecode dispatches")
    print("    - Larger instruction encoding")
    print("    - Compiler must do register allocation")


if __name__ == "__main__":
    print("=" * 60)
    print("=== Exercise 1: Tree-Walking Interpreter ===")
    print("=" * 60)
    exercise_1()

    print("\n" + "=" * 60)
    print("=== Exercise 2: Bytecode Compiler and Stack VM ===")
    print("=" * 60)
    exercise_2()

    print("\n" + "=" * 60)
    print("=== Exercise 3: Register-Based VM ===")
    print("=" * 60)
    exercise_3()

    print("\n" + "=" * 60)
    print("=== Exercise 4: Instruction Dispatch Techniques ===")
    print("=" * 60)
    exercise_4()

    print("\n" + "=" * 60)
    print("=== Exercise 5: Simple JIT Concept ===")
    print("=" * 60)
    exercise_5()

    print("\n" + "=" * 60)
    print("=== Exercise 6: VM Design Comparison ===")
    print("=" * 60)
    exercise_6()

    print("\nAll exercises completed!")
