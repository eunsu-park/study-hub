# Interpreters and Virtual Machines

**Previous**: [14. Garbage Collection](./14_Garbage_Collection.md) | **Next**: [16. Modern Compiler Infrastructure](./16_Modern_Compiler_Infrastructure.md)

---

Not all language implementations compile to native machine code. Many of the most widely used languages -- Python, Java, JavaScript, Ruby, Erlang, Lua -- rely on interpreters or virtual machines (or a combination of both) to execute programs. Understanding how interpreters and VMs work is essential for language implementers and for any programmer who wants to understand what happens when their code runs.

This lesson covers the spectrum of execution strategies, from simple tree-walking interpreters to sophisticated JIT-compiling virtual machines. We will build a working bytecode compiler and stack-based VM for a simple language in Python, providing a concrete foundation for understanding how production VMs operate.

**Difficulty**: ⭐⭐⭐⭐

**Prerequisites**: [07. Abstract Syntax Trees](./07_Abstract_Syntax_Trees.md), [09. Intermediate Representations](./09_Intermediate_Representations.md), [11. Code Generation](./11_Code_Generation.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Compare interpreters and compilers along multiple axes (speed, portability, development cycle)
2. Implement a tree-walking interpreter for a simple language
3. Design a bytecode instruction set and implement a bytecode compiler
4. Build a stack-based virtual machine from scratch
5. Understand register-based VM design and its advantages
6. Explain instruction dispatch techniques and their performance impact
7. Describe JIT compilation strategies (method JIT, tracing JIT)
8. Explain runtime optimization techniques (inline caching, type specialization)
9. Analyze the design of real VMs (JVM, CPython, V8, BEAM)
10. Understand metacircular interpreters

---

## Table of Contents

1. [Interpreters vs Compilers](#1-interpreters-vs-compilers)
2. [Tree-Walking Interpreters](#2-tree-walking-interpreters)
3. [Bytecode and Bytecode Compilation](#3-bytecode-and-bytecode-compilation)
4. [Stack-Based Virtual Machines](#4-stack-based-virtual-machines)
5. [Register-Based Virtual Machines](#5-register-based-virtual-machines)
6. [Instruction Dispatch Techniques](#6-instruction-dispatch-techniques)
7. [A Complete Bytecode Compiler and VM](#7-a-complete-bytecode-compiler-and-vm)
8. [JIT Compilation](#8-jit-compilation)
9. [Runtime Optimization Techniques](#9-runtime-optimization-techniques)
10. [Real Virtual Machines](#10-real-virtual-machines)
11. [Metacircular Interpreters](#11-metacircular-interpreters)
12. [Summary](#12-summary)
13. [Exercises](#13-exercises)
14. [References](#14-references)

---

## 1. Interpreters vs Compilers

### 1.1 The Execution Spectrum

Language implementations exist on a spectrum from pure interpretation to pure compilation:

```
Pure Interpreter                                    Native Compiler
     |                                                    |
     v                                                    v
┌──────────┬──────────┬──────────┬──────────┬──────────────┐
│  Tree    │ Bytecode │ Bytecode │  AOT     │    Static    │
│  Walking │ Interp.  │ + JIT    │ Compile  │   Compiler   │
│          │          │          │          │              │
│  Ruby 1  │ CPython  │  JVM     │  GraalVM │   GCC/LLVM  │
│  (old)   │  Lua     │  V8      │  Native  │   Rust      │
│  Bash    │          │  PyPy    │  Image   │   Go        │
└──────────┴──────────┴──────────┴──────────┴──────────────┘
   Slow                                            Fast
   Portable                                        Platform-specific
   Quick startup                                   Slow startup
   Easy to implement                               Complex implementation
```

### 1.2 Trade-offs

| Aspect | Interpreter | Compiler |
|--------|------------|----------|
| **Execution speed** | Slow (10-100x slower) | Fast (near hardware speed) |
| **Startup time** | Fast (no compilation phase) | Slow (must compile first) |
| **Memory usage** | Lower (no generated code) | Higher (generated code + data) |
| **Portability** | High (VM abstracts hardware) | Low (target-specific) |
| **Error messages** | Better (has source info) | Often cryptic |
| **Debugging** | Easier (inspect live state) | Harder (optimized away) |
| **Development cycle** | Fast (edit-run) | Slow (edit-compile-run) |
| **Dynamic features** | Easy (`eval`, metaprogramming) | Hard or impossible |
| **Optimization** | Limited | Extensive |

### 1.3 Hybrid Approaches

Most modern systems use hybrid approaches:

- **Java**: Compile to bytecode (AOT), then JIT compile hot methods to native code at runtime.
- **JavaScript (V8)**: Parse to AST, compile to bytecode, JIT compile hot functions with TurboFan.
- **Python (PyPy)**: Interpret bytecode, trace hot loops, JIT compile traces.
- **.NET**: Compile to CIL bytecode, JIT compile to native code at load time or lazily.

---

## 2. Tree-Walking Interpreters

### 2.1 The Simplest Interpreter

A tree-walking interpreter executes a program by traversing its AST. Each node type has an associated evaluation rule.

```python
from dataclasses import dataclass
from typing import Any, Union


# AST node definitions
@dataclass
class NumberLit:
    value: float

@dataclass
class StringLit:
    value: str

@dataclass
class BoolLit:
    value: bool

@dataclass
class Identifier:
    name: str

@dataclass
class BinOp:
    op: str
    left: Any
    right: Any

@dataclass
class UnaryOp:
    op: str
    operand: Any

@dataclass
class Assign:
    name: str
    value: Any

@dataclass
class If:
    condition: Any
    then_body: list
    else_body: list

@dataclass
class While:
    condition: Any
    body: list

@dataclass
class FuncDef:
    name: str
    params: list
    body: list

@dataclass
class FuncCall:
    name: str
    args: list

@dataclass
class Return:
    value: Any

@dataclass
class Print:
    value: Any


class ReturnException(Exception):
    """Used to implement return from functions."""
    def __init__(self, value):
        self.value = value


class Environment:
    """Variable scope with lexical scoping."""

    def __init__(self, parent=None):
        self.bindings = {}
        self.parent = parent

    def get(self, name):
        if name in self.bindings:
            return self.bindings[name]
        if self.parent:
            return self.parent.get(name)
        raise NameError(f"Undefined variable: {name}")

    def set(self, name, value):
        self.bindings[name] = value

    def update(self, name, value):
        """Update existing binding (search up scope chain)."""
        if name in self.bindings:
            self.bindings[name] = value
            return
        if self.parent:
            self.parent.update(name, value)
            return
        # If not found anywhere, create in current scope
        self.bindings[name] = value


class TreeWalkInterpreter:
    """
    A tree-walking interpreter that directly executes AST nodes.
    """

    def __init__(self):
        self.global_env = Environment()
        self.output = []  # Captured output for testing

    def interpret(self, program: list):
        """Interpret a list of statements."""
        result = None
        for stmt in program:
            result = self.execute(stmt, self.global_env)
        return result

    def execute(self, node, env):
        """Execute a single AST node."""
        method_name = f"exec_{type(node).__name__}"
        method = getattr(self, method_name, None)
        if method is None:
            raise RuntimeError(f"Unknown node type: {type(node).__name__}")
        return method(node, env)

    def exec_NumberLit(self, node, env):
        return node.value

    def exec_StringLit(self, node, env):
        return node.value

    def exec_BoolLit(self, node, env):
        return node.value

    def exec_Identifier(self, node, env):
        return env.get(node.name)

    def exec_BinOp(self, node, env):
        left = self.execute(node.left, env)
        right = self.execute(node.right, env)

        ops = {
            '+': lambda a, b: a + b,
            '-': lambda a, b: a - b,
            '*': lambda a, b: a * b,
            '/': lambda a, b: a / b,
            '%': lambda a, b: a % b,
            '==': lambda a, b: a == b,
            '!=': lambda a, b: a != b,
            '<': lambda a, b: a < b,
            '>': lambda a, b: a > b,
            '<=': lambda a, b: a <= b,
            '>=': lambda a, b: a >= b,
            'and': lambda a, b: a and b,
            'or': lambda a, b: a or b,
        }

        if node.op not in ops:
            raise RuntimeError(f"Unknown operator: {node.op}")
        return ops[node.op](left, right)

    def exec_UnaryOp(self, node, env):
        operand = self.execute(node.operand, env)
        if node.op == '-':
            return -operand
        if node.op == 'not':
            return not operand
        raise RuntimeError(f"Unknown unary operator: {node.op}")

    def exec_Assign(self, node, env):
        value = self.execute(node.value, env)
        env.set(node.name, value)
        return value

    def exec_If(self, node, env):
        condition = self.execute(node.condition, env)
        if condition:
            for stmt in node.then_body:
                self.execute(stmt, env)
        elif node.else_body:
            for stmt in node.else_body:
                self.execute(stmt, env)

    def exec_While(self, node, env):
        while self.execute(node.condition, env):
            for stmt in node.body:
                self.execute(stmt, env)

    def exec_FuncDef(self, node, env):
        # Store function as closure (captures defining environment)
        env.set(node.name, ('function', node.params, node.body, env))

    def exec_FuncCall(self, node, env):
        func = env.get(node.name)
        if not isinstance(func, tuple) or func[0] != 'function':
            raise RuntimeError(f"{node.name} is not a function")

        _, params, body, closure_env = func
        args = [self.execute(arg, env) for arg in node.args]

        if len(args) != len(params):
            raise RuntimeError(
                f"{node.name} expects {len(params)} args, got {len(args)}")

        # Create new scope for function body
        func_env = Environment(parent=closure_env)
        for param, arg in zip(params, args):
            func_env.set(param, arg)

        # Execute function body
        try:
            for stmt in body:
                self.execute(stmt, func_env)
        except ReturnException as ret:
            return ret.value
        return None

    def exec_Return(self, node, env):
        value = self.execute(node.value, env) if node.value else None
        raise ReturnException(value)

    def exec_Print(self, node, env):
        value = self.execute(node.value, env)
        self.output.append(str(value))
        print(value)


def demonstrate_tree_walker():
    """Demonstrate the tree-walking interpreter."""
    print("=== Tree-Walking Interpreter Demo ===\n")

    interp = TreeWalkInterpreter()

    # Program: compute factorial
    program = [
        FuncDef('factorial', ['n'], [
            If(
                BinOp('<=', Identifier('n'), NumberLit(1)),
                [Return(NumberLit(1))],
                [Return(BinOp('*', Identifier('n'),
                              FuncCall('factorial',
                                       [BinOp('-', Identifier('n'),
                                               NumberLit(1))])))]
            )
        ]),
        Assign('result', FuncCall('factorial', [NumberLit(10)])),
        Print(Identifier('result')),
    ]

    interp.interpret(program)

    # Program: Fibonacci
    program2 = [
        FuncDef('fib', ['n'], [
            If(
                BinOp('<', Identifier('n'), NumberLit(2)),
                [Return(Identifier('n'))],
                [Return(BinOp('+',
                              FuncCall('fib', [BinOp('-', Identifier('n'),
                                                     NumberLit(1))]),
                              FuncCall('fib', [BinOp('-', Identifier('n'),
                                                     NumberLit(2))])))]
            )
        ]),
        # Print first 10 Fibonacci numbers
        Assign('i', NumberLit(0)),
        While(BinOp('<', Identifier('i'), NumberLit(10)), [
            Print(FuncCall('fib', [Identifier('i')])),
            Assign('i', BinOp('+', Identifier('i'), NumberLit(1))),
        ]),
    ]

    interp2 = TreeWalkInterpreter()
    interp2.interpret(program2)

demonstrate_tree_walker()
```

### 2.2 Advantages and Disadvantages

**Advantages**:
- Simple to implement (a few hundred lines)
- Direct access to AST (great for debugging, error messages)
- Easy to add features (just add a new `exec_` method)
- Natural for languages with complex semantics

**Disadvantages**:
- **Slow**: Each node requires a virtual dispatch (method lookup)
- **Deep recursion**: Deeply nested expressions cause stack overflow
- **No optimization**: Every expression is re-evaluated every time
- **Cache unfriendly**: AST nodes scattered in memory

Tree-walking interpreters are typically 50-200x slower than compiled code.

---

## 3. Bytecode and Bytecode Compilation

### 3.1 What is Bytecode?

**Bytecode** is a compact binary representation of a program, designed to be executed by a virtual machine rather than real hardware. It sits between source code and machine code:

```
Source Code       AST          Bytecode         Machine Code
  x = a + b  -->  Assign  -->  LOAD_VAR a   -->  mov eax, [rbp-8]
                   / \          LOAD_VAR b       add eax, [rbp-16]
                  x   +         ADD              mov [rbp-24], eax
                     / \        STORE_VAR x
                    a   b
```

### 3.2 Bytecode Design Principles

Good bytecode design balances several goals:

1. **Compactness**: Fewer bytes = faster loading, less memory, better cache use
2. **Simplicity**: Simple instructions are easy to decode and execute
3. **Completeness**: Must express all language features
4. **Performance**: Common operations should be efficient
5. **Verifiability**: Should be possible to validate before execution

### 3.3 Instruction Encoding

```
Fixed-width (e.g., 32-bit):
┌────────┬────────┬────────┬────────┐
│ opcode │ arg1   │ arg2   │ arg3   │
│ 8 bits │ 8 bits │ 8 bits │ 8 bits │
└────────┴────────┴────────┴────────┘
  Simple to decode, wastes space for simple instructions

Variable-width (e.g., CPython):
┌────────┐         ┌────────┬────────┐
│ opcode │    or   │ opcode │  arg   │
│ 8 bits │         │ 8 bits │ 8 bits │
└────────┘         └────────┴────────┘
  Compact, but harder to decode
```

### 3.4 Defining an Instruction Set

```python
from enum import IntEnum, auto


class OpCode(IntEnum):
    """Bytecode instruction opcodes for our simple VM."""

    # Stack operations
    CONST = 0        # Push constant: CONST <index>
    POP = 1          # Pop top of stack

    # Arithmetic
    ADD = 2          # Pop two, push sum
    SUB = 3          # Pop two, push difference
    MUL = 4          # Pop two, push product
    DIV = 5          # Pop two, push quotient
    MOD = 6          # Pop two, push remainder
    NEG = 7          # Negate top of stack

    # Comparison
    EQ = 8           # Equal
    NE = 9           # Not equal
    LT = 10          # Less than
    GT = 11          # Greater than
    LE = 12          # Less or equal
    GE = 13          # Greater or equal

    # Logical
    NOT = 14         # Logical not

    # Variables
    LOAD = 15        # Push variable value: LOAD <slot>
    STORE = 16       # Pop and store to variable: STORE <slot>
    LOAD_GLOBAL = 17 # Push global value: LOAD_GLOBAL <index>
    STORE_GLOBAL = 18# Store global: STORE_GLOBAL <index>

    # Control flow
    JUMP = 19        # Unconditional jump: JUMP <offset>
    JUMP_IF_FALSE = 20  # Conditional jump: JUMP_IF_FALSE <offset>
    JUMP_IF_TRUE = 21   # Conditional jump: JUMP_IF_TRUE <offset>

    # Functions
    CALL = 22        # Call function: CALL <num_args>
    RETURN = 23      # Return from function

    # I/O
    PRINT = 24       # Print top of stack

    # Special
    HALT = 25        # Stop execution

    # Constants
    TRUE = 26        # Push True
    FALSE = 27       # Push False
    NONE = 28        # Push None


# Instruction metadata
INSTRUCTION_INFO = {
    OpCode.CONST: ('CONST', 1),          # 1 argument (constant index)
    OpCode.POP: ('POP', 0),
    OpCode.ADD: ('ADD', 0),
    OpCode.SUB: ('SUB', 0),
    OpCode.MUL: ('MUL', 0),
    OpCode.DIV: ('DIV', 0),
    OpCode.MOD: ('MOD', 0),
    OpCode.NEG: ('NEG', 0),
    OpCode.EQ: ('EQ', 0),
    OpCode.NE: ('NE', 0),
    OpCode.LT: ('LT', 0),
    OpCode.GT: ('GT', 0),
    OpCode.LE: ('LE', 0),
    OpCode.GE: ('GE', 0),
    OpCode.NOT: ('NOT', 0),
    OpCode.LOAD: ('LOAD', 1),
    OpCode.STORE: ('STORE', 1),
    OpCode.LOAD_GLOBAL: ('LOAD_GLOBAL', 1),
    OpCode.STORE_GLOBAL: ('STORE_GLOBAL', 1),
    OpCode.JUMP: ('JUMP', 1),
    OpCode.JUMP_IF_FALSE: ('JUMP_IF_FALSE', 1),
    OpCode.JUMP_IF_TRUE: ('JUMP_IF_TRUE', 1),
    OpCode.CALL: ('CALL', 1),
    OpCode.RETURN: ('RETURN', 0),
    OpCode.PRINT: ('PRINT', 0),
    OpCode.HALT: ('HALT', 0),
    OpCode.TRUE: ('TRUE', 0),
    OpCode.FALSE: ('FALSE', 0),
    OpCode.NONE: ('NONE', 0),
}
```

---

## 4. Stack-Based Virtual Machines

### 4.1 How Stack VMs Work

A stack-based VM uses an operand stack for all computations. Operands are pushed onto the stack, operations pop their arguments and push results.

```
Computing x = a + b * c:

Instructions:          Stack (grows right →)
                       []
LOAD a                 [3]
LOAD b                 [3, 4]
LOAD c                 [3, 4, 5]
MUL                    [3, 20]      (4 * 5 = 20)
ADD                    [23]         (3 + 20 = 23)
STORE x                []           (x = 23)
```

### 4.2 Advantages of Stack VMs

1. **Simple code generation**: No register allocation needed
2. **Compact bytecode**: Instructions don't need to specify operand locations
3. **Easy to implement**: The stack provides a natural evaluation order
4. **Portable**: No assumptions about number of hardware registers

### 4.3 Disadvantages

1. **More instructions**: `LOAD a; LOAD b; ADD` vs `ADD r1, r2, r3`
2. **Memory traffic**: Every operation reads/writes the stack (memory, not registers)
3. **Harder to optimize**: Stack positions are implicit, making analysis difficult

### 4.4 Basic Stack VM Implementation

```python
class CodeObject:
    """
    Compiled code object (like Python's code object).
    Contains bytecode, constants, and metadata.
    """

    def __init__(self, name='<module>'):
        self.name = name
        self.bytecode = []       # List of (opcode, arg) tuples
        self.constants = []      # Constant pool
        self.local_names = []    # Local variable names
        self.num_locals = 0

    def emit(self, opcode, arg=None):
        """Emit a bytecode instruction."""
        self.bytecode.append((opcode, arg))
        return len(self.bytecode) - 1  # Return instruction index

    def add_constant(self, value):
        """Add a constant to the pool, return its index."""
        if value in self.constants:
            return self.constants.index(value)
        self.constants.append(value)
        return len(self.constants) - 1

    def add_local(self, name):
        """Add a local variable, return its slot index."""
        if name in self.local_names:
            return self.local_names.index(name)
        self.local_names.append(name)
        self.num_locals += 1
        return len(self.local_names) - 1

    def disassemble(self):
        """Print human-readable bytecode."""
        print(f"\n=== Disassembly of {self.name} ===")
        print(f"Constants: {self.constants}")
        print(f"Locals: {self.local_names}")
        print(f"Instructions:")

        for i, (opcode, arg) in enumerate(self.bytecode):
            name, num_args = INSTRUCTION_INFO.get(opcode, ('???', 0))
            if arg is not None:
                # Add human-readable annotation
                if opcode == OpCode.CONST:
                    extra = f" ({self.constants[arg]})"
                elif opcode in (OpCode.LOAD, OpCode.STORE):
                    extra = f" ({self.local_names[arg]})" if arg < len(self.local_names) else ""
                elif opcode in (OpCode.JUMP, OpCode.JUMP_IF_FALSE, OpCode.JUMP_IF_TRUE):
                    extra = f" (-> {arg})"
                else:
                    extra = ""
                print(f"  {i:4d}  {name:<20s} {arg}{extra}")
            else:
                print(f"  {i:4d}  {name}")
```

---

## 5. Register-Based Virtual Machines

### 5.1 Register VM Design

Register-based VMs use virtual registers instead of a stack. Instructions explicitly name source and destination registers.

```
Computing x = a + b * c:

Stack VM:              Register VM:
  LOAD a                 MUL  r2, r1, r2    (r2 = b * c)
  LOAD b                 ADD  r0, r0, r2    (r0 = a + r2)
  LOAD c
  MUL
  ADD
  STORE x

  6 instructions         2 instructions
```

### 5.2 Register VM Example

```python
class RegisterVM:
    """
    Simple register-based VM.

    Instructions: (opcode, dest, src1, src2)
    """

    def __init__(self, num_registers=256):
        self.registers = [None] * num_registers
        self.pc = 0

    def execute(self, instructions, constants):
        """Execute register-based instructions."""
        self.pc = 0

        while self.pc < len(instructions):
            instr = instructions[self.pc]
            opcode = instr[0]

            if opcode == 'LOADK':     # Load constant: LOADK dest, const_idx
                dest, const_idx = instr[1], instr[2]
                self.registers[dest] = constants[const_idx]

            elif opcode == 'MOVE':     # Move: MOVE dest, src
                dest, src = instr[1], instr[2]
                self.registers[dest] = self.registers[src]

            elif opcode == 'ADD':      # Add: ADD dest, src1, src2
                dest, src1, src2 = instr[1], instr[2], instr[3]
                self.registers[dest] = self.registers[src1] + self.registers[src2]

            elif opcode == 'MUL':
                dest, src1, src2 = instr[1], instr[2], instr[3]
                self.registers[dest] = self.registers[src1] * self.registers[src2]

            elif opcode == 'SUB':
                dest, src1, src2 = instr[1], instr[2], instr[3]
                self.registers[dest] = self.registers[src1] - self.registers[src2]

            elif opcode == 'LT':       # Less than: LT dest, src1, src2
                dest, src1, src2 = instr[1], instr[2], instr[3]
                self.registers[dest] = self.registers[src1] < self.registers[src2]

            elif opcode == 'JMP':      # Jump: JMP offset
                self.pc += instr[1]
                continue

            elif opcode == 'JMPF':     # Jump if false: JMPF test, offset
                if not self.registers[instr[1]]:
                    self.pc += instr[2]
                    continue

            elif opcode == 'PRINT':    # Print: PRINT src
                print(f"  Output: {self.registers[instr[1]]}")

            elif opcode == 'HALT':
                break

            self.pc += 1

        return self.registers


def demonstrate_register_vm():
    """Demonstrate register-based VM computing sum 1..10."""
    print("=== Register VM Demo ===")
    print("Computing sum of 1 to 10:\n")

    vm = RegisterVM()

    # sum = 0; i = 1; while i <= 10: sum += i; i += 1
    constants = [0, 1, 10]  # 0: zero, 1: one, 2: ten

    instructions = [
        ('LOADK', 0, 0),        # r0 = 0 (sum)
        ('LOADK', 1, 1),        # r1 = 1 (i)
        ('LOADK', 2, 2),        # r2 = 10 (limit)
        ('LOADK', 3, 1),        # r3 = 1 (increment)
        # Loop start (pc=4):
        ('LT', 4, 2, 1),       # r4 = (10 < i), i.e., i > 10
        ('JMPF', 4, 1),        # if not (i > 10), skip next
        ('JMP', 4),             # jump to end (pc=10)
        ('ADD', 0, 0, 1),       # sum += i
        ('ADD', 1, 1, 3),       # i += 1
        ('JMP', -5),            # jump back to loop start (pc=4)
        # End (pc=10):
        ('PRINT', 0),           # print sum
        ('HALT',),
    ]

    vm.execute(instructions, constants)

demonstrate_register_vm()
```

### 5.3 Stack vs Register Comparison

| Aspect | Stack-Based | Register-Based |
|--------|-------------|----------------|
| **Instruction count** | More (implicit operands) | Fewer (explicit operands) |
| **Instruction size** | Smaller (no register fields) | Larger (2-3 register operands) |
| **Code size** | Often smaller overall | Often larger overall |
| **Dispatches** | More (more instructions) | Fewer |
| **Implementation** | Simpler | More complex |
| **Optimization** | Harder (stack is implicit) | Easier (registers are explicit) |
| **Examples** | JVM, CPython, CLR, WASM | Lua 5, Dalvik, BEAM |

Research by Shi et al. (2008) showed that register-based VMs execute about 47% fewer instructions, and despite larger code size, are typically 20-30% faster.

---

## 6. Instruction Dispatch Techniques

### 6.1 Switch Dispatch

The simplest dispatch mechanism: a large `switch` (or `if/elif`) statement.

```python
def switch_dispatch(bytecode, constants):
    """
    Execute bytecode using switch dispatch.
    This is the simplest but slowest dispatch method.
    """
    stack = []
    pc = 0

    while pc < len(bytecode):
        opcode, arg = bytecode[pc]

        if opcode == OpCode.CONST:
            stack.append(constants[arg])
        elif opcode == OpCode.ADD:
            b, a = stack.pop(), stack.pop()
            stack.append(a + b)
        elif opcode == OpCode.SUB:
            b, a = stack.pop(), stack.pop()
            stack.append(a - b)
        elif opcode == OpCode.MUL:
            b, a = stack.pop(), stack.pop()
            stack.append(a * b)
        elif opcode == OpCode.PRINT:
            print(stack.pop())
        elif opcode == OpCode.HALT:
            break
        # ... more cases

        pc += 1
```

**Problem**: The CPU's branch predictor sees one branch point for all opcodes. It can only predict the next opcode based on the previous one (poor accuracy).

### 6.2 Direct Threaded Code

Replace opcode values with direct pointers to handler code. After each handler executes, it jumps directly to the next handler (no central dispatch loop).

In C (not possible in pure Python due to language limitations):

```c
// C implementation of direct threading
void* dispatch_table[] = {
    &&op_const, &&op_add, &&op_sub, &&op_mul, /* ... */
};

// Initial dispatch
goto *dispatch_table[bytecode[pc]];

op_const:
    stack[sp++] = constants[bytecode[pc+1]];
    pc += 2;
    goto *dispatch_table[bytecode[pc]];  // Direct jump to next handler

op_add:
    sp--;
    stack[sp-1] += stack[sp];
    pc += 1;
    goto *dispatch_table[bytecode[pc]];
```

**Advantage**: Each handler has its own indirect branch, giving the CPU branch predictor more context. This typically gives 15-25% speedup over switch dispatch.

### 6.3 Computed Goto (GCC Extension)

GCC's `&&label` extension enables direct threading in C. CPython uses this when available:

```python
# Python simulation of computed goto dispatch
# (In practice, this is done in C with goto *dispatch_table[opcode])

def computed_goto_simulation(bytecode, constants):
    """
    Simulate computed goto dispatch in Python.

    In real C implementations, this uses GCC's &&label extension
    for indirect branches, which enables better branch prediction.
    """
    stack = []
    pc = 0

    # Handler functions (simulate goto targets)
    def handle_const():
        nonlocal pc
        stack.append(constants[bytecode[pc][1]])
        pc += 1

    def handle_add():
        nonlocal pc
        b, a = stack.pop(), stack.pop()
        stack.append(a + b)
        pc += 1

    def handle_mul():
        nonlocal pc
        b, a = stack.pop(), stack.pop()
        stack.append(a * b)
        pc += 1

    def handle_print():
        nonlocal pc
        print(f"  Output: {stack.pop()}")
        pc += 1

    def handle_halt():
        nonlocal pc
        pc = len(bytecode)  # Exit

    # Dispatch table (simulates array of goto labels)
    dispatch = {
        OpCode.CONST: handle_const,
        OpCode.ADD: handle_add,
        OpCode.MUL: handle_mul,
        OpCode.PRINT: handle_print,
        OpCode.HALT: handle_halt,
    }

    while pc < len(bytecode):
        opcode = bytecode[pc][0]
        dispatch[opcode]()  # "Computed goto"
```

### 6.4 Subroutine Threading

Each bytecode instruction is compiled into a call to its handler subroutine. Faster than switch, slower than direct threading (call/return overhead).

### 6.5 Dispatch Performance Comparison

```python
import time

def benchmark_dispatch():
    """
    Compare dispatch techniques (simplified Python benchmark).
    Real-world differences are more pronounced in C/C++.
    """
    # Create a simple program: push 1, push 2, add, repeated 1M times
    n = 100_000

    bytecode = []
    constants = [1, 2]
    for _ in range(n):
        bytecode.append((OpCode.CONST, 0))
        bytecode.append((OpCode.CONST, 1))
        bytecode.append((OpCode.ADD, None))
        bytecode.append((OpCode.POP, None))
    bytecode.append((OpCode.HALT, None))

    # Method 1: if/elif chain
    start = time.perf_counter()
    stack = []
    pc = 0
    while pc < len(bytecode):
        op, arg = bytecode[pc]
        if op == OpCode.CONST:
            stack.append(constants[arg])
        elif op == OpCode.ADD:
            b, a = stack.pop(), stack.pop()
            stack.append(a + b)
        elif op == OpCode.POP:
            stack.pop()
        elif op == OpCode.HALT:
            break
        pc += 1
    time_switch = time.perf_counter() - start

    # Method 2: Dictionary dispatch
    def do_const(s, c, a): s.append(c[a])
    def do_add(s, c, a): b, a2 = s.pop(), s.pop(); s.append(a2 + b)
    def do_pop(s, c, a): s.pop()

    dispatch_table = {
        OpCode.CONST: do_const,
        OpCode.ADD: do_add,
        OpCode.POP: do_pop,
    }

    start = time.perf_counter()
    stack = []
    pc = 0
    while pc < len(bytecode):
        op, arg = bytecode[pc]
        if op == OpCode.HALT:
            break
        dispatch_table[op](stack, constants, arg)
        pc += 1
    time_dict = time.perf_counter() - start

    print(f"=== Dispatch Benchmark ({n} iterations) ===")
    print(f"if/elif chain: {time_switch:.3f}s")
    print(f"Dict dispatch: {time_dict:.3f}s")
    print(f"Ratio: {time_dict/time_switch:.2f}x")

# benchmark_dispatch()  # Uncomment to run
```

---

## 7. A Complete Bytecode Compiler and VM

This section builds a complete bytecode compiler and stack-based VM for a simple language.

### 7.1 The Language

Our language (called "Mini") supports:
- Integers, floats, booleans, strings
- Arithmetic and comparison operators
- Variables and assignment
- `if`/`else` conditionals
- `while` loops
- Functions with parameters and return values
- Print statement

### 7.2 The Bytecode Compiler

```python
class Compiler:
    """
    Bytecode compiler: AST -> CodeObject.
    Walks the AST and emits bytecode instructions.
    """

    def __init__(self):
        self.code = CodeObject('<module>')
        self.functions = {}  # name -> CodeObject

    def compile(self, program):
        """Compile a list of AST statements to bytecode."""
        for stmt in program:
            self.compile_node(stmt)
        self.code.emit(OpCode.HALT)
        return self.code

    def compile_node(self, node):
        """Compile a single AST node."""
        method = getattr(self, f'compile_{type(node).__name__}', None)
        if method is None:
            raise CompileError(f"Cannot compile {type(node).__name__}")
        method(node)

    def compile_NumberLit(self, node):
        idx = self.code.add_constant(node.value)
        self.code.emit(OpCode.CONST, idx)

    def compile_StringLit(self, node):
        idx = self.code.add_constant(node.value)
        self.code.emit(OpCode.CONST, idx)

    def compile_BoolLit(self, node):
        if node.value:
            self.code.emit(OpCode.TRUE)
        else:
            self.code.emit(OpCode.FALSE)

    def compile_Identifier(self, node):
        slot = self.code.add_local(node.name)
        self.code.emit(OpCode.LOAD, slot)

    def compile_BinOp(self, node):
        # Compile left operand
        self.compile_node(node.left)
        # Compile right operand
        self.compile_node(node.right)
        # Emit operation
        op_map = {
            '+': OpCode.ADD, '-': OpCode.SUB,
            '*': OpCode.MUL, '/': OpCode.DIV,
            '%': OpCode.MOD,
            '==': OpCode.EQ, '!=': OpCode.NE,
            '<': OpCode.LT, '>': OpCode.GT,
            '<=': OpCode.LE, '>=': OpCode.GE,
        }
        if node.op not in op_map:
            raise CompileError(f"Unknown operator: {node.op}")
        self.code.emit(op_map[node.op])

    def compile_UnaryOp(self, node):
        self.compile_node(node.operand)
        if node.op == '-':
            self.code.emit(OpCode.NEG)
        elif node.op == 'not':
            self.code.emit(OpCode.NOT)

    def compile_Assign(self, node):
        self.compile_node(node.value)
        slot = self.code.add_local(node.name)
        self.code.emit(OpCode.STORE, slot)

    def compile_If(self, node):
        # Compile condition
        self.compile_node(node.condition)

        # Jump to else/end if false
        jump_to_else = self.code.emit(OpCode.JUMP_IF_FALSE, 0)  # Placeholder

        # Compile then body
        for stmt in node.then_body:
            self.compile_node(stmt)

        if node.else_body:
            # Jump over else body
            jump_to_end = self.code.emit(OpCode.JUMP, 0)  # Placeholder

            # Patch jump to else
            else_start = len(self.code.bytecode)
            self.code.bytecode[jump_to_else] = (OpCode.JUMP_IF_FALSE, else_start)

            # Compile else body
            for stmt in node.else_body:
                self.compile_node(stmt)

            # Patch jump to end
            end_pos = len(self.code.bytecode)
            self.code.bytecode[jump_to_end] = (OpCode.JUMP, end_pos)
        else:
            # Patch jump to end (no else)
            end_pos = len(self.code.bytecode)
            self.code.bytecode[jump_to_else] = (OpCode.JUMP_IF_FALSE, end_pos)

    def compile_While(self, node):
        # Loop start
        loop_start = len(self.code.bytecode)

        # Compile condition
        self.compile_node(node.condition)

        # Jump to end if false
        jump_to_end = self.code.emit(OpCode.JUMP_IF_FALSE, 0)  # Placeholder

        # Compile body
        for stmt in node.body:
            self.compile_node(stmt)

        # Jump back to start
        self.code.emit(OpCode.JUMP, loop_start)

        # Patch jump to end
        end_pos = len(self.code.bytecode)
        self.code.bytecode[jump_to_end] = (OpCode.JUMP_IF_FALSE, end_pos)

    def compile_Print(self, node):
        self.compile_node(node.value)
        self.code.emit(OpCode.PRINT)

    def compile_FuncDef(self, node):
        # Compile function to a separate CodeObject
        func_code = CodeObject(node.name)

        # Add parameters as locals
        for param in node.params:
            func_code.add_local(param)

        # Save current code object, switch to function's
        parent_code = self.code
        self.code = func_code

        # Compile function body
        for stmt in node.body:
            self.compile_node(stmt)

        # Ensure function returns None if no explicit return
        self.code.emit(OpCode.NONE)
        self.code.emit(OpCode.RETURN)

        # Restore parent code object
        self.code = parent_code

        # Store function in constants
        func_idx = self.code.add_constant(func_code)
        func_slot = self.code.add_local(node.name)
        self.code.emit(OpCode.CONST, func_idx)
        self.code.emit(OpCode.STORE, func_slot)

    def compile_FuncCall(self, node):
        # Push function object
        slot = self.code.add_local(node.name)
        self.code.emit(OpCode.LOAD, slot)

        # Push arguments
        for arg in node.args:
            self.compile_node(arg)

        # Call with number of arguments
        self.code.emit(OpCode.CALL, len(node.args))

    def compile_Return(self, node):
        if node.value:
            self.compile_node(node.value)
        else:
            self.code.emit(OpCode.NONE)
        self.code.emit(OpCode.RETURN)


class CompileError(Exception):
    pass
```

### 7.3 The Virtual Machine

```python
class Frame:
    """
    Call frame: represents a function invocation.
    Contains local variables and a return address.
    """

    def __init__(self, code, return_addr=0):
        self.code = code
        self.pc = 0
        self.locals = [None] * (code.num_locals + 16)
        self.return_addr = return_addr


class VirtualMachine:
    """
    Stack-based virtual machine for our bytecode.
    """

    def __init__(self):
        self.stack = []
        self.frames = []
        self.current_frame = None
        self.output = []       # Captured output

    def run(self, code):
        """Execute a CodeObject."""
        self.current_frame = Frame(code)
        self.frames.append(self.current_frame)

        while True:
            frame = self.current_frame
            if frame.pc >= len(frame.code.bytecode):
                break

            opcode, arg = frame.code.bytecode[frame.pc]
            frame.pc += 1

            # Dispatch
            if opcode == OpCode.CONST:
                self.stack.append(frame.code.constants[arg])

            elif opcode == OpCode.POP:
                self.stack.pop()

            elif opcode == OpCode.ADD:
                b, a = self.stack.pop(), self.stack.pop()
                self.stack.append(a + b)

            elif opcode == OpCode.SUB:
                b, a = self.stack.pop(), self.stack.pop()
                self.stack.append(a - b)

            elif opcode == OpCode.MUL:
                b, a = self.stack.pop(), self.stack.pop()
                self.stack.append(a * b)

            elif opcode == OpCode.DIV:
                b, a = self.stack.pop(), self.stack.pop()
                if b == 0:
                    raise RuntimeError("Division by zero")
                self.stack.append(a / b)

            elif opcode == OpCode.MOD:
                b, a = self.stack.pop(), self.stack.pop()
                self.stack.append(a % b)

            elif opcode == OpCode.NEG:
                self.stack.append(-self.stack.pop())

            elif opcode == OpCode.EQ:
                b, a = self.stack.pop(), self.stack.pop()
                self.stack.append(a == b)

            elif opcode == OpCode.NE:
                b, a = self.stack.pop(), self.stack.pop()
                self.stack.append(a != b)

            elif opcode == OpCode.LT:
                b, a = self.stack.pop(), self.stack.pop()
                self.stack.append(a < b)

            elif opcode == OpCode.GT:
                b, a = self.stack.pop(), self.stack.pop()
                self.stack.append(a > b)

            elif opcode == OpCode.LE:
                b, a = self.stack.pop(), self.stack.pop()
                self.stack.append(a <= b)

            elif opcode == OpCode.GE:
                b, a = self.stack.pop(), self.stack.pop()
                self.stack.append(a >= b)

            elif opcode == OpCode.NOT:
                self.stack.append(not self.stack.pop())

            elif opcode == OpCode.LOAD:
                self.stack.append(frame.locals[arg])

            elif opcode == OpCode.STORE:
                frame.locals[arg] = self.stack.pop()

            elif opcode == OpCode.JUMP:
                frame.pc = arg

            elif opcode == OpCode.JUMP_IF_FALSE:
                if not self.stack.pop():
                    frame.pc = arg

            elif opcode == OpCode.JUMP_IF_TRUE:
                if self.stack.pop():
                    frame.pc = arg

            elif opcode == OpCode.CALL:
                num_args = arg
                args = []
                for _ in range(num_args):
                    args.insert(0, self.stack.pop())

                func_code = self.stack.pop()  # Pop function object

                if not isinstance(func_code, CodeObject):
                    raise RuntimeError(f"Not callable: {func_code}")

                # Create new frame
                new_frame = Frame(func_code, return_addr=0)

                # Bind arguments to parameters
                for i, val in enumerate(args):
                    new_frame.locals[i] = val

                # Push current frame's return info
                self.frames.append(new_frame)
                self.current_frame = new_frame

            elif opcode == OpCode.RETURN:
                return_value = self.stack.pop()

                # Pop frame
                self.frames.pop()
                if not self.frames:
                    return return_value

                self.current_frame = self.frames[-1]
                self.stack.append(return_value)

            elif opcode == OpCode.PRINT:
                value = self.stack.pop()
                self.output.append(str(value))
                print(f"  >> {value}")

            elif opcode == OpCode.TRUE:
                self.stack.append(True)

            elif opcode == OpCode.FALSE:
                self.stack.append(False)

            elif opcode == OpCode.NONE:
                self.stack.append(None)

            elif opcode == OpCode.HALT:
                break

            else:
                raise RuntimeError(f"Unknown opcode: {opcode}")

        return self.stack[-1] if self.stack else None
```

### 7.4 Putting It All Together

```python
def run_mini_program():
    """Compile and run a complete Mini program."""
    print("=== Complete Bytecode Compiler + VM Demo ===\n")

    # Program: compute factorial using a loop
    program = [
        # n = 10
        Assign('n', NumberLit(10)),

        # result = 1
        Assign('result', NumberLit(1)),

        # i = 1
        Assign('i', NumberLit(1)),

        # while i <= n:
        While(
            BinOp('<=', Identifier('i'), Identifier('n')),
            [
                # result = result * i
                Assign('result', BinOp('*', Identifier('result'),
                                       Identifier('i'))),
                # i = i + 1
                Assign('i', BinOp('+', Identifier('i'), NumberLit(1))),
            ]
        ),

        # print(result)
        Print(Identifier('result')),
    ]

    # Compile
    compiler = Compiler()
    code = compiler.compile(program)

    # Disassemble
    code.disassemble()

    # Execute
    print("\n--- Execution ---")
    vm = VirtualMachine()
    vm.run(code)

    # Program 2: Recursive function
    print("\n" + "=" * 50)
    print("Program 2: Recursive Fibonacci\n")

    program2 = [
        FuncDef('fib', ['n'], [
            If(
                BinOp('<', Identifier('n'), NumberLit(2)),
                [Return(Identifier('n'))],
                [Return(BinOp('+',
                              FuncCall('fib', [BinOp('-', Identifier('n'),
                                                     NumberLit(1))]),
                              FuncCall('fib', [BinOp('-', Identifier('n'),
                                                     NumberLit(2))])))]
            )
        ]),
        Print(FuncCall('fib', [NumberLit(10)])),
    ]

    compiler2 = Compiler()
    code2 = compiler2.compile(program2)
    code2.disassemble()

    print("\n--- Execution ---")
    vm2 = VirtualMachine()
    vm2.run(code2)

run_mini_program()
```

---

## 8. JIT Compilation

### 8.1 Why JIT?

Bytecode interpretation is still 5-20x slower than native code due to dispatch overhead, lack of register usage, and inability to apply traditional compiler optimizations. **Just-In-Time (JIT) compilation** bridges this gap by compiling bytecode to native code at runtime.

```
              Startup Speed  ──────────────────▶  Steady-state Speed
Interpretation  ████████                         ░░░░░░░░
Bytecode        ██████████                       ████████
Method JIT      ████████████                     ████████████████
Tracing JIT     ██████████                       ██████████████████████
AOT Compile     ░░░░                             ████████████████████████
```

### 8.2 Method JIT

A **method JIT** compiler identifies hot methods (frequently called functions) and compiles them to native code.

```python
class MethodJITSimulator:
    """
    Simulates method JIT compilation behavior.

    Hot methods are "compiled" (represented by optimized Python functions)
    when their call count exceeds a threshold.
    """

    def __init__(self, hot_threshold=10):
        self.hot_threshold = hot_threshold
        self.call_counts = {}       # method name -> count
        self.compiled = {}          # method name -> compiled version
        self.compilation_log = []

    def call_method(self, name, interpreted_func, *args):
        """
        Call a method, potentially triggering JIT compilation.
        """
        # Track call count
        self.call_counts[name] = self.call_counts.get(name, 0) + 1

        # Check if we should compile
        if (name not in self.compiled and
                self.call_counts[name] >= self.hot_threshold):
            self.compile_method(name, interpreted_func)

        # Execute compiled version if available
        if name in self.compiled:
            return self.compiled[name](*args)
        else:
            return interpreted_func(*args)

    def compile_method(self, name, func):
        """Simulate JIT compilation of a method."""
        self.compilation_log.append(name)
        # In reality, this would generate native code
        # We simulate "optimization" by creating an optimized version
        self.compiled[name] = func  # In practice, a native code version
        print(f"  [JIT] Compiled method: {name} "
              f"(after {self.call_counts[name]} calls)")


def demonstrate_method_jit():
    """Show method JIT behavior."""
    print("=== Method JIT Simulation ===\n")

    jit = MethodJITSimulator(hot_threshold=5)

    def fibonacci(n):
        if n < 2:
            return n
        return fibonacci(n - 1) + fibonacci(n - 2)

    # First few calls: interpreted
    for i in range(20):
        result = jit.call_method('fibonacci', fibonacci, 10)

    print(f"\nCall counts: {jit.call_counts}")
    print(f"Compiled methods: {list(jit.compiled.keys())}")

demonstrate_method_jit()
```

### 8.3 Tracing JIT

A **tracing JIT** records the actual execution path through a hot loop (a "trace"), then compiles that specific path to native code.

```python
class TracingJITSimulator:
    """
    Simulates tracing JIT compilation.

    Records execution traces of hot loops and "compiles" them.
    """

    def __init__(self, hot_threshold=3):
        self.hot_threshold = hot_threshold
        self.loop_counts = {}
        self.traces = {}
        self.recording = False
        self.current_trace = []

    def enter_loop(self, loop_id):
        """Called at the top of each loop iteration."""
        self.loop_counts[loop_id] = self.loop_counts.get(loop_id, 0) + 1

        if loop_id in self.traces:
            return 'compiled'  # Use compiled trace

        if self.loop_counts[loop_id] >= self.hot_threshold and not self.recording:
            self.recording = True
            self.current_trace = []
            self.current_loop = loop_id
            print(f"  [Trace] Starting trace recording for loop {loop_id}")
            return 'recording'

        return 'interpreting'

    def record_operation(self, op, *args):
        """Record an operation during tracing."""
        if self.recording:
            self.current_trace.append((op, args))

    def record_guard(self, condition, description):
        """Record a guard (type check, bounds check, etc.)."""
        if self.recording:
            self.current_trace.append(('GUARD', description, condition))

    def end_loop_iteration(self, loop_id):
        """Called at the end of each loop iteration."""
        if self.recording and loop_id == self.current_loop:
            # Compile the trace
            self.traces[loop_id] = list(self.current_trace)
            self.recording = False
            print(f"  [Trace] Compiled trace for loop {loop_id} "
                  f"({len(self.current_trace)} operations)")
            self.current_trace = []

    def show_trace(self, loop_id):
        """Display a compiled trace."""
        if loop_id not in self.traces:
            print(f"No trace for loop {loop_id}")
            return

        print(f"\n=== Trace for loop {loop_id} ===")
        for i, entry in enumerate(self.traces[loop_id]):
            if entry[0] == 'GUARD':
                print(f"  {i}: GUARD {entry[1]}")
            else:
                print(f"  {i}: {entry[0]} {entry[1]}")


def demonstrate_tracing_jit():
    """Show tracing JIT behavior."""
    print("=== Tracing JIT Simulation ===\n")

    jit = TracingJITSimulator(hot_threshold=3)

    # Simulate a loop: sum = 0; for i in range(10): sum += i * 2
    arr = list(range(10))

    for iteration in range(5):  # Run the same loop 5 times
        total = 0

        for i in range(len(arr)):
            status = jit.enter_loop('sum_loop')

            # Record operations
            val = arr[i]
            jit.record_guard(isinstance(val, int), "type(val) is int")
            jit.record_operation('LOAD_ARRAY', 'arr', i)

            doubled = val * 2
            jit.record_operation('MUL', val, 2)

            total += doubled
            jit.record_operation('ADD', 'total', doubled)

            jit.end_loop_iteration('sum_loop')

        print(f"  Iteration {iteration}: sum = {total}, "
              f"loop mode = {status}")

    jit.show_trace('sum_loop')

demonstrate_tracing_jit()
```

### 8.4 Trace Compilation Details

A trace is a linear sequence of operations with guards:

```
Trace for "for i in range(n): sum += arr[i] * 2":

  GUARD: i < n                    # Loop condition
  LOAD_ARRAY arr, i               # Load arr[i]
  GUARD: type(arr[i]) == int      # Type check
  MUL temp, arr[i], 2             # temp = arr[i] * 2
  ADD sum, sum, temp              # sum += temp
  ADD i, i, 1                     # i++
  JUMP loop_start                 # Back to start
```

If any guard fails, the trace is abandoned ("side exit"), and execution falls back to the interpreter. Common guard failures trigger new traces for alternative paths.

### 8.5 On-Stack Replacement (OSR)

**On-stack replacement** allows switching between interpreted and compiled code in the middle of a function (even inside a loop). Without OSR, a hot loop would have to complete an entire function call in the interpreter before the compiled version could be used.

```
Without OSR:
  Interpreter: ████████████████████████████  (entire loop interpreted)
  Next call:   ░░░░ (compiled)

With OSR:
  Interpreter: ████████░░░░░░░░░░░░░░░░░░░  (OSR into compiled code mid-loop)
               ↑ OSR point
```

---

## 9. Runtime Optimization Techniques

### 9.1 Inline Caching

**Inline caching** speeds up method dispatch in dynamic languages. The first time a method is called on an object, the runtime looks up the method and caches the result at the call site.

```python
class InlineCache:
    """
    Simulates inline caching for method dispatch.

    Three states:
    1. Uninitialized: No cache (first call triggers lookup)
    2. Monomorphic: One cached type (fastest, most common)
    3. Polymorphic: Multiple cached types (still fast)
    4. Megamorphic: Too many types (falls back to generic lookup)
    """

    def __init__(self, method_name, max_entries=4):
        self.method_name = method_name
        self.cache = {}  # type -> method
        self.max_entries = max_entries
        self.hits = 0
        self.misses = 0

    def lookup(self, obj):
        """Look up a method using the inline cache."""
        obj_type = type(obj)

        if obj_type in self.cache:
            self.hits += 1
            return self.cache[obj_type]

        # Cache miss -- do full lookup
        self.misses += 1
        method = getattr(obj, self.method_name, None)

        if method and len(self.cache) < self.max_entries:
            self.cache[obj_type] = method  # Cache it

        return method

    @property
    def state(self):
        n = len(self.cache)
        if n == 0:
            return "uninitialized"
        elif n == 1:
            return "monomorphic"
        elif n <= self.max_entries:
            return "polymorphic"
        else:
            return "megamorphic"

    def __repr__(self):
        return (f"IC({self.method_name}, state={self.state}, "
                f"hits={self.hits}, misses={self.misses})")


def demonstrate_inline_caching():
    """Show inline caching behavior."""
    print("=== Inline Caching Demo ===\n")

    class Dog:
        def speak(self): return "Woof!"

    class Cat:
        def speak(self): return "Meow!"

    class Duck:
        def speak(self): return "Quack!"

    cache = InlineCache('speak')

    animals = [Dog(), Dog(), Dog(), Cat(), Dog(), Duck(), Dog()]

    for animal in animals:
        method = cache.lookup(animal)
        result = method()
        print(f"  {type(animal).__name__}.speak() = {result} | Cache: {cache}")

demonstrate_inline_caching()
```

### 9.2 Type Specialization

JIT compilers generate specialized code for specific types:

```python
def demonstrate_type_specialization():
    """Show type specialization in a JIT compiler."""
    print("=== Type Specialization ===\n")

    # Generic add (what the interpreter does)
    def generic_add(a, b):
        """Must check types at runtime."""
        if isinstance(a, int) and isinstance(b, int):
            return a + b
        elif isinstance(a, float) and isinstance(b, float):
            return a + b
        elif isinstance(a, str) and isinstance(b, str):
            return a + b
        elif isinstance(a, list) and isinstance(b, list):
            return a + b
        else:
            raise TypeError(f"Cannot add {type(a)} and {type(b)}")

    # Specialized add (what the JIT generates after seeing types)
    def int_add(a, b):
        """No type checks needed -- we know both are ints."""
        return a + b  # Direct integer addition

    # In practice, the JIT generates:
    print("Generic version (interpreter):")
    print("  1. Check type of a")
    print("  2. Check type of b")
    print("  3. Look up appropriate + operator")
    print("  4. Perform addition")
    print("  5. Box result")

    print("\nSpecialized version (JIT, after seeing int+int):")
    print("  1. GUARD: type(a) == int  (deoptimize if not)")
    print("  2. GUARD: type(b) == int  (deoptimize if not)")
    print("  3. Native integer addition (single CPU instruction)")
    print("  4. Result already unboxed")

    # Benchmark
    import time

    n = 1_000_000
    start = time.perf_counter()
    total = 0
    for i in range(n):
        total = generic_add(total, i)
    time_generic = time.perf_counter() - start

    start = time.perf_counter()
    total = 0
    for i in range(n):
        total = int_add(total, i)
    time_specialized = time.perf_counter() - start

    print(f"\nGeneric add ({n} iterations): {time_generic:.3f}s")
    print(f"Specialized add ({n} iterations): {time_specialized:.3f}s")
    print(f"Speedup: {time_generic / time_specialized:.2f}x")

# demonstrate_type_specialization()  # Uncomment to run
```

### 9.3 Deoptimization

When assumptions made by the JIT are violated (guard failure), the runtime must **deoptimize** -- fall back from compiled code to the interpreter.

```
Compiled code:
  GUARD type(x) == int     ──── guard fails ────▶  Deoptimize
  native_int_add(x, y)                              │
  ...                                                ▼
                                                 Interpreter
                                                 (continue execution
                                                  with correct semantics)
```

Deoptimization requires:
1. Reconstructing the interpreter state (stack, locals) from the compiled state
2. Invalidating the compiled code (if the assumption is permanently wrong)
3. Possibly recompiling with less aggressive assumptions

### 9.4 Hidden Classes / Shapes

V8 and other VMs use **hidden classes** (called "shapes" or "maps") to optimize property access on dynamic objects:

```python
def demonstrate_hidden_classes():
    """Demonstrate the concept of hidden classes / shapes."""
    print("=== Hidden Classes / Shapes ===\n")

    # In JavaScript, objects are dictionaries:
    # let p = {};  p.x = 1;  p.y = 2;

    # V8 creates hidden classes for each "shape":
    shapes = {
        'Shape0': {},                          # Empty object
        'Shape1': {'x': 'offset 0'},           # After p.x = 1
        'Shape2': {'x': 'offset 0', 'y': 'offset 1'},  # After p.y = 2
    }

    print("Object shape transitions:")
    print("  let p = {}          -> Shape0 (empty)")
    print("  p.x = 1             -> Shape1 ({x: offset 0})")
    print("  p.y = 2             -> Shape2 ({x: offset 0, y: offset 1})")
    print()
    print("Objects with the same shape share the same hidden class.")
    print("Property access becomes a fixed-offset load (like a struct).")
    print()
    print("  let q = {}; q.x = 5; q.y = 10;")
    print("  q has the SAME Shape2 as p!")
    print("  Accessing q.y is: load [q + offset_of_y]  (no dictionary lookup)")

demonstrate_hidden_classes()
```

---

## 10. Real Virtual Machines

### 10.1 JVM (Java Virtual Machine)

```
JVM Architecture:
┌──────────────────────────────────────────────────┐
│                    JVM                            │
│  ┌──────────┐   ┌──────────────────────────┐     │
│  │ Class    │   │      Runtime Data Areas   │     │
│  │ Loader   │   │  ┌──────┐  ┌──────────┐  │     │
│  │          │   │  │Method│  │  Heap     │  │     │
│  └──────────┘   │  │Area  │  │(GC-managed│  │     │
│                 │  └──────┘  └──────────┘  │     │
│  ┌──────────┐   │  ┌──────┐  ┌──────────┐  │     │
│  │Execution │   │  │Stack │  │ PC Regs   │  │     │
│  │ Engine   │   │  │(per  │  │(per thread│  │     │
│  │ - Interp │   │  │thread│  └──────────┘  │     │
│  │ - JIT(C1)│   │  └──────┘                │     │
│  │ - JIT(C2)│   └──────────────────────────┘     │
│  └──────────┘                                     │
└──────────────────────────────────────────────────┘
```

Key characteristics:
- **Stack-based bytecode**: ~200 opcodes
- **Typed instructions**: `iadd` (int), `fadd` (float), `dadd` (double)
- **Tiered compilation**: Interpreter -> C1 (fast compile) -> C2 (optimizing compile)
- **GC**: Multiple collectors (G1 default, ZGC for low latency)
- **Bytecode verification**: Type-safe bytecodes verified before execution

### 10.2 CPython VM

```
CPython Architecture:
┌─────────────────────────────────────────────┐
│              CPython                         │
│  ┌──────────┐   ┌────────────┐              │
│  │ Parser   │──▶│ Compiler   │              │
│  │ (PEG)    │   │ (AST→BC)   │              │
│  └──────────┘   └──────────┬─┘              │
│                            │                │
│                            ▼                │
│  ┌──────────────────────────────────┐       │
│  │        Bytecode Interpreter      │       │
│  │  (ceval.c: giant switch stmt)    │       │
│  │                                  │       │
│  │  - Stack-based                   │       │
│  │  - ~120 opcodes                  │       │
│  │  - Reference counting + cycle GC │       │
│  │  - GIL (Global Interpreter Lock) │       │
│  └──────────────────────────────────┘       │
└─────────────────────────────────────────────┘
```

You can inspect CPython bytecode using the `dis` module:

```python
import dis

def example_function(x, y):
    return x * x + y * y

print("=== CPython Bytecode ===")
dis.dis(example_function)
```

Output (approximate):

```
  2           0 LOAD_FAST                0 (x)
              2 LOAD_FAST                0 (x)
              4 BINARY_MULTIPLY
              6 LOAD_FAST                1 (y)
              8 LOAD_FAST                1 (y)
             10 BINARY_MULTIPLY
             12 BINARY_ADD
             14 RETURN_VALUE
```

### 10.3 V8 (JavaScript)

```
V8 Architecture:
┌──────────────────────────────────────────────┐
│                    V8                         │
│  ┌──────────┐                                │
│  │ Parser   │──▶ AST                         │
│  └──────────┘     │                          │
│                   ▼                          │
│  ┌──────────────────────┐                    │
│  │  Ignition            │  (Bytecode         │
│  │  (Bytecode Compiler) │   Interpreter)     │
│  └──────────┬───────────┘                    │
│             │ Profile data                   │
│             ▼                                │
│  ┌──────────────────────┐                    │
│  │  TurboFan            │  (Optimizing       │
│  │  (JIT Compiler)      │   JIT Compiler)    │
│  └──────────────────────┘                    │
│                                              │
│  Key techniques:                             │
│  - Hidden classes (shapes/maps)              │
│  - Inline caching                            │
│  - On-stack replacement                      │
│  - Deoptimization                            │
│  - Generational GC (Orinoco)                 │
└──────────────────────────────────────────────┘
```

### 10.4 BEAM (Erlang VM)

The BEAM is unique among VMs for its focus on concurrency and fault tolerance:

- **Register-based**: 1024 virtual registers per process
- **Lightweight processes**: Millions of processes, each with ~2KB stack
- **Preemptive scheduling**: Reductions-based (not time-based)
- **Hot code loading**: Replace running code without stopping
- **Pattern matching**: First-class bytecode support for pattern matching
- **No shared state**: Processes communicate only via message passing

```
BEAM Process Model:
┌────────────────────────────────────────────┐
│  BEAM VM                                    │
│  ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐       │
│  │Proc │  │Proc │  │Proc │  │Proc │  ...  │
│  │  1  │  │  2  │  │  3  │  │  4  │       │
│  │ 2KB │  │ 2KB │  │ 2KB │  │ 2KB │       │
│  └──┬──┘  └──┬──┘  └──┬──┘  └──┬──┘       │
│     │ msg    │ msg    │ msg    │ msg       │
│     └────────┴────────┴────────┘           │
│  ┌──────────────────────────────┐          │
│  │  Scheduler (per CPU core)    │          │
│  │  Reduction counting          │          │
│  └──────────────────────────────┘          │
└────────────────────────────────────────────┘
```

### 10.5 VM Comparison

```python
def vm_comparison():
    """Compare real-world VM implementations."""
    vms = [
        {
            'name': 'JVM',
            'type': 'Stack',
            'jit': 'Method JIT (C1/C2)',
            'gc': 'Generational (G1/ZGC)',
            'concurrency': 'OS threads',
            'typing': 'Static',
        },
        {
            'name': 'CPython',
            'type': 'Stack',
            'jit': 'None (3.13+ experimental)',
            'gc': 'RefCount + Gen cycle',
            'concurrency': 'GIL (threads limited)',
            'typing': 'Dynamic',
        },
        {
            'name': 'V8',
            'type': 'Register',
            'jit': 'TurboFan (optimizing)',
            'gc': 'Generational (Orinoco)',
            'concurrency': 'Event loop + workers',
            'typing': 'Dynamic',
        },
        {
            'name': 'BEAM',
            'type': 'Register',
            'jit': 'JIT (OTP 24+)',
            'gc': 'Per-process copying',
            'concurrency': 'Actor model (millions)',
            'typing': 'Dynamic',
        },
        {
            'name': 'CLR/.NET',
            'type': 'Stack',
            'jit': 'RyuJIT',
            'gc': 'Generational compacting',
            'concurrency': 'OS threads + async',
            'typing': 'Static',
        },
    ]

    print("=== VM Comparison ===\n")
    print(f"{'VM':<10} {'Type':<10} {'JIT':<25} {'GC':<25}")
    print("-" * 70)
    for vm in vms:
        print(f"{vm['name']:<10} {vm['type']:<10} {vm['jit']:<25} {vm['gc']:<25}")

vm_comparison()
```

---

## 11. Metacircular Interpreters

### 11.1 What is a Metacircular Interpreter?

A **metacircular interpreter** is an interpreter for a language written in the same language. It is a powerful concept from the Lisp tradition:

- **Lisp in Lisp**: The original metacircular evaluator (McCarthy, 1960)
- **PyPy**: Python interpreter written in (a subset of) Python
- **Truffle/Graal**: Self-optimizing interpreters written in Java

### 11.2 A Simple Metacircular Evaluator

```python
def metacircular_eval(expr, env):
    """
    A simple metacircular evaluator for a Lisp-like language.

    The language supports:
    - Numbers and strings (self-evaluating)
    - Variables (looked up in environment)
    - (quote x) -> x
    - (if test then else)
    - (define name value)
    - (lambda (params) body)
    - (function arg1 arg2 ...)
    """

    # Self-evaluating
    if isinstance(expr, (int, float, str)):
        return expr

    # Variable reference
    if isinstance(expr, str) and not expr.startswith('('):
        return env.get(expr)

    # Must be a list (compound expression)
    if not isinstance(expr, list):
        raise ValueError(f"Cannot evaluate: {expr}")

    if len(expr) == 0:
        return None

    head = expr[0]

    # Special forms
    if head == 'quote':
        return expr[1]

    elif head == 'if':
        _, test, then_clause, *else_clause = expr
        if metacircular_eval(test, env):
            return metacircular_eval(then_clause, env)
        elif else_clause:
            return metacircular_eval(else_clause[0], env)
        return None

    elif head == 'define':
        _, name, value = expr
        env.set(name, metacircular_eval(value, env))
        return None

    elif head == 'lambda':
        _, params, body = expr
        return ('closure', params, body, env)

    elif head == 'begin':
        result = None
        for subexpr in expr[1:]:
            result = metacircular_eval(subexpr, env)
        return result

    elif head == 'let':
        # (let ((x 1) (y 2)) body)
        _, bindings, body = expr
        new_env = Environment(parent=env)
        for name, value_expr in bindings:
            new_env.set(name, metacircular_eval(value_expr, env))
        return metacircular_eval(body, new_env)

    else:
        # Function application
        func = metacircular_eval(head, env)
        args = [metacircular_eval(arg, env) for arg in expr[1:]]

        if isinstance(func, tuple) and func[0] == 'closure':
            _, params, body, closure_env = func
            new_env = Environment(parent=closure_env)
            for param, arg in zip(params, args):
                new_env.set(param, arg)
            return metacircular_eval(body, new_env)

        elif callable(func):
            return func(*args)

        raise ValueError(f"Not a function: {func}")


def demonstrate_metacircular():
    """Demonstrate the metacircular evaluator."""
    print("=== Metacircular Evaluator ===\n")

    # Create global environment with built-in functions
    global_env = Environment()
    global_env.set('+', lambda a, b: a + b)
    global_env.set('-', lambda a, b: a - b)
    global_env.set('*', lambda a, b: a * b)
    global_env.set('/', lambda a, b: a / b)
    global_env.set('<', lambda a, b: a < b)
    global_env.set('>', lambda a, b: a > b)
    global_env.set('=', lambda a, b: a == b)
    global_env.set('print', lambda x: print(f"  Output: {x}") or x)

    # Evaluate expressions
    programs = [
        # Simple arithmetic
        (['+', 3, ['*', 4, 5]], "3 + 4*5"),

        # Define and use a variable
        (['begin',
          ['define', 'x', 42],
          ['print', 'x']], "define x = 42"),

        # Lambda function
        (['begin',
          ['define', 'square', ['lambda', ['x'], ['*', 'x', 'x']]],
          ['print', ['square', 7]]], "define square, compute square(7)"),

        # Recursive factorial (using begin for multiple defines)
        (['begin',
          ['define', 'fact',
           ['lambda', ['n'],
            ['if', ['<', 'n', 2],
             1,
             ['*', 'n', ['fact', ['-', 'n', 1]]]]]],
          ['print', ['fact', 10]]], "factorial(10)"),

        # Higher-order function
        (['begin',
          ['define', 'apply-twice',
           ['lambda', ['f', 'x'],
            ['f', ['f', 'x']]]],
          ['define', 'add3', ['lambda', ['x'], ['+', 'x', 3]]],
          ['print', ['apply-twice', 'add3', 10]]], "apply-twice(add3, 10)"),
    ]

    for expr, description in programs:
        print(f"Program: {description}")
        result = metacircular_eval(expr, global_env)
        if result is not None:
            print(f"  Result: {result}")
        print()

demonstrate_metacircular()
```

### 11.3 Why Metacircular Interpreters Matter

1. **Self-hosting**: A language that can implement itself demonstrates completeness and power.
2. **Reflection**: The interpreter is available to the running program (macros, eval, introspection).
3. **Bootstrapping**: Start with a simple interpreter, then build a better one in the language itself.
4. **Education**: They demonstrate language semantics in the clearest way possible.
5. **Optimization**: PyPy's meta-tracing approach writes the interpreter in Python, then automatically generates a JIT compiler.

---

## 12. Summary

This lesson covered the full spectrum of program execution strategies:

| Approach | Speed | Complexity | Key Example |
|----------|-------|-----------|-------------|
| **Tree-walking** | Slowest (50-200x) | Simple | Early Ruby, Bash |
| **Bytecode interp.** | Slow (5-20x) | Moderate | CPython, Lua |
| **Bytecode + JIT** | Near-native | Complex | JVM, V8, PyPy |
| **AOT compile** | Native | Complex | GCC, Rust, Go |

Key concepts:

- **Bytecode** is a compact, portable intermediate form designed for VM execution
- **Stack-based VMs** are simple to implement but generate more instructions; **register-based VMs** are more efficient but more complex
- **Instruction dispatch** technique (switch vs. threaded code vs. computed goto) significantly impacts interpreter performance
- **JIT compilation** bridges the gap between interpretation and native compilation:
  - Method JIT compiles hot functions
  - Tracing JIT compiles hot execution paths
  - Both rely on profiling, speculation, and deoptimization
- **Runtime optimizations** like inline caching, type specialization, and hidden classes make dynamic languages competitive with static ones
- **Metacircular interpreters** demonstrate language power and enable advanced techniques like meta-tracing

---

## 13. Exercises

### Exercise 1: Tree-Walking Extensions

Extend the tree-walking interpreter from Section 2 to support:
(a) Arrays with `push`, `pop`, and indexing.
(b) For loops: `for i in range(n): body`.
(c) Closures that correctly capture variables from enclosing scopes.

Test with a program that creates a list of closures:
```
funcs = []
for i in range(5):
    define make_adder(x): return lambda(y): x + y
    push(funcs, make_adder(i))
print(funcs[3](10))  # Should print 13
```

### Exercise 2: Bytecode Optimization

Implement a bytecode peephole optimizer that handles:
(a) Constant folding: `CONST 3; CONST 4; ADD` -> `CONST 7`
(b) Dead store elimination: `STORE x; STORE x` -> keep only the second
(c) Redundant load elimination: `STORE x; LOAD x` -> `STORE x; DUP`

Apply it to the output of the compiler from Section 7 and measure the reduction in instruction count.

### Exercise 3: Register Allocation for Register VM

Implement a simple register allocator that converts stack-based bytecode to register-based bytecode:
(a) Use a simple stack simulation to track which values are in which registers.
(b) Handle register spilling when you run out of registers.
(c) Compare instruction counts between the stack and register versions for several programs.

### Exercise 4: Simple JIT Compiler

Implement a simple JIT-like system that:
(a) Profiles which functions are called most frequently.
(b) For "hot" functions, generates a specialized Python function (using `exec`) that eliminates the dispatch overhead.
(c) Benchmark the interpreted vs "JIT-compiled" versions.

### Exercise 5: Debugger for the VM

Add debugging support to the VM from Section 7:
(a) Single-step execution (execute one instruction, then pause).
(b) Breakpoints (pause when reaching a specific instruction index).
(c) Stack and local variable inspection at any point.
(d) Call stack display (show the chain of function calls).

### Exercise 6: VM Performance Analysis

Instrument the VM to collect execution statistics:
(a) Count how many times each opcode is executed.
(b) Identify the most common opcode pairs (superinstructions).
(c) Propose and implement 3 superinstructions that combine common pairs.
(d) Measure the reduction in dispatch count and execution time.

---

## 14. References

1. Aho, A. V., Lam, M. S., Sethi, R., & Ullman, J. D. (2006). *Compilers: Principles, Techniques, and Tools* (2nd ed.), Chapter 8.
2. Nystrom, R. (2021). *Crafting Interpreters*. Available at [craftinginterpreters.com](https://craftinginterpreters.com/).
3. Smith, J. E., & Nair, R. (2005). *Virtual Machines: Versatile Platforms for Systems and Processes*. Morgan Kaufmann.
4. Ertl, M. A., & Gregg, D. (2003). "The Structure and Performance of Efficient Interpreters." *Journal of Instruction-Level Parallelism*, 5.
5. Shi, Y., Gregg, D., Beatty, A., & Ertl, M. A. (2008). "Virtual Machine Showdown: Stack versus Registers." *ACM TOPLAS*, 30(4).
6. Bolz, C. F., Cuni, A., Fijalkowski, M., & Rigo, A. (2009). "Tracing the Meta-level: PyPy's Tracing JIT Compiler." *ICOOOLPS*.
7. Holzle, U., Chambers, C., & Ungar, D. (1991). "Optimizing Dynamically-Typed Object-Oriented Languages With Polymorphic Inline Caches." *ECOOP*.
8. Deutsch, L. P., & Schiffman, A. M. (1984). "Efficient Implementation of the Smalltalk-80 System." *POPL*.

---

[Previous: 14. Garbage Collection](./14_Garbage_Collection.md) | [Next: 16. Modern Compiler Infrastructure](./16_Modern_Compiler_Infrastructure.md) | [Overview](./00_Overview.md)
