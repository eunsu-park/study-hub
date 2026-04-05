# 인터프리터와 가상 머신(Interpreters and Virtual Machines)

**이전**: [14. 가비지 컬렉션](./14_Garbage_Collection.md) | **다음**: [16. 현대 컴파일러 인프라](./16_Modern_Compiler_Infrastructure.md)

---

모든 언어 구현이 네이티브 기계 코드로 컴파일되는 것은 아닙니다. Python, Java, JavaScript, Ruby, Erlang, Lua 같이 가장 널리 사용되는 많은 언어들은 프로그램을 실행하기 위해 인터프리터 또는 가상 머신(또는 둘의 조합)에 의존합니다. 인터프리터와 VM이 어떻게 작동하는지 이해하는 것은 언어 구현자와 코드가 실행될 때 무슨 일이 일어나는지 이해하고 싶은 모든 프로그래머에게 필수적입니다.

이 레슨은 단순한 트리 순회 인터프리터부터 정교한 JIT 컴파일 가상 머신까지 실행 전략의 스펙트럼을 다룹니다. 간단한 언어를 위한 바이트코드 컴파일러와 스택 기반 VM을 Python으로 구현하여 프로덕션 VM이 어떻게 작동하는지 이해하기 위한 구체적인 기반을 제공합니다.

**난이도**: ⭐⭐⭐⭐

**선수 지식**: [07. 추상 구문 트리](./07_Abstract_Syntax_Trees.md), [09. 중간 표현](./09_Intermediate_Representations.md), [11. 코드 생성](./11_Code_Generation.md)

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 여러 축(속도, 이식성, 개발 사이클)에서 인터프리터와 컴파일러를 비교한다
2. 간단한 언어를 위한 트리 순회 인터프리터를 구현한다
3. 바이트코드 명령 집합을 설계하고 바이트코드 컴파일러를 구현한다
4. 스택 기반 가상 머신을 처음부터 만든다
5. 레지스터 기반 VM 설계와 그 장점을 이해한다
6. 명령 디스패치 기법과 성능 영향을 설명한다
7. JIT 컴파일 전략(메서드 JIT, 추적 JIT)을 기술한다
8. 런타임 최적화 기법(인라인 캐싱, 타입 특수화)을 설명한다
9. 실제 VM(JVM, CPython, V8, BEAM)의 설계를 분석한다
10. 메타순환 인터프리터를 이해한다

---

## 목차

1. [인터프리터 vs 컴파일러](#1-인터프리터-vs-컴파일러)
2. [트리 순회 인터프리터](#2-트리-순회-인터프리터)
3. [바이트코드와 바이트코드 컴파일](#3-바이트코드와-바이트코드-컴파일)
4. [스택 기반 가상 머신](#4-스택-기반-가상-머신)
5. [레지스터 기반 가상 머신](#5-레지스터-기반-가상-머신)
6. [명령 디스패치 기법](#6-명령-디스패치-기법)
7. [완전한 바이트코드 컴파일러와 VM](#7-완전한-바이트코드-컴파일러와-vm)
8. [JIT 컴파일](#8-jit-컴파일)
9. [런타임 최적화 기법](#9-런타임-최적화-기법)
10. [실제 가상 머신](#10-실제-가상-머신)
11. [메타순환 인터프리터](#11-메타순환-인터프리터)
12. [요약](#12-요약)
13. [연습 문제](#13-연습-문제)
14. [참고 자료](#14-참고-자료)

---

## 1. 인터프리터 vs 컴파일러

### 1.1 실행 스펙트럼

언어 구현은 순수 해석부터 순수 컴파일까지의 스펙트럼에 존재합니다:

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

### 1.2 트레이드오프

| 측면 | 인터프리터 | 컴파일러 |
|------|-----------|---------|
| **실행 속도** | 느림 (10-100배 느림) | 빠름 (하드웨어 속도에 근접) |
| **시작 시간** | 빠름 (컴파일 단계 없음) | 느림 (먼저 컴파일해야 함) |
| **메모리 사용** | 낮음 (생성된 코드 없음) | 높음 (생성된 코드 + 데이터) |
| **이식성** | 높음 (VM이 하드웨어 추상화) | 낮음 (대상 특화) |
| **에러 메시지** | 더 좋음 (소스 정보 있음) | 종종 난해함 |
| **디버깅** | 쉬움 (라이브 상태 검사) | 어려움 (최적화되어 없어짐) |
| **개발 사이클** | 빠름 (편집-실행) | 느림 (편집-컴파일-실행) |
| **동적 기능** | 쉬움 (`eval`, 메타프로그래밍) | 어렵거나 불가능 |
| **최적화** | 제한적 | 광범위 |

### 1.3 하이브리드 접근

대부분의 현대 시스템은 하이브리드 접근을 사용합니다:

- **Java**: 바이트코드로 컴파일(AOT), 그런 다음 런타임에 핫 메서드를 네이티브 코드로 JIT 컴파일.
- **JavaScript (V8)**: AST로 파싱, 바이트코드로 컴파일, TurboFan으로 핫 함수를 JIT 컴파일.
- **Python (PyPy)**: 바이트코드 해석, 핫 루프 추적, 추적을 JIT 컴파일.
- **.NET**: CIL 바이트코드로 컴파일, 로드 시 또는 지연으로 네이티브 코드로 JIT 컴파일.

---

## 2. 트리 순회 인터프리터

### 2.1 가장 단순한 인터프리터

트리 순회 인터프리터는 AST를 순회하여 프로그램을 실행합니다. 각 노드 유형에는 관련된 평가 규칙이 있습니다.

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

### 2.2 장점과 단점

**장점**:
- 구현이 간단함 (수백 줄)
- AST에 직접 접근 (디버깅, 에러 메시지에 훌륭)
- 기능 추가가 쉬움 (새 `exec_` 메서드만 추가)
- 복잡한 의미론의 언어에 자연스러움

**단점**:
- **느림**: 각 노드에 가상 디스패치(메서드 조회) 필요
- **깊은 재귀**: 깊이 중첩된 표현식은 스택 오버플로우 유발
- **최적화 없음**: 모든 표현식이 매번 재평가됨
- **캐시 비친화적**: AST 노드가 메모리에 분산

트리 순회 인터프리터는 일반적으로 컴파일된 코드보다 50-200배 느립니다.

---

## 3. 바이트코드와 바이트코드 컴파일

### 3.1 바이트코드란?

**바이트코드(bytecode)**는 실제 하드웨어가 아닌 가상 머신에 의해 실행되도록 설계된 프로그램의 컴팩트한 이진 표현입니다. 소스 코드와 기계 코드 사이에 위치합니다:

```
Source Code       AST          Bytecode         Machine Code
  x = a + b  -->  Assign  -->  LOAD_VAR a   -->  mov eax, [rbp-8]
                   / \          LOAD_VAR b       add eax, [rbp-16]
                  x   +         ADD              mov [rbp-24], eax
                     / \        STORE_VAR x
                    a   b
```

### 3.2 바이트코드 설계 원칙

좋은 바이트코드 설계는 여러 목표의 균형을 맞춥니다:

1. **컴팩트성**: 더 적은 바이트 = 더 빠른 로딩, 더 적은 메모리, 더 좋은 캐시 활용
2. **단순성**: 단순한 명령어는 디코딩과 실행이 쉬움
3. **완전성**: 모든 언어 기능을 표현 가능해야 함
4. **성능**: 공통 연산이 효율적이어야 함
5. **검증 가능성**: 실행 전에 유효성 검사가 가능해야 함

### 3.3 명령어 인코딩

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

### 3.4 명령어 집합 정의

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

## 4. 스택 기반 가상 머신

### 4.1 스택 VM 작동 방식

스택 기반 VM은 모든 계산을 위해 피연산자 스택을 사용합니다. 피연산자는 스택에 푸시되고, 연산은 인수를 팝하고 결과를 푸시합니다.

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

### 4.2 스택 VM의 장점

1. **단순한 코드 생성**: 레지스터 할당 불필요
2. **컴팩트 바이트코드**: 명령어가 피연산자 위치를 지정할 필요 없음
3. **구현 용이**: 스택이 자연스러운 평가 순서 제공
4. **이식성**: 하드웨어 레지스터 수에 대한 가정 없음

### 4.3 단점

1. **더 많은 명령어**: `LOAD a; LOAD b; ADD` vs `ADD r1, r2, r3`
2. **메모리 트래픽**: 모든 연산이 스택을 읽고 씀 (레지스터가 아닌 메모리)
3. **최적화 어려움**: 스택 위치가 암묵적이어서 분석이 어려움

### 4.4 기본 스택 VM 구현

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

## 5. 레지스터 기반 가상 머신

### 5.1 레지스터 VM 설계

레지스터 기반 VM은 스택 대신 가상 레지스터를 사용합니다. 명령어는 소스와 대상 레지스터를 명시적으로 지정합니다.

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

### 5.2 레지스터 VM 예제

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

### 5.3 스택 vs 레지스터 비교

| 측면 | 스택 기반 | 레지스터 기반 |
|------|----------|-------------|
| **명령어 수** | 많음 (암묵적 피연산자) | 적음 (명시적 피연산자) |
| **명령어 크기** | 작음 (레지스터 필드 없음) | 큼 (2-3 레지스터 피연산자) |
| **코드 크기** | 종종 더 작음 | 종종 더 큼 |
| **디스패치 횟수** | 더 많음 (더 많은 명령어) | 더 적음 |
| **구현** | 더 단순 | 더 복잡 |
| **최적화** | 어려움 (스택이 암묵적) | 쉬움 (레지스터가 명시적) |
| **예시** | JVM, CPython, CLR, WASM | Lua 5, Dalvik, BEAM |

Shi et al. (2008)의 연구에 따르면 레지스터 기반 VM은 약 47% 더 적은 명령어를 실행하며, 더 큰 코드 크기에도 불구하고 일반적으로 20-30% 더 빠릅니다.

---

## 6. 명령어 디스패치 기법

### 6.1 스위치 디스패치(Switch Dispatch)

가장 단순한 디스패치 메커니즘: 큰 `switch` (또는 `if/elif`) 문.

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

**문제**: CPU의 분기 예측기는 모든 옵코드에 대해 하나의 분기 지점을 봅니다. 이전 옵코드를 기반으로만 다음 옵코드를 예측할 수 있습니다(낮은 정확도).

### 6.2 직접 스레드 코드(Direct Threaded Code)

옵코드 값을 핸들러 코드에 대한 직접 포인터로 교체합니다. 각 핸들러가 실행된 후 다음 핸들러로 직접 점프합니다(중앙 디스패치 루프 없음).

C로 작성 (순수 Python에서는 언어 제한으로 불가):

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

**장점**: 각 핸들러에는 자체 간접 분기가 있어 CPU 분기 예측기에 더 많은 컨텍스트를 제공합니다. 이는 일반적으로 스위치 디스패치보다 15-25% 속도 향상을 제공합니다.

### 6.3 Computed Goto (GCC 확장)

GCC의 `&&label` 확장은 C에서 직접 스레딩을 가능하게 합니다. CPython은 가능할 때 이를 사용합니다:

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

### 6.4 서브루틴 스레딩(Subroutine Threading)

각 바이트코드 명령어가 해당 핸들러 서브루틴에 대한 호출로 컴파일됩니다. 스위치보다 빠르고, 직접 스레딩보다 느립니다 (call/return 오버헤드).

### 6.5 디스패치 성능 비교

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

## 7. 완전한 바이트코드 컴파일러와 VM

이 절에서는 간단한 언어를 위한 완전한 바이트코드 컴파일러와 스택 기반 VM을 구축합니다.

### 7.1 언어

우리의 언어("Mini"라고 불림)는 다음을 지원합니다:
- 정수, 부동소수점, 불리언, 문자열
- 산술 및 비교 연산자
- 변수와 할당
- `if`/`else` 조건문
- `while` 루프
- 매개변수와 반환값이 있는 함수
- Print 문

### 7.2 바이트코드 컴파일러

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

### 7.3 가상 머신

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

### 7.4 모두 합치기

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

## 8. JIT 컴파일

### 8.1 왜 JIT인가?

바이트코드 해석은 디스패치 오버헤드, 레지스터 사용 부족, 전통적인 컴파일러 최적화 불가 등으로 인해 여전히 네이티브 코드보다 5-20배 느립니다. **JIT(Just-In-Time) 컴파일**은 런타임에 바이트코드를 네이티브 코드로 컴파일하여 이 격차를 줄입니다.

```
              Startup Speed  ──────────────────▶  Steady-state Speed
Interpretation  ████████                         ░░░░░░░░
Bytecode        ██████████                       ████████
Method JIT      ████████████                     ████████████████
Tracing JIT     ██████████                       ██████████████████████
AOT Compile     ░░░░                             ████████████████████████
```

### 8.2 메서드 JIT(Method JIT)

**메서드 JIT** 컴파일러는 핫 메서드(자주 호출되는 함수)를 식별하여 네이티브 코드로 컴파일합니다.

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

### 8.3 추적 JIT(Tracing JIT)

**추적 JIT**는 핫 루프를 통한 실제 실행 경로("트레이스")를 기록한 후, 그 특정 경로를 네이티브 코드로 컴파일합니다.

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

### 8.4 트레이스 컴파일 세부 사항

트레이스는 가드(guard)가 있는 선형 연산 시퀀스입니다:

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

가드가 실패하면 트레이스가 중단("사이드 이그짓")되고 인터프리터로 폴백됩니다. 공통 가드 실패는 대체 경로에 대한 새 트레이스를 트리거합니다.

### 8.5 온-스택 대체(On-Stack Replacement, OSR)

**온-스택 대체(OSR)**는 함수 중간(루프 내부도 포함)에서 해석 코드와 컴파일 코드 사이의 전환을 허용합니다. OSR 없이는 컴파일된 버전을 사용하기 전에 인터프리터에서 전체 함수 호출을 완료해야 합니다.

```
Without OSR:
  Interpreter: ████████████████████████████  (entire loop interpreted)
  Next call:   ░░░░ (compiled)

With OSR:
  Interpreter: ████████░░░░░░░░░░░░░░░░░░░  (OSR into compiled code mid-loop)
               ↑ OSR point
```

---

## 9. 런타임 최적화 기법

### 9.1 인라인 캐싱(Inline Caching)

**인라인 캐싱(inline caching)**은 동적 언어에서 메서드 디스패치를 가속합니다. 객체에서 메서드가 처음 호출될 때 런타임이 메서드를 찾고 호출 사이트에 결과를 캐시합니다.

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

### 9.2 타입 특수화(Type Specialization)

JIT 컴파일러는 특정 타입에 대한 특수화된 코드를 생성합니다:

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

### 9.3 역최적화(Deoptimization)

JIT가 세운 가정이 위반될 때(가드 실패), 런타임은 **역최적화(deoptimize)**해야 합니다 -- 컴파일된 코드에서 인터프리터로 폴백합니다.

```
Compiled code:
  GUARD type(x) == int     ──── guard fails ────▶  Deoptimize
  native_int_add(x, y)                              │
  ...                                                ▼
                                                 Interpreter
                                                 (continue execution
                                                  with correct semantics)
```

역최적화에는 다음이 필요합니다:
1. 컴파일된 상태에서 인터프리터 상태(스택, 로컬) 재구성
2. 컴파일된 코드 무효화 (가정이 영구적으로 잘못된 경우)
3. 덜 공격적인 가정으로 재컴파일

### 9.4 숨겨진 클래스(Hidden Classes) / 형태(Shapes)

V8과 다른 VM은 동적 객체에 대한 프로퍼티 접근을 최적화하기 위해 **숨겨진 클래스(hidden classes)**(형태(shapes) 또는 맵(maps)이라고도 함)를 사용합니다:

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

## 10. 실제 가상 머신

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

주요 특성:
- **스택 기반 바이트코드**: 약 200개 옵코드
- **타입화된 명령어**: `iadd` (int), `fadd` (float), `dadd` (double)
- **계층적 컴파일**: 인터프리터 -> C1 (빠른 컴파일) -> C2 (최적화 컴파일)
- **GC**: 여러 컬렉터 (G1 기본, ZGC는 저지연)
- **바이트코드 검증**: 실행 전 타입 안전 바이트코드 검증

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

`dis` 모듈을 사용하여 CPython 바이트코드를 검사할 수 있습니다:

```python
import dis

def example_function(x, y):
    return x * x + y * y

print("=== CPython Bytecode ===")
dis.dis(example_function)
```

출력 (근사치):

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

BEAM은 동시성과 내결함성에 초점을 맞춘 독특한 VM입니다:

- **레지스터 기반**: 프로세스당 1024개의 가상 레지스터
- **경량 프로세스**: 수백만 개의 프로세스, 각각 ~2KB 스택
- **선점형 스케줄링**: 리덕션 기반 (시간 기반이 아님)
- **핫 코드 로딩**: 실행을 멈추지 않고 실행 중인 코드 교체
- **패턴 매칭**: 패턴 매칭을 위한 일급 바이트코드 지원
- **공유 상태 없음**: 프로세스는 메시지 전달로만 통신

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

### 10.5 VM 비교

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

## 11. 메타순환 인터프리터

### 11.1 메타순환 인터프리터란?

**메타순환 인터프리터(metacircular interpreter)**는 같은 언어로 작성된 언어를 위한 인터프리터입니다. Lisp 전통에서 나온 강력한 개념입니다:

- **Lisp in Lisp**: 원래의 메타순환 평가기 (McCarthy, 1960)
- **PyPy**: (Python의 부분 집합으로) Python으로 작성된 Python 인터프리터
- **Truffle/Graal**: Java로 작성된 자기 최적화 인터프리터

### 11.2 간단한 메타순환 평가기

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

### 11.3 메타순환 인터프리터가 중요한 이유

1. **자기 호스팅(Self-hosting)**: 자신을 구현할 수 있는 언어는 완전성과 강력함을 입증합니다.
2. **반영(Reflection)**: 인터프리터가 실행 중인 프로그램에서 사용 가능합니다 (매크로, eval, 인트로스펙션).
3. **부트스트래핑**: 간단한 인터프리터로 시작한 다음 그 언어로 더 나은 인터프리터를 구축합니다.
4. **교육**: 언어 의미론을 가장 명확한 방식으로 보여줍니다.
5. **최적화**: PyPy의 메타-추적 방식은 Python으로 인터프리터를 작성한 다음 자동으로 JIT 컴파일러를 생성합니다.

---

## 12. 요약

이 레슨은 프로그램 실행 전략의 전체 스펙트럼을 다루었습니다:

| 접근 방식 | 속도 | 복잡성 | 주요 예시 |
|---------|------|-------|---------|
| **트리 순회** | 가장 느림 (50-200배) | 단순 | 초기 Ruby, Bash |
| **바이트코드 해석** | 느림 (5-20배) | 보통 | CPython, Lua |
| **바이트코드 + JIT** | 네이티브에 근접 | 복잡 | JVM, V8, PyPy |
| **AOT 컴파일** | 네이티브 | 복잡 | GCC, Rust, Go |

핵심 개념:

- **바이트코드**는 VM 실행을 위해 설계된 컴팩트하고 이식 가능한 중간 형태입니다
- **스택 기반 VM**은 구현이 단순하지만 더 많은 명령어를 생성합니다; **레지스터 기반 VM**은 더 효율적이지만 더 복잡합니다
- **명령어 디스패치** 기법(스위치 vs 스레드 코드 vs computed goto)은 인터프리터 성능에 크게 영향을 미칩니다
- **JIT 컴파일**은 해석과 네이티브 컴파일의 격차를 줄입니다:
  - 메서드 JIT는 핫 함수를 컴파일합니다
  - 추적 JIT는 핫 실행 경로를 컴파일합니다
  - 둘 다 프로파일링, 추측, 역최적화에 의존합니다
- 인라인 캐싱, 타입 특수화, 숨겨진 클래스 같은 **런타임 최적화**는 동적 언어를 정적 언어와 경쟁력 있게 만듭니다
- **메타순환 인터프리터**는 언어의 강력함을 보여주고 메타-추적 같은 고급 기법을 가능하게 합니다

---

## 13. 연습 문제

### 연습 1: 트리 순회 확장

2절의 트리 순회 인터프리터를 다음을 지원하도록 확장하세요:
(a) `push`, `pop`, 인덱싱이 있는 배열.
(b) For 루프: `for i in range(n): body`.
(c) 둘러싼 스코프의 변수를 올바르게 캡처하는 클로저.

다음 클로저 목록을 생성하는 프로그램으로 테스트하세요:
```
funcs = []
for i in range(5):
    define make_adder(x): return lambda(y): x + y
    push(funcs, make_adder(i))
print(funcs[3](10))  # Should print 13
```

### 연습 2: 바이트코드 최적화

다음을 처리하는 바이트코드 핍홀 옵티마이저를 구현하세요:
(a) 상수 폴딩: `CONST 3; CONST 4; ADD` -> `CONST 7`
(b) 죽은 저장 제거: `STORE x; STORE x` -> 두 번째만 유지
(c) 중복 로드 제거: `STORE x; LOAD x` -> `STORE x; DUP`

7절의 컴파일러 출력에 적용하고 명령어 수 감소를 측정하세요.

### 연습 3: 레지스터 VM을 위한 레지스터 할당

스택 기반 바이트코드를 레지스터 기반 바이트코드로 변환하는 간단한 레지스터 할당기를 구현하세요:
(a) 간단한 스택 시뮬레이션을 사용하여 어떤 값이 어떤 레지스터에 있는지 추적합니다.
(b) 레지스터가 부족할 때 레지스터 스필링을 처리합니다.
(c) 여러 프로그램에 대해 스택과 레지스터 버전의 명령어 수를 비교합니다.

### 연습 4: 간단한 JIT 컴파일러

다음을 구현하는 간단한 JIT 유사 시스템을 구현하세요:
(a) 가장 자주 호출되는 함수를 프로파일링합니다.
(b) "핫" 함수에 대해 디스패치 오버헤드를 제거하는 특수화된 Python 함수를 (`exec` 사용) 생성합니다.
(c) 해석 vs "JIT 컴파일" 버전을 벤치마크합니다.

### 연습 5: VM 디버거

7절의 VM에 디버깅 지원을 추가하세요:
(a) 단일 단계 실행 (명령어 하나 실행 후 일시 정지).
(b) 브레이크포인트 (특정 명령어 인덱스에 도달 시 일시 정지).
(c) 어느 지점에서든 스택과 지역 변수 검사.
(d) 콜 스택 표시 (함수 호출 체인 표시).

### 연습 6: VM 성능 분석

실행 통계를 수집하는 VM 계측:
(a) 각 옵코드가 실행되는 횟수를 세어봅니다.
(b) 가장 일반적인 옵코드 쌍(슈퍼명령어)을 식별합니다.
(c) 공통 쌍을 결합하는 슈퍼명령어 3개를 제안하고 구현합니다.
(d) 디스패치 수와 실행 시간의 감소를 측정합니다.

---

## 14. 참고 자료

1. Aho, A. V., Lam, M. S., Sethi, R., & Ullman, J. D. (2006). *Compilers: Principles, Techniques, and Tools* (2nd ed.), Chapter 8.
2. Nystrom, R. (2021). *Crafting Interpreters*. Available at [craftinginterpreters.com](https://craftinginterpreters.com/).
3. Smith, J. E., & Nair, R. (2005). *Virtual Machines: Versatile Platforms for Systems and Processes*. Morgan Kaufmann.
4. Ertl, M. A., & Gregg, D. (2003). "The Structure and Performance of Efficient Interpreters." *Journal of Instruction-Level Parallelism*, 5.
5. Shi, Y., Gregg, D., Beatty, A., & Ertl, M. A. (2008). "Virtual Machine Showdown: Stack versus Registers." *ACM TOPLAS*, 30(4).
6. Bolz, C. F., Cuni, A., Fijalkowski, M., & Rigo, A. (2009). "Tracing the Meta-level: PyPy's Tracing JIT Compiler." *ICOOOLPS*.
7. Holzle, U., Chambers, C., & Ungar, D. (1991). "Optimizing Dynamically-Typed Object-Oriented Languages With Polymorphic Inline Caches." *ECOOP*.
8. Deutsch, L. P., & Schiffman, A. M. (1984). "Efficient Implementation of the Smalltalk-80 System." *POPL*.

---

[이전: 14. 가비지 컬렉션](./14_Garbage_Collection.md) | [다음: 16. 현대 컴파일러 인프라](./16_Modern_Compiler_Infrastructure.md) | [개요](./00_Overview.md)
