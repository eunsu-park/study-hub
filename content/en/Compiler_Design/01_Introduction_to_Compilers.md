# Lesson 1: Introduction to Compilers

## Learning Objectives

After completing this lesson, you will be able to:

1. Define what a compiler is and distinguish it from an interpreter
2. Trace the historical development of compilers from Fortran to modern systems
3. Identify and describe each phase of the compilation process
4. Explain the distinction between front-end and back-end
5. Understand single-pass vs. multi-pass compilation strategies
6. Describe bootstrapping and cross-compilation
7. Read and construct T-diagrams (tombstone diagrams)
8. Trace a simple program through every phase of compilation

---

## 1. What Is a Compiler?

A **compiler** is a program that translates source code written in one language (the **source language**) into another language (the **target language**), while reporting errors detected during the translation process.

$$\text{Compiler}: \text{Source Language} \longrightarrow \text{Target Language}$$

More precisely, a compiler is a function that maps programs:

$$C: \mathcal{P}_S \rightarrow \mathcal{P}_T$$

where $\mathcal{P}_S$ is the set of valid programs in the source language and $\mathcal{P}_T$ is the set of programs in the target language, such that for every input $x$:

$$\text{meaning}(P_S, x) = \text{meaning}(C(P_S), x)$$

The **semantic equivalence** requirement is crucial: the compiled program must produce the same observable behavior as the source program for all valid inputs.

### Common Source-Target Pairs

| Source Language | Target Language | Example |
|-----------------|-----------------|---------|
| C | x86 machine code | GCC, Clang |
| Java | JVM bytecode | javac |
| TypeScript | JavaScript | tsc |
| Python | Python bytecode (.pyc) | CPython |
| Rust | LLVM IR, then machine code | rustc |
| SASS | CSS | sass compiler |
| LaTeX | DVI/PDF | pdflatex |

### Compiler vs. Transpiler

A **transpiler** (source-to-source compiler) translates between languages at a similar level of abstraction. TypeScript to JavaScript and CoffeeScript to JavaScript are transpilation. The boundary between compilation and transpilation is blurry; what matters is that both preserve semantics.

---

## 2. Compiler vs. Interpreter

An **interpreter** executes the source program directly, without producing a separate target program.

```
Compiler workflow:
  Source Code  --->  [Compiler]  --->  Target Code  --->  [CPU/VM]  --->  Output
                                          (stored)

Interpreter workflow:
  Source Code  --->  [Interpreter + Input]  --->  Output
                     (no separate target)
```

### Key Differences

| Aspect | Compiler | Interpreter |
|--------|----------|-------------|
| Output | Produces target program | No target program produced |
| Execution | Target runs independently | Interpreter must be present |
| Speed (startup) | Slow (compilation overhead) | Fast (immediate execution) |
| Speed (runtime) | Fast (optimized target code) | Slow (repeated analysis) |
| Error reporting | All errors before execution | Errors at runtime |
| Memory | Target program stored | Source re-analyzed each time |

### Hybrid Approaches

Most modern language implementations use a **hybrid** approach:

1. **Java**: Compile to bytecode (javac), then interpret/JIT-compile on JVM
2. **Python**: Compile to bytecode (.pyc), then interpret on CPython VM
3. **JavaScript**: Parse, then JIT-compile hot paths (V8, SpiderMonkey)
4. **C#/.NET**: Compile to CIL, then JIT-compile to native code

```python
# Python's hybrid approach: you can see the bytecode
import dis

def add(a, b):
    return a + b

# Show the compiled bytecode instructions
dis.dis(add)
# Output:
#   2           0 LOAD_FAST                0 (a)
#               2 LOAD_FAST                1 (b)
#               4 BINARY_ADD
#               6 RETURN_VALUE
```

### The Compilation Spectrum

Rather than a binary distinction, there is a spectrum:

```
Pure Interpreter  <----------------------------->  Ahead-of-Time Compiler
     |                    |              |                |
   Shell scripts    CPython (bytecode)  JIT (V8, JVM)   GCC/Clang
```

---

## 3. Historical Development

### The Early Days (1950s)

Before compilers, programmers wrote machine code or assembly language by hand. The idea that a machine could translate high-level notation into efficient machine code was met with skepticism.

**1957 -- FORTRAN Compiler** (John Backus, IBM)
- The first optimizing compiler
- Took 18 person-years to develop
- Goal: produce code as efficient as hand-written assembly
- Success proved that high-level languages were practical

**1960 -- COBOL Compiler**
- Grace Hopper's work on A-0 (1952) laid groundwork
- First language designed for portability across machines

**1962 -- LISP Compiler**
- First self-hosting compiler (compiler written in the language it compiles)
- Introduced garbage collection

### Formal Foundations (1960s-1970s)

**1960s -- Chomsky Hierarchy and Parsing Theory**
- Noam Chomsky's classification of formal languages
- Direct application to compiler design
- Development of context-free grammars for syntax

**1965 -- Knuth's LR Parsing**
- Donald Knuth introduced LR parsing
- Provided a systematic way to build parsers for deterministic context-free languages

**1970 -- Yacc (Yet Another Compiler-Compiler)**
- Stephen C. Johnson at Bell Labs
- LALR(1) parser generator
- Made parser construction practical

**1975 -- Lex**
- Mike Lesk and Eric Schmidt at Bell Labs
- Lexical analyzer generator
- Works with Yacc: Lex handles tokens, Yacc handles grammar

### Modern Era (1980s-Present)

**1987 -- GCC (GNU Compiler Collection)**
- Richard Stallman
- First major open-source compiler
- Supports multiple languages and targets

**2003 -- LLVM**
- Chris Lattner at UIUC
- Modular compiler infrastructure
- Revolutionized compiler construction with reusable components

**2010s -- Rust, Go, Swift**
- Languages designed with modern compiler technology
- Rust: ownership-based memory safety (no GC)
- Go: fast compilation as a design goal
- Swift: built on LLVM

**2020s -- AI-Assisted Compilation**
- Machine learning for optimization heuristics
- Neural program synthesis
- LLM-based code generation (a different paradigm from traditional compilation)

---

## 4. Phases of Compilation

A compiler is organized as a sequence of **phases**, each of which transforms the program from one representation to another.

```
Source Code (characters)
    |
    v
+-------------------+
| Lexical Analysis  |  (Scanner / Lexer)
+-------------------+
    |  tokens
    v
+-------------------+
| Syntax Analysis   |  (Parser)
+-------------------+
    |  parse tree / AST
    v
+-------------------+
| Semantic Analysis |  (Type checker)
+-------------------+
    |  annotated AST
    v
+-------------------+
| IR Generation     |  (Intermediate Representation)
+-------------------+
    |  IR (three-address code, SSA, ...)
    v
+-------------------+
| Optimization      |  (Machine-independent)
+-------------------+
    |  optimized IR
    v
+-------------------+
| Code Generation   |  (Machine-dependent)
+-------------------+
    |  target code (assembly / machine code)
    v
+-------------------+
| Target Optimization |  (Peephole, scheduling)
+-------------------+
    |
    v
Target Code (machine instructions)
```

Two additional components run throughout all phases:

- **Symbol Table Manager**: Maintains information about identifiers (names, types, scopes)
- **Error Handler**: Detects, reports, and (where possible) recovers from errors

### Phase 1: Lexical Analysis (Scanning)

The **lexer** (or scanner) reads the source program as a stream of characters and groups them into **tokens** -- the smallest meaningful units of the language.

**Input**: Stream of characters
**Output**: Stream of tokens

```
Source:  position = initial + rate * 60

Tokens:  [ID:"position"] [ASSIGN:"="] [ID:"initial"] [PLUS:"+"]
         [ID:"rate"] [STAR:"*"] [INT:"60"]
```

Each token has:
- A **token type** (or token class): `ID`, `INT`, `ASSIGN`, `PLUS`, etc.
- An optional **attribute value**: the actual lexeme or its value

The lexer also:
- Strips whitespace and comments
- Handles preprocessing directives (in C)
- Reports lexical errors (illegal characters, unterminated strings)

### Phase 2: Syntax Analysis (Parsing)

The **parser** takes the token stream and builds a **parse tree** (or directly an AST) according to the grammar rules of the language.

**Input**: Stream of tokens
**Output**: Parse tree or AST

```
Parse tree for: position = initial + rate * 60

        assignment_stmt
       /       |       \
     ID       '='      expr
  "position"          /  |  \
                   expr  '+'  term
                    |        / | \
                   ID    term '*' factor
                "initial"  |       |
                          ID     INT
                        "rate"   60
```

The parser:
- Verifies that the token sequence conforms to the language grammar
- Reports syntax errors ("expected `;` before `}`")
- May perform error recovery to continue parsing after errors

### Phase 3: Semantic Analysis

**Semantic analysis** checks the program for semantic consistency that cannot be captured by context-free grammars.

**Input**: AST
**Output**: Annotated/decorated AST

Key tasks:
- **Type checking**: Is `rate * 60` valid? (Can you multiply a float by an int?)
- **Type coercion**: Insert implicit conversions (int 60 -> float 60.0)
- **Name resolution**: Which declaration does each use of `rate` refer to?
- **Scope checking**: Is `position` visible in this scope?
- **Definite assignment**: Is `initial` initialized before use?

```
Annotated AST:

        assignment_stmt
       /       |       \
     ID       '='      expr : float
  "position"          /  |  \
  (type:float)     expr  '+'  term : float
                    |        / | \
                   ID    term '*' factor : float
                "initial"  |       |
               (float)    ID    intToFloat(INT)
                        "rate"     60 -> 60.0
                       (float)
```

### Phase 4: Intermediate Representation (IR) Generation

The compiler generates a machine-independent **intermediate representation** of the program.

**Input**: Annotated AST
**Output**: IR (e.g., three-address code)

```
Three-address code for: position = initial + rate * 60

    t1 = intToFloat(60)
    t2 = rate * t1
    t3 = initial + t2
    position = t3
```

Each instruction has at most three operands (hence "three-address code"). This flat representation is much easier to optimize than a tree.

### Phase 5: Optimization

The **optimizer** transforms the IR to improve performance without changing the program's semantics.

**Input**: IR
**Output**: Optimized IR

```
Before optimization:           After optimization:
    t1 = intToFloat(60)           t1 = rate * 60.0
    t2 = rate * t1                position = initial + t1
    t3 = initial + t2
    position = t3
```

Optimizations applied:
- **Constant folding**: `intToFloat(60)` -> `60.0` (computed at compile time)
- **Dead code elimination**: removed unnecessary temporaries
- **Copy propagation**: replaced uses of t3 with its definition

### Phase 6: Code Generation

The **code generator** maps the optimized IR to the target machine's instruction set.

**Input**: Optimized IR
**Output**: Target code (assembly or machine code)

```asm
; x86-64 assembly for: position = initial + rate * 60.0
    movsd   xmm0, QWORD PTR [rate]       ; load rate into xmm0
    mulsd   xmm0, QWORD PTR [.LC0]       ; xmm0 = rate * 60.0
    addsd   xmm0, QWORD PTR [initial]    ; xmm0 = initial + rate * 60.0
    movsd   QWORD PTR [position], xmm0   ; store result in position
.LC0:
    .double 60.0
```

The code generator must:
- Select appropriate instructions (**instruction selection**)
- Assign variables to registers (**register allocation**)
- Determine the order of instructions (**instruction scheduling**)

---

## 5. A Complete Example: From Source to Target

Let us trace the expression `a = b + c * 2` through every phase.

### Source Code

```c
int a, b, c;
b = 3;
c = 5;
a = b + c * 2;
```

### Phase 1: Lexical Analysis

```
Token Stream:
  [KW_INT:"int"] [ID:"a"] [COMMA:","] [ID:"b"] [COMMA:","] [ID:"c"] [SEMI:";"]
  [ID:"b"] [ASSIGN:"="] [INT:"3"] [SEMI:";"]
  [ID:"c"] [ASSIGN:"="] [INT:"5"] [SEMI:";"]
  [ID:"a"] [ASSIGN:"="] [ID:"b"] [PLUS:"+"] [ID:"c"] [STAR:"*"] [INT:"2"] [SEMI:";"]
```

### Phase 2: Syntax Analysis (AST)

```
Program
 |- VarDecl(type=int, names=[a, b, c])
 |- Assign(target=b, value=IntLit(3))
 |- Assign(target=c, value=IntLit(5))
 |- Assign(target=a,
           value=BinOp(+,
                        left=Var(b),
                        right=BinOp(*,
                                     left=Var(c),
                                     right=IntLit(2))))
```

### Phase 3: Semantic Analysis

```
Symbol Table:
  a : int (declared, scope=global)
  b : int (declared, scope=global)
  c : int (declared, scope=global)

Type checking:
  b + c * 2  =>  int + (int * int)  =>  int + int  =>  int  ✓
  a = (int)  =>  int = int  ✓
```

### Phase 4: IR Generation (Three-Address Code)

```
    b = 3
    c = 5
    t1 = c * 2
    t2 = b + t1
    a = t2
```

### Phase 5: Optimization

```
Constant propagation: b=3, c=5 are known constants

    t1 = 5 * 2     (c replaced with 5)
    t2 = 3 + t1    (b replaced with 3)
    a = t2

Constant folding:

    t1 = 10         (5 * 2 computed at compile time)
    t2 = 13         (3 + 10 computed at compile time)
    a = 13

Copy propagation + dead code elimination:

    a = 13
```

The entire computation was resolved at compile time.

### Phase 6: Code Generation

```asm
; x86-64
    mov DWORD PTR [a], 13
```

### Python Simulation of All Phases

```python
"""
A simplified demonstration of compiler phases.
This processes the expression: a = b + c * 2
"""

# ============================================================
# Phase 1: Lexical Analysis
# ============================================================

import re
from dataclasses import dataclass
from typing import List, Optional, Any

@dataclass
class Token:
    type: str
    value: str
    line: int = 0
    col: int = 0

def tokenize(source: str) -> List[Token]:
    """Simple lexer for a tiny language."""
    token_spec = [
        ('INT',     r'\d+'),
        ('ID',      r'[a-zA-Z_]\w*'),
        ('ASSIGN',  r'='),
        ('PLUS',    r'\+'),
        ('STAR',    r'\*'),
        ('SEMI',    r';'),
        ('COMMA',   r','),
        ('SKIP',    r'[ \t]+'),
        ('NEWLINE', r'\n'),
        ('MISMATCH', r'.'),
    ]
    keywords = {'int'}
    tok_regex = '|'.join(f'(?P<{name}>{pattern})' for name, pattern in token_spec)
    tokens = []
    for mo in re.finditer(tok_regex, source):
        kind = mo.lastgroup
        value = mo.group()
        if kind == 'SKIP' or kind == 'NEWLINE':
            continue
        if kind == 'MISMATCH':
            raise SyntaxError(f'Unexpected character: {value!r}')
        if kind == 'ID' and value in keywords:
            kind = f'KW_{value.upper()}'
        tokens.append(Token(kind, value))
    return tokens

source = """int a, b, c;
b = 3;
c = 5;
a = b + c * 2;"""

tokens = tokenize(source)
print("=== Phase 1: Lexical Analysis ===")
for tok in tokens:
    print(f"  {tok.type:10s} {tok.value!r}")


# ============================================================
# Phase 2: Syntax Analysis (Build AST)
# ============================================================

@dataclass
class ASTNode:
    """Base class for AST nodes."""
    pass

@dataclass
class IntLiteral(ASTNode):
    value: int

@dataclass
class Identifier(ASTNode):
    name: str
    resolved_type: Optional[str] = None

@dataclass
class BinOp(ASTNode):
    op: str
    left: ASTNode
    right: ASTNode
    resolved_type: Optional[str] = None

@dataclass
class Assignment(ASTNode):
    target: str
    value: ASTNode

@dataclass
class VarDecl(ASTNode):
    type_name: str
    names: List[str]

@dataclass
class Program(ASTNode):
    statements: List[ASTNode]

class Parser:
    """Recursive descent parser for our tiny language."""

    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.pos = 0

    def peek(self) -> Optional[Token]:
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return None

    def consume(self, expected_type: str = None) -> Token:
        tok = self.tokens[self.pos]
        if expected_type and tok.type != expected_type:
            raise SyntaxError(
                f"Expected {expected_type}, got {tok.type} ({tok.value!r})"
            )
        self.pos += 1
        return tok

    def parse_program(self) -> Program:
        stmts = []
        while self.pos < len(self.tokens):
            stmts.append(self.parse_statement())
        return Program(stmts)

    def parse_statement(self):
        tok = self.peek()
        if tok.type == 'KW_INT':
            return self.parse_var_decl()
        else:
            return self.parse_assignment()

    def parse_var_decl(self) -> VarDecl:
        self.consume('KW_INT')
        names = [self.consume('ID').value]
        while self.peek() and self.peek().type == 'COMMA':
            self.consume('COMMA')
            names.append(self.consume('ID').value)
        self.consume('SEMI')
        return VarDecl('int', names)

    def parse_assignment(self) -> Assignment:
        name = self.consume('ID').value
        self.consume('ASSIGN')
        expr = self.parse_expr()
        self.consume('SEMI')
        return Assignment(name, expr)

    def parse_expr(self) -> ASTNode:
        """expr = term (('+') term)*"""
        left = self.parse_term()
        while self.peek() and self.peek().type == 'PLUS':
            self.consume('PLUS')
            right = self.parse_term()
            left = BinOp('+', left, right)
        return left

    def parse_term(self) -> ASTNode:
        """term = factor (('*') factor)*"""
        left = self.parse_factor()
        while self.peek() and self.peek().type == 'STAR':
            self.consume('STAR')
            right = self.parse_factor()
            left = BinOp('*', left, right)
        return left

    def parse_factor(self) -> ASTNode:
        """factor = INT | ID"""
        tok = self.peek()
        if tok.type == 'INT':
            self.consume()
            return IntLiteral(int(tok.value))
        elif tok.type == 'ID':
            self.consume()
            return Identifier(tok.value)
        else:
            raise SyntaxError(f"Unexpected token: {tok}")

parser = Parser(tokens)
ast = parser.parse_program()

def print_ast(node, indent=0):
    prefix = "  " * indent
    if isinstance(node, Program):
        print(f"{prefix}Program")
        for s in node.statements:
            print_ast(s, indent + 1)
    elif isinstance(node, VarDecl):
        print(f"{prefix}VarDecl(type={node.type_name}, names={node.names})")
    elif isinstance(node, Assignment):
        print(f"{prefix}Assign(target={node.target})")
        print_ast(node.value, indent + 1)
    elif isinstance(node, BinOp):
        print(f"{prefix}BinOp({node.op})")
        print_ast(node.left, indent + 1)
        print_ast(node.right, indent + 1)
    elif isinstance(node, IntLiteral):
        print(f"{prefix}IntLit({node.value})")
    elif isinstance(node, Identifier):
        print(f"{prefix}Id({node.name})")

print("\n=== Phase 2: Syntax Analysis (AST) ===")
print_ast(ast)


# ============================================================
# Phase 3: Semantic Analysis
# ============================================================

class SemanticAnalyzer:
    def __init__(self):
        self.symbol_table = {}  # name -> type

    def analyze(self, node: ASTNode):
        if isinstance(node, Program):
            for stmt in node.statements:
                self.analyze(stmt)
        elif isinstance(node, VarDecl):
            for name in node.names:
                if name in self.symbol_table:
                    raise NameError(f"Variable '{name}' already declared")
                self.symbol_table[name] = node.type_name
        elif isinstance(node, Assignment):
            if node.target not in self.symbol_table:
                raise NameError(f"Variable '{node.target}' not declared")
            expr_type = self.check_type(node.value)
            target_type = self.symbol_table[node.target]
            if expr_type != target_type:
                raise TypeError(
                    f"Cannot assign {expr_type} to {target_type} variable '{node.target}'"
                )
        return node

    def check_type(self, node: ASTNode) -> str:
        if isinstance(node, IntLiteral):
            return 'int'
        elif isinstance(node, Identifier):
            if node.name not in self.symbol_table:
                raise NameError(f"Undeclared variable '{node.name}'")
            node.resolved_type = self.symbol_table[node.name]
            return node.resolved_type
        elif isinstance(node, BinOp):
            left_type = self.check_type(node.left)
            right_type = self.check_type(node.right)
            if left_type != right_type:
                raise TypeError(
                    f"Type mismatch: {left_type} {node.op} {right_type}"
                )
            node.resolved_type = left_type
            return left_type
        raise TypeError(f"Unknown node type: {type(node)}")

analyzer = SemanticAnalyzer()
analyzer.analyze(ast)

print("\n=== Phase 3: Semantic Analysis ===")
print("Symbol table:", analyzer.symbol_table)
print("Type checking passed!")


# ============================================================
# Phase 4: IR Generation (Three-Address Code)
# ============================================================

class IRGenerator:
    def __init__(self):
        self.instructions = []
        self.temp_count = 0

    def new_temp(self) -> str:
        self.temp_count += 1
        return f"t{self.temp_count}"

    def generate(self, node: ASTNode):
        if isinstance(node, Program):
            for stmt in node.statements:
                self.generate(stmt)
        elif isinstance(node, VarDecl):
            pass  # Declarations don't generate code
        elif isinstance(node, Assignment):
            result = self.gen_expr(node.value)
            self.instructions.append(f"{node.target} = {result}")

    def gen_expr(self, node: ASTNode) -> str:
        if isinstance(node, IntLiteral):
            return str(node.value)
        elif isinstance(node, Identifier):
            return node.name
        elif isinstance(node, BinOp):
            left = self.gen_expr(node.left)
            right = self.gen_expr(node.right)
            temp = self.new_temp()
            self.instructions.append(f"{temp} = {left} {node.op} {right}")
            return temp
        raise ValueError(f"Unknown node: {type(node)}")

ir_gen = IRGenerator()
ir_gen.generate(ast)

print("\n=== Phase 4: IR Generation ===")
for instr in ir_gen.instructions:
    print(f"  {instr}")


# ============================================================
# Phase 5: Optimization (Constant Folding + Propagation)
# ============================================================

def optimize(instructions: List[str]) -> List[str]:
    """Simple constant folding and propagation optimizer."""
    constants = {}  # variable -> known constant value
    optimized = []

    for instr in instructions:
        parts = instr.split(' = ', 1)
        target = parts[0].strip()
        expr = parts[1].strip()

        # Substitute known constants
        for var, val in constants.items():
            # Replace variable with its constant value (whole word only)
            import re as _re
            expr = _re.sub(r'\b' + _re.escape(var) + r'\b', str(val), expr)

        # Try to evaluate the expression (constant folding)
        try:
            # Only evaluate if it looks like a pure arithmetic expression
            result = eval(expr)
            if isinstance(result, (int, float)):
                constants[target] = result
                optimized.append(f"{target} = {result}")
                continue
        except:
            pass

        optimized.append(f"{target} = {expr}")

    # Remove assignments to temporaries that are only used once
    # (simplified dead code elimination)
    final = []
    used_vars = set()
    for instr in reversed(optimized):
        parts = instr.split(' = ', 1)
        target = parts[0].strip()
        if target.startswith('t') and target not in used_vars:
            continue  # Dead temporary
        final.append(instr)
        # Track variable uses in the expression
        for word in re.findall(r'\b[a-zA-Z_]\w*\b', parts[1]):
            used_vars.add(word)
    final.reverse()

    return final

optimized_ir = optimize(ir_gen.instructions)

print("\n=== Phase 5: Optimization ===")
for instr in optimized_ir:
    print(f"  {instr}")


# ============================================================
# Phase 6: Code Generation (Simple Stack Machine)
# ============================================================

def generate_code(instructions: List[str]) -> List[str]:
    """Generate simple stack-machine instructions."""
    code = []
    for instr in instructions:
        parts = instr.split(' = ', 1)
        target = parts[0].strip()
        expr = parts[1].strip()

        # Try to parse as "left op right"
        match = re.match(r'(\w+)\s*([+\-*/])\s*(\w+)', expr)
        if match:
            left, op, right = match.groups()
            code.append(f"  LOAD  {left}")
            code.append(f"  LOAD  {right}")
            op_map = {'+': 'ADD', '-': 'SUB', '*': 'MUL', '/': 'DIV'}
            code.append(f"  {op_map[op]}")
            code.append(f"  STORE {target}")
        else:
            # Simple assignment: target = value
            code.append(f"  LOAD  {expr}")
            code.append(f"  STORE {target}")
    return code

target_code = generate_code(optimized_ir)

print("\n=== Phase 6: Code Generation (Stack Machine) ===")
for line in target_code:
    print(line)
```

Running this produces:

```
=== Phase 1: Lexical Analysis ===
  KW_INT     'int'
  ID         'a'
  COMMA      ','
  ID         'b'
  ...

=== Phase 2: Syntax Analysis (AST) ===
Program
  VarDecl(type=int, names=['a', 'b', 'c'])
  Assign(target=b)
    IntLit(3)
  Assign(target=c)
    IntLit(5)
  Assign(target=a)
    BinOp(+)
      Id(b)
      BinOp(*)
        Id(c)
        IntLit(2)

=== Phase 3: Semantic Analysis ===
Symbol table: {'a': 'int', 'b': 'int', 'c': 'int'}
Type checking passed!

=== Phase 4: IR Generation ===
  b = 3
  c = 5
  t1 = c * 2
  t2 = b + t1
  a = t2

=== Phase 5: Optimization ===
  a = 13

=== Phase 6: Code Generation (Stack Machine) ===
  LOAD  13
  STORE a
```

The optimizer reduced the entire computation to a single constant assignment.

---

## 6. Front-End vs. Back-End

The compiler is logically divided into two halves:

```
                    Front-End                    Back-End
              (language-dependent)          (machine-dependent)
         +---------------------------+  +---------------------------+
Source -->| Lexer | Parser | Semantic |->| Optimizer | Code Gen     |--> Target
         | Analysis       | Analysis |  |           | Register Alloc|
         +---------------------------+  +---------------------------+
                     |                              |
                     v                              v
               Intermediate                    Target Code
              Representation (IR)
```

### Front-End

The front-end is **source-language dependent** and **target-machine independent**:

- Lexical analysis
- Syntax analysis
- Semantic analysis
- IR generation

The front-end produces an intermediate representation that captures the meaning of the program without committing to any particular target machine.

### Back-End

The back-end is **source-language independent** and **target-machine dependent**:

- Machine-independent optimization
- Machine-dependent optimization
- Instruction selection
- Register allocation
- Instruction scheduling

### The $M \times N$ Problem

Without an IR, supporting $M$ source languages and $N$ target machines requires $M \times N$ compilers. With a shared IR, you need only $M$ front-ends and $N$ back-ends:

```
Without IR:  M x N compilers needed

  C    ---->  x86
  C    ---->  ARM
  C    ---->  MIPS
  Java ---->  x86
  Java ---->  ARM
  Java ---->  MIPS
  Rust ---->  x86
  Rust ---->  ARM
  Rust ---->  MIPS

  3 languages x 3 targets = 9 compilers


With IR:  M + N components needed

  C    ---->  \
  Java ---->   ----> IR ----> x86
  Rust ---->  /          ---> ARM
                         ---> MIPS

  3 front-ends + 3 back-ends = 6 components
```

This is exactly the design philosophy behind **LLVM**: many front-ends (Clang for C/C++, rustc for Rust, swiftc for Swift) share a common IR (LLVM IR) and common back-ends.

---

## 7. Single-Pass vs. Multi-Pass Compilation

### Single-Pass Compiler

A **single-pass compiler** processes the source code exactly once, performing all phases simultaneously as it reads the input from left to right.

**Advantages**:
- Fast compilation (only one pass over the source)
- Low memory usage
- Simple implementation

**Disadvantages**:
- Limited optimization opportunities
- Language must be designed for single-pass compilation (e.g., forward declarations in C/Pascal)
- Cannot perform global analysis

**Example**: Early Pascal compilers were single-pass. This is why Pascal requires forward declarations:

```pascal
{ Pascal: must declare before use }
procedure B; forward;  { forward declaration needed }

procedure A;
begin
    B;  { calls B, which hasn't been fully defined yet }
end;

procedure B;
begin
    A;
end;
```

### Multi-Pass Compiler

A **multi-pass compiler** processes the source code multiple times, each pass performing a specific transformation.

**Advantages**:
- Better optimization (global view of the program)
- Cleaner separation of concerns
- Language doesn't need forward declarations

**Disadvantages**:
- Slower compilation
- Higher memory usage (must store intermediate representations)

**Example**: Modern compilers like GCC and Clang use many passes:

```
GCC compilation passes (simplified):
  Pass 1: Preprocessing (cpp)
  Pass 2: Parsing -> GENERIC IR
  Pass 3: GENERIC -> GIMPLE (lowering)
  Pass 4: GIMPLE optimizations (SSA, constant prop, DCE, ...)
  Pass 5: GIMPLE -> RTL (Register Transfer Language)
  Pass 6: RTL optimizations (CSE, loop opt, register alloc)
  Pass 7: RTL -> Assembly
```

---

## 8. Bootstrapping

**Bootstrapping** is the process of writing a compiler for a language using the language itself. This creates a chicken-and-egg problem: how do you compile the compiler if the compiler doesn't exist yet?

### The Bootstrap Process

```
Step 1: Write a simple compiler C0 for language L in an existing language E
        (e.g., write a C compiler in assembly)

        C0 written in E,  compiles subset of L

Step 2: Use C0 to compile a better compiler C1 written in L
        (e.g., write a C compiler in C, compile it with C0)

        C1 written in L,  compiled by C0

Step 3: Use C1 to recompile itself
        (the compiler compiles its own source code)

        C1 written in L,  compiled by C1

Step 4: Iterate to improve the compiler
        C2, C3, ... each version compiled by the previous
```

### Historical Example: The First C Compiler

Ken Thompson and Dennis Ritchie bootstrapped C as follows:

1. Wrote a minimal C compiler in PDP-11 assembly (B language actually preceded C)
2. Wrote a better C compiler in C
3. Compiled the C compiler with the assembly version
4. Recompiled the C compiler with itself

### Trust Problem (Thompson's Hack)

In his 1984 Turing Award lecture "Reflections on Trusting Trust," Ken Thompson demonstrated that a bootstrapped compiler can contain hidden backdoors:

1. Modify the compiler to insert a backdoor when compiling a specific program (e.g., `login`)
2. Modify the compiler to insert modification (1) when compiling *itself*
3. Remove the modifications from the source code
4. The compiled compiler binary still contains both modifications, even though the source is clean

This is why **reproducible builds** and **diverse double-compilation** are important for security.

---

## 9. Cross-Compilation

A **cross-compiler** runs on one platform (the **host**) but generates code for a different platform (the **target**).

$$\text{Cross-compiler}: \text{Source} \xrightarrow{\text{runs on Host}} \text{Target code for Target platform}$$

### Terminology

We use three terms:
- **Build**: The platform where the compiler is built
- **Host**: The platform where the compiler runs
- **Target**: The platform for which the compiler generates code

| Build = Host = Target | Name |
|-----------------------|------|
| All same | Native compiler |
| Build = Host, different Target | Cross-compiler |
| All different | Canadian cross |

### Use Cases

- **Embedded systems**: Develop on x86 PC, deploy on ARM microcontroller
- **Mobile development**: Compile on macOS, run on iOS/Android
- **Operating systems**: Compile kernel for a platform that has no compiler yet

```bash
# Example: Cross-compiling C for ARM on an x86 Linux host
arm-linux-gnueabihf-gcc -o hello hello.c

# The resulting 'hello' binary runs on ARM, not on the x86 host
file hello
# hello: ELF 32-bit LSB executable, ARM, EABI5, ...
```

---

## 10. T-Diagrams (Tombstone Diagrams)

**T-diagrams** are a visual notation for describing compilers, interpreters, and programs in terms of their source language, target language, and implementation language.

### Notation

A **compiler** is drawn as a T-shape:

```
+-------------------+
|  Source  | Target  |
+----+-----+----+---+
     | Impl     |
     +----------+
```

- **Source**: Language the compiler accepts
- **Target**: Language the compiler produces
- **Impl**: Language the compiler is written in

A **program** is drawn as an inverted trapezoid:

```
+---------+
| Program |
+----+----+
     | Lang
     +----+
```

An **interpreter** is drawn as a rounded T:

```
+-------------------+
|  Language          |
+----+---------+----+
     | Impl    |
     +---------+
```

### Example: Bootstrapping with T-Diagrams

**Step 1**: Write a C compiler in assembly that targets x86.

```
+-------------------+
|    C     |  x86   |
+----+-----+----+---+
     |   ASM    |
     +----------+
```

**Step 2**: Write a C compiler in C that targets x86.

```
+-------------------+
|    C     |  x86   |
+----+-----+----+---+
     |    C      |
     +----------+
```

**Step 3**: Compile the C compiler (Step 2) using the assembly compiler (Step 1). The result is a C compiler written in x86 machine code.

```
+-------------------+       +-------------------+       +-------------------+
|    C     |  x86   |       |    C     |  x86   |       |    C     |  x86   |
+----+-----+----+---+  ==>  +----+-----+----+---+  ==>  +----+-----+----+---+
     |    C      |                |   ASM    |                |   x86    |
     +----+------+                +----------+                +----------+
          |
    compiled by
          |
     +----+-----+----+
     |    C     | x86 |
     +----+-----+----+
          | ASM  |
          +------+
```

### Python Representation of T-Diagrams

```python
from dataclasses import dataclass

@dataclass
class Compiler:
    """Represents a compiler as a T-diagram."""
    source: str      # Source language
    target: str      # Target language
    impl: str        # Implementation language

    def __repr__(self):
        return f"Compiler({self.source} -> {self.target}, written in {self.impl})"

@dataclass
class Program:
    """Represents a program."""
    name: str
    language: str

    def __repr__(self):
        return f"Program({self.name}, in {self.language})"

def compile_with(compiler: Compiler, artifact):
    """
    Simulate compilation:
    - If artifact is a Program in compiler.source, produce Program in compiler.target
    - If artifact is a Compiler in compiler.source, produce Compiler with impl=compiler.target
    """
    if isinstance(artifact, Program):
        if artifact.language != compiler.source:
            raise ValueError(
                f"Cannot compile {artifact.language} program "
                f"with {compiler.source} compiler"
            )
        return Program(artifact.name, compiler.target)

    elif isinstance(artifact, Compiler):
        if artifact.impl != compiler.source:
            raise ValueError(
                f"Cannot compile compiler written in {artifact.impl} "
                f"with {compiler.source} compiler"
            )
        return Compiler(artifact.source, artifact.target, compiler.target)

# Bootstrapping example
print("=== Bootstrapping a C Compiler ===\n")

# Step 1: We have a C compiler written in ASM that targets x86
c_compiler_asm = Compiler("C", "x86", "ASM")
print(f"Step 1 (existing):  {c_compiler_asm}")

# Step 2: We write a C compiler in C that targets x86
c_compiler_in_c = Compiler("C", "x86", "C")
print(f"Step 2 (source):    {c_compiler_in_c}")

# Step 3: Compile the C compiler with the ASM compiler
c_compiler_native = compile_with(c_compiler_asm, c_compiler_in_c)
print(f"Step 3 (compiled):  {c_compiler_native}")

# Step 4: The native compiler can now compile itself
c_compiler_self = compile_with(c_compiler_native, c_compiler_in_c)
print(f"Step 4 (self-comp): {c_compiler_self}")
```

Output:

```
=== Bootstrapping a C Compiler ===

Step 1 (existing):  Compiler(C -> x86, written in ASM)
Step 2 (source):    Compiler(C -> x86, written in C)
Step 3 (compiled):  Compiler(C -> x86, written in x86)
Step 4 (self-comp): Compiler(C -> x86, written in x86)
```

---

## 11. Compiler Construction Tools

Over the decades, many tools have been developed to automate parts of compiler construction:

### Lexer Generators

| Tool | Input | Output | Notes |
|------|-------|--------|-------|
| Lex | Regular expressions | C scanner | Original Unix tool (1975) |
| Flex | Regular expressions | C scanner | Fast Lex (GNU replacement) |
| re2c | Regular expressions | C/C++ scanner | Generates direct code, no tables |
| ANTLR | Grammar | Java/Python/... scanner+parser | Combined lexer+parser |
| PLY | Python regex | Python scanner | Python Lex-Yacc |

### Parser Generators

| Tool | Grammar type | Output | Notes |
|------|-------------|--------|-------|
| Yacc | LALR(1) | C parser | Original Unix tool (1975) |
| Bison | LALR(1)/GLR | C/C++/Java parser | GNU replacement for Yacc |
| ANTLR | LL(*) | Multi-language parser | Popular, user-friendly |
| Lark | Earley/LALR | Python parser | Modern Python tool |
| tree-sitter | LR(1)/GLR | C parser + bindings | Incremental parsing for editors |

### Compiler Frameworks

| Framework | Type | Notes |
|-----------|------|-------|
| LLVM | Compiler infrastructure | Modular, widely used |
| GCC | Compiler collection | Mature, many targets |
| Cranelift | Code generator | Used in Wasmtime, fast compile times |
| QBE | Compiler backend | Lightweight alternative to LLVM |
| MLIR | IR framework | Multi-level IR (part of LLVM project) |

---

## 12. Why Study Compilers?

Even if you never build a production compiler, compiler techniques appear everywhere in software engineering:

1. **Configuration file parsers**: YAML, TOML, JSON, INI parsers use lexer+parser techniques
2. **Query languages**: SQL, GraphQL, Elasticsearch queries are compiled/interpreted
3. **Template engines**: Jinja2, Handlebars, JSX all involve parsing and code generation
4. **Build systems**: Make, CMake, Bazel parse their own DSLs
5. **Regular expressions**: Every regex engine implements finite automata
6. **IDEs**: Syntax highlighting, code completion, and refactoring use compiler front-ends
7. **Static analysis**: Linters and security scanners perform semantic analysis
8. **Code generation**: ORMs, protocol buffers, and API generators produce code from specifications
9. **Optimization**: Understanding compiler optimizations helps you write faster code
10. **Formal verification**: Type systems and program analysis build on compiler techniques

---

## Summary

- A **compiler** translates source code from one language to another while preserving semantics
- Compilation proceeds through well-defined **phases**: lexical analysis, parsing, semantic analysis, IR generation, optimization, and code generation
- The **front-end** (language-dependent) and **back-end** (machine-dependent) are separated by an **intermediate representation**
- **Bootstrapping** allows a compiler to compile itself, but raises trust issues
- **Cross-compilation** generates code for a platform different from the one the compiler runs on
- **T-diagrams** provide a visual notation for reasoning about compilers, programs, and their relationships
- Modern compilers like GCC and LLVM/Clang use dozens of passes over multiple intermediate representations
- Compiler techniques are broadly applicable beyond compiler construction itself

---

## Exercises

### Exercise 1: Phase Identification

For the following program, identify what each compiler phase would do. Write down the output of each phase (tokens, AST structure, type checking results, three-address code, optimized code).

```c
float area;
float radius = 5.0;
area = 3.14159 * radius * radius;
```

### Exercise 2: Compiler vs. Interpreter

Consider the following scenarios. For each, explain whether a compiler, interpreter, or hybrid approach would be most appropriate and why:

1. A scripting language for automating system administration tasks
2. A language for writing high-performance game engines
3. A configuration language for specifying build rules
4. A language running in a web browser (sandboxed)

### Exercise 3: T-Diagrams

Draw T-diagrams for the following scenario:

You have:
- A Python interpreter written in C that runs on x86
- A C compiler written in C that targets x86

Show how you would:
1. Run a Python program on an x86 machine
2. Compile the C compiler using itself
3. Create a cross-compiler that runs on x86 but targets ARM

### Exercise 4: Bootstrapping Sequence

Suppose you are creating a new language called "Nova" and you want to write the Nova compiler in Nova itself. Describe a concrete bootstrapping strategy with at least three steps. What is the minimum you need to implement in an existing language?

### Exercise 5: Front-End / Back-End Separation

A company has compilers for 4 source languages targeting 3 architectures. How many compiler components are needed:
1. Without a shared IR?
2. With a shared IR?

If they add 2 more languages and 2 more architectures, how do the numbers change in each case?

### Exercise 6: Compilation Phases in Practice

Use Python's `dis` module and `ast` module to explore how CPython compiles Python code:

```python
import ast
import dis

source = "x = [i**2 for i in range(10)]"

# 1. Parse to AST
tree = ast.parse(source)
print(ast.dump(tree, indent=2))

# 2. Compile to bytecode
code = compile(source, "<string>", "exec")

# 3. Disassemble
dis.dis(code)
```

Study the output and identify which compiler phases are visible. What optimizations (if any) did CPython perform?

---

[Next: Lexical Analysis](./02_Lexical_Analysis.md) | [Overview](./00_Overview.md)
