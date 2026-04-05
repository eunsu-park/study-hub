# Capstone Compiler Project

**Previous**: [27. Modern Parser Generators](./27_Modern_Parser_Generators.md)

---

This capstone lesson guides you through building a complete compiler for a toy language called **MiniLang**. You will implement every phase: lexical analysis, parsing, type checking, intermediate representation, optimization, and code generation targeting LLVM IR. By the end, you will have a working compiler that translates MiniLang source code into executable native code via LLVM.

**Difficulty**: ⭐⭐⭐⭐⭐

**Prerequisites**: All previous lessons, especially [02. Lexical Analysis](./02_Lexical_Analysis.md), [05. Top-Down Parsing](./05_Top_Down_Parsing.md), [08. Semantic Analysis](./08_Semantic_Analysis.md), [18. SSA Form](./18_SSA_Form.md), [20. LLVM IR Introduction](./20_LLVM_IR_Introduction.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Design a small but complete programming language
2. Build a lexer, parser, type checker, and code generator from scratch
3. Generate LLVM IR and compile it to native executables
4. Apply optimization passes to the generated IR
5. Integrate all compiler phases into a cohesive pipeline

---

## Table of Contents

1. [MiniLang Language Specification](#1-minilang-language-specification)
2. [Project Architecture](#2-project-architecture)
3. [Phase 1: Lexer](#3-phase-1-lexer)
4. [Phase 2: Parser](#4-phase-2-parser)
5. [Phase 3: Type Checker](#5-phase-3-type-checker)
6. [Phase 4: IR Generation](#6-phase-4-ir-generation)
7. [Phase 5: LLVM Code Generation](#7-phase-5-llvm-code-generation)
8. [Phase 6: Optimization](#8-phase-6-optimization)
9. [Putting It All Together](#9-putting-it-all-together)
10. [Extension Ideas](#10-extension-ideas)
11. [Summary](#11-summary)
12. [References](#12-references)

---

## 1. MiniLang Language Specification

### 1.1 Overview

MiniLang is a statically-typed, imperative language with:
- Integer and boolean types
- Variables with type annotations
- Arithmetic and logical expressions
- If/else statements
- While loops
- Functions with parameters and return values
- Print statement for output

### 1.2 Example Program

```
// Compute factorial iteratively
fn factorial(n: int) -> int {
    var result: int = 1;
    var i: int = 1;
    while i <= n {
        result = result * i;
        i = i + 1;
    }
    return result;
}

fn main() -> int {
    var x: int = 10;
    print factorial(x);
    return 0;
}
```

### 1.3 Grammar (EBNF)

```ebnf
program      = { function } ;
function     = "fn" IDENT "(" [ params ] ")" "->" type block ;
params       = param { "," param } ;
param        = IDENT ":" type ;
type         = "int" | "bool" | "void" ;
block        = "{" { statement } "}" ;
statement    = var_decl | assignment | if_stmt | while_stmt
             | return_stmt | print_stmt | expr_stmt ;
var_decl     = "var" IDENT ":" type "=" expression ";" ;
assignment   = IDENT "=" expression ";" ;
if_stmt      = "if" expression block [ "else" block ] ;
while_stmt   = "while" expression block ;
return_stmt  = "return" expression ";" ;
print_stmt   = "print" expression ";" ;
expr_stmt    = expression ";" ;
expression   = logic_or ;
logic_or     = logic_and { "||" logic_and } ;
logic_and    = equality { "&&" equality } ;
equality     = comparison { ( "==" | "!=" ) comparison } ;
comparison   = addition { ( "<" | ">" | "<=" | ">=" ) addition } ;
addition     = multiplication { ( "+" | "-" ) multiplication } ;
multiplication = unary { ( "*" | "/" | "%" ) unary } ;
unary        = [ "-" | "!" ] primary ;
primary      = INTEGER | "true" | "false" | IDENT
             | IDENT "(" [ args ] ")" | "(" expression ")" ;
args         = expression { "," expression } ;
```

### 1.4 Type Rules

- Arithmetic operators (`+`, `-`, `*`, `/`, `%`) require `int` operands and produce `int`
- Comparison operators (`<`, `>`, `<=`, `>=`) require `int` operands and produce `bool`
- Equality operators (`==`, `!=`) require matching types and produce `bool`
- Logical operators (`&&`, `||`) require `bool` operands and produce `bool`
- Unary `-` requires `int`, `!` requires `bool`
- Conditions in `if` and `while` must be `bool`
- Return type must match function declaration

---

## 2. Project Architecture

```
minilang/
├── lexer.py          # Tokenizer
├── parser.py         # Recursive descent parser -> AST
├── ast_nodes.py      # AST node definitions
├── type_checker.py   # Static type checking
├── codegen.py        # LLVM IR generation (via llvmlite)
├── compiler.py       # Main driver
└── tests/
    ├── test_lexer.py
    ├── test_parser.py
    ├── test_type_checker.py
    ├── test_codegen.py
    └── programs/     # Sample MiniLang programs
        ├── factorial.mini
        ├── fibonacci.mini
        └── gcd.mini
```

---

## 3. Phase 1: Lexer

### 3.1 Token Types

```python
from enum import Enum, auto
from dataclasses import dataclass

class TokenType(Enum):
    # Literals
    INTEGER = auto()
    TRUE = auto()
    FALSE = auto()
    IDENTIFIER = auto()

    # Keywords
    FN = auto()
    VAR = auto()
    IF = auto()
    ELSE = auto()
    WHILE = auto()
    RETURN = auto()
    PRINT = auto()
    INT_TYPE = auto()    # "int"
    BOOL_TYPE = auto()   # "bool"
    VOID_TYPE = auto()   # "void"

    # Operators
    PLUS = auto()
    MINUS = auto()
    STAR = auto()
    SLASH = auto()
    PERCENT = auto()
    ASSIGN = auto()       # =
    EQ = auto()           # ==
    NEQ = auto()          # !=
    LT = auto()
    GT = auto()
    LTE = auto()          # <=
    GTE = auto()          # >=
    AND = auto()          # &&
    OR = auto()           # ||
    NOT = auto()          # !

    # Delimiters
    LPAREN = auto()
    RPAREN = auto()
    LBRACE = auto()
    RBRACE = auto()
    COMMA = auto()
    COLON = auto()
    SEMICOLON = auto()
    ARROW = auto()        # ->

    # Special
    EOF = auto()

@dataclass
class Token:
    type: TokenType
    value: str
    line: int
    column: int
```

### 3.2 Lexer Implementation

```python
class Lexer:
    KEYWORDS = {
        'fn': TokenType.FN, 'var': TokenType.VAR,
        'if': TokenType.IF, 'else': TokenType.ELSE,
        'while': TokenType.WHILE, 'return': TokenType.RETURN,
        'print': TokenType.PRINT, 'int': TokenType.INT_TYPE,
        'bool': TokenType.BOOL_TYPE, 'void': TokenType.VOID_TYPE,
        'true': TokenType.TRUE, 'false': TokenType.FALSE,
    }

    def __init__(self, source):
        self.source = source
        self.pos = 0
        self.line = 1
        self.column = 1
        self.tokens = []

    def tokenize(self):
        while self.pos < len(self.source):
            self.skip_whitespace_and_comments()
            if self.pos >= len(self.source):
                break

            ch = self.source[self.pos]

            if ch.isdigit():
                self.read_number()
            elif ch.isalpha() or ch == '_':
                self.read_identifier()
            elif ch == '=' and self.peek(1) == '=':
                self.add_token(TokenType.EQ, '==', advance=2)
            elif ch == '!' and self.peek(1) == '=':
                self.add_token(TokenType.NEQ, '!=', advance=2)
            elif ch == '<' and self.peek(1) == '=':
                self.add_token(TokenType.LTE, '<=', advance=2)
            elif ch == '>' and self.peek(1) == '=':
                self.add_token(TokenType.GTE, '>=', advance=2)
            elif ch == '&' and self.peek(1) == '&':
                self.add_token(TokenType.AND, '&&', advance=2)
            elif ch == '|' and self.peek(1) == '|':
                self.add_token(TokenType.OR, '||', advance=2)
            elif ch == '-' and self.peek(1) == '>':
                self.add_token(TokenType.ARROW, '->', advance=2)
            else:
                single = {
                    '+': TokenType.PLUS, '-': TokenType.MINUS,
                    '*': TokenType.STAR, '/': TokenType.SLASH,
                    '%': TokenType.PERCENT, '=': TokenType.ASSIGN,
                    '<': TokenType.LT, '>': TokenType.GT,
                    '!': TokenType.NOT, '(': TokenType.LPAREN,
                    ')': TokenType.RPAREN, '{': TokenType.LBRACE,
                    '}': TokenType.RBRACE, ',': TokenType.COMMA,
                    ':': TokenType.COLON, ';': TokenType.SEMICOLON,
                }
                if ch in single:
                    self.add_token(single[ch], ch, advance=1)
                else:
                    raise LexError(f"Unexpected character '{ch}' "
                                   f"at line {self.line}, col {self.column}")

        self.tokens.append(Token(TokenType.EOF, '', self.line, self.column))
        return self.tokens

    def read_number(self):
        start = self.pos
        while self.pos < len(self.source) and self.source[self.pos].isdigit():
            self.pos += 1
            self.column += 1
        self.tokens.append(Token(TokenType.INTEGER,
                                  self.source[start:self.pos],
                                  self.line, self.column - (self.pos - start)))

    def read_identifier(self):
        start = self.pos
        while self.pos < len(self.source) and (self.source[self.pos].isalnum()
                                                 or self.source[self.pos] == '_'):
            self.pos += 1
            self.column += 1
        word = self.source[start:self.pos]
        ttype = self.KEYWORDS.get(word, TokenType.IDENTIFIER)
        self.tokens.append(Token(ttype, word, self.line,
                                  self.column - (self.pos - start)))

    def skip_whitespace_and_comments(self):
        while self.pos < len(self.source):
            ch = self.source[self.pos]
            if ch in ' \t\r':
                self.pos += 1
                self.column += 1
            elif ch == '\n':
                self.pos += 1
                self.line += 1
                self.column = 1
            elif ch == '/' and self.peek(1) == '/':
                while self.pos < len(self.source) and self.source[self.pos] != '\n':
                    self.pos += 1
            else:
                break

    def peek(self, offset):
        idx = self.pos + offset
        return self.source[idx] if idx < len(self.source) else '\0'

    def add_token(self, ttype, value, advance):
        self.tokens.append(Token(ttype, value, self.line, self.column))
        self.pos += advance
        self.column += advance
```

---

## 4. Phase 2: Parser

### 4.1 AST Node Definitions

```python
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class Program:
    functions: List['FunctionDef']

@dataclass
class FunctionDef:
    name: str
    params: List['Param']
    return_type: str
    body: 'Block'
    line: int = 0

@dataclass
class Param:
    name: str
    type_name: str

@dataclass
class Block:
    statements: List['Statement']

# Statements
@dataclass
class VarDecl:
    name: str
    type_name: str
    initializer: 'Expr'

@dataclass
class Assignment:
    name: str
    value: 'Expr'

@dataclass
class IfStmt:
    condition: 'Expr'
    then_block: Block
    else_block: Optional[Block] = None

@dataclass
class WhileStmt:
    condition: 'Expr'
    body: Block

@dataclass
class ReturnStmt:
    value: 'Expr'

@dataclass
class PrintStmt:
    value: 'Expr'

# Expressions
@dataclass
class BinaryExpr:
    op: str
    left: 'Expr'
    right: 'Expr'

@dataclass
class UnaryExpr:
    op: str
    operand: 'Expr'

@dataclass
class IntLiteral:
    value: int

@dataclass
class BoolLiteral:
    value: bool

@dataclass
class Identifier:
    name: str

@dataclass
class FunctionCall:
    name: str
    args: List['Expr']
```

### 4.2 Recursive Descent Parser

```python
class Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.pos = 0

    def current(self):
        return self.tokens[self.pos]

    def expect(self, ttype):
        tok = self.current()
        if tok.type != ttype:
            raise ParseError(f"Expected {ttype}, got {tok.type} "
                             f"('{tok.value}') at line {tok.line}")
        self.pos += 1
        return tok

    def match(self, ttype):
        if self.current().type == ttype:
            return self.expect(ttype)
        return None

    def parse_program(self):
        functions = []
        while self.current().type != TokenType.EOF:
            functions.append(self.parse_function())
        return Program(functions)

    def parse_function(self):
        self.expect(TokenType.FN)
        name = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.LPAREN)
        params = self.parse_params()
        self.expect(TokenType.RPAREN)
        self.expect(TokenType.ARROW)
        ret_type = self.parse_type()
        body = self.parse_block()
        return FunctionDef(name, params, ret_type, body)

    def parse_params(self):
        params = []
        if self.current().type != TokenType.RPAREN:
            params.append(self.parse_param())
            while self.match(TokenType.COMMA):
                params.append(self.parse_param())
        return params

    def parse_param(self):
        name = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.COLON)
        type_name = self.parse_type()
        return Param(name, type_name)

    def parse_type(self):
        tok = self.current()
        if tok.type in (TokenType.INT_TYPE, TokenType.BOOL_TYPE, TokenType.VOID_TYPE):
            self.pos += 1
            return tok.value
        raise ParseError(f"Expected type, got {tok.value} at line {tok.line}")

    def parse_block(self):
        self.expect(TokenType.LBRACE)
        stmts = []
        while self.current().type != TokenType.RBRACE:
            stmts.append(self.parse_statement())
        self.expect(TokenType.RBRACE)
        return Block(stmts)

    def parse_statement(self):
        tok = self.current()
        if tok.type == TokenType.VAR:
            return self.parse_var_decl()
        elif tok.type == TokenType.IF:
            return self.parse_if_stmt()
        elif tok.type == TokenType.WHILE:
            return self.parse_while_stmt()
        elif tok.type == TokenType.RETURN:
            return self.parse_return_stmt()
        elif tok.type == TokenType.PRINT:
            return self.parse_print_stmt()
        elif (tok.type == TokenType.IDENTIFIER
              and self.tokens[self.pos + 1].type == TokenType.ASSIGN):
            return self.parse_assignment()
        else:
            expr = self.parse_expression()
            self.expect(TokenType.SEMICOLON)
            return expr

    def parse_expression(self):
        return self.parse_logic_or()

    def parse_logic_or(self):
        left = self.parse_logic_and()
        while self.current().type == TokenType.OR:
            self.pos += 1
            right = self.parse_logic_and()
            left = BinaryExpr('||', left, right)
        return left

    def parse_logic_and(self):
        left = self.parse_equality()
        while self.current().type == TokenType.AND:
            self.pos += 1
            right = self.parse_equality()
            left = BinaryExpr('&&', left, right)
        return left

    def parse_equality(self):
        left = self.parse_comparison()
        while self.current().type in (TokenType.EQ, TokenType.NEQ):
            op = self.current().value
            self.pos += 1
            right = self.parse_comparison()
            left = BinaryExpr(op, left, right)
        return left

    def parse_comparison(self):
        left = self.parse_addition()
        while self.current().type in (TokenType.LT, TokenType.GT,
                                       TokenType.LTE, TokenType.GTE):
            op = self.current().value
            self.pos += 1
            right = self.parse_addition()
            left = BinaryExpr(op, left, right)
        return left

    def parse_addition(self):
        left = self.parse_multiplication()
        while self.current().type in (TokenType.PLUS, TokenType.MINUS):
            op = self.current().value
            self.pos += 1
            right = self.parse_multiplication()
            left = BinaryExpr(op, left, right)
        return left

    def parse_multiplication(self):
        left = self.parse_unary()
        while self.current().type in (TokenType.STAR, TokenType.SLASH,
                                       TokenType.PERCENT):
            op = self.current().value
            self.pos += 1
            right = self.parse_unary()
            left = BinaryExpr(op, left, right)
        return left

    def parse_unary(self):
        if self.current().type in (TokenType.MINUS, TokenType.NOT):
            op = self.current().value
            self.pos += 1
            operand = self.parse_unary()
            return UnaryExpr(op, operand)
        return self.parse_primary()

    def parse_primary(self):
        tok = self.current()
        if tok.type == TokenType.INTEGER:
            self.pos += 1
            return IntLiteral(int(tok.value))
        elif tok.type == TokenType.TRUE:
            self.pos += 1
            return BoolLiteral(True)
        elif tok.type == TokenType.FALSE:
            self.pos += 1
            return BoolLiteral(False)
        elif tok.type == TokenType.IDENTIFIER:
            self.pos += 1
            if self.current().type == TokenType.LPAREN:
                # Function call
                self.pos += 1
                args = []
                if self.current().type != TokenType.RPAREN:
                    args.append(self.parse_expression())
                    while self.match(TokenType.COMMA):
                        args.append(self.parse_expression())
                self.expect(TokenType.RPAREN)
                return FunctionCall(tok.value, args)
            return Identifier(tok.value)
        elif tok.type == TokenType.LPAREN:
            self.pos += 1
            expr = self.parse_expression()
            self.expect(TokenType.RPAREN)
            return expr
        raise ParseError(f"Unexpected token '{tok.value}' at line {tok.line}")

    # ... var_decl, assignment, if, while, return, print parsers
    def parse_var_decl(self):
        self.expect(TokenType.VAR)
        name = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.COLON)
        type_name = self.parse_type()
        self.expect(TokenType.ASSIGN)
        init = self.parse_expression()
        self.expect(TokenType.SEMICOLON)
        return VarDecl(name, type_name, init)

    def parse_assignment(self):
        name = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.ASSIGN)
        value = self.parse_expression()
        self.expect(TokenType.SEMICOLON)
        return Assignment(name, value)

    def parse_if_stmt(self):
        self.expect(TokenType.IF)
        cond = self.parse_expression()
        then_block = self.parse_block()
        else_block = None
        if self.match(TokenType.ELSE):
            else_block = self.parse_block()
        return IfStmt(cond, then_block, else_block)

    def parse_while_stmt(self):
        self.expect(TokenType.WHILE)
        cond = self.parse_expression()
        body = self.parse_block()
        return WhileStmt(cond, body)

    def parse_return_stmt(self):
        self.expect(TokenType.RETURN)
        value = self.parse_expression()
        self.expect(TokenType.SEMICOLON)
        return ReturnStmt(value)

    def parse_print_stmt(self):
        self.expect(TokenType.PRINT)
        value = self.parse_expression()
        self.expect(TokenType.SEMICOLON)
        return PrintStmt(value)
```

---

## 5. Phase 3: Type Checker

```python
class TypeChecker:
    def __init__(self):
        self.functions = {}     # name -> (param_types, return_type)
        self.variables = {}     # name -> type (current scope)
        self.scope_stack = []
        self.current_return_type = None
        self.errors = []

    def check_program(self, program):
        # First pass: register all function signatures
        for func in program.functions:
            param_types = [p.type_name for p in func.params]
            self.functions[func.name] = (param_types, func.return_type)

        # Second pass: type check each function body
        for func in program.functions:
            self.check_function(func)

        if self.errors:
            raise TypeErrors(self.errors)

    def check_function(self, func):
        self.push_scope()
        self.current_return_type = func.return_type
        for param in func.params:
            self.define(param.name, param.type_name)
        self.check_block(func.body)
        self.pop_scope()

    def check_block(self, block):
        self.push_scope()
        for stmt in block.statements:
            self.check_statement(stmt)
        self.pop_scope()

    def check_statement(self, stmt):
        if isinstance(stmt, VarDecl):
            init_type = self.check_expr(stmt.initializer)
            if init_type != stmt.type_name:
                self.error(f"Cannot assign {init_type} to {stmt.type_name}")
            self.define(stmt.name, stmt.type_name)

        elif isinstance(stmt, Assignment):
            var_type = self.lookup(stmt.name)
            val_type = self.check_expr(stmt.value)
            if var_type != val_type:
                self.error(f"Cannot assign {val_type} to {var_type}")

        elif isinstance(stmt, IfStmt):
            cond_type = self.check_expr(stmt.condition)
            if cond_type != 'bool':
                self.error(f"If condition must be bool, got {cond_type}")
            self.check_block(stmt.then_block)
            if stmt.else_block:
                self.check_block(stmt.else_block)

        elif isinstance(stmt, WhileStmt):
            cond_type = self.check_expr(stmt.condition)
            if cond_type != 'bool':
                self.error(f"While condition must be bool, got {cond_type}")
            self.check_block(stmt.body)

        elif isinstance(stmt, ReturnStmt):
            ret_type = self.check_expr(stmt.value)
            if ret_type != self.current_return_type:
                self.error(f"Return type {ret_type} doesn't match "
                          f"declared {self.current_return_type}")

        elif isinstance(stmt, PrintStmt):
            self.check_expr(stmt.value)

    def check_expr(self, expr):
        if isinstance(expr, IntLiteral):
            return 'int'
        elif isinstance(expr, BoolLiteral):
            return 'bool'
        elif isinstance(expr, Identifier):
            return self.lookup(expr.name)
        elif isinstance(expr, BinaryExpr):
            return self.check_binary(expr)
        elif isinstance(expr, UnaryExpr):
            return self.check_unary(expr)
        elif isinstance(expr, FunctionCall):
            return self.check_call(expr)

    def check_binary(self, expr):
        left = self.check_expr(expr.left)
        right = self.check_expr(expr.right)

        if expr.op in ('+', '-', '*', '/', '%'):
            if left != 'int' or right != 'int':
                self.error(f"Operator {expr.op} requires int operands")
            return 'int'
        elif expr.op in ('<', '>', '<=', '>='):
            if left != 'int' or right != 'int':
                self.error(f"Operator {expr.op} requires int operands")
            return 'bool'
        elif expr.op in ('==', '!='):
            if left != right:
                self.error(f"Cannot compare {left} with {right}")
            return 'bool'
        elif expr.op in ('&&', '||'):
            if left != 'bool' or right != 'bool':
                self.error(f"Operator {expr.op} requires bool operands")
            return 'bool'

    def check_unary(self, expr):
        operand_type = self.check_expr(expr.operand)
        if expr.op == '-' and operand_type != 'int':
            self.error(f"Unary - requires int, got {operand_type}")
            return 'int'
        if expr.op == '!' and operand_type != 'bool':
            self.error(f"Unary ! requires bool, got {operand_type}")
            return 'bool'
        return operand_type

    def check_call(self, expr):
        if expr.name not in self.functions:
            self.error(f"Undefined function: {expr.name}")
            return 'int'
        param_types, ret_type = self.functions[expr.name]
        if len(expr.args) != len(param_types):
            self.error(f"Expected {len(param_types)} args, got {len(expr.args)}")
        for arg, expected in zip(expr.args, param_types):
            actual = self.check_expr(arg)
            if actual != expected:
                self.error(f"Arg type mismatch: expected {expected}, got {actual}")
        return ret_type

    def push_scope(self):
        self.scope_stack.append(dict(self.variables))

    def pop_scope(self):
        self.variables = self.scope_stack.pop()

    def define(self, name, type_name):
        self.variables[name] = type_name

    def lookup(self, name):
        if name in self.variables:
            return self.variables[name]
        self.error(f"Undefined variable: {name}")
        return 'int'

    def error(self, msg):
        self.errors.append(msg)
```

---

## 6. Phase 4: IR Generation

### 6.1 LLVM IR with llvmlite

```python
from llvmlite import ir

class CodeGenerator:
    def __init__(self):
        self.module = ir.Module(name="minilang")
        self.builder = None
        self.variables = {}     # name -> alloca instruction
        self.functions = {}     # name -> ir.Function
        self.printf_func = None

    def generate(self, program):
        self.declare_printf()
        # First pass: declare all functions
        for func in program.functions:
            self.declare_function(func)
        # Second pass: generate bodies
        for func in program.functions:
            self.generate_function(func)
        return self.module

    def declare_printf(self):
        printf_type = ir.FunctionType(ir.IntType(32), [ir.IntType(8).as_pointer()],
                                       var_arg=True)
        self.printf_func = ir.Function(self.module, printf_type, name="printf")

    def type_to_llvm(self, type_name):
        if type_name == 'int':
            return ir.IntType(32)
        elif type_name == 'bool':
            return ir.IntType(1)
        elif type_name == 'void':
            return ir.VoidType()

    def declare_function(self, func):
        param_types = [self.type_to_llvm(p.type_name) for p in func.params]
        ret_type = self.type_to_llvm(func.return_type)
        func_type = ir.FunctionType(ret_type, param_types)
        llvm_func = ir.Function(self.module, func_type, name=func.name)
        self.functions[func.name] = llvm_func

    def generate_function(self, func):
        llvm_func = self.functions[func.name]
        block = llvm_func.append_basic_block(name="entry")
        self.builder = ir.IRBuilder(block)
        self.variables = {}

        # Allocate space for parameters
        for param, llvm_arg in zip(func.params, llvm_func.args):
            llvm_arg.name = param.name
            alloca = self.builder.alloca(self.type_to_llvm(param.type_name),
                                          name=param.name)
            self.builder.store(llvm_arg, alloca)
            self.variables[param.name] = alloca

        # Generate body
        self.generate_block(func.body)

        # Ensure function has a terminator
        if not self.builder.block.is_terminated:
            if func.return_type == 'void':
                self.builder.ret_void()
            else:
                self.builder.ret(ir.Constant(self.type_to_llvm(func.return_type), 0))

    def generate_block(self, block):
        for stmt in block.statements:
            self.generate_statement(stmt)
            if self.builder.block.is_terminated:
                break

    def generate_statement(self, stmt):
        if isinstance(stmt, VarDecl):
            alloca = self.builder.alloca(self.type_to_llvm(stmt.type_name),
                                          name=stmt.name)
            val = self.generate_expr(stmt.initializer)
            self.builder.store(val, alloca)
            self.variables[stmt.name] = alloca

        elif isinstance(stmt, Assignment):
            val = self.generate_expr(stmt.value)
            self.builder.store(val, self.variables[stmt.name])

        elif isinstance(stmt, ReturnStmt):
            val = self.generate_expr(stmt.value)
            self.builder.ret(val)

        elif isinstance(stmt, PrintStmt):
            val = self.generate_expr(stmt.value)
            fmt = self.get_format_string()
            self.builder.call(self.printf_func, [fmt, val])

        elif isinstance(stmt, IfStmt):
            self.generate_if(stmt)

        elif isinstance(stmt, WhileStmt):
            self.generate_while(stmt)

    def generate_if(self, stmt):
        cond = self.generate_expr(stmt.condition)
        func = self.builder.function

        then_bb = func.append_basic_block("if.then")
        else_bb = func.append_basic_block("if.else") if stmt.else_block else None
        merge_bb = func.append_basic_block("if.merge")

        if else_bb:
            self.builder.cbranch(cond, then_bb, else_bb)
        else:
            self.builder.cbranch(cond, then_bb, merge_bb)

        # Then block
        self.builder.position_at_start(then_bb)
        self.generate_block(stmt.then_block)
        if not self.builder.block.is_terminated:
            self.builder.branch(merge_bb)

        # Else block
        if else_bb:
            self.builder.position_at_start(else_bb)
            self.generate_block(stmt.else_block)
            if not self.builder.block.is_terminated:
                self.builder.branch(merge_bb)

        self.builder.position_at_start(merge_bb)

    def generate_while(self, stmt):
        func = self.builder.function
        cond_bb = func.append_basic_block("while.cond")
        body_bb = func.append_basic_block("while.body")
        exit_bb = func.append_basic_block("while.exit")

        self.builder.branch(cond_bb)
        self.builder.position_at_start(cond_bb)
        cond = self.generate_expr(stmt.condition)
        self.builder.cbranch(cond, body_bb, exit_bb)

        self.builder.position_at_start(body_bb)
        self.generate_block(stmt.body)
        if not self.builder.block.is_terminated:
            self.builder.branch(cond_bb)

        self.builder.position_at_start(exit_bb)

    def generate_expr(self, expr):
        if isinstance(expr, IntLiteral):
            return ir.Constant(ir.IntType(32), expr.value)
        elif isinstance(expr, BoolLiteral):
            return ir.Constant(ir.IntType(1), int(expr.value))
        elif isinstance(expr, Identifier):
            return self.builder.load(self.variables[expr.name], name=expr.name)
        elif isinstance(expr, BinaryExpr):
            return self.generate_binary(expr)
        elif isinstance(expr, UnaryExpr):
            return self.generate_unary(expr)
        elif isinstance(expr, FunctionCall):
            return self.generate_call(expr)

    def generate_binary(self, expr):
        left = self.generate_expr(expr.left)
        right = self.generate_expr(expr.right)

        ops = {
            '+': self.builder.add, '-': self.builder.sub,
            '*': self.builder.mul, '/': self.builder.sdiv,
            '%': self.builder.srem,
        }
        if expr.op in ops:
            return ops[expr.op](left, right)

        cmp_ops = {
            '<': '<', '>': '>', '<=': '<=', '>=': '>=',
            '==': '==', '!=': '!=',
        }
        if expr.op in cmp_ops:
            return self.builder.icmp_signed(cmp_ops[expr.op], left, right)

        if expr.op == '&&':
            return self.builder.and_(left, right)
        if expr.op == '||':
            return self.builder.or_(left, right)

    def generate_unary(self, expr):
        operand = self.generate_expr(expr.operand)
        if expr.op == '-':
            return self.builder.neg(operand)
        if expr.op == '!':
            return self.builder.not_(operand)

    def generate_call(self, expr):
        func = self.functions[expr.name]
        args = [self.generate_expr(a) for a in expr.args]
        return self.builder.call(func, args)

    def get_format_string(self):
        fmt = "%d\n\0"
        fmt_const = ir.Constant(ir.ArrayType(ir.IntType(8), len(fmt)),
                                 bytearray(fmt.encode("utf8")))
        global_fmt = ir.GlobalVariable(self.module, fmt_const.type, name=".fmt")
        global_fmt.linkage = "internal"
        global_fmt.global_constant = True
        global_fmt.initializer = fmt_const
        return self.builder.bitcast(global_fmt, ir.IntType(8).as_pointer())
```

---

## 7. Phase 5: LLVM Code Generation

```python
from llvmlite import binding

def compile_to_native(module, output_path):
    """Compile LLVM IR module to native executable."""
    binding.initialize()
    binding.initialize_native_target()
    binding.initialize_native_asmprinter()

    # Verify the module
    llvm_ir = str(module)
    mod = binding.parse_assembly(llvm_ir)
    mod.verify()

    # Create target machine
    target = binding.Target.from_default_triple()
    target_machine = target.create_target_machine(opt=2)

    # Emit object code
    obj_path = output_path + ".o"
    with open(obj_path, "wb") as f:
        f.write(target_machine.emit_object(mod))

    # Link with system linker
    import subprocess
    subprocess.run(["cc", obj_path, "-o", output_path], check=True)
    print(f"Compiled to {output_path}")
```

---

## 8. Phase 6: Optimization

```python
def optimize_module(module_ir):
    """Apply LLVM optimization passes."""
    mod = binding.parse_assembly(module_ir)
    mod.verify()

    # Create pass manager with O2 optimizations
    pm_builder = binding.create_pass_manager_builder()
    pm_builder.opt_level = 2

    pm = binding.create_module_pass_manager()
    pm_builder.populate(pm)

    # Run passes
    pm.run(mod)

    return mod
```

---

## 9. Putting It All Together

```python
def compile_file(source_path, output_path):
    """Compile a MiniLang source file to a native executable."""
    # Read source
    with open(source_path) as f:
        source = f.read()

    # Phase 1: Lex
    tokens = Lexer(source).tokenize()
    print(f"Lexer: {len(tokens)} tokens")

    # Phase 2: Parse
    ast = Parser(tokens).parse_program()
    print(f"Parser: {len(ast.functions)} functions")

    # Phase 3: Type Check
    TypeChecker().check_program(ast)
    print("Type checker: OK")

    # Phase 4: Generate LLVM IR
    codegen = CodeGenerator()
    module = codegen.generate(ast)
    print(f"IR generation: {len(str(module))} bytes")

    # Phase 5: Optimize
    optimized = optimize_module(str(module))
    print("Optimization: OK")

    # Phase 6: Compile to native
    compile_to_native_from_mod(optimized, output_path)
    print(f"Output: {output_path}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python compiler.py <input.mini> <output>")
        sys.exit(1)
    compile_file(sys.argv[1], sys.argv[2])
```

---

## 10. Extension Ideas

Once your basic compiler works, consider these extensions:

1. **String type**: Add string literals and concatenation
2. **Arrays**: Add array types with indexing
3. **For loops**: Add `for i in range(n)` style loops
4. **Structs**: Add user-defined struct types
5. **Closures**: Add anonymous functions and closures (see Lesson 24)
6. **Type inference**: Infer types from initializers (see Lesson 23)
7. **Error messages**: Add source location to all error messages
8. **Debug info**: Generate DWARF debug info (see Lesson 26)
9. **Standard library**: Add built-in functions (abs, min, max)
10. **Interpreter mode**: Add an interpreter backend alongside LLVM

---

## 11. Summary

- Building a compiler end-to-end requires integrating many subsystems
- The **lexer** converts source text to tokens
- The **parser** builds an AST from the token stream
- The **type checker** validates types and catches errors before code generation
- The **code generator** translates the AST to LLVM IR
- **Optimization passes** improve the generated code
- **LLVM** handles the heavy lifting of machine code generation
- A complete compiler for MiniLang is approximately 800-1000 lines of Python

---

## 12. References

1. Nystrom, R. (2021). *Crafting Interpreters*. https://craftinginterpreters.com/
2. LLVM Tutorial: Implementing a Language with LLVM: https://llvm.org/docs/tutorial/
3. llvmlite documentation: https://llvmlite.readthedocs.io/
4. Cooper, K., Torczon, L. (2011). *Engineering a Compiler*, 2nd ed. Morgan Kaufmann.
5. Appel, A. W. (2004). *Modern Compiler Implementation in ML*. Cambridge University Press.

---

**Previous**: [27. Modern Parser Generators](./27_Modern_Parser_Generators.md)
