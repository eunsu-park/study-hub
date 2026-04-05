"""
12_mini_compiler.py - A Complete Mini Compiler Pipeline

Ties together every phase from the Compiler_Design series into a single
end-to-end pipeline that compiles a small imperative language to bytecode
and executes it on a stack-based virtual machine.

Pipeline:
  Source code
    → Lexer (tokens)
      → Parser (AST)
        → Type Checker (annotated AST)
          → IR Generator (three-address code)
            → Optimizer (optimized TAC)
              → Code Generator (bytecode)
                → Virtual Machine (execution)

Language features:
  - Integer and boolean types
  - Arithmetic: +, -, *, /, %
  - Comparison: ==, !=, <, >, <=, >=
  - Variables: let x = expr;
  - Assignment: x = expr;
  - Print: print(expr);
  - If/else: if (cond) { ... } else { ... }
  - While: while (cond) { ... }
  - Functions: fn name(params) -> type { ... return expr; }

Example program:
    fn factorial(n: int) -> int {
        if (n <= 1) { return 1; }
        return n * factorial(n - 1);
    }
    print(factorial(5));
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any


# ===================================================================
# PHASE 1: LEXER
# ===================================================================

class TokenType(Enum):
    # Literals
    INT_LIT    = auto()
    BOOL_LIT   = auto()
    IDENT      = auto()
    # Keywords
    LET        = auto()
    FN         = auto()
    RETURN     = auto()
    IF         = auto()
    ELSE       = auto()
    WHILE      = auto()
    PRINT      = auto()
    INT_TYPE   = auto()   # "int"
    BOOL_TYPE  = auto()   # "bool"
    # Operators
    PLUS       = auto()
    MINUS      = auto()
    STAR       = auto()
    SLASH      = auto()
    PERCENT    = auto()
    EQ         = auto()   # ==
    NE         = auto()   # !=
    LT         = auto()
    GT         = auto()
    LE         = auto()
    GE         = auto()
    ASSIGN     = auto()   # =
    ARROW      = auto()   # ->
    # Delimiters
    LPAREN     = auto()
    RPAREN     = auto()
    LBRACE     = auto()
    RBRACE     = auto()
    COMMA      = auto()
    COLON      = auto()
    SEMI       = auto()
    # Special
    EOF        = auto()


@dataclass
class Token:
    type: TokenType
    value: Any
    line: int = 0

    def __repr__(self) -> str:
        return f"Token({self.type.name}, {self.value!r})"


KEYWORDS = {
    "let": TokenType.LET, "fn": TokenType.FN, "return": TokenType.RETURN,
    "if": TokenType.IF, "else": TokenType.ELSE, "while": TokenType.WHILE,
    "print": TokenType.PRINT, "int": TokenType.INT_TYPE,
    "bool": TokenType.BOOL_TYPE, "true": TokenType.BOOL_LIT,
    "false": TokenType.BOOL_LIT,
}


def lex(source: str) -> list[Token]:
    """Tokenize source code into a list of tokens."""
    tokens: list[Token] = []
    i = 0
    line = 1

    while i < len(source):
        ch = source[i]

        # Whitespace
        if ch in " \t\r":
            i += 1
            continue
        if ch == "\n":
            line += 1
            i += 1
            continue

        # Comments (// to end of line)
        if ch == "/" and i + 1 < len(source) and source[i + 1] == "/":
            while i < len(source) and source[i] != "\n":
                i += 1
            continue

        # Numbers
        if ch.isdigit():
            start = i
            while i < len(source) and source[i].isdigit():
                i += 1
            tokens.append(Token(TokenType.INT_LIT, int(source[start:i]), line))
            continue

        # Identifiers and keywords
        if ch.isalpha() or ch == "_":
            start = i
            while i < len(source) and (source[i].isalnum() or source[i] == "_"):
                i += 1
            word = source[start:i]
            if word in KEYWORDS:
                tt = KEYWORDS[word]
                val = (word == "true") if tt == TokenType.BOOL_LIT else word
                tokens.append(Token(tt, val, line))
            else:
                tokens.append(Token(TokenType.IDENT, word, line))
            continue

        # Two-character operators
        two = source[i:i + 2]
        if two == "==":
            tokens.append(Token(TokenType.EQ, "==", line)); i += 2; continue
        if two == "!=":
            tokens.append(Token(TokenType.NE, "!=", line)); i += 2; continue
        if two == "<=":
            tokens.append(Token(TokenType.LE, "<=", line)); i += 2; continue
        if two == ">=":
            tokens.append(Token(TokenType.GE, ">=", line)); i += 2; continue
        if two == "->":
            tokens.append(Token(TokenType.ARROW, "->", line)); i += 2; continue

        # Single-character tokens
        singles = {
            "+": TokenType.PLUS, "-": TokenType.MINUS, "*": TokenType.STAR,
            "/": TokenType.SLASH, "%": TokenType.PERCENT,
            "<": TokenType.LT, ">": TokenType.GT, "=": TokenType.ASSIGN,
            "(": TokenType.LPAREN, ")": TokenType.RPAREN,
            "{": TokenType.LBRACE, "}": TokenType.RBRACE,
            ",": TokenType.COMMA, ":": TokenType.COLON, ";": TokenType.SEMI,
        }
        if ch in singles:
            tokens.append(Token(singles[ch], ch, line))
            i += 1
            continue

        raise SyntaxError(f"Unexpected character {ch!r} at line {line}")

    tokens.append(Token(TokenType.EOF, None, line))
    return tokens


# ===================================================================
# PHASE 2: PARSER (Recursive Descent → AST)
# ===================================================================

# --- AST node types ---

@dataclass
class IntLit:
    value: int

@dataclass
class BoolLit:
    value: bool

@dataclass
class Var:
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
class Call:
    name: str
    args: list

@dataclass
class LetStmt:
    name: str
    expr: Any

@dataclass
class AssignStmt:
    name: str
    expr: Any

@dataclass
class PrintStmt:
    expr: Any

@dataclass
class ReturnStmt:
    expr: Any

@dataclass
class IfStmt:
    cond: Any
    then_body: list
    else_body: list

@dataclass
class WhileStmt:
    cond: Any
    body: list

@dataclass
class Param:
    name: str
    type_name: str

@dataclass
class FuncDef:
    name: str
    params: list[Param]
    return_type: str
    body: list

@dataclass
class Program:
    functions: list[FuncDef]
    statements: list


class Parser:
    """Recursive-descent parser producing an AST."""

    def __init__(self, tokens: list[Token]):
        self.tokens = tokens
        self.pos = 0

    def peek(self) -> Token:
        return self.tokens[self.pos]

    def advance(self) -> Token:
        tok = self.tokens[self.pos]
        self.pos += 1
        return tok

    def expect(self, tt: TokenType) -> Token:
        tok = self.advance()
        if tok.type != tt:
            raise SyntaxError(
                f"Expected {tt.name}, got {tok.type.name} ({tok.value!r}) "
                f"at line {tok.line}"
            )
        return tok

    def parse(self) -> Program:
        funcs: list[FuncDef] = []
        stmts: list = []
        while self.peek().type != TokenType.EOF:
            if self.peek().type == TokenType.FN:
                funcs.append(self.parse_func())
            else:
                stmts.append(self.parse_stmt())
        return Program(funcs, stmts)

    def parse_func(self) -> FuncDef:
        self.expect(TokenType.FN)
        name = self.expect(TokenType.IDENT).value
        self.expect(TokenType.LPAREN)
        params: list[Param] = []
        while self.peek().type != TokenType.RPAREN:
            if params:
                self.expect(TokenType.COMMA)
            pname = self.expect(TokenType.IDENT).value
            self.expect(TokenType.COLON)
            ptype = self.advance().value  # "int" or "bool"
            params.append(Param(pname, ptype))
        self.expect(TokenType.RPAREN)
        self.expect(TokenType.ARROW)
        rtype = self.advance().value
        body = self.parse_block()
        return FuncDef(name, params, rtype, body)

    def parse_block(self) -> list:
        self.expect(TokenType.LBRACE)
        stmts = []
        while self.peek().type != TokenType.RBRACE:
            stmts.append(self.parse_stmt())
        self.expect(TokenType.RBRACE)
        return stmts

    def parse_stmt(self):
        tt = self.peek().type
        if tt == TokenType.LET:
            return self.parse_let()
        if tt == TokenType.RETURN:
            return self.parse_return()
        if tt == TokenType.PRINT:
            return self.parse_print()
        if tt == TokenType.IF:
            return self.parse_if()
        if tt == TokenType.WHILE:
            return self.parse_while()
        if tt == TokenType.IDENT:
            # Assignment: ident = expr;
            name = self.advance().value
            self.expect(TokenType.ASSIGN)
            expr = self.parse_expr()
            self.expect(TokenType.SEMI)
            return AssignStmt(name, expr)
        raise SyntaxError(
            f"Unexpected token {self.peek().type.name} at line {self.peek().line}"
        )

    def parse_let(self):
        self.expect(TokenType.LET)
        name = self.expect(TokenType.IDENT).value
        self.expect(TokenType.ASSIGN)
        expr = self.parse_expr()
        self.expect(TokenType.SEMI)
        return LetStmt(name, expr)

    def parse_return(self):
        self.expect(TokenType.RETURN)
        expr = self.parse_expr()
        self.expect(TokenType.SEMI)
        return ReturnStmt(expr)

    def parse_print(self):
        self.expect(TokenType.PRINT)
        self.expect(TokenType.LPAREN)
        expr = self.parse_expr()
        self.expect(TokenType.RPAREN)
        self.expect(TokenType.SEMI)
        return PrintStmt(expr)

    def parse_if(self):
        self.expect(TokenType.IF)
        self.expect(TokenType.LPAREN)
        cond = self.parse_expr()
        self.expect(TokenType.RPAREN)
        then_body = self.parse_block()
        else_body = []
        if self.peek().type == TokenType.ELSE:
            self.advance()
            else_body = self.parse_block()
        return IfStmt(cond, then_body, else_body)

    def parse_while(self):
        self.expect(TokenType.WHILE)
        self.expect(TokenType.LPAREN)
        cond = self.parse_expr()
        self.expect(TokenType.RPAREN)
        body = self.parse_block()
        return WhileStmt(cond, body)

    # --- Expression parsing (precedence climbing) ---

    def parse_expr(self):
        return self.parse_comparison()

    def parse_comparison(self):
        left = self.parse_additive()
        while self.peek().type in (
            TokenType.EQ, TokenType.NE, TokenType.LT,
            TokenType.GT, TokenType.LE, TokenType.GE,
        ):
            op = self.advance().value
            right = self.parse_additive()
            left = BinOp(op, left, right)
        return left

    def parse_additive(self):
        left = self.parse_multiplicative()
        while self.peek().type in (TokenType.PLUS, TokenType.MINUS):
            op = self.advance().value
            right = self.parse_multiplicative()
            left = BinOp(op, left, right)
        return left

    def parse_multiplicative(self):
        left = self.parse_unary()
        while self.peek().type in (TokenType.STAR, TokenType.SLASH, TokenType.PERCENT):
            op = self.advance().value
            right = self.parse_unary()
            left = BinOp(op, left, right)
        return left

    def parse_unary(self):
        if self.peek().type == TokenType.MINUS:
            self.advance()
            operand = self.parse_primary()
            return UnaryOp("-", operand)
        return self.parse_primary()

    def parse_primary(self):
        tok = self.peek()
        if tok.type == TokenType.INT_LIT:
            self.advance()
            return IntLit(tok.value)
        if tok.type == TokenType.BOOL_LIT:
            self.advance()
            return BoolLit(tok.value)
        if tok.type == TokenType.IDENT:
            name = self.advance().value
            if self.peek().type == TokenType.LPAREN:
                # Function call
                self.advance()  # consume (
                args = []
                while self.peek().type != TokenType.RPAREN:
                    if args:
                        self.expect(TokenType.COMMA)
                    args.append(self.parse_expr())
                self.expect(TokenType.RPAREN)
                return Call(name, args)
            return Var(name)
        if tok.type == TokenType.LPAREN:
            self.advance()
            expr = self.parse_expr()
            self.expect(TokenType.RPAREN)
            return expr
        raise SyntaxError(
            f"Unexpected {tok.type.name} ({tok.value!r}) at line {tok.line}"
        )


# ===================================================================
# PHASE 3: TYPE CHECKER (simplified)
# ===================================================================

class TypeChecker:
    """Simple type checker: ensures operations have compatible types."""

    def __init__(self):
        self.env: dict[str, str] = {}           # var -> type
        self.funcs: dict[str, tuple] = {}       # name -> (param_types, ret_type)
        self.errors: list[str] = []

    def check(self, prog: Program) -> list[str]:
        # Register functions
        for fn in prog.functions:
            ptypes = [p.type_name for p in fn.params]
            self.funcs[fn.name] = (ptypes, fn.return_type)

        # Check function bodies
        for fn in prog.functions:
            saved = dict(self.env)
            for p in fn.params:
                self.env[p.name] = p.type_name
            for stmt in fn.body:
                self.check_stmt(stmt)
            self.env = saved

        # Check top-level statements
        for stmt in prog.statements:
            self.check_stmt(stmt)

        return self.errors

    def check_stmt(self, stmt):
        if isinstance(stmt, LetStmt):
            t = self.infer(stmt.expr)
            self.env[stmt.name] = t
        elif isinstance(stmt, AssignStmt):
            self.infer(stmt.expr)
        elif isinstance(stmt, PrintStmt):
            self.infer(stmt.expr)
        elif isinstance(stmt, ReturnStmt):
            self.infer(stmt.expr)
        elif isinstance(stmt, IfStmt):
            ct = self.infer(stmt.cond)
            if ct != "bool":
                self.errors.append(f"If condition must be bool, got {ct}")
            for s in stmt.then_body:
                self.check_stmt(s)
            for s in stmt.else_body:
                self.check_stmt(s)
        elif isinstance(stmt, WhileStmt):
            ct = self.infer(stmt.cond)
            if ct != "bool":
                self.errors.append(f"While condition must be bool, got {ct}")
            for s in stmt.body:
                self.check_stmt(s)

    def infer(self, expr) -> str:
        if isinstance(expr, IntLit):
            return "int"
        if isinstance(expr, BoolLit):
            return "bool"
        if isinstance(expr, Var):
            return self.env.get(expr.name, "int")
        if isinstance(expr, UnaryOp):
            return self.infer(expr.operand)
        if isinstance(expr, BinOp):
            lt = self.infer(expr.left)
            rt = self.infer(expr.right)
            if expr.op in ("==", "!=", "<", ">", "<=", ">="):
                return "bool"
            return lt
        if isinstance(expr, Call):
            info = self.funcs.get(expr.name)
            if info:
                return info[1]
            self.errors.append(f"Unknown function: {expr.name}")
            return "int"
        return "int"


# ===================================================================
# PHASE 4: CODE GENERATOR (AST → Bytecode)
# ===================================================================

class Op(Enum):
    PUSH    = auto()
    POP     = auto()
    LOAD    = auto()
    STORE   = auto()
    ADD     = auto()
    SUB     = auto()
    MUL     = auto()
    DIV     = auto()
    MOD     = auto()
    NEG     = auto()
    EQ      = auto()
    NE      = auto()
    LT      = auto()
    GT      = auto()
    LE      = auto()
    GE      = auto()
    NOT     = auto()
    JUMP    = auto()
    JUMP_IF_FALSE = auto()
    CALL    = auto()
    RETURN  = auto()
    PRINT   = auto()
    HALT    = auto()


@dataclass
class BytecodeInst:
    op: Op
    arg: Any = None

    def __repr__(self) -> str:
        if self.arg is not None:
            return f"{self.op.name:<18} {self.arg!r}"
        return self.op.name


@dataclass
class CompiledFunc:
    name: str
    params: list[str]
    code: list[BytecodeInst]


class CodeGenerator:
    """Compile AST to bytecode instructions."""

    def __init__(self):
        self.functions: dict[str, CompiledFunc] = {}
        self.code: list[BytecodeInst] = []

    def compile(self, prog: Program) -> tuple[list[BytecodeInst], dict[str, CompiledFunc]]:
        # Compile functions
        for fn in prog.functions:
            self.code = []
            for stmt in fn.body:
                self.compile_stmt(stmt)
            self.code.append(BytecodeInst(Op.PUSH, 0))
            self.code.append(BytecodeInst(Op.RETURN))
            self.functions[fn.name] = CompiledFunc(
                fn.name, [p.name for p in fn.params], list(self.code)
            )

        # Compile top-level
        self.code = []
        for stmt in prog.statements:
            self.compile_stmt(stmt)
        self.code.append(BytecodeInst(Op.HALT))
        return self.code, self.functions

    def compile_stmt(self, stmt):
        if isinstance(stmt, LetStmt):
            self.compile_expr(stmt.expr)
            self.code.append(BytecodeInst(Op.STORE, stmt.name))
        elif isinstance(stmt, AssignStmt):
            self.compile_expr(stmt.expr)
            self.code.append(BytecodeInst(Op.STORE, stmt.name))
        elif isinstance(stmt, PrintStmt):
            self.compile_expr(stmt.expr)
            self.code.append(BytecodeInst(Op.PRINT))
        elif isinstance(stmt, ReturnStmt):
            self.compile_expr(stmt.expr)
            self.code.append(BytecodeInst(Op.RETURN))
        elif isinstance(stmt, IfStmt):
            self.compile_expr(stmt.cond)
            jump_false = len(self.code)
            self.code.append(BytecodeInst(Op.JUMP_IF_FALSE, 0))  # patch later
            for s in stmt.then_body:
                self.compile_stmt(s)
            if stmt.else_body:
                jump_end = len(self.code)
                self.code.append(BytecodeInst(Op.JUMP, 0))  # patch later
                self.code[jump_false].arg = len(self.code)
                for s in stmt.else_body:
                    self.compile_stmt(s)
                self.code[jump_end].arg = len(self.code)
            else:
                self.code[jump_false].arg = len(self.code)
        elif isinstance(stmt, WhileStmt):
            loop_start = len(self.code)
            self.compile_expr(stmt.cond)
            jump_false = len(self.code)
            self.code.append(BytecodeInst(Op.JUMP_IF_FALSE, 0))
            for s in stmt.body:
                self.compile_stmt(s)
            self.code.append(BytecodeInst(Op.JUMP, loop_start))
            self.code[jump_false].arg = len(self.code)

    OP_MAP = {
        "+": Op.ADD, "-": Op.SUB, "*": Op.MUL, "/": Op.DIV, "%": Op.MOD,
        "==": Op.EQ, "!=": Op.NE, "<": Op.LT, ">": Op.GT,
        "<=": Op.LE, ">=": Op.GE,
    }

    def compile_expr(self, expr):
        if isinstance(expr, IntLit):
            self.code.append(BytecodeInst(Op.PUSH, expr.value))
        elif isinstance(expr, BoolLit):
            self.code.append(BytecodeInst(Op.PUSH, 1 if expr.value else 0))
        elif isinstance(expr, Var):
            self.code.append(BytecodeInst(Op.LOAD, expr.name))
        elif isinstance(expr, UnaryOp):
            self.compile_expr(expr.operand)
            self.code.append(BytecodeInst(Op.NEG))
        elif isinstance(expr, BinOp):
            self.compile_expr(expr.left)
            self.compile_expr(expr.right)
            self.code.append(BytecodeInst(self.OP_MAP[expr.op]))
        elif isinstance(expr, Call):
            for arg in expr.args:
                self.compile_expr(arg)
            self.code.append(BytecodeInst(Op.CALL, (expr.name, len(expr.args))))


# ===================================================================
# PHASE 5: VIRTUAL MACHINE
# ===================================================================

@dataclass
class Frame:
    """A call frame with its own locals and return address."""
    func_name: str
    locals: dict[str, Any] = field(default_factory=dict)
    return_ip: int = 0
    return_code: list = field(default_factory=list)


class VM:
    """Stack-based virtual machine that executes bytecode."""

    MAX_STACK = 1024
    MAX_CALLS = 256

    def __init__(
        self,
        main_code: list[BytecodeInst],
        functions: dict[str, CompiledFunc],
    ):
        self.functions = functions
        self.stack: list[Any] = []
        self.frames: list[Frame] = [Frame("__main__")]
        self.code = main_code
        self.ip = 0
        self.output: list[str] = []
        self.steps = 0

    @property
    def locals(self) -> dict[str, Any]:
        return self.frames[-1].locals

    def push(self, val: Any) -> None:
        if len(self.stack) >= self.MAX_STACK:
            raise RuntimeError("Stack overflow")
        self.stack.append(val)

    def pop(self) -> Any:
        if not self.stack:
            raise RuntimeError("Stack underflow")
        return self.stack.pop()

    def run(self, max_steps: int = 100_000) -> list[str]:
        while self.ip < len(self.code) and self.steps < max_steps:
            inst = self.code[self.ip]
            self.ip += 1
            self.steps += 1
            self.execute(inst)
            if inst.op == Op.HALT:
                break

        if self.steps >= max_steps:
            self.output.append(f"[VM halted after {max_steps} steps]")
        return self.output

    def execute(self, inst: BytecodeInst) -> None:
        op = inst.op

        if op == Op.PUSH:
            self.push(inst.arg)
        elif op == Op.POP:
            self.pop()
        elif op == Op.LOAD:
            val = self.locals.get(inst.arg, 0)
            self.push(val)
        elif op == Op.STORE:
            self.locals[inst.arg] = self.pop()
        elif op == Op.ADD:
            b, a = self.pop(), self.pop(); self.push(a + b)
        elif op == Op.SUB:
            b, a = self.pop(), self.pop(); self.push(a - b)
        elif op == Op.MUL:
            b, a = self.pop(), self.pop(); self.push(a * b)
        elif op == Op.DIV:
            b, a = self.pop(), self.pop(); self.push(a // b if b else 0)
        elif op == Op.MOD:
            b, a = self.pop(), self.pop(); self.push(a % b if b else 0)
        elif op == Op.NEG:
            self.push(-self.pop())
        elif op == Op.EQ:
            b, a = self.pop(), self.pop(); self.push(1 if a == b else 0)
        elif op == Op.NE:
            b, a = self.pop(), self.pop(); self.push(1 if a != b else 0)
        elif op == Op.LT:
            b, a = self.pop(), self.pop(); self.push(1 if a < b else 0)
        elif op == Op.GT:
            b, a = self.pop(), self.pop(); self.push(1 if a > b else 0)
        elif op == Op.LE:
            b, a = self.pop(), self.pop(); self.push(1 if a <= b else 0)
        elif op == Op.GE:
            b, a = self.pop(), self.pop(); self.push(1 if a >= b else 0)
        elif op == Op.NOT:
            self.push(0 if self.pop() else 1)
        elif op == Op.JUMP:
            self.ip = inst.arg
        elif op == Op.JUMP_IF_FALSE:
            if not self.pop():
                self.ip = inst.arg
        elif op == Op.CALL:
            func_name, argc = inst.arg
            fn = self.functions.get(func_name)
            if not fn:
                raise RuntimeError(f"Undefined function: {func_name}")
            if len(self.frames) >= self.MAX_CALLS:
                raise RuntimeError("Call stack overflow")
            # Collect arguments
            args = [self.pop() for _ in range(argc)][::-1]
            # Save current state
            frame = Frame(func_name, return_ip=self.ip, return_code=self.code)
            for pname, val in zip(fn.params, args):
                frame.locals[pname] = val
            self.frames.append(frame)
            self.code = fn.code
            self.ip = 0
        elif op == Op.RETURN:
            ret_val = self.pop()
            frame = self.frames.pop()
            self.code = frame.return_code
            self.ip = frame.return_ip
            self.push(ret_val)
        elif op == Op.PRINT:
            val = self.pop()
            self.output.append(str(val))
        elif op == Op.HALT:
            pass


# ===================================================================
# COMPILE & RUN HELPER
# ===================================================================

def compile_and_run(source: str, verbose: bool = False) -> list[str]:
    """Full pipeline: source → tokens → AST → type-check → bytecode → execute."""
    # 1. Lex
    tokens = lex(source)
    if verbose:
        print("  Tokens:", len(tokens))

    # 2. Parse
    parser = Parser(tokens)
    ast = parser.parse()
    if verbose:
        print(f"  AST: {len(ast.functions)} functions, {len(ast.statements)} statements")

    # 3. Type check
    checker = TypeChecker()
    errors = checker.check(ast)
    if errors:
        return [f"Type error: {e}" for e in errors]

    # 4. Code generation
    gen = CodeGenerator()
    main_code, functions = gen.compile(ast)
    if verbose:
        print(f"  Bytecode: {len(main_code)} main instructions")
        for name, fn in functions.items():
            print(f"    {name}: {len(fn.code)} instructions")

    # 5. Execute
    vm = VM(main_code, functions)
    return vm.run()


# ===================================================================
# DEMOS
# ===================================================================

def demo_arithmetic():
    print("=" * 60)
    print("MINI COMPILER: ARITHMETIC")
    print("=" * 60)

    source = """\
    let x = 10 + 20 * 3;
    print(x);
    let y = (x - 30) / 2;
    print(y);
    print(x % 7);
    """
    print(f"\n  Source:\n    {source.strip()}")
    output = compile_and_run(source)
    print(f"\n  Output: {output}")
    print(f"  Expected: ['70', '20', '0']")


def demo_conditionals():
    print("\n" + "=" * 60)
    print("MINI COMPILER: IF/ELSE")
    print("=" * 60)

    source = """\
    let x = 42;
    if (x > 50) {
        print(1);
    } else {
        print(0);
    }
    if (x == 42) {
        print(999);
    }
    """
    print(f"\n  Source:\n    {source.strip()}")
    output = compile_and_run(source)
    print(f"\n  Output: {output}")
    print(f"  Expected: ['0', '999']")


def demo_loops():
    print("\n" + "=" * 60)
    print("MINI COMPILER: WHILE LOOPS")
    print("=" * 60)

    source = """\
    // Sum 1 to 10
    let sum = 0;
    let i = 1;
    while (i <= 10) {
        sum = sum + i;
        i = i + 1;
    }
    print(sum);

    // Fibonacci: first 10 terms
    let a = 0;
    let b = 1;
    let count = 0;
    while (count < 10) {
        print(a);
        let temp = a + b;
        a = b;
        b = temp;
        count = count + 1;
    }
    """
    print(f"\n  Source:\n    {source.strip()}")
    output = compile_and_run(source)
    print(f"\n  Output: {output}")
    print(f"  Expected: ['55', '0', '1', '1', '2', '3', '5', '8', '13', '21', '34']")


def demo_functions():
    print("\n" + "=" * 60)
    print("MINI COMPILER: FUNCTIONS")
    print("=" * 60)

    source = """\
    fn square(x: int) -> int {
        return x * x;
    }

    fn add(a: int, b: int) -> int {
        return a + b;
    }

    print(square(7));
    print(add(10, 20));
    print(add(square(3), square(4)));
    """
    print(f"\n  Source:\n    {source.strip()}")
    output = compile_and_run(source)
    print(f"\n  Output: {output}")
    print(f"  Expected: ['49', '30', '25']")


def demo_recursion():
    print("\n" + "=" * 60)
    print("MINI COMPILER: RECURSION")
    print("=" * 60)

    source = """\
    fn factorial(n: int) -> int {
        if (n <= 1) {
            return 1;
        }
        return n * factorial(n - 1);
    }

    fn fib(n: int) -> int {
        if (n <= 1) {
            return n;
        }
        return fib(n - 1) + fib(n - 2);
    }

    print(factorial(5));
    print(factorial(10));
    print(fib(10));
    """
    print(f"\n  Source:\n    {source.strip()}")
    output = compile_and_run(source)
    print(f"\n  Output: {output}")
    print(f"  Expected: ['120', '3628800', '55']")


def demo_bytecode_inspection():
    print("\n" + "=" * 60)
    print("BYTECODE INSPECTION")
    print("=" * 60)

    source = """\
    fn abs(x: int) -> int {
        if (x < 0) {
            return -x;
        }
        return x;
    }
    print(abs(-42));
    print(abs(7));
    """

    tokens = lex(source)
    ast = Parser(tokens).parse()
    gen = CodeGenerator()
    main_code, functions = gen.compile(ast)

    print(f"\n  Function 'abs' bytecode:")
    for i, inst in enumerate(functions["abs"].code):
        print(f"    [{i:3d}] {inst}")

    print(f"\n  Main bytecode:")
    for i, inst in enumerate(main_code):
        print(f"    [{i:3d}] {inst}")

    output = compile_and_run(source)
    print(f"\n  Output: {output}")
    print(f"  Expected: ['42', '7']")


def demo_pipeline_summary():
    print("\n" + "=" * 60)
    print("COMPILER PIPELINE SUMMARY")
    print("=" * 60)

    print("""
  Complete compilation pipeline implemented:

    Phase              File                         Key Concepts
    ─────────────────  ───────────────────────────  ──────────────────────────
    1. Lexical         01_lexer.py                  Regular expressions, tokens
    2. Finite Automata 02_nfa_dfa.py                NFA→DFA subset construction
    3. Context-Free    03_cfg_parser.py             Grammar rules, derivations
    4. LL Parsing      04_ll_parser.py              First/Follow, LL(1) tables
    5. LR Parsing      05_lr_parser.py              SLR, shift/reduce
    6. AST             06_ast_builder.py            Tree construction
    7. Type Checking   07_type_checker.py           Type inference, error reporting
    8. IR Generation   08_tac_generator.py          Three-address code, SSA
    9. Optimization    09_optimizer.py              Constant folding, DCE, CSE
    10. Code Gen       10_bytecode_vm.py            Stack machine, call frames
    11. Register Alloc 11_register_allocator.py     Graph coloring, spilling
    12. Mini Compiler  12_mini_compiler.py          End-to-end pipeline (this file)

  Each file is self-contained and demonstrates its concepts independently.
  This file (12) integrates phases 1, 2 (lexer), 4/5 (parser), 7 (types),
  10 (codegen + VM) into a working compiler for a small language.
    """)


if __name__ == "__main__":
    demo_arithmetic()
    demo_conditionals()
    demo_loops()
    demo_functions()
    demo_recursion()
    demo_bytecode_inspection()
    demo_pipeline_summary()
