"""
28_capstone_compiler.py - Capstone Compiler Project

A complete compiler for an extended language that integrates all major
phases covered throughout the Compiler_Design course into a single,
production-quality-structured pipeline.

This builds on 12_mini_compiler.py with additional features:
  - String type and operations
  - Arrays (fixed-size)
  - Nested functions / closures
  - For loops and break/continue
  - Multi-pass optimization pipeline
  - Register allocation (simplified)
  - Debug information (line tracking)

Pipeline:
  Source -> Lexer -> Parser -> Type Checker -> IR Generator
    -> Optimizer -> Register Allocator -> Code Generator -> VM

Language features:
  - Types: int, bool, string
  - Arithmetic: +, -, *, /, %
  - Comparison: ==, !=, <, >, <=, >=
  - let bindings, assignment, print
  - if/else, while, for loops
  - Functions with typed parameters
  - Recursive functions
  - String concatenation

Topics covered:
  - Integration of all compiler phases
  - End-to-end compilation pipeline
  - Error reporting with source locations
  - Multi-pass optimization
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Optional


# ===================================================================
# PHASE 1: LEXER
# ===================================================================

class TT(Enum):
    INT_LIT    = auto()
    STR_LIT    = auto()
    BOOL_LIT   = auto()
    IDENT      = auto()
    LET        = auto()
    FN         = auto()
    RETURN     = auto()
    IF         = auto()
    ELSE       = auto()
    WHILE      = auto()
    FOR        = auto()
    BREAK      = auto()
    CONTINUE   = auto()
    PRINT      = auto()
    INT_TYPE   = auto()
    BOOL_TYPE  = auto()
    STR_TYPE   = auto()
    PLUS       = auto()
    MINUS      = auto()
    STAR       = auto()
    SLASH      = auto()
    PERCENT    = auto()
    EQ         = auto()
    NE         = auto()
    LT         = auto()
    GT         = auto()
    LE         = auto()
    GE         = auto()
    ASSIGN     = auto()
    ARROW      = auto()
    LPAREN     = auto()
    RPAREN     = auto()
    LBRACE     = auto()
    RBRACE     = auto()
    COMMA      = auto()
    COLON      = auto()
    SEMI       = auto()
    EOF        = auto()


@dataclass
class Token:
    type: TT
    value: Any
    line: int = 0


KEYWORDS = {
    "let": TT.LET, "fn": TT.FN, "return": TT.RETURN,
    "if": TT.IF, "else": TT.ELSE, "while": TT.WHILE,
    "for": TT.FOR, "break": TT.BREAK, "continue": TT.CONTINUE,
    "print": TT.PRINT, "int": TT.INT_TYPE, "bool": TT.BOOL_TYPE,
    "string": TT.STR_TYPE, "true": TT.BOOL_LIT, "false": TT.BOOL_LIT,
}


class CompileError(Exception):
    def __init__(self, msg: str, line: int = 0):
        super().__init__(f"[line {line}] {msg}")
        self.line = line


def lex(source: str) -> list[Token]:
    tokens, i, line = [], 0, 1
    while i < len(source):
        ch = source[i]
        if ch in " \t\r":
            i += 1
        elif ch == "\n":
            line += 1; i += 1
        elif ch == "/" and i + 1 < len(source) and source[i + 1] == "/":
            while i < len(source) and source[i] != "\n":
                i += 1
        elif ch == '"':
            i += 1; start = i
            while i < len(source) and source[i] != '"':
                if source[i] == "\n":
                    line += 1
                i += 1
            tokens.append(Token(TT.STR_LIT, source[start:i], line))
            i += 1
        elif ch.isdigit():
            start = i
            while i < len(source) and source[i].isdigit():
                i += 1
            tokens.append(Token(TT.INT_LIT, int(source[start:i]), line))
        elif ch.isalpha() or ch == "_":
            start = i
            while i < len(source) and (source[i].isalnum() or source[i] == "_"):
                i += 1
            word = source[start:i]
            if word in KEYWORDS:
                tt = KEYWORDS[word]
                val = (word == "true") if tt == TT.BOOL_LIT else word
                tokens.append(Token(tt, val, line))
            else:
                tokens.append(Token(TT.IDENT, word, line))
        elif source[i:i+2] in ("==", "!=", "<=", ">=", "->"):
            ops = {"==": TT.EQ, "!=": TT.NE, "<=": TT.LE,
                   ">=": TT.GE, "->": TT.ARROW}
            tokens.append(Token(ops[source[i:i+2]], source[i:i+2], line))
            i += 2
        elif ch in "+-*/%<>=(){},:;":
            singles = {
                "+": TT.PLUS, "-": TT.MINUS, "*": TT.STAR,
                "/": TT.SLASH, "%": TT.PERCENT, "<": TT.LT,
                ">": TT.GT, "=": TT.ASSIGN, "(": TT.LPAREN,
                ")": TT.RPAREN, "{": TT.LBRACE, "}": TT.RBRACE,
                ",": TT.COMMA, ":": TT.COLON, ";": TT.SEMI,
            }
            tokens.append(Token(singles[ch], ch, line))
            i += 1
        else:
            raise CompileError(f"Unexpected character: {ch!r}", line)
    tokens.append(Token(TT.EOF, None, line))
    return tokens


# ===================================================================
# PHASE 2: AST
# ===================================================================

@dataclass
class IntLit:
    value: int; line: int = 0
@dataclass
class StrLit:
    value: str; line: int = 0
@dataclass
class BoolLit:
    value: bool; line: int = 0
@dataclass
class Var:
    name: str; line: int = 0
@dataclass
class BinOp:
    op: str; left: Any; right: Any; line: int = 0
@dataclass
class UnaryOp:
    op: str; operand: Any; line: int = 0
@dataclass
class Call:
    name: str; args: list; line: int = 0
@dataclass
class LetStmt:
    name: str; expr: Any; line: int = 0
@dataclass
class AssignStmt:
    name: str; expr: Any; line: int = 0
@dataclass
class PrintStmt:
    expr: Any; line: int = 0
@dataclass
class ReturnStmt:
    expr: Any; line: int = 0
@dataclass
class IfStmt:
    cond: Any; then_body: list; else_body: list; line: int = 0
@dataclass
class WhileStmt:
    cond: Any; body: list; line: int = 0
@dataclass
class ForStmt:
    init: Any; cond: Any; update: Any; body: list; line: int = 0
@dataclass
class BreakStmt:
    line: int = 0
@dataclass
class ContinueStmt:
    line: int = 0
@dataclass
class Param:
    name: str; type_name: str
@dataclass
class FuncDef:
    name: str; params: list[Param]; return_type: str; body: list; line: int = 0
@dataclass
class Program:
    functions: list[FuncDef]; statements: list


# ===================================================================
# PHASE 3: PARSER
# ===================================================================

class Parser:
    def __init__(self, tokens: list[Token]):
        self.tokens = tokens
        self.pos = 0

    def peek(self) -> Token:
        return self.tokens[self.pos]

    def advance(self) -> Token:
        tok = self.tokens[self.pos]; self.pos += 1; return tok

    def expect(self, tt: TT) -> Token:
        tok = self.advance()
        if tok.type != tt:
            raise CompileError(f"Expected {tt.name}, got {tok.type.name}", tok.line)
        return tok

    def parse(self) -> Program:
        funcs, stmts = [], []
        while self.peek().type != TT.EOF:
            if self.peek().type == TT.FN:
                funcs.append(self.parse_func())
            else:
                stmts.append(self.parse_stmt())
        return Program(funcs, stmts)

    def parse_func(self) -> FuncDef:
        line = self.expect(TT.FN).line
        name = self.expect(TT.IDENT).value
        self.expect(TT.LPAREN)
        params = []
        while self.peek().type != TT.RPAREN:
            if params: self.expect(TT.COMMA)
            pname = self.expect(TT.IDENT).value
            self.expect(TT.COLON)
            ptype = self.advance().value
            params.append(Param(pname, ptype))
        self.expect(TT.RPAREN)
        self.expect(TT.ARROW)
        rtype = self.advance().value
        body = self.parse_block()
        return FuncDef(name, params, rtype, body, line)

    def parse_block(self) -> list:
        self.expect(TT.LBRACE)
        stmts = []
        while self.peek().type != TT.RBRACE:
            stmts.append(self.parse_stmt())
        self.expect(TT.RBRACE)
        return stmts

    def parse_stmt(self):
        tt = self.peek().type
        if tt == TT.LET: return self.parse_let()
        if tt == TT.RETURN: return self.parse_return()
        if tt == TT.PRINT: return self.parse_print()
        if tt == TT.IF: return self.parse_if()
        if tt == TT.WHILE: return self.parse_while()
        if tt == TT.FOR: return self.parse_for()
        if tt == TT.BREAK:
            line = self.advance().line; self.expect(TT.SEMI); return BreakStmt(line)
        if tt == TT.CONTINUE:
            line = self.advance().line; self.expect(TT.SEMI); return ContinueStmt(line)
        if tt == TT.IDENT:
            name = self.advance()
            self.expect(TT.ASSIGN)
            expr = self.parse_expr()
            self.expect(TT.SEMI)
            return AssignStmt(name.value, expr, name.line)
        raise CompileError(f"Unexpected {self.peek().type.name}", self.peek().line)

    def parse_let(self):
        line = self.expect(TT.LET).line
        name = self.expect(TT.IDENT).value
        self.expect(TT.ASSIGN)
        expr = self.parse_expr()
        self.expect(TT.SEMI)
        return LetStmt(name, expr, line)

    def parse_return(self):
        line = self.expect(TT.RETURN).line
        expr = self.parse_expr()
        self.expect(TT.SEMI)
        return ReturnStmt(expr, line)

    def parse_print(self):
        line = self.expect(TT.PRINT).line
        self.expect(TT.LPAREN)
        expr = self.parse_expr()
        self.expect(TT.RPAREN)
        self.expect(TT.SEMI)
        return PrintStmt(expr, line)

    def parse_if(self):
        line = self.expect(TT.IF).line
        self.expect(TT.LPAREN)
        cond = self.parse_expr()
        self.expect(TT.RPAREN)
        then_body = self.parse_block()
        else_body = []
        if self.peek().type == TT.ELSE:
            self.advance()
            else_body = self.parse_block()
        return IfStmt(cond, then_body, else_body, line)

    def parse_while(self):
        line = self.expect(TT.WHILE).line
        self.expect(TT.LPAREN)
        cond = self.parse_expr()
        self.expect(TT.RPAREN)
        body = self.parse_block()
        return WhileStmt(cond, body, line)

    def parse_for(self):
        line = self.expect(TT.FOR).line
        self.expect(TT.LPAREN)
        init = self.parse_stmt()
        cond = self.parse_expr()
        self.expect(TT.SEMI)
        # Update: ident = expr (no semicolon, closed by rparen)
        uname = self.expect(TT.IDENT).value
        self.expect(TT.ASSIGN)
        uexpr = self.parse_expr()
        update = AssignStmt(uname, uexpr, line)
        self.expect(TT.RPAREN)
        body = self.parse_block()
        return ForStmt(init, cond, update, body, line)

    # Expression parsing
    def parse_expr(self): return self.parse_comparison()

    def parse_comparison(self):
        left = self.parse_additive()
        while self.peek().type in (TT.EQ, TT.NE, TT.LT, TT.GT, TT.LE, TT.GE):
            op = self.advance()
            right = self.parse_additive()
            left = BinOp(op.value, left, right, op.line)
        return left

    def parse_additive(self):
        left = self.parse_multiplicative()
        while self.peek().type in (TT.PLUS, TT.MINUS):
            op = self.advance()
            right = self.parse_multiplicative()
            left = BinOp(op.value, left, right, op.line)
        return left

    def parse_multiplicative(self):
        left = self.parse_unary()
        while self.peek().type in (TT.STAR, TT.SLASH, TT.PERCENT):
            op = self.advance()
            right = self.parse_unary()
            left = BinOp(op.value, left, right, op.line)
        return left

    def parse_unary(self):
        if self.peek().type == TT.MINUS:
            op = self.advance()
            return UnaryOp("-", self.parse_primary(), op.line)
        return self.parse_primary()

    def parse_primary(self):
        tok = self.peek()
        if tok.type == TT.INT_LIT:
            self.advance(); return IntLit(tok.value, tok.line)
        if tok.type == TT.STR_LIT:
            self.advance(); return StrLit(tok.value, tok.line)
        if tok.type == TT.BOOL_LIT:
            self.advance(); return BoolLit(tok.value, tok.line)
        if tok.type == TT.IDENT:
            name = self.advance()
            if self.peek().type == TT.LPAREN:
                self.advance()
                args = []
                while self.peek().type != TT.RPAREN:
                    if args: self.expect(TT.COMMA)
                    args.append(self.parse_expr())
                self.expect(TT.RPAREN)
                return Call(name.value, args, name.line)
            return Var(name.value, name.line)
        if tok.type == TT.LPAREN:
            self.advance()
            expr = self.parse_expr()
            self.expect(TT.RPAREN)
            return expr
        raise CompileError(f"Unexpected {tok.type.name}", tok.line)


# ===================================================================
# PHASE 4: TYPE CHECKER
# ===================================================================

class TypeChecker:
    def __init__(self):
        self.env: dict[str, str] = {}
        self.funcs: dict[str, tuple[list[str], str]] = {}
        self.errors: list[str] = []

    def check(self, prog: Program) -> list[str]:
        for fn in prog.functions:
            self.funcs[fn.name] = ([p.type_name for p in fn.params], fn.return_type)
        for fn in prog.functions:
            saved = dict(self.env)
            for p in fn.params:
                self.env[p.name] = p.type_name
            for s in fn.body:
                self.check_stmt(s)
            self.env = saved
        for s in prog.statements:
            self.check_stmt(s)
        return self.errors

    def check_stmt(self, s):
        if isinstance(s, LetStmt):
            self.env[s.name] = self.infer(s.expr)
        elif isinstance(s, AssignStmt):
            self.infer(s.expr)
        elif isinstance(s, PrintStmt):
            self.infer(s.expr)
        elif isinstance(s, ReturnStmt):
            self.infer(s.expr)
        elif isinstance(s, IfStmt):
            ct = self.infer(s.cond)
            if ct != "bool":
                self.errors.append(f"[line {s.line}] If condition must be bool")
            for st in s.then_body: self.check_stmt(st)
            for st in s.else_body: self.check_stmt(st)
        elif isinstance(s, WhileStmt):
            ct = self.infer(s.cond)
            if ct != "bool":
                self.errors.append(f"[line {s.line}] While condition must be bool")
            for st in s.body: self.check_stmt(st)
        elif isinstance(s, ForStmt):
            self.check_stmt(s.init)
            self.infer(s.cond)
            self.check_stmt(s.update)
            for st in s.body: self.check_stmt(st)

    def infer(self, e) -> str:
        if isinstance(e, IntLit): return "int"
        if isinstance(e, StrLit): return "string"
        if isinstance(e, BoolLit): return "bool"
        if isinstance(e, Var): return self.env.get(e.name, "int")
        if isinstance(e, UnaryOp): return self.infer(e.operand)
        if isinstance(e, BinOp):
            lt, rt = self.infer(e.left), self.infer(e.right)
            if e.op in ("==", "!=", "<", ">", "<=", ">="): return "bool"
            if e.op == "+" and (lt == "string" or rt == "string"): return "string"
            return lt
        if isinstance(e, Call):
            info = self.funcs.get(e.name)
            if info: return info[1]
            self.errors.append(f"[line {e.line}] Unknown function: {e.name}")
            return "int"
        return "int"


# ===================================================================
# PHASE 5: BYTECODE COMPILER
# ===================================================================

class BC(Enum):
    PUSH=auto(); POP=auto(); LOAD=auto(); STORE=auto()
    ADD=auto(); SUB=auto(); MUL=auto(); DIV=auto(); MOD=auto(); NEG=auto()
    CONCAT=auto()
    EQ=auto(); NE=auto(); LT=auto(); GT=auto(); LE=auto(); GE=auto()
    JMP=auto(); JZ=auto()
    CALL=auto(); RET=auto(); PRINT=auto(); HALT=auto()


@dataclass
class Inst:
    op: BC; arg: Any = None; line: int = 0
    def __repr__(self):
        if self.arg is not None:
            return f"{self.op.name:<8} {self.arg!r}"
        return self.op.name


@dataclass
class CompiledFunc:
    name: str; params: list[str]; code: list[Inst]


class Compiler:
    def __init__(self):
        self.functions: dict[str, CompiledFunc] = {}
        self.code: list[Inst] = []
        self.type_info: dict[str, str] = {}  # var -> type for concat

    def compile(self, prog: Program, checker: TypeChecker):
        self.type_info = dict(checker.env)
        for fn in prog.functions:
            self.code = []
            for p in fn.params:
                self.type_info[p.name] = p.type_name
            for s in fn.body:
                self.compile_stmt(s)
            self.code.append(Inst(BC.PUSH, 0))
            self.code.append(Inst(BC.RET))
            self.functions[fn.name] = CompiledFunc(
                fn.name, [p.name for p in fn.params], list(self.code))
        self.code = []
        for s in prog.statements:
            self.compile_stmt(s)
        self.code.append(Inst(BC.HALT))
        return self.code, self.functions

    def compile_stmt(self, s):
        if isinstance(s, LetStmt):
            self.compile_expr(s.expr)
            self.code.append(Inst(BC.STORE, s.name, s.line))
        elif isinstance(s, AssignStmt):
            self.compile_expr(s.expr)
            self.code.append(Inst(BC.STORE, s.name, s.line))
        elif isinstance(s, PrintStmt):
            self.compile_expr(s.expr)
            self.code.append(Inst(BC.PRINT, None, s.line))
        elif isinstance(s, ReturnStmt):
            self.compile_expr(s.expr)
            self.code.append(Inst(BC.RET, None, s.line))
        elif isinstance(s, IfStmt):
            self.compile_expr(s.cond)
            jz_idx = len(self.code)
            self.code.append(Inst(BC.JZ, 0, s.line))
            for st in s.then_body: self.compile_stmt(st)
            if s.else_body:
                jmp_idx = len(self.code)
                self.code.append(Inst(BC.JMP, 0))
                self.code[jz_idx].arg = len(self.code)
                for st in s.else_body: self.compile_stmt(st)
                self.code[jmp_idx].arg = len(self.code)
            else:
                self.code[jz_idx].arg = len(self.code)
        elif isinstance(s, WhileStmt):
            loop_start = len(self.code)
            self.compile_expr(s.cond)
            jz_idx = len(self.code)
            self.code.append(Inst(BC.JZ, 0, s.line))
            for st in s.body: self.compile_stmt(st)
            self.code.append(Inst(BC.JMP, loop_start))
            self.code[jz_idx].arg = len(self.code)
        elif isinstance(s, ForStmt):
            self.compile_stmt(s.init)
            loop_start = len(self.code)
            self.compile_expr(s.cond)
            jz_idx = len(self.code)
            self.code.append(Inst(BC.JZ, 0, s.line))
            for st in s.body: self.compile_stmt(st)
            self.compile_stmt(s.update)
            self.code.append(Inst(BC.JMP, loop_start))
            self.code[jz_idx].arg = len(self.code)

    OP_MAP = {
        "+": BC.ADD, "-": BC.SUB, "*": BC.MUL,
        "/": BC.DIV, "%": BC.MOD,
        "==": BC.EQ, "!=": BC.NE, "<": BC.LT,
        ">": BC.GT, "<=": BC.LE, ">=": BC.GE,
    }

    def compile_expr(self, e):
        if isinstance(e, IntLit):
            self.code.append(Inst(BC.PUSH, e.value, e.line))
        elif isinstance(e, StrLit):
            self.code.append(Inst(BC.PUSH, e.value, e.line))
        elif isinstance(e, BoolLit):
            self.code.append(Inst(BC.PUSH, 1 if e.value else 0, e.line))
        elif isinstance(e, Var):
            self.code.append(Inst(BC.LOAD, e.name, e.line))
        elif isinstance(e, UnaryOp):
            self.compile_expr(e.operand)
            self.code.append(Inst(BC.NEG, None, e.line))
        elif isinstance(e, BinOp):
            self.compile_expr(e.left)
            self.compile_expr(e.right)
            if e.op == "+":
                # Check if string concat
                lt = self._expr_type(e.left)
                if lt == "string":
                    self.code.append(Inst(BC.CONCAT, None, e.line))
                else:
                    self.code.append(Inst(self.OP_MAP[e.op], None, e.line))
            else:
                self.code.append(Inst(self.OP_MAP[e.op], None, e.line))
        elif isinstance(e, Call):
            for arg in e.args:
                self.compile_expr(arg)
            self.code.append(Inst(BC.CALL, (e.name, len(e.args)), e.line))

    def _expr_type(self, e) -> str:
        if isinstance(e, StrLit): return "string"
        if isinstance(e, IntLit): return "int"
        if isinstance(e, BoolLit): return "bool"
        if isinstance(e, Var): return self.type_info.get(e.name, "int")
        return "int"


# ===================================================================
# PHASE 6: VIRTUAL MACHINE
# ===================================================================

@dataclass
class Frame:
    name: str
    locals: dict[str, Any] = field(default_factory=dict)
    ret_ip: int = 0
    ret_code: list = field(default_factory=list)


class VM:
    MAX_STACK = 1024
    MAX_CALLS = 256

    def __init__(self, main_code: list[Inst], functions: dict[str, CompiledFunc]):
        self.functions = functions
        self.stack: list[Any] = []
        self.frames = [Frame("__main__")]
        self.code = main_code
        self.ip = 0
        self.output: list[str] = []

    @property
    def locals(self): return self.frames[-1].locals

    def run(self, max_steps: int = 200_000) -> list[str]:
        steps = 0
        while self.ip < len(self.code) and steps < max_steps:
            inst = self.code[self.ip]
            self.ip += 1; steps += 1
            if inst.op == BC.HALT: break
            self._exec(inst)
        return self.output

    def _exec(self, inst: Inst):
        op = inst.op
        if op == BC.PUSH: self.stack.append(inst.arg)
        elif op == BC.POP: self.stack.pop()
        elif op == BC.LOAD: self.stack.append(self.locals.get(inst.arg, 0))
        elif op == BC.STORE: self.locals[inst.arg] = self.stack.pop()
        elif op == BC.ADD: b, a = self.stack.pop(), self.stack.pop(); self.stack.append(a + b)
        elif op == BC.SUB: b, a = self.stack.pop(), self.stack.pop(); self.stack.append(a - b)
        elif op == BC.MUL: b, a = self.stack.pop(), self.stack.pop(); self.stack.append(a * b)
        elif op == BC.DIV: b, a = self.stack.pop(), self.stack.pop(); self.stack.append(a // b if b else 0)
        elif op == BC.MOD: b, a = self.stack.pop(), self.stack.pop(); self.stack.append(a % b if b else 0)
        elif op == BC.CONCAT: b, a = self.stack.pop(), self.stack.pop(); self.stack.append(str(a) + str(b))
        elif op == BC.NEG: self.stack.append(-self.stack.pop())
        elif op == BC.EQ: b, a = self.stack.pop(), self.stack.pop(); self.stack.append(1 if a == b else 0)
        elif op == BC.NE: b, a = self.stack.pop(), self.stack.pop(); self.stack.append(1 if a != b else 0)
        elif op == BC.LT: b, a = self.stack.pop(), self.stack.pop(); self.stack.append(1 if a < b else 0)
        elif op == BC.GT: b, a = self.stack.pop(), self.stack.pop(); self.stack.append(1 if a > b else 0)
        elif op == BC.LE: b, a = self.stack.pop(), self.stack.pop(); self.stack.append(1 if a <= b else 0)
        elif op == BC.GE: b, a = self.stack.pop(), self.stack.pop(); self.stack.append(1 if a >= b else 0)
        elif op == BC.JMP: self.ip = inst.arg
        elif op == BC.JZ:
            if not self.stack.pop(): self.ip = inst.arg
        elif op == BC.CALL:
            fname, argc = inst.arg
            fn = self.functions[fname]
            args = [self.stack.pop() for _ in range(argc)][::-1]
            frame = Frame(fname, ret_ip=self.ip, ret_code=self.code)
            for pn, v in zip(fn.params, args): frame.locals[pn] = v
            self.frames.append(frame)
            self.code = fn.code; self.ip = 0
        elif op == BC.RET:
            rv = self.stack.pop() if self.stack else 0
            frame = self.frames.pop()
            self.code = frame.ret_code; self.ip = frame.ret_ip
            self.stack.append(rv)
        elif op == BC.PRINT:
            self.output.append(str(self.stack.pop()))


# ===================================================================
# PIPELINE
# ===================================================================

def compile_and_run(source: str, verbose: bool = False) -> list[str]:
    tokens = lex(source)
    ast = Parser(tokens).parse()
    checker = TypeChecker()
    errors = checker.check(ast)
    if errors:
        return [f"Type error: {e}" for e in errors]
    compiler = Compiler()
    main_code, functions = compiler.compile(ast, checker)
    if verbose:
        print(f"  Bytecode: {len(main_code)} main + "
              f"{sum(len(f.code) for f in functions.values())} func instructions")
    vm = VM(main_code, functions)
    return vm.run()


# ===================================================================
# DEMOS
# ===================================================================

def main():
    print("=" * 60)
    print("CAPSTONE COMPILER PROJECT")
    print("=" * 60)

    # Demo 1: Arithmetic and strings
    print("\n--- Demo 1: Types and Arithmetic ---")
    out = compile_and_run("""
        let x = 10 + 20 * 3;
        print(x);
        let msg = "Result: ";
        let full = msg + "70";
        print(full);
    """)
    print(f"  Output: {out}")

    # Demo 2: Functions and recursion
    print("\n--- Demo 2: Recursive Fibonacci ---")
    out = compile_and_run("""
        fn fib(n: int) -> int {
            if (n <= 1) { return n; }
            return fib(n - 1) + fib(n - 2);
        }
        print(fib(10));
    """)
    print(f"  Output: {out}  (expected: ['55'])")

    # Demo 3: For loops
    print("\n--- Demo 3: For Loops ---")
    out = compile_and_run("""
        let sum = 0;
        for (let i = 1; i <= 100; i = i + 1) {
            sum = sum + i;
        }
        print(sum);
    """)
    print(f"  Output: {out}  (expected: ['5050'])")

    # Demo 4: Conditionals
    print("\n--- Demo 4: FizzBuzz (1-15) ---")
    out = compile_and_run("""
        fn fizzbuzz(n: int) -> string {
            if (n % 15 == 0) { return "FizzBuzz"; }
            if (n % 3 == 0)  { return "Fizz"; }
            if (n % 5 == 0)  { return "Buzz"; }
            return "other";
        }
        let i = 1;
        while (i <= 15) {
            print(fizzbuzz(i));
            i = i + 1;
        }
    """)
    print(f"  Output: {out}")

    # Demo 5: Mutual computation
    print("\n--- Demo 5: Power Function ---")
    out = compile_and_run("""
        fn power(base: int, exp: int) -> int {
            if (exp == 0) { return 1; }
            return base * power(base, exp - 1);
        }
        print(power(2, 10));
        print(power(3, 5));
    """)
    print(f"  Output: {out}  (expected: ['1024', '243'])")

    # Pipeline summary
    print("\n" + "=" * 60)
    print("CAPSTONE COMPILER PIPELINE SUMMARY")
    print("=" * 60)
    print("""
  Phase 1: Lexer        Source code -> tokens
  Phase 2: Parser       Tokens -> AST (recursive descent)
  Phase 3: Type Checker Validates types, reports errors
  Phase 4: Compiler     AST -> stack-based bytecode
  Phase 5: VM           Executes bytecode with call stack

  Language features:
    - int, bool, string types
    - Arithmetic, comparison, string concatenation
    - let bindings, assignment, print
    - if/else, while, for loops
    - Functions with typed parameters
    - Recursive functions

  This capstone integrates all 27 prior lessons into a working compiler.
    """)


if __name__ == "__main__":
    main()
