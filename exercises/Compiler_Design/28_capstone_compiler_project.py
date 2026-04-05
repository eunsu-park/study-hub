"""
Exercises for Lesson 28: Capstone Compiler Project
Topic: Compiler_Design

A minimal but complete compiler for MiniLang expressions.
Demonstrates: lexer -> parser -> type checker -> evaluator.
"""

from enum import Enum, auto
from dataclasses import dataclass
from typing import List, Optional, Dict, Union


# === Token Types ===

class TT(Enum):
    INT = auto(); BOOL_TRUE = auto(); BOOL_FALSE = auto(); IDENT = auto()
    PLUS = auto(); MINUS = auto(); STAR = auto(); SLASH = auto()
    EQ = auto(); LT = auto(); GT = auto(); ASSIGN = auto()
    LPAREN = auto(); RPAREN = auto(); SEMI = auto(); COLON = auto()
    VAR = auto(); IF = auto(); ELSE = auto(); PRINT = auto()
    LBRACE = auto(); RBRACE = auto(); INT_TYPE = auto(); BOOL_TYPE = auto()
    EOF = auto()


@dataclass
class Token:
    type: TT
    value: str


# === Lexer ===

class Lexer:
    KEYWORDS = {'var': TT.VAR, 'if': TT.IF, 'else': TT.ELSE,
                'print': TT.PRINT, 'true': TT.BOOL_TRUE, 'false': TT.BOOL_FALSE,
                'int': TT.INT_TYPE, 'bool': TT.BOOL_TYPE}

    def __init__(self, src):
        self.src = src
        self.pos = 0

    def tokenize(self):
        tokens = []
        while self.pos < len(self.src):
            c = self.src[self.pos]
            if c in ' \t\n\r':
                self.pos += 1
            elif c.isdigit():
                start = self.pos
                while self.pos < len(self.src) and self.src[self.pos].isdigit():
                    self.pos += 1
                tokens.append(Token(TT.INT, self.src[start:self.pos]))
            elif c.isalpha() or c == '_':
                start = self.pos
                while self.pos < len(self.src) and (self.src[self.pos].isalnum() or self.src[self.pos] == '_'):
                    self.pos += 1
                word = self.src[start:self.pos]
                tokens.append(Token(self.KEYWORDS.get(word, TT.IDENT), word))
            elif c == '=' and self.pos + 1 < len(self.src) and self.src[self.pos + 1] == '=':
                tokens.append(Token(TT.EQ, '==')); self.pos += 2
            elif c == '=':
                tokens.append(Token(TT.ASSIGN, '=')); self.pos += 1
            else:
                simple = {'+': TT.PLUS, '-': TT.MINUS, '*': TT.STAR, '/': TT.SLASH,
                          '<': TT.LT, '>': TT.GT, '(': TT.LPAREN, ')': TT.RPAREN,
                          ';': TT.SEMI, ':': TT.COLON, '{': TT.LBRACE, '}': TT.RBRACE}
                if c in simple:
                    tokens.append(Token(simple[c], c)); self.pos += 1
                else:
                    raise SyntaxError(f"Unexpected: '{c}'")
        tokens.append(Token(TT.EOF, ''))
        return tokens


# === AST ===

@dataclass
class IntLit:
    value: int

@dataclass
class BoolLit:
    value: bool

@dataclass
class Ident:
    name: str

@dataclass
class BinExpr:
    op: str; left: object; right: object

@dataclass
class VarDecl:
    name: str; type_name: str; init: object

@dataclass
class Assign:
    name: str; value: object

@dataclass
class IfStmt:
    cond: object; then_body: list; else_body: list

@dataclass
class PrintStmt:
    value: object


# === Parser ===

class Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.pos = 0

    def cur(self): return self.tokens[self.pos]
    def eat(self, tt):
        if self.cur().type != tt:
            raise SyntaxError(f"Expected {tt}, got {self.cur().type}")
        t = self.cur(); self.pos += 1; return t

    def parse(self):
        stmts = []
        while self.cur().type != TT.EOF:
            stmts.append(self.stmt())
        return stmts

    def stmt(self):
        if self.cur().type == TT.VAR:
            return self.var_decl()
        elif self.cur().type == TT.IF:
            return self.if_stmt()
        elif self.cur().type == TT.PRINT:
            self.eat(TT.PRINT); e = self.expr(); self.eat(TT.SEMI)
            return PrintStmt(e)
        elif self.cur().type == TT.IDENT and self.tokens[self.pos + 1].type == TT.ASSIGN:
            name = self.eat(TT.IDENT).value; self.eat(TT.ASSIGN)
            e = self.expr(); self.eat(TT.SEMI)
            return Assign(name, e)
        raise SyntaxError(f"Unexpected: {self.cur()}")

    def var_decl(self):
        self.eat(TT.VAR); name = self.eat(TT.IDENT).value
        self.eat(TT.COLON); ty = self.eat(self.cur().type).value
        self.eat(TT.ASSIGN); init = self.expr(); self.eat(TT.SEMI)
        return VarDecl(name, ty, init)

    def if_stmt(self):
        self.eat(TT.IF); cond = self.expr()
        self.eat(TT.LBRACE); then_ = []
        while self.cur().type != TT.RBRACE: then_.append(self.stmt())
        self.eat(TT.RBRACE)
        else_ = []
        if self.cur().type == TT.ELSE:
            self.eat(TT.ELSE); self.eat(TT.LBRACE)
            while self.cur().type != TT.RBRACE: else_.append(self.stmt())
            self.eat(TT.RBRACE)
        return IfStmt(cond, then_, else_)

    def expr(self):
        left = self.term()
        while self.cur().type in (TT.PLUS, TT.MINUS, TT.EQ, TT.LT, TT.GT):
            op = self.eat(self.cur().type).value; right = self.term()
            left = BinExpr(op, left, right)
        return left

    def term(self):
        left = self.primary()
        while self.cur().type in (TT.STAR, TT.SLASH):
            op = self.eat(self.cur().type).value; right = self.primary()
            left = BinExpr(op, left, right)
        return left

    def primary(self):
        if self.cur().type == TT.INT:
            return IntLit(int(self.eat(TT.INT).value))
        elif self.cur().type == TT.BOOL_TRUE:
            self.eat(TT.BOOL_TRUE); return BoolLit(True)
        elif self.cur().type == TT.BOOL_FALSE:
            self.eat(TT.BOOL_FALSE); return BoolLit(False)
        elif self.cur().type == TT.IDENT:
            return Ident(self.eat(TT.IDENT).value)
        elif self.cur().type == TT.LPAREN:
            self.eat(TT.LPAREN); e = self.expr(); self.eat(TT.RPAREN); return e
        raise SyntaxError(f"Unexpected: {self.cur()}")


# === Type Checker ===

def type_check(stmts):
    env = {}
    for s in stmts:
        check_stmt(s, env)

def check_stmt(s, env):
    if isinstance(s, VarDecl):
        t = check_expr(s.init, env)
        if t != s.type_name: raise TypeError(f"Cannot assign {t} to {s.type_name}")
        env[s.name] = s.type_name
    elif isinstance(s, Assign):
        if s.name not in env: raise TypeError(f"Undefined: {s.name}")
        t = check_expr(s.value, env)
        if t != env[s.name]: raise TypeError(f"Type mismatch for {s.name}")
    elif isinstance(s, PrintStmt):
        check_expr(s.value, env)
    elif isinstance(s, IfStmt):
        ct = check_expr(s.cond, env)
        if ct != 'bool': raise TypeError(f"If condition must be bool")
        for st in s.then_body: check_stmt(st, dict(env))
        for st in s.else_body: check_stmt(st, dict(env))

def check_expr(e, env):
    if isinstance(e, IntLit): return 'int'
    if isinstance(e, BoolLit): return 'bool'
    if isinstance(e, Ident):
        if e.name not in env: raise TypeError(f"Undefined: {e.name}")
        return env[e.name]
    if isinstance(e, BinExpr):
        lt = check_expr(e.left, env)
        rt = check_expr(e.right, env)
        if e.op in ('+', '-', '*', '/'):
            if lt != 'int' or rt != 'int': raise TypeError(f"Arithmetic requires int")
            return 'int'
        if e.op in ('<', '>', '=='):
            return 'bool'
    raise TypeError(f"Unknown: {e}")


# === Interpreter (stand-in for codegen) ===

def interpret(stmts):
    env = {}
    output = []
    for s in stmts:
        exec_stmt(s, env, output)
    return output

def exec_stmt(s, env, output):
    if isinstance(s, VarDecl): env[s.name] = eval_expr(s.init, env)
    elif isinstance(s, Assign): env[s.name] = eval_expr(s.value, env)
    elif isinstance(s, PrintStmt): output.append(eval_expr(s.value, env))
    elif isinstance(s, IfStmt):
        if eval_expr(s.cond, env):
            for st in s.then_body: exec_stmt(st, env, output)
        else:
            for st in s.else_body: exec_stmt(st, env, output)

def eval_expr(e, env):
    if isinstance(e, IntLit): return e.value
    if isinstance(e, BoolLit): return e.value
    if isinstance(e, Ident): return env[e.name]
    if isinstance(e, BinExpr):
        l, r = eval_expr(e.left, env), eval_expr(e.right, env)
        ops = {'+': lambda: l+r, '-': lambda: l-r, '*': lambda: l*r,
               '/': lambda: l//r, '<': lambda: l<r, '>': lambda: l>r,
               '==': lambda: l==r}
        return ops[e.op]()


# === Main: Complete Pipeline ===

def main():
    source = """
    var x: int = 10;
    var y: int = 3;
    var result: int = x * y + 5;
    print result;
    if result > 30 {
        print 1;
    } else {
        print 0;
    }
    """

    print("=" * 60)
    print("MiniLang Compiler Pipeline")
    print("=" * 60)
    print()
    print(f"Source:\n{source}")

    # Phase 1: Lex
    tokens = Lexer(source).tokenize()
    print(f"Phase 1 - Lexer: {len(tokens)} tokens")
    print(f"  {[(t.type.name, t.value) for t in tokens[:10]]}...")
    print()

    # Phase 2: Parse
    ast = Parser(tokens).parse()
    print(f"Phase 2 - Parser: {len(ast)} statements")
    for s in ast:
        print(f"  {type(s).__name__}: {s}")
    print()

    # Phase 3: Type Check
    type_check(ast)
    print("Phase 3 - Type Checker: OK")
    print()

    # Phase 4: Execute (substitute for codegen)
    output = interpret(ast)
    print(f"Phase 4 - Execution output: {output}")
    print()

    print("In a real compiler, Phase 4 would generate LLVM IR")
    print("and compile it to native code via llc/gcc.")


if __name__ == "__main__":
    main()
