"""
15_tree_walk_interpreter.py - Tree-Walking Interpreter

Demonstrates an interpreter that directly walks the AST to execute
a program, without compiling to bytecode or machine code first.

Components:
  1. Lexer and Parser (minimal, reused from earlier examples)
  2. Tree-Walk Evaluator with environment chains
  3. Closures and first-class functions
  4. Built-in functions (print, len, clock)

The interpreted language supports:
  - Integer and string literals
  - Arithmetic and comparison operators
  - Variables (let bindings)
  - If/else expressions
  - While loops
  - Function definitions and calls
  - Closures (functions capturing outer variables)
  - Recursive functions

Topics covered:
  - Direct AST interpretation vs compilation
  - Environment model (scope chains)
  - Closure representation
  - Tail-call considerations
  - Interpreter vs VM performance trade-offs
"""

from __future__ import annotations
from dataclasses import dataclass, field
from time import time
from typing import Any, Callable, Optional


# ---------------------------------------------------------------------------
# AST Nodes
# ---------------------------------------------------------------------------

@dataclass
class NumLit:
    value: int

@dataclass
class StrLit:
    value: str

@dataclass
class BoolLit:
    value: bool

@dataclass
class Var:
    name: str

@dataclass
class BinExpr:
    op: str
    left: Any
    right: Any

@dataclass
class UnaryExpr:
    op: str
    operand: Any

@dataclass
class CallExpr:
    callee: Any
    args: list

@dataclass
class LetStmt:
    name: str
    value: Any

@dataclass
class AssignStmt:
    name: str
    value: Any

@dataclass
class PrintStmt:
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
class FuncDef:
    name: str
    params: list[str]
    body: list

@dataclass
class ReturnStmt:
    value: Any

@dataclass
class Program:
    statements: list


# ---------------------------------------------------------------------------
# Simple Lexer
# ---------------------------------------------------------------------------

def tokenize(source: str) -> list[tuple[str, Any]]:
    """Tokenize source into (type, value) pairs."""
    tokens = []
    i = 0
    keywords = {"let", "fn", "if", "else", "while", "return",
                "print", "true", "false"}

    while i < len(source):
        ch = source[i]
        if ch in " \t\r\n":
            i += 1
        elif ch == '/' and i + 1 < len(source) and source[i + 1] == '/':
            while i < len(source) and source[i] != '\n':
                i += 1
        elif ch == '"':
            i += 1
            start = i
            while i < len(source) and source[i] != '"':
                i += 1
            tokens.append(("STR", source[start:i]))
            i += 1
        elif ch.isdigit():
            start = i
            while i < len(source) and source[i].isdigit():
                i += 1
            tokens.append(("NUM", int(source[start:i])))
        elif ch.isalpha() or ch == '_':
            start = i
            while i < len(source) and (source[i].isalnum() or source[i] == '_'):
                i += 1
            word = source[start:i]
            if word in ("true", "false"):
                tokens.append(("BOOL", word == "true"))
            elif word in keywords:
                tokens.append((word.upper(), word))
            else:
                tokens.append(("IDENT", word))
        elif source[i:i+2] in ("==", "!=", "<=", ">=", "->"):
            tokens.append(("OP", source[i:i+2]))
            i += 2
        elif ch in "+-*/%<>=!":
            tokens.append(("OP", ch))
            i += 1
        elif ch in "(){},;:":
            tokens.append((ch, ch))
            i += 1
        else:
            raise SyntaxError(f"Unexpected character: {ch!r}")

    tokens.append(("EOF", None))
    return tokens


# ---------------------------------------------------------------------------
# Simple Recursive Descent Parser
# ---------------------------------------------------------------------------

class Parser:
    def __init__(self, tokens: list[tuple[str, Any]]):
        self.tokens = tokens
        self.pos = 0

    def peek(self) -> tuple[str, Any]:
        return self.tokens[self.pos]

    def advance(self) -> tuple[str, Any]:
        tok = self.tokens[self.pos]
        self.pos += 1
        return tok

    def expect(self, typ: str) -> tuple[str, Any]:
        tok = self.advance()
        if tok[0] != typ:
            raise SyntaxError(f"Expected {typ}, got {tok}")
        return tok

    def parse(self) -> Program:
        stmts = []
        while self.peek()[0] != "EOF":
            stmts.append(self.parse_stmt())
        return Program(stmts)

    def parse_stmt(self):
        tt = self.peek()[0]
        if tt == "LET":
            return self.parse_let()
        if tt == "FN":
            return self.parse_func()
        if tt == "IF":
            return self.parse_if()
        if tt == "WHILE":
            return self.parse_while()
        if tt == "RETURN":
            return self.parse_return()
        if tt == "PRINT":
            return self.parse_print()
        if tt == "IDENT":
            name = self.advance()[1]
            self.expect("OP")  # =
            expr = self.parse_expr()
            self.expect(";")
            return AssignStmt(name, expr)
        raise SyntaxError(f"Unexpected token: {self.peek()}")

    def parse_let(self):
        self.expect("LET")
        name = self.expect("IDENT")[1]
        self.expect("OP")  # =
        expr = self.parse_expr()
        self.expect(";")
        return LetStmt(name, expr)

    def parse_func(self):
        self.expect("FN")
        name = self.expect("IDENT")[1]
        self.expect("(")
        params = []
        while self.peek()[0] != ")":
            if params:
                self.expect(",")
            params.append(self.expect("IDENT")[1])
        self.expect(")")
        body = self.parse_block()
        return FuncDef(name, params, body)

    def parse_if(self):
        self.expect("IF")
        self.expect("(")
        cond = self.parse_expr()
        self.expect(")")
        then_body = self.parse_block()
        else_body = []
        if self.peek()[0] == "ELSE":
            self.advance()
            else_body = self.parse_block()
        return IfStmt(cond, then_body, else_body)

    def parse_while(self):
        self.expect("WHILE")
        self.expect("(")
        cond = self.parse_expr()
        self.expect(")")
        body = self.parse_block()
        return WhileStmt(cond, body)

    def parse_return(self):
        self.expect("RETURN")
        expr = self.parse_expr()
        self.expect(";")
        return ReturnStmt(expr)

    def parse_print(self):
        self.expect("PRINT")
        self.expect("(")
        expr = self.parse_expr()
        self.expect(")")
        self.expect(";")
        return PrintStmt(expr)

    def parse_block(self) -> list:
        self.expect("{")
        stmts = []
        while self.peek()[0] != "}":
            stmts.append(self.parse_stmt())
        self.expect("}")
        return stmts

    def parse_expr(self):
        return self.parse_comparison()

    def parse_comparison(self):
        left = self.parse_additive()
        while self.peek() == ("OP", "==") or self.peek() == ("OP", "!=") or \
              self.peek() == ("OP", "<") or self.peek() == ("OP", ">") or \
              self.peek() == ("OP", "<=") or self.peek() == ("OP", ">="):
            op = self.advance()[1]
            right = self.parse_additive()
            left = BinExpr(op, left, right)
        return left

    def parse_additive(self):
        left = self.parse_multiplicative()
        while self.peek() in (("OP", "+"), ("OP", "-")):
            op = self.advance()[1]
            right = self.parse_multiplicative()
            left = BinExpr(op, left, right)
        return left

    def parse_multiplicative(self):
        left = self.parse_unary()
        while self.peek() in (("OP", "*"), ("OP", "/"), ("OP", "%")):
            op = self.advance()[1]
            right = self.parse_unary()
            left = BinExpr(op, left, right)
        return left

    def parse_unary(self):
        if self.peek() == ("OP", "-"):
            self.advance()
            return UnaryExpr("-", self.parse_primary())
        return self.parse_primary()

    def parse_primary(self):
        tok = self.peek()
        if tok[0] == "NUM":
            self.advance()
            return NumLit(tok[1])
        if tok[0] == "STR":
            self.advance()
            return StrLit(tok[1])
        if tok[0] == "BOOL":
            self.advance()
            return BoolLit(tok[1])
        if tok[0] == "IDENT":
            name = self.advance()[1]
            if self.peek()[0] == "(":
                self.advance()
                args = []
                while self.peek()[0] != ")":
                    if args:
                        self.expect(",")
                    args.append(self.parse_expr())
                self.expect(")")
                return CallExpr(Var(name), args)
            return Var(name)
        if tok[0] == "(":
            self.advance()
            expr = self.parse_expr()
            self.expect(")")
            return expr
        raise SyntaxError(f"Unexpected in expression: {tok}")


# ---------------------------------------------------------------------------
# Environment (scope chain)
# ---------------------------------------------------------------------------

class Environment:
    """Variable environment with parent scope chain for closures."""

    def __init__(self, parent: Optional[Environment] = None):
        self.bindings: dict[str, Any] = {}
        self.parent = parent

    def get(self, name: str) -> Any:
        if name in self.bindings:
            return self.bindings[name]
        if self.parent:
            return self.parent.get(name)
        raise NameError(f"Undefined variable: {name}")

    def set(self, name: str, value: Any) -> None:
        self.bindings[name] = value

    def assign(self, name: str, value: Any) -> None:
        """Assign to existing variable, walking up scope chain."""
        if name in self.bindings:
            self.bindings[name] = value
            return
        if self.parent:
            self.parent.assign(name, value)
            return
        raise NameError(f"Undefined variable: {name}")


# ---------------------------------------------------------------------------
# Closure representation
# ---------------------------------------------------------------------------

@dataclass
class Closure:
    """A function paired with its defining environment."""
    name: str
    params: list[str]
    body: list
    env: Environment

    def __repr__(self):
        return f"<fn {self.name}({', '.join(self.params)})>"


class ReturnSignal(Exception):
    """Used to unwind the call stack on return statements."""
    def __init__(self, value: Any):
        self.value = value


# ---------------------------------------------------------------------------
# Tree-Walk Interpreter
# ---------------------------------------------------------------------------

class Interpreter:
    """Evaluates AST nodes by walking the tree directly."""

    def __init__(self):
        self.global_env = Environment()
        self.output: list[str] = []
        self._install_builtins()

    def _install_builtins(self):
        self.global_env.set("clock", lambda: int(time() * 1000))

    def run(self, program: Program) -> list[str]:
        self.execute_block(program.statements, self.global_env)
        return self.output

    def execute_block(self, stmts: list, env: Environment) -> None:
        for stmt in stmts:
            self.execute(stmt, env)

    def execute(self, node: Any, env: Environment) -> None:
        if isinstance(node, LetStmt):
            val = self.evaluate(node.value, env)
            env.set(node.name, val)
        elif isinstance(node, AssignStmt):
            val = self.evaluate(node.value, env)
            env.assign(node.name, val)
        elif isinstance(node, PrintStmt):
            val = self.evaluate(node.expr, env)
            self.output.append(str(val))
        elif isinstance(node, IfStmt):
            if self.evaluate(node.cond, env):
                self.execute_block(node.then_body, Environment(env))
            else:
                self.execute_block(node.else_body, Environment(env))
        elif isinstance(node, WhileStmt):
            while self.evaluate(node.cond, env):
                self.execute_block(node.body, Environment(env))
        elif isinstance(node, FuncDef):
            closure = Closure(node.name, node.params, node.body, env)
            env.set(node.name, closure)
        elif isinstance(node, ReturnStmt):
            val = self.evaluate(node.value, env)
            raise ReturnSignal(val)

    def evaluate(self, node: Any, env: Environment) -> Any:
        if isinstance(node, NumLit):
            return node.value
        if isinstance(node, StrLit):
            return node.value
        if isinstance(node, BoolLit):
            return node.value
        if isinstance(node, Var):
            return env.get(node.name)
        if isinstance(node, UnaryExpr):
            val = self.evaluate(node.operand, env)
            if node.op == "-":
                return -val
            return val
        if isinstance(node, BinExpr):
            left = self.evaluate(node.left, env)
            right = self.evaluate(node.right, env)
            return self._apply_op(node.op, left, right)
        if isinstance(node, CallExpr):
            callee = self.evaluate(node.callee, env)
            args = [self.evaluate(a, env) for a in node.args]
            return self._call(callee, args)
        raise RuntimeError(f"Unknown node type: {type(node).__name__}")

    def _apply_op(self, op: str, left: Any, right: Any) -> Any:
        ops = {
            "+": lambda a, b: a + b,
            "-": lambda a, b: a - b,
            "*": lambda a, b: a * b,
            "/": lambda a, b: a // b if isinstance(a, int) else a / b,
            "%": lambda a, b: a % b,
            "==": lambda a, b: a == b,
            "!=": lambda a, b: a != b,
            "<": lambda a, b: a < b,
            ">": lambda a, b: a > b,
            "<=": lambda a, b: a <= b,
            ">=": lambda a, b: a >= b,
        }
        return ops[op](left, right)

    def _call(self, callee: Any, args: list) -> Any:
        if callable(callee):
            return callee(*args)
        if isinstance(callee, Closure):
            call_env = Environment(callee.env)
            for name, val in zip(callee.params, args):
                call_env.set(name, val)
            try:
                self.execute_block(callee.body, call_env)
            except ReturnSignal as ret:
                return ret.value
            return None
        raise RuntimeError(f"Cannot call {callee}")


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def run_program(source: str) -> list[str]:
    tokens = tokenize(source)
    ast = Parser(tokens).parse()
    interp = Interpreter()
    return interp.run(ast)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Tree-Walking Interpreter Demo")
    print("=" * 60)

    # Demo 1: Basic arithmetic and variables
    print("\n--- Arithmetic and Variables ---")
    out = run_program("""
        let x = 10 + 20 * 3;
        print(x);
        let y = (x - 30) / 2;
        print(y);
    """)
    print(f"  Output: {out}")

    # Demo 2: Functions and recursion
    print("\n--- Recursive Factorial ---")
    out = run_program("""
        fn factorial(n) {
            if (n <= 1) {
                return 1;
            }
            return n * factorial(n - 1);
        }
        print(factorial(5));
        print(factorial(10));
    """)
    print(f"  Output: {out}")

    # Demo 3: Closures
    print("\n--- Closures ---")
    out = run_program("""
        fn make_counter(start) {
            let count = start;
            fn increment() {
                count = count + 1;
                return count;
            }
            return increment;
        }
        let counter = make_counter(0);
        print(counter());
        print(counter());
        print(counter());
    """)
    print(f"  Output: {out}")

    # Demo 4: Fibonacci with while loop
    print("\n--- Fibonacci (iterative) ---")
    out = run_program("""
        fn fib(n) {
            let a = 0;
            let b = 1;
            let i = 0;
            while (i < n) {
                let temp = a + b;
                a = b;
                b = temp;
                i = i + 1;
            }
            return a;
        }
        print(fib(10));
        print(fib(20));
    """)
    print(f"  Output: {out}")

    # Demo 5: String operations
    print("\n--- Strings ---")
    out = run_program("""
        let greeting = "Hello" + " " + "World";
        print(greeting);
    """)
    print(f"  Output: {out}")

    print("\n--- Interpreter vs Compiler Trade-offs ---")
    print("""
  Tree-walk interpreter:
    + Simple implementation, easy to debug
    + No intermediate compilation step
    + Natural support for closures via environment chains
    - Slow: AST traversal overhead on every evaluation
    - No optimization opportunities

  Bytecode compiler + VM:
    + Much faster execution (10-100x)
    + Enables optimizations (constant folding, etc.)
    - More complex implementation
    - Compilation step adds latency for short programs
    """)


if __name__ == "__main__":
    main()
