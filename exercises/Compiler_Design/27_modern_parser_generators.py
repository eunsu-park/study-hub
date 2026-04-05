"""
Exercises for Lesson 27: Modern Parser Generators
Topic: Compiler_Design

Implements PEG parser with Packrat memoization and error recovery.
"""

from dataclasses import dataclass
from typing import Optional, List, Any, Tuple


# === AST Nodes ===

@dataclass
class Num:
    value: int
    def __repr__(self): return str(self.value)

@dataclass
class BinOp:
    op: str
    left: Any
    right: Any
    def __repr__(self): return f"({self.left} {self.op} {self.right})"


# === Exercise 1: PEG Parser with Packrat Memoization ===

class PackratParser:
    """PEG parser for arithmetic with memoization."""

    def __init__(self, text):
        self.text = text.replace(" ", "")
        self.pos = 0
        self.memo = {}

    def parse(self):
        result = self.expr()
        if self.pos < len(self.text):
            raise SyntaxError(f"Unexpected '{self.text[self.pos]}' at pos {self.pos}")
        return result

    def _memoize(self, rule_name, func):
        key = (rule_name, self.pos)
        if key in self.memo:
            result, new_pos = self.memo[key]
            self.pos = new_pos
            return result
        result = func()
        self.memo[key] = (result, self.pos)
        return result

    def expr(self):
        return self._memoize("expr", self._expr)

    def _expr(self):
        left = self.term()
        while self.pos < len(self.text) and self.text[self.pos] in '+-':
            op = self.text[self.pos]
            self.pos += 1
            right = self.term()
            left = BinOp(op, left, right)
        return left

    def term(self):
        return self._memoize("term", self._term)

    def _term(self):
        left = self.factor()
        while self.pos < len(self.text) and self.text[self.pos] in '*/':
            op = self.text[self.pos]
            self.pos += 1
            right = self.factor()
            left = BinOp(op, left, right)
        return left

    def factor(self):
        return self._memoize("factor", self._factor)

    def _factor(self):
        if self.pos < len(self.text) and self.text[self.pos] == '(':
            self.pos += 1
            result = self.expr()
            if self.pos < len(self.text) and self.text[self.pos] == ')':
                self.pos += 1
            return result
        return self.number()

    def number(self):
        start = self.pos
        while self.pos < len(self.text) and self.text[self.pos].isdigit():
            self.pos += 1
        if self.pos == start:
            raise SyntaxError(f"Expected number at pos {self.pos}")
        return Num(int(self.text[start:self.pos]))


def exercise_1():
    """PEG parser with Packrat memoization."""
    print("Exercise 1: Packrat PEG Parser")
    print()

    tests = [
        "1 + 2 * 3",
        "(1 + 2) * 3",
        "10 - 3 + 2",
        "2 * 3 + 4 * 5",
    ]

    for expr_str in tests:
        parser = PackratParser(expr_str)
        ast = parser.parse()
        print(f"  {expr_str:20s} -> {ast}")
        print(f"    Memo entries: {len(parser.memo)}")
    print()


# === Exercise 2: Tree-sitter Grammar (design) ===

def exercise_2():
    """Design a tree-sitter grammar for a simple language."""
    print("Exercise 2: Tree-sitter Grammar Design")
    print()
    grammar = '''
module.exports = grammar({
  name: 'simple_lang',
  rules: {
    source_file: $ => repeat($._statement),
    _statement: $ => choice(
      $.assignment,
      $.if_statement,
      $.while_statement,
    ),
    assignment: $ => seq(
      field('name', $.identifier), '=',
      field('value', $.expression), ';'
    ),
    if_statement: $ => seq(
      'if', field('condition', $.expression),
      field('body', $.block),
      optional(seq('else', field('else', $.block)))
    ),
    while_statement: $ => seq(
      'while', field('condition', $.expression),
      field('body', $.block)
    ),
    block: $ => seq('{', repeat($._statement), '}'),
    expression: $ => choice(
      $.binary_expression, $.number, $.identifier,
      seq('(', $.expression, ')')
    ),
    binary_expression: $ => prec.left(1, seq(
      $.expression, choice('+','-','*','/','<','>','=='),
      $.expression
    )),
    number: $ => /\\d+/,
    identifier: $ => /[a-zA-Z_]\\w*/,
  },
});
'''.strip()
    print(grammar)
    print()


# === Exercise 3: Incremental Edit Simulation ===

def exercise_3():
    """Simulate incremental parsing."""
    print("Exercise 3: Incremental Parsing Simulation")
    print()

    print("  Original source:")
    print("    x = 1 + 2;")
    print("    y = x * 3;")
    print()
    print("  Edit: change '2' to '20' at byte offset 8")
    print()
    print("  Changed nodes:")
    print("    - number: '2' -> '20' (leaf changed)")
    print("    - binary_expression: '1 + 2' -> '1 + 20' (parent updated)")
    print("    - assignment: 'x = 1 + 2;' -> 'x = 1 + 20;' (grandparent)")
    print()
    print("  Unchanged nodes:")
    print("    - 'y = x * 3;' (completely unchanged, subtree reused)")
    print()
    print("  tree-sitter API:")
    print("    tree.edit(start_byte=8, old_end_byte=9, new_end_byte=10, ...)")
    print("    new_tree = parser.parse(new_source, tree)")
    print()


# === Exercise 4: Error Recovery Parser ===

class RecoveringParser:
    """Recursive descent parser with synchronization-based error recovery."""

    def __init__(self, tokens):
        self.tokens = tokens
        self.pos = 0
        self.errors = []

    def current(self):
        return self.tokens[self.pos] if self.pos < len(self.tokens) else ('EOF', '')

    def expect(self, ttype):
        if self.current()[0] == ttype:
            tok = self.current()
            self.pos += 1
            return tok
        self.errors.append(f"Expected {ttype}, got {self.current()}")
        return None

    def parse_statements(self):
        stmts = []
        while self.current()[0] != 'EOF':
            try:
                stmt = self.parse_statement()
                if stmt:
                    stmts.append(stmt)
            except SyntaxError as e:
                self.errors.append(str(e))
                self.synchronize()
        return stmts

    def parse_statement(self):
        if self.current()[0] == 'ID':
            name = self.current()[1]
            self.pos += 1
            if not self.expect('ASSIGN'):
                raise SyntaxError(f"Expected '=' after '{name}'")
            val = self.current()[1]
            self.pos += 1
            self.expect('SEMI')
            return ('assign', name, val)
        raise SyntaxError(f"Unexpected token: {self.current()}")

    def synchronize(self):
        """Skip tokens until semicolon or EOF."""
        while self.current()[0] not in ('SEMI', 'EOF'):
            self.pos += 1
        if self.current()[0] == 'SEMI':
            self.pos += 1


def exercise_4():
    """Parser with error recovery."""
    print("Exercise 4: Error Recovery")
    print()

    tokens = [
        ('ID', 'x'), ('ASSIGN', '='), ('NUM', '5'), ('SEMI', ';'),
        ('ID', 'y'), ('NUM', '3'), ('SEMI', ';'),  # missing '='
        ('ID', 'z'), ('ASSIGN', '='), ('NUM', '7'), ('SEMI', ';'),
        ('EOF', ''),
    ]

    parser = RecoveringParser(tokens)
    stmts = parser.parse_statements()

    print(f"  Parsed {len(stmts)} statements:")
    for s in stmts:
        print(f"    {s}")
    print(f"  Errors ({len(parser.errors)}):")
    for e in parser.errors:
        print(f"    {e}")
    print()


# === Exercise 5: Mini LSP Server ===

def exercise_5():
    """Sketch of a minimal LSP server."""
    print("Exercise 5: Mini LSP Server Design")
    print()
    code = '''
import json, sys

symbols = {
    "factorial": {"type": "function", "line": 1, "doc": "int -> int"},
    "main": {"type": "function", "line": 8, "doc": "() -> int"},
}

def handle(request):
    method = request["method"]
    if method == "initialize":
        return {"capabilities": {"hoverProvider": True, "definitionProvider": True}}
    elif method == "textDocument/hover":
        pos = request["params"]["position"]
        word = get_word_at(pos)
        if word in symbols:
            return {"contents": symbols[word]["doc"]}
    elif method == "textDocument/definition":
        pos = request["params"]["position"]
        word = get_word_at(pos)
        if word in symbols:
            return {"uri": "file:///test.mini", "range": {"start": {"line": symbols[word]["line"]}}}
    return None
'''.strip()
    print(code)
    print()


def main():
    for i, ex in enumerate([exercise_1, exercise_2, exercise_3, exercise_4, exercise_5], 1):
        print(f"{'=' * 60}")
        print(f"Exercise {i}")
        print(f"{'=' * 60}")
        ex()


if __name__ == "__main__":
    main()
