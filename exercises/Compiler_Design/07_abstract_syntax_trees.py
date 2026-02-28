"""
Exercises for Lesson 07: Abstract Syntax Trees
Topic: Compiler_Design

Solutions to practice problems from the lesson.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Union
from abc import ABC, abstractmethod
import json


# === Shared AST Infrastructure ===

@dataclass
class SourceLocation:
    line: int
    col: int


@dataclass
class ASTNode(ABC):
    loc: Optional[SourceLocation] = field(default=None, repr=False, init=False)

    def __eq__(self, other):
        if type(self) != type(other):
            return False
        # Compare all fields except loc
        for f in self.__dataclass_fields__:
            if f == 'loc':
                continue
            if getattr(self, f) != getattr(other, f):
                return False
        return True

    def __hash__(self):
        return id(self)


# Expressions
@dataclass
class IntLiteral(ASTNode):
    value: int = 0


@dataclass
class FloatLiteral(ASTNode):
    value: float = 0.0


@dataclass
class StringLiteral(ASTNode):
    value: str = ""


@dataclass
class BoolLiteral(ASTNode):
    value: bool = False


@dataclass
class Identifier(ASTNode):
    name: str = ""


@dataclass
class BinaryExpr(ASTNode):
    op: str = ""
    left: ASTNode = field(default_factory=lambda: IntLiteral())
    right: ASTNode = field(default_factory=lambda: IntLiteral())


@dataclass
class UnaryExpr(ASTNode):
    op: str = ""
    operand: ASTNode = field(default_factory=lambda: IntLiteral())


@dataclass
class CallExpr(ASTNode):
    func: ASTNode = field(default_factory=lambda: Identifier())
    args: List[ASTNode] = field(default_factory=list)


# Statements
@dataclass
class LetStmt(ASTNode):
    name: str = ""
    value: ASTNode = field(default_factory=lambda: IntLiteral())


@dataclass
class AssignStmt(ASTNode):
    name: str = ""
    value: ASTNode = field(default_factory=lambda: IntLiteral())


@dataclass
class PrintStmt(ASTNode):
    expr: ASTNode = field(default_factory=lambda: IntLiteral())


@dataclass
class IfStmt(ASTNode):
    condition: ASTNode = field(default_factory=lambda: BoolLiteral())
    then_body: List[ASTNode] = field(default_factory=list)
    else_body: Optional[List[ASTNode]] = None


@dataclass
class WhileStmt(ASTNode):
    condition: ASTNode = field(default_factory=lambda: BoolLiteral())
    body: List[ASTNode] = field(default_factory=list)


@dataclass
class ReturnStmt(ASTNode):
    value: Optional[ASTNode] = None


@dataclass
class FuncDecl(ASTNode):
    name: str = ""
    params: List[str] = field(default_factory=list)
    body: List[ASTNode] = field(default_factory=list)


@dataclass
class Program(ASTNode):
    stmts: List[ASTNode] = field(default_factory=list)


# === Exercise 1: AST Node Design ===
# Problem: Design AST nodes for arrays, dicts, slices, try-catch, pattern matching.

@dataclass
class ArrayLiteral(ASTNode):
    """Array literal: [1, 2, 3]"""
    elements: List[ASTNode] = field(default_factory=list)


@dataclass
class DictLiteral(ASTNode):
    """Dictionary literal: {key: value, ...}"""
    keys: List[ASTNode] = field(default_factory=list)
    values: List[ASTNode] = field(default_factory=list)


@dataclass
class SliceExpr(ASTNode):
    """Slice expression: a[1:5], a[:3], a[::2]"""
    obj: ASTNode = field(default_factory=lambda: Identifier())
    start: Optional[ASTNode] = None
    stop: Optional[ASTNode] = None
    step: Optional[ASTNode] = None


@dataclass
class TryCatchStmt(ASTNode):
    """Try-catch-finally statement."""
    try_body: List[ASTNode] = field(default_factory=list)
    catch_clauses: List['CatchClause'] = field(default_factory=list)
    finally_body: Optional[List[ASTNode]] = None


@dataclass
class CatchClause(ASTNode):
    """Single catch clause: catch (e: Error) { ... }"""
    param_name: str = ""
    error_type: str = ""
    body: List[ASTNode] = field(default_factory=list)


@dataclass
class MatchExpr(ASTNode):
    """Pattern matching: match x { 1 => "one", _ => "other" }"""
    subject: ASTNode = field(default_factory=lambda: Identifier())
    arms: List['MatchArm'] = field(default_factory=list)


@dataclass
class MatchArm(ASTNode):
    """Single match arm: pattern => expression"""
    pattern: ASTNode = field(default_factory=lambda: IntLiteral())
    body: ASTNode = field(default_factory=lambda: IntLiteral())


@dataclass
class WildcardPattern(ASTNode):
    """Wildcard pattern: _"""
    pass


def exercise_1():
    """Design AST node types for advanced language features."""
    print("1. Array literal: [1, 2, 3]")
    arr = ArrayLiteral(elements=[IntLiteral(1), IntLiteral(2), IntLiteral(3)])
    print(f"   {arr}")
    print()

    print("2. Dictionary literal: {\"name\": \"John\", \"age\": 30}")
    d = DictLiteral(
        keys=[StringLiteral("name"), StringLiteral("age")],
        values=[StringLiteral("John"), IntLiteral(30)]
    )
    print(f"   {d}")
    print()

    print("3. Slice expression: a[1:5], a[:3], a[::2]")
    s1 = SliceExpr(obj=Identifier("a"), start=IntLiteral(1), stop=IntLiteral(5))
    s2 = SliceExpr(obj=Identifier("a"), stop=IntLiteral(3))
    s3 = SliceExpr(obj=Identifier("a"), step=IntLiteral(2))
    print(f"   a[1:5] -> {s1}")
    print(f"   a[:3]  -> {s2}")
    print(f"   a[::2] -> {s3}")
    print()

    print("4. Try-catch-finally:")
    tc = TryCatchStmt(
        try_body=[CallExpr(func=Identifier("risky_op"), args=[])],
        catch_clauses=[
            CatchClause(param_name="e", error_type="Error",
                        body=[PrintStmt(expr=Identifier("e"))])
        ],
        finally_body=[CallExpr(func=Identifier("cleanup"), args=[])]
    )
    print(f"   {tc}")
    print()

    print("5. Pattern matching:")
    match = MatchExpr(
        subject=Identifier("x"),
        arms=[
            MatchArm(pattern=IntLiteral(1), body=StringLiteral("one")),
            MatchArm(pattern=IntLiteral(2), body=StringLiteral("two")),
            MatchArm(pattern=WildcardPattern(), body=StringLiteral("other")),
        ]
    )
    print(f"   {match}")


# === Exercise 2: Complete Visitor - DepthCalculator ===
# Problem: Compute the maximum depth of an expression AST.

class ASTVisitor(ABC):
    """Base visitor for AST nodes."""
    def visit(self, node):
        method_name = f"visit_{type(node).__name__}"
        visitor = getattr(self, method_name, self.generic_visit)
        return visitor(node)

    def generic_visit(self, node):
        raise NotImplementedError(f"No visitor for {type(node).__name__}")


class DepthCalculator(ASTVisitor):
    """Compute maximum depth of an expression AST."""

    def visit_IntLiteral(self, node):
        return 1

    def visit_FloatLiteral(self, node):
        return 1

    def visit_StringLiteral(self, node):
        return 1

    def visit_BoolLiteral(self, node):
        return 1

    def visit_Identifier(self, node):
        return 1

    def visit_BinaryExpr(self, node):
        return 1 + max(self.visit(node.left), self.visit(node.right))

    def visit_UnaryExpr(self, node):
        return 1 + self.visit(node.operand)

    def visit_CallExpr(self, node):
        if not node.args:
            return 1 + self.visit(node.func)
        arg_depth = max(self.visit(a) for a in node.args)
        return 1 + max(self.visit(node.func), arg_depth)

    def visit_ArrayLiteral(self, node):
        if not node.elements:
            return 1
        return 1 + max(self.visit(e) for e in node.elements)


def exercise_2():
    """DepthCalculator visitor for AST depth computation."""
    calc = DepthCalculator()

    # IntLiteral(5) -> depth 1
    e1 = IntLiteral(5)
    d1 = calc.visit(e1)
    print(f"  IntLiteral(5) -> depth {d1}")

    # BinaryExpr(ADD, IntLiteral(2), IntLiteral(3)) -> depth 2
    e2 = BinaryExpr("+", IntLiteral(2), IntLiteral(3))
    d2 = calc.visit(e2)
    print(f"  2 + 3 -> depth {d2}")

    # (2 + 3) * 4 -> depth 3
    e3 = BinaryExpr("*", e2, IntLiteral(4))
    d3 = calc.visit(e3)
    print(f"  (2 + 3) * 4 -> depth {d3}")

    # -(2 + 3) -> depth 3
    e4 = UnaryExpr("-", e2)
    d4 = calc.visit(e4)
    print(f"  -(2 + 3) -> depth {d4}")

    # ((1 + 2) * (3 + 4)) + 5 -> depth 4
    e5 = BinaryExpr("+",
        BinaryExpr("*",
            BinaryExpr("+", IntLiteral(1), IntLiteral(2)),
            BinaryExpr("+", IntLiteral(3), IntLiteral(4))),
        IntLiteral(5))
    d5 = calc.visit(e5)
    print(f"  ((1 + 2) * (3 + 4)) + 5 -> depth {d5}")

    # f(1, 2+3) -> depth 3
    e6 = CallExpr(func=Identifier("f"),
                  args=[IntLiteral(1), BinaryExpr("+", IntLiteral(2), IntLiteral(3))])
    d6 = calc.visit(e6)
    print(f"  f(1, 2+3) -> depth {d6}")


# === Exercise 3: Pretty Printer Enhancement ===
# Problem: Handle multi-line function calls, trailing commas, comments.

class PrettyPrinter(ASTVisitor):
    """Pretty printer that can reconstruct source from AST."""

    def __init__(self, max_line_width=80, indent_str="    "):
        self.max_line_width = max_line_width
        self.indent_str = indent_str
        self.indent_level = 0

    def _indent(self):
        return self.indent_str * self.indent_level

    def visit_IntLiteral(self, node):
        return str(node.value)

    def visit_FloatLiteral(self, node):
        return str(node.value)

    def visit_StringLiteral(self, node):
        return f'"{node.value}"'

    def visit_BoolLiteral(self, node):
        return "true" if node.value else "false"

    def visit_Identifier(self, node):
        return node.name

    def visit_BinaryExpr(self, node):
        left = self.visit(node.left)
        right = self.visit(node.right)
        return f"{left} {node.op} {right}"

    def visit_UnaryExpr(self, node):
        operand = self.visit(node.operand)
        return f"{node.op}{operand}"

    def visit_CallExpr(self, node):
        func_name = self.visit(node.func)
        args = [self.visit(a) for a in node.args]

        # Try single-line first
        single_line = f"{func_name}({', '.join(args)})"
        if len(single_line) + len(self._indent()) <= self.max_line_width:
            return single_line

        # Multi-line with trailing comma
        self.indent_level += 1
        indent = self._indent()
        lines = [f"{func_name}("]
        for i, arg in enumerate(args):
            comma = ","  # trailing comma on all items
            lines.append(f"{indent}{arg}{comma}")
        self.indent_level -= 1
        lines.append(f"{self._indent()})")
        return "\n".join(lines)

    def visit_ArrayLiteral(self, node):
        elements = [self.visit(e) for e in node.elements]
        single = f"[{', '.join(elements)}]"
        if len(single) + len(self._indent()) <= self.max_line_width:
            return single

        self.indent_level += 1
        indent = self._indent()
        lines = ["["]
        for elem in elements:
            lines.append(f"{indent}{elem},")  # trailing comma
        self.indent_level -= 1
        lines.append(f"{self._indent()}]")
        return "\n".join(lines)

    def visit_LetStmt(self, node):
        value = self.visit(node.value)
        return f"{self._indent()}let {node.name} = {value};"

    def visit_PrintStmt(self, node):
        expr = self.visit(node.expr)
        return f"{self._indent()}print({expr});"

    def visit_Program(self, node):
        return "\n".join(self.visit(s) for s in node.stmts)


def exercise_3():
    """Enhanced pretty printer with multi-line calls and trailing commas."""
    pp = PrettyPrinter(max_line_width=40)

    # Short call - fits on one line
    short_call = CallExpr(
        func=Identifier("add"),
        args=[IntLiteral(1), IntLiteral(2)]
    )
    print("Short call (fits on one line):")
    print(f"  {pp.visit(short_call)}")
    print()

    # Long call - needs multi-line
    long_call = CallExpr(
        func=Identifier("create_very_long_function"),
        args=[
            StringLiteral("first_argument_value"),
            StringLiteral("second_argument_value"),
            IntLiteral(42),
        ]
    )
    print("Long call (multi-line with trailing commas):")
    result = pp.visit(long_call)
    for line in result.split('\n'):
        print(f"  {line}")
    print()

    # Array literal
    short_array = ArrayLiteral(elements=[IntLiteral(i) for i in range(3)])
    print("Short array:")
    print(f"  {pp.visit(short_array)}")
    print()

    long_array = ArrayLiteral(
        elements=[StringLiteral(f"item_{i}") for i in range(5)]
    )
    print("Long array (multi-line with trailing commas):")
    result = pp.visit(long_array)
    for line in result.split('\n'):
        print(f"  {line}")


# === Exercise 4: Constant Propagation ===
# Problem: Track variable assignments and substitute known constants.

class ConstantPropagator(ASTVisitor):
    """Propagate known constant values through the AST."""

    def __init__(self):
        self.env = {}  # variable name -> known constant value (as ASTNode)

    def visit_IntLiteral(self, node):
        return node

    def visit_FloatLiteral(self, node):
        return node

    def visit_StringLiteral(self, node):
        return node

    def visit_BoolLiteral(self, node):
        return node

    def visit_Identifier(self, node):
        if node.name in self.env:
            return self.env[node.name]
        return node

    def visit_BinaryExpr(self, node):
        left = self.visit(node.left)
        right = self.visit(node.right)

        # Constant folding if both sides are integer literals
        if isinstance(left, IntLiteral) and isinstance(right, IntLiteral):
            ops = {
                '+': lambda a, b: a + b,
                '-': lambda a, b: a - b,
                '*': lambda a, b: a * b,
                '/': lambda a, b: a // b if b != 0 else None,
            }
            if node.op in ops:
                result = ops[node.op](left.value, right.value)
                if result is not None:
                    return IntLiteral(result)

        return BinaryExpr(node.op, left, right)

    def visit_UnaryExpr(self, node):
        operand = self.visit(node.operand)
        if isinstance(operand, IntLiteral) and node.op == '-':
            return IntLiteral(-operand.value)
        return UnaryExpr(node.op, operand)

    def visit_LetStmt(self, node):
        value = self.visit(node.value)
        # Track if value is a constant
        if isinstance(value, (IntLiteral, FloatLiteral, StringLiteral, BoolLiteral)):
            self.env[node.name] = value
        else:
            # Remove from env if reassigned to non-constant
            self.env.pop(node.name, None)
        return LetStmt(node.name, value)

    def visit_PrintStmt(self, node):
        return PrintStmt(self.visit(node.expr))

    def visit_Program(self, node):
        return Program([self.visit(s) for s in node.stmts])


def exercise_4():
    """Constant propagation and folding transformer."""
    pp = PrettyPrinter()

    # Build AST for:
    #   let x = 5;
    #   let y = x + 3;
    #   print(y * 2);
    program = Program(stmts=[
        LetStmt("x", IntLiteral(5)),
        LetStmt("y", BinaryExpr("+", Identifier("x"), IntLiteral(3))),
        PrintStmt(BinaryExpr("*", Identifier("y"), IntLiteral(2))),
    ])

    print("Before constant propagation:")
    print(f"  {pp.visit(program)}")
    print()

    propagator = ConstantPropagator()
    optimized = propagator.visit(program)

    print("After constant propagation + folding:")
    print(f"  {pp.visit(optimized)}")
    print()

    # More complex example
    program2 = Program(stmts=[
        LetStmt("a", IntLiteral(10)),
        LetStmt("b", IntLiteral(20)),
        LetStmt("c", BinaryExpr("+", Identifier("a"), Identifier("b"))),
        LetStmt("d", BinaryExpr("*", Identifier("c"), IntLiteral(2))),
        PrintStmt(BinaryExpr("-", Identifier("d"), IntLiteral(5))),
    ])

    print("Complex example before:")
    print(f"  {pp.visit(program2)}")
    print()

    propagator2 = ConstantPropagator()
    optimized2 = propagator2.visit(program2)
    print("Complex example after:")
    print(f"  {pp.visit(optimized2)}")


# === Exercise 5: AST Diff ===
# Problem: Compute differences between two ASTs.

@dataclass
class Change:
    """Represents a difference between two ASTs."""
    kind: str  # 'modified', 'added', 'removed'
    path: str  # path to the node
    old_value: Optional[str] = None
    new_value: Optional[str] = None


def ast_diff(old, new, path="root"):
    """Compute differences between two AST nodes."""
    changes = []

    if type(old) != type(new):
        changes.append(Change('modified', path,
                              type(old).__name__, type(new).__name__))
        return changes

    if isinstance(old, IntLiteral):
        if old.value != new.value:
            changes.append(Change('modified', f"{path}.value",
                                  str(old.value), str(new.value)))
    elif isinstance(old, StringLiteral):
        if old.value != new.value:
            changes.append(Change('modified', f"{path}.value",
                                  repr(old.value), repr(new.value)))
    elif isinstance(old, Identifier):
        if old.name != new.name:
            changes.append(Change('modified', f"{path}.name",
                                  old.name, new.name))
    elif isinstance(old, BinaryExpr):
        if old.op != new.op:
            changes.append(Change('modified', f"{path}.op", old.op, new.op))
        changes.extend(ast_diff(old.left, new.left, f"{path}.left"))
        changes.extend(ast_diff(old.right, new.right, f"{path}.right"))
    elif isinstance(old, UnaryExpr):
        if old.op != new.op:
            changes.append(Change('modified', f"{path}.op", old.op, new.op))
        changes.extend(ast_diff(old.operand, new.operand, f"{path}.operand"))
    elif isinstance(old, LetStmt):
        if old.name != new.name:
            changes.append(Change('modified', f"{path}.name", old.name, new.name))
        changes.extend(ast_diff(old.value, new.value, f"{path}.value"))
    elif isinstance(old, PrintStmt):
        changes.extend(ast_diff(old.expr, new.expr, f"{path}.expr"))
    elif isinstance(old, Program):
        old_len = len(old.stmts)
        new_len = len(new.stmts)
        for i in range(min(old_len, new_len)):
            changes.extend(ast_diff(old.stmts[i], new.stmts[i], f"{path}.stmts[{i}]"))
        for i in range(old_len, new_len):
            changes.append(Change('added', f"{path}.stmts[{i}]",
                                  new_value=type(new.stmts[i]).__name__))
        for i in range(new_len, old_len):
            changes.append(Change('removed', f"{path}.stmts[{i}]",
                                  old_value=type(old.stmts[i]).__name__))

    return changes


def exercise_5():
    """AST diff: compute differences between two ASTs."""
    old_program = Program(stmts=[
        LetStmt("x", IntLiteral(5)),
        LetStmt("y", BinaryExpr("+", Identifier("x"), IntLiteral(3))),
        PrintStmt(Identifier("y")),
    ])

    new_program = Program(stmts=[
        LetStmt("x", IntLiteral(10)),                              # changed value
        LetStmt("y", BinaryExpr("*", Identifier("x"), IntLiteral(3))),  # changed op
        PrintStmt(BinaryExpr("+", Identifier("y"), IntLiteral(1))),     # changed expr
        LetStmt("z", IntLiteral(0)),                               # added stmt
    ])

    pp = PrettyPrinter()
    print("Old program:")
    print(f"  {pp.visit(old_program)}")
    print()
    print("New program:")
    print(f"  {pp.visit(new_program)}")
    print()

    changes = ast_diff(old_program, new_program)
    print(f"Differences ({len(changes)} changes):")
    for change in changes:
        if change.kind == 'modified':
            print(f"  MODIFIED {change.path}: {change.old_value} -> {change.new_value}")
        elif change.kind == 'added':
            print(f"  ADDED    {change.path}: {change.new_value}")
        elif change.kind == 'removed':
            print(f"  REMOVED  {change.path}: {change.old_value}")


# === Exercise 6: Round-Trip Test ===
# Problem: Parse -> pretty-print -> parse -> verify structural identity.

def simple_parse(source):
    """Minimal parser for round-trip testing."""
    # Supports: let x = expr; and print(expr);
    # Expressions: int literals, identifiers, binary +, *, -
    tokens = source.replace(';', ' ; ').replace('(', ' ( ').replace(')', ' ) ').split()
    pos = [0]

    def peek():
        return tokens[pos[0]] if pos[0] < len(tokens) else None

    def advance():
        t = tokens[pos[0]]
        pos[0] += 1
        return t

    def parse_expr():
        left = parse_term()
        while peek() in ('+', '-'):
            op = advance()
            right = parse_term()
            left = BinaryExpr(op, left, right)
        return left

    def parse_term():
        left = parse_atom()
        while peek() == '*':
            op = advance()
            right = parse_atom()
            left = BinaryExpr(op, left, right)
        return left

    def parse_atom():
        t = peek()
        if t == '(':
            advance()
            expr = parse_expr()
            advance()  # ')'
            return expr
        advance()
        try:
            return IntLiteral(int(t))
        except ValueError:
            return Identifier(t)

    stmts = []
    while pos[0] < len(tokens):
        t = peek()
        if t == 'let':
            advance()
            name = advance()
            advance()  # '='
            value = parse_expr()
            advance()  # ';'
            stmts.append(LetStmt(name, value))
        elif t == 'print':
            advance()
            advance()  # '('
            expr = parse_expr()
            advance()  # ')'
            advance()  # ';'
            stmts.append(PrintStmt(expr))
        else:
            break

    return Program(stmts)


def simple_print(program):
    """Minimal printer for round-trip testing."""
    pp = PrettyPrinter()
    return pp.visit(program)


def exercise_6():
    """Round-trip test: parse -> print -> parse -> compare."""
    test_sources = [
        "let x = 5;",
        "let y = 3 + 4;",
        "print(x);",
        "let a = 1; let b = 2; print(a + b);",
        "let c = 2 * 3 + 4;",
    ]

    for source in test_sources:
        ast1 = simple_parse(source)
        regenerated = simple_print(ast1)
        ast2 = simple_parse(regenerated)

        match = ast1 == ast2
        status = "PASS" if match else "FAIL"
        print(f"  [{status}] '{source}'")
        if not match:
            print(f"         Regenerated: '{regenerated}'")
            print(f"         AST1: {ast1}")
            print(f"         AST2: {ast2}")

    print()
    print("Cases where round-tripping may fail:")
    print("  1. Comments: lost during parsing (not preserved in AST)")
    print("  2. Whitespace: formatting differences between original and regenerated")
    print("  3. Parenthesization: (2 + 3) might round-trip as 2 + 3 if printer")
    print("     relies on precedence instead of explicit parens")
    print("  4. Syntactic sugar: x++ might desugar to x = x + 1 in the AST")
    print()
    print("Mitigation strategies:")
    print("  - Store comments in AST nodes (attach to nearest node)")
    print("  - Use a CST (Concrete Syntax Tree) for lossless representation")
    print("  - Pretty printer should add parens when precedence is ambiguous")
    print("  - Track original formatting hints in AST metadata")


if __name__ == "__main__":
    print("=" * 60)
    print("=== Exercise 1: AST Node Design ===")
    print("=" * 60)
    exercise_1()

    print("\n" + "=" * 60)
    print("=== Exercise 2: Complete Visitor (DepthCalculator) ===")
    print("=" * 60)
    exercise_2()

    print("\n" + "=" * 60)
    print("=== Exercise 3: Pretty Printer Enhancement ===")
    print("=" * 60)
    exercise_3()

    print("\n" + "=" * 60)
    print("=== Exercise 4: Constant Propagation ===")
    print("=" * 60)
    exercise_4()

    print("\n" + "=" * 60)
    print("=== Exercise 5: AST Diff ===")
    print("=" * 60)
    exercise_5()

    print("\n" + "=" * 60)
    print("=== Exercise 6: Round-Trip Test ===")
    print("=" * 60)
    exercise_6()

    print("\nAll exercises completed!")
