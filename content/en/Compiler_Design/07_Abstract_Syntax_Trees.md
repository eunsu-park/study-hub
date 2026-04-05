# Lesson 7: Abstract Syntax Trees

## Learning Objectives

After completing this lesson, you will be able to:

1. **Distinguish** between concrete syntax trees (parse trees) and abstract syntax trees (ASTs)
2. **Design** AST node types using algebraic data types and Python dataclasses
3. **Implement** the visitor pattern for traversing and transforming ASTs
4. **Apply** multiple tree traversal strategies (pre-order, post-order, in-order)
5. **Build** a pretty printer that reconstructs source code from an AST
6. **Perform** AST transformations such as constant folding and desugaring
7. **Track** source locations in AST nodes for error reporting
8. **Serialize** ASTs to JSON and S-expression formats

---

## 1. Introduction: Why ASTs?

When a parser processes source code, it produces a tree representation of the program's structure. There are two kinds of trees we need to distinguish:

### 1.1 Concrete Syntax Tree (Parse Tree)

A **concrete syntax tree** (CST), also called a **parse tree**, faithfully mirrors the grammar. Every nonterminal and terminal appears as a node, including syntactic noise like parentheses, commas, and keywords that exist only for disambiguation.

```
Parse tree for "2 + 3 * 4":

            E
           /|\
          E  +  T
          |    /|\
          T  T  *  F
          |  |     |
          F  F     4
          |  |
          2  3
```

### 1.2 Abstract Syntax Tree (AST)

An **abstract syntax tree** strips away all syntactic details and retains only the **essential structure** of the program. Parentheses, operator-precedence scaffolding, and intermediate nonterminals are all gone.

```
AST for "2 + 3 * 4":

        Add
       /   \
      2    Mul
          /   \
         3     4
```

### 1.3 CST vs AST Comparison

| Aspect | Concrete Syntax Tree | Abstract Syntax Tree |
|--------|---------------------|---------------------|
| Structure | Mirrors grammar exactly | Captures semantic structure |
| Nodes | One per grammar symbol | One per meaningful construct |
| Parentheses | Explicitly represented | Implicit in tree structure |
| Precedence | Encoded via nonterminals ($E$, $T$, $F$) | Encoded by tree depth |
| Size | Larger (many internal nodes) | Smaller (only essential nodes) |
| Use cases | Parsing theory, CST-based tools | Compilers, interpreters, linters |

### 1.4 The AST as Central Data Structure

The AST is the **lingua franca** of compiler internals. Nearly every subsequent phase operates on it:

```
Source Code
    │
    ▼
┌─────────┐
│  Lexer  │
└────┬────┘
     │ tokens
     ▼
┌─────────┐
│ Parser  │
└────┬────┘
     │ AST                    ◄── We are here
     ▼
┌──────────────┐
│ Semantic     │ ──▶ annotated AST (types, scopes)
│ Analysis     │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ IR           │ ──▶ intermediate representation
│ Generation   │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Optimization │
│ & Code Gen   │
└──────────────┘
```

---

## 2. AST Node Design

### 2.1 Algebraic Data Types

In languages with algebraic data types (Haskell, OCaml, Rust), ASTs are naturally expressed as sum types (tagged unions):

```haskell
-- Haskell example
data Expr
    = IntLit Int
    | BoolLit Bool
    | Var String
    | BinOp Op Expr Expr
    | UnaryOp Op Expr
    | Call String [Expr]
    | IfExpr Expr Expr Expr

data Op = Add | Sub | Mul | Div | Eq | Lt | And | Or

data Stmt
    = ExprStmt Expr
    | LetStmt String Expr
    | ReturnStmt (Maybe Expr)
    | Block [Stmt]
    | WhileStmt Expr Stmt
    | IfStmt Expr Stmt (Maybe Stmt)
```

### 2.2 Python Dataclasses

Python does not have native algebraic data types, but we can achieve a similar design using `dataclasses` and inheritance. This is the approach used by CPython's own `ast` module.

```python
"""
AST Node Definitions Using Python Dataclasses

This module defines a complete AST for a small imperative language
with expressions, statements, and a simple type system.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional


# ─── Source Location Tracking ───

@dataclass(frozen=True)
class SourceLocation:
    """Tracks where in the source code a construct appears."""
    line: int
    column: int
    end_line: int = -1
    end_column: int = -1
    filename: str = "<unknown>"

    def __repr__(self):
        if self.end_line > 0:
            return (
                f"{self.filename}:{self.line}:{self.column}"
                f"-{self.end_line}:{self.end_column}"
            )
        return f"{self.filename}:{self.line}:{self.column}"


# ─── Base AST Node ───

@dataclass
class ASTNode:
    """Base class for all AST nodes."""
    loc: Optional[SourceLocation] = field(
        default=None, repr=False, compare=False
    )


# ─── Operators ───

class BinOpKind(Enum):
    ADD = "+"
    SUB = "-"
    MUL = "*"
    DIV = "/"
    MOD = "%"
    POW = "**"
    EQ = "=="
    NE = "!="
    LT = "<"
    GT = ">"
    LE = "<="
    GE = ">="
    AND = "and"
    OR = "or"


class UnaryOpKind(Enum):
    NEG = "-"
    NOT = "not"


# ─── Type Annotations ───

@dataclass
class TypeAnnotation(ASTNode):
    """Base class for type annotations in the AST."""
    pass

@dataclass
class SimpleType(TypeAnnotation):
    """A simple named type: int, float, bool, str."""
    name: str

@dataclass
class ListType(TypeAnnotation):
    """A list type: list[T]."""
    element_type: TypeAnnotation

@dataclass
class FuncType(TypeAnnotation):
    """A function type: (T1, T2) -> R."""
    param_types: list[TypeAnnotation]
    return_type: TypeAnnotation

@dataclass
class OptionalType(TypeAnnotation):
    """An optional type: T?."""
    inner_type: TypeAnnotation


# ─── Expressions ───

@dataclass
class Expr(ASTNode):
    """Base class for all expression nodes."""
    pass

@dataclass
class IntLiteral(Expr):
    """Integer literal: 42."""
    value: int

@dataclass
class FloatLiteral(Expr):
    """Floating-point literal: 3.14."""
    value: float

@dataclass
class BoolLiteral(Expr):
    """Boolean literal: true, false."""
    value: bool

@dataclass
class StringLiteral(Expr):
    """String literal: "hello"."""
    value: str

@dataclass
class NilLiteral(Expr):
    """Nil/null literal."""
    pass

@dataclass
class Identifier(Expr):
    """Variable reference: x, foo."""
    name: str

@dataclass
class BinaryExpr(Expr):
    """Binary operation: a + b, x == y."""
    op: BinOpKind
    left: Expr
    right: Expr

@dataclass
class UnaryExpr(Expr):
    """Unary operation: -x, not y."""
    op: UnaryOpKind
    operand: Expr

@dataclass
class CallExpr(Expr):
    """Function call: f(a, b, c)."""
    callee: Expr
    arguments: list[Expr]

@dataclass
class IndexExpr(Expr):
    """Indexing: a[i]."""
    obj: Expr
    index: Expr

@dataclass
class MemberExpr(Expr):
    """Member access: a.b."""
    obj: Expr
    member: str

@dataclass
class IfExpr(Expr):
    """Conditional expression: if cond then a else b."""
    condition: Expr
    then_expr: Expr
    else_expr: Expr

@dataclass
class ListExpr(Expr):
    """List literal: [1, 2, 3]."""
    elements: list[Expr]

@dataclass
class LambdaExpr(Expr):
    """Lambda expression: fn(x, y) => x + y."""
    params: list[Parameter]
    body: Expr


# ─── Statements ───

@dataclass
class Stmt(ASTNode):
    """Base class for all statement nodes."""
    pass

@dataclass
class Parameter(ASTNode):
    """Function parameter with optional type annotation."""
    name: str
    type_ann: Optional[TypeAnnotation] = None

@dataclass
class ExprStmt(Stmt):
    """Expression used as a statement: f(x);."""
    expr: Expr

@dataclass
class LetStmt(Stmt):
    """Variable declaration: let x: int = 42;."""
    name: str
    type_ann: Optional[TypeAnnotation] = None
    initializer: Optional[Expr] = None

@dataclass
class AssignStmt(Stmt):
    """Assignment: x = expr;."""
    target: Expr
    value: Expr

@dataclass
class ReturnStmt(Stmt):
    """Return statement: return expr;."""
    value: Optional[Expr] = None

@dataclass
class IfStmt(Stmt):
    """If statement: if (cond) { ... } else { ... }."""
    condition: Expr
    then_body: Block
    else_body: Optional[Block] = None

@dataclass
class WhileStmt(Stmt):
    """While loop: while (cond) { ... }."""
    condition: Expr
    body: Block

@dataclass
class ForStmt(Stmt):
    """For loop: for (x in collection) { ... }."""
    var_name: str
    iterable: Expr
    body: Block

@dataclass
class Block(Stmt):
    """Block of statements: { stmt1; stmt2; ... }."""
    statements: list[Stmt]

@dataclass
class FuncDecl(Stmt):
    """Function declaration: fn name(params) -> RetType { body }."""
    name: str
    params: list[Parameter]
    return_type: Optional[TypeAnnotation] = None
    body: Optional[Block] = None

@dataclass
class PrintStmt(Stmt):
    """Print statement: print(expr);."""
    value: Expr


# ─── Program ───

@dataclass
class Program(ASTNode):
    """The top-level AST node representing an entire program."""
    statements: list[Stmt]
```

### 2.3 Design Principles

**1. Favor immutability.** AST nodes should be treated as immutable after construction. Transformations create new nodes rather than modifying existing ones.

**2. Separate expressions and statements.** Even though some languages blur the line (Rust, Kotlin), keeping them separate in the AST simplifies analysis.

**3. Include source locations.** Every node should carry the source position for error reporting. Using `field(default=None, repr=False, compare=False)` keeps the location from cluttering output and comparisons.

**4. Use enums for finite choices.** Operators, type kinds, and other finite sets should be enums, not strings. This prevents typos and enables exhaustive matching.

**5. Keep it minimal.** The AST should not include syntactic sugar. Desugar during parsing or in a separate pass:

| Source Syntax | Desugared AST |
|---|---|
| `x += 5` | `AssignStmt(x, BinaryExpr(ADD, x, 5))` |
| `for x in range(10)` | `WhileStmt(...)` (or keep ForStmt if semantically distinct) |
| `a?.b` | `IfExpr(a != nil, a.b, nil)` |

---

## 3. The Visitor Pattern

### 3.1 Motivation

Once we have an AST, we need to traverse it for various purposes: type checking, code generation, pretty printing, optimization, and more. The **visitor pattern** separates the traversal logic from the AST node definitions, allowing us to add new operations without modifying the AST classes.

### 3.2 Classic Visitor Pattern

```python
"""
Visitor Pattern for AST Traversal

The visitor pattern allows defining new operations on the AST
without modifying the node classes. Each visitor class implements
a visit method for each node type.
"""

from typing import TypeVar, Generic

T = TypeVar("T")


class ASTVisitor(Generic[T]):
    """
    Base visitor class.

    For each AST node type, define a visit_<NodeType> method.
    The generic dispatch method routes based on the node's class name.
    """

    def visit(self, node: ASTNode) -> T:
        """Dispatch to the appropriate visit method."""
        method_name = f"visit_{type(node).__name__}"
        visitor_method = getattr(self, method_name, self.generic_visit)
        return visitor_method(node)

    def generic_visit(self, node: ASTNode) -> T:
        """Called when no specific visitor method exists."""
        raise NotImplementedError(
            f"No visit_{type(node).__name__} method defined "
            f"in {type(self).__name__}"
        )


class ExprVisitor(Generic[T]):
    """Visitor specialized for expression nodes."""

    def visit(self, node: Expr) -> T:
        method_name = f"visit_{type(node).__name__}"
        visitor_method = getattr(self, method_name, self.generic_visit)
        return visitor_method(node)

    def generic_visit(self, node: Expr) -> T:
        raise NotImplementedError(
            f"No visit_{type(node).__name__} method"
        )

    def visit_IntLiteral(self, node: IntLiteral) -> T:
        return self.generic_visit(node)

    def visit_FloatLiteral(self, node: FloatLiteral) -> T:
        return self.generic_visit(node)

    def visit_BoolLiteral(self, node: BoolLiteral) -> T:
        return self.generic_visit(node)

    def visit_StringLiteral(self, node: StringLiteral) -> T:
        return self.generic_visit(node)

    def visit_Identifier(self, node: Identifier) -> T:
        return self.generic_visit(node)

    def visit_BinaryExpr(self, node: BinaryExpr) -> T:
        return self.generic_visit(node)

    def visit_UnaryExpr(self, node: UnaryExpr) -> T:
        return self.generic_visit(node)

    def visit_CallExpr(self, node: CallExpr) -> T:
        return self.generic_visit(node)

    def visit_IfExpr(self, node: IfExpr) -> T:
        return self.generic_visit(node)
```

### 3.3 Evaluator Visitor

A simple interpreter that evaluates expressions by visiting nodes:

```python
class Evaluator(ExprVisitor[object]):
    """
    Evaluates expression ASTs to produce values.

    This is the simplest useful visitor: a tree-walking interpreter.
    """

    def __init__(self):
        self.environment: dict[str, object] = {}

    def visit_IntLiteral(self, node: IntLiteral) -> int:
        return node.value

    def visit_FloatLiteral(self, node: FloatLiteral) -> float:
        return node.value

    def visit_BoolLiteral(self, node: BoolLiteral) -> bool:
        return node.value

    def visit_StringLiteral(self, node: StringLiteral) -> str:
        return node.value

    def visit_Identifier(self, node: Identifier) -> object:
        if node.name not in self.environment:
            raise NameError(f"Undefined variable: {node.name}")
        return self.environment[node.name]

    def visit_BinaryExpr(self, node: BinaryExpr) -> object:
        left = self.visit(node.left)
        right = self.visit(node.right)

        ops = {
            BinOpKind.ADD: lambda a, b: a + b,
            BinOpKind.SUB: lambda a, b: a - b,
            BinOpKind.MUL: lambda a, b: a * b,
            BinOpKind.DIV: lambda a, b: a / b,
            BinOpKind.MOD: lambda a, b: a % b,
            BinOpKind.POW: lambda a, b: a ** b,
            BinOpKind.EQ: lambda a, b: a == b,
            BinOpKind.NE: lambda a, b: a != b,
            BinOpKind.LT: lambda a, b: a < b,
            BinOpKind.GT: lambda a, b: a > b,
            BinOpKind.LE: lambda a, b: a <= b,
            BinOpKind.GE: lambda a, b: a >= b,
            BinOpKind.AND: lambda a, b: a and b,
            BinOpKind.OR: lambda a, b: a or b,
        }

        op_func = ops.get(node.op)
        if op_func is None:
            raise ValueError(f"Unknown operator: {node.op}")
        return op_func(left, right)

    def visit_UnaryExpr(self, node: UnaryExpr) -> object:
        operand = self.visit(node.operand)
        if node.op == UnaryOpKind.NEG:
            return -operand
        elif node.op == UnaryOpKind.NOT:
            return not operand
        raise ValueError(f"Unknown unary operator: {node.op}")

    def visit_IfExpr(self, node: IfExpr) -> object:
        condition = self.visit(node.condition)
        if condition:
            return self.visit(node.then_expr)
        else:
            return self.visit(node.else_expr)


# ─── Demo ───

if __name__ == "__main__":
    # Build AST for: (2 + 3) * 4
    ast = BinaryExpr(
        op=BinOpKind.MUL,
        left=BinaryExpr(
            op=BinOpKind.ADD,
            left=IntLiteral(2),
            right=IntLiteral(3),
        ),
        right=IntLiteral(4),
    )

    evaluator = Evaluator()
    result = evaluator.visit(ast)
    print(f"(2 + 3) * 4 = {result}")  # Output: 20

    # Build AST for: if 3 > 2 then 10 else 20
    ast2 = IfExpr(
        condition=BinaryExpr(
            BinOpKind.GT,
            IntLiteral(3),
            IntLiteral(2),
        ),
        then_expr=IntLiteral(10),
        else_expr=IntLiteral(20),
    )

    result2 = evaluator.visit(ast2)
    print(f"if 3 > 2 then 10 else 20 = {result2}")  # Output: 10
```

### 3.4 Statement Visitor (Full Interpreter)

```python
class Interpreter(ASTVisitor[None]):
    """
    A tree-walking interpreter that executes statement ASTs.

    Handles variable declarations, assignments, control flow,
    and function calls.
    """

    def __init__(self):
        self.env: dict[str, object] = {}
        self.expr_eval = Evaluator()
        self.expr_eval.environment = self.env
        self.functions: dict[str, FuncDecl] = {}

    def eval_expr(self, expr: Expr) -> object:
        return self.expr_eval.visit(expr)

    def visit_Program(self, node: Program):
        for stmt in node.statements:
            self.visit(stmt)

    def visit_Block(self, node: Block):
        for stmt in node.statements:
            self.visit(stmt)

    def visit_ExprStmt(self, node: ExprStmt):
        self.eval_expr(node.expr)

    def visit_LetStmt(self, node: LetStmt):
        value = None
        if node.initializer is not None:
            value = self.eval_expr(node.initializer)
        self.env[node.name] = value

    def visit_AssignStmt(self, node: AssignStmt):
        if isinstance(node.target, Identifier):
            value = self.eval_expr(node.value)
            self.env[node.target.name] = value
        else:
            raise RuntimeError("Can only assign to identifiers")

    def visit_PrintStmt(self, node: PrintStmt):
        value = self.eval_expr(node.value)
        print(value)

    def visit_IfStmt(self, node: IfStmt):
        condition = self.eval_expr(node.condition)
        if condition:
            self.visit(node.then_body)
        elif node.else_body is not None:
            self.visit(node.else_body)

    def visit_WhileStmt(self, node: WhileStmt):
        while self.eval_expr(node.condition):
            self.visit(node.body)

    def visit_FuncDecl(self, node: FuncDecl):
        self.functions[node.name] = node


# ─── Demo ───

if __name__ == "__main__":
    # Build AST for:
    #   let x = 10;
    #   let y = 20;
    #   if (x < y) { print(x + y); } else { print(0); }
    program = Program(statements=[
        LetStmt("x", initializer=IntLiteral(10)),
        LetStmt("y", initializer=IntLiteral(20)),
        IfStmt(
            condition=BinaryExpr(
                BinOpKind.LT,
                Identifier("x"),
                Identifier("y"),
            ),
            then_body=Block([
                PrintStmt(BinaryExpr(
                    BinOpKind.ADD,
                    Identifier("x"),
                    Identifier("y"),
                ))
            ]),
            else_body=Block([
                PrintStmt(IntLiteral(0))
            ]),
        ),
    ])

    interpreter = Interpreter()
    interpreter.visit(program)
    # Output: 30
```

---

## 4. Tree Traversal Strategies

### 4.1 Pre-Order Traversal

Visit the node **before** its children. Used for: printing, serialization, tree copying.

$$\text{visit}(n) = \text{process}(n); \quad \text{visit}(n.\text{child}_1); \quad \text{visit}(n.\text{child}_2); \quad \ldots$$

```python
def preorder(node: ASTNode, depth: int = 0):
    """Pre-order traversal: visit node, then children."""
    print("  " * depth + type(node).__name__)

    if isinstance(node, BinaryExpr):
        preorder(node.left, depth + 1)
        preorder(node.right, depth + 1)
    elif isinstance(node, UnaryExpr):
        preorder(node.operand, depth + 1)
    elif isinstance(node, CallExpr):
        preorder(node.callee, depth + 1)
        for arg in node.arguments:
            preorder(arg, depth + 1)
    elif isinstance(node, Block):
        for stmt in node.statements:
            preorder(stmt, depth + 1)
    # ... other node types
```

### 4.2 Post-Order Traversal

Visit children **before** the node. Used for: evaluation, code generation, computing synthesized attributes.

$$\text{visit}(n) = \text{visit}(n.\text{child}_1); \quad \text{visit}(n.\text{child}_2); \quad \ldots; \quad \text{process}(n)$$

```python
def postorder_eval(node: Expr) -> object:
    """Post-order evaluation: evaluate children, then compute node."""
    if isinstance(node, IntLiteral):
        return node.value
    elif isinstance(node, BinaryExpr):
        # Evaluate children first (post-order)
        left_val = postorder_eval(node.left)
        right_val = postorder_eval(node.right)
        # Then process this node
        if node.op == BinOpKind.ADD:
            return left_val + right_val
        elif node.op == BinOpKind.MUL:
            return left_val * right_val
        # ... other operators
    raise ValueError(f"Cannot evaluate: {type(node).__name__}")
```

### 4.3 In-Order Traversal

Visit left child, then node, then right child. Primarily used for: printing infix expressions, binary search trees.

$$\text{visit}(n) = \text{visit}(n.\text{left}); \quad \text{process}(n); \quad \text{visit}(n.\text{right})$$

```python
def inorder_print(node: Expr, needs_parens: bool = False) -> str:
    """In-order traversal to produce infix notation."""
    if isinstance(node, IntLiteral):
        return str(node.value)
    elif isinstance(node, Identifier):
        return node.name
    elif isinstance(node, BinaryExpr):
        left_str = inorder_print(node.left, True)
        right_str = inorder_print(node.right, True)
        result = f"{left_str} {node.op.value} {right_str}"
        if needs_parens:
            result = f"({result})"
        return result
    return "?"
```

### 4.4 Generic Tree Walker

A reusable walker that visits all children of any AST node:

```python
import dataclasses


class TreeWalker(ASTVisitor[None]):
    """
    Generic walker that visits all children of every node.

    Override specific visit methods to add behavior.
    Default behavior: just recurse into children.
    """

    def generic_visit(self, node: ASTNode) -> None:
        """Visit all children by inspecting dataclass fields."""
        if not dataclasses.is_dataclass(node):
            return

        for f in dataclasses.fields(node):
            if f.name == "loc":
                continue

            value = getattr(node, f.name)
            if isinstance(value, ASTNode):
                self.visit(value)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, ASTNode):
                        self.visit(item)


class VariableCollector(TreeWalker):
    """Collects all variable names referenced in the AST."""

    def __init__(self):
        self.variables: set[str] = set()

    def visit_Identifier(self, node: Identifier):
        self.variables.add(node.name)


# Usage:
# collector = VariableCollector()
# collector.visit(program_ast)
# print(collector.variables)  # {'x', 'y', 'z', ...}
```

---

## 5. Pretty Printing

### 5.1 What is Pretty Printing?

**Pretty printing** converts an AST back into human-readable source code. It is the inverse of parsing. A good pretty printer:

- Produces syntactically valid output
- Handles indentation consistently
- Minimizes unnecessary parentheses
- Preserves the program's semantics

### 5.2 Implementation

```python
"""
AST Pretty Printer

Converts AST nodes back into readable source code.
Handles indentation, operator precedence (for minimal parentheses),
and multi-line formatting.
"""


class PrettyPrinter(ASTVisitor[str]):
    """
    Pretty-prints AST nodes to source code strings.
    """

    def __init__(self, indent_str: str = "    "):
        self.indent_str = indent_str
        self.indent_level = 0

    @property
    def indent(self) -> str:
        return self.indent_str * self.indent_level

    # ─── Operator precedence for minimal parentheses ───

    PRECEDENCE = {
        BinOpKind.OR: 1,
        BinOpKind.AND: 2,
        BinOpKind.EQ: 3, BinOpKind.NE: 3,
        BinOpKind.LT: 4, BinOpKind.GT: 4,
        BinOpKind.LE: 4, BinOpKind.GE: 4,
        BinOpKind.ADD: 5, BinOpKind.SUB: 5,
        BinOpKind.MUL: 6, BinOpKind.DIV: 6, BinOpKind.MOD: 6,
        BinOpKind.POW: 7,
    }

    def _expr_precedence(self, node: Expr) -> int:
        if isinstance(node, BinaryExpr):
            return self.PRECEDENCE.get(node.op, 0)
        return 100  # literals, identifiers, calls: never need parens

    def _paren_if_needed(
        self, child: Expr, parent_prec: int, is_right: bool = False
    ) -> str:
        """Add parentheses if child has lower precedence than parent."""
        child_str = self.visit(child)
        child_prec = self._expr_precedence(child)

        needs_parens = False
        if child_prec < parent_prec:
            needs_parens = True
        elif child_prec == parent_prec and is_right:
            # For left-associative operators, right child at same
            # precedence needs parens: a - (b - c)
            if isinstance(child, BinaryExpr) and child.op in (
                BinOpKind.SUB, BinOpKind.DIV, BinOpKind.MOD
            ):
                needs_parens = True

        return f"({child_str})" if needs_parens else child_str

    # ─── Expressions ───

    def visit_IntLiteral(self, node: IntLiteral) -> str:
        return str(node.value)

    def visit_FloatLiteral(self, node: FloatLiteral) -> str:
        return str(node.value)

    def visit_BoolLiteral(self, node: BoolLiteral) -> str:
        return "true" if node.value else "false"

    def visit_StringLiteral(self, node: StringLiteral) -> str:
        # Escape special characters
        escaped = (
            node.value
            .replace("\\", "\\\\")
            .replace('"', '\\"')
            .replace("\n", "\\n")
            .replace("\t", "\\t")
        )
        return f'"{escaped}"'

    def visit_NilLiteral(self, node: NilLiteral) -> str:
        return "nil"

    def visit_Identifier(self, node: Identifier) -> str:
        return node.name

    def visit_BinaryExpr(self, node: BinaryExpr) -> str:
        prec = self.PRECEDENCE.get(node.op, 0)
        left_str = self._paren_if_needed(node.left, prec)
        right_str = self._paren_if_needed(node.right, prec, is_right=True)
        return f"{left_str} {node.op.value} {right_str}"

    def visit_UnaryExpr(self, node: UnaryExpr) -> str:
        operand_str = self.visit(node.operand)
        if node.op == UnaryOpKind.NEG:
            if isinstance(node.operand, BinaryExpr):
                return f"-({operand_str})"
            return f"-{operand_str}"
        elif node.op == UnaryOpKind.NOT:
            return f"not {operand_str}"
        return f"{node.op.value}{operand_str}"

    def visit_CallExpr(self, node: CallExpr) -> str:
        callee_str = self.visit(node.callee)
        args_str = ", ".join(self.visit(arg) for arg in node.arguments)
        return f"{callee_str}({args_str})"

    def visit_IndexExpr(self, node: IndexExpr) -> str:
        obj_str = self.visit(node.obj)
        idx_str = self.visit(node.index)
        return f"{obj_str}[{idx_str}]"

    def visit_MemberExpr(self, node: MemberExpr) -> str:
        obj_str = self.visit(node.obj)
        return f"{obj_str}.{node.member}"

    def visit_IfExpr(self, node: IfExpr) -> str:
        cond = self.visit(node.condition)
        then = self.visit(node.then_expr)
        els = self.visit(node.else_expr)
        return f"if {cond} then {then} else {els}"

    def visit_ListExpr(self, node: ListExpr) -> str:
        elems = ", ".join(self.visit(e) for e in node.elements)
        return f"[{elems}]"

    def visit_LambdaExpr(self, node: LambdaExpr) -> str:
        params = ", ".join(self._format_param(p) for p in node.params)
        body = self.visit(node.body)
        return f"fn({params}) => {body}"

    # ─── Statements ───

    def visit_Program(self, node: Program) -> str:
        return "\n".join(self.visit(stmt) for stmt in node.statements)

    def visit_Block(self, node: Block) -> str:
        if not node.statements:
            return f"{self.indent}{{}}"

        lines = [f"{self.indent}{{"]
        self.indent_level += 1
        for stmt in node.statements:
            lines.append(self.visit(stmt))
        self.indent_level -= 1
        lines.append(f"{self.indent}}}")
        return "\n".join(lines)

    def visit_ExprStmt(self, node: ExprStmt) -> str:
        return f"{self.indent}{self.visit(node.expr)};"

    def visit_LetStmt(self, node: LetStmt) -> str:
        parts = [f"{self.indent}let {node.name}"]
        if node.type_ann is not None:
            parts.append(f": {self.visit(node.type_ann)}")
        if node.initializer is not None:
            parts.append(f" = {self.visit(node.initializer)}")
        parts.append(";")
        return "".join(parts)

    def visit_AssignStmt(self, node: AssignStmt) -> str:
        target = self.visit(node.target)
        value = self.visit(node.value)
        return f"{self.indent}{target} = {value};"

    def visit_ReturnStmt(self, node: ReturnStmt) -> str:
        if node.value is not None:
            return f"{self.indent}return {self.visit(node.value)};"
        return f"{self.indent}return;"

    def visit_IfStmt(self, node: IfStmt) -> str:
        cond = self.visit(node.condition)
        then_str = self.visit(node.then_body)
        result = f"{self.indent}if ({cond}) {then_str.lstrip()}"
        if node.else_body is not None:
            else_str = self.visit(node.else_body)
            result += f" else {else_str.lstrip()}"
        return result

    def visit_WhileStmt(self, node: WhileStmt) -> str:
        cond = self.visit(node.condition)
        body = self.visit(node.body)
        return f"{self.indent}while ({cond}) {body.lstrip()}"

    def visit_ForStmt(self, node: ForStmt) -> str:
        iterable = self.visit(node.iterable)
        body = self.visit(node.body)
        return (
            f"{self.indent}for ({node.var_name} in {iterable}) "
            f"{body.lstrip()}"
        )

    def visit_FuncDecl(self, node: FuncDecl) -> str:
        params = ", ".join(self._format_param(p) for p in node.params)
        ret = ""
        if node.return_type is not None:
            ret = f" -> {self.visit(node.return_type)}"
        if node.body is not None:
            body = self.visit(node.body)
            return (
                f"{self.indent}fn {node.name}({params}){ret} "
                f"{body.lstrip()}"
            )
        return f"{self.indent}fn {node.name}({params}){ret};"

    def visit_PrintStmt(self, node: PrintStmt) -> str:
        return f"{self.indent}print({self.visit(node.value)});"

    # ─── Types ───

    def visit_SimpleType(self, node: SimpleType) -> str:
        return node.name

    def visit_ListType(self, node: ListType) -> str:
        inner = self.visit(node.element_type)
        return f"list[{inner}]"

    def visit_FuncType(self, node: FuncType) -> str:
        params = ", ".join(self.visit(t) for t in node.param_types)
        ret = self.visit(node.return_type)
        return f"({params}) -> {ret}"

    def visit_OptionalType(self, node: OptionalType) -> str:
        return f"{self.visit(node.inner_type)}?"

    # ─── Helpers ───

    def _format_param(self, param: Parameter) -> str:
        if param.type_ann is not None:
            return f"{param.name}: {self.visit(param.type_ann)}"
        return param.name


# ─── Demo ───

if __name__ == "__main__":
    program = Program(statements=[
        FuncDecl(
            name="factorial",
            params=[Parameter("n", SimpleType("int"))],
            return_type=SimpleType("int"),
            body=Block([
                IfStmt(
                    condition=BinaryExpr(
                        BinOpKind.LE,
                        Identifier("n"),
                        IntLiteral(1),
                    ),
                    then_body=Block([ReturnStmt(IntLiteral(1))]),
                    else_body=Block([
                        ReturnStmt(BinaryExpr(
                            BinOpKind.MUL,
                            Identifier("n"),
                            CallExpr(
                                Identifier("factorial"),
                                [BinaryExpr(
                                    BinOpKind.SUB,
                                    Identifier("n"),
                                    IntLiteral(1),
                                )],
                            ),
                        ))
                    ]),
                ),
            ]),
        ),
        LetStmt("result", SimpleType("int"),
                CallExpr(Identifier("factorial"), [IntLiteral(5)])),
        PrintStmt(Identifier("result")),
    ])

    printer = PrettyPrinter()
    print(printer.visit(program))
```

**Output:**

```
fn factorial(n: int) -> int {
    if (n <= 1) {
        return 1;
    } else {
        return n * factorial(n - 1);
    }
}
let result: int = factorial(5);
print(result);
```

---

## 6. AST Transformations

### 6.1 Constant Folding

**Constant folding** evaluates expressions with known values at compile time:

```python
class ConstantFolder(ASTVisitor[ASTNode]):
    """
    Constant folding: evaluate constant expressions at compile time.

    Examples:
        2 + 3       =>  5
        4 * 1       =>  4 (identity)
        0 * x       =>  0 (zero multiplication)
        x + 0       =>  x (identity)
        true and x  =>  x (short-circuit simplification)
    """

    def visit_IntLiteral(self, node: IntLiteral) -> Expr:
        return node

    def visit_FloatLiteral(self, node: FloatLiteral) -> Expr:
        return node

    def visit_BoolLiteral(self, node: BoolLiteral) -> Expr:
        return node

    def visit_StringLiteral(self, node: StringLiteral) -> Expr:
        return node

    def visit_Identifier(self, node: Identifier) -> Expr:
        return node

    def visit_BinaryExpr(self, node: BinaryExpr) -> Expr:
        left = self.visit(node.left)
        right = self.visit(node.right)

        # Both operands are integer literals: compute at compile time
        if isinstance(left, IntLiteral) and isinstance(right, IntLiteral):
            try:
                result = self._eval_int_op(node.op, left.value, right.value)
                if isinstance(result, bool):
                    return BoolLiteral(result, loc=node.loc)
                return IntLiteral(result, loc=node.loc)
            except (ZeroDivisionError, OverflowError):
                pass  # Cannot fold; leave as-is

        # Identity simplifications
        if node.op == BinOpKind.ADD:
            if isinstance(left, IntLiteral) and left.value == 0:
                return right
            if isinstance(right, IntLiteral) and right.value == 0:
                return left

        if node.op == BinOpKind.MUL:
            if isinstance(left, IntLiteral) and left.value == 1:
                return right
            if isinstance(right, IntLiteral) and right.value == 1:
                return left
            if isinstance(left, IntLiteral) and left.value == 0:
                return IntLiteral(0, loc=node.loc)
            if isinstance(right, IntLiteral) and right.value == 0:
                return IntLiteral(0, loc=node.loc)

        if node.op == BinOpKind.SUB:
            if isinstance(right, IntLiteral) and right.value == 0:
                return left

        # Return simplified node
        return BinaryExpr(node.op, left, right, loc=node.loc)

    def visit_UnaryExpr(self, node: UnaryExpr) -> Expr:
        operand = self.visit(node.operand)

        if node.op == UnaryOpKind.NEG and isinstance(operand, IntLiteral):
            return IntLiteral(-operand.value, loc=node.loc)

        if node.op == UnaryOpKind.NOT and isinstance(operand, BoolLiteral):
            return BoolLiteral(not operand.value, loc=node.loc)

        # Double negation elimination: --x => x
        if (
            node.op == UnaryOpKind.NEG
            and isinstance(operand, UnaryExpr)
            and operand.op == UnaryOpKind.NEG
        ):
            return operand.operand

        return UnaryExpr(node.op, operand, loc=node.loc)

    def _eval_int_op(self, op: BinOpKind, a: int, b: int):
        ops = {
            BinOpKind.ADD: lambda: a + b,
            BinOpKind.SUB: lambda: a - b,
            BinOpKind.MUL: lambda: a * b,
            BinOpKind.DIV: lambda: a // b,
            BinOpKind.MOD: lambda: a % b,
            BinOpKind.POW: lambda: a ** b,
            BinOpKind.EQ: lambda: a == b,
            BinOpKind.NE: lambda: a != b,
            BinOpKind.LT: lambda: a < b,
            BinOpKind.GT: lambda: a > b,
            BinOpKind.LE: lambda: a <= b,
            BinOpKind.GE: lambda: a >= b,
        }
        return ops[op]()

    # Pass through for other nodes (transform children)

    def visit_CallExpr(self, node: CallExpr) -> Expr:
        callee = self.visit(node.callee)
        args = [self.visit(arg) for arg in node.arguments]
        return CallExpr(callee, args, loc=node.loc)

    def visit_IfExpr(self, node: IfExpr) -> Expr:
        cond = self.visit(node.condition)
        if isinstance(cond, BoolLiteral):
            if cond.value:
                return self.visit(node.then_expr)
            else:
                return self.visit(node.else_expr)
        return IfExpr(
            cond, self.visit(node.then_expr),
            self.visit(node.else_expr), loc=node.loc,
        )


# ─── Demo ───

if __name__ == "__main__":
    # AST for: (2 + 3) * x + 0
    ast = BinaryExpr(
        BinOpKind.ADD,
        BinaryExpr(
            BinOpKind.MUL,
            BinaryExpr(BinOpKind.ADD, IntLiteral(2), IntLiteral(3)),
            Identifier("x"),
        ),
        IntLiteral(0),
    )

    printer = PrettyPrinter()
    print(f"Before: {printer.visit(ast)}")
    # Before: (2 + 3) * x + 0

    folder = ConstantFolder()
    folded = folder.visit(ast)
    print(f"After:  {printer.visit(folded)}")
    # After:  5 * x
```

### 6.2 Desugaring

**Desugaring** transforms syntactic sugar into simpler, core constructs:

```python
class Desugarer(ASTVisitor[ASTNode]):
    """
    Transform syntactic sugar into core language constructs.

    Examples:
        for (x in list) { body }
            => let _iter = list;
               let _i = 0;
               while (_i < len(_iter)) {
                   let x = _iter[_i];
                   body;
                   _i = _i + 1;
               }

        x += 5  =>  x = x + 5
    """

    def __init__(self):
        self._temp_counter = 0

    def _fresh_name(self, prefix: str = "_tmp") -> str:
        self._temp_counter += 1
        return f"{prefix}{self._temp_counter}"

    def visit_ForStmt(self, node: ForStmt) -> Block:
        """Desugar for-in loop to while loop."""
        iter_name = self._fresh_name("_iter")
        idx_name = self._fresh_name("_i")

        iterable = self.visit(node.iterable)
        body = self.visit(node.body)

        return Block(statements=[
            # let _iter = iterable;
            LetStmt(iter_name, initializer=iterable),
            # let _i = 0;
            LetStmt(idx_name, initializer=IntLiteral(0)),
            # while (_i < len(_iter)) { ... }
            WhileStmt(
                condition=BinaryExpr(
                    BinOpKind.LT,
                    Identifier(idx_name),
                    CallExpr(Identifier("len"), [Identifier(iter_name)]),
                ),
                body=Block(statements=[
                    # let x = _iter[_i];
                    LetStmt(
                        node.var_name,
                        initializer=IndexExpr(
                            Identifier(iter_name),
                            Identifier(idx_name),
                        ),
                    ),
                    # original body
                    body,
                    # _i = _i + 1;
                    AssignStmt(
                        Identifier(idx_name),
                        BinaryExpr(
                            BinOpKind.ADD,
                            Identifier(idx_name),
                            IntLiteral(1),
                        ),
                    ),
                ]),
            ),
        ])

    # ... other visit methods pass through unchanged
```

---

## 7. Source Location Tracking

### 7.1 Why Track Locations?

Source locations are essential for:

1. **Error messages**: "Type error at line 42, column 15"
2. **Debugger support**: Mapping compiled code back to source lines
3. **IDE features**: Go-to-definition, hover information, refactoring
4. **Source maps**: Mapping generated JavaScript back to TypeScript

### 7.2 Propagating Locations

```python
def parse_binary_expr(self) -> Expr:
    """Parse a binary expression, tracking source locations."""
    start = self.current_token.loc  # Remember start position

    left = self.parse_unary_expr()

    while self.current_token.type in (
        TokenType.PLUS, TokenType.MINUS,
        TokenType.STAR, TokenType.SLASH,
    ):
        op_token = self.current_token
        self.advance()
        right = self.parse_unary_expr()

        # Create node with location spanning left to right
        left = BinaryExpr(
            op=token_to_binop(op_token),
            left=left,
            right=right,
            loc=SourceLocation(
                line=start.line,
                column=start.column,
                end_line=right.loc.end_line if right.loc else -1,
                end_column=right.loc.end_column if right.loc else -1,
                filename=start.filename,
            ),
        )

    return left
```

### 7.3 Error Reporting with Locations

```python
class CompilerError:
    """A compiler error with source location and context."""

    def __init__(
        self,
        message: str,
        loc: SourceLocation,
        source_lines: list[str],
        severity: str = "error",
    ):
        self.message = message
        self.loc = loc
        self.source_lines = source_lines
        self.severity = severity

    def format(self) -> str:
        """Format the error like Rust/GCC-style diagnostics."""
        lines = []
        lines.append(
            f"{self.severity}: {self.message}"
        )
        lines.append(
            f"  --> {self.loc.filename}:{self.loc.line}:{self.loc.column}"
        )

        if 0 < self.loc.line <= len(self.source_lines):
            line_num = self.loc.line
            source_line = self.source_lines[line_num - 1]
            lines.append(f"   |")
            lines.append(f"{line_num:>3} | {source_line}")

            # Underline the error location
            padding = " " * (self.loc.column - 1)
            length = max(
                1,
                (self.loc.end_column - self.loc.column)
                if self.loc.end_column > 0 else 1,
            )
            underline = "^" * length
            lines.append(f"   | {padding}{underline}")

        return "\n".join(lines)


# Example output:
#
# error: Type mismatch: expected int, found string
#   --> main.lang:42:15
#    |
#  42 |     let x: int = "hello";
#    |                ^^^^^^^
```

---

## 8. Pattern Matching on ASTs

### 8.1 Python 3.10+ Structural Pattern Matching

Python 3.10 introduced `match`/`case` statements that work beautifully with dataclass-based ASTs:

```python
def simplify(expr: Expr) -> Expr:
    """
    Simplify expressions using pattern matching.

    Python 3.10+ structural pattern matching provides a clean
    way to match and transform AST nodes.
    """
    match expr:
        # Constant folding: literal + literal
        case BinaryExpr(
            op=BinOpKind.ADD,
            left=IntLiteral(value=a),
            right=IntLiteral(value=b),
        ):
            return IntLiteral(a + b)

        case BinaryExpr(
            op=BinOpKind.MUL,
            left=IntLiteral(value=a),
            right=IntLiteral(value=b),
        ):
            return IntLiteral(a * b)

        # Identity: x + 0 => x
        case BinaryExpr(
            op=BinOpKind.ADD,
            left=e,
            right=IntLiteral(value=0),
        ):
            return simplify(e)

        # Identity: 0 + x => x
        case BinaryExpr(
            op=BinOpKind.ADD,
            left=IntLiteral(value=0),
            right=e,
        ):
            return simplify(e)

        # Identity: x * 1 => x
        case BinaryExpr(
            op=BinOpKind.MUL,
            left=e,
            right=IntLiteral(value=1),
        ):
            return simplify(e)

        # Absorbing: x * 0 => 0
        case BinaryExpr(
            op=BinOpKind.MUL,
            right=IntLiteral(value=0),
        ):
            return IntLiteral(0)

        # Double negation: --x => x
        case UnaryExpr(
            op=UnaryOpKind.NEG,
            operand=UnaryExpr(op=UnaryOpKind.NEG, operand=inner),
        ):
            return simplify(inner)

        # Not not x => x
        case UnaryExpr(
            op=UnaryOpKind.NOT,
            operand=UnaryExpr(op=UnaryOpKind.NOT, operand=inner),
        ):
            return simplify(inner)

        # If true => then branch
        case IfExpr(
            condition=BoolLiteral(value=True),
            then_expr=then_e,
        ):
            return simplify(then_e)

        # If false => else branch
        case IfExpr(
            condition=BoolLiteral(value=False),
            else_expr=else_e,
        ):
            return simplify(else_e)

        # Default: recurse into children
        case BinaryExpr(op=op, left=left, right=right):
            return BinaryExpr(op, simplify(left), simplify(right))

        case UnaryExpr(op=op, operand=operand):
            return UnaryExpr(op, simplify(operand))

        case _:
            return expr
```

### 8.2 Pre-3.10 Alternative: isinstance Chains

For Python versions before 3.10, we use the traditional approach:

```python
def simplify_compat(expr: Expr) -> Expr:
    """Pattern matching using isinstance (Python < 3.10)."""
    if isinstance(expr, BinaryExpr):
        left = simplify_compat(expr.left)
        right = simplify_compat(expr.right)

        # Constant folding
        if (
            isinstance(left, IntLiteral)
            and isinstance(right, IntLiteral)
        ):
            if expr.op == BinOpKind.ADD:
                return IntLiteral(left.value + right.value)
            elif expr.op == BinOpKind.MUL:
                return IntLiteral(left.value * right.value)

        # Identity: x + 0 => x
        if (
            expr.op == BinOpKind.ADD
            and isinstance(right, IntLiteral)
            and right.value == 0
        ):
            return left

        return BinaryExpr(expr.op, left, right)

    elif isinstance(expr, UnaryExpr):
        operand = simplify_compat(expr.operand)

        if (
            expr.op == UnaryOpKind.NEG
            and isinstance(operand, UnaryExpr)
            and operand.op == UnaryOpKind.NEG
        ):
            return operand.operand

        return UnaryExpr(expr.op, operand)

    return expr
```

---

## 9. AST Serialization

### 9.1 JSON Serialization

JSON is the most common serialization format for ASTs, used by tools like Babel (JavaScript), rustc, and many LSP implementations.

```python
"""
AST Serialization to JSON and S-expressions.
"""

import json
import dataclasses


class ASTSerializer:
    """Serialize AST nodes to JSON-compatible dictionaries."""

    def to_dict(self, node: ASTNode) -> dict:
        """Convert an AST node to a JSON-serializable dictionary."""
        result = {"_type": type(node).__name__}

        if not dataclasses.is_dataclass(node):
            return result

        for f in dataclasses.fields(node):
            value = getattr(node, f.name)
            result[f.name] = self._serialize_value(value)

        return result

    def _serialize_value(self, value) -> object:
        """Serialize a field value."""
        if value is None:
            return None
        elif isinstance(value, ASTNode):
            return self.to_dict(value)
        elif isinstance(value, list):
            return [self._serialize_value(item) for item in value]
        elif isinstance(value, Enum):
            return value.value
        elif isinstance(value, (int, float, str, bool)):
            return value
        elif isinstance(value, SourceLocation):
            return {
                "line": value.line,
                "column": value.column,
                "end_line": value.end_line,
                "end_column": value.end_column,
            }
        else:
            return str(value)

    def to_json(self, node: ASTNode, indent: int = 2) -> str:
        """Convert an AST node to a JSON string."""
        return json.dumps(self.to_dict(node), indent=indent)

    @staticmethod
    def from_dict(data: dict) -> ASTNode:
        """Reconstruct an AST node from a dictionary."""
        # This requires a registry of node types
        node_types = {
            "IntLiteral": IntLiteral,
            "FloatLiteral": FloatLiteral,
            "BoolLiteral": BoolLiteral,
            "StringLiteral": StringLiteral,
            "Identifier": Identifier,
            "BinaryExpr": BinaryExpr,
            "UnaryExpr": UnaryExpr,
            "CallExpr": CallExpr,
            "LetStmt": LetStmt,
            "PrintStmt": PrintStmt,
            "Program": Program,
            # ... register all node types
        }

        type_name = data.get("_type")
        if type_name not in node_types:
            raise ValueError(f"Unknown AST node type: {type_name}")

        cls = node_types[type_name]
        # Reconstruct fields (simplified -- real implementation
        # would need recursive deserialization)
        kwargs = {}
        for f in dataclasses.fields(cls):
            if f.name in data and f.name != "_type":
                kwargs[f.name] = data[f.name]

        return cls(**kwargs)


# ─── Demo ───

if __name__ == "__main__":
    ast = BinaryExpr(
        BinOpKind.ADD,
        IntLiteral(2),
        BinaryExpr(BinOpKind.MUL, IntLiteral(3), IntLiteral(4)),
    )

    serializer = ASTSerializer()
    json_str = serializer.to_json(ast)
    print("JSON representation:")
    print(json_str)
```

**Output:**

```json
{
  "_type": "BinaryExpr",
  "op": "+",
  "left": {
    "_type": "IntLiteral",
    "value": 2,
    "loc": null
  },
  "right": {
    "_type": "BinaryExpr",
    "op": "*",
    "left": {
      "_type": "IntLiteral",
      "value": 3,
      "loc": null
    },
    "right": {
      "_type": "IntLiteral",
      "value": 4,
      "loc": null
    },
    "loc": null
  },
  "loc": null
}
```

### 9.2 S-Expression Serialization

S-expressions provide a compact, Lisp-like representation that is easy to read and parse:

```python
class SExprSerializer:
    """Serialize AST to S-expression format."""

    def to_sexpr(self, node: ASTNode) -> str:
        """Convert AST to S-expression string."""
        if isinstance(node, IntLiteral):
            return str(node.value)
        elif isinstance(node, FloatLiteral):
            return str(node.value)
        elif isinstance(node, BoolLiteral):
            return "#t" if node.value else "#f"
        elif isinstance(node, StringLiteral):
            return f'"{node.value}"'
        elif isinstance(node, NilLiteral):
            return "nil"
        elif isinstance(node, Identifier):
            return node.name
        elif isinstance(node, BinaryExpr):
            left = self.to_sexpr(node.left)
            right = self.to_sexpr(node.right)
            return f"({node.op.value} {left} {right})"
        elif isinstance(node, UnaryExpr):
            operand = self.to_sexpr(node.operand)
            return f"({node.op.value} {operand})"
        elif isinstance(node, CallExpr):
            callee = self.to_sexpr(node.callee)
            args = " ".join(self.to_sexpr(a) for a in node.arguments)
            return f"(call {callee} {args})"
        elif isinstance(node, LetStmt):
            init = self.to_sexpr(node.initializer) if node.initializer else "nil"
            return f"(let {node.name} {init})"
        elif isinstance(node, IfStmt):
            cond = self.to_sexpr(node.condition)
            then = self.to_sexpr(node.then_body)
            if node.else_body:
                els = self.to_sexpr(node.else_body)
                return f"(if {cond} {then} {els})"
            return f"(if {cond} {then})"
        elif isinstance(node, Block):
            stmts = " ".join(self.to_sexpr(s) for s in node.statements)
            return f"(block {stmts})"
        elif isinstance(node, PrintStmt):
            return f"(print {self.to_sexpr(node.value)})"
        elif isinstance(node, Program):
            stmts = " ".join(self.to_sexpr(s) for s in node.statements)
            return f"(program {stmts})"
        elif isinstance(node, FuncDecl):
            params = " ".join(p.name for p in node.params)
            body = self.to_sexpr(node.body) if node.body else "()"
            return f"(fn {node.name} ({params}) {body})"
        elif isinstance(node, ReturnStmt):
            if node.value:
                return f"(return {self.to_sexpr(node.value)})"
            return "(return)"
        elif isinstance(node, WhileStmt):
            return (
                f"(while {self.to_sexpr(node.condition)} "
                f"{self.to_sexpr(node.body)})"
            )
        elif isinstance(node, AssignStmt):
            return (
                f"(set! {self.to_sexpr(node.target)} "
                f"{self.to_sexpr(node.value)})"
            )
        elif isinstance(node, ExprStmt):
            return self.to_sexpr(node.expr)
        else:
            return f"(<unknown> {type(node).__name__})"


# ─── Demo ───

if __name__ == "__main__":
    ast = BinaryExpr(
        BinOpKind.ADD,
        IntLiteral(2),
        BinaryExpr(BinOpKind.MUL, IntLiteral(3), Identifier("x")),
    )

    sexpr = SExprSerializer()
    print(sexpr.to_sexpr(ast))
    # Output: (+ 2 (* 3 x))

    # A more complex program
    program = Program(statements=[
        LetStmt("x", initializer=IntLiteral(10)),
        WhileStmt(
            condition=BinaryExpr(
                BinOpKind.GT, Identifier("x"), IntLiteral(0)
            ),
            body=Block([
                PrintStmt(Identifier("x")),
                AssignStmt(
                    Identifier("x"),
                    BinaryExpr(
                        BinOpKind.SUB, Identifier("x"), IntLiteral(1)
                    ),
                ),
            ]),
        ),
    ])

    print(sexpr.to_sexpr(program))
    # Output: (program (let x 10) (while (> x 0) (block (print x) (set! x (- x 1)))))
```

---

## 10. Real-World AST Examples

### 10.1 Python's ast Module

Python provides a built-in `ast` module that exposes the AST used by CPython:

```python
import ast

source = """
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)
"""

tree = ast.parse(source)
print(ast.dump(tree, indent=2))
```

**Output (abbreviated):**

```
Module(
  body=[
    FunctionDef(
      name='factorial',
      args=arguments(args=[arg(arg='n')]),
      body=[
        If(
          test=Compare(
            left=Name(id='n'),
            ops=[LtE()],
            comparators=[Constant(value=1)]
          ),
          body=[Return(value=Constant(value=1))],
          orelse=[]
        ),
        Return(
          value=BinOp(
            left=Name(id='n'),
            op=Mult(),
            right=Call(
              func=Name(id='factorial'),
              args=[BinOp(left=Name(id='n'), op=Sub(), right=Constant(value=1))]
            )
          )
        )
      ]
    )
  ]
)
```

### 10.2 Rust's syn Crate

Rust's `syn` crate parses Rust source code into AST types. The AST closely mirrors Rust's syntax with types like `ExprBinary`, `ExprIf`, `ItemFn`, etc.

### 10.3 Babel (JavaScript)

Babel uses the **ESTree** AST specification, which defines standard node types for JavaScript. ASTs are represented as JSON objects with a `type` field.

---

## 11. Summary

The abstract syntax tree is the central data structure in any compiler or language tool. It provides a clean, structural representation of the source program that is amenable to analysis and transformation.

**Key concepts:**

| Concept | Description |
|---------|-------------|
| **CST vs AST** | CST mirrors grammar; AST captures semantic structure |
| **Node design** | Use dataclasses/algebraic types with clear type hierarchies |
| **Visitor pattern** | Separate traversal logic from node definitions |
| **Pre/post/in-order** | Different traversal orders for different purposes |
| **Pretty printing** | Convert AST back to readable source code |
| **Constant folding** | Evaluate compile-time known expressions |
| **Desugaring** | Simplify syntactic sugar into core constructs |
| **Source locations** | Track where each construct appears for error reporting |
| **Serialization** | JSON and S-expressions for tool interoperability |

**Design guidelines:**

1. Keep AST nodes minimal -- desugar during or after parsing
2. Use algebraic data types (or dataclasses + enums) for type safety
3. Always include source locations for good error messages
4. Make nodes immutable; transformations create new trees
5. The visitor pattern scales well to many analysis passes
6. Consider serialization early if interoperability with tools is needed

---

## Exercises

### Exercise 1: AST Node Design

Design AST node types (using Python dataclasses) for the following language features:

1. Array literals: `[1, 2, 3]`
2. Dictionary literals: `{key: value, ...}`
3. Slice expressions: `a[1:5]`, `a[:3]`, `a[::2]`
4. Try-catch: `try { ... } catch (e: Error) { ... } finally { ... }`
5. Pattern matching: `match x { 1 => "one", 2 => "two", _ => "other" }`

### Exercise 2: Complete Visitor

Implement a `DepthCalculator` visitor that computes the maximum depth of an expression AST. For example, `IntLiteral(5)` has depth 1, and `BinaryExpr(ADD, IntLiteral(2), IntLiteral(3))` has depth 2.

### Exercise 3: Pretty Printer Enhancement

Extend the pretty printer to handle:

1. Multi-line function calls when arguments exceed a line width (e.g., 80 characters)
2. Trailing commas in list literals
3. Comments (you will need to add a comment field to appropriate AST nodes)

### Exercise 4: Constant Propagation

Implement a `ConstantPropagator` that tracks variable assignments and substitutes known values:

```
let x = 5;           let x = 5;
let y = x + 3;  =>   let y = 8;
print(y * 2);         print(16);
```

This requires combining the constant folder with a simple environment that tracks variable-to-value mappings.

### Exercise 5: AST Diff

Implement a function `ast_diff(old: ASTNode, new: ASTNode) -> list[Change]` that computes the differences between two ASTs. Each `Change` should record what was added, removed, or modified and where (source location).

This is useful for incremental compilation and intelligent code review tools.

### Exercise 6: Round-Trip Test

Write a test that verifies the **round-trip property**: parse source code to an AST, pretty-print the AST back to source code, parse again, and verify that both ASTs are structurally identical.

```python
def test_round_trip(source: str):
    ast1 = parse(source)
    regenerated = pretty_print(ast1)
    ast2 = parse(regenerated)
    assert ast1 == ast2, "Round-trip failed!"
```

Identify cases where round-tripping may fail (e.g., comments, whitespace, parenthesization differences) and discuss how to handle them.

---

[Previous: Bottom-Up Parsing](./06_Bottom_Up_Parsing.md) | [Next: Semantic Analysis](./08_Semantic_Analysis.md) | [Overview](./00_Overview.md)
