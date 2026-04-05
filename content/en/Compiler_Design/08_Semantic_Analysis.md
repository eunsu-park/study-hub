# Lesson 8: Semantic Analysis

## Learning Objectives

After completing this lesson, you will be able to:

1. **Explain** the role of semantic analysis in the compilation pipeline
2. **Distinguish** between syntax-directed translation using S-attributed and L-attributed grammars
3. **Implement** symbol tables with proper scope management using hash tables
4. **Describe** type systems along the dimensions of static/dynamic and strong/weak
5. **Implement** type checking rules for expressions, statements, and functions
6. **Understand** the basics of Hindley-Milner type inference
7. **Handle** type compatibility, coercion, overloading resolution, and declaration processing
8. **Report** semantic errors with clear messages and source locations

---

## 1. Introduction: The Role of Semantic Analysis

After parsing produces an AST, the compiler must verify that the program is **meaningful** -- not just syntactically correct, but semantically valid. Semantic analysis bridges the gap between syntax and code generation.

### 1.1 What Parsing Cannot Check

Parsing ensures the program follows the grammar, but many important properties are beyond the grammar's reach:

| Property | Example of Violation | Detected By |
|----------|---------------------|-------------|
| Variable declared before use | `x = y + 1;` (y not declared) | Semantic analysis |
| Type compatibility | `"hello" + 42` (string + int) | Type checker |
| Function arity | `f(1, 2)` when f takes 3 args | Type checker |
| Return type | `fn foo() -> int { return "hi"; }` | Type checker |
| Break outside loop | `break;` in global scope | Semantic analysis |
| Duplicate declarations | `let x = 1; let x = 2;` in same scope | Symbol table |
| Access control | Reading a private field | Semantic analysis |

### 1.2 Semantic Analysis Pipeline

```
            AST (from parser)
             │
             ▼
    ┌────────────────────┐
    │  Name Resolution   │  Build symbol tables,
    │  (Scope Analysis)  │  resolve identifiers
    └────────┬───────────┘
             │
             ▼
    ┌────────────────────┐
    │  Type Checking /   │  Verify type correctness,
    │  Type Inference    │  annotate AST with types
    └────────┬───────────┘
             │
             ▼
    ┌────────────────────┐
    │  Other Checks      │  Control flow, reachability,
    │                    │  initialization, etc.
    └────────┬───────────┘
             │
             ▼
        Annotated AST
    (ready for IR generation)
```

---

## 2. Attribute Grammars

### 2.1 Concept

An **attribute grammar** augments a context-free grammar with **attributes** attached to grammar symbols and **semantic rules** that define how to compute them. This formalism provides a rigorous way to define semantic analysis.

Each grammar symbol $X$ can have:
- **Synthesized attributes**: Computed from the attributes of $X$'s children (information flows up)
- **Inherited attributes**: Computed from the attributes of $X$'s parent or siblings (information flows down)

### 2.2 S-Attributed Grammars

An **S-attributed grammar** uses only **synthesized** attributes. Information flows strictly bottom-up. These are the simplest and can be evaluated in a single post-order traversal.

**Example: Calculator with synthesized `val` attribute**

$$
\begin{aligned}
E &\to E_1 + T & \quad E.\text{val} &= E_1.\text{val} + T.\text{val} \\
E &\to T & \quad E.\text{val} &= T.\text{val} \\
T &\to T_1 * F & \quad T.\text{val} &= T_1.\text{val} \times F.\text{val} \\
T &\to F & \quad T.\text{val} &= F.\text{val} \\
F &\to (\ E\ ) & \quad F.\text{val} &= E.\text{val} \\
F &\to \textbf{num} & \quad F.\text{val} &= \textbf{num}.\text{lexval}
\end{aligned}
$$

```python
"""
S-Attributed Grammar Evaluation

Demonstrates synthesized attribute computation
for a simple expression evaluator.
"""


def evaluate_s_attributed(node) -> int:
    """
    Post-order evaluation (S-attributed).
    All attributes are synthesized (flow bottom-up).
    """
    match node:
        case IntLiteral(value=v):
            return v

        case BinaryExpr(op=BinOpKind.ADD, left=l, right=r):
            return evaluate_s_attributed(l) + evaluate_s_attributed(r)

        case BinaryExpr(op=BinOpKind.MUL, left=l, right=r):
            return evaluate_s_attributed(l) * evaluate_s_attributed(r)

        case _:
            raise ValueError(f"Unsupported node: {type(node).__name__}")
```

### 2.3 L-Attributed Grammars

An **L-attributed grammar** allows inherited attributes, but with a restriction: for a production $A \to X_1 X_2 \cdots X_n$, the inherited attributes of $X_i$ can depend only on:

1. Inherited attributes of $A$ (the parent)
2. Attributes (synthesized or inherited) of $X_1, X_2, \ldots, X_{i-1}$ (the left siblings)

This ensures attributes can be computed in a single **left-to-right** traversal.

**Example: Type declaration with inherited type attribute**

$$
\begin{aligned}
D &\to T\ L & \quad L.\text{type} &= T.\text{type} \\
T &\to \textbf{int} & \quad T.\text{type} &= \text{integer} \\
T &\to \textbf{float} & \quad T.\text{type} &= \text{float} \\
L &\to L_1 ,\ \textbf{id} & \quad L_1.\text{type} &= L.\text{type}; \quad \text{addtype}(\textbf{id}, L.\text{type}) \\
L &\to \textbf{id} & \quad & \text{addtype}(\textbf{id}, L.\text{type})
\end{aligned}
$$

Here, `L.type` is an **inherited** attribute that flows from left to right, carrying the declared type to each identifier.

```python
def process_declaration(type_node, id_list, symbol_table):
    """
    L-attributed evaluation: the type flows from left (T) to right (L).

    For "int x, y, z":
      T.type = int         (synthesized)
      L.type = T.type      (inherited from left sibling)
      addtype(x, int)      (uses inherited type)
      addtype(y, int)
      addtype(z, int)
    """
    declared_type = resolve_type(type_node)  # synthesized from T
    for name in id_list:
        symbol_table.define(name, declared_type)  # inherited to L
```

### 2.4 Comparison

| Property | S-Attributed | L-Attributed |
|----------|-------------|--------------|
| Attribute direction | Bottom-up only | Bottom-up + restricted top-down |
| Evaluation order | Post-order | Left-to-right depth-first |
| Complexity | Simpler | More flexible |
| LR parsers | Directly supported | Requires adaptation |
| LL parsers | Supported | Directly supported |
| Use cases | Expression evaluation, code gen | Type propagation, scope management |

---

## 3. Symbol Tables

### 3.1 Purpose

A **symbol table** is the data structure that maps names (identifiers) to their attributes (type, scope, memory location, etc.). It is the backbone of semantic analysis.

Every time the compiler encounters:
- A **declaration** (`let x: int = 5`): add a new entry
- A **reference** (`print(x)`): look up an existing entry

### 3.2 Scope Management

Most languages support **nested scopes**: a block can contain inner blocks, each with their own declarations that may shadow outer ones.

```
fn outer() {
    let x = 1;           // scope level 1
    {
        let y = 2;       // scope level 2
        let x = 10;      // shadows outer x
        {
            let z = 3;   // scope level 3
            print(x);    // resolves to inner x (= 10)
        }
        // z is no longer visible
    }
    // y and inner x are no longer visible
    print(x);            // resolves to outer x (= 1)
}
```

### 3.3 Implementation Strategies

**Strategy 1: Chained Hash Tables (scope stack)**

Each scope gets its own hash table. Lookup searches from the innermost scope outward.

```
Scope Stack:

    ┌──────────────┐
    │ Scope 3      │ ← current scope
    │ z -> int     │
    └──────┬───────┘
           │ parent
    ┌──────┴───────┐
    │ Scope 2      │
    │ y -> int     │
    │ x -> int     │  (shadows Scope 1's x)
    └──────┬───────┘
           │ parent
    ┌──────┴───────┐
    │ Scope 1      │
    │ x -> int     │
    └──────────────┘
```

**Strategy 2: Single hash table with scope markers**

Use one hash table but maintain a stack of scope markers. When exiting a scope, remove all entries added since the last marker.

### 3.4 Complete Implementation

```python
"""
Symbol Table with Scope Management

Implements a chained-scope symbol table using hash tables.
Each scope has its own dictionary, and lookup walks up the chain.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Any


# ─── Types ───

class TypeKind(Enum):
    INT = auto()
    FLOAT = auto()
    BOOL = auto()
    STRING = auto()
    VOID = auto()
    FUNC = auto()
    LIST = auto()
    ERROR = auto()      # special type for error recovery


@dataclass(frozen=True)
class Type:
    """Represents a type in the type system."""
    kind: TypeKind

    def __repr__(self):
        return self.kind.name.lower()


@dataclass(frozen=True)
class FuncType(Type):
    """Function type: (param_types) -> return_type."""
    kind: TypeKind = field(default=TypeKind.FUNC, init=False)
    param_types: tuple[Type, ...] = ()
    return_type: Type = field(default_factory=lambda: Type(TypeKind.VOID))

    def __repr__(self):
        params = ", ".join(str(p) for p in self.param_types)
        return f"({params}) -> {self.return_type}"


@dataclass(frozen=True)
class ListType(Type):
    """List type: list[element_type]."""
    kind: TypeKind = field(default=TypeKind.LIST, init=False)
    element_type: Type = field(
        default_factory=lambda: Type(TypeKind.INT)
    )

    def __repr__(self):
        return f"list[{self.element_type}]"


# Commonly used types
INT_TYPE = Type(TypeKind.INT)
FLOAT_TYPE = Type(TypeKind.FLOAT)
BOOL_TYPE = Type(TypeKind.BOOL)
STRING_TYPE = Type(TypeKind.STRING)
VOID_TYPE = Type(TypeKind.VOID)
ERROR_TYPE = Type(TypeKind.ERROR)


# ─── Symbol ───

class SymbolKind(Enum):
    VARIABLE = auto()
    FUNCTION = auto()
    PARAMETER = auto()
    TYPE_ALIAS = auto()


@dataclass
class Symbol:
    """
    Represents a named entity in the program.

    Attributes:
        name: the identifier
        type: the entity's type
        kind: variable, function, parameter, etc.
        scope_level: nesting depth where defined
        is_mutable: whether the binding can be reassigned
        is_initialized: whether a value has been assigned
        loc: source location of the declaration
    """
    name: str
    type: Type
    kind: SymbolKind
    scope_level: int = 0
    is_mutable: bool = True
    is_initialized: bool = False
    loc: Optional[Any] = None  # SourceLocation


# ─── Scope ───

@dataclass
class Scope:
    """
    A single scope level containing symbol definitions.

    Attributes:
        symbols: hash table mapping names to symbols
        parent: enclosing scope (None for global scope)
        level: nesting depth (0 = global)
        name: optional scope name (function name, block, etc.)
    """
    symbols: dict[str, Symbol] = field(default_factory=dict)
    parent: Optional[Scope] = None
    level: int = 0
    name: str = "<anonymous>"

    def define(self, symbol: Symbol) -> Optional[Symbol]:
        """
        Define a new symbol in this scope.

        Returns the previous symbol if there was a duplicate
        in THIS scope (not parent scopes), or None.
        """
        existing = self.symbols.get(symbol.name)
        self.symbols[symbol.name] = symbol
        symbol.scope_level = self.level
        return existing

    def lookup_local(self, name: str) -> Optional[Symbol]:
        """Look up a symbol in THIS scope only (no parent chain)."""
        return self.symbols.get(name)

    def lookup(self, name: str) -> Optional[Symbol]:
        """
        Look up a symbol, searching up the scope chain.

        Returns the first matching symbol found (innermost scope wins).
        """
        scope = self
        while scope is not None:
            sym = scope.symbols.get(name)
            if sym is not None:
                return sym
            scope = scope.parent
        return None

    def all_symbols(self) -> list[Symbol]:
        """Return all symbols defined in this scope."""
        return list(self.symbols.values())


# ─── Symbol Table ───

class SymbolTable:
    """
    Manages a stack of scopes for name resolution.

    Usage:
        table = SymbolTable()
        table.define("x", INT_TYPE, SymbolKind.VARIABLE)

        table.enter_scope("if-body")
        table.define("y", BOOL_TYPE, SymbolKind.VARIABLE)
        table.define("x", FLOAT_TYPE, SymbolKind.VARIABLE)  # shadows outer x

        sym = table.lookup("x")  # finds inner x (float)
        table.exit_scope()

        sym = table.lookup("x")  # finds outer x (int)
        sym = table.lookup("y")  # None (out of scope)
    """

    def __init__(self):
        self.global_scope = Scope(level=0, name="<global>")
        self.current_scope = self.global_scope
        self.errors: list[str] = []

    @property
    def scope_level(self) -> int:
        return self.current_scope.level

    def enter_scope(self, name: str = "<block>"):
        """Enter a new nested scope."""
        new_scope = Scope(
            parent=self.current_scope,
            level=self.current_scope.level + 1,
            name=name,
        )
        self.current_scope = new_scope

    def exit_scope(self) -> Scope:
        """
        Exit the current scope and return to the enclosing scope.

        Returns the exited scope (useful for inspecting its symbols).
        """
        if self.current_scope.parent is None:
            raise RuntimeError("Cannot exit global scope")
        exited = self.current_scope
        self.current_scope = self.current_scope.parent
        return exited

    def define(
        self,
        name: str,
        type: Type,
        kind: SymbolKind = SymbolKind.VARIABLE,
        mutable: bool = True,
        initialized: bool = False,
        loc: Optional[Any] = None,
    ) -> Symbol:
        """
        Define a new symbol in the current scope.

        Reports an error if the name is already defined in the
        current scope (not parent scopes -- shadowing is allowed).
        """
        symbol = Symbol(
            name=name,
            type=type,
            kind=kind,
            scope_level=self.scope_level,
            is_mutable=mutable,
            is_initialized=initialized,
            loc=loc,
        )

        existing = self.current_scope.define(symbol)
        if existing is not None:
            self.errors.append(
                f"Duplicate definition: '{name}' is already defined "
                f"in scope '{self.current_scope.name}' "
                f"(previous definition at {existing.loc})"
            )
        return symbol

    def lookup(self, name: str) -> Optional[Symbol]:
        """
        Look up a symbol by name, searching from current to global scope.
        """
        return self.current_scope.lookup(name)

    def lookup_local(self, name: str) -> Optional[Symbol]:
        """Look up a symbol in the current scope only."""
        return self.current_scope.lookup_local(name)

    def resolve(self, name: str, loc: Optional[Any] = None) -> Symbol:
        """
        Resolve a name reference, reporting an error if not found.

        Returns the symbol if found, or a dummy ERROR symbol.
        """
        sym = self.lookup(name)
        if sym is None:
            self.errors.append(
                f"Undefined name: '{name}' is not defined "
                f"(referenced at {loc})"
            )
            return Symbol(
                name=name,
                type=ERROR_TYPE,
                kind=SymbolKind.VARIABLE,
            )
        return sym

    def print_scopes(self):
        """Debug: print all scopes and their symbols."""
        scope = self.current_scope
        while scope is not None:
            print(
                f"Scope '{scope.name}' (level {scope.level}):"
            )
            for name, sym in scope.symbols.items():
                mut = "mut" if sym.is_mutable else "const"
                init = "init" if sym.is_initialized else "uninit"
                print(
                    f"  {name}: {sym.type} "
                    f"({sym.kind.name}, {mut}, {init})"
                )
            scope = scope.parent


# ─── Demo ───

if __name__ == "__main__":
    table = SymbolTable()

    # Global scope
    table.define("print", FuncType(
        param_types=(STRING_TYPE,), return_type=VOID_TYPE
    ), SymbolKind.FUNCTION, mutable=False)

    table.define("x", INT_TYPE, initialized=True)

    # Enter function scope
    table.enter_scope("main")
    table.define("y", FLOAT_TYPE, SymbolKind.PARAMETER, initialized=True)
    table.define("x", BOOL_TYPE, initialized=True)  # shadows global x

    # Enter block scope
    table.enter_scope("if-body")
    table.define("z", STRING_TYPE, initialized=True)

    print("Current scope state:")
    table.print_scopes()
    print()

    # Lookups
    for name in ["z", "y", "x", "print", "w"]:
        sym = table.resolve(name)
        if sym.type != ERROR_TYPE:
            print(f"  {name} -> {sym.type} (level {sym.scope_level})")
        else:
            print(f"  {name} -> UNDEFINED")

    print()
    if table.errors:
        print("Errors:")
        for err in table.errors:
            print(f"  {err}")

    # Exit scopes
    table.exit_scope()
    table.exit_scope()

    # Now x refers to global int again
    sym = table.lookup("x")
    print(f"\nAfter exiting scopes: x -> {sym.type}")
```

**Expected output:**

```
Current scope state:
Scope 'if-body' (level 2):
  z: string (VARIABLE, mut, init)
Scope 'main' (level 1):
  y: float (PARAMETER, mut, init)
  x: bool (VARIABLE, mut, init)
Scope '<global>' (level 0):
  print: () -> void (FUNCTION, const, uninit)
  x: int (VARIABLE, mut, init)

  z -> string (level 2)
  y -> float (level 1)
  x -> bool (level 1)
  print -> () -> void (level 0)
  w -> UNDEFINED

Errors:
  Undefined name: 'w' is not defined (referenced at None)

After exiting scopes: x -> int
```

---

## 4. Type Systems

### 4.1 Static vs Dynamic Typing

**Static typing** checks types at **compile time**. Every expression has a known type before the program runs.

| Language | Typing |
|----------|--------|
| C, C++, Java, Rust, Go, Haskell | Static |
| Python, JavaScript, Ruby, Lisp | Dynamic |
| TypeScript, Mypy, Dart | Gradually typed (static + dynamic) |

**Dynamic typing** checks types at **runtime**. Variables can hold values of any type.

```python
# Static typing (pseudo-code)
let x: int = 5
x = "hello"         # COMPILE ERROR: type mismatch

# Dynamic typing (Python)
x = 5
x = "hello"         # OK at runtime
```

### 4.2 Strong vs Weak Typing

**Strong typing** means implicit type conversions are restricted. Operations on incompatible types produce errors.

**Weak typing** means implicit conversions (coercions) happen freely.

| | Strong | Weak |
|---|---|---|
| **Static** | Rust, Haskell, Java | C, C++ |
| **Dynamic** | Python, Ruby | JavaScript, PHP |

**Examples of weak typing (JavaScript):**

```javascript
"5" + 3       // "53" (number coerced to string)
"5" - 3       // 2    (string coerced to number)
true + true   // 2    (booleans coerced to numbers)
```

**Examples of strong typing (Python):**

```python
"5" + 3       # TypeError: can only concatenate str to str
True + True   # 2 (bool is a subclass of int -- deliberate design)
```

### 4.3 Type System Design Spectrum

```
    ┌──────────────────────────────────────────────────────┐
    │           Type System Design Space                   │
    │                                                      │
    │  Weak ◄────────────────────────────────────► Strong  │
    │                                                      │
    │   C         Java       Haskell     Rust              │
    │   │          │           │          │                 │
    │   ▼          ▼           ▼          ▼                 │
    │  implicit   some        no         no implicit       │
    │  casts      coercion    coercion   conversion        │
    │                                    + ownership       │
    │                                                      │
    │  JavaScript  Python     OCaml                        │
    │   │          │           │                            │
    │   ▼          ▼           ▼                            │
    │  lots of    few         no                           │
    │  coercion   coercion    coercion                     │
    └──────────────────────────────────────────────────────┘
```

---

## 5. Type Checking

### 5.1 Type Checking Rules

Type checking verifies that operations are applied to operands of compatible types. We express type checking rules as **judgments**:

$$\Gamma \vdash e : \tau$$

Read as: "Under type environment $\Gamma$, expression $e$ has type $\tau$."

**Core typing rules (expressed semi-formally):**

$$
\frac{}{\Gamma \vdash n : \text{int}} \quad (\text{Int Literal})
$$

$$
\frac{}{\Gamma \vdash \text{true} : \text{bool}} \quad (\text{Bool Literal})
$$

$$
\frac{x : \tau \in \Gamma}{\Gamma \vdash x : \tau} \quad (\text{Variable})
$$

$$
\frac{\Gamma \vdash e_1 : \text{int} \quad \Gamma \vdash e_2 : \text{int}}{\Gamma \vdash e_1 + e_2 : \text{int}} \quad (\text{Int Addition})
$$

$$
\frac{\Gamma \vdash e_1 : \tau_1 \quad \Gamma \vdash e_2 : \tau_2 \quad \tau_1 = \tau_2 \quad \tau_1 \in \{\text{int}, \text{float}\}}{\Gamma \vdash e_1 < e_2 : \text{bool}} \quad (\text{Comparison})
$$

$$
\frac{\Gamma \vdash e : \text{bool} \quad \Gamma \vdash s_1 : \text{ok} \quad \Gamma \vdash s_2 : \text{ok}}{\Gamma \vdash \textbf{if}\ (e)\ s_1\ \textbf{else}\ s_2 : \text{ok}} \quad (\text{If Statement})
$$

$$
\frac{\Gamma \vdash f : (\tau_1, \ldots, \tau_n) \to \tau_r \quad \Gamma \vdash e_i : \tau_i \text{ for } i = 1 \ldots n}{\Gamma \vdash f(e_1, \ldots, e_n) : \tau_r} \quad (\text{Function Call})
$$

### 5.2 Type Checker Implementation

```python
"""
Type Checker Implementation

Walks the AST, verifies type correctness, and annotates
nodes with their resolved types.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


class TypeChecker:
    """
    Performs type checking on an AST.

    Uses the symbol table for name resolution and type lookup.
    Reports type errors with source locations.
    """

    def __init__(self):
        self.symbol_table = SymbolTable()
        self.errors: list[str] = []
        self.current_function_return_type: Optional[Type] = None

        # Register built-in functions
        self._register_builtins()

    def _register_builtins(self):
        """Register built-in functions in the global scope."""
        self.symbol_table.define(
            "print",
            FuncType(param_types=(STRING_TYPE,), return_type=VOID_TYPE),
            SymbolKind.FUNCTION,
            mutable=False,
            initialized=True,
        )
        self.symbol_table.define(
            "len",
            FuncType(param_types=(STRING_TYPE,), return_type=INT_TYPE),
            SymbolKind.FUNCTION,
            mutable=False,
            initialized=True,
        )
        self.symbol_table.define(
            "str",
            FuncType(param_types=(INT_TYPE,), return_type=STRING_TYPE),
            SymbolKind.FUNCTION,
            mutable=False,
            initialized=True,
        )

    def error(self, message: str, loc=None):
        """Record a type error."""
        loc_str = f" at {loc}" if loc else ""
        self.errors.append(f"TypeError{loc_str}: {message}")

    def check_program(self, program) -> bool:
        """
        Type-check an entire program.

        Returns True if no type errors were found.
        """
        for stmt in program.statements:
            self.check_stmt(stmt)
        return len(self.errors) == 0

    # ─── Expression Type Checking ───

    def check_expr(self, expr) -> Type:
        """
        Type-check an expression and return its type.

        This is the core of the type checker. Each expression
        form has specific typing rules.
        """
        if isinstance(expr, IntLiteral):
            return INT_TYPE

        elif isinstance(expr, FloatLiteral):
            return FLOAT_TYPE

        elif isinstance(expr, BoolLiteral):
            return BOOL_TYPE

        elif isinstance(expr, StringLiteral):
            return STRING_TYPE

        elif isinstance(expr, NilLiteral):
            return VOID_TYPE

        elif isinstance(expr, Identifier):
            return self._check_identifier(expr)

        elif isinstance(expr, BinaryExpr):
            return self._check_binary(expr)

        elif isinstance(expr, UnaryExpr):
            return self._check_unary(expr)

        elif isinstance(expr, CallExpr):
            return self._check_call(expr)

        elif isinstance(expr, IndexExpr):
            return self._check_index(expr)

        elif isinstance(expr, IfExpr):
            return self._check_if_expr(expr)

        elif isinstance(expr, ListExpr):
            return self._check_list_expr(expr)

        else:
            self.error(
                f"Unknown expression type: {type(expr).__name__}",
                getattr(expr, 'loc', None),
            )
            return ERROR_TYPE

    def _check_identifier(self, expr) -> Type:
        """Type-check a variable reference."""
        sym = self.symbol_table.resolve(
            expr.name, getattr(expr, 'loc', None)
        )
        # Propagate symbol table errors
        if self.symbol_table.errors:
            self.errors.extend(self.symbol_table.errors)
            self.symbol_table.errors.clear()

        if not sym.is_initialized and sym.type != ERROR_TYPE:
            self.error(
                f"Variable '{expr.name}' may not be initialized",
                getattr(expr, 'loc', None),
            )
        return sym.type

    def _check_binary(self, expr) -> Type:
        """
        Type-check a binary expression.

        Rules:
        - Arithmetic (+, -, *, /, %, **): both operands numeric, result is
          the wider type (int < float)
        - String concatenation (+): both operands string, result string
        - Comparison (<, >, <=, >=): both operands numeric, result bool
        - Equality (==, !=): operands same type, result bool
        - Logical (and, or): both operands bool, result bool
        """
        left_type = self.check_expr(expr.left)
        right_type = self.check_expr(expr.right)

        # Skip further checking if either side has errors
        if left_type == ERROR_TYPE or right_type == ERROR_TYPE:
            return ERROR_TYPE

        op = expr.op

        # Arithmetic operators
        if op in (
            BinOpKind.ADD, BinOpKind.SUB,
            BinOpKind.MUL, BinOpKind.DIV,
            BinOpKind.MOD, BinOpKind.POW,
        ):
            # String concatenation
            if (
                op == BinOpKind.ADD
                and left_type == STRING_TYPE
                and right_type == STRING_TYPE
            ):
                return STRING_TYPE

            # Numeric operations
            if self._is_numeric(left_type) and self._is_numeric(right_type):
                return self._wider_numeric(left_type, right_type)

            self.error(
                f"Operator '{op.value}' not supported for types "
                f"'{left_type}' and '{right_type}'",
                getattr(expr, 'loc', None),
            )
            return ERROR_TYPE

        # Comparison operators
        if op in (
            BinOpKind.LT, BinOpKind.GT,
            BinOpKind.LE, BinOpKind.GE,
        ):
            if self._is_numeric(left_type) and self._is_numeric(right_type):
                return BOOL_TYPE
            self.error(
                f"Cannot compare '{left_type}' and '{right_type}'",
                getattr(expr, 'loc', None),
            )
            return ERROR_TYPE

        # Equality operators
        if op in (BinOpKind.EQ, BinOpKind.NE):
            if self._types_compatible(left_type, right_type):
                return BOOL_TYPE
            self.error(
                f"Cannot compare '{left_type}' and '{right_type}' "
                f"for equality",
                getattr(expr, 'loc', None),
            )
            return ERROR_TYPE

        # Logical operators
        if op in (BinOpKind.AND, BinOpKind.OR):
            if left_type == BOOL_TYPE and right_type == BOOL_TYPE:
                return BOOL_TYPE
            self.error(
                f"Logical operator '{op.value}' requires bool operands, "
                f"got '{left_type}' and '{right_type}'",
                getattr(expr, 'loc', None),
            )
            return ERROR_TYPE

        self.error(f"Unknown operator: {op}", getattr(expr, 'loc', None))
        return ERROR_TYPE

    def _check_unary(self, expr) -> Type:
        """Type-check a unary expression."""
        operand_type = self.check_expr(expr.operand)

        if operand_type == ERROR_TYPE:
            return ERROR_TYPE

        if expr.op == UnaryOpKind.NEG:
            if self._is_numeric(operand_type):
                return operand_type
            self.error(
                f"Unary '-' requires numeric operand, got '{operand_type}'",
                getattr(expr, 'loc', None),
            )
            return ERROR_TYPE

        if expr.op == UnaryOpKind.NOT:
            if operand_type == BOOL_TYPE:
                return BOOL_TYPE
            self.error(
                f"Unary 'not' requires bool operand, got '{operand_type}'",
                getattr(expr, 'loc', None),
            )
            return ERROR_TYPE

        return ERROR_TYPE

    def _check_call(self, expr) -> Type:
        """
        Type-check a function call.

        Verifies that:
        1. The callee has a function type
        2. The number of arguments matches
        3. Each argument type matches the parameter type
        """
        callee_type = self.check_expr(expr.callee)

        if callee_type == ERROR_TYPE:
            return ERROR_TYPE

        if not isinstance(callee_type, FuncType):
            self.error(
                f"'{expr.callee}' is not callable "
                f"(type is '{callee_type}')",
                getattr(expr, 'loc', None),
            )
            return ERROR_TYPE

        # Check argument count
        expected = len(callee_type.param_types)
        actual = len(expr.arguments)
        if expected != actual:
            callee_name = (
                expr.callee.name
                if isinstance(expr.callee, Identifier) else "function"
            )
            self.error(
                f"'{callee_name}' expects {expected} argument(s), "
                f"got {actual}",
                getattr(expr, 'loc', None),
            )
            return ERROR_TYPE

        # Check argument types
        for i, (arg, param_type) in enumerate(
            zip(expr.arguments, callee_type.param_types)
        ):
            arg_type = self.check_expr(arg)
            if not self._type_assignable(param_type, arg_type):
                self.error(
                    f"Argument {i + 1}: expected '{param_type}', "
                    f"got '{arg_type}'",
                    getattr(arg, 'loc', None),
                )

        return callee_type.return_type

    def _check_index(self, expr) -> Type:
        """Type-check an index expression: a[i]."""
        obj_type = self.check_expr(expr.obj)
        idx_type = self.check_expr(expr.index)

        if obj_type == ERROR_TYPE or idx_type == ERROR_TYPE:
            return ERROR_TYPE

        if isinstance(obj_type, ListType):
            if idx_type != INT_TYPE:
                self.error(
                    f"List index must be int, got '{idx_type}'",
                    getattr(expr, 'loc', None),
                )
            return obj_type.element_type

        if obj_type == STRING_TYPE:
            if idx_type != INT_TYPE:
                self.error(
                    f"String index must be int, got '{idx_type}'",
                    getattr(expr, 'loc', None),
                )
            return STRING_TYPE

        self.error(
            f"Type '{obj_type}' does not support indexing",
            getattr(expr, 'loc', None),
        )
        return ERROR_TYPE

    def _check_if_expr(self, expr) -> Type:
        """Type-check a conditional expression."""
        cond_type = self.check_expr(expr.condition)
        then_type = self.check_expr(expr.then_expr)
        else_type = self.check_expr(expr.else_expr)

        if cond_type != BOOL_TYPE and cond_type != ERROR_TYPE:
            self.error(
                f"Condition must be bool, got '{cond_type}'",
                getattr(expr, 'loc', None),
            )

        if then_type == ERROR_TYPE or else_type == ERROR_TYPE:
            return ERROR_TYPE

        if then_type != else_type:
            self.error(
                f"If branches have different types: "
                f"'{then_type}' and '{else_type}'",
                getattr(expr, 'loc', None),
            )
            return ERROR_TYPE

        return then_type

    def _check_list_expr(self, expr) -> Type:
        """Type-check a list literal."""
        if not expr.elements:
            return ListType(element_type=VOID_TYPE)  # empty list

        elem_type = self.check_expr(expr.elements[0])
        for elem in expr.elements[1:]:
            t = self.check_expr(elem)
            if t != elem_type and t != ERROR_TYPE and elem_type != ERROR_TYPE:
                self.error(
                    f"List elements have inconsistent types: "
                    f"'{elem_type}' and '{t}'",
                    getattr(elem, 'loc', None),
                )
                return ERROR_TYPE

        return ListType(element_type=elem_type)

    # ─── Statement Type Checking ───

    def check_stmt(self, stmt):
        """Type-check a statement."""
        if isinstance(stmt, LetStmt):
            self._check_let(stmt)
        elif isinstance(stmt, AssignStmt):
            self._check_assign(stmt)
        elif isinstance(stmt, ExprStmt):
            self.check_expr(stmt.expr)
        elif isinstance(stmt, PrintStmt):
            self.check_expr(stmt.value)
        elif isinstance(stmt, IfStmt):
            self._check_if_stmt(stmt)
        elif isinstance(stmt, WhileStmt):
            self._check_while(stmt)
        elif isinstance(stmt, Block):
            self._check_block(stmt)
        elif isinstance(stmt, FuncDecl):
            self._check_func_decl(stmt)
        elif isinstance(stmt, ReturnStmt):
            self._check_return(stmt)
        elif isinstance(stmt, ForStmt):
            self._check_for(stmt)
        else:
            self.error(
                f"Unknown statement type: {type(stmt).__name__}",
                getattr(stmt, 'loc', None),
            )

    def _check_let(self, stmt):
        """Type-check a let declaration."""
        declared_type = None
        if stmt.type_ann is not None:
            declared_type = self._resolve_type_annotation(stmt.type_ann)

        init_type = None
        if stmt.initializer is not None:
            init_type = self.check_expr(stmt.initializer)

        # Determine the variable's type
        if declared_type is not None and init_type is not None:
            if not self._type_assignable(declared_type, init_type):
                self.error(
                    f"Cannot assign '{init_type}' to variable "
                    f"'{stmt.name}' of type '{declared_type}'",
                    getattr(stmt, 'loc', None),
                )
            var_type = declared_type
        elif declared_type is not None:
            var_type = declared_type
        elif init_type is not None:
            var_type = init_type  # type inference from initializer
        else:
            self.error(
                f"Variable '{stmt.name}' has no type annotation "
                f"and no initializer",
                getattr(stmt, 'loc', None),
            )
            var_type = ERROR_TYPE

        self.symbol_table.define(
            stmt.name,
            var_type,
            SymbolKind.VARIABLE,
            initialized=(stmt.initializer is not None),
            loc=getattr(stmt, 'loc', None),
        )

    def _check_assign(self, stmt):
        """Type-check an assignment statement."""
        if isinstance(stmt.target, Identifier):
            sym = self.symbol_table.resolve(
                stmt.target.name, getattr(stmt, 'loc', None)
            )
            if self.symbol_table.errors:
                self.errors.extend(self.symbol_table.errors)
                self.symbol_table.errors.clear()

            if not sym.is_mutable and sym.type != ERROR_TYPE:
                self.error(
                    f"Cannot assign to immutable variable "
                    f"'{stmt.target.name}'",
                    getattr(stmt, 'loc', None),
                )

            value_type = self.check_expr(stmt.value)
            if not self._type_assignable(sym.type, value_type):
                self.error(
                    f"Cannot assign '{value_type}' to '{sym.type}'",
                    getattr(stmt, 'loc', None),
                )

            sym.is_initialized = True
        else:
            # Complex assignment target (e.g., a[i] = ...)
            target_type = self.check_expr(stmt.target)
            value_type = self.check_expr(stmt.value)
            if not self._type_assignable(target_type, value_type):
                self.error(
                    f"Cannot assign '{value_type}' to '{target_type}'",
                    getattr(stmt, 'loc', None),
                )

    def _check_if_stmt(self, stmt):
        """Type-check an if statement."""
        cond_type = self.check_expr(stmt.condition)
        if cond_type != BOOL_TYPE and cond_type != ERROR_TYPE:
            self.error(
                f"If condition must be bool, got '{cond_type}'",
                getattr(stmt, 'loc', None),
            )

        self._check_block(stmt.then_body)
        if stmt.else_body is not None:
            self._check_block(stmt.else_body)

    def _check_while(self, stmt):
        """Type-check a while loop."""
        cond_type = self.check_expr(stmt.condition)
        if cond_type != BOOL_TYPE and cond_type != ERROR_TYPE:
            self.error(
                f"While condition must be bool, got '{cond_type}'",
                getattr(stmt, 'loc', None),
            )
        self._check_block(stmt.body)

    def _check_for(self, stmt):
        """Type-check a for loop."""
        iter_type = self.check_expr(stmt.iterable)
        if isinstance(iter_type, ListType):
            elem_type = iter_type.element_type
        elif iter_type == STRING_TYPE:
            elem_type = STRING_TYPE
        elif iter_type != ERROR_TYPE:
            self.error(
                f"Cannot iterate over type '{iter_type}'",
                getattr(stmt, 'loc', None),
            )
            elem_type = ERROR_TYPE
        else:
            elem_type = ERROR_TYPE

        self.symbol_table.enter_scope("for-body")
        self.symbol_table.define(
            stmt.var_name, elem_type,
            SymbolKind.VARIABLE, initialized=True,
        )
        self._check_block(stmt.body)
        self.symbol_table.exit_scope()

    def _check_block(self, block):
        """Type-check a block of statements."""
        self.symbol_table.enter_scope("block")
        for stmt in block.statements:
            self.check_stmt(stmt)
        self.symbol_table.exit_scope()

    def _check_func_decl(self, stmt):
        """Type-check a function declaration."""
        # Resolve parameter types
        param_types = []
        for param in stmt.params:
            if param.type_ann is not None:
                pt = self._resolve_type_annotation(param.type_ann)
            else:
                self.error(
                    f"Parameter '{param.name}' has no type annotation",
                    getattr(param, 'loc', None),
                )
                pt = ERROR_TYPE
            param_types.append(pt)

        # Resolve return type
        return_type = VOID_TYPE
        if stmt.return_type is not None:
            return_type = self._resolve_type_annotation(stmt.return_type)

        # Define function in current scope
        func_type = FuncType(
            param_types=tuple(param_types),
            return_type=return_type,
        )
        self.symbol_table.define(
            stmt.name, func_type,
            SymbolKind.FUNCTION, mutable=False, initialized=True,
            loc=getattr(stmt, 'loc', None),
        )

        # Check body in new scope
        if stmt.body is not None:
            prev_return = self.current_function_return_type
            self.current_function_return_type = return_type

            self.symbol_table.enter_scope(f"fn:{stmt.name}")
            for param, pt in zip(stmt.params, param_types):
                self.symbol_table.define(
                    param.name, pt,
                    SymbolKind.PARAMETER, initialized=True,
                )

            for s in stmt.body.statements:
                self.check_stmt(s)

            self.symbol_table.exit_scope()
            self.current_function_return_type = prev_return

    def _check_return(self, stmt):
        """Type-check a return statement."""
        if self.current_function_return_type is None:
            self.error(
                "Return statement outside of function",
                getattr(stmt, 'loc', None),
            )
            return

        if stmt.value is not None:
            value_type = self.check_expr(stmt.value)
            expected = self.current_function_return_type
            if not self._type_assignable(expected, value_type):
                self.error(
                    f"Return type mismatch: expected '{expected}', "
                    f"got '{value_type}'",
                    getattr(stmt, 'loc', None),
                )
        else:
            if self.current_function_return_type != VOID_TYPE:
                self.error(
                    f"Function should return '{self.current_function_return_type}', "
                    f"but return has no value",
                    getattr(stmt, 'loc', None),
                )

    # ─── Type Resolution Helpers ───

    def _resolve_type_annotation(self, ann) -> Type:
        """Convert a type annotation AST node to a Type."""
        if isinstance(ann, SimpleType):
            type_map = {
                "int": INT_TYPE,
                "float": FLOAT_TYPE,
                "bool": BOOL_TYPE,
                "string": STRING_TYPE,
                "void": VOID_TYPE,
            }
            t = type_map.get(ann.name)
            if t is None:
                self.error(f"Unknown type: '{ann.name}'")
                return ERROR_TYPE
            return t
        elif isinstance(ann, ListType):
            elem_t = self._resolve_type_annotation(ann.element_type)
            return ListType(element_type=elem_t)
        elif isinstance(ann, FuncType):
            params = tuple(
                self._resolve_type_annotation(p) for p in ann.param_types
            )
            ret = self._resolve_type_annotation(ann.return_type)
            return FuncType(param_types=params, return_type=ret)
        else:
            self.error(
                f"Unsupported type annotation: {type(ann).__name__}"
            )
            return ERROR_TYPE

    def _is_numeric(self, t: Type) -> bool:
        return t.kind in (TypeKind.INT, TypeKind.FLOAT)

    def _wider_numeric(self, a: Type, b: Type) -> Type:
        """Return the wider of two numeric types (int < float)."""
        if a == FLOAT_TYPE or b == FLOAT_TYPE:
            return FLOAT_TYPE
        return INT_TYPE

    def _type_assignable(self, target: Type, source: Type) -> bool:
        """Check if source type can be assigned to target type."""
        if target == ERROR_TYPE or source == ERROR_TYPE:
            return True  # Don't cascade errors
        if target == source:
            return True
        # Implicit widening: int -> float
        if target == FLOAT_TYPE and source == INT_TYPE:
            return True
        return False

    def _types_compatible(self, a: Type, b: Type) -> bool:
        """Check if two types can be compared for equality."""
        if a == ERROR_TYPE or b == ERROR_TYPE:
            return True
        return a == b or (self._is_numeric(a) and self._is_numeric(b))


# ─── Demo ───

if __name__ == "__main__":
    # Build AST for a program with type errors
    program = Program(statements=[
        # let x: int = 42;
        LetStmt("x", SimpleType("int"), IntLiteral(42)),
        # let y: string = "hello";
        LetStmt("y", SimpleType("string"), StringLiteral("hello")),
        # let z = x + y;  -- TYPE ERROR: int + string
        LetStmt("z", None, BinaryExpr(
            BinOpKind.ADD, Identifier("x"), Identifier("y"),
        )),
        # fn add(a: int, b: int) -> int { return a + b; }
        FuncDecl(
            "add",
            [Parameter("a", SimpleType("int")),
             Parameter("b", SimpleType("int"))],
            SimpleType("int"),
            Block([
                ReturnStmt(BinaryExpr(
                    BinOpKind.ADD,
                    Identifier("a"), Identifier("b"),
                ))
            ]),
        ),
        # let result = add(1, "two");  -- TYPE ERROR: string for int param
        LetStmt("result", None, CallExpr(
            Identifier("add"),
            [IntLiteral(1), StringLiteral("two")],
        )),
        # if (x) { ... }  -- TYPE ERROR: int used as bool
        IfStmt(
            condition=Identifier("x"),
            then_body=Block([
                PrintStmt(Identifier("x")),
            ]),
        ),
    ])

    checker = TypeChecker()
    success = checker.check_program(program)

    print(f"Type check {'passed' if success else 'FAILED'}!")
    print()

    if checker.errors:
        print(f"Found {len(checker.errors)} error(s):")
        for err in checker.errors:
            print(f"  {err}")
```

**Expected output:**

```
Type check FAILED!

Found 3 error(s):
  TypeError: Operator '+' not supported for types 'int' and 'string'
  TypeError: Argument 2: expected 'int', got 'string'
  TypeError: If condition must be bool, got 'int'
```

---

## 6. Type Inference

### 6.1 Local Type Inference

The simplest form of type inference: deduce a variable's type from its initializer.

```python
# The type checker already supports this:
let x = 42;           # inferred as int
let y = 3.14;         # inferred as float
let z = x + y;        # inferred as float (int + float -> float)
let msg = "hello";    # inferred as string
```

This is what TypeScript, Kotlin, Rust, and Go use for variable declarations.

### 6.2 Hindley-Milner Type Inference

**Hindley-Milner (HM)** type inference is more powerful: it can infer the types of function parameters and return types without annotations. It is used by Haskell, OCaml, F#, and (in a restricted form) Rust.

**Core idea:** Use **type variables** and **unification** to determine types.

$$
\frac{\Gamma, x:\alpha \vdash e : \tau}{\Gamma \vdash \lambda x.e : \alpha \to \tau}
$$

If we do not know the type of parameter $x$, we assign it a fresh type variable $\alpha$ and let unification determine its concrete type.

### 6.3 Unification Algorithm

**Unification** finds a substitution $\sigma$ that makes two types equal: $\sigma(\tau_1) = \sigma(\tau_2)$.

```python
"""
Simplified Hindley-Milner Type Inference

Implements type variables and unification for basic
type inference without explicit annotations.
"""


class TypeVar:
    """A type variable that can be unified with other types."""

    _counter = 0

    def __init__(self, name: str = None):
        if name is None:
            TypeVar._counter += 1
            name = f"T{TypeVar._counter}"
        self.name = name
        self.bound_to: Optional["InferType"] = None

    def resolve(self) -> "InferType":
        """Follow the chain of bindings to find the actual type."""
        if self.bound_to is None:
            return self
        if isinstance(self.bound_to, TypeVar):
            resolved = self.bound_to.resolve()
            self.bound_to = resolved  # path compression
            return resolved
        return self.bound_to

    def __repr__(self):
        resolved = self.resolve()
        if resolved is self:
            return f"?{self.name}"
        return repr(resolved)


# For HM inference, types are:
InferType = TypeVar | Type | FuncType


def unify(t1: InferType, t2: InferType) -> bool:
    """
    Unify two types, binding type variables as needed.

    Returns True if unification succeeds, False otherwise.
    """
    # Resolve any bound type variables
    if isinstance(t1, TypeVar):
        t1 = t1.resolve()
    if isinstance(t2, TypeVar):
        t2 = t2.resolve()

    # Same type variable: trivially unified
    if t1 is t2:
        return True

    # One is a type variable: bind it
    if isinstance(t1, TypeVar):
        if occurs_in(t1, t2):
            return False  # Occurs check: prevent infinite types
        t1.bound_to = t2
        return True

    if isinstance(t2, TypeVar):
        if occurs_in(t2, t1):
            return False
        t2.bound_to = t1
        return True

    # Both are concrete types: must be the same kind
    if isinstance(t1, Type) and isinstance(t2, Type):
        return t1 == t2

    # Both are function types: unify parameter and return types
    if isinstance(t1, FuncType) and isinstance(t2, FuncType):
        if len(t1.param_types) != len(t2.param_types):
            return False
        for p1, p2 in zip(t1.param_types, t2.param_types):
            if not unify(p1, p2):
                return False
        return unify(t1.return_type, t2.return_type)

    return False


def occurs_in(var: TypeVar, t: InferType) -> bool:
    """
    Occurs check: does type variable `var` appear in type `t`?

    This prevents creating infinite types like T = T -> T.
    """
    if isinstance(t, TypeVar):
        t = t.resolve()
        return t is var
    if isinstance(t, FuncType):
        for pt in t.param_types:
            if occurs_in(var, pt):
                return True
        return occurs_in(var, t.return_type)
    return False


# ─── Simple Type Inferencer ───

class TypeInferencer:
    """
    Simple Hindley-Milner style type inference.

    Assigns type variables to unknown types and uses
    unification to determine concrete types.
    """

    def __init__(self):
        self.env: dict[str, InferType] = {}
        self.errors: list[str] = []

    def fresh_var(self) -> TypeVar:
        """Create a fresh type variable."""
        return TypeVar()

    def infer(self, expr) -> InferType:
        """Infer the type of an expression."""
        if isinstance(expr, IntLiteral):
            return INT_TYPE

        elif isinstance(expr, BoolLiteral):
            return BOOL_TYPE

        elif isinstance(expr, StringLiteral):
            return STRING_TYPE

        elif isinstance(expr, Identifier):
            if expr.name in self.env:
                return self.env[expr.name]
            self.errors.append(f"Undefined: {expr.name}")
            return self.fresh_var()

        elif isinstance(expr, BinaryExpr):
            left_t = self.infer(expr.left)
            right_t = self.infer(expr.right)

            if expr.op in (
                BinOpKind.ADD, BinOpKind.SUB,
                BinOpKind.MUL, BinOpKind.DIV,
            ):
                # Both sides must be the same numeric type
                result_var = self.fresh_var()
                if not unify(left_t, result_var):
                    self.errors.append(
                        f"Type error in left operand of '{expr.op.value}'"
                    )
                if not unify(right_t, result_var):
                    self.errors.append(
                        f"Type error in right operand of '{expr.op.value}'"
                    )
                return result_var.resolve()

            elif expr.op in (
                BinOpKind.EQ, BinOpKind.LT,
                BinOpKind.GT, BinOpKind.LE, BinOpKind.GE,
            ):
                if not unify(left_t, right_t):
                    self.errors.append("Comparison type mismatch")
                return BOOL_TYPE

            return self.fresh_var()

        elif isinstance(expr, LambdaExpr):
            # λ(x, y) => body
            param_vars = []
            for param in expr.params:
                tv = self.fresh_var()
                self.env[param.name] = tv
                param_vars.append(tv)

            body_type = self.infer(expr.body)

            return FuncType(
                param_types=tuple(
                    v.resolve() if isinstance(v, TypeVar) else v
                    for v in param_vars
                ),
                return_type=(
                    body_type.resolve()
                    if isinstance(body_type, TypeVar) else body_type
                ),
            )

        elif isinstance(expr, CallExpr):
            func_type = self.infer(expr.callee)
            arg_types = [self.infer(arg) for arg in expr.arguments]

            ret_var = self.fresh_var()
            expected_func = FuncType(
                param_types=tuple(arg_types),
                return_type=ret_var,
            )

            if not unify(func_type, expected_func):
                self.errors.append("Function call type mismatch")

            return ret_var.resolve()

        return self.fresh_var()


# ─── Demo ───

if __name__ == "__main__":
    inferencer = TypeInferencer()

    # Infer type of: fn(x) => x + 1
    # Expected: (int) -> int
    expr = LambdaExpr(
        params=[Parameter("x")],
        body=BinaryExpr(
            BinOpKind.ADD,
            Identifier("x"),
            IntLiteral(1),
        ),
    )

    result_type = inferencer.infer(expr)
    print(f"fn(x) => x + 1 : {result_type}")
    # Output: fn(x) => x + 1 : (int) -> int

    # Infer: fn(a, b) => a + b
    # Expected: (?T, ?T) -> ?T  (polymorphic)
    inferencer2 = TypeInferencer()
    expr2 = LambdaExpr(
        params=[Parameter("a"), Parameter("b")],
        body=BinaryExpr(
            BinOpKind.ADD,
            Identifier("a"),
            Identifier("b"),
        ),
    )

    result_type2 = inferencer2.infer(expr2)
    print(f"fn(a, b) => a + b : {result_type2}")
```

---

## 7. Type Compatibility and Coercion

### 7.1 Type Compatibility Rules

| Rule | Description | Example |
|------|-------------|---------|
| **Identity** | Same type is always compatible | `int` = `int` |
| **Widening** | Smaller numeric type fits larger | `int` -> `float` |
| **Subtyping** | Subtype compatible with supertype | `Cat` -> `Animal` |
| **Structural** | Same structure = compatible | `{x: int, y: int}` = `{x: int, y: int}` |
| **Nominal** | Same name = compatible | Java class types |

### 7.2 Implicit Coercion

```python
def can_coerce(source: Type, target: Type) -> bool:
    """
    Check if source can be implicitly coerced to target.

    Coercion rules (ordered by safety):
    1. int -> float    (widening, safe)
    2. int -> string   (via str(), if language supports it)
    3. float -> int    (narrowing, UNSAFE -- not allowed implicitly)
    """
    COERCION_TABLE = {
        (TypeKind.INT, TypeKind.FLOAT): True,   # safe widening
        (TypeKind.INT, TypeKind.BOOL): False,    # no implicit
        (TypeKind.FLOAT, TypeKind.INT): False,   # narrowing, rejected
        (TypeKind.BOOL, TypeKind.INT): True,     # true=1, false=0
    }

    if source == target:
        return True

    return COERCION_TABLE.get(
        (source.kind, target.kind), False
    )


def insert_coercion(expr, source_type: Type, target_type: Type):
    """
    Wrap an expression in an explicit coercion node if needed.

    This makes the implicit coercion explicit in the AST,
    which simplifies code generation.
    """
    if source_type == target_type:
        return expr

    if can_coerce(source_type, target_type):
        return CallExpr(
            callee=Identifier(f"__coerce_{source_type}_to_{target_type}"),
            arguments=[expr],
        )

    raise TypeError(
        f"Cannot coerce '{source_type}' to '{target_type}'"
    )
```

---

## 8. Overloading Resolution

### 8.1 What is Overloading?

**Overloading** allows multiple functions (or operators) to share the same name but have different parameter types. The compiler must resolve which version to call based on the argument types.

### 8.2 Resolution Algorithm

```python
def resolve_overload(
    name: str,
    arg_types: list[Type],
    candidates: list[FuncType],
) -> Optional[FuncType]:
    """
    Resolve an overloaded function call.

    Strategy:
    1. Find exact matches
    2. If no exact match, find matches with implicit coercion
    3. If multiple matches, prefer the most specific one
    4. If still ambiguous, report an error

    Returns the resolved function type, or None if ambiguous/no match.
    """
    # Phase 1: Exact matches
    exact = []
    for candidate in candidates:
        if len(candidate.param_types) != len(arg_types):
            continue
        if all(
            p == a for p, a in zip(candidate.param_types, arg_types)
        ):
            exact.append(candidate)

    if len(exact) == 1:
        return exact[0]
    if len(exact) > 1:
        return None  # ambiguous

    # Phase 2: Matches with coercion
    coercible = []
    for candidate in candidates:
        if len(candidate.param_types) != len(arg_types):
            continue
        if all(
            can_coerce(a, p)
            for p, a in zip(candidate.param_types, arg_types)
        ):
            coercible.append(candidate)

    if len(coercible) == 1:
        return coercible[0]

    # Phase 3: Most specific match
    if len(coercible) > 1:
        # Prefer the candidate requiring fewer coercions
        def coercion_count(candidate):
            return sum(
                0 if p == a else 1
                for p, a in zip(candidate.param_types, arg_types)
            )

        coercible.sort(key=coercion_count)
        if coercion_count(coercible[0]) < coercion_count(coercible[1]):
            return coercible[0]

    return None  # no match or ambiguous


# ─── Example ───

if __name__ == "__main__":
    # Overloaded "add" function
    candidates = [
        FuncType(param_types=(INT_TYPE, INT_TYPE), return_type=INT_TYPE),
        FuncType(
            param_types=(FLOAT_TYPE, FLOAT_TYPE),
            return_type=FLOAT_TYPE,
        ),
        FuncType(
            param_types=(STRING_TYPE, STRING_TYPE),
            return_type=STRING_TYPE,
        ),
    ]

    # add(1, 2) -> exact match: (int, int) -> int
    result = resolve_overload("add", [INT_TYPE, INT_TYPE], candidates)
    print(f"add(int, int) resolves to: {result}")

    # add(1, 2.0) -> coercion match: (float, float) -> float
    result = resolve_overload("add", [INT_TYPE, FLOAT_TYPE], candidates)
    print(f"add(int, float) resolves to: {result}")

    # add("a", "b") -> exact match: (string, string) -> string
    result = resolve_overload(
        "add", [STRING_TYPE, STRING_TYPE], candidates
    )
    print(f'add(string, string) resolves to: {result}')

    # add(1, "b") -> no match
    result = resolve_overload("add", [INT_TYPE, STRING_TYPE], candidates)
    print(f'add(int, string) resolves to: {result}')
```

---

## 9. Declaration Processing

### 9.1 Forward References

Many languages allow declarations to reference names that appear later in the source:

```
fn is_even(n: int) -> bool {
    if (n == 0) return true;
    return is_odd(n - 1);      // forward reference to is_odd
}

fn is_odd(n: int) -> bool {
    if (n == 0) return false;
    return is_even(n - 1);     // backward reference to is_even
}
```

### 9.2 Two-Pass Strategy

To handle forward references, use two passes over declarations:

```python
def check_declarations_two_pass(declarations: list):
    """
    Two-pass declaration processing.

    Pass 1: Register all names and their types (signatures)
    Pass 2: Check function bodies (using all registered names)
    """
    symbol_table = SymbolTable()

    # Pass 1: Register signatures
    for decl in declarations:
        if isinstance(decl, FuncDecl):
            param_types = []
            for param in decl.params:
                pt = resolve_type_annotation(param.type_ann)
                param_types.append(pt)

            return_type = VOID_TYPE
            if decl.return_type:
                return_type = resolve_type_annotation(decl.return_type)

            func_type = FuncType(
                param_types=tuple(param_types),
                return_type=return_type,
            )
            symbol_table.define(
                decl.name, func_type,
                SymbolKind.FUNCTION, initialized=True,
            )

        elif isinstance(decl, LetStmt):
            if decl.type_ann:
                var_type = resolve_type_annotation(decl.type_ann)
            else:
                var_type = ERROR_TYPE  # Need initializer for inference
            symbol_table.define(
                decl.name, var_type, SymbolKind.VARIABLE,
            )

    # Pass 2: Check bodies
    for decl in declarations:
        if isinstance(decl, FuncDecl) and decl.body is not None:
            check_function_body(decl, symbol_table)
        elif isinstance(decl, LetStmt) and decl.initializer is not None:
            check_initializer(decl, symbol_table)
```

### 9.3 Topological Sorting

For non-function declarations that depend on each other, topological sorting ensures declarations are processed in dependency order:

```
let a = b + 1;    // depends on b
let b = 10;       // no dependencies
let c = a + b;    // depends on a and b

Dependency graph:
  c --> a --> b
  c --> b

Topological order: b, a, c
```

---

## 10. Semantic Error Reporting

### 10.1 Error Categories

| Category | Example | Severity |
|----------|---------|----------|
| Undefined name | `print(x)` where x not defined | Error |
| Type mismatch | `"hello" + 42` | Error |
| Wrong arity | `f(1, 2)` but f takes 1 arg | Error |
| Duplicate definition | `let x = 1; let x = 2;` | Error (or warning) |
| Unused variable | `let x = 5;` (x never read) | Warning |
| Unreachable code | Code after `return` | Warning |
| Implicit narrowing | `let x: int = 3.14;` | Warning (or error) |
| Shadowing | Inner `x` shadows outer `x` | Info/Warning |

### 10.2 Error Recovery in Type Checking

The type checker should continue after errors to report as many issues as possible in one pass. The key technique is the **error type**:

```python
# ERROR_TYPE acts as a "universal" type that is compatible with everything.
# This prevents cascading errors.

def _type_assignable(self, target: Type, source: Type) -> bool:
    if target == ERROR_TYPE or source == ERROR_TYPE:
        return True  # Suppress cascading errors
    # ... normal checking
```

### 10.3 Helpful Error Messages

Good compiler errors should be:

1. **Precise**: Point to the exact location of the problem
2. **Clear**: Explain what is wrong in plain language
3. **Helpful**: Suggest how to fix the issue

```python
class DiagnosticFormatter:
    """Format semantic errors with context and suggestions."""

    @staticmethod
    def type_mismatch(
        expected: Type,
        actual: Type,
        context: str,
        loc,
        source_lines: list[str],
    ) -> str:
        lines = [
            f"error: type mismatch in {context}",
            f"  --> {loc}",
        ]

        if loc and 0 < loc.line <= len(source_lines):
            line = source_lines[loc.line - 1]
            lines.append(f"   |")
            lines.append(f"{loc.line:>3} | {line}")
            lines.append(
                f"   | {' ' * (loc.column - 1)}"
                f"{'~' * max(1, loc.end_column - loc.column)}"
            )

        lines.append(f"   = expected: {expected}")
        lines.append(f"   = found:    {actual}")

        # Suggestions
        if expected == INT_TYPE and actual == FLOAT_TYPE:
            lines.append(
                f"   = help: use an explicit cast: "
                f"int(value)"
            )
        elif expected == STRING_TYPE and actual == INT_TYPE:
            lines.append(
                f"   = help: convert to string: "
                f"str(value)"
            )

        return "\n".join(lines)
```

**Example formatted error:**

```
error: type mismatch in function argument
  --> main.lang:15:20
   |
 15 |     let result = add(1, "two");
   |                       ~~~~~
   = expected: int
   = found:    string
   = help: convert to int if possible: int("two")
```

---

## 11. Summary

Semantic analysis is the bridge between syntactic correctness (parsing) and meaningful program behavior (code generation). It verifies that the program makes sense: names are defined, types are consistent, and language rules are followed.

**Key concepts:**

| Concept | Description |
|---------|-------------|
| **Attribute grammars** | Formalize how semantic information flows in the parse tree |
| **S-attributed** | Synthesized attributes only; bottom-up evaluation |
| **L-attributed** | Synthesized + left-to-right inherited; single-pass evaluation |
| **Symbol table** | Maps names to their types, scopes, and attributes |
| **Scope management** | Chained hash tables for nested scopes with shadowing |
| **Type checking** | Verify that operations are applied to compatible types |
| **Type inference** | Deduce types from usage (local inference or HM unification) |
| **Coercion** | Implicit type conversion (e.g., int to float) |
| **Overloading** | Resolving the right function from multiple candidates |
| **Error recovery** | Using ERROR_TYPE to prevent cascading type errors |

**Design guidelines:**

1. Use a two-pass approach (register declarations, then check bodies) for forward references
2. Design an error type that suppresses cascading errors
3. Track source locations throughout for precise error reporting
4. Keep coercion rules explicit and minimal to avoid surprises
5. Support both explicit type annotations and local type inference
6. Report all errors in a single pass (do not stop at the first error)

---

## Exercises

### Exercise 1: Symbol Table Extension

Extend the `SymbolTable` class to track:

1. **Unused variables**: After type-checking, report variables that were defined but never referenced
2. **Shadowing warnings**: Report when an inner scope variable shadows an outer one
3. **Constant propagation**: Track which variables are bound to known constants

### Exercise 2: Full Type Checker

Extend the type checker to handle these additional constructs:

1. Array/list operations: `append`, `pop`, `len`
2. String operations: `+` (concatenation), `len`, indexing
3. Compound assignment: `+=`, `-=`, `*=`, `/=`
4. Ternary operator: `cond ? then_expr : else_expr`

Ensure proper type checking for each operation and write test cases with both valid and invalid programs.

### Exercise 3: Type Inference

Implement local type inference that handles:

```
let x = 5;                 // inferred as int
let y = [1, 2, 3];         // inferred as list[int]
let z = fn(a) => a + 1;    // inferred as (int) -> int
let w = if true then 1 else 2;  // inferred as int
```

Your inferencer should produce helpful errors when inference fails (e.g., `let a = [];` -- cannot infer element type of empty list).

### Exercise 4: Semantic Error Catalog

Create a test suite that exercises at least 15 different semantic errors. For each error, write:

1. A minimal program that triggers it
2. The expected error message
3. A corrected version of the program

Example errors to include: undefined variable, type mismatch in assignment, wrong number of arguments, return outside function, break outside loop, duplicate function parameter names, recursive type definition.

### Exercise 5: Overloading with Generics

Extend the overloading resolution to handle simple generics:

```
fn identity<T>(x: T) -> T { return x; }
fn pair<T, U>(a: T, b: U) -> (T, U) { return (a, b); }

identity(5)          // T = int, returns int
identity("hello")    // T = string, returns string
pair(1, "two")       // T = int, U = string, returns (int, string)
```

Implement the generic instantiation logic: when calling `identity(5)`, determine that `T = int` and verify the return type is `int`.

### Exercise 6: Control Flow Analysis

Implement a semantic analysis pass that checks:

1. Every function with a non-void return type returns on all paths
2. `break` and `continue` only appear inside loops
3. Code after `return` is unreachable (emit a warning)
4. Variables are initialized before use on all paths

This requires analyzing the control flow graph of each function. Start with a simplified version that handles if-else and while loops.

---

[Previous: Abstract Syntax Trees](./07_Abstract_Syntax_Trees.md) | [Next: Intermediate Representations](./09_Intermediate_Representations.md) | [Overview](./00_Overview.md)
