"""
24_closure_conversion.py - Closure Conversion and Lambda Lifting

Demonstrates how compilers implement closures (functions that capture
variables from enclosing scopes) by transforming them into plain
functions with explicit environment parameters.

Components:
  1. Free Variable Analysis
     Determine which variables a function body references from outer
     scopes (its "free variables").

  2. Closure Conversion
     Transform closures into pairs of (function_pointer, environment).
     The function receives the environment as an extra parameter and
     loads captured variables from it.

  3. Lambda Lifting
     An alternative to closure conversion: lift nested functions to
     the top level by adding free variables as extra parameters.

  4. Environment Representation
     Compare flat closures (copy all free vars into a record) with
     linked closures (chain of environment frames).

Topics covered:
  - Free variable computation
  - Closure representation strategies
  - Flat vs linked environments
  - Lambda lifting transformation
  - Defunctionalization (converting higher-order to first-order)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Optional, Union


# ---------------------------------------------------------------------------
# AST for a small functional language
# ---------------------------------------------------------------------------

@dataclass
class Num:
    value: int

@dataclass
class Var:
    name: str

@dataclass
class BinOp:
    op: str
    left: Any
    right: Any

@dataclass
class Lam:
    param: str
    body: Any

@dataclass
class App:
    func: Any
    arg: Any

@dataclass
class Let:
    name: str
    value: Any
    body: Any

@dataclass
class If:
    cond: Any
    then_expr: Any
    else_expr: Any


# ---------------------------------------------------------------------------
# Pretty Printer
# ---------------------------------------------------------------------------

def pretty(node: Any, indent: int = 0) -> str:
    pad = "  " * indent
    if isinstance(node, Num):
        return f"{node.value}"
    if isinstance(node, Var):
        return node.name
    if isinstance(node, BinOp):
        return f"({pretty(node.left)} {node.op} {pretty(node.right)})"
    if isinstance(node, Lam):
        return f"(fn {node.param} -> {pretty(node.body)})"
    if isinstance(node, App):
        return f"({pretty(node.func)} {pretty(node.arg)})"
    if isinstance(node, Let):
        return f"let {node.name} = {pretty(node.value)} in {pretty(node.body)}"
    if isinstance(node, If):
        return (f"if {pretty(node.cond)} then {pretty(node.then_expr)} "
                f"else {pretty(node.else_expr)}")
    return str(node)


# ---------------------------------------------------------------------------
# Free Variable Analysis
# ---------------------------------------------------------------------------

def free_vars(node: Any) -> set[str]:
    """Compute the set of free variables in an expression."""
    if isinstance(node, Num):
        return set()
    if isinstance(node, Var):
        return {node.name}
    if isinstance(node, BinOp):
        return free_vars(node.left) | free_vars(node.right)
    if isinstance(node, Lam):
        return free_vars(node.body) - {node.param}
    if isinstance(node, App):
        return free_vars(node.func) | free_vars(node.arg)
    if isinstance(node, Let):
        return free_vars(node.value) | (free_vars(node.body) - {node.name})
    if isinstance(node, If):
        return (free_vars(node.cond) | free_vars(node.then_expr) |
                free_vars(node.else_expr))
    return set()


# ---------------------------------------------------------------------------
# Closure Conversion
# ---------------------------------------------------------------------------

@dataclass
class ClosureCreate:
    """Create a closure: (func_label, [captured_var1, captured_var2, ...])"""
    func_label: str
    captured: list[str]

    def __str__(self):
        caps = ", ".join(self.captured)
        return f"make_closure({self.func_label}, [{caps}])"


@dataclass
class ClosureCall:
    """Call a closure: closure(arg)"""
    closure: Any
    arg: Any

    def __str__(self):
        return f"call_closure({self.closure}, {self.arg})"


@dataclass
class EnvLoad:
    """Load a captured variable from the closure environment."""
    env_var: str
    index: int

    def __str__(self):
        return f"{self.env_var}[{self.index}]"


@dataclass
class TopLevelFunc:
    """A top-level function after closure conversion."""
    label: str
    env_param: str
    param: str
    body: Any
    free_vars: list[str]

    def __str__(self):
        fvs = ", ".join(self.free_vars)
        return (f"func {self.label}({self.env_param}, {self.param}) "
                f"[captures: {fvs}]:\n"
                f"  {pretty(self.body)}")


class ClosureConverter:
    """
    Transform closures into top-level functions with explicit environments.
    Each lambda becomes:
      - A top-level function with an extra 'env' parameter
      - A closure creation that captures free variables
    """

    def __init__(self):
        self.func_counter = 0
        self.top_level_funcs: list[TopLevelFunc] = []
        self.log: list[str] = []

    def _fresh_label(self) -> str:
        self.func_counter += 1
        return f"__closure_{self.func_counter}"

    def convert(self, node: Any, bound: set[str] = None) -> Any:
        if bound is None:
            bound = set()

        if isinstance(node, Num):
            return node

        if isinstance(node, Var):
            return node

        if isinstance(node, BinOp):
            return BinOp(node.op,
                         self.convert(node.left, bound),
                         self.convert(node.right, bound))

        if isinstance(node, Lam):
            fvs = sorted(free_vars(node) & bound)
            label = self._fresh_label()
            env_param = f"__env_{label}"

            # Convert body, replacing free var references with env loads
            new_bound = bound | {node.param}
            converted_body = self.convert(node.body, new_bound)

            # Replace captured variables with environment loads
            for i, fv in enumerate(fvs):
                converted_body = self._substitute(
                    converted_body, fv, EnvLoad(env_param, i))

            self.top_level_funcs.append(
                TopLevelFunc(label, env_param, node.param,
                             converted_body, fvs))

            self.log.append(
                f"  Lambda ({node.param}) -> {label}, "
                f"captures: {fvs}")

            return ClosureCreate(label, fvs)

        if isinstance(node, App):
            func = self.convert(node.func, bound)
            arg = self.convert(node.arg, bound)
            return ClosureCall(func, arg)

        if isinstance(node, Let):
            val = self.convert(node.value, bound)
            body = self.convert(node.body, bound | {node.name})
            return Let(node.name, val, body)

        if isinstance(node, If):
            return If(self.convert(node.cond, bound),
                      self.convert(node.then_expr, bound),
                      self.convert(node.else_expr, bound))

        return node

    def _substitute(self, node: Any, name: str, replacement: Any) -> Any:
        """Replace occurrences of variable 'name' with 'replacement'."""
        if isinstance(node, Var) and node.name == name:
            return replacement
        if isinstance(node, Num):
            return node
        if isinstance(node, BinOp):
            return BinOp(node.op,
                         self._substitute(node.left, name, replacement),
                         self._substitute(node.right, name, replacement))
        if isinstance(node, Let):
            val = self._substitute(node.value, name, replacement)
            body = node.body if node.name == name else \
                   self._substitute(node.body, name, replacement)
            return Let(node.name, val, body)
        if isinstance(node, ClosureCall):
            return ClosureCall(
                self._substitute(node.closure, name, replacement),
                self._substitute(node.arg, name, replacement))
        if isinstance(node, ClosureCreate):
            return node
        if isinstance(node, EnvLoad):
            return node
        if isinstance(node, If):
            return If(self._substitute(node.cond, name, replacement),
                      self._substitute(node.then_expr, name, replacement),
                      self._substitute(node.else_expr, name, replacement))
        return node


# ---------------------------------------------------------------------------
# Lambda Lifting
# ---------------------------------------------------------------------------

class LambdaLifter:
    """
    Alternative to closure conversion: lift nested functions to the
    top level by passing free variables as extra parameters.
    """

    def __init__(self):
        self.func_counter = 0
        self.lifted_funcs: list[tuple[str, list[str], Any]] = []
        self.log: list[str] = []

    def _fresh_name(self) -> str:
        self.func_counter += 1
        return f"__lifted_{self.func_counter}"

    def lift(self, node: Any, bound: set[str] = None) -> Any:
        if bound is None:
            bound = set()

        if isinstance(node, Num) or isinstance(node, Var):
            return node

        if isinstance(node, BinOp):
            return BinOp(node.op,
                         self.lift(node.left, bound),
                         self.lift(node.right, bound))

        if isinstance(node, Lam):
            fvs = sorted(free_vars(node) & bound)
            name = self._fresh_name()
            lifted_body = self.lift(node.body, bound | {node.param})
            all_params = fvs + [node.param]
            self.lifted_funcs.append((name, all_params, lifted_body))

            self.log.append(
                f"  Lifted (fn {node.param} -> ...) to "
                f"{name}({', '.join(all_params)})")

            # Replace lambda with partial application of lifted function
            result: Any = Var(name)
            for fv in fvs:
                result = App(result, Var(fv))
            return result

        if isinstance(node, App):
            return App(self.lift(node.func, bound),
                       self.lift(node.arg, bound))

        if isinstance(node, Let):
            val = self.lift(node.value, bound)
            body = self.lift(node.body, bound | {node.name})
            return Let(node.name, val, body)

        return node


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Closure Conversion and Lambda Lifting Demo")
    print("=" * 60)

    # Example 1: Simple closure
    # let x = 10 in let add_x = fn y -> x + y in add_x 5
    expr1 = Let("x", Num(10),
                Let("add_x", Lam("y", BinOp("+", Var("x"), Var("y"))),
                    App(Var("add_x"), Num(5))))

    print(f"\n--- Example 1: Simple Closure ---")
    print(f"  Source: {pretty(expr1)}")
    fvs1 = free_vars(Lam("y", BinOp("+", Var("x"), Var("y"))))
    print(f"  Free vars of (fn y -> x + y): {fvs1}")

    # Closure conversion
    cc = ClosureConverter()
    converted1 = cc.convert(expr1, set())
    print(f"\n  After closure conversion:")
    for entry in cc.log:
        print(entry)
    for func in cc.top_level_funcs:
        print(f"  {func}")

    # Example 2: Nested closures (counter factory)
    # let make_adder = fn x -> fn y -> x + y
    # in let add5 = make_adder 5
    # in add5 3
    expr2 = Let("make_adder",
                Lam("x", Lam("y", BinOp("+", Var("x"), Var("y")))),
                Let("add5", App(Var("make_adder"), Num(5)),
                    App(Var("add5"), Num(3))))

    print(f"\n--- Example 2: Nested Closures ---")
    print(f"  Source: {pretty(expr2)}")

    cc2 = ClosureConverter()
    converted2 = cc2.convert(expr2, set())
    print(f"\n  After closure conversion:")
    for entry in cc2.log:
        print(entry)
    for func in cc2.top_level_funcs:
        print(f"  {func}")

    # Lambda lifting alternative
    print(f"\n--- Lambda Lifting (Alternative) ---")
    ll = LambdaLifter()
    lifted = ll.lift(expr1, set())
    print(f"  Source: {pretty(expr1)}")
    for entry in ll.log:
        print(entry)
    for name, params, body in ll.lifted_funcs:
        print(f"  func {name}({', '.join(params)}): {pretty(body)}")

    # Example 3: Free variable analysis
    print(f"\n--- Free Variable Analysis ---")
    examples = [
        ("fn x -> x + 1", Lam("x", BinOp("+", Var("x"), Num(1)))),
        ("fn x -> x + y", Lam("x", BinOp("+", Var("x"), Var("y")))),
        ("fn x -> fn y -> x + y + z",
         Lam("x", Lam("y", BinOp("+", BinOp("+", Var("x"), Var("y")),
                                  Var("z"))))),
        ("let a = 1 in fn x -> a + x",
         Let("a", Num(1), Lam("x", BinOp("+", Var("a"), Var("x"))))),
    ]
    for desc, expr in examples:
        fvs = free_vars(expr)
        print(f"  {desc}  ->  FV = {fvs}")

    print(f"\n--- Closure Conversion vs Lambda Lifting ---")
    print("""
  Closure Conversion:
    - Creates (function_ptr, environment) pairs
    - Environment stores captured values
    - Uniform calling convention (all closures look the same)
    - Used by: OCaml, Haskell (STG), JavaScript engines

  Lambda Lifting:
    - Adds free variables as extra parameters
    - No runtime environment allocation
    - Caller must pass all captured values explicitly
    - Used by: GHC (as optimization), some C compilers for nested functions

  Flat vs Linked Environments:
    - Flat: copy all captured vars into one record (fast access, copy cost)
    - Linked: chain of scope frames (sharing, slower access via indirection)
    """)


if __name__ == "__main__":
    main()
