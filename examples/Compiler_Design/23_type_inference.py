"""
23_type_inference.py - Hindley-Milner Type Inference

Demonstrates the Hindley-Milner type inference algorithm, the
foundation of type systems in ML, Haskell, Rust (partial), and
other languages with parametric polymorphism.

Components:
  1. Type Representation
     Concrete types (Int, Bool, String), type variables, and
     function types (arrow types).

  2. Unification
     The core algorithm that finds the most general substitution
     making two types equal.

  3. Algorithm W
     The classic type inference algorithm that walks an expression
     tree, generates type constraints, and solves them via unification.

  4. Let Polymorphism
     Generalize types at let-bindings to enable polymorphic reuse
     (e.g., `let id = fn x -> x` gives id the type `forall a. a -> a`).

  5. Type Schemes and Instantiation
     Represent polymorphic types as type schemes (universally quantified
     types) and instantiate them with fresh variables at each use site.

Topics covered:
  - Type variables and substitution
  - Unification algorithm (Robinson's)
  - Occurs check (prevents infinite types)
  - Algorithm W for type inference
  - Let-polymorphism (generalization/instantiation)
  - Constraint-based type inference
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Type Representation
# ---------------------------------------------------------------------------

class Type:
    """Base class for types."""
    pass


@dataclass(frozen=True)
class TConst(Type):
    """Concrete type: Int, Bool, String, etc."""
    name: str

    def __str__(self):
        return self.name


@dataclass
class TVar(Type):
    """Type variable (e.g., 'a, 'b)."""
    name: str

    def __str__(self):
        return self.name

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return isinstance(other, TVar) and self.name == other.name


@dataclass
class TArrow(Type):
    """Function type: param -> result."""
    param: Type
    result: Type

    def __str__(self):
        p = f"({self.param})" if isinstance(self.param, TArrow) else str(self.param)
        return f"{p} -> {self.result}"


@dataclass
class TList(Type):
    """List type: [a]."""
    elem: Type

    def __str__(self):
        return f"[{self.elem}]"


# Convenience constants
INT = TConst("Int")
BOOL = TConst("Bool")
STRING = TConst("String")


# ---------------------------------------------------------------------------
# Type Scheme (polymorphic types)
# ---------------------------------------------------------------------------

@dataclass
class Scheme:
    """
    Type scheme: forall a1 a2 ... . type
    Represents a polymorphic type with universally quantified variables.
    """
    vars: list[str]
    type: Type

    def __str__(self):
        if self.vars:
            vs = " ".join(self.vars)
            return f"forall {vs}. {self.type}"
        return str(self.type)


# ---------------------------------------------------------------------------
# Substitution
# ---------------------------------------------------------------------------

class Substitution:
    """A mapping from type variables to types."""

    def __init__(self, mapping: Optional[dict[str, Type]] = None):
        self.mapping: dict[str, Type] = mapping or {}

    def apply(self, t: Type) -> Type:
        if isinstance(t, TConst):
            return t
        if isinstance(t, TVar):
            if t.name in self.mapping:
                return self.apply(self.mapping[t.name])
            return t
        if isinstance(t, TArrow):
            return TArrow(self.apply(t.param), self.apply(t.result))
        if isinstance(t, TList):
            return TList(self.apply(t.elem))
        return t

    def apply_scheme(self, scheme: Scheme) -> Scheme:
        # Don't substitute bound variables
        restricted = Substitution(
            {k: v for k, v in self.mapping.items() if k not in scheme.vars}
        )
        return Scheme(scheme.vars, restricted.apply(scheme.type))

    def compose(self, other: Substitution) -> Substitution:
        """Compose two substitutions: self after other."""
        new_mapping = {k: self.apply(v) for k, v in other.mapping.items()}
        new_mapping.update(self.mapping)
        return Substitution(new_mapping)

    def __str__(self):
        if not self.mapping:
            return "{}"
        entries = ", ".join(f"{k} := {v}" for k, v in self.mapping.items())
        return f"{{{entries}}}"


# ---------------------------------------------------------------------------
# Unification
# ---------------------------------------------------------------------------

class UnificationError(Exception):
    pass


def occurs_in(var_name: str, t: Type) -> bool:
    """Check if a type variable occurs in a type (prevents infinite types)."""
    if isinstance(t, TVar):
        return t.name == var_name
    if isinstance(t, TArrow):
        return occurs_in(var_name, t.param) or occurs_in(var_name, t.result)
    if isinstance(t, TList):
        return occurs_in(var_name, t.elem)
    return False


def unify(t1: Type, t2: Type) -> Substitution:
    """
    Find the most general unifier of two types.
    Returns a substitution that makes t1 and t2 equal.
    """
    if isinstance(t1, TConst) and isinstance(t2, TConst) and t1.name == t2.name:
        return Substitution()

    if isinstance(t1, TVar):
        if t1 == t2:
            return Substitution()
        if occurs_in(t1.name, t2):
            raise UnificationError(
                f"Occurs check failed: {t1} occurs in {t2}")
        return Substitution({t1.name: t2})

    if isinstance(t2, TVar):
        return unify(t2, t1)

    if isinstance(t1, TArrow) and isinstance(t2, TArrow):
        s1 = unify(t1.param, t2.param)
        s2 = unify(s1.apply(t1.result), s1.apply(t2.result))
        return s2.compose(s1)

    if isinstance(t1, TList) and isinstance(t2, TList):
        return unify(t1.elem, t2.elem)

    raise UnificationError(f"Cannot unify {t1} with {t2}")


# ---------------------------------------------------------------------------
# Expression AST
# ---------------------------------------------------------------------------

@dataclass
class EInt:
    value: int

@dataclass
class EBool:
    value: bool

@dataclass
class EStr:
    value: str

@dataclass
class EVar:
    name: str

@dataclass
class EApp:
    func: object
    arg: object

@dataclass
class ELam:
    param: str
    body: object

@dataclass
class ELet:
    name: str
    value: object
    body: object

@dataclass
class EIf:
    cond: object
    then_expr: object
    else_expr: object


# ---------------------------------------------------------------------------
# Type Environment
# ---------------------------------------------------------------------------

class TypeEnv:
    """Maps variable names to type schemes."""

    def __init__(self, bindings: Optional[dict[str, Scheme]] = None):
        self.bindings: dict[str, Scheme] = bindings or {}

    def extend(self, name: str, scheme: Scheme) -> TypeEnv:
        new_bindings = dict(self.bindings)
        new_bindings[name] = scheme
        return TypeEnv(new_bindings)

    def lookup(self, name: str) -> Optional[Scheme]:
        return self.bindings.get(name)

    def free_vars(self) -> set[str]:
        result = set()
        for scheme in self.bindings.values():
            result |= _free_vars_type(scheme.type) - set(scheme.vars)
        return result


def _free_vars_type(t: Type) -> set[str]:
    if isinstance(t, TConst):
        return set()
    if isinstance(t, TVar):
        return {t.name}
    if isinstance(t, TArrow):
        return _free_vars_type(t.param) | _free_vars_type(t.result)
    if isinstance(t, TList):
        return _free_vars_type(t.elem)
    return set()


# ---------------------------------------------------------------------------
# Algorithm W
# ---------------------------------------------------------------------------

class InferenceEngine:
    """Hindley-Milner type inference using Algorithm W."""

    def __init__(self):
        self.var_counter = 0
        self.log: list[str] = []

    def fresh_var(self) -> TVar:
        self.var_counter += 1
        return TVar(f"t{self.var_counter}")

    def instantiate(self, scheme: Scheme) -> Type:
        """Replace bound variables with fresh type variables."""
        mapping = {v: self.fresh_var() for v in scheme.vars}
        subst = Substitution(mapping)
        return subst.apply(scheme.type)

    def generalize(self, env: TypeEnv, t: Type) -> Scheme:
        """Generalize a type by quantifying free variables not in env."""
        env_fv = env.free_vars()
        type_fv = _free_vars_type(t)
        gen_vars = sorted(type_fv - env_fv)
        return Scheme(gen_vars, t)

    def infer(self, env: TypeEnv, expr: object) -> tuple[Substitution, Type]:
        """
        Algorithm W: infer the type of an expression.
        Returns (substitution, inferred_type).
        """
        if isinstance(expr, EInt):
            return Substitution(), INT

        if isinstance(expr, EBool):
            return Substitution(), BOOL

        if isinstance(expr, EStr):
            return Substitution(), STRING

        if isinstance(expr, EVar):
            scheme = env.lookup(expr.name)
            if scheme is None:
                raise UnificationError(f"Unbound variable: {expr.name}")
            t = self.instantiate(scheme)
            self.log.append(f"  Var '{expr.name}': {scheme} -> {t}")
            return Substitution(), t

        if isinstance(expr, ELam):
            param_type = self.fresh_var()
            new_env = env.extend(expr.param, Scheme([], param_type))
            s, body_type = self.infer(new_env, expr.body)
            result_type = TArrow(s.apply(param_type), body_type)
            self.log.append(f"  Lambda {expr.param}: {result_type}")
            return s, result_type

        if isinstance(expr, EApp):
            result_type = self.fresh_var()
            s1, func_type = self.infer(env, expr.func)
            s2, arg_type = self.infer(
                TypeEnv({k: s1.apply_scheme(v) for k, v in env.bindings.items()}),
                expr.arg
            )
            s3 = unify(s2.apply(func_type), TArrow(arg_type, result_type))
            final_type = s3.apply(result_type)
            return s3.compose(s2.compose(s1)), final_type

        if isinstance(expr, ELet):
            s1, val_type = self.infer(env, expr.value)
            applied_env = TypeEnv(
                {k: s1.apply_scheme(v) for k, v in env.bindings.items()})
            scheme = self.generalize(applied_env, val_type)
            self.log.append(f"  Let '{expr.name}': {scheme}")
            new_env = applied_env.extend(expr.name, scheme)
            s2, body_type = self.infer(new_env, expr.body)
            return s2.compose(s1), body_type

        if isinstance(expr, EIf):
            s1, cond_type = self.infer(env, expr.cond)
            s2 = unify(cond_type, BOOL).compose(s1)
            env2 = TypeEnv({k: s2.apply_scheme(v) for k, v in env.bindings.items()})
            s3, then_type = self.infer(env2, expr.then_expr)
            env3 = TypeEnv({k: s3.apply_scheme(v) for k, v in env2.bindings.items()})
            s4, else_type = self.infer(env3, expr.else_expr)
            s5 = unify(s4.apply(then_type), else_type)
            return s5.compose(s4.compose(s3.compose(s2))), s5.apply(else_type)

        raise UnificationError(f"Unknown expression: {type(expr).__name__}")


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def infer_and_print(name: str, expr: object,
                    env: Optional[TypeEnv] = None) -> None:
    """Infer and display the type of an expression."""
    engine = InferenceEngine()
    if env is None:
        env = TypeEnv()
    try:
        subst, t = engine.infer(env, expr)
        final = subst.apply(t)
        print(f"  {name}: {final}")
    except UnificationError as e:
        print(f"  {name}: TYPE ERROR - {e}")


def main():
    print("=" * 60)
    print("Hindley-Milner Type Inference Demo")
    print("=" * 60)

    # Basic literals
    print("\n--- Literals ---")
    infer_and_print("42", EInt(42))
    infer_and_print("true", EBool(True))
    infer_and_print('"hi"', EStr("hi"))

    # Lambda expressions
    print("\n--- Lambda Expressions ---")
    # fn x -> x  (identity)
    infer_and_print("fn x -> x", ELam("x", EVar("x")))
    # fn x -> fn y -> x  (const)
    infer_and_print("fn x -> fn y -> x",
                    ELam("x", ELam("y", EVar("x"))))

    # Application with built-in add
    print("\n--- Application ---")
    add_type = Scheme([], TArrow(INT, TArrow(INT, INT)))
    env = TypeEnv({"add": add_type, "not": Scheme([], TArrow(BOOL, BOOL))})

    # add 1 2
    infer_and_print("add 1",
                    EApp(EVar("add"), EInt(1)), env)
    infer_and_print("add 1 2",
                    EApp(EApp(EVar("add"), EInt(1)), EInt(2)), env)

    # Let polymorphism
    print("\n--- Let Polymorphism ---")
    # let id = fn x -> x in (id 42, id true)
    # id gets type forall a. a -> a
    let_id = ELet("id", ELam("x", EVar("x")),
                  EApp(EVar("id"), EInt(42)))
    infer_and_print("let id = fn x -> x in id 42", let_id)

    # Type error detection
    print("\n--- Type Errors ---")
    # if 42 then 1 else 2  (condition must be Bool)
    infer_and_print("if 42 then 1 else 2",
                    EIf(EInt(42), EInt(1), EInt(2)))

    # not 42  (not expects Bool, got Int)
    infer_and_print("not 42",
                    EApp(EVar("not"), EInt(42)), env)

    # Unification demo
    print("\n--- Unification Examples ---")
    examples = [
        (INT, INT, "Int ~ Int"),
        (TVar("a"), INT, "a ~ Int"),
        (TArrow(TVar("a"), TVar("b")), TArrow(INT, BOOL),
         "(a -> b) ~ (Int -> Bool)"),
    ]
    for t1, t2, desc in examples:
        try:
            s = unify(t1, t2)
            print(f"  {desc}: {s}")
        except UnificationError as e:
            print(f"  {desc}: FAIL - {e}")

    # Occurs check
    print("\n--- Occurs Check ---")
    try:
        unify(TVar("a"), TArrow(TVar("a"), INT))
        print("  a ~ (a -> Int): should not succeed")
    except UnificationError as e:
        print(f"  a ~ (a -> Int): {e}")

    print("\n--- Summary ---")
    print("""
  Hindley-Milner type inference:
    1. Fresh type variables for unknown types
    2. Generate constraints by walking the AST
    3. Solve constraints via unification
    4. Generalize at let-bindings for polymorphism
    5. Instantiate polymorphic types at use sites

  Properties:
    - Principal types: infers the most general type
    - Decidable: always terminates
    - Sound: inferred types are correct
    - Complete: finds a type if one exists (for simply-typed lambda calculus)
    """)


if __name__ == "__main__":
    main()
