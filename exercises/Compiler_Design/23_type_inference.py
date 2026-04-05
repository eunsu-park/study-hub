"""
Exercises for Lesson 23: Type Inference
Topic: Compiler_Design

Implements unification and Algorithm W for a small lambda calculus.
"""

from dataclasses import dataclass
from typing import Union, Dict, Set, Optional, Tuple


# === Type representation ===

@dataclass(frozen=True)
class TVar:
    name: str
    def __repr__(self): return self.name

@dataclass(frozen=True)
class TCon:
    name: str
    def __repr__(self): return self.name

@dataclass(frozen=True)
class TFun:
    arg: 'Type'
    ret: 'Type'
    def __repr__(self): return f"({self.arg} -> {self.ret})"

Type = Union[TVar, TCon, TFun]
INT = TCon("Int")
BOOL = TCon("Bool")


# === Substitution ===

class Substitution:
    def __init__(self, mapping=None):
        self.mapping = mapping or {}

    def apply(self, ty):
        if isinstance(ty, TVar):
            return self.apply(self.mapping[ty.name]) if ty.name in self.mapping else ty
        elif isinstance(ty, TCon):
            return ty
        elif isinstance(ty, TFun):
            return TFun(self.apply(ty.arg), self.apply(ty.ret))
        return ty

    def compose(self, other):
        new = {v: self.apply(t) for v, t in other.mapping.items()}
        new.update(self.mapping)
        return Substitution(new)


def free_vars(ty):
    if isinstance(ty, TVar): return {ty.name}
    elif isinstance(ty, TCon): return set()
    elif isinstance(ty, TFun): return free_vars(ty.arg) | free_vars(ty.ret)
    return set()


# === Exercise 1: Unification ===

class UnificationError(Exception): pass

def unify(t1, t2):
    if isinstance(t1, TVar):
        if isinstance(t2, TVar) and t1.name == t2.name:
            return Substitution()
        if t1.name in free_vars(t2):
            raise UnificationError(f"Occurs check: {t1} in {t2}")
        return Substitution({t1.name: t2})
    elif isinstance(t2, TVar):
        return unify(t2, t1)
    elif isinstance(t1, TCon) and isinstance(t2, TCon):
        if t1.name == t2.name:
            return Substitution()
        raise UnificationError(f"Cannot unify {t1} with {t2}")
    elif isinstance(t1, TFun) and isinstance(t2, TFun):
        s1 = unify(t1.arg, t2.arg)
        s2 = unify(s1.apply(t1.ret), s1.apply(t2.ret))
        return s2.compose(s1)
    raise UnificationError(f"Cannot unify {t1} with {t2}")


def exercise_1():
    """Test unification algorithm."""
    print("Exercise 1: Unification")
    tests = [
        (INT, INT, "Int ~ Int"),
        (TVar("a"), INT, "a ~ Int"),
        (TFun(TVar("a"), TVar("b")), TFun(INT, BOOL), "(a->b) ~ (Int->Bool)"),
        (TVar("a"), TFun(INT, TVar("a")), "a ~ (Int->a) [occurs check]"),
        (INT, BOOL, "Int ~ Bool [type error]"),
    ]
    for t1, t2, desc in tests:
        try:
            s = unify(t1, t2)
            print(f"  {desc}")
            print(f"    Result: {s.mapping}")
        except UnificationError as e:
            print(f"  {desc}")
            print(f"    Error: {e}")
    print()


# === Expression AST ===

@dataclass
class Var: name: str
@dataclass
class Lam: param: str; body: 'Expr'
@dataclass
class App: func: 'Expr'; arg: 'Expr'
@dataclass
class Let: name: str; value: 'Expr'; body: 'Expr'
@dataclass
class Lit: value: object
Expr = Union[Var, Lam, App, Let, Lit]


# === Exercise 2: Algorithm W ===

@dataclass
class Scheme:
    vars: list
    type: Type

    def instantiate(self, fresh):
        s = {v: fresh() for v in self.vars}
        return Substitution(s).apply(self.type)


class TypeInferencer:
    def __init__(self):
        self.counter = 0

    def fresh(self):
        self.counter += 1
        return TVar(f"t{self.counter}")

    def infer(self, env, expr):
        if isinstance(expr, Lit):
            if isinstance(expr.value, int): return Substitution(), INT
            if isinstance(expr.value, bool): return Substitution(), BOOL
        elif isinstance(expr, Var):
            if expr.name not in env:
                raise TypeError(f"Unbound: {expr.name}")
            return Substitution(), env[expr.name].instantiate(self.fresh)
        elif isinstance(expr, Lam):
            tv = self.fresh()
            new_env = {**env, expr.param: Scheme([], tv)}
            s, body_ty = self.infer(new_env, expr.body)
            return s, TFun(s.apply(tv), body_ty)
        elif isinstance(expr, App):
            s1, fun_ty = self.infer(env, expr.func)
            env2 = {k: Scheme(v.vars, s1.apply(v.type)) for k, v in env.items()}
            s2, arg_ty = self.infer(env2, expr.arg)
            tv = self.fresh()
            s3 = unify(s2.apply(fun_ty), TFun(arg_ty, tv))
            return s3.compose(s2).compose(s1), s3.apply(tv)
        elif isinstance(expr, Let):
            s1, t1 = self.infer(env, expr.value)
            env1 = {k: Scheme(v.vars, s1.apply(v.type)) for k, v in env.items()}
            env_fv = set()
            for sc in env1.values():
                env_fv |= free_vars(sc.type)
            gen_vars = list(free_vars(t1) - env_fv)
            env1[expr.name] = Scheme(gen_vars, t1)
            s2, t2 = self.infer(env1, expr.body)
            return s2.compose(s1), t2
        raise TypeError(f"Unknown: {expr}")


def exercise_2():
    """Test Algorithm W on several expressions."""
    print("Exercise 2: Algorithm W")
    inf = TypeInferencer()

    tests = [
        ("identity", Lam("x", Var("x"))),
        ("const 42", Lit(42)),
        ("apply id 5", Let("id", Lam("x", Var("x")), App(Var("id"), Lit(5)))),
        ("compose", Lam("f", Lam("g", Lam("x", App(Var("f"), App(Var("g"), Var("x"))))))),
    ]

    for name, expr in tests:
        inf.counter = 0
        try:
            s, t = inf.infer({}, expr)
            result = s.apply(t)
            print(f"  {name} : {result}")
        except (TypeError, UnificationError) as e:
            print(f"  {name} : ERROR - {e}")
    print()


# === Exercise 3: Constraint-Based Inference ===

def exercise_3():
    """Generate and solve type constraints."""
    print("Exercise 3: Constraint-Based Inference")
    print()
    print("  Expression: \\f -> \\x -> f x")
    print()
    print("  Constraint generation:")
    print("    f : t1, x : t2")
    print("    f x : t3 with constraint t1 = t2 -> t3")
    print("    \\x -> f x : t2 -> t3")
    print("    \\f -> \\x -> f x : t1 -> t2 -> t3")
    print()
    print("  Constraints: {t1 = t2 -> t3}")
    print("  Solution: t1 = t2 -> t3")
    print("  Result type: (t2 -> t3) -> t2 -> t3")
    print()


# === Exercise 4: Let-Polymorphism ===

def exercise_4():
    """Show let-polymorphism vs lambda-bound."""
    print("Exercise 4: Let-Polymorphism")
    inf = TypeInferencer()

    # Works: let id = \\x -> x in (id 5, id True)
    # Simulated as: let id = \\x -> x in id 5
    expr_let = Let("id", Lam("x", Var("x")), App(Var("id"), Lit(42)))
    inf.counter = 0
    s, t = inf.infer({}, expr_let)
    print(f"  let id = \\x -> x in id 42 : {s.apply(t)}")

    # Would fail: (\\id -> id 42) (\\x -> x) if id also used with Bool
    expr_lam = App(Lam("id", App(Var("id"), Lit(42))), Lam("x", Var("x")))
    inf.counter = 0
    try:
        s, t = inf.infer({}, expr_lam)
        print(f"  (\\id -> id 42) (\\x -> x) : {s.apply(t)}")
    except (TypeError, UnificationError) as e:
        print(f"  (\\id -> id 42) (\\x -> x) : ERROR - {e}")
    print()
    print("  With let: id is polymorphic (forall a. a -> a)")
    print("  With lambda: id is monomorphic (fixed to one type)")
    print()


# === Exercise 5: Extend with List Types ===

def exercise_5():
    """Sketch: extending inference with list types."""
    print("Exercise 5: List Type Extension (design)")
    print()
    rules = [
        "nil  : forall a. [a]",
        "cons : forall a. a -> [a] -> [a]",
        "head : forall a. [a] -> a",
        "tail : forall a. [a] -> [a]",
        "map  : forall a b. (a -> b) -> [a] -> [b]",
    ]
    print("  Built-in type schemes:")
    for r in rules:
        print(f"    {r}")
    print()
    print("  Example inference:")
    print("    map (\\x -> x + 1) [1, 2, 3]")
    print("    (\\x -> x + 1) : Int -> Int")
    print("    [1, 2, 3] : [Int]")
    print("    Unify: a = Int, b = Int")
    print("    Result: [Int]")
    print()


def main():
    for i, ex in enumerate([exercise_1, exercise_2, exercise_3, exercise_4, exercise_5], 1):
        print(f"{'=' * 60}")
        print(f"Exercise {i}")
        print(f"{'=' * 60}")
        ex()


if __name__ == "__main__":
    main()
