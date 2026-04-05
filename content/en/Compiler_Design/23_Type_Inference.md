# Type Inference

**Previous**: [22. JIT Compilation](./22_JIT_Compilation.md) | **Next**: [24. Closure Conversion](./24_Closure_Conversion.md)

---

Type inference allows compilers to automatically deduce the types of expressions without requiring explicit type annotations from the programmer. The Hindley-Milner type system, used in ML, Haskell, and Rust (partially), provides a principled framework for inferring the most general type of every expression. This lesson covers the theory of type inference, the unification algorithm, Algorithm W, and constraint-based approaches used in modern languages.

**Difficulty**: ⭐⭐⭐⭐

**Prerequisites**: [08. Semantic Analysis](./08_Semantic_Analysis.md), [07. Abstract Syntax Trees](./07_Abstract_Syntax_Trees.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain the difference between type checking and type inference
2. Implement the unification algorithm for type expressions
3. Describe the Hindley-Milner type system and its properties
4. Implement Algorithm W for type inference
5. Handle let-polymorphism and generalization
6. Understand constraint-based type inference used in modern compilers

---

## Table of Contents

1. [Type Systems Overview](#1-type-systems-overview)
2. [Type Expressions and Substitutions](#2-type-expressions-and-substitutions)
3. [Unification](#3-unification)
4. [Hindley-Milner Type System](#4-hindley-milner-type-system)
5. [Algorithm W](#5-algorithm-w)
6. [Let-Polymorphism](#6-let-polymorphism)
7. [Constraint-Based Inference](#7-constraint-based-inference)
8. [Extensions and Modern Languages](#8-extensions-and-modern-languages)
9. [Summary](#9-summary)
10. [Exercises](#10-exercises)
11. [References](#11-references)

---

## 1. Type Systems Overview

### 1.1 Type Checking vs. Type Inference

```
Type Checking:   Given expression e and type T, verify that e : T
Type Inference:  Given expression e, find the most general type T such that e : T
```

Examples:

```python
# Type checking (Java, C): programmer provides types
int add(int x, int y) { return x + y; }

# Type inference (ML, Haskell): compiler infers types
let add x y = x + y
(* Inferred: add : int -> int -> int *)
```

### 1.2 Why Type Inference Matters

- **Less annotation burden**: Programmers write less boilerplate
- **More polymorphism**: Inferred types are often more general than hand-written ones
- **Safety**: Automatic inference catches errors a programmer might miss
- **Expressiveness**: Enables powerful generic programming

### 1.3 Languages with Type Inference

| Language | Inference Scope | System |
|----------|----------------|--------|
| Haskell | Full (Hindley-Milner + extensions) | System F with type classes |
| ML/OCaml | Full (Hindley-Milner) | Damas-Milner |
| Rust | Local (within functions) | Bidirectional + unification |
| TypeScript | Structural + flow | Flow-sensitive |
| C++ | `auto` keyword | Template deduction |
| Kotlin/Swift | Local | Bidirectional |

---

## 2. Type Expressions and Substitutions

### 2.1 Type Language

```python
from dataclasses import dataclass
from typing import Union

@dataclass(frozen=True)
class TVar:
    """Type variable: a, b, t1, t2, ..."""
    name: str

    def __repr__(self):
        return self.name

@dataclass(frozen=True)
class TCon:
    """Type constructor: Int, Bool, String, ..."""
    name: str

    def __repr__(self):
        return self.name

@dataclass(frozen=True)
class TFun:
    """Function type: a -> b"""
    arg: 'Type'
    ret: 'Type'

    def __repr__(self):
        return f"({self.arg} -> {self.ret})"

@dataclass(frozen=True)
class TList:
    """List type: [a]"""
    elem: 'Type'

    def __repr__(self):
        return f"[{self.elem}]"

Type = Union[TVar, TCon, TFun, TList]

# Examples:
# Int
INT = TCon("Int")
BOOL = TCon("Bool")
# a -> b
a, b = TVar("a"), TVar("b")
fun_ab = TFun(a, b)
# Int -> Int
fun_ii = TFun(INT, INT)
```

### 2.2 Substitutions

A **substitution** is a mapping from type variables to types:

```python
class Substitution:
    """A mapping from type variables to types."""

    def __init__(self, mapping=None):
        self.mapping = mapping or {}

    def apply(self, ty):
        """Apply this substitution to a type."""
        if isinstance(ty, TVar):
            if ty.name in self.mapping:
                return self.apply(self.mapping[ty.name])
            return ty
        elif isinstance(ty, TCon):
            return ty
        elif isinstance(ty, TFun):
            return TFun(self.apply(ty.arg), self.apply(ty.ret))
        elif isinstance(ty, TList):
            return TList(self.apply(ty.elem))
        return ty

    def compose(self, other):
        """Compose two substitutions: self after other."""
        # Apply self to all bindings in other, then add self's bindings
        new_mapping = {v: self.apply(t) for v, t in other.mapping.items()}
        new_mapping.update(self.mapping)
        return Substitution(new_mapping)

    def __repr__(self):
        return str(self.mapping)
```

### 2.3 Free Variables

```python
def free_vars(ty):
    """Return the set of free type variables in a type."""
    if isinstance(ty, TVar):
        return {ty.name}
    elif isinstance(ty, TCon):
        return set()
    elif isinstance(ty, TFun):
        return free_vars(ty.arg) | free_vars(ty.ret)
    elif isinstance(ty, TList):
        return free_vars(ty.elem)
    return set()
```

---

## 3. Unification

### 3.1 The Unification Problem

Given two types `t1` and `t2`, find a substitution `S` such that `S(t1) = S(t2)`, or report that no such substitution exists.

```
Unify(Int, Int) = {}                    -- trivial
Unify(a, Int) = {a -> Int}             -- bind variable
Unify(a -> b, Int -> Bool) = {a -> Int, b -> Bool}
Unify(Int, Bool) = FAIL                 -- conflict
Unify(a, a -> Int) = FAIL              -- occurs check (infinite type)
```

### 3.2 Implementation

```python
class UnificationError(Exception):
    pass

def unify(t1, t2):
    """
    Unify two types, returning the most general unifier (MGU).
    Raises UnificationError if types are incompatible.
    """
    if isinstance(t1, TVar):
        return unify_var(t1, t2)
    elif isinstance(t2, TVar):
        return unify_var(t2, t1)
    elif isinstance(t1, TCon) and isinstance(t2, TCon):
        if t1.name == t2.name:
            return Substitution()
        raise UnificationError(f"Cannot unify {t1} with {t2}")
    elif isinstance(t1, TFun) and isinstance(t2, TFun):
        s1 = unify(t1.arg, t2.arg)
        s2 = unify(s1.apply(t1.ret), s1.apply(t2.ret))
        return s2.compose(s1)
    elif isinstance(t1, TList) and isinstance(t2, TList):
        return unify(t1.elem, t2.elem)
    else:
        raise UnificationError(f"Cannot unify {t1} with {t2}")


def unify_var(var, ty):
    """Unify a type variable with a type."""
    if isinstance(ty, TVar) and var.name == ty.name:
        return Substitution()  # Same variable
    elif var.name in free_vars(ty):
        raise UnificationError(f"Occurs check: {var} in {ty}")
    else:
        return Substitution({var.name: ty})
```

### 3.3 Occurs Check

The **occurs check** prevents infinite types:

```
Unify(a, [a])  -->  a = [a] = [[a]] = [[[a]]] = ...  -- infinite!

Without occurs check: unification succeeds but creates a cyclic type.
With occurs check: unification fails, preventing unsoundness.
```

---

## 4. Hindley-Milner Type System

### 4.1 Language

The HM system covers a small but powerful core language:

```
e ::= x                    -- variable
    | \x -> e              -- lambda abstraction
    | e1 e2                -- application
    | let x = e1 in e2     -- let binding
    | literal               -- integer, boolean, etc.
```

### 4.2 Type Schemes (Polymorphism)

A **type scheme** (or polytype) quantifies over type variables:

```
sigma ::= forall a1 ... an . tau

Examples:
  id    : forall a . a -> a
  const : forall a b . a -> b -> a
  map   : forall a b . (a -> b) -> [a] -> [b]
```

### 4.3 Instantiation and Generalization

```python
@dataclass
class Scheme:
    """Type scheme: forall vars . type"""
    vars: list      # quantified type variables
    type: Type

    def instantiate(self, fresh_var_gen):
        """Replace quantified variables with fresh type variables."""
        subst = {}
        for v in self.vars:
            subst[v] = fresh_var_gen()
        return Substitution(subst).apply(self.type)


def generalize(env, ty):
    """
    Generalize a type by quantifying over free variables
    not in the environment.
    """
    env_free = set()
    for scheme in env.values():
        env_free |= free_vars_scheme(scheme)

    gen_vars = list(free_vars(ty) - env_free)
    return Scheme(gen_vars, ty)
```

### 4.4 Principal Types

A key property of HM: every typeable expression has a **principal type** -- the most general type from which all other valid types can be obtained by substitution.

```
id = \x -> x
Principal type: forall a . a -> a

-- All these are instances:
-- Int -> Int
-- Bool -> Bool
-- (Int -> Bool) -> (Int -> Bool)
```

---

## 5. Algorithm W

### 5.1 Overview

Algorithm W (Damas and Milner, 1982) infers the principal type of an expression:

```python
class TypeInferencer:
    """Hindley-Milner type inference using Algorithm W."""

    def __init__(self):
        self.counter = 0

    def fresh_var(self):
        """Generate a fresh type variable."""
        self.counter += 1
        return TVar(f"t{self.counter}")

    def infer(self, env, expr):
        """
        Infer the type of expr in environment env.
        Returns (substitution, type).
        """
        if isinstance(expr, Var):
            return self.infer_var(env, expr)
        elif isinstance(expr, Lam):
            return self.infer_lambda(env, expr)
        elif isinstance(expr, App):
            return self.infer_app(env, expr)
        elif isinstance(expr, Let):
            return self.infer_let(env, expr)
        elif isinstance(expr, Lit):
            return self.infer_lit(expr)
        else:
            raise TypeError(f"Unknown expression: {expr}")

    def infer_var(self, env, expr):
        """Variable: look up in environment and instantiate."""
        if expr.name not in env:
            raise TypeError(f"Unbound variable: {expr.name}")
        scheme = env[expr.name]
        ty = scheme.instantiate(self.fresh_var)
        return Substitution(), ty

    def infer_lambda(self, env, expr):
        """Lambda: \\x -> body"""
        arg_type = self.fresh_var()
        # Extend environment with x : arg_type (monomorphic)
        new_env = dict(env)
        new_env[expr.param] = Scheme([], arg_type)
        s, body_type = self.infer(new_env, expr.body)
        return s, TFun(s.apply(arg_type), body_type)

    def infer_app(self, env, expr):
        """Application: f x"""
        s1, fun_type = self.infer(env, expr.func)
        s2, arg_type = self.infer(apply_env(s1, env), expr.arg)
        result_type = self.fresh_var()
        s3 = unify(s2.apply(fun_type), TFun(arg_type, result_type))
        return s3.compose(s2).compose(s1), s3.apply(result_type)

    def infer_let(self, env, expr):
        """Let: let x = e1 in e2"""
        # Infer type of e1
        s1, t1 = self.infer(env, expr.value)
        # Generalize t1 in the updated environment
        env1 = apply_env(s1, env)
        scheme = generalize(env1, t1)
        # Extend environment with generalized type
        env1[expr.name] = scheme
        # Infer type of e2
        s2, t2 = self.infer(env1, expr.body)
        return s2.compose(s1), t2

    def infer_lit(self, expr):
        """Literal: integer, boolean, etc."""
        if isinstance(expr.value, int):
            return Substitution(), TCon("Int")
        elif isinstance(expr.value, bool):
            return Substitution(), TCon("Bool")
        elif isinstance(expr.value, str):
            return Substitution(), TCon("String")
        else:
            raise TypeError(f"Unknown literal type: {expr.value}")
```

### 5.2 Helper: Apply Substitution to Environment

```python
def apply_env(subst, env):
    """Apply a substitution to all type schemes in an environment."""
    return {name: Scheme(s.vars, subst.apply(s.type)) for name, s in env.items()}
```

### 5.3 Complete Example

```python
# Infer: let id = \x -> x in id 42
expr = Let("id",
           Lam("x", Var("x")),
           App(Var("id"), Lit(42)))

inferencer = TypeInferencer()
s, t = inferencer.infer({}, expr)
print(s.apply(t))
# Output: Int

# Infer: \f -> \x -> f (f x)
expr = Lam("f", Lam("x", App(Var("f"), App(Var("f"), Var("x")))))
s, t = inferencer.infer({}, expr)
print(s.apply(t))
# Output: (t1 -> t1) -> t1 -> t1
```

---

## 6. Let-Polymorphism

### 6.1 The Key Insight

In HM, `let` bindings get **generalized** (polymorphic) types, but lambda parameters do not:

```
-- Let-bound: polymorphic
let id = \x -> x in (id 42, id True)
-- id : forall a . a -> a   (OK: used at Int and Bool)

-- Lambda-bound: monomorphic
(\id -> (id 42, id True)) (\x -> x)
-- FAILS: id would need to be both Int -> Int and Bool -> Bool
```

### 6.2 Why This Restriction?

Unrestricted polymorphism for lambda-bound variables makes type inference undecidable (equivalent to System F type inference). The let-polymorphism restriction keeps inference decidable and complete.

### 6.3 Value Restriction

In the presence of mutable references, naive let-polymorphism is unsound:

```ml
(* Dangerous in ML without value restriction *)
let r = ref []     (* r : forall a . ref (a list) ??? *)
r := [1]           (* r : ref (int list) *)
let s = !r         (* s : forall a . a list ??? *)
let _ = hd s + 1   (* treats s as int list... but could be anything! *)
```

The **value restriction** (Wright, 1995): only generalize `let`-bound expressions that are syntactic values (not computations).

---

## 7. Constraint-Based Inference

### 7.1 Motivation

Instead of unifying eagerly (Algorithm W), constraint-based inference collects type constraints first, then solves them:

```
Phase 1: Generate constraints from the AST
Phase 2: Solve constraints using unification
```

This separation enables better error messages and modular type checking.

### 7.2 Constraint Generation

```python
def generate_constraints(expr, env):
    """
    Generate type constraints from an expression.
    Returns (type, constraints) where constraints is a list of (t1 == t2).
    """
    if isinstance(expr, Var):
        ty = env[expr.name].instantiate(fresh_var)
        return ty, []

    elif isinstance(expr, Lam):
        arg_ty = fresh_var()
        new_env = {**env, expr.param: Scheme([], arg_ty)}
        body_ty, cs = generate_constraints(expr.body, new_env)
        return TFun(arg_ty, body_ty), cs

    elif isinstance(expr, App):
        fun_ty, cs1 = generate_constraints(expr.func, env)
        arg_ty, cs2 = generate_constraints(expr.arg, env)
        result_ty = fresh_var()
        constraint = (fun_ty, TFun(arg_ty, result_ty))
        return result_ty, cs1 + cs2 + [constraint]

    elif isinstance(expr, Lit):
        if isinstance(expr.value, int):
            return TCon("Int"), []
        elif isinstance(expr.value, bool):
            return TCon("Bool"), []
```

### 7.3 Constraint Solving

```python
def solve_constraints(constraints):
    """Solve a list of type equality constraints."""
    subst = Substitution()

    for t1, t2 in constraints:
        t1 = subst.apply(t1)
        t2 = subst.apply(t2)
        s = unify(t1, t2)
        subst = s.compose(subst)

    return subst
```

---

## 8. Extensions and Modern Languages

### 8.1 Type Classes (Haskell)

Type classes add constrained polymorphism:

```haskell
-- Without type classes: (+) only works on one type
add :: Int -> Int -> Int

-- With type classes: (+) works on any Num type
add :: Num a => a -> a -> a
```

### 8.2 Bidirectional Type Inference

Modern languages (Scala, Rust, Swift) use bidirectional inference: information flows both "up" (synthesis) and "down" (checking):

```
Synthesis: infer type from expression bottom-up
Checking:  push expected type into expression top-down

Example (Rust):
let x: Vec<i32> = vec![1, 2, 3];
// vec! knows to create Vec<i32> because expected type flows down
```

### 8.3 Row Polymorphism and Structural Types

```
-- Record with row polymorphism
getName : { name : String | r } -> String
getName record = record.name

-- Works for any record with at least a name field
```

---

## 9. Summary

- **Type inference** automatically deduces types without annotations
- **Unification** finds substitutions that make two types equal
- The **Hindley-Milner** system provides complete, decidable type inference with principal types
- **Algorithm W** implements HM inference using unification and generalization
- **Let-polymorphism** allows polymorphic types for let-bound but not lambda-bound variables
- **Constraint-based inference** separates constraint generation from solving for better modularity
- Modern languages extend HM with type classes, bidirectional checking, and row polymorphism

---

## 10. Exercises

1. **Unification**: Implement the unification algorithm and test it on pairs of type expressions.

2. **Algorithm W**: Implement Algorithm W for a small lambda calculus with let bindings, and infer types for several test expressions.

3. **Constraint generation**: Implement constraint-based inference and compare with Algorithm W on the same expressions.

4. **Let-polymorphism**: Show an expression that types correctly with `let` but fails with lambda, and explain why.

5. **Extend the type system**: Add list types and inference rules for `cons`, `head`, `tail`, and `map`.

---

## 11. References

1. Damas, L., Milner, R. (1982). "Principal Type-Schemes for Functional Programs." *POPL*.
2. Milner, R. (1978). "A Theory of Type Polymorphism in Programming." *JCSS*, 17(3).
3. Robinson, J. A. (1965). "A Machine-Oriented Logic Based on the Resolution Principle." *JACM*, 12(1).
4. Pierce, B. C. (2002). *Types and Programming Languages*. MIT Press. Chapters 22-23.
5. Odersky, M., Sulzmann, M., Wehr, M. (1999). "Type Inference with Constrained Types." *TAPOS*, 5(1).

---

**Previous**: [22. JIT Compilation](./22_JIT_Compilation.md) | **Next**: [24. Closure Conversion](./24_Closure_Conversion.md)
