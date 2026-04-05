# 타입 추론(Type Inference)

**이전**: [22. JIT 컴파일](./22_JIT_Compilation.md) | **다음**: [24. 클로저 변환](./24_Closure_Conversion.md)

---

타입 추론(type inference)은 프로그래머의 명시적 타입 어노테이션 없이 컴파일러가 표현식의 타입을 자동으로 추론할 수 있게 합니다. ML, Haskell, Rust(부분적)에서 사용되는 힌들리-밀너 타입 시스템(Hindley-Milner type system)은 모든 표현식의 가장 일반적인 타입을 추론하기 위한 원칙적인 프레임워크를 제공합니다. 이 레슨은 타입 추론 이론, 단일화(unification) 알고리즘, Algorithm W, 그리고 현대 언어에서 사용되는 제약 기반(constraint-based) 접근 방식을 다룹니다.

**난이도**: ⭐⭐⭐⭐

**선수 지식**: [08. 의미 분석](./08_Semantic_Analysis.md), [07. 추상 구문 트리](./07_Abstract_Syntax_Trees.md)

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 타입 검사(type checking)와 타입 추론(type inference)의 차이를 설명한다
2. 타입 표현식에 대한 단일화(unification) 알고리즘을 구현한다
3. 힌들리-밀너 타입 시스템과 그 속성을 기술한다
4. 타입 추론을 위한 Algorithm W를 구현한다
5. let-다형성(let-polymorphism)과 일반화(generalization)를 처리한다
6. 현대 컴파일러에서 사용되는 제약 기반 타입 추론을 이해한다

---

## 목차

1. [타입 시스템 개요](#1-타입-시스템-개요)
2. [타입 표현식과 치환](#2-타입-표현식과-치환)
3. [단일화](#3-단일화)
4. [힌들리-밀너 타입 시스템](#4-힌들리-밀너-타입-시스템)
5. [Algorithm W](#5-algorithm-w)
6. [Let-다형성](#6-let-다형성)
7. [제약 기반 추론](#7-제약-기반-추론)
8. [확장과 현대 언어](#8-확장과-현대-언어)
9. [요약](#9-요약)
10. [연습 문제](#10-연습-문제)
11. [참고 자료](#11-참고-자료)

---

## 1. 타입 시스템 개요

### 1.1 타입 검사 vs. 타입 추론

```
Type Checking:   Given expression e and type T, verify that e : T
Type Inference:  Given expression e, find the most general type T such that e : T
```

예시:

```python
# Type checking (Java, C): programmer provides types
int add(int x, int y) { return x + y; }

# Type inference (ML, Haskell): compiler infers types
let add x y = x + y
(* Inferred: add : int -> int -> int *)
```

### 1.2 타입 추론이 중요한 이유

- **어노테이션 부담 감소**: 프로그래머가 보일러플레이트를 덜 작성합니다
- **더 많은 다형성(polymorphism)**: 추론된 타입이 수동으로 작성한 것보다 더 일반적인 경우가 많습니다
- **안전성**: 자동 추론이 프로그래머가 놓칠 수 있는 오류를 포착합니다
- **표현력**: 강력한 제네릭 프로그래밍을 가능하게 합니다

### 1.3 타입 추론을 갖춘 언어들

| 언어 | 추론 범위 | 시스템 |
|------|----------|--------|
| Haskell | 전체 (힌들리-밀너 + 확장) | 타입 클래스를 가진 System F |
| ML/OCaml | 전체 (힌들리-밀너) | Damas-Milner |
| Rust | 지역 (함수 내부) | 양방향 + 단일화 |
| TypeScript | 구조적 + 흐름 | 흐름 민감(flow-sensitive) |
| C++ | `auto` 키워드 | 템플릿 추론 |
| Kotlin/Swift | 지역 | 양방향(bidirectional) |

---

## 2. 타입 표현식과 치환

### 2.1 타입 언어

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

### 2.2 치환(Substitutions)

**치환(substitution)**은 타입 변수에서 타입으로의 매핑입니다:

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

### 2.3 자유 변수(Free Variables)

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

## 3. 단일화

### 3.1 단일화 문제

두 타입 `t1`과 `t2`가 주어지면, `S(t1) = S(t2)`를 만족하는 치환 `S`를 찾거나, 그러한 치환이 존재하지 않음을 보고합니다.

```
Unify(Int, Int) = {}                    -- trivial
Unify(a, Int) = {a -> Int}             -- bind variable
Unify(a -> b, Int -> Bool) = {a -> Int, b -> Bool}
Unify(Int, Bool) = FAIL                 -- conflict
Unify(a, a -> Int) = FAIL              -- occurs check (infinite type)
```

### 3.2 구현

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

### 3.3 순환 검사(Occurs Check)

**순환 검사(occurs check)**는 무한 타입을 방지합니다:

```
Unify(a, [a])  -->  a = [a] = [[a]] = [[[a]]] = ...  -- infinite!

Without occurs check: unification succeeds but creates a cyclic type.
With occurs check: unification fails, preventing unsoundness.
```

---

## 4. 힌들리-밀너 타입 시스템

### 4.1 언어

HM 시스템은 작지만 강력한 핵심 언어를 다룹니다:

```
e ::= x                    -- variable
    | \x -> e              -- lambda abstraction
    | e1 e2                -- application
    | let x = e1 in e2     -- let binding
    | literal               -- integer, boolean, etc.
```

### 4.2 타입 스킴(Type Schemes, 다형성)

**타입 스킴(type scheme)**(또는 다형 타입(polytype))은 타입 변수를 한정합니다:

```
sigma ::= forall a1 ... an . tau

Examples:
  id    : forall a . a -> a
  const : forall a b . a -> b -> a
  map   : forall a b . (a -> b) -> [a] -> [b]
```

### 4.3 인스턴스화(Instantiation)와 일반화(Generalization)

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

### 4.4 주요 타입(Principal Types)

HM의 핵심 속성: 타입을 가질 수 있는 모든 표현식은 **주요 타입(principal type)**을 가집니다 -- 치환을 통해 다른 모든 유효한 타입을 얻을 수 있는 가장 일반적인 타입입니다.

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

### 5.1 개요

Algorithm W (Damas와 Milner, 1982)는 표현식의 주요 타입을 추론합니다:

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

### 5.2 헬퍼: 환경에 치환 적용

```python
def apply_env(subst, env):
    """Apply a substitution to all type schemes in an environment."""
    return {name: Scheme(s.vars, subst.apply(s.type)) for name, s in env.items()}
```

### 5.3 전체 예제

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

## 6. Let-다형성

### 6.1 핵심 통찰

HM에서 `let` 바인딩은 **일반화된**(다형적) 타입을 얻지만, 람다 매개변수는 그렇지 않습니다:

```
-- Let-bound: polymorphic
let id = \x -> x in (id 42, id True)
-- id : forall a . a -> a   (OK: used at Int and Bool)

-- Lambda-bound: monomorphic
(\id -> (id 42, id True)) (\x -> x)
-- FAILS: id would need to be both Int -> Int and Bool -> Bool
```

### 6.2 이 제한이 필요한 이유

람다 바인딩 변수에 대한 무제한 다형성은 타입 추론을 결정 불가능하게 만듭니다(System F 타입 추론과 동등). let-다형성 제한은 추론을 결정 가능하고 완전하게 유지합니다.

### 6.3 값 제한(Value Restriction)

가변 참조(mutable reference)가 있을 때 순진한 let-다형성은 건전하지 않습니다:

```ml
(* Dangerous in ML without value restriction *)
let r = ref []     (* r : forall a . ref (a list) ??? *)
r := [1]           (* r : ref (int list) *)
let s = !r         (* s : forall a . a list ??? *)
let _ = hd s + 1   (* treats s as int list... but could be anything! *)
```

**값 제한(value restriction)**(Wright, 1995): 구문적으로 값(syntactic value)인 `let` 바인딩 표현식만 일반화합니다(계산이 아닌).

---

## 7. 제약 기반 추론

### 7.1 동기

즉시 단일화하는 대신(Algorithm W), 제약 기반 추론은 먼저 타입 제약을 수집한 후 이를 풀이합니다:

```
Phase 1: Generate constraints from the AST
Phase 2: Solve constraints using unification
```

이 분리는 더 나은 오류 메시지와 모듈식 타입 검사를 가능하게 합니다.

### 7.2 제약 생성

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

### 7.3 제약 풀이

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

## 8. 확장과 현대 언어

### 8.1 타입 클래스(Type Classes, Haskell)

타입 클래스는 제약된 다형성(constrained polymorphism)을 추가합니다:

```haskell
-- Without type classes: (+) only works on one type
add :: Int -> Int -> Int

-- With type classes: (+) works on any Num type
add :: Num a => a -> a -> a
```

### 8.2 양방향 타입 추론(Bidirectional Type Inference)

현대 언어(Scala, Rust, Swift)는 양방향 추론을 사용합니다: 정보가 "위로"(합성, synthesis)와 "아래로"(검사, checking) 모두 흐릅니다:

```
Synthesis: infer type from expression bottom-up
Checking:  push expected type into expression top-down

Example (Rust):
let x: Vec<i32> = vec![1, 2, 3];
// vec! knows to create Vec<i32> because expected type flows down
```

### 8.3 행 다형성(Row Polymorphism)과 구조적 타입(Structural Types)

```
-- Record with row polymorphism
getName : { name : String | r } -> String
getName record = record.name

-- Works for any record with at least a name field
```

---

## 9. 요약

- **타입 추론**은 어노테이션 없이 자동으로 타입을 추론합니다
- **단일화**는 두 타입을 같게 만드는 치환을 찾습니다
- **힌들리-밀너** 시스템은 주요 타입과 함께 완전하고 결정 가능한 타입 추론을 제공합니다
- **Algorithm W**는 단일화와 일반화를 사용하여 HM 추론을 구현합니다
- **Let-다형성**은 let 바인딩 변수에는 다형적 타입을, 람다 바인딩 변수에는 단형적 타입을 허용합니다
- **제약 기반 추론**은 더 나은 모듈성을 위해 제약 생성과 풀이를 분리합니다
- 현대 언어는 타입 클래스, 양방향 검사, 행 다형성으로 HM을 확장합니다

---

## 10. 연습 문제

1. **단일화**: 단일화 알고리즘을 구현하고 타입 표현식 쌍에 대해 테스트하세요.

2. **Algorithm W**: let 바인딩이 있는 작은 람다 계산법(lambda calculus)에 Algorithm W를 구현하고, 여러 테스트 표현식에 대해 타입을 추론하세요.

3. **제약 생성**: 제약 기반 추론을 구현하고 동일한 표현식에 대해 Algorithm W와 비교하세요.

4. **Let-다형성**: `let`으로는 올바르게 타입이 지정되지만 람다로는 실패하는 표현식을 보이고, 그 이유를 설명하세요.

5. **타입 시스템 확장**: 리스트 타입과 `cons`, `head`, `tail`, `map`에 대한 추론 규칙을 추가하세요.

---

## 11. 참고 자료

1. Damas, L., Milner, R. (1982). "Principal Type-Schemes for Functional Programs." *POPL*.
2. Milner, R. (1978). "A Theory of Type Polymorphism in Programming." *JCSS*, 17(3).
3. Robinson, J. A. (1965). "A Machine-Oriented Logic Based on the Resolution Principle." *JACM*, 12(1).
4. Pierce, B. C. (2002). *Types and Programming Languages*. MIT Press. Chapters 22-23.
5. Odersky, M., Sulzmann, M., Wehr, M. (1999). "Type Inference with Constrained Types." *TAPOS*, 5(1).

---

**이전**: [22. JIT 컴파일](./22_JIT_Compilation.md) | **다음**: [24. 클로저 변환](./24_Closure_Conversion.md)
