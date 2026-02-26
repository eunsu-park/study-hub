# 레슨 8: 의미 분석(Semantic Analysis)

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 컴파일 파이프라인에서 의미 분석(semantic analysis)의 역할을 **설명**할 수 있다
2. S-속성(S-attributed) 문법과 L-속성(L-attributed) 문법을 사용하는 구문 지향 번역(syntax-directed translation)을 **구별**할 수 있다
3. 해시 테이블을 사용하여 적절한 스코프 관리(scope management)가 있는 심벌 테이블(symbol table)을 **구현**할 수 있다
4. 정적/동적, 강/약 타입 시스템(type system)을 **설명**할 수 있다
5. 표현식, 문장, 함수에 대한 타입 검사(type checking) 규칙을 **구현**할 수 있다
6. 힌들리-밀너(Hindley-Milner) 타입 추론(type inference)의 기초를 **이해**할 수 있다
7. 타입 호환성(type compatibility), 강제 변환(coercion), 오버로딩 해결(overloading resolution), 선언 처리를 **다룰** 수 있다
8. 명확한 메시지와 소스 위치를 포함한 의미 오류를 **보고**할 수 있다

---

## 1. 소개: 의미 분석의 역할

파서(parser)가 AST를 생성한 후, 컴파일러는 프로그램이 **의미 있는지** 검증해야 합니다 -- 단순히 구문적으로 올바른 것뿐만 아니라 의미적으로도 유효해야 합니다. 의미 분석은 구문과 코드 생성(code generation) 사이의 간극을 메워줍니다.

### 1.1 파싱이 확인할 수 없는 것들

파싱은 프로그램이 문법을 따르는지 확인하지만, 많은 중요한 속성들은 문법의 범위를 벗어납니다:

| 속성 | 위반 예시 | 감지 주체 |
|------|---------|----------|
| 변수 사용 전 선언 | `x = y + 1;` (y 미선언) | 의미 분석 |
| 타입 호환성 | `"hello" + 42` (문자열 + 정수) | 타입 검사기 |
| 함수 인자 수 | `f(1, 2)` (f가 3개 인자를 받을 때) | 타입 검사기 |
| 반환 타입 | `fn foo() -> int { return "hi"; }` | 타입 검사기 |
| 루프 외부의 break | 전역 스코프에서 `break;` | 의미 분석 |
| 중복 선언 | 같은 스코프에서 `let x = 1; let x = 2;` | 심벌 테이블 |
| 접근 제어 | private 필드 읽기 | 의미 분석 |

### 1.2 의미 분석 파이프라인

```
            AST (파서로부터)
             │
             ▼
    ┌────────────────────┐
    │  이름 해결         │  심벌 테이블 구축,
    │  (스코프 분석)      │  식별자 해결
    └────────┬───────────┘
             │
             ▼
    ┌────────────────────┐
    │  타입 검사 /        │  타입 정확성 검증,
    │  타입 추론          │  AST에 타입 주석 추가
    └────────┬───────────┘
             │
             ▼
    ┌────────────────────┐
    │  기타 검사          │  제어 흐름, 도달 가능성,
    │                    │  초기화 등
    └────────┬───────────┘
             │
             ▼
        주석이 달린 AST
    (IR 생성 준비 완료)
```

---

## 2. 속성 문법(Attribute Grammars)

### 2.1 개념

**속성 문법(attribute grammar)**은 문맥 자유 문법(context-free grammar)에 문법 기호(grammar symbol)에 부착된 **속성(attribute)**과 속성을 계산하는 방법을 정의하는 **의미 규칙(semantic rule)**을 추가하여 확장합니다. 이 형식은 의미 분석을 정의하는 엄밀한 방법을 제공합니다.

각 문법 기호 $X$는 다음을 가질 수 있습니다:
- **합성 속성(Synthesized attributes)**: $X$의 자식들의 속성으로 계산됩니다 (정보가 위로 흐름)
- **상속 속성(Inherited attributes)**: $X$의 부모나 형제의 속성으로 계산됩니다 (정보가 아래로 흐름)

### 2.2 S-속성 문법(S-Attributed Grammars)

**S-속성 문법(S-attributed grammar)**은 **합성** 속성만 사용합니다. 정보는 엄격하게 아래에서 위로 흐릅니다. 이것이 가장 단순하며 단일 후위 순회(post-order traversal)로 평가할 수 있습니다.

**예시: 합성된 `val` 속성을 가진 계산기**

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

### 2.3 L-속성 문법(L-Attributed Grammars)

**L-속성 문법(L-attributed grammar)**은 상속 속성을 허용하지만, 제약이 있습니다: 생성 규칙 $A \to X_1 X_2 \cdots X_n$에서 $X_i$의 상속 속성은 다음에만 의존할 수 있습니다:

1. $A$의 상속 속성 (부모)
2. $X_1, X_2, \ldots, X_{i-1}$의 속성 (합성 또는 상속) (왼쪽 형제들)

이로써 속성을 단일 **왼쪽에서 오른쪽** 순회로 계산할 수 있습니다.

**예시: 상속된 타입 속성을 가진 타입 선언**

$$
\begin{aligned}
D &\to T\ L & \quad L.\text{type} &= T.\text{type} \\
T &\to \textbf{int} & \quad T.\text{type} &= \text{integer} \\
T &\to \textbf{float} & \quad T.\text{type} &= \text{float} \\
L &\to L_1 ,\ \textbf{id} & \quad L_1.\text{type} &= L.\text{type}; \quad \text{addtype}(\textbf{id}, L.\text{type}) \\
L &\to \textbf{id} & \quad & \text{addtype}(\textbf{id}, L.\text{type})
\end{aligned}
$$

여기서 `L.type`은 선언된 타입을 각 식별자에게 전달하면서 왼쪽에서 오른쪽으로 흐르는 **상속** 속성입니다.

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

### 2.4 비교

| 속성 | S-속성 | L-속성 |
|------|--------|--------|
| 속성 방향 | 아래에서 위로만 | 아래에서 위로 + 제한된 위에서 아래로 |
| 평가 순서 | 후위 순서 | 왼쪽-오른쪽 깊이 우선 |
| 복잡도 | 더 단순 | 더 유연 |
| LR 파서 | 직접 지원 | 적응 필요 |
| LL 파서 | 지원됨 | 직접 지원 |
| 사용 사례 | 표현식 평가, 코드 생성 | 타입 전파, 스코프 관리 |

---

## 3. 심벌 테이블(Symbol Tables)

### 3.1 목적

**심벌 테이블(symbol table)**은 이름(식별자)을 해당 속성(타입, 스코프, 메모리 위치 등)에 매핑하는 자료구조입니다. 의미 분석의 핵심 구조입니다.

컴파일러가 다음과 같은 경우를 만날 때마다:
- **선언** (`let x: int = 5`): 새 항목을 추가
- **참조** (`print(x)`): 기존 항목을 조회

### 3.2 스코프 관리(Scope Management)

대부분의 언어는 **중첩 스코프(nested scopes)**를 지원합니다: 블록은 내부 블록을 포함할 수 있으며, 각각 외부 스코프의 선언을 가릴 수 있는 자체 선언을 가집니다.

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

### 3.3 구현 전략

**전략 1: 연쇄 해시 테이블 (스코프 스택)**

각 스코프는 자체 해시 테이블을 가집니다. 조회는 가장 안쪽 스코프에서 바깥쪽으로 탐색합니다.

```
스코프 스택:

    ┌──────────────┐
    │ Scope 3      │ ← 현재 스코프
    │ z -> int     │
    └──────┬───────┘
           │ parent
    ┌──────┴───────┐
    │ Scope 2      │
    │ y -> int     │
    │ x -> int     │  (Scope 1의 x를 가림)
    └──────┬───────┘
           │ parent
    ┌──────┴───────┐
    │ Scope 1      │
    │ x -> int     │
    └──────────────┘
```

**전략 2: 스코프 마커가 있는 단일 해시 테이블**

하나의 해시 테이블을 사용하되 스코프 마커의 스택을 유지합니다. 스코프를 종료할 때, 마지막 마커 이후에 추가된 모든 항목을 제거합니다.

### 3.4 완전한 구현

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

**예상 출력:**

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

## 4. 타입 시스템(Type Systems)

### 4.1 정적 타입과 동적 타입(Static vs Dynamic Typing)

**정적 타입(static typing)**은 **컴파일 시간**에 타입을 검사합니다. 모든 표현식은 프로그램이 실행되기 전에 알려진 타입을 가집니다.

| 언어 | 타입 방식 |
|------|---------|
| C, C++, Java, Rust, Go, Haskell | 정적 |
| Python, JavaScript, Ruby, Lisp | 동적 |
| TypeScript, Mypy, Dart | 점진적 타입(gradual typing) (정적 + 동적) |

**동적 타입(dynamic typing)**은 **런타임**에 타입을 검사합니다. 변수는 어떤 타입의 값도 담을 수 있습니다.

```python
# Static typing (pseudo-code)
let x: int = 5
x = "hello"         # COMPILE ERROR: type mismatch

# Dynamic typing (Python)
x = 5
x = "hello"         # OK at runtime
```

### 4.2 강한 타입과 약한 타입(Strong vs Weak Typing)

**강한 타입(strong typing)**은 암묵적 타입 변환이 제한됨을 의미합니다. 호환되지 않는 타입에 대한 연산은 오류를 생성합니다.

**약한 타입(weak typing)**은 암묵적 변환(강제 변환)이 자유롭게 발생함을 의미합니다.

| | 강한 타입 | 약한 타입 |
|---|---|---|
| **정적** | Rust, Haskell, Java | C, C++ |
| **동적** | Python, Ruby | JavaScript, PHP |

**약한 타입의 예시 (JavaScript):**

```javascript
"5" + 3       // "53" (number coerced to string)
"5" - 3       // 2    (string coerced to number)
true + true   // 2    (booleans coerced to numbers)
```

**강한 타입의 예시 (Python):**

```python
"5" + 3       # TypeError: can only concatenate str to str
True + True   # 2 (bool is a subclass of int -- deliberate design)
```

### 4.3 타입 시스템 설계 스펙트럼

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

## 5. 타입 검사(Type Checking)

### 5.1 타입 검사 규칙

타입 검사는 연산이 호환 가능한 타입의 피연산자에 적용되는지 검증합니다. 타입 검사 규칙을 **판단(judgment)**으로 표현합니다:

$$\Gamma \vdash e : \tau$$

"타입 환경(type environment) $\Gamma$ 하에서, 표현식 $e$는 타입 $\tau$를 가진다"로 읽습니다.

**핵심 타입 규칙 (반형식적 표현):**

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

### 5.2 타입 검사기 구현

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

**예상 출력:**

```
Type check FAILED!

Found 3 error(s):
  TypeError: Operator '+' not supported for types 'int' and 'string'
  TypeError: Argument 2: expected 'int', got 'string'
  TypeError: If condition must be bool, got 'int'
```

---

## 6. 타입 추론(Type Inference)

### 6.1 지역 타입 추론(Local Type Inference)

가장 단순한 형태의 타입 추론: 초기화 식으로부터 변수의 타입을 추론합니다.

```python
# The type checker already supports this:
let x = 42;           # inferred as int
let y = 3.14;         # inferred as float
let z = x + y;        # inferred as float (int + float -> float)
let msg = "hello";    # inferred as string
```

이것은 TypeScript, Kotlin, Rust, Go가 변수 선언에 사용하는 방식입니다.

### 6.2 힌들리-밀너 타입 추론(Hindley-Milner Type Inference)

**힌들리-밀너(HM)** 타입 추론은 더 강력합니다: 어노테이션 없이 함수 매개변수와 반환 타입의 타입을 추론할 수 있습니다. Haskell, OCaml, F#, 그리고 (제한된 형태로) Rust에서 사용됩니다.

**핵심 아이디어:** **타입 변수(type variables)**와 **단일화(unification)**를 사용하여 타입을 결정합니다.

$$
\frac{\Gamma, x:\alpha \vdash e : \tau}{\Gamma \vdash \lambda x.e : \alpha \to \tau}
$$

매개변수 $x$의 타입을 모를 경우, 신선한 타입 변수 $\alpha$를 할당하고 단일화가 구체적인 타입을 결정하게 합니다.

### 6.3 단일화 알고리즘(Unification Algorithm)

**단일화(unification)**는 두 타입을 같게 만드는 치환(substitution) $\sigma$를 찾습니다: $\sigma(\tau_1) = \sigma(\tau_2)$.

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

## 7. 타입 호환성과 강제 변환(Type Compatibility and Coercion)

### 7.1 타입 호환성 규칙

| 규칙 | 설명 | 예시 |
|------|------|------|
| **동일성(Identity)** | 같은 타입은 항상 호환됨 | `int` = `int` |
| **확장(Widening)** | 더 작은 수치 타입이 더 큰 타입에 맞음 | `int` -> `float` |
| **서브타이핑(Subtyping)** | 서브타입은 슈퍼타입과 호환됨 | `Cat` -> `Animal` |
| **구조적(Structural)** | 같은 구조 = 호환됨 | `{x: int, y: int}` = `{x: int, y: int}` |
| **명목적(Nominal)** | 같은 이름 = 호환됨 | Java 클래스 타입 |

### 7.2 암묵적 강제 변환(Implicit Coercion)

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

## 8. 오버로딩 해결(Overloading Resolution)

### 8.1 오버로딩이란?

**오버로딩(overloading)**은 여러 함수(또는 연산자)가 동일한 이름을 공유하지만 서로 다른 매개변수 타입을 가질 수 있도록 합니다. 컴파일러는 인자 타입을 기반으로 어떤 버전을 호출할지 해결해야 합니다.

### 8.2 해결 알고리즘

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

## 9. 선언 처리(Declaration Processing)

### 9.1 전방 참조(Forward References)

많은 언어들은 소스에서 나중에 등장하는 이름을 참조하는 선언을 허용합니다:

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

### 9.2 2-패스 전략(Two-Pass Strategy)

전방 참조를 처리하기 위해 선언에 대해 두 번의 패스를 사용합니다:

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

### 9.3 위상 정렬(Topological Sorting)

서로 의존하는 비함수 선언의 경우, 위상 정렬은 선언이 의존성 순서로 처리되도록 보장합니다:

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

## 10. 의미 오류 보고(Semantic Error Reporting)

### 10.1 오류 범주

| 범주 | 예시 | 심각도 |
|------|------|--------|
| 정의되지 않은 이름 | x가 정의되지 않은 상태에서 `print(x)` | 오류 |
| 타입 불일치 | `"hello" + 42` | 오류 |
| 잘못된 인자 수 | `f(1, 2)` (f는 1개 인자를 받음) | 오류 |
| 중복 정의 | `let x = 1; let x = 2;` | 오류 (또는 경고) |
| 사용되지 않는 변수 | `let x = 5;` (x가 읽히지 않음) | 경고 |
| 도달 불가능한 코드 | `return` 후의 코드 | 경고 |
| 암묵적 축소 | `let x: int = 3.14;` | 경고 (또는 오류) |
| 가림(Shadowing) | 내부 `x`가 외부 `x`를 가림 | 정보/경고 |

### 10.2 타입 검사에서의 오류 복구(Error Recovery)

타입 검사기는 오류 후에도 계속 진행하여 한 번의 패스에서 가능한 많은 문제를 보고해야 합니다. 핵심 기법은 **오류 타입(error type)**입니다:

```python
# ERROR_TYPE acts as a "universal" type that is compatible with everything.
# This prevents cascading errors.

def _type_assignable(self, target: Type, source: Type) -> bool:
    if target == ERROR_TYPE or source == ERROR_TYPE:
        return True  # Suppress cascading errors
    # ... normal checking
```

### 10.3 유용한 오류 메시지

좋은 컴파일러 오류는 다음과 같아야 합니다:

1. **정확함**: 문제의 정확한 위치를 가리킴
2. **명확함**: 무엇이 잘못되었는지 평이한 언어로 설명함
3. **유용함**: 문제를 수정하는 방법을 제안함

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

**형식화된 오류 예시:**

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

## 11. 요약

의미 분석(semantic analysis)은 구문적 정확성(파싱)과 의미 있는 프로그램 동작(코드 생성) 사이의 다리입니다. 프로그램이 의미 있는지 검증합니다: 이름이 정의되어 있고, 타입이 일관되며, 언어 규칙을 따르는지.

**핵심 개념:**

| 개념 | 설명 |
|------|------|
| **속성 문법(Attribute grammars)** | 의미 정보가 파스 트리에서 흐르는 방식을 형식화함 |
| **S-속성(S-attributed)** | 합성 속성만 사용; 아래에서 위로 평가 |
| **L-속성(L-attributed)** | 합성 + 왼쪽-오른쪽 상속; 단일 패스 평가 |
| **심벌 테이블(Symbol table)** | 이름을 타입, 스코프, 속성에 매핑 |
| **스코프 관리(Scope management)** | 가림이 있는 중첩 스코프를 위한 연쇄 해시 테이블 |
| **타입 검사(Type checking)** | 연산이 호환 가능한 타입에 적용되는지 검증 |
| **타입 추론(Type inference)** | 사용법으로부터 타입 추론 (지역 추론 또는 HM 단일화) |
| **강제 변환(Coercion)** | 암묵적 타입 변환 (예: int에서 float으로) |
| **오버로딩(Overloading)** | 여러 후보 중 올바른 함수 해결 |
| **오류 복구(Error recovery)** | 연쇄 타입 오류를 방지하기 위한 ERROR_TYPE 사용 |

**설계 가이드라인:**

1. 전방 참조를 위해 2-패스 접근 방식 사용 (선언 등록, 그 다음 본문 검사)
2. 연쇄 오류를 억제하는 오류 타입 설계
3. 정확한 오류 보고를 위해 전체적으로 소스 위치 추적
4. 놀라움을 방지하기 위해 강제 변환 규칙을 명시적이고 최소한으로 유지
5. 명시적 타입 어노테이션과 지역 타입 추론 모두 지원
6. 단일 패스에서 모든 오류를 보고 (첫 번째 오류에서 중단하지 않음)

---

## 연습 문제

### 연습 1: 심벌 테이블 확장

`SymbolTable` 클래스를 확장하여 다음을 추적하세요:

1. **사용되지 않는 변수**: 타입 검사 후, 정의되었지만 참조되지 않은 변수를 보고
2. **가림 경고**: 내부 스코프 변수가 외부 스코프 변수를 가릴 때 보고
3. **상수 전파**: 어떤 변수가 알려진 상수에 바인딩되어 있는지 추적

### 연습 2: 완전한 타입 검사기

다음 추가 구조를 처리하도록 타입 검사기를 확장하세요:

1. 배열/리스트 연산: `append`, `pop`, `len`
2. 문자열 연산: `+` (연결), `len`, 인덱싱
3. 복합 할당: `+=`, `-=`, `*=`, `/=`
4. 삼항 연산자: `cond ? then_expr : else_expr`

각 연산에 대한 적절한 타입 검사를 보장하고 유효한 프로그램과 무효한 프로그램 모두에 대한 테스트 케이스를 작성하세요.

### 연습 3: 타입 추론

다음을 처리하는 지역 타입 추론을 구현하세요:

```
let x = 5;                 // inferred as int
let y = [1, 2, 3];         // inferred as list[int]
let z = fn(a) => a + 1;    // inferred as (int) -> int
let w = if true then 1 else 2;  // inferred as int
```

추론이 실패할 때 유용한 오류를 생성해야 합니다 (예: `let a = [];` -- 빈 리스트의 원소 타입을 추론할 수 없음).

### 연습 4: 의미 오류 카탈로그

최소 15개의 다른 의미 오류를 실행하는 테스트 스위트를 작성하세요. 각 오류에 대해 다음을 작성하세요:

1. 오류를 유발하는 최소 프로그램
2. 예상 오류 메시지
3. 수정된 버전의 프로그램

포함할 오류 예시: 정의되지 않은 변수, 할당에서 타입 불일치, 잘못된 인자 수, 함수 외부의 return, 루프 외부의 break, 중복 함수 매개변수 이름, 재귀적 타입 정의.

### 연습 5: 제네릭을 사용한 오버로딩

단순 제네릭을 처리하도록 오버로딩 해결을 확장하세요:

```
fn identity<T>(x: T) -> T { return x; }
fn pair<T, U>(a: T, b: U) -> (T, U) { return (a, b); }

identity(5)          // T = int, returns int
identity("hello")    // T = string, returns string
pair(1, "two")       // T = int, U = string, returns (int, string)
```

제네릭 인스턴스화 로직을 구현하세요: `identity(5)`를 호출할 때 `T = int`를 결정하고 반환 타입이 `int`인지 검증합니다.

### 연습 6: 제어 흐름 분석(Control Flow Analysis)

다음을 검사하는 의미 분석 패스를 구현하세요:

1. 비-void 반환 타입을 가진 모든 함수가 모든 경로에서 반환하는지
2. `break`와 `continue`가 루프 내부에서만 나타나는지
3. `return` 후의 코드가 도달 불가능한지 (경고 발생)
4. 변수가 모든 경로에서 사용 전에 초기화되는지

이를 위해 각 함수의 제어 흐름 그래프를 분석해야 합니다. if-else와 while 루프를 처리하는 단순화된 버전으로 시작하세요.

---

[이전: 추상 구문 트리](./07_Abstract_Syntax_Trees.md) | [다음: 중간 표현](./09_Intermediate_Representations.md) | [개요](./00_Overview.md)
