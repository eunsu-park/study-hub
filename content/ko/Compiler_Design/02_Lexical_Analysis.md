# 레슨 2: 어휘 분석(Lexical Analysis)

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 컴파일 파이프라인에서 렉서(Lexer, 스캐너)의 역할을 설명할 수 있다
2. 토큰(Token), 어휘소(Lexeme), 패턴(Pattern)의 차이를 구분할 수 있다
3. 정규 표현식(Regular Expression)을 형식적으로 정의하고 토큰 패턴 명세에 활용할 수 있다
4. 톰슨 구성법(Thompson's Construction)을 적용하여 정규 표현식을 NFA로 변환할 수 있다
5. 부분집합 구성법(Subset Construction)을 적용하여 NFA를 DFA로 변환할 수 있다
6. 홉크로프트 알고리즘(Hopcroft's Algorithm)을 적용하여 DFA를 최소화할 수 있다
7. 최장 일치 규칙(Longest Match Rule)을 구현하고 토큰 우선순위를 처리할 수 있다
8. 간단한 프로그래밍 언어용 완전한 렉서를 Python으로 구현할 수 있다

---

## 1. 렉서(Lexer)의 역할

**렉서(Lexer)**는 **스캐너(Scanner)** 또는 **토크나이저(Tokenizer)**라고도 불리며, 컴파일러의 첫 번째 단계입니다. 소스 프로그램을 문자 스트림으로 읽어 **토큰(Token)**이라 불리는 의미 있는 단위로 묶어 줍니다.

```
소스 문자:          "if (x >= 42) return y + 1;"
                            |
                    [Lexer / Scanner]
                            |
                            v
토큰 스트림:         KW_IF  LPAREN  ID(x)  GEQ  INT(42)  RPAREN
                    KW_RETURN  ID(y)  PLUS  INT(1)  SEMI
```

### 어휘 분석을 별도 단계로 분리하는 이유

1. **단순성**: 파서는 원시 문자가 아닌 토큰(유한하고 구조화된 알파벳)을 대상으로 동작합니다. 이를 통해 문법이 대폭 단순화됩니다.

2. **효율성**: 식별자, 숫자, 문자열 같은 어휘 패턴은 정규적(Regular)이므로 유한 오토마톤(Finite Automaton)으로 인식할 수 있어 문맥 자유 파싱(Context-Free Parsing)보다 훨씬 빠릅니다.

3. **이식성**: 문자 집합 관련 문제(ASCII, UTF-8, 줄바꿈 방식 등)가 렉서 내부에만 국한됩니다.

4. **모듈성**: 렉서와 파서를 독립적으로 개발할 수 있습니다. 다른 소스 인코딩을 지원할 때 렉서만 수정하면 됩니다.

### 렉서가 하는 일

- 문자를 토큰으로 묶음
- 공백(Whitespace)과 주석(Comment) 제거
- 줄 번호 추적(오류 메시지용)
- 키워드(Keyword)와 식별자(Identifier) 구분
- 문자열 및 문자 리터럴의 이스케이프 처리
- 어휘 오류(불법 문자, 끝나지 않은 문자열 등) 보고

### 렉서가 하지 않는 일

- 구문(Syntax) 검사 → 파서의 역할
- 타입(Type) 검사 → 의미 분석기(Semantic Analyzer)의 역할
- 연산자 우선순위(Operator Precedence) 처리 → 파서의 역할

---

## 2. 토큰(Token), 어휘소(Lexeme), 패턴(Pattern)

서로 연관되어 있지만 구별되는 세 가지 개념입니다:

### 토큰(Token)

**토큰**은 어휘 단위의 한 종류를 나타내는 추상 기호입니다. 렉서의 출력이자 파서의 입력입니다.

토큰은 일반적으로 다음 요소로 구성됩니다:
- **토큰 타입**(또는 토큰 이름): `ID`, `INT`, `PLUS`, `KW_IF` 등
- 선택적 **속성 값(Attribute Value)**: 실제 텍스트, 숫자 값, 또는 심볼 테이블 포인터

### 어휘소(Lexeme)

**어휘소**는 토큰 패턴과 일치하는 소스 프로그램의 실제 부분 문자열입니다.

### 패턴(Pattern)

**패턴**은 토큰 타입에 속하는 어휘소들의 집합을 기술하는 규칙(일반적으로 정규 표현식)입니다.

### 예시

| 토큰 타입 | 패턴 (비형식적) | 어휘소 예시 |
|------------|---------------------|-----------------|
| `ID` | 문자로 시작하고 문자/숫자가 이어지는 것 | `x`, `count`, `myVar` |
| `INT` | 하나 이상의 숫자 | `0`, `42`, `1000` |
| `FLOAT` | 숫자, 점, 숫자 | `3.14`, `0.001` |
| `STRING` | `"` ... `"` | `"hello"`, `""` |
| `KW_IF` | `if` | `if` |
| `KW_WHILE` | `while` | `while` |
| `PLUS` | `+` | `+` |
| `GEQ` | `>=` | `>=` |
| `ASSIGN` | `=` | `=` |
| `LPAREN` | `(` | `(` |
| `SEMI` | `;` | `;` |

### 토큰 자료 구조

```python
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Optional

class TokenType(Enum):
    # Literals
    INT_LIT = auto()
    FLOAT_LIT = auto()
    STRING_LIT = auto()
    CHAR_LIT = auto()

    # Identifiers
    IDENTIFIER = auto()

    # Keywords
    KW_IF = auto()
    KW_ELSE = auto()
    KW_WHILE = auto()
    KW_FOR = auto()
    KW_RETURN = auto()
    KW_INT = auto()
    KW_FLOAT = auto()
    KW_VOID = auto()

    # Operators
    PLUS = auto()       # +
    MINUS = auto()      # -
    STAR = auto()       # *
    SLASH = auto()      # /
    ASSIGN = auto()     # =
    EQ = auto()         # ==
    NEQ = auto()        # !=
    LT = auto()         # <
    GT = auto()         # >
    LEQ = auto()        # <=
    GEQ = auto()        # >=
    AND = auto()        # &&
    OR = auto()         # ||
    NOT = auto()        # !

    # Delimiters
    LPAREN = auto()     # (
    RPAREN = auto()     # )
    LBRACE = auto()     # {
    RBRACE = auto()     # }
    LBRACKET = auto()   # [
    RBRACKET = auto()   # ]
    COMMA = auto()      # ,
    SEMI = auto()       # ;

    # Special
    EOF = auto()
    ERROR = auto()

@dataclass
class Token:
    type: TokenType
    lexeme: str
    value: Any = None      # Computed value (e.g., int 42, not string "42")
    line: int = 0
    column: int = 0

    def __repr__(self):
        if self.value is not None:
            return f"Token({self.type.name}, {self.lexeme!r}, value={self.value})"
        return f"Token({self.type.name}, {self.lexeme!r})"
```

---

## 3. 정규 표현식(Regular Expressions) — 형식적 정의

토큰 패턴은 **정규 표현식(Regular Expression, regex)**을 사용하여 명세합니다. 여기서는 컴파일러 이론에서 사용하는 형식 정의를 소개합니다. Python의 `re` 모듈처럼 확장된 정규 표현식 문법과는 다릅니다.

### 알파벳(Alphabet)과 문자열(String)

- **알파벳** $\Sigma$는 기호(문자)의 유한하고 비어 있지 않은 집합입니다.
- $\Sigma$ **위의 문자열**은 $\Sigma$에서 온 기호들의 유한 수열입니다.
- **빈 문자열**은 $\epsilon$ (엡실론)으로 표기합니다.
- 문자열 $w$의 **길이**는 $|w|$이며, $|\epsilon| = 0$입니다.
- $\Sigma^*$는 $\Sigma$ 위의 모든 문자열의 집합($\epsilon$ 포함)을 나타냅니다.
- $\Sigma^+ = \Sigma^* \setminus \{\epsilon\}$은 모든 비어 있지 않은 문자열의 집합입니다.

### 언어(Language)

$\Sigma$ **위의 언어** $L$은 $\Sigma^*$의 임의의 부분집합, 즉 임의의 문자열 집합입니다.

### 정규 표현식의 정의

알파벳 $\Sigma$ 위의 정규 표현식 $r$은 귀납적으로 정의됩니다:

**기저 사례(Base Cases):**

1. $\epsilon$은 언어 $\{\epsilon\}$을 나타내는 정규 표현식입니다.
2. 각 $a \in \Sigma$에 대해, 기호 $a$는 $\{a\}$를 나타내는 정규 표현식입니다.

**귀납적 사례 ($r$과 $s$가 각각 $L(r)$과 $L(s)$를 나타내는 정규 표현식일 때):**

3. **합집합(Union, 교대)**: $r \mid s$는 $L(r) \cup L(s)$를 나타냅니다.
4. **연접(Concatenation)**: $r \cdot s$ (또는 간략히 $rs$)는 $L(r) \cdot L(s) = \{xy \mid x \in L(r), y \in L(s)\}$를 나타냅니다.
5. **클리니 스타(Kleene Star, 폐쇄)**: $r^*$는 $L(r)^* = \bigcup_{i=0}^{\infty} L(r)^i$를 나타냅니다.

여기서 $L^0 = \{\epsilon\}$이고 $L^{i+1} = L^i \cdot L$입니다.

**연산자 우선순위** (높은 것부터 낮은 것 순):
1. 클리니 스타 $*$ (후위, 가장 강하게 결합)
2. 연접 (묵시적, 나란히 쓰기)
3. 합집합 $\mid$ (가장 약하게 결합)

모든 연산자는 좌결합적(Left-Associative)입니다.

### 예시

| 정규 표현식 | 언어 | 설명 |
|-------|----------|-------------|
| $a$ | $\{a\}$ | 문자열 "a"만 |
| $a \mid b$ | $\{a, b\}$ | "a" 또는 "b" |
| $ab$ | $\{ab\}$ | 문자열 "ab" |
| $a^*$ | $\{\epsilon, a, aa, aaa, \ldots\}$ | a가 0개 이상 |
| $(a \mid b)^*$ | $\{a, b\}$ 위의 모든 문자열 | a와 b의 임의 조합 |
| $a(a \mid b)^*$ | a로 시작하는 $\{a,b\}$ 위의 문자열 | a로 시작 |
| $(0 \mid 1)^* 0$ | 0으로 끝나는 이진 문자열 | 짝수 이진수 |

### 편의 표기(Shorthands)

다음은 형식적 정의의 일부는 아니지만 자주 사용되는 표기입니다:

| 표기 | 의미 | 형식적 동치 |
|-----------|---------|-------------------|
| $r^+$ | 하나 이상 | $rr^*$ |
| $r?$ | 0개 또는 1개 | $r \mid \epsilon$ |
| $[a\text{-}z]$ | 문자 클래스 | $a \mid b \mid \cdots \mid z$ |
| $[0\text{-}9]$ | 숫자 | $0 \mid 1 \mid \cdots \mid 9$ |
| $.$ | 임의 문자 | $\Sigma$ (전체 합집합) |
| $r\{n\}$ | 정확히 $n$번 반복 | $\underbrace{rr\cdots r}_{n}$ |

### 정규 표현식으로 표현한 토큰 패턴

```
Identifier:  [a-zA-Z_][a-zA-Z0-9_]*
Integer:     [0-9]+
Float:       [0-9]+\.[0-9]+([eE][+-]?[0-9]+)?
String:      "[^"\\]*(\\.[^"\\]*)*"
Comment:     //[^\n]*
Whitespace:  [ \t\n\r]+
```

### 정규 표현식의 대수 법칙

정규 표현식은 다음 항등식을 따릅니다(단순화에 유용):

| 법칙 | 내용 |
|-----|-----------|
| 합집합의 교환법칙 | $r \mid s = s \mid r$ |
| 합집합의 결합법칙 | $(r \mid s) \mid t = r \mid (s \mid t)$ |
| 연접의 결합법칙 | $(rs)t = r(st)$ |
| 연접의 합집합 분배법칙 | $r(s \mid t) = rs \mid rt$ |
| $\epsilon$은 연접의 항등원 | $\epsilon r = r\epsilon = r$ |
| $\emptyset$은 합집합의 항등원 | $r \mid \emptyset = r$ |
| $\emptyset$은 연접의 영원(Zero) | $\emptyset r = r\emptyset = \emptyset$ |
| 스타의 멱등성 | $(r^*)^* = r^*$ |
| 엡실론의 스타 | $\epsilon^* = \epsilon$ |

---

## 4. 정규 표현식에서 NFA로: 톰슨 구성법(Thompson's Construction)

**톰슨 구성법**(Ken Thompson, 1968)은 정규 표현식을 $\epsilon$-전이(epsilon-transition)를 포함하는 동치 NFA(비결정적 유한 오토마톤, Nondeterministic Finite Automaton)로 변환합니다.

### NFA 정의 (간략)

NFA는 5-튜플 $(Q, \Sigma, \delta, q_0, F)$이며:
- $Q$: 유한 상태 집합
- $\Sigma$: 입력 알파벳
- $\delta: Q \times (\Sigma \cup \{\epsilon\}) \rightarrow \mathcal{P}(Q)$: 전이 함수
- $q_0 \in Q$: 시작 상태
- $F \subseteq Q$: 수락 상태 집합

### 구성 규칙

각 규칙은 **시작 상태 하나**와 **수락 상태 하나**를 가진 NFA를 생성합니다. 시작 상태로 들어오는 전이나 수락 상태에서 나가는 전이는 없습니다.

**기저 사례 1: 빈 문자열 $\epsilon$**

```
  -->(q0)---ε--->(q1)
     start       accept
```

**기저 사례 2: 단일 기호 $a$**

```
  -->(q0)---a--->(q1)
     start       accept
```

**귀납적 사례 1: 합집합 $r \mid s$**

상태 $(r_0, r_f)$를 가진 NFA $N(r)$과 상태 $(s_0, s_f)$를 가진 NFA $N(s)$에 대해:

```
              ε ---> [N(r)] ---ε--->
            /                        \
  -->(q0)                              (q_f)
            \                        /
              ε ---> [N(s)] ---ε--->
     start                          accept
```

**귀납적 사례 2: 연접 $rs$**

$N(r)$의 수락 상태와 $N(s)$의 시작 상태를 합칩니다:

```
  -->(r0)---[N(r)]--->(r_f = s_0)---[N(s)]--->(s_f)
     start                                     accept
```

**귀납적 사례 3: 클리니 스타 $r^*$**

```
              ε ---> [N(r)] ---ε--->
            /           |            \
  -->(q0)     <---ε-----+             (q_f)
            \                        /
              ----------ε---------->
     start                          accept
```

### 톰슨 구성법의 특성

1. 길이 $n$의 정규 표현식에 대해 NFA가 최대 $2n$개의 상태를 가집니다.
2. 각 상태는 최대 두 개의 나가는 전이를 가집니다.
3. NFA는 정확히 하나의 시작 상태와 하나의 수락 상태를 가집니다.
4. 구성 과정은 정규 표현식 크기에 선형적: $O(n)$

### Python 구현

```python
from dataclasses import dataclass, field
from typing import Dict, Set, Tuple, List, Optional

@dataclass
class NFAState:
    """A state in an NFA."""
    id: int
    is_accept: bool = False
    transitions: Dict[str, List['NFAState']] = field(default_factory=dict)

    def add_transition(self, symbol: str, target: 'NFAState'):
        """Add a transition on symbol (use 'ε' for epsilon)."""
        if symbol not in self.transitions:
            self.transitions[symbol] = []
        self.transitions[symbol].append(target)

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return isinstance(other, NFAState) and self.id == other.id

    def __repr__(self):
        suffix = " (accept)" if self.is_accept else ""
        return f"q{self.id}{suffix}"


class NFA:
    """Nondeterministic Finite Automaton."""

    _state_counter = 0

    def __init__(self, start: NFAState, accept: NFAState):
        self.start = start
        self.accept = accept

    @classmethod
    def _new_state(cls, is_accept=False) -> NFAState:
        state = NFAState(id=cls._state_counter, is_accept=is_accept)
        cls._state_counter += 1
        return state

    @classmethod
    def reset_counter(cls):
        cls._state_counter = 0

    @classmethod
    def from_epsilon(cls) -> 'NFA':
        """Thompson's construction: epsilon."""
        start = cls._new_state()
        accept = cls._new_state(is_accept=True)
        start.add_transition('ε', accept)
        return cls(start, accept)

    @classmethod
    def from_symbol(cls, symbol: str) -> 'NFA':
        """Thompson's construction: single symbol."""
        start = cls._new_state()
        accept = cls._new_state(is_accept=True)
        start.add_transition(symbol, accept)
        return cls(start, accept)

    @classmethod
    def union(cls, nfa1: 'NFA', nfa2: 'NFA') -> 'NFA':
        """Thompson's construction: r | s."""
        start = cls._new_state()
        accept = cls._new_state(is_accept=True)

        start.add_transition('ε', nfa1.start)
        start.add_transition('ε', nfa2.start)

        nfa1.accept.is_accept = False
        nfa1.accept.add_transition('ε', accept)

        nfa2.accept.is_accept = False
        nfa2.accept.add_transition('ε', accept)

        return cls(start, accept)

    @classmethod
    def concatenation(cls, nfa1: 'NFA', nfa2: 'NFA') -> 'NFA':
        """Thompson's construction: r s."""
        nfa1.accept.is_accept = False
        nfa1.accept.add_transition('ε', nfa2.start)
        return cls(nfa1.start, nfa2.accept)

    @classmethod
    def kleene_star(cls, nfa1: 'NFA') -> 'NFA':
        """Thompson's construction: r*."""
        start = cls._new_state()
        accept = cls._new_state(is_accept=True)

        start.add_transition('ε', nfa1.start)
        start.add_transition('ε', accept)

        nfa1.accept.is_accept = False
        nfa1.accept.add_transition('ε', nfa1.start)
        nfa1.accept.add_transition('ε', accept)

        return cls(start, accept)


def regex_to_nfa(regex: str) -> NFA:
    """
    Convert a simple regex to NFA using Thompson's construction.

    Supported operators: | (union), * (star), concatenation (implicit),
                         ( ) (grouping)

    This parser handles operator precedence correctly:
      * > concatenation > |
    """
    NFA.reset_counter()
    pos = 0

    def peek():
        nonlocal pos
        return regex[pos] if pos < len(regex) else None

    def consume():
        nonlocal pos
        ch = regex[pos]
        pos += 1
        return ch

    def parse_expr() -> NFA:
        """expr = term ('|' term)*"""
        left = parse_term()
        while peek() == '|':
            consume()  # eat '|'
            right = parse_term()
            left = NFA.union(left, right)
        return left

    def parse_term() -> NFA:
        """term = factor+"""
        # A term is one or more concatenated factors
        result = parse_factor()
        while peek() is not None and peek() not in ('|', ')'):
            right = parse_factor()
            result = NFA.concatenation(result, right)
        return result

    def parse_factor() -> NFA:
        """factor = base ('*')?"""
        base = parse_base()
        while peek() == '*':
            consume()
            base = NFA.kleene_star(base)
        return base

    def parse_base() -> NFA:
        """base = '(' expr ')' | symbol"""
        if peek() == '(':
            consume()  # eat '('
            nfa = parse_expr()
            if peek() != ')':
                raise SyntaxError("Expected ')'")
            consume()  # eat ')'
            return nfa
        elif peek() is not None and peek() not in ('|', ')', '*'):
            ch = consume()
            return NFA.from_symbol(ch)
        else:
            # Empty: return epsilon NFA
            return NFA.from_epsilon()

    result = parse_expr()
    if pos != len(regex):
        raise SyntaxError(f"Unexpected character at position {pos}: {regex[pos]}")
    return result


def epsilon_closure(states: Set[NFAState]) -> Set[NFAState]:
    """Compute the epsilon-closure of a set of NFA states."""
    closure = set(states)
    stack = list(states)
    while stack:
        state = stack.pop()
        for target in state.transitions.get('ε', []):
            if target not in closure:
                closure.add(target)
                stack.append(target)
    return closure


def nfa_simulate(nfa: NFA, input_string: str) -> bool:
    """Simulate an NFA on an input string."""
    current = epsilon_closure({nfa.start})
    for ch in input_string:
        next_states = set()
        for state in current:
            for target in state.transitions.get(ch, []):
                next_states.add(target)
        current = epsilon_closure(next_states)
    return any(s.is_accept for s in current)


# Example usage
nfa = regex_to_nfa("(a|b)*abb")
print("NFA for (a|b)*abb")
print(f"  'abb':   {nfa_simulate(nfa, 'abb')}")      # True
print(f"  'aabb':  {nfa_simulate(nfa, 'aabb')}")     # True
print(f"  'babb':  {nfa_simulate(nfa, 'babb')}")     # True
print(f"  'ab':    {nfa_simulate(nfa, 'ab')}")       # False
print(f"  'abab':  {nfa_simulate(nfa, 'abab')}")     # False
print(f"  '':      {nfa_simulate(nfa, '')}")          # False
```

---

## 5. NFA에서 DFA로: 부분집합 구성법(Subset Construction)

NFA는 정규 표현식에서 만들기는 쉽지만 비결정적입니다. 즉, 같은 기호에 대해 하나의 상태에서 여러 전이가 가능합니다. **DFA**(결정적 유한 오토마톤, Deterministic Finite Automaton)는 각 상태가 기호당 정확히 하나의 전이만 가지므로 시뮬레이션 효율이 더 높습니다.

**부분집합 구성법**(또는 **멱집합 구성법**, Powerset Construction)은 NFA를 동치 DFA로 변환합니다.

### 알고리즘

```
Input:  NFA N = (Q_N, Σ, δ_N, q_0, F_N)
Output: DFA D = (Q_D, Σ, δ_D, d_0, F_D)

1. d_0 = ε-closure({q_0})                  // DFA의 시작 상태
2. Q_D = {d_0}                              // DFA 상태 집합 (각 원소는 NFA 상태 집합)
3. worklist = [d_0]                         // 처리할 상태
4. while worklist is not empty:
5.     S = worklist.pop()
6.     for each symbol a in Σ:
7.         T = ε-closure(move(S, a))        // move(S,a) = S에 속한 모든 s에 대한 δ_N(s,a)의 합집합
8.         if T is not empty:
9.             if T not in Q_D:
10.                Q_D.add(T)
11.                worklist.append(T)
12.            δ_D(S, a) = T
13. F_D = {S in Q_D | S ∩ F_N ≠ ∅}         // NFA 수락 상태를 포함하는 DFA 상태가 수락 상태
```

### 복잡도

최악의 경우, 상태 수 $n$인 NFA에서 $2^n$개의 상태를 가진 DFA가 생성될 수 있습니다(따라서 "멱집합" 구성법이라 불립니다). 실제로는 도달 가능한 상태의 수가 훨씬 적습니다.

### Python 구현

```python
from typing import FrozenSet

@dataclass
class DFAState:
    """A state in a DFA, corresponding to a set of NFA states."""
    id: int
    nfa_states: FrozenSet[NFAState]
    is_accept: bool = False
    transitions: Dict[str, 'DFAState'] = field(default_factory=dict)

    def __repr__(self):
        nfa_ids = sorted(s.id for s in self.nfa_states)
        suffix = " (accept)" if self.is_accept else ""
        return f"D{self.id}{{{','.join(map(str, nfa_ids))}}}{suffix}"


class DFA:
    """Deterministic Finite Automaton."""

    def __init__(self, start: DFAState, states: List[DFAState], alphabet: Set[str]):
        self.start = start
        self.states = states
        self.alphabet = alphabet

    @classmethod
    def from_nfa(cls, nfa: NFA) -> 'DFA':
        """Convert NFA to DFA using subset construction."""

        # Collect alphabet (all symbols except epsilon)
        alphabet = set()
        visited_nfa = set()
        stack = [nfa.start]
        while stack:
            state = stack.pop()
            if state in visited_nfa:
                continue
            visited_nfa.add(state)
            for symbol, targets in state.transitions.items():
                if symbol != 'ε':
                    alphabet.add(symbol)
                for t in targets:
                    stack.append(t)

        # Subset construction
        dfa_id = 0
        start_nfa_states = frozenset(epsilon_closure({nfa.start}))
        start_is_accept = any(s.is_accept for s in start_nfa_states)
        start_dfa = DFAState(dfa_id, start_nfa_states, start_is_accept)
        dfa_id += 1

        # Map from frozenset of NFA states -> DFA state
        state_map = {start_nfa_states: start_dfa}
        dfa_states = [start_dfa]
        worklist = [start_dfa]

        while worklist:
            current = worklist.pop()

            for symbol in sorted(alphabet):
                # Compute move(current.nfa_states, symbol)
                move_result = set()
                for nfa_state in current.nfa_states:
                    for target in nfa_state.transitions.get(symbol, []):
                        move_result.add(target)

                if not move_result:
                    continue

                # Compute epsilon-closure of the move result
                closure = frozenset(epsilon_closure(move_result))

                if closure not in state_map:
                    is_accept = any(s.is_accept for s in closure)
                    new_state = DFAState(dfa_id, closure, is_accept)
                    dfa_id += 1
                    state_map[closure] = new_state
                    dfa_states.append(new_state)
                    worklist.append(new_state)

                current.transitions[symbol] = state_map[closure]

        return cls(start_dfa, dfa_states, alphabet)

    def simulate(self, input_string: str) -> bool:
        """Simulate the DFA on an input string."""
        current = self.start
        for ch in input_string:
            if ch not in current.transitions:
                return False
            current = current.transitions[ch]
        return current.is_accept

    def print_table(self):
        """Print the DFA transition table."""
        symbols = sorted(self.alphabet)
        header = f"{'State':>10} | {'Accept':>6} | " + " | ".join(
            f"{s:>5}" for s in symbols
        )
        print(header)
        print("-" * len(header))
        for state in sorted(self.states, key=lambda s: s.id):
            row = f"{'D' + str(state.id):>10} | {'yes' if state.is_accept else 'no':>6} | "
            cells = []
            for s in symbols:
                if s in state.transitions:
                    cells.append(f"{'D' + str(state.transitions[s].id):>5}")
                else:
                    cells.append(f"{'--':>5}")
            row += " | ".join(cells)
            print(row)


# Example: Convert NFA for (a|b)*abb to DFA
print("=== NFA to DFA: (a|b)*abb ===\n")
nfa = regex_to_nfa("(a|b)*abb")
dfa = DFA.from_nfa(nfa)
dfa.print_table()

print()
test_strings = ["abb", "aabb", "babb", "ababb", "ab", "ba", ""]
for s in test_strings:
    result = dfa.simulate(s)
    print(f"  '{s}': {'accepted' if result else 'rejected'}")
```

---

## 6. DFA 최소화(DFA Minimization): 홉크로프트 알고리즘(Hopcroft's Algorithm)

부분집합 구성법으로 생성된 DFA는 불필요하게 많은 상태를 가질 수 있습니다. **DFA 최소화**는 동일한 언어를 인식하는 가장 작은 DFA를 생성합니다.

### 동치 상태(Equivalent States)

두 상태 $p$와 $q$가 **동치**(구별 불가능)인 것은 모든 문자열 $w$에 대해:

$$\hat{\delta}(p, w) \in F \iff \hat{\delta}(q, w) \in F$$

즉, $p$나 $q$에서 출발하더라도 DFA가 정확히 같은 문자열 집합을 수락합니다.

### 홉크로프트 알고리즘 (분할 정제, Partition Refinement)

알고리즘은 거친 분할로 시작하여 점차 세밀하게 정제합니다:

```
Input:  DFA D = (Q, Σ, δ, q_0, F)
Output: Minimized DFA D' with equivalent states merged

1. Initial partition P = {F, Q \ F}     // 수락 상태와 비수락 상태
2. worklist W = {F}                      // (또는 F, Q\F 중 더 작은 것)
3. while W is not empty:
4.     A = W.pop()
5.     for each symbol c in Σ:
6.         X = {q in Q | δ(q, c) in A}  // c로 A에 전이하는 상태들
7.         for each group Y in P:
8.             Y1 = Y ∩ X                // A로 가는 Y의 상태들
9.             Y2 = Y \ X               // A로 가지 않는 Y의 상태들
10.            if Y1 ≠ ∅ and Y2 ≠ ∅:    // Y가 분할됨
11.                Replace Y in P with Y1 and Y2
12.                if Y in W:
13.                    Replace Y in W with Y1 and Y2
14.                else:
15.                    Add smaller of Y1, Y2 to W
16. Return DFA with each partition group as a single state
```

**시간 복잡도**: $O(n \log n)$, $n = |Q|$

### Python 구현

```python
def minimize_dfa(dfa: DFA) -> DFA:
    """
    Minimize a DFA using Hopcroft's algorithm (partition refinement).
    Returns a new minimized DFA.
    """
    states = dfa.states
    alphabet = sorted(dfa.alphabet)

    # Remove unreachable states
    reachable = set()
    stack = [dfa.start]
    while stack:
        s = stack.pop()
        if s in reachable:
            continue
        reachable.add(s)
        for sym in alphabet:
            if sym in s.transitions:
                stack.append(s.transitions[sym])
    states = [s for s in states if s in reachable]

    # Add dead state if needed (for completeness)
    dead_state = None
    for s in states:
        for sym in alphabet:
            if sym not in s.transitions:
                if dead_state is None:
                    dead_state = DFAState(
                        id=-1,
                        nfa_states=frozenset(),
                        is_accept=False
                    )
                    for a in alphabet:
                        dead_state.transitions[a] = dead_state
                    states.append(dead_state)
                s.transitions[sym] = dead_state

    # Initial partition: {accept states, non-accept states}
    accept_group = frozenset(s for s in states if s.is_accept)
    non_accept_group = frozenset(s for s in states if not s.is_accept)

    partition = set()
    if accept_group:
        partition.add(accept_group)
    if non_accept_group:
        partition.add(non_accept_group)

    # Worklist
    worklist = set()
    if accept_group and non_accept_group:
        # Add the smaller group
        if len(accept_group) <= len(non_accept_group):
            worklist.add(accept_group)
        else:
            worklist.add(non_accept_group)
    elif accept_group:
        worklist.add(accept_group)
    elif non_accept_group:
        worklist.add(non_accept_group)

    # Refinement
    while worklist:
        A = worklist.pop()

        for c in alphabet:
            # X = states that transition into A on symbol c
            X = frozenset(
                s for s in states
                if c in s.transitions and s.transitions[c] in A
            )

            new_partition = set()
            for Y in partition:
                Y1 = Y & X
                Y2 = Y - X

                if Y1 and Y2:
                    new_partition.add(Y1)
                    new_partition.add(Y2)
                    if Y in worklist:
                        worklist.discard(Y)
                        worklist.add(Y1)
                        worklist.add(Y2)
                    else:
                        if len(Y1) <= len(Y2):
                            worklist.add(Y1)
                        else:
                            worklist.add(Y2)
                else:
                    new_partition.add(Y)

            partition = new_partition

    # Build minimized DFA
    # Map each old state to its partition representative
    state_to_group = {}
    for group in partition:
        representative = min(group, key=lambda s: s.id)
        for s in group:
            state_to_group[s] = representative

    # Create new DFA states
    new_dfa_id = 0
    group_to_new_state = {}

    for group in partition:
        rep = min(group, key=lambda s: s.id)
        if rep == dead_state:
            continue  # Skip the dead state
        is_accept = any(s.is_accept for s in group)
        new_state = DFAState(
            id=new_dfa_id,
            nfa_states=frozenset().union(*(s.nfa_states for s in group)),
            is_accept=is_accept
        )
        group_to_new_state[frozenset(group)] = new_state
        new_dfa_id += 1

    # Add transitions
    for group in partition:
        rep = min(group, key=lambda s: s.id)
        if rep == dead_state:
            continue
        new_state = group_to_new_state[frozenset(group)]
        for c in alphabet:
            if c in rep.transitions:
                target = rep.transitions[c]
                target_group = None
                for g in partition:
                    if target in g:
                        target_group = g
                        break
                if target_group and frozenset(target_group) in group_to_new_state:
                    new_state.transitions[c] = group_to_new_state[frozenset(target_group)]

    # Find the new start state
    start_group = None
    for group in partition:
        if dfa.start in group:
            start_group = group
            break

    new_start = group_to_new_state[frozenset(start_group)]
    new_states = list(group_to_new_state.values())

    return DFA(new_start, new_states, dfa.alphabet)


# Example
print("\n=== DFA Minimization ===\n")
print("Before minimization:")
dfa.print_table()

min_dfa = minimize_dfa(dfa)
print(f"\nAfter minimization ({len(min_dfa.states)} states):")
min_dfa.print_table()

# Verify correctness
print("\nVerification:")
for s in ["abb", "aabb", "babb", "ababb", "ab", "ba", ""]:
    orig = dfa.simulate(s)
    mini = min_dfa.simulate(s)
    status = "OK" if orig == mini else "MISMATCH!"
    print(f"  '{s}': original={orig}, minimized={mini}  [{status}]")
```

---

## 7. 토큰 인식(Token Recognition)

모든 토큰 패턴에 대한 DFA가 준비되었을 때, 렉서는 어떤 토큰을 반환할지를 어떻게 결정할까요?

### 최장 일치 규칙(Longest Match Rule)

렉서는 항상 현재 위치에서 **가능한 가장 긴 토큰**을 반환합니다.

```
Input:  "iffy = 10;"

최장 일치 없이:  KW_IF("if") ID("fy") ...   잘못됨!
최장 일치 적용:  ID("iffy") ASSIGN("=") ...  올바름!
```

### 우선순위 규칙(Priority Rule)

두 패턴이 동일한 최장 어휘소와 일치할 때, **먼저 나열된**(우선순위 높은) 것이 선택됩니다. 키워드는 일반적으로 식별자보다 먼저 나열됩니다.

```
Input:  "if"

패턴 순서:
  1. KW_IF:   "if"       <--- 이긴다 (높은 우선순위)
  2. ID:      [a-z]+     <--- "if"와도 일치
```

### 결합 DFA 방식(Combined DFA Approach)

실제로는 모든 토큰 패턴 DFA를 하나의 DFA로 결합합니다:

1. 각 토큰 패턴에 대해 NFA를 만들고, 수락 상태에 토큰 타입을 표시합니다.
2. 새 시작 상태와 각 패턴 NFA로의 엡실론 전이를 가진 결합 NFA를 만듭니다.
3. 부분집합 구성법으로 DFA로 변환합니다.
4. DFA 상태가 여러 패턴의 수락 상태를 포함할 때, 가장 높은 우선순위를 선택합니다.

```
            ε ---> NFA(KW_IF)
           /
  start --ε ---> NFA(KW_WHILE)
           \
            ε ---> NFA(ID)
                   ...
```

### 최장 일치 구현

```python
def longest_match(dfa: DFA, source: str, pos: int) -> Tuple[Optional[str], int]:
    """
    Find the longest match starting at pos.
    Returns (token_type, length) or (None, 0) if no match.
    """
    state = dfa.start
    last_accept = None
    last_accept_pos = pos
    current_pos = pos

    while current_pos < len(source):
        ch = source[current_pos]
        if ch not in state.transitions:
            break
        state = state.transitions[ch]
        current_pos += 1
        if state.is_accept:
            last_accept = state  # Remember this accept position
            last_accept_pos = current_pos

    if last_accept is not None:
        return (last_accept, last_accept_pos - pos)
    return (None, 0)
```

### 렉서의 오류 복구(Error Recovery)

렉서가 어떤 토큰 패턴과도 일치하지 않는 문자를 만나면 여러 선택지가 있습니다:

1. **패닉 모드(Panic Mode)**: 문제가 되는 문자를 건너뛰고 오류를 보고
2. **삽입(Insert)**: 빠진 문자를 삽입 (드묾)
3. **삭제(Delete)**: 문제가 되는 문자를 삭제
4. **교체(Replace)**: 문제가 되는 문자를 유효한 문자로 교체

대부분의 컴파일러는 패닉 모드를 사용합니다: 오류를 보고하고 한 문자를 건너뜁니다.

```python
def scan_with_recovery(source: str) -> List[Token]:
    """Scan with panic-mode error recovery."""
    tokens = []
    pos = 0
    line = 1
    col = 1

    while pos < len(source):
        # Skip whitespace
        if source[pos] in ' \t\r':
            pos += 1
            col += 1
            continue
        if source[pos] == '\n':
            pos += 1
            line += 1
            col = 1
            continue

        # Try to match a token
        best_match = None
        best_length = 0
        best_type = None

        for token_type, pattern_dfa in token_patterns:
            match_state, length = longest_match(pattern_dfa, source, pos)
            if length > best_length:
                best_match = match_state
                best_length = length
                best_type = token_type

        if best_match:
            lexeme = source[pos:pos + best_length]
            tokens.append(Token(best_type, lexeme, line=line, column=col))
            pos += best_length
            col += best_length
        else:
            # Error recovery: skip one character
            print(f"Lexical error at line {line}, col {col}: "
                  f"unexpected character '{source[pos]}'")
            pos += 1
            col += 1

    return tokens
```

---

## 8. 완전한 렉서 구현

다음은 앞서 다룬 기법을 활용하여 처음부터 만든 C 유사 언어(C-like Language)용 완전한 자족형 렉서입니다.

```python
"""
Complete lexer for a simple C-like language.
Uses a table-driven approach with longest-match and priority rules.
"""

import re
from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Optional, Tuple, Any


class TT(Enum):
    """Token types."""
    # Keywords (listed first for priority)
    KW_IF = auto()
    KW_ELSE = auto()
    KW_WHILE = auto()
    KW_FOR = auto()
    KW_RETURN = auto()
    KW_INT = auto()
    KW_FLOAT = auto()
    KW_VOID = auto()
    KW_CHAR = auto()
    KW_BREAK = auto()
    KW_CONTINUE = auto()

    # Literals
    FLOAT_LIT = auto()
    INT_LIT = auto()
    STRING_LIT = auto()
    CHAR_LIT = auto()

    # Identifiers
    IDENT = auto()

    # Multi-character operators (before single-char ones for longest match)
    LEQ = auto()        # <=
    GEQ = auto()        # >=
    EQ = auto()         # ==
    NEQ = auto()        # !=
    AND = auto()        # &&
    OR = auto()         # ||
    ARROW = auto()      # ->
    INC = auto()        # ++
    DEC = auto()        # --
    PLUS_ASSIGN = auto()   # +=
    MINUS_ASSIGN = auto()  # -=
    STAR_ASSIGN = auto()   # *=
    SLASH_ASSIGN = auto()  # /=

    # Single-character operators and delimiters
    PLUS = auto()       # +
    MINUS = auto()      # -
    STAR = auto()       # *
    SLASH = auto()      # /
    PERCENT = auto()    # %
    ASSIGN = auto()     # =
    LT = auto()         # <
    GT = auto()         # >
    NOT = auto()        # !
    AMP = auto()        # &
    PIPE = auto()       # |
    LPAREN = auto()     # (
    RPAREN = auto()     # )
    LBRACE = auto()     # {
    RBRACE = auto()     # }
    LBRACKET = auto()   # [
    RBRACKET = auto()   # ]
    COMMA = auto()      # ,
    SEMI = auto()       # ;
    DOT = auto()        # .

    # Special
    EOF = auto()
    ERROR = auto()


@dataclass
class Token:
    type: TT
    lexeme: str
    value: Any = None
    line: int = 0
    column: int = 0

    def __repr__(self):
        val = f", value={self.value!r}" if self.value is not None else ""
        return f"Token({self.type.name}, {self.lexeme!r}{val}, {self.line}:{self.column})"


class LexerError(Exception):
    def __init__(self, message, line, column):
        super().__init__(message)
        self.line = line
        self.column = column


class Lexer:
    """
    A complete lexer for a C-like language.

    Uses a hand-written state machine approach for correctness
    and efficiency. Implements longest-match and keyword priority.
    """

    KEYWORDS = {
        'if':       TT.KW_IF,
        'else':     TT.KW_ELSE,
        'while':    TT.KW_WHILE,
        'for':      TT.KW_FOR,
        'return':   TT.KW_RETURN,
        'int':      TT.KW_INT,
        'float':    TT.KW_FLOAT,
        'void':     TT.KW_VOID,
        'char':     TT.KW_CHAR,
        'break':    TT.KW_BREAK,
        'continue': TT.KW_CONTINUE,
    }

    TWO_CHAR_OPS = {
        '<=': TT.LEQ,
        '>=': TT.GEQ,
        '==': TT.EQ,
        '!=': TT.NEQ,
        '&&': TT.AND,
        '||': TT.OR,
        '->': TT.ARROW,
        '++': TT.INC,
        '--': TT.DEC,
        '+=': TT.PLUS_ASSIGN,
        '-=': TT.MINUS_ASSIGN,
        '*=': TT.STAR_ASSIGN,
        '/=': TT.SLASH_ASSIGN,
    }

    SINGLE_CHAR_OPS = {
        '+': TT.PLUS,
        '-': TT.MINUS,
        '*': TT.STAR,
        '/': TT.SLASH,
        '%': TT.PERCENT,
        '=': TT.ASSIGN,
        '<': TT.LT,
        '>': TT.GT,
        '!': TT.NOT,
        '&': TT.AMP,
        '|': TT.PIPE,
        '(': TT.LPAREN,
        ')': TT.RPAREN,
        '{': TT.LBRACE,
        '}': TT.RBRACE,
        '[': TT.LBRACKET,
        ']': TT.RBRACKET,
        ',': TT.COMMA,
        ';': TT.SEMI,
        '.': TT.DOT,
    }

    def __init__(self, source: str, filename: str = "<input>"):
        self.source = source
        self.filename = filename
        self.pos = 0
        self.line = 1
        self.column = 1
        self.errors: List[str] = []

    def peek(self, offset=0) -> Optional[str]:
        """Look at the character at current position + offset."""
        idx = self.pos + offset
        if idx < len(self.source):
            return self.source[idx]
        return None

    def advance(self) -> str:
        """Consume and return the current character."""
        ch = self.source[self.pos]
        self.pos += 1
        if ch == '\n':
            self.line += 1
            self.column = 1
        else:
            self.column += 1
        return ch

    def make_token(self, token_type: TT, lexeme: str,
                   value: Any = None, line: int = 0, col: int = 0) -> Token:
        return Token(token_type, lexeme, value, line, col)

    def skip_whitespace_and_comments(self):
        """Skip whitespace, single-line comments, and multi-line comments."""
        while self.pos < len(self.source):
            ch = self.peek()

            # Whitespace
            if ch in ' \t\r\n':
                self.advance()
                continue

            # Single-line comment: //
            if ch == '/' and self.peek(1) == '/':
                self.advance()  # skip /
                self.advance()  # skip /
                while self.pos < len(self.source) and self.peek() != '\n':
                    self.advance()
                continue

            # Multi-line comment: /* ... */
            if ch == '/' and self.peek(1) == '*':
                start_line = self.line
                start_col = self.column
                self.advance()  # skip /
                self.advance()  # skip *
                depth = 1
                while self.pos < len(self.source) and depth > 0:
                    if self.peek() == '/' and self.peek(1) == '*':
                        depth += 1
                        self.advance()
                        self.advance()
                    elif self.peek() == '*' and self.peek(1) == '/':
                        depth -= 1
                        self.advance()
                        self.advance()
                    else:
                        self.advance()
                if depth > 0:
                    self.errors.append(
                        f"{self.filename}:{start_line}:{start_col}: "
                        f"unterminated comment"
                    )
                continue

            break  # Not whitespace or comment

    def scan_number(self) -> Token:
        """Scan an integer or float literal."""
        start_line = self.line
        start_col = self.column
        start_pos = self.pos
        is_float = False

        # Integer part
        while self.pos < len(self.source) and self.peek().isdigit():
            self.advance()

        # Fractional part
        if self.peek() == '.' and self.peek(1) is not None and self.peek(1).isdigit():
            is_float = True
            self.advance()  # skip '.'
            while self.pos < len(self.source) and self.peek().isdigit():
                self.advance()

        # Exponent part
        if self.peek() in ('e', 'E'):
            is_float = True
            self.advance()  # skip 'e'/'E'
            if self.peek() in ('+', '-'):
                self.advance()
            if not (self.pos < len(self.source) and self.peek().isdigit()):
                self.errors.append(
                    f"{self.filename}:{self.line}:{self.column}: "
                    f"expected digit in exponent"
                )
            while self.pos < len(self.source) and self.peek().isdigit():
                self.advance()

        lexeme = self.source[start_pos:self.pos]
        if is_float:
            return self.make_token(TT.FLOAT_LIT, lexeme, float(lexeme),
                                   start_line, start_col)
        else:
            return self.make_token(TT.INT_LIT, lexeme, int(lexeme),
                                   start_line, start_col)

    def scan_string(self) -> Token:
        """Scan a string literal (double-quoted)."""
        start_line = self.line
        start_col = self.column
        self.advance()  # skip opening "
        chars = []

        while self.pos < len(self.source) and self.peek() != '"':
            if self.peek() == '\n':
                self.errors.append(
                    f"{self.filename}:{self.line}:{self.column}: "
                    f"unterminated string literal"
                )
                break
            if self.peek() == '\\':
                self.advance()  # skip backslash
                esc = self.advance() if self.pos < len(self.source) else ''
                escape_map = {
                    'n': '\n', 't': '\t', 'r': '\r',
                    '\\': '\\', '"': '"', '0': '\0',
                }
                chars.append(escape_map.get(esc, esc))
            else:
                chars.append(self.advance())

        if self.pos < len(self.source) and self.peek() == '"':
            self.advance()  # skip closing "
        else:
            self.errors.append(
                f"{self.filename}:{start_line}:{start_col}: "
                f"unterminated string literal"
            )

        value = ''.join(chars)
        lexeme = self.source[
            (start_col - 1) + sum(1 for _ in range(start_line - 1)):self.pos
        ]
        return self.make_token(TT.STRING_LIT, f'"{value}"', value,
                               start_line, start_col)

    def scan_char(self) -> Token:
        """Scan a character literal (single-quoted)."""
        start_line = self.line
        start_col = self.column
        self.advance()  # skip opening '

        if self.peek() == '\\':
            self.advance()
            esc = self.advance() if self.pos < len(self.source) else ''
            escape_map = {
                'n': '\n', 't': '\t', 'r': '\r',
                '\\': '\\', "'": "'", '0': '\0',
            }
            value = escape_map.get(esc, esc)
        elif self.pos < len(self.source):
            value = self.advance()
        else:
            value = ''
            self.errors.append(
                f"{self.filename}:{start_line}:{start_col}: "
                f"empty character literal"
            )

        if self.pos < len(self.source) and self.peek() == "'":
            self.advance()  # skip closing '
        else:
            self.errors.append(
                f"{self.filename}:{start_line}:{start_col}: "
                f"unterminated character literal"
            )

        return self.make_token(TT.CHAR_LIT, f"'{value}'", value,
                               start_line, start_col)

    def scan_identifier_or_keyword(self) -> Token:
        """Scan an identifier or keyword."""
        start_line = self.line
        start_col = self.column
        start_pos = self.pos

        while (self.pos < len(self.source) and
               (self.peek().isalnum() or self.peek() == '_')):
            self.advance()

        lexeme = self.source[start_pos:self.pos]

        # Check if it's a keyword
        if lexeme in self.KEYWORDS:
            return self.make_token(self.KEYWORDS[lexeme], lexeme,
                                   line=start_line, col=start_col)
        else:
            return self.make_token(TT.IDENT, lexeme,
                                   line=start_line, col=start_col)

    def next_token(self) -> Token:
        """Return the next token from the source."""
        self.skip_whitespace_and_comments()

        if self.pos >= len(self.source):
            return self.make_token(TT.EOF, "", line=self.line, col=self.column)

        start_line = self.line
        start_col = self.column
        ch = self.peek()

        # Numbers
        if ch.isdigit():
            return self.scan_number()

        # Identifiers and keywords
        if ch.isalpha() or ch == '_':
            return self.scan_identifier_or_keyword()

        # String literals
        if ch == '"':
            return self.scan_string()

        # Character literals
        if ch == "'":
            return self.scan_char()

        # Two-character operators (check before single-char)
        two_char = self.source[self.pos:self.pos + 2]
        if two_char in self.TWO_CHAR_OPS:
            self.advance()
            self.advance()
            return self.make_token(self.TWO_CHAR_OPS[two_char], two_char,
                                   line=start_line, col=start_col)

        # Single-character operators and delimiters
        if ch in self.SINGLE_CHAR_OPS:
            self.advance()
            return self.make_token(self.SINGLE_CHAR_OPS[ch], ch,
                                   line=start_line, col=start_col)

        # Error: unexpected character
        self.advance()
        self.errors.append(
            f"{self.filename}:{start_line}:{start_col}: "
            f"unexpected character '{ch}'"
        )
        return self.make_token(TT.ERROR, ch, line=start_line, col=start_col)

    def tokenize(self) -> List[Token]:
        """Tokenize the entire source and return a list of tokens."""
        tokens = []
        while True:
            tok = self.next_token()
            tokens.append(tok)
            if tok.type == TT.EOF:
                break
        return tokens


# ============================================================
# Example usage
# ============================================================

source_code = """
// Compute factorial iteratively
int factorial(int n) {
    int result = 1;
    for (int i = 2; i <= n; i++) {
        result *= i;
    }
    return result;
}

/* Multi-line
   comment */
int main() {
    int x = factorial(10);
    float pi = 3.14159;
    char ch = 'A';
    char newline = '\\n';

    if (x >= 3628800 && pi != 0.0) {
        return 0;
    } else {
        return -1;
    }
}
"""

print("=== Complete Lexer Demo ===\n")
lexer = Lexer(source_code, "example.c")
tokens = lexer.tokenize()

for tok in tokens:
    print(f"  {tok}")

if lexer.errors:
    print("\nErrors:")
    for err in lexer.errors:
        print(f"  {err}")
else:
    print(f"\nTokenization successful: {len(tokens)} tokens, no errors.")

# Summary statistics
from collections import Counter
type_counts = Counter(tok.type for tok in tokens)
print("\nToken type distribution:")
for tt, count in type_counts.most_common():
    print(f"  {tt.name:20s} {count}")
```

---

## 9. 렉서 생성기(Lexer Generator) 명세

직접 작성하는 렉서는 최대한의 제어를 제공하지만, **렉서 생성기**(Lex, Flex, PLY)는 토큰 패턴 명세를 받아 자동으로 렉서 코드를 생성하여 작업을 자동화합니다.

### Lex/Flex 명세 형식

```lex
%{
/* C declarations */
#include "parser.tab.h"
%}

/* Definitions */
DIGIT   [0-9]
LETTER  [a-zA-Z_]
ID      {LETTER}({LETTER}|{DIGIT})*

%%
/* Rules: pattern    action */
"if"            { return KW_IF; }
"else"          { return KW_ELSE; }
"while"         { return KW_WHILE; }
"return"        { return KW_RETURN; }
"int"           { return KW_INT; }
"float"         { return KW_FLOAT; }

{DIGIT}+        { yylval.ival = atoi(yytext); return INT_LIT; }
{DIGIT}+"."{DIGIT}+ { yylval.fval = atof(yytext); return FLOAT_LIT; }
{ID}            { yylval.sval = strdup(yytext); return IDENT; }

"<="            { return LEQ; }
">="            { return GEQ; }
"=="            { return EQ; }
"!="            { return NEQ; }

"+"             { return PLUS; }
"-"             { return MINUS; }
"*"             { return STAR; }
"/"             { return SLASH; }
"="             { return ASSIGN; }

"("             { return LPAREN; }
")"             { return RPAREN; }
"{"             { return LBRACE; }
"}"             { return RBRACE; }
";"             { return SEMI; }
","             { return COMMA; }

"//".*          { /* skip single-line comment */ }
[ \t\r]+        { /* skip whitespace */ }
\n              { yylineno++; }

.               { fprintf(stderr, "Unexpected: %s\n", yytext); }
%%
```

### PLY (Python Lex-Yacc) 명세

```python
"""
Token specification using PLY (Python Lex-Yacc).
Install: pip install ply
"""
import ply.lex as lex

# List of token names (required by PLY)
tokens = [
    'IDENT', 'INT_LIT', 'FLOAT_LIT', 'STRING_LIT',
    'PLUS', 'MINUS', 'STAR', 'SLASH',
    'ASSIGN', 'EQ', 'NEQ', 'LT', 'GT', 'LEQ', 'GEQ',
    'LPAREN', 'RPAREN', 'LBRACE', 'RBRACE',
    'SEMI', 'COMMA',
]

# Reserved words
reserved = {
    'if':     'KW_IF',
    'else':   'KW_ELSE',
    'while':  'KW_WHILE',
    'for':    'KW_FOR',
    'return': 'KW_RETURN',
    'int':    'KW_INT',
    'float':  'KW_FLOAT',
    'void':   'KW_VOID',
}

tokens += list(reserved.values())

# Simple tokens (single characters)
t_PLUS    = r'\+'
t_MINUS   = r'-'
t_STAR    = r'\*'
t_SLASH   = r'/'
t_ASSIGN  = r'='
t_LPAREN  = r'\('
t_RPAREN  = r'\)'
t_LBRACE  = r'\{'
t_RBRACE  = r'\}'
t_SEMI    = r';'
t_COMMA   = r','

# Multi-character operators (functions, checked before simple tokens)
def t_LEQ(t):
    r'<='
    return t

def t_GEQ(t):
    r'>='
    return t

def t_EQ(t):
    r'=='
    return t

def t_NEQ(t):
    r'!='
    return t

def t_LT(t):
    r'<'
    return t

def t_GT(t):
    r'>'
    return t

# Float literal (must be before INT_LIT)
def t_FLOAT_LIT(t):
    r'\d+\.\d+([eE][+-]?\d+)?'
    t.value = float(t.value)
    return t

# Integer literal
def t_INT_LIT(t):
    r'\d+'
    t.value = int(t.value)
    return t

# String literal
def t_STRING_LIT(t):
    r'"[^"\\]*(\\.[^"\\]*)*"'
    t.value = t.value[1:-1]  # Strip quotes
    return t

# Identifier (and keyword check)
def t_IDENT(t):
    r'[a-zA-Z_][a-zA-Z0-9_]*'
    t.type = reserved.get(t.value, 'IDENT')  # Check for keywords
    return t

# Track line numbers
def t_newline(t):
    r'\n+'
    t.lexer.lineno += len(t.value)

# Ignored characters (whitespace)
t_ignore = ' \t\r'

# Comments
def t_COMMENT(t):
    r'//[^\n]*'
    pass  # Discard

def t_BLOCK_COMMENT(t):
    r'/\*[\s\S]*?\*/'
    t.lexer.lineno += t.value.count('\n')

# Error handling
def t_error(t):
    print(f"Illegal character '{t.value[0]}' at line {t.lineno}")
    t.lexer.skip(1)

# Build the lexer
lexer = lex.lex()

# Test it
lexer.input("int x = 42 + y * 3.14;")
while True:
    tok = lexer.token()
    if not tok:
        break
    print(tok)
```

---

## 10. 특수 사례 처리

### 이스케이프 시퀀스가 포함된 문자열 리터럴

문자열 스캔은 이스케이프 시퀀스 처리가 필요하여 패턴이 문자 수준에서 문맥 의존적이 됩니다:

```python
def scan_string_literal(source: str, pos: int) -> Tuple[str, str, int]:
    """
    Scan a string literal starting at pos (which should be '"').
    Returns (raw_lexeme, processed_value, new_pos).
    """
    assert source[pos] == '"'
    start = pos
    pos += 1  # skip opening quote
    chars = []

    ESCAPE_MAP = {
        'n': '\n',
        't': '\t',
        'r': '\r',
        '\\': '\\',
        '"': '"',
        '0': '\0',
        'a': '\a',
        'b': '\b',
        'f': '\f',
        'v': '\v',
    }

    while pos < len(source):
        ch = source[pos]
        if ch == '"':
            pos += 1  # skip closing quote
            raw = source[start:pos]
            return (raw, ''.join(chars), pos)
        elif ch == '\\':
            pos += 1
            if pos >= len(source):
                raise LexerError("Unterminated escape in string", 0, 0)
            esc = source[pos]
            if esc in ESCAPE_MAP:
                chars.append(ESCAPE_MAP[esc])
            elif esc == 'x':
                # Hex escape: \xNN
                hex_str = source[pos+1:pos+3]
                chars.append(chr(int(hex_str, 16)))
                pos += 2
            elif esc == 'u':
                # Unicode escape: \uNNNN
                hex_str = source[pos+1:pos+5]
                chars.append(chr(int(hex_str, 16)))
                pos += 4
            else:
                chars.append(esc)  # Unknown escape, keep literal
            pos += 1
        elif ch == '\n':
            raise LexerError("Unterminated string (newline)", 0, 0)
        else:
            chars.append(ch)
            pos += 1

    raise LexerError("Unterminated string (EOF)", 0, 0)
```

### 여러 줄 문자열(Multi-Line Strings)

일부 언어는 여러 줄 문자열을 지원합니다:

```python
def scan_multiline_string(source: str, pos: int) -> Tuple[str, str, int]:
    """Scan a triple-quoted string (Python-style)."""
    assert source[pos:pos+3] == '"""'
    start = pos
    pos += 3  # skip opening """

    while pos < len(source) - 2:
        if source[pos:pos+3] == '"""':
            pos += 3
            raw = source[start:pos]
            value = raw[3:-3]  # strip quotes
            return (raw, value, pos)
        pos += 1

    raise LexerError("Unterminated multi-line string", 0, 0)
```

### 중첩 주석(Nested Comments)

일부 언어(Haskell, Swift, Rust)는 중첩 주석을 지원합니다:

```python
def skip_nested_comment(source: str, pos: int) -> int:
    """Skip a nested block comment /* ... /* ... */ ... */"""
    assert source[pos:pos+2] == '/*'
    pos += 2
    depth = 1

    while pos < len(source) - 1 and depth > 0:
        if source[pos:pos+2] == '/*':
            depth += 1
            pos += 2
        elif source[pos:pos+2] == '*/':
            depth -= 1
            pos += 2
        else:
            pos += 1

    if depth > 0:
        raise LexerError("Unterminated nested comment", 0, 0)

    return pos
```

### 의미 있는 공백 (Python 스타일 들여쓰기)

Python의 렉서는 공백 변화를 기반으로 `INDENT`와 `DEDENT` 토큰을 생성합니다:

```python
def tokenize_with_indentation(source: str) -> List[Token]:
    """
    Tokenize with Python-style indentation handling.
    Generates INDENT and DEDENT tokens.
    """
    tokens = []
    indent_stack = [0]  # Stack of indentation levels
    lines = source.split('\n')

    for line_no, line in enumerate(lines, 1):
        # Skip blank lines
        stripped = line.lstrip()
        if not stripped or stripped.startswith('#'):
            continue

        # Calculate indentation level
        indent = len(line) - len(stripped)

        if indent > indent_stack[-1]:
            indent_stack.append(indent)
            tokens.append(Token(TT.IDENT, "<INDENT>", line=line_no, column=1))
        else:
            while indent < indent_stack[-1]:
                indent_stack.pop()
                tokens.append(Token(TT.IDENT, "<DEDENT>", line=line_no, column=1))
            if indent != indent_stack[-1]:
                raise LexerError(
                    f"Inconsistent indentation at line {line_no}", line_no, 1
                )

        # Tokenize the rest of the line normally
        # (omitted for brevity -- use the main lexer)

    # Emit remaining DEDENTs
    while len(indent_stack) > 1:
        indent_stack.pop()
        tokens.append(Token(TT.IDENT, "<DEDENT>", line=len(lines), column=1))

    return tokens
```

---

## 11. 성능 고려 사항

### 테이블 구동(Table-Driven) vs. 직접 코딩(Direct-Coded) 렉서

**테이블 구동** (Lex/Flex로 생성):
- 전이 테이블을 2차원 배열로 저장: `next_state = table[state][char]`
- 간결하고 생성하기 쉬움
- 문자마다 간접 메모리 접근 발생

**직접 코딩** (수동 작성 또는 re2c):
- 전이를 `switch` 문 또는 `goto` 체인으로 인코딩
- 더 빠름: 분기 예측이 더 잘 작동하고, 테이블 조회 없음
- 코드 크기가 더 큼

```c
// Table-driven (simplified)
int state = 0;
while (state != -1) {
    char c = *input++;
    state = table[state][c];  // indirect lookup
}

// Direct-coded
s0:
    char c = *input++;
    if (c >= 'a' && c <= 'z') goto s1;
    if (c >= '0' && c <= '9') goto s2;
    goto error;
s1:
    ...
```

**re2c**는 직접 코딩 렉서를 생성하며, 일반적으로 Flex보다 2~3배 빠릅니다.

### 버퍼 관리(Buffer Management)

파일 시스템에서 문자를 한 번에 하나씩 읽는 것은 느립니다. 렉서는 **이중 버퍼링(Double Buffering)**을 사용합니다:

```
+-----------------------------------+-----------------------------------+
| Buffer 1 (4096 bytes)             | Buffer 2 (4096 bytes)             |
+-----------------------------------+-----------------------------------+
                    ^                                    ^
                lexeme_begin                          forward

When forward reaches the end of Buffer 1, load next block into Buffer 2.
When forward reaches the end of Buffer 2, load next block into Buffer 1.
```

이를 **센티넬(Sentinel)** 기법이라 합니다: 각 버퍼 끝에 특수 EOF 문자를 배치하여 문자마다 경계 검사를 하지 않아도 됩니다.

---

## 요약

- **렉서**는 컴파일의 첫 번째 단계로, 문자를 토큰으로 변환합니다.
- **토큰**은 타입과 선택적 속성 값을 가지며, **어휘소(Lexeme)**는 실제 텍스트, **패턴(Pattern)**은 규칙입니다.
- 토큰 패턴은 합집합, 연접, 클리니 스타를 지원하는 **정규 표현식**으로 명세됩니다.
- **톰슨 구성법**은 정규 표현식을 $O(n)$ 시간과 공간으로 NFA로 변환합니다.
- **부분집합 구성법**은 NFA를 DFA로 변환합니다. 최악의 경우 $2^n$개의 상태이지만 실제로는 훨씬 적습니다.
- **홉크로프트 알고리즘**은 $O(n \log n)$ 시간으로 DFA를 최소화합니다.
- **최장 일치 규칙**은 렉서가 항상 가능한 가장 긴 토큰을 반환하도록 보장합니다.
- **우선순위 규칙**은 패턴 간의 동점을 해결합니다(키워드가 식별자보다 우선).
- 직접 작성하는 렉서는 유연성을 제공하고, 렉서 생성기(Lex, Flex, PLY)는 편의성을 제공합니다.
- 특수 사례로는 문자열 이스케이프, 중첩 주석, 의미 있는 공백이 있습니다.

---

## 연습 문제

### 연습 1: 정규 표현식

다음 각 항목에 대한 정규 표현식을 작성하세요:

1. 0의 개수가 짝수인 이진 문자열
2. 밑줄(_)로 시작하는 C 스타일 식별자
3. 이메일 주소 (간략화: `name@domain.tld`)
4. 과학적 표기법의 부동소수점 수 (예: `1.5e-3`, `-2.0E+10`)
5. C 스타일 주석: `/* ... */` (비중첩)

### 연습 2: 톰슨 구성법

정규 표현식 `a(b|c)*d`에 톰슨 구성법을 적용하세요. NFA를 그리고(모든 상태에 레이블 표시), 다음 입력에 대해 NFA를 시뮬레이션하세요:
- `"ad"` -- 수락해야 하는가?
- `"abcd"` -- 수락해야 하는가?
- `"abbd"` -- 수락해야 하는가?
- `"abc"` -- 수락해야 하는가?

### 연습 3: 부분집합 구성법

연습 2의 NFA에 부분집합 구성 알고리즘을 적용하세요. DFA 전이 테이블을 작성하고, 결과 DFA의 상태 수는 몇 개인지 구하세요.

### 연습 4: 렉서 확장

섹션 8의 완전한 렉서를 확장하여 다음을 처리하도록 수정하세요:
1. 16진수 정수 리터럴 (예: `0xFF`, `0x1A3`)
2. 8진수 정수 리터럴 (예: `0o77`, `0o12`)
3. 2진수 정수 리터럴 (예: `0b1010`, `0b11001`)
4. 가독성을 위한 숫자 구분자 (예: `1_000_000`, `0xFF_FF`)

### 연습 5: 오류 복구

다음 기능을 갖춘 오류 복구 전략을 설계하고 구현하세요:
1. 문자열이 시작된 줄 번호와 함께 끝나지 않은 문자열 리터럴을 보고
2. 흔한 오타에 대한 수정 제안 (예: `retrun` -> `return`)
3. 연속된 불법 문자들을 개별적으로 보고하지 않고 하나의 오류 메시지로 그룹화

### 연습 6: 성능 비교

다음을 수행하는 간단한 벤치마크를 구현하세요:
1. 무작위 토큰으로 대용량(1MB 이상) 소스 파일 생성
2. 섹션 8의 직접 작성 렉서로 토큰화
3. 결합 정규 표현식을 사용한 Python의 `re` 모듈로 토큰화
4. 토큰화 시간을 비교하고 두 결과가 동일한 토큰을 생성하는지 검증

---

[이전: 컴파일러 개론](./01_Introduction_to_Compilers.md) | [다음: 유한 오토마톤](./03_Finite_Automata.md) | [개요](./00_Overview.md)
