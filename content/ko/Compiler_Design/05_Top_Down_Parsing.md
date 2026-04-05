# 레슨 5: 하향식 파싱(Top-Down Parsing)

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. **이해**: 하향식 파싱의 원리와 파스 트리를 루트에서 아래로 구성하는 방법
2. **계산**: 임의의 문맥 자유 문법에 대한 FIRST 및 FOLLOW 집합 계산
3. **구성**: LL(1) 파싱 테이블을 구성하고 LL(1) 문법 식별
4. **구현**: 소규모 언어에 대한 재귀 하강 파서(recursive descent parser) 직접 구현
5. **구현**: 구성된 파싱 테이블에서 테이블 구동 LL(1) 파서 구현
6. **제거**: 좌재귀(left recursion)를 제거하고 좌인수분해(left factoring)를 적용하여 문법을 LL(1) 호환으로 만들기
7. **해소**: LL(1) 충돌 해소 및 오류 복구 전략 적용
8. **설명**: LL(k)와 ALL(*) 파싱이 LL(1)을 넘어 제공하는 확장 기능 설명

---

## 1. 하향식 파싱 소개

하향식 파싱(top-down parsing)은 **루트**(시작 기호)에서 파스 트리를 구성하여 **잎**(단말 기호) 방향으로 내려가는 파싱 전략입니다. 각 단계에서 파서는 현재 입력 토큰을 기반으로 적용할 생성 규칙을 예측합니다.

하향식 파싱의 핵심 질문:

> 파싱 스택 맨 위에 비단말 $A$가 있고 현재 룩어헤드(lookahead) 토큰이 $a$일 때, 파서는 어떤 생성 규칙 $A \to \alpha$를 적용해야 하는가?

하향식 파서는 두 가지 주요 형태로 나뉩니다:

1. **재귀 하강 파서(Recursive descent parser)** -- 비단말당 하나씩, 상호 재귀적인 프로시저들의 모음
2. **테이블 구동 예측 파서(Table-driven predictive parser)** -- LL(1) 파싱 테이블과 명시적 스택으로 구동

두 형태 모두 문법이 **LL(1)** (또는 그에 가깝게)이어야 합니다. LL(1)이라는 이름은:

- **L**: 입력을 **왼쪽(Left)**에서 오른쪽으로 스캔
- **L**: **최좌단(Leftmost)** 유도 생성
- **1**: **1**개의 토큰 룩어헤드 사용

```
Top-Down Parse Tree Construction (for input "a + b * c"):

Step 1:         E              Step 2:         E
               /|\                            /|\
              ? ? ?                          T  E'
                                            /
                                           ?

Step 3:         E              Step 4:         E
               /|\                            /|\
              T  E'                          T  E'
             /   |                          /|  |\
            F    T'                        F T' + T E'
            |                              |  |
            a    ...                       a  ε  ...
```

파서는 **최좌단 유도(leftmost derivation)** 방식으로 진행합니다: 매 단계마다 가장 왼쪽의 비단말을 전개합니다.

---

## 2. 재귀 하강 파싱(Recursive Descent Parsing)

### 2.1 기본 개념

재귀 하강 파서에서, 문법의 각 비단말은 하나의 함수에 대응합니다. 비단말 $A$에 대한 함수는 현재 룩어헤드 토큰을 검사하여 사용할 생성 규칙을 결정한 다음, 선택된 생성 규칙 우변의 각 기호에 대한 함수를 호출합니다.

좌재귀 제거 후의 고전적인 표현식 문법을 고려합니다:

$$
\begin{aligned}
E &\to T \; E' \\
E' &\to + \; T \; E' \;\mid\; \varepsilon \\
T &\to F \; T' \\
T' &\to * \; F \; T' \;\mid\; \varepsilon \\
F &\to ( \; E \; ) \;\mid\; \textbf{id}
\end{aligned}
$$

### 2.2 구현

산술 표현식을 위한 완전한 재귀 하강 파서:

```python
"""
Recursive Descent Parser for Arithmetic Expressions

Grammar (LL(1)-compatible):
    E  -> T E'
    E' -> + T E' | ε
    T  -> F T'
    T' -> * F T' | ε
    F  -> ( E ) | id | num
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional


# ─── Token Types ───

class TokenType(Enum):
    PLUS = auto()
    STAR = auto()
    LPAREN = auto()
    RPAREN = auto()
    ID = auto()
    NUM = auto()
    EOF = auto()


@dataclass
class Token:
    type: TokenType
    value: str
    pos: int  # position in input for error reporting


# ─── Lexer ───

class Lexer:
    """Simple tokenizer for arithmetic expressions."""

    def __init__(self, text: str):
        self.text = text
        self.pos = 0

    def _skip_whitespace(self):
        while self.pos < len(self.text) and self.text[self.pos].isspace():
            self.pos += 1

    def next_token(self) -> Token:
        self._skip_whitespace()

        if self.pos >= len(self.text):
            return Token(TokenType.EOF, "", self.pos)

        ch = self.text[self.pos]
        start = self.pos

        if ch == '+':
            self.pos += 1
            return Token(TokenType.PLUS, "+", start)
        elif ch == '*':
            self.pos += 1
            return Token(TokenType.STAR, "*", start)
        elif ch == '(':
            self.pos += 1
            return Token(TokenType.LPAREN, "(", start)
        elif ch == ')':
            self.pos += 1
            return Token(TokenType.RPAREN, ")", start)
        elif ch.isdigit():
            while self.pos < len(self.text) and self.text[self.pos].isdigit():
                self.pos += 1
            return Token(TokenType.NUM, self.text[start:self.pos], start)
        elif ch.isalpha() or ch == '_':
            while self.pos < len(self.text) and (
                self.text[self.pos].isalnum() or self.text[self.pos] == '_'
            ):
                self.pos += 1
            return Token(TokenType.ID, self.text[start:self.pos], start)
        else:
            raise SyntaxError(
                f"Unexpected character '{ch}' at position {start}"
            )

    def tokenize(self) -> list[Token]:
        tokens = []
        while True:
            tok = self.next_token()
            tokens.append(tok)
            if tok.type == TokenType.EOF:
                break
        return tokens


# ─── AST Nodes ───

@dataclass
class ASTNode:
    pass

@dataclass
class NumLit(ASTNode):
    value: int

@dataclass
class Ident(ASTNode):
    name: str

@dataclass
class BinOp(ASTNode):
    op: str
    left: ASTNode
    right: ASTNode


# ─── Recursive Descent Parser ───

class RecursiveDescentParser:
    """
    Parses arithmetic expressions using recursive descent.

    Each nonterminal in the grammar has a corresponding method:
        E  -> T E'        =>  parse_E()
        E' -> + T E' | ε  =>  parse_E_prime(left)
        T  -> F T'        =>  parse_T()
        T' -> * F T' | ε  =>  parse_T_prime(left)
        F  -> ( E ) | id  =>  parse_F()
    """

    def __init__(self, tokens: list[Token]):
        self.tokens = tokens
        self.pos = 0

    @property
    def current(self) -> Token:
        return self.tokens[self.pos]

    def eat(self, expected: TokenType) -> Token:
        """Consume the current token if it matches the expected type."""
        tok = self.current
        if tok.type != expected:
            raise SyntaxError(
                f"Expected {expected.name} but found {tok.type.name} "
                f"('{tok.value}') at position {tok.pos}"
            )
        self.pos += 1
        return tok

    def parse(self) -> ASTNode:
        """Entry point: parse the entire expression."""
        tree = self.parse_E()
        self.eat(TokenType.EOF)
        return tree

    def parse_E(self) -> ASTNode:
        """E -> T E'"""
        left = self.parse_T()
        return self.parse_E_prime(left)

    def parse_E_prime(self, left: ASTNode) -> ASTNode:
        """E' -> + T E' | ε"""
        if self.current.type == TokenType.PLUS:
            self.eat(TokenType.PLUS)
            right = self.parse_T()
            node = BinOp("+", left, right)
            return self.parse_E_prime(node)  # left-associate
        else:
            # ε-production: return what we have
            return left

    def parse_T(self) -> ASTNode:
        """T -> F T'"""
        left = self.parse_F()
        return self.parse_T_prime(left)

    def parse_T_prime(self, left: ASTNode) -> ASTNode:
        """T' -> * F T' | ε"""
        if self.current.type == TokenType.STAR:
            self.eat(TokenType.STAR)
            right = self.parse_F()
            node = BinOp("*", left, right)
            return self.parse_T_prime(node)  # left-associate
        else:
            return left

    def parse_F(self) -> ASTNode:
        """F -> ( E ) | id | num"""
        if self.current.type == TokenType.LPAREN:
            self.eat(TokenType.LPAREN)
            node = self.parse_E()
            self.eat(TokenType.RPAREN)
            return node
        elif self.current.type == TokenType.NUM:
            tok = self.eat(TokenType.NUM)
            return NumLit(int(tok.value))
        elif self.current.type == TokenType.ID:
            tok = self.eat(TokenType.ID)
            return Ident(tok.name if hasattr(tok, 'name') else tok.value)
        else:
            raise SyntaxError(
                f"Unexpected token {self.current.type.name} "
                f"('{self.current.value}') at position {self.current.pos}"
            )


# ─── Pretty Printer ───

def pretty_print(node: ASTNode, indent: int = 0) -> str:
    """Produce a human-readable representation of the AST."""
    prefix = "  " * indent
    if isinstance(node, NumLit):
        return f"{prefix}NumLit({node.value})"
    elif isinstance(node, Ident):
        return f"{prefix}Ident({node.name})"
    elif isinstance(node, BinOp):
        left_str = pretty_print(node.left, indent + 1)
        right_str = pretty_print(node.right, indent + 1)
        return (
            f"{prefix}BinOp({node.op})\n"
            f"{left_str}\n"
            f"{right_str}"
        )
    return f"{prefix}Unknown({node})"


# ─── Demo ───

if __name__ == "__main__":
    examples = [
        "3 + 4 * 5",
        "(a + b) * c",
        "x + y + z",
        "2 * (3 + 4)",
    ]
    for expr in examples:
        print(f"\nInput: {expr}")
        tokens = Lexer(expr).tokenize()
        print(f"Tokens: {[(t.type.name, t.value) for t in tokens]}")
        ast = RecursiveDescentParser(tokens).parse()
        print(f"AST:\n{pretty_print(ast)}")
```

**"3 + 4 * 5"에 대한 출력:**

```
Input: 3 + 4 * 5
Tokens: [('NUM', '3'), ('PLUS', '+'), ('NUM', '4'), ('STAR', '*'), ('NUM', '5'), ('EOF', '')]
AST:
BinOp(+)
  NumLit(3)
  BinOp(*)
    NumLit(4)
    NumLit(5)
```

### 2.3 장점과 한계

**장점:**

- 직접 이해하고 구현하기 쉬움
- 자연스러운 오류 메시지(각 함수가 기대하는 것을 알고 있음)
- 의미 동작(AST 구성, 타입 검사)을 쉽게 삽입 가능
- 수동 역추적(backtracking)으로 일부 비LL(1) 구문 처리 가능

**한계:**

- 문법이 (대략) LL(1)이어야 함
- 좌재귀 문법을 직접 처리할 수 없음
- 과도한 역추적으로 성능이 저하될 수 있음
- 대규모 파서를 직접 유지하는 것은 오류 가능성이 높음

---

## 3. FIRST 집합과 FOLLOW 집합

FIRST 집합과 FOLLOW 집합은 예측 파싱을 가능하게 하는 핵심 자료 구조입니다. 이들은 두 가지 본질적인 질문에 답합니다:

- **FIRST($\alpha$)**: $\alpha$에서 유도되는 문자열의 첫 번째 기호로 나타날 수 있는 단말은 무엇인가?
- **FOLLOW($A$)**: 어떤 문장 형식에서 비단말 $A$ 바로 다음에 나타날 수 있는 단말은 무엇인가?

### 3.1 FIRST 집합

**정의.** 문법 기호들의 문자열 $\alpha$에 대해, $\text{FIRST}(\alpha)$는 $\alpha$에서 유도되는 문자열의 시작이 될 수 있는 단말들의 집합입니다. $\alpha \Rightarrow^* \varepsilon$이면 $\varepsilon \in \text{FIRST}(\alpha)$입니다.

**FIRST 집합 계산 알고리즘:**

```
Algorithm: Compute FIRST(X) for all grammar symbols X

1. If X is a terminal:
       FIRST(X) = {X}

2. If X is a nonterminal:
       For each production X -> Y1 Y2 ... Yk:
           Add FIRST(Y1) - {ε} to FIRST(X)
           If ε ∈ FIRST(Y1):
               Add FIRST(Y2) - {ε} to FIRST(X)
               If ε ∈ FIRST(Y2):
                   Add FIRST(Y3) - {ε} to FIRST(X)
                   ...continue until Yi where ε ∉ FIRST(Yi)
           If ε ∈ FIRST(Yi) for all i = 1..k:
               Add ε to FIRST(X)

3. If X -> ε is a production:
       Add ε to FIRST(X)

Repeat until no FIRST set changes.
```

**문자열에 대한 FIRST 확장:**

문자열 $\alpha = X_1 X_2 \cdots X_n$에 대해:

$$
\text{FIRST}(X_1 X_2 \cdots X_n) = \begin{cases}
\text{FIRST}(X_1) & \text{if } \varepsilon \notin \text{FIRST}(X_1) \\
(\text{FIRST}(X_1) - \{\varepsilon\}) \cup \text{FIRST}(X_2 \cdots X_n) & \text{if } \varepsilon \in \text{FIRST}(X_1)
\end{cases}
$$

### 3.2 FOLLOW 집합

**정의.** 비단말 $A$에 대해, $\text{FOLLOW}(A)$는 어떤 문장 형식에서 $A$ 바로 오른쪽에 나타날 수 있는 단말들의 집합입니다. 입력 끝 표시자 $\$$는 시작 기호 $S$의 $\text{FOLLOW}(S)$에 포함됩니다.

**FOLLOW 집합 계산 알고리즘:**

```
Algorithm: Compute FOLLOW(A) for all nonterminals A

1. Add $ to FOLLOW(S), where S is the start symbol.

2. For each production A -> α B β:
       Add FIRST(β) - {ε} to FOLLOW(B)

3. For each production A -> α B, or A -> α B β where ε ∈ FIRST(β):
       Add FOLLOW(A) to FOLLOW(B)

Repeat steps 2-3 until no FOLLOW set changes.
```

### 3.3 Python 구현

```python
"""
FIRST and FOLLOW Set Computation

This module computes FIRST and FOLLOW sets for any context-free grammar,
which are essential for constructing LL(1) parsing tables.
"""

from typing import Optional

# We use a special sentinel for epsilon
EPSILON = "ε"
EOF = "$"


class Grammar:
    """
    Represents a context-free grammar.

    Attributes:
        productions: dict mapping nonterminal -> list of right-hand sides
                     Each RHS is a list of symbols (strings)
        terminals: set of terminal symbols
        nonterminals: set of nonterminal symbols
        start: the start symbol
    """

    def __init__(self):
        self.productions: dict[str, list[list[str]]] = {}
        self.terminals: set[str] = set()
        self.nonterminals: set[str] = set()
        self.start: Optional[str] = None

    def add_production(self, lhs: str, rhs: list[str]):
        """Add a production rule LHS -> RHS."""
        if lhs not in self.productions:
            self.productions[lhs] = []
        self.productions[lhs].append(rhs)
        self.nonterminals.add(lhs)

        if self.start is None:
            self.start = lhs

        for symbol in rhs:
            if symbol != EPSILON and symbol not in self.productions:
                # Tentatively mark as terminal; may be reclassified later
                pass

    def finalize(self):
        """Call after all productions are added to classify symbols."""
        all_symbols = set()
        for lhs, rhs_list in self.productions.items():
            for rhs in rhs_list:
                for sym in rhs:
                    if sym != EPSILON:
                        all_symbols.add(sym)
        self.terminals = all_symbols - self.nonterminals

    def __repr__(self):
        lines = []
        for lhs, rhs_list in self.productions.items():
            for rhs in rhs_list:
                lines.append(f"  {lhs} -> {' '.join(rhs)}")
        return "Grammar:\n" + "\n".join(lines)


def compute_first(grammar: Grammar) -> dict[str, set[str]]:
    """
    Compute FIRST sets for all symbols in the grammar.

    Returns:
        Dictionary mapping each symbol to its FIRST set.
    """
    first: dict[str, set[str]] = {}

    # Initialize FIRST for terminals
    for t in grammar.terminals:
        first[t] = {t}

    # Initialize FIRST for nonterminals
    for nt in grammar.nonterminals:
        first[nt] = set()

    changed = True
    while changed:
        changed = False

        for lhs, rhs_list in grammar.productions.items():
            for rhs in rhs_list:
                # Handle epsilon production
                if rhs == [EPSILON]:
                    if EPSILON not in first[lhs]:
                        first[lhs].add(EPSILON)
                        changed = True
                    continue

                # Process each symbol in the RHS
                all_have_epsilon = True
                for symbol in rhs:
                    symbol_first = first.get(symbol, set())

                    # Add FIRST(symbol) - {ε} to FIRST(lhs)
                    additions = symbol_first - {EPSILON}
                    if not additions.issubset(first[lhs]):
                        first[lhs] |= additions
                        changed = True

                    # If ε not in FIRST(symbol), stop
                    if EPSILON not in symbol_first:
                        all_have_epsilon = False
                        break

                # If all symbols can derive ε, add ε
                if all_have_epsilon:
                    if EPSILON not in first[lhs]:
                        first[lhs].add(EPSILON)
                        changed = True

    return first


def first_of_string(
    symbols: list[str], first: dict[str, set[str]]
) -> set[str]:
    """
    Compute FIRST of a string of grammar symbols.

    Args:
        symbols: list of grammar symbols
        first: precomputed FIRST sets

    Returns:
        FIRST set for the symbol string
    """
    result = set()

    if not symbols or symbols == [EPSILON]:
        return {EPSILON}

    all_have_epsilon = True
    for sym in symbols:
        sym_first = first.get(sym, {sym})  # terminals map to themselves
        result |= (sym_first - {EPSILON})

        if EPSILON not in sym_first:
            all_have_epsilon = False
            break

    if all_have_epsilon:
        result.add(EPSILON)

    return result


def compute_follow(
    grammar: Grammar, first: dict[str, set[str]]
) -> dict[str, set[str]]:
    """
    Compute FOLLOW sets for all nonterminals in the grammar.

    Args:
        grammar: the grammar
        first: precomputed FIRST sets

    Returns:
        Dictionary mapping each nonterminal to its FOLLOW set.
    """
    follow: dict[str, set[str]] = {}

    for nt in grammar.nonterminals:
        follow[nt] = set()

    # Rule 1: Add $ to FOLLOW(start)
    follow[grammar.start].add(EOF)

    changed = True
    while changed:
        changed = False

        for lhs, rhs_list in grammar.productions.items():
            for rhs in rhs_list:
                if rhs == [EPSILON]:
                    continue

                for i, symbol in enumerate(rhs):
                    if symbol not in grammar.nonterminals:
                        continue

                    # beta = everything after this symbol
                    beta = rhs[i + 1:]

                    if beta:
                        # Rule 2: Add FIRST(β) - {ε} to FOLLOW(symbol)
                        first_beta = first_of_string(beta, first)
                        additions = first_beta - {EPSILON}
                        if not additions.issubset(follow[symbol]):
                            follow[symbol] |= additions
                            changed = True

                        # Rule 3: If ε ∈ FIRST(β), add FOLLOW(LHS)
                        if EPSILON in first_beta:
                            if not follow[lhs].issubset(follow[symbol]):
                                follow[symbol] |= follow[lhs]
                                changed = True
                    else:
                        # Symbol is at the end: Rule 3
                        if not follow[lhs].issubset(follow[symbol]):
                            follow[symbol] |= follow[lhs]
                            changed = True

    return follow


# ─── Example Usage ───

def build_expression_grammar() -> Grammar:
    """
    Build the classic expression grammar (LL(1)):
        E  -> T E'
        E' -> + T E' | ε
        T  -> F T'
        T' -> * F T' | ε
        F  -> ( E ) | id
    """
    g = Grammar()
    g.add_production("E",  ["T", "E'"])
    g.add_production("E'", ["+", "T", "E'"])
    g.add_production("E'", [EPSILON])
    g.add_production("T",  ["F", "T'"])
    g.add_production("T'", ["*", "F", "T'"])
    g.add_production("T'", [EPSILON])
    g.add_production("F",  ["(", "E", ")"])
    g.add_production("F",  ["id"])
    g.finalize()
    return g


if __name__ == "__main__":
    grammar = build_expression_grammar()
    print(grammar)
    print()

    first = compute_first(grammar)
    print("FIRST sets:")
    for sym in sorted(grammar.nonterminals):
        print(f"  FIRST({sym}) = {{ {', '.join(sorted(first[sym]))} }}")
    print()

    follow = compute_follow(grammar, first)
    print("FOLLOW sets:")
    for sym in sorted(grammar.nonterminals):
        print(f"  FOLLOW({sym}) = {{ {', '.join(sorted(follow[sym]))} }}")
```

**예상 출력:**

```
FIRST sets:
  FIRST(E)  = { (, id }
  FIRST(E') = { +, ε }
  FIRST(F)  = { (, id }
  FIRST(T)  = { (, id }
  FIRST(T') = { *, ε }

FOLLOW sets:
  FOLLOW(E)  = { ), $ }
  FOLLOW(E') = { ), $ }
  FOLLOW(F)  = { +, *, ), $ }
  FOLLOW(T)  = { +, ), $ }
  FOLLOW(T') = { +, ), $ }
```

### 3.4 계산 예시

표현식 문법에 대한 FIRST 집합 계산을 단계별로 추적합니다.

**반복 1:**

| 비단말 | 생성 규칙 | FIRST 추가 | 반복 후 FIRST 집합 |
|-------------|-----------|-----------------|-----------------|
| $E'$ | $E' \to \varepsilon$ | $\varepsilon$ | $\{\varepsilon\}$ |
| $T'$ | $T' \to \varepsilon$ | $\varepsilon$ | $\{\varepsilon\}$ |
| $F$ | $F \to (\ E\ )$ | $($ | $\{(\}$ |
| $F$ | $F \to \textbf{id}$ | $\textbf{id}$ | $\{(, \textbf{id}\}$ |

**반복 2:**

| 비단말 | 생성 규칙 | 근거 | 반복 후 FIRST 집합 |
|-------------|-----------|-----------|-----------------|
| $T$ | $T \to F\ T'$ | $\text{FIRST}(F) = \{(, \textbf{id}\}$ | $\{(, \textbf{id}\}$ |
| $E'$ | $E' \to +\ T\ E'$ | $\text{FIRST}(+) = \{+\}$ | $\{+, \varepsilon\}$ |
| $T'$ | $T' \to *\ F\ T'$ | $\text{FIRST}(*) = \{*\}$ | $\{*, \varepsilon\}$ |

**반복 3:**

| 비단말 | 생성 규칙 | 근거 | 반복 후 FIRST 집합 |
|-------------|-----------|-----------|-----------------|
| $E$ | $E \to T\ E'$ | $\text{FIRST}(T) = \{(, \textbf{id}\}$ | $\{(, \textbf{id}\}$ |

**반복 4:** 변화 없음. 고정점(fixed point) 도달.

---

## 4. LL(1) 파싱 테이블 구성

### 4.1 알고리즘

LL(1) 파싱 테이블 $M$은 비단말(행)과 단말 및 $\$$(열)로 인덱싱된 2차원 배열입니다. 각 항목 $M[A, a]$는 적용할 생성 규칙을 포함합니다.

**알고리즘: LL(1) 테이블 구성**

```
For each production A -> α in the grammar:
    1. For each terminal a in FIRST(α):
           Add "A -> α" to M[A, a]

    2. If ε ∈ FIRST(α):
           For each terminal b in FOLLOW(A):
               Add "A -> α" to M[A, b]
           If $ ∈ FOLLOW(A):
               Add "A -> α" to M[A, $]
```

**정의.** 문법이 **LL(1)**이려면 파싱 테이블 $M$의 모든 항목에 생성 규칙이 최대 하나만 있어야 합니다.

### 4.2 Python 구현

```python
def build_ll1_table(
    grammar: Grammar,
    first: dict[str, set[str]],
    follow: dict[str, set[str]],
) -> tuple[dict[tuple[str, str], list[str]], list[str]]:
    """
    Build the LL(1) parsing table.

    Returns:
        (table, conflicts) where:
        - table: dict mapping (nonterminal, terminal) -> production RHS
        - conflicts: list of conflict descriptions (empty if grammar is LL(1))
    """
    table: dict[tuple[str, str], list[list[str]]] = {}
    conflicts: list[str] = []

    for lhs, rhs_list in grammar.productions.items():
        for rhs in rhs_list:
            # Compute FIRST of the RHS
            rhs_first = first_of_string(rhs, first)

            # Rule 1: For each terminal a in FIRST(rhs)
            for a in rhs_first:
                if a == EPSILON:
                    continue
                key = (lhs, a)
                if key not in table:
                    table[key] = []
                table[key].append(rhs)

            # Rule 2: If ε in FIRST(rhs), for each b in FOLLOW(lhs)
            if EPSILON in rhs_first:
                for b in follow[lhs]:
                    key = (lhs, b)
                    if key not in table:
                        table[key] = []
                    table[key].append(rhs)

    # Check for conflicts
    final_table: dict[tuple[str, str], list[str]] = {}
    for key, prods in table.items():
        if len(prods) > 1:
            conflict_strs = [" ".join(p) for p in prods]
            conflicts.append(
                f"Conflict at M[{key[0]}, {key[1]}]: "
                f"{' vs '.join(conflict_strs)}"
            )
        final_table[key] = prods[0]

    return final_table, conflicts


def print_ll1_table(
    table: dict[tuple[str, str], list[str]],
    grammar: Grammar,
):
    """Pretty-print the LL(1) parsing table."""
    terminals_list = sorted(grammar.terminals) + [EOF]
    nonterminals_list = sorted(grammar.nonterminals)

    # Compute column widths
    col_width = max(
        max(len(t) for t in terminals_list),
        max(
            len(" ".join(table.get((nt, t), [])))
            for nt in nonterminals_list
            for t in terminals_list
        ),
        8
    ) + 2

    # Header
    header = f"{'':>6}" + "".join(f"{t:>{col_width}}" for t in terminals_list)
    print(header)
    print("-" * len(header))

    # Rows
    for nt in nonterminals_list:
        row = f"{nt:>6}"
        for t in terminals_list:
            entry = table.get((nt, t))
            if entry:
                cell = f"{nt}->{' '.join(entry)}"
            else:
                cell = ""
            row += f"{cell:>{col_width}}"
        print(row)


if __name__ == "__main__":
    grammar = build_expression_grammar()
    first = compute_first(grammar)
    follow = compute_follow(grammar, first)
    table, conflicts = build_ll1_table(grammar, first, follow)

    if conflicts:
        print("LL(1) conflicts found:")
        for c in conflicts:
            print(f"  {c}")
    else:
        print("Grammar is LL(1)!")

    print("\nLL(1) Parsing Table:")
    print_ll1_table(table, grammar)
```

### 4.3 표현식 문법 테이블

표현식 문법의 LL(1) 테이블:

```
             (          )          *          +         id          $
────────────────────────────────────────────────────────────────────────
   E     E->T E'                                    E->T E'
   E'               E'->ε                E'->+T E'              E'->ε
   F     F->(E)                                     F->id
   T     T->F T'                                    T->F T'
   T'               T'->ε     T'->*F T'   T'->ε                 T'->ε
```

**테이블 읽는 법:** 파서의 스택 맨 위에 비단말 $E$가 있고 룩어헤드가 `id`이면, 생성 규칙 $E \to T\ E'$를 적용합니다.

---

## 5. 테이블 구동 LL(1) 파서

### 5.1 파싱 알고리즘

테이블 구동 LL(1) 파서는 명시적 스택과 파싱 테이블을 사용하여 최좌단 유도를 시뮬레이션합니다.

```
Algorithm: Table-Driven LL(1) Parsing

Input:  string w$, parsing table M
Output: leftmost derivation of w or error

Initialize stack = [S, $]  (start symbol on top)
Set ip to first symbol of w$

repeat:
    let X = top of stack, a = current input symbol

    if X is a terminal:
        if X == a:
            pop X from stack
            advance ip          ("match")
        else:
            error()

    else if X == $:
        if a == $:
            accept()            ("success")
        else:
            error()

    else:  (X is a nonterminal)
        if M[X, a] is defined as X -> Y1 Y2 ... Yk:
            pop X from stack
            push Yk, ..., Y2, Y1 onto stack  (Y1 on top)
            output "X -> Y1 Y2 ... Yk"
        else:
            error()
```

### 5.2 완전한 구현

```python
"""
Table-Driven LL(1) Parser

Implements a complete LL(1) parser using the parsing table constructed
from FIRST and FOLLOW sets.
"""


class LL1Parser:
    """
    A table-driven LL(1) parser.

    Uses an explicit stack and a parsing table to parse input strings.
    """

    def __init__(
        self,
        grammar: Grammar,
        table: dict[tuple[str, str], list[str]],
    ):
        self.grammar = grammar
        self.table = table

    def parse(self, tokens: list[str], verbose: bool = False) -> bool:
        """
        Parse a list of terminal symbols.

        Args:
            tokens: list of terminal symbols (without $)
            verbose: if True, print each step

        Returns:
            True if the input is accepted, False otherwise.
        """
        # Add end marker
        input_symbols = tokens + [EOF]
        ip = 0  # input pointer

        # Initialize stack with $ and start symbol
        stack = [EOF, self.grammar.start]

        step = 0
        if verbose:
            print(f"{'Step':>4}  {'Stack':<30} {'Input':<30} {'Action'}")
            print("-" * 90)

        while True:
            top = stack[-1]
            current = input_symbols[ip]
            remaining = " ".join(input_symbols[ip:])

            if verbose:
                stack_str = " ".join(reversed(stack))
                print(
                    f"{step:>4}  {stack_str:<30} {remaining:<30}",
                    end="",
                )

            # Case 1: Top is $ (end marker)
            if top == EOF:
                if current == EOF:
                    if verbose:
                        print("ACCEPT")
                    return True
                else:
                    if verbose:
                        print("ERROR: unexpected input after stack empty")
                    return False

            # Case 2: Top is a terminal
            if top in self.grammar.terminals:
                if top == current:
                    stack.pop()
                    ip += 1
                    if verbose:
                        print(f"Match '{top}'")
                else:
                    if verbose:
                        print(
                            f"ERROR: expected '{top}', found '{current}'"
                        )
                    return False

            # Case 3: Top is a nonterminal
            elif top in self.grammar.nonterminals:
                key = (top, current)
                if key in self.table:
                    production = self.table[key]
                    stack.pop()

                    # Push RHS in reverse order (so first symbol is on top)
                    if production != [EPSILON]:
                        for symbol in reversed(production):
                            stack.append(symbol)

                    prod_str = " ".join(production)
                    if verbose:
                        print(f"Apply {top} -> {prod_str}")
                else:
                    if verbose:
                        print(
                            f"ERROR: no entry for M[{top}, {current}]"
                        )
                    return False

            else:
                if verbose:
                    print(f"ERROR: unknown symbol '{top}'")
                return False

            step += 1

            # Safety: prevent infinite loops
            if step > 10000:
                if verbose:
                    print("ERROR: too many steps, possible infinite loop")
                return False


# ─── Demo ───

def demo_ll1_parser():
    """Demonstrate the LL(1) parser on the expression grammar."""
    grammar = build_expression_grammar()
    first = compute_first(grammar)
    follow = compute_follow(grammar, first)
    table, conflicts = build_ll1_table(grammar, first, follow)

    parser = LL1Parser(grammar, table)

    # Test inputs
    tests = [
        (["id", "+", "id", "*", "id"], True),
        (["(", "id", "+", "id", ")", "*", "id"], True),
        (["id", "+", "+"], False),
        (["(", "id"], False),
    ]

    for tokens, expected in tests:
        print(f"\n{'='*90}")
        print(f"Input: {' '.join(tokens)}")
        print(f"{'='*90}")
        result = parser.parse(tokens, verbose=True)
        status = "ACCEPTED" if result else "REJECTED"
        assert result == expected, f"Expected {expected}, got {result}"
        print(f"\nResult: {status}")


if __name__ == "__main__":
    demo_ll1_parser()
```

**"id + id * id"에 대한 추적 예시:**

```
Step  Stack                          Input                          Action
------------------------------------------------------------------------------------------
   0  E $                            id + id * id $                 Apply E -> T E'
   1  T E' $                         id + id * id $                 Apply T -> F T'
   2  F T' E' $                      id + id * id $                 Apply F -> id
   3  id T' E' $                     id + id * id $                 Match 'id'
   4  T' E' $                        + id * id $                    Apply T' -> ε
   5  E' $                           + id * id $                    Apply E' -> + T E'
   6  + T E' $                       + id * id $                    Match '+'
   7  T E' $                         id * id $                      Apply T -> F T'
   8  F T' E' $                      id * id $                      Apply F -> id
   9  id T' E' $                     id * id $                      Match 'id'
  10  T' E' $                        * id $                         Apply T' -> * F T'
  11  * F T' E' $                    * id $                         Match '*'
  12  F T' E' $                      id $                           Apply F -> id
  13  id T' E' $                     id $                           Match 'id'
  14  T' E' $                        $                              Apply T' -> ε
  15  E' $                           $                              Apply E' -> ε
  16  $                              $                              ACCEPT
```

---

## 6. 좌재귀 제거(Left Recursion Elimination)

### 6.1 문제

문법에 $A \Rightarrow^+ A\alpha$인 어떤 문자열 $\alpha$에 대한 비단말 $A$가 존재하면 **좌재귀(left-recursive)**입니다. 하향식 파서는 좌재귀를 처리할 수 없습니다. $A$를 파싱하면 즉시 $A$를 다시 파싱해야 하므로 무한 재귀가 발생합니다.

**직접 좌재귀(Immediate left recursion)**는 다음 형태입니다:

$$A \to A\alpha_1 \mid A\alpha_2 \mid \cdots \mid A\alpha_m \mid \beta_1 \mid \beta_2 \mid \cdots \mid \beta_n$$

여기서 $\beta_i$는 $A$로 시작하지 않습니다.

### 6.2 직접 좌재귀 제거

**변환 규칙:**

다음을:
$$A \to A\alpha_1 \mid A\alpha_2 \mid \cdots \mid \beta_1 \mid \beta_2 \mid \cdots$$

다음으로 교체합니다:
$$
\begin{aligned}
A &\to \beta_1 A' \mid \beta_2 A' \mid \cdots \\
A' &\to \alpha_1 A' \mid \alpha_2 A' \mid \cdots \mid \varepsilon
\end{aligned}
$$

**예시:** 좌재귀 표현식 문법:

$$E \to E + T \mid T$$

다음이 됩니다:

$$
\begin{aligned}
E &\to T\ E' \\
E' &\to +\ T\ E' \mid \varepsilon
\end{aligned}
$$

### 6.3 간접 좌재귀 제거

간접(또는 숨겨진) 좌재귀는 순환이 여러 비단말을 포함할 때 발생합니다:

$$
\begin{aligned}
A &\to B\alpha \\
B &\to A\beta
\end{aligned}
$$

여기서 $A \Rightarrow B\alpha \Rightarrow A\beta\alpha$이므로 $A$는 간접적으로 좌재귀적입니다.

**알고리즘: 모든 좌재귀 제거**

```
Input: Grammar G with no cycles or ε-productions
       (except ε-productions introduced by the algorithm)

1. Order the nonterminals: A1, A2, ..., An

2. For i = 1 to n:
       For j = 1 to i-1:
           Replace each production Ai -> Aj γ with:
               Ai -> δ1 γ | δ2 γ | ... | δk γ
           where Aj -> δ1 | δ2 | ... | δk are all Aj-productions

       Eliminate immediate left recursion from Ai productions
```

### 6.4 Python 구현

```python
def eliminate_immediate_left_recursion(
    lhs: str, productions: list[list[str]]
) -> tuple[str, list[list[str]], list[list[str]]]:
    """
    Eliminate immediate left recursion from productions of a nonterminal.

    Args:
        lhs: the nonterminal
        productions: list of RHS alternatives

    Returns:
        (new_nonterminal, new_productions_for_lhs, productions_for_new)
    """
    left_recursive = []
    non_recursive = []

    for rhs in productions:
        if rhs[0] == lhs:
            # A -> A α  =>  α part
            left_recursive.append(rhs[1:])
        else:
            non_recursive.append(rhs)

    if not left_recursive:
        # No left recursion to eliminate
        return lhs, productions, []

    # Create new nonterminal A'
    new_nt = lhs + "'"

    # A -> β1 A' | β2 A' | ...
    new_lhs_prods = []
    for beta in non_recursive:
        if beta == [EPSILON]:
            new_lhs_prods.append([new_nt])
        else:
            new_lhs_prods.append(beta + [new_nt])

    # A' -> α1 A' | α2 A' | ... | ε
    new_nt_prods = []
    for alpha in left_recursive:
        new_nt_prods.append(alpha + [new_nt])
    new_nt_prods.append([EPSILON])

    return new_nt, new_lhs_prods, new_nt_prods


def eliminate_all_left_recursion(grammar: Grammar) -> Grammar:
    """
    Eliminate all left recursion (immediate and indirect) from a grammar.

    Returns a new grammar with no left recursion.
    """
    new_grammar = Grammar()
    nonterminals = list(grammar.nonterminals)

    # Working copy of productions
    current_prods: dict[str, list[list[str]]] = {
        nt: list(rhs_list) for nt, rhs_list in grammar.productions.items()
    }

    for i, ai in enumerate(nonterminals):
        # Step 1: Replace Ai -> Aj γ for j < i
        for j in range(i):
            aj = nonterminals[j]
            new_prods = []
            for rhs in current_prods[ai]:
                if rhs[0] == aj:
                    # Substitute: Ai -> Aj γ becomes Ai -> δk γ
                    gamma = rhs[1:]
                    for delta in current_prods[aj]:
                        if delta == [EPSILON]:
                            new_prods.append(gamma if gamma else [EPSILON])
                        else:
                            new_prods.append(delta + gamma)
                else:
                    new_prods.append(rhs)
            current_prods[ai] = new_prods

        # Step 2: Eliminate immediate left recursion
        new_nt, lhs_prods, nt_prods = eliminate_immediate_left_recursion(
            ai, current_prods[ai]
        )
        current_prods[ai] = lhs_prods
        if nt_prods:
            current_prods[new_nt] = nt_prods

    # Build new grammar
    for lhs, rhs_list in current_prods.items():
        for rhs in rhs_list:
            new_grammar.add_production(lhs, rhs)

    new_grammar.start = grammar.start
    new_grammar.finalize()
    return new_grammar


# Example:
if __name__ == "__main__":
    g = Grammar()
    # Left-recursive expression grammar
    g.add_production("E", ["E", "+", "T"])
    g.add_production("E", ["T"])
    g.add_production("T", ["T", "*", "F"])
    g.add_production("T", ["F"])
    g.add_production("F", ["(", "E", ")"])
    g.add_production("F", ["id"])
    g.finalize()

    print("Original grammar:")
    print(g)
    print()

    g2 = eliminate_all_left_recursion(g)
    print("After left-recursion elimination:")
    print(g2)
```

---

## 7. 좌인수분해(Left Factoring)

### 7.1 문제

좌인수분해는 동일한 비단말에 대한 두 개 이상의 생성 규칙이 공통 접두사를 공유할 때 필요합니다. 이 경우 파서는 단일 룩어헤드 토큰으로 어떤 생성 규칙을 사용할지 결정할 수 없습니다.

**예시:**

$$
\text{Stmt} \to \textbf{if}\ E\ \textbf{then}\ S\ \textbf{else}\ S \mid \textbf{if}\ E\ \textbf{then}\ S
$$

두 대안 모두 `if E then S`로 시작하므로, 단일 룩어헤드 토큰으로는 파서가 선택할 수 없습니다.

### 7.2 변환

생성 규칙 $A \to \alpha\beta_1 \mid \alpha\beta_2$에 대해, 다음으로 교체합니다:

$$
\begin{aligned}
A &\to \alpha\ A' \\
A' &\to \beta_1 \mid \beta_2
\end{aligned}
$$

**if-then-else 좌인수분해 후:**

$$
\begin{aligned}
\text{Stmt} &\to \textbf{if}\ E\ \textbf{then}\ S\ \text{Stmt'} \\
\text{Stmt'} &\to \textbf{else}\ S \mid \varepsilon
\end{aligned}
$$

### 7.3 구현

```python
def left_factor(grammar: Grammar) -> Grammar:
    """
    Apply left factoring to the grammar.

    Returns a new grammar with common prefixes factored out.
    """
    new_grammar = Grammar()
    counter = 0

    for lhs, rhs_list in grammar.productions.items():
        remaining = list(rhs_list)

        while remaining:
            # Find the longest common prefix among remaining productions
            if len(remaining) == 1:
                new_grammar.add_production(lhs, remaining[0])
                remaining = []
                continue

            # Group by first symbol
            groups: dict[str, list[list[str]]] = {}
            for rhs in remaining:
                key = rhs[0] if rhs else EPSILON
                if key not in groups:
                    groups[key] = []
                groups[key].append(rhs)

            new_remaining = []
            for first_sym, group in groups.items():
                if len(group) == 1:
                    new_grammar.add_production(lhs, group[0])
                else:
                    # Find longest common prefix
                    prefix = list(group[0])
                    for rhs in group[1:]:
                        new_prefix = []
                        for a, b in zip(prefix, rhs):
                            if a == b:
                                new_prefix.append(a)
                            else:
                                break
                        prefix = new_prefix

                    if not prefix:
                        # No common prefix; just add all
                        for rhs in group:
                            new_grammar.add_production(lhs, rhs)
                    else:
                        # Factor out the prefix
                        counter += 1
                        new_nt = f"{lhs}_F{counter}"
                        new_grammar.add_production(
                            lhs, prefix + [new_nt]
                        )
                        for rhs in group:
                            suffix = rhs[len(prefix):]
                            if not suffix:
                                suffix = [EPSILON]
                            new_grammar.add_production(new_nt, suffix)

            remaining = new_remaining

    new_grammar.start = grammar.start
    new_grammar.finalize()
    return new_grammar
```

---

## 8. LL(1) 충돌과 해소

### 8.1 충돌 유형

LL(1) 충돌은 파싱 테이블 항목 $M[A, a]$에 생성 규칙이 여러 개 있을 때 발생합니다. 이는 두 가지 이유로 발생할 수 있습니다:

**FIRST-FIRST 충돌:** $A$에 대한 두 생성 규칙이 겹치는 FIRST 집합을 가집니다.

$$A \to \alpha \mid \beta \quad \text{여기서 } \text{FIRST}(\alpha) \cap \text{FIRST}(\beta) \neq \emptyset$$

**FIRST-FOLLOW 충돌:** 널가능(nullable) 생성 규칙의 FIRST 집합이 비단말의 FOLLOW 집합과 겹칩니다.

$$A \to \alpha \mid \varepsilon \quad \text{여기서 } \text{FIRST}(\alpha) \cap \text{FOLLOW}(A) \neq \emptyset$$

### 8.2 해소 전략

| 충돌 유형 | 해소 전략 |
|---|---|
| FIRST-FIRST | 좌인수분해, 문법 재작성 |
| FIRST-FOLLOW | 문법 재구성, 또는 LL(k)/ALL(*) 사용 |
| 매달린 else | 관례: 가장 가까운 매칭되지 않은 `if`와 연결 |
| 본질적 모호성 | 모호성을 제거하기 위해 문법 재작성 |

### 8.3 고전적인 매달린 else 문제

매달린 else 문제는 아마도 가장 유명한 LL(1) 충돌입니다:

```
Stmt -> if Expr then Stmt else Stmt
      | if Expr then Stmt
      | other
```

좌인수분해 후:

```
Stmt  -> if Expr then Stmt Else | other
Else  -> else Stmt | ε
```

룩어헤드가 `else`일 때 $\text{Else}$에 대한 FIRST-FOLLOW 충돌이 여전히 있습니다: $\text{Else} \to \textbf{else}\ \text{Stmt}$와 $\text{Else} \to \varepsilon$ 모두 적용됩니다.

**해소:** 관례적으로 항상 비엡실론 생성 규칙을 선택합니다(`else`를 가장 가까운 `if`와 연결). 파싱 테이블에서 단순히 `else Stmt` 생성 규칙을 우선시합니다.

---

## 9. 오류 복구(Error Recovery)

파서가 오류(파싱 테이블에 항목 없음, 또는 단말 불일치)를 만나면, 단순히 중단하는 대신 우아하게 복구해야 합니다.

### 9.1 패닉 모드 복구(Panic Mode Recovery)

패닉 모드는 가장 단순하고 가장 일반적으로 사용되는 오류 복구 전략입니다. 오류가 발생하면 파서는 **동기화 토큰(synchronization token)**을 찾을 때까지 입력 기호를 버립니다. 동기화 토큰은 일반적으로 전개 중인 비단말의 FOLLOW 집합의 원소입니다.

```python
def panic_mode_recovery(
    self,
    nonterminal: str,
    input_symbols: list[str],
    ip: int,
    follow: dict[str, set[str]],
) -> int:
    """
    Panic mode error recovery.

    Skip input tokens until we find one in FOLLOW(nonterminal),
    then pop the nonterminal from the stack and continue.

    Returns the new input pointer position.
    """
    sync_set = follow.get(nonterminal, set())
    print(
        f"  Error: skipping input, looking for sync tokens "
        f"in FOLLOW({nonterminal}) = {sync_set}"
    )

    while ip < len(input_symbols):
        if input_symbols[ip] in sync_set:
            print(f"  Synchronized on '{input_symbols[ip]}'")
            return ip
        print(f"  Skipping '{input_symbols[ip]}'")
        ip += 1

    return ip
```

### 9.2 구문 수준 복구(Phrase-Level Recovery)

구문 수준 복구는 파싱 테이블의 오류 항목에 특정 오류 처리 루틴을 채워 넣습니다. 예를 들어:

| 상황 | 복구 동작 |
|---|---|
| 피연산자 누락 (`+ * x`) | 더미 피연산자 삽입 |
| 연산자 누락 (`x y`) | 더미 연산자 삽입 |
| 괄호 불균형 | 누락된 `)` 삽입 또는 여분의 `(` 폐기 |
| 세미콜론 누락 | `;` 삽입 후 계속 |

```python
# Phrase-level error recovery entries for expression grammar
ERROR_ACTIONS = {
    # M[E, +]: missing operand before +
    ("E", "+"): ("insert_id", "Missing operand before '+'"),
    # M[F, +]: missing operand
    ("F", "+"): ("insert_id", "Missing operand"),
    # M[F, *]: missing operand
    ("F", "*"): ("insert_id", "Missing operand"),
    # M[E, )]: extra closing paren
    ("E", ")"): ("skip", "Unexpected ')'"),
    # M[F, )]: missing operand before )
    ("F", ")"): ("insert_id", "Missing operand before ')'"),
}
```

### 9.3 오류 생성 규칙(Error Productions)

일부 파서 생성기는 문법에 **오류 생성 규칙(error productions)**을 추가할 수 있습니다:

```
Stmt -> error ;    // skip everything until ';'
```

파서가 어떤 정상 생성 규칙도 일치시킬 수 없을 때, `error` 토큰이 동기화 토큰(이 경우 `;`)까지 입력을 소비하며 매칭됩니다.

---

## 10. LL(k)와 ALL(*) 파싱

### 10.1 LL(k) 파싱

LL(1)은 단일 룩어헤드 토큰을 사용합니다. **LL(k)**는 이를 $k$개의 토큰 룩어헤드로 확장합니다. 파싱 테이블은 비단말과 남은 입력의 길이 $k$ 접두사로 인덱싱됩니다.

**정의.** 임의의 두 최좌단 유도에 대해 문법이 LL(k)이려면:

$$
\begin{aligned}
S &\Rightarrow^*_{lm} wA\alpha \Rightarrow_{lm} w\beta\alpha \Rightarrow^*_{lm} wx \\
S &\Rightarrow^*_{lm} wA\alpha \Rightarrow_{lm} w\gamma\alpha \Rightarrow^*_{lm} wy
\end{aligned}
$$

$\text{FIRST}_k(x) = \text{FIRST}_k(y)$이면 $\beta = \gamma$ (동일한 생성 규칙이 사용됨)여야 합니다.

**실용적 영향:** $k$가 증가할수록 파싱 테이블은 지수적으로 커집니다 ($O(|\Sigma|^k)$개의 열). 실제로 손으로 작성하는 LL 파서에서는 $k > 2$는 드뭅니다.

### 10.2 ALL(*) 파싱

**ALL(*)** (Adaptive LL)은 ANTLR 4가 사용하는 파싱 알고리즘입니다. 지수적 테이블 크기 없이 무제한 룩어헤드의 능력을 제공합니다.

**핵심 아이디어:**

1. **임의 룩어헤드**: 고정된 $k$ 대신, ALL(*)는 예측을 해소하는 데 필요한 만큼 토큰을 검사합니다
2. **증강 전이 네트워크(Augmented Transition Networks, ATNs)**: 문법은 재귀적 상태 기계의 집합으로 표현됩니다
3. **동적 분석**: 런타임에 ATN 시뮬레이션이 더 많은 문맥이 필요한 예측을 만나면, **룩어헤드 DFA** 구성을 수행합니다
4. **캐싱**: 예측이 특정 룩어헤드 문맥에 대해 해소되면, 결과가 향후 재사용을 위해 DFA로 캐싱됩니다

```
ALL(*) Decision Process:

    Regular LL(1)                     ALL(*) Adaptive
    ┌─────────┐                       ┌─────────────┐
    │ Single  │                       │  ATN        │
    │ token   │ ──resolve──▶ prod     │  simulation │
    │ lookup  │                       │             │
    └─────────┘                       │ If ambig:   │
                                      │  build DFA  │──▶ prod
                                      │  cache it   │
                                      └─────────────┘
```

**ALL(*)의 장점:**

- 임의의 결정적 문맥 자유 언어를 처리합니다
- 수동 좌인수분해나 좌재귀 제거가 필요 없습니다(ANTLR 4는 직접 좌재귀를 자동으로 처리)
- 오류 메시지가 매우 정확할 수 있습니다
- 실용적 성능은 종종 선형에 가깝습니다

**ANTLR 4 예시:**

```
// ANTLR 4 grammar -- no left factoring needed
expr : expr '*' expr    // direct left recursion is OK
     | expr '+' expr
     | '(' expr ')'
     | ID
     | NUM
     ;
```

### 10.3 하향식 파싱 전략 비교

| 기능 | LL(1) | LL(k) | ALL(*) |
|---------|-------|-------|--------|
| 룩어헤드 | 1 토큰 | $k$ 토큰 | 무제한 |
| 테이블 크기 | $O(|\Sigma|)$ | $O(|\Sigma|^k)$ | 동적 DFA |
| 좌재귀 | 제거 필요 | 제거 필요 | 직접 LR 자동 처리 |
| 좌인수분해 | 필요 | 경우에 따라 | 불필요 |
| 파서 생성기 | 직접 작성 / 단순 | 드뭄 | ANTLR 4 |
| 오류 복구 | 수동 | 수동 | 자동 (ANTLR) |
| 성능 | $O(n)$ | $O(n)$ | 일반적 $O(n)$, 최악 $O(n^2)$ |

---

## 11. 모두 합치기

완전한 파이프라인을 시연하는 완전한 미니 언어 파서를 구성합니다.

```python
"""
Complete LL(1) Parser for a Mini-Language

Language syntax:
    Program   -> StmtList
    StmtList  -> Stmt StmtList | ε
    Stmt      -> id = Expr ;
               | print ( Expr ) ;
    Expr      -> Term Expr'
    Expr'     -> + Term Expr' | ε
    Term      -> Factor Term'
    Term'     -> * Factor Term' | ε
    Factor    -> ( Expr ) | id | num
"""


class MiniLangParser:
    """LL(1) parser for a mini-language with assignments and print."""

    def __init__(self):
        self.grammar = Grammar()
        self._build_grammar()
        self.grammar.finalize()

        self.first = compute_first(self.grammar)
        self.follow = compute_follow(self.grammar, self.first)
        self.table, conflicts = build_ll1_table(
            self.grammar, self.first, self.follow
        )

        if conflicts:
            raise ValueError(f"Grammar is not LL(1): {conflicts}")

    def _build_grammar(self):
        g = self.grammar
        g.add_production("Program",   ["StmtList"])
        g.add_production("StmtList",  ["Stmt", "StmtList"])
        g.add_production("StmtList",  [EPSILON])
        g.add_production("Stmt",      ["id", "=", "Expr", ";"])
        g.add_production("Stmt",      ["print", "(", "Expr", ")", ";"])
        g.add_production("Expr",      ["Term", "Expr'"])
        g.add_production("Expr'",     ["+", "Term", "Expr'"])
        g.add_production("Expr'",     [EPSILON])
        g.add_production("Term",      ["Factor", "Term'"])
        g.add_production("Term'",     ["*", "Factor", "Term'"])
        g.add_production("Term'",     [EPSILON])
        g.add_production("Factor",    ["(", "Expr", ")"])
        g.add_production("Factor",    ["id"])
        g.add_production("Factor",    ["num"])

    def parse(self, tokens: list[str]) -> list[str]:
        """
        Parse input tokens and return the list of productions applied.
        """
        parser = LL1Parser(self.grammar, self.table)
        result = parser.parse(tokens, verbose=True)
        return result


if __name__ == "__main__":
    mini = MiniLangParser()

    # Parse: x = 3 + 4 * 5 ; print ( x ) ;
    tokens = [
        "id", "=", "num", "+", "num", "*", "num", ";",
        "print", "(", "id", ")", ";",
    ]

    print("Parsing: x = 3 + 4 * 5 ; print ( x ) ;")
    print("=" * 90)
    mini.parse(tokens)
```

---

## 12. 요약

하향식 파싱은 룩어헤드 토큰과 예측 테이블의 안내를 받아 루트에서 잎 방향으로 파스 트리를 구성합니다.

**핵심 개념:**

| 개념 | 설명 |
|---------|-------------|
| **재귀 하강(Recursive descent)** | 비단말당 하나의 함수; 가장 직관적인 파싱 방법 |
| **FIRST 집합** | 문법 기호에서 유도되는 문자열의 시작이 될 수 있는 단말들 |
| **FOLLOW 집합** | 임의의 유도에서 비단말 뒤에 나타날 수 있는 단말들 |
| **LL(1) 테이블** | (비단말, 단말) 쌍을 생성 규칙에 매핑 |
| **좌재귀 제거** | 하향식 파싱에 필요; $A \to A\alpha$ 패턴 변환 |
| **좌인수분해** | 공통 접두사를 인수분해하여 FIRST-FIRST 충돌 해소 |
| **패닉 모드** | 동기화 토큰까지 건너뛰는 오류 복구 |
| **ALL(*)** | 무제한 룩어헤드를 가진 적응형 LL 파싱 (ANTLR 4) |

**하향식 파싱을 사용해야 할 때:**

- 문법이 자연스럽게 LL(1)이거나 그에 가까울 때
- 파서를 직접 작성하고 싶을 때 (재귀 하강)
- 오류 메시지에 세밀한 제어가 필요할 때
- ANTLR 4를 사용할 때 (내부적으로 ALL(*)를 사용)

**하향식 파싱을 사용하지 말아야 할 때:**

- 제거하기 어려운 본질적 좌재귀를 가진 문법
- 우선순위 상승(precedence climbing)이나 Pratt 파싱이 더 간단한 연산자 중심 언어
- Yacc/Bison (LR 기반) 같은 파서 생성기가 이미 도구 체인에 있을 때

---

## 연습 문제

### 연습 1: FIRST 및 FOLLOW 집합 계산

다음 문법에 대한 FIRST 및 FOLLOW 집합을 계산하세요:

$$
\begin{aligned}
S &\to A\ B \\
A &\to a\ A \mid \varepsilon \\
B &\to b\ B \mid c
\end{aligned}
$$

제공된 Python 코드에 문법을 구현하고 계산을 실행하여 답을 검증하세요.

### 연습 2: LL(1) 테이블 구성

다음 문법이 주어졌을 때:

$$
\begin{aligned}
S &\to i\ E\ t\ S\ S' \mid a \\
S' &\to e\ S \mid \varepsilon \\
E &\to b
\end{aligned}
$$

(여기서 $i$ = `if`, $t$ = `then`, $e$ = `else`, $a$ = 할당, $b$ = 불리언 표현식)

1. FIRST 및 FOLLOW 집합을 계산합니다.
2. LL(1) 파싱 테이블을 구성합니다.
3. 충돌을 찾습니다. 매달린 else 문제를 어떻게 해소하겠습니까?

### 연습 3: 좌재귀 제거

다음 문법에서 모든 좌재귀를 제거하세요:

$$
\begin{aligned}
S &\to A\ a \mid b \\
A &\to A\ c \mid S\ d \mid e
\end{aligned}
$$

$A \to S\ d$가 간접 좌재귀를 생성함에 주목하세요. 알고리즘의 각 단계를 보이세요.

### 연습 4: 재귀 하강 파서 확장

2.2절에서 제공된 재귀 하강 파서를 다음을 지원하도록 확장하세요:

1. 덧셈과 같은 우선순위의 뺄셈(`-`)
2. 곱셈과 같은 우선순위의 나눗셈(`/`)
3. 단항 부정(`-x`)
4. 정수 리터럴과 변수 이름

수정된 문법과 해당 Python 코드를 작성하세요.

### 연습 5: 오류 복구

테이블 구동 LL(1) 파서에 패닉 모드 오류 복구를 구현하세요. 구현은 다음을 해야 합니다:

1. $M[A, a]$가 비어있을 때, 위치와 함께 오류를 보고합니다
2. $\text{FOLLOW}(A)$의 원소를 찾을 때까지 입력 토큰을 건너뜁니다
3. 스택에서 $A$를 팝하고 파싱을 계속합니다
4. (첫 번째 오류만이 아닌) 발견된 모든 오류를 보고합니다

입력 `"id + * id"` (`+`와 `*` 사이에 피연산자 누락)로 테스트하세요.

### 연습 6: 문법 설계 도전

다음 언어 기능에 대한 LL(1) 문법을 설계하세요:

- 변수 선언: `let x = expr;`
- 할당: `x = expr;`
- if-else 구문: `if (expr) { stmts } else { stmts }`
- while 루프: `while (expr) { stmts }`
- print 구문: `print(expr);`
- `+`, `-`, `*`, `/`를 사용한 산술 표현식

1. 문법을 작성합니다.
2. FIRST 및 FOLLOW 집합을 계산하여 LL(1)인지 검증합니다.
3. LL(1)이 아니면 좌인수분해 및/또는 좌재귀 제거를 적용합니다.

---

[이전: 문맥 자유 문법](./04_Context_Free_Grammars.md) | [다음: 상향식 파싱](./06_Bottom_Up_Parsing.md) | [개요](./00_Overview.md)
