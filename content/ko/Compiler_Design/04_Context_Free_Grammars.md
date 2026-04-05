# 레슨 4: 문맥 자유 문법

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 문맥 자유 문법(Context-Free Grammar, CFG)의 형식적 정의를 제시할 수 있다
2. BNF 및 EBNF 표기법을 사용하여 문법을 작성할 수 있다
3. 주어진 문자열에 대한 최좌단(leftmost) 및 최우단(rightmost) 유도를 구성할 수 있다
4. 파스 트리(parse tree)를 구성하고 유도와의 관계를 이해할 수 있다
5. 문법의 모호성(ambiguity)을 식별하고 해소할 수 있다
6. 문법을 촘스키 정규형(Chomsky Normal Form, CNF) 및 그라이바흐 정규형(Greibach Normal Form, GNF)으로 변환할 수 있다
7. CYK 파싱 알고리즘을 적용할 수 있다
8. CFG와 푸시다운 오토마타(pushdown automata)의 연관성을 이해할 수 있다
9. 문맥 자유 언어에 대한 펌핑 보조 정리(pumping lemma)를 적용할 수 있다
10. Python으로 문법 조작 및 CYK 파싱을 구현할 수 있다

---

## 1. 문맥 자유 문법의 형식적 정의

**문맥 자유 문법(Context-Free Grammar, CFG)**은 4-튜플(4-tuple)입니다:

$$G = (V, \Sigma, P, S)$$

여기서:

- $V$는 **변수(variable)**(또는 **비단말(nonterminal)**이라고도 함)의 유한 집합
- $\Sigma$는 **단말 기호(terminal symbol)**(알파벳)의 유한 집합이며, $V \cap \Sigma = \emptyset$
- $P$는 **생성 규칙(production)**(규칙)의 유한 집합으로, 각 규칙은 $A \rightarrow \alpha$ 형태 ($A \in V$, $\alpha \in (V \cup \Sigma)^*$)
- $S \in V$는 **시작 기호(start symbol)**

### 표기 규약

이 레슨 전체에서 다음 표기를 사용합니다:

| 기호 | 의미 |
|--------|------------|
| $A, B, C, S$ | 변수(비단말), 대문자 |
| $a, b, c$ | 단말, 소문자 |
| $\alpha, \beta, \gamma$ | $(V \cup \Sigma)^*$의 문자열 |
| $w, x, y, z$ | $\Sigma^*$의 문자열 (단말 문자열) |
| $\epsilon$ | 빈 문자열 |

### CFG 예시

간단한 산술 표현식을 위한 문법:

$$G = (\{E, T, F\}, \{+, *, (, ), \texttt{id}\}, P, E)$$

생성 규칙 $P$:

$$E \rightarrow E + T \mid T$$
$$T \rightarrow T * F \mid F$$
$$F \rightarrow (E) \mid \texttt{id}$$

이 문법은 `+`, `*`, 괄호, 식별자를 사용한 산술 표현식의 언어를 정의하며, 일반적인 우선순위(`*`가 `+`보다 강하게 결합)를 따릅니다.

### 유도(Derivation)

**유도(derivation)**는 시작 기호를 단말 기호로 이루어진 문자열로 변환하는 단계들의 연속입니다. 각 단계에서 변수를 해당 생성 규칙의 우변으로 반복적으로 대체합니다.

$A \rightarrow \beta$가 생성 규칙이고 $\alpha A \gamma$가 문장 형식(sentential form)이면:

$$\alpha A \gamma \Rightarrow \alpha \beta \gamma$$

$\alpha \xRightarrow{*} \beta$는 0번 이상의 유도 단계를 나타냅니다.

### 문법의 언어

$G$가 생성하는 **언어**:

$$L(G) = \{w \in \Sigma^* \mid S \xRightarrow{*} w\}$$

시작 기호에서 유도 가능한 모든 단말 문자열의 집합입니다.

### Python 표현

```python
from dataclasses import dataclass, field
from typing import List, Set, Dict, Tuple, Optional


@dataclass
class Production:
    """A grammar production A -> symbols."""
    head: str                 # Left-hand side (nonterminal)
    body: Tuple[str, ...]     # Right-hand side (tuple of symbols)

    def __repr__(self):
        body_str = ' '.join(self.body) if self.body else 'ε'
        return f"{self.head} -> {body_str}"

    def __hash__(self):
        return hash((self.head, self.body))

    def __eq__(self, other):
        return (isinstance(other, Production) and
                self.head == other.head and self.body == other.body)


class Grammar:
    """Context-Free Grammar."""

    def __init__(self, variables: Set[str], terminals: Set[str],
                 productions: List[Production], start: str):
        self.variables = variables
        self.terminals = terminals
        self.productions = productions
        self.start = start

        # Index productions by head
        self.prod_index: Dict[str, List[Production]] = {}
        for p in productions:
            self.prod_index.setdefault(p.head, []).append(p)

    @classmethod
    def from_string(cls, grammar_str: str, start: str = None) -> 'Grammar':
        """
        Parse a grammar from a multi-line string.
        Format: 'A -> α | β | γ' (one nonterminal per line).
        Nonterminals are uppercase single letters or quoted names.
        Use 'ε' or 'epsilon' for epsilon productions.
        """
        variables = set()
        terminals = set()
        productions = []

        for line in grammar_str.strip().split('\n'):
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            head, _, body_str = line.partition('->')
            head = head.strip()
            variables.add(head)

            for alt in body_str.split('|'):
                alt = alt.strip()
                if alt in ('ε', 'epsilon', ''):
                    productions.append(Production(head, ()))
                else:
                    symbols = tuple(alt.split())
                    productions.append(Production(head, symbols))

        # Infer terminals
        for p in productions:
            for sym in p.body:
                if sym not in variables:
                    terminals.add(sym)

        if start is None:
            start = productions[0].head

        return cls(variables, terminals, productions, start)

    def __repr__(self):
        lines = [f"Grammar(start={self.start})"]
        for var in sorted(self.variables):
            prods = self.prod_index.get(var, [])
            bodies = [' '.join(p.body) if p.body else 'ε' for p in prods]
            lines.append(f"  {var} -> {' | '.join(bodies)}")
        return '\n'.join(lines)

    def is_terminal(self, symbol: str) -> bool:
        return symbol in self.terminals

    def is_variable(self, symbol: str) -> bool:
        return symbol in self.variables


# Example: Arithmetic expression grammar
expr_grammar = Grammar.from_string("""
E -> E + T | T
T -> T * F | F
F -> ( E ) | id
""")

print(expr_grammar)
```

---

## 2. BNF와 EBNF 표기법

### 배커스-나우르 형식(Backus-Naur Form, BNF)

**BNF**(Backus-Naur Form)는 John Backus가 ALGOL 60 보고서를 위해 도입한 문맥 자유 문법의 표준 표기법입니다.

```bnf
<expression>  ::= <expression> "+" <term> | <term>
<term>        ::= <term> "*" <factor> | <factor>
<factor>      ::= "(" <expression> ")" | <identifier>
<identifier>  ::= <letter> | <identifier> <letter> | <identifier> <digit>
<letter>      ::= "a" | "b" | ... | "z"
<digit>       ::= "0" | "1" | ... | "9"
```

BNF 표기 규약:
- 비단말은 꺾쇠 괄호로 감쌈: `<expression>`
- 단말은 따옴표로 감쌈: `"+"`
- `::=`는 "~로 정의된다"를 의미
- `|`는 대안을 구분
- 각 규칙은 하나의 비단말을 정의

### 확장 BNF(Extended BNF, EBNF)

**EBNF**는 BNF에 편의 표기를 추가하여 필요한 규칙 수를 줄입니다:

| EBNF 표기 | 의미 | BNF 동등 표현 |
|---------------|---------|----------------|
| `{X}` 또는 `X*` | X의 0번 이상 반복 | 재귀를 사용한 새 비단말 |
| `[X]` 또는 `X?` | 선택적 (0번 또는 1번) | 엡실론 대안을 가진 새 비단말 |
| `(X \| Y)` | 그룹화 | 새 비단말 |
| `X+` | X의 1번 이상 반복 | 재귀를 사용한 새 비단말 |

**예시**: EBNF로 표현한 표현식 문법:

```ebnf
expression = term { "+" term } .
term       = factor { "*" factor } .
factor     = "(" expression ")" | identifier .
identifier = letter { letter | digit } .
```

반복(`{...}`)이 명시적 재귀를 대체하므로 더 간결합니다.

### EBNF를 BNF로 변환

```
EBNF: A = α { β } γ .
BNF:  A  -> α A' γ
      A' -> β A' | ε

EBNF: A = α [ β ] γ .
BNF:  A -> α β γ | α γ

EBNF: A = α ( β | γ ) δ .
BNF:  A  -> α A' δ
      A' -> β | γ
```

### Python 구현

```python
def ebnf_to_bnf(ebnf_rules: str) -> str:
    """
    Convert EBNF notation to BNF.
    Handles: { } (repetition), [ ] (optional), ( | ) (grouping).

    This is a simplified converter for demonstration.
    """
    bnf_rules = []
    aux_counter = [0]

    def new_aux(base_name: str) -> str:
        aux_counter[0] += 1
        return f"{base_name}_{aux_counter[0]}"

    for line in ebnf_rules.strip().split('\n'):
        line = line.strip()
        if not line:
            continue

        head, _, body = line.partition('=')
        head = head.strip()
        body = body.strip().rstrip('.')

        # For simplicity, just handle the basic transformations
        # In practice, you'd need a proper EBNF parser

        # Handle { X } -> X_rep where X_rep -> X X_rep | ε
        import re
        while '{' in body:
            match = re.search(r'\{([^{}]+)\}', body)
            if not match:
                break
            inner = match.group(1).strip()
            aux = new_aux(head + "_rep")
            body = body[:match.start()] + aux + body[match.end():]
            bnf_rules.append(f"{aux} -> {inner} {aux} | ε")

        # Handle [ X ] -> X_opt where X_opt -> X | ε
        while '[' in body:
            match = re.search(r'\[([^\[\]]+)\]', body)
            if not match:
                break
            inner = match.group(1).strip()
            aux = new_aux(head + "_opt")
            body = body[:match.start()] + aux + body[match.end():]
            bnf_rules.append(f"{aux} -> {inner} | ε")

        bnf_rules.insert(0, f"{head} -> {body}")

    return '\n'.join(bnf_rules)


ebnf = """
expr = term { + term } .
term = factor { * factor } .
factor = ( expr ) | id .
"""

print("=== EBNF to BNF Conversion ===\n")
print("EBNF:")
print(ebnf)
print("BNF:")
print(ebnf_to_bnf(ebnf))
```

---

## 3. 유도(Derivations)

### 최좌단 유도(Leftmost Derivation)

**최좌단 유도(leftmost derivation)**에서는 각 단계마다 문장 형식에서 **가장 왼쪽** 변수를 대체합니다.

**표기**: $\alpha \xRightarrow{lm} \beta$

**예시**: 표현식 문법을 사용하여 `id + id * id`를 유도합니다.

$$E \xRightarrow{lm} E + T \xRightarrow{lm} T + T \xRightarrow{lm} F + T \xRightarrow{lm} \texttt{id} + T$$
$$\xRightarrow{lm} \texttt{id} + T * F \xRightarrow{lm} \texttt{id} + F * F \xRightarrow{lm} \texttt{id} + \texttt{id} * F \xRightarrow{lm} \texttt{id} + \texttt{id} * \texttt{id}$$

### 최우단 유도(Rightmost Derivation)

**최우단 유도(rightmost derivation)**(또는 **정규 유도(canonical derivation)**라고도 함)에서는 각 단계마다 **가장 오른쪽** 변수를 대체합니다.

**표기**: $\alpha \xRightarrow{rm} \beta$

$$E \xRightarrow{rm} E + T \xRightarrow{rm} E + T * F \xRightarrow{rm} E + T * \texttt{id} \xRightarrow{rm} E + F * \texttt{id}$$
$$\xRightarrow{rm} E + \texttt{id} * \texttt{id} \xRightarrow{rm} T + \texttt{id} * \texttt{id} \xRightarrow{rm} F + \texttt{id} * \texttt{id} \xRightarrow{rm} \texttt{id} + \texttt{id} * \texttt{id}$$

두 유도 모두 동일한 파스 트리를 생성합니다(비모호 문법에서).

### 유도 순서가 중요한 이유

- **하향식 파서(top-down parser)**(LL)는 최좌단 유도를 구성합니다
- **상향식 파서(bottom-up parser)**(LR)는 최우단 유도를 역순으로 구성합니다
- 비모호 문법에서 두 방식 모두 동일한 파스 트리를 산출합니다
- 모호 문법에서는 다른 유도 순서가 다른 파스 트리를 산출할 수 있습니다

### Python 구현

```python
def leftmost_derivation(grammar: Grammar, target: str) -> Optional[List[str]]:
    """
    Find a leftmost derivation of target string.
    Uses brute-force search (only practical for short strings).
    Returns list of sentential forms, or None if not derivable.
    """
    target_symbols = tuple(target.split())

    # BFS
    from collections import deque

    queue = deque()
    queue.append(((grammar.start,), [(grammar.start,)]))
    visited = {(grammar.start,)}

    max_len = len(target_symbols) * 3  # Bound to prevent infinite search

    while queue:
        current, history = queue.popleft()

        # Check if we've derived the target
        if current == target_symbols:
            return [' '.join(sf) for sf in history]

        # If current is all terminals, no more derivations possible
        if all(grammar.is_terminal(s) for s in current):
            continue

        # If too long, skip
        if len(current) > max_len:
            continue

        # Find leftmost variable
        for i, sym in enumerate(current):
            if grammar.is_variable(sym):
                # Try all productions for this variable
                for prod in grammar.prod_index.get(sym, []):
                    new_form = current[:i] + prod.body + current[i+1:]
                    if new_form not in visited:
                        visited.add(new_form)
                        queue.append((new_form, history + [new_form]))
                break  # Only expand leftmost variable

    return None


# Example
print("=== Leftmost Derivation ===\n")
derivation = leftmost_derivation(expr_grammar, "id + id * id")
if derivation:
    for i, step in enumerate(derivation):
        arrow = "  " if i == 0 else "=>"
        print(f"  {arrow} {step}")
```

---

## 4. 파스 트리(Parse Trees)

**파스 트리(parse tree)**(또는 **유도 트리(derivation tree)** 혹은 **구체 구문 트리(concrete syntax tree)**라고도 함)는 유도의 그래픽 표현입니다.

### 정의

문법 $G = (V, \Sigma, P, S)$에 대해 파스 트리는 다음 특성을 가집니다:

1. **루트(root)**는 시작 기호 $S$로 레이블됩니다
2. 각 **내부 노드(interior node)**는 변수 $A \in V$로 레이블됩니다
3. 각 **잎(leaf)**은 단말 $a \in \Sigma$ 또는 $\epsilon$으로 레이블됩니다
4. 내부 노드 $A$가 자식 $X_1, X_2, \ldots, X_k$를 가지면, $A \rightarrow X_1 X_2 \cdots X_k \in P$

### 파스 트리 예시

표현식 문법으로 `id + id * id`에 대한 파스 트리:

```
             E
           / | \
          E  +  T
          |    /|\
          T  T  *  F
          |  |     |
          F  F    id
          |  |
         id  id
```

잎을 왼쪽에서 오른쪽으로 읽으면 유도된 문자열 `id + id * id`가 나옵니다.

### 유도와의 관계

- 모든 유도는 정확히 하나의 파스 트리에 대응합니다(비모호 문법에서)
- 모든 파스 트리는 정확히 하나의 최좌단 유도와 하나의 최우단 유도에 대응합니다
- 서로 다른 유도(최좌단, 최우단, 또는 임의 순서)가 **동일한** 파스 트리에 대응할 수 있습니다

### Python 구현

```python
@dataclass
class ParseTreeNode:
    """A node in a parse tree."""
    symbol: str
    children: List['ParseTreeNode'] = field(default_factory=list)
    is_terminal: bool = False

    def __repr__(self):
        if self.is_terminal:
            return f"'{self.symbol}'"
        return f"{self.symbol}"

    def leaves(self) -> List[str]:
        """Return the leaves (yield) of the parse tree."""
        if self.is_terminal:
            return [self.symbol] if self.symbol != 'ε' else []
        result = []
        for child in self.children:
            result.extend(child.leaves())
        return result

    def pretty_print(self, prefix="", is_last=True):
        """Print the tree in a readable format."""
        connector = "└── " if is_last else "├── "
        label = f"'{self.symbol}'" if self.is_terminal else self.symbol
        print(prefix + connector + label)
        new_prefix = prefix + ("    " if is_last else "│   ")
        for i, child in enumerate(self.children):
            child.pretty_print(new_prefix, i == len(self.children) - 1)


def build_parse_tree(grammar: Grammar, derivation_steps: list) -> ParseTreeNode:
    """
    Build a parse tree from a leftmost derivation.
    Each step is a sentential form (list of symbols).
    """
    # Start with root
    root = ParseTreeNode(grammar.start)

    # Track which nodes need expansion
    def find_leftmost_unexpanded(node):
        """Find the leftmost nonterminal leaf that hasn't been expanded."""
        if node.is_terminal:
            return None
        if not node.children:
            if grammar.is_variable(node.symbol):
                return node
            return None
        for child in node.children:
            result = find_leftmost_unexpanded(child)
            if result is not None:
                return result
        return None

    # Process each derivation step
    for i in range(1, len(derivation_steps)):
        current = derivation_steps[i].split()
        prev = derivation_steps[i-1].split()

        # Find which variable was expanded
        node = find_leftmost_unexpanded(root)
        if node is None:
            break

        # Find the production used
        for prod in grammar.prod_index.get(node.symbol, []):
            # Check if this production could have been applied
            expanded = []
            for j, sym in enumerate(prev):
                if sym == node.symbol and j == prev.index(node.symbol):
                    expanded.extend(prod.body if prod.body else ['ε'])
                else:
                    expanded.append(sym)

            if expanded == current or (not prod.body and len(current) == len(prev) - 1):
                # This is the right production
                if prod.body:
                    for sym in prod.body:
                        child = ParseTreeNode(
                            sym,
                            is_terminal=grammar.is_terminal(sym)
                        )
                        node.children.append(child)
                else:
                    node.children.append(
                        ParseTreeNode('ε', is_terminal=True)
                    )
                break

    return root


# Manually build a parse tree for "id + id * id"
def make_expr_tree():
    """Build the parse tree for 'id + id * id' manually."""
    # F -> id (three instances)
    id1 = ParseTreeNode('id', is_terminal=True)
    id2 = ParseTreeNode('id', is_terminal=True)
    id3 = ParseTreeNode('id', is_terminal=True)

    f1 = ParseTreeNode('F', [id1])
    f2 = ParseTreeNode('F', [id2])
    f3 = ParseTreeNode('F', [id3])

    # T -> F (left), T -> T * F (right)
    t1 = ParseTreeNode('T', [f1])
    t2 = ParseTreeNode('T', [
        ParseTreeNode('T', [f2]),
        ParseTreeNode('*', is_terminal=True),
        f3
    ])

    # E -> E + T
    plus = ParseTreeNode('+', is_terminal=True)
    e = ParseTreeNode('E', [
        ParseTreeNode('E', [t1]),
        plus,
        t2
    ])

    return e

tree = make_expr_tree()
print("=== Parse Tree for 'id + id * id' ===\n")
tree.pretty_print(prefix="", is_last=True)
print(f"\nYield: {' '.join(tree.leaves())}")
```

---

## 5. 모호성(Ambiguity)

### 정의

문법 $G$는 $L(G)$에 속하는 어떤 문자열 $w$가 **두 개 이상의 서로 다른 파스 트리**(또는 두 개 이상의 서로 다른 최좌단 유도)를 가지면 **모호(ambiguous)**합니다.

### 모호한 문법 예시

다음과 같은 더 간단한 표현식 문법을 고려합니다:

$$E \rightarrow E + E \mid E * E \mid (E) \mid \texttt{id}$$

문자열 `id + id * id`는 두 개의 파스 트리를 가집니다:

**트리 1** (곱셈이 더 강하게 결합):
```
       E
      /|\
     E + E
     |  /|\
    id E * E
       |   |
      id  id
```

해석: `id + (id * id)`

**트리 2** (덧셈이 더 강하게 결합):
```
       E
      /|\
     E * E
    /|\   |
   E + E  id
   |   |
  id  id
```

해석: `(id + id) * id`

이 두 파스 트리는 동일한 표현식에 서로 다른 의미를 부여합니다!

### 모호성이 중요한 이유

- 서로 다른 파스 트리 $\rightarrow$ 서로 다른 의미론(semantics)
- 컴파일러는 **유일한** 해석을 생성해야 합니다
- 문법의 모호성은 파싱을 비결정적(nondeterministic)으로 만듭니다

### 본질적 모호성(Inherent Ambiguity)

문맥 자유 **언어**가 **본질적으로 모호(inherently ambiguous)**하다는 것은 그 언어에 대한 모든 문법이 모호하다는 의미입니다. (이는 문법이 모호한 것과 다릅니다 -- 같은 언어에 대해 비모호 문법을 찾을 수도 있습니다.)

고전적인 예시:

$$L = \{a^i b^j c^k \mid i = j \text{ 또는 } j = k\}$$

이 언어는 본질적으로 모호합니다. 문자열 $a^n b^n c^n$은 $i = j$를 맞추거나 $j = k$를 맞추는 방식으로 파싱될 수 있으며, 이 언어에 대한 비모호 문법은 존재하지 않습니다.

### 모호성 감지

**결정 불가(Undecidable)**: 임의의 CFG가 주어질 때 그것이 모호한지 결정하는 알고리즘은 존재하지 않습니다. (이는 Post의 대응 문제(Post's Correspondence Problem)로 환원될 수 있습니다.)

그러나 특정 종류의 문법(LL(1), LR(1))은 비모호임이 보장됩니다.

```python
def check_ambiguity_brute_force(grammar: Grammar, max_length: int = 6) -> list:
    """
    Brute-force check: enumerate all leftmost derivations up to a given
    string length and check for strings with multiple derivations.

    WARNING: Exponential complexity. Only for very small grammars.
    Returns list of (string, count) for ambiguous strings.
    """
    from collections import defaultdict

    derivation_count = defaultdict(int)

    def derive(sentential_form, depth):
        """Enumerate all leftmost derivations."""
        # Check if all terminals
        if all(grammar.is_terminal(s) for s in sentential_form):
            if len(sentential_form) <= max_length:
                key = ' '.join(sentential_form)
                derivation_count[key] += 1
            return

        # Prune: too many symbols
        terminal_count = sum(1 for s in sentential_form if grammar.is_terminal(s))
        if terminal_count > max_length:
            return

        if depth > max_length * 3:
            return

        # Find leftmost variable
        for i, sym in enumerate(sentential_form):
            if grammar.is_variable(sym):
                for prod in grammar.prod_index.get(sym, []):
                    new_form = sentential_form[:i] + list(prod.body) + sentential_form[i+1:]
                    derive(new_form, depth + 1)
                break

    derive([grammar.start], 0)

    ambiguous = [(s, c) for s, c in derivation_count.items() if c > 1]
    return ambiguous


# Test with the ambiguous grammar
ambig_grammar = Grammar.from_string("""
E -> E + E | E * E | ( E ) | id
""")

print("=== Ambiguity Check ===\n")
results = check_ambiguity_brute_force(ambig_grammar, max_length=5)
if results:
    print("Ambiguous strings found:")
    for s, count in sorted(results, key=lambda x: len(x[0])):
        print(f"  '{s}' has {count} distinct leftmost derivations")
else:
    print("No ambiguity found (up to given length)")
```

---

## 6. 모호성 해소

### 전략 1: 연산자 우선순위(Operator Precedence)

우선순위 수준별로 비단말 계층을 도입합니다:

```
Level 0 (lowest):   E -> E + T | T        (addition)
Level 1:            T -> T * F | F        (multiplication)
Level 2 (highest):  F -> ( E ) | id       (atoms)
```

우선순위가 높은 연산자는 문법에서 더 깊이 위치하므로 더 강하게 결합됩니다.

네 수준의 우선순위를 가진 언어:

```
expr   -> expr || or_t | or_t            // Level 0: logical OR
or_t   -> or_t && and_t | and_t          // Level 1: logical AND
and_t  -> and_t == rel_t | rel_t         // Level 2: equality
rel_t  -> rel_t < factor | factor        // Level 3: relational
factor -> ( expr ) | id | num            // Level 4: atoms
```

### 전략 2: 결합성(Associativity)

- **좌결합(left-associative)** 연산자는 **좌재귀(left recursion)** 사용: $E \rightarrow E + T$
  - `a + b + c`는 `(a + b) + c`로 파싱됩니다

- **우결합(right-associative)** 연산자는 **우재귀(right recursion)** 사용: $E \rightarrow T = E$
  - `a = b = c`는 `a = (b = c)`로 파싱됩니다

- **비결합(non-associative)** 연산자: 같은 수준에서 재귀 없음
  - `a < b < c`는 구문 오류

```
# Left-associative: a + b + c = (a + b) + c
E -> E + T | T

# Right-associative: a = b = c = a = (b = c)
E -> T = E | T

# Non-associative: a < b is ok, a < b < c is error
E -> T < T | T
```

### 전략 3: 외부 모호성 해소 규칙

일부 파서 생성기는 우선순위와 결합성을 문법에 인코딩하는 대신 **외부적으로** 지정할 수 있습니다:

```yacc
/* Yacc/Bison precedence declarations */
%left '+' '-'         /* lowest precedence, left-associative */
%left '*' '/'         /* higher precedence, left-associative */
%right UMINUS         /* highest precedence, right-associative (unary minus) */

%%
expr : expr '+' expr
     | expr '-' expr
     | expr '*' expr
     | expr '/' expr
     | '-' expr %prec UMINUS
     | '(' expr ')'
     | NUMBER
     ;
```

이 방법은 문법을 단순하게(그리고 모호하게) 유지하면서 명시적 규칙으로 모호성을 해소합니다.

### 전략 4: 매달린 else(Dangling Else) 문제

**매달린 else**는 고전적인 모호성 문제입니다:

```
stmt -> if expr then stmt
      | if expr then stmt else stmt
      | other
```

`if E1 then if E2 then S1 else S2`에 대해 두 개의 파스 트리가 존재합니다:

```
Parse tree 1: if E1 then (if E2 then S1 else S2)     -- else matches inner if
Parse tree 2: if E1 then (if E2 then S1) else S2      -- else matches outer if
```

**해소**: 관례적으로 `else`는 **가장 가까운 매칭되지 않은** `if`와 연결됩니다. 이는 문법으로 인코딩할 수 있습니다:

```
stmt         -> matched | unmatched
matched      -> if expr then matched else matched | other
unmatched    -> if expr then stmt
              | if expr then matched else unmatched
```

또는 외부적으로 해소: 대부분의 파서 생성기는 기본적으로 `else`를 가장 가까운 `if`와 연결합니다.

### Python 예시: 모호성 해소

```python
# Ambiguous grammar for expressions
ambiguous = Grammar.from_string("""
E -> E + E | E * E | ( E ) | id
""")

# Unambiguous grammar (precedence + left-associativity)
unambiguous = Grammar.from_string("""
E -> E + T | T
T -> T * F | F
F -> ( E ) | id
""")

print("=== Ambiguous Grammar ===")
print(ambiguous)
print()

print("=== Unambiguous Grammar (same language) ===")
print(unambiguous)
print()

# Verify they generate the same strings (for small strings)
def generate_strings(grammar, max_depth=8):
    """Generate all strings derivable up to a certain derivation depth."""
    strings = set()

    def derive(form, depth):
        if depth > max_depth:
            return
        if all(grammar.is_terminal(s) for s in form):
            strings.add(tuple(form))
            return
        for i, sym in enumerate(form):
            if grammar.is_variable(sym):
                for prod in grammar.prod_index.get(sym, []):
                    new_form = form[:i] + list(prod.body) + form[i+1:]
                    derive(new_form, depth + 1)
                break

    derive([grammar.start], 0)
    return {' '.join(s) for s in strings}

strings_ambig = generate_strings(ambiguous, 6)
strings_unambig = generate_strings(unambiguous, 8)

print(f"Ambiguous grammar generates {len(strings_ambig)} strings (depth 6)")
print(f"Unambiguous grammar generates {len(strings_unambig)} strings (depth 8)")
print(f"Strings in ambiguous but not unambiguous: {strings_ambig - strings_unambig}")
print(f"Strings in unambiguous but not ambiguous: {strings_unambig - strings_ambig}")
```

---

## 7. 촘스키 정규형(Chomsky Normal Form, CNF)

CFG가 **촘스키 정규형(Chomsky Normal Form)**이려면 모든 생성 규칙이 다음 형태 중 하나여야 합니다:

1. $A \rightarrow BC$ (두 비단말, $B, C \in V$)
2. $A \rightarrow a$ ($a \in \Sigma$, 하나의 단말)
3. $S \rightarrow \epsilon$ (오직 $\epsilon \in L(G)$인 경우에만, 그리고 $S$는 어떤 우변에도 나타나지 않음)

### CNF가 중요한 이유

- **CYK 파싱 알고리즘**은 CNF를 요구합니다 (9절 참조)
- CNF의 모든 파스 트리는 **이진 트리(binary tree)**입니다 (분석에 유용)
- 모든 CFG는 동등한 CNF 문법으로 변환될 수 있습니다
- CNF에서 길이 $n$인 문자열의 유도는 정확히 $2n - 1$ 단계를 가집니다

### CNF로 변환

**입력**: CFG $G = (V, \Sigma, P, S)$
**출력**: CNF의 동등한 CFG $G'$

**단계 1: $\epsilon$-생성 규칙 제거** ($S \rightarrow \epsilon$은 필요에 따라 예외)

**널가능(nullable)** 변수 찾기: $A$가 널가능하면 $A \xRightarrow{*} \epsilon$입니다.

각 생성 규칙 $B \rightarrow \alpha$에 대해, 널가능 기호의 가능한 모든 부분 집합이 제거된 새 생성 규칙을 추가합니다. 그런 다음 모든 $\epsilon$-생성 규칙을 삭제합니다(필요한 경우 $S \rightarrow \epsilon$ 제외).

**단계 2: 단위 생성 규칙 제거** ($A \rightarrow B$)

모든 단위 쌍 찾기: $(A, B)$ (단위 생성 규칙만 사용하여 $A \xRightarrow{*} B$인 경우). 각 단위 체인을 직접 생성 규칙으로 대체합니다.

**단계 3: 긴 생성 규칙 교체**

$A \rightarrow B_1 B_2 \cdots B_k$ ($k > 2$)를 다음으로 교체합니다:

$$A \rightarrow B_1 C_1$$
$$C_1 \rightarrow B_2 C_2$$
$$\vdots$$
$$C_{k-2} \rightarrow B_{k-1} B_k$$

**단계 4: 단말-비단말 혼합 교체**

$|\alpha| \geq 2$인 생성 규칙 $A \rightarrow \alpha$에서, 각 단말 $a$를 새 변수 $T_a$로 교체하고 $T_a \rightarrow a$를 추가합니다.

### Python 구현

```python
def to_cnf(grammar: Grammar) -> Grammar:
    """
    Convert a grammar to Chomsky Normal Form.
    """
    variables = set(grammar.variables)
    terminals = set(grammar.terminals)
    productions = list(grammar.productions)
    start = grammar.start

    aux_counter = [0]
    def new_var(prefix="X"):
        aux_counter[0] += 1
        name = f"{prefix}{aux_counter[0]}"
        variables.add(name)
        return name

    # ============================================================
    # Step 0: New start symbol (ensure S doesn't appear on RHS)
    # ============================================================
    rhs_symbols = set()
    for p in productions:
        rhs_symbols.update(p.body)
    if start in rhs_symbols:
        new_start = new_var("S")
        productions.append(Production(new_start, (start,)))
        start = new_start

    # ============================================================
    # Step 1: Remove ε-productions
    # ============================================================

    # Find nullable variables
    nullable = set()
    changed = True
    while changed:
        changed = False
        for p in productions:
            if not p.body or all(s in nullable for s in p.body):
                if p.head not in nullable:
                    nullable.add(p.head)
                    changed = True

    # Generate new productions with nullable symbols optionally removed
    new_prods = set()
    for p in productions:
        if not p.body:
            # ε-production: only keep for start symbol
            if p.head == start:
                new_prods.add(p)
            continue

        # Find all positions with nullable symbols
        nullable_positions = [i for i, s in enumerate(p.body) if s in nullable]

        # Generate all subsets of nullable positions
        from itertools import combinations
        for r in range(len(nullable_positions) + 1):
            for positions_to_remove in combinations(nullable_positions, r):
                new_body = tuple(
                    s for i, s in enumerate(p.body)
                    if i not in positions_to_remove
                )
                if new_body:  # Don't add empty unless it's start
                    new_prods.add(Production(p.head, new_body))
                elif p.head == start:
                    new_prods.add(Production(p.head, ()))

    productions = list(new_prods)

    # ============================================================
    # Step 2: Remove unit productions (A -> B)
    # ============================================================

    # Find unit pairs using transitive closure
    unit_pairs = set()
    for v in variables:
        unit_pairs.add((v, v))  # Reflexive

    changed = True
    while changed:
        changed = False
        for p in productions:
            if len(p.body) == 1 and p.body[0] in variables:
                for (a, b) in list(unit_pairs):
                    if b == p.head:
                        new_pair = (a, p.body[0])
                        if new_pair not in unit_pairs:
                            unit_pairs.add(new_pair)
                            changed = True

    # Replace unit productions
    new_prods = set()
    for p in productions:
        if len(p.body) == 1 and p.body[0] in variables:
            continue  # Skip unit productions

        # For each unit pair (A, B) where B is the head of this production
        for (a, b) in unit_pairs:
            if b == p.head:
                new_prods.add(Production(a, p.body))

    productions = list(new_prods)

    # ============================================================
    # Step 3: Replace terminals in mixed productions
    # ============================================================
    term_vars = {}  # terminal -> variable name
    additional = []

    final_prods = []
    for p in productions:
        if len(p.body) <= 1:
            final_prods.append(p)
            continue

        # Replace terminals with new variables
        new_body = []
        for sym in p.body:
            if sym in terminals:
                if sym not in term_vars:
                    tv = new_var("T")
                    term_vars[sym] = tv
                    additional.append(Production(tv, (sym,)))
                new_body.append(term_vars[sym])
            else:
                new_body.append(sym)
        final_prods.append(Production(p.head, tuple(new_body)))

    productions = final_prods + additional

    # ============================================================
    # Step 4: Break long productions into binary
    # ============================================================
    final_prods = []
    for p in productions:
        if len(p.body) <= 2:
            final_prods.append(p)
            continue

        # A -> B1 B2 B3 ... Bk  =>  A -> B1 C1, C1 -> B2 C2, ..., Ck-2 -> Bk-1 Bk
        symbols = list(p.body)
        current_head = p.head
        while len(symbols) > 2:
            new_head = new_var("C")
            final_prods.append(Production(current_head, (symbols[0], new_head)))
            symbols = symbols[1:]
            current_head = new_head
        final_prods.append(Production(current_head, tuple(symbols)))

    productions = final_prods

    return Grammar(variables, terminals, productions, start)


# Example
print("=== CNF Conversion ===\n")

expr_g = Grammar.from_string("""
E -> E + T | T
T -> T * F | F
F -> ( E ) | id
""")

print("Original grammar:")
print(expr_g)
print()

cnf = to_cnf(expr_g)
print("CNF grammar:")
for p in sorted(cnf.productions, key=lambda p: (p.head, p.body)):
    print(f"  {p}")

# Verify: check that some strings are still derivable
print("\nVerification (derivability check):")
for w in ["id", "id + id", "id * id", "id + id * id"]:
    symbols = w.split()
    # We'll verify using CYK (next section)
```

---

## 8. 그라이바흐 정규형(Greibach Normal Form, GNF)

CFG가 **그라이바흐 정규형(Greibach Normal Form)**이려면 모든 생성 규칙이 다음 형태여야 합니다:

$$A \rightarrow a B_1 B_2 \cdots B_k$$

여기서 $a \in \Sigma$이고 $B_1, \ldots, B_k \in V$ ($k \geq 0$).

즉, 모든 생성 규칙은 정확히 하나의 단말로 시작하고, 그 뒤에 0개 이상의 비단말이 따라옵니다.

### 속성

1. GNF의 각 유도 단계는 정확히 하나의 단말을 소비합니다
2. 길이 $n$인 문자열은 정확히 $n$번의 유도 단계를 필요로 합니다($\epsilon$-생성 규칙 없음, $S \rightarrow \epsilon$은 가능)
3. GNF는 푸시다운 오토마타 구성에 유용합니다
4. GNF로의 변환은 좌재귀 제거가 필요합니다 (레슨 5에서 다룸)

### 정규형 비교

| 속성 | CNF | GNF |
|----------|-----|-----|
| 생성 규칙 형태 | $A \rightarrow BC$ 또는 $A \rightarrow a$ | $A \rightarrow a\alpha$ ($\alpha \in V^*$) |
| 파스 트리 형태 | 이진 트리 | 높은 분기 인수 |
| $|w| = n$에 대한 유도 길이 | $2n - 1$ | $n$ |
| 주요 용도 | CYK 알고리즘 | PDA 구성 |
| 좌재귀 허용 여부 | 허용 | 불허 |

---

## 9. CYK 알고리즘

**코크-영거-카사미(Cocke-Younger-Kasami, CYK) 알고리즘**은 CNF의 문법 $G$에 대해 문자열 $w$가 $L(G)$에 속하는지 판단하는 동적 프로그래밍 알고리즘입니다. 파스 트리도 구성합니다.

### 알고리즘

**입력**: CNF의 문법 $G$, 문자열 $w = a_1 a_2 \cdots a_n$
**출력**: $w \in L(G)$ 여부, 및 파스 테이블

**아이디어**: 삼각형 테이블 $T$를 구성합니다. $T[i][j]$는 부분 문자열 $a_i a_{i+1} \cdots a_j$를 유도할 수 있는 변수들의 집합입니다.

$$T[i][j] = \{A \in V \mid A \xRightarrow{*} a_i a_{i+1} \cdots a_j\}$$

**알고리즘**:

```
1. // Base case: substrings of length 1
   for i = 1 to n:
       T[i][i] = {A | A -> aᵢ ∈ P}

2. // Inductive case: substrings of length 2, 3, ..., n
   for length = 2 to n:
       for i = 1 to n - length + 1:
           j = i + length - 1
           T[i][j] = ∅
           for k = i to j - 1:
               for each production A -> BC:
                   if B ∈ T[i][k] and C ∈ T[k+1][j]:
                       T[i][j] = T[i][j] ∪ {A}

3. // Accept if start symbol is in top-right cell
   return S ∈ T[1][n]
```

**시간 복잡도**: $O(n^3 \cdot |P|)$
**공간 복잡도**: $O(n^2)$

### 계산 예시

CNF의 문법:

$$S \rightarrow AB \mid BC$$
$$A \rightarrow BA \mid a$$
$$B \rightarrow CC \mid b$$
$$C \rightarrow AB \mid a$$

`"baaba"` 파싱:

```
String:  b   a   a   b   a
Index:   1   2   3   4   5

Table T[i][j]:

i\j |  1      2       3       4       5
----+-------------------------------------------
 1  | {B}   {S,A}  {B}     {S,A,C} {B}
 2  |        {A,C}  {S,C}   {B}    {S,A,C}
 3  |               {A,C}   {S,A}  {B}
 4  |                        {B}   {S,A,C}
 5  |                               {A,C}
```

단계별 계산:

**길이 1** (대각선):
- $T[1][1]$: $b$는 $B \rightarrow b$와 일치, 따라서 $\{B\}$
- $T[2][2]$: $a$는 $A \rightarrow a$와 $C \rightarrow a$와 일치, 따라서 $\{A, C\}$
- 위치 3, 4, 5에 대해 마찬가지

**길이 2**:
- $T[1][2]$: $k=1$에서 분할: $T[1][1] \times T[2][2] = \{B\} \times \{A,C\}$
  - $B, A$: $? \rightarrow BA$ 확인: $A \rightarrow BA$는 $A$를, $S \rightarrow BC$는 $(B,C)$ 필요: $B \in T[1][1]$이고 $C \in T[2][2]$이므로 $S$.
  - $A \rightarrow BA$: $B \in T[1][1]$이고 $A \in T[2][2]$이므로 $A$.
  - 결과: $\{S, A\}$

(나머지 셀들에 대해 마찬가지로 계속...)

$S \in T[1][5]$이므로 문자열 `"baaba"`는 언어에 속합니다.

### Python 구현

```python
def cyk_parse(grammar: Grammar, word: str) -> Tuple[bool, list]:
    """
    CYK parsing algorithm.
    Grammar must be in Chomsky Normal Form.
    word is a space-separated string of terminals.

    Returns (accepted, table) where table[i][j] is the set of
    variables that can derive word[i..j].
    """
    symbols = word.split()
    n = len(symbols)

    if n == 0:
        # Check if S -> ε exists
        for p in grammar.prod_index.get(grammar.start, []):
            if not p.body:
                return (True, [])
        return (False, [])

    # Initialize table: T[i][j] = set of variables
    # Using 0-indexed
    T = [[set() for _ in range(n)] for _ in range(n)]

    # Also store back-pointers for parse tree construction
    # back[i][j][A] = (B, C, k) meaning A -> BC, B derives [i..k], C derives [k+1..j]
    back = [[{} for _ in range(n)] for _ in range(n)]

    # Step 1: Base case (length 1)
    for i in range(n):
        for p in grammar.productions:
            if len(p.body) == 1 and p.body[0] == symbols[i]:
                T[i][i].add(p.head)
                back[i][i][p.head] = ('terminal', symbols[i])

    # Step 2: Fill table for increasing lengths
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            for k in range(i, j):
                # Try all productions A -> BC
                for p in grammar.productions:
                    if len(p.body) == 2:
                        B, C = p.body
                        if B in T[i][k] and C in T[k+1][j]:
                            T[i][j].add(p.head)
                            if p.head not in back[i][j]:
                                back[i][j][p.head] = ('split', B, C, k)

    accepted = grammar.start in T[0][n-1]

    # Print table
    print(f"CYK Table for '{word}':\n")
    col_width = 12
    header = "i\\j |" + "|".join(f"{j:^{col_width}}" for j in range(n))
    print(header)
    print("-" * len(header))
    for i in range(n):
        cells = []
        for j in range(n):
            if j < i:
                cells.append(" " * col_width)
            else:
                cell = "{" + ",".join(sorted(T[i][j])) + "}" if T[i][j] else "∅"
                cells.append(f"{cell:^{col_width}}")
        print(f" {i}  |" + "|".join(cells))

    print(f"\nAccepted: {accepted}")

    # Build parse tree
    def build_tree(var, i, j):
        info = back[i][j].get(var)
        if info is None:
            return ParseTreeNode(var, [ParseTreeNode("?", is_terminal=True)])
        if info[0] == 'terminal':
            return ParseTreeNode(var, [ParseTreeNode(info[1], is_terminal=True)])
        else:
            _, B, C, k = info
            left = build_tree(B, i, k)
            right = build_tree(C, k+1, j)
            return ParseTreeNode(var, [left, right])

    if accepted:
        tree = build_tree(grammar.start, 0, n-1)
        print("\nParse tree:")
        tree.pretty_print()

    return (accepted, T)


# Example: CYK parsing
cnf_grammar = Grammar.from_string("""
S -> A B | B C
A -> B A | a
B -> C C | b
C -> A B | a
""")

print("=== CYK Parsing ===\n")
print("Grammar:")
print(cnf_grammar)
print()

cyk_parse(cnf_grammar, "b a a b a")
print()
cyk_parse(cnf_grammar, "a b")
print()
cyk_parse(cnf_grammar, "b b b")
```

---

## 10. 푸시다운 오토마타(Pushdown Automata) 개요

**푸시다운 오토마타(Pushdown Automaton, PDA)**는 **스택(stack)**으로 강화된 유한 오토마타입니다. PDA는 정확히 문맥 자유 언어를 인식합니다.

### 형식적 정의

PDA는 7-튜플(7-tuple)입니다:

$$M = (Q, \Sigma, \Gamma, \delta, q_0, Z_0, F)$$

여기서:

- $Q$는 상태의 유한 집합
- $\Sigma$는 입력 알파벳
- $\Gamma$는 **스택 알파벳(stack alphabet)**
- $\delta: Q \times (\Sigma \cup \{\epsilon\}) \times \Gamma \rightarrow \mathcal{P}(Q \times \Gamma^*)$는 전이 함수
- $q_0 \in Q$는 시작 상태
- $Z_0 \in \Gamma$는 초기 스택 기호
- $F \subseteq Q$는 수락 상태의 집합

### PDA의 작동 방식

각 단계에서 다음에 기반하여:
1. 현재 상태
2. 현재 입력 기호 (또는 $\epsilon$, 입력 없음)
3. 스택의 맨 위

PDA는:
1. 새로운 상태로 이동합니다
2. 선택적으로 입력 기호를 소비합니다
3. 스택의 맨 위를 0개 이상의 기호로 교체합니다

### CFG와의 동등성

**정리**: 언어가 문맥 자유인 것은 어떤 PDA에 의해 인식되는 것과 동치입니다.

**CFG $\rightarrow$ PDA**: 모든 CFG에 대해, 최좌단 유도를 시뮬레이션하는 PDA를 구성할 수 있습니다. PDA는 스택을 사용하여 유도될 남은 기호들을 추적합니다.

**PDA $\rightarrow$ CFG**: 모든 PDA에 대해, CFG를 구성할 수 있습니다. 이 구성은 "PDA가 $A$를 스택에서 팝하면서 상태 $q_i$에서 상태 $q_j$로 이동한다"를 나타내는 변수 $[q_i, A, q_j]$를 생성합니다.

### Python: 간단한 PDA 시뮬레이션

```python
class PDA:
    """Simple Pushdown Automaton simulation."""

    def __init__(self, transitions, start_state, start_stack, accept_states):
        """
        transitions: dict mapping (state, input_or_eps, stack_top) ->
                     list of (new_state, stack_push)
        stack_push is a tuple: () means pop, ('A',) means replace with A,
                                ('A', 'B') means replace top with A then push B
        """
        self.transitions = transitions
        self.start_state = start_state
        self.start_stack = start_stack
        self.accept_states = accept_states

    def accepts(self, word: str) -> bool:
        """Check if the PDA accepts the word (by final state)."""
        # Configuration: (state, remaining_input_pos, stack)
        # Use BFS to explore all nondeterministic choices
        from collections import deque

        initial = (self.start_state, 0, (self.start_stack,))
        queue = deque([initial])
        visited = set()

        symbols = list(word) if word else []

        while queue:
            state, pos, stack = queue.popleft()

            config = (state, pos, stack)
            if config in visited:
                continue
            visited.add(config)

            # Check acceptance
            if pos == len(symbols) and state in self.accept_states:
                return True

            if not stack:
                continue

            top = stack[-1]
            rest = stack[:-1]

            # Try epsilon transitions
            for new_state, push in self.transitions.get(
                (state, 'ε', top), []
            ):
                new_stack = rest + push
                if len(new_stack) < 100:  # Bound stack size
                    queue.append((new_state, pos, new_stack))

            # Try input transitions
            if pos < len(symbols):
                ch = symbols[pos]
                for new_state, push in self.transitions.get(
                    (state, ch, top), []
                ):
                    new_stack = rest + push
                    if len(new_stack) < 100:
                        queue.append((new_state, pos + 1, new_stack))

        return False


# PDA for {a^n b^n | n >= 0}
# States: q0 (push a's), q1 (pop a's matching b's), q_accept
pda_anbn = PDA(
    transitions={
        # In q0, push a's onto stack
        ('q0', 'a', 'Z'): [('q0', ('Z', 'A'))],    # Push A, keep Z
        ('q0', 'a', 'A'): [('q0', ('A', 'A'))],    # Push A
        # Switch to q1 when seeing b
        ('q0', 'b', 'A'): [('q1', ())],             # Pop A
        # In q0, accept empty string
        ('q0', 'ε', 'Z'): [('q_accept', ('Z',))],
        # In q1, pop a's matching b's
        ('q1', 'b', 'A'): [('q1', ())],             # Pop A
        # Accept when stack has only Z
        ('q1', 'ε', 'Z'): [('q_accept', ('Z',))],
    },
    start_state='q0',
    start_stack='Z',
    accept_states={'q_accept'}
)

print("=== PDA for {a^n b^n | n >= 0} ===\n")
for w in ["", "ab", "aabb", "aaabbb", "aab", "abb", "ba", "abab"]:
    result = "accept" if pda_anbn.accepts(w) else "reject"
    print(f"  '{w}': {result}")
```

---

## 11. 문맥 자유 언어에 대한 펌핑 보조 정리

### 서술

$L$이 문맥 자유 언어이면, $|w| \geq p$인 $L$의 모든 문자열 $w$를 다섯 부분으로 분할할 수 있는 상수 $p \geq 1$이 존재합니다:

$$w = uvxyz$$

다음 조건을 만족하면서:

1. $|vy| > 0$ ($v$와 $y$ 중 적어도 하나는 비어있지 않음)
2. $|vxy| \leq p$
3. $\forall i \geq 0: uv^ixy^iz \in L$ ($v$와 $y$를 함께 펌핑)

### 증명 아이디어

충분히 긴 문자열은 높이가 $|V|$ (변수의 수)보다 큰 유도 트리를 가져야 합니다. 비둘기집 원리(pigeonhole principle)에 의해, 루트에서 잎까지의 어떤 경로에서 변수 $A$가 반복되어야 합니다:

```
        S
       / \
      /   \
     /     \
    A       <-- first occurrence of A
   / \
  v   A    <-- second occurrence of A
     / \
    x   y
```

첫 번째 $A$를 루트로 하는 부분 트리는 $vxy$를 생성하고, 두 번째 $A$를 루트로 하는 부분 트리는 $x$를 생성합니다. 두 $A$ 사이의 부분을 반복하거나 제거함으로써 "펌핑"할 수 있습니다.

### CFL 펌핑 보조 정리 사용법

언어가 문맥 자유가 아님을 증명하기 위해:

1. $L$이 문맥 자유라고 가정합니다
2. $p$를 펌핑 길이라고 합니다
3. $|w| \geq p$인 특정 $w \in L$을 선택합니다
4. $|vy| > 0$이고 $|vxy| \leq p$인 **모든** 분해 $w = uvxyz$에 대해, $uv^ixy^iz \notin L$인 $i$가 존재함을 보입니다
5. 모순 도출

### 예시: $L = \{a^n b^n c^n \mid n \geq 0\}$은 문맥 자유가 아님

**증명**:

1. 펌핑 길이 $p$로 $L$이 문맥 자유라고 가정합니다.
2. $w = a^p b^p c^p \in L$을 선택합니다. $|w| = 3p \geq p$.
3. $|vy| > 0$이고 $|vxy| \leq p$인 $w = uvxyz$로 분해합니다.
4. $|vxy| \leq p$이므로, 부분 문자열 $vxy$는 세 그룹($a$, $b$, $c$) 중 최대 두 그룹에 걸칠 수 있습니다.
5. 경우 분석:
   - $vxy$가 $a^p$ 내에: 펌핑하면 $a$ 개수는 변하지만 $b$나 $c$는 변하지 않습니다.
   - $vxy$가 $a^p b^p$ 내에: 펌핑하면 $a$와/또는 $b$는 변하지만 $c$는 변하지 않습니다.
   - $vxy$가 $b^p$ 내에: 펌핑하면 $b$는 변하지만 $a$나 $c$는 변하지 않습니다.
   - $vxy$가 $b^p c^p$ 내에: 펌핑하면 $b$와/또는 $c$는 변하지만 $a$는 변하지 않습니다.
   - $vxy$가 $c^p$ 내에: 펌핑하면 $c$는 변하지만 $a$나 $b$는 변하지 않습니다.
6. 모든 경우에서, 펌핑 ($i = 2$)은 세 카운트가 불균등한 문자열을 생성하므로 $uv^2xy^2z \notin L$입니다.
7. 모순. $\blacksquare$

### 문맥 자유를 넘어서는 언어들

| 언어 | 유형 | 비문맥 자유인 이유 |
|----------|------|------------|
| $\{a^n b^n c^n \mid n \geq 0\}$ | 문맥 감응적 | 세 방향 매칭 필요 |
| $\{ww \mid w \in \{a,b\}^*\}$ | 문맥 감응적 | 복사 언어 |
| $\{a^{2^n} \mid n \geq 0\}$ | 문맥 감응적 | 지수적 증가 |
| 임의의 결정 가능한 언어 | Type 0 이하 | 튜링 기계 필요 |

참고: $\{a^n b^n \mid n \geq 0\}$은 문맥 자유입니다. 한계는 세 개 이상의 매칭된 그룹에 있습니다.

---

## 12. 촘스키 계층(Chomsky Hierarchy)

**촘스키 계층(Chomsky hierarchy)**은 형식 언어를 네 수준으로 분류합니다:

| 유형 | 이름 | 문법 제약 | 오토마타 | 예시 |
|------|------|---------------------|-----------|---------|
| 3 | 정규(Regular) | $A \rightarrow aB$ 또는 $A \rightarrow a$ | 유한 오토마타(DFA/NFA) | $a^*b^*$ |
| 2 | 문맥 자유(Context-free) | $A \rightarrow \alpha$ ($\alpha \in (V \cup \Sigma)^*$) | 푸시다운 오토마타 | $a^nb^n$ |
| 1 | 문맥 감응(Context-sensitive) | $\alpha A \beta \rightarrow \alpha \gamma \beta$ ($|\gamma| \geq 1$) | 선형 유계 오토마타 | $a^nb^nc^n$ |
| 0 | 재귀 열거 가능(Recursively enumerable) | 제한 없음 | 튜링 기계 | 정지 문제의 여(complement) |

```
Hierarchy (proper inclusions):

Regular  ⊂  Context-Free  ⊂  Context-Sensitive  ⊂  Recursively Enumerable
  (Type 3)    (Type 2)         (Type 1)              (Type 0)
```

### 컴파일러 설계와의 관련성

| 단계 | 언어 유형 | 형식론 |
|-------|-------------- |-----------|
| 어휘 분석(Lexical analysis) | 정규 (Type 3) | DFA, 정규 표현식 |
| 구문 분석(Syntax analysis) | 문맥 자유 (Type 2) | CFG, 푸시다운 오토마타 |
| 의미 분석(Semantic analysis) | 문맥 감응 (Type 1) | 속성 문법, 타입 시스템 |
| 전체 언어 의미론 | CF 이상 | 튜링 기계 (일반적으로 결정 불가) |

---

## 13. 문법 변환

### 좌재귀 제거(Eliminating Left Recursion)

문법에 $A \xRightarrow{+} A\alpha$인 어떤 $\alpha$에 대한 유도가 존재하면 **좌재귀(left recursion)**가 있습니다. 좌재귀는 하향식 파서에 문제를 일으킵니다(무한 루프).

**직접 좌재귀(Direct left recursion)** ($A \rightarrow A\alpha \mid \beta$):

다음으로 교체합니다:
$$A \rightarrow \beta A'$$
$$A' \rightarrow \alpha A' \mid \epsilon$$

**예시**:
$$E \rightarrow E + T \mid T$$

는 다음이 됩니다:
$$E \rightarrow T E'$$
$$E' \rightarrow + T E' \mid \epsilon$$

### 좌인수분해(Left Factoring)

동일한 비단말에 대한 두 생성 규칙이 공통 접두사를 공유할 때:

$$A \rightarrow \alpha \beta_1 \mid \alpha \beta_2$$

다음으로 교체합니다:
$$A \rightarrow \alpha A'$$
$$A' \rightarrow \beta_1 \mid \beta_2$$

### Python 구현

```python
def eliminate_left_recursion(grammar: Grammar) -> Grammar:
    """
    Eliminate immediate left recursion from all productions.
    (Does not handle indirect left recursion.)
    """
    new_productions = []
    new_variables = set(grammar.variables)

    for var in sorted(grammar.variables):
        prods = grammar.prod_index.get(var, [])

        # Separate left-recursive and non-left-recursive productions
        left_recursive = []  # A -> A α
        non_recursive = []   # A -> β

        for p in prods:
            if p.body and p.body[0] == var:
                left_recursive.append(p.body[1:])  # α (without leading A)
            else:
                non_recursive.append(p.body)

        if not left_recursive:
            # No left recursion for this variable
            new_productions.extend(prods)
            continue

        # Eliminate left recursion
        # A -> β1 | β2 | ... | A α1 | A α2 | ...
        # becomes:
        # A  -> β1 A' | β2 A' | ...
        # A' -> α1 A' | α2 A' | ... | ε

        new_var = var + "'"
        while new_var in new_variables:
            new_var += "'"
        new_variables.add(new_var)

        # A -> βi A'
        for beta in non_recursive:
            new_body = beta + (new_var,) if beta else (new_var,)
            new_productions.append(Production(var, new_body))

        # A' -> αi A'
        for alpha in left_recursive:
            new_body = alpha + (new_var,)
            new_productions.append(Production(new_var, new_body))

        # A' -> ε
        new_productions.append(Production(new_var, ()))

    return Grammar(
        new_variables, grammar.terminals,
        new_productions, grammar.start
    )


def left_factor(grammar: Grammar) -> Grammar:
    """
    Apply left factoring to a grammar.
    """
    new_productions = list(grammar.productions)
    new_variables = set(grammar.variables)
    aux_counter = [0]

    changed = True
    while changed:
        changed = False

        # Index by head
        prod_index = {}
        for p in new_productions:
            prod_index.setdefault(p.head, []).append(p)

        next_prods = []
        processed_vars = set()

        for var in sorted(prod_index.keys()):
            if var in processed_vars:
                continue
            prods = prod_index[var]

            # Find longest common prefix among any two productions
            best_prefix = None
            best_group = None

            for i in range(len(prods)):
                for j in range(i + 1, len(prods)):
                    # Find common prefix
                    prefix = []
                    for k in range(min(len(prods[i].body), len(prods[j].body))):
                        if prods[i].body[k] == prods[j].body[k]:
                            prefix.append(prods[i].body[k])
                        else:
                            break

                    if prefix and (best_prefix is None or len(prefix) > len(best_prefix)):
                        best_prefix = tuple(prefix)
                        # Find all productions sharing this prefix
                        group = [p for p in prods if p.body[:len(prefix)] == best_prefix]
                        best_group = group

            if best_prefix and best_group and len(best_group) > 1:
                changed = True
                processed_vars.add(var)

                # Create new variable
                aux_counter[0] += 1
                new_var = f"{var}_LF{aux_counter[0]}"
                new_variables.add(new_var)

                # A -> prefix A_LF
                next_prods.append(Production(var, best_prefix + (new_var,)))

                # A_LF -> remaining1 | remaining2 | ...
                for p in best_group:
                    remaining = p.body[len(best_prefix):]
                    next_prods.append(Production(new_var, remaining if remaining else ()))

                # Keep non-matching productions
                for p in prods:
                    if p not in best_group:
                        next_prods.append(p)
            else:
                next_prods.extend(prods)

        new_productions = next_prods

    return Grammar(
        new_variables, grammar.terminals,
        new_productions, grammar.start
    )


# Example: Eliminate left recursion
print("=== Eliminating Left Recursion ===\n")
print("Original:")
print(expr_grammar)
print()

no_lr = eliminate_left_recursion(expr_grammar)
print("After eliminating left recursion:")
print(no_lr)
print()

# Example: Left factoring
lf_grammar = Grammar.from_string("""
S -> if E then S else S | if E then S | other
E -> id
""")

print("=== Left Factoring ===\n")
print("Original:")
print(lf_grammar)
print()

factored = left_factor(lf_grammar)
print("After left factoring:")
print(factored)
```

---

## 요약

- **문맥 자유 문법(context-free grammar)** $G = (V, \Sigma, P, S)$는 유도를 통해 언어를 생성합니다
- **BNF**와 **EBNF**는 문법 작성을 위한 표준 표기법입니다
- **최좌단**과 **최우단** 유도는 서로 다른 파싱 전략에 대응합니다
- **파스 트리(parse tree)**는 유도의 구조적 표현입니다
- 문법이 **모호(ambiguous)**하다는 것은 어떤 문자열이 여러 파스 트리를 가진다는 것이며, 모호성은 일반적으로 결정 불가입니다
- 모호성은 **우선순위(precedence)**, **결합성(associativity)**, 문법 재구성을 통해 해소됩니다
- **촘스키 정규형(Chomsky Normal Form)**(이진 생성 규칙)은 $O(n^3)$에 파싱하는 **CYK 알고리즘**을 가능하게 합니다
- **그라이바흐 정규형(Greibach Normal Form)**(단말 우선 생성 규칙)은 PDA 구성에 유용합니다
- **푸시다운 오토마타(Pushdown automata)**는 정확히 문맥 자유 언어를 인식합니다
- **CFL 펌핑 보조 정리(CFL pumping lemma)**는 $\{a^n b^n c^n\}$ 같은 언어들이 문맥 자유가 아님을 증명합니다
- **촘스키 계층(Chomsky hierarchy)**은 언어를 정규, 문맥 자유, 문맥 감응, 재귀 열거 가능의 네 유형으로 분류합니다
- **좌재귀 제거(left recursion elimination)**와 **좌인수분해(left factoring)**는 하향식 파싱을 위한 문법을 준비합니다

---

## 연습 문제

### 연습 1: 문법 작성

다음 각 언어에 대한 문맥 자유 문법을 작성하세요:

1. $\{a, b\}$에 대한 회문(palindrome)
2. (중첩 가능한) 균형 잡힌 괄호
3. $\{a^i b^j c^k \mid i = j \text{ 또는 } j = k\}$
4. 단순화된 Python `if`/`elif`/`else` 구문의 부분 집합
5. `+`, `-`, `*`, `/`, 단항 `-`, 괄호를 가진 산술 표현식으로, 표준 우선순위와 좌결합성을 따름

### 연습 2: 유도와 파스 트리

문법 $E \rightarrow E + T \mid T$, $T \rightarrow T * F \mid F$, $F \rightarrow (E) \mid \texttt{id} \mid \texttt{num}$을 사용하여:

1. `id * (num + id)`에 대한 최좌단 유도를 구합니다
2. 같은 문자열에 대한 최우단 유도를 구합니다
3. 파스 트리를 그립니다
4. 두 유도가 동일한 파스 트리에 대응함을 확인합니다

### 연습 3: 모호성 분석

다음 문법을 고려합니다:

$$S \rightarrow aSb \mid aSbb \mid \epsilon$$

1. 이 문법이 생성하는 언어는 무엇입니까?
2. 이 문법은 모호합니까? 모호하다면, 두 개의 파스 트리를 가지는 문자열을 찾으세요.
3. 동일한 언어에 대한 비모호 문법을 찾을 수 있습니까?

### 연습 4: CNF 변환

다음 문법을 촘스키 정규형으로 변환하세요:

$$S \rightarrow ASA \mid aB$$
$$A \rightarrow B \mid S$$
$$B \rightarrow b \mid \epsilon$$

변환 과정의 각 단계를 보이세요.

### 연습 5: CYK 알고리즘

연습 4에서 구한 CNF 문법(또는 9절에서 제공된 것)을 사용하여 다음 문자열에 CYK 알고리즘을 실행하세요:
1. `"aabb"`
2. `"abab"`
3. `"baba"`

각 문자열에 대한 전체 테이블을 보이세요.

### 연습 6: 펌핑 보조 정리

CFL 펌핑 보조 정리를 사용하여 다음 언어들이 문맥 자유가 아님을 증명하세요:

1. $L_1 = \{a^n b^n c^n d^n \mid n \geq 0\}$
2. $L_2 = \{a^i b^j c^k \mid 0 \leq i \leq j \leq k\}$
3. $L_3 = \{ww \mid w \in \{a, b\}^*\}$ (복사 언어)

---

[Previous: Finite Automata](./03_Finite_Automata.md) | [Next: Top-Down Parsing](./05_Top_Down_Parsing.md) | [Overview](./00_Overview.md)
