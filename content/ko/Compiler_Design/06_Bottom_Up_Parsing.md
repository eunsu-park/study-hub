# 레슨 6: 상향식 파싱(Bottom-Up Parsing)

## 학습 목표

이 레슨을 완료하면 다음을 수행할 수 있습니다:

1. **이해**: 시프트-리듀스 파싱(Shift-Reduce Parsing)의 원리와 잎(리프)에서 위로 파스 트리를 구성하는 방법
2. **정의**: 핸들(Handle)이 무엇인지 설명하고 리듀스 결정을 유도하는 방식 설명
3. **구성**: LR(0) 아이템 집합과 해당 오토마톤(Automaton) 구성
4. **구축**: SLR(1), 정규 LR(1), LALR(1) 파싱 테이블 구축
5. **비교**: 각 LR 변형의 장단점 비교
6. **활용**: 파서 생성기 도구(Yacc, Bison, PLY) 효과적으로 사용
7. **해결**: 우선순위(Precedence)와 결합성(Associativity)을 이용한 시프트-리듀스/리듀스-리듀스 충돌 해결
8. **구현**: LR 파서에서의 오류 복구(Error Recovery) 전략 구현

---

## 1. 상향식 파싱(Bottom-Up Parsing) 소개

상향식 파싱은 **잎(리프)** (터미널 심볼)에서 **루트**(시작 심볼)까지 파스 트리를 구성합니다. 센텐셜 폼(Sentential Form)의 부분 문자열 중 생산 규칙의 우변(Right-Hand Side)과 일치하는 **핸들(Handle)**을 반복적으로 찾아 리듀스(Reduce)합니다.

상향식 파싱의 가장 일반적인 형태는 **LR 파싱**이며, 다음을 의미합니다:

- **L**: 입력을 **L**eft(왼쪽)에서 오른쪽으로 스캔
- **R**: **R**ightmost(최우측) 유도를 역순으로 생성

```
"id + id * id"에 대한 상향식 파스 구성:

Step 1: id + id * id     (shift id)
Step 2: F  + id * id     (reduce F -> id)
Step 3: T  + id * id     (reduce T -> F)
Step 4: E  + id * id     (reduce E -> T)
Step 5: E  + id * id     (shift +)
Step 6: E  + id * id     (shift id)
Step 7: E  + F  * id     (reduce F -> id)
Step 8: E  + T  * id     (reduce T -> F)
Step 9: E  + T  * id     (shift *)
Step 10: E + T  * id     (shift id)
Step 11: E + T  * F      (reduce F -> id)
Step 12: E + T           (reduce T -> T * F)
Step 13: E               (reduce E -> E + T)
Step 14: ACCEPT
```

### 1.1 왜 상향식인가?

상향식 파서는 하향식 파서에 비해 다음과 같은 중요한 장점이 있습니다:

| 특징 | 하향식(Top-Down, LL) | 상향식(Bottom-Up, LR) |
|------|---------------------|----------------------|
| 문법 클래스 | LL(1) 부분집합 | 실질적으로 모든 CFG |
| 좌재귀(Left Recursion) | 반드시 제거 필요 | 자연스럽게 처리 |
| 룩어헤드 결정 | 어느 생산 규칙을 예측 | 언제 리듀스할지 결정 |
| 모호성(Ambiguity) | 처리 어려움 | 우선순위 규칙으로 해결 |
| 도구 지원 | ANTLR | Yacc, Bison, PLY, tree-sitter |

---

## 2. 시프트-리듀스 파싱(Shift-Reduce Parsing)

### 2.1 네 가지 동작

시프트-리듀스 파서는 문법 심볼의 **스택(Stack)**과 **입력 버퍼(Input Buffer)**를 유지합니다. 각 단계에서 네 가지 동작 중 하나를 수행합니다:

1. **시프트(Shift)**: 다음 입력 심볼을 스택에 푸시
2. **리듀스(Reduce)**: 스택에서 핸들을 팝(Pop)하고 해당 비터미널(Nonterminal)을 푸시
3. **수락(Accept)**: 파싱 완료 -- 스택에 시작 심볼이 있고 입력이 비어 있음
4. **오류(Error)**: 유효한 동작 없음

```
시프트-리듀스 파싱 시각화:

    STACK                     INPUT                ACTION
    ─────                     ─────                ──────
    $                         id + id * id $       Shift
    $ id                      + id * id $          Reduce F -> id
    $ F                       + id * id $          Reduce T -> F
    $ T                       + id * id $          Reduce E -> T
    $ E                       + id * id $          Shift
    $ E +                     id * id $            Shift
    $ E + id                  * id $               Reduce F -> id
    $ E + F                   * id $               Reduce T -> F
    $ E + T                   * id $               Shift
    $ E + T *                 id $                 Shift
    $ E + T * id              $                    Reduce F -> id
    $ E + T * F               $                    Reduce T -> T * F
    $ E + T                   $                    Reduce E -> E + T
    $ E                       $                    Accept
```

### 2.2 핸들(Handle)

오른쪽 센텐셜 폼(Right-Sentential Form) $\gamma$의 **핸들**은 생산 규칙 $A \to \beta$와 $\gamma$ 내의 위치로서, $\beta$를 $A$로 대체했을 때 최우측 유도의 이전 오른쪽 센텐셜 폼이 생성됩니다.

**형식적으로:** $S \Rightarrow^*_{rm} \alpha A w \Rightarrow_{rm} \alpha \beta w$이면, $\alpha$ 다음 위치에서의 $A \to \beta$는 $\alpha \beta w$의 핸들입니다.

**핵심 통찰:** 핸들은 항상 스택의 맨 위에 있습니다. 이것이 상향식 파싱이 스택과 함께 동작하는 이유입니다 -- 핸들을 찾기 위해 스택 깊숙이 들여다볼 필요가 없습니다.

### 2.3 비어블 프리픽스(Viable Prefix)

**비어블 프리픽스(Viable Prefix)**는 파싱 스택에 나타날 수 있는 오른쪽 센텐셜 폼의 임의 접두사입니다. 동등하게, 최우측 핸들의 오른쪽 끝을 넘어가지 않는 접두사입니다.

비어블 프리픽스의 집합은 정규 언어(Regular Language)이므로 유한 오토마톤(Finite Automaton)으로 인식할 수 있습니다. 이 오토마톤이 바로 다음에 구성할 LR(0) 오토마톤입니다.

---

## 3. LR(0) 아이템과 LR(0) 오토마톤

### 3.1 LR(0) 아이템

**LR(0) 아이템**(또는 간단히 "아이템")은 우변의 어느 위치에 점(.)이 있는 생산 규칙입니다:

$$A \to \alpha \cdot \beta$$

점은 지금까지 생산 규칙의 얼마만큼을 인식했는지를 나타냅니다:

- $A \to \cdot \alpha\beta$: 아직 아무것도 매칭되지 않음
- $A \to \alpha \cdot \beta$: $\alpha$가 매칭됨, $\beta$가 예상됨
- $A \to \alpha\beta \cdot$: 우변 전체가 매칭됨 (리듀스 준비 완료)

**예시:** 생산 규칙 $E \to E + T$는 네 가지 아이템을 생성합니다:

$$E \to \cdot E + T \qquad E \to E \cdot + T \qquad E \to E + \cdot T \qquad E \to E + T \cdot$$

### 3.2 클로저(Closure) 연산

아이템 집합의 **클로저**는 비터미널 앞에 점이 있을 때 암시된 모든 아이템을 추가합니다.

**알고리즘:**

```
CLOSURE(I):
    repeat:
        for each item [A -> α . B β] in I:
            for each production B -> γ:
                add [B -> . γ] to I
    until no new items are added
    return I
```

**직관:** 다음에 $B$가 올 것으로 예상한다면(점이 $B$ 앞에 있음), $B$가 시작할 수 있는 모든 것에 대비해야 합니다.

### 3.3 GOTO 연산

**GOTO** 함수는 아이템 집합 간의 전이(Transition)를 정의합니다.

$$\text{GOTO}(I, X) = \text{CLOSURE}(\{[A \to \alpha X \cdot \beta] \mid [A \to \alpha \cdot X \beta] \in I\})$$

즉, $I$ 내의 모든 아이템 중 점이 심볼 $X$ 앞에 있는 것을 찾아 점을 $X$ 뒤로 이동시킨 후 클로저를 계산합니다.

### 3.4 LR(0) 오토마톤 구성

LR(0) 아이템 집합의 정규 컬렉션(Canonical Collection)은 비어블 프리픽스를 인식하는 유한 오토마톤의 상태를 형성합니다.

**알고리즘:**

```
ITEMS(G'):
    C = { CLOSURE({[S' -> . S]}) }    // 초기 상태
    repeat:
        for each set of items I in C:
            for each grammar symbol X:
                if GOTO(I, X) is not empty and not in C:
                    add GOTO(I, X) to C
    until no new sets are added
    return C
```

### 3.5 Python 구현

```python
"""
LR(0) Automaton Construction

Builds the canonical collection of LR(0) item sets, which forms
the basis for SLR, canonical LR, and LALR parsing table construction.
"""

from dataclasses import dataclass, field
from typing import Optional, FrozenSet


EPSILON = "ε"
EOF = "$"
DOT = "."


@dataclass(frozen=True)
class Item:
    """
    An LR(0) item: a production with a dot position.

    Example: Item("E", ("E", "+", "T"), 1) represents E -> E . + T
    """
    lhs: str
    rhs: tuple[str, ...]
    dot: int

    def __post_init__(self):
        assert 0 <= self.dot <= len(self.rhs)

    @property
    def next_symbol(self) -> Optional[str]:
        """Symbol immediately after the dot, or None if dot is at end."""
        if self.dot < len(self.rhs):
            sym = self.rhs[self.dot]
            return None if sym == EPSILON else sym
        return None

    @property
    def is_reduce(self) -> bool:
        """True if the dot is at the end (ready to reduce)."""
        return self.dot >= len(self.rhs) or (
            len(self.rhs) == 1 and self.rhs[0] == EPSILON
        )

    def advance(self) -> "Item":
        """Return a new item with the dot advanced by one position."""
        return Item(self.lhs, self.rhs, self.dot + 1)

    def __repr__(self):
        rhs_list = list(self.rhs)
        if rhs_list == [EPSILON]:
            return f"[{self.lhs} -> {DOT}]"
        rhs_list.insert(self.dot, DOT)
        return f"[{self.lhs} -> {' '.join(rhs_list)}]"


class LRAutomaton:
    """
    Constructs the canonical collection of LR(0) item sets.
    """

    def __init__(self, grammar: "Grammar"):
        """
        Initialize with an augmented grammar.

        The grammar should already have an augmented start production
        S' -> S added.
        """
        self.grammar = grammar
        self.states: list[frozenset[Item]] = []
        self.goto_table: dict[tuple[int, str], int] = {}
        self._build()

    def _get_productions(self, nonterminal: str) -> list[tuple[str, ...]]:
        """Get all productions for a nonterminal as tuples."""
        result = []
        for rhs in self.grammar.productions.get(nonterminal, []):
            result.append(tuple(rhs))
        return result

    def closure(self, items: set[Item]) -> frozenset[Item]:
        """Compute the closure of a set of items."""
        result = set(items)
        worklist = list(items)

        while worklist:
            item = worklist.pop()
            next_sym = item.next_symbol

            if next_sym and next_sym in self.grammar.nonterminals:
                for rhs in self._get_productions(next_sym):
                    new_item = Item(next_sym, rhs, 0)
                    if new_item not in result:
                        result.add(new_item)
                        worklist.append(new_item)

        return frozenset(result)

    def goto(self, items: frozenset[Item], symbol: str) -> frozenset[Item]:
        """Compute GOTO(items, symbol)."""
        moved = set()
        for item in items:
            if item.next_symbol == symbol:
                moved.add(item.advance())

        if not moved:
            return frozenset()
        return self.closure(moved)

    def _build(self):
        """Build the canonical collection of LR(0) item sets."""
        # Find the augmented start production S' -> S
        start = self.grammar.start
        start_rhs = self._get_productions(start)[0]
        initial_item = Item(start, start_rhs, 0)
        initial_state = self.closure({initial_item})

        self.states = [initial_state]
        state_map = {initial_state: 0}

        worklist = [0]
        all_symbols = self.grammar.terminals | self.grammar.nonterminals

        while worklist:
            state_idx = worklist.pop()
            state = self.states[state_idx]

            for symbol in all_symbols:
                next_state = self.goto(state, symbol)

                if not next_state:
                    continue

                if next_state not in state_map:
                    state_map[next_state] = len(self.states)
                    self.states.append(next_state)
                    worklist.append(state_map[next_state])

                self.goto_table[(state_idx, symbol)] = state_map[next_state]

    def print_states(self):
        """Pretty-print all states of the automaton."""
        for i, state in enumerate(self.states):
            print(f"\nState {i}:")
            for item in sorted(state, key=repr):
                print(f"  {item}")

    def print_transitions(self):
        """Print all transitions."""
        print("\nTransitions:")
        for (state, symbol), target in sorted(self.goto_table.items()):
            print(f"  State {state} --{symbol}--> State {target}")


# ─── Grammar Helper ───

class Grammar:
    """Grammar representation (same as Lesson 5, augmented)."""

    def __init__(self):
        self.productions: dict[str, list[list[str]]] = {}
        self.terminals: set[str] = set()
        self.nonterminals: set[str] = set()
        self.start: Optional[str] = None

    def add_production(self, lhs: str, rhs: list[str]):
        if lhs not in self.productions:
            self.productions[lhs] = []
        self.productions[lhs].append(rhs)
        self.nonterminals.add(lhs)

        if self.start is None:
            self.start = lhs

    def finalize(self):
        all_symbols = set()
        for lhs, rhs_list in self.productions.items():
            for rhs in rhs_list:
                for sym in rhs:
                    if sym != EPSILON:
                        all_symbols.add(sym)
        self.terminals = all_symbols - self.nonterminals

    def augment(self) -> "Grammar":
        """Create an augmented grammar with S' -> S."""
        aug = Grammar()
        new_start = self.start + "'"
        aug.add_production(new_start, [self.start])
        for lhs, rhs_list in self.productions.items():
            for rhs in rhs_list:
                aug.add_production(lhs, rhs)
        aug.finalize()
        return aug


# ─── Demo ───

def build_expression_grammar() -> Grammar:
    """
    Expression grammar:
        E -> E + T | T
        T -> T * F | F
        F -> ( E ) | id
    """
    g = Grammar()
    g.add_production("E", ["E", "+", "T"])
    g.add_production("E", ["T"])
    g.add_production("T", ["T", "*", "F"])
    g.add_production("T", ["F"])
    g.add_production("F", ["(", "E", ")"])
    g.add_production("F", ["id"])
    g.finalize()
    return g


if __name__ == "__main__":
    grammar = build_expression_grammar()
    augmented = grammar.augment()
    automaton = LRAutomaton(augmented)
    automaton.print_states()
    automaton.print_transitions()
```

**예시 출력 (일부):**

```
State 0:
  [E' -> . E]
  [E -> . E + T]
  [E -> . T]
  [F -> . ( E )]
  [F -> . id]
  [T -> . F]
  [T -> . T * F]

State 1:
  [E' -> E .]
  [E -> E . + T]

State 2:
  [E -> T .]
  [T -> T . * F]

State 3:
  [T -> F .]

...
```

---

## 4. SLR(1) 파싱

### 4.1 SLR(1) 테이블 구성

**단순 LR (SLR, Simple LR)** 파싱은 LR(0) 오토마톤과 FOLLOW 집합을 사용하여 파싱 테이블을 구성합니다. 가장 간단한 LR 변형입니다.

**알고리즘:**

```
SLR(1) 테이블 구성:

정규 컬렉션의 각 상태 I_i에 대해:

  1. SHIFT: [A -> α . a β]가 I_i에 있고 GOTO(I_i, a) = I_j이면:
       ACTION[i, a] = "shift j"로 설정

  2. REDUCE: [A -> α .]가 I_i에 있고 (A ≠ S')이면:
       FOLLOW(A)의 각 터미널 a에 대해:
           ACTION[i, a] = "reduce A -> α"로 설정

  3. ACCEPT: [S' -> S .]가 I_i에 있으면:
       ACTION[i, $] = "accept"로 설정

  4. GOTO: GOTO(I_i, A) = I_j인 비터미널 A에 대해:
       GOTO[i, A] = j로 설정

어떤 항목이 중복 정의되면 해당 문법은 SLR(1)이 아닙니다.
```

### 4.2 SLR(1) 파서 구현

```python
"""
SLR(1) Parser Implementation

Constructs an SLR(1) parsing table from the LR(0) automaton
and FOLLOW sets, then uses it to parse input strings.
"""


# Reuse FIRST/FOLLOW from Lesson 5 (imported or redefined)
def compute_first(grammar: Grammar) -> dict[str, set[str]]:
    """Compute FIRST sets (same algorithm as Lesson 5)."""
    first: dict[str, set[str]] = {}
    for t in grammar.terminals:
        first[t] = {t}
    for nt in grammar.nonterminals:
        first[nt] = set()

    changed = True
    while changed:
        changed = False
        for lhs, rhs_list in grammar.productions.items():
            for rhs in rhs_list:
                if rhs == [EPSILON]:
                    if EPSILON not in first[lhs]:
                        first[lhs].add(EPSILON)
                        changed = True
                    continue
                all_eps = True
                for sym in rhs:
                    sf = first.get(sym, set())
                    additions = sf - {EPSILON}
                    if not additions.issubset(first[lhs]):
                        first[lhs] |= additions
                        changed = True
                    if EPSILON not in sf:
                        all_eps = False
                        break
                if all_eps and EPSILON not in first[lhs]:
                    first[lhs].add(EPSILON)
                    changed = True
    return first


def compute_follow(
    grammar: Grammar, first: dict[str, set[str]]
) -> dict[str, set[str]]:
    """Compute FOLLOW sets (same algorithm as Lesson 5)."""
    follow: dict[str, set[str]] = {nt: set() for nt in grammar.nonterminals}
    follow[grammar.start].add(EOF)

    def first_of_seq(symbols):
        result = set()
        for s in symbols:
            sf = first.get(s, {s})
            result |= (sf - {EPSILON})
            if EPSILON not in sf:
                return result
        result.add(EPSILON)
        return result

    changed = True
    while changed:
        changed = False
        for lhs, rhs_list in grammar.productions.items():
            for rhs in rhs_list:
                if rhs == [EPSILON]:
                    continue
                for i, sym in enumerate(rhs):
                    if sym not in grammar.nonterminals:
                        continue
                    beta = rhs[i+1:]
                    if beta:
                        fb = first_of_seq(beta)
                        add = fb - {EPSILON}
                        if not add.issubset(follow[sym]):
                            follow[sym] |= add
                            changed = True
                        if EPSILON in fb:
                            if not follow[lhs].issubset(follow[sym]):
                                follow[sym] |= follow[lhs]
                                changed = True
                    else:
                        if not follow[lhs].issubset(follow[sym]):
                            follow[sym] |= follow[lhs]
                            changed = True
    return follow


@dataclass
class Action:
    """Represents a parsing action."""
    kind: str          # "shift", "reduce", "accept"
    state: int = -1    # target state for shift
    lhs: str = ""      # LHS for reduce
    rhs: tuple = ()    # RHS for reduce
    prod_num: int = -1 # production number for reduce

    def __repr__(self):
        if self.kind == "shift":
            return f"s{self.state}"
        elif self.kind == "reduce":
            return f"r({self.lhs}->{' '.join(self.rhs)})"
        elif self.kind == "accept":
            return "acc"
        return f"?{self.kind}"


class SLRParser:
    """
    Complete SLR(1) parser.

    Constructs the SLR(1) table from a grammar and uses it
    to parse input token streams.
    """

    def __init__(self, grammar: Grammar):
        self.original = grammar
        self.augmented = grammar.augment()
        self.automaton = LRAutomaton(self.augmented)

        # Compute FIRST and FOLLOW on the augmented grammar
        self.first = compute_first(self.augmented)
        self.follow = compute_follow(self.augmented, self.first)

        # Number productions for reduce actions
        self.prod_list: list[tuple[str, tuple[str, ...]]] = []
        for lhs, rhs_list in self.augmented.productions.items():
            for rhs in rhs_list:
                self.prod_list.append((lhs, tuple(rhs)))

        # Build the ACTION and GOTO tables
        self.action_table: dict[tuple[int, str], Action] = {}
        self.goto_table: dict[tuple[int, str], int] = {}
        self.conflicts: list[str] = []
        self._build_tables()

    def _find_prod_num(self, lhs: str, rhs: tuple) -> int:
        """Find the production number for a given production."""
        for i, (l, r) in enumerate(self.prod_list):
            if l == lhs and r == rhs:
                return i
        return -1

    def _set_action(self, state: int, symbol: str, action: Action):
        """Set an action, detecting conflicts."""
        key = (state, symbol)
        if key in self.action_table:
            existing = self.action_table[key]
            if repr(existing) != repr(action):
                conflict_type = (
                    "shift-reduce"
                    if {existing.kind, action.kind} == {"shift", "reduce"}
                    else "reduce-reduce"
                )
                self.conflicts.append(
                    f"{conflict_type} conflict in state {state} "
                    f"on '{symbol}': {existing} vs {action}"
                )
                # Default resolution: prefer shift over reduce
                if action.kind == "shift":
                    self.action_table[key] = action
                return
        self.action_table[key] = action

    def _build_tables(self):
        """Build the SLR(1) ACTION and GOTO tables."""
        start_nt = self.augmented.start  # S'
        original_start = self.original.start  # S

        for i, state in enumerate(self.automaton.states):
            for item in state:
                next_sym = item.next_symbol

                if next_sym is not None:
                    # Dot is before a symbol
                    if next_sym in self.augmented.terminals:
                        # SHIFT
                        target = self.automaton.goto_table.get(
                            (i, next_sym)
                        )
                        if target is not None:
                            self._set_action(
                                i, next_sym,
                                Action("shift", state=target)
                            )
                    elif next_sym in self.augmented.nonterminals:
                        # GOTO
                        target = self.automaton.goto_table.get(
                            (i, next_sym)
                        )
                        if target is not None:
                            self.goto_table[(i, next_sym)] = target

                elif item.is_reduce:
                    if item.lhs == start_nt:
                        # ACCEPT: S' -> S .
                        self._set_action(i, EOF, Action("accept"))
                    else:
                        # REDUCE: A -> α .
                        prod_num = self._find_prod_num(
                            item.lhs, item.rhs
                        )
                        for terminal in self.follow.get(item.lhs, set()):
                            self._set_action(
                                i, terminal,
                                Action(
                                    "reduce",
                                    lhs=item.lhs,
                                    rhs=item.rhs,
                                    prod_num=prod_num,
                                )
                            )

    def parse(self, tokens: list[str], verbose: bool = False) -> bool:
        """
        Parse a list of terminal symbols.

        Args:
            tokens: list of terminal symbols (without $)
            verbose: if True, print each step

        Returns:
            True if accepted, False otherwise.
        """
        input_syms = tokens + [EOF]
        ip = 0

        # Stack holds alternating states and symbols:
        # [state0, sym1, state1, sym2, state2, ...]
        # We use a simpler model: stack of states, separate symbol stack
        state_stack = [0]
        symbol_stack: list[str] = []

        step = 0
        if verbose:
            print(
                f"{'Step':>4}  {'Stack':<35} "
                f"{'Input':<25} {'Action'}"
            )
            print("-" * 95)

        while True:
            state = state_stack[-1]
            current = input_syms[ip]

            if verbose:
                stack_str = " ".join(
                    f"{s}/{symbol_stack[i]}"
                    if i < len(symbol_stack) else str(s)
                    for i, s in enumerate(state_stack)
                )
                input_str = " ".join(input_syms[ip:])
                print(
                    f"{step:>4}  {stack_str:<35} "
                    f"{input_str:<25}",
                    end="",
                )

            action = self.action_table.get((state, current))

            if action is None:
                if verbose:
                    print(f"ERROR: no action for state {state}, '{current}'")
                return False

            if action.kind == "shift":
                symbol_stack.append(current)
                state_stack.append(action.state)
                ip += 1
                if verbose:
                    print(f"Shift {action.state}")

            elif action.kind == "reduce":
                rhs_len = len(action.rhs)
                if action.rhs == (EPSILON,):
                    rhs_len = 0

                # Pop |rhs| symbols and states
                for _ in range(rhs_len):
                    state_stack.pop()
                    symbol_stack.pop()

                # Push LHS
                symbol_stack.append(action.lhs)
                goto_state = self.goto_table.get(
                    (state_stack[-1], action.lhs)
                )
                if goto_state is None:
                    if verbose:
                        print(
                            f"ERROR: no GOTO for state "
                            f"{state_stack[-1]}, '{action.lhs}'"
                        )
                    return False
                state_stack.append(goto_state)

                if verbose:
                    print(
                        f"Reduce {action.lhs} -> "
                        f"{' '.join(action.rhs)}"
                    )

            elif action.kind == "accept":
                if verbose:
                    print("ACCEPT")
                return True

            step += 1
            if step > 100000:
                if verbose:
                    print("ERROR: too many steps")
                return False

    def print_tables(self):
        """Pretty-print the ACTION and GOTO tables."""
        terminals = sorted(self.augmented.terminals) + [EOF]
        nonterminals = sorted(
            self.augmented.nonterminals - {self.augmented.start}
        )

        col_w = 12
        header = f"{'State':>6}"
        header += "".join(f"{t:>{col_w}}" for t in terminals)
        header += " |"
        header += "".join(f"{nt:>{col_w}}" for nt in nonterminals)

        print(header)
        print("-" * len(header))

        for i in range(len(self.automaton.states)):
            row = f"{i:>6}"
            for t in terminals:
                action = self.action_table.get((i, t))
                cell = repr(action) if action else ""
                row += f"{cell:>{col_w}}"
            row += " |"
            for nt in nonterminals:
                goto = self.goto_table.get((i, nt))
                cell = str(goto) if goto is not None else ""
                row += f"{cell:>{col_w}}"
            print(row)


# ─── Demo ───

if __name__ == "__main__":
    grammar = build_expression_grammar()

    print("Original Grammar:")
    for lhs, rhs_list in grammar.productions.items():
        for rhs in rhs_list:
            print(f"  {lhs} -> {' '.join(rhs)}")
    print()

    parser = SLRParser(grammar)

    if parser.conflicts:
        print("Conflicts found:")
        for c in parser.conflicts:
            print(f"  {c}")
    else:
        print("No conflicts -- grammar is SLR(1)!")
    print()

    print("SLR(1) Parsing Table:")
    parser.print_tables()
    print()

    # Parse some inputs
    test_cases = [
        (["id", "+", "id", "*", "id"], True),
        (["(", "id", "+", "id", ")", "*", "id"], True),
        (["id", "*", "(", "id", "+", "id", ")"], True),
        (["id", "+", "+"], False),
    ]

    for tokens, expected in test_cases:
        print(f"\n{'='*95}")
        print(f"Input: {' '.join(tokens)}")
        print(f"{'='*95}")
        result = parser.parse(tokens, verbose=True)
        status = "ACCEPTED" if result else "REJECTED"
        print(f"Result: {status}")
        assert result == expected
```

---

## 5. 정규 LR(1) 파싱

### 5.1 LR(1) 아이템

**LR(1) 아이템**은 LR(0) 아이템에 **룩어헤드(Lookahead)** 터미널을 추가한 것입니다:

$$[A \to \alpha \cdot \beta, a]$$

이는 "현재 $A \to \alpha\beta$를 인식하는 중이며, $\alpha$를 매칭했고, 리듀스를 완료하면 다음 입력 심볼이 $a$여야 한다"는 의미입니다.

룩어헤드는 점이 끝에 도달했을 때(리듀스 결정을 위해)만 사용되며, 시프트 결정은 LR(0)과 동일합니다.

### 5.2 LR(1) 클로저

LR(1) 클로저는 룩어헤드를 전파합니다:

```
CLOSURE(I):
    repeat:
        for each item [A -> α . B β, a] in I:
            for each production B -> γ:
                for each terminal b in FIRST(βa):
                    add [B -> . γ, b] to I
    until no new items are added
    return I
```

LR(0) 클로저와의 핵심 차이는 룩어헤드 집합 $\text{FIRST}(\beta a)$의 계산입니다. 이것이 파서를 통해 문맥이 흘러가는 방식입니다.

### 5.3 LR(1) 테이블 구성

테이블 구성은 SLR(1)과 유사하지만, 리듀스 동작에 FOLLOW 집합 대신 아이템별 룩어헤드를 사용합니다:

```
각 상태 I_i에 대해:
  SHIFT:  [A -> α . a β, b]가 I_i에 있고 GOTO(I_i, a) = I_j이면:
              ACTION[i, a] = "shift j"

  REDUCE: [A -> α ., a]가 I_i에 있고 (A ≠ S')이면:
              ACTION[i, a] = "reduce A -> α"
              (FOLLOW(A) 전체가 아닌 룩어헤드 'a'에만 적용)

  ACCEPT: [S' -> S ., $]가 I_i에 있으면:
              ACTION[i, $] = "accept"
```

### 5.4 LR(1) vs SLR(1)

LR(1)의 핵심 장점은 리듀스 동작의 정밀도입니다:

```
SLR(1):  FOLLOW(A)의 모든 터미널에서 A -> α 리듀스
LR(1):   특정 룩어헤드 터미널에서만 A -> α 리듀스

즉, LR(1)은 SLR(1)이 처리할 수 없는 문법을 처리할 수 있습니다.
```

**SLR(1)은 실패하지만 LR(1)은 동작하는 예시:**

$$
\begin{aligned}
S &\to L = R \mid R \\
L &\to * R \mid \textbf{id} \\
R &\to L
\end{aligned}
$$

SLR(1)에서는 $[R \to L \cdot]$과 $[S \to L \cdot = R]$을 포함하는 상태에서 시프트-리듀스 충돌이 발생합니다. $=$가 $\text{FOLLOW}(R)$에 있기 때문입니다. LR(1)에서는 아이템 $[R \to L \cdot, \$]$의 룩어헤드가 $=$가 아닌 $\$$이므로 충돌이 없습니다.

### 5.5 실용적 고려사항: 테이블 크기

정규 LR(1)의 주요 단점은 오토마톤의 크기입니다. 서로 다른 룩어헤드를 가진 아이템이 분리되어 유지되므로 LR(1) 상태의 수가 LR(0) 상태보다 훨씬 많을 수 있습니다. 일반적인 프로그래밍 언어 문법의 경우 LR(1) 테이블은 수천 개의 상태를 가질 수 있는 반면, LR(0)/SLR은 수백 개입니다.

---

## 6. LALR(1) 파싱

### 6.1 동기

**LALR(1)** (Look-Ahead LR)은 SLR(1)과 정규 LR(1) 사이의 실용적인 절충안입니다:

- **SLR(1)보다 강력**: 문맥 민감 룩어헤드 사용
- **SLR(1)과 동일한 상태 수**: 동일한 코어(Core)를 가진 LR(1) 상태 병합
- **LR(1)에 거의 근접**: 실용적인 문법 중 LR(1)이지만 LALR(1)이 아닌 경우는 매우 드뭄

### 6.2 병합을 통한 구성

LALR(1) 상태는 동일한 **코어**(룩어헤드를 무시한 동일한 LR(0) 아이템 집합)를 가진 LR(1) 상태를 병합하여 생성됩니다. 룩어헤드 집합은 합집합(Union)으로 병합됩니다.

```
LR(1) 상태:                         LALR(1) (병합 후):
  State 4a: {[R -> L ., $]}           State 4: {[R -> L ., {$, =}]}
  State 4b: {[R -> L ., =]}
```

### 6.3 새로운 충돌 가능성

병합은 정규 LR(1) 테이블에 없었던 **리듀스-리듀스 충돌**을 유발할 수 있습니다. 그러나 코어가 시프트 동작을 결정하므로 새로운 시프트-리듀스 충돌은 **발생할 수 없습니다**.

```
LALR 병합이 리듀스-리듀스 충돌을 유발하는 예시:

LR(1) State A:  { [X -> α ., a],  [Y -> β ., b] }
LR(1) State B:  { [X -> α ., b],  [Y -> β ., a] }

병합 후:  { [X -> α ., {a,b}],  [Y -> β ., {a,b}] }
                         ↑ 리듀스-리듀스 충돌!
```

실제로 이 상황은 실제 프로그래밍 언어 문법에서 매우 드뭅니다.

### 6.4 LR 변형 비교

```
문법 클래스 계층:

    ┌─────────────────────────────────────────────┐
    │              All CFGs                        │
    │  ┌───────────────────────────────────────┐   │
    │  │         Unambiguous CFGs              │   │
    │  │  ┌─────────────────────────────────┐  │   │
    │  │  │        LR(1) Grammars           │  │   │
    │  │  │  ┌───────────────────────────┐  │  │   │
    │  │  │  │    LALR(1) Grammars       │  │  │   │
    │  │  │  │  ┌─────────────────────┐  │  │  │   │
    │  │  │  │  │   SLR(1) Grammars   │  │  │  │   │
    │  │  │  │  │  ┌───────────────┐  │  │  │  │   │
    │  │  │  │  │  │  LR(0)       │  │  │  │  │   │
    │  │  │  │  │  └───────────────┘  │  │  │  │   │
    │  │  │  │  └─────────────────────┘  │  │  │   │
    │  │  │  └───────────────────────────┘  │  │   │
    │  │  └─────────────────────────────────┘  │   │
    │  └───────────────────────────────────────┘   │
    └─────────────────────────────────────────────┘
```

| 변형 | 리듀스 결정 | 상태 수 | 표현력 | 도구 |
|------|------------|---------|--------|------|
| LR(0) | 항상 리듀스 (룩어헤드 없음) | 최소 | 가장 약함 | 이론적 |
| SLR(1) | FOLLOW 집합 | LR(0)과 동일 | 양호 | 간단한 생성기 |
| LALR(1) | 병합된 LR(1) 룩어헤드 | LR(0)과 동일 | 매우 양호 | Yacc, Bison, PLY |
| LR(1) | 정확한 LR(1) 룩어헤드 | 훨씬 클 수 있음 | 최고 | 직접 사용 드뭄 |

---

## 7. 파서 생성기 도구

### 7.1 Yacc와 Bison

**Yacc**(Yet Another Compiler Compiler)는 클래식 유닉스 파서 생성기입니다. **Bison**은 GNU 버전입니다. 둘 다 문법 명세서로부터 LALR(1) 파서를 생성합니다.

**Yacc/Bison 문법 형식:**

```yacc
/* Declarations section */
%token ID NUM
%left '+' '-'
%left '*' '/'

%%
/* Grammar rules section */
expr : expr '+' expr    { $$ = $1 + $3; }
     | expr '-' expr    { $$ = $1 - $3; }
     | expr '*' expr    { $$ = $1 * $3; }
     | expr '/' expr    { $$ = $1 / $3; }
     | '(' expr ')'    { $$ = $2; }
     | NUM              { $$ = $1; }
     ;
%%
/* C code section */
```

주요 기능:
- **`%left`, `%right`, `%nonassoc`**: 연산자 우선순위(Precedence)와 결합성(Associativity) 선언
- **`$$`**: LHS 심볼의 값 (결과)
- **`$1`, `$2`, `$3`, ...**: RHS 심볼들의 값
- **`%prec`**: 특정 규칙의 기본 우선순위 재정의

### 7.2 PLY (Python Lex-Yacc)

**PLY**는 Lex와 Yacc의 Python 구현체입니다. LALR(1) 파서를 생성합니다.

```python
"""
PLY Example: Expression Parser

PLY (Python Lex-Yacc) provides lex and yacc functionality for Python.
Install with: pip install ply
"""

# ─── Lexer ───

import ply.lex as lex

tokens = ('NUM', 'PLUS', 'TIMES', 'LPAREN', 'RPAREN', 'ID')

t_PLUS = r'\+'
t_TIMES = r'\*'
t_LPAREN = r'\('
t_RPAREN = r'\)'
t_ID = r'[a-zA-Z_][a-zA-Z0-9_]*'
t_ignore = ' \t'

def t_NUM(t):
    r'\d+'
    t.value = int(t.value)
    return t

def t_error(t):
    print(f"Illegal character '{t.value[0]}'")
    t.lexer.skip(1)

lexer = lex.lex()


# ─── Parser ───

import ply.yacc as yacc

# Precedence (lowest to highest)
precedence = (
    ('left', 'PLUS'),
    ('left', 'TIMES'),
)

def p_expr_binop(p):
    '''expr : expr PLUS expr
            | expr TIMES expr'''
    if p[2] == '+':
        p[0] = ('add', p[1], p[3])
    else:
        p[0] = ('mul', p[1], p[3])

def p_expr_group(p):
    '''expr : LPAREN expr RPAREN'''
    p[0] = p[2]

def p_expr_num(p):
    '''expr : NUM'''
    p[0] = ('num', p[1])

def p_expr_id(p):
    '''expr : ID'''
    p[0] = ('id', p[1])

def p_error(p):
    if p:
        print(f"Syntax error at '{p.value}'")
    else:
        print("Syntax error at end of input")

parser = yacc.yacc()


# ─── Usage ───

if __name__ == "__main__":
    test_inputs = [
        "3 + 4 * 5",
        "(a + b) * c",
        "x + y + z",
    ]

    for text in test_inputs:
        result = parser.parse(text)
        print(f"{text:20s} => {result}")
```

**출력:**

```
3 + 4 * 5            => ('add', ('num', 3), ('mul', ('num', 4), ('num', 5)))
(a + b) * c          => ('mul', ('add', ('id', 'a'), ('id', 'b')), ('id', 'c'))
x + y + z            => ('add', ('add', ('id', 'x'), ('id', 'y')), ('id', 'z'))
```

### 7.3 기타 파서 생성기 도구

| 도구 | 언어 | 파서 유형 | 주요 특징 |
|------|------|----------|----------|
| **Yacc/Bison** | C/C++ | LALR(1) | 업계 표준 |
| **PLY** | Python | LALR(1) | Python식 Yacc |
| **Lark** | Python | Earley/LALR | 우아한 EBNF 문법 |
| **ANTLR** | Java/Python/... | ALL(*) | 가장 강력한 LL |
| **tree-sitter** | C (바인딩 포함) | GLR | 증분식(Incremental), 오류 허용 |
| **Menhir** | OCaml | LR(1) | LALR이 아닌 완전한 LR(1) |
| **Happy** | Haskell | LALR(1) | 모나딕 파서 동작 |

---

## 8. 충돌 해결

### 8.1 시프트-리듀스 충돌

**시프트-리듀스 충돌**은 파서가 다음 입력 심볼을 시프트하거나 스택의 핸들을 리듀스할 수 있을 때 발생합니다.

**전형적인 예시: 댕글링 else(Dangling Else).**

```
stmt : IF expr THEN stmt ELSE stmt
     | IF expr THEN stmt
     | other
     ;
```

파서가 룩어헤드 `ELSE`로 `IF expr THEN stmt`를 인식할 때:
- **시프트** `ELSE` (이 `if`와 `else` 연결)
- **리듀스** `stmt -> IF expr THEN stmt` (외부 `if`와 `else` 연결)

**해결:** 대부분의 파서 생성기는 기본적으로 **시프트**를 선택합니다 (`else`를 가장 가까운 `if`와 연결). 이것이 사실상 모든 프로그래밍 언어에서 올바른 동작입니다.

### 8.2 리듀스-리듀스 충돌

**리듀스-리듀스 충돌**은 파서가 두 가지 다른 생산 규칙으로 리듀스할 수 있을 때 발생합니다.

```
stmt : ID '(' expr_list ')'    // 함수 호출
     | ID '(' expr_list ')'    // 배열 서브스크립트 (가상)
     ;
```

**해결 전략:**
- 모호성을 제거하도록 문법 재작성
- Yacc/Bison에서는 먼저 나열된 생산 규칙이 우선
- 의미 분석(Semantic Analysis)을 통해 경우를 구분 (파싱 수준이 아닌)

### 8.3 우선순위와 결합성 선언

파서 생성기는 표현식 문법에서 시프트-리듀스 충돌을 체계적으로 해결하기 위해 **우선순위**와 **결합성** 선언을 제공합니다.

```yacc
/* 우선순위: 낮은 것부터 */
%right '='                  /* 대입: 우결합 */
%left OR                    /* 논리 OR */
%left AND                   /* 논리 AND */
%left EQ NE                 /* 동등 비교 */
%left '<' '>' LE GE         /* 크기 비교 */
%left '+' '-'               /* 덧셈 */
%left '*' '/' '%'           /* 곱셈 */
%right UMINUS               /* 단항 빼기 (가상 토큰) */
```

**작동 방식:**

1. 각 토큰은 (선언 위치에 따른) **우선순위 레벨**을 가짐
2. 각 생산 규칙은 **가장 오른쪽 터미널**의 우선순위를 가짐
3. 시프트-리듀스 충돌에서:
   - 시프트 토큰의 우선순위가 리듀스 생산 규칙보다 높으면: **시프트**
   - 리듀스 생산 규칙의 우선순위가 더 높으면: **리듀스**
   - 동일한 우선순위이면: **결합성** 사용 (`%left` = 리듀스, `%right` = 시프트, `%nonassoc` = 오류)

**예시:** 입력 `3 + 4 * 5`에서:

```
Stack: ... 3 + 4    Lookahead: *

Production to reduce: expr -> expr + expr  (precedence of '+')
Token to shift: '*'                        (precedence of '*')

Since * > + in precedence: SHIFT

Result: 3 + (4 * 5)  ✓
```

### 8.4 `%prec` 지시어

때로는 생산 규칙이 가장 오른쪽 터미널이 제시하는 것과 다른 우선순위가 필요합니다. `%prec` 지시어로 이를 재정의할 수 있습니다:

```yacc
expr : '-' expr  %prec UMINUS    /* 단항 빼기 */
     {
         $$ = -$2;
     }
     ;
```

`%prec UMINUS` 없이는 생산 규칙 `expr -> '-' expr`가 `-`의 우선순위(덧셈 레벨)를 가집니다. `%prec UMINUS`를 사용하면 더 높은 단항 빼기 우선순위를 가집니다.

---

## 9. LR 파서의 오류 복구

### 9.1 패닉 모드(Panic Mode)

하향식 파싱과 유사하지만, 시프트-리듀스 프레임워크에 맞게 조정됩니다:

1. 오류 복구 비터미널 $A$에 대해 GOTO($s$, $A$)가 정의된 상태 $s$를 찾을 때까지 스택에서 상태를 팝
2. $A$와 GOTO 상태를 스택에 푸시
3. 파서가 계속할 수 있는 입력 심볼을 찾을 때까지 입력 심볼 버림

### 9.2 오류 생산 규칙(Error Productions)

Yacc/Bison은 오류 복구를 위해 특별한 `error` 토큰을 제공합니다:

```yacc
stmt : expr ';'
     | error ';'      /* 오류 시 다음 세미콜론까지 건너뜀 */
     {
         yyerrok;      /* 오류 상태 초기화 */
         printf("Recovered from error\n");
     }
     ;

block : '{' stmt_list '}'
      | '{' error '}'    /* 오류 시 매칭되는 중괄호까지 건너뜀 */
      ;
```

**`error` 동작 방식:**

1. 구문 오류가 발생하면 파서는 `error`를 시프트할 수 있는 상태를 찾을 때까지 상태를 팝
2. `error` 토큰을 시프트
3. 오류 생산 규칙 이후 성공적으로 시프트할 수 있을 때까지 입력 토큰 버림
4. `yyerrok`은 파서에게 정상 오류 보고를 재개하도록 지시 (그렇지 않으면 몇 개의 토큰 동안 오류를 억제)

### 9.3 구현

```python
class SLRParserWithRecovery(SLRParser):
    """
    SLR parser with error recovery support.

    Uses a synchronization-based strategy:
    when an error occurs, pop states until we find one that
    can handle a synchronizing token.
    """

    def __init__(self, grammar: Grammar, sync_tokens: set[str] = None):
        super().__init__(grammar)
        self.sync_tokens = sync_tokens or {";", ")", "}"}
        self.errors: list[str] = []

    def parse_with_recovery(
        self, tokens: list[str], verbose: bool = False
    ) -> tuple[bool, list[str]]:
        """
        Parse with error recovery.

        Returns:
            (success, errors) where success is True if parsing completed
            and errors is a list of error messages.
        """
        self.errors = []
        input_syms = tokens + [EOF]
        ip = 0
        state_stack = [0]
        symbol_stack: list[str] = []

        while True:
            state = state_stack[-1]
            current = input_syms[ip]

            action = self.action_table.get((state, current))

            if action is None:
                # ERROR
                error_msg = (
                    f"Syntax error at position {ip}: "
                    f"unexpected '{current}' in state {state}"
                )
                self.errors.append(error_msg)
                if verbose:
                    print(f"  ERROR: {error_msg}")

                # Recovery: skip input until sync token
                recovered = False
                while ip < len(input_syms) - 1:  # Don't skip $
                    if input_syms[ip] in self.sync_tokens:
                        # Try to find a state that can handle this
                        while len(state_stack) > 1:
                            test_action = self.action_table.get(
                                (state_stack[-1], input_syms[ip])
                            )
                            if test_action is not None:
                                recovered = True
                                break
                            state_stack.pop()
                            if symbol_stack:
                                symbol_stack.pop()

                        if recovered:
                            if verbose:
                                print(
                                    f"  Recovered at '{input_syms[ip]}' "
                                    f"in state {state_stack[-1]}"
                                )
                            break
                    ip += 1

                if not recovered:
                    return False, self.errors
                continue

            if action.kind == "shift":
                symbol_stack.append(current)
                state_stack.append(action.state)
                ip += 1
            elif action.kind == "reduce":
                rhs_len = len(action.rhs)
                if action.rhs == (EPSILON,):
                    rhs_len = 0
                for _ in range(rhs_len):
                    state_stack.pop()
                    symbol_stack.pop()
                symbol_stack.append(action.lhs)
                goto_state = self.goto_table.get(
                    (state_stack[-1], action.lhs)
                )
                if goto_state is None:
                    return False, self.errors
                state_stack.append(goto_state)
            elif action.kind == "accept":
                return len(self.errors) == 0, self.errors
```

---

## 10. 고급 주제

### 10.1 GLR 파싱

**일반화 LR (GLR, Generalized LR)** 파싱은 충돌이 발생할 때 여러 파서 스택을 동시에 유지하여 모호하고 비결정론적인 문법을 처리합니다. 충돌이 발생하면 파서가 여러 개의 병렬 파서로 **분기(Fork)**됩니다.

```
GLR 파싱: 충돌 시 분기

    충돌 전:         분기 후:
    ┌─────────┐         ┌─────────┐
    │ Stack A │         │ Stack A │ ──shift──▶  Stack A'
    └─────────┘         │         │ ──reduce──▶ Stack A''
                        └─────────┘

    두 스택이 독립적으로 계속됨.
    유효하지 않은 파스는 사라지고, 유효한 것들이 수렴함.
```

**사용처:** tree-sitter (에디터에서의 증분 파싱), Elkhound, Bison의 `%glr-parser` 모드.

### 10.2 증분 파싱(Incremental Parsing)

현대 에디터(VS Code, Neovim)는 매 키 입력마다 파일을 다시 파싱해야 합니다. **증분 파싱(Incremental Parsing)**은 파일의 변경되지 않은 부분에 대해 이전 파스 결과를 재사용합니다.

**tree-sitter**는 가장 유명한 증분 파서입니다:

1. 이전 편집으로부터 파스 트리를 유지
2. 사용자가 타이핑할 때 영향받은 트리 노드만 다시 파싱
3. 문법적으로 잘못된 코드에 대한 강인성을 위해 GLR 알고리즘 사용
4. 일반적인 편집에 대해 밀리초 미만의 파싱 시간 달성

### 10.3 연산자 우선순위 파싱(Operator Precedence Parsing)

표현식이 많은 언어의 경우, **연산자 우선순위 파싱** (또는 **프랫 파싱(Pratt Parsing)**)이 완전한 LR 파싱의 더 간단한 대안을 제공합니다:

```python
def pratt_parse(tokens, min_precedence=0):
    """
    Pratt parser / precedence climbing for expressions.

    A simple, elegant alternative to LR for expression parsing.
    """
    left = parse_atom(tokens)

    while (
        tokens.peek() is not None
        and get_precedence(tokens.peek()) >= min_precedence
    ):
        op = tokens.next()
        prec = get_precedence(op)
        assoc = get_associativity(op)

        # For left-associative: next_min = prec + 1
        # For right-associative: next_min = prec
        next_min = prec + 1 if assoc == "left" else prec

        right = pratt_parse(tokens, next_min)
        left = BinOp(op, left, right)

    return left
```

---

## 11. 요약

상향식 파싱은 프로덕션 컴파일러와 파서 생성기에서 지배적인 전략입니다. 좌재귀 문법을 처리하는 능력과 모호성 해결에 대한 체계적인 접근 방식이 매우 실용적입니다.

**핵심 개념:**

| 개념 | 설명 |
|------|------|
| **시프트-리듀스** | 기본 동작: 입력을 스택에 시프트하거나 핸들을 리듀스 |
| **핸들** | 각 단계에서 리듀스해야 할 RHS (항상 스택 맨 위에 있음) |
| **LR(0) 아이템** | 파스 진행 상황을 표시하는 점이 있는 생산 규칙 |
| **클로저** | 비터미널 앞에 점이 있을 때 암시된 아이템 추가 |
| **GOTO** | 점을 이동시켜 아이템 집합 간 전이 |
| **SLR(1)** | LR(0) 오토마톤 + 리듀스 결정을 위한 FOLLOW 집합 |
| **LR(1)** | 아이템이 특정 룩어헤드를 가짐; 가장 정밀 |
| **LALR(1)** | 병합된 LR(1) 상태; 실용적 최적점 (Yacc, Bison) |
| **우선순위** | 연산자 문법을 위한 체계적인 충돌 해결 |

**어떤 LR 변형을 사용해야 하는가?**

- **SLR(1)**: 간단한 문법과 교육 목적
- **LALR(1)**: 대부분의 실용적인 파서 생성기 (Yacc, Bison, PLY)
- **LR(1)**: LALR(1)에 가짜 리듀스-리듀스 충돌이 있을 때
- **GLR**: 모호한 문법이나 최대한의 유연성이 필요할 때

---

## 연습 문제

### 연습 1: LR(0) 오토마톤 구성

다음 문법에 대한 완전한 LR(0) 오토마톤을 구성하세요:

$$
\begin{aligned}
S' &\to S \\
S &\to A\ B \\
A &\to a \\
B &\to b
\end{aligned}
$$

모든 상태를 아이템 집합과 함께 그리고 모든 전이에 레이블을 붙이세요. 오토마톤은 몇 개의 상태를 가집니까?

### 연습 2: SLR(1) 테이블 구성

증강 문법에 대해:

$$
\begin{aligned}
S' &\to S \\
S &\to C\ C \\
C &\to c\ C \mid d
\end{aligned}
$$

1. LR(0) 오토마톤을 구성하세요.
2. FIRST와 FOLLOW 집합을 계산하세요.
3. SLR(1) 파싱 테이블을 구축하세요.
4. 입력 문자열 `c d c d`의 파스를 추적하세요.

### 연습 3: SLR vs LR(1)

다음 문법을 고려하세요:

$$
\begin{aligned}
S &\to L = R \mid R \\
L &\to * R \mid \textbf{id} \\
R &\to L
\end{aligned}
$$

1. SLR 테이블을 구성하고 충돌을 식별하여 이 문법이 **SLR(1)이 아님**을 보이세요.
2. 이 문법이 **LR(1)임**을 설명하세요 (전체 오토마톤을 구축하지 않고 관련 LR(1) 아이템을 설명해도 됩니다).

### 연습 4: PLY 파서

PLY를 사용하여 (또는 동등한 LALR 파서를 직접 작성하여) 다음을 지원하는 간단한 계산기 언어의 파서를 구현하세요:

- 정수와 부동소수점 숫자
- 산술 연산자: `+`, `-`, `*`, `/`, `**` (거듭제곱)
- 단항 부정: `-x`
- 괄호 표현식
- 변수 대입: `x = expr`
- 출력: `print expr`

모든 연산자에 적절한 우선순위와 결합성을 정의하세요. `x = 2 + 3 * 4`와 `print x ** 2` 같은 입력으로 테스트하세요.

### 연습 5: 충돌 분석

다음 문법에 대해:

```
stmt : IF expr THEN stmt
     | IF expr THEN stmt ELSE stmt
     | OTHER
     ;
```

1. 시프트-리듀스 충돌을 식별하기에 충분한 LALR(1) 오토마톤을 구성하세요.
2. `ELSE`에 `%left` 또는 `%nonassoc` 선언을 적용하면 어떻게 되는지 설명하세요.
3. 댕글링 else에서 "충돌 시 시프트"가 올바른 기본값인 이유를 설명하세요.

### 연습 6: 오류 복구

4.2절의 SLR 파서 구현을 확장하여 오류 생산 규칙을 사용한 오류 복구를 지원하세요. 표현식 문법에 다음 오류 규칙을 추가하세요:

```
E -> error + T     // 잘못된 좌측 피연산자에서 복구
E -> E + error     // 잘못된 우측 피연산자에서 복구
F -> ( error )     // 잘못된 괄호 표현식에서 복구
```

입력 `+ id * id`, `id + * id`, `( + ) * id`로 테스트하세요.

---

[이전: 하향식 파싱](./05_Top_Down_Parsing.md) | [다음: 추상 구문 트리](./07_Abstract_Syntax_Trees.md) | [개요](./00_Overview.md)
