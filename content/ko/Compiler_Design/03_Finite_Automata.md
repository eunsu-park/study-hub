# 레슨 3: 유한 오토마톤(Finite Automata)

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. DFA, NFA, $\epsilon$-NFA의 형식적 정의를 제시할 수 있다
2. 부분집합 구성법(Subset Construction)을 통해 NFA와 DFA의 동치 관계를 증명할 수 있다
3. 부분집합 구성 알고리즘을 단계별로 적용할 수 있다
4. 홉크로프트 알고리즘(Hopcroft's Algorithm)으로 DFA를 최소화하고 마이힐-네로드 정리(Myhill-Nerode Theorem)를 이해할 수 있다
5. 정규 언어의 펌핑 보조 정리(Pumping Lemma)를 기술하고 적용할 수 있다
6. 정규 언어의 한계(유한 오토마톤이 인식할 수 없는 것)를 파악할 수 있다
7. 렉서 생성을 위한 실용 도구(Lex, Flex, re2c)를 사용할 수 있다
8. Python으로 NFA 시뮬레이션, 부분집합 구성, DFA 최소화를 구현할 수 있다

---

## 1. DFA의 형식적 정의

**결정적 유한 오토마톤(Deterministic Finite Automaton, DFA)**은 5-튜플입니다:

$$M = (Q, \Sigma, \delta, q_0, F)$$

여기서:

- $Q$는 유한하고 비어 있지 않은 **상태** 집합
- $\Sigma$는 유한한 **입력 알파벳**
- $\delta: Q \times \Sigma \rightarrow Q$는 **전이 함수**(전체 함수)
- $q_0 \in Q$는 **시작 상태**(초기 상태라고도 함)
- $F \subseteq Q$는 **수락 상태** 집합(최종 상태라고도 함)

### DFA의 핵심 특성

1. **결정적(Deterministic)**: 모든 상태와 모든 입력 기호에 대해 다음 상태가 정확히 하나 존재합니다.
2. **전체 함수**: $\delta$는 모든 $(q, a) \in Q \times \Sigma$에 대해 정의됩니다. (DFA에 "빠진" 전이가 있으면 모든 $a$에 대해 $\delta(q_d, a) = q_d$인 **죽은 상태(Dead State)** $q_d$로 묵시적으로 전이됩니다.)
3. **$\epsilon$-전이 없음**: DFA는 빈 문자열에 대한 전이가 없습니다.

### 확장 전이 함수

$\delta$를 문자열에 대해 귀납적으로 확장합니다:

$$\hat{\delta}(q, \epsilon) = q$$
$$\hat{\delta}(q, wa) = \delta(\hat{\delta}(q, w), a)$$

여기서 $w \in \Sigma^*$이고 $a \in \Sigma$입니다.

### DFA의 언어

DFA $M$이 **인식(수락)하는 언어**는:

$$L(M) = \{w \in \Sigma^* \mid \hat{\delta}(q_0, w) \in F\}$$

### DFA 예시

`01`로 끝나는 이진 문자열을 수락하는 DFA:

$$M = (\{q_0, q_1, q_2\}, \{0, 1\}, \delta, q_0, \{q_2\})$$

| | 0 | 1 |
|---|---|---|
| $\rightarrow q_0$ | $q_1$ | $q_0$ |
| $q_1$ | $q_1$ | $q_2$ |
| $*q_2$ | $q_1$ | $q_0$ |

```
      0           1           0
q0 -------> q1 -------> q2 -------> q1
 ^   1       |   0       ^   1
 |           |           |
 +-----------+           +--- q0
     (self)
```

더 정확하게는:

```
         1
     +--------+
     |        |
     v    0   |    1          0
   (q0) ----> q1 ----> ((q2)) ----> q1
     ^        |          |
     |   0    |     1    |
     +--------+     +----+
     (goes to q1)   (goes to q0)
```

### Python 구현

```python
from dataclasses import dataclass, field
from typing import Dict, Set, Tuple, Optional, FrozenSet, List


class DFA:
    """
    Deterministic Finite Automaton.
    States are represented as strings for clarity.
    """

    def __init__(
        self,
        states: Set[str],
        alphabet: Set[str],
        transitions: Dict[Tuple[str, str], str],
        start: str,
        accepting: Set[str]
    ):
        self.states = states
        self.alphabet = alphabet
        self.transitions = transitions  # (state, symbol) -> state
        self.start = start
        self.accepting = accepting

        # Validate
        assert start in states, f"Start state {start} not in states"
        assert accepting <= states, "Accepting states must be a subset of states"

    def delta(self, state: str, symbol: str) -> Optional[str]:
        """Transition function. Returns None if undefined (implicit dead state)."""
        return self.transitions.get((state, symbol))

    def delta_hat(self, state: str, word: str) -> Optional[str]:
        """Extended transition function on strings."""
        current = state
        for ch in word:
            current = self.delta(current, ch)
            if current is None:
                return None
        return current

    def accepts(self, word: str) -> bool:
        """Check if the DFA accepts the given word."""
        final_state = self.delta_hat(self.start, word)
        return final_state is not None and final_state in self.accepting

    def print_table(self):
        """Print the transition table."""
        symbols = sorted(self.alphabet)
        header = f"{'State':>10} | " + " | ".join(f"{s:>5}" for s in symbols)
        print(header)
        print("-" * len(header))
        for state in sorted(self.states):
            prefix = "->" if state == self.start else "  "
            suffix = "*" if state in self.accepting else " "
            label = f"{prefix}{suffix}{state}"
            row = f"{label:>10} | "
            cells = []
            for s in symbols:
                target = self.delta(state, s)
                cells.append(f"{target if target else '--':>5}")
            row += " | ".join(cells)
            print(row)


# Example: DFA accepting binary strings ending in "01"
dfa_01 = DFA(
    states={"q0", "q1", "q2"},
    alphabet={"0", "1"},
    transitions={
        ("q0", "0"): "q1",
        ("q0", "1"): "q0",
        ("q1", "0"): "q1",
        ("q1", "1"): "q2",
        ("q2", "0"): "q1",
        ("q2", "1"): "q0",
    },
    start="q0",
    accepting={"q2"}
)

print("=== DFA: Binary strings ending in '01' ===")
dfa_01.print_table()

test_words = ["01", "001", "101", "0101", "10", "1", "0", ""]
print()
for w in test_words:
    result = "accept" if dfa_01.accepts(w) else "reject"
    print(f"  '{w}': {result}")
```

---

## 2. NFA의 형식적 정의

**비결정적 유한 오토마톤(Nondeterministic Finite Automaton, NFA)**은 5-튜플입니다:

$$N = (Q, \Sigma, \delta, q_0, F)$$

여기서:

- $Q$는 유한 상태 집합
- $\Sigma$는 입력 알파벳
- $\delta: Q \times \Sigma \rightarrow \mathcal{P}(Q)$는 전이 함수(상태의 **집합**을 반환)
- $q_0 \in Q$는 시작 상태
- $F \subseteq Q$는 수락 상태 집합

### DFA와의 핵심 차이점

1. **비결정적**: $\delta(q, a)$는 **여러** 상태를 반환할 수 있습니다(빈 집합 포함).
2. **전체 함수가 아닐 수 있음**: 전이가 상태 0개로 연결될 수 있습니다($\delta(q, a) = \emptyset$).
3. 문자열은 $q_0$에서 어떤 $f \in F$까지 **적어도 하나의 경로가 존재**하면 수락됩니다.

### $\epsilon$-NFA

**$\epsilon$-NFA**는 빈 문자열 $\epsilon$에 대한 전이를 허용함으로써 NFA를 확장합니다:

$$\delta: Q \times (\Sigma \cup \{\epsilon\}) \rightarrow \mathcal{P}(Q)$$

$\epsilon$-전이는 오토마톤이 입력 기호를 소비하지 않고 상태를 변경할 수 있게 합니다.

### $\epsilon$-클로저(Closure)

상태 $q$의 **$\epsilon$-클로저**는 0개 이상의 $\epsilon$-전이를 따라 $q$에서 도달할 수 있는 모든 상태의 집합입니다:

$$\text{ECLOSE}(q) = \{q\} \cup \{p \mid q \xrightarrow{\epsilon} \cdots \xrightarrow{\epsilon} p\}$$

상태 집합 $S$에 대해:

$$\text{ECLOSE}(S) = \bigcup_{q \in S} \text{ECLOSE}(q)$$

### NFA의 언어

NFA $N$이 문자열 $w = a_1 a_2 \cdots a_n$을 수락하는 것은, 다음 조건을 만족하는 상태 수열 $r_0, r_1, \ldots, r_n$이 존재함을 의미합니다:

1. $r_0 = q_0$ (초기 상태에서 시작)
2. $r_{i+1} \in \delta(r_i, a_{i+1})$ ($0 \leq i < n$에 대해 전이를 따름)
3. $r_n \in F$ (수락 상태에서 종료)

$\epsilon$-NFA의 경우, 단계 사이에 $\epsilon$-전이도 허용됩니다.

### Python 구현

```python
class NFA:
    """
    Nondeterministic Finite Automaton (with epsilon transitions).
    """

    EPSILON = 'ε'

    def __init__(
        self,
        states: Set[str],
        alphabet: Set[str],
        transitions: Dict[Tuple[str, str], Set[str]],
        start: str,
        accepting: Set[str]
    ):
        self.states = states
        self.alphabet = alphabet  # Should NOT include epsilon
        self.transitions = transitions  # (state, symbol_or_epsilon) -> set of states
        self.start = start
        self.accepting = accepting

    def delta(self, state: str, symbol: str) -> Set[str]:
        """Transition function. Returns set of next states."""
        return self.transitions.get((state, symbol), set())

    def epsilon_closure(self, states: Set[str]) -> Set[str]:
        """Compute epsilon-closure of a set of states using DFS."""
        closure = set(states)
        stack = list(states)
        while stack:
            state = stack.pop()
            for target in self.delta(state, self.EPSILON):
                if target not in closure:
                    closure.add(target)
                    stack.append(target)
        return closure

    def move(self, states: Set[str], symbol: str) -> Set[str]:
        """Compute the set of states reachable from 'states' on 'symbol'."""
        result = set()
        for state in states:
            result |= self.delta(state, symbol)
        return result

    def accepts(self, word: str) -> bool:
        """Check if the NFA accepts the given word."""
        current = self.epsilon_closure({self.start})
        for ch in word:
            current = self.epsilon_closure(self.move(current, ch))
        return bool(current & self.accepting)

    def print_table(self):
        """Print the NFA transition table."""
        symbols = sorted(self.alphabet) + [self.EPSILON]
        header = f"{'State':>10} | " + " | ".join(f"{s:>10}" for s in symbols)
        print(header)
        print("-" * len(header))
        for state in sorted(self.states):
            prefix = "->" if state == self.start else "  "
            suffix = "*" if state in self.accepting else " "
            label = f"{prefix}{suffix}{state}"
            row = f"{label:>10} | "
            cells = []
            for s in symbols:
                targets = self.delta(state, s)
                cell = "{" + ",".join(sorted(targets)) + "}" if targets else "∅"
                cells.append(f"{cell:>10}")
            row += " | ".join(cells)
            print(row)


# Example: NFA accepting strings ending in "01"
nfa_01 = NFA(
    states={"q0", "q1", "q2"},
    alphabet={"0", "1"},
    transitions={
        ("q0", "0"): {"q0", "q1"},  # Nondeterministic: two choices
        ("q0", "1"): {"q0"},
        ("q1", "1"): {"q2"},
    },
    start="q0",
    accepting={"q2"}
)

print("\n=== NFA: Strings ending in '01' ===")
nfa_01.print_table()
print()

for w in ["01", "001", "101", "0101", "10", "1", "0", ""]:
    result = "accept" if nfa_01.accepts(w) else "reject"
    print(f"  '{w}': {result}")


# Example: ε-NFA
nfa_eps = NFA(
    states={"q0", "q1", "q2", "q3"},
    alphabet={"a", "b"},
    transitions={
        ("q0", "ε"): {"q1", "q2"},
        ("q1", "a"): {"q1"},
        ("q1", "b"): {"q3"},
        ("q2", "b"): {"q2"},
        ("q2", "a"): {"q3"},
    },
    start="q0",
    accepting={"q3"}
)

print("\n=== ε-NFA Example ===")
nfa_eps.print_table()
print()

for w in ["a", "b", "ab", "ba", "aab", "bba", ""]:
    result = "accept" if nfa_eps.accepts(w) else "reject"
    print(f"  '{w}': {result}")
```

---

## 3. NFA와 DFA의 동치 관계

**정리**: 모든 NFA $N$에 대해 $L(N) = L(D)$인 DFA $D$가 존재합니다.

**증명 전략**: 부분집합 구성 알고리즘이 $N$으로부터 $D$를 명시적으로 구성합니다.

### 동치 관계가 중요한 이유

- NFA는 구성하기 쉽습니다 (톰슨 구성법)
- DFA는 시뮬레이션하기 쉽습니다 (결정적, $O(n)$ 시간)
- 동치 관계 덕분에 NFA는 명세에, DFA는 실행에 사용할 수 있습니다

### 부분집합 구성 알고리즘 (상세)

**입력**: NFA $N = (Q_N, \Sigma, \delta_N, q_0, F_N)$

**출력**: DFA $D = (Q_D, \Sigma, \delta_D, d_0, F_D)$

**알고리즘**:

```
1. d₀ = ECLOSE({q₀})
2. Q_D = {d₀}
3. unmarked = {d₀}
4. δ_D = {}
5.
6. while unmarked is not empty:
7.     pick and mark state S from unmarked
8.     for each symbol a ∈ Σ:
9.         T = ECLOSE(move(S, a))
10.        if T = ∅:
11.            continue  (or map to dead state)
12.        if T ∉ Q_D:
13.            Q_D = Q_D ∪ {T}
14.            add T to unmarked
15.        δ_D(S, a) = T
16.
17. F_D = {S ∈ Q_D | S ∩ F_N ≠ ∅}
18. return (Q_D, Σ, δ_D, d₀, F_D)
```

각 DFA 상태는 NFA 상태의 **부분집합**입니다(이름의 유래). 최대 $2^{|Q_N|}$개의 DFA 상태가 있을 수 있지만, 실제로는 도달 가능한 상태가 훨씬 적습니다.

### 풀이 예시

$(a|b)^*abb$를 수락하는 NFA를 고려합니다:

```
NFA States: {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
Start: 0
Accept: {10}

Transitions:
  0 --ε--> 1, 7
  1 --ε--> 2, 4
  2 --a--> 3
  3 --ε--> 6
  4 --b--> 5
  5 --ε--> 6
  6 --ε--> 1, 7
  7 --a--> 8
  8 --b--> 9
  9 --b--> 10
```

**1단계**: $d_0 = \text{ECLOSE}(\{0\})$ 계산

상태 0에서 시작하여 $\epsilon$-전이를 따릅니다:
$0 \xrightarrow{\epsilon} 1 \xrightarrow{\epsilon} 2, 4$
$0 \xrightarrow{\epsilon} 7$

$d_0 = \{0, 1, 2, 4, 7\}$

**2단계**: 각 기호에 대해 $d_0$ 처리

**$d_0$에서 $a$**:
$\text{move}(d_0, a) = \delta(2, a) \cup \delta(7, a) = \{3\} \cup \{8\} = \{3, 8\}$
$\text{ECLOSE}(\{3, 8\}) = \{3, 6, 1, 2, 4, 7, 8\} = \{1, 2, 3, 4, 6, 7, 8\}$

이를 $d_1 = \{1, 2, 3, 4, 6, 7, 8\}$이라 합니다.

**$d_0$에서 $b$**:
$\text{move}(d_0, b) = \delta(4, b) = \{5\}$
$\text{ECLOSE}(\{5\}) = \{5, 6, 1, 2, 4, 7\} = \{1, 2, 4, 5, 6, 7\}$

이를 $d_2 = \{1, 2, 4, 5, 6, 7\}$이라 합니다.

**3단계**: 각 기호에 대해 $d_1$ 처리

**$d_1$에서 $a$**:
$\text{move}(d_1, a) = \{3, 8\}$
$\text{ECLOSE}(\{3, 8\}) = d_1$ (이전과 같은 집합)

**$d_1$에서 $b$**:
$\text{move}(d_1, b) = \{5, 9\}$
$\text{ECLOSE}(\{5, 9\}) = \{1, 2, 4, 5, 6, 7, 9\}$

이를 $d_3 = \{1, 2, 4, 5, 6, 7, 9\}$이라 합니다.

**4단계**: 각 기호에 대해 $d_2$ 처리

**$d_2$에서 $a$**: $\text{move}(d_2, a) = \{3, 8\}$, ECLOSE = $d_1$
**$d_2$에서 $b$**: $\text{move}(d_2, b) = \{5\}$, ECLOSE = $d_2$

**5단계**: 각 기호에 대해 $d_3$ 처리

**$d_3$에서 $a$**: $\text{move}(d_3, a) = \{3, 8\}$, ECLOSE = $d_1$
**$d_3$에서 $b$**: $\text{move}(d_3, b) = \{5, 10\}$
$\text{ECLOSE}(\{5, 10\}) = \{1, 2, 4, 5, 6, 7, 10\}$

이를 $d_4 = \{1, 2, 4, 5, 6, 7, 10\}$이라 합니다. $10 \in F_N$이므로 $d_4$는 수락 상태입니다.

**6단계**: $d_4$ 처리

**$d_4$에서 $a$**: $\text{move}(d_4, a) = \{3, 8\}$, ECLOSE = $d_1$
**$d_4$에서 $b$**: $\text{move}(d_4, b) = \{5\}$, ECLOSE = $d_2$

**결과 DFA**:

| DFA 상태 | NFA 상태들 | $a$ | $b$ | 수락? |
|-----------|------------|-----|-----|---------|
| $d_0$ | $\{0,1,2,4,7\}$ | $d_1$ | $d_2$ | 아니오 |
| $d_1$ | $\{1,2,3,4,6,7,8\}$ | $d_1$ | $d_3$ | 아니오 |
| $d_2$ | $\{1,2,4,5,6,7\}$ | $d_1$ | $d_2$ | 아니오 |
| $d_3$ | $\{1,2,4,5,6,7,9\}$ | $d_1$ | $d_4$ | 아니오 |
| $d_4$ | $\{1,2,4,5,6,7,10\}$ | $d_1$ | $d_2$ | **예** |

### 부분집합 구성의 Python 구현

```python
def subset_construction(nfa: NFA) -> DFA:
    """
    Convert an NFA to a DFA using the subset construction algorithm.
    """
    # Start state of DFA
    start_set = frozenset(nfa.epsilon_closure({nfa.start}))

    # Mapping from frozensets to DFA state names
    dfa_states: Dict[FrozenSet[str], str] = {}
    state_counter = 0

    def get_name(state_set: FrozenSet[str]) -> str:
        nonlocal state_counter
        if state_set not in dfa_states:
            dfa_states[state_set] = f"d{state_counter}"
            state_counter += 1
        return dfa_states[state_set]

    start_name = get_name(start_set)
    dfa_transitions: Dict[Tuple[str, str], str] = {}
    dfa_accepting: Set[str] = set()

    unmarked = [start_set]
    marked: Set[FrozenSet[str]] = set()

    while unmarked:
        current_set = unmarked.pop()
        marked.add(current_set)
        current_name = get_name(current_set)

        # Check if this is an accepting state
        if current_set & nfa.accepting:
            dfa_accepting.add(current_name)

        # For each input symbol
        for symbol in sorted(nfa.alphabet):
            # Compute move
            move_result = set()
            for state in current_set:
                move_result |= nfa.delta(state, symbol)

            if not move_result:
                continue

            # Compute epsilon-closure
            target_set = frozenset(nfa.epsilon_closure(move_result))
            target_name = get_name(target_set)

            dfa_transitions[(current_name, symbol)] = target_name

            if target_set not in marked and target_set not in unmarked:
                unmarked.append(target_set)

    all_dfa_states = set(dfa_states.values())

    result = DFA(
        states=all_dfa_states,
        alphabet=nfa.alphabet,
        transitions=dfa_transitions,
        start=start_name,
        accepting=dfa_accepting
    )

    # Print the mapping for reference
    print("State mapping (DFA state -> NFA states):")
    for nfa_set, dfa_name in sorted(dfa_states.items(), key=lambda x: x[1]):
        is_accept = "  (accepting)" if dfa_name in dfa_accepting else ""
        print(f"  {dfa_name} = {{{', '.join(sorted(nfa_set))}}}{is_accept}")
    print()

    return result


# Example: Build NFA for (a|b)*abb using explicit transitions
nfa_abb = NFA(
    states={str(i) for i in range(11)},
    alphabet={"a", "b"},
    transitions={
        ("0", "ε"): {"1", "7"},
        ("1", "ε"): {"2", "4"},
        ("2", "a"): {"3"},
        ("3", "ε"): {"6"},
        ("4", "b"): {"5"},
        ("5", "ε"): {"6"},
        ("6", "ε"): {"1", "7"},
        ("7", "a"): {"8"},
        ("8", "b"): {"9"},
        ("9", "b"): {"10"},
    },
    start="0",
    accepting={"10"}
)

print("=== Subset Construction: (a|b)*abb ===\n")
dfa_abb = subset_construction(nfa_abb)
print("DFA Transition Table:")
dfa_abb.print_table()

# Verify
print("\nVerification:")
for w in ["abb", "aabb", "babb", "ababb", "ab", "ba", ""]:
    nfa_result = nfa_abb.accepts(w)
    dfa_result = dfa_abb.accepts(w)
    match = "OK" if nfa_result == dfa_result else "MISMATCH!"
    print(f"  '{w}': NFA={nfa_result}, DFA={dfa_result} [{match}]")
```

---

## 4. DFA 최소화(DFA Minimization)

### 동치 상태(Equivalent States)

두 DFA 상태 $p$와 $q$가 **동치**(표기: $p \equiv q$)인 것은:

$$\forall w \in \Sigma^*: \hat{\delta}(p, w) \in F \iff \hat{\delta}(q, w) \in F$$

두 상태가 **구별 가능(Distinguishable)**한 것은, $\hat{\delta}(p, w)$와 $\hat{\delta}(q, w)$ 중 하나는 수락 상태이고 다른 하나는 아닌 문자열 $w$(**구별 문자열** 또는 **증인(Witness)**)가 존재하는 것입니다.

### 마이힐-네로드 정리(The Myhill-Nerode Theorem)

**마이힐-네로드 정리**는 언어가 정규(Regular)이기 위한 필요충분조건을 제공합니다.

**정의**: $\Sigma$ 위의 언어 $L$에 대해, $\Sigma^*$ 위의 **마이힐-네로드 동치 관계** $\equiv_L$을 다음과 같이 정의합니다:

$$x \equiv_L y \iff (\forall z \in \Sigma^*: xz \in L \iff yz \in L)$$

즉, 모든 접미사에 의해 구별되지 않는 두 문자열은 동치입니다. 모든 연속 $z$에 대해 $xz$와 $yz$ 모두 $L$에 속하거나, 둘 다 속하지 않습니다.

**정리 (마이힐-네로드)**: 언어 $L \subseteq \Sigma^*$가 정규(Regular)인 것은, $\equiv_L$이 **유한한** 수의 동치류(Equivalence Classes)를 가질 때 동치입니다. 또한 동치류의 수는 $L$에 대한 **최소 DFA**의 상태 수와 같습니다.

### 따름 정리

1. 정규 언어의 최소 DFA는 **유일**합니다(동형(Isomorphism) 기준).
2. 언어가 정규가 아님을 증명하려면, $\equiv_L$이 무한히 많은 동치류를 가짐을 보이면 됩니다.
3. 최소화는 동치 상태를 합치는 과정입니다.

### 예시: 마이힐-네로드 동치류

$L = \{w \in \{0,1\}^* \mid w \text{가 } 01\text{로 끝남}\}$에 대해:

접두사 $x$에서 중요한 것을 생각해봅니다:
- $x$의 끝이 유용하지 않은가? (완성하려면 0과 1 모두 필요)
- $x$가 0으로 끝나는가? (완성하려면 1만 필요)
- $x$가 01로 끝나는가? (이미 $L$에 속하지만 더 입력이 들어오면 달라질 수 있음)

세 개의 동치류:
- $[\epsilon]$: 0으로 끝나지 않는 문자열. ($\epsilon$, $1$, $11$, $01$, ... 포함)
- $[0]$: 0으로 끝나지만 01로 끝나지 않는 문자열. ($0$, $10$, $00$, $110$, ... 포함)
- $[01]$: 01로 끝나는 문자열. 단, $01 \equiv_L \epsilon$은 틀립니다. $01 \in L$이지만 $\epsilon \notin L$이기 때문입니다.

정확히 분류하면:
- $C_0$: $\hat{\delta}(q_0, x) = q_0$인 문자열: 0으로 끝나지 않는 문자열 (예: $\epsilon$, $1$, $11$, $011$)
- $C_1$: $\hat{\delta}(q_0, x) = q_1$인 문자열: 0으로 끝나는 문자열 (예: $0$, $10$, $00$, $010$)
- $C_2$: $\hat{\delta}(q_0, x) = q_2$인 문자열: 01로 끝나는 문자열 (예: $01$, $001$, $101$)

동치류 3개 $\Rightarrow$ 최소 DFA의 상태 3개.

### 홉크로프트 알고리즘(Hopcroft's Algorithm)

**홉크로프트 알고리즘**은 상태 분할을 반복적으로 정제하여 최소 DFA를 계산합니다.

```
Algorithm: Hopcroft's DFA Minimization

Input:  DFA M = (Q, Σ, δ, q₀, F)
Output: Minimum DFA M' = (Q', Σ, δ', q₀', F')

1.  P = {F, Q \ F}                    // 초기 분할
2.  W = {min(F, Q \ F) by size}       // 워크리스트 (더 작은 집합으로 시작)
3.
4.  while W ≠ ∅:
5.      A = some element from W; remove A from W
6.      for each c ∈ Σ:
7.          X = {q ∈ Q | δ(q, c) ∈ A}    // 기호 c로 A의 사전 이미지
8.          for each group Y ∈ P such that Y ∩ X ≠ ∅ and Y \ X ≠ ∅:
9.              // Y를 Y ∩ X와 Y \ X로 분할해야 함
10.             replace Y in P with (Y ∩ X) and (Y \ X)
11.             if Y ∈ W:
12.                 replace Y in W with (Y ∩ X) and (Y \ X)
13.             else:
14.                 add min(Y ∩ X, Y \ X) to W   // 더 작은 절반 추가
15.
16. // 분할 P에서 최소화된 DFA 구성
17. For each group G ∈ P, create one state in M'
18. q₀' = group containing q₀
19. F' = {G ∈ P | G ∩ F ≠ ∅}
20. δ'(G₁, c) = G₂ where δ(q, c) ∈ G₂ for any q ∈ G₁
```

**시간 복잡도**: $O(|\Sigma| \cdot n \log n)$, $n = |Q|$

### Python 구현

```python
def minimize_dfa(dfa: DFA) -> DFA:
    """
    Minimize a DFA using Hopcroft's partition refinement algorithm.
    Returns a new minimized DFA.
    """
    # Step 0: Remove unreachable states
    reachable = set()
    stack = [dfa.start]
    while stack:
        state = stack.pop()
        if state in reachable:
            continue
        reachable.add(state)
        for symbol in dfa.alphabet:
            target = dfa.delta(state, symbol)
            if target is not None:
                stack.append(target)

    states = reachable
    accepting = dfa.accepting & reachable

    # Make the DFA complete (add dead state if needed)
    dead = None
    for state in list(states):
        for symbol in dfa.alphabet:
            if dfa.delta(state, symbol) is None or dfa.delta(state, symbol) not in states:
                if dead is None:
                    dead = "__dead__"
                    states = states | {dead}
                # We'll handle this in the transitions

    # Build a complete transition function
    def complete_delta(state, symbol):
        if state == dead:
            return dead
        target = dfa.delta(state, symbol)
        if target is None or target not in reachable:
            return dead if dead else None
        return target

    # Step 1: Initial partition
    non_accepting = states - accepting
    partition = []
    if accepting:
        partition.append(set(accepting))
    if non_accepting:
        partition.append(set(non_accepting))

    # Helper: find which group a state belongs to
    def find_group(state):
        for i, group in enumerate(partition):
            if state in group:
                return i
        return -1

    # Step 2: Worklist initialization
    worklist = []
    if len(partition) == 2:
        # Add the smaller group
        if len(partition[0]) <= len(partition[1]):
            worklist.append(set(partition[0]))
        else:
            worklist.append(set(partition[1]))
    elif len(partition) == 1:
        worklist.append(set(partition[0]))

    # Step 3: Refinement
    while worklist:
        A = worklist.pop()

        for c in dfa.alphabet:
            # X = states that go to A on symbol c
            X = set()
            for state in states:
                target = complete_delta(state, c)
                if target in A:
                    X.add(state)

            # Try to split each group
            new_partition = []
            for Y in partition:
                Y1 = Y & X
                Y2 = Y - X

                if Y1 and Y2:
                    new_partition.append(Y1)
                    new_partition.append(Y2)

                    # Update worklist
                    if Y in worklist:
                        worklist.remove(Y)
                        worklist.append(Y1)
                        worklist.append(Y2)
                    else:
                        if len(Y1) <= len(Y2):
                            worklist.append(Y1)
                        else:
                            worklist.append(Y2)
                else:
                    new_partition.append(Y)

            partition = new_partition

    # Step 4: Build minimized DFA
    # Remove groups containing only the dead state
    partition = [g for g in partition if g != {dead}] if dead else partition

    # Name the groups
    group_names = {}
    for i, group in enumerate(partition):
        group_names[frozenset(group)] = f"m{i}"

    # State-to-group mapping
    state_to_group = {}
    for group in partition:
        name = group_names[frozenset(group)]
        for state in group:
            state_to_group[state] = name

    # Build transitions
    new_transitions = {}
    new_accepting = set()
    new_start = state_to_group[dfa.start]

    for group in partition:
        name = group_names[frozenset(group)]
        representative = next(iter(group))

        # Check if accepting
        if group & dfa.accepting:
            new_accepting.add(name)

        # Add transitions
        for symbol in dfa.alphabet:
            target = complete_delta(representative, symbol)
            if target is not None and target in state_to_group:
                new_transitions[(name, symbol)] = state_to_group[target]

    new_states = set(group_names.values())

    result = DFA(
        states=new_states,
        alphabet=dfa.alphabet,
        transitions=new_transitions,
        start=new_start,
        accepting=new_accepting
    )

    # Print partition info
    print("Partition groups:")
    for group in partition:
        name = group_names[frozenset(group)]
        is_accept = " (accepting)" if name in new_accepting else ""
        is_start = " (start)" if name == new_start else ""
        print(f"  {name} = {{{', '.join(sorted(group))}}}{is_accept}{is_start}")
    print()

    return result


# Example: Minimize the DFA for (a|b)*abb
print("=== DFA Minimization Example ===\n")
print("Original DFA:")
dfa_abb.print_table()
print()

min_dfa = minimize_dfa(dfa_abb)
print("Minimized DFA:")
min_dfa.print_table()

# Verify
print("\nVerification:")
for w in ["abb", "aabb", "babb", "ababb", "ab", "ba", "", "aababb"]:
    orig = dfa_abb.accepts(w)
    mini = min_dfa.accepts(w)
    match = "OK" if orig == mini else "MISMATCH!"
    print(f"  '{w}': original={orig}, minimized={mini} [{match}]")
```

---

## 5. 테이블 채우기 알고리즘(Table-Filling Algorithm)

**테이블 채우기 알고리즘**(또는 **표시 알고리즘(Marking Algorithm)**)은 구별 가능한 상태 쌍을 찾는 것을 기반으로 하는 대안적(더 단순한) 최소화 방법입니다.

### 알고리즘

```
1. p ∈ F이고 q ∉ F인 (또는 그 반대인) 모든 쌍 (p, q)에 대해:
       (p, q)를 구별 가능으로 표시

2. 변화가 없을 때까지 반복:
       표시되지 않은 모든 쌍 (p, q)에 대해:
           각 기호 a ∈ Σ에 대해:
               (δ(p, a), δ(q, a))가 구별 가능으로 표시되어 있으면:
                   (p, q)를 구별 가능으로 표시

3. 표시되지 않은 모든 쌍은 동치이며 합칠 수 있음
```

**시간 복잡도**: $O(n^2 |\Sigma|)$ — 홉크로프트의 $O(n \log n)$보다 단순하지만 느림

### Python 구현

```python
def table_filling_minimize(dfa: DFA) -> Dict[Tuple[str, str], bool]:
    """
    Table-filling algorithm for finding distinguishable state pairs.
    Returns a dictionary mapping pairs to True (distinguishable) or False (equivalent).
    """
    states = sorted(dfa.states)
    n = len(states)

    # Initialize: all pairs unmarked
    distinguishable = {}
    for i in range(n):
        for j in range(i + 1, n):
            p, q = states[i], states[j]
            distinguishable[(p, q)] = False

    # Step 1: Mark pairs where one is accepting and the other is not
    changed = True
    for i in range(n):
        for j in range(i + 1, n):
            p, q = states[i], states[j]
            if (p in dfa.accepting) != (q in dfa.accepting):
                distinguishable[(p, q)] = True

    # Step 2: Iterate until no changes
    changed = True
    while changed:
        changed = False
        for i in range(n):
            for j in range(i + 1, n):
                p, q = states[i], states[j]
                if distinguishable[(p, q)]:
                    continue

                for symbol in dfa.alphabet:
                    tp = dfa.delta(p, symbol)
                    tq = dfa.delta(q, symbol)

                    if tp is None or tq is None:
                        if tp != tq:
                            distinguishable[(p, q)] = True
                            changed = True
                            break
                        continue

                    if tp == tq:
                        continue

                    # Check if (tp, tq) is marked
                    pair = tuple(sorted([tp, tq]))
                    if pair in distinguishable and distinguishable[pair]:
                        distinguishable[(p, q)] = True
                        changed = True
                        break

    return distinguishable


# Example
print("\n=== Table-Filling Algorithm ===\n")
table = table_filling_minimize(dfa_abb)

states = sorted(dfa_abb.states)
print("Distinguishability table (X = distinguishable, . = equivalent):")
print(f"{'':>5}", end="")
for s in states[:-1]:
    print(f" {s:>4}", end="")
print()

for j in range(1, len(states)):
    print(f"{states[j]:>5}", end="")
    for i in range(j):
        pair = tuple(sorted([states[i], states[j]]))
        mark = "  X " if table.get(pair, False) else "  . "
        print(mark, end="")
    print()

# Find equivalent groups
print("\nEquivalent state pairs:")
for (p, q), is_dist in sorted(table.items()):
    if not is_dist:
        print(f"  {p} ≡ {q}")
```

---

## 6. 정규 언어의 펌핑 보조 정리(The Pumping Lemma)

**펌핑 보조 정리**는 정규 언어에 대한 필요 조건입니다. 주로 **언어가 정규가 아님을 증명**하는 데 사용됩니다.

### 명제

$L$이 정규 언어라면, 모든 문자열 $w \in L$로서 $|w| \geq p$인 것을 세 부분으로 분할할 수 있는 상수 $p \geq 1$ (**펌핑 길이**)가 존재합니다:

$$w = xyz$$

다음 조건을 만족하면서:

1. $|y| > 0$ ("펌프"가 비어 있지 않음)
2. $|xy| \leq p$ (펌프가 시작 부분 근처에 있음)
3. $\forall i \geq 0: xy^iz \in L$ ($y$를 몇 번이든 반복해도 문자열이 $L$에 속함)

### 증명 아이디어

$L$이 정규라면, 상태 수 $p$인 DFA가 인식합니다. $|w| \geq p$인 문자열 $w$에 대해, 비둘기집 원리(Pigeonhole Principle)에 의해 첫 $p$번의 단계에서 어떤 상태가 두 번 이상 방문되어야 합니다. 같은 상태를 두 번 방문하는 사이의 $w$ 부분이 $y$이며, 몇 번이든 "펌핑"(반복)할 수 있습니다.

```
States visited:  q₀ → q₁ → q₂ → ... → qₖ → ... → qⱼ → ... → qₙ
                 |------ x ------|       |---- y ----|---- z ----|
                                  qₖ = qⱼ (pigeonhole!)

Since qₖ = qⱼ, we can:
- Skip y:     x z          (i=0)
- Keep y:     x y z        (i=1, original)
- Repeat y:   x y y z      (i=2)
- Repeat y:   x y y y z    (i=3)
...
```

### 펌핑 보조 정리 사용 (귀류법)

$L$이 정규가 아님을 증명하려면:

1. $L$이 정규라고 **가정**합니다 (귀류를 위해).
2. 그러면 어떤 펌핑 길이 $p$에 대해 펌핑 보조 정리가 성립합니다.
3. $|w| \geq p$인 특정 문자열 $w \in L$을 선택합니다 (신중하게 고를 것!).
4. 조건 1과 2를 만족하는 **모든** 분해 $w = xyz$에 대해, $xy^iz \notin L$인 $i$가 존재함을 보입니다.
5. 이는 펌핑 보조 정리에 모순이 되므로 $L$은 정규가 아닙니다.

### 예시 1: $L = \{a^n b^n \mid n \geq 0\}$는 정규가 아닙니다.

**증명**:

1. 펌핑 길이 $p$를 가진 정규 언어 $L$을 가정합니다.
2. $w = a^p b^p$를 선택합니다. 분명히 $w \in L$이고 $|w| = 2p \geq p$입니다.
3. 펌핑 보조 정리에 의해, $w = xyz$로서 $|y| > 0$, $|xy| \leq p$이고 모든 $i \geq 0$에 대해 $xy^iz \in L$입니다.
4. $|xy| \leq p$이고 $w$가 $p$개의 a로 시작하므로, $y$는 전부 a로 구성됩니다: 어떤 $k \geq 1$에 대해 $y = a^k$.
5. $i = 2$를 고려하면: $xy^2z = a^{p+k} b^p$. $k \geq 1$이므로 $p + k > p$이고, $xy^2z \notin L$입니다.
6. 모순. 따라서 $L$은 정규가 아닙니다. $\blacksquare$

### 예시 2: $L = \{0^{n^2} \mid n \geq 0\}$는 정규가 아닙니다.

**증명**:

1. 펌핑 길이 $p$를 가진 정규 언어 $L$을 가정합니다.
2. $w = 0^{p^2}$를 선택합니다. $p^2 = p \cdot p$는 완전제곱수이므로 $w \in L$입니다.
3. $w = xyz$로서 $|y| = k > 0$, $|xy| \leq p$라 합니다.
4. $i = 2$를 고려하면: $|xy^2z| = p^2 + k$.
5. $1 \leq k \leq p$이므로 $p^2 < p^2 + k \leq p^2 + p < (p+1)^2 = p^2 + 2p + 1$.
6. 따라서 $p^2 < |xy^2z| < (p+1)^2$이고, $|xy^2z|$는 연속하는 두 완전제곱수 사이에 있어 완전제곱수가 아닙니다.
7. 따라서 $xy^2z \notin L$. 모순. $\blacksquare$

### Python 검증

```python
def verify_pumping_lemma(dfa: DFA, p: int):
    """
    Experimentally verify the pumping lemma for a DFA.
    Try all strings of length p and check that they can be pumped.
    """
    from itertools import product

    symbols = sorted(dfa.alphabet)
    violations = 0

    for word_tuple in product(symbols, repeat=p):
        word = ''.join(word_tuple)
        if not dfa.accepts(word):
            continue

        # Try all valid decompositions xyz
        can_pump = False
        for xy_len in range(1, p + 1):
            for y_len in range(1, xy_len + 1):
                x_len = xy_len - y_len
                x = word[:x_len]
                y = word[x_len:xy_len]
                z = word[xy_len:]

                # Check pumping for i = 0, 1, 2, 3
                all_accepted = True
                for i in range(4):
                    pumped = x + y * i + z
                    if not dfa.accepts(pumped):
                        all_accepted = False
                        break

                if all_accepted:
                    can_pump = True
                    break
            if can_pump:
                break

        if not can_pump:
            violations += 1
            print(f"  Warning: '{word}' cannot be pumped!")

    if violations == 0:
        print(f"  All strings of length {p} can be pumped.")
    else:
        print(f"  {violations} violations found.")


# Verify with our DFA
print("\n=== Pumping Lemma Verification ===")
print(f"DFA has {len(dfa_abb.states)} states, so p = {len(dfa_abb.states)}")
verify_pumping_lemma(dfa_abb, len(dfa_abb.states))
```

---

## 7. 정규 언어의 한계

정규 언어(와 유한 오토마톤)에는 근본적인 한계가 있습니다. 다음은 **정규가 아닌** 중요한 언어들입니다:

### 비정규(Non-Regular) 언어들

| 언어 | 정규가 아닌 이유 |
|----------|----------------|
| $\{a^n b^n \mid n \geq 0\}$ | a와 b의 수를 셀 수 없음 |
| $\{ww \mid w \in \{a,b\}^*\}$ | 임의의 접두사를 기억할 수 없음 |
| $\{ww^R \mid w \in \{a,b\}^*\}$ | 회문(Palindrome)을 대응할 수 없음 |
| $\{a^p \mid p \text{는 소수}\}$ | 소수는 주기적이지 않음 |
| $\{a^{n^2} \mid n \geq 0\}$ | 완전제곱수는 주기적이지 않음 |
| $\{a^{2^n} \mid n \geq 0\}$ | 2의 거듭제곱은 너무 빠르게 증가 |

### 정규 언어가 할 수 있는 것

| 패턴 | 예시 | 정규? |
|---------|---------|----------|
| 고정 문자열 | `"while"` | 예 |
| 대안 | `"if" | "else" | "while"` | 예 |
| 반복 | `[a-z]+` | 예 |
| 유계 카운팅 | $a^{\leq 100}$ | 예 (거대한 DFA이지만 유한) |
| 모듈러 카운팅 | $\{a^n \mid n \equiv 0 \pmod{3}\}$ | 예 |
| 토큰 패턴 | 식별자, 숫자, 문자열 | 예 |

### 컴파일러 설계에 대한 함의

정규 표현식(과 DFA)은 토큰이 단순하고 정규적인 구조를 가지므로 **어휘 분석**에 완벽합니다. 그러나 다음은 처리할 수 **없습니다**:

- 괄호 대응: $\{( ^n ) ^n\}$
- 중첩 구조: if-then-else 중첩
- 범위(Scope)를 넘나드는 변수 선언과 사용

이러한 것들은 **문맥 자유 문법(Context-Free Grammar)**과 푸시다운 오토마톤(Pushdown Automaton)을 필요로 하며, 다음 레슨에서 학습합니다.

### 정규 언어의 폐쇄 성질(Closure Properties)

정규 언어는 많은 연산에 대해 닫혀 있습니다:

| 연산 | 닫혀 있음? | 구성 방법 |
|-----------|---------|-------------|
| 합집합 $L_1 \cup L_2$ | 예 | 곱 구성(Product Construction) 또는 NFA 합집합 |
| 교집합 $L_1 \cap L_2$ | 예 | 곱 구성 |
| 여집합(Complement) $\overline{L}$ | 예 | DFA에서 수락/비수락 상태 교환 |
| 연접 $L_1 \cdot L_2$ | 예 | NFA 연접 |
| 클리니 스타 $L^*$ | 예 | NFA 스타 구성 |
| 역전(Reversal) $L^R$ | 예 | 모든 전이 역전, 시작/수락 상태 교환 |
| 차집합 $L_1 \setminus L_2$ | 예 | $L_1 \cap \overline{L_2}$ |
| 동형(Homomorphism) | 예 | 각 기호 교체 |
| 역동형(Inverse Homomorphism) | 예 | 각 기호를 집합으로 교체 |

```python
def dfa_complement(dfa: DFA) -> DFA:
    """
    Construct a DFA that accepts the complement of dfa's language.
    Swap accepting and non-accepting states.
    Note: DFA must be complete (total transition function).
    """
    new_accepting = dfa.states - dfa.accepting
    return DFA(
        states=dfa.states,
        alphabet=dfa.alphabet,
        transitions=dict(dfa.transitions),
        start=dfa.start,
        accepting=new_accepting
    )


def dfa_intersection(dfa1: DFA, dfa2: DFA) -> DFA:
    """
    Construct a DFA that accepts the intersection of two DFAs' languages.
    Uses the product construction.
    """
    assert dfa1.alphabet == dfa2.alphabet

    new_states = set()
    new_transitions = {}
    new_accepting = set()

    # Product states: (q1, q2)
    for s1 in dfa1.states:
        for s2 in dfa2.states:
            state_name = f"({s1},{s2})"
            new_states.add(state_name)

            if s1 in dfa1.accepting and s2 in dfa2.accepting:
                new_accepting.add(state_name)

            for symbol in dfa1.alphabet:
                t1 = dfa1.delta(s1, symbol)
                t2 = dfa2.delta(s2, symbol)
                if t1 is not None and t2 is not None:
                    new_transitions[(state_name, symbol)] = f"({t1},{t2})"

    new_start = f"({dfa1.start},{dfa2.start})"

    return DFA(
        states=new_states,
        alphabet=dfa1.alphabet,
        transitions=new_transitions,
        start=new_start,
        accepting=new_accepting
    )
```

---

## 8. 렉서 생성을 위한 실용 도구

### Lex와 Flex

**Lex**(1975, Mike Lesk)와 그 현대적 대체재인 **Flex**(Fast Lex)는 C/C++용 표준 렉서 생성기입니다.

**워크플로우**:

```
tokens.l  --->  [Flex]  --->  lex.yy.c  --->  [GCC]  --->  lexer binary
(spec)                        (C source)
```

**Flex 명세 구조**:

```
%{
  /* C declarations: includes, globals */
%}

/* Definitions: named patterns */
DIGIT [0-9]
LETTER [a-zA-Z_]

%%
/* Rules: pattern  action */
{DIGIT}+    { return INT_LIT; }
{LETTER}({LETTER}|{DIGIT})*  { return IDENT; }
%%

/* User code: helper functions */
```

**Flex 내부 동작 방식**:

1. 각 패턴(정규 표현식)을 NFA로 변환
2. 모든 NFA를 하나로 결합 (새 시작 상태 추가)
3. 부분집합 구성법으로 DFA 생성
4. DFA 최소화
5. DFA를 전이 테이블로 인코딩한 C 파일 생성

### re2c

**re2c**는 C, C++, Go, Rust용 **직접 코딩(Direct-Coded)** 렉서를 생성합니다. Flex보다 빠른 렉서를 생성하는 이유:

1. 전이가 `switch` 또는 `if-else` 체인으로 컴파일됨
2. 테이블 조회를 위한 간접 메모리 접근 없음
3. CPU의 분기 예측이 더 잘 작동

```c
// re2c specification
/*!re2c
    re2c:define:YYCTYPE = char;
    re2c:define:YYCURSOR = cursor;

    [0-9]+          { return INT_LIT; }
    [a-zA-Z_][a-zA-Z0-9_]* { return IDENT; }
    [ \t\n]+        { goto start; }
    .               { return ERROR; }
*/
```

### ANTLR

**ANTLR**(ANother Tool for Language Recognition)은 여러 대상 언어(Java, Python, C++, JavaScript, Go 등)용 렉서+파서 결합 코드를 생성합니다.

```antlr
// ANTLR lexer grammar
lexer grammar ExprLexer;

INT     : [0-9]+ ;
FLOAT   : [0-9]+ '.' [0-9]+ ;
ID      : [a-zA-Z_] [a-zA-Z0-9_]* ;
PLUS    : '+' ;
MINUS   : '-' ;
STAR    : '*' ;
SLASH   : '/' ;
LPAREN  : '(' ;
RPAREN  : ')' ;
WS      : [ \t\r\n]+ -> skip ;
```

### Python 도구들

| 도구 | 유형 | 비고 |
|------|------|-------|
| **PLY** (`ply`) | Lex+Yacc 클론 | 순수 Python, 패턴에 독스트링 사용 |
| **Lark** (`lark`) | Earley/LALR 파서 | 렉서 포함, 현대적 API |
| **SLY** (`sly`) | Lex+Yacc (현대화된 PLY) | 데코레이터와 클래스 사용 |
| **rply** | PLY 호환 | RPython/PyPy용 설계 |
| **Pygments** | 렉서 라이브러리 | 구문 강조 표시, 다양한 언어 지원 |

### 방식 비교

| 방식 | 장점 | 단점 |
|----------|------|------|
| 직접 작성 | 최대 제어, 최상의 오류 메시지 | 작성/유지 코드가 많음 |
| Flex/Lex | 표준, 빠른 출력 | C 전용, 오류 커스터마이징 어려움 |
| re2c | 가장 빠른 출력, 직접 코드 | 이식성 낮음 |
| ANTLR | 다중 언어, 렉서+파서 결합 | 의존성이 큼 |
| PLY/Lark | Python 네이티브, 사용 쉬움 | C 기반 도구보다 느림 |

---

## 9. 완전한 동작 예시: 정규식 → 최소화된 DFA 파이프라인

다음은 정규 표현식을 받아 최소화된 DFA를 생성하는 완전한 엔드-투-엔드 예시로, 레슨 2와 3의 전체 파이프라인을 보여줍니다.

```python
"""
Complete pipeline: Regex -> NFA -> DFA -> Minimized DFA
Demonstrates Thompson's construction, subset construction, and Hopcroft minimization.
"""


def full_pipeline(regex: str, test_strings: list):
    """Run the complete regex-to-minimized-DFA pipeline."""

    print(f"{'=' * 60}")
    print(f"Regular Expression: {regex}")
    print(f"{'=' * 60}")

    # Step 1: Parse regex and build NFA (Thompson's construction)
    # Using the NFA classes from Lesson 2
    from dataclasses import dataclass, field
    from typing import Dict, Set, List, FrozenSet, Optional, Tuple

    # --- Simplified NFA for this pipeline ---
    class SimpleState:
        _counter = 0

        def __init__(self, is_accept=False):
            self.id = SimpleState._counter
            SimpleState._counter += 1
            self.is_accept = is_accept
            self.transitions = {}  # symbol -> list of states

        def add(self, symbol, target):
            self.transitions.setdefault(symbol, []).append(target)

        def __hash__(self):
            return self.id

        def __eq__(self, other):
            return isinstance(other, SimpleState) and self.id == other.id

        def __repr__(self):
            return f"s{self.id}"

    SimpleState._counter = 0

    class SimpleNFA:
        def __init__(self, start, accept):
            self.start = start
            self.accept = accept

    # Thompson's construction
    def build_nfa(regex):
        pos = [0]

        def peek():
            return regex[pos[0]] if pos[0] < len(regex) else None

        def advance():
            ch = regex[pos[0]]
            pos[0] += 1
            return ch

        def expr():
            left = term()
            while peek() == '|':
                advance()
                right = term()
                s = SimpleState()
                a = SimpleState(True)
                s.add('ε', left.start)
                s.add('ε', right.start)
                left.accept.is_accept = False
                left.accept.add('ε', a)
                right.accept.is_accept = False
                right.accept.add('ε', a)
                left = SimpleNFA(s, a)
            return left

        def term():
            result = factor()
            while peek() not in (None, '|', ')'):
                right = factor()
                result.accept.is_accept = False
                result.accept.add('ε', right.start)
                result = SimpleNFA(result.start, right.accept)
            return result

        def factor():
            base = atom()
            while peek() == '*':
                advance()
                s = SimpleState()
                a = SimpleState(True)
                s.add('ε', base.start)
                s.add('ε', a)
                base.accept.is_accept = False
                base.accept.add('ε', base.start)
                base.accept.add('ε', a)
                base = SimpleNFA(s, a)
            return base

        def atom():
            if peek() == '(':
                advance()
                n = expr()
                advance()  # ')'
                return n
            ch = advance()
            s = SimpleState()
            a = SimpleState(True)
            s.add(ch, a)
            return SimpleNFA(s, a)

        return expr()

    nfa = build_nfa(regex)

    # Collect all states and alphabet
    all_states = set()
    alphabet = set()
    stack = [nfa.start]
    while stack:
        s = stack.pop()
        if s in all_states:
            continue
        all_states.add(s)
        for sym, targets in s.transitions.items():
            if sym != 'ε':
                alphabet.add(sym)
            for t in targets:
                stack.append(t)

    print(f"\n1. NFA: {len(all_states)} states, alphabet = {sorted(alphabet)}")

    # Epsilon closure
    def eclose(states):
        closure = set(states)
        work = list(states)
        while work:
            s = work.pop()
            for t in s.transitions.get('ε', []):
                if t not in closure:
                    closure.add(t)
                    work.append(t)
        return frozenset(closure)

    # NFA simulation
    def nfa_accepts(w):
        current = eclose({nfa.start})
        for ch in w:
            next_s = set()
            for s in current:
                next_s.update(s.transitions.get(ch, []))
            current = eclose(next_s)
        return any(s.is_accept for s in current)

    # Step 2: Subset construction
    start_set = eclose({nfa.start})
    dfa_map = {start_set: 'D0'}
    dfa_trans = {}
    dfa_accept = set()
    counter = 1
    worklist = [start_set]
    processed = set()

    while worklist:
        current = worklist.pop()
        if current in processed:
            continue
        processed.add(current)
        current_name = dfa_map[current]

        if any(s.is_accept for s in current):
            dfa_accept.add(current_name)

        for sym in sorted(alphabet):
            move = set()
            for s in current:
                move.update(s.transitions.get(sym, []))
            if not move:
                continue
            target = eclose(move)
            if target not in dfa_map:
                dfa_map[target] = f'D{counter}'
                counter += 1
            dfa_trans[(current_name, sym)] = dfa_map[target]
            if target not in processed:
                worklist.append(target)

    dfa_states = set(dfa_map.values())
    print(f"2. DFA (subset construction): {len(dfa_states)} states")

    # Print DFA table
    syms = sorted(alphabet)
    print(f"\n   {'State':>8} | " + " | ".join(f"{s:>4}" for s in syms))
    print("   " + "-" * (12 + 7 * len(syms)))
    for state in sorted(dfa_states):
        prefix = "->" if state == 'D0' else "  "
        suffix = "*" if state in dfa_accept else " "
        row = f"   {prefix}{suffix}{state:>5} | "
        cells = []
        for s in syms:
            t = dfa_trans.get((state, s), '--')
            cells.append(f"{t:>4}")
        print(row + " | ".join(cells))

    # Step 3: DFA Minimization (simplified Hopcroft)
    # ... (using the same approach as in Section 4)
    # For brevity, use a simple partition refinement

    # Make DFA complete
    dead = '__dead__'
    need_dead = False
    for state in list(dfa_states):
        for sym in alphabet:
            if (state, sym) not in dfa_trans:
                dfa_trans[(state, sym)] = dead
                need_dead = True
    if need_dead:
        dfa_states.add(dead)
        for sym in alphabet:
            dfa_trans[(dead, sym)] = dead

    # Partition refinement
    non_acc = dfa_states - dfa_accept
    if need_dead and dead in non_acc:
        pass  # dead state is non-accepting

    partition = []
    if dfa_accept:
        partition.append(set(dfa_accept))
    if non_acc:
        partition.append(set(non_acc))

    changed = True
    while changed:
        changed = False
        new_part = []
        for group in partition:
            # Try to split this group
            split = {}
            for state in group:
                sig = tuple(
                    next(
                        (i for i, g in enumerate(partition)
                         if dfa_trans.get((state, sym)) in g),
                        -1
                    )
                    for sym in sorted(alphabet)
                )
                split.setdefault(sig, set()).add(state)
            if len(split) > 1:
                changed = True
            new_part.extend(split.values())
        partition = new_part

    # Remove dead-state group
    partition = [g for g in partition if g != {dead}]

    # Build minimized DFA
    group_name = {}
    for i, g in enumerate(partition):
        for s in g:
            group_name[s] = f'M{i}'

    min_states = set()
    min_trans = {}
    min_accept = set()
    min_start = group_name.get('D0', 'M0')

    for g in partition:
        rep = next(iter(g))
        name = group_name[rep]
        min_states.add(name)
        if g & dfa_accept:
            min_accept.add(name)
        for sym in alphabet:
            target = dfa_trans.get((rep, sym))
            if target and target in group_name:
                min_trans[(name, sym)] = group_name[target]

    print(f"\n3. Minimized DFA: {len(min_states)} states")
    print(f"\n   {'State':>8} | " + " | ".join(f"{s:>4}" for s in syms))
    print("   " + "-" * (12 + 7 * len(syms)))
    for state in sorted(min_states):
        prefix = "->" if state == min_start else "  "
        suffix = "*" if state in min_accept else " "
        row = f"   {prefix}{suffix}{state:>5} | "
        cells = []
        for s in syms:
            t = min_trans.get((state, s), '--')
            cells.append(f"{t:>4}")
        print(row + " | ".join(cells))

    # Step 4: Test
    def min_dfa_accepts(w):
        state = min_start
        for ch in w:
            state = min_trans.get((state, ch))
            if state is None:
                return False
        return state in min_accept

    print(f"\n4. Testing:")
    for w in test_strings:
        nfa_r = nfa_accepts(w)
        min_r = min_dfa_accepts(w)
        match = "OK" if nfa_r == min_r else "MISMATCH"
        status = "accept" if min_r else "reject"
        print(f"   '{w}': {status} [{match}]")

    print()


# Run examples
full_pipeline(
    "(a|b)*abb",
    ["abb", "aabb", "babb", "ababb", "ab", "ba", "", "aababb"]
)

full_pipeline(
    "(a|b)*a(a|b)",
    ["aa", "ab", "ba", "bab", "aab", "bba", "a", "b", ""]
)

full_pipeline(
    "a*b*",
    ["", "a", "b", "ab", "aab", "abb", "aabb", "ba", "aba"]
)
```

---

## 10. 정규 언어의 결정 문제(Decision Problems)

유한 오토마톤과 정규 언어에는 여러 **결정 가능한(Decidable)** 문제들이 있습니다:

| 문제 | 질문 | 결정 가능? | 알고리즘 |
|---------|----------|------------|-----------|
| 멤버십(Membership) | $w \in L(M)$인가? | 예 | $w$에 대해 DFA 시뮬레이션: $O(|w|)$ |
| 공집합(Emptiness) | $L(M) = \emptyset$인가? | 예 | 도달 가능한 수락 상태 존재 여부 확인: $O(|Q|)$ |
| 유한성(Finiteness) | $L(M)$이 유한한가? | 예 | 수락 상태로 가는 경로에 사이클 존재 여부 확인 |
| 동치(Equivalence) | $L(M_1) = L(M_2)$인가? | 예 | 둘 다 최소화 후 동형 여부 확인: $O(n \log n)$ |
| 부분집합(Subset) | $L(M_1) \subseteq L(M_2)$인가? | 예 | $L(M_1) \cap \overline{L(M_2)} = \emptyset$ 확인 |
| 전체성(Universality) | $L(M) = \Sigma^*$인가? | 예 | 여집합이 공집합인지 확인 |

```python
def is_empty(dfa: DFA) -> bool:
    """Check if the DFA's language is empty."""
    reachable = set()
    stack = [dfa.start]
    while stack:
        state = stack.pop()
        if state in reachable:
            continue
        reachable.add(state)
        if state in dfa.accepting:
            return False  # Found reachable accepting state
        for sym in dfa.alphabet:
            target = dfa.delta(state, sym)
            if target and target not in reachable:
                stack.append(target)
    return True


def is_finite(dfa: DFA) -> bool:
    """Check if the DFA's language is finite."""
    # A language is infinite iff there's a cycle on a path
    # from start to some accepting state.
    # 1. Find states reachable from start
    reachable = set()
    stack = [dfa.start]
    while stack:
        state = stack.pop()
        if state in reachable:
            continue
        reachable.add(state)
        for sym in dfa.alphabet:
            target = dfa.delta(state, sym)
            if target:
                stack.append(target)

    # 2. Find states that can reach an accepting state (reverse graph)
    can_reach_accept = set()
    reverse = {s: [] for s in dfa.states}
    for (s, sym), t in dfa.transitions.items():
        reverse.setdefault(t, []).append(s)

    stack = [s for s in dfa.accepting if s in reachable]
    while stack:
        state = stack.pop()
        if state in can_reach_accept:
            continue
        can_reach_accept.add(state)
        for pred in reverse.get(state, []):
            if pred not in can_reach_accept:
                stack.append(pred)

    # 3. Check for cycles among "useful" states
    useful = reachable & can_reach_accept

    # DFS cycle detection on useful states
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {s: WHITE for s in useful}

    def has_cycle(state):
        color[state] = GRAY
        for sym in dfa.alphabet:
            target = dfa.delta(state, sym)
            if target in useful:
                if color[target] == GRAY:
                    return True
                if color[target] == WHITE and has_cycle(target):
                    return True
        color[state] = BLACK
        return False

    for state in useful:
        if color[state] == WHITE:
            if has_cycle(state):
                return False  # Language is infinite

    return True  # Language is finite
```

---

## 요약

- **DFA** $(Q, \Sigma, \delta, q_0, F)$는 $\delta$가 정확히 하나의 상태를 반환하는 전체 함수인 유한 오토마톤입니다.
- **NFA**는 비결정성을 허용합니다: $\delta$는 상태의 **집합**을 반환합니다. $\epsilon$-NFA는 추가로 $\epsilon$에 대한 전이를 허용합니다.
- NFA와 DFA는 **표현 능력이 동치**입니다: **부분집합 구성법**은 임의의 NFA를 DFA로 변환합니다(상태 수가 지수적으로 늘어날 수 있음).
- **DFA 최소화**(홉크로프트 알고리즘)는 주어진 정규 언어에 대해 유일한 가장 작은 DFA를 $O(n \log n)$ 시간에 생성합니다.
- **마이힐-네로드 정리**는 동치류의 수로 정규 언어를 특성화하고 최소 DFA의 유일성을 증명합니다.
- **펌핑 보조 정리**는 정규성의 필요 조건으로, 언어가 정규가 아님을 증명하는 데 사용됩니다.
- 정규 언어는 대응/카운팅(예: $a^n b^n$)이나 임의적 기억을 처리할 수 없습니다. 이를 위해서는 문맥 자유 문법(다음 레슨)이 필요합니다.
- 실용 도구(Flex, re2c, ANTLR, PLY)는 정규식 명세에서 렉서 생성을 자동화합니다.
- 정규 언어는 합집합, 교집합, 여집합, 연접, 클리니 스타 등 다양한 연산에 대해 닫혀 있습니다.

---

## 연습 문제

### 연습 1: NFA 구성과 시뮬레이션

정규 표현식 $(ab|ba)^*$에 대한 NFA를 구성(톰슨 구성법 또는 직접 구성)하세요. 그런 다음:
1. 모든 상태와 전이를 나열하세요.
2. 문자열 `"abba"`, `"abab"`, `"aabb"`, `""`에 대해 NFA를 시뮬레이션하세요.
3. 어떤 문자열이 수락되나요?

### 연습 2: 부분집합 구성

다음 NFA가 주어졌을 때:

| 상태 | $a$ | $b$ | $\epsilon$ |
|-------|-----|-----|------------|
| $\rightarrow q_0$ | $\{q_1\}$ | $\emptyset$ | $\{q_2\}$ |
| $q_1$ | $\emptyset$ | $\{q_1, q_3\}$ | $\emptyset$ |
| $q_2$ | $\{q_3\}$ | $\emptyset$ | $\emptyset$ |
| $*q_3$ | $\emptyset$ | $\emptyset$ | $\emptyset$ |

1. 각 상태의 $\epsilon$-클로저를 계산하세요.
2. 부분집합 구성 알고리즘을 단계별로 적용하세요.
3. 결과 DFA 전이 테이블을 작성하세요.
4. 이 오토마톤이 인식하는 언어는 무엇인가요?

### 연습 3: DFA 최소화

다음 DFA를 테이블 채우기 알고리즘으로 최소화하세요:

| 상태 | $0$ | $1$ |
|-------|-----|-----|
| $\rightarrow A$ | $B$ | $C$ |
| $B$ | $D$ | $E$ |
| $*C$ | $F$ | $C$ |
| $D$ | $B$ | $E$ |
| $*E$ | $F$ | $C$ |
| $*F$ | $F$ | $C$ |

1. 테이블 채우기 단계를 보이세요.
2. 동치 상태 쌍을 파악하세요.
3. 최소화된 DFA를 그리세요.

### 연습 4: 펌핑 보조 정리 증명

펌핑 보조 정리를 사용하여 다음 언어들이 정규가 아님을 증명하세요:
1. $L_1 = \{a^n b^{2n} \mid n \geq 0\}$
2. $L_2 = \{w \in \{a,b\}^* \mid w\text{에서 }a\text{와 }b\text{의 수가 같음}\}$
3. $L_3 = \{a^{n!} \mid n \geq 1\}$ (길이가 팩토리얼인 a의 문자열)

### 연습 5: 폐쇄 성질

1. $\{a,b\}$ 위에서 $\{w \mid w\text{에 "ab"가 포함됨}\}$을 수락하는 DFA $D_1$과 $\{w \mid |w|\text{가 짝수}\}$를 수락하는 DFA $D_2$가 주어졌을 때, 곱 구성을 사용하여 $L(D_1) \cap L(D_2)$를 수락하는 DFA를 구성하세요.
2. $L(D_1)$의 여집합을 수락하는 DFA를 구성하세요.

### 연습 6: 구현 도전

Python으로 완전한 파이프라인을 구현하세요:
1. 정규 표현식 파싱 (`|`, `*`, `+`, `?`, `.`, 문자 클래스 `[a-z]` 지원)
2. 톰슨 구성법으로 NFA 구성
3. 부분집합 구성법으로 DFA 변환
4. 홉크로프트 알고리즘으로 DFA 최소화
5. 최소화된 DFA로 입력 문자열 시뮬레이션
6. 다음 정규 표현식으로 구현을 테스트하세요:
   - `[a-z]+[0-9]*` (식별자 형태)
   - `(0|1(01*0)*1)*` (3의 배수인 이진수)
   - `"([^"\\]|\\.)*"` (C 스타일 문자열 리터럴)

---

[이전: 어휘 분석](./02_Lexical_Analysis.md) | [다음: 문맥 자유 문법](./04_Context_Free_Grammars.md) | [개요](./00_Overview.md)
