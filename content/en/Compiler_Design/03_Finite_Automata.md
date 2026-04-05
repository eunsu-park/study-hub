# Lesson 3: Finite Automata

## Learning Objectives

After completing this lesson, you will be able to:

1. Give formal definitions of DFA, NFA, and $\epsilon$-NFA
2. Prove the equivalence of NFA and DFA via subset construction
3. Apply the subset construction algorithm step by step
4. Minimize a DFA using Hopcroft's algorithm and understand the Myhill-Nerode theorem
5. State and apply the pumping lemma for regular languages
6. Identify the limitations of regular languages (what finite automata cannot recognize)
7. Use practical tools (Lex, Flex, re2c) for lexer generation
8. Implement NFA simulation, subset construction, and DFA minimization in Python

---

## 1. Formal Definition of a DFA

A **Deterministic Finite Automaton (DFA)** is a 5-tuple:

$$M = (Q, \Sigma, \delta, q_0, F)$$

where:

- $Q$ is a finite, nonempty set of **states**
- $\Sigma$ is a finite **input alphabet**
- $\delta: Q \times \Sigma \rightarrow Q$ is the **transition function** (total function)
- $q_0 \in Q$ is the **start state** (also called initial state)
- $F \subseteq Q$ is the set of **accepting states** (also called final states)

### Key Properties of DFA

1. **Deterministic**: For every state and every input symbol, there is exactly one next state.
2. **Total function**: $\delta$ is defined for all $(q, a) \in Q \times \Sigma$. (If a DFA has "missing" transitions, they implicitly go to a **dead state** $q_d$ with $\delta(q_d, a) = q_d$ for all $a$.)
3. **No $\epsilon$-transitions**: DFAs do not have transitions on the empty string.

### Extended Transition Function

We extend $\delta$ to strings inductively:

$$\hat{\delta}(q, \epsilon) = q$$
$$\hat{\delta}(q, wa) = \delta(\hat{\delta}(q, w), a)$$

where $w \in \Sigma^*$ and $a \in \Sigma$.

### Language of a DFA

The **language recognized** (accepted) by a DFA $M$ is:

$$L(M) = \{w \in \Sigma^* \mid \hat{\delta}(q_0, w) \in F\}$$

### Example DFA

DFA that accepts binary strings ending in `01`:

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

More precisely:

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

### Python Implementation

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

## 2. Formal Definition of an NFA

A **Nondeterministic Finite Automaton (NFA)** is a 5-tuple:

$$N = (Q, \Sigma, \delta, q_0, F)$$

where:

- $Q$ is a finite set of states
- $\Sigma$ is the input alphabet
- $\delta: Q \times \Sigma \rightarrow \mathcal{P}(Q)$ is the transition function (returns a **set** of states)
- $q_0 \in Q$ is the start state
- $F \subseteq Q$ is the set of accepting states

### Key Differences from DFA

1. **Nondeterministic**: $\delta(q, a)$ can return **multiple** states (including the empty set)
2. **Not necessarily total**: A transition may lead to zero states ($\delta(q, a) = \emptyset$)
3. A string is accepted if **there exists** at least one path from $q_0$ to some $f \in F$

### $\epsilon$-NFA

An **$\epsilon$-NFA** extends the NFA by allowing transitions on the empty string $\epsilon$:

$$\delta: Q \times (\Sigma \cup \{\epsilon\}) \rightarrow \mathcal{P}(Q)$$

An $\epsilon$-transition allows the automaton to change state without consuming any input symbol.

### $\epsilon$-Closure

The **$\epsilon$-closure** of a state $q$ is the set of all states reachable from $q$ by following zero or more $\epsilon$-transitions:

$$\text{ECLOSE}(q) = \{q\} \cup \{p \mid q \xrightarrow{\epsilon} \cdots \xrightarrow{\epsilon} p\}$$

For a set of states $S$:

$$\text{ECLOSE}(S) = \bigcup_{q \in S} \text{ECLOSE}(q)$$

### Language of an NFA

An NFA $N$ accepts a string $w = a_1 a_2 \cdots a_n$ if there exists a sequence of states $r_0, r_1, \ldots, r_n$ such that:

1. $r_0 = q_0$ (start in the initial state)
2. $r_{i+1} \in \delta(r_i, a_{i+1})$ for $0 \leq i < n$ (follow transitions)
3. $r_n \in F$ (end in an accepting state)

For an $\epsilon$-NFA, we also allow $\epsilon$-transitions between steps.

### Python Implementation

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

## 3. Equivalence of NFA and DFA

**Theorem**: For every NFA $N$, there exists a DFA $D$ such that $L(N) = L(D)$.

**Proof strategy**: The subset construction algorithm explicitly builds $D$ from $N$.

### Why Equivalence Matters

- NFAs are easier to construct (Thompson's construction)
- DFAs are easier to simulate (deterministic, $O(n)$ time)
- Equivalence lets us use NFAs for specification and DFAs for execution

### The Subset Construction Algorithm (Detailed)

**Input**: NFA $N = (Q_N, \Sigma, \delta_N, q_0, F_N)$

**Output**: DFA $D = (Q_D, \Sigma, \delta_D, d_0, F_D)$

**Algorithm**:

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

Each DFA state is a **subset** of NFA states (hence the name). There are at most $2^{|Q_N|}$ DFA states, but in practice many fewer are reachable.

### Worked Example

Consider the NFA accepting $(a|b)^*abb$:

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

**Step 1**: Compute $d_0 = \text{ECLOSE}(\{0\})$

Starting from state 0, follow $\epsilon$-transitions:
$0 \xrightarrow{\epsilon} 1 \xrightarrow{\epsilon} 2, 4$
$0 \xrightarrow{\epsilon} 7$

$d_0 = \{0, 1, 2, 4, 7\}$

**Step 2**: Process $d_0$ on each symbol

**$d_0$ on $a$**:
$\text{move}(d_0, a) = \delta(2, a) \cup \delta(7, a) = \{3\} \cup \{8\} = \{3, 8\}$
$\text{ECLOSE}(\{3, 8\}) = \{3, 6, 1, 2, 4, 7, 8\} = \{1, 2, 3, 4, 6, 7, 8\}$

Call this $d_1 = \{1, 2, 3, 4, 6, 7, 8\}$.

**$d_0$ on $b$**:
$\text{move}(d_0, b) = \delta(4, b) = \{5\}$
$\text{ECLOSE}(\{5\}) = \{5, 6, 1, 2, 4, 7\} = \{1, 2, 4, 5, 6, 7\}$

Call this $d_2 = \{1, 2, 4, 5, 6, 7\}$.

**Step 3**: Process $d_1$ on each symbol

**$d_1$ on $a$**:
$\text{move}(d_1, a) = \{3, 8\}$
$\text{ECLOSE}(\{3, 8\}) = d_1$ (same set as before)

**$d_1$ on $b$**:
$\text{move}(d_1, b) = \{5, 9\}$
$\text{ECLOSE}(\{5, 9\}) = \{1, 2, 4, 5, 6, 7, 9\}$

Call this $d_3 = \{1, 2, 4, 5, 6, 7, 9\}$.

**Step 4**: Process $d_2$ on each symbol

**$d_2$ on $a$**: $\text{move}(d_2, a) = \{3, 8\}$, ECLOSE = $d_1$
**$d_2$ on $b$**: $\text{move}(d_2, b) = \{5\}$, ECLOSE = $d_2$

**Step 5**: Process $d_3$ on each symbol

**$d_3$ on $a$**: $\text{move}(d_3, a) = \{3, 8\}$, ECLOSE = $d_1$
**$d_3$ on $b$**: $\text{move}(d_3, b) = \{5, 10\}$
$\text{ECLOSE}(\{5, 10\}) = \{1, 2, 4, 5, 6, 7, 10\}$

Call this $d_4 = \{1, 2, 4, 5, 6, 7, 10\}$. Since $10 \in F_N$, $d_4$ is an accepting state.

**Step 6**: Process $d_4$

**$d_4$ on $a$**: $\text{move}(d_4, a) = \{3, 8\}$, ECLOSE = $d_1$
**$d_4$ on $b$**: $\text{move}(d_4, b) = \{5\}$, ECLOSE = $d_2$

**Result DFA**:

| DFA State | NFA States | $a$ | $b$ | Accept? |
|-----------|------------|-----|-----|---------|
| $d_0$ | $\{0,1,2,4,7\}$ | $d_1$ | $d_2$ | No |
| $d_1$ | $\{1,2,3,4,6,7,8\}$ | $d_1$ | $d_3$ | No |
| $d_2$ | $\{1,2,4,5,6,7\}$ | $d_1$ | $d_2$ | No |
| $d_3$ | $\{1,2,4,5,6,7,9\}$ | $d_1$ | $d_4$ | No |
| $d_4$ | $\{1,2,4,5,6,7,10\}$ | $d_1$ | $d_2$ | **Yes** |

### Python Implementation of Subset Construction

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

## 4. DFA Minimization

### Equivalent States

Two DFA states $p$ and $q$ are **equivalent** (written $p \equiv q$) if:

$$\forall w \in \Sigma^*: \hat{\delta}(p, w) \in F \iff \hat{\delta}(q, w) \in F$$

They are **distinguishable** if there exists a string $w$ (called a **distinguishing string** or **witness**) such that one of $\hat{\delta}(p, w)$ and $\hat{\delta}(q, w)$ is accepting and the other is not.

### The Myhill-Nerode Theorem

The **Myhill-Nerode theorem** provides a necessary and sufficient condition for a language to be regular.

**Definition**: For a language $L$ over $\Sigma$, define the **Myhill-Nerode equivalence relation** $\equiv_L$ on $\Sigma^*$:

$$x \equiv_L y \iff (\forall z \in \Sigma^*: xz \in L \iff yz \in L)$$

In words: two strings are equivalent if they are indistinguishable by any suffix -- for every continuation $z$, either both $xz$ and $yz$ are in $L$, or neither is.

**Theorem (Myhill-Nerode)**: A language $L \subseteq \Sigma^*$ is regular if and only if $\equiv_L$ has a **finite** number of equivalence classes. Moreover, the number of equivalence classes equals the number of states in the **minimum DFA** for $L$.

### Consequences

1. The minimum DFA for a regular language is **unique** (up to isomorphism)
2. To prove a language is NOT regular, show that $\equiv_L$ has infinitely many equivalence classes
3. Minimization is the process of merging equivalent states

### Example: Myhill-Nerode Equivalence Classes

For $L = \{w \in \{0,1\}^* \mid w \text{ ends in } 01\}$:

Consider what matters about a prefix $x$:
- Does $x$ end with nothing useful? (Need both 0 and 1 to finish)
- Does $x$ end with 0? (Need just 1 to finish)
- Does $x$ end with 01? (Already in $L$, but more input could change this)

Three equivalence classes:
- $[\epsilon]$: strings not ending in 0. (Includes $\epsilon$, $1$, $11$, $01$, ...)
- $[0]$: strings ending in 0 but not 01. (Includes $0$, $10$, $00$, $110$, ...)
- $[01]$: strings ending in 01. Wait -- $01 \equiv_L \epsilon$ is incorrect since $01 \in L$ but $\epsilon \notin L$.

Let's be more careful. The classes are:
- $C_0$: strings such that $\hat{\delta}(q_0, x) = q_0$: strings not ending in 0 (e.g., $\epsilon$, $1$, $11$, $011$)
- $C_1$: strings such that $\hat{\delta}(q_0, x) = q_1$: strings ending in 0 (e.g., $0$, $10$, $00$, $010$)
- $C_2$: strings such that $\hat{\delta}(q_0, x) = q_2$: strings ending in 01 (e.g., $01$, $001$, $101$)

Three classes $\Rightarrow$ three states in the minimum DFA.

### Hopcroft's Algorithm

**Hopcroft's algorithm** computes the minimum DFA by iteratively refining a partition of states.

```
Algorithm: Hopcroft's DFA Minimization

Input:  DFA M = (Q, Σ, δ, q₀, F)
Output: Minimum DFA M' = (Q', Σ, δ', q₀', F')

1.  P = {F, Q \ F}                    // Initial partition
2.  W = {min(F, Q \ F) by size}       // Worklist (start with smaller set)
3.
4.  while W ≠ ∅:
5.      A = some element from W; remove A from W
6.      for each c ∈ Σ:
7.          X = {q ∈ Q | δ(q, c) ∈ A}    // Pre-image of A on symbol c
8.          for each group Y ∈ P such that Y ∩ X ≠ ∅ and Y \ X ≠ ∅:
9.              // Y must be split into Y ∩ X and Y \ X
10.             replace Y in P with (Y ∩ X) and (Y \ X)
11.             if Y ∈ W:
12.                 replace Y in W with (Y ∩ X) and (Y \ X)
13.             else:
14.                 add min(Y ∩ X, Y \ X) to W   // Add smaller half
15.
16. // Build minimized DFA from partition P
17. For each group G ∈ P, create one state in M'
18. q₀' = group containing q₀
19. F' = {G ∈ P | G ∩ F ≠ ∅}
20. δ'(G₁, c) = G₂ where δ(q, c) ∈ G₂ for any q ∈ G₁
```

**Time complexity**: $O(|\Sigma| \cdot n \log n)$ where $n = |Q|$

### Python Implementation

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

## 5. The Table-Filling Algorithm

An alternative (and simpler) minimization approach is the **table-filling algorithm** (also called the **marking algorithm**), based on finding distinguishable state pairs.

### Algorithm

```
1. For all pairs (p, q) where p ∈ F and q ∉ F (or vice versa):
       Mark (p, q) as distinguishable

2. Repeat until no more changes:
       For all unmarked pairs (p, q):
           For each symbol a ∈ Σ:
               If (δ(p, a), δ(q, a)) is marked as distinguishable:
                   Mark (p, q) as distinguishable

3. All unmarked pairs are equivalent and can be merged
```

**Time complexity**: $O(n^2 |\Sigma|)$ -- simpler but slower than Hopcroft's $O(n \log n)$

### Python Implementation

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

## 6. The Pumping Lemma for Regular Languages

The **pumping lemma** is a necessary condition for regular languages. It is primarily used to **prove that a language is NOT regular**.

### Statement

If $L$ is a regular language, then there exists a constant $p \geq 1$ (the **pumping length**) such that every string $w \in L$ with $|w| \geq p$ can be split into three parts:

$$w = xyz$$

satisfying:

1. $|y| > 0$ (the "pump" is nonempty)
2. $|xy| \leq p$ (the pump is near the beginning)
3. $\forall i \geq 0: xy^iz \in L$ (pumping $y$ any number of times keeps the string in $L$)

### Proof Idea

If $L$ is regular, it is recognized by a DFA with $p$ states. For any string $w$ with $|w| \geq p$, by the pigeonhole principle, some state must be visited twice in the first $p$ steps. The portion of $w$ between the two visits to the same state is $y$, and it can be "pumped" (repeated any number of times).

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

### Using the Pumping Lemma (Proof by Contradiction)

To prove $L$ is not regular:

1. **Assume** $L$ is regular (for contradiction)
2. Then the pumping lemma holds with some pumping length $p$
3. Choose a specific string $w \in L$ with $|w| \geq p$ (choose wisely!)
4. Show that **for all** decompositions $w = xyz$ satisfying conditions 1 and 2, there exists some $i$ where $xy^iz \notin L$
5. This contradicts the pumping lemma, so $L$ is not regular

### Example 1: $L = \{a^n b^n \mid n \geq 0\}$ Is Not Regular

**Proof**:

1. Assume $L$ is regular with pumping length $p$.
2. Choose $w = a^p b^p$. Clearly $w \in L$ and $|w| = 2p \geq p$.
3. By the pumping lemma, $w = xyz$ where $|y| > 0$, $|xy| \leq p$, and $xy^iz \in L$ for all $i \geq 0$.
4. Since $|xy| \leq p$ and $w$ starts with $p$ a's, $y$ consists entirely of a's: $y = a^k$ for some $k \geq 1$.
5. Consider $i = 2$: $xy^2z = a^{p+k} b^p$. Since $k \geq 1$, we have $p + k > p$, so $xy^2z \notin L$.
6. Contradiction. Therefore $L$ is not regular. $\blacksquare$

### Example 2: $L = \{0^{n^2} \mid n \geq 0\}$ Is Not Regular

**Proof**:

1. Assume $L$ is regular with pumping length $p$.
2. Choose $w = 0^{p^2}$. Since $p^2 = p \cdot p$ is a perfect square, $w \in L$.
3. Let $w = xyz$ with $|y| = k > 0$ and $|xy| \leq p$.
4. Consider $i = 2$: $|xy^2z| = p^2 + k$.
5. Since $1 \leq k \leq p$, we have $p^2 < p^2 + k \leq p^2 + p < (p+1)^2 = p^2 + 2p + 1$.
6. So $p^2 < |xy^2z| < (p+1)^2$, meaning $|xy^2z|$ lies between two consecutive perfect squares and is not itself a perfect square.
7. Thus $xy^2z \notin L$. Contradiction. $\blacksquare$

### Python Verification

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

## 7. Limitations of Regular Languages

Regular languages (and finite automata) have fundamental limitations. Here are important languages that are **NOT regular**:

### Non-Regular Languages

| Language | Why Not Regular |
|----------|----------------|
| $\{a^n b^n \mid n \geq 0\}$ | Cannot count matching a's and b's |
| $\{ww \mid w \in \{a,b\}^*\}$ | Cannot remember an arbitrary prefix |
| $\{ww^R \mid w \in \{a,b\}^*\}$ | Cannot match palindromes |
| $\{a^p \mid p \text{ is prime}\}$ | Primes are not periodic |
| $\{a^{n^2} \mid n \geq 0\}$ | Perfect squares are not periodic |
| $\{a^{2^n} \mid n \geq 0\}$ | Powers of 2 grow too fast |

### What Regular Languages CAN Do

| Pattern | Example | Regular? |
|---------|---------|----------|
| Fixed string | `"while"` | Yes |
| Alternatives | `"if" | "else" | "while"` | Yes |
| Repetition | `[a-z]+` | Yes |
| Bounded counting | $a^{\leq 100}$ | Yes (huge DFA, but finite) |
| Modular counting | $\{a^n \mid n \equiv 0 \pmod{3}\}$ | Yes |
| Token patterns | Identifiers, numbers, strings | Yes |

### Implication for Compiler Design

Regular expressions (and DFAs) are perfect for **lexical analysis** because tokens have simple, regular structure. However, they **cannot** handle:

- Matching parentheses: $\{( ^n ) ^n\}$
- Nested structures: if-then-else nesting
- Variable declarations and uses across scopes

These require **context-free grammars** and pushdown automata, which we study in the next lesson.

### Closure Properties of Regular Languages

Regular languages are closed under many operations:

| Operation | Closed? | Construction |
|-----------|---------|-------------|
| Union $L_1 \cup L_2$ | Yes | Product construction or NFA union |
| Intersection $L_1 \cap L_2$ | Yes | Product construction |
| Complement $\overline{L}$ | Yes | Swap accepting/non-accepting in DFA |
| Concatenation $L_1 \cdot L_2$ | Yes | NFA concatenation |
| Kleene star $L^*$ | Yes | NFA star construction |
| Reversal $L^R$ | Yes | Reverse all transitions, swap start/accept |
| Difference $L_1 \setminus L_2$ | Yes | $L_1 \cap \overline{L_2}$ |
| Homomorphism | Yes | Replace each symbol |
| Inverse homomorphism | Yes | Replace each symbol with a set |

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

## 8. Practical Tools for Lexer Generation

### Lex and Flex

**Lex** (1975, Mike Lesk) and its modern replacement **Flex** (Fast Lex) are the standard lexer generators for C/C++.

**Workflow**:

```
tokens.l  --->  [Flex]  --->  lex.yy.c  --->  [GCC]  --->  lexer binary
(spec)                        (C source)
```

**Flex specification structure**:

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

**How Flex works internally**:

1. Converts each pattern (regex) to an NFA
2. Combines all NFAs into one (with a new start state)
3. Applies subset construction to get a DFA
4. Minimizes the DFA
5. Generates a C file with the DFA encoded as a transition table

### re2c

**re2c** generates **direct-coded** lexers (no table lookup) for C, C++, Go, and Rust. It produces faster lexers than Flex because:

1. Transitions are compiled into `switch` or `if-else` chains
2. No indirect memory access for table lookups
3. Better branch prediction by the CPU

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

**ANTLR** (ANother Tool for Language Recognition) generates combined lexer+parser code for multiple target languages (Java, Python, C++, JavaScript, Go, etc.).

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

### Python Tools

| Tool | Type | Notes |
|------|------|-------|
| **PLY** (`ply`) | Lex+Yacc clone | Pure Python, uses docstrings for patterns |
| **Lark** (`lark`) | Earley/LALR parser | Includes lexer, modern API |
| **SLY** (`sly`) | Lex+Yacc (modernized PLY) | Uses decorators and classes |
| **rply** | PLY-compatible | Designed for RPython/PyPy |
| **Pygments** | Lexer library | Syntax highlighting, many languages |

### Comparison of Approaches

| Approach | Pros | Cons |
|----------|------|------|
| Hand-written | Maximum control, best errors | More code to write/maintain |
| Flex/Lex | Standard, fast output | C only, hard to customize errors |
| re2c | Fastest output, direct code | Less portable |
| ANTLR | Multi-language, combined lex+parse | Larger dependency |
| PLY/Lark | Python-native, easy to use | Slower than C-based tools |

---

## 9. Complete Working Example: Regex-to-Minimized-DFA Pipeline

Here is a complete, end-to-end example that takes a regular expression and produces a minimized DFA, demonstrating the entire pipeline from Lessons 2 and 3.

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

## 10. Decision Problems for Regular Languages

Finite automata and regular languages have several **decidable** problems:

| Problem | Question | Decidable? | Algorithm |
|---------|----------|------------|-----------|
| Membership | Is $w \in L(M)$? | Yes | Simulate DFA on $w$: $O(|w|)$ |
| Emptiness | Is $L(M) = \emptyset$? | Yes | Check if any accepting state is reachable: $O(|Q|)$ |
| Finiteness | Is $L(M)$ finite? | Yes | Check for cycles on paths to accepting states |
| Equivalence | Is $L(M_1) = L(M_2)$? | Yes | Minimize both and check isomorphism: $O(n \log n)$ |
| Subset | Is $L(M_1) \subseteq L(M_2)$? | Yes | Check if $L(M_1) \cap \overline{L(M_2)} = \emptyset$ |
| Universality | Is $L(M) = \Sigma^*$? | Yes | Check if complement is empty |

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

## Summary

- A **DFA** $(Q, \Sigma, \delta, q_0, F)$ is a finite automaton where $\delta$ is a total function returning exactly one state.
- An **NFA** allows nondeterminism: $\delta$ returns a **set** of states. An $\epsilon$-NFA additionally allows transitions on $\epsilon$.
- NFAs and DFAs are **equivalent in power**: the **subset construction** converts any NFA to a DFA (potentially with exponentially more states).
- **DFA minimization** (Hopcroft's algorithm) produces the unique smallest DFA for a given regular language, in $O(n \log n)$ time.
- The **Myhill-Nerode theorem** characterizes regular languages by their number of equivalence classes and proves the uniqueness of the minimum DFA.
- The **pumping lemma** is a necessary condition for regularity, used to prove languages are NOT regular.
- Regular languages cannot handle matching/counting (e.g., $a^n b^n$) or arbitrary memory. These require context-free grammars (next lesson).
- Practical tools (Flex, re2c, ANTLR, PLY) automate lexer generation from regex specifications.
- Regular languages are closed under union, intersection, complement, concatenation, Kleene star, and many other operations.

---

## Exercises

### Exercise 1: NFA Construction and Simulation

Construct an NFA (using Thompson's construction or directly) for the regular expression $(ab|ba)^*$. Then:
1. List all states and transitions
2. Simulate the NFA on the strings: `"abba"`, `"abab"`, `"aabb"`, `""`
3. Which strings are accepted?

### Exercise 2: Subset Construction

Given the following NFA:

| State | $a$ | $b$ | $\epsilon$ |
|-------|-----|-----|------------|
| $\rightarrow q_0$ | $\{q_1\}$ | $\emptyset$ | $\{q_2\}$ |
| $q_1$ | $\emptyset$ | $\{q_1, q_3\}$ | $\emptyset$ |
| $q_2$ | $\{q_3\}$ | $\emptyset$ | $\emptyset$ |
| $*q_3$ | $\emptyset$ | $\emptyset$ | $\emptyset$ |

1. Compute the $\epsilon$-closure of each state
2. Apply the subset construction algorithm step by step
3. Draw the resulting DFA transition table
4. What language does this automaton recognize?

### Exercise 3: DFA Minimization

Minimize the following DFA using the table-filling algorithm:

| State | $0$ | $1$ |
|-------|-----|-----|
| $\rightarrow A$ | $B$ | $C$ |
| $B$ | $D$ | $E$ |
| $*C$ | $F$ | $C$ |
| $D$ | $B$ | $E$ |
| $*E$ | $F$ | $C$ |
| $*F$ | $F$ | $C$ |

1. Show the table-filling steps
2. Identify equivalent state pairs
3. Draw the minimized DFA

### Exercise 4: Pumping Lemma Proofs

Use the pumping lemma to prove that the following languages are NOT regular:
1. $L_1 = \{a^n b^{2n} \mid n \geq 0\}$
2. $L_2 = \{w \in \{a,b\}^* \mid w \text{ has equal numbers of } a\text{'s and } b\text{'s}\}$
3. $L_3 = \{a^{n!} \mid n \geq 1\}$ (strings of $a$'s whose length is a factorial)

### Exercise 5: Closure Properties

1. Given DFAs $D_1$ accepting $\{w \mid w \text{ contains "ab"}\}$ over $\{a,b\}$ and $D_2$ accepting $\{w \mid |w| \text{ is even}\}$ over $\{a,b\}$, construct a DFA for $L(D_1) \cap L(D_2)$ using the product construction.
2. Construct a DFA for the complement of $L(D_1)$.

### Exercise 6: Implementation Challenge

Implement the full pipeline in Python:
1. Parse a regular expression (supporting `|`, `*`, `+`, `?`, `.`, character classes `[a-z]`)
2. Build an NFA using Thompson's construction
3. Convert to DFA using subset construction
4. Minimize the DFA using Hopcroft's algorithm
5. Simulate the minimized DFA on input strings
6. Test your implementation on these regular expressions:
   - `[a-z]+[0-9]*` (identifier-like)
   - `(0|1(01*0)*1)*` (binary multiples of 3)
   - `"([^"\\]|\\.)*"` (C-style string literals)

---

[Previous: Lexical Analysis](./02_Lexical_Analysis.md) | [Next: Context-Free Grammars](./04_Context_Free_Grammars.md) | [Overview](./00_Overview.md)
