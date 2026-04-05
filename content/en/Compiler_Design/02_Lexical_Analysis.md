# Lesson 2: Lexical Analysis

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain the role of the lexer (scanner) in the compilation pipeline
2. Distinguish between tokens, lexemes, and patterns
3. Define regular expressions formally and use them to specify token patterns
4. Apply Thompson's construction to convert a regular expression to an NFA
5. Apply the subset construction to convert an NFA to a DFA
6. Apply Hopcroft's algorithm to minimize a DFA
7. Implement the longest-match rule and handle token priorities
8. Build a complete lexer for a simple programming language in Python

---

## 1. The Role of the Lexer

The **lexer** (also called **scanner** or **tokenizer**) is the first phase of a compiler. It reads the source program as a stream of characters and groups them into meaningful sequences called **tokens**.

```
Source characters:          "if (x >= 42) return y + 1;"
                                    |
                            [Lexer / Scanner]
                                    |
                                    v
Token stream:               KW_IF  LPAREN  ID(x)  GEQ  INT(42)  RPAREN
                            KW_RETURN  ID(y)  PLUS  INT(1)  SEMI
```

### Why Separate Lexical Analysis?

1. **Simplicity**: The parser operates on tokens (a finite, structured alphabet) rather than raw characters. This drastically simplifies the grammar.

2. **Efficiency**: Lexical patterns (identifiers, numbers, strings) are regular and can be recognized by finite automata -- much faster than context-free parsing.

3. **Portability**: Character-set issues (ASCII, UTF-8, line endings) are confined to the lexer.

4. **Modularity**: The lexer and parser can be developed independently. Different source encodings require changes only in the lexer.

### What the Lexer Does

- Groups characters into tokens
- Strips whitespace and comments
- Handles line counting (for error messages)
- Recognizes keywords vs. identifiers
- Handles string and character literal escaping
- Reports lexical errors (illegal characters, unterminated strings)

### What the Lexer Does NOT Do

- Check syntax (that is the parser's job)
- Check types (that is the semantic analyzer's job)
- Handle operator precedence (that is the parser's job)

---

## 2. Tokens, Lexemes, and Patterns

Three related but distinct concepts:

### Token

A **token** is an abstract symbol representing a class of lexical units. It is the output of the lexer and the input to the parser.

A token typically has:
- A **token type** (or token name): `ID`, `INT`, `PLUS`, `KW_IF`, etc.
- An optional **attribute value**: the actual text, numeric value, or symbol table pointer

### Lexeme

A **lexeme** is the actual substring of the source program that matches a token pattern.

### Pattern

A **pattern** is a rule (usually a regular expression) that describes the set of lexemes belonging to a token type.

### Example

| Token Type | Pattern (informal) | Example Lexemes |
|------------|---------------------|-----------------|
| `ID` | Letter followed by letters/digits | `x`, `count`, `myVar` |
| `INT` | One or more digits | `0`, `42`, `1000` |
| `FLOAT` | Digits, dot, digits | `3.14`, `0.001` |
| `STRING` | `"` ... `"` | `"hello"`, `""` |
| `KW_IF` | `if` | `if` |
| `KW_WHILE` | `while` | `while` |
| `PLUS` | `+` | `+` |
| `GEQ` | `>=` | `>=` |
| `ASSIGN` | `=` | `=` |
| `LPAREN` | `(` | `(` |
| `SEMI` | `;` | `;` |

### Token Data Structure

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

## 3. Regular Expressions -- Formal Definition

Token patterns are specified using **regular expressions** (regex). Here we define them formally, as used in compiler theory -- not the extended regex syntax of tools like Python's `re` module.

### Alphabet and Strings

- An **alphabet** $\Sigma$ is a finite, nonempty set of symbols (characters).
- A **string** over $\Sigma$ is a finite sequence of symbols from $\Sigma$.
- The **empty string** is denoted $\epsilon$ (epsilon).
- The **length** of string $w$ is $|w|$. We have $|\epsilon| = 0$.
- $\Sigma^*$ denotes the set of all strings over $\Sigma$ (including $\epsilon$).
- $\Sigma^+ = \Sigma^* \setminus \{\epsilon\}$ is the set of all nonempty strings.

### Language

A **language** $L$ over $\Sigma$ is any subset of $\Sigma^*$ -- that is, any set of strings.

### Regular Expression Definition

A regular expression $r$ over alphabet $\Sigma$ is defined inductively:

**Base cases:**

1. $\epsilon$ is a regular expression denoting the language $\{\epsilon\}$
2. For each $a \in \Sigma$, the symbol $a$ is a regular expression denoting $\{a\}$

**Inductive cases (if $r$ and $s$ are regular expressions denoting $L(r)$ and $L(s)$):**

3. **Union (alternation)**: $r \mid s$ denotes $L(r) \cup L(s)$
4. **Concatenation**: $r \cdot s$ (or simply $rs$) denotes $L(r) \cdot L(s) = \{xy \mid x \in L(r), y \in L(s)\}$
5. **Kleene star (closure)**: $r^*$ denotes $L(r)^* = \bigcup_{i=0}^{\infty} L(r)^i$

where $L^0 = \{\epsilon\}$ and $L^{i+1} = L^i \cdot L$.

**Operator precedence** (highest to lowest):
1. Kleene star $*$ (postfix, binds tightest)
2. Concatenation (implicit, juxtaposition)
3. Union $\mid$ (binds loosest)

All operators are left-associative.

### Examples

| Regex | Language | Description |
|-------|----------|-------------|
| $a$ | $\{a\}$ | Just the string "a" |
| $a \mid b$ | $\{a, b\}$ | Either "a" or "b" |
| $ab$ | $\{ab\}$ | The string "ab" |
| $a^*$ | $\{\epsilon, a, aa, aaa, \ldots\}$ | Zero or more a's |
| $(a \mid b)^*$ | All strings over $\{a, b\}$ | Any combination of a's and b's |
| $a(a \mid b)^*$ | Strings over $\{a,b\}$ starting with $a$ | Starts with a |
| $(0 \mid 1)^* 0$ | Binary strings ending in 0 | Even binary numbers |

### Convenient Shorthands

These are not part of the formal definition but are commonly used:

| Shorthand | Meaning | Formal equivalent |
|-----------|---------|-------------------|
| $r^+$ | One or more | $rr^*$ |
| $r?$ | Zero or one | $r \mid \epsilon$ |
| $[a\text{-}z]$ | Character class | $a \mid b \mid \cdots \mid z$ |
| $[0\text{-}9]$ | Digit | $0 \mid 1 \mid \cdots \mid 9$ |
| $.$ | Any character | $\Sigma$ (union of all) |
| $r\{n\}$ | Exactly $n$ copies | $\underbrace{rr\cdots r}_{n}$ |

### Token Patterns as Regular Expressions

```
Identifier:  [a-zA-Z_][a-zA-Z0-9_]*
Integer:     [0-9]+
Float:       [0-9]+\.[0-9]+([eE][+-]?[0-9]+)?
String:      "[^"\\]*(\\.[^"\\]*)*"
Comment:     //[^\n]*
Whitespace:  [ \t\n\r]+
```

### Algebraic Laws of Regular Expressions

Regular expressions obey the following identities (useful for simplification):

| Law | Statement |
|-----|-----------|
| Union is commutative | $r \mid s = s \mid r$ |
| Union is associative | $(r \mid s) \mid t = r \mid (s \mid t)$ |
| Concatenation is associative | $(rs)t = r(st)$ |
| Concatenation distributes over union | $r(s \mid t) = rs \mid rt$ |
| $\epsilon$ is identity for concatenation | $\epsilon r = r\epsilon = r$ |
| $\emptyset$ is identity for union | $r \mid \emptyset = r$ |
| $\emptyset$ is zero for concatenation | $\emptyset r = r\emptyset = \emptyset$ |
| Star idempotence | $(r^*)^* = r^*$ |
| Star of epsilon | $\epsilon^* = \epsilon$ |

---

## 4. From Regular Expressions to NFA: Thompson's Construction

**Thompson's construction** (Ken Thompson, 1968) converts a regular expression into an equivalent NFA (nondeterministic finite automaton) with $\epsilon$-transitions.

### NFA Definition (Brief)

An NFA is a 5-tuple $(Q, \Sigma, \delta, q_0, F)$ where:
- $Q$ is a finite set of states
- $\Sigma$ is the input alphabet
- $\delta: Q \times (\Sigma \cup \{\epsilon\}) \rightarrow \mathcal{P}(Q)$ is the transition function
- $q_0 \in Q$ is the start state
- $F \subseteq Q$ is the set of accept states

### Construction Rules

Each rule produces an NFA with exactly **one start state** and **one accept state**. No transitions enter the start state or leave the accept state.

**Base case 1: Empty string $\epsilon$**

```
  -->(q0)---ε--->(q1)
     start       accept
```

**Base case 2: Single symbol $a$**

```
  -->(q0)---a--->(q1)
     start       accept
```

**Inductive case 1: Union $r \mid s$**

Given NFAs $N(r)$ with states $(r_0, r_f)$ and $N(s)$ with states $(s_0, s_f)$:

```
              ε ---> [N(r)] ---ε--->
            /                        \
  -->(q0)                              (q_f)
            \                        /
              ε ---> [N(s)] ---ε--->
     start                          accept
```

**Inductive case 2: Concatenation $rs$**

Merge the accept state of $N(r)$ with the start state of $N(s)$:

```
  -->(r0)---[N(r)]--->(r_f = s_0)---[N(s)]--->(s_f)
     start                                     accept
```

**Inductive case 3: Kleene star $r^*$**

```
              ε ---> [N(r)] ---ε--->
            /           |            \
  -->(q0)     <---ε-----+             (q_f)
            \                        /
              ----------ε---------->
     start                          accept
```

### Properties of Thompson's Construction

1. The resulting NFA has at most $2n$ states for a regex of length $n$
2. Each state has at most two outgoing transitions
3. The NFA has exactly one start state and one accept state
4. Construction is linear in the size of the regex: $O(n)$

### Python Implementation

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

## 5. From NFA to DFA: Subset Construction

While NFAs are easy to construct from regular expressions, they are nondeterministic -- a state can have multiple transitions on the same symbol. **DFAs** (deterministic finite automata) are more efficient to simulate because each state has exactly one transition per symbol.

The **subset construction** (also called the **powerset construction**) converts an NFA to an equivalent DFA.

### Algorithm

```
Input:  NFA N = (Q_N, Σ, δ_N, q_0, F_N)
Output: DFA D = (Q_D, Σ, δ_D, d_0, F_D)

1. d_0 = ε-closure({q_0})                  // Start state of DFA
2. Q_D = {d_0}                              // Set of DFA states (each is a set of NFA states)
3. worklist = [d_0]                         // States to process
4. while worklist is not empty:
5.     S = worklist.pop()
6.     for each symbol a in Σ:
7.         T = ε-closure(move(S, a))        // move(S,a) = union of δ_N(s,a) for all s in S
8.         if T is not empty:
9.             if T not in Q_D:
10.                Q_D.add(T)
11.                worklist.append(T)
12.            δ_D(S, a) = T
13. F_D = {S in Q_D | S ∩ F_N ≠ ∅}         // DFA state is accepting if it contains an NFA accept state
```

### Complexity

In the worst case, a DFA can have $2^n$ states for an NFA with $n$ states (hence "powerset" construction). In practice, only a small fraction of these states are reachable.

### Python Implementation

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

## 6. DFA Minimization: Hopcroft's Algorithm

A DFA produced by subset construction may have more states than necessary. **DFA minimization** produces the smallest DFA that recognizes the same language.

### Equivalent States

Two states $p$ and $q$ are **equivalent** (indistinguishable) if for every string $w$:

$$\hat{\delta}(p, w) \in F \iff \hat{\delta}(q, w) \in F$$

In other words, starting from $p$ or $q$, the DFA accepts exactly the same set of strings.

### Hopcroft's Algorithm (Partition Refinement)

The algorithm starts with a coarse partition and refines it:

```
Input:  DFA D = (Q, Σ, δ, q_0, F)
Output: Minimized DFA D' with equivalent states merged

1. Initial partition P = {F, Q \ F}     // Accepting and non-accepting states
2. worklist W = {F}                      // (or the smaller of F, Q\F)
3. while W is not empty:
4.     A = W.pop()
5.     for each symbol c in Σ:
6.         X = {q in Q | δ(q, c) in A}  // States that transition to A on c
7.         for each group Y in P:
8.             Y1 = Y ∩ X                // States in Y that go to A on c
9.             Y2 = Y \ X               // States in Y that don't go to A on c
10.            if Y1 ≠ ∅ and Y2 ≠ ∅:    // Y is split
11.                Replace Y in P with Y1 and Y2
12.                if Y in W:
13.                    Replace Y in W with Y1 and Y2
14.                else:
15.                    Add smaller of Y1, Y2 to W
16. Return DFA with each partition group as a single state
```

**Time complexity**: $O(n \log n)$ where $n = |Q|$

### Python Implementation

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

## 7. Token Recognition

Given DFAs for all token patterns, how does the lexer decide which token to return?

### The Longest Match Rule

The lexer always returns the **longest possible token** starting at the current position.

```
Input:  "iffy = 10;"

Without longest match:  KW_IF("if") ID("fy") ...   WRONG!
With longest match:     ID("iffy") ASSIGN("=") ...  CORRECT!
```

### Priority Rule

When two patterns match the same longest lexeme, the one listed **first** (highest priority) wins. Keywords are typically listed before identifiers.

```
Input:  "if"

Pattern order:
  1. KW_IF:   "if"       <--- wins (higher priority)
  2. ID:      [a-z]+     <--- also matches "if"
```

### Combined DFA Approach

In practice, all token pattern DFAs are combined into a single DFA:

1. Build an NFA for each token pattern, marking the accept state with the token type
2. Create a combined NFA with a new start state and epsilon-transitions to each pattern NFA
3. Convert to DFA using subset construction
4. When a DFA state contains accept states from multiple patterns, choose the highest-priority one

```
            ε ---> NFA(KW_IF)
           /
  start --ε ---> NFA(KW_WHILE)
           \
            ε ---> NFA(ID)
                   ...
```

### Implementing the Longest Match

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

### Error Recovery in Lexers

When the lexer encounters a character that doesn't match any token pattern, it has several options:

1. **Panic mode**: Skip the offending character and report an error
2. **Insert**: Insert a missing character (rare)
3. **Delete**: Delete the offending character
4. **Replace**: Replace the offending character with a valid one

Most compilers use panic mode: report the error and skip one character.

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

## 8. A Complete Lexer Implementation

Here is a complete, self-contained lexer for a C-like language, built from the ground up using the techniques discussed.

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

## 9. Lexer Generator Specification

While hand-written lexers offer maximum control, **lexer generators** (Lex, Flex, PLY) automate the process by taking a specification of token patterns and generating lexer code.

### Lex/Flex Specification Format

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

### PLY (Python Lex-Yacc) Specification

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

## 10. Handling Special Cases

### String Literals with Escapes

String scanning requires handling escape sequences, which makes the pattern context-sensitive at the character level:

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

### Multi-Line Strings

Some languages support multi-line strings:

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

### Nested Comments

Some languages (Haskell, Swift, Rust) support nested comments:

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

### Significant Whitespace (Python-style Indentation)

Python's lexer generates `INDENT` and `DEDENT` tokens based on whitespace changes:

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

## 11. Performance Considerations

### Table-Driven vs. Direct-Coded Lexers

**Table-driven** (generated by Lex/Flex):
- Transition table stored as a 2D array: `next_state = table[state][char]`
- Compact, easy to generate
- Indirect memory access on every character

**Direct-coded** (hand-written or re2c):
- Transitions encoded as `switch` statements or `goto` chains
- Faster: branch prediction works better, no table lookup
- Larger code size

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

**re2c** generates direct-coded lexers and is typically 2-3x faster than Flex.

### Buffer Management

Reading one character at a time from the file system is slow. Lexers use **double buffering**:

```
+-----------------------------------+-----------------------------------+
| Buffer 1 (4096 bytes)             | Buffer 2 (4096 bytes)             |
+-----------------------------------+-----------------------------------+
                    ^                                    ^
                lexeme_begin                          forward

When forward reaches the end of Buffer 1, load next block into Buffer 2.
When forward reaches the end of Buffer 2, load next block into Buffer 1.
```

This is called the **sentinel** technique: place a special EOF character at the end of each buffer so the main scanning loop doesn't need a bounds check on every character.

---

## Summary

- The **lexer** is the first phase of compilation, transforming characters into tokens
- **Tokens** have a type and an optional attribute value; **lexemes** are the actual text; **patterns** are the rules
- Token patterns are specified using **regular expressions**, which support union, concatenation, and Kleene star
- **Thompson's construction** converts a regex to an NFA in $O(n)$ time and space
- **Subset construction** converts an NFA to a DFA; worst case $2^n$ states, but usually much fewer
- **Hopcroft's algorithm** minimizes the DFA in $O(n \log n)$ time
- The **longest match rule** ensures the lexer always returns the longest possible token
- **Priority rules** resolve ties between patterns (keywords before identifiers)
- Hand-written lexers offer flexibility; lexer generators (Lex, Flex, PLY) offer convenience
- Special cases include string escapes, nested comments, and significant whitespace

---

## Exercises

### Exercise 1: Regular Expressions

Write regular expressions for each of the following:

1. Binary strings with an even number of 0's
2. C-style identifiers that start with an underscore
3. Email addresses (simplified: `name@domain.tld`)
4. Floating-point numbers in scientific notation (e.g., `1.5e-3`, `-2.0E+10`)
5. C-style comments: `/* ... */` (non-nested)

### Exercise 2: Thompson's Construction

Apply Thompson's construction to the regex `a(b|c)*d`. Draw the NFA (label all states). Then simulate the NFA on the inputs:
- `"ad"` -- should it accept?
- `"abcd"` -- should it accept?
- `"abbd"` -- should it accept?
- `"abc"` -- should it accept?

### Exercise 3: Subset Construction

Given the NFA from Exercise 2, apply the subset construction algorithm. Show the DFA transition table. How many states does the resulting DFA have?

### Exercise 4: Lexer Extension

Extend the complete lexer in Section 8 to handle:
1. Hexadecimal integer literals (e.g., `0xFF`, `0x1A3`)
2. Octal integer literals (e.g., `0o77`, `0o12`)
3. Binary integer literals (e.g., `0b1010`, `0b11001`)
4. Numeric separators for readability (e.g., `1_000_000`, `0xFF_FF`)

### Exercise 5: Error Recovery

Design and implement an error recovery strategy for the lexer that:
1. Reports unterminated string literals with the line number where the string started
2. Suggests corrections for common typos (e.g., `retrun` -> `return`)
3. Groups consecutive illegal characters into a single error message rather than reporting each one individually

### Exercise 6: Performance Comparison

Implement a simple benchmark that:
1. Generates a large (1MB+) source file with random tokens
2. Tokenizes it using the hand-written lexer from Section 8
3. Tokenizes it using Python's `re` module with a combined regex
4. Compare the tokenization times and verify both produce the same tokens

---

[Previous: Introduction to Compilers](./01_Introduction_to_Compilers.md) | [Next: Finite Automata](./03_Finite_Automata.md) | [Overview](./00_Overview.md)
