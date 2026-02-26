# Lesson 6: Bottom-Up Parsing

## Learning Objectives

After completing this lesson, you will be able to:

1. **Understand** the principles of shift-reduce parsing and how it constructs parse trees from the leaves upward
2. **Define** handles and explain how they guide reduction decisions
3. **Construct** LR(0) item sets and the corresponding automaton
4. **Build** SLR(1), canonical LR(1), and LALR(1) parsing tables
5. **Compare** the strengths and limitations of each LR variant
6. **Use** parser generator tools (Yacc, Bison, PLY) effectively
7. **Resolve** shift-reduce and reduce-reduce conflicts using precedence and associativity
8. **Implement** error recovery strategies in LR parsers

---

## 1. Introduction to Bottom-Up Parsing

Bottom-up parsing builds the parse tree from the **leaves** (terminal symbols) up to the **root** (start symbol). It works by repeatedly finding and reducing **handles** -- substrings of the sentential form that match the right-hand side of a production.

The most common form of bottom-up parsing is **LR parsing**, which stands for:

- **L**: scan input **L**eft to right
- **R**: produce a **R**ightmost derivation (in reverse)

```
Bottom-Up Parse Construction for "id + id * id":

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

### 1.1 Why Bottom-Up?

Bottom-up parsers have significant advantages over top-down parsers:

| Feature | Top-Down (LL) | Bottom-Up (LR) |
|---------|--------------|-----------------|
| Grammar class | LL(1) subset | Virtually all practical CFGs |
| Left recursion | Must be eliminated | Handled naturally |
| Lookahead decisions | Predict which production | Decide when to reduce |
| Ambiguity | Difficult to handle | Resolved by precedence rules |
| Tool support | ANTLR | Yacc, Bison, PLY, tree-sitter |

---

## 2. Shift-Reduce Parsing

### 2.1 The Four Actions

A shift-reduce parser maintains a **stack** of grammar symbols and an **input buffer**. At each step, it performs one of four actions:

1. **Shift**: Push the next input symbol onto the stack
2. **Reduce**: Pop a handle from the stack and push the corresponding nonterminal
3. **Accept**: Parsing is complete -- the stack contains the start symbol and the input is empty
4. **Error**: No valid action exists

```
Shift-Reduce Parsing Visualization:

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

### 2.2 Handles

A **handle** of a right-sentential form $\gamma$ is a production $A \to \beta$ and a position in $\gamma$ where $\beta$ may be found, such that replacing $\beta$ by $A$ produces the previous right-sentential form in a rightmost derivation.

**Formally:** If $S \Rightarrow^*_{rm} \alpha A w \Rightarrow_{rm} \alpha \beta w$, then $A \to \beta$ in the position following $\alpha$ is a handle of $\alpha \beta w$.

**Key insight:** The handle always appears at the top of the stack. This is what makes bottom-up parsing work with a stack -- we never need to look deep into the stack to find the handle.

### 2.3 Viable Prefixes

A **viable prefix** is any prefix of a right-sentential form that can appear on the parsing stack. Equivalently, it is a prefix that does not extend past the right end of the rightmost handle.

The set of viable prefixes is a regular language, which means it can be recognized by a finite automaton. This automaton is exactly the LR(0) automaton we will construct next.

---

## 3. LR(0) Items and the LR(0) Automaton

### 3.1 LR(0) Items

An **LR(0) item** (or simply "item") is a production with a dot (.) at some position in the right-hand side:

$$A \to \alpha \cdot \beta$$

The dot indicates how much of the production has been seen so far:

- $A \to \cdot \alpha\beta$: nothing has been matched yet
- $A \to \alpha \cdot \beta$: $\alpha$ has been matched, $\beta$ is expected
- $A \to \alpha\beta \cdot$: the entire right-hand side has been matched (ready to reduce)

**Example:** The production $E \to E + T$ generates four items:

$$E \to \cdot E + T \qquad E \to E \cdot + T \qquad E \to E + \cdot T \qquad E \to E + T \cdot$$

### 3.2 Closure Operation

The **closure** of a set of items adds all items implied by having a dot before a nonterminal.

**Algorithm:**

```
CLOSURE(I):
    repeat:
        for each item [A -> α . B β] in I:
            for each production B -> γ:
                add [B -> . γ] to I
    until no new items are added
    return I
```

**Intuition:** If we expect to see $B$ next (dot is before $B$), then we must also be prepared to see anything that $B$ can begin with.

### 3.3 GOTO Operation

The **GOTO** function defines transitions between item sets.

$$\text{GOTO}(I, X) = \text{CLOSURE}(\{[A \to \alpha X \cdot \beta] \mid [A \to \alpha \cdot X \beta] \in I\})$$

In words: take all items in $I$ where the dot is before symbol $X$, advance the dot past $X$, then take the closure.

### 3.4 Constructing the LR(0) Automaton

The canonical collection of LR(0) item sets forms the states of a finite automaton that recognizes viable prefixes.

**Algorithm:**

```
ITEMS(G'):
    C = { CLOSURE({[S' -> . S]}) }    // initial state
    repeat:
        for each set of items I in C:
            for each grammar symbol X:
                if GOTO(I, X) is not empty and not in C:
                    add GOTO(I, X) to C
    until no new sets are added
    return C
```

### 3.5 Python Implementation

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

**Example output (partial):**

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

## 4. SLR(1) Parsing

### 4.1 SLR(1) Table Construction

**Simple LR (SLR)** parsing uses the LR(0) automaton and the FOLLOW sets to construct the parsing table. It is the simplest LR variant.

**Algorithm:**

```
Construct SLR(1) Table:

For each state I_i in the canonical collection:

  1. SHIFT: If [A -> α . a β] is in I_i and GOTO(I_i, a) = I_j:
       Set ACTION[i, a] = "shift j"

  2. REDUCE: If [A -> α .] is in I_i (A ≠ S'):
       For each terminal a in FOLLOW(A):
           Set ACTION[i, a] = "reduce A -> α"

  3. ACCEPT: If [S' -> S .] is in I_i:
       Set ACTION[i, $] = "accept"

  4. GOTO: If GOTO(I_i, A) = I_j for nonterminal A:
       Set GOTO[i, A] = j

If any entry is multiply defined, the grammar is not SLR(1).
```

### 4.2 SLR(1) Parser Implementation

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

## 5. Canonical LR(1) Parsing

### 5.1 LR(1) Items

An **LR(1) item** augments the LR(0) item with a **lookahead** terminal:

$$[A \to \alpha \cdot \beta, a]$$

This means: "We are in the process of recognizing $A \to \alpha\beta$, have matched $\alpha$, and if we complete the reduction, the next input symbol should be $a$."

The lookahead is only used when the dot reaches the end (for reduce decisions); shift decisions are the same as LR(0).

### 5.2 LR(1) Closure

The LR(1) closure propagates lookaheads:

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

The critical difference from LR(0) closure is the computation of the lookahead set $\text{FIRST}(\beta a)$. This is how context flows through the parser.

### 5.3 LR(1) Table Construction

The table construction is similar to SLR(1) but uses item-specific lookaheads instead of FOLLOW sets for reduce actions:

```
For each state I_i:
  SHIFT:  If [A -> α . a β, b] in I_i and GOTO(I_i, a) = I_j:
              ACTION[i, a] = "shift j"

  REDUCE: If [A -> α ., a] in I_i (A ≠ S'):
              ACTION[i, a] = "reduce A -> α"
              (Only on lookahead 'a', NOT all of FOLLOW(A))

  ACCEPT: If [S' -> S ., $] in I_i:
              ACTION[i, $] = "accept"
```

### 5.4 LR(1) vs SLR(1)

The key advantage of LR(1) is precision in reduce actions:

```
SLR(1):  Reduce A -> α on ALL terminals in FOLLOW(A)
LR(1):   Reduce A -> α on ONLY the specific lookahead terminal

This means LR(1) can handle grammars that SLR(1) cannot.
```

**Example where SLR(1) fails but LR(1) works:**

$$
\begin{aligned}
S &\to L = R \mid R \\
L &\to * R \mid \textbf{id} \\
R &\to L
\end{aligned}
$$

In SLR(1), there is a shift-reduce conflict in the state containing $[R \to L \cdot]$ and $[S \to L \cdot = R]$ because $=$ is in $\text{FOLLOW}(R)$. In LR(1), the item $[R \to L \cdot, \$]$ has lookahead $\$$, not $=$, so there is no conflict.

### 5.5 Practical Consideration: Table Size

The main disadvantage of canonical LR(1) is the size of the automaton. The number of LR(1) states can be much larger than LR(0) states because items with different lookaheads are kept separate. For typical programming language grammars, LR(1) tables can have thousands of states, while LR(0)/SLR has hundreds.

---

## 6. LALR(1) Parsing

### 6.1 Motivation

**LALR(1)** (Look-Ahead LR) is a practical compromise between SLR(1) and canonical LR(1):

- **More powerful than SLR(1)**: Uses context-sensitive lookaheads
- **Same number of states as SLR(1)**: Merges LR(1) states with identical cores
- **Almost as powerful as LR(1)**: Very few practical grammars are LR(1) but not LALR(1)

### 6.2 Construction by Merging

LALR(1) states are created by merging LR(1) states that have the same **core** (same set of LR(0) items, ignoring lookaheads). The lookahead sets are merged (unioned).

```
LR(1) states:                         LALR(1) (merged):
  State 4a: {[R -> L ., $]}           State 4: {[R -> L ., {$, =}]}
  State 4b: {[R -> L ., =]}
```

### 6.3 Potential New Conflicts

Merging can introduce **reduce-reduce conflicts** that were not present in the canonical LR(1) table. However, it **cannot** introduce new shift-reduce conflicts (because the core determines shift actions).

```
Example of LALR merging introducing a reduce-reduce conflict:

LR(1) State A:  { [X -> α ., a],  [Y -> β ., b] }
LR(1) State B:  { [X -> α ., b],  [Y -> β ., a] }

After merging:  { [X -> α ., {a,b}],  [Y -> β ., {a,b}] }
                                 ↑ reduce-reduce conflict!
```

In practice, this situation is extremely rare for real programming language grammars.

### 6.4 Comparison of LR Variants

```
Grammar Class Hierarchy:

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

| Variant | Reduce decision | # States | Power | Tool |
|---------|----------------|----------|-------|------|
| LR(0) | Always reduce (no lookahead) | Minimal | Weakest | Theoretical |
| SLR(1) | FOLLOW sets | Same as LR(0) | Good | Simple generators |
| LALR(1) | Merged LR(1) lookaheads | Same as LR(0) | Very good | Yacc, Bison, PLY |
| LR(1) | Exact LR(1) lookaheads | Can be much larger | Best | Rarely used directly |

---

## 7. Parser Generator Tools

### 7.1 Yacc and Bison

**Yacc** (Yet Another Compiler Compiler) is the classic Unix parser generator. **Bison** is the GNU version. Both generate LALR(1) parsers from a grammar specification.

**Yacc/Bison grammar format:**

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

Key features:
- **`%left`, `%right`, `%nonassoc`**: Declare operator precedence and associativity
- **`$$`**: Value of the LHS symbol (the result)
- **`$1`, `$2`, `$3`, ...**: Values of RHS symbols
- **`%prec`**: Override default precedence for a specific rule

### 7.2 PLY (Python Lex-Yacc)

**PLY** is a Python implementation of Lex and Yacc. It generates LALR(1) parsers.

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

**Output:**

```
3 + 4 * 5            => ('add', ('num', 3), ('mul', ('num', 4), ('num', 5)))
(a + b) * c          => ('mul', ('add', ('id', 'a'), ('id', 'b')), ('id', 'c'))
x + y + z            => ('add', ('add', ('id', 'x'), ('id', 'y')), ('id', 'z'))
```

### 7.3 Other Parser Generator Tools

| Tool | Language | Parser Type | Notable Feature |
|------|----------|-------------|-----------------|
| **Yacc/Bison** | C/C++ | LALR(1) | Industry standard |
| **PLY** | Python | LALR(1) | Pythonic Yacc |
| **Lark** | Python | Earley/LALR | Elegant EBNF syntax |
| **ANTLR** | Java/Python/... | ALL(*) | Most powerful LL |
| **tree-sitter** | C (with bindings) | GLR | Incremental, error-tolerant |
| **Menhir** | OCaml | LR(1) | Full LR(1), not LALR |
| **Happy** | Haskell | LALR(1) | Monadic parser actions |

---

## 8. Conflict Resolution

### 8.1 Shift-Reduce Conflicts

A **shift-reduce conflict** occurs when the parser can either shift the next input symbol or reduce a handle on the stack.

**Classic example: the dangling else.**

```
stmt : IF expr THEN stmt ELSE stmt
     | IF expr THEN stmt
     | other
     ;
```

When the parser sees `IF expr THEN stmt` with lookahead `ELSE`, it can:
- **Shift** `ELSE` (associate `else` with this `if`)
- **Reduce** `stmt -> IF expr THEN stmt` (associate `else` with an outer `if`)

**Resolution:** Most parser generators default to **shift** (match `else` with the nearest `if`). This is the correct behavior for virtually all programming languages.

### 8.2 Reduce-Reduce Conflicts

A **reduce-reduce conflict** occurs when the parser can reduce by two different productions.

```
stmt : ID '(' expr_list ')'    // function call
     | ID '(' expr_list ')'    // array subscript (hypothetical)
     ;
```

**Resolution strategies:**
- Rewrite the grammar to eliminate the ambiguity
- In Yacc/Bison, the production listed first wins
- Use semantic analysis to distinguish cases (not at the parsing level)

### 8.3 Precedence and Associativity Declarations

Parser generators provide **precedence** and **associativity** declarations to systematically resolve shift-reduce conflicts in expression grammars.

```yacc
/* Precedence: lowest first */
%right '='                  /* assignment: right-associative */
%left OR                    /* logical OR */
%left AND                   /* logical AND */
%left EQ NE                 /* equality */
%left '<' '>' LE GE         /* comparison */
%left '+' '-'               /* additive */
%left '*' '/' '%'           /* multiplicative */
%right UMINUS               /* unary minus (pseudo-token) */
```

**How it works:**

1. Each token gets a **precedence level** (from its declaration position)
2. Each production gets the precedence of its **rightmost terminal**
3. On a shift-reduce conflict:
   - If the shift token has higher precedence than the reduce production: **shift**
   - If the reduce production has higher precedence: **reduce**
   - If equal precedence: use **associativity** (`%left` = reduce, `%right` = shift, `%nonassoc` = error)

**Example:** For the input `3 + 4 * 5`:

```
Stack: ... 3 + 4    Lookahead: *

Production to reduce: expr -> expr + expr  (precedence of '+')
Token to shift: '*'                        (precedence of '*')

Since * > + in precedence: SHIFT

Result: 3 + (4 * 5)  ✓
```

### 8.4 The `%prec` Directive

Sometimes a production needs a different precedence than its rightmost terminal suggests. The `%prec` directive overrides this:

```yacc
expr : '-' expr  %prec UMINUS    /* unary minus */
     {
         $$ = -$2;
     }
     ;
```

Without `%prec UMINUS`, the production `expr -> '-' expr` would have the precedence of `-` (additive level). With `%prec UMINUS`, it gets the higher unary minus precedence.

---

## 9. Error Recovery in LR Parsers

### 9.1 Panic Mode

Similar to top-down parsing, but adapted for the shift-reduce framework:

1. Pop states from the stack until finding a state $s$ where GOTO($s$, $A$) is defined for some error-recovery nonterminal $A$
2. Push $A$ and the GOTO state onto the stack
3. Discard input symbols until finding one on which the parser can continue

### 9.2 Error Productions

Yacc/Bison provide the special `error` token for error recovery:

```yacc
stmt : expr ';'
     | error ';'      /* on error, skip to next semicolon */
     {
         yyerrok;      /* reset error state */
         printf("Recovered from error\n");
     }
     ;

block : '{' stmt_list '}'
      | '{' error '}'    /* on error, skip to matching brace */
      ;
```

**How `error` works:**

1. When a syntax error occurs, the parser pops states until it finds one that can shift `error`
2. It shifts the `error` token
3. It discards input tokens until it can successfully shift after the error production
4. `yyerrok` tells the parser to resume normal error reporting (otherwise it suppresses errors for a few tokens)

### 9.3 Implementation

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

## 10. Advanced Topics

### 10.1 GLR Parsing

**Generalized LR (GLR)** parsing handles ambiguous and nondeterministic grammars by maintaining multiple parse stacks simultaneously. When a conflict occurs, the parser **forks** into multiple parallel parsers.

```
GLR Parsing: Fork on Conflict

    Before conflict:     After fork:
    ┌─────────┐         ┌─────────┐
    │ Stack A │         │ Stack A │ ──shift──▶  Stack A'
    └─────────┘         │         │ ──reduce──▶ Stack A''
                        └─────────┘

    Both stacks continue independently.
    Invalid parses die out; valid ones converge.
```

**Used by:** tree-sitter (for incremental parsing in editors), Elkhound, Bison's `%glr-parser` mode.

### 10.2 Incremental Parsing

Modern editors (VS Code, Neovim) need to re-parse files after every keystroke. **Incremental parsing** reuses previous parse results for unchanged portions of the file.

**tree-sitter** is the most prominent incremental parser:

1. Maintains the parse tree from the previous edit
2. When the user types, only the affected tree nodes are re-parsed
3. Uses a GLR algorithm for robustness with syntactically incorrect code
4. Achieves sub-millisecond parse times for typical edits

### 10.3 Operator Precedence Parsing

For expression-heavy languages, **operator precedence parsing** (also called **Pratt parsing**) provides a simpler alternative to full LR parsing:

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

## 11. Summary

Bottom-up parsing is the dominant strategy in production compilers and parser generators. Its ability to handle left-recursive grammars and its systematic approach to ambiguity resolution make it highly practical.

**Key concepts:**

| Concept | Description |
|---------|-------------|
| **Shift-reduce** | The basic operation: shift input onto stack or reduce a handle |
| **Handle** | The RHS that should be reduced at each step (always at stack top) |
| **LR(0) items** | Productions with a dot marking parse progress |
| **Closure** | Adding implied items when dot is before a nonterminal |
| **GOTO** | Transitioning between item sets by advancing the dot |
| **SLR(1)** | LR(0) automaton + FOLLOW sets for reduce decisions |
| **LR(1)** | Items carry specific lookaheads; most precise |
| **LALR(1)** | Merged LR(1) states; practical sweet spot (Yacc, Bison) |
| **Precedence** | Systematic conflict resolution for operator grammars |

**Which LR variant to use?**

- **SLR(1)**: For simple grammars and educational purposes
- **LALR(1)**: For most practical parser generators (Yacc, Bison, PLY)
- **LR(1)**: When LALR(1) has spurious reduce-reduce conflicts
- **GLR**: For ambiguous grammars or when you need maximum flexibility

---

## Exercises

### Exercise 1: LR(0) Automaton Construction

Construct the complete LR(0) automaton for the following grammar:

$$
\begin{aligned}
S' &\to S \\
S &\to A\ B \\
A &\to a \\
B &\to b
\end{aligned}
$$

Draw all states with their item sets and label all transitions. How many states does the automaton have?

### Exercise 2: SLR(1) Table Construction

For the augmented grammar:

$$
\begin{aligned}
S' &\to S \\
S &\to C\ C \\
C &\to c\ C \mid d
\end{aligned}
$$

1. Construct the LR(0) automaton.
2. Compute FIRST and FOLLOW sets.
3. Build the SLR(1) parsing table.
4. Trace the parse of the input string `c d c d`.

### Exercise 3: SLR vs LR(1)

Consider the grammar:

$$
\begin{aligned}
S &\to L = R \mid R \\
L &\to * R \mid \textbf{id} \\
R &\to L
\end{aligned}
$$

1. Show that this grammar is **not** SLR(1) by constructing the SLR table and identifying the conflict.
2. Explain why the grammar **is** LR(1) (you may describe the relevant LR(1) items without building the full automaton).

### Exercise 4: PLY Parser

Using PLY (or writing an equivalent LALR parser by hand), implement a parser for a simple calculator language that supports:

- Integer and floating-point numbers
- Arithmetic operators: `+`, `-`, `*`, `/`, `**` (power)
- Unary negation: `-x`
- Parenthesized expressions
- Variable assignment: `x = expr`
- Print: `print expr`

Define appropriate precedence and associativity for all operators. Test with inputs like `x = 2 + 3 * 4` and `print x ** 2`.

### Exercise 5: Conflict Analysis

For the grammar:

```
stmt : IF expr THEN stmt
     | IF expr THEN stmt ELSE stmt
     | OTHER
     ;
```

1. Construct enough of the LALR(1) automaton to identify the shift-reduce conflict.
2. Explain what the `%left` or `%nonassoc` declaration would do if applied to `ELSE`.
3. Why is "shift on conflict" the right default for the dangling else?

### Exercise 6: Error Recovery

Extend the SLR parser implementation from Section 4.2 to support error recovery using error productions. Add these error rules to the expression grammar:

```
E -> error + T     // recover from bad left operand
E -> E + error     // recover from bad right operand
F -> ( error )     // recover from bad parenthesized expression
```

Test with inputs: `+ id * id`, `id + * id`, `( + ) * id`.

---

[Previous: Top-Down Parsing](./05_Top_Down_Parsing.md) | [Next: Abstract Syntax Trees](./07_Abstract_Syntax_Trees.md) | [Overview](./00_Overview.md)
