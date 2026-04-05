# Lesson 5: Top-Down Parsing

## Learning Objectives

After completing this lesson, you will be able to:

1. **Understand** the principles of top-down parsing and how it constructs parse trees from the root downward
2. **Compute** FIRST and FOLLOW sets for any context-free grammar
3. **Construct** LL(1) parsing tables and identify LL(1) grammars
4. **Implement** recursive descent parsers by hand for small languages
5. **Implement** table-driven LL(1) parsers from constructed parsing tables
6. **Eliminate** left recursion and apply left factoring to make grammars LL(1)-compatible
7. **Resolve** LL(1) conflicts and apply error recovery strategies
8. **Explain** the extensions LL(k) and ALL(*) parsing provide beyond LL(1)

---

## 1. Introduction to Top-Down Parsing

Top-down parsing is a parsing strategy that builds the parse tree starting from the **root** (the start symbol) and works its way down toward the **leaves** (the terminal symbols). At each step, the parser attempts to predict which production to apply based on the current input token.

The fundamental question in top-down parsing is:

> Given a nonterminal $A$ on top of the parsing stack and the current lookahead token $a$, which production $A \to \alpha$ should the parser apply?

Top-down parsers come in two main forms:

1. **Recursive descent parsers** -- a collection of mutually recursive procedures, one per nonterminal
2. **Table-driven predictive parsers** -- driven by an LL(1) parsing table and an explicit stack

Both forms require the grammar to be **LL(1)** (or close to it). The name LL(1) stands for:

- **L**: scan input **L**eft to right
- **L**: produce a **L**eftmost derivation
- **1**: use **1** token of lookahead

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

The parser proceeds in a **leftmost derivation**: at every step, it expands the leftmost nonterminal.

---

## 2. Recursive Descent Parsing

### 2.1 Basic Concept

In a recursive descent parser, each nonterminal in the grammar corresponds to a function. The function for nonterminal $A$ examines the current lookahead token and decides which production to use, then calls functions for each symbol in the chosen production's right-hand side.

Consider the classic expression grammar (after eliminating left recursion):

$$
\begin{aligned}
E &\to T \; E' \\
E' &\to + \; T \; E' \;\mid\; \varepsilon \\
T &\to F \; T' \\
T' &\to * \; F \; T' \;\mid\; \varepsilon \\
F &\to ( \; E \; ) \;\mid\; \textbf{id}
\end{aligned}
$$

### 2.2 Implementation

Here is a complete recursive descent parser for arithmetic expressions:

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

**Output for "3 + 4 * 5":**

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

### 2.3 Advantages and Limitations

**Advantages:**

- Simple to understand and implement by hand
- Natural error messages (each function knows what it expects)
- Easy to embed semantic actions (AST construction, type checking)
- Can handle some non-LL(1) constructs with manual backtracking

**Limitations:**

- Requires the grammar to be (approximately) LL(1)
- Cannot handle left-recursive grammars directly
- Performance may degrade with excessive backtracking
- Maintaining large parsers by hand is error-prone

---

## 3. FIRST and FOLLOW Sets

The FIRST and FOLLOW sets are the key data structures that make predictive parsing possible. They answer two essential questions:

- **FIRST($\alpha$)**: What terminals can appear as the first symbol of a string derived from $\alpha$?
- **FOLLOW($A$)**: What terminals can appear immediately after nonterminal $A$ in some sentential form?

### 3.1 FIRST Sets

**Definition.** For a string of grammar symbols $\alpha$, $\text{FIRST}(\alpha)$ is the set of terminals that can begin strings derived from $\alpha$. If $\alpha \Rightarrow^* \varepsilon$, then $\varepsilon \in \text{FIRST}(\alpha)$.

**Algorithm to compute FIRST sets:**

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

**Extending FIRST to strings:**

For a string $\alpha = X_1 X_2 \cdots X_n$:

$$
\text{FIRST}(X_1 X_2 \cdots X_n) = \begin{cases}
\text{FIRST}(X_1) & \text{if } \varepsilon \notin \text{FIRST}(X_1) \\
(\text{FIRST}(X_1) - \{\varepsilon\}) \cup \text{FIRST}(X_2 \cdots X_n) & \text{if } \varepsilon \in \text{FIRST}(X_1)
\end{cases}
$$

### 3.2 FOLLOW Sets

**Definition.** For a nonterminal $A$, $\text{FOLLOW}(A)$ is the set of terminals that can appear immediately to the right of $A$ in some sentential form. The end-of-input marker $\$$ is in $\text{FOLLOW}(S)$ where $S$ is the start symbol.

**Algorithm to compute FOLLOW sets:**

```
Algorithm: Compute FOLLOW(A) for all nonterminals A

1. Add $ to FOLLOW(S), where S is the start symbol.

2. For each production A -> α B β:
       Add FIRST(β) - {ε} to FOLLOW(B)

3. For each production A -> α B, or A -> α B β where ε ∈ FIRST(β):
       Add FOLLOW(A) to FOLLOW(B)

Repeat steps 2-3 until no FOLLOW set changes.
```

### 3.3 Python Implementation

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

**Expected output:**

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

### 3.4 Worked Example

Let us trace the FIRST set computation for the expression grammar step by step.

**Iteration 1:**

| Nonterminal | Production | FIRST additions | FIRST set after |
|-------------|-----------|-----------------|-----------------|
| $E'$ | $E' \to \varepsilon$ | $\varepsilon$ | $\{\varepsilon\}$ |
| $T'$ | $T' \to \varepsilon$ | $\varepsilon$ | $\{\varepsilon\}$ |
| $F$ | $F \to (\ E\ )$ | $($ | $\{(\}$ |
| $F$ | $F \to \textbf{id}$ | $\textbf{id}$ | $\{(, \textbf{id}\}$ |

**Iteration 2:**

| Nonterminal | Production | Reasoning | FIRST set after |
|-------------|-----------|-----------|-----------------|
| $T$ | $T \to F\ T'$ | $\text{FIRST}(F) = \{(, \textbf{id}\}$ | $\{(, \textbf{id}\}$ |
| $E'$ | $E' \to +\ T\ E'$ | $\text{FIRST}(+) = \{+\}$ | $\{+, \varepsilon\}$ |
| $T'$ | $T' \to *\ F\ T'$ | $\text{FIRST}(*) = \{*\}$ | $\{*, \varepsilon\}$ |

**Iteration 3:**

| Nonterminal | Production | Reasoning | FIRST set after |
|-------------|-----------|-----------|-----------------|
| $E$ | $E \to T\ E'$ | $\text{FIRST}(T) = \{(, \textbf{id}\}$ | $\{(, \textbf{id}\}$ |

**Iteration 4:** No changes. Fixed point reached.

---

## 4. LL(1) Parsing Table Construction

### 4.1 The Algorithm

An LL(1) parsing table $M$ is a two-dimensional array indexed by nonterminals (rows) and terminals plus $\$$ (columns). Each entry $M[A, a]$ contains a production to apply.

**Algorithm: LL(1) Table Construction**

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

**Definition.** A grammar is **LL(1)** if and only if every entry in the parsing table $M$ contains at most one production.

### 4.2 Python Implementation

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

### 4.3 The Expression Grammar Table

For the expression grammar, the LL(1) table is:

```
             (          )          *          +         id          $
────────────────────────────────────────────────────────────────────────
   E     E->T E'                                    E->T E'
   E'               E'->ε                E'->+T E'              E'->ε
   F     F->(E)                                     F->id
   T     T->F T'                                    T->F T'
   T'               T'->ε     T'->*F T'   T'->ε                 T'->ε
```

**Reading the table:** If the parser sees nonterminal $E$ on top of the stack and the lookahead is `id`, it applies the production $E \to T\ E'$.

---

## 5. Table-Driven LL(1) Parser

### 5.1 The Parsing Algorithm

The table-driven LL(1) parser uses an explicit stack and the parsing table to simulate a leftmost derivation.

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

### 5.2 Complete Implementation

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

**Example trace for "id + id * id":**

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

## 6. Left Recursion Elimination

### 6.1 The Problem

A grammar is **left-recursive** if there exists a nonterminal $A$ such that $A \Rightarrow^+ A\alpha$ for some string $\alpha$. Top-down parsers cannot handle left recursion because parsing $A$ would immediately require parsing $A$ again, leading to infinite recursion.

**Immediate left recursion** has the form:

$$A \to A\alpha_1 \mid A\alpha_2 \mid \cdots \mid A\alpha_m \mid \beta_1 \mid \beta_2 \mid \cdots \mid \beta_n$$

where none of the $\beta_i$ starts with $A$.

### 6.2 Eliminating Immediate Left Recursion

**Transformation rule:**

Replace:
$$A \to A\alpha_1 \mid A\alpha_2 \mid \cdots \mid \beta_1 \mid \beta_2 \mid \cdots$$

With:
$$
\begin{aligned}
A &\to \beta_1 A' \mid \beta_2 A' \mid \cdots \\
A' &\to \alpha_1 A' \mid \alpha_2 A' \mid \cdots \mid \varepsilon
\end{aligned}
$$

**Example:** The left-recursive expression grammar:

$$E \to E + T \mid T$$

Becomes:

$$
\begin{aligned}
E &\to T\ E' \\
E' &\to +\ T\ E' \mid \varepsilon
\end{aligned}
$$

### 6.3 Eliminating Indirect Left Recursion

Indirect (or hidden) left recursion occurs when the cycle involves multiple nonterminals:

$$
\begin{aligned}
A &\to B\alpha \\
B &\to A\beta
\end{aligned}
$$

Here $A \Rightarrow B\alpha \Rightarrow A\beta\alpha$, so $A$ is indirectly left-recursive.

**Algorithm: Eliminate All Left Recursion**

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

### 6.4 Python Implementation

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

## 7. Left Factoring

### 7.1 The Problem

Left factoring is needed when two or more productions for a nonterminal share a common prefix, causing the parser to be unable to decide which production to use.

**Example:**

$$
\text{Stmt} \to \textbf{if}\ E\ \textbf{then}\ S\ \textbf{else}\ S \mid \textbf{if}\ E\ \textbf{then}\ S
$$

Both alternatives start with `if E then S`, so with a single lookahead token the parser cannot choose.

### 7.2 The Transformation

For productions $A \to \alpha\beta_1 \mid \alpha\beta_2$, replace with:

$$
\begin{aligned}
A &\to \alpha\ A' \\
A' &\to \beta_1 \mid \beta_2
\end{aligned}
$$

**After left factoring the if-then-else:**

$$
\begin{aligned}
\text{Stmt} &\to \textbf{if}\ E\ \textbf{then}\ S\ \text{Stmt'} \\
\text{Stmt'} &\to \textbf{else}\ S \mid \varepsilon
\end{aligned}
$$

### 7.3 Implementation

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

## 8. LL(1) Conflicts and Resolution

### 8.1 Types of Conflicts

An LL(1) conflict occurs when a parsing table entry $M[A, a]$ has multiple productions. This can happen for two reasons:

**FIRST-FIRST Conflict:** Two productions for $A$ have overlapping FIRST sets.

$$A \to \alpha \mid \beta \quad \text{where } \text{FIRST}(\alpha) \cap \text{FIRST}(\beta) \neq \emptyset$$

**FIRST-FOLLOW Conflict:** A nullable production's FIRST set overlaps with the nonterminal's FOLLOW set.

$$A \to \alpha \mid \varepsilon \quad \text{where } \text{FIRST}(\alpha) \cap \text{FOLLOW}(A) \neq \emptyset$$

### 8.2 Resolution Strategies

| Conflict Type | Resolution Strategy |
|---|---|
| FIRST-FIRST | Left factoring, grammar rewriting |
| FIRST-FOLLOW | Grammar restructuring, or use LL(k)/ALL(*) |
| Dangling else | Convention: match with nearest unmatched `if` |
| Inherent ambiguity | Rewrite grammar to remove ambiguity |

### 8.3 The Classic Dangling Else

The dangling else problem is perhaps the most famous LL(1) conflict:

```
Stmt -> if Expr then Stmt else Stmt
      | if Expr then Stmt
      | other
```

After left factoring:

```
Stmt  -> if Expr then Stmt Else | other
Else  -> else Stmt | ε
```

This still has a FIRST-FOLLOW conflict for $\text{Else}$ when the lookahead is `else`: both $\text{Else} \to \textbf{else}\ \text{Stmt}$ and $\text{Else} \to \varepsilon$ apply.

**Resolution:** By convention, always choose the non-epsilon production (match `else` with the nearest `if`). In the parsing table, simply prefer the `else Stmt` production.

---

## 9. Error Recovery

When the parser encounters an error (no entry in the parsing table, or a terminal mismatch), it must recover gracefully rather than simply halting.

### 9.1 Panic Mode Recovery

Panic mode is the simplest and most commonly used error recovery strategy. When an error occurs, the parser discards input symbols until it finds a **synchronization token** -- typically a member of the FOLLOW set of the nonterminal being expanded.

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

### 9.2 Phrase-Level Recovery

Phrase-level recovery fills in error entries in the parsing table with specific error-handling routines. For example:

| Situation | Recovery Action |
|---|---|
| Missing operand (`+ * x`) | Insert a dummy operand |
| Missing operator (`x y`) | Insert a dummy operator |
| Unbalanced parenthesis | Insert missing `)` or discard extra `(` |
| Missing semicolon | Insert `;` and continue |

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

### 9.3 Error Productions

Some parser generators allow adding **error productions** to the grammar:

```
Stmt -> error ;    // skip everything until ';'
```

When the parser cannot match any normal production, the `error` token matches and consumes input until the synchronizing token (`;` in this case).

---

## 10. LL(k) and ALL(*) Parsing

### 10.1 LL(k) Parsing

LL(1) uses a single lookahead token. **LL(k)** extends this to $k$ tokens of lookahead. The parsing table becomes indexed by a nonterminal and a $k$-length prefix of the remaining input.

**Definition.** A grammar is LL(k) if for any two leftmost derivations:

$$
\begin{aligned}
S &\Rightarrow^*_{lm} wA\alpha \Rightarrow_{lm} w\beta\alpha \Rightarrow^*_{lm} wx \\
S &\Rightarrow^*_{lm} wA\alpha \Rightarrow_{lm} w\gamma\alpha \Rightarrow^*_{lm} wy
\end{aligned}
$$

If $\text{FIRST}_k(x) = \text{FIRST}_k(y)$, then $\beta = \gamma$ (the same production was used).

**Practical impact:** As $k$ increases, the parsing table grows exponentially ($O(|\Sigma|^k)$ columns). In practice, $k > 2$ is rare for hand-built LL parsers.

### 10.2 ALL(*) Parsing

**ALL(*)** (Adaptive LL) is the parsing algorithm used by ANTLR 4. It provides the power of unlimited lookahead without the exponential table size.

**Key ideas:**

1. **Arbitrary lookahead**: Instead of a fixed $k$, ALL(*) examines as many tokens as needed to resolve the prediction
2. **Augmented Transition Networks (ATNs)**: The grammar is represented as a set of recursive state machines
3. **Dynamic analysis**: At runtime, if the ATN simulation encounters a prediction that requires more context, it performs a **lookahead DFA** construction
4. **Caching**: Once a prediction is resolved for a given lookahead context, the result is cached as a DFA for future reuse

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

**Advantages of ALL(*):**

- Handles any deterministic context-free language
- No manual left-factoring or left-recursion elimination needed (ANTLR 4 handles these automatically for direct left recursion)
- Error messages can be very precise
- Practical performance is often near-linear

**Example in ANTLR 4:**

```
// ANTLR 4 grammar -- no left factoring needed
expr : expr '*' expr    // direct left recursion is OK
     | expr '+' expr
     | '(' expr ')'
     | ID
     | NUM
     ;
```

### 10.3 Comparison of Top-Down Parsing Strategies

| Feature | LL(1) | LL(k) | ALL(*) |
|---------|-------|-------|--------|
| Lookahead | 1 token | $k$ tokens | Unbounded |
| Table size | $O(|\Sigma|)$ | $O(|\Sigma|^k)$ | Dynamic DFA |
| Left recursion | Must eliminate | Must eliminate | Direct LR auto-handled |
| Left factoring | Required | Sometimes | Not needed |
| Parser generator | Hand-written / simple | Uncommon | ANTLR 4 |
| Error recovery | Manual | Manual | Automatic (ANTLR) |
| Performance | $O(n)$ | $O(n)$ | $O(n)$ typical, $O(n^2)$ worst |

---

## 11. Putting It All Together

Let us build a complete mini-language parser that demonstrates the full pipeline from grammar to parsed output.

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

## 12. Summary

Top-down parsing builds parse trees from the root to the leaves, guided by lookahead tokens and prediction tables.

**Key concepts:**

| Concept | Description |
|---------|-------------|
| **Recursive descent** | One function per nonterminal; the most intuitive parsing method |
| **FIRST sets** | Terminals that can begin strings derived from a grammar symbol |
| **FOLLOW sets** | Terminals that can appear after a nonterminal in any derivation |
| **LL(1) table** | Maps (nonterminal, terminal) pairs to productions |
| **Left recursion elimination** | Required for top-down parsing; transforms $A \to A\alpha$ patterns |
| **Left factoring** | Factors out common prefixes to resolve FIRST-FIRST conflicts |
| **Panic mode** | Error recovery by skipping to synchronization tokens |
| **ALL(*)** | Adaptive LL parsing with unbounded lookahead (ANTLR 4) |

**When to use top-down parsing:**

- When the grammar is naturally LL(1) or close to it
- When you want to write a parser by hand (recursive descent)
- When you need fine-grained control over error messages
- When using ANTLR 4 (which uses ALL(*) internally)

**When NOT to use top-down parsing:**

- Grammars with inherent left recursion that is hard to eliminate
- Operator-heavy languages where precedence climbing or Pratt parsing is simpler
- When a parser generator like Yacc/Bison (LR-based) is already in your toolchain

---

## Exercises

### Exercise 1: FIRST and FOLLOW Computation

Compute the FIRST and FOLLOW sets for the following grammar:

$$
\begin{aligned}
S &\to A\ B \\
A &\to a\ A \mid \varepsilon \\
B &\to b\ B \mid c
\end{aligned}
$$

Verify your answer by implementing the grammar in the Python code provided and running the computation.

### Exercise 2: LL(1) Table Construction

Given the grammar:

$$
\begin{aligned}
S &\to i\ E\ t\ S\ S' \mid a \\
S' &\to e\ S \mid \varepsilon \\
E &\to b
\end{aligned}
$$

(where $i$ = `if`, $t$ = `then`, $e$ = `else`, $a$ = assignment, $b$ = boolean expression)

1. Compute FIRST and FOLLOW sets.
2. Construct the LL(1) parsing table.
3. Identify any conflicts. How would you resolve the dangling else?

### Exercise 3: Left Recursion Elimination

Eliminate all left recursion from the following grammar:

$$
\begin{aligned}
S &\to A\ a \mid b \\
A &\to A\ c \mid S\ d \mid e
\end{aligned}
$$

Note that $A \to S\ d$ creates indirect left recursion. Show each step of the algorithm.

### Exercise 4: Recursive Descent Parser Extension

Extend the recursive descent parser provided in Section 2.2 to support:

1. Subtraction (`-`) with the same precedence as addition
2. Division (`/`) with the same precedence as multiplication
3. Unary negation (`-x`)
4. Integer literals and variable names

Write the modified grammar and the corresponding Python code.

### Exercise 5: Error Recovery

Implement panic-mode error recovery in the table-driven LL(1) parser. Your implementation should:

1. When $M[A, a]$ is empty, report the error with the position
2. Skip input tokens until finding one in $\text{FOLLOW}(A)$
3. Pop $A$ from the stack and continue parsing
4. Report all errors found (not just the first one)

Test with the input `"id + * id"` (missing operand between `+` and `*`).

### Exercise 6: Grammar Design Challenge

Design an LL(1) grammar for the following language features:

- Variable declarations: `let x = expr;`
- Assignment: `x = expr;`
- If-else statements: `if (expr) { stmts } else { stmts }`
- While loops: `while (expr) { stmts }`
- Print statements: `print(expr);`
- Arithmetic expressions with `+`, `-`, `*`, `/`

1. Write the grammar.
2. Verify it is LL(1) by computing FIRST and FOLLOW sets.
3. If not LL(1), apply left factoring and/or left recursion elimination.

---

[Previous: Context-Free Grammars](./04_Context_Free_Grammars.md) | [Next: Bottom-Up Parsing](./06_Bottom_Up_Parsing.md) | [Overview](./00_Overview.md)
