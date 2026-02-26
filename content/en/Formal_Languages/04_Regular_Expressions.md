# Lesson 4: Regular Expressions

**Previous**: [Nondeterministic Finite Automata](./03_Nondeterministic_Finite_Automata.md) | **Next**: [Properties of Regular Languages](./05_Properties_of_Regular_Languages.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Define regular expressions formally with their syntax and semantics
2. Convert between regular expressions and finite automata
3. Apply algebraic laws for regular expression equivalence
4. Distinguish between theoretical regular expressions and practical regex engines

---

If you have ever typed `grep 'error.*timeout' server.log` in a terminal or used `*.txt` to match files in a shell, you have already used the practical descendants of regular expressions. The theoretical version studied here strips away convenience features and asks a precise question: **what can be described with just three operations -- choice, concatenation, and repetition?** The answer turns out to be exactly the regular languages, the same class recognized by finite automata.

Regular expressions are the algebraic counterpart to finite automata. While automata describe languages procedurally (as machines), regular expressions describe them declaratively (as patterns). Every programmer encounters regex in tools like `grep`, text editors, and input validators -- but the theoretical foundation explains exactly what patterns they can and cannot express.

## 1. Formal Definition

A **regular expression** over alphabet $\Sigma$ is defined recursively:

**Base cases:**
- $\emptyset$ is a regex denoting the empty language $\emptyset$
- $\varepsilon$ is a regex denoting $\{\varepsilon\}$
- For each $a \in \Sigma$, the symbol $a$ is a regex denoting $\{a\}$

**Recursive cases:** If $R$ and $S$ are regexes, then:
- $(R \cup S)$ (union) denotes $L(R) \cup L(S)$
- $(RS)$ (concatenation) denotes $L(R) \cdot L(S)$
- $(R^*)$ (Kleene star) denotes $L(R)^*$

**Intuition for Kleene star:** Think of `*` as meaning "repeat zero or more times." It works just like the `*` wildcard in shell globs: `*.txt` matches any filename (including the empty prefix) followed by `.txt`. Similarly, $a^*$ matches the empty string, `a`, `aa`, `aaa`, and so on -- any number of repetitions, including none.

**Precedence** (highest to lowest): star > concatenation > union.

So $ab^* \cup c$ means $(a(b^*)) \cup c$, not $a(b^*\cup c)$ or $(ab)^* \cup c$.

## 2. Examples

Over $\Sigma = \{0, 1\}$:

| Regex | Language |
|-------|---------|
| $0^*$ | $\{\varepsilon, 0, 00, 000, \ldots\}$ |
| $(0 \cup 1)^*$ | $\Sigma^*$ (all strings) |
| $0(0 \cup 1)^*$ | Strings starting with 0 |
| $(0 \cup 1)^* 1$ | Strings ending with 1 |
| $(0 \cup 1)^* 01 (0 \cup 1)^*$ | Strings containing 01 as a substring |
| $1^* (01^+)^*$ | Strings where every 0 is followed by at least one 1 |
| $(01 \cup 10)^*$ | $\{\varepsilon, 01, 10, 0101, 0110, 1001, 1010, \ldots\}$ |

## 3. Equivalence with Finite Automata

**Theorem (Kleene's Theorem).** A language is regular if and only if it can be described by a regular expression.

This requires two directions:
1. Regex → NFA (constructive, via Thompson's construction)
2. DFA → Regex (constructive, via state elimination)

### 3.1 Regex to NFA: Thompson's Construction

For each regex construct, build a small NFA fragment with one start state and one accept state:

**Base cases:**

- $\varepsilon$: `→(start)--ε-->(accept)`
- $a$: `→(start)--a-->(accept)`

**Union** $R \cup S$:
- New start state with $\varepsilon$-transitions to start states of $N_R$ and $N_S$
- Accept states of $N_R$ and $N_S$ have $\varepsilon$-transitions to a new accept state

**Concatenation** $RS$:
- Connect accept state of $N_R$ via $\varepsilon$ to start state of $N_S$
- Start of $N_R$ is the new start; accept of $N_S$ is the new accept

**Star** $R^*$:
- New start state (also accepting) with $\varepsilon$ to start of $N_R$
- Accept state of $N_R$ has $\varepsilon$ back to start of $N_R$
- Accept state of $N_R$ also has $\varepsilon$ to new accept state

Each step adds at most 2 states, so the NFA has $O(n)$ states for a regex of length $n$.

### 3.2 DFA to Regex: State Elimination

Given a DFA (or NFA) with $n$ states, we can eliminate states one by one, labeling transitions with regular expressions:

**Algorithm:**
1. Add a new start state $q_s$ with $\varepsilon$-transition to the original start
2. Add a new accept state $q_a$ with $\varepsilon$-transitions from all original accept states
3. Repeat until only $q_s$ and $q_a$ remain:
   - Pick a state $q$ to eliminate (not $q_s$ or $q_a$)
   - For each pair $(p, r)$ with transitions $p \to q$ and $q \to r$:
     - If $q$ has a self-loop labeled $S$, the new label from $p$ to $r$ is: $R_1 S^* R_2 \cup R_3$ where $R_1$ is the label $p \to q$, $R_2$ is the label $q \to r$, and $R_3$ is any existing label $p \to r$
4. The label on the single remaining transition $q_s \to q_a$ is the regex

### 3.3 Example: State Elimination

Consider the DFA for "even number of 1s" with states $q_e$ (start, accept) and $q_o$:

After adding $q_s$ and $q_a$:
- $q_s \xrightarrow{\varepsilon} q_e$
- $q_e \xrightarrow{0} q_e$, $q_e \xrightarrow{1} q_o$
- $q_o \xrightarrow{0} q_o$, $q_o \xrightarrow{1} q_e$
- $q_e \xrightarrow{\varepsilon} q_a$

Eliminate $q_o$: $q_o$ has self-loop $0$ and transitions $q_e \xrightarrow{1} q_o \xrightarrow{1} q_e$.

New transition $q_e \to q_e$: $1 \cdot 0^* \cdot 1$ (go to $q_o$, loop on 0s, come back).

Combined with existing self-loop: $q_e \to q_e$ is now $0 \cup 1 0^* 1$.

Eliminate $q_e$: Result is $(0 \cup 1 0^* 1)^*$.

So the regex is $(0 \cup 10^*1)^*$ — strings where 1s come in pairs (with any 0s between them).

## 4. Algebraic Laws

Regular expressions satisfy these algebraic identities:

### 4.1 Union Laws
- $R \cup S = S \cup R$ (commutativity)
- $R \cup (S \cup T) = (R \cup S) \cup T$ (associativity)
- $R \cup R = R$ (idempotence)
- $R \cup \emptyset = R$ (identity)

### 4.2 Concatenation Laws
- $R(ST) = (RS)T$ (associativity)
- $R\varepsilon = \varepsilon R = R$ (identity)
- $R\emptyset = \emptyset R = \emptyset$ (annihilation)

### 4.3 Distribution
- $R(S \cup T) = RS \cup RT$ (left distribution)
- $(R \cup S)T = RT \cup ST$ (right distribution)

### 4.4 Star Laws
- $\emptyset^* = \varepsilon$
- $\varepsilon^* = \varepsilon$
- $R^* = R^* R^* = (R^*)^*$
- $R^* = \varepsilon \cup R R^*$
- $R^* = \varepsilon \cup R^* R$
- $(R \cup S)^* = (R^* S^*)^*$

### 4.5 Arden's Lemma

If $X = AX \cup B$ where $\varepsilon \notin L(A)$, then $X = A^*B$.

This lemma is useful for solving systems of equations over regular expressions (e.g., converting DFAs to regexes by solving state equations).

## 5. Practical vs. Theoretical Regex

**Theoretical** regular expressions use only: $\cup$, concatenation, $*$, $\varepsilon$, $\emptyset$, and alphabet symbols.

**Practical** regex engines (PCRE, Python `re`, etc.) add:

| Feature | Theory | Practice |
|---------|--------|----------|
| Character classes | — | `[a-z]`, `\d`, `\w` |
| Quantifiers | $R^*$ only | `R+`, `R?`, `R{n,m}` |
| Anchors | — | `^`, `$`, `\b` |
| Backreferences | — | `\1`, `(?: ...)` |
| Lookahead/behind | — | `(?=...)`, `(?<=...)` |

**Important:** Backreferences go **beyond** regular languages! The pattern `(a+)\1` matches $\{a^n a^n \mid n \geq 1\}$, which is not regular. Practical regex engines are more powerful than theoretical regular expressions but lose the guarantee of linear-time matching.

## 6. Regular Expression Complexity

| Operation | Time Complexity |
|-----------|----------------|
| Thompson's construction (regex → NFA) | $O(n)$ states/transitions |
| Subset construction (NFA → DFA) | $O(2^n)$ worst case |
| DFA simulation | $O(m)$ for input length $m$ |
| NFA simulation (on-the-fly subset) | $O(nm)$ for regex size $n$, input $m$ |

The NFA simulation approach avoids the exponential DFA construction while still providing polynomial-time matching — unlike backtracking engines that can take exponential time.

## Practice Exercises

### Exercise 1: Regex Design

Write regular expressions for these languages over $\Sigma = \{0, 1\}$:

1. Strings of length at least 2 that begin and end with the same symbol
2. Strings that do not contain the substring "11"
3. Strings where every pair of adjacent 0s appears before any 1

### Exercise 2: Regex to NFA

Apply Thompson's construction to build an NFA for $(ab \cup a)^*b$. Show the intermediate NFA fragments for each step.

### Exercise 3: State Elimination

Convert this DFA to a regular expression using state elimination:

```
States: {q0, q1}, Start: q0, Accept: {q1}
δ(q0, a) = q0    δ(q0, b) = q1
δ(q1, a) = q0    δ(q1, b) = q1
```

### Exercise 4: Algebraic Simplification

Use the algebraic laws to simplify:

1. $(a^* b^*)^* $
2. $\emptyset^* \cup a$
3. $(a \cup b)^* a (a \cup b)^*$

---

**Previous**: [Nondeterministic Finite Automata](./03_Nondeterministic_Finite_Automata.md) | **Next**: [Properties of Regular Languages](./05_Properties_of_Regular_Languages.md)
