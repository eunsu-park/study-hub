# Lesson 2: Deterministic Finite Automata (DFA)

**Previous**: [Introduction to Formal Languages](./01_Introduction_to_Formal_Languages.md) | **Next**: [Nondeterministic Finite Automata](./03_Nondeterministic_Finite_Automata.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Define a DFA formally as a 5-tuple
2. Trace computation on input strings using transition functions
3. Design DFAs for given languages
4. Prove correctness of DFA constructions
5. Apply the extended transition function

---

A DFA is the simplest computational model in the Chomsky hierarchy, yet it powers some of the most widely used tools in computing -- from lexical analyzers in compilers to pattern matching in text editors. Understanding DFAs builds the foundation for every other automaton we will study.

## 1. Informal Description

A **Deterministic Finite Automaton (DFA)** is the simplest computational model. It reads an input string one symbol at a time, transitions between a finite number of states, and either accepts or rejects the input.

Key properties:
- **Deterministic**: exactly one transition for each state-symbol pair
- **Finite**: bounded memory (finite number of states)
- No auxiliary storage beyond the current state

> **Analogy -- The Turnstile.** A DFA is like a subway turnstile. It has a fixed set of states (locked, unlocked), reads one input at a time (coin, push), and transitions deterministically -- the same input in the same state always produces the same result. There is no guessing, no backtracking, just a simple mechanism that either accepts your token or stays locked.

## 2. Formal Definition

A DFA is a 5-tuple $M = (Q, \Sigma, \delta, q_0, F)$ where:

| Component | Description |
|-----------|-------------|
| $Q$ | A finite set of **states** |
| $\Sigma$ | A finite **input alphabet** |
| $\delta: Q \times \Sigma \to Q$ | The **transition function** |
| $q_0 \in Q$ | The **start state** |
| $F \subseteq Q$ | The set of **accept (final) states** |

The transition function $\delta$ is **total**: it must be defined for every state-symbol pair.

## 3. State Diagrams

DFAs are commonly represented as directed graphs:
- Nodes represent states
- Edges labeled with symbols represent transitions
- The start state has an incoming arrow from nowhere
- Accept states are drawn with double circles

**Example:** DFA accepting strings over $\{0, 1\}$ with an even number of 1s.

```
States: Q = {q_even, q_odd}
Alphabet: Σ = {0, 1}
Start: q_even
Accept: F = {q_even}

Transitions:
  δ(q_even, 0) = q_even
  δ(q_even, 1) = q_odd
  δ(q_odd, 0) = q_odd
  δ(q_odd, 1) = q_even
```

## 4. Extended Transition Function

The transition function $\delta$ processes one symbol. We extend it to process strings:

**Definition.** The **extended transition function** $\hat{\delta}: Q \times \Sigma^* \to Q$ is defined recursively:

$$\hat{\delta}(q, \varepsilon) = q$$
$$\hat{\delta}(q, wa) = \delta(\hat{\delta}(q, w), a) \quad \text{for } w \in \Sigma^*, a \in \Sigma$$

Intuitively, $\hat{\delta}(q, w)$ is the state reached after processing string $w$ starting from state $q$.

## 5. Language of a DFA

The **language recognized** (or accepted) by a DFA $M$ is:

$$L(M) = \{w \in \Sigma^* \mid \hat{\delta}(q_0, w) \in F\}$$

A language $L$ is **regular** if there exists a DFA $M$ such that $L = L(M)$.

## 6. DFA Design Examples

### 6.1 Strings Ending in "01"

**Language:** $L = \{w \in \{0,1\}^* \mid w \text{ ends with } 01\}$

**Approach:** Track the last two symbols read.

```
States: {q_start, q_0, q_01}
Start: q_start
Accept: {q_01}

δ(q_start, 0) = q_0      δ(q_start, 1) = q_start
δ(q_0, 0) = q_0           δ(q_0, 1) = q_01
δ(q_01, 0) = q_0          δ(q_01, 1) = q_start
```

### 6.2 Strings with "ab" as a Substring

**Language:** $L = \{w \in \{a,b\}^* \mid ab \text{ is a substring of } w\}$

```
States: {q0, q_a, q_ab}
Start: q0
Accept: {q_ab}

δ(q0, a) = q_a      δ(q0, b) = q0
δ(q_a, a) = q_a     δ(q_a, b) = q_ab
δ(q_ab, a) = q_ab   δ(q_ab, b) = q_ab
```

State `q_ab` is a **trap state** (or **absorbing state**) — once entered, the DFA stays there regardless of input.

### 6.3 Binary Numbers Divisible by 3

**Language:** $L = \{w \in \{0,1\}^* \mid w \text{ represents a number divisible by 3}\}$

**Key insight:** If the number so far has value $n$ (mod 3), reading bit $b$ gives value $2n + b$ (mod 3).

```
States: {r0, r1, r2}  (remainder mod 3)
Start: r0
Accept: {r0}

δ(r0, 0) = r0   δ(r0, 1) = r1
δ(r1, 0) = r2   δ(r1, 1) = r0
δ(r2, 0) = r1   δ(r2, 1) = r2
```

## 7. DFA Product Construction

Given DFAs $M_1$ and $M_2$, we can construct DFAs for $L(M_1) \cap L(M_2)$ and $L(M_1) \cup L(M_2)$.

**Product construction** for $M_1 = (Q_1, \Sigma, \delta_1, q_{01}, F_1)$ and $M_2 = (Q_2, \Sigma, \delta_2, q_{02}, F_2)$:

Define $M = (Q_1 \times Q_2, \Sigma, \delta, (q_{01}, q_{02}), F)$ where:
- $\delta((p, q), a) = (\delta_1(p, a), \delta_2(q, a))$
- For **intersection**: $F = F_1 \times F_2$
- For **union**: $F = (F_1 \times Q_2) \cup (Q_1 \times F_2)$

**Theorem.** If $L_1$ and $L_2$ are regular, then $L_1 \cap L_2$ and $L_1 \cup L_2$ are regular.

## 8. Complement

Given a DFA $M = (Q, \Sigma, \delta, q_0, F)$ for $L$, the DFA for $\overline{L}$ is:

$$M' = (Q, \Sigma, \delta, q_0, Q \setminus F)$$

Simply swap accept and non-accept states!

**Theorem.** If $L$ is regular, then $\overline{L}$ is regular.

Note: This construction requires a **complete** DFA (total transition function). If starting from an NFA, first convert to a DFA.

## 9. Proving DFA Correctness

To prove a DFA $M$ correctly recognizes a language $L$, show $L(M) = L$:

**Method — Loop Invariant:** Define a claim about what each state represents, then prove by induction on the input length.

**Example:** Proving the "even number of 1s" DFA is correct.

**Claim:** After reading $w$, the DFA is in state $q_{even}$ if and only if $w$ contains an even number of 1s.

*Proof by induction on $|w|$:*

**Base:** $|w| = 0$, i.e., $w = \varepsilon$. The DFA is in $q_{even}$ (start state), and $\varepsilon$ has zero 1s (even). ✓

**Inductive step:** Assume the claim holds for string $w$. Consider $wa$ where $a \in \{0, 1\}$.

- Case $a = 0$: The parity doesn't change, and the DFA stays in the same state. ✓
- Case $a = 1$: The parity flips, and the DFA transitions between $q_{even}$ and $q_{odd}$. ✓

Since $F = \{q_{even}\}$, the DFA accepts $w$ iff $w$ has an even number of 1s. $\square$

## 10. Limitations of DFAs

DFAs have **finite memory** (only the current state). This means they cannot:

- Count unboundedly (e.g., cannot match $n$ a's with $n$ b's)
- Remember the full input
- Solve problems requiring a stack or other auxiliary storage

For example, no DFA can recognize $L = \{a^n b^n \mid n \geq 0\}$. We will prove this formally using the pumping lemma in Lesson 5.

## Practice Exercises

### Exercise 1: DFA Construction

Design DFAs (specify all 5 components) for each language over $\Sigma = \{0, 1\}$:

1. $L = \{w \mid w \text{ contains the substring } 110\}$
2. $L = \{w \mid |w| \text{ is divisible by 3}\}$
3. $L = \{w \mid w \text{ does not contain two consecutive 1s}\}$

### Exercise 2: Product Construction

Given:
- $M_1$ accepts strings with an even number of 0s
- $M_2$ accepts strings with an odd number of 1s

Construct the product DFA for $L(M_1) \cap L(M_2)$. How many states does it have?

### Exercise 3: Correctness Proof

Prove by induction that the "divisible by 3" DFA from Section 6.3 correctly recognizes binary numbers divisible by 3. (Hint: prove that state $r_i$ corresponds to the input having value $\equiv i \pmod{3}$.)

---

**Previous**: [Introduction to Formal Languages](./01_Introduction_to_Formal_Languages.md) | **Next**: [Nondeterministic Finite Automata](./03_Nondeterministic_Finite_Automata.md)
