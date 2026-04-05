# Lesson 7: Pushdown Automata

**Previous**: [Context-Free Grammars](./06_Context_Free_Grammars.md) | **Next**: [Properties of Context-Free Languages](./08_Properties_of_Context_Free_Languages.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Define pushdown automata formally
2. Explain the role of the stack in extending finite automata
3. Convert between CFGs and PDAs
4. Distinguish between acceptance by final state and acceptance by empty stack
5. Compare deterministic PDAs with nondeterministic PDAs and identify their limitations

## 1. Motivation

Finite automata cannot recognize $\{0^n 1^n \mid n \geq 0\}$ because they have no memory beyond the current state. A **pushdown automaton (PDA)** extends a finite automaton with a **stack** — an unbounded LIFO data structure.

**Analogy:** Picture a cafeteria tray dispenser -- you can only add a tray on top (push) or remove the topmost tray (pop), and you cannot see or reach the trays below. A PDA's stack works exactly the same way: last in, first out. This simple mechanism is surprisingly powerful: it lets the machine "remember" things like how many opening parentheses it has seen and match them against closing ones, which is precisely what finite automata cannot do.

The stack provides exactly the right amount of power to recognize context-free languages.

## 2. Formal Definition

A **pushdown automaton** is a 6-tuple $P = (Q, \Sigma, \Gamma, \delta, q_0, F)$ where:

| Component | Description |
|-----------|-------------|
| $Q$ | A finite set of states |
| $\Sigma$ | A finite input alphabet |
| $\Gamma$ | A finite **stack alphabet** |
| $\delta: Q \times (\Sigma \cup \{\varepsilon\}) \times (\Gamma \cup \{\varepsilon\}) \to \mathcal{P}(Q \times (\Gamma \cup \{\varepsilon\}))$ | Transition function |
| $q_0 \in Q$ | Start state |
| $F \subseteq Q$ | Accept states |

A transition $\delta(q, a, X) \ni (p, Y)$ means: in state $q$, reading input $a$ (or $\varepsilon$), with $X$ on top of stack (or $\varepsilon$ for no stack read), move to state $p$ and replace $X$ with $Y$ (or push $Y$ / pop $X$).

### 2.1 Stack Operations

The transition function encodes stack operations:

| Transition | Stack Effect |
|------------|-------------|
| $\delta(q, a, \varepsilon) \ni (p, X)$ | **Push** $X$ (don't read stack) |
| $\delta(q, a, X) \ni (p, \varepsilon)$ | **Pop** $X$ |
| $\delta(q, a, X) \ni (p, Y)$ | **Replace** top $X$ with $Y$ |
| $\delta(q, a, X) \ni (p, YZ)$ | **Replace** $X$ with $YZ$ ($Y$ on top) |
| $\delta(q, a, \varepsilon) \ni (p, \varepsilon)$ | No stack change |

Note: PDAs are inherently **nondeterministic** — $\delta$ returns a set.

## 3. Configuration and Computation

A **configuration** (or instantaneous description) of a PDA is a triple $(q, w, \gamma)$ where:
- $q$ is the current state
- $w$ is the remaining input
- $\gamma$ is the stack contents (top on the left)

The **yields** relation $\vdash$ describes one computation step:
$$(q, aw, X\gamma) \vdash (p, w, Y\gamma) \quad \text{if } (p, Y) \in \delta(q, a, X)$$

We write $\vdash^*$ for the reflexive transitive closure.

## 4. Acceptance Modes

### 4.1 Acceptance by Final State

The PDA accepts $w$ if $(q_0, w, Z_0) \vdash^* (q, \varepsilon, \gamma)$ for some $q \in F$ and any $\gamma$.

$$L(P) = \{w \mid (q_0, w, Z_0) \vdash^* (q, \varepsilon, \gamma), q \in F\}$$

(Here $Z_0$ is an initial stack symbol, sometimes included as part of the definition.)

### 4.2 Acceptance by Empty Stack

The PDA accepts $w$ if $(q_0, w, Z_0) \vdash^* (q, \varepsilon, \varepsilon)$ for any state $q$.

$$N(P) = \{w \mid (q_0, w, Z_0) \vdash^* (q, \varepsilon, \varepsilon)\}$$

### 4.3 Equivalence of Acceptance Modes

**Theorem.** For every PDA $P_1$ accepting by final state, there exists a PDA $P_2$ accepting by empty stack with $N(P_2) = L(P_1)$, and vice versa.

The conversions use standard constructions:
- **Final state → Empty stack**: After reaching an accept state, pop everything
- **Empty stack → Final state**: Detect empty stack (using a bottom marker) and transition to an accept state

## 5. PDA Design Examples

### 5.1 $L = \{0^n 1^n \mid n \geq 0\}$

**Strategy:** Push a symbol for each 0, pop for each 1. Accept if stack empties after reading all input.

```
States: {q0, q1, q2}
Start: q0, Accept: {q2}
Stack alphabet: {Z, X}  (Z = bottom marker)

δ(q0, ε, ε) = {(q1, Z)}     -- push bottom marker
δ(q1, 0, ε) = {(q1, X)}     -- push X for each 0
δ(q1, 1, X) = {(q2, ε)}     -- start popping
δ(q2, 1, X) = {(q2, ε)}     -- keep popping
δ(q2, ε, Z) = {(q2, ε)}     -- pop marker, accept by empty stack
```

### 5.2 $L = \{ww^R \mid w \in \{a,b\}^*\}$ (Even-Length Palindromes)

**Strategy:** Push first half, then match second half against stack.

```
States: {q_push, q_pop, q_accept}
Start: q_push, Accept: {q_accept}

-- Push phase: nondeterministically guess the middle
δ(q_push, a, ε) = {(q_push, a)}
δ(q_push, b, ε) = {(q_push, b)}
δ(q_push, ε, ε) = {(q_pop, ε)}    -- guess: we're at the middle

-- Pop phase: match input with stack
δ(q_pop, a, a) = {(q_pop, ε)}
δ(q_pop, b, b) = {(q_pop, ε)}
δ(q_pop, ε, Z) = {(q_accept, ε)}  -- stack empty = matched
```

The nondeterminism is essential: the PDA guesses the midpoint.

### 5.3 $L = \{a^i b^j c^k \mid i = j \text{ or } j = k\}$

The PDA nondeterministically chooses which equality to verify:

- **Branch 1:** Push $a$'s, match with $b$'s, skip $c$'s
- **Branch 2:** Skip $a$'s, push $b$'s, match with $c$'s

## 6. Equivalence of PDA and CFG

**Theorem.** A language is context-free if and only if it is accepted by some PDA.

### 6.1 CFG → PDA

Given CFG $G = (V, \Sigma, R, S)$, construct a PDA that simulates leftmost derivations:

1. Push $S$ onto the stack
2. Repeat:
   - If top of stack is a variable $A$: nondeterministically choose a rule $A \to \alpha$ and replace $A$ with $\alpha$
   - If top of stack is a terminal $a$: read $a$ from input and pop

This PDA has just 3 states (start, loop, accept) and does all the work on the stack.

**Worked trace:** Let us convert the grammar $S \to aSb \mid \varepsilon$ and trace the PDA on input `aabb`. The PDA starts with $S$ on the stack and uses the rules above.

| Step | State | Remaining Input | Stack (top on left) | Action |
|------|-------|-----------------|---------------------|--------|
| 0 | start | `aabb` | (empty) | Push $S$, go to loop |
| 1 | loop | `aabb` | $S$ | Replace $S$ with $aSb$ (choose rule $S \to aSb$) |
| 2 | loop | `aabb` | $a\,S\,b$ | Top is terminal $a$; match with input $a$, pop |
| 3 | loop | `abb` | $S\,b$ | Replace $S$ with $aSb$ (choose rule $S \to aSb$) |
| 4 | loop | `abb` | $a\,S\,b\,b$ | Top is terminal $a$; match with input $a$, pop |
| 5 | loop | `bb` | $S\,b\,b$ | Replace $S$ with $\varepsilon$ (choose rule $S \to \varepsilon$), pop $S$ |
| 6 | loop | `bb` | $b\,b$ | Top is terminal $b$; match with input $b$, pop |
| 7 | loop | `b` | $b$ | Top is terminal $b$; match with input $b$, pop |
| 8 | loop | (empty) | (empty) | Input exhausted, stack empty -- accept! |

At each step, the PDA either expands a variable using a grammar rule or matches a terminal against the next input symbol. The nondeterminism is in choosing which grammar rule to apply (step 1 vs. step 5).

### 6.2 PDA → CFG

Given PDA $P$, construct a CFG whose variables correspond to triples $[q, A, p]$ representing "starting in state $q$ with $A$ on top of stack, reaching state $p$ with $A$ popped."

The construction is more complex but systematic:
- For each transition that pushes: create rules linking the push to eventual pops
- For each pair of states: create a variable representing the computation between them

## 7. Deterministic Pushdown Automata

A PDA is **deterministic (DPDA)** if for every configuration, at most one transition is possible:

For all $q \in Q$, $a \in \Sigma$, $X \in \Gamma$:
- $|\delta(q, a, X)| + |\delta(q, \varepsilon, X)| + |\delta(q, a, \varepsilon)| + |\delta(q, \varepsilon, \varepsilon)| \leq 1$

(At most one applicable transition for each state, input, and stack combination.)

### 7.1 DPDA Power

**Important:** DPDAs are **strictly weaker** than nondeterministic PDAs!

- DPDAs recognize exactly the **deterministic context-free languages (DCFL)**
- DCFL $\subsetneq$ CFL
- Example: $\{ww^R \mid w \in \{a,b\}^*\}$ is CFL but not DCFL
- Example: $\{wcw^R \mid w \in \{a,b\}^*\}$ is DCFL (the center marker $c$ eliminates the need to guess)

This contrasts with finite automata, where DFA = NFA in power.

### 7.2 Properties of DCFL

- Closed under complement (unlike general CFL!)
- **Not** closed under union, intersection, concatenation, or Kleene star
- Every DCFL has an unambiguous grammar
- DCFL are LR(k)-parsable — the basis of practical parser generators

## 8. PDA Variants

| Variant | Description | Power |
|---------|-------------|-------|
| Standard PDA | As defined above | CFL |
| Two-stack PDA | Two independent stacks | Equal to Turing machine! |
| PDA with queue | Replace stack with queue | Equal to Turing machine! |
| Counter automaton | Stack over $\{Z, X\}$ only | Strictly between DFA and PDA |

Adding a second stack or replacing the stack with a queue makes the model Turing-complete — a hint at the power of unrestricted memory.

## Practice Exercises

### Exercise 1: PDA Design

Design PDAs for:

1. $L = \{a^n b^{2n} \mid n \geq 0\}$
2. $L = \{w \in \{a,b\}^* \mid |w|_a = |w|_b\}$
3. $L = \{a^i b^j c^k \mid i + k = j\}$

### Exercise 2: CFG to PDA

Convert this grammar to a PDA using the construction from Section 6.1:

$$S \to aSb \mid \varepsilon$$

Trace the PDA's computation on input $aabb$.

### Exercise 3: Determinism

Which of these languages are deterministic CFL? Justify your answer.

1. $\{a^n b^n \mid n \geq 0\}$
2. $\{ww^R \mid w \in \{a,b\}^*\}$
3. $\{a^n b^n \mid n \geq 0\} \cup \{a^n b^{2n} \mid n \geq 0\}$

---

**Previous**: [Context-Free Grammars](./06_Context_Free_Grammars.md) | **Next**: [Properties of Context-Free Languages](./08_Properties_of_Context_Free_Languages.md)
