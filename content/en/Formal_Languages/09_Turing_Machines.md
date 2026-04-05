# Lesson 9: Turing Machines

**Previous**: [Properties of Context-Free Languages](./08_Properties_of_Context_Free_Languages.md) | **Next**: [Decidability](./10_Decidability.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Define Turing machines formally
2. Trace TM computation on example inputs
3. Explain TM variants and prove their equivalence
4. State the Church-Turing thesis and its implications
5. Distinguish between decidable and recognizable languages

## 1. Motivation

Finite automata (finite memory) and pushdown automata (stack memory) have limited power. A **Turing machine** has an **infinite tape** — unrestricted read/write memory — and can compute anything that is intuitively "computable."

Turing machines are the foundation of computability theory and define the boundary between what computers can and cannot do.

> **Analogy -- Infinite Scratch Paper.** A Turing machine is the theoretical equivalent of a person with unlimited scratch paper, a pencil, an eraser, and a simple rule book. The person reads one symbol, consults the rule book, writes a new symbol, and moves left or right. Despite this simplicity, Church and Turing proved that anything computable by any algorithm can be computed by this person -- making the Turing machine the universal yardstick for computability.

## 2. Formal Definition

A **Turing machine (TM)** is a 7-tuple $M = (Q, \Sigma, \Gamma, \delta, q_0, q_{accept}, q_{reject})$ where:

| Component | Description |
|-----------|-------------|
| $Q$ | A finite set of states |
| $\Sigma$ | The input alphabet (does not contain the blank symbol $\sqcup$) |
| $\Gamma$ | The tape alphabet, where $\sqcup \in \Gamma$ and $\Sigma \subset \Gamma$ |
| $\delta: Q \times \Gamma \to Q \times \Gamma \times \{L, R\}$ | Transition function |
| $q_0 \in Q$ | Start state |
| $q_{accept} \in Q$ | Accept state |
| $q_{reject} \in Q$ | Reject state ($q_{reject} \neq q_{accept}$) |

A transition $\delta(q, a) = (r, b, D)$ means: in state $q$, reading symbol $a$, write $b$, move head in direction $D \in \{L, R\}$, and enter state $r$.

The machine halts immediately upon entering $q_{accept}$ or $q_{reject}$.

## 3. Computation

### 3.1 Configuration

A **configuration** of a TM is a snapshot: $uqv$ where:
- $u \in \Gamma^*$: tape contents to the left of the head
- $q \in Q$: current state
- $v \in \Gamma^+$: tape contents from the head position rightward

The head is scanning the first symbol of $v$.

### 3.2 Computation Steps

Configuration $C_1$ **yields** $C_2$ (written $C_1 \vdash C_2$) if $C_2$ follows from $C_1$ by one transition.

Example: if $\delta(q, b) = (r, c, R)$, then $uaq\mathbf{b}v \vdash uacr\mathbf{v}$.

### 3.3 Accepting, Rejecting, Looping

On input $w$, a TM may:
- **Accept**: Enter $q_{accept}$ (halts)
- **Reject**: Enter $q_{reject}$ (halts)
- **Loop**: Run forever without halting

This three-way outcome (accept/reject/loop) is crucial — it's why we distinguish **decidable** from **recognizable**.

## 4. Languages and Decidability

### 4.1 Turing-Recognizable (Recursively Enumerable)

A language $L$ is **Turing-recognizable** (or **recursively enumerable**, RE) if there exists a TM $M$ such that:
- $w \in L \Rightarrow M$ accepts $w$
- $w \notin L \Rightarrow M$ rejects or loops on $w$

### 4.2 Turing-Decidable (Recursive)

A language $L$ is **Turing-decidable** (or **recursive**, or simply **decidable**) if there exists a TM $M$ that halts on every input and:
- $w \in L \Rightarrow M$ accepts $w$
- $w \notin L \Rightarrow M$ rejects $w$

Such a TM is called a **decider**.

### 4.3 Hierarchy

$$\text{Regular} \subsetneq \text{Context-Free} \subsetneq \text{Decidable} \subsetneq \text{Recognizable} \subsetneq \text{All Languages}$$

## 5. TM Design Examples

### 5.1 $L = \{0^n 1^n 2^n \mid n \geq 0\}$

**Strategy:** Repeatedly mark one 0, one 1, and one 2. Accept if all are marked simultaneously.

```
1. Scan right. If tape is blank, accept (n=0 case).
2. If current symbol is 0, mark it (write X). Move right to find unmarked 1.
3. Mark that 1 (write Y). Move right to find unmarked 2.
4. Mark that 2 (write Z). Move left all the way back to start.
5. Repeat from step 2.
6. If no unmarked 0s remain, scan right to verify no unmarked 1s or 2s remain.
7. If verified, accept. Otherwise, reject.
```

This TM runs in $O(n^2)$ time — it makes $n$ passes, each scanning $O(n)$ tape cells.

### 5.2 $L = \{w\#w \mid w \in \{0,1\}^*\}$ (String Matching)

**Strategy:** Zigzag between the two copies, comparing symbols one at a time.

```
1. Read and mark the first unmarked symbol before #.
2. Cross over # and find the corresponding position after #.
3. If symbols match, mark and return left. If not, reject.
4. Repeat until all symbols before # are marked.
5. Verify all symbols after # are also marked. Accept if so.
```

### 5.3 $L = \{w \mid w \text{ is a binary palindrome}\}$

**Strategy:** Compare first and last symbols, then shrink the string.

```
1. Read the leftmost symbol, remember it, mark it.
2. Scan right to find the rightmost unmarked symbol.
3. If they match, mark it and go back to step 1.
4. If they don't match, reject.
5. If all symbols are marked, accept.
```

## 6. TM Variants

### 6.1 Multitape Turing Machine

Has $k$ tapes, each with its own head. Transition: $\delta: Q \times \Gamma^k \to Q \times \Gamma^k \times \{L, R, S\}^k$.

**Theorem.** Every multitape TM has an equivalent single-tape TM.

*Proof idea:* Simulate $k$ tapes on one tape by interleaving contents with separators. Mark head positions with dotted symbols. Each step of the $k$-tape TM requires one scan of the single tape.

**Time blowup:** A $t(n)$-time $k$-tape TM can be simulated in $O(t(n)^2)$ time on a single tape.

### 6.2 Nondeterministic Turing Machine

Transition function returns a set: $\delta: Q \times \Gamma \to \mathcal{P}(Q \times \Gamma \times \{L, R\})$.

**Theorem.** Every nondeterministic TM has an equivalent deterministic TM.

*Proof idea:* Simulate the NTM's computation tree using breadth-first search on a 3-tape DTM.

**Key difference from finite automata:** NTMs don't add power (same as for NFAs), but the simulation has an **exponential** time overhead, which matters for complexity theory.

### 6.3 Other Variants

| Variant | Equivalent to standard TM? |
|---------|---------------------------|
| Two-way infinite tape | Yes |
| Multi-head TM | Yes |
| Multi-dimensional tape | Yes |
| Random access TM | Yes |
| Counter machines (2+ counters) | Yes |
| Queue automata | Yes |

All these variants recognize exactly the same class of languages — the Turing-recognizable languages.

## 7. The Church-Turing Thesis

**Thesis:** The class of functions computable by a Turing machine is exactly the class of functions that are "effectively computable" (by any algorithmic method).

This is a **thesis**, not a theorem — it cannot be proven because "effectively computable" is an informal notion. However:
- Every proposed formal model of computation (lambda calculus, recursive functions, Post systems, register machines, etc.) has been shown equivalent to Turing machines
- No one has found a counterexample
- It is universally accepted in computer science

**Implication:** If a function cannot be computed by a Turing machine, it cannot be computed by **any** algorithm on **any** computer.

## 8. Universal Turing Machine

A **universal Turing machine** $U$ takes as input the description $\langle M, w \rangle$ of a TM $M$ and a string $w$, and simulates $M$ on $w$.

- $U$ accepts $\langle M, w \rangle$ iff $M$ accepts $w$
- $U$ rejects $\langle M, w \rangle$ iff $M$ rejects $w$
- $U$ loops on $\langle M, w \rangle$ iff $M$ loops on $w$

The universal TM is the theoretical foundation of stored-program computers: a single machine that can simulate any other machine.

## 9. Encoding Turing Machines

To work with TMs as inputs, we need an encoding scheme $\langle M \rangle$ that represents a TM as a string over some fixed alphabet (e.g., $\{0, 1\}$).

Any reasonable encoding works (polynomial-time convertible). We write:
- $\langle M \rangle$: encoding of TM $M$
- $\langle M, w \rangle$: encoding of TM $M$ and input $w$
- $\langle G \rangle$: encoding of grammar $G$

## Practice Exercises

### Exercise 1: TM Design

Design Turing machines (describe the high-level algorithm) for:

1. $L = \{a^{2^n} \mid n \geq 0\}$ (strings of $a$'s whose length is a power of 2)
2. Addition: given $0^m 1 0^n$, produce $0^{m+n}$ on the tape
3. $L = \{w\#w\#w \mid w \in \{a,b\}^*\}$ (three copies)

### Exercise 2: Multitape Simulation

Describe how a 2-tape TM can recognize $\{0^n 1^n 2^n \mid n \geq 0\}$ in $O(n)$ time (versus $O(n^2)$ for a single tape). What is the time overhead of the single-tape simulation?

### Exercise 3: Church-Turing Thesis

Consider a programming language with variables, conditionals, loops, and arrays. Argue informally that any function computable in this language is also computable by a Turing machine. (Hint: how would you simulate each language construct?)

---

**Previous**: [Properties of Context-Free Languages](./08_Properties_of_Context_Free_Languages.md) | **Next**: [Decidability](./10_Decidability.md)
