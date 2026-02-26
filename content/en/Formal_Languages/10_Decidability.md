# Lesson 10: Decidability

**Previous**: [Turing Machines](./09_Turing_Machines.md) | **Next**: [Reducibility](./11_Reducibility.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Distinguish decidable from undecidable languages
2. Prove the undecidability of the halting problem using diagonalization
3. Apply diagonalization arguments to new problems
4. Identify decidable problems for regular and context-free languages
5. Explain the relationship between decidability and recognizability

---

Decidability draws the sharpest line in all of computer science: the boundary between problems that algorithms can solve and problems that no algorithm can ever solve. The halting problem -- can we determine in general whether a program will finish running? -- turns out to be undecidable. This is not a limitation of current technology; it is a mathematical impossibility that applies to all computers past, present, and future.

## 1. Decidable Languages

A language $L$ is **decidable** if there exists a TM that halts on every input and correctly accepts/rejects membership. We call such a TM a **decider**.

**Key distinction:**
- Decidable = always halts (accepts or rejects)
- Recognizable = halts and accepts for members, but may loop for non-members

## 2. Decidable Problems for Regular Languages

**Theorem.** The following problems are decidable:

### 2.1 $A_{DFA}$: DFA Acceptance

$A_{DFA} = \{\langle B, w \rangle \mid B \text{ is a DFA that accepts } w\}$

*Decider:* Simulate $B$ on $w$. Since DFAs always halt after $|w|$ steps, this terminates.

### 2.2 $E_{DFA}$: DFA Emptiness

$E_{DFA} = \{\langle A \rangle \mid A \text{ is a DFA and } L(A) = \emptyset\}$

*Decider:* Check if any accept state is reachable from the start state (graph reachability).

### 2.3 $EQ_{DFA}$: DFA Equivalence

$EQ_{DFA} = \{\langle A, B \rangle \mid A, B \text{ are DFAs and } L(A) = L(B)\}$

*Decider:* Construct a DFA for the symmetric difference $L(A) \triangle L(B) = (L(A) \cap \overline{L(B)}) \cup (\overline{L(A)} \cap L(B))$ and check emptiness.

## 3. Decidable Problems for Context-Free Languages

### 3.1 $A_{CFG}$: CFG Membership

$A_{CFG} = \{\langle G, w \rangle \mid G \text{ is a CFG that generates } w\}$

*Decider:* Convert $G$ to CNF and run the CYK algorithm.

### 3.2 $E_{CFG}$: CFG Emptiness

$E_{CFG} = \{\langle G \rangle \mid G \text{ is a CFG and } L(G) = \emptyset\}$

*Decider:* Mark terminals as "generating." Iteratively mark variables whose rules have all symbols already marked. Accept if $S$ is never marked.

### 3.3 What about $EQ_{CFG}$?

$EQ_{CFG} = \{\langle G, H \rangle \mid L(G) = L(H)\}$ is **undecidable**!

This is a key difference between regular and context-free languages.

## 4. The Halting Problem

### 4.1 $A_{TM}$: TM Acceptance

$A_{TM} = \{\langle M, w \rangle \mid M \text{ is a TM that accepts } w\}$

**Theorem.** $A_{TM}$ is Turing-recognizable but not decidable.

*Recognizability:* A universal TM $U$ recognizes $A_{TM}$ — it simulates $M$ on $w$ and accepts if $M$ accepts. But if $M$ loops, $U$ loops too.

### 4.2 Undecidability Proof (Diagonalization)

**Theorem.** $A_{TM}$ is undecidable.

*Proof by contradiction.* Assume $H$ is a decider for $A_{TM}$:

$$H(\langle M, w \rangle) = \begin{cases} \text{accept} & \text{if } M \text{ accepts } w \\ \text{reject} & \text{if } M \text{ does not accept } w \end{cases}$$

Construct a new TM $D$:

> $D$ on input $\langle M \rangle$:
> 1. Run $H$ on $\langle M, \langle M \rangle \rangle$
> 2. If $H$ accepts, **reject**. If $H$ rejects, **accept**.

Now consider $D$ on input $\langle D \rangle$:
- If $D$ accepts $\langle D \rangle$, then $H$ accepted $\langle D, \langle D \rangle \rangle$, so $D$ does accept $\langle D \rangle$. But $D$'s construction says it rejects. **Contradiction.**
- If $D$ rejects $\langle D \rangle$, then $H$ rejected $\langle D, \langle D \rangle \rangle$, so $D$ does not accept $\langle D \rangle$. But $D$'s construction says it accepts. **Contradiction.**

Both cases lead to contradiction, so $H$ cannot exist. $\square$

### 4.3 The Diagonalization Argument

This proof uses **diagonalization**, the same technique Cantor used to prove the uncountability of the reals:

| | $\langle M_1 \rangle$ | $\langle M_2 \rangle$ | $\langle M_3 \rangle$ | $\cdots$ |
|---|---|---|---|---|
| $M_1$ | accept | reject | accept | $\cdots$ |
| $M_2$ | reject | reject | accept | $\cdots$ |
| $M_3$ | accept | accept | reject | $\cdots$ |
| $\vdots$ | | | | $\ddots$ |
| $D$ | **reject** | **accept** | **accept** | $\cdots$ |

$D$ flips the diagonal entries, so $D$ cannot appear in any row — it differs from every $M_i$ on input $\langle M_i \rangle$.

## 5. Consequences

### 5.1 The Complement of $A_{TM}$

$\overline{A_{TM}} = \{\langle M, w \rangle \mid M \text{ does not accept } w\}$

**Theorem.** $\overline{A_{TM}}$ is not Turing-recognizable.

*Proof.* If both $A_{TM}$ and $\overline{A_{TM}}$ were recognizable, we could build a decider for $A_{TM}$ by running both recognizers in parallel. Since $A_{TM}$ is undecidable, $\overline{A_{TM}}$ cannot be recognizable. $\square$

### 5.2 Recognizable vs. Co-Recognizable

**Theorem.** $L$ is decidable if and only if both $L$ and $\overline{L}$ are Turing-recognizable.

*Proof.*
- ($\Rightarrow$) If $L$ is decidable, a decider for $L$ recognizes $L$, and flipping its output gives a decider (hence recognizer) for $\overline{L}$.
- ($\Leftarrow$) Given recognizers $M_1$ for $L$ and $M_2$ for $\overline{L}$, run both in parallel. One must accept on every input. If $M_1$ accepts first, accept. If $M_2$ accepts first, reject. $\square$

## 6. More Undecidable Problems

### 6.1 $HALT_{TM}$

$HALT_{TM} = \{\langle M, w \rangle \mid M \text{ halts on } w\}$

**Undecidable.** Reducible from $A_{TM}$: if we could decide halting, we could decide acceptance by first checking if $M$ halts on $w$, then simulating to see if it accepts.

### 6.2 $E_{TM}$: TM Emptiness

$E_{TM} = \{\langle M \rangle \mid L(M) = \emptyset\}$

**Undecidable.** To decide $A_{TM}$, given $\langle M, w \rangle$, construct $M'$ that ignores its input and simulates $M$ on $w$. Then $L(M') = \emptyset$ iff $M$ doesn't accept $w$.

### 6.3 $EQ_{TM}$: TM Equivalence

$EQ_{TM} = \{\langle M_1, M_2 \rangle \mid L(M_1) = L(M_2)\}$

**Undecidable.** Reduce from $E_{TM}$: is $L(M) = L(M_\emptyset)$ where $M_\emptyset$ rejects everything?

## 7. The Landscape of Decidability

```
All languages (uncountably many)
├── Not Turing-recognizable (e.g., complement of A_TM)
├── Turing-recognizable but not decidable (e.g., A_TM)
└── Decidable
    ├── Context-free
    │   └── Regular
    └── Non-context-free decidable (e.g., {a^n b^n c^n})
```

## Practice Exercises

### Exercise 1: Decidability Proofs

Prove that the following languages are decidable:

1. $\{\langle M \rangle \mid M \text{ is a DFA that accepts some string of length } \leq 5\}$
2. $\{\langle G \rangle \mid G \text{ is a CFG and } L(G) \text{ is infinite}\}$
3. $\{\langle M \rangle \mid M \text{ is a DFA that accepts at least one palindrome}\}$

### Exercise 2: Undecidability

Show that $L = \{\langle M \rangle \mid M \text{ is a TM that accepts } \varepsilon\}$ is undecidable. (Hint: reduce from $A_{TM}$.)

### Exercise 3: Recognizability

Classify each language as decidable, recognizable (but not decidable), or not recognizable:

1. $\{\langle M, w \rangle \mid M \text{ halts on } w \text{ within 1000 steps}\}$
2. $\{\langle M \rangle \mid M \text{ accepts some string}\}$
3. $\{\langle M \rangle \mid L(M) = \Sigma^*\}$

---

**Previous**: [Turing Machines](./09_Turing_Machines.md) | **Next**: [Reducibility](./11_Reducibility.md)
