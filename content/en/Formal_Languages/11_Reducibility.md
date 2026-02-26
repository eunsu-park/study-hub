# Lesson 11: Reducibility

**Previous**: [Decidability](./10_Decidability.md) | **Next**: [Computational Complexity](./12_Computational_Complexity.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Define and apply mapping reductions between languages
2. Use reductions to prove undecidability of new problems
3. State and apply Rice's theorem
4. Describe the Post Correspondence Problem and its significance
5. Distinguish Turing reductions from mapping reductions

## 1. The Concept of Reduction

A **reduction** transforms one problem into another. If problem $A$ reduces to problem $B$, then:
- A solution to $B$ gives a solution to $A$
- If $A$ is hard (undecidable), then $B$ must also be hard

Reductions are the primary tool for proving undecidability: reduce a known undecidable problem to the target problem.

## 2. Mapping Reductions

### 2.1 Definition

A function $f: \Sigma^* \to \Sigma^*$ is **computable** if there exists a TM that, on input $w$, halts with $f(w)$ on its tape.

Language $A$ is **mapping reducible** to language $B$, written $A \leq_m B$, if there exists a computable function $f$ such that:

$$w \in A \iff f(w) \in B$$

The function $f$ is called the **reduction**.

### 2.2 Properties

**Theorem.** If $A \leq_m B$ and $B$ is decidable, then $A$ is decidable.

*Proof.* Given a decider $D_B$ for $B$, construct a decider for $A$: on input $w$, compute $f(w)$ and run $D_B$ on $f(w)$. $\square$

**Contrapositive.** If $A$ is undecidable and $A \leq_m B$, then $B$ is undecidable.

Similarly for recognizability:
- If $A \leq_m B$ and $B$ is recognizable, then $A$ is recognizable
- If $A$ is not recognizable and $A \leq_m B$, then $B$ is not recognizable

## 3. Reduction Examples

### 3.1 $HALT_{TM}$ is Undecidable

$HALT_{TM} = \{\langle M, w \rangle \mid M \text{ halts on input } w\}$

**Reduction:** $A_{TM} \leq_m HALT_{TM}$

Define $f(\langle M, w \rangle) = \langle M', w \rangle$ where $M'$ is:

> $M'$ on input $x$:
> 1. Simulate $M$ on $x$
> 2. If $M$ accepts, accept
> 3. If $M$ rejects, enter an infinite loop

Now:
- $\langle M, w \rangle \in A_{TM}$ (M accepts w) $\iff$ $M'$ halts on $w$ (accepting) $\iff$ $\langle M', w \rangle \in HALT_{TM}$
- $\langle M, w \rangle \notin A_{TM}$ $\iff$ $M'$ either loops (if $M$ loops) or loops (if $M$ rejects) $\iff$ $\langle M', w \rangle \notin HALT_{TM}$ $\square$

### 3.2 $E_{TM}$ is Undecidable

$E_{TM} = \{\langle M \rangle \mid L(M) = \emptyset\}$

**Reduction:** $A_{TM} \leq_m \overline{E_{TM}}$

Define $f(\langle M, w \rangle) = \langle M' \rangle$ where $M'$ is:

> $M'$ on input $x$:
> 1. Ignore $x$
> 2. Simulate $M$ on $w$
> 3. If $M$ accepts $w$, accept

Now:
- $\langle M, w \rangle \in A_{TM}$ $\iff$ $L(M') = \Sigma^*$ $\iff$ $\langle M' \rangle \notin E_{TM}$ $\iff$ $\langle M' \rangle \in \overline{E_{TM}}$

So $A_{TM} \leq_m \overline{E_{TM}}$, making $\overline{E_{TM}}$ undecidable, and therefore $E_{TM}$ is undecidable. $\square$

### 3.3 $EQ_{TM}$ is Undecidable

$EQ_{TM} = \{\langle M_1, M_2 \rangle \mid L(M_1) = L(M_2)\}$

**Reduction:** $E_{TM} \leq_m EQ_{TM}$

Define $f(\langle M \rangle) = \langle M, M_\emptyset \rangle$ where $M_\emptyset$ rejects all inputs.

$\langle M \rangle \in E_{TM} \iff L(M) = \emptyset = L(M_\emptyset) \iff \langle M, M_\emptyset \rangle \in EQ_{TM}$ $\square$

### 3.4 $REGULAR_{TM}$ is Undecidable

$REGULAR_{TM} = \{\langle M \rangle \mid L(M) \text{ is regular}\}$

**Reduction:** $A_{TM} \leq_m REGULAR_{TM}$ (actually to its complement, or use Rice's theorem — see Section 4).

## 4. Rice's Theorem

Rice's theorem is a powerful meta-result that settles many undecidability questions at once.

### 4.1 Property of Languages

A **property** of Turing-recognizable languages is a set $P$ of Turing-recognizable languages. A TM $M$ "has property $P$" if $L(M) \in P$.

A property is **nontrivial** if:
- Some TM has the property (i.e., $P \neq \emptyset$)
- Some TM does not have the property (i.e., $P$ is not the set of all RE languages)

### 4.2 Statement

**Theorem (Rice's Theorem).** If $P$ is any nontrivial property of Turing-recognizable languages, then the language

$$L_P = \{\langle M \rangle \mid L(M) \in P\}$$

is undecidable.

### 4.3 Proof

Assume WLOG that $\emptyset \notin P$ (otherwise consider $\overline{P}$).

Since $P$ is nontrivial, there exists a TM $T$ with $L(T) \in P$.

Reduce $A_{TM}$ to $L_P$: given $\langle M, w \rangle$, construct $M'$:

> $M'$ on input $x$:
> 1. Simulate $M$ on $w$
> 2. If $M$ accepts $w$, simulate $T$ on $x$ and output its answer
> 3. If $M$ rejects or loops on $w$, loop forever

Now:
- If $M$ accepts $w$: $L(M') = L(T) \in P$, so $\langle M' \rangle \in L_P$
- If $M$ doesn't accept $w$: $L(M') = \emptyset \notin P$, so $\langle M' \rangle \notin L_P$

Therefore $\langle M, w \rangle \in A_{TM} \iff \langle M' \rangle \in L_P$, giving $A_{TM} \leq_m L_P$. $\square$

**Plain-English summary:** Rice's theorem says that **you cannot write a program that reliably answers any "interesting" question about what another program computes**. Here "interesting" means the question is about the program's input-output behavior (its language), not about its source code, and the answer is not trivially always-yes or always-no. For example, no algorithm can look at arbitrary source code and reliably determine whether it computes the empty function, whether it ever outputs `"hello"`, or whether it recognizes a regular language. The moment you care about **what** a program does rather than **how it looks**, decidability is lost.

### 4.4 Applications of Rice's Theorem

These are all undecidable (nontrivial semantic properties):

| Language | Property |
|----------|----------|
| $\{\langle M \rangle \mid L(M) = \emptyset\}$ | Language is empty |
| $\{\langle M \rangle \mid L(M) = \Sigma^*\}$ | Language is everything |
| $\{\langle M \rangle \mid L(M) \text{ is regular}\}$ | Language is regular |
| $\{\langle M \rangle \mid L(M) \text{ is context-free}\}$ | Language is CF |
| $\{\langle M \rangle \mid L(M) \text{ is finite}\}$ | Language is finite |
| $\{\langle M \rangle \mid |L(M)| = 42\}$ | Language has exactly 42 strings |
| $\{\langle M \rangle \mid L(M) \text{ contains } \varepsilon\}$ | Language contains ε |

**What Rice's theorem does NOT cover:**
- Properties of TMs (not languages): e.g., "does $M$ have fewer than 100 states?" — this is decidable
- Trivial properties: "does $M$ accept some RE language?" — always yes, trivially decidable

## 5. The Post Correspondence Problem

### 5.1 Definition

An instance of the **Post Correspondence Problem (PCP)** is a collection of **dominos**: pairs $(t_1, b_1), (t_2, b_2), \ldots, (t_k, b_k)$ where $t_i, b_i \in \Sigma^+$.

A **match** is a sequence of indices $i_1, i_2, \ldots, i_n$ (with repetition allowed) such that:

$$t_{i_1} t_{i_2} \cdots t_{i_n} = b_{i_1} b_{i_2} \cdots b_{i_n}$$

The **decision problem**: given an instance, does a match exist?

### 5.2 Example

Dominos: $(a, ab)$, $(b, a)$, $(ab, b)$

Match: indices 1, 2, 1, 3:
- Top: $a \cdot b \cdot a \cdot ab = abaab$
- Bottom: $ab \cdot a \cdot ab \cdot b = abaab$ ✓

### 5.3 Undecidability

**Theorem.** PCP is undecidable.

*Proof idea:* Reduce from $A_{TM}$. Encode a TM computation as a PCP instance where a match exists iff the TM accepts. The dominos encode the transition function and the initial/accepting configurations.

### 5.4 Applications

PCP undecidability is used to prove undecidability of:
- Ambiguity of CFGs
- Equivalence of CFGs
- Whether the intersection of two CFLs is empty
- Various problems in formal verification

## 6. Turing Reductions

### 6.1 Definition

Language $A$ is **Turing reducible** to $B$, written $A \leq_T B$, if there exists an oracle TM $M^B$ that decides $A$ using $B$ as an oracle.

An **oracle TM** $M^B$ can, in one step, query "is $w \in B$?" and get the correct answer.

### 6.2 Comparison

| | Mapping Reduction ($\leq_m$) | Turing Reduction ($\leq_T$) |
|---|---|---|
| Mechanism | Computable function | Oracle queries |
| Queries to $B$ | Exactly one, at the end | Any number, anywhere |
| Strength | Stronger (more restrictive) | Weaker (more general) |
| If $A \leq B$ | $A \leq_m B \implies A \leq_T B$ | Not vice versa |

### 6.3 Example

$A_{TM} \leq_T HALT_{TM}$: With a halting oracle, we can decide acceptance — first ask if $M$ halts on $w$, then simulate if it does.

$\overline{A_{TM}} \leq_T A_{TM}$: With an $A_{TM}$ oracle, just negate the answer.

Note: $\overline{A_{TM}} \leq_m A_{TM}$ does NOT hold (mapping reduction preserves recognizability).

## Practice Exercises

### Exercise 1: Mapping Reductions

Construct mapping reductions to prove:

1. $A_{TM} \leq_m HALT_{TM}$ (give the explicit construction of $M'$)
2. $A_{TM} \leq_m \{\langle M \rangle \mid L(M) \text{ is infinite}\}$

### Exercise 2: Rice's Theorem

For each language, determine if Rice's theorem applies and conclude decidability:

1. $\{\langle M \rangle \mid M \text{ has at most 5 states}\}$
2. $\{\langle M \rangle \mid L(M)$ contains only even-length strings$\}$
3. $\{\langle M \rangle \mid M \text{ halts on all inputs within 100 steps}\}$

### Exercise 3: PCP

Find a match for this PCP instance (or argue that no match exists):

Dominos: $(ab, a)$, $(b, ab)$, $(a, b)$

---

**Previous**: [Decidability](./10_Decidability.md) | **Next**: [Computational Complexity](./12_Computational_Complexity.md)
