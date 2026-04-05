# Lesson 12: Computational Complexity

**Previous**: [Reducibility](./11_Reducibility.md) | **Next**: [The Chomsky Hierarchy](./13_The_Chomsky_Hierarchy.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Define time and space complexity classes
2. Characterize the classes P, NP, coNP, and PSPACE
3. State the Cook-Levin theorem and explain its significance
4. Perform polynomial-time reductions between NP-complete problems
5. Articulate the P vs NP question and its implications

---

Decidability tells us whether a problem can be solved at all; complexity theory asks how *efficiently* it can be solved. The distinction between polynomial-time and exponential-time algorithms is the difference between a program that finishes in seconds and one that would outlast the universe. The P vs NP question -- whether every problem whose solution can be quickly verified can also be quickly solved -- is the most important open question in computer science, with a one-million-dollar prize awaiting its resolution.

## 1. Measuring Complexity

### 1.1 Time Complexity

The **time complexity** of a decider $M$ is the function $t: \mathbb{N} \to \mathbb{N}$ where $t(n)$ is the maximum number of steps $M$ takes on any input of length $n$.

We say $M$ runs in time $O(f(n))$ if $t(n) \leq c \cdot f(n)$ for some constant $c$ and all sufficiently large $n$.

### 1.2 Space Complexity

The **space complexity** of a decider $M$ is $s(n)$: the maximum number of tape cells $M$ uses on any input of length $n$.

### 1.3 Complexity Classes

A **complexity class** is a set of languages defined by resource bounds:

$$\text{TIME}(f(n)) = \{L \mid L \text{ is decided by a TM in } O(f(n)) \text{ time}\}$$
$$\text{NTIME}(f(n)) = \{L \mid L \text{ is decided by an NTM in } O(f(n)) \text{ time}\}$$
$$\text{SPACE}(f(n)) = \{L \mid L \text{ is decided by a TM using } O(f(n)) \text{ space}\}$$

## 2. The Class P

$$\mathbf{P} = \bigcup_{k \geq 0} \text{TIME}(n^k)$$

$\mathbf{P}$ is the class of languages decidable in **polynomial time** by a deterministic TM.

$\mathbf{P}$ is considered the class of "efficiently solvable" problems.

### 2.1 Examples in P

| Problem | Time | Algorithm |
|---------|------|-----------|
| DFA acceptance | $O(n)$ | Simulate DFA |
| CFG membership | $O(n^3)$ | CYK algorithm |
| Graph connectivity | $O(n + m)$ | BFS/DFS |
| Shortest path | $O(n^2)$ | Dijkstra |
| 2-SAT | $O(n)$ | Implication graph SCCs |
| Primality testing | $O(n^6)$ | AKS algorithm |
| Linear programming | $O(n^3)$ | Interior point methods |

### 2.2 Robustness of P

$\mathbf{P}$ is the same whether we use:
- Single-tape or multi-tape TMs (polynomial slowdown only)
- Any "reasonable" deterministic model of computation

This supports the **Extended Church-Turing thesis**: any "reasonable" model of computation can simulate any other with at most polynomial overhead.

## 3. The Class NP

$$\mathbf{NP} = \bigcup_{k \geq 0} \text{NTIME}(n^k)$$

$\mathbf{NP}$ is the class of languages decidable in polynomial time by a **nondeterministic** TM.

### 3.1 Verifier Characterization

Equivalently, $L \in \mathbf{NP}$ if and only if there exists a polynomial-time **verifier** $V$:

$$L = \{w \mid \exists c \in \{0,1\}^{p(|w|)}, V(w, c) = \text{accept}\}$$

where $c$ is a polynomial-length **certificate** (or witness) and $V$ runs in polynomial time.

**Intuition:** NP problems are those whose solutions are hard to find but easy to verify.

### 3.2 Examples in NP

| Problem | Certificate | Verification |
|---------|-------------|-------------|
| SAT | Satisfying assignment | Evaluate formula |
| Hamiltonian Path | Path listing | Check it visits all vertices |
| Clique | Set of $k$ vertices | Check all pairs connected |
| Subset Sum | Subset of numbers | Sum and compare |
| Graph Coloring | Color assignment | Check no adjacent same color |
| Traveling Salesman (decision) | Tour | Sum edge weights, compare to bound |

### 3.3 P ⊆ NP

Every deterministic TM is a (trivial) nondeterministic TM. So $\mathbf{P} \subseteq \mathbf{NP}$.

The question $\mathbf{P} \stackrel{?}{=} \mathbf{NP}$ is the most important open problem in theoretical computer science.

## 4. NP-Completeness

### 4.1 Polynomial-Time Reductions

Language $A$ is **polynomial-time reducible** to $B$, written $A \leq_P B$, if there exists a polynomial-time computable function $f$ such that $w \in A \iff f(w) \in B$.

### 4.2 Definition

A language $B$ is **NP-complete** if:
1. $B \in \mathbf{NP}$
2. Every $A \in \mathbf{NP}$ satisfies $A \leq_P B$

A language satisfying only condition 2 is **NP-hard**.

**Significance:** If ANY NP-complete problem has a polynomial-time algorithm, then $\mathbf{P} = \mathbf{NP}$. If any NP-complete problem is NOT in P, then $\mathbf{P} \neq \mathbf{NP}$.

### 4.3 The Cook-Levin Theorem

**Theorem (Cook-Levin).** SAT is NP-complete.

**SAT (Boolean Satisfiability):** Given a Boolean formula $\phi$, is there an assignment of truth values that makes $\phi$ true?

*Proof idea:* Given an NTM $M$ running in time $n^k$ and input $w$, construct a Boolean formula $\phi$ that is satisfiable iff $M$ accepts $w$. The formula encodes:
- Variables for the computation tableau (state, tape contents, head position at each time step)
- Clauses ensuring valid start configuration
- Clauses ensuring each step follows the transition function
- Clauses ensuring an accepting configuration is reached

The construction is polynomial in $|w|$, proving SAT is NP-hard. SAT is in NP (the satisfying assignment is the certificate). $\square$

## 5. Classic NP-Complete Problems

### 5.1 Reductions Chain

```
SAT
 └── 3-SAT (restrict clauses to 3 literals)
      ├── CLIQUE (k-clique in graph)
      ├── VERTEX COVER (k vertices covering all edges)
      ├── INDEPENDENT SET
      ├── 3-COLORING
      └── SUBSET SUM
           └── PARTITION
                └── BIN PACKING
```

### 5.2 3-SAT

**3-SAT:** SAT restricted to formulas in 3-CNF (each clause has exactly 3 literals).

**Theorem.** 3-SAT is NP-complete.

*Reduction from SAT:* Replace each long clause with multiple 3-literal clauses using auxiliary variables.

### 5.3 CLIQUE

**CLIQUE:** Given graph $G$ and integer $k$, does $G$ contain a complete subgraph of $k$ vertices?

**Reduction from 3-SAT:** For each clause, create a node per literal. Connect nodes from different clauses unless they are contradictory ($x_i$ and $\neg x_i$). A satisfying assignment corresponds to a $k$-clique where $k$ = number of clauses.

### 5.4 VERTEX COVER

**VERTEX COVER:** Given graph $G$ and integer $k$, is there a set of $k$ vertices that touches every edge?

**Reduction from CLIQUE:** $G$ has a $k$-clique iff $\overline{G}$ has a vertex cover of size $|V| - k$.

### 5.5 SUBSET SUM

**SUBSET SUM:** Given integers $a_1, \ldots, a_n$ and target $t$, is there a subset summing to $t$?

**Reduction from 3-SAT:** Encode variables and clauses as carefully chosen integers where digit positions correspond to variables and clauses.

## 6. coNP

$$\mathbf{coNP} = \{L \mid \overline{L} \in \mathbf{NP}\}$$

coNP is the class of languages whose **complements** are in NP. Equivalently, problems with efficiently verifiable **disproofs**.

**Examples:**
- TAUTOLOGY (is a formula always true?) — in coNP
- UNSAT (is a formula unsatisfiable?) — coNP-complete

**Relationship:** $\mathbf{P} \subseteq \mathbf{NP} \cap \mathbf{coNP}$. Whether $\mathbf{NP} = \mathbf{coNP}$ is open.

## 7. Space Complexity Classes

### 7.1 PSPACE

$$\mathbf{PSPACE} = \bigcup_{k \geq 0} \text{SPACE}(n^k)$$

### 7.2 Key Relationships

$$\mathbf{P} \subseteq \mathbf{NP} \subseteq \mathbf{PSPACE} \subseteq \mathbf{EXPTIME}$$

**Savitch's Theorem:** $\text{NSPACE}(f(n)) \subseteq \text{SPACE}(f(n)^2)$

Therefore $\mathbf{NPSPACE} = \mathbf{PSPACE}$ (nondeterminism doesn't help for polynomial space).

### 7.3 PSPACE-Complete Problems

- **TQBF** (True Quantified Boolean Formula): $\exists x_1 \forall x_2 \exists x_3 \cdots \phi(x_1, \ldots, x_n)$
- **Generalized geography**, **generalized chess**, etc.

## 8. Summary of Complexity Classes

```
EXPTIME
└── PSPACE = NPSPACE (Savitch)
    └── NP ∪ coNP
        ├── NP
        │   └── P
        └── coNP
            └── P
```

At least one containment must be strict (since $\mathbf{P} \neq \mathbf{EXPTIME}$), but we don't know which!

## Practice Exercises

### Exercise 1: Complexity Classification

For each problem, identify the best known complexity class (P, NP-complete, or PSPACE-complete):

1. Shortest path in an unweighted graph
2. Determine if a Boolean formula in 3-CNF is satisfiable
3. Determine if a QBF (quantified Boolean formula) is true
4. 2-coloring a graph

### Exercise 2: Polynomial Reduction

Show that INDEPENDENT SET $\leq_P$ VERTEX COVER. (Hint: $S$ is an independent set iff $V \setminus S$ is a vertex cover.)

### Exercise 3: NP Membership

Show that each problem is in NP by describing the certificate and the polynomial-time verifier:

1. **COMPOSITE**: Given $n$, is $n$ composite (not prime)?
2. **GRAPH ISOMORPHISM**: Given $G_1, G_2$, are they isomorphic?
3. **SET COVER**: Given sets $S_1, \ldots, S_m$ and integer $k$, can $k$ sets cover all elements?

---

**Previous**: [Reducibility](./11_Reducibility.md) | **Next**: [The Chomsky Hierarchy](./13_The_Chomsky_Hierarchy.md)
