# Lesson 8: Properties of Context-Free Languages

**Previous**: [Pushdown Automata](./07_Pushdown_Automata.md) | **Next**: [Turing Machines](./09_Turing_Machines.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. State and apply the pumping lemma for context-free languages
2. Prove that specific languages are not context-free
3. Enumerate closure and non-closure properties of CFLs
4. Distinguish decidable from undecidable problems for CFLs

---

Just as the pumping lemma for regular languages revealed their boundaries, context-free languages have their own limits. Languages requiring three-way coordination (like $a^n b^n c^n$) lie beyond the reach of any CFG or PDA. Understanding these limits -- along with what is and is not decidable for CFLs -- is critical for knowing when you need a more powerful formalism than a context-free grammar.

## 1. The Pumping Lemma for CFLs

**Theorem (Pumping Lemma for Context-Free Languages).** If $L$ is a context-free language, then there exists a constant $p \geq 1$ such that every $w \in L$ with $|w| \geq p$ can be written as $w = uvxyz$ where:

1. $|vy| \geq 1$ (at least one of $v, y$ is nonempty)
2. $|vxy| \leq p$ ($v, x, y$ together are bounded)
3. For all $i \geq 0$, $uv^ixy^iz \in L$ (pumping $v$ and $y$ together preserves membership)

### 1.1 Intuition

In a CNF grammar, a parse tree for a long string must be tall. In a tall tree, some variable $A$ must repeat on a path from root to leaf. The subtree between the two occurrences of $A$ produces the "pump" portions $v$ and $y$.

### 1.2 Proof Sketch

Let $G$ be a CNF grammar for $L$ with $|V|$ variables. Set $p = 2^{|V|+1}$.

For any $w \in L$ with $|w| \geq p$, the parse tree has height $> |V|$. By the pigeonhole principle, some path from root to leaf contains a repeated variable $A$.

The two occurrences of $A$ define:
- $u$: yield to the left of the outer $A$'s subtree
- $v$: yield between outer and inner $A$'s left portions
- $x$: yield of the inner $A$'s subtree
- $y$: yield between inner and outer $A$'s right portions
- $z$: yield to the right of the outer $A$'s subtree

Since $A \Rightarrow^* vAy$ and $A \Rightarrow^* x$, we can pump: $A \Rightarrow^* v^i x y^i$ for any $i \geq 0$. $\square$

## 2. Pumping Lemma Applications

### 2.1 $L = \{a^n b^n c^n \mid n \geq 0\}$ is not context-free

*Proof.* Assume CFL with pumping length $p$. Choose $w = a^p b^p c^p$.

Write $w = uvxyz$ with $|vy| \geq 1$, $|vxy| \leq p$.

Since $|vxy| \leq p$, the substring $vxy$ cannot span all three symbol types. So $v$ and $y$ together contain at most two of $\{a, b, c\}$.

Pumping up ($i = 2$): $uv^2xy^2z$ has more of at most two symbols but not the third. The counts are no longer equal, so $uv^2xy^2z \notin L$. Contradiction. $\square$

### 2.2 $L = \{ww \mid w \in \{a,b\}^*\}$ is not context-free

*Proof.* Assume CFL with pumping length $p$. Choose $w = a^p b^p a^p b^p$.

Write $w = uvxyz$ with $|vxy| \leq p$.

Since $|vxy| \leq p$, the pump region falls within one of these zones:
- Within the first $a^p b^p$: Pumping changes the first half but not the second
- Spanning the boundary of the two halves: Pumping disrupts the structure

In either case, pumping produces a string not of the form $ww$. (The detailed case analysis confirms all positions lead to contradiction.) $\square$

### 2.3 $L = \{a^{n!} \mid n \geq 0\}$ is not context-free

*Proof.* Assume CFL with pumping length $p$. Choose $w = a^{p!}$ where $p! \geq p$.

Write $w = uvxyz$ with $|vy| = k$ where $1 \leq k \leq p$.

Then $|uv^ixy^iz| = p! + (i-1)k$.

For $i = 2$: $|uv^2xy^2z| = p! + k$. We need $p! + k$ to be a factorial.

But $p! < p! + k \leq p! + p < p! + p \cdot p! = (p+1)!$.

So $p! + k$ lies strictly between $p!$ and $(p+1)!$ and is not a factorial. Contradiction. $\square$

## 3. Closure Properties

### 3.1 Closed Operations

| Operation | Closed? | Proof Technique |
|-----------|---------|-----------------|
| Union $L_1 \cup L_2$ | Yes | $S \to S_1 \mid S_2$ |
| Concatenation $L_1 \cdot L_2$ | Yes | $S \to S_1 S_2$ |
| Kleene star $L^*$ | Yes | $S \to S S_1 \mid \varepsilon$ |
| Reversal $L^R$ | Yes | Reverse all production rules |
| Homomorphism $h(L)$ | Yes | Replace terminals in grammar |
| Inverse homomorphism $h^{-1}(L)$ | Yes | PDA construction |
| Intersection with regular $L_1 \cap R$ | Yes | PDA × DFA product |
| Substitution | Yes | Replace terminals with CFGs |

### 3.2 Non-Closed Operations

| Operation | Closed? | Counterexample |
|-----------|---------|---------------|
| Intersection $L_1 \cap L_2$ | **No** | $\{a^n b^n c^m\} \cap \{a^m b^n c^n\} = \{a^n b^n c^n\}$ |
| Complement $\overline{L}$ | **No** | If closed under complement and union, would be closed under intersection (De Morgan) |

### 3.3 Intersection with Regular Languages

**Theorem.** If $L$ is context-free and $R$ is regular, then $L \cap R$ is context-free.

*Proof.* Construct a PDA for $L \cap R$ by running the PDA for $L$ and the DFA for $R$ in parallel:
- States: $Q_{PDA} \times Q_{DFA}$
- Stack: same as the PDA
- Transitions: advance both machines simultaneously
- Accept: when both machines accept

This is extremely useful for proving non-context-freeness using closure:

**Example:** $L = \{w \in \{a,b,c\}^* \mid |w|_a = |w|_b = |w|_c\}$ is not CFL.

*Proof.* Suppose $L$ is CFL. Then $L \cap a^*b^*c^* = \{a^n b^n c^n \mid n \geq 0\}$ would be CFL (intersection with the regular language $a^*b^*c^*$). Contradiction. $\square$

## 4. Ogden's Lemma

**Ogden's Lemma** is a strengthened version of the CFL pumping lemma where we "mark" at least $p$ positions and the pump must include marked positions.

**Theorem.** If $L$ is CFL, then $\exists p$ such that for any $w \in L$ with at least $p$ marked positions, $w = uvxyz$ where:
1. $v$ and $y$ together have at least one marked position
2. $vxy$ has at most $p$ marked positions
3. $\forall i \geq 0$, $uv^ixy^iz \in L$

Ogden's Lemma can prove non-context-freeness in cases where the standard pumping lemma fails.

## 5. Decision Problems for CFLs

### 5.1 Decidable Problems

| Problem | Decidable? | Method |
|---------|------------|--------|
| Membership: $w \in L(G)$? | Yes | CYK algorithm, $O(n^3)$ |
| Emptiness: $L(G) = \emptyset$? | Yes | Check if start variable is generating |
| Finiteness: Is $L(G)$ finite? | Yes | Check for cycles in the "unit graph" of useful variables |

### 5.2 Undecidable Problems

| Problem | Decidable? |
|---------|------------|
| Universality: $L(G) = \Sigma^*$? | **No** |
| Equivalence: $L(G_1) = L(G_2)$? | **No** |
| Inclusion: $L(G_1) \subseteq L(G_2)$? | **No** |
| Ambiguity: Is $G$ ambiguous? | **No** |
| Inherent ambiguity: Is $L$ inherently ambiguous? | **No** |
| Regularity: Is $L(G)$ regular? | **No** |
| Intersection emptiness: $L(G_1) \cap L(G_2) = \emptyset$? | **No** |

The undecidability of these problems contrasts sharply with regular languages, where all are decidable.

## 6. Context-Free Languages vs. Regular Languages

| Property | Regular | Context-Free |
|----------|---------|-------------|
| Automaton | DFA/NFA | PDA |
| Grammar | Right-linear | Context-free |
| Algebraic | Regular expression | — |
| Pumping lemma | 3-part split | 5-part split |
| Closed under ∩ | Yes | **No** |
| Closed under complement | Yes | **No** |
| Equivalence decidable | Yes | **No** |
| DFA = NFA | Yes | DPDA ≠ NPDA |

## Practice Exercises

### Exercise 1: Pumping Lemma

Prove that these languages are not context-free:

1. $L = \{a^n b^n c^n d^n \mid n \geq 0\}$
2. $L = \{a^n b^n c^i \mid n \leq i \leq 2n\}$
3. $L = \{a^{2^n} \mid n \geq 0\}$

### Exercise 2: Closure Properties

1. Prove that $\{a^n b^m c^n d^m \mid n, m \geq 0\}$ is not CFL using closure with regular languages.
2. Show that $L_1 = \{a^n b^n c^m \mid n, m \geq 0\}$ and $L_2 = \{a^m b^n c^n \mid n, m \geq 0\}$ are both CFL by giving grammars.
3. Verify that $L_1 \cap L_2 = \{a^n b^n c^n \mid n \geq 0\}$ and conclude that CFLs are not closed under intersection.

### Exercise 3: Decision Problems

For each problem, state whether it is decidable for CFLs and briefly explain why:

1. Given CFG $G$, is $|L(G)| \geq 100$?
2. Given CFGs $G_1, G_2$, is $L(G_1) \cap L(G_2) = \emptyset$?
3. Given CFG $G$ and regular expression $R$, is $L(G) \subseteq L(R)$?

---

**Previous**: [Pushdown Automata](./07_Pushdown_Automata.md) | **Next**: [Turing Machines](./09_Turing_Machines.md)
