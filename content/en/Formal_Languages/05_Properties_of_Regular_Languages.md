# Lesson 5: Properties of Regular Languages

**Previous**: [Regular Expressions](./04_Regular_Expressions.md) | **Next**: [Context-Free Grammars](./06_Context_Free_Grammars.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. State and apply the pumping lemma for regular languages
2. Prove that specific languages are not regular
3. Enumerate closure properties of regular languages
4. State and apply the Myhill-Nerode theorem
5. Identify decidable problems for regular languages

---

Knowing what regular languages *can* do is only half the story. Equally important is understanding their limits -- what they *cannot* express. The pumping lemma and Myhill-Nerode theorem give us rigorous tools to draw this boundary, while closure properties let us build new regular languages from existing ones. These results are essential for choosing the right formalism for any language-recognition task.

## 1. The Pumping Lemma

The pumping lemma provides a necessary condition for regularity. It is the primary tool for proving languages are **not** regular.

**Theorem (Pumping Lemma for Regular Languages).** If $L$ is a regular language, then there exists a constant $p \geq 1$ (the pumping length) such that every string $w \in L$ with $|w| \geq p$ can be written as $w = xyz$ where:

1. $|y| \geq 1$ (the "pump" is nonempty)
2. $|xy| \leq p$ (the pump occurs within the first $p$ symbols)
3. For all $i \geq 0$, $xy^iz \in L$ (pumping $y$ preserves membership)

**Intuition:** In a DFA with $p$ states, any string of length $\geq p$ must revisit some state (pigeonhole principle). The substring between the two visits to the same state can be "pumped" — repeated any number of times.

### 1.1 Proof of the Pumping Lemma

Let $M$ be a DFA for $L$ with $p = |Q|$ states. Consider $w = a_1 a_2 \cdots a_n$ with $n \geq p$.

The computation visits states $q_0, q_1, \ldots, q_n$ where $q_i = \hat{\delta}(q_0, a_1 \cdots a_i)$.

Among the first $p + 1$ states $q_0, q_1, \ldots, q_p$, by the pigeonhole principle, some state must repeat: $q_i = q_j$ for some $0 \leq i < j \leq p$.

Set:
- $x = a_1 \cdots a_i$ (before the repeated state)
- $y = a_{i+1} \cdots a_j$ (the loop)
- $z = a_{j+1} \cdots a_n$ (after the loop)

Then $|y| = j - i \geq 1$, $|xy| = j \leq p$, and pumping $y$ any number of times stays in $L$ because the loop returns to the same state. $\square$

## 2. Using the Pumping Lemma

The pumping lemma is used in **proof by contradiction**: assume $L$ is regular, derive a contradiction.

### 2.1 Example: $L = \{0^n 1^n \mid n \geq 0\}$ is not regular

*Proof.* Assume $L$ is regular with pumping length $p$. Consider $w = 0^p 1^p \in L$ with $|w| = 2p \geq p$.

By the pumping lemma, $w = xyz$ with $|y| \geq 1$, $|xy| \leq p$.

Since $|xy| \leq p$, both $x$ and $y$ consist entirely of 0s. Write $y = 0^k$ where $k \geq 1$.

Pumping down: $xy^0z = xz = 0^{p-k}1^p$. Since $k \geq 1$, this has fewer 0s than 1s, so $xz \notin L$. Contradiction. $\square$

### 2.2 Example: $L = \{ww \mid w \in \{0,1\}^*\}$ is not regular

*Proof.* Assume $L$ is regular with pumping length $p$. Consider $w = 0^p 1 0^p 1 \in L$.

By the pumping lemma, $w = xyz$ with $|xy| \leq p$, so $y = 0^k$ for some $k \geq 1$ (within the first block of 0s).

Then $xy^2z = 0^{p+k} 1 0^p 1$. For this to be in $L$, it must equal $uu$ for some $u$. But $|xy^2z| = 2p + k + 2$, so $|u| = p + k/2 + 1$, which is not an integer when $k$ is odd. Even when $k$ is even, the split doesn't produce equal halves. Contradiction. $\square$

### 2.3 Example: $L = \{0^{n^2} \mid n \geq 0\}$ is not regular

*Proof.* Assume regular with pumping length $p$. Choose $w = 0^{p^2}$.

Write $w = xyz$ with $1 \leq |y| \leq p$. Then $|xy^2z| = p^2 + |y|$.

Since $1 \leq |y| \leq p$: $p^2 < p^2 + |y| \leq p^2 + p < (p+1)^2$.

So $|xy^2z|$ is between consecutive perfect squares and is not itself a perfect square. Thus $xy^2z \notin L$. Contradiction. $\square$

## 3. Closure Properties

Regular languages are closed under many operations:

### 3.1 Boolean Operations

| Operation | Closed? | Proof Technique |
|-----------|---------|-----------------|
| Union $L_1 \cup L_2$ | Yes | NFA construction or product DFA |
| Intersection $L_1 \cap L_2$ | Yes | Product DFA |
| Complement $\overline{L}$ | Yes | Swap accept/reject in DFA |
| Difference $L_1 \setminus L_2$ | Yes | $L_1 \cap \overline{L_2}$ |
| Symmetric diff. $L_1 \triangle L_2$ | Yes | $(L_1 \setminus L_2) \cup (L_2 \setminus L_1)$ |

### 3.2 Regular Operations

| Operation | Closed? | Proof Technique |
|-----------|---------|-----------------|
| Concatenation $L_1 \cdot L_2$ | Yes | NFA construction |
| Kleene star $L^*$ | Yes | NFA construction |
| Reversal $L^R$ | Yes | Reverse all transitions, swap start/accept |

### 3.3 Other Operations

| Operation | Closed? | Proof Technique |
|-----------|---------|-----------------|
| Homomorphism $h(L)$ | Yes | Replace each symbol in NFA |
| Inverse homomorphism $h^{-1}(L)$ | Yes | Simulate DFA on $h(a)$ for each input $a$ |
| Quotient $L_1 / L_2$ | Yes | Modify accept states |

### 3.4 Using Closure for Proofs

Closure properties can prove non-regularity without the pumping lemma.

**Example:** Show $L = \{0^n 1^m \mid n \neq m\}$ is not regular.

*Proof.* Assume $L$ is regular. Since regular languages are closed under complement, $\overline{L}$ is regular. We can write $\overline{L} = \{0^n 1^n \mid n \geq 0\} \cup \{w \mid w \text{ is not of the form } 0^*1^*\}$.

The set $\{w \mid w \text{ is of the form } 0^*1^*\} = 0^*1^*$ is regular. So $\overline{L} \cap 0^*1^* = \{0^n 1^n \mid n \geq 0\}$ would be regular. But we proved it is not. Contradiction. $\square$

## 4. The Myhill-Nerode Theorem

The Myhill-Nerode theorem gives an **exact characterization** of regularity (unlike the pumping lemma, which only provides a necessary condition).

### 4.1 Distinguishability

For a language $L \subseteq \Sigma^*$, two strings $x, y \in \Sigma^*$ are **$L$-distinguishable** if there exists a string $z \in \Sigma^*$ such that exactly one of $xz, yz$ is in $L$.

If no such $z$ exists, $x$ and $y$ are **$L$-indistinguishable**, written $x \equiv_L y$.

### 4.2 The Equivalence Relation $\equiv_L$

$\equiv_L$ is an equivalence relation on $\Sigma^*$ (reflexive, symmetric, transitive). It partitions $\Sigma^*$ into equivalence classes.

### 4.3 The Theorem

**Theorem (Myhill-Nerode).** The following are equivalent:
1. $L$ is regular
2. $L$ is the union of some equivalence classes of a right-invariant equivalence relation of finite index
3. $\equiv_L$ has finite index (finitely many equivalence classes)

Moreover, the number of equivalence classes of $\equiv_L$ equals the number of states in the **minimum DFA** for $L$.

### 4.4 Example: $\{0^n 1^n\}$ via Myhill-Nerode

Consider $L = \{0^n 1^n \mid n \geq 0\}$. For any $i \neq j$, the strings $0^i$ and $0^j$ are $L$-distinguishable: the string $z = 1^i$ gives $0^i 1^i \in L$ but $0^j 1^i \notin L$.

Since $\{0^i \mid i \geq 0\}$ are all pairwise distinguishable, $\equiv_L$ has infinitely many equivalence classes. By Myhill-Nerode, $L$ is not regular.

### 4.5 Example: Minimum DFA Size

For $L = \{w \in \{0,1\}^* \mid w \text{ ends in } 01\}$, the Myhill-Nerode classes are:
- $[\varepsilon]$: strings not ending in 0 or 01 (distinguished by appending $01$)
- $[0]$: strings ending in 0 but not 01 (distinguished by appending $1$)
- $[01]$: strings ending in 01

Three classes → minimum DFA has 3 states.

## 5. Decision Problems for Regular Languages

All these problems are **decidable** for regular languages:

| Problem | Question | Decidable? | Method |
|---------|----------|------------|--------|
| Membership | Is $w \in L$? | Yes | Simulate DFA on $w$ |
| Emptiness | Is $L = \emptyset$? | Yes | Check reachability to accept states |
| Finiteness | Is $L$ finite? | Yes | Check for cycles on paths to accept states |
| Equivalence | Is $L_1 = L_2$? | Yes | Minimize both DFAs and check isomorphism |
| Containment | Is $L_1 \subseteq L_2$? | Yes | Check if $L_1 \cap \overline{L_2} = \emptyset$ |
| Universality | Is $L = \Sigma^*$? | Yes | Check if $\overline{L} = \emptyset$ |

## Practice Exercises

### Exercise 1: Pumping Lemma Proofs

Prove that the following languages are not regular:

1. $L = \{a^n b^{2n} \mid n \geq 0\}$
2. $L = \{w \in \{0,1\}^* \mid w \text{ has equal numbers of 0s and 1s}\}$
3. $L = \{a^p \mid p \text{ is prime}\}$

### Exercise 2: Closure Properties

Use closure properties (not the pumping lemma) to prove:

1. $L = \{w \in \{a,b\}^* \mid |w|_a = |w|_b\}$ is not regular, using the fact that $L \cap a^*b^* = \{a^nb^n\}$
2. If $L$ is regular, then $\text{HALF}(L) = \{x \mid \exists y, |x| = |y|, xy \in L\}$ is regular

### Exercise 3: Myhill-Nerode

Determine the Myhill-Nerode equivalence classes for $L = \{w \in \{a,b\}^* \mid |w|_a \text{ is even}\}$. What is the minimum DFA size?

---

**Previous**: [Regular Expressions](./04_Regular_Expressions.md) | **Next**: [Context-Free Grammars](./06_Context_Free_Grammars.md)
