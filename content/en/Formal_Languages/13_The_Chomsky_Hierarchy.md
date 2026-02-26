# Lesson 13: The Chomsky Hierarchy

**Previous**: [Computational Complexity](./12_Computational_Complexity.md) | **Next**: [Advanced Topics and Applications](./14_Advanced_Topics_and_Applications.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Describe all four levels of the Chomsky hierarchy in detail
2. Define context-sensitive grammars and linear bounded automata
3. Compare closure properties, decidability, and power across all levels
4. Classify languages into the appropriate level of the hierarchy

---

The Chomsky hierarchy is the organizing framework for everything we have studied. It arranges formal languages into four nested classes -- regular, context-free, context-sensitive, and recursively enumerable -- each with a corresponding grammar type and automaton model. This lesson brings together all the individual results into a single, unified picture of what each level of computational power can and cannot express.

## 1. Overview

The **Chomsky hierarchy** classifies grammars (and their corresponding languages) into four types, each strictly more powerful than the previous:

$$\text{Type 3 (Regular)} \subsetneq \text{Type 2 (Context-Free)} \subsetneq \text{Type 1 (Context-Sensitive)} \subsetneq \text{Type 0 (Unrestricted)}$$

| Type | Grammar | Automaton | Rule Form |
|------|---------|-----------|-----------|
| 3 | Regular | DFA/NFA | $A \to aB$ or $A \to a$ |
| 2 | Context-free | PDA | $A \to \alpha$ |
| 1 | Context-sensitive | LBA | $\alpha A \beta \to \alpha \gamma \beta$, $|\gamma| \geq 1$ |
| 0 | Unrestricted | TM | $\alpha \to \beta$ (no restrictions) |

## 2. Type 3: Regular Grammars

### 2.1 Definition

A **regular grammar** (or **right-linear grammar**) has rules restricted to:
- $A \to aB$ (terminal followed by variable)
- $A \to a$ (single terminal)
- $A \to \varepsilon$ (only for start variable if needed)

A **left-linear grammar** uses $A \to Ba$ and $A \to a$ instead. Both generate exactly the regular languages.

### 2.2 Equivalence with Finite Automata

Regular grammars, DFAs, NFAs, and regular expressions all define the same class:

$$\text{Regular grammars} = \text{DFA} = \text{NFA} = \text{Regex}$$

### 2.3 Properties Summary

- Closed under: union, intersection, complement, concatenation, Kleene star, reversal, homomorphism
- Decidable: membership, emptiness, equivalence, containment, finiteness
- Cannot express: counting ($a^nb^n$), nesting, cross-serial dependencies

## 3. Type 2: Context-Free Grammars

### 3.1 Definition

A **context-free grammar** has rules of the form:
$$A \to \alpha \quad \text{where } A \in V, \alpha \in (V \cup \Sigma)^*$$

The variable $A$ can be replaced by $\alpha$ **regardless of surrounding context**.

### 3.2 Equivalence with PDA

$$\text{CFG} = \text{PDA (nondeterministic)}$$

Note: DPDA $\subsetneq$ NPDA for PDAs, unlike DFA = NFA for finite automata.

### 3.3 Properties Summary

- Closed under: union, concatenation, Kleene star, reversal, homomorphism, intersection with regular
- **NOT** closed under: intersection, complement
- Decidable: membership (CYK, $O(n^3)$), emptiness, finiteness
- **Undecidable**: equivalence, universality, ambiguity, regularity
- Cannot express: cross-serial dependencies ($a^nb^nc^n$), copy language ($ww$)

## 4. Type 1: Context-Sensitive Grammars

### 4.1 Definition

A **context-sensitive grammar (CSG)** has rules of the form:
$$\alpha A \beta \to \alpha \gamma \beta$$

where $A \in V$, $\alpha, \beta \in (V \cup \Sigma)^*$, and $\gamma \in (V \cup \Sigma)^+$.

The key constraints:
- Variable $A$ is replaced by $\gamma$ only in the **context** of $\alpha$ and $\beta$
- $|\gamma| \geq 1$: rules cannot shorten the sentential form (non-contracting)
- Exception: $S \to \varepsilon$ is allowed if $S$ doesn't appear on the right side of any rule

An equivalent formulation uses **monotone grammars**: $\alpha \to \beta$ where $|\alpha| \leq |\beta|$.

### 4.2 Context-Sensitive Languages (CSL)

**Examples:**
- $\{a^n b^n c^n \mid n \geq 1\}$ — the classic non-CFL
- $\{ww \mid w \in \{a,b\}^*\}$ — the copy language
- $\{a^{n^2} \mid n \geq 1\}$ — perfect squares
- $\{a^{2^n} \mid n \geq 0\}$ — powers of 2
- $\{a^p \mid p \text{ is prime}\}$ — primes

### 4.3 Grammar for $\{a^n b^n c^n\}$

```
S → aSBC | aBC
CB → BC          (context-sensitive: reorder B and C)
aB → ab
bB → bb
bC → bc
cC → cc
```

The rule $CB \to BC$ "sorts" $B$'s before $C$'s, which requires context sensitivity.

### 4.4 Linear Bounded Automata (LBA)

A **linear bounded automaton** is a nondeterministic TM restricted to use only the tape cells containing the input (plus constant overhead).

Formally: an LBA is an NTM where the head never moves beyond the input boundaries (marked by special end markers).

**Theorem.** A language is context-sensitive iff it is accepted by an LBA.

### 4.5 Properties of CSL

- Closed under: union, intersection, complement, concatenation, Kleene star
- Closure under complement was a major open problem, settled by Immerman-Szelepcsényi (1987-88)
- Decidable: membership ($O(n)$ space by LBA simulation)
- **Undecidable**: emptiness, equivalence, containment

### 4.6 The LBA Problem

**Open question (partially resolved):** Is the emptiness problem for LBAs decidable?

This is still open. It is known that $E_{LBA}$ is undecidable for the standard formulation.

## 5. Type 0: Unrestricted Grammars

### 5.1 Definition

An **unrestricted grammar** (or **phrase-structure grammar**) has rules of the form:
$$\alpha \to \beta$$

where $\alpha \in (V \cup \Sigma)^* V (V \cup \Sigma)^*$ (must contain at least one variable) and $\beta \in (V \cup \Sigma)^*$.

No restrictions on the relationship between $|\alpha|$ and $|\beta|$ — rules can shorten strings.

### 5.2 Equivalence with Turing Machines

$$\text{Unrestricted grammars} = \text{Turing machines (recognizers)}$$

The languages generated by unrestricted grammars are exactly the **recursively enumerable** (Turing-recognizable) languages.

### 5.3 Properties

- Closed under: union, intersection, concatenation, Kleene star
- **NOT** closed under: complement (that would make all RE languages decidable)
- **Undecidable**: membership, emptiness, equivalence, everything

## 6. Comprehensive Comparison

### 6.1 Closure Properties

| Operation | Regular | CF | CS | RE |
|-----------|---------|-----|-----|-----|
| Union | ✓ | ✓ | ✓ | ✓ |
| Intersection | ✓ | ✗ | ✓ | ✓ |
| Complement | ✓ | ✗ | ✓ | ✗ |
| Concatenation | ✓ | ✓ | ✓ | ✓ |
| Kleene star | ✓ | ✓ | ✓ | ✓ |
| Homomorphism | ✓ | ✓ | ✗ | ✓ |
| Inverse homomorphism | ✓ | ✓ | ✓ | ✓ |
| Intersection with regular | ✓ | ✓ | ✓ | ✓ |

### 6.2 Decision Problems

| Problem | Regular | CF | CS | RE |
|---------|---------|-----|-----|-----|
| Membership | ✓ | ✓ | ✓ | ✗ |
| Emptiness | ✓ | ✓ | ✗ | ✗ |
| Finiteness | ✓ | ✓ | ✗ | ✗ |
| Equivalence | ✓ | ✗ | ✗ | ✗ |
| Universality | ✓ | ✗ | ✗ | ✗ |
| Containment | ✓ | ✗ | ✗ | ✗ |

### 6.3 Language Examples by Type

| Language | Smallest Type |
|----------|--------------|
| $a^*b^*$ | Regular (Type 3) |
| $a^n b^n$ | Context-free (Type 2) |
| $a^n b^n c^n$ | Context-sensitive (Type 1) |
| $A_{TM}$ | Recursively enumerable (Type 0) |
| $\overline{A_{TM}}$ | Not in any type |

## 7. Beyond the Hierarchy

### 7.1 Decidable (Recursive) Languages

Between CS and RE sits an important class:

$$\text{CS} \subsetneq \text{Decidable} \subsetneq \text{RE}$$

Decidable languages are recognized by TMs that always halt. Some decidable languages are not context-sensitive (e.g., languages requiring superlinear space).

### 7.2 Languages Outside the Hierarchy

There are uncountably many languages over any alphabet, but only countably many TMs. Therefore, **most** languages are not even recursively enumerable.

**Cardinality argument:**
- $|\{0,1\}^*|$ is countable
- $|\mathcal{P}(\{0,1\}^*)|$ (set of all languages) is uncountable
- $|\{\text{TMs}\}|$ is countable
- Therefore, most languages have no TM recognizer

## Practice Exercises

### Exercise 1: Classification

Classify each language into the smallest Chomsky hierarchy level:

1. $\{a^n b^m \mid n > m\}$
2. $\{a^n b^n c^n d^n \mid n \geq 0\}$
3. $\{w \in \{a,b\}^* \mid |w|_a = |w|_b\}$
4. $\{a^p \mid p \text{ is prime}\}$
5. $\{\langle M \rangle \mid M \text{ halts on } \varepsilon\}$

### Exercise 2: Context-Sensitive Grammar

Write a context-sensitive grammar for $\{a^n b^n c^n \mid n \geq 1\}$. Trace the derivation of $aabbcc$.

### Exercise 3: Closure Properties

1. Show that context-sensitive languages are closed under union by constructing a grammar.
2. Explain why closure of CSL under complement is nontrivial (unlike for regular languages where you just flip states in a DFA).

---

**Previous**: [Computational Complexity](./12_Computational_Complexity.md) | **Next**: [Advanced Topics and Applications](./14_Advanced_Topics_and_Applications.md)
