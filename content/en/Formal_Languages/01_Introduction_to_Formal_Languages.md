# Lesson 1: Introduction to Formal Languages

**Next**: [Deterministic Finite Automata](./02_Deterministic_Finite_Automata.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Define alphabets, strings, and formal languages mathematically
2. Perform fundamental operations on strings and languages
3. Explain the role of formal languages in computer science
4. Distinguish between different ways of specifying languages

## 1. Motivation

Every programming language, communication protocol, and data format is built on the concept of a **formal language** — a precisely defined set of strings over some alphabet. Understanding formal languages lets us answer fundamental questions:

- What can be computed? (Computability)
- How efficiently can it be computed? (Complexity)
- How do we recognize valid programs? (Parsing)
- What are the limits of mechanical reasoning? (Undecidability)

This topic provides the mathematical framework for these questions.

## 2. Alphabets and Strings

### 2.1 Alphabet

An **alphabet** $\Sigma$ is a finite, nonempty set of symbols.

**Examples:**
- Binary alphabet: $\Sigma = \{0, 1\}$
- Latin alphabet: $\Sigma = \{a, b, c, \ldots, z\}$
- ASCII: $\Sigma = \{$ all 128 ASCII characters $\}$
- DNA: $\Sigma = \{A, C, G, T\}$

### 2.2 String (Word)

A **string** (or **word**) over an alphabet $\Sigma$ is a finite sequence of symbols from $\Sigma$.

- The **empty string** is denoted $\varepsilon$ (sometimes $\lambda$).
- The **length** of a string $w$ is denoted $|w|$.
  - $|\varepsilon| = 0$
  - $|abba| = 4$
- $\Sigma^*$ denotes the set of **all strings** over $\Sigma$, including $\varepsilon$.
- $\Sigma^+$ denotes all **nonempty** strings: $\Sigma^+ = \Sigma^* \setminus \{\varepsilon\}$.

### 2.3 String Operations

**Concatenation:** If $x = a_1 a_2 \cdots a_m$ and $y = b_1 b_2 \cdots b_n$, then $xy = a_1 a_2 \cdots a_m b_1 b_2 \cdots b_n$.

Properties:
- $|xy| = |x| + |y|$
- $x \varepsilon = \varepsilon x = x$ (identity)
- $(xy)z = x(yz)$ (associativity)
- In general, $xy \neq yx$ (not commutative)

**Power:** $w^0 = \varepsilon$, $w^{n+1} = w^n w$ for $n \geq 0$.

**Reversal:** $w^R$ is $w$ written backwards. $(abc)^R = cba$, $\varepsilon^R = \varepsilon$.

**Substring, prefix, suffix:**
- $v$ is a **substring** of $w$ if $w = xvy$ for some $x, y$.
- $v$ is a **prefix** of $w$ if $w = vy$ for some $y$.
- $v$ is a **suffix** of $w$ if $w = xv$ for some $x$.

## 3. Formal Languages

### 3.1 Definition

A **formal language** $L$ over alphabet $\Sigma$ is any subset of $\Sigma^*$:

$$L \subseteq \Sigma^*$$

A language can be finite or infinite.

**Examples** (over $\Sigma = \{0, 1\}$):
- $L_1 = \{0, 01, 011, 0111, \ldots\} = \{0 1^n \mid n \geq 0\}$
- $L_2 = \{w \in \{0,1\}^* \mid w = w^R\}$ (binary palindromes)
- $L_3 = \emptyset$ (the empty language — contains no strings)
- $L_4 = \{\varepsilon\}$ (contains exactly the empty string)

Note: $\emptyset \neq \{\varepsilon\}$ — the empty language versus the language containing only the empty string.

### 3.2 Language Operations

Languages, being sets, support all set operations:

**Union:** $L_1 \cup L_2 = \{w \mid w \in L_1 \text{ or } w \in L_2\}$

**Intersection:** $L_1 \cap L_2 = \{w \mid w \in L_1 \text{ and } w \in L_2\}$

**Complement:** $\overline{L} = \Sigma^* \setminus L = \{w \in \Sigma^* \mid w \notin L\}$

**Difference:** $L_1 \setminus L_2 = L_1 \cap \overline{L_2}$

Additionally, languages have concatenation and closure operations:

**Concatenation:** $L_1 \cdot L_2 = \{xy \mid x \in L_1, y \in L_2\}$

**Power:** $L^0 = \{\varepsilon\}$, $L^{n+1} = L^n \cdot L$

**Kleene star (closure):** $L^* = \bigcup_{n=0}^{\infty} L^n = L^0 \cup L^1 \cup L^2 \cup \cdots$

**Positive closure:** $L^+ = \bigcup_{n=1}^{\infty} L^n = L \cdot L^*$

### 3.3 Examples of Language Operations

Let $L_1 = \{a, ab\}$ and $L_2 = \{b, ba\}$:

- $L_1 \cup L_2 = \{a, ab, b, ba\}$
- $L_1 \cdot L_2 = \{ab, aba, abb, abba\}$
- $L_1^* = \{\varepsilon, a, ab, aa, aab, aba, abab, \ldots\}$

## 4. Specifying Languages

Languages can be specified in several ways:

### 4.1 Enumeration

List all strings (only practical for finite languages):

$$L = \{cat, dog, fish\}$$

### 4.2 Set-Builder Notation

Describe with a predicate:

$$L = \{a^n b^n \mid n \geq 0\} = \{\varepsilon, ab, aabb, aaabbb, \ldots\}$$

### 4.3 Grammars

A set of production rules that generate all strings in the language. Different grammar types (regular, context-free, context-sensitive, unrestricted) define different language classes.

### 4.4 Automata

A computational model that accepts exactly the strings in the language. Different automata (DFA, PDA, TM) recognize different language classes.

### 4.5 Regular Expressions

An algebraic notation for describing patterns. Equivalent to finite automata in expressive power.

## 5. The Chomsky Hierarchy (Preview)

Noam Chomsky classified grammars (and their corresponding languages) into four types:

| Type | Grammar | Automaton | Language Class | Example |
|------|---------|-----------|---------------|---------|
| 3 | Regular | Finite automaton | Regular | $a^*b^*$ |
| 2 | Context-free | Pushdown automaton | Context-free | $a^n b^n$ |
| 1 | Context-sensitive | Linear bounded automaton | Context-sensitive | $a^n b^n c^n$ |
| 0 | Unrestricted | Turing machine | Recursively enumerable | Halting problem |

Each type is strictly contained in the next: Regular $\subsetneq$ Context-free $\subsetneq$ Context-sensitive $\subsetneq$ Recursively enumerable.

We will study each level in detail throughout this topic.

## 6. Decision Problems for Languages

Given a language $L$, fundamental questions include:

- **Membership:** Is $w \in L$? (Can we recognize strings in the language?)
- **Emptiness:** Is $L = \emptyset$?
- **Finiteness:** Is $L$ finite?
- **Equivalence:** Given $L_1, L_2$, is $L_1 = L_2$?
- **Containment:** Is $L_1 \subseteq L_2$?

Whether these questions are **decidable** depends on the language class — a central theme of this topic.

## Practice Exercises

### Exercise 1: String Operations

Let $\Sigma = \{a, b\}$, $x = abba$, $y = bab$.

1. Compute $xy$, $yx$, $x^R$, $y^R$.
2. List all prefixes of $x$.
3. List all substrings of $y$ (no duplicates).

### Exercise 2: Language Operations

Let $L_1 = \{a, bb\}$ and $L_2 = \{b, ab\}$ over $\Sigma = \{a, b\}$.

1. Compute $L_1 \cdot L_2$.
2. Compute $L_1^2 = L_1 \cdot L_1$.
3. List strings in $L_1^*$ of length $\leq 3$.

### Exercise 3: Language Classification

Classify each language and justify informally (we'll prove these formally later):

1. $L = \{w \in \{0,1\}^* \mid w \text{ has an even number of 0s}\}$
2. $L = \{0^n 1^n \mid n \geq 0\}$
3. $L = \{0^n 1^n 2^n \mid n \geq 0\}$
4. The set of all valid C programs

---

**Next**: [Deterministic Finite Automata](./02_Deterministic_Finite_Automata.md)
