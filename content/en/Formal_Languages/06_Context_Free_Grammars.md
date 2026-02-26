# Lesson 6: Context-Free Grammars

**Previous**: [Properties of Regular Languages](./05_Properties_of_Regular_Languages.md) | **Next**: [Pushdown Automata](./07_Pushdown_Automata.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Define context-free grammars formally
2. Construct grammars for given languages
3. Analyze derivations, parse trees, and ambiguity
4. Convert grammars to Chomsky Normal Form and Greibach Normal Form
5. Apply the CYK parsing algorithm

---

Context-free grammars are the backbone of programming language design. Every time a compiler parses your source code, it uses a CFG (or a close variant) to determine whether the syntax is valid and to build the parse tree that drives code generation. CFGs can express nested and recursive structures -- like matching parentheses and nested function calls -- that regular languages cannot.

## 1. Formal Definition

A **context-free grammar (CFG)** is a 4-tuple $G = (V, \Sigma, R, S)$ where:

| Component | Description |
|-----------|-------------|
| $V$ | A finite set of **variables** (nonterminals) |
| $\Sigma$ | A finite set of **terminals** (disjoint from $V$) |
| $R$ | A finite set of **production rules** of the form $A \to \alpha$ where $A \in V$ and $\alpha \in (V \cup \Sigma)^*$ |
| $S \in V$ | The **start variable** |

The key restriction: the left side of each rule is a **single** variable. This is what makes it "context-free" — a variable can be replaced regardless of context.

## 2. Derivations

A **derivation** is a sequence of rule applications that transforms the start variable into a string of terminals.

**Notation:**
- $\alpha \Rightarrow \beta$: one-step derivation ($\alpha$ yields $\beta$ by one rule)
- $\alpha \Rightarrow^* \beta$: zero or more derivation steps

**The language** of a grammar: $L(G) = \{w \in \Sigma^* \mid S \Rightarrow^* w\}$

### 2.1 Example

Grammar $G_1$ for $\{0^n 1^n \mid n \geq 0\}$:

$$S \to 0S1 \mid \varepsilon$$

Derivation of $000111$:
$$S \Rightarrow 0S1 \Rightarrow 00S11 \Rightarrow 000S111 \Rightarrow 000111$$

### 2.2 Leftmost and Rightmost Derivations

- **Leftmost derivation**: Always expand the leftmost variable first
- **Rightmost derivation**: Always expand the rightmost variable first

**Example:** Grammar $S \to AB$, $A \to aA \mid a$, $B \to bB \mid b$

Leftmost derivation of $aabb$:
$$S \Rightarrow AB \Rightarrow aAB \Rightarrow aaB \Rightarrow aabB \Rightarrow aabb$$

Rightmost derivation of $aabb$:
$$S \Rightarrow AB \Rightarrow AbB \Rightarrow Abb \Rightarrow aAbb \Rightarrow aabb$$

## 3. Parse Trees

A **parse tree** is a graphical representation of a derivation:
- Root is the start variable
- Internal nodes are variables
- Leaves are terminals or $\varepsilon$
- For each internal node $A$ with children $X_1, \ldots, X_k$, there is a rule $A \to X_1 \cdots X_k$

The **yield** of a parse tree is the string formed by reading the leaves left to right.

A string may have multiple leftmost derivations but they always correspond to different parse trees.

## 4. Ambiguity

A grammar $G$ is **ambiguous** if some string $w \in L(G)$ has two or more distinct parse trees (equivalently, two or more leftmost derivations).

### 4.1 Example: Ambiguous Grammar

$$E \to E + E \mid E \times E \mid (E) \mid a$$

The string $a + a \times a$ has two parse trees:
1. $(a + a) \times a$ — addition first
2. $a + (a \times a)$ — multiplication first

### 4.2 Resolving Ambiguity

Introduce precedence and associativity through grammar structure:

$$E \to E + T \mid T$$
$$T \to T \times F \mid F$$
$$F \to (E) \mid a$$

This grammar is **unambiguous** and encodes $\times$ before $+$ (standard precedence).

### 4.3 Inherent Ambiguity

A **context-free language** is **inherently ambiguous** if every grammar for it is ambiguous.

**Example:** $L = \{a^i b^j c^k \mid i = j \text{ or } j = k\}$ is inherently ambiguous.

Not all CFLs are inherently ambiguous — most practical languages have unambiguous grammars.

## 5. Grammar Transformations

### 5.1 Eliminating $\varepsilon$-Productions

A production $A \to \varepsilon$ is an **$\varepsilon$-production**. To eliminate them:

1. Find all **nullable** variables (those that can derive $\varepsilon$)
2. For each rule $A \to X_1 X_2 \cdots X_k$, add rules with all combinations of nullable variables present/absent
3. Remove all $\varepsilon$-productions (except possibly $S \to \varepsilon$ if $\varepsilon \in L$; in that case, ensure $S$ doesn't appear on the right side of any rule)

### 5.2 Eliminating Unit Productions

A **unit production** has the form $A \to B$ where $B \in V$.

1. Compute the unit-pair relation: all pairs $(A, B)$ where $A \Rightarrow^* B$ using only unit productions
2. For each unit pair $(A, B)$ and each non-unit production $B \to \alpha$, add $A \to \alpha$
3. Remove all unit productions

### 5.3 Eliminating Useless Symbols

A symbol is **useful** if it appears in some derivation $S \Rightarrow^* w$ for $w \in \Sigma^*$.

1. Remove **non-generating** variables (those that cannot derive any terminal string)
2. Remove **unreachable** symbols (those not reachable from $S$)

Order matters: do generating check first, then reachability.

## 6. Chomsky Normal Form (CNF)

A CFG is in **Chomsky Normal Form** if every rule has one of these forms:
- $A \to BC$ (two variables)
- $A \to a$ (single terminal)
- $S \to \varepsilon$ (only if $\varepsilon \in L$, and $S$ doesn't appear on any right side)

**Theorem.** Every CFL has a grammar in CNF.

**Conversion algorithm:**
1. Eliminate $\varepsilon$-productions (except $S \to \varepsilon$)
2. Eliminate unit productions
3. Replace terminals in long rules: for $A \to aBC$, add $X_a \to a$ and write $A \to X_a BC$
4. Break long rules: for $A \to BCD$, add $A \to BX$, $X \to CD$

### 6.1 CNF Example

Original: $S \to aSb \mid ab$

Step 3: Replace terminals
- $X_a \to a$, $X_b \to b$
- $S \to X_a S X_b \mid X_a X_b$

Step 4: Break long rule $S \to X_a S X_b$
- $S \to X_a Y$, $Y \to S X_b$
- $S \to X_a X_b$

CNF result:
$$S \to X_a Y \mid X_a X_b, \quad Y \to S X_b, \quad X_a \to a, \quad X_b \to b$$

## 7. Greibach Normal Form (GNF)

A CFG is in **Greibach Normal Form** if every rule has the form:
- $A \to a B_1 B_2 \cdots B_k$ (terminal followed by zero or more variables, $k \geq 0$)
- $S \to \varepsilon$ (if needed)

**Theorem.** Every CFL has a grammar in GNF.

GNF is useful because:
- Each derivation step consumes exactly one input symbol
- The derivation length equals the string length (no $\varepsilon$-productions in body)
- Directly corresponds to a PDA without $\varepsilon$-transitions (except for accepting $\varepsilon$)

## 8. The CYK Algorithm

The **Cocke-Younger-Kasami (CYK)** algorithm determines whether $w \in L(G)$ for a grammar $G$ in CNF. It uses dynamic programming.

**Input:** Grammar $G$ in CNF, string $w = a_1 a_2 \cdots a_n$

**Table:** $T[i][j]$ = set of variables that can derive $a_i a_{i+1} \cdots a_j$

**Algorithm:**
```
// Base case: substrings of length 1
for i = 1 to n:
    T[i][i] = {A | A → a_i is a rule}

// Inductive case: substrings of length 2, 3, ..., n
for length = 2 to n:
    for i = 1 to n - length + 1:
        j = i + length - 1
        T[i][j] = ∅
        for k = i to j - 1:
            for each rule A → BC:
                if B ∈ T[i][k] and C ∈ T[k+1][j]:
                    T[i][j] = T[i][j] ∪ {A}

// Accept if start variable derives entire string
accept if S ∈ T[1][n]
```

**Time complexity:** $O(n^3 \cdot |G|)$

**Space complexity:** $O(n^2)$

### 8.1 CYK Example

Grammar in CNF: $S \to AB$, $A \to a$, $B \to b$, $S \to a$ (for the language $\{a, ab\}$, simplified).

For more complex grammars, the CYK table fills systematically bottom-up — see the example code.

## Practice Exercises

### Exercise 1: Grammar Design

Write CFGs for:

1. $L = \{a^i b^j \mid i \geq j \geq 0\}$
2. $L = \{w \in \{a,b\}^* \mid |w|_a = |w|_b\}$
3. $L = \{a^i b^j c^k \mid i + k = j\}$

### Exercise 2: CNF Conversion

Convert this grammar to Chomsky Normal Form:

$$S \to aAb \mid B$$
$$A \to aA \mid a$$
$$B \to bB \mid \varepsilon$$

Show each transformation step.

### Exercise 3: CYK Parsing

Apply the CYK algorithm to check if $w = aabb$ is in the language generated by:

$$S \to AB \mid BC, \quad A \to BA \mid a, \quad B \to CC \mid b, \quad C \to AB \mid a$$

Fill in the complete CYK table.

---

**Previous**: [Properties of Regular Languages](./05_Properties_of_Regular_Languages.md) | **Next**: [Pushdown Automata](./07_Pushdown_Automata.md)
