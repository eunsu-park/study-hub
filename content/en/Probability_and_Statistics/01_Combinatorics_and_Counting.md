# Combinatorics and Counting

## Learning Objectives

After completing this lesson, you will be able to:

1. Apply the addition and multiplication rules to count outcomes of compound experiments
2. Compute permutations with and without repetition
3. Compute combinations with and without repetition (stars and bars)
4. Use multinomial coefficients to count arrangements of multiset elements
5. Apply the inclusion-exclusion principle to union-of-events problems
6. Define derangements and compute $D_n$ using the subfactorial formula
7. Implement counting functions in Python using `math.factorial` and `itertools`

---

## Overview

Combinatorics is the mathematical foundation of probability. Before we can assign probabilities to events, we must be able to count the number of ways those events can occur. This lesson covers the essential counting techniques that underpin all of discrete probability theory.

---

## Table of Contents

1. [Fundamental Counting Principles](#1-fundamental-counting-principles)
2. [Permutations](#2-permutations)
3. [Combinations](#3-combinations)
4. [Multinomial Coefficients](#4-multinomial-coefficients)
5. [Inclusion-Exclusion Principle](#5-inclusion-exclusion-principle)
6. [Derangements](#6-derangements)
7. [Python Examples](#7-python-examples)
8. [Key Takeaways](#8-key-takeaways)

---

## 1. Fundamental Counting Principles

### Addition Rule (Rule of Sum)

If event $A$ can occur in $m$ ways and event $B$ can occur in $n$ ways, and if $A$ and $B$ are **mutually exclusive** (cannot both happen), then $A$ or $B$ can occur in:

$$m + n \text{ ways}$$

**Example**: A student can choose a math elective (5 options) or a physics elective (3 options), but not both. Total choices: $5 + 3 = 8$.

### Multiplication Rule (Rule of Product)

If a procedure consists of two **sequential stages**, where stage 1 can be done in $m$ ways and stage 2 can be done in $n$ ways (regardless of the stage-1 outcome), then the complete procedure can be done in:

$$m \times n \text{ ways}$$

**Example**: A license plate has 2 letters followed by 3 digits. Total plates: $26^2 \times 10^3 = 676{,}000$.

### Generalized Multiplication Rule

For $k$ sequential stages with $n_1, n_2, \ldots, n_k$ choices respectively:

$$\text{Total outcomes} = n_1 \times n_2 \times \cdots \times n_k = \prod_{i=1}^{k} n_i$$

---

## 2. Permutations

A **permutation** is an ordered arrangement of objects.

### Permutations without Repetition

The number of ways to arrange $r$ objects chosen from $n$ distinct objects, where **order matters** and no object is reused:

$$P(n, r) = \frac{n!}{(n - r)!}$$

Special case -- arranging all $n$ objects: $P(n, n) = n!$

**Example**: How many 3-letter "words" from the English alphabet (no repeated letters)?

$$P(26, 3) = \frac{26!}{23!} = 26 \times 25 \times 24 = 15{,}600$$

### Permutations with Repetition

When each of $n$ types of objects can be used **any number of times**, the number of $r$-length sequences is:

$$n^r$$

**Example**: A 4-digit PIN using digits 0--9: $10^4 = 10{,}000$.

### Permutations of Multisets

The number of distinct permutations of $n$ objects where object type $i$ appears $n_i$ times (and $n_1 + n_2 + \cdots + n_k = n$):

$$\frac{n!}{n_1! \, n_2! \, \cdots \, n_k!}$$

**Example**: Distinct arrangements of the letters in "MISSISSIPPI":

- Total letters: 11
- M: 1, I: 4, S: 4, P: 2

$$\frac{11!}{1! \cdot 4! \cdot 4! \cdot 2!} = \frac{39{,}916{,}800}{1 \cdot 24 \cdot 24 \cdot 2} = 34{,}650$$

---

## 3. Combinations

A **combination** is an unordered selection of objects.

### Combinations without Repetition

The number of ways to choose $r$ objects from $n$ distinct objects, where **order does not matter**:

$$\binom{n}{r} = C(n, r) = \frac{n!}{r!(n - r)!}$$

Key properties:

- $\binom{n}{0} = \binom{n}{n} = 1$
- $\binom{n}{r} = \binom{n}{n-r}$ (symmetry)
- $\binom{n}{r} = \binom{n-1}{r-1} + \binom{n-1}{r}$ (Pascal's identity)

**Example**: Choosing a committee of 4 from 10 people:

$$\binom{10}{4} = \frac{10!}{4! \cdot 6!} = \frac{10 \times 9 \times 8 \times 7}{4 \times 3 \times 2 \times 1} = 210$$

### Combinations with Repetition (Stars and Bars)

The number of ways to choose $r$ items from $n$ types **with repetition allowed** (equivalently, distributing $r$ identical balls into $n$ distinct bins):

$$\binom{n + r - 1}{r} = \binom{n + r - 1}{n - 1}$$

**Stars and Bars Intuition**: Represent $r$ items as stars ($\star$) and use $n-1$ bars ($|$) to separate them into $n$ groups. The total number of symbols is $r + n - 1$, and we choose positions for the $n-1$ bars.

**Example**: How many ways to buy 8 donuts from 3 flavors?

$$\binom{3 + 8 - 1}{8} = \binom{10}{8} = 45$$

### The Binomial Theorem

The connection between combinations and algebra:

$$(x + y)^n = \sum_{k=0}^{n} \binom{n}{k} x^k y^{n-k}$$

Setting $x = y = 1$: $\sum_{k=0}^{n} \binom{n}{k} = 2^n$ (total subsets of an $n$-element set).

---

## 4. Multinomial Coefficients

The **multinomial coefficient** generalizes the binomial coefficient to partitions into more than two groups. The number of ways to divide $n$ objects into $k$ groups of sizes $n_1, n_2, \ldots, n_k$ (where $\sum n_i = n$):

$$\binom{n}{n_1, n_2, \ldots, n_k} = \frac{n!}{n_1! \, n_2! \, \cdots \, n_k!}$$

### Multinomial Theorem

$$(x_1 + x_2 + \cdots + x_k)^n = \sum_{\substack{n_1 + n_2 + \cdots + n_k = n \\ n_i \geq 0}} \binom{n}{n_1, n_2, \ldots, n_k} \prod_{i=1}^{k} x_i^{n_i}$$

**Example**: Find the coefficient of $x^2 y^3 z$ in $(x + y + z)^6$:

$$\binom{6}{2, 3, 1} = \frac{6!}{2! \cdot 3! \cdot 1!} = \frac{720}{2 \cdot 6 \cdot 1} = 60$$

---

## 5. Inclusion-Exclusion Principle

### Two-Set Version

For any two events (or sets) $A$ and $B$:

$$|A \cup B| = |A| + |B| - |A \cap B|$$

### Three-Set Version

$$|A \cup B \cup C| = |A| + |B| + |C| - |A \cap B| - |A \cap C| - |B \cap C| + |A \cap B \cap C|$$

### General Formula

For $n$ sets $A_1, A_2, \ldots, A_n$:

$$\left|\bigcup_{i=1}^{n} A_i\right| = \sum_{k=1}^{n} (-1)^{k+1} \sum_{1 \leq i_1 < i_2 < \cdots < i_k \leq n} |A_{i_1} \cap A_{i_2} \cap \cdots \cap A_{i_k}|$$

### Worked Example: Counting with Constraints

**Problem**: How many integers from 1 to 1000 are divisible by 2, 3, or 5?

Let $A$ = multiples of 2, $B$ = multiples of 3, $C$ = multiples of 5.

- $|A| = \lfloor 1000/2 \rfloor = 500$
- $|B| = \lfloor 1000/3 \rfloor = 333$
- $|C| = \lfloor 1000/5 \rfloor = 200$
- $|A \cap B| = \lfloor 1000/6 \rfloor = 166$
- $|A \cap C| = \lfloor 1000/10 \rfloor = 100$
- $|B \cap C| = \lfloor 1000/15 \rfloor = 66$
- $|A \cap B \cap C| = \lfloor 1000/30 \rfloor = 33$

$$|A \cup B \cup C| = 500 + 333 + 200 - 166 - 100 - 66 + 33 = 734$$

---

## 6. Derangements

A **derangement** is a permutation where **no element appears in its original position**. The number of derangements of $n$ elements is denoted $D_n$ (or $!n$):

$$D_n = n! \sum_{k=0}^{n} \frac{(-1)^k}{k!}$$

This can also be written recursively:

$$D_n = (n - 1)(D_{n-1} + D_{n-2}), \quad D_0 = 1, \quad D_1 = 0$$

For large $n$, $D_n \approx n!/e$.

**Example**: Four people each put a hat in a box. Hats are redistributed randomly. What is the probability nobody gets their own hat?

$$D_4 = 4!\left(\frac{1}{0!} - \frac{1}{1!} + \frac{1}{2!} - \frac{1}{3!} + \frac{1}{4!}\right) = 24\left(1 - 1 + \frac{1}{2} - \frac{1}{6} + \frac{1}{24}\right) = 9$$

$$P(\text{derangement}) = \frac{D_4}{4!} = \frac{9}{24} = \frac{3}{8} = 0.375$$

---

## 7. Python Examples

### Basic Counting Functions

```python
import math
from itertools import permutations, combinations, combinations_with_replacement

# --- Permutations and Combinations ---
def P(n, r):
    """Permutations: P(n, r) = n! / (n-r)!"""
    return math.factorial(n) // math.factorial(n - r)

def C(n, r):
    """Combinations: C(n, r) = n! / (r! * (n-r)!)"""
    return math.factorial(n) // (math.factorial(r) * math.factorial(n - r))

# Python 3.8+ has math.comb and math.perm
print(f"P(26, 3) = {math.perm(26, 3)}")   # 15600
print(f"C(10, 4) = {math.comb(10, 4)}")    # 210

# Stars and bars: C(n+r-1, r)
n_types, r_items = 3, 8
print(f"Stars & bars: {math.comb(n_types + r_items - 1, r_items)}")  # 45
```

### Multinomial Coefficient

```python
import math

def multinomial(n, groups):
    """Compute n! / (n1! * n2! * ... * nk!)"""
    assert sum(groups) == n, "Group sizes must sum to n"
    result = math.factorial(n)
    for g in groups:
        result //= math.factorial(g)
    return result

# MISSISSIPPI arrangements
print(f"MISSISSIPPI: {multinomial(11, [1, 4, 4, 2])}")  # 34650

# Coefficient of x^2 y^3 z in (x+y+z)^6
print(f"Multinomial coeff: {multinomial(6, [2, 3, 1])}")  # 60
```

### Inclusion-Exclusion

```python
def divisible_count(limit, divisors):
    """Count integers in [1, limit] divisible by at least one divisor."""
    from itertools import combinations
    n = len(divisors)
    total = 0
    for k in range(1, n + 1):
        sign = (-1) ** (k + 1)
        for combo in combinations(divisors, k):
            # LCM of the combo (for pairwise coprime, it is the product)
            lcm = combo[0]
            for d in combo[1:]:
                lcm = lcm * d // math.gcd(lcm, d)
            total += sign * (limit // lcm)
    return total

result = divisible_count(1000, [2, 3, 5])
print(f"Integers 1-1000 divisible by 2, 3, or 5: {result}")  # 734
```

### Derangements

```python
import math

def derangements(n):
    """Compute D_n using the inclusion-exclusion formula."""
    return sum((-1)**k * math.factorial(n) // math.factorial(k)
               for k in range(n + 1))

for n in range(1, 9):
    d = derangements(n)
    ratio = d / math.factorial(n)
    print(f"D_{n} = {d:>6},  D_{n}/{n}! = {ratio:.6f}")

# Output shows ratio converging to 1/e ~ 0.367879...
```

### Enumeration with itertools

```python
from itertools import permutations, combinations

# All 2-element permutations of {A, B, C}
perms = list(permutations("ABC", 2))
print(f"P(3,2) = {len(perms)}: {perms}")
# 6: [('A','B'), ('A','C'), ('B','A'), ('B','C'), ('C','A'), ('C','B')]

# All 2-element combinations of {A, B, C}
combs = list(combinations("ABC", 2))
print(f"C(3,2) = {len(combs)}: {combs}")
# 3: [('A','B'), ('A','C'), ('B','C')]
```

---

## 8. Key Takeaways

| Concept | Formula | When to Use |
|---------|---------|-------------|
| Multiplication rule | $n_1 \times n_2 \times \cdots \times n_k$ | Sequential independent stages |
| Permutations (no rep.) | $P(n,r) = n!/(n-r)!$ | Ordered selection, distinct items |
| Permutations (with rep.) | $n^r$ | Ordered selection, items reusable |
| Combinations (no rep.) | $\binom{n}{r} = n!/[r!(n-r)!]$ | Unordered selection, distinct items |
| Combinations (with rep.) | $\binom{n+r-1}{r}$ | Unordered selection, items reusable |
| Multinomial coefficient | $n!/(n_1! \cdots n_k!)$ | Partitioning into groups |
| Inclusion-exclusion | $\sum (-1)^{k+1} \cdots$ | Counting unions with overlaps |
| Derangements | $D_n = n! \sum (-1)^k/k!$ | Permutations with no fixed points |

**Decision guide**: Ask two questions about your counting problem:

1. **Does order matter?** Yes -> permutation; No -> combination
2. **Is repetition allowed?** This determines which formula variant to use.

---

*Next lesson: [02 - Probability Axioms and Rules](./02_Probability_Axioms_and_Rules.md)*
