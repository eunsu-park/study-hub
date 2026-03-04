# Probability Axioms and Rules

## Learning Objectives

After completing this lesson, you will be able to:

1. Define experiments, sample spaces, and events using set-theoretic language
2. State the three Kolmogorov axioms and derive fundamental probability properties from them
3. Compute conditional probabilities and apply the multiplication rule
4. Apply the Law of Total Probability to partition problems
5. Use Bayes' theorem to update probabilities given new evidence
6. Define and verify statistical independence for two or more events
7. Implement probability calculations in Python using simulation

---

## Overview

Probability theory provides a rigorous mathematical framework for reasoning about uncertainty. Rather than relying on intuition, we build probability from three axioms proposed by Andrey Kolmogorov in 1933. Every result in probability -- from Bayes' theorem to the Central Limit Theorem -- follows from these axioms.

---

## Table of Contents

1. [Experiments, Sample Spaces, and Events](#1-experiments-sample-spaces-and-events)
2. [The Kolmogorov Axioms](#2-the-kolmogorov-axioms)
3. [Properties Derived from the Axioms](#3-properties-derived-from-the-axioms)
4. [Conditional Probability](#4-conditional-probability)
5. [Law of Total Probability](#5-law-of-total-probability)
6. [Bayes' Theorem](#6-bayes-theorem)
7. [Independence](#7-independence)
8. [Python Examples](#8-python-examples)
9. [Key Takeaways](#9-key-takeaways)

---

## 1. Experiments, Sample Spaces, and Events

### Random Experiment

A **random experiment** (or trial) is a procedure that:

- Can be repeated under identical conditions
- Has a well-defined set of possible outcomes
- Has an outcome that cannot be predicted with certainty before the experiment

**Examples**: Rolling a die, flipping a coin, measuring a patient's blood pressure.

### Sample Space

The **sample space** $\Omega$ (or $S$) is the set of all possible outcomes of an experiment.

| Experiment | Sample Space |
|-----------|-------------|
| Coin flip | $\Omega = \{H, T\}$ |
| Die roll | $\Omega = \{1, 2, 3, 4, 5, 6\}$ |
| Two coin flips | $\Omega = \{HH, HT, TH, TT\}$ |
| Lifetime of a bulb | $\Omega = [0, \infty)$ |

Sample spaces can be **finite**, **countably infinite** (e.g., number of coin flips until first head), or **uncountably infinite** (e.g., continuous measurements).

### Events

An **event** $A$ is a subset of the sample space: $A \subseteq \Omega$.

- **Simple event**: Contains exactly one outcome, e.g., $\{3\}$
- **Compound event**: Contains multiple outcomes, e.g., "roll an even number" = $\{2, 4, 6\}$
- **Certain event**: $\Omega$ (always occurs)
- **Impossible event**: $\emptyset$ (never occurs)

### Set Operations on Events

| Operation | Notation | Meaning |
|-----------|----------|---------|
| Union | $A \cup B$ | $A$ or $B$ (or both) occurs |
| Intersection | $A \cap B$ | Both $A$ and $B$ occur |
| Complement | $A^c$ or $\bar{A}$ | $A$ does not occur |
| Difference | $A \setminus B$ | $A$ occurs but $B$ does not |
| Mutually exclusive | $A \cap B = \emptyset$ | $A$ and $B$ cannot both occur |

**De Morgan's Laws**:

$$
(A \cup B)^c = A^c \cap B^c, \qquad (A \cap B)^c = A^c \cup B^c
$$

---

## 2. The Kolmogorov Axioms

A **probability function** $P$ assigns a real number to each event in a sigma-algebra $\mathcal{F}$ of subsets of $\Omega$. It must satisfy:

### Axiom 1: Non-Negativity

$$P(A) \geq 0 \quad \text{for every event } A$$

### Axiom 2: Normalization

$$P(\Omega) = 1$$

### Axiom 3: Countable Additivity

If $A_1, A_2, A_3, \ldots$ are pairwise mutually exclusive events (i.e., $A_i \cap A_j = \emptyset$ for $i \neq j$), then:

$$P\left(\bigcup_{i=1}^{\infty} A_i\right) = \sum_{i=1}^{\infty} P(A_i)$$

The triple $(\Omega, \mathcal{F}, P)$ is called a **probability space**.

---

## 3. Properties Derived from the Axioms

Every property below follows logically from the three axioms.

### Complement Rule

$$P(A^c) = 1 - P(A)$$

*Proof*: $A$ and $A^c$ are mutually exclusive with $A \cup A^c = \Omega$. By Axiom 3: $P(A) + P(A^c) = P(\Omega) = 1$.

### Probability of the Empty Set

$$P(\emptyset) = 0$$

### Monotonicity

If $A \subseteq B$, then $P(A) \leq P(B)$.

### Addition Rule (General)

For any two events $A$ and $B$:

$$P(A \cup B) = P(A) + P(B) - P(A \cap B)$$

This is a direct application of inclusion-exclusion from Lesson 01.

### Boole's Inequality (Union Bound)

$$P\left(\bigcup_{i=1}^{n} A_i\right) \leq \sum_{i=1}^{n} P(A_i)$$

### Probability Bounds

$$0 \leq P(A) \leq 1 \quad \text{for all events } A$$

---

## 4. Conditional Probability

### Definition

The **conditional probability** of $A$ given $B$ (where $P(B) > 0$) is:

$$P(A \mid B) = \frac{P(A \cap B)}{P(B)}$$

**Interpretation**: We restrict the sample space to outcomes in $B$ and ask how likely $A$ is within that restricted space.

**Example**: Roll a fair die. Let $A = \{6\}$ and $B = \{\text{even}\} = \{2, 4, 6\}$.

$$P(A \mid B) = \frac{P(\{6\})}{P(\{2,4,6\})} = \frac{1/6}{3/6} = \frac{1}{3}$$

### Multiplication Rule

Rearranging the definition:

$$P(A \cap B) = P(A \mid B) \, P(B) = P(B \mid A) \, P(A)$$

**Chain rule** for multiple events:

$$P(A_1 \cap A_2 \cap \cdots \cap A_n) = P(A_1) \, P(A_2 \mid A_1) \, P(A_3 \mid A_1 \cap A_2) \cdots P(A_n \mid A_1 \cap \cdots \cap A_{n-1})$$

**Example**: Draw 2 cards without replacement from a standard deck. Probability both are aces:

$$P(A_1 \cap A_2) = P(A_1) \cdot P(A_2 \mid A_1) = \frac{4}{52} \cdot \frac{3}{51} = \frac{12}{2652} = \frac{1}{221}$$

---

## 5. Law of Total Probability

### Partition of the Sample Space

Events $B_1, B_2, \ldots, B_n$ form a **partition** of $\Omega$ if:

1. They are mutually exclusive: $B_i \cap B_j = \emptyset$ for $i \neq j$
2. They are exhaustive: $B_1 \cup B_2 \cup \cdots \cup B_n = \Omega$
3. Each has positive probability: $P(B_i) > 0$

### The Law

If $\{B_1, B_2, \ldots, B_n\}$ is a partition of $\Omega$, then for any event $A$:

$$P(A) = \sum_{i=1}^{n} P(A \mid B_i) \, P(B_i)$$

**Example**: A factory has two machines. Machine 1 produces 60% of items (2% defective). Machine 2 produces 40% of items (5% defective). What is the probability a randomly chosen item is defective?

$$P(D) = P(D \mid M_1)P(M_1) + P(D \mid M_2)P(M_2) = (0.02)(0.60) + (0.05)(0.40) = 0.032$$

---

## 6. Bayes' Theorem

### Statement

Given a partition $\{B_1, B_2, \ldots, B_n\}$ and an event $A$ with $P(A) > 0$:

$$P(B_j \mid A) = \frac{P(A \mid B_j) \, P(B_j)}{\sum_{i=1}^{n} P(A \mid B_i) \, P(B_i)} = \frac{P(A \mid B_j) \, P(B_j)}{P(A)}$$

**Terminology**:

- $P(B_j)$: **prior** probability (before observing evidence)
- $P(A \mid B_j)$: **likelihood** (probability of evidence given hypothesis)
- $P(B_j \mid A)$: **posterior** probability (after observing evidence)
- $P(A)$: **marginal likelihood** or **evidence**

### Medical Test Example

A disease affects 1 in 1000 people ($P(D) = 0.001$). A test has:

- Sensitivity (true positive rate): $P(+ \mid D) = 0.99$
- Specificity (true negative rate): $P(- \mid D^c) = 0.95$

If a person tests positive, what is $P(D \mid +)$?

**Step 1**: Compute $P(+)$ using the Law of Total Probability:

$$P(+) = P(+ \mid D)P(D) + P(+ \mid D^c)P(D^c) = (0.99)(0.001) + (0.05)(0.999) = 0.05094$$

**Step 2**: Apply Bayes' theorem:

$$P(D \mid +) = \frac{(0.99)(0.001)}{0.05094} = \frac{0.00099}{0.05094} \approx 0.0194$$

Despite the apparently good test, a positive result gives only a 1.94% chance of actually having the disease. This is called the **base rate fallacy** -- when the disease is rare, most positives are false positives.

---

## 7. Independence

### Definition for Two Events

Events $A$ and $B$ are **independent** if:

$$P(A \cap B) = P(A) \cdot P(B)$$

Equivalently (when $P(B) > 0$): $P(A \mid B) = P(A)$ -- knowing $B$ occurred does not change the probability of $A$.

### Mutual Independence

Events $A_1, A_2, \ldots, A_n$ are **mutually independent** if for every subset $\{i_1, i_2, \ldots, i_k\} \subseteq \{1, 2, \ldots, n\}$:

$$P(A_{i_1} \cap A_{i_2} \cap \cdots \cap A_{i_k}) = P(A_{i_1}) \cdot P(A_{i_2}) \cdots P(A_{i_k})$$

**Warning**: Pairwise independence does not imply mutual independence.

### Properties of Independent Events

If $A$ and $B$ are independent, then so are:

- $A$ and $B^c$
- $A^c$ and $B$
- $A^c$ and $B^c$

### Independence vs. Mutual Exclusivity

- **Mutually exclusive**: $P(A \cap B) = 0$ -- if $A$ occurs, $B$ cannot (maximum negative dependence for non-trivial events)
- **Independent**: $P(A \cap B) = P(A)P(B)$ -- occurrence of one tells nothing about the other

If $P(A) > 0$ and $P(B) > 0$, mutually exclusive events are **never** independent (since $0 \neq P(A)P(B)$).

---

## 8. Python Examples

### Simulating Conditional Probability

```python
import random

def simulate_conditional(n_trials=1_000_000):
    """Simulate P(die=6 | die is even) by rejection sampling."""
    random.seed(42)
    even_count = 0
    six_and_even = 0

    for _ in range(n_trials):
        roll = random.randint(1, 6)
        if roll % 2 == 0:          # Condition: even
            even_count += 1
            if roll == 6:
                six_and_even += 1

    estimate = six_and_even / even_count
    print(f"P(6 | even) ~ {estimate:.4f}  (exact: 0.3333)")

simulate_conditional()
```

### Bayes' Theorem: Medical Test

```python
def bayes_medical_test(prevalence, sensitivity, specificity):
    """Compute P(Disease | Positive test) using Bayes' theorem."""
    p_d = prevalence
    p_pos_given_d = sensitivity
    p_pos_given_not_d = 1 - specificity

    # Law of total probability
    p_pos = p_pos_given_d * p_d + p_pos_given_not_d * (1 - p_d)

    # Bayes' theorem
    p_d_given_pos = (p_pos_given_d * p_d) / p_pos

    print(f"Prevalence:   {p_d}")
    print(f"Sensitivity:  {sensitivity}")
    print(f"Specificity:  {specificity}")
    print(f"P(+):         {p_pos:.6f}")
    print(f"P(D | +):     {p_d_given_pos:.6f}")
    return p_d_given_pos

bayes_medical_test(prevalence=0.001, sensitivity=0.99, specificity=0.95)
# P(D | +) ~ 0.019417
```

### Verifying Independence by Simulation

```python
import random

def test_independence(n_trials=500_000):
    """Test whether two coin flips are independent via simulation."""
    random.seed(0)
    count_a = 0   # First flip is H
    count_b = 0   # Second flip is H
    count_ab = 0  # Both are H

    for _ in range(n_trials):
        flip1 = random.choice(["H", "T"])
        flip2 = random.choice(["H", "T"])
        a = (flip1 == "H")
        b = (flip2 == "H")
        count_a += a
        count_b += b
        count_ab += (a and b)

    p_a = count_a / n_trials
    p_b = count_b / n_trials
    p_ab = count_ab / n_trials

    print(f"P(A)      = {p_a:.4f}")
    print(f"P(B)      = {p_b:.4f}")
    print(f"P(A)*P(B) = {p_a * p_b:.4f}")
    print(f"P(A & B)  = {p_ab:.4f}")
    print(f"Independent? P(AB) ~ P(A)P(B): {abs(p_ab - p_a * p_b) < 0.005}")

test_independence()
```

### Law of Total Probability: Factory Example

```python
def factory_defect():
    """Compute defect probability using the Law of Total Probability."""
    # Machine 1: 60% of production, 2% defect rate
    # Machine 2: 40% of production, 5% defect rate
    machines = [
        {"name": "M1", "share": 0.60, "defect_rate": 0.02},
        {"name": "M2", "share": 0.40, "defect_rate": 0.05},
    ]

    p_defect = sum(m["share"] * m["defect_rate"] for m in machines)
    print(f"P(Defective) = {p_defect:.4f}")  # 0.0320

    # Reverse: given defective, which machine? (Bayes)
    for m in machines:
        posterior = (m["defect_rate"] * m["share"]) / p_defect
        print(f"P({m['name']} | Defective) = {posterior:.4f}")

factory_defect()
```

---

## 9. Key Takeaways

1. **Probability is built on three axioms**: non-negativity, normalization, and countable additivity. Every theorem in probability follows from these.

2. **Conditional probability** $P(A \mid B) = P(A \cap B)/P(B)$ refines our probability assessment after gaining partial information.

3. **Bayes' theorem** inverts conditional probabilities -- it computes the probability of a cause given an observed effect.

4. **The Law of Total Probability** lets us compute $P(A)$ by summing over a partition of the sample space.

5. **Independence** ($P(A \cap B) = P(A)P(B)$) is the mathematical formalization of "no influence." It is fundamentally different from mutual exclusivity.

6. **Base rate matters**: Even highly accurate tests produce mostly false positives when the underlying condition is rare (Bayes' theorem quantifies this precisely).

---

*Previous: [01 - Combinatorics and Counting](./01_Combinatorics_and_Counting.md) | Next: [03 - Random Variables and Distributions](./03_Random_Variables_and_Distributions.md)*
