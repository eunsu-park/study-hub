# Lesson 3: Nondeterministic Finite Automata (NFA)

**Previous**: [Deterministic Finite Automata](./02_Deterministic_Finite_Automata.md) | **Next**: [Regular Expressions](./04_Regular_Expressions.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Define NFAs formally, including epsilon-transitions
2. Trace NFA computation as a tree of possibilities
3. Convert an NFA to an equivalent DFA using subset construction
4. Prove that NFAs and DFAs recognize the same class of languages
5. Apply nondeterminism as a design tool

## 1. Motivation

Deterministic machines make exactly one choice at each step. **Nondeterministic** machines can make multiple choices simultaneously — they explore all possible paths in parallel.

Nondeterminism doesn't add computational power to finite automata (NFAs accept exactly the regular languages), but it often makes automata design dramatically simpler.

## 2. Formal Definition

An **NFA** is a 5-tuple $N = (Q, \Sigma, \delta, q_0, F)$ where:

| Component | Description |
|-----------|-------------|
| $Q$ | A finite set of states |
| $\Sigma$ | A finite input alphabet |
| $\delta: Q \times (\Sigma \cup \{\varepsilon\}) \to \mathcal{P}(Q)$ | The transition function |
| $q_0 \in Q$ | The start state |
| $F \subseteq Q$ | The set of accept states |

Key differences from DFA:
- $\delta$ returns a **set** of states (possibly empty): $\delta(q, a) \subseteq Q$
- $\delta$ is defined on $\Sigma \cup \{\varepsilon\}$, allowing **epsilon-transitions** (transitions without consuming input)

## 3. NFA Computation

An NFA **accepts** a string $w$ if **there exists** at least one sequence of transitions leading to an accept state. It **rejects** $w$ only if **all** paths lead to non-accept states or die (reach states with no available transition).

Think of it as a tree of possibilities:
- Each branch represents a choice
- If **any** branch reaches an accept state, the string is accepted
- "Dead" branches (no transition available) simply terminate

## 4. Epsilon-Closure

The **epsilon-closure** of a state $q$, denoted $\text{ECLOSE}(q)$, is the set of states reachable from $q$ by following zero or more $\varepsilon$-transitions:

$$\text{ECLOSE}(q) = \{q\} \cup \{p \mid p \text{ is reachable from } q \text{ via } \varepsilon\text{-transitions}\}$$

For a set of states $S$: $\text{ECLOSE}(S) = \bigcup_{q \in S} \text{ECLOSE}(q)$

## 5. Extended Transition Function for NFA

$$\hat{\delta}(q, \varepsilon) = \text{ECLOSE}(q)$$
$$\hat{\delta}(q, wa) = \text{ECLOSE}\left(\bigcup_{p \in \hat{\delta}(q, w)} \delta(p, a)\right)$$

The language of an NFA: $L(N) = \{w \in \Sigma^* \mid \hat{\delta}(q_0, w) \cap F \neq \emptyset\}$

## 6. NFA Design Examples

### 6.1 Strings Ending in "01"

```
States: {q0, q1, q2}
Start: q0
Accept: {q2}

δ(q0, 0) = {q0, q1}   δ(q0, 1) = {q0}
δ(q1, 0) = ∅           δ(q1, 1) = {q2}
δ(q2, 0) = ∅           δ(q2, 1) = ∅
```

Compare this 3-state NFA with the 3-state DFA from Lesson 2. The NFA "guesses" when the suffix "01" begins.

### 6.2 Strings Containing "aba" or "bab"

An NFA can nondeterministically guess which pattern to look for:

```
States: {q0, qa1, qa2, qa3, qb1, qb2, qb3}
Start: q0
Accept: {qa3, qb3}

From q0: ε-transitions to qa1 and qb1 (start both searches)
  also: δ(q0, a) = {q0}, δ(q0, b) = {q0}  (stay in q0)

qa1 --a--> qa2 --b--> qa3(accept) --a--> qa3 --b--> qa3
qb1 --b--> qb2 --a--> qb3(accept) --a--> qb3 --b--> qb3
```

With a DFA, handling both patterns simultaneously requires careful state tracking. The NFA design is more intuitive.

## 7. Equivalence of DFA and NFA

**Theorem.** A language is accepted by some NFA if and only if it is accepted by some DFA.

*Proof direction: NFA → DFA* (Subset Construction)

Given NFA $N = (Q, \Sigma, \delta, q_0, F)$, construct DFA $D = (Q', \Sigma, \delta', q_0', F')$:

- $Q' = \mathcal{P}(Q)$ — each DFA state is a **set** of NFA states
- $q_0' = \text{ECLOSE}(q_0)$
- $\delta'(S, a) = \text{ECLOSE}\left(\bigcup_{q \in S} \delta(q, a)\right)$
- $F' = \{S \in Q' \mid S \cap F \neq \emptyset\}$

*Proof direction: DFA → NFA* is trivial — every DFA is already an NFA (with singleton transition sets and no $\varepsilon$-transitions).

### 7.1 Subset Construction Example

Consider the NFA for strings ending in "01":

```
NFA states: {q0, q1, q2}
Start: q0, Accept: {q2}
```

Subset construction:

| DFA State | On 0 | On 1 |
|-----------|------|------|
| $\{q_0\}$ | $\{q_0, q_1\}$ | $\{q_0\}$ |
| $\{q_0, q_1\}$ | $\{q_0, q_1\}$ | $\{q_0, q_2\}$ |
| $\{q_0, q_2\}$ | $\{q_0, q_1\}$ | $\{q_0\}$ |

Accept DFA states: $\{q_0, q_2\}$ (contains $q_2 \in F$).

This gives a 3-state DFA, matching the earlier design.

### 7.2 Exponential Blowup

The subset construction can produce up to $2^{|Q|}$ DFA states. This blowup is sometimes unavoidable:

**Example:** The language $L_n = \{w \in \{0,1\}^* \mid \text{the } n\text{th-from-last symbol is } 1\}$

- An NFA needs only $n + 1$ states (guess the position)
- Any DFA requires at least $2^n$ states (must track a window of $n$ symbols)

This exponential gap shows that nondeterminism provides genuine **succinctness** even though it doesn't increase power.

## 8. Closure Under Concatenation

NFAs make it easy to prove closure properties.

**Theorem.** If $L_1$ and $L_2$ are regular, then $L_1 \cdot L_2$ is regular.

*Proof.* Given NFA $N_1$ for $L_1$ and $N_2$ for $L_2$:
1. Add $\varepsilon$-transitions from every accept state of $N_1$ to the start state of $N_2$
2. Make the accept states of $N_1$ non-accepting
3. The start state is $N_1$'s start state; accept states are $N_2$'s accept states

The resulting NFA "guesses" where to split the input between $L_1$ and $L_2$. $\square$

## 9. Closure Under Kleene Star

**Theorem.** If $L$ is regular, then $L^*$ is regular.

*Proof.* Given NFA $N$ for $L$:
1. Add a new start state $q_s$ (also an accept state, for $\varepsilon \in L^*$)
2. Add $\varepsilon$-transition from $q_s$ to the original start state
3. Add $\varepsilon$-transitions from every accept state of $N$ back to the original start state

The NFA can loop through $N$ any number of times. $\square$

## 10. Nondeterminism as a Proof Technique

Nondeterminism is not physically realizable but is invaluable as:
- A **design tool**: Build simple automata that "guess" correctly
- A **proof technique**: Show closure properties by combining automata
- A **theoretical concept**: Central to complexity theory (P vs NP)

The DFA-NFA equivalence means we can freely use nondeterminism when convenient, knowing a (possibly larger) deterministic equivalent exists.

## Practice Exercises

### Exercise 1: NFA Design

Design NFAs (draw or specify formally) for:

1. $L = \{w \in \{a,b\}^* \mid w \text{ contains } aab \text{ as a substring}\}$
2. $L = \{w \in \{0,1\}^* \mid \text{the third symbol from the end is } 0\}$
3. $L = \{w \in \{a,b\}^* \mid |w| \equiv 0 \pmod{2} \text{ or } w \text{ ends with } ab\}$

### Exercise 2: Subset Construction

Perform the subset construction on this NFA:

```
States: {q0, q1, q2}, Start: q0, Accept: {q2}
δ(q0, a) = {q0, q1}   δ(q0, b) = {q0}
δ(q1, a) = ∅           δ(q1, b) = {q2}
δ(q2, a) = ∅           δ(q2, b) = ∅
```

List all reachable DFA states and their transitions.

### Exercise 3: Concatenation Construction

Given NFAs for $L_1 = \{a^n \mid n \geq 1\}$ and $L_2 = \{b^n \mid n \geq 1\}$, construct an NFA for $L_1 \cdot L_2$ using the concatenation construction from Section 8.

---

**Previous**: [Deterministic Finite Automata](./02_Deterministic_Finite_Automata.md) | **Next**: [Regular Expressions](./04_Regular_Expressions.md)
