# Formal Languages and Automata Theory

## Overview

This topic covers the mathematical foundations of computation: formal languages, automata, computability, and complexity theory. While Compiler_Design applies these concepts to build compilers, this topic focuses on the theoretical framework — rigorous definitions, proofs, closure properties, and fundamental limits of computation.

## Prerequisites

- Basic discrete mathematics (sets, functions, relations, proof techniques)
- Programming fundamentals (for running example code)
- Recommended: Compiler_Design (for practical context)

## Learning Path

### Part I: Finite Automata and Regular Languages (L01-L05)

| Lesson | Title | Key Concepts |
|--------|-------|--------------|
| 01 | Introduction to Formal Languages | Alphabets, strings, languages, language operations |
| 02 | Deterministic Finite Automata | DFA definition, state diagrams, language recognition |
| 03 | Nondeterministic Finite Automata | NFA, ε-transitions, subset construction, DFA equivalence |
| 04 | Regular Expressions | Syntax, semantics, equivalence with finite automata |
| 05 | Properties of Regular Languages | Pumping lemma, closure properties, Myhill-Nerode theorem, minimization |

### Part II: Context-Free Languages (L06-L08)

| Lesson | Title | Key Concepts |
|--------|-------|--------------|
| 06 | Context-Free Grammars | Productions, derivations, parse trees, ambiguity, normal forms |
| 07 | Pushdown Automata | PDA definition, acceptance modes, equivalence with CFG |
| 08 | Properties of Context-Free Languages | CFL pumping lemma, closure properties, decidability |

### Part III: Turing Machines and Computability (L09-L11)

| Lesson | Title | Key Concepts |
|--------|-------|--------------|
| 09 | Turing Machines | TM definition, variants, Church-Turing thesis |
| 10 | Decidability | Decidable and recognizable languages, halting problem |
| 11 | Reducibility | Mapping reductions, Rice's theorem, Post correspondence problem |

### Part IV: Complexity Theory and Applications (L12-L14)

| Lesson | Title | Key Concepts |
|--------|-------|--------------|
| 12 | Computational Complexity | Time complexity, P, NP, NP-completeness, Cook-Levin theorem |
| 13 | The Chomsky Hierarchy | Type 0-3 languages, closure summary, decidability summary |
| 14 | Advanced Topics and Applications | Lambda calculus, context-sensitive grammars, applications |

## Relationship to Other Topics

- **Compiler_Design**: Applies DFA/NFA for lexing and CFG for parsing — practical orientation
- **Algorithm**: Complexity classes (P, NP) connect to algorithm design and analysis
- **Math_for_AI**: Formal proof techniques, mathematical reasoning
- **Quantum_Computing**: Quantum complexity classes extend classical complexity theory

## Example Code

Runnable Python examples are available in [`examples/Formal_Languages/`](../../../examples/Formal_Languages/). These implement key algorithms and simulators for hands-on experimentation.

## References

- *Introduction to the Theory of Computation* by Michael Sipser (primary reference)
- *Introduction to Automata Theory, Languages, and Computation* by Hopcroft, Motwani, Ullman
- *Computational Complexity: A Modern Approach* by Arora and Barak
- *Automata and Computability* by Dexter Kozen
