# Formal Languages Examples

This directory contains 8 Python examples demonstrating key concepts in formal languages and automata theory, from finite automata through Turing machines and the Chomsky hierarchy. All examples use only Python's standard library (no external dependencies).

## Files Overview

### 1. `01_dfa_simulator.py` - Deterministic Finite Automata
**Concepts:**
- Formal DFA definition as a 5-tuple (Q, Σ, δ, q₀, F)
- String acceptance via extended transition function
- DFA product construction for intersection/union
- DFA complement (swap accept/reject states)
- DFA minimization via Hopcroft's partition refinement

**Run:** `python 01_dfa_simulator.py`

---

### 2. `02_nfa_subset_construction.py` - NFA and Subset Construction
**Concepts:**
- NFA definition with epsilon-transitions
- Epsilon-closure computation
- NFA simulation (parallel state tracking)
- Subset construction algorithm (NFA → DFA conversion)
- Exponential blowup demonstration
- Closure constructions (union, concatenation, Kleene star)

**Run:** `python 02_nfa_subset_construction.py`

---

### 3. `03_regular_expressions.py` - Regular Expression Engine
**Concepts:**
- Regex parsing into an Abstract Syntax Tree
- Thompson's construction: regex → NFA
- NFA simulation for pattern matching
- Algebraic identity verification
- Linear NFA size guarantee O(n)

**Run:** `python 03_regular_expressions.py`

---

### 4. `04_pumping_lemma.py` - Pumping Lemma Demonstrations
**Concepts:**
- Pumping lemma for regular languages (3-part decomposition)
- Pumping lemma for context-free languages (5-part decomposition)
- Pumping game simulation (adversary vs prover)
- Non-regularity proofs: {a^n b^n}, {a^p | p prime}
- Non-context-freeness proof: {a^n b^n c^n}

**Run:** `python 04_pumping_lemma.py`

---

### 5. `05_cfg_cyk_parser.py` - CFG, CNF, and CYK Parser
**Concepts:**
- Context-free grammar representation and derivation
- Chomsky Normal Form (CNF) conversion
- CYK (Cocke-Younger-Kasami) parsing algorithm O(n³)
- Parse tree reconstruction from CYK table
- Grammar ambiguity demonstration

**Run:** `python 05_cfg_cyk_parser.py`

---

### 6. `06_pushdown_automaton.py` - Pushdown Automata
**Concepts:**
- PDA definition with stack operations (push, pop, replace)
- Nondeterministic PDA simulation via BFS
- Acceptance by final state
- PDA designs: {a^n b^n}, palindromes, balanced parentheses, equal counts
- DPDA vs NPDA: center-marked palindromes

**Run:** `python 06_pushdown_automaton.py`

---

### 7. `07_turing_machine.py` - Turing Machine Simulator
**Concepts:**
- Standard single-tape deterministic Turing machine
- TM computation with tape visualization
- TM designs: {a^n b^n}, {a^n b^n c^n}, palindromes, {a^(2^n)}
- Time complexity analysis (step counting)
- O(n²) typical for single-tape matching problems

**Run:** `python 07_turing_machine.py`

---

### 8. `08_chomsky_hierarchy.py` - Chomsky Hierarchy Overview
**Concepts:**
- All four levels: Regular, Context-Free, Context-Sensitive, RE
- Language membership testing at each level
- Strict containment demonstrations
- Closure property verification (including CF non-closure under ∩)
- Decision problem decidability comparison across levels

**Run:** `python 08_chomsky_hierarchy.py`

---

## Requirements

- Python 3.8 or higher
- Standard library only (no external dependencies)

## Usage

Each file is self-contained and can be run independently:

```bash
# Run a specific example
python 01_dfa_simulator.py

# Or run all examples sequentially
for f in *.py; do echo "=== $f ==="; python "$f"; echo; done
```

## Learning Path

Recommended order:

1. **Finite Automata (Part I):**
   - 01_dfa_simulator.py (DFA basics, product construction, minimization)
   - 02_nfa_subset_construction.py (NFA, epsilon-closure, subset construction)
   - 03_regular_expressions.py (Thompson's construction, regex ↔ NFA)
   - 04_pumping_lemma.py (limitations of regular/CF languages)

2. **Context-Free Languages (Part II):**
   - 05_cfg_cyk_parser.py (grammars, CNF, parsing)
   - 06_pushdown_automaton.py (stack-based computation)

3. **Computability and Hierarchy (Part III-IV):**
   - 07_turing_machine.py (universal computation model)
   - 08_chomsky_hierarchy.py (complete classification)

## Key Takeaways

- **DFA/NFA equivalence**: Nondeterminism adds succinctness but not power for finite automata
- **Regular expressions = finite automata**: Kleene's theorem (Thompson + state elimination)
- **Pumping lemma**: Necessary condition for regularity/context-freeness; used for non-membership proofs
- **PDA = CFG**: Pushdown automata recognize exactly the context-free languages
- **DPDA ⊊ NPDA**: Unlike DFA = NFA, deterministic PDAs are strictly weaker
- **Turing machines**: The most powerful standard computational model; define the limits of computation
- **Chomsky hierarchy**: Regular ⊊ CF ⊊ CS ⊊ RE ⊊ All Languages
- **Decidability decreases with power**: More expressive language classes have fewer decidable properties
