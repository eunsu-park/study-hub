# Logic Gates

**Previous**: [Integer and Floating-Point Representation](./03_Integer_Float_Representation.md) | **Next**: [Combinational Logic Circuits](./05_Combinational_Logic.md)

**Difficulty**: ⭐ (Beginner)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Describe the function of each basic logic gate (AND, OR, NOT, NAND, NOR, XOR, XNOR)
2. Construct truth tables for any combination of gates
3. Prove that NAND and NOR are universal gates by implementing AND, OR, and NOT from each
4. Simplify Boolean expressions using algebraic rules (identity, null, idempotent, absorption, consensus)
5. Apply De Morgan's theorems to transform logic expressions between AND/OR forms

---

Logic gates are the atoms of digital computing. Every operation your CPU performs -- from adding two numbers to rendering a video frame -- is ultimately built from combinations of simple gates that output 0 or 1. Understanding gates is understanding the physical foundation on which all of computing rests.

## Table of Contents

1. [Logic Gate Basics](#1-logic-gate-basics)
2. [Basic Gates (AND, OR, NOT)](#2-basic-gates-and-or-not)
3. [Universal Gates (NAND, NOR)](#3-universal-gates-nand-nor)
4. [XOR and XNOR Gates](#4-xor-and-xnor-gates)
5. [Truth Table Construction](#5-truth-table-construction)
6. [Boolean Algebra Basics](#6-boolean-algebra-basics)
7. [Boolean Algebra Laws](#7-boolean-algebra-laws)
8. [Logic Expression Simplification](#8-logic-expression-simplification)
9. [Practice Problems](#9-practice-problems)

---

## 1. Logic Gate Basics

### Digital Signals

```
┌─────────────────────────────────────────────────────────────┐
│                     Digital Signals                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Analog vs Digital:                                         │
│                                                             │
│  Analog: Continuous values   Digital: Discrete values       │
│                                                             │
│     │ /\    /\               │ ┌──┐  ┌──┐                  │
│     │/  \  /  \              │ │  │  │  │                  │
│  ───┼────\/────\──        ───┼─┘  └──┘  └──                │
│     │                        │                             │
│                                                             │
│  In digital logic:                                          │
│  - HIGH (1, True):  High voltage (e.g., 5V, 3.3V)          │
│  - LOW (0, False):  Low voltage (e.g., 0V)                 │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### What is a Logic Gate?

```
Logic Gate:
Electronic circuit that performs logical operations on input signals to generate output

┌─────────────────────────────────────────────────────────────┐
│                                                             │
│    Input A ─────┐                                           │
│                 │                                           │
│                 ├───[ Logic Gate ]─────→ Output Y           │
│                 │                                           │
│    Input B ─────┘                                           │
│                                                             │
│    Y = f(A, B)  ← Logic function                            │
│                                                             │
└─────────────────────────────────────────────────────────────┘

Physical implementation of gates:
- Transistors (CMOS, TTL, etc.)
- Relays (early computers)
- Vacuum tubes (1st generation computers)
```

### Summary of Logic Gate Types

```
┌─────────────┬────────────┬────────────────────────────────────┐
│    Gate     │   Symbol   │           Description              │
├─────────────┼────────────┼────────────────────────────────────┤
│    AND      │   A·B      │  Output 1 only when both are 1     │
│    OR       │   A+B      │  Output 1 if at least one is 1     │
│    NOT      │   A'       │  Inverts input                     │
├─────────────┼────────────┼────────────────────────────────────┤
│    NAND     │  (A·B)'    │  Inverted AND                      │
│    NOR      │  (A+B)'    │  Inverted OR                       │
├─────────────┼────────────┼────────────────────────────────────┤
│    XOR      │   A⊕B      │  Output 1 when inputs differ       │
│    XNOR     │  (A⊕B)'    │  Output 1 when inputs are equal    │
└─────────────┴────────────┴────────────────────────────────────┘
```

---

## 2. Basic Gates (AND, OR, NOT)

### AND Gate

```
AND Gate: Output is 1 only when all inputs are 1

Circuit Symbol:
              ┌────┐
         A ───┤    │
              │ &  ├───Y   Y = A · B = A AND B
         B ───┤    │
              └────┘

IEEE/ANSI Symbol:
         A ───┬────╮
              │    ├───Y
         B ───┴────╯

Truth Table:
┌───┬───┬───────┐
│ A │ B │ A·B   │
├───┼───┼───────┤
│ 0 │ 0 │   0   │
│ 0 │ 1 │   0   │
│ 1 │ 0 │   0   │
│ 1 │ 1 │   1   │
└───┴───┴───────┘

Real-world analogy:
- Switches connected in series
- Light bulb lights only when both switches are ON

    A       B      Bulb
   ─/ ─────/ ─────◯─
```

### OR Gate

```
OR Gate: Output is 1 if at least one input is 1

Circuit Symbol:
              ┌────╲
         A ───┤     ╲
              │  ≥1  >───Y   Y = A + B = A OR B
         B ───┤     ╱
              └────╱

IEEE/ANSI Symbol:
         A ───╲╲
              ╲╲────Y
         B ───╱╱

Truth Table:
┌───┬───┬───────┐
│ A │ B │ A+B   │
├───┼───┼───────┤
│ 0 │ 0 │   0   │
│ 0 │ 1 │   1   │
│ 1 │ 0 │   1   │
│ 1 │ 1 │   1   │
└───┴───┴───────┘

Real-world analogy:
- Switches connected in parallel
- Light bulb lights when at least one is ON

        ┌──/ ──┐  A
    ────┤      ├────◯──
        └──/ ──┘  B
            Bulb
```

### NOT Gate (Inverter)

```
NOT Gate: Inverts the input

Circuit Symbol:
              ┌────╲
         A ───┤  1  o───Y   Y = A' = Ā = NOT A = ¬A
              └────╱

Truth Table:
┌───┬───────┐
│ A │  A'   │
├───┼───────┤
│ 0 │   1   │
│ 1 │   0   │
└───┴───────┘

Notation:
- A' (prime)
- Ā (overbar)
- NOT A
- ¬A
- !A (programming)
- ~A (bitwise operation)

Real-world analogy:
- Normally closed (NC) switch
- Circuit breaks when button is pressed
```

### Basic Gate Combinations

```
3-input AND Gate:

         A ───┐
              │    ┌────┐
         B ───┼────┤    │
              │    │ &  ├───Y   Y = A · B · C
         C ───┘    │    │
                   └────┘

Truth Table (8 combinations):
┌───┬───┬───┬───────┐
│ A │ B │ C │ A·B·C │
├───┼───┼───┼───────┤
│ 0 │ 0 │ 0 │   0   │
│ 0 │ 0 │ 1 │   0   │
│ 0 │ 1 │ 0 │   0   │
│ 0 │ 1 │ 1 │   0   │
│ 1 │ 0 │ 0 │   0   │
│ 1 │ 0 │ 1 │   0   │
│ 1 │ 1 │ 0 │   0   │
│ 1 │ 1 │ 1 │   1   │
└───┴───┴───┴───────┘

3-input OR Gate:

Y = A + B + C

Output is 1 if at least one is 1
Output is 0 only when all are 0
```

---

## 3. Universal Gates (NAND, NOR)

### NAND Gate

```
NAND Gate: Inverted AND (NOT AND)

Circuit Symbol:
              ┌────╲
         A ───┤     o───Y   Y = (A · B)' = A NAND B
         B ───┤    ╱
              └────╱

Truth Table:
┌───┬───┬──────────┐
│ A │ B │ (A·B)'   │
├───┼───┼──────────┤
│ 0 │ 0 │    1     │
│ 0 │ 1 │    1     │
│ 1 │ 0 │    1     │
│ 1 │ 1 │    0     │
└───┴───┴──────────┘

Comparison with AND:
┌───┬───┬───────┬──────────┐
│ A │ B │ A·B   │ (A·B)'   │
├───┼───┼───────┼──────────┤
│ 0 │ 0 │   0   │    1     │
│ 0 │ 1 │   0   │    1     │
│ 1 │ 0 │   0   │    1     │
│ 1 │ 1 │   1   │    0     │
└───┴───┴───────┴──────────┘
```

### NOR Gate

```
NOR Gate: Inverted OR (NOT OR)

Circuit Symbol:
              ┌────╲
         A ───╲     o───Y   Y = (A + B)' = A NOR B
         B ───╱    ╱
              └────╱

Truth Table:
┌───┬───┬──────────┐
│ A │ B │ (A+B)'   │
├───┼───┼──────────┤
│ 0 │ 0 │    1     │
│ 0 │ 1 │    0     │
│ 1 │ 0 │    0     │
│ 1 │ 1 │    0     │
└───┴───┴──────────┘

Comparison with OR:
┌───┬───┬───────┬──────────┐
│ A │ B │ A+B   │ (A+B)'   │
├───┼───┼───────┼──────────┤
│ 0 │ 0 │   0   │    1     │
│ 0 │ 1 │   1   │    0     │
│ 1 │ 0 │   1   │    0     │
│ 1 │ 1 │   1   │    0     │
└───┴───┴───────┴──────────┘
```

### Meaning of Universal Gates

```
┌─────────────────────────────────────────────────────────────┐
│                    Universal Gate                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  NAND and NOR are "Universal Gates".                        │
│  → All logical operations can be implemented with these     │
│     gates alone.                                            │
│                                                             │
│  Implementing other gates with NAND:                        │
│                                                             │
│  NOT:                                                       │
│       A ───┬───┤ NAND ├───→ A'                              │
│            └───┤      │                                     │
│                                                             │
│  AND:                                                       │
│       A ───┤ NAND ├───┤ NAND ├───→ A·B                      │
│       B ───┤      │───┤      │                              │
│                                                             │
│  OR:                                                        │
│       A ───┤ NAND ├─┐                                       │
│       A ───┤      │ └───┤ NAND ├───→ A+B                    │
│       B ───┤ NAND ├─────┤      │                            │
│       B ───┤      │                                         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Implementing Basic Gates with NAND

```
1. NOT from NAND:
   ┌───────┐
A ─┤       │
   │ NAND  ├─── Y = A'
A ─┤       │
   └───────┘

   (A·A)' = A'


2. AND from NAND:
   ┌───────┐    ┌───────┐
A ─┤       │    │       │
   │ NAND  ├────┤ NAND  ├─── Y = A·B
B ─┤       │    │       │
   └───────┘    └───────┘
                    │
               (same input)

   ((A·B)')' = A·B


3. OR from NAND:
   ┌───────┐
A ─┤       │
   │ NAND  ├─────┐
A ─┤       │     │   ┌───────┐
   └───────┘     ├───┤       │
                 │   │ NAND  ├─── Y = A+B
   ┌───────┐     ├───┤       │
B ─┤       │     │   └───────┘
   │ NAND  ├─────┘
B ─┤       │
   └───────┘

   (A'·B')' = A+B  (De Morgan)
```

---

## 4. XOR and XNOR Gates

### XOR Gate (Exclusive OR)

```
XOR Gate: Output is 1 when inputs differ

Circuit Symbol:
              ┌────╲
         A ───╲ =1  ╲───Y   Y = A ⊕ B = A XOR B
         B ───╱     ╱
              └────╱

IEEE Symbol: =1 or ⊕

Truth Table:
┌───┬───┬───────┐
│ A │ B │ A⊕B   │
├───┼───┼───────┤
│ 0 │ 0 │   0   │ ← Equal
│ 0 │ 1 │   1   │ ← Different
│ 1 │ 0 │   1   │ ← Different
│ 1 │ 1 │   0   │ ← Equal
└───┴───┴───────┘

Equivalent expressions:
A ⊕ B = A'B + AB'  (only one is 1)
      = (A + B)(A' + B')
      = (A + B)(AB)'

XOR properties:
- A ⊕ 0 = A
- A ⊕ 1 = A'
- A ⊕ A = 0
- A ⊕ A' = 1
- A ⊕ B = B ⊕ A (commutative)
- (A ⊕ B) ⊕ C = A ⊕ (B ⊕ C) (associative)
```

### XNOR Gate (Equivalence)

```
XNOR Gate: Output is 1 when inputs are equal

Circuit Symbol:
              ┌────╲
         A ───╲ =1  o───Y   Y = (A ⊕ B)' = A XNOR B
         B ───╱     ╱
              └────╱

Truth Table:
┌───┬───┬─────────┐
│ A │ B │ (A⊕B)'  │
├───┼───┼─────────┤
│ 0 │ 0 │    1    │ ← Equal
│ 0 │ 1 │    0    │ ← Different
│ 1 │ 0 │    0    │ ← Different
│ 1 │ 1 │    1    │ ← Equal
└───┴───┴─────────┘

Equivalent expressions:
(A ⊕ B)' = A'B' + AB  (both equal)
         = A ⊙ B (equivalence symbol)
```

### Applications of XOR

```
┌─────────────────────────────────────────────────────────────┐
│                    Major XOR Applications                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. Adder                                                   │
│     - Sum calculation in half adder                         │
│     - LSB of A + B = A ⊕ B                                  │
│                                                             │
│  2. Parity Check                                            │
│     - XOR of multiple bits = 1 if odd number of 1s         │
│     - Used for error detection                              │
│                                                             │
│  3. Comparator                                              │
│     - A ⊕ B = 0 if A = B                                    │
│     - A ⊕ B = 1 if A ≠ B                                    │
│                                                             │
│  4. Toggle                                                  │
│     - A ⊕ 1 = A' (bit inversion)                            │
│     - A ⊕ 0 = A (maintain)                                  │
│                                                             │
│  5. Encryption                                              │
│     - Message ⊕ Key = Ciphertext                            │
│     - Ciphertext ⊕ Key = Message                            │
│                                                             │
│  6. Swap (without temp variable)                            │
│     - a = a ⊕ b                                             │
│     - b = a ⊕ b                                             │
│     - a = a ⊕ b                                             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Implementing XOR with Basic Gates

```
XOR = A'B + AB'

Circuit implementation:
                    ┌─────┐
         A ────┬────┤ NOT ├────┐
               │    └─────┘    │    ┌─────┐
               │               └────┤     │
               │                    │ AND ├────┐
               │    ┌──────────────┤     │    │    ┌────┐
               │    │              └─────┘    ├────┤    │
         B ────┼────┼──────────┐              │    │ OR ├───Y
               │    │          │              │    │    │
               │    │    ┌─────┐    ┌─────┐   │    └────┘
               │    │    │ NOT ├────┤     │   │
               │    └────┤     │    │ AND ├───┘
               └─────────┴─────┘    │     │
                                    └─────┘

Gate count:
- 2 NOT + 2 AND + 1 OR = 5 gates

Implementing XOR with NAND only:
- Requires 4 NAND gates
```

---

## 5. Truth Table Construction

### Truth Table Basics

```
┌─────────────────────────────────────────────────────────────┐
│                        Truth Table                           │
├─────────────────────────────────────────────────────────────┤
│  Lists all possible input combinations and corresponding     │
│  outputs for a logic function                                │
│                                                             │
│  n inputs → 2ⁿ rows                                         │
│  - 1 input: 2 rows                                          │
│  - 2 inputs: 4 rows                                         │
│  - 3 inputs: 8 rows                                         │
│  - 4 inputs: 16 rows                                        │
└─────────────────────────────────────────────────────────────┘

Input listing rule (counting order):
┌───┬───┬───┐
│ A │ B │ C │
├───┼───┼───┤
│ 0 │ 0 │ 0 │  ← 0
│ 0 │ 0 │ 1 │  ← 1
│ 0 │ 1 │ 0 │  ← 2
│ 0 │ 1 │ 1 │  ← 3
│ 1 │ 0 │ 0 │  ← 4
│ 1 │ 0 │ 1 │  ← 5
│ 1 │ 1 │ 0 │  ← 6
│ 1 │ 1 │ 1 │  ← 7
└───┴───┴───┘
```

### Truth Tables for Complex Logic Expressions

```
Example: Y = AB + A'C

Add intermediate columns for step-by-step calculation:

┌───┬───┬───┬─────┬──────┬─────┬───────────┐
│ A │ B │ C │ A'  │  AB  │ A'C │ Y=AB+A'C  │
├───┼───┼───┼─────┼──────┼─────┼───────────┤
│ 0 │ 0 │ 0 │  1  │  0   │  0  │     0     │
│ 0 │ 0 │ 1 │  1  │  0   │  1  │     1     │
│ 0 │ 1 │ 0 │  1  │  0   │  0  │     0     │
│ 0 │ 1 │ 1 │  1  │  0   │  1  │     1     │
│ 1 │ 0 │ 0 │  0  │  0   │  0  │     0     │
│ 1 │ 0 │ 1 │  0  │  0   │  0  │     0     │
│ 1 │ 1 │ 0 │  0  │  1   │  0  │     1     │
│ 1 │ 1 │ 1 │  0  │  1   │  0  │     1     │
└───┴───┴───┴─────┴──────┴─────┴───────────┘
```

### Deriving Logic Expressions from Truth Tables

```
Deriving logic expressions from a given truth table:

┌───┬───┬───┐
│ A │ B │ Y │
├───┼───┼───┤
│ 0 │ 0 │ 1 │  ← Row 0: A'B'
│ 0 │ 1 │ 0 │
│ 1 │ 0 │ 1 │  ← Row 2: AB'
│ 1 │ 1 │ 1 │  ← Row 3: AB
└───┴───┴───┘

Method 1: Sum of Products (SOP)
- Select only rows where output is 1
- Express each row as AND term (add NOT for 0s)
- Connect with OR

Y = A'B' + AB' + AB

Method 2: Product of Sums (POS)
- Select only rows where output is 0
- Express each row as OR term (add NOT for 1s)
- Connect with AND

Y = (A + B')  ← from row 0,1
```

---

## 6. Boolean Algebra Basics

### What is Boolean Algebra?

```
┌─────────────────────────────────────────────────────────────┐
│                      Boolean Algebra                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Algebraic system developed by George Boole in 1847         │
│  Deals with two values (0, 1) and logical operations        │
│                                                             │
│  Basic operations:                                          │
│  - AND (conjunction): · or omitted                          │
│  - OR (disjunction): +                                      │
│  - NOT (complement): ' or ̄ (overbar)                       │
│                                                             │
│  Operation precedence:                                      │
│  1. Parentheses ()                                          │
│  2. NOT (')                                                 │
│  3. AND (·)                                                 │
│  4. OR (+)                                                  │
│                                                             │
│  Example: A + B · C' = A + (B · (C'))                       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Basic Axioms

```
Boolean Algebra Axioms:

1. Closure:
   - A · B ∈ {0, 1}
   - A + B ∈ {0, 1}

2. Identity:
   - A · 1 = A
   - A + 0 = A

3. Commutative:
   - A · B = B · A
   - A + B = B + A

4. Distributive:
   - A · (B + C) = A·B + A·C
   - A + (B · C) = (A+B) · (A+C)  ← Different from regular algebra!

5. Complement:
   - A · A' = 0
   - A + A' = 1
```

---

## 7. Boolean Algebra Laws

### Basic Theorems

```
┌─────────────────────────────────────────────────────────────┐
│                      Basic Theorems                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Idempotent:                                                │
│  - A · A = A                                                │
│  - A + A = A                                                │
│                                                             │
│  Null/Domination:                                           │
│  - A · 0 = 0                                                │
│  - A + 1 = 1                                                │
│                                                             │
│  Involution (Double Negation):                              │
│  - (A')' = A                                                │
│                                                             │
│  Absorption:                                                │
│  - A + A·B = A                                              │
│  - A · (A + B) = A                                          │
│                                                             │
│  Associative:                                               │
│  - (A · B) · C = A · (B · C)                                │
│  - (A + B) + C = A + (B + C)                                │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### De Morgan's Laws

```
┌─────────────────────────────────────────────────────────────┐
│                    De Morgan's Laws                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Theorem 1: (A · B)' = A' + B'                              │
│  "NOT of AND = OR of NOTs"                                  │
│                                                             │
│  Theorem 2: (A + B)' = A' · B'                              │
│  "NOT of OR = AND of NOTs"                                  │
│                                                             │
│  Generalization:                                            │
│  (A₁ · A₂ · ... · Aₙ)' = A₁' + A₂' + ... + Aₙ'             │
│  (A₁ + A₂ + ... + Aₙ)' = A₁' · A₂' · ... · Aₙ'             │
│                                                             │
│  Application: NAND/NOR conversion                           │
│  (AB)' = A' + B'  → NAND to NOR with inverters             │
│  (A+B)' = A'B'    → NOR to NAND with inverters             │
│                                                             │
└─────────────────────────────────────────────────────────────┘

Proof of De Morgan's Laws (truth table):

(A · B)' = A' + B'

┌───┬───┬─────┬────────┬─────┬─────┬──────────┐
│ A │ B │ A·B │ (A·B)' │ A'  │ B'  │ A' + B'  │
├───┼───┼─────┼────────┼─────┼─────┼──────────┤
│ 0 │ 0 │  0  │   1    │  1  │  1  │    1     │
│ 0 │ 1 │  0  │   1    │  1  │  0  │    1     │
│ 1 │ 0 │  0  │   1    │  0  │  1  │    1     │
│ 1 │ 1 │  1  │   0    │  0  │  0  │    0     │
└───┴───┴─────┴────────┴─────┴─────┴──────────┘
         Equal!
```

### Consensus Theorem

```
Consensus:

A·B + A'·C + B·C = A·B + A'·C

"Remove redundant term"

Proof:
B·C = B·C·(A + A')           (A + A' = 1)
    = A·B·C + A'·B·C
    = A·B·C + A'·B·C         (already included in AB and A'C)

Dual form:
(A + B)·(A' + C)·(B + C) = (A + B)·(A' + C)
```

### Boolean Algebra Laws Summary Table

```
┌──────────────────────┬─────────────────────┬─────────────────────┐
│        Law           │        AND          │        OR           │
├──────────────────────┼─────────────────────┼─────────────────────┤
│ Identity             │ A · 1 = A           │ A + 0 = A           │
├──────────────────────┼─────────────────────┼─────────────────────┤
│ Null                 │ A · 0 = 0           │ A + 1 = 1           │
├──────────────────────┼─────────────────────┼─────────────────────┤
│ Idempotent           │ A · A = A           │ A + A = A           │
├──────────────────────┼─────────────────────┼─────────────────────┤
│ Complement           │ A · A' = 0          │ A + A' = 1          │
├──────────────────────┼─────────────────────┼─────────────────────┤
│ Commutative          │ A · B = B · A       │ A + B = B + A       │
├──────────────────────┼─────────────────────┼─────────────────────┤
│ Associative          │ (AB)C = A(BC)       │ (A+B)+C = A+(B+C)   │
├──────────────────────┼─────────────────────┼─────────────────────┤
│ Distributive         │ A(B+C) = AB + AC    │ A+BC = (A+B)(A+C)   │
├──────────────────────┼─────────────────────┼─────────────────────┤
│ Absorption           │ A(A+B) = A          │ A + AB = A          │
├──────────────────────┼─────────────────────┼─────────────────────┤
│ De Morgan            │ (AB)' = A' + B'     │ (A+B)' = A'B'       │
└──────────────────────┴─────────────────────┴─────────────────────┘
```

---

## 8. Logic Expression Simplification

### Algebraic Simplification

```
Example 1: Y = AB + AB'

Y = AB + AB'
  = A(B + B')    (reverse distributive)
  = A · 1        (B + B' = 1)
  = A            (identity)


Example 2: Y = A'B'C + A'BC + AB'C + ABC

Y = A'B'C + A'BC + AB'C + ABC
  = A'C(B' + B) + AC(B' + B)    (reverse distributive)
  = A'C · 1 + AC · 1             (B' + B = 1)
  = A'C + AC
  = C(A' + A)                    (reverse distributive)
  = C · 1
  = C


Example 3: Y = AB + A'C + BC

Y = AB + A'C + BC
  = AB + A'C + BC(A + A')        (A + A' = 1)
  = AB + A'C + ABC + A'BC
  = AB(1 + C) + A'C(1 + B)       (1 + X = 1)
  = AB + A'C                      (consensus theorem)
```

### Karnaugh Map

```
┌─────────────────────────────────────────────────────────────┐
│                    Karnaugh Map (K-Map)                      │
├─────────────────────────────────────────────────────────────┤
│  Method to visually simplify by arranging truth table       │
│  in 2D format. Group adjacent 1s to eliminate variables.    │
└─────────────────────────────────────────────────────────────┘

2-variable K-Map:
        B
      0   1
   ┌─────┬─────┐
 0 │ A'B'│ A'B │
A  ├─────┼─────┤
 1 │ AB' │ AB  │
   └─────┴─────┘

3-variable K-Map (Gray code order):
         BC
       00  01  11  10
    ┌────┬────┬────┬────┐
  0 │ 0  │ 1  │ 3  │ 2  │
A   ├────┼────┼────┼────┤
  1 │ 4  │ 5  │ 7  │ 6  │
    └────┴────┴────┴────┘
           (minterm numbers)

4-variable K-Map:
          CD
        00  01  11  10
     ┌────┬────┬────┬────┐
  00 │ 0  │ 1  │ 3  │ 2  │
     ├────┼────┼────┼────┤
  01 │ 4  │ 5  │ 7  │ 6  │
AB   ├────┼────┼────┼────┤
  11 │ 12 │ 13 │ 15 │ 14 │
     ├────┼────┼────┼────┤
  10 │ 8  │ 9  │ 11 │ 10 │
     └────┴────┴────┴────┘
```

### K-Map Usage

```
Example: Y = Σm(0, 1, 3, 5, 7)  (3 variables)

Step 1: Mark 1s on K-Map
         BC
       00  01  11  10
    ┌────┬────┬────┬────┐
  0 │ 1  │ 1  │ 1  │    │   ← m0, m1, m3
A   ├────┼────┼────┼────┤
  1 │    │ 1  │ 1  │    │   ← m5, m7
    └────┴────┴────┴────┘

Step 2: Group adjacent 1s (power-of-2 sizes)
         BC
       00  01  11  10
    ┌────┬────┬────┬────┐
  0 │[1] │ 1──┼─1  │    │   Group1: m1,m3,m5,m7 (vertical 2 cells)
A   ├────┼────┼────┼────┤
  1 │    │ 1──┼─1  │    │
    └────┴────┴────┴────┘
                │
         Group2: m0,m1 (horizontal 2 cells)

Step 3: Extract common variables for each group
- Group1 (4 cells): Only C is common → C
- Group2 (2 cells): A' and B' are common → A'B'

Result: Y = C + A'B'

Verification:
Original: A'B'C' + A'B'C + A'BC + AB'C + ABC
         = A'B'(C'+C) + C(A'B + AB' + AB)
         = A'B' + C(A'B + A) = A'B' + C
```

### K-Map Grouping Rules

```
┌─────────────────────────────────────────────────────────────┐
│                    K-Map Grouping Rules                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. Group size must be power of 2 (1, 2, 4, 8, 16...)      │
│                                                             │
│  2. Only adjacent cells can be grouped (no diagonal)        │
│     - K-Map edges wrap around (torus shape)                │
│                                                             │
│  3. All 1s must be included in at least one group          │
│                                                             │
│  4. Groups should be as large as possible                   │
│     (larger group = fewer variables)                        │
│                                                             │
│  5. Same 1 can be included in multiple groups if needed    │
│                                                             │
│  6. Don't Care (X) can be treated as 1 when needed         │
│                                                             │
└─────────────────────────────────────────────────────────────┘

Edge wrapping example (4 variables):

          CD
        00  01  11  10
     ┌────┬────┬────┬────┐
  00 │ 1  │    │    │ 1  │  ← left-right wrap
     ├────┼────┼────┼────┤
  01 │    │    │    │    │
AB   ├────┼────┼────┼────┤
  11 │    │    │    │    │
     ├────┼────┼────┼────┤
  10 │ 1  │    │    │ 1  │  ← top-bottom wrap
     └────┴────┴────┴────┘
       ↑              ↑
       └──────────────┘
         left-right wrap

4 ones form one group (four corners)
→ B'D' (common variables)
```

---

## 9. Practice Problems

### Basic Problems

**1. Find the result of the following logical operations.**
   - (a) 1 AND 0
   - (b) 1 OR 0
   - (c) NOT 1
   - (d) 1 NAND 1
   - (e) 0 NOR 0
   - (f) 1 XOR 1

**2. Create truth tables for the following logic expressions.**
   - (a) Y = A + B'
   - (b) Y = AB + A'B'
   - (c) Y = (A + B)(A' + C)

**3. Derive the logic expression in SOP form for the following truth table.**

```
┌───┬───┬───┐
│ A │ B │ Y │
├───┼───┼───┤
│ 0 │ 0 │ 0 │
│ 0 │ 1 │ 1 │
│ 1 │ 0 │ 1 │
│ 1 │ 1 │ 0 │
└───┴───┴───┘
```

### Boolean Algebra Problems

**4. Simplify the following logic expressions.**
   - (a) Y = A + A'B
   - (b) Y = AB + AB'
   - (c) Y = (A + B)(A + B')
   - (d) Y = A'B + AB' + AB + A'B'

**5. Use De Morgan's laws to transform the following.**
   - (a) (ABC)' = ?
   - (b) (A + B + C)' = ?
   - (c) ((A + B)C)' = ?

**6. Prove the following identities using Boolean algebra.**
   - (a) A + AB = A
   - (b) A(A + B) = A
   - (c) A + A'B = A + B

### K-Map Problems

**7. Simplify the following logic expressions using K-Maps.**
   - (a) Y = Σm(0, 2, 4, 6)  (3 variables)
   - (b) Y = Σm(0, 1, 2, 3, 5, 7)  (3 variables)
   - (c) Y = Σm(0, 1, 2, 5, 8, 9, 10)  (4 variables)

**8. Simplify the following function with Don't Care conditions.**
   Y = Σm(1, 3, 7) + d(0, 5)  (3 variables)

### Advanced Problems

**9. Implement the following using only NAND gates.**
   - (a) NOT gate
   - (b) AND gate
   - (c) OR gate
   - (d) XOR gate

**10. Express the output of the following circuit as a logic expression and simplify.**

```
        ┌─────┐
A ──────┤     │
        │ AND ├─────┐
B ──────┤     │     │    ┌─────┐
        └─────┘     ├────┤     │
                    │    │ OR  ├──── Y
        ┌─────┐     ├────┤     │
A ──┬───┤ NOT ├─────┘    └─────┘
    │   └─────┘
    │   ┌─────┐
    └───┤     │
        │ AND ├─────────────────┘
C ──────┤     │
        └─────┘
```

---

<details>
<summary>Answers</summary>

**1. Logical operation results**
- (a) 1 AND 0 = 0
- (b) 1 OR 0 = 1
- (c) NOT 1 = 0
- (d) 1 NAND 1 = 0
- (e) 0 NOR 0 = 1
- (f) 1 XOR 1 = 0

**2. Truth tables**
- (a) Y = A + B': [1,0,1,1] (in order)
- (b) Y = AB + A'B': [1,0,0,1]
- (c) Y = (A+B)(A'+C): [0,C,B,1] i.e. [0,0,0,1,0,1,0,1]

**3.** Y = A'B + AB' = A XOR B

**4. Simplification**
- (a) Y = A + A'B = A + B (absorption)
- (b) Y = AB + AB' = A(B + B') = A
- (c) Y = (A+B)(A+B') = A + BB' = A
- (d) Y = A'B + AB' + AB + A'B' = A'(B+B') + A(B'+B) = A' + A = 1

**5. De Morgan**
- (a) (ABC)' = A' + B' + C'
- (b) (A+B+C)' = A'B'C'
- (c) ((A+B)C)' = (A+B)' + C' = A'B' + C'

**6. Proof**
- (a) A + AB = A(1 + B) = A·1 = A
- (b) A(A+B) = AA + AB = A + AB = A
- (c) A + A'B = (A+A')(A+B) = 1·(A+B) = A+B

**7. K-Map simplification**
- (a) Y = B' (m0,m2,m4,m6 are B=0 column)
- (b) Y = A' + C (m0,1,2,3=A', m1,3,5,7=C)
- (c) Y = B'D' + A'B'C' + A'CD'

**8.** With Don't Care: Y = B' or Y = A' + B' (depending on d usage)

**9. NAND implementation**
- (a) NOT: A NAND A = A'
- (b) AND: (A NAND B) NAND (A NAND B)
- (c) OR: (A NAND A) NAND (B NAND B)
- (d) XOR: Requires 4 NANDs

**10.** Y = AB + A'C = AND of A and B OR AND of NOT A and C, simplification result remains AB + A'C (cannot be simplified further)

</details>

---

## Next Steps

- [05_Combinational_Logic.md](./05_Combinational_Logic.md) - Adders, multiplexers, decoders

---

## References

- Digital Design (Morris Mano)
- Computer Organization and Design (Patterson & Hennessy)
- [Logic Gate Simulator](https://logic.ly/)
- [Boolean Algebra Calculator](https://www.dcode.fr/boolean-algebra)
- [Karnaugh Map Solver](https://www.charlie-coleman.com/experiments/kmap/)
