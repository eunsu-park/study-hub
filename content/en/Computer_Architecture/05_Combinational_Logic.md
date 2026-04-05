# Combinational Logic Circuits

**Previous**: [Logic Gates](./04_Logic_Gates.md) | **Next**: [Sequential Logic Circuits](./06_Sequential_Logic.md)

**Difficulty**: ⭐⭐ (Intermediate)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Design half-adder and full-adder circuits from logic gates
2. Explain ripple-carry addition and its propagation delay limitation
3. Implement a multiplexer and explain its role in data selection
4. Design a decoder and explain its use in memory addressing
5. Describe how an ALU combines arithmetic and logic operations using control signals
6. Compare encoders and decoders and identify their applications in computer hardware

---

Individual logic gates can only answer yes/no questions. Combinational circuits combine gates into components that can add numbers, select data paths, and decode addresses -- the building blocks of every processor. The ALU you will design in this lesson is the "calculator" at the heart of every CPU.

## Table of Contents

1. [Characteristics of Combinational Logic Circuits](#1-characteristics-of-combinational-logic-circuits)
2. [Half Adder](#2-half-adder)
3. [Full Adder](#3-full-adder)
4. [Ripple Carry Adder](#4-ripple-carry-adder)
5. [Multiplexer (MUX)](#5-multiplexer-mux)
6. [Demultiplexer (DEMUX)](#6-demultiplexer-demux)
7. [Decoder](#7-decoder)
8. [Encoder](#8-encoder)
9. [Comparator and Other Circuits](#9-comparator-and-other-circuits)
10. [Practice Problems](#10-practice-problems)

---

## 1. Characteristics of Combinational Logic Circuits

### Combinational Circuits vs Sequential Circuits

```
┌─────────────────────────────────────────────────────────────┐
│                Classification of Digital Circuits            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────────┐    ┌─────────────────────┐        │
│  │  Combinational      │    │    Sequential        │        │
│  │    Logic Circuit    │    │    Logic Circuit     │        │
│  ├─────────────────────┤    ├─────────────────────┤        │
│  │ - No memory         │    │ - Has memory         │        │
│  │ - Output=f(inputs)  │    │ - Output=f(in,state) │        │
│  │ - No feedback       │    │ - Has feedback       │        │
│  │                     │    │                     │        │
│  │ Ex: Adder, MUX,     │    │ Ex: Flip-flop,      │        │
│  │     Decoder, Encoder│    │     Register, Counter│        │
│  └─────────────────────┘    └─────────────────────┘        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Characteristics of Combinational Circuits

```
Characteristics of Combinational Logic Circuits:

1. Outputs depend only on current inputs
   ┌────────────────┐
   │                │
Input →│ Combinational│→ Output
   │    Circuit     │
   └────────────────┘
   Y = f(X₁, X₂, ..., Xₙ)

2. No memory elements
   - Does not store previous inputs or states

3. Only propagation delay exists
   - Time from input change → output change
   - Sum of gate delays

4. No feedback path
   - Output does not connect back to input
```

### Combinational Circuit Design Procedure

```
┌─────────────────────────────────────────────────────────────┐
│            Combinational Circuit Design Procedure            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. Problem Definition                                      │
│     - Clearly define inputs and outputs                     │
│     - Understand operating conditions                       │
│                                                             │
│  2. Truth Table Construction                                │
│     - Determine outputs for all input combinations          │
│     - Identify Don't Care conditions                        │
│                                                             │
│  3. Logic Expression Derivation                             │
│     - SOP (Sum of Products) or POS (Product of Sums) form   │
│                                                             │
│  4. Logic Expression Simplification                         │
│     - Use Boolean algebra or K-maps                         │
│                                                             │
│  5. Circuit Implementation                                  │
│     - Design circuit with logic gates                       │
│     - Convert to NAND/NOR gates if needed                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Half Adder

### Half Adder Concept

```
Half Adder:
Adds two 1-bit inputs to output Sum and Carry

┌─────────────────────────────────────────────────────────────┐
│                                                             │
│       A ─────┐     ┌─────────────┐                         │
│              ├─────┤     HA      ├───── S (Sum)            │
│       B ─────┘     │             ├───── C (Carry)          │
│                    └─────────────┘                         │
│                                                             │
│  Binary addition:  A + B = CS                              │
│                    0 + 0 = 00  (0)                          │
│                    0 + 1 = 01  (1)                          │
│                    1 + 0 = 01  (1)                          │
│                    1 + 1 = 10  (2)                          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Truth Table and Logic Expression

```
Half Adder Truth Table:

┌───┬───┬───────┬─────────┐
│ A │ B │ S(Sum)│ C(Carry)│
├───┼───┼───────┼─────────┤
│ 0 │ 0 │   0   │    0    │
│ 0 │ 1 │   1   │    0    │
│ 1 │ 0 │   1   │    0    │
│ 1 │ 1 │   0   │    1    │
└───┴───┴───────┴─────────┘

Logic expressions:
S = A ⊕ B  (XOR)
C = A · B  (AND)
```

### Circuit Implementation

```
Half Adder Circuit:

         A ───┬─────┬───────────────────┐
              │     │                   │
              │     │     ┌──────┐      │
              │     └─────┤ XOR  ├──────┼─── S
              │     ┌─────┤      │      │
         B ───┼─────┼─────┴──────┘      │
              │     │                   │
              │     │     ┌──────┐      │
              │     └─────┤ AND  ├──────┼─── C
              └───────────┤      │      │
                          └──────┘

Block Symbol:
        ┌───────┐
   A ───┤       ├─── S (Sum)
        │  HA   │
   B ───┤       ├─── C (Carry)
        └───────┘

Gate count: 1 XOR + 1 AND = 2 gates
```

### Implementation with NAND Gates

```
Half Adder using only NAND gates:

XOR with NAND: 4 gates needed
AND with NAND: 2 gates needed

But with circuit optimization:

        ┌───────┐
   A ───┤ NAND  ├───┬───────────────────┐
   B ───┤   1   │   │                   │
        └───────┘   │   ┌───────┐       │   ┌───────┐
                    ├───┤ NAND  ├───────┼───┤ NAND  ├─── S
   A ───┬───────────┘   │   3   │       │   │   5   │
        │   ┌───────┐   └───────┘       │   └───────┘
        └───┤ NAND  ├───────────────────┘
   B ───────┤   2   │
            └───────┘   ┌───────┐
                    ┌───┤ NAND  ├─── C
                    │   │   4   │
   (NAND1 output) ──┴───┤       │
                        └───────┘

Total: 5 NAND gates
```

---

## 3. Full Adder

### Full Adder Concept

```
Full Adder:
Adds three 1-bit inputs (A, B, Cᵢₙ) to output Sum and Carry

┌─────────────────────────────────────────────────────────────┐
│                                                             │
│       A ─────┐                                              │
│              │     ┌─────────────┐                         │
│       B ─────┼─────┤     FA      ├───── S (Sum)            │
│              │     │             ├───── Cₒᵤₜ (Carry Out)   │
│      Cᵢₙ ────┘     └─────────────┘                         │
│                                                             │
│  Binary addition:  A + B + Cᵢₙ = CₒᵤₜS                     │
│                                                             │
│  Cᵢₙ = Carry from lower bit                                │
│  Cₒᵤₜ = Carry to higher bit                                │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Truth Table and Logic Expression

```
Full Adder Truth Table:

┌───┬───┬─────┬───────┬──────────┐
│ A │ B │ Cᵢₙ │   S   │   Cₒᵤₜ   │
├───┼───┼─────┼───────┼──────────┤
│ 0 │ 0 │  0  │   0   │    0     │
│ 0 │ 0 │  1  │   1   │    0     │
│ 0 │ 1 │  0  │   1   │    0     │
│ 0 │ 1 │  1  │   0   │    1     │
│ 1 │ 0 │  0  │   1   │    0     │
│ 1 │ 0 │  1  │   0   │    1     │
│ 1 │ 1 │  0  │   0   │    1     │
│ 1 │ 1 │  1  │   1   │    1     │
└───┴───┴─────┴───────┴──────────┘

Logic expressions:
S = A ⊕ B ⊕ Cᵢₙ
Cₒᵤₜ = AB + BCᵢₙ + ACᵢₙ = AB + Cᵢₙ(A ⊕ B)

Explanation:
- S: Output 1 if odd number of 1s (successive XOR)
- Cₒᵤₜ: Output 1 if two or more inputs are 1
```

### Implementation with Two Half Adders

```
Full Adder = 2 Half Adders + OR gate

              ┌───────┐        ┌───────┐
    A ────────┤       ├────────┤       ├─────── S
              │  HA1  │        │  HA2  │
    B ────────┤       ├──┬─────┤       ├─────┐
              └───────┘  │     └───────┘     │
                 C1      │        C2         │
                 │       │        │          │
   Cᵢₙ ──────────┼───────┘        │          │
                 │                │    ┌─────┴────┐
                 └────────────────┴────┤   OR     ├─── Cₒᵤₜ
                                       └──────────┘

Operation:
1. HA1: A ⊕ B = P (partial sum), A·B = G (generate)
2. HA2: P ⊕ Cᵢₙ = S (final sum), P·Cᵢₙ = propagate carry
3. Cₒᵤₜ = G + P·Cᵢₙ = AB + (A⊕B)·Cᵢₙ
```

### Circuit Implementation

```
Full Adder Detailed Circuit:

                           ┌───────┐
    A ────────┬────────────┤ XOR   ├──────┬──────────────┐
              │            │       │      │              │
    B ────────┼───┬────────┤       │      │  ┌───────┐   │
              │   │        └───────┘      └──┤ XOR   ├───┼── S
              │   │           (P)         ┌──┤       │   │
   Cᵢₙ ───────┼───┼───────────────────────┘  └───────┘   │
              │   │                                      │
              │   │        ┌───────┐                     │
              │   └────────┤ AND   ├──┐                  │
              │   ┌────────┤       │  │   ┌───────┐      │
              │   │        └───────┘  └───┤       │      │
              │   │           (G)         │  OR   ├──────┴── Cₒᵤₜ
              │   │        ┌───────┐  ┌───┤       │
              │   └────────┤ AND   ├──┘   └───────┘
              │            │       │
   (P) ───────┴────────────┤       │
                           └───────┘

Block Symbol:
           ┌───────┐
    A  ────┤       ├──── S
    B  ────┤  FA   │
   Cᵢₙ ────┤       ├──── Cₒᵤₜ
           └───────┘
```

---

## 4. Ripple Carry Adder

### Ripple Carry Adder Concept

```
Ripple Carry Adder:
Multi-bit addition by connecting full adders in series

4-bit Ripple Carry Adder:

   A₃ B₃    A₂ B₂    A₁ B₁    A₀ B₀
    │ │      │ │      │ │      │ │
    ▼ ▼      ▼ ▼      ▼ ▼      ▼ ▼
  ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐
  │ FA₃ │←─┤ FA₂ │←─┤ FA₁ │←─┤ FA₀ │←─ Cᵢₙ (0)
  └──┬──┘  └──┬──┘  └──┬──┘  └──┬──┘
     │        │        │        │
     ▼        ▼        ▼        ▼
   Cₒᵤₜ      S₃       S₂       S₁       S₀

Result: Cₒᵤₜ S₃ S₂ S₁ S₀ = A₃A₂A₁A₀ + B₃B₂B₁B₀
```

### 8-bit Adder

```
8-bit Ripple Carry Adder:

A[7:0] ───┬──────────────────────────────────────────────┐
          │                                              │
B[7:0] ───┼──────────────────────────────────────────────┼──┐
          │                                              │  │
          ▼                                              ▼  ▼
        ┌───┐   ┌───┐   ┌───┐   ┌───┐   ┌───┐   ┌───┐ ┌───┐┌───┐
 0 ─────┤FA0├───┤FA1├───┤FA2├───┤FA3├───┤FA4├───┤FA5├─┤FA6├┤FA7├─── Cₒᵤₜ
        └─┬─┘   └─┬─┘   └─┬─┘   └─┬─┘   └─┬─┘   └─┬─┘ └─┬─┘└─┬─┘
          │       │       │       │       │       │     │    │
          ▼       ▼       ▼       ▼       ▼       ▼     ▼    ▼
         S₀      S₁      S₂      S₃      S₄      S₅    S₆   S₇

Carry propagation delay:
- Worst case: Carry propagates from LSB to MSB
- Delay time = n × (FA propagation delay)
```

### Limitations of Ripple Carry Adder

```
┌─────────────────────────────────────────────────────────────┐
│        Ripple Carry Adder Advantages and Disadvantages       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Advantages:                                                │
│  - Simple design                                            │
│  - Low gate count (n FAs for n-bit)                         │
│  - Regular structure                                        │
│                                                             │
│  Disadvantages:                                             │
│  - Carry propagation delay increases linearly               │
│  - Very slow for 32-bit, 64-bit                             │
│                                                             │
│  Delay time analysis:                                       │
│  - Carry delay of 1 FA: approximately 2 gate delays         │
│  - Total delay for n-bit adder: approximately 2n gate delays│
│                                                             │
│  Improved adders:                                           │
│  - Carry Lookahead Adder                                    │
│  - Carry Select Adder                                       │
│  - Carry Save Adder                                         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Subtractor

```
Subtraction using adder:

A - B = A + (-B) = A + (2's complement of B) = A + B' + 1

4-bit Adder/Subtractor:

         Sub(0=Add, 1=Subtract)
              │
   B₃ ───────⊕───┐        B₂ ───────⊕───┐
              │  │                   │  │
   A₃ ────┐   │  │     A₂ ────┐     │  │
          │   │  │            │     │  │
          ▼   ▼  │            ▼     ▼  │
        ┌─────┐  │          ┌─────┐   │
  ←─────┤ FA₃ │←─┼──────────┤ FA₂ │←──┼─ ...
        └──┬──┘  │          └──┬──┘   │
           │                   │
          S₃                  S₂

Sub = 0: B unchanged, Cᵢₙ = 0 → A + B
Sub = 1: B inverted, Cᵢₙ = 1 → A + B' + 1 = A - B
```

---

## 5. Multiplexer (MUX)

### Multiplexer Concept

```
Multiplexer (MUX):
Data selector that selects one of multiple inputs to output

┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  2ⁿ inputs → 1 output                                       │
│  n select lines                                             │
│                                                             │
│       D₀ ───┐                                               │
│       D₁ ───┼───┐     ┌─────────┐                          │
│       D₂ ───┼───┼─────┤         │                          │
│       D₃ ───┼───┼─────┤   MUX   ├───── Y                   │
│         ...  │   │     │  n:1    │                          │
│      D₂ⁿ⁻₁──┼───┼─────┤         │                          │
│             │   │     └────┬────┘                          │
│             │   │          │                               │
│             │   │     S₀ S₁...Sₙ₋₁                         │
│             │   │          │                               │
│             │   │     (Select lines)                        │
│                                                             │
│  Y = D_S (S-th input goes to output)                        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 2:1 MUX

```
2:1 Multiplexer:

        ┌─────────┐
   D₀ ──┤ 0       │
        │   MUX   ├─── Y
   D₁ ──┤ 1       │
        └────┬────┘
             │
             S

Truth Table:
┌───┬──────┐
│ S │  Y   │
├───┼──────┤
│ 0 │  D₀  │
│ 1 │  D₁  │
└───┴──────┘

Logic expression:
Y = S'·D₀ + S·D₁

Circuit:
                    ┌───────┐
   D₀ ──────────────┤       │
                    │  AND  ├────┐
   S ───┬───[NOT]───┤       │    │    ┌───────┐
        │           └───────┘    ├────┤  OR   ├─── Y
        │           ┌───────┐    │    │       │
        │           │       │    │    └───────┘
   D₁ ──┼───────────┤  AND  ├────┘
        │           │       │
        └───────────┤       │
                    └───────┘
```

### 4:1 MUX

```
4:1 Multiplexer:

           ┌─────────┐
   D₀  ────┤ 00      │
   D₁  ────┤ 01      │
           │   MUX   ├─── Y
   D₂  ────┤ 10      │
   D₃  ────┤ 11      │
           └────┬────┘
                │
             S₁  S₀

Truth Table:
┌────┬────┬──────┐
│ S₁ │ S₀ │  Y   │
├────┼────┼──────┤
│ 0  │ 0  │  D₀  │
│ 0  │ 1  │  D₁  │
│ 1  │ 0  │  D₂  │
│ 1  │ 1  │  D₃  │
└────┴────┴──────┘

Logic expression:
Y = S₁'S₀'D₀ + S₁'S₀D₁ + S₁S₀'D₂ + S₁S₀D₃
```

### 4:1 MUX Circuit

```
4:1 Multiplexer Circuit:

   D₀ ────┐     ┌─────┐
          └─────┤ AND ├────┐
   S₁'────┬─────┤     │    │
   S₀'────┼─────┤     │    │
          │     └─────┘    │
   D₁ ────│     ┌─────┐    │
          └─────┤ AND ├────┼────┐
   S₁'────┬─────┤     │    │    │
   S₀ ────┼─────┤     │    │    │    ┌─────┐
          │     └─────┘    │    ├────┤     │
   D₂ ────│     ┌─────┐    │    │    │ OR  ├─── Y
          └─────┤ AND ├────┼────┼────┤     │
   S₁ ────┬─────┤     │    │    │    │     │
   S₀'────┼─────┤     │    │    │    └─────┘
          │     └─────┘    │    │
   D₃ ────│     ┌─────┐    │    │
          └─────┤ AND ├────┘────┘
   S₁ ────┬─────┤     │
   S₀ ────┴─────┤     │
                └─────┘

Constructed with two 2:1 MUXes:
        ┌───────┐
   D₀ ──┤ 0     │
        │ MUX   ├──┐    ┌───────┐
   D₁ ──┤ 1     │  └────┤ 0     │
        └───┬───┘       │ MUX   ├─── Y
            │      ┌────┤ 1     │
        ┌───┴───┐  │    └───┬───┘
   D₂ ──┤ 0     │  │        │
        │ MUX   ├──┘        S₁
   D₃ ──┤ 1     │
        └───┬───┘
            S₀
```

### Applications of MUX

```
┌─────────────────────────────────────────────────────────────┐
│                    Applications of MUX                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. Data Selection                                          │
│     - Select one of multiple sources                        │
│     - CPU data paths                                        │
│                                                             │
│  2. Parallel-to-Serial Conversion                           │
│     - Output parallel data sequentially                     │
│                                                             │
│  3. Logic Function Implementation                           │
│     - Truth table outputs as data inputs                    │
│     - Any n-variable function can be implemented with       │
│       2ⁿ:1 MUX                                              │
│                                                             │
│  4. Conditional Data Transfer                               │
│     - Hardware implementation of if-else statements         │
│                                                             │
└─────────────────────────────────────────────────────────────┘

Example: Implementing logic function with MUX

Implement function Y = A'B + AB' (XOR) with 4:1 MUX:

Truth Table:
┌───┬───┬───┐
│ A │ B │ Y │
├───┼───┼───┤
│ 0 │ 0 │ 0 │ → D₀ = 0
│ 0 │ 1 │ 1 │ → D₁ = 1
│ 1 │ 0 │ 1 │ → D₂ = 1
│ 1 │ 1 │ 0 │ → D₃ = 0
└───┴───┴───┘

        ┌─────────┐
    0 ──┤ 00      │
    1 ──┤ 01      │
        │  4:1    ├─── Y = A ⊕ B
    1 ──┤ 10  MUX │
    0 ──┤ 11      │
        └────┬────┘
          A    B
```

---

## 6. Demultiplexer (DEMUX)

### Demultiplexer Concept

```
Demultiplexer (DEMUX):
Transfers one input to one of multiple outputs (reverse of MUX)

┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  1 input → 2ⁿ outputs                                       │
│  n select lines                                             │
│                                                             │
│                    ┌─────────┐───── Y₀                     │
│                    │         │───── Y₁                     │
│           D ───────┤  DEMUX  │───── Y₂                     │
│                    │  1:2ⁿ   │───── Y₃                     │
│                    │         │  ... │                      │
│                    └────┬────┘───── Y₂ⁿ⁻₁                  │
│                         │                                   │
│                    S₀ S₁...Sₙ₋₁                             │
│                                                             │
│  Only selected output receives input D, others are 0        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 1:4 DEMUX

```
1:4 Demultiplexer:

              ┌─────────┐───── Y₀
              │         │───── Y₁
        D ────┤  DEMUX  │
              │  1:4    │───── Y₂
              └────┬────┘───── Y₃
                   │
                S₁  S₀

Truth Table (when D=1):
┌────┬────┬────┬────┬────┬────┐
│ S₁ │ S₀ │ Y₀ │ Y₁ │ Y₂ │ Y₃ │
├────┼────┼────┼────┼────┼────┤
│ 0  │ 0  │ 1  │ 0  │ 0  │ 0  │
│ 0  │ 1  │ 0  │ 1  │ 0  │ 0  │
│ 1  │ 0  │ 0  │ 0  │ 1  │ 0  │
│ 1  │ 1  │ 0  │ 0  │ 0  │ 1  │
└────┴────┴────┴────┴────┴────┘

Logic expressions:
Y₀ = D · S₁' · S₀'
Y₁ = D · S₁' · S₀
Y₂ = D · S₁  · S₀'
Y₃ = D · S₁  · S₀
```

### DEMUX Circuit

```
1:4 DEMUX Circuit:

                     ┌───────┐
        D ───┬───────┤       │
             │       │  AND  ├─── Y₀
   S₁'───────┼───────┤       │
   S₀'───────┼───────┤       │
             │       └───────┘
             │       ┌───────┐
             ├───────┤       │
             │       │  AND  ├─── Y₁
   S₁'───────┼───────┤       │
   S₀ ───────┼───────┤       │
             │       └───────┘
             │       ┌───────┐
             ├───────┤       │
             │       │  AND  ├─── Y₂
   S₁ ───────┼───────┤       │
   S₀'───────┼───────┤       │
             │       └───────┘
             │       ┌───────┐
             └───────┤       │
                     │  AND  ├─── Y₃
   S₁ ───────────────┤       │
   S₀ ───────────────┤       │
                     └───────┘
```

---

## 7. Decoder

### Decoder Concept

```
Decoder:
Activates one of 2ⁿ outputs from n-bit input

┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  n inputs → 2ⁿ outputs                                      │
│  Only output corresponding to input value is 1, rest are 0  │
│                                                             │
│                    ┌─────────┐───── Y₀                     │
│           A₀ ──────┤         │───── Y₁                     │
│           A₁ ──────┤ DECODER │───── Y₂                     │
│           ...      │  n:2ⁿ   │───── Y₃                     │
│           Aₙ₋₁ ────┤         │  ... │                      │
│                    └─────────┘───── Y₂ⁿ⁻₁                  │
│                                                             │
│  Y_i = 1 if input = i (binary)                              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 2:4 Decoder

```
2:4 Decoder (2-to-4 Decoder):

           ┌─────────┐───── Y₀
    A₀ ────┤         │───── Y₁
    A₁ ────┤ 2:4 DEC │───── Y₂
           └─────────┘───── Y₃

Truth Table:
┌────┬────┬────┬────┬────┬────┐
│ A₁ │ A₀ │ Y₀ │ Y₁ │ Y₂ │ Y₃ │
├────┼────┼────┼────┼────┼────┤
│ 0  │ 0  │ 1  │ 0  │ 0  │ 0  │
│ 0  │ 1  │ 0  │ 1  │ 0  │ 0  │
│ 1  │ 0  │ 0  │ 0  │ 1  │ 0  │
│ 1  │ 1  │ 0  │ 0  │ 0  │ 1  │
└────┴────┴────┴────┴────┴────┘

Logic expressions:
Y₀ = A₁' · A₀'
Y₁ = A₁' · A₀
Y₂ = A₁  · A₀'
Y₃ = A₁  · A₀
```

### Decoder Circuit

```
2:4 Decoder Circuit:

                ┌───────┐
   A₁'──────────┤       │
                │  AND  ├─── Y₀ (=A₁'A₀')
   A₀'──────────┤       │
                └───────┘

                ┌───────┐
   A₁'──────────┤       │
                │  AND  ├─── Y₁ (=A₁'A₀)
   A₀ ──────────┤       │
                └───────┘

                ┌───────┐
   A₁ ──────────┤       │
                │  AND  ├─── Y₂ (=A₁A₀')
   A₀'──────────┤       │
                └───────┘

                ┌───────┐
   A₁ ──────────┤       │
                │  AND  ├─── Y₃ (=A₁A₀)
   A₀ ──────────┤       │
                └───────┘
```

### Decoder with Enable Input

```
3:8 Decoder with Enable:

              ┌──────────┐───── Y₀
    A₀ ───────┤          │───── Y₁
    A₁ ───────┤  3:8     │───── Y₂
    A₂ ───────┤  DEC     │───── Y₃
              │          │───── Y₄
    E ────────┤ (Enable) │───── Y₅
              └──────────┘───── Y₆
                          ───── Y₇

E=0: All outputs are 0 (disabled)
E=1: Normal operation

Purpose of Enable:
- Select when connecting multiple decoders
- Chip Select
- Timing control
```

### Applications of Decoder

```
┌─────────────────────────────────────────────────────────────┐
│                   Applications of Decoder                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. Memory Address Decoding                                 │
│     - Select 2ⁿ memory locations with n-bit address        │
│                                                             │
│  2. Instruction Decoding                                    │
│     - Convert operation code (opcode) to control signals    │
│                                                             │
│  3. 7-Segment Display                                       │
│     - Convert BCD to 7-segment pattern                      │
│                                                             │
│  4. Minterm Generator                                       │
│     - Implement logic functions (Decoder + OR gate)         │
│                                                             │
└─────────────────────────────────────────────────────────────┘

Implementing logic function with decoder:

Example: Implement Y = Σm(1, 2, 4, 7) with 3:8 decoder

         ┌──────────┐
   A ────┤          │───── m₀
   B ────┤  3:8     │───── m₁ ──┐
   C ────┤  DEC     │───── m₂ ──┼──┐
         │          │───── m₃   │  │    ┌──────┐
         │          │───── m₄ ──┼──┼────┤      │
         │          │───── m₅   │  │    │  OR  ├─── Y
         │          │───── m₆   │  └────┤      │
         └──────────┘───── m₇ ──┴───────┤      │
                                        └──────┘

Y = m₁ + m₂ + m₄ + m₇
```

---

## 8. Encoder

### Encoder Concept

```
Encoder:
Converts one active input among 2ⁿ inputs to n-bit binary code (reverse of decoder)

┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  2ⁿ inputs → n outputs                                      │
│  Outputs number of active input in binary                   │
│                                                             │
│      D₀ ──────┐                                             │
│      D₁ ──────┼────┐     ┌─────────┐                       │
│      D₂ ──────┼────┼─────┤         ├───── A₀              │
│      D₃ ──────┼────┼─────┤ ENCODER ├───── A₁              │
│      ...      │    │     │  2ⁿ:n   │  ... │                │
│      D₂ⁿ⁻₁───┼────┼─────┤         ├───── Aₙ₋₁            │
│               │    │     └─────────┘                       │
│                                                             │
│  Assumption: Only one input is active at a time             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 4:2 Encoder

```
4:2 Encoder:

           ┌─────────┐
    D₀ ────┤         │───── A₀
    D₁ ────┤  4:2    │
    D₂ ────┤  ENC    │───── A₁
    D₃ ────┤         │
           └─────────┘

Truth Table:
┌────┬────┬────┬────┬────┬────┐
│ D₃ │ D₂ │ D₁ │ D₀ │ A₁ │ A₀ │
├────┼────┼────┼────┼────┼────┤
│ 0  │ 0  │ 0  │ 1  │ 0  │ 0  │
│ 0  │ 0  │ 1  │ 0  │ 0  │ 1  │
│ 0  │ 1  │ 0  │ 0  │ 1  │ 0  │
│ 1  │ 0  │ 0  │ 0  │ 1  │ 1  │
└────┴────┴────┴────┴────┴────┘

Logic expressions:
A₀ = D₁ + D₃
A₁ = D₂ + D₃
```

### Priority Encoder

```
Priority Encoder:
Encodes only the highest priority input even when multiple inputs are active

┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  Problem with regular encoder:                              │
│  - Incorrect output if two or more inputs are 1             │
│                                                             │
│  Priority Encoder:                                          │
│  - Higher numbered input has higher priority                │
│  - Valid output added (indicates if any input is present)   │
│                                                             │
└─────────────────────────────────────────────────────────────┘

4:2 Priority Encoder:

           ┌─────────┐───── A₀
    D₀ ────┤         │───── A₁
    D₁ ────┤  4:2    │
    D₂ ────┤  P.ENC  │───── V (Valid)
    D₃ ────┤         │
           └─────────┘

Truth Table (X = don't care):
┌────┬────┬────┬────┬────┬────┬───┐
│ D₃ │ D₂ │ D₁ │ D₀ │ A₁ │ A₀ │ V │
├────┼────┼────┼────┼────┼────┼───┤
│ 0  │ 0  │ 0  │ 0  │ X  │ X  │ 0 │ ← No input
│ 0  │ 0  │ 0  │ 1  │ 0  │ 0  │ 1 │
│ 0  │ 0  │ 1  │ X  │ 0  │ 1  │ 1 │ ← D₁ has priority
│ 0  │ 1  │ X  │ X  │ 1  │ 0  │ 1 │ ← D₂ has priority
│ 1  │ X  │ X  │ X  │ 1  │ 1  │ 1 │ ← D₃ has highest priority
└────┴────┴────┴────┴────┴────┴───┘

Logic expressions:
A₁ = D₃ + D₂
A₀ = D₃ + D₂'D₁
V  = D₃ + D₂ + D₁ + D₀
```

### Applications of Encoder

```
┌─────────────────────────────────────────────────────────────┐
│                   Applications of Encoder                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. Keyboard Encoder                                        │
│     - Convert pressed key to scan code                      │
│     - Handle simultaneous keys with priority encoder        │
│                                                             │
│  2. Interrupt Controller                                    │
│     - Select highest priority among multiple interrupt      │
│       requests                                              │
│     - Generate interrupt number                             │
│                                                             │
│  3. Position Encoder                                        │
│     - Extract position information from sensor array        │
│                                                             │
│  4. Data Compression                                        │
│     - Convert one-hot code to binary code                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 9. Comparator and Other Circuits

### Comparator

```
1-bit Comparator:
Compares two bits A and B and outputs size relationship

           ┌─────────┐───── G (A > B)
    A ─────┤         │───── E (A = B)
    B ─────┤  COMP   │───── L (A < B)
           └─────────┘

Truth Table:
┌───┬───┬───┬───┬───┐
│ A │ B │ G │ E │ L │
├───┼───┼───┼───┼───┤
│ 0 │ 0 │ 0 │ 1 │ 0 │
│ 0 │ 1 │ 0 │ 0 │ 1 │
│ 1 │ 0 │ 1 │ 0 │ 0 │
│ 1 │ 1 │ 0 │ 1 │ 0 │
└───┴───┴───┴───┴───┘

Logic expressions:
G = A · B'
E = A ⊙ B = (A ⊕ B)'
L = A' · B
```

### n-bit Comparator

```
4-bit Comparator:

   A[3:0] ────────┐     ┌─────────┐───── A > B
                  ├─────┤         │
   B[3:0] ────────┘     │  4-bit  │───── A = B
                        │  COMP   │
                        │         │───── A < B
                        └─────────┘

Operating principle:
Compare from MSB and decide at first different bit

A = 1011, B = 1010
     │        │
     A₃=B₃    A₂=B₂    A₁>B₁ → A > B

Logic expression (cascade):
G = A₃B₃' + (A₃⊙B₃)(A₂B₂' + (A₂⊙B₂)(A₁B₁' + (A₁⊙B₁)A₀B₀'))
E = (A₃⊙B₃)(A₂⊙B₂)(A₁⊙B₁)(A₀⊙B₀)
L = E' · G'
```

### BCD to 7-Segment Decoder

```
7-Segment Display:

     ─a─
    │   │
    f   b
    │   │
     ─g─
    │   │
    e   c
    │   │
     ─d─

BCD input (4 bits) → 7 segment outputs

       ┌─────────────┐───── a
  D₀ ──┤             │───── b
  D₁ ──┤  BCD to     │───── c
  D₂ ──┤  7-Segment  │───── d
  D₃ ──┤  Decoder    │───── e
       │             │───── f
       └─────────────┘───── g

Partial Truth Table:
┌────────┬────────────────────────────┐
│  BCD   │    Segments (a-g)          │
├────────┼────────────────────────────┤
│  0000  │    1 1 1 1 1 1 0  (0)      │
│  0001  │    0 1 1 0 0 0 0  (1)      │
│  0010  │    1 1 0 1 1 0 1  (2)      │
│  0011  │    1 1 1 1 0 0 1  (3)      │
│  0100  │    0 1 1 0 0 1 1  (4)      │
│  0101  │    1 0 1 1 0 1 1  (5)      │
│  0110  │    1 0 1 1 1 1 1  (6)      │
│  0111  │    1 1 1 0 0 0 0  (7)      │
│  1000  │    1 1 1 1 1 1 1  (8)      │
│  1001  │    1 1 1 1 0 1 1  (9)      │
└────────┴────────────────────────────┘
```

### Parity Generator/Checker

```
Parity:
Bit added to data for error detection

Even Parity:
- Add parity bit so total number of 1s is even
- If data has odd number of 1s, P=1; if even, P=0

Odd Parity:
- Add parity bit so total number of 1s is odd

4-bit Even Parity Generator:

   D₀ ───┬───⊕───┬───⊕───┬───⊕─── P (Parity bit)
         │       │       │
   D₁ ───┘       │       │
                 │       │
   D₂ ───────────┘       │
                         │
   D₃ ───────────────────┘

P = D₀ ⊕ D₁ ⊕ D₂ ⊕ D₃

Example: D = 1011
P = 1 ⊕ 0 ⊕ 1 ⊕ 1 = 1
Transmit: 1011 + 1 = 10111 (number of 1s = 4, even)
```

---

## 10. Practice Problems

### Basic Problems

**1. Explain the difference between a half adder and a full adder.**

**2. Calculate 0111 + 0011 in a 4-bit ripple carry adder. Show S and Cₒᵤₜ for each FA.**

**3. In an 8:1 multiplexer, which input goes to the output when select lines are S₂S₁S₀ = 101?**

### Design Problems

**4. Implement a full adder using only NAND gates. What is the minimum number of gates needed?**

**5. Implement the following function using only 4:1 MUX.**
```
Y = A'B + AB'C + ABC'
```

**6. Implement the following function using a 3:8 decoder and OR gate.**
```
Y = Σm(0, 2, 5, 7)
```

### Analysis Problems

**7. Analyze the operation of the following circuit and create a truth table.**
```
       ┌─────────┐
  A ───┤ 0       │
       │   MUX   ├─── Y
  B ───┤ 1       │
       └────┬────┘
            │
            A
```

**8. In an 8-bit ripple carry adder where each FA has a 10ns delay, what is the total delay in the worst case?**

### Application Problems

**9. Design a 4-bit adder/subtractor. When Sub input is 0, perform addition; when 1, perform subtraction.**

**10. Design a circuit that selects the highest priority among 8 interrupt requests using a priority encoder.**

---

<details>
<summary>Answers</summary>

**1.**
- Half Adder: 2 inputs (A, B) output sum and carry. No carry input from lower bit.
- Full Adder: 3 inputs (A, B, Cᵢₙ) output sum and carry. Processes carry from lower bit.

**2.** 0111 + 0011:
- FA₀: 1+1+0 = S₀=0, C₀=1
- FA₁: 1+1+1 = S₁=1, C₁=1
- FA₂: 1+0+1 = S₂=0, C₂=1
- FA₃: 0+0+1 = S₃=1, Cₒᵤₜ=0
- Result: 01010 (7+3=10)

**3.** S₂S₁S₀ = 101 = 5₁₀, so D₅ goes to output.

**4.** Full adder can be implemented with 9 NAND gates.

**5.** Using A, B as select lines:
- D₀(A=0,B=0) = 0
- D₁(A=0,B=1) = C'
- D₂(A=1,B=0) = C
- D₃(A=1,B=1) = C'

**6.** Connect outputs m₀, m₂, m₅, m₇ of 3:8 decoder to OR gate.

**7.** Circuit analysis:
- S=A=0: Y=D₀=A
- S=A=1: Y=D₁=B
- Therefore Y = A'·A + A·B = A·B

**8.** In ripple carry adder when carry propagates from LSB to MSB:
8 FA × 10ns = 80ns

**9.** 4-bit Adder/Subtractor:
- Connect B input through XOR gates with Sub
- Sub=0: B unchanged, Cᵢₙ=0 (addition)
- Sub=1: B inverted, Cᵢₙ=1 (subtraction)

**10.** Use 8:3 priority encoder:
- D₀~D₇: Interrupt requests (D₇ has highest priority)
- A₂A₁A₀: Selected interrupt number
- V: Valid interrupt request present

</details>

---

## Next Steps

- [06_Sequential_Logic.md](./06_Sequential_Logic.md) - Latches, flip-flops, registers, counters

---

## References

- Digital Design (Morris Mano)
- Computer Organization and Design (Patterson & Hennessy)
- [Logisim - Digital Circuit Simulator](http://www.cburch.com/logisim/)
- [CircuitVerse - Online Digital Circuit Simulator](https://circuitverse.org/)
- [Digital Circuits - All About Circuits](https://www.allaboutcircuits.com/textbook/digital/)
