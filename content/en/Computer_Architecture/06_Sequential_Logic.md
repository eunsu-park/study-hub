# Sequential Logic Circuits

**Previous**: [Combinational Logic Circuits](./05_Combinational_Logic.md) | **Next**: [CPU Architecture Basics](./07_CPU_Architecture_Basics.md)

**Difficulty**: ⭐⭐ (Intermediate)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Distinguish combinational logic from sequential logic
2. Explain the SR, D, JK, and T flip-flops and their characteristic tables
3. Describe how registers store data using groups of flip-flops
4. Design a synchronous counter using flip-flops
5. Model a finite state machine with a state diagram and state table
6. Explain the role of the clock signal in synchronizing sequential circuits
7. Calculate the maximum clock frequency from setup time, hold time, and propagation delay

---

Combinational circuits have no memory -- their output depends only on current inputs. Sequential circuits add the crucial ability to remember past inputs, enabling the storage of data and the step-by-step execution of instructions. Registers, counters, and state machines are the building blocks that transform a passive circuit into an active, programmable computer.

## Table of Contents

1. [Characteristics of Sequential Logic Circuits](#1-characteristics-of-sequential-logic-circuits)
2. [SR Latch](#2-sr-latch)
3. [D Latch](#3-d-latch)
4. [D Flip-Flop](#4-d-flip-flop)
5. [JK Flip-Flop](#5-jk-flip-flop)
6. [T Flip-Flop](#6-t-flip-flop)
7. [Registers](#7-registers)
8. [Counters](#8-counters)
9. [Clock and Timing](#9-clock-and-timing)
10. [Practice Problems](#10-practice-problems)

---

## 1. Characteristics of Sequential Logic Circuits

### Combinational Circuits vs Sequential Circuits

```
┌─────────────────────────────────────────────────────────────┐
│                     Circuit Type Comparison                  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Combinational Logic Circuit:                               │
│  ┌────────┐                                                │
│  │        │                                                │
│  │ Input ─┤ Comb.   ├── Output                             │
│  │        │  Circuit│                                      │
│  └────────┘                                                │
│  Output = f(current inputs)                                 │
│                                                             │
│  Sequential Logic Circuit:                                  │
│  ┌────────────────────────────────────────────┐            │
│  │                                            │            │
│  │ Input ──┤ Comb.   ├── Output                │            │
│  │           Circuit                          │            │
│  │            │                               │            │
│  │            ▼                               │            │
│  │        ┌───────┐                           │            │
│  │        │ Memory│                           │            │
│  │        │(State)│◄───────────────────────── │           │
│  │        └───────┘        Feedback           │            │
│  │                                            │            │
│  └────────────────────────────────────────────┘            │
│  Output = f(current inputs, current state)                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Characteristics of Sequential Circuits

```
┌─────────────────────────────────────────────────────────────┐
│          Characteristics of Sequential Logic Circuits        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. Contains Memory Elements                                │
│     - Latches, flip-flops                                   │
│     - Store previous state                                  │
│                                                             │
│  2. Feedback Path Exists                                    │
│     - Output affects input                                  │
│                                                             │
│  3. Time-dependent Operation                                │
│     - Synchronous: State changes by clock                   │
│     - Asynchronous: State changes immediately by input      │
│                                                             │
│  4. State Transition                                        │
│     - Current state + Input → Next state                   │
│     - Represented by State Diagram                          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Synchronous vs Asynchronous

```
Synchronous Sequential Circuit:

        Input ────┐
                  │    ┌──────────┐
                  ├────┤          ├──── Output
                  │    │  Comb.   │
        State ────┼────┤  Logic   │
           ▲      │    └──────────┘
           │      │
           │      │    ┌──────────┐
           └──────┴────┤ Flip-Flop│
                  │    │  (State) │
        CLK ──────┴────┤          │
                       └──────────┘

  - State changes only at clock edge
  - Predictable operation
  - Easy design and analysis


Asynchronous Sequential Circuit:

        Input ────┐
                  │    ┌──────────┐
                  ├────┤          ├──── Output
                  │    │  Comb.   │
        State ────┼────┤  Logic   │
           ▲      │    └──────────┘
           │      │
           │      │    ┌──────────┐
           └──────┴────┤  Latch   │
                       │  (State) │
                       └──────────┘

  - Responds immediately to input changes
  - Glitches possible
  - Complex analysis
```

---

## 2. SR Latch

### SR Latch Concept

```
SR Latch (Set-Reset Latch):
Most basic memory element, stores 1 bit

┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  S (Set):   Set Q to 1                                      │
│  R (Reset): Set Q to 0                                      │
│  Q:         Current state (output)                          │
│  Q':        Complement of Q                                 │
│                                                             │
│  Operation:                                                 │
│  - S=1, R=0: Q=1 (Set)                                      │
│  - S=0, R=1: Q=0 (Reset)                                    │
│  - S=0, R=0: Q holds (Store)                                │
│  - S=1, R=1: Forbidden                                      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### NOR Gate SR Latch

```
SR Latch implemented with NOR gates:

                    ┌───────┐
        R ──────────┤       │
                    │  NOR  ├──────┬───── Q
              ┌─────┤       │      │
              │     └───────┘      │
              │                    │
              │     ┌───────┐      │
              │     │       │      │
              └─────┤  NOR  ├──────┼───── Q'
                    │       │      │
        S ──────────┤       │      │
                    └───────┘      │
                         │         │
                         └─────────┘
                          Feedback

Logic expressions:
Q  = (R + Q')' = R' · Q
Q' = (S + Q)'  = S' · Q'
```

### NAND Gate SR Latch

```
SR Latch implemented with NAND gates (Active-Low):

                    ┌───────┐
        S' ─────────┤       │
                    │ NAND  ├──────┬───── Q
              ┌─────┤       │      │
              │     └───────┘      │
              │                    │
              │     ┌───────┐      │
              │     │       │      │
              └─────┤ NAND  ├──────┼───── Q'
                    │       │      │
        R' ─────────┤       │      │
                    └───────┘      │
                         │         │
                         └─────────┘

NAND SR Latch is Active-Low:
- S'=0, R'=1: Q=1 (Set)
- S'=1, R'=0: Q=0 (Reset)
- S'=1, R'=1: Q holds (Store)
- S'=0, R'=0: Forbidden
```

### SR Latch Truth Table

```
NOR Gate SR Latch Truth Table:

┌───┬───┬────────┬───────────────────────────────┐
│ S │ R │  Q(t+1)│          Operation            │
├───┼───┼────────┼───────────────────────────────┤
│ 0 │ 0 │  Q(t)  │  Hold (No change)             │
│ 0 │ 1 │   0    │  Reset                        │
│ 1 │ 0 │   1    │  Set                          │
│ 1 │ 1 │   ?    │  Forbidden/Invalid            │
└───┴───┴────────┴───────────────────────────────┘

Problem with forbidden state (S=R=1):
1. Both Q and Q' become 0 (Q ≠ Q')
2. Race condition if S, R return to 0 simultaneously
3. Final state is indeterminate
```

### SR Latch Timing Diagram

```
        │
   S ───┼──┐   ┌───┐       ┌───┐
        │  └───┘   └───────┘   └───────────
        │
   R ───┼──────────┐   ┌───┐
        │          └───┘   └───────────────
        │
   Q ───┼──────┐       ┌───────────┐
        │      └───────┘           └───────
        │
   Q'───┼──┐       ┌───┐       ┌───────────
        │  └───────┘   └───────┘
        │
        └─────────────────────────────────────→ Time
            Set   Reset  Set   Reset
```

---

## 3. D Latch

### D Latch Concept

```
D Latch (Data Latch / Gated D Latch):
Solves forbidden state of SR latch, single data input

┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  D (Data):   Data to store                                  │
│  EN (Enable): Gate control signal                           │
│                                                             │
│  Operation:                                                 │
│  - EN=1: Q = D (Transparent)                                │
│  - EN=0: Q holds (Latched)                                  │
│                                                             │
│  D latch passes input through when Enable is 1              │
│  Also called "Transparent Latch"                            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### D Latch Circuit

```
D Latch Circuit (based on SR latch):

                       ┌───────┐
        D ──────┬──────┤       │
                │      │  AND  ├──────── S ───┐
        EN ─────┼──────┤       │              │
                │      └───────┘              │
                │                             │
                │      ┌───────┐         ┌────┴────┐
                │      │       │         │         │
                └─[NOT]┤  AND  ├──────── R │  SR    ├── Q
                       │       │           │ Latch  │
        EN ────────────┤       │           │        ├── Q'
                       └───────┘           └────────┘

Logic expressions:
S = D · EN
R = D' · EN

When EN=1:
- D=1 → S=1, R=0 → Q=1
- D=0 → S=0, R=1 → Q=0
- i.e., Q = D

When EN=0:
- S=0, R=0 → Q holds
```

### D Latch Truth Table

```
D Latch Truth Table:

┌────┬───┬────────┬───────────────────────────────┐
│ EN │ D │  Q(t+1)│          Operation            │
├────┼───┼────────┼───────────────────────────────┤
│ 0  │ 0 │  Q(t)  │  Hold                         │
│ 0  │ 1 │  Q(t)  │  Hold                         │
│ 1  │ 0 │   0    │  Store D (Q=0)                │
│ 1  │ 1 │   1    │  Store D (Q=1)                │
└────┴───┴────────┴───────────────────────────────┘

Simply:
EN=0: Q(t+1) = Q(t)    (Hold)
EN=1: Q(t+1) = D       (Pass through)
```

### D Latch Symbol

```
D Latch Symbol:

        ┌───────────┐
   D ───┤ D       Q ├─── Q
        │           │
   EN ──┤ EN     Q' ├─── Q'
        └───────────┘

Or:

        ┌───────────┐
   D ───┤ D       Q ├─── Q
        │    >o     │
   EN ──┤           ├─── Q'
        └───────────┘
        (Level-triggered)
```

### D Latch Timing

```
        │
   EN ──┼──┐       ┌───────┐       ┌───────────
        │  └───────┘       └───────┘
        │     ↑ Transparent   ↑ Latched
        │
   D ───┼────┐   ┌───┐   ┌─────────┐
        │    └───┘   └───┘         └───────────
        │
   Q ───┼────┐   ┌───────┐   ┌─────────────────
        │    └───┘       └───┘
        │
        └─────────────────────────────────────────→ Time

EN=1 (Transparent): Q follows D
EN=0 (Latched): Q holds last D value
```

---

## 4. D Flip-Flop

### Flip-Flop vs Latch

```
┌─────────────────────────────────────────────────────────────┐
│                    Latch vs Flip-Flop                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Latch:                                                     │
│  - Level-triggered                                          │
│  - Responds to input while Enable is 1                      │
│  - Transparent state exists                                 │
│                                                             │
│  Flip-Flop:                                                 │
│  - Edge-triggered                                           │
│  - Samples input only at clock edge moment                  │
│  - More predictable operation                               │
│                                                             │
│      Level-triggered         Edge-triggered                 │
│         (Latch)              (Flip-Flop)                    │
│          │                      │                           │
│   EN ────┼──┐    ┌────     CLK ─┼──┐    ┌────               │
│          │  └────┘              │  └────┘                   │
│          │  ↑~~~~↑              │  ↑                        │
│          │  Response period     │  Only responds at moment  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### D Flip-Flop Concept

```
D Flip-Flop:
Stores D value to Q only at rising/falling edge of clock

┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  Operation:                                                 │
│  - Rising edge trigger: Store D to Q when CLK goes 0→1      │
│  - Falling edge trigger: Store D to Q when CLK goes 1→0     │
│  - Q holds at all other times                               │
│                                                             │
│  Characteristic equation: Q(t+1) = D                        │
│                          (at clock edge)                    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Master-Slave Structure

```
D Flip-Flop (Master-Slave structure):

         ┌─────────────────────────────────────────┐
         │                                         │
   D ────┼────┬───────┐                           │
         │    │       │                           │
         │    │   ┌───┴───┐     ┌───────┐        │
         │    │   │       │     │       │        │
         │    └───┤ D   Q ├─────┤ D   Q ├────────┼─── Q
         │        │Master │     │Slave  │        │
   CLK ──┼────────┤ EN    │  ┌──┤ EN    │        │
         │        └───────┘  │  └───────┘        │
         │                   │                    │
         │        ┌──────────┘                   │
         │        │  [NOT]                       │
         │        │                              │
         └────────┴──────────────────────────────┘

Operation:
1. CLK=0: Master latch transparent, Slave latch fixed
   - D passes to Master's Q
   - Slave holds previous value

2. CLK=1: Master latch fixed, Slave latch transparent
   - Master holds value
   - Master's value passes to Slave (final Q)

Result: D passes to Q at rising edge
```

### D Flip-Flop Symbol

```
Rising edge-triggered D Flip-Flop:

        ┌───────────┐
   D ───┤ D       Q ├─── Q
        │           │
   CLK ─┤ >      Q' ├─── Q'
        └───────────┘
          ↑
        Triangle = Edge-triggered (rising)


Falling edge-triggered D Flip-Flop:

        ┌───────────┐
   D ───┤ D       Q ├─── Q
        │           │
   CLK ─┤ >o     Q' ├─── Q'
        └───────────┘
          ↑
        Circle = Inverted (falling edge)
```

### D Flip-Flop Truth Table

```
Rising edge-triggered D Flip-Flop Truth Table:

┌───────┬───┬─────────┬───────────────────────────┐
│  CLK  │ D │  Q(t+1) │        Operation          │
├───────┼───┼─────────┼───────────────────────────┤
│   0   │ X │   Q(t)  │  Hold                     │
│   1   │ X │   Q(t)  │  Hold                     │
│   ↓   │ X │   Q(t)  │  Hold (falling edge)      │
│   ↑   │ 0 │    0    │  Store D (rising edge)    │
│   ↑   │ 1 │    1    │  Store D (rising edge)    │
└───────┴───┴─────────┴───────────────────────────┘

↑ = Rising edge (0→1)
↓ = Falling edge (1→0)
X = Don't care
```

### D Flip-Flop Timing

```
        │
  CLK ──┼──┐  ┌──┐  ┌──┐  ┌──┐  ┌──┐  ┌──┐
        │  └──┘  └──┘  └──┘  └──┘  └──┘  └──
        │     ↑     ↑     ↑     ↑     ↑
        │
   D ───┼───────┐        ┌─────────────┐
        │       └────────┘             └───
        │
   Q ───┼───────────┐        ┌─────────────
        │           └────────┘
        │        ↑        ↑        ↑
        │     Rising   Rising   Rising
        │      Edge     Edge     Edge
        └─────────────────────────────────────→ Time

Q stores D value at rising edge moment
Holds until next rising edge
```

### Reset/Preset Function

```
D Flip-Flop with Asynchronous Reset/Preset:

        ┌─────────────────┐
  PRE ──┤ PR            Q ├─── Q
        │                 │
   D ───┤ D               │
        │                 │
  CLK ──┤ >            Q' ├─── Q'
        │                 │
  CLR ──┤ CLR             │
        └─────────────────┘

Operation:
- CLR=1 (Active): Q=0 (Asynchronous reset)
- PRE=1 (Active): Q=1 (Asynchronous preset)
- CLR=0, PRE=0: Normal operation (store D at clock edge)

Truth Table:
┌─────┬─────┬───────┬───┬────────┐
│ CLR │ PRE │  CLK  │ D │  Q     │
├─────┼─────┼───────┼───┼────────┤
│  1  │  0  │   X   │ X │   0    │ ← Async reset
│  0  │  1  │   X   │ X │   1    │ ← Async preset
│  1  │  1  │   X   │ X │   ?    │ ← Forbidden
│  0  │  0  │   ↑   │ D │   D    │ ← Normal operation
│  0  │  0  │  0/1  │ X │  Q(t)  │ ← Hold
└─────┴─────┴───────┴───┴────────┘
```

---

## 5. JK Flip-Flop

### JK Flip-Flop Concept

```
JK Flip-Flop:
Universal flip-flop that replaces SR's forbidden state with "toggle"

┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  J (Jump):  Similar to Set                                  │
│  K (Kill):  Similar to Reset                                │
│                                                             │
│  Operation (at clock edge):                                 │
│  - J=0, K=0: Q holds (No change)                            │
│  - J=0, K=1: Q=0 (Reset)                                    │
│  - J=1, K=0: Q=1 (Set)                                      │
│  - J=1, K=1: Q toggles (Q' = Q(t)')                         │
│                                                             │
│  Characteristic equation: Q(t+1) = J·Q' + K'·Q              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### JK Flip-Flop Truth Table

```
JK Flip-Flop Truth Table:

┌───┬───┬─────────┬───────────────────────────────┐
│ J │ K │  Q(t+1) │          Operation            │
├───┼───┼─────────┼───────────────────────────────┤
│ 0 │ 0 │   Q(t)  │  Hold                         │
│ 0 │ 1 │    0    │  Reset                        │
│ 1 │ 0 │    1    │  Set                          │
│ 1 │ 1 │  Q(t)'  │  Toggle                       │
└───┴───┴─────────┴───────────────────────────────┘

Expanded Truth Table:
┌───┬───┬───────┬─────────┐
│ J │ K │  Q(t) │  Q(t+1) │
├───┼───┼───────┼─────────┤
│ 0 │ 0 │   0   │    0    │
│ 0 │ 0 │   1   │    1    │
│ 0 │ 1 │   0   │    0    │
│ 0 │ 1 │   1   │    0    │
│ 1 │ 0 │   0   │    1    │
│ 1 │ 0 │   1   │    1    │
│ 1 │ 1 │   0   │    1    │
│ 1 │ 1 │   1   │    0    │
└───┴───┴───────┴─────────┘
```

### JK Flip-Flop Symbol

```
JK Flip-Flop Symbol:

        ┌───────────┐
   J ───┤ J       Q ├─── Q
        │           │
  CLK ──┤ >         │
        │           │
   K ───┤ K      Q' ├─── Q'
        └───────────┘
```

### JK Flip-Flop Circuit

```
JK Flip-Flop (SR-based):

                ┌───────┐
   J ─────────┬─┤       │
              │ │  AND  ├───────── S ───┐
   Q' ────────┼─┤       │               │
              │ └───────┘               │
              │                    ┌────┴────┐
              │                    │         │
              │                    │   SR    ├── Q
   CLK ───────┼────────────────────┤  F/F    │
              │                    │         ├── Q'
              │                    └────┬────┘
              │ ┌───────┐               │
   K ─────────┼─┤       │               │
              │ │  AND  ├───────── R ───┘
   Q ─────────┴─┤       │
                └───────┘

S = J · Q'
R = K · Q

When J=K=1:
- If Q=0: S=1, R=0 → Q=1
- If Q=1: S=0, R=1 → Q=0
→ Toggle!
```

---

## 6. T Flip-Flop

### T Flip-Flop Concept

```
T Flip-Flop (Toggle Flip-Flop):
Flip-flop specialized for toggle function

┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  Operation (at clock edge):                                 │
│  - T=0: Q holds                                             │
│  - T=1: Q toggles                                           │
│                                                             │
│  Characteristic equation: Q(t+1) = T ⊕ Q(t) = T·Q' + T'·Q   │
│                                                             │
│  Use: Core component of counters                            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### T Flip-Flop Truth Table

```
T Flip-Flop Truth Table:

┌───┬─────────┬───────────────────────────────┐
│ T │  Q(t+1) │          Operation            │
├───┼─────────┼───────────────────────────────┤
│ 0 │   Q(t)  │  Hold                         │
│ 1 │  Q(t)'  │  Toggle                       │
└───┴─────────┴───────────────────────────────┘

Expanded:
┌───┬───────┬─────────┐
│ T │  Q(t) │  Q(t+1) │
├───┼───────┼─────────┤
│ 0 │   0   │    0    │
│ 0 │   1   │    1    │
│ 1 │   0   │    1    │
│ 1 │   1   │    0    │
└───┴───────┴─────────┘
```

### T Flip-Flop Implementation

```
Implementing T Flip-Flop with JK Flip-Flop:

        ┌───────────┐
   T ───┤ J       Q ├─── Q
        │           │
  CLK ──┤ >         │
        │           │
   T ───┤ K      Q' ├─── Q'
        └───────────┘

Connect J = K = T


Implementing T Flip-Flop with D Flip-Flop:

                    ┌───────┐
   T ───────────────┤       │
                    │  XOR  ├───┐
   Q ──────┬────────┤       │   │
           │        └───────┘   │
           │                    │
           │     ┌──────────┐   │
           │     │          │   │
           └─────┤ D      Q ├───┴─── Q
                 │          │
   CLK ──────────┤ >     Q' ├─────── Q'
                 └──────────┘

D = T ⊕ Q
```

### Flip-Flop Comparison

```
┌───────────────────────────────────────────────────────────────┐
│                    Flip-Flop Comparison                        │
├───────────┬───────────────────────────────────────────────────┤
│   Type    │               Characteristics                     │
├───────────┼───────────────────────────────────────────────────┤
│           │  Q(t+1) = D                                       │
│     D     │  - Stores input directly                          │
│           │  - Mainly used in registers                       │
├───────────┼───────────────────────────────────────────────────┤
│           │  Q(t+1) = J·Q' + K'·Q                             │
│    JK     │  - Most versatile                                 │
│           │  - Can be converted to any other flip-flop        │
├───────────┼───────────────────────────────────────────────────┤
│           │  Q(t+1) = T ⊕ Q                                   │
│     T     │  - Toggle function                                │
│           │  - Mainly used in counters                        │
├───────────┼───────────────────────────────────────────────────┤
│           │  Q(t+1) = S + R'·Q (condition S·R=0)              │
│    SR     │  - Basic latch                                    │
│           │  - Forbidden state exists (S=R=1)                 │
└───────────┴───────────────────────────────────────────────────┘

Conversions:
┌─────────────┬─────────────────────────────────────────────────┐
│ Conversion  │                    Method                        │
├─────────────┼─────────────────────────────────────────────────┤
│  D → JK     │  J = D, K = D'                                  │
│  D → T      │  T = D ⊕ Q                                      │
│  JK → D     │  D = J                                          │
│  JK → T     │  J = K = T                                      │
│  T → JK     │  J = K = T                                      │
│  T → D      │  D = T ⊕ Q                                      │
└─────────────┴─────────────────────────────────────────────────┘
```

---

## 7. Registers

### Register Concept

```
Register:
Group of flip-flops that stores multiple bits of data simultaneously

┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  Uses:                                                      │
│  - Temporary storage of data inside CPU                     │
│  - Data transfer (parallel/serial conversion)               │
│  - Address storage                                          │
│  - Instruction storage                                      │
│                                                             │
│  Types:                                                     │
│  - Parallel In/Parallel Out (PIPO)                          │
│  - Serial In/Serial Out (SISO)                              │
│  - Serial In/Parallel Out (SIPO)                            │
│  - Parallel In/Serial Out (PISO)                            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 4-bit Parallel Register

```
4-bit Parallel In/Parallel Out Register:

        D₃         D₂         D₁         D₀
         │          │          │          │
         ▼          ▼          ▼          ▼
    ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐
    │  D  Q  │ │  D  Q  │ │  D  Q  │ │  D  Q  │
    │        │ │        │ │        │ │        │
CLK─┤ >      │─┤ >      │─┤ >      │─┤ >      │
    │        │ │        │ │        │ │        │
    │     Q' │ │     Q' │ │     Q' │ │     Q' │
    └────────┘ └────────┘ └────────┘ └────────┘
         │          │          │          │
         ▼          ▼          ▼          ▼
        Q₃         Q₂         Q₁         Q₀

Operation:
- At clock edge, D₃D₂D₁D₀ are simultaneously stored to Q₃Q₂Q₁Q₀
- Load function only
```

### Register with Load Function

```
4-bit Register with Load Control:

                 D₃                D₀
                  │                 │
                  ▼                 ▼
              ┌───────┐         ┌───────┐
              │ 0     │         │ 0     │
   Load ──────┤  MUX  │  ...    │  MUX  │
              │ 1     │         │ 1     │
        ┌─────┤       │   ┌─────┤       │
        │     └───┬───┘   │     └───┬───┘
        │         │       │         │
        │         ▼       │         ▼
        │    ┌────────┐   │    ┌────────┐
        │    │  D  Q  │   │    │  D  Q  │
        │    │        │   │    │        │
   CLK ─┼────┤ >      │───┼────┤ >      │
        │    │        │   │    │        │
        │    │     Q' │   │    │     Q' │
        │    └────────┘   │    └────────┘
        │         │       │         │
        └─────────┴───────┴─────────┘
                  │                 │
                  ▼                 ▼
                 Q₃                Q₀

Operation:
- Load=0: Q holds (MUX selects Q)
- Load=1: Store D (MUX selects D)
```

### Shift Register

```
4-bit Serial In/Parallel Out Shift Register (SIPO):

                                                    Serial Out
                                                         ↑
   ┌────────┐    ┌────────┐    ┌────────┐    ┌────────┐ │
   │  D  Q  │───→│  D  Q  │───→│  D  Q  │───→│  D  Q  │─┘
   │        │    │        │    │        │    │        │
   │ >      │    │ >      │    │ >      │    │ >      │
   │        │    │        │    │        │    │        │
   └───┬────┘    └───┬────┘    └───┬────┘    └───┬────┘
       │             │             │             │
       ▲             │             │             │
  Serial In         CLK           CLK           CLK
                     └─────────────┴─────────────┘

Operation (at each clock edge):
- Data shifts one position from left to right
- New bit enters via Serial In

Example (initial value 0000, input 1101):
CLK  Serial_In  Q₃Q₂Q₁Q₀
 0      -       0 0 0 0
 1      1       1 0 0 0
 2      1       1 1 0 0
 3      0       0 1 1 0
 4      1       1 0 1 1
```

### Bidirectional Shift Register

```
Bidirectional Shift Register:

              ┌─────────────────────────────────────┐
              │                                     │
 Left_In ─────┼─┐                               ┌───┼───── Right_Out
              │ │   ┌─────────┐   ┌─────────┐   │   │
              │ └───┤ 00      │   │         │   │   │
              │     │ 01  MUX ├───┤ D     Q ├───┼───┼─┐
 Right_In ────┼─────┤ 10      │   │         │   │   │ │
              │ ┌───┤ 11      │   │ >       │   │   │ │
              │ │   └────┬────┘   └─────────┘   │   │ │
              │ │        │              ↑       │   │ │
              │ │     S₁ S₀            CLK      │   │ │
              │ │                               │   │ │
              │ └───────────────────────────────┘   │ │
              │                                     │ │
              └─────────────────────────────────────┼─┘
                                                    │
                                                Right_In

Mode selection (S₁S₀):
- 00: Hold
- 01: Right shift
- 10: Left shift
- 11: Parallel load
```

---

## 8. Counters

### Counter Concept

```
Counter:
Sequential circuit that counts clock pulses

┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  Types:                                                     │
│  - Asynchronous Counter (Ripple Counter): Simple but slow   │
│  - Synchronous Counter: Fast                                │
│                                                             │
│  Operation:                                                 │
│  - Up Counter: 0, 1, 2, 3, ...                             │
│  - Down Counter: 7, 6, 5, 4, ...                           │
│  - Up/Down Counter                                          │
│                                                             │
│  Modulus:                                                   │
│  - MOD-n counter: Counts from 0 to n-1                      │
│  - n-bit counter: Counts from 0 to 2ⁿ-1 (MOD-2ⁿ)           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Asynchronous Counter (Ripple Counter)

```
3-bit Asynchronous Up Counter:

              ┌────────┐      ┌────────┐      ┌────────┐
   CLK ───────┤ T    Q ├──────┤ T    Q ├──────┤ T    Q ├
              │        │      │        │      │        │
      1 ──────┤        │   1──┤        │   1──┤        │
              │     Q' │      │     Q' │      │     Q' │
              └───┬────┘      └───┬────┘      └───┬────┘
                  │               │               │
                  Q₀              Q₁              Q₂
                 (LSB)                           (MSB)

Operation:
- Each T flip-flop has T=1 (always toggles)
- Q₀ operates directly from CLK
- Q₁ operates from Q₀ falling edge
- Q₂ operates from Q₁ falling edge

Timing:
        │
  CLK ──┼─┐ ┐ ┐ ┐ ┐ ┐ ┐ ┐ ┐
        │ └─┘ └─┘ └─┘ └─┘ └─┘
        │
   Q₀ ──┼───┐   ┌───┐   ┌───┐   ┌───
        │   └───┘   └───┘   └───┘
        │
   Q₁ ──┼───────┐       ┌───────┐
        │       └───────┘       └───
        │
   Q₂ ──┼───────────────┐
        │               └───────────
        │
Count:  0   1   2   3   4   5   6   7   0...
```

### Problems with Asynchronous Counter

```
Ripple Delay:

┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  Delay accumulates through each flip-flop                   │
│                                                             │
│  4-bit counter example (0111 → 1000 transition):            │
│                                                             │
│     Q₃  Q₂  Q₁  Q₀                                          │
│      0   1   1   1  (7)                                     │
│           ↓   ↓   ↓                                         │
│      0   1   1   0  (6)  ← Glitch!                          │
│           ↓   ↓                                             │
│      0   1   0   0  (4)  ← Glitch!                          │
│           ↓                                                 │
│      0   0   0   0  (0)  ← Glitch!                          │
│                                                             │
│      1   0   0   0  (8)  ← Final state                      │
│                                                             │
│  Total delay = n × (flip-flop delay)                        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Synchronous Counter

```
3-bit Synchronous Up Counter:

                   AND
                    │
        ┌──────────┬┴─────────┐
        │          │          │
  ┌────────┐  ┌────────┐  ┌────────┐
  │ T    Q ├──┤ T    Q ├──┤ T    Q ├
  │        │  │        │  │        │
1─┤        │  │        │  │        │
  │     Q' │  │     Q' │  │     Q' │
  └───┬────┘  └───┬────┘  └───┬────┘
      │  ↑        │  ↑        │  ↑
     Q₀ CLK      Q₁ CLK      Q₂ CLK
                      │
           ┌──────────┴──────────┐
           │                     │
   CLK ────┴─────────────────────┘

Logic:
T₀ = 1                    (always toggle)
T₁ = Q₀                   (toggle when Q₀=1)
T₂ = Q₀ · Q₁              (toggle when Q₀=Q₁=1)

Operation:
- All flip-flops operate simultaneously with same CLK
- No glitches
- Can operate at high speed
```

### MOD-N Counter

```
MOD-6 Counter (counts 0~5):

3-bit counter + reset logic

  ┌────────┐  ┌────────┐  ┌────────┐
  │ D    Q ├──┤ D    Q ├──┤ D    Q ├
  │        │  │        │  │        │
  │        │  │        │  │        │
  │     Q' │  │     Q' │  │     Q' │
  └───┬────┘  └───┬────┘  └───┬────┘
      │           │           │
      Q₀          Q₁          Q₂
      │           │           │
      └─────┬─────┴───────────┘
            │
        ┌───┴───┐
        │       │
        │  AND  ├───→ Reset (Detect Q₂·Q₁ = 6)
        │       │
        └───────┘

Operation:
- 0, 1, 2, 3, 4, 5, 0, 1, 2, ...
- Resets to 0 immediately when reaches 6 (110)

Or design with sequential logic:
States: 0→1→2→3→4→5→0...
```

### Up/Down Counter

```
Synchronous Up/Down Counter:

Determines count direction based on Up/Down control signal

             Up/Down
                │
        ┌───────┼───────┐
        │       │       │
   ┌────┴───┐   │   ┌───┴────┐
   │ Q₀     │   │   │    Q₀' │
   │    AND │   │   │ AND    │
   │ Up     │   │   │   Down │
   └────┬───┘   │   └───┬────┘
        │       │       │
        └───────┼───────┘
                │
            ┌───┴───┐
            │  OR   ├────→ T₁
            └───────┘

Logic:
Up=1:   T₁ = Q₀ (toggle when previous bit is 1)
Down=1: T₁ = Q₀' (toggle when previous bit is 0)
```

---

## 9. Clock and Timing

### Clock Signal

```
Clock:
Periodic signal controlling timing of synchronous circuits

┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  Clock waveform:                                            │
│                                                             │
│       ┌────┐    ┌────┐    ┌────┐    ┌────┐                 │
│       │    │    │    │    │    │    │    │                 │
│   ────┘    └────┘    └────┘    └────┘    └────             │
│       │    │    │                                          │
│       ├────┼────┤                                          │
│       │    │    │                                          │
│       T_H  T_L  T (Period)                                 │
│                                                             │
│  - Period (T): Time of one cycle                           │
│  - Frequency (f): f = 1/T                                  │
│  - Duty cycle: T_H / T × 100%                              │
│                                                             │
│  Example: 1GHz clock                                        │
│  - T = 1ns (nanosecond)                                    │
│  - 1 billion cycles per second                             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Setup Time and Hold Time

```
Flip-Flop Timing Parameters:

┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  Setup Time (t_su):                                         │
│  - Minimum time input must be stable before clock edge     │
│                                                             │
│  Hold Time (t_h):                                           │
│  - Minimum time input must be maintained after clock edge  │
│                                                             │
│  Propagation Delay (t_pd):                                  │
│  - Time from clock edge to output change                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘

Timing Diagram:

        │←──t_su──→│←t_h→│
        │          │     │
   D ───┼───┐      │     │
        │   └──────┼─────┼─────────
        │          │     │
        │          │ ↑   │
  CLK ──┼──────────┼─┼───┼─────────
        │          │ │   │
        │          │ │   │
        │          │←t_pd→│
   Q ───┼──────────┼─────┼───┐
        │          │     │   └─────
        │
        └──────────────────────────→ Time
```

### Timing Violations

```
Setup Time Violation:

        │←t_su→│   (required)
        │      │
   D ───┼──────┼──────┐
        │  ↑   │      └─────
        │  │   │
        │  Actual change (late!)
        │      │
  CLK ──┼──────┼──────────────
        │      │ ↑
               Clock edge

Result: Output indeterminate (metastable state possible)


Hold Time Violation:

        │      │←t_h→│  (required)
        │      │     │
   D ───┼──────┼─────┼┐
        │      │     │└─────
        │      │  ↑
        │      │  Actual change (early!)
        │      │
  CLK ──┼──────┼──────────────
        │      │ ↑
               Clock edge

Result: Either previous or new input may be sampled
```

### Maximum Clock Frequency

```
Maximum Operating Frequency Calculation:

┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  ┌────────┐         ┌──────────┐         ┌────────┐        │
│  │ FF₁    │         │  Comb.   │         │ FF₂    │        │
│  │        ├─────────┤  Logic   ├─────────┤        │        │
│  │      Q │         │          │         │ D      │        │
│  └────┬───┘         └──────────┘         └───┬────┘        │
│       │                                      │             │
│       └──────────────────────────────────────┘             │
│                      │                                     │
│  CLK ────────────────┴─────────────────────────            │
│                                                             │
│  Minimum clock period:                                     │
│  T_min = t_pd(FF₁) + t_comb + t_su(FF₂)                    │
│                                                             │
│  Maximum frequency:                                         │
│  f_max = 1 / T_min                                         │
│                                                             │
│  Example:                                                   │
│  t_pd = 2ns, t_comb = 5ns, t_su = 1ns                      │
│  T_min = 2 + 5 + 1 = 8ns                                   │
│  f_max = 1 / 8ns = 125MHz                                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Metastability

```
Metastable State:

┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  Unstable state occurring on setup/hold time violation      │
│                                                             │
│  Normal state:           Metastable state:                  │
│                                                             │
│      1 ──●──             1 ──          ●                   │
│          │                   ╲        ╱                    │
│          │                    ╲●●●●●╱                      │
│          │                         │                       │
│      0 ──┴──             0 ──      │                       │
│                                    Indeterminate            │
│                                                             │
│  Danger:                                                    │
│  - Output at intermediate value for indeterminate time     │
│  - Next stage cannot determine if 0 or 1                   │
│  - System malfunction possible                              │
│                                                             │
│  Solution:                                                  │
│  - Use synchronizer circuits                                │
│  - Design timing with sufficient margin                     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 10. Practice Problems

### Basic Problems

**1. Explain the difference between a latch and a flip-flop.**

**2. Write the characteristic equation of a D flip-flop and explain its operation.**

**3. Explain the operation when J=K=1 in a JK flip-flop.**

### Analysis Problems

**4. Analyze the operation of the following circuit and create a state transition table.**

```
        ┌───────────┐
   D ───┤ D       Q ├───┬─── Q
        │           │   │
  CLK ──┤ >      Q' ├───┼─── Q'
        └───────────┘   │
              ↑         │
              └─────────┘
```

**5. Draw a timing diagram for a 4-bit ripple counter. Initial state is 0000.**

**6. Explain the operation of the following register. Initial value is 0000, input sequence is 1, 0, 1, 1.**

```
4-bit right shift register (serial input)
```

### Design Problems

**7. Implement a T flip-flop using only D flip-flops.**

**8. Design a MOD-5 synchronous counter. (0, 1, 2, 3, 4, 0, ...)**

**9. Design a 4-bit bidirectional shift register. (Left, Right, Hold modes)**

### Timing Problems

**10. Calculate the maximum clock frequency given the following conditions.**
- Flip-flop propagation delay: 5ns
- Combinational circuit delay: 15ns
- Setup time: 3ns
- Hold time: 2ns

---

<details>
<summary>Answers</summary>

**1.**
- Latch: Level-triggered, responds to input while Enable is active (transparent)
- Flip-flop: Edge-triggered, samples input only at clock edge moment

**2.** Q(t+1) = D (at clock edge). At the rising (or falling) edge of the clock, store D input value to Q and hold until the next edge.

**3.** When J=K=1, Q toggles. i.e., Q(t+1) = Q(t)'. If currently 0, becomes 1; if 1, becomes 0.

**4.** The circuit is a D flip-flop with D input connected to Q'. This operates identically to a T flip-flop, toggling at every clock edge.
State transition: 0→1→0→1→...

**5.**
```
CLK: _|‾|_|‾|_|‾|_|‾|_|‾|_|‾|_|‾|_|‾|_
Q₀:  __|‾‾|__|‾‾|__|‾‾|__|‾‾|__|‾‾|__
Q₁:  ____|‾‾‾‾|____|‾‾‾‾|____|‾‾‾‾|__
Q₂:  ________|‾‾‾‾‾‾‾‾|________|‾‾‾‾
Q₃:  ________________|‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾
```

**6.** Right shift operation:
- Initial: 0000
- Input 1: 1000
- Input 0: 0100
- Input 1: 1010
- Input 1: 1101

**7.** T flip-flop with D flip-flop:
- D = T XOR Q
- T=0: D=Q (hold)
- T=1: D=Q' (toggle)

**8.** MOD-5 counter (using JK flip-flops):
- States: 000→001→010→011→100→000
- J₀=1, K₀=1 (always toggle)
- J₁=Q₀·Q₂', K₁=Q₀
- J₂=Q₀·Q₁, K₂=Q₀

**9.** Bidirectional shift register:
- Place MUX in front of each flip-flop
- Select Left/Right/Hold with control input
- Left: Select next bit
- Right: Select previous bit
- Hold: Select current Q

**10.** Maximum clock frequency:
T_min = t_pd + t_comb + t_su = 5 + 15 + 3 = 23ns
f_max = 1 / 23ns ≈ 43.5MHz

</details>

---

## Next Steps

- [07_CPU_Architecture_Basics.md](./07_CPU_Architecture_Basics.md) - CPU components and instruction execution cycle

---

## References

- Digital Design (Morris Mano)
- Computer Organization and Design (Patterson & Hennessy)
- [Logic Gate Simulator](https://logic.ly/)
- [Digital Circuits Tutorial](https://www.electronics-tutorials.ws/sequential/sequential.html)
- [Nand2Tetris - Building a Computer from First Principles](https://www.nand2tetris.org/)
