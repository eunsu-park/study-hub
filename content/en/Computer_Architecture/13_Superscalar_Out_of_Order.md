# Superscalar and Out-of-Order Execution

**Previous**: [12_Branch_Prediction.md](./12_Branch_Prediction.md) | **Next**: [14_Memory_Hierarchy.md](./14_Memory_Hierarchy.md)

**Difficulty**: ⭐⭐⭐⭐

**Prerequisites**: Pipelining, Branch Prediction, CPU Architecture Basics

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain superscalar execution and how it issues multiple instructions per cycle
2. Describe out-of-order execution and why it improves performance
3. Explain register renaming and how it eliminates false dependencies (WAR, WAW)
4. Distinguish true data dependencies (RAW) from false dependencies
5. Describe Tomasulo's algorithm at a conceptual level
6. Explain the role of the reorder buffer in maintaining program order

---

Modern CPUs do not execute instructions in the order you wrote them. Out-of-order, superscalar execution lets the processor find and exploit instruction-level parallelism automatically -- executing independent instructions simultaneously even when they appear sequential in your code. This is how a single CPU core achieves performance that would otherwise require explicit parallelization.

## Table of Contents

1. [Instruction-Level Parallelism (ILP)](#1-instruction-level-parallelism-ilp)
2. [Superscalar Processors](#2-superscalar-processors)
3. [Need for Out-of-Order Execution](#3-need-for-out-of-order-execution)
4. [Register Renaming](#4-register-renaming)
5. [Tomasulo Algorithm](#5-tomasulo-algorithm)
6. [Reorder Buffer (ROB)](#6-reorder-buffer-rob)
7. [Modern Processor Implementations](#7-modern-processor-implementations)
8. [Register Renaming: Deep Dive](#8-register-renaming-deep-dive)
9. [Practice Problems](#9-practice-problems)

---

## 1. Instruction-Level Parallelism (ILP)

### 1.1 ILP Concept

Instruction-Level Parallelism (ILP) refers to the potential parallelism of instructions within a program that can be executed simultaneously.

```
Sequential vs Parallel Execution:

Sequential Execution:
  Time →
  t1    t2    t3    t4    t5    t6
  ├─────┼─────┼─────┼─────┼─────┼─────┤
  │ I1  │ I2  │ I3  │ I4  │ I5  │ I6  │
  └─────┴─────┴─────┴─────┴─────┴─────┘
  Total 6 cycles

Parallel Execution (ILP = 2):
  Time →
  t1    t2    t3
  ├─────┼─────┼─────┤
  │ I1  │ I3  │ I5  │
  │ I2  │ I4  │ I6  │
  └─────┴─────┴─────┘
  Total 3 cycles
```

### 1.2 Data Dependence

The most important factor limiting ILP is data dependence.

#### RAW (Read After Write) - True Dependence

```assembly
I1: ADD R1, R2, R3    ; R1 = R2 + R3
I2: SUB R4, R1, R5    ; R4 = R1 - R5  (uses R1, needs I1's result)
```

```
I1 ────────→ I2
   R1 dependence

I2 can only execute after I1 writes to R1
```

#### WAR (Write After Read) - Anti-dependence

```assembly
I1: ADD R1, R2, R3    ; Read R2
I2: SUB R2, R4, R5    ; Write R2 (must write after I1 reads R2)
# Why this is "false": I2's computation has nothing to do with I1's R2 value.
# The conflict exists only because the ISA has limited register names —
# the compiler reused "R2". Register renaming gives I2 a fresh physical
# register, letting it execute in parallel with I1.
```

```
I2 must not overwrite R2 before I1 reads it
→ Can be resolved by register renaming
```

#### WAW (Write After Write) - Output Dependence

```assembly
I1: ADD R1, R2, R3    ; Write R1
I2: SUB R1, R4, R5    ; Write R1 (writing to same register)
# Why this is "false": I1 and I2 compute completely independent results.
# The only constraint is that the final value of R1 must be I2's result
# (since I2 comes later in program order). With renaming, each gets its
# own physical register — I1 writes P10, I2 writes P11 — and they can
# execute simultaneously.
```

```
If I2 completes before I1, final R1 value is wrong
→ Can be resolved by register renaming
```

### 1.3 Dependence Graph

```
Program:
I1: LD   R1, 0(R10)     ; Load from memory
I2: ADD  R2, R1, R3     ; RAW on R1
I3: LD   R4, 8(R10)     ; Independent
I4: MUL  R5, R4, R6     ; RAW on R4
I5: ADD  R7, R2, R5     ; RAW on R2, R5
I6: ST   R7, 16(R10)    ; RAW on R7

Dependence graph:
        I1          I3
         │           │
         ▼           ▼
        I2          I4
         │           │
         └─────┬─────┘
               │
               ▼
              I5
               │
               ▼
              I6

Parallelizable groups:
- Level 1: I1, I3 (can execute simultaneously)
- Level 2: I2, I4 (can execute simultaneously)
- Level 3: I5
- Level 4: I6

Minimum execution time: 4 levels (4 cycles) vs sequential: 6 cycles
ILP = 6/4 = 1.5
```

### 1.4 Control Dependence

```assembly
      BEQ  R1, R2, LABEL    ; Branch instruction
      ADD  R3, R4, R5       ; Execute if branch not taken
      ...
LABEL:
      SUB  R6, R7, R8       ; Execute if branch taken
```

```
Don't know which instruction to execute before branch result
→ Resolved by branch prediction
```

---

## 2. Superscalar Processors

### 2.1 Superscalar Concept

A superscalar processor can fetch, decode, and execute multiple instructions per cycle.

```
Scalar vs Superscalar:

Scalar Pipeline (IPC ≤ 1):
┌─────┬─────┬─────┬─────┬─────┐
│ IF  │ ID  │ EX  │ MEM │ WB  │ I1
└─────┴─────┴─────┴─────┴─────┘
      ┌─────┬─────┬─────┬─────┬─────┐
      │ IF  │ ID  │ EX  │ MEM │ WB  │ I2
      └─────┴─────┴─────┴─────┴─────┘

2-way Superscalar (IPC ≤ 2):
┌─────┬─────┬─────┬─────┬─────┐
│ IF  │ ID  │ EX  │ MEM │ WB  │ I1
├─────┼─────┼─────┼─────┼─────┤
│ IF  │ ID  │ EX  │ MEM │ WB  │ I2
└─────┴─────┴─────┴─────┴─────┘
      ┌─────┬─────┬─────┬─────┬─────┐
      │ IF  │ ID  │ EX  │ MEM │ WB  │ I3
      ├─────┼─────┼─────┼─────┼─────┤
      │ IF  │ ID  │ EX  │ MEM │ WB  │ I4
      └─────┴─────┴─────┴─────┴─────┘
```

### 2.2 Superscalar Structure

```
┌─────────────────────────────────────────────────────────────────┐
│                     4-way Superscalar Processor                   │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              Instruction Fetch Unit                       │   │
│  │  ┌──────────┐    ┌──────────────────────────────────┐    │   │
│  │  │    PC    │───→│  Instruction Cache (I-Cache)      │    │   │
│  │  └──────────┘    └──────────────────────────────────┘    │   │
│  │                              │                            │   │
│  │                    4 instructions/cycle                   │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                │                                 │
│                                ▼                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              Instruction Decode Unit                      │   │
│  │  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐         │   │
│  │  │Decoder1│  │Decoder2│  │Decoder3│  │Decoder4│         │   │
│  │  └────────┘  └────────┘  └────────┘  └────────┘         │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                │                                 │
│                                ▼                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    Issue Unit                             │   │
│  │          (Dependence checking and execution unit alloc)   │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                │                                 │
│        ┌───────────┬──────────┼──────────┬───────────┐         │
│        ▼           ▼          ▼          ▼           ▼         │
│  ┌──────────┐┌──────────┐┌──────────┐┌──────────┐┌──────────┐  │
│  │  ALU 1   ││  ALU 2   ││  FPU     ││  Load    ││  Store   │  │
│  │          ││          ││          ││  Unit    ││  Unit    │  │
│  └──────────┘└──────────┘└──────────┘└──────────┘└──────────┘  │
│        │           │          │          │           │         │
│        └───────────┴──────────┴──────────┴───────────┘         │
│                                │                                 │
│                                ▼                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              Write-back / Commit                          │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### 2.3 Issue Policies

#### In-Order Issue

```
Instructions issued to execution units in program order

Program:
I1: ADD R1, R2, R3
I2: MUL R4, R1, R5    ; Depends on I1 (stall)
I3: SUB R6, R7, R8    ; Independent but waits behind I2

Timeline:
Cycle 1: Issue I1
Cycle 2: I1 executing, I2 waiting (RAW hazard)
Cycle 3: Issue I2 (after I1 completes)
Cycle 4: Issue I3

Problem: I3 is independent but delayed by I2
```

#### Out-of-Order Issue

```
Independent instructions issued regardless of order

Program:
I1: ADD R1, R2, R3
I2: MUL R4, R1, R5    ; Depends on I1
I3: SUB R6, R7, R8    ; Independent

Timeline:
Cycle 1: Issue I1, I3 (simultaneous)
Cycle 2: I1 completes, Issue I2
Cycle 3: I2 executing

Performance gain: I3 doesn't wait for I2
```

### 2.4 Diverse Execution Units

```
Modern processor execution units example:

┌─────────────────────────────────────────────────┐
│           Execution Units (Intel Core)           │
├────────────────┬────────────────────────────────┤
│ Port 0         │ ALU, FP MUL, FP DIV, Branch    │
├────────────────┼────────────────────────────────┤
│ Port 1         │ ALU, FP ADD, LEA               │
├────────────────┼────────────────────────────────┤
│ Port 2         │ Load (Address Gen)             │
├────────────────┼────────────────────────────────┤
│ Port 3         │ Load (Address Gen)             │
├────────────────┼────────────────────────────────┤
│ Port 4         │ Store Data                     │
├────────────────┼────────────────────────────────┤
│ Port 5         │ ALU, Vector Shuffle, Branch    │
├────────────────┼────────────────────────────────┤
│ Port 6         │ ALU, Branch                    │
├────────────────┼────────────────────────────────┤
│ Port 7         │ Store (Address Gen)            │
└────────────────┴────────────────────────────────┘

Total: 8 ports, max 8 micro-ops per cycle
```

---

## 3. Need for Out-of-Order Execution

### 3.1 Limitations of In-Order Execution

```
Program:
I1: LD   R1, 0(R10)     ; Cache miss - 100 cycles
I2: ADD  R2, R1, R3     ; Depends on I1
I3: LD   R4, 8(R10)     ; Independent - cache hit 4 cycles
I4: MUL  R5, R4, R6     ; Depends on I3
I5: ADD  R7, R8, R9     ; Completely independent

In-Order Execution:
Cycle 1-100: I1 executing (cache miss wait)
Cycle 101:   I2 executes
Cycle 102-105: I3 executes
Cycle 106:   I4 executes
Cycle 107:   I5 executes

Total: 107 cycles

Out-of-Order Execution:
Cycle 1:     I1 starts (cache miss)
Cycle 2-5:   I3 executes (parallel with I1)
Cycle 6:     I4 executes
Cycle 7:     I5 executes
...
Cycle 100:   I1 completes
Cycle 101:   I2 executes

Total: 101 cycles

Speedup: 107/101 = 1.06x (simple example)
Actual difference is much larger
```

### 3.2 Three Stages of OoO Execution

```
┌─────────────────────────────────────────────────────────────┐
│              Out-of-Order Execution Pipeline                 │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   In-Order       Out-of-Order       In-Order                │
│   Front-end      Execution          Back-end                │
│                                                              │
│  ┌─────────┐    ┌─────────────┐    ┌─────────────┐         │
│  │ Fetch   │    │ Issue/      │    │ Commit/     │         │
│  │ Decode  │───→│ Execute     │───→│ Retire      │         │
│  │ Rename  │    │ (OoO)       │    │ (In-Order)  │         │
│  └─────────┘    └─────────────┘    └─────────────┘         │
│                                                              │
│  Program order   Data flow order    Program order           │
│                                                              │
└─────────────────────────────────────────────────────────────┘

Stage 1: Front-end (In-Order)
- Instruction fetch, decode
- Register renaming
- Insert instructions into Issue Queue

Stage 2: Execution (Out-of-Order)
- Execute instructions when operands ready
- Execution order determined by data dependencies
- Parallel execution across multiple units

Stage 3: Back-end (In-Order)
- Commit results in program order
- Ensures precise exception handling
- Update architectural registers
```

---

## 4. Register Renaming

### 4.1 Need for Renaming

```
Program:
I1: ADD R1, R2, R3    ; Write R1
I2: MUL R4, R1, R5    ; Read R1 (RAW)
I3: ADD R1, R6, R7    ; Write R1 (WAW with I1)
I4: SUB R8, R1, R9    ; Read R1 (RAW with I3)

Problem:
- I3 should be able to execute independently of I1, I2
- But they share R1, causing WAW dependence
- If I3 completes before I1, I2 reads wrong value
```

### 4.2 Renaming Operation

```
Architectural Registers: R1-R8 (programmer-visible)
Physical Registers: P1-P64 (actual hardware registers)

Before renaming:
I1: ADD R1, R2, R3
I2: MUL R4, R1, R5
I3: ADD R1, R6, R7
I4: SUB R8, R1, R9

After renaming:
I1: ADD P10, P2, P3    ; R1 → P10
I2: MUL P11, P10, P5   ; R4 → P11, R1 → P10
I3: ADD P12, P6, P7    ; R1 → P12 (new physical register!)
I4: SUB P13, P12, P9   ; R8 → P13, R1 → P12

Result:
- WAW dependence between I1 and I3 eliminated (different physical regs)
- I2 reads P10 (I1's result)
- I4 reads P12 (I3's result)
- I1 and I3 can now execute in parallel!
```

### 4.3 Register Alias Table (RAT)

```
┌─────────────────────────────────────────────────────────┐
│              Register Alias Table (RAT)                  │
├─────────────────────────────────────────────────────────┤
│  Architectural Reg  │  Physical Reg  │  Valid           │
├─────────────────────┼────────────────┼──────────────────┤
│        R0           │      P0        │    1             │
│        R1           │      P12       │    0 (pending)   │
│        R2           │      P2        │    1             │
│        R3           │      P3        │    1             │
│        R4           │      P11       │    0 (pending)   │
│        R5           │      P5        │    1             │
│        R6           │      P6        │    1             │
│        R7           │      P7        │    1             │
│        R8           │      P13       │    0 (pending)   │
│        ...          │      ...       │    ...           │
└─────────────────────┴────────────────┴──────────────────┘

Free List: P14, P15, P16, ...  (available physical registers)
```

### 4.4 Renaming Algorithm

```
Renaming process (instruction: ADD Rd, Rs1, Rs2):

1. Source register renaming:
   - Look up physical registers for Rs1, Rs2 in RAT

2. Destination register renaming:
   - Allocate new physical register from Free List
   - Update RAT mapping for Rd to new physical register

3. Record dependence info:
   - Check if source physical registers are still being produced
   - Establish links to producer instructions

Example:
Instruction: ADD R1, R2, R3

Before:
  RAT[R1] = P5, RAT[R2] = P2, RAT[R3] = P3
  Free List: P10, P11, P12, ...

Renaming:
  1. Rs1(R2) → P2, Rs2(R3) → P3
  2. Rd(R1) → P10 (new allocation)
  3. RAT[R1] = P10

After:
  RAT[R1] = P10, RAT[R2] = P2, RAT[R3] = P3
  Free List: P11, P12, ...

Renamed instruction: ADD P10, P2, P3
```

---

## 5. Tomasulo Algorithm

### 5.1 Background

Algorithm developed by Robert Tomasulo in 1967 for the IBM 360/91 floating-point unit. Forms the basis of modern out-of-order execution processors.

### 5.2 Key Components

```
┌─────────────────────────────────────────────────────────────────┐
│                  Tomasulo Architecture                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                   Instruction Queue                       │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              │                                   │
│                              ▼                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                   Issue Logic                             │   │
│  │            (Reservation Station allocation)               │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              │                                   │
│            ┌─────────────────┴─────────────────┐                │
│            ▼                                   ▼                │
│  ┌──────────────────────┐          ┌──────────────────────┐    │
│  │  Reservation Stations │          │  Reservation Stations│    │
│  │      (Add/Sub)        │          │      (Mul/Div)       │    │
│  ├──────────────────────┤          ├──────────────────────┤    │
│  │ RS1: Op Vj Vk Qj Qk  │          │ RS4: Op Vj Vk Qj Qk  │    │
│  │ RS2: Op Vj Vk Qj Qk  │          │ RS5: Op Vj Vk Qj Qk  │    │
│  │ RS3: Op Vj Vk Qj Qk  │          │ RS6: Op Vj Vk Qj Qk  │    │
│  └──────────┬───────────┘          └──────────┬───────────┘    │
│             │                                  │                │
│             ▼                                  ▼                │
│  ┌──────────────────────┐          ┌──────────────────────┐    │
│  │      FP Adder        │          │     FP Multiplier    │    │
│  │    (2 cycles)        │          │     (10 cycles)      │    │
│  └──────────┬───────────┘          └──────────┬───────────┘    │
│             │                                  │                │
│             └─────────────────┬────────────────┘                │
│                               │                                  │
│                               ▼                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                   Common Data Bus (CDB)                   │   │
│  │               (Result broadcast)                          │   │
│  └──────────────────────────────────────────────────────────┘   │
│                               │                                  │
│                               ▼                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                   Register File                           │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 5.3 Reservation Station Structure

```
┌─────────────────────────────────────────────────────────────┐
│              Reservation Station Entry                       │
├──────┬──────┬──────┬──────┬──────┬──────┬──────┬───────────┤
│ Busy │  Op  │  Vj  │  Vk  │  Qj  │  Qk  │  A   │   Dest    │
├──────┼──────┼──────┼──────┼──────┼──────┼──────┼───────────┤
│  1   │ ADD  │ 3.5  │  -   │  -   │ RS4  │  -   │    F2     │
└──────┴──────┴──────┴──────┴──────┴──────┴──────┴───────────┘

Field descriptions:
- Busy: Entry in use
- Op: Operation to perform
- Vj, Vk: Source operand values (if already available)
- Qj, Qk: RS that will produce value (if not ready yet)
- A: Memory address (for Load/Store)
- Dest: Destination register for result
```

### 5.4 Operation Process

```
Three-stage processing:

1. Issue
   - Get instruction from Instruction Queue
   - Allocate appropriate Reservation Station
   - Record operand values or producer RS tags

2. Execute
   - Start execution when all operands ready
   - Execute when Qj = 0 AND Qk = 0
   - Perform operation in execution unit

3. Write Result
   - Broadcast result via CDB
   - Waiting RSs receive result
   - Update Register File
```

### 5.5 Example: Tomasulo Execution Trace

```
Program:
I1: LD   F6, 34(R2)
I2: LD   F2, 45(R3)
I3: MUL  F0, F2, F4
I4: SUB  F8, F6, F2
I5: DIV  F10, F0, F6
I6: ADD  F6, F8, F2

Initial state:
  F4 = 2.5 (available)
  Load: 2 cycles, Mul: 10 cycles, Add/Sub: 2 cycles, Div: 40 cycles

=== Cycle 1 ===
Issue I1: LD F6, 34(R2)
  Load1: Busy=1, A=34+R2, Dest=F6
  Register[F6]: Qi=Load1

=== Cycle 2 ===
Issue I2: LD F2, 45(R3)
  Load2: Busy=1, A=45+R3, Dest=F2
  Register[F2]: Qi=Load2
Execute I1: Memory access starts

=== Cycle 3 ===
Issue I3: MUL F0, F2, F4
  Mult1: Busy=1, Op=MUL, Vk=2.5, Qj=Load2, Dest=F0
  Register[F0]: Qi=Mult1
Execute I2: Memory access starts
Write I1: Broadcast F6 value on CDB
  Load1: Busy=0
  Register[F6]: Qi=0, Value=M[34+R2]

=== Cycle 4 ===
Issue I4: SUB F8, F6, F2
  Add1: Busy=1, Op=SUB, Vj=M[34+R2], Qk=Load2, Dest=F8
Write I2: Broadcast F2 value on CDB
  Mult1: Vj=M[45+R3], Qj=0  (value received)
  Add1: Vk=M[45+R3], Qk=0   (value received)

=== Cycle 5 ===
Issue I5: DIV F10, F0, F6
  Mult2: Busy=1, Op=DIV, Vk=M[34+R2], Qj=Mult1, Dest=F10
Execute I3: MUL starts (Vj, Vk ready)
Execute I4: SUB starts (Vj, Vk ready)

=== Cycle 6 ===
Issue I6: ADD F6, F8, F2
  Add2: Busy=1, Op=ADD, Vk=M[45+R3], Qj=Add1, Dest=F6
  Register[F6]: Qi=Add2

=== Cycle 7 ===
Write I4: Broadcast F8 value on CDB
  Add2: Vj=(F6-F2), Qj=0  (value received)

=== Cycle 8 ===
Execute I6: ADD starts

... (continues)

=== Cycle 15 ===
Write I3: MUL complete (cycle 5+10)
  Mult2: Vj=(F2*F4), Qj=0  (value received)

=== Cycle 16 ===
Execute I5: DIV starts

=== Cycle 56 ===
Write I5: DIV complete (cycle 16+40)
```

---

## 6. Reorder Buffer (ROB)

### 6.1 Need for ROB

```
Problem: Precise exception handling impossible with OoO execution

Example:
I1: LD   R1, 0(R2)     ; May cause page fault
I2: ADD  R3, R4, R5    ; Completes before I1

If I2 completes first and I1 page faults:
- I2's result already written to R3
- State inconsistent after exception handling and restart

Solution: Reorder Buffer
- Temporarily store results
- Commit in program order
- Discard uncommitted results on exception
```

### 6.2 ROB Structure

```
┌─────────────────────────────────────────────────────────────┐
│                    Reorder Buffer                            │
├─────┬─────────┬──────────┬─────────┬────────┬──────────────┤
│Entry│  Busy   │  State   │  Dest   │ Value  │ Instruction  │
├─────┼─────────┼──────────┼─────────┼────────┼──────────────┤
│  1  │    1    │ Commit   │   F6    │  10.5  │ LD F6,34(R2) │
│  2  │    1    │ Commit   │   F2    │   5.0  │ LD F2,45(R3) │
│  3  │    1    │ Execute  │   F0    │   -    │ MUL F0,F2,F4 │
│  4  │    1    │ Write    │   F8    │   5.5  │ SUB F8,F6,F2 │
│  5  │    1    │ Issue    │  F10    │   -    │ DIV F10,F0,F6│
│  6  │    1    │ Issue    │   F6    │   -    │ ADD F6,F8,F2 │
│  7  │    0    │   -      │   -     │   -    │      -       │
│  8  │    0    │   -      │   -     │   -    │      -       │
└─────┴─────────┴──────────┴─────────┴────────┴──────────────┘
      ↑                                                ↑
    Head                                             Tail
  (Commit)                                         (Issue)

State:
- Issue: Issued, waiting to execute
- Execute: Executing
- Write: Execution complete, result recorded
- Commit: Ready to commit
```

### 6.3 Integrating ROB with Tomasulo

```
┌─────────────────────────────────────────────────────────────────┐
│            Modern Out-of-Order Processor                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                  Instruction Fetch/Decode                   │ │
│  └────────────────────────────────────────────────────────────┘ │
│                              │                                   │
│                              ▼                                   │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                    Register Rename                          │ │
│  │              (RAT + Physical Register File)                 │ │
│  └────────────────────────────────────────────────────────────┘ │
│                              │                                   │
│                              ▼                                   │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │              Reorder Buffer (ROB) Allocation               │ │
│  │                  (Entry allocation, order tracking)         │ │
│  └────────────────────────────────────────────────────────────┘ │
│                              │                                   │
│                              ▼                                   │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │              Issue Queue / Reservation Stations             │ │
│  │                  (Dependence wait, issue control)           │ │
│  └────────────────────────────────────────────────────────────┘ │
│                              │                                   │
│         ┌────────────────────┼────────────────────┐             │
│         ▼                    ▼                    ▼             │
│  ┌────────────┐      ┌────────────┐      ┌────────────┐        │
│  │ Execution  │      │ Execution  │      │   Memory   │        │
│  │  Unit 1    │      │  Unit 2    │      │   Unit     │        │
│  └─────┬──────┘      └─────┬──────┘      └─────┬──────┘        │
│        │                   │                   │                │
│        └───────────────────┼───────────────────┘                │
│                            │                                    │
│                            ▼                                    │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │              Write Back to ROB                              │ │
│  │              (Record results in ROB)                        │ │
│  └────────────────────────────────────────────────────────────┘ │
│                            │                                    │
│                            ▼                                    │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │              Commit (In-Order)                              │ │
│  │     (Update architectural registers from ROB Head)          │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 6.4 Commit Process

```
Commit rules:
1. Only ROB Head instruction can commit
2. Instruction must be in complete state
3. All previous instructions must be committed
4. No exception present

On exception:
┌──────────────────────────────────────────────────┐
│  I1  │  I2  │  I3  │  I4  │  I5  │  I6  │       │
│ Done │ Done │ Done │ Exc! │ Done │ Done │       │
└──────────────────────────────────────────────────┘
   ↑
 Head

Exception at I4:
1. Commit I1, I2, I3 complete
2. Exception detected at I4
3. Discard I5, I6 results (not committed yet)
4. Jump to exception handler at I4's address
5. Precise exception state maintained
```

### 6.5 Branch Misprediction Recovery

```
Recovery on branch misprediction:

ROB state:
┌────────────────────────────────────────────────────┐
│  I1  │  I2  │  BR  │  I4  │  I5  │  I6  │         │
│ Done │ Done │Mis-P│ Done │ Done │ Done │         │
└────────────────────────────────────────────────────┘
   ↑           ↑
 Head        branch

Misprediction confirmed at BR:
1. Commit I1, I2
2. Detect misprediction at BR
3. Flush I4, I5, I6 (speculative execution results)
4. Restore RAT to BR checkpoint (checkpoint or ROB reverse walk)
5. Restart fetch at correct branch target

Recovery mechanisms:
- Checkpoint: Save RAT snapshot at each branch
- Gradual recovery: Walk ROB backwards to restore RAT
```

---

## 7. Modern Processor Implementations

### 7.1 Intel Core Architecture (Arrow Lake — Lion Cove P-core)

Arrow Lake (2024) introduces Intel's tile-based architecture, separating P-cores and E-cores into distinct compute tiles. The Lion Cove P-core is the first Intel core to reach 8-wide decode, closing the gap with Apple's wide-issue designs.

```
┌─────────────────────────────────────────────────────────────────┐
│          Intel Arrow Lake — Lion Cove P-core (2024)              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Front-end (In-Order)                                           │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  - Branch prediction: improved TAGE-like, ~12K BTB         │ │
│  │  - L1 I-Cache: 64KB, 8-way                                 │ │
│  │  - Decode: 8-wide (up from 6-wide in Raptor Cove)          │ │
│  │  - Micro-op Cache: ~4K micro-ops                           │ │
│  │  - Allocation: 8 micro-ops/cycle                           │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  Out-of-Order Engine                                            │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  - ROB: 576 entries (up from 512 in Raptor Cove)           │ │
│  │  - Scheduler: 192 entries (doubled from Raptor Cove)       │ │
│  │  - Physical Registers: 280 integer + 332 vector            │ │
│  │  - Load Buffer: 192 entries                                │ │
│  │  - Store Buffer: 128 entries                               │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  Execution Units (12 Ports)                                     │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  Port 0: ALU, FMA, FP Div                                  │ │
│  │  Port 1: ALU, FMA, AES                                     │ │
│  │  Port 2: ALU, Branch                                       │ │
│  │  Port 3: ALU, Branch                                       │ │
│  │  Port 4: Load AGU                                          │ │
│  │  Port 5: Load AGU                                          │ │
│  │  Port 6: Store AGU                                         │ │
│  │  Port 7: Store AGU                                         │ │
│  │  Port 8: Store Data                                        │ │
│  │  Port 9: Store Data                                        │ │
│  │  Port 10: ALU                                              │ │
│  │  Port 11: ALU                                              │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  Memory Subsystem                                               │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  - L1 D-Cache: 48KB, 12-way, 5 cycles                      │ │
│  │  - L2 Cache: 3MB/core, 12-way, ~17 cycles                  │ │
│  │  - L3 Cache: up to 36MB shared, ~45 cycles                  │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  Performance Metrics                                            │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  - Theoretical max: 8 micro-ops execution/cycle            │ │
│  │  - Actual IPC: ~3-5 depending on workload                  │ │
│  │  - Pipeline depth: ~20 stages                              │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

> **Note**: Skylake (2015) was the seminal microarchitecture that established the 4-wide decode, 224-entry ROB baseline that Intel iterated upon for nearly a decade. Arrow Lake represents a major architectural leap with tile-based disaggregation and significantly widened front-end/back-end structures.

### 7.2 ARM Cortex-A77 Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                 ARM Cortex-A77 Microarchitecture                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Front-end                                                      │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  - Fetch: 4 instructions/cycle                             │ │
│  │  - Decode: 4-wide                                          │ │
│  │  - Macro-op Cache: 1.5K entries                            │ │
│  │  - Branch prediction: TAGE-based                           │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  Out-of-Order Engine                                            │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  - ROB: 160 entries                                        │ │
│  │  - Dispatch: 6 micro-ops/cycle                             │ │
│  │  - Issue Queue: 120 entries                                │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  Execution Units (10 pipelines)                                 │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  - 2x Branch                                               │ │
│  │  - 3x Integer ALU                                          │ │
│  │  - 2x Integer Multi-Cycle                                  │ │
│  │  - 2x FP/NEON                                              │ │
│  │  - 2x Load + 1x Store                                      │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 7.3 Performance Comparison

```
┌──────────────────────────────────────────────────────────────────────┐
│       Modern High-Performance Processor Comparison (2025)             │
├───────────────┬────────────────┬────────────────┬────────────────────┤
│     Feature   │ Intel Arrow    │  AMD Zen 5     │  Apple M4 P-core   │
│               │ Lake(Lion Cove)│  (Nirvana)     │                    │
├───────────────┼────────────────┼────────────────┼────────────────────┤
│  Decode Width │     8-wide     │  8-wide(2×4)   │     10-wide        │
├───────────────┼────────────────┼────────────────┼────────────────────┤
│  ROB Size     │   576 entry    │   448 entry    │    ~650 entry      │
├───────────────┼────────────────┼────────────────┼────────────────────┤
│  Issue Width  │    12 ports    │   12 ports     │    ~16 ports       │
├───────────────┼────────────────┼────────────────┼────────────────────┤
│  L1 I-Cache   │     64KB       │     32KB       │     192KB          │
├───────────────┼────────────────┼────────────────┼────────────────────┤
│  L1 D-Cache   │     48KB       │     48KB       │     128KB          │
├───────────────┼────────────────┼────────────────┼────────────────────┤
│  L2 Cache     │     3MB        │     1MB        │     16MB           │
├───────────────┼────────────────┼────────────────┼────────────────────┤
│ Actual IPC    │     ~3-5       │     ~3-5       │      ~5-6          │
├───────────────┼────────────────┼────────────────┼────────────────────┤
│  Key Feature  │  Tile-based    │  Native        │  Unified memory    │
│               │  disaggregation│  AVX-512       │  architecture      │
└───────────────┴────────────────┴────────────────┴────────────────────┘
```

### 7.4 Limitations of ILP

```
Limiting factors for ILP exploitation:

1. True Data Dependence (RAW)
   - Cannot be resolved by renaming
   - Long dependency chains determine performance

2. Control Dependence
   - Branch prediction accuracy limits (~95-97%)
   - Pipeline flush on misprediction

3. Memory Dependence
   - Load-Store dependence detection difficulty
   - Constraints on memory instruction reordering

4. Window Size Limits
   - ROB, Issue Queue size constraints
   - Cannot exploit ILP far apart

Actual program ILP:

┌────────────────────────────────────────────────────┐
│  Program Type        │  Avg ILP    │  Limiting Factor │
├────────────────────────────────────────────────────┤
│  Integer (SPEC INT)  │   1.5-2.5  │  Dependency chains│
│  FP (SPEC FP)        │   2.0-4.0  │  Memory bandwidth │
│  Media processing    │   3.0-6.0  │  SIMD utilization │
│  Database            │   1.0-2.0  │  Branch prediction│
│  Web browser         │   1.5-2.5  │  Control flow     │
└────────────────────────────────────────────────────┘
```

---

## 8. Register Renaming: Deep Dive

While Section 4 introduced the concept of register renaming, this section provides a detailed walkthrough with a concrete simulation and explores the hardware mechanisms in depth.

### 8.1 WAR and WAW: The False Dependency Problem

False dependencies arise because the ISA has a limited number of architectural registers (e.g., 16 in x86-64, 32 in ARM/RISC-V). Programmers and compilers reuse the same register names, creating artificial ordering constraints:

```
Example showing how register scarcity creates false dependencies:

I1: MUL R1, R2, R3      ; R1 = R2 * R3          (produces R1)
I2: ADD R4, R1, R5      ; R4 = R1 + R5          (RAW: reads R1 from I1 ← TRUE dep)
I3: SUB R1, R6, R7      ; R1 = R6 - R7          (WAW: writes R1 like I1)
                                                  (WAR: writes R1 that I2 reads)
I4: ADD R8, R1, R9      ; R8 = R1 + R9          (RAW: reads R1 from I3 ← TRUE dep)
I5: MUL R4, R10, R11    ; R4 = R10 * R11        (WAW: writes R4 like I2)

Without renaming:
- I3 must wait for I2 to read R1 (WAR) — but I3's computation is independent!
- I5 must wait for I2 to complete (WAW on R4) — but I5 is independent!
- Only I2→I1 and I4→I3 are true data dependencies

With renaming, I3 and I5 can execute in parallel with I1 and I2.
```

### 8.2 The Register Alias Table (RAT) in Detail

The RAT is the central bookkeeping structure for register renaming. It maps each architectural register to its current physical register:

```
Hardware structures for register renaming:

┌─────────────────────────────────────────────────────────────────┐
│                   Register Renaming Hardware                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────┐                                            │
│  │   RAT (Mapping)   │  Arch Reg → Physical Reg                  │
│  │   R0 → P0         │  Updated on every WRITE (destination)     │
│  │   R1 → P7         │                                           │
│  │   R2 → P2         │  Consulted on every READ (source)         │
│  │   ...             │                                           │
│  └──────────────────┘                                            │
│                                                                  │
│  ┌──────────────────┐                                            │
│  │  Free List (FIFO) │  Pool of unallocated physical registers   │
│  │  [P14, P15, P16,  │  New physical reg allocated for each      │
│  │   P17, P18, ...]  │  instruction that writes a result         │
│  └──────────────────┘                                            │
│                                                                  │
│  ┌──────────────────┐                                            │
│  │  Physical Reg File │  Actual storage (180+ integer regs       │
│  │  P0: 42            │  in modern x86 processors)               │
│  │  P1: 17            │                                           │
│  │  P2: 99            │  Much larger than architectural set       │
│  │  ...               │  (x86-64 has only 16 arch regs)          │
│  └──────────────────┘                                            │
│                                                                  │
│  Physical regs freed when the old mapping is no longer needed    │
│  (after the instruction that overwrote it has committed)         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 8.3 Detailed Renaming Trace (5 Instructions)

Let us trace the renaming of 5 instructions step by step, showing the RAT state after each instruction:

```
Initial state:
  RAT: R1→P1(val=10), R2→P2(val=20), R3→P3(val=30),
       R4→P4(val=40), R5→P5(val=50), R6→P6(val=60)
  Free List: [P20, P21, P22, P23, P24, ...]

─── Instruction 1: ADD R1, R2, R3  (R1 = R2 + R3) ───

  Source renaming:
    R2 → look up RAT → P2 (value 20, ready)
    R3 → look up RAT → P3 (value 30, ready)
  Destination renaming:
    R1 → allocate P20 from Free List
    Old mapping: R1→P1 (remember P1 for later freeing)
    Update RAT: R1→P20

  Renamed: ADD P20, P2, P3
  RAT: R1→P20, R2→P2, R3→P3, R4→P4, R5→P5, R6→P6
  Free List: [P21, P22, P23, P24, ...]

─── Instruction 2: MUL R4, R1, R5  (R4 = R1 * R5) ───

  Source renaming:
    R1 → look up RAT → P20 (NOT READY yet — I1 hasn't finished)
    R5 → look up RAT → P5 (value 50, ready)
  Destination renaming:
    R4 → allocate P21
    Old mapping: R4→P4
    Update RAT: R4→P21

  Renamed: MUL P21, P20, P5     ← will wait for P20
  RAT: R1→P20, R2→P2, R3→P3, R4→P21, R5→P5, R6→P6
  Free List: [P22, P23, P24, ...]

─── Instruction 3: SUB R1, R6, R2  (R1 = R6 - R2) ───

  Source renaming:
    R6 → look up RAT → P6 (value 60, ready)
    R2 → look up RAT → P2 (value 20, ready)
  Destination renaming:
    R1 → allocate P22 (NEW physical register!)
    Old mapping: R1→P20
    Update RAT: R1→P22

  Renamed: SUB P22, P6, P2
  RAT: R1→P22, R2→P2, R3→P3, R4→P21, R5→P5, R6→P6

  KEY INSIGHT: I3 writes to P22, not P20.
  → I2 still reads P20 (I1's result) — WAR eliminated!
  → I1 writes P20, I3 writes P22 — WAW eliminated!
  → I3 can execute IN PARALLEL with I1!

─── Instruction 4: ADD R5, R1, R4  (R5 = R1 + R4) ───

  Source renaming:
    R1 → look up RAT → P22 (I3's result, not ready yet)
    R4 → look up RAT → P21 (I2's result, not ready yet)
  Destination renaming:
    R5 → allocate P23
    Update RAT: R5→P23

  Renamed: ADD P23, P22, P21    ← waits for both P22 and P21
  RAT: R1→P22, R2→P2, R3→P3, R4→P21, R5→P23, R6→P6

─── Instruction 5: MUL R2, R1, R3  (R2 = R1 * R3) ───

  Source renaming:
    R1 → P22, R3 → P3
  Destination renaming:
    R2 → allocate P24
    Update RAT: R2→P24

  Renamed: MUL P24, P22, P3

Final RAT: R1→P22, R2→P24, R3→P3, R4→P21, R5→P23, R6→P6
Free List: [P25, P26, ...]

Summary of renamed program:
  I1: ADD P20, P2, P3
  I2: MUL P21, P20, P5     (true dep on I1 via P20)
  I3: SUB P22, P6, P2      (INDEPENDENT of I1, I2!)
  I4: ADD P23, P22, P21    (true dep on I2, I3)
  I5: MUL P24, P22, P3     (true dep on I3 only)

Parallelism exposed:
  Cycle 1: I1, I3 execute in parallel (no dependency)
  Cycle 2: I5 can start (depends on I3 only)
  Cycle 2+: I2 starts after I1 completes
  Later:   I4 starts after both I2 and I3 complete
```

### 8.4 Connection to Tomasulo's Algorithm

Register renaming and Tomasulo's algorithm solve the same problem (eliminating false dependencies) but in different ways:

```
Tomasulo (1967):                    Modern Renaming (post-1990):
  - Renaming via RS tags              - Explicit RAT + physical registers
  - Tags = Reservation Station IDs    - Tags = physical register numbers
  - Implicit renaming                 - Explicit renaming stage in pipeline
  - Results broadcast on CDB          - Results written to physical reg file

Modern processors combine BOTH ideas:
  1. RAT performs explicit renaming (front-end)
  2. Issue Queue (evolved from RS) tracks dependencies
  3. Physical Register File stores values
  4. Results forwarded via bypass network (evolved from CDB)
```

### 8.5 Physical Register Freeing

A critical question: when can a physical register be returned to the Free List?

```
Rule: Physical register Pold can be freed when:
  1. The instruction that OVERWROTE the mapping (Pold → Pnew) has COMMITTED
  2. AND no older in-flight instruction still needs Pold

In practice:
  I1: ADD R1, R2, R3  → R1 mapped to P20 (old mapping P1)
      When I1 COMMITS, P1 can be freed
      (because all instructions before I1 that used P1 have also committed)

This is why ROB + renaming work together:
  - ROB tracks program order and commit point
  - On commit: free the OLD physical register from the overwritten mapping
  - On flush (misprediction): restore the OLD mapping and free the NEW registers
```

### 8.6 Python Simulation: Register Renaming

```python
"""
Register Renaming Simulator
Demonstrates how the RAT eliminates false dependencies.
"""
from collections import deque

class RegisterRenamer:
    def __init__(self, num_arch_regs=8, num_phys_regs=32):
        self.num_arch = num_arch_regs
        self.num_phys = num_phys_regs

        # RAT: maps architectural register name -> physical register number
        # Initially, Ri -> Pi (identity mapping)
        self.rat = {f"R{i}": f"P{i}" for i in range(num_arch_regs)}

        # Free list: physical registers available for allocation
        self.free_list = deque(f"P{i}" for i in range(num_arch_regs, num_phys_regs))

        # Track old mappings for freeing on commit
        self.old_mappings = []  # (instruction_id, old_phys_reg)

    def rename_instruction(self, inst_id, op, rd, rs1, rs2=None):
        """Rename one instruction. Returns the renamed instruction."""
        # Step 1: Rename source registers (look up current mapping)
        phys_rs1 = self.rat[rs1]
        phys_rs2 = self.rat[rs2] if rs2 else None

        # Step 2: Allocate new physical register for destination
        if not self.free_list:
            raise RuntimeError("Out of physical registers! Pipeline stall.")
        old_phys = self.rat[rd]
        new_phys = self.free_list.popleft()

        # Step 3: Update RAT
        self.rat[rd] = new_phys
        self.old_mappings.append((inst_id, old_phys))

        # Build renamed instruction string
        if phys_rs2:
            renamed = f"{op} {new_phys}, {phys_rs1}, {phys_rs2}"
        else:
            renamed = f"{op} {new_phys}, {phys_rs1}"

        return {
            "id": inst_id,
            "original": f"{op} {rd}, {rs1}" + (f", {rs2}" if rs2 else ""),
            "renamed": renamed,
            "dest_old": old_phys,
            "dest_new": new_phys,
        }

    def print_rat(self):
        """Print current RAT state."""
        entries = [f"{arch}->{phys}" for arch, phys in sorted(self.rat.items())]
        print(f"  RAT: {', '.join(entries)}")
        print(f"  Free: [{', '.join(list(self.free_list)[:6])}{'...' if len(self.free_list) > 6 else ''}]")


def main():
    renamer = RegisterRenamer(num_arch_regs=8, num_phys_regs=20)

    # Program with WAR and WAW hazards
    instructions = [
        # (op, dest, src1, src2)
        ("ADD", "R1", "R2", "R3"),   # I1: R1 = R2 + R3
        ("MUL", "R4", "R1", "R5"),   # I2: R4 = R1 * R5  (RAW on R1)
        ("SUB", "R1", "R6", "R7"),   # I3: R1 = R6 - R7  (WAW I1, WAR I2)
        ("ADD", "R5", "R1", "R4"),   # I4: R5 = R1 + R4  (RAW on R1, R4)
        ("MUL", "R2", "R1", "R3"),   # I5: R2 = R1 * R3
    ]

    print("=" * 60)
    print("Register Renaming Simulation")
    print("=" * 60)
    print("\nInitial state:")
    renamer.print_rat()

    results = []
    for i, (op, rd, rs1, rs2) in enumerate(instructions):
        inst_id = f"I{i+1}"
        print(f"\n--- {inst_id}: {op} {rd}, {rs1}, {rs2} ---")
        result = renamer.rename_instruction(inst_id, op, rd, rs1, rs2)
        results.append(result)
        print(f"  Renamed: {result['renamed']}")
        print(f"  ({rd}: {result['dest_old']} -> {result['dest_new']})")
        renamer.print_rat()

    # Analyze dependencies after renaming
    print("\n" + "=" * 60)
    print("Dependency Analysis After Renaming")
    print("=" * 60)

    print("\nRenamed program:")
    for r in results:
        print(f"  {r['id']}: {r['renamed']}")

    print("\nTrue dependencies only (RAW):")
    print("  I2 depends on I1 (reads P8, produced by I1)")
    print("  I4 depends on I3 (reads P10) and I2 (reads P9)")
    print("  I5 depends on I3 (reads P10)")
    print("\nFalse dependencies eliminated:")
    print("  I1 vs I3: WAW on R1 -> now P8 vs P10 (independent!)")
    print("  I2 vs I3: WAR on R1 -> I2 reads P8, I3 writes P10 (independent!)")
    print("\nParallel execution possible: I1 || I3, then I2 || I5")


if __name__ == "__main__":
    main()
```

---

## 9. Practice Problems

### Basic Problems

1. Classify the data dependence types:
   ```assembly
   I1: ADD R1, R2, R3
   I2: SUB R4, R1, R5
   I3: MUL R1, R6, R7
   I4: DIV R8, R1, R9
   ```

2. What is the theoretical maximum IPC in a 3-way superscalar processor?

3. Which dependence types can register renaming resolve?
   - (a) RAW
   - (b) WAR
   - (c) WAW
   - (d) Control

### Intermediate Problems

4. Apply register renaming to this program:
   ```assembly
   I1: ADD R1, R2, R3
   I2: MUL R4, R1, R5
   I3: ADD R1, R6, R7
   I4: SUB R8, R1, R4
   ```
   (Use physical registers starting from P10)

5. List 3 reasons why ROB is needed.

6. What is the role of CDB (Common Data Bus) in Tomasulo algorithm?

### Advanced Problems

7. Trace Reservation Station states for this Tomasulo execution:
   ```assembly
   I1: LD   F2, 0(R1)     ; 3 cycles
   I2: MUL  F4, F2, F0    ; 5 cycles
   I3: ADD  F6, F4, F2    ; 2 cycles
   ```
   (Initial: F0 = 2.0, RS: Load1, Mult1, Add1, trace cycles 1-10)

8. Explain ROB-based recovery process on branch misprediction.

9. Intel Arrow Lake's Lion Cove P-core has 576 ROB entries (up from Skylake's 224). What are the effects of making the ROB larger or smaller, and why has the trend been toward larger ROBs?

<details>
<summary>Answers</summary>

1. Dependence classification:
   - I1→I2: RAW (R1)
   - I1→I3: WAW (R1)
   - I2→I3: WAR (R1 - I2 reads, I3 writes)
   - I3→I4: RAW (R1)

2. Theoretical max IPC of 3-way superscalar = 3

3. (b) WAR, (c) WAW
   - RAW is true dependence, cannot be resolved
   - Control resolved by branch prediction

4. Register renaming:
   ```assembly
   I1: ADD P10, P2, P3    ; R1 → P10
   I2: MUL P11, P10, P5   ; R4 → P11
   I3: ADD P12, P6, P7    ; R1 → P12 (new register)
   I4: SUB P13, P12, P11  ; R8 → P13
   ```
   I1 and I3 can now execute in parallel

5. ROB needed for:
   - Precise exception handling (in-order commit)
   - Branch misprediction recovery
   - Precise interrupt handling

6. CDB role:
   - Broadcast completed results to all RSs
   - Waiting instructions receive operands
   - Update register file

7. Tomasulo execution trace:
   ```
   Cycle 1: Issue I1 → Load1
   Cycle 2: Execute I1, Issue I2 → Mult1 (Qj=Load1)
   Cycle 3: Execute I1 continues, Issue I3 → Add1 (Qj=Mult1, Qk=Load1)
   Cycle 4: Write I1, Mult1: Vj update, Add1: Vk update
   Cycle 5: Execute I2 starts
   ...
   Cycle 9: Write I2, Add1: Vj update
   Cycle 10: Execute I3 starts
   Cycle 11-12: Execute I3 completes
   ```

8. Branch misprediction recovery:
   - Invalidate results after branch in ROB
   - Restore RAT to branch checkpoint state
   - Restart fetch at correct branch target
   - Pipeline flush

9. ROB size tradeoffs:
   - Larger: More ILP exploitation, tolerates longer memory latency, exploits wider issue
   - Smaller: Reduced area/power, faster recovery on misprediction
   - Trend toward larger ROBs: wider decode (8-wide+) requires more in-flight instructions to keep execution units fed; memory latency (~100+ cycles for DRAM) demands a large window to find independent work

</details>

---

## Next Steps

- [14_Memory_Hierarchy.md](./14_Memory_Hierarchy.md) - Memory System Hierarchy and Locality Principles

---

## References

- Computer Architecture: A Quantitative Approach (Hennessy & Patterson)
- [Intel Optimization Manual](https://www.intel.com/content/www/us/en/developer/articles/technical/intel-sdm.html)
- [WikiChip - Microarchitectures](https://en.wikichip.org/wiki/WikiChip)
- [Agner Fog's microarchitecture](https://www.agner.org/optimize/)
- Tomasulo, R.M. "An Efficient Algorithm for Exploiting Multiple Arithmetic Units" (1967)
