# Control Unit

**Previous**: [07_CPU_Architecture_Basics.md](./07_CPU_Architecture_Basics.md) | **Next**: [09_Instruction_Set_Architecture.md](./09_Instruction_Set_Architecture.md)

**Difficulty**: ⭐⭐⭐

**Prerequisites**: CPU Architecture Basics, Logic Circuits, State Machines

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain the role of the control unit in the instruction execution cycle
2. Distinguish hardwired control from microprogrammed control
3. Trace how an instruction's opcode maps to specific control signals
4. Describe the structure of a microinstruction (micro-ops, sequencing, condition)
5. Compare the tradeoffs between hardwired (speed) and microprogrammed (flexibility) control
6. Explain how control signals activate the datapath components

---

If the ALU is the calculator and registers are the scratch paper, the control unit is the brain that decides what to calculate and when. It transforms the abstract meaning of an instruction into the concrete electrical signals that make the hardware act. Understanding control unit design reveals why RISC processors can be simpler and faster than CISC designs.

## Table of Contents

1. [Role of the Control Unit](#1-role-of-the-control-unit)
2. [Hardwired Control](#2-hardwired-control)
3. [Microprogrammed Control](#3-microprogrammed-control)
4. [Control Signal Generation](#4-control-signal-generation)
5. [Microinstructions](#5-microinstructions)
6. [Practice Problems](#6-practice-problems)

---

> The control unit is like an orchestra conductor. It does not play any instrument itself, but its signals -- the tempo, the cue to start, the gesture to soften -- determine what every other part of the CPU does and when. Just as a conductor interprets a musical score (the instruction) and coordinates the violins (ALU), percussion (memory), and brass (registers) to produce a symphony, the control unit decodes each instruction and generates the precise control signals that orchestrate the datapath.

## 1. Role of the Control Unit

### 1.1 Basic Functions of the Control Unit

```
┌──────────────────────────────────────────────────────────────────────┐
│                      Role of Control Unit                             │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                    Inputs                                    │    │
│  │                                                             │    │
│  │    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐   │    │
│  │    │Instruction  │    │   Clock     │    │  Status     │   │    │
│  │    │  IR (Opcode,│    │             │    │  Signals    │   │    │
│  │    │   Funct)    │    │             │    │  (Flags,    │   │    │
│  │    └──────┬──────┘    └──────┬──────┘    │   Ready)    │   │    │
│  └───────────┼──────────────────┼──────────────────┬───────────┘    │
│              │                  │                  │                 │
│              ▼                  ▼                  ▼                 │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                      Control Unit                            │    │
│  │                                                             │    │
│  │    ┌────────────────────────────────────────────────┐      │    │
│  │    │       Control Logic / Microprogram             │      │    │
│  │    └────────────────────────────────────────────────┘      │    │
│  └─────────────────────────────────────────────────────────────┘    │
│              │                  │                  │                 │
│              ▼                  ▼                  ▼                 │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                Outputs - Control Signals                     │    │
│  │                                                             │    │
│  │    ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────────────┐ │    │
│  │    │ RegWrite│ │ ALUOp   │ │ MemRead │ │ MemWrite        │ │    │
│  │    │ ALUSrc  │ │ Branch  │ │ MemtoReg│ │ PCSrc           │ │    │
│  │    └─────────┘ └─────────┘ └─────────┘ └─────────────────┘ │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

### 1.2 Main Tasks of the Control Unit

| Task | Description |
|------|-------------|
| Instruction Decoding | Analyze opcode and function fields in IR to identify instruction type |
| Timing Generation | Determine execution sequence and timing for each operation |
| Control Signal Generation | Output signals to control each component in the datapath |
| State Management | Track current execution state and determine next state |
| Exception Handling | Detect and handle exceptional situations like interrupts and errors |

### 1.3 Control Signals and Datapath

```
┌───────────────────────────────────────────────────────────────────────────┐
│               Relationship between Control Signals and Datapath           │
│                                                                           │
│                          ┌────────────────┐                               │
│                          │  Control Unit  │                               │
│                          └───────┬────────┘                               │
│                                  │                                        │
│           ┌──────────────────────┼──────────────────────┐                 │
│           │                      │                      │                 │
│           ▼                      ▼                      ▼                 │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐           │
│  │Register Control │  │   ALU Control   │  │  Memory Control │           │
│  │                 │  │                 │  │                 │           │
│  │  - RegWrite     │  │  - ALUOp        │  │  - MemRead      │           │
│  │  - RegDst       │  │  - ALUSrc       │  │  - MemWrite     │           │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘           │
│           │                    │                    │                     │
│           ▼                    ▼                    ▼                     │
│  ┌─────────────────────────────────────────────────────────────┐         │
│  │                       Datapath                               │         │
│  │                                                             │         │
│  │  ┌──────────┐     ┌──────────┐     ┌──────────────────┐    │         │
│  │  │ Register │────►│   ALU    │────►│     Memory       │    │         │
│  │  │   File   │     │          │     │                  │    │         │
│  │  └──────────┘     └──────────┘     └──────────────────┘    │         │
│  │       ▲                                     │              │         │
│  │       └─────────────────────────────────────┘              │         │
│  │                    (Write Back)                            │         │
│  └─────────────────────────────────────────────────────────────┘         │
│                                                                           │
└───────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Hardwired Control

### 2.1 Hardwired Control Concept

Hardwired control implements control logic directly using combinational and sequential logic circuits.

```
┌─────────────────────────────────────────────────────────────────────┐
│                     Hardwired Control Unit                           │
│                                                                     │
│    IR [31:26]        ┌─────────────────────────────────────────┐   │
│    (Opcode)   ──────►│                                         │   │
│                      │     Combinational Logic                 │   │
│    IR [5:0]          │                                         │   │
│    (Funct)    ──────►│                                         │──► Control Signals
│                      │     - Decoders                          │   │
│    Status Flags      │     - Logic Gates (AND, OR)             │   │
│    (Flags)    ──────►│     - Multiplexers                      │   │
│                      └─────────────────────────────────────────┘   │
│                                    ▲                               │
│                                    │                               │
│                      ┌─────────────┴─────────────┐                 │
│                      │                           │                 │
│    ┌─────────────────┴───────────────────────────┴───────────┐    │
│    │                  Timing/Sequencer                       │    │
│    │       (State Registers, Counters, Decoders)            │    │
│    │                                                         │    │
│    │    ┌────────┐    ┌────────┐    ┌────────┐              │    │
│    │    │ State  │───►│ State  │───►│ Timing │              │    │
│    │    │Register│    │Decoder │    │ Signals│              │    │
│    │    └────────┘    └────────┘    └────────┘              │    │
│    └─────────────────────────────────────────────────────────┘    │
│                      ▲                                             │
│    Clock ────────────┘                                             │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 MIPS Single-Cycle Control Unit

```
┌─────────────────────────────────────────────────────────────────────┐
│              MIPS Single-Cycle Control (Simplified)                  │
│                                                                     │
│    Opcode [5:0]                                                     │
│        │                                                            │
│        ▼                                                            │
│    ┌───────────────────────────────────────────────────────────┐   │
│    │                      Main Decoder                          │   │
│    ├───────────────────────────────────────────────────────────┤   │
│    │  Opcode  │ RegDst │ ALUSrc │ MemtoReg │ RegWrite │ ...    │   │
│    ├──────────┼────────┼────────┼──────────┼──────────┼────────┤   │
│    │ R-type   │   1    │   0    │    0     │    1     │        │   │
│    │ (000000) │        │        │          │          │        │   │
│    ├──────────┼────────┼────────┼──────────┼──────────┼────────┤   │
│    │   lw     │   0    │   1    │    1     │    1     │        │   │
│    │ (100011) │        │        │          │          │        │   │
│    ├──────────┼────────┼────────┼──────────┼──────────┼────────┤   │
│    │   sw     │   x    │   1    │    x     │    0     │        │   │
│    │ (101011) │        │        │          │          │        │   │
│    ├──────────┼────────┼────────┼──────────┼──────────┼────────┤   │
│    │   beq    │   x    │   0    │    x     │    0     │        │   │
│    │ (000100) │        │        │          │          │        │   │
│    └──────────┴────────┴────────┴──────────┴──────────┴────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.3 ALU Control Unit

```
┌─────────────────────────────────────────────────────────────────────┐
│                        ALU Control Unit                              │
│                                                                     │
│    ALUOp (2-bit)      Funct (6-bit)                                 │
│        │                  │                                         │
│        │                  │                                         │
│        ▼                  ▼                                         │
│    ┌──────────────────────────────────────────┐                    │
│    │            ALU Control Unit              │                    │
│    │                                          │                    │
│    │    ┌────────┬────────┬────────────────┐ │                    │
│    │    │ ALUOp  │ Funct  │ ALU Operation  │ │                    │
│    │    ├────────┼────────┼────────────────┤ │                    │
│    │    │  00    │xxxxxx  │ Add (lw/sw)    │ │                    │
│    │    │  01    │xxxxxx  │ Sub (beq)      │ │                    │
│    │    │  10    │100000  │ Add            │ │                    │
│    │    │  10    │100010  │ Sub            │ │                    │
│    │    │  10    │100100  │ And            │ │                    │
│    │    │  10    │100101  │ Or             │ │                    │
│    │    │  10    │101010  │ Slt            │ │                    │
│    │    └────────┴────────┴────────────────┘ │                    │
│    └──────────────────────────────────────────┘                    │
│                    │                                                │
│                    ▼                                                │
│            ALU Control (4-bit)                                      │
│                                                                     │
│    ┌───────────────────────────────────────────────────────┐       │
│    │  ALU Control │     Operation      │                   │       │
│    ├──────────────┼────────────────────┤                   │       │
│    │     0000     │       AND          │                   │       │
│    │     0001     │       OR           │                   │       │
│    │     0010     │       Add          │                   │       │
│    │     0110     │       Sub          │                   │       │
│    │     0111     │       Slt          │                   │       │
│    │     1100     │       NOR          │                   │       │
│    └──────────────┴────────────────────┘                   │       │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.4 Multi-Cycle Hardwired Control

```
┌─────────────────────────────────────────────────────────────────────┐
│            Multi-Cycle Control: Finite State Machine (FSM)           │
│                                                                     │
│                          ┌──────────┐                               │
│                     ┌───►│  State 0 │◄────────────────┐             │
│                     │    │  (IF)    │                 │             │
│                     │    └────┬─────┘                 │             │
│                     │         │                       │             │
│                     │         ▼                       │             │
│                     │    ┌──────────┐                 │             │
│                     │    │  State 1 │                 │             │
│                     │    │  (ID)    │                 │             │
│                     │    └────┬─────┘                 │             │
│                     │         │                       │             │
│            ┌────────┴─────────┼─────────┬─────────────┤             │
│            │                  │         │             │             │
│            ▼                  ▼         ▼             │             │
│       ┌──────────┐      ┌──────────┐ ┌──────────┐    │             │
│       │  State 2 │      │  State 6 │ │  State 8 │    │             │
│       │ (MemAddr)│      │  (R-EX)  │ │  (BEQ)   │────┘             │
│       └────┬─────┘      └────┬─────┘ └──────────┘                   │
│            │                 │                                       │
│       ┌────┴────┐            │                                       │
│       │         │            │                                       │
│       ▼         ▼            ▼                                       │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐                             │
│  │  State 3 │ │  State 5 │ │  State 7 │                             │
│  │ (MemRead)│ │(MemWrite)│ │  (R-WB)  │                             │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘                             │
│       │            │            │                                    │
│       ▼            │            │                                    │
│  ┌──────────┐      │            │                                    │
│  │  State 4 │      │            │                                    │
│  │  (LW-WB) │      │            │                                    │
│  └────┬─────┘      │            │                                    │
│       │            │            │                                    │
│       └────────────┴────────────┴──────────────────────►(State 0)   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.5 Control Signals by State

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                   Control Signal Table by State                              │
├───────┬───────┬───────┬───────┬───────┬───────┬───────┬───────┬────────────┤
│ State │PCWrite│IRWrite│RegWrite│ALUSrcA│ALUSrcB│ALUOp │MemRead│Description │
├───────┼───────┼───────┼───────┼───────┼───────┼───────┼───────┼────────────┤
│   0   │   1   │   1   │   0   │   0   │  01   │  00   │   1   │ IF         │
│  (IF) │       │       │       │       │       │       │       │Instr Fetch │
├───────┼───────┼───────┼───────┼───────┼───────┼───────┼───────┼────────────┤
│   1   │   0   │   0   │   0   │   0   │  11   │  00   │   0   │ ID         │
│  (ID) │       │       │       │       │       │       │       │Instr Decode│
├───────┼───────┼───────┼───────┼───────┼───────┼───────┼───────┼────────────┤
│   2   │   0   │   0   │   0   │   1   │  10   │  00   │   0   │ MemAddr    │
│       │       │       │       │       │       │       │       │Addr Calc   │
├───────┼───────┼───────┼───────┼───────┼───────┼───────┼───────┼────────────┤
│   3   │   0   │   0   │   0   │   0   │  00   │  00   │   1   │ MemRead    │
│       │       │       │       │       │       │       │       │Memory Read │
├───────┼───────┼───────┼───────┼───────┼───────┼───────┼───────┼────────────┤
│   4   │   0   │   0   │   1   │   0   │  00   │  00   │   0   │ LW WB      │
│       │       │       │       │       │       │       │       │Load Write  │
├───────┼───────┼───────┼───────┼───────┼───────┼───────┼───────┼────────────┤
│   6   │   0   │   0   │   0   │   1   │  00   │  10   │   0   │ R-type EX  │
│       │       │       │       │       │       │       │       │R-type Exec │
├───────┼───────┼───────┼───────┼───────┼───────┼───────┼───────┼────────────┤
│   7   │   0   │   0   │   1   │   0   │  00   │  00   │   0   │ R-type WB  │
│       │       │       │       │       │       │       │       │R-type Write│
└───────┴───────┴───────┴───────┴───────┴───────┴───────┴───────┴────────────┘
```

### 2.6 Advantages and Disadvantages of Hardwired Control

| Advantages | Disadvantages |
|------------|---------------|
| Fast speed (only propagation delay) | High design complexity |
| Can be optimized | Difficult to modify/extend |
| Area efficient (for simple instruction sets) | Difficult to debug |
| Suitable for RISC processors | Requires circuit redesign for new instructions |

---

## 3. Microprogrammed Control

### 3.1 Microprogrammed Control Concept

Microprogrammed control stores control signals as a microprogram (firmware) and generates control signals by sequentially executing them.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                   Microprogrammed Control Unit                           │
│                                                                         │
│    IR (Opcode)                                                          │
│        │                                                                │
│        ▼                                                                │
│    ┌──────────────────┐                                                │
│    │   Mapping ROM    │                                                │
│    │ (Address Mapping)│                                                │
│    └────────┬─────────┘                                                │
│             │ Start Address                                             │
│             ▼                                                           │
│    ┌────────────────────────────────────────────────────────────┐      │
│    │                                                            │      │
│    │    ┌─────────────┐                                        │      │
│    │    │  MicroPC    │◄─────────────────────────────┐         │      │
│    │    │ (Micro      │                              │         │      │
│    │    │  Program    │                              │         │      │
│    │    │  Counter)   │                              │         │      │
│    │    └──────┬──────┘                              │         │      │
│    │           │                                      │         │      │
│    │           ▼                                      │         │      │
│    │    ┌────────────────────────────────────┐      │         │      │
│    │    │      Control Store (ROM)           │      │         │      │
│    │    │   (Microprogram Storage)           │      │         │      │
│    │    │                                    │      │         │      │
│    │    │   Address   Microinstruction       │      │         │      │
│    │    │   0x00    [Control Signals | Next] │      │         │      │
│    │    │   0x01    [Control Signals | Next] │      │         │      │
│    │    │   0x02    [Control Signals | Next] │      │         │      │
│    │    │    ...                             │      │         │      │
│    │    └───────────────┬────────────────────┘      │         │      │
│    │                    │                            │         │      │
│    │    ┌───────────────┴───────────────┐           │         │      │
│    │    │                               │           │         │      │
│    │    ▼                               ▼           │         │      │
│    │  ┌─────────────┐           ┌────────────────┐  │         │      │
│    │  │  Control    │           │   Sequencer    │──┘         │      │
│    │  │  Signals    │           │ (Next Address) │            │      │
│    │  │  (Output)   │           └────────────────┘            │      │
│    │  └─────────────┘                                         │      │
│    │                                                            │      │
│    └────────────────────────────────────────────────────────────┘      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Control Store Structure

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Control Store (ROM)                               │
│                                                                         │
│   Address   Microinstruction (e.g., 32-bit)                             │
│    ┌──────┬─────────────────────────────────────────────────────────┐   │
│    │      │ ALU │Reg │Mem │Mem │ALU │Reg │PC  │ IR │Next│Sequence  │   │
│    │ Addr │ Op  │Dst │Read│Wrt │Src │Wrt │Src │Wrt │Addr│ Control  │   │
│    ├──────┼─────┼────┼────┼────┼────┼────┼────┼────┼────┼──────────┤   │
│    │ 0x00 │ 00  │ x  │ 1  │ 0  │ 0  │ 0  │ 0  │ 1  │0x01│ SEQ      │   │
│    │      │     │    │    │    │    │    │    │    │    │(Sequential)│   │
│    ├──────┼─────┼────┼────┼────┼────┼────┼────┼────┼────┼──────────┤   │
│    │ 0x01 │ 00  │ x  │ 0  │ 0  │ 0  │ 0  │ x  │ 0  │DISP│ DISPATCH │   │
│    │      │     │    │    │    │    │    │    │    │    │ (Branch) │   │
│    ├──────┼─────┼────┼────┼────┼────┼────┼────┼────┼────┼──────────┤   │
│    │ 0x02 │ 00  │ 0  │ 0  │ 0  │ 1  │ 0  │ x  │ 0  │0x03│ SEQ      │   │
│    │      │     │    │    │    │    │    │    │    │    │ (lw/sw)  │   │
│    ├──────┼─────┼────┼────┼────┼────┼────┼────┼────┼────┼──────────┤   │
│    │ 0x03 │ xx  │ x  │ 1  │ 0  │ x  │ 0  │ x  │ 0  │0x04│ SEQ      │   │
│    │      │     │    │    │    │    │    │    │    │    │ (lw mem) │   │
│    ├──────┼─────┼────┼────┼────┼────┼────┼────┼────┼────┼──────────┤   │
│    │ 0x04 │ xx  │ 0  │ 0  │ 0  │ x  │ 1  │ x  │ 0  │0x00│ FETCH    │   │
│    │      │     │    │    │    │    │    │    │    │    │ (lw wb)  │   │
│    ├──────┼─────┼────┼────┼────┼────┼────┼────┼────┼────┼──────────┤   │
│    │ ...  │     │    │    │    │    │    │    │    │    │          │   │
│    └──────┴─────┴────┴────┴────┴────┴────┴────┴────┴────┴──────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.3 Sequencer Operation

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Sequencer                                        │
│                                                                         │
│   Determines next address based on sequence control field of current    │
│   microinstruction                                                      │
│                                                                         │
│    ┌────────────────────────────────────────────────────────────┐       │
│    │ Sequence Control│        Next Address Determination        │       │
│    ├─────────────────┼────────────────────────────────────────┤       │
│    │    SEQ          │  MicroPC ← MicroPC + 1                 │       │
│    │ (Sequential)    │  Proceed to next microinstruction      │       │
│    ├─────────────────┼────────────────────────────────────────┤       │
│    │   BRANCH        │  MicroPC ← Next address field          │       │
│    │   (Branch)      │  Jump to specified address             │       │
│    ├─────────────────┼────────────────────────────────────────┤       │
│    │   DISPATCH      │  MicroPC ← Mapping ROM[IR.Opcode]      │       │
│    │   (Dispatch)    │  Branch to routine based on instr type │       │
│    ├─────────────────┼────────────────────────────────────────┤       │
│    │   FETCH         │  MicroPC ← 0 (Return to IF state)      │       │
│    │   (Fetch)       │  Start next instruction fetch          │       │
│    └─────────────────┴────────────────────────────────────────┘       │
│                                                                         │
│    Next address selection via MUX:                                      │
│                                                                         │
│    MicroPC+1 ────►┌───────┐                                            │
│    Branch Addr ──►│  MUX  │────► Next MicroPC                          │
│    Dispatch Addr─►│       │                                            │
│    0 (Fetch) ────►└───────┘                                            │
│                      ▲                                                  │
│                      │                                                  │
│              Sequence Control Signal                                    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.4 Advantages and Disadvantages of Microprogrammed Control

| Advantages | Disadvantages |
|------------|---------------|
| Flexible design (only ROM needs modification) | Slower than hardwired |
| Easy to implement complex instructions | ROM access delay |
| Easy to debug | Additional hardware (Control Store) needed |
| Suitable for CISC processors | Increased power consumption |
| Microcode can be updated | Increased area |

### 3.5 Real-World Examples

```
┌─────────────────────────────────────────────────────────────────────────┐
│              Microcode Usage in Real Processors                          │
│                                                                         │
│    Intel x86 family:                                                    │
│    ┌───────────────────────────────────────────────────────────────┐   │
│    │  - Implements complex CISC instructions with microcode        │   │
│    │  - Simple instructions handled quickly with hardwired logic   │   │
│    │  - Microcode updates for security patches (e.g., Spectre/    │   │
│    │    Meltdown)                                                  │   │
│    └───────────────────────────────────────────────────────────────┘   │
│                                                                         │
│    AMD processors:                                                      │
│    ┌───────────────────────────────────────────────────────────────┐   │
│    │  - Similarly handles complex instructions with microcode      │   │
│    │  - Microcode updates via BIOS/UEFI                            │   │
│    └───────────────────────────────────────────────────────────────┘   │
│                                                                         │
│    Hybrid approach (most modern processors):                            │
│    ┌───────────────────────────────────────────────────────────────┐   │
│    │  - Frequently used instructions: Hardwired (fast)             │   │
│    │  - Complex instructions: Microcode (flexibility)              │   │
│    │  - Example: x86 ADD is hardwired, REP MOVS is microcode       │   │
│    └───────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Control Signal Generation

### 4.1 Main Control Signals List

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Main Control Signals                             │
├──────────────┬──────────────────────────────────────────────────────────┤
│ Control Signal│                    Function                              │
├──────────────┼──────────────────────────────────────────────────────────┤
│   RegDst     │ Select write register (rt=0, rd=1)                       │
├──────────────┼──────────────────────────────────────────────────────────┤
│   RegWrite   │ Enable writing to register file                          │
├──────────────┼──────────────────────────────────────────────────────────┤
│   ALUSrc     │ Select second ALU input (Register=0, Immediate=1)        │
├──────────────┼──────────────────────────────────────────────────────────┤
│   ALUOp      │ Specify ALU operation (00=add, 01=sub, 10=R-type)        │
├──────────────┼──────────────────────────────────────────────────────────┤
│   MemRead    │ Enable memory read                                       │
├──────────────┼──────────────────────────────────────────────────────────┤
│   MemWrite   │ Enable memory write                                      │
├──────────────┼──────────────────────────────────────────────────────────┤
│   MemtoReg   │ Select data to write to register (ALU=0, Memory=1)       │
├──────────────┼──────────────────────────────────────────────────────────┤
│   PCSrc      │ Select PC value (PC+4=0, Branch target=1)                │
├──────────────┼──────────────────────────────────────────────────────────┤
│   Branch     │ Is conditional branch instruction                        │
├──────────────┼──────────────────────────────────────────────────────────┤
│   Jump       │ Is unconditional jump instruction                        │
└──────────────┴──────────────────────────────────────────────────────────┘
```

### 4.2 Control Signal Values by Instruction

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                   Control Signal Truth Table by Instruction                      │
├──────────┬───────┬────────┬───────┬───────┬────────┬─────────┬────────┬────────┤
│Instruction│RegDst │ALUSrc  │MemtoReg│RegWrite│MemRead│MemWrite│Branch  │ALUOp   │
├──────────┼───────┼────────┼───────┼───────┼────────┼─────────┼────────┼────────┤
│ R-type   │   1   │   0    │   0   │   1   │   0    │    0    │   0    │  10    │
│ (add,sub)│       │        │       │       │        │         │        │        │
├──────────┼───────┼────────┼───────┼───────┼────────┼─────────┼────────┼────────┤
│   lw     │   0   │   1    │   1   │   1   │   1    │    0    │   0    │  00    │
├──────────┼───────┼────────┼───────┼───────┼────────┼─────────┼────────┼────────┤
│   sw     │   x   │   1    │   x   │   0   │   0    │    1    │   0    │  00    │
├──────────┼───────┼────────┼───────┼───────┼────────┼─────────┼────────┼────────┤
│   beq    │   x   │   0    │   x   │   0   │   0    │    0    │   1    │  01    │
├──────────┼───────┼────────┼───────┼───────┼────────┼─────────┼────────┼────────┤
│  addi    │   0   │   1    │   0   │   1   │   0    │    0    │   0    │  00    │
├──────────┼───────┼────────┼───────┼───────┼────────┼─────────┼────────┼────────┤
│    j     │   x   │   x    │   x   │   0   │   0    │    0    │   x    │  xx    │
└──────────┴───────┴────────┴───────┴───────┴────────┴─────────┴────────┴────────┘

x = don't care (any value is acceptable)
```

### 4.3 Control Signal Generation Logic Circuit

```
┌─────────────────────────────────────────────────────────────────────────┐
│                Control Signal Generation Logic Circuit                   │
│                                                                         │
│    Opcode [5:0]                                                         │
│       │                                                                 │
│       ├─────────────────────────────────────────────────────────────┐   │
│       │                                                             │   │
│       │   ┌─────────────────────────────────────────────────────┐   │   │
│       │   │                    Decoder                           │   │   │
│       │   │                                                     │   │   │
│       │   │   Opcode    R-type  lw   sw   beq   addi   j       │   │   │
│       │   │   000000  ─────►                                    │   │   │
│       │   │   100011  ─────────►                                │   │   │
│       │   │   101011  ──────────────►                           │   │   │
│       │   │   000100  ───────────────────►                      │   │   │
│       │   │   001000  ────────────────────────►                 │   │   │
│       │   │   000010  ─────────────────────────────►            │   │   │
│       │   └─────────────────────────────────────────────────────┘   │   │
│       │           │       │      │      │       │       │           │   │
│       │           ▼       ▼      ▼      ▼       ▼       ▼           │   │
│       │                                                             │   │
│       │   RegDst  = R-type                                          │   │
│       │   ALUSrc  = lw | sw | addi                                  │   │
│       │   MemtoReg = lw                                             │   │
│       │   RegWrite = R-type | lw | addi                             │   │
│       │   MemRead  = lw                                             │   │
│       │   MemWrite = sw                                             │   │
│       │   Branch   = beq                                            │   │
│       │   ALUOp[1] = R-type                                         │   │
│       │   ALUOp[0] = beq                                            │   │
│       │   Jump     = j                                              │   │
│       │                                                             │   │
│       └─────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.4 Timing Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│               LW Instruction Execution Timing (Single-Cycle)             │
│                                                                         │
│   CLK    ────┐     ┌─────────────────────────────┐     ┌─────          │
│              │     │                             │     │               │
│              └─────┘                             └─────┘               │
│                                                                         │
│   PC     ════╪═════════════════════════════════════════╪═══            │
│          0x04│                                         │0x08           │
│                                                                         │
│   IR     ────╪═════════════════════════════════════════╪───            │
│              │          lw $t0, 4($s0)                 │               │
│                                                                         │
│   RegRead ───────────────────────────────────────────────────          │
│              │    ▼ Read $s0 value                                     │
│                                                                         │
│   ALU    ────────────────────────────────────────────────────          │
│                      │    ▼ Calculate $s0 + 4                          │
│                                                                         │
│   MemRead ───────────────────────────────────────────────────          │
│                              │    ▼ Read Memory[$s0+4]                 │
│                                                                         │
│   RegWrite ──────────────────────────────────────────────────          │
│                                      │    ▼ Write to $t0               │
│                                                                         │
│          ├───IF───┼───ID───┼───EX───┼───MEM──┼───WB───┤                │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Microinstructions

### 5.1 Microinstruction Format

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Microinstruction Format                              │
│                                                                         │
│    ┌────────────────────────────────────────────────────────────────┐  │
│    │              Horizontal Microinstruction                        │  │
│    │                                                                │  │
│    │  ┌────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────────┐ │  │
│    │  │ALU │Reg │Reg │Mem │Mem │ALU │ALU │PC  │ IR │Seq │ Next   │ │  │
│    │  │Op  │Dst │Wrt │Rd  │Wrt │SrcA│SrcB│Wrt │Wrt │Ctl │ Addr   │ │  │
│    │  │4bit│1bit│1bit│1bit│1bit│1bit│2bit│1bit│1bit│2bit│ 6bit   │ │  │
│    │  └────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────────┘ │  │
│    │                                                                │  │
│    │  Features: All control signals explicitly encoded             │  │
│    │       - Fast decoding                                         │  │
│    │       - Wide bit width (requires more ROM)                    │  │
│    │       - Parallel operations possible                          │  │
│    └────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│    ┌────────────────────────────────────────────────────────────────┐  │
│    │              Vertical Microinstruction                          │  │
│    │                                                                │  │
│    │  ┌────────────┬────────────┬────────────┬──────────────────┐  │  │
│    │  │  Operation │  Source    │Destination │  Next Address    │  │  │
│    │  │   (op)     │  (src)     │   (dst)    │                  │  │  │
│    │  │   4bit     │   4bit     │   4bit     │     6bit         │  │  │
│    │  └────────────┴────────────┴────────────┴──────────────────┘  │  │
│    │                                                                │  │
│    │  Features: Control signals represented in encoded format      │  │
│    │       - Narrow bit width (less ROM)                           │  │
│    │       - Additional decoding required (slower)                 │  │
│    │       - One operation at a time                               │  │
│    └────────────────────────────────────────────────────────────────┘  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 5.2 Microinstruction Example (LW Instruction)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                 LW $t0, offset($s0) Microprogram                         │
│                                                                         │
│   Address   Microinstruction                   Description              │
│   ────────────────────────────────────────────────────────────────      │
│                                                                         │
│   0x00:  PCWrite=1, IRWrite=1, ALUSrcA=0,     // Instruction Fetch(IF) │
│          ALUSrcB=01, ALUOp=00, MemRead=1,                               │
│          NextAddr=0x01, SeqCtl=SEQ                                      │
│                                                                         │
│   0x01:  ALUSrcA=0, ALUSrcB=11, ALUOp=00,     // Instruction Decode(ID)│
│          NextAddr=DISPATCH, SeqCtl=DISPATCH                             │
│                                                                         │
│   --- lw instruction dispatch →                                         │
│                                                                         │
│   0x10:  ALUSrcA=1, ALUSrcB=10, ALUOp=00      // Address calculation   │
│          NextAddr=0x11, SeqCtl=SEQ            // A = $s0 + offset      │
│                                                                         │
│   0x11:  MemRead=1                             // Memory read           │
│          NextAddr=0x12, SeqCtl=SEQ            // MDR ← Mem[A]          │
│                                                                         │
│   0x12:  RegDst=0, RegWrite=1, MemtoReg=1     // Write Back (WB)       │
│          NextAddr=0x00, SeqCtl=FETCH          // $t0 ← MDR             │
│                                                                         │
│   Micro-operation sequence:                                             │
│   1. MAR ← PC; IR ← Mem[MAR]; PC ← PC + 4                              │
│   2. A ← Regs[$s0]                                                      │
│   3. ALUOut ← A + sign_extend(offset)                                  │
│   4. MDR ← Mem[ALUOut]                                                 │
│   5. Regs[$t0] ← MDR                                                   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 5.3 Microinstruction Sequencing

```
┌─────────────────────────────────────────────────────────────────────────┐
│                   Microinstruction Sequencing Example                    │
│                                                                         │
│    Start                                                                │
│      │                                                                  │
│      ▼                                                                  │
│    ┌────────┐                                                          │
│    │ 0x00   │  IF: Instruction fetch                                   │
│    │ (IF)   │  MAR←PC, IR←Mem[MAR], PC←PC+4                            │
│    └───┬────┘                                                          │
│        │ SEQ                                                           │
│        ▼                                                                │
│    ┌────────┐                                                          │
│    │ 0x01   │  ID: Instruction decode, register read                   │
│    │ (ID)   │  A←Regs[rs], B←Regs[rt]                                  │
│    └───┬────┘                                                          │
│        │ DISPATCH (based on opcode)                                    │
│        │                                                                │
│   ┌────┴────┬────────────┬────────────┬────────────┐                  │
│   │         │            │            │            │                   │
│   ▼         ▼            ▼            ▼            ▼                   │
│ ┌──────┐ ┌──────┐    ┌──────┐    ┌──────┐    ┌──────┐                 │
│ │ 0x10 │ │ 0x20 │    │ 0x30 │    │ 0x40 │    │ 0x50 │                 │
│ │ lw   │ │ sw   │    │R-type│    │ beq  │    │  j   │                 │
│ └──┬───┘ └──┬───┘    └──┬───┘    └──┬───┘    └──┬───┘                 │
│    │        │           │           │           │                      │
│    ▼        ▼           ▼           │           │                      │
│ ┌──────┐ ┌──────┐    ┌──────┐      │           │                      │
│ │ 0x11 │ │ 0x21 │    │ 0x31 │      │           │                      │
│ │MemRd │ │MemWr │    │ R-WB │      │           │                      │
│ └──┬───┘ └──┬───┘    └──┬───┘      │           │                      │
│    │        │           │           │           │                      │
│    ▼        │           │           │           │                      │
│ ┌──────┐    │           │           │           │                      │
│ │ 0x12 │    │           │           │           │                      │
│ │ LW-WB│    │           │           │           │                      │
│ └──┬───┘    │           │           │           │                      │
│    │        │           │           │           │                      │
│    └────────┴───────────┴───────────┴───────────┴───►(Return to 0x00) │
│                          FETCH                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 5.4 Horizontal vs Vertical Microinstruction Comparison

| Feature | Horizontal | Vertical |
|---------|-----------|----------|
| Bit Width | Wide (50-100 bits) | Narrow (16-32 bits) |
| Decoding | Not needed/minimal | Additional decoding required |
| Speed | Fast | Slow |
| Parallelism | High (simultaneous control) | Low (sequential) |
| ROM Size | Large | Small |
| Flexibility | High | Medium |

### 5.5 Nanoprogramming

```
┌─────────────────────────────────────────────────────────────────────────┐
│                       Nanoprogramming                                    │
│                                                                         │
│   Uses two-level control storage to optimize Control Store size         │
│                                                                         │
│    ┌─────────────────────────────────────────────────────────────────┐ │
│    │                                                                 │ │
│    │   MicroPC ───►┌─────────────────────┐                          │ │
│    │               │   Control Store     │                          │ │
│    │               │   (Micro ROM)       │                          │ │
│    │               │                     │                          │ │
│    │               │   ┌──────┬────────┐ │                          │ │
│    │               │   │NanoPC│NextAddr│ │──► Sequence Control      │ │
│    │               │   └──┬───┴────────┘ │                          │ │
│    │               └──────┼──────────────┘                          │ │
│    │                      │                                          │ │
│    │                      ▼                                          │ │
│    │               ┌─────────────────────┐                          │ │
│    │               │   Nano Store        │                          │ │
│    │               │   (Nano ROM)        │                          │ │
│    │               │                     │                          │ │
│    │               │ Actual control      │──► Control Signal Output │ │
│    │               │ signal encoding     │                          │ │
│    │               └─────────────────────┘                          │ │
│    │                                                                 │ │
│    └─────────────────────────────────────────────────────────────────┘ │
│                                                                         │
│    Advantage: Save ROM by sharing repeated control signal patterns      │
│    Disadvantage: Additional access delay                                │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Practice Problems

### Basic Problems

1. Explain three main roles of the control unit.

2. Compare the differences between hardwired control and microprogrammed control.

3. Explain the role of the following control signals:
   - (a) RegWrite
   - (b) ALUSrc
   - (c) MemtoReg

### Control Signal Analysis Problems

4. Determine the value of each control signal when executing the following MIPS instruction:

   `ADD $t0, $t1, $t2`

   - RegDst = ?
   - ALUSrc = ?
   - MemtoReg = ?
   - RegWrite = ?
   - MemRead = ?
   - MemWrite = ?
   - Branch = ?
   - ALUOp = ?

5. Determine all control signals for the `SW $t0, 100($s0)` instruction.

### Microprogram Problems

6. Explain the role of the sequencer in microprogrammed control.

7. Compare the advantages and disadvantages of horizontal and vertical microinstructions.

8. Write the microinstruction sequence for the `BEQ $t0, $t1, label` instruction.

### Advanced Problems

9. Explain which control method (hardwired/microprogrammed) is more suitable for the following situations:
   - (a) Designing a simple RISC processor
   - (b) Designing a complex CISC processor
   - (c) When microcode updates are needed

10. List all states that the lw instruction goes through in a multi-cycle CPU FSM (Finite State Machine), and explain the control signals activated in each state.

<details>
<summary>Answers</summary>

1. Main roles of the control unit:
   - Instruction decoding: Analyze IR opcode to identify instruction type
   - Timing generation: Determine execution sequence and timing for each operation
   - Control signal generation: Output signals to control each component in datapath

2. Comparison:
   - Hardwired: Implemented directly with logic circuits, fast but difficult to modify
   - Microprogrammed: Executes program stored in ROM, flexible but slower

3. Control signal roles:
   - (a) RegWrite: Enable writing data to register file
   - (b) ALUSrc: Select second ALU input source (0=register, 1=immediate)
   - (c) MemtoReg: Select data to write to register (0=ALU, 1=memory)

4. ADD instruction control signals:
   - RegDst = 1 (use rd)
   - ALUSrc = 0 (register)
   - MemtoReg = 0 (ALU result)
   - RegWrite = 1
   - MemRead = 0
   - MemWrite = 0
   - Branch = 0
   - ALUOp = 10 (R-type)

5. SW instruction control signals:
   - RegDst = x (don't care)
   - ALUSrc = 1 (immediate 100)
   - MemtoReg = x
   - RegWrite = 0
   - MemRead = 0
   - MemWrite = 1
   - Branch = 0
   - ALUOp = 00 (add)

6. Sequencer role:
   - Interprets sequence control field of current microinstruction
   - Determines next microinstruction address (sequential, branch, dispatch, fetch)
   - Updates MicroPC value

7. Horizontal vs Vertical:
   - Horizontal: Fast execution, parallel control possible, wide bit width, large ROM
   - Vertical: Narrow bit width, small ROM, additional decoding required, slower

8. BEQ microinstructions:
   - IF: MAR←PC, IR←Mem[MAR], PC←PC+4
   - ID: A←Regs[$t0], B←Regs[$t1], ALUOut←PC+(offset<<2)
   - EX: if (A==B) PC←ALUOut → 0x00 (return to IF)

9.
   - (a) RISC: Hardwired (instructions are simple and speed is important)
   - (b) CISC: Microprogrammed (easy to implement complex instructions)
   - (c) Updates needed: Microprogrammed (only ROM needs modification)

10. lw instruction states:
    - State 0 (IF): MemRead=1, IRWrite=1, PCWrite=1
    - State 1 (ID): (register read)
    - State 2 (MemAddr): ALUSrcA=1, ALUSrcB=10, ALUOp=00
    - State 3 (MemRead): MemRead=1
    - State 4 (LW-WB): RegDst=0, MemtoReg=1, RegWrite=1

</details>

---

## Next Steps

- [09_Instruction_Set_Architecture.md](./09_Instruction_Set_Architecture.md) - CISC vs RISC, Addressing Modes

---

## References

- Computer Organization and Design (Patterson & Hennessy)
- Computer Architecture: A Quantitative Approach (Hennessy & Patterson)
- [MIPS Control Unit Implementation](https://www.cs.cornell.edu/courses/cs3410/2019sp/schedule/slides/11-control-mc.pdf)
- Digital Design and Computer Architecture (Harris & Harris)
