# CPU Architecture Basics

**Previous**: [06_Sequential_Logic.md](./06_Sequential_Logic.md) | **Next**: [08_Control_Unit.md](./08_Control_Unit.md)

**Difficulty**: ⭐⭐

**Prerequisites**: Logic Gates, Sequential Logic Circuits, Combinational Logic Circuits

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Identify the major components of a CPU (ALU, registers, control unit, buses)
2. Trace the fetch-decode-execute cycle step by step
3. Explain the role of the program counter (PC), instruction register (IR), and MAR/MBR
4. Describe how the datapath connects CPU components
5. Distinguish between register-register and register-memory architectures
6. Compare single-cycle and multi-cycle CPU implementations
7. Calculate execution time and CPI for different CPU designs

---

The CPU is the brain of every computer -- but unlike a human brain, its operation can be understood completely by tracing signals through a handful of components. Understanding CPU architecture explains why some programs run faster than others, why instruction sets matter, and how hardware and software co-evolve. In this lesson, we examine the internal components of the CPU, types of registers, the datapath structure, and the instruction execution cycle in detail.

## Table of Contents

1. [CPU Components](#1-cpu-components)
2. [Register Types](#2-register-types)
3. [Datapath](#3-datapath)
4. [Instruction Execution Cycle Details](#4-instruction-execution-cycle-details)
5. [Single-Cycle vs Multi-Cycle](#5-single-cycle-vs-multi-cycle)
6. [Practice Problems](#6-practice-problems)

---

## 1. CPU Components

### 1.1 Overall CPU Structure

```
┌──────────────────────────────────────────────────────────────────┐
│                            CPU                                    │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │                    Control Unit                             │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │  │
│  │  │ Instruction │  │   Control   │  │  Timing & Sequencing│ │  │
│  │  │   Decoder   │  │   Signal    │  │    (Sequencer)      │ │  │
│  │  │             │  │  Generator  │  │                     │ │  │
│  │  └─────────────┘  └─────────────┘  └─────────────────────┘ │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                   │
│  ┌─────────────────────┐    ┌─────────────────────────────────┐  │
│  │   Register File      │    │      ALU (Arithmetic Logic      │  │
│  │  ┌─────┐ ┌─────┐   │    │            Unit)                │  │
│  │  │ R0  │ │ R1  │   │    │  ┌─────────────────────────┐    │  │
│  │  ├─────┤ ├─────┤   │    │  │   Arithmetic Unit       │    │  │
│  │  │ R2  │ │ R3  │   │    │  │   (+, -, *, /)          │    │  │
│  │  ├─────┤ ├─────┤   │    │  ├─────────────────────────┤    │  │
│  │  │ ... │ │ ... │   │    │  │   Logic Unit            │    │  │
│  │  ├─────┤ ├─────┤   │    │  │   (AND, OR, XOR, NOT)   │    │  │
│  │  │ PC  │ │ IR  │   │    │  ├─────────────────────────┤    │  │
│  │  └─────┘ └─────┘   │    │  │   Shift Unit            │    │  │
│  │                     │    │  │   (<<, >>)              │    │  │
│  └─────────────────────┘    └─────────────────────────────────┘  │
│                                                                   │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │                   Internal Bus                              │  │
│  └────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
                              │
                    ┌─────────┴─────────┐
                    │    System Bus     │
                    └───────────────────┘
```

### 1.2 ALU (Arithmetic Logic Unit)

The ALU is the core component in the CPU that performs actual computations.

```
                    ┌──────────────┐
        A ─────────►│              │
        (Operand 1) │              │
                    │     ALU      │────────► Result
        B ─────────►│              │
        (Operand 2) │              │────────► Status Flags
                    │              │          (Zero, Carry,
        Op Code  ───►│              │           Overflow, Sign)
                    └──────────────┘
```

#### Operations Performed by ALU

| Operation Type | Operation | Description |
|----------------|-----------|-------------|
| Arithmetic | ADD | Addition |
| | SUB | Subtraction |
| | MUL | Multiplication |
| | DIV | Division |
| | INC | Increment by 1 |
| | DEC | Decrement by 1 |
| Logical | AND | Logical AND |
| | OR | Logical OR |
| | XOR | Exclusive OR |
| | NOT | Logical NOT |
| Shift | SHL | Shift Left |
| | SHR | Shift Right (Logical) |
| | SAR | Shift Right (Arithmetic) |
| Comparison | CMP | Compare (subtract and set flags) |

#### Status Flags Register

```
┌───┬───┬───┬───┬───┬───┬───┬───┐
│ S │ Z │ - │ A │ - │ P │ - │ C │
└───┴───┴───┴───┴───┴───┴───┴───┘
  │   │       │       │       │
  │   │       │       │       └─ Carry (carry/borrow)
  │   │       │       └───────── Parity
  │   │       └───────────────── Auxiliary Carry
  │   └───────────────────────── Zero (result is 0)
  └───────────────────────────── Sign (negative if 1)

Additional flags:
- Overflow (O): Overflow occurred
- Interrupt (I): Interrupt enable
- Direction (D): String operation direction
```

### 1.3 Control Unit

The control unit decodes instructions and sends appropriate control signals to each component.

```
┌─────────────────────────────────────────────────────────────┐
│                      Control Unit (CU)                       │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Instruction Register (IR)               │   │
│  │    ┌────────────┬────────────┬────────────────┐     │   │
│  │    │  Opcode    │   Rs/Rd    │   Immediate    │     │   │
│  │    │ (Operation)│ (Registers)│    (Value)     │     │   │
│  │    └────────────┴────────────┴────────────────┘     │   │
│  └─────────────────────────────────────────────────────┘   │
│                           │                                 │
│                           ▼                                 │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Instruction Decoder                     │   │
│  └─────────────────────────────────────────────────────┘   │
│                           │                                 │
│           ┌───────────────┼───────────────┐                │
│           ▼               ▼               ▼                │
│     ┌──────────┐   ┌──────────┐   ┌──────────┐            │
│     │ ALU      │   │ Register │   │ Memory   │            │
│     │ Control  │   │ Control  │   │ Control  │            │
│     └──────────┘   └──────────┘   └──────────┘            │
└─────────────────────────────────────────────────────────────┘
```

#### Control Signal Examples

| Control Signal | Function |
|----------------|----------|
| RegWrite | Enable register write |
| ALUSrc | Select ALU input source |
| ALUOp | Specify ALU operation type |
| MemRead | Enable memory read |
| MemWrite | Enable memory write |
| MemtoReg | Select memory-to-register path |
| Branch | Branch decision |
| Jump | Jump decision |

---

## 2. Register Types

### 2.1 General Purpose Registers

Registers that programmers can use freely.

```
┌─────────────────────────────────────────────────────────────────┐
│                 General Purpose Registers (x86-64)               │
├─────────────┬─────────┬─────────┬──────────────────────────────┤
│   64-bit    │  32-bit │  16-bit │           Purpose            │
├─────────────┼─────────┼─────────┼──────────────────────────────┤
│    RAX      │   EAX   │   AX    │ Accumulator (arithmetic)     │
│    RBX      │   EBX   │   BX    │ Base (memory address base)   │
│    RCX      │   ECX   │   CX    │ Counter (loop counter)       │
│    RDX      │   EDX   │   DX    │ Data (I/O operations)        │
│    RSI      │   ESI   │   SI    │ Source Index                 │
│    RDI      │   EDI   │   DI    │ Destination Index            │
│    RBP      │   EBP   │   BP    │ Base Pointer (stack frame)   │
│    RSP      │   ESP   │   SP    │ Stack Pointer                │
│   R8-R15    │ R8D-R15D│ R8W-R15W│ Additional GPRs (x64)        │
└─────────────┴─────────┴─────────┴──────────────────────────────┘
```

#### Register Size Relationships (x86)

```
64-bit: ├──────────────────────────────────────────────────────────┤ RAX
32-bit:                                 ├──────────────────────────┤ EAX
16-bit:                                                 ├──────────┤ AX
 8-bit:                                                 ├────┬─────┤
                                                         AH    AL
```

### 2.2 Special Purpose Registers

Registers essential for processor operation.

```
┌─────────────────────────────────────────────────────────────────┐
│                   Special Purpose Registers                      │
├──────────────┬──────────────────────────────────────────────────┤
│   Register   │                    Role                          │
├──────────────┼──────────────────────────────────────────────────┤
│ PC (Program  │ Stores address of next instruction to execute    │
│  Counter)    │ Automatically incremented after fetch            │
├──────────────┼──────────────────────────────────────────────────┤
│ IR (Instruc- │ Stores the currently executing instruction       │
│ tion Reg.)   │ Target for control unit decoding                 │
├──────────────┼──────────────────────────────────────────────────┤
│ MAR (Memory  │ Stores memory address to access                  │
│ Address Reg.)│ Connected to address bus                         │
├──────────────┼──────────────────────────────────────────────────┤
│ MBR/MDR      │ Stores data to read or write from memory         │
│ (Memory Data)│ Connected to data bus                            │
├──────────────┼──────────────────────────────────────────────────┤
│ SP (Stack    │ Stores address of top of stack                   │
│  Pointer)    │ Automatically adjusted during PUSH/POP           │
├──────────────┼──────────────────────────────────────────────────┤
│ PSW/FLAGS    │ Stores processor status information              │
│ (Status Reg.)│ Flags like Zero, Carry, Overflow                 │
└──────────────┴──────────────────────────────────────────────────┘
```

#### PC (Program Counter) Operation

```
Memory:                              PC Operation:
┌─────────────┐
│ 0x1000: ADD │  ◄─── PC = 0x1000 (current)
├─────────────┤
│ 0x1004: SUB │  ◄─── PC = 0x1004 (next)
├─────────────┤
│ 0x1008: MUL │  ◄─── PC = 0x1008
├─────────────┤
│ 0x100C: JMP │  ◄─── PC value changes on branch
├─────────────┤
│    ...      │
└─────────────┘

Sequential execution: PC = PC + instruction size
Branch execution: PC = branch target address
```

### 2.3 ARM Register Structure

```
┌─────────────────────────────────────────────────────────────────┐
│                      ARM Registers (AArch64)                     │
├──────────────┬──────────────────────────────────────────────────┤
│   Register   │                    Role                          │
├──────────────┼──────────────────────────────────────────────────┤
│   X0 - X7    │ Argument passing and return value registers      │
│   X8         │ Indirect result register                         │
│   X9 - X15   │ Temporary registers (caller-saved)               │
│  X16 - X17   │ Intra-procedure call registers                   │
│   X18        │ Platform register                                │
│  X19 - X28   │ Callee-saved registers                           │
│   X29 (FP)   │ Frame Pointer                                    │
│   X30 (LR)   │ Link Register (return address)                   │
│   SP         │ Stack Pointer                                    │
│   PC         │ Program Counter                                  │
└──────────────┴──────────────────────────────────────────────────┘
```

---

## 3. Datapath

### 3.1 Datapath Concept

The datapath is the collection of paths through which data moves within the CPU and the functional units that process data along these paths.

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Basic Datapath                              │
│                                                                     │
│    ┌──────┐         ┌──────────────┐                               │
│    │  PC  │────────►│  Instruction │                               │
│    └──────┘         │    Memory    │                               │
│        │            └──────┬───────┘                               │
│        │                   │                                        │
│        ▼                   ▼                                        │
│    ┌──────┐         ┌──────────────┐         ┌──────────────┐      │
│    │ +4   │         │  Instruction │────────►│  Control     │      │
│    └──────┘         │      IR      │         │    Unit      │      │
│                     └──────┬───────┘         └──────┬───────┘      │
│                            │                        │               │
│                            ▼                        ▼               │
│                    ┌───────────────┐         Control Signals       │
│         ┌─────────┤  Register     │                                │
│         │         │    File       │                                │
│         │         └───┬───────┬───┘                                │
│         │             │       │                                     │
│         │             ▼       ▼                                     │
│         │         ┌───────────────┐                                │
│         │         │     ALU       │                                │
│         │         └───────┬───────┘                                │
│         │                 │                                         │
│         │                 ▼                                         │
│         │         ┌───────────────┐                                │
│         └────────►│     Data      │                                │
│                   │    Memory     │                                │
│                   └───────────────┘                                │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.2 MIPS Datapath (Detailed)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              MIPS Datapath                                   │
│                                                                             │
│                                                                             │
│    ┌────────────────────────────────────────────────────────────────────┐  │
│    │                   Instruction Fetch (IF)                            │  │
│    │                                                                    │  │
│    │   ┌──────┐    ┌──────┐    ┌────────────────┐                       │  │
│    │   │  PC  │───►│ +4   │    │   Instruction  │                       │  │
│    │   └──────┘    └───┬──┘    │     Memory     │                       │  │
│    │       │           │       └────────┬───────┘                       │  │
│    │       └───────────┴────────────────┘                               │  │
│    └────────────────────────────────────────────────────────────────────┘  │
│                                    │                                        │
│                                    ▼ [31:0] Instruction                     │
│    ┌────────────────────────────────────────────────────────────────────┐  │
│    │                      Instruction Decode (ID)                        │  │
│    │                                                                    │  │
│    │    [25:21]  ┌──────────────┐                                       │  │
│    │    ────────►│   Read       │───► Read Data 1                       │  │
│    │    [20:16]  │   Register   │                                       │  │
│    │    ────────►│   File       │───► Read Data 2                       │  │
│    │    [15:11]  │              │                                       │  │
│    │    ────────►│   Write Reg  │◄── Write Data                         │  │
│    │             └──────────────┘                                       │  │
│    │                                                                    │  │
│    │    [15:0]   ┌──────────────┐                                       │  │
│    │    ────────►│  Sign        │───► 32-bit immediate                  │  │
│    │             │  Extend      │                                       │  │
│    │             └──────────────┘                                       │  │
│    └────────────────────────────────────────────────────────────────────┘  │
│                                    │                                        │
│                                    ▼                                        │
│    ┌────────────────────────────────────────────────────────────────────┐  │
│    │                         Execute (EX)                                │  │
│    │                                                                    │  │
│    │    Read Data 1 ───►┌──────────────┐                                │  │
│    │                    │              │                                │  │
│    │    MUX Output  ───►│     ALU      │───► ALU Result                 │  │
│    │                    │              │───► Zero Flag                  │  │
│    │                    └──────────────┘                                │  │
│    │                                                                    │  │
│    │    Read Data 2 ───►┌──────┐                                        │  │
│    │    Immediate   ───►│ MUX  │───► ALU Input B                        │  │
│    │                    └──────┘                                        │  │
│    │                     ▲ ALUSrc                                       │  │
│    └────────────────────────────────────────────────────────────────────┘  │
│                                    │                                        │
│                                    ▼                                        │
│    ┌────────────────────────────────────────────────────────────────────┐  │
│    │                      Memory Access (MEM)                            │  │
│    │                                                                    │  │
│    │    ALU Result ─────►┌────────────────┐                             │  │
│    │    (Address)        │     Data       │───► Read Data               │  │
│    │                     │    Memory      │                             │  │
│    │    Write Data ─────►│                │                             │  │
│    │                     └────────────────┘                             │  │
│    │                     ▲ MemRead/MemWrite                             │  │
│    └────────────────────────────────────────────────────────────────────┘  │
│                                    │                                        │
│                                    ▼                                        │
│    ┌────────────────────────────────────────────────────────────────────┐  │
│    │                       Write Back (WB)                               │  │
│    │                                                                    │  │
│    │    ALU Result  ───►┌──────┐                                        │  │
│    │    Memory Data ───►│ MUX  │───► Register File Write Data           │  │
│    │                    └──────┘                                        │  │
│    │                     ▲ MemtoReg                                     │  │
│    └────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.3 Datapath Components

| Component | Function | Control Signal |
|-----------|----------|----------------|
| PC | Store next instruction address | - |
| Instruction Memory | Store and provide instructions | - |
| Register File | Register read/write | RegWrite |
| ALU | Arithmetic/logical operations | ALUOp |
| Data Memory | Store/load data | MemRead, MemWrite |
| MUX | Select data path | ALUSrc, MemtoReg |
| Sign Extend | 16-bit → 32-bit extension | - |

---

## 4. Instruction Execution Cycle Details

### 4.1 Basic Execution Cycle

```
┌─────────────────────────────────────────────────────────────────┐
│                    Instruction Execution Cycle                   │
│                                                                 │
│   ┌────────┐   ┌────────┐   ┌────────┐   ┌────────┐   ┌──────┐ │
│   │ Fetch  │──►│ Decode │──►│Execute │──►│ Memory │──►│Write │ │
│   │        │   │        │   │        │   │ Access │   │Back  │ │
│   └────────┘   └────────┘   └────────┘   └────────┘   └──────┘ │
│       │                                                   │     │
│       └───────────────────────────────────────────────────┘     │
│                           Repeat                                │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 Each Stage in Detail

#### Stage 1: Instruction Fetch (IF)

```
Operation:
1. Copy PC value to MAR
2. Read Memory[MAR] content to MBR
3. Copy MBR content to IR
4. PC = PC + instruction size

Micro-operations:
MAR ← PC
MBR ← Memory[MAR]
IR ← MBR
PC ← PC + 4

Timing Diagram:
─────┬─────┬─────┬─────┬─────
 T0  │ T1  │ T2  │ T3  │
─────┴─────┴─────┴─────┴─────
MAR←PC    MBR←Mem  IR←MBR  PC←PC+4
```

#### Stage 2: Instruction Decode (ID)

```
Operation:
1. Analyze IR opcode field
2. Calculate operand addresses
3. Read required register values

Micro-operations:
A ← Regs[IR[25:21]]      // Read rs register
B ← Regs[IR[20:16]]      // Read rt register
ALUOut ← PC + (sign-extend(IR[15:0]) << 2)  // Calculate branch address

Instruction Format Analysis (MIPS):
┌────────┬───────┬───────┬───────┬───────┬────────┐
│ opcode │  rs   │  rt   │  rd   │ shamt │ funct  │  R-type
│  6-bit │ 5-bit │ 5-bit │ 5-bit │ 5-bit │ 6-bit  │
└────────┴───────┴───────┴───────┴───────┴────────┘

┌────────┬───────┬───────┬─────────────────────────┐
│ opcode │  rs   │  rt   │       immediate         │  I-type
│  6-bit │ 5-bit │ 5-bit │        16-bit           │
└────────┴───────┴───────┴─────────────────────────┘
```

#### Stage 3: Execute (EX)

```
Execution by instruction type:

1. Arithmetic/Logic Operations (R-type):
   ALUOut ← A op B

   Example: ADD $t0, $t1, $t2
   ALUOut ← Regs[$t1] + Regs[$t2]

2. Memory Reference:
   ALUOut ← A + sign-extend(IR[15:0])

   Example: LW $t0, 100($t1)
   ALUOut ← Regs[$t1] + 100

3. Branch:
   if (A == B) PC ← ALUOut

   Example: BEQ $t0, $t1, label
   if (Regs[$t0] == Regs[$t1])
       PC ← PC + offset × 4
```

#### Stage 4: Memory Access (MEM)

```
Load Instruction:
MDR ← Memory[ALUOut]

Example: LW $t0, 100($t1)
MDR ← Memory[Regs[$t1] + 100]

Store Instruction:
Memory[ALUOut] ← B

Example: SW $t0, 100($t1)
Memory[Regs[$t1] + 100] ← Regs[$t0]
```

#### Stage 5: Write Back (WB)

```
R-type Instruction:
Regs[IR[15:11]] ← ALUOut

Example: ADD $t0, $t1, $t2
Regs[$t0] ← ALUOut

Load Instruction:
Regs[IR[20:16]] ← MDR

Example: LW $t0, 100($t1)
Regs[$t0] ← MDR
```

### 4.3 Instruction Execution Examples

```
Example: ADD $t0, $t1, $t2 (R-type)

Stage 1 (IF):
  MAR ← PC
  IR ← Memory[MAR]
  PC ← PC + 4

Stage 2 (ID):
  A ← Regs[$t1]
  B ← Regs[$t2]

Stage 3 (EX):
  ALUOut ← A + B

Stage 4 (MEM):
  (None - no memory access needed)

Stage 5 (WB):
  Regs[$t0] ← ALUOut


Example: LW $t0, 100($t1) (I-type, Load)

Stage 1 (IF):
  MAR ← PC
  IR ← Memory[MAR]
  PC ← PC + 4

Stage 2 (ID):
  A ← Regs[$t1]

Stage 3 (EX):
  ALUOut ← A + 100

Stage 4 (MEM):
  MDR ← Memory[ALUOut]

Stage 5 (WB):
  Regs[$t0] ← MDR
```

---

## 5. Single-Cycle vs Multi-Cycle

### 5.1 Single-Cycle Implementation

```
┌─────────────────────────────────────────────────────────────────┐
│                    Single-Cycle                                  │
│                                                                 │
│  Feature: All instructions complete in one clock cycle          │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                    1 Clock Cycle                          │  │
│  │  ┌─────┬─────┬─────┬─────┬─────┐                         │  │
│  │  │ IF  │ ID  │ EX  │ MEM │ WB  │                         │  │
│  │  └─────┴─────┴─────┴─────┴─────┘                         │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
│  Clock Period = Execution time of longest instruction (Load)    │
│                                                                 │
│  Example (time for each stage):                                 │
│  - IF: 200ps                                                   │
│  - ID: 100ps                                                   │
│  - EX: 200ps                                                   │
│  - MEM: 200ps                                                  │
│  - WB: 100ps                                                   │
│                                                                 │
│  Clock Period = 200 + 100 + 200 + 200 + 100 = 800ps            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

#### Single-Cycle Advantages and Disadvantages

| Advantages | Disadvantages |
|------------|---------------|
| Simple implementation | Long clock period (based on slowest instruction) |
| Simple control logic | All instructions take same time |
| CPI = 1 (1 cycle per instruction) | Low hardware resource utilization |

### 5.2 Multi-Cycle Implementation

```
┌─────────────────────────────────────────────────────────────────┐
│                    Multi-Cycle                                   │
│                                                                 │
│  Feature: Instructions execute over multiple clock cycles       │
│                                                                 │
│  Each stage in a separate cycle:                                │
│  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐                       │
│  │ IF  │ │ ID  │ │ EX  │ │ MEM │ │ WB  │                       │
│  └─────┘ └─────┘ └─────┘ └─────┘ └─────┘                       │
│  Cycle 1 Cycle 2 Cycle 3 Cycle 4 Cycle 5                        │
│                                                                 │
│  Clock Period = Time of longest stage                           │
│  = max(200, 100, 200, 200, 100) = 200ps                        │
│                                                                 │
│  Cycles per instruction type:                                   │
│  - Load:  5 cycles (IF, ID, EX, MEM, WB)                       │
│  - Store: 4 cycles (IF, ID, EX, MEM)                           │
│  - R-type: 4 cycles (IF, ID, EX, WB)                           │
│  - Branch: 3 cycles (IF, ID, EX)                               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

#### Multi-Cycle State Diagram

```
                    ┌─────────────────────────────────┐
                    │                                 │
                    ▼                                 │
              ┌──────────┐                           │
              │  IF      │                           │
              │ (Fetch)  │                           │
              └────┬─────┘                           │
                   │                                 │
                   ▼                                 │
              ┌──────────┐                           │
              │  ID      │                           │
              │ (Decode) │                           │
              └────┬─────┘                           │
                   │                                 │
         ┌─────────┼─────────┐                       │
         │         │         │                       │
         ▼         ▼         ▼                       │
    ┌────────┐ ┌────────┐ ┌────────┐                │
    │ MEM    │ │ R-type │ │ Branch │───────────────┘
    │Address │ │ Exec   │ │Complete│
    └────┬───┘ └────┬───┘ └────────┘
         │         │
    ┌────┴────┐    │
    │         │    │
    ▼         ▼    │
┌───────┐ ┌───────┐│
│  Load │ │ Store ││
│  MEM  │ │  MEM  ││
└───┬───┘ └───────┘│
    │              │
    ▼              │
┌───────┐    ┌─────┴─────┐
│ Load  │    │  R-type   │
│  WB   │    │    WB     │
└───────┘    └───────────┘
```

### 5.3 Performance Comparison

```
┌─────────────────────────────────────────────────────────────────┐
│                    Performance Comparison Example                │
│                                                                 │
│  Assumptions:                                                   │
│  - Instruction mix: Load 25%, Store 10%, R-type 45%, Branch 20%│
│  - Single-cycle clock: 800ps                                    │
│  - Multi-cycle clock: 200ps                                     │
│                                                                 │
│  Single-Cycle:                                                  │
│  Average time = 800ps × 1 = 800ps/instruction                  │
│                                                                 │
│  Multi-Cycle:                                                   │
│  Average CPI = 0.25×5 + 0.10×4 + 0.45×4 + 0.20×3               │
│           = 1.25 + 0.40 + 1.80 + 0.60 = 4.05                   │
│  Average time = 200ps × 4.05 = 810ps/instruction               │
│                                                                 │
│  Conclusion: Similar performance in this example                │
│  (But multi-cycle uses hardware resources more efficiently)     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 5.4 Comparison Summary

| Feature | Single-Cycle | Multi-Cycle |
|---------|-------------|-------------|
| CPI | 1 | Variable (3~5) |
| Clock Period | Long (longest instruction) | Short (longest stage) |
| Control Logic | Simple | Complex (FSM required) |
| Hardware Usage | Inefficient | Efficient (shared) |
| Memory | Instruction/data separate | Can be unified |
| Implementation Complexity | Low | Medium |

---

## 6. Practice Problems

### Basic Problems

1. What are the three main components of a CPU?

2. Explain the role of the following registers:
   - (a) PC (Program Counter)
   - (b) IR (Instruction Register)
   - (c) MAR (Memory Address Register)

3. List three types of operations performed by the ALU.

### Datapath Problems

4. For the following MIPS instruction, select all control signals that are activated:

   `ADD $t0, $t1, $t2`

   - (a) RegWrite
   - (b) ALUSrc
   - (c) MemRead
   - (d) MemWrite
   - (e) MemtoReg

5. Explain the 5-stage execution process for the instruction `LW $t0, 100($t1)` in order.

### Performance Analysis Problems

6. In a single-cycle CPU where each stage takes the following time:
   - IF: 250ps
   - ID: 150ps
   - EX: 200ps
   - MEM: 300ps
   - WB: 100ps

   (a) What is the clock period?
   (b) How many instructions can be executed in 1 second?

7. In a multi-cycle CPU with the following instruction distribution, calculate the average CPI:
   - Load: 30% (5 cycles)
   - Store: 15% (4 cycles)
   - R-type: 40% (4 cycles)
   - Branch: 15% (3 cycles)

### Advanced Problems

8. Explain three CPU design techniques to solve the Von Neumann bottleneck.

9. Calculate the total execution time for the following code sequence in both single-cycle and multi-cycle CPUs (using time assumptions from problem 6):

```assembly
LW   $t0, 0($s0)
ADD  $t1, $t0, $t2
SW   $t1, 4($s0)
```

<details>
<summary>Answers</summary>

1. ALU (Arithmetic Logic Unit), Control Unit, Registers

2.
   - (a) PC: Stores the address of the next instruction to execute
   - (b) IR: Stores the currently executing instruction
   - (c) MAR: Stores the memory address to access

3. Arithmetic operations (addition, subtraction, etc.), Logical operations (AND, OR, etc.), Shift operations (or Comparison operations)

4. (a) RegWrite - Activated because result is written to register
   - ALUSrc = 0 (use register value)
   - MemtoReg = 0 (select ALU result)

5.
   - IF: Fetch instruction from PC, PC+4
   - ID: Read $t1 register value, sign-extend offset(100)
   - EX: Calculate address $t1 + 100
   - MEM: Read data from calculated address
   - WB: Store read data to $t0

6.
   - (a) Clock period = 250 + 150 + 200 + 300 + 100 = 1000ps = 1ns
   - (b) 1 second / 1ns = 10^9 = 1 GIPS (Giga Instructions Per Second)

7. Average CPI = 0.30×5 + 0.15×4 + 0.40×4 + 0.15×3
           = 1.50 + 0.60 + 1.60 + 0.45 = 4.15

8.
   - Cache memory: Place high-speed memory between CPU and memory
   - Pipelining: Process multiple instructions simultaneously
   - Prefetching: Fetch required data in advance

9.
   - Single-cycle: 3 × 1000ps = 3000ps = 3ns
   - Multi-cycle: (5 + 4 + 4) × 300ps = 13 × 300ps = 3900ps = 3.9ns
     (Clock period = max(250, 150, 200, 300, 100) = 300ps)

</details>

---

## Next Steps

- [08_Control_Unit.md](./08_Control_Unit.md) - Hardwired/Microprogrammed Control

---

## References

- Computer Organization and Design (Patterson & Hennessy)
- Computer Architecture: A Quantitative Approach (Hennessy & Patterson)
- [CPU Visual Simulator](https://cpuvisualsimulator.github.io/)
- [MIPS Datapath Simulator](https://courses.cs.washington.edu/courses/cse378/09au/lectures/datapath.html)
